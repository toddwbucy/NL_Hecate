use std::path::Path;
use std::time::Instant;

use nl_hecate_core::checkpoint::{load_stacked_safetensors, save_stacked_safetensors};
use nl_hecate_core::conductor::{Conductor, Pulse, ConductorState};
use nl_hecate_core::context_stream::StreamCursor;
use nl_hecate_core::model::{
    MAGConfig, SWAConfig, MemoryRuleKind, CompositionKind, HopeVariant,
    LevelTapeStrategy, BuildResumeState,
};
use nl_hecate_core::parallel::{ParallelConfig, ParallelStrategy};

#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_forward::gpu_stacked_forward;
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward;
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_optimizer::{
    GpuStackedAdamWState, gpu_stacked_adamw_update, gpu_stacked_sync_embed_weights,
};

use crate::config::Config;
use crate::data::BpeTokenStream;
use crate::log::MetricsLogger;

/// Cosine annealing with linear warmup.
fn cosine_lr(step: usize, warmup_steps: usize, total_steps: usize, lr_peak: f32) -> f32 {
    if step < warmup_steps {
        return lr_peak * step as f32 / warmup_steps.max(1) as f32;
    }
    let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps).max(1) as f32;
    let progress = progress.min(1.0);
    0.5 * lr_peak * (1.0 + (std::f32::consts::PI * progress).cos())
}

pub fn run(config_path: &str, _resume: bool) {
    let cfg = Config::from_file(config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    let seq_len = cfg.seq_len();
    let d = cfg.model.d_model;
    let nh = cfg.model.num_heads;
    let hd = d / nh;
    let v = cfg.model.vocab_size;
    let k = cfg.model.k;
    let n_blocks = cfg.model.n_blocks;
    let ws = cfg.model.window_size;

    // ── Resolve paths ────────────────────────────────────────────────
    let run_dir = cfg.build.run_dir.as_deref().unwrap_or("runs/default");
    std::fs::create_dir_all(format!("{run_dir}/checkpoints")).ok();

    let save_path = cfg.build.save_path.clone()
        .unwrap_or_else(|| format!("{run_dir}/checkpoints/model.safetensors"));
    let log_file = cfg.build.log_file.clone()
        .unwrap_or_else(|| format!("{run_dir}/metrics.jsonl"));

    // ── Build MAGConfig ──────────────────────────────────────────────
    let chunk_sizes = cfg.model.chunk_sizes.clone()
        .unwrap_or_else(|| default_chunk_sizes(k));

    let tape_strategies: Vec<LevelTapeStrategy> = cfg.model.tape_strategies
        .as_ref()
        .map(|ts| ts.iter().map(|s| match s.as_str() {
            "exact" => LevelTapeStrategy::Exact,
            _ => LevelTapeStrategy::Proxy,
        }).collect())
        .unwrap_or_else(|| {
            let mut v = vec![LevelTapeStrategy::Exact];
            v.extend((1..k).map(|_| LevelTapeStrategy::Proxy));
            v
        });

    let hope_variant = match cfg.model.hope_variant.as_str() {
        "chained" => HopeVariant::Chained,
        "independent" => HopeVariant::Independent,
        "nested" => HopeVariant::Nested,
        "sequential" => HopeVariant::Sequential,
        _ => HopeVariant::FreqGated,
    };

    let memory_rule = match cfg.model.memory_rule.as_str() {
        "titans" => MemoryRuleKind::TitansLMM,
        "delta" => MemoryRuleKind::DeltaRule,
        "hebbian" => MemoryRuleKind::HebbianRule,
        _ => MemoryRuleKind::TitansLMM,
    };

    let composition = match cfg.model.composition.as_str() {
        "mal" => CompositionKind::MAL,
        "mac" => CompositionKind::MAC,
        _ => CompositionKind::MAG,
    };

    let parallel = cfg.model.parallel_strategy.as_ref().map(|ps| {
        ParallelConfig {
            strategy: match ps.as_str() {
                "tnt_hierarchical" => ParallelStrategy::TNTHierarchical,
                _ => ParallelStrategy::TNTHierarchical,
            },
            chunk_size: cfg.model.tnt_global_chunk_size.unwrap_or(64),
            tnt_global_chunk_size: cfg.model.tnt_global_chunk_size.unwrap_or(64),
            tnt_local_chunk_size: cfg.model.tnt_local_chunk_size.unwrap_or(8),
        }
    });

    let mut mag_cfg = MAGConfig::test_config();
    mag_cfg.swa = SWAConfig {
        d_model: d,
        num_heads: nh,
        head_dim: hd,
        seq_len,
        window_size: ws,
        vocab_size: v,
    };
    mag_cfg.memory_enabled = true;
    mag_cfg.composition = composition;
    mag_cfg.memory_rule = memory_rule;
    mag_cfg.k = k;
    mag_cfg.chunk_sizes = chunk_sizes.clone();
    mag_cfg.hope_variant = hope_variant;
    mag_cfg.residual = cfg.model.residual;
    mag_cfg.parallel = parallel;
    mag_cfg.tape_multiplier = cfg.model.tape_multiplier;
    mag_cfg.tape_strategies = tape_strategies;

    // Gate clamps
    if let Some(ref af) = cfg.build.alpha_floor {
        mag_cfg.alpha_floor = af.clone();
    }
    if let Some(ref tc) = cfg.build.theta_ceil {
        mag_cfg.theta_ceil = tc.clone();
    }
    if let Some(ref mnm) = cfg.model.m_norm_max {
        mag_cfg.m_norm_max = mnm.clone();
    }
    if let Some(ref ec) = cfg.model.error_clip {
        mag_cfg.error_clip = ec.clone();
    }

    // ── Load checkpoint or init ──────────────────────────────────────
    let mut resume_step: usize = 0;

    #[cfg(feature = "cuda")]
    let (mut gpu_params, mut gpu_context, mut adamw_state);

    if let Some(ref load_path) = cfg.build.load {
        let p = Path::new(load_path);
        if !p.exists() {
            eprintln!("ERROR: checkpoint not found: {load_path}");
            std::process::exit(1);
        }

        let (host_params, loaded_cfg, loaded_n_blocks, build_state) =
            load_stacked_safetensors(p).unwrap_or_else(|e| {
                eprintln!("ERROR: failed to load checkpoint: {e}");
                std::process::exit(1);
            });

        // Use loaded config values but override seq_len if specified
        if cfg.build.seq_len_override.is_some() {
            mag_cfg.swa.seq_len = seq_len;
        }

        if let Some(bs) = &build_state {
            resume_step = bs.global_step;
        }

        eprintln!("Loading checkpoint: {load_path}");
        eprintln!("  Stacked checkpoint: n_blocks={loaded_n_blocks}, k={} ({}build state)",
            loaded_cfg.k, if build_state.is_some() { "with " } else { "no " });

        #[cfg(feature = "cuda")]
        {
            gpu_params = GpuStackedParams::from_host(&host_params);
            gpu_context = GpuStackedContext::new(
                n_blocks, k, d, cfg.build.batch_size, Some(&mag_cfg),
            );
            adamw_state = None;
        }
    } else {
        // Fresh init
        let host_params = nl_hecate_core::stacked_model::StackedMAGParams::init(
            &mag_cfg, n_blocks, cfg.build.seed,
        );

        #[cfg(feature = "cuda")]
        {
            gpu_params = GpuStackedParams::from_host(&host_params);
            gpu_context = GpuStackedContext::new(
                n_blocks, k, d, cfg.build.batch_size, Some(&mag_cfg),
            );
            adamw_state = None;
        }
    }

    // ── Reset intervals (spec 57) ────────────────────────────────────
    let reset_intervals: Vec<usize> = cfg.model.reset_intervals.clone()
        .unwrap_or_default();
    let mut fire_counts = vec![0usize; k];

    // ── Data loading ─────────────────────────────────────────────────
    eprintln!("Loading data: {}", cfg.data.path);
    let mut loaders: Vec<BpeTokenStream> = Vec::new();
    let bs = cfg.build.batch_size;

    for b in 0..bs {
        let mut loader = BpeTokenStream::load(&cfg.data.path).unwrap_or_else(|e| {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        });
        // Distribute slots across the corpus
        if bs > 1 {
            let offset = b * (loader.total_tokens / bs);
            loader.seek(offset);
        }
        loaders.push(loader);
    }

    let total_tokens = loaders[0].total_tokens;

    // ── Conductor ────────────────────────────────────────────────────
    let mut conductor = Conductor::new(k, chunk_sizes.clone());
    // Advance conductor to match resume step
    for _ in 0..resume_step {
        conductor.advance();
    }

    // ── Logger ───────────────────────────────────────────────────────
    let mut logger = MetricsLogger::new(&log_file).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    // ── Print banner ─────────────────────────────────────────────────
    let total_params = {
        #[cfg(feature = "cuda")]
        { gpu_params.to_host(d, v, k).num_params() }
    };

    eprintln!("============================================================");
    eprintln!("NL-Hecate Build");
    eprintln!("============================================================");
    eprintln!("  Model:    d={d}, heads={nh}, seq_len={seq_len}, vocab={v}");
    eprintln!("  Memory:   rule={:?}, composition={:?}, k={k}", memory_rule, composition);
    eprintln!("  CMS:      chunk_sizes={chunk_sizes:?}");
    if let Some(ref mnm) = cfg.model.m_norm_max {
        eprintln!("  M-norm:   max={mnm:?}");
    }
    if let Some(ref ec) = cfg.model.error_clip {
        eprintln!("  ErrClip:  max={ec:?}");
    }
    eprintln!("  Data:     {} tokens ({} BPE)", fmt_num(total_tokens), cfg.data.format);
    eprintln!("  Build:    {} steps (from step {resume_step}), lr={}", cfg.build.steps, cfg.build.lr);
    eprintln!("  Optimizer: {} (b1={}, b2={}, wd={}, warmup={})",
        cfg.build.optimizer, cfg.build.beta1, cfg.build.beta2,
        cfg.build.weight_decay, cfg.build.warmup_steps);
    eprintln!("  Grad clip: max_norm={}", cfg.build.max_grad_norm);
    eprintln!("  Checkpoint: every {} steps", cfg.build.save_every);
    if cfg.build.flashcard {
        eprintln!("  Flashcard: {}% × {} rounds @ {} gen tokens",
            cfg.build.flashcard_pct, cfg.build.flashcard_rounds, cfg.build.flashcard_gen_tokens);
    }
    eprintln!("  Log:      {log_file}");
    eprintln!("============================================================");
    eprintln!();
    eprintln!("  Stacked:  {n_blocks} blocks x k={k} CMS levels  ({} params)", fmt_num(total_params));
    if !reset_intervals.is_empty() {
        eprintln!("  Reset:    selective intervals={reset_intervals:?} (spec 57)");
    }

    logger.log_build_start(d, nh, seq_len, k, n_blocks, cfg.build.steps, cfg.build.lr, total_params);

    // ── Flashcard state ──────────────────────────────────────────────
    let mut flashcard_interval_positions: Vec<usize> = loaders.iter().map(|l| l.position).collect();

    // ── Training loop ────────────────────────────────────────────────
    let end_step = resume_step + cfg.build.steps;
    let t_start = Instant::now();
    let mut loss_first: Option<f32> = None;
    let mut loss_last: f32 = 0.0;
    let mut step_tokens: usize = 0;

    for step in resume_step..end_step {
        // ── LR schedule ──────────────────────────────────────────
        let lr = cosine_lr(step, cfg.build.warmup_steps, end_step, cfg.build.lr);

        // ── Assemble batch ───────────────────────────────────────
        let mut all_input = Vec::with_capacity(bs * seq_len);
        let mut all_target = Vec::with_capacity(bs * seq_len);
        for loader in &mut loaders {
            let (inp, tgt) = loader.next_chunk(seq_len).unwrap_or_else(|| {
                eprintln!("ERROR: data exhausted");
                std::process::exit(1);
            });
            all_input.extend_from_slice(&inp);
            all_target.extend_from_slice(&tgt);
        }

        // ── Pulse ────────────────────────────────────────────────
        let pulse = conductor.pulse();

        // ── Forward + Backward + Optimizer ───────────────────────
        #[cfg(feature = "cuda")]
        let (loss, grad_norm) = {
            let (loss, cache) = gpu_stacked_forward(
                &gpu_params, &mag_cfg, &all_input, &all_target,
                &pulse, &mut gpu_context, &mut None,
            );

            let mut grads = gpu_stacked_backward(
                &gpu_params, &mag_cfg, &cache, &mut None,
            );

            // Lazy-init AdamW
            if adamw_state.is_none() {
                adamw_state = Some(GpuStackedAdamWState::from_params(&gpu_params));
            }
            let state = adamw_state.as_mut().unwrap();

            let gnorm = gpu_stacked_adamw_update(
                &mut gpu_params, &mut grads, state,
                &pulse,
                lr, cfg.build.beta1, cfg.build.beta2, 1e-8,
                cfg.build.weight_decay, cfg.build.max_grad_norm,
                false, // freeze_embed
                &mut None, // profiler
            );

            // Weight tying
            gpu_stacked_sync_embed_weights(&mut gpu_params, d, v);

            // Dormancy tracking
            gpu_context.update_m_norm_tracking();

            // Selective periodic reset (spec 57)
            maybe_reset_levels(&pulse, &reset_intervals, &mut fire_counts, &mut gpu_context);

            (loss, gnorm)
        };

        // ── Advance ──────────────────────────────────────────────
        conductor.advance();
        step_tokens += bs * seq_len;

        if loss_first.is_none() { loss_first = Some(loss); }
        loss_last = loss;

        // ── NaN check ────────────────────────────────────────────
        if loss.is_nan() || loss.is_infinite() {
            eprintln!("  ABORTING: NaN/Inf at step {step}");
            break;
        }

        // ── Logging ──────────────────────────────────────────────
        let log_this = cfg.build.log_every > 0 && step % cfg.build.log_every == 0;
        if log_this || step == resume_step {
            let elapsed = t_start.elapsed().as_secs_f64();
            let tok_s = step_tokens as f64 / elapsed;
            let ppl = (loss as f64).exp();
            let rss_mb = get_rss_mb();

            eprintln!("  step {:>6}  loss={loss:.4}  ppl={ppl:.1}  tok/s={tok_s:.0}  gnorm={grad_norm:.4}  lr={lr:.6}  rss={rss_mb}MB",
                step);

            logger.log_step(step, loss, grad_norm, lr, elapsed, &pulse_to_active(&pulse));
        }

        // ── Checkpoint ───────────────────────────────────────────
        let do_checkpoint = cfg.build.save_every > 0
            && step > resume_step
            && (step + 1 - resume_step) % cfg.build.save_every == 0;

        if do_checkpoint {
            // Flashcard session before checkpoint
            if cfg.build.flashcard {
                #[cfg(feature = "cuda")]
                run_flashcard(
                    &cfg, &mag_cfg, step,
                    &mut gpu_params, &mut gpu_context,
                    &mut adamw_state,
                    &mut loaders, &flashcard_interval_positions,
                    &mut conductor,
                    &reset_intervals, &mut fire_counts,
                );
                // Update interval positions for next checkpoint
                flashcard_interval_positions = loaders.iter().map(|l| l.position).collect();
            }

            // Save checkpoint
            let ckpt_path = save_path.replace(
                ".safetensors",
                &format!("_step{}.safetensors", step + 1),
            );

            #[cfg(feature = "cuda")]
            {
                let host_params = gpu_params.to_host(d, v, k);
                let host_context = gpu_context.blocks[0].to_host(k);
                let build_state = BuildResumeState {
                    conductor: ConductorState {
                        k,
                        chunk_sizes: chunk_sizes.clone(),
                        step: conductor.step(),
                    },
                    stream_cursor: StreamCursor {
                        position: loaders[0].position as u64,
                        chunk_id: 0,
                        pulse_id: 0,
                        rng_state: None,
                        content_hash: 0,
                    },
                    context: host_context,
                    global_step: step + 1,
                };
                save_stacked_safetensors(
                    Path::new(&ckpt_path), &host_params, &mag_cfg,
                    Some(&build_state),
                ).unwrap_or_else(|e| {
                    eprintln!("ERROR: checkpoint save failed: {e}");
                });
            }

            eprintln!("  [checkpoint saved: {ckpt_path}]");
            logger.log_checkpoint(step + 1, &ckpt_path);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    let elapsed = t_start.elapsed().as_secs_f64();
    let tok_s = step_tokens as f64 / elapsed;
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  Steps:    {} ({elapsed:.0}s)", cfg.build.steps);
    eprintln!("  Tok/s:    {tok_s:.0}");
    eprintln!("  Loss:     {:.4} → {loss_last:.4}", loss_first.unwrap_or(0.0));
    eprintln!("============================================================");

    logger.log_build_end(cfg.build.steps, elapsed, tok_s,
        loss_first.unwrap_or(0.0), loss_last);
}

/// Format number with comma separators.
fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

/// Default chunk sizes for k levels.
fn default_chunk_sizes(k: usize) -> Vec<usize> {
    match k {
        1 => vec![1],
        2 => vec![1, 8],
        3 => vec![1, 8, 64],
        _ => vec![1, 8, 64, 512],
    }
}

/// Convert Pulse to Vec<bool> for logging.
fn pulse_to_active(pulse: &Pulse) -> Vec<bool> {
    pulse.active_levels.clone()
}

/// Selective periodic reset (spec 57).
#[cfg(feature = "cuda")]
fn maybe_reset_levels(
    pulse: &Pulse,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
    context: &mut GpuStackedContext,
) {
    if reset_intervals.is_empty() { return; }
    for (level, active) in pulse.active_levels.iter().enumerate() {
        if !active { continue; }
        if level >= reset_intervals.len() { continue; }
        fire_counts[level] += 1;
        if reset_intervals[level] > 0 && fire_counts[level] >= reset_intervals[level] {
            fire_counts[level] = 0;
            // Zero M for this level across all blocks
            for block_ctx in &mut context.blocks {
                if level < block_ctx.memory.len() {
                    block_ctx.memory[level].zero();
                }
            }
        }
    }
}

/// Run flashcard session: sample chunks from the last checkpoint interval,
/// re-present each chunk for N rounds.
#[cfg(feature = "cuda")]
fn run_flashcard(
    cfg: &Config,
    mag_cfg: &MAGConfig,
    step: usize,
    gpu_params: &mut GpuStackedParams,
    gpu_context: &mut GpuStackedContext,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    loaders: &mut [BpeTokenStream],
    interval_positions: &[usize],
    conductor: &mut Conductor,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
) {
    let seq_len = mag_cfg.swa.seq_len;
    let d = mag_cfg.swa.d_model;
    let v = mag_cfg.swa.vocab_size;
    let pct = cfg.build.flashcard_pct;
    let rounds = cfg.build.flashcard_rounds;
    let lr = cosine_lr(step, cfg.build.warmup_steps,
        step + cfg.build.steps, cfg.build.lr); // approximate

    // Build deck: sample pct% of chunks from interval [start_pos, current_pos)
    let mut deck: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();

    // Use first loader for now (batch_size=1 primary path)
    let loader = &loaders[0];
    let start_pos = interval_positions[0];
    let end_pos = loader.position;

    if end_pos <= start_pos { return; }

    let total_chunks = (end_pos - start_pos) / seq_len;
    let n_sample = ((total_chunks as f32 * pct / 100.0).ceil() as usize).max(1);
    let n_sample = n_sample.min(total_chunks);

    // Build a temporary loader for sampling
    let mut fc_loader = BpeTokenStream::load(&cfg.data.path).unwrap();
    fc_loader.seek(start_pos);

    // Collect all chunks in range, then sample
    let mut all_chunks: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    for _ in 0..total_chunks {
        if let Some(chunk) = fc_loader.next_chunk(seq_len) {
            all_chunks.push(chunk);
        }
    }

    // Simple deterministic sampling: take every N-th chunk
    let stride = (all_chunks.len() / n_sample).max(1);
    for (i, chunk) in all_chunks.into_iter().enumerate() {
        if i % stride == 0 && deck.len() < n_sample {
            deck.push(chunk);
        }
    }

    if deck.is_empty() { return; }

    eprintln!("  Flashcard deck: {} chunks available in range [{start_pos}, {end_pos})",
        total_chunks);
    eprintln!("  [flashcard] {} chunks × {rounds} rounds (step {})", deck.len(), step + 1);

    // Create separate conductor for flashcard (doesn't affect training conductor)
    let mut fc_conductor = Conductor::new(mag_cfg.k, mag_cfg.chunk_sizes.clone());

    // Process each chunk for N rounds
    for (input_ids, target_ids) in &deck {
        for _round in 0..rounds {
            let pulse = fc_conductor.pulse();

            let (_, cache) = gpu_stacked_forward(
                gpu_params, mag_cfg, input_ids, target_ids,
                &pulse, gpu_context, &mut None,
            );

            let mut grads = gpu_stacked_backward(
                gpu_params, mag_cfg, &cache, &mut None,
            );

            if adamw_state.is_none() {
                *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
            }
            let state = adamw_state.as_mut().unwrap();

            gpu_stacked_adamw_update(
                gpu_params, &mut grads, state,
                &pulse,
                lr, cfg.build.beta1, cfg.build.beta2, 1e-8,
                cfg.build.weight_decay, cfg.build.max_grad_norm,
                false, &mut None,
            );

            gpu_stacked_sync_embed_weights(gpu_params, d, v);
            gpu_context.update_m_norm_tracking();
            maybe_reset_levels(&pulse, reset_intervals, fire_counts, gpu_context);

            fc_conductor.advance();
        }
    }

    eprintln!("  [flashcard] done: {} chunks reinforced", deck.len());
}

/// Get resident set size in MB (Linux).
fn get_rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
                .map(|kb| kb / 1024)
        })
        .unwrap_or(0)
}
