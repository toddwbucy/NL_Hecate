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

use crate::config::{Config, OptimizerConfig, PhaseDuration};
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

    // ── Resolve phases ───────────────────────────────────────────────
    let phases = cfg.resolved_phases().unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

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

    // ── Conductor ────────────────────────────────────────────────────
    let mut conductor = Conductor::new(k, chunk_sizes.clone());
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

    let default_opt = &cfg.build.optimizer;

    eprintln!("============================================================");
    eprintln!("NL-Hecate Build (spec 61: phase list)");
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
    eprintln!("  Optimizer: {} (lr={}, b1={}, b2={}, wd={})",
        default_opt.optimizer_type, default_opt.lr,
        default_opt.beta1(), default_opt.beta2(), default_opt.weight_decay());
    eprintln!("  Grad clip: max_norm={}", cfg.build.max_grad_norm);
    eprintln!("  Phases:   {}", phases.len());
    for (i, phase) in phases.iter().enumerate() {
        let dur = match &phase.duration {
            PhaseDuration::Steps(s) => format!("{s} steps"),
            PhaseDuration::ThinkRounds(r) => format!("{r} think_rounds"),
        };
        let opt_info = phase.optimizer.as_ref()
            .map(|o| format!(" [{}@{}]", o.optimizer_type, o.lr))
            .unwrap_or_default();
        eprintln!("    [{i}] {}: {}{}", phase.label, dur, opt_info);
    }
    eprintln!("  Stacked:  {n_blocks} blocks x k={k} CMS levels  ({} params)", fmt_num(total_params));
    if !reset_intervals.is_empty() {
        eprintln!("  Reset:    selective intervals={reset_intervals:?} (spec 57)");
    }
    eprintln!("  Log:      {log_file}");
    eprintln!("============================================================");

    logger.log_build_start(d, nh, seq_len, k, n_blocks, cfg.build.steps, default_opt.lr, total_params);

    // ── Phase loop (spec 61) ─────────────────────────────────────────
    let t_start = Instant::now();
    let mut global_step = resume_step;
    let mut loss_first: Option<f32> = None;
    let mut loss_last: f32 = 0.0;
    let mut step_tokens: usize = 0;
    let mut aborted = false;

    for (phase_idx, phase) in phases.iter().enumerate() {
        // Resolve per-phase overrides (fall back to build defaults)
        let opt = phase.optimizer.as_ref().unwrap_or(default_opt);
        let batch_size = phase.batch_size.unwrap_or(cfg.build.batch_size);
        let phase_seq_len = phase.seq_len.unwrap_or(seq_len);
        let save_every = phase.save_every.unwrap_or(cfg.build.save_every);
        let log_every = phase.log_every.unwrap_or(cfg.build.log_every);
        let max_grad_norm = phase.max_grad_norm.unwrap_or(cfg.build.max_grad_norm);
        let warmup_steps = phase.warmup_steps.unwrap_or(cfg.build.warmup_steps);

        eprintln!();
        eprintln!("── Phase {phase_idx}: {} ──", phase.label);

        match &phase.duration {
            PhaseDuration::Steps(total_phase_steps) => {
                // ── Steps mode: streaming consumption ──────────────
                let mut loaders: Vec<BpeTokenStream> = Vec::new();
                for b in 0..batch_size {
                    let mut loader = BpeTokenStream::load(&phase.data).unwrap_or_else(|e| {
                        eprintln!("ERROR: {e}");
                        std::process::exit(1);
                    });
                    if batch_size > 1 {
                        let offset = b * (loader.total_tokens / batch_size);
                        loader.seek(offset);
                    }
                    loaders.push(loader);
                }

                let total_tokens_phase = loaders[0].total_tokens;
                eprintln!("  Data:  {} tokens, batch_size={batch_size}, seq_len={phase_seq_len}",
                    fmt_num(total_tokens_phase));
                eprintln!("  Opt:   {} lr={} wd={} gnorm_clip={}",
                    opt.optimizer_type, opt.lr, opt.weight_decay(), max_grad_norm);

                // Compute total steps for LR schedule within this phase
                let phase_end_step = global_step + total_phase_steps;

                for phase_step in 0..*total_phase_steps {
                    let lr = cosine_lr(phase_step, warmup_steps, *total_phase_steps, opt.lr);

                    // Assemble batch
                    let mut all_input = Vec::with_capacity(batch_size * phase_seq_len);
                    let mut all_target = Vec::with_capacity(batch_size * phase_seq_len);
                    for loader in &mut loaders {
                        let (inp, tgt) = loader.next_chunk(phase_seq_len).unwrap_or_else(|| {
                            eprintln!("ERROR: data exhausted");
                            std::process::exit(1);
                        });
                        all_input.extend_from_slice(&inp);
                        all_target.extend_from_slice(&tgt);
                    }

                    let pulse = conductor.pulse();

                    // Forward + Backward + Optimizer
                    #[cfg(feature = "cuda")]
                    let (loss, grad_norm) = {
                        run_step(
                            &mut gpu_params, &mag_cfg, &mut gpu_context,
                            &mut adamw_state,
                            &all_input, &all_target, &pulse,
                            opt, lr, max_grad_norm,
                            d, v,
                            &reset_intervals, &mut fire_counts,
                        )
                    };

                    conductor.advance();
                    step_tokens += batch_size * phase_seq_len;
                    global_step += 1;

                    if loss_first.is_none() { loss_first = Some(loss); }
                    loss_last = loss;

                    if loss.is_nan() || loss.is_infinite() {
                        eprintln!("  ABORTING: NaN/Inf at step {global_step}");
                        aborted = true;
                        break;
                    }

                    // Logging
                    let log_this = log_every > 0 && phase_step % log_every == 0;
                    if log_this || phase_step == 0 {
                        let elapsed = t_start.elapsed().as_secs_f64();
                        let tok_s = step_tokens as f64 / elapsed;
                        let ppl = (loss as f64).exp();
                        let rss_mb = get_rss_mb();

                        eprintln!("  step {:>6}  loss={loss:.4}  ppl={ppl:.1}  tok/s={tok_s:.0}  gnorm={grad_norm:.4}  lr={lr:.6}  rss={rss_mb}MB",
                            global_step);

                        logger.log_step(global_step, loss, grad_norm, lr, elapsed, &pulse_to_active(&pulse));
                    }

                    // Checkpoint
                    let do_checkpoint = save_every > 0
                        && phase_step > 0
                        && (phase_step + 1) % save_every == 0;

                    if do_checkpoint {
                        save_checkpoint(
                            &save_path, global_step,
                            #[cfg(feature = "cuda")]
                            &gpu_params,
                            #[cfg(feature = "cuda")]
                            &gpu_context,
                            &mag_cfg, &conductor, &chunk_sizes,
                            &loaders, d, v, k,
                            &mut logger,
                        );
                    }
                }
            }

            PhaseDuration::ThinkRounds(rounds) => {
                // ── Think rounds: iterative self-refinement ────────
                // Load the data once as the initial input
                let mut loader = BpeTokenStream::load(&phase.data).unwrap_or_else(|e| {
                    eprintln!("ERROR: {e}");
                    std::process::exit(1);
                });
                let total_tokens_phase = loader.total_tokens;
                eprintln!("  Data:  {} tokens, {rounds} think_rounds",
                    fmt_num(total_tokens_phase));
                eprintln!("  Opt:   {} lr={} wd={} gnorm_clip={}",
                    opt.optimizer_type, opt.lr, opt.weight_decay(), max_grad_norm);

                // Load all data as initial input
                let (mut input, mut target) = loader.next_chunk(phase_seq_len)
                    .unwrap_or_else(|| {
                        eprintln!("ERROR: data exhausted in think_rounds");
                        std::process::exit(1);
                    });

                for round in 0..*rounds {
                    eprintln!("  [think round {}/{}]", round + 1, rounds);

                    let lr = opt.lr; // think_rounds uses constant lr (no schedule)
                    let pulse = conductor.pulse();

                    // LEARN from current input
                    #[cfg(feature = "cuda")]
                    let (loss, grad_norm) = {
                        run_step(
                            &mut gpu_params, &mag_cfg, &mut gpu_context,
                            &mut adamw_state,
                            &input, &target, &pulse,
                            opt, lr, max_grad_norm,
                            d, v,
                            &reset_intervals, &mut fire_counts,
                        )
                    };

                    conductor.advance();
                    step_tokens += phase_seq_len;
                    global_step += 1;

                    if loss_first.is_none() { loss_first = Some(loss); }
                    loss_last = loss;

                    eprintln!("    loss={loss:.4}  gnorm={grad_norm:.4}");
                    logger.log_step(global_step, loss, grad_norm, lr,
                        t_start.elapsed().as_secs_f64(), &pulse_to_active(&pulse));

                    if loss.is_nan() || loss.is_infinite() {
                        eprintln!("  ABORTING: NaN/Inf at think round {round}");
                        aborted = true;
                        break;
                    }

                    // SPEAK — generate output from what was just learned
                    // TODO: implement prefill + decode_token loop for think_rounds
                    // For now, re-present the same data (deferred until generation is in CLI)
                    // When generation lands:
                    //   let logits = prefill(&input, &pulse);
                    //   let output = decode_loop(logits, max_gen_tokens);
                    //   input = output; target = shifted(output);
                    eprintln!("    [speak phase: deferred — generation not yet in CLI]");
                }
            }
        }

        if aborted { break; }

        // ── Phase boundary checkpoint ─────────────────────────────
        eprintln!("  [phase {phase_idx} complete — checkpoint at step {global_step}]");
        #[cfg(feature = "cuda")]
        {
            // Synthesize a minimal loader list for checkpoint metadata
            let loaders_empty: Vec<BpeTokenStream> = Vec::new();
            save_checkpoint(
                &save_path, global_step,
                &gpu_params,
                &gpu_context,
                &mag_cfg, &conductor, &chunk_sizes,
                &loaders_empty, d, v, k,
                &mut logger,
            );
        }
    }

    // ── Summary ──────────────────────────────────────────────────────
    let elapsed = t_start.elapsed().as_secs_f64();
    let tok_s = step_tokens as f64 / elapsed;
    eprintln!();
    eprintln!("============================================================");
    eprintln!("  Phases:   {} complete", phases.len());
    eprintln!("  Steps:    {global_step} ({elapsed:.0}s)");
    eprintln!("  Tok/s:    {tok_s:.0}");
    eprintln!("  Loss:     {:.4} → {loss_last:.4}", loss_first.unwrap_or(0.0));
    eprintln!("============================================================");

    logger.log_build_end(global_step - resume_step, elapsed, tok_s,
        loss_first.unwrap_or(0.0), loss_last);
}

// ── Extracted helpers ────────────────────────────────────────────────

/// Run a single training step: forward + backward + optimizer update.
#[cfg(feature = "cuda")]
fn run_step(
    gpu_params: &mut GpuStackedParams,
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    input: &[usize],
    target: &[usize],
    pulse: &Pulse,
    opt: &OptimizerConfig,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    v: usize,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
) -> (f32, f32) {
    let (loss, cache) = gpu_stacked_forward(
        gpu_params, mag_cfg, input, target,
        pulse, gpu_context, &mut None,
    );

    let mut grads = gpu_stacked_backward(
        gpu_params, mag_cfg, &cache, &mut None,
    );

    // Lazy-init AdamW
    if adamw_state.is_none() {
        *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
    }
    let state = adamw_state.as_mut().unwrap();

    let gnorm = gpu_stacked_adamw_update(
        gpu_params, &mut grads, state,
        pulse,
        lr, opt.beta1(), opt.beta2(), 1e-8,
        opt.weight_decay(), max_grad_norm,
        false, // freeze_embed
        &mut None, // profiler
    );

    // Weight tying
    gpu_stacked_sync_embed_weights(gpu_params, d, v);

    // Dormancy tracking
    gpu_context.update_m_norm_tracking();

    // Selective periodic reset (spec 57)
    maybe_reset_levels(pulse, reset_intervals, fire_counts, gpu_context);

    (loss, gnorm)
}

/// Save checkpoint with build state.
fn save_checkpoint(
    save_path: &str,
    global_step: usize,
    #[cfg(feature = "cuda")] gpu_params: &GpuStackedParams,
    #[cfg(feature = "cuda")] gpu_context: &GpuStackedContext,
    mag_cfg: &MAGConfig,
    conductor: &Conductor,
    chunk_sizes: &[usize],
    loaders: &[BpeTokenStream],
    d: usize,
    v: usize,
    k: usize,
    logger: &mut MetricsLogger,
) {
    let ckpt_path = save_path.replace(
        ".safetensors",
        &format!("_step{global_step}.safetensors"),
    );

    #[cfg(feature = "cuda")]
    {
        let host_params = gpu_params.to_host(d, v, k);
        let host_context = gpu_context.blocks[0].to_host(k);
        let stream_position = loaders.first().map(|l| l.position as u64).unwrap_or(0);
        let build_state = BuildResumeState {
            conductor: ConductorState {
                k,
                chunk_sizes: chunk_sizes.to_vec(),
                step: conductor.step(),
            },
            stream_cursor: StreamCursor {
                position: stream_position,
                chunk_id: 0,
                pulse_id: 0,
                rng_state: None,
                content_hash: 0,
            },
            context: host_context,
            global_step,
        };
        save_stacked_safetensors(
            Path::new(&ckpt_path), &host_params, mag_cfg,
            Some(&build_state),
        ).unwrap_or_else(|e| {
            eprintln!("ERROR: checkpoint save failed: {e}");
        });
    }

    eprintln!("  [checkpoint saved: {ckpt_path}]");
    logger.log_checkpoint(global_step, &ckpt_path);
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
            for block_ctx in &mut context.blocks {
                if level < block_ctx.memory.len() {
                    block_ctx.memory[level].zero();
                }
            }
        }
    }
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
