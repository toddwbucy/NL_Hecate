use std::path::Path;
use std::time::Instant;

use nl_hecate_core::checkpoint::{load_stacked_safetensors, save_stacked_safetensors};
use nl_hecate_core::conductor::{Conductor, Pulse, ConductorState};
use nl_hecate_core::context_stream::StreamCursor;
use nl_hecate_core::model::{
    MAGConfig, SWAConfig, MemoryRuleKind, CompositionKind, HopeVariant,
    LevelTapeStrategy, BuildResumeState, PushUpInit,
};
use nl_hecate_core::parallel::{ParallelConfig, ParallelStrategy};

#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_backward::{gpu_accumulate_grads, GpuStackedGrads};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_optimizer::{
    GpuStackedAdamWState, gpu_read_grad_norm, gpu_stacked_scale_grads_ex,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_profiler::GpuProfiler;

use crate::checkpoint_policy::{TriggerState, checkpoint_filename};
use crate::config::{Config, PhaseDuration, TriggerConfig, CheckpointNaming};
use crate::state_file::{
    self, StateFile, CheckpointEntry, HealthSnapshot, SessionInfo,
};
use crate::data::BpeTokenStream;
#[cfg(feature = "cuda")]
use crate::probe::run_inline_probes;
#[cfg(feature = "cuda")]
use crate::step::{step, step_micro, step_update, generate, BlockLevelGnorms};
use crate::log::{MetricsLogger, CmsDiagnostics, CmsTapeLogger, TOKENS_PER_SEGMENT};

/// Cosine annealing with linear warmup.
fn cosine_lr(step: usize, warmup_steps: usize, total_steps: usize, lr_peak: f32) -> f32 {
    if step < warmup_steps {
        return lr_peak * step as f32 / warmup_steps.max(1) as f32;
    }
    let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps).max(1) as f32;
    let progress = progress.min(1.0);
    0.5 * lr_peak * (1.0 + (std::f32::consts::PI * progress).cos())
}

pub fn feed(config_path: &str, resume: bool) {
    let mut cfg = Config::from_file(config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    // --resume: find latest checkpoint in run_dir if build.load is not set
    if resume && cfg.build.load.is_none() {
        let run_dir = cfg.build.run_dir.as_deref().unwrap_or("runs/default");
        let ckpt_dir = format!("{run_dir}/checkpoints");
        if let Ok(entries) = std::fs::read_dir(&ckpt_dir) {
            let mut latest: Option<(std::time::SystemTime, String)> = None;
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "safetensors") {
                    if let Ok(meta) = entry.metadata() {
                        if let Ok(modified) = meta.modified() {
                            if latest.as_ref().map_or(true, |(t, _)| modified > *t) {
                                latest = Some((modified, path.to_string_lossy().into_owned()));
                            }
                        }
                    }
                }
            }
            if let Some((_, path)) = latest {
                eprintln!("  [resume: loading {path}]");
                cfg.build.load = Some(path);
            } else {
                eprintln!("  [resume: no checkpoints found in {ckpt_dir}, starting fresh]");
            }
        }
    }

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
    let mut resume_conductor_step: usize = 0;
    let mut resume_tokens: usize = 0;
    let mut resume_cursors: Vec<StreamCursor> = Vec::new();

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

        // Validate checkpoint shape matches config
        if loaded_cfg.swa.d_model != d {
            eprintln!("ERROR: checkpoint d_model={} but config d_model={d}", loaded_cfg.swa.d_model);
            std::process::exit(1);
        }
        if loaded_cfg.swa.num_heads != nh {
            eprintln!("ERROR: checkpoint num_heads={} but config num_heads={nh}", loaded_cfg.swa.num_heads);
            std::process::exit(1);
        }
        if loaded_cfg.swa.vocab_size != v {
            eprintln!("ERROR: checkpoint vocab_size={} but config vocab_size={v}", loaded_cfg.swa.vocab_size);
            std::process::exit(1);
        }
        if loaded_cfg.k != k {
            eprintln!("ERROR: checkpoint k={} but config k={k}", loaded_cfg.k);
            std::process::exit(1);
        }
        if loaded_n_blocks != n_blocks {
            eprintln!("ERROR: checkpoint n_blocks={loaded_n_blocks} but config n_blocks={n_blocks}");
            std::process::exit(1);
        }

        if cfg.build.seq_len_override.is_some() {
            mag_cfg.swa.seq_len = seq_len;
        }

        if let Some(bs) = &build_state {
            resume_step = bs.global_step;
            resume_conductor_step = bs.conductor.step;
            resume_tokens = bs.total_tokens_seen;
            // Restore per-slot cursors for batch>1 resume
            if !bs.stream_cursors.is_empty() {
                resume_cursors = bs.stream_cursors.clone();
            } else if bs.stream_cursor.position > 0 {
                // Backward compat: single-cursor checkpoints
                resume_cursors = vec![bs.stream_cursor.clone()];
            }
        }
        if cfg.build.reset_step {
            eprintln!("  [reset_step: overriding checkpoint step {} → 0]", resume_step);
            resume_step = 0;
            resume_conductor_step = 0;
            resume_tokens = 0;
            resume_cursors.clear();
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

    // ── k-extension (spec 7 / spec 22) ─────────────────────────────
    // Mutables that extend_k may modify
    let mut k = k;
    let mut chunk_sizes = chunk_sizes;

    if let Some(target_k) = cfg.build.extend_k {
        if cfg.build.load.is_none() {
            eprintln!("ERROR: extend_k requires a checkpoint (set build.load)");
            std::process::exit(1);
        }
        if target_k != k + 1 {
            eprintln!("ERROR: extend_k={target_k} must be loaded_k+1={}", k + 1);
            std::process::exit(1);
        }
        if !cfg.build.push_up && !cfg.build.stack_up {
            eprintln!("ERROR: extend_k set but neither push_up nor stack_up — set one of them");
            std::process::exit(1);
        }
        if cfg.build.push_up && cfg.build.stack_up {
            eprintln!("ERROR: push_up and stack_up are mutually exclusive");
            std::process::exit(1);
        }

        let chunk_template = [1, 8, 64, 512];
        if target_k > chunk_template.len() {
            eprintln!("ERROR: extend_k={target_k} exceeds max supported k={}", chunk_template.len());
            std::process::exit(1);
        }

        let new_chunks: Vec<usize> = if cfg.build.push_up {
            chunk_template[..target_k].to_vec()
        } else {
            let mut c = chunk_sizes.clone();
            c.push(chunk_template[target_k - 1]);
            c
        };

        // Stacked checkpoints only support push_up (not stack_up) per spec 22
        if cfg.build.stack_up {
            eprintln!("ERROR: stacked extend_k only supports push_up (not stack_up)");
            std::process::exit(1);
        }

        // Snapshot pre-extension config for to_host — GPU buffers still have old k
        #[cfg(feature = "cuda")]
        let pre_extend_cfg = mag_cfg.clone();

        // Update MAGConfig for new k
        mag_cfg.k = target_k;
        mag_cfg.chunk_sizes = new_chunks.clone();

        // Extend per-level arrays
        while mag_cfg.m_norm_max.len() < target_k {
            mag_cfg.m_norm_max.push(*mag_cfg.m_norm_max.last().unwrap_or(&100.0));
        }
        while mag_cfg.error_clip.len() < target_k {
            mag_cfg.error_clip.push(*mag_cfg.error_clip.last().unwrap_or(&100.0));
        }

        // Perform the extension
        #[cfg(feature = "cuda")]
        {
            let host = gpu_params.to_host(&pre_extend_cfg);
            let init = match cfg.build.push_up_init.as_str() {
                "clone" => PushUpInit::Clone,
                _ => PushUpInit::Random,
            };
            let new_host = host.extend_push_up(&mag_cfg, cfg.build.seed, init);
            gpu_params = GpuStackedParams::from_host(&new_host);
            gpu_context = GpuStackedContext::new(
                n_blocks, target_k, d, cfg.build.batch_size, Some(&mag_cfg),
            );
            adamw_state = None; // Fresh optimizer for new level structure
        }

        let mode = if cfg.build.push_up { "push-up" } else { "stack-up" };
        eprintln!("  k-extension ({mode}): k={k} → k={target_k}, chunks={new_chunks:?}");

        k = target_k;
        chunk_sizes = new_chunks;
        resume_step = 0; // New phase starts from step 0
        resume_conductor_step = 0;
    }

    // ── Spec 02: State file — lifecycle record ──────────────────────
    // Placed after extend_k so that mag_cfg captures post-extension architecture.
    let state_basename = std::path::Path::new(&save_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let state_file_path = state_file::state_file_path(run_dir, state_basename);
    let mut model_state = if state_file_path.exists() {
        match state_file::load_state_file(&state_file_path) {
            Ok(sf) => {
                eprintln!("  [state file loaded: {} ({} checkpoints, {} tokens)]",
                    state_file_path.display(), sf.checkpoints.len(), sf.tokens_total);
                sf
            }
            Err(e) => {
                eprintln!("WARNING: failed to load state file {}: {e} — creating fresh",
                    state_file_path.display());
                state_file::init_state_file(state_basename, &mag_cfg)
            }
        }
    } else {
        let sf = if cfg.build.load.is_some() {
            // Resuming from checkpoint without a state file — create one with parent ref
            let mut sf = state_file::init_state_file(state_basename, &mag_cfg);
            sf.tokens_total = resume_tokens as u64;
            eprintln!("  [state file created from checkpoint resume: {}]", state_file_path.display());
            sf
        } else {
            eprintln!("  [state file created: {}]", state_file_path.display());
            state_file::init_state_file(state_basename, &mag_cfg)
        };
        // Persist immediately so the state file exists even if the run crashes before first checkpoint
        if let Err(e) = state_file::save_state_file(&state_file_path, &sf) {
            eprintln!("WARNING: failed to write state file: {e}");
        }
        sf
    };

    // ── Spec 04: Inline cursor — state file is authoritative cursor source ──
    // Priority: state file cursor > safetensors BuildResumeState > legacy .cursor.json
    // Guard: state file cursor only valid if it matches the loaded checkpoint.
    if cfg.build.load.is_some() && !cfg.build.reset_step {
        let load_path = cfg.build.load.as_deref().unwrap_or("");
        let state_cursor_matches = !model_state.cursor.slots.is_empty()
            && model_state.current_checkpoint.as_ref()
                .map(|cc| cc.path == load_path)
                .unwrap_or(false);

        if state_cursor_matches {
            eprintln!("  [cursor: from state file ({} slots)]", model_state.cursor.slots.len());
            resume_cursors = model_state.cursor.slots.clone();
        } else {
            if !model_state.cursor.slots.is_empty() {
                eprintln!("  [cursor: state file cursor skipped — checkpoint mismatch]");
            }
            if resume_cursors.is_empty() {
                // Safetensors had no cursors either — try legacy sidecar
                let legacy = state_file::load_legacy_cursor(load_path);
                if !legacy.is_empty() {
                    eprintln!("  [cursor: from legacy .cursor.json sidecar]");
                    resume_cursors = legacy;
                }
            }
            // else: resume_cursors already populated from safetensors BuildResumeState
        }
    }

    // ── Reset intervals (spec 57) ────────────────────────────────────
    let reset_intervals: Vec<usize> = cfg.model.reset_intervals.clone()
        .unwrap_or_default();
    let mut fire_counts = vec![0usize; k];

    // ── Conductor ────────────────────────────────────────────────────
    // Replay conductor to its persisted per-token step (not global_step which
    // counts optimizer steps). step()/generate() advance conductor per-token.
    let mut conductor = Conductor::new(k, chunk_sizes.clone());
    conductor.set_seq_len(seq_len);
    for _ in 0..resume_conductor_step {
        conductor.advance();
    }

    // ── Dormancy detection config ─────────────────────────────────
    #[cfg(feature = "cuda")]
    if let Some(ref floors) = cfg.model.dormancy_floor {
        let consecutive = cfg.model.dormancy_consecutive;
        if consecutive > 0 {
            gpu_context.set_dormancy_config(floors.clone(), consecutive);
            eprintln!("  Dormancy: floors={floors:?}, consecutive={consecutive}");
        }
    }

    // ── Logger ───────────────────────────────────────────────────────
    let mut logger = MetricsLogger::new(&log_file).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    // ── CMS tape sidecar ──────────────────────────────────────────
    let mut cms_tape: Option<CmsTapeLogger> = if cfg.build.cms_sidecar {
        let tape_path = format!("{}/cms_tape.jsonl", run_dir);
        match CmsTapeLogger::new(&tape_path) {
            Ok(t) => {
                eprintln!("  CMS tape: {tape_path}");
                Some(t)
            }
            Err(e) => {
                eprintln!("WARNING: could not open CMS tape: {e}");
                None
            }
        }
    } else {
        None
    };

    // ── Step profiler ──────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    let profile_every = cfg.build.profile_every;
    #[cfg(feature = "cuda")]
    let mut profile_logger: Option<crate::log::ProfileLogger> = if profile_every > 0 {
        let prof_path = format!("{}/step_profile.jsonl", run_dir);
        match crate::log::ProfileLogger::new(&prof_path) {
            Ok(pl) => {
                eprintln!("  Profiler: every {profile_every} steps → {prof_path}");
                Some(pl)
            }
            Err(e) => {
                eprintln!("WARNING: could not open profile log: {e}");
                None
            }
        }
    } else {
        None
    };

    // ── Print banner ─────────────────────────────────────────────────
    let total_params = {
        #[cfg(feature = "cuda")]
        { gpu_params.to_host(&mag_cfg).num_params() }
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
        default_opt.optimizer_type(), default_opt.lr(),
        default_opt.beta1(), default_opt.beta2(), default_opt.weight_decay());
    eprintln!("  Grad clip: max_norm={}", cfg.build.max_grad_norm);
    eprintln!("  Phases:   {}", phases.len());
    for (i, phase) in phases.iter().enumerate() {
        let dur = match &phase.duration {
            PhaseDuration::Steps(s) => format!("{s} steps"),
            PhaseDuration::ThinkRounds(r) => format!("{r} think_rounds"),
        };
        let opt_info = phase.optimizer.as_ref()
            .map(|o| format!(" [{}@{}]", o.optimizer_type(), o.lr()))
            .unwrap_or_default();
        eprintln!("    [{i}] {}: {}{}", phase.label, dur, opt_info);
    }
    eprintln!("  Stacked:  {n_blocks} blocks x k={k} CMS levels  ({} params)", fmt_num(total_params));
    if !reset_intervals.is_empty() {
        eprintln!("  Reset:    selective intervals={reset_intervals:?} (spec 57)");
    }
    eprintln!("  Log:      {log_file}");
    eprintln!("============================================================");

    logger.log_build_start(d, nh, seq_len, k, n_blocks, cfg.build.steps, default_opt.lr(), total_params);

    // Spec 73: enable GPU buffer pool to recycle allocations across steps.
    // After the first warm-up step, all ~572 per-step cudaMalloc/cudaFree calls
    // become free-list hits (zero CUDA runtime overhead).
    #[cfg(feature = "cuda")]
    {
        nl_hecate_core::gpu_buf::gpu_pool_enable();
        eprintln!("  GPU pool: enabled (spec 73)");
    }

    // ── Phase loop (spec 61) ─────────────────────────────────────────
    let t_start = Instant::now();
    let mut global_step = resume_step;
    let mut loss_first: Option<f32> = None;
    let mut loss_last: f32 = 0.0;
    let mut step_tokens: usize = 0;
    let mut total_tokens_seen: usize = resume_tokens; // CG-6: cumulative tokens for segment accounting
    let mut aborted = false;
    let mut trigger_state = TriggerState::new(resume_tokens as u64);
    let mut last_loaders: Vec<BpeTokenStream> = Vec::new();

    for (phase_idx, phase) in phases.iter().enumerate() {
        // Resolve per-phase overrides (fall back to build defaults)
        let opt = phase.optimizer.as_ref().unwrap_or(default_opt);
        let batch_size = phase.batch_size.unwrap_or(cfg.build.batch_size);
        let phase_seq_len = phase.seq_len.unwrap_or(seq_len);
        let save_every = phase.save_every.unwrap_or(cfg.build.save_every);
        let log_every = phase.log_every.unwrap_or(cfg.build.log_every);

        // Spec 78: resolve effective checkpoint policy for this phase.
        // Priority: phase.checkpoint > build.checkpoint.triggers > save_every fallback
        let phase_checkpoint_policy = phase.checkpoint.as_ref()
            .unwrap_or(&cfg.build.checkpoint);
        let effective_triggers: Vec<TriggerConfig> = if !phase_checkpoint_policy.triggers.is_empty() {
            phase_checkpoint_policy.triggers.clone()
        } else if save_every > 0 {
            vec![TriggerConfig::StepCount { every: save_every }]
        } else {
            vec![]
        };
        let checkpoint_naming = &phase_checkpoint_policy.naming;
        let has_new_style_triggers = effective_triggers.iter().any(|t| !matches!(t, TriggerConfig::StepCount { .. }));
        let max_grad_norm = phase.max_grad_norm.unwrap_or(cfg.build.max_grad_norm);
        let warmup_steps = phase.warmup_steps.unwrap_or(cfg.build.warmup_steps);

        eprintln!();
        eprintln!("── Phase {phase_idx}: {} ──", phase.label);

        // Reallocate GPU context if phase overrides batch_size or seq_len
        #[cfg(feature = "cuda")]
        {
            let needs_realloc = batch_size != cfg.build.batch_size
                || phase_seq_len != seq_len;
            if needs_realloc {
                mag_cfg.swa.seq_len = phase_seq_len;
                gpu_context = GpuStackedContext::new(
                    n_blocks, k, d, batch_size, Some(&mag_cfg),
                );
                // Re-apply dormancy config after reallocation
                if let Some(ref floors) = cfg.model.dormancy_floor {
                    let consecutive = cfg.model.dormancy_consecutive;
                    if consecutive > 0 {
                        gpu_context.set_dormancy_config(floors.clone(), consecutive);
                    }
                }
                // AdamW state is param-sized (not batch/seq dependent), keep it
                eprintln!("  [GPU context reallocated: batch_size={batch_size}, seq_len={phase_seq_len}]");
            }
        }

        match &phase.duration {
            PhaseDuration::Steps(total_phase_steps) => {
                // ── Steps mode: streaming consumption ──────────────
                let mut loaders: Vec<BpeTokenStream> = Vec::new();
                for b in 0..batch_size {
                    let mut loader = BpeTokenStream::load(&phase.data).unwrap_or_else(|e| {
                        eprintln!("ERROR: {e}");
                        std::process::exit(1);
                    });
                    // Restore from checkpoint cursors if available, else use even spacing
                    if b < resume_cursors.len() {
                        let pos = resume_cursors[b].position as usize;
                        loader.seek(pos);
                        if b == 0 {
                            eprintln!("  [resuming {batch_size} loaders from checkpoint cursors]");
                        }
                    } else if batch_size > 1 {
                        let offset = b * (loader.total_tokens / batch_size);
                        loader.seek(offset);
                    }
                    loaders.push(loader);
                }
                // Clear resume cursors after first use (subsequent phases start fresh)
                resume_cursors.clear();

                let accum_steps = phase.accum_steps.unwrap_or(cfg.build.accum_steps);

                let total_tokens_phase = loaders[0].total_tokens;
                let tok_per_logical_step = accum_steps * batch_size * phase_seq_len;
                eprintln!("  Data:  {} tokens, batch_size={batch_size}, accum_steps={accum_steps}, seq_len={phase_seq_len}",
                    fmt_num(total_tokens_phase));
                eprintln!("  Eff:   {} tokens per logical step",
                    fmt_num(tok_per_logical_step));
                eprintln!("  Opt:   {} lr={} wd={} gnorm_clip={}",
                    opt.optimizer_type(), opt.lr(), opt.weight_decay(), max_grad_norm);

                // Compute total steps for LR schedule within this phase
                let _phase_end_step = global_step + total_phase_steps;

                for phase_step in 0..*total_phase_steps {
                    let lr = cosine_lr(phase_step, warmup_steps, *total_phase_steps, opt.lr());

                    // Capture pulse snapshot for logging (conductor advances inside step_micro())
                    let pulse = conductor.pulse();

                    // Spec 63: compute log_this early so backward can skip gnorm readback
                    let log_this = log_every > 0 && phase_step % log_every == 0;

                    // Profiler setup
                    #[cfg(feature = "cuda")]
                    let do_profile = profile_every > 0 && phase_step > 0
                        && phase_step % profile_every == 0;
                    #[cfg(feature = "cuda")]
                    let mut profiler: Option<GpuProfiler> = if do_profile {
                        Some(GpuProfiler::new(true))
                    } else {
                        None
                    };

                    // ── Spec 76: Gradient accumulation loop ──────────────────────
                    // Accumulate gradients across accum_steps micro-steps, each with
                    // batch_size samples. ONE optimizer step on averaged gradients.
                    #[cfg(feature = "cuda")]
                    let (loss, _gnorm_deferred, block_level_gnorms) = {
                        let mut total_loss = 0.0f32;
                        let mut total_micro_samples = 0usize;
                        let mut acc_gnorms: BlockLevelGnorms = Vec::new();
                        // First micro-step produces the accumulator (grads from backward)
                        let mut grad_accum: Option<GpuStackedGrads> = None;

                        for _micro in 0..accum_steps {
                            // Assemble fresh batch for this micro-step (no data replay)
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

                            // Each batch sample: forward + backward (NO optimizer)
                            for sample_idx in 0..batch_size {
                                let start = sample_idx * phase_seq_len;
                                let end = start + phase_seq_len;

                                let micro_result = step_micro(
                                    &gpu_params, &mag_cfg, &mut gpu_context,
                                    &all_input[start..end], &all_target[start..end],
                                    &mut conductor,
                                    &mut profiler,
                                    log_this && total_micro_samples == 0,
                                );

                                // Accumulate gradients
                                match grad_accum {
                                    None => {
                                        // First micro-step: take ownership of grads as accumulator
                                        acc_gnorms = micro_result.block_level_gnorms;
                                        grad_accum = Some(micro_result.grads);
                                    }
                                    Some(ref mut accum) => {
                                        gpu_accumulate_grads(accum, &micro_result.grads);
                                        // Accumulate gnorms for diagnostics
                                        for (acc, val) in acc_gnorms.iter_mut().zip(micro_result.block_level_gnorms.iter()) {
                                            for (a, v) in acc.iter_mut().zip(val.iter()) {
                                                *a += v;
                                            }
                                        }
                                    }
                                }

                                total_loss += micro_result.loss;
                                total_micro_samples += 1;
                            }
                        }

                        let n_samples_f = total_micro_samples as f32;
                        // Average gnorms across all micro-step samples
                        for block in acc_gnorms.iter_mut() {
                            for g in block.iter_mut() { *g /= n_samples_f; }
                        }

                        // Scale accumulated gradients: average over all micro-steps × batch samples
                        let mut grads = grad_accum.expect("at least one micro-step");
                        gpu_stacked_scale_grads_ex(&mut grads, 1.0 / n_samples_f, false);

                        // ONE optimizer step on averaged gradients
                        let gnorm = step_update(
                            &mut gpu_params, &mut grads, &mut adamw_state,
                            &pulse, opt, lr, max_grad_norm, d, v,
                            &reset_intervals, &mut fire_counts,
                            &mut gpu_context, &mut profiler, log_this,
                        );

                        (total_loss / n_samples_f, gnorm, acc_gnorms)
                    };

                    // Collect and log profile if active
                    #[cfg(feature = "cuda")]
                    if let Some(ref mut prof) = profiler {
                        let step_profile = prof.collect(n_blocks);
                        if let Some(ref mut pl) = profile_logger {
                            let by_cat: Vec<serde_json::Value> = step_profile.by_category.iter()
                                .map(|c| serde_json::json!({
                                    "category": c.category.as_str(),
                                    "ms": c.ms,
                                    "pct": c.pct,
                                }))
                                .collect();
                            let per_block: Vec<serde_json::Value> = step_profile.per_block.iter()
                                .map(|b| serde_json::json!({
                                    "block": b.block_idx,
                                    "fwd_ms": b.fwd_ms,
                                    "bwd_ms": b.bwd_ms,
                                    "opt_ms": b.opt_ms,
                                }))
                                .collect();
                            pl.log_profile(global_step + 1, serde_json::json!({
                                "total_ms": step_profile.total_ms,
                                "by_category": by_cat,
                                "per_block": per_block,
                            }));
                        }
                        prof.reset();
                    }

                    // Spec 76: tokens per logical step = accum_steps × batch_size × seq_len
                    let tokens_this_step = accum_steps * batch_size * phase_seq_len;
                    step_tokens += tokens_this_step;
                    total_tokens_seen += tokens_this_step;
                    global_step += 1;

                    if loss_first.is_none() { loss_first = Some(loss); }
                    loss_last = loss;

                    if loss.is_nan() || loss.is_infinite() {
                        eprintln!("  ABORTING: NaN/Inf at step {global_step}");
                        aborted = true;
                        break;
                    }

                    // Logging + CMS diagnostics (gated to log_every to avoid per-step GPU stalls)
                    if log_this || phase_step == 0 {
                        // Spec 64: update_m_norm_tracking() now called inside step_update(),
                        // before maybe_reset_levels(), so we read pre-reset norms.

                        #[cfg(feature = "cuda")]
                        let cms_diag = collect_cms_diagnostics(&gpu_context, &block_level_gnorms, k);

                        // Check for dormancy transitions and log events
                        #[cfg(feature = "cuda")]
                        for (l, status) in cms_diag.dormancy_status.iter().enumerate() {
                            if status != "active" {
                                let max_count = gpu_context.dormancy_below_count.iter()
                                    .filter_map(|b| b.get(l))
                                    .copied()
                                    .max()
                                    .unwrap_or(0);
                                logger.log_dormancy(global_step, 0, l, status, max_count);
                            }
                        }

                        // CMS tape sidecar
                        #[cfg(feature = "cuda")]
                        if let Some(ref mut tape) = cms_tape {
                            tape.log_step(global_step, &cms_diag);
                        }

                        // Spec 62: read grad norm from GPU only on log steps (deferred readback)
                        #[cfg(feature = "cuda")]
                        let grad_norm = if let Some(ref state) = adamw_state {
                            gpu_read_grad_norm(state)
                        } else { 0.0 };

                        let elapsed = t_start.elapsed().as_secs_f64();
                        let tok_s = step_tokens as f64 / elapsed;
                        let ppl = (loss as f64).exp();
                        let rss_mb = get_rss_mb();

                        let segments = total_tokens_seen / TOKENS_PER_SEGMENT;
                        eprintln!("  step {:>6}  seg={segments:<8}  loss={loss:.4}  ppl={ppl:.1}  tok/s={tok_s:.0}  gnorm={grad_norm:.4}  lr={lr:.6}  rss={rss_mb}MB",
                            global_step);

                        logger.log_step(global_step, loss, grad_norm, lr, elapsed,
                            &pulse_to_active(&pulse), &level_firings(&pulse, phase_seq_len, &chunk_sizes),
                            #[cfg(feature = "cuda")]
                            Some(&cms_diag),
                            #[cfg(not(feature = "cuda"))]
                            None,
                            total_tokens_seen,
                            tokens_this_step,
                            accum_steps,
                        );
                    }

                    // Spec 78: Checkpoint trigger evaluation
                    trigger_state.record_loss(loss);
                    let do_checkpoint = if has_new_style_triggers {
                        // New-style triggers: token_count, elapsed_minutes, loss_plateau
                        trigger_state.should_checkpoint(&effective_triggers, total_tokens_seen as u64)
                    } else {
                        // Legacy StepCount fallback: exact modulo semantics
                        save_every > 0 && phase_step > 0 && (phase_step + 1) % save_every == 0
                    };

                    if do_checkpoint {
                        save_checkpoint(
                            &save_path, global_step, total_tokens_seen,
                            checkpoint_naming, loss,
                            &mut model_state, &state_file_path, &phase.label,
                            #[cfg(feature = "cuda")]
                            &gpu_params,
                            #[cfg(feature = "cuda")]
                            &gpu_context,
                            &mag_cfg, &conductor, &chunk_sizes,
                            &loaders, d, v, k,
                            &mut logger,
                        );
                        trigger_state.record_checkpoint(total_tokens_seen as u64, loss);

                        // Run inline probes if tokenizer is configured
                        #[cfg(feature = "cuda")]
                        if let Some(ref tok_path) = cfg.build.tokenizer_path {
                            let snapshot = gpu_params.to_host(&mag_cfg);
                            let probe_results = run_inline_probes(
                                &snapshot, &mag_cfg, tok_path, global_step,
                                d, v, k, n_blocks, &chunk_sizes,
                                cfg.build.probe_max_tokens,
                                cfg.build.probe_temperature,
                                opt.lr(), opt.beta1(), opt.beta2(),
                                opt.weight_decay(), max_grad_norm,
                            );
                            logger.log_probe_results(global_step, probe_results);
                        }
                    }
                }
                // Preserve loaders for on_unload cursor state
                last_loaders = loaders;
            }

            PhaseDuration::ThinkRounds(rounds) => {
                // ── Think rounds: iterative self-refinement ────────
                if batch_size != 1 {
                    eprintln!("ERROR: think_rounds requires batch_size=1 (got {batch_size}) — \
                        generation is singleton. Add \"batch_size\": 1 to the phase.");
                    std::process::exit(1);
                }

                // Load the data once as the initial input
                let mut loader = BpeTokenStream::load(&phase.data).unwrap_or_else(|e| {
                    eprintln!("ERROR: {e}");
                    std::process::exit(1);
                });
                let total_tokens_phase = loader.total_tokens;
                if total_tokens_phase < phase_seq_len {
                    eprintln!("ERROR: think_rounds data has {} tokens but seq_len={phase_seq_len}",
                        total_tokens_phase);
                    std::process::exit(1);
                }
                eprintln!("  Data:  {} tokens, {rounds} think_rounds",
                    fmt_num(total_tokens_phase));
                eprintln!("  Opt:   {} lr={} wd={} gnorm_clip={}",
                    opt.optimizer_type(), opt.lr(), opt.weight_decay(), max_grad_norm);

                // Load all data as initial input
                let (mut input, mut target) = loader.next_chunk(phase_seq_len)
                    .unwrap_or_else(|| {
                        eprintln!("ERROR: data exhausted in think_rounds");
                        std::process::exit(1);
                    });

                for round in 0..*rounds {
                    eprintln!("  [think round {}/{}]", round + 1, rounds);

                    let lr = opt.lr(); // think_rounds uses constant lr (no schedule)
                    let pulse = conductor.pulse(); // snapshot for logging

                    // LEARN from current input (unified forward path)
                    #[cfg(feature = "cuda")]
                    let (loss, _gnorm_deferred, block_level_gnorms) = {
                        let result = step(
                            &mut gpu_params, &mag_cfg, &mut gpu_context,
                            &mut adamw_state,
                            &input, &target,
                            &mut conductor,
                            opt, lr, max_grad_norm,
                            d, v,
                            &reset_intervals, &mut fire_counts,
                            &mut None, // no profiling in think_rounds (too few steps)
                            true, // think_rounds always log
                        );
                        (result.loss, result.grad_norm, result.block_level_gnorms)
                    };

                    // conductor.advance() removed — happens per-token inside step()
                    step_tokens += phase_seq_len;
                    global_step += 1;

                    if loss_first.is_none() { loss_first = Some(loss); }
                    loss_last = loss;

                    // Spec 62: deferred grad norm readback (think_rounds always log)
                    #[cfg(feature = "cuda")]
                    let grad_norm = if let Some(ref state) = adamw_state {
                        gpu_read_grad_norm(state)
                    } else { 0.0 };

                    #[cfg(feature = "cuda")]
                    let cms_diag = collect_cms_diagnostics(&gpu_context, &block_level_gnorms, k);

                    #[cfg(feature = "cuda")]
                    if let Some(ref mut tape) = cms_tape {
                        tape.log_step(global_step, &cms_diag);
                    }

                    total_tokens_seen += phase_seq_len; // think_rounds: bs=1
                    let segments = total_tokens_seen / TOKENS_PER_SEGMENT;
                    eprintln!("    seg={segments}  loss={loss:.4}  gnorm={grad_norm:.4}");
                    logger.log_step(global_step, loss, grad_norm, lr,
                        t_start.elapsed().as_secs_f64(), &pulse_to_active(&pulse),
                        &level_firings(&pulse, phase_seq_len, &chunk_sizes),
                        #[cfg(feature = "cuda")]
                        Some(&cms_diag),
                        #[cfg(not(feature = "cuda"))]
                        None,
                        total_tokens_seen,
                        phase_seq_len, // think_rounds: batch_size=1, accum_steps=1
                        1, // no accumulation in think_rounds
                    );

                    if loss.is_nan() || loss.is_infinite() {
                        eprintln!("  ABORTING: NaN/Inf at think round {round}");
                        aborted = true;
                        break;
                    }

                    // SPEAK — generate via step() with deferred backward
                    // Then REDIRECT — generated output becomes next round's input
                    #[cfg(feature = "cuda")]
                    if round + 1 < *rounds {
                        let gen_tokens = phase.max_gen_tokens.unwrap_or(phase_seq_len);
                        let gen_temp = phase.temperature.unwrap_or(0.0);
                        let gen_top_k = phase.top_k.unwrap_or(0);

                        let gen_result = generate(
                            &mut gpu_params, &mag_cfg, &mut gpu_context,
                            &mut adamw_state, &mut conductor,
                            opt, lr, max_grad_norm, d, v,
                            &input, gen_tokens, gen_temp, gen_top_k,
                            None, // no stop token for think rounds
                            &mut None, // no profiling
                        );

                        // Account for generate()'s optimizer step in bookkeeping.
                        // generate() forwarded prompt + generated tokens and ran backward.
                        let gen_tok_count = gen_result.tokens.len();
                        step_tokens += gen_tok_count;
                        total_tokens_seen += gen_tok_count;

                        // REDIRECT — truncate output to seq_len, use as next input
                        let output = gen_result.tokens;
                        let new_input = if output.len() >= phase_seq_len {
                            output[output.len() - phase_seq_len..].to_vec()
                        } else {
                            output.clone()
                        };
                        // Target is input shifted by one (next-token prediction)
                        target = new_input[1..].to_vec();
                        target.push(new_input[0]); // wrap-around for last position
                        input = new_input;

                        eprintln!("    [speak: generated {} tokens, loss={:.4} → redirect as round {} input]",
                            output.len(), gen_result.loss, round + 2);
                    }
                }
            }
        }

        if aborted { break; }

        // Restore GPU context to build defaults if phase overrode them
        #[cfg(feature = "cuda")]
        {
            let needs_restore = batch_size != cfg.build.batch_size
                || phase_seq_len != seq_len;
            if needs_restore {
                mag_cfg.swa.seq_len = seq_len;
                gpu_context = GpuStackedContext::new(
                    n_blocks, k, d, cfg.build.batch_size, Some(&mag_cfg),
                );
                // Re-apply dormancy config after restore
                if let Some(ref floors) = cfg.model.dormancy_floor {
                    let consecutive = cfg.model.dormancy_consecutive;
                    if consecutive > 0 {
                        gpu_context.set_dormancy_config(floors.clone(), consecutive);
                    }
                }
            }
        }

        // ── Phase boundary checkpoint ─────────────────────────────
        eprintln!("  [phase {phase_idx} complete — checkpoint at step {global_step}]");
        #[cfg(feature = "cuda")]
        {
            save_checkpoint(
                &save_path, global_step, total_tokens_seen,
                checkpoint_naming, loss_last,
                &mut model_state, &state_file_path, &phase.label,
                &gpu_params,
                &gpu_context,
                &mag_cfg, &conductor, &chunk_sizes,
                &last_loaders, d, v, k,
                &mut logger,
            );
            trigger_state.record_checkpoint(total_tokens_seen as u64, loss_last);

            // Run inline probes at phase boundary
            if let Some(ref tok_path) = cfg.build.tokenizer_path {
                let default_opt = &cfg.build.optimizer;
                let snapshot = gpu_params.to_host(&mag_cfg);
                let probe_results = run_inline_probes(
                    &snapshot, &mag_cfg, tok_path, global_step,
                    d, v, k, n_blocks, &chunk_sizes,
                    cfg.build.probe_max_tokens,
                    cfg.build.probe_temperature,
                    default_opt.lr(), default_opt.beta1(), default_opt.beta2(),
                    default_opt.weight_decay(), cfg.build.max_grad_norm,
                );
                logger.log_probe_results(global_step, probe_results);
            }
        }
    }

    // Spec 78: on_unload — save if work is unsaved (covers abort and graceful exit)
    #[cfg(feature = "cuda")]
    if cfg.build.checkpoint.on_unload && trigger_state.is_stale(total_tokens_seen as u64) {
        eprintln!("  [on_unload: saving final checkpoint at step {global_step}]");
        let naming = &cfg.build.checkpoint.naming;
        save_checkpoint(
            &save_path, global_step, total_tokens_seen,
            naming, loss_last,
            &mut model_state, &state_file_path, "on_unload",
            &gpu_params,
            &gpu_context,
            &mag_cfg, &conductor, &chunk_sizes,
            &last_loaders, d, v, k,
            &mut logger,
        );
    }

    // ── Summary ──────────────────────────────────────────────────────
    let elapsed = t_start.elapsed().as_secs_f64();
    let tok_s = step_tokens as f64 / elapsed;
    eprintln!();
    eprintln!("============================================================");
    let total_segments = total_tokens_seen / TOKENS_PER_SEGMENT;
    eprintln!("  Phases:   {} complete", phases.len());
    eprintln!("  Steps:    {global_step} ({elapsed:.0}s)");
    eprintln!("  Segments: {total_segments} ({} tokens)", fmt_num(total_tokens_seen));
    eprintln!("  Tok/s:    {tok_s:.0}");
    eprintln!("  Loss:     {:.4} → {loss_last:.4}", loss_first.unwrap_or(0.0));
    // Spec 73: drain GPU buffer pool and report statistics
    #[cfg(feature = "cuda")]
    {
        let pool_stats = nl_hecate_core::gpu_buf::gpu_pool_drain();
        eprintln!("  GPU pool: {} hits, {} misses, {} unique sizes, {} cached",
            pool_stats.hits, pool_stats.misses, pool_stats.unique_sizes, pool_stats.cached_buffers);
    }
    eprintln!("============================================================");

    logger.log_build_end(global_step - resume_step, elapsed, tok_s,
        loss_first.unwrap_or(0.0), loss_last, total_tokens_seen);
}

// ── Extracted helpers ────────────────────────────────────────────────

/// Build CMS diagnostics from gpu_context state and per-block gradient norms.
/// Aggregates across blocks: gnorms and m-deltas are averaged, dormancy uses worst-case.
#[cfg(feature = "cuda")]
fn collect_cms_diagnostics(
    gpu_context: &GpuStackedContext,
    block_level_gnorms: &BlockLevelGnorms,
    k: usize,
) -> CmsDiagnostics {
    let n_blocks = block_level_gnorms.len();
    let nb = n_blocks.max(1) as f32;

    // Mean per-level gnorms across blocks
    let mut level_gnorms = vec![0.0f32; k];
    for block_gnorms in block_level_gnorms {
        for (l, &g) in block_gnorms.iter().enumerate() {
            if l < k { level_gnorms[l] += g; }
        }
    }
    for g in &mut level_gnorms { *g /= nb; }

    // M-norms: use prev_m_norms (spec 64: captured inside step() BEFORE maybe_reset_levels
    // zeros the buffers). These are pre-reset norms — the actual end-of-sequence memory state.
    let mut level_m_norms = vec![0.0f32; k];
    for block_norms in &gpu_context.prev_m_norms {
        for (l, &n) in block_norms.iter().enumerate() {
            if l < k { level_m_norms[l] += n; }
        }
    }
    for n in &mut level_m_norms { *n /= nb; }

    let mut level_m_deltas = vec![0.0f32; k];
    for block_deltas in &gpu_context.m_norm_deltas {
        for (l, &d) in block_deltas.iter().enumerate() {
            if l < k { level_m_deltas[l] += d; }
        }
    }
    for d in &mut level_m_deltas { *d /= nb; }

    // Dormancy status: worst-case across blocks per level
    let status_per_block = gpu_context.dormancy_status();
    let mut dormancy_status = vec!["active".to_string(); k];
    for block_status in &status_per_block {
        for (l, s) in block_status.iter().enumerate() {
            if l < k {
                // Escalate: active < warning < dormant
                if s == "dormant" || (s == "warning" && dormancy_status[l] == "active") {
                    dormancy_status[l] = s.clone();
                }
            }
        }
    }

    CmsDiagnostics { level_gnorms, level_m_norms, level_m_deltas, dormancy_status }
}

/// Save checkpoint with build state, and update the state file.
fn save_checkpoint(
    save_path: &str,
    global_step: usize,
    total_tokens_seen: usize,
    naming: &CheckpointNaming,
    loss: f32,
    model_state: &mut StateFile,
    state_file_path: &Path,
    phase_label: &str,
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
    let ckpt_path = checkpoint_filename(save_path, naming, global_step, total_tokens_seen);

    #[cfg(feature = "cuda")]
    {
        let host_params = gpu_params.to_host(mag_cfg);
        let host_context = gpu_context.blocks[0].to_host(k);
        let stream_position = loaders.first().map(|l| l.position as u64).unwrap_or(0);

        // Save all loader positions for batch>1 resume
        let stream_cursors: Vec<StreamCursor> = loaders.iter().map(|l| {
            StreamCursor {
                position: l.position as u64,
                chunk_id: 0,
                pulse_id: conductor.step() as u64,
                rng_state: None,
                content_hash: 0,
            }
        }).collect();

        let build_state = BuildResumeState {
            conductor: ConductorState {
                k,
                chunk_sizes: chunk_sizes.to_vec(),
                step: conductor.step(),
            },
            stream_cursor: StreamCursor {
                position: stream_position,
                chunk_id: 0,
                pulse_id: conductor.step() as u64,
                rng_state: None,
                content_hash: 0,
            },
            context: host_context,
            global_step,
            stream_cursors: stream_cursors.clone(),
            total_tokens_seen,
        };
        match save_stacked_safetensors(
            Path::new(&ckpt_path), &host_params, mag_cfg,
            Some(&build_state),
        ) {
            Ok(()) => {
                eprintln!("  [checkpoint saved: {ckpt_path}]");
                logger.log_checkpoint(global_step, &ckpt_path);

                // Spec 02: update state file with checkpoint entry
                let entry = CheckpointEntry {
                    path: ckpt_path.clone(),
                    tokens: total_tokens_seen as u64,
                    content_hash: String::new(), // TODO: SHA-256 of weight bytes (future)
                    timestamp: state_file::iso8601_now(),
                    health: HealthSnapshot {
                        loss,
                        m_norm_per_level: vec![],
                        gate_alpha_mean_per_level: vec![],
                        gate_theta_mean_per_level: vec![],
                        cms_activations_since_restore: vec![],
                    },
                    session: SessionInfo::Build {
                        label: phase_label.into(),
                        dataset: String::new(),
                        dataset_hash: String::new(),
                        config_snapshot: String::new(),
                    },
                };
                state_file::record_checkpoint(model_state, entry, &stream_cursors);
                if let Err(e) = state_file::save_state_file(state_file_path, model_state) {
                    eprintln!("WARNING: state file update failed: {e}");
                }
            }
            Err(e) => {
                eprintln!("ERROR: checkpoint save failed: {e}");
            }
        }
    }
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

/// Compute actual per-level firing counts for this step.
/// An active level fires seq_len / chunk_size times per step.
fn level_firings(pulse: &Pulse, seq_len: usize, chunk_sizes: &[usize]) -> Vec<usize> {
    pulse.active_levels.iter().enumerate().map(|(i, &active)| {
        if active {
            let cs = chunk_sizes.get(i).copied().unwrap_or(1);
            seq_len / cs
        } else {
            0
        }
    }).collect()
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
