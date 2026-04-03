/// Spec 70 — Unified CLI Loop: one core method, output handlers.
/// Spec 76 — Split into step_micro() + step_update() for gradient accumulation.
///
/// `step_micro()` runs forward → backward only (immutable weights).
/// `step_update()` runs optimizer on accumulated gradients (mutates weights).
/// `step()` is a convenience wrapper: step_micro() + step_update() in one call.
/// `generate()` is autoregressive sampling via step() with deferred backward.
///
/// Every token the model sees goes through step(). There is no inference-without-learning.
/// CS-10: no train/eval distinction. CS-18: forward pass IS the only API.

#[cfg(feature = "cuda")]
use nl_hecate_core::conductor::Pulse;

#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_forward::{
    gpu_stacked_forward_tokens, gpu_stacked_forward_sequence,
    gpu_cross_entropy_loss,
    StackedDecodeWorkspace, ActivationWindow,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_backward::{gpu_stacked_backward, GpuStackedGrads};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_optimizer::{
    GpuStackedAdamWState, gpu_stacked_adamw_update, gpu_stacked_sync_embed_weights,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_forward::GpuKVCache;
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_profiler::GpuProfiler;
#[cfg(feature = "cuda")]
use nl_hecate_core::model::MAGConfig;
#[cfg(feature = "cuda")]
use nl_hecate_core::conductor::Conductor;

use crate::config::OptimizerConfig;
use crate::sample::sample_token;

// ── Types ───────────────────────────────────────────────────────────

/// Per-level gradient norms extracted from backward pass (one Vec<f32> per block).
#[cfg(feature = "cuda")]
pub type BlockLevelGnorms = Vec<Vec<f32>>;

/// Result of step_micro(): forward → backward only. No optimizer.
#[cfg(feature = "cuda")]
pub struct MicroStepResult {
    pub logits: Vec<f32>,           // [vocab_size] — last token's logits
    pub loss: f32,
    pub grads: GpuStackedGrads,
    pub block_level_gnorms: BlockLevelGnorms,
    pub pulse: Pulse,               // snapshot for logging
}

/// Result of a single step: forward → backward → update.
#[cfg(feature = "cuda")]
pub struct StepResult {
    pub logits: Vec<f32>,           // [vocab_size] — last token's logits for sampling
    pub loss: f32,
    pub grad_norm: f32,
    pub block_level_gnorms: BlockLevelGnorms,
    pub pulse: Pulse,               // snapshot for logging
}

/// Result of autoregressive generation via step().
#[cfg(feature = "cuda")]
pub struct GenerateResult {
    pub tokens: Vec<usize>,
    pub loss: f32,                  // average loss across generated tokens (from deferred backward)
    pub grad_norm: f32,             // grad norm from deferred backward
    pub block_level_gnorms: BlockLevelGnorms,
}

// ── Core: step_micro() ─────────────────────────────────────────────

/// Spec 76: Forward + backward only. Returns gradients. Does NOT run optimizer.
///
/// Weights are immutable during micro-steps — gradients accumulate across
/// multiple calls before one optimizer step fires.
#[cfg(feature = "cuda")]
pub fn step_micro(
    gpu_params: &GpuStackedParams,     // immutable — weights don't change during accumulation
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    tokens: &[usize],
    targets: &[usize],
    conductor: &mut Conductor,
    profiler: &mut Option<GpuProfiler>,
    log_this: bool,
) -> MicroStepResult {
    if let Some(ref mut p) = profiler { p.step_start(); }

    // Spec 71: full-sequence forward for build mode (s > 1).
    let (last_logits, cache) = gpu_stacked_forward_sequence(
        gpu_params, mag_cfg, tokens, targets,
        conductor, gpu_context,
    );

    // Spec 72: GPU-side cross-entropy
    let v = mag_cfg.swa.vocab_size;
    let loss = gpu_cross_entropy_loss(&cache.logits, &cache.target_ids_gpu, targets, v, tokens.len());

    // Capture pulse before backward consumes it
    let pulse = cache.pulse.clone();

    let grads = gpu_stacked_backward(
        gpu_params, mag_cfg, &cache, profiler, log_this,
    );

    // Extract per-level gradient norms
    let block_level_gnorms: BlockLevelGnorms = grads.blocks.iter()
        .map(|bg| bg.level_output_gnorms.clone())
        .collect();

    if let Some(ref mut p) = profiler { p.step_stop(); }

    MicroStepResult { logits: last_logits, loss, grads, block_level_gnorms, pulse }
}

// ── Core: step_update() ────────────────────────────────────────────

/// Spec 76: Optimizer step on accumulated gradients. Mutates weights.
///
/// Called once per logical step after all micro-steps have accumulated.
/// Handles: grad clip → AdamW → weight tying → M-norm tracking → level resets.
#[cfg(feature = "cuda")]
pub fn step_update(
    gpu_params: &mut GpuStackedParams,
    grads: &mut GpuStackedGrads,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    pulse: &Pulse,
    opt: &OptimizerConfig,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    v: usize,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
    gpu_context: &mut GpuStackedContext,
    profiler: &mut Option<GpuProfiler>,
    log_this: bool,
) -> f32 {
    // Dispatch optimizer by type
    let gnorm = match opt.optimizer_type() {
        "adamw" => {
            if adamw_state.is_none() {
                *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
            }
            let state = adamw_state.as_mut().unwrap();
            gpu_stacked_adamw_update(
                gpu_params, grads, state,
                pulse,
                lr, opt.beta1(), opt.beta2(), 1e-8,
                opt.weight_decay(), max_grad_norm,
                false, // freeze_embed
                profiler,
            )
        }
        other => {
            panic!("Unsupported optimizer type: \"{other}\". Only \"adamw\" is currently implemented.");
        }
    };

    // Weight tying
    gpu_stacked_sync_embed_weights(gpu_params, d, v);

    // Spec 64: capture pre-reset M norms on log steps (before reset zeros the buffers)
    if log_this {
        gpu_context.update_m_norm_tracking();
    }

    // Selective periodic reset (spec 57)
    maybe_reset_levels(pulse, reset_intervals, fire_counts, gpu_context);

    gnorm
}

// ── Core: step() — convenience wrapper ─────────────────────────────

/// Process a chunk of tokens: forward → backward → update.
/// Convenience wrapper around step_micro() + step_update() for callers
/// that don't need gradient accumulation (generate, think_rounds, etc.).
#[cfg(feature = "cuda")]
pub fn step(
    gpu_params: &mut GpuStackedParams,
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    tokens: &[usize],
    targets: &[usize],
    conductor: &mut Conductor,
    opt: &OptimizerConfig,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    v: usize,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
    profiler: &mut Option<GpuProfiler>,
    log_this: bool,
) -> StepResult {
    let MicroStepResult { logits, loss, mut grads, block_level_gnorms, pulse } = step_micro(
        gpu_params, mag_cfg, gpu_context,
        tokens, targets, conductor, profiler, log_this,
    );

    let gnorm = step_update(
        gpu_params, &mut grads, adamw_state,
        &pulse, opt, lr, max_grad_norm, d, v,
        reset_intervals, fire_counts, gpu_context, profiler, log_this,
    );

    StepResult { logits, loss, grad_norm: gnorm, block_level_gnorms, pulse }
}

// ── Core: generate() ────────────────────────────────────────────────

/// Generate tokens autoregressively with deferred backward.
///
/// Each generated token goes through forward (memory updates). Once generation is
/// complete, the full generated sequence runs backward so the model learns from its
/// own output. This is NOT a separate code path — it uses the same forward pipeline
/// as step().
///
/// Deferred backward approach (spec 70 option 2): accumulate tokens in ActivationWindow
/// during generation, run one backward on the full sequence. More GPU-efficient than
/// per-token backward.
#[cfg(feature = "cuda")]
pub fn generate(
    gpu_params: &mut GpuStackedParams,
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    conductor: &mut Conductor,
    opt: &OptimizerConfig,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    v: usize,
    prompt: &[usize],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    stop_token: Option<usize>,
    profiler: &mut Option<GpuProfiler>,
) -> GenerateResult {
    let n_blocks = gpu_params.n_blocks();
    let kv_len = prompt.len().max(mag_cfg.swa.seq_len) + max_tokens;
    let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
        .map(|_| GpuKVCache::new(kv_len, d, 1))
        .collect();
    let mut ws = StackedDecodeWorkspace::new(n_blocks, d, v);

    // ActivationWindow collects all tokens (prompt + generated) for deferred backward
    let window_cap = prompt.len() + max_tokens;
    let mut window = ActivationWindow::new(window_cap);

    // Forward the prompt — model learns prompt context via memory updates
    let mut last_logits = gpu_stacked_forward_tokens(
        gpu_params, mag_cfg, prompt,
        conductor, gpu_context, &mut kv_caches, &mut ws,
        &mut window,
    );

    // Autoregressive generation loop
    let mut generated = Vec::new();

    for _ in 0..max_tokens {
        let next_tok = sample_token(&last_logits, temperature, top_k);
        generated.push(next_tok);

        // Forward the token so the model sees it (memory updates + activation saved)
        last_logits = gpu_stacked_forward_tokens(
            gpu_params, mag_cfg, &[next_tok],
            conductor, gpu_context, &mut kv_caches, &mut ws,
            &mut window,
        );

        // Stop AFTER forwarding — the model learns the stop decision.
        // Remove the stop token from returned tokens (callers don't want it in output)
        // but it remains in the window for deferred backward.
        if let Some(stop) = stop_token {
            if next_tok == stop {
                generated.pop(); // strip from output
                break;
            }
        }
    }

    // ── Deferred backward: learn from the full generated sequence ────
    // Build targets: shifted by 1 (next-token prediction).
    // Window contains prompt + generated tokens. Targets = window[1..] + mask.
    let win_len = window.len();
    let mut target_ids: Vec<usize> = Vec::with_capacity(win_len);
    for i in 1..win_len {
        target_ids.push(window.entries[i].token_id);
    }
    target_ids.push(v); // mask last position (no target available)

    let cache = window.assemble_cache(mag_cfg, &target_ids);
    let loss = gpu_cross_entropy_loss(&cache.logits, &cache.target_ids_gpu, &target_ids, v, win_len);

    let (grad_norm, block_level_gnorms) = if !loss.is_nan() && !loss.is_infinite() {
        let mut grads = gpu_stacked_backward(
            gpu_params, mag_cfg, &cache, profiler, false,
        );

        let blg: BlockLevelGnorms = grads.blocks.iter()
            .map(|bg| bg.level_output_gnorms.clone())
            .collect();

        if adamw_state.is_none() {
            *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
        }
        let state = adamw_state.as_mut().unwrap();
        let gnorm = gpu_stacked_adamw_update(
            gpu_params, &mut grads, state,
            &cache.pulse,
            lr, opt.beta1(), opt.beta2(), 1e-8,
            opt.weight_decay(), max_grad_norm,
            false, profiler,
        );
        gpu_stacked_sync_embed_weights(gpu_params, d, v);
        gpu_context.update_m_norm_tracking();

        (gnorm, blg)
    } else {
        (0.0, vec![vec![]; n_blocks])
    };

    GenerateResult { tokens: generated, loss, grad_norm, block_level_gnorms }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Selective periodic reset (spec 57).
#[cfg(feature = "cuda")]
pub fn maybe_reset_levels(
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
