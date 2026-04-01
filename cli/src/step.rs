/// Spec 70 — Unified CLI Loop: one core method, output handlers.
///
/// `step()` is the atomic NLM operation: forward → backward → update → logits.
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
    gpu_stacked_forward_tokens, StackedDecodeWorkspace, ActivationWindow,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward;
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

// ── Core: step() ────────────────────────────────────────────────────

/// Process a chunk of tokens: forward → backward → update.
/// This is the ONE thing an NLM does. Every token goes through this.
///
/// Returns StepResult with logits (for sampling), loss, grad_norm, and per-block gnorms.
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
    if let Some(ref mut p) = profiler { p.step_start(); }

    // Fresh KV caches per chunk (no cross-chunk attention leaking)
    let n_blocks = gpu_params.n_blocks();
    let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
        .map(|_| GpuKVCache::new(tokens.len(), d, 1))
        .collect();
    let mut ws = StackedDecodeWorkspace::new(n_blocks, d, v);

    // Forward: process all tokens through unified path, saving activations
    let mut window = ActivationWindow::new(tokens.len());
    let last_logits = gpu_stacked_forward_tokens(
        gpu_params, mag_cfg, tokens,
        conductor, gpu_context, &mut kv_caches, &mut ws,
        &mut window,
    );

    // Assemble activation cache for backward (recomputes batched SWA attention)
    let cache = window.assemble_cache(mag_cfg, targets);

    // Loss (host-side cross-entropy from assembled logits)
    let loss = host_cross_entropy_loss(&cache.logits, targets, v, tokens.len());

    // Capture pulse before backward consumes it
    let pulse = cache.pulse.clone();

    let mut grads = gpu_stacked_backward(
        gpu_params, mag_cfg, &cache, profiler, log_this,
    );

    // Extract per-level gradient norms before optimizer consumes grads
    let block_level_gnorms: BlockLevelGnorms = grads.blocks.iter()
        .map(|bg| bg.level_output_gnorms.clone())
        .collect();

    // Dispatch optimizer by type
    let gnorm = match opt.optimizer_type() {
        "adamw" => {
            if adamw_state.is_none() {
                *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
            }
            let state = adamw_state.as_mut().unwrap();
            gpu_stacked_adamw_update(
                gpu_params, &mut grads, state,
                &cache.pulse,
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

    if let Some(ref mut p) = profiler { p.step_stop(); }

    // Weight tying
    gpu_stacked_sync_embed_weights(gpu_params, d, v);

    // Spec 64: capture pre-reset M norms on log steps (before reset zeros the buffers)
    if log_this {
        gpu_context.update_m_norm_tracking();
    }

    // Selective periodic reset (spec 57)
    maybe_reset_levels(&pulse, reset_intervals, fire_counts, gpu_context);

    StepResult { logits: last_logits, loss, grad_norm: gnorm, block_level_gnorms, pulse }
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
        if let Some(stop) = stop_token {
            if next_tok == stop {
                break;
            }
        }
        generated.push(next_tok);

        last_logits = gpu_stacked_forward_tokens(
            gpu_params, mag_cfg, &[next_tok],
            conductor, gpu_context, &mut kv_caches, &mut ws,
            &mut window,
        );
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
    let loss = host_cross_entropy_loss(&cache.logits, &target_ids, v, win_len);

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

/// Host-side cross-entropy loss from GPU logits buffer + target IDs.
#[cfg(feature = "cuda")]
pub fn host_cross_entropy_loss(
    logits_gpu: &nl_hecate_core::gpu_buf::GpuBuf<f32>,
    target_ids: &[usize],
    vocab_size: usize,
    seq_len: usize,
) -> f32 {
    let mut logits_host = vec![0.0f32; seq_len * vocab_size];
    logits_gpu.copy_to_host(&mut logits_host);

    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for t in 0..seq_len {
        let target = target_ids[t];
        if target >= vocab_size { continue; } // masked position

        let offset = t * vocab_size;
        let row = &logits_host[offset..offset + vocab_size];

        // log-sum-exp for numerical stability
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln() + max_logit;
        let log_prob = row[target] - lse;
        total_loss -= log_prob;
        count += 1;
    }

    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

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
