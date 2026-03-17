/// GPU-resident AdamW optimizer for stacked multi-block models.
///
/// Maintains moment buffers for shared params + per-block SWA/LN/CMS weights.
/// Reuses `adamw_one` kernel from gpu_optimizer.rs.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuStackedParams, GpuBlockParams};
#[cfg(feature = "cuda")]
use crate::gpu_stacked_backward::GpuStackedGrads;
#[cfg(feature = "cuda")]
use crate::gpu_optimizer::adamw_one;
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;

// ══════════════════════════════════════════════════════════════════════
// Moment buffer structs
// ══════════════════════════════════════════════════════════════════════

/// Moment buffers for one block's SWA + LN weights.
#[cfg(feature = "cuda")]
struct MomentBlock {
    m_q: GpuBuf<f32>, v_q: GpuBuf<f32>,
    m_k: GpuBuf<f32>, v_k: GpuBuf<f32>,
    m_v: GpuBuf<f32>, v_v: GpuBuf<f32>,
    m_o: GpuBuf<f32>, v_o: GpuBuf<f32>,
    m_ln_attn_gamma: GpuBuf<f32>, v_ln_attn_gamma: GpuBuf<f32>,
    m_ln_attn_beta: GpuBuf<f32>, v_ln_attn_beta: GpuBuf<f32>,
    m_ln_mem_gamma: GpuBuf<f32>, v_ln_mem_gamma: GpuBuf<f32>,
    m_ln_mem_beta: GpuBuf<f32>, v_ln_mem_beta: GpuBuf<f32>,
    // Learnable level aggregation (host-side scalars)
    m_alpha_mem: Vec<f32>, v_alpha_mem: Vec<f32>,
    levels: Vec<MomentLevelStacked>,
}

/// Moment buffers for one CMS level within a block.
#[cfg(feature = "cuda")]
struct MomentLevelStacked {
    m_w_k_mem: GpuBuf<f32>, v_w_k_mem: GpuBuf<f32>,
    m_w_v_mem: GpuBuf<f32>, v_w_v_mem: GpuBuf<f32>,
    m_w_q_mem: GpuBuf<f32>, v_w_q_mem: GpuBuf<f32>,
    m_w_alpha: GpuBuf<f32>, v_w_alpha: GpuBuf<f32>,
    m_b_alpha: GpuBuf<f32>, v_b_alpha: GpuBuf<f32>,
    m_w_theta: GpuBuf<f32>, v_w_theta: GpuBuf<f32>,
    m_b_theta: GpuBuf<f32>, v_b_theta: GpuBuf<f32>,
    m_w_eta: GpuBuf<f32>, v_w_eta: GpuBuf<f32>,
    m_b_eta: GpuBuf<f32>, v_b_eta: GpuBuf<f32>,
    level_step: u32,
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedAdamWState
// ══════════════════════════════════════════════════════════════════════

/// AdamW state for stacked model: shared embed/unembed/ln_final + N blocks.
#[cfg(feature = "cuda")]
pub struct GpuStackedAdamWState {
    // Shared param moments
    m_embed: GpuBuf<f32>, v_embed: GpuBuf<f32>,
    m_unembed: GpuBuf<f32>, v_unembed: GpuBuf<f32>,
    m_ln_final_gamma: GpuBuf<f32>, v_ln_final_gamma: GpuBuf<f32>,
    m_ln_final_beta: GpuBuf<f32>, v_ln_final_beta: GpuBuf<f32>,
    // Per-block moments
    blocks: Vec<MomentBlock>,
    pub step: u32,
    // Scratch buffers for gradient norm computation
    norm_scratch: GpuBuf<f32>,
    norm_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuStackedAdamWState {
    /// Create zero-initialized optimizer state matching stacked param shapes.
    pub fn from_params(params: &GpuStackedParams) -> Self {
        let blocks = params.blocks.iter().map(|bp| {
            let levels = bp.levels.iter().map(|lp| MomentLevelStacked {
                m_w_k_mem: GpuBuf::zeros(lp.w_k_mem.len()),
                v_w_k_mem: GpuBuf::zeros(lp.w_k_mem.len()),
                m_w_v_mem: GpuBuf::zeros(lp.w_v_mem.len()),
                v_w_v_mem: GpuBuf::zeros(lp.w_v_mem.len()),
                m_w_q_mem: GpuBuf::zeros(lp.w_q_mem.len()),
                v_w_q_mem: GpuBuf::zeros(lp.w_q_mem.len()),
                m_w_alpha: GpuBuf::zeros(lp.w_alpha.len()),
                v_w_alpha: GpuBuf::zeros(lp.w_alpha.len()),
                m_b_alpha: GpuBuf::zeros(lp.b_alpha.len()),
                v_b_alpha: GpuBuf::zeros(lp.b_alpha.len()),
                m_w_theta: GpuBuf::zeros(lp.w_theta.len()),
                v_w_theta: GpuBuf::zeros(lp.w_theta.len()),
                m_b_theta: GpuBuf::zeros(lp.b_theta.len()),
                v_b_theta: GpuBuf::zeros(lp.b_theta.len()),
                m_w_eta: GpuBuf::zeros(lp.w_eta.len()),
                v_w_eta: GpuBuf::zeros(lp.w_eta.len()),
                m_b_eta: GpuBuf::zeros(lp.b_eta.len()),
                v_b_eta: GpuBuf::zeros(lp.b_eta.len()),
                level_step: 0,
            }).collect();

            MomentBlock {
                m_q: GpuBuf::zeros(bp.w_q.len()), v_q: GpuBuf::zeros(bp.w_q.len()),
                m_k: GpuBuf::zeros(bp.w_k.len()), v_k: GpuBuf::zeros(bp.w_k.len()),
                m_v: GpuBuf::zeros(bp.w_v.len()), v_v: GpuBuf::zeros(bp.w_v.len()),
                m_o: GpuBuf::zeros(bp.w_o.len()), v_o: GpuBuf::zeros(bp.w_o.len()),
                m_ln_attn_gamma: GpuBuf::zeros(bp.ln_attn_gamma.len()),
                v_ln_attn_gamma: GpuBuf::zeros(bp.ln_attn_gamma.len()),
                m_ln_attn_beta: GpuBuf::zeros(bp.ln_attn_beta.len()),
                v_ln_attn_beta: GpuBuf::zeros(bp.ln_attn_beta.len()),
                m_ln_mem_gamma: GpuBuf::zeros(bp.ln_mem_gamma.len()),
                v_ln_mem_gamma: GpuBuf::zeros(bp.ln_mem_gamma.len()),
                m_ln_mem_beta: GpuBuf::zeros(bp.ln_mem_beta.len()),
                v_ln_mem_beta: GpuBuf::zeros(bp.ln_mem_beta.len()),
                m_alpha_mem: vec![0.0f32; bp.alpha_mem.len()],
                v_alpha_mem: vec![0.0f32; bp.alpha_mem.len()],
                levels,
            }
        }).collect();

        // Compute max buffer length for norm scratch allocation
        let mut max_len = params.w_embed.len()
            .max(params.w_unembed.len())
            .max(params.ln_final_gamma.len());
        for bp in &params.blocks {
            for buf in [&bp.w_q, &bp.w_k, &bp.w_v, &bp.w_o,
                        &bp.ln_attn_gamma, &bp.ln_attn_beta,
                        &bp.ln_mem_gamma, &bp.ln_mem_beta] {
                max_len = max_len.max(buf.len());
            }
            for lp in &bp.levels {
                for buf in [&lp.w_k_mem, &lp.w_v_mem, &lp.w_q_mem,
                            &lp.w_alpha, &lp.b_alpha, &lp.w_theta, &lp.b_theta,
                            &lp.w_eta, &lp.b_eta] {
                    max_len = max_len.max(buf.len());
                }
            }
        }
        let max_partials = max_len / 256 + 1;

        GpuStackedAdamWState {
            m_embed: GpuBuf::zeros(params.w_embed.len()),
            v_embed: GpuBuf::zeros(params.w_embed.len()),
            m_unembed: GpuBuf::zeros(params.w_unembed.len()),
            v_unembed: GpuBuf::zeros(params.w_unembed.len()),
            m_ln_final_gamma: GpuBuf::zeros(params.ln_final_gamma.len()),
            v_ln_final_gamma: GpuBuf::zeros(params.ln_final_gamma.len()),
            m_ln_final_beta: GpuBuf::zeros(params.ln_final_beta.len()),
            v_ln_final_beta: GpuBuf::zeros(params.ln_final_beta.len()),
            blocks,
            step: 0,
            norm_scratch: GpuBuf::zeros(max_partials),
            norm_host: vec![0.0f32; max_partials],
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_adamw_update — full weight update
// ══════════════════════════════════════════════════════════════════════

/// Compute L2 norm of all stacked gradient buffers on GPU.
/// When `skip_embed` is true, excludes d_w_embed and d_w_unembed from the norm
/// to avoid over-clipping trainable gradients when embeddings are frozen.
#[cfg(feature = "cuda")]
fn gpu_stacked_grad_norm_ex(grads: &GpuStackedGrads, state: &mut GpuStackedAdamWState, skip_embed: bool) -> f32 {
    let mut total_sq = 0.0f64;

    // NOTE: This accumulates partial sums with a host sync per tensor. For large
    // n_blocks * k this becomes many round-trips. A single-kernel reduction across
    // all gradient buffers would be faster but requires a flat buffer layout.
    // Acceptable for shakedown builds; optimize when profiling shows it matters.
    let mut accum = |g: &GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let mut num_blocks: i32 = 0;
        let err = unsafe {
            crate::cuda_ffi::grad_norm_sq_cuda(
                g.as_ptr(), state.norm_scratch.ptr(),
                n, &mut num_blocks,
            )
        };
        assert_eq!(err, 0, "grad_norm_sq_cuda failed with cudaError_t={}", err);
        crate::dispatch::cuda_sync();
        let nb = num_blocks as usize;
        state.norm_scratch.slice(0, nb).copy_to_host(&mut state.norm_host[..nb]);
        for i in 0..nb {
            total_sq += state.norm_host[i] as f64;
        }
    };

    // Shared grads (skip embed/unembed when frozen)
    if !skip_embed {
        accum(&grads.d_w_embed);
        accum(&grads.d_w_unembed);
    }
    accum(&grads.d_ln_final_gamma);
    accum(&grads.d_ln_final_beta);

    // Per-block grads
    for bg in &grads.blocks {
        accum(&bg.d_w_q);
        accum(&bg.d_w_k);
        accum(&bg.d_w_v);
        accum(&bg.d_w_o);
        accum(&bg.d_ln_attn_gamma);
        accum(&bg.d_ln_attn_beta);
        accum(&bg.d_ln_mem_gamma);
        accum(&bg.d_ln_mem_beta);
        for lg in &bg.levels {
            accum(&lg.d_w_k_mem);
            accum(&lg.d_w_v_mem);
            accum(&lg.d_w_q_mem);
            accum(&lg.d_w_alpha);
            accum(&lg.d_b_alpha);
            accum(&lg.d_w_theta);
            accum(&lg.d_b_theta);
            accum(&lg.d_w_eta);
            accum(&lg.d_b_eta);
        }
    }
    drop(accum);

    // alpha_mem gradients (host-side, accumulated after closure dropped)
    for bg in &grads.blocks {
        for &g in &bg.d_alpha_mem {
            total_sq += (g as f64) * (g as f64);
        }
    }

    total_sq.sqrt() as f32
}

/// Per-block gradient norm results.
#[cfg(feature = "cuda")]
pub struct PerBlockGradNorms {
    /// Aggregate L2 norm per block (SWA + all levels + alpha_mem).
    /// Length = n_blocks. Used for depth specialization CV metric.
    pub block_norms: Vec<f32>,
    /// L0-only L2 norm per block (only level[0] memory params).
    /// Length = n_blocks. Used for the "L0 gnorm per block > 0.01" floor
    /// check in spec 19 promotion criteria.
    pub l0_block_norms: Vec<f32>,
}

/// Compute per-block L2 gradient norms. Returns both aggregate (SWA + all
/// levels) and L0-only norms per block. Called before global clipping so
/// values reflect the true per-block learning signal. Shared params (embed,
/// unembed, ln_final) are excluded because they contribute equally to every
/// block's gradient and would mask depth specialization.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_per_block_grad_norms(
    grads: &GpuStackedGrads,
    state: &mut GpuStackedAdamWState,
) -> PerBlockGradNorms {
    let n = grads.blocks.len();
    let mut block_norms = Vec::with_capacity(n);
    let mut l0_block_norms = Vec::with_capacity(n);

    for bg in &grads.blocks {
        let mut block_sq = 0.0f64;
        let mut l0_sq = 0.0f64;

        let mut accum_to = |g: &GpuBuf<f32>, target: &mut f64| {
            let n = g.len() as i32;
            if n == 0 { return; }
            let mut num_blocks: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::grad_norm_sq_cuda(
                    g.as_ptr(), state.norm_scratch.ptr(),
                    n, &mut num_blocks,
                )
            };
            assert_eq!(err, 0, "grad_norm_sq_cuda failed with cudaError_t={}", err);
            crate::dispatch::cuda_sync();
            let nb = num_blocks as usize;
            state.norm_scratch.slice(0, nb).copy_to_host(&mut state.norm_host[..nb]);
            for i in 0..nb {
                *target += state.norm_host[i] as f64;
            }
        };

        // SWA projections for this block (aggregate only)
        accum_to(&bg.d_w_q, &mut block_sq);
        accum_to(&bg.d_w_k, &mut block_sq);
        accum_to(&bg.d_w_v, &mut block_sq);
        accum_to(&bg.d_w_o, &mut block_sq);
        accum_to(&bg.d_ln_attn_gamma, &mut block_sq);
        accum_to(&bg.d_ln_attn_beta, &mut block_sq);
        accum_to(&bg.d_ln_mem_gamma, &mut block_sq);
        accum_to(&bg.d_ln_mem_beta, &mut block_sq);

        // All levels within this block. For L0, accumulate into both
        // block_sq and l0_sq without redundant kernel launches.
        for (li, lg) in bg.levels.iter().enumerate() {
            let target = if li == 0 { &mut l0_sq } else { &mut block_sq };
            accum_to(&lg.d_w_k_mem, target);
            accum_to(&lg.d_w_v_mem, target);
            accum_to(&lg.d_w_q_mem, target);
            accum_to(&lg.d_w_alpha, target);
            accum_to(&lg.d_b_alpha, target);
            accum_to(&lg.d_w_theta, target);
            accum_to(&lg.d_b_theta, target);
            accum_to(&lg.d_w_eta, target);
            accum_to(&lg.d_b_eta, target);
        }
        drop(accum_to);

        // L0 energy is part of the aggregate
        block_sq += l0_sq;

        // alpha_mem gradients (host-side)
        for &g in &bg.d_alpha_mem {
            block_sq += (g as f64) * (g as f64);
        }

        block_norms.push(block_sq.sqrt() as f32);
        l0_block_norms.push(l0_sq.sqrt() as f32);
    }

    PerBlockGradNorms { block_norms, l0_block_norms }
}

/// Scale all stacked gradient buffers by a constant factor (for clipping).
#[cfg(feature = "cuda")]
fn gpu_stacked_scale_grads_ex(grads: &mut GpuStackedGrads, scale: f32, skip_embed: bool) {
    let scale_buf = |g: &mut GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let err = unsafe {
            crate::cuda_ffi::grad_scale_cuda(g.ptr(), scale, n)
        };
        assert_eq!(err, 0, "grad_scale_cuda failed with cudaError_t={}", err);
    };

    // Shared grads (skip embed/unembed when frozen)
    if !skip_embed {
        scale_buf(&mut grads.d_w_embed);
        scale_buf(&mut grads.d_w_unembed);
    }
    scale_buf(&mut grads.d_ln_final_gamma);
    scale_buf(&mut grads.d_ln_final_beta);

    // Per-block grads
    for bg in &mut grads.blocks {
        scale_buf(&mut bg.d_w_q);
        scale_buf(&mut bg.d_w_k);
        scale_buf(&mut bg.d_w_v);
        scale_buf(&mut bg.d_w_o);
        scale_buf(&mut bg.d_ln_attn_gamma);
        scale_buf(&mut bg.d_ln_attn_beta);
        scale_buf(&mut bg.d_ln_mem_gamma);
        scale_buf(&mut bg.d_ln_mem_beta);
        for lg in &mut bg.levels {
            scale_buf(&mut lg.d_w_k_mem);
            scale_buf(&mut lg.d_w_v_mem);
            scale_buf(&mut lg.d_w_q_mem);
            scale_buf(&mut lg.d_w_alpha);
            scale_buf(&mut lg.d_b_alpha);
            scale_buf(&mut lg.d_w_theta);
            scale_buf(&mut lg.d_b_theta);
            scale_buf(&mut lg.d_w_eta);
            scale_buf(&mut lg.d_b_eta);
        }
        // alpha_mem gradients (host-side)
        for g in &mut bg.d_alpha_mem {
            *g *= scale;
        }
    }

    crate::dispatch::cuda_sync();
}

/// Full AdamW update for stacked model. Updates all params in-place.
/// Returns pre-clip gradient norm (for logging). 0.0 if clipping disabled.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_adamw_update(
    params: &mut GpuStackedParams,
    grads: &mut GpuStackedGrads,
    state: &mut GpuStackedAdamWState,
    pulse: &Pulse,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    max_grad_norm: f32,
    freeze_embed: bool,
) -> f32 {
    state.step += 1;
    let t = state.step as f32;
    let bc1_inv = 1.0 / (1.0 - beta1.powf(t));
    let bc2_inv = 1.0 / (1.0 - beta2.powf(t));

    // Gradient clipping (exclude frozen embed grads from norm to avoid over-clipping)
    let grad_norm = if max_grad_norm > 0.0 {
        let norm = gpu_stacked_grad_norm_ex(grads, state, freeze_embed);
        if norm > max_grad_norm {
            let scale = max_grad_norm / norm;
            gpu_stacked_scale_grads_ex(grads, scale, freeze_embed);
        }
        norm
    } else {
        0.0
    };

    // ── Shared params (always active, unless frozen) ──────────────────
    if !freeze_embed {
        adamw_one(&mut params.w_embed, &grads.d_w_embed,
                  &mut state.m_embed, &mut state.v_embed,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut params.w_unembed, &grads.d_w_unembed,
                  &mut state.m_unembed, &mut state.v_unembed,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    }
    adamw_one(&mut params.ln_final_gamma, &grads.d_ln_final_gamma,
              &mut state.m_ln_final_gamma, &mut state.v_ln_final_gamma,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.ln_final_beta, &grads.d_ln_final_beta,
              &mut state.m_ln_final_beta, &mut state.v_ln_final_beta,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

    // ── Per-block params ───────────────────────────────────────────────
    for (b, (bp, bg)) in params.blocks.iter_mut().zip(grads.blocks.iter()).enumerate() {
        let mb = &mut state.blocks[b];

        // SWA projections (always active)
        adamw_one(&mut bp.w_q, &bg.d_w_q, &mut mb.m_q, &mut mb.v_q,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_k, &bg.d_w_k, &mut mb.m_k, &mut mb.v_k,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_v, &bg.d_w_v, &mut mb.m_v, &mut mb.v_v,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_o, &bg.d_w_o, &mut mb.m_o, &mut mb.v_o,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

        // LayerNorms
        adamw_one(&mut bp.ln_attn_gamma, &bg.d_ln_attn_gamma,
                  &mut mb.m_ln_attn_gamma, &mut mb.v_ln_attn_gamma,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.ln_attn_beta, &bg.d_ln_attn_beta,
                  &mut mb.m_ln_attn_beta, &mut mb.v_ln_attn_beta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.ln_mem_gamma, &bg.d_ln_mem_gamma,
                  &mut mb.m_ln_mem_gamma, &mut mb.v_ln_mem_gamma,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.ln_mem_beta, &bg.d_ln_mem_beta,
                  &mut mb.m_ln_mem_beta, &mut mb.v_ln_mem_beta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

        // alpha_mem: learnable level aggregation weights (host-side AdamW)
        // Spec: specs/infrastructure/21_stacked_alpha_aggregation.md
        {
            let k = bg.d_alpha_mem.len();
            let mut alpha_host = vec![0.0f32; k];
            bp.alpha_mem.slice(0, k).copy_to_host(&mut alpha_host);
            for i in 0..k {
                let g = bg.d_alpha_mem[i];
                mb.m_alpha_mem[i] = beta1 * mb.m_alpha_mem[i] + (1.0 - beta1) * g;
                mb.v_alpha_mem[i] = beta2 * mb.v_alpha_mem[i] + (1.0 - beta2) * g * g;
                let m_hat = mb.m_alpha_mem[i] * bc1_inv;
                let v_hat = mb.v_alpha_mem[i] * bc2_inv;
                alpha_host[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * alpha_host[i]);
            }
            bp.alpha_mem.copy_from_host(&alpha_host);
        }

        // Per-level CMS weights (pulse-gated)
        for (i, (lp, lg)) in bp.levels.iter_mut().zip(bg.levels.iter()).enumerate() {
            if i >= pulse.active_levels.len() || !pulse.active_levels[i] {
                continue;
            }
            let ml = &mut mb.levels[i];
            ml.level_step += 1;
            let lt = ml.level_step as f32;
            let lbc1_inv = 1.0 / (1.0 - beta1.powf(lt));
            let lbc2_inv = 1.0 / (1.0 - beta2.powf(lt));

            adamw_one(&mut lp.w_k_mem, &lg.d_w_k_mem, &mut ml.m_w_k_mem, &mut ml.v_w_k_mem,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.w_v_mem, &lg.d_w_v_mem, &mut ml.m_w_v_mem, &mut ml.v_w_v_mem,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.w_q_mem, &lg.d_w_q_mem, &mut ml.m_w_q_mem, &mut ml.v_w_q_mem,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);

            adamw_one(&mut lp.w_alpha, &lg.d_w_alpha, &mut ml.m_w_alpha, &mut ml.v_w_alpha,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.b_alpha, &lg.d_b_alpha, &mut ml.m_b_alpha, &mut ml.v_b_alpha,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.w_theta, &lg.d_w_theta, &mut ml.m_w_theta, &mut ml.v_w_theta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.b_theta, &lg.d_b_theta, &mut ml.m_b_theta, &mut ml.v_b_theta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            // CS-39: clamp b_theta bias component (see gpu_optimizer.rs for full rationale).
            // w_theta · [k,v] can still push per-token theta higher; theta_ceil is the full fix.
            unsafe {
                crate::cuda_ffi::clamp_f32_cuda(lp.b_theta.ptr(), lp.b_theta.len() as i32, -10.0, 2.0);
            }
            crate::dispatch::cuda_sync(); // surface any async error from clamp kernel
            adamw_one(&mut lp.w_eta, &lg.d_w_eta, &mut ml.m_w_eta, &mut ml.v_w_eta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.b_eta, &lg.d_b_eta, &mut ml.m_b_eta, &mut ml.v_b_eta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        }
    }

    grad_norm
}

/// Weight tying for stacked model: copy w_unembed^T → w_embed on GPU.
/// Mirrors gpu_backward::gpu_sync_embed_weights for single-block models.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_sync_embed_weights(params: &mut GpuStackedParams, d: usize, vocab: usize) {
    unsafe {
        crate::cuda_ffi::transpose_copy_cuda(
            params.w_unembed.as_ptr(),  // src: [d, vocab]
            params.w_embed.ptr(),        // dst: [vocab, d]
            d as i32,
            vocab as i32,
        );
    }
    crate::dispatch::cuda_sync();
}
