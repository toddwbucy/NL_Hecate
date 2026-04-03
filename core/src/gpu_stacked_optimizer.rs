/// GPU-resident AdamW optimizer for stacked multi-block models.
///
/// Maintains moment buffers for shared params + per-block SWA/LN/CMS weights.
/// Reuses `adamw_one` kernel from gpu_optimizer.rs.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::GpuStackedParams;
#[cfg(feature = "cuda")]
use crate::gpu_stacked_backward::GpuStackedGrads;
#[cfg(feature = "cuda")]
use crate::gpu_optimizer::adamw_one;
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;
#[cfg(feature = "cuda")]
use crate::gpu_profiler::GpuProfiler;
#[cfg(feature = "cuda")]
use crate::{prof_start, prof_stop};

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
    m_persistent_tokens: GpuBuf<f32>, v_persistent_tokens: GpuBuf<f32>,
    // Per-block moments
    blocks: Vec<MomentBlock>,
    pub step: u32,
    // Scratch buffers for gradient norm computation
    norm_scratch: GpuBuf<f32>,
    norm_host: Vec<f32>,
    // Spec 62: GPU-side norm + clip results (avoid per-block D2H)
    pub(crate) norm_result: GpuBuf<f32>,   // 1 element: computed L2 grad norm
    pub(crate) clip_scale: GpuBuf<f32>,    // 1 element: min(1.0, max_norm / norm)
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

        // Spec 54: Compute TOTAL partials across ALL gradient tensors so we can
        // launch all grad_norm_sq_cuda kernels at different offsets, then sync once.
        let partials_for = |len: usize| -> usize { (len + 255) / 256 };
        let mut total_partials = 0usize;
        total_partials += partials_for(params.w_embed.len());
        total_partials += partials_for(params.w_unembed.len());
        total_partials += partials_for(params.ln_final_gamma.len());
        total_partials += partials_for(params.ln_final_beta.len());
        total_partials += partials_for(params.persistent_tokens.len());
        for bp in &params.blocks {
            for buf in [&bp.w_q, &bp.w_k, &bp.w_v, &bp.w_o,
                        &bp.ln_attn_gamma, &bp.ln_attn_beta,
                        &bp.ln_mem_gamma, &bp.ln_mem_beta] {
                total_partials += partials_for(buf.len());
            }
            for lp in &bp.levels {
                for buf in [&lp.w_k_mem, &lp.w_v_mem, &lp.w_q_mem,
                            &lp.w_alpha, &lp.b_alpha, &lp.w_theta, &lp.b_theta,
                            &lp.w_eta, &lp.b_eta] {
                    total_partials += partials_for(buf.len());
                }
            }
        }
        // Need 2x because both grad_norm_ex and per_block_grad_norms run sequentially
        // reusing the same buffer. The max of the two is total_partials (same tensors).
        let max_partials = total_partials;

        GpuStackedAdamWState {
            m_embed: GpuBuf::zeros(params.w_embed.len()),
            v_embed: GpuBuf::zeros(params.w_embed.len()),
            m_unembed: GpuBuf::zeros(params.w_unembed.len()),
            v_unembed: GpuBuf::zeros(params.w_unembed.len()),
            m_ln_final_gamma: GpuBuf::zeros(params.ln_final_gamma.len()),
            v_ln_final_gamma: GpuBuf::zeros(params.ln_final_gamma.len()),
            m_ln_final_beta: GpuBuf::zeros(params.ln_final_beta.len()),
            v_ln_final_beta: GpuBuf::zeros(params.ln_final_beta.len()),
            m_persistent_tokens: GpuBuf::zeros(params.persistent_tokens.len()),
            v_persistent_tokens: GpuBuf::zeros(params.persistent_tokens.len()),
            blocks,
            step: 0,
            norm_scratch: GpuBuf::zeros(max_partials),
            norm_host: vec![0.0f32; max_partials],
            norm_result: GpuBuf::zeros(1),
            clip_scale: GpuBuf::zeros(1),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_adamw_update — full weight update
// ══════════════════════════════════════════════════════════════════════

/// Compute L2 norm of all stacked gradient buffers on GPU and apply clipping
/// entirely on-device (spec 62). Replaces the old gpu_stacked_grad_norm_ex which
/// did 9 cuda_sync + D2H per step. This version does ONE sync (for alpha_mem
/// host-side clip scale readback).
///
/// All grad_norm_sq_cuda kernels fire into a single contiguous scratch buffer
/// without intermediate syncs. A single-block reduction kernel sums all partials
/// and computes the clip scale on GPU. Conditional scaling reads the scale from
/// device memory — if no clipping needed, the kernel exits immediately.
#[cfg(feature = "cuda")]
fn gpu_stacked_grad_norm_clip_fused(
    grads: &mut GpuStackedGrads,
    state: &mut GpuStackedAdamWState,
    max_grad_norm: f32,
    skip_embed: bool,
) {
    let scratch_ptr = state.norm_scratch.ptr();
    let mut offset = 0usize;

    // ── Launch all grad_norm_sq_cuda into scratch without sync ────────

    // Shared grads
    let mut shared_ptrs: Vec<(*const f32, i32)> = if skip_embed {
        vec![(grads.d_ln_final_gamma.as_ptr(), grads.d_ln_final_gamma.len() as i32),
             (grads.d_ln_final_beta.as_ptr(), grads.d_ln_final_beta.len() as i32)]
    } else {
        vec![(grads.d_w_embed.as_ptr(), grads.d_w_embed.len() as i32),
             (grads.d_w_unembed.as_ptr(), grads.d_w_unembed.len() as i32),
             (grads.d_ln_final_gamma.as_ptr(), grads.d_ln_final_gamma.len() as i32),
             (grads.d_ln_final_beta.as_ptr(), grads.d_ln_final_beta.len() as i32)]
    };
    if grads.d_persistent_tokens.len() > 0 {
        shared_ptrs.push((grads.d_persistent_tokens.as_ptr(), grads.d_persistent_tokens.len() as i32));
    }
    for &(ptr, n) in &shared_ptrs {
        if n == 0 { continue; }
        let mut nb: i32 = 0;
        let err = unsafe {
            crate::cuda_ffi::grad_norm_sq_cuda(ptr, scratch_ptr.add(offset), n, &mut nb)
        };
        assert_eq!(err, 0, "grad_norm_sq_cuda failed");
        offset += nb as usize;
    }

    // Per-block grads (all blocks, all levels — NO per-block sync)
    for bg in grads.blocks.iter() {
        for ptr_n in [
            (bg.d_w_q.as_ptr(), bg.d_w_q.len()),
            (bg.d_w_k.as_ptr(), bg.d_w_k.len()),
            (bg.d_w_v.as_ptr(), bg.d_w_v.len()),
            (bg.d_w_o.as_ptr(), bg.d_w_o.len()),
            (bg.d_ln_attn_gamma.as_ptr(), bg.d_ln_attn_gamma.len()),
            (bg.d_ln_attn_beta.as_ptr(), bg.d_ln_attn_beta.len()),
            (bg.d_ln_mem_gamma.as_ptr(), bg.d_ln_mem_gamma.len()),
            (bg.d_ln_mem_beta.as_ptr(), bg.d_ln_mem_beta.len()),
        ] {
            let n = ptr_n.1 as i32;
            if n == 0 { continue; }
            let mut nb: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::grad_norm_sq_cuda(ptr_n.0, scratch_ptr.add(offset), n, &mut nb)
            };
            assert_eq!(err, 0, "grad_norm_sq_cuda failed");
            offset += nb as usize;
        }
        for lg in &bg.levels {
            for ptr_n in [
                (lg.d_w_k_mem.as_ptr(), lg.d_w_k_mem.len()),
                (lg.d_w_v_mem.as_ptr(), lg.d_w_v_mem.len()),
                (lg.d_w_q_mem.as_ptr(), lg.d_w_q_mem.len()),
                (lg.d_w_alpha.as_ptr(), lg.d_w_alpha.len()),
                (lg.d_b_alpha.as_ptr(), lg.d_b_alpha.len()),
                (lg.d_w_theta.as_ptr(), lg.d_w_theta.len()),
                (lg.d_b_theta.as_ptr(), lg.d_b_theta.len()),
                (lg.d_w_eta.as_ptr(), lg.d_w_eta.len()),
                (lg.d_b_eta.as_ptr(), lg.d_b_eta.len()),
            ] {
                let n = ptr_n.1 as i32;
                if n == 0 { continue; }
                let mut nb: i32 = 0;
                let err = unsafe {
                    crate::cuda_ffi::grad_norm_sq_cuda(ptr_n.0, scratch_ptr.add(offset), n, &mut nb)
                };
                assert_eq!(err, 0, "grad_norm_sq_cuda failed");
                offset += nb as usize;
            }
        }
    }

    // ── GPU-side reduction + clip scale computation (1 kernel, no sync) ──
    if offset > 0 {
        let err = unsafe {
            crate::cuda_ffi::reduce_partials_clip_cuda(
                scratch_ptr,
                offset as i32,
                max_grad_norm,
                state.norm_result.ptr(),
                state.clip_scale.ptr(),
            )
        };
        assert_eq!(err, 0, "reduce_partials_clip_cuda failed");
    }

    // ── Conditional scale: all GPU grad buffers (reads scale from device) ──
    let scale_ptr = state.clip_scale.as_ptr();
    let cond_scale = |g: &mut GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let err = unsafe {
            crate::cuda_ffi::grad_scale_conditional_cuda(g.ptr(), scale_ptr, n)
        };
        assert_eq!(err, 0, "grad_scale_conditional_cuda failed");
    };

    if !skip_embed {
        cond_scale(&mut grads.d_w_embed);
        cond_scale(&mut grads.d_w_unembed);
    }
    cond_scale(&mut grads.d_ln_final_gamma);
    cond_scale(&mut grads.d_ln_final_beta);
    cond_scale(&mut grads.d_persistent_tokens);
    for bg in &mut grads.blocks {
        cond_scale(&mut bg.d_w_q);
        cond_scale(&mut bg.d_w_k);
        cond_scale(&mut bg.d_w_v);
        cond_scale(&mut bg.d_w_o);
        cond_scale(&mut bg.d_ln_attn_gamma);
        cond_scale(&mut bg.d_ln_attn_beta);
        cond_scale(&mut bg.d_ln_mem_gamma);
        cond_scale(&mut bg.d_ln_mem_beta);
        for lg in &mut bg.levels {
            cond_scale(&mut lg.d_w_k_mem);
            cond_scale(&mut lg.d_w_v_mem);
            cond_scale(&mut lg.d_w_q_mem);
            cond_scale(&mut lg.d_w_alpha);
            cond_scale(&mut lg.d_b_alpha);
            cond_scale(&mut lg.d_w_theta);
            cond_scale(&mut lg.d_b_theta);
            cond_scale(&mut lg.d_w_eta);
            cond_scale(&mut lg.d_b_eta);
        }
    }

    // ── alpha_mem: host-side grads need actual scale value (1 D2H) ───
    // This is the ONE sync per step. alpha_mem grads are tiny host-side
    // scalars (k=4 per block). Read clip_scale to apply them.
    let mut scale_host = [1.0f32];
    crate::dispatch::cuda_sync();
    state.clip_scale.copy_to_host(&mut scale_host);
    if scale_host[0] < 1.0 {
        for bg in &mut grads.blocks {
            for g in &mut bg.d_alpha_mem {
                *g *= scale_host[0];
            }
        }
    }
}

/// Read the grad norm computed by the last gpu_stacked_grad_norm_clip_fused call.
/// Only call on log steps to avoid unnecessary D2H sync.
#[cfg(feature = "cuda")]
pub fn gpu_read_grad_norm(state: &GpuStackedAdamWState) -> f32 {
    let mut norm = [0.0f32];
    state.norm_result.copy_to_host(&mut norm);
    norm[0]
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
///
/// Spec 54: Per-block batched launch — all kernels within each block fire at
/// different offsets in norm_scratch, then sync once per block. This avoids
/// the command-queue saturation from launching all ~352 kernels at once.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_per_block_grad_norms(
    grads: &GpuStackedGrads,
    state: &mut GpuStackedAdamWState,
) -> PerBlockGradNorms {
    let n_blk = grads.blocks.len();
    let scratch_ptr = state.norm_scratch.ptr();

    let mut block_sq = vec![0.0f64; n_blk];
    let mut l0_sq = vec![0.0f64; n_blk];

    for (bi, bg) in grads.blocks.iter().enumerate() {
        let mut offset = 0usize;

        // Track level boundaries for L0 detection
        // (start_offset, end_offset, is_l0)
        let mut level_segs: Vec<(usize, usize, bool)> = Vec::new();
        let swa_start = offset;

        // SWA projections (aggregate only, not L0)
        for g in [&bg.d_w_q, &bg.d_w_k, &bg.d_w_v, &bg.d_w_o,
                  &bg.d_ln_attn_gamma, &bg.d_ln_attn_beta,
                  &bg.d_ln_mem_gamma, &bg.d_ln_mem_beta] {
            let n = g.len() as i32;
            if n == 0 { continue; }
            let mut nb: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::grad_norm_sq_cuda(
                    g.as_ptr(), scratch_ptr.add(offset), n, &mut nb,
                )
            };
            assert_eq!(err, 0, "grad_norm_sq_cuda failed");
            offset += nb as usize;
        }
        // SWA segment: aggregate only (is_l0 = false)
        level_segs.push((swa_start, offset, false));

        // Per-level params
        for (li, lg) in bg.levels.iter().enumerate() {
            let lev_start = offset;
            for g in [&lg.d_w_k_mem, &lg.d_w_v_mem, &lg.d_w_q_mem,
                      &lg.d_w_alpha, &lg.d_b_alpha,
                      &lg.d_w_theta, &lg.d_b_theta,
                      &lg.d_w_eta, &lg.d_b_eta] {
                let n = g.len() as i32;
                if n == 0 { continue; }
                let mut nb: i32 = 0;
                let err = unsafe {
                    crate::cuda_ffi::grad_norm_sq_cuda(
                        g.as_ptr(), scratch_ptr.add(offset), n, &mut nb,
                    )
                };
                assert_eq!(err, 0, "grad_norm_sq_cuda failed");
                offset += nb as usize;
            }
            level_segs.push((lev_start, offset, li == 0));
        }

        // One sync + D2H per block
        if offset > 0 {
            crate::dispatch::cuda_sync();
            state.norm_scratch.slice(0, offset).copy_to_host(&mut state.norm_host[..offset]);
        }

        // Accumulate from segments
        for &(start, end, is_l0) in &level_segs {
            let partial_sum: f64 = (start..end)
                .map(|i| state.norm_host[i] as f64)
                .sum();
            block_sq[bi] += partial_sum;
            if is_l0 {
                l0_sq[bi] += partial_sum;
            }
        }

        // alpha_mem gradients (host-side scalars)
        for &g in &bg.d_alpha_mem {
            block_sq[bi] += (g as f64) * (g as f64);
        }
    }

    let block_norms: Vec<f32> = block_sq.iter().map(|s| s.sqrt() as f32).collect();
    let l0_block_norms: Vec<f32> = l0_sq.iter().map(|s| s.sqrt() as f32).collect();

    PerBlockGradNorms { block_norms, l0_block_norms }
}

/// Scale all stacked gradient buffers by a constant factor.
/// Spec 76: now public for gradient averaging after accumulation.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_scale_grads_ex(grads: &mut GpuStackedGrads, scale: f32, skip_embed: bool) {
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
    scale_buf(&mut grads.d_persistent_tokens);

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
    // Spec 54: removed unnecessary cuda_sync() — next op is AdamW GPU kernels
    // on the same default stream, which guarantees ordering.
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
    profiler: &mut Option<GpuProfiler>,
) -> f32 {
    state.step += 1;
    let t = state.step as f32;
    let bc1_inv = 1.0 / (1.0 - beta1.powf(t));
    let bc2_inv = 1.0 / (1.0 - beta2.powf(t));

    // Gradient clipping: fused GPU-side norm + conditional scale (spec 62).
    // Norm + clip happen entirely on-device. Only 1 cuda_sync for alpha_mem
    // host-side scale readback, down from 9 syncs in the old path.
    // Grad norm value is deferred — call gpu_read_grad_norm() on log steps.
    prof_start!(profiler, "grad_clip", GradClip, None, None);
    if max_grad_norm > 0.0 {
        gpu_stacked_grad_norm_clip_fused(grads, state, max_grad_norm, freeze_embed);
    }
    prof_stop!(profiler);

    // ── Shared params (always active, unless frozen) ──────────────────
    prof_start!(profiler, "opt_shared", Optimizer, None, None);
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
    if params.persistent_tokens.len() > 1 {
        adamw_one(&mut params.persistent_tokens, &grads.d_persistent_tokens,
                  &mut state.m_persistent_tokens, &mut state.v_persistent_tokens,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    }
    prof_stop!(profiler);

    // ── Per-block params ───────────────────────────────────────────────
    for (b, (bp, bg)) in params.blocks.iter_mut().zip(grads.blocks.iter()).enumerate() {
        let mb = &mut state.blocks[b];

        // SWA projections (always active)
        prof_start!(profiler, "opt_swa", Optimizer, Some(b), None);
        adamw_one(&mut bp.w_q, &bg.d_w_q, &mut mb.m_q, &mut mb.v_q,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_k, &bg.d_w_k, &mut mb.m_k, &mut mb.v_k,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_v, &bg.d_w_v, &mut mb.m_v, &mut mb.v_v,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut bp.w_o, &bg.d_w_o, &mut mb.m_o, &mut mb.v_o,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        prof_stop!(profiler);

        // LayerNorms
        prof_start!(profiler, "opt_ln", Optimizer, Some(b), None);
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
        prof_stop!(profiler);

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
            prof_start!(profiler, "opt_memory", Optimizer, Some(b), Some(i));
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
            // Spec 54: removed cuda_sync() — was error-check only; next ops are on same stream.
            adamw_one(&mut lp.w_eta, &lg.d_w_eta, &mut ml.m_w_eta, &mut ml.v_w_eta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.b_eta, &lg.d_b_eta, &mut ml.m_b_eta, &mut ml.v_b_eta,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            prof_stop!(profiler);
        }
    }

    // Spec 62: grad_norm deferred — caller reads via gpu_read_grad_norm() on log steps.
    0.0
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
    // Spec 54: removed cuda_sync() — next op is forward pass on same stream.
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedM3State — M3 optimizer for stacked multi-block models
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
use crate::gpu_optimizer::{GpuM3Config, m3_ema_one, m3_apply_1d_one, m3_ns_apply, gpu_frob_norm};

/// M3 moment buffers for one block's SWA + LN weights.
/// Three buffers per param: m1 (fast momentum), m2 (slow momentum), v (second moment).
#[cfg(feature = "cuda")]
struct M3MomentBlockStacked {
    // 2D SWA projections
    m1_q: GpuBuf<f32>, m2_q: GpuBuf<f32>, v_q: GpuBuf<f32>,
    m1_k: GpuBuf<f32>, m2_k: GpuBuf<f32>, v_k: GpuBuf<f32>,
    m1_v: GpuBuf<f32>, m2_v: GpuBuf<f32>, v_v: GpuBuf<f32>,
    m1_o: GpuBuf<f32>, m2_o: GpuBuf<f32>, v_o: GpuBuf<f32>,
    // 1D LayerNorms
    m1_ln_attn_gamma: GpuBuf<f32>, m2_ln_attn_gamma: GpuBuf<f32>, v_ln_attn_gamma: GpuBuf<f32>,
    m1_ln_attn_beta: GpuBuf<f32>,  m2_ln_attn_beta: GpuBuf<f32>,  v_ln_attn_beta: GpuBuf<f32>,
    m1_ln_mem_gamma: GpuBuf<f32>,  m2_ln_mem_gamma: GpuBuf<f32>,  v_ln_mem_gamma: GpuBuf<f32>,
    m1_ln_mem_beta: GpuBuf<f32>,   m2_ln_mem_beta: GpuBuf<f32>,   v_ln_mem_beta: GpuBuf<f32>,
    // alpha_mem: host-side (1D, small — k scalars)
    m1_alpha_mem: Vec<f32>, m2_alpha_mem: Vec<f32>, v_alpha_mem: Vec<f32>,
    // Per-level CMS
    levels: Vec<M3MomentLevelStacked>,
}

/// M3 moment buffers for one CMS level within a stacked block.
#[cfg(feature = "cuda")]
struct M3MomentLevelStacked {
    // 2D memory projections
    m1_w_k_mem: GpuBuf<f32>, m2_w_k_mem: GpuBuf<f32>, v_w_k_mem: GpuBuf<f32>,
    m1_w_v_mem: GpuBuf<f32>, m2_w_v_mem: GpuBuf<f32>, v_w_v_mem: GpuBuf<f32>,
    m1_w_q_mem: GpuBuf<f32>, m2_w_q_mem: GpuBuf<f32>, v_w_q_mem: GpuBuf<f32>,
    // 1D gate params
    m1_w_alpha: GpuBuf<f32>, m2_w_alpha: GpuBuf<f32>, v_w_alpha: GpuBuf<f32>,
    m1_b_alpha: GpuBuf<f32>, m2_b_alpha: GpuBuf<f32>, v_b_alpha: GpuBuf<f32>,
    m1_w_theta: GpuBuf<f32>, m2_w_theta: GpuBuf<f32>, v_w_theta: GpuBuf<f32>,
    m1_b_theta: GpuBuf<f32>, m2_b_theta: GpuBuf<f32>, v_b_theta: GpuBuf<f32>,
    m1_w_eta: GpuBuf<f32>,   m2_w_eta: GpuBuf<f32>,   v_w_eta: GpuBuf<f32>,
    m1_b_eta: GpuBuf<f32>,   m2_b_eta: GpuBuf<f32>,   v_b_eta: GpuBuf<f32>,
    level_step: u32,
}

/// M3 optimizer state for stacked model: shared embed/unembed/ln_final + N blocks.
#[cfg(feature = "cuda")]
pub struct GpuStackedM3State {
    // Shared param moments
    m1_embed: GpuBuf<f32>,   m2_embed: GpuBuf<f32>,   v_embed: GpuBuf<f32>,
    m1_unembed: GpuBuf<f32>, m2_unembed: GpuBuf<f32>, v_unembed: GpuBuf<f32>,
    m1_ln_final_gamma: GpuBuf<f32>, m2_ln_final_gamma: GpuBuf<f32>, v_ln_final_gamma: GpuBuf<f32>,
    m1_ln_final_beta: GpuBuf<f32>,  m2_ln_final_beta: GpuBuf<f32>,  v_ln_final_beta: GpuBuf<f32>,
    m1_persistent_tokens: GpuBuf<f32>, m2_persistent_tokens: GpuBuf<f32>, v_persistent_tokens_m3: GpuBuf<f32>,
    // Per-block moments
    blocks: Vec<M3MomentBlockStacked>,
    pub step: u32,
    pub config: GpuM3Config,
    // NS scratch buffers (allocated once, reused every step)
    ns_ata: GpuBuf<f32>,
    ns_work: GpuBuf<f32>,
    ns_ax: GpuBuf<f32>,
    ns_aax: GpuBuf<f32>,
    norm_scratch: GpuBuf<f32>,
    norm_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuStackedM3State {
    /// Create zero-initialized M3 state matching stacked param shapes.
    pub fn from_params(params: &GpuStackedParams, config: GpuM3Config, d: usize) -> Self {
        let z3 = |len: usize| -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
            (GpuBuf::zeros(len), GpuBuf::zeros(len), GpuBuf::zeros(len))
        };

        let (m1_embed, m2_embed, v_embed) = z3(params.w_embed.len());
        let (m1_unembed, m2_unembed, v_unembed) = z3(params.w_unembed.len());
        let (m1_ln_final_gamma, m2_ln_final_gamma, v_ln_final_gamma) = z3(params.ln_final_gamma.len());
        let (m1_ln_final_beta, m2_ln_final_beta, v_ln_final_beta) = z3(params.ln_final_beta.len());
        let (m1_persistent_tokens, m2_persistent_tokens, v_persistent_tokens_m3) = z3(params.persistent_tokens.len());

        // Pre-compute max 2D buffer size for NS scratch allocation
        let mut max_2d = params.w_embed.len().max(params.w_unembed.len()).max(params.persistent_tokens.len());
        for bp in &params.blocks {
            for buf in [&bp.w_q, &bp.w_k, &bp.w_v, &bp.w_o] {
                max_2d = max_2d.max(buf.len());
            }
            for lp in &bp.levels {
                for buf in [&lp.w_k_mem, &lp.w_v_mem, &lp.w_q_mem] {
                    max_2d = max_2d.max(buf.len());
                }
            }
        }

        let blocks = params.blocks.iter().map(|bp| {
            let levels = bp.levels.iter().map(|lp| {
                let (m1_w_k_mem, m2_w_k_mem, v_w_k_mem) = z3(lp.w_k_mem.len());
                let (m1_w_v_mem, m2_w_v_mem, v_w_v_mem) = z3(lp.w_v_mem.len());
                let (m1_w_q_mem, m2_w_q_mem, v_w_q_mem) = z3(lp.w_q_mem.len());
                let (m1_w_alpha, m2_w_alpha, v_w_alpha) = z3(lp.w_alpha.len());
                let (m1_b_alpha, m2_b_alpha, v_b_alpha) = z3(lp.b_alpha.len());
                let (m1_w_theta, m2_w_theta, v_w_theta) = z3(lp.w_theta.len());
                let (m1_b_theta, m2_b_theta, v_b_theta) = z3(lp.b_theta.len());
                let (m1_w_eta, m2_w_eta, v_w_eta) = z3(lp.w_eta.len());
                let (m1_b_eta, m2_b_eta, v_b_eta) = z3(lp.b_eta.len());
                M3MomentLevelStacked {
                    m1_w_k_mem, m2_w_k_mem, v_w_k_mem,
                    m1_w_v_mem, m2_w_v_mem, v_w_v_mem,
                    m1_w_q_mem, m2_w_q_mem, v_w_q_mem,
                    m1_w_alpha, m2_w_alpha, v_w_alpha,
                    m1_b_alpha, m2_b_alpha, v_b_alpha,
                    m1_w_theta, m2_w_theta, v_w_theta,
                    m1_b_theta, m2_b_theta, v_b_theta,
                    m1_w_eta, m2_w_eta, v_w_eta,
                    m1_b_eta, m2_b_eta, v_b_eta,
                    level_step: 0,
                }
            }).collect();

            let k = bp.alpha_mem.len();
            let (m1_q, m2_q, v_q) = z3(bp.w_q.len());
            let (m1_k, m2_k, v_k) = z3(bp.w_k.len());
            let (m1_v, m2_v, v_v) = z3(bp.w_v.len());
            let (m1_o, m2_o, v_o) = z3(bp.w_o.len());
            let (m1_ln_attn_gamma, m2_ln_attn_gamma, v_ln_attn_gamma) = z3(bp.ln_attn_gamma.len());
            let (m1_ln_attn_beta, m2_ln_attn_beta, v_ln_attn_beta) = z3(bp.ln_attn_beta.len());
            let (m1_ln_mem_gamma, m2_ln_mem_gamma, v_ln_mem_gamma) = z3(bp.ln_mem_gamma.len());
            let (m1_ln_mem_beta, m2_ln_mem_beta, v_ln_mem_beta) = z3(bp.ln_mem_beta.len());

            M3MomentBlockStacked {
                m1_q, m2_q, v_q,
                m1_k, m2_k, v_k,
                m1_v, m2_v, v_v,
                m1_o, m2_o, v_o,
                m1_ln_attn_gamma, m2_ln_attn_gamma, v_ln_attn_gamma,
                m1_ln_attn_beta, m2_ln_attn_beta, v_ln_attn_beta,
                m1_ln_mem_gamma, m2_ln_mem_gamma, v_ln_mem_gamma,
                m1_ln_mem_beta, m2_ln_mem_beta, v_ln_mem_beta,
                m1_alpha_mem: vec![0.0f32; k],
                m2_alpha_mem: vec![0.0f32; k],
                v_alpha_mem: vec![0.0f32; k],
                levels,
            }
        }).collect();

        let max_partials = max_2d / 256 + 1;

        GpuStackedM3State {
            m1_embed, m2_embed, v_embed,
            m1_unembed, m2_unembed, v_unembed,
            m1_ln_final_gamma, m2_ln_final_gamma, v_ln_final_gamma,
            m1_ln_final_beta, m2_ln_final_beta, v_ln_final_beta,
            m1_persistent_tokens, m2_persistent_tokens, v_persistent_tokens_m3,
            blocks,
            step: 0,
            config,
            ns_ata: GpuBuf::zeros(d * d),
            ns_work: GpuBuf::zeros(max_2d),
            ns_ax: GpuBuf::zeros(max_2d),
            ns_aax: GpuBuf::zeros(max_2d),
            norm_scratch: GpuBuf::zeros(max_partials),
            norm_host: vec![0.0f32; max_partials],
        }
    }
}

/// Compute gradient norm for stacked grads using M3's scratch buffers.
/// When `skip_embed` is true, excludes embed/unembed from the norm (frozen case).
#[cfg(feature = "cuda")]
fn gpu_stacked_m3_grad_norm(
    grads: &GpuStackedGrads,
    scratch: &mut GpuBuf<f32>,
    host: &mut Vec<f32>,
    skip_embed: bool,
) -> f32 {
    let mut total = 0.0f64;
    let mut add = |g: &GpuBuf<f32>| {
        let n = gpu_frob_norm(g, scratch, host) as f64;
        total += n * n;
    };
    if !skip_embed {
        add(&grads.d_w_embed);
        add(&grads.d_w_unembed);
    }
    add(&grads.d_ln_final_gamma);
    add(&grads.d_ln_final_beta);
    if grads.d_persistent_tokens.len() > 0 {
        add(&grads.d_persistent_tokens);
    }
    for bg in &grads.blocks {
        add(&bg.d_w_q);
        add(&bg.d_w_k);
        add(&bg.d_w_v);
        add(&bg.d_w_o);
        add(&bg.d_ln_attn_gamma);
        add(&bg.d_ln_attn_beta);
        add(&bg.d_ln_mem_gamma);
        add(&bg.d_ln_mem_beta);
        for lg in &bg.levels {
            add(&lg.d_w_k_mem);
            add(&lg.d_w_v_mem);
            add(&lg.d_w_q_mem);
            add(&lg.d_w_alpha);
            add(&lg.d_b_alpha);
            add(&lg.d_w_theta);
            add(&lg.d_b_theta);
            add(&lg.d_w_eta);
            add(&lg.d_b_eta);
        }
    }
    drop(add);
    // alpha_mem gradients (host-side scalars, accumulated after closure dropped)
    for bg in &grads.blocks {
        for &g in &bg.d_alpha_mem {
            let g64 = g as f64;
            total += g64 * g64;
        }
    }
    total.sqrt() as f32
}

/// Scale all stacked grads in-place for gradient clipping.
/// When `skip_embed` is true, excludes embed/unembed from scaling (frozen case).
#[cfg(feature = "cuda")]
fn gpu_stacked_m3_scale_grads(grads: &mut GpuStackedGrads, scale: f32, skip_embed: bool) {
    let scale_buf = |g: &mut GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let err = unsafe {
            crate::cuda_ffi::grad_scale_cuda(g.ptr(), scale, n)
        };
        assert_eq!(err, 0, "grad_scale_cuda failed with cudaError_t={}", err);
    };
    if !skip_embed {
        scale_buf(&mut grads.d_w_embed);
        scale_buf(&mut grads.d_w_unembed);
    }
    scale_buf(&mut grads.d_ln_final_gamma);
    scale_buf(&mut grads.d_ln_final_beta);
    scale_buf(&mut grads.d_persistent_tokens);
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
        // alpha_mem (host-side scalars)
        for g in &mut bg.d_alpha_mem {
            *g *= scale;
        }
    }
}

/// M3 optimizer update for stacked multi-block model.
/// Mirrors `gpu_m3_update` from `gpu_optimizer.rs` but iterates over blocks.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_m3_update(
    params: &mut GpuStackedParams,
    grads: &mut GpuStackedGrads,
    state: &mut GpuStackedM3State,
    pulse: &Pulse,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    freeze_embed: bool,
    profiler: &mut Option<GpuProfiler>,
) -> f32 {
    state.step += 1;
    let cfg = &state.config;
    let update_m2 = state.step % cfg.chunk_size == 0;
    let bc2 = 1.0 - cfg.beta2.powi(state.step as i32);

    // ── Gradient clipping ─────────────────────────────────────────────
    prof_start!(profiler, "grad_clip", GradClip, None, None);
    let grad_norm = if max_grad_norm > 0.0 {
        let norm = gpu_stacked_m3_grad_norm(grads, &mut state.norm_scratch, &mut state.norm_host, freeze_embed);
        if norm > max_grad_norm {
            let scale = max_grad_norm / norm;
            gpu_stacked_m3_scale_grads(grads, scale, freeze_embed);
        }
        norm
    } else {
        0.0
    };
    prof_stop!(profiler);

    let ns_iters = cfg.ns_iterations;
    let m2_scale = -lr * cfg.alpha;
    let vocab = params.w_embed.len() / d;

    // Helper macro for 2D NS apply (M1 then M2)
    macro_rules! ns_2d {
        ($w:expr, $m1:expr, $m2:expr, $rows:expr, $cols:expr) => {
            m3_ns_apply(&mut $w, &$m1, $rows, $cols, -lr, ns_iters,
                        &mut state.ns_ata, &mut state.ns_work,
                        &mut state.ns_ax, &mut state.ns_aax,
                        &mut state.norm_scratch, &mut state.norm_host);
            m3_ns_apply(&mut $w, &$m2, $rows, $cols, m2_scale, ns_iters,
                        &mut state.ns_ata, &mut state.ns_work,
                        &mut state.ns_ax, &mut state.ns_aax,
                        &mut state.norm_scratch, &mut state.norm_host);
        };
    }

    // ── Shared params (always active, unless frozen) ──────────────────
    prof_start!(profiler, "opt_shared", Optimizer, None, None);
    if !freeze_embed {
        m3_ema_one(&mut state.m1_embed, &mut state.m2_embed, &mut state.v_embed, &grads.d_w_embed,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut state.m1_unembed, &mut state.m2_unembed, &mut state.v_unembed, &grads.d_w_unembed,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        ns_2d!(params.w_embed, state.m1_embed, state.m2_embed, vocab, d);
        ns_2d!(params.w_unembed, state.m1_unembed, state.m2_unembed, d, vocab);
    }

    // ln_final: 1D
    m3_ema_one(&mut state.m1_ln_final_gamma, &mut state.m2_ln_final_gamma, &mut state.v_ln_final_gamma, &grads.d_ln_final_gamma,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut state.m1_ln_final_beta, &mut state.m2_ln_final_beta, &mut state.v_ln_final_beta, &grads.d_ln_final_beta,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_apply_1d_one(&mut params.ln_final_gamma, &state.m1_ln_final_gamma, &state.m2_ln_final_gamma, &state.v_ln_final_gamma,
                    lr, cfg.alpha, cfg.eps, bc2);
    m3_apply_1d_one(&mut params.ln_final_beta, &state.m1_ln_final_beta, &state.m2_ln_final_beta, &state.v_ln_final_beta,
                    lr, cfg.alpha, cfg.eps, bc2);
    if params.persistent_tokens.len() > 1 {
        let n_p = params.persistent_tokens.len() / d;
        m3_ema_one(&mut state.m1_persistent_tokens, &mut state.m2_persistent_tokens, &mut state.v_persistent_tokens_m3, &grads.d_persistent_tokens,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        ns_2d!(params.persistent_tokens, state.m1_persistent_tokens, state.m2_persistent_tokens, n_p, d);
    }
    prof_stop!(profiler);

    // ── Per-block params ──────────────────────────────────────────────
    for (b, bg) in grads.blocks.iter().enumerate() {
        let bp = &mut params.blocks[b];
        let mb = &mut state.blocks[b];

        // 2D SWA projections: EMA + NS apply
        prof_start!(profiler, "opt_swa", Optimizer, Some(b), None);
        m3_ema_one(&mut mb.m1_q, &mut mb.m2_q, &mut mb.v_q, &bg.d_w_q,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_k, &mut mb.m2_k, &mut mb.v_k, &bg.d_w_k,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_v, &mut mb.m2_v, &mut mb.v_v, &bg.d_w_v,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_o, &mut mb.m2_o, &mut mb.v_o, &bg.d_w_o,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        prof_stop!(profiler);

        prof_start!(profiler, "opt_ns", OptimizerNS, Some(b), None);
        ns_2d!(bp.w_q, mb.m1_q, mb.m2_q, d, d);
        ns_2d!(bp.w_k, mb.m1_k, mb.m2_k, d, d);
        ns_2d!(bp.w_v, mb.m1_v, mb.m2_v, d, d);
        ns_2d!(bp.w_o, mb.m1_o, mb.m2_o, d, d);
        prof_stop!(profiler);

        // 1D LayerNorms: EMA + 1D apply
        prof_start!(profiler, "opt_ln", Optimizer, Some(b), None);
        m3_ema_one(&mut mb.m1_ln_attn_gamma, &mut mb.m2_ln_attn_gamma, &mut mb.v_ln_attn_gamma, &bg.d_ln_attn_gamma,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_ln_attn_beta, &mut mb.m2_ln_attn_beta, &mut mb.v_ln_attn_beta, &bg.d_ln_attn_beta,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_ln_mem_gamma, &mut mb.m2_ln_mem_gamma, &mut mb.v_ln_mem_gamma, &bg.d_ln_mem_gamma,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
        m3_ema_one(&mut mb.m1_ln_mem_beta, &mut mb.m2_ln_mem_beta, &mut mb.v_ln_mem_beta, &bg.d_ln_mem_beta,
                   cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

        m3_apply_1d_one(&mut bp.ln_attn_gamma, &mb.m1_ln_attn_gamma, &mb.m2_ln_attn_gamma, &mb.v_ln_attn_gamma,
                        lr, cfg.alpha, cfg.eps, bc2);
        m3_apply_1d_one(&mut bp.ln_attn_beta, &mb.m1_ln_attn_beta, &mb.m2_ln_attn_beta, &mb.v_ln_attn_beta,
                        lr, cfg.alpha, cfg.eps, bc2);
        m3_apply_1d_one(&mut bp.ln_mem_gamma, &mb.m1_ln_mem_gamma, &mb.m2_ln_mem_gamma, &mb.v_ln_mem_gamma,
                        lr, cfg.alpha, cfg.eps, bc2);
        m3_apply_1d_one(&mut bp.ln_mem_beta, &mb.m1_ln_mem_beta, &mb.m2_ln_mem_beta, &mb.v_ln_mem_beta,
                        lr, cfg.alpha, cfg.eps, bc2);
        prof_stop!(profiler);

        // alpha_mem: host-side 1D M3 (same pattern as AdamW host-side, but with 3 moments)
        {
            let k = bg.d_alpha_mem.len();
            let mut alpha_host = vec![0.0f32; k];
            bp.alpha_mem.slice(0, k).copy_to_host(&mut alpha_host);
            for i in 0..k {
                let g = bg.d_alpha_mem[i];
                mb.m1_alpha_mem[i] = cfg.beta1 * mb.m1_alpha_mem[i] + (1.0 - cfg.beta1) * g;
                if update_m2 {
                    mb.m2_alpha_mem[i] = cfg.beta3 * mb.m2_alpha_mem[i] + (1.0 - cfg.beta3) * g;
                }
                mb.v_alpha_mem[i] = cfg.beta2 * mb.v_alpha_mem[i] + (1.0 - cfg.beta2) * g * g;
                let v_hat = mb.v_alpha_mem[i] / bc2;
                let denom = v_hat.sqrt() + cfg.eps;
                // Combined M1 + alpha*M2 update (Adam-style for 1D scalars)
                let update = mb.m1_alpha_mem[i] + cfg.alpha * mb.m2_alpha_mem[i];
                alpha_host[i] -= lr * update / denom;
            }
            bp.alpha_mem.copy_from_host(&alpha_host);
        }

        // Per-level CMS weights (pulse-gated)
        for (i, lg) in bg.levels.iter().enumerate() {
            if i >= pulse.active_levels.len() || !pulse.active_levels[i] {
                continue;
            }
            let lp = &mut bp.levels[i];
            let ml = &mut mb.levels[i];
            ml.level_step += 1;
            let level_bc2 = 1.0 - cfg.beta2.powi(ml.level_step as i32);

            // 2D level params: EMA
            prof_start!(profiler, "opt_memory", Optimizer, Some(b), Some(i));
            m3_ema_one(&mut ml.m1_w_k_mem, &mut ml.m2_w_k_mem, &mut ml.v_w_k_mem, &lg.d_w_k_mem,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_w_v_mem, &mut ml.m2_w_v_mem, &mut ml.v_w_v_mem, &lg.d_w_v_mem,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_w_q_mem, &mut ml.m2_w_q_mem, &mut ml.v_w_q_mem, &lg.d_w_q_mem,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

            // 1D level params: EMA
            m3_ema_one(&mut ml.m1_w_alpha, &mut ml.m2_w_alpha, &mut ml.v_w_alpha, &lg.d_w_alpha,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_b_alpha, &mut ml.m2_b_alpha, &mut ml.v_b_alpha, &lg.d_b_alpha,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_w_theta, &mut ml.m2_w_theta, &mut ml.v_w_theta, &lg.d_w_theta,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_b_theta, &mut ml.m2_b_theta, &mut ml.v_b_theta, &lg.d_b_theta,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_w_eta, &mut ml.m2_w_eta, &mut ml.v_w_eta, &lg.d_w_eta,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_b_eta, &mut ml.m2_b_eta, &mut ml.v_b_eta, &lg.d_b_eta,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

            // Apply 1D level params
            m3_apply_1d_one(&mut lp.w_alpha, &ml.m1_w_alpha, &ml.m2_w_alpha, &ml.v_w_alpha,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            m3_apply_1d_one(&mut lp.b_alpha, &ml.m1_b_alpha, &ml.m2_b_alpha, &ml.v_b_alpha,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            m3_apply_1d_one(&mut lp.w_theta, &ml.m1_w_theta, &ml.m2_w_theta, &ml.v_w_theta,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            m3_apply_1d_one(&mut lp.b_theta, &ml.m1_b_theta, &ml.m2_b_theta, &ml.v_b_theta,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            // CS-39: clamp b_theta
            unsafe {
                crate::cuda_ffi::clamp_f32_cuda(lp.b_theta.ptr(), lp.b_theta.len() as i32, -10.0, 2.0);
            }
            // Spec 54: removed cuda_sync() — error-check only, next ops on same stream.
            m3_apply_1d_one(&mut lp.w_eta, &ml.m1_w_eta, &ml.m2_w_eta, &ml.v_w_eta,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            m3_apply_1d_one(&mut lp.b_eta, &ml.m1_b_eta, &ml.m2_b_eta, &ml.v_b_eta,
                            lr, cfg.alpha, cfg.eps, level_bc2);
            prof_stop!(profiler);

            // Apply 2D level params via NS
            prof_start!(profiler, "opt_ns", OptimizerNS, Some(b), Some(i));
            ns_2d!(lp.w_k_mem, ml.m1_w_k_mem, ml.m2_w_k_mem, d, d);
            ns_2d!(lp.w_v_mem, ml.m1_w_v_mem, ml.m2_w_v_mem, d, d);
            ns_2d!(lp.w_q_mem, ml.m1_w_q_mem, ml.m2_w_q_mem, d, d);
            prof_stop!(profiler);
        }
    }
    // Spec 54: removed cuda_sync() — next op is forward pass on same stream.
    grad_norm
}
