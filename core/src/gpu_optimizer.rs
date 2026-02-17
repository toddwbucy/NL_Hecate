/// GPU-resident AdamW optimizer state and update functions.
///
/// Maintains first-moment (m) and second-moment (v) buffers on GPU,
/// mirroring every parameter buffer in GpuMAGParams. The fused AdamW
/// kernel updates weights, m, and v in a single pass — zero PCIe traffic.
///
/// This is the outer-loop optimizer only (CS-27: optimizer frequency must
/// match architecture). Inner-loop memory updates (Titans GD+momentum,
/// Delta Rule, etc.) are handled by the memory rules themselves.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuMAGParams, GpuMemoryLevelParams};
#[cfg(feature = "cuda")]
use crate::gpu_backward::GpuMAGGrads;

// ══════════════════════════════════════════════════════════════════════
// Moment buffer structs (mirror GpuMAGParams layout)
// ══════════════════════════════════════════════════════════════════════

/// First/second moment buffers for SWA weights.
#[cfg(feature = "cuda")]
struct MomentSWA {
    m_embed: GpuBuf<f32>,   v_embed: GpuBuf<f32>,
    m_q: GpuBuf<f32>,       v_q: GpuBuf<f32>,
    m_k: GpuBuf<f32>,       v_k: GpuBuf<f32>,
    m_v: GpuBuf<f32>,       v_v: GpuBuf<f32>,
    m_o: GpuBuf<f32>,       v_o: GpuBuf<f32>,
    m_unembed: GpuBuf<f32>, v_unembed: GpuBuf<f32>,
}

/// First/second moment buffers for one memory level.
#[cfg(feature = "cuda")]
struct MomentLevel {
    m_w_k_mem: GpuBuf<f32>,  v_w_k_mem: GpuBuf<f32>,
    m_w_v_mem: GpuBuf<f32>,  v_w_v_mem: GpuBuf<f32>,
    m_w_q_mem: GpuBuf<f32>,  v_w_q_mem: GpuBuf<f32>,
    m_w_alpha: GpuBuf<f32>,  v_w_alpha: GpuBuf<f32>,
    m_b_alpha: GpuBuf<f32>,  v_b_alpha: GpuBuf<f32>,
    m_w_theta: GpuBuf<f32>,  v_w_theta: GpuBuf<f32>,
    m_b_theta: GpuBuf<f32>,  v_b_theta: GpuBuf<f32>,
    m_w_eta: GpuBuf<f32>,    v_w_eta: GpuBuf<f32>,
    m_b_eta: GpuBuf<f32>,    v_b_eta: GpuBuf<f32>,
}

// ══════════════════════════════════════════════════════════════════════
// GpuAdamWState — complete optimizer state on GPU
// ══════════════════════════════════════════════════════════════════════

/// AdamW optimizer state resident on GPU. Zero-initialized moment buffers
/// mirror every learnable parameter in GpuMAGParams.
#[cfg(feature = "cuda")]
pub struct GpuAdamWState {
    swa: MomentSWA,
    levels: Vec<MomentLevel>,
    /// Optimizer step counter (for bias correction).
    pub step: u32,
    /// Scratch buffer for gradient norm reduction (reused across steps).
    norm_scratch: GpuBuf<f32>,
    /// Host-side buffer for reading back partial sums.
    norm_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuAdamWState {
    /// Create zero-initialized optimizer state matching param shapes.
    pub fn from_params(params: &GpuMAGParams) -> Self {
        let swa = MomentSWA {
            m_embed: GpuBuf::zeros(params.swa.w_embed.len()),
            v_embed: GpuBuf::zeros(params.swa.w_embed.len()),
            m_q: GpuBuf::zeros(params.swa.w_q.len()),
            v_q: GpuBuf::zeros(params.swa.w_q.len()),
            m_k: GpuBuf::zeros(params.swa.w_k.len()),
            v_k: GpuBuf::zeros(params.swa.w_k.len()),
            m_v: GpuBuf::zeros(params.swa.w_v.len()),
            v_v: GpuBuf::zeros(params.swa.w_v.len()),
            m_o: GpuBuf::zeros(params.swa.w_o.len()),
            v_o: GpuBuf::zeros(params.swa.w_o.len()),
            m_unembed: GpuBuf::zeros(params.swa.w_unembed.len()),
            v_unembed: GpuBuf::zeros(params.swa.w_unembed.len()),
        };

        let levels = params.levels.iter().map(|lp| MomentLevel {
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
        }).collect();

        // Max buffer for norm reduction: find largest param buffer across all weights.
        // ceil(max_param_len / 256) partials needed.
        let mut max_len = params.swa.w_embed.len();
        for buf in [&params.swa.w_q, &params.swa.w_k, &params.swa.w_v,
                     &params.swa.w_o, &params.swa.w_unembed] {
            max_len = max_len.max(buf.len());
        }
        for lp in &params.levels {
            for buf in [&lp.w_k_mem, &lp.w_v_mem, &lp.w_q_mem,
                         &lp.w_alpha, &lp.b_alpha, &lp.w_theta, &lp.b_theta,
                         &lp.w_eta, &lp.b_eta] {
                max_len = max_len.max(buf.len());
            }
        }
        let max_partials = max_len / 256 + 1;

        GpuAdamWState {
            swa,
            levels,
            step: 0,
            norm_scratch: GpuBuf::zeros(max_partials),
            norm_host: vec![0.0f32; max_partials],
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// AdamW update — calls fused kernel for each param/grad/m/v pair
// ══════════════════════════════════════════════════════════════════════

/// Helper: call the fused AdamW kernel for one (w, g, m, v) pair.
#[cfg(feature = "cuda")]
#[inline]
fn adamw_one(
    w: &mut GpuBuf<f32>, g: &GpuBuf<f32>,
    m: &mut GpuBuf<f32>, v: &mut GpuBuf<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1_inv: f32, bc2_inv: f32, weight_decay: f32,
) {
    let n = w.len() as i32;
    debug_assert_eq!(w.len(), g.len());
    debug_assert_eq!(w.len(), m.len());
    debug_assert_eq!(w.len(), v.len());
    unsafe {
        crate::cuda_ffi::adamw_update_cuda(
            w.ptr(), g.as_ptr(), m.ptr(), v.ptr(),
            n, lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay,
        );
    }
}

/// Full AdamW weight update on GPU. Updates all params in-place, advances step counter.
/// Zero PCIe traffic — everything stays on device.
///
/// Returns the pre-clip gradient L2 norm (for logging). Returns 0.0 if clipping disabled.
#[cfg(feature = "cuda")]
pub fn gpu_adamw_update(
    params: &mut GpuMAGParams,
    grads: &mut GpuMAGGrads,
    state: &mut GpuAdamWState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    max_grad_norm: f32,
) -> f32 {
    state.step += 1;
    let t = state.step as f32;
    let bc1_inv = 1.0 / (1.0 - beta1.powf(t));
    let bc2_inv = 1.0 / (1.0 - beta2.powf(t));

    // ── Gradient clipping (if enabled) ───────────────────────────────
    let grad_norm = if max_grad_norm > 0.0 {
        let norm = gpu_grad_norm(grads, state);
        if norm > max_grad_norm {
            let scale = max_grad_norm / norm;
            gpu_scale_grads(grads, scale);
        }
        norm
    } else {
        0.0
    };

    // ── SWA weights ──────────────────────────────────────────────────
    let s = &mut state.swa;
    adamw_one(&mut params.swa.w_embed, &grads.d_w_embed, &mut s.m_embed, &mut s.v_embed,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.w_q, &grads.d_w_q, &mut s.m_q, &mut s.v_q,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.w_k, &grads.d_w_k, &mut s.m_k, &mut s.v_k,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.w_v, &grads.d_w_v, &mut s.m_v, &mut s.v_v,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.w_o, &grads.d_w_o, &mut s.m_o, &mut s.v_o,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.w_unembed, &grads.d_w_unembed, &mut s.m_unembed, &mut s.v_unembed,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

    // ── Per-level memory weights ─────────────────────────────────────
    for (i, lg) in grads.levels.iter().enumerate() {
        let lp = &mut params.levels[i];
        let ml = &mut state.levels[i];

        adamw_one(&mut lp.w_k_mem, &lg.d_w_k_mem, &mut ml.m_w_k_mem, &mut ml.v_w_k_mem,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.w_v_mem, &lg.d_w_v_mem, &mut ml.m_w_v_mem, &mut ml.v_w_v_mem,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.w_q_mem, &lg.d_w_q_mem, &mut ml.m_w_q_mem, &mut ml.v_w_q_mem,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

        // Gate weights
        adamw_one(&mut lp.w_alpha, &lg.d_w_alpha, &mut ml.m_w_alpha, &mut ml.v_w_alpha,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.b_alpha, &lg.d_b_alpha, &mut ml.m_b_alpha, &mut ml.v_b_alpha,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.w_theta, &lg.d_w_theta, &mut ml.m_w_theta, &mut ml.v_w_theta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.b_theta, &lg.d_b_theta, &mut ml.m_b_theta, &mut ml.v_b_theta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        // TODO(CS-39): clamp b_theta after update to prevent decay divergence
        adamw_one(&mut lp.w_eta, &lg.d_w_eta, &mut ml.m_w_eta, &mut ml.v_w_eta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
        adamw_one(&mut lp.b_eta, &lg.d_b_eta, &mut ml.m_b_eta, &mut ml.v_b_eta,
                  lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    }

    crate::dispatch::cuda_sync();
    grad_norm
}

// ══════════════════════════════════════════════════════════════════════
// GPU gradient norm computation + clipping
// ══════════════════════════════════════════════════════════════════════

/// Compute L2 norm of all gradient buffers on GPU.
/// Uses partial-reduction kernel + host-side sum of partials.
/// Only one small D2H copy (a few hundred f32 partial sums).
#[cfg(feature = "cuda")]
fn gpu_grad_norm(grads: &GpuMAGGrads, state: &mut GpuAdamWState) -> f32 {
    let mut total_sq = 0.0f64;

    // Helper: accumulate norm^2 from one gradient buffer
    let mut accum = |g: &GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let mut num_blocks: i32 = 0;
        unsafe {
            crate::cuda_ffi::grad_norm_sq_cuda(
                g.as_ptr(), state.norm_scratch.ptr(),
                n, &mut num_blocks,
            );
        }
        crate::dispatch::cuda_sync();
        let nb = num_blocks as usize;
        state.norm_scratch.slice(0, nb).copy_to_host(&mut state.norm_host[..nb]);
        for i in 0..nb {
            total_sq += state.norm_host[i] as f64;
        }
    };

    // SWA grads
    accum(&grads.d_w_embed);
    accum(&grads.d_w_q);
    accum(&grads.d_w_k);
    accum(&grads.d_w_v);
    accum(&grads.d_w_o);
    accum(&grads.d_w_unembed);

    // Level grads
    for lg in &grads.levels {
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

    (total_sq).sqrt() as f32
}

/// Scale all gradient buffers by a constant factor (for clipping).
#[cfg(feature = "cuda")]
fn gpu_scale_grads(grads: &mut GpuMAGGrads, scale: f32) {
    let scale_buf = |g: &mut GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        unsafe {
            crate::cuda_ffi::grad_scale_cuda(g.ptr(), scale, n);
        }
    };

    scale_buf(&mut grads.d_w_embed);
    scale_buf(&mut grads.d_w_q);
    scale_buf(&mut grads.d_w_k);
    scale_buf(&mut grads.d_w_v);
    scale_buf(&mut grads.d_w_o);
    scale_buf(&mut grads.d_w_unembed);

    for lg in &mut grads.levels {
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

    crate::dispatch::cuda_sync();
}
