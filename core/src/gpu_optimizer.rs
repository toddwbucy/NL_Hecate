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
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;

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
    m_ln_attn_gamma: GpuBuf<f32>, v_ln_attn_gamma: GpuBuf<f32>,
    m_ln_attn_beta: GpuBuf<f32>,  v_ln_attn_beta: GpuBuf<f32>,
    m_ln_mem_gamma: GpuBuf<f32>,  v_ln_mem_gamma: GpuBuf<f32>,
    m_ln_mem_beta: GpuBuf<f32>,   v_ln_mem_beta: GpuBuf<f32>,
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
    /// Level-local step counter for per-level bias correction.
    /// Counts how many times this level has actually fired, not global steps.
    level_step: u32,
    // SwiGluMlp moment buffers. zeros(1) for non-SwiGLU levels.
    m_gate_proj: GpuBuf<f32>,  v_gate_proj: GpuBuf<f32>,
    m_up_proj:   GpuBuf<f32>,  v_up_proj:   GpuBuf<f32>,
    m_down_proj: GpuBuf<f32>,  v_down_proj: GpuBuf<f32>,
    has_mlp: bool,
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
            m_ln_attn_gamma: GpuBuf::zeros(params.swa.ln_attn_gamma.len()),
            v_ln_attn_gamma: GpuBuf::zeros(params.swa.ln_attn_gamma.len()),
            m_ln_attn_beta: GpuBuf::zeros(params.swa.ln_attn_beta.len()),
            v_ln_attn_beta: GpuBuf::zeros(params.swa.ln_attn_beta.len()),
            m_ln_mem_gamma: GpuBuf::zeros(params.swa.ln_mem_gamma.len()),
            v_ln_mem_gamma: GpuBuf::zeros(params.swa.ln_mem_gamma.len()),
            m_ln_mem_beta: GpuBuf::zeros(params.swa.ln_mem_beta.len()),
            v_ln_mem_beta: GpuBuf::zeros(params.swa.ln_mem_beta.len()),
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
            level_step: 0,
            m_gate_proj: GpuBuf::zeros(lp.gate_proj.len().max(1)),
            v_gate_proj: GpuBuf::zeros(lp.gate_proj.len().max(1)),
            m_up_proj:   GpuBuf::zeros(lp.up_proj.len().max(1)),
            v_up_proj:   GpuBuf::zeros(lp.up_proj.len().max(1)),
            m_down_proj: GpuBuf::zeros(lp.down_proj.len().max(1)),
            v_down_proj: GpuBuf::zeros(lp.down_proj.len().max(1)),
            has_mlp: lp.has_mlp,
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
            if lp.has_mlp {
                for buf in [&lp.gate_proj, &lp.up_proj, &lp.down_proj] {
                    max_len = max_len.max(buf.len());
                }
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
pub(crate) fn adamw_one(
    w: &mut GpuBuf<f32>, g: &GpuBuf<f32>,
    m: &mut GpuBuf<f32>, v: &mut GpuBuf<f32>,
    lr: f32, beta1: f32, beta2: f32, eps: f32,
    bc1_inv: f32, bc2_inv: f32, weight_decay: f32,
) {
    let n = w.len() as i32;
    debug_assert_eq!(w.len(), g.len());
    debug_assert_eq!(w.len(), m.len());
    debug_assert_eq!(w.len(), v.len());
    let err = unsafe {
        crate::cuda_ffi::adamw_update_cuda(
            w.ptr(), g.as_ptr(), m.ptr(), v.ptr(),
            n, lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay,
        )
    };
    assert_eq!(err, 0, "adamw_update_cuda failed with cudaError_t={}", err);
}

/// Full AdamW weight update on GPU. Updates all params in-place, advances step counter.
/// Zero PCIe traffic — everything stays on device.
///
/// Pulse-gated: SWA params always update; CMS level params only update when
/// the Pulse fires for that level. Per-level step counters drive bias correction.
///
/// Returns the pre-clip gradient L2 norm (for logging). Returns 0.0 if clipping disabled.
#[cfg(feature = "cuda")]
pub fn gpu_adamw_update(
    params: &mut GpuMAGParams,
    grads: &mut GpuMAGGrads,
    state: &mut GpuAdamWState,
    pulse: &Pulse,
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

    // ── SWA weights (always active) ──────────────────────────────────
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
    adamw_one(&mut params.swa.ln_attn_gamma, &grads.d_ln_attn_gamma, &mut s.m_ln_attn_gamma, &mut s.v_ln_attn_gamma,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.ln_attn_beta, &grads.d_ln_attn_beta, &mut s.m_ln_attn_beta, &mut s.v_ln_attn_beta,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.ln_mem_gamma, &grads.d_ln_mem_gamma, &mut s.m_ln_mem_gamma, &mut s.v_ln_mem_gamma,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
    adamw_one(&mut params.swa.ln_mem_beta, &grads.d_ln_mem_beta, &mut s.m_ln_mem_beta, &mut s.v_ln_mem_beta,
              lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);

    // ── Per-level memory weights (Pulse-gated) ───────────────────────
    for (i, lg) in grads.levels.iter().enumerate() {
        if i >= pulse.active_levels.len() || !pulse.active_levels[i] {
            continue; // Level frozen: no update, no step increment
        }

        let lp = &mut params.levels[i];
        let ml = &mut state.levels[i];

        // Per-level bias correction
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

        // Gate weights
        adamw_one(&mut lp.w_alpha, &lg.d_w_alpha, &mut ml.m_w_alpha, &mut ml.v_w_alpha,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        adamw_one(&mut lp.b_alpha, &lg.d_b_alpha, &mut ml.m_b_alpha, &mut ml.v_b_alpha,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        adamw_one(&mut lp.w_theta, &lg.d_w_theta, &mut ml.m_w_theta, &mut ml.v_w_theta,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        adamw_one(&mut lp.b_theta, &lg.d_b_theta, &mut ml.m_b_theta, &mut ml.v_b_theta,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        // CS-39: clamp b_theta to bound the bias component of inner-loop learning rate.
        // softplus(b_theta) = theta_bias; clamping b_theta ∈ [-10, 2] keeps bias ∈ [~0, ~2.13].
        // NOTE: w_theta · [k,v] can still push per-token theta higher. A full theta_ceil
        // enforcement in the forward pass (via config.theta_ceil) is the proper long-term fix.
        // This bias clamp is a first-order stabilizer — sufficient for shakedown builds.
        unsafe {
            crate::cuda_ffi::clamp_f32_cuda(lp.b_theta.ptr(), lp.b_theta.len() as i32, -10.0, 2.0);
        }
        adamw_one(&mut lp.w_eta, &lg.d_w_eta, &mut ml.m_w_eta, &mut ml.v_w_eta,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        adamw_one(&mut lp.b_eta, &lg.d_b_eta, &mut ml.m_b_eta, &mut ml.v_b_eta,
                  lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);

        // SwiGluMlp projection weights (Pulse-gated: same level step as matrix rules)
        if ml.has_mlp {
            adamw_one(&mut lp.gate_proj, &lg.d_gate_proj,
                      &mut ml.m_gate_proj, &mut ml.v_gate_proj,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.up_proj, &lg.d_up_proj,
                      &mut ml.m_up_proj, &mut ml.v_up_proj,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
            adamw_one(&mut lp.down_proj, &lg.d_down_proj,
                      &mut ml.m_down_proj, &mut ml.v_down_proj,
                      lr, beta1, beta2, eps, lbc1_inv, lbc2_inv, weight_decay);
        }
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

    // SWA grads
    accum(&grads.d_w_embed);
    accum(&grads.d_w_q);
    accum(&grads.d_w_k);
    accum(&grads.d_w_v);
    accum(&grads.d_w_o);
    accum(&grads.d_w_unembed);
    accum(&grads.d_ln_attn_gamma);
    accum(&grads.d_ln_attn_beta);
    accum(&grads.d_ln_mem_gamma);
    accum(&grads.d_ln_mem_beta);

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
        if lg.has_mlp {
            accum(&lg.d_gate_proj);
            accum(&lg.d_up_proj);
            accum(&lg.d_down_proj);
        }
    }

    (total_sq).sqrt() as f32
}

/// Compute per-level L2 gradient norms. Returns Vec<f32> of length k.
/// Called before global clipping so the values reflect the true per-level
/// learning signal (post-clip all levels are scaled by the same factor,
/// losing relative dead-level information).
#[cfg(feature = "cuda")]
pub fn gpu_per_level_grad_norms(grads: &GpuMAGGrads, state: &mut GpuAdamWState) -> Vec<f32> {
    let mut level_norms = Vec::with_capacity(grads.levels.len());

    for lg in &grads.levels {
        let mut level_sq = 0.0f64;

        macro_rules! accum_level {
            ($g:expr) => {{
                let n = $g.len() as i32;
                if n > 0 {
                    let mut num_blocks: i32 = 0;
                    let err = unsafe {
                        crate::cuda_ffi::grad_norm_sq_cuda(
                            $g.as_ptr(), state.norm_scratch.ptr(), n, &mut num_blocks,
                        )
                    };
                    assert_eq!(err, 0, "grad_norm_sq_cuda failed with cudaError_t={}", err);
                    crate::dispatch::cuda_sync();
                    let nb = num_blocks as usize;
                    state.norm_scratch.slice(0, nb).copy_to_host(&mut state.norm_host[..nb]);
                    for i in 0..nb {
                        level_sq += state.norm_host[i] as f64;
                    }
                }
            }};
        }

        accum_level!(lg.d_w_k_mem);
        accum_level!(lg.d_w_v_mem);
        accum_level!(lg.d_w_q_mem);
        accum_level!(lg.d_w_alpha);
        accum_level!(lg.d_b_alpha);
        accum_level!(lg.d_w_theta);
        accum_level!(lg.d_b_theta);
        accum_level!(lg.d_w_eta);
        accum_level!(lg.d_b_eta);
        if lg.has_mlp {
            accum_level!(lg.d_gate_proj);
            accum_level!(lg.d_up_proj);
            accum_level!(lg.d_down_proj);
        }

        level_norms.push(level_sq.sqrt() as f32);
    }

    level_norms
}

/// Scale all gradient buffers by a constant factor (for clipping).
#[cfg(feature = "cuda")]
fn gpu_scale_grads(grads: &mut GpuMAGGrads, scale: f32) {
    let scale_buf = |g: &mut GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let err = unsafe {
            crate::cuda_ffi::grad_scale_cuda(g.ptr(), scale, n)
        };
        assert_eq!(err, 0, "grad_scale_cuda failed with cudaError_t={}", err);
    };

    scale_buf(&mut grads.d_w_embed);
    scale_buf(&mut grads.d_w_q);
    scale_buf(&mut grads.d_w_k);
    scale_buf(&mut grads.d_w_v);
    scale_buf(&mut grads.d_w_o);
    scale_buf(&mut grads.d_w_unembed);
    scale_buf(&mut grads.d_ln_attn_gamma);
    scale_buf(&mut grads.d_ln_attn_beta);
    scale_buf(&mut grads.d_ln_mem_gamma);
    scale_buf(&mut grads.d_ln_mem_beta);

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
        if lg.has_mlp {
            scale_buf(&mut lg.d_gate_proj);
            scale_buf(&mut lg.d_up_proj);
            scale_buf(&mut lg.d_down_proj);
        }
    }

    crate::dispatch::cuda_sync();
}
