/// GPU-resident optimizer state and update functions (AdamW + M3).
///
/// AdamW: single-frequency, diagonal scaling. Suitable for k=1 CMS.
/// M3: multi-scale momentum with Newton-Schulz orthogonalization (Eq 75).
///     Suitable for k>=2 CMS. See spec 34 (34_m3_gpu_integration.md).
///
/// Both are outer-loop optimizers only (CS-27). Inner-loop memory updates
/// are handled by the memory rules themselves.
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
use crate::gpu_params::GpuMAGParams;
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
// Host-side optimizer snapshot (spec 33)
// ══════════════════════════════════════════════════════════════════════

/// Host-resident copy of AdamW optimizer state for snapshot/restore.
/// Flat concatenation avoids mirroring every field name — layout is
/// deterministic within a single (params, k) shape.
#[cfg(feature = "cuda")]
pub struct HostOptimizerState {
    /// Concatenated m+v for all SWA params.
    pub swa_moments: Vec<f32>,
    /// Concatenated m+v per CMS level.
    pub level_moments: Vec<Vec<f32>>,
    /// Per-level step counters (for bias correction).
    pub level_steps: Vec<u32>,
    /// Global optimizer step counter.
    pub step: u32,
}

/// Download one GpuBuf to the end of a host Vec.
#[cfg(feature = "cuda")]
#[inline]
fn download_buf(buf: &GpuBuf<f32>, dst: &mut Vec<f32>) {
    let offset = dst.len();
    dst.resize(offset + buf.len(), 0.0);
    buf.copy_to_host(&mut dst[offset..]);
}

/// Upload a slice from a host Vec into a new GpuBuf, advancing the cursor.
#[cfg(feature = "cuda")]
#[inline]
fn upload_buf(src: &[f32], cursor: &mut usize, len: usize) -> GpuBuf<f32> {
    let buf = GpuBuf::from_host(&src[*cursor..*cursor + len]);
    *cursor += len;
    buf
}

#[cfg(feature = "cuda")]
impl GpuAdamWState {
    /// Download all optimizer state to host (spec 33).
    pub fn to_host(&self) -> HostOptimizerState {
        let mut swa_moments = Vec::new();
        // SWA: m then v for each param, in field order
        for buf in [
            &self.swa.m_embed, &self.swa.v_embed,
            &self.swa.m_q, &self.swa.v_q,
            &self.swa.m_k, &self.swa.v_k,
            &self.swa.m_v, &self.swa.v_v,
            &self.swa.m_o, &self.swa.v_o,
            &self.swa.m_unembed, &self.swa.v_unembed,
            &self.swa.m_ln_attn_gamma, &self.swa.v_ln_attn_gamma,
            &self.swa.m_ln_attn_beta, &self.swa.v_ln_attn_beta,
            &self.swa.m_ln_mem_gamma, &self.swa.v_ln_mem_gamma,
            &self.swa.m_ln_mem_beta, &self.swa.v_ln_mem_beta,
        ] {
            download_buf(buf, &mut swa_moments);
        }

        let mut level_moments = Vec::with_capacity(self.levels.len());
        let mut level_steps = Vec::with_capacity(self.levels.len());
        for lv in &self.levels {
            let mut lm = Vec::new();
            for buf in [
                &lv.m_w_k_mem, &lv.v_w_k_mem,
                &lv.m_w_v_mem, &lv.v_w_v_mem,
                &lv.m_w_q_mem, &lv.v_w_q_mem,
                &lv.m_w_alpha, &lv.v_w_alpha,
                &lv.m_b_alpha, &lv.v_b_alpha,
                &lv.m_w_theta, &lv.v_w_theta,
                &lv.m_b_theta, &lv.v_b_theta,
                &lv.m_w_eta, &lv.v_w_eta,
                &lv.m_b_eta, &lv.v_b_eta,
                &lv.m_gate_proj, &lv.v_gate_proj,
                &lv.m_up_proj, &lv.v_up_proj,
                &lv.m_down_proj, &lv.v_down_proj,
            ] {
                download_buf(buf, &mut lm);
            }
            level_moments.push(lm);
            level_steps.push(lv.level_step);
        }

        HostOptimizerState {
            swa_moments,
            level_moments,
            level_steps,
            step: self.step,
        }
    }

    /// Reconstruct GPU optimizer state from host snapshot (spec 33).
    /// `params` provides buffer sizes for the norm_scratch allocation.
    pub fn from_host(host: &HostOptimizerState, params: &GpuMAGParams) -> Self {
        let mut c = 0usize;
        let s = &host.swa_moments;
        let swa = MomentSWA {
            m_embed: upload_buf(s, &mut c, params.swa.w_embed.len()),
            v_embed: upload_buf(s, &mut c, params.swa.w_embed.len()),
            m_q: upload_buf(s, &mut c, params.swa.w_q.len()),
            v_q: upload_buf(s, &mut c, params.swa.w_q.len()),
            m_k: upload_buf(s, &mut c, params.swa.w_k.len()),
            v_k: upload_buf(s, &mut c, params.swa.w_k.len()),
            m_v: upload_buf(s, &mut c, params.swa.w_v.len()),
            v_v: upload_buf(s, &mut c, params.swa.w_v.len()),
            m_o: upload_buf(s, &mut c, params.swa.w_o.len()),
            v_o: upload_buf(s, &mut c, params.swa.w_o.len()),
            m_unembed: upload_buf(s, &mut c, params.swa.w_unembed.len()),
            v_unembed: upload_buf(s, &mut c, params.swa.w_unembed.len()),
            m_ln_attn_gamma: upload_buf(s, &mut c, params.swa.ln_attn_gamma.len()),
            v_ln_attn_gamma: upload_buf(s, &mut c, params.swa.ln_attn_gamma.len()),
            m_ln_attn_beta: upload_buf(s, &mut c, params.swa.ln_attn_beta.len()),
            v_ln_attn_beta: upload_buf(s, &mut c, params.swa.ln_attn_beta.len()),
            m_ln_mem_gamma: upload_buf(s, &mut c, params.swa.ln_mem_gamma.len()),
            v_ln_mem_gamma: upload_buf(s, &mut c, params.swa.ln_mem_gamma.len()),
            m_ln_mem_beta: upload_buf(s, &mut c, params.swa.ln_mem_beta.len()),
            v_ln_mem_beta: upload_buf(s, &mut c, params.swa.ln_mem_beta.len()),
        };
        assert_eq!(c, s.len(), "SWA moment size mismatch: expected {} but snapshot has {}", c, s.len());

        assert!(
            host.level_moments.len() == host.level_steps.len()
                && host.level_steps.len() == params.levels.len(),
            "Level count mismatch: snapshot has {} moment vecs, {} step counters, but model has {} levels",
            host.level_moments.len(), host.level_steps.len(), params.levels.len(),
        );

        let levels: Vec<MomentLevel> = host.level_moments.iter()
            .zip(host.level_steps.iter())
            .zip(params.levels.iter())
            .map(|((lm, &ls), lp)| {
                let mut c = 0usize;
                let ml = MomentLevel {
                    m_w_k_mem: upload_buf(lm, &mut c, lp.w_k_mem.len()),
                    v_w_k_mem: upload_buf(lm, &mut c, lp.w_k_mem.len()),
                    m_w_v_mem: upload_buf(lm, &mut c, lp.w_v_mem.len()),
                    v_w_v_mem: upload_buf(lm, &mut c, lp.w_v_mem.len()),
                    m_w_q_mem: upload_buf(lm, &mut c, lp.w_q_mem.len()),
                    v_w_q_mem: upload_buf(lm, &mut c, lp.w_q_mem.len()),
                    m_w_alpha: upload_buf(lm, &mut c, lp.w_alpha.len()),
                    v_w_alpha: upload_buf(lm, &mut c, lp.w_alpha.len()),
                    m_b_alpha: upload_buf(lm, &mut c, lp.b_alpha.len()),
                    v_b_alpha: upload_buf(lm, &mut c, lp.b_alpha.len()),
                    m_w_theta: upload_buf(lm, &mut c, lp.w_theta.len()),
                    v_w_theta: upload_buf(lm, &mut c, lp.w_theta.len()),
                    m_b_theta: upload_buf(lm, &mut c, lp.b_theta.len()),
                    v_b_theta: upload_buf(lm, &mut c, lp.b_theta.len()),
                    m_w_eta: upload_buf(lm, &mut c, lp.w_eta.len()),
                    v_w_eta: upload_buf(lm, &mut c, lp.w_eta.len()),
                    m_b_eta: upload_buf(lm, &mut c, lp.b_eta.len()),
                    v_b_eta: upload_buf(lm, &mut c, lp.b_eta.len()),
                    m_gate_proj: upload_buf(lm, &mut c, lp.gate_proj.len().max(1)),
                    v_gate_proj: upload_buf(lm, &mut c, lp.gate_proj.len().max(1)),
                    m_up_proj: upload_buf(lm, &mut c, lp.up_proj.len().max(1)),
                    v_up_proj: upload_buf(lm, &mut c, lp.up_proj.len().max(1)),
                    m_down_proj: upload_buf(lm, &mut c, lp.down_proj.len().max(1)),
                    v_down_proj: upload_buf(lm, &mut c, lp.down_proj.len().max(1)),
                    level_step: ls,
                    has_mlp: lp.has_mlp,
                };
                assert_eq!(c, lm.len(), "Level moment size mismatch: expected {} but snapshot has {}", c, lm.len());
                ml
            })
            .collect();

        // Recompute norm_scratch from params (same logic as from_params)
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
            step: host.step,
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

// ══════════════════════════════════════════════════════════════════════
// M3 Optimizer — multi-scale momentum with Newton-Schulz (spec 34)
// ══════════════════════════════════════════════════════════════════════

/// M3 optimizer configuration.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub struct GpuM3Config {
    pub beta1: f32,         // fast momentum (default 0.9)
    pub beta2: f32,         // second moment (default 0.999)
    pub beta3: f32,         // slow momentum (default 0.99)
    pub alpha: f32,         // slow momentum weight in combined update (default 0.5)
    pub chunk_size: u32,    // Ĉ — M2 update frequency (default 8)
    pub ns_iterations: u32, // Newton-Schulz iterations T (default 5)
    pub eps: f32,           // numerical stability (default 1e-8)
}

#[cfg(feature = "cuda")]
impl Default for GpuM3Config {
    fn default() -> Self {
        GpuM3Config {
            beta1: 0.9,
            beta2: 0.999,
            beta3: 0.99,
            alpha: 0.5,
            chunk_size: 8,
            ns_iterations: 5,
            eps: 1e-8,
        }
    }
}

/// M3 moment buffers for SWA weights.
/// Three buffers per param: m1 (fast), m2 (slow), v (second moment).
#[cfg(feature = "cuda")]
struct M3MomentSWA {
    // 2D params — will get NS orthogonalization
    m1_embed: GpuBuf<f32>,   m2_embed: GpuBuf<f32>,   v_embed: GpuBuf<f32>,
    m1_q: GpuBuf<f32>,       m2_q: GpuBuf<f32>,       v_q: GpuBuf<f32>,
    m1_k: GpuBuf<f32>,       m2_k: GpuBuf<f32>,       v_k: GpuBuf<f32>,
    m1_v: GpuBuf<f32>,       m2_v: GpuBuf<f32>,       v_v: GpuBuf<f32>,
    m1_o: GpuBuf<f32>,       m2_o: GpuBuf<f32>,       v_o: GpuBuf<f32>,
    m1_unembed: GpuBuf<f32>, m2_unembed: GpuBuf<f32>, v_unembed: GpuBuf<f32>,
    // 1D params — Adam-style V division
    m1_ln_attn_gamma: GpuBuf<f32>, m2_ln_attn_gamma: GpuBuf<f32>, v_ln_attn_gamma: GpuBuf<f32>,
    m1_ln_attn_beta: GpuBuf<f32>,  m2_ln_attn_beta: GpuBuf<f32>,  v_ln_attn_beta: GpuBuf<f32>,
    m1_ln_mem_gamma: GpuBuf<f32>,  m2_ln_mem_gamma: GpuBuf<f32>,  v_ln_mem_gamma: GpuBuf<f32>,
    m1_ln_mem_beta: GpuBuf<f32>,   m2_ln_mem_beta: GpuBuf<f32>,   v_ln_mem_beta: GpuBuf<f32>,
}

/// M3 moment buffers for one memory level.
#[cfg(feature = "cuda")]
struct M3MomentLevel {
    // 2D params
    m1_w_k_mem: GpuBuf<f32>,  m2_w_k_mem: GpuBuf<f32>,  v_w_k_mem: GpuBuf<f32>,
    m1_w_v_mem: GpuBuf<f32>,  m2_w_v_mem: GpuBuf<f32>,  v_w_v_mem: GpuBuf<f32>,
    m1_w_q_mem: GpuBuf<f32>,  m2_w_q_mem: GpuBuf<f32>,  v_w_q_mem: GpuBuf<f32>,
    // 1D params (gate biases)
    m1_w_alpha: GpuBuf<f32>,  m2_w_alpha: GpuBuf<f32>,  v_w_alpha: GpuBuf<f32>,
    m1_b_alpha: GpuBuf<f32>,  m2_b_alpha: GpuBuf<f32>,  v_b_alpha: GpuBuf<f32>,
    m1_w_theta: GpuBuf<f32>,  m2_w_theta: GpuBuf<f32>,  v_w_theta: GpuBuf<f32>,
    m1_b_theta: GpuBuf<f32>,  m2_b_theta: GpuBuf<f32>,  v_b_theta: GpuBuf<f32>,
    m1_w_eta: GpuBuf<f32>,    m2_w_eta: GpuBuf<f32>,    v_w_eta: GpuBuf<f32>,
    m1_b_eta: GpuBuf<f32>,    m2_b_eta: GpuBuf<f32>,    v_b_eta: GpuBuf<f32>,
    // MLP weights (2D)
    m1_gate_proj: GpuBuf<f32>,  m2_gate_proj: GpuBuf<f32>,  v_gate_proj: GpuBuf<f32>,
    m1_up_proj: GpuBuf<f32>,    m2_up_proj: GpuBuf<f32>,    v_up_proj: GpuBuf<f32>,
    m1_down_proj: GpuBuf<f32>,  m2_down_proj: GpuBuf<f32>,  v_down_proj: GpuBuf<f32>,
    level_step: u32,
    has_mlp: bool,
}

/// GPU-resident M3 optimizer state. Spec 34.
#[cfg(feature = "cuda")]
pub struct GpuM3State {
    swa: M3MomentSWA,
    levels: Vec<M3MomentLevel>,
    pub step: u32,
    pub config: GpuM3Config,
    /// NS scratch buffers (allocated once in from_params, reused every step).
    /// Layout: ns_ata is [d × d] for X @ X^T. The others are [max_2d]
    /// to hold working copies during NS iteration.
    ns_ata: GpuBuf<f32>,    // X @ X^T intermediate [d × d]
    ns_work: GpuBuf<f32>,   // working copy of normalized X [max_2d]
    ns_ax: GpuBuf<f32>,     // A @ X intermediate [max_2d]
    ns_aax: GpuBuf<f32>,    // A @ (A @ X) intermediate [max_2d]
    /// Scratch for Frobenius norm partial reduction
    norm_scratch: GpuBuf<f32>,
    norm_host: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuM3State {
    /// Create zero-initialized M3 state matching param shapes.
    ///
    /// `d` is the model dimension — needed to size the NS ATA scratch buffer
    /// (the min dimension of all weight matrices is d, so ATA is d×d).
    pub fn from_params(params: &GpuMAGParams, config: GpuM3Config, d: usize) -> Self {
        let z3 = |len: usize| -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
            (GpuBuf::zeros(len), GpuBuf::zeros(len), GpuBuf::zeros(len))
        };

        let (m1_embed, m2_embed, v_embed) = z3(params.swa.w_embed.len());
        let (m1_q, m2_q, v_q) = z3(params.swa.w_q.len());
        let (m1_k, m2_k, v_k) = z3(params.swa.w_k.len());
        let (m1_v, m2_v, v_v) = z3(params.swa.w_v.len());
        let (m1_o, m2_o, v_o) = z3(params.swa.w_o.len());
        let (m1_unembed, m2_unembed, v_unembed) = z3(params.swa.w_unembed.len());
        let (m1_ln_attn_gamma, m2_ln_attn_gamma, v_ln_attn_gamma) = z3(params.swa.ln_attn_gamma.len());
        let (m1_ln_attn_beta, m2_ln_attn_beta, v_ln_attn_beta) = z3(params.swa.ln_attn_beta.len());
        let (m1_ln_mem_gamma, m2_ln_mem_gamma, v_ln_mem_gamma) = z3(params.swa.ln_mem_gamma.len());
        let (m1_ln_mem_beta, m2_ln_mem_beta, v_ln_mem_beta) = z3(params.swa.ln_mem_beta.len());

        let swa = M3MomentSWA {
            m1_embed, m2_embed, v_embed,
            m1_q, m2_q, v_q,
            m1_k, m2_k, v_k,
            m1_v, m2_v, v_v,
            m1_o, m2_o, v_o,
            m1_unembed, m2_unembed, v_unembed,
            m1_ln_attn_gamma, m2_ln_attn_gamma, v_ln_attn_gamma,
            m1_ln_attn_beta, m2_ln_attn_beta, v_ln_attn_beta,
            m1_ln_mem_gamma, m2_ln_mem_gamma, v_ln_mem_gamma,
            m1_ln_mem_beta, m2_ln_mem_beta, v_ln_mem_beta,
        };

        let levels: Vec<M3MomentLevel> = params.levels.iter().map(|lp| {
            let (m1_w_k_mem, m2_w_k_mem, v_w_k_mem) = z3(lp.w_k_mem.len());
            let (m1_w_v_mem, m2_w_v_mem, v_w_v_mem) = z3(lp.w_v_mem.len());
            let (m1_w_q_mem, m2_w_q_mem, v_w_q_mem) = z3(lp.w_q_mem.len());
            let (m1_w_alpha, m2_w_alpha, v_w_alpha) = z3(lp.w_alpha.len());
            let (m1_b_alpha, m2_b_alpha, v_b_alpha) = z3(lp.b_alpha.len());
            let (m1_w_theta, m2_w_theta, v_w_theta) = z3(lp.w_theta.len());
            let (m1_b_theta, m2_b_theta, v_b_theta) = z3(lp.b_theta.len());
            let (m1_w_eta, m2_w_eta, v_w_eta) = z3(lp.w_eta.len());
            let (m1_b_eta, m2_b_eta, v_b_eta) = z3(lp.b_eta.len());
            let (m1_gate_proj, m2_gate_proj, v_gate_proj) = z3(lp.gate_proj.len().max(1));
            let (m1_up_proj, m2_up_proj, v_up_proj) = z3(lp.up_proj.len().max(1));
            let (m1_down_proj, m2_down_proj, v_down_proj) = z3(lp.down_proj.len().max(1));
            M3MomentLevel {
                m1_w_k_mem, m2_w_k_mem, v_w_k_mem,
                m1_w_v_mem, m2_w_v_mem, v_w_v_mem,
                m1_w_q_mem, m2_w_q_mem, v_w_q_mem,
                m1_w_alpha, m2_w_alpha, v_w_alpha,
                m1_b_alpha, m2_b_alpha, v_b_alpha,
                m1_w_theta, m2_w_theta, v_w_theta,
                m1_b_theta, m2_b_theta, v_b_theta,
                m1_w_eta, m2_w_eta, v_w_eta,
                m1_b_eta, m2_b_eta, v_b_eta,
                m1_gate_proj, m2_gate_proj, v_gate_proj,
                m1_up_proj, m2_up_proj, v_up_proj,
                m1_down_proj, m2_down_proj, v_down_proj,
                level_step: 0,
                has_mlp: lp.has_mlp,
            }
        }).collect();

        // NS scratch: sized for largest 2D param buffer
        let mut max_2d = params.swa.w_embed.len();
        for buf in [&params.swa.w_q, &params.swa.w_k, &params.swa.w_v,
                     &params.swa.w_o, &params.swa.w_unembed] {
            max_2d = max_2d.max(buf.len());
        }
        for lp in &params.levels {
            for buf in [&lp.w_k_mem, &lp.w_v_mem, &lp.w_q_mem] {
                max_2d = max_2d.max(buf.len());
            }
            if lp.has_mlp {
                for buf in [&lp.gate_proj, &lp.up_proj, &lp.down_proj] {
                    max_2d = max_2d.max(buf.len());
                }
            }
        }

        let max_partials = max_2d / 256 + 1;

        GpuM3State {
            swa,
            levels,
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

// ── M3 helper: EMA update for one param group ──────────────────────────

/// Call the fused M3 EMA kernel for one param group.
#[cfg(feature = "cuda")]
#[inline]
fn m3_ema_one(
    m1: &mut GpuBuf<f32>, m2: &mut GpuBuf<f32>, v: &mut GpuBuf<f32>,
    g: &GpuBuf<f32>,
    beta1: f32, beta2: f32, beta3: f32,
    update_m2: bool,
) {
    let n = g.len() as i32;
    let err = unsafe {
        crate::cuda_ffi::m3_ema_update_cuda(
            m1.ptr(), m2.ptr(), v.ptr(), g.as_ptr(),
            n, beta1, beta2, beta3,
            if update_m2 { 1 } else { 0 },
        )
    };
    assert_eq!(err, 0, "m3_ema_update_cuda failed: {}", err);
}

/// Apply 1D (Adam-style) param update.
#[cfg(feature = "cuda")]
#[inline]
fn m3_apply_1d_one(
    w: &mut GpuBuf<f32>,
    m1: &GpuBuf<f32>, m2: &GpuBuf<f32>, v: &GpuBuf<f32>,
    lr: f32, alpha: f32, eps: f32, bc2: f32,
) {
    let n = w.len() as i32;
    let err = unsafe {
        crate::cuda_ffi::m3_apply_1d_cuda(
            w.ptr(), m1.as_ptr(), m2.as_ptr(), v.as_ptr(),
            n, lr, alpha, eps, bc2,
        )
    };
    assert_eq!(err, 0, "m3_apply_1d_cuda failed: {}", err);
}

/// Compute Frobenius norm of a GPU buffer using partial reduction.
#[cfg(feature = "cuda")]
fn gpu_frob_norm(buf: &GpuBuf<f32>, scratch: &mut GpuBuf<f32>, host: &mut [f32]) -> f32 {
    let n = buf.len() as i32;
    let mut num_blocks: i32 = 0;
    let err = unsafe {
        crate::cuda_ffi::frob_norm_sq_cuda(
            buf.as_ptr(), scratch.ptr(), n, &mut num_blocks,
        )
    };
    assert_eq!(err, 0, "frob_norm_sq_cuda failed: {}", err);
    crate::dispatch::cuda_sync();
    let nb = num_blocks as usize;
    scratch.copy_to_host(&mut host[..nb]);
    host[..nb].iter().sum::<f32>().sqrt()
}

/// Apply one NS-orthogonalized momentum to a 2D weight matrix.
///
/// Computes: w += lr_scale * ||m|| * NS(m / ||m||)
///
/// For M1: call with lr_scale = -lr
/// For M2: call with lr_scale = -lr * alpha
///
/// Tall matrices (rows > cols) are transposed before NS so the iteration
/// always works on the smaller (d × d) ATA matrix. Uses existing
/// `transpose_copy_cuda` — the O(rows*cols) copy is negligible vs T matmuls.
#[cfg(feature = "cuda")]
fn m3_ns_apply(
    w: &mut GpuBuf<f32>,
    m: &GpuBuf<f32>,         // momentum buffer (M1 or M2)
    rows: usize, cols: usize,
    lr_scale: f32,            // -lr for M1, -lr*alpha for M2
    ns_iters: u32,
    ns_ata: &mut GpuBuf<f32>,
    ns_work: &mut GpuBuf<f32>,
    ns_ax: &mut GpuBuf<f32>,
    ns_aax: &mut GpuBuf<f32>,
    norm_scratch: &mut GpuBuf<f32>,
    norm_host: &mut [f32],
) {
    let n = rows * cols;

    // Frobenius norm for pre-normalization
    let frob = gpu_frob_norm(m, norm_scratch, norm_host);
    if frob < 1e-7 {
        return; // near-zero momentum — skip
    }

    let tall = rows > cols;
    let (r, c) = if tall { (cols, rows) } else { (rows, cols) };

    // Copy m into ns_work and normalize. For tall matrices, transpose first
    // so ns_work holds a fat [r, c] matrix (r <= c).
    if tall {
        unsafe {
            crate::cuda_ffi::transpose_copy_cuda(
                m.as_ptr(), ns_work.ptr(), rows as i32, cols as i32,
            );
        }
    } else {
        ns_work.copy_from_device(m);
    }
    let err = unsafe { crate::cuda_ffi::scale_buf_cuda(ns_work.ptr(), 1.0 / frob, n as i32) };
    assert_eq!(err, 0);

    // Newton-Schulz iterations: X_new = a*X + b*(A@X) + c*(A@(A@X))
    // where A = X @ X^T. Muon polynomial coefficients (T=5 convergence).
    let (a, b, c_coeff) = (3.4445f32, -4.7750f32, 2.0315f32);

    for _ in 0..ns_iters {
        // A = X @ X^T → ns_ata[r, r]
        crate::dispatch::cublas_matmul_transb_dd(
            ns_work, ns_work, ns_ata, r, c, r, 0.0,
        );
        // AX = A @ X → ns_ax[r, c]
        crate::dispatch::cublas_matmul_dd(ns_ata, ns_work, ns_ax, r, r, c, 0.0);
        // A²X = A @ AX → ns_aax[r, c]
        crate::dispatch::cublas_matmul_dd(ns_ata, ns_ax, ns_aax, r, r, c, 0.0);
        // Polynomial: X = a*X + b*AX + c*A²X
        let err = unsafe {
            crate::cuda_ffi::m3_ns_poly_cuda(
                ns_work.ptr(), ns_ax.as_ptr(), ns_aax.as_ptr(),
                (r * c) as i32, a, b, c_coeff,
            )
        };
        assert_eq!(err, 0, "m3_ns_poly_cuda failed");
    }

    // Apply: w += lr_scale * ||m|| * NS(m/||m||)
    let final_scale = lr_scale * frob;
    if tall {
        // Transpose ns_work[cols,rows] → ns_ax[rows,cols], then apply via saxpy
        unsafe {
            crate::cuda_ffi::transpose_copy_cuda(
                ns_work.as_ptr(), ns_ax.ptr(), cols as i32, rows as i32,
            );
            crate::cuda_ffi::saxpy_cuda(final_scale, ns_ax.as_ptr(), w.ptr(), n as i32);
        }
    } else {
        unsafe {
            crate::cuda_ffi::saxpy_cuda(final_scale, ns_work.as_ptr(), w.ptr(), n as i32);
        }
    }
}

/// Full M3 optimizer step on GPU. Updates all params in-place.
///
/// Pulse-gated: SWA params always update; CMS level params only update
/// when the Pulse fires for that level.
///
/// Returns the pre-clip gradient L2 norm (for logging). Returns 0.0 if clipping disabled.
#[cfg(feature = "cuda")]
pub fn gpu_m3_update(
    params: &mut GpuMAGParams,
    grads: &mut GpuMAGGrads,
    state: &mut GpuM3State,
    pulse: &Pulse,
    lr: f32,
    max_grad_norm: f32,
    d: usize,  // model dimension (for NS reshape: weight matrices are d x d or vocab x d)
) -> f32 {
    state.step += 1;
    let cfg = &state.config;
    let update_m2 = state.step % cfg.chunk_size == 0;
    let bc2 = 1.0 - cfg.beta2.powi(state.step as i32);

    // ── Gradient clipping (reuse AdamW norm infrastructure) ───────────
    let grad_norm = if max_grad_norm > 0.0 {
        // Borrow norm scratch temporarily — this is the same pattern as AdamW
        let norm = gpu_grad_norm_with_scratch(grads, &mut state.norm_scratch, &mut state.norm_host);
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

    // EMA updates (all 2D SWA)
    m3_ema_one(&mut s.m1_embed, &mut s.m2_embed, &mut s.v_embed, &grads.d_w_embed,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_q, &mut s.m2_q, &mut s.v_q, &grads.d_w_q,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_k, &mut s.m2_k, &mut s.v_k, &grads.d_w_k,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_v, &mut s.m2_v, &mut s.v_v, &grads.d_w_v,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_o, &mut s.m2_o, &mut s.v_o, &grads.d_w_o,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_unembed, &mut s.m2_unembed, &mut s.v_unembed, &grads.d_w_unembed,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

    // 1D SWA params: EMA + Adam-style apply
    m3_ema_one(&mut s.m1_ln_attn_gamma, &mut s.m2_ln_attn_gamma, &mut s.v_ln_attn_gamma, &grads.d_ln_attn_gamma,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_ln_attn_beta, &mut s.m2_ln_attn_beta, &mut s.v_ln_attn_beta, &grads.d_ln_attn_beta,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_ln_mem_gamma, &mut s.m2_ln_mem_gamma, &mut s.v_ln_mem_gamma, &grads.d_ln_mem_gamma,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
    m3_ema_one(&mut s.m1_ln_mem_beta, &mut s.m2_ln_mem_beta, &mut s.v_ln_mem_beta, &grads.d_ln_mem_beta,
               cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

    // Apply 1D params (LN)
    m3_apply_1d_one(&mut params.swa.ln_attn_gamma, &s.m1_ln_attn_gamma, &s.m2_ln_attn_gamma, &s.v_ln_attn_gamma,
                    lr, cfg.alpha, cfg.eps, bc2);
    m3_apply_1d_one(&mut params.swa.ln_attn_beta, &s.m1_ln_attn_beta, &s.m2_ln_attn_beta, &s.v_ln_attn_beta,
                    lr, cfg.alpha, cfg.eps, bc2);
    m3_apply_1d_one(&mut params.swa.ln_mem_gamma, &s.m1_ln_mem_gamma, &s.m2_ln_mem_gamma, &s.v_ln_mem_gamma,
                    lr, cfg.alpha, cfg.eps, bc2);
    m3_apply_1d_one(&mut params.swa.ln_mem_beta, &s.m1_ln_mem_beta, &s.m2_ln_mem_beta, &s.v_ln_mem_beta,
                    lr, cfg.alpha, cfg.eps, bc2);

    // Apply 2D SWA params via NS (M1 and M2 applied separately via saxpy)
    // embed: vocab × d, q/k/v/o: d × d, unembed: d × vocab
    let vocab = params.swa.w_embed.len() / d;
    let ns_iters = cfg.ns_iterations;
    let m2_scale = -lr * cfg.alpha;

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

    ns_2d!(params.swa.w_embed, s.m1_embed, s.m2_embed, vocab, d);
    ns_2d!(params.swa.w_q, s.m1_q, s.m2_q, d, d);
    ns_2d!(params.swa.w_k, s.m1_k, s.m2_k, d, d);
    ns_2d!(params.swa.w_v, s.m1_v, s.m2_v, d, d);
    ns_2d!(params.swa.w_o, s.m1_o, s.m2_o, d, d);
    ns_2d!(params.swa.w_unembed, s.m1_unembed, s.m2_unembed, d, vocab);

    // ── Per-level memory weights (Pulse-gated) ───────────────────────
    for (i, lg) in grads.levels.iter().enumerate() {
        if i >= pulse.active_levels.len() || !pulse.active_levels[i] {
            continue;
        }

        let lp = &mut params.levels[i];
        let ml = &mut state.levels[i];
        ml.level_step += 1;
        let level_bc2 = 1.0 - cfg.beta2.powi(ml.level_step as i32);

        // 2D level params: EMA
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
        m3_apply_1d_one(&mut lp.w_eta, &ml.m1_w_eta, &ml.m2_w_eta, &ml.v_w_eta,
                        lr, cfg.alpha, cfg.eps, level_bc2);
        m3_apply_1d_one(&mut lp.b_eta, &ml.m1_b_eta, &ml.m2_b_eta, &ml.v_b_eta,
                        lr, cfg.alpha, cfg.eps, level_bc2);

        // Apply 2D level params via NS
        ns_2d!(lp.w_k_mem, ml.m1_w_k_mem, ml.m2_w_k_mem, d, d);
        ns_2d!(lp.w_v_mem, ml.m1_w_v_mem, ml.m2_w_v_mem, d, d);
        ns_2d!(lp.w_q_mem, ml.m1_w_q_mem, ml.m2_w_q_mem, d, d);

        // MLP weights (2D)
        if ml.has_mlp {
            m3_ema_one(&mut ml.m1_gate_proj, &mut ml.m2_gate_proj, &mut ml.v_gate_proj, &lg.d_gate_proj,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_up_proj, &mut ml.m2_up_proj, &mut ml.v_up_proj, &lg.d_up_proj,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);
            m3_ema_one(&mut ml.m1_down_proj, &mut ml.m2_down_proj, &mut ml.v_down_proj, &lg.d_down_proj,
                       cfg.beta1, cfg.beta2, cfg.beta3, update_m2);

            // MLP shapes per MAGParams: gate_proj/up_proj are [inter, d], down_proj is [d, inter]
            let ff_dim = lp.gate_proj.len() / d;
            ns_2d!(lp.gate_proj, ml.m1_gate_proj, ml.m2_gate_proj, ff_dim, d);
            ns_2d!(lp.up_proj, ml.m1_up_proj, ml.m2_up_proj, ff_dim, d);
            ns_2d!(lp.down_proj, ml.m1_down_proj, ml.m2_down_proj, d, ff_dim);
        }
    }

    crate::dispatch::cuda_sync();
    grad_norm
}

/// Compute gradient norm using provided scratch buffers (for M3).
#[cfg(feature = "cuda")]
fn gpu_grad_norm_with_scratch(
    grads: &GpuMAGGrads,
    scratch: &mut GpuBuf<f32>,
    host: &mut Vec<f32>,
) -> f32 {
    let mut total = 0.0f32;
    let mut add_norm = |g: &GpuBuf<f32>| {
        total += gpu_frob_norm(g, scratch, host).powi(2);
    };
    add_norm(&grads.d_w_embed);
    add_norm(&grads.d_w_q);
    add_norm(&grads.d_w_k);
    add_norm(&grads.d_w_v);
    add_norm(&grads.d_w_o);
    add_norm(&grads.d_w_unembed);
    add_norm(&grads.d_ln_attn_gamma);
    add_norm(&grads.d_ln_attn_beta);
    add_norm(&grads.d_ln_mem_gamma);
    add_norm(&grads.d_ln_mem_beta);
    for lg in &grads.levels {
        add_norm(&lg.d_w_k_mem);
        add_norm(&lg.d_w_v_mem);
        add_norm(&lg.d_w_q_mem);
        add_norm(&lg.d_w_alpha);
        add_norm(&lg.d_b_alpha);
        add_norm(&lg.d_w_theta);
        add_norm(&lg.d_b_theta);
        add_norm(&lg.d_w_eta);
        add_norm(&lg.d_b_eta);
        if lg.has_mlp {
            add_norm(&lg.d_gate_proj);
            add_norm(&lg.d_up_proj);
            add_norm(&lg.d_down_proj);
        }
    }
    total.sqrt()
}
