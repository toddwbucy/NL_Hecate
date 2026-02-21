/// MONETA memory system — 2-layer MLP with l_p attentional bias.
///
/// First MIRAS variant to use MLP memory instead of d×d matrix.
/// y = W2 @ silu(W1 @ q) — nonlinear associative memory with higher capacity.
///
/// MIRAS knobs: MLP structure, l_p attentional bias, L2 retention (L_q planned), GD algorithm.
/// Source: MIRAS (2504.13173) Eqs 24-25, Table 2.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   pre_act = W1 @ k_t;  h = silu(pre_act)
///   prediction = W2 @ h;  error = prediction - v_t
///   lp_grad = p * sign(error) * |error|^(p-1)
///   grad_W2 = outer(lp_grad, h);  grad_W1 = outer(silu'(pre_act) * (W2^T @ lp_grad), k_t)
///   W1 = alpha_t * W1 - theta_t * (grad_W1 + lambda_2 * 2 * W1)
///   W2 = alpha_t * W2 - theta_t * (grad_W2 + lambda_2 * 2 * W2)
///   y_t = W2 @ silu(W1 @ q_t)
///
/// Backward: reverse token loop with accumulated d_W1, d_W2.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    silu_f32, silu_prime_f32, frobenius_dot_f32,
};
use crate::retention::{l2_apply_retention, l2_retention_gradient};
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};

// ── MONETA implementation ───────────────────────────────────────────

/// MONETA: 2-layer MLP memory with l_p loss and L2 retention.
/// Carries config params to avoid changing the MemoryRule trait signature.
pub struct Moneta {
    pub d_hidden: usize,
    pub lp_p: f32,
    pub lambda_2: f32,
    /// Sharpness parameter for smooth Sign approximation: tanh(a * x).
    /// Default: 10.0. Higher values → sharper transition, closer to true signum.
    /// At a=10: tanh(10 * 0.01) ≈ 0.1 (smooth near 0), tanh(10 * 0.5) ≈ 1.0 (saturated).
    /// See specs/algorithms/attentional_biases/01_l1_sign.md (MIRAS Remark 5).
    pub sign_sharpness: f32,
}

/// All intermediate values from a MONETA forward pass, needed for backward.
pub struct MonetaCache {
    pub seq_len: usize,
    pub d: usize,
    pub d_hidden: usize,
    /// W1 states for t=0..seq_len: [(seq_len+1) * d_hidden * d]
    pub w1_states: Vec<f32>,
    /// W2 states for t=0..seq_len: [(seq_len+1) * d * d_hidden]
    pub w2_states: Vec<f32>,
    /// Per-token projected keys: [seq_len, d]
    pub k_mem: Vec<f32>,
    /// Per-token projected values: [seq_len, d]
    pub v_mem: Vec<f32>,
    /// Per-token projected queries: [seq_len, d]
    pub q_mem: Vec<f32>,
    /// Per-token concatenated (k,v): [seq_len, 2*d]
    pub concat_kv: Vec<f32>,
    /// Pre-sigmoid alpha values: [seq_len]
    pub alpha_pre: Vec<f32>,
    /// Sigmoid alpha values: [seq_len]
    pub alpha: Vec<f32>,
    /// Pre-softplus theta values: [seq_len]
    pub theta_pre: Vec<f32>,
    /// Softplus theta values: [seq_len]
    pub theta: Vec<f32>,
    /// Pre-activation: W1 @ k_t, [seq_len, d_hidden]
    pub pre_act: Vec<f32>,
    /// Hidden activation: silu(pre_act), [seq_len, d_hidden]
    pub hidden: Vec<f32>,
    /// MLP prediction: W2 @ h, [seq_len, d]
    pub prediction: Vec<f32>,
    /// Error: prediction - v_t, [seq_len, d]
    pub error: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
    /// l_p exponent (cached for backward)
    pub lp_p: f32,
    /// L2 retention strength (cached for backward)
    pub lambda_2: f32,
    /// Sign sharpness (cached for backward)
    pub sign_sharpness: f32,
}

/// Compute l_p gradient: p * smooth_sign(e) * |e|^(p-1).
///
/// Uses tanh(a*e) as smooth Sign approximation and (e^2+eps)^{(p-1)/2}
/// as smooth absolute-power approximation (MIRAS §5.1 Remark 5).
/// Both are fully differentiable, enabling Wengert tape integration.
///
/// Fast-path dispatch (spec: 03_lp_dispatch.md):
/// - p=1 (L1): |e|^0 = 1 vanishes, only Sign needed → tanh(a*e)
/// - p=2 (L2): Sign(e)⊙|e| = e, no approximators → 2*e
/// - General p: both smooth approximators required
#[inline]
fn lp_grad(e: f32, p: f32, sign_sharpness: f32) -> f32 {
    if (p - 2.0).abs() < 1e-6 {
        // L2 fast-path: Sign(e) ⊙ |e|^1 = e, so p * e = 2 * e.
        // No tanh, no power — standard Delta rule gradient.
        2.0 * e
    } else if (p - 1.0).abs() < 1e-6 {
        // L1 fast-path: |e|^0 = 1, magnitude term vanishes.
        // Only Sign approximator survives.
        (sign_sharpness * e).tanh()
    } else {
        // General l_p: p * tanh(a*e) * (e^2 + eps)^{(p-1)/2}
        // Smooth power approximator: (e^2 + eps)^{(p-1)/2} ≈ |e|^{p-1}
        // Fully differentiable at e=0 (unlike |e|^{p-1}).
        let smooth_sign = (sign_sharpness * e).tanh();
        let power_approx = (e * e + 1e-6_f32).powf((p - 1.0) / 2.0);
        p * smooth_sign * power_approx
    }
}

impl MemoryRule for Moneta {
    type Cache = MonetaCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        crate::parallel::supported_strategies(crate::model::MemoryRuleKind::Moneta)
    }

    fn init(&self, d: usize) -> MemoryState {
        // For API compatibility — actual MONETA state is W1+W2, not a d×d matrix.
        // The init/write/read API isn't used by the full step() path.
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, _state: &mut MemoryState, _k: &[f32], _v: &[f32], _gates: &Gates) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation)
    }

    fn read(&self, _state: &MemoryState, _q: &[f32], _out: &mut [f32]) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation)
    }

    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, MonetaCache) {
        let dh = self.d_hidden;
        let p = self.lp_p;
        let l2 = self.lambda_2;
        let a = self.sign_sharpness;
        debug_assert_eq!(embedded.len(), seq_len * d);

        // Project embedded → k_mem, v_mem, q_mem via W^T
        let mut w_k_mem_t = vec![0.0f32; d * d];
        let mut w_v_mem_t = vec![0.0f32; d * d];
        let mut w_q_mem_t = vec![0.0f32; d * d];
        transpose_f32(&level_params.w_k_mem, &mut w_k_mem_t, d, d);
        transpose_f32(&level_params.w_v_mem, &mut w_v_mem_t, d, d);
        transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);

        let mut k_mem = vec![0.0f32; seq_len * d];
        let mut v_mem = vec![0.0f32; seq_len * d];
        let mut q_mem = vec![0.0f32; seq_len * d];
        matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
        matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
        matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

        // Allocate W1, W2 states — seed from initial_m if provided
        let w1_size = dh * d;
        let w2_size = d * dh;
        let mut w1_states = vec![0.0f32; (seq_len + 1) * w1_size];
        let mut w2_states = vec![0.0f32; (seq_len + 1) * w2_size];

        if let Some(m0) = initial_m {
            // CMS context memory format: W1_flat ++ W2_flat
            debug_assert_eq!(m0.len(), w1_size + w2_size);
            w1_states[..w1_size].copy_from_slice(&m0[..w1_size]);
            w2_states[..w2_size].copy_from_slice(&m0[w1_size..w1_size + w2_size]);
        } else {
            // Xavier-like init for W1 to break the zero saddle point.
            // With W1=0, W2=0: h=silu(0)=0, so grad_W2=outer(lp_g,0)=0 — MLP can never
            // escape zero. Small W1 init produces nonzero h, allowing W2 to learn.
            let scale = (2.0 / (d + dh) as f32).sqrt() * 0.1;
            for i in 0..w1_size {
                // Deterministic pseudo-random: mix index bits for variety
                let hash = ((i as u32).wrapping_mul(2654435761)) as f32 / u32::MAX as f32;
                w1_states[i] = scale * (hash - 0.5);
            }
            // W2 stays zero — gets nonzero gradient from first step via outer(lp_g, h≠0)
        }

        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut theta_pre = vec![0.0f32; seq_len];
        let mut theta = vec![0.0f32; seq_len];
        let mut pre_act_all = vec![0.0f32; seq_len * dh];
        let mut hidden_all = vec![0.0f32; seq_len * dh];
        let mut prediction_all = vec![0.0f32; seq_len * d];
        let mut error_all = vec![0.0f32; seq_len * d];
        let mut y = vec![0.0f32; seq_len * d];

        for t in 0..seq_len {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];

            // Concatenate (k_t, v_t)
            let c_base = t * 2 * d;
            concat_kv[c_base..c_base + d].copy_from_slice(k_t);
            concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
            let concat_t = &concat_kv[c_base..c_base + 2 * d];

            // alpha_t = sigmoid(concat @ w_alpha + b_alpha)
            let mut alpha_pre_t = level_params.b_alpha[0];
            for i in 0..(2 * d) {
                alpha_pre_t += concat_t[i] * level_params.w_alpha[i];
            }
            alpha_pre[t] = alpha_pre_t;
            alpha[t] = sigmoid_f32(alpha_pre_t);

            // theta_t = softplus(concat @ w_theta + b_theta)
            let mut theta_pre_t = level_params.b_theta[0];
            for i in 0..(2 * d) {
                theta_pre_t += concat_t[i] * level_params.w_theta[i];
            }
            theta_pre[t] = theta_pre_t;
            theta[t] = softplus_f32(theta_pre_t);

            // MLP forward: pre_act = W1 @ k_t, h = silu(pre_act)
            // Use split_at_mut to get non-overlapping borrows for current (immutable)
            // and next (mutable) slices of the W1/W2 state arrays.
            let (w1_left, w1_right) = w1_states.split_at_mut((t + 1) * w1_size);
            let w1_t = &w1_left[t * w1_size..];
            let w1_next = &mut w1_right[..w1_size];

            let (w2_left, w2_right) = w2_states.split_at_mut((t + 1) * w2_size);
            let w2_t = &w2_left[t * w2_size..];
            let w2_next = &mut w2_right[..w2_size];

            let pa_base = t * dh;
            // W1[dh, d] @ k_t[d, 1] = pre_act[dh, 1]
            matmul_f32(w1_t, k_t, &mut pre_act_all[pa_base..pa_base + dh], dh, d, 1);
            for i in 0..dh {
                hidden_all[pa_base + i] = silu_f32(pre_act_all[pa_base + i]);
            }
            let h_t = &hidden_all[pa_base..pa_base + dh];

            // prediction = W2 @ h
            let pred_base = t * d;
            matmul_f32(w2_t, h_t, &mut prediction_all[pred_base..pred_base + d], d, dh, 1);

            // error = prediction - v_t
            for i in 0..d {
                error_all[pred_base + i] = prediction_all[pred_base + i] - v_t[i];
            }

            // l_p gradient on error
            let mut lp_g = vec![0.0f32; d];
            for i in 0..d {
                lp_g[i] = lp_grad(error_all[pred_base + i], p, a);
            }

            // grad_W2 = outer(lp_grad, h) → [d, dh]
            let mut grad_w2 = vec![0.0f32; w2_size];
            for i in 0..d {
                for j in 0..dh {
                    grad_w2[i * dh + j] = lp_g[i] * h_t[j];
                }
            }

            // grad_h = W2^T @ lp_grad → [dh]
            let mut grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += w2_t[j * dh + i] * lp_g[j];
                }
                grad_h[i] = sum;
            }

            // grad_pre = grad_h * silu'(pre_act) → [dh]
            let mut grad_pre = vec![0.0f32; dh];
            for i in 0..dh {
                grad_pre[i] = grad_h[i] * silu_prime_f32(pre_act_all[pa_base + i]);
            }

            // grad_W1 = outer(grad_pre, k_t) → [dh, d]
            let mut grad_w1 = vec![0.0f32; w1_size];
            for i in 0..dh {
                for j in 0..d {
                    grad_w1[i * d + j] = grad_pre[i] * k_t[j];
                }
            }

            // Retention: L2 global penalty gradient + multiplicative decay
            let ret_grad_w1 = l2_retention_gradient(w1_t, l2);
            let ret_grad_w2 = l2_retention_gradient(w2_t, l2);

            // Update: W1 = alpha_t * W1 - theta_t * (grad_W1 + ret_grad_W1)
            //         W2 = alpha_t * W2 - theta_t * (grad_W2 + ret_grad_W2)
            let alpha_t = alpha[t];
            let theta_t = theta[t];
            w1_next.copy_from_slice(w1_t);
            l2_apply_retention(w1_next, alpha_t);
            for i in 0..w1_size {
                w1_next[i] -= theta_t * (grad_w1[i] + ret_grad_w1[i]);
            }
            w2_next.copy_from_slice(w2_t);
            l2_apply_retention(w2_next, alpha_t);
            for i in 0..w2_size {
                w2_next[i] -= theta_t * (grad_w2[i] + ret_grad_w2[i]);
            }

            // Read: y_t = W2_next @ silu(W1_next @ q_t)
            let mut q_pre = vec![0.0f32; dh];
            matmul_f32(w1_next, q_t, &mut q_pre, dh, d, 1);
            let mut q_hidden = vec![0.0f32; dh];
            for i in 0..dh {
                q_hidden[i] = silu_f32(q_pre[i]);
            }
            matmul_f32(w2_next, &q_hidden, &mut y[t * d..(t + 1) * d], d, dh, 1);
        }

        let cache = MonetaCache {
            seq_len, d, d_hidden: dh,
            w1_states, w2_states,
            k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta,
            pre_act: pre_act_all, hidden: hidden_all,
            prediction: prediction_all, error: error_all,
            y: y.clone(),
            lp_p: p, lambda_2: l2, sign_sharpness: a,
        };

        (y, cache)
    }

    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &MonetaCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        let dh = cache.d_hidden;
        let p = cache.lp_p;
        let a = cache.sign_sharpness;
        let l2 = cache.lambda_2;
        let l2_2 = l2 * 2.0;
        let w1_size = dh * d;
        let w2_size = d * dh;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like(d);
        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // Accumulated gradients on W1 and W2 (the MLP "memory state")
        let mut d_w1 = vec![0.0f32; w1_size];
        let mut d_w2 = vec![0.0f32; w2_size];

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let _v_t = &cache.v_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let theta_pre_t = cache.theta_pre[t];
            let pa_base = t * dh;
            let h_t = &cache.hidden[pa_base..pa_base + dh];
            let w1_t = &cache.w1_states[t * w1_size..(t + 1) * w1_size];
            let w2_t = &cache.w2_states[t * w2_size..(t + 1) * w2_size];
            let w1_next = &cache.w1_states[(t + 1) * w1_size..(t + 2) * w1_size];
            let w2_next = &cache.w2_states[(t + 1) * w2_size..(t + 2) * w2_size];

            // ── y_t = W2_next @ silu(W1_next @ q_t) backward ──
            let d_y_t = &d_y[t * d..(t + 1) * d];

            // Forward cache for read: q_pre = W1_next @ q_t, q_hidden = silu(q_pre)
            let mut q_pre = vec![0.0f32; dh];
            matmul_f32(w1_next, q_t, &mut q_pre, dh, d, 1);
            let mut q_hidden = vec![0.0f32; dh];
            for i in 0..dh {
                q_hidden[i] = silu_f32(q_pre[i]);
            }

            // d_W2_next += outer(d_y_t, q_hidden)
            for i in 0..d {
                for j in 0..dh {
                    d_w2[i * dh + j] += d_y_t[i] * q_hidden[j];
                }
            }

            // d_q_hidden = W2_next^T @ d_y_t
            let mut d_q_hidden = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += w2_next[j * dh + i] * d_y_t[j];
                }
                d_q_hidden[i] = sum;
            }

            // d_q_pre = d_q_hidden * silu'(q_pre)
            let mut d_q_pre = vec![0.0f32; dh];
            for i in 0..dh {
                d_q_pre[i] = d_q_hidden[i] * silu_prime_f32(q_pre[i]);
            }

            // d_W1_next += outer(d_q_pre, q_t)
            for i in 0..dh {
                for j in 0..d {
                    d_w1[i * d + j] += d_q_pre[i] * q_t[j];
                }
            }

            // d_q_t = W1_next^T @ d_q_pre
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..dh {
                    sum += w1_next[i * d + j] * d_q_pre[i];
                }
                d_q_mem[t * d + j] = sum;
            }

            // ── W_next = alpha * W_t - theta * (grad_W + l2*2*W_t) backward ──
            // d_W1_next is our current d_w1, d_W2_next is our current d_w2
            //
            // d_alpha from W1: sum_i(d_W1_next_i * W1_t_i)
            // d_alpha from W2: sum_i(d_W2_next_i * W2_t_i)
            let d_alpha_w1 = frobenius_dot_f32(&d_w1, w1_t);
            let d_alpha_w2 = frobenius_dot_f32(&d_w2, w2_t);
            let d_alpha_scalar = d_alpha_w1 + d_alpha_w2;

            // d_theta from W1: sum_i(d_W1_next_i * -(grad_W1_i + l2_2*W1_t_i))
            // d_theta from W2: sum_i(d_W2_next_i * -(grad_W2_i + l2_2*W2_t_i))
            // We need grad_W1 and grad_W2 from the forward pass — recompute them.
            let pred_base = t * d;
            let mut lp_g = vec![0.0f32; d];
            for i in 0..d {
                lp_g[i] = lp_grad(cache.error[pred_base + i], p, a);
            }

            // Recompute grad_W2 = outer(lp_g, h_t)
            let mut grad_w2 = vec![0.0f32; w2_size];
            for i in 0..d {
                for j in 0..dh {
                    grad_w2[i * dh + j] = lp_g[i] * h_t[j];
                }
            }
            // Recompute grad_h, grad_pre, grad_W1
            let mut grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += w2_t[j * dh + i] * lp_g[j];
                }
                grad_h[i] = sum;
            }
            let mut grad_pre = vec![0.0f32; dh];
            for i in 0..dh {
                grad_pre[i] = grad_h[i] * silu_prime_f32(cache.pre_act[pa_base + i]);
            }
            let mut grad_w1 = vec![0.0f32; w1_size];
            for i in 0..dh {
                for j in 0..d {
                    grad_w1[i * d + j] = grad_pre[i] * k_t[j];
                }
            }

            // d_theta_scalar
            let mut d_theta_scalar = 0.0f32;
            for i in 0..w1_size {
                d_theta_scalar -= d_w1[i] * (grad_w1[i] + l2_2 * w1_t[i]);
            }
            for i in 0..w2_size {
                d_theta_scalar -= d_w2[i] * (grad_w2[i] + l2_2 * w2_t[i]);
            }

            // ── Backprop d_W_next through the update to get d_W_t and d_grad ──
            // d_grad_W1 = -theta * d_W1_next
            // d_grad_W2 = -theta * d_W2_next
            let mut d_grad_w1 = vec![0.0f32; w1_size];
            let mut d_grad_w2 = vec![0.0f32; w2_size];
            for i in 0..w1_size {
                d_grad_w1[i] = -theta_t * d_w1[i];
            }
            for i in 0..w2_size {
                d_grad_w2[i] = -theta_t * d_w2[i];
            }

            // d_W1_t = alpha * d_W1_next - theta * l2_2 * d_W1_next
            //        = (alpha - theta * l2_2) * d_W1_next
            // But we also need d_W_t += d_theta_grad contribution through W_t in retention term
            // Actually: d_W1_t from update eq = alpha_t * d_W1_next + (-theta_t * l2_2) * d_W1_next
            // = (alpha_t - theta_t * l2_2) * d_W1_next
            let coeff = alpha_t - theta_t * l2_2;
            let mut d_w1_prev = vec![0.0f32; w1_size];
            let mut d_w2_prev = vec![0.0f32; w2_size];
            for i in 0..w1_size {
                d_w1_prev[i] = coeff * d_w1[i];
            }
            for i in 0..w2_size {
                d_w2_prev[i] = coeff * d_w2[i];
            }

            // ── Backprop through MLP gradient computation ──
            // grad_W2 = outer(lp_g, h) → d_lp_g from d_grad_W2, d_h from d_grad_W2
            // d_lp_g[i] = sum_j(d_grad_W2[i,j] * h[j])
            let mut d_lp_g = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..dh {
                    sum += d_grad_w2[i * dh + j] * h_t[j];
                }
                d_lp_g[i] = sum;
            }
            // d_h from grad_W2: d_h[j] = sum_i(d_grad_W2[i,j] * lp_g[i])
            let mut d_h_from_gw2 = vec![0.0f32; dh];
            for j in 0..dh {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_grad_w2[i * dh + j] * lp_g[i];
                }
                d_h_from_gw2[j] = sum;
            }

            // grad_h = W2^T @ lp_g → d_lp_g from d_grad_h contribution, d_W2_t from d_grad_h
            // d_grad_h = silu'(pre_act) * d_grad_pre → but we need d_grad_pre first
            // Actually the chain is:
            //   grad_W1 = outer(grad_pre, k_t)
            //   grad_pre = grad_h * silu'(pre_act)
            //   grad_h = W2^T @ lp_g
            //
            // d_grad_pre[i] = sum_j(d_grad_W1[i,j] * k_t[j])
            let mut d_grad_pre = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_grad_w1[i * d + j] * k_t[j];
                }
                d_grad_pre[i] = sum;
            }
            // d_k from grad_W1: d_k[j] += sum_i(d_grad_W1[i,j] * grad_pre[i])
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..dh {
                    sum += d_grad_w1[i * d + j] * grad_pre[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // grad_pre = grad_h * silu'(pre_act) backward
            // d_grad_h[i] = d_grad_pre[i] * silu'(pre_act[i])
            let mut d_grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                d_grad_h[i] = d_grad_pre[i] * silu_prime_f32(cache.pre_act[pa_base + i]);
            }
            // Note: we skip d_pre_act from silu'' (second derivative) — this is a first-order
            // backward pass through the update rule. The silu_prime is a coefficient, not
            // a function of pre_act that we differentiate again.

            // d_h_total = d_h_from_gw2 + contribution from grad_h path
            // grad_h = W2^T @ lp_g backward:
            //   d_lp_g[j] += sum_i(W2[j,i] * d_grad_h[i]) ... wait, W2^T @ lp_g:
            //   grad_h[i] = sum_j(W2[j, i] * lp_g[j])  (W2 is [d, dh])
            //   d_lp_g[j] += sum_i(d_grad_h[i] * W2[j, i]) = sum_i(W2[j,i] * d_grad_h[i])
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..dh {
                    sum += w2_t[j * dh + i] * d_grad_h[i];
                }
                d_lp_g[j] += sum;
            }
            // d_W2_t from grad_h: d_W2_t[j, i] += lp_g[j] * d_grad_h[i]
            for j in 0..d {
                for i in 0..dh {
                    d_w2_prev[j * dh + i] += lp_g[j] * d_grad_h[i];
                }
            }

            // ── lp_grad backward: d/de [p * Sign(e) ⊙ |e|^{p-1}] ──
            let mut d_err = vec![0.0f32; d];
            for i in 0..d {
                d_err[i] = d_lp_g[i] * lp_grad_deriv(cache.error[pred_base + i], p, a);
            }

            // error = prediction - v_t backward
            // d_prediction = d_err, d_v_t -= d_err
            for i in 0..d {
                d_v_mem[t * d + i] -= d_err[i];
            }

            // prediction = W2_t @ h backward (through d_err)
            // d_W2_t[i,j] += d_err[i] * h_t[j]
            for i in 0..d {
                for j in 0..dh {
                    d_w2_prev[i * dh + j] += d_err[i] * h_t[j];
                }
            }
            // d_h_from_pred[j] = sum_i(W2_t[i,j] * d_err[i])
            let mut d_h_total = d_h_from_gw2;
            for j in 0..dh {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += w2_t[i * dh + j] * d_err[i];
                }
                d_h_total[j] += sum;
            }

            // h = silu(pre_act) backward: d_pre_act = d_h_total * silu'(pre_act)
            let mut d_pre_act = vec![0.0f32; dh];
            for i in 0..dh {
                d_pre_act[i] = d_h_total[i] * silu_prime_f32(cache.pre_act[pa_base + i]);
            }

            // pre_act = W1_t @ k_t backward
            // d_W1_t[i,j] += d_pre_act[i] * k_t[j]
            for i in 0..dh {
                for j in 0..d {
                    d_w1_prev[i * d + j] += d_pre_act[i] * k_t[j];
                }
            }
            // d_k_t[j] += sum_i(W1_t[i,j] * d_pre_act[i])
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..dh {
                    sum += w1_t[i * d + j] * d_pre_act[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_scalar * sig_deriv;

            // ── Gate backward: theta_t = softplus(theta_pre_t) ──
            let softplus_deriv = sigmoid_f32(theta_pre_t);
            let d_theta_pre = d_theta_scalar * softplus_deriv;

            // ── w_alpha, b_alpha gradient ──
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;

            // ── w_theta, b_theta gradient ──
            for i in 0..(2 * d) {
                grads.w_theta[i] += d_theta_pre * concat_t[i];
            }
            grads.b_theta[0] += d_theta_pre;

            // ── concat backward → d_k_mem, d_v_mem ──
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i]
                                    + d_theta_pre * level_params.w_theta[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i]
                                    + d_theta_pre * level_params.w_theta[d + i];
            }

            // Swap: d_w1_prev becomes d_w1 for next (earlier) token
            d_w1 = d_w1_prev;
            d_w2 = d_w2_prev;
        }

        // ── Projection backward: k_mem = embedded @ W_K_mem^T ──
        let mut d_embedded = vec![0.0f32; s * d];

        let mut d_k_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_k_mem, &mut d_k_mem_t, s, d);
        matmul_f32(&d_k_mem_t, embedded, &mut grads.w_k_mem, d, s, d);

        let mut d_v_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_v_mem, &mut d_v_mem_t, s, d);
        matmul_f32(&d_v_mem_t, embedded, &mut grads.w_v_mem, d, s, d);

        let mut d_q_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_q_mem, &mut d_q_mem_t, s, d);
        matmul_f32(&d_q_mem_t, embedded, &mut grads.w_q_mem, d, s, d);

        crate::tensor::matmul_acc_f32(&d_k_mem, &level_params.w_k_mem, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_v_mem, &level_params.w_v_mem, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_q_mem, &level_params.w_q_mem, &mut d_embedded, s, d, d);

        (grads, d_embedded)
    }
}

/// Derivative of lp_grad w.r.t. error: d/de [p * Sign(e) ⊙ |e|^{p-1}].
///
/// Used by both `step_backward` (MONETA MLP) and `lp_bias_gradient_backward`
/// (standalone dispatch) to avoid duplicating the per-element VJP logic.
#[inline]
fn lp_grad_deriv(e: f32, p: f32, sign_sharpness: f32) -> f32 {
    let a = sign_sharpness;
    if (p - 2.0).abs() < 1e-6 {
        2.0
    } else if (p - 1.0).abs() < 1e-6 {
        let tanh_ae = (a * e).tanh();
        a * (1.0 - tanh_ae * tanh_ae)
    } else {
        let tanh_ae = (a * e).tanh();
        let sech2 = 1.0 - tanh_ae * tanh_ae;
        let e2_eps = e * e + 1e-6_f32;
        let power = e2_eps.powf((p - 1.0) / 2.0);
        let d_power = (p - 1.0) * e * e2_eps.powf((p - 3.0) / 2.0);
        p * (a * sech2 * power + tanh_ae * d_power)
    }
}

// ── Standalone l_p dispatch (linear-memory rules) ────────────────────

/// Compute the l_p gradient vector: p * Sign(e) ⊙ |e|^{p-1} per element.
///
/// Standalone dispatch for matrix-memory rules (Delta, Titans, Hebbian, etc.)
/// where error = W @ k - v. The caller computes the error vector; this function
/// applies the element-wise l_p gradient with fast-path dispatch.
///
/// Returns a d-dimensional gradient vector. The caller forms the outer product
/// `grad_vec @ k^T` to get the d×d weight gradient.
///
/// Source: MIRAS §5.1 Eq 11, specs/algorithms/attentional_biases/03_lp_dispatch.md
pub fn lp_bias_gradient(error: &[f32], p: f32, sign_sharpness: f32) -> Vec<f32> {
    let d = error.len();
    let mut grad = vec![0.0f32; d];
    for i in 0..d {
        grad[i] = lp_grad(error[i], p, sign_sharpness);
    }
    grad
}

/// Backward through the l_p gradient vector (VJP).
///
/// Given upstream gradient d_lp (d-dimensional, flowing back through the l_p
/// gradient computation), returns d_error (d-dimensional gradient w.r.t. the
/// error vector).
///
/// Fast-path dispatch mirrors the forward:
/// - p=2: d/de(2*e) = 2
/// - p=1: d/de(tanh(a*e)) = a * sech^2(a*e)
/// - General: product rule on p * tanh(a*e) * (e^2+eps)^{(p-1)/2}
///
/// Source: specs/algorithms/attentional_biases/03_lp_dispatch.md (VJP section)
pub fn lp_bias_gradient_backward(
    d_lp: &[f32],
    error: &[f32],
    p: f32,
    sign_sharpness: f32,
) -> Vec<f32> {
    assert_eq!(d_lp.len(), error.len(),
        "lp_bias_gradient_backward: d_lp.len() ({}) != error.len() ({})", d_lp.len(), error.len());
    let d = error.len();
    let mut d_error = vec![0.0f32; d];
    for i in 0..d {
        d_error[i] = d_lp[i] * lp_grad_deriv(error[i], p, sign_sharpness);
    }
    d_error
}

// ── AttentionalBias dispatch (all matrix-memory rules) ──────────────

/// Apply attentional bias transformation to raw error vector.
///
/// Maps AttentionalBias enum → element-wise gradient transformation:
/// - L2: identity (raw error, factor of 2 absorbed into learning rate)
/// - L1: tanh(a * error) per element (smooth Sign approximation)
/// - Lp(p): p * tanh(a*e) * (e²+eps)^{(p-1)/2} per element
///
/// The caller forms outer(result, k) to get the d×d weight gradient.
/// Source: MIRAS §5.1 Eq 11, specs/algorithms/attentional_biases/03_lp_dispatch.md
pub fn apply_attentional_bias(
    error: &[f32],
    bias: crate::model::AttentionalBias,
    sign_sharpness: f32,
) -> Vec<f32> {
    use crate::model::AttentionalBias;
    match bias {
        AttentionalBias::L2 => error.to_vec(),
        AttentionalBias::L1 => lp_bias_gradient(error, 1.0, sign_sharpness),
        AttentionalBias::Lp(p) => lp_bias_gradient(error, p, sign_sharpness),
        AttentionalBias::KL => {
            panic!("KL attentional bias not yet implemented — see specs/algorithms/attentional_biases/02_kl_objective.md. \
                    Call validate_bias() at config load to catch this early.")
        }
        AttentionalBias::Huber => {
            panic!("Huber attentional bias not yet implemented — see specs/algorithms/attentional_biases/02_kl_objective.md. \
                    Call validate_bias() at config load to catch this early.")
        }
    }
}

/// Backward (VJP) through attentional bias transformation.
///
/// Given upstream gradient flowing back through the bias transform,
/// returns gradient w.r.t. the raw error vector.
/// - L2: identity (d/de(e) = 1)
/// - L1: a * sech²(a*e) per element
/// - Lp(p): product rule on tanh and power terms
pub fn apply_attentional_bias_backward(
    d_biased: &[f32],
    error: &[f32],
    bias: crate::model::AttentionalBias,
    sign_sharpness: f32,
) -> Vec<f32> {
    use crate::model::AttentionalBias;
    match bias {
        AttentionalBias::L2 => d_biased.to_vec(),
        AttentionalBias::L1 => lp_bias_gradient_backward(d_biased, error, 1.0, sign_sharpness),
        AttentionalBias::Lp(p) => lp_bias_gradient_backward(d_biased, error, p, sign_sharpness),
        AttentionalBias::KL => {
            panic!("KL attentional bias not yet implemented — see specs/algorithms/attentional_biases/02_kl_objective.md. \
                    Call validate_bias() at config load to catch this early.")
        }
        AttentionalBias::Huber => {
            panic!("Huber attentional bias not yet implemented — see specs/algorithms/attentional_biases/02_kl_objective.md. \
                    Call validate_bias() at config load to catch this early.")
        }
    }
}

/// Normalize AttentionalBias: Lp(2.0)→L2, Lp(1.0)→L1 to avoid ambiguity.
/// L2 returns identity while Lp(2.0) returns 2*error — semantically different.
/// Call this at config load/construction to prevent downstream confusion.
pub fn normalize_bias(bias: crate::model::AttentionalBias) -> crate::model::AttentionalBias {
    use crate::model::AttentionalBias;
    match bias {
        AttentionalBias::Lp(p) if (p - 2.0).abs() < 1e-6 => AttentionalBias::L2,
        AttentionalBias::Lp(p) if (p - 1.0).abs() < 1e-6 => AttentionalBias::L1,
        other => other,
    }
}

/// Validate that the bias is supported by apply_attentional_bias.
/// KL and Huber are separate specs (02_kl_objective.md) and not yet implemented.
pub fn validate_bias(bias: crate::model::AttentionalBias) -> Result<(), String> {
    use crate::model::AttentionalBias;
    match bias {
        AttentionalBias::L2 | AttentionalBias::L1 | AttentionalBias::Lp(_) => Ok(()),
        AttentionalBias::KL => Err("KL attentional bias not yet implemented (see 02_kl_objective.md)".into()),
        AttentionalBias::Huber => Err("Huber attentional bias not yet implemented (see 02_kl_objective.md)".into()),
    }
}

/// Encode AttentionalBias as a single f32 for tape metadata.
/// L2 → 2.0, L1 → 1.0, Lp(p) → p, KL → -1.0, Huber → -2.0.
/// Panics on Lp(1.0) or Lp(2.0) — call normalize_bias() first.
pub fn bias_to_f32(bias: crate::model::AttentionalBias) -> f32 {
    use crate::model::AttentionalBias;
    match bias {
        AttentionalBias::L2 => 2.0,
        AttentionalBias::L1 => 1.0,
        AttentionalBias::Lp(p) => {
            assert!((p - 2.0).abs() >= 1e-6 && (p - 1.0).abs() >= 1e-6,
                "Lp({p}) collides with L2/L1 encoding — call normalize_bias() first");
            p
        }
        AttentionalBias::KL => -1.0,
        AttentionalBias::Huber => -2.0,
    }
}

/// Decode a single f32 back to AttentionalBias (inverse of bias_to_f32).
/// L2 returns identity (factor of 2 absorbed into learning rate), while
/// Lp(2.0) returns 2*error — so we must distinguish them.
pub fn f32_to_bias(v: f32) -> crate::model::AttentionalBias {
    use crate::model::AttentionalBias;
    if (v - 2.0).abs() < 1e-6 { AttentionalBias::L2 }
    else if (v - 1.0).abs() < 1e-6 { AttentionalBias::L1 }
    else if (v - (-1.0)).abs() < 1e-6 { AttentionalBias::KL }
    else if (v - (-2.0)).abs() < 1e-6 { AttentionalBias::Huber }
    else { AttentionalBias::Lp(v) }
}

// ── Read-only functions (for frozen CMS levels) ─────────────────────

/// Forward pass for a frozen MONETA level: y_t = W2 @ silu(W1 @ q_t).
/// W1, W2 are frozen (not updated). Returns (y, q_mem).
pub fn moneta_read_only(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    frozen_m: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let w1_size = d_hidden * d;
    let w2_size = d * d_hidden;
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(frozen_m.len(), w1_size + w2_size);

    let w1 = &frozen_m[..w1_size];
    let w2 = &frozen_m[w1_size..w1_size + w2_size];

    // Project embedded → q_mem via W_Q_mem^T
    let mut w_q_mem_t = vec![0.0f32; d * d];
    transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

    // y_t = W2 @ silu(W1 @ q_t)
    let mut y = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let mut pre_act = vec![0.0f32; d_hidden];
        matmul_f32(w1, q_t, &mut pre_act, d_hidden, d, 1);
        let mut h = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            h[i] = silu_f32(pre_act[i]);
        }
        matmul_f32(w2, &h, &mut y[t * d..(t + 1) * d], d, d_hidden, 1);
    }

    (y, q_mem)
}

/// Backward pass for a frozen MONETA level.
/// Only d_embedded (through W_Q_mem) flows back, plus d_W_Q_mem gradients.
pub fn moneta_read_only_backward(
    level_params: &MemoryLevelParams,
    frozen_m: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
) -> (MemoryLevelParams, Vec<f32>) {
    let w1_size = d_hidden * d;
    let w2_size = d * d_hidden;
    debug_assert_eq!(d_y.len(), seq_len * d);

    let w1 = &frozen_m[..w1_size];
    let w2 = &frozen_m[w1_size..w1_size + w2_size];

    let mut grads = MemoryLevelParams::zeros_like(d);

    // y_t = W2 @ silu(W1 @ q_t) → d_q_t via chain rule
    let mut d_q_mem = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];

        // Recompute forward intermediates
        let mut pre_act = vec![0.0f32; d_hidden];
        matmul_f32(w1, q_t, &mut pre_act, d_hidden, d, 1);
        let mut h = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            h[i] = silu_f32(pre_act[i]);
        }

        // d_h = W2^T @ d_y_t
        let mut d_h = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += w2[j * d_hidden + i] * d_y_t[j];
            }
            d_h[i] = sum;
        }

        // d_pre_act = d_h * silu'(pre_act)
        let mut d_pre_act = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            d_pre_act[i] = d_h[i] * silu_prime_f32(pre_act[i]);
        }

        // d_q_t = W1^T @ d_pre_act
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d_hidden {
                sum += w1[i * d + j] * d_pre_act[i];
            }
            d_q_mem[t * d + j] = sum;
        }
    }

    // q_mem = embedded @ W_Q_mem^T → d_W_Q_mem, d_embedded
    let mut d_q_mem_t = vec![0.0f32; d * seq_len];
    transpose_f32(&d_q_mem, &mut d_q_mem_t, seq_len, d);
    matmul_f32(&d_q_mem_t, embedded, &mut grads.w_q_mem, d, seq_len, d);

    let mut d_embedded = vec![0.0f32; seq_len * d];
    crate::tensor::matmul_acc_f32(&d_q_mem, &level_params.w_q_mem, &mut d_embedded, seq_len, d, d);

    (grads, d_embedded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

    fn test_config() -> MAGConfig {
        MAGConfig::moneta_test_config()
    }

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    fn make_moneta(cfg: &MAGConfig) -> Moneta {
        Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2, sign_sharpness: cfg.sign_sharpness }
    }

    #[test]
    fn test_moneta_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_moneta(&cfg);
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_moneta_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_moneta(&cfg);
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "MONETA forward should be deterministic");
    }

    #[test]
    fn test_moneta_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_moneta(&cfg);
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let dh = cfg.d_hidden;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.w1_states.len(), (s + 1) * dh * d);
        assert_eq!(cache.w2_states.len(), (s + 1) * d * dh);
        assert_eq!(cache.pre_act.len(), s * dh);
        assert_eq!(cache.hidden.len(), s * dh);
    }

    #[test]
    fn test_moneta_forward_mlp_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_moneta(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let dh = cfg.d_hidden;
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // W1 should start small (Xavier init) and evolve to different values
        let w1_0_norm: f32 = cache.w1_states[0..dh * d].iter().map(|x| x * x).sum::<f32>().sqrt();
        let w1_t_norm: f32 = cache.w1_states[s * dh * d..(s + 1) * dh * d].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(w1_0_norm > 1e-6, "W1_0 should be initialized (Xavier), norm={w1_0_norm}");
        assert!((w1_t_norm - w1_0_norm).abs() > 1e-8, "W1_T should have evolved from init, W1_0={w1_0_norm:.4e}, W1_T={w1_t_norm:.4e}");
    }

    #[test]
    fn test_moneta_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_moneta(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            let th = cache.theta[t];
            assert!(th >= 0.0, "theta[{t}]={th} should be non-negative");
        }
    }

    #[test]
    fn test_moneta_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_moneta(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta),
        ] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "d_embedded[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_moneta_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_moneta(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "grad_{name} is all zeros (max_abs={max_abs})");
        }
        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "d_embedded is all zeros");
    }

    #[test]
    fn test_moneta_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_moneta(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        assert_eq!(grads.w_k_mem.len(), d * d);
        assert_eq!(grads.w_v_mem.len(), d * d);
        assert_eq!(grads.w_q_mem.len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
        assert_eq!(grads.w_theta.len(), 2 * d);
        assert_eq!(grads.b_theta.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }

    // ── Trait API tests ──────────────────────────────────────────────

    #[test]
    fn test_moneta_init() {
        let rule = Moneta { d_hidden: 4, lp_p: 2.0, lambda_2: 0.01, sign_sharpness: 10.0 };
        let state = rule.init(8);
        assert_eq!(state.m.len(), 64);
        assert_eq!(state.d, 8);
        assert!(state.m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_moneta_level_and_parallelization() {
        let rule = Moneta { d_hidden: 4, lp_p: 2.0, lambda_2: 0.01, sign_sharpness: 10.0 };
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"chunkwise_gd"));
        assert!(strategies.contains(&"tnt"));
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_moneta_read_only_zero_memory() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let frozen_m = vec![0.0f32; dh * d + d * dh];
        let (y, _q_mem) = moneta_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        // Zero W1/W2 → zero pre_act → silu(0)=0 → zero output
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_moneta_read_only_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        // Initialize W1/W2 with some non-zero values
        let mut rng = SimpleRng::new(77);
        let mut frozen_m = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut frozen_m, 0.1);
        let (y, _q_mem) = moneta_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        let y_norm: f32 = y.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(y_norm > 1e-6, "Non-zero W1/W2 should produce non-zero output");
    }

    #[test]
    fn test_moneta_read_only_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let mut rng = SimpleRng::new(77);
        let mut frozen_m = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut frozen_m, 0.1);
        let (_, q_mem) = moneta_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = moneta_read_only_backward(
            &params.levels[0], &frozen_m, &q_mem, &d_y, &embedded, s, d, dh,
        );
        for &v in grads.w_q_mem.iter() {
            assert!(v.is_finite(), "read_only_backward grad not finite");
        }
        for &v in d_emb.iter() {
            assert!(v.is_finite(), "read_only_backward d_emb not finite");
        }
    }

    // ── Initial memory seeding test ──────────────────────────────────

    #[test]
    fn test_moneta_initial_m_seeding() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let rule = make_moneta(&cfg);

        // Without initial_m
        let (y1, _) = rule.step(&params.levels[0], &embedded, s, d, None);
        // With non-zero initial_m
        let mut rng = SimpleRng::new(77);
        let mut m0 = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut m0, 0.1);
        let (y2, _) = rule.step(&params.levels[0], &embedded, s, d, Some(m0));

        // Outputs should differ with different initial memory
        assert_ne!(y1, y2, "Initial memory seeding should change output");
    }

    // ── l_p dispatch tests ──────────────────────────────────────────

    #[test]
    fn test_lp_grad_l2_fast_path() {
        // p=2: lp_grad should return 2*e exactly
        let errors = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        for &e in &errors {
            let g = lp_grad(e, 2.0, 10.0);
            let expected = 2.0 * e;
            assert!((g - expected).abs() < 1e-10,
                "p=2 fast-path: lp_grad({e}) = {g}, expected {expected}");
        }
    }

    #[test]
    fn test_lp_grad_l1_fast_path() {
        // p=1: lp_grad should return tanh(a*e)
        let a = 10.0;
        let errors = [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0];
        for &e in &errors {
            let g = lp_grad(e, 1.0, a);
            let expected = (a * e).tanh();
            assert!((g - expected).abs() < 1e-10,
                "p=1 fast-path: lp_grad({e}) = {g}, expected {expected}");
        }
    }

    #[test]
    fn test_lp_grad_general_p3() {
        // p=3: gradient should be 3 * tanh(a*e) * (e^2+eps)^1.0
        let a = 10.0;
        let e = 0.5;
        let g = lp_grad(e, 3.0, a);
        let expected = 3.0 * (a * e).tanh() * (e * e + 1e-6_f32).powf(1.0);
        assert!((g - expected).abs() < 1e-6,
            "p=3: lp_grad({e}) = {g}, expected {expected}");
    }

    #[test]
    fn test_lp_grad_smooth_at_zero() {
        // All p values should produce finite, non-NaN results at e=0
        for p in [1.0, 1.5, 2.0, 3.0, 4.0] {
            let g = lp_grad(0.0, p, 10.0);
            assert!(g.is_finite(), "lp_grad(0, p={p}) should be finite, got {g}");
        }
    }

    #[test]
    fn test_lp_bias_gradient_standalone() {
        // Verify standalone dispatch matches per-element lp_grad
        let error = vec![-0.5, 0.0, 0.3, 1.0];
        let p = 3.0;
        let a = 10.0;
        let grad = lp_bias_gradient(&error, p, a);
        assert_eq!(grad.len(), 4);
        for i in 0..4 {
            let expected = lp_grad(error[i], p, a);
            assert!((grad[i] - expected).abs() < 1e-10,
                "lp_bias_gradient[{i}] = {}, expected {expected}", grad[i]);
        }
    }

    #[test]
    fn test_lp_bias_gradient_backward_l2() {
        // p=2: d/de(2*e) = 2, so d_error = 2 * d_lp
        let d_lp = vec![1.0, -0.5, 0.3];
        let error = vec![0.1, -0.2, 0.5];
        let d_err = lp_bias_gradient_backward(&d_lp, &error, 2.0, 10.0);
        for i in 0..3 {
            let expected = d_lp[i] * 2.0;
            assert!((d_err[i] - expected).abs() < 1e-10,
                "p=2 backward: d_err[{i}] = {}, expected {expected}", d_err[i]);
        }
    }

    #[test]
    fn test_lp_bias_gradient_backward_fd_check() {
        // Finite-difference check for p=3 at a few error values
        let error = vec![0.3, -0.5, 1.0, 0.01];
        let p = 3.0;
        let a = 10.0;
        let eps = 1e-3;

        let d_lp = vec![1.0; error.len()];
        let d_err = lp_bias_gradient_backward(&d_lp, &error, p, a);

        for i in 0..error.len() {
            let mut e_plus = error.clone();
            let mut e_minus = error.clone();
            e_plus[i] += eps;
            e_minus[i] -= eps;
            let g_plus = lp_grad(e_plus[i], p, a);
            let g_minus = lp_grad(e_minus[i], p, a);
            let fd = (g_plus - g_minus) / (2.0 * eps);
            let rel_err = if fd.abs() > 1e-4 {
                ((d_err[i] - fd) / fd).abs()
            } else {
                (d_err[i] - fd).abs()
            };
            assert!(rel_err < 0.05,
                "FD check p={p} e={}: analytic={}, fd={fd}, rel_err={rel_err:.4}",
                error[i], d_err[i]);
        }
    }
}
