/// YAAD memory system — 2-layer MLP with Huber attentional bias and decoupled retention.
///
/// Second MLP-family MIRAS variant, sibling to MONETA. Same 2-layer MLP structure
/// (y = W2 @ silu(W1 @ q)), same GD algorithm. Two targeted changes:
///
/// 1. Huber loss replaces l_p — gradient is `e` for |e| < delta, `delta * sign(e)` for |e| >= delta.
///    Bounded gradient (never exceeds delta) protects memory from outlier tokens.
///
/// 2. Decoupled retention — local L2 (stay near chunk-boundary snapshot) + global L2 (keep small),
///    replacing MONETA's global-only L2.
///
/// MIRAS knobs: MLP structure, Huber attentional bias, decoupled retention, GD algorithm.
/// Source: MIRAS (2504.13173) Eq 26, Table 2.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   pre_act = W1 @ k_t;  h = silu(pre_act)
///   prediction = W2 @ h;  error = prediction - v_t
///   huber_grad = e if |e| < delta, delta * sign(e) otherwise
///   grad_W2 = outer(huber_grad, h);  grad_W1 = outer(silu'(pre_act) * (W2^T @ huber_grad), k_t)
///   ret_local = lambda_local * 2 * (W - W_boundary)
///   ret_global = lambda_2 * 2 * W
///   W1 = alpha_t * W1 - theta_t * (grad_W1 + ret_local_W1 + ret_global_W1)
///   W2 = alpha_t * W2 - theta_t * (grad_W2 + ret_local_W2 + ret_global_W2)
///   y_t = W2 @ silu(W1 @ q_t)
///
/// Backward: reverse token loop with accumulated d_W1, d_W2.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    silu_f32, silu_prime_f32, frobenius_dot_f32,
};
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates};

// ── YAAD implementation ─────────────────────────────────────────────

/// YAAD: 2-layer MLP memory with Huber loss and decoupled retention.
pub struct YAAD {
    pub d_hidden: usize,
    pub delta: f32,
    pub lambda_local: f32,
    pub lambda_2: f32,
}

/// All intermediate values from a YAAD forward pass, needed for backward.
pub struct YAADCache {
    pub seq_len: usize,
    pub d: usize,
    pub d_hidden: usize,
    /// W1 states for t=0..seq_len: [(seq_len+1) * d_hidden * d]
    pub w1_states: Vec<f32>,
    /// W2 states for t=0..seq_len: [(seq_len+1) * d * d_hidden]
    pub w2_states: Vec<f32>,
    /// Chunk-boundary snapshot of W1: [d_hidden * d]
    pub w1_boundary: Vec<f32>,
    /// Chunk-boundary snapshot of W2: [d * d_hidden]
    pub w2_boundary: Vec<f32>,
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
    /// Huber delta threshold (cached for backward)
    pub delta: f32,
    /// Local retention strength (cached for backward)
    pub lambda_local: f32,
    /// Global L2 retention strength (cached for backward)
    pub lambda_2: f32,
}

/// Compute Huber gradient: e if |e| < delta, delta * sign(e) otherwise.
/// Bounded gradient magnitude — never exceeds delta.
#[inline]
fn huber_grad(e: f32, delta: f32) -> f32 {
    if e.abs() < delta {
        e
    } else {
        delta * e.signum()
    }
}

impl MemoryRule for YAAD {
    type Cache = YAADCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] { &["sequential"] }

    fn init(&self, d: usize) -> MemoryState {
        // For API compatibility — actual YAAD state is W1+W2+boundaries, not a d×d matrix.
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, _state: &mut MemoryState, _k: &[f32], _v: &[f32], _gates: &Gates) {
        unimplemented!("YAAD does not support direct write — use step() instead");
    }

    fn read(&self, _state: &MemoryState, _q: &[f32], _out: &mut [f32]) {
        unimplemented!("YAAD does not support direct read — use step() instead");
    }

    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<&[f32]>,
    ) -> (Vec<f32>, YAADCache) {
        let dh = self.d_hidden;
        let hub_delta = self.delta;
        let l_local = self.lambda_local;
        let l2 = self.lambda_2;
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
            let scale = (2.0 / (d + dh) as f32).sqrt() * 0.1;
            for i in 0..w1_size {
                let hash = ((i as u32).wrapping_mul(2654435761)) as f32 / u32::MAX as f32;
                w1_states[i] = scale * (hash - 0.5);
            }
        }

        // Snapshot W1_0 and W2_0 as chunk boundary state
        let w1_boundary = w1_states[..w1_size].to_vec();
        let w2_boundary = w2_states[..w2_size].to_vec();

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
            let (w1_left, w1_right) = w1_states.split_at_mut((t + 1) * w1_size);
            let w1_t = &w1_left[t * w1_size..];
            let w1_next = &mut w1_right[..w1_size];

            let (w2_left, w2_right) = w2_states.split_at_mut((t + 1) * w2_size);
            let w2_t = &w2_left[t * w2_size..];
            let w2_next = &mut w2_right[..w2_size];

            let pa_base = t * dh;
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

            // Huber gradient on error
            let mut hub_g = vec![0.0f32; d];
            for i in 0..d {
                hub_g[i] = huber_grad(error_all[pred_base + i], hub_delta);
            }

            // grad_W2 = outer(huber_grad, h) → [d, dh]
            let mut grad_w2 = vec![0.0f32; w2_size];
            for i in 0..d {
                for j in 0..dh {
                    grad_w2[i * dh + j] = hub_g[i] * h_t[j];
                }
            }

            // grad_h = W2^T @ huber_grad → [dh]
            let mut grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += w2_t[j * dh + i] * hub_g[j];
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

            // Decoupled retention:
            //   local:  lambda_local * 2 * (W - W_boundary)
            //   global: lambda_2 * 2 * W
            let l_local_2 = l_local * 2.0;
            let l2_2 = l2 * 2.0;

            // Update: W1 = alpha_t * W1 - theta_t * (grad_W1 + ret_local_W1 + ret_global_W1)
            //         W2 = alpha_t * W2 - theta_t * (grad_W2 + ret_local_W2 + ret_global_W2)
            let alpha_t = alpha[t];
            let theta_t = theta[t];
            for i in 0..w1_size {
                let ret_local = l_local_2 * (w1_t[i] - w1_boundary[i]);
                let ret_global = l2_2 * w1_t[i];
                w1_next[i] = alpha_t * w1_t[i] - theta_t * (grad_w1[i] + ret_local + ret_global);
            }
            for i in 0..w2_size {
                let ret_local = l_local_2 * (w2_t[i] - w2_boundary[i]);
                let ret_global = l2_2 * w2_t[i];
                w2_next[i] = alpha_t * w2_t[i] - theta_t * (grad_w2[i] + ret_local + ret_global);
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

        let cache = YAADCache {
            seq_len, d, d_hidden: dh,
            w1_states, w2_states,
            w1_boundary, w2_boundary,
            k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta,
            pre_act: pre_act_all, hidden: hidden_all,
            prediction: prediction_all, error: error_all,
            y: y.clone(),
            delta: hub_delta, lambda_local: l_local, lambda_2: l2,
        };

        (y, cache)
    }

    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &YAADCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        let dh = cache.d_hidden;
        let hub_delta = cache.delta;
        let l_local = cache.lambda_local;
        let l2 = cache.lambda_2;
        let l_local_2 = l_local * 2.0;
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

            // ── W_next = alpha * W_t - theta * (grad_W + ret_local + ret_global) backward ──
            // d_alpha from W1: sum_i(d_W1_next_i * W1_t_i)
            // d_alpha from W2: sum_i(d_W2_next_i * W2_t_i)
            let d_alpha_w1 = frobenius_dot_f32(&d_w1, w1_t);
            let d_alpha_w2 = frobenius_dot_f32(&d_w2, w2_t);
            let d_alpha_scalar = d_alpha_w1 + d_alpha_w2;

            // Recompute huber_grad and MLP gradients for this token
            let pred_base = t * d;
            let mut hub_g = vec![0.0f32; d];
            for i in 0..d {
                hub_g[i] = huber_grad(cache.error[pred_base + i], hub_delta);
            }

            // Recompute grad_W2 = outer(hub_g, h_t)
            let mut grad_w2 = vec![0.0f32; w2_size];
            for i in 0..d {
                for j in 0..dh {
                    grad_w2[i * dh + j] = hub_g[i] * h_t[j];
                }
            }
            // Recompute grad_h, grad_pre, grad_W1
            let mut grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += w2_t[j * dh + i] * hub_g[j];
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

            // d_theta_scalar: d_W_next_i * -(grad_W_i + ret_local_i + ret_global_i)
            let mut d_theta_scalar = 0.0f32;
            for i in 0..w1_size {
                let ret_local = l_local_2 * (w1_t[i] - cache.w1_boundary[i]);
                let ret_global = l2_2 * w1_t[i];
                d_theta_scalar -= d_w1[i] * (grad_w1[i] + ret_local + ret_global);
            }
            for i in 0..w2_size {
                let ret_local = l_local_2 * (w2_t[i] - cache.w2_boundary[i]);
                let ret_global = l2_2 * w2_t[i];
                d_theta_scalar -= d_w2[i] * (grad_w2[i] + ret_local + ret_global);
            }

            // ── Backprop d_W_next through the update to get d_W_t and d_grad ──
            // d_grad_W1 = -theta * d_W1_next, d_grad_W2 = -theta * d_W2_next
            let mut d_grad_w1 = vec![0.0f32; w1_size];
            let mut d_grad_w2 = vec![0.0f32; w2_size];
            for i in 0..w1_size {
                d_grad_w1[i] = -theta_t * d_w1[i];
            }
            for i in 0..w2_size {
                d_grad_w2[i] = -theta_t * d_w2[i];
            }

            // d_W1_t from update eq:
            //   W1_next = alpha_t * W1_t - theta_t * (grad_W1 + l_local_2*(W1_t - W1_b) + l2_2*W1_t)
            //   dW_next/dW_t = alpha_t - theta_t * (l_local_2 + l2_2) [for the retention terms]
            //   Plus contributions from grad_W terms that depend on W_t
            let coeff = alpha_t - theta_t * (l_local_2 + l2_2);
            let mut d_w1_prev = vec![0.0f32; w1_size];
            let mut d_w2_prev = vec![0.0f32; w2_size];
            for i in 0..w1_size {
                d_w1_prev[i] = coeff * d_w1[i];
            }
            for i in 0..w2_size {
                d_w2_prev[i] = coeff * d_w2[i];
            }

            // ── Backprop through MLP gradient computation ──
            // grad_W2 = outer(hub_g, h) → d_hub_g from d_grad_W2, d_h from d_grad_W2
            let mut d_hub_g = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..dh {
                    sum += d_grad_w2[i * dh + j] * h_t[j];
                }
                d_hub_g[i] = sum;
            }
            let mut d_h_from_gw2 = vec![0.0f32; dh];
            for j in 0..dh {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_grad_w2[i * dh + j] * hub_g[i];
                }
                d_h_from_gw2[j] = sum;
            }

            // grad_W1 = outer(grad_pre, k_t) backward
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
            let mut d_grad_h = vec![0.0f32; dh];
            for i in 0..dh {
                d_grad_h[i] = d_grad_pre[i] * silu_prime_f32(cache.pre_act[pa_base + i]);
            }

            // grad_h = W2^T @ hub_g backward
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..dh {
                    sum += w2_t[j * dh + i] * d_grad_h[i];
                }
                d_hub_g[j] += sum;
            }
            // d_W2_t from grad_h: d_W2_t[j, i] += hub_g[j] * d_grad_h[i]
            for j in 0..d {
                for i in 0..dh {
                    d_w2_prev[j * dh + i] += hub_g[j] * d_grad_h[i];
                }
            }

            // ── Huber gradient backward: hub_g(e) ──
            // For |e| < delta: hub_g = e, so d_hub_g/de = 1.0
            // For |e| >= delta: hub_g = delta * sign(e), so d_hub_g/de = 0.0
            let mut d_err = vec![0.0f32; d];
            for i in 0..d {
                let e = cache.error[pred_base + i];
                let huber_second = if e.abs() < hub_delta { 1.0 } else { 0.0 };
                d_err[i] = d_hub_g[i] * huber_second;
            }

            // error = prediction - v_t backward
            for i in 0..d {
                d_v_mem[t * d + i] -= d_err[i];
            }

            // prediction = W2_t @ h backward (through d_err)
            for i in 0..d {
                for j in 0..dh {
                    d_w2_prev[i * dh + j] += d_err[i] * h_t[j];
                }
            }
            let mut d_h_total = d_h_from_gw2;
            for j in 0..dh {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += w2_t[i * dh + j] * d_err[i];
                }
                d_h_total[j] += sum;
            }

            // h = silu(pre_act) backward
            let mut d_pre_act = vec![0.0f32; dh];
            for i in 0..dh {
                d_pre_act[i] = d_h_total[i] * silu_prime_f32(cache.pre_act[pa_base + i]);
            }

            // pre_act = W1_t @ k_t backward
            for i in 0..dh {
                for j in 0..d {
                    d_w1_prev[i * d + j] += d_pre_act[i] * k_t[j];
                }
            }
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

// ── Read-only functions (for frozen CMS levels) ─────────────────────

/// Forward pass for a frozen YAAD level: y_t = W2 @ silu(W1 @ q_t).
/// Same as MONETA read-only — MLP structure is identical when frozen.
pub fn yaad_read_only(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    frozen_m: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Frozen MLP forward is identical to MONETA — same y = W2 @ silu(W1 @ q)
    crate::moneta::moneta_read_only(level_params, embedded, frozen_m, seq_len, d, d_hidden)
}

/// Backward pass for a frozen YAAD level.
/// Same as MONETA read-only backward — only q_mem projection flows back.
pub fn yaad_read_only_backward(
    level_params: &MemoryLevelParams,
    frozen_m: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
) -> (MemoryLevelParams, Vec<f32>) {
    crate::moneta::moneta_read_only_backward(
        level_params, frozen_m, q_mem, d_y, embedded, seq_len, d, d_hidden,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

    fn test_config() -> MAGConfig {
        MAGConfig::yaad_test_config()
    }

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    fn make_yaad(cfg: &MAGConfig) -> YAAD {
        YAAD {
            d_hidden: cfg.d_hidden,
            delta: cfg.delta,
            lambda_local: cfg.lambda_local,
            lambda_2: cfg.lambda_2,
        }
    }

    #[test]
    fn test_yaad_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_yaad_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "YAAD forward should be deterministic");
    }

    #[test]
    fn test_yaad_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
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
        assert_eq!(cache.w1_boundary.len(), dh * d);
        assert_eq!(cache.w2_boundary.len(), d * dh);
        assert_eq!(cache.pre_act.len(), s * dh);
        assert_eq!(cache.hidden.len(), s * dh);
    }

    #[test]
    fn test_yaad_forward_mlp_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let dh = cfg.d_hidden;
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let w1_0_norm: f32 = cache.w1_states[0..dh * d].iter().map(|x| x * x).sum::<f32>().sqrt();
        let w1_t_norm: f32 = cache.w1_states[s * dh * d..(s + 1) * dh * d].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(w1_0_norm > 1e-6, "W1_0 should be initialized (Xavier), norm={w1_0_norm}");
        assert!((w1_t_norm - w1_0_norm).abs() > 1e-8, "W1_T should have evolved from init");
    }

    #[test]
    fn test_yaad_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            let th = cache.theta[t];
            assert!(th >= 0.0, "theta[{t}]={th} should be non-negative");
        }
    }

    #[test]
    fn test_yaad_boundary_snapshot() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = make_yaad(&cfg);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let dh = cfg.d_hidden;
        let d = cfg.swa.d_model;

        // Boundary should match W1_0
        assert_eq!(&cache.w1_boundary, &cache.w1_states[0..dh * d]);
        assert_eq!(&cache.w2_boundary, &cache.w2_states[0..d * dh]);
    }

    #[test]
    fn test_yaad_huber_gradient_bounded() {
        // Verify huber_grad clips large errors at delta
        let delta = 1.0f32;
        // Small error: gradient = error
        assert!((huber_grad(0.5, delta) - 0.5).abs() < 1e-8);
        assert!((huber_grad(-0.3, delta) - (-0.3)).abs() < 1e-8);
        // Large error: gradient = delta * sign(e)
        assert!((huber_grad(5.0, delta) - 1.0).abs() < 1e-8);
        assert!((huber_grad(-10.0, delta) - (-1.0)).abs() < 1e-8);
        // At boundary
        assert!((huber_grad(0.99, delta) - 0.99).abs() < 1e-8);
    }

    #[test]
    fn test_yaad_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_yaad(&cfg);
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
    fn test_yaad_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_yaad(&cfg);
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
    fn test_yaad_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_yaad(&cfg);
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
    fn test_yaad_init() {
        let rule = YAAD { d_hidden: 4, delta: 1.0, lambda_local: 0.01, lambda_2: 0.01 };
        let state = rule.init(8);
        assert_eq!(state.m.len(), 64);
        assert_eq!(state.d, 8);
        assert!(state.m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_yaad_level_and_parallelization() {
        let rule = YAAD { d_hidden: 4, delta: 1.0, lambda_local: 0.01, lambda_2: 0.01 };
        assert_eq!(rule.level(), 0);
        assert_eq!(rule.supported_parallelization(), &["sequential"]);
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_yaad_read_only_zero_memory() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let frozen_m = vec![0.0f32; dh * d + d * dh];
        let (y, _q_mem) = yaad_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_yaad_read_only_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let mut rng = SimpleRng::new(77);
        let mut frozen_m = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut frozen_m, 0.1);
        let (y, _q_mem) = yaad_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        let y_norm: f32 = y.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(y_norm > 1e-6, "Non-zero W1/W2 should produce non-zero output");
    }

    #[test]
    fn test_yaad_read_only_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let mut rng = SimpleRng::new(77);
        let mut frozen_m = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut frozen_m, 0.1);
        let (_, q_mem) = yaad_read_only(&params.levels[0], &embedded, &frozen_m, s, d, dh);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = yaad_read_only_backward(
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
    fn test_yaad_initial_m_seeding() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let dh = cfg.d_hidden;
        let rule = make_yaad(&cfg);

        let (y1, _) = rule.step(&params.levels[0], &embedded, s, d, None);
        let mut rng = SimpleRng::new(77);
        let mut m0 = vec![0.0f32; dh * d + d * dh];
        rng.fill_uniform(&mut m0, 0.1);
        let (y2, _) = rule.step(&params.levels[0], &embedded, s, d, Some(&m0));

        assert_ne!(y1, y2, "Initial memory seeding should change output");
    }
}
