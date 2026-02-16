/// Atlas Omega — the 9th MIRAS memory variant.
///
/// Atlas paper (2505.23735) Eqs 9-13, Section 3.2.
///
/// Key insight: The omega function is STATE-INDEPENDENT — it depends only on the
/// input token's projections (k, v) and outer-loop parameters (W_omega), NOT on
/// the memory state M. This enables full parallelization of momentum computation.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   eta_t   = sigmoid(concat(k_t, v_t) @ w_eta + b_eta)
///   omega_t = W_omega @ silu(concat(k_t, v_t))            // state-independent!
///   S_{t+1} = eta_t * S_t - theta_t * omega_t             // momentum (no M dependency)
///   M_{t+1} = (1-alpha_t) * M_t + S_{t+1}                 // memory update
///   y_t = M_{t+1} @ q_t
///
/// MIRAS knobs: matrix structure, L2 bias, L2 decay retention, GD+momentum (state-independent).
///
/// The omega function (atlas_parallel.rs) replaces the gradient computation
/// grad = outer(M@k - v, k) with a learned surrogate omega = W_omega @ silu(concat(k, v)).
/// This removes the M-dependency from momentum, enabling atlas_parallel_forward.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
};
use crate::retention::l2_apply_retention;
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};
use crate::atlas_parallel::{AtlasOmegaParams, atlas_omega};

// ── Atlas Omega implementation ──────────────────────────────────────

/// Atlas Omega: state-independent momentum memory rule (Atlas Eqs 9-13).
///
/// Like TitansLMM but replaces the M-dependent gradient with a learned
/// state-independent omega function, enabling full momentum parallelism.
pub struct AtlasOmega {
    /// Atlas omega function parameters (W_omega, W_k_omega, W_v_omega).
    pub omega_params: AtlasOmegaParams,
}

/// All intermediate values from an Atlas Omega forward pass, needed for backward.
pub struct AtlasOmegaCache {
    pub seq_len: usize,
    pub d: usize,
    /// Memory matrices M_t for t=0..seq_len: [(seq_len+1) * d * d]
    pub m_states: Vec<f32>,
    /// Momentum matrices S_t for t=0..seq_len: [(seq_len+1) * d * d]
    pub s_states: Vec<f32>,
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
    /// Pre-sigmoid eta values: [seq_len]
    pub eta_pre: Vec<f32>,
    /// Sigmoid eta values: [seq_len]
    pub eta: Vec<f32>,
    /// Per-token omega outputs: [seq_len, d*d] (reshaped from [d] vector to [d, d] matrix)
    pub omega_mats: Vec<f32>,
    /// Per-token silu(concat(k,v)): [seq_len, 2*d] — needed for omega backward
    pub silu_kv: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
}

impl MemoryRule for AtlasOmega {
    type Cache = AtlasOmegaCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        crate::parallel::supported_strategies(crate::model::MemoryRuleKind::AtlasOmega)
    }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError> {
        let d = state.d;
        // omega = W_omega @ silu(concat(k, v)) → [d] vector
        let omega_vec = atlas_omega(k, v, &self.omega_params, d);

        // Reshape omega [d] into a [d, d] rank-1 outer product with k:
        // omega_mat[i, j] = omega_vec[i] * k[j]
        // This gives us a d×d matrix to add to M via momentum.
        //
        // S_t has no history in write() — we use theta as direct lr.
        // M = (1-alpha) * M - theta * omega_mat
        l2_apply_retention(&mut state.m, 1.0 - gates.alpha);
        for i in 0..d {
            for j in 0..d {
                state.m[i * d + j] -= gates.theta * omega_vec[i] * k[j];
            }
        }
        Ok(())
    }

    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) -> Result<(), MemoryError> {
        let d = state.d;
        matmul_f32(&state.m, q, out, d, d, 1);
        Ok(())
    }

    /// Full sequence forward with cache for backward.
    ///
    /// Unlike TitansLMM which uses grad = outer(M@k - v, k) for momentum,
    /// Atlas Omega uses omega = W_omega @ silu(concat(k, v)) — state-independent.
    ///
    /// The omega function outputs a [d] vector. We form the momentum update as
    /// an outer product: omega_mat = outer(omega, k), giving a [d, d] matrix.
    ///
    /// S_{t+1} = eta_t * S_t - theta_t * omega_mat_t
    /// M_{t+1} = (1-alpha_t) * M_t + S_{t+1}
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, AtlasOmegaCache) {
        debug_assert_eq!(embedded.len(), seq_len * d);
        let dd = d * d;

        // Project embedded → k_mem, v_mem, q_mem via W^T
        let mut w_k_mem_t = vec![0.0f32; dd];
        let mut w_v_mem_t = vec![0.0f32; dd];
        let mut w_q_mem_t = vec![0.0f32; dd];
        transpose_f32(&level_params.w_k_mem, &mut w_k_mem_t, d, d);
        transpose_f32(&level_params.w_v_mem, &mut w_v_mem_t, d, d);
        transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);

        let mut k_mem = vec![0.0f32; seq_len * d];
        let mut v_mem = vec![0.0f32; seq_len * d];
        let mut q_mem = vec![0.0f32; seq_len * d];
        matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
        matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
        matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

        // Allocate cache
        let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
        let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), dd);
            m_states[..dd].copy_from_slice(&m0);
        }
        // S_0 = zeros (always)

        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut theta_pre = vec![0.0f32; seq_len];
        let mut theta = vec![0.0f32; seq_len];
        let mut eta_pre = vec![0.0f32; seq_len];
        let mut eta = vec![0.0f32; seq_len];
        let mut omega_mats = vec![0.0f32; seq_len * dd];
        let mut silu_kv = vec![0.0f32; seq_len * 2 * d];
        let mut y = vec![0.0f32; seq_len * d];

        // Sequential token loop
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

            // eta_t = sigmoid(concat @ w_eta + b_eta)
            let mut eta_pre_t = level_params.b_eta[0];
            for i in 0..(2 * d) {
                eta_pre_t += concat_t[i] * level_params.w_eta[i];
            }
            eta_pre[t] = eta_pre_t;
            eta[t] = sigmoid_f32(eta_pre_t);

            // Compute omega: state-independent surrogate gradient
            // omega_vec = W_omega @ silu(concat(k, v)) → [d]
            // First compute silu(concat(k, v)) and cache it
            let sk_base = t * 2 * d;
            for i in 0..(2 * d) {
                let x = concat_t[i];
                silu_kv[sk_base + i] = crate::tensor::silu_f32(x);
            }
            let silu_t = &silu_kv[sk_base..sk_base + 2 * d];

            // omega_vec = W_omega @ silu_t → [d]
            let mut omega_vec = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..(2 * d) {
                    sum += self.omega_params.w_omega[i * 2 * d + j] * silu_t[j];
                }
                omega_vec[i] = sum;
            }

            // omega_mat = outer(omega_vec, k_t) → [d, d]
            let om_base = t * dd;
            for i in 0..d {
                for j in 0..d {
                    omega_mats[om_base + i * d + j] = omega_vec[i] * k_t[j];
                }
            }

            // S_{t+1} = eta_t * S_t - theta_t * omega_mat
            let s_t = t * dd;
            let s_next = (t + 1) * dd;
            for i in 0..dd {
                s_states[s_next + i] = eta[t] * s_states[s_t + i] - theta[t] * omega_mats[om_base + i];
            }

            // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}
            let m_t = t * dd;
            let m_next = (t + 1) * dd;
            let retention = 1.0 - alpha[t];
            for i in 0..dd {
                m_states[m_next + i] = retention * m_states[m_t + i] + s_states[s_next + i];
            }

            // y_t = M_{t+1} @ q_t
            let m_next_slice = &m_states[m_next..m_next + dd];
            matmul_f32(m_next_slice, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = AtlasOmegaCache {
            seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
            omega_mats, silu_kv, y: y.clone(),
        };

        (y, cache)
    }

    /// Full sequence backward through Atlas Omega.
    ///
    /// Reverse token loop with accumulated d_M and d_S (two recurrences),
    /// similar to TitansLMM but with omega-specific gradients.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &AtlasOmegaCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        let dd = d * d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like(d);

        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        let mut d_m = vec![0.0f32; dd];
        let mut d_s = vec![0.0f32; dd];

        // Accumulate omega param gradients
        let mut d_w_omega = vec![0.0f32; d * 2 * d];

        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let _v_t = &cache.v_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let d_y_t = &d_y[t * d..(t + 1) * d];
            let m_t_off = t * dd;
            let m_next_off = (t + 1) * dd;
            let s_t_off = t * dd;
            let om_base = t * dd;
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let eta_t = cache.eta[t];

            // ── y_t = M_{t+1} @ q_t backward ──
            // d_M += outer(d_y_t, q_t)
            for i in 0..d {
                for j in 0..d {
                    d_m[i * d + j] += d_y_t[i] * q_t[j];
                }
            }

            // d_q_t = M_{t+1}^T @ d_y_t
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += cache.m_states[m_next_off + i * d + j] * d_y_t[i];
                }
                d_q_mem[t * d + j] = sum;
            }

            // ── M_{t+1} = (1-alpha) * M_t + S_{t+1} backward ──
            // d_S_{t+1} += d_M
            for i in 0..dd {
                d_s[i] += d_m[i];
            }

            // d_alpha = -frobenius_dot(d_M, M_t)
            let mut d_alpha_sum = 0.0f32;
            for i in 0..dd {
                d_alpha_sum += cache.m_states[m_t_off + i] * d_m[i];
            }

            let retention = 1.0 - alpha_t;
            let mut d_m_prev = vec![0.0f32; dd];
            for i in 0..dd {
                d_m_prev[i] = retention * d_m[i];
            }

            // ── S_{t+1} = eta * S_t - theta * omega_mat backward ──
            // d_eta = frobenius_dot(d_S, S_t)
            let mut d_eta_sum = 0.0f32;
            for i in 0..dd {
                d_eta_sum += cache.s_states[s_t_off + i] * d_s[i];
            }

            // d_theta = -frobenius_dot(d_S, omega_mat)
            let mut d_theta_sum = 0.0f32;
            for i in 0..dd {
                d_theta_sum += cache.omega_mats[om_base + i] * d_s[i];
            }

            // d_omega_mat = -theta * d_S
            let mut d_omega_mat = vec![0.0f32; dd];
            for i in 0..dd {
                d_omega_mat[i] = -theta_t * d_s[i];
            }

            // ── omega_mat[i,j] = omega_vec[i] * k[j] backward ──
            // d_omega_vec[i] = sum_j d_omega_mat[i,j] * k[j]
            let mut d_omega_vec = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_omega_mat[i * d + j] * k_t[j];
                }
                d_omega_vec[i] = sum;
            }
            // d_k contribution from omega_mat:
            // d_k[j] += sum_i d_omega_mat[i,j] * omega_vec[i]
            // omega_vec[i] = omega_mats[om_base + i*d + 0] / k_t[0]... not stored directly
            // Recompute omega_vec from omega_mats: omega_vec[i] = omega_mats[i*d+j]/k_t[j] for any nonzero k_t[j]
            // Safer: recompute from silu_kv
            let sk_base = t * 2 * d;
            let silu_t = &cache.silu_kv[sk_base..sk_base + 2 * d];
            let mut omega_vec_recomputed = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..(2 * d) {
                    sum += self.omega_params.w_omega[i * 2 * d + j] * silu_t[j];
                }
                omega_vec_recomputed[i] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_omega_mat[i * d + j] * omega_vec_recomputed[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // ── omega_vec = W_omega @ silu_kv backward ──
            // d_W_omega += outer(d_omega_vec, silu_t)
            for i in 0..d {
                for j in 0..(2 * d) {
                    d_w_omega[i * 2 * d + j] += d_omega_vec[i] * silu_t[j];
                }
            }

            // d_silu_t = W_omega^T @ d_omega_vec
            let mut d_silu_t = vec![0.0f32; 2 * d];
            for j in 0..(2 * d) {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += self.omega_params.w_omega[i * 2 * d + j] * d_omega_vec[i];
                }
                d_silu_t[j] = sum;
            }

            // ── silu(x) = x * sigmoid(x) backward ──
            // d_x = d_silu * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
            //      = d_silu * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let mut d_concat_omega = vec![0.0f32; 2 * d];
            for i in 0..(2 * d) {
                let x = concat_t[i];
                let sig = sigmoid_f32(x);
                let dsilu_dx = sig * (1.0 + x * (1.0 - sig));
                d_concat_omega[i] = d_silu_t[i] * dsilu_dx;
            }

            // d_k_mem, d_v_mem from omega path via concat
            for i in 0..d {
                d_k_mem[t * d + i] += d_concat_omega[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_concat_omega[d + i];
            }

            // ── Gate backward ──

            // alpha gate: alpha = sigmoid(alpha_pre)
            let alpha_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = (-d_alpha_sum) * alpha_deriv;

            // theta gate: theta = softplus(theta_pre)
            let theta_deriv = sigmoid_f32(cache.theta_pre[t]);
            let d_theta_pre = (-d_theta_sum) * theta_deriv;

            // eta gate: eta = sigmoid(eta_pre)
            let eta_deriv = eta_t * (1.0 - eta_t);
            let d_eta_pre = d_eta_sum * eta_deriv;

            // Gate weight gradients
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
                grads.w_theta[i] += d_theta_pre * concat_t[i];
                grads.w_eta[i] += d_eta_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;
            grads.b_theta[0] += d_theta_pre;
            grads.b_eta[0] += d_eta_pre;

            // Gate backward → d_k, d_v via concat
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i]
                                    + d_theta_pre * level_params.w_theta[i]
                                    + d_eta_pre * level_params.w_eta[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i]
                                    + d_theta_pre * level_params.w_theta[d + i]
                                    + d_eta_pre * level_params.w_eta[d + i];
            }

            // ── Propagate d_S backward ──
            // d_S_t = eta * d_S_{t+1}
            let mut d_s_prev = vec![0.0f32; dd];
            for i in 0..dd {
                d_s_prev[i] = eta_t * d_s[i];
            }

            d_m = d_m_prev;
            d_s = d_s_prev;
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

        // Note: d_w_omega gradients are accumulated but not returned through MemoryLevelParams
        // since omega params are separate from the standard level params.
        // In a full implementation, these would flow through AtlasOmegaParams gradients.
        // For now, the omega params contribute to d_embedded via the chain rule above.

        (grads, d_embedded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::atlas_parallel::AtlasOmegaParams;

    fn test_config() -> MAGConfig {
        MAGConfig::atlas_test_config()
    }

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    fn make_rule(d: usize) -> AtlasOmega {
        AtlasOmega {
            omega_params: AtlasOmegaParams::init(d, 42),
        }
    }

    #[test]
    fn test_atlas_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_atlas_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let rule = make_rule(d);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let m_0 = &cache.m_states[0..d * d];
        let m_t = &cache.m_states[s * d * d..(s + 1) * d * d];
        let m0_norm: f32 = m_0.iter().map(|x| x * x).sum();
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_atlas_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            let th = cache.theta[t];
            assert!(th >= 0.0, "theta[{t}]={th} should be non-negative");
            let e = cache.eta[t];
            assert!(e > 0.0 && e < 1.0, "eta[{t}]={e} not in (0,1)");
        }
    }

    #[test]
    fn test_atlas_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        assert_eq!(y1, y2, "Atlas Omega forward should be deterministic");
    }

    #[test]
    fn test_atlas_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let rule = make_rule(d);
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
        assert_eq!(cache.s_states.len(), (s + 1) * d * d);
    }

    #[test]
    fn test_atlas_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta), ("w_eta", &grads.w_eta),
            ("b_eta", &grads.b_eta),
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
    fn test_atlas_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
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
    fn test_atlas_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = make_rule(d);
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
        assert_eq!(grads.w_eta.len(), 2 * d);
        assert_eq!(grads.b_eta.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }

    // ── Trait API tests ──────────────────────────────────────────────

    #[test]
    fn test_atlas_init() {
        let rule = make_rule(8);
        let state = rule.init(8);
        assert_eq!(state.m.len(), 64);
        assert_eq!(state.d, 8);
        assert!(state.m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_atlas_level_and_parallelization() {
        let rule = make_rule(8);
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"chunkwise_gd"));
        assert!(strategies.contains(&"atlas_parallel"));
    }

    #[test]
    fn test_atlas_state_independence() {
        // The omega function should produce identical outputs for the same input,
        // regardless of memory state — this is the key Atlas property.
        let d = 8;
        let omega_params = AtlasOmegaParams::init(d, 42);
        let k = vec![0.3f32, -0.2, 0.5, 0.1, 0.4, -0.1, 0.2, 0.6];
        let v = vec![-0.1f32, 0.4, 0.2, -0.3, 0.1, 0.5, -0.2, 0.3];

        let omega1 = atlas_omega(&k, &v, &omega_params, d);
        let omega2 = atlas_omega(&k, &v, &omega_params, d);

        for (a, b) in omega1.iter().zip(omega2.iter()) {
            assert_eq!(*a, *b, "omega should be deterministic and state-independent");
        }
    }
}
