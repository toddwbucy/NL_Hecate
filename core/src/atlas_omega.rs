/// Atlas Omega — state-independent memory update rule.
///
/// MIRAS knobs: matrix structure, L2 bias, L2 decay retention, GD+momentum algorithm.
/// Key difference from Titans LMM: replaces state-dependent gradient
/// `grad = outer(M@k - v, k)` with learned state-independent omega function:
/// `omega_mat = outer(omega(k, v), k)` where `omega(k, v) = W_omega @ silu(concat(k, v))`.
///
/// Because omega is independent of M, all tokens' omega values can be precomputed
/// in parallel — enabling the AtlasParallel parallelization strategy.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   eta_t   = sigmoid(concat(k_t, v_t) @ w_eta + b_eta)
///   omega_t = W_omega @ silu(concat(k_t, v_t))              // state-independent
///   omega_mat_t = outer(omega_t, k_t)
///   S_{t+1} = eta_t * S_t - theta_t * omega_mat_t           // momentum accumulator
///   M_{t+1} = (1-alpha_t) * M_t + S_{t+1}                   // memory update
///   y_t = M_{t+1} @ q_t
///
/// Backward: reverse token loop with accumulated d_M and d_S (two recurrences),
/// plus d_W_omega gradient through omega function.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    outer_product_f32, frobenius_dot_f32, silu_f32,
};
use crate::retention::l2_apply_retention;
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};

// ── Atlas Omega implementation ──────────────────────────────────────

/// Atlas Omega: state-independent memory update rule (ATLAS paper, 2505.23735).
pub struct AtlasOmega;

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
    /// Per-token silu(concat(k,v)): [seq_len, 2*d] (needed for W_omega backward)
    pub silu_kv: Vec<f32>,
    /// Per-token omega vectors: [seq_len, d]
    pub omega_vecs: Vec<f32>,
    /// Per-token omega outer products: [seq_len, d*d]
    pub omega_mats: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
    /// Conv1D cache for key preprocessing (None when kernel_size=0)
    pub k_conv_cache: Option<crate::conv1d::Conv1DCache>,
    /// Conv1D cache for query preprocessing (None when kernel_size=0)
    pub q_conv_cache: Option<crate::conv1d::Conv1DCache>,
}

/// Compute omega(k, v) = W_omega @ silu(concat(k, v)).
///
/// Returns (omega_vec [d], silu_kv [2*d]) — silu_kv needed for backward.
fn compute_omega(k: &[f32], v: &[f32], w_omega: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);
    debug_assert_eq!(w_omega.len(), d * 2 * d);

    // concat(k, v) → silu → [2*d]
    let mut silu_kv = vec![0.0f32; 2 * d];
    for i in 0..d {
        silu_kv[i] = silu_f32(k[i]);
    }
    for i in 0..d {
        silu_kv[d + i] = silu_f32(v[i]);
    }

    // W_omega @ silu_kv → [d]
    let mut omega = vec![0.0f32; d];
    for i in 0..d {
        let mut sum = 0.0f32;
        for j in 0..(2 * d) {
            sum += w_omega[i * 2 * d + j] * silu_kv[j];
        }
        omega[i] = sum;
    }

    (omega, silu_kv)
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
        // Simplified write (no momentum, no omega — use step() for full path)
        let d = state.d;
        let retention = 1.0 - gates.alpha;
        let lr = gates.theta;
        // Without W_omega, fall back to delta rule behavior for write()
        for i in 0..d {
            let mut pred_i = 0.0f32;
            for j in 0..d {
                pred_i += state.m[i * d + j] * k[j];
            }
            let err_i = pred_i - v[i];
            for j in 0..d {
                state.m[i * d + j] = retention * state.m[i * d + j] - lr * err_i * k[j];
            }
        }
        Ok(())
    }

    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) -> Result<(), MemoryError> {
        let d = state.d;
        matmul_f32(&state.m, q, out, d, d, 1);
        Ok(())
    }

    /// Full sequence forward with omega function and momentum accumulator S.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, AtlasOmegaCache) {
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

        // Conv1D key/query preprocessing (after projection, before memory loop)
        let (k_conv_cache, q_conv_cache) = crate::conv1d::apply_conv1d_to_kq(
            &mut k_mem, &mut q_mem, level_params, seq_len, d);

        // Allocate cache
        let mut m_states = vec![0.0f32; (seq_len + 1) * d * d];
        let mut s_states = vec![0.0f32; (seq_len + 1) * d * d];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), d * d);
            m_states[..d * d].copy_from_slice(&m0);
        }
        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut theta_pre = vec![0.0f32; seq_len];
        let mut theta = vec![0.0f32; seq_len];
        let mut eta_pre = vec![0.0f32; seq_len];
        let mut eta = vec![0.0f32; seq_len];
        let mut silu_kv = vec![0.0f32; seq_len * 2 * d];
        let mut omega_vecs = vec![0.0f32; seq_len * d];
        let mut omega_mats = vec![0.0f32; seq_len * d * d];
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

            // eta_t = sigmoid(concat @ w_eta + b_eta) — momentum gate
            let mut eta_pre_t = level_params.b_eta[0];
            for i in 0..(2 * d) {
                eta_pre_t += concat_t[i] * level_params.w_eta[i];
            }
            eta_pre[t] = eta_pre_t;
            eta[t] = sigmoid_f32(eta_pre_t);

            // omega_t = W_omega @ silu(concat(k_t, v_t)) — state-independent!
            let (omega_vec, silu_kv_t) = compute_omega(k_t, v_t, &level_params.w_omega, d);
            silu_kv[t * 2 * d..(t + 1) * 2 * d].copy_from_slice(&silu_kv_t);
            omega_vecs[t * d..(t + 1) * d].copy_from_slice(&omega_vec);

            // omega_mat = outer(omega_t, k_t)
            let g_base = t * d * d;
            outer_product_f32(&omega_vec, k_t, &mut omega_mats[g_base..g_base + d * d]);

            // S_{t+1} = eta_t * S_t - theta_t * omega_mat  (momentum update)
            let eta_t = eta[t];
            let theta_t = theta[t];
            let s_t_off = t * d * d;
            let s_next_off = (t + 1) * d * d;
            s_states.copy_within(s_t_off..s_t_off + d * d, s_next_off);
            l2_apply_retention(&mut s_states[s_next_off..s_next_off + d * d], eta_t);
            for i in 0..(d * d) {
                s_states[s_next_off + i] -= theta_t * omega_mats[g_base + i];
            }

            // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}  (memory update with momentum)
            let m_next_off = (t + 1) * d * d;
            let m_t_off = t * d * d;
            m_states.copy_within(m_t_off..m_t_off + d * d, m_next_off);
            l2_apply_retention(&mut m_states[m_next_off..m_next_off + d * d], 1.0 - alpha[t]);
            for i in 0..(d * d) {
                m_states[m_next_off + i] += s_states[s_next_off + i];
            }

            // y_t = M_{t+1} @ q_t
            let m_next = &m_states[m_next_off..m_next_off + d * d];
            matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = AtlasOmegaCache {
            seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
            silu_kv, omega_vecs, omega_mats, y: y.clone(),
            k_conv_cache, q_conv_cache,
        };

        (y, cache)
    }

    /// Full sequence backward through the Atlas Omega memory.
    ///
    /// Two interacting recurrences: d_M and d_S propagate backward.
    /// Additionally: d_W_omega gradient through the omega function.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &AtlasOmegaCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like_from(level_params, d);

        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // d_M and d_S: accumulated gradients on memory and momentum state
        let mut d_m = vec![0.0f32; d * d];
        let mut d_s = vec![0.0f32; d * d];

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let m_t = &cache.m_states[t * d * d..(t + 1) * d * d];
            let m_next = &cache.m_states[(t + 1) * d * d..(t + 2) * d * d];
            let s_t = &cache.s_states[t * d * d..(t + 1) * d * d];
            let omega_mat_t = &cache.omega_mats[t * d * d..(t + 1) * d * d];
            let omega_vec_t = &cache.omega_vecs[t * d..(t + 1) * d];
            let silu_kv_t = &cache.silu_kv[t * 2 * d..(t + 1) * 2 * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let theta_pre_t = cache.theta_pre[t];
            let eta_t = cache.eta[t];

            // ── y_t = M_{t+1} @ q_t backward ──
            let d_y_t = &d_y[t * d..(t + 1) * d];

            // d_M += outer(d_y_t, q_t)
            for i in 0..d {
                for j in 0..d {
                    d_m[i * d + j] += d_y_t[i] * q_t[j];
                }
            }

            // d_q_t = M_{t+1}^T @ d_y_t
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_next[j * d + i] * d_y_t[j];
                }
                d_q_mem[t * d + i] = sum;
            }

            // ── M_{t+1} = (1-alpha) * M_t + S_{t+1} backward ──
            for i in 0..(d * d) {
                d_s[i] += d_m[i];
            }

            let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);

            let mut d_m_prev = d_m.clone();
            l2_apply_retention(&mut d_m_prev, 1.0 - alpha_t);

            // ── S_{t+1} = eta * S_t - theta * omega_mat backward ──
            let d_eta_scalar = frobenius_dot_f32(s_t, &d_s);
            let d_theta_scalar = -frobenius_dot_f32(omega_mat_t, &d_s);

            // d_omega_mat = -theta * d_S
            let mut d_omega_mat = vec![0.0f32; d * d];
            for i in 0..(d * d) {
                d_omega_mat[i] = -theta_t * d_s[i];
            }

            // d_S_prev = eta * d_S (propagate to previous step)
            let mut d_s_prev = d_s.clone();
            l2_apply_retention(&mut d_s_prev, eta_t);

            // ── omega_mat = outer(omega_vec, k_t) backward ──
            // d_omega_vec[i] = sum_j(d_omega_mat[i,j] * k_t[j])
            let mut d_omega_vec = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_omega_mat[i * d + j] * k_t[j];
                }
                d_omega_vec[i] = sum;
            }
            // d_k_t += sum_i(d_omega_mat[i,j] * omega_vec[i])
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_omega_mat[i * d + j] * omega_vec_t[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // ── omega_vec = W_omega @ silu_kv backward ──
            // d_W_omega[i,j] += d_omega_vec[i] * silu_kv[j]
            for i in 0..d {
                for j in 0..(2 * d) {
                    grads.w_omega[i * 2 * d + j] += d_omega_vec[i] * silu_kv_t[j];
                }
            }

            // d_silu_kv = W_omega^T @ d_omega_vec
            let mut d_silu_kv = vec![0.0f32; 2 * d];
            for j in 0..(2 * d) {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += level_params.w_omega[i * 2 * d + j] * d_omega_vec[i];
                }
                d_silu_kv[j] = sum;
            }

            // silu(x) = x * sigmoid(x), silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //                                     = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            // We need d_concat_kv from silu backward
            // concat_kv = [k_t, v_t], silu_kv = silu(concat_kv)
            // d_concat_kv[i] = d_silu_kv[i] * silu_deriv(concat_kv[i])
            for i in 0..(2 * d) {
                let x = concat_t[i];
                let sig = sigmoid_f32(x);
                let silu_deriv = sig * (1.0 + x * (1.0 - sig));
                let d_input = d_silu_kv[i] * silu_deriv;
                // First d elements go to d_k_mem, next d to d_v_mem
                if i < d {
                    d_k_mem[t * d + i] += d_input;
                } else {
                    d_v_mem[t * d + (i - d)] += d_input;
                }
            }

            // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_scalar * sig_deriv;

            // ── Gate backward: theta_t = softplus(theta_pre_t) ──
            let softplus_deriv = sigmoid_f32(theta_pre_t);
            let d_theta_pre = d_theta_scalar * softplus_deriv;

            // ── Gate backward: eta_t = sigmoid(eta_pre_t) ──
            let eta_sig_deriv = eta_t * (1.0 - eta_t);
            let d_eta_pre = d_eta_scalar * eta_sig_deriv;

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

            // ── w_eta, b_eta gradient ──
            for i in 0..(2 * d) {
                grads.w_eta[i] += d_eta_pre * concat_t[i];
            }
            grads.b_eta[0] += d_eta_pre;

            // ── concat backward → d_k_mem, d_v_mem ──
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

            // Update d_m and d_s for next (earlier) token
            d_m = d_m_prev;
            d_s = d_s_prev;
        }

        // ── Conv1D backward (before projection backward) ──
        crate::conv1d::backward_conv1d_kq(
            &mut d_k_mem, &mut d_q_mem,
            &cache.k_conv_cache, &cache.q_conv_cache,
            level_params, &mut grads, s, d);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

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

    #[test]
    fn test_atlas_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_atlas_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let m_0 = &cache.m_states[0..d * d];
        let m_t = &cache.m_states[s * d * d..(s + 1) * d * d];
        let m0_norm: f32 = m_0.iter().map(|x| x * x).sum();
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_atlas_forward_momentum_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let s_0 = &cache.s_states[0..d * d];
        let s0_norm: f32 = s_0.iter().map(|x| x * x).sum();
        assert!(s0_norm < 1e-12, "S_0 should be zero");

        let s_t = &cache.s_states[s * d * d..(s + 1) * d * d];
        let st_norm: f32 = s_t.iter().map(|x| x * x).sum();
        assert!(st_norm > 1e-12, "S_T should have evolved, norm={st_norm}");
    }

    #[test]
    fn test_atlas_forward_omega_state_independent() {
        // Same inputs → same omega regardless of memory state
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let d = cfg.swa.d_model;

        // Run with zero initial M
        let (_, cache1) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);

        // Run with non-zero initial M
        let m0: Vec<f32> = (0..d*d).map(|i| (i as f32) * 0.01).collect();
        let (_, cache2) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, Some(m0));

        // Omega vectors should be identical (state-independent)
        assert_eq!(cache1.omega_vecs, cache2.omega_vecs,
            "omega should be state-independent");
    }

    #[test]
    fn test_atlas_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Atlas Omega forward should be deterministic");
    }

    #[test]
    fn test_atlas_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = AtlasOmega;
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
        assert_eq!(cache.s_states.len(), (s + 1) * d * d);
        assert_eq!(cache.omega_vecs.len(), s * d);
        assert_eq!(cache.omega_mats.len(), s * d * d);
        assert_eq!(cache.silu_kv.len(), s * 2 * d);
    }

    #[test]
    fn test_atlas_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = AtlasOmega;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta), ("w_eta", &grads.w_eta),
            ("b_eta", &grads.b_eta), ("w_omega", &grads.w_omega),
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
        let rule = AtlasOmega;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_omega", &grads.w_omega),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "grad_{name} is all zeros (max_abs={max_abs})");
        }
        let eta_max = grads.w_eta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(eta_max > 1e-10, "w_eta grads should be non-zero");

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
        let rule = AtlasOmega;
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
        assert_eq!(grads.w_omega.len(), d * 2 * d);
        assert_eq!(d_emb.len(), s * d);
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_atlas_read_only_zero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let frozen_m = vec![0.0f32; d * d];
        let (y, _q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_atlas_level_and_parallelization() {
        let rule = AtlasOmega;
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"atlas_parallel"));
    }
}
