/// Titans LMM (Long-term Memory Module) — GD + Momentum memory rule.
///
/// Extends Delta Rule with a momentum accumulator S (Titans Eqs 12-15, 32-33).
/// MIRAS knobs: matrix structure, L2 bias, L2 decay retention, GD+momentum algorithm.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   eta_t   = sigmoid(concat(k_t, v_t) @ w_eta + b_eta)        // NEW: momentum gate
///   prediction = M_t @ k_t
///   error = prediction - v_t
///   grad = outer(error, k_t)
///   S_{t+1} = eta_t * S_t - theta_t * grad                     // NEW: momentum accumulator
///   M_{t+1} = (1-alpha_t) * M_t + S_{t+1}                      // Memory uses momentum
///   y_t = M_{t+1} @ q_t
///
/// Backward: reverse token loop with accumulated d_M AND d_S (two recurrences).

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    outer_product_f32, frobenius_dot_f32,
};
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};

// ── Titans LMM implementation ───────────────────────────────────────

/// Titans LMM: GD + momentum memory rule (Titans Eqs 12-15, 32-33).
pub struct TitansLMM;

/// All intermediate values from a Titans LMM forward pass, needed for backward.
pub struct TitansLMMCache {
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
    /// Prediction errors: [seq_len, d]
    pub error: Vec<f32>,
    /// Gradient outer products: [seq_len, d*d]
    pub grad_outer: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
}

impl MemoryRule for TitansLMM {
    type Cache = TitansLMMCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] { &["sequential"] }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError> {
        // Simplified write (no momentum tracking — use step() for full path)
        let d = state.d;
        let mut prediction = vec![0.0f32; d];
        matmul_f32(&state.m, k, &mut prediction, d, d, 1);

        let retention = 1.0 - gates.alpha;
        let lr = gates.theta;
        for i in 0..d {
            let err_i = prediction[i] - v[i];
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

    /// Full sequence forward with momentum accumulator S.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, TitansLMMCache) {
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

        // Allocate cache — seed M_0 from initial_m if provided, else zeros
        // S_0 is always zeros (no initial momentum)
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
        let mut error = vec![0.0f32; seq_len * d];
        let mut grad_outer = vec![0.0f32; seq_len * d * d];
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

            // eta_t = sigmoid(concat @ w_eta + b_eta) — NEW: momentum gate
            let mut eta_pre_t = level_params.b_eta[0];
            for i in 0..(2 * d) {
                eta_pre_t += concat_t[i] * level_params.w_eta[i];
            }
            eta_pre[t] = eta_pre_t;
            eta[t] = sigmoid_f32(eta_pre_t);

            // prediction = M_t @ k_t
            let m_t = &m_states[t * d * d..(t + 1) * d * d];
            let mut prediction = vec![0.0f32; d];
            matmul_f32(m_t, k_t, &mut prediction, d, d, 1);

            // error = prediction - v_t
            let e_base = t * d;
            for i in 0..d {
                error[e_base + i] = prediction[i] - v_t[i];
            }

            // grad = outer(error, k_t)
            let g_base = t * d * d;
            outer_product_f32(&error[e_base..e_base + d], k_t, &mut grad_outer[g_base..g_base + d * d]);

            // S_{t+1} = eta_t * S_t - theta_t * grad  (momentum update)
            let eta_t = eta[t];
            let theta_t = theta[t];
            let s_t_off = t * d * d;
            let s_next_off = (t + 1) * d * d;
            for i in 0..(d * d) {
                s_states[s_next_off + i] = eta_t * s_states[s_t_off + i]
                    - theta_t * grad_outer[g_base + i];
            }

            // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}  (memory update with momentum)
            let retention = 1.0 - alpha[t];
            let m_next_off = (t + 1) * d * d;
            for i in 0..(d * d) {
                m_states[m_next_off + i] = retention * m_states[t * d * d + i]
                    + s_states[s_next_off + i];
            }

            // y_t = M_{t+1} @ q_t
            let m_next = &m_states[m_next_off..m_next_off + d * d];
            matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = TitansLMMCache {
            seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
            error, grad_outer, y: y.clone(),
        };

        (y, cache)
    }

    /// Full sequence backward through the Titans LMM memory.
    ///
    /// Two interacting recurrences: d_M and d_S propagate backward.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &TitansLMMCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like(d);

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
            let err_t = &cache.error[t * d..(t + 1) * d];
            let grad_t = &cache.grad_outer[t * d * d..(t + 1) * d * d];
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
            // d_S_{t+1} = d_M  (S contributes additively to M)
            // Note: d_s already accumulates from future — add current d_m contribution
            for i in 0..(d * d) {
                d_s[i] += d_m[i];
            }

            let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);

            let retention = 1.0 - alpha_t;
            let mut d_m_prev = vec![0.0f32; d * d];
            for i in 0..(d * d) {
                d_m_prev[i] = retention * d_m[i];
            }

            // ── S_{t+1} = eta * S_t - theta * grad backward ──
            let d_eta_scalar = frobenius_dot_f32(s_t, &d_s);
            let d_theta_scalar = -frobenius_dot_f32(grad_t, &d_s);

            // d_grad = -theta * d_S
            let mut d_grad = vec![0.0f32; d * d];
            for i in 0..(d * d) {
                d_grad[i] = -theta_t * d_s[i];
            }

            // d_S_prev = eta * d_S (propagate to previous step)
            let mut d_s_prev = vec![0.0f32; d * d];
            for i in 0..(d * d) {
                d_s_prev[i] = eta_t * d_s[i];
            }

            // ── grad = outer(error, k) backward ──
            let mut d_err = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_grad[i * d + j] * k_t[j];
                }
                d_err[i] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_grad[i * d + j] * err_t[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // ── error = prediction - v backward ──
            for i in 0..d {
                d_v_mem[t * d + i] -= d_err[i];
            }

            // ── prediction = M_t @ k_t backward ──
            for i in 0..d {
                for j in 0..d {
                    d_m_prev[i * d + j] += d_err[i] * k_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += m_t[i * d + j] * d_err[i];
                }
                d_k_mem[t * d + j] += sum;
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
        MAGConfig::titans_test_config()
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
    fn test_titans_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_titans_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
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
    fn test_titans_forward_momentum_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // S_0 should be zero
        let s_0 = &cache.s_states[0..d * d];
        let s0_norm: f32 = s_0.iter().map(|x| x * x).sum();
        assert!(s0_norm < 1e-12, "S_0 should be zero");

        // S_T should be non-zero (momentum accumulated)
        let s_t = &cache.s_states[s * d * d..(s + 1) * d * d];
        let st_norm: f32 = s_t.iter().map(|x| x * x).sum();
        assert!(st_norm > 1e-12, "S_T should have evolved, norm={st_norm}");
    }

    #[test]
    fn test_titans_forward_eta_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
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
    fn test_titans_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Titans LMM forward should be deterministic");
    }

    #[test]
    fn test_titans_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM;
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
        assert_eq!(cache.s_states.len(), (s + 1) * d * d);
        assert_eq!(cache.eta.len(), s);
    }

    #[test]
    fn test_titans_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM;
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
    fn test_titans_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM;
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
        // eta grads should also be non-zero
        let eta_max = grads.w_eta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(eta_max > 1e-10, "w_eta grads should be non-zero");

        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "d_embedded is all zeros");
    }

    #[test]
    fn test_titans_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM;
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

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_titans_read_only_zero_memory() {
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
    fn test_titans_read_only_nonzero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let mut frozen_m = vec![0.0f32; d * d];
        for i in 0..d { frozen_m[i * d + i] = 1.0; }
        let (y, q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        for i in 0..(s * d) {
            assert!((y[i] - q_mem[i]).abs() < 1e-6, "y[{i}]={} != q_mem[{i}]={}", y[i], q_mem[i]);
        }
    }

    #[test]
    fn test_titans_level_and_parallelization() {
        let rule = TitansLMM;
        assert_eq!(rule.level(), 0);
        assert_eq!(rule.supported_parallelization(), &["sequential"]);
    }
}
