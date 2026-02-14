/// Delta Rule memory system — forward and backward.
///
/// Titans Eq 34: simplest NL memory (no momentum).
/// Sequential per-token: M evolves as tokens are processed.
///
/// Forward:
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   prediction = M_t @ k_t
///   error = prediction - v_t
///   grad = outer(error, k_t)
///   M_{t+1} = (1-alpha_t) * M_t - theta_t * grad
///   y_t = M_{t+1} @ q_t
///
/// Backward: reverse token loop with accumulated d_M.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    outer_product_f32, frobenius_dot_f32,
};
use crate::model::MAGParams;

/// All intermediate values from a Delta Rule forward pass, needed for backward.
pub struct DeltaRuleCache {
    pub seq_len: usize,
    pub d: usize,
    /// Memory matrices M_t for t=0..seq_len (M_0 = zeros, M_t is state BEFORE token t's update)
    /// Layout: [(seq_len+1) * d * d], indexed as m_states[t * d * d .. (t+1) * d * d]
    pub m_states: Vec<f32>,
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
    /// Prediction errors: [seq_len, d]
    pub error: Vec<f32>,
    /// Gradient outer products: [seq_len, d*d]
    pub grad_outer: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
}

/// Delta Rule forward pass (sequential).
///
/// `embedded`: [seq_len, d] — input embeddings (shared with attention branch).
/// Returns (y, cache) where y is [seq_len, d] memory output.
pub fn delta_rule_forward(
    params: &MAGParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
) -> (Vec<f32>, DeltaRuleCache) {
    debug_assert_eq!(embedded.len(), seq_len * d);

    // Project embedded → k_mem, v_mem, q_mem via W^T
    let mut w_k_mem_t = vec![0.0f32; d * d];
    let mut w_v_mem_t = vec![0.0f32; d * d];
    let mut w_q_mem_t = vec![0.0f32; d * d];
    transpose_f32(&params.w_k_mem, &mut w_k_mem_t, d, d);
    transpose_f32(&params.w_v_mem, &mut w_v_mem_t, d, d);
    transpose_f32(&params.w_q_mem, &mut w_q_mem_t, d, d);

    let mut k_mem = vec![0.0f32; seq_len * d];
    let mut v_mem = vec![0.0f32; seq_len * d];
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
    matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
    matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

    // Allocate cache
    let mut m_states = vec![0.0f32; (seq_len + 1) * d * d]; // M_0 = zeros
    let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
    let mut alpha_pre = vec![0.0f32; seq_len];
    let mut alpha = vec![0.0f32; seq_len];
    let mut theta_pre = vec![0.0f32; seq_len];
    let mut theta = vec![0.0f32; seq_len];
    let mut error = vec![0.0f32; seq_len * d];
    let mut grad_outer = vec![0.0f32; seq_len * d * d];
    let mut y = vec![0.0f32; seq_len * d];

    // Sequential token loop
    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let m_t = &m_states[t * d * d..(t + 1) * d * d];

        // Concatenate (k_t, v_t)
        let c_base = t * 2 * d;
        concat_kv[c_base..c_base + d].copy_from_slice(k_t);
        concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
        let concat_t = &concat_kv[c_base..c_base + 2 * d];

        // alpha_t = sigmoid(concat @ w_alpha + b_alpha)
        let mut alpha_pre_t = params.b_alpha[0];
        for i in 0..(2 * d) {
            alpha_pre_t += concat_t[i] * params.w_alpha[i];
        }
        alpha_pre[t] = alpha_pre_t;
        alpha[t] = sigmoid_f32(alpha_pre_t);

        // theta_t = softplus(concat @ w_theta + b_theta)
        let mut theta_pre_t = params.b_theta[0];
        for i in 0..(2 * d) {
            theta_pre_t += concat_t[i] * params.w_theta[i];
        }
        theta_pre[t] = theta_pre_t;
        theta[t] = softplus_f32(theta_pre_t);

        // prediction = M_t @ k_t
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

        // M_{t+1} = (1-alpha_t) * M_t - theta_t * grad
        let retention = 1.0 - alpha[t];
        let lr = theta[t];
        // Copy M_t first to avoid overlapping borrows on m_states
        for i in 0..(d * d) {
            m_states[(t + 1) * d * d + i] = retention * m_states[t * d * d + i]
                - lr * grad_outer[g_base + i];
        }

        // y_t = M_{t+1} @ q_t
        let m_next = &m_states[(t + 1) * d * d..(t + 2) * d * d];
        matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
    }

    let y_out = y.clone();

    let cache = DeltaRuleCache {
        seq_len, d, m_states, k_mem, v_mem, q_mem, concat_kv,
        alpha_pre, alpha, theta_pre, theta, error, grad_outer, y,
    };

    (y_out, cache)
}

/// Delta Rule backward pass.
///
/// `d_y`: [seq_len, d] — upstream gradient on memory output y.
/// `embedded`: [seq_len, d] — input embeddings (for projection backward).
///
/// Returns gradient on MAGParams memory fields and d_embedded contribution.
pub fn delta_rule_backward(
    params: &MAGParams,
    cache: &DeltaRuleCache,
    d_y: &[f32],
    embedded: &[f32],
) -> (MAGParams, Vec<f32>) {
    let s = cache.seq_len;
    let d = cache.d;
    debug_assert_eq!(d_y.len(), s * d);
    debug_assert_eq!(embedded.len(), s * d);

    // Initialize gradient accumulators
    let mut grads = crate::model::MAGParams::zeros_like(&crate::model::MAGConfig {
        swa: crate::model::SWAConfig {
            d_model: d,
            num_heads: 0, // not used for zeros_like sizing
            head_dim: 0,
            seq_len: 0,
            window_size: 0,
            vocab_size: 0,
        },
        memory_enabled: true,
    });
    // Fix: zeros_like uses d for sizing memory weights but SWA parts need proper config.
    // We only accumulate into memory fields here; SWA grads come from mag.rs.
    grads.w_k_mem = vec![0.0f32; d * d];
    grads.w_v_mem = vec![0.0f32; d * d];
    grads.w_q_mem = vec![0.0f32; d * d];
    grads.w_alpha = vec![0.0f32; 2 * d];
    grads.b_alpha = vec![0.0f32; 1];
    grads.w_theta = vec![0.0f32; 2 * d];
    grads.b_theta = vec![0.0f32; 1];

    // Gradient buffers for projected memory k/v/q
    let mut d_k_mem = vec![0.0f32; s * d];
    let mut d_v_mem = vec![0.0f32; s * d];
    let mut d_q_mem = vec![0.0f32; s * d];

    // d_M: accumulated gradient on memory state, starts from zero after last token
    let mut d_m = vec![0.0f32; d * d];

    // Reverse token loop
    for t in (0..s).rev() {
        let k_t = &cache.k_mem[t * d..(t + 1) * d];
        let _v_t = &cache.v_mem[t * d..(t + 1) * d];
        let q_t = &cache.q_mem[t * d..(t + 1) * d];
        let m_t = &cache.m_states[t * d * d..(t + 1) * d * d];
        let m_next = &cache.m_states[(t + 1) * d * d..(t + 2) * d * d];
        let err_t = &cache.error[t * d..(t + 1) * d];
        let grad_t = &cache.grad_outer[t * d * d..(t + 1) * d * d];
        let c_base = t * 2 * d;
        let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
        let alpha_t = cache.alpha[t];
        let theta_t = cache.theta[t];
        let _alpha_pre_t = cache.alpha_pre[t];
        let theta_pre_t = cache.theta_pre[t];

        // ── y_t = M_{t+1} @ q_t backward ──
        // d_M_{t+1} += outer(d_y_t, q_t)    (from y_t = M @ q)
        // d_q_t = M_{t+1}^T @ d_y_t
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
                sum += m_next[j * d + i] * d_y_t[j]; // M^T[i,j] = M[j,i]
            }
            d_q_mem[t * d + i] = sum;
        }

        // ── M_{t+1} = (1-alpha) * M_t - theta * grad backward ──
        // d_alpha_t = -frobenius(d_M, M_t)  (the -(−1) from derivative of (1-alpha))
        let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);
        // Actually: d/d_alpha of (1-alpha)*M_t = -M_t, so d_alpha = frobenius(d_M, -M_t)
        // = -frobenius(d_M, M_t). Correct.

        // d_theta_t = -frobenius(d_M, grad_t)
        let d_theta_scalar = -frobenius_dot_f32(&d_m, grad_t);

        // d_grad = -theta * d_M
        let mut d_grad = vec![0.0f32; d * d];
        for i in 0..(d * d) {
            d_grad[i] = -theta_t * d_m[i];
        }

        // d_M_t = (1 - alpha) * d_M  (propagate to previous step)
        let retention = 1.0 - alpha_t;
        let mut d_m_prev = vec![0.0f32; d * d];
        for i in 0..(d * d) {
            d_m_prev[i] = retention * d_m[i];
        }

        // ── grad = outer(error, k) backward ──
        // d_err_t[i] = sum_j d_grad[i,j] * k_t[j]
        // d_k_t[j] += sum_i d_grad[i,j] * err_t[i]
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
        // d_prediction = d_err, d_v_mem -= d_err
        for i in 0..d {
            d_v_mem[t * d + i] -= d_err[i];
        }

        // ── prediction = M_t @ k_t backward ──
        // d_M_t += outer(d_prediction, k_t)  — but d_prediction = d_err
        // d_k_t += M_t^T @ d_prediction
        for i in 0..d {
            for j in 0..d {
                d_m_prev[i * d + j] += d_err[i] * k_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_t[i * d + j] * d_err[i]; // M_t^T[j,i] * d_err[i]
            }
            d_k_mem[t * d + j] += sum;
        }

        // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
        let sig_deriv = alpha_t * (1.0 - alpha_t);
        let d_alpha_pre = d_alpha_scalar * sig_deriv;

        // ── Gate backward: theta_t = softplus(theta_pre_t) ──
        // softplus'(x) = sigmoid(x)
        let softplus_deriv = sigmoid_f32(theta_pre_t);
        let d_theta_pre = d_theta_scalar * softplus_deriv;

        // ── w_alpha, b_alpha gradient ──
        // alpha_pre = concat @ w_alpha + b_alpha
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
        // d_concat = d_alpha_pre * w_alpha + d_theta_pre * w_theta
        for i in 0..d {
            d_k_mem[t * d + i] += d_alpha_pre * params.w_alpha[i]
                                + d_theta_pre * params.w_theta[i];
        }
        for i in 0..d {
            d_v_mem[t * d + i] += d_alpha_pre * params.w_alpha[d + i]
                                + d_theta_pre * params.w_theta[d + i];
        }

        // Update d_m for next (earlier) token
        d_m = d_m_prev;
    }

    // ── Projection backward: k_mem = embedded @ W_K_mem^T ──
    // d_W_K_mem = d_k_mem^T @ embedded
    // d_embedded += d_k_mem @ W_K_mem (and similarly for v, q)
    let mut d_embedded = vec![0.0f32; s * d];

    // d_W_K_mem
    let mut d_k_mem_t = vec![0.0f32; d * s];
    transpose_f32(&d_k_mem, &mut d_k_mem_t, s, d);
    matmul_f32(&d_k_mem_t, embedded, &mut grads.w_k_mem, d, s, d);

    // d_W_V_mem
    let mut d_v_mem_t = vec![0.0f32; d * s];
    transpose_f32(&d_v_mem, &mut d_v_mem_t, s, d);
    matmul_f32(&d_v_mem_t, embedded, &mut grads.w_v_mem, d, s, d);

    // d_W_Q_mem
    let mut d_q_mem_t = vec![0.0f32; d * s];
    transpose_f32(&d_q_mem, &mut d_q_mem_t, s, d);
    matmul_f32(&d_q_mem_t, embedded, &mut grads.w_q_mem, d, s, d);

    // d_embedded += d_k_mem @ W_K_mem + d_v_mem @ W_V_mem + d_q_mem @ W_Q_mem
    crate::tensor::matmul_acc_f32(&d_k_mem, &params.w_k_mem, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_v_mem, &params.w_v_mem, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_q_mem, &params.w_q_mem, &mut d_embedded, s, d, d);

    (grads, d_embedded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;

    fn test_config() -> MAGConfig {
        MAGConfig::test_config()
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
    fn test_delta_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let (y, _cache) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_delta_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let (_y, cache) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // M_0 is zeros, M_T should be non-zero
        let m_0 = &cache.m_states[0..d * d];
        let m_t = &cache.m_states[s * d * d..(s + 1) * d * d];
        let m0_norm: f32 = m_0.iter().map(|x| x * x).sum();
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_delta_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let (_y, cache) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            let th = cache.theta[t];
            assert!(th >= 0.0, "theta[{t}]={th} should be non-negative");
        }
    }

    #[test]
    fn test_delta_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let (y1, _) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        let (y2, _) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        assert_eq!(y1, y2, "Delta rule forward should be deterministic");
    }

    #[test]
    fn test_delta_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let (y, cache) = delta_rule_forward(&params, &embedded, cfg.swa.seq_len, cfg.swa.d_model);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
    }

    #[test]
    fn test_delta_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let (_y, cache) = delta_rule_forward(&params, &embedded, s, d);

        // Simple upstream gradient: all ones
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = delta_rule_backward(&params, &cache, &d_y, &embedded);

        // Check all gradients finite
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
    fn test_delta_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let (_y, cache) = delta_rule_forward(&params, &embedded, s, d);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = delta_rule_backward(&params, &cache, &d_y, &embedded);

        // Key memory weights should have non-zero gradients
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
    fn test_delta_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let (_y, cache) = delta_rule_forward(&params, &embedded, s, d);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = delta_rule_backward(&params, &cache, &d_y, &embedded);

        assert_eq!(grads.w_k_mem.len(), d * d);
        assert_eq!(grads.w_v_mem.len(), d * d);
        assert_eq!(grads.w_q_mem.len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
        assert_eq!(grads.w_theta.len(), 2 * d);
        assert_eq!(grads.b_theta.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }
}
