/// Delta Rule memory system — trait-based implementation.
///
/// Implements the MemoryRule trait from the MIRAS framework (spec 00_interface.md).
/// Titans Eq 34: simplest NL memory (no momentum).
/// Sequential per-token: M evolves as tokens are processed.
///
/// Forward (per token):
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
use crate::retention::l2_apply_retention;
use crate::model::MemoryLevelParams;

// ── Memory rule trait ────────────────────────────────────────────────

/// Per-token memory state: wraps the d×d memory matrix M.
pub struct MemoryState {
    /// Flat [d, d] memory matrix in row-major layout.
    pub m: Vec<f32>,
    pub d: usize,
}

/// Data-dependent gates for a single token.
pub struct Gates {
    /// Retention gate: sigmoid output in (0, 1).
    pub alpha: f32,
    /// Learning rate: softplus output, non-negative.
    pub theta: f32,
}

/// Error type for memory operations that may not be supported by all rules.
///
/// MLP-family rules (MONETA, YAAD, MEMORA) fuse write+read into `step()` because
/// their 2-layer MLP structure doesn't fit the per-token matrix API (M @ q).
/// Instead of panicking at runtime, these rules return `Err(UnsupportedOperation)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// The memory rule does not support this operation — use `step()` instead.
    UnsupportedOperation,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::UnsupportedOperation =>
                write!(f, "this memory rule does not support direct write/read — use step() instead"),
        }
    }
}

/// The core abstraction from the MIRAS framework (spec 00_interface.md).
///
/// Every named memory variant (Delta Rule, Titans LMM, MONETA, etc.)
/// implements this trait. Static dispatch only — Enzyme requires
/// monomorphization, not vtable indirection.
pub trait MemoryRule {
    /// Cache type for backward pass.
    type Cache;

    /// CMS frequency level index. Level 0 fires every token.
    fn level(&self) -> usize;

    /// Which parallelization strategies this rule supports.
    fn supported_parallelization(&self) -> &'static [&'static str];

    /// Create initial memory state (M_0 = zeros).
    fn init(&self, d: usize) -> MemoryState;

    /// Per-token WRITE: mutate memory state given projected k, v and gates.
    /// Does NOT produce output — use `read` after write.
    ///
    /// Returns `Err(MemoryError::UnsupportedOperation)` for MLP-family rules
    /// (MONETA, YAAD, MEMORA) where write is fused into `step()`.
    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError>;

    /// Per-token READ: query memory with q, write result to `out`.
    /// Pure function — does NOT mutate state.
    ///
    /// Returns `Err(MemoryError::UnsupportedOperation)` for MLP-family rules
    /// (MONETA, YAAD, MEMORA) where read is fused into `step()`.
    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) -> Result<(), MemoryError>;

    /// Full sequence forward: project → gate → write → read for all tokens.
    /// Returns (output [seq_len, d], cache for backward).
    ///
    /// `initial_m`: Optional initial memory state. If None, starts from zeros.
    /// Used by CMS to seed active levels with persisted context memory.
    /// Ownership is transferred so the callee can reuse the allocation.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, Self::Cache);

    /// Full sequence backward through the memory rule.
    /// Returns (level param gradients, d_embedded contribution).
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &Self::Cache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>);
}

// ── Delta Rule implementation ────────────────────────────────────────

/// Delta Rule: simplest NL memory — GD without momentum (Titans Eq 34).
///
/// MIRAS knobs: matrix structure, L2 bias, L2 decay retention, GD algorithm.
pub struct DeltaRule;

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

impl MemoryRule for DeltaRule {
    type Cache = DeltaRuleCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        crate::parallel::supported_strategies(crate::model::MemoryRuleKind::DeltaRule)
    }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError> {
        let d = state.d;
        // prediction = M @ k
        let mut prediction = vec![0.0f32; d];
        matmul_f32(&state.m, k, &mut prediction, d, d, 1);

        // error = prediction - v; grad = outer(error, k)
        // M = (1-alpha) * M - theta * outer(error, k)
        let lr = gates.theta;
        l2_apply_retention(&mut state.m, 1.0 - gates.alpha);
        for i in 0..d {
            let err_i = prediction[i] - v[i];
            for j in 0..d {
                state.m[i * d + j] -= lr * err_i * k[j];
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
    /// `embedded`: [seq_len, d] — input embeddings (shared with attention branch).
    /// Returns (y, cache) where y is [seq_len, d] memory output.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, DeltaRuleCache) {
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
        let mut m_states = vec![0.0f32; (seq_len + 1) * d * d];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), d * d);
            m_states[..d * d].copy_from_slice(&m0);
        }
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

            // M_{t+1} = (1-alpha_t) * M_t - theta_t * grad
            let lr = theta[t];
            let m_next_off = (t + 1) * d * d;
            let m_t_off = t * d * d;
            m_states.copy_within(m_t_off..m_t_off + d * d, m_next_off);
            l2_apply_retention(&mut m_states[m_next_off..m_next_off + d * d], 1.0 - alpha[t]);
            for i in 0..(d * d) {
                m_states[m_next_off + i] -= lr * grad_outer[g_base + i];
            }

            // y_t = M_{t+1} @ q_t
            let m_next = &m_states[(t + 1) * d * d..(t + 2) * d * d];
            matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = DeltaRuleCache {
            seq_len, d, m_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, error, grad_outer, y: y.clone(),
        };

        (y, cache)
    }

    /// Full sequence backward through the Delta Rule memory.
    ///
    /// `d_y`: [seq_len, d] — upstream gradient on memory output y.
    /// `embedded`: [seq_len, d] — input embeddings (for projection backward).
    ///
    /// Returns (MemoryLevelParams gradients, d_embedded contribution).
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &DeltaRuleCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        // Initialize gradient accumulators for memory level weights only.
        let mut grads = MemoryLevelParams::zeros_like(d);

        // Gradient buffers for projected memory k/v/q
        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // d_M: accumulated gradient on memory state, starts from zero after last token
        let mut d_m = vec![0.0f32; d * d];

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let m_t = &cache.m_states[t * d * d..(t + 1) * d * d];
            let m_next = &cache.m_states[(t + 1) * d * d..(t + 2) * d * d];
            let err_t = &cache.error[t * d..(t + 1) * d];
            let grad_t = &cache.grad_outer[t * d * d..(t + 1) * d * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let theta_pre_t = cache.theta_pre[t];

            // ── y_t = M_{t+1} @ q_t backward ──
            // d_M_{t+1} += outer(d_y_t, q_t)
            // d_q_t = M_{t+1}^T @ d_y_t
            let d_y_t = &d_y[t * d..(t + 1) * d];

            for i in 0..d {
                for j in 0..d {
                    d_m[i * d + j] += d_y_t[i] * q_t[j];
                }
            }

            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_next[j * d + i] * d_y_t[j];
                }
                d_q_mem[t * d + i] = sum;
            }

            // ── M_{t+1} = (1-alpha) * M_t - theta * grad backward ──
            let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);
            let d_theta_scalar = -frobenius_dot_f32(&d_m, grad_t);

            // d_grad = -theta * d_M
            let mut d_grad = vec![0.0f32; d * d];
            for i in 0..(d * d) {
                d_grad[i] = -theta_t * d_m[i];
            }

            // d_M_t = (1 - alpha) * d_M  (propagate to previous step)
            let mut d_m_prev = d_m.clone();
            l2_apply_retention(&mut d_m_prev, 1.0 - alpha_t);

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

            // Update d_m for next (earlier) token
            d_m = d_m_prev;
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

/// Forward pass for a frozen (read-only) level: uses persisted M without writing.
///
/// For each token: y_t = M @ q_t, where M is the frozen memory matrix.
/// Returns (y [seq_len, d], projected q_mem [seq_len, d]).
pub fn delta_rule_read_only(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    frozen_m: &[f32],
    seq_len: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(frozen_m.len(), d * d);

    // Project embedded → q_mem via W_Q_mem^T
    let mut w_q_mem_t = vec![0.0f32; d * d];
    transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

    // y_t = M @ q_t for each token
    let mut y = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        matmul_f32(frozen_m, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
    }

    (y, q_mem)
}

/// Backward pass for a frozen (read-only) level.
///
/// Only d_embedded (through W_Q_mem) flows back — no memory weight gradients
/// because the frozen level doesn't write. But we DO compute d_W_Q_mem gradients
/// since the query projection parameters are still learnable.
///
/// Returns (MemoryLevelParams grads [only w_q_mem populated], d_embedded contribution).
pub fn delta_rule_read_only_backward(
    level_params: &MemoryLevelParams,
    frozen_m: &[f32],
    _q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
) -> (MemoryLevelParams, Vec<f32>) {
    debug_assert_eq!(d_y.len(), seq_len * d);

    let mut grads = MemoryLevelParams::zeros_like(d);

    // y_t = M @ q_t → d_q_t = M^T @ d_y_t
    let mut d_q_mem = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let d_y_t = &d_y[t * d..(t + 1) * d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += frozen_m[j * d + i] * d_y_t[j];
            }
            d_q_mem[t * d + i] = sum;
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
        let rule = DeltaRule;
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_delta_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = DeltaRule;
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
    fn test_delta_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = DeltaRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
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
        let rule = DeltaRule;
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Delta rule forward should be deterministic");
    }

    #[test]
    fn test_delta_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = DeltaRule;
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
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
        let rule = DeltaRule;
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
    fn test_delta_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = DeltaRule;
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
    fn test_delta_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = DeltaRule;
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
    fn test_delta_init() {
        let rule = DeltaRule;
        let state = rule.init(8);
        assert_eq!(state.m.len(), 64);
        assert_eq!(state.d, 8);
        assert!(state.m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_delta_write_read() {
        let rule = DeltaRule;
        let mut state = rule.init(4);
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.0f32, 1.0, 0.0, 0.0];
        let gates = Gates { alpha: 0.0, theta: 1.0 };

        // Write: M = 0 - 1.0 * outer(0*k - v, k) = outer(v, k)
        // prediction = M@k = 0, error = 0-v = -v, grad = outer(-v, k)
        // M_new = (1-0)*M - 1.0*outer(-v,k) = outer(v, k)
        rule.write(&mut state, &k, &v, &gates).unwrap();

        // Read: M @ q should give v[0]*q for q aligned with k
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        rule.read(&state, &q, &mut out).unwrap();
        // M = outer(v, k) = [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
        // M @ q = [0, 1, 0, 0]
        assert!((out[1] - 1.0).abs() < 1e-6, "read should return stored value");
    }

    #[test]
    fn test_delta_level_and_parallelization() {
        let rule = DeltaRule;
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"chunkwise_gd"));
        assert!(strategies.contains(&"tnt"));
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_read_only_zero_memory() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let frozen_m = vec![0.0f32; d * d];
        let (y, _q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        // Zero memory → zero output
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_read_only_nonzero_memory() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        // Identity memory → y_t = q_t
        let mut frozen_m = vec![0.0f32; d * d];
        for i in 0..d { frozen_m[i * d + i] = 1.0; }
        let (y, q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        // y should equal q_mem when M = I
        for i in 0..(s * d) {
            assert!((y[i] - q_mem[i]).abs() < 1e-6, "y[{i}]={} != q_mem[{i}]={}", y[i], q_mem[i]);
        }
    }
}
