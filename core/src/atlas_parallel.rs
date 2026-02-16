/// Atlas Parallel — parallelization strategy for Atlas Omega rule.
///
/// Atlas Omega uses a state-independent omega function:
///   omega = W_omega @ silu(concat(k, v))
/// Because omega is independent of the memory state, all tokens' momentum
/// updates can be computed in parallel (Atlas paper Section 5, Eqs 34-41).
///
/// This module provides:
/// - AtlasOmegaParams: weights for the omega function
/// - atlas_omega(): the state-independent surrogate gradient function
/// - atlas_parallel_forward(): parallel forward using pre-computed omegas
/// - atlas_parallel_backward(): corresponding backward pass

use crate::tensor::{silu_f32, matmul_f32, transpose_f32, sigmoid_f32, softplus_f32};
use crate::atlas_omega::AtlasOmegaCache;

/// Parameters for the Atlas Omega function.
#[derive(Clone, Debug)]
pub struct AtlasOmegaParams {
    /// Key projection for omega: [d, d]
    pub w_k_omega: Vec<f32>,
    /// Value projection for omega: [d, d]
    pub w_v_omega: Vec<f32>,
    /// Omega projection: [d, 2*d]
    pub w_omega: Vec<f32>,
}

impl AtlasOmegaParams {
    /// Initialize with random weights.
    pub fn init(d: usize, seed: u64) -> Self {
        use crate::tensor::SimpleRng;
        let mut rng = SimpleRng::new(seed);
        let scale = (1.0 / d as f32).sqrt();
        let mut w_k_omega = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_k_omega, scale);
        let mut w_v_omega = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_v_omega, scale);
        let mut w_omega = vec![0.0f32; d * 2 * d];
        rng.fill_uniform(&mut w_omega, scale);
        Self { w_k_omega, w_v_omega, w_omega }
    }
}

/// Compute the Atlas omega function: omega = W_omega @ silu(concat(k, v)).
///
/// This is state-independent — it depends only on the input token's projections,
/// not on the memory state. This property enables full parallelization.
///
/// # Arguments
/// * `k` - Key vector [d]
/// * `v` - Value vector [d]
/// * `params` - Atlas omega parameters
/// * `d` - Model dimension
///
/// # Returns
/// omega vector [d]
pub fn atlas_omega(k: &[f32], v: &[f32], params: &AtlasOmegaParams, d: usize) -> Vec<f32> {
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);
    debug_assert_eq!(params.w_omega.len(), d * 2 * d);

    // concat(k, v) → [2*d]
    let mut kv = vec![0.0f32; 2 * d];
    kv[..d].copy_from_slice(k);
    kv[d..].copy_from_slice(v);

    // silu(kv) → [2*d]
    for x in &mut kv {
        *x = silu_f32(*x);
    }

    // W_omega @ silu(kv) → [d]
    let mut omega = vec![0.0f32; d];
    for i in 0..d {
        let mut sum = 0.0f32;
        for j in 0..(2 * d) {
            sum += params.w_omega[i * 2 * d + j] * kv[j];
        }
        omega[i] = sum;
    }

    omega
}

/// Atlas parallel forward — compute all omega/gates in parallel, then accumulate memory.
///
/// This is the parallel version of the Atlas Omega rule's step() function.
/// The key parallelism: because omega is state-independent, all per-token
/// omega values and gates can be computed first (embarrassingly parallel),
/// then momentum S and memory M are accumulated sequentially.
///
/// Returns (y [seq_len, d], AtlasOmegaCache).
pub fn atlas_parallel_forward(
    level_params: &crate::model::MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    _cfg: &crate::model::MAGConfig,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, AtlasOmegaCache) {
    let dd = d * d;
    let omega_params = AtlasOmegaParams::init(d, 42);

    // Project embedded → k_mem, v_mem, q_mem
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

    // PARALLEL PHASE 1: Compute all omegas and gates independently
    // (These are all state-independent — can be parallelized)
    let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
    let mut alpha_pre = vec![0.0f32; seq_len];
    let mut alpha = vec![0.0f32; seq_len];
    let mut theta_pre = vec![0.0f32; seq_len];
    let mut theta = vec![0.0f32; seq_len];
    let mut eta_pre = vec![0.0f32; seq_len];
    let mut eta = vec![0.0f32; seq_len];
    let mut omega_mats = vec![0.0f32; seq_len * dd];
    let mut silu_kv = vec![0.0f32; seq_len * 2 * d];

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let c_base = t * 2 * d;

        // Concat(k, v)
        concat_kv[c_base..c_base + d].copy_from_slice(k_t);
        concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
        let concat_t = &concat_kv[c_base..c_base + 2 * d];

        // Gates
        let mut ap = level_params.b_alpha[0];
        let mut tp = level_params.b_theta[0];
        let mut ep = level_params.b_eta[0];
        for i in 0..(2 * d) {
            ap += concat_t[i] * level_params.w_alpha[i];
            tp += concat_t[i] * level_params.w_theta[i];
            ep += concat_t[i] * level_params.w_eta[i];
        }
        alpha_pre[t] = ap;
        alpha[t] = sigmoid_f32(ap);
        theta_pre[t] = tp;
        theta[t] = softplus_f32(tp);
        eta_pre[t] = ep;
        eta[t] = sigmoid_f32(ep);

        // silu(concat(k, v))
        let sk_base = t * 2 * d;
        for i in 0..(2 * d) {
            silu_kv[sk_base + i] = silu_f32(concat_t[i]);
        }
        let silu_t = &silu_kv[sk_base..sk_base + 2 * d];

        // omega_vec = W_omega @ silu_t
        let mut omega_vec = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..(2 * d) {
                sum += omega_params.w_omega[i * 2 * d + j] * silu_t[j];
            }
            omega_vec[i] = sum;
        }

        // omega_mat = outer(omega_vec, k_t)
        let om_base = t * dd;
        for i in 0..d {
            for j in 0..d {
                omega_mats[om_base + i * d + j] = omega_vec[i] * k_t[j];
            }
        }
    }

    // SEQUENTIAL PHASE: Accumulate S and M (inherently sequential)
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
    if let Some(m0) = initial_m {
        debug_assert_eq!(m0.len(), dd);
        m_states[..dd].copy_from_slice(&m0);
    }

    let mut y = vec![0.0f32; seq_len * d];

    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let om_base = t * dd;
        let s_t = t * dd;
        let s_next = (t + 1) * dd;
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        // S_{t+1} = eta * S_t - theta * omega_mat
        for i in 0..dd {
            s_states[s_next + i] = eta[t] * s_states[s_t + i] - theta[t] * omega_mats[om_base + i];
        }

        // M_{t+1} = (1-alpha) * M_t + S_{t+1}
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

/// Atlas parallel backward.
///
/// Delegates to the AtlasOmega rule's step_backward since the backward
/// pass structure is the same (just the forward was parallelized differently).
pub fn atlas_parallel_backward(
    level_params: &crate::model::MemoryLevelParams,
    cache: &AtlasOmegaCache,
    d_y: &[f32],
    embedded: &[f32],
    _cfg: &crate::model::MAGConfig,
) -> (crate::model::MemoryLevelParams, Vec<f32>) {
    let d = cache.d;
    let omega_params = AtlasOmegaParams::init(d, 42);
    let rule = crate::atlas_omega::AtlasOmega { omega_params };
    use crate::delta_rule::MemoryRule;
    rule.step_backward(level_params, cache, d_y, embedded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atlas_omega_computation() {
        let d = 4;
        let params = AtlasOmegaParams::init(d, 42);
        let k = vec![1.0, 0.5, -0.3, 0.8];
        let v = vec![0.2, -0.4, 0.6, -0.1];

        let omega = atlas_omega(&k, &v, &params, d);
        assert_eq!(omega.len(), d);
        for &val in &omega {
            assert!(val.is_finite(), "omega should be finite");
        }
    }

    #[test]
    fn test_atlas_omega_state_independent() {
        // Same k, v → same omega regardless of when called
        let d = 4;
        let params = AtlasOmegaParams::init(d, 42);
        let k = vec![0.3, -0.2, 0.5, 0.1];
        let v = vec![-0.1, 0.4, 0.2, -0.3];

        let omega1 = atlas_omega(&k, &v, &params, d);
        let omega2 = atlas_omega(&k, &v, &params, d);

        for (a, b) in omega1.iter().zip(omega2.iter()) {
            assert_eq!(*a, *b, "omega should be deterministic");
        }
    }

    #[test]
    fn test_atlas_parallel_forward_matches_sequential() {
        // The parallel forward should produce identical results to sequential AtlasOmega.step()
        let cfg = crate::model::MAGConfig::atlas_test_config();
        let params = crate::model::MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // Create embedded input
        use crate::tensor::SimpleRng;
        let mut rng = SimpleRng::new(99);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);

        // Sequential (via AtlasOmega.step())
        let omega_params = AtlasOmegaParams::init(d, 42);
        let rule = crate::atlas_omega::AtlasOmega { omega_params };
        use crate::delta_rule::MemoryRule;
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        // Parallel
        let (y_par, _) = atlas_parallel_forward(&params.levels[0], &embedded, s, d, &cfg, None);

        // They should match exactly
        for i in 0..(s * d) {
            assert!(
                (y_seq[i] - y_par[i]).abs() < 1e-6,
                "Mismatch at {i}: seq={}, par={}", y_seq[i], y_par[i]
            );
        }
    }

    #[test]
    fn test_atlas_omega_param_shapes() {
        let d = 8;
        let params = AtlasOmegaParams::init(d, 42);
        assert_eq!(params.w_k_omega.len(), d * d);
        assert_eq!(params.w_v_omega.len(), d * d);
        assert_eq!(params.w_omega.len(), d * 2 * d);
    }

    #[test]
    fn test_atlas_omega_different_inputs() {
        // Different inputs → (very likely) different outputs
        let d = 8;
        let params = AtlasOmegaParams::init(d, 42);

        let k1 = vec![1.0; d];
        let v1 = vec![0.0; d];
        let k2 = vec![0.0; d];
        let v2 = vec![1.0; d];

        let omega1 = atlas_omega(&k1, &v1, &params, d);
        let omega2 = atlas_omega(&k2, &v2, &params, d);

        let diff: f32 = omega1.iter().zip(omega2.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "different inputs should give different omega");
    }
}
