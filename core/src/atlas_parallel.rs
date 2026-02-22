/// Atlas Parallel — parallelization strategy for Atlas Omega rule.
///
/// Because omega(k,v) is state-independent (doesn't depend on M), ALL tokens'
/// omega values can be precomputed in parallel. The sequential recurrence only
/// applies to M and S updates, which use precomputed omega_mats.
///
/// This module provides the parallel forward/backward that processes omega
/// computation in batch, then runs the sequential M/S recurrence.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    outer_product_f32, silu_f32,
};
use crate::retention::l2_apply_retention;
use crate::delta_rule::MemoryRule;
use crate::atlas_omega::AtlasOmegaCache;

/// Compute all omega vectors in parallel (batch version).
///
/// Given projected k_mem [seq_len, d] and v_mem [seq_len, d],
/// computes omega_t = W_omega @ silu(concat(k_t, v_t)) for all t.
///
/// Returns (omega_vecs [seq_len, d], silu_kv [seq_len, 2*d]).
pub fn batch_compute_omega(
    k_mem: &[f32],
    v_mem: &[f32],
    w_omega: &[f32],
    seq_len: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(k_mem.len(), seq_len * d);
    debug_assert_eq!(v_mem.len(), seq_len * d);
    debug_assert_eq!(w_omega.len(), d * 2 * d);

    // Build silu_kv [seq_len, 2*d] = silu(concat(k, v))
    let mut silu_kv = vec![0.0f32; seq_len * 2 * d];
    for t in 0..seq_len {
        for i in 0..d {
            silu_kv[t * 2 * d + i] = silu_f32(k_mem[t * d + i]);
        }
        for i in 0..d {
            silu_kv[t * 2 * d + d + i] = silu_f32(v_mem[t * d + i]);
        }
    }

    // omega_vecs [seq_len, d] = silu_kv [seq_len, 2*d] @ W_omega^T [2*d, d]
    let mut w_omega_t = vec![0.0f32; 2 * d * d];
    transpose_f32(w_omega, &mut w_omega_t, d, 2 * d);
    let mut omega_vecs = vec![0.0f32; seq_len * d];
    matmul_f32(&silu_kv, &w_omega_t, &mut omega_vecs, seq_len, 2 * d, d);

    (omega_vecs, silu_kv)
}

/// Atlas parallel forward pass.
///
/// Phase 1 (parallel): precompute all projections, gates, and omega values.
/// Phase 2 (sequential): run the M/S recurrence using precomputed omega_mats.
pub fn atlas_parallel_forward(
    level_params: &crate::model::MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    _cfg: &crate::model::MAGConfig,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, AtlasOmegaCache) {
    debug_assert_eq!(embedded.len(), seq_len * d);

    // ── Phase 1: Parallel batch projections ──
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

    // Apply Conv1D preprocessing to k/q (must match atlas_omega::step)
    let (k_conv_cache, q_conv_cache) = crate::conv1d::apply_conv1d_to_kq(
        &mut k_mem, &mut q_mem, level_params, seq_len, d);

    // Batch compute omega (the key parallelizable operation)
    let (omega_vecs, silu_kv) = batch_compute_omega(&k_mem, &v_mem, &level_params.w_omega, seq_len, d);

    // Batch compute gates
    let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
    let mut alpha_pre = vec![0.0f32; seq_len];
    let mut alpha = vec![0.0f32; seq_len];
    let mut theta_pre = vec![0.0f32; seq_len];
    let mut theta = vec![0.0f32; seq_len];
    let mut eta_pre = vec![0.0f32; seq_len];
    let mut eta = vec![0.0f32; seq_len];

    for t in 0..seq_len {
        let c_base = t * 2 * d;
        concat_kv[c_base..c_base + d].copy_from_slice(&k_mem[t * d..(t + 1) * d]);
        concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(&v_mem[t * d..(t + 1) * d]);
        let concat_t = &concat_kv[c_base..c_base + 2 * d];

        let mut a_pre = level_params.b_alpha[0];
        let mut t_pre = level_params.b_theta[0];
        let mut e_pre = level_params.b_eta[0];
        for i in 0..(2 * d) {
            a_pre += concat_t[i] * level_params.w_alpha[i];
            t_pre += concat_t[i] * level_params.w_theta[i];
            e_pre += concat_t[i] * level_params.w_eta[i];
        }
        alpha_pre[t] = a_pre;
        alpha[t] = sigmoid_f32(a_pre);
        theta_pre[t] = t_pre;
        theta[t] = softplus_f32(t_pre);
        eta_pre[t] = e_pre;
        eta[t] = sigmoid_f32(e_pre);
    }

    // Batch compute omega outer products
    let mut omega_mats = vec![0.0f32; seq_len * d * d];
    for t in 0..seq_len {
        outer_product_f32(
            &omega_vecs[t * d..(t + 1) * d],
            &k_mem[t * d..(t + 1) * d],
            &mut omega_mats[t * d * d..(t + 1) * d * d],
        );
    }

    // ── Phase 2: Sequential M/S recurrence ──
    let mut m_states = vec![0.0f32; (seq_len + 1) * d * d];
    let mut s_states = vec![0.0f32; (seq_len + 1) * d * d];
    if let Some(m0) = initial_m {
        debug_assert_eq!(m0.len(), d * d);
        m_states[..d * d].copy_from_slice(&m0);
    }
    let mut y = vec![0.0f32; seq_len * d];

    for t in 0..seq_len {
        let g_base = t * d * d;
        let s_t_off = t * d * d;
        let s_next_off = (t + 1) * d * d;

        // S_{t+1} = eta_t * S_t - theta_t * omega_mat_t
        s_states.copy_within(s_t_off..s_t_off + d * d, s_next_off);
        l2_apply_retention(&mut s_states[s_next_off..s_next_off + d * d], eta[t]);
        for i in 0..(d * d) {
            s_states[s_next_off + i] -= theta[t] * omega_mats[g_base + i];
        }

        // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}
        let m_t_off = t * d * d;
        let m_next_off = (t + 1) * d * d;
        m_states.copy_within(m_t_off..m_t_off + d * d, m_next_off);
        l2_apply_retention(&mut m_states[m_next_off..m_next_off + d * d], 1.0 - alpha[t]);
        for i in 0..(d * d) {
            m_states[m_next_off + i] += s_states[s_next_off + i];
        }

        // y_t = M_{t+1} @ q_t
        let m_next = &m_states[m_next_off..m_next_off + d * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
    }

    let cache = AtlasOmegaCache {
        seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
        alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
        silu_kv, omega_vecs, omega_mats, y: y.clone(),
        k_conv_cache,
        q_conv_cache,
    };

    (y, cache)
}

/// Atlas parallel backward pass.
///
/// Identical to AtlasOmega::step_backward() since the backward
/// recurrence is inherently sequential (d_M and d_S propagate backward).
pub fn atlas_parallel_backward(
    level_params: &crate::model::MemoryLevelParams,
    cache: &AtlasOmegaCache,
    d_y: &[f32],
    embedded: &[f32],
) -> (crate::model::MemoryLevelParams, Vec<f32>) {
    // Reuse the sequential backward — both produce identical gradients.
    crate::atlas_omega::AtlasOmega.step_backward(level_params, cache, d_y, embedded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;

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
    fn test_batch_omega_matches_sequential() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // Sequential via MemoryRule::step
        let rule = crate::atlas_omega::AtlasOmega;
        let (y_seq, cache_seq) = crate::delta_rule::MemoryRule::step(
            &rule, &params.levels[0], &embedded, s, d, None,
        );

        // Parallel via atlas_parallel_forward
        let (y_par, cache_par) = atlas_parallel_forward(
            &params.levels[0], &embedded, s, d, &cfg, None,
        );

        // Outputs must match exactly (same arithmetic, same order)
        for i in 0..(s * d) {
            assert!((y_seq[i] - y_par[i]).abs() < 1e-6,
                "y mismatch at {i}: seq={} par={}", y_seq[i], y_par[i]);
        }

        // Omega vectors must match
        assert_eq!(cache_seq.omega_vecs, cache_par.omega_vecs);
    }

    #[test]
    fn test_atlas_parallel_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let (y, _cache) = atlas_parallel_forward(
            &params.levels[0], &embedded, s, d, &cfg, None,
        );
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_atlas_parallel_backward_matches_sequential() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // Forward
        let rule = crate::atlas_omega::AtlasOmega;
        let (_, cache) = crate::delta_rule::MemoryRule::step(
            &rule, &params.levels[0], &embedded, s, d, None,
        );

        let d_y = vec![1.0f32; s * d];

        // Sequential backward
        let (grads_seq, d_emb_seq) = crate::delta_rule::MemoryRule::step_backward(
            &rule, &params.levels[0], &cache, &d_y, &embedded,
        );

        // Parallel backward
        let (grads_par, d_emb_par) = atlas_parallel_backward(
            &params.levels[0], &cache, &d_y, &embedded,
        );

        // Must match exactly (same function)
        assert_eq!(grads_seq.w_omega, grads_par.w_omega);
        assert_eq!(d_emb_seq, d_emb_par);
    }

    #[test]
    fn test_batch_compute_omega_shapes() {
        let d = 8;
        let s = 4;
        let mut rng = SimpleRng::new(42);
        let mut k_mem = vec![0.0f32; s * d];
        let mut v_mem = vec![0.0f32; s * d];
        let mut w_omega = vec![0.0f32; d * 2 * d];
        rng.fill_uniform(&mut k_mem, 0.1);
        rng.fill_uniform(&mut v_mem, 0.1);
        rng.fill_uniform(&mut w_omega, 0.1);

        let (omega_vecs, silu_kv) = batch_compute_omega(&k_mem, &v_mem, &w_omega, s, d);
        assert_eq!(omega_vecs.len(), s * d);
        assert_eq!(silu_kv.len(), s * 2 * d);

        for &v in &omega_vecs {
            assert!(v.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn test_atlas_parallel_forward_detects_nan() {
        // Verify forward propagates NaN detection
        let cfg = test_config();
        let mut params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // Inject NaN
        params.levels[0].w_omega[0] = f32::NAN;
        let (y, _) = atlas_parallel_forward(
            &params.levels[0], &embedded, s, d, &cfg, None,
        );
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] not finite: {v}");
        }
    }
}
