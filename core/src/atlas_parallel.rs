/// Atlas Parallel — placeholder for Atlas Omega parallelization strategy.
///
/// Atlas Omega is a planned MIRAS variant that uses a state-independent omega
/// function: omega = W_omega @ silu(concat(k, v)). Because omega is independent
/// of the memory state, all tokens can be computed in parallel.
///
/// This module defines the types and omega function but panics on forward/backward
/// since the Atlas Omega rule is not yet implemented.

use crate::tensor::silu_f32;

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

/// Atlas parallel forward — NOT YET IMPLEMENTED.
///
/// Panics with a clear message. Atlas Omega rule must first be implemented
/// as a MIRAS variant before this parallelization strategy can be used.
pub fn atlas_parallel_forward(
    _level_params: &crate::model::MemoryLevelParams,
    _embedded: &[f32],
    _seq_len: usize,
    _d: usize,
    _cfg: &crate::model::MAGConfig,
    _initial_m: Option<Vec<f32>>,
) -> ! {
    unimplemented!(
        "Atlas parallel forward requires the Atlas Omega MIRAS variant, \
         which is not yet implemented. Add AtlasOmega to the MemoryRuleKind enum \
         and implement the MemoryRule trait first."
    )
}

/// Atlas parallel backward — NOT YET IMPLEMENTED.
pub fn atlas_parallel_backward(
    _level_params: &crate::model::MemoryLevelParams,
    _d_y: &[f32],
    _embedded: &[f32],
    _cfg: &crate::model::MAGConfig,
) -> ! {
    unimplemented!(
        "Atlas parallel backward requires the Atlas Omega MIRAS variant, \
         which is not yet implemented."
    )
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
    #[should_panic(expected = "Atlas Omega MIRAS variant")]
    fn test_atlas_forward_panics() {
        let cfg = crate::model::MAGConfig::test_config();
        let params = crate::model::MAGParams::init(&cfg, 42);
        let embedded = vec![0.0f32; cfg.swa.seq_len * cfg.swa.d_model];
        atlas_parallel_forward(
            &params.levels[0], &embedded,
            cfg.swa.seq_len, cfg.swa.d_model, &cfg, None,
        );
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
