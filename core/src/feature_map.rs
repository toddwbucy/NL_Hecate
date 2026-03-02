/// Feature map phi(k) — optional key preprocessing before memory operations.
///
/// Spec: specs/algorithms/self_referential/02_feature_maps.md
/// Source: HOPE (2512.24695) §2 Eq 5 (FWP with phi), §4.4 Eq 51 (higher-order momentum).
///
/// The feature map slots between key projection and memory operations:
///   k_t (from W_K) → phi(k_t) → enters error/outer-product/read
///
/// Gates (alpha/theta/eta) continue using raw k_t — gates control update
/// magnitude, not key content.
///
/// Key invariant: both k and q use the same phi instance. Write and read
/// must share feature space or associative retrieval breaks.
///
/// RandomFourier stability:
///   ||phi(k)|| <= sqrt(2) for ALL k (bounded by cos^2 <= 1).
///   Since ||phi(k)||^2 <= 2, we have 1/||phi(k)||^2 >= 1/2.
///   A conservative global bound is therefore: theta < 1/2.
///   This makes carry-forward safe for slow CMS levels (L2/L3).

use serde::{Serialize, Deserialize};
use crate::tensor::SimpleRng;

// ── Feature map kind ─────────────────────────────────────────────────

/// Which feature map phi to apply to keys and queries before memory operations.
///
/// Identity is the serde/backward-compat default — all existing checkpoints
/// and configs continue to work unchanged. RandomFourier is the recommended
/// default for new HOPE configs (see spec stability argument).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FeatureMapKind {
    /// phi(k) = k. No-op — bitwise identical to pre-feature-map behavior.
    Identity,
    /// phi(k) = sqrt(2/d) * cos(W_rand @ k + b_rand).
    /// W_rand ~ N(0, sigma^2), b_rand ~ U[0, 2*pi]. Both frozen after init.
    /// Guarantees ||phi(k)|| <= sqrt(2) for all k.
    RandomFourier {
        /// Bandwidth parameter. Default 1.0 approximates N(0,1) Gaussian kernel.
        sigma: f32,
    },
    /// phi(k) = elu(k) + 1. Element-wise, ensures phi(k) > 0.
    /// From linear attention literature (Katharopoulos et al. 2020).
    ELU,
}

impl Default for FeatureMapKind {
    fn default() -> Self {
        FeatureMapKind::Identity
    }
}

// ── Init ─────────────────────────────────────────────────────────────

/// Initialize frozen RandomFourier weights.
///
/// w_rand: [d, d] drawn from N(0, sigma^2). Implemented via Box-Muller
/// transform from our uniform SimpleRng (avoids adding a Gaussian RNG dep).
/// b_rand: [d] drawn from U[0, 2*pi].
///
/// Both are frozen — they are stored in MemoryLevelParams but not updated
/// by the outer-loop optimizer. They don't need gradients.
pub fn init_random_fourier(d: usize, sigma: f32, rng: &mut SimpleRng) -> (Vec<f32>, Vec<f32>) {
    // w_rand: [d * d] — Box-Muller transform for N(0, sigma^2)
    // Pairs: (u1, u2) -> (sqrt(-2*ln(u1))*cos(2*pi*u2), sqrt(-2*ln(u1))*sin(2*pi*u2))
    let mut w_rand = vec![0.0f32; d * d];
    let two_pi = 2.0 * std::f32::consts::PI;
    let n = d * d;
    let pairs = (n + 1) / 2;
    for i in 0..pairs {
        // Draw two uniform samples in (0, 1]. Use scale=0.5 → rng gives [-0.5, 0.5],
        // then shift to (0, 1]. Clamp away from 0 to prevent ln(0).
        let u1 = (rng.uniform(0.5) + 0.5).max(1e-7_f32);
        let u2 = (rng.uniform(0.5) + 0.5).max(1e-7_f32);
        let r = (-2.0 * u1.ln()).sqrt() * sigma;
        let g1 = r * (two_pi * u2).cos();
        let g2 = r * (two_pi * u2).sin();
        w_rand[2 * i] = g1;
        if 2 * i + 1 < n {
            w_rand[2 * i + 1] = g2;
        }
    }

    // b_rand: [d] — U[0, 2*pi]
    let mut b_rand = vec![0.0f32; d];
    for v in b_rand.iter_mut() {
        *v = (rng.uniform(0.5) + 0.5) * two_pi;
    }

    (w_rand, b_rand)
}

// ── Forward ──────────────────────────────────────────────────────────

/// Apply feature map to a single token key/query vector x: [d].
///
/// Returns (phi_x, z) where:
/// - phi_x: [d] the mapped vector (same dim; d_phi = d for all current maps)
/// - z: [d] the pre-activation (needed for backward). Empty Vec for Identity only.
///
/// For Identity: returns (x.to_vec(), vec![]) — zero allocation hot path.
/// For RandomFourier: z = W_rand @ x + b_rand, phi_x = sqrt(2/d) * cos(z).
/// For ELU: phi_x = elu(x) + 1 = max(x, 0) + exp(min(x, 0)), z = x (the input,
///          cached as the pre-activation for VJP: d(phi)/dx = 1 if x>0 else exp(x)).
pub fn apply(x: &[f32], kind: &FeatureMapKind, w_rand: &[f32], b_rand: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    match kind {
        FeatureMapKind::Identity => {
            (x.to_vec(), vec![])
        }
        FeatureMapKind::RandomFourier { .. } => {
            debug_assert_eq!(w_rand.len(), d * d, "w_rand shape mismatch");
            debug_assert_eq!(b_rand.len(), d, "b_rand shape mismatch");
            // z = W_rand @ x + b_rand  (W_rand is [d, d], x is [d])
            let mut z = b_rand.to_vec();
            for i in 0..d {
                for j in 0..d {
                    z[i] += w_rand[i * d + j] * x[j];
                }
            }
            // phi_x = sqrt(2/d) * cos(z)
            let scale = (2.0 / d as f32).sqrt();
            let phi_x: Vec<f32> = z.iter().map(|&zi| scale * zi.cos()).collect();
            (phi_x, z)
        }
        FeatureMapKind::ELU => {
            // phi(k) = elu(k) + 1 = k+1 if k>0, else exp(k)
            let phi_x: Vec<f32> = x.iter().map(|&xi| {
                if xi > 0.0 { xi + 1.0 } else { xi.exp() }
            }).collect();
            // z = x (needed for backward: d(phi)/dx = 1 if x>0, else exp(x))
            (phi_x, x.to_vec())
        }
    }
}

/// Apply feature map in-place into pre-allocated output buffers.
///
/// Eliminates per-token Vec allocations in hot loops. Callers preallocate
/// `phi_out: [d]` and `z_out: [d]` once before the token loop and reuse them.
///
/// - Identity: copies x into phi_out; z_out is left unchanged (unused).
/// - RandomFourier: computes z = W_rand @ x + b_rand into z_out, phi = sqrt(2/d)*cos(z) into phi_out.
/// - ELU: computes phi = elu(x)+1 into phi_out, z = x into z_out.
pub fn apply_into(
    x: &[f32],
    kind: &FeatureMapKind,
    w_rand: &[f32],
    b_rand: &[f32],
    phi_out: &mut [f32],
    z_out: &mut [f32],
    d: usize,
) {
    match kind {
        FeatureMapKind::Identity => {
            phi_out.copy_from_slice(x);
        }
        FeatureMapKind::RandomFourier { .. } => {
            debug_assert_eq!(w_rand.len(), d * d, "w_rand shape mismatch");
            debug_assert_eq!(b_rand.len(), d, "b_rand shape mismatch");
            z_out.copy_from_slice(b_rand);
            for i in 0..d {
                for j in 0..d {
                    z_out[i] += w_rand[i * d + j] * x[j];
                }
            }
            let scale = (2.0 / d as f32).sqrt();
            for i in 0..d {
                phi_out[i] = scale * z_out[i].cos();
            }
        }
        FeatureMapKind::ELU => {
            for i in 0..d {
                let xi = x[i];
                phi_out[i] = if xi > 0.0 { xi + 1.0 } else { xi.exp() };
                z_out[i] = xi;
            }
        }
    }
}

/// Apply feature map to a batch of seq_len tokens: x_mem is [seq_len * d].
///
/// Returns (phi_mem, z_mem) where each is [seq_len * d].
/// For Identity: phi_mem = x_mem.to_vec(), z_mem = vec![].
pub fn apply_batch(
    x_mem: &[f32],
    kind: &FeatureMapKind,
    w_rand: &[f32],
    b_rand: &[f32],
    seq_len: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    match kind {
        FeatureMapKind::Identity => {
            (x_mem.to_vec(), vec![])
        }
        _ => {
            let mut phi_mem = vec![0.0f32; seq_len * d];
            let mut z_mem = vec![0.0f32; seq_len * d];
            // Preallocate per-token scratch to avoid per-iteration allocation.
            let mut phi_buf = vec![0.0f32; d];
            let mut z_buf = vec![0.0f32; d];
            for t in 0..seq_len {
                let x_t = &x_mem[t * d..(t + 1) * d];
                apply_into(x_t, kind, w_rand, b_rand, &mut phi_buf, &mut z_buf, d);
                phi_mem[t * d..(t + 1) * d].copy_from_slice(&phi_buf);
                z_mem[t * d..(t + 1) * d].copy_from_slice(&z_buf);
            }
            (phi_mem, z_mem)
        }
    }
}

// ── Backward (VJP) ───────────────────────────────────────────────────

/// VJP through phi: compute d_x given d_phi_x and cached z.
///
/// Identity: d_x = d_phi_x (pass-through, no-op copy).
/// RandomFourier: d_x = W_rand^T @ (-sqrt(2/d) * sin(z) ⊙ d_phi_x).
///   w_rand is frozen — no d_w_rand accumulated.
/// ELU: d_x[i] = d_phi_x[i] * (1 if x[i]>0 else exp(x[i])).
///   z = x (as stored in the ELU forward pass).
///
/// Note: for Identity, z should be empty. For RFF/ELU, z has length d.
pub fn vjp(d_phi_x: &[f32], z: &[f32], kind: &FeatureMapKind, w_rand: &[f32], d: usize) -> Vec<f32> {
    match kind {
        FeatureMapKind::Identity => {
            d_phi_x.to_vec()
        }
        FeatureMapKind::RandomFourier { .. } => {
            debug_assert_eq!(z.len(), d, "RFF vjp: z must have length d");
            debug_assert_eq!(w_rand.len(), d * d, "RFF vjp: w_rand shape mismatch");
            // d_z = -sqrt(2/d) * sin(z) ⊙ d_phi_x  (chain through cos)
            let scale = (2.0 / d as f32).sqrt();
            let mut d_z = vec![0.0f32; d];
            for i in 0..d {
                d_z[i] = -scale * z[i].sin() * d_phi_x[i];
            }
            // d_x = W_rand^T @ d_z  (W_rand is [d, d])
            let mut d_x = vec![0.0f32; d];
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += w_rand[i * d + j] * d_z[i];
                }
                d_x[j] = sum;
            }
            d_x
        }
        FeatureMapKind::ELU => {
            // z = x (stored original x).
            // d(phi)/dx = 1 if x>0, else exp(x)
            debug_assert_eq!(z.len(), d, "ELU vjp: z must have length d");
            let mut d_x = vec![0.0f32; d];
            for i in 0..d {
                let deriv = if z[i] > 0.0 { 1.0 } else { z[i].exp() };
                d_x[i] = d_phi_x[i] * deriv;
            }
            d_x
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::SimpleRng;

    #[test]
    fn test_identity_is_noop() {
        let d = 8;
        let x: Vec<f32> = (0..d).map(|i| i as f32 * 0.1).collect();
        let (phi_x, z) = apply(&x, &FeatureMapKind::Identity, &[], &[], d);
        assert_eq!(phi_x, x, "Identity must return x unchanged");
        assert!(z.is_empty(), "Identity must return empty z");
    }

    #[test]
    fn test_identity_vjp_passthrough() {
        let d = 8;
        let d_phi: Vec<f32> = (0..d).map(|i| i as f32 * 0.3).collect();
        let d_x = vjp(&d_phi, &[], &FeatureMapKind::Identity, &[], d);
        assert_eq!(d_x, d_phi, "Identity VJP must pass gradient through unchanged");
    }

    #[test]
    fn test_rff_norm_bound() {
        let d = 16;
        let sigma = 1.0_f32;
        let mut rng = SimpleRng::new(42);
        let (w_rand, b_rand) = init_random_fourier(d, sigma, &mut rng);
        let kind = FeatureMapKind::RandomFourier { sigma };

        let mut rng2 = SimpleRng::new(99);
        for _ in 0..1000 {
            // Random k with varying norms
            let scale = rng2.uniform(5.0);
            let k: Vec<f32> = (0..d).map(|_| rng2.uniform(scale)).collect();
            let (phi_k, _) = apply(&k, &kind, &w_rand, &b_rand, d);
            let norm_sq: f32 = phi_k.iter().map(|&x| x * x).sum();
            assert!(
                norm_sq <= 2.0 + 1e-5,
                "RFF norm^2 = {} > 2.0 for d={} — bound violated",
                norm_sq, d
            );
        }
    }

    #[test]
    fn test_rff_backward_fd() {
        let d = 8;
        let sigma = 1.0_f32;
        let mut rng = SimpleRng::new(1234);
        let (w_rand, b_rand) = init_random_fourier(d, sigma, &mut rng);
        let kind = FeatureMapKind::RandomFourier { sigma };

        // Pick a random x and upstream gradient
        let mut rng2 = SimpleRng::new(77);
        let x: Vec<f32> = (0..d).map(|_| rng2.uniform(1.0)).collect();
        let d_phi: Vec<f32> = (0..d).map(|_| rng2.uniform(1.0)).collect();

        // Analytic VJP
        let (_, z) = apply(&x, &kind, &w_rand, &b_rand, d);
        let d_x_analytic = vjp(&d_phi, &z, &kind, &w_rand, d);

        // Finite difference
        let eps = 1e-2_f32;
        let mut d_x_fd = vec![0.0f32; d];
        for j in 0..d {
            let mut xp = x.clone();
            xp[j] += eps;
            let mut xm = x.clone();
            xm[j] -= eps;
            let (phi_p, _) = apply(&xp, &kind, &w_rand, &b_rand, d);
            let (phi_m, _) = apply(&xm, &kind, &w_rand, &b_rand, d);
            // Directional derivative: d_phi^T @ (phi(x+eps*e_j) - phi(x-eps*e_j)) / (2*eps)
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += d_phi[i] * (phi_p[i] - phi_m[i]);
            }
            d_x_fd[j] = dot / (2.0 * eps);
        }

        // Check: each component within 10% relative or 5e-4 absolute
        for j in 0..d {
            let a = d_x_analytic[j];
            let fd = d_x_fd[j];
            let abs_err = (a - fd).abs();
            let rel_err = abs_err / (fd.abs().max(1e-6));
            assert!(
                abs_err < 5e-4 || rel_err < 0.10,
                "RFF backward FD check failed at j={}: analytic={:.6} fd={:.6} abs_err={:.2e} rel_err={:.2e}",
                j, a, fd, abs_err, rel_err
            );
        }
    }

    #[test]
    fn test_elu_forward_positive() {
        let d = 4;
        let x = vec![1.0f32, 2.0, 0.5, 3.0];
        let (phi_x, z) = apply(&x, &FeatureMapKind::ELU, &[], &[], d);
        // phi(k) = k + 1 for k > 0
        for i in 0..d {
            assert!((phi_x[i] - (x[i] + 1.0)).abs() < 1e-6, "ELU positive case failed");
        }
        assert_eq!(z, x, "ELU must cache z=x for backward");
    }

    #[test]
    fn test_elu_forward_negative() {
        let d = 4;
        let x = vec![-1.0f32, -0.5, -2.0, -0.1];
        let (phi_x, _) = apply(&x, &FeatureMapKind::ELU, &[], &[], d);
        for i in 0..d {
            let expected = x[i].exp();
            assert!((phi_x[i] - expected).abs() < 1e-6, "ELU negative case: got {} expected {}", phi_x[i], expected);
        }
    }

    #[test]
    fn test_elu_backward_fd() {
        let d = 8;
        let mut rng = SimpleRng::new(555);
        let x: Vec<f32> = (0..d).map(|_| rng.uniform(2.0)).collect();
        let d_phi: Vec<f32> = (0..d).map(|_| rng.uniform(1.0)).collect();

        let (_, z) = apply(&x, &FeatureMapKind::ELU, &[], &[], d);
        let d_x_analytic = vjp(&d_phi, &z, &FeatureMapKind::ELU, &[], d);

        let eps = 1e-3_f32;
        for j in 0..d {
            let mut xp = x.clone();
            xp[j] += eps;
            let mut xm = x.clone();
            xm[j] -= eps;
            let (phi_p, _) = apply(&xp, &FeatureMapKind::ELU, &[], &[], d);
            let (phi_m, _) = apply(&xm, &FeatureMapKind::ELU, &[], &[], d);
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += d_phi[i] * (phi_p[i] - phi_m[i]);
            }
            let fd = dot / (2.0 * eps);
            let abs_err = (d_x_analytic[j] - fd).abs();
            assert!(
                abs_err < 1e-3,
                "ELU backward j={}: analytic={:.6} fd={:.6} err={:.2e}",
                j, d_x_analytic[j], fd, abs_err
            );
        }
    }

    #[test]
    fn test_batch_matches_single() {
        let d = 8;
        let seq_len = 4;
        let sigma = 1.0f32;
        let mut rng = SimpleRng::new(42);
        let (w_rand, b_rand) = init_random_fourier(d, sigma, &mut rng);
        let kind = FeatureMapKind::RandomFourier { sigma };

        let x_mem: Vec<f32> = (0..seq_len * d).map(|i| i as f32 * 0.05).collect();
        let (phi_batch, z_batch) = apply_batch(&x_mem, &kind, &w_rand, &b_rand, seq_len, d);

        for t in 0..seq_len {
            let x_t = &x_mem[t * d..(t + 1) * d];
            let (phi_t, z_t) = apply(x_t, &kind, &w_rand, &b_rand, d);
            let batch_phi_t = &phi_batch[t * d..(t + 1) * d];
            let batch_z_t = &z_batch[t * d..(t + 1) * d];
            for i in 0..d {
                assert!((batch_phi_t[i] - phi_t[i]).abs() < 1e-6, "batch/single phi mismatch t={} i={}", t, i);
                assert!((batch_z_t[i] - z_t[i]).abs() < 1e-6, "batch/single z mismatch t={} i={}", t, i);
            }
        }
    }

    #[test]
    fn test_init_random_fourier_shapes() {
        let d = 16;
        let mut rng = SimpleRng::new(42);
        let (w, b) = init_random_fourier(d, 1.0, &mut rng);
        assert_eq!(w.len(), d * d, "w_rand must be [d*d]");
        assert_eq!(b.len(), d, "b_rand must be [d]");
        // b_rand in [0, 2*pi]
        let two_pi = 2.0 * std::f32::consts::PI;
        for &bi in &b {
            assert!(bi >= 0.0 && bi <= two_pi, "b_rand[i]={} out of [0, 2pi]", bi);
        }
    }
}
