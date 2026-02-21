/// Pluggable retention mechanisms — MIRAS Knob #3.
///
/// Retention controls how memory decays/evolves between updates. Previously
/// fused inline into each rule's `step()` method; now extracted into standalone
/// functions so any rule can use any retention mechanism (CS-36 compliance).
///
/// Four mechanisms:
/// - L2 weight decay: multiplicative `w *= retain` (most rules)
/// - KL divergence: softmax(alpha*log(w) - theta*grad) per row (MEMORA)
/// - Elastic net: L2 decay + L1 soft thresholding (NEW)
/// - Sphere normalization: orthogonal projection + normalize (Lattice OSR)

use serde::{Serialize, Deserialize};
use crate::tensor::{log_f32, softmax_f32, vec_norm_f32};

/// Which retention mechanism to apply.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RetentionKind {
    /// Multiplicative decay: w[i] *= retain_factor.
    /// Used by Delta, Hebbian, Titans, Trellis, Moneta, YAAD.
    L2WeightDecay,
    /// KL-optimal softmax: w_new = softmax(alpha*log(w) - theta*grad) per row.
    /// Used by MEMORA. Constrains rows to probability simplex.
    KLDivergence,
    /// L2 decay + L1 soft thresholding: encourages sparsity.
    /// NEW mechanism enabled by the pluggable extraction.
    ElasticNet,
    /// Orthogonal projection onto tangent space + normalize to unit sphere.
    /// Used by Lattice OSR. Constrains slots to S^{d-1}.
    SphereNormalization,
}

/// Configuration for retention mechanisms.
///
/// Not all fields are used by all mechanisms:
/// - L2WeightDecay: uses lambda_2 (for penalty gradient, e.g. Moneta/Trellis)
/// - KLDivergence: no config needed (alpha/theta come from gates)
/// - ElasticNet: uses lambda_1 (L1 strength) and lambda_2 (L2 strength)
/// - SphereNormalization: no config needed
#[derive(Clone, Debug)]
pub struct RetentionConfig {
    /// L1 penalty strength (elastic net only).
    pub lambda_1: f32,
    /// L2/global penalty strength (Moneta, YAAD, Trellis, elastic net).
    pub lambda_2: f32,
    /// Local L2 penalty toward chunk-boundary snapshot (YAAD only).
    pub lambda_local: f32,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        RetentionConfig { lambda_1: 0.0, lambda_2: 0.0, lambda_local: 0.0 }
    }
}

/// Map each rule to its historically-used retention mechanism.
pub fn default_retention(rule: crate::model::MemoryRuleKind) -> RetentionKind {
    use crate::model::MemoryRuleKind::*;
    match rule {
        DeltaRule | TitansLMM | HebbianRule | Moneta | YAAD | Trellis | AtlasOmega => RetentionKind::L2WeightDecay,
        MEMORA => RetentionKind::KLDivergence,
        LatticeOSR => RetentionKind::SphereNormalization,
    }
}

// ── L2 Weight Decay ─────────────────────────────────────────────────

/// Apply multiplicative L2 retention in-place: `w[i] *= retain`.
///
/// `retain` is the retain factor (NOT the forget rate):
/// - Delta/Hebbian/Titans pass `1.0 - alpha` (alpha = forget gate)
/// - Moneta/YAAD pass `alpha` directly (alpha = retain gate)
///
/// This is the forward-pass decay — separate from the L2 penalty gradient
/// which is an additive term in the gradient (see `l2_retention_gradient`).
#[inline]
pub fn l2_apply_retention(w: &mut [f32], retain: f32) {
    for x in w.iter_mut() {
        *x *= retain;
    }
}

/// Compute L2 retention penalty gradient: `2 * lambda * w[i]`.
///
/// Returns a new Vec with the gradient contribution to be added to the
/// MLP gradient before the weight update. Used by Moneta and Trellis.
pub fn l2_retention_gradient(w: &[f32], lambda: f32) -> Vec<f32> {
    let c = 2.0 * lambda;
    w.iter().map(|&x| c * x).collect()
}

/// Compute decoupled L2 retention gradient: local + global.
///
/// `local = 2 * lambda_local * (w - w_boundary)`
/// `global = 2 * lambda_2 * w`
///
/// Used by YAAD. The local term penalizes drift from the chunk-boundary
/// snapshot; the global term keeps weights small overall.
pub fn l2_decoupled_gradient(
    w: &[f32],
    w_boundary: &[f32],
    lambda_local: f32,
    lambda_2: f32,
) -> Vec<f32> {
    debug_assert_eq!(w.len(), w_boundary.len());
    let ll2 = 2.0 * lambda_local;
    let lg2 = 2.0 * lambda_2;
    w.iter()
        .zip(w_boundary.iter())
        .map(|(&wi, &bi)| ll2 * (wi - bi) + lg2 * wi)
        .collect()
}

// ── KL Divergence (MEMORA) ──────────────────────────────────────────

/// Apply KL-optimal softmax retention for one matrix.
///
/// For each row r:
///   `z[c] = alpha * log(w_prev[r*cols + c]) - theta * grad[r*cols + c]`
///   `w_next[r] = softmax(z)`
///
/// Constrains each row to the probability simplex. This is the closed-form
/// solution to the KL-regularized update (MIRAS Proposition 3.1).
///
/// Returns a new Vec<f32> of length rows*cols with the updated weights.
pub fn kl_apply_retention(
    w_prev: &[f32],
    grad: &[f32],
    alpha: f32,
    theta: f32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    debug_assert_eq!(w_prev.len(), rows * cols);
    debug_assert_eq!(grad.len(), rows * cols);

    let mut log_w = vec![0.0f32; rows * cols];
    log_f32(w_prev, &mut log_w);

    let mut z_buf = vec![0.0f32; cols];
    let mut softmax_buf = vec![0.0f32; cols];
    let mut result = vec![0.0f32; rows * cols];

    for r in 0..rows {
        let row_base = r * cols;
        for c in 0..cols {
            z_buf[c] = alpha * log_w[row_base + c] - theta * grad[row_base + c];
        }
        softmax_f32(&z_buf[..cols], &mut softmax_buf[..cols], 1, cols);
        result[row_base..row_base + cols].copy_from_slice(&softmax_buf[..cols]);
    }

    result
}

/// In-place variant: writes result into `out`, uses `log_buf` and `z_buf`
/// as scratch space. Avoids per-call allocations.
///
/// `out`:     mutable slice of length >= rows*cols (receives the result)
/// `log_buf`: mutable slice of length >= rows*cols (scratch for log(w_prev))
/// `z_buf`:   mutable slice of length >= cols (scratch for per-row z/softmax)
#[inline]
pub fn kl_apply_retention_inplace(
    w_prev: &[f32],
    grad: &[f32],
    alpha: f32,
    theta: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    log_buf: &mut [f32],
    z_buf: &mut [f32],
) {
    debug_assert_eq!(w_prev.len(), rows * cols);
    debug_assert_eq!(grad.len(), rows * cols);
    debug_assert!(out.len() >= rows * cols);
    debug_assert!(log_buf.len() >= rows * cols);
    debug_assert!(z_buf.len() >= cols);

    log_f32(w_prev, &mut log_buf[..rows * cols]);

    for r in 0..rows {
        let row_base = r * cols;
        for c in 0..cols {
            z_buf[c] = alpha * log_buf[row_base + c] - theta * grad[row_base + c];
        }
        softmax_f32(&z_buf[..cols], &mut out[row_base..row_base + cols], 1, cols);
    }
}

// ── Elastic Net (NEW) ───────────────────────────────────────────────

/// Apply elastic net retention in-place:
/// 1. L2 decay: `w[i] *= retain`
/// 2. L1 soft thresholding: `w[i] = sign(w[i]) * max(0, |w[i]| - lambda_1 * theta)`
///
/// The L2 decay is applied first (same as `l2_apply_retention`), then the
/// soft threshold promotes sparsity. `theta` is the inner-loop learning rate
/// gate output — it scales the L1 threshold to match the gradient step size.
///
/// Spec: `03_elastic_net.md` — `lambda_1 * ||W||_1 + lambda_2 * ||W||_2^2`
/// with the L2 part absorbed into the multiplicative decay.
#[inline]
pub fn elastic_net_apply(w: &mut [f32], retain: f32, lambda_1: f32, theta: f32) {
    let threshold = lambda_1 * theta;
    debug_assert!(threshold >= 0.0, "elastic_net threshold must be non-negative");
    for x in w.iter_mut() {
        *x *= retain;
        let abs_x = x.abs();
        if abs_x <= threshold {
            *x = 0.0;
        } else {
            *x = x.signum() * (abs_x - threshold);
        }
    }
}

// ── FTRL Accumulator Pattern (MIRAS §3.2, Eq 23) ────────────────────

/// FTRL elastic net step: accumulate into A, then derive W via soft thresholding.
///
/// This implements the two-step FTRL update from MIRAS Eq 23:
///   Step 1: `A[i] -= eta * grad[i]`  (gradient accumulation)
///   Step 2: `W[i] = sign(A[i]) * max(0, |A[i]| - lambda)`  (proximal step)
///
/// The accumulator A is an `inner_loop_state` tensor that persists alongside
/// the memory matrix W. It tracks the cumulative gradient sum, while W is the
/// regularized solution derived from A. The L1 component drives small entries
/// in A to exactly zero in W, producing sparse memory matrices.
///
/// # Arguments
/// * `accum` — gradient accumulator A, mutated in-place (same shape as W)
/// * `w` — memory matrix W, overwritten with soft_threshold(A) (same shape as A)
/// * `grad` — current gradient (same shape as A)
/// * `eta` — inner-loop learning rate (theta gate output)
/// * `lambda` — L1 threshold: `eta / alpha` where alpha controls L1 strength
pub fn ftrl_elastic_net_step(
    accum: &mut [f32],
    w: &mut [f32],
    grad: &[f32],
    eta: f32,
    lambda: f32,
) {
    debug_assert_eq!(accum.len(), w.len());
    debug_assert_eq!(accum.len(), grad.len());
    debug_assert!(lambda >= 0.0, "FTRL lambda must be non-negative, got {lambda}");

    for i in 0..accum.len() {
        // Step 1: Accumulate
        accum[i] -= eta * grad[i];

        // Step 2: Soft threshold → W
        let abs_a = accum[i].abs();
        if abs_a <= lambda {
            w[i] = 0.0;
        } else {
            w[i] = accum[i].signum() * (abs_a - lambda);
        }
    }
}

/// Backward pass through FTRL soft thresholding (straight-through estimator).
///
/// The soft thresholding `W = sign(A) * max(0, |A| - lambda)` has gradient:
///   `dL/dA = dL/dW * indicator(|A| > lambda)`
///
/// Active entries (|A| > lambda, W ≠ 0): gradient passes through unchanged.
/// Killed entries (|A| <= lambda, W = 0): gradient is zeroed.
///
/// This is the STE approach: at the discontinuity |A| = lambda, we use 0
/// (conservative — don't revive dead entries via gradient noise).
///
/// # Arguments
/// * `d_w` — upstream gradient dL/dW
/// * `accum` — accumulator A (needed to compute the indicator)
/// * `lambda` — same threshold used in the forward pass
///
/// # Returns
/// `d_accum` — gradient dL/dA (same shape as d_w)
pub fn ftrl_soft_threshold_backward(
    d_w: &[f32],
    accum: &[f32],
    lambda: f32,
) -> Vec<f32> {
    debug_assert_eq!(d_w.len(), accum.len());
    debug_assert!(lambda >= 0.0, "FTRL lambda must be non-negative, got {lambda}");

    d_w.iter()
        .zip(accum.iter())
        .map(|(&dw, &a)| {
            if a.abs() > lambda { dw } else { 0.0 }
        })
        .collect()
}

/// In-place variant of `ftrl_soft_threshold_backward`.
///
/// Writes `d_accum[i] = d_w[i] * indicator(|accum[i]| > lambda)` into `d_accum`.
#[inline]
pub fn ftrl_soft_threshold_backward_inplace(
    d_w: &[f32],
    accum: &[f32],
    lambda: f32,
    d_accum: &mut [f32],
) {
    debug_assert_eq!(d_w.len(), accum.len());
    debug_assert_eq!(d_w.len(), d_accum.len());
    debug_assert!(lambda >= 0.0, "FTRL lambda must be non-negative, got {lambda}");

    for i in 0..d_w.len() {
        d_accum[i] = if accum[i].abs() > lambda { d_w[i] } else { 0.0 };
    }
}

// ── Sphere Normalization (Lattice OSR) ──────────────────────────────

/// Apply orthogonal projection + normalize for one slot.
///
/// Given slot `s` (unit vector on S^{d-1}) and update `delta_s`:
/// 1. Project delta_s onto tangent space: `ortho = delta_s - dot(s, delta_s) * s`
/// 2. Step: `s_unnorm = s + ortho`
/// 3. Normalize: `s_new = s_unnorm / ||s_unnorm||`
///
/// Returns (s_new, unnorm_norm) where unnorm_norm is ||s_unnorm|| (needed for backward).
/// If ||s_unnorm|| < 1e-8, returns the original slot unchanged.
pub fn sphere_project_and_normalize(
    slot: &[f32],
    delta_s: &[f32],
    d: usize,
) -> (Vec<f32>, f32) {
    let mut s_new = vec![0.0f32; d];
    let norm = sphere_project_and_normalize_inplace(slot, delta_s, d, &mut s_new);
    (s_new, norm)
}

/// In-place variant: writes result into `s_new_out`, returns unnorm_norm.
///
/// Avoids per-call Vec allocation. Callers allocate `s_new_out` once (len >= d)
/// and reuse across iterations.
#[inline]
pub fn sphere_project_and_normalize_inplace(
    slot: &[f32],
    delta_s: &[f32],
    d: usize,
    s_new_out: &mut [f32],
) -> f32 {
    debug_assert_eq!(slot.len(), d);
    debug_assert_eq!(delta_s.len(), d);
    debug_assert!(s_new_out.len() >= d);

    // Compute parallel component: p = dot(s, delta_s)
    let mut p = 0.0f32;
    for j in 0..d {
        p += slot[j] * delta_s[j];
    }

    // s_unnorm = s + (delta_s - p * s)
    for j in 0..d {
        s_new_out[j] = slot[j] + delta_s[j] - p * slot[j];
    }

    let norm = vec_norm_f32(&s_new_out[..d]);
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for j in 0..d {
            s_new_out[j] *= inv;
        }
    } else {
        s_new_out[..d].copy_from_slice(slot);
    }

    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── L2 apply retention ──────────────────────────────────────────

    #[test]
    fn test_l2_apply_retention_basic() {
        let mut w = vec![1.0, 2.0, 3.0, 4.0];
        l2_apply_retention(&mut w, 0.9);
        assert!((w[0] - 0.9).abs() < 1e-6);
        assert!((w[1] - 1.8).abs() < 1e-6);
        assert!((w[2] - 2.7).abs() < 1e-6);
        assert!((w[3] - 3.6).abs() < 1e-6);
    }

    #[test]
    fn test_l2_apply_retention_zero() {
        let mut w = vec![1.0, 2.0, 3.0];
        l2_apply_retention(&mut w, 0.0);
        assert!(w.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_apply_retention_one() {
        let original = vec![1.0, -2.0, 3.0];
        let mut w = original.clone();
        l2_apply_retention(&mut w, 1.0);
        assert_eq!(w, original);
    }

    #[test]
    fn test_l2_apply_retention_empty() {
        let mut w: Vec<f32> = vec![];
        l2_apply_retention(&mut w, 0.5);
        assert!(w.is_empty());
    }

    // ── L2 retention gradient ───────────────────────────────────────

    #[test]
    fn test_l2_retention_gradient_basic() {
        let w = vec![1.0, -2.0, 0.5];
        let g = l2_retention_gradient(&w, 0.01);
        assert!((g[0] - 0.02).abs() < 1e-7);
        assert!((g[1] - (-0.04)).abs() < 1e-7);
        assert!((g[2] - 0.01).abs() < 1e-7);
    }

    #[test]
    fn test_l2_retention_gradient_zero_lambda() {
        let w = vec![1.0, 2.0, 3.0];
        let g = l2_retention_gradient(&w, 0.0);
        assert!(g.iter().all(|&x| x == 0.0));
    }

    // ── L2 decoupled gradient ───────────────────────────────────────

    #[test]
    fn test_l2_decoupled_gradient_basic() {
        let w = vec![1.0, 2.0];
        let wb = vec![0.5, 1.5];
        let g = l2_decoupled_gradient(&w, &wb, 0.01, 0.01);
        // local: 2*0.01*(1.0-0.5) = 0.01, global: 2*0.01*1.0 = 0.02 → 0.03
        assert!((g[0] - 0.03).abs() < 1e-7);
        // local: 2*0.01*(2.0-1.5) = 0.01, global: 2*0.01*2.0 = 0.04 → 0.05
        assert!((g[1] - 0.05).abs() < 1e-7);
    }

    #[test]
    fn test_l2_decoupled_gradient_zero_local() {
        let w = vec![1.0, 2.0];
        let wb = vec![0.5, 1.5];
        let g = l2_decoupled_gradient(&w, &wb, 0.0, 0.01);
        // Only global: 2*0.01*w
        assert!((g[0] - 0.02).abs() < 1e-7);
        assert!((g[1] - 0.04).abs() < 1e-7);
    }

    // ── KL apply retention ──────────────────────────────────────────

    #[test]
    fn test_kl_apply_retention_rows_sum_to_one() {
        // Start from uniform distribution
        let w = vec![0.25f32; 8]; // 2 rows x 4 cols
        let grad = vec![0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.1, -0.1];
        let result = kl_apply_retention(&w, &grad, 0.95, 0.01, 2, 4);
        // Each row should sum to ~1.0
        let row0_sum: f32 = result[0..4].iter().sum();
        let row1_sum: f32 = result[4..8].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5, "row0 sum = {row0_sum}");
        assert!((row1_sum - 1.0).abs() < 1e-5, "row1 sum = {row1_sum}");
    }

    #[test]
    fn test_kl_apply_retention_all_positive() {
        let w = vec![0.25f32; 4]; // 1 row x 4 cols
        let grad = vec![1.0, -1.0, 0.5, -0.5];
        let result = kl_apply_retention(&w, &grad, 0.9, 0.1, 1, 4);
        for &v in &result {
            assert!(v > 0.0, "KL retention should produce positive values");
        }
    }

    // ── KL apply retention inplace ─────────────────────────────────

    #[test]
    fn test_kl_apply_retention_inplace_matches_allocating() {
        let w = vec![0.25f32; 12]; // 3 rows x 4 cols
        let grad = vec![0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.1, -0.1, -0.3, 0.3, 0.0, 0.0];
        let (rows, cols) = (3, 4);

        // Allocating version
        let expected = kl_apply_retention(&w, &grad, 0.9, 0.05, rows, cols);

        // Inplace version
        let mut out = vec![0.0f32; rows * cols];
        let mut log_buf = vec![0.0f32; rows * cols];
        let mut z_buf = vec![0.0f32; cols];
        kl_apply_retention_inplace(&w, &grad, 0.9, 0.05, rows, cols, &mut out, &mut log_buf, &mut z_buf);

        for i in 0..rows * cols {
            assert!(
                (out[i] - expected[i]).abs() < 1e-7,
                "mismatch at [{i}]: inplace={} expected={}", out[i], expected[i]
            );
        }
    }

    // ── Elastic net ─────────────────────────────────────────────────

    #[test]
    fn test_elastic_net_apply_basic() {
        let mut w = vec![1.0, -0.5, 0.01, -0.01, 2.0];
        elastic_net_apply(&mut w, 0.9, 0.1, 1.0);
        // After L2: [0.9, -0.45, 0.009, -0.009, 1.8]
        // threshold = 0.1*1.0 = 0.1
        // 0.9 > 0.1: 0.9 - 0.1 = 0.8
        // 0.45 > 0.1: -(0.45 - 0.1) = -0.35
        // 0.009 <= 0.1: 0.0
        // 0.009 <= 0.1: 0.0
        // 1.8 > 0.1: 1.8 - 0.1 = 1.7
        assert!((w[0] - 0.8).abs() < 1e-6);
        assert!((w[1] - (-0.35)).abs() < 1e-6);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);
        assert!((w[4] - 1.7).abs() < 1e-6);
    }

    #[test]
    fn test_elastic_net_no_l1() {
        let original = vec![1.0, -2.0, 3.0];
        let mut w = original.clone();
        elastic_net_apply(&mut w, 0.9, 0.0, 1.0);
        // lambda_1=0 → no soft threshold, just L2 decay
        assert!((w[0] - 0.9).abs() < 1e-6);
        assert!((w[1] - (-1.8)).abs() < 1e-6);
        assert!((w[2] - 2.7).abs() < 1e-6);
    }

    // ── Sphere project and normalize ────────────────────────────────

    #[test]
    fn test_sphere_project_unit_norm() {
        // Start from unit vector [1, 0, 0, 0]
        let slot = vec![1.0, 0.0, 0.0, 0.0];
        let delta = vec![0.0, 0.1, 0.0, 0.0]; // purely orthogonal
        let (s_new, _norm) = sphere_project_and_normalize(&slot, &delta, 4);
        let norm: f32 = s_new.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "result should be unit norm, got {norm}");
    }

    #[test]
    fn test_sphere_project_parallel_ignored() {
        // Delta purely parallel to slot should not change direction
        let slot = vec![1.0, 0.0, 0.0, 0.0];
        let delta = vec![0.5, 0.0, 0.0, 0.0]; // parallel to slot
        let (s_new, _norm) = sphere_project_and_normalize(&slot, &delta, 4);
        // After projection, orthogonal component is zero → s_unnorm = slot + 0 = slot
        assert!((s_new[0] - 1.0).abs() < 1e-6);
        assert!(s_new[1].abs() < 1e-6);
    }

    #[test]
    fn test_sphere_project_orthogonal_moves() {
        let slot = vec![1.0, 0.0, 0.0, 0.0];
        let delta = vec![0.0, 0.5, 0.0, 0.0]; // perpendicular
        let (s_new, _) = sphere_project_and_normalize(&slot, &delta, 4);
        // s_unnorm = [1.0, 0.5, 0, 0], normalized
        let expected_norm = (1.0f32 + 0.25).sqrt();
        assert!((s_new[0] - 1.0 / expected_norm).abs() < 1e-6);
        assert!((s_new[1] - 0.5 / expected_norm).abs() < 1e-6);
    }

    // ── FTRL accumulator pattern ────────────────────────────────────

    #[test]
    fn test_ftrl_elastic_net_step_basic() {
        let mut accum = vec![0.0f32; 4];
        let mut w = vec![0.0f32; 4];
        let grad = vec![1.0, -2.0, 0.5, -0.1];

        // eta=0.1, lambda=0.05
        ftrl_elastic_net_step(&mut accum, &mut w, &grad, 0.1, 0.05);

        // accum: [0 - 0.1*1.0, 0 - 0.1*(-2.0), 0 - 0.1*0.5, 0 - 0.1*(-0.1)]
        //      = [-0.1, 0.2, -0.05, 0.01]
        assert!((accum[0] - (-0.1)).abs() < 1e-7);
        assert!((accum[1] - 0.2).abs() < 1e-7);
        assert!((accum[2] - (-0.05)).abs() < 1e-7);
        assert!((accum[3] - 0.01).abs() < 1e-7);

        // W = soft_threshold(accum, 0.05):
        // |−0.1| > 0.05 → −(0.1−0.05) = −0.05
        // |0.2| > 0.05 → +(0.2−0.05) = 0.15
        // |−0.05| <= 0.05 → 0.0  (exactly at threshold → killed)
        // |0.01| <= 0.05 → 0.0
        assert!((w[0] - (-0.05)).abs() < 1e-7);
        assert!((w[1] - 0.15).abs() < 1e-7);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);
    }

    #[test]
    fn test_ftrl_accumulates_across_steps() {
        let mut accum = vec![0.0f32; 2];
        let mut w = vec![0.0f32; 2];

        // Step 1: small gradient
        ftrl_elastic_net_step(&mut accum, &mut w, &[0.5, -0.3], 0.1, 0.02);
        assert!((accum[0] - (-0.05)).abs() < 1e-7);
        assert!((accum[1] - 0.03).abs() < 1e-7);

        // Step 2: accumulates on top
        ftrl_elastic_net_step(&mut accum, &mut w, &[0.5, -0.3], 0.1, 0.02);
        assert!((accum[0] - (-0.10)).abs() < 1e-7);
        assert!((accum[1] - 0.06).abs() < 1e-7);

        // W derived from final A
        // |−0.10| > 0.02 → −(0.10−0.02) = −0.08
        // |0.06| > 0.02 → +(0.06−0.02) = 0.04
        assert!((w[0] - (-0.08)).abs() < 1e-7);
        assert!((w[1] - 0.04).abs() < 1e-7);
    }

    #[test]
    fn test_ftrl_zero_lambda_no_thresholding() {
        let mut accum = vec![0.0f32; 3];
        let mut w = vec![0.0f32; 3];

        ftrl_elastic_net_step(&mut accum, &mut w, &[1.0, -1.0, 0.001], 0.1, 0.0);
        // lambda=0 → W = A (no thresholding, recovers pure L2)
        for i in 0..3 {
            assert_eq!(w[i], accum[i], "zero lambda should make W=A");
        }
    }

    #[test]
    fn test_ftrl_backward_ste_basic() {
        let d_w = vec![1.0, 2.0, 3.0, 4.0];
        let accum = vec![0.5, 0.01, -0.3, -0.001];
        let lambda = 0.05;

        let d_a = ftrl_soft_threshold_backward(&d_w, &accum, lambda);

        // |0.5| > 0.05 → pass through: 1.0
        // |0.01| <= 0.05 → killed: 0.0
        // |−0.3| > 0.05 → pass through: 3.0
        // |−0.001| <= 0.05 → killed: 0.0
        assert_eq!(d_a[0], 1.0);
        assert_eq!(d_a[1], 0.0);
        assert_eq!(d_a[2], 3.0);
        assert_eq!(d_a[3], 0.0);
    }

    #[test]
    fn test_ftrl_backward_inplace_matches() {
        let d_w = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let accum = vec![0.5, 0.01, -0.3, -0.001, 1.0];
        let lambda = 0.05;

        let expected = ftrl_soft_threshold_backward(&d_w, &accum, lambda);

        let mut d_a = vec![0.0f32; 5];
        ftrl_soft_threshold_backward_inplace(&d_w, &accum, lambda, &mut d_a);

        assert_eq!(d_a, expected);
    }

    #[test]
    fn test_ftrl_backward_zero_lambda_all_pass() {
        let d_w = vec![1.0, 2.0, 3.0];
        let accum = vec![0.001, -0.001, 0.0]; // tiny but nonzero
        let lambda = 0.0;

        let d_a = ftrl_soft_threshold_backward(&d_w, &accum, lambda);
        // lambda=0: |A| > 0 for nonzero entries → pass through
        // |0.0| is NOT > 0 → killed
        assert_eq!(d_a[0], 1.0);
        assert_eq!(d_a[1], 2.0);
        assert_eq!(d_a[2], 0.0); // exactly zero accumulator → killed
    }

    // ── default_retention mapping ───────────────────────────────────

    #[test]
    fn test_default_retention_mapping() {
        use crate::model::MemoryRuleKind;
        assert_eq!(default_retention(MemoryRuleKind::DeltaRule), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::TitansLMM), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::HebbianRule), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::Moneta), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::YAAD), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::Trellis), RetentionKind::L2WeightDecay);
        assert_eq!(default_retention(MemoryRuleKind::MEMORA), RetentionKind::KLDivergence);
        assert_eq!(default_retention(MemoryRuleKind::LatticeOSR), RetentionKind::SphereNormalization);
    }
}
