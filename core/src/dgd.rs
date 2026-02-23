/// Delta Gradient Descent (DGD) — standalone composable primitive.
///
/// DGD upgrades the inner-loop optimizer from state-independent (dot-product)
/// to state-dependent (L2 regression). The update depends on what M has already
/// learned: error = M@k - v, grad = outer(error, k).
///
/// Source: HOPE (2512.24695) §4.5, Appendix C; Eq 88, Eq 121.
///
/// These functions are bias-agnostic: attentional bias dispatch (L2/L1/Lp)
/// happens in the caller between `dgd_error` and `dgd_update`. This preserves
/// MIRAS 4-knob separation (CS-33).

use crate::tensor::{matmul_f32, outer_product_f32, frobenius_dot_f32, transpose_f32};
use crate::retention::l2_apply_retention;

/// Gradients returned by `dgd_step_backward`.
#[derive(Clone, Debug)]
pub struct DgdGrads {
    /// dL/dM_t: [d, d] gradient w.r.t. input memory state.
    pub d_m: Vec<f32>,
    /// dL/dk_t: [d] gradient w.r.t. key.
    pub d_k: Vec<f32>,
    /// dL/dv_t: [d] gradient w.r.t. value.
    pub d_v: Vec<f32>,
    /// dL/dalpha_t: scalar gradient w.r.t. retention gate.
    pub d_alpha: f32,
    /// dL/dtheta_t: scalar gradient w.r.t. learning rate gate.
    pub d_theta: f32,
}

/// Compute prediction error: `e = M @ k - v`.
///
/// - `m`: [d, d] memory matrix (row-major).
/// - `k`: [d] key vector.
/// - `v`: [d] value vector.
/// - `d`: dimension.
///
/// Returns: [d] error vector.
pub fn dgd_error(m: &[f32], k: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    debug_assert_eq!(m.len(), d * d);
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);

    let mut prediction = vec![0.0f32; d];
    matmul_f32(m, k, &mut prediction, d, d, 1);

    for i in 0..d {
        prediction[i] -= v[i];
    }
    prediction
}

/// Apply DGD update in-place: `M = (1 - alpha) * M - theta * outer(biased_error, k)`.
///
/// The caller is responsible for applying attentional bias to the error before
/// passing it here (MIRAS 4-knob separation).
///
/// - `m`: [d, d] memory matrix, updated in-place.
/// - `biased_error`: [d] error after attentional bias.
/// - `k`: [d] key vector.
/// - `alpha`: retention gate (sigmoid output).
/// - `theta`: learning rate gate (softplus output).
/// - `d`: dimension.
pub fn dgd_update(m: &mut [f32], biased_error: &[f32], k: &[f32], alpha: f32, theta: f32, d: usize) {
    debug_assert_eq!(m.len(), d * d);
    debug_assert_eq!(biased_error.len(), d);
    debug_assert_eq!(k.len(), d);

    // grad = outer(biased_error, k): [d, d]
    let mut grad = vec![0.0f32; d * d];
    outer_product_f32(biased_error, k, &mut grad);

    // M = (1 - alpha) * M - theta * grad
    l2_apply_retention(m, 1.0 - alpha);
    for i in 0..(d * d) {
        m[i] -= theta * grad[i];
    }
}

/// Convenience: combined error + update for L2 bias (identity).
///
/// Equivalent to `dgd_error` followed by `dgd_update` with biased_error = error.
/// Only correct for L2 attentional bias; other biases must call the two-step API.
///
/// Returns the error vector (useful for caching).
pub fn dgd_step(m: &mut [f32], k: &[f32], v: &[f32], alpha: f32, theta: f32, d: usize) -> Vec<f32> {
    let error = dgd_error(m, k, v, d);
    dgd_update(m, &error, k, alpha, theta, d);
    error
}

/// Analytical VJP for `dgd_step` (L2 bias only).
///
/// Given upstream gradient dL/dM_{t+1}, computes gradients w.r.t. all inputs.
///
/// Derivation (HOPE Appendix C, spec lines 171-186):
///   dL/dM_t     = (1-alpha) * dM_out - theta * dM_out @ (k @ k^T)
///   dL/dk_t     = -theta * (M_t^T @ dM_out @ k + E_t^T @ dM_out)
///   dL/dv_t     = theta * dM_out @ k
///   dL/dalpha_t = -trace(M_t^T @ dM_out)
///   dL/dtheta_t = -trace((E @ k^T)^T @ dM_out)
///
/// - `d_m_out`: [d, d] upstream gradient dL/dM_{t+1}.
/// - `m_prev`: [d, d] memory state BEFORE the update (M_t).
/// - `k`: [d] key vector.
/// - `v`: [d] value vector.
/// - `alpha`: retention gate.
/// - `theta`: learning rate gate.
/// - `d`: dimension.
pub fn dgd_step_backward(
    d_m_out: &[f32],
    m_prev: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    theta: f32,
    d: usize,
) -> DgdGrads {
    debug_assert_eq!(d_m_out.len(), d * d);
    debug_assert_eq!(m_prev.len(), d * d);
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);

    // E = M_prev @ k - v (prediction error at time t)
    let error = dgd_error(m_prev, k, v, d);

    // d_m = (1 - alpha) * d_m_out - theta * d_m_out @ (k @ k^T)
    // The second term: d_m_out @ k gives [d,1], then outer with k gives [d,d]
    let mut dm_out_k = vec![0.0f32; d]; // d_m_out @ k: [d]
    matmul_f32(d_m_out, k, &mut dm_out_k, d, d, 1);

    let mut d_m = vec![0.0f32; d * d];
    let mut kkt_term = vec![0.0f32; d * d]; // outer(d_m_out @ k, k)
    outer_product_f32(&dm_out_k, k, &mut kkt_term);

    for i in 0..(d * d) {
        d_m[i] = (1.0 - alpha) * d_m_out[i] - theta * kkt_term[i];
    }

    // d_k = -theta * (M_t^T @ d_m_out @ k + E^T @ d_m_out)
    // Term 1: M_t^T @ (d_m_out @ k) = M_t^T @ dm_out_k
    let mut m_t_transposed = vec![0.0f32; d * d];
    transpose_f32(m_prev, &mut m_t_transposed, d, d);
    let mut mt_dm_k = vec![0.0f32; d]; // M_t^T @ dm_out_k
    matmul_f32(&m_t_transposed, &dm_out_k, &mut mt_dm_k, d, d, 1);

    // Term 2: E^T @ d_m_out (E is [d], d_m_out is [d,d]) → treat E as [1,d] @ [d,d] → [1,d]
    // This is sum_i error[i] * d_m_out[i*d + j] for each j
    let mut et_dm = vec![0.0f32; d];
    // E^T @ d_m_out: [1,d] @ [d,d] → [1,d]
    for j in 0..d {
        let mut sum = 0.0f32;
        for i in 0..d {
            sum += error[i] * d_m_out[i * d + j];
        }
        et_dm[j] = sum;
    }

    let mut d_k = vec![0.0f32; d];
    for i in 0..d {
        d_k[i] = -theta * (mt_dm_k[i] + et_dm[i]);
    }

    // d_v = theta * d_m_out @ k
    let mut d_v = vec![0.0f32; d];
    for i in 0..d {
        d_v[i] = theta * dm_out_k[i];
    }

    // d_alpha = -trace(M_t^T @ d_m_out) = -frobenius_dot(M_t, d_m_out)
    let d_alpha = -frobenius_dot_f32(m_prev, d_m_out);

    // d_theta = -trace((E @ k^T)^T @ d_m_out) = -frobenius_dot(E @ k^T, d_m_out)
    let mut grad_ekt = vec![0.0f32; d * d];
    outer_product_f32(&error, k, &mut grad_ekt);
    let d_theta = -frobenius_dot_f32(&grad_ekt, d_m_out);

    DgdGrads { d_m, d_k, d_v, d_alpha, d_theta }
}

/// DGD with momentum accumulation (Delta Momentum, HOPE §4.4).
///
/// S = beta * S + theta * grad
/// M = (1 - alpha) * M - S
///
/// - `m`: [d, d] memory matrix, updated in-place.
/// - `s`: [d, d] momentum accumulator, updated in-place.
/// - `k`: [d] key vector.
/// - `v`: [d] value vector.
/// - `alpha`: retention gate.
/// - `theta`: learning rate gate.
/// - `beta`: momentum gate.
/// - `d`: dimension.
pub fn dgd_momentum_step(
    m: &mut [f32],
    s: &mut [f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    theta: f32,
    beta: f32,
    d: usize,
) {
    debug_assert_eq!(m.len(), d * d);
    debug_assert_eq!(s.len(), d * d);
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);

    let error = dgd_error(m, k, v, d);

    // grad = outer(error, k)
    let mut grad = vec![0.0f32; d * d];
    outer_product_f32(&error, k, &mut grad);

    // S = beta * S + theta * grad
    for i in 0..(d * d) {
        s[i] = beta * s[i] + theta * grad[i];
    }

    // M = (1 - alpha) * M - S
    l2_apply_retention(m, 1.0 - alpha);
    for i in 0..(d * d) {
        m[i] -= s[i];
    }
}

/// Sherman-Morrison closed-form DGD (HOPE Appendix C, Eq 121).
///
/// eta' = eta / (1 + eta)
/// M = (I - eta' * k @ k^T) * M + eta' * v @ k^T
///
/// Assumes ||k|| = phi (normalized keys from layer-norm).
///
/// - `m`: [d, d] memory matrix, updated in-place.
/// - `k`: [d] key vector (normalized).
/// - `v`: [d] value vector.
/// - `eta`: proximal step size.
/// - `d`: dimension.
pub fn dgd_sherman_morrison(m: &mut [f32], k: &[f32], v: &[f32], eta: f32, d: usize) {
    debug_assert_eq!(m.len(), d * d);
    debug_assert_eq!(k.len(), d);
    debug_assert_eq!(v.len(), d);

    let eta_prime = eta / (1.0 + eta);

    // Compute M @ k first (before modifying M)
    let mut mk = vec![0.0f32; d];
    matmul_f32(m, k, &mut mk, d, d, 1);

    // M_{t+1}[i][j] = M[i][j] - eta' * (M@k)[i] * k[j] + eta' * v[i] * k[j]
    //               = M[i][j] + eta' * (v[i] - (M@k)[i]) * k[j]
    //               = M[i][j] - eta' * error[i] * k[j]
    // where error = M@k - v
    for i in 0..d {
        let correction = eta_prime * (v[i] - mk[i]);
        for j in 0..d {
            m[i * d + j] += correction * k[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgd_error_basic() {
        let d = 2;
        // M = [[1,0],[0,1]] (identity), k = [1,0], v = [0.5, 0.5]
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        let error = dgd_error(&m, &k, &v, d);
        // M@k = [1,0], error = [1-0.5, 0-0.5] = [0.5, -0.5]
        assert!((error[0] - 0.5).abs() < 1e-6);
        assert!((error[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dgd_step_modifies_m() {
        let d = 2;
        let mut m = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        let m_before = m.clone();
        dgd_step(&mut m, &k, &v, 0.1, 0.5, d);
        assert_ne!(m, m_before);
    }

    #[test]
    fn test_sherman_morrison_reduces_error() {
        let d = 3;
        let mut m = vec![0.0f32; d * d];
        let k = vec![1.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0];

        let err_before: f32 = dgd_error(&m, &k, &v, d).iter().map(|e| e * e).sum();
        dgd_sherman_morrison(&mut m, &k, &v, 1.0, d);
        let err_after: f32 = dgd_error(&m, &k, &v, d).iter().map(|e| e * e).sum();

        assert!(err_after < err_before, "SM should reduce error: {err_after} >= {err_before}");
    }
}
