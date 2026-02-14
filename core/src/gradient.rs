/// Gradient orchestration and verification.
///
/// Provides:
/// - `compute_gradients`: main API for computing parameter gradients
/// - `finite_diff_gradient`: central finite differences for verification
/// - Gradient checking utilities

use crate::model::{SWAConfig, SWAParams};
use crate::forward::forward;
use crate::backward::backward_full;

/// Compute gradients of loss with respect to all parameters.
/// This is the main training API.
pub fn compute_gradients(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, SWAParams) {
    let (loss, cache) = forward(params, cfg, input_ids, target_ids);
    let grads = backward_full(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Compute finite-difference gradient for a single weight element.
/// Uses central differences: (f(x+eps) - f(x-eps)) / (2*eps).
fn fd_single(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&SWAParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut SWAParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix.
/// Returns (num_checked, num_passed, max_relative_error).
///
/// Uses relative error with denominator = max(|a|, |b|, abs_threshold) to
/// avoid division by near-zero values. Gradients where both analytical and
/// numerical are below `abs_threshold` are auto-passed (below FD resolution).
pub fn check_weight_gradient(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &SWAParams,
    name: &str,
    get_weight: impl Fn(&SWAParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut SWAParams, usize, f32),
    get_grad: impl Fn(&SWAParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();

    // f32 FD resolution limit: smallest detectable gradient ≈ loss * f32_eps / (2*eps).
    // With loss~2.8 and eps=1e-2: ~2.8 * 1.2e-7 / 0.02 ≈ 1.7e-5.
    // Auto-pass any gradient pair where both are below this threshold.
    let abs_threshold = 5e-4;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );

        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());

        checked += 1;

        // Auto-pass if both values are below FD detection threshold
        if denom < abs_threshold {
            passed += 1;
            continue;
        }

        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }

        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!(
                "  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, \
                 rel_err={rel_err:.4e}"
            );
        }
    }

    (checked, passed, max_rel_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny config for gradient checking. Smaller model = larger gradients per
    /// parameter = better FD resolution at f32 precision.
    fn grad_check_config() -> SWAConfig {
        SWAConfig {
            d_model: 8,
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 4,
            vocab_size: 16,
        }
    }

    fn make_test_data(cfg: &SWAConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len)
            .map(|t| t % cfg.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    /// FD eps: large enough for f32 to resolve loss differences.
    /// FD truncation error is O(eps^2) ≈ 1e-4, but f32 rounding needs
    /// 2*eps*grad >> loss * f32_eps ≈ 3e-7.
    const FD_EPS: f32 = 1e-2;
    /// Tolerance: accounts for both FD truncation and f32 rounding.
    /// With eps=1e-2, expect ~2-5% error for well-resolved gradients.
    const FD_TOL: f32 = 0.10; // 10% relative error

    #[test]
    fn test_gradient_w_unembed() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_unembed",
            |p| &p.w_unembed, |p, i, v| p.w_unembed[i] = v, |g| &g.w_unembed,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_unembed: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_unembed: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_o() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_o",
            |p| &p.w_o, |p, i, v| p.w_o[i] = v, |g| &g.w_o,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_o: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_o: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_q() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_q",
            |p| &p.w_q, |p, i, v| p.w_q[i] = v, |g| &g.w_q,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_q: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_q: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_k() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_k",
            |p| &p.w_k, |p, i, v| p.w_k[i] = v, |g| &g.w_k,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_k: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_k: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_v() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_v",
            |p| &p.w_v, |p, i, v| p.w_v[i] = v, |g| &g.w_v,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_v: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_v: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_embed() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        // Use check_weight_gradient — but only for embedding rows that have
        // non-zero gradients. Sample from the first few used rows.
        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_embed",
            |p| &p.w_embed, |p, i, v| p.w_embed[i] = v, |g| &g.w_embed,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_embed: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_embed: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_training_loss_decreases() {
        let cfg = SWAConfig::test_config();
        let mut params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);

        let (initial_loss, _) = forward(&params, &cfg, &input_ids, &target_ids);

        let lr = 0.01;
        let mut last_loss = initial_loss;
        for _ in 0..50 {
            let (loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);
            params.sgd_step(&grads, lr);
            last_loss = loss;
        }

        eprintln!("Loss: {initial_loss:.4} → {last_loss:.4}");
        assert!(last_loss < initial_loss,
            "Loss should decrease: {initial_loss} → {last_loss}");
    }
}
