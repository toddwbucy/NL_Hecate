/// Gradient orchestration and verification.
///
/// Provides:
/// - `compute_gradients`: main API for computing SWA parameter gradients
/// - `mag_compute_gradients`: main API for computing MAG parameter gradients
/// - `finite_diff_gradient`: central finite differences for verification
/// - Gradient checking utilities

use crate::model::{SWAConfig, SWAParams, MAGConfig, MAGParams};
use crate::forward::forward;
use crate::backward::backward_full;
use crate::mag::{mag_forward, mag_backward};

/// Compute gradients of loss with respect to all SWA parameters.
/// This is the main training API — used by Python bindings in Phase 3.
#[allow(dead_code)]
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

/// Compute gradients of loss with respect to all MAG parameters.
#[allow(dead_code)]
pub fn mag_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MAGParams) {
    let (loss, cache) = mag_forward(params, cfg, input_ids, target_ids);
    let grads = mag_backward(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Compute finite-difference gradient for a single weight element (SWA).
/// Uses central differences: (f(x+eps) - f(x-eps)) / (2*eps).
#[allow(dead_code)]
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

/// Compute finite-difference gradient for a single weight element (MAG).
#[allow(dead_code)]
fn mag_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = mag_forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = mag_forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (SWA).
/// Returns (num_checked, num_passed, max_relative_error).
///
/// Uses relative error with denominator = max(|a|, |b|, abs_threshold) to
/// avoid division by near-zero values. Gradients where both analytical and
/// numerical are below `abs_threshold` are auto-passed (below FD resolution).
#[allow(dead_code)]
pub(crate) fn check_weight_gradient(
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

    #[cfg(feature = "cuda")]
    let abs_threshold = 5e-3;
    #[cfg(not(feature = "cuda"))]
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

/// Check gradient for a specific weight matrix (MAG).
/// Returns (num_checked, num_passed, max_relative_error).
#[allow(dead_code)]
pub(crate) fn mag_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();

    let abs_threshold = 5e-4;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = mag_fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );

        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());

        checked += 1;

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
    const FD_EPS: f32 = 1e-2;
    /// Tolerance: accounts for both FD truncation and f32 rounding.
    #[cfg(not(feature = "cuda"))]
    const FD_TOL: f32 = 0.10;
    #[cfg(feature = "cuda")]
    const FD_TOL: f32 = 0.20;

    // ── SWA gradient checks (unchanged from Zero-A) ──────────────────

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

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_embed",
            |p| &p.w_embed, |p, i, v| p.w_embed[i] = v, |g| &g.w_embed,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_embed: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_embed: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── MAG gradient checks ──────────────────────────────────────────

    fn mag_grad_check_config() -> MAGConfig {
        MAGConfig::test_config()
    }

    fn mag_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    /// Init MAG params with neutral gate biases for gradient checking.
    /// b_alpha=0 → sigmoid(0)=0.5 (50/50 retention) gives larger memory gradients.
    /// b_theta=0 → softplus(0)=ln(2)≈0.69 (larger learning rate) gives larger signal.
    fn mag_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        params.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        params.b_theta = vec![0.0f32];  // softplus(0)=ln(2)≈0.69
        params
    }

    /// Crown jewel: W_K_mem gradient flows THROUGH memory recurrence.
    #[test]
    fn test_mag_gradient_w_k_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_k_mem",
            |p| &p.w_k_mem, |p, i, v| p.w_k_mem[i] = v, |g| &g.w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_v_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_v_mem",
            |p| &p.w_v_mem, |p, i, v| p.w_v_mem[i] = v, |g| &g.w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_q_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_q_mem",
            |p| &p.w_q_mem, |p, i, v| p.w_q_mem[i] = v, |g| &g.w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_alpha() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_alpha",
            |p| &p.w_alpha, |p, i, v| p.w_alpha[i] = v, |g| &g.w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_theta() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_theta",
            |p| &p.w_theta, |p, i, v| p.w_theta[i] = v, |g| &g.w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_b_alpha() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "b_alpha",
            |p| &p.b_alpha, |p, i, v| p.b_alpha[i] = v, |g| &g.b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_b_theta() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "b_theta",
            |p| &p.b_theta, |p, i, v| p.b_theta[i] = v, |g| &g.b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    /// Regression: SWA weights still get correct gradients when embedded in MAG.
    #[test]
    fn test_mag_gradient_swa_w_o() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "swa_w_o",
            |p| &p.swa.w_o, |p, i, v| p.swa.w_o[i] = v, |g| &g.swa.w_o,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("swa_w_o: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "swa_w_o: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    /// Training convergence: loss decreases over SGD steps.
    #[test]
    fn test_mag_training_convergence() {
        let cfg = mag_grad_check_config();
        let mut params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);

        let (initial_loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        eprintln!("MAG initial loss: {initial_loss:.4}");

        let lr = 0.01;
        let steps = 1000;
        for _ in 0..steps {
            let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);
            params.sgd_step(&grads, lr);
        }

        let (final_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        eprintln!("MAG final loss after {steps} steps: {final_loss:.4}");

        assert!(final_loss < initial_loss,
            "Loss should decrease: initial={initial_loss:.4}, final={final_loss:.4}");

        // Verify memory evolved meaningfully
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let m_t = &cache.delta_cache.m_states[s * d * d..(s + 1) * d * d];
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("MAG final memory norm: {mt_norm:.4e}");
        assert!(mt_norm > 1e-6, "Memory should evolve during training, norm={mt_norm}");
    }
}
