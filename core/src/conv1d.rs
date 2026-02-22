/// Depthwise causal Conv1D with SiLU activation for key/query preprocessing.
///
/// Per spec: specs/infrastructure/attention/02_short_conv.md
/// HOPE §6 / Atlas §2.1 / Mamba convention: a short depthwise causal conv
/// (kernel_size=4, SiLU) is applied to keys and queries AFTER linear projection
/// but BEFORE the memory module.
///
/// Operation:
///   1. Left-pad input with K-1 zeros for causality
///   2. Depthwise convolution (each channel independent)
///   3. SiLU activation: out = z * sigmoid(z)
///
/// Values are NOT convolved — only keys and queries get local receptive field.

use crate::model::MemoryLevelParams;

/// Cached intermediates needed for backward pass through Conv1D + SiLU.
pub struct Conv1DCache {
    /// Input before convolution: [seq_len, d]
    pub pre_conv: Vec<f32>,
    /// Convolution output before SiLU: [seq_len, d]
    pub pre_silu: Vec<f32>,
}

/// Depthwise causal Conv1D forward with SiLU activation.
///
/// `x`: [seq_len, d] — input (projected k or q)
/// `w`: [d, kernel_size] — depthwise conv weights
/// `bias`: [d] — per-channel bias
///
/// Returns (output [seq_len, d], pre_silu [seq_len, d]).
pub fn causal_conv1d_forward(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    seq_len: usize,
    d: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(x.len(), seq_len * d);
    debug_assert_eq!(w.len(), d * kernel_size);
    debug_assert_eq!(bias.len(), d);

    let mut z = vec![0.0f32; seq_len * d]; // pre-SiLU
    let mut out = vec![0.0f32; seq_len * d];

    // Depthwise causal conv: out[t,c] = sum_{j=0}^{K-1} w[c,j] * x_pad[t+j,c] + bias[c]
    // x_pad is left-padded with K-1 zeros, so x_pad[t+j, c] = x[t+j-(K-1), c] when t+j >= K-1
    for t in 0..seq_len {
        for c in 0..d {
            let mut acc = bias[c];
            for j in 0..kernel_size {
                // In padded coordinates: index = t + j
                // In original coordinates: orig = t + j - (K-1) = t - (K-1-j)
                let orig = t as isize - (kernel_size as isize - 1 - j as isize);
                if orig >= 0 {
                    acc += w[c * kernel_size + j] * x[orig as usize * d + c];
                }
            }
            z[t * d + c] = acc;
        }
    }

    // SiLU: out = z * sigmoid(z)
    for i in 0..seq_len * d {
        let sig = 1.0 / (1.0 + (-z[i]).exp());
        out[i] = z[i] * sig;
    }

    (out, z)
}

/// Depthwise causal Conv1D backward through SiLU and convolution.
///
/// `d_out`: [seq_len, d] — upstream gradient
/// `x`: [seq_len, d] — pre-conv input (saved from forward)
/// `w`: [d, kernel_size] — conv weights
/// `pre_silu`: [seq_len, d] — z values before SiLU (saved from forward)
///
/// Returns (d_x [seq_len, d], d_w [d, kernel_size], d_bias [d]).
pub fn causal_conv1d_backward(
    d_out: &[f32],
    x: &[f32],
    w: &[f32],
    pre_silu: &[f32],
    seq_len: usize,
    d: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    debug_assert_eq!(d_out.len(), seq_len * d);
    debug_assert_eq!(x.len(), seq_len * d);
    debug_assert_eq!(w.len(), d * kernel_size);
    debug_assert_eq!(pre_silu.len(), seq_len * d);

    // Step 1: SiLU backward — dL/dz = dL/dout * silu'(z)
    // silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
    let mut d_z = vec![0.0f32; seq_len * d];
    for i in 0..seq_len * d {
        let z = pre_silu[i];
        let sig = 1.0 / (1.0 + (-z).exp());
        let silu_grad = sig * (1.0 + z * (1.0 - sig));
        d_z[i] = d_out[i] * silu_grad;
    }

    // Step 2: Conv backward
    let mut d_x = vec![0.0f32; seq_len * d];
    let mut d_w = vec![0.0f32; d * kernel_size];
    let mut d_bias = vec![0.0f32; d];

    for t in 0..seq_len {
        for c in 0..d {
            let dz_tc = d_z[t * d + c];
            d_bias[c] += dz_tc;
            for j in 0..kernel_size {
                let orig = t as isize - (kernel_size as isize - 1 - j as isize);
                if orig >= 0 {
                    let orig_idx = orig as usize;
                    // d_w[c, j] += dz[t, c] * x[orig, c]
                    d_w[c * kernel_size + j] += dz_tc * x[orig_idx * d + c];
                    // d_x[orig, c] += dz[t, c] * w[c, j]
                    d_x[orig_idx * d + c] += dz_tc * w[c * kernel_size + j];
                }
            }
        }
    }

    (d_x, d_w, d_bias)
}

// ── DRY helpers for memory rules ────────────────────────────────────

/// Apply Conv1D to projected k_mem and q_mem in-place if conv weights are present.
///
/// Returns (k_conv_cache, q_conv_cache) — None if conv is not active (empty weights).
pub fn apply_conv1d_to_kq(
    k_mem: &mut Vec<f32>,
    q_mem: &mut Vec<f32>,
    level_params: &MemoryLevelParams,
    seq_len: usize,
    d: usize,
) -> (Option<Conv1DCache>, Option<Conv1DCache>) {
    if level_params.w_k_conv.is_empty() {
        return (None, None);
    }
    let kernel_size = level_params.w_k_conv.len() / d;
    debug_assert_eq!(level_params.w_k_conv.len(), d * kernel_size);
    debug_assert_eq!(level_params.w_q_conv.len(), d * kernel_size);

    // Save pre-conv inputs
    let k_pre = k_mem.clone();
    let q_pre = q_mem.clone();

    // Apply conv to keys
    let (k_out, k_pre_silu) = causal_conv1d_forward(
        &k_pre, &level_params.w_k_conv, &level_params.b_k_conv,
        seq_len, d, kernel_size,
    );
    *k_mem = k_out;

    // Apply conv to queries
    let (q_out, q_pre_silu) = causal_conv1d_forward(
        &q_pre, &level_params.w_q_conv, &level_params.b_q_conv,
        seq_len, d, kernel_size,
    );
    *q_mem = q_out;

    let k_cache = Conv1DCache { pre_conv: k_pre, pre_silu: k_pre_silu };
    let q_cache = Conv1DCache { pre_conv: q_pre, pre_silu: q_pre_silu };
    (Some(k_cache), Some(q_cache))
}

/// Backward through Conv1D for k_mem and q_mem gradients.
///
/// Transforms d_k_mem and d_q_mem in-place to be gradients w.r.t. the pre-conv
/// projections, and accumulates weight/bias gradients into `grads`.
pub fn backward_conv1d_kq(
    d_k_mem: &mut Vec<f32>,
    d_q_mem: &mut Vec<f32>,
    k_cache: &Option<Conv1DCache>,
    q_cache: &Option<Conv1DCache>,
    level_params: &MemoryLevelParams,
    grads: &mut MemoryLevelParams,
    seq_len: usize,
    d: usize,
) {
    let (Some(k_c), Some(q_c)) = (k_cache, q_cache) else { return };
    let kernel_size = level_params.w_k_conv.len() / d;

    // Backward through key conv
    let (d_k_input, d_wk, d_bk) = causal_conv1d_backward(
        d_k_mem, &k_c.pre_conv, &level_params.w_k_conv, &k_c.pre_silu,
        seq_len, d, kernel_size,
    );
    *d_k_mem = d_k_input;

    // Accumulate key conv weight gradients
    for i in 0..d_wk.len() { grads.w_k_conv[i] += d_wk[i]; }
    for i in 0..d_bk.len() { grads.b_k_conv[i] += d_bk[i]; }

    // Backward through query conv
    let (d_q_input, d_wq, d_bq) = causal_conv1d_backward(
        d_q_mem, &q_c.pre_conv, &level_params.w_q_conv, &q_c.pre_silu,
        seq_len, d, kernel_size,
    );
    *d_q_mem = d_q_input;

    // Accumulate query conv weight gradients
    for i in 0..d_wq.len() { grads.w_q_conv[i] += d_wq[i]; }
    for i in 0..d_bq.len() { grads.b_q_conv[i] += d_bq[i]; }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[test]
    fn test_conv1d_identity_kernel1() {
        // kernel_size=1, w=1, bias=0 → output should be SiLU(x)
        let seq_len = 4;
        let d = 3;
        let x: Vec<f32> = (0..seq_len * d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w = vec![1.0f32; d]; // [d, 1]
        let bias = vec![0.0f32; d];

        let (out, pre_silu) = causal_conv1d_forward(&x, &w, &bias, seq_len, d, 1);

        // With kernel_size=1, w=1, bias=0: z = x, out = SiLU(x)
        for i in 0..seq_len * d {
            assert!((pre_silu[i] - x[i]).abs() < 1e-6,
                "pre_silu[{i}]: expected {}, got {}", x[i], pre_silu[i]);
            let expected = silu(x[i]);
            assert!((out[i] - expected).abs() < 1e-6,
                "out[{i}]: expected {expected}, got {}", out[i]);
        }
    }

    #[test]
    fn test_conv1d_causality() {
        // Output[t] must depend only on input[max(0, t-K+1)..=t]
        let seq_len = 8;
        let d = 2;
        let kernel_size = 4;
        let w: Vec<f32> = (0..d * kernel_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let bias = vec![0.0f32; d];

        // Create base input
        let x_base = vec![1.0f32; seq_len * d];
        let (out_base, _) = causal_conv1d_forward(&x_base, &w, &bias, seq_len, d, kernel_size);

        // Perturb token at position 4 (0-indexed)
        let perturb_t = 4;
        let mut x_perturbed = x_base.clone();
        for c in 0..d {
            x_perturbed[perturb_t * d + c] = 5.0;
        }
        let (out_perturbed, _) = causal_conv1d_forward(&x_perturbed, &w, &bias, seq_len, d, kernel_size);

        // Tokens before perturb_t should be unchanged (causal: no future leakage)
        for t in 0..perturb_t {
            for c in 0..d {
                assert!((out_base[t * d + c] - out_perturbed[t * d + c]).abs() < 1e-6,
                    "Non-causal leakage at t={t}, c={c}");
            }
        }

        // Tokens at perturb_t..min(perturb_t+K, seq_len) should differ
        for t in perturb_t..std::cmp::min(perturb_t + kernel_size, seq_len) {
            let any_diff = (0..d).any(|c| {
                (out_base[t * d + c] - out_perturbed[t * d + c]).abs() > 1e-6
            });
            assert!(any_diff, "Expected output change at t={t} but found none");
        }

        // Tokens beyond perturb_t + K - 1 should be unchanged
        for t in (perturb_t + kernel_size)..seq_len {
            for c in 0..d {
                assert!((out_base[t * d + c] - out_perturbed[t * d + c]).abs() < 1e-6,
                    "Unexpected influence beyond receptive field at t={t}, c={c}");
            }
        }
    }

    #[test]
    fn test_conv1d_known_values() {
        // Hand-computed: kernel_size=2, seq_len=3, d=1
        // w = [0.5, 0.3] (shape [1, 2]), bias = [0.1]
        // x = [1.0, 2.0, 3.0] (shape [3, 1])
        //
        // Left-pad: x_pad = [0.0, 1.0, 2.0, 3.0]
        // z[0] = 0.5 * 0.0 + 0.3 * 1.0 + 0.1 = 0.4
        // z[1] = 0.5 * 1.0 + 0.3 * 2.0 + 0.1 = 1.2
        // z[2] = 0.5 * 2.0 + 0.3 * 3.0 + 0.1 = 2.0
        let x = vec![1.0f32, 2.0, 3.0];
        let w = vec![0.5f32, 0.3];
        let bias = vec![0.1f32];

        let (out, pre_silu) = causal_conv1d_forward(&x, &w, &bias, 3, 1, 2);

        assert!((pre_silu[0] - 0.4).abs() < 1e-5, "z[0] = {}", pre_silu[0]);
        assert!((pre_silu[1] - 1.2).abs() < 1e-5, "z[1] = {}", pre_silu[1]);
        assert!((pre_silu[2] - 2.0).abs() < 1e-5, "z[2] = {}", pre_silu[2]);

        for i in 0..3 {
            let expected = silu(pre_silu[i]);
            assert!((out[i] - expected).abs() < 1e-5,
                "out[{i}]: expected {expected}, got {}", out[i]);
        }
    }

    #[test]
    fn test_conv1d_backward_fd() {
        // Finite-difference gradient check
        let seq_len = 4;
        let d = 3;
        let kernel_size = 2;
        let eps = 1e-3f32;
        let tol = 0.02; // 2% relative tolerance

        let x: Vec<f32> = (0..seq_len * d).map(|i| ((i as f32) * 0.37 - 1.0)).collect();
        let w: Vec<f32> = (0..d * kernel_size).map(|i| (i as f32 + 1.0) * 0.15).collect();
        let bias: Vec<f32> = (0..d).map(|i| i as f32 * 0.05).collect();

        let (out, pre_silu) = causal_conv1d_forward(&x, &w, &bias, seq_len, d, kernel_size);

        // Use sum(out) as loss for FD
        let d_out = vec![1.0f32; seq_len * d];
        let (d_x, d_w, d_bias) = causal_conv1d_backward(
            &d_out, &x, &w, &pre_silu, seq_len, d, kernel_size);

        // FD check for d_x
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            x_plus[i] += eps;
            let (out_plus, _) = causal_conv1d_forward(&x_plus, &w, &bias, seq_len, d, kernel_size);
            let mut x_minus = x.clone();
            x_minus[i] -= eps;
            let (out_minus, _) = causal_conv1d_forward(&x_minus, &w, &bias, seq_len, d, kernel_size);
            let fd: f32 = (out_plus.iter().sum::<f32>() - out_minus.iter().sum::<f32>()) / (2.0 * eps);
            let ana = d_x[i];
            if ana.abs() > 1e-4 {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol, "d_x[{i}]: ana={ana}, fd={fd}, rel_err={rel_err}");
            } else {
                assert!((ana - fd).abs() < 1e-3, "d_x[{i}]: ana={ana}, fd={fd}");
            }
        }

        // FD check for d_w
        for i in 0..w.len() {
            let mut w_plus = w.clone();
            w_plus[i] += eps;
            let (out_plus, _) = causal_conv1d_forward(&x, &w_plus, &bias, seq_len, d, kernel_size);
            let mut w_minus = w.clone();
            w_minus[i] -= eps;
            let (out_minus, _) = causal_conv1d_forward(&x, &w_minus, &bias, seq_len, d, kernel_size);
            let fd: f32 = (out_plus.iter().sum::<f32>() - out_minus.iter().sum::<f32>()) / (2.0 * eps);
            let ana = d_w[i];
            if ana.abs() > 1e-4 {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol, "d_w[{i}]: ana={ana}, fd={fd}, rel_err={rel_err}");
            } else {
                assert!((ana - fd).abs() < 1e-3, "d_w[{i}]: ana={ana}, fd={fd}");
            }
        }

        // FD check for d_bias
        for i in 0..bias.len() {
            let mut b_plus = bias.clone();
            b_plus[i] += eps;
            let (out_plus, _) = causal_conv1d_forward(&x, &w, &b_plus, seq_len, d, kernel_size);
            let mut b_minus = bias.clone();
            b_minus[i] -= eps;
            let (out_minus, _) = causal_conv1d_forward(&x, &w, &b_minus, seq_len, d, kernel_size);
            let fd: f32 = (out_plus.iter().sum::<f32>() - out_minus.iter().sum::<f32>()) / (2.0 * eps);
            let ana = d_bias[i];
            if ana.abs() > 1e-4 {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol, "d_bias[{i}]: ana={ana}, fd={fd}, rel_err={rel_err}");
            } else {
                assert!((ana - fd).abs() < 1e-3, "d_bias[{i}]: ana={ana}, fd={fd}");
            }
        }
    }

    // ── Integration tests: conv1d with memory rules ──────────────────

    use crate::tensor::SimpleRng;
    use crate::delta_rule::{DeltaRule, MemoryRule};
    use crate::tape::OpaqueVjp;

    /// Helper: create level params with conv1d weights initialized.
    fn params_with_conv(d: usize, kernel_size: usize, seed: u64) -> crate::model::MemoryLevelParams {
        let mut rng = SimpleRng::new(seed);
        let mut params = crate::model::MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0);
        params.init_conv(d, kernel_size, &mut rng);
        params
    }

    #[test]
    fn test_delta_rule_conv1d() {
        // Conv1D with kernel_size=4 should change output compared to no conv
        let d = 8;
        let seq_len = 6;
        let mut rng = SimpleRng::new(42);
        let params_no_conv = crate::model::MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0);
        let params_conv = params_with_conv(d, 4, 42);
        let mut embedded = vec![0.0f32; seq_len * d];
        SimpleRng::new(99).fill_uniform(&mut embedded, 0.5);

        let rule = DeltaRule::l2();
        let (y_no_conv, _) = rule.step(&params_no_conv, &embedded, seq_len, d, None);
        let (y_conv, cache_conv) = rule.step(&params_conv, &embedded, seq_len, d, None);

        // Outputs should differ (conv transforms k/q before memory loop)
        let diff: f32 = y_no_conv.iter().zip(y_conv.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "Conv1D should change output, diff={diff}");

        // Conv cache should be populated
        assert!(cache_conv.k_conv_cache.is_some(), "k_conv_cache should be Some");
        assert!(cache_conv.q_conv_cache.is_some(), "q_conv_cache should be Some");
    }

    #[test]
    fn test_titans_conv1d() {
        // Conv1D with kernel_size=4 should change Titans output compared to no conv
        use crate::titans_lmm::TitansLMM;
        let d = 8;
        let seq_len = 6;
        let mut rng = SimpleRng::new(42);
        let params_no_conv = crate::model::MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0);
        let params_conv = params_with_conv(d, 4, 42);
        let mut embedded = vec![0.0f32; seq_len * d];
        SimpleRng::new(99).fill_uniform(&mut embedded, 0.5);

        let rule = TitansLMM::l2();
        let (y_no_conv, _) = rule.step(&params_no_conv, &embedded, seq_len, d, None);
        let (y_conv, cache_conv) = rule.step(&params_conv, &embedded, seq_len, d, None);

        let diff: f32 = y_no_conv.iter().zip(y_conv.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "Conv1D should change Titans output, diff={diff}");
        assert!(cache_conv.k_conv_cache.is_some());
        assert!(cache_conv.q_conv_cache.is_some());
    }

    #[test]
    fn test_conv1d_full_gradient_fd() {
        // FD gradient check through full Delta Rule with Conv1D active
        let d = 4;
        let seq_len = 3;
        let eps = 1e-2f32;
        let tol = 0.10; // 10% tolerance for FD through complex pipeline
        let abs_thr = 5e-4;

        let params = params_with_conv(d, 2, 42);
        let mut embedded = vec![0.0f32; seq_len * d];
        SimpleRng::new(99).fill_uniform(&mut embedded, 0.5);

        let rule = DeltaRule::l2();
        let (y_base, cache) = rule.step(&params, &embedded, seq_len, d, None);
        let loss_base: f32 = y_base.iter().map(|&v| v * v).sum::<f32>() * 0.5;

        // Analytical gradient via backward
        let d_y: Vec<f32> = y_base.clone(); // d(0.5*||y||^2)/dy = y
        let (param_grads, _) = rule.step_backward(&params, &cache, &d_y, &embedded);

        // Check conv weight gradients via FD
        let kernel_size = 2;
        for i in 0..d * kernel_size {
            let mut p_plus = params.clone();
            p_plus.w_k_conv[i] += eps;
            let (y_plus, _) = rule.step(&p_plus, &embedded, seq_len, d, None);
            let loss_plus: f32 = y_plus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let mut p_minus = params.clone();
            p_minus.w_k_conv[i] -= eps;
            let (y_minus, _) = rule.step(&p_minus, &embedded, seq_len, d, None);
            let loss_minus: f32 = y_minus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let ana = param_grads.w_k_conv[i];
            if ana.abs() < abs_thr && fd.abs() < abs_thr { continue; }
            if ana.abs() > abs_thr {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol,
                    "w_k_conv[{i}]: ana={ana:.6}, fd={fd:.6}, rel_err={rel_err:.4}");
            }
        }

        // Check query conv weight gradients
        for i in 0..d * kernel_size {
            let mut p_plus = params.clone();
            p_plus.w_q_conv[i] += eps;
            let (y_plus, _) = rule.step(&p_plus, &embedded, seq_len, d, None);
            let loss_plus: f32 = y_plus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let mut p_minus = params.clone();
            p_minus.w_q_conv[i] -= eps;
            let (y_minus, _) = rule.step(&p_minus, &embedded, seq_len, d, None);
            let loss_minus: f32 = y_minus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let ana = param_grads.w_q_conv[i];
            if ana.abs() < abs_thr && fd.abs() < abs_thr { continue; }
            if ana.abs() > abs_thr {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol,
                    "w_q_conv[{i}]: ana={ana:.6}, fd={fd:.6}, rel_err={rel_err:.4}");
            }
        }

        // Check conv bias gradients
        for i in 0..d {
            let mut p_plus = params.clone();
            p_plus.b_k_conv[i] += eps;
            let (y_plus, _) = rule.step(&p_plus, &embedded, seq_len, d, None);
            let loss_plus: f32 = y_plus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let mut p_minus = params.clone();
            p_minus.b_k_conv[i] -= eps;
            let (y_minus, _) = rule.step(&p_minus, &embedded, seq_len, d, None);
            let loss_minus: f32 = y_minus.iter().map(|&v| v * v).sum::<f32>() * 0.5;

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let ana = param_grads.b_k_conv[i];
            if ana.abs() < abs_thr && fd.abs() < abs_thr { continue; }
            if ana.abs() > abs_thr {
                let rel_err = ((ana - fd) / ana).abs();
                assert!(rel_err < tol,
                    "b_k_conv[{i}]: ana={ana:.6}, fd={fd:.6}, rel_err={rel_err:.4}");
            }
        }
    }

    #[test]
    fn test_opaque_vjp_delta_conv1d() {
        // Tape roundtrip with Conv1D active: opaque backward should match direct backward
        use crate::opaque_adapters::{register_opaque_vjps, level_params_grads_to_flat};
        let d = 4;
        let seq_len = 3;
        let params = params_with_conv(d, 4, 42);
        let mut embedded = vec![0.0f32; seq_len * d];
        SimpleRng::new(99).fill_uniform(&mut embedded, 0.5);

        let rule = DeltaRule::l2();
        let (_, cache) = rule.step(&params, &embedded, seq_len, d, None);
        let d_y = vec![1.0f32; seq_len * d];
        let (pg_direct, de_direct) = rule.step_backward(&params, &cache, &d_y, &embedded);

        let registry = register_opaque_vjps();
        crate::tape::with_tape(registry, |tape| {
            let (_, y_id, emb_in, lp_in) = rule.record_on_tape(tape, &params, &embedded, seq_len, d, None);
            tape.seed_grad(y_id, d_y.clone());
            tape.backward(y_id);
            let de_tape = tape.get_grad(emb_in).unwrap();
            let dlp_tape = tape.get_grad(lp_in).unwrap();
            for (i, (&t, &d)) in de_tape.iter().zip(de_direct.iter()).enumerate() {
                assert!((t - d).abs() < 1e-5,
                    "Conv1D Delta tape d_emb[{i}]: tape={t} direct={d}");
            }
            let lp_direct = level_params_grads_to_flat(&pg_direct);
            for (i, (&t, &d)) in dlp_tape.iter().zip(lp_direct.iter()).enumerate() {
                assert!((t - d).abs() < 1e-5,
                    "Conv1D Delta tape d_lp[{i}]: tape={t} direct={d}");
            }
        });
    }

    #[test]
    fn test_opaque_vjp_titans_conv1d() {
        // Tape roundtrip with Conv1D active for Titans
        use crate::titans_lmm::TitansLMM;
        use crate::opaque_adapters::{register_opaque_vjps, level_params_grads_to_flat};
        let d = 4;
        let seq_len = 3;
        let params = params_with_conv(d, 4, 42);
        let mut embedded = vec![0.0f32; seq_len * d];
        SimpleRng::new(99).fill_uniform(&mut embedded, 0.5);

        let rule = TitansLMM::l2();
        let (_, cache) = rule.step(&params, &embedded, seq_len, d, None);
        let d_y = vec![1.0f32; seq_len * d];
        let (pg_direct, de_direct) = rule.step_backward(&params, &cache, &d_y, &embedded);

        let registry = register_opaque_vjps();
        crate::tape::with_tape(registry, |tape| {
            let (_, y_id, emb_in, lp_in) = rule.record_on_tape(tape, &params, &embedded, seq_len, d, None);
            tape.seed_grad(y_id, d_y.clone());
            tape.backward(y_id);
            let de_tape = tape.get_grad(emb_in).unwrap();
            let dlp_tape = tape.get_grad(lp_in).unwrap();
            for (i, (&t, &d)) in de_tape.iter().zip(de_direct.iter()).enumerate() {
                assert!((t - d).abs() < 1e-5,
                    "Conv1D Titans tape d_emb[{i}]: tape={t} direct={d}");
            }
            let lp_direct = level_params_grads_to_flat(&pg_direct);
            for (i, (&t, &d)) in dlp_tape.iter().zip(lp_direct.iter()).enumerate() {
                assert!((t - d).abs() < 1e-5,
                    "Conv1D Titans tape d_lp[{i}]: tape={t} direct={d}");
            }
        });
    }
}
