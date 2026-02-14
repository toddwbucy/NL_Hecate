// CUDA SWA Kernel Tests — Phase 2 Track Zero-A
//
// Three test classes:
//   1. Forward match: CUDA bf16 output vs Rust f32 reference (per-element < 1e-2)
//   2. Backward match: CUDA bf16 dQ/dK/dV vs Rust f32 backward (per-element < 5e-2)
//   3. Full pipeline FD check through CUDA path (all 6 weight matrices)
//
// Tolerances are wider than f32-vs-f32 because the CUDA path uses bf16 storage
// (~7 mantissa bits, ~0.8% relative precision) for Q/K/V/out/attn_weights.
//
// All tests gated behind #[cfg(feature = "cuda")].

#![cfg(feature = "cuda")]

use nl_hecate_core::swa::{swa_forward, swa_backward_rust};
use nl_hecate_core::tensor::SimpleRng;

/// Generate random f32 buffer with small values.
fn rand_buf(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = SimpleRng::new(seed);
    let mut buf = vec![0.0f32; len];
    rng.fill_uniform(&mut buf, 0.5);
    buf
}

/// Check per-element tolerance between two slices.
fn check_close(name: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch {} vs {}", a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff < tol,
        "{name}: max diff {max_diff:.6e} at idx {max_idx} (a={:.6e}, b={:.6e}), tol={tol:.0e}",
        a[max_idx], b[max_idx]
    );
    eprintln!("  {name}: max_diff={max_diff:.6e} (tol={tol:.0e}) ✓");
}

// ── Test Class 1: Forward match ─────────────────────────────────────

/// Use the dispatch module to call CUDA forward, then compare against Rust.
/// The dispatch module handles all device memory management.
fn cuda_forward_via_dispatch(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32], attn_weights: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    // When compiled with --features cuda, dispatch calls the CUDA kernel
    nl_hecate_core::dispatch::swa_forward_dispatch(
        q, k, v, out, attn_weights,
        seq_len, num_heads, head_dim, window_size,
    );
}

fn cuda_backward_via_dispatch(
    q: &[f32], k: &[f32], v: &[f32],
    attn_weights: &[f32], d_attn_out: &[f32],
    d_q: &mut [f32], d_k: &mut [f32], d_v: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    nl_hecate_core::dispatch::swa_backward_dispatch(
        q, k, v, attn_weights, d_attn_out, d_q, d_k, d_v,
        seq_len, num_heads, head_dim, window_size,
    );
}

#[test]
fn test_cuda_forward_matches_rust_test_config() {
    // Use the same dimensions as SWAConfig::test_config()
    let seq_len = 24;
    let num_heads = 4;
    let head_dim = 16;
    let window_size = 16;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 100);
    let k = rand_buf(seq_len * total_dim, 200);
    let v = rand_buf(seq_len * total_dim, 300);

    // Rust reference
    let mut out_rust = vec![0.0f32; seq_len * total_dim];
    let mut aw_rust = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut out_rust, &mut aw_rust,
                seq_len, num_heads, head_dim, window_size);

    // CUDA via dispatch
    let mut out_cuda = vec![0.0f32; seq_len * total_dim];
    let mut aw_cuda = vec![0.0f32; aw_len];
    cuda_forward_via_dispatch(&q, &k, &v, &mut out_cuda, &mut aw_cuda,
                              seq_len, num_heads, head_dim, window_size);

    check_close("forward_out", &out_rust, &out_cuda, 1e-2);
    check_close("forward_aw", &aw_rust, &aw_cuda, 1e-2);
}

#[test]
fn test_cuda_forward_matches_rust_small() {
    // Minimal case: seq_len=2, 1 head, head_dim=4, window=2
    let seq_len = 2;
    let num_heads = 1;
    let head_dim = 4;
    let window_size = 2;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 400);
    let k = rand_buf(seq_len * total_dim, 500);
    let v = rand_buf(seq_len * total_dim, 600);

    let mut out_rust = vec![0.0f32; seq_len * total_dim];
    let mut aw_rust = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut out_rust, &mut aw_rust,
                seq_len, num_heads, head_dim, window_size);

    let mut out_cuda = vec![0.0f32; seq_len * total_dim];
    let mut aw_cuda = vec![0.0f32; aw_len];
    cuda_forward_via_dispatch(&q, &k, &v, &mut out_cuda, &mut aw_cuda,
                              seq_len, num_heads, head_dim, window_size);

    check_close("small_forward_out", &out_rust, &out_cuda, 1e-2);
    check_close("small_forward_aw", &aw_rust, &aw_cuda, 1e-2);
}

#[test]
fn test_cuda_forward_single_position() {
    // seq_len=1: attention is trivially 1.0, output = V
    let seq_len = 1;
    let num_heads = 2;
    let head_dim = 8;
    let window_size = 4;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 700);
    let k = rand_buf(seq_len * total_dim, 800);
    let v = rand_buf(seq_len * total_dim, 900);

    let mut out_rust = vec![0.0f32; seq_len * total_dim];
    let mut aw_rust = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut out_rust, &mut aw_rust,
                seq_len, num_heads, head_dim, window_size);

    let mut out_cuda = vec![0.0f32; seq_len * total_dim];
    let mut aw_cuda = vec![0.0f32; aw_len];
    cuda_forward_via_dispatch(&q, &k, &v, &mut out_cuda, &mut aw_cuda,
                              seq_len, num_heads, head_dim, window_size);

    check_close("single_pos_out", &out_rust, &out_cuda, 1e-2);
    check_close("single_pos_aw", &aw_rust, &aw_cuda, 1e-2);
}

#[test]
fn test_cuda_forward_head_dim_32_warp_boundary() {
    // Edge case: head_dim=32 fills exactly one warp — exercises the warp
    // reduction boundary where every thread participates.
    let seq_len = 8;
    let num_heads = 1;
    let head_dim = 32;
    let window_size = 4;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 1100);
    let k = rand_buf(seq_len * total_dim, 1200);
    let v = rand_buf(seq_len * total_dim, 1300);

    let mut out_rust = vec![0.0f32; seq_len * total_dim];
    let mut aw_rust = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut out_rust, &mut aw_rust,
                seq_len, num_heads, head_dim, window_size);

    let mut out_cuda = vec![0.0f32; seq_len * total_dim];
    let mut aw_cuda = vec![0.0f32; aw_len];
    cuda_forward_via_dispatch(&q, &k, &v, &mut out_cuda, &mut aw_cuda,
                              seq_len, num_heads, head_dim, window_size);

    check_close("hd32_forward_out", &out_rust, &out_cuda, 1e-2);
    check_close("hd32_forward_aw", &aw_rust, &aw_cuda, 1e-2);
}

#[test]
fn test_cuda_backward_head_dim_32_warp_boundary() {
    // Edge case: head_dim=32 fills exactly one warp for backward too.
    let seq_len = 8;
    let num_heads = 1;
    let head_dim = 32;
    let window_size = 4;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 1400);
    let k = rand_buf(seq_len * total_dim, 1500);
    let v = rand_buf(seq_len * total_dim, 1600);

    let mut attn_out = vec![0.0f32; seq_len * total_dim];
    let mut attn_weights = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut attn_out, &mut attn_weights,
                seq_len, num_heads, head_dim, window_size);

    let d_attn_out = rand_buf(seq_len * total_dim, 1700);

    let mut dq_rust = vec![0.0f32; seq_len * total_dim];
    let mut dk_rust = vec![0.0f32; seq_len * total_dim];
    let mut dv_rust = vec![0.0f32; seq_len * total_dim];
    swa_backward_rust(&q, &k, &v, &attn_weights, &d_attn_out,
                      &mut dq_rust, &mut dk_rust, &mut dv_rust,
                      seq_len, num_heads, head_dim, window_size);

    let mut dq_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dk_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dv_cuda = vec![0.0f32; seq_len * total_dim];
    cuda_backward_via_dispatch(&q, &k, &v, &attn_weights, &d_attn_out,
                               &mut dq_cuda, &mut dk_cuda, &mut dv_cuda,
                               seq_len, num_heads, head_dim, window_size);

    check_close("hd32_backward_dQ", &dq_rust, &dq_cuda, 5e-2);
    check_close("hd32_backward_dK", &dk_rust, &dk_cuda, 5e-2);
    check_close("hd32_backward_dV", &dv_rust, &dv_cuda, 5e-2);
}

// ── Test Class 2: Backward match ────────────────────────────────────

#[test]
fn test_cuda_backward_matches_rust_test_config() {
    let seq_len = 24;
    let num_heads = 4;
    let head_dim = 16;
    let window_size = 16;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 1000);
    let k = rand_buf(seq_len * total_dim, 1001);
    let v = rand_buf(seq_len * total_dim, 1002);

    // Run forward to get attn_weights (use Rust reference for consistency)
    let mut attn_out = vec![0.0f32; seq_len * total_dim];
    let mut attn_weights = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut attn_out, &mut attn_weights,
                seq_len, num_heads, head_dim, window_size);

    // Random upstream gradient
    let d_attn_out = rand_buf(seq_len * total_dim, 1003);

    // Rust backward
    let mut dq_rust = vec![0.0f32; seq_len * total_dim];
    let mut dk_rust = vec![0.0f32; seq_len * total_dim];
    let mut dv_rust = vec![0.0f32; seq_len * total_dim];
    swa_backward_rust(&q, &k, &v, &attn_weights, &d_attn_out,
                      &mut dq_rust, &mut dk_rust, &mut dv_rust,
                      seq_len, num_heads, head_dim, window_size);

    // CUDA backward via dispatch
    let mut dq_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dk_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dv_cuda = vec![0.0f32; seq_len * total_dim];
    cuda_backward_via_dispatch(&q, &k, &v, &attn_weights, &d_attn_out,
                               &mut dq_cuda, &mut dk_cuda, &mut dv_cuda,
                               seq_len, num_heads, head_dim, window_size);

    check_close("backward_dQ", &dq_rust, &dq_cuda, 5e-2);
    check_close("backward_dK", &dk_rust, &dk_cuda, 5e-2);
    check_close("backward_dV", &dv_rust, &dv_cuda, 5e-2);
}

#[test]
fn test_cuda_backward_matches_rust_small() {
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = 4;
    let window_size = 4;
    let total_dim = num_heads * head_dim;
    let aw_len = num_heads * seq_len * window_size;

    let q = rand_buf(seq_len * total_dim, 2000);
    let k = rand_buf(seq_len * total_dim, 2001);
    let v = rand_buf(seq_len * total_dim, 2002);

    let mut attn_out = vec![0.0f32; seq_len * total_dim];
    let mut attn_weights = vec![0.0f32; aw_len];
    swa_forward(&q, &k, &v, &mut attn_out, &mut attn_weights,
                seq_len, num_heads, head_dim, window_size);

    let d_attn_out = rand_buf(seq_len * total_dim, 2003);

    let mut dq_rust = vec![0.0f32; seq_len * total_dim];
    let mut dk_rust = vec![0.0f32; seq_len * total_dim];
    let mut dv_rust = vec![0.0f32; seq_len * total_dim];
    swa_backward_rust(&q, &k, &v, &attn_weights, &d_attn_out,
                      &mut dq_rust, &mut dk_rust, &mut dv_rust,
                      seq_len, num_heads, head_dim, window_size);

    let mut dq_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dk_cuda = vec![0.0f32; seq_len * total_dim];
    let mut dv_cuda = vec![0.0f32; seq_len * total_dim];
    cuda_backward_via_dispatch(&q, &k, &v, &attn_weights, &d_attn_out,
                               &mut dq_cuda, &mut dk_cuda, &mut dv_cuda,
                               seq_len, num_heads, head_dim, window_size);

    check_close("small_backward_dQ", &dq_rust, &dq_cuda, 5e-2);
    check_close("small_backward_dK", &dk_rust, &dk_cuda, 5e-2);
    check_close("small_backward_dV", &dv_rust, &dv_cuda, 5e-2);
}

// ── Test Class 3: Full pipeline FD check ────────────────────────────

use nl_hecate_core::model::{SWAConfig, SWAParams};
use nl_hecate_core::forward::forward;

/// FD gradient check for a single weight element through the full CUDA pipeline.
fn fd_single_cuda(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: &dyn Fn(&SWAParams) -> &Vec<f32>,
    set_weight: &dyn Fn(&mut SWAParams, usize, f32),
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

fn check_weight_gradient_cuda(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &SWAParams,
    name: &str,
    get_weight: &dyn Fn(&SWAParams) -> &Vec<f32>,
    set_weight: &dyn Fn(&mut SWAParams, usize, f32),
    get_grad: &dyn Fn(&SWAParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();
    // bf16 quantization corrupts FD for small gradients — raise threshold
    let abs_threshold = 5e-3;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = fd_single_cuda(
            params, cfg, input_ids, target_ids,
            get_weight, set_weight, idx, eps,
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

    eprintln!("  {name}: {passed}/{checked} pass, max_rel_err={max_rel_err:.4e}");
    assert!(passed == checked, "{name}: {passed}/{checked} passed, max_rel_err={max_rel_err:.4e}");
}

/// Compute gradients through the full CUDA pipeline.
fn compute_gradients_cuda(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, SWAParams) {
    use nl_hecate_core::backward::backward_full;
    let (loss, cache) = forward(params, cfg, input_ids, target_ids);
    let grads = backward_full(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

const FD_EPS: f32 = 1e-2;
// bf16 quantization adds ~1% error per load/store, compounding through the pipeline
const FD_TOL: f32 = 0.20;

#[test]
fn test_cuda_pipeline_gradient_w_q() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_q",
        &|p| &p.w_q, &|p, i, v| p.w_q[i] = v, &|g| &g.w_q,
        20, FD_EPS, FD_TOL,
    );
}

#[test]
fn test_cuda_pipeline_gradient_w_k() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_k",
        &|p| &p.w_k, &|p, i, v| p.w_k[i] = v, &|g| &g.w_k,
        20, FD_EPS, FD_TOL,
    );
}

#[test]
fn test_cuda_pipeline_gradient_w_v() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_v",
        &|p| &p.w_v, &|p, i, v| p.w_v[i] = v, &|g| &g.w_v,
        20, FD_EPS, FD_TOL,
    );
}

#[test]
fn test_cuda_pipeline_gradient_w_o() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_o",
        &|p| &p.w_o, &|p, i, v| p.w_o[i] = v, &|g| &g.w_o,
        20, FD_EPS, FD_TOL,
    );
}

#[test]
fn test_cuda_pipeline_gradient_w_unembed() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_unembed",
        &|p| &p.w_unembed, &|p, i, v| p.w_unembed[i] = v, &|g| &g.w_unembed,
        20, FD_EPS, FD_TOL,
    );
}

#[test]
fn test_cuda_pipeline_gradient_w_embed() {
    let cfg = grad_check_config();
    let params = SWAParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();
    let (_loss, grads) = compute_gradients_cuda(&params, &cfg, &input_ids, &target_ids);

    check_weight_gradient_cuda(
        &params, &cfg, &input_ids, &target_ids, &grads,
        "cuda_w_embed",
        &|p| &p.w_embed, &|p, i, v| p.w_embed[i] = v, &|g| &g.w_embed,
        20, FD_EPS, FD_TOL,
    );
}
