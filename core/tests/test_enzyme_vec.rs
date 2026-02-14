// Sub-spike: Enzyme + Vec<f32> Duplicated Slices
//
// PURPOSE: Verify Enzyme can differentiate through functions that operate
//          on &[f32] slices derived from Vec<f32>. The Phase 0 spike only
//          tested fixed-size [f32; 4] arrays. Track Zero-A needs variable-
//          size weight matrices (e.g. 64×64 = 4096 elements).
//
// TESTS:
//   1. Simple dot product with &[f32] and Active return
//   2. Struct with Vec<f32> field, Duplicated
//   3. Matmul-like operation on slices
//   4. Chain: struct with Vec → operation → scalar loss → gradient
//
// If ANY of these fail, we need fallback (raw pointers or fixed arrays).

#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

// ── Test 1: Dot product on slices ────────────────────────────────────

/// Dot product of two f32 slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Enzyme-annotated: both slices are Duplicated, return Active.
#[autodiff_reverse(d_dot_both, Duplicated, Duplicated, Active)]
fn dot_ad(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

/// Enzyme-annotated: a is Duplicated, b is Const, return Active.
#[autodiff_reverse(d_dot_wrt_a, Duplicated, Const, Active)]
fn dot_wrt_a_ad(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

// ── Test 2: Struct with Vec<f32> field ───────────────────────────────

#[derive(Clone)]
struct Weights {
    w: Vec<f32>,
}

/// Weighted sum: sum(w[i] * x[i]).
fn weighted_sum(weights: &Weights, x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..weights.w.len() {
        sum += weights.w[i] * x[i];
    }
    sum
}

#[autodiff_reverse(d_weighted_sum, Duplicated, Const, Active)]
fn weighted_sum_ad(weights: &Weights, x: &[f32]) -> f32 {
    weighted_sum(weights, x)
}

// ── Test 3: Matrix-vector multiply on slices ─────────────────────────

/// Matrix-vector: out = W[m,n] @ x[n], returns sum(out) as scalar.
/// W is row-major flat slice, x is vector.
fn matvec_sum(w: &[f32], x: &[f32], m: usize, n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..m {
        let mut row_sum = 0.0f32;
        for j in 0..n {
            row_sum += w[i * n + j] * x[j];
        }
        total += row_sum;
    }
    total
}

#[autodiff_reverse(d_matvec_sum, Duplicated, Const, Const, Const, Active)]
fn matvec_sum_ad(w: &[f32], x: &[f32], m: usize, n: usize) -> f32 {
    matvec_sum(w, x, m, n)
}

// ── Test 4: Full chain — struct with Vec → matmul → loss ─────────────

#[derive(Clone)]
struct LinearLayer {
    weight: Vec<f32>,  // [out_dim, in_dim] row-major
    out_dim: usize,
    in_dim: usize,
}

/// Forward: y = W @ x, loss = sum((y - target)^2)
fn linear_loss(layer: &LinearLayer, x: &[f32], target: &[f32]) -> f32 {
    let mut loss = 0.0f32;
    for i in 0..layer.out_dim {
        let mut yi = 0.0f32;
        for j in 0..layer.in_dim {
            yi += layer.weight[i * layer.in_dim + j] * x[j];
        }
        let diff = yi - target[i];
        loss += diff * diff;
    }
    loss
}

#[autodiff_reverse(d_linear_loss, Duplicated, Const, Const, Active)]
fn linear_loss_ad(layer: &LinearLayer, x: &[f32], target: &[f32]) -> f32 {
    linear_loss(layer, x, target)
}

// ── Finite difference helper ─────────────────────────────────────────

fn finite_diff_slice(
    f: impl Fn(&[f32]) -> f32,
    x: &[f32],
    idx: usize,
    eps: f32,
) -> f32 {
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    x_plus[idx] += eps;
    x_minus[idx] -= eps;
    (f(&x_plus) - f(&x_minus)) / (2.0 * eps)
}

fn check_gradient(name: &str, enzyme_grad: f32, fd_grad: f32, tol: f32) -> bool {
    let abs_diff = (enzyme_grad - fd_grad).abs();
    let rel_diff = if fd_grad.abs() > 1e-10 {
        abs_diff / fd_grad.abs()
    } else {
        abs_diff
    };
    let pass = rel_diff < tol || abs_diff < 1e-5;
    if !pass {
        eprintln!(
            "  FAIL {}: enzyme={:.6}, fd={:.6}, rel_err={:.6}",
            name, enzyme_grad, fd_grad, rel_diff
        );
    }
    pass
}

// ── Tests ────────────────────────────────────────────────────────────

#[test]
fn test_dot_product_slices() {
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![4.0f32, 5.0, 6.0];
    let mut da = vec![0.0f32; 3];
    let mut db = vec![0.0f32; 3];

    let _val = d_dot_both(&a, &mut da, &b, &mut db, 1.0);

    // d(dot)/d(a[i]) = b[i], d(dot)/d(b[i]) = a[i]
    let eps = 1e-4;
    let tol = 0.01;
    let mut all_pass = true;

    for i in 0..3 {
        let fd_da = finite_diff_slice(|a_| dot(a_, &b), &a, i, eps);
        all_pass &= check_gradient(&format!("da[{}]", i), da[i], fd_da, tol);
    }
    for i in 0..3 {
        let fd_db = finite_diff_slice(|b_| dot(&a, b_), &b, i, eps);
        all_pass &= check_gradient(&format!("db[{}]", i), db[i], fd_db, tol);
    }
    assert!(all_pass, "Dot product slice gradients failed");
}

#[test]
fn test_dot_wrt_a_only() {
    let a = vec![0.5f32, -1.0, 2.0, 0.3];
    let b = vec![1.0f32, 0.5, -0.5, 2.0];
    let mut da = vec![0.0f32; 4];

    let _val = d_dot_wrt_a(&a, &mut da, &b, 1.0);

    let eps = 1e-4;
    let tol = 0.01;
    let mut all_pass = true;
    for i in 0..4 {
        let fd = finite_diff_slice(|a_| dot(a_, &b), &a, i, eps);
        all_pass &= check_gradient(&format!("da[{}]", i), da[i], fd, tol);
    }
    assert!(all_pass, "Dot wrt_a gradients failed");
}

#[test]
fn test_struct_with_vec_field() {
    let weights = Weights { w: vec![0.3f32, -0.7, 1.2, 0.1] };
    let x = vec![1.0f32, 2.0, 0.5, -1.0];
    let mut d_weights = Weights { w: vec![0.0f32; 4] };

    let _val = d_weighted_sum(&weights, &mut d_weights, &x, 1.0);

    // d(weighted_sum)/d(w[i]) = x[i]
    let eps = 1e-4;
    let tol = 0.01;
    let mut all_pass = true;
    for i in 0..4 {
        let fd = {
            let mut w_plus = weights.clone();
            let mut w_minus = weights.clone();
            w_plus.w[i] += eps;
            w_minus.w[i] -= eps;
            (weighted_sum(&w_plus, &x) - weighted_sum(&w_minus, &x)) / (2.0 * eps)
        };
        all_pass &= check_gradient(&format!("dw[{}]", i), d_weights.w[i], fd, tol);
    }
    assert!(all_pass, "Struct with Vec field gradients failed");
}

#[test]
fn test_matvec_sum_gradients() {
    let m = 3usize;
    let n = 4usize;
    let w: Vec<f32> = vec![
        0.1, -0.2, 0.3, 0.4,
        0.5, 0.6, -0.7, 0.8,
        -0.1, 0.2, 0.3, -0.4,
    ];
    let x: Vec<f32> = vec![1.0, 0.5, -1.0, 2.0];
    let mut dw = vec![0.0f32; m * n];

    let _val = d_matvec_sum(&w, &mut dw, &x, m, n, 1.0);

    let eps = 1e-4;
    let tol = 0.01;
    let mut all_pass = true;
    for i in 0..(m * n) {
        let fd = finite_diff_slice(|w_| matvec_sum(w_, &x, m, n), &w, i, eps);
        all_pass &= check_gradient(&format!("dw[{}]", i), dw[i], fd, tol);
    }
    assert!(all_pass, "Matvec sum gradients failed");
}

#[test]
fn test_linear_loss_struct_vec() {
    let layer = LinearLayer {
        weight: vec![
            0.1, -0.3, 0.5,
            0.2, 0.4, -0.1,
        ],
        out_dim: 2,
        in_dim: 3,
    };
    let x = vec![1.0f32, -0.5, 2.0];
    let target = vec![0.5f32, -0.3];
    let mut d_layer = LinearLayer {
        weight: vec![0.0f32; 6],
        out_dim: 2,
        in_dim: 3,
    };

    let _loss = d_linear_loss(&layer, &mut d_layer, &x, &target, 1.0);

    let eps = 1e-4;
    let tol = 0.01;
    let mut all_pass = true;
    for i in 0..6 {
        let fd = {
            let mut l_plus = layer.clone();
            let mut l_minus = layer.clone();
            l_plus.weight[i] += eps;
            l_minus.weight[i] -= eps;
            (linear_loss(&l_plus, &x, &target) - linear_loss(&l_minus, &x, &target)) / (2.0 * eps)
        };
        all_pass &= check_gradient(&format!("dweight[{}]", i), d_layer.weight[i], fd, tol);
    }
    assert!(all_pass, "Linear loss struct Vec gradients failed");
}
