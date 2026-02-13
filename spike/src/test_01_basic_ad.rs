// Test 01: Basic Scalar Reverse-Mode AD
//
// PURPOSE: Verify that Enzyme's #[autodiff] attribute works at all.
//          This is the absolute minimum: scalar in, scalar out, reverse mode.
//
// WHAT THIS PROVES: The toolchain is correctly built and Enzyme can
//          differentiate a simple Rust function.
//
// EXPECTED: d_square(3.0) = 6.0  (derivative of x^2 is 2x)
//           d_cubic(2.0) = 12.0  (derivative of x^3 is 3x^2)

use std::autodiff::autodiff;

// --- Test functions ---

#[autodiff(d_square, Reverse, Active, Active)]
fn square(x: f32) -> f32 {
    x * x
}

#[autodiff(d_cubic, Reverse, Active, Active)]
fn cubic(x: f32) -> f32 {
    x * x * x
}

// Multi-operation chain: tests that Enzyme handles intermediate values
#[autodiff(d_chain, Reverse, Active, Active)]
fn chain(x: f32) -> f32 {
    let a = x * x;       // x^2
    let b = a + x;       // x^2 + x
    let c = b * 2.0;     // 2x^2 + 2x
    c                     // d/dx = 4x + 2
}

// --- Verification ---

/// Finite difference gradient: (f(x+eps) - f(x-eps)) / (2*eps)
fn finite_diff(f: fn(f32) -> f32, x: f32, eps: f32) -> f32 {
    (f(x + eps) - f(x - eps)) / (2.0 * eps)
}

fn check_gradient(name: &str, enzyme_grad: f32, fd_grad: f32, tol: f32) -> bool {
    let abs_diff = (enzyme_grad - fd_grad).abs();
    let rel_diff = if fd_grad.abs() > 1e-10 {
        abs_diff / fd_grad.abs()
    } else {
        abs_diff
    };

    if rel_diff < tol {
        println!("  [PASS] {}: enzyme={:.6}, fd={:.6}, rel_err={:.2e}",
                 name, enzyme_grad, fd_grad, rel_diff);
        true
    } else {
        println!("  [FAIL] {}: enzyme={:.6}, fd={:.6}, rel_err={:.2e} (tol={:.2e})",
                 name, enzyme_grad, fd_grad, rel_diff, tol);
        false
    }
}

pub fn run() -> (usize, usize) {
    println!("\n=== Test 01: Basic Scalar AD ===\n");

    let eps = 1e-4_f32;
    let tol = 1e-3_f32;
    let mut pass = 0;
    let mut fail = 0;

    // Test square: d/dx(x^2) = 2x
    {
        let x = 3.0_f32;
        let (val, grad) = d_square(x, 1.0);
        assert!((val - 9.0).abs() < 1e-6, "square(3.0) should be 9.0, got {}", val);
        let fd = finite_diff(square, x, eps);
        if check_gradient("square'(3.0)", grad, fd, tol) { pass += 1; } else { fail += 1; }
    }

    // Test square at different point
    {
        let x = -2.5_f32;
        let (val, grad) = d_square(x, 1.0);
        assert!((val - 6.25).abs() < 1e-6);
        let fd = finite_diff(square, x, eps);
        if check_gradient("square'(-2.5)", grad, fd, tol) { pass += 1; } else { fail += 1; }
    }

    // Test cubic: d/dx(x^3) = 3x^2
    {
        let x = 2.0_f32;
        let (val, grad) = d_cubic(x, 1.0);
        assert!((val - 8.0).abs() < 1e-6);
        let fd = finite_diff(cubic, x, eps);
        if check_gradient("cubic'(2.0)", grad, fd, tol) { pass += 1; } else { fail += 1; }
    }

    // Test chain: d/dx(2x^2 + 2x) = 4x + 2
    {
        let x = 1.5_f32;
        let (val, grad) = d_chain(x, 1.0);
        let expected_val = 2.0 * 1.5 * 1.5 + 2.0 * 1.5;  // 6.5 + 3.0 = 7.5
        assert!((val - expected_val).abs() < 1e-5, "chain(1.5) = {}, expected {}", val, expected_val);
        let fd = finite_diff(chain, x, eps);
        if check_gradient("chain'(1.5)", grad, fd, tol) { pass += 1; } else { fail += 1; }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
