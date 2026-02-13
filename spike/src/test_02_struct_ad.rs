// Test 02: Struct Field Differentiation
//
// PURPOSE: Verify that Enzyme can compute gradients with respect to
//          fields of a Rust struct passed by reference.
//
// WHY IT MATTERS: NL_Hecate's outer-loop parameters (W_K, W_V, W_Q, gates)
//          live in structs. Enzyme must produce d(loss)/d(W_K) by flowing
//          gradients into struct fields. If this doesn't work, the entire
//          architecture needs restructuring.
//
// EXPECTED: Gradients flow to individual struct fields correctly.
//           Shadow struct accumulates per-field gradients.

use std::autodiff::autodiff;

// --- Structs modeling NL parameters ---

/// Minimal projection parameters (models W_K, W_V from the spec)
#[derive(Clone)]
struct ProjectionParams {
    w_k: f32,
    w_v: f32,
}

/// Two-parameter linear model: out = w_k * x + w_v
/// This models the simplest case of "project input through learned weights"
#[autodiff(d_project, Reverse, Duplicated, Active, Active)]
fn project(params: &ProjectionParams, x: f32) -> f32 {
    params.w_k * x + params.w_v
}

/// Nested struct: models a small model with multiple parameter groups
#[derive(Clone)]
struct MiniModel {
    proj: ProjectionParams,
    bias: f32,
}

/// Forward pass through mini model: proj(x) + bias
#[autodiff(d_mini_forward, Reverse, Duplicated, Active, Active)]
fn mini_forward(model: &MiniModel, x: f32) -> f32 {
    let projected = model.proj.w_k * x + model.proj.w_v;
    projected + model.bias
}

/// Loss function wrapping projection: (project(x) - target)^2
/// This tests gradient flow through a chain: loss -> project -> params
#[autodiff(d_project_loss, Reverse, Duplicated, Active, Const, Active)]
fn project_loss(params: &ProjectionParams, x: f32, target: f32) -> f32 {
    let out = params.w_k * x + params.w_v;
    let diff = out - target;
    diff * diff
}

// --- Verification ---

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
    println!("\n=== Test 02: Struct Field Differentiation ===\n");

    let eps = 1e-4_f32;
    let tol = 1e-3_f32;
    let mut pass = 0;
    let mut fail = 0;

    // --- Test 1: Simple projection, gradient w.r.t. w_k ---
    // f(w_k, w_v, x) = w_k * x + w_v
    // df/dw_k = x, df/dw_v = 1
    {
        let params = ProjectionParams { w_k: 2.0, w_v: 0.5 };
        let x = 3.0_f32;

        // Shadow struct starts zeroed — Enzyme accumulates into it
        let mut d_params = ProjectionParams { w_k: 0.0, w_v: 0.0 };

        let (val, _d_x) = d_project(&params, &mut d_params, x, 1.0);
        assert!((val - 6.5).abs() < 1e-6, "project should be 2*3+0.5=6.5, got {}", val);

        // Finite diff for w_k: vary w_k by eps
        let fd_wk = {
            let p_plus = ProjectionParams { w_k: params.w_k + eps, ..params.clone() };
            let p_minus = ProjectionParams { w_k: params.w_k - eps, ..params.clone() };
            (project(&p_plus, x) - project(&p_minus, x)) / (2.0 * eps)
        };
        if check_gradient("d_project/d_w_k", d_params.w_k, fd_wk, tol) { pass += 1; } else { fail += 1; }

        // Finite diff for w_v
        let fd_wv = {
            let p_plus = ProjectionParams { w_v: params.w_v + eps, ..params.clone() };
            let p_minus = ProjectionParams { w_v: params.w_v - eps, ..params.clone() };
            (project(&p_plus, x) - project(&p_minus, x)) / (2.0 * eps)
        };
        if check_gradient("d_project/d_w_v", d_params.w_v, fd_wv, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Test 2: Nested struct ---
    {
        let model = MiniModel {
            proj: ProjectionParams { w_k: 1.5, w_v: -0.3 },
            bias: 0.7,
        };
        let x = 2.0_f32;

        let mut d_model = MiniModel {
            proj: ProjectionParams { w_k: 0.0, w_v: 0.0 },
            bias: 0.0,
        };

        let (val, _d_x) = d_mini_forward(&model, &mut d_model, x, 1.0);
        let expected = 1.5 * 2.0 + (-0.3) + 0.7;  // 3.0 - 0.3 + 0.7 = 3.4
        assert!((val - expected).abs() < 1e-5, "mini_forward={}, expected={}", val, expected);

        // Finite diff for nested w_k
        let fd_wk = {
            let mut m_plus = model.clone();
            m_plus.proj.w_k += eps;
            let mut m_minus = model.clone();
            m_minus.proj.w_k -= eps;
            (mini_forward(&m_plus, x) - mini_forward(&m_minus, x)) / (2.0 * eps)
        };
        if check_gradient("d_mini/d_proj.w_k", d_model.proj.w_k, fd_wk, tol) { pass += 1; } else { fail += 1; }

        // Finite diff for bias
        let fd_bias = {
            let mut m_plus = model.clone();
            m_plus.bias += eps;
            let mut m_minus = model.clone();
            m_minus.bias -= eps;
            (mini_forward(&m_plus, x) - mini_forward(&m_minus, x)) / (2.0 * eps)
        };
        if check_gradient("d_mini/d_bias", d_model.bias, fd_bias, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Test 3: Loss function (gradient through a chain) ---
    {
        let params = ProjectionParams { w_k: 1.0, w_v: 0.0 };
        let x = 2.0_f32;
        let target = 3.0_f32;

        let mut d_params = ProjectionParams { w_k: 0.0, w_v: 0.0 };

        // loss = (w_k * x + w_v - target)^2 = (2 - 3)^2 = 1
        // d_loss/d_w_k = 2 * (w_k*x + w_v - target) * x = 2 * (-1) * 2 = -4
        let (val, _d_x) = d_project_loss(&params, &mut d_params, x, target, 1.0);
        assert!((val - 1.0).abs() < 1e-5, "loss should be 1.0, got {}", val);

        let fd_wk = {
            let p_plus = ProjectionParams { w_k: params.w_k + eps, ..params.clone() };
            let p_minus = ProjectionParams { w_k: params.w_k - eps, ..params.clone() };
            (project_loss(&p_plus, x, target) - project_loss(&p_minus, x, target)) / (2.0 * eps)
        };
        if check_gradient("d_loss/d_w_k (chain)", d_params.w_k, fd_wk, tol) { pass += 1; } else { fail += 1; }

        let fd_wv = {
            let p_plus = ProjectionParams { w_v: params.w_v + eps, ..params.clone() };
            let p_minus = ProjectionParams { w_v: params.w_v - eps, ..params.clone() };
            (project_loss(&p_plus, x, target) - project_loss(&p_minus, x, target)) / (2.0 * eps)
        };
        if check_gradient("d_loss/d_w_v (chain)", d_params.w_v, fd_wv, tol) { pass += 1; } else { fail += 1; }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
