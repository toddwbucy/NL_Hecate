// Test 06: Chain Rule Composition — Enzyme + Hand-Written Backward
//
// PURPOSE: Test whether we can manually compose Enzyme-computed gradients
//          with hand-written backward kernels. This is the realistic path
//          for NL_Hecate's kernel-pair architecture:
//
//   Forward:  Rust_projection → kernel_memory_update → Rust_loss
//   Backward: d_loss(Enzyme) × d_kernel(hand) × d_proj(Enzyme)
//
// WHY IT MATTERS: Even if Enzyme can't see inside CUDA kernels (or FFI),
//          we can compose gradients manually using the chain rule IF we can:
//          (a) Get Enzyme to give us d_loss/d_kernel_output
//          (b) Get Enzyme to give us d_kernel_input/d_W (projection params)
//          (c) Multiply: (a) × hand_backward × (c)
//
// This test validates the manual composition approach.

use std::autodiff::autodiff;

// ============================================================
// The three-stage forward pass
// ============================================================

/// Stage 1 (Rust, Enzyme-differentiable): Project input through learned weight
fn project(w: f32, x: f32) -> f32 {
    w * x
}

/// Stage 2 (Simulated kernel): Memory update — the "opaque" middle
/// In production this would be a CUDA kernel.
/// The backward is hand-written (not Enzyme).
fn kernel_update(state: f32, projected: f32, lr: f32) -> f32 {
    state + lr * (projected - state)
}

/// Hand-written backward for kernel_update
/// Given d_out, compute d_state, d_projected, d_lr
fn kernel_update_backward(
    state: f32, projected: f32, lr: f32, d_out: f32,
) -> (f32, f32, f32) {
    let d_state = d_out * (1.0 - lr);
    let d_projected = d_out * lr;
    let d_lr = d_out * (projected - state);
    (d_state, d_projected, d_lr)
}

/// Stage 3 (Rust, Enzyme-differentiable): Compute loss
fn loss_fn(output: f32, target: f32) -> f32 {
    let diff = output - target;
    diff * diff
}

// ============================================================
// Full forward pass (for finite difference reference)
// ============================================================

fn full_forward(w: f32, x: f32, state: f32, lr: f32, target: f32) -> f32 {
    let projected = project(w, x);
    let updated = kernel_update(state, projected, lr);
    loss_fn(updated, target)
}

// ============================================================
// Enzyme-differentiable pieces
// ============================================================

/// Loss function that Enzyme differentiates — gives us d_loss/d_output
#[autodiff(d_loss_fn, Reverse, Active, Const, Active)]
fn loss_ad(output: f32, target: f32) -> f32 {
    loss_fn(output, target)
}

/// Projection that Enzyme differentiates — gives us d_projected/d_w
#[autodiff(d_project, Reverse, Active, Active, Active)]
fn project_ad(w: f32, x: f32) -> f32 {
    project(w, x)
}

// ============================================================
// Manual chain rule composition
// ============================================================

/// Compute d(loss)/d(w) through the full chain:
///   loss = loss_fn(kernel_update(project(w, x), state, lr), target)
///
/// Using chain rule:
///   d_loss/d_w = d_loss/d_updated × d_updated/d_projected × d_projected/d_w
///
/// Where:
///   d_loss/d_updated    — from Enzyme (d_loss_fn)
///   d_updated/d_projected — from hand-written backward (kernel_update_backward)
///   d_projected/d_w     — from Enzyme (d_project)
fn manual_chain_rule_dw(
    w: f32, x: f32, state: f32, lr: f32, target: f32,
) -> (f32, f32) {
    // Forward pass
    let projected = project(w, x);
    let updated = kernel_update(state, projected, lr);
    let loss = loss_fn(updated, target);

    // Backward pass — manual chain rule
    // Step 1: d_loss/d_updated (Enzyme on loss function)
    let (_loss_val, d_updated) = d_loss_fn(updated, target, 1.0);

    // Step 2: d_updated/d_projected (hand-written kernel backward)
    let (_d_state, d_projected, _d_lr) = kernel_update_backward(state, projected, lr, d_updated);

    // Step 3: d_projected/d_w (Enzyme on projection)
    let (_proj_val, d_w, _d_x) = d_project(w, x, 1.0);
    // But d_project gives d(w*x)/d(w) with seed=1.0
    // We need to scale by d_projected (chain rule)
    let d_loss_dw = d_projected * d_w;

    (loss, d_loss_dw)
}

/// Also compute d(loss)/d(state) and d(loss)/d(lr) for completeness
fn manual_chain_rule_full(
    w: f32, x: f32, state: f32, lr: f32, target: f32,
) -> (f32, f32, f32, f32) {
    // Forward pass
    let projected = project(w, x);
    let updated = kernel_update(state, projected, lr);
    let loss = loss_fn(updated, target);

    // d_loss/d_updated
    let (_loss_val, d_updated) = d_loss_fn(updated, target, 1.0);

    // Hand-written kernel backward
    let (d_state, d_projected, d_lr) = kernel_update_backward(state, projected, lr, d_updated);

    // d_projected/d_w via Enzyme
    let (_proj_val, d_w_raw, _d_x) = d_project(w, x, 1.0);
    let d_loss_dw = d_projected * d_w_raw;

    // d_state and d_lr already flow directly from kernel backward
    // (they don't go through the projection stage)
    (loss, d_loss_dw, d_state, d_lr)
}

// ============================================================
// Verification
// ============================================================

fn check_gradient(name: &str, computed: f32, reference: f32, tol: f32) -> bool {
    let abs_diff = (computed - reference).abs();
    let rel_diff = if reference.abs() > 1e-10 {
        abs_diff / reference.abs()
    } else {
        abs_diff
    };

    if rel_diff < tol {
        println!("  [PASS] {}: computed={:.6}, ref={:.6}, rel_err={:.2e}",
                 name, computed, reference, rel_diff);
        true
    } else {
        println!("  [FAIL] {}: computed={:.6}, ref={:.6}, rel_err={:.2e} (tol={:.2e})",
                 name, computed, reference, rel_diff, tol);
        false
    }
}

pub fn run() -> (usize, usize) {
    println!("\n=== Test 06: Chain Rule Composition ===\n");

    let eps = 1e-4_f32;
    let tol = 1e-3_f32;
    let mut pass = 0;
    let mut fail = 0;

    let w = 1.5_f32;
    let x = 2.0_f32;
    let state = 0.5_f32;
    let lr = 0.1_f32;
    let target = 3.0_f32;

    // --- Finite differences (ground truth) ---
    let fd_w = (full_forward(w + eps, x, state, lr, target) - full_forward(w - eps, x, state, lr, target)) / (2.0 * eps);
    let fd_state = (full_forward(w, x, state + eps, lr, target) - full_forward(w, x, state - eps, lr, target)) / (2.0 * eps);
    let fd_lr = (full_forward(w, x, state, lr + eps, target) - full_forward(w, x, state, lr - eps, target)) / (2.0 * eps);

    println!("  Finite difference reference:");
    println!("    d_loss/d_w     = {:.6}", fd_w);
    println!("    d_loss/d_state = {:.6}", fd_state);
    println!("    d_loss/d_lr    = {:.6}", fd_lr);
    println!();

    // --- Manual chain rule ---
    let (loss, d_w_manual) = manual_chain_rule_dw(w, x, state, lr, target);
    println!("  Forward loss = {:.6}", loss);
    println!();

    if check_gradient("chain_rule: d_loss/d_w", d_w_manual, fd_w, tol) { pass += 1; } else { fail += 1; }

    let (_loss, d_w_full, d_state_full, d_lr_full) = manual_chain_rule_full(w, x, state, lr, target);
    if check_gradient("chain_rule: d_loss/d_state", d_state_full, fd_state, tol) { pass += 1; } else { fail += 1; }
    if check_gradient("chain_rule: d_loss/d_lr", d_lr_full, fd_lr, tol) { pass += 1; } else { fail += 1; }

    // --- Test at a different point to avoid symmetry accidents ---
    println!();
    let w2 = 0.7_f32;
    let x2 = -1.5_f32;
    let state2 = 2.0_f32;
    let lr2 = 0.3_f32;
    let target2 = 1.0_f32;

    let fd_w2 = (full_forward(w2 + eps, x2, state2, lr2, target2) - full_forward(w2 - eps, x2, state2, lr2, target2)) / (2.0 * eps);
    let fd_state2 = (full_forward(w2, x2, state2 + eps, lr2, target2) - full_forward(w2, x2, state2 - eps, lr2, target2)) / (2.0 * eps);
    let fd_lr2 = (full_forward(w2, x2, state2, lr2 + eps, target2) - full_forward(w2, x2, state2, lr2 - eps, target2)) / (2.0 * eps);

    let (_loss2, d_w2, d_state2_comp, d_lr2_comp) = manual_chain_rule_full(w2, x2, state2, lr2, target2);
    if check_gradient("chain_rule_2: d_loss/d_w", d_w2, fd_w2, tol) { pass += 1; } else { fail += 1; }
    if check_gradient("chain_rule_2: d_loss/d_state", d_state2_comp, fd_state2, tol) { pass += 1; } else { fail += 1; }
    if check_gradient("chain_rule_2: d_loss/d_lr", d_lr2_comp, fd_lr2, tol) { pass += 1; } else { fail += 1; }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
