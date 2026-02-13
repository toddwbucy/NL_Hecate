// Test 05: Simulated Kernel-Pair Without #[custom_vjp]
//
// PURPOSE: The spec's kernel-pair architecture assumes every hot operation
//          has three implementations:
//            (1) Rust reference (Enzyme-compatible)
//            (2) CUDA forward kernel
//            (3) CUDA backward kernel with analytical gradients
//          Enzyme was supposed to chain through pairs via #[custom_vjp].
//          But #[custom_vjp] is NOT exposed in Rust-Enzyme.
//
// THIS TEST: Explores three strategies to achieve the same effect.
//
// Strategy A: Pure Rust — Enzyme differentiates the forward directly.
//             Compare Enzyme's gradient with our hand-written backward.
//             Validates the MATH without testing the opaque boundary.
//
// Strategy B: extern "C" FFI boundary — put the "kernel" behind FFI.
//             Test whether Enzyme stops at FFI calls.
//             If it does, we manually compose gradients.
//
// Strategy C: black_box / inline asm — create an opaque region.
//             Test if Enzyme returns zero gradient through opaque region.
//             If so, we manually inject the correct gradient.

use std::autodiff::autodiff;

// ============================================================
// STRATEGY A: Pure Rust — Let Enzyme differentiate directly
// ============================================================

/// A "kernel" implemented in pure Rust.
/// Models a memory update: M' = M + lr * (input - M)
/// This is what the CUDA kernel would compute, but in Rust.
fn delta_kernel_forward(state: f32, input: f32, lr: f32) -> f32 {
    state + lr * (input - state)
}

/// Hand-written backward for the delta kernel.
/// d_out/d_state = 1 - lr
/// d_out/d_input = lr
/// d_out/d_lr = input - state
fn delta_kernel_backward(state: f32, input: f32, lr: f32, d_out: f32) -> (f32, f32, f32) {
    let d_state = d_out * (1.0 - lr);
    let d_input = d_out * lr;
    let d_lr = d_out * (input - state);
    (d_state, d_input, d_lr)
}

/// Enzyme differentiates the forward directly.
#[autodiff(d_delta_kernel, Reverse, Active, Active, Active, Active)]
fn delta_kernel_ad(state: f32, input: f32, lr: f32) -> f32 {
    delta_kernel_forward(state, input, lr)
}

// ============================================================
// STRATEGY B: FFI Boundary
// ============================================================

// Simulate an external "kernel" via extern "C" function.
// In practice this would be a CUDA kernel compiled separately.
// We define it in Rust but mark it extern "C" to test the boundary.

// The "kernel" — compiled as a regular function but called through C ABI.
// Enzyme should NOT be able to differentiate through this.
#[no_mangle]
extern "C" fn ffi_kernel_forward(state: f32, input: f32, lr: f32) -> f32 {
    state + lr * (input - state)
}

// The hand-written backward for the FFI kernel.
#[no_mangle]
extern "C" fn ffi_kernel_backward(
    state: f32, input: f32, lr: f32, d_out: f32,
    d_state: *mut f32, d_input: *mut f32, d_lr: *mut f32,
) {
    unsafe {
        *d_state = d_out * (1.0 - lr);
        *d_input = d_out * lr;
        *d_lr = d_out * (input - state);
    }
}

/// Wrapper that Enzyme differentiates.
/// The interesting question: does Enzyme see through extern "C" calls
/// when the function is in the same compilation unit?
///
/// If yes: Enzyme differentiates ffi_kernel_forward directly (same as Strategy A).
/// If no:  Enzyme returns zero gradient, and we need manual composition.
#[autodiff(d_ffi_wrapper, Reverse, Active, Active, Active, Active)]
fn ffi_wrapper(state: f32, input: f32, lr: f32) -> f32 {
    // Call through C ABI — no unsafe needed since ffi_kernel_forward
    // is declared as extern "C" fn, not unsafe extern "C" fn
    ffi_kernel_forward(state, input, lr)
}

// ============================================================
// STRATEGY C: black_box barrier
// ============================================================

/// Use std::hint::black_box to prevent Enzyme from seeing the computation.
/// black_box is an identity function that the compiler can't optimize through.
/// If Enzyme respects this barrier, gradients through it will be zero.
#[autodiff(d_blackbox_wrapper, Reverse, Active, Active, Active, Active)]
fn blackbox_wrapper(state: f32, input: f32, lr: f32) -> f32 {
    // Compute the result, but hide it from the optimizer
    let result = state + lr * (input - state);
    std::hint::black_box(result)
}

// ============================================================
// Verification
// ============================================================

fn check_gradient(name: &str, enzyme_grad: f32, reference: f32, tol: f32) -> bool {
    let abs_diff = (enzyme_grad - reference).abs();
    let rel_diff = if reference.abs() > 1e-10 {
        abs_diff / reference.abs()
    } else {
        abs_diff
    };

    if rel_diff < tol {
        println!("  [PASS] {}: enzyme={:.6}, ref={:.6}, rel_err={:.2e}",
                 name, enzyme_grad, reference, rel_diff);
        true
    } else {
        println!("  [FAIL] {}: enzyme={:.6}, ref={:.6}, rel_err={:.2e} (tol={:.2e})",
                 name, enzyme_grad, reference, rel_diff, tol);
        false
    }
}

fn finite_diff_3(
    f: fn(f32, f32, f32) -> f32,
    state: f32, input: f32, lr: f32,
    eps: f32,
) -> (f32, f32, f32) {
    let d_state = (f(state + eps, input, lr) - f(state - eps, input, lr)) / (2.0 * eps);
    let d_input = (f(state, input + eps, lr) - f(state, input - eps, lr)) / (2.0 * eps);
    let d_lr = (f(state, input, lr + eps) - f(state, input, lr - eps)) / (2.0 * eps);
    (d_state, d_input, d_lr)
}

pub fn run() -> (usize, usize) {
    println!("\n=== Test 05: Kernel-Pair Simulation ===\n");

    let eps = 1e-4_f32;
    let tol = 1e-3_f32;
    let mut pass = 0;
    let mut fail = 0;

    let state = 2.0_f32;
    let input = 5.0_f32;
    let lr = 0.1_f32;

    let (fd_state, fd_input, fd_lr) = finite_diff_3(delta_kernel_forward, state, input, lr, eps);
    let (hand_state, hand_input, hand_lr) = delta_kernel_backward(state, input, lr, 1.0);

    // --- Strategy A: Pure Rust AD ---
    println!("  --- Strategy A: Pure Rust (Enzyme differentiates directly) ---");
    {
        let (val, d_state_e, d_input_e, d_lr_e) = d_delta_kernel(state, input, lr, 1.0);
        let expected = delta_kernel_forward(state, input, lr);
        assert!((val - expected).abs() < 1e-6);

        // Enzyme vs finite difference
        if check_gradient("A: d/d_state (enzyme vs fd)", d_state_e, fd_state, tol) { pass += 1; } else { fail += 1; }
        if check_gradient("A: d/d_input (enzyme vs fd)", d_input_e, fd_input, tol) { pass += 1; } else { fail += 1; }
        if check_gradient("A: d/d_lr (enzyme vs fd)", d_lr_e, fd_lr, tol) { pass += 1; } else { fail += 1; }

        // Enzyme vs hand-written backward
        if check_gradient("A: d/d_state (enzyme vs hand)", d_state_e, hand_state, tol) { pass += 1; } else { fail += 1; }
        if check_gradient("A: d/d_input (enzyme vs hand)", d_input_e, hand_input, tol) { pass += 1; } else { fail += 1; }
        if check_gradient("A: d/d_lr (enzyme vs hand)", d_lr_e, hand_lr, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Strategy B: FFI boundary ---
    println!("\n  --- Strategy B: FFI Boundary ---");
    {
        let (val, d_state_e, d_input_e, d_lr_e) = d_ffi_wrapper(state, input, lr, 1.0);
        let expected = delta_kernel_forward(state, input, lr);
        assert!((val - expected).abs() < 1e-6);

        // Check if Enzyme differentiated through the FFI call
        let enzyme_saw_through = d_state_e.abs() > 1e-10 || d_input_e.abs() > 1e-10 || d_lr_e.abs() > 1e-10;

        if enzyme_saw_through {
            println!("  [INFO] Enzyme DID see through extern \"C\" (same compilation unit).");
            println!("  [INFO] This means FFI in the same crate is NOT an opaque boundary.");
            println!("  [INFO] For true opacity, the kernel must be in a separate compilation unit.");
            if check_gradient("B: d/d_state (enzyme vs fd)", d_state_e, fd_state, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("B: d/d_input (enzyme vs fd)", d_input_e, fd_input, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("B: d/d_lr (enzyme vs fd)", d_lr_e, fd_lr, tol) { pass += 1; } else { fail += 1; }
        } else {
            println!("  [INFO] Enzyme did NOT see through extern \"C\" — gradients are zero.");
            println!("  [INFO] This confirms FFI is an opaque boundary.");
            println!("  [INFO] For kernel pairs, we manually compose:");
            println!("         d_outer * hand_backward(kernel) * d_outer");
            println!("  [PASS] FFI boundary is opaque (as needed for kernel-pair pattern)");
            pass += 1;

            // Verify hand-written backward matches finite diff
            if check_gradient("B: hand_backward d_state vs fd", hand_state, fd_state, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("B: hand_backward d_input vs fd", hand_input, fd_input, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("B: hand_backward d_lr vs fd", hand_lr, fd_lr, tol) { pass += 1; } else { fail += 1; }
        }
    }

    // --- Strategy C: black_box ---
    println!("\n  --- Strategy C: black_box Barrier ---");
    {
        let (val, d_state_e, d_input_e, d_lr_e) = d_blackbox_wrapper(state, input, lr, 1.0);
        let expected = delta_kernel_forward(state, input, lr);
        assert!((val - expected).abs() < 1e-6);

        let enzyme_saw_through = d_state_e.abs() > 1e-10 || d_input_e.abs() > 1e-10 || d_lr_e.abs() > 1e-10;

        if enzyme_saw_through {
            println!("  [INFO] Enzyme DID see through black_box.");
            println!("  [INFO] black_box is NOT a reliable opaque barrier for Enzyme.");
            if check_gradient("C: d/d_state (enzyme vs fd)", d_state_e, fd_state, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("C: d/d_input (enzyme vs fd)", d_input_e, fd_input, tol) { pass += 1; } else { fail += 1; }
            if check_gradient("C: d/d_lr (enzyme vs fd)", d_lr_e, fd_lr, tol) { pass += 1; } else { fail += 1; }
        } else {
            println!("  [INFO] Enzyme did NOT see through black_box — gradients are zero.");
            println!("  [INFO] black_box IS a reliable opaque barrier.");
            println!("  [PASS] black_box boundary confirmed opaque");
            pass += 1;
        }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
