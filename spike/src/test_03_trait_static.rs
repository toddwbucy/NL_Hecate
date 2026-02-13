// Test 03: Static Dispatch Through Trait Bounds
//
// PURPOSE: Verify that Enzyme can differentiate through generic functions
//          with trait bounds, where the compiler monomorphizes at compile time.
//
// WHY IT MATTERS: NL_Hecate's MemoryRule trait system relies on static dispatch.
//          Each memory update rule (Delta, Titans LMM, MONETA, etc.) implements
//          the same trait. The spec requires Enzyme to differentiate d(loss)/d(W)
//          through whichever concrete rule is used. If monomorphization works,
//          Enzyme sees the concrete function body and can differentiate it.
//
// PATTERN: Generic function with trait bound -> monomorphized at call site ->
//          Enzyme sees concrete code -> differentiation works (hypothesis).

use std::autodiff::autodiff;

// --- The trait (models MemoryRule from the spec) ---

trait MemoryRule {
    fn forward(&self, state: f32, input: f32) -> f32;
}

// --- Concrete implementations ---

/// Delta Rule: M_{t+1} = M_t + lr * (input - M_t)
/// Simplest memory update. Linear in both state and input.
struct DeltaRule {
    lr: f32,
}

impl MemoryRule for DeltaRule {
    fn forward(&self, state: f32, input: f32) -> f32 {
        state + self.lr * (input - state)
    }
}

/// Momentum Rule: M_{t+1} = momentum * M_t + (1 - momentum) * input
/// Models exponential moving average — common in Titans LMM.
struct MomentumRule {
    momentum: f32,
}

impl MemoryRule for MomentumRule {
    fn forward(&self, state: f32, input: f32) -> f32 {
        self.momentum * state + (1.0 - self.momentum) * input
    }
}

/// Gated Rule: M_{t+1} = gate * M_t + (1 - gate) * lr * input
/// Two learnable parameters. Tests multi-field struct through trait.
struct GatedRule {
    gate: f32,
    lr: f32,
}

impl MemoryRule for GatedRule {
    fn forward(&self, state: f32, input: f32) -> f32 {
        self.gate * state + (1.0 - self.gate) * self.lr * input
    }
}

// --- Monomorphized functions that Enzyme should differentiate ---

// Strategy: We can't put #[autodiff] on a generic function directly
// (Enzyme needs a concrete function). So we create concrete wrappers
// that the compiler monomorphizes.

#[autodiff(d_apply_delta, Reverse, Duplicated, Active, Active, Active)]
fn apply_delta(rule: &DeltaRule, state: f32, input: f32) -> f32 {
    rule.forward(state, input)
}

#[autodiff(d_apply_momentum, Reverse, Duplicated, Active, Active, Active)]
fn apply_momentum(rule: &MomentumRule, state: f32, input: f32) -> f32 {
    rule.forward(state, input)
}

#[autodiff(d_apply_gated, Reverse, Duplicated, Active, Active, Active)]
fn apply_gated(rule: &GatedRule, state: f32, input: f32) -> f32 {
    rule.forward(state, input)
}

// --- Test: generic function called with concrete type ---
// This is closer to how NL_Hecate would use it: a generic forward pass
// that calls rule.forward(), monomorphized per rule type.

fn apply_generic<R: MemoryRule>(rule: &R, state: f32, input: f32) -> f32 {
    rule.forward(state, input)
}

// We can't directly annotate the generic, but we can annotate a
// concrete instantiation via a wrapper:
#[autodiff(d_generic_delta, Reverse, Duplicated, Active, Active, Active)]
fn generic_delta_wrapper(rule: &DeltaRule, state: f32, input: f32) -> f32 {
    apply_generic(rule, state, input)
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
    println!("\n=== Test 03: Static Trait Dispatch ===\n");

    let eps = 1e-4_f32;
    let tol = 1e-3_f32;
    let mut pass = 0;
    let mut fail = 0;

    // --- Delta Rule ---
    // f(lr, state, input) = state + lr * (input - state)
    // df/d_lr = input - state
    // df/d_state = 1 - lr
    // df/d_input = lr
    {
        let rule = DeltaRule { lr: 0.1 };
        let state = 2.0_f32;
        let input = 5.0_f32;
        let mut d_rule = DeltaRule { lr: 0.0 };

        let (val, d_state, d_input) = d_apply_delta(&rule, &mut d_rule, state, input, 1.0);

        // Verify forward
        let expected = 2.0 + 0.1 * (5.0 - 2.0);  // 2.3
        assert!((val - expected).abs() < 1e-5, "delta forward: {} vs {}", val, expected);

        // Verify gradients via finite diff
        let fd_lr = {
            let r_plus = DeltaRule { lr: rule.lr + eps };
            let r_minus = DeltaRule { lr: rule.lr - eps };
            (apply_delta(&r_plus, state, input) - apply_delta(&r_minus, state, input)) / (2.0 * eps)
        };
        if check_gradient("delta: d/d_lr", d_rule.lr, fd_lr, tol) { pass += 1; } else { fail += 1; }

        let fd_state = (apply_delta(&rule, state + eps, input) - apply_delta(&rule, state - eps, input)) / (2.0 * eps);
        if check_gradient("delta: d/d_state", d_state, fd_state, tol) { pass += 1; } else { fail += 1; }

        let fd_input = (apply_delta(&rule, state, input + eps) - apply_delta(&rule, state, input - eps)) / (2.0 * eps);
        if check_gradient("delta: d/d_input", d_input, fd_input, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Momentum Rule ---
    {
        let rule = MomentumRule { momentum: 0.9 };
        let state = 1.0_f32;
        let input = 3.0_f32;
        let mut d_rule = MomentumRule { momentum: 0.0 };

        let (val, d_state, d_input) = d_apply_momentum(&rule, &mut d_rule, state, input, 1.0);
        let expected = 0.9 * 1.0 + 0.1 * 3.0;  // 1.2
        assert!((val - expected).abs() < 1e-5);

        let fd_mom = {
            let r_plus = MomentumRule { momentum: rule.momentum + eps };
            let r_minus = MomentumRule { momentum: rule.momentum - eps };
            (apply_momentum(&r_plus, state, input) - apply_momentum(&r_minus, state, input)) / (2.0 * eps)
        };
        if check_gradient("momentum: d/d_momentum", d_rule.momentum, fd_mom, tol) { pass += 1; } else { fail += 1; }

        let fd_state = (apply_momentum(&rule, state + eps, input) - apply_momentum(&rule, state - eps, input)) / (2.0 * eps);
        if check_gradient("momentum: d/d_state", d_state, fd_state, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Gated Rule (two learnable params) ---
    {
        let rule = GatedRule { gate: 0.6, lr: 0.2 };
        let state = 1.5_f32;
        let input = 4.0_f32;
        let mut d_rule = GatedRule { gate: 0.0, lr: 0.0 };

        let (val, d_state, d_input) = d_apply_gated(&rule, &mut d_rule, state, input, 1.0);
        let expected = 0.6 * 1.5 + 0.4 * 0.2 * 4.0;  // 0.9 + 0.32 = 1.22
        assert!((val - expected).abs() < 1e-5);

        let fd_gate = {
            let r_plus = GatedRule { gate: rule.gate + eps, lr: rule.lr };
            let r_minus = GatedRule { gate: rule.gate - eps, lr: rule.lr };
            (apply_gated(&r_plus, state, input) - apply_gated(&r_minus, state, input)) / (2.0 * eps)
        };
        if check_gradient("gated: d/d_gate", d_rule.gate, fd_gate, tol) { pass += 1; } else { fail += 1; }

        let fd_lr = {
            let r_plus = GatedRule { gate: rule.gate, lr: rule.lr + eps };
            let r_minus = GatedRule { gate: rule.gate, lr: rule.lr - eps };
            (apply_gated(&r_plus, state, input) - apply_gated(&r_minus, state, input)) / (2.0 * eps)
        };
        if check_gradient("gated: d/d_lr", d_rule.lr, fd_lr, tol) { pass += 1; } else { fail += 1; }

        // Verify state and input gradients too
        let fd_state = (apply_gated(&rule, state + eps, input) - apply_gated(&rule, state - eps, input)) / (2.0 * eps);
        if check_gradient("gated: d/d_state", d_state, fd_state, tol) { pass += 1; } else { fail += 1; }

        let fd_input = (apply_gated(&rule, state, input + eps) - apply_gated(&rule, state, input - eps)) / (2.0 * eps);
        if check_gradient("gated: d/d_input", d_input, fd_input, tol) { pass += 1; } else { fail += 1; }
    }

    // --- Generic wrapper (monomorphized through apply_generic) ---
    {
        let rule = DeltaRule { lr: 0.3 };
        let state = 1.0_f32;
        let input = 4.0_f32;
        let mut d_rule = DeltaRule { lr: 0.0 };

        let (val, d_state, d_input) = d_generic_delta(&rule, &mut d_rule, state, input, 1.0);
        let expected = 1.0 + 0.3 * (4.0 - 1.0);  // 1.9
        assert!((val - expected).abs() < 1e-5);

        let fd_lr = {
            let r_plus = DeltaRule { lr: rule.lr + eps };
            let r_minus = DeltaRule { lr: rule.lr - eps };
            (generic_delta_wrapper(&r_plus, state, input) - generic_delta_wrapper(&r_minus, state, input)) / (2.0 * eps)
        };
        if check_gradient("generic_delta: d/d_lr", d_rule.lr, fd_lr, tol) { pass += 1; } else { fail += 1; }

        let fd_state = (generic_delta_wrapper(&rule, state + eps, input) - generic_delta_wrapper(&rule, state - eps, input)) / (2.0 * eps);
        if check_gradient("generic_delta: d/d_state", d_state, fd_state, tol) { pass += 1; } else { fail += 1; }

        let fd_input = (generic_delta_wrapper(&rule, state, input + eps) - generic_delta_wrapper(&rule, state, input - eps)) / (2.0 * eps);
        if check_gradient("generic_delta: d/d_input", d_input, fd_input, tol) { pass += 1; } else { fail += 1; }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
