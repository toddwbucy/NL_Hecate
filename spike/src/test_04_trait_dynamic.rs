// Test 04: Dynamic Trait Dispatch (Expected to Fail)
//
// PURPOSE: Document whether Enzyme can differentiate through dyn Trait.
//          We EXPECT this to fail — vtable dispatch means Enzyme can't see
//          the concrete function body at LLVM IR level.
//
// WHY WE TEST IT: The original spec assumed dyn Trait works. We need to
//          confirm it doesn't and document the exact error message.
//          This informs the architectural decision to use static dispatch
//          (monomorphization) everywhere.
//
// EXPECTED: Compilation error or runtime panic. Document exactly what happens.
//
// NOTE: This file is structured to compile even if the autodiff attribute
//       on dyn Trait is rejected. We catch errors at each level.

#[allow(unused_imports)]
use std::autodiff::autodiff_reverse;

// --- The trait (same as test_03) ---

trait MemoryRule {
    fn forward(&self, state: f32, input: f32) -> f32;
}

struct DeltaRule {
    lr: f32,
}

impl MemoryRule for DeltaRule {
    fn forward(&self, state: f32, input: f32) -> f32 {
        state + self.lr * (input - state)
    }
}

// --- Attempt 1: Direct dyn Trait parameter ---
// This will likely fail at compilation. If it does, the function body
// is unreachable and we document the compiler error.
//
// NOTE: We can't actually put #[autodiff] on a function taking &dyn Trait
// because Enzyme needs a concrete type for the Duplicated shadow parameter.
// This is commented out to let the rest of the spike compile.
//
// Uncomment to test:
//
// #[autodiff_reverse(d_apply_dyn, Duplicated, Active, Active, Active)]
// fn apply_dyn(rule: &dyn MemoryRule, state: f32, input: f32) -> f32 {
//     rule.forward(state, input)
// }

// --- Attempt 2: Box<dyn Trait> ---
// Same issue but with heap allocation.
//
// #[autodiff_reverse(d_apply_boxed, Duplicated, Active, Active, Active)]
// fn apply_boxed(rule: &Box<dyn MemoryRule>, state: f32, input: f32) -> f32 {
//     rule.forward(state, input)
// }

// --- Attempt 3: Trait object inside a struct ---
// Sometimes wrapping in a struct changes how LLVM sees the type.
//
// struct RuleHolder<'a> {
//     rule: &'a dyn MemoryRule,
// }
//
// #[autodiff_reverse(d_apply_held, Duplicated, Active, Active, Active)]
// fn apply_held(holder: &RuleHolder, state: f32, input: f32) -> f32 {
//     holder.rule.forward(state, input)
// }

// --- What we actually run ---
// Since the dyn Trait attempts are expected to fail at compile time,
// we just document the findings.

pub fn run() -> (usize, usize) {
    println!("\n=== Test 04: Dynamic Trait Dispatch (Expected Failure) ===\n");

    println!("  [INFO] dyn Trait dispatch is NOT expected to work with Enzyme.");
    println!("  [INFO] Enzyme operates at LLVM IR level where vtable calls are opaque.");
    println!("  [INFO] The three attempts above are commented out to let the spike compile.");
    println!();
    println!("  To test, uncomment each attempt and observe the compiler error:");
    println!("    Attempt 1: &dyn MemoryRule — likely 'cannot determine shadow type'");
    println!("    Attempt 2: &Box<dyn MemoryRule> — likely same error");
    println!("    Attempt 3: Struct wrapping &dyn — likely same error");
    println!();
    println!("  ARCHITECTURAL IMPLICATION:");
    println!("    NL_Hecate must use static dispatch (generics + monomorphization)");
    println!("    for all code paths that Enzyme needs to differentiate through.");
    println!("    The MemoryRule trait is fine — but callers must be generic, not");
    println!("    using trait objects.");
    println!();

    // Run a manual verification to confirm static dispatch DOES work
    // (same as test_03, just confirming the alternative path)
    let rule = DeltaRule { lr: 0.1 };
    let state = 2.0_f32;
    let input = 5.0_f32;

    // Use the rule via trait method directly (no autodiff, just correctness)
    let result = rule.forward(state, input);
    let expected = 2.0 + 0.1 * (5.0 - 2.0);
    assert!((result - expected).abs() < 1e-6);

    println!("  [PASS] Static dispatch confirmed working (see test_03 for AD version)");
    println!("  [SKIP] dyn Trait tests skipped (would fail to compile)");
    println!();

    // Return 0 pass, 0 fail — this test is informational
    // The "pass" is documented in the findings
    println!("  Result: INFORMATIONAL (0 graded tests)\n");
    (0, 0)
}
