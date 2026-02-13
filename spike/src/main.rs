// NL_Hecate Enzyme Spike — Phase 0 Test Harness
//
// This is THE deliverable of the 2-week spike.
// Run with: RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --release
//
// Each test module exercises a specific Enzyme capability that
// NL_Hecate's architecture depends on. Results are printed with
// [PASS]/[FAIL] for each gradient check.
//
// See spike/README.md for what each test proves.

#![feature(autodiff)]

mod test_01_basic_ad;
mod test_02_struct_ad;
mod test_03_trait_static;
mod test_04_trait_dynamic;
mod test_05_custom_backward;
mod test_06_chain_rule;
mod test_07_integration;

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  NL_Hecate Enzyme Spike — Phase 0 Test Suite    ║");
    println!("╚══════════════════════════════════════════════════╝");

    let mut total_pass = 0_usize;
    let mut total_fail = 0_usize;

    let tests: Vec<(&str, fn() -> (usize, usize))> = vec![
        ("01: Basic Scalar AD",           test_01_basic_ad::run),
        ("02: Struct Field AD",           test_02_struct_ad::run),
        ("03: Static Trait Dispatch",     test_03_trait_static::run),
        ("04: Dynamic Trait (Expected Fail)", test_04_trait_dynamic::run),
        ("05: Kernel-Pair Simulation",    test_05_custom_backward::run),
        ("06: Chain Rule Composition",    test_06_chain_rule::run),
        ("07: Integration (Mini NL)",     test_07_integration::run),
    ];

    let mut results = Vec::new();

    for (name, test_fn) in &tests {
        let (p, f) = test_fn();
        total_pass += p;
        total_fail += f;
        results.push((*name, p, f));
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                        ║");
    println!("╠══════════════════════════════════════════════════╣");

    for (name, p, f) in &results {
        let status = if *f == 0 && *p > 0 {
            "PASS"
        } else if *f == 0 && *p == 0 {
            "SKIP"
        } else {
            "FAIL"
        };
        println!("║  [{}] {:40} {:>2}/{:<2} ║", status, name, p, p + f);
    }

    println!("╠══════════════════════════════════════════════════╣");
    println!("║  Total: {}/{} gradient checks passed              ║", total_pass, total_pass + total_fail);
    println!("╚══════════════════════════════════════════════════╝");

    // Determine outcome
    println!();
    if total_fail == 0 && total_pass > 0 {
        println!("OUTCOME: All gradient checks passed.");
        println!("  → If tests 01-03 + 05-07 pass: OUTCOME 1 (proceed to Track Zero-A)");
        println!("  → Pin this toolchain version in TOOLCHAIN.md");
    } else if total_fail > 0 {
        println!("OUTCOME: {} gradient checks failed.", total_fail);
        println!("  → Investigate failures before making go/no-go decision");
        println!("  → If only test_04 fails: expected (dyn Trait), proceed with static dispatch");
        println!("  → If structural tests fail: likely OUTCOME 2 or 3");
    } else {
        println!("OUTCOME: No gradient checks ran.");
        println!("  → Likely toolchain issue. Check TOOLCHAIN.md for build instructions.");
    }

    // Exit with appropriate code
    std::process::exit(if total_fail > 0 { 1 } else { 0 });
}
