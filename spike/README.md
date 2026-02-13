# Enzyme Spike — Phase 0

**Goal**: Answer "Can Enzyme differentiate through the Rust patterns NL_Hecate needs?"

**Time-boxed**: 14 days. If still debugging LLVM linking on day 14, that's Outcome 3.

## What We're Testing

| Test | File | What It Proves |
|------|------|----------------|
| 01 | `test_01_basic_ad.rs` | Scalar reverse-mode AD works at all |
| 02 | `test_02_struct_ad.rs` | Gradient flows to struct fields (W_K, W_V) |
| 03 | `test_03_trait_static.rs` | Enzyme sees through monomorphized trait dispatch |
| 04 | `test_04_trait_dynamic.rs` | `dyn Trait` dispatch (expected to fail) |
| 05 | `test_05_custom_backward.rs` | Simulated kernel-pair without `#[custom_vjp]` |
| 06 | `test_06_chain_rule.rs` | Enzyme + hand-written backward compose correctly |
| 07 | `test_07_integration.rs` | Mini NL forward pass: project -> update -> loss |

## Why This Matters

The spec assumes three Enzyme features that **don't exist** in the Rust wrapper:
- `#[custom_vjp]` — not surfaced from Enzyme internals
- `#[enzyme_opaque]` — not documented
- `dyn Trait` dispatch — vtable limitation

The spike finds workarounds or proves they're needed, before we build 10K lines on wrong assumptions.

## Verification

Every gradient is verified against finite differences:
```
enzyme_grad = autodiff(f, x, seed=1.0)
finite_diff = (f(x + eps) - f(x - eps)) / (2 * eps)   // eps = 1e-4
assert |enzyme_grad - finite_diff| / |finite_diff| < 1e-3
```

## Outcomes

**Outcome 1: Works** — Pin toolchain, proceed to Track Zero-A.

**Outcome 2: Works with simplifications** — Document constraints, revise specs.

**Outcome 3: Doesn't work** — Evaluate Plan B (manual chain rule / PyTorch prototype / C++ FFI).

## Running

```bash
# After toolchain is built (see TOOLCHAIN.md):
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme run --release
```

## Log

See `findings/` for daily progress and final outcome.
