# Enzyme Spike Outcome

**Date**: 2026-02-13
**Duration**: 1 day (of 14 budgeted)
**Toolchain SHA**: d7daac06

## Outcome: 1

### Outcome 1: Works
> Enzyme differentiates through static trait dispatch, and we have a viable
> pattern for kernel pairs via manual chain rule composition.

**Action**: Pin toolchain. Proceed to Track Zero-A.

---

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| 01: Basic scalar AD | **4/4 PASS** | Enzyme gives exact analytical gradients |
| 02: Struct field gradients | **6/6 PASS** | Duplicated shadow accumulation works |
| 03: Static trait dispatch | **13/13 PASS** | Monomorphization works for Delta, Momentum, Gated rules |
| 04: Dynamic trait dispatch | **SKIP** | Expected to fail — vtable opaque |
| 05: Kernel-pair simulation | **10/10 PASS** | Strategy A+B+C all informative |
| 06: Chain rule composition | **6/6 PASS** | Enzyme(loss) × hand(kernel) × Enzyme(proj) |
| 07: Integration (mini NL) | **18/18 PASS** | Full project → memory → gate → loss chain |

**Total: 57/57 gradient checks passed**

## Key Findings

### What Works

1. **Exact analytical gradients** — Enzyme is more accurate than finite differences
2. **Struct field differentiation** — `Duplicated` annotation with shadow `&mut` struct
3. **Nested struct fields** — gradients flow through `model.proj.w_k` correctly
4. **Static trait dispatch** — monomorphized trait calls are fully transparent to Enzyme
5. **Manual chain rule** — Enzyme(outer) × hand_backward(inner) × Enzyme(outer) produces correct end-to-end gradients
6. **Multi-stage composition** — 3-stage backward (gate+loss → memory → projection) verified across 2 test points

### What Doesn't Work

1. **`dyn Trait` dispatch** — vtable calls opaque at LLVM IR level (expected)
2. **`#[custom_vjp]`** — not exposed in Rust-Enzyme wrapper
3. **`#[enzyme_opaque]`** — not exposed in Rust-Enzyme wrapper
4. **`black_box` with `#[autodiff_reverse]`** — crashes Enzyme (assertion failure)

### Workarounds Discovered

1. **Duplicated shadow seed fix** — shadow params always get unit-seed Jacobian; manually multiply by upstream gradient for chain rule
2. **Manual chain rule replaces `#[custom_vjp]`** — call Enzyme with seed=1.0, hand-backward for kernel, Enzyme with seed=1.0, then compose gradients explicitly
3. **`extern "C"` is NOT opaque in same crate** — Enzyme sees through it with fat LTO; for true opacity, use separate compilation units

## Architectural Implications for NL_Hecate

### Spec Changes Needed

| Spec File | Change Required |
|-----------|----------------|
| `specs/infrastructure/differentiation/00_enzyme_integration.md` | Replace `#[custom_vjp]` pattern with manual chain rule composition |
| `specs/infrastructure/differentiation/00_enzyme_integration.md` | Remove `#[enzyme_opaque]` references; document separate compilation unit pattern |
| `specs/contract.md` | Note that kernel-pair backward composition is explicit, not automatic |

### Confirmed Patterns

- [x] Static dispatch through trait bounds
- [x] Struct field differentiation (including nested)
- [x] Manual chain rule at kernel boundaries
- [x] Loss function differentiation with Active scalars
- [x] Multi-parameter gradient accumulation via Duplicated shadows

### Rejected Patterns

- [x] Dynamic trait dispatch (`dyn Trait`) — use static dispatch
- [x] `#[custom_vjp]` — use manual chain rule composition
- [x] `#[enzyme_opaque]` — use separate compilation units
- [x] `black_box` as soft barrier — it's a hard crash, not soft
- [x] Passing upstream gradient as seed to Duplicated params — always use seed=1.0 and multiply manually

## Build Notes

**Build time**: 11 minutes 24 seconds (48 cores, 247GB RAM)
**Disk usage**: ~30 GB (rust-enzyme source + build)
**Patches needed**: None
**rustc version**: 1.95.0-nightly (d7daac06d 2026-02-13)
**Build method**: `bootstrap.toml` with `llvm.enzyme = true`

## Recommendations

1. **Pin toolchain** at SHA d7daac06 for all NL_Hecate development
2. **Proceed to Track Zero-A** — pure SWA attention, no memory, full pipeline validation
3. **Update differentiation spec** — document manual chain rule pattern as the canonical approach
4. **Create chain-rule helper utilities** — wrap the "call with seed=1.0, multiply shadow" pattern in ergonomic helpers to prevent future bugs
5. **Test separate compilation unit opacity** in Track Zero-A when CUDA kernels come online
