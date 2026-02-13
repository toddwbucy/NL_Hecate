# Enzyme Spike Outcome

**Date**: ____
**Duration**: ____ days (of 14 budgeted)
**Toolchain SHA**: ____

## Outcome: [ 1 / 2 / 3 ]

### Outcome 1: Works
> Enzyme differentiates through static trait dispatch, and we have a viable
> pattern for kernel pairs (even if not `#[custom_vjp]`).

**Action**: Pin toolchain. Proceed to Track Zero-A.

### Outcome 2: Works with simplifications
> Enzyme works but requires architectural changes.

**Required changes**:
- [ ] ____
- [ ] ____

**Action**: Revise affected specs. Proceed to Track Zero-A with constraints.

### Outcome 3: Doesn't work
> Enzyme-Rust too immature for this use case.

**Plan B recommendation**: [ B-1: Manual chain rule / B-2: PyTorch first / B-3: C++ FFI ]

---

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| 01: Basic scalar AD | | |
| 02: Struct field gradients | | |
| 03: Static trait dispatch | | |
| 04: Dynamic trait dispatch | | Expected to fail |
| 05: Kernel-pair simulation | | |
| 06: Chain rule composition | | |
| 07: Integration (mini NL) | | |

## Key Findings

### What Works

1. ____
2. ____

### What Doesn't Work

1. ____
2. ____

### Workarounds Discovered

1. ____
2. ____

## Architectural Implications for NL_Hecate

### Spec Changes Needed

| Spec File | Change Required |
|-----------|----------------|
| | |

### Confirmed Patterns

- [ ] Static dispatch through trait bounds
- [ ] Struct field differentiation
- [ ] Manual chain rule at kernel boundaries
- [ ] FFI boundary as opaque barrier
- [ ] Tuple return differentiation

### Rejected Patterns

- [ ] Dynamic trait dispatch (`dyn Trait`)
- [ ] `#[custom_vjp]` (not available)
- [ ] `#[enzyme_opaque]` (not available)

## Build Notes

**Build time**: ____ minutes
**Disk usage**: ____ GB
**Patches needed**: ____
**LLVM version**: ____
**Enzyme version**: ____

## Recommendations

____
