# Track Zero: First Implementation Milestone

```
CONTRACT
  Purpose:    Before any NL-specific complexity, validate that the infrastructure
              works end-to-end: Rust forward → Enzyme backward → CUDA attention →
              Python orchestration. Track Zero is a spike, not a product.
              If Track Zero doesn't work, nothing else matters.
  Expects:    Pinned toolchain (Phase 0 complete).
              Enzyme successfully differentiating through Rust trait patterns.
  Guarantees: A working end-to-end pipeline from Rust code to trained model.
              A regression anchor: k=1 NL_Hecate matches PyTorch baseline.
              Every infrastructure layer exercised before NL complexity enters.
  Cost:       2-4 weeks (after Phase 0 toolchain spike).
  Trade-off:  Track Zero delays NL-specific work. But it surfaces build system,
              Enzyme, and precision issues immediately — problems that would
              otherwise compound behind layers of CMS complexity.
  Position:   specs/infrastructure/track_zero/00_track_zero.md
  Source:     Committee review cycle; TNT two-stage training wisdom applied to
              implementation planning; "contact with reality" principle
```

## Phase 0: Enzyme Spike (2 weeks, time-boxed)

```
GOAL: Answer the question "Can Enzyme differentiate through the Rust patterns
      we need?" Everything else is contingent on this answer.

PROOF-OF-CONCEPT PROGRAM (~200 lines):

  trait MemoryRule {
    fn forward(&self, state: &Tensor, input: &Tensor) -> Tensor;
  }

  struct DeltaRule { w_k: Tensor, w_v: Tensor }
  impl MemoryRule for DeltaRule {
    #[enzyme_opaque]
    #[custom_vjp(delta_rule_backward)]
    fn forward(&self, state: &Tensor, input: &Tensor) -> Tensor {
      // inner loop: analytical gradient, state mutation
    }
  }

  #[autodiff]
  fn model_forward(rule: &dyn MemoryRule, x: &Tensor) -> Tensor {
    // project x, compute gates, call rule.forward(), compute loss
    // Enzyme needs to differentiate d(loss)/d(w_k) through this
  }

  // THE TEST: does d(loss)/d(w_k) flow correctly through trait dispatch
  // into a #[custom_vjp] function?

OUTCOMES (exactly one of three):

  OUTCOME 1: Works.
    Pin the toolchain at whatever nightly/LLVM/flags made it work.
    Document the working configuration. Move to Track Zero-A.

  OUTCOME 2: Works with simplifications.
    Enzyme differentiates Rust code, but not through trait objects,
    or not with specific generic bound patterns. Document exactly
    which patterns fail. Redesign affected trait system components
    to stay within what Enzyme supports.
    The spec adapts to reality, not the reverse.

  OUTCOME 3: Doesn't work.
    Enzyme on Rust is too immature for this use case.
    This is the MOST VALUABLE outcome to discover in week 2 vs month 6.
    Proceed to Plan B (see below).

ABORT CRITERIA: If still debugging LLVM linking on day 14, that's Outcome 3.
                Do not extend the spike. Make the go/no-go decision.

DELIVERABLE: A spike notebook documenting everything that broke,
             every workaround, every version incompatibility.
             Published to the repository regardless of outcome.
```

## Track Zero-A: Pure Infrastructure (no memory branch)

```
GOAL: Validate Rust → Enzyme → CUDA → Python pipeline with standard attention.
      No memory update rule. No inner loop. No CMS.
      This is a standard sliding-window attention Transformer built through
      the NL_Hecate infrastructure stack.

ARCHITECTURE:
  - Single-block model
  - Sliding window attention (SWA) only — no memory branch
  - Rust forward pass (projections, attention, loss)
  - CUDA kernel for SWA attention (forward + backward pair)
  - Enzyme computes d(loss)/d(W) through the Rust code
  - Python orchestration via PyO3

REGRESSION ANCHOR:
  Train an equivalent SWA Transformer in PyTorch on the same data.
  Compare loss curves. They must match within tolerance:
    TOLERANCE: <1% relative difference at convergence
  Any deviation is an infrastructure bug — the two systems implement
  the same math, so they must produce the same results.

PASS CRITERIA:
  1. Forward pass produces valid output (not NaN)
  2. Enzyme backward produces valid gradients
  3. Loss decreases over 1K steps
  4. Loss curve matches PyTorch baseline within tolerance
  5. Integration gradient test passes (see below)

INTEGRATION GRADIENT TEST:
  Compute d(loss)/d(W_K) through a complete small forward pass
  (single block, 64x64 matrices, 3 chunks of 8 tokens) via:
    (a) Enzyme (the production path)
    (b) Finite differences (numerical ground truth)
  These must match within tolerance (rtol=1e-4, atol=1e-6).
  This catches chain-rule composition bugs that kernel-level tests miss.
```

## Track Zero-B: Memory Introduction

```
GOAL: Validate that Enzyme correctly differentiates through the memory
      update rule — the NL-specific gradient flow.

ARCHITECTURE:
  - Single-block model
  - Delta Rule (simplest memory update: no momentum, no MLP)
  - MAG composition (memory gates attention output)
  - Chunkwise parallelization (simplest strategy)
  - Single CMS level (k=1 — degenerate case, no frequency scheduling)

WHY DELTA RULE + MAG:
  Delta Rule has half the state of full Titans LMM (no momentum accumulator).
  MAG is simpler than MAC (no full-causal attention, just SWA + gate).
  Chunkwise GD is the simplest parallelization (no scan, no hierarchy).
  k=1 means no CMS — the frequency scheduler is trivially "always active."
  This exercises every NL-specific layer without CMS complexity.

REGRESSION ANCHOR:
  Implement Delta Rule + MAG in PyTorch as a reference.
  This is NOT a vanilla Transformer — it's the same NL architecture
  implemented in PyTorch using torch.autograd.Function.
  Compare loss curves. Must match within tolerance.
  If Zero-A passes and Zero-B fails, the bug is in the memory gradient flow.

PASS CRITERIA:
  1-5. Same as Track Zero-A
  6. Memory state evolves meaningfully (not constant, not diverging)
  7. Gate values are in [0,1] and data-dependent (not collapsed to 0 or 1)
  8. Integration gradient test includes d(loss)/d(W_K) flowing THROUGH
     the Delta Rule inner loop via #[custom_vjp]
```

## Phase 2: CMS Introduction (k=2)

```
PREREQUISITE: Track Zero-B passes all criteria.

ARCHITECTURE:
  - Delta Rule + MAG + chunkwise (same as Zero-B)
  - k=2 CMS levels: frequencies [1, 8]
  - Error buffer accumulation for level 1 (frozen 7 out of 8 steps)

TRANSITION CRITERIA (when is Phase 2 "done"):
  The k=2 model must demonstrate MEASURABLE IMPROVEMENT over k=1 on at
  least one task metric (perplexity or downstream accuracy).
  If k=2 doesn't beat k=1, the multi-scale mechanism needs debugging —
  adding more levels won't help. This is the FALSIFICATION TEST for CMS.

VALIDATION:
  Test at three horizons:
    100 steps   — smoke test (catches crashes, NaN, divergence)
    1K steps    — single-level convergence
    10K steps   — multi-level interaction (covers 1250 cycles of level 1)
  Four criteria at each horizon:
    (a) Forward produces valid output
    (b) Backward produces valid gradients
    (c) Gradient magnitudes are bounded
    (d) Loss decreases (convergence)

ERROR BUFFER HEALTH:
  Monitor norm_ratio = ||accumulated_error|| / ||single_step_gradient||
  at every sync point (when level 1 fires).
  If norm_ratio > threshold (configurable, default 10.0):
    LOG WARNING and optionally clip the accumulated error.
  This prevents the "error buffer bomb" (512 accumulated steps as one giant update).
```

## Phase 3: Full Design Space

```
PREREQUISITE: Phase 2 passes transition criteria (k=2 beats k=1).

ARCHITECTURE:
  - k=4 CMS: frequencies [1, 8, 64, 512]
  - Multiple memory update rules (Titans LMM, MONETA, Lattice OSR)
  - Multiple parallelization strategies

VALIDATION PROTOCOL:
  Combinatorial sweep of compilable pairings against:
    (a) Valid output  (b) Valid gradients  (c) Bounded magnitudes  (d) Convergence
  At THREE horizons: 100 steps (smoke), 1K steps (convergence), 10K steps (stability)
  For k=4 with C_max=512, 10K steps covers ~20 full cycles of slowest level.

  DISTINCTION:
    Smoke test (10K steps): gates integration into the build.
    Stability test (100K steps): gates any claim that a config is "validated."
    Smoke catches crashes and divergence.
    Stability catches slow drift, gradual retention interference, accumulation pathologies.

  FALSIFICATION CRITERION (from committee review):
    If >20% of "valid by constraint matrix" combinations produce degenerate dynamics,
    the orthogonality framing is wrong and the constraint matrix needs rebuilding
    from empirical evidence rather than paper claims.
```

## Plan B: If Enzyme Doesn't Work

```
TWO FALLBACK PATHS (not mutually exclusive):

PATH A: Manual Chain-Rule Composition
  The kernel-pair architecture already specifies every backward.
  If Enzyme can't compose them, write the composition manually.
  This is the chain rule applied to ~5-8 operations per forward pass.
  Tedious, error-prone, verified against finite differences.
  The Rust code still compiles; the AD is just manual instead of automatic.

PATH B: PyTorch First
  Implement Delta Rule + MAG as a PyTorch autograd.Function.
  Validate the algorithms. Get a model converging.
  Characterize numerical behavior. Understand the math.
  Port to Rust later when the AD landscape matures.

  This is NOT abandoning the Rust vision.
  The spec's value is in the algorithms and constraints, not the
  implementation language. A working PyTorch prototype that validates
  CMS frequency scheduling, error buffer dynamics, and multi-level
  retention is scientifically valuable regardless of language.

  The NL_Hecate spec remains the architecture document.
  The PyTorch prototype becomes the reference implementation.
  The Rust build follows when toolchain maturity allows.
```

## Axiom Compliance

- **Contact with reality**: Track Zero forces infrastructure validation before NL complexity
- **TNT IS #4** (architecture-agnostic): Track Zero validates infrastructure, not architecture
- **Falsification**: Every phase has pass/fail criteria and regression anchors
