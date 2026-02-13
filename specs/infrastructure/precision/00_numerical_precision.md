# Numerical Precision Strategy

```
CONTRACT
  Purpose:    NL models accumulate state across hundreds of steps in the inner
              loop. Precision errors compound. This spec defines which operations
              require which precision, and why.
  Expects:    Operations spanning inner loop, outer loop, gates, attention,
              error buffers, and CMS scheduling.
  Guarantees: No silent precision-related training failures.
              Inner-loop accumulation in fp32 (bf16 drift kills memory over 512 steps).
              Error buffers in fp32 unconditionally (dynamic range requirement).
              Clear precision annotation on every operation category.
  Cost:       fp32 inner loop uses 2x memory vs bf16 for memory matrices.
              Mixed-precision attention (bf16) saves memory where safe.
  Trade-off:  Higher precision = more memory, slower computation.
              Lower precision = faster but accumulation errors compound.
              The inner loop's multi-step nature makes this trade-off sharper
              than in conventional models where each layer is independent.
  Position:   specs/infrastructure/precision/00_numerical_precision.md
  Source:     Atlas precision finding (10B@fp32 beats 100B@fp8);
              Track A OOM experience; committee review (#10)
```

## Precision Map

```
OPERATION                           PRECISION    WHY
─────────────────────────────────── ──────────── ──────────────────────────────────
Inner-loop memory update (M, S)     fp32         Accumulates across 100s of steps.
                                                 bf16 drift corrupts memory silently.

Inner-loop gradient (analytical)    fp32         Input to memory update. Precision
                                                 loss here propagates into M.

Gate computations (sigmoid, softplus) fp32       sigmoid(x) near 0 or 1 is precision-
                                                 sensitive. softplus near 0 is precision-
                                                 sensitive. bf16 quantization at extremes
                                                 causes gate collapse.

Error buffers (CMS accumulated grad) fp32        Accumulates up to 512 steps of gradient.
                                                 bf16 CANNOT represent the dynamic range.
                                                 UNCONDITIONAL — never bf16.

Attention (QKV, softmax, output)    bf16         Standard FlashAttention practice.
                                                 Each attention op is independent (no
                                                 accumulation across steps). bf16 is safe.

Projection matrices (W_K, W_V, W_Q) bf16 stored  Stored in bf16 for memory savings.
                                    fp32 master   fp32 master copy held by optimizer.
                                                  Standard mixed-precision pattern.

Outer-loop gradient accumulation    fp32         Enzyme produces fp32 gradients.
                                                 Allreduce in fp32. No bf16 reduction.

Loss computation                    fp32         Small numerical differences in loss
                                                 compound through the optimizer.

Optimizer state (M3 momentum)       fp32         Momentum accumulators are long-lived.
                                                 One per CMS level, persists across
                                                 the entire build.
```

## Why Inner Loop fp32 Is Non-Negotiable

```
In a conventional Transformer:
  - Each layer processes input independently
  - bf16 error in layer 5 doesn't accumulate with bf16 error in layer 6
  - Mixed precision is safe because errors are bounded per-layer

In an NL inner loop:
  - Memory M is updated at every token: M_{t+1} = f(M_t, k_t, v_t)
  - At token 100, M reflects accumulated updates from tokens 0-99
  - bf16 rounding at each step: error = O(epsilon) per step
  - After 512 steps: accumulated error = O(512 * epsilon)
  - For bf16 (epsilon ≈ 4e-3): accumulated error ≈ 2.0
  - This is NOT small — it's on the order of the memory values themselves

  fp32 (epsilon ≈ 6e-8): accumulated error after 512 steps ≈ 3e-5
  This IS small. fp32 is safe for inner-loop accumulation.

The Atlas paper found that 10B parameters at fp32 outperform 100B at fp8.
This is a direct consequence: parameter EFFICIENCY matters more than parameter
COUNT when the model self-modifies at test time. Precision IS performance.
```

## Mixed-Precision Strategy

```
PHASE: Build (outer loop active)
  Forward pass:
    - Projections: bf16 (W_K, W_V, W_Q are stored bf16)
    - Attention: bf16 (FlashAttention standard)
    - Gates: fp32 (upcast from bf16 input)
    - Inner loop: fp32 (memory, momentum, gradients)
    - Loss: fp32

  Backward pass (Enzyme):
    - Enzyme produces fp32 gradients for outer-loop params
    - Gradient allreduce in fp32 (no bf16 reduction)
    - Optimizer updates fp32 master copy
    - fp32 master → bf16 stored copy (for next forward)

PHASE: Test / Stream (outer loop frozen)
  Forward pass: identical precision to Build
  No backward pass (no Enzyme, no gradient)
  Memory savings: no gradient tensors, no optimizer state
  But inner-loop memory matrices are still fp32
```

## Precision Violations (What Goes Wrong)

```
VIOLATION 1: bf16 inner loop
  Symptom: memory values saturate or oscillate after ~100 steps
  Cause: accumulated rounding error exceeds signal magnitude
  Fix: fp32 inner loop (mandatory)

VIOLATION 2: bf16 error buffers
  Symptom: level-3 update (every 512 steps) produces NaN or inf
  Cause: 512 accumulated bf16 gradients overflow representable range
  Fix: fp32 error buffers (unconditional)

VIOLATION 3: bf16 gate computation
  Symptom: gates collapse to exactly 0.0 or 1.0
  Cause: sigmoid(x) for |x| > 10 rounds to 0/1 in bf16
  Fix: fp32 gate computation, then cast output to bf16 if needed

VIOLATION 4: bf16 gradient allreduce
  Symptom: parameters diverge across ranks after many steps
  Cause: bf16 reduction introduces different rounding per rank
  Fix: fp32 allreduce (standard practice)
```

## Axiom Compliance

- **Atlas IS #4** (precision tradeoff): 10B@fp32 > 100B@fp8 is the empirical grounding
- **CS-47** (in-place modification): fp32 master copy prevents precision loss from repeated cast
- **NL IS #7** (self-modifying): Self-modification requires precision — the modifications ARE the model
