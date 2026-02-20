# L_q Norm Retention (Bounded Memory)

```text
CONTRACT
  Purpose:    L_q norm retention combines L2 local proximity with an L_q global
              penalty that constrains the overall magnitude of memory. Where L2
              decay penalizes all entries uniformly, L_q norm retention shapes the
              memory's magnitude distribution: high q concentrates penalty on
              large entries (bounding peaks), low q spreads penalty more evenly
              (bounding total mass). MONETA uses (p,q) = (3,4) — the L_4 global
              penalty prevents any single memory entry from dominating.
  Expects:    Memory state W (matrix or MLP), previous state W_{t-1}, gates
              (alpha_t, eta_t), norm order q >= 1.
  Guarantees: Local retention D_t(W, W') = ||W - W'||^2_F keeps updates small.
              Global retention G_t(W) = ||W||_q^q bounds memory magnitude.
              The combination produces a normalization factor ||A||_q^{q-2} that
              rescales the gradient accumulator before writing to memory.
              At q = 2, recovers standard L2 weight decay (no normalization).
  Cost:       O(d^2) — element-wise power and norm computation. The q-norm
              requires |W_{jl}|^q per element plus a reduction for the norm.
              Negligible next to matmuls in the memory update.
  Trade-off:  More expressive than L2 (controls magnitude distribution, not just
              total magnitude). But introduces a normalization step that depends
              on the current accumulator state, making the update nonlinear in A.
              The q parameter is a design choice, not learned.
  Position:   specs/algorithms/retention_mechanisms/07_lq_norm.md
              Child of: 06_bregman.md (Bregman divergence general framework)
              Sibling of: 01_l2_weight_decay.md (q=2 special case),
                          03_elastic_net.md (L2+L1 vs L2+L_q)
  Source:     MIRAS (2504.13173) §5.3 Eqs 24-25 (MONETA variant);
              §5.1 Eqs 10-11 (l_p attentional bias, companion to L_q retention)
```

## MIRAS Decomposition

The retention decomposes into local proximity and global norm penalty:

<!-- HADES: miras_equations/eq-vp-retention-decomposition (§3.3, applied to L_q) -->
```text
Ret_t(W, W_{t-1}) = (1/eta_t) * D_t(W, W_{t-1})  +  (1/alpha_t) * G_t(W)
                     |---- Local Retention ----|     |-- Global Retention --|

-- Local: D_t(W, W') = ||W - W'||^2_F
--   Standard L2 proximity — same as L2 weight decay.
--   Controls per-step deviation from previous memory.

-- Global: G_t(W) = ||W||_q^q = sum_{j,l} |W_{jl}|^q
--   The L_q norm raised to the q-th power.
--   q > 2: penalizes large entries MORE than small ones (peak suppression)
--   q = 2: standard L2 global penalty (uniform)
--   q < 2 (q >= 1): penalizes small entries relatively MORE (promotes sparsity)
--   q = 1: recovers L1 penalty (elastic net without L2 global)
```

## The L_q Norm Effect

Different values of q shape the memory's magnitude distribution differently:

```text
-- Penalty per element: |W_{jl}|^q
--
-- q = 1:  |W|     → constant gradient sign(W). Drives to sparsity (L1).
-- q = 2:  |W|^2   → gradient 2W. Uniform decay. Standard L2.
-- q = 4:  |W|^4   → gradient 4|W|^3 sign(W). Large entries penalized cubically.
-- q → ∞: max|W|   → only the largest entry is penalized (L_inf limit).
--
-- MONETA default: q = 4
--   Large entries face 4x steeper penalty gradient than L2.
--   Small entries face gentler penalty than L2.
--   Net effect: bounded peaks, preserved fine structure.
```

## MONETA Update Rule (Eqs 24-25)

MONETA combines l_p attentional bias with L_q retention:

<!-- HADES: miras_equations/eq-024-025-moneta-spec (§5.3 Eqs 24-25) -->
```text
-- MONETA (p,q)-variant update rule (MIRAS Eqs 24-25):
FUNCTION: lq_retention_step(A: &mut Tensor, W: &mut Tensor,
                             grad_lp: &Tensor,
                             alpha_t: f32, eta_t: f32, q: f32) -> ()
  -- A: gradient accumulator (FTRL-style)
  -- W: memory state (derived from A via normalization)
  -- grad_lp: gradient of l_p loss (Eq 11)
  -- q: norm order (default: 4.0)

  -- Step 1: Accumulate with decay
  A = alpha_t * A - eta_t * grad_lp

  -- Step 2: L_q normalization
  -- The global penalty ||W||_q^q produces a normalization factor
  norm_q = (sum |A_{jl}|^q)^{1/q}                 -- L_q norm of A
  W = A / norm_q^{q-2}                             -- normalization by ||A||_q^{q-2}

  -- At q = 2: norm_q^{q-2} = norm_q^0 = 1 → W = A (no normalization)
  -- At q = 4: norm_q^{q-2} = norm_q^2 → W = A / ||A||_4^2
  -- The normalization constrains the output memory's magnitude.
```

## Connection to l_p Attentional Bias

MONETA pairs L_q retention with l_p attentional bias. The gradient that feeds
into the L_q retention step comes from the l_p loss:

<!-- HADES: miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10); miras_equations/eq-011-lp-closed-form (§5.1 Eq 11) -->
```text
-- l_p attentional bias (MIRAS Eq 10):
L(W; k_t, v_t) = ||W k_t - v_t||_p^p

-- Closed-form gradient (MIRAS Eq 11):
grad_lp = p * (Sign(W k_t - v_t) * |W k_t - v_t|^{p-1}) @ k_t^T

-- At p = 2: recovers standard Delta rule gradient 2*(W k - v) @ k^T
-- At p = 1: reduces to Sign(W k - v) @ k^T (value-less memory)
-- At p = 3 (MONETA default): uses quadratic error scaling

-- Smooth approximators (MIRAS Eq 25):
--   Sign(x) ≈ tanh(a * x)        (a controls sharpness)
--   |x|^{p-1} ≈ (x^2 + eps)^{(p-1)/2}  (eps for numerical stability)
```

The (p, q) pair is a joint design choice:
- p controls the **loss geometry** (how errors are measured)
- q controls the **retention geometry** (how memory magnitude is constrained)
- MONETA default: (p, q) = (3, 4)

## Bregman Divergence Connection

L_q norm retention is a special case of the Bregman divergence framework
(06_bregman.md):

<!-- HADES: miras_equations/eq-vp-retention-decomposition (§3.3); see also 06_bregman.md retention taxonomy -->
```text
-- Generator function:
phi(W) = (1/q) * ||W||_q^q = (1/q) * sum |W_{jl}|^q

-- Bregman divergence induced by phi:
D_phi(W, W') = (1/q) * sum (|W_{jl}|^q - |W'_{jl}|^q
               - q * |W'_{jl}|^{q-2} * W'_{jl} * (W_{jl} - W'_{jl}))

-- At q = 2: phi = (1/2)||W||^2_F → D_phi = (1/2)||W - W'||^2_F
--   Recovers standard L2 Bregman divergence.

-- MONETA simplifies by using L2 for local and L_q for global only,
-- rather than using the full L_q Bregman for both terms.
-- This is computationally cheaper and still provides the bounded-peak property.
```

## Properties

```text
-- Magnitude bounding:
--   After normalization: ||W||_q^q = ||A||_q^q / ||A||_q^{q(q-2)/q}
--   The normalization prevents unbounded accumulation in A from
--   translating to unbounded memory magnitude in W.

-- Selective penalty:
--   Unlike L2 which penalizes all entries equally (per-magnitude),
--   L_q with q > 2 preferentially suppresses outlier entries.
--   This is useful when memory should spread information across
--   many dimensions rather than concentrating in a few.

-- Accumulator vs memory:
--   MONETA maintains TWO states: A (accumulator) and W (normalized memory).
--   A accumulates raw gradient signal without normalization.
--   W is derived from A by applying the L_q normalization.
--   The query M(W, k) = W @ k uses the normalized W.
--   Gradients flow back through W to A via the normalization backward.
```

## Gradient Derivation (for tape integration)

<!-- HADES: Derived from miras_equations/eq-024-025-moneta-spec (§5.3 Eqs 24-25), analytical VJP -->
```text
-- Forward:
--   A_t = alpha_t * A_{t-1} - eta_t * grad_lp_t
--   norm_q = ||A_t||_q
--   W_t = A_t / norm_q^{q-2}

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dA_{t-1}, dL/dgrad_lp_t, dL/dalpha_t, dL/deta_t

-- Step 1: Backward through normalization W = A / norm_q^{q-2}
--   Let s = norm_q^{q-2} (scalar)
--   dL/dA_t (partial, through W) = dL/dW_t / s
--            - (q-2) * norm_q^{q-3} * (d(norm_q)/dA_t) * (A_t / s^2) . dL/dW_t
--
--   d(norm_q)/dA_{jl} = |A_{jl}|^{q-1} * sign(A_{jl}) / norm_q^{q-1}
--
--   Simplification for q = 4 (MONETA default):
--     s = norm_4^2
--     dL/dA_t = dL/dW_t / norm_4^2
--             - 2 * (sum dL/dW_t . A_t . |A_t|^2) / norm_4^6 * |A_t|^2 * sign(A_t)

-- Step 2: Backward through accumulation A_t = alpha * A_{t-1} - eta * grad
dL/dA_{t-1} = alpha_t * dL/dA_t
dL/dgrad_lp_t = -eta_t * dL/dA_t

dL/dalpha_t = trace(A_{t-1}^T @ dL/dA_t)     -- scalar gate gradient
dL/deta_t = -trace(grad_lp_t^T @ dL/dA_t)    -- scalar gate gradient
```

## Smooth Approximators

The l_p gradient (Eq 11) requires Sign and absolute value, which are
non-differentiable at zero. MIRAS Eq 25 specifies smooth approximators:

```text
-- Sign approximator:
Sign(x) ≈ tanh(a * x)
-- a = 10 (default): sharp but smooth transition near zero
-- Gradient: a * (1 - tanh(a*x)^2)

-- Absolute value power approximator:
|x|^{p-1} ≈ (x^2 + eps)^{(p-1)/2}
-- eps = 1e-6 (default): prevents division by zero
-- Gradient: (p-1) * x * (x^2 + eps)^{(p-3)/2}

-- These approximators ensure the full (p,q) pipeline is differentiable,
-- allowing the Wengert tape to propagate gradients through the l_p loss
-- into the L_q retention normalization without discontinuities.
```

## Implementation Notes

1. **MONETA already exists**: The MONETA variant in `core/src/moneta.rs` already
   implements (p,q) = (3,4). This spec documents the L_q retention mechanism
   in isolation from the MONETA variant, enabling other rules to use L_q
   retention with different attentional biases.

2. **Accumulator lifetime**: A is an `inner_loop_state` (same as momentum S).
   It persists within a forward pass but is NOT serialized with checkpoints.
   W (the normalized memory) is also `inner_loop_state` — both are reconstructed
   from outer_loop_params at each forward pass start.

3. **q = 2 degeneracy**: When q = 2, norm_q^{q-2} = 1, so W = A. The L_q
   normalization becomes identity, recovering standard L2 decay. Code should
   fast-path this case to avoid unnecessary norm computation.

4. **Numerical stability**: For large q and small entries, |W_{jl}|^q can
   underflow to zero. Use log-space computation: `q * log(|W| + eps)` then
   exp, or work with `(x^2 + eps)^{q/2}` as the smooth approximator.

5. **Pluggable retention dispatch**: When S3b-M1 (pluggable retention
   infrastructure) is implemented, L_q registers as a `RetentionKind::LqNorm(q)`
   variant. The q parameter is part of the variant configuration, not a
   runtime gate.

6. **Interaction with CMS**: At higher CMS levels (slower frequencies), the
   accumulator A has more steps between normalizations. The L_q constraint
   becomes relatively weaker per-step at slow levels, allowing slow memories
   to accumulate larger magnitudes before normalization clips them.

## Axiom Compliance

- **NL IS #4** (compressing context): The L_q norm bounds memory capacity, forcing compression of less important associations while preserving dominant patterns.
- **NL IS #9** (principled not ad hoc): The normalization factor emerges from the convex optimization of the L_q penalty, not from architectural intuition.
- **MIRAS IS #1** (orthogonal design choices): L_q retention is independent of the attentional bias — any (p, q) pair is valid, not just MONETA's (3, 4).
