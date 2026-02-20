# Follow the Regularized Leader (FTRL)

```text
CONTRACT
  Purpose:    FTRL is one of three equivalent viewpoints for deriving memory
              update rules (alongside Online GD and Learning-Retaining). Where
              GD takes one gradient step per token, FTRL solves a global
              optimization over ALL past linearized losses plus a regularizer.
              This viewpoint is essential because: (1) it reveals that different
              regularizers R(W) produce different memory rules as special cases,
              (2) it connects inner-loop memory updates to online learning theory,
              and (3) it enables elastic net retention via soft thresholding.
  Expects:    Sequence of loss functions l(W; k_t, v_t), regularizer R_t(W),
              learning rate eta_t. Losses may use any attentional bias (L2, l_p).
  Guarantees: Regret-bounded memory updates. The FTRL solution minimizes
              cumulative linearized loss subject to a regularization penalty.
              Under mild convexity assumptions, FTRL and Learning-Retaining
              viewpoints are equivalent (MIRAS Proposition 3.2).
  Cost:       Depends on R_t. Quadratic R → O(d^2) closed-form. L1/elastic net
              R → O(d^2) with soft thresholding. General R → may require
              iterative solve (mirror descent step).
  Trade-off:  More principled than GD (global objective, regret bounds) but
              requires solving an optimization per token instead of taking a
              single gradient step. In practice, the closed-form solutions
              (when they exist) have identical cost to GD.
  Position:   specs/algorithms/optimization_machinery/05_ftrl.md
              Sibling of 03_dgd.md, 04_dmgd.md. Provides the theoretical
              framework that unifies GD, DGD, and elastic net retention.
  Source:     MIRAS (2504.13173) §3.2, Proposition 3.2; HOPE (2512.24695) §2 Eq 3
```

## Three Viewpoints on Memory Updates

The MIRAS framework identifies three equivalent ways to derive memory rules.
FTRL is viewpoint #2:

```text
Viewpoint 1: Online GD (MIRAS §3.1)
  W_t = W_{t-1} - eta_t * grad l(W_{t-1}; k_t, v_t)
  -- One gradient step per token. Simplest. Our current default.

Viewpoint 2: FTRL (MIRAS §3.2)
  W_t = argmin_W  sum_{i=1}^t l_tilde_i(W) + (1/eta_t) * R_t(W)
  -- Global optimization over all past losses + regularizer.

Viewpoint 3: Learning-Retaining (MIRAS §3.3)
  W_t = argmin_W  l_tilde_t(W; k_t, v_t) + Ret_t(W, W_{t-1})
  -- One-step: learn from current token + retain past knowledge.
  -- Most general (subsumes FTRL under convexity, Prop 3.2).
```

## The FTRL Formulation

### Specific Form (GD as FTRL)

Standard gradient descent (Eq 5) is equivalent to FTRL with linear loss
approximations and quadratic regularization:

<!-- HADES: miras_equations/eq-007-ftrl-specific (§3.2); hope_equations/eq-003-ftrl-form (§2 Eq 3) -->
```text
W_t = argmin_W  sum_{i=1}^t <W - W_{i-1}, grad l(W_{i-1}; k_i, v_i)>
              + (1 / 2*eta) * ||W||^2

-- Component 1: sum of linearized losses (attentional bias)
--   Each term measures: "how well does W explain token i?"
--   Linear approximation → tractable sum

-- Component 2: quadratic regularizer (memory stability)
--   (1/2eta)||W||^2 penalizes large memory states
--   Prevents unbounded accumulation

-- Closed-form solution: W_t = -eta * sum_{i=1}^t grad l(W_{i-1}; k_i, v_i)
-- This IS gradient descent with accumulation.
```

### Generalized Form

Replace the linear loss approximation and quadratic regularizer with arbitrary
choices:

<!-- HADES: miras_equations/eq-vp-ftrl-general (§3.2) -->
```text
W_t = argmin_{W in W}  sum_{i=1}^t l_tilde_i(W; k_i, v_i)  +  (1/eta_t) * R_t(W)
                        |----- Attentional Bias -----|     |-- Memory Stability --|

-- l_tilde_i: any approximation of the true loss l(W; k_i, v_i)
--   l_tilde = linear approx → standard FTRL (Eq 7)
--   l_tilde = exact loss → full optimization (expensive but optimal)
--   l_tilde = L2 regression → DGD-style state-dependent updates

-- R_t(W): any regularization function
--   R = (1/2)||W||^2    → L2 decay (standard)
--   R = ||W||_1         → L1 sparsity
--   R = elastic net     → L2 + L1 (Eq 23)
--   R = KL divergence   → information-theoretic retention
--   R = Bregman div     → online mirror descent (most general)

-- eta_t: can be data-dependent (adaptive learning rate)
```

## FTRL ↔ Learning-Retaining Equivalence

MIRAS Proposition 3.2 proves that under mild convexity assumptions, the FTRL
and Learning-Retaining viewpoints produce identical update rules:

<!-- HADES: miras_definitions/prop-3-2-viewpoint-equivalence (§3.2, Proposition 3.2) -->
```text
PROPOSITION 3.2 (Viewpoint Equivalence):
  Let eta_t = eta (constant) and define:
    h_t(W) = sum_{i=1}^{t-1} l_tilde_i(W) + (1/eta) * R(W)

  Assume W = R^d and h_t is strictly convex.
  Let D_h be the Bregman divergence induced by h_t.

  Set Ret_t(W, W') = D_h(W, W') in the Learning-Retaining viewpoint.

  THEN: Learning-Retaining update = FTRL update.

-- Consequence: Learning-Retaining is the MORE GENERAL viewpoint.
-- It can express FTRL (under convexity) but also handles cases
-- where FTRL cannot (non-convex R, data-dependent retention).
-- This is why MIRAS derives most rules via Learning-Retaining.
```

## FTRL as Algorithm Knob

In the MIRAS 4-knob framework, FTRL is a value for the **Algorithm** knob:

```text
| Knob              | Value for FTRL                               |
|-------------------|----------------------------------------------|
| Memory Structure  | Any (matrix, MLP — FTRL is structure-agnostic)|
| Attentional Bias  | Any l_tilde (L2, dot-product, l_p, Huber)    |
| Retention         | Determined by R_t (L2, L1, elastic, Bregman) |
| Algorithm         | FTRL (global argmin over cumulative losses)   |
```

The key distinction from GD/DGD: FTRL solves a **global** optimization each
step (over all past losses), while GD takes a **local** step (one gradient).
When the closed-form exists (which it does for quadratic R), the cost is
identical — but the theoretical guarantees differ (FTRL has regret bounds).

## Elastic Net via FTRL

The most important practical application of the FTRL viewpoint: deriving the
elastic net retention mechanism as a special case.

<!-- HADES: miras_equations/eq-023-elastic-net-ftrl (§5.2 Eq 23) -->
```text
-- Elastic net regularizer:
R_t(W) = (1/eta) * ||W||^2  +  (1/alpha) * ||W||_1

-- FTRL with elastic net yields a two-step update (MIRAS Eq 23):
FUNCTION: ftrl_elastic_net_step(A: &mut Tensor, W: &mut Tensor,
                                 grad: &Tensor,
                                 eta: f32, alpha: f32) -> ()
  -- Step 1: Accumulate gradients (FTRL gradient sum)
  A = A - eta * grad

  -- Step 2: Soft thresholding (proximal operator for L1)
  W = soft_threshold(A, eta / alpha)

  -- soft_threshold(x, lambda) = sign(x) * max(|x| - lambda, 0)
  -- Elements below threshold → exactly zero (sparse memory)
  -- Elements above threshold → shrunk toward zero
```

This produces **sparse memory matrices** — entries that contribute little
are driven exactly to zero by the L1 term, while important associations
are preserved (shrunk but nonzero) by the L2 term. The sparsity pattern
is data-dependent and emerges from the optimization, not from pruning.

## Gradient Derivation (for tape integration)

The backward pass through FTRL depends on which closed-form is used:

### Quadratic R (standard FTRL = accumulated GD)

<!-- HADES: Derived from miras_equations/eq-007-ftrl-specific (§3.2), analytical VJP -->
```text
-- Forward: W_t = W_{t-1} - eta * grad_t  (when solved in closed form)
-- This is identical to GD — same backward pass as existing implementation.
-- No new VJP adapter needed.
```

### Elastic Net FTRL

<!-- HADES: Derived from miras_equations/eq-023-elastic-net-ftrl (§5.2 Eq 23), analytical VJP -->
```text
-- Forward:
--   A_t = A_{t-1} - eta * grad_t
--   W_t = soft_threshold(A_t, lambda)   where lambda = eta/alpha

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dA_{t-1}, dL/dgrad_t

-- Soft thresholding gradient:
--   d(soft_threshold)/dA = indicator(|A| > lambda)
--   i.e., gradient passes through for active (nonzero) entries,
--   is zeroed for entries killed by thresholding.

dL/dA_t = dL/dW_t * indicator(|A_t| > lambda)
dL/dA_{t-1} = dL/dA_t                        -- accumulator passthrough
dL/dgrad_t = -eta * dL/dA_t                  -- scale by learning rate
```

The soft thresholding has a **discontinuous gradient** at |A| = lambda.
In practice, this is handled by treating the indicator as a straight-through
estimator (gradient is 1 for active entries, 0 for zeroed entries).

## Parallelization

FTRL with quadratic R reduces to accumulated GD → same parallelization as
existing GD (associative scan for dot-product objective, chunkwise for L2).

FTRL with elastic net adds a pointwise soft thresholding step after the
gradient accumulation — this is embarrassingly parallel (element-wise).

```text
-- Chunkwise FTRL (elastic net):
-- For each chunk of size C:
--   1. Accumulate gradients A within chunk (parallel or sequential)
--   2. Apply soft_threshold(A, lambda) — element-wise, fully parallel
--   3. Pass A and W to next chunk
```

## Comparison: Algorithm Knob Values

```text
| Algorithm | Inner Objective    | Solve Method     | State-Dep | Parallelization |
|-----------|--------------------|------------------|-----------|-----------------|
| GD        | Dot-product        | One grad step    | No        | Assoc scan      |
| DGD       | L2 regression      | One grad step    | Yes (M@k) | Chunkwise       |
| FTRL      | Cumulative losses  | Global argmin    | Yes (sum) | Assoc scan/CW   |
| DMGD      | L2 (on momentum)   | One grad step    | Yes (S@g) | Chunkwise       |
```

FTRL is the theoretical umbrella. GD and DGD are special cases when the
FTRL argmin has a closed-form that reduces to a single gradient step.
The FTRL viewpoint adds value when: (1) the regularizer R induces
structure (sparsity from L1, bounded norms from constraints), or
(2) regret analysis is needed for theoretical guarantees.

## Implementation Notes

1. **Start with elastic net**: The primary practical value of FTRL for our
   codebase is enabling elastic net retention (S3b-S8). The FTRL machinery
   itself (gradient accumulator A + soft thresholding) is simple — the spec
   for elastic net retention will detail the implementation.

2. **Accumulator state**: FTRL requires an additional state tensor A (the
   gradient accumulator) alongside the memory W. This is an `inner_loop_state`
   (same lifetime as M and S). Size: same as W (d x d for matrix memory).

3. **Soft thresholding stability**: The threshold lambda = eta/alpha must be
   positive. When alpha → infinity (no L1), lambda → 0 and soft thresholding
   becomes identity (recovers pure L2). When eta → 0, lambda → 0 similarly.
   Both limits are numerically safe.

4. **Connection to existing code**: The elastic net retention in S3b-S8 will
   use this FTRL accumulate-then-threshold pattern. The current L2 decay
   retention (`(1-alpha) * M`) is the special case where R = quadratic only.

## Axiom Compliance

- **NL IS #4** (compressing context): FTRL compresses the full gradient history into a regularized optimum.
- **NL IS #6** (optimizers are associative memory): FTRL makes explicit that the memory update is an optimization problem, not just a heuristic.
- **NL IS #9** (principled not ad hoc): FTRL provides regret bounds — the memory rule has provable online learning guarantees.
