# Implicit Gradient Descent (Proximal / Closed-Form Updates)

```text
CONTRACT
  Purpose:    Implicit GD solves the memory update problem exactly rather than
              taking an explicit gradient step. Where explicit GD approximates
              the Learning-Retaining argmin with one gradient step, implicit GD
              computes the exact solution (when a closed form exists). This
              produces "locally optimal" memory updates — the best single-step
              change given the current token and retention constraint.
  Expects:    Loss function l(W; k_t, v_t), retention function Ret_t(W, W_{t-1}),
              both assumed convex with tractable structure (e.g., quadratic).
  Guarantees: Exact minimizer of the per-token learning-retaining objective.
              No approximation error from finite step size. The effective
              learning rate emerges from the solution rather than being a
              hyperparameter. Directional operations (decay, write) are
              optimally balanced.
  Cost:       Depends on the closed form. For L2 + quadratic retention:
              O(d^2) via Sherman-Morrison (same as explicit GD). For general
              convex problems: iterative solver needed (e.g., Newton-Schulz).
  Trade-off:  Eliminates step-size tuning but restricts to problems where the
              argmin is tractable. General nonlinear losses or non-convex
              retention have no closed form — these require explicit GD or
              approximate solvers like Newton-Schulz (S3b-S5).
  Position:   specs/algorithms/optimization_machinery/06_implicit_gd.md
              Sibling of 03_dgd.md (DGD is implicit GD with L2 + quadratic Ret),
              05_ftrl.md (FTRL is the cumulative version), 07_newton_schulz.md
  Source:     MIRAS (2504.13173) §3.3 Learning-Retaining Viewpoint;
              HOPE (2512.24695) Appendix C (Sherman-Morrison);
              Atlas (2505.23735) Table 1 ("Locally Optimal" characteristic)
```

## Explicit vs Implicit: The Core Distinction

Every memory update rule in the MIRAS framework can be viewed as solving:

<!-- HADES: miras_equations/eq-vp-learning-retaining (§3.3) -->
```text
W_t = argmin_{W}  l_tilde_t(W; k_t, v_t)  +  Ret_t(W, W_{t-1})
                   |-- learn new token --|     |-- retain old ----|
```

The **Algorithm knob** determines HOW this argmin is solved:

```text
| Method           | How it solves the argmin              | Quality    |
|------------------|---------------------------------------|------------|
| Explicit GD      | One gradient step at W_{t-1}          | Approximate|
| Explicit DGD     | One gradient step (state-dependent)   | Better     |
| FTRL             | Cumulative linearized losses + argmin | Exact*     |
| Implicit GD      | Exact solution of per-token argmin    | Exact      |
| Newton-Schulz    | Iterative second-order approximation  | Near-exact |
```

(*FTRL is exact for the linearized problem, not the original problem)

## The Implicit Update: General Form

When l_tilde and Ret are both differentiable and the combined objective is
strictly convex, the argmin has a unique solution characterized by:

<!-- HADES: miras_equations/eq-vp-learning-retaining (§3.3), first-order optimality -->
```text
-- First-order optimality condition:
nabla_W l_tilde_t(W_t; k_t, v_t) + nabla_W Ret_t(W_t, W_{t-1}) = 0

-- The implicit update IS the W_t that satisfies this equation.
-- For explicit GD, we linearize around W_{t-1} instead of solving exactly.
```

## Case 1: L2 Loss + Quadratic Retention (Sherman-Morrison)

The most important closed form. This IS what DGD computes:

<!-- HADES: hope_equations/eq-121-delta-gd-final (Appendix C Eq 121); miras_equations/eq-009-delta-rule (§4 Eq 9) -->
```text
-- Loss:      l(W; k, v) = ||W k - v||^2_2  (L2 regression)
-- Retention:  Ret(W, W') = (1/eta) * ||W - W'||^2_F  (quadratic proximity)

-- Combined objective:
W_t = argmin_W  ||W k_t - v_t||^2  +  (1/eta) * ||W - W_{t-1}||^2_F

-- Taking gradient, setting to zero:
(k_t k_t^T + (1/eta) I) * (W_t - W_{t-1}) = -(W_{t-1} k_t - v_t) k_t^T

-- By Sherman-Morrison (assuming ||k_t|| = phi):
eta' = eta / (1 + eta)
W_t = (I - eta' * k_t k_t^T) W_{t-1}  +  eta' * v_t k_t^T
      |---- directional decay ----|     |-- Hebbian write --|

-- The effective learning rate eta' is NOT a hyperparameter —
-- it emerges from the solution. eta' in (0, 1) always.
```

Key insight: the **directional decay** `(I - eta' k k^T)` acts only along
the input direction k_t. Information orthogonal to k_t is perfectly preserved.
This is strictly better than scalar decay `(1 - alpha) W` which decays
everything uniformly.

## Case 2: L2 Loss + L2 Decay Retention (Delta Rule)

When retention is the standard L2 weight decay:

<!-- HADES: miras_equations/eq-009-delta-rule (§4 Eq 9) -->
```text
-- Loss:      l(W; k, v) = ||W k - v||^2_2
-- Retention:  Ret(W, W') = ||W - alpha * W'||^2_F  (L2 decay)

-- Argmin solution:
W_t = alpha * (I - eta_t * k_t k_t^T) * W_{t-1}  +  eta_t * v_t k_t^T

-- This IS the Delta rule (MIRAS Eq 9).
-- The Delta rule is NOT just "a heuristic recurrence" — it is the
-- EXACT solution to L2 regression with L2 decay retention.
```

## Case 3: Dot-Product Loss + Quadratic Retention (Hebbian)

<!-- HADES: miras_equations/eq-008-hebbian-rule (§4 Eq 8) -->
```text
-- Loss:      l(W; k, v) = -2<W k, v>  (dot-product / Hebbian)
-- Retention:  Ret(W, W') = ||W - alpha * W'||^2_F

-- Gradient: nabla_W l = -2 v k^T  (independent of W — no implicit gain)
-- Argmin solution: W_t = alpha * W_{t-1} + v_t k_t^T

-- The Hebbian rule. Here implicit = explicit because the loss is
-- LINEAR in W. The argmin has the same solution as one gradient step.
-- This is why DGD (L2 loss, quadratic in W) gains from implicit
-- but Hebbian (dot-product, linear in W) does not.
```

## Case 4: L_p Loss (General)

For the general l_p attentional bias:

<!-- HADES: miras_equations/eq-011-lp-closed-form (§5.1 Eq 11) -->
```text
-- Loss: l(W; k, v) = ||W k - v||_p^p
-- Closed form depends on p:

p = 2:  Standard L2 → Sherman-Morrison (Case 1)
p = 1:  Sign-based update → sign(W k - v) k^T
        (value-less memory: maps entities to +/- direction)
p → inf: Max-norm → only the largest-error component updates

-- General p: the closed-form involves |W k - v|^{p-1} * sign(W k - v)
-- which specializes the gradient shape but preserves the implicit
-- structure when retention is quadratic.
```

## Case 5: Sliding Window (Omega Rule)

The Atlas Omega rule extends implicit GD to a window of past tokens:

<!-- HADES: atlas_equations/eq-009-omega-rule (§3.2 Eq 9) -->
```text
-- Omega objective (Atlas Eq 9):
W_t = argmin_W  sum_{i=t-c+1}^{t}  gamma_i * ||W k_i - v_i||^2_2

-- Window c >= 1:
--   c = 1: recovers per-token implicit GD (Delta rule / DGD)
--   c = context: global least squares (impractical)
--   c in between: local optimality over recent context

-- For linear memory with c > 1, the exact solution requires
-- inverting a (c x c) Gram matrix — O(c^3) per step.
-- Atlas avoids this cost by using momentum + Newton-Schulz
-- instead of exact inversion (S3b-S5).
```

## Atlas Table 1: "Locally Optimal" Characteristic

The Atlas paper classifies all recurrent memory models on five binary
characteristics. "Locally Optimal" means the model uses implicit or
second-order updates:

<!-- HADES: atlas_definitions/def-five-memory-characteristics (Table 1) -->
```text
| Characteristic        | Meaning                              | Implicit GD? |
|-----------------------|--------------------------------------|--------------|
| Dynamic Decay         | Data-dependent forgetting gate       | Orthogonal   |
| Deep Neural Memory    | MLP with L >= 2 layers               | Orthogonal   |
| Nonlinear Capacity    | Polynomial features boost capacity   | Orthogonal   |
| Locally Optimal       | Second-order / exact / proximal      | YES          |
| Flexible Context      | Window of tokens, not just current   | Orthogonal   |

-- Only Atlas achieves all 5/5.
-- Standard Attention achieves 3/5 (nonlinear, local opt, flexible).
-- Titans achieves 2/5 (dynamic decay + deep memory).
-- DGD (Sherman-Morrison) achieves local optimality for per-token case.
```

## Gradient Derivation (for tape integration)

Implicit GD's backward pass depends on the closed form used:

### Sherman-Morrison (L2 + quadratic Ret)

<!-- HADES: Derived from hope_equations/eq-121-delta-gd-final (Appendix C), same as DGD VJP -->
```text
-- Forward: W_t = (I - eta' k k^T) W_{t-1} + eta' v k^T
-- This is identical to the DGD forward — same backward pass.
-- See 03_dgd.md for the full analytical gradient derivation.
-- The opaque VJP adapter for DGD IS the implicit GD backward.
```

### Delta Rule (L2 + L2 decay)

```text
-- Forward: W_t = alpha * (I - eta k k^T) W_{t-1} + eta v k^T
-- Already implemented in core/src/delta.rs
-- Existing backward pass handles this case.
```

### General Case

```text
-- For general implicit updates W_t = argmin_W F(W; k_t, v_t, W_{t-1}):
-- The backward pass uses the Implicit Function Theorem:
--   nabla F(W_t) = 0  →  dW_t/dparam = -(nabla^2 F)^{-1} (nabla^2_{W,param} F)
-- This requires the Hessian of the combined objective.
-- For L2 + quadratic: Hessian = k k^T + (1/eta) I (constant, cheap).
-- For general: may need iterative Hessian-vector products.
```

## Implementation Notes

1. **DGD IS implicit GD**: The DGD spec (03_dgd.md) already documents the
   Sherman-Morrison closed form. This spec provides the theoretical framing
   that makes DGD's nature as an implicit solver explicit, and extends to
   other loss/retention combinations.

2. **No new Rust code for Case 1-2**: The Delta rule and DGD implementations
   already compute the implicit solution. This spec documents WHY they work,
   not a new algorithm to implement.

3. **New code needed for**: (a) l_p implicit updates with p != 2 (Case 4),
   which requires the sign-based gradient from MIRAS Eq 11; (b) sliding
   window implicit updates (Case 5, Omega rule), which requires multi-token
   gradient accumulation — covered by S3b-S18 (Atlas Omega spec).

4. **Connection to Newton-Schulz**: When the exact closed form is unavailable
   (nonlinear MLP memory, general loss), Newton-Schulz iterations approximate
   the implicit solution iteratively. This is why Atlas uses Muon — it's an
   approximate implicit solver. Spec S3b-S5 covers this.

5. **Locally optimal vs globally optimal**: Implicit GD is optimal for the
   current token (or window). It is NOT globally optimal across the full
   sequence — that would require solving the FTRL global objective (05_ftrl.md).
   The Omega rule with c > 1 interpolates between local and global optimality.

## Axiom Compliance

- **NL IS #6** (optimizers are associative memory): Implicit GD solves the associative memory objective exactly, making the connection maximally explicit.
- **NL IS #7** (self-modifying): The implicit solution depends on the current state W_{t-1}, making the update inherently self-referential.
- **NL IS #9** (principled not ad hoc): No hyperparameter approximation — the effective learning rate emerges from the optimization, not from tuning.
