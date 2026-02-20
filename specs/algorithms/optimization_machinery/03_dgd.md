# Delta Gradient Descent (DGD)

```
CONTRACT
  Purpose:    DGD is to GD what the Delta rule is to Hebbian — it upgrades the
              inner-loop optimizer from state-independent to state-dependent by
              swapping the dot-product objective for L2 regression. The update now
              depends on what the memory has ALREADY learned, not just the current
              input. This is the core inner-loop optimizer for the Hope architecture.
  Expects:    Memory state M, input key k, target value v, gates (alpha, theta).
              Inputs assumed normalized (||k||_2 = phi).
  Guarantees: State-dependent update. Adaptive, directional weight decay emerges
              naturally from the L2 objective. The decay selectively forgets along
              the current input direction, preserving orthogonal information.
              Closed-form via Sherman-Morrison (no iterative solve).
  Cost:       Per token: O(d^2) — same asymptotic cost as standard GD.
              One additional matrix-vector product (M @ k) for the state-dependent
              term. Negligible compared to the outer product in the update.
  Trade-off:  More expressive than GD (captures cross-sample dependencies) but
              cannot use associative scan parallelization (nonlinear in M).
              Relies on chunkwise GD parallelization instead.
  Position:   specs/algorithms/optimization_machinery/03_dgd.md
              Sibling of 01_momentum.md (DGD can compose with momentum)
  Source:     HOPE (2512.24695) §4.5, Appendix C; Table 6 ablation
```

## The Problem DGD Solves

Standard GD, viewed as an associative memory, uses a dot-product objective:

```
L_inner(M; k_t, v_t) = -<M k_t, v_t>

-- Gradient: nabla_M L = -v_t @ k_t^T
-- Update:   M_{t+1} = M_t + theta_t * v_t @ k_t^T
```

This update is **independent of M_t** — it adds the same outer product regardless
of what M already knows. For i.i.d. data this suffices. For sequences where tokens
are highly correlated, it wastes capacity by re-storing information M already has.

## DGD: The L2 Regression Upgrade

Replace dot-product similarity with L2 regression:

```
L_inner(M; k_t, v_t) = ||M k_t - v_t||^2_2

-- Gradient: nabla_M L = 2 * (M_t k_t - v_t) @ k_t^T
-- This gradient DEPENDS ON M_t via the (M_t k_t) term.
```

The DGD update with retention becomes:

```
FUNCTION: dgd_step(M: &mut Tensor, k: &Tensor, v: &Tensor,
                    alpha_t: f32, theta_t: f32) -> ()
  -- alpha_t: retention gate (data-dependent, sigmoid)
  -- theta_t: learning rate (data-dependent, softplus)
  -- HOPE §4.5 closed-form via Sherman-Morrison

  -- State-dependent gradient
  error = M @ k - v                          -- [d] what M predicts minus target
  grad = error @ k^T                         -- [d, d] outer product

  -- Combined update: retention + DGD
  -- Equivalent to: M_{t+1} = (1 - alpha_t) * M_t - theta_t * grad
  -- But the theta_t here is the effective learning rate eta'_t from Appendix C
  M = (1 - alpha_t) * M - theta_t * grad
```

## Closed-Form Derivation (HOPE Appendix C)

The proximal viewpoint derives DGD as the exact solution to:

```
M_{t+1} = argmin_M { ||M k_t - v_t||^2 + eta_t^{-1} ||M - M_t||^2_F }
```

Taking the gradient and setting to zero:

```
(k_t k_t^T + eta_t^{-1} I) (M_{t+1} - M_t) = -(M_t k_t - v_t) k_t^T
```

By Sherman-Morrison (assuming ||k_t|| = phi):

```
eta'_t = eta_t / (1 + eta_t)

M_{t+1} = (1 - eta'_t * k_t k_t^T) M_t + eta'_t * v_t @ k_t^T
```

This reveals DGD's structure: **directional decay** `(1 - eta'_t * k_t k_t^T)` applied
to M, plus Hebbian-style write. The decay only acts along the `k_t` direction.

## Comparison Table

```
| Property             | GD (dot-product)        | DGD (L2 regression)              |
|----------------------|-------------------------|----------------------------------|
| Inner objective      | -<M k, v>               | ||M k - v||^2                    |
| State dependence     | None                    | Yes (M_t k_t term)              |
| Gradient             | -v @ k^T                | 2(M k - v) @ k^T                |
| Implicit decay       | None                    | Directional: eta' * k k^T * M   |
| Parallelization      | Associative scan OK     | Chunkwise GD only (nonlinear)    |
| Sequence analogy     | Hebbian rule            | Delta rule                       |
| Optimizer analogy    | SGD                     | "DeltaSGD"                       |
```

## DGD as a MIRAS Algorithm Knob

DGD fits the MIRAS 4-knob framework as a new value for the **Algorithm** knob:

```
| Knob               | Value                                    |
|---------------------|------------------------------------------|
| Memory Structure    | Any (matrix, MLP — DGD is structure-agnostic) |
| Attentional Bias    | L2 regression (defining property of DGD) |
| Retention           | Any (L2 decay, KL, elastic net, etc.)    |
| Algorithm           | DGD (gradient descent with L2 inner obj) |
```

Note the subtle point: the "attentional bias" and "algorithm" knobs conflate for DGD
because the algorithm IS defined by the choice of inner objective. DGD = GD with L2
objective. Standard GD = GD with dot-product objective. The algorithm knob is really
choosing the (objective, solver) pair.

## Interaction with Momentum

DGD composes with momentum (the **Delta Momentum** variant, HOPE §4.4):

```
FUNCTION: dgd_momentum_step(M: &mut Tensor, S: &mut Tensor,
                             k: &Tensor, v: &Tensor,
                             alpha_t: f32, theta_t: f32, eta_t: f32) -> ()
  -- DGD for the memory update
  error = M @ k - v
  grad = error @ k^T

  -- Momentum accumulation (surprise decomposition, Titans Eq 10)
  S = eta_t * S + theta_t * grad

  -- Apply accumulated surprise with retention
  M = (1 - alpha_t) * M - S
```

The momentum term S can itself use DGD-style state-dependence (DMGD, HOPE Eq 33+),
but that is specified separately in `04_dmgd.md`.

## Gradient Derivation (for tape integration)

The backward pass through `dgd_step` requires gradients of loss w.r.t. the
outer-loop parameters (W_K, W_V, W_Q, gate params) that produce k, v, alpha, theta.

```
-- Forward: M_{t+1} = (1 - alpha_t) * M_t - theta_t * (M_t @ k_t - v_t) @ k_t^T
-- Let E_t = M_t @ k_t - v_t (the prediction error)

-- Given: dL/dM_{t+1} (upstream gradient from later tokens)
-- Need: dL/dM_t, dL/dk_t, dL/dv_t, dL/dalpha_t, dL/dtheta_t

dL/dM_t = (1 - alpha_t) * dL/dM_{t+1}
         - theta_t * dL/dM_{t+1} @ (k_t @ k_t^T)
         -- ^^^ state-dependent term: gradient flows through M_t @ k_t

dL/dk_t = -theta_t * (M_t^T @ dL/dM_{t+1} @ k_t  +  E_t^T @ dL/dM_{t+1})
         -- Two terms: one through M_t@k_t, one through the outer product

dL/dv_t = theta_t * dL/dM_{t+1} @ k_t
         -- v_t appears with negative sign in error, so positive here

dL/dalpha_t = -trace(M_t^T @ dL/dM_{t+1})
             -- Scalar: how much the retention gate affects loss

dL/dtheta_t = -trace((E_t @ k_t^T)^T @ dL/dM_{t+1})
             -- Scalar: how much the learning rate gate affects loss
```

These are the analytical gradients that the opaque VJP adapter must implement.
The tape records the forward pass; the adapter provides these backward equations.

## Chunkwise Parallelization

DGD is nonlinear in M (the `M @ k` term), so associative scan is inapplicable.
Use chunkwise GD (existing `chunkwise_gd.rs` infrastructure):

```
-- Split sequence into chunks of size C
-- For each chunk:
--   1. Freeze M at chunk boundary (M_chunk_start)
--   2. Compute all k, v, gates in parallel (they don't depend on M)
--   3. Sequential DGD recurrence within the chunk
--   4. Pass final M to next chunk
```

This is identical to how existing chunkwise GD works for the Delta rule.
No new parallelization strategy is needed — DGD slots into the existing
chunkwise framework because it shares the same nonlinear structure.

## Ablation Evidence (HOPE Table 6)

```
| Configuration      | Perplexity | Reasoning Accuracy |
|--------------------|-----------|-------------------|
| Hope (full, DGD)   | 12.24     | 58.1%             |
| w/o DGD (plain GD) | 13.41     | 56.5%             |
| Delta              | +1.17 ppl | -1.6 points       |
```

DGD's impact is comparable to removing momentum (+1.34 ppl) and larger than
removing CMS (+0.80 ppl) or weight decay (+1.47 ppl). It is one of the most
impactful single components in Hope.

## Implementation Notes

1. **Stability**: The `M @ k` product can grow unbounded if alpha clamping is
   insufficient. CS-39 documents a historical DGD stability issue traced to
   unclamped learnable decay. Always clamp alpha_t to [eps, 1-eps].

2. **Normalization**: The Sherman-Morrison derivation assumes `||k|| = phi`.
   In practice, layer-norm on the input ensures this. If inputs are not
   normalized, the effective learning rate changes.

3. **Cost**: One extra matrix-vector product (`M @ k`) per token compared to
   GD. For d=2048 this is 4M FLOPs — negligible next to the d^2 outer product.

4. **Existing code reuse**: The Delta rule in `core/src/delta.rs` already
   implements `M_{t+1} = (1-alpha) M_t - theta * (M k - v) k^T`. DGD for the
   inner-loop optimizer IS this same recurrence applied at the weight-update
   level rather than the sequence-memory level.

## Axiom Compliance

- **NL IS #4** (compressing context): DGD compresses gradient history with state awareness.
- **NL IS #6** (optimizers are associative memory): DGD upgrades the memory's objective.
- **NL IS #7** (self-modifying): The M-dependent term means the optimizer adapts
  based on what it has already learned.
