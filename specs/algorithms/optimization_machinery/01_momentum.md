# Momentum as Associative Memory

```
CONTRACT
  Purpose:    Momentum is NOT an optimizer trick — it IS an associative memory
              that compresses past gradients. This reframing is foundational to
              the NL program. The hierarchy from EMA to Muon to Atlas represents
              increasing memory capacity, not increasing optimizer complexity.
  Expects:    Gradient signal, momentum state, data-dependent gates.
  Guarantees: Compressed gradient history. Temporal smoothing.
              Standard EMA is the closed-form solution of an associative memory
              objective (HOPE Eq 13). This is proven, not a metaphor.
  Cost:       Varies: O(d) for EMA to O(d^2 * k_ns) for Muon (Newton-Schulz).
  Trade-off:  More expressive momentum = higher capacity = more compute.
              EMA is cheapest but compresses via forgetting (oldest gradients decay).
              Muon preserves gradient direction information via orthogonalization.
  Position:   specs/algorithms/optimization_machinery/01_momentum.md
  Source:     HOPE Eqs 10-13, 32-53; Titans Eqs 9-10, 14, 18
```

## The Reframing

```
-- Standard framing:
m_t = beta * m_{t-1} + (1-beta) * g_t      -- "exponential moving average"

-- NL reframing (HOPE Eq 13):
m_t = argmin_m  sum_{i<=t} ||m * g_{i+1} - P_i||^2 + lambda * ||m||^2
-- "find the momentum that best maps gradients to global properties"

-- These are the SAME MATH. The EMA form is the closed-form solution
-- of the associative memory objective.
```

## The Hierarchy

From simplest to most expressive. Each level adds capacity.

```
No momentum (GD)
  S_t = -theta_t * grad_t
  |
  v
EMA momentum (HOPE Eq 33)
  S_t = eta_t * S_{t-1} + theta_t * grad_t
  |
  v
Hebbian momentum (HOPE Eq 34)
  S_t = S_{t-1} + grad_t @ (global_property)^T    -- dot-product objective
  |
  v
Delta momentum (HOPE Eq 49)
  S_t = eta * S_{t-1} + theta * grad_t, V_t = update(V_{t-1}, S_t)
  -- L2 regression with explicit preconditioner
  |
  v
Deep momentum (HOPE Eq 50)
  S_t = MLP(S_{t-1}, grad_t)    -- nonlinear compression
  |
  v
Preconditioned momentum (HOPE Eq 46)
  S_t, V_t updated jointly, V preconditions the update direction
  |
  v
Muon (HOPE Eq 42, Atlas Eq 32)
  S_t = eta * S_{t-1} + theta * grad_t
  X = Newton_Schulz_5(S_t)    -- orthogonalize momentum
  -- Preserves gradient DIRECTION, not just magnitude
  |
  v
Atlas Omega (Atlas Eqs 32-33)
  S_t = eta * S_{t-1} + theta * omega_t    -- omega is state-independent
  X = Newton_Schulz_5(S_t)
  -- Fully parallel momentum + orthogonalization
```

## Surprise Decomposition (Titans Eq 10)

```
FUNCTION: surprise_decomposition(S_prev: &Tensor, grad_t: &Tensor,
                                  eta_t: f32, theta_t: f32) -> Tensor
  -- Surprise = past surprise (decayed) + momentary surprise (scaled)
  S_t = eta_t * S_prev + theta_t * grad_t

  -- eta_t ≈ 0: context changed, ignore past (reset momentum)
  -- eta_t ≈ 1: continuation, incorporate past
  -- theta_t scales current gradient contribution
  return S_t
```

## Newton-Schulz Orthogonalization (Muon)

```
FUNCTION: newton_schulz_5(S: &Tensor, k: usize) -> Tensor
  -- Map momentum to the nearest orthogonal matrix
  -- X_{k+1} = X_k * (3I - X_k^T @ X_k) / 2
  -- Converges in ~5 iterations for well-conditioned input

  X = S / max(norm(S), eps)     -- initial normalization
  FOR iter = 0 to k-1:
    X = 0.5 * X @ (3 * I - X^T @ X)
  return X

  -- Why orthogonalize? Gradient magnitude varies wildly across dimensions.
  -- SGD follows the noisy gradient. Adam normalizes per-dimension.
  -- Muon normalizes the FULL matrix, preserving directional information.
  -- This is why Muon trains faster on language tasks — it follows the
  -- gradient DIRECTION, not the noisy magnitude.
```

## Connection to Standard Optimizers

HOPE Eq 111 proves ALL standard optimizers are associative memories:

```
AdaGrad  = associative memory + diagonal preconditioner
Adam     = AdaGrad + EMA momentum
Muon     = Adam + Newton-Schulz (full matrix instead of diagonal)
M3       = Muon + CMS frequency scheduling (multi-scale)
```

## Axiom Compliance

- **NL IS #6** (optimizers are associative memory): THIS IS the axiom.
- **NL IS #4** (compressing context): Momentum compresses gradient history.
- **NL IS #7** (self-modifying): Data-dependent gates modify accumulation behavior.
