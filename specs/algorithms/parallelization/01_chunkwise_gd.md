# Chunkwise GD (Freeze State at Boundary)

```
CONTRACT
  Purpose:    Most universal parallelization strategy. Freeze memory state
              at chunk boundary, compute ALL gradients in the chunk simultaneously
              w.r.t. the frozen state. The approximation: gradients use the
              chunk-START state, not the rolling state.
  Expects:    Any MemoryUpdateRule with differentiable WRITE.
              Chunk of C tokens. State at chunk boundary.
  Guarantees: Outputs for all C tokens. New state at chunk boundary.
              Approximation error bounded by O(C * lr * ||grad||).
              Fully parallel within chunk: C gradient computations at once.
  Cost:       O(C * per_token_cost) total, but parallel across C.
              Wall-clock: O(per_token_cost) if C fits in parallel hardware.
  Trade-off:  Larger C = more parallelism = worse approximation.
              C=1 recovers exact sequential form.
              C=T processes entire sequence in one chunk (maximally approximate).
  Position:   specs/algorithms/parallelization/01_chunkwise_gd.md
  Source:     Titans (2501.00663) Eqs 16-17, Atlas (2505.23735) Eqs 15-16
```

## The Approximation

```
-- Exact sequential:
FOR t = 0 to C-1:
  grad_t = gradient(loss(state_{t-1}, x_t))     -- grad uses ROLLING state
  state_t = state_{t-1} + update(grad_t)

-- Approximate parallel:
state_boundary = state at chunk start
grad_all = gradient(loss(state_boundary, x_all)) -- ALL grads use BOUNDARY state
state_new = accumulate(state_boundary, grad_all) -- accumulate in one pass

-- The error: grad_t should use state_{t-1} but uses state_boundary instead.
-- For small learning rates and short chunks, this error is bounded:
-- ||grad_exact - grad_approx|| <= C * lr * L * ||grad||
-- where L is the Lipschitz constant of the gradient operator.
```

## Pseudocode

```
ALGORITHM: chunkwise_gd_forward(state_boundary: &Tensor, chunk: &[Tensor],
                                 rule: &dyn MemoryUpdateRule,
                                 pulse: &Pulse) -> (Vec<Tensor>, Tensor)
  C = chunk.len()

  -- Step 1: Compute ALL keys, values, queries at once
  -- These don't depend on state — only on input and outer params
  keys   = [rule.project_key(x) for x in chunk]     -- C vectors, parallel
  values = [rule.project_value(x) for x in chunk]   -- C vectors, parallel
  queries = [rule.project_query(x) for x in chunk]  -- C vectors, parallel

  -- Step 2: Compute ALL gates at once
  gates = [rule.compute_gates(k, v) for (k, v) in zip(keys, values)]  -- parallel

  -- Step 3: Compute ALL gradients w.r.t. BOUNDARY state
  -- THIS IS THE APPROXIMATION: all use state_boundary, not rolling state
  grads = [rule.compute_inner_gradient(state_boundary, k, v)
           for (k, v) in zip(keys, values)]          -- parallel

  -- Step 4: Compute decay products for accumulation
  alphas = [g.alpha for g in gates]
  beta = compute_cumulative_decay(alphas)            -- (C, C) matrix

  -- Step 5: Accumulate updates using decay products
  -- state_new = decay^C * state_boundary + sum_t(decay^(C-t) * S_t)
  state_new = state_boundary
  FOR t = 0 to C-1:
    S_t = gates[t].eta * momentum_at_boundary - gates[t].theta * grads[t]
    state_new = (1 - alphas[t]) * state_new + S_t

  -- Step 6: Compute outputs using boundary state (approximate)
  outputs = [state_boundary @ q for q in queries]    -- parallel

  return (outputs, state_new)

FUNCTION: compute_cumulative_decay(alphas: &[f32]) -> Tensor
  -- beta[i][j] = product of (1-alpha_t) for t = j+1 to i
  -- Used for accumulating updates across the chunk
  C = alphas.len()
  beta = identity(C)
  FOR i = 1 to C-1:
    FOR j = 0 to i-1:
      beta[i][j] = beta[i-1][j] * (1 - alphas[i])
  return beta
```

## Compatibility

```
SUPPORTED_BY:
  - TitansLMM:    YES (matrix memory, L2 gradient, straightforward)
  - DeltaRule:     YES (simpler — no momentum state)
  - Hebbian:       YES (trivially — no gradient to approximate)
  - MONETA:        YES (MLP gradient computed at boundary)
  - YAAD:          YES (same as MONETA)
  - MEMORA:        YES (KL update at boundary)
  - LatticeOSR:    YES (orthogonal update at boundary)
  - Trellis:       YES (two-pass at boundary)
```

## Axiom Compliance

- **TNT IS #7** (parallel within chunks): This IS chunk-wise parallelism
- **NL IS #2** (parallel optimization): C inner-loop steps run simultaneously
