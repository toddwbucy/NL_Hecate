# Atlas Parallel Momentum

```
CONTRACT
  Purpose:    Atlas achieves EXACT parallelism for momentum by making momentum
              independent of the current memory state. In standard Titans,
              momentum uses grad(M, x) — which depends on M, creating a
              sequential dependency. Atlas's Omega rule restructures this
              so momentum depends only on the OUTER-LOOP parameters (fixed
              during the inner loop), enabling full parallelism.
  Expects:    Atlas Omega rule memory. Chunk of C tokens.
              Outer-loop parameters (fixed within inner loop).
  Guarantees: EXACT momentum computation in parallel (no approximation).
              Memory update is still approximate (same as chunkwise GD).
              But momentum — the dominant cost — is fully parallel.
  Cost:       Momentum: O(C * d) total, O(1) depth (fully parallel).
              Memory: O(C * d^2) total, O(1) depth (parallel at boundary).
  Trade-off:  Exact momentum parallelism (better than Titans associative scan
              which is also exact but O(log C) depth). But requires the Omega
              rule structure — doesn't work for arbitrary memory rules.
  Position:   specs/algorithms/parallelization/05_atlas_parallel.md
  Source:     Atlas (2505.23735) Eqs 34-41, Section 5
```

## Why Atlas Momentum Is Special

```
-- Titans momentum: S_t = eta_t * S_{t-1} - theta_t * grad(M_{t-1}, x_t)
--   Sequential because grad depends on M_{t-1}, which depends on S_{t-1}

-- Atlas insight: restructure so momentum doesn't depend on M
-- Atlas Omega rule: S_t = eta_t * S_{t-1} - theta_t * omega(x_t, outer_params)
--   where omega uses ONLY outer_loop_params (fixed during inner loop)
--   S_t depends on S_{t-1} (linear recurrence) but NOT on M

-- Result: S can be computed for all t in parallel
-- Then M is updated using the pre-computed S values
```

## Pseudocode

```
ALGORITHM: atlas_parallel_forward(state_boundary: &Tensor,
                                   momentum_boundary: &Tensor,
                                   chunk: &[Tensor],
                                   outer: &AtlasParams,
                                   pulse: &Pulse) -> (Vec<Tensor>, Tensor, Tensor)
  C = chunk.len()

  -- Step 1: Compute ALL omega updates in parallel
  -- These depend only on input and outer_params (both fixed)
  omegas = PARALLEL_FOR t = 0 to C-1:
    yield atlas_omega(chunk[t], outer)     -- O(d) each, fully parallel

  -- Step 2: Compute ALL gates in parallel
  gates = PARALLEL_FOR t = 0 to C-1:
    k = chunk[t] @ outer.W_K^T
    v = chunk[t] @ outer.W_V^T
    yield compute_gates(k, v, outer.gate_params)

  -- Step 3: Parallel momentum via associative scan
  -- S_t = eta_t * S_{t-1} - theta_t * omega_t
  -- This is a linear recurrence (a = eta_t, b = -theta_t * omega_t)
  a_seq = [g.eta for g in gates]
  b_seq = [-g.theta * omega for (g, omega) in zip(gates, omegas)]
  S_all = associative_scan(a_seq, b_seq, momentum_boundary)  -- O(log C)

  -- Step 4: Accumulate memory update (parallel, approximate)
  -- M uses pre-computed S values — no sequential dependency on M
  alphas = [g.alpha for g in gates]
  decay_matrix = compute_cumulative_products(alphas)

  M_new = state_boundary
  FOR t = 0 to C-1:
    M_new = (1 - alphas[t]) * M_new + S_all[t]

  -- Step 5: Compute outputs (parallel)
  outputs = PARALLEL_FOR t = 0 to C-1:
    q_t = chunk[t] @ outer.W_Q^T
    yield state_boundary @ q_t   -- approximate: uses boundary state

  return (outputs, M_new, S_all[C-1])
```

## The Omega Rule

```
FUNCTION: atlas_omega(x: &Tensor, outer: &AtlasParams) -> Tensor
  -- Atlas Eq 9: omega computes a "virtual gradient" from the input
  -- WITHOUT referencing the current memory state
  --
  -- In Titans: grad = d/dM ||M@k - v||^2 = 2*(M@k-v)@k^T  (depends on M)
  -- In Atlas:  omega = f(k, v, outer_params)                 (depends on outer_params only)
  --
  -- The outer params learn to approximate what the gradient WOULD be
  -- This is the Omega rule's key insight: replace state-dependent gradient
  -- with a learned, state-independent surrogate

  k = x @ outer.W_K_omega^T
  v = x @ outer.W_V_omega^T
  omega = outer.W_omega @ silu(concat(k, v))    -- learned surrogate
  return omega
```

## Polynomial Extension (Atlas Eqs 19-25)

Atlas can extend to polynomial capacity O(d_k^p) via polynomial kernels:

```
-- Standard (c=1): linear capacity O(d_k)
-- Polynomial (c>1): capacity O(d_k^c)
-- Atlas Eq 22: polynomial feature map phi_c(k) = k^{\otimes c}
-- CAUTION: polynomial terms vanish in fp16 for c>=3
--          10B@fp32 beats 100B@fp8 (parameter efficiency)
```

## Compatibility

```
SUPPORTED_BY:
  - Atlas Omega Rule: YES  (designed for this)

NOT SUPPORTED:
  - TitansLMM: momentum depends on M (use associative scan instead)
  - Any rule where grad depends on state
```

## Axiom Compliance

- **Atlas IS #4** (strict Transformer generalization): Proven by Eq 26
- **Atlas IS #6** (composable): Momentum, memory, attention are independent modules
- **NL IS #2** (parallel): Momentum is fully parallel within chunks
