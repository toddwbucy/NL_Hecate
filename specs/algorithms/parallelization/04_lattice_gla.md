# Lattice GLA (Linearized Parallel Form)

```
CONTRACT
  Purpose:    Linearize Lattice/Trellis memory updates into a form compatible
              with Gated Linear Attention (GLA) parallelization. The orthogonal
              update and normalization are approximated as linear operations
              over a chunk, enabling matmul-based parallel processing.
  Expects:    Lattice OSR or Trellis memory rule. Chunk of C tokens.
              State at chunk boundary. Gating scalars.
  Guarantees: Parallel processing within chunk via GLA formulation.
              Approximation quality controlled by C (smaller = better).
              C=4 approximate matches C=1 exact (Lattice empirical finding).
  Cost:       O(C * m * d) total, parallel across C via matrix operations.
              Uses cumulative product matrices (same as chunkwise GD).
  Trade-off:  The linearization approximation is TIGHTER than chunkwise GD
              for Lattice/Trellis because the orthogonal update has bounded
              step size (unit sphere constraint limits drift). But it requires
              Lattice-specific algebraic structure to derive.
  Position:   specs/algorithms/parallelization/04_lattice_gla.md
  Source:     Lattice (2504.05646) Eqs 15-17, Appendix B
```

## Linearization of OSR

```
-- Exact sequential (Lattice Eq 9-10):
s_new = normalize(s + beta * orthogonal_project(delta_s, s))

-- Linearized approximation:
-- Factor 1: Drop the normalization for tokens WITHIN the chunk
-- Factor 2: Compute orthogonal projection w.r.t. boundary state
-- Factor 3: Apply normalization only at chunk boundaries

-- This turns the update into:
s_t = (1 - beta_t * (s_boundary^T @ delta_s_t)) * s_boundary + beta_t * delta_s_t
-- Which is a LINEAR recurrence in s_t when s_boundary is frozen
```

## Pseudocode

```
ALGORITHM: lattice_gla_forward(state_boundary: &Tensor, chunk: &[Tensor],
                                rule: &LatticeOSR, pulse: &Pulse) -> (Vec<Tensor>, Tensor)
  C = chunk.len()
  m = state_boundary.shape[0]  -- number of memory slots

  -- Step 1: Compute all updates w.r.t. boundary state (parallel)
  FOR i = 0 to m-1:
    s_i = state_boundary[i]     -- boundary state for slot i (unit vector)

    FOR t = 0 to C-1:           -- all parallel
      k_t = rule.project_key(chunk[t])
      v_t = rule.project_value(chunk[t])

      -- Score and gate w.r.t. boundary
      score = dot(s_i, k_t)
      gate = sigmoid(score)

      -- Linearized update coefficients
      a[i][t] = 1 - gate * dot(s_i, v_t)    -- decay (from orthogonal projection)
      b[i][t] = gate * v_t                    -- update direction

  -- Step 2: Cumulative products for parallel accumulation
  -- Same structure as chunkwise GD but with Lattice-specific a, b
  FOR i = 0 to m-1:
    beta_matrix = compute_cumulative_products(a[i])

    -- Accumulate: s_new = prod(a) * s_boundary + sum(prod(a_remaining) * b)
    state_new[i] = accumulate_linear(s_i, beta_matrix, b[i])

    -- Normalize at chunk boundary only
    state_new[i] = state_new[i] / norm(state_new[i])

  -- Step 3: Compute outputs (parallel)
  outputs = [lattice_read(state_boundary, rule.project_query(chunk[t]))
             for t in 0..C]

  return (outputs, state_new)
```

## Key Result: C=4 Matches C=1

Lattice's empirical finding: using C=4 (process 4 tokens at a time, approximate)
produces results nearly identical to C=1 (exact sequential). This is because:

1. The unit sphere constraint bounds how far state can drift in C steps
2. Orthogonal updates are inherently small (only the novel component)
3. The normalization at boundaries corrects accumulated drift

This means 4x parallelism with negligible quality loss.

## Compatibility

```
SUPPORTED_BY:
  - LatticeOSR:    YES  (designed for this)
  - Trellis:        YES  (same linearization applies to two-pass)

NOT SUPPORTED:
  - TitansLMM, DeltaRule, Hebbian: Use chunkwise GD instead
  - MONETA, YAAD, MEMORA: Use chunkwise GD instead
```

## Axiom Compliance

- **Lattice IS #8** (memory efficiency): GLA form is hardware-efficient
- **NL IS #2** (parallel): C tokens processed simultaneously
