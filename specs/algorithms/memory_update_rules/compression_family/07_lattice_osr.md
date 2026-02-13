# Lattice OSR (Orthogonal State Recurrence)

```
CONTRACT
  Purpose:    Memory compression via orthogonal projection onto unit sphere.
              Each memory slot is updated ONLY with information orthogonal
              to its current state. Proposition 3.1 proves this is Riemannian
              gradient descent on the unit sphere manifold.
  Expects:    Input token embedding x_t. Pulse.
              Memory slots S on the unit sphere (||S[i]|| = 1).
              Number of memory slots m (a compression hyperparameter).
  Guarantees: Output y_t. Memory slots remain on unit sphere after update.
              Normalization acts as implicit forgetting (no explicit decay gate).
              m slots compress arbitrarily long context.
  Cost:       Per token: O(m * d^2) for m slots of dimension d.
              Memory: O(m * d) — m unit vectors of dimension d.
              Key result: m=16 beats full-memory baselines (quarter memory).
  Trade-off:  Extreme compression (m << d) vs information loss.
              No explicit forgetting gate — normalization IS forgetting.
              Works best for compression-heavy tasks (long context, small memory).
              Less expressive than Titans LMM for short-context exact recall.
  Position:   specs/algorithms/memory_update_rules/compression_family/07_lattice_osr.md
              Child of memory_update_rules/00_interface.md
  Source:     Lattice (2504.05646) Eqs 5-10, Proposition 3.1
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | m unit vectors on S^{d-1} (unit sphere) |
| Attentional Bias | L2 or dot-product (configurable) |
| Retention | Normalization (implicit — projection back to sphere) |
| Algorithm | Riemannian gradient descent (Proposition 3.1) |

CS-36 compliance: Retention is sphere normalization, not L2 weight decay.

## State

```
STATE: LatticeOSRState
  S: Tensor(m, d)           -- memory slots (inner_loop_state)
                             -- INVARIANT: ||S[i]|| = 1 for all i
  beta: Tensor(m)           -- per-slot gating (data-dependent, inner_loop_state)

OUTER_PARAMS: LatticeOSRParams
  W_K: Tensor(d, d_model)
  W_V: Tensor(d, d_model)
  W_Q: Tensor(d, d_model)
  gate_params: GateParams
```

## Core Operation: Orthogonal Update

```
FUNCTION: orthogonal_update(s: &Tensor, delta_s: &Tensor) -> Tensor
  -- The defining operation of Lattice.
  -- Project delta_s onto the tangent plane of the sphere at s,
  -- then normalize back to the sphere.

  -- Step 1: Remove component parallel to s
  -- proj_s(delta_s) = (s^T @ delta_s) * s
  parallel = dot(s, delta_s) * s
  orthogonal = delta_s - parallel

  -- Step 2: Update and renormalize
  s_new = s + orthogonal
  s_new = s_new / norm(s_new)

  return s_new
  -- Proposition 3.1: This is gradient descent on the unit sphere manifold
  -- (Riemannian GD with the natural metric inherited from R^d)
```

## Pseudocode

### Decoding Variant (Lattice Eqs 5-6)

```
ALGORITHM: lattice_osr_decode_step(state: &mut LatticeOSRState, x_t: &Tensor,
                                    outer: &LatticeOSRParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T     -- (d,)
  v_t = x_t @ outer.W_V^T     -- (d,)
  q_t = x_t @ outer.W_Q^T     -- (d,)

  gates = compute_gates(k_t, v_t, outer.gate_params)

  IF NOT pulse.is_active(self.level):
    return lattice_read(state, q_t)

  -- Write: update each memory slot (Lattice Eqs 5-6)
  FOR i = 0 to m-1:
    -- Compute association strength
    score_i = dot(state.S[i], k_t)        -- how relevant is this slot?
    gate_i = sigmoid(score_i)              -- state-dependent gating

    -- Compute update direction
    delta_s = gate_i * v_t                 -- scaled value

    -- Orthogonal update: only NOVEL information modifies the slot
    state.S[i] = orthogonal_update(state.S[i], delta_s)
    -- INVARIANT: ||state.S[i]|| = 1 (preserved by normalization)

  -- Read (Lattice Eq 6)
  y_t = lattice_read(state, q_t)
  return y_t

FUNCTION: lattice_read(state: &LatticeOSRState, q: &Tensor) -> Tensor
  -- Weighted sum of memory slots, scored by query similarity
  scores = state.S @ q                    -- (m,)
  weights = softmax(scores)               -- attention over slots
  y = weights^T @ state.S                 -- (d,)  weighted combination
  return y
```

### Encoding Variant (Lattice Eqs 24-25)

```
ALGORITHM: lattice_osr_encode_step(state, x_t, outer, pulse) -> Tensor
  -- Same structure but update uses key similarity (not value)
  -- delta_s = gate_i * k_t (encode the KEY, not the VALUE)
  -- Read uses stored keys to retrieve values
  -- See Lattice Section 4.2 for full derivation
```

### Similarity Variant (Lattice Eqs 7-8)

```
ALGORITHM: lattice_osr_similarity_step(state, x_t, outer, pulse) -> Tensor
  -- Update uses cosine similarity between slots
  -- delta_s = gate_i * (v_t - dot(S[i], v_t) * S[i])
  -- Explicit orthogonal projection before normalization
  -- See Lattice Section 3.2 for full derivation
```

## Unified Form (Lattice Eq 26)

All three variants unify under:

```
FUNCTION: lattice_unified_update(s, input, gate)
  -- input = v_t (decode) or k_t (encode) or similarity-weighted combination
  delta_s = gate * input
  s_new = orthogonal_update(s, delta_s)
  return s_new
```

## Why Orthogonal?

Standard memory update: M += v @ k^T
This allows redundant information to accumulate — writing the same association twice
doubles its strength. Memory slots fill with repetitive information.

Orthogonal update: Only NOVEL information (orthogonal to current state) modifies the slot.
Repeated information projects to zero along the current direction.
This is optimal compression — each bit of state stores maximum unique information.

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (freeze slots at chunk boundary)
  - LatticeGLA:       YES  (linearized form enables GLA-style parallelism, see parallelization/04)
  - AssociativeScan:   NO   (normalization is nonlinear)
  - TNTHierarchical:   YES  (excellent as local memory due to low state size)
```

## Axiom Compliance

- **NL IS #4** (compressing context): m slots compress unbounded context
- **Lattice IS #1** (compression as optimization): Update IS optimization on sphere manifold
- **Lattice IS #5** (Riemannian GD): Proven equivalence to gradient descent on S^{d-1}
- **Lattice IS #7** (normalization = forgetting): No explicit decay; normalization controls retention
