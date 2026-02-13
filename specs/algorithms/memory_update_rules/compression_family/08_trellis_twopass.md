# Trellis Two-Pass KV Compression

```
CONTRACT
  Purpose:    Two-pass memory compression: first pass compresses keys via
              OGD on state matrix, normalized SiLU activation produces
              compressed output, second pass compresses values using the
              compressed key output. Evolution of Lattice — same bilevel
              optimization, drops orthogonal projection, adds two-pass + SiLU.
  Expects:    Input token embedding x_t. Pulse.
              Two state matrices S_K (key compression) and S_V (value compression).
  Guarantees: Output y_t. Key and value compression are decoupled.
              Normalized SiLU provides smooth, bounded activation.
              Outperforms Mamba2 and GLA at 340M-1.3B params.
  Cost:       Per token: O(d_k * d + d_v * d) for two-pass update + read.
              Memory: O(d_k * d + d_v * d) for two state matrices.
              ~2x cost of single-pass Lattice but better compression quality.
  Trade-off:  Two passes = better compression quality (keys and values compressed
              separately with appropriate objectives). But 2x the state and compute.
              Normalized SiLU is smoother than Lattice's hard normalization.
  Position:   specs/algorithms/memory_update_rules/compression_family/08_trellis_twopass.md
              Sibling of 07_lattice_osr.md (Lattice evolution)
  Source:     Trellis (2512.23852) Eqs 13-14, Section 3
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | Two matrices: S_K(d_k, d), S_V(d_v, d) |
| Attentional Bias | Nonlinear: normalized SiLU activation |
| Retention | L2 state decay: lambda * \|\|S\|\|^2 |
| Algorithm | Online Gradient Descent (OGD) — one step per token |

## State

```
STATE: TrellisState
  S_K: Tensor(d_k, d)      -- key compression matrix (inner_loop_state)
  S_V: Tensor(d_v, d)      -- value compression matrix (inner_loop_state)

OUTER_PARAMS: TrellisParams
  W_K: Tensor(d_k, d_model)   -- key projection
  W_V: Tensor(d_v, d_model)   -- value projection
  W_Q: Tensor(d_v, d_model)   -- query projection (matches value dim)
  gate_params: GateParams
  lambda_k: f32                -- key state decay rate
  lambda_v: f32                -- value state decay rate
```

## Pseudocode

```
ALGORITHM: trellis_step(state: &mut TrellisState, x_t: &Tensor,
                        outer: &TrellisParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T     -- (d_k,)
  v_t = x_t @ outer.W_V^T     -- (d_v,)
  q_t = x_t @ outer.W_Q^T     -- (d_v,)

  gates = compute_gates(k_t, v_t, outer.gate_params)
  alpha_t = gates.alpha
  theta_t = gates.theta

  IF NOT pulse.is_active(self.level):
    return trellis_read(state, q_t, k_t)

  -- === PASS 1: Key Compression (Trellis Eq 13) ===

  -- OGD step on key state with L2 decay
  -- Loss: ||S_K @ x - k||^2 + lambda_k * ||S_K||^2
  -- Gradient: (S_K @ x - k) @ x^T + lambda_k * S_K
  x_embed = x_t                -- use raw input for key compression
  pred_k = state.S_K @ x_embed
  error_k = pred_k - k_t
  grad_S_K = outer_product(error_k, x_embed) + outer.lambda_k * state.S_K

  -- State decay + gradient step
  state.S_K = (1 - alpha_t) * state.S_K - theta_t * grad_S_K

  -- Compressed key output via normalized SiLU
  compressed_k = normalized_silu(state.S_K @ x_embed)   -- (d_k,)

  -- === PASS 2: Value Compression (Trellis Eq 14) ===

  -- OGD step on value state, using compressed key as input
  -- Loss: ||S_V @ compressed_k - v||^2 + lambda_v * ||S_V||^2
  pred_v = state.S_V @ compressed_k
  error_v = pred_v - v_t
  grad_S_V = outer_product(error_v, compressed_k) + outer.lambda_v * state.S_V

  -- State decay + gradient step
  state.S_V = (1 - alpha_t) * state.S_V - theta_t * grad_S_V

  -- === READ ===
  y_t = trellis_read(state, q_t, k_t)
  return y_t

FUNCTION: trellis_read(state: &TrellisState, q: &Tensor, k_raw: &Tensor) -> Tensor
  -- Compress query through the same key path, then read from value state
  compressed_q = normalized_silu(state.S_K @ k_raw)   -- use key path for query
  y = state.S_V @ compressed_q                         -- (d_v,)
  return y

FUNCTION: normalized_silu(x: &Tensor) -> Tensor
  -- SiLU(x) / ||SiLU(x)|| * sqrt(d)
  -- Normalized to preserve scale. Smooth alternative to Lattice's hard normalization.
  sx = x * sigmoid(x)                     -- SiLU activation
  return sx / norm(sx) * sqrt(x.len())    -- normalize to maintain expected magnitude
```

## Why Two Passes?

Lattice uses one pass: compress input → store association.
Problem: the same compression serves both key matching AND value retrieval.
These are different objectives — what makes a good key (discriminative features)
isn't what makes a good value (complete information).

Trellis decouples them:
- Pass 1 (S_K): Learn to compress inputs into discriminative keys
- Pass 2 (S_V): Learn to compress values using the already-compressed keys

The compressed key output from Pass 1 becomes the input to Pass 2.
This creates a two-stage pipeline where each stage specializes.

## Why Normalized SiLU?

Lattice uses hard normalization (project to unit sphere).
This is discontinuous at the origin and can create gradient issues.

Normalized SiLU is smooth everywhere:
- SiLU(x) = x * sigmoid(x) — smooth, bounded below by 0
- Normalization preserves direction while controlling magnitude
- The sigmoid gating naturally suppresses small activations

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (freeze both S_K and S_V at chunk boundary)
  - LatticeGLA:       YES  (linearized form, same as Lattice)
  - AssociativeScan:   NO   (normalized SiLU is nonlinear)
  - TNTHierarchical:   YES
```

## Axiom Compliance

- **Trellis IS #1** (bilevel optimization): Outer loop learns projections, inner loop updates S_K, S_V
- **Trellis IS #3** (two-pass): Key and value compression are decoupled
- **Trellis IS #4** (normalized SiLU): Smooth activation, not hard normalization
- **NL IS #4** (compressing context): Two-stage compression pipeline
- **Lattice → Trellis evolution**: Same bilevel + OGD + L2 decay + chunkwise. New: two-pass + SiLU.
