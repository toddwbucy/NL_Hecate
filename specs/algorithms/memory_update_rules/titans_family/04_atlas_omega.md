# Atlas Omega Rule (Context-Window Memory)

<!-- HADES: atlas_equations/eq-009-omega-rule (Atlas Eq 9); atlas_equations/eq-010-omeganet-update (Atlas Eq 10); atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32); atlas_equations/eq-033-atlas-momentum (Atlas Eq 33); atlas_equations/eq-034-parallel-memory-recurrence (Atlas Eq 34) -->
```text
CONTRACT
  Purpose:    The Omega rule is Atlas's core memory update — strictly more
              powerful than the Delta rule (Titans). Where Delta computes surprise
              from a SINGLE token (||M(k_t) - v_t||^2), Omega computes surprise
              from a LOCAL CONTEXT WINDOW of c tokens:
                sum_{i=t-c+1}^{t} gamma_i ||M(k_i) - v_i||^2
              This context-window gradient gives the memory a richer signal about
              what to learn, reducing noise from individual tokens. Combined with
              Muon momentum (Newton-Schulz orthogonalization, Eq 32), Atlas achieves
              state-independent momentum — the momentum computation depends only on
              outer_loop_params, not on M. This enables EXACT parallel momentum
              (see 05_atlas_parallel.md), unlike Titans where momentum is sequential.
  Expects:    Input token embedding x_t of dimension d_model. A sliding window of
              c recent (key, value) pairs. Feature map phi (optional, identity default).
              Attention weights gamma_i^(t) within the window. Pulse from Conductor.
              Outer-loop parameters: W_K, W_V, W_Q, gate weights, phi params.
  Guarantees: Output y_t of dimension d_out. Memory state updated in place.
              Gradient is analytical (context-window sum). Momentum is state-independent
              — depends only on outer_loop_params and the token stream, NOT on M.
              Supports: atlas_parallel (exact), chunkwise_gd (approximate).
              When c = 1: reduces to Titans LMM (single-token surprise).
              When phi = identity: reduces to standard Omega (no feature map).
  Cost:       Per token: O(c * d_out * d_in) for context-window gradient (c matmuls).
              Momentum: O(d_out * d_in) for EMA + O(d_out * d_in * k_ns) for NS.
              Memory update: O(d_out * d_in) for retention + momentum application.
              Total: O(c * d^2 + k_ns * d^2) per token.
  Trade-off:  Richer gradient signal (c tokens) vs higher per-token cost (c× more
              matmuls). State-independent momentum enables exact parallelism but
              requires the Omega rule structure — the gradient must not depend on M
              for the parallel form. With c = 1, cost matches Titans but loses the
              context-window advantage.
  Position:   specs/algorithms/memory_update_rules/titans_family/04_atlas_omega.md
              Child of memory_update_rules/00_interface.md
              Sibling of 01_titans_lmm.md (Atlas extends Titans)
              See also: parallelization/05_atlas_parallel.md (parallel momentum)
              See also: optimization_machinery/09_adamuon.md (Muon/AdaMuon outer loop)
  Source:     Atlas (2505.23735) Eqs 9-11, 15, 32-34, 56; Section 5
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | Matrix M in R^{d_out x d_in} |
| Attentional Bias | L2 context-window: sum gamma_i \|\|M(phi(k_i)) - v_i\|\|^2 |
| Retention | L2 weight decay: alpha_t * M (data-dependent, per-dimension) |
| Algorithm | GD + Muon momentum (Newton-Schulz orthogonalization) |

Key difference from Titans LMM: the attentional bias sums over a context
window of c tokens instead of a single token, and the algorithm uses Muon
(Newton-Schulz) instead of standard EMA momentum.

## State

```text
STATE: AtlasOmegaState
  M: Tensor(d_out, d_in)     -- memory matrix (inner_loop_state)
  S: Tensor(d_out, d_in)     -- momentum accumulator (inner_loop_state)
  window_keys: RingBuffer(c, d_in)   -- recent keys for context window
  window_vals: RingBuffer(c, d_out)  -- recent values for context window

OUTER_PARAMS: AtlasOmegaParams
  W_K: Tensor(d_in, d_model)     -- key projection (outer_loop_param)
  W_V: Tensor(d_out, d_model)    -- value projection (outer_loop_param)
  W_Q: Tensor(d_out, d_model)    -- query projection (outer_loop_param)
  gate_params: GateParams        -- alpha, eta, theta projections (outer_loop_param)
  phi_params: Option<PhiParams>  -- feature map parameters (outer_loop_param, if learned)
  -- Attention weights gamma within the window:
  --   Fixed (uniform 1/c) or learned (outer_loop_param)
```

## The Omega Rule (Atlas Eq 9)

The core learning rule — surprise measured over a local context:

<!-- HADES: atlas_equations/eq-009-omega-rule (Atlas Eq 9) -->
```text
-- Omega Rule (Atlas Eq 9):
--   min_M sum_{i=t-c+1}^{t} gamma_i^(t) ||M(k_i) - v_i||^2
--
-- Compare to Delta Rule (Titans):
--   min_M ||M(k_t) - v_t||^2    (single token, c = 1)
--
-- The gradient:
--   grad = sum_{i=t-c+1}^{t} gamma_i * 2 * (M(k_i) - v_i) @ k_i^T
--
-- With phi feature map (Atlas Eq 10/56):
--   grad = sum_{i=t-c+1}^{t} gamma_i * 2 * (M @ phi(k_i) - v_i) @ phi(k_i)^T
--
-- gamma_i^(t) are attention weights within the window:
--   Uniform: gamma_i = 1/c (all tokens equally weighted)
--   Decaying: gamma_i = decay^(t-i) (recent tokens weighted more)
--   Learned: gamma from a softmax over window positions (outer_loop_param)
```

## Per-Token Update (Sequential Form)

<!-- HADES: atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32); atlas_equations/eq-033-atlas-momentum (Atlas Eq 33) -->
```text
ALGORITHM: atlas_omega_step(state: &mut AtlasOmegaState, x_t: &Tensor,
                             outer: &AtlasOmegaParams, pulse: &Pulse) -> Tensor

  -- Project input to key, value, query
  k_t = x_t @ outer.W_K^T                          -- (d_in,)
  v_t = x_t @ outer.W_V^T                          -- (d_out,)
  q_t = x_t @ outer.W_Q^T                          -- (d_out,)

  -- Apply feature map (if configured)
  phi_k_t = phi(k_t, outer.phi_params)              -- (d_phi,) or (d_in,) if identity

  -- Update sliding window
  state.window_keys.push(phi_k_t)
  state.window_vals.push(v_t)

  -- Compute gates (data-dependent, outer_loop_param projections)
  gates = compute_gates(k_t, v_t, outer.gate_params)
  alpha_t = gates.alpha                              -- retain gate [0,1]
  eta_t   = gates.eta                                -- momentum decay [0,1]
  theta_t = gates.theta                              -- learning rate (positive)

  -- Check frequency schedule
  IF NOT pulse.is_active(self.level):
    y_t = state.M @ phi(q_t, outer.phi_params)
    return y_t

  -- Compute context-window gradient (Atlas Eq 9/56)
  grad = zeros(d_out, d_in)
  FOR i in 0..state.window_keys.len():
    ki = state.window_keys[i]
    vi = state.window_vals[i]
    gamma_i = outer.gamma[i]                        -- or 1/c for uniform
    error_i = state.M @ ki - vi                     -- (d_out,)
    grad += gamma_i * outer_product(error_i, ki)    -- (d_out, d_in)

  -- Update momentum with Muon (Atlas Eq 33)
  state.S = eta_t * state.S + theta_t * grad        -- (d_out, d_in)

  -- Apply Newton-Schulz orthogonalization (Atlas Eq 32)
  ns_update = newton_schulz_k(state.S, k_ns)        -- (d_out, d_in)

  -- Update memory with retention (Atlas Eq 32)
  state.M = alpha_t * state.M - theta_t * ns_update  -- (d_out, d_in)

  -- Read: query memory
  y_t = state.M @ phi(q_t, outer.phi_params)        -- (d_out,)

  return y_t
```

## State-Independent Momentum (Atlas Key Insight)

The Omega gradient's dependency on M is what prevents exact parallel momentum
in Titans. Atlas restructures this:

<!-- HADES: atlas_equations/eq-033-atlas-momentum (Atlas Eq 33); atlas_equations/eq-034-parallel-memory-recurrence (Atlas Eq 34) -->
```text
-- Titans momentum (state-DEPENDENT):
--   grad_t = (M_{t-1} @ k_t - v_t) @ k_t^T     -- depends on M_{t-1}
--   S_t = eta_t * S_{t-1} - theta_t * grad_t    -- depends on M through grad
--   Sequential: S_t depends on M_{t-1} depends on S_{t-1}
--
-- Atlas momentum (state-INDEPENDENT):
--   omega_t = f(k_t, v_t, outer_params)          -- depends only on outer_params
--   S_t = eta_t * S_{t-1} + theta_t * omega_t   -- linear recurrence in S only
--   Parallel: S_t is a linear recurrence, solvable by associative scan
--
-- How Atlas achieves this:
--   The context-window gradient at chunk boundaries uses M_frozen (chunk start).
--   Within a chunk, omega_t = sum gamma_i (M_frozen @ phi(k_i) - v_i) @ phi(k_i)^T
--   M_frozen is CONSTANT within the chunk → omega depends only on:
--     - M_frozen (known at chunk start)
--     - k_i, v_i (input tokens, known)
--     - outer_params (fixed during inner loop)
--   Therefore omega_t is state-independent within a chunk.
--
-- Parallel memory recurrence (Atlas Eq 34):
--   M_t = alpha_t * M_{t-1} + S_t
--   This is a linear recurrence: M_t = A_t * M_{t-1} + B_t
--   where A_t = alpha_t * I, B_t = S_t (pre-computed in parallel)
--   Solvable by associative scan in O(log C) depth.
```

## OmegaNet: Linear Memory Expansion (Atlas Eq 11)

For matrix-valued memory with feature maps, the expanded form:

<!-- HADES: atlas_equations/eq-011-omeganet-linear-case (Atlas Eq 11) -->
```text
-- OmegaNet linear case (Atlas Eq 11):
--   M_t = (diag(alpha_t) - sum gamma_i phi(k_i) @ phi(k_i)^T) @ M_{t-1}
--         - sum gamma_i v_i @ phi(k_i)^T
--
-- This separates into:
--   Data-dependent decay: (diag(alpha_t) - sum gamma_i phi(k_i)phi(k_i)^T) @ M_{t-1}
--   Context-aware update: - sum gamma_i v_i @ phi(k_i)^T
--
-- The decay matrix is NOT a scalar times identity — it is data-dependent.
-- Directions with high key density (sum phi(k_i)phi(k_i)^T) decay faster.
-- This is selective forgetting: the memory preferentially forgets what it
-- is about to overwrite, preserving orthogonal information.
--
-- When phi = identity and c = 1:
--   M_t = (alpha_t * I - k_t @ k_t^T) @ M_{t-1} - v_t @ k_t^T
--   This is the Titans Delta rule with directional decay.
```

## Chunkwise Parallel Training (Atlas Eq 15)

<!-- HADES: atlas_equations/eq-015-chunkwise-omega-update (Atlas Eq 15) -->
```text
-- Chunkwise parallel update (Atlas Eq 15):
--   Partition input into chunks of size b.
--   Within each chunk, compute gradients w.r.t. last state of previous chunk.
--   When b = 1: fully recurrent (exact, sequential).
--   When b > 1: parallel within chunk, sequential across chunks.
--
-- This is the SAME pattern as chunkwise GD (01_chunkwise_gd.md) but applied
-- to the context-window gradient. The frozen M at chunk start makes ALL
-- omega values within the chunk computable in parallel.
--
-- See 05_atlas_parallel.md for the full parallel algorithm including:
--   Step 1: Compute all omega_t in parallel (state-independent)
--   Step 2: Scan momentum S (linear recurrence, O(log C) depth)
--   Step 3: Scan memory M (linear recurrence, O(log C) depth)
--   Step 4: Read all y_t in parallel
```

## VJP Gradient Derivation (for Wengert tape)

<!-- HADES: Derived from atlas_equations/eq-009-omega-rule (Atlas Eq 9); atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32), VJP for tape integration -->
```text
-- Forward (one token, sequential form):
--   phi_k = phi(k_t)
--   FOR i in window: error_i = M @ phi_k_i - v_i
--   grad = sum gamma_i * error_i @ phi_k_i^T
--   S = eta * S + theta * grad
--   ns = newton_schulz_k(S, k_ns)
--   M_new = alpha * M - theta * ns
--   y = M_new @ phi(q_t)

-- Given: dL/dy (upstream gradient from composition pattern)
-- Need: dL/d(W_K), dL/d(W_V), dL/d(W_Q), dL/d(gate_params), dL/d(phi_params)

-- Through read (y = M_new @ phi_q):
--   dL/dM_new = dL/dy @ phi_q^T
--   dL/dphi_q = M_new^T @ dL/dy → dL/dW_Q via chain rule

-- Through memory update (M_new = alpha * M - theta * ns):
--   dL/dM (through retention) = alpha * dL/dM_new
--   dL/dns = -theta * dL/dM_new
--   dL/dalpha = sum(dL/dM_new * M)  → dL/d(gate_params) via chain rule

-- Through Newton-Schulz (ns = NS_k(S)):
--   dL/dS = ns_backward(dL/dns, S, k_ns)
--   (NS backward is k iterations of reverse-mode through the NS recurrence)

-- Through momentum (S = eta * S_prev + theta * grad):
--   dL/dS_prev = eta * dL/dS
--   dL/dgrad = theta * dL/dS
--   dL/deta = sum(dL/dS * S_prev)  → dL/d(gate_params)
--   dL/dtheta = sum(dL/dS * grad)  → dL/d(gate_params)

-- Through context-window gradient:
--   FOR i in window:
--     dL/derror_i = gamma_i * dL/dgrad @ phi_k_i    -- (d_out,)
--     dL/dM (through error_i) += gamma_i * dL/derror_i @ phi_k_i^T
--     dL/dphi_k_i (through error) = M^T @ dL/derror_i
--     dL/dphi_k_i (through outer product) = error_i^T @ (gamma_i * dL/dgrad)
--     dL/dv_i = -dL/derror_i
--   Total dL/dM = dL/dM (through retention) + sum_i dL/dM (through error_i)

-- Through projections and feature map:
--   dL/dk_t = dphi/dk @ dL/dphi_k_t  → dL/dW_K = dL/dk_t @ x_t^T
--   dL/dv_t = dL/dv_t (from window)  → dL/dW_V = dL/dv_t @ x_t^T
```

## Special Cases and Relationships

| Configuration | Reduces To | How |
|---|---|---|
| c = 1, phi = identity, no NS | Titans LMM | Single-token surprise, standard momentum |
| c = 1, phi = identity, eta = 0, no NS | Delta Rule | Single-token, no momentum |
| c = 1, phi = identity, eta = 0, no NS, skip grad | Hebbian | Direct outer product |
| c > 1, phi = identity, no NS | "Omega without Muon" | Context-window + standard momentum |
| c > 1, phi = polynomial | OmegaNet (Eq 10) | Full Atlas with feature maps |
| c > 1, phi = polynomial, with NS | Full Atlas (Eq 32) | State-of-the-art Atlas |

## Hyperparameter Defaults

<!-- HADES: Derived from atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32), experimental defaults from Atlas paper -->
```text
-- From Atlas (2505.23735) experiments:
--   c = 8 or 16          (context window size)
--   k_ns = 5             (Newton-Schulz iterations)
--   phi = identity       (feature maps optional, identity is baseline)
--   gamma = uniform 1/c  (equal weighting within window)
--   Gate biases: same as Titans (b_alpha=3.0, b_theta=-4.6)
--
-- Window size c interacts with chunk size C (parallelization):
--   c <= C: window fits within one chunk (simple)
--   c > C: window spans chunks (requires cross-chunk state, complex)
--   Recommended: c <= C to avoid cross-chunk dependencies.
```

## Implementation Notes

1. **Not yet implemented**: Atlas Omega is a Stage 3 extension. The existing Titans
   LMM (`core/src/titans_lmm.rs`) provides the foundation. Adding Atlas requires:
   the sliding window buffer, context-window gradient loop, and Newton-Schulz call
   after momentum. The parallel form (05_atlas_parallel.md) is already specced.

2. **Sliding window buffer**: The ring buffer for (key, value) pairs within the
   context window is `inner_loop_state` — it resets per forward pass (or persists
   as `context_memory` across chunks if c > 1 chunk). Size: O(c * d) per buffer,
   two buffers (keys + values) = O(2 * c * d).

3. **Newton-Schulz in inner loop**: Unlike outer-loop AdaMuon (09_adamuon.md),
   the NS call here is INSIDE the forward pass and must be recorded on the
   Wengert tape as an opaque VJP block. The backward through NS is k_ns iterations
   of reverse-mode through the 3-degree polynomial recurrence.

4. **Feature map dispatch**: phi is configured at initialization via `FeatureMapKind`
   (see self_referential/02_feature_maps.md). Identity is the default and fast path.
   When phi = identity, all phi() calls are no-ops and the memory shape stays [d, d].

5. **Interaction with CMS**: Each CMS level can independently use Titans or Atlas.
   A natural configuration: fast levels use Titans (cheaper, c=1), slow levels use
   Atlas (richer gradient signal justifies the c× cost for infrequent updates).

## Parallelization Support

```text
SUPPORTED_PARALLELIZATION:
  - AtlasParallel:    YES  (exact momentum, see 05_atlas_parallel.md)
  - ChunkwiseGD:      YES  (freeze M at chunk boundary, compute all omega in parallel)
  - AssociativeScan:   YES  (both S and M are linear recurrences within chunk)
  - TNTHierarchical:   YES  (can serve as both global and local memory in TNT)
```

## Axiom Compliance

- **NL IS #2** (nested, multi-level): Inner loop (Omega GD + Muon momentum) nested inside outer loop (tape AD). Three levels: NS orthogonalization → momentum → memory.
- **NL IS #4** (compressing context): Context-window gradient compresses c tokens into a single update direction. Memory compresses the full token stream into matrix weights.
- **NL IS #6** (optimizers are memory): M IS the associative memory. Omega's context-window gradient IS the optimizer. Muon's NS IS a memory of gradient direction.
- **NL IS #7** (self-modifying): Data-dependent gates, context-aware decay matrix (Eq 11) selectively forgets based on input.
- **NL IS NOT #1** (not single-level): Three nested levels — NS within momentum within memory.
- **MIRAS IS #1** (orthogonal design choices): Atlas fills all four knobs with specific choices (matrix, L2-window, directional decay, GD+Muon). Each can be varied independently.
