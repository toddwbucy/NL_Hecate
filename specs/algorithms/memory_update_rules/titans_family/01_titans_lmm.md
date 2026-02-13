# Titans LMM (Neural Long-Term Memory)

```
CONTRACT
  Purpose:    Foundational memory module. Matrix-valued associative memory
              with gradient descent, momentum, and data-dependent gating.
              Delta rule and Hebbian are special cases (eta=0, no gradient).
  Expects:    Input token embedding x_t of dimension d_model.
              A Pulse with frequency level activity status.
              Outer-loop parameters: W_K, W_V, W_Q, gate weights.
  Guarantees: Output y_t of dimension d_out.
              Memory state updated in place (inner_loop_state lifetime).
              Gradient is analytical (no Enzyme needed for inner loop).
              Supports: chunkwise_gd, associative_scan (momentum only).
  Cost:       Per token: O(d_out * d_in) for gradient + update + read.
              Memory: O(d_out * d_in) for M + O(d_out * d_in) for S.
  Trade-off:  Momentum adds memory cost (doubles state) but enables temporal
              smoothing and associative scan parallelization. Without momentum
              (Delta Rule), state is halved but gradients are noisier.
  Position:   specs/algorithms/memory_update_rules/titans_family/01_titans_lmm.md
              Child of memory_update_rules/00_interface.md
              Parent of 02_delta_rule.md, 03_hebbian_rule.md (special cases)
  Source:     Titans (2501.00663) Section 3.1, Appendix C, Eqs 8-15, 32-33
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | Matrix M in R^{d_out x d_in} |
| Attentional Bias | L2 associative: \|\|M(k) - v\|\|^2 |
| Retention | L2 weight decay: (1 - alpha_t) * M |
| Algorithm | Gradient descent + momentum |

## State

```
STATE: TitansLMMState
  M: Tensor(d_out, d_in)     -- memory matrix (inner_loop_state)
  S: Tensor(d_out, d_in)     -- momentum accumulator (inner_loop_state)

OUTER_PARAMS: TitansLMMParams
  W_K: Tensor(d_in, d_model)     -- key projection (outer_loop_param)
  W_V: Tensor(d_out, d_model)    -- value projection (outer_loop_param)
  W_Q: Tensor(d_out, d_model)    -- query projection (outer_loop_param)
  gate_params: GateParams        -- alpha, eta, theta projections (outer_loop_param)
```

## Per-Token Update (Sequential Form)

This is the conceptual algorithm. In practice, use chunked forms from parallelization/.

```
ALGORITHM: titans_lmm_step(state: &mut TitansLMMState, x_t: &Tensor,
                            outer: &TitansLMMParams, pulse: &Pulse) -> Tensor
  -- #[no_autodiff] on inner-loop operations
  -- Enzyme differentiates through outer (W_K, W_V, W_Q, gate_params)

  -- Project input to key, value, query
  k_t = x_t @ outer.W_K^T                          -- (d_in,)
  v_t = x_t @ outer.W_V^T                          -- (d_out,)
  q_t = x_t @ outer.W_Q^T                          -- (d_out,)

  -- Compute gates (data-dependent, outer_loop_param projections)
  gates = compute_gates(k_t, v_t, outer.gate_params)
  alpha_t = gates.alpha                              -- retain gate [0,1]
  eta_t   = gates.eta                                -- momentum decay [0,1]
  theta_t = gates.theta                              -- learning rate (positive)

  -- Check frequency schedule
  IF NOT pulse.is_active(self.level):
    -- This level is frozen. Read only, no update.
    y_t = state.M @ q_t
    return y_t

  -- Compute gradient of associative memory loss (Titans Eq 12)
  -- Loss: ||M @ k_t - v_t||^2
  -- Gradient w.r.t. M: 2 * (M @ k_t - v_t) @ k_t^T
  -- This is ANALYTICAL — no autodiff.
  prediction = state.M @ k_t                         -- (d_out,)
  error = prediction - v_t                           -- (d_out,)
  grad = outer_product(error, k_t)                   -- (d_out, d_in)

  -- Update momentum: surprise decomposition (Titans Eq 14)
  state.S = eta_t * state.S - theta_t * grad         -- (d_out, d_in)

  -- Update memory with retention (Titans Eq 13)
  state.M = (1 - alpha_t) * state.M + state.S        -- (d_out, d_in)

  -- Read: query memory (Titans Eq 15)
  y_t = state.M @ q_t                                -- (d_out,)

  return y_t
```

## Implementation Form (Per-Dimension Gates)

Titans Appendix C, Eqs 32-33. Gates are vectors, not scalars.

```
ALGORITHM: titans_lmm_step_impl(state, x_t, outer, pulse) -> Tensor
  -- Same flow as above, but gates are per-dimension:
  alpha_t: Tensor(d_out)     -- per-dimension retain
  eta_t:   Tensor(d_out)     -- per-dimension momentum decay
  theta_t: Tensor(d_out)     -- per-dimension learning rate

  -- Momentum (Eq 33): per-dimension scaling
  state.S = diag(eta_t) @ state.S - diag(theta_t) @ (state.M @ k_t @ k_t^T - v_t @ k_t^T)

  -- Memory (Eq 32): per-dimension retention
  state.M = diag(1 - alpha_t) @ state.M + state.S

  -- Read
  y_t = state.M @ q_t
  return y_t
```

## Special Cases

| Variant | How to Derive | File |
|---|---|---|
| **Delta Rule** | Set eta_t = 0 (no momentum). Titans Eq 34. | 02_delta_rule.md |
| **Hebbian** | Set eta_t = 0, skip gradient. Direct: M += v @ k^T. MIRAS Eq 8. | 03_hebbian_rule.md |
| **Linear Attention** | alpha_t = 0, eta_t = 0. Titans Eq 4. | (degenerate case) |
| **Gated DeltaNet** | eta_t = 0, alpha_t = per-dim vector. Titans Eq 34 + gating. | (degenerate case) |

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (freeze state at chunk boundary, compute all grads at once)
  - AssociativeScan:   PARTIAL  (momentum update is linear recurrence, memory update is not)
  - TNTHierarchical:   YES  (can serve as both global and local memory in TNT)
```

## Key Properties

1. **Inner loop**: WRITE is the inner optimization. It runs during forward pass. No external optimizer.
2. **Outer loop**: W_K, W_V, W_Q, gate_params learn via Enzyme AD through the entire sequence.
3. **No learning rate hyperparameter**: theta_t is data-dependent. The "learning rate" is OUTPUT of the gate network.
4. **No mode distinction**: WRITE runs identically always. Memory always updates. (CS-10, CS-38)
5. **Gradient is analytical**: 2*(Mk - v)@k^T. No Enzyme for inner loop. (CS-40)

## Axiom Compliance

- **NL IS #2** (nested, multi-level): Inner loop (GD+momentum) nested inside outer loop (Enzyme AD)
- **NL IS #4** (compressing context): Memory compresses key-value associations into matrix weights
- **NL IS #6** (optimizers are memory): M IS the associative memory. The update rule IS the optimizer.
- **NL IS #7** (self-modifying): Memory modifies itself based on its own prediction error
- **NL IS NOT #1** (not single-level): Two levels — inner (GD+momentum) and outer (Enzyme AD)
- **NL IS NOT #3** (not static): Update rule runs always, gates are data-dependent
- **NL IS NOT #4** (not discrete memory): Continuous matrix, not slot-based
