# Delta Rule

```
CONTRACT
  Purpose:    Momentum-free special case of Titans LMM. Simplest gradient-based
              memory update. Removes momentum accumulator, halving state size.
  Expects:    Same as Titans LMM (input token, Pulse, outer params).
  Guarantees: Same output contract as Titans LMM.
              State size halved (no momentum S).
              Strictly a subset of Titans LMM behavior (eta=0).
  Cost:       Per token: O(d_out * d_in) for gradient + update + read.
              Memory: O(d_out * d_in) for M only (no S).
  Trade-off:  Half the state memory. Noisier updates (no temporal smoothing).
              Cannot use associative scan parallelization (no linear recurrence).
              Simpler to reason about — each update is independent of history.
  Position:   specs/algorithms/memory_update_rules/titans_family/02_delta_rule.md
              Child of 01_titans_lmm.md (special case: eta=0)
  Source:     Titans (2501.00663) Eq 34; MIRAS (2504.13173) Table 1 row "Delta"
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | Matrix M in R^{d_out x d_in} |
| Attentional Bias | L2 associative: \|\|M(k) - v\|\|^2 |
| Retention | L2 weight decay: (1 - alpha_t) * M |
| Algorithm | Gradient descent (NO momentum) |

## State

```
STATE: DeltaRuleState
  M: Tensor(d_out, d_in)     -- memory matrix (inner_loop_state)
  -- No S (momentum). This is the defining difference from Titans LMM.

OUTER_PARAMS: same as TitansLMMParams
```

## Pseudocode

```
ALGORITHM: delta_rule_step(state: &mut DeltaRuleState, x_t: &Tensor,
                           outer: &TitansLMMParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T
  v_t = x_t @ outer.W_V^T
  q_t = x_t @ outer.W_Q^T

  gates = compute_gates(k_t, v_t, outer.gate_params)
  alpha_t = gates.alpha
  theta_t = gates.theta
  -- gates.eta is IGNORED (momentum disabled)

  IF NOT pulse.is_active(self.level):
    return state.M @ q_t

  -- Analytical gradient (Titans Eq 12)
  prediction = state.M @ k_t
  error = prediction - v_t
  grad = outer_product(error, k_t)

  -- Direct update: no momentum accumulation (Titans Eq 34)
  state.M = (1 - alpha_t) * state.M - theta_t * grad

  y_t = state.M @ q_t
  return y_t
```

## Derivation from Titans LMM

```
-- Titans LMM:
S_t = eta_t * S_{t-1} - theta_t * grad
M_t = (1 - alpha_t) * M_{t-1} + S_t

-- Set eta_t = 0:
S_t = -theta_t * grad              -- momentum is just scaled gradient
M_t = (1 - alpha_t) * M_{t-1} + (-theta_t * grad)
M_t = (1 - alpha_t) * M_{t-1} - theta_t * grad    -- Delta Rule (Eq 34)

-- S is no longer needed as state — it's computed fresh each step.
```

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (standard freeze-at-boundary)
  - AssociativeScan:   NO   (no linear recurrence to exploit — memory update is nonlinear)
  - TNTHierarchical:   YES  (can serve as local or global memory)
```

## When to Use Delta Rule vs Titans LMM

- **Delta Rule**: When state memory is constrained. When temporal smoothing isn't needed.
  When the task is "remember exact associations" rather than "track slowly drifting patterns."
- **Titans LMM**: When you need temporal smoothing. When gradients are noisy.
  When you want associative scan parallelization on the momentum dimension.

## Axiom Compliance

- Same as Titans LMM (IS #2, #4, #6, #7; IS NOT #1, #3, #4)
- Still two-level (inner GD, outer Enzyme AD) even without momentum
