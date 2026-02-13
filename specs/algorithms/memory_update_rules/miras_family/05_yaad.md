# YAAD (Huber Loss + Decoupled Retention)

```
CONTRACT
  Purpose:    Robust memory update with Huber attentional bias and
              decoupled local + global retention. Huber loss transitions
              from L2 (small errors) to L1 (large errors), providing
              robustness to outlier tokens while maintaining smooth gradients.
  Expects:    Input token embedding x_t. Pulse. MLP-structured outer params.
              Huber threshold delta (hyperparameter).
  Guarantees: Output y_t. Memory updated with bounded gradient magnitude.
              Gradient never exceeds delta (unlike L2 which is unbounded).
  Cost:       Same as MONETA: O(d_in * d_hidden + d_hidden * d_out).
              Huber gradient is marginally cheaper than l_p (no power computation).
  Trade-off:  More robust to outlier tokens than L2 (Titans) or l_p (MONETA).
              But the threshold delta is a hyperparameter that must be tuned.
              Too small = acts like L1 (ignores large errors).
              Too large = acts like L2 (sensitive to outliers).
  Position:   specs/algorithms/memory_update_rules/miras_family/05_yaad.md
              Sibling of 04_moneta.md (same structure, different bias + retention)
  Source:     MIRAS (2504.13173) Eq 26, Table 2
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | 2-layer MLP (same as MONETA) |
| Attentional Bias | Huber: L2 for |e| < delta, L1 for |e| >= delta |
| Retention | Decoupled: L2 local + L2 global (independent strengths) |
| Algorithm | Gradient descent |

## State

```
STATE: YAADState
  W1: Tensor(d_hidden, d_in)    -- (inner_loop_state)
  W2: Tensor(d_out, d_hidden)   -- (inner_loop_state)
  W1_boundary: Tensor(d_hidden, d_in)  -- snapshot at chunk start (inner_loop_state)
  W2_boundary: Tensor(d_out, d_hidden) -- snapshot at chunk start (inner_loop_state)
  sigma: ActivationFn

OUTER_PARAMS: YAADParams
  W_K, W_V, W_Q: (same as MONETA)
  gate_params: GateParams
  delta: f32                     -- Huber threshold
  lambda_local: f32              -- local retention strength (stay close to boundary)
  lambda_global: f32             -- global retention strength (keep small)
```

## Pseudocode

```
ALGORITHM: yaad_init_chunk(state: &mut YAADState)
  -- Called at the start of each chunk to snapshot boundary state
  state.W1_boundary = state.W1.clone()
  state.W2_boundary = state.W2.clone()

ALGORITHM: yaad_step(state: &mut YAADState, x_t: &Tensor,
                     outer: &YAADParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T
  v_t = x_t @ outer.W_V^T
  q_t = x_t @ outer.W_Q^T

  gates = compute_gates(k_t, v_t, outer.gate_params)

  IF NOT pulse.is_active(self.level):
    return mlp_forward(state, q_t)

  -- MLP forward
  h = state.sigma(state.W1 @ k_t)
  prediction = state.W2 @ h
  error = prediction - v_t

  -- Huber gradient (MIRAS Eq 26)
  -- L_huber(e) = 0.5 * e^2         if |e| < delta
  --            = delta * (|e| - 0.5 * delta)  if |e| >= delta
  -- Gradient:
  --   e                if |e| < delta    (same as L2)
  --   delta * sign(e)  if |e| >= delta   (same as L1, bounded)
  huber_grad = WHERE(abs(error) < outer.delta,
                     error,
                     outer.delta * sign(error))

  -- Backprop through MLP (analytical)
  grad_W2 = outer_product(huber_grad, h)
  grad_h = state.W2^T @ huber_grad
  grad_pre_act = grad_h * sigma_prime(state.W1 @ k_t)
  grad_W1 = outer_product(grad_pre_act, k_t)

  -- Decoupled retention (MIRAS Eq 26, Table 2)
  -- Local: stay close to BOUNDARY state (snapshot at chunk start)
  --   L_local = lambda_local * ||W - W_boundary||^2
  --   grad_local = lambda_local * 2 * (W - W_boundary)
  -- Global: keep overall magnitude bounded
  --   L_global = lambda_global * ||W||^2
  --   grad_global = lambda_global * 2 * W
  ret_local_W1 = outer.lambda_local * 2 * (state.W1 - state.W1_boundary)
  ret_global_W1 = outer.lambda_global * 2 * state.W1
  ret_local_W2 = outer.lambda_local * 2 * (state.W2 - state.W2_boundary)
  ret_global_W2 = outer.lambda_global * 2 * state.W2

  -- Update
  state.W1 = state.W1 - gates.theta * (grad_W1 + ret_local_W1 + ret_global_W1)
  state.W2 = state.W2 - gates.theta * (grad_W2 + ret_local_W2 + ret_global_W2)

  h_q = state.sigma(state.W1 @ q_t)
  y_t = state.W2 @ h_q
  return y_t
```

## Why Huber?

The gradient of L2 loss grows linearly with error magnitude: grad = 2*error.
An outlier token with large error produces a large gradient that can destabilize memory.
Huber clips the gradient at delta, providing:
- L2 behavior for normal tokens (smooth, efficient learning)
- L1 behavior for outliers (bounded update, stability)

This matters in long-context models where a single unusual token shouldn't overwrite
good associations built from thousands of prior tokens.

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES
  - AssociativeScan:   NO   (nonlinear MLP)
  - TNTHierarchical:   YES
```

## Axiom Compliance

Same as MONETA. Additionally:
- **MIRAS IS #1** (orthogonal choices): Huber bias chosen independently of retention
