# MONETA (2-Layer MLP + l_p Loss + L_q Retention)

```
CONTRACT
  Purpose:    Deep memory model with 2-layer MLP structure, l_p attentional bias,
              and mixed L_q + L2 retention. First MIRAS variant to demonstrate
              that MLP memory outperforms matrix memory at the same parameter count.
  Expects:    Input token embedding x_t. Pulse. MLP-structured outer params.
  Guarantees: Output y_t. Memory updated via gradient descent through 2-layer MLP.
              l_p loss provides robust error signal (less sensitive to outliers
              than L2 when p < 2).
  Cost:       Per token: O(d_in * d_hidden + d_hidden * d_out) for MLP forward + grad.
              Memory: O(d_in * d_hidden + d_hidden * d_out) for W1, W2.
              More expensive than matrix rules but higher capacity per parameter.
  Trade-off:  Higher capacity (MLP can learn nonlinear associations) vs higher cost.
              l_p loss with p < 2 is more robust to outliers but gradient undefined at 0.
              L_q retention with q < 2 promotes sparsity (hard forgetting).
  Position:   specs/algorithms/memory_update_rules/miras_family/04_moneta.md
              Child of memory_update_rules/00_interface.md
  Source:     MIRAS (2504.13173) Eqs 24-25, Table 2
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | 2-layer MLP: W1(d_hidden, d_in), sigma, W2(d_out, d_hidden) |
| Attentional Bias | l_p norm: \|\|MLP(k) - v\|\|_p^p |
| Retention | L_q local + L2 global: lambda_q\|\|W-W_prev\|\|_q^q + lambda_2\|\|W\|\|_2^2 |
| Algorithm | Gradient descent (single step per token) |

CS-34 compliance: Memory is 2-layer MLP, not restricted to matrix.
CS-35 compliance: Algorithm is GD, but declared as a choice, not hardcoded.

## State

```
STATE: MONETAState
  W1: Tensor(d_hidden, d_in)    -- first layer (inner_loop_state)
  W2: Tensor(d_out, d_hidden)   -- second layer (inner_loop_state)
  sigma: ActivationFn           -- nonlinearity (typically SiLU or GELU)

OUTER_PARAMS: MONETAParams
  W_K: Tensor(d_in, d_model)
  W_V: Tensor(d_out, d_model)
  W_Q: Tensor(d_out, d_model)
  gate_params: GateParams
  p: f32                         -- l_p norm exponent (typically 1.0 or 1.5)
  q: f32                         -- L_q retention exponent
  lambda_q: f32                  -- local retention strength
  lambda_2: f32                  -- global retention strength
```

## Pseudocode

```
ALGORITHM: moneta_step(state: &mut MONETAState, x_t: &Tensor,
                       outer: &MONETAParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T
  v_t = x_t @ outer.W_V^T
  q_t = x_t @ outer.W_Q^T

  gates = compute_gates(k_t, v_t, outer.gate_params)

  IF NOT pulse.is_active(self.level):
    return mlp_forward(state, q_t)

  -- MLP forward pass
  h = state.sigma(state.W1 @ k_t)            -- hidden layer (d_hidden,)
  prediction = state.W2 @ h                    -- output (d_out,)

  -- l_p loss gradient (MIRAS Eq 24)
  -- d/dW ||MLP(k) - v||_p^p
  -- Requires chain rule through MLP layers
  error = prediction - v_t                     -- (d_out,)
  lp_grad_error = p * sign(error) * abs(error)^(p-1)   -- l_p gradient

  -- Backprop through MLP (analytical, #[no_autodiff])
  grad_W2 = outer_product(lp_grad_error, h)             -- (d_out, d_hidden)
  grad_h = state.W2^T @ lp_grad_error                   -- (d_hidden,)
  grad_pre_act = grad_h * sigma_prime(state.W1 @ k_t)   -- (d_hidden,)
  grad_W1 = outer_product(grad_pre_act, k_t)             -- (d_hidden, d_in)

  -- Retention gradient (MIRAS Eq 25)
  -- L_q local: lambda_q * q * sign(W - W_prev) * |W - W_prev|^(q-1)
  -- L2 global: 2 * lambda_2 * W
  -- W_prev is the state at chunk boundary (parallelization uses this)
  ret_grad_W1 = outer.lambda_2 * 2 * state.W1
  ret_grad_W2 = outer.lambda_2 * 2 * state.W2

  -- Combined update
  state.W1 = state.W1 - gates.theta * (grad_W1 + ret_grad_W1)
  state.W2 = state.W2 - gates.theta * (grad_W2 + ret_grad_W2)

  -- Read via updated MLP
  h_q = state.sigma(state.W1 @ q_t)
  y_t = state.W2 @ h_q

  return y_t

FUNCTION: mlp_forward(state: &MONETAState, input: &Tensor) -> Tensor
  h = state.sigma(state.W1 @ input)
  return state.W2 @ h
```

## Why 2-Layer MLP?

MIRAS Section 4.2 shows that matrix memory has capacity O(min(d_in, d_out)).
MLP memory has capacity that scales with the number of parameters AND the
representational power of the nonlinearity. For the same parameter count,
MLP stores more distinct associations.

The cost: gradient computation requires backprop through the MLP (still analytical
but more operations). And the nonlinearity breaks the linear structure that
enables associative scan parallelization.

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (freeze MLP weights at chunk boundary)
  - AssociativeScan:   NO   (nonlinear MLP prevents linear recurrence)
  - TNTHierarchical:   YES  (can be local or global memory)
```

## Axiom Compliance

- **NL IS #4** (compressing context): MLP compresses input-output associations
- **NL IS #6** (optimizers are memory): The MLP IS the associative memory
- **MIRAS IS #1** (orthogonal design choices): Each knob independently chosen
- **NL IS NOT #4** (not discrete): Continuous MLP weights, not slot-based
