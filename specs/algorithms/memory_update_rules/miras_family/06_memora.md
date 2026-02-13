# MEMORA (KL Divergence + Softmax Emergence)

```
CONTRACT
  Purpose:    Memory update where KL divergence retention causes softmax
              structure to EMERGE from the optimization objective. The memory
              weights live on the probability simplex — each row is a
              distribution. This is the only rule where the mathematical
              structure of attention emerges from first principles.
  Expects:    Input token embedding x_t. Pulse. MLP-structured outer params.
              Memory initialized on the probability simplex (rows sum to 1).
  Guarantees: Output y_t. Memory stays on probability simplex at all times.
              Softmax appears in the update rule as a CONSEQUENCE, not an assumption.
  Cost:       Per token: O(d_in * d_hidden + d_hidden * d_out) + softmax over d_in.
              Slightly more expensive than MONETA due to softmax normalization.
  Trade-off:  Mathematically elegant (softmax emerges, not imposed).
              But probability simplex constraint limits representational capacity —
              memory can only store distributions, not arbitrary matrices.
              Works well when associations ARE distributions (e.g., next-token probs).
  Position:   specs/algorithms/memory_update_rules/miras_family/06_memora.md
              Sibling of 04_moneta.md, 05_yaad.md
  Source:     MIRAS (2504.13173) Eq 27, Proposition 3.1
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | 2-layer MLP on probability simplex |
| Attentional Bias | L2 associative: \|\|MLP(k) - v\|\|^2 |
| Retention | KL divergence: D_KL(W \|\| W_{t-1}) |
| Algorithm | Gradient descent (with closed-form KL solution) |

CS-36 compliance: Retention is KL divergence, not restricted to L2.

## The Softmax Emergence

MIRAS Proposition 3.1 proves: when you optimize L2 loss subject to KL retention,
the closed-form solution contains a softmax:

```
-- Objective at each step:
W_t = argmin_W  ||MLP_W(k_t) - v_t||^2 + (1/alpha_t) * D_KL(W || W_{t-1})

-- Closed-form solution (proven in MIRAS):
W_t[i] = softmax(alpha_t * log(W_{t-1}[i]) - eta_t * grad[i])

-- The softmax is NOT a design choice — it is a mathematical consequence
-- of combining L2 loss with KL retention.
```

This means standard attention (which uses softmax by design choice) is actually
a special case of MEMORA (where softmax emerges from optimization theory).

## State

```
STATE: MEMORAState
  W1: Tensor(d_hidden, d_in)    -- on probability simplex (inner_loop_state)
  W2: Tensor(d_out, d_hidden)   -- on probability simplex (inner_loop_state)
  sigma: ActivationFn

OUTER_PARAMS: MEMORAParams
  W_K, W_V, W_Q: (same as MONETA)
  gate_params: GateParams
```

## Pseudocode

```
ALGORITHM: memora_step(state: &mut MEMORAState, x_t: &Tensor,
                       outer: &MEMORAParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T
  v_t = x_t @ outer.W_V^T
  q_t = x_t @ outer.W_Q^T

  gates = compute_gates(k_t, v_t, outer.gate_params)
  alpha_t = gates.alpha
  eta_t = gates.eta

  IF NOT pulse.is_active(self.level):
    return mlp_forward(state, q_t)

  -- MLP forward + gradient (same as MONETA, L2 bias)
  h = state.sigma(state.W1 @ k_t)
  prediction = state.W2 @ h
  error = prediction - v_t
  grad_W2 = outer_product(error, h)
  grad_h = state.W2^T @ error
  grad_pre_act = grad_h * sigma_prime(state.W1 @ k_t)
  grad_W1 = outer_product(grad_pre_act, k_t)

  -- KL-optimal update (MIRAS Eq 27, Proposition 3.1)
  -- For each row of W1, W2:
  --   W_new[i] = softmax(alpha_t * log(W_prev[i]) - eta_t * grad[i])
  state.W1 = row_wise_softmax(alpha_t * log(state.W1) - eta_t * grad_W1)
  state.W2 = row_wise_softmax(alpha_t * log(state.W2) - eta_t * grad_W2)

  -- Read via updated MLP (still on simplex)
  h_q = state.sigma(state.W1 @ q_t)
  y_t = state.W2 @ h_q
  return y_t

FUNCTION: row_wise_softmax(X: Tensor) -> Tensor
  -- Apply softmax independently to each row
  -- Ensures each row remains a valid probability distribution
  FOR each row i:
    X[i] = softmax(X[i])
  return X
```

## Initialization

Memory must start on the probability simplex:

```
FUNCTION: init_memora_state(d_hidden, d_in, d_out) -> MEMORAState
  -- Uniform initialization: each row is 1/d (maximum entropy)
  W1 = ones(d_hidden, d_in) / d_in
  W2 = ones(d_out, d_hidden) / d_hidden
  return MEMORAState { W1, W2, sigma: SiLU }
```

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (freeze at boundary, but KL requires log of boundary state)
  - AssociativeScan:   NO   (softmax is nonlinear)
  - TNTHierarchical:   YES
```

## Axiom Compliance

- **MIRAS IS #3** (proven optimality): KL retention + L2 bias has a closed-form optimal solution
- **NL IS #6** (optimizers are memory): The softmax-structured memory IS an attention mechanism
- **NL IS #4** (compressing context): KL retention controls information retention rate
