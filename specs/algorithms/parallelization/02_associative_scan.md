# Associative Scan (Blelloch Algorithm)

```
CONTRACT
  Purpose:    EXACT parallelization for LINEAR recurrences. Uses parallel
              prefix sum to compute all prefix products in O(log C) steps.
              Works for momentum (linear recurrence) but NOT for memory
              update (nonlinear due to forgetting gate).
  Expects:    A linear recurrence of the form s_t = a_t * s_{t-1} + b_t.
              Chunk of C tokens with pre-computed a_t and b_t values.
  Guarantees: EXACT results (no approximation, unlike chunkwise GD).
              O(log C) parallel steps (vs O(C) sequential).
              But ONLY works for linear recurrences.
  Cost:       O(C * d) total work, O(log C * d) parallel depth.
              Uses 2x memory for the scan (intermediate prefix products).
  Trade-off:  Exact but limited scope. Only applicable to linear recurrences.
              Memory update (M = (1-alpha)*M + S) is NOT linear in M when
              alpha is data-dependent. Momentum update (S = eta*S + theta*g)
              IS linear in S.
  Position:   specs/algorithms/parallelization/02_associative_scan.md
  Source:     Titans (2501.00663) Eq 18, Blelloch (1990) parallel prefix algorithms
```

## The Linear Recurrence

```
-- Any recurrence of this form can be parallelized:
s_t = a_t * s_{t-1} + b_t

-- Where:
-- s_t: state at step t
-- a_t: multiplicative factor (can be data-dependent)
-- b_t: additive factor (can be data-dependent)

-- This includes:
-- Momentum:  S_t = eta_t * S_{t-1} + theta_t * g_t     (a=eta, b=theta*g)
-- EMA:       m_t = beta * m_{t-1} + (1-beta) * x_t     (a=beta, b=(1-beta)*x)
-- Linear attention decay: h_t = gamma * h_{t-1} + v @ k^T  (a=gamma, b=v@k^T)

-- This does NOT include:
-- Memory update: M_t = (1-alpha_t)*M_{t-1} + S_t  where S_t depends on M_{t-1}
--   (S_t is computed from grad(M_{t-1}, x_t) — nonlinear in M)
-- Lattice OSR: normalization is nonlinear
-- Softmax (MEMORA): nonlinear
```

## Blelloch Algorithm (Parallel Prefix Sum)

```
ALGORITHM: associative_scan(a: &[Tensor], b: &[Tensor], s_init: &Tensor) -> Vec<Tensor>
  -- Computes s_t for all t in O(log C) parallel steps
  -- Uses the associative property: (a_i, b_i) * (a_j, b_j) = (a_i * a_j, a_i * b_j + b_i)

  C = a.len()
  -- Represent each step as a (a_t, b_t) pair
  pairs = [(a[t], b[t]) for t in 0..C]

  -- Up-sweep: combine pairs in parallel
  FOR depth = 0 to log2(C) - 1:
    stride = 2^(depth + 1)
    FOR i = stride-1, stride*2-1, ... C-1:  -- parallel
      j = i - 2^depth
      pairs[i] = combine(pairs[j], pairs[i])

  -- Down-sweep: propagate from root
  pairs[C-1] = (1, s_init)  -- identity
  FOR depth = log2(C)-1 down to 0:
    stride = 2^(depth + 1)
    FOR i = stride-1, stride*2-1, ... C-1:  -- parallel
      j = i - 2^depth
      temp = pairs[j]
      pairs[j] = pairs[i]
      pairs[i] = combine(temp, pairs[i])

  return [pairs[t].b for t in 0..C]

FUNCTION: combine(left: (Tensor, Tensor), right: (Tensor, Tensor)) -> (Tensor, Tensor)
  (a_l, b_l) = left
  (a_r, b_r) = right
  return (a_l * a_r, a_l * b_r + b_l)
```

## Application to Titans Momentum

```
-- Titans momentum (Eq 18 parallel form):
-- S_t = eta_t * S_{t-1} - theta_t * grad_t
-- This is linear recurrence: a = eta_t, b = -theta_t * grad_t

FUNCTION: parallel_momentum(etas: &[Tensor], thetas: &[Tensor],
                            grads: &[Tensor], S_init: &Tensor) -> Vec<Tensor>
  a_sequence = etas                               -- decay factors
  b_sequence = [-theta * g for (theta, g) in zip(thetas, grads)]  -- update terms
  return associative_scan(a_sequence, b_sequence, S_init)
```

## Compatibility

```
SUPPORTED_BY:
  - TitansLMM:     PARTIAL  (momentum update only, not memory update)
  - Hebbian:        YES     (M_t = (1-alpha_t)*M + v@k^T is linear in M)
  - EMA momentum:   YES     (standard EMA is a linear recurrence)
  - Linear attention: YES   (decay + additive update)
  - Mamba/SSM:      YES     (diagonal state update is linear)

NOT SUPPORTED:
  - Any rule with nonlinear state dependency
  - Lattice OSR (normalization)
  - MEMORA (softmax)
  - MLP-based rules (MONETA, YAAD) — gradient depends on state
```

## Axiom Compliance

- **NL IS #2** (parallel optimization): O(log C) parallel depth
- Complements chunkwise GD: use associative scan for momentum, chunkwise for memory
