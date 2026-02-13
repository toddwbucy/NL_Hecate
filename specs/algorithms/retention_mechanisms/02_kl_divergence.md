# KL Divergence Retention

```
CONTRACT
  Purpose:    Retention via KL divergence from previous state. Constrains
              memory to the probability simplex. Softmax emerges as a
              CONSEQUENCE of the optimization, not an architectural choice.
  Expects:    Memory on probability simplex (rows sum to 1).
  Guarantees: Memory stays on simplex. Update contains softmax naturally.
              Connections to information-theoretic optimality (minimizing
              information loss under capacity constraints).
  Cost:       O(d) per row for log + exp (softmax).
              More expensive than L2 decay per element.
  Trade-off:  Mathematically principled (softmax emerges from theory).
              But memory is constrained to distributions — cannot store
              arbitrary matrices. Trade representational power for
              theoretical elegance.
  Position:   specs/algorithms/retention_mechanisms/02_kl_divergence.md
  Source:     MIRAS (2504.13173) Eq 27, Proposition 3.1
```

## MIRAS Decomposition

```
D_t(W, W_{t-1}) = D_KL(W || W_{t-1})     -- local: stay close in KL sense
                = sum(W * log(W / W_{t-1}))
G_t(W) = -Entropy(W)                       -- global: don't spread too thin
       = sum(W * log(W))
```

## Closed-Form Solution

```
FUNCTION: kl_apply_retention(W: &Tensor, grad: &Tensor,
                             alpha_t: f32, eta_t: f32) -> Tensor
  -- MIRAS Proposition 3.1:
  -- W_new[i] = softmax(alpha_t * log(W_old[i]) - eta_t * grad[i])

  -- Per row:
  FOR each row i of W:
    logits = alpha_t * log(W[i]) - eta_t * grad[i]
    W_new[i] = softmax(logits)

  return W_new

  -- The softmax appears because:
  -- argmin_p D_KL(p || q) + <grad, p>  subject to sum(p)=1, p>=0
  -- has solution p = softmax(log(q) - grad)
  -- The probability simplex constraint + KL retention = softmax
```

## Properties

- **Simplex constraint**: Memory rows are valid probability distributions at all times.
- **Softmax emergence**: Not designed in — it falls out of the math.
- **Information-theoretic**: KL measures information lost by moving from W_{t-1} to W_new.
  Alpha controls how much information loss is tolerated per step.
- **Initialization**: Must start on simplex (uniform = max entropy starting point).
- **Numerical stability**: Requires log-space computation. log(0) must be handled (add epsilon or use log-softmax).

## Connection to Standard Attention

MIRAS Proposition 3.1 shows that standard softmax attention is a single-step
MEMORA with identity memory: softmax(Q @ K^T / sqrt(d)).
Standard attention is MEMORA with no memory accumulation — just one-step retrieval.

## Axiom Compliance

- **MIRAS IS #3** (proven optimality): Closed-form optimal under KL constraints
- **NL IS #6** (optimizers are memory): Softmax attention IS a degenerate memory rule
- CS-36 compliance: Retention is KL, not restricted to L2
