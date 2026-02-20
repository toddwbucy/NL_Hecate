# Bregman Divergence Retention (General Framework)

```text
CONTRACT
  Purpose:    Bregman divergence is the most general retention mechanism in the
              MIRAS framework. Every retention function used in practice (L2 decay,
              KL divergence, elastic net, L_q norm) is a special case of a Bregman
              divergence induced by a convex generator. This spec documents the
              general framework and the f-divergence specialization that produces
              constrained (simplex) memory updates.
  Expects:    Generator function f (strictly convex, f(1) = 0 for f-divergence),
              memory state W, previous state W_{t-1}, learning rate eta_t,
              retention strength alpha_t.
  Guarantees: The retention decomposition Ret = Local + Global always holds.
              Local retention D_t(W, W_{t-1}) penalizes deviation from previous state.
              Global retention G_t(W) constrains overall memory size/structure.
              Different generators f produce different retention behaviors as
              closed-form special cases.
  Cost:       Depends on f. Quadratic f → O(d^2) (element-wise). KL f → O(d^2)
              (element-wise exp/log). Elastic net → O(d^2) (soft thresholding).
              All are element-wise operations — negligible next to matmuls.
  Trade-off:  More expressive retention (e.g., KL, Bregman) can enforce useful
              constraints (probability simplex, sparsity) but introduces
              nonlinearity in the update, which may complicate parallelization.
  Position:   specs/algorithms/retention_mechanisms/06_bregman.md
              Parent of: KL (existing MEMORA), elastic net (FTRL spec), L_q (S3b-S7),
              sigmoid-bounded (S3b-S8)
  Source:     MIRAS (2504.13173) §3.3, §5.2 Eqs 18-23; Proposition 3.2
```

## Retention Decomposition

Every retention function in MIRAS decomposes into two components:

<!-- HADES: miras_equations/eq-vp-retention-decomposition (§3.3) -->
```text
Ret_t(W, W_{t-1}) = (1/eta_t) * D_t(W, W_{t-1})  +  (1/alpha_t) * G_t(W)
                     |---- Local Retention ----|     |-- Global Retention --|

-- Local Retention D_t(W, W_{t-1}):
--   A premetric measuring deviation from previous state.
--   Controls HOW MUCH the memory can change per token.
--   eta_t = meta in-context learning rate:
--     large eta_t → learn more new, forget more old
--     small eta_t → conservative, retain more

-- Global Retention G_t(W):
--   Constrains the overall memory structure/size.
--   Independent of W_{t-1} — applies regardless of history.
--   alpha_t controls strength:
--     large alpha_t → weak global constraint
--     small alpha_t → strong global constraint (smaller memory)
```

## The Bregman Divergence

Given a strictly convex, differentiable generator function phi:

```text
-- Bregman divergence:
D_phi(W, W') = phi(W) - phi(W') - <nabla phi(W'), W - W'>

-- Properties:
--   D_phi(W, W') >= 0          (non-negativity)
--   D_phi(W, W') = 0 iff W=W'  (identity of indiscernibles)
--   NOT symmetric in general: D_phi(W, W') != D_phi(W', W)
--   NOT a true metric — but sufficient for retention

-- Special cases by generator:
--   phi(W) = (1/2)||W||^2_F    → D_phi = (1/2)||W - W'||^2_F  (L2, standard)
--   phi(W) = sum W log(W)      → D_phi = KL(W || W')           (KL divergence)
--   phi(W) = (1/q)||W||^q_q    → D_phi = L_q Bregman           (L_q norm)
```

## Connection to FTRL (Proposition 3.2)

Bregman divergence is the bridge between FTRL and Learning-Retaining viewpoints:

<!-- HADES: miras_definitions/prop-3-2-viewpoint-equivalence (§3.2, Proposition 3.2) -->
```text
-- Proposition 3.2:
-- Define h_t(W) = sum l_tilde_i(W) + (1/eta) * R(W)
-- Let D_h = Bregman divergence induced by h_t
-- Set Ret_t(W, W') = D_h(W, W') in Learning-Retaining

-- THEN: Learning-Retaining = FTRL

-- The Bregman divergence "converts" the FTRL regularizer R(W)
-- into a Learning-Retaining retention function.
-- This is why Bregman is the most general retention: it
-- encompasses ANY convex regularizer from the FTRL viewpoint.
```

## f-Divergence Retention (Constrained Memory)

When the memory must lie on a probability simplex (non-negative, ||W||_1 = c),
use f-divergence as the local retention:

<!-- HADES: miras_equations/eq-018-f-divergence-update (§5.2 Eq 18) -->
```text
-- f-divergence local retention:
D_t(W, W') = sum W'_{jl} * f(W_{jl} / W'_{jl})

-- f: strictly convex with f(1) = 0
-- g(.): inverse of f'(.)

-- General update rule (MIRAS Eq 18):
W_t = W_{t-1} * g(-zeta_t - eta_t * grad l(W_{t-1}; k_t, v_t))

-- where * is element-wise (Hadamard) product
-- zeta_t: chosen so that ||W_t||_1 = c (simplex projection)
-- g(.): the nonlinearity induced by the divergence choice

-- Special cases:
--   f(tau) = tau^2/2     → g = identity → standard GD (no nonlinearity)
--   f(tau) = tau ln(tau)  → g = exp      → KL retention (Softmax update)
--   f(tau) = |tau - 1|^p  → g = sign*|.|^{1/(p-1)} → L_p retention
```

## KL Divergence Retention (MEMORA)

The most important f-divergence specialization:

<!-- HADES: miras_equations/eq-021-kl-softmax-update (§5.2 Eq 21) -->
```text
-- KL retention update (MIRAS Eq 21):
FUNCTION: kl_retention_step(W: &mut Tensor, grad: &Tensor,
                             alpha_t: f32, eta_t: f32, c: f32) -> ()
  -- Derived from f(tau) = tau ln(tau) → g(.) = exp(.)
  lambda_t = (1/alpha_t) / (1/alpha_t + 1/eta_t)    -- in (0, 1)
  eta_prime = 1.0 / (1/alpha_t + 1/eta_t)           -- in R+

  -- Softmax update (naturally constrains to simplex)
  logits = (1.0 - lambda_t) * log(W) - eta_prime * grad
  W = c * softmax(logits)

  -- Properties:
  --   W >= 0 (softmax output is non-negative)
  --   ||W||_1 = c (softmax sums to 1, scaled by c)
  --   KL retention acts as multiplicative decay (via log space)
```

This is the retention mechanism used by MEMORA (MIRAS Eq 27).

## Elastic Net Retention (Sparse Memory)

Combines L2 (local) + L1 (global) for sparse memory:

<!-- HADES: miras_equations/eq-022-elastic-net-soft-threshold (§5.2 Eq 22) -->
```text
-- Elastic net retention (MIRAS Eq 22):
FUNCTION: elastic_net_step(W: &mut Tensor, grad: &Tensor,
                            eta: f32, alpha: f32, beta: f32) -> ()
  -- gamma = eta*beta / (alpha*(eta+beta))   -- soft threshold
  -- lambda = beta / (beta+eta)              -- decay factor in (0,1)
  -- zeta = eta * lambda                     -- effective LR

  -- Two-stage update: decay + threshold
  z = lambda * W - zeta * grad               -- soft decay + gradient
  W = sign(z) * max(abs(z) - gamma, 0)       -- soft thresholding

  -- Properties:
  --   |W_{jl}| < gamma → zeroed (hard sparsity)
  --   |W_{jl}| >= gamma → shrunk toward zero (soft decay)
  --   Combines soft forgetting (lambda < 1) with hard forgetting (threshold)
```

## Retention Taxonomy

```text
| Retention       | Local D_t           | Global G_t          | Constraint   | MIRAS Eq |
|-----------------|---------------------|---------------------|--------------|----------|
| L2 decay        | ||W - alpha*W'||^2  | —                   | None         | Eq 8-9   |
| L_q norm        | ||W - W'||^2        | ||W||_q^q           | Bounded norm | Eq 24-25 |
| KL divergence   | KL(W || W')         | Shannon entropy     | Simplex      | Eq 21    |
| Elastic net     | ||W - W'||^2        | ||W||^2 + ||W||_1   | Sparse       | Eq 22    |
| Sigmoid-bounded | ||W - W'||^2        | sigmoid barrier     | [0, 1]       | S3b-S8   |
| General Bregman | D_phi(W, W')        | phi(W)              | Depends on f | Eq 18    |
```

## Gradient Derivation (for tape integration)

### f-Divergence Update

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), analytical VJP -->
```text
-- Forward: W_t = W_{t-1} * g(-zeta_t - eta_t * grad_t)
-- Let h_t = -zeta_t - eta_t * grad_t  (the argument to g)

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dW_{t-1}, dL/dgrad_t, dL/dzeta_t

dL/dW_{t-1} = dL/dW_t * g(h_t)
             + dL/dW_t * W_{t-1} * g'(h_t) * d(h_t)/dW_{t-1}
             -- First term: through the Hadamard product
             -- Second term: through h_t's dependence on W_{t-1} via grad

dL/dgrad_t = -eta_t * dL/dW_t * W_{t-1} * g'(h_t)
            -- Through the learning rate scaling in h_t

dL/dzeta_t = -sum(dL/dW_t * W_{t-1} * g'(h_t))
            -- Scalar: how the simplex projection parameter affects loss
```

### KL (Softmax) Update

```text
-- Forward: W_t = c * softmax((1-lambda) * log(W_{t-1}) - eta' * grad_t)
-- The softmax Jacobian is well-known:
--   d(softmax(x))_i / dx_j = softmax(x)_i * (delta_ij - softmax(x)_j)
-- Standard softmax backward — the tape handles this natively.
```

## Implementation Notes

1. **Start with existing variants**: L2 decay is already implemented. KL is
   used by MEMORA (already implemented). Elastic net comes from FTRL spec.
   This spec provides the unifying framework — no new Rust needed until
   we implement a truly novel generator f.

2. **Simplex projection**: The zeta_t parameter in Eq 18 enforces ||W||_1 = c.
   For KL (softmax), this is automatic. For general f-divergence, computing
   zeta_t requires a 1D root-finding step — bisection on the constraint.

3. **Numerical stability**: KL retention requires log(W_{t-1}), which is
   undefined for W = 0. Clamp W to [eps, inf) before taking log. The softmax
   output is always positive, so this is only an initialization concern.

4. **Element-wise operations**: All Bregman/f-divergence updates are element-wise
   (Hadamard products, element-wise g(.)). These are embarrassingly parallel
   and have no interaction with the chunkwise parallelization strategy.

5. **Pluggable retention dispatch**: The S3-M1 (pluggable retention) milestone
   provides the dispatch infrastructure. Each Bregman variant registers as a
   `RetentionKind` enum value. This spec fills the mathematical content that
   S3-M1's infrastructure dispatches to.

## Axiom Compliance

- **NL IS #4** (compressing context): Bregman retention controls memory compression — different divergences compress differently (L2 shrinks, KL redistributes, elastic net sparsifies).
- **NL IS #9** (principled not ad hoc): Every retention is derived from a convex generator, not hand-tuned. The framework guarantees consistency.
- **NL IS #6** (optimizers are associative memory): Retention IS the memory's forgetting policy — it determines what the memory retains and what it discards.
