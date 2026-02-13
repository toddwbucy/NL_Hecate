# Elastic Net Retention

```
CONTRACT
  Purpose:    Combined L1 + L2 retention. L1 provides hard forgetting
              (drives small associations to exactly zero). L2 provides
              soft forgetting (continuous decay). Together: selective
              retention — important associations persist, unimportant
              ones are pruned.
  Expects:    Memory state W, boundary state W_prev, L1/L2 strengths.
  Guarantees: Soft thresholding + exponential decay.
              Small associations go to exactly zero (sparse memory).
              Large associations decay slowly (important memories persist).
  Cost:       O(1) per element — soft threshold + multiply.
  Trade-off:  More expressive than L2 (selective). More interpretable
              than KL (explicit sparsity vs spread). But L1 gradient
              is discontinuous at zero — requires careful handling.
  Position:   specs/algorithms/retention_mechanisms/03_elastic_net.md
  Source:     MIRAS (2504.13173) general framework
```

## MIRAS Decomposition

```
D_t(W, W_{t-1}) = ||W - W_{t-1}||^2                  -- local: L2
G_t(W) = lambda_1 * ||W||_1 + lambda_2 * ||W||^2     -- global: L1 + L2

-- L1 term promotes sparsity (drives small values to zero)
-- L2 term prevents any single element from growing too large
```

## Closed-Form Solution

```
FUNCTION: elastic_net_apply_retention(W: &Tensor, alpha_t: f32,
                                      lambda_1: f32, lambda_2: f32) -> Tensor
  -- Step 1: L2 decay (same as l2_weight_decay)
  W_decayed = (1.0 - alpha_t * lambda_2) * W

  -- Step 2: Soft thresholding (from L1 term)
  -- S_lambda(x) = sign(x) * max(|x| - lambda, 0)
  threshold = alpha_t * lambda_1
  W_new = sign(W_decayed) * max(abs(W_decayed) - threshold, 0.0)

  return W_new
```

## Soft Thresholding Visualization

```
-- Input value:   ...-0.3  -0.2  -0.1  0.0  0.1  0.2  0.3...
-- After L2 decay: scaled by (1-alpha*lambda_2)
-- After L1 threshold (lambda=0.15):
--                 ...-0.15 -0.05  0.0  0.0  0.0  0.05 0.15...
-- Small values (|x| < threshold) go to EXACTLY zero.
-- Large values are shifted toward zero by threshold amount.
```

## When to Use Elastic Net

- When memory should be SPARSE (many zero entries)
- When you want to distinguish "important" from "noise" associations
- When L2 alone retains too much noise (everything decays but nothing is zeroed)
- Useful at slow-frequency CMS levels where memory should accumulate only the most persistent patterns

## Axiom Compliance

- CS-36 compliance: Uses L1+L2, not restricted to L2 only
- **NL IS #4** (compressing context): L1 actively compresses by zeroing small associations
