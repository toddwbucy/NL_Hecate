# L2 Weight Decay Retention

```
CONTRACT
  Purpose:    Exponential decay of memory toward zero. The default retention
              mechanism for Titans, Atlas, and Trellis. Simplest, fastest,
              most widely used.
  Expects:    Memory state W, data-dependent alpha gate.
  Guarantees: Decayed state (1-alpha)*W. Exponential forgetting curve.
              After n steps without reinforcement, signal = (1-alpha)^n * original.
  Cost:       O(1) per element — single multiply.
  Trade-off:  Cheapest and simplest. But one-size-fits-all: same decay rate
              for all associations. Cannot selectively retain important
              associations while forgetting unimportant ones (elastic net can).
  Position:   specs/algorithms/retention_mechanisms/01_l2_weight_decay.md
  Source:     MIRAS (2504.13173) Table 1; Titans (2501.00663) Eqs 13, 32
```

## MIRAS Decomposition

```
D_t(W, W_{t-1}) = ||W - W_{t-1}||^2     -- local: stay close to previous
G_t(W) = ||W||^2                          -- global: keep weights small

Ret_t = (1/eta_t) * ||W - W_{t-1}||^2 + (1/alpha_t) * ||W||^2
```

## Closed-Form Solution

```
FUNCTION: l2_apply_retention(W: &Tensor, alpha_t: f32) -> Tensor
  -- alpha_t in [0,1]: 0 = full retain, 1 = full forget
  return (1.0 - alpha_t) * W

  -- Derivation:
  -- Minimize ||W_new - W_old||^2 + lambda * ||W_new||^2 + loss(W_new)
  -- Setting gradient to zero gives: W_new = (1-alpha) * W_old - lr * grad_loss
  -- The (1-alpha) factor IS the retention, applied BEFORE the update.
```

## As Optimization Penalty

```
FUNCTION: l2_compute_retention(W_current: &Tensor, W_boundary: &Tensor,
                                alpha_t: f32) -> Scalar
  local_penalty = sum((W_current - W_boundary)^2)
  global_penalty = sum(W_current^2)
  return (1.0 / (1.0 - alpha_t)) * local_penalty + alpha_t * global_penalty
```

## Properties

- **Exponential forgetting**: Signal decays as (1-alpha)^n. Half-life = -ln(2)/ln(1-alpha).
- **Data-dependent**: alpha_t is learned (outer loop) and input-dependent (inner loop).
  The INPUT decides the forgetting rate. (CS-39: must be clamped to [0,1].)
- **Uniform**: Same rate for all memory elements. Cannot distinguish important from unimportant.
- **Parallelization-friendly**: Decay products form a geometric series. Chunkwise parallel
  form pre-computes cumulative products.

## Axiom Compliance

- **NL IS NOT #4** (not discrete): Continuous exponential decay, not slot-based eviction
