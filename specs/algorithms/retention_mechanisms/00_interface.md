# Retention Mechanism Interface

```
CONTRACT
  Purpose:    Defines MIRAS Knob #3 — how a memory update rule controls
              what it retains vs what becomes inaccessible. Every retention
              mechanism plugs into the same slot in the MemoryUpdateRule trait.
  Expects:    Current memory state, previous memory state (chunk boundary),
              data-dependent control signals.
  Guarantees: Returns either a retention penalty (for the objective) or
              a decayed state (closed-form result). The two forms are equivalent.
  Cost:       Zero — trait definition. Implementor costs vary.
  Trade-off:  Expressiveness vs compute. L2 decay is cheapest but least
              flexible. KL divergence is most mathematically elegant but
              constrains memory to probability simplex. f-divergence
              subsumes all others but is hardest to implement.
  Position:   specs/algorithms/retention_mechanisms/00_interface.md
              Parent of all retention mechanism implementations.
  Source:     MIRAS (2504.13173) Section 3.2, def-retention-gate, eq-vp-retention-decomposition
```

## Key Reframing: Retention, Not Forgetting

From MIRAS def-retention-gate:
There is no memory erasing or forgetting. Memories become inaccessible
due to retrieval failure.

The retention gate controls ACCESSIBILITY, not existence:
- "Forgetting" implies deletion (irreversible)
- "Retention" implies prioritization (memories are still there, harder to retrieve)

In the math, these are identical. But the framing changes implementation:
you never zero out state, you decay it. The information is asymptotically
unreachable, not deleted.

## The MIRAS Retention Decomposition

Every retention mechanism decomposes into two orthogonal forces:

```
Ret_t(W, W_{t-1}) = (1/eta_t) * D_t(W, W_{t-1})  +  (1/alpha_t) * G_t(W)
                     |___________________________|     |__________________|
                          LOCAL RETENTION                GLOBAL RETENTION
                     "stay close to previous"          "keep memory bounded"
```

- **D_t (Local)**: Penalizes deviation from the previous state W_{t-1}
- **G_t (Global)**: Penalizes the overall size/complexity of memory

## Trait Definition

```
TRAIT: RetentionMechanism

  COMPUTE_RETENTION(W_current: &Tensor, W_previous: &Tensor,
                    gates: &RetentionGates) -> Scalar
    Returns the retention penalty to add to the memory update objective.
    Used in the optimization form.

  APPLY_RETENTION(W_previous: &Tensor, gates: &RetentionGates) -> Tensor
    Returns the decayed state. The "effective previous state."
    Used in the closed-form update.

  -- The two forms are equivalent:
  -- COMPUTE_RETENTION feeds into the optimization objective
  -- APPLY_RETENTION is the closed-form result of that optimization
```

## Control Signals

```
FUNCTION: compute_retention_gates(x_t: &Tensor, gate_params: &RetentionGateParams) -> RetentionGates
  -- gate_params are outer_loop_param (learned via Enzyme AD)
  -- gate VALUES are data-dependent (the input decides how much to retain)
  alpha_t = sigmoid(x_t @ gate_params.W_alpha^T)    -- local retention [0,1]
  eta_t   = sigmoid(x_t @ gate_params.W_eta^T)      -- global retention [0,1]
  return RetentionGates { alpha: alpha_t, eta: eta_t }

  -- Both gates are data-dependent. This is the key distinction from
  -- fixed-schedule retention (e.g., constant weight decay in SGD).
  -- CS-39: Learnable decay parameters must be clamped.
```

## Named Configurations

| Retention | D_t (Local) | G_t (Global) | Closed Form | File |
|---|---|---|---|---|
| L2 Weight Decay | \|\|W-W_{t-1}\|\|^2 | \|\|W\|\|^2 | (1-alpha)W_{t-1} | 01_l2_weight_decay.md |
| KL Divergence | KL(W\|\|W_{t-1}) | Entropy(W) | softmax(alpha*log(W_{t-1})-eta*grad) | 02_kl_divergence.md |
| Elastic Net | \|\|W-W_{t-1}\|\|^2 | L1+L2 | soft_threshold + decay | 03_elastic_net.md |
| f-Divergence | D_f(W\|\|W_{t-1}) | depends on f | varies | 04_f_divergence.md |
| Sphere Norm | implicit | unit norm | beta*(s+Delta_s) | 05_sphere_normalization.md |

## What This Interface Does NOT Specify

- **Loss function**: Knob #2 (attentional bias) is separate
- **Algorithm**: Knob #4 (GD, OGD, etc.) — retention feeds into objective, algorithm solves it
- **Memory structure**: Knob #1 — retention works on any differentiable state
