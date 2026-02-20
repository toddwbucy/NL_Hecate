# Sigmoid-Bounded Retention (Box-Constrained Memory)

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), log-barrier specialization of Bregman framework -->
```text
CONTRACT
  Purpose:    Sigmoid-bounded retention constrains memory entries to the interval
              [0, 1] using a log-barrier penalty derived from the Bregman framework.
              Where L2 decay allows unbounded memory and KL constrains to a simplex,
              sigmoid-bounded retention enforces per-element box constraints. The
              update operates in logit space — applying sigmoid to map back to [0,1]
              — so the barrier is never violated. This is useful for memories that
              function as soft gates, attention masks, or probability-like quantities
              where entries must remain bounded.
  Expects:    Memory state W in [0, 1]^{d x d}, previous state W_{t-1},
              gates (alpha_t, eta_t), gradient of loss w.r.t. W.
  Guarantees: Local retention D_t(W, W') = ||W - W'||^2_F keeps updates small.
              Global retention G_t(W) = -sum(log(W) + log(1-W)) is a log-barrier
              that repels memory entries from the boundaries 0 and 1.
              The combined update maps to logit space, applies gradient + decay,
              then maps back via sigmoid — entries never leave [0, 1].
  Cost:       O(d^2) — element-wise log, exp, sigmoid. Same asymptotic cost as
              KL retention. The logit/sigmoid operations are well-optimized on GPU.
  Trade-off:  Bounded memory prevents runaway growth (no clamping needed). But
              representational capacity is limited to [0, 1] per entry — cannot
              store arbitrary real-valued associations. Best suited for memories
              that naturally represent probabilities, gates, or soft masks.
  Position:   specs/algorithms/retention_mechanisms/08_sigmoid_bounded.md
              Child of: 06_bregman.md (Bregman divergence general framework)
              Sibling of: 02_kl_divergence.md (simplex constraint vs box constraint)
  Source:     Derived from MIRAS (2504.13173) §5.2 Bregman framework (Eq 18) +
              standard log-barrier interior point methods (Boyd & Vandenberghe 2004)
```

## Motivation

Several memory use cases require bounded entries:

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), motivation for box constraint -->
```text
-- Gate memories: alpha_t, theta_t are sigmoid outputs in [0, 1].
--   If a memory stores learned gate patterns, entries must stay bounded.
--
-- Attention masks: Soft masks indicating which positions to attend to.
--   Must be in [0, 1] to serve as multiplicative weights.
--
-- Probability memories: When each entry represents P(feature | context).
--   Must be non-negative and bounded above by 1.
--
-- Unlike KL retention (simplex: rows sum to 1), sigmoid-bounded imposes
-- INDEPENDENT per-element constraints. Each W_{jl} in [0,1] separately.
-- No normalization across rows or columns.
```

## MIRAS Decomposition

<!-- HADES: miras_equations/eq-vp-retention-decomposition (§3.3, applied to sigmoid-bounded) -->
```text
Ret_t(W, W_{t-1}) = (1/eta_t) * D_t(W, W_{t-1})  +  (1/alpha_t) * G_t(W)
                     |---- Local Retention ----|     |-- Global Retention --|

-- Local: D_t(W, W') = ||W - W'||^2_F
--   Standard L2 proximity — same as L2 weight decay and L_q norm.
--   Controls per-step deviation from previous memory.

-- Global: G_t(W) = -sum_{j,l} (log(W_{jl}) + log(1 - W_{jl}))
--   Log-barrier function for the box constraint [0, 1].
--   As W_{jl} → 0: -log(W_{jl}) → +inf (repels from 0)
--   As W_{jl} → 1: -log(1 - W_{jl}) → +inf (repels from 1)
--   At W_{jl} = 0.5: minimum of barrier (max entropy point)
--   The barrier keeps entries STRICTLY inside (0, 1) — never on boundary.
```

## The Log-Barrier Generator

Sigmoid-bounded retention is a Bregman divergence with the negative entropy
of a Bernoulli distribution as the generator:

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), log-barrier specialization -->
```text
-- Generator function (negative binary entropy):
phi(W) = sum_{j,l} (W_{jl} * log(W_{jl}) + (1 - W_{jl}) * log(1 - W_{jl}))

-- This is strictly convex on (0, 1) with:
--   phi'(w) = log(w) - log(1 - w) = logit(w)
--   phi''(w) = 1/(w(1-w)) > 0  (strictly convex)

-- The induced Bregman divergence:
D_phi(W, W') = sum_{j,l} (W_{jl} * log(W_{jl}/W'_{jl})
                          + (1 - W_{jl}) * log((1 - W_{jl})/(1 - W'_{jl})))

-- This is the KL divergence between Bernoulli(W_{jl}) and Bernoulli(W'_{jl}).
-- Each element is treated as an independent Bernoulli parameter.
```

## Update Rule (Logit-Space)

The update operates in logit space where the constraint is automatically
satisfied:

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), sigmoid-bounded specialization -->
```text
-- Sigmoid-bounded retention update:
FUNCTION: sigmoid_bounded_step(W: &mut Tensor, grad: &Tensor,
                                alpha_t: f32, eta_t: f32) -> ()
  -- W: memory state in (0, 1)^{d x d}
  -- grad: gradient of loss w.r.t. W

  -- Step 1: Map to logit space (unconstrained)
  Z = logit(W)                    -- Z = log(W / (1 - W)), Z in (-inf, +inf)

  -- Step 2: Compute logit-space gradient (chain rule: dL/dZ = dL/dW * dW/dZ)
  grad_logit = grad * W * (1 - W)   -- dW/dZ = sigmoid(Z)*(1-sigmoid(Z)) = W*(1-W)

  -- Step 3: Decay + gradient step in logit space
  Z = (1 - alpha_t) * Z - eta_t * grad_logit

  -- alpha_t: retention gate (same as L2 decay). Controls logit-space decay.
  --   alpha_t → 0: full retention (Z unchanged). alpha_t → 1: full decay (Z → 0).
  -- eta_t: learning rate gate. Scales the logit-space gradient.

  -- Step 4: Map back to (0, 1)
  W = sigmoid(Z)                  -- W = 1 / (1 + exp(-Z))

  -- Properties:
  --   sigmoid maps R → (0, 1), so W stays bounded regardless of Z.
  --   Decay in logit space: (1-alpha_t) * Z shrinks logits toward 0,
  --   which maps to W → 0.5 (the maximum-entropy point).
  --   This is analogous to how L2 decay shrinks W toward 0,
  --   but here the "zero" of logit space is W = 0.5.
```

## Comparison with Other Bounded Retentions

<!-- HADES: Derived from miras_equations/eq-vp-retention-decomposition (§3.3), constraint comparison -->
```text
| Retention        | Constraint Set           | Decay Target | Mechanism        |
|------------------|--------------------------|--------------|------------------|
| L2 weight decay  | R^{d x d} (unbounded)    | W → 0        | Scalar multiply  |
| KL divergence    | Simplex (rows sum to c)  | W → uniform  | Softmax          |
| Sphere norm      | S^{d-1} (unit sphere)    | Emergent     | Normalization    |
| L_q norm         | Bounded L_q ball         | W → 0        | q-normalization  |
| Sigmoid-bounded  | [0, 1]^{d x d} (box)    | W → 0.5      | Logit + sigmoid  |

-- Key distinction from KL:
--   KL: rows are coupled (must sum to c). Interdependent.
--   Sigmoid: elements are independent. Each W_{jl} in [0,1] separately.
--   KL redistributes within rows. Sigmoid constrains each element alone.
--
-- Key distinction from L2:
--   L2 decay target is W = 0 (off). High retention → small W.
--   Sigmoid decay target is W = 0.5 (uncertain). High retention → W → 0.5.
--   This makes sigmoid retention natural for uncertainty-tracking memories.
```

## Gradient Derivation (for tape integration)

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), analytical VJP for sigmoid-bounded -->
```text
-- Forward:
--   Z_t = logit(W_{t-1})                          -- to logit space
--   grad_logit_t = grad_t ⊙ W_{t-1} ⊙ (1 - W_{t-1})
--   Z_t = (1 - alpha_t) * Z_t - eta_t * grad_logit_t
--   W_t = sigmoid(Z_t)                            -- back to [0,1]

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dW_{t-1}, dL/dgrad_t, dL/dalpha_t, dL/deta_t

-- Step 1: Backward through sigmoid
--   dL/dZ_t = dL/dW_t ⊙ W_t ⊙ (1 - W_t)        (⊙ = element-wise product)

-- Step 2: Backward through logit-space update
--   Let g_logit = grad_t ⊙ W_{t-1} ⊙ (1 - W_{t-1})  (the logit-space gradient)
--   dL/dZ_{prev} = (1 - alpha_t) * dL/dZ_t        (through decay)
--   dL/dg_logit = -eta_t * dL/dZ_t               (through gradient step)

-- Step 3: Backward through logit
--   dL/dW_{t-1} (through Z_prev) = dL/dZ_{prev} / (W_{t-1} ⊙ (1 - W_{t-1}))
--   dL/dW_{t-1} (through g_logit) = dL/dg_logit ⊙ (grad_t ⊙ (1 - 2*W_{t-1}))
--   dL/dW_{t-1} = sum of both terms

-- Gate gradients (scalars):
dL/dalpha_t = -trace(Z_{prev}^T @ dL/dZ_t)
dL/deta_t = -trace(g_logit^T @ dL/dZ_t)
```

## Properties

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), sigmoid-bounded properties -->
```text
-- Automatic boundedness:
--   No clamping needed. sigmoid(Z) is in (0, 1) for all Z in R.
--   Even with large gradients or aggressive learning rates,
--   memory entries cannot escape the box constraint.

-- Maximum entropy equilibrium:
--   Without gradient input, decay drives Z → 0, hence W → 0.5.
--   This represents maximum uncertainty (Bernoulli entropy maximized at p=0.5).
--   Memory "forgets" toward uncertainty, not toward zero.

-- Gradient scaling near boundaries:
--   The logit transform compresses gradients near 0 and 1.
--   dW/dZ = W(1-W) → 0 as W → 0 or W → 1.
--   This is a natural learning rate annealing near boundaries:
--   entries near saturation (0 or 1) change more slowly.
--   Entries near 0.5 (uncertain) change more quickly.

-- Connection to Hopfield energy:
--   The log-barrier sum(W*log(W) + (1-W)*log(1-W)) is the negative
--   binary entropy. Minimizing it (the decay target) maximizes entropy.
--   This connects to information-theoretic forgetting: decay erases
--   information by moving toward the maximum-entropy state.
```

## Implementation Notes

<!-- HADES: Derived from miras_equations/eq-018-f-divergence-update (§5.2 Eq 18), implementation guidance for log-barrier specialization -->

1. **Novel retention mechanism**: Sigmoid-bounded retention does not correspond
   to a named variant in the MIRAS paper. It is derived from the Bregman
   framework (06_bregman.md) using the negative binary entropy generator. The
   MIRAS §5.2 f-divergence machinery (Eq 18) provides the mathematical
   foundation; this spec instantiates it for the box constraint.

2. **Initialization**: Memory must start in (0, 1). Initialize to W = 0.5
   (maximum entropy, no information). Avoid exact 0 or 1 — logit is undefined
   at boundaries. Use `W_init = sigmoid(0) = 0.5` or small perturbation.

3. **Numerical stability**: `logit(W)` requires W in (eps, 1-eps). Clamp W
   before logit: `W_clamped = clamp(W, 1e-6, 1 - 1e-6)`. The sigmoid output
   is always in (0, 1), so clamping is only needed at initialization or after
   checkpoint restore.

4. **Logit-space operations**: Working in Z = logit(W) space throughout the
   forward pass avoids repeated logit/sigmoid conversions. Store Z as the
   primary `inner_loop_state`, compute W = sigmoid(Z) only for the memory
   query M(W, k) = W @ k. This is analogous to log-space computation for KL.

5. **Pluggable retention dispatch**: Registers as
   `RetentionKind::SigmoidBounded` in the S3b-M1 infrastructure. No
   additional parameters beyond the standard gates (alpha_t, eta_t).

6. **Interaction with CMS (Continuous Memory System)**: At slow CMS levels
   (see `specs/infrastructure/scheduling/00_conductor.md`), the decay toward
   W = 0.5 is slower, allowing slow-frequency memories to maintain stronger
   commitments (entries closer to 0 or 1) for longer. Fast levels decay
   quickly toward uncertainty.

## Axiom Compliance

- **NL IS #4** (compressing context): The box constraint bounds total information capacity per entry to 1 bit, forcing efficient use of the memory matrix.
- **NL IS #9** (principled not ad hoc): The sigmoid update emerges from the Bregman framework with a well-defined convex generator (negative binary entropy), not from an architectural choice to "just clamp."
- **MIRAS IS #1** (orthogonal design choices): Sigmoid-bounded retention is independent of the memory structure and attentional bias — any rule can use it when bounded entries are appropriate.
