# AdaMuon (Adam + Muon)

<!-- HADES: atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32); atlas_equations/eq-033-atlas-momentum (Atlas Eq 33); hope_equations/eq-042-muon-optimizer (§4.3 Eq 42); hope_equations/eq-044-newton-schulz-iteration (§4.3 Eq 44) -->
```text
CONTRACT
  Purpose:    AdaMuon combines Adam's per-element second-moment estimate with
              Muon's Newton-Schulz orthogonalization of the first moment. Adam
              normalizes per dimension (diagonal preconditioning). Muon normalizes
              the full gradient matrix (matrix preconditioning). AdaMuon does BOTH:
              it uses the second moment v for per-element scaling AND Newton-Schulz
              for directional normalization of the first moment.

              In the NL hierarchy (01_momentum.md): AdaMuon is the most expressive
              element-wise + matrix optimizer. Adam captures magnitude information
              (how big gradients are per dimension). Muon captures directional
              information (which directions gradients point). AdaMuon captures both.

              Atlas (2505.23735) uses Muon as the inner-loop momentum optimizer
              (Eq 32) for the Omega rule memory update. AdaMuon extends this to
              the outer loop where the second-moment estimate provides additional
              stability for training on noisy gradient streams across CMS levels.
  Expects:    Gradient from tape backward. First-moment buffer S [d_out, d_in]
              (matrix-valued). Second-moment buffer v [d_out * d_in] (element-wise,
              flattened). Newton-Schulz iteration count k_ns (typically 5).
              Pulse from Conductor for frequency gating.
  Guarantees: The first moment is orthogonalized via Newton-Schulz, preserving
              gradient direction while normalizing magnitude. The second moment
              provides per-element adaptive learning rates. Combined update:
              theta -= lr * NS_k(S_hat) / (sqrt(v_hat) + eps).
              When k_ns = 0: reduces to Adam (no orthogonalization).
              When v = 1 everywhere: reduces to Muon (no per-element scaling).
  Cost:       O(d^2 * k_ns) per step for Newton-Schulz (k_ns matrix multiplies).
              O(d) for second-moment update (element-wise, same as Adam).
              Total: O(d^2 * k_ns + d) ≈ O(d^2 * k_ns) since k_ns ≥ 1.
              This is k_ns × more expensive than Adam per parameter update.
              With k_ns = 5: 5× the cost of Adam for matrix-shaped parameters.
  Trade-off:  More expressive than Adam (captures cross-dimension correlations)
              but significantly more expensive. Best suited for large matrix-valued
              parameters (W_K, W_V, projections) where directional information
              matters. For bias vectors and small parameters, Adam suffices.
              CMS frequency gating amortizes the cost: slow levels pay the NS cost
              less frequently.
  Position:   specs/algorithms/optimization_machinery/09_adamuon.md
              Sibling of: 08_adamw_outer.md (AdamW), 01_momentum.md (hierarchy)
  Source:     Atlas (2505.23735) Eqs 32-33 (Muon in memory update);
              HOPE (2512.24695) §4.3 Eqs 42-44 (Muon, Newton-Schulz, objective);
              Jordan et al. 2024 (original Muon)
```

## Muon: Matrix-Valued Momentum with Orthogonalization

Muon replaces the standard EMA momentum with Newton-Schulz orthogonalization:

<!-- HADES: hope_equations/eq-042-muon-optimizer (§4.3 Eq 42); hope_equations/eq-044-newton-schulz-iteration (§4.3 Eq 44) -->
```text
-- Muon optimizer (HOPE Eq 42):
--   S_t = beta * S_{t-1} + (1 - beta) * grad_t    -- EMA momentum (matrix-valued)
--   update = NewtonSchulz_k(S_t)                    -- orthogonalize
--   theta -= lr * update
--
-- Newton-Schulz iteration (HOPE Eq 44):
FUNCTION: newton_schulz_k(S: &Tensor, k: usize) -> Tensor
  -- Maps momentum matrix to nearest orthogonal matrix
  -- Solves: min_X ||X^T X - I||_F  subject to column space of X = column space of S
  X = S / max(frobenius_norm(S), eps)   -- initial normalization
  FOR iter in 0..k:
    X = 0.5 * X @ (3 * I - X^T @ X)
  return X

-- Why this works (HOPE Eq 43, orthogonalization objective):
--   J(P) = ||P(g)^T P(g) - I||  — measures deviation from orthogonality
--   Newton-Schulz is gradient descent on J, converging in ~5 iterations
--   for well-conditioned input.
--
-- Why orthogonalize momentum?
--   SGD: follows raw gradient (noisy magnitude + direction)
--   Adam: normalizes per-element (fixes magnitude, keeps noisy direction)
--   Muon: normalizes the full matrix (fixes both magnitude AND direction)
--   The gradient DIRECTION is what matters for optimization.
--   Magnitude is noise. Muon strips the noise while preserving the signal.
```

## AdaMuon: Combining Both

AdaMuon fuses the two preconditioning strategies:

<!-- HADES: Derived from hope_equations/eq-042-muon-optimizer (§4.3 Eq 42); nl_optimizers/opt-adam (§4.2), combined preconditioning -->
```text
FUNCTION: adamuon_step(theta: &mut Tensor,        -- [d_out, d_in] parameter matrix
                       grad: &Tensor,              -- [d_out, d_in] gradient
                       s: &mut Tensor,             -- [d_out, d_in] first moment (matrix)
                       v: &mut Vec<f32>,           -- [d_out * d_in] second moment (flat)
                       t: &mut u32,                -- step counter
                       lr: f32, beta1: f32, beta2: f32,
                       eps: f32, weight_decay: f32,
                       k_ns: usize) -> ()

  *t += 1

  -- Step 1: Update first moment (matrix-valued EMA)
  s = beta1 * s + (1 - beta1) * grad            -- [d_out, d_in]

  -- Step 2: Update second moment (element-wise EMA)
  FOR i in 0..(d_out * d_in):
    v[i] = beta2 * v[i] + (1 - beta2) * grad_flat[i] * grad_flat[i]

  -- Step 3: Bias correction
  bc1 = 1 - beta1^(*t)
  bc2 = 1 - beta2^(*t)
  s_hat = s / bc1                                -- [d_out, d_in]
  -- v_hat computed element-wise below

  -- Step 4: Newton-Schulz orthogonalization of first moment
  ns_update = newton_schulz_k(s_hat, k_ns)       -- [d_out, d_in]

  -- Step 5: Combined update (element-wise division)
  FOR i in 0..(d_out * d_in):
    v_hat_i = v[i] / bc2
    -- Muon-normalized direction / Adam-normalized scale
    theta_flat[i] -= lr * (ns_update_flat[i] / (sqrt(v_hat_i) + eps)
                          + weight_decay * theta_flat[i])

-- Interpretation:
--   ns_update provides the DIRECTION (orthogonalized, cross-dim correlations)
--   1/sqrt(v_hat) provides the SCALE (per-element, adaptive)
--   The combined update follows the optimal direction at the optimal per-element rate.
--
-- Degenerate cases:
--   k_ns = 0: ns_update = s_hat (no orthogonalization) → standard Adam
--   beta2 = 0: v = grad^2 (no history) → single-step Muon with per-element scaling
--   weight_decay = 0: no decoupled regularization
```

## Atlas Inner-Loop Usage (Eq 32-33)

In Atlas, Muon (without the Adam second moment) is used as the inner-loop
momentum optimizer for the Omega rule:

<!-- HADES: atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32); atlas_equations/eq-033-atlas-momentum (Atlas Eq 33) -->
```text
-- Atlas memory update (Eq 32):
--   M_t = alpha_t * M_{t-1} - eta_t * NewtonSchulz_k(S_t)
--
-- Atlas momentum (Eq 33):
--   S_t = theta_t * S_{t-1} + grad(sum_{i=t-c+1}^{t} gamma_i ||M(phi(k_i)) - v_i||^2)
--
-- The momentum S accumulates Omega-rule gradients over a sliding window.
-- Newton-Schulz then orthogonalizes the accumulated momentum before
-- applying it to the memory update.
--
-- Key Atlas insight: S_t depends on S_{t-1} (linear) but NOT on M_{t-1}
-- (because the Omega rule gradient uses outer_loop_params, not M).
-- This enables EXACT parallel momentum computation (see 05_atlas_parallel.md).
--
-- Inner-loop (Atlas) vs outer-loop (AdaMuon):
--   Inner loop: Muon only (no second moment needed — inner loop is short)
--   Outer loop: AdaMuon (second moment stabilizes across many gradient steps)
--   The inner loop sees a handful of tokens per chunk.
--   The outer loop sees the full training stream.
--   Adam's variance estimate only helps with long gradient histories.
```

## Frequency-Gated AdaMuon

Like AdamW (08_adamw_outer.md), AdaMuon must be frequency-aware for NL:

<!-- HADES: Derived from hope_equations/eq-071-arch-variant2 (§6 Eq 71), frequency gating for AdaMuon -->
```text
-- Per-level AdaMuon state:
STRUCT LevelAdaMuonState {
  s: Tensor,             -- first moment [d_out, d_in] (matrix-valued)
  v: Vec<f32>,           -- second moment [d_out * d_in] (element-wise)
  level_step: u32,       -- level-local step count for bias correction
}

-- Frequency gating follows the same pattern as AdamW:
--   Active level: run adamuon_step with accumulated gradient
--   Frozen level: NO moment update, NO step increment
--   Level-local bias correction (same reasoning as 08_adamw_outer.md)

-- Cost amortization:
--   Level 0 (C=1): AdaMuon runs every step — expensive but most critical
--   Level 3 (C=512): AdaMuon runs every 512 steps — 5 NS iterations × 1/512 frequency
--     Amortized cost per step: O(d^2 * k_ns / 512) ≈ negligible
--   Slow levels are natural candidates for AdaMuon because the cost is amortized
--   over many frozen steps, while the directional accuracy matters more for
--   parameters that update infrequently (each update must count).
```

## Parameter Group Strategy

Not all parameters benefit from Newton-Schulz. Practical configuration:

```text
-- Matrix-valued parameters (W_K, W_V, W_Q, W_O, embed, unembed):
--   Shape: [d_out, d_in] — cross-dimension correlations matter
--   Use AdaMuon with k_ns = 5
--   The orthogonalization captures gradient structure across dimensions
--
-- Bias vectors (b_alpha, b_theta, b_eta):
--   Shape: [d] — no cross-dimension structure to orthogonalize
--   Use Adam (k_ns = 0, which disables NS)
--   Newton-Schulz on a vector is equivalent to normalization — wasteful
--
-- Gate projections (W_alpha, W_theta, W_eta):
--   Shape: [d, d] — matrix-valued but through sigmoid/softplus saturation
--   Use Adam or AdaMuon depending on dimension
--   For small d: Adam (NS overhead not worth it)
--   For large d: AdaMuon (directional accuracy helps)
--
-- Configuration pattern:
--   param_groups = [
--     {params: matrix_params, optimizer: "adamuon", k_ns: 5},
--     {params: bias_params, optimizer: "adam", k_ns: 0},
--   ]
--   Both groups share beta1, beta2, eps, weight_decay, lr schedule.
--   The only difference is whether NS runs on the first moment.
```

## Gradient Through Newton-Schulz (Outer-Loop Only)

Newton-Schulz is applied to the outer-loop momentum, not to inner-loop state.
The tape does NOT differentiate through NS — it is part of the optimizer, which
operates AFTER the tape backward completes:

```text
-- Execution order per step:
--   1. Forward pass (tape records operations)
--   2. Tape backward (computes dL/d(outer_loop_params))
--   3. AdaMuon step (uses computed gradients to update params)
--
-- Step 3 is NOT recorded on the tape. The optimizer is a POST-PROCESSING
-- step that modifies parameters using the gradients from Step 2.
-- No second-order gradients through the optimizer are needed.
--
-- This is the same for Adam, Muon, AdaMuon, or any outer-loop optimizer.
-- The tape computes first-order gradients. The optimizer uses them.
--
-- (Inner-loop Muon in Atlas IS on the tape — but that's the memory rule's
-- responsibility, not the optimizer's. See 05_atlas_parallel.md.)
```

## Hyperparameter Defaults

<!-- HADES: Derived from atlas_equations/eq-032-atlas-memory-muon (Atlas Eq 32), experimental defaults -->
```text
-- From Atlas experiments and Muon literature:
--   beta1 = 0.95      (first moment — slightly higher than Adam's 0.9)
--   beta2 = 0.999     (second moment — same as Adam)
--   eps = 1e-8        (same as Adam)
--   weight_decay = 0.1 (same as AdamW)
--   k_ns = 5          (Newton-Schulz iterations — standard from Jordan et al.)
--   lr = 2e-2 (Muon) or 4e-4 (Adam-scale, adjusted for NS normalization)
--
-- Learning rate note:
--   Muon's NS normalization changes the effective gradient scale.
--   Raw Muon uses lr ~ 2e-2 (10-50× larger than Adam).
--   AdaMuon with 1/sqrt(v) division may need intermediate lr.
--   The exact lr depends on the interaction between NS and v-scaling.
--   Tuning required — no universal default exists yet for AdaMuon in NL.
```

## Implementation Notes

1. **Not yet implemented**: Unlike AdamW (which has Python and GPU implementations),
   AdaMuon is a Stage 3 extension. The existing Newton-Schulz code in the momentum
   spec (01_momentum.md) provides the building block. Implementation requires adding
   the matrix-valued first moment to the optimizer state and the NS call after bias
   correction.

2. **Memory overhead**: The first moment S is matrix-valued [d_out, d_in], same size
   as the parameter itself. Adam's first moment m is also the same size (flattened).
   So AdaMuon's first-moment memory is the SAME as Adam's — just reshaped as a matrix
   for the NS call. No additional memory beyond Adam + the NS scratch space (one
   temporary matrix for the iteration).

3. **NS convergence**: Newton-Schulz converges in 5 iterations for well-conditioned
   inputs. If the momentum matrix is poorly conditioned (early training, after lr
   warmup), more iterations may be needed. Monitor `||X^T X - I||_F` to verify
   convergence. If it exceeds 0.1 after k_ns iterations, increase k_ns or improve
   the initial normalization.

4. **Mixed optimizer groups**: The parameter group strategy (AdaMuon for matrices,
   Adam for biases) requires the optimizer to dispatch per parameter shape. This
   is a configuration-time decision, not a per-step branch — the optimizer state
   layout is fixed at initialization.

5. **Interaction with M3**: M3 (02_m3.md) allows different optimizers per CMS level.
   The natural M3 configuration: AdamW for fast levels (cheap, frequent), AdaMuon
   for slow levels (expensive but infrequent, directional accuracy critical).

## Axiom Compliance

- **NL IS #6** (optimizers are associative memory): Muon's momentum S is a matrix-valued associative memory with Newton-Schulz as the output nonlinearity (HOPE §4.3). AdaMuon composes two associative memories: S (directional, matrix) and v (magnitude, element-wise).
- **MIRAS IS #1** (orthogonal design choices): AdaMuon is orthogonal to the 4-knob framework — it is an outer-loop optimizer choice independent of inner-loop memory structure, bias, retention, and algorithm.
- **CS-27/28** (frequency-aware optimizer): Same per-level state and frequency gating as AdamW. The NS cost is amortized by CMS frequency scheduling.
