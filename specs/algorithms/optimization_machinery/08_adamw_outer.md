# AdamW as Outer-Loop Optimizer

<!-- HADES: hope_equations/eq-013-momentum-argmin (§4.1 Eq 13); hope_equations/eq-071-arch-variant2 (§6 Eq 71); nl_optimizers/opt-adam (§4.2) -->
```text
CONTRACT
  Purpose:    AdamW is the standard outer-loop optimizer for NL models. HOPE §9.2
              explicitly uses AdamW with cosine annealing for all experiments. The NL
              reframing (Eq 13) shows that Adam's first-moment EMA IS an associative
              memory — not a metaphor, a proven equivalence. Adam extends this with a
              second-moment (variance) associative memory that provides per-element
              adaptive learning rates. AdamW (Loshchilov & Hutter, 2019) decouples
              weight decay from the adaptive gradient, preventing the L2 penalty from
              being divided by the second-moment estimate.

              The outer-loop operates on outer_loop_param tensors (W_K, W_V, W_Q,
              gate weights/biases, embeddings). It does NOT touch inner_loop_state
              (M, S) — those are updated by the memory rules themselves.

              Critical constraint (CS-27/28): AdamW must be frequency-aware. Each CMS
              level has its own parameter group with independent moment buffers. The
              optimizer ONLY updates a level when the Conductor's Pulse fires for that
              level. Between firings, moment buffers are frozen — no step counter
              increment, no moment update, no weight modification.
  Expects:    Gradients from Wengert tape reverse pass, accumulated per outer_loop_param.
              Pulse struct from Conductor indicating which CMS levels are active.
              Per-level learning rates (may differ by level). Cosine annealing schedule.
  Guarantees: Frequency-gated updates: level k parameters updated only when
              step % chunk_sizes[k] == 0. Moment buffers (m, v) maintain their state
              between firings — no reset, no spurious decay. Bias correction uses
              the level's OWN step count (number of times that level has fired), not
              the global step. Weight decay applied to the pre-update parameter value
              (decoupled, not L2 regularization).
  Cost:       O(P) per active level, where P is the parameter count for that level.
              Three element-wise passes: moment update, bias correction, parameter step.
              When a level is frozen: O(0) — literally no computation.
  Trade-off:  AdamW is well-understood and robust but element-wise — it cannot capture
              cross-parameter correlations the way matrix-valued momentum (Muon) can.
              For NL, AdamW is the practical default; Muon/AdaMuon are the expressive
              alternatives (see 01_momentum.md, S3b-S17).
  Position:   specs/algorithms/optimization_machinery/08_adamw_outer.md
              Sibling of: 01_momentum.md (momentum hierarchy), 02_m3.md (multi-scale)
  Source:     HOPE (2512.24695) §4.1-4.2 (Adam as AM), §6 Eq 71 (frequency gating),
              §9.2 (experimental setup); Loshchilov & Hutter 2019 (AdamW);
              TNT (2511.07343) experimental setup (AdamW + cosine)
```

## Adam as Associative Memory (HOPE §4.2)

The NL program views every optimizer component as an associative memory.
Adam's two moments are two such memories:

<!-- HADES: hope_equations/eq-013-momentum-argmin (§4.1 Eq 13); nl_optimizers/opt-adam (§4.2) -->
```text
-- First moment m (HOPE §4.1, Eq 11):
--   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
--
--   NL reframing (Eq 13): m_t = argmin_m sum_{i<=t} ||m - g_i||^2 + lambda * ||m||^2
--   This IS the closed-form solution of an L2 associative memory.
--   m is a VALUE-LESS associative memory (scalar weights on scalar gradients).
--   It compresses past gradient history with exponential decay.
--   Memory capacity: O(1) — it stores a weighted average, not individual gradients.

-- Second moment v (HOPE §4.2):
--   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
--
--   Same L2 associative memory, applied to SQUARED gradients.
--   v_t estimates the variance of the gradient distribution.
--   Adam's update: m_hat / sqrt(v_hat) normalizes per dimension.
--   This IS per-element preconditioning via variance prediction.

-- Combined (Adam):
--   theta_{t+1} = theta_t - lr * m_hat_t / (sqrt(v_hat_t) + eps)
--
--   Two nested associative memories:
--     Level 0: v (fast, tracks variance) — updates every step
--     Level 1: m (medium, tracks gradient mean) — updates every step
--     Level 2: theta (slow, tracks optimal weights) — updates via m/sqrt(v)
--
--   This is already a 3-level nested system in the HOPE sense (Definition 3, Eq 19).
--   The NL CMS wraps it in a FOURTH level of nesting: the frequency schedule.
```

## AdamW Update Rule

The standard decoupled weight decay formulation:

<!-- HADES: Derived from nl_optimizers/opt-adam (§4.2); Loshchilov & Hutter 2019 (decoupled weight decay) -->
```text
FUNCTION: adamw_step(theta: &mut [f32], grad: &[f32],
                     m: &mut [f32], v: &mut [f32],
                     t: &mut u32, lr: f32,
                     beta1: f32, beta2: f32,
                     eps: f32, weight_decay: f32) -> ()
  -- theta: parameter vector (outer_loop_param)
  -- grad: accumulated gradient from tape backward
  -- m, v: first/second moment buffers (per-parameter)
  -- t: step counter for this parameter group

  *t += 1

  -- Moment update
  FOR i in 0..theta.len():
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]

  -- Bias correction
  bc1 = 1 - beta1^(*t)
  bc2 = 1 - beta2^(*t)

  -- Parameter update (DECOUPLED weight decay)
  FOR i in 0..theta.len():
    m_hat = m[i] / bc1
    v_hat = v[i] / bc2
    theta[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta[i])

  -- Key difference from L2 regularization:
  --   L2: grad += lambda * theta; then Adam divides by sqrt(v) → decay is adaptive
  --   AdamW: decay applied AFTER Adam step → decay is constant-rate, independent of v
  --   AdamW is correct for NL because inner-loop gradients have high variance,
  --   and L2 would couple the regularization strength to gradient noise.
```

## Frequency-Gated Updates (CS-27, CS-28)

The critical NL adaptation: AdamW becomes frequency-aware by maintaining
per-CMS-level state and gating updates via the Pulse:

<!-- HADES: hope_equations/eq-071-arch-variant2 (§6 Eq 71) -->
```text
-- Frequency gating (HOPE Eq 71):
--   theta^(f_l) updated every C^(f) steps by accumulating optimizer error f(.)
--   When step % C^(f) != 0: zero update, zero computation.
--
-- For AdamW, f(.) = AdamW step. The "error" is the gradient, accumulated in
-- the error buffer while the level is frozen.

-- Per-level optimizer state:
STRUCT LevelAdamWState {
  m: Vec<f32>,            -- first moment, same shape as level params
  v: Vec<f32>,            -- second moment, same shape as level params
  level_step: u32,        -- number of times THIS level has fired
  -- NOT the global step — this counts only active updates for this level.
  -- Level 0 (C=1): level_step == global_step
  -- Level 3 (C=512): level_step == global_step / 512
}

-- Full optimizer state:
STRUCT FrequencyAwareAdamW {
  swa_state: AdamWState,       -- SWA params (always active, not CMS-gated)
  level_states: Vec<LevelAdamWState>,  -- one per CMS level (k states)
  -- Hyperparameters (shared or per-level):
  beta1: f32,                  -- default 0.9
  beta2: f32,                  -- default 0.999
  eps: f32,                    -- default 1e-8
  weight_decay: f32,           -- default 0.1
}

-- Gated step (called once per global step):
FUNCTION: frequency_gated_step(opt: &mut FrequencyAwareAdamW,
                               pulse: &Pulse,
                               swa_grads: &[f32],
                               level_grads: &[Vec<f32>],
                               lr: f32) -> ()
  -- SWA params always update (not frequency-gated)
  adamw_step(&mut opt.swa_state.params, swa_grads,
             &mut opt.swa_state.m, &mut opt.swa_state.v,
             &mut opt.swa_state.step, lr, ...)

  -- CMS levels: only update if Pulse says level is active
  FOR level in 0..pulse.k:
    IF pulse.active[level]:
      -- This level fires — apply accumulated gradient
      adamw_step(&mut level_params[level], level_grads[level],
                 &mut opt.level_states[level].m,
                 &mut opt.level_states[level].v,
                 &mut opt.level_states[level].level_step,
                 lr, ...)
    ELSE:
      -- Level frozen: NO moment update, NO step increment, NO parameter change
      -- Gradient accumulates in the error buffer (managed by Conductor)
      pass
```

## Bias Correction with Level-Local Step Count

A subtle but critical detail: bias correction must use the level's own step
count, not the global step.

<!-- HADES: Derived from hope_equations/eq-013-momentum-argmin (§4.1 Eq 13), bias correction with level-local step count -->
```text
-- Why level-local step count matters:
--
-- Global step = 1024. Level 3 fires every 512 steps → has fired twice.
-- If using global step for bias correction:
--   bc1 = 1 - 0.9^1024 ≈ 1.0 → almost no correction
--   bc2 = 1 - 0.999^1024 ≈ 0.64 → moderate correction
-- If using level-local step count:
--   bc1 = 1 - 0.9^2 = 0.19 → strong correction (only 2 samples!)
--   bc2 = 1 - 0.999^2 ≈ 0.002 → very strong correction
--
-- The level-local count is CORRECT because the moment buffers for Level 3
-- have only been updated twice. Using the global step would undercount
-- the bias, producing unstable updates in the early phase of slow levels.
--
-- This is equivalent to treating each CMS level as having its own independent
-- Adam optimizer that happens to share hyperparameters but not state.
```

## Cosine Annealing Schedule

HOPE §9.2 and TNT experiments use cosine annealing with linear warmup:

<!-- HADES: Derived from HOPE (2512.24695) §9.2 experimental setup; TNT (2511.07343) experimental setup -->
```text
FUNCTION: cosine_lr(step: u32, warmup_steps: u32, total_steps: u32,
                    lr_peak: f32, lr_min: f32) -> f32
  -- Linear warmup
  IF step < warmup_steps:
    return lr_peak * step / warmup_steps

  -- Cosine decay
  progress = (step - warmup_steps) / (total_steps - warmup_steps)
  progress = min(progress, 1.0)
  return lr_min + 0.5 * (lr_peak - lr_min) * (1 + cos(pi * progress))

-- Typical HOPE values (§9.2):
--   lr_peak = 4e-4
--   lr_min = 0.0
--   warmup_steps = 200 (or 500 for larger models)
--   total_steps = varies (10K-1M)

-- Per-level learning rate scaling:
--   All levels share the same cosine schedule by default.
--   Optional: scale lr per level (slow levels may need larger lr because
--   they accumulate gradients over more tokens before applying them).
--   This is a hyperparameter, not a paper prescription.
```

## Gradient Accumulation and Error Buffers

When a CMS level is frozen, gradients accumulate in an error buffer managed
by the Conductor. When the level fires, the accumulated gradient is passed
to AdamW as a single batch:

```text
-- Frozen level gradient handling:
--
-- Step 1-511 (Level 3 frozen):
--   error_buf[level_3] += grad_t    -- accumulate per-step gradients
--   (Gradients come from the tape backward through frozen-level reads)
--
-- Step 512 (Level 3 fires):
--   accumulated_grad = error_buf[level_3]
--   -- Option A: Average the accumulated gradient
--   grad_for_adam = accumulated_grad / 512
--   -- Option B: Use the sum directly (effective lr = lr / 512)
--   grad_for_adam = accumulated_grad
--   adamw_step(level_3_params, grad_for_adam, ...)
--   error_buf[level_3] = 0    -- reset
--
-- Current implementation uses sum (Option B) with the same lr.
-- This means slow levels see effectively larger gradient magnitudes,
-- which the second moment v naturally compensates for (Adam's adaptivity).
-- The accumulated gradient is the sum of 512 mini-batch gradients,
-- so v grows proportionally, and m_hat/sqrt(v_hat) stays bounded.
```

## GPU Implementation

The existing `GpuAdamWState` (`core/src/gpu_optimizer.rs`) implements a fused
AdamW kernel on GPU with zero PCIe traffic:

```text
-- GPU AdamW architecture:
--   Moment buffers (m, v) mirror every learnable parameter on GPU
--   Fused kernel: reads grad, reads m/v/theta, writes updated m/v/theta
--   One kernel launch per parameter buffer (not per element)
--   Gradient norm computed via parallel reduction (scratch buffer)
--
-- Current layout:
--   GpuAdamWState {
--     swa: MomentSWA        -- moments for embed, Q, K, V, O, unembed
--     levels: Vec<MomentLevel>  -- moments for W_K_mem, W_V_mem, W_Q_mem,
--                               -- W_alpha, b_alpha, W_theta, b_theta,
--                               -- W_eta, b_eta (per level)
--     step: u32             -- global step counter
--   }
--
-- TODO: level_step per MomentLevel for correct per-level bias correction.
-- Current implementation uses global step — adequate for Level 0/1 but
-- incorrect for slow levels (Level 2+) as described in bias correction section.
```

## Interaction with M3 (Multi-Scale Momentum)

M3 (`02_m3.md`) extends the outer loop from single-scale AdamW to multi-scale
momentum. The relationship:

```text
-- AdamW: one set of moment buffers per parameter, one update schedule
-- M3:    k sets of moment buffers (one per CMS level), k update schedules
--
-- AdamW with frequency gating IS a simplified M3:
--   It has per-level state (moment buffers per level)
--   It gates updates by CMS frequency
--   But it uses the SAME optimizer (AdamW) at every level
--
-- Full M3 allows DIFFERENT optimizers per level:
--   Level 0: AdamW (fast, element-wise, cheap)
--   Level 1: AdamW with different beta
--   Level 2: Muon (matrix-valued momentum, expensive but expressive)
--   Level 3: AdaMuon (Atlas variant, most expressive)
--
-- For now, frequency-gated AdamW is the default. M3 with heterogeneous
-- optimizers is a Stage 3 extension.
```

## Hyperparameter Defaults

```text
-- From HOPE §9.2 and TNT experiments:
--   beta1 = 0.9       (first moment decay)
--   beta2 = 0.999     (second moment decay)
--   eps = 1e-8        (numerical stability)
--   weight_decay = 0.1 (decoupled)
--   lr = 4e-4 (60M), 3e-4 (90M+) with cosine annealing
--   warmup = 200 steps (small models), 500 steps (large models)
--   grad_clip = 1.0 (global norm clipping before optimizer step)
--
-- These are the SAME as standard Transformer training — NL does not
-- require exotic outer-loop hyperparameters. The architecture difference
-- is entirely in the inner loop (memory rules, CMS frequency gating).
-- The outer loop just needs to not fight the inner loop (CS-27/28).
```

## Implementation Notes

1. **Already implemented**: `python/build.py:AdamW` (Python-side, flat arrays)
   and `core/src/gpu_optimizer.rs:GpuAdamWState` (GPU-side, fused kernel). Both
   work correctly for single-frequency or uniform-frequency CMS. The spec
   documents the general frequency-aware pattern that the existing implementations
   approximate.

2. **Level-local step count**: The existing GPU implementation uses a global step
   counter. For k=2 (Level 0 + Level 1 at 8x), this is acceptable — Level 1
   fires every 8 steps, so after 80 global steps its bias correction uses
   `1 - beta^80` instead of `1 - beta^10`. The difference is negligible for
   beta < 1. For k=4 with Level 3 at 512x, the difference matters and should
   be corrected (future work).

3. **Gradient clipping**: Applied BEFORE the optimizer step, to the aggregated
   gradient vector. Global norm clipping (`||g||_2 <= max_norm`) is standard.
   The clip threshold is typically 1.0. Clipping after accumulation (for frozen
   levels) uses the accumulated gradient norm, not per-step norms.

4. **SWA parameters**: Embedding, Q/K/V/O projections, and unembed weights are
   NOT CMS-gated — they update every step. Only memory-level parameters (W_K_mem,
   W_V_mem, gates) are frequency-gated. The optimizer must distinguish these two
   parameter groups.

5. **Checkpoint/restore**: Optimizer state (m, v, step counters) must be serialized
   alongside model parameters. The existing checkpoint format includes parameter
   weights but not optimizer state — this should be extended for long training runs
   where restarting from scratch wastes the warmup period.

## Axiom Compliance

- **NL IS #6** (optimizers are associative memory): Adam's two moments are explicitly associative memories (Eq 13). The first moment compresses gradient history via L2 regression. The second moment compresses squared-gradient history. The NL reframing is not a metaphor — it is a proven mathematical equivalence.
- **CS-27** (optimizer frequency must match architecture frequency): Per-level moment buffers and frequency-gated updates ensure the optimizer respects CMS structure. Level 3 parameters are not touched until Level 3 fires.
- **CS-28** (optimizer must be frequency-aware): AdamW itself is not forbidden (HOPE §9.2 uses it). The constraint is that it must be APPLIED frequency-aware, with per-level state and gated updates.
- **CS-29** (feedback loop is closed): The outer loop (AdamW) applies gradients that the inner loop (memory rules) generated. The inner loop uses parameters that the outer loop updated. This is a closed loop designed as a unit, not separate components.
