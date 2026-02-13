# Frequency Scheduler

```
CONTRACT
  Purpose:    Controls WHEN parameters at each CMS frequency level update.
              The gating mechanism that creates multi-scale parallelism.
              Off-schedule levels require ZERO computation — frozen completely.
  Expects:    Global step counter. Chunk sizes per level. Number of levels k.
  Guarantees: Boolean mask indicating which levels are active at each step.
              Level 0 (fastest, C=1) is ALWAYS active.
              Higher levels fire at geometrically decreasing rates.
              Error accumulation: frozen levels' gradients are stored, not lost.
  Cost:       O(k) per step for modular arithmetic check.
  Trade-off:  More levels = finer temporal resolution = more parameter groups.
              Geometric spacing [1,8,64,512] ensures distinct timescales.
              The conventional Transformer is the k=1 special case.
  Position:   specs/algorithms/frequency_scheduling/01_frequency_scheduler.md
  Source:     HOPE (2512.24695) Eq 71, Section 7.1
```

## The Frequency Gate (Eq 71)

```
FUNCTION: frequency_gate(step: u64, level: usize, chunk_sizes: &[u64]) -> bool
  C = chunk_sizes[level]
  return step % C == 0

  -- Level 0 (C=1): active EVERY step
  -- Level 1 (C=8): active every 8th step
  -- Level 2 (C=64): active every 64th step
  -- Level 3 (C=512): active every 512th step
```

## Active Levels at Step t

```
FUNCTION: active_levels(step: u64, chunk_sizes: &[u64]) -> Vec<bool>
  return chunk_sizes.iter().map(|C| step % C == 0).collect()

  -- Example with [1, 8, 64, 512]:
  -- Step 1:   [true, false, false, false]   (1 level active)
  -- Step 8:   [true, true,  false, false]   (2 levels active)
  -- Step 64:  [true, true,  true,  false]   (3 levels active)
  -- Step 512: [true, true,  true,  true]    (all 4 levels — rarest)
```

## CMS Parameter Update (Eq 71)

```
ALGORITHM: cms_parameter_update(params: &mut [Tensor], error_buffers: &mut [Tensor],
                                 grad_fn: Fn(level) -> Tensor,
                                 pulse: &Pulse)
  FOR level = 0 to k-1:
    IF pulse.is_active(level):
      -- Active: apply accumulated error + current gradient
      update = error_buffers[level] + grad_fn(level)
      params[level] = params[level] + update
      error_buffers[level] = zeros_like(params[level])  -- reset
    ELSE:
      -- Frozen: accumulate error
      error_buffers[level] = error_buffers[level] + grad_fn(level)
```

## Error Accumulation

When a level is frozen, its gradient signal is NOT lost. It accumulates in an
error buffer. When the level next fires, the accumulated error is applied all at once.

```
-- This means: NO INFORMATION IS WASTED
-- The frozen level contributes the SAME total gradient as if it had run every step
-- The only difference: it's applied in a batch instead of token-by-token

-- Eq 71 says f(.) is arbitrary — ANY optimizer can be used per level
-- The frequency scheduler makes ANY optimizer frequency-aware
-- (CS-28: optimizer must be frequency-aware)
```

## Chunk Size Computation

```
FUNCTION: compute_chunk_sizes(base: u64, frequencies: &[u64], T: u64) -> Vec<u64>
  -- Each level's chunk size = base * frequency multiplier
  chunk_sizes = frequencies.iter().map(|f| min(base * f, T)).collect()
  return chunk_sizes

  -- Typical: base=1, frequencies=[1, 8, 64, 512]
  -- Gives: chunk_sizes=[1, 8, 64, 512]
```

## Parallelism Implications

```
-- At most steps, only level 0 is active → 1 block computes
-- The other k-1 blocks are completely skipped (zero compute)
-- When multiple levels activate (rare), they're INDEPENDENT

-- Average computation per step:
-- level 0: every step (always)
-- level 1: 1/8 of steps
-- level 2: 1/64 of steps
-- level 3: 1/512 of steps
-- Average active levels: 1 + 1/8 + 1/64 + 1/512 ≈ 1.14

-- A 4-level CMS model does ~1.14x the work of a single-level model
-- but captures 4 distinct temporal scales
```

## Integration with Pulse

The frequency scheduler CREATES the Pulse that flows through the entire system:

```
FUNCTION: advance_pulse(pulse: &mut Pulse, chunk_sizes: &[u64])
  pulse.global_step += 1
  pulse.active_levels = active_levels(pulse.global_step, chunk_sizes)
  -- Every component reads pulse.active_levels to decide whether to update
```

## Error Buffer Health Invariant

```
-- When a level is frozen for C steps, its error buffer accumulates C gradients.
-- If these gradients are correlated (same sign), the accumulated norm grows linearly.
-- Applied all at once, this creates a "bomb" — one giant update at sync time.

INVARIANT: Error Buffer Norm Ratio
  At every sync point (when level i fires), monitor:

  norm_ratio = ||error_buffers[level]|| / ||grad_fn(level)||
    where the denominator is the CURRENT single-step gradient magnitude

  IF norm_ratio > threshold (configurable, default 10.0):
    LOG WARNING: "Error buffer for level {level} at {norm_ratio}x single-step norm"
    OPTIONALLY: clip error_buffers[level] to threshold * ||grad_fn(level)||

  WHY 10.0: For level 3 (C=512), accumulating 512 random gradients gives
  expected norm_ratio ≈ sqrt(512) ≈ 22.6 (random walk).
  If norm_ratio >> sqrt(C), gradients are correlated (systematic signal).
  If norm_ratio < sqrt(C), some cancellation occurred (expected).
  10.0 as default catches pathological accumulation while allowing normal behavior.

  THIS IS A HEALTH CHECK, NOT A TRAINING SIGNAL.
  It monitors for pathology — it does not steer the optimizer.
```

## Axiom Compliance

- **NL IS #2** (nested, multi-level, parallel): Frequency scheduler CREATES the multi-level structure
- **NL IS #8** (continuum memory): Discrete approximation to continuum of timescales
- **CS-28** (frequency-aware optimizer): Scheduler makes ANY optimizer frequency-aware
