# Spec 64: Pre-Reset M-Norm Diagnostics

```text
CONTRACT
Purpose:    Capture memory state norms BEFORE periodic reset zeros the buffers.
            Current diagnostics read post-reset zeros for all levels when
            reset_intervals=[1,1,1,1], making it impossible to verify that:
            (a) memory accumulates meaningful state during the forward pass,
            (b) backward gradients flow through memory correctly,
            (c) different head_dim configurations affect memory utilization.

Expects:    Spec 57 delivered: selective periodic reset via maybe_reset_levels().
            Spec 63 delivered: deferred backward gnorm readback (log-step gating).
            update_m_norm_tracking() runs in main loop AFTER run_step() returns,
            but run_step() calls maybe_reset_levels() as its last action.
            Result: all memory buffers are zero when norms are measured.

Guarantees: Pre-reset M norms captured on log steps with zero additional GPU syncs
            on non-log steps. Numerically identical training — no change to forward,
            backward, or optimizer paths. Existing post-reset norm tracking unchanged.
            CMS tape sidecar and metrics.jsonl emit both pre-reset and post-reset
            (zero) norms for each level.

Cost:       One memory_norms() call per log step (already existed, just moved earlier).
            No new GPU buffers. No new CUDA kernels. ~48 kernel launches + 1 D2H
            per log step (same as current cost, just at a different point in the step).

Trade-off:  Pre-reset norms are captured after optimizer update but before reset.
            This means they include the optimizer's effect on M-projections (W_K, W_V)
            but reflect the memory state that was actually used for prediction.
            This is the correct measurement point — it's what the model "knew" at
            end of sequence, before slate is wiped for the next sequence.

Position:   specs/infrastructure/64_pre_reset_m_norm_diagnostics.md
Extends:    specs/infrastructure/57_selective_periodic_reset.md
            specs/infrastructure/63_deferred_backward_gnorm.md

Source:     hd=32 vs hd=64 ablation (2026-03-30): all level_m_norms read 0.0 after
            reset_intervals corrected to [1,1,1,1]. Cannot diagnose whether memory
            system is contributing to learning or if model relies entirely on SWA.
```

## Problem

With `reset_intervals=[1,1,1,1]` (correct per gear curriculum design — all levels
reset at sequence boundary), the execution order in `run_step` is:

```
1. gpu_stacked_forward()     — memory accumulates across seq_len tokens
2. gpu_stacked_backward()    — gradients flow through memory trajectory
3. extract block_level_gnorms
4. gpu_stacked_adamw_update() — optimizer updates weights
5. gpu_stacked_sync_embed_weights()
6. maybe_reset_levels()      — zeros ALL memory buffers    ← inside run_step
```

Then in the main loop (log steps only):
```
7. update_m_norm_tracking()  — calls memory_norms()        ← reads zeros
8. collect_cms_diagnostics() — reports zeros to metrics
```

Every logged `level_m_norms` entry is `[0.0, 0.0, 0.0, 0.0]`. This blinds us to:
- Whether L0 memory is accumulating at hd=64 vs hd=32
- Whether higher levels (L1-L3) develop meaningful state within a sequence
- Whether the backward pass produces gradients that actually improve memory projections
- Dormancy detection (spec 28) — all levels appear dormant when they may be active

## Solution

### Move norm capture before reset, inside run_step

On log steps (`log_this=true`), capture pre-reset M norms inside `run_step`,
after the optimizer update but before `maybe_reset_levels`. Store them in
`GpuStackedContext` for the main loop to read.

#### Changes to `run_step` (cli/src/run.rs)

```rust
// After optimizer update, before reset:
if log_this {
    gpu_context.update_m_norm_tracking();  // captures pre-reset norms
}

// Selective periodic reset (spec 57)
maybe_reset_levels(pulse, reset_intervals, fire_counts, gpu_context);
```

#### Changes to main loop (cli/src/run.rs)

Remove the `update_m_norm_tracking()` call from the main loop logging block
(line 609). It's now called inside `run_step` at the correct point.

#### No changes to `update_m_norm_tracking` or `memory_norms`

The functions are correct — they compute Frobenius norms of `context.memory[level]`
buffers on GPU. The only issue was timing: called after reset instead of before.

## Files Modified

| File | Change |
|------|--------|
| `cli/src/run.rs` | Move `update_m_norm_tracking()` from main loop into `run_step`, gated by `log_this`, positioned after optimizer but before `maybe_reset_levels`. Remove duplicate call from main loop. |

## Diagnostic Value

After this fix, metrics.jsonl will show:

```json
{
  "level_m_norms": [0.14, 3.3e-05, 1.7e-09, 6.8e-18],
  "level_m_deltas": [0.002, 1.1e-06, 3.2e-11, 0.0]
}
```

This enables:
1. **Memory utilization comparison**: hd=32 vs hd=64 — does L0 accumulate differently?
2. **Backward effectiveness**: do m_deltas grow over training? (memory is learning)
3. **Level hierarchy validation**: L0 >> L1 >> L2 >> L3 norms (expected at seq_len=512)
4. **Dormancy detection**: non-zero deltas = active; zero deltas = dormant (spec 28)

## Acceptance Criteria

1. `level_m_norms` shows non-zero values for active levels on log steps
2. `level_m_deltas` reflects step-over-step changes in memory state
3. Zero additional GPU syncs on non-log steps (no perf regression)
4. Training is numerically identical — same loss curve with and without the fix
5. `cargo test --features cuda --lib` passes
6. Dormancy detection (spec 28) functional again

## Ontological Compliance

- **CS-18**: Diagnostic timing change in orchestration tier (cli/src/run.rs). No core changes.
- **CS-40**: No tape involvement — measurement path only.
- **CS-32**: Observe-then-advance preserved — norms captured after advance (optimizer),
  before next observe (reset prepares for next sequence).
