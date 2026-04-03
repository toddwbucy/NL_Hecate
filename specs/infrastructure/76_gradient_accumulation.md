# Spec 76: Gradient Accumulation + Batch Gradient Fix

## CONTRACT

| Field | Value |
|-------|-------|
| **Purpose** | Fix per-sample optimizer firing (bug) and add micro-step gradient accumulation to bridge effective batch size toward paper-scale (500K tokens/step) |
| **Expects** | Spec 71 full-sequence forward, spec 72 GPU-side cross-entropy, existing `GpuStackedGrads` structure, `gpu_stacked_scale_grads_ex` |
| **Guarantees** | One optimizer step per logical step (not per batch sample). `accum_steps` config field scales effective batch size without additional GPU memory. Backward compatible: `accum_steps=1` is bit-identical to corrected single-step behavior. |
| **Cost** | Gradient buffers persist across micro-steps (already allocated, no new memory). One additional `gpu_zero_grads` call per logical step. |
| **Trade-off** | Longer wall-clock per logical step (N micro-steps × batch_size forward/backward passes). Accepted — effective batch size is the bottleneck, not wall-clock per step. |
| **Position** | `cli/src/step.rs` (split into micro/update), `cli/src/feed.rs` (restructured loop), `core/src/gpu_stacked_backward.rs` (accumulate + zero), `cli/src/config.rs` (accum_steps field) |
| **Source** | TNT (2511.07343) batch=0.5M-1M tokens; Titans (2501.00663) ~500K tokens/step |

---

## Problem

### Bug: Per-Sample Optimizer Firing

The current build loop (feed.rs:598-628) calls `step()` once per batch sample:

```rust
for sample_idx in 0..batch_size {
    let result = step(..., &all_input[start..end], ...);
    // step() runs: forward → backward → OPTIMIZER → weight sync
}
```

`step()` includes the optimizer (gpu_stacked_adamw_update). With `batch_size=3`, the model receives **3 separate weight updates per iteration** — each on a single sample's gradients. This is not gradient averaging. It is 3 independent SGD steps with batch_size=1.

The comment in the code says "this matches the NL paradigm (forward IS optimization)" but that refers to the **inner loop** (memory M updates during forward). The **outer loop** optimizer (AdamW on W_K, W_V, W_Q, gates) should update once on averaged gradients, not once per sample.

### Gap: Effective Batch Size

Even after the fix, `batch_size=3` at `seq_len=4096` gives 12K tokens per optimizer step. Papers use 500K-1M. We need ~40x more tokens per update. Gradient accumulation closes this gap without requiring more GPU memory.

With `accum_steps=16, batch_size=3, seq_len=4096`:
- 16 × 3 × 4096 = **197K tokens per logical step**
- 4 A6000s with this config = ~800K tokens/step (paper scale)

---

## Design

### Split `step()` into Two Functions

```rust
/// Forward + backward only. Returns gradients. Does NOT run optimizer.
fn step_micro(
    gpu_params: &GpuStackedParams,   // immutable — weights don't change during accumulation
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    tokens: &[usize],
    targets: &[usize],
    conductor: &mut Conductor,
    profiler: &mut Option<GpuProfiler>,
    log_this: bool,
) -> MicroStepResult {
    let (last_logits, cache) = gpu_stacked_forward_sequence(...);
    let loss = gpu_cross_entropy_loss(...);
    let grads = gpu_stacked_backward(...);
    MicroStepResult { logits: last_logits, loss, grads, pulse: cache.pulse, block_level_gnorms }
}

/// Optimizer step on accumulated gradients. Mutates weights.
fn step_update(
    gpu_params: &mut GpuStackedParams,
    grads: &mut GpuStackedGrads,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    pulse: &Pulse,
    opt: &OptimizerConfig,
    lr: f32,
    max_grad_norm: f32,
    d: usize,
    v: usize,
    reset_intervals: &[usize],
    fire_counts: &mut [usize],
    gpu_context: &mut GpuStackedContext,
    profiler: &mut Option<GpuProfiler>,
    log_this: bool,
) -> f32 {  // returns grad_norm
    let gnorm = gpu_stacked_adamw_update(...);
    gpu_stacked_sync_embed_weights(gpu_params, d, v);
    if log_this { gpu_context.update_m_norm_tracking(); }
    maybe_reset_levels(pulse, reset_intervals, fire_counts, gpu_context);
    gnorm
}
```

### New Gradient Operations

```rust
/// Zero all gradient buffers (called once at start of each logical step).
pub fn gpu_zero_grads(grads: &mut GpuStackedGrads) {
    grads.d_w_embed.zero();
    grads.d_w_unembed.zero();
    grads.d_ln_final_gamma.zero();
    grads.d_ln_final_beta.zero();
    for block in &mut grads.blocks {
        // zero all per-block gradient buffers
    }
}

/// Accumulate: grads_accum += grads_micro (element-wise add on GPU).
pub fn gpu_accumulate_grads(accum: &mut GpuStackedGrads, micro: &GpuStackedGrads) {
    // gpu_add_inplace for each buffer pair
}
```

Alternatively, `gpu_stacked_backward` can be modified to **add into** existing gradient buffers rather than overwriting. This avoids a separate accumulate step — backward writes `d_w += new_grad` instead of `d_w = new_grad`. A `backward_accumulate: bool` parameter or a separate `gpu_stacked_backward_into()` function controls this.

The simpler approach: backward always overwrites (current behavior), and we add `gpu_accumulate_grads` to sum micro-step results into an accumulator. The accumulator is zeroed once per logical step. This is cleaner because it doesn't change the backward function's contract.

### Restructured Build Loop (feed.rs)

```rust
for phase_step in 0..total_phase_steps {
    let lr = cosine_lr(phase_step, warmup_steps, total_phase_steps, opt.lr());
    let log_this = log_every > 0 && phase_step % log_every == 0;

    // Persistent gradient accumulator (allocated once, zeroed each logical step)
    gpu_zero_grads(&mut grad_accum);
    let mut total_loss = 0.0f32;
    let mut total_micro_steps = 0usize;

    // Accumulation window: accum_steps micro-steps
    for micro in 0..accum_steps {
        // Assemble batch for this micro-step
        let (all_input, all_target) = assemble_batch(&mut loaders, batch_size, phase_seq_len);

        // Each batch sample: forward + backward (NO optimizer)
        for sample_idx in 0..batch_size {
            let start = sample_idx * phase_seq_len;
            let end = start + phase_seq_len;

            let micro_result = step_micro(
                &gpu_params, &mag_cfg, &mut gpu_context,
                &all_input[start..end], &all_target[start..end],
                &mut conductor,
                &mut profiler, log_this && micro == 0,
            );

            // Accumulate gradients
            gpu_accumulate_grads(&mut grad_accum, &micro_result.grads);
            total_loss += micro_result.loss;
            total_micro_steps += 1;

            // Track tokens
            tokens_this_step += phase_seq_len;
        }
    }

    // Scale accumulated gradients: average over all micro-steps × batch samples
    let scale = 1.0 / total_micro_steps as f32;
    gpu_stacked_scale_grads_ex(&mut grad_accum, scale, false);

    // ONE optimizer step on averaged gradients
    let gnorm = step_update(
        &mut gpu_params, &mut grad_accum, &mut adamw_state,
        &pulse, opt, lr, max_grad_norm, d, v,
        &reset_intervals, &mut fire_counts,
        &mut gpu_context, &mut profiler, log_this,
    );

    let avg_loss = total_loss / total_micro_steps as f32;

    // Token-count-based checkpointing (spec 02: state file lifecycle)
    total_tokens_seen += tokens_this_step;
    let do_checkpoint = should_checkpoint_by_tokens(total_tokens_seen, ...);
}
```

### Config

```json
{
  "build": {
    "accum_steps": 16,
    "batch_size": 3,
    "seq_len": 4096
  }
}
```

New field in `BuildConfig`:

```rust
#[serde(default = "default_accum_steps")]
pub accum_steps: usize,

fn default_accum_steps() -> usize { 1 }
```

With `accum_steps=1`, the loop runs one micro-step, which is exactly the corrected single-step behavior (all batch samples averaged, one optimizer step). No behavioral change for existing configs.

---

## Conductor Semantics

The conductor advances **once per logical step** (one optimizer update), not per micro-step or per batch sample.

All tokens within a logical step see the same `pulse.active_levels`. For k=1 this is a no-op. For k>1, this means one conductor tick per `accum_steps × batch_size × seq_len` tokens processed.

This is intentional — the conductor determines *which CMS levels are active*, and that decision should be stable across the full gradient accumulation window. The inner-loop memory M updates happen per-token inside `forward_sequence()`.

---

## Checkpoint Policy (OUT OF SCOPE)

Checkpoint cadence is **not** part of this spec. The current step-based `save_every` remains
as a build-time convenience. The real checkpoint policy — when to persist, what triggers it,
how the user configures it — belongs in the State File Lifecycle epic (spec 02). Checkpoint
triggers include model unload, user-initiated save, and config-driven policies composable
in the JSON state file. That design is orthogonal to gradient accumulation.

---

## Token Accounting

`total_tokens_seen` tracks the cumulative token count for this instance. It starts from `BuildResumeState.total_tokens_seen` on resume, and advances by `accum_steps × batch_size × seq_len` per logical step.

Metrics logging includes:
- `tokens_this_step`: total tokens in this logical step (`accum_steps × batch_size × seq_len`)
- `total_tokens_seen`: cumulative since model init
- `accum_steps`: for interpreting effective batch size

---

## Memory

Zero additional GPU memory. Gradient accumulation reuses the same `GpuStackedGrads` buffer. The accumulator is zeroed at the start of each logical step, accumulated across micro-steps, then consumed by the optimizer. At no point do two sets of gradients exist simultaneously.

---

## Acceptance Criteria

1. `accum_steps=1` with corrected batch loop is bit-identical to properly averaged single-step behavior (one optimizer step per iteration, gradients averaged across batch samples)
2. `accum_steps=N, batch_size=B` produces equivalent gradients to `accum_steps=1, batch_size=N*B` (within f32 tolerance)
3. Constant GPU memory regardless of `accum_steps`
4. No data replay — cursor advances each micro-step, each batch sample sees fresh data
5. Grad norm computed on fully accumulated and averaged gradients
6. LR schedule uses logical step count (not micro-step count)
7. `tokens_this_step`, `total_tokens_seen`, and `accum_steps` present in metrics JSONL

---

## Implementation Phases

### Phase A: Fix batch gradient bug ✅

Split `step()` into `step_micro()` + `step_update()`. Restructure feed.rs loop so all batch samples run forward+backward, gradients are averaged, then one optimizer step fires. `accum_steps` defaults to 1 (no accumulation yet, but the loop structure is correct).

This phase alone fixes the per-sample optimizer bug. Every existing config produces correct gradient-averaged behavior.

### Phase B: Gradient accumulation + metrics ✅

Add `gpu_zero_grads`, `gpu_accumulate_grads`. Add outer `accum_steps` loop in feed.rs. Scale gradients by `1/(accum_steps * batch_size)`. Add `accum_steps` to config. Add `tokens_this_step`, `total_tokens_seen`, `accum_steps` to metrics JSONL.

### Phase C: Validation

Run TitansMAG d=1024 with `accum_steps=1` (verify loss matches old config). Then `accum_steps=16` (verify throughput scaling and loss trajectory).

---

## Files to Modify

| File | Change |
|------|--------|
| `cli/src/step.rs` | Split `step()` → `step_micro()` + `step_update()`. Keep `step()` as convenience wrapper. |
| `cli/src/feed.rs` | Restructure build loop: zero grads → micro-steps × batch samples → scale → optimizer. Add `accum_steps` + token counts to metrics. |
| `core/src/gpu_stacked_backward.rs` | Add `gpu_zero_grads()`, `gpu_accumulate_grads()` |
| `core/src/gpu_stacked_optimizer.rs` | Make `gpu_stacked_scale_grads_ex` public (for gradient averaging after accumulation) |
| `cli/src/config.rs` | Add `accum_steps` to `BuildConfig` + `PhaseConfig` |

---

## Traced Equations

- TNT (2511.07343) §4: batch size 0.5M-1M tokens
- Titans (2501.00663) §5: ~500K tokens per optimizer step
- HOPE (2512.24695) §6: outer-loop gradient = average over inner-loop segment
