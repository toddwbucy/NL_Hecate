# Spec 63: Deferred Per-Block Backward Gnorm Readback

```text
CONTRACT
Purpose:    Eliminate n_blocks per-step cuda_sync + D2H pipeline stalls in
            gpu_stacked_backward by deferring gnorm/dot-product readback.
            Chain mode: skip readback entirely on non-log steps (pure diagnostic).
            MAG mode: batch all blocks' readback into ONE sync after the block loop.

Expects:    Spec 62 delivered: GPU-side grad norm + clip in optimizer (10→2 syncs).
            Backward still has n_blocks syncs (one per block) for:
              Chain: block_level_gnorms (diagnostic only).
              MAG: block_level_gnorms (diagnostic) + dot products → d_alpha_mem (optimizer).
            Forward has n_blocks implicit syncs for alpha_mem D2H (MAG only).

Guarantees: Gradient computation is numerically identical — same kernels, same
            partials, same reductions. Only the host-side readback timing changes.
            Chain mode: zero backward syncs on non-log steps.
            MAG mode: 1 backward sync total (down from n_blocks).
            Forward alpha_mem: 0 syncs (GPU-side softmax + weighted sum).

Cost:       Scratch buffer grows by n_blocks factor (~4-25 MB depending on config).
            Two new CUDA kernels (~50 lines total). One new GPU buffer per block
            for cached alpha_weights (k floats each, negligible).

Trade-off:  block_level_gnorms zeroed on non-log steps in chain mode. Acceptable —
            they are only used for CMS diagnostic logging, never for control flow.
            MAG d_alpha_mem computation moves from per-block to post-loop batched.

Position:   specs/infrastructure/63_deferred_backward_gnorm.md
Extends:    specs/infrastructure/62_gpu_side_grad_norm_clip.md
            specs/infrastructure/54_batched_grad_norm.md

Source:     nvtop sawtooth VRAM pattern on A6000 during hd64 ablation runs.
            Profile: 6-8 cuda_sync per step from backward gnorm readback.
```

## Problem

Spec 62 reduced optimizer syncs from 9 to 1, but nvtop still shows sawtooth
VRAM utilization. The remaining pipeline stalls come from the backward pass:

### Chain mode (current ablation configs)

Each block in the backward loop (line 376 of `gpu_stacked_backward.rs`) does:
1. Launch `grad_norm_sq_cuda` per level into shared `gnorm_scratch`
2. `cuda_sync()` — **pipeline drain**
3. `copy_to_host()` — read partials
4. Sum partials → `block_level_gnorms[level]`

These gnorms are **purely diagnostic** — stored in `GpuStackedBlockGrads` and
used only for CMS tape logging on log steps. They never affect gradient flow.

### MAG/Independent mode (not current ablations, but common)

Each block (line 433) does:
1. Launch `grad_norm_sq_cuda` + k `dot_product_partial_f32_cuda` into scratch
2. `cuda_sync()` — **pipeline drain**
3. `copy_to_host()` — read partials
4. Reduce gnorm → `block_level_gnorms` (diagnostic)
5. Reduce dots → `d_alpha_mem` (needed for optimizer, but NOT for inter-block gradients)

### Forward alpha_mem (MAG only)

Each block (line 526 of `gpu_stacked_forward.rs`) does:
1. `alpha_mem.copy_to_host()` — **implicit sync** (cudaMemcpy D2H is synchronous)
2. Host softmax on k=4 values
3. k `saxpy_cuda` calls with host scalars

### Sync budget before this spec (post-spec 62)

| Source | Chain syncs | MAG syncs | Purpose |
|--------|:-----------:|:---------:|---------|
| Backward gnorm (per-block) | n_blocks | n_blocks | Diagnostic gnorms + d_alpha_mem |
| Forward alpha_mem D2H | 0 | n_blocks | Softmax weights for weighted sum |
| Forward loss D2H | 1 | 1 | NaN detection |
| Optimizer clip_scale D2H | 1 | 1 | alpha_mem host-side scaling |
| **Total (d768, 6 blocks)** | **8** | **14** | |
| **Total (d1024, 8 blocks)** | **10** | **18** | |

## Solution

### Phase 1: Deferred backward gnorm readback (chain + MAG)

**Key insight**: `block_level_gnorms` and `d_alpha_mem` are stored in
`GpuStackedBlockGrads` but never used for inter-block gradient flow. The gradient
to the next block flows through `d_residual_stream`, which is independent.

#### Chain mode: skip entirely on non-log steps

The gnorm values are purely diagnostic. On non-log steps:
- **Skip** `grad_norm_sq_cuda` kernel launches entirely (no point computing
  values that won't be read)
- **Skip** `cuda_sync()` + `copy_to_host()` entirely
- `block_level_gnorms` remains zeros (default initialization)

On log steps (`need_gnorms=true`): each block launches gnorm kernels at
block-specific offsets in the expanded scratch buffer (`b * per_block_slots`).
ONE sync + ONE D2H after the entire block loop reads all blocks' partials,
then host-side reduction fills `block_level_gnorms` for each block.

#### MAG mode: batch across blocks

d_alpha_mem is needed every step (optimizer input). But the sync can move:

1. **Expand scratch buffer**: `n_blocks * max_norm_blocks * (1 + k)` partials
2. **Per-block offset**: block `b` writes to `b * per_block_slots` in scratch
3. **Remove per-block sync**: kernels are ordered on the same CUDA stream
4. **ONE sync + ONE D2H** after the entire block loop
5. **Post-process**: reduce all blocks' partials on host in one pass

### Phase 2: GPU-side forward alpha_mem (MAG only, future)

Replace per-block `copy_to_host` + host softmax + k saxpy calls with:

1. **`softmax_weighted_sum_k4_cuda`** — Fused kernel: reads alpha_mem[k] on GPU,
   computes softmax in shared memory, weighted sum of y_levels in one pass.
2. Cache alpha_weights on GPU for backward use.
3. Backward reads weights from GPU cache instead of host Vec.

This phase is deferred — chain mode doesn't use alpha_mem, and the current
ablation runs are all chained.

## Files Modified

| File | Change |
|------|--------|
| `core/src/gpu_stacked_backward.rs` | Pass `log_this` flag. Chain: skip sync on non-log steps. MAG: per-block offsets in expanded scratch, ONE post-loop sync. |
| `core/src/gpu_stacked_forward.rs` | Pass `log_this` to backward call signature. |
| `cli/src/run.rs` | Pass `log_this` through `run_step` → backward. |

## Sync Budget After This Spec

### Chain mode (Phase 1)

| Source | Before | After (non-log) | After (log) |
|--------|:------:|:----------------:|:-----------:|
| Backward gnorm | n_blocks | **0** | **1** |
| Loss D2H | 1 | 1 | 1 |
| Optimizer clip_scale | 1 | 1 | 1 |
| **Total (d768)** | **8** | **2** | **3** |
| **Total (d1024)** | **10** | **2** | **3** |

### MAG mode (Phase 1)

| Source | Before | After |
|--------|:------:|:-----:|
| Backward gnorm+dots | n_blocks | **1** |
| Forward alpha_mem | n_blocks | n_blocks (Phase 2) |
| Loss D2H | 1 | 1 |
| Optimizer clip_scale | 1 | 1 |
| **Total (d768)** | **14** | **9** |

## Acceptance Criteria

1. Chain mode: zero backward syncs on non-log steps (measurable via nvtop)
2. MAG mode: 1 backward sync total (down from n_blocks)
3. Gradient computation numerically identical — loss curves match within f32 tolerance
4. `cargo test --features cuda --lib` passes
5. tok/s improvement measurable on A6000 at d=768 or d=1024 (chain mode ablations)

## Ontological Compliance

- **CS-18**: Infrastructure optimization in Rust/CUDA tier. No orchestration changes.
- **CS-42**: Scratch buffer expansion. Proportional to existing allocation.
- **CS-40**: No tape involvement — backward/optimizer path only.

## Equations Traced

No paper equations — pure infrastructure optimization. The backward gradient
computation is unchanged; only the host-side readback timing changes.
Extends specs 54 and 62.
