# Spec 62: GPU-Side Gradient Norm + Clip

```text
CONTRACT
Purpose:    Eliminate per-block cuda_sync + D2H pipeline stalls in gradient norm
            computation by performing the entire norm→clip→scale sequence on GPU.
            Extends spec 54 (batched grad norm) by removing the host-side reduction
            entirely — partial sums are reduced on-device, clip scale computed on GPU,
            and conditional scaling reads the scale from device memory.

Expects:    Spec 54 delivered: per-block batched launch into shared norm_scratch.
            gpu_stacked_grad_norm_ex does 9 cuda_sync + D2H per step (1 shared + 8 blocks).
            grad_norm_sq_cuda kernel produces partial squared sums.
            grad_scale_cuda scales grads with a host-provided scalar.

Guarantees: Gradient clipping is numerically equivalent (same partial sums, same sqrt,
            same scale factor). Per-step syncs reduced from 9 to 1 (alpha_mem host-side
            clip readback). Grad norm value available for deferred readback on log steps.

Cost:       2 extra GPU scalars (8 bytes). Two new kernels (~40 lines CUDA total).
            Negligible vs existing scratch buffer.

Trade-off:  Grad norm is no longer returned synchronously from the optimizer. Callers
            must call gpu_read_grad_norm() on log steps. This is intentional — the norm
            is only needed for display/logging, not for control flow.

Position:   specs/infrastructure/62_gpu_side_grad_norm_clip.md
Extends:    specs/infrastructure/54_batched_grad_norm.md

Source:     Profile data: 9 cuda_sync per step from grad norm (spec 54 reduced from 708
            but left per-block syncs). nvtop sawtooth pattern visible on A6000.
```

## Problem

Spec 54 reduced gradient norm syncs from 708/step to ~10/step by batching kernel
launches within each block. But each block still requires one `cuda_sync()` + D2H
to accumulate partials on the host. For an 8-block model this is 9 pipeline drains
per step (1 shared + 8 blocks). Combined with the loss readback sync, this accounts
for 10 pipeline stalls per training step.

### Sync budget before this spec (post-spec 54)

| Source | Syncs/step | Purpose |
|--------|-----------|---------|
| `gpu_stacked_grad_norm_ex` (shared) | 1 | Sum shared param partials on host |
| `gpu_stacked_grad_norm_ex` (per-block) | 8 | Sum per-block partials on host |
| Forward loss D2H | 1 | Loss scalar for NaN detection |
| M-norm tracking | 0 | Already gated to log_every |
| **Total** | **10** | |

## Solution

### Phase 1: GPU-Side Reduction + Conditional Scale (this spec)

Keep all existing `grad_norm_sq_cuda` launches into the shared `norm_scratch` buffer,
but launch ALL of them (shared + all blocks) without any intermediate sync. Then:

1. **`reduce_partials_clip_cuda`** — Single-block kernel (256 threads).
   Grid-stride accumulates all partials from `norm_scratch`, tree-reduces in shared
   memory, computes `norm = sqrt(sum)`, writes norm to `out_norm` and
   `scale = min(1.0, max_grad_norm / norm)` to `out_scale`.

2. **`grad_scale_conditional_cuda`** — Per-element kernel.
   Reads `*scale_ptr` from device memory. If `scale >= 1.0`, entire warp exits
   immediately (uniform branch, near-zero cost). Otherwise `g[i] *= scale`.
   Replaces `grad_scale_cuda` which required a host-side scale value.

3. **alpha_mem scaling** — The only remaining sync. `alpha_mem` gradients are
   host-side scalars (k=4 per block, ~32 values total). A single D2H copy of
   `clip_scale` (1 float) enables host-side scaling. This is 1 sync vs 9.

### Phase 2: Deferred Loss Readback (future spec)

Make loss buffer persistent, remove `cuda_sync()` after cross-entropy. Gate NaN
detection to configurable interval. Takes total per-step syncs from 2 to 1 or 0.

## New CUDA Kernels

### `reduce_partials_clip_cuda`

```c
// Grid: 1 block of 256 threads
// Input:  partial_sums[0..total_partials] — squared norm partials from grad_norm_sq_cuda
// Output: out_norm = sqrt(sum), out_scale = min(1.0, max_grad_norm / norm)
__global__ void reduce_partials_clip_kernel(
    const float* partial_sums, int total_partials, float max_grad_norm,
    float* out_norm, float* out_scale)
{
    extern __shared__ float smem[];
    // Grid-stride accumulation, tree reduction, sqrt, scale computation
}
```

### `grad_scale_conditional_cuda`

```c
// Grid: ceil(n/256) blocks of 256 threads
// Input:  g[n] — gradient buffer, scale_ptr — device pointer to clip scale
// If *scale_ptr >= 1.0: all threads exit (uniform warp branch)
// Otherwise: g[i] *= *scale_ptr
__global__ void grad_scale_conditional_kernel(
    float* g, const float* scale_ptr, int n)
{
    float scale = *scale_ptr;
    if (scale >= 1.0f) return;  // uniform branch: zero divergence
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] *= scale;
}
```

## Deferred Grad Norm Readback

The optimizer no longer returns `grad_norm` synchronously. Instead:

- `gpu_stacked_adamw_update()` returns 0.0 for grad_norm (placeholder).
- `gpu_read_grad_norm(state)` reads `state.norm_result` via D2H — call on log steps only.
- The norm value in `norm_result` is valid immediately after the fused clip call.

## Files Modified

| File | Change |
|------|--------|
| `core/kernels/adamw.cu` | Add `reduce_partials_clip_cuda` + `grad_scale_conditional_cuda` |
| `core/src/cuda_ffi.rs` | FFI declarations for both new kernels |
| `core/src/gpu_stacked_optimizer.rs` | New `gpu_stacked_grad_norm_clip_fused()` replaces `gpu_stacked_grad_norm_ex`. Add `norm_result`, `clip_scale` to `GpuStackedAdamWState`. Add `gpu_read_grad_norm()`. |
| `cli/src/run.rs` | Import `gpu_read_grad_norm`, read on log steps only |

## Sync Budget After This Spec

| Source | Before | After | Eliminated |
|--------|--------|-------|-----------|
| Grad norm (shared + blocks) | 9 | 0 | 9 |
| alpha_mem clip scale D2H | 0 | 1 | -1 (new) |
| Forward loss D2H | 1 | 1 | 0 |
| **Total** | **10** | **2** | **8** |

## Acceptance Criteria

1. Gradient clipping is numerically equivalent — loss curves match pre/post within f32 tolerance
2. `reduce_partials_clip_cuda` output matches host-side reference (existing test suite)
3. Per-step syncs reduced from 10 to 2 (measurable via nvtop)
4. `cargo test --features cuda --lib` passes (all existing tests)
5. tok/s improvement measurable on A6000 at d=768 or d=1024

## Ontological Compliance

- **CS-18**: Gradient math in Rust/CUDA tier. No orchestration changes.
- **CS-42**: Scratch buffer reuse. Two new persistent GPU scalars (8 bytes total).
- **CS-40**: No tape involvement — optimizer path only.

## Equations Traced

No paper equations — pure infrastructure optimization. The gradient norm clipping
algorithm is unchanged; only the synchronization pattern changes. Extends spec 54.
