# Spec 54: Batched Gradient Norm Reduction

```text
CONTRACT
  Purpose:    Replace per-tensor sync-and-accumulate gradient norm with batched
              kernel launches and a single sync, eliminating ~708 cudaDeviceSynchronize
              calls per training step (94% of all syncs).
  Expects:    gpu_stacked_optimizer.rs has `gpu_stacked_grad_norm_ex` and
              `gpu_stacked_per_block_grad_norms` which each launch `grad_norm_sq_cuda`
              per tensor with a sync+D2H after each launch. The norm_scratch buffer
              is currently sized for the largest single tensor's partials.
  Guarantees: Gradient norm and per-block gradient norms are bit-identical.
              Total syncs per step drop from ~753 to ~49 (for k=4, 8 blocks).
              No new CUDA kernels — reuses existing `grad_norm_sq_cuda`.
  Cost:       ~3.5 MB additional GPU scratch memory (at d=1024, k=4, 8 blocks).
              ~10 MB at d=2048, 12 blocks. Negligible vs model VRAM.
  Trade-off:  Scratch buffer grows from ~16 KB to ~3.5 MB. The 708 eliminated
              pipeline drains vastly outweigh the memory cost. Also removes 4
              unnecessary safety fences (B4, O3, O4, O7) for 4 more eliminated syncs.
  Position:   specs/infrastructure/54_batched_grad_norm.md
  Source:     Profile data: GPU compute sawtooth (task_bdefc4). Self-documented
              bottleneck at gpu_stacked_optimizer.rs:172-175.
```

## Problem

### The sawtooth root cause

Every training step executes ~753 `cudaDeviceSynchronize()` calls on a k=4,
8-block model. Each sync drains the entire GPU kernel pipeline, forcing the GPU
to idle until the CPU processes the result and launches the next kernel. This
creates the sawtooth pattern visible in `nvidia-smi` GPU utilization.

### Sync budget breakdown (k=4, n_blocks=8)

| Source | Syncs/step | % | Purpose |
|--------|-----------|---|---------|
| `gpu_stacked_grad_norm_ex` | 356 | 47% | Per-tensor partial reduction for global gradient L2 norm |
| `gpu_stacked_per_block_grad_norms` | 352 | 47% | Per-tensor partial reduction for per-block gradient norms |
| Backward (gnorm + dots) | 41 | 5% | Per-level gnorm diagnostics + spec 53 dot products |
| Forward (loss D2H) | 1 | <1% | Necessary — loss must reach CPU for logging |
| Unnecessary fences | 4 | <1% | Safety fences between GPU-only operations |

**94% of syncs** come from the gradient norm computation's per-tensor
sync-accumulate pattern.

### Current implementation

`gpu_stacked_optimizer.rs:176-193`:
```rust
let mut accum = |g: &GpuBuf<f32>| {
    let n = g.len() as i32;
    let mut num_blocks: i32 = 0;
    let err = unsafe {
        crate::cuda_ffi::grad_norm_sq_cuda(
            g.as_ptr(), state.norm_scratch.ptr(),
            n, &mut num_blocks,
        )
    };
    crate::dispatch::cuda_sync();       // ← sync per tensor
    let nb = num_blocks as usize;
    state.norm_scratch.slice(0, nb).copy_to_host(&mut state.norm_host[..nb]);
    for i in 0..nb {
        total_sq += state.norm_host[i] as f64;
    }
};
```

This is called once per gradient tensor. For 8 blocks × (8 SWA params + 4 levels
× 9 params) + 4 shared params = **356 times** for the global norm, and 352 times
for per-block norms. Each call:
1. Launches `grad_norm_sq_cuda` (fast GPU kernel)
2. `cudaDeviceSynchronize()` (drains GPU pipeline — **the bottleneck**)
3. D2H copy of partial sums (~4-16 KB)
4. Host-side accumulation

The problem: **step 2 kills GPU pipelining.** The next kernel can't launch until
the CPU finishes step 4 for the current tensor.

### Why the scratch buffer is the constraint

All tensors write their partial sums to the **same** `norm_scratch` buffer.
Without offsetting, launching all kernels before syncing would overwrite earlier
results.

## Solution

### 2.1 Batched launch with offset scratch

Resize `norm_scratch` to hold ALL tensors' partial sums simultaneously. Each
kernel writes to its own offset within the scratch buffer. After all kernels
launch, do ONE sync and ONE D2H copy.

```rust
fn gpu_stacked_grad_norm_ex(grads: &GpuStackedGrads, state: &mut GpuStackedAdamWState, skip_embed: bool) -> f32 {
    // Phase 1: Launch all partial reduction kernels with offsets
    let mut offset = 0usize;
    let mut segments: Vec<(usize, usize)> = Vec::new(); // (offset, num_blocks)

    let mut launch = |g: &GpuBuf<f32>| {
        let n = g.len() as i32;
        if n == 0 { return; }
        let mut num_blocks: i32 = 0;
        let err = unsafe {
            crate::cuda_ffi::grad_norm_sq_cuda(
                g.as_ptr(),
                state.norm_scratch.ptr().add(offset),  // offset output
                n, &mut num_blocks,
            )
        };
        assert_eq!(err, 0);
        let nb = num_blocks as usize;
        segments.push((offset, nb));
        offset += nb;
    };

    // Launch all (same order as before)
    if !skip_embed {
        launch(&grads.d_w_embed);
        launch(&grads.d_w_unembed);
    }
    launch(&grads.d_ln_final_gamma);
    launch(&grads.d_ln_final_beta);
    for bg in &grads.blocks {
        launch(&bg.d_w_q);
        // ... all tensors ...
    }

    // Phase 2: Single sync + single D2H copy
    crate::dispatch::cuda_sync();
    state.norm_scratch.slice(0, offset).copy_to_host(&mut state.norm_host[..offset]);

    // Phase 3: Host-side accumulation (same as before, just over the full buffer)
    let mut total_sq = 0.0f64;
    for &(seg_off, nb) in &segments {
        for i in 0..nb {
            total_sq += state.norm_host[seg_off + i] as f64;
        }
    }

    // Add alpha_mem (host-side, unchanged)
    for bg in &grads.blocks {
        for &g in &bg.d_alpha_mem {
            total_sq += (g as f64) * (g as f64);
        }
    }

    total_sq.sqrt() as f32
}
```

### 2.2 Same pattern for per-block norms

Apply the identical batched-launch pattern to `gpu_stacked_per_block_grad_norms`.
Per-block accumulation requires tracking which segments belong to which block,
but the launch-sync-copy pattern is the same.

```rust
fn gpu_stacked_per_block_grad_norms(...) -> PerBlockGradNorms {
    let mut offset = 0usize;
    // Vec of (block_idx, offset, num_blocks, is_l0)
    let mut segments: Vec<(usize, usize, usize, bool)> = Vec::new();

    for (b, bg) in grads.blocks.iter().enumerate() {
        // Launch all tensors for this block (same kernel, different scratch offsets)
        let mut launch = |g: &GpuBuf<f32>, is_l0: bool| {
            // ... same launch pattern with offset ...
            segments.push((b, offset, nb, is_l0));
            offset += nb;
        };
        launch(&bg.d_w_q, false);
        // ...
        for (l, lg) in bg.levels.iter().enumerate() {
            let is_l0 = l == 0;
            launch(&lg.d_w_k_mem, is_l0);
            // ...
        }
    }

    // ONE sync, ONE D2H copy
    crate::dispatch::cuda_sync();
    state.norm_scratch.slice(0, offset).copy_to_host(&mut state.norm_host[..offset]);

    // Accumulate per-block
    // ...
}
```

### 2.3 Scratch buffer sizing

Resize `norm_scratch` and `norm_host` to hold the total partials across ALL
tensors:

```rust
// In GpuStackedAdamWState::new():
// Current: max_partials = max_single_tensor_len / 256 + 1
// New: total_partials = sum of ceil(tensor_len / 256) for ALL tensors

let mut total_partials = 0usize;
total_partials += (params.w_embed.len() + 255) / 256;
total_partials += (params.w_unembed.len() + 255) / 256;
// ... all shared + per-block + per-level tensors ...

norm_scratch: GpuBuf::zeros(total_partials),
norm_host: vec![0.0f32; total_partials],
```

Memory cost:

| Config | Total partials | Scratch size |
|--------|---------------|-------------|
| d=768, k=4, 2 blocks | ~120K | 0.5 MB |
| d=1024, k=4, 8 blocks | ~920K | 3.5 MB |
| d=2048, k=4, 12 blocks | ~2.8M | 10.7 MB |

### 2.4 Remove unnecessary fences

Additionally remove these 4 unnecessary `cuda_sync()` calls:

| ID | File | Line | Context |
|----|------|------|---------|
| B4 | gpu_stacked_backward.rs | 608 | End-of-backward fence — grads stay on GPU for optimizer |
| O3 | gpu_stacked_optimizer.rs | 374 | End of `scale_grads` — next op is AdamW GPU kernels |
| O4 | gpu_stacked_optimizer.rs | 512 | After `clamp_f32_cuda` on `b_theta` — error-check only |
| O7 | gpu_stacked_optimizer.rs | 536 | `sync_embed_weights` — transpose and gather on same stream |

### 2.5 Backward gnorm + dot product batching (Phase 2)

In the independent/FreqGated aggregation backward path, each block computes:
- 1 `compute_gnorm` call on `d_y_combined` (diagnostic per-level output gnorm)
- k `dot_product_partial_f32_cuda` calls for softmax Jacobian (spec 53)

All k+1 operations read from the same-sized buffers ([bs*s, d]) and write
partial sums. None are consumed until after all are computed. Batch them
with offset scratch and ONE sync per block.

```rust
// Independent path: batch gnorm + k dot products, 1 sync
let total_slots = 1 + cfg.k; // 1 gnorm + k dots
let gnorm_scratch = GpuBuf::zeros(max_norm_blocks * total_slots);

// Launch gnorm at offset 0
let mut offset = 0usize;
let mut gnorm_nb: i32 = 0;
grad_norm_sq_cuda(d_y_combined.as_ptr(), scratch.ptr(), bsd_i32, &mut gnorm_nb);
offset += gnorm_nb as usize;

// Launch k dot products at offset positions
let mut dot_nbs = vec![0i32; cfg.k];
for l in 0..cfg.k {
    dot_product_partial_f32_cuda(
        d_y_combined.as_ptr(), y_per_level[l].as_ptr(),
        scratch.ptr().add(offset), bsd_i32, &mut dot_nbs[l],
    );
    offset += dot_nbs[l] as usize;
}

// ONE sync + D2H
cuda_sync();
scratch.slice(0, offset).copy_to_host(&mut host[..offset]);

// Accumulate gnorm from first segment
// Accumulate dots from subsequent segments
```

Chain mode gnorms remain per-level (1 sync each) because `d_upstream` is
mutated between levels — the norm must capture the pre-mutation value, and
cloning the full buffer ([bs*s, d]) to defer would cost more than the sync.

**Scratch sizing**: `gnorm_scratch` grows from `max_norm_blocks` to
`max_norm_blocks * (1 + k)`. At d=2048, k=4: 40,960 → 204,800 floats
(800 KB). Negligible.

## Sync budget after fix (Phase 1 + Phase 2)

| Source | Original | Phase 1 | Phase 2 | Eliminated |
|--------|----------|---------|---------|-----------|
| `gpu_stacked_grad_norm_ex` | 356 | ~10 | ~10 | 346 |
| `gpu_stacked_per_block_grad_norms` | 352 | ~8 | ~8 | 344 |
| Unnecessary fences | 4 | 0 | 0 | 4 |
| Backward (independent: gnorm+dots) | 60 | 60 | 12 | 48 |
| Backward (chain: gnorms) | 48 | 48 | 48 | 0 |
| Forward (loss D2H) | 1 | 1 | 1 | 0 |
| **Total (independent)** | **773** | **79** | **31** | **742** |
| **Total (chain)** | **761** | **67** | **67** | **694** |

Note: independent vs chain depends on `hope_variant`. Current active models
use Chained, so Phase 2 primarily benefits future independent-mode runs.
Chain mode already got its main wins from Phase 1.

**96% reduction in pipeline drains (independent mode).**

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_optimizer.rs` | Batched launch pattern for both norm functions; resize scratch; remove O3, O4, O7 fences |
| `core/src/gpu_stacked_backward.rs` | Remove B4 fence (line 608) |

No new CUDA kernels. No FFI changes. No Python changes.

## Acceptance Criteria

1. `gpu_stacked_grad_norm_ex` uses ONE sync+D2H instead of per-tensor
2. `gpu_stacked_per_block_grad_norms` uses ONE sync+D2H instead of per-tensor
3. Gradient norm values match the old implementation (bit-exact)
4. Per-block gradient norms match (bit-exact)
5. 4 unnecessary fences removed (B4, O3, O4, O7)
6. norm_scratch sized for total partials across all tensors
7. Profile shows reduced sawtooth amplitude in GPU utilization
8. All existing tests pass

## Risk Assessment

**Zero correctness risk.** The partial reduction kernels are independent — they
read from different gradient buffers and write to non-overlapping scratch
regions. CUDA guarantees all kernels on the default stream execute in order,
so the last kernel completes before the sync. The host-side accumulation is
identical — same partials, same summation order, same f64 precision.

**Memory risk: low.** 3.5 MB at d=1024 is 0.01% of a 48 GB GPU's memory.
10.7 MB at d=2048 is 0.007% of a 141 GB H200.

## Ontological Compliance

- **CS-18**: Gradient norm is math in the Rust tier.
- **CS-42**: Scratch buffer reuse pattern (single allocation, partitioned).
- **CS-40**: No tape involvement — optimizer path.

## Equations Traced

No paper equations — this is a pure infrastructure optimization. The gradient
norm clipping algorithm is unchanged; only the synchronization pattern changes.
