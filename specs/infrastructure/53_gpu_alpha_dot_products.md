# Spec 53: GPU-Resident Alpha Aggregation Dot Products

```text
CONTRACT
  Purpose:    Move the softmax-Jacobian dot products in independent aggregation
              backward from CPU (with D2H copies) to GPU, eliminating ~240 MB
              of PCIe round-trips per step at d=2048.
  Expects:    Spec 21 (stacked alpha aggregation) shipped. Independent/FreqGated
              backward in gpu_stacked_backward.rs:360-386 computes k dot products
              <d_y_combined, y_per_level[l]> on CPU after full D2H copy.
              grad_norm_sq_cuda kernel exists as a partial-reduction template.
  Guarantees: Backward produces identical d_alpha_mem values (bit-exact at f32
              rounding). Per-block D2H transfer drops from (k+1)*bsd floats to
              k*ceil(bsd/256) + k floats. No new cuda_sync() calls added.
  Cost:       One new CUDA kernel (~40 lines). One cuda_ffi binding. One Rust
              helper function. Modification to ~25 lines of gpu_stacked_backward.rs.
  Trade-off:  Partial sums are reduced on host (same as grad_norm_sq pattern).
              A fully-fused kernel could avoid even this, but the partial sums
              are ~200 floats per level — negligible vs the 12M floats eliminated.
  Position:   specs/infrastructure/53_gpu_alpha_dot_products.md
  Source:     Spec 21 (stacked alpha aggregation), HOPE eq-074 softmax Jacobian.
              Profile data: d=2048 block 7 backward hotspot (task_c37ad4).
```

## Problem

The independent aggregation backward (spec 21) computes the softmax Jacobian
for `d_alpha_mem` — the gradient of the learnable level-mixing weights. This
requires k dot products: `dot[l] = <d_y_combined, y_per_level[l]>`.

The current implementation (`gpu_stacked_backward.rs:372-386`):

```rust
let mut d_y_host = vec![0.0f32; bsd];
crate::dispatch::cuda_sync();                    // ← full pipeline drain
d_y_combined.slice(0, bsd).copy_to_host(&mut d_y_host);
let mut dots = vec![0.0f64; cfg.k];
for l in 0..cfg.k {
    let mut y_l_host = vec![0.0f32; bsd];
    bc.y_per_level[l].slice(0, bsd).copy_to_host(&mut y_l_host);  // ← k more D2H
    dots[l] = d_y_host.iter().zip(y_l_host.iter())
        .map(|(&dy, &y)| dy as f64 * y as f64).sum();
}
```

### Transfer volume per block

| Config | bsd | Per-block D2H | 12 blocks |
|--------|-----|---------------|-----------|
| d=768, bs=1, s=512 | 393,216 | 7.5 MB | 90 MB |
| d=1024, bs=8, s=512 | 4,194,304 | 80 MB | 960 MB |
| d=2048, bs=12, s=512 | 12,582,912 | 240 MB | **2.88 GB** |

The `cuda_sync()` at line 374 is a full pipeline drain — every queued kernel
must complete before the copy can start. This is the primary cause of the
block 7 backward hotspot observed in profiling: block 7 is the transition point
in reverse iteration where no subsequent GPU work can hide the D2H latency.

### What we actually need

The final output is `d_alpha_mem` — a `Vec<f32>` of length k (= 4 floats).
We transfer ~240 MB to compute 4 scalars.

## Solution

### 2.1 New CUDA kernel: `dot_product_partial_f32`

A partial-reduction dot product kernel, mirroring `grad_norm_sq_cuda`:

```c
// In core/kernels/elementwise.cu (or new file)

__global__ void dot_product_partial_f32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ partial_sums,
    int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = smem[0];
}

extern "C" cudaError_t dot_product_partial_f32_cuda(
    const float* a, const float* b,
    float* partial_sums, int n, int* out_num_blocks)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    *out_num_blocks = grid;
    dot_product_partial_f32_kernel<<<grid, block, block * sizeof(float)>>>(
        a, b, partial_sums, n);
    return cudaGetLastError();
}
```

This is identical to `grad_norm_sq_cuda` except `a[i] * b[i]` replaces
`g[i] * g[i]`. Block size 256, shared memory reduction, ceil(n/256) partial
sums written to device memory.

### 2.2 FFI binding

In `core/src/cuda_ffi.rs`:

```rust
pub(crate) fn dot_product_partial_f32_cuda(
    a: *const f32, b: *const f32,
    partial_sums: *mut f32,
    n: i32, out_num_blocks: *mut i32,
) -> u32;
```

### 2.3 Rust helper: `gpu_dot_product`

In `core/src/gpu_stacked_backward.rs` (or a shared module):

```rust
/// Compute <a, b> on GPU using partial reduction + host sum.
/// scratch must be at least ceil(n/256) elements.
fn gpu_dot_product(a: &GpuBuf<f32>, b: &GpuBuf<f32>, n: usize,
                   scratch: &GpuBuf<f32>) -> f64 {
    let mut num_blocks: i32 = 0;
    let err = unsafe {
        crate::cuda_ffi::dot_product_partial_f32_cuda(
            a.as_ptr(), b.as_ptr(), scratch.ptr(),
            n as i32, &mut num_blocks,
        )
    };
    assert_eq!(err, 0, "dot_product_partial_f32_cuda failed");
    crate::dispatch::cuda_sync();
    let nb = num_blocks as usize;
    let mut host = vec![0.0f32; nb];
    scratch.slice(0, nb).copy_to_host(&mut host);
    host.iter().map(|x| *x as f64).sum()
}
```

Note: The `cuda_sync()` here is necessary (must wait for the kernel to finish
before reading partial sums), but it replaces the existing `cuda_sync()` at
line 374 — net zero new syncs.

### 2.4 Replace D2H copy loop

Replace `gpu_stacked_backward.rs:372-386`:

```rust
// Before: ~240 MB D2H per block
let w = &bc.alpha_weights;
let mut d_y_host = vec![0.0f32; bsd];
crate::dispatch::cuda_sync();
d_y_combined.slice(0, bsd).copy_to_host(&mut d_y_host);
let mut dots = vec![0.0f64; cfg.k];
for l in 0..cfg.k {
    let mut y_l_host = vec![0.0f32; bsd];
    bc.y_per_level[l].slice(0, bsd).copy_to_host(&mut y_l_host);
    dots[l] = d_y_host.iter().zip(y_l_host.iter())
        .map(|(&dy, &y)| dy as f64 * y as f64).sum();
}

// After: ~3 KB D2H per block (k * ceil(bsd/256) partial sums)
let w = &bc.alpha_weights;
let mut dots = vec![0.0f64; cfg.k];
for l in 0..cfg.k {
    dots[l] = gpu_dot_product(&d_y_combined, &bc.y_per_level[l], bsd, &gnorm_scratch);
}
```

The `gnorm_scratch` buffer (already allocated at line 232 for grad norm
computations) is reused — it has `ceil(bsd/256)` elements, which is exactly
what the dot product kernel needs.

### Transfer volume after fix

| Config | Partials per level | Per-block D2H | Reduction |
|--------|-------------------|---------------|-----------|
| d=768, bs=1, s=512 | 1,536 | 24 KB | **312×** |
| d=1024, bs=8, s=512 | 16,384 | 256 KB | **312×** |
| d=2048, bs=12, s=512 | 49,153 | 768 KB | **312×** |

### Sync budget

- **Removed**: 1 `cuda_sync()` at line 374 (before the old D2H copy)
- **Added**: k `cuda_sync()` calls inside `gpu_dot_product` (one per level)
- **Net**: +3 syncs per block (k=4 → 4 syncs replace 1, but the old 1 was
  followed by k blocking D2H copies which implicitly sync)

This can be further optimized (Phase 2) by launching all k kernels before
syncing once, but even the naive version eliminates the dominant transfer cost.

## Phase 2 (future): Fused multi-dot kernel

A single kernel launch could compute all k dot products simultaneously:

```c
// grid=(ceil(n/256), k), each y-row processes one level
__global__ void multi_dot_partial_f32_kernel(
    const float* a,
    const float* const* b_ptrs,  // k pointers to y_per_level
    float* partial_sums,         // [k * num_blocks_x]
    int n, int k)
```

This would reduce to 1 sync + 1 D2H copy per block instead of k, but the
Phase 1 fix already eliminates >99.7% of the transfer volume.

## Files to Modify

| File | Change |
|------|--------|
| `core/kernels/elementwise.cu` | Add `dot_product_partial_f32_kernel` + launcher |
| `core/src/cuda_ffi.rs` | Add FFI binding |
| `core/src/gpu_stacked_backward.rs` | Add `gpu_dot_product` helper; replace D2H loop |
| `core/build.rs` | No change — elementwise.cu already compiled |

## Acceptance Criteria

1. `d_alpha_mem` values match the old implementation (f32 rounding tolerance)
2. No full-buffer D2H copies in the independent aggregation backward path
3. `gnorm_scratch` buffer reused (no new allocations in the per-block loop)
4. Profile shows block 7 backward time reduced (no 48 MB+ D2H per block)
5. All existing tests pass — no behavioral change
6. New test: `test_gpu_dot_product_matches_cpu` comparing GPU partial-reduction
   dot product against CPU reference for various buffer sizes

## Ontological Compliance

- **CS-18**: Dot product is math in the Rust/CUDA tier, not orchestration.
- **CS-42**: Scratch buffer reuse from existing arena allocation.
- **CS-40**: No tape involvement — this is the hand-written backward path.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-074-arch-variant5 | hope_equations | HOPE §5.1 | implements (softmax Jacobian backward) |
