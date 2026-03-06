# Large Dimension Kernel Restructuring (d > 1024)

## CONTRACT
- **Purpose**: Remove d <= 1024 ceiling from all memory update kernels and D <= 512 ceiling from gate_backward. Enable d=2048 for 700M-1B HOPE model on H100.
- **Expects**: All 10 kernel files with `if (tid < d)` patterns that assume blockDim.x >= d.
- **Guarantees**: (1) All matvec and elementwise-over-d operations use strided loops. (2) Kernels produce identical output when blockDim.x >= d. (3) Kernels produce correct output when blockDim.x < d. (4) Shared memory guard rejects d values that exceed GPU smem capacity.
- **Cost**: Zero performance regression at d <= 1024 (strided loop with single iteration = same code path). Negligible overhead at d > 1024 (extra loop iterations amortized over d^2 matrix operations).
- **Trade-off**: Simplicity over specialization. A single code path handles all d values rather than separate kernels for different ranges.
- **Position**: `specs/infrastructure/cuda/05_large_dimension_kernel_restructuring.md`
- **Source**: Empirical — H100 testing of d=2048 HOPE model hit silent data corruption from unwritten elements. Pure CUDA infrastructure (block size limits, shared memory carve-outs); no paper equations apply. NVIDIA CUDA Programming Guide §5.2.3 (Thread Block Size), §16.5.1 (Shared Memory). Cross-ref: `specs/infrastructure/cuda/04_hopper_kernel_optimization.md` §4 (shared memory budget tables).

## Root Cause

Prediction/readout loops use `if (tid < d)` — this requires blockDim.x >= d. CUDA maximum block size is 1024. At d=2048, threads with tid in [1024, 2048) never exist, so elements in that range are never written.

The M-matrix update loops already use strided patterns (`for (int idx = tid; idx < dd; idx += blockDim.x)`) and work at any d. Only the matvec and elementwise-over-d patterns need the same treatment.

## The Fix: One Pattern Transformation

Every `if (tid < d)` becomes a strided loop:

```cuda
// BEFORE (breaks when blockDim.x < d):
if (tid < d) {
    float sum = 0.0f;
    for (int j = 0; j < d; j++)
        sum += M[tid * d + j] * vec[j];
    output[tid] = sum;
}
__syncthreads();

// AFTER (works for any d):
for (int row = tid; row < d; row += blockDim.x) {
    float sum = 0.0f;
    for (int j = 0; j < d; j++)
        sum += M[row * d + j] * vec[j];
    output[row] = sum;
}
__syncthreads();
```

When blockDim.x >= d, the strided loop executes exactly once with row=tid — identical behavior to the original `if (tid < d)`.

## Files Modified

### Memory Forward Kernels (4 files)
- `core/kernels/hebbian_forward.cu` — readout only
- `core/kernels/delta_forward.cu` — prediction + error + readout
- `core/kernels/titans_forward.cu` — prediction + error + readout
- `core/kernels/dgd_forward.cu` — prediction + error + readout

### Memory Backward Kernels (4 files)
- `core/kernels/hebbian_backward.cu` — d_q, d_v, d_k
- `core/kernels/delta_backward.cu` — d_q, prediction recompute, error, d_error, d_k, d_v
- `core/kernels/titans_backward.cu` — d_q, prediction recompute, error, d_error, d_k, d_v
- `core/kernels/dgd_backward.cu` — d_q, prediction recompute, error, d_error, d_k, d_v

### Gate Kernels (2 files)
- `core/kernels/gate_backward.cu` — remove assert(D <= 512), strided loop
- `core/kernels/elementwise.cu` — raise gate_compute block cap from 512 to 1024

## Dimension Guard

C wrapper guards change from hard `d > 1024` rejection to shared-memory-based limit. The maximum d is determined by the Ampere path's shared memory requirement:

- Forward kernels: 8*d floats = 32*d bytes. At d=5120: 160KB (within Hopper 228KB).
- Backward kernels: (3*d + block + 8*d) floats. At d=2048, block=1024: ~91KB (within Ampere 99KB).

Guard formula: `smem_multiplier * d * sizeof(float) > 163840` rejects configurations exceeding 160KB (conservative Hopper limit). Per-kernel multipliers match actual shared memory layout: 8 for forward (Delta/Titans/DGD), 6 for Hebbian forward, 2 for checkpoint-mode forward. Source: NVIDIA CUDA Programming Guide §16.5.1, validated empirically on sm_86 (Ampere, 100KB) and sm_90 (Hopper, 228KB).

## Block Size Policy

- Forward kernels: `min(d*d, 1024)` — unchanged
- Backward kernels: `min(d, 1024)` with power-of-2 rounding — unchanged
- Gate backward: `min(2*D, 1024)` — raised from hard D <= 512

## Shared Memory at d=2048

| Kernel | Layout | Bytes |
|--------|--------|-------|
| Forward (Ampere) | 8*d floats | 64KB |
| Forward (ckpt, delta/titans) | 2*d floats | 16KB |
| Forward (ckpt, dgd) | 8*d floats | 64KB |
| Forward (ckpt, hebbian) | 0 | 0 |
| Backward (main) | (3*d + block + 8*d) floats | ~91KB |
| Backward (segment) | (3*d + block) floats | ~28KB |

All within Ampere 99KB / Hopper 228KB limits.

## Build System Change

`core/build.rs` forces optimized CUDA compilation (`opt_level(2)`, `debug(false)`) even in Rust debug profile. Without this, the `cc` crate passes nvcc `-G` (device debug) which doubles register usage from ~40 to 80+ registers per thread. At 1024 threads per block, 80 registers × 1024 = 81920 exceeds the 65536-register-per-SM limit on Ampere, causing "too many resources requested for launch" at d >= 1536.

## Regression Safety

When blockDim.x >= d (the d <= 1024 case that all existing tests exercise), the strided loop body executes exactly once per thread with loop variable equal to tid. This is algebraically identical to the original `if (tid < d)` guard. No existing test can observe different behavior.

## Test Coverage

`core/tests/test_cuda_large_d.rs` — 10 tests validating d > 1024:

| Test | d | What |
|------|---|------|
| delta forward | 2048 | Full numerical match vs Rust reference |
| delta backward | 2048 | Full numerical match vs Rust reference (all 6 gradients) |
| hebbian forward | 2048 | Full numerical match vs Rust reference |
| hebbian backward | 2048 | Non-zero gradient sanity check |
| titans forward | 2048 | Full numerical match vs Rust reference |
| titans backward | 2048 | Non-zero gradient sanity check |
| dgd forward | 2048 | Full numerical match vs Rust reference |
| dgd backward | 2048 | Non-zero gradient sanity check |
| gate backward | D=1024 | Verifies upper-half weights [D, 2D) written |
| delta forward | 1536 | Odd dimension between 1024 and 2048 |
