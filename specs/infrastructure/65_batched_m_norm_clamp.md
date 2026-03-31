# 65 — Batched M-Norm Clamp

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Replace per-batch-head `m_norm_clamp` loop (up to `bs*nh` separate kernel launches per level) with a single batched kernel launch |
| Expects   | Working `m_norm_clamp_f32_cuda` single-matrix kernel; per-head memory layout (spec 45) with heads folded into batch dim |
| Guarantees | Bit-identical output vs single-matrix kernel called in a loop; massive launch count reduction (e.g. 576 → 24 launches per step at d=768 nh=12 bs=2 k=4 n_blocks=6) |
| Cost      | One new kernel + one FFI declaration + one dispatch wrapper |
| Trade-off | Grid=(batch_size) means one block per matrix — no inter-matrix parallelism within a block, but this matches the existing single-matrix kernel's strategy |
| Position  | Prerequisite for spec 66 (multi-head fused recurrence kernel); immediate throughput improvement |
| Source    | gpu_forward.rs: 10 separate clamp-loop sites, each iterating `bs_mem` times; task_ed9b3d |

## Problem

The existing `m_norm_clamp_f32_cuda` clamps a single d×d matrix per launch. After each
forward pass, Rust code loops over all batch-head elements:

```rust
for bh in 0..bs_mem {
    m_norm_clamp_f32_cuda(context_m + bh * dd, d, m_norm_max);
}
```

At d_model=768, nh=12, bs=2: each head has an hd×hd memory matrix where hd = d_model/nh = 64.
The batch-head count is `bs_mem = bs * nh = 24`, so the loop fires 24 times per level,
× 4 levels × 6 blocks = **576 separate launches per step** just for M-norm clamping.
Each launch clamps a single 64×64 (hd×hd) matrix but pays full kernel launch overhead (~5-10 µs).

## Design: Batched Kernel

New kernel `m_norm_clamp_batch_kernel` with `grid=(batch_size)`. Each block independently
clamps its own d×d matrix — identical math to the single-matrix kernel, just indexed by
`blockIdx.x`.

### Kernel

```c
__global__ void m_norm_clamp_batch_kernel(float* m, int dd, float m_norm_max) {
    extern __shared__ float s_norm[];
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float* m_b = m + b * dd;

    // Sum-of-squares reduction (identical to single-matrix kernel)
    float local = 0.0f;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        float v = m_b[idx];
        local += v * v;
    }
    s_norm[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm[tid] += s_norm[tid + s];
        __syncthreads();
    }

    float fnorm = sqrtf(s_norm[0]);
    if (fnorm > m_norm_max) {
        float scale = m_norm_max / fnorm;
        for (int idx = tid; idx < dd; idx += blockDim.x)
            m_b[idx] *= scale;
    }
}
```

### C wrapper

```c
extern "C" void m_norm_clamp_batch_f32_cuda(float* m, int d, int batch_size, float m_norm_max) {
    if (m_norm_max <= 0.0f || m_norm_max >= 1e30f) return;
    if (batch_size <= 0) return;
    int dd = d * d;
    int block_size = 1;
    while (block_size * 2 <= d && block_size < 1024) block_size <<= 1;
    int smem_bytes = block_size * sizeof(float);
    m_norm_clamp_batch_kernel<<<batch_size, block_size, smem_bytes>>>(m, dd, m_norm_max);
}
```

## Dispatch Changes (Rust side)

### cuda_ffi.rs

```rust
pub(crate) fn m_norm_clamp_batch_f32_cuda(m: *mut f32, d: i32, batch_size: i32, m_norm_max: f32);
```

### dispatch.rs

Public wrappers for both single and batched variants:

```rust
pub fn m_norm_clamp(m: &mut GpuBuf<f32>, d: i32, m_norm_max: f32);
pub fn m_norm_clamp_batch(m: &mut GpuBuf<f32>, d: i32, batch_size: i32, m_norm_max: f32);
```

### gpu_forward.rs

Replace all 10 clamp-loop sites with batched calls:

```rust
// Before (repeated 10 times):
for bh in 0..bs_mem {
    m_norm_clamp_f32_cuda(context_m + bh * dd * 4, d, m_norm_max);
}

// After:
m_norm_clamp_batch_f32_cuda(context_m, d, bs_mem as i32, m_norm_max);
```

5 main-path sites use `bs_mem as i32`, 2 fused-path sites use `bs as i32`,
3 checkpoint-path sites use `nh as i32` (extracted from per-head copy loop).

## Files Modified

| File | Change |
|------|--------|
| `core/kernels/m_norm_clamp.cu` | Add `m_norm_clamp_batch_kernel` + C wrapper |
| `core/src/cuda_ffi.rs` | Add 1 FFI declaration |
| `core/src/dispatch.rs` | Add 2 dispatch wrappers (single + batch) |
| `core/src/gpu_forward.rs` | Replace 10 clamp loops with batched calls |

## Validation

1. `cargo test --features cuda` — all existing tests pass
2. New test: `test_batched_clamp_matches_single_loop` — for (d, batch_size) in
   [(32,8), (64,4), (128,2), (64,1)], run both paths and compare element-wise (tol 1e-6)
3. New test: `test_batched_clamp_noop_when_below_threshold` — high m_norm_max, verify no-op
4. Training equivalence: loss/gnorm unchanged vs pre-batched baseline

## Success Criteria

1. All 10 clamp-loop sites replaced with single batched calls
2. Bit-identical output vs single-matrix loop (validated by unit test)
3. `cargo test --features cuda` passes
4. No training regression

## Non-Goals

- No changes to the fused recurrence kernels (that's spec 66)
- No changes to gate computation or head transpose
- The single-matrix `m_norm_clamp_f32_cuda` kernel is preserved (backward compat)
