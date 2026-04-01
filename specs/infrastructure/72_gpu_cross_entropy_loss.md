# Spec 72: GPU-Side Cross-Entropy Loss for Stacked Forward Path

## CONTRACT
- **Purpose**: Eliminate the host-side logits readback bottleneck by wiring the existing GPU cross-entropy kernel into the stacked forward path.
- **Expects**: `cross_entropy_forward_cuda` kernel (already in `core/kernels/cross_entropy.cu`), `GpuStackedCache` with `logits` (GPU), `target_ids_gpu` (GPU), `target_ids_i32` (host).
- **Guarantees**: Loss computation stays entirely on GPU. D2H transfer drops from `seq_len × vocab_size × 4` bytes to 4 bytes per step. Loss values match within f32 tolerance (GPU `atomicAdd` reduction order may differ from sequential host sum). No change to backward path (already uses `cross_entropy_backward_cuda`).
- **Cost**: Zero new kernels, zero new CUDA code. Wiring change only.
- **Trade-off**: None — strictly eliminates waste.
- **Position**: Infrastructure optimization. Unblocks sustained GPU utilization for build runs.
- **Source**: nvtop profiling showing GPU idle during host CE computation. At seq_len=4096 v=49152: 768 MB D2H per step. At seq_len=32768: 6.4 GB D2H per step.

## Problem

`host_cross_entropy_loss` in `cli/src/step.rs` copies the **entire** logits buffer from GPU to host, then computes log-sum-exp on the CPU:

```rust
let mut logits_host = vec![0.0f32; seq_len * vocab_size];  // 768 MB at s=4096
logits_gpu.copy_to_host(&mut logits_host);                  // sync + D2H
// ... CPU loop over seq_len × vocab_size elements
```

This is the dominant pipeline stall visible in nvtop — the GPU sits idle while the host processes hundreds of millions of floats.

Meanwhile, `core/kernels/cross_entropy.cu` already implements fused softmax + NLL on GPU, producing a single scalar via `atomicAdd`. The backward path (`gpu_stacked_backward.rs:121`) already uses this kernel. The forward path simply never got wired to it.

## Solution

Replace `host_cross_entropy_loss(...)` with `cross_entropy_forward_cuda(...)` + 4-byte scalar readback.

### Changes

| File | Change |
|------|--------|
| `cli/src/step.rs` | `step()`: replace `host_cross_entropy_loss` call with GPU CE forward + scalar D2H. Compute `valid_count` from host-side `targets` (same pattern as backward). |
| `cli/src/step.rs` | `generate()`: same replacement for the deferred-backward loss computation. |
| `cli/src/step.rs` | Delete `host_cross_entropy_loss` function (dead code after wiring). |

### step() loss computation — before

```rust
let loss = host_cross_entropy_loss(&cache.logits, targets, v, tokens.len());
```

### step() loss computation — after

```rust
let loss = gpu_cross_entropy_loss(&cache.logits, &cache.target_ids_gpu, targets, v, tokens.len());
```

Where `gpu_cross_entropy_loss` is a thin wrapper:

```rust
fn gpu_cross_entropy_loss(
    logits_gpu: &GpuBuf<f32>,
    target_ids_gpu: &GpuBuf<f32>,
    target_ids_host: &[usize],
    vocab_size: usize,
    seq_len: usize,
) -> f32 {
    let valid_count = target_ids_host.iter()
        .filter(|&&t| t < vocab_size)
        .count();
    if valid_count == 0 { return 0.0; }

    let loss_buf = GpuBuf::<f32>::zeros(1);
    unsafe {
        nl_hecate_core::cuda_ffi::cross_entropy_forward_cuda(
            logits_gpu.as_ptr(),
            target_ids_gpu.ptr() as *const i32,
            loss_buf.ptr(),
            seq_len as i32, vocab_size as i32,
        );
    }
    nl_hecate_core::dispatch::cuda_sync();

    let mut loss_host = [0.0f32; 1];
    loss_buf.copy_to_host(&mut loss_host);
    loss_host[0] / valid_count as f32
}
```

### Data flow

```text
Before:  logits [s×v GPU] → copy_to_host [s×v host] → CPU log-sum-exp → scalar
After:   logits [s×v GPU] → GPU kernel → scalar [1 GPU] → copy_to_host [1 host]

D2H bytes:  s × v × 4  →  4
At s=4096:  768 MB      →  4 bytes
At s=32768: 6.4 GB      →  4 bytes
```

### valid_count on host

The host-side count over `target_ids` (a few thousand `usize` values) takes microseconds — negligible vs the eliminated D2H copy. The backward path already uses the same pattern (`cache.target_ids_i32.iter().filter(...).count()`).

## Backward — no changes

`gpu_stacked_backward.rs:121` already calls `cross_entropy_backward_cuda` on the GPU logits buffer. The backward path is already correct.

## NaN detection

The 4-byte scalar loss is still read back every step. NaN/Inf detection works exactly as before — check `loss.is_nan() || loss.is_infinite()` on the host-side scalar.

## Verification

1. `cargo test --features cuda --lib` — all existing tests pass
2. Loss values match `host_cross_entropy_loss` within f32 tolerance (`atomicAdd` reduction order differs from sequential host sum)
3. nvtop: GPU idle gaps between forward and backward should shrink dramatically
4. tok/s improvement measurable on d=1024 build
