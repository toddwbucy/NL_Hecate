# SwiGLU CUDA Kernel Pair

```text
CONTRACT
  Purpose:    GPU-accelerated SwiGLU MLP forward and backward for CMS level stacking
              (HOPE §7.3). Host-pointer variant for checkpoint I/O; device-to-device
              (_dd) variant for zero-PCIe step_adamw() GPU training path.
  Expects:    GpuMemoryLevelParams with has_mlp=true and gate_proj/up_proj/down_proj
              allocated on device. Shared cuBLAS handles (g_cublas_handle_fwd/_bwd).
  Guarantees: Forward parity ≤ 1e-5 vs Rust reference. Backward weight grads ≤ 1e-4.
              fp32 throughout (inner-loop non-negotiable). No cudaDeviceSynchronize in
              _dd variants — caller syncs via dispatch::cuda_sync().
  Cost:       O(s·d·inter) per level per step — 3 cuBLAS GEMMs + elementwise SiLU.
              d=2048, inter=8192: ~25 GFLOPs/level at seq_len=512.
  Trade-off:  host-pointer: simple, H2D/D2H each call, uses persistent g_fwd/bwd_pool.
              _dd: zero PCIe, caller provides GpuBufs, no pool allocation inside kernel.
  Position:   specs/infrastructure/cuda/03_swiglu_kernels.md
  Source:     HOPE (2512.24695) §7.3, Eq 71, Eq 72
```

## 1. SwiGLU Math (HOPE §7.3, Eq 71–72)

SwiGLU MLP as a stateless CMS level — no inner-loop M state. The three weight matrices
(gate_proj, up_proj, down_proj) are **outer-loop AdamW parameters**.

```text
gate_out = X @ gate_proj.T             [s × inter]
up_out   = X @ up_proj.T               [s × inter]
sig      = sigmoid(gate_out)           [s × inter]
fused    = gate_out * sig * up_out     [s × inter]   (SwiGLU activation)
Y        = fused @ down_proj.T         [s × d_model]
```

Backward (standard backprop):
```text
d_fused     = d_Y @ down_proj
d_down_proj = fused.T @ d_Y
dsilu[i]    = sig[i] * (1 + gate_out[i] * (1 - sig[i]))
d_up[i]     = d_fused[i] * gate_out[i] * sig[i]
d_gate[i]   = d_fused[i] * up_out[i] * dsilu[i]
d_gate_proj = d_gate.T @ X
d_up_proj   = d_up.T @ X
d_X         = d_gate @ gate_proj + d_up @ up_proj
```

## 2. Kernel-Pair Variants

Two variants, four files:

| File | Function | Variant | Notes |
|------|----------|---------|-------|
| `swiglu_forward.cu` | `swiglu_forward_f32_cuda` | host-pointer | PCIe each call; uses persistent pool |
| `swiglu_forward.cu` | `swiglu_forward_f32_cuda_dd` | device-to-device | zero PCIe; caller provides GpuBufs |
| `swiglu_backward.cu` | `swiglu_backward_f32_cuda` | host-pointer | PCIe each call; uses persistent pool |
| `swiglu_backward.cu` | `swiglu_backward_f32_cuda_dd` | device-to-device | zero PCIe; caller provides GpuBufs |

## 3. Forward Signatures

### 3.1 Host-pointer variant (existing)

```cpp
extern "C" void swiglu_forward_f32_cuda(
    const float* X,           // host: [seq_len × d_model]
    const float* gate_proj,   // host: [intermediate × d_model]
    const float* up_proj,     // host: [intermediate × d_model]
    const float* down_proj,   // host: [d_model × intermediate]
    float* Y,                 // host: [seq_len × d_model]
    float* gate_buf,          // host: [seq_len × intermediate] saved for bwd
    float* up_buf,            // host: [seq_len × intermediate]
    float* fused_buf,         // host: [seq_len × intermediate]
    float* cache_buf,         // host: [seq_len × intermediate] sigmoid cache
    int seq_len, int d_model, int intermediate);
```

**Buffer lifecycle**: Allocates persistent device buffers (`g_fwd_pool`) on first call.
All H2D/D2H copies happen inside the function. Caller receives host-side activation copies.

### 3.2 Device-to-device variant (_dd, new)

```cpp
extern "C" void swiglu_forward_f32_cuda_dd(
    const float* X,           // device: [seq_len × d_model]
    const float* gate_proj,   // device: [intermediate × d_model]
    const float* up_proj,     // device: [intermediate × d_model]
    const float* down_proj,   // device: [d_model × intermediate]
    float* Y,                 // device: [seq_len × d_model]
    float* gate_buf,          // device: [seq_len × intermediate] saved for bwd
    float* up_buf,            // device: [seq_len × intermediate]
    float* fused_buf,         // device: [seq_len × intermediate]
    float* cache_buf,         // device: [seq_len × intermediate] sigmoid cache
    int seq_len, int d_model, int intermediate);
```

**Buffer lifecycle**: All pointers are device memory provided by the caller (GpuBuf<f32>
fields in GpuMemoryCache::SwiGlu). No pool allocation inside. No cudaDeviceSynchronize —
caller syncs via `dispatch::cuda_sync()` after the full level loop.

**Same cuBLAS handle**: Reuses `get_cublas_handle_fwd()` — no new handle allocation.

## 4. Backward Signatures

### 4.1 Host-pointer variant (existing)

```cpp
extern "C" void swiglu_backward_f32_cuda(
    const float* d_Y,         // host: [seq_len × d_model]
    const float* X,           // host: [seq_len × d_model]
    const float* gate_proj,   // host: [intermediate × d_model]
    const float* up_proj,     // host: [intermediate × d_model]
    const float* down_proj,   // host: [d_model × intermediate]
    const float* fused_buf,   // host: [seq_len × intermediate]
    const float* gate_buf,    // host: [seq_len × intermediate]
    const float* up_buf,      // host: [seq_len × intermediate]
    const float* cache_buf,   // host: [seq_len × intermediate] sigmoid
    float* d_X,               // host: [seq_len × d_model]
    float* d_gate_proj,       // host: [intermediate × d_model]
    float* d_up_proj,         // host: [intermediate × d_model]
    float* d_down_proj,       // host: [d_model × intermediate]
    int seq_len, int d_model, int intermediate);
```

### 4.2 Device-to-device variant (_dd, new)

```cpp
extern "C" void swiglu_backward_f32_cuda_dd(
    const float* d_Y,         // device: [seq_len × d_model]
    const float* X,           // device: [seq_len × d_model]
    const float* gate_proj,   // device: [intermediate × d_model]
    const float* up_proj,     // device: [intermediate × d_model]
    const float* down_proj,   // device: [d_model × intermediate]
    const float* fused_buf,   // device: [seq_len × intermediate]
    const float* gate_buf,    // device: [seq_len × intermediate]
    const float* up_buf,      // device: [seq_len × intermediate]
    const float* cache_buf,   // device: [seq_len × intermediate] sigmoid
    float* d_X,               // device: [seq_len × d_model]
    float* d_gate_proj,       // device: [intermediate × d_model]
    float* d_up_proj,         // device: [intermediate × d_model]
    float* d_down_proj,       // device: [d_model × intermediate]
    int seq_len, int d_model, int intermediate);
```

**Weight grads written with beta=0**: d_gate_proj/d_up_proj/d_down_proj are freshly
computed each call (no accumulation across CMS levels). The GpuBuf::zeros() output buffers
in GpuLevelGrads::zeros_mlp() provide pre-zeroed storage.

## 5. cuBLAS Layout Rationale

All weight matrices stored row-major in Rust. cuBLAS is column-major. The _dd variants
use the same cuBLAS row-major tricks as the host-pointer variants:

- `gate_proj[inter × d]` with `lda=d_model` → cuBLAS sees `gate_proj.T` (col-major [d × inter])
  - `transa=T` transposes back → result `gate_buf[seq × inter]` ✓
- `down_proj[d × inter]` with `lda=inter` → cuBLAS sees `down_proj.T` (col-major [inter × d])
  - `transa=T` transposes back → result `Y[seq × d]` ✓

No new scratch buffers needed in _dd variant — all intermediates (dDFused, dDGateOut,
dDUpOut) are allocated from device scratch GpuBuf<f32> locals inside the function.

## 6. Rust Integration (gpu_forward.rs)

SwiGLU is **always active** regardless of the Pulse signal — mirrors CPU path in `mag.rs:615`:
```rust
let active = active || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);
```

The GPU path implements this via `effective_active` in the level loop:
```rust
let effective_active = pulse.active_levels[level]
    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);
```

`GpuContextState` provides `zeros(d*d)` dummy M buffers for SwiGLU levels. The
`SwiGluMlp` match arm in `gpu_memory_forward` ignores `context_m` entirely.

## 7. Optimizer Integration (gpu_optimizer.rs)

Pulse-gated optimizer still applies: SwiGLU Level 1 weights only update every 8 steps,
Level 2 every 64 steps, etc. Per-level bias correction uses `level_step` counter as with
matrix rules.
