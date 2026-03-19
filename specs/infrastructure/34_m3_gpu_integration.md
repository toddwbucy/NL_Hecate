# Spec 34: M3 GPU Integration

## CONTRACT

| Field | Value |
|-------|-------|
| **Purpose** | GPU-resident M3 optimizer replacing AdamW for CMS k>=2 builds, with per-param 2D/1D update split, Newton-Schulz orthogonalization, and Python config/dispatch |
| **Expects** | Existing `GpuMAGParams` + `GpuMAGGrads` from forward/backward pass; `Pulse` with active CMS levels; M3Config from Python tier |
| **Guarantees** | Frequency-aware optimization (CS-27/CS-28): momentum levels match CMS architecture levels. 2D params get NS-orthogonalized updates; 1D params get Adam-style V-divided updates. Checkpoint snapshot/restore follows spec 33 pattern. |
| **Cost** | Per step: k EMA updates (M1) + 1 V update + conditional M2 update. NS: T iterations of d x d matmul per 2D param group (GPU-bound, overlappable). Memory: 3 buffers per param (M1 + M2 + V) in f32 = 12 bytes/param. |
| **Trade-off** | NS adds O(T * d^2) compute per 2D param group per step. For d=512, T=5: ~5 small matmuls. Acceptable: replaces AdamW's per-element bias correction which doesn't scale to multi-frequency. |
| **Position** | Rust tier (gpu_optimizer.rs, m3.rs) + CUDA tier (kernels/) + Python tier (lib.rs, config.py, loop.py) |
| **Source** | HOPE (2512.24695) Eq 42, 44, 75; spec 02_m3.md (algorithm); CS-27, CS-28 |

## Design

### Phase 1: Rust GPU State (`gpu_optimizer.rs`)

```rust
pub struct GpuM3State {
    /// Per-param-group M1 (fast momentum), M2 (slow momentum), V (second moment)
    pub swa: M3MomentSWA,          // SWA-level params (embed, Q, K, V, O, unembed, LN)
    pub levels: Vec<M3MomentLevel>, // per CMS level (k_mem, v_mem, q_mem, gates)
    pub step: u32,
    pub chunk_size: u32,            // Ĉ — slow momentum update frequency
    pub config: GpuM3Config,
}

pub struct GpuM3Config {
    pub beta1: f32,    // fast momentum coefficient (default 0.9)
    pub beta2: f32,    // second moment coefficient (default 0.999)
    pub beta3: f32,    // slow momentum coefficient (default 0.99)
    pub alpha: f32,    // weight of slow momentum (default 0.5)
    pub ns_iterations: u32, // Newton-Schulz iterations T (default 5)
    pub eps: f32,      // numerical stability (default 1e-8)
}
```

Each `M3MomentSWA` and `M3MomentLevel` mirrors the existing `MomentSWA`/`MomentLevel`
structure but with three buffers per param instead of two (m, v → m1, m2, v).

### Phase 2: CUDA Kernels

**Kernel 1: `m3_ema_update`** — Fused M1 + V + conditional M2 update.
```
For each element i:
  m1[i] = beta1 * m1[i] + (1 - beta1) * grad[i]
  v[i]  = beta2 * v[i]  + (1 - beta2) * grad[i]^2
  if step % chunk_size == 0:
    m2[i] = beta3 * m2[i] + (1 - beta3) * grad[i]
```
One kernel launch per param group. Grid: ceil(n / 256). Simple elementwise, high bandwidth utilization.

**Kernel 2: `m3_apply_1d`** — Adam-style param update for 1D params.
```
For each element i:
  bc = 1.0 - beta2^step
  update = (m1[i] + alpha * m2[i]) / (sqrt(v[i] / bc) + eps)
  param[i] -= lr * update
```

**Kernel 3: `m3_apply_2d`** — Muon-style param update for 2D params.
NS is called from Rust (cuBLAS matmul) rather than a custom kernel — the matmul
is the hot path and cuBLAS is already optimized for it. The combination + param
update is a simple elementwise kernel:
```
For each element i:
  param[i] -= lr * (o1[i] + alpha * o2[i])
```

**Newton-Schulz (Rust + cuBLAS)**:
```rust
fn gpu_newton_schulz(m: &GpuBuf<f32>, rows: usize, cols: usize,
                     iterations: usize, scratch: &mut NSScratch) -> GpuBuf<f32> {
    // Transpose tall matrices: work on min(rows, cols) dimension
    // T=5 polynomial: a=3.4445, b=-4.7750, c=2.0315
    // Uses cublasSgemm for A = X @ X^T and X_new = a*X + b*(A@X) + c*(A@(A@X))
}
```
cuBLAS is already linked (used by existing matmul paths). NS scratch buffers
are allocated once in GpuM3State and reused.

### Phase 3: Python Integration

**Config** (`config.py`):
```python
# In BuildConfig:
optimizer: str = "adamw_gpu_stacked"  # existing default
m3_beta1: float = 0.9
m3_beta2: float = 0.999
m3_beta3: float = 0.99
m3_alpha: float = 0.5
m3_chunk_size: int = 8
m3_ns_iterations: int = 5
```

When `optimizer: "m3"` is set, loop.py creates `GpuM3State` instead of
`GpuAdamWState`. The optimizer field accepts: `"adamw"`, `"adamw_gpu"`,
`"adamw_gpu_stacked"`, `"m3"`.

**Loop dispatch** (`loop.py`):
```python
if bcfg.optimizer == "m3":
    gpu_model.step_m3(lr=bcfg.lr)  # calls gpu_m3_update in Rust
else:
    gpu_model.step_adamw(lr=bcfg.lr)  # existing path
```

**PyO3 bindings** (`lib.rs`):
- `GpuModel.init_m3(config)` — creates GpuM3State from GpuM3Config
- `GpuModel.step_m3(lr)` — one M3 optimizer step (EMA + NS + apply)
- M3 state participates in spec 33 snapshot/restore

### Phase 4: Checkpoint

M3 state snapshot/restore follows the spec 33 `HostOptimizerState` pattern:
- `GpuM3State::to_host()` → flat Vec<f32> concatenation of all M1, M2, V buffers
- `GpuM3State::from_host()` → upload and reconstruct

The Python `full_snapshot`/`full_restore` path handles M3 transparently — the
`snapshot_optimizer`/`restore_optimizer` methods dispatch based on which optimizer
is active.

## Scope

### Files to modify
- `core/src/gpu_optimizer.rs` — GpuM3State, GpuM3Config, gpu_m3_update(), m3_ns_apply()
- `core/kernels/m3_optimizer.cu` — EMA, 1D/2D apply, Frobenius norm, scale, NS polynomial kernels
- `core/src/cuda_ffi.rs` — FFI declarations for M3 CUDA kernels
- `python/src/lib.rs` — PyO3 bindings (init_m3, step_m3)
- `python/engine/config.py` — M3 config fields
- `python/engine/loop.py` — optimizer dispatch

### Files NOT modified
- No changes to forward/backward pass
- No changes to existing AdamW path (M3 is additive, not replacement)
- No changes to CMS/Conductor/Pulse (M3 uses existing Pulse)

## Constraints

- **CS-27**: M3 frequency levels MUST match CMS architecture levels
- **CS-28**: M3 replaces AdamW for k>=2 (AdamW remains valid for k=1)
- **CS-44**: NS kernel uses cuBLAS (hardware-optimized), not custom matmul
- **fp32 only**: All M3 state is f32 (consistent with inner-loop precision spec)
