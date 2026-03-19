# Spec 33: Optimizer State Snapshot/Restore

## CONTRACT

| Field | Value |
|-------|-------|
| **Purpose** | Preserve AdamW optimizer moments across learning probes so build training resumes without cold-start transient |
| **Expects** | GpuModel with lazy-init `adamw_state: Option<GpuAdamWState>` |
| **Guarantees** | After probe sequence: params, context, AND optimizer moments are identical to pre-probe state |
| **Cost** | One GPU→host download + one host→GPU upload per checkpoint event (~2× param size, overlappable with tape I/O) |
| **Trade-off** | Extra host memory for moment snapshot (~2× model params in f32). Acceptable: moments are same size as params, and snapshot is transient (freed after restore) |
| **Position** | Rust tier (gpu_optimizer.rs) + Python tier (lib.rs PyO3 bindings, evaluation.py, loop.py) |
| **Source** | GitHub issue #206; PR #205 review finding (h) |

## Motivation

Learning probes at checkpoint time (spec 32) require modifying model state:
- `full_restore()` restores params + context but NOT optimizer moments
- `reset_optimizer()` zeroes all AdamW m1/m2 buffers and step counter
- After the final restore, training resumes with cold optimizer moments

Cold-restart effect: AdamW bias correction with step=0 causes ~100 steps of
elevated effective learning rate before moments reconverge. With `probe_max_tokens`
increasing, the corruption window grows.

## Design

### Host-side optimizer state struct

```rust
pub struct HostOptimizerState {
    swa_moments: Vec<f32>,       // concatenated m+v for all SWA params
    level_moments: Vec<Vec<f32>>, // concatenated m+v per level
    level_steps: Vec<u32>,        // per-level step counters
    step: u32,                    // global step counter
}
```

A flat concatenation avoids needing to mirror every field name. The layout is
deterministic: same struct that created the snapshot restores it.

### Rust API (gpu_optimizer.rs)

```rust
impl GpuAdamWState {
    pub fn to_host(&self) -> HostOptimizerState { ... }
    pub fn from_host(host: &HostOptimizerState, params: &GpuMAGParams) -> Self { ... }
}
```

`to_host()` concatenates all GpuBuf m/v pairs into flat Vec<f32> via cudaMemcpy D→H.
`from_host()` slices the flat vectors and uploads via cudaMemcpy H→D, reconstructing
norm_scratch from params (same as `from_params()`).

### PyO3 bindings (lib.rs)

```rust
#[pyclass]
struct OptimizerState { inner: HostOptimizerState }

#[pymethods]
impl GpuModel {
    fn snapshot_optimizer(&self) -> Option<OptimizerState> { ... }
    fn restore_optimizer(&mut self, state: &OptimizerState) -> PyResult<()> { ... }
}
```

`snapshot_optimizer()` returns `None` if optimizer hasn't been initialized yet
(no training steps taken). This is safe — the probe path only runs after
training has started.

### Python integration (evaluation.py)

```python
def full_snapshot(gpu_model):
    return {
        "params": gpu_model.to_host_params(),
        "context": gpu_model.to_host_context(),
        "optimizer": gpu_model.snapshot_optimizer(),
    }

def full_restore(gpu_model, snapshot):
    gpu_model.upload_params(snapshot["params"])
    gpu_model.upload_context(snapshot["context"])
    if snapshot.get("optimizer") is not None:
        gpu_model.restore_optimizer(snapshot["optimizer"])
```

All `reset_optimizer()` calls in the probe path become unnecessary and are removed.

## Scope

### Files to modify
- `core/src/gpu_optimizer.rs` — add `HostOptimizerState`, `to_host()`, `from_host()`
- `python/src/lib.rs` — add `OptimizerState` pyclass, `snapshot_optimizer()`, `restore_optimizer()`
- `python/engine/evaluation.py` — update `full_snapshot`/`full_restore`
- `python/engine/loop.py` — remove `reset_optimizer()` calls from probe paths

### Files NOT modified
- No CUDA kernel changes (uses existing cudaMemcpy via GpuBuf)
- No config changes
- No checkpoint serialization changes (optimizer state is transient, not persisted)

## What gets removed

- All `gpu_model.reset_optimizer()` calls in probe paths (loop.py, evaluation.py)
- The cold-restart transient after each checkpoint event
