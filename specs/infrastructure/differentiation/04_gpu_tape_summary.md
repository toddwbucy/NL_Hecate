# GPU Tape Summary — Fast-Path Level Diagnostics

```text
CONTRACT
  Purpose:    Provide a GPU-resident fast path for per-level output gradient
              norms, eliminating the GPU→CPU round-trip that makes the current
              tape_forward_summary() ~10x slower than a normal training step.
              The CPU tape path (spec 03) is preserved for deep observability
              (DGD delta inspection, M state readout, named saved buffers).
              This spec adds a lightweight alternative that captures the same
              output_grad_norm signal directly during the existing GPU backward.

  Expects:    - core/src/gpu_backward.rs — gpu_cms_backward() already computes
                  d_y_combined per level and passes it to gpu_memory_backward()
              - core/src/gpu_optimizer.rs — grad_norm_sq_cuda kernel and
                  gpu_per_level_grad_norms() pattern for GPU-resident L2 norms
              - python/src/lib.rs — PyGpuModel with tape_forward_summary()
                  (spec 03, CPU path) and step_adamw_gpu() (GPU training step)
              - engine/evaluation.py — print_tape_summary() already handles
                  the dict schema
              - engine/loop.py — tape_every gating for tape diagnostic calls

  Guarantees: - gpu_tape_forward_summary() returns the same dict schema as
                  tape_forward_summary() (spec 03):
                  {
                    "loss": float,
                    "total_blocks": int,    # number of active levels this step
                    "levels": [
                      {
                        "level": int,
                        "opaque_key": str,          # memory_rule from config
                        "block_count": int,          # 1 if active, 0 if inactive
                        "output_grad_norm": float,   # L2 norm of d_y_combined
                        "dgd_delta_norm": float,     # always 0.0 (GPU path)
                      }, ...
                    ]
                  }
              - Same throughput as a normal training step: no GPU→CPU param copy,
                  no traced forward, no CPU tape allocation
              - Does NOT update weights or optimizer state (diagnostic only)
              - CPU tape path (tape_forward_summary) remains available and
                  unchanged for DGD delta, M state readout, saved buffer queries
              - Python callers (loop.py, evaluation.py) work unchanged — same
                  dict keys, same print_tape_summary() formatting

  Cost:       - k calls to grad_norm_sq_cuda (one per active level) during the
                  diagnostic forward+backward — negligible vs full backward cost
              - One extra GPU forward+backward per tape_every step (same as
                  CPU path, but no PCIe transfer)
              - level_output_gnorms: Vec<f32> of length k on GpuMAGGrads — k*4 bytes

  Trade-off:  - dgd_delta_norm is always 0.0 on the GPU path. The DGD delta
                  is computed inside the Rust traced forward's inner loop and is
                  only accessible via the CPU tape's saved buffer mechanism.
                  This is acceptable: output_grad_norm is the primary dormancy
                  signal (spec 02, Section 5.1). DGD delta inspection is a
                  secondary use case that can use the CPU path at lower frequency.
              - total_blocks is synthetic (= number of active levels, not true
                  tape block count). Adequate for the diagnostic use case.
              - opaque_key is read from config.memory_rule rather than from tape
                  OpaqueKey registry. Same string for all levels (homogeneous CMS).

  Position:   specs/infrastructure/differentiation/04_gpu_tape_summary.md
              Extends:    03_tape_summary_pyo3.md (adds GPU fast path)
              Depends on: 02_tape_observation.md (defines the dict schema)
              Used by:    engine/loop.py (tape_every gating)

  Source:     HOPE (2512.24695) Eq. 97 — CMS chain, per-level gradient flow
              CS-32 (observe-then-advance) — summary read after backward
              CS-40 (opt-in tape) — GPU path does not activate CPU tape
              CS-18 (orchestration in Python tier) — path selection in loop.py
```

---

## 1. Problem: CPU Round-Trip Bottleneck

The current `tape_forward_summary()` (spec 03) does this:

```text
GPU params ──copy──▶ CPU params ──traced_forward──▶ CPU tape ──backward──▶ query
   (PCIe)              (slow)                        (slow)               (fast)
```

At d=512, k=4, seq_len=512, this drops throughput from ~650 tok/s to ~49 tok/s
at eval steps. The `to_host()` copies alone are ~1.7 GB of parameters, and the
CPU traced forward is ~10x slower than the GPU forward.

The primary signal extracted — `output_grad_norm` per level — is the L2 norm of
the gradient flowing into each level's backward pass. This gradient
(`d_y_combined`) already exists on GPU during `gpu_cms_backward()`.

---

## 2. Solution: Capture d_y_combined Norms During GPU Backward

### 2.1 Add `level_output_gnorms` to `GpuMAGGrads`

In `core/src/gpu_backward.rs`, add a host-side Vec to `GpuMAGGrads`:

```rust
pub struct GpuMAGGrads {
    // ... existing fields ...
    pub levels: Vec<GpuLevelGrads>,
    /// Per-level L2 norm of d_y_combined (the output gradient entering each
    /// level's backward). Length k. Populated during gpu_cms_backward().
    /// 0.0 for inactive levels (CMS frequency gate).
    pub level_output_gnorms: Vec<f32>,
}
```

### 2.2 Compute norms in `gpu_cms_backward()`

After `d_y_combined` is computed (post-sigmoid-backward, post-1/sqrt(k) scaling)
and before the per-level loop, compute its L2 norm once. Then for each active
level, record the norm. Since all active levels receive the same `d_y_combined`
(MAG composition: memory gates attention, all levels see the same gated output
gradient), the norm is the same for all active levels in the current
architecture.

```rust
// In gpu_cms_backward(), after d_y_combined is finalized:
let mut level_output_gnorms = vec![0.0f32; cfg.k];

// Compute L2 norm of d_y_combined on GPU using existing kernel
let d_y_norm = {
    let mut num_blocks: i32 = 0;
    let max_blocks = (bsd + 255) / 256;
    let mut scratch = GpuBuf::zeros(max_blocks);
    let err = unsafe {
        crate::cuda_ffi::grad_norm_sq_cuda(
            d_y_combined.as_ptr(), scratch.ptr(), bsd as i32, &mut num_blocks,
        )
    };
    assert_eq!(err, 0, "grad_norm_sq_cuda for d_y_combined failed");
    crate::dispatch::cuda_sync();
    let nb = num_blocks as usize;
    let mut host = vec![0.0f32; nb];
    scratch.slice(0, nb).copy_to_host(&mut host);
    let sq_sum: f64 = host.iter().map(|x| *x as f64).sum();
    sq_sum.sqrt() as f32
};

for level in 0..cfg.k {
    if cache.pulse.active_levels[level] {
        level_output_gnorms[level] = d_y_norm;
    }
}

// ... rest of backward ...

// At grads construction:
grads.level_output_gnorms = level_output_gnorms;
```

### 2.3 GPU-side `gpu_tape_forward_summary()` in `python/src/lib.rs`

New method on `PyGpuModel` that runs a GPU forward+backward (no optimizer step)
and reads `level_output_gnorms` from the grads struct:

```rust
/// GPU-resident tape summary — fast path.
///
/// Runs one GPU forward+backward (no optimizer step) and captures per-level
/// output gradient norms from d_y_combined. Same dict schema as
/// tape_forward_summary() but ~10x faster (no CPU round-trip).
///
/// dgd_delta_norm is always 0.0 on this path — use tape_forward_summary()
/// for DGD delta inspection.
fn gpu_tape_forward_summary(
    &mut self,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    pulse: &Pulse,
    py: Python<'_>,
) -> PyResult<PyObject> {
    // Validation (same as tape_forward_summary)
    let s = self.cfg.swa.seq_len;
    let v = self.cfg.swa.vocab_size;
    // ... input validation ...

    // GPU forward (non-destructive: context is saved/restored)
    let saved_ctx = self.context.clone();
    let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
        &self.params, &self.cfg, &input_ids, &target_ids,
        &pulse.inner, &mut self.context,
    );

    // GPU backward (collect_output_gnorms=true for diagnostic)
    let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
        &self.params, &self.cfg, &cache, true,
    );

    // Restore context (diagnostic must not modify state)
    self.context = saved_ctx;

    // Build same dict schema as tape_forward_summary
    let rule_name = format!("{:?}", self.cfg.memory_rule);
    let dict = PyDict::new(py);
    dict.set_item("loss", loss)?;

    let active_count: usize = (0..self.cfg.k)
        .filter(|&l| pulse.inner.active_levels[l])
        .count();
    dict.set_item("total_blocks", active_count)?;

    let levels_list = pyo3::types::PyList::empty(py);
    for level in 0..self.cfg.k {
        let ldict = PyDict::new(py);
        ldict.set_item("level", level)?;
        ldict.set_item("opaque_key", &rule_name)?;
        ldict.set_item("block_count",
            if pulse.inner.active_levels[level] { 1usize } else { 0usize })?;
        ldict.set_item("output_grad_norm",
            grads.level_output_gnorms[level])?;
        ldict.set_item("dgd_delta_norm", 0.0f32)?;
        levels_list.append(ldict)?;
    }
    dict.set_item("levels", levels_list)?;

    Ok(dict.into())
}
```

### 2.4 Python caller — transparent selection in `loop.py`

The caller in `loop.py` switches to `gpu_tape_forward_summary()` when available,
falling back to `tape_forward_summary()`. No config change needed — the method
name is the gate:

```python
# In loop.py, at tape_every interval:
if hasattr(gpu_model, "gpu_tape_forward_summary"):
    tape_sum = gpu_model.gpu_tape_forward_summary(
        input_ids, target_ids, pulse
    )
else:
    tape_sum = gpu_model.tape_forward_summary(
        input_ids, target_ids, pulse
    )
```

`print_tape_summary()` and JSONL logging work unchanged — same dict schema.

---

## 3. What This Does NOT Change

| Component | Status |
|---|---|
| CPU `tape_forward_summary()` | Preserved — still available for DGD delta, M state readout |
| `print_tape_summary()` in evaluation.py | Unchanged — same dict schema |
| JSONL log format | Unchanged — same keys |
| Wengert tape infrastructure (specs 01-03) | Unchanged |
| GPU training step (`step_adamw_gpu`) | Unchanged — norms are only captured in the diagnostic call |
| `gpu_per_level_grad_norms()` | Unchanged — continues to measure weight gradient norms |

---

## 4. Context Save/Restore

The diagnostic forward+backward modifies `GpuContextState` (memory M states
advance during forward). This must be invisible to the training loop.

Strategy: clone context before the diagnostic call, restore after.
`GpuContextState` contains `Vec<GpuBuf<f32>>` for `context_m` — these are
device pointers. Clone must be a deep copy (cudaMemcpy D2D), not pointer copy.

If `GpuContextState` does not currently implement deep clone, add it:

```rust
impl GpuContextState {
    pub fn deep_clone(&self) -> Self {
        GpuContextState {
            context_m: self.context_m.iter().map(|buf| {
                let mut copy = GpuBuf::zeros(buf.len());
                gpu_buf_memcpy_d2d(copy.ptr(), buf.as_ptr(), buf.len() * 4);
                copy
            }).collect(),
        }
    }
}
```

---

## 5. Relationship Between output_grad_norm Sources

| Source | What it measures | When computed |
|---|---|---|
| CPU tape `opaque_output_grad_norm()` | Gradient on the first output BufId of the last opaque block at each level | Post-backward, on CPU tape arena |
| GPU `level_output_gnorms` | L2 norm of `d_y_combined` entering each level's backward | During GPU backward |
| `gpu_per_level_grad_norms()` | L2 norm of accumulated weight gradients per level | Post-backward, before clipping |

The first two measure the same quantity (gradient flowing *into* the level),
just computed in different places. The third measures gradient flowing *through*
the level into the weights — a downstream consequence. Both are useful:
`output_grad_norm` detects dormancy at the input, `grad_norms` detects
effective parameter updates.

---

## 6. Files to Create/Modify

| File | Change |
|---|---|
| `core/src/gpu_backward.rs` | Add `level_output_gnorms: Vec<f32>` to `GpuMAGGrads`; compute `d_y_combined` norm via `grad_norm_sq_cuda`; populate per active level |
| `python/src/lib.rs` | Add `gpu_tape_forward_summary()` method on `PyGpuModel` |
| `python/engine/loop.py` | Prefer `gpu_tape_forward_summary` when available |
| `core/src/gpu_forward.rs` | Add `deep_clone()` on `GpuContextState` if not present |

No new files required. All changes are additive.

---

## 7. Testing

### Rust tests (core)

```rust
#[test]
fn test_level_output_gnorms_populated() {
    // After gpu_cms_backward(), grads.level_output_gnorms.len() == cfg.k
    // Active levels have gnorm > 0, inactive have gnorm == 0
}

#[test]
fn test_level_output_gnorms_match_d_y_combined() {
    // For k=1 (all active), gnorm must match manual L2 norm of d_y_combined
    // extracted via copy_to_host after the backward
}
```

### Python tests

```python
def test_gpu_tape_summary_schema():
    """gpu_tape_forward_summary returns same dict keys as tape_forward_summary."""
    gpu_sum = gpu_model.gpu_tape_forward_summary(input_ids, target_ids, pulse)
    cpu_sum = gpu_model.tape_forward_summary(input_ids, target_ids, pulse)
    assert set(gpu_sum.keys()) == set(cpu_sum.keys())
    assert len(gpu_sum["levels"]) == len(cpu_sum["levels"])
    for g, c in zip(gpu_sum["levels"], cpu_sum["levels"]):
        assert set(g.keys()) == set(c.keys())

def test_gpu_tape_summary_does_not_modify_context():
    """Context M states must be identical before and after the call."""
    ctx_before = gpu_model.to_host_context()
    gpu_model.gpu_tape_forward_summary(input_ids, target_ids, pulse)
    ctx_after = gpu_model.to_host_context()
    # Compare M state tensors
```

---

## 8. HADES Registration

```json
{
  "_key": "gpu-tape-summary",
  "title": "GPU Tape Summary — Fast-Path Level Diagnostics",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/differentiation/04_gpu_tape_summary.md",
  "purpose": "GPU-resident fast path for per-level output gradient norms, eliminating CPU round-trip from tape_forward_summary",
  "paper_source": ["2512.24695"],
  "traced_to_equations": [
    "hope_equations/eq-097-hope-cms-chain"
  ],
  "traced_to_axioms": [],
  "depends_on_specs": [
    "hecate_specs/tape-summary-pyo3",
    "hecate_specs/tape-observation"
  ],
  "status": "v0.4.0"
}
```
