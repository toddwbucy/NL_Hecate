# Stacked Tape Diagnostics — Per-Block × Per-Level Gradient Observability

```text
CONTRACT
  Purpose:    Extend the Wengert tape and GPU tape summary infrastructure to
              N-block stacked models. Provides per-(block, level) gradient norms
              and diagnostic signals, with a switchable CPU/GPU mode:
              - CPU mode: full Rust reference forward+backward with Wengert tape
                (for validation, cross-checking GPU path, DGD delta inspection)
              - GPU mode: lightweight norm extraction from existing CUDA backward
                (for production monitoring, NaN tracing, dormancy detection)
              - Off mode: no diagnostic overhead (default for maximum throughput)

  Expects:    - Stacked forward/backward: gpu_stacked_forward.rs, gpu_stacked_backward.rs
              - Stacked optimizer: gpu_stacked_optimizer.rs (GpuStackedAdamWState)
              - Single-block tape: tape.rs, traced_forward.rs, tape_summary.rs,
                opaque_adapters.rs (CPU path)
              - Single-block GPU tape: gpu_backward.rs level_output_gnorms (spec 04)
              - Stacked params: GpuStackedParams with Vec<GpuBlockParams>
              - Stacked cache: GpuStackedCache with Vec<GpuStackedBlockCache>
              - Stacked grads: GpuStackedGrads with Vec<GpuStackedBlockGrads>

  Guarantees: - Per-(block, level) output_grad_norm in every diagnostic call
              - CPU and GPU modes produce compatible dict schemas (superset of
                spec 04 schema, with added block_index field)
              - CPU mode produces bitwise-identical loss to GPU training step
              - GPU mode has zero parameter PCIe transfer (norms computed on-device)
              - Diagnostic calls do NOT modify model state (context save/restore)
              - tape_device config switch selects mode at build start, not per-step
              - n_blocks=1 with tape_device=cpu falls through to existing
                tape_forward_summary() (backward compat)

  Cost:       CPU mode: ~10-50x slower than GPU training step (PCIe + Rust forward).
                At d=512, n_blocks=4, k=4: ~4x the single-block CPU tape cost.
              GPU mode: one extra GPU forward+backward per tape_every step.
                At d=512, n_blocks=4, k=4: ~7s overhead per diagnostic step.
                Negligible vs training when tape_every >= 256.
              Off mode: zero overhead.

  Trade-off:  CPU mode gives full DGD delta, M state readout, named saved buffers,
              and serves as the gradient oracle for cross-checking the GPU path.
              GPU mode sacrifices DGD delta (always 0.0) and saved buffer access
              for 10-50x faster diagnostics. The primary NaN tracing signal —
              per-(block, level) output_grad_norm — is available in both modes.

  Position:   specs/infrastructure/15_stacked_tape_diagnostics.md
              Extends:    14_multi_block_stacking.md (adds tape to stacked arch)
              Extends:    differentiation/04_gpu_tape_summary.md (multi-block)
              Depends on: differentiation/01_wengert_tape.md (tape engine)
              Depends on: differentiation/03_tape_summary_pyo3.md (schema)

  Source:     HOPE (2512.24695) Section 6 — HOPE architecture with CMS
              HOPE (2512.24695) Eq. 97 — CMS chain, per-level gradient flow
              CS-32 (observe-then-advance) — summary read after backward
              CS-40 (opt-in tape) — diagnostic is opt-in via tape_device config
              CS-18 (orchestration in Python tier) — mode selection in loop.py
```

---

## 1. Problem: NaN Tracing in Stacked Models

Stacked 4-block × 4-level models exhibit gradient pathology:
- Full-trajectory: NaN at step 442 (fixed by adding grad clipping)
- TNT chunkwise: NaN at step 1449 despite grad clipping (gnorm 18673 → inf → NaN)

The existing tape infrastructure is single-block only. `traced_cms_forward()` records
one block's ops (embed → SWA → k levels → unembed). For stacked models we need to
see *which block* and *which level* is the source of gradient explosion. Without
per-(block, level) visibility, debugging is blind.

The TNT NaN demonstrates that grad clipping alone is insufficient — we need to
understand where the pathological gradients originate to fix the root cause, not
just clamp the symptom.

---

## 2. Architecture: Switchable CPU/GPU Tape Device

```text
                    ┌── tape_device: "cpu" ──────────────────────────────┐
                    │                                                     │
                    │  GPU params ──copy──▶ CPU params (N blocks)         │
                    │       │                                             │
                    │       ▼                                             │
                    │  traced_stacked_forward()                           │
                    │    ┌── Block 0 ──────────────────────┐              │
                    │    │  embed → LN_attn → SWA          │              │
config.tape_device ─┤    │  LN_mem → [level 0..k-1]       │──┐ tape ops  │
                    │    └─────────────────────────────────┘  │           │
                    │    ┌── Block 1 ──────────────────────┐  │           │
                    │    │  LN_attn → SWA                  │  │           │
                    │    │  LN_mem → [level 0..k-1]        │──┤           │
                    │    └─────────────────────────────────┘  │           │
                    │    ...                                   │           │
                    │    unembed → cross_entropy              │           │
                    │       │                                  │           │
                    │       ▼                                  │           │
                    │  tape.backward() ◄──────────────────────┘           │
                    │       │                                              │
                    │       ▼                                              │
                    │  extract per-(block, level) norms + DGD delta        │
                    │                                                      │
                    └──────────────────────────────────────────────────────┘

                    ┌── tape_device: "gpu" ──────────────────────────────┐
                    │                                                     │
                    │  gpu_stacked_forward() ──▶ gpu_stacked_backward()   │
                    │       │                         │                    │
                    │       │    per-block d_y_combined norms captured     │
                    │       │         │                                    │
                    │       ▼         ▼                                    │
                    │  {loss, per-(block,level) output_grad_norm}          │
                    │                                                      │
                    └──────────────────────────────────────────────────────┘

                    ┌── tape_device: "off" (default) ────────────────────┐
                    │  No diagnostic overhead. Normal training step only. │
                    └──────────────────────────────────────────────────────┘
```

### 2.1 Config Field

In `python/engine/config.py`, add:

```python
tape_device: str = "off"   # "cpu", "gpu", or "off"
```

Validation: `tape_device in ("cpu", "gpu", "off")`. If `tape_device != "off"`,
`tape_every` must be > 0 (otherwise no diagnostic steps are ever triggered).

The switch is set at build start and applies for the entire run. It does NOT
change per-step — the mode selects which code path handles `tape_every` intervals.

---

## 3. GPU Mode: Per-Block Gradient Norm Extraction

### 3.1 Extend `GpuStackedBlockGrads`

Add per-level output gradient norms to each block's gradient struct:

```rust
// In gpu_stacked_backward.rs
pub struct GpuStackedBlockGrads {
    // ... existing fields ...
    /// Per-level L2 norm of the gradient entering each level's backward pass.
    /// Length k. 0.0 for inactive levels.
    pub level_output_gnorms: Vec<f32>,
}
```

### 3.2 Capture Norms in `gpu_stacked_backward()`

Inside the per-block backward loop (line 162, `for b in (0..n_blocks).rev()`),
after `d_y_combined` is computed (line 183-198) and before the per-level memory
backward loop (line 202):

```rust
// Compute L2 norm of d_y_combined for this block
let mut block_level_gnorms = vec![0.0f32; cfg.k];
let d_y_norm = {
    let mut num_blocks_out: i32 = 0;
    let max_blocks = (bsd + 255) / 256;
    let mut scratch = GpuBuf::zeros(max_blocks);
    unsafe {
        crate::cuda_ffi::grad_norm_sq_cuda(
            d_y_combined.as_ptr(), scratch.ptr(), bsd as i32, &mut num_blocks_out,
        );
    }
    crate::dispatch::cuda_sync();
    let nb = num_blocks_out as usize;
    let mut host = vec![0.0f32; nb];
    scratch.slice(0, nb).copy_to_host(&mut host);
    let sq_sum: f64 = host.iter().map(|x| *x as f64).sum();
    sq_sum.sqrt() as f32
};

for level in 0..cfg.k {
    if cache.pulse.active_levels[level] {
        block_level_gnorms[level] = d_y_norm;
    }
}
```

Store in `GpuStackedBlockGrads` alongside existing gradient buffers.

### 3.3 Stacked Grad Norm Roll-Up

Add a method on `GpuStackedGrads` to extract the full (block, level) matrix:

```rust
impl GpuStackedGrads {
    /// Returns per-(block, level) output gradient norms as a flat Vec.
    /// Layout: [block_0_level_0, block_0_level_1, ..., block_N-1_level_k-1]
    pub fn all_level_output_gnorms(&self) -> Vec<f32> {
        self.blocks.iter()
            .flat_map(|bg| bg.level_output_gnorms.iter().copied())
            .collect()
    }
}
```

### 3.4 PyO3 Method: `gpu_stacked_tape_summary()`

New method on `PyGpuStackedModel`:

```rust
/// GPU-resident stacked tape summary.
///
/// Runs one GPU forward+backward (no optimizer step) and captures
/// per-(block, level) output gradient norms from d_y_combined.
/// Context is saved and restored — diagnostic does not modify state.
fn gpu_stacked_tape_summary(
    &mut self,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    pulse: &Pulse,
    py: Python<'_>,
) -> PyResult<PyObject> {
    // Save context (deep clone — cudaMemcpy D2D for each block's context_m)
    let saved_ctx = self.context.deep_clone();

    // Forward
    let (loss, cache) = gpu_stacked_forward(
        &self.params, &self.cfg, &input_ids, &target_ids,
        &pulse.inner, &mut self.context,
    );

    // Backward (non-destructive — we only read grad norms)
    let grads = gpu_stacked_backward(&self.params, &self.cfg, &cache);

    // Restore context
    self.context = saved_ctx;

    // Build dict
    build_stacked_tape_dict(py, loss, &grads, &self.cfg, &pulse.inner)
}
```

### 3.5 Dict Schema (extends spec 04)

```python
{
    "loss": float,
    "n_blocks": int,
    "total_blocks": int,           # sum of active levels across all blocks
    "blocks": [                    # length n_blocks
        {
            "block_index": int,
            "levels": [            # length k
                {
                    "level": int,
                    "opaque_key": str,         # memory_rule from config
                    "block_count": int,        # 1 if active, 0 if inactive
                    "output_grad_norm": float, # L2 norm of d_y_combined
                    "dgd_delta_norm": float,   # 0.0 on GPU path
                }, ...
            ]
        }, ...
    ],
    # Flat summary for backward compat with single-block consumers
    "levels": [                    # aggregated across all blocks (max gnorm per level)
        {
            "level": int,
            "opaque_key": str,
            "block_count": int,
            "output_grad_norm": float,  # max across blocks
            "dgd_delta_norm": float,
        }, ...
    ]
}
```

The `blocks` array is the new per-(block, level) signal. The top-level `levels`
array is retained for backward compatibility with `print_tape_summary()` and
JSONL logging — it aggregates by taking the max output_grad_norm across blocks
for each level.

---

## 4. CPU Mode: Traced Stacked Forward

### 4.1 `traced_stacked_forward()` in `core/src/traced_forward.rs`

Extends `traced_cms_forward()` to handle N blocks. The tape records ops in the
same linear sequence as the stacked forward:

```text
Tape recording order:
  embed
  ┌─ Block 0 ─────────────────────┐
  │  ln_attn → q,k,v projections  │
  │  SWA (opaque, key=SWA)        │
  │  residual skip 1               │
  │  ln_mem                        │
  │  level 0 (opaque, key=Titans)  │  ← tagged (block=0, level=0)
  │  level 1 (opaque, key=Titans)  │  ← tagged (block=0, level=1)
  │  ...                           │
  │  residual skip 2               │
  └────────────────────────────────┘
  ┌─ Block 1 ─────────────────────┐
  │  (same pattern)                │
  │  level 0 (opaque)              │  ← tagged (block=1, level=0)
  │  ...                           │
  └────────────────────────────────┘
  ...
  ln_final → unembed → cross_entropy
```

Each opaque block records both `level: Option<usize>` (existing) and a new
`block_index: Option<usize>` field on TapeBuf:

```rust
// In tape.rs, extend TapeBuf:
pub struct TapeBuf {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub is_param: bool,
    pub role: Option<&'static str>,
    pub level: Option<usize>,
    /// Block index within stacked model (0..n_blocks-1).
    /// None for shared params (embed, unembed, ln_final).
    pub block_index: Option<usize>,
}
```

### 4.2 `TracedStackedParamIds`

Maps all registered parameters back to their tape BufIds:

```rust
pub struct TracedStackedParamIds {
    pub w_embed: BufId,
    pub w_unembed: BufId,
    pub ln_final_gamma: BufId,
    pub ln_final_beta: BufId,
    /// Per-block parameter IDs, length n_blocks.
    pub blocks: Vec<TracedBlockParamIds>,
}

pub struct TracedBlockParamIds {
    pub w_q: BufId,
    pub w_k: BufId,
    pub w_v: BufId,
    pub w_o: BufId,
    pub ln_attn_gamma: BufId,
    pub ln_attn_beta: BufId,
    pub ln_mem_gamma: BufId,
    pub ln_mem_beta: BufId,
    /// Per-level level_params BufIds.
    pub level_params: Vec<BufId>,
    pub frozen_w_q_mem: Vec<Option<BufId>>,
}
```

### 4.3 `extract_stacked_tape_summary()`

New function in `tape_summary.rs`:

```rust
pub fn extract_stacked_tape_summary(
    params:     &MAGStackedParams,   // CPU-side stacked params
    cfg:        &MAGConfig,
    input_ids:  &[usize],
    target_ids: &[usize],
    pulse:      &Pulse,
    contexts:   &mut Vec<ContextState>,  // per-block context
) -> StackedTapeSummary {
    let registry = register_opaque_vjps();

    with_tape(registry, |tape| {
        let (loss, _cache, loss_id, _param_ids) =
            traced_stacked_forward(tape, params, cfg, input_ids, target_ids, pulse, contexts);

        tape.backward(loss_id);

        // Extract per-(block, level) norms
        let all_blocks = tape.enumerate_opaque_blocks();
        // ... group by (block_index, level) ...
    })
}
```

### 4.4 `StackedTapeSummary`

```rust
pub struct StackedTapeSummary {
    pub loss: f32,
    pub n_blocks: usize,
    pub total_blocks: usize,
    pub blocks: Vec<BlockTapeSummary>,
}

pub struct BlockTapeSummary {
    pub block_index: usize,
    pub levels: Vec<LevelSummary>,  // reuses existing LevelSummary
}
```

---

## 5. Stacked Context Deep Clone

Both CPU and GPU diagnostic paths must save/restore context to avoid side effects.

### GPU path

`GpuStackedContext` (or equivalent) contains per-block `Vec<GpuBuf<f32>>` for
`context_m`. Deep clone via `cudaMemcpy` device-to-device:

```rust
impl GpuStackedContext {
    pub fn deep_clone(&self) -> Self {
        GpuStackedContext {
            blocks: self.blocks.iter().map(|bc| {
                GpuBlockContext {
                    context_m: bc.context_m.iter().map(|buf| {
                        let mut copy = GpuBuf::zeros(buf.len());
                        gpu_buf_memcpy_d2d(copy.ptr(), buf.as_ptr(), buf.len() * 4);
                        copy
                    }).collect(),
                }
            }).collect(),
        }
    }
}
```

### CPU path

`Vec<ContextState>` — each `ContextState` contains `Vec<Vec<f32>>` for `context_m`.
Standard `clone()` suffices (heap copy).

---

## 6. Python Integration: loop.py

### 6.1 Tape Device Dispatch

At `tape_every` intervals in the training loop:

```python
if bcfg.tape_device == "gpu" and is_stacked:
    tape_sum = gpu_model.gpu_stacked_tape_summary(input_ids, target_ids, pulse)
    print_stacked_tape_summary(tape_sum, step)
elif bcfg.tape_device == "gpu" and not is_stacked:
    tape_sum = gpu_model.gpu_tape_forward_summary(input_ids, target_ids, pulse)
    print_tape_summary(tape_sum, step)
elif bcfg.tape_device == "cpu" and is_stacked:
    tape_sum = gpu_model.cpu_stacked_tape_summary(input_ids, target_ids, pulse)
    print_stacked_tape_summary(tape_sum, step)
elif bcfg.tape_device == "cpu" and not is_stacked:
    tape_sum = gpu_model.tape_forward_summary(input_ids, target_ids, pulse)
    print_tape_summary(tape_sum, step)
# tape_device == "off": skip entirely
```

### 6.2 `print_stacked_tape_summary()` in `evaluation.py`

```python
def print_stacked_tape_summary(summary: dict, step: int):
    """Print per-(block, level) gradient diagnostics."""
    print(f"  [tape] step {step}  loss={summary['loss']:.4f}  "
          f"n_blocks={summary['n_blocks']}  total_opaque={summary['total_blocks']}")
    for block in summary["blocks"]:
        bi = block["block_index"]
        for lev in block["levels"]:
            li = lev["level"]
            gnorm = lev["output_grad_norm"]
            status = "ACTIVE" if lev["block_count"] > 0 else "frozen"
            flag = " ◀ DEAD" if gnorm < 1e-6 and lev["block_count"] > 0 else ""
            flag = " ◀ EXPLODING" if gnorm > 100.0 else flag
            print(f"    block[{bi}] level[{li}] {status:>6}  "
                  f"gnorm={gnorm:.6f}{flag}")
```

This gives immediate visibility into which (block, level) pair is the gradient
pathology source — the exact signal needed to diagnose the TNT NaN.

### 6.3 JSONL Logging

At tape_every steps, append a `tape_summary` event:

```python
{
    "event": "tape_summary",
    "step": 1424,
    "loss": 6.5729,
    "n_blocks": 4,
    "blocks": [
        {"block_index": 0, "levels": [{"level": 0, "output_grad_norm": 0.0312}, ...]},
        {"block_index": 1, "levels": [{"level": 0, "output_grad_norm": 0.4821}, ...]},
        {"block_index": 2, "levels": [{"level": 0, "output_grad_norm": 15.234}, ...]},
        {"block_index": 3, "levels": [{"level": 0, "output_grad_norm": 18673.9}, ...]}
    ]
}
```

---

## 7. NaN Tracing Use Case

With stacked tape diagnostics, the TNT NaN at step 1449 could be traced:

1. Set `tape_device: "gpu"`, `tape_every: 8` (frequent sampling near expected crash)
2. At each tape step, log per-(block, level) output_grad_norm
3. The exploding block/level pair becomes visible in JSONL:
   - If block 3 level 0 shows gnorm 18673 while blocks 0-2 are normal → the last
     block's fast-frequency level is the pathology source
   - If all blocks show escalating gnorms → the residual stream is amplifying
     through depth (compounding gradient problem)
   - If only TNT-chunked levels explode → the chunkwise approximation introduces
     gradient error that compounds through blocks

This determines the fix:
- **Single block pathology**: per-block M-norm clamp or per-block grad clipping
- **Residual amplification**: residual scaling (1/sqrt(n_blocks)) or per-block
  gradient checkpointing
- **TNT approximation error**: larger chunk size, per-chunk momentum reset, or
  gradient scaling for TNT levels

---

## 8. CPU/GPU Cross-Validation

At low frequency (e.g., every 10× tape_every), run both CPU and GPU paths and
compare output_grad_norms:

```python
if step % (bcfg.tape_every * 10) == 0 and bcfg.tape_device == "gpu":
    cpu_sum = gpu_model.cpu_stacked_tape_summary(input_ids, target_ids, pulse)
    gpu_sum = gpu_model.gpu_stacked_tape_summary(input_ids, target_ids, pulse)
    for b in range(n_blocks):
        for l in range(k):
            cpu_gn = cpu_sum["blocks"][b]["levels"][l]["output_grad_norm"]
            gpu_gn = gpu_sum["blocks"][b]["levels"][l]["output_grad_norm"]
            rel_err = abs(cpu_gn - gpu_gn) / max(cpu_gn, 1e-8)
            if rel_err > 0.05:
                print(f"  WARNING: tape cross-check block[{b}] level[{l}] "
                      f"cpu={cpu_gn:.6f} gpu={gpu_gn:.6f} rel_err={rel_err:.4f}")
```

This serves as a continuous validation gate: if the GPU backward and CPU tape
disagree on gradient norms, either the CUDA kernel has a bug or the CPU reference
path has diverged. Both are critical to catch.

---

## 9. Files to Create/Modify

| File | Change |
|---|---|
| `core/src/tape.rs` | Add `block_index: Option<usize>` to `TapeBuf` |
| `core/src/traced_forward.rs` | Add `traced_stacked_forward()` — N-block tape recording |
| `core/src/tape_summary.rs` | Add `StackedTapeSummary`, `BlockTapeSummary`, `extract_stacked_tape_summary()` |
| `core/src/gpu_stacked_backward.rs` | Add `level_output_gnorms: Vec<f32>` to `GpuStackedBlockGrads`; compute `d_y_combined` norms per block |
| `python/src/lib.rs` | Add `gpu_stacked_tape_summary()` and `cpu_stacked_tape_summary()` on `PyGpuStackedModel` |
| `python/engine/config.py` | Add `tape_device: str = "off"` field + validation |
| `python/engine/loop.py` | Tape device dispatch at `tape_every` intervals |
| `python/engine/evaluation.py` | Add `print_stacked_tape_summary()` |

New files: none (all additions to existing files).

---

## 10. Build Order

1. `tape.rs` — add `block_index` field to `TapeBuf` (backward compat: all existing code passes `None`)
2. `gpu_stacked_backward.rs` — add `level_output_gnorms` to `GpuStackedBlockGrads`, compute norms (GPU mode)
3. `python/src/lib.rs` — add `gpu_stacked_tape_summary()` method (GPU mode PyO3)
4. `cargo build --release --features cuda` — verify GPU mode compiles
5. `config.py` — add `tape_device` field
6. `loop.py` + `evaluation.py` — wire tape device dispatch + stacked summary printing
7. `maturin develop --release --features cuda` — build Python bindings
8. **Smoke test**: run stacked shakedown with `tape_device: "gpu"`, `tape_every: 64`
9. `traced_forward.rs` — add `traced_stacked_forward()` (CPU mode, Phase 2)
10. `tape_summary.rs` — add `extract_stacked_tape_summary()` (CPU mode, Phase 2)
11. `python/src/lib.rs` — add `cpu_stacked_tape_summary()` (CPU mode PyO3, Phase 2)

GPU mode (steps 1-8) is the priority — it gives NaN tracing capability immediately.
CPU mode (steps 9-11) is Phase 2 — needed for validation cross-checking but not
for the immediate NaN debugging need.

---

## 11. Testing

### Rust tests

```rust
#[test]
fn test_stacked_block_grads_have_level_gnorms() {
    // After gpu_stacked_backward(), each block's level_output_gnorms.len() == cfg.k
    // Active levels have gnorm > 0, inactive have gnorm == 0
}

#[test]
fn test_tape_buf_block_index() {
    // TapeBuf with block_index=Some(2) preserves the value through alloc/query
}

#[test]
fn test_traced_stacked_forward_bitwise() {
    // traced_stacked_forward() produces identical loss to gpu_stacked_forward()
    // for same inputs, params, context
}

#[test]
fn test_stacked_tape_summary_per_block_levels() {
    // extract_stacked_tape_summary() returns n_blocks entries,
    // each with k level summaries
}
```

### Python tests

```python
def test_gpu_stacked_tape_summary_schema():
    """Verify dict schema has blocks array with per-level entries."""
    summary = gpu_model.gpu_stacked_tape_summary(input_ids, target_ids, pulse)
    assert "blocks" in summary
    assert len(summary["blocks"]) == n_blocks
    for block in summary["blocks"]:
        assert "block_index" in block
        assert len(block["levels"]) == k

def test_gpu_stacked_tape_no_context_mutation():
    """Context M states identical before and after diagnostic call."""
    # Compare context checksums pre/post

def test_tape_device_config_validation():
    """tape_device must be 'cpu', 'gpu', or 'off'."""
    cfg = BuildConfig(tape_device="invalid")  # should raise
```

---

## 12. HADES Registration

```json
{
  "_key": "stacked-tape-diagnostics",
  "title": "Stacked Tape Diagnostics — Per-Block × Per-Level Gradient Observability",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/15_stacked_tape_diagnostics.md",
  "purpose": "Switchable CPU/GPU tape for N-block stacked models with per-(block, level) gradient norm extraction",
  "paper_source": ["2512.24695"],
  "traced_to_equations": [
    "hope_equations/eq-097-hope-cms-chain"
  ],
  "traced_to_axioms": [],
  "depends_on_specs": [
    "hecate_specs/multi-block-stacking",
    "hecate_specs/gpu-tape-summary",
    "hecate_specs/wengert-tape"
  ],
  "status": "v0.4.0"
}
```
