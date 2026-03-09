# DGD Delta Norm — GPU-Resident Inner-Loop Error Observability

```text
CONTRACT
  Purpose:    Expose the DGD prediction error norm ‖M@k - v‖₂ from the CUDA
              forward kernels as a per-(block, level) diagnostic signal. This is
              the inner-loop self-modification signal — when it collapses to zero,
              the memory has stopped learning. When it explodes, NaN is imminent.
              Currently only available on the CPU tape path (obs::DGD_DELTA buffer
              role in opaque_adapters.rs). This spec adds the GPU path.

  Expects:    - Titans/DGD forward CUDA kernels: titans_forward.cu, dgd_forward.cu
                (both compute error[i] = prediction[i] - v_t[i] in shared memory)
              - GPU stacked tape infrastructure: spec 15 (gpu_stacked_tape_summary)
              - Per-(block, level) diagnostic dict schema from spec 15
              - grad_norm_sq_cuda kernel (core/kernels/elementwise.cu) for norm reduction

  Guarantees: - Per-(block, level) dgd_delta_norm in tape summary dict (replaces 0.0 placeholder)
              - GPU-only: no PCIe transfer of error vectors, only final scalar norms
              - Norm computed from the LAST token's PRE-UPDATE error:
                ‖M_{s-1} @ k_{s-1} - v_{s-1}‖₂ (the error that drove the final M-update,
                using M's state BEFORE the last update — not the post-update M_s)
              - CPU tape path (obs::DGD_DELTA) unchanged — still captures full error buffer
              - Forward kernel output buffer is OPTIONAL: controlled by non-null pointer
                (zero overhead when diagnostic is off)

  Cost:       Per level per block: one d-element warp reduction (~20 cycles at d=512).
              At n_blocks=4, k=4, 12 active levels: ~240 cycles total.
              Negligible vs the d² M-update that dominates each forward token step.

  Trade-off:  Reports only the LAST token's error norm, not per-token trace.
              Per-token would require seq_len output floats per level — useful but
              not needed for the primary diagnostic (collapse/explosion detection).
              The last-token norm is the signal that matters: it reflects whether M
              has learned to predict after seeing the full context.

  Position:   specs/infrastructure/16_dgd_delta_norm_gpu.md
              Extends:    15_stacked_tape_diagnostics.md (fills dgd_delta_norm=0.0 gap)
              Extends:    cuda/02_dgd_kernels.md (adds output to forward kernels)
              Depends on: algorithms/optimization_machinery/03_dgd.md (defines the error)

  Source:     HOPE (2512.24695) §4.5, Eq 88 — DGD update rule
              HOPE (2512.24695) Appendix C, Eq 121 — error = M@k - v (prediction minus target)
              Titans (2501.00663) §3.2 — L2 regression inner objective
              CS-32 (observe-then-advance) — diagnostic read after forward, before backward
              CS-40 (opt-in tape) — only computed when tape_device != "off"
```

---

## 1. Problem: The Missing Inner-Loop Signal

The stacked tape (spec 15) captures **output gradient norms** — how much each level
contributes to the backward pass. This answers "where does gradient explosion start?"
but NOT "why does the memory stop learning?"

The DGD error vector `e_t = M_t @ k_t - v_t` is the **prediction residual** — what
memory predicts (`M_t @ k_t`) minus what it should predict (`v_t`). Its norm is the
fundamental health indicator of the inner loop:

| ‖e_t‖₂ behavior | Diagnosis |
|---|---|
| Decreasing over tokens | Memory is learning — DGD is working |
| Flat near zero | Memory has converged for this context (healthy) |
| Flat near ‖v‖₂ | Memory is NOT learning — M@k ≈ 0 (dead memory) |
| Increasing | Memory predictions diverging — M-norm explosion imminent |
| NaN/Inf | Already crashed |

The CPU tape captures this via `obs::DGD_DELTA` in `opaque_adapters.rs` (line ~1095),
but the GPU tape returns `dgd_delta_norm: 0.0` as a placeholder. This spec fills that gap.

### Source Equations

From HOPE (2512.24695) Eq 88 / Eq 121, the DGD update:

```text
M_{t+1} = (1 - α_t) · M_t - θ_t · (M_t @ k_t - v_t) @ k_t^T
                                ^^^^^^^^^^^^^^^^
                                  error vector e_t
```

The error is computed explicitly in both Titans and DGD CUDA forward kernels:
- `titans_forward.cu` line 191-193: `error_buf[row] = prediction[row] - v_t[row]`
- `dgd_forward.cu` line 7-8: same pattern

The error lives in shared memory (`error_buf[d]`) for exactly one `__syncthreads`
window before it's consumed by the M-update. We compute its norm in that window.

---

## 2. Architecture: Separate Post-Pass Kernel + Cache Extractor

### 2.1 Design Decision: Post-Pass Over In-Kernel

The original spec proposed adding `delta_norm_out` to the forward kernel signatures.
The implementation chose a **separate post-pass kernel** instead, for three reasons:

1. **Zero impact on forward kernels**: The Titans/DGD/Delta forward kernels are
   performance-critical and already register-heavy. Adding a conditional norm
   reduction inside the token loop risks register spills and instruction cache
   pressure even when the diagnostic is off.
2. **Simpler maintenance**: Forward kernel signatures are shared across full-trajectory,
   checkpointed, and segment-based variants (12 kernels). Adding a parameter to all
   12 would be high-touch. The post-pass kernel is a single function.
3. **Cache already stores M and k/v**: The `GpuMemoryCache` variants retain `m_states`
   (or `m_checkpoints`) and `k_mem`/`v_mem` from the forward pass. Computing
   `‖M@k - v‖₂` from these buffers after the forward is complete requires no
   additional data flow — just a separate kernel invocation.

### 2.2 Post-Pass Kernel: `dgd_delta_norm_cuda`

Implemented in `core/kernels/elementwise.cu`:

```cuda
// Single-block kernel: computes ‖M @ k - v‖₂
// M is [d,d], k is [d], v is [d]. Writes scalar to norm_out[0].
extern "C" void dgd_delta_norm_cuda(
    const float* M, const float* k, const float* v,
    float* norm_out, int d)
```

Algorithm:
1. **Matvec**: `prediction[row] = M[row,:] · k` (strided, shared memory)
2. **Error + sum-of-squares**: `local_sq += (prediction[row] - v[row])²`
3. **Warp reduction**: `__shfl_down_sync` within each warp
4. **Inter-warp reduction**: via shared memory scratch (reuses `prediction[]`)
5. **Final output**: `norm_out[0] = sqrt(total)`

Block size is rounded up to the nearest warp boundary to ensure full-warp
shuffle semantics. Single block launch — sufficient for d ≤ 1024.

### 2.3 Cache-Side Extractor: `GpuMemoryCache::dgd_delta_norm()`

Implemented in `core/src/gpu_forward.rs` as a method on `GpuMemoryCache`.
Extracts M, k, v from the forward cache and calls `dgd_delta_norm_cuda`:

- **Delta / DGD / Titans**: Uses pre-update `M_{s-1}` at offset `(s-1)*dd` in
  `m_states`, with `k_{s-1}` and `v_{s-1}`. This matches the error that drove
  the final M-update: `e_{s-1} = M_{s-1} @ k_{s-1} - v_{s-1}`.
- **Checkpointed variants** (DeltaCkpt / DGDCkpt / TitansCkpt): Uses the
  second-to-last checkpoint as an approximation of the pre-update M. Exact when
  `s` falls on a checkpoint boundary; at most `checkpoint_interval` tokens stale
  otherwise. This is acceptable for a diagnostic signal.
- **TNT**: Iterates shard inner caches, calls `dgd_delta_norm` recursively on each
  shard's Titans/Delta cache, returns max across all shards.
- **Hebbian / SwiGlu**: Returns 0.0 (no error vector in these rules).

### 2.4 Wiring: GPU Tape Summary

In `python/src/lib.rs`, `gpu_tape_forward_summary()` calls
`cache.memory_caches[level].dgd_delta_norm(s, d, bs)` after the diagnostic
forward pass and populates the `dgd_delta_norm` field in the returned dict.
No forward kernel signature changes required.

---

## 4. Integration with Tape Summary

### 4.1 GpuModel::gpu_tape_forward_summary()

The GPU tape summary runs a diagnostic forward+backward. After the forward pass,
the cache-side extractor computes `dgd_delta_norm` from the retained M and k/v
buffers. No forward kernel signature changes needed — the norm is a post-pass
computation on cached data.

```rust
// In python/src/lib.rs, gpu_tape_forward_summary()
let delta_norm = cache.memory_caches[level]
    .as_ref()
    .map(|mc| mc.dgd_delta_norm(s, d, bs))
    .unwrap_or(0.0);
ldict.set_item("dgd_delta_norm", delta_norm)?;
```

### 4.2 Dict Schema Update

The `dgd_delta_norm` field in the tape summary dict (previously hardcoded to 0.0
on the GPU path) is now populated with the actual norm:

```python
{
    "blocks": [
        {
            "block_index": 0,
            "levels": [
                {
                    "level": 0,
                    "output_grad_norm": 0.020020,
                    "m_norm": 0.9344,
                    "dgd_delta_norm": 1.2345,   # ◀─── was 0.0, now real
                    "opaque_key": "Titans",
                    "block_count": 1,
                }, ...
            ]
        }, ...
    ]
}
```

### 4.3 Diagnostic Interpretation (Extended from spec 15)

| gnorm | ‖M‖_F | dgd_delta_norm | Interpretation |
|---|---|---|---|
| > 0 | growing | growing | Active memory, healthy gradient — learning |
| > 0 | growing | ≈ 0 | M growing but error collapsed — overfitting to context |
| > 0 | stable | stable | Converged — memory at equilibrium |
| ≈ 0 | stable | > 0 | DEAD level: error exists but gradient not flowing back |
| ≈ 0 | stable | ≈ 0 | Truly dead: no error, no gradient, no learning |
| exploding | exploding | exploding | NaN imminent — M-norm clamp insufficient |
| > 0 | small | ≈ ‖v‖₂ | Memory not learning — M@k ≈ 0 (cold start) |

The combination of `gnorm`, `m_norm`, and `dgd_delta_norm` gives a three-axis
diagnostic: backward signal (gnorm), memory state (m_norm), forward signal (delta_norm).
Together they disambiguate "dead level" (no gradient flow) from "collapsed memory"
(no prediction error) from "healthy convergence" (small error, small gradient, stable M).

---

## 5. Python Display

### 5.1 Enhanced `print_stacked_tape_summary()`

```python
def print_stacked_tape_summary(tape_summary: dict, step: int) -> None:
    for block in tape_summary["blocks"]:
        bi = block["block_index"]
        for lev in block["levels"]:
            li = lev["level"]
            gnorm = lev["output_grad_norm"]
            m_norm = lev.get("m_norm")
            delta = lev.get("dgd_delta_norm", 0.0)
            status = "ACTIVE" if lev["block_count"] > 0 else "frozen"

            m_str = f"  ‖M‖={m_norm:.4f}" if m_norm is not None else ""
            d_str = f"  Δ={delta:.4f}" if delta > 0 else ""
            flag = ""
            if gnorm < 1e-6 and lev["block_count"] > 0:
                flag = " ◀ DEAD"
            elif gnorm > 100.0:
                flag = " ◀ EXPLODING"
            elif delta > 0 and delta < 1e-6 and lev["block_count"] > 0:
                flag = " ◀ COLLAPSED"

            print(f"    block[{bi}] level[{li}] {status:>6}  "
                  f"gnorm={gnorm:.6f}{m_str}{d_str}{flag}")
```

### 5.2 JSONL Tape Event

```python
{
    "event": "tape_summary",
    "step": 512,
    "loss": 6.234,
    "n_blocks": 4,
    "blocks": [
        {
            "block_index": 0,
            "levels": [
                {"level": 0, "output_grad_norm": 0.020, "m_norm": 0.93,
                 "dgd_delta_norm": 1.23},
                {"level": 1, "output_grad_norm": 0.018, "m_norm": 0.33,
                 "dgd_delta_norm": 0.87},
                ...
            ]
        }, ...
    ]
}
```

---

## 6. Non-DGD Memory Rules

The `delta_norm_out` mechanism applies to any rule that computes `error = M@k - v`:
- **Titans LMM**: yes (same error, plus momentum S)
- **DGD**: yes (identical error)
- **Delta Rule**: yes (identical error, different gate semantics)
- **Hebbian**: NO — Hebbian uses `v @ k^T` directly, no error term
- **MONETA/YAAD/MEMORA**: MLP rules — error is implicit in the MLP loss, not a simple vector

For Hebbian levels, `dgd_delta_norm` remains 0.0 (no error vector exists).
For MLP rules, a future spec could expose the MLP's internal loss as the diagnostic.

The cache extractor returns 0.0 for these variants — zero overhead.

---

## 7. Files Modified

| File | Change |
|---|---|
| `core/kernels/elementwise.cu` | `dgd_delta_norm_kernel` + `dgd_delta_norm_cuda` C wrapper |
| `core/src/cuda_ffi.rs` | FFI declaration for `dgd_delta_norm_cuda` |
| `core/src/gpu_forward.rs` | `GpuMemoryCache::dgd_delta_norm()` — cache-side extractor for all 10 variants |
| `python/src/lib.rs` | Wire `dgd_delta_norm` into `gpu_tape_forward_summary()` |

Forward kernel files (`titans_forward.cu`, `dgd_forward.cu`, etc.) are **unchanged**.
New files: none.

---

## 8. Build Order

1. Add `dgd_delta_norm_kernel` + C wrapper to `core/kernels/elementwise.cu`
2. Add FFI declaration to `core/src/cuda_ffi.rs`
3. Add `GpuMemoryCache::dgd_delta_norm()` method in `core/src/gpu_forward.rs`
4. `cargo build --release --features cuda` — verify compiles, all existing tests pass
5. Wire into `gpu_tape_forward_summary()` in `python/src/lib.rs`
6. `maturin develop --release --features cuda` — build Python bindings
7. Smoke test: run shakedown with `tape_device: "gpu"`, verify `dgd_delta_norm > 0`

---

## 9. Testing

### CUDA kernel tests

```rust
#[test]
fn test_dgd_delta_norm_matches_cpu() {
    // Construct known M [d,d], k [d], v [d] on GPU.
    // Call dgd_delta_norm_cuda, read back scalar.
    // Compare against CPU: sqrt(sum((M@k - v)^2)). Tolerance: 1e-5.
}

#[test]
fn test_dgd_delta_norm_zero_error() {
    // Set v = M @ k exactly. Verify norm_out ≈ 0.0.
}
```

### Python integration tests

```python
def test_tape_dgd_delta_nonzero():
    """After tape summary, Titans levels should have dgd_delta_norm > 0."""
    summary = gpu_model.gpu_tape_forward_summary(input_ids, target_ids, pulse)
    for lev in summary["levels"]:
        if lev["block_count"] > 0:
            assert lev["dgd_delta_norm"] > 0, \
                f"level {lev['level']}: delta=0"
```

---

## 10. HADES Registration

```json
{
  "_key": "dgd-delta-norm-gpu",
  "title": "DGD Delta Norm — GPU-Resident Inner-Loop Error Observability",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/16_dgd_delta_norm_gpu.md",
  "purpose": "Expose DGD prediction error norm from CUDA forward kernels as per-(block, level) diagnostic",
  "paper_source": ["2512.24695", "2501.00663"],
  "traced_to_equations": [
    "hope_equations/eq-088-practical-dgd-update",
    "hope_equations/eq-121-delta-gd-final",
    "hope_equations/eq-057-delta-gd"
  ],
  "traced_to_axioms": [],
  "depends_on_specs": [
    "hecate_specs/stacked-tape-diagnostics",
    "hecate_specs/dgd"
  ],
  "status": "v0.4.0"
}
```
