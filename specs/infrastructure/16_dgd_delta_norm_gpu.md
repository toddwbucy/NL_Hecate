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
              - Norm computed from the LAST token's error in each level's forward pass
                (the most diagnostic: it reflects M's state after processing the full sequence)
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

```
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

## 2. Architecture: Side-Channel Norm Output

### 2.1 Kernel Signature Change

Add an optional output pointer to both forward kernels:

```cuda
// titans_forward_kernel — MODIFIED signature
__global__ void titans_forward_kernel(
    const float* __restrict__ k_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ alpha,      // [batch_size, seq_len]
    const float* __restrict__ theta,      // [batch_size, seq_len]
    const float* __restrict__ eta,        // [batch_size, seq_len]
    const float* __restrict__ m_initial,  // [batch_size, d*d]
    const float* __restrict__ s_initial,  // [batch_size, d*d]
    float* __restrict__ m_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ s_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ y,                // [batch_size, seq_len, d]
    float* __restrict__ delta_norm_out,   // [batch_size] or NULL  ◀─── NEW
    int seq_len, int d)
```

When `delta_norm_out == NULL`, no norm computation occurs — identical to current behavior.
When non-null, the kernel writes `‖error_{seq_len-1}‖₂` (last token's error norm) for
each batch element.

Same change for `dgd_forward_kernel` (unbatched: `delta_norm_out` is `float*` to a single float, or NULL).

### 2.2 Norm Computation in Token Loop

After `error_buf` is populated (line 194 in titans_forward.cu), and only on the
**last token** (`t == seq_len - 1`), compute the L2 norm:

```cuda
// After error computation, before M-update
// Only on last token — the most diagnostic error sample
if (t == seq_len - 1 && delta_norm_out != NULL) {
    // Parallel sum-of-squares reduction using shared memory
    // error_buf[d] already in smem — reuse prediction[] as scratch
    float local_sq = 0.0f;
    for (int row = tid; row < d; row += blockDim.x) {
        local_sq += error_buf[row] * error_buf[row];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
    }

    // Inter-warp reduction via shared memory (reuse prediction[] as scratch)
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) prediction[warp_id] = local_sq;
    __syncthreads();

    // Final reduce in warp 0
    if (warp_id == 0) {
        float val = (lane < (blockDim.x + warpSize - 1) / warpSize)
                    ? prediction[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) {
            delta_norm_out[b] = sqrtf(val);  // L2 norm
        }
    }
    __syncthreads();
}
```

Key design choices:
- **Last token only**: The error after processing the full sequence is the most
  meaningful diagnostic. Per-token norms would require `seq_len × batch_size` output
  floats — overkill for NaN tracing.
- **Reuse `prediction[]` as scratch**: After the error is computed, `prediction[]` is
  dead until the next token. Safe to use for the reduction without additional smem.
- **Warp shuffle + smem**: Standard two-phase reduction pattern. At d=512, blockDim=1024,
  this is 32 warps → ~5 smem entries → negligible.

### 2.3 Checkpointed Forward Variants

The checkpointed forward kernels (`titans_forward_ckpt_kernel`, `dgd_forward_ckpt_kernel`)
have the same token loop structure. Apply the identical norm computation on the last
token of the LAST chunk (the chunk containing `seq_len - 1`).

For the segment-based variants (TNT chunkwise), the error of the last token in the
last segment is used. This is slightly different from the full-trajectory last token
but still diagnostically meaningful.

---

## 3. Rust FFI Changes

### 3.1 C Wrapper Signatures

```c
// In core/kernels/titans_forward.cu (C wrapper)
extern "C" void titans_forward_cuda(
    const float* k, const float* v, const float* q,
    const float* alpha, const float* theta, const float* eta,
    const float* m_init, const float* s_init,
    float* m_states, float* s_states, float* y,
    float* delta_norm_out,   // NULL disables norm output
    int batch_size, int seq_len, int d)

// In core/kernels/dgd_forward.cu (C wrapper)
extern "C" void dgd_forward_cuda(
    const float* k, const float* v, const float* q,
    const float* alpha, const float* theta,
    const float* m_init,
    float* m_states, float* y,
    float* delta_norm_out,   // NULL disables norm output
    int seq_len, int d)
```

### 3.2 Rust cuda_ffi Bindings

```rust
// In core/src/cuda_ffi.rs
extern "C" {
    pub fn titans_forward_cuda(
        k: *const f32, v: *const f32, q: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_init: *const f32, s_init: *const f32,
        m_states: *mut f32, s_states: *mut f32, y: *mut f32,
        delta_norm_out: *mut f32,  // null_mut() disables
        batch_size: i32, seq_len: i32, d: i32,
    );

    pub fn dgd_forward_cuda(
        k: *const f32, v: *const f32, q: *const f32,
        alpha: *const f32, theta: *const f32,
        m_init: *const f32,
        m_states: *mut f32, y: *mut f32,
        delta_norm_out: *mut f32,  // null_mut() disables
        seq_len: i32, d: i32,
    );
}
```

### 3.3 Forward Call Sites

In `gpu_forward.rs` and `gpu_stacked_forward.rs`, the forward dispatch currently
passes the existing arguments. Change to:

```rust
// Normal training step — no delta norm needed
unsafe {
    titans_forward_cuda(
        /* existing args... */
        std::ptr::null_mut(),  // delta_norm_out = NULL → no overhead
        batch_size, seq_len, d,
    );
}

// Diagnostic tape step — capture delta norm
let mut delta_norm_buf = GpuBuf::<f32>::zeros(batch_size);
unsafe {
    titans_forward_cuda(
        /* existing args... */
        delta_norm_buf.ptr(),  // non-NULL → kernel computes norm
        batch_size, seq_len, d,
    );
}
// Read back the single float per batch element
let mut delta_norms = vec![0.0f32; batch_size];
delta_norm_buf.copy_to_host(&mut delta_norms);
// delta_norms[0] = ‖error_{last_token}‖₂ for batch element 0
```

---

## 4. Integration with Stacked Tape Summary

### 4.1 GpuStackedModel::gpu_stacked_tape_summary()

Currently (spec 15 implementation), the tape summary method runs a diagnostic
forward+backward. Modify the diagnostic forward to pass non-null `delta_norm_out`:

```rust
// In python/src/lib.rs, gpu_stacked_tape_summary()
// For each block b, for each active level l:
//   Run forward with delta_norm_out enabled
//   Read back delta_norm scalar
//   Store in tape dict as dgd_delta_norm

// The forward call already dispatches per-level to Titans/DGD kernel.
// Just pass the delta_norm_out pointer on diagnostic calls.
```

### 4.2 Dict Schema Update

The `dgd_delta_norm` field in the tape summary dict (currently hardcoded to 0.0 on
GPU path) gets populated:

```python
{
    "blocks": [
        {
            "block_index": 0,
            "levels": [
                {
                    "level": 0,
                    "gnorm": 0.020020,
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
                {"level": 0, "gnorm": 0.020, "m_norm": 0.93,
                 "dgd_delta_norm": 1.23},
                {"level": 1, "gnorm": 0.018, "m_norm": 0.33,
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

The kernel-level NULL-pointer guard ensures zero overhead for rules that don't compute
an error vector.

---

## 7. Files to Create/Modify

| File | Change |
|---|---|
| `core/kernels/titans_forward.cu` | Add `delta_norm_out` parameter; warp reduction on last token |
| `core/kernels/dgd_forward.cu` | Same as above |
| `core/kernels/titans_forward.cu` (ckpt variant) | Same, on last token of last chunk |
| `core/kernels/dgd_forward.cu` (ckpt variant) | Same |
| `core/src/cuda_ffi.rs` | Update extern "C" signatures with `delta_norm_out` |
| `core/src/gpu_forward.rs` | Pass `null_mut()` on normal calls |
| `core/src/gpu_stacked_forward.rs` | Pass `null_mut()` on normal calls |
| `python/src/lib.rs` | Pass non-null on diagnostic calls; populate `dgd_delta_norm` |
| `python/engine/evaluation.py` | Show `Δ=` in stacked tape display |

New files: none.

---

## 8. Build Order

1. Modify `titans_forward_kernel` signature + add norm reduction (CUDA)
2. Modify `dgd_forward_kernel` same way (CUDA)
3. Update C wrappers in both `.cu` files
4. Update `cuda_ffi.rs` extern declarations
5. Update all call sites in `gpu_forward.rs` / `gpu_stacked_forward.rs` to pass `null_mut()`
6. `cargo build --release --features cuda` — verify compiles, all existing tests pass
7. Wire non-null pointer in `gpu_stacked_tape_summary()` diagnostic path
8. Update `evaluation.py` display
9. `maturin develop --release --features cuda` — build Python bindings
10. Smoke test: run stacked shakedown, verify `dgd_delta_norm > 0` in tape output

---

## 9. Testing

### CUDA kernel tests

```rust
#[test]
fn test_titans_forward_delta_norm_null() {
    // Pass delta_norm_out = null_mut(). Verify output y matches existing test.
    // Ensures NULL path has zero behavior change.
}

#[test]
fn test_titans_forward_delta_norm_value() {
    // Pass delta_norm_out = allocated buffer. Run forward on known input.
    // Verify delta_norm_out[0] = ||error_{last_token}||_2 matches CPU computation.
}

#[test]
fn test_dgd_forward_delta_norm_matches_cpu() {
    // Run both CPU dgd_step (core/src/dgd.rs) and CUDA dgd_forward_cuda.
    // Compare final-token error norm. Tolerance: 1e-5.
}
```

### Python integration tests

```python
def test_stacked_tape_dgd_delta_nonzero():
    """After tape summary, Titans levels should have dgd_delta_norm > 0."""
    summary = gpu_model.gpu_stacked_tape_summary(input_ids, target_ids, pulse)
    for block in summary["blocks"]:
        for lev in block["levels"]:
            if lev["block_count"] > 0:
                assert lev["dgd_delta_norm"] > 0, \
                    f"block {block['block_index']} level {lev['level']}: delta=0"
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
