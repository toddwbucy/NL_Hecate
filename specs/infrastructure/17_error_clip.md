# Per-Token Error Clipping — Inner-Loop Gradient Bound

```text
CONTRACT
Purpose:   Clip the per-token prediction error e_t = M@k - v inside CUDA
           forward (and backward) kernels to prevent inner-loop gradient
           explosion that leads to NaN.  When ‖e_t‖₂ > error_clip, rescale
           e_t so ‖e_t‖₂ = error_clip.  Straight-through estimator on
           backward (identity Jacobian through the clamp, same as M-norm
           clamp CS-39).

Expects:
  - Titans/Delta/DGD forward kernels computing error_buf[i] = prediction[i] - v_t[i]
  - Titans/Delta/DGD backward kernels recomputing the same error for gradient
  - MAGConfig with per-level configuration (m_norm_max pattern)
  - Spec 16 (DGD delta norm) providing ‖e_t‖ observability in the tape

Guarantees:
  - ‖e_t‖₂ ≤ error_clip at every token in every forward and backward kernel
  - Forward and backward clip identically (gradient consistency)
  - error_clip=0.0 disables clipping (zero overhead: branch never entered)
  - Per-level config: different levels may have different clip thresholds
  - No new device memory: norm reduction reuses existing prediction[] smem scratch
  - Existing tests unaffected (error_clip defaults to 0.0)

Cost:
  Per token per level: one d-element warp reduction (~20 cycles at d=512)
  plus conditional rescale (d multiplies).  The M-update that follows is
  O(d²), so the clip is <0.1% overhead.  When error_clip=0.0 the branch
  is skipped entirely — zero cost for unconfigured runs.

Trade-off:
  Clipping the error is inner-loop gradient clipping.  It bounds the
  M-update magnitude: ‖θ·e_t⊗k_t‖_F ≤ θ·error_clip·‖k_t‖, regardless
  of M's spectral structure.  This is strictly weaker than a spectral
  norm clamp on M (which would prevent the error from growing in the
  first place) but is 100x cheaper and directly targets the quantity
  observed to explode in tape diagnostics.  If error clipping alone is
  insufficient, spectral norm clamp can be layered on top.

  The straight-through backward means gradients for theta/alpha/eta
  see the clipped error, not the true error.  This introduces bias
  in outer-loop gate gradients when the clip activates.  In practice,
  the clip only fires during pathological divergence (Δ > 50 when
  healthy range is 5-15), so the bias affects only steps that would
  otherwise produce NaN.

Position:  specs/infrastructure/17_error_clip.md
Extends:   specs/infrastructure/16_dgd_delta_norm_gpu.md (observable signal)
Depends:   specs/constraints/code_smells/CS-39 (clamp pattern precedent)
Source:
  HOPE (2512.24695) Eq 88 — DGD update: M -= θ·(M@k - v)⊗k^T
  HOPE (2512.24695) Eq 121 — error = M@k - v
  Titans (2501.00663) §3.2 — L2 regression inner objective
  CS-39 (theta clamp) — precedent for in-kernel clamping with straight-through
```

---

## 1. Problem: Spectral Concentration Defeats Frobenius Clamp

The M-norm clamp (CS-39, `m_norm_clamp.cu`) bounds ‖M‖_F but not the
spectral norm σ_max(M).  A rank-1 matrix with ‖M‖_F=100 has σ_max=100,
producing ‖M@k‖ ≤ 100·‖k‖ ≈ 1500 at d=512.  The prediction error
e_t = M@k - v therefore grows unboundedly even though ‖M‖_F is clamped.

Tape evidence from shakedown runs (spec 16 diagnostics, 2026-03-09):

| Run | Step | Block | Level | ‖e_t‖₂ (Δ) | ‖M‖_F | Next event |
|-----|------|-------|-------|-------------|-------|------------|
| Titans | 384 | B2 L0 | 0 | 15.6 | 26.3 | — |
| Titans | 448 | B1 L0 | 0 | 11.9 | 55.6 | — |
| Titans | 512 | B2 L0 | 0 | **324.4** | 100.0 | NaN at 520 |
| TNT | 1792 | B3 L0 | 0 | 147.1 | 100.0 | — |
| TNT | 1856 | B0 L0 | 0 | **2,565.7** | 100.0 | — |
| TNT | 1920 | B0 L0 | 0 | **43,233.5** | 100.0 | NaN at 1932 |

The error escalation is 12 → 324 → 2,566 → 43,234 — exponential growth
while ‖M‖ sits at the 100.0 ceiling.  The Frobenius clamp cannot prevent
this because it constrains total energy, not directional concentration.

---

## 2. Solution: Clip e_t In-Place

After computing `error_buf[row] = prediction[row] - v_t[row]` and before
the M-update outer product, insert:

```rust
let norm = error_buf.iter().map(|e| e * e).sum::<f32>().sqrt();
if norm > error_clip {
    let scale = error_clip / norm;
    for e in error_buf.iter_mut() { *e *= scale; }
}
```

This bounds the M-update:
```text
‖θ·e_t ⊗ k_t‖_F = θ · ‖e_t‖ · ‖k_t‖ ≤ θ · error_clip · ‖k_t‖
```

At error_clip=50, θ≈0.01, ‖k‖≈15: update magnitude ≤ 7.5 per token.
Compared to unclamped: θ·43233·15 ≈ 6,485 — a 865x reduction.

---

## 3. CUDA Insertion Pattern

Applied identically in all 12 kernel functions (3 rules × 2 variants × fwd/bwd).

After the existing error computation:
```cuda
for (int row = tid; row < d; row += blockDim.x) {
    error_buf[row] = prediction[row] - v_t[row];
}
__syncthreads();
```

Insert:
```cuda
// ── Per-token error clipping (spec 17) ──────────────────
if (error_clip > 0.0f) {
    // Step 1: partial ‖error‖²
    float local_sq = 0.0f;
    for (int row = tid; row < d; row += blockDim.x)
        local_sq += error_buf[row] * error_buf[row];

    // Warp reduction
    for (int off = warpSize / 2; off > 0; off >>= 1)
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);

    // Inter-warp via prediction[] scratch (dead until next token)
    int warp_id = tid / warpSize, lane = tid % warpSize;
    if (lane == 0) prediction[warp_id] = local_sq;
    __syncthreads();

    if (tid == 0) {
        int nw = (blockDim.x + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += prediction[w];
        prediction[0] = total;  // ‖error‖² in prediction[0]
    }
    __syncthreads();

    float err_norm = sqrtf(prediction[0]);
    if (err_norm > error_clip) {
        float scale = error_clip / err_norm;
        for (int row = tid; row < d; row += blockDim.x)
            error_buf[row] *= scale;
    }
    __syncthreads();
}
```

Key properties:
- `prediction[]` is safe to reuse: it's dead after error computation,
  before M-update.  Same scratch reuse as in spec 16 delta norm kernel.
- When `error_clip == 0.0f`, the branch is never entered — zero overhead.
- The `__syncthreads()` count matches the existing kernel pattern (one
  after error, one after M-update).  The clip inserts two additional
  syncs inside the conditional block.

---

## 4. Backward Consistency

Backward kernels recompute `error_buf` identically (same M states from
forward cache).  They MUST apply the same clipping to maintain gradient
consistency.  If forward clips error from 43,233 → 50 but backward uses
the raw 43,233, the VJP is wrong.

Straight-through estimator: the Jacobian of the clamp is identity (same
as CS-39 theta clamp).  Gradients for d_theta, d_alpha, d_eta, d_k, d_v
flow through the clipped error value, not through the clamp operation
itself.  This is standard practice for gradient clipping in meta-learning
inner loops.

---

## 5. Files to Modify

### CUDA kernels (6 forward + 6 backward = 12 kernel functions)

| File | Kernel functions | Change |
|------|-----------------|--------|
| `core/kernels/titans_forward.cu` | `titans_forward_kernel`, `titans_forward_ckpt_kernel` | Add `float error_clip` param; insert clip block |
| `core/kernels/dgd_forward.cu` | `dgd_forward_kernel`, `dgd_forward_ckpt_kernel` | Same |
| `core/kernels/delta_forward.cu` | `delta_forward_kernel`, `delta_forward_ckpt_kernel` | Same |
| `core/kernels/titans_backward.cu` | `titans_backward_kernel`, `titans_backward_segment_kernel` | Same |
| `core/kernels/dgd_backward.cu` | `dgd_backward_kernel`, `dgd_backward_segment_kernel` | Same |
| `core/kernels/delta_backward.cu` | `delta_backward_kernel`, `delta_backward_segment_kernel` | Same |

Each file's `extern "C"` wrapper also gets `float error_clip` added to its
signature and passed through to the kernel launch.

### Rust tier

| File | Change |
|------|--------|
| `core/src/model.rs` | Add `pub error_clip: Vec<f32>` to `MAGConfig`; add `error_clip_for_level(level) -> f32` method; default empty vec (0.0 = disabled) |
| `core/src/cuda_ffi.rs` | Add `error_clip: f32` to all 12 `extern "C"` function signatures |
| `core/src/gpu_forward.rs` | Pass `cfg.error_clip_for_level(level)` to forward kernel dispatch |
| `core/src/gpu_backward.rs` | Pass `cfg.error_clip_for_level(level)` to backward kernel dispatch |
| `core/src/gpu_stacked_forward.rs` | Thread `error_clip` per level through stacked forward |
| `core/src/gpu_stacked_backward.rs` | Thread `error_clip` per level through stacked backward |

### Python tier

| File | Change |
|------|--------|
| `python/engine/config.py` | Add `error_clip: list[float] \| None = None` to model config |
| `python/src/lib.rs` | Wire `error_clip` from Python config into `MAGConfig` construction |

### No new files

All modifications to existing files.

---

## 6. Config

```json
{
    "model": {
        "error_clip": [50.0, 50.0, 50.0, 50.0],
        "m_norm_max": [100.0, 100.0, 100.0, 100.0]
    }
}
```

Per-level array, same as `m_norm_max`.  Recommended starting value: 50.0
(3-4x the healthy Δ range of 12-15, well below the pathological 300+ range).

When omitted or empty: no clipping (backward compatible).

---

## 7. Build Order

1. CUDA forward kernels — add `error_clip` param + clip block (3 files)
2. CUDA backward kernels — same (3 files)
3. `cuda_ffi.rs` — update 12 extern signatures
4. `model.rs` — add `error_clip` field + accessor
5. `gpu_forward.rs`, `gpu_backward.rs` — pass through to kernel calls
6. `gpu_stacked_forward.rs`, `gpu_stacked_backward.rs` — pass through
7. `cargo build --release --features cuda` — verify compiles
8. `config.py`, `lib.rs` — Python config wiring
9. `maturin develop --release --features cuda` — build bindings
10. Smoke test: run with `error_clip: [50.0, ...]`, verify Δ capped at 50
11. Re-run both shakedowns with error_clip enabled

---

## 8. Verification

1. **Clip fires**: tape shows Δ ≤ error_clip at all (block, level) pairs
2. **No NaN**: both Titans and TNT runs survive past their previous NaN steps
3. **Learning preserved**: loss continues decreasing after clip activates
4. **Backward consistency**: FD gradient check at d=64 with error_clip=5.0 —
   GPU backward matches FD within 10% tolerance
5. **Zero overhead when disabled**: benchmark with error_clip=0.0 shows
   identical tok/s to current builds
6. **Per-token clip**: inject a synthetic sequence where tokens 0..s/2 have
   ‖e_t‖ < error_clip (should pass through unchanged) and tokens s/2..s have
   ‖e_t‖ >> error_clip (should be rescaled to error_clip).  Dump error_buf
   norms before/after the clip block via a debug kernel; verify the bound
   holds at every token position, not just the last

---

## 9. Diagnostic Interaction with Spec 16

The DGD delta norm (spec 16) reports ‖e_t‖ from the LAST token.  With
error clipping, the reported Δ will be min(actual_error, error_clip).
This is correct — the tape shows the effective error that drives the
M-update, not the hypothetical unclipped value.

To observe the unclipped error for research purposes, run with
`error_clip: []` (disabled) and `tape_device: "gpu"`.

---

## 10. HADES Registration

```json
{
    "_key": "error-clip",
    "title": "Per-Token Error Clipping — Inner-Loop Gradient Bound",
    "category": "infrastructure",
    "version": "0.4.0",
    "path": "specs/infrastructure/17_error_clip.md",
    "purpose": "Clip prediction error e_t = M@k-v in CUDA kernels to prevent inner-loop gradient explosion",
    "paper_source": ["2512.24695", "2501.00663"],
    "traced_to_equations": [
        "hope_equations/eq-088-practical-dgd-update",
        "hope_equations/eq-121-delta-gd-final"
    ],
    "traced_to_axioms": [],
    "depends_on_specs": [
        "hecate_specs/dgd-delta-norm-gpu",
        "hecate_specs/m-norm-clamp"
    ],
    "status": "v0.4.0"
}
```
