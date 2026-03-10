# BUG-03: Wire Learnable Alpha Aggregation in Stacked Multi-Block Forward/Backward

```text
CONTRACT
  Purpose:    Replace the uniform-sum level combination in the stacked multi-block
              forward and backward paths with softmax-weighted aggregation using
              the existing alpha_mem parameter, matching the single-block MAC path.
  Expects:    BUG-01 (W_O) and BUG-02 (MAG sigmoid gating) resolved. alpha_mem
              already allocated in BlockParams [k] (stacked_model.rs:36), uploaded
              to GpuBlockParams (gpu_params.rs:445), serialized in checkpoints
              (checkpoint.rs:69-73), and initialized to zeros (uniform 1/k at
              softmax). Reference softmax + backward in mac.rs:31-47 and 904-915.
  Guarantees: After fix, each block computes:
                weights = softmax(alpha_mem)  — on host, [k] vector
                y_combined = Σ_l weights[l] * y_level[l]
              Backward computes d_alpha_mem via softmax Jacobian:
                d_alpha[l] = w[l] * (dot_l - Σ_j w[j] * dot_j)
              where dot_l = <d_y_combined, y_level[l]>.
              The 1/sqrt(k) normalization is removed — softmax weights sum to 1,
              providing implicit normalization that adapts during training.
              Optimizer wires alpha_mem with AdamW moment buffers.
  Cost:       One host-side softmax per block per forward (k=4: negligible).
              One k-element dot-product reduction per level per backward.
              k AdamW scalar updates per block per step. No new CUDA kernels.
  Trade-off:  Replaces fixed 1/sqrt(k) normalization with learned [0,1] weights
              that sum to 1. At init (alpha_mem=zeros → uniform 1/k), equivalent
              to mean rather than sum/sqrt(k). The model can learn to suppress
              dead levels (weight→0) or amplify active ones (weight→1). The slight
              numerical difference at init (1/k vs 1/sqrt(k)) is absorbed by the
              first few training steps.
  Position:   specs/infrastructure/21_stacked_alpha_aggregation.md
  Source:     HOPE (2512.24695) Section 5.1, eq-074 (Independent CMS Aggregation):
              y_t = Agg(MLP^(f_k)(x_t), ..., MLP^(f_1)(x_t))
              "A simple design choice for Agg() is a learnable weighted sum."
              Reference implementation: mac.rs:31-47 (softmax), mac.rs:904-915
              (softmax Jacobian backward).
```

## Bug Description

The stacked multi-block forward pass (`gpu_stacked_forward.rs:309-324`) combines
CMS level outputs via uniform sum with a fixed `1/sqrt(k)` normalization for k>2.
The `alpha_mem` parameter — allocated, uploaded, serialized, and initialized to
zeros in `BlockParams` — is never read during forward or backward. It is a dead
parameter: AdamW applies weight decay (pulling toward zero, which is already the
init value) but receives no gradient signal.

HOPE eq-074 specifies a "learnable weighted sum" as the aggregation function for
independent CMS levels. The single-block MAC path (`mac.rs:632-639`) already
implements this correctly using `softmax(alpha_mem)`. The stacked path does not.

## Paper Source

HOPE (2512.24695) Section 5.1, equation `hope_equations/eq-074-arch-variant5`:
```text
y_t = Agg(MLP^(f_k)(x_t), MLP^(f_{k-1})(x_t), ..., MLP^(f_1)(x_t))
```
Where:
- Each `MLP^(f_l)` is a frequency-indexed memory level processing input independently
- `Agg()` is an arbitrary aggregation function
- "A simple design choice for Agg() is a learnable weighted sum of the inputs"

The learnable weighted sum uses `softmax(alpha_mem)` to produce non-negative weights
that sum to 1, then computes `y_combined = Σ_l w[l] * y_level[l]`.

## Why It Matters

### At k=1 (Phase 1)
Not relevant — with one level, alpha_mem is a single-element vector and softmax
gives weight 1.0 regardless. BUG-03 is correctly identified as a soft blocker
that only matters at k≥2.

### At k=2 (Phase 2 onward)
With uniform weighting, both levels contribute equally to `y_combined` regardless
of whether one level has learned useful features and the other hasn't. In the
push-up curriculum:
- L1 (promoted from Phase 1's L0) has a trained M matrix producing useful outputs
- L0 (fresh, Xavier init) produces near-random outputs

Uniform weighting forces the sigmoid gate to see 50% noise from fresh L0. With
learnable aggregation, the model can initially weight L1 heavily (preserving
the warm start) and gradually increase L0's weight as it bootstraps.

### At k=4 (Phase 4)
L3 fires every 512 steps. Even with push-up warm starts, L3's contribution
per-token is small. Learnable weights let the model suppress L3's output until
it has accumulated enough memory updates to be useful, rather than adding noise
to every token's representation.

### Removing 1/sqrt(k)
The current `1/sqrt(k)` normalization is a fixed heuristic that keeps the
combined output magnitude roughly constant regardless of k. Softmax weights
sum to 1, so `y_combined` magnitude is bounded by `max_l ‖y_level[l]‖` — a
tighter, adaptive bound. The model starts at uniform 1/k (mean, not
sum/sqrt(k)) and can learn any convex combination.

At init:
- k=2: uniform = 1/2 = 0.5 vs 1/sqrt(2) ≈ 0.707 per level. Small difference.
- k=4: uniform = 1/4 = 0.25 vs 1/sqrt(4) = 0.5 per level. 2x difference.

The k=4 init difference means y_combined magnitude starts 2x smaller. This is
absorbed within the first few hundred steps as alpha_mem learns. The sigmoid
gate (BUG-02) bounds the impact on the residual stream regardless.

## Current Code (broken)

`gpu_stacked_forward.rs:309-324`:
```rust
// ── Combine levels: uniform sum with 1/sqrt(k) for k>2 ────
// NOTE: alpha_mem/alpha_refl are not yet used here. Level outputs are
// combined via uniform sum. When learnable aggregation is added, replace
// the uniform sum with softmax(alpha) weighted combination.
let mut y_combined = GpuBuf::<f32>::zeros(total);
for y_level in &y_per_level {
    unsafe {
        crate::cuda_ffi::saxpy_cuda(1.0, y_level.as_ptr(), y_combined.ptr(), total_i32);
    }
}
if cfg.k > 2 {
    let scale = 1.0 / (cfg.k as f32).sqrt();
    unsafe {
        crate::cuda_ffi::saxpy_cuda(scale - 1.0, y_combined.as_ptr(), y_combined.ptr(), total_i32);
    }
}
```

`gpu_stacked_backward.rs:211-217` (scale backward):
```rust
if cfg.k > 2 {
    let scale = 1.0 / (cfg.k as f32).sqrt();
    unsafe {
        crate::cuda_ffi::saxpy_cuda(scale - 1.0, d_y_combined.as_ptr(), d_y_combined.ptr(), bsd_i32);
    }
}
```

`gpu_stacked_backward.rs:244-269` (per-level backward passes same `d_y_combined`
to all levels uniformly).

`gpu_stacked_optimizer.rs:335-338`:
```rust
// NOTE: alpha_mem/alpha_refl are declared in BlockParams but not yet wired
// into the forward/backward passes (levels are combined via uniform sum with
// 1/sqrt(k) scaling). Once learnable aggregation is implemented, add moment
// buffers here and update the grad norm / scale functions accordingly.
```

`gradient.rs:445-448` (tape path returns zeros):
```rust
// TODO: Tape path doesn't yet extract alpha_mem/alpha_refl gradients.
// Hand-written backward (cms_mac_backward) computes these via softmax Jacobian.
// Tape-based alpha grads are a Stage 3 task (TapeOp::WeightedSum registration).
```

`traced_forward.rs:1383-1391` (uniform sum in traced path):
```rust
let mut combined_id = y_ids[0];
for i in 1..cfg.k {
    combined_id = traced_add(tape, combined_id, y_ids[i]);
}
if cfg.k > 2 {
    let scale = 1.0 / (cfg.k as f32).sqrt();
    combined_id = traced_scale(tape, combined_id, scale);
}
```

## Fix Specification

### Forward (gpu_stacked_forward.rs)

Replace the uniform sum block (lines 309-324) with softmax-weighted combination:

```rust
// ── Learnable level aggregation: weights = softmax(alpha_mem) ────
// Spec: specs/infrastructure/21_stacked_alpha_aggregation.md
// HOPE eq-074: y_t = Agg(...), "learnable weighted sum"
let alpha_host = block.alpha_mem.to_host(cfg.k);
let weights = softmax(&alpha_host);

let mut y_combined = GpuBuf::<f32>::zeros(total);
for (l, y_level) in y_per_level.iter().enumerate() {
    unsafe {
        crate::cuda_ffi::saxpy_cuda(weights[l], y_level.as_ptr(), y_combined.ptr(), total_i32);
    }
}
```

The `softmax()` function is computed on the host (k=4 elements — negligible cost
vs GPU kernel launch overhead). The result is k scalar weights used in k
`saxpy_cuda` calls, which are already the mechanism used today.

The `1/sqrt(k)` normalization block is removed entirely — softmax weights sum to
1, providing implicit normalization.

### Cache (GpuStackedBlockCache)

Add `alpha_weights: Vec<f32>` to the cache for backward use:

```rust
pub struct GpuStackedBlockCache {
    // ... existing fields ...
    pub alpha_weights: Vec<f32>,  // [k] — softmax(alpha_mem), needed for backward
}
```

### Backward (gpu_stacked_backward.rs)

Replace the uniform `d_y_combined` distribution and `1/sqrt(k)` scale backward
with weighted distribution + softmax Jacobian:

```rust
// ── Learnable aggregation backward ──────────────────────────────
// Forward: y_combined = Σ_l w[l] * y_level[l], w = softmax(alpha_mem)

// 1. Remove the 1/sqrt(k) scale backward (delete lines 211-217)

// 2. Per-level backward: scale d_y_combined by w[l] for each level
//    (replaces uniform d_y_combined distribution at lines 244-269)
for level in 0..cfg.k {
    // d_y_level[l] = w[l] * d_y_combined
    // Scale d_y_combined by w[l] before passing to memory backward
    ...
}

// 3. d_alpha_mem via softmax Jacobian (reference: mac.rs:904-915)
let mut d_alpha_mem = vec![0.0f32; cfg.k];
let dots: Vec<f32> = (0..cfg.k).map(|l| {
    // dot_l = <d_y_combined, y_level[l]> — inner product on GPU
    gpu_dot_product(&d_y_combined, &bc.y_per_level[l], bsd)
}).collect();
let weighted_dot_sum: f32 = (0..cfg.k)
    .map(|j| bc.alpha_weights[j] * dots[j])
    .sum();
for level in 0..cfg.k {
    d_alpha_mem[level] = bc.alpha_weights[level] * (dots[level] - weighted_dot_sum);
}
```

The inner product `<d_y_combined, y_level[l]>` can be computed via the existing
`grad_norm_sq_cuda` pattern (reduce on GPU, sum on host) or a dedicated dot
product kernel. Since k≤4 and this runs once per block per step, even a simple
`cublasSdot` call suffices.

### Optimizer (gpu_stacked_optimizer.rs)

Wire alpha_mem into AdamW at the NOTE comment (line 335):

```rust
// alpha_mem: k scalar parameters per block
// Moment buffers: m_alpha_mem [k], v_alpha_mem [k] per block
adamw_scalar_update(
    &mut block.alpha_mem, &d_alpha_mem,
    &mut mb.m_alpha_mem, &mut mb.v_alpha_mem,
    lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay,
);
```

Since alpha_mem is a tiny vector (k=4 elements), the AdamW update runs on the
host. No GPU kernel needed.

### CPU Traced Path (traced_forward.rs)

Replace the uniform add + scale (lines 1383-1391) with weighted sum:

```rust
// ── Learnable level aggregation ──────────────────────────────
let alpha_host = &block.alpha_mem;
let weights = softmax(alpha_host);
let mut combined_id = traced_scale(tape, y_ids[0], weights[0]);
for i in 1..cfg.k {
    let scaled = traced_scale(tape, y_ids[i], weights[i]);
    combined_id = traced_add(tape, combined_id, scaled);
}
```

The tape records `traced_scale` and `traced_add` operations. Gradients for
`alpha_mem` flow through the scale operations automatically via the Wengert tape.

### Gradient extraction (gradient.rs)

The TODO at line 445 about tape-based alpha gradients remains deferred — the
stacked GPU path uses hand-written backward (not the tape), so d_alpha_mem is
computed directly in `gpu_stacked_backward.rs`. The tape path (traced forward)
records the operations but gradient extraction for alpha_mem specifically requires
knowing which tape nodes correspond to the scale weights, which is a traced-path
concern separate from the GPU backward.

For the stacked GPU path (the production path), d_alpha_mem is computed analytically
via the softmax Jacobian as specified above. No tape involvement needed.

## alpha_refl: Not Applicable

`alpha_refl` is the MAC composition's "reflective memory" aggregation weight —
it weights the per-level outputs in the MAC's assembled-input construction
(`mac.rs:640-650`). The stacked path uses MAG composition (not MAC), which has no
reflective memory path. `alpha_refl` remains unused in the stacked forward/backward.

This spec does NOT wire `alpha_refl`. If a stacked MAC composition is added in
the future, alpha_refl would follow the same pattern.

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | Replace uniform sum + 1/sqrt(k) with softmax(alpha_mem) weighted sum; add alpha_weights to cache |
| `core/src/gpu_stacked_backward.rs` | Replace uniform d_y_combined with weighted per-level; add d_alpha_mem via softmax Jacobian; remove 1/sqrt(k) scale backward |
| `core/src/gpu_stacked_optimizer.rs` | Add AdamW moment buffers for alpha_mem; wire scalar update |
| `core/src/traced_forward.rs` | Replace traced_add sum + traced_scale(1/sqrt(k)) with traced_scale(w[l]) weighted sum |

## Acceptance Criteria

1. Forward uses `softmax(alpha_mem)` weights per block: `y_combined = Σ w[l] * y_level[l]`
2. `1/sqrt(k)` normalization removed from both forward and backward
3. Backward computes `d_alpha_mem` via softmax Jacobian (matches mac.rs:904-915 pattern)
4. Per-level backward receives `w[l] * d_y_combined` (not uniform `d_y_combined`)
5. Optimizer updates alpha_mem via AdamW with moment buffers
6. CPU traced path mirrors GPU path (traced_scale per level)
7. `d_alpha_mem` is nonzero after one backward pass (no longer a dead parameter)
8. At init (alpha_mem=zeros), weights are uniform 1/k — mathematically equivalent to mean
9. No regressions in single-block or stacked tests
10. alpha_refl remains unwired (MAC-only, not used in MAG stacked path)

## Dependencies

- BUG-01 (spec 18, PR #188, merged): W_O must be wired — attn_proj feeds into gating
- BUG-02 (spec 20, PR #189, merged): MAG sigmoid gating must be wired — d_y_combined
  flows from gating backward through alpha aggregation backward to per-level backward

## Ontological Compliance

- **CS-18**: Softmax and dot products are math in the Rust tier, not orchestration.
- **CS-10**: No mode flag — alpha aggregation is applied identically in all phases.
- **CS-32**: Observe-then-advance — aggregation is a stateless weighted sum.
- **CS-40**: Opt-in AD — traced_scale records weights only when tape active.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-074-arch-variant5 | hope_equations | HOPE §5.1 | implements |
