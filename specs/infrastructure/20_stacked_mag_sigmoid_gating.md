# BUG-02: Wire MAG Sigmoid Gating in Stacked Multi-Block Forward/Backward

```
CONTRACT
  Purpose:    Add the missing MAG sigmoid gating to the stacked multi-block
              GPU forward and backward paths, replacing the broken two-additive-
              skip residual structure with memory-gated attention.
  Expects:    BUG-01 (W_O output projection) resolved — sigmoid gating operates
              on attn_proj (W_O output), not raw attn_out. Existing CUDA kernels
              sigmoid_cuda, elemwise_mul_cuda, gating_backward_cuda, and
              sigmoid_backward_cuda available in core/kernels/elementwise.cu.
  Guarantees: After fix, each block computes:
                gate = sigmoid(y_combined)
                gated_out = attn_proj * gate
                residual_out = block_input + gated_out
              Backward computes d_attn_proj, d_gate, d_y_combined correctly.
              CPU traced path mirrors GPU path. Memory output is bounded to [0,1]
              via sigmoid, preventing unbounded residual growth.
  Cost:       Two elementwise ops per block per forward (sigmoid + mul), negligible
              vs SWA and memory matmuls. Three elementwise ops per block per backward
              (gating_backward + sigmoid_backward). Two new GPU buffers per block in
              cache (gate, attn_proj — both [bs*s, d]).
  Trade-off:  Replaces unbounded additive memory with bounded [0,1] multiplicative
              gating. Memory can now suppress attention (gate≈0) or pass it through
              (gate≈1), but cannot amplify it beyond 1x. This matches the MAG paper
              equation and prevents the NaN from unbounded residual growth.
  Position:   specs/infrastructure/20_stacked_mag_sigmoid_gating.md
  Source:     Titans (2501.00663) Section 3.3, eq-028 (MAG composition):
              o = y ⊙ σ(M(x̃)) — attention output gated by sigmoid of memory.
              Single-block GPU path (gpu_forward.rs:720-723) implements this
              correctly in the legacy (non-residual) path.
```

## Bug Description

The stacked multi-block forward pass (`gpu_stacked_forward.rs:323-328`) adds
raw CMS memory output (`y_combined`) to the residual stream additively via two
separate residual skips. The Titans MAG composition (eq-028) specifies
`o = y ⊙ σ(M(x̃))` — attention output element-wise multiplied by sigmoid of
memory output. Without the sigmoid bound, memory magnitudes are unbounded.

## Paper Source

Titans (2501.00663) Section 3.3, equation collection `titans_equations/eq-028`:
```
o = y ⊙ σ(M(x̃))
```
Where:
- `y` is the attention output (after W_O projection: `attn_proj`)
- `M(x̃)` is the memory output (`y_combined` from CMS levels)
- `σ` is the sigmoid function (bounds output to [0,1])
- `⊙` is elementwise multiplication

The single-block GPU path (`gpu_forward.rs:720-723`) implements this correctly:
```rust
sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), total_i32);
elemwise_mul_cuda(attn_out.as_ptr(), gate.as_ptr(), gated_out.ptr(), total_i32);
```

## Why It Matters

Without sigmoid gating, each M matrix (clamped to ‖M‖_F ≤ 100) can produce
‖y_level‖ ≤ 100. Sum 4 levels → up to 400. Scale by 1/√4 → 200 per block.
Sum 4 blocks → up to 800 added to the residual. The final LN input becomes
enormous. With sigmoid, each gate element is bounded to [0,1], so the maximum
contribution per block is ‖attn_proj‖ (already regulated by W_O from BUG-01).

## Current Code (broken)

`gpu_stacked_forward.rs:323-328`:
```rust
// Residual skip 2: residual_stream = residual_after_attn + y_combined
let mut new_residual = GpuBuf::<f32>::zeros(total);
unsafe {
    saxpy_cuda(1.0, residual_after_attn.as_ptr(), new_residual.ptr(), total_i32);
    saxpy_cuda(1.0, y_combined.as_ptr(), new_residual.ptr(), total_i32);
}
```

Two additive skips: `residual_out = block_input + attn_proj + y_combined`.
No sigmoid, no gating, unbounded memory magnitude.

## Fix Specification

### Forward (gpu_stacked_forward.rs)

Replace residual skip 2 (lines 323-328) with MAG sigmoid gating:

```rust
// ── MAG sigmoid gating: gate = σ(y_combined), gated_out = attn_proj * gate ──
let mut gate = GpuBuf::<f32>::zeros(total);
let mut gated_out = GpuBuf::<f32>::zeros(total);
unsafe {
    sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), total_i32);
    elemwise_mul_cuda(attn_proj.as_ptr(), gate.as_ptr(), gated_out.ptr(), total_i32);
}

// ── Residual: residual_out = block_input + gated_out ──
let mut new_residual = GpuBuf::<f32>::zeros(total);
unsafe {
    saxpy_cuda(1.0, block_input.as_ptr(), new_residual.ptr(), total_i32);
    saxpy_cuda(1.0, gated_out.as_ptr(), new_residual.ptr(), total_i32);
}
```

### Cache (GpuStackedBlockCache)

Add `gate` and `attn_proj` to the cache for backward:

```rust
pub struct GpuStackedBlockCache {
    // ... existing fields ...
    pub attn_proj: GpuBuf<f32>,  // [bs*s, d] — attn_out @ W_O^T, needed for gating backward
    pub gate: GpuBuf<f32>,       // [bs*s, d] — sigmoid(y_combined), needed for backward
}
```

### Backward (gpu_stacked_backward.rs)

Replace the additive skip 2 backward with gating backward:

```rust
// ── Output residual skip backward: residual_out = block_input + gated_out ──
// d_gated_out = d_residual_stream
// d_block_input accumulates d_residual_stream (handled at end)

// ── Gating backward: gated_out = attn_proj * gate ──
let mut d_attn_proj_from_gate = GpuBuf::zeros(bsd);
let mut d_gate = GpuBuf::zeros(bsd);
unsafe {
    gating_backward_cuda(
        d_residual_stream.as_ptr(), bc.attn_proj.as_ptr(), bc.gate.as_ptr(),
        d_attn_proj_from_gate.ptr(), d_gate.ptr(), bsd_i32,
    );
}

// ── Sigmoid backward: gate = sigmoid(y_combined) ──
let mut d_y_combined = GpuBuf::zeros(bsd);
unsafe {
    sigmoid_backward_cuda(
        d_gate.as_ptr(), bc.gate.as_ptr(), d_y_combined.ptr(), bsd_i32,
    );
}
```

Then `d_y_combined` feeds into per-level memory backward (unchanged).

After LN_mem backward produces `d_residual_after_attn`, the attn_proj gradient
accumulates from both paths:

```rust
// d_attn_proj_total = d_attn_proj_from_gate + d_residual_after_attn
// (attn_proj is used in both gated_out and residual_after_attn)
```

### CPU Traced Path (traced_forward.rs)

Replace the additive residual skip 2 with traced sigmoid + mul:

```rust
let gate_id = traced_sigmoid(tape, combined_id);
let gated_out_id = traced_mul(tape, attn_proj_id, gate_id);
residual_id = traced_add(tape, residual_id, gated_out_id);
```

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | Replace additive skip 2 with sigmoid gating; add gate/attn_proj to cache |
| `core/src/gpu_stacked_backward.rs` | Replace additive skip 2 backward with gating+sigmoid backward |
| `core/src/traced_forward.rs` | Replace traced_add skip 2 with traced_sigmoid + traced_mul |

## Acceptance Criteria

1. Sigmoid gating applied per block: `gate = sigmoid(y_combined)`, `gated_out = attn_proj * gate`
2. Single residual skip: `residual_out = block_input + gated_out`
3. Backward computes d_gate, d_y_combined, d_attn_proj correctly
4. CPU traced path mirrors GPU path (traced_sigmoid + traced_mul)
5. No regressions in single-block or stacked tests
6. `attn_proj` (not raw `attn_out`) is the gated quantity

## Dependencies

- BUG-01 (spec 18, PR #188, merged): W_O must be wired — gating operates on `attn_proj`

## Ontological Compliance

- **CS-18**: Sigmoid/mul are math in the Rust tier, not orchestration.
- **CS-10**: No mode flag — gating is applied identically in all phases.
- **CS-32**: Observe-then-advance — gating is a stateless transform.
- **CS-40**: Opt-in AD — traced_sigmoid/traced_mul recorded only when tape active.
