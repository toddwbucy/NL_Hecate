# BUG-01: Wire W_O Output Projection in Stacked Multi-Block Forward/Backward

```
CONTRACT
  Purpose:    Add the missing W_O output projection to the stacked multi-block
              GPU forward and backward paths, matching the single-block path.
  Expects:    Existing stacked forward (gpu_stacked_forward.rs) producing raw
              attn_out into the residual stream. W_O allocated in BlockParams
              and GpuBlockParams but unused. Single-block path (gpu_forward.rs)
              correctly applies W_O via cublas_matmul_transb_dd.
  Guarantees: After fix, each block computes attn_proj = attn_out @ W_O^T and
              uses attn_proj (not raw attn_out) in residual skip 1. Backward
              computes d_w_o and d_attn_out correctly. CPU traced path mirrors
              GPU path. W_O is no longer a dead parameter.
  Cost:       One d×d matmul per block per forward pass (negligible vs SWA).
              One d×d matmul + one outer product per block per backward pass.
              No new GPU buffers — reuses existing attn_out buffer or adds one
              attn_proj buffer per block in cache.
  Trade-off:  Adds compute (N × d² FLOPs) but removes the unbounded residual
              growth that causes NaN at ~460 steps in 4-block builds.
  Position:   specs/infrastructure/18_stacked_w_o_output_projection.md
  Source:     Standard multi-head attention output projection (Vaswani et al.
              2017). W_O does not appear as a numbered equation in any of the 7
              NL papers (Titans, MIRAS, HOPE, Lattice, Atlas, TNT, Trellis).
              Titans Section 4.4 detail (5) mentions "gating with normalization
              and linear layer before final output projection (Mehta et al.
              2023)" but that refers to the MEMORY MODULE's output pipeline,
              not SWA's W_O — it applies to all variants including LMM (pure
              memory, no attention). SWA's W_O is assumed transformer
              infrastructure. The single-block path (gpu_forward.rs:710-711)
              already implements it correctly.
```

## Bug Description

The stacked multi-block forward pass (`gpu_stacked_forward.rs:239-241`) omits
the W_O output projection that the single-block GPU path applies at
`gpu_forward.rs:710-711`. W_O is allocated in `BlockParams` (line 25 of
`stacked_model.rs`), the optimizer applies weight decay to it, but it receives
zero gradient — a dead parameter.

## Paper Source

W_O does not appear as a numbered equation in any of the 7 NL papers (Titans,
MIRAS, HOPE, Lattice, Atlas, TNT, Trellis). All 7 paper equation collections
were searched — zero hits for W_O, output projection, or W_{O}.

**Titans Section 4.4 detail (5)** mentions *"gating with normalization and
linear layer before final output projection (Mehta et al. 2023)"*, but this
describes the **memory module's** output pipeline — it is listed as a detail
*"shared across all Titans variants"*, including LMM (pure memory, no
attention). This is NOT SWA's W_O.

SWA's W_O is standard multi-head attention infrastructure (Vaswani et al. 2017,
"Attention Is All You Need"). The NL papers assume the reader knows how
multi-head attention works and do not re-derive it. Our single-block GPU path
correctly implements W_O; the stacked path omits it.

## Why It Matters

Without W_O, raw SWA attention output enters the residual stream unregulated.
In a 4-block model, each block adds unscaled attn_out to the residual. The
residual magnitude grows without a learned projection to control it. This is the
primary contributor to NaN at ~460 steps in full-trajectory stacked builds.

W_O provides:
1. **Scale regulation** — a learned d×d projection that can attenuate or rotate
   the attention output before it enters the residual stream.
2. **Head mixing** — in multi-head attention, W_O recombines head outputs into
   the model dimension. Without it, heads cannot mix information.
3. **Gradient signal** — d_w_o is currently all-zeros, meaning AdamW applies
   weight decay to W_O (pulling it toward zero) but receives no learning signal.
   This creates a consistent but meaningless loss contribution.

## Single-Block Path (reference implementation)

`gpu_forward.rs:710-711` (residual path):
```rust
// Output projection on residual
cublas_matmul_transb_dd(&res_final, &params.swa.w_o, &mut projected, bs * s, d, d, 0.0);
```

`gpu_forward.rs:725` (legacy sigmoid gating path):
```rust
cublas_matmul_transb_dd(&gated_out, &params.swa.w_o, &mut projected, bs * s, d, d, 0.0);
```

Both paths apply `projected = input @ W_O^T` as the final step before unembed.

## Stacked Path (current — broken)

`gpu_stacked_forward.rs:239-241`:
```rust
// NOTE: w_o (output projection) is not applied after attention in the
// initial stacked architecture. The residual stream carries raw attn_out.
// When w_o is wired in, add: attn_proj = attn_out @ w_o, use attn_proj below.
```

`gpu_stacked_backward.rs:327-329`:
```rust
// Note: d_w_o not used in stacked model — output projection removed.
// Each block's output goes directly to residual stream.
// d_w_o stays zeros.
```

`traced_forward.rs:1395`:
```rust
w_o: None, // not applied in stacked path (see gpu_stacked_forward.rs:239-241)
```

## Fix Specification

### Forward (gpu_stacked_forward.rs)

After SWA produces `attn_out` and before residual skip 1, insert:

```rust
// Output projection: attn_proj = attn_out @ W_O^T
let mut attn_proj = GpuBuf::<f32>::zeros(total);
cublas_matmul_transb_dd(&attn_out, &block.w_o, &mut attn_proj, bs * s, d, d, 0.0);
```

Then use `attn_proj` (not `attn_out`) in the residual:

```rust
// Residual skip 1: residual_after_attn = block_input + attn_proj
saxpy_cuda(1.0, block_input.as_ptr(), residual_after_attn.ptr(), total_i32);
saxpy_cuda(1.0, attn_proj.as_ptr(), residual_after_attn.ptr(), total_i32);
```

### Backward (gpu_stacked_backward.rs)

In the backward pass, at the point where d_attn flows back through W_O:

```rust
// d_attn_out = d_residual_skip1 @ W_O  (gradient through projection)
cublas_matmul_dd(&d_residual_skip1, &block.w_o, &mut d_attn_out, bs * s, d, d, 0.0);

// d_w_o += attn_out^T @ d_residual_skip1  (gradient for W_O)
gpu_matmul_transa_dd(&d_residual_skip1, &attn_out, &mut d_w_o, d, bs * s, d);
```

Note: `d_w_o` accumulates with `beta=1.0` — each block contributes its own
gradient to its own W_O.

### Cache (GpuBlockCache)

Add `attn_proj` to `GpuBlockCache` if `attn_out` is needed separately in the
backward pass. If BUG-02 (MAG gating) will also need `attn_proj`, storing it in
the cache avoids recomputation:

```rust
pub struct GpuBlockCache {
    // ... existing fields ...
    pub attn_proj: GpuBuf<f32>,  // attn_out @ W_O^T — needed by backward + BUG-02
}
```

### CPU Traced Path (traced_forward.rs)

In `traced_stacked_forward()`, register W_O and apply the projection on the tape:

```rust
let w_o_id = tape.register_param(&block.w_o, vec![d, d]);
let attn_proj_id = traced_matmul_transb(tape, attn_out_id, w_o_id, bs * s, d, d);
// Use attn_proj_id (not attn_out_id) in residual skip 1
```

Change `TracedBlockParamIds::w_o` from `None` to `Some(w_o_id)`.

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | Add W_O matmul after SWA, before residual skip 1 |
| `core/src/gpu_stacked_backward.rs` | Add W_O backward (d_attn_out, d_w_o gradients) |
| `core/src/traced_forward.rs` | Register W_O, apply matmul in traced_stacked_forward |
| `core/src/gpu_stacked_forward.rs` | Add attn_proj to GpuBlockCache |

## Acceptance Criteria

1. W_O projection applied in stacked forward: `attn_proj = attn_out @ W_O^T` per block
2. Backward computes `d_w_o` and `d_attn_out` correctly (FD gradient check passes)
3. CPU traced path mirrors GPU path exactly (tape records W_O matmul)
4. `d_w_o` is nonzero after one backward pass (no longer a dead parameter)
5. Stacked build runs 1000+ steps without NaN on dolmino_smoke (regression test)
6. Single-block behavior unchanged (no regressions in existing tests)

## Dependencies

None — this is independent of BUG-02 (MAG gating) and BUG-03 (alpha aggregation).

## Interaction with BUG-02

BUG-02 (MAG sigmoid gating) will change the residual structure from two additive
skips to `residual_out = block_input + (attn_proj * sigmoid(y_combined))`. That
fix operates on `attn_proj` (the W_O output from this bug fix), not raw
`attn_out`. This bug must be fixed first.

## Ontological Compliance

- **CS-18**: The W_O matmul is math in the Rust tier, not orchestration. No
  change to Python tier.
- **CS-10**: No mode flag — W_O is applied identically in all phases.
- **CS-32**: Observe-then-advance — W_O is a stateless projection applied to the
  observation (attn_out), no ordering dependency with memory update.
- **CS-40**: Opt-in AD — W_O is recorded on the Wengert tape only when tape is
  active (traced path). GPU path uses hand-written backward.
