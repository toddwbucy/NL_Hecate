# GPU Batch Context Continuity

```text
CONTRACT
  Purpose:    Specify the semantics of CMS context carry-forward (inner_loop_state M)
              when batch_size > 1, and define the per-slot context architecture that
              gives every batch element a dense, gapless sequential M stream.

  Expects:    - batch_size is a build config parameter (>= 1)
              - GpuContextState exists and holds per-level M on GPU
              - CUDA kernels (delta_forward, titans_forward) accept m_initial[d*d]
              - BpeTokenStream serves chunks via a single position cursor

  Guarantees: - With batch_size=B and seq_len=S, every token in the corpus flows
                through SOME batch slot's M update (no token is skipped)
              - Each batch slot b has its own sequential M stream through sub-corpus b
              - Slot b sees tokens at corpus positions: {b*S, b*S+B*S, b*S+2*B*S, ...}
                (strided by B*S per step, dense within each sub-corpus of size 1/B)
              - batch_size=1 is identical to the prior single-stream behavior
              - Checkpoint/restore: each slot's cursor and M state is independently
                serializable (deferred to a future checkpoint spec)

  Cost:       - GpuContextState VRAM: B * k * d*d * 4 bytes (e.g., B=8, k=4, d=512 = 32 MB)
              - Forward kernel: no extra FLOPs; m_initial offset read is O(1)
              - copy_final_m: B D2D memcpy calls per level (each d*d floats, ~1 MB)
              - loop.py: B separate BpeTokenStream instances (same corpus mmap'd once)
              - No change to backward kernels (d_m_initial is a state gradient,
                not a weight gradient, and is not consumed by AdamW)

  Trade-off:  Sub-corpus coverage: each slot sees 1/B of the total corpus in its
              sequential M stream. For B=8, slot 0 sees 12.8M tokens at 25K steps
              (vs 102.4M total corpus-equivalent tokens consumed for gradient).
              Cross-document associations that span the 1/B corpus boundary of a
              given slot are missed by that slot's M. This is an inherent constraint
              of batch-parallel M updates — removed only by sequential M computation,
              which defeats the throughput benefit of batching.

  Position:   specs/infrastructure/06_batch_context_continuity.md
  Source:     HOPE (2512.24695) §3 CMS inner_loop_state semantics
                HADES: hope_equations/eq-088-practical-dgd-update (DGD update rule)
              Titans (2501.00663) §3 Sequential memory update
                HADES: titans_equations/eq-034-deltanet-update (Delta Rule)
  Related:    specs/infrastructure/05_ablation_study.md (ablation prereq)
              core/src/gpu_params.rs (GpuContextState)
              core/src/gpu_forward.rs (broadcast_m_initial, copy_final_m)
              core/kernels/delta_forward.cu, titans_forward.cu
              python/engine/loop.py (batch chunk collection)
              python/engine/data.py (BpeTokenStream)
```

## Problem Statement

With `batch_size=B > 1`, the current implementation:

1. **Broadcasts** a single context M to all B batch elements at step start.
2. Runs B elements **in parallel** through the forward kernel (each starting from M_carry).
3. **Copies only element-0's** final M back as M_carry for the next step.

This creates a **(B-1)*S token gap** in M's sequential context stream per step:
- Element-0 processes tokens `[t*B*S .. t*B*S + S - 1]`
- Elements 1..B-1 process tokens `[t*B*S + S .. t*B*S + B*S - 1]` (these tokens update weights but not M)
- M_carry for step t+1 = M after processing tokens `[t*B*S .. t*B*S + S - 1]` only

At B=8, S=512: M sees 512 of 4096 tokens per step (12.5% M coverage). M cannot accumulate associations across chunk boundaries missed by element-0.

## Fix: Per-Slot Context States

Assign each batch slot b its own sequential M stream through sub-corpus b.

### Sub-corpus layout

Total corpus: N tokens. Sub-corpus b = tokens at positions {b*S, b*S + B*S, b*S + 2*B*S, ...}.

At step t:
- Slot b processes token range `[b*S + t*B*S .. b*S + t*B*S + S - 1]`
- M_b carry for step t+1 = slot b's final M from step t

Each slot sees a **dense sequential stream** through 1/B of the corpus (stride = B*S, but within each sub-corpus position the stream is contiguous from one step to the next). Cross-document associations are preserved within each sub-corpus.

### Changes

#### `core/src/gpu_params.rs` — GpuContextState

Add `batch_size: usize` field. Change `memory: Vec<GpuBuf<f32>>` from:
- k elements, each `[d*d]` (one M per level)

To:
- k elements, each `[batch_size * d*d]` (one M per batch slot per level)

```rust
pub struct GpuContextState {
    pub memory: Vec<GpuBuf<f32>>,  // k bufs, each [batch_size * d*d]
    pub d: usize,
    pub batch_size: usize,
}

impl GpuContextState {
    pub fn new(k: usize, d: usize, batch_size: usize) -> Self {
        let memory = (0..k).map(|_| GpuBuf::zeros(batch_size * d * d)).collect();
        GpuContextState { memory, d, batch_size }
    }
    // reset(): zero all batch_size * d*d floats per level
    // to_host(): export slot-0's M (primary context for checkpoint/inference)
    // from_host_context(): broadcast host M to all batch_size slots
}
```

#### `core/kernels/delta_forward.cu` and `titans_forward.cu` — Per-batch initial M

**Before**: `const float* m_initial` is `[d*d]`, same for all batch elements (broadcast).

**After**: `const float* m_initial` is `[batch_size * d*d]`. Element `b` reads from:
```c
const float* m_b = m_initial + b * d * d;
// Use m_b instead of m_initial for initializing m_prev in kernel
```

This is the **only** kernel change: a pointer offset in the initialization code. No grid/block changes needed.

#### `core/src/gpu_forward.rs` — Remove broadcast, fix copy_final_m

**Remove `broadcast_m_initial`**: instead of creating a new `[batch_size * d*d]` buffer by broadcasting,
pass `context_m[level]` directly to the kernel as `m_initial[batch_size * d*d]`.

**Update `copy_final_m`** for all batch slots:
```rust
fn copy_final_m_batch(m_states: &GpuBuf<f32>, context_m: &mut GpuBuf<f32>,
                      seq_len: usize, dd: usize, batch_size: usize) {
    for b in 0..batch_size {
        // Element b's final M in m_states: offset = b*(seq_len+1)*dd + seq_len*dd
        let src_offset = (b * (seq_len + 1) + seq_len) * dd;
        let dst_offset = b * dd;
        // D2D memcpy: m_states[src_offset..] → context_m[dst_offset..]
        unsafe {
            cuda_memcpy_d2d(
                context_m.ptr().add(dst_offset),
                m_states.ptr().add(src_offset),
                dd * std::mem::size_of::<f32>(),
            );
        }
    }
}
```

#### `python/src/lib.rs` — GpuModel init

`GpuContextState::new` now takes `batch_size`. Pass it from `cfg.batch_size` (the model's batch_size field, mirroring `BuildConfig.batch_size`). This requires either:
- Adding `batch_size` to `MAGConfig` (cleanest), or
- Passing it separately to `GpuModel::new`

Recommended: keep `batch_size` in `BuildConfig` (build time, not model arch), and pass it to `GpuModel::new` explicitly.

#### `python/engine/loop.py` — Per-slot BpeTokenStream

**Before** (batch_size > 1):
```python
for _ in range(bcfg.batch_size):
    chunk = bpe_loader.next_chunk(bcfg.seq_len)
    all_input.extend(chunk[0])
    all_target.extend(chunk[1])
```
This yields B consecutive chunks from ONE cursor (positions t*B*S .. (t+1)*B*S).

**After**:
```python
# Initialization: create B loaders at B different starting positions
bpe_loaders = []
slot_size = bpe_loader.total_tokens // bcfg.batch_size
for b in range(bcfg.batch_size):
    loader_b = BpeTokenStream(bcfg.data_path, split="train")
    loader_b.position = b * slot_size  # start of sub-corpus b
    bpe_loaders.append(loader_b)

# At each step: yield chunk b from loader b
all_input, all_target = [], []
for b, loader_b in enumerate(bpe_loaders):
    chunk = loader_b.next_chunk(bcfg.seq_len)
    if chunk is None:
        break
    all_input.extend(chunk[0])
    all_target.extend(chunk[1])
```

Slot b's initial position: `b * (total_tokens // batch_size)`. The primary slot (b=0) starts at 0 and processes the first 1/B of the corpus sequentially. Slot 1 starts at 1/B of the corpus, etc.

**Checkpoint behavior**: save each loader_b's cursor separately. During restore, re-initialize each loader at its saved position. (Ablation runs don't resume from checkpoint, so cursor restore can be a stub for now.)

## Correctness vs Prior Behavior

| Property | Prior (batch_size>1) | After fix |
|---|---|---|
| M token coverage | 1/B of corpus | 100% (distributed across B sub-corpora) |
| M sequential gaps | (B-1)*S tokens per step | 0 within each sub-corpus |
| Gradient coverage | 100% (all B chunks contribute) | 100% (unchanged) |
| VRAM (context state) | k*d*d*4 bytes | B*k*d*d*4 bytes (8x for B=8) |
| Kernel changes | None | m_initial pointer offset only |
| Backward changes | None | None (d_m_initial unused by AdamW) |

## Inference / Checkpoint Semantics

At checkpoint and inference time, **slot 0** (b=0) is the primary context. This matches the prior single-stream semantics for the checkpoint format: `GpuContextState::to_host()` exports slot-0's M only. Loading a checkpoint initializes slot-0 from the checkpoint and broadcasts to all other slots.

For evaluation (non-batch forward), only slot-0's M is used. This is consistent with the NL papers' single-context-stream model.

## Acceptance Criteria

1. `GpuContextState::new(k=1, d=32, batch_size=4)` allocates 4 * 32*32 * 4 = 65,536 bytes per level.
2. After one forward step at batch_size=4, slots 0..3 have distinct M states (they started at distinct sub-corpus positions).
3. At batch_size=1, behavior is bit-identical to the prior implementation.
4. `verify_grad_scaling.py` still passes (gradient normalization is unchanged).
5. Ablation A still runs correctly with batch_size=8 (memory_enabled=False path unchanged).
