# Segment-Scoped Wengert Tape

```text
CONTRACT
Purpose:    Decouple the Wengert tape's intermediate storage from raw token count
            and scope it to CMS chunk boundaries (segments). This makes tape memory
            O(segments × d²) instead of O(seq_len × d²), enabling deep multi-block
            models to train and self-modify during inference without exhausting GPU
            memory. The tape measures the same quantities regardless of whether the
            model is streaming training data or running in a chat window.

Expects:    Existing Wengert tape (core/src/tape.rs) that records per-token
            intermediates during forward and replays in reverse for backward.
            CMS Conductor/Pulse system that tracks chunk boundaries per level.
            GPU stacked forward/backward paths that store per-timestep M states
            for the backward pass.

Guarantees: 1. Tape intermediate storage is proportional to the number of CMS
               segment boundaries, not the number of tokens processed.
            2. For a k-level CMS with chunk_sizes [c_0, c_1, ..., c_{k-1}] and
               sequence length S, the segment count is:
               N_segments = Σ_{l=0}^{k-1} ⌊S / c_l⌋
               For k=4 [1,8,64,512] S=512: N = 512+64+8+1 = 585
               For k=8 [1,8,64,512,4096,...] S=4096: N dominates at L0 = 4096
            3. Gradient correctness is preserved — the chain rule across segment
               boundaries produces identical gradients to the per-token tape for
               all outer-loop parameters.
            4. The tape structure is identical during training and inference. A model
               self-modifying in a chat window uses the same segment-scoped tape as
               during a build run. The observability metrics (gnorms, memory norms,
               gate values) are measured per-segment in both contexts.
            5. Memory footprint for the backward pass scales with model depth
               (n_blocks) and CMS level count (k), NOT with input length.

Cost:       Requires restructuring the backward pass to operate at chunk-boundary
            granularity rather than per-token granularity. Within-chunk token
            processing uses a rolling state that overwrites rather than accumulates.
            Inter-chunk state is checkpointed for backward replay.

Trade-off:  Within-chunk gradients for M are accumulated at the chunk boundary
            rather than stored per-token. This is mathematically equivalent for
            the Titans memory update (which applies at chunk boundaries anyway)
            but changes the intermediate representation. The per-token M trajectory
            within a chunk is not stored — only the boundary states and the
            accumulated gradient contribution.

Position:   specs/infrastructure/25_segment_scoped_tape.md

Source:     TNT (2511.07343) — chunkwise training approximation. Memory updates
            occur at chunk boundaries. Within-chunk processing accumulates error
            signals that are applied at the boundary.
            HOPE (2512.24695) eq-074 — CMS level independence. Each level's tape
            storage is scoped to its own chunk boundaries.
            CS-40 — Opt-in AD. The tape is activated explicitly; segment scoping
            does not change the opt-in contract.
            CS-42 — Gradient checkpointing hurts NL. This is NOT gradient
            checkpointing. Segment scoping stores boundary states, not recomputing
            them. No recomputation, no statefulness hazard.
```

## Problem Statement

The current Wengert tape stores per-token intermediates for every timestep in the
sequence, for every block. At d=1024, 8 blocks, seq_len=512:

- Per-token M state: d × d × 4 bytes = 4MB
- Per-token across sequence: 512 × 4MB = 2GB per level per block
- With momentum S: 2× = 4GB per level per block
- 8 blocks × k=1: 32GB just for memory state history

This consumed 40.5GB for a 178.5M parameter model — a 225:1 ratio of tape storage
to model parameters. This ratio grows linearly with seq_len and n_blocks, making
it unsustainable for production-scale models.

More critically, this design creates an asymmetry between training and inference.
During inference (chat, serving), the model must self-modify (NL's core property —
CS-10, no train/eval distinction). If the tape is active during inference for
self-modification, the same O(seq_len × d²) memory applies to every conversation
turn. A multi-turn chat at seq_len=2048 would require 4× the memory of training.

## Key Insight: Memory Updates Happen at Chunk Boundaries

The Titans memory rule (and all MIRAS variants) updates M at chunk boundaries,
not at every token. Within a chunk, the model:
1. Processes tokens through attention (SWA)
2. Queries M for each token: y_t = M @ q_t (read-only)
3. Accumulates error signals for the chunk
4. Applies the memory update at the chunk boundary: M_{t+C} = M_t + ΔM

The backward pass needs to differentiate through steps 1-4. But it does NOT need
the per-token M state for step 2 — it only needs the M state at the chunk boundary
(step 4) and the accumulated error (step 3).

## Design: Segment-Scoped Storage

### What is a Segment?

A **segment** is the interval between two consecutive chunk boundaries for a given
CMS level. Level l with chunk_size c_l produces ⌊S/c_l⌋ segments per sequence.

### What the Tape Stores Per Segment

For each segment boundary (per level, per block):

1. **M_boundary**: The memory matrix M at the chunk boundary [d × d]
2. **S_boundary**: The momentum matrix S at the chunk boundary [d × d] (if momentum enabled)
3. **error_accum**: The accumulated error signal for the chunk [d × d or d, depending on rule]
4. **gate_values**: alpha and theta gate outputs at the boundary [2 scalars]
5. **query_cache**: The queries q_t within the chunk [chunk_size × d] — needed for
   the M update backward, but bounded by chunk_size (not seq_len)

### What the Tape Does NOT Store

- Per-token M states within a chunk (overwritten during forward, not checkpointed)
- Per-token attention intermediates beyond the SWA window
- Duplicate copies of the embedding or LayerNorm intermediates

### Memory Budget

For k=4 [1,8,64,512], seq_len=512, d=1024, n_blocks=8:

**Current (per-token)**:
- M states: 512 × 1024² × 4 bytes × 8 blocks × 1 level = 17.2 GB (k=1)

**Proposed (per-segment)**:
- L0 (chunk=1): 512 boundaries × (M + S + error + queries) per block
  - M: 512 × 1024² × 4 = 2.1 GB — same as per-token for L0 (chunk=1 means every token IS a boundary)
- L1 (chunk=8): 64 boundaries × (M + S + 8×d queries) per block
  - M: 64 × 1024² × 4 = 268 MB per block
- L2 (chunk=64): 8 boundaries — 33.5 MB per block
- L3 (chunk=512): 1 boundary — 4.2 MB per block

**Key observation**: L0 at chunk=1 gets no benefit — every token is a boundary.
The savings come entirely from L1+ levels. For k=1 models, this change has no
effect. The benefit scales with k and with higher chunk sizes.

**For the 8-block d=1024 model that OOM'd**:
- k=1 (chunk=1): No savings — same as current. The OOM was caused by L0's
  per-token storage being identical to per-segment storage.
- k=4: L1-L3 savings would reclaim ~80% of their tape storage.

**Implication**: The real fix for the d=1024 OOM requires BOTH segment scoping
(for k≥2) AND a within-chunk memory optimization for L0 (chunk=1). L0's within-chunk
processing can use a rolling M state that doesn't store history — the backward pass
for L0 can recompute M_t from M_{t-1} since chunk_size=1 means each "chunk" is a
single step with a simple backward.

## Interaction with Training vs Inference

### Unified Tape Contract

The segment-scoped tape operates identically in both contexts:

| Aspect | Training (build) | Inference (chat/serve) |
|--------|-----------------|----------------------|
| Tape active? | Yes (with_tape) | Yes (self-modification) |
| Storage scope | Per-segment boundaries | Per-segment boundaries |
| Backward pass | Full — compute gradients | Partial — update M only |
| Observability | gnorms, gate values, M norms per segment | Same metrics, same granularity |
| Memory budget | Fixed per sequence | Fixed per sequence |

The tape measures the same quantities in both contexts. A monitoring dashboard
showing CMS level activity during training shows the same metrics during inference.
This is critical for alignment monitoring — the observability doesn't change when
the model goes to production.

### Inference-Specific Optimization

During inference, the backward pass only needs to update M (the inner loop), not
compute gradients for outer-loop parameters (W_K, W_V, W_Q, W_O, gates). This
means the tape can skip storing intermediates needed only for outer-loop gradient
computation, further reducing memory. A `TapeMode::InnerLoopOnly` flag would
enable this without changing the segment-scoped structure.

## Implementation Phases

### Phase 1: Segment Boundary Extraction (Python tier)
- Modify the Python build loop to pass segment boundary indices to the Rust tier
- The Conductor already knows when each level fires — expose this as a boundary map
- No Rust changes — just plumbing

### Phase 2: GPU Backward Restructure (Rust/CUDA tier)
- Restructure `gpu_stacked_backward.rs` to iterate over segment boundaries
  instead of per-token
- Within-chunk backward: recompute M states from boundary M (for chunk_size=1,
  this is trivial — no recomputation needed since each chunk IS one token)
- Cross-chunk backward: standard chain rule across boundary states
- Cache structure changes: `GpuStackedBlockCache` stores boundary states, not
  per-token states

### Phase 3: L0 Rolling State Optimization (Rust/CUDA tier)
- For chunk_size=1 (L0), the "segment" is every token — no storage savings from
  segment scoping alone
- Implement a rolling state for L0: M_t overwrites M_{t-1}, backward recomputes
  from M_{t-1} = M_t - ΔM (invertible for simple update rules)
- This is level-specific optimization, not segment scoping per se, but addresses
  the dominant memory consumer

### Phase 4: Tape Observability Unification
- Ensure tape summary metrics (gnorms, gate values, M norms) are computed at
  segment boundaries in both training and inference
- Python-tier `print_tape_summary` reads the same data structure regardless of
  context
- Monitoring hooks emit per-segment metrics for external dashboards

## Files to Modify

| File | Change |
|------|--------|
| `core/src/tape.rs` | Add `SegmentBoundary` struct, modify `TapeArena` to store per-segment |
| `core/src/gpu_stacked_forward.rs` | Emit boundary states to cache instead of per-token states |
| `core/src/gpu_stacked_backward.rs` | Iterate over segment boundaries, not tokens |
| `core/src/tape_summary.rs` | Compute summary metrics at segment granularity |
| `python/src/lib.rs` | Update `cpu_stacked_tape_summary` to work with segment-scoped data |
| `python/engine/loop.py` | Pass boundary maps from Conductor to Rust tier |

## Acceptance Criteria

1. Tape storage is O(N_segments × d²) not O(seq_len × d²) for levels with chunk_size > 1
2. Gradient correctness: FD gradient check passes at same tolerances as current tape
3. Training and inference use the same tape structure and produce the same observability metrics
4. 8-block d=1024 k=4 model fits on A6000 (49GB) — currently OOMs at k=1
5. `print_tape_summary` output is identical in format (per-level gnorms, gate values)
6. No regression in existing 4-block d=512 training performance

## Ontological Compliance

- **CS-10**: No mode flag. The tape is the same structure in training and inference. No `if training:` branch.
- **CS-18**: Segment boundary logic is math (Rust tier). Python tier only passes boundary indices.
- **CS-32**: Observe-then-advance. Boundary states are observed (stored) before the next chunk advances.
- **CS-40**: Opt-in AD. Segment scoping doesn't change the opt-in contract — `with_tape()` still activates recording.
- **CS-42**: NOT gradient checkpointing. Boundary states are stored, not recomputed. No statefulness hazard.
- **CS-48**: Per-level retention parameters remain independent. Segment boundaries are per-level by definition.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| TNT chunkwise approximation | tnt_equations | TNT §3 (2511.07343) | implements |
| eq-074 CMS independence | hope_equations | HOPE §5.1 (2512.24695) | cites |
