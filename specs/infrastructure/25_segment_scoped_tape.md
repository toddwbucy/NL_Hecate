# Boundary-Scoped Wengert Tape with Tape Multiplier

```text
CONTRACT
Purpose:    Decouple the Wengert tape's intermediate storage from raw token count
            and tie it to the model's CMS architecture. Tape memory becomes a
            deterministic function of (d, k, chunk_sizes, n_blocks, tape_multiplier)
            — a constant readable from the model config at deploy time.
            The tape_multiplier (1x–8x) controls the observation density within
            chunks: 1x stores boundary checkpoints only, Nx stores N evenly-spaced
            checkpoints per chunk. Same tape structure in training and inference.

Expects:    Existing Wengert tape (core/src/tape.rs) with opt-in recording.
            CMS Conductor/Pulse system that tracks chunk boundaries per level.
            GPU stacked forward/backward with existing checkpointed variants
            (GpuMemoryCache::*Ckpt) that store M at intervals and recompute
            between checkpoints during backward.

Guarantees: 1. Tape memory is a deterministic function of model architecture and
               tape_multiplier — NOT of input length or content.
               tape_size = n_blocks × Σ_l(checkpoints_per_level(l)) × d² × sizeof(f32)
               where checkpoints_per_level(l) = ⌊S / (chunk_sizes[l] / multiplier)⌋ + 1
            2. At multiplier=1x, tape stores only chunk boundary states.
               At multiplier=Nx, N evenly-spaced checkpoints per chunk.
               At multiplier=chunk_size (level-specific max), equivalent to
               current full per-token storage.
            3. Gradient correctness is preserved — the existing checkpointed
               backward kernels handle recomputation from checkpoints.
            4. The tape structure is identical during training and inference.
               Observability metrics (gnorms, M norms, gate values) are computed
               at checkpoint positions in both contexts.
            5. The multiplier is a deployment config parameter, not a model
               parameter. Changing it does not change model behavior — only
               the density of trajectory observation.

Cost:       Higher-k levels with multiplier=1x use recomputation in backward
            (existing Ckpt kernel behavior). This trades compute for memory.
            At multiplier=max (full trajectory), no recomputation — same as
            current behavior.

Trade-off:  At multiplier=1x, the within-chunk M trajectory is not stored.
            This is the path M took between boundaries — useful for diagnosing
            oscillation, instability, or pathological L0 behavior. The multiplier
            knob lets operators choose their point on the memory/visibility curve:
            - 1x: Production default. Predictable memory. Boundary-only visibility.
            - 2x–4x: Monitored production. Midpoint visibility. Catch gross issues.
            - 8x+: Development/diagnostics. Near-full trajectory. Maximum insight.

Position:   specs/infrastructure/25_segment_scoped_tape.md

Source:     TNT (2511.07343) — chunkwise training approximation. Memory updates
            occur at chunk boundaries. Within-chunk processing accumulates error
            signals that are applied at the boundary.
            HOPE (2512.24695) eq-074 — CMS level independence. Each level's tape
            storage is scoped to its own chunk boundaries.
            CS-40 — Opt-in AD. The tape is activated explicitly; boundary scoping
            does not change the opt-in contract.
            CS-42 — Gradient checkpointing hurts NL. At multiplier=1x, boundary
            states ARE stored (not recomputed). Between-checkpoint recomputation
            is bounded and deterministic — not the unbounded recomputation that
            CS-42 warns against.
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

More critically, the tape size is a function of **input data** (seq_len), not
**model architecture**. This means:
- Memory budgets cannot be determined at deploy time from model config alone.
- Capacity planning for serving requires worst-case input assumptions.
- A multi-turn chat session's memory varies with message length.
- There is no fixed number to put on a model spec sheet.

Production requires predictable resource consumption derived from architecture,
not data. The rate of input is unstable by nature — the model's memory footprint
must not inherit that instability.

## Key Insight: Existing Ckpt Infrastructure

The codebase already contains checkpointed variants of every memory rule's GPU
cache (`GpuMemoryCache::DeltaCkpt`, `TitansCkpt`, `HebbianCkpt`, `DGDCkpt`).
These store `m_checkpoints` at a fixed interval instead of per-token `m_states`,
and the backward kernels (`titans_backward_checkpointed`, etc.) handle
recomputation from checkpoints.

The checkpoint interval is currently a global config field
(`MAGConfig::checkpoint_interval: Option<usize>`). Boundary-scoped tape is
achieved by setting this interval to `chunk_sizes[level]` per level — or a
fraction thereof controlled by the tape multiplier.

No new CUDA kernels required. No new backward logic. The infrastructure exists.

## Design: Tape Multiplier

### Base Unit

The **base checkpoint count** for level l with chunk_size c_l and sequence S is:

```
base_checkpoints(l) = ⌊S / c_l⌋ + 1
```

This is the boundary-only count (1x multiplier). For k=4 [1,8,64,512], S=512:
- L0: 513  (every token + initial)
- L1: 65   (every 8th + initial)
- L2: 9    (every 64th + initial)
- L3: 2    (start + end)

### Multiplier Effect

The `tape_multiplier` (integer, 1–max) controls intra-chunk checkpoint density:

```
effective_interval(l) = max(1, chunk_sizes[l] / tape_multiplier)
checkpoints(l) = ⌊S / effective_interval(l)⌋ + 1
```

| Multiplier | L0 (c=1) | L1 (c=8) | L2 (c=64) | L3 (c=512) | Total  |
|------------|----------|----------|-----------|------------|--------|
| 1x         | 513      | 65       | 9         | 2          | 589    |
| 2x         | 513      | 129      | 17        | 3          | 662    |
| 4x         | 513      | 257      | 33        | 5          | 808    |
| 8x (max L1)| 513      | 513      | 65        | 9          | 1100   |
| full       | 513      | 513      | 513       | 513        | 2052   |

L0 is unaffected by the multiplier — its chunk_size is 1, so every token is
already a boundary. The multiplier only increases density for L1+.

### Memory Budget

For d=512, n_blocks=4, k=4 (current running model):

| Multiplier | M checkpoints total | Memory (M only) | vs current |
|------------|--------------------:|----------------:|------------|
| 1x         | 589 × 4 = 2,356    | 2.5 GB          | 48% of full |
| 2x         | 662 × 4 = 2,648    | 2.8 GB          | 54%         |
| 4x         | 808 × 4 = 3,232    | 3.4 GB          | 62%         |
| full       | 2052 × 4 = 8,208   | 8.6 GB          | 100%        |

For d=1024, n_blocks=8, k=4 (the model that OOM'd at 40.5GB):

| Multiplier | M checkpoints total  | Memory (M only) | vs current |
|------------|---------------------:|----------------:|------------|
| 1x         | 589 × 8 = 4,712     | 19.7 GB         | 48%         |
| 2x         | 662 × 8 = 5,296     | 22.1 GB         | 54%         |
| full       | 2052 × 8 = 16,416   | 68.6 GB         | 100% (OOM) |

At 1x, the d=1024 model's M tape drops from ~40GB to ~20GB — within A6000 range.

### Tape Size Formula

Deterministic from model config:

```
tape_bytes(d, k, chunk_sizes, n_blocks, seq_len, multiplier) =
    n_blocks × Σ_{l=0}^{k-1} checkpoints(l, multiplier) × d² × 4

    where checkpoints(l, m) = ⌊seq_len / max(1, chunk_sizes[l] / m)⌋ + 1
```

This is a **model constant** for fixed seq_len. Print it on the model card.

### Deployment Profiles

| Environment          | Multiplier | Rationale                                    |
|----------------------|------------|----------------------------------------------|
| Production (serving) | 1x         | Minimum memory. Boundary visibility only.    |
| Monitored production | 2x–3x     | Midpoint visibility. Catch oscillation.      |
| Staging/eval         | 4x         | Good trajectory resolution for validation.   |
| Development/build    | 8x or full | Maximum visibility. Memory is available.     |

The multiplier never changes model behavior — only observation density.

## Implementation

### What Changes

The change is surgical: in `gpu_stacked_forward.rs`, before calling
`gpu_memory_forward()` for each level, compute the effective checkpoint interval
from `chunk_sizes[level]` and `tape_multiplier`. If the interval > 1, use the
existing Ckpt codepath. If interval == 1 (L0, or any level at max multiplier),
use the existing full-trajectory codepath.

### Per-Level Checkpoint Interval

```rust
// In gpu_stacked_forward, per-level memory forward:
let effective_interval = match tape_multiplier {
    0 | 1 => chunk_sizes[level],   // 1x: boundary only
    m     => max(1, chunk_sizes[level] / m),  // Nx: N checkpoints per chunk
};
let level_ckpt = if effective_interval > 1 {
    Some(effective_interval)
} else {
    None  // full trajectory (L0, or multiplier >= chunk_size)
};
```

### Config Addition

```rust
// In MAGConfig (model.rs):
/// Tape multiplier: controls intra-chunk checkpoint density.
/// 1 = boundary-only (minimum memory), N = N checkpoints per chunk.
/// 0 or None = full trajectory (current behavior, backward compatible).
#[serde(default)]
pub tape_multiplier: Option<usize>,
```

Python config:
```python
# In ModelConfig (config.py):
tape_multiplier: int | None = None  # None = full trajectory (current)
```

JSON config:
```json
{
    "model": {
        "tape_multiplier": 1
    }
}
```

### Files to Modify

**Phase 1 (this PR — GPU path):**

| File | Change |
|------|--------|
| `core/src/model.rs` | Add `tape_multiplier: Option<usize>` to MAGConfig |
| `core/src/gpu_forward.rs` | Compute per-level checkpoint interval, route to Ckpt path |
| `python/engine/config.py` | Add `tape_multiplier` to ModelConfig |
| `python/src/lib.rs` | Pass tape_multiplier through PyO3 |

**Phase 2 (future — GPU tape summary, eliminates CPU bottleneck):**

| File | Change |
|------|--------|
| `core/src/tape_summary.rs` | Compute gnorms/gate stats from GPU Ckpt cache directly |
| `core/src/traced_forward.rs` | Apply multiplier to CPU tape allocations (parity) |
| `core/src/tape.rs` | Multiplier-aware arena allocation for CPU path |
| `python/src/lib.rs` | GPU tape summary binding (replace `cpu_stacked_tape_summary`) |

### GPU vs CPU Tape Paths

The tape multiplier applies to both paths:

| Path | Current behavior | With multiplier |
|------|-----------------|-----------------||
| **GPU** (`gpu_stacked_forward/backward`) | Full `m_states[(s+1)*d*d]` per level | `Ckpt` variants with per-level interval |
| **CPU** (`traced_forward` → `tape.rs`) | Full cache copied to CPU, traced forward | Same multiplier controls cache allocation |

**Implementation priority: GPU first.** The CPU path is used only for tape
summaries (every `tape_every` steps). It currently copies all params to CPU
and runs a full traced forward — this is why tok/s drops from ~605 to ~25
during tape steps. The GPU path handles all training forward/backward.

Future work: move tape summary computation to GPU entirely (read checkpoint
data directly from GPU cache, compute gnorms/gate stats on GPU, return only
scalar summary values to Python). This eliminates the CPU copy bottleneck
and makes the tape multiplier's memory savings apply to the summary path too.

### Files NOT Modified

| File | Why unchanged |
|------|---------------|
| `core/src/gpu_backward.rs` | Ckpt backward kernels already exist for all rules |
| `core/kernels/*.cu` | No new CUDA kernels needed |
| `core/src/conductor.rs` | Pulse/chunk_sizes unchanged |

## Acceptance Criteria

1. `tape_multiplier: 1` produces boundary-only checkpoints for L1+ levels
2. `tape_multiplier: None` (or omitted) preserves current full-trajectory behavior
3. Gradient correctness: FD gradient check passes at same tolerances for multiplier=1
4. Tape memory matches the formula: `n_blocks × Σ checkpoints(l) × d² × 4`
5. Same tape summary output format (per-level gnorms, gate values) at all multiplier values
6. 8-block d=1024 k=4 model fits on A6000 at multiplier=1 (currently OOMs at full)
7. No regression in existing 4-block d=512 training loss or tok/s at any multiplier
8. Tape size is deterministic from model config — can be computed before forward pass

## Ontological Compliance

- **CS-10**: No mode flag. The tape multiplier is a deployment parameter, not a model mode. Same model, different observation density. No `if training:` branch.
- **CS-18**: Checkpoint interval computation is math (Rust tier). Python tier only passes the multiplier value.
- **CS-32**: Observe-then-advance. Checkpoint states are observed (stored) before the next chunk advances.
- **CS-40**: Opt-in AD. Boundary scoping doesn't change the opt-in contract — `with_tape()` still activates recording.
- **CS-42**: NOT gradient checkpointing in the problematic sense. Between-checkpoint recomputation is bounded by chunk_size (max 512 steps for L3). The recomputation is deterministic and finite — not the unbounded recomputation that CS-42 warns against.
- **CS-48**: Per-level parameters remain independent. Checkpoint intervals are per-level by definition (derived from per-level chunk_sizes).

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| TNT chunkwise approximation | tnt_equations | TNT §3 (2511.07343) | implements |
| eq-074 CMS independence | hope_equations | HOPE §5.1 (2512.24695) | cites |
