# Cycle-Scoped Tape with Tape Multiplier

```text
CONTRACT
Purpose:    The Wengert tape's intermediate cache is scoped to CMS cycles — the
            natural clock of the memory system. A "cycle" is one complete CMS
            rotation: the period of the slowest active level (e.g., 8 tokens for
            k=2 [1,8], 512 tokens for k=4 [1,8,64,512]). The tape_multiplier
            controls how many cycles of cache are retained at any time.
            tape_multiplier=1 is the default — it keeps one cycle of cache live,
            which is the minimum required for backward to function. Higher values
            retain more cycles for deeper gradient flow or diagnostic visibility.
            The tape is not optional — it is the mechanism that enables gradient
            flow through memory. Without it, there is no backward pass.

Expects:    Existing Wengert tape (core/src/tape.rs) with opt-in recording.
            CMS Conductor/Pulse system that tracks chunk boundaries per level.
            GPU stacked forward/backward with existing checkpointed variants
            (GpuMemoryCache::*Ckpt) that store M at intervals and recompute
            between checkpoints during backward.
            TNT periodic reset (spec 08): local memory shards are independent
            given M_init — each shard's forward+backward is self-contained.
            (TNT eq-006, 2511.07343)

Guarantees: 1. Peak tape memory is a deterministic function of model architecture
               and tape_multiplier — NOT of input length or content.
               tape_bytes = n_blocks × k × cycles_retained × cycle_cache_size
               where cycle_cache_size = tokens_per_cycle × d² × sizeof(f32) × 2 (M+S)
            2. tape_multiplier=1 is the default. It retains one CMS cycle of
               cache — the minimum for backward to compute gradients through
               memory. This is not an optimization — it is the architectural
               baseline.
            3. Cache drops occur at cycle boundaries on a rolling basis. When a
               new cycle completes, the oldest cycle beyond the retention window
               is freed. Cache management is synchronized to the CMS clock.
            4. Gradient correctness is preserved within the retained window.
               Gradients flow through tape_multiplier cycles of memory updates.
               Beyond the window, gradients are truncated at the cycle boundary
               (analogous to truncated BPTT, but aligned to CMS structure).
            5. Higher tape_multiplier values retain more cycles:
               - 1: one cycle (default, minimum for backward)
               - 2: two cycles (gradient flows through 2 rotations)
               - N: N cycles (deeper gradient flow, more memory)
               The full-trajectory behavior of the naive implementation is
               equivalent to tape_multiplier = ceil(seq_len / cycle_length).
            6. The tape structure is identical during training and inference.
               Observability metrics (gnorms, M norms, gate values) are computed
               from the retained cache in both contexts.

Cost:       At tape_multiplier=1, gradients are truncated at one cycle boundary.
            For k=2 [1,8], one cycle = 8 tokens — gradients flow through 8 tokens
            of memory updates before truncation. Higher multipliers extend this
            at the cost of proportionally more memory.

Trade-off:  tape_multiplier=1 gives the minimum gradient window (one cycle).
            This is sufficient for most training because:
            - L0 (every token) gets full gradient within the cycle
            - Higher levels fire at cycle boundaries where gradients are retained
            - The CMS architecture is designed so that each cycle is self-contained
            Higher multipliers are useful for:
            - Diagnosing cross-cycle gradient pathologies
            - Research into longer-range memory gradient flow
            - Observability of M trajectory across multiple cycles

Position:   specs/infrastructure/25_segment_scoped_tape.md

Source:     TNT (2511.07343) — chunkwise training approximation. Memory updates
            occur at chunk boundaries. Within-chunk processing accumulates error
            signals that are applied at the boundary.
            TNT (2511.07343) eq-006 — Local memory periodic reset. Shards are
            independent given M_init. Forward+backward can be fused per shard
            without cross-shard cache retention.
            HOPE (2512.24695) eq-074 — CMS level independence. Each level's tape
            storage is scoped to its own chunk boundaries.
            CS-40 — Opt-in AD. The tape is activated explicitly; cycle scoping
            does not change the opt-in contract.
            CS-42 — Gradient checkpointing hurts NL. At multiplier=1, boundary
            states ARE stored (not recomputed). Between-checkpoint recomputation
            is bounded and deterministic — not the unbounded recomputation that
            CS-42 warns against.
```

## Problem Statement

The naive implementation stores per-token intermediates for every timestep in the
sequence, for every block. This is effectively tape_multiplier=∞ — retaining the
entire trajectory. At d=1024, 8 blocks, seq_len=512:

- Per-token M state: d × d × 4 bytes = 4MB
- Per-token across sequence: 512 × 4MB = 2GB per level per block
- With momentum S: 2× = 4GB per level per block
- 8 blocks × k=1: 32GB just for memory state history

This consumed 40.5GB for a 178.5M parameter model — a 225:1 ratio of tape storage
to model parameters. The naive approach inherited instability from input data
(tape size scaled with seq_len), making memory budgets unpredictable.

The fix is not to add checkpointing as an optimization. The fix is to recognize
that the tape is architecturally scoped to CMS cycles, and always has been. The
naive implementation simply failed to enforce that scoping.

## Key Insight: CMS Cycles as the Natural Cache Unit

The CMS operates like a Swiss watch. Multiple levels fire at different rates, but
they are synchronized: when the slowest level fires, all levels fire together.

```text
k=2, chunk_sizes=[1, 8]:
  L0: ████████ ████████ ████████ ████████  (every token)
  L1: █───────█───────█───────█───────█    (every 8 tokens)
      ^cycle 1^cycle 2^cycle 3^cycle 4^

k=4, chunk_sizes=[1, 8, 64, 512]:
  L0: every token
  L1: every 8 tokens
  L2: every 64 tokens
  L3: every 512 tokens (= 1 cycle per sequence)
      ^─────────────── one cycle ───────────────^
```

One **cycle** = the period of the slowest active level. For k=2 [1,8], one cycle
is 8 tokens. Within a cycle, all levels that fire have their updates cached.
At the cycle boundary, the cache for the oldest cycle beyond the retention window
is freed.

The tape_multiplier controls how many cycles to retain:

| Multiplier | Cycles retained | Cache window (k=2 [1,8]) | Gradient depth |
|------------|----------------|--------------------------|----------------|
| 1 (default)| 1              | 8 tokens                 | 1 cycle        |
| 2          | 2              | 16 tokens                | 2 cycles       |
| 4          | 4              | 32 tokens                | 4 cycles       |
| 8          | 8              | 64 tokens                | 8 cycles       |
| 64 (max)   | 64             | 512 tokens (full seq)    | full trajectory |

## Key Insight 2: Cycle Length Scales with k

As levels are added via push-up, the cycle length grows exponentially:

| k | chunk_sizes | cycle_len | cycles/seq (s=512) |
|---|-------------|-----------|-------------------|
| 1 | [1] | 1 | 512 |
| 2 | [1,8] | 8 | 64 |
| 3 | [1,8,64] | 64 | 8 |
| 4 | [1,8,64,512] | 512 | 1 |
| 5 | [1,8,64,512,4096] | 4096 | 0 (cycle > seq) |

**Critical consequence at k=4**: cycle_length = seq_len = 512. There is exactly
one cycle per sequence. tape_multiplier=1 means "keep everything" — the multiplier
provides no memory reduction. This is inherent to the CMS architecture: when the
slowest level fires once per sequence, the entire sequence IS one cycle.

**At k=5+**: cycle_length exceeds seq_len. The cycle never completes within a
single forward pass. Cache management must handle partial cycles — the drop
never fires because the cycle boundary is never reached. Effectively equivalent
to full trajectory, but the code must not break.

Per-cycle cache cost at d=1024, 8 blocks, tape_multiplier=1:

| k | cycle_len | M/S per cycle | Projs per cycle | Total/cycle | × 8 blocks × k |
|---|-----------|--------------|-----------------|-------------|-----------------|
| 2 | 8 | 16 MB | ~96 KB | ~16 MB | 256 MB |
| 3 | 64 | 24 MB | ~768 KB | ~25 MB | 600 MB |
| 4 | 512 | 32 MB | ~6 MB | ~38 MB | 1.2 GB |

This is a scaling cost the architecture pays as k increases. The code must:
1. Derive cycle_length from `max(chunk_sizes)` dynamically — never hardcode
2. Handle the degenerate case where cycles_in_seq <= 1 (no drops possible)
3. Report the computed tape budget at model init so operators see the cost
4. Make it easy to see the VRAM impact of adding a level before committing

The tape_multiplier becomes most valuable at low k (many short cycles to manage)
and less relevant at high k (few long cycles, or cycle > seq_len). At high k,
VRAM scaling is managed by other means: larger GPUs, model parallelism, or
longer seq_len that creates more cycles.

## Key Insight 3: TNT Shard Independence (eq-006)

The TNT parallel strategy (`tnt_hierarchical`) splits the sequence into shards of
size `tnt_global_chunk_size` (C_G). Each shard is subdivided into `n_batch = C_G / C_L`
local chunks of size `tnt_local_chunk_size` (C_L), processed as a batch.

The TNT paper's eq-006 states that local memory **resets to M_init at each shard
boundary**. This means:

- Each shard's local memory forward is **self-contained**: it starts from M_init,
  processes n_batch local chunks, and produces outputs + gradients independently.
- There is **no cross-shard dependency** in the local memory state. Shard 3's
  backward does not depend on shard 0's caches.
- The only cross-shard state is the **global memory** V, which evolves via small
  summary vectors (k_sum, v_sum — O(d) each). These are negligible.

The cycle-scoped cache integrates naturally with TNT shards: each shard contains
one or more cycles. Rolling eviction operates during TNT forward — after each
shard's inner cache is pushed, caches exceeding the retention window are freed.
Shard independence (TNT eq-006) guarantees this is safe: evicted shards still
contribute global M gradients via their retained summary vectors.

### Global Memory Backward

The global memory V evolves sequentially across shards via summary outer products
(spec 08, TNT eq-005). The backward for the global update chain requires only:
- `k_summaries[shard_idx]`, `v_summaries[shard_idx]`: mean k/v vectors (d each)

No per-shard full-matrix snapshots (e.g., `global_m_before`) are kept. The global
backward propagates `d_m_carry` through the summary update chain in reverse,
using only the retained summary vectors and the inner caches of retained shards.
Summaries are O(d) per shard — negligible even at d=1024.

The global backward runs within the reverse shard loop: every shard (including
evicted ones) contributes global M gradients via its summaries. Only retained
shards also run the inner backward for projection/gate gradients.

## Design: Cycle-Scoped Cache Management

### Cache Lifecycle

For each level, within each block:

```rust
// ── Per-cycle cached state ──────────────────────────────────────────
struct CycleCache {
    projections: Projections,  // k_mem, v_mem, q_mem — HOPE eq-009
    gates: Gates,              // alpha (Titans eq-012), theta (HOPE eq-088), eta (Titans eq-014)
    m_trajectory: Vec<M>,      // M_t at each step — full trajectory (CS-42: no recompute)
    s_trajectory: Vec<M>,      // S_t (momentum, Titans eq-013 — Titans only)
    k_norms: Vec<f32>,         // ‖k_t‖₂ per step
    q_norms: Vec<f32>,         // ‖q_t‖₂ per step
}

struct Projections { k_mem: [f32; d], v_mem: [f32; d], q_mem: [f32; d] }
struct Gates { alpha: f32, theta: f32, eta: f32 }

// ── Forward: one step within a cycle ────────────────────────────────
fn forward_step(t: usize, cache: &mut CycleCache, m: &mut M,
                tape_multiplier: usize, retained: &mut VecDeque<CycleCache>)
{
    // 1. Projections + gates (HOPE eq-009, Titans eq-012/014)
    let (proj, gates) = compute_projections_and_gates(input_t);

    // 2. Inner memory kernel → update M, produce y_t (Titans eq-004)
    let y_t = memory_update(m, &proj, &gates);

    // 3. Store in current cycle cache (CS-42: arena-allocated)
    cache.projections = proj;
    cache.gates = gates;
    cache.m_trajectory.push(m.clone());

    // 4. Cycle boundary → rolling eviction (spec 25)
    if is_cycle_boundary(t) {
        retained.push_back(std::mem::take(cache));
        while retained.len() > tape_multiplier {
            retained.pop_front();  // free oldest cycle's cache
        }
    }
}

// ── Backward: reverse over retained cycles ──────────────────────────
fn backward_pass(retained: &mut VecDeque<CycleCache>, grads: &mut GradBuffers) {
    // Newest to oldest — gradient flows through tape_multiplier cycles
    for cache in retained.drain(..).rev() {
        // 1. Read projections/gates directly (no recomputation — CS-42)
        // 2. Read M/S trajectory directly (full trajectory stored)
        // 3. Compute gradients via chain rule (HOPE eq-088, Titans eq-012/013/014)
        accumulate_gradients(&cache, grads);
        // 4. Cache freed on drop (cycle-scoped lifetime)
    }
}
```

The key: **projections, gates, and M/S trajectories are never recomputed**. They
are always stored in the cache. The full M trajectory within each retained shard
is available directly — no boundary-checkpoint recomputation is needed (CS-42).

### What the Cache Stores per Cycle

| Component | Size per cycle | Notes |
|-----------|---------------|-------|
| k_mem, v_mem, q_mem | 3 × cycle_len × d | Projections — always cached |
| alpha, theta, eta | 3 × cycle_len | Gates — always cached |
| M trajectory | cycle_len × d² | Full M at each step (no recomputation) |
| S trajectory | cycle_len × d² | Momentum (Titans only) |
| k_norms, q_norms | 2 × cycle_len | L2 norms — always cached |

For k=2 [1,8] at d=512 (cycle_len=8):
- Projections + gates: 8 × 512 × 3 + 8 × 3 = 12,312 floats ≈ 48 KB
- M+S trajectories: 2 × 8 × 512² = 4,194,304 floats ≈ 16 MB
- **Total per cycle: ~16 MB**

For k=2 [1,8] at d=1024 (cycle_len=8):
- Projections + gates: ~96 KB
- M+S trajectories: 2 × 8 × 1024² = 16,777,216 floats ≈ 64 MB
- **Total per cycle: ~64 MB**

### Memory Budget

For d=512, 4 blocks, k=2 [1,8], seq_len=512 (64 cycles):

| Multiplier | Retained | Cache total | Notes |
|------------|----------|-------------|-------|
| 1 (default)| 1 cycle  | 4 × 2 × 4 MB = 32 MB | Minimum viable |
| 4          | 4 cycles | 4 × 2 × 16 MB = 128 MB | Development |
| 64 (full)  | 64 cycles| 4 × 2 × 256 MB = 2 GB | Full trajectory |

For d=1024, 8 blocks, k=2 [1,8], seq_len=512 (64 cycles):

| Multiplier | Retained | Cache total | Notes |
|------------|----------|-------------|-------|
| 1 (default)| 1 cycle  | 8 × 2 × 16 MB = 256 MB | Fits easily |
| 4          | 4 cycles | 8 × 2 × 64 MB = 1 GB | Comfortable |
| 64 (full)  | 64 cycles| 8 × 2 × 1 GB = 16 GB | Tight on A6000 |

### Tape Size Formula

Deterministic from model config:

```text
cycle_length = max(chunk_sizes[0..k])   # period of slowest level
cycles_in_seq = ceil(seq_len / cycle_length)  # partial cycles count
retained_cycles = min(tape_multiplier, max(1, cycles_in_seq))

per_cycle_cache = cycle_length × d² × sizeof(f32)  # M trajectory (full, no recomputation)
                + cycle_length × d² × sizeof(f32)  # S trajectory (Titans only)
                + cycle_length × d × 3 × sizeof(f32)  # projections (k, v, q)
                + cycle_length × 3 × sizeof(f32)       # gates (alpha, theta, eta)

tape_bytes = n_blocks × k × retained_cycles × per_cycle_cache
```

This is a **model constant** for fixed seq_len. Print it on the model card.

## Config

### Default: tape_multiplier = 1

```rust
// In MAGConfig (model.rs):
/// Tape multiplier: how many CMS cycles of cache to retain.
/// 1 = one cycle (default, minimum for backward).
/// N = N cycles (deeper gradient flow, more memory).
/// The tape is required for backward — this is not optional.
#[serde(default = "default_tape_multiplier")]
pub tape_multiplier: usize,

fn default_tape_multiplier() -> usize { 1 }
```

Python config:
```python
# In BuildConfig (config.py):
tape_multiplier: int = 1  # CMS cycles of cache to retain (1 = default)
```

JSON config:
```json
{
    "model": {
        "tape_multiplier": 1
    }
}
```

Note: `tape_multiplier` is no longer `Option<usize>`. It is always present,
always >= 1. The old `None` / `Some(0)` values mapped to the naive full-trajectory
behavior, which is now expressed as `tape_multiplier = seq_len / cycle_length`.

## Implementation

### What Changes

**Cache management layer**: Rolling eviction of shard inner caches based on the
cycle-scoped retention window. This does NOT change the forward or backward
kernels — only WHEN caches are freed.

**Sequential path** (`gpu_memory_forward`): Unchanged. Full M trajectory is
stored within each call. Cycle-scoped eviction is a future extension for the
sequential path. Currently, all production configs use TNT.

**TNT path** (`gpu_tnt_forward`): Rolling eviction at the shard level. After each
shard's forward completes, if the accumulated shard count exceeds
`retained_shards(shard_size)`, the oldest shard's inner cache is freed. The
backward only computes inner gradients for retained shards; evicted shards
contribute global M gradients only (gradient truncation). Summaries (O(d) each)
are retained for ALL shards to preserve the global M backward chain rule.

**Default change**: `tape_multiplier` defaults to 1, not None. All existing
configs without `tape_multiplier` get the cycle-scoped behavior automatically.
To recover the old full-trajectory behavior, set `tape_multiplier` to
`seq_len / max(chunk_sizes)`.

### Files to Modify

| File | Change |
|------|--------|
| `core/src/model.rs` | Change `tape_multiplier` from `Option<usize>` to `usize`, default 1 |
| `core/src/gpu_forward.rs` | Cycle-aware cache retention in both sequential and TNT paths |
| `core/src/gpu_backward.rs` | Consume caches cycle-by-cycle, free after each cycle's backward |
| `python/engine/config.py` | Default `tape_multiplier = 1` |
| `python/src/lib.rs` | Update PyO3 binding (no longer Option) |

### Files NOT Modified

| File | Why unchanged |
|------|---------------|
| `core/kernels/*.cu` | No new CUDA kernels needed |
| `core/src/conductor.rs` | Pulse/chunk_sizes unchanged — cycle timing is derived |

## Acceptance Criteria

1. `tape_multiplier=1` (default) retains one CMS cycle of cache — backward works correctly
2. No separate "replay" path — projections and gates are always cached, never recomputed
3. Cache drops happen at cycle boundaries, synchronized to the CMS clock
4. Gradient correctness: FD gradient check passes within the retained window
5. d=512 4-block k=2 at tape_multiplier=1: no throughput regression vs naive full-trajectory (~580 tok/s)
6. d=1024 8-block k=2 at tape_multiplier=1: fits on A6000 (~256 MB tape)
7. Higher tape_multiplier values proportionally increase cache and gradient depth
8. Tape size is deterministic from model config — can be computed before forward pass
9. Same model behavior at all multiplier values (only gradient truncation depth changes)
10. TNT shard caches retained up to retained_shards window; oldest shard caches evicted when window exceeded

## Ontological Compliance

- **CS-10**: No mode flag. The tape multiplier is an architectural parameter, not a mode. Same model, same forward path, same backward path. Only the cache retention window differs.
- **CS-18**: Cache management is math (Rust tier). Python tier only passes the multiplier value.
- **CS-32**: Observe-then-advance. Cache entries are stored (observed) before the cycle advances. Drop happens after backward consumes the cycle.
- **CS-40**: Opt-in AD. Cycle scoping doesn't change the opt-in contract — `with_tape()` still activates recording.
- **CS-42**: NOT gradient checkpointing in the problematic sense. Within-cycle recomputation (from M boundary states) is bounded by cycle_length. This is deterministic and finite — not the unbounded recomputation that CS-42 warns against.
- **CS-48**: Per-level parameters remain independent. All levels within a cycle are cached together because they fire together (CMS synchronization).

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| TNT chunkwise approximation | tnt_equations | TNT §3 (2511.07343) | implements |
| eq-006 local memory periodic reset | tnt_equations | TNT §3.2 (2511.07343) | implements |
| eq-005 global memory update | tnt_equations | TNT §3.1 (2511.07343) | cites |
| eq-074 CMS independence | hope_equations | HOPE §5.1 (2512.24695) | cites |
