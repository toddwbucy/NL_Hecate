# Per-Head Observation in Tape Summary

```text
CONTRACT
  Purpose:    Extend LevelSummary and BlockLevelSummary with per-head memory
              norms so that head-level imbalances — one head's M dominating or
              collapsing while others are healthy — are visible in tape
              diagnostics and CmsTape sidecar files.

  Expects:    - core/src/tape_summary.rs — LevelSummary with m_norm (aggregate
                Frobenius norm), freq_gate_value, is_frozen (spec 48 shipped).
              - core/src/conductor.rs — ContextState.memory[level] stores
                final M as flat [d*d] Vec<f32>.
              - core/src/model.rs — MAGConfig with swa.num_heads, swa.head_dim,
                d_model = num_heads * head_dim.
              - core/src/gpu_params.rs — memory_norms() computes per-(block,level)
                aggregate Frobenius norms on GPU.
              - Memory matrix M is [d × d] row-major, where d = d_model.
                Per-head decomposition treats this as num_heads independent
                [head_dim × head_dim] diagonal blocks: head h owns rows
                [h*head_dim..(h+1)*head_dim] × cols [h*head_dim..(h+1)*head_dim].
              - python/src/lib.rs — PyO3 tape summary methods expose LevelSummary
                fields as Python dicts.
              - python/engine/cms_tape.py — CmsTape accumulator with
                l0_per_head_sample_rate parameter (currently dead, spec 49).

  Guarantees: - LevelSummary gains `head_m_norms: Vec<f32>` (length = num_heads)
                when MemoryRuleKind::has_square_m() is true, num_heads > 1,
                and final_memory has d_model² elements. Empty Vec otherwise.
              - BlockLevelSummary gains `head_m_norms: Vec<f32>` (same).
              - GPU path gains `memory_norms_per_head()` returning
                Vec<Vec<Vec<f32>>> — [n_blocks][k][num_heads]. Empty inner Vec
                for non-square rules or num_heads <= 1.
              - PyO3 tape summary dicts include "head_m_norms": [float, ...].
              - Existing aggregate m_norm unchanged — head norms satisfy
                sum(head_m_norms_i²) ≤ m_norm² (equality when cross-head
                coupling is zero, which holds for Delta/Titans/Hebbian rules).
              - Zero overhead when head_m_norms is not populated (Vec is empty
                for non-square rules, k=1, or num_heads=1).
              - CmsTape per-head accumulation via l0_per_head_sample_rate is
                DEFERRED to spec 49 (task_a1fafb), which is blocked on this spec.

  Cost:       - CPU tape path: num_heads Frobenius norms per level per diagnostic
                call. Each norm is O(head_dim²) — negligible vs forward pass.
              - GPU path: D2H copy of mem_dd = num_heads × head_dim² floats per
                level, then CPU-side per-tile Frobenius norm reduction. No GPU
                kernel launch — host reduction is sufficient at diagnostic cadence.
              - Sidecar size (when spec 49 lands): +num_heads floats per level
                per recorded step. At num_heads=16, k=4, 1024 L0 samples: +256KB
                (~10% increase over aggregate-only budget).

  Trade-off:  - Two memory layouts coexist:
                  CPU tape path: M is [d × d] row-major. Per-head decomposition
                    extracts diagonal sub-blocks [h*hd..(h+1)*hd]².
                  GPU path: M is packed per-head tiles at h*hd*hd offset, each
                    tile contiguous [hd × hd]. Per-head norm is a simple slice.
                Both layouts give equivalent per-head norms when cross-head
                coupling is zero (true for Delta/Titans/Hebbian rules).
              - MLP-based rules (MONETA/YAAD/MEMORA) have non-square M
                (w1: [d,4d], w2: [4d,d]). Per-head decomposition of MLP
                weights is less meaningful — head_m_norms is empty for these
                rules (graceful degradation).

  Position:   specs/infrastructure/differentiation/06_per_head_observation.md
              Implements: spec 48 Phase 4 (per-head extension to LevelSummary)
              Used by:    spec 49 (CmsTape l0_per_head_sample_rate accumulation)
              Blocks:     task_a1fafb (spec 49 completion)

  Source:     Titans (2501.00663) Eq. 32 — M update uses outer(k,v) where
                k,v ∈ R^d; per-head decomposition follows from the attention
                head structure of W_K, W_V projections.
              HOPE (2512.24695) Eq. 97 — CMS chain per-level structure;
                per-head norms add a finer granularity within each level.
              CS-32 (observe-then-advance) — head norms read post-backward.
              CS-40 (opt-in AD) — observation does not enter the tape.
```

---

## 1. Problem

### Current state

`LevelSummary.m_norm` reports the Frobenius norm of the entire d×d memory
matrix M. When `d_model=768` and `num_heads=12`, this single scalar aggregates
12 independent `64×64` head memory blocks into one number.

This hides three diagnostic signals:

1. **Head imbalance**: One head's M can dominate the aggregate norm while
   others collapse to near-zero. The aggregate norm looks healthy.

2. **Per-head dormancy**: A frozen level's M may be stale in some heads but
   not others. The aggregate `m_norm` stays constant, masking partial staleness.

3. **Head-level gradient flow**: Combined with per-head output gradient norms
   (future work), per-head M norms would identify which heads receive gradient
   flow and which are dormant — a finer signal than the current per-level
   `output_grad_norm`.

### What exists

| Component | Granularity | Location |
|-----------|-------------|----------|
| `LevelSummary.m_norm` | per-level aggregate | `tape_summary.rs` |
| `memory_norms()` | per-(block, level) aggregate | `gpu_params.rs` |
| `update_m_norm_tracking()` | per-(block, level) delta | `gpu_params.rs` |
| `CmsTape.l0_per_head_sample_rate` | dead parameter | `cms_tape.py` |

None of these decompose to per-head granularity.

---

## 2. Solution

### 2.1 Per-Head Norm Extraction (CPU Tape Path)

**File:** `core/src/tape_summary.rs`

Add `head_m_norms` to both summary structs:

```rust
pub struct LevelSummary {
    // ... existing fields unchanged ...

    /// Per-head Frobenius norms of the block-diagonal M sub-matrices.
    /// Length = num_heads for square-M rules (Delta, Titans, Hebbian, etc.).
    /// Empty for MLP-based rules (MONETA/YAAD/MEMORA) where M is non-square.
    pub head_m_norms: Vec<f32>,
}

pub struct BlockLevelSummary {
    // ... existing fields unchanged ...
    pub head_m_norms: Vec<f32>,
}
```

**Extraction logic** in `extract_level()`:

```rust
fn per_head_m_norms(m: &[f32], d: usize, num_heads: usize) -> Vec<f32> {
    let head_dim = d / num_heads;
    // M is [d × d] row-major. Head h owns the diagonal block:
    //   rows [h*hd .. (h+1)*hd], cols [h*hd .. (h+1)*hd]
    (0..num_heads).map(|h| {
        let row_start = h * head_dim;
        let col_start = h * head_dim;
        let mut sum_sq = 0.0f32;
        for r in 0..head_dim {
            let row_off = (row_start + r) * d + col_start;
            for c in 0..head_dim {
                let val = m[row_off + c];
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }).collect()
}
```

The function signature of `extract_level()` gains `num_heads` and `d_model`:

```rust
fn extract_level(
    tape: &Tape,
    all_blocks: &[(usize, OpaqueKey, Option<usize>, Option<usize>)],
    lev: usize,
    block_filter: Option<usize>,
    final_memory: Option<&[f32]>,
    gate_values: Option<&[f32]>,
    num_heads: usize,        // NEW
    d_model: usize,          // NEW — validates square-M layout
    has_square_m: bool,      // NEW — from MemoryRuleKind::has_square_m()
) -> LevelSummary
```

Per-head norms are computed only when `has_square_m` is true,
`final_memory.len() == d_model * d_model`, and `num_heads > 1`. Otherwise
`head_m_norms` is empty (graceful degradation for MLP rules, non-square M,
or `num_heads=1`). The `has_square_m` gate is derived from the memory rule
kind — MLP-based rules (Moneta/YAAD/MEMORA/SwiGluMlp) return false.

### 2.2 Per-Head Norms on GPU

**File:** `core/src/gpu_params.rs`

Add alongside existing `memory_norms()`:

```rust
/// Compute per-(block, level, head) Frobenius norms of M sub-matrices on GPU.
/// Returns [n_blocks][k][num_heads].
/// GPU layout: packed per-head tiles, head h at offset h*hd*hd.
pub fn memory_norms_per_head(&self) -> Vec<Vec<Vec<f32>>> {
    // D2H copy of slot-0 (mem_dd floats), then per-tile Frobenius norm.
    // Each tile is contiguous [hd × hd] — simple slice reduction.
}
```

**Why CPU-side**: GPU memory uses packed per-head tiles (`mem_dd = num_heads *
head_dim²`), so each tile is contiguous — a simple slice reduction on the host
after D2H copy. No custom CUDA kernel needed. Runs at eval cadence only.

### 2.3 PyO3 Surface

**File:** `python/src/lib.rs`

In all 4 tape summary methods (and helper `set_core_level_fields()`), add
the new field:

```rust
ldict.set_item("head_m_norms", &lvl.head_m_norms)?;
```

This is a `Vec<f32>` which PyO3 converts to a Python `list[float]`.

### 2.4 CmsTape Per-Head Accumulation (Python)

**File:** `python/engine/cms_tape.py`

This change lives in spec 49 (task_a1fafb) which is blocked on this spec.
Once per-head norms are available in the tape summary dict, the CmsTape
accumulator will:

1. Record `head_m_norms` per level at `l0_per_head_sample_rate` (1/16)
2. Store as `[sampled_steps, num_heads]` parallel arrays per level
3. Flush to sidecar `.cms.json` alongside aggregate data

The `l0_per_head_sample_rate` parameter and its validation already exist
in `CmsTape.__init__()`. The `record()` method will consume `head_m_norms`
from the tape summary dict when present.

---

## 3. Memory Layouts

Two M layouts coexist. Per-head norm extraction handles both.

### 3.1 CPU Tape Path: Row-Major `[d × d]`

M is stored row-major as `[d × d]` where `d = num_heads × head_dim`.
Per-head norms extract **diagonal blocks** at `rows [h*hd..(h+1)*hd],
cols [h*hd..(h+1)*hd]`:

```text
         col 0..hd    col hd..2hd   col 2hd..3hd  ...  col (H-1)hd..d
row 0    ┌──────────┬─────────────┬──────────────┬───┬──────────────┐
  ..     │  Head 0  │  cross 0→1  │  cross 0→2   │   │  cross 0→H-1 │
row hd   │  hd×hd   │             │              │   │              │
         ├──────────┼─────────────┼──────────────┼───┤              │
row hd   │cross 1→0 │   Head 1    │  cross 1→2   │   │              │
  ..     │          │   hd×hd     │              │   │              │
         └──────────┴─────────────┴──────────────┴───┴──────────────┘
```

**Invariant:** `sum(head_norm_h² for h in 0..H) ≤ m_norm²`. Equality holds
only when all cross-head blocks are zero.

### 3.2 GPU Path: Packed Per-Head Tiles

GPU stores M as `mem_dd = num_heads × head_dim²` contiguous tiles:

```text
offset 0          hd²          2·hd²              (H-1)·hd²    H·hd²
┌─────────────┬─────────────┬─────────────┬───┬─────────────┐
│   Head 0    │   Head 1    │   Head 2    │...│  Head H-1   │
│  hd×hd flat │  hd×hd flat │  hd×hd flat │   │  hd×hd flat │
└─────────────┴─────────────┴─────────────┴───┴─────────────┘
```

Per-head norm is a simple contiguous slice reduction: `||buf[h*hd²..
(h+1)*hd²]||_F`. No cross-head coupling exists in this layout.

---

## 4. Scope Boundary

### In scope (this spec)

| Change | File | Type |
|--------|------|------|
| `head_m_norms: Vec<f32>` on `LevelSummary` | `tape_summary.rs` | Additive field |
| `head_m_norms: Vec<f32>` on `BlockLevelSummary` | `tape_summary.rs` | Additive field |
| `per_head_m_norms()` helper function | `tape_summary.rs` | New function |
| `num_heads`, `d_model`, `has_square_m` on `extract_level()` | `tape_summary.rs` | Signature change |
| `MemoryRuleKind::has_square_m()` method | `model.rs` | New method |
| `memory_norms_per_head()` method | `gpu_params.rs` | New method |
| `memory_norms()` uses `ctx.mem_dd()` not `d*d` | `gpu_params.rs` | Bug fix |
| `head_m_norms` in PyO3 tape summary dicts | `lib.rs` | Additive field |

### Out of scope (deferred)

| Item | Reason | Where |
|------|--------|-------|
| Per-head output gradient norms | Requires per-head gradient accumulation on tape | Future spec |
| Per-head alpha/theta gate stats | Already aggregated in gate diagnostics (spec 44) | Spec 44 extension |
| CmsTape per-head accumulation | Spec 49 (blocked on this spec) | task_a1fafb |
| Cross-head coupling norms | Low diagnostic value, high compute cost | Not planned |
| CUDA kernel for per-head norms | CPU-side sufficient at diagnostic cadence | Optimize if needed |

---

## 5. Tests

```rust
// Per-head norms sum to ≤ aggregate norm squared
#[test]
fn test_head_norms_consistent_with_aggregate()

// Per-head norms length == num_heads for square-M rules
#[test]
fn test_head_norms_length_matches_num_heads()

// Per-head norms empty for MLP rules (non-square M)
#[test]
fn test_head_norms_empty_for_mlp_rules()

// Stacked model: per-(block, level, head) norms populated
#[test]
fn test_stacked_head_norms()

// Identity-like M: all head norms should be approximately equal
#[test]
fn test_head_norms_uniform_for_identity_m()
```

---

## 6. HADES Registration

```json
{
  "_key": "per-head-observation",
  "title": "Per-Head Observation in Tape Summary",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/differentiation/06_per_head_observation.md",
  "purpose": "Extend LevelSummary with per-head memory norms for head-level imbalance and dormancy diagnosis",
  "paper_source": ["2501.00663", "2512.24695"],
  "traced_to_equations": ["titans_equations/eq-032-associative-memory-update", "hope_equations/eq-097-hope-cms-chain"],
  "traced_to_axioms": ["hope_axioms/CS-32", "hope_axioms/CS-40"],
  "depends_on_specs": ["hecate_specs/tape-observation"],
  "status": "v0.4.0"
}
```
