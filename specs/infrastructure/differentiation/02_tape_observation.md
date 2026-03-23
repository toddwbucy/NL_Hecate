# Tape Observation Infrastructure — Native Interpretability Layer (v2)

```text
CONTRACT
  Purpose:    Close four remaining gaps in the Wengert tape observation surface
              so that every CMS level's memory state, retention dynamics, and
              freeze status are queryable post-backward — without disrupting
              backward correctness or imposing overhead when disabled.
  Expects:    core/src/tape.rs — TapeBuf (with role/level fields), TapeOp::Opaque
              (with level: Option<usize>, block_index: Option<usize>),
              alloc_named(), obs::* constants, and the full query API
              (find_opaque_ops, find_opaque_at_level, get_saved_by_role,
              enumerate_opaque_blocks, opaque_output_grad_norm) — all implemented
              and passing existing tests.
              All 9 memory rules + 9 frozen variants registered as opaque VJP blocks.
              DeltaRule adapter already uses alloc_named for obs::DGD_DELTA.
              tape_summary.rs provides extract_tape_summary and
              extract_stacked_tape_summary with LevelSummary containing
              level, opaque_key, block_count, output_grad_norm, dgd_delta_norm.
  Guarantees: All existing tape tests continue to pass unmodified.
              After migration, get_saved_by_role() returns data for all 9 adapters
              (not just DeltaRule) — at minimum obs::M_STATES and obs::ERROR.
              LevelSummary gains m_norm, freq_gate_value, is_frozen fields.
              Single parameterized extract function eliminates code duplication.
              Zero overhead when observation fields are not queried.
  Cost:       alloc → alloc_named: zero compute overhead (metadata-only change).
              m_norm computation: one L2 norm per level per diagnostic call.
              freq_gate_value: one sigmoid read per level per diagnostic call.
              is_frozen: one OpaqueKey pattern match per level (zero cost).
              Unified extract: code reduction, no performance change.
  Trade-off:  Named metadata on tape buffers uses string role tags, not typed enums.
              Sufficient for observation; not a general tensor naming system.
              freq_gate_value is NaN for Fixed schedules — callers must check.
              m_norm requires Phase 2 (alloc_named) to land before it can be
              populated — strict phase ordering.
  Position:   specs/infrastructure/differentiation/02_tape_observation.md
  Source:     HOPE (2512.24695) Eq. 88 (practical DGD update) for delta definition.
              HOPE (2512.24695) Eq. 97 (CMS chain) for per-level structure.
              Titans (2501.00663) Eq. 13 (forgetting gate) for alpha/theta semantics.
              CS-40 (opt-in AD), CS-42 (all intermediates stored),
              CS-47 (no in-place mutation of tracked tensors).
```

---

## 0. Status of Original Spec (v1)

The original v1 spec (2026-02-26) identified three gaps:

1. **Saved buffers have no semantic names** → **SHIPPED.** `TapeBuf.role`, `TapeBuf.level`,
   `alloc_named()`, and `obs::*` constants are implemented in `tape.rs`.
2. **CMS levels produce indistinguishable opaque ops** → **SHIPPED.** `TapeOp::Opaque` has
   `level: Option<usize>` and `block_index: Option<usize>`. `record_on_tape()` accepts level.
3. **DGD delta never saved** → **SHIPPED.** DeltaRule adapter saves `cache.error` via
   `alloc_named(..., obs::DGD_DELTA, level)`.

All v1 query API methods are implemented and tested: `find_opaque_ops`, `find_opaque_at_level`,
`get_saved_by_role`, `enumerate_opaque_blocks`, `opaque_output_grad_norm`.

**This v2 revision addresses four new gaps revealed by a graph-informed codebase survey
(5,368 rust-analyzer symbols, 10,948 edges).**

---

## 1. Problem: Four Remaining Gaps

### Gap A — 8 of 9 adapters use `alloc()` instead of `alloc_named()`

The infrastructure for named buffers exists, but only the DeltaRule adapter uses it (one
`alloc_named` call for `obs::DGD_DELTA`). The other 8 active adapters — TitansLMM, Hebbian,
Moneta, YAAD, MEMORA, LatticeOSR, Trellis, AtlasOmega — allocate all saved buffers via
plain `alloc()`. This means:

- `get_saved_by_role(op_idx, obs::M_STATES)` returns `None` for 8 out of 9 rules
- `get_saved_by_role(op_idx, obs::ERROR)` returns `None` for 7 out of 8 rules that have an error buffer
- `get_saved_by_role(op_idx, obs::ALPHA)` returns `None` for all rules with learned decay

The query API works correctly — it simply has no named data to find.

### Gap B — `tape_summary.rs` duplicates extraction logic

`extract_tape_summary()` (single-block) and `extract_stacked_tape_summary()` (multi-block)
are near-identical functions that both:

1. Call `with_tape` → `register_opaque_vjps`
2. Run a traced forward (CMS or stacked)
3. Call `tape.backward(loss_id)`
4. Iterate levels × blocks, querying `enumerate_opaque_blocks`, `opaque_output_grad_norm`,
   and `get_saved_by_role(_, obs::DGD_DELTA)`

The only structural difference is the stacked variant adds a block-index loop.
Code duplication: ~90 lines of near-identical query/assembly logic.

### Gap C — Frequency gate values not in `LevelSummary`

`traced_cms_forward()` records `w_freq`/`b_freq` tape ops for learned frequency gates,
and gradients flow correctly. But `LevelSummary` does not expose the gate's sigmoid output.
For Fixed schedules, levels are either active or inactive (binary); for Learned schedules,
the gate value (0.0–1.0) carries information about how strongly the model wants to fire
that level — a key interpretability signal invisible to the summary.

### Gap D — Frozen-level M staleness invisible

Frozen levels (read-only M) use `FrozenDeltaRule`, `FrozenTitansLMM`, etc. — they compute
M @ q_t but never update M. The current `LevelSummary` has no way to distinguish a frozen
level from an active level that simply didn't fire. Additionally, frozen levels have no
M-norm signal (since M doesn't change, its staleness is invisible). Without these signals,
training diagnostics cannot detect when a frozen level's M has become stale relative to the
data distribution.

---

## 2. Solution: Four Targeted Changes

### 2.1 Adapter `alloc_named` Migration

**File:** `core/src/opaque_adapters.rs`

Convert key saved buffers from `alloc()` to `alloc_named()` across all 8 remaining adapters.
The `level` parameter is already available in every `record_on_tape()` signature.

**Buffers to name per adapter:**

| Buffer | Constant | Adapters |
|--------|----------|----------|
| `cache.m_states` | `obs::M_STATES` | DeltaRule, TitansLMM, Hebbian, Trellis, AtlasOmega |
| `cache.s_states` | `obs::S_STATES` | TitansLMM, AtlasOmega (momentum) |
| `cache.k_mem` | `obs::K_MEM` | TitansLMM, Hebbian, Moneta, YAAD, MEMORA, LatticeOSR, Trellis |
| `cache.v_mem` | `obs::V_MEM` | TitansLMM, Hebbian, Moneta, YAAD, MEMORA, LatticeOSR, Trellis |
| `cache.alpha` | `obs::ALPHA` | All rules with learned decay |
| `cache.theta` | `obs::THETA` | TitansLMM, Moneta, YAAD, MEMORA, Trellis, AtlasOmega |
| `cache.error` | `obs::ERROR` | TitansLMM, Moneta, YAAD, MEMORA, LatticeOSR, Trellis |
| `cache.w1_states` | `obs::W1_STATES` | Moneta, YAAD, MEMORA (MLP rules) |
| `cache.w2_states` | `obs::W2_STATES` | Moneta, YAAD, MEMORA (MLP rules) |

New constants to add to `obs::*` module: `S_STATES`, `W1_STATES`, `W2_STATES`.

**Critical invariant:** Positional index in the `saved` Vec must NOT change. Backward
adapters read by position (`saved[0]`, `saved[1]`, ...). `alloc_named` only adds metadata
to the `TapeBuf` struct — the backward function never sees it.

**Change pattern (mechanical, per adapter):**
```rust
// Before:
tape.alloc(cache.m_states, vec![])
// After:
tape.alloc_named(cache.m_states, vec![], obs::M_STATES, level)
```

### 2.2 Unified Extract Implementation

**File:** `core/src/tape_summary.rs`

Extract the shared query/assembly logic into a single `extract_tape_summary_impl()`
that is parameterized by `n_blocks`:

```rust
fn extract_tape_summary_impl(
    tape: &Tape,
    loss: f32,
    k: usize,
    n_blocks: usize,  // 1 for single-block, N for stacked
) -> (f32, usize, Vec<Vec<LevelSummary>>)
```

The existing public functions become thin wrappers:

```rust
pub fn extract_tape_summary(...) -> TapeSummary {
    // ... setup + traced_cms_forward + backward ...
    let (loss, total, levels) = extract_tape_summary_impl(tape, loss, cfg.k, 1);
    TapeSummary { loss, total_blocks: total, levels: levels[0].clone() }
}

pub fn extract_stacked_tape_summary(...) -> StackedTapeSummary {
    // ... setup + traced_stacked_forward + backward ...
    let (loss, total, block_levels) = extract_tape_summary_impl(tape, loss, cfg.k, n_blocks);
    // ... wrap into StackedTapeSummary ...
}
```

This eliminates ~90 lines of duplicated level-iteration and query logic.

### 2.3 Frequency Gate Value in `LevelSummary`

**File:** `core/src/tape_summary.rs`

Add `freq_gate_value: f32` to `LevelSummary`:

```rust
pub struct LevelSummary {
    // ... existing fields ...
    /// Sigmoid output of the learned frequency gate (0.0–1.0).
    /// NaN if the schedule is Fixed (gate does not exist for this level).
    pub freq_gate_value: f32,
}
```

Population: inside the unified extract, after backward, query the tape for the frequency
gate sigmoid op at each level. The gate is recorded in `traced_cms_forward()` as a
`traced_sigmoid` op on the `pre = dot(mean, w_freq) + b_freq` chain. The sigmoid output
BufId is accessible via the `TracedParamIds.freq_gate_ids` returned by `traced_cms_forward`.

For Fixed schedules: set to `f32::NAN` (the sigmoid op does not exist).

### 2.4 Frozen Flag and M-Norm in `LevelSummary`

**File:** `core/src/tape_summary.rs`

Add two fields:

```rust
pub struct LevelSummary {
    // ... existing fields ...
    /// L2 (Frobenius) norm of the M state saved in this level's opaque block.
    /// 0.0 if no obs::M_STATES role is present (requires alloc_named migration).
    pub m_norm: f32,
    /// True if this level's OpaqueKey is a Frozen* variant (read-only M).
    pub is_frozen: bool,
}
```

**m_norm:** Query `get_saved_by_role(op_idx, obs::M_STATES)` post-backward. Compute
L2 norm. This field is the primary staleness indicator — when a frozen level's `m_norm`
stays constant across checkpoints while `output_grad_norm` changes, M is stale.

**is_frozen:** Check if the `OpaqueKey` debug string starts with "Frozen" (or match
against the `Frozen*` variants directly). Zero compute cost.

**Dependency:** `m_norm` requires Gap A (alloc_named migration) to be completed first.
Before migration, `get_saved_by_role` returns `None` for most adapters and `m_norm` will
be 0.0.

---

## 3. Existing Infrastructure (Unchanged)

These components are complete and require no modification:

| Component | Status |
|---|---|
| `TapeBuf.role`, `TapeBuf.level` fields | Shipped |
| `alloc_named()` method | Shipped |
| `obs::*` constants (M_STATES, ERROR, DGD_DELTA, K_MEM, V_MEM, ALPHA, THETA, EMBEDDED, LEVEL_PARAMS) | Shipped |
| `TapeOp::Opaque { level, block_index }` | Shipped |
| `record_opaque()` with level parameter | Shipped |
| `OpaqueVjp::record_on_tape()` with level parameter | Shipped |
| Query API (5 methods) | Shipped |
| DeltaRule `alloc_named` for DGD_DELTA | Shipped |
| Backward VJP rules | Unchanged |
| `with_tape()` entry point | Unchanged |
| GPU path (`gpu_cms_backward`) | Unaffected — tape is CPU/Rust-path only |

---

## 4. Implementation Phases

### Phase 2: Adapter `alloc_named` Migration (~2 hours)

**File:** `core/src/opaque_adapters.rs`

Mechanical conversion: 8 adapters × ~3-10 key buffers each. Add `obs::S_STATES`,
`obs::W1_STATES`, `obs::W2_STATES` constants to `tape.rs`.

**Tests:** 8 new tests — one per adapter verifying `get_saved_by_role(op_idx, obs::M_STATES)`
(or equivalent primary buffer) returns data after traced forward+backward.

**Risk:** Zero — metadata-only, backward adapters read by position.

### Phase 3: Unified Extract + LevelSummary Extensions (~2 hours)

**File:** `core/src/tape_summary.rs`

a. Extract shared logic into `extract_tape_summary_impl`
b. Add `m_norm`, `freq_gate_value`, `is_frozen` to `LevelSummary`
c. Populate new fields in the unified extract

**Tests:** 3 new — `test_m_norm_from_named_buffer`, `test_frozen_level_flag`,
`test_unified_matches_stacked`.

### Phase 4: PyO3 Surface (~1.5 hours)

**Files:** `python/src/lib.rs`, `python/engine/evaluation.py`

Surface new `LevelSummary` fields in all tape summary PyO3 methods. Update
`print_tape_summary` display. Additive only — no API breakage.

**Deferred to Spec 49:** PyO3 method consolidation (4 → 1) and unified live GPU
observation API (7 scattered methods → 1).

---

## 5. Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `core/src/tape.rs` | 2 | Add `obs::S_STATES`, `obs::W1_STATES`, `obs::W2_STATES` constants |
| `core/src/opaque_adapters.rs` | 2 | 8 adapters: `alloc` → `alloc_named` for key saved buffers |
| `core/src/tape_summary.rs` | 3 | Unified extract + LevelSummary fields |
| `python/src/lib.rs` | 4 | Surface new fields in PyO3 dicts |
| `python/engine/evaluation.py` | 4 | Display new fields |

No new files required.

---

## 6. Testing

### Phase 2 Tests: Named Buffer Queries (8 tests)

One test per adapter verifying that after a traced forward+backward pass,
`get_saved_by_role()` returns non-empty data for the adapter's primary saved buffer.

```rust
#[test]
fn test_titans_lmm_named_buffers() {
    // Run TitansLMM through tape with alloc_named
    // Verify: get_saved_by_role(op_idx, obs::M_STATES) is Some
    // Verify: get_saved_by_role(op_idx, obs::ALPHA) is Some
    // Verify: get_saved_by_role(op_idx, obs::THETA) is Some
}
```

### Phase 3 Tests: LevelSummary Extensions (3 tests)

```rust
#[test]
fn test_m_norm_from_named_buffer() {
    // After extract, LevelSummary.m_norm > 0.0 for active levels
    // (requires Phase 2 migration to have landed)
}

#[test]
fn test_frozen_level_flag() {
    // Configure one frozen level, extract summary
    // Verify: is_frozen == true for frozen level, false for active levels
}

#[test]
fn test_unified_matches_stacked() {
    // Single-block stacked model must produce identical LevelSummary
    // as non-stacked extract on same inputs
}
```

---

## 7. Dependency Graph

```text
Phase 2 (alloc_named) → Phase 3 (unified extract + LevelSummary) → Phase 4 (PyO3)
```

Phase 2 must land before Phase 3 because `m_norm` depends on named M_STATES buffers.
Phase 3 must land before Phase 4 because PyO3 exposes the extended LevelSummary.

---

## 8. Risk to Running Experiments

**Zero.** All changes affect the tape observation path, which is activated only during
eval via `tape_forward_summary` calls. The hot training loop (`forward` + `step_adamw`)
is completely untouched. GPU0 and GPU1 jobs continue undisturbed.

---

## 9. HADES Spec Registration

```json
{
  "_key": "tape-observation",
  "title": "Tape Observation Infrastructure — Native Interpretability Layer (v2)",
  "category": "infrastructure",
  "version": "0.4.0-v2",
  "path": "specs/infrastructure/differentiation/02_tape_observation.md",
  "purpose": "Close 4 remaining observation gaps: adapter alloc_named migration (8 rules), unified extract, freq_gate_value + is_frozen + m_norm in LevelSummary",
  "paper_source": ["2512.24695", "2501.00663"],
  "traced_to_equations": [
    "hope_equations/eq-088-practical-dgd-update",
    "hope_equations/eq-097-hope-cms-chain",
    "titans_equations/eq-013-forgetting-gate"
  ],
  "traced_to_axioms": [],
  "status": "v0.4.0-v2"
}
```
