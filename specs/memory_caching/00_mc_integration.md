# Memory Caching Module — Integration Contract

```text
CONTRACT
  Purpose:    Extend NL_Hecate with the four Memory Caching aggregation variants
              from arXiv:2602.24281: Residual Memory (RM), Gated Residual Memory
              (GRM), Memory Soup, and Sparse Selective Caching (SSC). Each variant
              defines a distinct strategy for how segment-level compressed
              checkpoints are retrieved and composed into the current context —
              making segment access semantics an architecture choice, not a fixed
              operation.
  Expects:    core/src/tape.rs — Wengert tape with full observation infrastructure
              (TapeBuf named roles, per-level OpaqueKey, DGD delta exposure, query
              API) per specs/infrastructure/differentiation/02_tape_observation.md.
              All 4 CMS levels (k=4) producing distinct per-level OpaqueKey entries.
              OpaqueKey enum extended with MC variant keys:
                OpaqueKey::ResidualMemory, OpaqueKey::GatedResidualMemory,
                OpaqueKey::MemorySoup, OpaqueKey::SparseSelectiveCaching.
              Segment boundary infrastructure: checkpoint cache (BufId-indexed),
              segment index (step count → segment id), and retrieval API.
  Guarantees: Each MC variant implements the MemoryCachingRule trait with:
                segment_forward(tape, m_state, segment_idx, level) → BufId
                aggregate(tape, cache, query, connector, level) → BufId
              The tape records both store and aggregate operations as opaque VJP
              blocks, making cache access patterns inspectable via the observation
              query API.
              GRM gate uses u_t (connector vector projection), NEVER q_t (query).
              This is enforced by trait signature separation — u_t and q_t are
              distinct buffer arguments. Sharing them collapses the model to zeros
              across all reported metrics (MC paper Table 3 ablation).
  Cost:       Segment checkpoint cache: O(N × d²) memory, where N = seq_len / L
              and L = segment length. At N=8, d=512: ~8MB per CMS level.
              GRM gate: one similarity pass over cached key projections per segment
              per forward call — O(N × d) per gate activation.
              Tape overhead: one arena allocation per segment boundary, same pattern
              as existing opaque blocks — zero hot-path allocation mid-segment.
  Trade-off:  RM is lowest-cost (no gate, fixed composition).
              GRM adds gate computation and the W_u connector projection.
              Memory Soup is highest-capacity but O(N²) over cached checkpoints.
              SSC adds a learnable selection mask — O(N) overhead but requires
              careful initialization to avoid premature sparse collapse.
  Position:   specs/memory_caching/00_mc_integration.md
  Source:     arXiv:2602.24281 (Memory Caching for Efficient Sequence Modeling).
              Ablation: Table 3, u_t/q_t separation (GRM gate collapse study).
              CS-32 (observe-then-advance), CS-42 (no gradient checkpointing).
              Tape infrastructure: specs/infrastructure/differentiation/01_wengert_tape.md
              Observation layer:  specs/infrastructure/differentiation/02_tape_observation.md
```

**Version**: 0.4.0
**Repository**: NL_Hecate
**Paper**: arXiv:2602.24281
**Status**: Designed — implementation pending tape observation layer completion

---

## 1. What This Is

Memory Caching is the paper's answer to the question: once you have compressed
segment-level checkpoints, how do you use them?

The NL architecture builds compressed representations of prior context through
its inner optimization loop. At segment boundaries — when the model transitions
from one chunk to the next — the MC paper asks what gets carried forward and how.
Four strategies are defined:

| Variant | Aggregation Strategy | Gate Type | Complexity |
|---|---|---|---|
| **Residual Memory (RM)** | Direct residual addition of prior checkpoint | None | O(L) |
| **Gated Residual Memory (GRM)** | Residual weighted by content-based gate | u_t similarity gate | O(NL) |
| **Memory Soup** | Full attention over all cached checkpoints | Soft attention | O(N²L) |
| **Sparse Selective Caching (SSC)** | Learned sparse selection over segment index | Learnable mask | O(NL) |

These variants live at the **segment aggregation layer** — above the inner
optimization loop, below the outer loop parameter update. They are not memory
update rules (those live in `MemoryUpdateRule`). They are checkpoint retrieval
policies.

This module does not yet exist in NL_Hecate. What exists is the infrastructure
it requires: the Wengert tape with observation slots, CUDA kernel pairs for the
inner memory operations, and the CMS level scheduling. The MC variants are the
next layer to build.

---

## 2. Implementation Status

### COMPLETE — Tape Core

The Wengert tape core is operational. Per `specs/infrastructure/differentiation/`:

```text
✓ TapeBuf with arena storage (BufId-indexed)
✓ 17 VJP operations registered via OpaqueKey enum
✓ Full saved[] arrays per opaque block
✓ CUDA ops registered as opaque backward functions
✓ Kernel pairs: DGD, gate, SwiGLU, and others
```

### IN PROGRESS — Tape Observation Layer

Per `specs/infrastructure/differentiation/02_tape_observation.md`:

```text
◐ DeltaRuleLevel0..DeltaRuleLevel3 — per-level key registration
◐ SavedBufferMetadata (role + level fields on TapeBuf)
◐ DGD delta as named intermediate (alloc_named with obs::DGD_DELTA)
◐ Query API: enumerate_opaque_blocks(), find_opaque_at_level(),
             get_saved_by_role(), opaque_output_grad_norm()
```

The MC module depends on this layer being complete. Segment boundary
snapshots use the same `alloc_named` / per-level `OpaqueKey` pattern.
Do not begin MC variant implementation until Class 4 tape observation
tests pass (see §8 of `02_tape_observation.md`).

### PENDING — This Module

All four MC variants are designed but not implemented:

```text
✗ MemoryCachingRule trait (segment_forward + aggregate signatures)
✗ OpaqueKey variants for MC operations
✗ SegmentCache data structure (segment_idx → BufId mapping)
✗ ResidualMemory (RM) — simplest, lowest-risk starting point
✗ GatedResidualMemory (GRM) — u_t connector projection + similarity gate
✗ MemorySoup — full checkpoint attention
✗ SparseSelectiveCaching (SSC) — learnable selection mask
✗ Segment boundary detection in traced_forward.rs
✗ MC tape observation: segment boundary snapshots, gate weight logging
✗ mc_equations extraction (MC paper equations into HADES graph)
```

---

## 3. Trait Interface

Every MC variant implements one trait:

```rust
pub trait MemoryCachingRule {
    /// Forward pass for one segment. Records segment checkpoint on tape.
    /// Returns BufId of the compressed checkpoint (added to segment cache).
    fn segment_forward(
        &self,
        tape: &mut Tape,
        m_state: BufId,       // current M matrix (inner loop output for this segment)
        segment_idx: usize,   // which segment this is (0-indexed)
        level: usize,         // CMS level (0..k-1)
    ) -> BufId;               // checkpoint BufId

    /// Aggregate cached checkpoints into current context.
    /// Returns BufId of the aggregated representation.
    fn aggregate(
        &self,
        tape: &mut Tape,
        cache: &SegmentCache,       // all prior checkpoint BufIds for this level
        query: BufId,               // q_t — for content-based retrieval
        connector: Option<BufId>,   // u_t — GRM only (NEVER aliased to query)
        level: usize,
    ) -> BufId;

    /// OpaqueKey for this variant's tape registration.
    fn opaque_key(&self) -> OpaqueKey;
}
```

The `connector: Option<BufId>` field enforces u_t/q_t separation at the API
level. GRM passes `Some(u_t_buf)`. RM and Soup pass `None`. If `connector` and
`query` resolve to the same `BufId`, the backward adapter panics — aliased inputs
break the VJP computation graph.

---

## 4. Critical Constraints

### 4.1 u_t / q_t Separation (GRM Hard Constraint)

The GRM gate computes:

```text
gate_i = similarity(u_t, MeanPooling(K_i))
```

where `u_t` is a **connector vector projection** (separate learned matrix W_u)
and `K_i` is the mean-pooled key matrix for segment i.

`u_t` is NOT `q_t`. The paper's ablation (Table 3) shows that sharing these
projections produces **zeros across every reported metric** — not performance
degradation, but complete model collapse.

Enforcement:

```text
CS-SHARED-U-Q: GRM aggregate() connector argument must resolve to a BufId
distinct from the query argument. If they are the same buffer, the backward
adapter panics with: "GRM u_t and q_t aliased — model collapse guaranteed".
The OpaqueKey::GatedResidualMemory VJP block's saved[] array must contain
BOTH u_t_buf and q_t_buf as distinct entries for VJP correctness verification.
```

This will be registered as a formal code smell in `hope_code_smells` when
the `02_gated_residual_memory.md` component spec is finalized.

### 4.2 Segment Boundary Snapshots Required

Every call to `segment_forward()` must record a `TapeOp::Opaque` block:

```text
CS-NO-SEGMENT-SNAPSHOT: Any MC variant that skips tape recording at a segment
boundary makes that segment's checkpoint invisible to backward — gradient cannot
flow through cache access that wasn't recorded. This applies even for frozen
segments: a frozen segment records an opaque block with zero gradient output,
not silence. Silence is a correctness bug.
```

### 4.3 Content-Based vs. Position-Based Gating

GRM and SSC both involve selection. The distinction matters for correctness:

```text
GRM: content-based gate — gate_i = f(u_t, K_i)
     Gate output varies with query. Same cache, different u_t → different weights.
     This is learned content sensitivity.

SSC: position-based mask — mask = g(segment_idx)
     Selection is over segment indices, not content.
     SSC does NOT have a query-dependent gate.
```

Position-only gating in GRM (where `gate_i = f(i)` with no `u_t` dependency)
is a correctness failure that passes the type system. Test Class 5 (§7) includes
a gate-dependency probe: perturbing `u_t` must change `gate_i` by at least ε.

---

## 5. Component Specs

Individual variant specs to be created in this directory:

| File | Variant | Status |
|---|---|---|
| `01_residual_memory.md` | Residual Memory (RM) | Not written |
| `02_gated_residual_memory.md` | Gated Residual Memory (GRM) | Not written |
| `03_memory_soup.md` | Memory Soup | Not written |
| `04_sparse_selective_caching.md` | Sparse Selective Caching (SSC) | Not written |

**Implementation order**: RM → GRM → Soup → SSC.

RM validates the segment boundary infrastructure before any gate or attention
mechanism is added. GRM adds the u_t constraint. Soup adds full checkpoint
attention. SSC adds the sparsity schedule — highest risk of premature collapse.

---

## 6. IS / IS NOT Compliance

| IS Container Item | MC Compliance | Notes |
|---|---|---|
| New learning paradigm | ✓ | Segment caching ≠ KV-cache or sliding-window |
| Nested, multi-level, parallel optimization | ✓ | Segment boundaries at higher frequency than token steps |
| Each level with its own context flow | ✓ | Each CMS level maintains its own segment cache |
| Compressing its own context flow | ✓ | Checkpoints ARE compressed M states, not raw token histories |
| In-context learning naturally emerges | ✓ | Gate weights are learned, not hand-tuned retrieval rules |
| Optimizers ARE associative memory modules | ✓ | The checkpoint IS the optimizer state at segment end |
| Self-modifying learning module | ✓ | Gate parameters updated via outer loop |
| Continuum memory system | ✓ | MC adds segment-level continuity above per-token inner loop |

| IS NOT Container Item | MC Compliance | Notes |
|---|---|---|
| NOT single-level optimization | ✓ | Each CMS level has its own segment cache |
| NOT shared/global context flow | ✓ | Per-level caches are never merged across levels |
| NOT static/fixed update rules | ✓ | GRM gate and SSC mask are both learned |
| NOT discrete long/short-term memory | ✓ | Segment caches are compressed checkpoints, not fixed slots |
| NOT optimizers as just optimizers | ✓ | Checkpoint retrieval IS optimization |

---

## 7. Test Classes

Per the NL_Hecate testing convention (Demand / Guarantee / Numerical):

### Test Class 5: MC Correctness

```rust
// Demand: invalid segment_idx panics
fn test_segment_idx_out_of_range_panics()

// Guarantee: segment_forward returns a BufId that appears in tape.ops
fn test_segment_forward_records_on_tape()

// Guarantee: aggregate output shape matches M state shape
fn test_aggregate_output_shape()

// Numerical: RM aggregate is exactly prev_checkpoint + current_m
fn test_rm_aggregate_exact_residual()

// Numerical: GRM gate ∈ [0, 1] for all inputs
fn test_grm_gate_bounded()

// Numerical: perturbing u_t changes GRM gate (content sensitivity probe)
fn test_grm_gate_depends_on_connector_not_just_position()

// Demand: GRM with aliased connector == query panics with CS-SHARED-U-Q message
fn test_grm_aliased_u_t_q_t_panics()

// Guarantee: enumerate_opaque_blocks() includes MC variant keys after forward
fn test_mc_ops_visible_on_tape()
```

---

## 8. Equations (Pending Extraction)

The MC paper (2602.24281) equations have not yet been extracted into an
`mc_equations` collection. Until extraction is complete, references use PDF
section markers directly.

Key equations required for implementation:

| Approx. Location | Description | Variant |
|---|---|---|
| MC §3, Eq. (4) | Segment checkpoint definition | All |
| MC §3.1, Eq. (7) | Residual Memory aggregation | RM |
| MC §3.2, Eq. (9) | GRM gate: similarity(u_t, MeanPool(K_i)) | GRM |
| MC §3.2, Eq. (10) | GRM weighted aggregation | GRM |
| MC §3.3, Eq. (14) | Memory Soup attention over checkpoints | Soup |
| MC §3.4, Eq. (17) | SSC selection mask | SSC |

Equation extraction into `mc_equations` is required before formal signature
registration in `hecate_specs` for the component specs (§5). Extraction uses
the standard methodology (see `methodology/versions/METHODOLOGY_V4.2.md`).

---

## 9. Contribution Opportunities

This module is the current build frontier. The tape infrastructure is being
finalized; the MC variants are the next implementation layer.

Entry points ranked by prerequisite depth:

| Task | Prerequisite | Difficulty |
|---|---|---|
| Draft `01_residual_memory.md` component spec | This file + MC paper §3.1 | Low |
| Extract MC equations into `mc_equations` | HADES extraction methodology | Medium |
| Implement `SegmentCache` struct | `core/src/tape.rs` familiarity | Medium |
| Implement `ResidualMemory` + tape registration | Tape observation layer complete | Medium |
| Implement `GatedResidualMemory` with u_t separation | RM working + ablation verified | High |
| Write Test Class 5 (MC correctness suite) | At least RM implemented | Medium |
| Implement `MemorySoup` | GRM working | High |
| Implement `SparseSelectiveCaching` | Soup working + sparsity schedule design | High |

Access to the HADES graph (ArangoDB + MCP server) is available to active
contributors via the shared endpoint. Contact the project maintainer.

---

## 10. HADES Spec Registration

```json
{
  "_key": "memory-caching-module",
  "title": "Memory Caching Module — Integration Contract",
  "category": "module",
  "version": "0.4.0",
  "path": "specs/memory_caching/00_mc_integration.md",
  "purpose": "Four MC aggregation variants (RM, GRM, Soup, SSC) from arXiv:2602.24281 — segment checkpoint retrieval and composition strategies for NL_Hecate",
  "paper_source": ["2602.24281"],
  "traced_to_equations": [],
  "traced_to_axioms": [
    "hope_axioms/IS",
    "hope_axioms/IS_NOT"
  ],
  "depends_on_specs": [
    "hecate_specs/wengert_tape",
    "hecate_specs/tape-observation"
  ],
  "status": "designed",
  "implementation_status": {
    "tape_infrastructure": "complete",
    "tape_observation": "in_progress",
    "mc_variants": "pending"
  }
}
```
