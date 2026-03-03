# Tape Observation Infrastructure — Native Interpretability Layer

```text
CONTRACT
  Purpose:    Extend the Wengert tape with named observation slots that expose
              per-level gradient flow, the DGD self-modification delta, and
              saved-buffer semantics — without disrupting backward correctness
              or imposing overhead when disabled. This is interpretability
              designed into the AD infrastructure, not bolted on afterwards.
  Expects:    core/src/tape.rs — TapeBuf, TapeOp::Opaque, OpaqueKey, with_tape()
              — all implemented and passing Class 1-3 tests per 01_wengert_tape.md.
              All 8 memory rules registered as opaque VJP blocks.
              OpaqueKey::DGD already registered in the tape.
              CMS k=4 forward records one TapeOp::Opaque per level call site
              (the level call site already passes through record_on_tape()).
  Guarantees: All existing tape tests continue to pass unmodified.
              Observation slots add zero runtime overhead when not queried
              (metadata lives on TapeBuf, which is already allocated).
              Per-level observability: each CMS level records a distinct
              (OpaqueKey, level) pair — making it addressable by level index.
              DGD delta (M@k - v error before application) is saved as a named
              buffer in the DGD opaque block's saved array.
              Query API is post-backward only — no hot-path allocation.
  Cost:       SavedBufferMetadata: two Option fields added to TapeBuf — 16 bytes
              per buffer, zero compute overhead.
              Per-level key: one additional usize field on TapeOp::Opaque —
              8 bytes per op, zero compute overhead.
              DGD delta buffer: already computed inside step() — saving it adds
              one arena allocation (d² f32s, ~1MB at d=512). Negligible.
              Query API: O(ops) linear scan, called post-backward, never on hot path.
  Trade-off:  Named metadata on tape buffers is a weak form of provenance
              (string role tag, not a typed enum). Sufficient for observation;
              not a general tensor naming system.
              Per-level keys require the call site to pass level: usize into
              record_on_tape() — a one-line change per rule adapter.
              DGD delta exposure means the DGD adapter must save the error
              buffer explicitly rather than computing it inline and discarding it.
  Position:   specs/infrastructure/differentiation/02_tape_observation.md
  Source:     HOPE (2512.24695) Eq. 88 (practical DGD update) for delta definition.
              HOPE (2512.24695) Eq. 97 (CMS chain) for per-level structure.
              task_4acd05 exploration_note (2026-02-26) for gap analysis.
              CS-40 (opt-in AD), CS-42 (all intermediates stored),
              CS-47 (no in-place mutation of tracked tensors).
```

---

## 1. Problem: Three Gaps in the Current Tape

The Wengert tape records the computation graph with full fidelity. However
three structural gaps prevent native observation of the quantities that make
NL interpretable:

### Gap 1 — Saved buffers have no semantic names

Every `TapeOp::Opaque` block saves tensors in a positional `Vec<BufId>`:

```rust
Opaque { key, inputs, outputs, saved: vec![buf_0, buf_1, buf_2] }
//                                         ^       ^       ^
//                                    embedded  params   cache_flat
//                         (which is which requires reading the adapter source)
```

`saved[2]` might be `m_states`, `k_mem`, or `error` depending on which rule
registered this block. There is no metadata. A query like "show me the M state
after level 1 at step t" requires knowing the adapter's positional convention.

### Gap 2 — All CMS levels produce a single indistinguishable `TapeOp::Opaque`

At k=4, `cms_forward()` calls `record_on_tape()` four times — one per level.
All four produce `TapeOp::Opaque { key: OpaqueKey::DeltaRule, ... }`. The tape
has no way to distinguish Level 0 from Level 3 without tracing call order, which
is fragile and breaks if level ordering changes.

### Gap 3 — The DGD self-modification delta is never a named intermediate

Inside `step()`, the DGD adapter computes:

```rust
let error = matmul(m_prev, k) - v;   // M @ k - v  ← this IS the delta (HOPE Eq. 88)
let delta_m = outer(error, k);       // error ⊗ k
let m_new = m_prev * (I - eta * outer(k, k)) - eta * delta_m;
```

The `error` tensor — the self-modification signal — is computed, used, and
discarded. It is never saved to the tape arena. Observing the DGD delta
requires re-running the forward pass with extra instrumentation.

---

## 2. Solution: Three Minimal, Additive Changes

None of these changes modify the backward pass or existing VJP rules. They are
pure metadata additions and one saved-buffer registration.

### 2.1 `SavedBufferMetadata` — named roles on tape buffers

Add two optional fields to `TapeBuf`:

```rust
pub struct TapeBuf {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub is_param: bool,
    // ── observation metadata (new fields) ──
    /// Semantic role of this buffer within its opaque block.
    /// Examples: "m_states", "error", "k_mem", "dgd_delta", "alpha", "theta".
    /// None for unnamed intermediates (standard ops, unnamed saved tensors).
    pub role: Option<&'static str>,
    /// CMS level index (0..k-1) this buffer belongs to.
    /// None for buffers that are not level-specific.
    pub level: Option<usize>,
}
```

All existing allocations default `role: None, level: None` — no behavioral
change. Named buffers are allocated via a new `alloc_named` method:

```rust
impl Tape {
    /// Allocate a named buffer. role and level are observation metadata only.
    pub fn alloc_named(
        &mut self,
        data: Vec<f32>,
        shape: Vec<usize>,
        role: &'static str,
        level: Option<usize>,
    ) -> BufId {
        let id = self.bufs.len();
        self.bufs.push(TapeBuf { data, shape, is_param: false,
                                  role: Some(role), level });
        self.grad_accum.push(None);
        id
    }
}
```

### 2.2 Per-level observability — `level` field on `TapeOp::Opaque`

Add a `level: Option<usize>` field to `TapeOp::Opaque`:

```rust
Opaque {
    key: OpaqueKey,
    inputs: Vec<BufId>,
    outputs: Vec<BufId>,
    saved: Vec<BufId>,
    level: Option<usize>,   // ← new field; None for non-CMS blocks (SWA, DGD)
},
```

The `record_opaque` method gains a `level` parameter with a default:

```rust
pub fn record_opaque(
    &mut self,
    key: OpaqueKey,
    inputs: Vec<BufId>,
    outputs: Vec<BufId>,
    saved: Vec<BufId>,
    level: Option<usize>,   // ← new; pass Some(l) from per-level call sites
) {
    ...
    self.record(TapeOp::Opaque { key, inputs, outputs, saved, level });
}
```

Every existing call site passes `None` (backward compatible). The CMS adapter
in `traced_forward.rs` passes `Some(level_idx)`:

```rust
// In traced_cms_forward(), for each level l in 0..k:
rule.record_on_tape(tape, &level_params, embedded, seq_len, d, initial_m,
                    Some(l));   // ← the one-line change per call site
```

`record_on_tape()` in the `OpaqueVjp` trait gains a `level: Option<usize>`
parameter, which it forwards to `record_opaque`.

### 2.3 DGD delta as named intermediate

The DGD backward adapter already receives the `error` tensor in its `saved`
array (required for its VJP). The only change is to allocate it via
`alloc_named` instead of `alloc`:

```rust
// In DGD's record_on_tape():
let error = matmul(&m_prev, &k) - &v;   // M @ k - v  (HOPE Eq. 88)
let error_buf = tape.alloc_named(
    error.clone(),
    vec![d],
    "dgd_delta",
    level,   // propagated from record_on_tape parameter
);
// error_buf is added to saved[] as before — backward adapter is unchanged
```

The backward adapter's positional convention does not change: `saved[N]` is
still `error`. The new metadata is carried alongside — the adapter is not aware
of it.

---

## 3. Query API

Post-backward queries. These are never called during forward or backward —
no hot-path impact. All O(ops) scans through `self.ops`.

```rust
impl Tape {
    /// All opaque ops matching key (any level).
    pub fn find_opaque_ops(&self, key: OpaqueKey) -> Vec<usize> {
        self.ops.iter().enumerate()
            .filter_map(|(i, op)| match op {
                TapeOp::Opaque { key: k, .. } if *k == key => Some(i),
                _ => None,
            })
            .collect()
    }

    /// All opaque ops matching key at a specific CMS level.
    pub fn find_opaque_at_level(&self, key: OpaqueKey, level: usize) -> Vec<usize> {
        self.ops.iter().enumerate()
            .filter_map(|(i, op)| match op {
                TapeOp::Opaque { key: k, level: Some(l), .. }
                    if *k == key && *l == level => Some(i),
                _ => None,
            })
            .collect()
    }

    /// Get the saved buffer with the given role from an opaque op.
    /// Returns None if the op is not Opaque or no saved buffer has that role.
    pub fn get_saved_by_role(&self, op_idx: usize, role: &str) -> Option<&[f32]> {
        match &self.ops[op_idx] {
            TapeOp::Opaque { saved, .. } => {
                saved.iter().find_map(|&id| {
                    if self.bufs[id].role.map_or(false, |r| r == role) {
                        Some(self.bufs[id].data.as_slice())
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    }

    /// Enumerate all opaque blocks: (op_idx, key, level).
    pub fn enumerate_opaque_blocks(&self) -> Vec<(usize, OpaqueKey, Option<usize>)> {
        self.ops.iter().enumerate()
            .filter_map(|(i, op)| match op {
                TapeOp::Opaque { key, level, .. } => Some((i, *key, *level)),
                _ => None,
            })
            .collect()
    }

    /// Get the gradient norm for an opaque block's first output.
    /// Useful for level-boundary gradient flow measurement after backward().
    pub fn opaque_output_grad_norm(&self, op_idx: usize) -> Option<f32> {
        match &self.ops[op_idx] {
            TapeOp::Opaque { outputs, .. } => {
                let out_id = *outputs.first()?;
                let grad = self.grad_accum[out_id].as_deref()?;
                let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
                Some(norm)
            }
            _ => None,
        }
    }
}
```

---

## 4. Named Roles — Canonical List

The following role strings are the canonical saved-buffer names across all adapters.
String constants are defined in `tape.rs` to avoid typos:

```rust
pub mod obs {
    pub const M_STATES:    &str = "m_states";
    pub const ERROR:       &str = "error";       // L2 loss error (for most rules)
    pub const DGD_DELTA:   &str = "dgd_delta";   // DGD: M@k - v  (HOPE Eq. 88)
    pub const K_MEM:       &str = "k_mem";       // key memory projection
    pub const V_MEM:       &str = "v_mem";       // value memory projection
    pub const ALPHA:       &str = "alpha";       // learned decay gate output
    pub const THETA:       &str = "theta";       // softplus-gated learning rate
    pub const EMBEDDED:    &str = "embedded";    // input token embedding
    pub const LEVEL_PARAMS:&str = "level_params";// flattened MemoryLevelParams
}
```

Rules allocate their key saved buffers via `alloc_named` using these constants.
Unnamed saved buffers (temporary intermediates not needed for observation)
continue using `alloc`.

---

## 5. Observation Points per Component

### 5.1 Level-boundary gradient norms (already partially shipped via PR #149)

With per-level `Opaque` keys, gradient norm at each level boundary is directly
queryable post-backward — no additional instrumentation needed:

```rust
// After tape.backward(loss_id):
for level in 0..k {
    let ops = tape.find_opaque_at_level(OpaqueKey::DeltaRule, level);
    for op_idx in ops {
        let norm = tape.opaque_output_grad_norm(op_idx);
        println!("Level {level} output grad norm: {norm:?}");
    }
}
```

### 5.2 DGD self-modification delta

```rust
// After tape.backward(loss_id):
let dgd_ops = tape.find_opaque_ops(OpaqueKey::DGD);
for op_idx in dgd_ops {
    if let Some(delta) = tape.get_saved_by_role(op_idx, obs::DGD_DELTA) {
        let norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("DGD delta norm (HOPE Eq. 88): {norm:.6}");
    }
}
```

### 5.3 M state readout at any level

```rust
// Forward value of M for level l, after tape.backward():
let ops = tape.find_opaque_at_level(OpaqueKey::DeltaRule, 2);
if let Some(op_idx) = ops.first() {
    if let Some(m) = tape.get_saved_by_role(*op_idx, obs::M_STATES) {
        // m is flat [d*d] — reshape as needed
        println!("Level 2 M state has {} elements", m.len());
    }
}
```

### 5.4 Per-level error buffer (frozen levels)

Frozen-level ops use `OpaqueKey::FrozenDeltaRule` (or equivalent frozen variant).
Their `saved` array records `M_FROZEN` and `Q_T`, which flow into ErrorBuffer
accumulation. Per-level keys make it possible to distinguish Level 1 frozen
accumulation from Level 3.

---

## 6. What This Does NOT Change

| Component | Status |
|---|---|
| Backward VJP rules | Unchanged — metadata is read-only |
| `with_tape()` entry point | Unchanged |
| `OpaqueBackwardFn` signature | Unchanged — adapters use positional saved[] as before |
| All existing test classes (1-3) | Unchanged — no behavioral difference |
| GPU path (`gpu_cms_backward`) | Unaffected — tape is CPU/Rust-path only |
| `MemoryRule` trait | Unchanged |
| `OpaqueVjp::record_on_tape` callers | `level` parameter added with `None` default — backward compatible |

---

## 7. Files to Create/Modify

| File | Change |
|---|---|
| `core/src/tape.rs` | Add `role`/`level` to `TapeBuf`; `alloc_named()`; `level` on `TapeOp::Opaque`; `record_opaque()` level param; `obs::*` constants; query API methods |
| `core/src/opaque_adapters.rs` | DGD adapter: save `error` via `alloc_named(..., obs::DGD_DELTA, level)`; other adapters: rename key saved buffers to `alloc_named` |
| `core/src/traced_forward.rs` | Pass `Some(level_idx)` to `record_on_tape()` at each CMS level call site |
| `core/src/tape.rs` (OpaqueVjp trait) | Add `level: Option<usize>` to `record_on_tape()` signature |

No new files required. All changes are additive to existing files.

---

## 8. Testing

### Test Class 4: Observation Correctness

```rust
// Verify that named saved buffers contain the correct values.

#[test]
fn test_dgd_delta_is_l2_error() {
    // Run DGD forward with tape.
    // After backward, get_saved_by_role(op_idx, DGD_DELTA) must equal M@k - v.
    let (m_prev, k_vec, v_vec) = random_dgd_inputs(d=16);
    let expected_delta: Vec<f32> = matmul_ref(&m_prev, &k_vec)
        .iter().zip(&v_vec).map(|(a, b)| a - b).collect();

    let tape_delta = /* ... run DGD through tape, read role ... */;
    assert_allclose!(tape_delta, expected_delta, atol=1e-6,
        "DGD delta in tape != M@k - v (HOPE Eq. 88)");
}

#[test]
fn test_per_level_keys_distinct() {
    // k=4, all levels active. After forward, enumerate_opaque_blocks()
    // must return 4 entries with level = Some(0), Some(1), Some(2), Some(3).
    let blocks = tape.enumerate_opaque_blocks();
    let memory_blocks: Vec<_> = blocks.iter()
        .filter(|(_, key, _)| *key == OpaqueKey::DeltaRule)
        .collect();
    assert_eq!(memory_blocks.len(), 4);
    for (expected_level, (_, _, level)) in memory_blocks.iter().enumerate() {
        assert_eq!(*level, Some(expected_level));
    }
}

#[test]
fn test_grad_norm_query_post_backward() {
    // After backward, opaque_output_grad_norm() must return a finite positive value
    // for active levels and near-zero for frozen levels (no update gradient).
    // Verifies that gradient flow is observable at level boundaries.
}
```

---

## 9. HADES Spec Registration

```json
{
  "_key": "tape-observation",
  "title": "Tape Observation Infrastructure — Native Interpretability Layer",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/differentiation/02_tape_observation.md",
  "purpose": "Named observation slots on the Wengert tape: per-level VJP keys, SavedBufferMetadata, DGD delta exposure, and post-backward query API",
  "paper_source": ["2512.24695"],
  "traced_to_equations": [
    "hope_equations/eq-088-practical-dgd-update",
    "hope_equations/eq-097-hope-cms-chain"
  ],
  "traced_to_axioms": [],
  "status": "v0.4.0"
}
```
