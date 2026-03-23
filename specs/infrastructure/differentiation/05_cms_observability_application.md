# CMS Observability Application — Unified API, Accumulator, Sidecar

```text
CONTRACT
  Purpose:    Consolidate four near-duplicate PyO3 tape summary methods into one
              unified schema, then build the Python-side accumulator that consumes
              it to provide CMS evolution data across training steps — flushed as
              .cms.json sidecar files at each checkpoint.

  Expects:    - Spec 48 (02_tape_observation.md) shipped: alloc_named on all 9
                adapters, LevelSummary with m_norm/freq_gate_value/is_frozen,
                unified extract_level() helper in tape_summary.rs
              - python/src/lib.rs has 4 PyO3 tape summary methods:
                  tape_forward_summary (CPU single-block)
                  gpu_tape_forward_summary (GPU single-block)
                  cpu_stacked_tape_summary (CPU multi-block)
                  gpu_stacked_tape_summary (GPU multi-block)
                Each independently constructs Python dicts with overlapping but
                inconsistent field sets
              - python/engine/loop.py checkpoint event (save_every) at lines
                1415-1737 with existing .cursor.json sidecar pattern
              - CS-32: observe-then-advance — tape summary is read-only diagnostic
              - CS-40: opt-in AD — tape only active during summary call
              - CS-10: no train/eval distinction — accumulator collects from the
                same build stream

  Guarantees: - Single TapeSummaryResult schema consumed by all Python code
              - CmsTape accumulator bounded to O(checkpoint_interval × k) memory
              - .cms.json sidecar written atomically at each checkpoint
              - Live probe returns current accumulator state without clearing
              - L0 sampling configurable; L1-L3 always full density
              - Zero impact on training loop hot path (summary calls already gated
                by save_every)
              - Backward-compatible: existing print_tape_summary and JSONL logging
                continue to work unchanged

  Cost:       - PyO3 consolidation: refactor, no runtime cost change
              - Accumulator: O(1) dict append per tape call, bounded by interval
              - Sidecar write: one JSON serialize per checkpoint (~300KB-4MB)
              - Live probe: dict copy, no tape replay

  Trade-off:  PyO3 consolidation introduces a breaking internal refactor of lib.rs
              but external Python API (method names, dict schema) stays stable.
              Accumulator trades host memory (~4MB worst case at k=4, heads=16,
              8192-step interval) for cross-step CMS evolution visibility.

  Position:   specs/infrastructure/differentiation/05_cms_observability_application.md
  Source:     HOPE (2512.24695) Eq. 97 (CMS chain) — per-level structure that
              the accumulator tracks over time.
              Titans (2501.00663) Eq. 13 (forgetting gate) — alpha/theta dynamics
              that the sidecar captures.
              CS-32 (observe-then-advance), CS-40 (opt-in AD), CS-10 (no train/eval).
```

---

## 0. Motivation

Spec 48 delivered per-level Rust primitives: every CMS level's M-norm, freeze
status, frequency gate value, and gradient norm are now queryable from a single
tape forward+backward call. But these primitives return data for **one invocation**
— they have no memory across steps.

To understand CMS level evolution during training (Does L2's M-norm plateau?
Does the frequency gate for L3 converge to a stable value? Is a frozen level's
M going stale?), we need infrastructure that:

1. **Accumulates** per-level metrics across steps
2. **Persists** them alongside checkpoints
3. **Exposes** a live read API for probes and notebooks

Before building that accumulator, the PyO3 surface must be unified. The current
4 near-duplicate methods independently construct Python dicts with overlapping
but inconsistent field sets — the GPU stacked path has `m_shard_diff`,
`dormancy_status`, `alpha`/`theta`/`eta` stats that the CPU paths lack. Building
an accumulator on top of this fragmented surface would inherit all inconsistencies.

---

## 1. Phase 1: PyO3 Consolidation (4 → 1 Unified Schema)

### 1.1 Problem

Four methods in `python/src/lib.rs`:

| Method | Lines | Device | Model | Extra fields |
|--------|-------|--------|-------|-------------|
| `tape_forward_summary` | 2579-2649 | CPU | single | baseline |
| `gpu_tape_forward_summary` | 2660-2780 | GPU | single | theta stats |
| `cpu_stacked_tape_summary` | 3358-3470 | CPU | stacked | blocks array |
| `gpu_stacked_tape_summary` | 3553-3881 | GPU | stacked | m_shard_diff, dormancy, alpha/theta/eta |

Each builds its own `PyDict` inline with copy-pasted key insertion. When spec 48
added `m_norm`/`freq_gate_value`/`is_frozen`, each method needed independent
edits. This scales poorly.

### 1.2 Solution: Canonical Schema + Builder

**Step 1: Define canonical dict schema (Python-facing, unchanged):**

```python
{
    "loss": float,
    "n_blocks": int,          # 1 for single-block models
    "total_blocks": int,      # count of active CMS blocks
    "levels": [               # aggregated per-level (backward compat)
        {
            "level": int,
            "opaque_key": str,
            "block_count": int,
            "output_grad_norm": float,
            "dgd_delta_norm": float,
            "m_norm": float,
            "freq_gate_value": float,   # NaN for Fixed schedule
            "is_frozen": bool,
            # Optional (GPU paths with cache access):
            "m_shard_diff": float,
            "m_shard_diff_relative": float,
            "dormancy_status": str,
            "dormancy_below_count": int,
            "alpha": {"count", "min", "max", "mean", "median", "p95", "p99", "frac_at_floor"},
            "theta": {"count", "min", "max", "mean", "median", "p95", "p99", "frac_at_ceil"},
            "eta":   {"count", "min", "max", "mean", "median", "p95", "p99"},
        }
    ],
    "blocks": [               # per-block detail (stacked models only)
        {
            "block_index": int,
            "levels": [...]   # same per-level schema
        }
    ]
}
```

**Step 2: Refactor lib.rs — extract `build_tape_summary_dict()`:**

A single Rust helper function that takes structured input and builds the PyDict:

```rust
fn build_tape_summary_dict(
    py: Python<'_>,
    loss: f32,
    n_blocks: usize,
    // Per-level data: Vec of (level_idx, LevelMetrics)
    aggregated_levels: &[LevelMetrics],
    // Per-block data: Option for stacked models
    block_levels: Option<&[Vec<LevelMetrics>]>,
) -> PyResult<PyObject>
```

Where `LevelMetrics` is a plain struct:

```rust
struct LevelMetrics {
    level: usize,
    opaque_key: String,
    block_count: usize,
    output_grad_norm: f32,
    dgd_delta_norm: f32,
    m_norm: f32,
    freq_gate_value: f32,
    is_frozen: bool,
    // Optional extended fields (None = omit from dict)
    m_shard_diff: Option<f32>,
    m_shard_diff_relative: Option<f32>,
    dormancy_status: Option<String>,
    dormancy_below_count: Option<i32>,
    alpha_stats: Option<GateStats>,
    theta_stats: Option<GateStats>,
    eta_stats: Option<GateStats>,
}
```

**Step 3: Each method becomes a thin wrapper:**

Each of the 4 existing methods retains its name and signature (backward compat)
but delegates dict construction to `build_tape_summary_dict()`. The per-method
logic is reduced to: (a) run the appropriate forward/backward path, (b) collect
`LevelMetrics` from whatever data source that path provides, (c) call the shared
builder.

### 1.3 Files Modified

| File | Change |
|------|--------|
| `python/src/lib.rs` | Extract `LevelMetrics` struct, `GateStats` struct, `build_tape_summary_dict()`. Refactor 4 methods to use shared builder. Net code reduction ~200 lines. |

### 1.4 Invariants

- **No Python API change.** Method names, parameter signatures, and returned dict
  keys are identical. Existing `print_tape_summary()` and JSONL logging work
  without modification.
- **No Rust core changes.** This is pure PyO3-tier refactoring.
- **Fields present in one path but not others** (e.g., `alpha` stats only in GPU
  stacked) remain conditionally present — `build_tape_summary_dict` omits keys
  with `None` values rather than inserting NaN/defaults.

### 1.5 Tests

- Existing Python tests that call tape summary methods must pass unchanged.
- New test: call all 4 methods on same input, verify shared keys produce
  identical values (within tolerance for GPU vs CPU float differences).

---

## 2. Phase 2: CmsTape Accumulator

### 2.1 Class Design

```python
class CmsTape:
    """Accumulates per-level CMS metrics across training steps."""

    def __init__(self, k: int, n_blocks: int, num_heads: int,
                 chunk_sizes: list[int],
                 l0_sample_rate: float = 0.125,
                 l0_per_head_sample_rate: float = 0.0625):
        ...

    def record(self, tape_summary: dict, step: int) -> None:
        """Append one tape summary to the accumulator.

        Respects sampling rates: L0 records at l0_sample_rate,
        L1-L3 always record (full density).
        """
        ...

    def flush(self, from_step: int, to_step: int) -> dict:
        """Return accumulated data as sidecar-ready dict and reset."""
        ...

    def probe(self) -> dict:
        """Return current accumulator state without clearing.

        Read-only view for live probes, Jupyter, callbacks.
        """
        ...

    def __len__(self) -> int:
        """Number of recorded samples."""
        ...
```

### 2.2 Accumulator Storage

Per level, the accumulator stores parallel arrays (one entry per recorded step):

```python
{
    "steps": [int],              # step indices where this level fired
    "m_norm": [float],           # Frobenius norm of M
    "output_grad_norm": [float], # ‖d_y‖_2
    "dgd_delta_norm": [float],   # ‖M@k - v‖_2
    "freq_gate_value": [float],  # sigmoid output (NaN for Fixed)
    "is_frozen": [bool],         # frozen flag
    # Per-block (stacked models):
    "blocks": [
        {
            "m_norm": [float],
            "output_grad_norm": [float],
            "dgd_delta_norm": [float],
            # Extended (when available):
            "m_shard_diff": [float],
            "alpha_mean": [float],
            "theta_mean": [float],
        }
    ]
}
```

### 2.3 Sampling

- **L0**: Fires every token. At 8192-step checkpoint intervals, that's 8192
  samples. Sample at `l0_sample_rate` (default 1/8) → 1024 recorded.
  Per-head data at `l0_per_head_sample_rate` (default 1/16) → 512 recorded.
- **L1** (every 8th): 1024 firings per interval → full density.
- **L2** (every 64th): 128 firings → full density.
- **L3** (every 512th): 16 firings → full density.

Sampling uses deterministic modular arithmetic (`step % sample_period == 0`),
not random sampling, for reproducibility.

### 2.4 Memory Budget

At 8192-step checkpoint interval, k=4, n_blocks=4, num_heads=16:

| Level | Firings | Sampled | Aggregate bytes | Per-head bytes |
|-------|---------|---------|-----------------|----------------|
| L0 | 8192 | 1024 | ~30KB | ~480KB |
| L1 | 1024 | 1024 | ~30KB | ~480KB |
| L2 | 128 | 128 | ~4KB | ~60KB |
| L3 | 16 | 16 | ~0.5KB | ~8KB |
| **Total** | | | **~65KB** | **~1MB** |

With 4 blocks × 6 metrics per block: ~4MB total worst case. Acceptable.

---

## 3. Phase 2 (cont.): Sidecar Format

### 3.1 Filename Convention

```
model_step5000.safetensors.cms.json
```

Alongside the existing:
```
model_step5000.safetensors
model_step5000.safetensors.cursor.json
```

### 3.2 Schema

```json
{
    "version": 1,
    "from_step": 0,
    "to_step": 5000,
    "token_count": 2560000,
    "k": 4,
    "chunk_sizes": [1, 8, 64, 512],
    "n_blocks": 4,
    "num_heads": 16,
    "l0_sample_rate": 0.125,
    "levels": [
        {
            "level": 0,
            "firings": 5000,
            "sampled": 625,
            "steps": [0, 8, 16, ...],
            "m_norm": [12.3, 12.4, ...],
            "output_grad_norm": [0.05, 0.04, ...],
            "dgd_delta_norm": [0.02, 0.02, ...],
            "freq_gate_value": [0.95, 0.95, ...],
            "is_frozen": [false, false, ...],
            "blocks": [
                {
                    "block_index": 0,
                    "m_norm": [...],
                    "output_grad_norm": [...],
                    "dgd_delta_norm": [...]
                }
            ]
        }
    ]
}
```

### 3.3 Write Path

In `loop.py`, immediately after the existing `.cursor.json` write:

```python
if cms_tape is not None:
    cms_sidecar_path = str(ckpt_path) + ".cms.json"
    cms_data = cms_tape.flush(from_step=last_flush_step, to_step=step)
    with open(cms_sidecar_path, "w") as f:
        json.dump(cms_data, f)
    last_flush_step = step
```

---

## 4. Phase 2 (cont.): Consumption Modes

### 4.1 Checkpoint Mode (default, always-on)

Integrated into the existing checkpoint event in `loop.py`:

1. At each `save_every` step, tape summary is already called (lines 1530-1573)
2. `cms_tape.record(tape_sum, step)` appends to accumulator
3. After checkpoint save, `cms_tape.flush()` writes sidecar and resets

This is **zero additional config** — if `tape_device != "off"`, the accumulator
records. If `save_every > 0`, the sidecar is written.

### 4.2 Live Probe Mode

```python
# From Jupyter, callback, or coherence probe:
current_state = gpu_model.cms_tape()  # returns dict, does NOT clear
```

This calls `cms_tape.probe()` which returns a copy of the current accumulator
state. The accumulator is only cleared by `flush()` at checkpoint time.

The `cms_tape()` method is added to `PyGpuModel` as a thin Python-side method
(not a PyO3 Rust method — it just returns the Python accumulator's state).

---

## 5. Phase 2 (cont.): Training Loop Integration

### 5.1 Initialization

In `loop.py`, after model and config are set up:

```python
from engine.cms_tape import CmsTape

cms_tape = None
if bcfg.tape_device != "off" and bcfg.save_every > 0:
    cms_tape = CmsTape(
        k=bcfg.k,
        n_blocks=getattr(bcfg, 'n_blocks', 1),
        num_heads=bcfg.num_heads,
        chunk_sizes=bcfg.chunk_sizes,
    )
    last_flush_step = 0
```

### 5.2 Recording

Inside the existing tape diagnostic block (lines 1530-1573):

```python
if tape_sum is not None and cms_tape is not None:
    cms_tape.record(tape_sum, step)
```

### 5.3 Flushing

Inside the checkpoint save block, after `.cursor.json` write:

```python
if cms_tape is not None and len(cms_tape) > 0:
    cms_path = str(ckpt_path) + ".cms.json"
    with open(cms_path, "w") as f:
        json.dump(cms_tape.flush(last_flush_step, step), f)
    last_flush_step = step
```

### 5.4 New File

```
python/engine/cms_tape.py    # CmsTape class (~150 lines)
```

This is the only new file. All other changes are edits to existing files.

---

## 6. Files Modified

| File | Phase | Change |
|------|-------|--------|
| `python/src/lib.rs` | 1 | Extract `LevelMetrics`, `GateStats`, `build_tape_summary_dict()`. Refactor 4 methods. ~200 line net reduction. |
| `python/engine/cms_tape.py` | 2 | **NEW.** CmsTape class. ~240 lines. |
| `python/engine/loop.py` | 2 | Init CmsTape, record at tape diagnostic, flush at checkpoint. ~15 lines added. |

Files **NOT** modified:
- No Rust core changes (core/src/*)
- No CUDA changes
- No changes to `evaluation.py` (print_tape_summary already handles the dict)
- No changes to existing config schema (CmsTape uses existing config fields)

---

## 7. Testing

### Phase 1 Tests (PyO3 consolidation)

1. **Schema parity test**: Call all 4 methods on same model/input, verify shared
   keys produce identical values (within GPU/CPU float tolerance).
2. **Backward compat test**: Existing `print_tape_summary()` works on output from
   each method without modification.

### Phase 2 Tests (CmsTape)

3. **Round-trip test**: Accumulate known data → flush → json.loads → verify values
   match input exactly.
4. **Sampling test**: Record 100 L0 steps at 1/8 rate → verify 12-13 samples
   stored. Record 100 L1 steps → verify all 100 stored.
5. **Probe-doesn't-clear test**: Record 10 steps → probe() → verify len unchanged
   → flush() → verify len == 0.
6. **Sidecar file test**: Simulate checkpoint flow → verify .cms.json file written
   with correct schema, from_step/to_step, level arrays.

---

## 8. Dependency Graph

```
Phase 1 (PyO3 consolidation) → Phase 2 (CmsTape + sidecar + loop)
```

Phase 1 must land first because the accumulator consumes the unified dict schema.
If the schema is fragmented, every accumulator method needs per-path special cases.

---

## 9. Risk to Running Experiments

**Zero.** Phase 1 is a refactor — same methods, same signatures, same output.
Phase 2 adds an optional accumulator that is only active when `tape_device != "off"`
and `save_every > 0` (both already true for current experiments). The accumulator
appends to a Python list — it cannot affect the training loop's forward/backward/
step_adamw hot path.

The sidecar write is a single `json.dump` at checkpoint time (~1ms for 4MB).
Negligible compared to the safetensors write.

---

## 10. HADES Registration

```json
{
    "_key": "cms-observability-application",
    "title": "CMS Observability Application — Unified API, Accumulator, Sidecar",
    "category": "infrastructure",
    "version": "0.4.0",
    "path": "specs/infrastructure/differentiation/05_cms_observability_application.md",
    "purpose": "Consolidate 4 PyO3 tape summary methods into 1 schema, build CmsTape accumulator with .cms.json sidecar persistence and live probe API",
    "paper_source": ["2512.24695", "2501.00663"],
    "traced_to_equations": [
        "hope_equations/eq-097-hope-cms-chain",
        "titans_equations/eq-013-forgetting-gate"
    ],
    "traced_to_axioms": [],
    "status": "v0.4.0"
}
```

Trace edges:
- `hecate_specs/cms-observability-application` → `hope_equations/eq-097-hope-cms-chain` (implements: CMS per-level structure)
- `hecate_specs/cms-observability-application` → `titans_equations/eq-013-forgetting-gate` (cites: alpha/theta dynamics)
