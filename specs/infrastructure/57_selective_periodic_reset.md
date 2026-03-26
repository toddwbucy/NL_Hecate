# Selective Periodic Reset: Per-Level Reset Intervals

<!-- HADES: tnt_equations/eq-006-local-memory-update (TNT §3.2);
            tnt_equations/eq-014-n-local-memories-update (TNT §3.3);
            hope_equations/eq-097-hope-cms-chain (HOPE §7) -->

```text
CONTRACT
  Purpose:    Extend spec 08 (TNT periodic reset) from an all-or-nothing boolean
              to a per-level reset interval. Each CMS level gets an independent
              reset period R_k (in steps). Default behavior: R_k=1 for all levels
              (current spec-08 behavior — every level resets every step it fires).
              Gear-shifting mode: R_0=1, R_1=8, R_2=64, R_3=512, equalizing the
              write budget across levels so each accumulates ~512 writes between
              resets, regardless of its chunk_size.

  Expects:    - Spec 08 (TNT periodic reset) fully implemented: memory_reset="periodic"
                causes all active levels to reset M after each step.
              - Conductor/Pulse delivering per-step active_levels mask.
              - GpuContextState.memory: per-level M buffers.
              - GpuStackedModel and GpuModel with memory_reset boolean field.
              - Python config (engine/config.py) with memory_reset: str field.
              - loop.py orchestration calling periodic_reset_level() after each step.

  Guarantees: - When reset_intervals is absent or all-ones: behavior is IDENTICAL
                to spec-08 periodic mode. No regression.
              - When reset_intervals = [1, 8, 64, 512]: level k is only reset
                every R_k steps where it fires. Between resets, M persists across
                fire boundaries, allowing the level to accumulate writes.
              - The reset decision is: reset level k at step t IFF
                pulse.active_levels[k] AND (fire_count_k % R_k == 0),
                where fire_count_k is the number of times level k has fired
                since the last reset (or since build start).
              - Per-level fire counters are maintained in the Conductor, serialized
                in ConductorState, and restored from checkpoints.
              - Backward compatibility: existing configs without reset_intervals
                behave identically to before (all-ones default).
              - carry_forward mode is unaffected — reset_intervals is ignored
                when memory_reset != "periodic".

  Cost:       - k additional usize counters in Conductor (trivial).
              - One modulo check per level per step (trivial).
              - No CUDA kernel changes — reset is already called from Python tier.
              - Config: one optional array field.

  Trade-off:  With R_k=1 (spec-08 default), each level starts every shard from
              M_init — maximum parallelism, minimum cross-shard memory.
              With R_k > 1, levels accumulate M across R_k shards before reset —
              more memory signal per level, but introduces sequential dependency
              across R_k consecutive shards. The gear-shifting profile [1,8,64,512]
              gives each level ~512 write operations between resets, equalizing
              the effective write budget that spec-08's all-ones profile denies
              to higher levels (L2 gets 8 writes, L3 gets 1 write before reset).

  Position:   specs/infrastructure/57_selective_periodic_reset.md
  Source:     TNT (2511.07343) §3.2 Local Memory with Periodic Reset
                HADES: tnt_equations/eq-006-local-memory-update (reset rule)
                       tnt_equations/eq-014-n-local-memories-update (N local memories)
              CMS gear-shifting analysis (runs/k4_chain_dolmino_d1024_32h/report/
                cms_gear_shifting_analysis.pdf) — empirical evidence that all-ones
                reset starves L2/L3.
  Related:    specs/infrastructure/08_tnt_periodic_reset.md (base periodic reset)
              EPIC task_6ebcb7 / CG-1 task_98bbc2
```

---

## Motivation

The k4_chain_dolmino_d1024_32h seed run (61K steps, 252M tokens) demonstrated that
spec-08's all-or-nothing periodic reset reduces a k=4 CMS hierarchy to effectively
k≈1. At seq_len=512 with chunk_sizes=[1,8,64,512], all four levels fire every step,
and all four are reset to M=0 after every step:

| Level | Chunk size | Writes per step | ‖M‖_F at 61K | θ_eff |
|-------|-----------|-----------------|--------------|-------|
| L0    | 1         | 512             | 0.1014       | 0.367 |
| L1    | 8         | 64              | 1.1e-5       | 0.011 |
| L2    | 64        | 8               | ≈0           | 0.002 |
| L3    | 512       | 1               | ≈0           | 0.001 |

L0 gets 512 write operations per step — enough to build meaningful M signal. L3 gets
exactly 1 write before being zeroed. The model has learned this limitation: L2/L3
gate biases have driven α_eff→1.0 (maximum retention) and θ_eff→0.0 (don't waste
capacity on signal that will be erased).

Selective periodic reset equalizes the write budget by letting each level accumulate
writes across multiple fire events before resetting. With R=[1,8,64,512]:

| Level | R_k | Fires between resets | Writes between resets |
|-------|-----|---------------------|-----------------------|
| L0    | 1   | 1                   | 512                   |
| L1    | 8   | 8                   | 8 × 64 = 512         |
| L2    | 64  | 64                  | 64 × 8 = 512         |
| L3    | 512 | 512                 | 512 × 1 = 512        |

Each level now gets ~512 writes between resets — a uniform budget.

---

## Design

### Reset Decision Rule

At each step t, for each CMS level k:

```text
if pulse.active_levels[k]:
    fire_count[k] += 1
    if fire_count[k] % reset_intervals[k] == 0:
        M_k ← 0          // or M_init_k when learnable-init is implemented
        fire_count[k] = 0 // reset counter
```

The fire counter tracks how many times level k has fired since its last reset
(or since build start). When the counter reaches R_k, the level resets and
the counter returns to zero.

### Conductor Changes

```rust
pub struct Conductor {
    pub k: usize,
    pub chunk_sizes: Vec<usize>,
    step: usize,
    stream: Option<Box<dyn ContextStream>>,
    // NEW: per-level fire counters for selective reset
    fire_counts: Vec<usize>,
    reset_intervals: Vec<usize>,
}

pub struct ConductorState {
    pub k: usize,
    pub chunk_sizes: Vec<usize>,
    pub step: usize,
    // NEW: serialized for checkpoint restore
    pub fire_counts: Vec<usize>,
    pub reset_intervals: Vec<usize>,
}
```

New method on Conductor:

```rust
/// Check whether level k should reset this step.
/// Returns true IFF level k fired AND its fire counter reached R_k.
/// Advances the fire counter as a side effect.
/// Must be called AFTER observe, BEFORE next advance (CS-32).
pub fn should_reset_level(&mut self, level: usize, active: bool) -> bool {
    if !active { return false; }
    self.fire_counts[level] += 1;
    if self.fire_counts[level] >= self.reset_intervals[level] {
        self.fire_counts[level] = 0;
        true
    } else {
        false
    }
}
```

### Python Config Changes

```python
# engine/config.py — BuildConfig
reset_intervals: list[int] | None = None
# None → default [1,1,...,1] (spec-08 behavior)
# [1,8,64,512] → gear-shifting mode
```

Validation:
- If set, length must equal k
- All values must be >= 1
- Ignored when memory_reset != "periodic"

### Python Orchestration Changes (loop.py / lib.rs)

Currently in `python/src/lib.rs` (GpuStackedModel::step_adamw):

```rust
if self.memory_reset {
    for (k, &active) in pulse.inner.active_levels.iter().enumerate() {
        if active {
            self.context.periodic_reset_level(k);
        }
    }
}
```

Changed to:

```rust
if self.memory_reset {
    for (k, &active) in pulse.inner.active_levels.iter().enumerate() {
        if self.conductor.should_reset_level(k, active) {
            self.context.periodic_reset_level(k);
        }
    }
}
```

This requires the GpuModel / GpuStackedModel to either:
(a) own a Conductor reference for reset decisions, or
(b) accept reset_intervals and maintain fire_counts internally.

**Option (b) is simpler** — the GPU model already has `memory_reset: bool`. Replace
with `reset_intervals: Vec<usize>` (empty = no reset). The fire counters are k
usizes stored alongside the model, no GPU memory needed.

### PyO3 Interface Changes

```python
# Current
GpuStackedModel(cfg, n_blocks, seed, batch_size=1, memory_reset=False)

# New (backward compatible)
GpuStackedModel(cfg, n_blocks, seed, batch_size=1,
                memory_reset=False, reset_intervals=None)
# memory_reset=True, reset_intervals=None → [1,1,...,1] (spec-08)
# memory_reset=True, reset_intervals=[1,8,64,512] → selective
# memory_reset=False → no reset regardless of intervals
```

---

## Config Schema

```json
{
  "model": {
    "memory_reset": "periodic",
    "reset_intervals": [1, 8, 64, 512]
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reset_intervals` | `[int]` or null | null | Per-level reset intervals. null = [1,...,1]. |

---

## Checkpoint Compatibility

The fire_counts are transient inner-loop state — they can be reset to zero on
checkpoint restore without loss of correctness (the model simply starts a fresh
reset cycle). No checkpoint format changes needed.

reset_intervals is a config field, stored in the config JSON that is already
serialized with every checkpoint.

---

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|--------------|
| eq-006-local-memory-update | tnt_equations | TNT §3.2 (2511.07343) | extends |
| eq-014-n-local-memories-update | tnt_equations | TNT §3.3 (2511.07343) | implements |
| eq-097-hope-cms-chain | hope_equations | HOPE §7 (2512.24695) | cites |

---

## Code Smell Constraints

- **CS-10** (no model.train/eval): reset_intervals is a structural config, not a
  mode flag. Same forward code path regardless of interval values.
- **CS-11** (no training loop in memory rule): Reset decision logic stays in the
  orchestration layer (Rust model struct or Python loop), not in memory rules.
- **CS-32** (observe-then-advance): Reset happens AFTER observing the step's final M
  and BEFORE the next step's first observe. fire_count update is part of this
  post-observe phase.
- **CS-48** (per-level parameter independence): Each level has its own independent
  fire counter and reset interval. No cross-level coupling.

---

## Test Plan

1. **Unit: fire counter logic** — verify should_reset_level returns true only every
   R_k fires. Test with R=[1,1,1,1] (always reset), R=[1,8,64,512] (selective),
   R=[1,1,1,1] vs [2,2,2,2] (half frequency).

2. **Unit: backward compatibility** — verify that reset_intervals=None produces
   identical behavior to current memory_reset=True (all-ones).

3. **Integration: M persistence** — with R=[1,8,64,512], verify L1 M norm > 0 after
   steps 1-7 (not yet reset), L1 M norm = 0 after step 8 (reset). Same for L2
   at step 64, L3 at step 512.

4. **Integration: checkpoint round-trip** — save checkpoint with fire_counts in
   flight, restore, verify reset cycle continues correctly (or restarts cleanly).

5. **Regression: spec-08 parity** — run 100 steps with reset_intervals=[1,1,1,1]
   and memory_reset=True (no intervals), verify loss trajectories are identical.
