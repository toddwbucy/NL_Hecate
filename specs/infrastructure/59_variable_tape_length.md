# Variable Tape Length: seq_len as Experimental Variable for Gear-Shifting

<!-- HADES: hope_equations/eq-097-hope-cms-chain (HOPE §7);
            tnt_equations/eq-006-local-memory-update (TNT §3.2);
            tnt_equations/eq-014-n-local-memories-update (TNT §3.3) -->

```text
CONTRACT
  Purpose:    Establish seq_len as a first-class experimental variable for the
              gear-shifting strategy (spec 57). The tape length determines how
              many writes each CMS level accumulates per step — this is the
              primary lever for activating higher levels. Rather than fixing
              seq_len=2048, we provide sweep configs at {1024, 2048, 4096} and
              document the VRAM budget so researchers can find the minimum tape
              length where L1/L2 build meaningful M signal.

  Expects:    - Spec 57: selective periodic reset with per-level intervals.
              - Spec 58: level activation metrics (‖M‖_F > ε AND θ_eff > δ)
                for measuring when levels "wake up."
              - Existing infrastructure: seq_len is already a config parameter
                with no hardcoded upper bound. The only constraint is
                seq_len % chunk_sizes[i] == 0 for all i.
              - chunk_sizes=[1, 8, 64, 512] for k=4 CMS hierarchy.

  Guarantees: - Sweep configs for seq_len={1024, 2048, 4096} with matched
                batch_size to fit A6000 48GB VRAM at d=1024.
              - VRAM budget formula documented for planning arbitrary configs.
              - Validation test confirming forward+backward+optimizer at each
                seq_len on the current hardware.
              - No code changes to core, kernels, or lib.rs — only configs,
                validation, and documentation.

  Cost:       - Zero code changes (seq_len is already fully parameterized).
              - 3 new config files.
              - 1 validation test.
              - VRAM scales linearly with seq_len: ~2x seq_len → ~2x memory
                for attention buffers and cache.

  Trade-off:  Longer tapes give higher levels more writes per step but reduce
              throughput (linear in seq_len) and increase VRAM. The gear-shifting
              hypothesis predicts a minimum tape length exists where L1/L2
              activation metrics (spec 58) cross threshold. We don't know this
              value — hence the sweep.

              With selective reset (spec 57, intervals=[1,8,64,512]):
              | seq_len | L0 writes/step | L1 writes/step | L2 writes/step |
              |---------|---------------|----------------|----------------|
              | 512     | 512           | 64             | 8              |
              | 1024    | 1024          | 128            | 16             |
              | 2048    | 2048          | 256            | 32             |
              | 4096    | 4096          | 512            | 64             |

              At seq_len=4096, L2 gets 64 writes/step — matching what L1 gets
              at seq_len=512. This may be the activation threshold for L2.

  Position:   specs/infrastructure/59_variable_tape_length.md
  Source:     HOPE (2512.24695) §7 — CMS chain structure
              TNT (2511.07343) §3.2 — local memory with periodic reset
              Internal: k4_chain_dolmino_d1024_32h seed run analysis
              Internal: curricula-gear strategy (memory/project_curricula_gear_strategy.md)
  Related:    specs/infrastructure/57_selective_periodic_reset.md (gear-shifting)
              specs/infrastructure/58_level_activation_metrics.md (activation measurement)
              EPIC task_6ebcb7 / CG-2 task_2f39bf
```

---

## Motivation

The k4_chain_dolmino_d1024_32h seed run (61K steps, seq_len=512) showed L2/L3
completely inactive. With all-ones periodic reset, L2 gets only 8 writes before
M is zeroed. Selective reset (spec 57) lets writes accumulate, but at seq_len=512
L2 still only gets 8 writes per step.

The tape length is the second lever (after reset intervals) for equalizing write
budgets. Doubling seq_len doubles the writes per step for every level. The question:
**what is the minimum tape length where higher levels activate?**

This is an experimental question, not an engineering one. The infrastructure already
supports arbitrary seq_len. What we need is:
1. Configs to sweep
2. VRAM budget to plan batch_size
3. Validation that the full pipeline works at each length

---

## VRAM Budget

### Formula (approximate, for d=1024, k=4, fp32 inner loop + bf16 attention)

```text
VRAM_total ≈ VRAM_params + VRAM_context + VRAM_cache + VRAM_optimizer

VRAM_params ≈ 4 * n_params_bytes                    # fp32 weights
VRAM_context ≈ n_blocks * k * d * d * 4             # M matrices (fp32)
VRAM_cache ≈ n_blocks * batch_size * seq_len * d * 4 * C  # forward cache
VRAM_optimizer ≈ 2 * VRAM_params                    # AdamW m + v
```

Where C ≈ 8-12 (multiple buffers: attention logits, memory intermediates, gradients).

### Estimated VRAM for d=1024, n_blocks=4, k=4

| seq_len | batch_size | Cache (GB) | Total est. (GB) | Fits A6000? |
|---------|-----------|------------|-----------------|-------------|
| 512     | 8         | ~8.5       | ~16             | Yes (48GB)  |
| 1024    | 4         | ~8.5       | ~16             | Yes         |
| 2048    | 2         | ~8.5       | ~16             | Yes         |
| 4096    | 1         | ~8.5       | ~16             | Yes         |

The key insight: halving batch_size when doubling seq_len keeps total VRAM
roughly constant. Tokens/step = batch_size * seq_len stays constant at ~4096.

For H200 (80GB), batch_size can be 2-4x larger at each seq_len.

---

## Sweep Configs

Three configs targeting A6000 48GB at d=1024:

### 1. seq_len=1024 (2x baseline)

```json
{
  "seq_len": 1024,
  "batch_size": 4,
  "memory_reset": "periodic",
  "reset_intervals": [1, 8, 64, 512]
}
```

### 2. seq_len=2048 (4x baseline)

```json
{
  "seq_len": 2048,
  "batch_size": 2,
  "memory_reset": "periodic",
  "reset_intervals": [1, 8, 64, 512]
}
```

### 3. seq_len=4096 (8x baseline)

```json
{
  "seq_len": 4096,
  "batch_size": 1,
  "memory_reset": "periodic",
  "reset_intervals": [1, 8, 64, 512]
}
```

All three share:
- `chunk_sizes: [1, 8, 64, 512]` — CMS frequencies unchanged
- `reset_intervals: [1, 8, 64, 512]` — gear-shifting from spec 57
- `k: 4, d_model: 1024, num_heads: 32, n_blocks: 4`
- Same learning rate, warmup, and optimizer settings

---

## Validation Test

A Python test that:
1. Creates a small model (d=32, k=4) at each target seq_len
2. Runs one forward+backward+optimizer step
3. Verifies no NaN/Inf in loss or gradients
4. Confirms memory_norms() returns k values

This validates the full pipeline at each seq_len without requiring GPU hours.

---

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|--------------|
| eq-097-hope-cms-chain | hope_equations | HOPE §7 (2512.24695) | cites |
| eq-006-local-memory-update | tnt_equations | TNT §3.2 (2511.07343) | cites |
| eq-014-n-local-memories-update | tnt_equations | TNT §3.3 (2511.07343) | cites |

---

## Code Smell Constraints

- **CS-10** (no train/eval): configs are structural, not mode flags.
- **CS-18** (orchestration in Python): all changes are config + Python-tier.
- **CS-48** (per-level independence): each level's write count scales with
  seq_len independently.

---

## Test Plan

1. **Validation: pipeline at seq_len=1024** — forward+backward+optimizer, no NaN.
2. **Validation: pipeline at seq_len=2048** — same.
3. **Validation: pipeline at seq_len=4096** — same.
4. **Regression: seq_len=512 unchanged** — existing tests still pass.
