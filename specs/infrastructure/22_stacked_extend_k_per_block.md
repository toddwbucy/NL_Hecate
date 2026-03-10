# Per-Block extend_k Push-Up for Stacked Checkpoints

```text
CONTRACT
  Purpose:    Enable push-up level stacking (spec 07, PR #176) for stacked
              multi-block models. Each block independently shifts its levels up
              and receives a fresh L0 with independent Xavier init, preserving
              depth-specialized weights. Also enables stacked checkpoint
              save/load (currently blocked).
  Expects:    BUG-01 (W_O, PR #188), BUG-02 (MAG sigmoid, PR #189), and BUG-03
              (alpha_mem aggregation, PR #190) all resolved. Single-block
              push-up logic (extend_push_up in model.rs:2254-2317) working and
              validated. StackedMAGParams (stacked_model.rs) with per-block
              BlockParams containing levels, alpha_mem, alpha_refl.
  Guarantees: After implementation:
              1. Stacked checkpoints save/load via safetensors with hierarchical
                 keys: block.{n}.level.{m}.* for per-block tensors, shared.*
                 for embed/unembed/ln_final.
              2. extend_k with push_up=true works for n_blocks > 1: each block's
                 level[i] → level[i+1], fresh level[0] per block with distinct
                 seeds. Shared params (embed, unembed, ln_final) preserved.
              3. alpha_mem logits shift per block: old alpha[i] → new alpha[i+1],
                 fresh alpha[0] = 0.0 per block (uniform initial weight for new L0).
              4. Optimizer state fully reset on promotion (fresh m_t=0, v_t=0,
                 warmup restarts) — same semantics as single-block push-up.
              5. loop.py blocking guards (lines 465-474) removed and replaced
                 with stacked-aware routing.
  Cost:       One-time checkpoint conversion overhead (read old single-block
              format + write stacked format). Per-promotion overhead: N blocks ×
              level-shift + Xavier init — negligible (<100ms for n_blocks=4, d=512).
  Trade-off:  Adds a second checkpoint format (stacked safetensors alongside
              single-block safetensors). Both formats share the same container
              (safetensors) but with different key hierarchies. The loader must
              detect which format is present and route accordingly.
  Position:   specs/infrastructure/22_stacked_extend_k_per_block.md
  Source:     Spec 07 (push_up_level_stacking) — single-block push-up protocol.
              Spec 14 (multi_block_stacking) — stacked architecture.
              Spec 19 (stacked_push_up_experiment) — experiment protocol.
              HOPE (2512.24695) Section 5.1 — ad-hoc level stacking.
```

## Motivation

### The blocker

The stacked push-up experiment (spec 19, EXP-15) requires 4 phases:
- Phase 1: n_blocks=4, k=1 (fresh init, trains from scratch) — **works today**
- Phase 2: n_blocks=4, k=2 (load phase 1 checkpoint, extend_k=2) — **blocked**
- Phase 3: n_blocks=4, k=3 (load phase 2 checkpoint, extend_k=3) — **blocked**
- Phase 4: n_blocks=4, k=4 (load phase 3 checkpoint, extend_k=4) — **blocked**

Phases 2-4 are blocked by three missing capabilities:
1. **Stacked checkpoint save**: `loop.py:1317` prints `"stacked model save not yet implemented"` and skips
2. **Stacked checkpoint load**: `loop.py:465-468` raises `RuntimeError` for `is_stacked and bcfg.load`
3. **Per-block extend_k**: `loop.py:470-474` raises `RuntimeError` for `is_stacked and extend_k`

All three must be resolved together — checkpoint save/load is a prerequisite for extend_k,
and extend_k is the whole point.

### Why single-block extend_push_up doesn't generalize

The current `extend_push_up` (`model.rs:2254-2317`) operates on `MAGParams`:
- Flat `levels: Vec<MemoryLevelParams>` — shifts `levels[i]` → `levels[i+1]`
- Flat `alpha_mem: Vec<f32>` — shifts `alpha_mem[i]` → `alpha_mem[i+1]`
- Single SWA (embed/unembed/W_Q/W_K/W_V/W_O) — preserved bitwise

For `StackedMAGParams`:
- Levels live inside `blocks[n].levels` — **per-block** level arrays
- Alpha lives inside `blocks[n].alpha_mem` — **per-block** aggregation logits
- SWA projections live inside `blocks[n].w_q/w_k/w_v/w_o` — **per-block**
- Shared params: `w_embed`, `w_unembed`, `ln_final_gamma/beta` — global

The level shift must happen independently per block. Each block gets its own fresh
L0 with a distinct Xavier seed — they must NOT be clones of each other, because
blocks at different depths in the residual stream have different activation
distributions (the depth-specialization hypothesis from spec 19).

## Stacked Checkpoint Format

### Key hierarchy

Stacked checkpoints use hierarchical keys under the same safetensors container:

```text
# Shared parameters (global)
shared.embed.weight          — [vocab × d]
shared.lm_head.weight        — [d × vocab]
shared.ln_final.gamma        — [d]
shared.ln_final.beta         — [d]

# Per-block parameters
block.0.swa.w_q              — [d × d]
block.0.swa.w_k              — [d × d]
block.0.swa.w_v              — [d × d]
block.0.swa.w_o              — [d × d]
block.0.ln_attn.gamma        — [d]
block.0.ln_attn.beta         — [d]
block.0.ln_mem.gamma         — [d]
block.0.ln_mem.beta          — [d]
block.0.alpha_mem            — [k]
block.0.alpha_refl           — [k]
block.0.level.0.w_k          — [d × d]
block.0.level.0.w_v          — [d × d]
block.0.level.0.w_q          — [d × d]
block.0.level.0.gate.alpha   — [d]
block.0.level.0.gate.b_alpha — [1]
block.0.level.0.gate.theta   — [d]
block.0.level.0.gate.b_theta — [1]
block.0.level.0.gate.eta     — [d]
block.0.level.0.gate.b_eta   — [1]
block.0.level.0.w_omega      — (if non-empty)
...
block.0.level.{k-1}.*        — same structure per level
block.1.*                     — same structure for block 1
...
block.{n-1}.*                 — same structure for block n-1
```

### Format detection

The loader detects stacked vs single-block by checking whether the header
contains keys starting with `block.` or `shared.`. If neither is found, fall
through to single-block `load_safetensors()`.

### __metadata__

Same `__metadata__` structure as single-block. Config JSON includes `n_blocks`
field (already part of config passed to Python). Add `"stacked": true` flag to
metadata for explicit format identification.

## Per-Block extend_push_up

### Algorithm

```text
fn extend_stacked_push_up(
    old: &StackedMAGParams,
    new_cfg: &MAGConfig,
    n_blocks: usize,
    seed: u64,
) -> StackedMAGParams:
    assert new_cfg.k == old.blocks[0].levels.len() + 1

    # 1. Preserve shared params exactly (bitwise)
    new.w_embed = old.w_embed.clone()
    new.w_unembed = old.w_unembed.clone()
    new.ln_final_gamma = old.ln_final_gamma.clone()
    new.ln_final_beta = old.ln_final_beta.clone()

    # 2. Per-block level shift (independent per block)
    for b in 0..n_blocks:
        old_block = old.blocks[b]
        new_block = BlockParams::init(new_cfg, seed + b * 10_000)

        # Preserve SWA projections and LayerNorms for this block
        new_block.w_q = old_block.w_q.clone()
        new_block.w_k = old_block.w_k.clone()
        new_block.w_v = old_block.w_v.clone()
        new_block.w_o = old_block.w_o.clone()
        new_block.ln_attn_gamma = old_block.ln_attn_gamma.clone()
        new_block.ln_attn_beta = old_block.ln_attn_beta.clone()
        new_block.ln_mem_gamma = old_block.ln_mem_gamma.clone()
        new_block.ln_mem_beta = old_block.ln_mem_beta.clone()

        # Shift levels: old level[i] → new level[i+1]
        for i in 0..old_k:
            new_block.levels[i+1] = old_block.levels[i].clone()

        # Fresh L0: clone old L0 projections for scale-matched init
        # (same logic as single-block: prevents signal swamping)
        donor = old_block.levels[0]
        new_block.levels[0].w_k_mem = donor.w_k_mem.clone()
        new_block.levels[0].w_v_mem = donor.w_v_mem.clone()
        new_block.levels[0].w_q_mem = donor.w_q_mem.clone()
        new_block.levels[0].w_alpha = donor.w_alpha.clone()
        new_block.levels[0].w_theta = donor.w_theta.clone()
        new_block.levels[0].w_eta = donor.w_eta.clone()
        new_block.levels[0].w_omega = donor.w_omega.clone()
        # Gate biases kept at level-0 defaults from init

        # Shift alpha logits per block
        for i in 0..old_k:
            new_block.alpha_mem[i+1] = old_block.alpha_mem[i]
            new_block.alpha_refl[i+1] = old_block.alpha_refl[i]
        # new alpha[0] = 0.0 (from init) — uniform initial weight

        new.blocks[b] = new_block

    return new
```

### Key differences from single-block push-up

| Aspect | Single-block (`extend_push_up`) | Stacked per-block |
|--------|-------------------------------|-------------------|
| Level array | `MAGParams.levels[i]` | `blocks[b].levels[i]` per block |
| Alpha logits | `MAGParams.alpha_mem[i]` | `blocks[b].alpha_mem[i]` per block |
| SWA preservation | Single `swa` field | Per-block `w_q/w_k/w_v/w_o` + LN |
| Shared params | Embed/unembed in `swa` | Separate `w_embed/w_unembed/ln_final` |
| Fresh L0 seeds | One seed | Distinct seed per block (`seed + b * 10_000`) |
| Persistent tokens | `MAGParams.persistent_tokens` | Not applicable (MAG, no persistent tokens) |

### Fresh L0 seed independence

Each block's fresh L0 must have a distinct Xavier initialization. Using the same
seed for all blocks would create identical L0s that then see different residual
stream depths — the initialization should already be independent so any
convergence to similar weights is a learned property, not an artifact.

Seed formula: `block_seed = base_seed + block_index * 10_000` — same offset
pattern as `StackedMAGParams::init()` (stacked_model.rs:179).

## Stacked Checkpoint Save/Load

### Save (`save_stacked_safetensors`)

New function in `checkpoint.rs`:

```rust
pub fn save_stacked_safetensors(
    path: &Path,
    params: &StackedMAGParams,
    config: &MAGConfig,
    n_blocks: usize,
    build_state: Option<&BuildResumeState>,
) -> io::Result<()>
```

Serializes with hierarchical keys. Per-level tensors follow the same structure
as single-block but prefixed with `block.{n}.`. Shared params use `shared.`
prefix. Metadata includes `"stacked": "true"` and `"n_blocks": "N"`.

### Load (`load_stacked_safetensors`)

New function in `checkpoint.rs`:

```rust
pub fn load_stacked_safetensors(
    path: &Path,
) -> io::Result<(StackedMAGParams, MAGConfig, usize, Option<BuildResumeState>)>
```

Returns n_blocks alongside the params. Detects block count by scanning header
keys for the highest `block.{n}` prefix.

### Format detection in loader

A top-level `load_checkpoint_auto(path)` function (or modify the existing Python
routing) checks:
1. If header contains `shared.embed.weight` → stacked format → `load_stacked_safetensors`
2. Else if header contains `embed.weight` → single-block format → `load_safetensors`

### PyO3 exposure

Two new Python-visible functions:
- `nl_hecate.save_stacked_checkpoint(params, config, path, build_state)`
- `nl_hecate.load_stacked_checkpoint(path)` → `(params, config, n_blocks, build_state)`
- `nl_hecate.extend_stacked_push_up(params, config, n_blocks, seed)` → new stacked params

The params object for stacked models needs a Python-side representation. The
simplest approach: pass `StackedMAGParams` as an opaque PyO3 object (like
`MAGParams` today) with the `GpuStackedModel` constructing from it.

## loop.py Changes

### Remove blocking guards

Replace lines 465-474:

```python
# OLD (blocking):
if is_stacked and bcfg.load:
    raise RuntimeError("Stacked model checkpoint loading is not yet implemented...")
if is_stacked and getattr(bcfg, "extend_k", None) is not None:
    raise RuntimeError("extend_k is not supported with n_blocks > 1...")

# NEW (routing):
if is_stacked and bcfg.load:
    params, cfg, n_blocks, build_state = nl_hecate.load_stacked_checkpoint(load_path)
    # ... same build_state handling as single-block ...
```

### Stacked extend_k routing

After loading a stacked checkpoint, if `extend_k` is set:

```python
if is_stacked and bcfg.extend_k is not None:
    loaded_k = cfg.k
    target_k = bcfg.extend_k
    if target_k != loaded_k + 1:
        print(f"  ERROR: extend_k={target_k} must be loaded_k+1={loaded_k + 1}")
        return
    # Build new MAGConfig with target_k (same logic as single-block lines 221-257)
    new_cfg = nl_hecate.MAGConfig(...)
    if bcfg.push_up:
        params = nl_hecate.extend_stacked_push_up(params, new_cfg, n_blocks, bcfg.seed)
        print(f"  Stacked push-up: k={loaded_k} → k={target_k}, n_blocks={n_blocks}")
    else:
        print("  ERROR: extend_k set but push_up not true (stack_up not supported for stacked)")
        return
    cfg = new_cfg
    resume_step = 0
    build_state = None
```

### Stacked checkpoint save

Replace the skip-print at line 1317 with actual save:

```python
if is_stacked:
    nl_hecate.save_stacked_checkpoint(params, cfg, save_path, build_state_dict)
else:
    # existing single-block save
```

### stack_up: NOT supported for stacked

Stack-up (keep levels in place, add at top) does not make sense for the stacked
push-up experiment. The spec 19 protocol exclusively uses push-up. The guard
for `stack_up and is_stacked` remains as a RuntimeError with a clear message.

## Optimizer State Reset

On promotion (extend_k), the optimizer state is fully reset:
- AdamW moment buffers `m_t`, `v_t` → all zeros
- Warmup counter → reset to 0
- Bias correction step counter → reset to 1

This is the same behavior as single-block push-up (spec 07). The
`GpuStackedModel::from_params()` constructor allocates fresh optimizer state
from the new params, so there is no carryover.

## Implementation Order

1. **Stacked checkpoint save** (`checkpoint.rs`): `save_stacked_safetensors`
2. **Stacked checkpoint load** (`checkpoint.rs`): `load_stacked_safetensors`
3. **PyO3 bindings** for save/load stacked checkpoints
4. **Remove load guard** (`loop.py:465-468`): route to stacked loader
5. **Replace save skip** (`loop.py:1317, 1617, 1653`): route to stacked saver
6. **`extend_stacked_push_up`** (`stacked_model.rs`): per-block level shift
7. **PyO3 binding** for `extend_stacked_push_up`
8. **Remove extend_k guard** (`loop.py:470-474`): route to stacked push-up
9. **Tests**: round-trip save/load, extend_k correctness, alpha_mem shift

## Files to Modify

| File | Change |
|------|--------|
| `core/src/checkpoint.rs` | Add `save_stacked_safetensors` and `load_stacked_safetensors` |
| `core/src/stacked_model.rs` | Add `StackedMAGParams::extend_push_up()` |
| `python/nl_hecate/lib.rs` (or bindings) | PyO3 wrappers for stacked checkpoint + extend |
| `python/engine/loop.py` | Remove 3 blocking guards, add stacked routing for load/save/extend_k |

## Acceptance Criteria

1. `save_stacked_safetensors` + `load_stacked_safetensors` round-trip: all params bitwise identical
2. Hierarchical key format: `block.{n}.level.{m}.*` in safetensors header
3. Format auto-detection: loader correctly routes stacked vs single-block files
4. `extend_stacked_push_up`: each block's level[i] shifted to level[i+1]
5. Fresh L0 per block: distinct Xavier init (different seeds per block)
6. L0 projection cloning: fresh L0 gets old L0's w_k_mem/w_v_mem/w_q_mem for scale match
7. L0 gate biases: level-0 defaults (not cloned from old L0)
8. Alpha logits shifted per block: old alpha[i] → new alpha[i+1], alpha[0] = 0.0
9. Shared params (embed, unembed, ln_final) preserved exactly across extend_k
10. Optimizer state reset: fresh m_t=0, v_t=0 after promotion
11. loop.py: `--load` works for stacked checkpoints
12. loop.py: `--extend_k` with `--push_up` works for stacked checkpoints
13. Periodic and final checkpoint saves work for stacked models (no more skip-prints)
14. No regressions: single-block save/load/extend_k unchanged

## Dependencies

- BUG-01 (spec 18, PR #188, merged): W_O in stacked forward
- BUG-02 (spec 20, PR #189, merged): MAG sigmoid gating in stacked forward
- BUG-03 (spec 21, PR #190, merged): alpha_mem aggregation in stacked forward
- Spec 07 (push_up_level_stacking, PR #176, merged): single-block push-up reference
- Spec 14 (multi_block_stacking): stacked architecture definition

## Falsification

This implementation is wrong if:
1. Stacked checkpoint save/load loses parameters (non-bitwise round-trip)
2. After extend_k, any block has identical level weights to another block's corresponding level (seed independence failure)
3. After extend_k, L0 gate biases match L1 gate biases (bias cloning leak — L0 should have level-0 defaults, L1 should have level-0 defaults from the prior phase)
4. The optimizer carries moment state across promotion (m_t/v_t nonzero at step 0 of new phase)

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-097-hope-cms-chain | hope_equations | HOPE §5.1 | adapts (stacked variant) |
| eq-070-arch-variant1 | hope_equations | HOPE CMS chain | implements |

## Code Smells

| Smell | Enforcement | Rationale |
|-------|-------------|-----------|
| CS-04/05/06 | ontological | "blocks" for depth, "levels" for CMS frequency |
| CS-10 | behavioral | No mode flag — extend_k logic is checkpoint manipulation, not forward path |
| CS-18 | architectural | Checkpoint I/O and orchestration in Rust/Python tiers, not CUDA |
| CS-32 | behavioral | Observe-then-advance — promotion is a discrete state transition |
