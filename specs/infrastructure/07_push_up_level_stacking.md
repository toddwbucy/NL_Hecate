# Push-Up Progressive Level Stacking

```text
CONTRACT
  Purpose   : Warm-start k=4 CMS by progressively adding levels from bottom,
              pushing trained levels to slower frequencies each phase
  Expects   : Safetensors checkpoint with k=N levels; config requesting k=N+1;
              data cursor sidecar with position tracking
  Guarantees: All existing level params shift up by 1 (level.i → level.i+1);
              fresh level 0 initialized with default gate biases;
              optimizer state reset for all params (clean AdamW on resume);
              data cursor seekable to arbitrary position via config
  Cost      : One checkpoint load + tensor remapping per phase (~seconds);
              no new CUDA kernels; Python-tier orchestration only
  Trade-off : Optimizer state not preserved across phases (acceptable —
              fresh AdamW with warmup handles level transitions cleanly)
  Position  : Research experiment infrastructure
  Source    : HOPE (2512.24695) §5.1 "Ad-hoc Level Stacking"
```

## Motivation

Cold-start k=4 CMS leaves L3 dead. In 100K-step runs (MAC, MAG, TNT), L3 shows:
- 0.2% activation, gnorm 0.000002 (5 orders of magnitude below L2)
- Brief activation at ~step 9K then collapse — gradient competition, not cold start

HOPE §5.1 proposes initializing higher levels from pretrained checkpoints.
This spec implements a **push-up variant**: trained memory is promoted to a
slower frequency level, and a fresh fast level is added at L0.

## Key Insight: Gate Frequency Independence

Gate weights (W_alpha, W_theta, W_eta) learn **content sensitivity** — which
dimensions of the input k,v representation matter for retention/learning rate
decisions. This is purely content-dependent and has nothing to do with firing
frequency. CMS frequency is external scheduling; the gate is oblivious to how
often it fires.

Therefore: gate-memory pairs transfer cleanly across frequency levels with
zero recalibration. The gate learned "which neurons matter" — that knowledge
is frequency-independent.

## Protocol: Push-Up Curriculum

### Phase 1 — k=1 seed (10K steps)
- Config: k=1, chunk_sizes=[1]
- Data: positions [0 → 10K×seq_len]
- Goal: L0 converges (gnorm stabilizes, loss plateau begins)
- Output: checkpoint_phase1.safetensors + cursor at position P1

### Phase 2 — k=2 (10K steps, 50% overlap)
- Load checkpoint_phase1, extend_k=2, push_up=true
- Level mapping: old L0 → new L1, fresh L0 initialized
- Data: rewind cursor to P1 - 5K×seq_len, run 10K steps
  - First 5K steps: L1 sees familiar data at new frequency
  - Last 5K steps: both levels see new data
- Goal: L1 contributing (gnorm > 0.01), L0 reconverged
- Output: checkpoint_phase2.safetensors + cursor at position P2

### Phase 3 — k=3 (10K steps, 50% overlap)
- Load checkpoint_phase2, extend_k=3, push_up=true
- Level mapping: old L0 → new L1, old L1 → new L2, fresh L0
- Data: rewind cursor to P2 - 5K×seq_len, run 10K steps
- Goal: L2 contributing, L1 adjusted, L0 reconverged
- Output: checkpoint_phase3.safetensors + cursor at position P3

### Phase 4 — k=4 production run (indefinite)
- Load checkpoint_phase3, extend_k=4, push_up=true
- Level mapping: old L0→L1, old L1→L2, old L2→L3, fresh L0
- Data: **rewind cursor to position 0** — full corpus from start
- All 4 levels warm. L3 has been through 3 rounds of progressive training.
- This IS the training run — do not stop.

### Step Budget Rationale
- 10K steps per phase = sufficient for L0 convergence (data shows L0 active
  by step 320, stable by ~2K)
- 50% overlap (5K old + 5K new) gives promoted levels familiar data to
  establish gradient flow at new frequency before encountering novelty
- Total warm-up cost: ~30K steps. Cheap relative to 100K+ production run.

## Implementation

### Config Changes

New fields in build config (Python tier, `engine/config.py`):

```python
extend_k: int | None = None      # Target k (must be current_k + 1)
push_up: bool = False             # If true, shift existing levels up; if false, add on top
data_seek: int | None = None      # Override cursor position (token offset)
```

### Checkpoint Extension Logic

In `engine/loop.py`, after loading checkpoint but before training loop:

```python
if bcfg.extend_k is not None:
    loaded_k = len(params.levels)  # or from loaded config
    target_k = bcfg.extend_k
    assert target_k == loaded_k + 1, f"extend_k must be current k+1"

    if bcfg.push_up:
        # Shift all existing levels up by 1
        # level.0 → level.1, level.1 → level.2, ...
        shifted_levels = [None] + params.levels  # index 0 is placeholder
        # Initialize fresh level 0 with default biases for level 0
        fresh_l0 = init_level_params(d_model, level_index=0, seed=new_seed)
        shifted_levels[0] = fresh_l0
        params.levels = shifted_levels
    else:
        # Stack up: add new level on top
        new_level = init_level_params(d_model, level_index=target_k-1, seed=new_seed)
        params.levels.append(new_level)

    # Resize CMS aggregation logits
    params.alpha_mem = resize_alpha(params.alpha_mem, target_k)
    params.alpha_refl = resize_alpha(params.alpha_refl, target_k)

    # Update config k and chunk_sizes
    cfg.k = target_k
    cfg.chunk_sizes = STANDARD_CHUNK_SIZES[:target_k]  # [1, 8, 64, 512]
```

### Data Cursor Seek

In `engine/loop.py`, after cursor restore:

```python
if bcfg.data_seek is not None:
    active_loader.seek(bcfg.data_seek)
    print(f"  Data cursor overridden: seeking to position {bcfg.data_seek:,}")
```

Requires adding `BpeDataLoader.seek(position)` — set internal position
directly, recalculate chunk_id, clear content_hash (no validation on
manual seek).

### Fresh Level Initialization

New level 0 gets:
- W_K, W_V, W_Q: Xavier uniform (same as normal init)
- Gate biases: level-0 defaults (b_alpha=3.0, b_theta=-4.6, b_eta=1.5)
- Gate weights: Xavier uniform
- Memory M: zeros (no prior state)
- Momentum S: zeros

### AdamW Handling

AdamW state is NOT persisted in checkpoints (confirmed in codebase).
On resume, optimizer reinitializes with fresh m_t=0, v_t=0 for all params.
Combined with warmup_steps=1000, this is clean — no special handling needed.

### Conductor/Pulse Sync

ConductorState.step resets to 0 on extend_k (new training phase).
chunk_sizes updated to match new k. Pulse generation picks up immediately.

## Configs for Each Phase

### Phase 1: `push_up_phase1_k1.json`
```json
{
  "model": { "d_model": 512, "k": 1, "chunk_sizes": [1],
             "memory_rule": "titans", "composition": "mag" },
  "build": { "lr": 0.0006, "steps": 10000, "warmup_steps": 500,
             "save_path": "checkpoints/push_up_phase1_k1.safetensors" },
  "data": { "path": "data/dolmino_100b", "format": "sharegpt" }
}
```

### Phase 2: `push_up_phase2_k2.json`
```json
{
  "model": { "d_model": 512, "k": 2, "chunk_sizes": [1, 8],
             "memory_rule": "titans", "composition": "mag" },
  "build": { "lr": 0.0006, "steps": 10000, "warmup_steps": 500,
             "load": "checkpoints/push_up_phase1_k1.safetensors",
             "extend_k": 2, "push_up": true,
             "data_seek": "<P1 - 5000*512>",
             "save_path": "checkpoints/push_up_phase2_k2.safetensors" },
  "data": { "path": "data/dolmino_100b", "format": "sharegpt" }
}
```

### Phase 3: `push_up_phase3_k3.json`
(Same pattern, extend_k=3, data_seek = P2 - 5000*512)

### Phase 4: `push_up_phase4_k4.json`
```json
{
  "model": { "d_model": 512, "k": 4, "chunk_sizes": [1, 8, 64, 512],
             "memory_rule": "titans", "composition": "mag" },
  "build": { "lr": 0.0006, "steps": 100000, "warmup_steps": 1000,
             "load": "checkpoints/push_up_phase3_k3.safetensors",
             "extend_k": 4, "push_up": true,
             "data_seek": 0,
             "save_path": "checkpoints/push_up_phase4_k4.safetensors" },
  "data": { "path": "data/dolmino_100b", "format": "sharegpt" },
  "m_norm_max": [100.0, 100.0, 100.0, 100.0]
}
```

## Success Metric

L3 gnorm sustained above 0.001 after 10K steps in Phase 4.
(Cold-start baseline: L3 peaks at 0.000128 then collapses to 0.000002.)

## Control

Cold-start k=4 runs already in flight (MAC, MAG 100K on A6000).
Direct comparison: same model, same data, same hyperparams.
Only difference: push-up warm start vs cold start.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-087-ad-hoc-level-stacking | hope_equations | HOPE §5.1, arXiv:2512.24695 | adapts |

Note: "adapts" not "implements" — this is a push-up variant of the paper's
stack-up method. The paper adds levels on top; we push trained levels to
slower frequencies and add fresh fast levels at L0.

## Deferred

- Stack-Up variant (paper method): add new level on top instead of pushing up.
  Same infrastructure, different level mapping. Config: push_up=false.
- Spiral curriculum: re-expose earlier data with increasing multiplier at each
  phase. Too complex for first test — keep it simple with 50% overlap.
- Automated phase transitions: detect gnorm stabilization and trigger next
  phase automatically. Manual stop-and-restart is fine for research.
