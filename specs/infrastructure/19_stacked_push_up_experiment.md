# EXP-15: Stacked Push-Up Level Curriculum

```
CONTRACT
  Purpose:    Validate the push-up level stacking strategy on a 4-block stacked
              model. Phase 1 trains n_blocks=4, k=1 (4 independent L0s, one per
              block). Subsequent phases push up within each block, adding levels
              progressively until n_blocks=4, k=4 (16 total level instances).
  Expects:    BUG-01 (W_O output projection) and BUG-02 (MAG sigmoid gating)
              resolved in the stacked forward/backward paths. Push-up checkpoint
              extension logic (spec 07, PR #176). Dolmino-100b corpus.
  Guarantees: Each block's L0 trains at its own depth in the residual stream,
              producing depth-specialized memory. On promotion, each block's
              levels inherit depth-appropriate weights — not clones of a
              single-block checkpoint.
  Cost:       Phase 1: ~20K steps (4 blocks × 1 level each).
              Phase 2-4: ~15K steps each. Total warm-up: ~65K steps.
              Production phase 4 run: 100K+ steps.
              GPU: single A6000 (d=512 fits in 24GB at n_blocks=4, k=4).
  Trade-off:  Longer warm-up (65K vs 30K for single-block push-up) because
              4 blocks need more steps for inter-block residual dynamics to
              stabilize. But each promoted level carries depth-specialized
              weights, which single-block push-up cannot provide.
  Position:   specs/infrastructure/19_stacked_push_up_experiment.md
  Source:     HOPE (2512.24695) Section 5.1 — ad-hoc level stacking.
              Spec 07 (push_up_level_stacking) — single-block push-up protocol.
              Spec 12 (metric_driven_promotion) — convergence signals.
              Spec 14 (multi_block_stacking) — stacked architecture.
              Internal: cold-start k=4 dead L3 (EXP-09, EXP-10).
              Internal: rate-bottleneck analysis (session 2026-03-10).
```

## Motivation

### Why single-block push-up is insufficient for stacked models

The existing push-up protocol (spec 07) trains a single block through k=1→k=4.
For a 4-block stacked model, the naive approach would be to clone that
single-block checkpoint 4 times into `block.0` through `block.3`. This loses
the key advantage of stacking: **depth specialization**.

In a stacked model:
- Block 0's memory sees raw embeddings
- Block 1's memory sees block 0's refined output
- Block 2's memory sees 2x refined output
- Block 3's memory sees 3x refined output

Each block's W_K, W_V, W_Q projections and memory M learn different patterns
because they operate at different depths. Cloning a single-block checkpoint
throws away this structure — all 4 blocks start identical and must
re-specialize, wasting the warm start.

### Why stacking alone doesn't fix dead L2/L3

The dead-level problem is a **rate bottleneck**, not a signal-depth problem:

| Factor | L0 | L3 | Ratio |
|--------|----|----|-------|
| Fires per step | 1 | 1/512 | 512x |
| Initial theta | 0.0325 | 0.0005 | 65x |
| Effective learning rate | 0.0325 | ~1e-6 | ~33,000x |

Adding blocks does not change L3's firing frequency or initial theta. L3 in
block 3 still fires every 512 steps with theta≈0.0005, regardless of how
refined its input is. The solution is push-up: give L3 a pre-trained M matrix
so it starts warm, bypassing the cold-start bootstrapping entirely.

### The combined strategy

Stacked push-up gives both:
1. **Depth specialization** — each block's levels learn at the correct depth
2. **Warm start** — slow levels inherit trained M matrices, bypassing the
   rate bottleneck

## Prerequisites

| Prereq | Status | Why needed |
|--------|--------|-----------|
| BUG-01 (W_O projection) | task_2a31af, in_progress | Stacked model NaNs without W_O |
| BUG-02 (MAG sigmoid gating) | task_a83090, open | Unbounded memory → residual divergence |
| BUG-03 (alpha_mem aggregation) | task_604923, open | Needed at k≥2 only; can defer to phase 2 |
| Push-up checkpoint extension | PR #176, merged | Level remapping logic |

BUG-01 and BUG-02 are hard blockers for phase 1. BUG-03 is soft — at k=1 there
is only one level per block, so alpha_mem aggregation is a no-op. BUG-03 must
be resolved before phase 2 (k=2).

## Experiment Protocol

### Phase 1 — n_blocks=4, k=1 (seed)

**Goal**: Train 4 blocks, each with a single L0. Establish inter-block residual
dynamics. Validate that the stacked model converges (no NaN past 1000 steps).

```text
Config:
  n_blocks: 4
  k: 1
  chunk_sizes: [1]
  d_model: 512
  optimizer: adamw_gpu_stacked
  steps: 20000
  lr: 0.0003
  warmup_steps: 1000
  save_every: 5000
```

**Why 20K steps** (not 10K like single-block phase 1):
- 4 blocks have 4x the parameters to stabilize
- Inter-block residual dynamics need time to settle
- The residual stream connects all blocks — block 3 can't converge until
  blocks 0-2 produce useful features

**Success criteria**:
- No NaN/Inf through 20K steps
- Loss decreasing (>30% reduction from step 0)
- All 4 blocks' W_O, W_Q, W_K, W_V have nonzero gradients
- L0 gnorm per block > 0.01 (all 4 blocks learning)

**Output**: `checkpoints/stacked_pushup_p1_4b_k1.safetensors`

### Phase 2 — n_blocks=4, k=2 (first promotion)

**Goal**: Push each block's L0 → L1, fresh L0 per block. Validate that all 4
L1s retain their depth-specialized weights and that fresh L0s bootstrap.

```text
Config:
  n_blocks: 4
  k: 2
  chunk_sizes: [1, 8]
  load: checkpoints/stacked_pushup_p1_4b_k1.safetensors
  extend_k: 2
  push_up: true
  steps: 15000
  lr: 0.0003
  warmup_steps: 500
  data_seek: <monotonic, no re-exposure>
```

**Key difference from single-block push-up**: The extend_k/push_up logic must
operate **per block**. Each block independently shifts its level.0 → level.1
and gets a fresh level.0. The shared embed/unembed and final LN are unchanged.

**BUG-03 becomes relevant here**: With k=2, `alpha_mem` must weight the two
level outputs per block. If BUG-03 is not yet fixed, uniform weighting applies
(acceptable for k=2 but not ideal).

**Success criteria**:
- L1 gnorm per block > 0.001 (promoted levels still contributing)
- L0 gnorm per block > 0.01 (fresh levels bootstrapping)
- Loss at end of phase 2 ≤ loss at end of phase 1

**Output**: `checkpoints/stacked_pushup_p2_4b_k2.safetensors`

### Phase 3 — n_blocks=4, k=3

```text
Config:
  n_blocks: 4
  k: 3
  chunk_sizes: [1, 8, 64]
  load: checkpoints/stacked_pushup_p2_4b_k2.safetensors
  extend_k: 3
  push_up: true
  steps: 15000
```

**Success criteria**:
- L2 gnorm per block > 0.001 (2x-promoted levels still viable)
- L2 fires per block ≥ 15 during phase (15000/64 = 234 fires — plenty)
- Loss continuing to decrease or flat (not increasing)

**Output**: `checkpoints/stacked_pushup_p3_4b_k3.safetensors`

### Phase 4 — n_blocks=4, k=4 (production)

```text
Config:
  n_blocks: 4
  k: 4
  chunk_sizes: [1, 8, 64, 512]
  load: checkpoints/stacked_pushup_p3_4b_k3.safetensors
  extend_k: 4
  push_up: true
  steps: 100000
  data_seek: 0  (full corpus restart for production)
  m_norm_max: [100.0, 100.0, 100.0, 100.0]
  error_clip: [50.0, 50.0, 50.0, 50.0]
  theta_ceil: [2.0, 2.0, 2.0, 2.0]
```

**The headline test**: All 16 level instances (4 blocks × 4 levels) active and
contributing. L3 gnorm > 0.001 sustained beyond 10K steps (cold-start baseline:
L3 peaks at 0.000128 then collapses to 0.000002).

**Success criteria**:
- L3 gnorm per block sustained > 0.001 after 10K steps in phase 4
- All 16 level instances have nonzero gradient
- No NaN through 100K steps
- Final eval loss better than cold-start k=4 stacked baseline (if one exists)
- NIAH test at depth 3584: positive lift (if NIAH infra from spec 12 is ready)

**Output**: `checkpoints/stacked_pushup_p4_4b_k4.safetensors`

## Implementation Considerations

### extend_k must work per-block in stacked checkpoints

The current push-up logic (spec 07, PR #176) operates on single-block
checkpoint keys:
```
level.0.w_k → level.1.w_k
level.0.w_v → level.1.w_v
...
```

For stacked checkpoints, the key remapping must operate per block:
```
block.0.level.0.w_k → block.0.level.1.w_k
block.1.level.0.w_k → block.1.level.1.w_k
block.2.level.0.w_k → block.2.level.1.w_k
block.3.level.0.w_k → block.3.level.1.w_k
```

Each block gets its own fresh level.0 — they are NOT clones of each other.
Independent Xavier init with different seeds per block.

### Optimizer: adamw_gpu_stacked

The stacked optimizer must handle per-block parameter groups. The extend_k
promotion resets AdamW state for all params (same as single-block: fresh m_t=0,
v_t=0, warmup restarts).

### Data advancement

Same monotonic cursor advancement as spec 07/12. No data re-exposure between
phases. Phase 4 rewinds to position 0 for a full production run (all levels are
warm, the model needs to see the full corpus).

## Relationship to Existing Experiments

| Experiment | Config | Status | Relationship |
|-----------|--------|--------|-------------|
| EXP-09 | MAC k=4 cold-start | Dead L3 | Baseline — stacked push-up should beat this |
| EXP-10 | MAG k=4 cold-start | Dead L3 | Baseline — stacked push-up should beat this |
| Single-block push-up | k=1→4 single block | Phase 1-2 run | Push-up works for single block; this extends to stacked |
| Stacked shakedown | n_blocks=4 k=4 cold | NaN at ~460 | Blocked by BUG-01/02/03; stacked push-up starts at k=1 |

## Control Experiment

If time permits, run a **cloned push-up** control: train single-block k=1→4,
then clone that block 4 times into a stacked checkpoint. Compare final loss and
L3 activation against the stacked push-up. If stacked push-up shows
meaningfully better L3 gnorm, the depth-specialization hypothesis is confirmed.

## Falsification Criteria

This experiment is falsified if:

1. **Phase 1 NaNs despite BUG-01/02 fixes** — the stacked architecture has a
   deeper instability not caused by W_O/gating. Would require architectural
   rethinking.

2. **L3 gnorm in phase 4 matches cold-start baseline** (~0.000002) — push-up
   does not transfer across the stacked architecture, the warm start is lost
   during promotion in stacked context.

3. **Cloned push-up matches stacked push-up** — depth specialization during
   phase 1 provides no benefit over cloning a single-block checkpoint. Would
   mean stacking's value is purely parameter count, not depth.

4. **All 4 blocks converge to identical weights** despite different
   initialization and different positions in the residual stream. Would mean
   the residual stream does not differentiate blocks.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-097-hope-cms-chain | hope_equations | HOPE §5.1 | adapts (stacked variant) |
| eq-070-arch-variant1 | hope_equations | HOPE CMS chain | implements |

## Code Smells

| Smell | Enforcement | Rationale |
|-------|-------------|-----------|
| CS-04/05/06 | ontological | "blocks" for depth, "levels" for CMS frequency |
| CS-10 | behavioral | No train/eval — forward identical in all phases |
| CS-18 | architectural | Push-up orchestration in Python, math in Rust |
| CS-32 | behavioral | Observe-then-advance per block per level |
