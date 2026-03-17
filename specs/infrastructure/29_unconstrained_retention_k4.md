# Unconstrained Retention k=4: Zero Alpha Floor Ablation

```text
CONTRACT
Purpose   : Ablation experiment — train k=4 from scratch with alpha_floor=0.0
            on all four levels, allowing each level to learn its own retention
            rate without constraint. Tests the hypothesis that the alpha_floor=0.8
            constraint prevents slow-frequency CMS levels (L2, L3) from adapting
            their retention to the timescale mismatch between their update frequency
            and the continuously-updating attention/embedding weights.

Expects   : GPU stacked optimizer (gpu_stacked_optimizer.rs) with per-level
            alpha_floor config. BuildConfig (python/engine/config.py) with
            per-level alpha_floor array. Spec 27 per-level tape strategy
            (L0=exact, L1-L3=proxy). Spec 28 dormancy sentinel (shard M-diff
            logging + dormancy tracking). Dolmino-100B data.

Guarantees: 1. k=4 model trains from scratch (no push-up) with
               alpha_floor=[0.0, 0.0, 0.0, 0.0].
            2. All four CMS levels receive unconstrained alpha gates (sigmoid
               output range [0.0, 1.0] — full forgetting to full retention).
            3. Tape summaries log alpha distributions per level, enabling
               comparison with alpha_floor=0.8 baselines.
            4. Dormancy sentinel monitors all proxy levels (L1-L3) for
               collapse detection.
            5. No code changes required — config-only experiment.

Cost      : ~100K steps on A6000. At k=4 d=512, estimated ~800 tok/s
            (lower than k=2 due to 4 levels). Wall time: ~18h.

Trade-off : Without the alpha floor, levels CAN drive alpha to zero —
            complete memory erasure each step. This risks training instability
            (no persistent memory). The experiment tests whether the model
            finds a useful retention regime on its own vs. collapsing.

Position  : specs/infrastructure/29_unconstrained_retention_k4.md

Source    : Titans (2501.00663) eq-013 — M_t = α_t · M_{t-1} + S_t
            Alpha is the data-dependent retention gate: α_t = σ(w_α · [k_t, v_t] + b_α).
            The Titans paper does not specify a floor — α is sigmoid-bounded [0,1] naturally.
            The alpha_floor=0.8 was our implementation choice (CS-39) to prevent
            catastrophic forgetting. This experiment tests whether it's helping or hurting.
            HOPE (2512.24695) §5.1 — CMS frequency structure
            HOPE eq-097 — CMS chained frequency rule
            HOPE eq-074 — CMS independence (levels do not share state)
            Internal: pushup_k3_from_spec27_30k report (2026-03-17) — L2 dormancy,
            alpha trending toward floor in all levels, floor-pinning in 10/12 block×level combos.
            Internal: spec27_d512_4b_k2 130K report (2026-03-17) — L1 eta collapse,
            universal alpha floor-pinning.
```

---

## 1. Hypothesis

The alpha_floor=0.8 constraint is preventing slow-frequency CMS levels from
finding their natural retention equilibrium. Evidence:

**Observation 1 — Universal floor-pinning.** In the k=3 push-up experiment
(95K steps), 10 of 12 block×level combinations hit the alpha floor (0.80).
In the k=2 130K experiment, all L1 alpha values pinned to floor. The model
is pressing against the constraint everywhere.

**Observation 2 — L2 dormancy.** L2 (chunk_size=64) fires every 64 steps.
Between fires, the attention weights it feeds into receive 64 AdamW updates.
L2's memory becomes stale relative to the current attention context. With
alpha_floor=0.80, L2 retains 80% of this stale memory and can only
overwrite 20% per fire. The slow overwrite rate may prevent L2 from
tracking the attention drift.

**Observation 3 — Timescale mismatch.** The alpha floor was calibrated for
L0 (fires every token). At L0's frequency, 80% retention means memory
decays with halflife ~3 tokens — reasonable. At L2's frequency (fires
every 64 tokens), 80% retention per-fire means effective halflife of
~3 fires = ~192 tokens measured in attention-weight-update steps. At L3
(every 512), halflife ~1536 steps. The same floor creates wildly different
effective retention timescales across levels.

**Hypothesis:** Removing the alpha floor allows slow levels to:
- Drive alpha lower (faster forgetting) when their memory is stale
- Drive alpha higher (slower forgetting) when they capture long-range patterns
- Find level-specific retention rates tuned to their actual update frequency

**Alternative hypothesis (failure mode):** Without the floor, all levels
drive alpha toward 0 early in training (before memory has useful content),
creating a "no-memory" degenerate solution where the model reduces to pure
attention. The sigmoid bias initialization (b_alpha=3.0 → sigmoid≈0.95)
may provide enough initial retention to avoid this, but it's not guaranteed.

---

## 2. Experiment Configuration

```json
{
    "description": "k=4 from scratch, zero alpha floor (unconstrained retention)",
    "model": {
        "d_model": 512,
        "num_heads": 8,
        "seq_len": 512,
        "window_size": 512,
        "vocab_size": 50257,
        "memory_rule": "titans",
        "composition": "mag",
        "k": 4,
        "chunk_sizes": [1, 8, 64, 512],
        "m_norm_max": [100.0, 100.0, 100.0, 100.0],
        "residual": true,
        "n_blocks": 4,
        "parallel_strategy": "tnt_hierarchical",
        "tape_strategies": ["exact", "proxy", "proxy", "proxy"]
    },
    "build": {
        "lr": 3e-4,
        "steps": 100000,
        "warmup_steps": 500,
        "alpha_floor": [0.0, 0.0, 0.0, 0.0],
        "theta_ceil": [1.0, 1.0, 1.0, 1.0],
        "tape_every": 1000,
        "save_every": 5000
    }
}
```

### Key design choices

- **k=4 from scratch, not push-up.** This avoids confounding the retention
  question with push-up dynamics. All four levels start from random init
  simultaneously. If the unconstrained retention helps, we'll see it in the
  raw convergence curve.

- **tape_strategies: [exact, proxy, proxy, proxy].** L0 gets exact gradients,
  L1-L3 get proxy. Same as all our k≥2 runs. This is the standard config
  for k=4 — the retention question is orthogonal to tape strategy.

- **b_alpha initialization.** The default b_alpha=3.0 gives sigmoid(3.0)≈0.95
  initial retention. Even without a floor, levels start with high retention
  and must learn to reduce it. This provides a soft initial bias toward
  retention that the model can override.

---

## 3. Metrics to Watch

### Primary: Alpha distribution per level over time

The central question: what alpha values does each level converge to?

| Outcome | Interpretation |
|---------|---------------|
| All levels α ≈ 0.8-0.95 | Floor was unnecessary — model naturally retains |
| L0 α ≈ 0.9, L2/L3 α ≈ 0.3-0.5 | Levels find frequency-appropriate retention |
| L2/L3 α → 0 | Catastrophic — slow levels erase memory, degenerate to no-memory |
| L0 α → 0, L2/L3 α ≈ 0.9 | Inverted — fast level forgets, slow levels retain (unexpected but interesting) |

### Secondary: Level differentiation

- **Theta per level.** Does unconstrained alpha change the theta (inner LR) dynamics?
  In the floor-constrained runs, L2/L3 theta went to ~0. With unconstrained alpha,
  levels can forget faster, which may allow them to write more meaningful new content
  (theta > 0) since the memory isn't saturated with stale state.

- **Loss trajectory.** Compare with k=2 spec27 130K baseline (final loss 2.60) and
  k=3 push-up baseline (best loss 2.53). k=4 from scratch has more parameters but
  also more levels to bootstrap.

- **Dormancy status.** Use spec 28 dormancy sentinel to detect if proxy levels
  go dormant even with unconstrained retention.

### Tertiary: Block coherency

- **Per-block alpha variance.** Do blocks converge to similar alpha profiles, or does
  the Block 2 anomaly persist?
- **Block CV trajectory.** Does unconstrained retention affect depth specialization?

---

## 4. Success Criteria

The experiment succeeds (informs next steps) regardless of whether the model
improves. Specific outcomes:

1. **If loss < 2.60** (beats k=2 130K baseline): Unconstrained retention is
   viable at k=4. Proceed to longer runs and scale-up.

2. **If loss > 3.0 but levels show differentiated alpha**: Retention freedom
   helps differentiation but training needs more steps. Extend to 200K.

3. **If any level α → 0 and loss diverges**: Floor IS load-bearing. Consider
   per-level floors scaled by frequency (e.g., L0=0.8, L1=0.6, L2=0.3, L3=0.1).

4. **If all levels α ≈ 0.8-0.95 naturally**: Floor was a safety net, not a
   constraint. Remove it permanently from default configs.

---

## 5. Comparison Runs

| Run | k | Alpha floor | Steps | Status |
|-----|---|-------------|-------|--------|
| spec25_30k_d512_4b_k2 | 2 | [0.8, 0.8] | 30K | Complete (loss 2.75) |
| spec27_d512_4b_k2_100k | 2 | [0.8, 0.8] | 130K | Complete (loss 2.60) |
| pushup_k3_from_spec27_30k | 3 | [0.8, 0.8, 0.8] | 95K | Complete (loss 2.53 best) |
| **This run** | **4** | **[0.0, 0.0, 0.0, 0.0]** | **100K** | **Planned** |

---

## 6. No Code Changes Required

This is a config-only experiment. The alpha_floor field already accepts 0.0 —
the sigmoid output simply goes unclamped. The gate diagnostics (spec 26) and
dormancy sentinel (spec 28) provide all needed observability.

---

## 7. Ontological Compliance

- **CS-10**: No mode flag. Same forward pass, same backward pass.
- **CS-18**: Alpha floor is an optimizer-tier constraint, configured in Python.
- **CS-32**: Observe-then-advance. Alpha is read from the gate buffer, clamped
  (or not) by the floor, then used in the memory update.
- **CS-39**: This experiment REMOVES CS-39's alpha floor. The spec acknowledges
  this is a deliberate ablation of a safety constraint. Results will inform
  whether CS-39 should be modified to use per-level scaling.

---

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-013-forgetting-gate | titans_equations | Titans §3.2 (2501.00663) | tests |
| eq-014-momentum-with-forgetting | titans_equations | Titans §3.2 (2501.00663) | cites |
| eq-097-hope-cms-chain | hope_equations | HOPE §7.1 (2512.24695) | cites |
| eq-074-cms-independence | hope_equations | HOPE §5.1 (2512.24695) | cites |
