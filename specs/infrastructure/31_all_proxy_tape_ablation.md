# All-Proxy Tape Ablation: L0 Exact vs L0 Proxy

```text
CONTRACT
Purpose   : Ablation experiment — train k=2 from scratch with
            tape_strategies=[proxy, proxy] (all levels proxy) and
            alpha_floor=0.0, comparing against the identical config
            with tape_strategies=[exact, proxy]. Tests whether exact
            Wengert tape on L0 provides measurable benefit over the
            chunkwise proxy approximation that the papers specify.

Expects   : GPU stacked optimizer (gpu_stacked_optimizer.rs) with
            per-level tape strategy dispatch (spec 27). BuildConfig
            (python/engine/config.py) with tape_strategies array.
            Dolmino-100B data. Control run (k2_zero_retention_d512)
            already running with [exact, proxy].

Guarantees: 1. k=2 model trains from scratch with
               tape_strategies=[proxy, proxy], alpha_floor=[0.0, 0.0].
            2. All other config parameters identical to control run
               (k2_zero_retention_d512): d=512, 4 blocks, 8 heads,
               lr=3e-4, seed=42, 200K steps, Dolmino-100B.
            3. Tape summaries log gate distributions per level at
               tape_every=1000, enabling direct comparison with
               control arm.
            4. No code changes required — config-only experiment.

Cost      : ~200K steps on A6000. At k=2 d=512, estimated ~1200 tok/s
            (possibly higher than control due to no tape record/replay
            overhead on L0). Wall time: ~24h.

Trade-off : Proxy gradients on L0 are anchored to chunk-start state,
            not current state. This introduces gradient staleness
            proportional to L0's chunk_size (1 token). At chunk_size=1,
            the staleness is minimal — proxy and exact should produce
            very similar gradients. The experiment tests whether this
            theoretical similarity holds in practice over 200K steps.

Position  : specs/infrastructure/31_all_proxy_tape_ablation.md

Source    : TNT (2511.07343) eq-003 — Chunkwise compression: gradient
            computed w.r.t. chunk-start state, not current state.
            TNT eq-005 — Global memory update: same chunkwise formulation.
            TNT eq-006 — Local memory update: same chunkwise formulation
            for each local memory at every frequency.
            Titans (2501.00663) eq-016 — Chunk-wise mini-batch GD:
            anchors gradients to chunk boundaries.
            NONE of the papers use exact/full BPTT for any level.
            Our exact-on-L0 is an implementation choice (spec 27),
            not paper-derived.
```

---

## 1. Hypothesis

The exact Wengert tape on L0 may provide negligible benefit over proxy
at chunk_size=1. At chunk_size=1, the proxy approximation computes
gradients w.r.t. the memory state at the start of the chunk — which IS
the current token's state. The "staleness" that makes proxy an
approximation for larger chunk sizes is essentially zero at chunk_size=1.

**Hypothesis A (no difference):** All-proxy matches exact within noise.
The exact tape on L0 was safety margin, not contributing signal. We can
drop it, save VRAM, and gain throughput.

**Hypothesis B (exact helps):** Exact tape gives L0 better gradients
that compound over 200K steps. The difference is small per-step but
accumulates. If so, quantify the gap to decide if the VRAM cost is
justified at scale.

**Hypothesis C (proxy helps):** The proxy's implicit staleness acts as
regularization, producing smoother optimization. Unlikely but would be
a significant finding.

---

## 2. Experiment Configuration

### Control arm (already running)

```json
{
    "description": "k=2 zero-floor, L0 exact tape (control)",
    "run_dir": "runs/k2_zero_retention_d512",
    "tape_strategies": ["exact", "proxy"],
    "alpha_floor": [0.0, 0.0]
}
```

### Test arm (this experiment)

```json
{
    "description": "k=2 zero-floor, all proxy tape (test)",
    "model": {
        "d_model": 512,
        "num_heads": 8,
        "seq_len": 512,
        "window_size": 512,
        "vocab_size": 50257,
        "memory_rule": "titans",
        "composition": "mag",
        "k": 2,
        "chunk_sizes": [1, 8],
        "m_norm_max": [100.0, 100.0],
        "error_clip": [0.0, 0.0],
        "residual": true,
        "n_blocks": 4,
        "parallel_strategy": "tnt_hierarchical",
        "tnt_global_chunk_size": 64,
        "tnt_local_chunk_size": 8,
        "memory_reset": "periodic",
        "tape_multiplier": 1,
        "tape_strategies": ["proxy", "proxy"]
    },
    "build": {
        "lr": 3e-4,
        "steps": 200000,
        "warmup_steps": 500,
        "alpha_floor": [0.0, 0.0],
        "theta_ceil": [1.0, 1.0],
        "tape_every": 1000,
        "save_every": 5000,
        "seed": 42
    }
}
```

### Only difference: tape_strategies

| Parameter | Control | Test |
|-----------|---------|------|
| tape_strategies | [exact, proxy] | [proxy, proxy] |
| Everything else | identical | identical |

---

## 3. Metrics to Watch

### Primary: Loss trajectory comparison

| Checkpoint | Control (exact+proxy) | Test (all proxy) | Delta |
|------------|----------------------|-------------------|-------|
| 10K | TBD | TBD | |
| 30K | TBD | TBD | |
| 50K | TBD | TBD | |
| 100K | TBD | TBD | |
| 130K | TBD (compare to 2.60 baseline) | TBD | |
| 200K | TBD | TBD | |

### Secondary: Throughput

All-proxy should be faster — no tape recording/replay on L0.
Measure tok/s delta as percentage.

### Tertiary: Gate dynamics

- Do L0 alpha/theta distributions differ without exact gradients?
- Does L0 find the same retention rate (~0.64 at step 3000)?
- Any difference in gradient norm patterns?

---

## 4. Success Criteria

1. **If |loss_proxy - loss_exact| < 5% at 130K**: All-proxy is viable.
   Recommend dropping exact tape from default configs. Proceed to
   scale-up with all-proxy.

2. **If loss_proxy > loss_exact by >10%**: Exact tape is load-bearing.
   Keep it for L0. Document the cost-benefit for scale-up decisions.

3. **If loss_proxy < loss_exact by >5%**: Proxy regularization effect.
   Needs replication. If confirmed, switch to all-proxy permanently.

---

## 5. VRAM and Throughput Implications

### At d=512 (this experiment)

- Exact L0 tape: stores full M+S trajectory per shard (8 timesteps × 2 × d² × 4 bytes = 16MB/block)
- Proxy L0: stores only M_final + S_final (2 × d² × 4 bytes = 2MB/block)
- Savings: ~56MB across 4 blocks — modest

### At scale (motivation for this experiment)

| d | Exact L0 overhead | Savings if all-proxy works |
|---|-------------------|---------------------------|
| 512 | 56MB | Modest |
| 1024 | 224MB | Meaningful |
| 2048 | 896MB | Significant |
| 4096 | 3.5GB | Game-changing |

### Throughput

Tape record/replay adds overhead per forward/backward pass. All-proxy
eliminates this for L0. Expected throughput gain: 5-15% (to be measured).

---

## 6. No Code Changes Required

This is a config-only experiment. The tape_strategies field already
accepts "proxy" for any level. The GPU stacked forward/backward paths
dispatch per-level based on the strategy array.

---

## 7. Ontological Compliance

- **CS-10**: No mode flag. Same forward pass.
- **CS-18**: Tape strategy is configured in Python, dispatched in Rust.
- **CS-32**: Observe-then-advance. Proxy still reads M at chunk start.
- **CS-40**: Opt-in AD. With all-proxy, tape recording is not activated
  for L0 — consistent with CS-40's "off by default" principle.

---

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-003-chunkwise-compression | tnt_equations | TNT §3 (2511.07343) | implements |
| eq-005-global-memory-update | tnt_equations | TNT §3.1 (2511.07343) | implements |
| eq-006-local-memory-update | tnt_equations | TNT §3.2 (2511.07343) | cites |
| eq-016-chunk-wise-gd | titans_equations | Titans §3.4 (2501.00663) | cites |
