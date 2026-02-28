# CMS Multi-Scale Ablation Study

```text
CONTRACT
  Purpose:    Determine whether CMS multi-scale structure (k=4, levels L0–L3) provides
              a measurable perplexity advantage over single-level (k=1) and no-memory
              baselines on a corpus with genuine long-range token structure, and whether
              the state-dependent DGD inner-loop optimizer improves over standard GD at
              the same scale.

  Expects:    - Build corpus selected and validated by specs/infrastructure/02_corpus_selection.md
                (lag-MI ESTR ratio ≥ 2.0× at lag=512 vs background). Selected: allenai/c4 (en).
              - Corpus prepared: python/data/c4/{train,val}_tokens.npy + meta.json
              - All 4 run configs present and validated:
                  python/configs/ablation_A_no_memory.json
                  python/configs/ablation_B_k1_titans.json
                  python/configs/ablation_C_k4_cms.json
                  python/configs/ablation_D_k4_dgd.json
              - GPU available; A6000 (46 GiB) sufficient for d=512 k=4 at 100K steps
              - HADES nl_experiments: hecate-experimental-plan-2026-02 node exists

  Guarantees: - All 4 runs share identical hyperparameters except the independent variable
                (memory_enabled, k, memory_rule). Cold-start only — no resumed checkpoints.
              - Primary metric: eval perplexity at step 100K on held-out C4 validation set.
              - Results interpreted only after all 4 runs complete (no partial conclusions).
              - Gate diagnostics (θ, ‖M‖ per level) logged every eval_every=1000 steps.
              - All checkpoints saved every 5000 steps for post-hoc analysis.

  Cost:       ~4–6 GPU-hours per run × 4 runs = ~16–24 GPU-hours total on A6000.
              ~1.5 GB disk per checkpoint × 20 checkpoints × 4 runs = ~120 GB checkpoints.

  Trade-off:  60M-parameter model at d=512 is well below HOPE paper scale (760M). The
              perplexity ratios may compress relative to Table 6. The study tests whether
              the _direction_ of the effect holds at smaller scale — not whether the
              absolute magnitude matches.

  Position:   specs/infrastructure/05_ablation_study.md
  Source:     HOPE (2512.24695) Table 6 ablation (§5.3)
                HADES: hope_equations/eq-088-practical-dgd-update (DGD update rule)
                       hope_equations/eq-096-hope-dgd-update (DGD variant)
                       hope_equations/eq-097-hope-cms-chain (CMS frequency chain)
              HOPE (2512.24695) §4.5 DGD inner-loop optimizer
              HOPE (2512.24695) §3 CMS frequency levels [1, 8, 64, 512]
              Titans (2501.00663) §3.2 Delta Rule memory update
                HADES: titans_equations/eq-034-deltanet-update (Delta Rule)
              MIRAS (2504.13173) §3 Delta Rule / standard GD
                HADES: miras_equations/eq-009-delta-rule (Delta Rule update)
                       miras_equations/eq-005-basic-gd-update (standard GD)
  Related:    specs/infrastructure/02_corpus_selection.md (corpus prerequisite)
              docs/research_notes/nlm_initialization_dynamics.md
              docs/committee_response_06.md (k=1 vs k=4 FineWeb-Edu findings)
```

---

## Hypothesis

The CMS frequency hierarchy (levels L0–L3 at periods [1, 8, 64, 512]) provides
a measurable perplexity advantage over single-level and no-memory baselines **when
the build corpus has genuine multi-timescale token structure** — that is, when
ESTR(lag=512) > 2× background rate.

FineWeb-Edu failed this criterion (ratio=1.00×). C4 passes (ratio=7.86×). The prior
k=1 vs k=4 comparison on FineWeb-Edu was therefore inconclusive about the architecture;
it reflected corpus choice, not model capacity.

**Primary prediction**: After 100K build steps on C4,
```text
ppl(Run D) < ppl(Run C) < ppl(Run B) < ppl(Run A)
```

with ppl(B)/ppl(D) matching the direction of HOPE Table 6:
```text
Table 6 (HOPE, 760M tokens, Wikitext-103):
  B (k=1 Titans):  14.68 ppl  (normalized to A=baseline)
  C (k=4 delta):   13.76 ppl  (−6.3% vs B)
  D (k=4 DGD):     13.41 ppl  (−8.6% vs B, −2.5% vs C)
```

At 60M-parameter scale the absolute ppl will be higher, but the rank order
and directional improvement percentages should hold.

**Secondary prediction**: L2/L3 gates activate on C4 (θ_L2 > 0.01, θ_L3 > 0.003
at step 20K for Runs C and D). This distinguishes the data-limited hypothesis
from the initialization-trap hypothesis identified in committee_response_06.

---

## Failure Criteria

The study is invalidated (conclusions inconclusive) if ANY of:
- Runs B, C, D converge to ppl within 1% of Run A (no memory benefit at all)
- L2/L3 gates remain dormant (θ_L2 < 0.003 and θ_L3 < 0.001 throughout 100K steps)
  in Runs C and D — this would indicate C4 preparation is broken or an initialization
  trap that requires additional warmup protocol (hypothesis B from committee_response_06)
- Any run diverges (loss NaN or ppl > 10× initial) before step 50K
- A run was resumed from a checkpoint rather than cold-started

If any failure criterion triggers, open a GitHub issue, mark downstream
ABLATION-A through ABLATION-D tasks as blocked with reason, and do not
report partial results as findings.

---

## The 4-Run Design

All runs share these controlled variables:

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| num_heads | 8 |
| seq_len | 512 |
| window_size | 512 |
| vocab_size | 32000 |
| optimizer | adamw_gpu |
| lr | 0.0004 |
| warmup_steps | 200 |
| weight_decay | 0.1 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| max_grad_norm | 1.0 |
| steps | 100000 |
| save_every | 5000 |
| eval_every | 1000 |
| eval_max_chunks | 100 |
| seed | 42 |
| corpus | allenai/c4 (en), prepared at python/data/c4/ |
| m_norm_max | [100.0] × k (NaN safety, never active at healthy scale) |
| theta_floor | **absent** (no artificial gate floors — see committee_response_06 §2) |

The **only** differences between runs are:

| Run | memory_enabled | k | chunk_sizes | memory_rule | composition | Notes |
|-----|---------------|---|------------|-------------|-------------|-------|
| **A** | false | 1 | [1] | delta | mag | SWA-only baseline |
| **B** | true | 1 | [1] | titans | mag | k=1 TitansLMM (DGD) |
| **C** | true | 4 | [1,8,64,512] | delta | mag | k=4 standard GD |
| **D** | true | 4 | [1,8,64,512] | titans | mag | k=4 DGD (L2 regression) |

### Why these 4 runs

**A → B** tests whether any memory module helps vs no-memory SWA (memory benefit
question). Expected: Run B better than Run A by ≥3%.

**B → C** is a mixed comparison (B=titans/k=1, C=delta/k=4) and is not the primary
axis. The cleanest comparisons are A→B (memory on/off at k=1) and B→D (k=1 vs k=4
under identical DGD optimizer). C is included to isolate the optimizer contribution
at k=4 scale via the C vs D axis.

**C vs D** tests the inner-loop optimizer: state-independent GD (delta rule,
`M_{t+1} = M_t + θ·v⊗k`) vs state-dependent DGD (TitansLMM,
`M_{t+1} = M_t·(I - θ·α·k⊗k) + θ·v⊗k`). Both are at k=4 for maximum
statistical power to observe the difference.

**B vs D** tests whether k=4 + DGD outperforms k=1 + DGD (the CMS multi-scale
benefit under identical optimizer). This is the cleanest test of the CMS claim.

### Memory rule semantics (CS-33/CS-34 compliant)

```text
memory_rule = "delta"
  Algorithm: standard GD (state-independent)
  Update:    M_{t+1} = (1 - alpha_t) * M_t + theta_t * v_t ⊗ k_t
  Source:    MIRAS (2504.13173) §3, Delta Rule (Hebbian with retention)
  HADES:     miras_equations/eq-009-delta-rule
             miras_equations/eq-005-basic-gd-update

memory_rule = "titans"
  Algorithm: DGD — L2 regression inner loop (state-dependent)
  Update:    error_t = M_t k_t - v_t
             M_{t+1} = (1 - alpha_t) * M_t - theta_t * biased(error_t) ⊗ k_t
  Source:    HOPE (2512.24695) §4.5, Eq 88; Titans (2501.00663) §3.2
  HADES:     hope_equations/eq-088-practical-dgd-update
             titans_equations/eq-034-deltanet-update
```

---

## Configuration Files

Authoritative configs are at:
- `python/configs/ablation_A_no_memory.json`
- `python/configs/ablation_B_k1_titans.json`
- `python/configs/ablation_C_k4_cms.json`
- `python/configs/ablation_D_k4_dgd.json`

They are the executable form of this spec — the spec and configs must stay in sync.

---

## Execution Order

Runs A and B can execute in parallel (independent). Runs C and D can execute
in parallel (independent). A+B must complete before drawing conclusions about
C+D, but C+D may start immediately.

Recommended execution with two GPUs:
```text
GPU 0: Run A and Run C sequentially (no-mem baseline + k=4 delta)
GPU 1: Run B and Run D sequentially (k=1 titans + k=4 titans/DGD)
```

---

## Logging and Diagnostics

Each run's JSONL log must capture:
- Step, loss, eval_ppl every `log_every` steps
- For memory-enabled runs (B, C, D): per-level θ, ‖M‖_F in the checkpoint JSON

Post-run analysis:
1. Extract ppl at step 100K from each run's JSONL
2. Compute ratios: B/A, C/A, D/A, D/B, D/C
3. Compare ratio D/B to HOPE Table 6 pattern (expected ~0.91)
4. Plot θ(t) per level for Runs C and D to verify gate activation on C4

---

## Acceptance Criteria

- [ ] All 4 runs complete to step 100K without NaN or early termination
- [ ] L2/L3 gate activation confirmed for Run D (θ_L2 > 0.01, θ_L3 > 0.003 by step 20K)
- [ ] Primary prediction direction confirmed: ppl(D) ≤ ppl(C) ≤ ppl(B) ≤ ppl(A)
  OR failure criteria triggered and documented
- [ ] HADES nl_experiments node updated with final ppl values for all 4 runs
- [ ] ABLATION-A through ABLATION-D tasks closed in Persephone

---

## Relationship to HOPE Table 6

HOPE Table 6 compares:
- SWA attention (no memory): baseline
- Titans at k=1: −6.0% vs baseline
- k=4 CMS: −7.8% vs baseline
- k=4 CMS + DGD: −9.5% vs baseline

Source: HOPE (2512.24695) §5.3, Wikitext-103, 760M-param model.

This study reproduces the same ablation design at 60M parameters on C4 (en).
The scale reduction means absolute ppl will differ; the study tests
whether the **rank order and directional improvement** replicate at smaller scale.

Failure to replicate the direction would be a finding: either the CMS benefit
only emerges at large scale (scale threshold hypothesis) or the corpus
selection (C4 vs Wikitext-103) changes the result.
