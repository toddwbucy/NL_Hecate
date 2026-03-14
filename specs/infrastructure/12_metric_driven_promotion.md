# Metric-Driven Level Promotion

```
CONTRACT
  Purpose:    Replace fixed step-count level promotion with convergence-signal-driven
              promotion. Promotion fires when a level has extracted available signal
              from its current data window — not after an arbitrary number of steps.
              Includes per-level validation on novel data and a needle-in-a-haystack
              (NIAH) evaluation that tests the CMS memory hierarchy's retrieval
              capability at each k.

  Expects:    - CMS k>=1 with push-up stacking (spec 07_push_up_level_stacking.md)
              - Per-level gnorm tracking (gpu_per_level_grad_norms in gpu_optimizer.rs)
              - JSONL logging infrastructure (loop.py, evaluation.py)
              - ContextStream with seekable data cursor (data_seek in BuildConfig)
              - dolmino_100b or equivalent lag-MI-validated corpus (spec 02)
              - Scale-matched initialization in extend_push_up (model.rs)

  Guarantees: - Promotion is triggered by multi-signal convergence, not step count
              - Each level lifetime sees only novel tokens (monotonic data advancement)
              - Per-level validation uses held-out data from the CURRENT data window
              - NIAH eval tests long-range retrieval at each promotion boundary
              - Fixed-step mode retained as fallback (promotion_mode: "steps"|"convergence")
              - All promotion events logged with full metric state for post-hoc analysis

  Cost:       - Convergence metric collection: ~0 overhead (reuses existing gnorm_l)
              - Plateau detection: O(window_size) per eval step, negligible
              - Per-level validation: one extra eval pass per eval_every on window-local data
              - NIAH eval: one forward pass over 4096-token synthetic sequence per promotion boundary
              - No new Rust or CUDA code required for Phase 1

  Trade-off:  Convergence-driven promotion produces variable-length phases. Early phases
              (k=1, k=2) may converge faster than the fixed 20K/15K steps, saving GPU time.
              Late phases (k=3, k=4) may run longer because slow levels need more fires
              to establish a gradient trend. Total training time is data-dependent rather
              than predetermined. The fixed-step fallback provides a predictable upper bound.

  Position:   specs/infrastructure/12_metric_driven_promotion.md

  Source:     HOPE (2512.24695) Section 7.1 — CMS frequency structure
              HADES: hope_equations/eq-097-hope-cms-chain (CMS chained frequency rule)
              HADES: hope_equations/eq-100-freq-equilibrium (CMS frequency equilibrium)
              HADES: hope_equations/eq-102-gated-memory (write gate theta_t)
              HADES: hope_equations/eq-106-freq-weights (per-level inner LR)
              Internal: pyramid phase 3/4 data re-exposure failure (2026-03-07)
              Internal: cold-start k=4 report (docs/reports/cold_start_k4_100k_mag_mac_report.md)
              Related: specs/infrastructure/09_gate_warmup.md (gate dormancy analysis)
              Related: specs/infrastructure/10_memory_manifold_analysis.md (JS divergence, rank)
              Related: specs/infrastructure/07_push_up_level_stacking.md (promotion mechanism)
              Related: task_39914e (staged initialization curriculum — spiral variant)
```

---

## 1. Problem: Step Counts Are a Meaningless Promotion Signal

The pyramid curriculum uses fixed step counts for each phase:

```text
Phase 1 (k=1): 20K steps on tokens 0–10M
Phase 2 (k=2): 15K steps on tokens 0–7.7M
Phase 3 (k=3): 10K steps on tokens 0–5.1M
Phase 4 (k=4): 100K steps on tokens 0–51M
```

Two compounding failures:

**F1 — Data re-exposure.** Promoted levels inherit weights trained on the same tokens during prior
lifetimes. By phase 4, L2 and L3 have seen tokens 0–5M three to four times through different level
lifetimes. The inner loop's memory update (M_t = alpha * M_{t-1} + theta * v otimes k) saturates:
the projections already map these tokens to near-optimal memory states, so the gradient is near zero.
Observed: phase 4 gnorm_l = [14.4, 0.4, 0.0, 0.0] — L2 and L3 completely dead.

**F2 — Fixed steps ignore convergence rate.** At k=1, the model may converge in 12K steps or need
25K. At k=4, slow levels fire every 512 steps and may need 50K+ steps to accumulate enough gradient
updates. A fixed schedule cannot adapt to either case.

The fix: promotion is driven by convergence metrics, and the data cursor advances monotonically so
no level ever re-encounters data from a prior lifetime.

---

## 2. Convergence Signals

Four signals, ranked by reliability and implementation cost.

### 2.1 Primary: Per-Level Gradient Norm Trend (gnorm_l)

Already computed by `gpu_per_level_grad_norms()` in `gpu_optimizer.rs` and logged every `log_every`
steps. This is the most direct signal of whether a level is still learning.

**Metric**: Rolling median of gnorm_l[i] over the last N fires for level i.

**Why median, not mean**: gnorm_l is spiky (a single batch can produce 10x the typical value).
Median is robust to outliers while still tracking the trend.

**Fire-aligned windowing**: Slow levels fire infrequently. L2 (chunk=64) fires every 64 steps;
L3 (chunk=512) fires every 512 steps. The rolling window must be measured in FIRES, not steps,
to ensure each level has enough samples for a meaningful trend.

```text
Window size: W = 20 fires (configurable)
L0 (chunk=1):   20 fires = 20 steps   — trend established quickly
L1 (chunk=8):   20 fires = 160 steps  — ~30 seconds at 600 tok/s
L2 (chunk=64):  20 fires = 1280 steps — ~18 minutes
L3 (chunk=512): 20 fires = 10240 steps — ~2.4 hours
```

**Saturation criterion**: Level i is saturated when:

```text
median(gnorm_l[i], last W fires) < saturation_threshold
```

Default `saturation_threshold = 0.05` (configurable). This is well above numerical noise (~1e-4)
but below meaningful learning (~0.5+).

**Hysteresis**: A level must remain below threshold for H consecutive windows (default H=2)
before being declared saturated. This prevents premature promotion from a temporary low-gradient
batch.

### 2.2 Secondary: Validation Loss Plateau

Standard patience-based plateau detection on eval loss.

**Critical requirement**: The validation data MUST be drawn from the CURRENT data window, not a
fixed holdout set. A fixed val set suffers the same re-exposure problem as training data — promoted
levels memorized it during prior lifetimes.

**Per-window validation split**: When advancing the data cursor to position P, reserve a validation
slice:

```text
Training data: tokens [P, P + phase_budget - val_size)
Validation data: tokens [P + phase_budget - val_size, P + phase_budget)

val_size = 50000 tokens (configurable, ~100 eval chunks at seq_len=512)
```

**Plateau detection**:

```text
IF eval_loss has not improved by min_delta in patience consecutive evals:
    plateau_detected = true

Default: patience = 5 evals, min_delta = 0.01
```

### 2.3 Tertiary: Level Contribution Delta

Measures whether the most-recently-promoted level is contributing to loss reduction.
Computed as the difference in eval loss between:
  (a) full model forward pass (all levels active)
  (b) forward pass with the target level's gate forced to zero (level ablated)

```text
contribution_delta[i] = eval_loss(level_i_ablated) - eval_loss(all_levels)
```

If `contribution_delta[i] < 0.01`, level i is not meaningfully helping. This is more expensive
(requires an extra forward pass per level per eval) and is optional. Enable with
`track_level_contribution: true` in config.

### 2.4 Diagnostic: Memory Norm Velocity

The rate of change of ||M||_F per level between consecutive fires. When the memory matrix norm
stabilizes (velocity approaches zero), the memory has reached a fixed point for this data.

```text
m_norm_velocity[i] = |norm(M_i, fire_t) - norm(M_i, fire_{t-1})| / norm(M_i, fire_t)
```

This is logged but NOT used for promotion decisions (the m_norm_clamp at 100.0 confounds the
signal — once M hits the clamp, velocity goes to zero regardless of learning state). It serves
as a diagnostic cross-check against gnorm_l.

---

## 3. Promotion Decision Function

```text
FUNCTION: should_promote(metrics: &ConvergenceState, config: &PromotionConfig) -> bool

  -- Mode gate
  IF config.promotion_mode == "steps":
    RETURN metrics.step >= config.fixed_step_budget

  -- Primary signal: fastest active level saturated
  -- "Fastest active" = L0 in normal training. After promotion, still L0 (push-up inserts fresh L0).
  IF NOT gnorm_saturated(level=0, metrics, config):
    RETURN false   -- L0 still learning, don't promote yet

  -- Secondary signal: validation plateau
  IF config.require_val_plateau AND NOT metrics.val_plateau_detected:
    RETURN false

  -- Minimum step guard: ensure slow levels had enough fires for meaningful trend
  min_fires_slowest = config.min_fires_before_promotion  -- default: 15
  slowest_level = metrics.k - 1
  IF metrics.level_fire_count[slowest_level] < min_fires_slowest:
    RETURN false

  RETURN true
```

**Key design choice**: We gate on L0 saturation, not slowest-level saturation. L0 fires every
step and converges first. When L0 has extracted all signal from this data window, there is no
benefit to continuing — the slow levels receive gradient through L0's contribution to the loss,
and if L0's gradient is flat, the slow levels' gradient is also diminishing.

The `min_fires_before_promotion` guard ensures the slowest level has had enough gradient updates
to establish its weights before being promoted to an even slower frequency.

---

## 4. Data Window Advancement

Each promotion advances the data cursor monotonically. No level ever re-encounters data from a
prior lifetime.

```text
Phase 1 (k=1): data_seek = 0,          trains on tokens [0, N1)
Phase 2 (k=2): data_seek = N1,         trains on tokens [N1, N1+N2)
Phase 3 (k=3): data_seek = N1+N2,      trains on tokens [N1+N2, N1+N2+N3)
Phase 4 (k=4): data_seek = N1+N2+N3,   trains on tokens [N1+N2+N3, ...)
```

Where N_i = steps_in_phase_i * seq_len (tokens consumed in phase i).

In convergence mode, N_i is variable — determined by when the promotion signal fires. The
checkpoint records the data cursor position so the next phase can compute its starting offset.

**Checkpoint contract**: When saving at promotion time, the checkpoint MUST include:

```json
{
  "promotion_event": {
    "from_k": 2,
    "to_k": 3,
    "step": 14832,
    "data_cursor": 7593984,
    "metrics_at_promotion": {
      "gnorm_l_median": [0.032, 0.41],
      "eval_loss": 4.21,
      "val_plateau_steps": 3,
      "level_fire_counts": [14832, 1854]
    }
  }
}
```

This enables post-hoc analysis of what signals predicted successful promotions vs failures.

---

## 5. Per-Level Validation

### 5.1 Window-Local Validation Set

Each phase carves a validation slice from the END of its data window:

```text
Total window: [data_seek, data_seek + max_tokens)
Training:     [data_seek, data_seek + max_tokens - val_tokens)
Validation:   [data_seek + max_tokens - val_tokens, data_seek + max_tokens)
```

`max_tokens` is either the fixed phase budget (steps mode) or a configurable upper bound
(convergence mode). Default `val_tokens = 50000`.

The validation slice is loaded once at phase start and reused for all evals within that phase.
It is discarded at promotion (the next phase gets its own val slice from its own data window).

### 5.2 Per-Level Eval Metrics

At each eval step, log per-level diagnostics:

```text
[eval] step 5000 loss=4.21 ppl=67.3
  L0: gnorm_median=0.032 fires=5000 alpha=0.88 theta=0.054 ||M||=99.99
  L1: gnorm_median=0.41  fires=625  alpha=0.88 theta=0.047 ||M||=99.99
  val_loss=4.35 (window-local)
  promotion_ready=false (L0 gnorm above threshold)
```

---

## 6. Needle-in-a-Haystack (NIAH) Evaluation

### 6.1 Purpose

Test whether the CMS memory hierarchy can retrieve a specific fact planted early in a long
context window. At 60M parameters, passing NIAH at 4096 tokens would demonstrate that the
multi-frequency memory enables long-range retrieval that standard attention at this scale
cannot achieve.

### 6.2 Design

**Haystack construction**: A continuous passage of 4096 tokens from the validation corpus.
Natural text, not synthetic noise — the model must distinguish the needle from real distractors.

**Needle**: A synthetic fact with a unique answer not guessable from the haystack context.
Format: `"The secret code for project alpha is 7492."` — a declarative statement with a
specific retrievable value.

**Needle placement**: Variable depth within the haystack. Test at positions:
  - 256 tokens from query (local — L0/L1 should handle)
  - 1024 tokens from query (medium — L1/L2 range)
  - 2048 tokens from query (long — L2/L3 range)
  - 3584 tokens from query (maximum — only L3 can plausibly retrieve)

**Query**: After the full haystack, prompt the model: `"What is the secret code for project alpha?"`
Score by whether the model assigns higher probability to "7492" than to other 4-digit numbers.

### 6.3 Scoring

```text
FUNCTION: niah_score(model, haystack, needle, needle_pos, query) -> NiahResult

  -- Construct input: haystack with needle inserted at needle_pos, followed by query
  input_tokens = insert_needle(haystack, needle, needle_pos) + query_tokens

  -- Forward pass through full sequence (4096 + query tokens)
  logits = model.forward(input_tokens)

  -- Score: log-probability of the correct answer token(s) at the query position
  target_tokens = tokenize("7492")
  answer_logprob = sum(log_softmax(logits[query_pos + i])[target_tokens[i]]
                       for i in range(len(target_tokens)))

  -- Baseline: average log-probability of 10 random 4-digit numbers at the same position
  baseline_logprob = mean(score_alternative(model, logits, query_pos, random_4digit)
                          for _ in range(10))

  RETURN NiahResult {
    needle_depth: needle_pos / len(haystack),
    answer_logprob: answer_logprob,
    baseline_logprob: baseline_logprob,
    lift: answer_logprob - baseline_logprob,  -- positive = retrieval success
    pass: answer_logprob > baseline_logprob,
  }
```

**Pass criterion**: `lift > 0` (model assigns higher probability to the planted answer than
to random alternatives). A stronger criterion for publication: `lift > 1.0` (e^1 = 2.7x more
likely than random alternatives).

### 6.4 When to Run NIAH

- At every promotion boundary (before and after promotion)
- At `niah_every` steps during the final k=4 phase (default: every 10K steps)
- On the final checkpoint

This creates a trajectory showing how retrieval capability develops across phases. The
prediction: NIAH scores should improve monotonically with k (more levels = longer retrieval
horizon). If NIAH at depth 3584 passes at k=4 but fails at k=2, that's direct evidence the
level hierarchy is functional.

### 6.5 Per-Level Attribution

For each NIAH test, optionally run the ablated version (Section 2.3): forward pass with each
level's gate forced to zero. This reveals WHICH level is responsible for retrieval at each depth:

```text
NIAH depth=256:  L0 ablation kills retrieval → L0 is the retriever (expected)
NIAH depth=3584: L3 ablation kills retrieval → L3 is the retriever (the headline result)
NIAH depth=3584: L0 ablation has no effect   → L0 is not involved (expected)
```

This is the publishable finding: level i retrieves information at timescale chunk_size[i],
and ablating it destroys retrieval at that timescale while leaving other timescales intact.

---

## 7. Configuration Schema

```json
{
  "promotion": {
    "promotion_mode": "convergence",

    "gnorm_window_fires": 20,
    "gnorm_saturation_threshold": 0.05,
    "gnorm_hysteresis_windows": 2,

    "require_val_plateau": true,
    "val_patience_evals": 5,
    "val_min_delta": 0.01,
    "val_tokens": 50000,

    "min_fires_before_promotion": 15,
    "max_steps": 100000,

    "track_level_contribution": false,

    "niah_enabled": true,
    "niah_haystack_tokens": 4096,
    "niah_depths": [256, 1024, 2048, 3584],
    "niah_every": 10000,
    "niah_num_needles": 5
  }
}
```

All fields have defaults. Omitting the `promotion` section entirely falls back to
`promotion_mode: "steps"` (current behavior, fully backward compatible).

---

## 8. Implementation Plan

### Phase 1: Convergence Metrics + Promotion Decision (Python only)

**Files to modify**:

| File | Change |
|------|--------|
| `python/engine/config.py` | Add `PromotionConfig` dataclass with all fields from Section 7 |
| `python/engine/loop.py` | Add gnorm_l rolling window, plateau detection, promotion check |
| `python/engine/evaluation.py` | Add window-local val set loading, per-level eval logging |
| `python/engine/loop.py` | On promotion: save checkpoint, advance data cursor, extend_k |

No Rust or CUDA changes. The promotion logic is purely Python-tier orchestration (CS-18).

### Phase 2: NIAH Evaluation

**Files to create**:

| File | Purpose |
|------|---------|
| `python/engine/niah.py` | Needle construction, haystack assembly, scoring, level attribution |

**Files to modify**:

| File | Change |
|------|--------|
| `python/engine/evaluation.py` | Call niah.run_niah() at promotion boundaries and niah_every |
| `python/engine/loop.py` | Wire niah results into JSONL log |

### Phase 3: Automated Multi-Phase Pipeline

**Files to modify**:

| File | Change |
|------|--------|
| `python/engine/loop.py` | Promotion triggers automatic phase transition within a single run |

Currently each phase is a separate `hecate.py --build` invocation with a hand-written config.
Phase 3 enables a single config that specifies `target_k: 4` and the system runs k=1 through
k=4 automatically, promoting when convergence signals fire.

---

## 9. Relationship to Existing Specs

| Spec | Relationship |
|------|-------------|
| 07_push_up_level_stacking.md | Provides the promotion mechanism (extend_push_up) |
| 09_gate_warmup.md | Gate dormancy analysis; warmup may be needed at each promotion |
| 10_memory_manifold_analysis.md | JS divergence, effective rank — complementary diagnostics |
| 02_corpus_selection.md | Lag-MI validation ensures data has structure at CMS frequencies |
| task_39914e (spiral curriculum) | Alternative curriculum strategy; metric-driven promotion subsumes the "when to advance" question |

---

## 10. Falsification Criteria

This spec is falsified if:

1. **Convergence mode produces worse final eval loss than fixed-step mode** at matched total
   training tokens (same data budget, same compute). If so, the convergence signals are
   triggering premature promotion.

2. **NIAH at depth 3584 never passes at k=4** even after 100K steps. This means the memory
   hierarchy is not performing long-range retrieval at this model scale, regardless of
   curriculum.

3. **Per-level gnorm_l does not decline over a phase** — if gnorm_l is flat or increasing
   throughout training, saturation detection is not a valid promotion signal and an
   alternative criterion is needed.

4. **Window-local validation produces identical scores to fixed validation** — if there is no
   measurable difference between val-on-current-window and val-on-fixed-holdout, the data
   re-exposure problem is less severe than hypothesized and the complexity of per-window
   validation is not justified.

---

## 11. Code Smells

| Smell | Enforcement | Rationale |
|-------|-------------|-----------|
| CS-18 | architectural | Promotion logic lives in Python orchestration, not Rust |
| CS-10 | behavioral | No train/eval mode flag — NIAH uses the same forward pass as training |
| CS-11 | behavioral | No training loop in memory rules — promotion is external scheduling |
| CS-32 | behavioral | Observe-then-advance — metrics are read before promotion decision |
| CS-47 | behavioral | NIAH needle seeds and haystack selection are deterministic given config |
