# Phase 0 Experiment Plan: Infrastructure Validation

## Philosophy

Before we can break this model, we need to confirm it works. Before we can
confirm it works, we need to confirm the *instruments* work. The committee
is asking for adversarial robustness (Step 4). We are finishing infrastructure
correctness (Step 2). The ordering matters:

```
Step 1: Code compiles and runs             ← DONE (805 tests, 0 failures)
Step 2: Infrastructure is correct          ← HERE (Experiments 0A/0B running)
Step 3: Model learns as NL predicts        ← NEXT
Step 4: Model is robust under adversarial  ← Committee's ask (premature)
```

Each step gates the next. If Step 2 shows the gates aren't moving, Step 3's
"adaptation rate" metric is meaningless. If Step 3 shows the model doesn't
learn differently with k=4, Step 4's adversarial probes have nothing to stress.

---

## Currently Running: Experiments 0A and 0B

Two matched runs on FineWeb-Edu (score 4+5, 76M tokens, 32K BPE vocab):

| | Experiment 0A | Experiment 0B |
|---|---|---|
| **GPU** | GPU 0 | GPU 1 |
| **k** | 1 (single level) | 4 (CMS hierarchy) |
| **Params** | ~36.7M | ~45.4M |
| **Data** | FineWeb-Edu (identical) | FineWeb-Edu (identical) |
| **Config** | `configs/fineweb_edu_k1.json` | `configs/fineweb_edu_k4.json` |
| **Log** | `runs/fineweb_edu_k1.jsonl` | `runs/fineweb_edu_k4.jsonl` |
| **Purpose** | Baseline (no frequency hierarchy) | Does CMS produce different dynamics? |

**Controlled variables**: Same data, tokenizer, optimizer, LR schedule, warmup,
d_model, num_heads, seq_len, memory_rule (Titans), composition (MAG),
projection_kind (adaptive), self_generated_values (true).

**Isolated variable**: k (1 vs 4). This is the cleanest possible comparison —
we are testing whether frequency hierarchy produces measurably different
behavior, not whether "NL beats transformers."

---

## Experiment 1: Instrument Validation ("Do the gauges work?")

**Goal**: Confirm that what we log faithfully represents what the model is doing.
This is about the *instruments*, not the model.

**When**: After 0A/0B reach 10K steps (both in warmup completion range).

### 1.1 Gate Bias Sanity Check

**What we have**: `gate_biases()` returns raw (b_alpha, b_theta, b_eta) per level.

**What to verify**:
- [ ] **Initial values match spec**: b_alpha should start at [3.0, 4.0, 4.5, 5.0]
      for k=4 (sigmoid ≈ 0.95-0.99). b_theta should start at [-4.6, -5.6, -6.6, -7.6]
      (softplus ≈ 0.01-0.0005). Verify by parsing step 0 from JSONL.
- [ ] **Biases change over training**: Plot b_alpha and b_theta per level vs step.
      If biases are FLAT, the outer loop isn't reaching them (wiring bug).
      If they diverge wildly in the first 100 steps, initialization is wrong.
- [ ] **k=1 has exactly one level**: Verify JSONL for 0A has 1-tuple gate biases.

**Tool**: Parse JSONL, plot with matplotlib. No new code needed.

### 1.2 Memory Norm Trajectory

**What we have**: `memory_norms()` returns Frobenius norm of M per level at eval steps.

**What to verify**:
- [ ] **M grows from zero**: Memory starts empty (reset_context). Norms should
      increase from 0 over the first few thousand steps as M accumulates.
- [ ] **M stabilizes**: Norms shouldn't grow unboundedly — retention decay should
      produce an equilibrium. If ||M|| doubles every 1K steps, retention is broken.
- [ ] **k=4 levels differ**: Higher levels (L2, L3) should show different norm
      trajectories than L0. If all 4 norms are identical, the frequency hierarchy
      isn't doing anything.

**Tool**: Parse JSONL eval events, plot memory_norms per level. No new code needed.

### 1.3 Fire Count Verification

**What we have**: `level_fire_counts` logged per eval block, reset after logging.

**What to verify**:
- [ ] **Frequencies match config**: With chunk_sizes=[1,8,64,512], between two eval
      points (5000 steps apart), expected fires: L0=5000, L1=625, L2≈78, L3≈10.
      Parse JSONL and verify counts match.
- [ ] **k=1 fires every step**: All 5000 fires should be L0.

**Tool**: Parse JSONL. No new code needed.

### 1.4 Level 3 Active Fire Threshold

**What we have**: Level 3 "active fire" counter (b_theta > 0.001 threshold).

**What to verify**:
- [ ] **Early training**: L3 active fires should be 0 or near-0. b_theta_l3 starts
      at -7.6 (softplus ≈ 0.0005), well below threshold.
- [ ] **If active fires appear**: This means L3's learning rate gate is opening —
      the model is deciding Level 3 should learn. This is a KEY signal.

**Tool**: Parse JSONL `event="level3_activity"`. No new code needed.

---

## Experiment 2: Convergence Comparison ("Does k matter?")

**Goal**: Determine whether k=4 produces meaningfully different learning dynamics
than k=1, or whether the extra 24% parameters are wasted.

**When**: After 0A/0B reach 25K steps.

### 2.1 Loss Curves

**What to measure**:
- [ ] **Convergence speed**: Steps to reach loss=6.0 (or whatever the plateau is).
      k=4 has more parameters, so faster convergence alone proves nothing —
      normalize by parameter count.
- [ ] **Final loss at 50K steps**: If k=4 is >5% lower loss than k=1 at 50K,
      frequency hierarchy is contributing beyond parameter count.
- [ ] **Loss variance**: Compute rolling stddev of loss over 100-step windows.
      If k=4 is smoother, the frequency hierarchy is providing gradient stability.

### 2.2 Gate Evolution Divergence

**What to measure**:
- [ ] **Do k=4 levels specialize?** Plot b_alpha and b_theta per level over time.
      If all 4 levels converge to the same bias values, CMS isn't creating
      frequency specialization. If they diverge, the model is learning different
      forget/learn rates per timescale.
- [ ] **Compare k=1 gate trajectory to k=4's L0**: k=1's single level and k=4's
      Level 0 both fire every step. Do they evolve the same gate values?
      If they differ, the presence of higher levels is influencing L0.

### 2.3 Memory Norm Stratification

**What to measure**:
- [ ] **Norm ordering**: In a working CMS, we'd expect L0 memory to be "fast
      and volatile" (moderate norm, high variance) and L3 to be "slow and stable"
      (lower norm, low variance). Plot norm trajectories per level.
- [ ] **Norm growth rate**: Compute d(||M||)/d(step) per level. Higher levels
      should show slower growth (they fire less often).

---

## Experiment 3: Context Accumulation ("Does memory do anything?")

**Goal**: Verify that memory accumulation across chunks produces measurably
different behavior than a memoryless model.

**When**: After 0A/0B reach 25K steps.

### 3.1 Fresh Context vs Accumulated Context

**Method**:
1. Take a checkpoint at step 25K
2. Run eval on 20 val chunks with accumulated context (normal)
3. Run eval on same 20 val chunks with reset_context() before each chunk
4. Compare loss

**Expected**: Accumulated context should produce lower loss on later chunks
(memory helps predict what comes next based on what came before).
If there's no difference, memory isn't contributing.

**What to add**: A small evaluation script (~50 lines) that loads a checkpoint
and runs both conditions. Not a loop.py change — a standalone diagnostic.

### 3.2 Memory Norm Before/After Document Boundaries

**Method**:
1. Log memory norms before and after reset_context() at document boundaries
2. If the norm drops significantly at boundaries and rebuilds within the
   next document, memory is accumulating document-level information

**What to add**: A few log lines in the document boundary handling code.

---

## Experiment 4: Domain Transition ("Does k=4 adapt faster?")

**Goal**: The core committee prediction — does frequency hierarchy help with
domain adaptation?

**When**: After 0A/0B reach 50K steps on FineWeb-Edu.

### 4.1 Setup

1. Take checkpoints from both k=1 and k=4 at step 50K
2. Prepare a small ARC-like reasoning dataset (or use a different text domain)
3. Continue training both from their 50K checkpoints on the new domain

### 4.2 Metrics

- [ ] **Steps to stabilization**: How many steps until loss on new domain
      plateaus? k=4 should stabilize faster if frequency hierarchy enables
      faster adaptation (L0 adapts immediately, L3 retains old knowledge).
- [ ] **Catastrophic forgetting**: After N steps on new domain, evaluate on
      FineWeb-Edu. k=4 should show LESS forgetting (L3 retains base knowledge).
- [ ] **Gate response to transition**: Do gate biases shift at the domain
      boundary? L0's theta should increase (learn faster on new material).
      L3's theta should stay low (preserve old knowledge).

### 4.3 Falsifiable Predictions (from committee_response_04)

| Prediction | Threshold | Measured By |
|---|---|---|
| L2 theta > 0.003 after 20K steps | softplus(b_theta_l2) | JSONL gate_biases |
| k=4 loss 5%+ below k=1 at 50K | (loss_k1 - loss_k4) / loss_k1 | JSONL eval_loss |
| k=4 stabilizes on ARC in 2x fewer steps | steps_to_plateau | Domain transition run |
| k=1 shows greater FineWeb regression after ARC | delta_eval_loss | Re-eval on FineWeb |

---

## Experiment 5: Adversarial Probes ("Try to break it")

**Goal**: The committee's ask. But only after Steps 2-3 confirm the model works.

**When**: After Experiment 4 completes.

**Probes** (from committee suggestions):
- Repetitive input designed to saturate memory
- Contradictory information across chunks
- Distribution shift within a single document
- Noise injection to test retention resilience

**Why it's last**: If Experiment 2 shows k=4 gates aren't moving, adversarial
probes will just confirm a broken model is broken. Fix the model first.

---

## Infrastructure Gaps to Fill

Based on diagnostic inventory, these are needed for the experiments above:

| Gap | Needed For | Effort | Priority |
|---|---|---|---|
| Eval script for Exp 3.1 (fresh vs accumulated) | Experiment 3 | ~50 lines Python | Medium |
| JSONL analysis/plotting script | Experiments 1-2 | ~200 lines Python | High |
| Domain transition data prep | Experiment 4 | ~100 lines Python | Later |
| Per-level gradient norms | Nice-to-have for Exp 2 | Rust + PyO3 change | Low |

The JSONL plotting script is the highest priority — we need it to analyze
Experiments 0A/0B once they reach 10K steps.

---

## Timeline

```
Now         → 0A/0B running (step 0, both GPUs)
~2 hours    → 10K steps → Experiment 1 (instrument validation)
~5 hours    → 25K steps → Experiments 2+3 (convergence + context)
~10 hours   → 50K steps → Begin Experiment 4 setup (domain transition)
~15 hours   → Experiment 4 data collected
~20 hours   → Experiment 5 (adversarial) IF steps 2-3 pass

Each step gates the next. If Experiment 1 shows broken instruments,
we fix them before proceeding. No skipping.
```

---

## Decision Points

**After Experiment 1**: If gate biases are flat → investigate outer-loop gradient
flow. If memory norms are zero → check M initialization and copy_final_m path.
If fire counts don't match config → Conductor bug.

**After Experiment 2**: If k=4 shows no advantage over k=1 (normalized for params)
→ CMS frequency hierarchy may not be helping at this scale. This is a legitimate
finding, not a failure. Report it honestly.

**After Experiment 3**: If fresh context = accumulated context → memory isn't
contributing. This would be a significant finding requiring investigation.

**After Experiment 4**: If k=1 adapts as fast as k=4 → frequency hierarchy
doesn't help with domain transition. Again, report honestly.

The experiments are designed to produce useful information regardless of outcome.
A null result on Experiment 2 ("k doesn't matter at 60M params") is just as
publishable as a positive result.
