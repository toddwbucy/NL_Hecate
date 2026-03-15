# NIAH Verification Tool — Standalone CMS Retrieval Probe

```
CONTRACT
  Purpose:    Standalone verification tool that tests CMS memory retrieval by
              planting synthetic factoids at random positions within a real-corpus
              haystack and querying them at configurable distances. Runs as an
              independent CLI against saved checkpoints — not coupled to the build
              loop. Designed to be run between push-up phases to confirm that the
              CMS frequency hierarchy enables long-range retrieval.

  Expects:    - Saved checkpoint (safetensors) with MAGConfig
              - BPE corpus data (same format as training data)
              - GPU with nl_hecate CUDA bindings
              - Tokenizer matching the checkpoint's vocabulary

  Guarantees: - Needle position is randomized per trial (never position 0)
              - Retrieval distance (tokens between needle and query) is the
                test variable, configurable and sweepable
              - Post-query corpus padding prevents positional shortcuts
              - Each trial uses a fresh CMS context (clean memory state)
              - Results include per-trial needle_depth, retrieval_distance,
                answer_logprob, baseline_logprob, lift, and pass/fail
              - JSON output suitable for automated analysis and plotting

  Cost:       - Per trial: ~(haystack_size / seq_len) forward passes through the
                loaded model. At haystack=8192, seq_len=512: ~16 forward calls.
              - 10 trials × 4 distances = 40 trial runs = ~640 forward calls total
              - At d=512, k=4: ~2 minutes on A6000 for a full sweep
              - Zero effect on training state (standalone process)

  Trade-off:  Logprob scoring measures whether the model's distribution FAVORS the
              correct answer, not whether it can generate it verbatim. At 60M params,
              generation quality is too low for text-matching evaluation. Logprob lift
              is the right metric: it detects retrieval signal even when the model
              can't produce fluent text.

  Position:   specs/infrastructure/26_niah_verification.md

  Source:     Spec 12 §6 (metric-driven promotion — NIAH design)
              Titans (2501.00663) — CMS frequency hierarchy
              HOPE (2512.24695) eq-097 — CMS chained frequency rule
              CS-10 — No train/eval distinction (same forward pass)
              CS-18 — Orchestration in Python, math in Rust
              CS-47 — Deterministic needle seeds given config
```

---

## 1. Why Standalone?

NIAH was originally embedded in the build loop (PR #196 first attempt). This was
wrong for three reasons:

1. **Contamination risk**: Running NIAH mid-training modifies CMS memory state.
   Even with save/restore, the context switch disrupts the memory's continuous
   learning trajectory.

2. **Scheduling coupling**: NIAH's schedule (every N steps, at promotion
   boundaries) is orthogonal to the training loop's step counter. Embedding it
   creates conditional complexity that doesn't belong in the forward loop.

3. **Reproducibility**: A standalone tool with a fixed checkpoint produces
   identical results regardless of when it's run. Build-loop NIAH depends on
   the exact training state at the moment of evaluation.

The correct workflow is:

```text
Phase 1 (k=1): train → save checkpoint
                          ↓
                    run niah_verify (baseline: should fail at 4096)
                          ↓
Phase 2 (k=2): train → save checkpoint
                          ↓
                    run niah_verify (should improve at 1024-2048)
                          ↓
Phase 3 (k=3): train → save checkpoint
                          ↓
                    run niah_verify (should improve at 2048-4096)
                          ↓
Phase 4 (k=4): train → save checkpoint
                          ↓
                    run niah_verify (headline: 4096 should pass)
```

---

## 2. Haystack Construction

Each trial constructs a sequence:

```text
[corpus_prefix] + [needle] + [corpus_gap] + [query] + [corpus_suffix]
 ←── random ──→              ←── distance ──→         ←── padding ──→
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `haystack_size` | 8192 | Total tokens in the haystack (prefix + gap + suffix) |
| `retrieval_distance` | 4096 | Tokens between needle injection and query |
| `min_prefix` | 512 | Minimum corpus tokens before needle |
| `min_suffix` | 256 | Minimum corpus tokens after query |

### Construction algorithm

```text
FUNCTION: build_trial_sequence(corpus, needle, query, rng, config)

  prefix_budget = haystack_size - retrieval_distance - len(needle) - len(query) - min_suffix
  IF prefix_budget < min_prefix:
      ERROR "haystack_size too small for retrieval_distance"

  -- Random needle depth: between min_prefix and prefix_budget
  needle_depth = rng.randint(min_prefix, prefix_budget)

  -- Carve non-overlapping corpus segments
  prefix  = corpus[offset : offset + needle_depth]
  gap     = corpus[offset + needle_depth : offset + needle_depth + retrieval_distance]
  suffix  = corpus[offset + needle_depth + retrieval_distance : offset + needle_depth + retrieval_distance + min_suffix]

  RETURN prefix + needle + gap + query + suffix
```

**Needle depth is randomized** so the model can't learn a positional shortcut.
The query is buried in corpus (not terminal) so the last-position bias doesn't
apply.

---

## 3. Needles

Synthetic factoids with unique, unguessable answers:

```python
NEEDLES = [
    ("The secret code for project alpha is 7492.",
     "What is the secret code for project alpha?", "7492"),
    ("The identification number for laboratory zeta is 3841.",
     "What is the identification number for laboratory zeta?", "3841"),
    ("The access key for server omega is 6205.",
     "What is the access key for server omega?", "6205"),
    ("The reference number for experiment delta is 1738.",
     "What is the reference number for experiment delta?", "1738"),
    ("The serial number for device sigma is 9054.",
     "What is the serial number for device sigma?", "9054"),
]
```

Each answer is a 4-digit number that will not appear in natural corpus text.
The tokenizer may encode these as 1-4 BPE tokens — scoring handles this
(Section 4).

---

## 4. Scoring

### 4.1 Forward pass

The full trial sequence is processed in `seq_len` chunks through the model's
`forward()` method. CMS memory accumulates naturally across chunks — this is
the same path as training (CS-10).

A fresh `Conductor` is created per trial. CMS context is reset before each
trial. No training state needs protection because this is a standalone process.

### 4.2 Answer logprob

After processing the full sequence, extract logits at the position immediately
before the first answer token. Compute log-softmax and score:

```text
answer_logprob = log_softmax(logits[query_answer_pos])[answer_token_ids[0]]
```

We score only the first answer token because we have logits at only one
position. The comparison is fair: baseline alternatives are also scored at
the same position using their first token.

### 4.3 Baseline comparison

Score 10 random 4-digit numbers at the same position:

```text
FOR i in 1..10:
    random_num = rng.randint(1000, 9999)
    random_tokens = tokenizer.encode(str(random_num))
    baseline_logprobs[i] = log_softmax(logits[query_answer_pos])[random_tokens[0]]

baseline_logprob = mean(baseline_logprobs)
```

### 4.4 Lift and pass/fail

```text
lift = answer_logprob - baseline_logprob
pass = lift > 0
```

- `lift > 0`: Model favors the planted answer over random alternatives
- `lift > 1.0`: Publication-grade (e^1 ≈ 2.7× more likely than random)

---

## 5. CLI Interface

```bash
python -m tools.niah_verify \
    --checkpoint runs/k2_phase1/checkpoints/model.safetensors \
    --data data/fineweb_edu \
    --distances 1024,2048,4096 \
    --num_trials 5 \
    --seed 42 \
    --output results/niah_k2.json \
    --gpu 0
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | yes | — | Path to safetensors checkpoint |
| `--data` | yes | — | Path to BPE corpus directory |
| `--distances` | no | `4096` | Comma-separated retrieval distances |
| `--num_trials` | no | `5` | Trials per distance per needle |
| `--haystack_size` | no | `8192` | Total haystack tokens |
| `--min_prefix` | no | `512` | Min corpus tokens before needle |
| `--min_suffix` | no | `256` | Min corpus tokens after query |
| `--seed` | no | `42` | RNG seed for reproducibility |
| `--output` | no | stdout | JSON output file path |
| `--gpu` | no | `0` | CUDA device index |

---

## 6. Output Format

### Console output

```text
NIAH Verification: runs/k2_phase1/checkpoints/model.safetensors
  Model: d=512, k=2, chunks=[1,8]
  Data:  data/fineweb_edu (12.4M tokens)
  Distances: [1024, 2048, 4096]
  Trials: 5 per distance

  distance=1024  pass=4/5  mean_lift=0.847
    trial 0: PASS  depth=1203  lift=1.24  answer=7492
    trial 1: PASS  depth=3891  lift=0.67  answer=3841
    trial 2: FAIL  depth=724   lift=-0.12 answer=6205
    trial 3: PASS  depth=2150  lift=0.93  answer=1738
    trial 4: PASS  depth=4401  lift=1.51  answer=9054

  distance=2048  pass=2/5  mean_lift=0.213
    ...

  distance=4096  pass=0/5  mean_lift=-0.341
    ...

  Summary:
    1024: 80% pass, mean_lift=0.847
    2048: 40% pass, mean_lift=0.213
    4096:  0% pass, mean_lift=-0.341
```

### JSON output

```json
{
  "checkpoint": "runs/k2_phase1/checkpoints/model.safetensors",
  "model": {"d_model": 512, "k": 2, "chunk_sizes": [1, 8]},
  "config": {
    "haystack_size": 8192,
    "min_prefix": 512,
    "min_suffix": 256,
    "seed": 42
  },
  "distances": {
    "1024": {
      "pass_rate": 0.8,
      "mean_lift": 0.847,
      "trials": [
        {
          "needle_idx": 0,
          "needle_depth": 1203,
          "retrieval_distance": 1024,
          "answer": "7492",
          "answer_logprob": -4.21,
          "baseline_logprob": -5.06,
          "lift": 0.847,
          "pass": true
        }
      ]
    },
    "2048": { "..." : "..." },
    "4096": { "..." : "..." }
  }
}
```

---

## 7. CMS Frequency Alignment

The retrieval distances map to CMS level boundaries:

| Distance | CMS levels in range | Expected retriever |
|----------|--------------------|--------------------|
| 256 | L0 (chunk=1) | L0 — fires every token |
| 1024 | L0, L1 (chunk=8) | L1 — 128 fires in 1024 tokens |
| 2048 | L0, L1, L2 (chunk=64) | L2 — 32 fires in 2048 tokens |
| 4096 | L0, L1, L2, L3 (chunk=512) | L3 — 8 fires in 4096 tokens |

The prediction: retrieval at distance D requires a CMS level whose chunk_size
allows enough fires across D tokens to encode and retain the needle. If NIAH
passes at 4096 only after k=4 is available, the level hierarchy is working.

---

## 8. Implementation Plan

### Files to create

| File | Purpose |
|------|---------|
| `python/tools/niah_verify.py` | Standalone offline CLI tool + core logic |

### Files unchanged

- No modifications to `loop.py`, `config.py`, or any Rust code
- Uses existing `nl_hecate.load_checkpoint`, `GpuModel.forward`, `Conductor`

### Implementation sequence

1. Argument parsing and checkpoint loading
2. Corpus loading (reuse `BpeTokenStream`)
3. Haystack construction with randomized needle depth
4. Forward pass loop (seq_len chunks, fresh Conductor per trial)
5. Logprob scoring at query answer position
6. Results aggregation and output

---

## 9. Falsification Criteria

1. **NIAH at 4096 never passes at k=4 after 100K steps**: CMS memory hierarchy
   does not enable long-range retrieval at this model scale.

2. **NIAH results are identical across k values**: The level hierarchy adds no
   retrieval capability — the model retrieves (or fails) regardless of CMS depth.

3. **NIAH passes at 4096 at k=1**: The sliding window attention alone handles
   4K retrieval, making CMS memory unnecessary for this task.

---

## 10. Code Smells

| Smell | Enforcement | Rationale |
|-------|-------------|-----------|
| CS-10 | behavioral | Same forward pass as training — no eval mode |
| CS-18 | architectural | Orchestration in Python, forward pass in Rust |
| CS-32 | behavioral | Observe-then-advance — score after full forward |
| CS-47 | behavioral | Deterministic given seed + checkpoint + corpus |
