# Corpus Selection for CMS Ablation Study

```
CONTRACT
  Purpose:    Select a build corpus that has genuine long-range token structure at
              CMS frequencies (lags 8, 64, 512 tokens) to distinguish the
              'data-limited' vs 'initialization trap' hypotheses from the k=1 vs k=4
              result. A corpus is valid iff empirical lag-MI at lag 512 exceeds 2×
              the background rate (lag 4096). Without this prerequisite, ABLATION
              runs A/B/C/D cannot be interpreted — dormant L2/L3 gates are ambiguous
              between "no signal in the data" and "model never learned to read the
              signal."

  Expects:    - Python environment with `datasets`, `tokenizers`, `numpy`, `tqdm`
              - HuggingFace Hub access or local copies of candidate corpora
              - LLaMA-3 tokenizer (32K vocab, BPE) — same tokenizer the build runs use
              - `python/data/` writable, ≥20GB free disk per corpus candidate
              - The lag-MI script `python/tools/lag_mi.py` (deliverable of this spec)
              - The corpus prep script `python/scripts/prepare_corpus.py` (deliverable of this spec)
              - HADES collection `nl_experiments` contains node `hecate-experimental-plan-2026-02`

  Guarantees: - Exactly one corpus is selected and documented in HADES before
                ABLATION-1 begins
              - The selection criterion is quantitative and falsifiable: an algorithm,
                not a judgment call
              - The lag-MI values for all evaluated corpora are stored in HADES so
                future corpus re-evaluations can cite the same baseline
              - The selected corpus is tokenized and sharded into `python/data/<name>/`
                at seq_len=512 before ABLATION-1 can be declared unblocked
              - If NO candidate passes the criterion, the spec mandates escalation
                (open GitHub issue, do not proceed with ABLATION runs)

  Cost:       - ~2–4 CPU-hours to download and tokenize 1B tokens per candidate
              - ~30 min to compute lag-MI on a 100M-token sample
              - No GPU required for this spec

  Trade-off:  - Evaluating lag-MI from raw token co-occurrence is an approximation to
                true mutual information (vocabulary size is too large for exact computation).
                The proxy is the expected PPMI at each lag, which is bounded below by
                true MI and zero when tokens are independent at that lag. The 2× threshold
                compensates for the approximation's conservative bias.
              - Restricting to 1B tokens per candidate sacrifices exactness for speed.
                The lag-MI signal is robust to sample size above ~100M tokens for a
                32K-vocabulary corpus.

  Position:   specs/infrastructure/02_corpus_selection.md
              Prerequisite for: specs/infrastructure/03_ablation_study.md (ABLATION-1)
              Related: docs/research_notes/nlm_initialization_dynamics.md

  Source:     HOPE (2512.24695) §3 CMS frequency levels [1, 8, 64, 512]
              Internal: k=1 vs k=4 result (issue #135), committee_response_06.md
```

---

## Why Corpus Selection is a Prerequisite

The k=1 vs k=4 comparison on FineWeb-Edu (72M tokens) showed dormant L2/L3 gates:
L2 θ ≈ 0.0019, L3 θ ≈ 0.0005. There are two structurally different explanations:

```
Hypothesis A (data-limited):
  FineWeb-Edu articles average 300–600 tokens. Cross-article structure
  does not exist at lag=512 because each article is independent. L2/L3
  gates are *correctly* dormant — there is nothing to memorize at those
  frequencies.

Hypothesis B (initialization trap):
  During the noisy early outer-loop phase (W_K, W_V near-random),
  L2/L3 gates collapsed toward zero. Once there, they receive minimal
  gradient signal and stay dormant even on corpora with 512-token structure.
  See: docs/research_notes/nlm_initialization_dynamics.md

These make distinct predictions:
  A → k=4 outperforms k=1 on long-range corpus, gates activate promptly
  B → k=4 still underperforms on long-range corpus unless warmup protocol used
```

A corpus that **fails** lag-MI validation makes the entire ABLATION study
uninterpretable — L2/L3 gate dormancy is not a finding, it is an artifact of
the corpus. No corpus = no ablation.

---

## Corpus Candidates

### Primary candidates (in evaluation order)

| Corpus | Source | Size | Rationale for long-range structure |
|--------|--------|------|-------------------------------------|
| **C4** | HuggingFace `allenai/c4` | ~180B tokens | Miras paper build corpus (§5). Web crawl, paragraph-structured, many docs > 512 tokens |
| **PG-19** | HuggingFace `pg19` | ~3B tokens | Project Gutenberg books. Strong inter-sentence and inter-paragraph dependencies at all lags |
| **SlimPajama-Books** | HuggingFace `cerebras/SlimPajama-627B`, books subset | ~25B tokens | Books subcorpus, similar to PG-19 but larger |

### Why FineWeb-Edu is excluded

FineWeb-Edu was selected for the original k=1/k=4 runs specifically because it is
well-filtered and high-quality. However, its articles are short-form educational
text (Wikipedia-style), typically 200–800 tokens per document. In a tokenized sequence
stream at seq_len=512, most sequences span at most 1–2 articles. Token correlations
at lag=512 are dominated by document boundaries, not coherent narrative structure.

This is not a quality judgment — FineWeb-Edu is excellent for many purposes. It
is specifically inadequate for testing whether CMS L2/L3 levels (periods 64 and 512)
carry useful signal.

---

## The Lag-MI Validation Protocol

### Definition: Empirical PPMI at Lag L

For a tokenized corpus stream X = [x_0, x_1, ...], define the **empirical
positive pointwise MI at lag L** as:

```
PPMI(L) = E_{t} [ max(0, log P̂(x_{t+L} | x_t) - log P̂(x_{t+L})) ]

Estimated as:
  1. Sample N=500K random indices t from the corpus
  2. For each pair (x_t, x_{t+L}), look up:
       p_joint(a, b)  = count(x_t=a, x_{t+L}=b) / N
       p_marginal(b)  = count(x_{t+L}=b) / N
  3. PPMI(L) = (1/N) * sum_t max(0, log p_joint(x_t, x_{t+L}) - log p_marginal(x_{t+L}))
```

PPMI(L) is zero when x_t and x_{t+L} are independent (uniform joint = product of
marginals). It is positive when there is genuine statistical dependency at lag L.

### Practical computation (vocabulary-efficient)

Full vocabulary PPMI over 32K tokens requires O(V^2) storage. Instead:

```
APPROACH: Vocabulary-bucketed PPMI
  1. Reduce to top-K tokens by frequency, bucket the rest as <UNK>
     K = 8192 (covers ~95% of token mass in BPE corpora)
  2. Build co-occurrence matrix C[K, K] from N=500K samples
  3. PPMI(L) = sum_{a,b} C[a,b]/N * max(0, log(C[a,b]*N / row_sum[a]*col_sum[b]))
  4. Normalize by vocabulary entropy H(X) to get a [0,1] bounded score:
       NMI(L) = PPMI(L) / H(X)
```

This is implemented in `python/tools/lag_mi.py` (deliverable).

### Evaluation lags

Evaluate at lags: `[1, 8, 64, 512, 4096]`

These correspond to:
- `1`: adjacent tokens — should be highest (bigram structure)
- `8`: L1 CMS period — medium-range phrase structure
- `64`: L2 CMS period — paragraph/section structure
- `512`: L3 CMS period — the critical test
- `4096`: Background / null — beyond most document boundaries, captures only
  corpus-wide topic bias (should be near zero for well-shuffled corpora)

### Pass criterion

```
PASS: NMI(lag=512) > 2.0 × NMI(lag=4096)

  Interpretation: lag-512 carries more than 2× the predictive signal of the
  background rate. This ensures L3 CMS has genuine information to memorize,
  not just topic-frequency artifacts.

FAIL: NMI(lag=512) ≤ 2.0 × NMI(lag=4096)

  Action: Corpus is excluded. Log the values in HADES and move to the next
  candidate. If all candidates fail, open a GitHub issue and halt ABLATION.
```

### Selection criterion (if multiple candidates pass)

Select the corpus with the highest `NMI(lag=512) / NMI(lag=4096)` ratio.
Among ties (ratio within 10%): prefer the larger corpus.

---

## Deliverables

### D1: `python/tools/lag_mi.py`

A standalone script. No build-time dependencies. Interface:

```bash
# Evaluate a corpus
python tools/lag_mi.py \
    --corpus allenai/c4 \
    --split train \
    --sample-tokens 100000000 \
    --lags 1 8 64 512 4096 \
    --vocab-k 8192 \
    --seed 42 \
    --out results/lag_mi_c4.json

# Output (example, not normative):
# {
#   "corpus": "allenai/c4",
#   "sample_tokens": 100000000,
#   "seed": 42,
#   "nmi": {"1": 0.041, "8": 0.018, "64": 0.009, "512": 0.006, "4096": 0.002},
#   "pass": true,
#   "ratio_512_4096": 3.0,
#   "threshold": 2.0
# }
# CS-47: seed is stored in output; same seed + same corpus = same result
```

The script streams from HuggingFace datasets (no full download required). It
tokenizes with LLaMA-3 tokenizer (`meta-llama/Meta-Llama-3-8B` tokenizer,
`vocab_size=32000`, same as the build configs).

### D2: Corpus evaluation results in HADES

For each evaluated corpus, insert a node into `nl_experiments`:

```json
{
  "_key": "corpus-eval-<name>-<date>",
  "type": "corpus_evaluation",
  "corpus": "<name>",
  "sample_tokens": 100000000,
  "nmi_by_lag": {"1": N, "8": N, "64": N, "512": N, "4096": N},
  "pass_criterion": "NMI(512) > 2x NMI(4096)",
  "passed": true|false,
  "ratio_512_4096": N,
  "date": "2026-XX-XX",
  "authority_basis": "human_measured"
}
```

### D3: Selected corpus tokenized and sharded

Output matches the format consumed by `BpeDataLoader` in `engine/data.py`:

```
python/data/<selected_corpus_name>/
  train_tokens.npy    # uint32 numpy array of token IDs (input)
  train_targets.npy   # uint32 numpy array of next-token targets (-1 → vocab_size)
  val_tokens.npy      # uint32 numpy array, validation split
  val_targets.npy     # uint32 numpy array, validation split
  meta.json           # {"vocab_size": 32000, "train_tokens": N, "val_tokens": M,
                      #  "tokenizer": "...", "corpus": "...", "seed": 42}
```

Produced by `python/scripts/prepare_corpus.py` (also a deliverable of this spec).
Total: ≥1B tokens for build, ≥1M tokens for evaluation.

### D4: Update HADES `hecate-experimental-plan-2026-02`

Merge the corpus selection outcome into the existing node:

```json
{
  "corpus_selection": {
    "selected": "<name>",
    "evaluated": ["c4", "pg19", ...],
    "nmi_results": {...},
    "selection_date": "2026-XX-XX",
    "shard_path": "python/data/<name>/"
  }
}
```

---

## Failure Protocol

If no candidate corpus passes the criterion:

1. Open a new GitHub issue: "Corpus search — lag-MI failure on all candidates"
2. Post the full `lag_mi.py` output for each evaluated corpus as a comment
3. Halt all ABLATION tasks (mark as blocked, reason: "no valid corpus found")
4. Consider: Wikipedia (English), OpenWebText, Pile-Books3, or custom curriculum

Do **not** relax the 2× threshold to force a corpus to pass. The threshold is
calibrated to ensure L3 has real signal. Relaxing it defeats the purpose of the
experiment.

---

## Acceptance Criteria for task_210707

- [ ] `python/tools/lag_mi.py` written and tested on a small sample
- [ ] At least 2 corpus candidates evaluated (≥100M tokens each)
- [ ] HADES: corpus evaluation nodes inserted for each candidate
- [ ] At least 1 corpus passes the criterion, OR failure protocol initiated
- [ ] Winning corpus tokenized and sharded in `python/data/<name>/`
- [ ] HADES: `hecate-experimental-plan-2026-02` updated with corpus selection
- [ ] ABLATION-1 task (task_5335bd) unblocked

---

## Implementation Notes

1. **Streaming, not downloading**: Use HuggingFace `datasets` with `streaming=True`.
   For 100M tokens at ~4 bytes/token this is 400MB of data transfer, not 180GB.

2. **Tokenizer must match**: The lag-MI is computed over the *same token IDs* that
   the build run will see. Use `meta-llama/Meta-Llama-3-8B` tokenizer with
   `add_bos_token=False, add_eos_token=False` to match `python/engine/data.py`.

3. **Do not shuffle before lag-MI**: Shuffling a corpus before computing lag-MI
   destroys exactly the signal being measured. Compute lag-MI on natural document
   order, then shuffle the *shards* for build.

4. **Shard format**: The existing `python/engine/data.py` ContextStream reads raw
   `int32` token arrays in contiguous files. Match this format exactly.

5. **The k=1/k=4 FineWeb-Edu runs already exist**: Resist the temptation to resume
   `fineweb_k1` at step 55K on a new corpus as a "quick check." That run is not
   a controlled comparison — its projections are trained on FineWeb-Edu statistics.
   All four ablation runs (A/B/C/D) must be fresh cold-start runs on the same corpus.
