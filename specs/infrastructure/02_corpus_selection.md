# Corpus Selection for CMS Ablation Study

```text
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
                HADES: hope_equations/eq-097-hope-cms-chain (CMS chained frequency rule)
                       hope_equations/eq-093-freq-transfer (frequency transfer between levels)
              Internal: k=1 vs k=4 result (issue #135), committee_response_06.md
```

---

## Why Corpus Selection is a Prerequisite

The k=1 vs k=4 comparison on FineWeb-Edu (72M tokens) showed dormant L2/L3 gates:
L2 θ ≈ 0.0019, L3 θ ≈ 0.0005. There are two structurally different explanations:

```text
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

### Definition: Excess Same-Token Rate (ESTR) at Lag L

> **Paper trace**:
> - **Lag values [1, 8, 64, 512]**: CMS frequency periods from HOPE (2512.24695) §3,
>   HADES: `hope_equations/eq-097-hope-cms-chain`, `hope_equations/eq-093-freq-transfer`
> - **Background lag 4096**: Internal diagnostic baseline — no paper source. Chosen as
>   a lag beyond most document boundaries so topic-frequency bias is the only remaining
>   signal.
> - **ESTR formula** (`same_rate / expected_collision - 1`): Custom proxy metric. The
>   expected collision rate `sum_v P(v)^2` is the standard birthday-problem self-overlap
>   (Cover & Thomas, *Elements of Information Theory*, §2.4). No HADES record — ESTR
>   is not from the NL paper set.
> - **Pass-criterion ratio 2.0×**: Empirically calibrated threshold, not from any paper.
>   Chosen so C4 (ratio=7.86×) clearly passes and FineWeb-Edu (ratio=1.00×) clearly
>   fails. No HADES record.

For a tokenized corpus stream X = [x_0, x_1, ...], define the **Excess
Same-Token Rate at lag L** as:

```rust
// ESTR(L): Excess Same-Token Rate at lag L
// tokens: &[u32]  — flat uint32 token stream
// lag: usize      — distance between compared positions
// exclude_top_n: usize — number of high-frequency tokens to ignore
fn estr(tokens: &[u32], lag: usize, exclude_top_n: usize, n_samples: usize) -> f64 {
    let freq = frequency_histogram(tokens);           // freq[v] = count of token v
    let stop_ids: HashSet<u32> = top_n_ids(&freq, exclude_top_n);

    // Sample (t, t+lag) pairs restricted to content-word positions
    let (same, total) = sample_pairs(tokens, lag, n_samples, &stop_ids);
    let same_rate: f64 = same as f64 / total as f64;

    // Birthday-problem baseline: expected collision rate under independence
    let expected_collision: f64 = content_vocab_probs(&freq, &stop_ids)
        .iter().map(|p| p * p).sum();

    same_rate / expected_collision - 1.0  // ESTR > 0 ↔ structured; ≈ 0 ↔ flat
}
```

ESTR(L) ≈ 0 when tokens repeat at chance frequency (independent at lag L).
ESTR(L) > 0 when the same content word recurs more than chance — the corpus
has genuine predictive structure at that lag distance.

**Why ESTR instead of PPMI**: PPMI over the full 32K BPE vocabulary is
dominated by high-frequency subwords ("the", "of", " a") which are flat at
all lags. ESTR with stop-word exclusion isolates content-word repetition
(proper nouns, technical terms) that reflects true long-range structure.

### Practical computation

```rust
// Stop-word-excluded same-token rate (practical ESTR estimation)
// Implemented in python/tools/lag_mi.py::_compute_estr
fn compute_estr_practical<R: Rng + ?Sized>(
    tokens: &[u32],
    lag: usize,
    exclude_top_n: usize,  // default: 200
    n_samples: usize,       // default: 500_000
    rng: &mut R,
) -> f64 {
    let freq = np_bincount(tokens);
    let stop_ids: HashSet<u32> = argsort_desc(&freq)[..exclude_top_n].collect();

    let positions = rng.integers(0, tokens.len() - lag, n_samples);
    let pairs: Vec<(u32, u32)> = positions
        .filter(|&t| !stop_ids.contains(&tokens[t]) && !stop_ids.contains(&tokens[t + lag]))
        .map(|t| (tokens[t], tokens[t + lag]))
        .collect();

    let same_rate = pairs.iter().filter(|(a, b)| a == b).count() as f64 / pairs.len() as f64;
    let p_content = content_unigram_probs(&freq, &stop_ids);
    let expected_collision: f64 = p_content.iter().map(|p| p * p).sum();

    same_rate / expected_collision - 1.0
}
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

```rust
// Corpus selection criterion (implemented in lag_mi.py::evaluate_criterion)
fn passes_selection(estr: &HashMap<usize, f64>, threshold: f64) -> bool {
    let estr_512 = estr[&512];           // L3 CMS period — the signal lag
    let estr_bg  = estr[&4096];          // background / null reference

    // Non-positive background is an immediate FAIL: estr_bg ≤ 0 means the
    // metric produced no meaningful background signal (broken computation or
    // degenerate corpus). Clamping to 1e-12 would produce a huge ratio and
    // incorrectly PASS a corpus with zero/negative background.
    if estr_bg <= 0.0 {
        return false;
    }
    let bg_safe = estr_bg.max(1e-12);    // defensive floor (unreachable after guard)
    let ratio = estr_512 / bg_safe;

    // PASS: signal exists AND exceeds threshold × background
    estr_512 > 0.0 && ratio > threshold  // threshold = 2.0 (PASS_THRESHOLD)
    // FAIL: estr_512 ≤ 0 OR ratio ≤ threshold
    //   → corpus excluded; log to HADES, try next candidate
}
```

### Selection criterion (if multiple candidates pass)

Select the corpus with the highest `ESTR(lag=512) / ESTR(lag=4096)` ratio.
Among ties (ratio within 10%): prefer the larger corpus.

---

## Deliverables

### D1: `python/tools/lag_mi.py`

A standalone script. No build-time dependencies. Interface:

```bash
# Evaluate a corpus
python python/tools/lag_mi.py \
    --corpus allenai/c4 \
    --config en \
    --split train \
    --sample-tokens 100000000 \
    --lags 1 8 64 512 4096 \
    --exclude-top-n 200 \
    --seed 42 \
    --out results/lag_mi_c4.json

# Output (example, not normative):
# {
#   "corpus": "allenai/c4",
#   "sample_tokens": 100000000,
#   "exclude_top_n": 200,
#   "seed": 42,
#   "estr": {"1": 0.041, "8": 0.018, "64": 0.009, "512": 0.006, "4096": 0.002},
#   "criterion": {"passed": true, "ratio_512_4096": 3.0, "threshold": 2.0}
# }
# CS-47: seed is stored in output; same seed + same corpus = same result
```

The script streams from HuggingFace datasets (no full download required). It
tokenizes with LLaMA-3 tokenizer (`hf-internal-testing/llama-tokenizer` tokenizer,
`vocab_size=32000`, same as the build configs).

### D2: Corpus evaluation results in HADES

For each evaluated corpus, insert a node into `nl_experiments`:

```json
{
  "_key": "corpus-eval-<name>-<date>",
  "type": "corpus_evaluation",
  "corpus": "<name>",
  "sample_tokens": 100000000,
  "estr_by_lag": {"1": N, "8": N, "64": N, "512": N, "4096": N},
  "pass_criterion": "ESTR(512) > 2x ESTR(4096)",
  "passed": true|false,
  "ratio_512_4096": N,
  "date": "2026-XX-XX",
  "authority_basis": "human_measured"
}
```

### D3: Selected corpus tokenized and sharded

Output matches the format consumed by `BpeTokenStream` in `engine/data.py`:

```text
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
    "estr_results": {...},
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
   the build run will see. Use `hf-internal-testing/llama-tokenizer` tokenizer with
   `add_bos_token=False, add_eos_token=False` to match `python/engine/data.py`.

3. **Do not shuffle before lag-MI**: Shuffling a corpus before computing lag-MI
   destroys exactly the signal being measured. Compute lag-MI on natural document
   order, then shuffle the *shards* for build.

4. **Shard format**: The existing `python/engine/data.py` ContextStream reads raw
   `uint32` token arrays in contiguous files. Match this format exactly.

5. **The k=1/k=4 FineWeb-Edu runs already exist**: Resist the temptation to resume
   `fineweb_k1` at step 55K on a new corpus as a "quick check." That run is not
   a controlled comparison — its projections are trained on FineWeb-Edu statistics.
   All four ablation runs (A/B/C/D) must be fresh cold-start runs on the same corpus.
