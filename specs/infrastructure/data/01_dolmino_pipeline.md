# Dolmino-Mix 100B Data Pipeline

```text
CONTRACT
  Purpose:    Prepare Dolmino-Mix 100B (OLMo 3 32B annealing dataset) as a
              flat numpy token stream compatible with BpeDataLoader for NL-Hecate
              build runs. Produces train/val splits in the same uint32/int32 format
              as the FineWeb-Edu pipeline — a drop-in replacement for longer-range,
              more diverse training contexts.

  Expects:    - Dolmino-Mix on disk at a configurable source path (default:
                /bulk-store/training-datasets/dolmino_mix_100B/)
              - Source format: subdirectories each containing *.jsonl.zst shards
              - Records have at minimum a `text` field (string)
              - Default --min_text_len=2048 chars (~512 tokens) discards short
                documents so every retained document spans at least one full L3
                CMS period. This is a diagnostic requirement, not a quality
                filter — the goal is to force L2/L3 to encounter long-range
                structure that could not come from FineWeb-Edu short articles.
              - An existing 32K BPE tokenizer at data/fineweb_edu/tokenizer.json
                with EOT id=3 — do NOT train a new tokenizer
              - Python environment with `zstandard`, `tokenizers`, `numpy`
              - Writable output directory (default: data/dolmino_100b/)

  Guarantees: - Reproducible: shards traversed in sorted order; seed controls
                train/val document split
              - Memory-bounded: shards decompressed and released one at a time;
                no full corpus loaded into RAM
              - Drop-in: output directory loadable by BpeDataLoader with zero changes
              - meta.json schema matches FineWeb-Edu pipeline exactly
              - All tokens are valid next-token prediction targets (no masking,
                mask_ratio=0.0)

  Cost:       - ~2–4 CPU-hours per 100M tokens depending on decompression speed
              - No GPU required
              - Disk: ~400MB per 100M tokens (uint32 + int32 arrays)

  Trade-off:  - Standard LM objective with no quality score filter: Dolmino-Mix
                is already curated upstream (OLMo 3 selection criteria). Applying
                an additional filter would require score metadata not present in the
                JSONL schema.
              - Single-process streaming (no multiprocess I/O): avoids zstd
                decompressor contention; sufficient for CPU-bound throughput.

  Position:   specs/infrastructure/data/01_dolmino_pipeline.md
              Implements: BpeDataLoader-compatible output (engine/data.py)
              Peer to: specs/infrastructure/02_corpus_selection.md

  Source:     HOPE (2512.24695) §3 — Continuous Memory System requires a continuous
                token stream without epoch boundaries (no DataLoader shuffle reset)
              Internal: engine/data.py BpeDataLoader interface contract
              Dataset: Dolmino-Mix 100B (dolmino-mix, OLMo3 dataset),
                       Wadden et al. 2024
```

---

## Source Dataset

Dolmino-Mix 100B is the high-quality annealing dataset used for OLMo 3 32B
stage-2 midtraining. It is a curated mix of web pages, code, math, QA, reasoning
traces, and instruction data in two ingredient variants (ingredient1, ingredient2).

Each ingredient occupies a separate subdirectory tree under `data/`:

```text
/bulk-store/training-datasets/dolmino_mix_100B/
├── data/                    ← 323 subdirectories
│   ├── ingredient1-common_crawl-high-quality_19_*/
│   ├── ingredient1-code-meta-reasoning/
│   ├── ingredient2-*/
│   └── ...                  ← each dir contains *.jsonl.zst shard files
└── README.md
```

JSONL record schema (minimum required field):

```json
{"id": "...", "metadata": {...}, "text": "..."}
```

## Output Format

```text
data/dolmino_100b/
├── train_tokens.npy       ← uint32 flat token array
├── train_targets.npy      ← int32 flat target array (tokens[1:])
├── val_tokens.npy
├── val_targets.npy
├── tokenizer.json         ← symlink or copy from data/fineweb_edu/tokenizer.json
└── meta.json              ← see schema below
```

### meta.json schema

```json
{
  "vocab_size": 32000,
  "tokenizer": "tokenizer.json",
  "special_tokens": {
    "<|im_start|>": 0,
    "<|im_end|>": 1,
    "<|pad|>": 2,
    "<|endoftext|>": 3
  },
  "train": {
    "split": "train",
    "documents": <int>,
    "total_tokens": <int>,
    "valid_targets": <int>,
    "masked_targets": 0,
    "mask_ratio": 0.0
  },
  "val": {
    "split": "val",
    "documents": <int>,
    "total_tokens": <int>,
    "valid_targets": <int>,
    "masked_targets": 0,
    "mask_ratio": 0.0
  },
  "seed": <int>,
  "val_ratio": <float>,
  "source": "dolmino-mix-100b (ingredient=ingredient1)",
  "ingredient": "ingredient1"
}
```

## Algorithm

```rust
fn prepare_dolmino<P>(
    source_dir:   P,
    output_dir:   P,
    ingredient:   &str,
    target_tokens: usize,
    val_ratio:    f32,
    min_text_len: usize,
    seed:         u64,
) -> Result<(), PipelineError>
where
    P: AsRef<std::path::Path>,
{
    // 1. Discover shards (sorted for reproducibility)
    let dirs: Vec<PathBuf> = glob(source_dir / "data" / format!("{ingredient}-*"))
        .sorted()
        .collect();
    if dirs.is_empty() {
        return Err(PipelineError::NoSourceDirs { ingredient });
    }
    let shards: Vec<PathBuf> = dirs.iter()
        .flat_map(|d| glob(d / "*.jsonl.zst").sorted())
        .collect();

    // 2. Stream documents; discard those shorter than min_text_len chars
    let mut all_docs: Vec<String> = Vec::new();
    let mut total_chars: usize = 0;
    let target_chars: usize = target_tokens * 5; // headroom for tokenization ratio
    'outer: for shard in &shards {
        for record in stream_jsonl_zst(shard)? {
            let text: String = record.get("text").unwrap_or_default();
            if text.len() >= min_text_len {
                total_chars += text.len();
                all_docs.push(text);
            }
        }
        if total_chars >= target_chars { break 'outer; }
    }

    // 3. Shuffle and split (seeded, reproducible)
    let mut rng = Rng::seed(seed);
    let indices = rng.permutation(all_docs.len());
    let n_val   = 1.max((all_docs.len() as f32 * val_ratio) as usize);
    let val_set: HashSet<usize> = indices[..n_val].iter().copied().collect();
    let train_docs: Vec<&str> = all_docs.iter().enumerate()
        .filter(|(i, _)| !val_set.contains(i)).map(|(_, s)| s.as_str()).collect();
    let val_docs: Vec<&str> = val_set.iter()
        .map(|&i| all_docs[i].as_str()).collect();
    let n_train_docs = train_docs.len();
    let n_val_docs   = val_docs.len();

    // 4. Load tokenizer (no training — reuse existing 32K BPE)
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    // 5. Tokenize into flat arrays with EOT separators
    let train_tokens = tokenize_stream(&train_docs, &tokenizer,
                                       (target_tokens as f32 * (1.0 - val_ratio)) as usize)?;
    let val_tokens   = tokenize_stream(&val_docs,   &tokenizer,
                                       (target_tokens as f32 * val_ratio) as usize)?;

    // 6. Write output
    // Standard next-token prediction: input[i] predicts tokens[i+1]
    save_npy(output_dir / "train_tokens.npy",  &train_tokens[..train_tokens.len()-1], Dtype::U32)?;
    save_npy(output_dir / "train_targets.npy", &train_tokens[1..],                    Dtype::I32)?;
    save_npy(output_dir / "val_tokens.npy",    &val_tokens[..val_tokens.len()-1],     Dtype::U32)?;
    save_npy(output_dir / "val_targets.npy",   &val_tokens[1..],                      Dtype::I32)?;
    save_meta(output_dir / "meta.json",
              n_train_docs, n_val_docs, &tokenizer, ingredient, min_text_len, seed)?;
    copy_tokenizer(output_dir / "tokenizer.json")?;
    Ok(())
}
```

### Streaming helper

```rust
fn stream_jsonl_zst(path: &Path) -> impl Iterator<Item = Result<JsonValue, ParseError>> {
    let file   = File::open(path)?;
    let reader = ZstdDecompressor::new().stream_reader(file)?;
    BufReader::new(reader)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| serde_json::from_str(&line).map_err(ParseError::Json))
}
```

## Constraint Compliance

| Constraint | Mechanism |
|-----------|-----------|
| CS-17 (no DataLoader) | BpeDataLoader wraps flat npy arrays — no shuffle reset between chunks |
| CS-01 (no MemoryModule class) | Pipeline is a standalone script, not a module class |

## CLI Interface

```bash
python scripts/prepare_dolmino.py [OPTIONS]

Options:
  --source         Path to Dolmino-Mix root dir (default: /bulk-store/training-datasets/dolmino_mix_100B/)
  --output         Output directory (default: data/dolmino_100b)
  --ingredient     Which ingredient to use: ingredient1 | ingredient2 | both (default: ingredient1)
  --target_tokens  Target total tokens across train+val (default: 100_000_000)
  --val_ratio      Fraction of documents held out for validation (default: 0.05)
  --seed           Random seed for shuffle/split (default: 42)
  --tokenizer      Path to BPE tokenizer json (default: data/fineweb_edu/tokenizer.json)
  --min_text_len   Minimum document length in chars (default: 2048 ≈ 512 tokens).
                   Documents shorter than this are discarded. 2048 ensures every
                   document spans at least one full L3 CMS period — the diagnostic
                   requirement for meaningful L2/L3 gate activation testing.
```
