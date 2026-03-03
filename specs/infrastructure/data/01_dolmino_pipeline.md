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

```
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

```
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

```
fn prepare_dolmino(source_dir, output_dir, ingredient, target_tokens, val_ratio, seed):
    # 1. Discover shards (sorted for reproducibility)
    dirs = sorted(glob(source_dir / "data" / f"{ingredient}-*"))
    if len(dirs) == 0:
        error("No source directories found for ingredient={ingredient}")

    shards = sorted(path for d in dirs for path in glob(d / "*.jsonl.zst"))

    # 2. Stream documents without accumulating corpus in RAM
    all_docs: list[str] = []
    for shard in shards:
        for record in stream_jsonl_zst(shard):
            text = record.get("text", "")
            if len(text) >= MIN_TEXT_LEN:
                all_docs.append(text)

    # 3. Shuffle and split (seeded, reproducible)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_docs))
    n_val = max(1, int(len(all_docs) * val_ratio))
    val_set  = {indices[i] for i in range(n_val)}
    train_docs = [all_docs[i] for i in range(len(all_docs)) if i not in val_set]
    val_docs   = [all_docs[i] for i in val_set]

    # 4. Load tokenizer (no training)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 5. Tokenize into flat arrays with EOT separators
    train_tokens = tokenize_stream(train_docs, tokenizer, target_tokens * (1 - val_ratio))
    val_tokens   = tokenize_stream(val_docs, tokenizer, target_tokens * val_ratio)

    # 6. Write output
    save_npy(output_dir / "train_tokens.npy", train_tokens, dtype=uint32)
    save_npy(output_dir / "train_targets.npy", train_tokens[1:] concat [EOT], dtype=int32)
    # (targets array is tokens shifted by 1 — standard next-token prediction)
    save_meta(output_dir / "meta.json", ...)
    copy_tokenizer(output_dir / "tokenizer.json")
```

### Streaming helper

```
fn stream_jsonl_zst(path):
    with open(path, "rb") as f:
        reader = ZstdDecompressor().stream_reader(f)
        for line in TextIOWrapper(reader):
            yield json.loads(line)
```

## Constraint Compliance

| Constraint | Mechanism |
|-----------|-----------|
| CS-17 (no DataLoader) | BpeDataLoader wraps flat npy arrays — no shuffle reset between chunks |
| CS-01 (no MemoryModule class) | Pipeline is a standalone script, not a module class |

## CLI Interface

```
python scripts/prepare_dolmino.py [OPTIONS]

Options:
  --source         Path to Dolmino-Mix root dir (default: /bulk-store/training-datasets/dolmino_mix_100B/)
  --output         Output directory (default: data/dolmino_100b)
  --ingredient     Which ingredient to use: ingredient1 | ingredient2 | both (default: ingredient1)
  --target_tokens  Target total tokens across train+val (default: 100_000_000)
  --val_ratio      Fraction of documents held out for validation (default: 0.05)
  --seed           Random seed for shuffle/split (default: 42)
  --tokenizer      Path to BPE tokenizer json (default: data/fineweb_edu/tokenizer.json)
```
