#!/usr/bin/env python3
"""
Prepare Dolmino-Mix 100B for NL-Hecate training.

Streams *.jsonl.zst shards from all source subdirectories (sorted for
reproducibility), tokenizes using the existing 32K BPE tokenizer from
FineWeb-Edu, and writes flat numpy arrays compatible with BpeDataLoader.

No quality score filter — Dolmino-Mix is already curated upstream.
No tokenizer training — reuses data/fineweb_edu/tokenizer.json.

Usage:
    python scripts/prepare_dolmino.py
    python scripts/prepare_dolmino.py --target_tokens 1_000_000_000
    python scripts/prepare_dolmino.py --ingredient ingredient2 --output data/dolmino_i2
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = "/bulk-store/training-datasets/dolmino_mix_100B/"
DEFAULT_TOKENIZER = "data/fineweb_edu/tokenizer.json"
EOT_ID = 3  # <|endoftext|> — same across all NL-Hecate pipelines
MIN_TEXT_LEN = 100  # skip very short fragments


def stream_jsonl_zst(path: Path):
    """Yield parsed JSON records from a single .jsonl.zst shard."""
    import zstandard as zstd

    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def collect_shards(source_dir: Path, ingredient: str) -> list[Path]:
    """Return sorted list of all .jsonl.zst shard paths for the ingredient(s)."""
    data_dir = source_dir / "data"
    if not data_dir.is_dir():
        print(f"ERROR: Source data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if ingredient == "both":
        patterns = ["ingredient1-*", "ingredient2-*"]
    else:
        patterns = [f"{ingredient}-*"]

    dirs = []
    for pattern in patterns:
        dirs.extend(sorted(data_dir.glob(pattern)))
    dirs.sort()

    if not dirs:
        print(
            f"ERROR: No source directories found for ingredient='{ingredient}' "
            f"under {data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    shards = []
    for d in dirs:
        shards.extend(sorted(d.glob("*.jsonl.zst")))

    print(f"  Found {len(dirs)} source directories, {len(shards)} shards")
    return shards


def stream_documents(shards: list[Path], target_chars: int) -> list[str]:
    """Stream documents from shards until target_chars reached.

    Documents are accumulated as strings (not tokens) to allow a seeded
    train/val split at the document level before tokenization.
    """
    docs: list[str] = []
    total_chars = 0
    total_records = 0
    skipped = 0

    for shard_idx, shard in enumerate(shards):
        for rec in stream_jsonl_zst(shard):
            total_records += 1
            text = rec.get("text", "")
            if not text or len(text) < MIN_TEXT_LEN:
                skipped += 1
                continue
            docs.append(text)
            total_chars += len(text)

            if total_records % 50_000 == 0:
                print(
                    f"    shard {shard_idx+1}/{len(shards)}: "
                    f"{total_records:,} records, {len(docs):,} kept, "
                    f"{total_chars:,} chars...",
                    end="\r",
                )

        if total_chars >= target_chars:
            break

    print(
        f"\n  Streamed {total_records:,} records → {len(docs):,} kept, "
        f"{skipped:,} skipped (<{MIN_TEXT_LEN} chars)"
    )
    print(f"  Total chars: {total_chars:,}")
    return docs


def tokenize_documents(
    docs: list[str], tokenizer, target_tokens: int
) -> tuple[list[int], list[int]]:
    """Tokenize docs into flat token/target arrays with EOT document separators."""
    all_tokens: list[int] = []

    for i, doc in enumerate(docs):
        ids = tokenizer.encode(doc).ids
        all_tokens.extend(ids)
        all_tokens.append(EOT_ID)

        if i % 10_000 == 0 and i > 0:
            print(
                f"    tokenized {i:,}/{len(docs):,} docs, "
                f"{len(all_tokens):,} tokens...",
                end="\r",
            )
        if len(all_tokens) >= target_tokens:
            break

    all_tokens = all_tokens[:target_tokens]
    print(f"\n  Total tokens: {len(all_tokens):,}")

    # Standard next-token prediction: input[i] predicts target[i] = input[i+1]
    input_tokens = all_tokens[:-1]
    target_tokens_list = all_tokens[1:]
    return input_tokens, target_tokens_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Dolmino-Mix 100B for NL-Hecate"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help=f"Path to Dolmino-Mix root directory (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dolmino_100b",
        help="Output directory (default: data/dolmino_100b)",
    )
    parser.add_argument(
        "--ingredient",
        type=str,
        default="ingredient1",
        choices=["ingredient1", "ingredient2", "both"],
        help="Ingredient variant to process (default: ingredient1)",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=100_000_000,
        help="Target total token count across train+val (default: 100M)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Fraction of documents held out for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val shuffle (default: 42)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Path to existing BPE tokenizer json (default: {DEFAULT_TOKENIZER})",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.output)
    tokenizer_path = Path(args.tokenizer)

    if not source_dir.is_dir():
        print(f"ERROR: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    if not tokenizer_path.exists():
        print(
            f"ERROR: Tokenizer not found: {tokenizer_path}\n"
            "       Run prepare_fineweb_edu.py first to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover shards ───────────────────────────────────────
    print(f"Step 1: Discovering shards (ingredient={args.ingredient})...")
    shards = collect_shards(source_dir, args.ingredient)

    # ── Step 2: Stream documents ──────────────────────────────────────
    # Accumulate strings (not tokens) — 3-5x smaller than token arrays.
    # At 100M tokens / ~4 chars-per-token: ~400MB string data.
    target_chars = int(args.target_tokens * 4.5)  # headroom for tokenization ratio
    print(f"\nStep 2: Streaming documents (target ≈ {args.target_tokens:,} tokens)...")
    t0 = time.time()
    docs = stream_documents(shards, target_chars)
    print(f"  Done in {time.time() - t0:.1f}s")

    if len(docs) < 10:
        print("ERROR: Too few documents extracted. Check source path and ingredient.",
              file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Shuffle and split ─────────────────────────────────────
    print(
        f"\nStep 3: Splitting {len(docs):,} docs "
        f"(seed={args.seed}, val_ratio={args.val_ratio})..."
    )
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * args.val_ratio))
    val_set = set(indices[:n_val].tolist())
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_set]
    val_docs = [docs[i] for i in val_set]
    print(f"  Train: {len(train_docs):,} docs, Val: {len(val_docs):,} docs")

    # Free original list immediately — we hold train/val references
    docs = []

    # ── Step 4: Load tokenizer ────────────────────────────────────────
    print(f"\nStep 4: Loading tokenizer: {tokenizer_path}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab:,}")

    # ── Step 5: Tokenize ──────────────────────────────────────────────
    print(f"\nStep 5: Tokenizing (target={args.target_tokens:,} total tokens)...")
    t0 = time.time()

    train_target = int(args.target_tokens * (1 - args.val_ratio))
    val_target = args.target_tokens - train_target

    print("  Train split:")
    train_input, train_targets = tokenize_documents(train_docs, tokenizer, train_target)
    train_docs = []  # free memory

    print("  Val split:")
    val_input, val_targets = tokenize_documents(val_docs, tokenizer, val_target)
    val_docs = []

    print(f"  Tokenized in {time.time() - t0:.1f}s")

    # ── Step 6: Save ──────────────────────────────────────────────────
    print("\nStep 6: Saving output files...")

    np.save(out_dir / "train_tokens.npy", np.array(train_input, dtype=np.uint32))
    np.save(out_dir / "train_targets.npy", np.array(train_targets, dtype=np.int32))
    np.save(out_dir / "val_tokens.npy", np.array(val_input, dtype=np.uint32))
    np.save(out_dir / "val_targets.npy", np.array(val_targets, dtype=np.int32))

    # Tokenizer copy (not symlink — portable across mounts)
    dest_tok = out_dir / "tokenizer.json"
    if not dest_tok.exists():
        shutil.copy2(tokenizer_path, dest_tok)

    source_label = f"dolmino-mix-100b (ingredient={args.ingredient})"
    meta = {
        "vocab_size": actual_vocab,
        "tokenizer": "tokenizer.json",
        "special_tokens": {
            "<|im_start|>": 0,
            "<|im_end|>": 1,
            "<|pad|>": 2,
            "<|endoftext|>": EOT_ID,
        },
        "train": {
            "split": "train",
            "documents": len(train_input) // 512 or 0,  # rough; exact below
            "total_tokens": len(train_input),
            "valid_targets": len(train_targets),
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "val": {
            "split": "val",
            "documents": len(val_input) // 512 or 0,
            "total_tokens": len(val_input),
            "valid_targets": len(val_targets),
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source": source_label,
        "ingredient": args.ingredient,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    total = len(train_input) + len(val_input)
    train_mb = (len(train_input) * 4) / 1e6  # uint32
    val_mb = (len(val_input) * 4) / 1e6
    print(f"\n{'=' * 60}")
    print("Dolmino-Mix data preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {out_dir}")
    print(f"  Source:       {source_label}")
    print(f"  Ingredient:   {args.ingredient}")
    print(f"  Vocab:        {actual_vocab:,}")
    print(f"  Total tokens: {total:,}")
    print(f"  Train tokens: {len(train_input):,}  ({train_mb:.1f} MB)")
    print(f"  Val tokens:   {len(val_input):,}  ({val_mb:.1f} MB)")
    print("  Mask ratio:   0.0% (all tokens are valid targets)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
