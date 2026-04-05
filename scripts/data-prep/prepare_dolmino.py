#!/usr/bin/env python3
"""
Prepare Dolmino-Mix 100B for NL-Hecate training.

Streams *.jsonl.zst shards from all source subdirectories (sorted for
reproducibility), tokenizes using the existing 32K BPE tokenizer from
FineWeb-Edu, and writes flat numpy arrays compatible with BpeTokenStream.

No quality score filter — Dolmino-Mix is already curated upstream.
No tokenizer training — reuses data/fineweb_edu/tokenizer.json.

Default --min_text_len=2048 (~512 tokens) ensures every document in the stream
can span at least one full L3 CMS period. Shorter documents are discarded.

Usage:
    python scripts/prepare_dolmino.py
    python scripts/prepare_dolmino.py --target_tokens 1_000_000_000
    python scripts/prepare_dolmino.py --ingredient ingredient2 --output data/dolmino_i2
    python scripts/prepare_dolmino.py --min_text_len 4096   # stricter: force 2x L3
"""

import argparse
import io
import json
import filecmp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_SOURCE = "/bulk-store/training-datasets/dolmino_mix_100B/"
DEFAULT_TOKENIZER = "data/fineweb_edu/tokenizer.json"
EOT_ID = 3  # <|endoftext|> — same across all NL-Hecate pipelines
# Default minimum document length in characters.
# 2048 chars ≈ 512 tokens at ~4 chars/token — ensures every document spans
# at least one full L3 CMS period. Shorter documents are discarded so the
# token stream is composed entirely of genuinely long-range content.
MIN_TEXT_LEN_DEFAULT = 2048


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
                except json.JSONDecodeError as exc:
                    print(
                        f"  warning: JSON decode error in {path.name}: {exc!r}",
                        file=sys.stderr,
                    )
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


def stream_documents(
    shards: list[Path], target_chars: int, min_text_len: int,
    skip_docs: int = 0,
) -> list[str]:
    """Stream documents from shards until target_chars reached.

    Documents shorter than min_text_len chars are discarded. With the default
    of 2048 chars (~512 tokens) every retained document spans at least one full
    L3 CMS period, making the stream suitable for diagnosing L2/L3 activation.

    Documents are accumulated as strings (not tokens) to allow a seeded
    train/val split at the document level before tokenization.

    If skip_docs > 0, that many qualifying documents (passing min_text_len
    filter) are skipped before accumulation begins. This lets volume N+1 pick
    up exactly where volume N left off in the sorted shard stream.
    """
    docs: list[str] = []
    total_chars = 0
    total_records = 0
    skipped_short = 0
    skipped_offset = 0
    kept = 0

    for shard_idx, shard in enumerate(shards):
        for rec in stream_jsonl_zst(shard):
            total_records += 1
            text = rec.get("text", "")
            if not isinstance(text, str) or len(text) < min_text_len:
                skipped_short += 1
                continue

            kept += 1
            if kept <= skip_docs:
                skipped_offset += 1
                if skipped_offset % 50_000 == 0:
                    print(
                        f"    skipping: {skipped_offset:,}/{skip_docs:,} docs...",
                        end="\r",
                    )
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

    if skip_docs > 0:
        print(f"\n  Skipped {skipped_offset:,} docs (volume offset)")
    print(
        f"  Streamed {total_records:,} records → {len(docs):,} kept, "
        f"{skipped_short:,} skipped (<{min_text_len} chars)"
    )
    print(f"  Total chars: {total_chars:,}")
    return docs


def tokenize_documents_to_mmap(
    docs: list[str], tokenizer, target_tokens: int, eot_id: int,
    tokens_path: Path, targets_path: Path,
    batch_size: int = 10_000,
) -> int:
    """Tokenize docs directly into memory-mapped numpy files.

    Writes tokens incrementally to avoid holding 10B+ ints in a Python list.
    Returns the actual number of token pairs written (may be < target_tokens).
    """
    # Pre-allocate mmap files at target size
    tokens_arr = np.lib.format.open_memmap(
        str(tokens_path), mode='w+', dtype=np.uint32, shape=(target_tokens,))
    targets_arr = np.lib.format.open_memmap(
        str(targets_path), mode='w+', dtype=np.int32, shape=(target_tokens,))

    # Small buffer to accumulate a batch before writing to mmap
    buf = np.empty(batch_size * 3000, dtype=np.uint32)  # ~12MB, generous per-batch
    buf_pos = 0
    written = 0  # token pairs written to mmap (offset by 1 for next-token pred)
    carry = -1   # last token from previous flush (needed for targets[0] of next chunk)

    def flush_buf():
        nonlocal buf_pos, written, carry
        if buf_pos == 0:
            return
        chunk = buf[:buf_pos]
        # Next-token prediction: tokens[i] predicts targets[i] = tokens[i+1]
        if carry >= 0:
            # Bridge: last token of previous flush -> first token of this chunk
            n = min(len(chunk), target_tokens - written)
            tokens_arr[written:written + n] = np.concatenate(
                [np.array([carry], dtype=np.uint32), chunk[:n - 1]]) if n > 1 else np.array([carry], dtype=np.uint32)[:n]
            targets_arr[written:written + n] = chunk[:n].astype(np.int32)
            written += n
        else:
            # First flush: tokens = chunk[:-1], targets = chunk[1:]
            n = min(len(chunk) - 1, target_tokens - written)
            tokens_arr[written:written + n] = chunk[:n]
            targets_arr[written:written + n] = chunk[1:n + 1].astype(np.int32)
            written += n
        carry = int(chunk[buf_pos - 1])
        buf_pos = 0

    for start in range(0, len(docs), batch_size):
        if written >= target_tokens:
            break
        batch = docs[start:start + batch_size]
        encoded = tokenizer.encode_batch(batch)
        for enc in encoded:
            ids = enc.ids
            need = len(ids) + 1  # +1 for EOT
            # Grow buffer if needed
            if buf_pos + need > len(buf):
                flush_buf()
                if need > len(buf):
                    buf = np.empty(need * 2, dtype=np.uint32)
            buf[buf_pos:buf_pos + len(ids)] = ids
            buf_pos += len(ids)
            buf[buf_pos] = eot_id
            buf_pos += 1

        # Flush periodically to keep memory low
        if buf_pos > 5_000_000:
            flush_buf()

        print(
            f"    tokenized {min(start + batch_size, len(docs)):,}/{len(docs):,} docs, "
            f"{written:,} tokens written...",
            end="\r",
        )

    flush_buf()  # final flush

    # Truncate mmap files to actual size if we wrote less than target
    actual = min(written, target_tokens)
    if actual < target_tokens:
        del tokens_arr, targets_arr
        # Rewrite with correct size
        tmp_tok = np.load(str(tokens_path), mmap_mode='r')[:actual].copy()
        tmp_tgt = np.load(str(targets_path), mmap_mode='r')[:actual].copy()
        np.save(str(tokens_path), tmp_tok)
        np.save(str(targets_path), tmp_tgt)
        del tmp_tok, tmp_tgt
    else:
        tokens_arr.flush()
        targets_arr.flush()

    print(f"\n  Total token pairs: {actual:,}")
    return actual


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
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=MIN_TEXT_LEN_DEFAULT,
        help=(
            "Minimum document length in characters. Documents shorter than this "
            "are discarded before tokenization. Default 2048 ≈ 512 tokens — "
            "ensures every document spans at least one full L3 CMS period."
        ),
    )
    parser.add_argument(
        "--skip_docs",
        type=int,
        default=0,
        help=(
            "Number of qualifying documents to skip before accumulating. "
            "Use this to create volume N+1 by skipping the docs volume N consumed. "
            "Example: volume 1 used 358524 docs → --skip_docs 358524 for volume 2."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        parser.error(f"--val_ratio must be in (0, 1), got {args.val_ratio}")
    if args.target_tokens <= 0:
        parser.error(f"--target_tokens must be > 0, got {args.target_tokens}")
    if args.min_text_len <= 0:
        parser.error(f"--min_text_len must be > 0, got {args.min_text_len}")

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
    print(
        f"\nStep 2: Streaming documents "
        f"(target ≈ {args.target_tokens:,} tokens, min_text_len={args.min_text_len:,} chars)..."
    )
    t0 = time.time()
    docs = stream_documents(shards, target_chars, args.min_text_len,
                            skip_docs=args.skip_docs)
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
    val_indices = indices[:n_val].tolist()
    val_set = set(val_indices)
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_set]
    val_docs = [docs[i] for i in val_indices]
    print(f"  Train: {len(train_docs):,} docs, Val: {len(val_docs):,} docs")

    # Free original list immediately — we hold train/val references
    docs = []

    # ── Step 4: Load tokenizer ────────────────────────────────────────
    print(f"\nStep 4: Loading tokenizer: {tokenizer_path}")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab:,}")

    # Resolve EOT token ID from the tokenizer itself (not hardcoded).
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        print("ERROR: Tokenizer has no <|endoftext|> token.", file=sys.stderr)
        sys.exit(1)
    print(f"  EOT token: <|endoftext|> = {eot_id}")

    # ── Step 5: Tokenize (streaming to mmap) ────────────────────────
    print(f"\nStep 5: Tokenizing (target={args.target_tokens:,} total tokens, streaming to mmap)...")
    t0 = time.time()

    train_target = int(args.target_tokens * (1 - args.val_ratio))
    val_target = args.target_tokens - train_target

    n_train_docs = len(train_docs)
    n_val_docs = len(val_docs)

    print("  Train split:")
    n_train = tokenize_documents_to_mmap(
        train_docs, tokenizer, train_target, eot_id,
        out_dir / "train_tokens.npy", out_dir / "train_targets.npy",
    )
    train_docs = []  # free memory

    print("  Val split:")
    n_val = tokenize_documents_to_mmap(
        val_docs, tokenizer, val_target, eot_id,
        out_dir / "val_tokens.npy", out_dir / "val_targets.npy",
    )
    val_docs = []

    print(f"  Tokenized in {time.time() - t0:.1f}s")

    # ── Step 6: Save metadata ─────────────────────────────────────────
    print("\nStep 6: Saving metadata...")

    # Tokenizer copy (not symlink — portable across mounts).
    dest_tok = out_dir / "tokenizer.json"
    if not dest_tok.exists() or not filecmp.cmp(tokenizer_path, dest_tok, shallow=False):
        shutil.copy2(tokenizer_path, dest_tok)

    source_label = (
        f"dolmino-mix-100b (ingredient={args.ingredient}, "
        f"min_text_len={args.min_text_len})"
    )
    meta = {
        "vocab_size": actual_vocab,
        "tokenizer": "tokenizer.json",
        "special_tokens": {
            "<|endoftext|>": eot_id,
        },
        "train": {
            "split": "train",
            "documents": n_train_docs,
            "total_tokens": n_train,
            "valid_targets": n_train,
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "val": {
            "split": "val",
            "documents": n_val_docs,
            "total_tokens": n_val,
            "valid_targets": n_val,
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source": source_label,
        "ingredient": args.ingredient,
        "min_text_len": args.min_text_len,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    total = n_train + n_val
    train_mb = (n_train * 4) / 1e6  # uint32
    val_mb = (n_val * 4) / 1e6
    print(f"\n{'=' * 60}")
    print("Dolmino-Mix data preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {out_dir}")
    print(f"  Source:       {source_label}")
    print(f"  Ingredient:   {args.ingredient}")
    print(f"  Vocab:        {actual_vocab:,}")
    print(f"  Total tokens: {total:,}")
    print(f"  Train tokens: {n_train:,}  ({train_mb:.1f} MB)")
    print(f"  Val tokens:   {n_val:,}  ({val_mb:.1f} MB)")
    print("  Mask ratio:   0.0% (all tokens are valid targets)")
    print(f"  Min doc len:  {args.min_text_len:,} chars (~{args.min_text_len//4} tokens)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
