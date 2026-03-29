#!/usr/bin/env python3
"""
Prepare Tulu-3 SFT data for CMS-aware curriculum learning.

Reads the tulu-3-sft subset from Dolmino-Mix 100B, tokenizes with the
same BPE tokenizer used for base training, and writes flat numpy arrays
compatible with BpeTokenStream.

This dataset is intended for Phase 2 of CMS curriculum learning:
feed instruction-response pairs once L1 is developing (~1.7x bias movement)
so that L2 fire events carry instruction-following structure.

Unlike the base dolmino prep, we use a LOWER min_text_len (256 chars)
because SFT instruction-response pairs are naturally shorter than
long-form documents, and that's fine — the instruction-response
*structure* is what L1/L2 need to encode, not raw length.

Usage:
    python scripts/prepare_tulu3_sft.py
    python scripts/prepare_tulu3_sft.py --target_tokens 50_000_000
    python scripts/prepare_tulu3_sft.py --min_text_len 512 --english_only
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

TULU3_DIR = Path("/bulk-store/training-datasets/dolmino_mix_100B/data/ingredient1-tulu-3-sft")
DEFAULT_TOKENIZER = "data/fineweb_edu/tokenizer.json"
EOT_ID = 3  # <|endoftext|>
MIN_TEXT_LEN_DEFAULT = 256  # Lower than base dolmino — SFT pairs are shorter


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


def is_english_ish(text: str) -> bool:
    """Quick heuristic: check if text is mostly ASCII (English/code)."""
    if len(text) == 0:
        return False
    ascii_count = sum(1 for c in text[:500] if ord(c) < 128)
    return ascii_count / min(len(text), 500) > 0.85


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tulu-3 SFT data for CMS curriculum learning"
    )
    parser.add_argument(
        "--source", type=str, default=str(TULU3_DIR),
        help=f"Path to tulu-3-sft directory (default: {TULU3_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default="data/tulu3_sft",
        help="Output directory (default: data/tulu3_sft)",
    )
    parser.add_argument(
        "--target_tokens", type=int, default=100_000_000,
        help="Target token count for train+val (default: 100M)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Fraction of documents for validation (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER,
        help=f"Path to BPE tokenizer json (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--min_text_len", type=int, default=MIN_TEXT_LEN_DEFAULT,
        help="Minimum document length in chars (default: 256)",
    )
    parser.add_argument(
        "--english_only", action="store_true",
        help="Filter to English-ish documents only (ASCII > 85%%)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    # Collect shards
    shards = sorted(source.glob("*.jsonl.zst"))
    if not shards:
        print(f"ERROR: No .jsonl.zst files found in {source}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(shards)} shards in {source}")

    # Stream and filter documents
    # ~4 chars per token, so target chars ≈ target_tokens * 4
    target_chars = args.target_tokens * 4
    docs = []
    total_records = 0
    skipped_short = 0
    skipped_lang = 0
    total_chars = 0

    t0 = time.time()
    for shard_idx, shard in enumerate(shards):
        for rec in stream_jsonl_zst(shard):
            total_records += 1
            text = rec.get("text", "")
            if not isinstance(text, str) or len(text) < args.min_text_len:
                skipped_short += 1
                continue
            if args.english_only and not is_english_ish(text):
                skipped_lang += 1
                continue

            docs.append(text)
            total_chars += len(text)

            if total_records % 50_000 == 0:
                print(
                    f"  shard {shard_idx+1}/{len(shards)}: "
                    f"{total_records:,} records, {len(docs):,} kept, "
                    f"{total_chars:,} chars...",
                    end="\r",
                )
            if total_chars >= target_chars:
                break
        if total_chars >= target_chars:
            break

    elapsed = time.time() - t0
    print(f"\n  Streamed {total_records:,} records in {elapsed:.1f}s")
    print(f"  Kept: {len(docs):,}  Skipped short: {skipped_short:,}  Skipped lang: {skipped_lang:,}")
    print(f"  Total chars: {total_chars:,}")

    if not docs:
        print("ERROR: No documents survived filtering", file=sys.stderr)
        sys.exit(1)

    # Shuffle and split
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(docs))
    val_count = max(1, int(len(docs) * args.val_ratio))
    val_indices = set(indices[:val_count].tolist())

    train_docs = [docs[i] for i in range(len(docs)) if i not in val_indices]
    val_docs = [docs[i] for i in range(len(docs)) if i in val_indices]
    print(f"  Split: {len(train_docs):,} train, {len(val_docs):,} val")

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"  Tokenizer: {args.tokenizer} (vocab={tokenizer.get_vocab_size()})")

    # Tokenize
    def tokenize_docs(doc_list, label, target):
        all_tokens = []
        for start in range(0, len(doc_list), 10_000):
            batch = doc_list[start:start + 10_000]
            encoded = tokenizer.encode_batch(batch)
            for enc in encoded:
                all_tokens.extend(enc.ids)
                all_tokens.append(EOT_ID)
            print(
                f"    {label}: {min(start+10_000, len(doc_list)):,}/{len(doc_list):,} docs, "
                f"{len(all_tokens):,} tokens...",
                end="\r",
            )
            if len(all_tokens) >= target:
                break
        all_tokens = all_tokens[:target]
        print(f"\n    {label}: {len(all_tokens):,} tokens")
        return all_tokens

    train_target = int(args.target_tokens * (1 - args.val_ratio))
    val_target = args.target_tokens - train_target

    print("Tokenizing...")
    train_tokens = tokenize_docs(train_docs, "train", train_target)
    val_tokens = tokenize_docs(val_docs, "val", val_target)

    # Write output
    output.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer
    import shutil
    tok_dst = output / "tokenizer.json"
    if not tok_dst.exists():
        shutil.copy2(args.tokenizer, tok_dst)
        print(f"  Copied tokenizer to {tok_dst}")

    # Write arrays (input = tokens[:-1], target = tokens[1:])
    for name, toks, doc_count in [
        ("train", train_tokens, len(train_docs)),
        ("val", val_tokens, len(val_docs)),
    ]:
        inp = np.array(toks[:-1], dtype=np.int32)
        tgt = np.array(toks[1:], dtype=np.int32)
        np.save(output / f"{name}_tokens.npy", inp)
        np.save(output / f"{name}_targets.npy", tgt)
        print(f"  Wrote {name}: {len(inp):,} tokens from {doc_count:,} docs")

    # Meta
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": "tokenizer.json",
        "special_tokens": {"<|endoftext|>": EOT_ID},
        "source": "tulu-3-sft (dolmino_mix_100B/ingredient1-tulu-3-sft)",
        "purpose": "CMS Phase 2 curriculum — instruction following for L1/L2 development",
        "min_text_len": args.min_text_len,
        "english_only": args.english_only,
        "train": {
            "tokens": len(train_tokens) - 1,
            "documents": len(train_docs),
        },
        "val": {
            "tokens": len(val_tokens) - 1,
            "documents": len(val_docs),
        },
    }
    with open(output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote meta.json")
    print(f"\nDone! Output: {output}")


if __name__ == "__main__":
    main()
