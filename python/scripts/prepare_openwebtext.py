#!/usr/bin/env python3
"""
Prepare OpenWebText for NL-Hecate using GPT-2's tokenizer.

Reads pre-tokenized OpenWebText arrow shards (already GPT-2 BPE tokenized),
concatenates them into the standard NL-Hecate format:
  train_tokens.npy, train_targets.npy, val_tokens.npy, val_targets.npy, meta.json

Source: /bulk-store/training-datasets/openwebtext/apollo-research___skylion007-openwebtext-tokenizer-gpt2/

Usage:
    python scripts/prepare_openwebtext.py \
        --input /bulk-store/training-datasets/openwebtext/apollo-research___skylion007-openwebtext-tokenizer-gpt2/default/0.0.0/<hash>/ \
        --output python/data/openwebtext_gpt2 \
        --target-tokens 100_000_000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


GPT2_VOCAB_SIZE = 50257
GPT2_EOS_ID = 50256
VAL_FRAC = 0.005  # 0.5% for validation


def prepare_targets(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """Next-token prediction targets. Last token masked with vocab_size."""
    if tokens.size == 0:
        return np.empty(0, dtype=np.uint32)
    targets = np.empty_like(tokens)
    targets[:-1] = tokens[1:]
    targets[-1] = vocab_size
    return targets


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenWebText for NL-Hecate")
    parser.add_argument("--input", required=True, help="Directory with .arrow shards")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target-tokens", type=int, default=100_000_000,
                        help="Target total tokens (default 100M)")
    parser.add_argument("--text-column", default="input_ids",
                        help="Column name with token IDs in arrow files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find arrow shards
    arrow_files = sorted(input_dir.glob("*.arrow"))
    if not arrow_files:
        sys.exit(f"No .arrow files found in {input_dir}")

    print(f"Found {len(arrow_files)} arrow shards")
    print(f"Target: {args.target_tokens:,} tokens")

    try:
        import pyarrow as pa
    except ImportError:
        sys.exit("pyarrow required. pip install pyarrow")

    # Stream through shards, collecting token IDs
    from array import array as _array
    all_tokens = _array("I")  # uint32, memory-efficient
    t0 = time.time()
    doc_count = 0

    for shard_path in arrow_files:
        if len(all_tokens) >= args.target_tokens:
            break

        table = pa.ipc.open_file(shard_path).read_all()
        column_names = table.column_names

        # Try different column names
        token_col = None
        for col_name in [args.text_column, "input_ids", "tokens", "token_ids", "text"]:
            if col_name in column_names:
                token_col = col_name
                break

        if token_col is None:
            print(f"  Warning: no token column found in {shard_path.name}")
            print(f"  Available columns: {column_names}")
            # If there's a 'text' column, we need to tokenize
            if "text" in column_names:
                token_col = "text"
            else:
                continue

        col = table.column(token_col)

        for i in range(len(col)):
            if len(all_tokens) >= args.target_tokens:
                break

            chunk = col[i]
            if token_col == "text":
                # Need to tokenize -- skip for now, handle below
                continue

            # chunk should be a list of ints
            if hasattr(chunk, "as_py"):
                ids = chunk.as_py()
            else:
                ids = list(chunk)

            if isinstance(ids, list):
                all_tokens.extend(ids)
                all_tokens.append(GPT2_EOS_ID)  # document boundary
                doc_count += 1
            elif isinstance(ids, (int, np.integer)):
                # Single token per row -- this is pre-flattened
                all_tokens.append(int(ids))
                doc_count += 1

        elapsed = time.time() - t0
        rate = len(all_tokens) / max(elapsed, 0.001)
        print(f"  {shard_path.name}: {len(all_tokens):,} tokens "
              f"({doc_count:,} docs, {rate:.0f} tok/s)")

    if len(all_tokens) == 0:
        # The arrow files might have a different structure -- try direct read
        print("\nNo tokens found via column scan. Trying datasets library...")
        try:
            from datasets import Dataset
            ds = Dataset.from_file(str(arrow_files[0]))
            print(f"  Columns: {ds.column_names}")
            print(f"  First row keys: {list(ds[0].keys()) if len(ds) > 0 else 'empty'}")
            if len(ds) > 0:
                first = ds[0]
                for k, v in first.items():
                    vtype = type(v).__name__
                    vlen = len(v) if hasattr(v, '__len__') else 'scalar'
                    print(f"    {k}: {vtype}, len={vlen}")
                    if isinstance(v, list) and len(v) > 0:
                        print(f"      sample: {v[:5]}")
        except Exception as e:
            print(f"  datasets fallback failed: {e}")

        sys.exit("Could not extract tokens. Check column structure.")

    total = len(all_tokens)
    tokens_arr = np.array(all_tokens, dtype=np.uint32)

    # Split train/val
    val_size = max(int(total * VAL_FRAC), 1024)
    train_tokens = tokens_arr[:-val_size]
    val_tokens = tokens_arr[-val_size:]

    train_targets = prepare_targets(train_tokens, GPT2_VOCAB_SIZE)
    val_targets = prepare_targets(val_tokens, GPT2_VOCAB_SIZE)

    # Save
    np.save(output_dir / "train_tokens.npy", train_tokens)
    np.save(output_dir / "train_targets.npy", train_targets)
    np.save(output_dir / "val_tokens.npy", val_tokens)
    np.save(output_dir / "val_targets.npy", val_targets)

    meta = {
        "vocab_size": GPT2_VOCAB_SIZE,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_tokens": total,
        "documents": doc_count,
        "source": "openwebtext (GPT-2 tokenized)",
        "tokenizer": "gpt2",
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
