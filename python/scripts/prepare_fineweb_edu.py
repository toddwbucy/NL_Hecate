#!/usr/bin/env python3
"""
Prepare FineWeb-Edu (score 4+5) for NL-Hecate training.

Reads high-quality educational documents from the local FineWeb-Edu-dedup
parquet, trains a 32K BPE tokenizer, and generates parallel token/target
arrays in the same format as prepare_sharegpt.py.

Unlike ShareGPT: no ChatML wrapping, no loss masking. Every token is a
valid next-token prediction target (standard LM objective).

Usage:
    python scripts/prepare_fineweb_edu.py
    python scripts/prepare_fineweb_edu.py --target_tokens 100_000_000
    python scripts/prepare_fineweb_edu.py --min_score 4 --output data/fineweb_edu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LOCAL_PARQUET = "/bulk-store/training-datasets/smollm-corpus/fineweb-edu-dedup/data.parquet"

# Special tokens — same IDs as ShareGPT pipeline for compatibility
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|pad|>", "<|endoftext|>"]
EOT_ID = 3  # document separator


def extract_documents(parquet_path: str, min_score: int,
                      target_chars: int) -> list[str]:
    """Extract documents with int_score >= min_score up to target_chars."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    docs = []
    total_chars = 0
    total_scanned = 0
    kept = 0

    print(f"  Scanning parquet (min_score={min_score})...")
    for batch in pf.iter_batches(batch_size=10_000, columns=["text", "metadata"]):
        texts = batch.column("text")
        metas = batch.column("metadata")
        for i in range(len(texts)):
            total_scanned += 1
            m = metas[i].as_py()
            score = m.get("int_score", 0)
            if score >= min_score:
                text = texts[i].as_py()
                if text and len(text) > 100:  # skip tiny fragments
                    docs.append(text)
                    total_chars += len(text)
                    kept += 1
            if total_scanned % 100_000 == 0:
                print(f"    scanned {total_scanned:,}, kept {kept:,}, "
                      f"{total_chars:,} chars...", end="\r")
            if total_chars >= target_chars:
                break
        if total_chars >= target_chars:
            break

    print(f"\n  Extracted {kept:,} docs from {total_scanned:,} scanned "
          f"({total_chars:,} chars)")
    return docs


def train_tokenizer(docs: list[str], vocab_size: int, output_path: str):
    """Train a BPE tokenizer on document text."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Sample docs for tokenizer training (full corpus may be huge)
    max_train_docs = min(len(docs), 200_000)
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(docs), max_train_docs, replace=False)

    def text_iterator():
        for idx in train_indices:
            yield docs[idx]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Verify special token IDs
    assert tokenizer.token_to_id("<|im_start|>") == 0
    assert tokenizer.token_to_id("<|im_end|>") == 1
    assert tokenizer.token_to_id("<|pad|>") == 2
    assert tokenizer.token_to_id("<|endoftext|>") == EOT_ID

    tokenizer.save(output_path)
    print(f"  Tokenizer saved: {output_path} "
          f"(vocab_size={tokenizer.get_vocab_size()}, "
          f"trained on {max_train_docs:,} docs)")
    return tokenizer


def tokenize_documents(docs: list[str], tokenizer,
                       target_tokens: int) -> tuple[list[int], list[int]]:
    """Tokenize documents into flat token/target arrays.

    Documents separated by <|endoftext|>. All tokens are valid targets
    (no masking — standard LM objective).
    """
    all_tokens = []

    for i, doc in enumerate(docs):
        ids = tokenizer.encode(doc).ids
        all_tokens.extend(ids)
        all_tokens.append(EOT_ID)  # document boundary
        if i % 10_000 == 0:
            print(f"    tokenized {i:,}/{len(docs):,} docs, "
                  f"{len(all_tokens):,} tokens...", end="\r")
        if len(all_tokens) >= target_tokens:
            break

    # Truncate to target
    all_tokens = all_tokens[:target_tokens]
    print(f"\n  Total tokens: {len(all_tokens):,}")

    # Standard next-token prediction: input[i] -> target[i] = input[i+1]
    input_tokens = all_tokens[:-1]
    target_tokens_list = all_tokens[1:]

    return input_tokens, target_tokens_list


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu (score 4+5) for NL-Hecate")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="BPE vocabulary size (default: 32000)")
    parser.add_argument("--output", type=str, default="data/fineweb_edu",
                        help="Output directory")
    parser.add_argument("--source", type=str, default=LOCAL_PARQUET,
                        help="Path to FineWeb-Edu parquet")
    parser.add_argument("--min_score", type=int, default=4,
                        help="Minimum int_score (default: 4)")
    parser.add_argument("--target_tokens", type=int, default=100_000_000,
                        help="Target token count (default: 100M)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to existing tokenizer (skip training)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"

    # Estimate chars needed: ~3 chars/token for BPE on English prose
    target_chars = int(args.target_tokens * 3.5)

    # ── Step 1: Extract documents ────────────────────────────────────
    print(f"Step 1: Extracting score >= {args.min_score} documents...")
    t0 = time.time()
    docs = extract_documents(args.source, args.min_score, target_chars)
    print(f"  Done in {time.time() - t0:.1f}s")

    if len(docs) < 100:
        print("ERROR: Too few documents extracted")
        sys.exit(1)

    # ── Step 2: Train or load tokenizer ──────────────────────────────
    if args.tokenizer and Path(args.tokenizer).exists():
        print(f"\nStep 2: Loading existing tokenizer: {args.tokenizer}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
    elif tokenizer_path.exists():
        print(f"\nStep 2: Loading existing tokenizer: {tokenizer_path}")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print(f"\nStep 2: Training {args.vocab_size} BPE tokenizer...")
        t0 = time.time()
        tokenizer = train_tokenizer(docs, args.vocab_size, str(tokenizer_path))
        print(f"  Trained in {time.time() - t0:.1f}s")

    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}")

    # ── Step 3: Shuffle and split documents ──────────────────────────
    print(f"\nStep 3: Splitting {len(docs):,} documents "
          f"(seed={args.seed}, val_ratio={args.val_ratio})...")
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * args.val_ratio))
    val_indices = set(indices[:n_val])
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_indices]
    val_docs = [docs[i] for i in val_indices]
    print(f"  Train: {len(train_docs):,} docs, Val: {len(val_docs):,} docs")

    # ── Step 4: Tokenize ─────────────────────────────────────────────
    print(f"\nStep 4: Tokenizing (target={args.target_tokens:,} tokens)...")
    t0 = time.time()

    train_target = int(args.target_tokens * (1 - args.val_ratio))
    val_target = args.target_tokens - train_target

    print("  Train split:")
    train_input, train_targets = tokenize_documents(
        train_docs, tokenizer, train_target)
    print("  Val split:")
    val_input, val_targets = tokenize_documents(
        val_docs, tokenizer, val_target)
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    # ── Step 5: Save ─────────────────────────────────────────────────
    print("\nStep 5: Saving output files...")

    train_tokens_arr = np.array(train_input, dtype=np.uint32)
    train_targets_arr = np.array(train_targets, dtype=np.int32)
    val_tokens_arr = np.array(val_input, dtype=np.uint32)
    val_targets_arr = np.array(val_targets, dtype=np.int32)

    np.save(out_dir / "train_tokens.npy", train_tokens_arr)
    np.save(out_dir / "train_targets.npy", train_targets_arr)
    np.save(out_dir / "val_tokens.npy", val_tokens_arr)
    np.save(out_dir / "val_targets.npy", val_targets_arr)

    print(f"  train_tokens.npy: {train_tokens_arr.nbytes / 1e6:.1f} MB "
          f"({len(train_tokens_arr):,} tokens)")
    print(f"  val_tokens.npy: {val_tokens_arr.nbytes / 1e6:.1f} MB "
          f"({len(val_tokens_arr):,} tokens)")

    # Save metadata (same schema as ShareGPT pipeline)
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
            "documents": len(train_docs),
            "total_tokens": len(train_input),
            "valid_targets": len(train_targets),
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "val": {
            "split": "val",
            "documents": len(val_docs),
            "total_tokens": len(val_input),
            "valid_targets": len(val_targets),
            "masked_targets": 0,
            "mask_ratio": 0.0,
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source": "fineweb-edu-dedup (score >= {})".format(args.min_score),
        "min_score": args.min_score,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    total = len(train_input) + len(val_input)
    print(f"\n{'=' * 60}")
    print("FineWeb-Edu data preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:        {out_dir}")
    print(f"  Source:        FineWeb-Edu score >= {args.min_score}")
    print(f"  Vocab:         {actual_vocab:,}")
    print(f"  Total tokens:  {total:,}")
    print(f"  Train tokens:  {len(train_input):,}")
    print(f"  Val tokens:    {len(val_input):,}")
    print(f"  Documents:     {len(train_docs) + len(val_docs):,}")
    print(f"  Mask ratio:    0.0% (all tokens are valid targets)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
