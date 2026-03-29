#!/usr/bin/env python3
"""
Generic HuggingFace corpus preparation for NL-Hecate build runs.

Streams documents from a HuggingFace dataset, tokenizes with the 32K BPE
tokenizer, and writes the standard NL-Hecate data format consumed by
BpeTokenStream in engine/data.py.

Output format (matches existing prepare_fineweb_edu.py convention):
  <output_dir>/
    train_tokens.npy    -- uint32 token IDs (input sequence)
    train_targets.npy   -- uint32 next-token targets (last pos -> vocab_size)
    val_tokens.npy      -- validation split
    val_targets.npy     -- validation split
    meta.json           -- vocab_size, token counts, provenance

Standard build corpus lags are at CMS level periods [1, 8, 64, 512] tokens.
Run lag_mi.py on the output to verify the corpus passes the selection criterion
before scheduling any build runs.

Spec: specs/infrastructure/02_corpus_selection.md
CS-38: 'build' vocabulary, not 'training' or 'train'
CS-37: CMS constructs called 'levels'
CS-47: seed stored in meta.json for reproducibility

Usage:
    # C4 (Miras build corpus — requires --config en)
    python scripts/prepare_corpus.py \\
        --corpus allenai/c4 --config en --split train \\
        --target-tokens 1_000_000_000 \\
        --output data/c4

    # PG-19 (Project Gutenberg books, long-range structure)
    python scripts/prepare_corpus.py \\
        --corpus pg19 --split train \\
        --target-tokens 1_000_000_000 \\
        --output data/pg19

    # Smoke test (10M tokens)
    python scripts/prepare_corpus.py \\
        --corpus allenai/c4 --config en --target-tokens 10_000_000 \\
        --output data/c4_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from array import array as _array
from pathlib import Path

import numpy as np

TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"  # 32K BPE
VOCAB_SIZE = 32000
VAL_FRAC = 0.005   # 0.5% for evaluation split
MIN_DOC_LEN = 64   # skip very short documents (fewer than 64 characters)


def _load_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers not installed. Run: pip install transformers")
    return AutoTokenizer.from_pretrained(tokenizer_id)


def _prepare_targets(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """Shift token IDs by one to produce next-token targets.

    The last position in each document has no next token — use vocab_size
    as the sentinel (BpeTokenStream treats target >= vocab_size as masked).
    Since tokens are a flat stream (not chunked here), the only masked
    position is the very last token.
    """
    if tokens.size == 0:
        return np.empty(0, dtype=np.uint32)
    targets = np.empty_like(tokens)
    targets[:-1] = tokens[1:]
    targets[-1] = vocab_size  # sentinel: no target for final token
    return targets


def stream_and_tokenize(
    corpus: str,
    split: str,
    text_column: str,
    target_tokens: int,
    tokenizer,
    seed: int,
    verbose: bool,
    config: str | None = None,
) -> np.ndarray:
    """Stream corpus documents and return a flat uint32 token array.

    Documents are concatenated in natural order (no shuffle) so that
    inter-document lag structure is preserved. A single EOS token is
    appended between documents as a boundary marker.

    CS-47: does not shuffle — natural order is required for lag-MI.
    Shuffling is deferred to the data loader (chunked random access).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed. Run: pip install datasets")

    eos_id = tokenizer.eos_token_id or 2
    # Use array.array instead of list to avoid OOM at 1B-token scale:
    # list[int] holds Python int objects (~28 bytes each); array.array('I')
    # stores raw 4-byte unsigned ints — ~7× lower peak memory.
    all_tokens: _array = _array("I")
    t0 = time.time()
    doc_count = 0

    load_kwargs: dict = {"split": split, "streaming": True, "trust_remote_code": False}
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(corpus, **load_kwargs)

    for doc in ds:
        text = doc.get(text_column, "")
        if not text or len(text) < MIN_DOC_LEN:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(ids)
        all_tokens.append(eos_id)
        doc_count += 1

        if len(all_tokens) >= target_tokens:
            break

        if verbose and doc_count % 50_000 == 0:
            elapsed = time.time() - t0
            tok_per_s = len(all_tokens) / max(elapsed, 1)
            pct = 100 * len(all_tokens) / target_tokens
            print(
                f"  {pct:5.1f}%  {len(all_tokens):>12,} tokens  "
                f"{doc_count:,} docs  {tok_per_s:,.0f} tok/s",
                end="\r", flush=True,
            )

    if verbose:
        print()

    return np.array(all_tokens[:target_tokens], dtype=np.uint32)


def save_split(
    out_dir: Path,
    prefix: str,
    tokens: np.ndarray,
    vocab_size: int,
) -> None:
    """Write {prefix}_tokens.npy and {prefix}_targets.npy."""
    targets = _prepare_targets(tokens, vocab_size)
    np.save(out_dir / f"{prefix}_tokens.npy", tokens)
    np.save(out_dir / f"{prefix}_targets.npy", targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", required=True, metavar="DATASET_ID",
        help="HuggingFace dataset ID (e.g. 'allenai/c4', 'pg19')",
    )
    parser.add_argument(
        "--config", default=None, metavar="NAME",
        help="Dataset config name if required (e.g. 'en' for allenai/c4)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--text-column", default="text",
        help="Column containing document text (default: 'text')",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=1_000_000_000,
        help="Target token count for the combined dataset (default: 1B)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=VAL_FRAC,
        help=f"Fraction of tokens for evaluation split (default: {VAL_FRAC})",
    )
    parser.add_argument(
        "--tokenizer", default=TOKENIZER_ID,
        help=f"HuggingFace tokenizer ID (default: {TOKENIZER_ID})",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Output directory (created if absent)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed stored in meta.json for CS-47 reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = _load_tokenizer(args.tokenizer)

    if tokenizer.vocab_size != VOCAB_SIZE:
        sys.exit(
            f"Tokenizer vocab mismatch: expected {VOCAB_SIZE}, "
            f"got {tokenizer.vocab_size}. Use the LLaMA tokenizer "
            f"({TOKENIZER_ID}) or update VOCAB_SIZE in this script."
        )
    vocab_size = VOCAB_SIZE

    if verbose:
        print(f"Streaming {args.target_tokens:,} tokens from {args.corpus}/{args.split} ...")
    t0 = time.time()

    all_tokens = stream_and_tokenize(
        corpus=args.corpus,
        split=args.split,
        text_column=args.text_column,
        target_tokens=args.target_tokens,
        tokenizer=tokenizer,
        seed=args.seed,
        verbose=verbose,
        config=args.config,
    )

    elapsed_stream = time.time() - t0
    if verbose:
        print(f"  Streamed {len(all_tokens):,} tokens in {elapsed_stream:.1f}s")

    # --- Split into build and evaluation ---
    val_n = max(10_000, int(len(all_tokens) * args.val_frac))
    val_tokens = all_tokens[:val_n]
    build_tokens = all_tokens[val_n:]

    if verbose:
        print(f"Saving to {out_dir}/ ...")
        print(f"  build: {len(build_tokens):,} tokens")
        print(f"  eval:  {len(val_tokens):,} tokens")

    save_split(out_dir, "train", build_tokens, vocab_size)
    save_split(out_dir, "val", val_tokens, vocab_size)

    meta = {
        "vocab_size": vocab_size,
        "train_tokens": int(len(build_tokens)),
        "val_tokens": int(len(val_tokens)),
        "tokenizer": args.tokenizer,
        "corpus": args.corpus,
        "config": args.config,
        "split": args.split,
        "seed": args.seed,   # CS-47: provenance for reproducibility
        "min_doc_len": MIN_DOC_LEN,
        "elapsed_stream_seconds": round(elapsed_stream, 1),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        total = time.time() - t0
        print(f"Done. Total elapsed: {total:.1f}s")
        print(f"  {out_dir}/train_tokens.npy  ({build_tokens.nbytes / 1e9:.2f} GB)")
        print(f"  {out_dir}/val_tokens.npy    ({val_tokens.nbytes / 1e6:.0f} MB)")
        print(f"  {out_dir}/meta.json")
        print(f"\nNext: verify corpus with lag_mi.py before scheduling build runs.")
        print(f"  python tools/lag_mi.py --npy {out_dir}/train_tokens.npy "
              f"--out results/lag_mi_{Path(args.output).name}.json")


if __name__ == "__main__":
    main()
