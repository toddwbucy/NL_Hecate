#!/usr/bin/env python3
"""
Prepare FineWeb-Edu (score 4+5) tokenized with Llama-3.2-1B tokenizer (128K vocab).

Re-tokenizes the same raw documents as prepare_fineweb_edu.py using the
Llama 3 tokenizer instead of the custom 32K BPE. Required for the HOPE §7.3
level stacking experiment (SwiGluMlp rule) where the embedding table is
initialized from Llama-3.2-1B and must match vocab_size=128256.

Output: data/fineweb_edu_llama3/
  train.npy   — token IDs (uint32)
  val.npy     — token IDs (uint32)
  meta.json   — {"vocab_size": 128256, "train_tokens": N, "val_tokens": M}

Usage:
    python data/prepare_fineweb_edu_llama3.py
    python data/prepare_fineweb_edu_llama3.py --target_tokens 100_000_000
    python data/prepare_fineweb_edu_llama3.py --output data/fineweb_edu_llama3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LOCAL_PARQUET = "/bulk-store/training-datasets/smollm-corpus/fineweb-edu-dedup/data.parquet"
LLAMA3_MODEL_ID = "unsloth/Llama-3.2-1B"
VOCAB_SIZE = 128256
VAL_FRAC = 0.005  # 0.5% of tokens for validation


def tokenize_docs(
    docs: list[str],
    tokenizer,
    batch_size: int = 512,
) -> np.ndarray:
    """Tokenize a list of documents, append EOS between each."""
    eos_id = tokenizer.eos_token_id
    all_ids: list[int] = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for ids in enc["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(eos_id)
        if i % (batch_size * 10) == 0:
            pct = min(100, 100 * i / max(len(docs), 1))
            print(f"  tokenizing {pct:.0f}%  ({i}/{len(docs)} docs, "
                  f"{len(all_ids):,} tokens)", end="\r", flush=True)
    print()
    return np.array(all_ids, dtype=np.uint32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_tokens", type=int, default=100_000_000,
                        help="Approximate target token count (default: 100M)")
    parser.add_argument("--min_score", type=int, default=4,
                        help="Minimum FineWeb-Edu quality score (default: 4)")
    parser.add_argument("--output", default="data/fineweb_edu_llama3",
                        help="Output directory (default: data/fineweb_edu_llama3)")
    parser.add_argument("--model", default=LLAMA3_MODEL_ID,
                        help=f"HF tokenizer model ID (default: {LLAMA3_MODEL_ID})")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.vocab_size <= VOCAB_SIZE, (
        f"Expected vocab_size <= {VOCAB_SIZE}, got {tokenizer.vocab_size}"
    )

    print(f"Extracting documents from {LOCAL_PARQUET} ...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(LOCAL_PARQUET)
    docs: list[str] = []
    total_chars = 0
    target_chars = args.target_tokens * 5  # rough estimate: 5 chars/token

    for batch in pf.iter_batches(batch_size=10_000, columns=["text", "metadata"]):
        texts = batch.column("text")
        metas = batch.column("metadata")
        for i in range(len(texts)):
            m = metas[i].as_py() if hasattr(metas[i], "as_py") else {}
            score = m.get("int_score", 0) if isinstance(m, dict) else 0
            if score < args.min_score:
                continue
            text = texts[i].as_py()
            docs.append(text)
            total_chars += len(text)
        if total_chars >= target_chars:
            break
    print(f"  Collected {len(docs):,} documents ({total_chars/1e6:.1f}M chars)")

    print("Tokenizing ...")
    t0 = time.time()
    all_ids = tokenize_docs(docs, tokenizer)
    print(f"  {len(all_ids):,} tokens in {time.time()-t0:.1f}s")

    # Split train/val
    val_n = max(1000, int(len(all_ids) * VAL_FRAC))
    val_ids = all_ids[:val_n]
    train_ids = all_ids[val_n:]

    train_path = out_dir / "train.npy"
    val_path   = out_dir / "val.npy"
    meta_path  = out_dir / "meta.json"

    np.save(train_path, train_ids)
    np.save(val_path, val_ids)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "tokenizer": args.model,
        "min_score": args.min_score,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir}/")
    print(f"  train.npy: {len(train_ids):,} tokens")
    print(f"  val.npy:   {len(val_ids):,} tokens")
    print(f"  meta.json: vocab_size={VOCAB_SIZE}")


if __name__ == "__main__":
    main()
