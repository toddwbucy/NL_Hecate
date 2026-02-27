#!/usr/bin/env python3
"""
Prepare shakedown corpus: TinyStories (simple) → Cosmopedia (complex).

Produces:
  corpus.bin      — flat bytes, all docs concatenated, no delimiters
  doc_starts.npy  — uint64[N] byte offset where each document begins
  meta.json       — summary stats

Documents are sorted by complexity (short/simple first) so the model
sees easy grammar before harder content. No special separator tokens —
the boundary array is the sole signal for document resets.

Usage:
    python data/prepare_shakedown.py --output data/shakedown_1b
    python data/prepare_shakedown.py \
        --tinystories /path/to/stories_raw.txt \
        --cosmopedia /path/to/data.parquet \
        --output data/shakedown_1b \
        --tinystories_bytes 300000000 \
        --cosmopedia_bytes 700000000
"""

import argparse
import json
import os
import re
import time

import numpy as np


# ── Audience complexity ranking ──────────────────────────────────────
# Typos (studnets) match the actual Cosmopedia data.
AUDIENCE_RANK = {
    "young_children": 0,
    "children": 1,
    "middle_school_students": 2,
    "high_school_studnets": 3,
    "general": 4,
    "college_students": 5,
    "college_studnets": 5,
    "researchers": 6,
    "alien": 4,
    "requires_details": 4,
}


def extract_tinystories(path: str, max_bytes: int) -> list[bytes]:
    """Read TinyStories, split on delimiters, sort by length (shortest first)."""
    print(f"Reading TinyStories from {path}...")
    t0 = time.perf_counter()
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    print(f"  Read {len(raw):,} chars in {time.perf_counter() - t0:.1f}s")

    # Split on === Story N === delimiters
    stories = re.split(r"^=== Story \d+ ===$", raw, flags=re.MULTILINE)
    # Filter empty/whitespace-only
    stories = [s.strip() for s in stories if s.strip()]
    print(f"  Found {len(stories):,} stories")

    # Sort by byte length (shortest = simplest grammar first)
    stories.sort(key=lambda s: len(s.encode("utf-8")))

    # Take up to max_bytes worth
    docs = []
    total = 0
    for s in stories:
        b = s.encode("utf-8")
        if total + len(b) > max_bytes:
            break
        docs.append(b)
        total += len(b)

    print(f"  Selected {len(docs):,} stories, {total:,} bytes")
    return docs


def extract_cosmopedia(path: str, max_bytes: int) -> list[bytes]:
    """Two-pass Cosmopedia extraction to avoid loading 112GB of text into RAM.

    Pass 1: Read only audience + token_length metadata (~700MB for 36M rows).
            Sort by (audience_rank, token_length), select rows that fit budget.
    Pass 2: Read text column ONLY for selected rows, one row group at a time.
    """
    import pyarrow.parquet as pq

    print(f"Reading Cosmopedia from {path}...")
    t0 = time.perf_counter()

    pf = pq.ParquetFile(path)
    num_groups = pf.metadata.num_row_groups

    # ── Pass 1: metadata only (audience + token_length) ──────────────
    print("  Pass 1: reading metadata (audience, token_length)...")
    # Store (audience_rank, token_length, row_group_idx, row_within_group)
    meta_entries = []
    for gi in range(num_groups):
        table = pf.read_row_group(gi, columns=["audience", "token_length"])
        audiences = table.column("audience").to_pylist()
        token_lengths = table.column("token_length").to_pylist()

        for ri, (aud, tlen) in enumerate(zip(audiences, token_lengths)):
            rank = AUDIENCE_RANK.get(aud, 4)
            meta_entries.append((rank, tlen or 0, gi, ri))

        if (gi + 1) % 200 == 0:
            print(f"    Row group {gi + 1}/{num_groups}, "
                  f"{len(meta_entries):,} entries...")

    print(f"  Pass 1 done: {len(meta_entries):,} entries in "
          f"{time.perf_counter() - t0:.1f}s")

    # Sort by (audience_rank, token_length)
    meta_entries.sort(key=lambda e: (e[0], e[1]))

    # Select rows within byte budget. Estimate ~4 bytes/token for UTF-8.
    BYTES_PER_TOKEN = 4
    selected = set()  # (row_group_idx, row_within_group)
    estimated_bytes = 0
    for rank, tlen, gi, ri in meta_entries:
        est = tlen * BYTES_PER_TOKEN
        if estimated_bytes + est > max_bytes:
            break
        selected.add((gi, ri))
        estimated_bytes += est

    print(f"  Selected {len(selected):,} rows "
          f"(~{estimated_bytes:,} estimated bytes)")

    # Group selected rows by row_group for efficient pass 2
    selected_by_group: dict[int, list[int]] = {}
    for gi, ri in selected:
        selected_by_group.setdefault(gi, []).append(ri)

    # Build ordering: we need to emit docs in sort order, so track
    # (sort_position) → (gi, ri) mapping
    order_map = {}  # (gi, ri) → sort_position
    for pos, (rank, tlen, gi, ri) in enumerate(meta_entries):
        if (gi, ri) in selected:
            order_map[(gi, ri)] = pos

    # Free metadata
    del meta_entries

    # ── Pass 2: read text only for selected rows ─────────────────────
    print("  Pass 2: reading text for selected rows...")
    t1 = time.perf_counter()

    # Collect (sort_position, bytes) then sort
    collected: list[tuple[int, bytes]] = []
    total_bytes = 0
    groups_with_data = sorted(selected_by_group.keys())

    for gi in groups_with_data:
        row_indices = selected_by_group[gi]
        table = pf.read_row_group(gi, columns=["text"])
        texts = table.column("text").to_pylist()

        for ri in row_indices:
            text = texts[ri]
            if text is None:
                continue
            b = text.encode("utf-8")
            sort_pos = order_map[(gi, ri)]
            collected.append((sort_pos, b))
            total_bytes += len(b)

        # Free the text column immediately
        del table, texts

        if len(collected) % 50000 < len(row_indices):
            print(f"    Collected {len(collected):,} docs, "
                  f"{total_bytes:,} bytes so far...")

    print(f"  Pass 2 done: {len(collected):,} docs, {total_bytes:,} bytes "
          f"in {time.perf_counter() - t1:.1f}s")

    # Sort by original complexity order
    collected.sort(key=lambda x: x[0])

    # Trim to exact byte budget (estimate was approximate)
    docs = []
    total = 0
    for _, b in collected:
        if total + len(b) > max_bytes:
            break
        docs.append(b)
        total += len(b)

    del collected

    print(f"  Final: {len(docs):,} documents, {total:,} bytes")
    return docs


def write_split(docs: list[bytes], prefix: str, output_dir: str) -> dict:
    """Write {prefix}_corpus.bin + {prefix}_doc_starts.npy for a split."""
    corpus_path = os.path.join(output_dir, f"{prefix}_corpus.bin")
    starts_path = os.path.join(output_dir, f"{prefix}_doc_starts.npy")

    doc_starts = []
    offset = 0
    with open(corpus_path, "wb") as f:
        for doc in docs:
            doc_starts.append(offset)
            f.write(doc)
            offset += len(doc)

    total_bytes = offset
    doc_starts_arr = np.array(doc_starts, dtype=np.uint64)
    np.save(starts_path, doc_starts_arr)

    return {"num_docs": len(docs), "total_bytes": total_bytes}


def write_corpus(docs: list[bytes], output_dir: str, val_every: int = 100):
    """Split docs into train/val (every Nth doc → val), write both.

    Stratified split: every val_every-th document goes to val.
    This preserves the complexity distribution in both splits since
    the docs are sorted by complexity (TinyStories short→long,
    then Cosmopedia young_children→researchers).

    Outputs:
      train_corpus.bin / train_doc_starts.npy  — training data
      val_corpus.bin   / val_doc_starts.npy    — validation data
      corpus.bin       / doc_starts.npy        — full corpus (for compat)
      meta.json        — summary stats
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Splitting {len(docs):,} documents (every {val_every}th → val)...")
    t0 = time.perf_counter()

    train_docs = []
    val_docs = []
    for i, doc in enumerate(docs):
        if i % val_every == 0:
            val_docs.append(doc)
        else:
            train_docs.append(doc)

    print(f"  Train: {len(train_docs):,} docs, "
          f"{sum(len(d) for d in train_docs):,} bytes")
    print(f"  Val:   {len(val_docs):,} docs, "
          f"{sum(len(d) for d in val_docs):,} bytes")

    # Write train split
    train_info = write_split(train_docs, "train", output_dir)
    # Write val split
    val_info = write_split(val_docs, "val", output_dir)
    # Write train as the main corpus too (VecStream trains on this)
    # Symlink for backward compat: corpus.bin → train_corpus.bin
    corpus_link = os.path.join(output_dir, "corpus.bin")
    starts_link = os.path.join(output_dir, "doc_starts.npy")
    for link in (corpus_link, starts_link):
        if os.path.exists(link):
            os.remove(link)
    os.symlink("train_corpus.bin", corpus_link)
    os.symlink("train_doc_starts.npy", starts_link)

    total_bytes = train_info["total_bytes"] + val_info["total_bytes"]
    meta = {
        "num_docs": len(docs),
        "total_bytes": total_bytes,
        "train_docs": train_info["num_docs"],
        "train_bytes": train_info["total_bytes"],
        "val_docs": val_info["num_docs"],
        "val_bytes": val_info["total_bytes"],
        "val_every": val_every,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"  Wrote all splits in {elapsed:.1f}s")
    print(f"  meta.json: {meta}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Prepare shakedown corpus")
    parser.add_argument(
        "--tinystories",
        default="/bulk-store/training-datasets/natural_language/tinystories/stories_raw.txt",
        help="Path to TinyStories stories_raw.txt",
    )
    parser.add_argument(
        "--cosmopedia",
        default="/bulk-store/training-datasets/smollm-corpus/cosmopedia-v2/data.parquet",
        help="Path to Cosmopedia data.parquet",
    )
    parser.add_argument(
        "--output",
        default="data/shakedown_1b",
        help="Output directory",
    )
    parser.add_argument(
        "--tinystories_bytes",
        type=int,
        default=300_000_000,
        help="Max bytes from TinyStories (default: 300M)",
    )
    parser.add_argument(
        "--cosmopedia_bytes",
        type=int,
        default=700_000_000,
        help="Max bytes from Cosmopedia (default: 700M)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Shakedown Corpus Preparation")
    print("=" * 60)

    # Extract TinyStories (sorted by length)
    ts_docs = extract_tinystories(args.tinystories, args.tinystories_bytes)

    # Extract Cosmopedia (sorted by audience rank + token length)
    cos_docs = extract_cosmopedia(args.cosmopedia, args.cosmopedia_bytes)

    # Concatenate: TinyStories first (simple), then Cosmopedia (complex)
    all_docs = ts_docs + cos_docs

    ts_bytes = sum(len(d) for d in ts_docs)
    cos_bytes = sum(len(d) for d in cos_docs)

    print(f"\nCombined: {len(all_docs):,} documents")
    print(f"  TinyStories: {len(ts_docs):,} docs, {ts_bytes:,} bytes")
    print(f"  Cosmopedia:  {len(cos_docs):,} docs, {cos_bytes:,} bytes")
    print(f"  Total:       {ts_bytes + cos_bytes:,} bytes")

    # Write output
    meta = write_corpus(all_docs, args.output)
    meta["tinystories_docs"] = len(ts_docs)
    meta["tinystories_bytes"] = ts_bytes
    meta["cosmopedia_docs"] = len(cos_docs)
    meta["cosmopedia_bytes"] = cos_bytes

    # Update meta with source info
    meta_path = os.path.join(args.output, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Output in {args.output}/")


if __name__ == "__main__":
    main()
