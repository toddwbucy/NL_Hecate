#!/usr/bin/env python3
"""
Prepare byte-level training data from local FineWeb-edu-dedup parquet.

Reads from /bulk-store/training-datasets/smollm-corpus/fineweb-edu-dedup/data.parquet,
concatenates text, encodes to UTF-8 bytes, and saves as a flat binary file where each
byte IS a token ID (vocab_size=256).

Usage:
    python data/download_fineweb.py                    # default: 100M bytes
    python data/download_fineweb.py --target_bytes 50_000_000
    python data/download_fineweb.py --output data/fineweb_50m.bin
"""

import argparse
import sys
from pathlib import Path

LOCAL_PARQUET = "/bulk-store/training-datasets/smollm-corpus/fineweb-edu-dedup/data.parquet"


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb byte tokens for NL-Hecate")
    parser.add_argument("--target_bytes", type=int, default=100_000_000,
                        help="Target number of byte tokens (default: 100M)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output binary file path")
    parser.add_argument("--source", type=str, default=LOCAL_PARQUET,
                        help="Path to FineWeb parquet file")
    args = parser.parse_args()

    output = args.output or f"data/fineweb_{args.target_bytes // 1_000_000}m.bin"
    output_path = Path(output)

    if output_path.exists():
        print(f"Output already exists: {output_path} ({output_path.stat().st_size:,} bytes)")
        print("Delete it to re-create, or use --output for a different path.")
        return

    source = Path(args.source)
    if not source.exists():
        print(f"Error: parquet not found at {source}")
        print("Expected: /bulk-store/training-datasets/smollm-corpus/fineweb-edu-dedup/data.parquet")
        sys.exit(1)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: 'pyarrow' required. Install with:")
        print("  pip install pyarrow")
        sys.exit(1)

    print(f"Reading from: {source}")
    print(f"Target: {args.target_bytes:,} byte tokens")

    # Stream row groups to avoid loading 504GB into memory
    pf = pq.ParquetFile(source)
    buf = bytearray()
    rows_read = 0

    for batch in pf.iter_batches(batch_size=10_000, columns=["text"]):
        for text in batch.column("text"):
            t = text.as_py()
            if not t:
                continue
            buf.extend(t.encode("utf-8"))
            rows_read += 1
            if rows_read % 10_000 == 0:
                print(f"  {rows_read:,} docs, {len(buf):,} bytes...", end="\r")
            if len(buf) >= args.target_bytes:
                break
        if len(buf) >= args.target_bytes:
            break

    # Truncate to exact target
    buf = buf[:args.target_bytes]

    # Write as raw bytes (each byte IS a token ID)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(bytes(buf))

    print(f"\nDone: {rows_read:,} docs -> {len(buf):,} byte tokens")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
