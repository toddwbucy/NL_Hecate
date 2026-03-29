#!/usr/bin/env python3
"""Convert existing .json checkpoints to .safetensors binary format.

Usage:
    python scripts/convert_checkpoint.py checkpoints/model.json
    python scripts/convert_checkpoint.py checkpoints/model.json --out checkpoints/model.safetensors
    python scripts/convert_checkpoint.py checkpoints/  # convert all .json in directory

Reduces file size by ~3x and load time from ~60s to <1s for 433M param models.
The original .json file is preserved (not deleted).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import nl_hecate


def convert_one(src: Path, dst: Path) -> None:
    print(f"Loading  {src}  ({src.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    t0 = time.time()
    try:
        params, cfg, build_state = nl_hecate.load_build_checkpoint(str(src))
        has_build_state = build_state is not None
    except Exception as e:
        print(f"  (load_build_checkpoint failed: {e}, retrying as params-only)", flush=True)
        params, cfg = nl_hecate.load_checkpoint(str(src))
        has_build_state = False
        build_state = None
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s  (d_model={cfg.d_model}, k={cfg.k})", flush=True)

    print(f"Writing  {dst} ...", flush=True)
    t0 = time.time()
    if has_build_state:
        # WARNING: build_state (conductor position, stream cursor, context) cannot be
        # round-tripped through Python — the Rust binding exposes no save_build_checkpoint.
        # The converted checkpoint contains params + config only; resuming from it
        # will reset the training cursor to step 0.
        print("  WARNING: build_state not preserved in conversion — converted checkpoint is params-only")
        nl_hecate.save_checkpoint(str(dst), params, cfg)
    else:
        nl_hecate.save_checkpoint(str(dst), params, cfg)
    save_time = time.time() - t0
    dst_mb = dst.stat().st_size / 1e6
    print(f"  Saved  {dst_mb:.0f} MB in {save_time:.1f}s  ({src.stat().st_size / dst.stat().st_size:.1f}x smaller)")

    # Copy cursor sidecar if it exists
    import shutil
    sidecar_candidate = Path(str(src) + ".cursor.json")
    if sidecar_candidate.exists():
        sidecar_dst = Path(str(dst) + ".cursor.json")
        shutil.copy2(sidecar_candidate, sidecar_dst)
        print(f"  Copied cursor sidecar → {sidecar_dst.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Source .json checkpoint or directory")
    parser.add_argument("--out", help="Output .safetensors path (default: same stem)")
    args = parser.parse_args()

    src = Path(args.src)

    if src.is_dir():
        # Convert all .json checkpoints in directory
        jsons = sorted(src.glob("*.json"))
        if not jsons:
            print(f"No .json files found in {src}")
            return
        print(f"Converting {len(jsons)} checkpoint(s) in {src}/")
        for j in jsons:
            if j.suffix == ".json" and not j.name.endswith(".cursor.json"):
                dst = j.with_suffix(".safetensors")
                if dst.exists():
                    print(f"  Skipping {j.name} (destination exists)")
                    continue
                convert_one(j, dst)
        print("Done.")
    else:
        if not src.exists():
            print(f"Error: {src} not found")
            sys.exit(1)
        if args.out:
            dst = Path(args.out)
        else:
            dst = src.with_suffix(".safetensors")
        if dst.exists():
            print(f"Warning: {dst} already exists, overwriting")
        convert_one(src, dst)
        print("Done.")


if __name__ == "__main__":
    main()
