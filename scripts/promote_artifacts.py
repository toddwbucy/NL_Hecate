#!/usr/bin/env python3
"""
promote_artifacts.py — copy the built PyO3 .so into artifacts/so/ and write
artifacts/.build-meta.json. Prints the hecate_artifacts HADES JSON for manual
insertion.

Usage:
    python scripts/promote_artifacts.py [--venv VENV_DIR]

Default venv: python/.venv
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SO_DIR = ARTIFACTS_DIR / "so"
WHEEL_DIR = ARTIFACTS_DIR / "wheels"
META_FILE = ARTIFACTS_DIR / ".build-meta.json"


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def find_so(venv: Path) -> Path:
    pattern = str(venv / "lib" / "python3*" / "site-packages" / "nl_hecate" / "nl_hecate*.so")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"ERROR: no nl_hecate*.so found under {venv}\n"
                 "       Run `maturin develop --release` first.")
    return Path(matches[0])


def find_wheel() -> Path:
    pattern = str(REPO_ROOT / "python" / "target" / "wheels" / "nl_hecate*.whl")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit("ERROR: no .whl found under python/target/wheels/\n"
                 "       Run `maturin build --release` first.")
    return max(Path(m) for m in matches)  # newest by name


def promote_so(venv: Path, sha: str) -> dict:
    src = find_so(venv)
    short = sha[:8]
    # e.g. nl_hecate.cpython-312-x86_64-linux-gnu.e001002.so
    stem = src.stem  # nl_hecate.cpython-312-x86_64-linux-gnu
    dest_name = f"{stem}.{short}.so"
    dest = SO_DIR / dest_name
    SO_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    print(f"  promoted .so → {dest.relative_to(REPO_ROOT)}  ({size // 1024}K)")
    return {
        "_key": f"nl-hecate-so-{short}",
        "artifact_type": "pyo3_so",
        "path": str(dest.relative_to(REPO_ROOT)),
        "git_sha": sha,
        "size_bytes": size,
        "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"],
        "produced_by_task": None,
        "build_date_epoch": int(time.time()),
        "notes": "Release build via maturin develop --release; all CUDA kernels embedded",
    }


def promote_wheel(sha: str) -> dict:
    src = find_wheel()
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = WHEEL_DIR / src.name
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    print(f"  promoted .whl → {dest.relative_to(REPO_ROOT)}  ({size // 1024}K)")
    short = sha[:8]
    return {
        "_key": f"nl-hecate-whl-{short}",
        "artifact_type": "wheel",
        "path": str(dest.relative_to(REPO_ROOT)),
        "git_sha": sha,
        "size_bytes": size,
        "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"],
        "produced_by_task": None,
        "build_date_epoch": int(time.time()),
        "notes": "Release wheel via maturin build --release",
    }


def write_meta(sha: str, artifact: dict) -> None:
    meta = {
        "git_sha": sha,
        "promoted_at_epoch": artifact["build_date_epoch"],
        "artifact_type": artifact["artifact_type"],
        "path": artifact["path"],
        "size_bytes": artifact["size_bytes"],
        "cuda_archs": artifact["cuda_archs"],
    }
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {META_FILE.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=str(REPO_ROOT / "python" / ".venv"),
                        help="Path to the active venv (default: python/.venv)")
    parser.add_argument("--wheel", action="store_true",
                        help="Promote a .whl from python/target/wheels/ instead of venv .so")
    args = parser.parse_args()

    sha = git_sha()
    print(f"git SHA: {sha[:8]}")

    if args.wheel:
        artifact = promote_wheel(sha)
    else:
        artifact = promote_so(Path(args.venv), sha)

    write_meta(sha, artifact)

    print()
    print("── HADES hecate_artifacts entry ────────────────────────────────")
    print("Insert this into the NL database to register artifact provenance:")
    print()
    print(f"  hades --database NL db insert --collection hecate_artifacts \\")
    print(f"    '{json.dumps(artifact)}'")
    print()
    print("Or via MCP:")
    print(f"  hades_db_insert(collection='hecate_artifacts', database='NL',")
    print(f"    data='{json.dumps(artifact)}')")


if __name__ == "__main__":
    main()
