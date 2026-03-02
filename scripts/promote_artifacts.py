#!/usr/bin/env python3
"""
promote_artifacts.py — copy the built PyO3 .so into artifacts/so/ and write
artifacts/.build-meta.json. Prints hecate_artifacts and hecate_artifact_edges
JSON for manual HADES insertion.

Usage:
    python scripts/promote_artifacts.py [--venv VENV_DIR] [--task-key TASK_KEY]
    python scripts/promote_artifacts.py --wheel [--task-key TASK_KEY]

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

# Source code nodes compiled into the release .so (Rust + CUDA only).
# Python engine files and test files are not compiled into the binary.
COMPILED_FROM_SOURCES = [
    "gpu-forward-rs",
    "cuda-graph-rs",
    "cuda-ffi-rs",
    "gpu-params-rs",
    "gpu-buf-rs",
    "core-lib-rs",
    "elementwise-cu",
    "python-lib-rs",
]


def git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        sys.exit("ERROR: git executable not found on PATH")
    return subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()


def find_so(venv: Path) -> Path:
    pattern = str(venv / "lib" / "python3*" / "site-packages" / "nl_hecate" / "nl_hecate*.so")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit(f"ERROR: no nl_hecate*.so found under {venv}\n"
                 "       Run `maturin develop --release` first.")
    return max((Path(m) for m in matches), key=lambda p: p.stat().st_mtime)


def find_wheel() -> Path:
    pattern = str(REPO_ROOT / "python" / "target" / "wheels" / "nl_hecate*.whl")
    matches = glob.glob(pattern)
    if not matches:
        sys.exit("ERROR: no .whl found under python/target/wheels/\n"
                 "       Run `maturin build --release` first.")
    return max((Path(m) for m in matches), key=lambda p: p.stat().st_mtime)


def promote_so(venv: Path, sha: str, produced_by_task: str | None) -> dict:
    src = find_so(venv)
    short = sha[:8]
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
        "build_hash": sha,
        "size_bytes": size,
        "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"],
        "produced_by_task": produced_by_task,
        "created_at": f"epoch:{int(time.time())}",
        "notes": "Release build via maturin develop --release; all CUDA kernels embedded",
    }


def promote_wheel(sha: str, produced_by_task: str | None) -> dict:
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
        "build_hash": sha,
        "size_bytes": size,
        "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"],
        "produced_by_task": produced_by_task,
        "created_at": f"epoch:{int(time.time())}",
        "notes": "Release wheel via maturin build --release",
    }


def write_meta(sha: str, artifact: dict) -> None:
    # Uses git_sha (not build_hash) — the staleness check in hecate.py reads
    # meta['git_sha'] to compare against `git rev-parse HEAD`.
    meta = {
        "git_sha": sha,
        "promoted_at_epoch": int(artifact["created_at"].split(":")[1]),
        "artifact_type": artifact["artifact_type"],
        "path": artifact["path"],
        "size_bytes": artifact["size_bytes"],
        "cuda_archs": artifact["cuda_archs"],
    }
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {META_FILE.relative_to(REPO_ROOT)}")


def build_edges(artifact: dict) -> list[dict]:
    """Return compiled_from edge dicts for all source nodes."""
    artifact_id = f"hecate_artifacts/{artifact['_key']}"
    return [
        {
            "_from": artifact_id,
            "_to": f"arxiv_metadata/{src}",
            "rel": "compiled_from",
        }
        for src in COMPILED_FROM_SOURCES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", default=str(REPO_ROOT / "python" / ".venv"),
                        help="Path to the active venv (default: python/.venv)")
    parser.add_argument("--wheel", action="store_true",
                        help="Promote a .whl from python/target/wheels/ instead of venv .so")
    parser.add_argument("--task-key", default=None,
                        help="Persephone task key, e.g. task_87a521 (optional)")
    args = parser.parse_args()

    produced_by_task = f"persephone_tasks/{args.task_key}" if args.task_key else None

    sha = git_sha()
    print(f"git SHA: {sha[:8]}")

    if args.wheel:
        artifact = promote_wheel(sha, produced_by_task)
    else:
        artifact = promote_so(Path(args.venv), sha, produced_by_task)

    write_meta(sha, artifact)
    edges = build_edges(artifact)

    # Derive extension for commit hint from the promoted artifact path.
    ext = Path(artifact["path"]).suffix  # .so or .whl

    print()
    print("── HADES hecate_artifacts node ─────────────────────────────────")
    print("Insert into NL database:")
    print()
    print("  hades --database NL db aql \"INSERT")
    print(f"    {json.dumps(artifact)}")
    print("    INTO hecate_artifacts\"")
    print()
    print("── HADES hecate_artifact_edges (compiled_from) ─────────────────")
    print("Insert each edge:")
    print()
    for edge in edges:
        print(f"  hades --database NL db aql \"INSERT {json.dumps(edge)} INTO hecate_artifact_edges\"")
    print()
    print("── Next step ───────────────────────────────────────────────────")
    print("Commit the promoted binary:")
    print(f"  git add artifacts/so/ artifacts/wheels/")
    print(f"  git commit -m \"chore: promote nl_hecate {ext} {sha[:8]}\"")


if __name__ == "__main__":
    main()
