# Canonical Artifact Directory Layout

```text
CONTRACT
  Purpose:    Define the canonical filesystem layout for compiled binary artifacts
              produced by the NL-Hecate build system, establish a staleness
              detection convention, and specify the HADES metadata schema for
              artifact provenance tracking including source-to-binary graph edges.
  Expects:    `core/build.rs` compiles CUDA kernels via nvcc and links them into
              the PyO3 .so through maturin. The .so embeds all CUDA kernel object
              code for sm_86/89/90 + PTX. `python/.venv` is the active venv.
              Git is initialized; `git rev-parse HEAD` returns a clean SHA.
              Source code nodes exist in HADES `arxiv_metadata` collection.
  Guarantees: Every promoted artifact has a known git SHA, size, and type
              recorded in `artifacts/.build-meta.json` (filesystem) and in the
              HADES `hecate_artifacts` collection (graph). The artifact node is
              connected to the source code nodes it was compiled from via
              `hecate_artifact_edges`. The mapping from source tree → binary is
              auditable by graph traversal. No artifact is silently stale.
  Cost:       One manual `scripts/promote_artifacts.py` invocation after each
              `maturin develop --release` build, followed by a git commit of the
              promoted binary. ~5 seconds total. No CI changes needed.
  Trade-off:  Binaries ARE committed to git. At 4.1MB the release .so (CUDA
              kernels + PyO3 glue, all archs) is small enough to distribute
              directly through the repository without git-lfs.
  Position:   specs/infrastructure/02_artifact_layout.md
  Source:     NL-Hecate: core/build.rs (nvcc compilation), python/src/lib.rs
              (PyO3 bindings), scripts/promote_artifacts.py (promotion script).
              HADES collections: hecate_artifacts, hecate_artifact_edges.
```

---

## 1. Problem

Three categories of compiled output exist in the project with no coherent home:

| Artifact | Current location | Problem |
|---|---|---|
| PyO3 binding `.so` | `python/.venv/lib/.../nl_hecate.cpython-312-*.so` | Buried in venv, silently stale when Rust source changes |
| Maturin `.whl` | `python/target/wheels/` (ephemeral) | Not staged, not versioned |
| CUDA fat binary | Inside the `.so` (linked at maturin build time) | No standalone cache; every GPU node must rebuild from source |

The `hecate_artifacts` HADES collection was pre-created to solve provenance tracking
but had no writes, no schema, no edge collection, and no filesystem convention.

---

## 2. Directory Layout

```text
artifacts/                          ← binary distribution directory
├── .gitignore                      ← ignores .build-meta.json only
├── .build-meta.json                ← gitignored; written by promote_artifacts.py
├── README.md                       ← rebuild instructions + HADES queries
├── cuda/                           ← reserved for standalone CUDA fat binaries (future)
├── so/                             ← SHA-tagged PyO3 .so files (committed to git)
│   └── nl_hecate.cpython-312-x86_64-linux-gnu.<sha>.so
└── wheels/                         ← version-tagged maturin .whl files (committed to git)
    └── nl_hecate-<version>-cp312-linux_x86_64.whl
```

`artifacts/cuda/` is reserved for standalone CUDA fat binaries if the build is
ever split to produce a distributable `.fatbin` separately from the `.so`. For
the current single-machine workflow, the CUDA kernels are embedded inside the
`.so` and `cuda/` remains empty.

---

## 3. Git Strategy

**Decision: commit binaries to git; the repository is the distribution channel.**

- `artifacts/so/*.so` and `artifacts/wheels/*.whl` are **committed to git**
- `artifacts/.build-meta.json` is gitignored (machine-local staleness pointer)
- `artifacts/README.md`, `.gitignore`, and subdirectory stubs are committed
- Rationale: the 4.1MB release `.so` embeds all CUDA kernels (sm_86/89/90 + PTX)
  and PyO3 glue. This is small enough to commit directly — no git-lfs required.
  Consumers clone the repo and use the pre-built binary immediately without
  needing nvcc, Rust, or maturin installed.

---

## 4. Build Integration

**Decision: manual promote script; NOT wired into `build.rs`.**

`core/build.rs` is a Cargo build script with a specific contract: it compiles CUDA
kernels and configures linker flags. Adding filesystem side-effects (copying to
`artifacts/`) would violate that contract and make incremental builds less
predictable. Promotion is a deliberate post-build action by the developer.

Workflow:
```bash
maturin develop --release                          # builds .so into venv
python scripts/promote_artifacts.py --task-key task_87a521  # stage + HADES JSON
git add artifacts/so/ && git commit -m "chore: promote .so <sha>"
```

---

## 5. Staleness Detection

`artifacts/.build-meta.json` is written at promote time and is gitignored:

```json
{
  "git_sha": "e001002...",
  "promoted_at_epoch": 1740000000,
  "artifact_type": "pyo3_so",
  "path": "artifacts/so/nl_hecate.cpython-312-x86_64-linux-gnu.e001002.so",
  "size_bytes": 4300000,
  "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"]
}
```

`hecate.py` should read `artifacts/.build-meta.json` at startup and compare the
recorded `git_sha` against `git rev-parse HEAD`. If they differ, emit a warning:

```text
WARNING: artifacts/.build-meta.json SHA e001002 != current HEAD abc1234.
         Run `maturin develop --release && python scripts/promote_artifacts.py`
         to rebuild.
```

Implementation of this startup check in `hecate.py` is deferred as a follow-on.

---

## 6. HADES Schema

### `hecate_artifacts` — metadata node (no binary content)

Every promoted artifact writes one document to `hecate_artifacts`:

```json
{
  "_key": "nl-hecate-so-<short-sha>",
  "artifact_type": "pyo3_so",
  "path": "artifacts/so/nl_hecate.cpython-312-x86_64-linux-gnu.<sha>.so",
  "git_sha": "<full 40-char sha>",
  "size_bytes": 4300000,
  "cuda_archs": ["sm_86", "sm_89", "sm_90", "compute_86"],
  "produced_by_task": "persephone_tasks/<task_key>",
  "build_date_epoch": 1740000000,
  "notes": "Release build, all CUDA kernels embedded"
}
```

`artifact_type` values: `pyo3_so`, `wheel`, `cuda_fatbin`.

### `hecate_artifact_edges` — source provenance edges

Each artifact node is connected to the `arxiv_metadata` source code nodes that
were compiled into it:

```json
{
  "_from": "hecate_artifacts/nl-hecate-so-<sha>",
  "_to": "arxiv_metadata/<source-key>",
  "rel": "compiled_from"
}
```

Source nodes for the PyO3 `.so` (Rust + CUDA files compiled by maturin):
`gpu-forward-rs`, `cuda-graph-rs`, `cuda-ffi-rs`, `gpu-params-rs`,
`gpu-buf-rs`, `core-lib-rs`, `elementwise-cu`, `python-lib-rs`.

Python engine files (`evaluation-py`, `loop-py`) and test files are NOT compiled
into the release binary and do not receive `compiled_from` edges.

The promote script prints the edge JSON for manual insertion. It does not call
HADES directly — avoiding a hard dependency on the HADES CLI in the build path.

---

## 7. `promote_artifacts.py` Contract

See `scripts/promote_artifacts.py`.

The script:
1. Resolves the current `.so` path from the active venv
2. Gets `git rev-parse HEAD`
3. Copies `.so` to `artifacts/so/<basename>.<short-sha>.so`
4. Writes `artifacts/.build-meta.json` (gitignored)
5. Prints the `hecate_artifacts` document JSON and all `hecate_artifact_edges`
   edge JSON for manual HADES insertion
