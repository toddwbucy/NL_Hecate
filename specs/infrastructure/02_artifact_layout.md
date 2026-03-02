# Canonical Artifact Directory Layout

```text
CONTRACT
  Purpose:    Define the canonical filesystem layout for compiled binary artifacts
              produced by the NL-Hecate build system, establish a staleness
              detection convention, and specify the HADES metadata schema for
              artifact provenance tracking.
  Expects:    `core/build.rs` compiles CUDA kernels via nvcc and links them into
              the PyO3 .so through maturin. The .so embeds all CUDA kernel object
              code for sm_86/89/90 + PTX. `python/.venv` is the active venv.
              Git is initialized; `git rev-parse HEAD` returns a clean SHA.
  Guarantees: Every promoted artifact has a known git SHA, size, and type
              recorded in `artifacts/.build-meta.json` (filesystem) and in the
              HADES `hecate_artifacts` collection (graph). The mapping from source
              tree → binary is auditable. No artifact is silently stale.
  Cost:       One manual `scripts/promote_artifacts.py` invocation after each
              `maturin develop --release` build. ~1 second. No CI changes needed.
  Trade-off:  Binaries are NOT committed to git (too large; gitignored). They live
              in `artifacts/` as a local cache. GitHub Releases is the distribution
              channel for versioned builds. The 4.1MB release .so is well under
              GitHub's 2GB release asset limit — no git-lfs needed.
  Position:   specs/infrastructure/02_artifact_layout.md
  Source:     NL-Hecate: core/build.rs (nvcc compilation), python/src/lib.rs
              (PyO3 bindings), scripts/promote_artifacts.py (promotion script).
              HADES collection: hecate_artifacts.
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
but had no writes, no schema, and no filesystem convention backing it.

---

## 2. Directory Layout

```text
artifacts/                          ← gitignored root for compiled outputs
├── .gitignore                      ← ignores *.so, *.whl, *.fatbin, *.build-meta.json
├── .build-meta.json                ← written by promote_artifacts.py at build time
├── README.md                       ← rebuild instructions + HADES link
├── cuda/                           ← placeholder for future standalone fat binaries
├── so/                             ← promoted .so files (SHA-tagged)
│   └── nl_hecate.cpython-312-x86_64-linux-gnu.<sha>.so
└── wheels/                         ← promoted .whl files (version-tagged)
    └── nl_hecate-<version>-cp312-linux_x86_64.whl
```

`artifacts/cuda/` is reserved for standalone CUDA fat binaries if the build is
ever split to produce a distributable `.fatbin` separately from the `.so`. For
the current single-machine workflow, the CUDA kernels are embedded inside the
`.so` and `cuda/` remains empty.

---

## 3. Git Strategy

**Decision: gitignore binaries; GitHub Releases for distribution.**

- `artifacts/*.so`, `artifacts/*.whl`, `artifacts/*.fatbin` are gitignored
- `artifacts/.build-meta.json` is gitignored (machine-local)
- `artifacts/README.md` and subdirectory stubs ARE committed (track the layout, not the binaries)
- Distribution: versioned builds published as GitHub Release assets via `gh release`
- Rationale: the 4.1MB release `.so` (CUDA kernels + PyO3 glue, all archs) is well
  under GitHub's 2GB per-asset limit. No git-lfs required. Binary content is fully
  reproducible from source + `scripts/promote_artifacts.py`.

---

## 4. Build Integration

**Decision: manual promote script; NOT wired into `build.rs`.**

`core/build.rs` is a Cargo build script with a specific contract: it compiles CUDA
kernels and configures linker flags. Adding filesystem side-effects (copying to
`artifacts/`) would violate that contract and make incremental builds less
predictable. Promotion is a deliberate post-build action by the developer.

Workflow:
```bash
maturin develop --release          # builds .so into venv (as usual)
python scripts/promote_artifacts.py  # promotes .so to artifacts/so/, writes .build-meta.json
```

---

## 5. Staleness Detection

`artifacts/.build-meta.json` is written at promote time:

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

```
WARNING: artifacts/.build-meta.json SHA e001002 != current HEAD abc1234.
         Run `maturin develop --release && python scripts/promote_artifacts.py`
         to rebuild.
```

Implementation of this startup check in `hecate.py` is deferred as a follow-on
(acceptance criterion 6 of task_87a521 requires only that the strategy is documented).

---

## 6. HADES `hecate_artifacts` Schema

Every promoted artifact writes one document to the `hecate_artifacts` collection:

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

The promote script writes this entry automatically via the `hades` CLI if
available; otherwise it prints the JSON for manual insertion.

---

## 7. `promote_artifacts.py` Contract

See `scripts/promote_artifacts.py`.

The script:
1. Resolves the current `.so` path from the active venv
2. Gets `git rev-parse HEAD`
3. Copies `.so` to `artifacts/so/<basename>.<short-sha>.so`
4. Writes `artifacts/.build-meta.json`
5. Prints the `hecate_artifacts` JSON for HADES insertion (does not call HADES
   directly — avoids a hard dependency on the hades CLI in the build path)
