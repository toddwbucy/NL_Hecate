## CONTRACT

**Purpose**: Define the canonical layout of the `python/` directory and govern how source artifacts
are organized across scripts, data, runs, and engine modules.

**Expects**:
- `python/` is the working root for all Python-tier operations
- `hecate.py` is the single entry point for all user-facing build/chat/generate commands
- `engine/` contains all reusable Python-tier orchestration modules (no math, CS-18)
- `configs/` contains JSON configuration files consumed by `BuildConfig.from_file()`
- `scripts/` exists as the designated home for one-shot operational scripts (data prep,
  profiling, research exploration)
- `runs/` exists as the single artifact sink for JSONL metrics logs and `.log` build logs
- `data/` contains raw and preprocessed token corpora and data subdirectories; it does NOT
  contain executable scripts

**Guarantees**:
- No data loss: all file moves are tracked by git; no files are deleted except explicitly
  deprecated backward-compat stubs (`build.py`, `serve.py`) that already exit with an error
- All tests pass after reorganization: no import path depends on `data/prepare_*` location,
  and tests reference only `engine/` modules
- Single entry point preserved: `hecate.py` at `python/` root remains the canonical CLI;
  no other script duplicates its function
- Log artifacts consolidated: `logs/` is merged into `runs/` so there is exactly one
  artifact sink directory; configs that previously referenced `runs/` are unchanged
- Config JSON files are not modified: existing `log_file` paths already point to `runs/`,
  so no config surgery is required

**Cost**:
- Any external script or cron job hard-coding `build.py`, `serve.py`, or `data/prepare_*.py`
  paths will break. These were already broken (stubs exit 1) or are operational scripts with
  no external callers tracked in this repository.
- `logs/` ceases to exist as a directory. Any external tooling watching `python/logs/` must
  be redirected to `python/runs/`.

**Trade-off**:
- Fewer top-level directories reduces cognitive overhead for new contributors at the cost of
  one migration event. The alternative (keeping both `logs/` and `runs/`) creates permanent
  ambiguity about where build artifacts land.

**Position**:
- This spec governs the `python/` directory layout only. It does not affect `core/` (Rust),
  `core/kernels/` (CUDA), or any spec files.
- `checkpoints/` is explicitly excluded from this reorganization — it is an untracked runtime
  artifact directory managed by the build loop, not a source layout concern.

**Source**:
- Internal infrastructure migration. No paper equations traced. Motivated by organic growth of
  `python/` as the project scaled from single-experiment to multi-experiment. Canonical layout
  follows the principle that data, scripts, and run artifacts are three distinct concerns and
  should not share directories.

---

## Directory Layout (post-migration canonical state)

```
python/
├── hecate.py              # single entry point (build / chat / generate)
├── engine/                # orchestration modules (no math, CS-18)
│   ├── config.py
│   ├── loop.py
│   ├── data.py
│   ├── generation.py
│   ├── chat.py
│   ├── evaluation.py
│   ├── logging_utils.py
│   ├── tokenizer.py
│   └── donor.py
├── scripts/               # one-shot operational scripts (data prep, profiling, research)
│   ├── convert_checkpoint.py
│   ├── extract_llama_donor.py
│   ├── prepare_fineweb_edu.py
│   ├── prepare_fineweb_edu_llama3.py
│   ├── prepare_sft.py
│   ├── prepare_shakedown.py
│   ├── prepare_sharegpt.py
│   ├── prepare_curriculum.py
│   ├── download_fineweb.py
│   ├── profile_step.py
│   ├── profile_tape.py
│   ├── baseline_pytorch.py
│   ├── launch_curriculum.sh
│   └── launch_sharegpt.sh
├── validate_run.py        # operational validation tool (keep at root)
├── configs/               # JSON build configurations
├── data/                  # token corpora and data subdirectories (no scripts)
│   ├── fineweb_edu/
│   ├── fineweb_edu_llama3/
│   ├── sft_phase2/
│   ├── shakedown_1b/
│   ├── sharegpt/
│   ├── curriculum/
│   └── phase0/
├── runs/                  # build artifact sink: JSONL metrics + .log files
├── tests/                 # pytest suite
├── checkpoints/           # runtime checkpoints (untracked, do not reorganize)
├── pyproject.toml
└── Cargo.toml
```

## Files Removed (deprecated stubs)

| File | Reason |
|---|---|
| `build.py` | Backward-compat stub; already prints deprecation message and exits 1 |
| `serve.py` | Backward-compat stub; already prints deprecation message and exits 1 |

## Files Moved

| From | To | Reason |
|---|---|---|
| `data/prepare_fineweb_edu.py` | `scripts/prepare_fineweb_edu.py` | Scripts belong in `scripts/` |
| `data/prepare_fineweb_edu_llama3.py` | `scripts/prepare_fineweb_edu_llama3.py` | Scripts belong in `scripts/` |
| `data/prepare_sft.py` | `scripts/prepare_sft.py` | Scripts belong in `scripts/` |
| `data/prepare_shakedown.py` | `scripts/prepare_shakedown.py` | Scripts belong in `scripts/` |
| `data/prepare_sharegpt.py` | `scripts/prepare_sharegpt.py` | Scripts belong in `scripts/` |
| `data/prepare_curriculum.py` | `scripts/prepare_curriculum.py` | Scripts belong in `scripts/` |
| `data/download_fineweb.py` | `scripts/download_fineweb.py` | Scripts belong in `scripts/` |
| `profile_step.py` | `scripts/profile_step.py` | Research/profiling script |
| `profile_tape.py` | `scripts/profile_tape.py` | Research/profiling script |
| `baseline_pytorch.py` | `scripts/baseline_pytorch.py` | Research/profiling script |
| `launch_curriculum.sh` | `scripts/launch_curriculum.sh` | Launch script |
| `launch_sharegpt.sh` | `scripts/launch_sharegpt.sh` | Launch script |
| `logs/*.log` | `runs/llama_stacking_k4/` | Consolidated artifact sink |

## Path Reference Updates

Updates required after the move:
1. `engine/config.py` `BuildConfig` has `log_file: str | None = None` — no change needed;
   value is set at runtime by config JSON, not hardcoded
2. All existing config JSON files already use `runs/` as the `log_file` prefix — no change needed
3. `engine/loop.py` and `hecate.py` contain no hardcoded `logs/` references — no change needed
4. `tests/test_baseline.py`: `sys.path.insert` updated to point to `scripts/` for `baseline_pytorch`
5. `scripts/profile_step.py`: `sys.path.insert` added to resolve `engine.*` from `python/` root
6. `scripts/launch_sharegpt.sh`: `cd` changed to `python/` root; `data/prepare_sharegpt.py`
   updated to `scripts/prepare_sharegpt.py`
7. `scripts/launch_curriculum.sh`: `cd` changed to `python/` root; `data/prepare_curriculum.py`
   updated to `scripts/prepare_curriculum.py`
8. Docstring usage examples in all moved prepare scripts updated from `data/prepare_*` to
   `scripts/prepare_*`

`validate_run.py` is kept at `python/` root (not moved to `scripts/`) as an operational tool
invoked directly by developers from the project root.
