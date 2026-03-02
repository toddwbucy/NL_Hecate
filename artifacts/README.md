# NL-Hecate Compiled Artifacts

Compiled binary outputs live here. **Binaries are committed to git** — at 4.1MB
the release `.so` is small enough to distribute directly through the repository.

See `specs/infrastructure/02_artifact_layout.md` for the full spec.

## Layout

```text
artifacts/
├── .build-meta.json      ← gitignored; written by promote_artifacts.py; local staleness pointer
├── cuda/                 ← reserved for standalone CUDA fat binaries (future)
├── so/                   ← SHA-tagged PyO3 .so files (committed)
│   └── nl_hecate.cpython-312-x86_64-linux-gnu.<sha>.so
└── wheels/               ← version-tagged maturin .whl files (committed)
    └── nl_hecate-<version>-cp312-linux_x86_64.whl
```

## Rebuild + Promote

```bash
# Build the release .so and install into the active venv:
cd python && maturin develop --release && cd ..

# Promote: copies .so to artifacts/so/, writes .build-meta.json, prints HADES JSON:
python scripts/promote_artifacts.py [--task-key task_87a521]

# Optional: promote a wheel instead of a dev install
cd python && maturin build --release && cd ..
python scripts/promote_artifacts.py --wheel [--task-key task_87a521]

# Commit the promoted binary:
git add artifacts/so/ && git commit -m "chore: promote nl_hecate .so <sha>"
```

## Staleness Check

`artifacts/.build-meta.json` records the git SHA at promote time.
Compare against current HEAD to know if the `.so` is stale:

```bash
python -c "
import json, subprocess
meta = json.load(open('artifacts/.build-meta.json'))
head = subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
if meta['git_sha'] != head:
    print(f'STALE: built at {meta[\"git_sha\"][:8]}, current HEAD {head[:8]}')
else:
    print('OK: .so is current')
"
```

## Artifact Provenance (HADES)

Every promoted artifact has a metadata node in the `hecate_artifacts` collection
of the NL HADES database, connected to the source code nodes it was compiled from
via `hecate_artifact_edges` (`rel: compiled_from`). Query:

```bash
hades --database NL db aql "FOR a IN hecate_artifacts SORT a.build_date_epoch DESC LIMIT 5 RETURN a"
```

Traverse source provenance:

```bash
hades --database NL db aql "
  FOR v, e IN 1..1 OUTBOUND 'hecate_artifacts/nl-hecate-so-e0010028'
    hecate_artifact_edges RETURN v._key
"
```
