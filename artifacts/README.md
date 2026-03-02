# NL-Hecate Compiled Artifacts

Compiled binary outputs live here. **Binaries are gitignored** — this directory
tracks the layout, not the content.

See `specs/infrastructure/02_artifact_layout.md` for the full spec.

## Layout

```
artifacts/
├── .build-meta.json      ← written by promote_artifacts.py; records git SHA + sizes
├── cuda/                 ← reserved for standalone CUDA fat binaries (future)
├── so/                   ← SHA-tagged PyO3 .so files
└── wheels/               ← version-tagged maturin .whl files
```

## Rebuild + Promote

```bash
# Build the release .so and install into the active venv:
cd python && maturin develop --release && cd ..

# Promote: copies .so to artifacts/so/, writes .build-meta.json, prints HADES JSON:
python scripts/promote_artifacts.py

# Optional: promote a wheel instead of a dev install
cd python && maturin build --release && cd ..
python scripts/promote_artifacts.py --wheel
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

Every promoted artifact registers a document in the `hecate_artifacts` collection
of the NL HADES database. Query:

```bash
hades --database NL db aql "FOR a IN hecate_artifacts SORT a.build_date_epoch DESC LIMIT 5 RETURN a"
```

## Distribution

Versioned builds are published as GitHub Release assets:

```bash
gh release create v0.4.0 artifacts/so/nl_hecate.*.so --title "v0.4.0" --notes "..."
```
