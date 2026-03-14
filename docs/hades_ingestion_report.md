# HADES Code Ingestion Report
# For: HADES Session — Full Codebase Ingestion (COMPLETE)

Date: 2026-02-26
Author: claude-sonnet-4-6 (ses_102a74 → continuation sessions)
Scope: Full NL_Hecate codebase ingested (73 code nodes, 60 compliance edges)

---

## What We Did

Ingested 10 source files from NL_Hecate into the HADES NL graph using `hades ingest`,
then manually created compliance edges in `nl_smell_compliance_edges` connecting each
file node to the code smell(s) it claims compliance with.

### Probe Files: Results

| File | ID | Edges | Smells |
|------|----|-------|--------|
| core/src/conductor.rs | conductor-rs | 1 | CS-32 |
| core/kernels/adamw.cu | adamw-cu | 0 | (NO_REFS — infra kernel) |
| core/src/adamw.rs | adamw-rs | 2 | CS-27, CS-28 |
| core/kernels/dgd_forward.cu | dgd-forward-cu | 1 | CS-33 |
| core/src/context_stream.rs | context-stream-rs | 1 | CS-11 |
| core/src/m3.rs | m3-rs | 2 | CS-27, CS-28 |
| core/src/tape.rs | tape-rs | 3 | CS-40, CS-42, CS-47 |
| core/src/cms_variants.rs | cms-variants-rs | 2 | CS-19, CS-23 |
| core/src/composition_safety.rs | composition-safety-rs | 5 | CS-31, CS-33, CS-34, CS-35, CS-36 |
| core/kernels/dgd_backward.cu | dgd-backward-cu | 0 | (NO_REFS — infra kernel) |

**Total**: 10 files ingested, 17 compliance edges created, 0 errors.

All nodes landed in `arxiv_metadata` collection with code-pipeline chunks in `arxiv_abstract_chunks`.
All edges in `nl_smell_compliance_edges`.

---

## Friction Points Encountered

### Friction #1 (HIGH): Two parallel smell collections

**Problem**: The graph has two collections containing code smell definitions:
- `nl_code_smells` — 47 smells (CS-01 through CS-47). Existing `nl_smell_compliance_edges`
  all use this as their `_to` target. Different key schema, e.g. `smell-032-observe-then-advance`.
- `hope_code_smells` — 50 smells (CS-01 through CS-50). Populated during task_093b0f to add
  missing smells CS-32 through CS-38, CS-41 through CS-50. Uses a different key schema.

**Impact**: During probe ingestion we had to decide which collection to use for edges.
We chose `nl_code_smells` since that's what existing infrastructure uses. But:
1. CS-48, CS-49, CS-50 don't exist in `nl_code_smells` — `gpu_forward.rs` claims CS-49
   in the codebase but we can't create a compliant edge for it without a target node.
2. `hope_code_smells` has more complete and recently-updated definitions (including the
   corrected CS-40 definition). `nl_code_smells` may be stale.
3. Future agents will not know which collection is canonical.

**Recommended Fix**: Merge and deduplicate. Either:
- (a) Make `nl_code_smells` the canonical collection: sync the 3 missing smells (CS-48, CS-49, CS-50)
  and the corrected CS-40 definition into it. Deprecate `hope_code_smells`.
- (b) Make `hope_code_smells` canonical: repoint all existing `nl_smell_compliance_edges` to use
  `hope_code_smells` key schema, then deprecate `nl_code_smells`.

**Recommendation**: Option (a). `nl_code_smells` already has all the existing edge infrastructure.
Add the 3 missing smells with the correct key format and update CS-40's definition.

---

### Friction #2 (MEDIUM): `hades db check` reports misleading collection name

**Problem**: `hades db check <id>` returns `"collection": "arxiv"` for files ingested via
`hades ingest`, but the actual ArangoDB collection is `arxiv_metadata`. This is a display
normalization that doesn't match what you need for building compliance edge `_from` fields.

**Impact**: When creating edges manually, we had to run a verification AQL query to confirm
the real `_id` was `arxiv_metadata/<id>` before writing the edge. Without this cross-check,
edge `_from` values would be wrong and edges would be invalid (pointing to non-existent nodes).

**Recommended Fix**: Either:
- (a) Return the actual ArangoDB collection name in `hades db check` output
- (b) Add a `document_id_full` field to the check output that gives the full `<collection>/<key>` path

---

### Friction #3 (MEDIUM): No CLI command to create compliance edges

**Problem**: There is no `hades` command for "link this file to this smell". Creating each
edge required:
1. Looking up the `_to` key manually in the smell collection
2. Constructing the composite `_key` string manually (`_from_coll_key__to_coll_key`)
3. Writing a full JSON document via `hades db insert nl_smell_compliance_edges --data '{...}'`

This is error-prone. During ingestion of all 10 files we had to:
- Construct 17 edge keys by hand
- Ensure the composite key format matched the existing convention
- Verify edges resolved correctly after creation

**Recommended Fix**: Add a high-level command:
```bash
hades link <source-id> --smell CS-32 --enforcement behavioral \
  --methods "next_chunk,advance" \
  --summary "Observe before advance — pulse read before step increments"
```
This would:
1. Look up `<source-id>` in `arxiv_metadata` to confirm it exists
2. Look up `CS-32` in the canonical smell collection to get the `_key`
3. Construct the composite edge key
4. Insert into `nl_smell_compliance_edges`
5. Return confirmation with the resolved edge

---

### Friction #4 (LOW): Ingested code nodes have no type/role metadata

**Problem**: Ingested files all land as generic documents in `arxiv_metadata`. There is no
field distinguishing a Rust source file from a CUDA kernel from an academic paper. The only
signal is the `document_id` we manually choose.

**Impact**: Queries like "show me all CUDA kernels" or "find Rust files that implement CS-33"
require knowing the naming convention we used (e.g., `-cu` suffix for CUDA). There is no
queryable type field.

**Recommended Fix**: `hades ingest` should attach type metadata based on file extension:
```json
{
  "document_id": "adamw-cu",
  "file_type": "cuda_kernel",
  "source_path": "core/kernels/adamw.cu",
  "language": "cuda",
  ...
}
```
Standard fields: `file_type` (`rust_source`, `cuda_kernel`, `python_source`, etc.),
`source_path` (relative path from repo root), `language`.

---

### Friction #5 (LOW): No batch-ingest-with-edges command

**Problem**: To ingest a file AND create its compliance edges requires:
1. `hades ingest <file> --id <id>` (one command)
2. One `hades db insert nl_smell_compliance_edges ...` per smell referenced (N more commands)

For a file claiming 5 smells (like composition_safety.rs), this is 6 separate commands.
For the full 71-file codebase, that's potentially 100+ commands.

**Recommended Fix**: Allow edge declarations inline with ingest:
```bash
hades ingest core/src/composition_safety.rs --id composition-safety-rs \
  --claims CS-31:architectural,CS-33:architectural,CS-34:architectural,CS-35:architectural,CS-36:architectural
```
Or via a sidecar YAML/JSON file:
```yaml
# composition_safety.claims.yaml
file: core/src/composition_safety.rs
id: composition-safety-rs
claims:
  - smell: CS-31
    enforcement: architectural
    methods: [CompositionSafetyMarker]
    summary: "NLM is indivisible — marker traits prevent subsystem extraction"
```

---

## Summary: What Works Well

1. **`hades ingest <file.rs>`** — auto-detects code files, Jina V4 Code LoRA, clean ingestion
2. **AQL access** — `hades db aql` gives full flexibility for verification queries
3. **nl_smell_compliance_edges** — the edge collection infrastructure is already there and works
4. **`hades db insert`** — reliable, returns clear success/failure

---

## Recommended Priority Order for HADES Improvements

| Priority | Friction | Effort | Impact |
|----------|----------|--------|--------|
| 1 | #1: Merge dual smell collections (add CS-48/49/50 to nl_code_smells, fix CS-40) | Low | High |
| 2 | #3: `hades link` command for compliance edges | Medium | High |
| 3 | #2: Fix `hades db check` collection display | Low | Medium |
| 4 | #4: Add type/role metadata to ingested code nodes | Low | Medium |
| 5 | #5: Batch ingest + claims command | Medium | Medium |

Once items 1 and 2 are complete, the remaining 61 files can be ingested efficiently.

---

## Workflow for Remaining 61 Files (Post-Improvements)

After HADES implements the above:

```bash
# For CLEAN files (have CS refs):
hades ingest core/src/<file>.rs --id <file>-rs --claims CS-XX:enforcement,...

# For NO_REFS files (infrastructure):
hades ingest core/kernels/<file>.cu --id <file>-cu
# (no --claims needed)
```

The `docs/audit_report.txt` already has the complete map of which files claim which smells.
It can serve as the input manifest for batch ingestion once the `--claims` mechanism exists.

---

## Paper Embeddings + Equation→Chunk Edges (Added Same Session)

After the 10-file probe, the following was also completed:

### 7 Core Papers Ingested into NL

All 7 NL research papers are now embedded in NL's `arxiv_abstract_chunks` with
full-text semantic search via Jina V4 late-chunking:

| Paper | arxiv_id | Chunks (NL) |
|-------|----------|-------------|
| Titans | 2501.00663 | 4 |
| MIRAS | 2504.13173 | 14 |
| HOPE/NL | 2512.24695 | 19 |
| Lattice | 2504.05646 | 6 |
| ATLAS | 2505.23735 | 5 |
| TNT | 2511.07343 | 2 |
| Trellis | 2512.23852 | 5 |

NL uses late-chunking with ~7.5K char windows (vs ~3K in arxiv_datastore). Total
coverage per paper is slightly lower in raw chars but semantically equivalent.

### 629 Equation→Chunk Edges Created

New edge collection: `nl_equation_source_edges` (proper ArangoDB edge type).

For each of the 317 equations across all 7 paper collections, semantic search was
run within the source paper's chunks using the equation's `description` field as
the query. Edges created for top matches above similarity=0.55, max 2 per equation.

Edge schema:
```json
{
  "_from": "hope_equations/eq-070-arch-variant1",
  "_to": "arxiv_abstract_chunks/2512_24695_chunk_14",
  "source_field": "equation_source",
  "similarity": 0.696,
  "equation_name": "Eq 70: CMS Chain Composition"
}
```

**Note**: ArangoDB chunk keys use underscores in the arxiv ID (`2512_24695`),
not dots. This was a trip-hazard during edge construction — keep in mind for
future edge builders targeting `arxiv_abstract_chunks`.

### Smell→Chunk Edges (134 edges)

New edge collection: `nl_smell_source_edges` (proper ArangoDB edge collection).

For each of the 49 smells in `nl_code_smells`, semantic search was run across all 7
papers using the smell's `description` field as query. Best match per paper above
similarity=0.60 kept.

- **37/49 smells** have chunk edges (75%)
- **12/49 smells** have no edges — these are nodes with no `description` field in
  `nl_code_smells`; they fell back to the smell name only (too short for good semantic
  retrieval). **This is a data gap in `nl_code_smells` fixed by the collection merge.**
  The richer `hope_code_smells` descriptions would close this.

### Full Traversal Chain Now Live

Two paths from code to paper context:

**Path A (via equations):**
```
source code → nl_smell_compliance_edges → nl_code_smells
           → (hecate_specs trace edges)  → {paper}_equations
           → nl_equation_source_edges    → arxiv_abstract_chunks
```

**Path B (direct smell→context):**
```
source code → nl_smell_compliance_edges → nl_code_smells
           → nl_smell_source_edges      → arxiv_abstract_chunks
```

Path B is simpler and works for any smell with a description. Path A is more
precise (pinpoints the specific equation) but requires the smell→equation
link to be manually curated. HADES should support both.

Verified via AQL: conductor.rs → CS-32 → HOPE paper chunk 11 (section on gating
in sequence models) and MIRAS chunk 0. Full `DOCUMENT()` resolution works correctly.

---

## Open Items After This Session

1. ~~**CS-49 in nl_code_smells**~~ — **RESOLVED**: HADES added CS-48/49/50 to `nl_code_smells`.
   `gpu_forward.rs` CS-49 edge created successfully.

2. ~~**CS-48 in nl_code_smells**~~ — **RESOLVED** (same as above).

3. **Fix build gaps** (independent of graph work):
   - Commit python/Cargo.lock (remove from .gitignore)
   - Add git hash to checkpoint schema in hecate.py
   - See `docs/build_assessment.md` for implementation details

4. **CS-28 compliance probe**: No test exists. Decision pending: write probe or accept code review.
   See `docs/compliance_probes.md`.

5. **`hades link --force` upsert**: When a compliance edge already exists (created by `--claims`),
   `hades link --force` fails. HADES needs upsert support. Workaround: delete bare edge first.

6. **gradient.rs size limit**: `gradient.rs` (306K, 6406 lines, mostly `#[cfg(test)]`) exceeds
   Jina V4's context window for late-chunking on a single GPU (needs 51 GiB, A6000 has 47.4 GiB).
   Ingested as 3 parts: `gradient-rs` (main code, CS-10/CS-17), `gradient-rs-tests-a`,
   `gradient-rs-tests-b`. Compliance edges point to `gradient-rs` (the main code section).

---

## Final Ingestion Status (2026-02-26, end of session)

**Total code nodes ingested**: 73 (in `arxiv_metadata`)
**Total compliance edges**: 60 (in `nl_smell_compliance_edges`)
**Papers embedded**: 7 (in `arxiv_abstract_chunks`, NL late-chunking)
**Equation→chunk edges**: 629 (in `nl_equation_source_edges`)
**Smell→chunk edges**: 163 (in `nl_smell_source_edges`)

### Code Node Breakdown

| Category | Count | Examples |
|----------|-------|---------|
| CLEAN Rust (with compliance edges) | 31 | conductor-rs, tape-rs, mag-rs, gradient-rs |
| NO_REFS CUDA kernels | 14 | titans_forward, delta_backward, m_norm_clamp |
| NO_REFS Rust (infra/no CS refs) | 25 | moneta, opaque_adapters, associative_scan |
| gradient.rs test splits | 2 | gradient-rs-tests-a, gradient-rs-tests-b |
| virtual NL_PATH nodes (pre-existing) | 1+ | NL_PATH_HYBRID, NL_PATH_PURE |

### Compliance Edge Distribution (by smell)

Most-referenced smells (>1 file claims them):
- CS-32 (observe-then-advance): 6 files
- CS-18 (forward-pass-only-api): 6 files
- CS-39 (unbounded-learnable-decay): 4 files
- CS-33 (no-forced-same-bias): 4 files
- CS-10 (no-train-eval-mode): 3 files
- CS-42 (grad-checkpoint-counterproductive): 3 files
- CS-27 (no-independent-optimizer): 3 files

### Graph Traversal Now Available

Full path from code to paper semantic context is operational:
```
source code → nl_smell_compliance_edges → nl_code_smells
           → nl_smell_source_edges      → arxiv_abstract_chunks (direct)
           → (hecate_specs trace edges) → {paper}_equations
           → nl_equation_source_edges   → arxiv_abstract_chunks (via equations)
```
