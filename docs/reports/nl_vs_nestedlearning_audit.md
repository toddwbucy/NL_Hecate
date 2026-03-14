# NL vs NestedLearning Database Audit

**Date**: 2026-03-07
**Auditor**: Claude (NL_Hecate core session)
**Reviewed by**: Todd (corrections applied inline)
**Context**: Discovered `NestedLearning` database alongside canonical `NL`. Conducted quality comparison to determine which should be canonical.

## TL;DR

NestedLearning is strictly superior to NL. Same structure (109 collections), same document counts — but NestedLearning has three improvements: (1) complete embedding coverage on code smells (60/60 vs 50/60), (2) 58 additional paper text chunks across all 8 source papers, (3) all 527 chunks have companion embeddings (vs 469 in NL). NestedLearning is a fully independent copy created via `arangodump`/`arangorestore`, not shared storage.

## Storage Model (CORRECTED)

**NestedLearning is an independent copy of NL**, created via `arangodump`/`arangorestore`. ArangoDB databases do not share collection storage. Documents appearing in both databases is the result of a manual sync script, not shared underlying storage. **Dropping NL would permanently delete NL's data.** NestedLearning has its own independent copy at parity due to sync.

## HADES Embedding Architecture (CORRECTED)

HADES uses a **split-collection pattern** for chunks:
- **Text**: stored in `arxiv_abstract_chunks` (no inline embedding field)
- **Vectors**: stored in `arxiv_abstract_embeddings` (companion collection, linked by key convention)

Checking for `embedding != null` on chunk documents will always return 0 — that's by design. The actual embedding coverage:

| Database | Chunks (text) | Companion embeddings (vectors) | Coverage |
|---|---|---|---|
| NL | 469 | 469 | **100%** |
| NestedLearning | 527 | 527 | **100%** |

All chunks in both databases are fully embedded. Semantic search over code and paper text is functional in both databases.

## Collection Counts (at parity via sync)

| Collection | NL | NestedLearning |
|---|---|---|
| nl_code_smells | 60 | 60 |
| nl_ethnographic_notes | 68 | 68 |
| hecate_specs | 104 | 104 |
| persephone_tasks | 248 | 248 |
| titans_equations | 35 | 35 |
| hope_equations | 128 | 128 |
| miras_equations | 38 | 38 |
| nl_smell_compliance_edges | 191 | 191 |
| nl_hecate_trace_edges | 399 | 399 |
| arxiv_metadata | 114 | 114 |
| arxiv_abstract_chunks | **469** | **527** |

## Difference 1: Embedding Coverage on Code Smells

| Collection | NL embedded | NestedLearning embedded |
|---|---|---|
| nl_code_smells | 50/60 (83%) | **60/60 (100%)** |
| hecate_specs | 85/104 | 85/104 |
| nl_ethnographic_notes | 64/68 | 64/68 |
| nl_reframings | 34/34 | 34/34 |
| nl_toolchain | 0/15 | 0/15 |
| nl_optimizers | 14/14 | 14/14 |
| titans_equations | 35/35 | 35/35 |
| hope_equations | 128/128 | 128/128 |

NL is missing inline embeddings on 10 code smells: `smell-033`, `smell-035`, `smell-036`, `smell-042`, `smell-043`, `smell-045`, `smell-046`, `smell-047`, `CS-49`, `CS-50`. These are invisible to `hades_query` in NL.

## Difference 2: Paper Text Chunks (NestedLearning only)

NestedLearning has 58 additional chunks — raw paper LaTeX text, chunked per paper:

| Paper | ArXiv ID | Chunks |
|---|---|---|
| HOPE | 2512.24695 | 32 |
| Lattice | 2504.05646 | 20 |
| Atlas | 2505.23735 | 16 |
| Titans | 2501.00663 | 14 |
| MIRAS | 2504.13173 | 14 |
| Conveyance | 2602.24281 | 12 |
| TNT | 2511.07343 | 10 |
| Trellis | 2512.23852 | 9 |

NL has zero paper text chunks. These are absent entirely. All 58 chunks in NestedLearning have companion embeddings — full-text paper search is operational.

## Code Chunks (identical in both)

Both databases have 85 code source files chunked (~370 chunks total), all with companion embeddings:

- **Rust core**: model.rs (17), tape.rs (13), lib.rs (12), gradient.rs (12), opaque-adapters.rs (12), dispatch.rs (11), mag.rs (11), moneta.rs (10), traced-forward.rs (9), self-ref.rs (8), trellis.rs (7), mal.rs (7), and 40+ more
- **CUDA kernels**: titans-backward.cu (3), delta-backward.cu (3), hebbian-backward.cu (3), swiglu-forward.cu (3), and 15+ more
- **Python engine**: loop.py (9), engine-loop.py (9), config.py (3), evaluation.py (3), and 10+ more

## Edge Integrity (identical in both)

- 11 orphan `_from` edges in `nl_hecate_trace_edges` (spec node deleted but edge remains)
- 65 `_to` edges pointing to non-standard collections
- 29/104 specs missing `traced_to_equations`
- 14/104 specs missing `paper_source`

## Recommended Action Plan (priority order)

### P0: Adopt NestedLearning as canonical
- NestedLearning is strictly superior: complete smell embeddings, paper text chunks with embeddings, all code chunks embedded
- Dump NL to `/bulk-store` as archive, then drop
- Update CLAUDE.md: `database="NL"` -> `database="NestedLearning"` everywhere
- Update MEMORY.md references
- **Critical**: NL and NestedLearning are independent — do NOT assume shared storage. Archive NL before dropping.

### P1: Backfill missing inline embeddings on non-chunk collections
- `nl_toolchain`: 0/15 embedded
- `hecate_specs`: 19/104 missing
- `nl_ethnographic_notes`: 4/68 missing

### P2: Fix edge integrity
- Remove 11 orphan trace edges
- Audit 65 non-standard `_to` targets
- Backfill `traced_to_equations` on 29 specs

## Notes for HADES Session

- NestedLearning's chunk embeddings (paper + code) are already complete — no embedding work needed on chunks
- The remaining embedding gaps are on non-chunk collections (toolchain, some specs, some ethnographic notes) which use inline embedding fields, not the companion collection pattern
- The `nl_toolchain` collection has 0/15 embeddings and should be prioritized
- Edge cleanup (P2) is non-blocking but improves graph traversal reliability

## Errors in Original Audit (transparency log)

1. **"Shared storage" claim**: WRONG. Inferred shared ArangoDB storage from observing a recently-synced document in both databases. In reality, NestedLearning is an independent `arangodump`/`arangorestore` copy kept at parity by a manual sync script. Dropping NL would be permanent data loss.
2. **"Zero chunk embeddings" claim**: MISLEADING. Checked for inline `embedding` fields on chunk documents (found 0). HADES uses a split-collection architecture: text in `arxiv_abstract_chunks`, vectors in `arxiv_abstract_embeddings`. Both databases have 100% companion embedding coverage. The P0/P1 embedding recommendations were unnecessary — the work was already done.
