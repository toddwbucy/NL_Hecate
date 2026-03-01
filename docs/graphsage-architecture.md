# GraphSAGE Architecture: RGCN over NL Knowledge Graph

**Task**: GraphSAGE Phase 4 (task_3d6023)
**Date**: 2026-03-01
**Status**: Implemented

## Overview

This document records the architecture decisions for learning over the NL knowledge graph
to predict compliance edges between code files and code smells.

---

## 4.1 Training Task: Link Prediction

**Task**: Binary link prediction on `nl_smell_compliance_edges`.

Given a code file node (`arxiv_metadata`) and a smell node (`nl_code_smells`), predict
whether the code file complies with the smell.

**Why link prediction, not node classification?**
The compliance relationship is inherently relational — it exists between a code artifact
and a constraint. Node classification would require forcing this relationship into a node
property, losing the directional structure. Link prediction preserves the semantics.

**Graph role of non-compliance edges**: All other edge types participate in message
passing (RGCN convolutions) but are NOT prediction targets. This lets the model use
the rich semantic connections (specs → equations → axioms → smells) to reason about
compliance without directly training on those edges.

**Positive examples**: 104 compliance edges from `arxiv_metadata` → `nl_code_smells`
(106 minus 2 dangling from gradient-rs-part1 which failed to ingest due to VRAM)

**Negative examples**: Corrupt one endpoint — hold smell fixed, swap code file to a
non-compliant file. Ratio 1:5 positive:negative (graph is small, more negatives help).

---

## 4.2 Node Type Projection Strategy

All node types have 2048-dimensional embeddings (Jina V4, retrieval.passage task).

**Embedding sources**:
- Concept nodes (`nl_code_smells`, `hecate_specs`, equations, abstractions, axioms,
  `nl_reframings`, `nl_ethnographic_notes`, `nl_optimizers`, `nl_probe_patterns`):
  inline `embedding` field added by `python/scripts/embed_concept_nodes.py`
- `arxiv_metadata` (code files, papers): embeddings are stored per-chunk in
  `arxiv_abstract_embeddings`; the document-level embedding is the mean of its chunks.

**Projection**: Per-type `Linear(2048 → 256, bias=False)` layer. Each node type gets
its own projection matrix to handle the distributional shift between concept types.

```
Input:  2048d (Jina V4)
            ↓  per-type Linear (no bias)
Hidden: 256d  ← same space for all types after projection
```

**hidden_dim = 256**: Chosen to fit all ~1000 nodes comfortably in GPU memory with
2-layer RGCN. At 256d, full graph fits in <1GB.

---

## 4.3 Edge Type Handling: Basis Decomposition

**13 edge collections** in `hecate_knowledge_graph`. The prediction target
(`nl_smell_compliance_edges`) is removed from the message-passing graph — it becomes
the label-only signal.

**Remaining edge types for message passing** (12 collections):
1. `nl_hecate_trace_edges`: specs → equations/axioms (implements/cites)
2. `nl_signature_equation_edges`: Python signatures → equations
3. `nl_axiom_basis_edges`: abstractions → axioms
4. `nl_definition_source_edges`: definitions → equations
5. `nl_structural_embodiment_edges`: axioms → definitions/equations
6. `nl_validated_against_edges`: many → axioms
7. `nl_equation_depends_edges`: equations → equations
8. `nl_lineage_chain_edges`: lineage → equations/abstractions
9. `nl_migration_edges`: nl_ concepts → hope_ concepts
10. `nl_axiom_inherits_edges`: paper-specific axioms → nl_axioms
11. `nl_equation_source_edges`: equations → arxiv_abstract_chunks
12. `nl_smell_source_edges`: smells → arxiv_abstract_chunks

**RGCN basis decomposition** (B=4):

Full RGCN uses 12 weight matrices W_r ∈ R^{256×256} = 12 × 65,536 = 786K params —
likely to overfit on a ~1000-node graph. Basis decomposition reduces this:

```
W_r = Σ_{b=1}^{B} a_{rb} · V_b
```

where V_b ∈ R^{256×256} are shared basis matrices and a_{rb} are per-relation
coefficients. With B=4: 4 × 65K + 12 × 4 = 261K params (3× reduction).

---

## 4.4 Negative Sampling

**Strategy**: For each positive edge (code_file, smell), create 5 negative edges by
randomly swapping the code file while keeping the smell fixed (or vice versa).

**Rationale**: Smell-fixed corruption tests whether the model has learned which code
files should comply with a given smell — the primary generalization target.

**Implementation**: Uniform random sampling from all `arxiv_metadata` nodes not
connected to the smell. With 58 code files and 50 smells, the corrupted pool is large
enough that false negatives are rare.

---

## 4.5 Train / Val / Test Split

**106 positive edges** → split 88 / 9 / 9 (random, seed=42).

Transductive setting: the full graph participates in message passing during training,
validation, and test. Only the compliance edge set is split. This is the only practical
approach given the small graph size (~1000 nodes, ~106 positive edges).

---

## 4.6 Full-Graph vs. Mini-Batch

**Full-graph training** (no NeighborSampler).

Total nodes: ~1000. Total edges: ~3000 (excluding compliance target edges).
At 256d fp32, full node tensor = 1000 × 256 × 4 bytes = 1 MB. Trivially fits.

No need for sampling, neighbor aggregation approximations, or mini-batching.
This eliminates a major source of variance and simplifies debugging.

---

## Implementation

```
python/training/
├── __init__.py
├── rgcn_model.py          # HeteroRGCN: per-type projection + RGCN layers
└── compliance_predictor.py  # Link prediction head + loss + negative sampling
```

### rgcn_model.py: HeteroRGCN

```
Input HeteroData
    ├── per-type Linear(2048 → 256)   ← no bias, separate per type
    ├── concat all nodes → [N_total, 256]
    ├── RGCNConv layer 1 (basis B=4)  → [N_total, 256]
    │   LayerNorm + ReLU + Dropout
    ├── RGCNConv layer 2 (basis B=4)  → [N_total, 256]
    │   LayerNorm + ReLU + Dropout
    └── split back to per-type dicts
```

### compliance_predictor.py: CompliancePredictor

```
src_emb [B, 256] + dst_emb [B, 256]
    → concat [B, 512]
    → Linear(512 → 256) + ReLU
    → Linear(256 → 1)
    → squeeze → logits [B]
```

Loss: Binary cross-entropy with logits. Positive weight: 5.0 (to compensate for 1:5
negative ratio without over-correcting).

---

## Acceptance Criteria (from task_3d6023)

1. ✅ Training task defined: link prediction on nl_smell_compliance_edges
2. ✅ RGCN model skeleton at python/training/rgcn_model.py
3. ✅ Link prediction head at python/training/compliance_predictor.py
4. ✅ This architecture doc covers all decisions 4.1–4.6
5. Forward pass runs without error on dummy HeteroData (verified below)
6. ✅ Basis decomposition B=4 implemented via RGCNConv(num_bases=4)

---

## Open Questions (deferred to Phase 5)

- **Node type selection**: Should `arxiv_abstract_chunks` be included in the graph?
  Including them adds 272 nodes and provides paths from smells → chunks → equations → specs.
  Decision: include them in the full export but make optional during training.

- **Embedding aggregation for arxiv_metadata**: Mean-pool chunk embeddings vs. first-chunk
  only vs. attention-pooled. Decision: mean-pool (simplest, most stable).

- **Hyperparameter search**: hidden_dim, num_layers, dropout, learning rate.
  Defer to Phase 5 training runs.
