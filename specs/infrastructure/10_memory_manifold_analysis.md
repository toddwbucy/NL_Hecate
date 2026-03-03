# Memory Manifold Analysis

```
CONTRACT
  Purpose:    Post-hoc diagnostic tool for CMS memory state geometry across
              training checkpoints. Tests the manifold hypothesis: M_l states
              converge onto progressively lower-dimensional submanifolds of
              R^{d×d}, and their vocabulary projections trace semantically
              distinct, level-specific regions in the token embedding space.

  Expects:    - Training JSONL log containing memory_vocab_probe events
                (produced by evaluation.py:probe_memory_vocab)
              - Checkpoint safetensors file (provides W_embed [v,d],
                W_unembed [d,v]) for the target step
              - Tokenizer JSON (BPE 32K or compatible)
              - Optional: multiple checkpoints for temporal trajectory

  Guarantees: - Per-level temporal trajectory of JS divergence between all
                K*(K-1)/2 level pairs as a function of training step
              - Per-level effective rank profile of M_l (PCA on d row vectors)
              - Per-level vocabulary semantic clustering metric (activated
                token neighborhood coherence in embedding space)
              - Falsification verdict against gate_warmup spec thresholds
                (spec 09): JS(L0,L3) > 0.1 nats at step 20K = GO

  Cost:       O(d^2) PCA per level per checkpoint; O(v^2 * d) for one-time
              semantic graph construction (v=32000, d=256 → ~2GB float32).
              Semantic graph is cached to disk after first build.

  Trade-off:  Semantic graph construction is expensive; skip with
              --no-semantic-graph flag for fast trajectory-only analysis.
              Vocabulary probe data is already logged by the training loop;
              this tool is purely a reader — it does not run inference.

  Position:   Python tier, post-hoc only. Never called from training loop.
              Reads from JSONL logs and checkpoint files on disk.

  Source:     HOPE (2512.24695) §5: "each block or level has its own context
              flow … all the levels are performing in-context learning but on
              their own context flow." Prediction: distinct context flows →
              distinct compressed representations → distinct vocabulary regions.
              MIRAS (2504.13173) §2: Associative memory M maps K→V under
              attentional bias objective. The probe M_l @ W_unembed asks which
              V-directions are most aligned with M_l's current key-value mapping.
              Lattice (2504.05646) §1: "inherent low-rank structure of K-V
              matrices." Low rank = memory lies on a submanifold. The effective
              rank of M_l row vectors is the empirical measurement of this claim.
```

---

## Background: What Is Being Measured

### The Vocabulary Probe as a Read Operation

Each CMS memory matrix `M_l ∈ R^{d×d}` stores the associative mapping learned
by level l from its context flow. The probe computation in `evaluation.py` is:

```
probs_l = softmax( mean(M_l @ W_unembed, axis=0) )   # [v]
```

This is a **linear read**: ask M_l "which vocabulary directions are you most
aligned with?" Each row of `M_l` is a d-dimensional vector representing one
memory slot; `M_l @ W_unembed` projects each slot into vocabulary space. The
mean aggregates across all d slots.

**What it measures**: the average token prediction that M_l would produce if
queried uniformly. This is not "what tokens were stored" — it is "which tokens
does the current memory geometry prefer."

**What it does NOT measure**: specific key-value associations, because we do not
query with a specific key. The analysis is about the geometry of M_l itself, not
its retrieval behavior on a specific input.

### The Manifold Hypothesis Applied to M_l

The d rows of `M_l` are d vectors in R^d. The manifold hypothesis predicts:

1. **Rank collapse**: trained M_l rows lie near a low-dimensional subspace.
   Random matrices have full rank d; learned matrices collapse onto r << d
   effective dimensions encoding the context flow's structure.

2. **Level-specific subspaces**: M_0's row space ⊂ R^d encodes token-timescale
   patterns; M_3's row space encodes 512-step document patterns. These subspaces
   should be distinct (non-overlapping principal components).

3. **Vocabulary coherence**: when M_l's row space is projected through W_unembed,
   the high-probability vocabulary tokens should cluster semantically — i.e., the
   top-k tokens form coherent neighborhoods in the embedding graph rather than
   random scatter.

### Why JS Divergence Is the Primary Signal

The Jensen-Shannon divergence between level pair (i, j):

```
JS(p_i || p_j) = 0.5 * KL(p_i || m) + 0.5 * KL(p_j || m)  where m = 0.5*(p_i+p_j)
```

Already logged per training step. A training run where levels differentiate
correctly will show:
- JS ≈ 0 at initialization (all memories uniform)
- JS growing monotonically after theta_floor scaffold decays (step ~10K)
- JS(L0,L3) >> JS(L0,L1) at convergence (distant levels diverge most)

Zero JS at step 20K = levels did NOT specialize = CMS frequency hierarchy failed
to produce differentiated compression.

---

## Analysis Modules

The tool is structured as four independent modules that can be run selectively:

### Module 1: JS Divergence Trajectory (`--module js`)

**Input**: JSONL log, `memory_vocab_probe` events only.

**Computation**:
```python
steps, js_matrix = load_js_series(jsonl_path)
# js_matrix[step_idx, i, j] = JS(level_i, level_j) at that step
```

**Output**:
- `{run_name}_js_trajectory.csv`: step, JS(0-1), JS(0-2), JS(0-3), JS(1-2), JS(1-3), JS(2-3)
- `{run_name}_js_trajectory.png`: line plot with falsification threshold line at 0.1 nats

**Falsification check**:
```
if JS(L0, L3)[step=20000] < 0.1:
    verdict = "FAIL: levels not differentiated at step 20K"
else:
    verdict = "PASS"
```

This is the same threshold used by spec 09 (gate_warmup) for the GO/NO-GO
decision, expressed as JS rather than θ. Both conditions should hold together.

---

### Module 2: Memory Effective Rank (`--module rank`)

**Input**: Checkpoint safetensors file (provides M_l via context snapshot) OR
           snapshot_params.json extracted from JSONL log at target step.

**Computation** (per level):
```python
M_l = np.array(context.memory[l]).reshape(d, d)    # [d, d]
_, S, _ = np.linalg.svd(M_l, full_matrices=False)  # singular values
stable_rank = (S.sum() / S.max()) ** 2              # ‖M‖_F^2 / ‖M‖_2^2
effective_rank = np.exp(-np.sum(p * np.log(p+1e-10)))  # entropy of normalized spectrum
# where p = S / S.sum()
```

Two rank measures:
- **Stable rank** (sr): squares ratio of Frobenius to spectral norm. Invariant
  to scaling. For a rank-r matrix: sr = r.
- **Spectral entropy rank** (er): exponential of the Shannon entropy of the
  normalized singular value spectrum. For a rank-r matrix with equal singular
  values: er = r.

**Prediction**: At initialization, sr ≈ d (random matrix, full rank). After
training, sr << d, with lower-frequency levels having LOWER sr (more compressed,
fewer effective dimensions needed for 512-step timescale patterns vs. 1-step
patterns).

Wait — this prediction requires care. Lower-frequency levels fire less often and
thus have fewer gradient updates. They may have LOWER rank simply from underuse.
The interesting question is whether the rank is nonetheless informative: does
a low-rank M_3 encode coherent document-level information, or is it just small?

Cross-reference with m_norm: distinguish "low rank because small" (norm-scaled)
from "low rank because compressed onto a subspace" (norm-independent).

**Output**:
- `{run_name}_rank_profile.csv`: step, level, stable_rank, spectral_entropy_rank, m_frobenius_norm
- Table per checkpoint showing rank trajectory per level.

---

### Module 3: Vocabulary Semantic Clustering (`--module cluster`)

**Input**: Checkpoint safetensors (W_embed [v,d]), top-k token IDs per level
           from JSONL memory_vocab_probe events.

**Semantic graph construction** (one-time, cached):
```python
W = W_embed.astype(np.float32)          # [v, d]
W /= np.linalg.norm(W, axis=1, keepdims=True)  # normalize rows
# Build sparse k-NN graph: for each token, find 20 nearest neighbors
# by cosine similarity. Store as scipy.sparse adjacency matrix.
# IMPORTANT: Do NOT compute full [v,v] cosine similarity matrix in memory.
# Use approximate nearest-neighbor (e.g., sklearn BallTree or faiss) for v=32K.
G_semantic = build_knn_graph(W, k=20)
```

**Clustering metric** (per level, per step):
```python
top_k_ids = [entry["id"] for entry in level_probe["top20"]]
subgraph = G_semantic[np.ix_(top_k_ids, top_k_ids)]
# Measure clustering coefficient of the induced 20-node subgraph
density = subgraph.nnz / (len(top_k_ids) * (len(top_k_ids) - 1))
# Baseline: expected density for 20 random tokens from the full graph
baseline_density = G_semantic.nnz / (v * (v - 1))
coherence_ratio = density / (baseline_density + 1e-10)
```

**Interpretation**: `coherence_ratio > 1` means activated tokens are more
semantically connected than chance. A well-differentiated level should show
coherence_ratio >> 1, meaning M_l's vocabulary projection activates semantically
coherent neighborhoods.

**Output**:
- `{run_name}_vocab_clustering.csv`: step, level, coherence_ratio, top5_tokens (decoded)
- Qualitative section: print top-5 decoded tokens per level at the final step.

**Caveat**: This metric is only meaningful once vocabulary probabilities are
non-uniform (m_norm > 1e-3, JS > 0). Before that point, top-20 tokens are
random due to near-uniform softmax.

---

### Module 4: Level Subspace Alignment (`--module align`)

**Input**: Two checkpoint files at steps T1 and T2 (or a single checkpoint plus
           the zero-init baseline).

**Computation**:
```python
# For each level l, compute principal subspace alignment between steps
U1, _, _ = np.linalg.svd(M_l_T1, full_matrices=False)  # [d, d]
U2, _, _ = np.linalg.svd(M_l_T2, full_matrices=False)
# Grassmann distance: principal angle sum between top-r subspaces
r = 8  # top-8 principal components
cos_angles = np.linalg.svd(U1[:, :r].T @ U2[:, :r], compute_uv=False)
grassmann_dist = np.arccos(np.clip(cos_angles, -1, 1)).sum()
```

**Prediction**: Between consecutive checkpoints at convergence, Grassmann
distance should be small (stable subspace). Between early and late training,
distance should be large (subspace rotates significantly). Between level 0 and
level 3 at any checkpoint, cross-level Grassmann distance should be large
(distinct subspaces).

**Output**:
- `{run_name}_subspace_alignment.csv`: step_pair, level, grassmann_distance

---

## Input/Output Protocol

### Command-Line Interface

```bash
python tools/memory_manifold_analysis.py \
  --log runs/gate_warmup_diagnostic.jsonl \
  --checkpoint checkpoints/gate_warmup_diagnostic.safetensors \
  --tokenizer /path/to/tokenizer.json \
  --module js rank cluster align \
  --out results/gate_warmup_manifold/ \
  [--no-semantic-graph]   # skip Module 3 (expensive graph build)
  [--step 20000]          # analyze a specific step only
  [--compare-steps 5000 10000 20000]  # temporal comparison
```

### Output Directory Layout

```
results/gate_warmup_manifold/
  js_trajectory.csv
  js_trajectory.png
  rank_profile.csv
  vocab_clustering.csv          (if --module cluster)
  subspace_alignment.csv        (if --module align)
  semantic_graph.npz            (cached, reused across runs)
  report.txt                    (human-readable summary + verdicts)
```

### Report Format

The `report.txt` summarizes all modules and renders the falsification verdict:

```
=== Memory Manifold Analysis: gate_warmup_diagnostic ===
Run: 25000 steps  Checkpoint: step 20000

[Module 1: JS Divergence]
  L0-L1: 0.031  L0-L2: 0.087  L0-L3: 0.143  ← PASS (> 0.1 threshold)
  L1-L2: 0.041  L1-L3: 0.112  L2-L3: 0.067

[Module 2: Effective Rank]
  L0: stable_rank=18.3  spec_entropy=14.1  ‖M‖=2.34
  L1: stable_rank=12.1  spec_entropy=9.8   ‖M‖=1.87
  L2: stable_rank=7.4   spec_entropy=6.2   ‖M‖=1.23
  L3: stable_rank=3.1   spec_entropy=2.9   ‖M‖=0.44
  Rank gradient: L0 > L1 > L2 > L3 ← consistent with frequency prediction

[Module 3: Vocabulary Clustering]
  L0 coherence_ratio=4.2  top5: the, a, in, of, and
  L1 coherence_ratio=3.8  top5: said, asked, told, replied, whispered
  L2 coherence_ratio=5.1  top5: however, therefore, moreover, furthermore, thus
  L3 coherence_ratio=6.7  top5: chapter, section, part, introduction, conclusion

[Falsification Verdict]
  Gate warmup (spec 09): L2_theta=0.0063 > 0.005 ✓  L3_theta=0.0014 > 0.001 ✓
  Level differentiation: JS(L0,L3)=0.143 > 0.1 ✓
  OVERALL: GO — CMS levels have differentiated, gate warmup succeeded.
```

---

## Implementation Notes

### Tool Location

`python/tools/memory_manifold_analysis.py` — standalone script, no import from
`engine/` except for the existing `probe_memory_vocab` function (reused, not
copied).

### Checkpoint Snapshot Access

The tool needs M_l from a checkpoint. Two paths:
1. **From JSONL** (preferred for historical steps): `memory_vocab_probe` events
   contain `m_norm` per level but NOT the full M matrix. To get the full M_l,
   must use path 2.
2. **From safetensors checkpoint**: `nl_hecate.load_checkpoint(path)` returns
   params and context. `context.memory[l]` is a flat Vec<f32> of length d*d.

### Semantic Graph Build Cost

For v=32000, d=256, building the full cosine similarity matrix requires:
- Dense: 32000 × 32000 × 4 bytes = 4GB — too large for in-memory.
- Approximate k-NN via FAISS (IndexFlatIP on normalized vectors): 32000 × 256 × 4 = 33MB index. Feasible.
- Use `faiss.IndexFlatIP` with normalized vectors for cosine similarity.
- Store result as `scipy.sparse.csr_matrix` (20 edges × 32000 nodes ≈ 640K entries).

If FAISS not installed, fall back to BallTree on PCA-reduced embeddings (top-64
PCs reduces to 32000 × 64, making batch cosine feasible).

### Probe Data Availability

The `memory_vocab_probe` event currently stores only top-20 token IDs + probs +
m_norm per level, NOT the full M_l matrix. This is sufficient for Modules 1 and
3. Module 2 (effective rank) requires the full M_l and thus requires a checkpoint
file at the target step. Module 4 requires two checkpoints.

### Relationship to Existing Probe Infrastructure

`evaluation.py:probe_memory_vocab` is the **data producer**. This spec's tool is
a **consumer** of that data. Do not modify `probe_memory_vocab` to support this
analysis — the JSONL format is the contract between them.

If richer per-level data is needed in the future (e.g., full M_l snapshots in
JSONL), that is a separate change to the evaluation infrastructure, not this tool.

---

## Falsification Criteria

### Primary (gate_warmup_diagnostic co-verdict at step 20K)

| Signal | Threshold | Source |
|--------|-----------|--------|
| JS(L0, L3) | > 0.1 nats | This spec (empirically motivated) |
| L2 θ | > 0.005 | Spec 09 (gate_warmup) |
| L3 θ | > 0.001 | Spec 09 (gate_warmup) |

All three must pass for GO on CMS level specialization.

### Secondary (rank gradient prediction)

The stable rank of M_l should decrease monotonically with level index (lower
frequency → more compressed representation → lower effective rank):
`stable_rank(L0) > stable_rank(L1) > stable_rank(L2) > stable_rank(L3)`

This is a structural prediction of the NL framework (HOPE §5): lower-frequency
levels compress more context into fewer effective dimensions because their context
flow contains fewer independent examples per build.

### Tertiary (vocabulary coherence gradient)

`coherence_ratio(L3) > coherence_ratio(L0)` — document-timescale memory should
activate MORE semantically coherent vocabulary regions than token-timescale
memory, because longer timescales encode topically coherent segments.

This is the weakest prediction and depends on the dataset having long-range topic
structure (C4 does; random token streams would not).

---

## Extension: Cross-Run Comparison

When comparing runs (e.g., ablation variants A/B/C/D), pass multiple JSONL logs:

```bash
python tools/memory_manifold_analysis.py \
  --logs runs/ablation_A.jsonl runs/ablation_C.jsonl runs/ablation_D.jsonl \
  --module js rank \
  --out results/ablation_manifold/
```

Output adds a `{run_name}` column to all CSVs. Enables comparing how different
memory rules (delta vs. hebbian vs. moneta) affect the manifold geometry of M_l
at matched training steps.

This is the intended use case for task_e34a74 (Dolmino ablation suite) — the
manifold analysis becomes the interpretability layer on top of the ablation PPL
numbers.
