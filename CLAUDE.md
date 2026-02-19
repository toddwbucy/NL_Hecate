# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NL_Hecate is a specification-first implementation of the Nested Learning (NL) research program from the Mirrokni/Behrouz research group at Google Research. It implements self-modifying neural networks where optimization IS the forward pass — no train/eval distinction, no external optimizer, no epochs.

**Current status**: Stages 0-1 complete (foundation + full algorithm core). 778 Rust + 27 Python tests passing. See `ROADMAP.md` for detailed milestone tracking and what's next (Stage 2: production infrastructure).

**Name origin**: Hecate — goddess of crossroads and thresholds — standing at the boundary between conventional ML frameworks and what the NL papers describe.

## Architecture: Three Layers

```text
Layer 3: Python — PyO3 bindings, orchestration, configuration. No math.
Layer 2: Rust   — All math, control flow, Wengert tape AD. Trait system enforces valid compositions.
Layer 1: CUDA   — Kernel pairs (forward + backward). Opaque to AD. Hardware-specific.
```

**Kernel-Pair Pattern**: Every hot operation has three implementations: (1) Rust reference (portable, AD-compatible), (2) CUDA forward kernel, (3) CUDA backward kernel with analytical gradients from papers. The Wengert tape (`core/src/tape.rs`) records operations during forward and replays in reverse for gradients. CUDA kernels are registered as opaque VJP blocks — the tape calls their hand-written backward kernels for gradient computation. This replaced the earlier Enzyme-based approach (archived to `Acheron/enzyme-archive/`).

## Repository Structure

- `specs/contract.md` — Top-level v0.4.0 specification (read this first)
- `specs/algorithms/` — Memory update rules (MIRAS 4-knob framework), composition patterns, retention, parallelization, optimization, frequency scheduling
- `specs/infrastructure/` — Differentiation (Wengert tape AD), scheduling (Conductor/Pulse), state lifecycle, context stream, distribution, attention, serving, compilation, precision, Track Zero
- `specs/constraints/` — 48 code smells (CS-01 through CS-48) and trait system safety
- `core/src/` — Rust core (stub, awaiting implementation)
- `core/kernels/` — CUDA kernels (stub)
- `python/nl_hecate/` — PyO3 bindings (stub)

## Key Concepts

### MIRAS Framework (4 Independent Knobs)

Every memory update rule is specified by four choices:
- **Memory Structure**: vector, matrix, 2-layer MLP
- **Attentional Bias**: L2, dot-product, Huber, l_p (the loss function)
- **Retention**: L2 decay, KL, elastic net, sphere normalization
- **Algorithm**: GD, GD+momentum, Newton-Schulz, FTRL

Eight named variants: Titans LMM, Delta Rule, Hebbian, MONETA, YAAD, MEMORA, Lattice OSR, Trellis. All implement the same `MemoryRule` trait (`specs/algorithms/memory_update_rules/00_interface.md`).

### Composition Patterns (MAC/MAG/MAL)

Three ways to combine memory with attention: MAC (memory-attention-memory, sequential), MAG (memory gates attention, parallel), MAL (memory preprocesses for attention, sequential).

### Continuous Memory Systems (CMS)

4 frequency levels (k=4): every token fires Level 0, every 8th fires Level 1, every 64th Level 2, every 512th Level 3. The **Conductor** generates a **Pulse** struct each step that tells every component which levels are active.

### State Lifetimes (Rust ownership enforced)

- **outer_loop_param**: W_K, W_V, W_Q, gates — persists across build, modified by AD, serialized
- **inner_loop_state**: M (memory), S (momentum) — scoped to forward pass, NOT serialized
- **context_memory**: chunk boundary state — persists across forward calls, moved

### Differentiation (Wengert tape + opaque VJP blocks)

- **Wengert tape** (`core/src/tape.rs`): Records operations during forward pass, replays in reverse for gradients. Opt-in via `with_tape()` (CS-40). All intermediates stored in arena (CS-42).
- **Opaque VJP blocks**: CUDA kernel pairs and memory rules register as opaque blocks on the tape. Each provides a hand-written backward function with analytical gradients from the papers.
- **OpaqueVjp trait** (`core/src/tape.rs`): All memory rules implement this trait, connecting the tape to their backward adapters via `opaque_key()` registry lookup.
- **Critical**: Inner-loop operations allow outer-loop gradients to flow THROUGH them via the opaque VJP mechanism — the tape orchestrates the chain rule across opaque boundaries.
- **History**: Originally designed around Enzyme (LLVM-level AD). Enzyme was archived to `Acheron/enzyme-archive/` after the Wengert tape proved more practical (no custom toolchain, no ICE crashes).

### Numerical Precision

Inner loop MUST be fp32 (non-negotiable). bf16 drift corrupts memory after ~100 steps. Attention uses bf16 (standard FlashAttention). Error buffers are unconditionally fp32.

## Constraints (Code Smells)

48 enforced constraints organized as:
- **Ontological** (CS-01–09, 20, 21, 37, 38): No MemoryModule class, no mode distinction, "levels" not "layers", "build" not "train"
- **Mode & Phase** (CS-10–17, 19): No train/eval, no epochs, forward code identical in all phases
- **Structural** (CS-18, 22–26, 31, 32): Forward pass IS the only API, observe-then-advance (CS-32)
- **MIRAS** (CS-33–36): Don't restrict 4-knob combinations
- **Infrastructure** (CS-39–48): Clamp learnable decay, opt-in AD, no DDP, gradient checkpointing hurts NL

Full index: `specs/constraints/code_smells/00_index.md`

## Implementation Roadmap

See `ROADMAP.md` for the full milestone-level roadmap with dependency graph.

**Completed**: Stage 0 (Foundation) and Stage 1 (Algorithm Core) — all 22 milestones.
**Next**: Stage 2 (Production Infrastructure) — compilation, multi-GPU, serving, edge deployment.
**Optional**: Stage 3 (Extensions) — pluggable retention, M3 optimizer, CMS variants, Atlas Omega.

## Spec File Conventions

Every spec file has a `CONTRACT` header block with: Purpose, Expects, Guarantees, Cost, Trade-off, Position, Source. All equations trace to specific papers. Pseudocode uses Rust-like syntax with trait bounds.

## HADES: Paper Traceability via Semantic Graph RAG

Every spec traces to paper equations. Use the `/hades` skill (or `hades` CLI) with the `NL` database to look up source material.

**Database**: `NL` — 73 collections, 1,699 documents across 7 decomposed papers + cross-cutting NL collections.

### Per-Paper Collections

Each core paper is decomposed into `{prefix}_equations`, `{prefix}_definitions`, `{prefix}_abstractions`, `{prefix}_axioms`, `{prefix}_lineage`:

| Prefix | Paper | ArXiv ID |
|---|---|---|
| `titans_` | Titans: Learning to Memorize at Test Time | 2501.00663 |
| `miras_` | MIRAS: It's All Connected | 2504.13173 |
| `hope_` | HOPE / Nested Learning | 2512.24695 |
| `lattice_` | Lattice: Learning to Efficiently Compress Memory | 2504.05646 |
| `atlas_` | ATLAS: Learning to Optimally Memorize | 2505.23735 |
| `tnt_` | TNT: Improving Chunkwise Training | 2511.07343 |
| `trellis_` | Trellis: Compress Key-Value Memory | 2512.23852 |

### Cross-Cutting Collections

- `nl_code_smells` (48) — all code smell constraints (CS-01 through CS-48)
- `nl_reframings` (33) — PyTorch→NL concept mappings
- `nl_optimizers` (14) — optimizer catalog
- `nl_toolchain` (15) — Rust/CUDA toolchain notes (includes historical Enzyme notes)
- `hecate_specs` (48) — mirror of spec files
- `paper_edges` (94) — graph edges connecting papers

### Common Queries

```bash
# Semantic search across ingested NL papers
hades --database NL db query "momentum accumulator in inner loop"

# Search within a specific paper
hades --database NL db query "chunkwise approximation error" --paper 2511.07343

# Look up a specific equation collection
hades --database NL db export titans_equations | head -20

# Trace a code smell back to its source
hades --database NL db aql "FOR doc IN nl_code_smells FILTER doc.id == 'CS-32' RETURN doc"

# Graph traversal — find what a paper connects to
hades --database NL db graph neighbors --start "arxiv_metadata/2504.13173" --graph paper_graph
```

## When Working in This Repo

- **Read `specs/contract.md` first** — it's the architectural overview and entry point
- **Every concept traces to a paper** — if something seems invented, check the source field
- **Compile-time safety is central** — invalid MIRAS combinations should be type errors, not runtime panics
- **No DDP** — NL requires CMS-aware gradient sync (only active levels synchronized)
- **No DataLoader** — replaced by ContextStream (continuous, no epochs, cursor serialized with checkpoint)
- **Terminology matters**: "build" not "train", "levels" not "layers", "context" not "dataset"
