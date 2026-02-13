# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NL_Hecate is a specification-first implementation of the Nested Learning (NL) research program from the Mirrokni/Behrouz research group at Google Research. It implements self-modifying neural networks where optimization IS the forward pass — no train/eval distinction, no external optimizer, no epochs.

**Current phase**: v0.4.0 specifications complete. Implementation has not begun. Next step is the Phase 0 Enzyme spike (2-week proof-of-concept).

**Name origin**: Hecate — goddess of crossroads and thresholds — standing at the boundary between conventional ML frameworks and what the NL papers describe.

## Architecture: Three Layers

```text
Layer 3: Python — PyO3 bindings, orchestration, configuration. No math.
Layer 2: Rust   — All math, control flow, Enzyme AD. Trait system enforces valid compositions.
Layer 1: CUDA   — Kernel pairs (forward + backward). Opaque to Enzyme. Hardware-specific.
```

**Kernel-Pair Pattern**: Every hot operation has three implementations: (1) Rust reference (portable, Enzyme-compatible), (2) CUDA forward kernel, (3) CUDA backward kernel with analytical gradients from papers. The spec envisions Enzyme chaining through pairs via `#[custom_vjp]` and `#[enzyme_opaque]`, but these annotations are **not yet exposed** in the Rust-Enzyme wrapper (they exist internally in Enzyme but are not surfaced). Currently the Rust integration only exposes `#[autodiff]` (expands to internal `#[rustc_autodiff]`) and `#[no_autodiff]`. The Phase 0 spike validates a workaround: manual chain-rule composition at kernel boundaries using FFI barriers as opaque regions, with Enzyme differentiating the Rust portions and hand-written backward kernels providing analytical gradients for the opaque CUDA portions. `#[custom_vjp]` and `#[enzyme_opaque]` remain planned future features — the architecture is designed for them but does not depend on them.

**Toolchain dependency chain**: LLVM version → Rust nightly → CUDA toolkit (all interdependent via Enzyme).

## Repository Structure

- `specs/contract.md` — Top-level v0.4.0 specification (read this first)
- `specs/algorithms/` — Memory update rules (MIRAS 4-knob framework), composition patterns, retention, parallelization, optimization, frequency scheduling
- `specs/infrastructure/` — Enzyme AD, scheduling (Conductor/Pulse), state lifecycle, context stream, distribution, attention, serving, compilation, precision, Track Zero
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

- **outer_loop_param**: W_K, W_V, W_Q, gates — persists across build, modified by Enzyme, serialized
- **inner_loop_state**: M (memory), S (momentum) — scoped to forward pass, NOT serialized
- **context_memory**: chunk boundary state — persists across forward calls, moved

### Differentiation (Two mechanisms composing via chain rule)

- **Enzyme AD**: Differentiates Rust code at LLVM IR level. Currently supported annotations: `#[autodiff]` (reverse/forward mode, expands to `#[rustc_autodiff]`), `#[no_autodiff]`. Planned but not yet available in Rust wrapper: `#[custom_vjp]`, `#[enzyme_opaque]`.
- **Hand-written backward kernels**: Analytical gradients from papers, opaque to Enzyme
- **Phase 0 strategy**: Since `#[custom_vjp]` is not yet available, the spike validates manual chain-rule composition — Enzyme differentiates Rust code on either side of a kernel boundary (FFI barrier), and hand-written backward kernels provide gradients for the opaque middle. The upstream gradient is passed as the seed to Enzyme's reverse-mode call, composing the full gradient in one shot.
- **Critical**: Inner-loop operations must allow outer-loop gradients to flow THROUGH them (the spec's `#[custom_vjp]` intent) — currently achieved via manual composition rather than annotation

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

## Implementation Roadmap (Track Zero)

1. **Phase 0** (2 weeks): Enzyme spike — prove Enzyme differentiates through Rust trait dispatch + manual chain-rule composition at kernel boundaries (see `spike/`)
2. **Track Zero-A** (2-4 weeks): Pure SWA attention, no memory — validate full Rust→Enzyme→CUDA→Python pipeline
3. **Track Zero-B**: Delta Rule + MAG — validate gradient flow through memory
4. **Phase 2**: CMS k=2 — verify multi-level scheduling
5. **Phase 3**: Full design space k=4 — combinatorial validation

See `specs/infrastructure/track_zero/00_track_zero.md`.

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

- `nl_code_smells` (47) — all code smell constraints
- `nl_reframings` (33) — PyTorch→NL concept mappings
- `nl_optimizers` (14) — optimizer catalog
- `nl_toolchain` (15) — Enzyme/Rust/CUDA toolchain notes
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
