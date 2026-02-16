# NL_Hecate

**Spec-first Rust implementation of the Nested Learning research program**

Stage 0-2 complete, Stage 3 in progress | ~921 tests | Apache 2.0

---

## What This Is

NL_Hecate implements the [Nested Learning](https://arxiv.org/abs/2512.24695) research program from the Mirrokni/Behrouz group at Google Research. These papers describe self-modifying neural networks where **optimization IS the forward pass** — there is no train/eval distinction, no external optimizer, no epochs. The memory updates itself at every token, at test time, as part of inference.

This is a **specification-first** implementation: every component traces to a paper equation, and 48 enforced constraints (code smells CS-01 through CS-48) prevent architectural drift. The codebase replaces what PyTorch provides (autograd, optimizer, DataLoader, `model.train()`/`model.eval()`) with a single unified forward pass that handles both learning and inference.

**Name origin**: Hecate — goddess of crossroads and thresholds — standing at the boundary between conventional ML frameworks and what the NL papers describe.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python Tier   PyO3 bindings, orchestration, config     │
│                No math. Forward→backward→sync→apply.    │
├─────────────────────────────────────────────────────────┤
│  Rust Tier     All math, control flow, Enzyme AD        │
│                Trait system enforces valid compositions  │
├─────────────────────────────────────────────────────────┤
│  CUDA Tier     Kernel pairs (forward + backward)        │
│                Opaque to Enzyme. Hand-written gradients. │
└─────────────────────────────────────────────────────────┘
```

**Kernel-pair pattern**: Every hot operation has three implementations — (1) Rust reference (portable, Enzyme-differentiable), (2) CUDA forward kernel, (3) CUDA backward kernel with analytical gradients from the papers. Enzyme differentiates the Rust code; CUDA kernels are opaque (compiled to SASS by nvcc). Gradients compose via manual chain rule at the FFI boundary.

**Differentiation**: [Enzyme](https://enzyme.mit.edu/) operates at the LLVM IR level, differentiating Rust code through trait dispatch, closures, and dynamic indexing. The `enzyme` toolchain is a custom nightly Rust compiler built with `llvm.enzyme = true`.

## Key Concepts

### [Miras](https://arxiv.org/abs/2504.13173) Framework (4 Independent Knobs)

Every memory update rule is specified by four orthogonal choices:

| Knob | Options |
|------|---------|
| **Memory Structure** | vector, d x d matrix, 2-layer MLP |
| **Attentional Bias** | L2, dot-product, Huber, l_p |
| **Retention** | L2 decay, KL divergence, elastic net, sphere normalization (all pluggable via `RetentionKind`) |
| **Algorithm** | GD, GD+momentum, Newton-Schulz, FTRL |

Eight named variants implement specific knob combinations: **Titans LMM**, **Delta Rule**, **Hebbian**, **MONETA**, **YAAD**, **MEMORA**, **Lattice OSR**, **Trellis**.

### [Titans](https://arxiv.org/abs/2501.00663) Composition Patterns

| Pattern | Description |
|---------|-------------|
| **MAG** | Memory gates attention output via sigmoid (parallel branches) |
| **MAL** | Memory preprocesses input for attention (sequential, residual) |
| **MAC** | Memory provides causal context, attention processes assembled input |

### Continuous Memory System (CMS)

4 frequency levels with periods [1, 8, 64, 512]. A **Conductor** generates a **Pulse** each step declaring which levels are active. Level 0 fires every token; Level 3 fires every 512th. Output normalization: 1/sqrt(k) for k > 2.

### State Lifetimes (Rust ownership enforced)

| Lifetime | Examples | Scope |
|----------|----------|-------|
| `outer_loop_param` | W_K, W_V, W_Q, gates | Persists across build, modified by Enzyme, serialized |
| `inner_loop_state` | M (memory), S (momentum) | Scoped to forward pass, NOT serialized |
| `context_memory` | Chunk boundary state | Persists across forward calls, moved |

## What's Implemented

| Stage | Description | Tests | Status |
|-------|-------------|-------|--------|
| **Stage 0** | Foundation — Enzyme spike, SWA pipeline, Delta Rule + MAG | 202 | Complete |
| **Stage 1** | Algorithm Core — all 8 Miras rules, 3 compositions, CMS k=1/2/4, 5 parallelization strategies, ContextStream, 100K stability sweep, PyO3 bindings | 805 | Complete |
| **Stage 2** | Production Infra — CUDA kernel pairs, multi-GPU sync, serving, edge deployment, architecture dispatch | ~131 | Complete |
| **Stage 3** | Extensions — pluggable retention (done), M3 optimizer, CMS variants | 22 | In progress (1/5) |

See [ROADMAP.md](ROADMAP.md) for per-milestone detail and [PROGRESS_REPORT.md](PROGRESS_REPORT.md) for the executive summary.

## Quick Start

### Toolchain Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Rust toolchain | `enzyme` (nightly, rustc d7daac06) | Custom-built with `llvm.enzyme = true` |
| LLVM | 20.0+ (built from source with Enzyme) | Ships with the `enzyme` toolchain |
| CUDA Toolkit | 12.8+ | Required for `cuda` feature only |

### Build Commands

```bash
# Rust reference only (no GPU, no Enzyme)
cargo +enzyme build --release
cargo +enzyme test --release --lib --tests

# CUDA build (requires GPU + CUDA Toolkit)
cargo +enzyme build --release --features cuda
cargo +enzyme test --release --features cuda --lib --tests

# Enzyme AD build (gradient tests)
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme build --release --features enzyme
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --release --features enzyme --lib --tests

# Full build (all features)
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --release \
  --features "cuda,distributed,serving,edge,enzyme" \
  --lib --tests

# Edge / WASM
cargo +enzyme build --release --features edge
cargo +enzyme build --release --features edge --target wasm32-unknown-unknown --lib
```

### Feature Flags

| Feature | What it enables |
|---------|----------------|
| `cuda` | GPU kernel dispatch (sm_86/89/90 + PTX fallback) |
| `enzyme` | Enzyme AD (`#[autodiff_reverse]` annotations) |
| `distributed` | CMS-aware multi-GPU gradient sync |
| `serving` | Session management, latency tracking, checkpoint/restore |
| `edge` | Zero-dependency micro model deployment (~34k tok/s at d=64) |
| `internal` | Exposes `gradient` module publicly (for testing tools) |

See [docs/build_matrix.md](docs/build_matrix.md) for the full feature combination matrix and GPU architecture support.

## Repository Structure

```
NL_Hecate/
├── specs/                    # Specification suite (read specs/contract.md first)
│   ├── contract.md           #   Top-level architecture contract
│   ├── algorithms/           #   Memory rules, composition, retention, parallelization
│   ├── infrastructure/       #   Enzyme AD, scheduling, distribution, serving
│   └── constraints/          #   48 code smell constraints (CS-01 through CS-48)
├── core/                     # Rust core crate (nl-hecate-core)
│   ├── src/                  #   31 modules: rules, compositions, retention, CMS, dispatch, ...
│   ├── kernels/              #   8 CUDA kernels (4 forward + 4 backward)
│   ├── tests/                #   Integration test suites
│   └── benches/              #   Criterion benchmarks (edge throughput)
├── python/                   # PyO3 bindings (nl_hecate Python package, Maturin build)
│   ├── src/lib.rs            #   PyO3 module: all 8 rules + 3 compositions
│   └── tests/                #   27 Python tests
├── spike/                    # Phase 0 Enzyme spike (57 tests, preserved)
├── docs/                     # Build matrix, architecture dispatch docs
├── ROADMAP.md                # Milestone-level progress tracking
├── PROGRESS_REPORT.md        # Executive summary with key discoveries
└── LICENSE                   # Apache 2.0
```

## Paper References

All components trace to equations in these papers:

| Paper | ArXiv | What NL_Hecate uses from it |
|-------|-------|-----------------------------|
| **Titans**: Learning to Memorize at Test Time | [2501.00663](https://arxiv.org/abs/2501.00663) | Delta Rule, Titans LMM, Hebbian, MAG/MAL/MAC |
| **Miras**: It's All Connected | [2504.13173](https://arxiv.org/abs/2504.13173) | 4-knob framework, MONETA, YAAD, MEMORA |
| **HOPE / Nested Learning** | [2512.24695](https://arxiv.org/abs/2512.24695) | Self-modifying forward pass, CMS frequency scheduling |
| **Lattice**: Learning to Efficiently Compress Memory | [2504.05646](https://arxiv.org/abs/2504.05646) | Lattice OSR, orthogonal state recurrence |
| **ATLAS**: Learning to Optimally Memorize | [2505.23735](https://arxiv.org/abs/2505.23735) | Atlas Omega rule (stub, Stage 3) |
| **TNT**: Improving Chunkwise Training | [2511.07343](https://arxiv.org/abs/2511.07343) | TNT hierarchical parallelization strategy |
| **Trellis**: Compress Key-Value Memory | [2512.23852](https://arxiv.org/abs/2512.23852) | Trellis two-pass KV compression |

## Terminology

NL_Hecate enforces ontological constraints to prevent conceptual drift from PyTorch conventions:

- **Levels**, not "layers" — CMS frequency hierarchy (Level 0, Level 1, ...)
- **Tiers**, not "layers" — codebase architecture (Python tier, Rust tier, CUDA tier)
- **Build**, not "train" — there is no training phase distinct from inference
- **Context**, not "dataset" — continuous streams, no epochs

See [specs/constraints/code_smells/00_index.md](specs/constraints/code_smells/00_index.md) for all 48 constraints.

## License

[Apache 2.0](LICENSE)
