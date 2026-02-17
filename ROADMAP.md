# NL_Hecate Roadmap

This document provides the single authoritative view of project progress and upcoming work. It replaces the scattered "Phase N" naming that was reused across different workstreams.

## Naming Convention

The project is organized into **Stages**, each containing **Milestones**. No two milestones share a name.

```
Stage 0: Foundation       — Toolchain, spike, pipeline validation
Stage 1: Algorithm Core   — Memory rules, compositions, scheduling, parallelization
Stage 2: Production Infra — Multi-GPU, serving, compilation, deployment
Stage 3: Extensions       — Pluggable retention, M3 optimizer, CMS variants
Stage 4: MVP              — Build a model, serve it locally
```

---

## Stage 0: Foundation (COMPLETE)

Validate that the toolchain works before adding NL complexity.

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S0-M1: Enzyme Spike** | Prove Enzyme differentiates through Rust trait dispatch. Pin toolchain. | 57 | COMPLETE |
| **S0-M2: SWA Pipeline** | Pure SWA attention, no memory. Rust + Enzyme + CUDA + Python end-to-end. | 47 | COMPLETE |
| **S0-M3: Memory Intro** | Delta Rule + MAG. Gradient flow through memory validated. PyO3 bindings. | 98 | COMPLETE |

**Toolchain pinned**: rustc `d7daac06` (1.95.0-nightly), fat LTO, `RUSTFLAGS="-Zautodiff=Enable"`.

---

## Stage 1: Algorithm Core (COMPLETE)

All algorithms from the NL paper suite, validated with FD gradient checks and convergence tests.

### CMS Scheduling

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S1-M1: CMS k=2** | Two-level frequency scheduling. Conductor, Pulse, ContextState, ErrorBuffer. Level 0 every step, Level 1 every 8th. | 140 | COMPLETE |
| **S1-M2: CMS k=2 Validation** | k=2 beats k=1 by 62% on multiscale data. Gate diagnostics. | 142 | COMPLETE |
| **S1-M3: CMS k=4** | Four levels [1, 8, 64, 512]. Output normalization (1/sqrt(k) for k>2). | 192 | COMPLETE |

### MIRAS Memory Rules (9/9)

| Milestone | Rule | Family | Memory Structure | Tests | Status |
|-----------|------|--------|-----------------|-------|--------|
| **S1-M4** | Delta Rule | Titans | d x d matrix | (included in S0-M3) | COMPLETE |
| **S1-M5** | Titans LMM | Titans | d x d matrix + momentum S | (included in S0-M3) | COMPLETE |
| **S1-M6** | Hebbian | Titans | d x d matrix (no error term) | 226 | COMPLETE |
| **S1-M7** | MONETA | MIRAS | 2-layer MLP (W1, W2) | 269 | COMPLETE |
| **S1-M8** | YAAD | MIRAS | 2-layer MLP + Huber loss | (PR #16) | COMPLETE |
| **S1-M9** | MEMORA | MIRAS | 2-layer MLP + KL retention | (PR #17) | COMPLETE |
| **S1-M10** | Lattice OSR | Compression | m unit vectors on S^(d-1) | 400 | COMPLETE |
| **S1-M11** | Trellis | Compression | Two-pass KV (S_K, S_V) | 466 | COMPLETE |
| **S3-M4** | Atlas Omega | Atlas | d x d matrix + learned omega | (PR #35) | COMPLETE |

### Composition Patterns (3/3)

| Milestone | Pattern | Description | Tests | Status |
|-----------|---------|-------------|-------|--------|
| **S1-M12** | MAG | Memory gates attention (parallel) | (included above) | COMPLETE |
| **S1-M13** | MAL | Memory preprocesses attention (sequential, residual) | 528 | COMPLETE |
| **S1-M14** | MAC | Memory-attention-memory (full causal) | 528 | COMPLETE |

### State & Streaming

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S1-M15: ContextStream** | VecStream, StreamCursor, BoundaryEvent. Atomic checkpoint/restore. | 551 | COMPLETE |
| **S1-M16: Stability Sweep** | 100K-step stability tests across all rule/composition/k combos. | 590 | COMPLETE |

### Parallelization Strategies (6/6)

| Milestone | Strategy | Scope | Tests | Status |
|-----------|----------|-------|-------|--------|
| **S1-M17** | Chunkwise GD | Universal (all 9 rules) | 93 | COMPLETE |
| **S1-M18** | Associative Scan | Hebbian (exact) + Titans momentum | 22 | COMPLETE |
| **S1-M19** | TNT Hierarchical | Global + local memories | 42 | COMPLETE |
| **S1-M20** | Lattice GLA | Lattice OSR + Trellis specialized | 17 | COMPLETE |
| **S1-M21** | Atlas Parallel | Batch omega + sequential recurrence (S3-M4) | (PR #35) | COMPLETE |

### PyO3 Bindings

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S1-M22: Python Bindings** | All 8 rules, 3 compositions, 10 kwargs. PyTorch baseline validation. | 27 Py | COMPLETE |

**Stage 1 totals**: 778 Rust + 27 Python = **805 tests, 0 failures**.

---

## Stage 2: Production Infrastructure (COMPLETE)

Move from validated algorithms to deployable system. Each milestone maps to a spec file.

### S2-M1: Compilation Strategy ✅

**Spec**: `specs/infrastructure/compilation/00_compilation.md`

Formalize the two-domain compilation boundary that already exists in practice:
- Enzyme handles Rust code (via `#[autodiff_reverse]`)
- Hand-written CUDA backward kernels handle opaque GPU operations
- No whole-model graph tracing (torch.compile cannot trace NL's inner loop)
- Multi-architecture fat binary (sm_86/89/90 SASS + PTX fallback)

**What was delivered**: CUDA kernel pairs for all 3 matrix-based memory rules (DeltaRule, TitansLMM, HebbianRule). Each has forward + backward kernels with analytical gradients from the papers. All fp32. Single-block design (Grid=1, Block=min(d², 1024)) with shared memory for M recurrence. Projections and gates remain in Rust (Enzyme-differentiable); only the sequential inner loop is CUDA. Multi-architecture fat binary via `-gencode` flags covers sm_86/89/90 with PTX fallback for future GPUs. Runtime `Backend` enum + `detect_gpu()` for diagnostics. `force_rust_reference()` override for testing.

| Phase | Kernels | Tests | Status |
|-------|---------|-------|--------|
| Phase 1: DeltaRule | `delta_forward.cu`, `delta_backward.cu` | 11 | COMPLETE |
| Phase 2: TitansLMM | `titans_forward.cu`, `titans_backward.cu` | 6 | COMPLETE |
| Phase 3: HebbianRule | `hebbian_forward.cu`, `hebbian_backward.cu` | 7 | COMPLETE |
| Phase 4: Integration | Composition-level CUDA parity tests | 5 | COMPLETE |
| Phase 5: Arch dispatch | `Backend` enum, `GpuInfo`, multi-arch build | ~13 | COMPLETE |

**Deliverables**:
- [x] CUDA kernel pairs for memory rule hot paths (Delta Rule, Titans LMM, Hebbian)
- [x] Architecture dispatch (sm_86, sm_89, sm_90 + PTX fallback)
- [x] `.cubin`/`.ptx` packaging (fat binary via multi-gencode)
- [x] Compilation documentation (`docs/build_matrix.md`)

**Dependencies**: None (builds on existing kernel-pair pattern)

---

### S2-M2: Multi-GPU Distribution ✅

**Spec**: `specs/infrastructure/distribution/00_multi_gpu.md`

NL cannot use DDP — CMS means different levels fire at different frequencies, so naive allreduce wastes bandwidth on frozen-level gradients.

**Deliverables**:
- [x] CMS-aware gradient synchronization (only active levels allreduce)
- [x] State isolation: OuterLoopParam synced across ranks, InnerLoopState rank-local
- [x] Conductor synchronization across ranks (deterministic Pulse from same step counter)
- [x] Throughput reporting (per-GPU, average vs worst-case) — CS-43/CS-44 compliant

**Implementation**: `core/src/distributed.rs` behind `#[cfg(feature = "distributed")]`
- ProcessGroup trait + MockProcessGroup (18 tests, all passing)
- sync_gradients: 6 SWA allreduces always + 9 per active level, ~1.14 level-allreduces/step for k=4
- ThroughputTracker with worst-case tracking
- distributed_step: full forward→backward→sync→apply→advance composition

**Dependencies**: S2-M1 (kernel pairs needed for GPU-resident compute)
**Estimated scope**: Medium

---

### S2-M3: Serving Non-Stationary Models ✅

**Spec**: `specs/infrastructure/serving/00_serving.md`

NL models self-modify during inference (inner loop runs at test time). Serving must handle stateful sessions with per-user context memory.

**Deliverables**:
- [x] Session struct: new_test, new_stream, process_chunk, checkpoint, restore
- [x] Two serving modes: Test (bounded context) and Stream (unbounded, ContextStream)
- [x] Per-token latency does NOT grow with context length (test_long_session_latency_stable)
- [x] Atomic checkpoint/resume with pulse_id verification
- [x] LatencyTracker with average, worst, p99 for SLA validation
- [x] 18 tests (LatencyTracker 3, Session Test 4, Session Stream 3, Checkpoint 4, Integration 4)

**Implementation**: `core/src/serving.rs` (feature-gated: `serving`)
**Dependencies**: S1-M15 (ContextStream), S2-M1 (compilation)
**Estimated scope**: Medium

---

### S2-M4: Edge Deployment ✅

**Spec**: `specs/infrastructure/edge_deployment/00_edge_deployment.md`

Deploy micro models (d <= 128) on CPU with zero external dependencies. The inner loop enables on-device adaptation without retraining.

**Deliverables**:
- [x] Profile 1: Inner-loop only (no Enzyme on target, pre-computed outer-loop weights)
- [x] Profile 2: Full NL (forward + backward + gradient apply on x86_64)
- [x] Profile 3: WASM (wasm32-unknown-unknown cross-compilation validated)
- [x] Target matrix validation: x86_64 native + wasm32 cross-compile
- [x] Benchmark: ~34k tok/s on x86_64 for d=64 (exceeds 18k target)
- [x] `#![feature(autodiff)]` gated behind `enzyme` feature (edge/wasm builds don't need nightly autodiff)

**Implementation**: `core/src/edge.rs` (feature-gated: `edge`), `core/benches/edge_bench.rs` (Criterion)
- EdgeConfig + EdgeModel: zero-dependency wrapper around cms_forward/cms_backward
- Criterion benchmarks: d=64/128/256 throughput sweep, adaptation cost, latency vs seq_len
- 11 integration tests + 9 unit tests (profiles, size, throughput, memory, cross-compile)

**Benchmark results** (x86_64, release):

| Dimension | Throughput (tok/s) | Forward latency | Deployment size |
|-----------|-------------------|-----------------|-----------------|
| d=64      | ~34,000           | ~471 µs         | ~230 KB         |
| d=128     | ~9,600            | ~1.67 ms        | ~530 KB         |
| d=256     | ~1,600            | ~10 ms          | ~1.6 MB         |

**Dependencies**: S2-M3 (serving primitives), Rust cross-compilation
**Estimated scope**: Small (primarily empirical validation, no new algorithms)

---

## Stage 3: Extensions (COMPLETE)

Algorithm-level extensions that enrich the design space. None are blockers for Stage 2.

### S3-M1: Pluggable Retention ✅

**Specs**: `specs/algorithms/retention_mechanisms/00_interface.md`, `03_elastic_net.md`, `04_f_divergence.md`

Retention was fused inline in each memory rule's `step()`. This milestone extracts it into standalone free functions dispatched via `RetentionKind` enum (matching existing `MemoryRuleKind`/`CompositionKind` patterns), enabling cross-rule retention swapping (e.g. DeltaRule+ElasticNet) for CS-36 compliance.

**What was delivered**: `core/src/retention.rs` (450 lines) with `RetentionKind` enum (L2WeightDecay, KLDivergence, ElasticNet, SphereNormalization), `RetentionConfig` struct, `default_retention()` mapping, and 6 free functions (`l2_apply_retention`, `l2_retention_gradient`, `l2_decoupled_gradient`, `kl_apply_retention[_inplace]`, `elastic_net_apply`, `sphere_project_and_normalize[_inplace]`). All 8 rules refactored. `RetentionKind` added to `MAGConfig`. PyO3 `retention` kwarg. In-place variants for hot-loop callers (lattice_osr, memora, trellis) eliminate per-iteration heap allocations.

**Deliverables**:
- [x] `RetentionKind` enum + free function dispatch (L2, KL, ElasticNet, Sphere)
- [x] Elastic Net retention (L1+L2 with soft thresholding for sparsity)
- [x] In-place variants for zero-alloc hot paths (`kl_apply_retention_inplace`, `sphere_project_and_normalize_inplace`)
- [x] All 8 rules refactored to use pluggable functions (bit-identical backward compatibility)
- [x] `RetentionKind` field on `MAGConfig` (24 constructors + all test files updated)
- [x] PyO3 bindings (`retention` kwarg: "l2", "kl", "elastic_net", "sphere")
- [x] 22 dedicated retention tests + 17 inline unit tests

**Dependencies**: None
**PR**: #31

---

### S3-M2: M3 Multi-Scale Optimizer ✅

**Spec**: `specs/algorithms/optimization_machinery/02_m3.md`

Apply CMS to the optimizer itself — k momentum accumulators at k frequency levels. Frozen levels buffer gradient errors; active levels apply combined gradient.

**What was delivered**: `core/src/m3.rs` (~400 lines) with `M3Config` (k levels, per-level etas/thetas/weights/frequencies), `M3State` (k momentum accumulators + error buffers), `m3_step()` core function, and Newton-Schulz orthogonalization (`newton_schulz_5()`). Flatten/unflatten helpers for MAGParams ↔ flat Vec<f32>. `apply_weight_gradients_m3()` on MAGParams. `M3Config` field on `MAGConfig` (Option, default None = plain SGD). PyO3 `m3` kwarg.

**Deliverables**:
- [x] Multi-scale gradient accumulation with error buffer reset
- [x] M3 + Newton-Schulz integration (opt-in via `use_newton_schulz`)
- [x] Validation: M3 k=2 improves convergence over flat SGD
- [x] Flatten/unflatten roundtrip for MAGParams ↔ flat gradient vector
- [x] 20 tests (unit, error buffering, integration, edge cases)

**Dependencies**: CMS, Conductor (both complete)
**PR**: #32

---

### S3-M3: CMS Deployment Variants ✅

**Spec**: `specs/algorithms/frequency_scheduling/02_cms_variants.md`

Five deployment patterns beyond the basic single-CMS we have today. S3-M3 is configuration schema + validation — the multi-block execution engine is Stage 4 work.

**What was delivered**: `core/src/cms_variants.rs` (~250 lines) with `CMSVariant` enum (Basic, Nested, Sequential, Independent, Hybrid), `BlockConfig` struct, `MultiBlockConfig` struct with per-variant constructors and validation. Nested variant requires M3 config (enforced). Sequential validates non-decreasing k. Hybrid requires mix of CMS/non-CMS blocks. PyO3 `MultiBlockConfig` class.

**Deliverables**:
- [x] Variant 1: Basic CMS (single block from existing MAGConfig)
- [x] Variant 2: Nested CMS (validates M3 config on each block)
- [x] Variant 3: Sequential CMS (validates k non-decreasing)
- [x] Variant 4: Independent CMS (per-block independent schedules)
- [x] Variant 5: Hybrid CMS (mix CMS and standard blocks)
- [x] 15 tests (construction, validation, helpers)

**Dependencies**: S3-M2 (M3 needed for Variant 2)
**PR**: #32

---

### S3-M4: Atlas Omega Rule ✅

**Spec**: `specs/algorithms/memory_update_rules/` (referenced in parallelization)

Implement Atlas Omega as 9th MIRAS variant, enabling the Atlas Parallel strategy (S1-M21 stub).

**What was delivered**: `core/src/atlas_omega.rs` (~700 lines) with full forward + backward implementation. State-independent omega function: `omega(k,v) = W_omega @ silu(concat(k_mem, v_mem))` — doesn't depend on M, enabling batch precomputation. `atlas_parallel.rs` rewritten with `batch_compute_omega()` + sequential M/S recurrence. Full dispatch wiring across mag.rs, mal.rs, mac.rs, chunkwise_gd.rs, tnt.rs (with explicit carry-forward path instead of outer-product global update). `w_omega` field on `MemoryLevelParams` (Xavier init for Atlas, zero for others). PyO3 `"atlas"` / `"atlas_omega"` memory_rule. Integration sweep expanded to 9 rules.

**Deliverables**:
- [x] AtlasOmega struct implementing MemoryRule trait
- [x] State-independent omega function with silu activation
- [x] Full parallelization via atlas_parallel_forward (batch Phase 1 + sequential Phase 2)
- [x] Forward + backward with d_M/d_S recurrences + d_W_omega gradient through silu chain
- [x] Dispatch wiring: MAG, MAL, MAC, chunkwise_gd, TNT, retention, M3 flatten
- [x] 12 unit tests + chunkwise macro tests + TNT macro tests + integration sweep (9 rules × 3 compositions × 3 k-values)

**Dependencies**: S1-M21 (Atlas stub, complete)
**PR**: #35

---

### S3-M5: Dynamic Frequency Scheduling ✅

**Spec**: `specs/algorithms/frequency_scheduling/01_frequency_scheduler.md`

CMS frequencies were hardcoded `[1, 8, 64, 512]` with pure modular arithmetic. This milestone adds learned sigmoid gates per level that decide when each level fires based on input embeddings, not just the step counter.

**What was delivered**: `core/src/dynamic_freq.rs` (~420 lines) with `FrequencySchedule` enum (Fixed default, Learned), `LearnedFreqConfig` (threshold, annealing), `FreqGateCache` for backward. Per-level sigmoid gate: `freq_gate_l = sigmoid(embedded_mean @ w_freq_l + b_freq_l)`. Hard threshold forward (gate > 0.5), straight-through estimator backward (gradient flows through sigmoid as if threshold weren't there). Level 0 always forced active (spec invariant). Higher levels initialized with more negative `b_freq` bias ([0.0, -1.0, -2.0, -3.0]) to fire less often, matching fixed schedule's geometric spacing. Empty-vec pattern for Fixed mode (zero overhead). Gate surrogate gradient uses dot product of d_y_combined and y_per_level as proxy signal.

**Deliverables**:
- [x] Learned frequency gates (sigmoid per level, threshold decision)
- [x] Straight-through estimator for backward through hard threshold
- [x] Level 0 forced-active override (spec invariant preserved)
- [x] Annealing support (optional fixed→learned blending period)
- [x] `FrequencySchedule` enum on `MAGConfig` (default: Fixed, zero behavioral change)
- [x] `w_freq`/`b_freq` on `MemoryLevelParams` (empty vecs for Fixed, [d]+[1] per level for Learned)
- [x] CMS integration: gate override in `cms_forward`, surrogate gradient in `cms_backward`
- [x] PyO3 bindings (`frequency_schedule` kwarg: "fixed", "learned", or dict with threshold/anneal_steps)
- [x] 22 tests (unit 6, integration 6, gradient 4, edge 6)

**Dependencies**: CMS k=4, Conductor (both complete)
**PR**: #36

---

## Stage 4: MVP — Build & Serve (IN PROGRESS)

Build a model and serve it locally. Not production-scale — just the primitives working end-to-end.

### S4-M1: Weight Serialization ✅

Serde JSON serialization for MAGParams + MAGConfig. `save_checkpoint()` / `load_checkpoint()` in `core/src/model.rs`.

**PR**: #37

---

### S4-M2: Stateful PyO3 Bindings ✅

Expose the stateful CMS build loop API to Python. Without this, Python can't do real builds with persistent memory — each `mag_forward()` created a fresh `ContextState`.

**What was delivered**: 6 new pyclass types (Conductor, Pulse, ContextState, ErrorBufferList, VecStream, CMSForwardCache) + 4 new pyfunctions (cms_forward, cms_backward, save_checkpoint, load_checkpoint). ~180 lines added to `python/src/lib.rs`.

**PR**: #38

---

### S4-M3: Build Script ✅

Python-tier orchestration (CS-18) for building a model on text data. Byte-level tokenizer (vocab=256), stateful CMS loop via Conductor + VecStream, periodic and final checkpoint saving.

**File**: `python/build.py` (~150 lines)
**PR**: #38

---

### S4-M4: Serve Script ✅

Load a checkpoint and interactively generate text. Autoregressive byte generation with temperature-controlled sampling. `--use_cms` for memory-augmented generation, `--interactive` for REPL mode.

**File**: `python/serve.py` (~150 lines)
**PR**: #38

---

### S4-M5: Declared Checkpoint Format (PLANNED)

Current checkpoint serialization is serde default (JSON dump of raw structs). Post-MVP, add a declared format with:
- Schema version for forward-compatible migration
- Optional ContextState persistence (for resuming builds mid-stream)
- Optional Conductor state (step counter, stream cursor) for exact resume
- Binary format option (the JSON checkpoint for a real model will be large)

Outer-loop params only is correct for serving (memory reconstructs at test time). But build resume and model distribution need a versioned, documented format.

**Dependencies**: S4-M1 through S4-M4 (MVP must work first)
**Status**: NOT STARTED

---

## Dependency Graph

```
Stage 0: Foundation ─────────────────────────── COMPLETE
    │
    ▼
Stage 1: Algorithm Core ─────────────────────── COMPLETE (805 tests)
    │
    ├─► S2-M1: Compilation Strategy ─────────── COMPLETE
    │       │
    │       ├─► S2-M2: Multi-GPU Distribution ── COMPLETE
    │       │
    │       └─► S2-M3: Serving ──────────────── COMPLETE
    │               │
    │               └─► S2-M4: Edge Deployment ─ COMPLETE
    │
    ├─► S3-M1: Pluggable Retention ─────────── COMPLETE (PR #31)
    │
    ├─► S3-M2: M3 Optimizer ────────────────── COMPLETE (PR #32)
    │       │
    │       └─► S3-M3: CMS Variants ──────── COMPLETE (PR #32)
    │
    ├─► S3-M4: Atlas Omega Rule ────────────── COMPLETE (PR #35)
    │
    ├─► S3-M5: Dynamic Frequency ───────────── COMPLETE (PR #36)
    │
    └─► S4-M1: Weight Serialization ────────── COMPLETE (PR #37)
            │
            └─► S4-M2: Stateful PyO3 ────────── COMPLETE (PR #38)
                    │
                    ├─► S4-M3: Build Script ──── COMPLETE (PR #38)
                    │
                    ├─► S4-M4: Serve Script ──── COMPLETE (PR #38)
                    │
                    └─► S4-M5: Declared Ckpt ─── PLANNED
```

Stage 2 milestones are sequential (each builds on the last). Stage 3 milestones are independent of each other and can be done in any order. Stage 3 does not block Stage 2. Stage 4 depends on S4-M1 (serialization) for the full build→save→load→serve pipeline.

---

## Summary

| Stage | Milestones | Tests | Status |
|-------|-----------|-------|--------|
| Stage 0: Foundation | 3 | 57 + 47 + 98 | COMPLETE |
| Stage 1: Algorithm Core | 19 | 778 Rust + 27 Python | COMPLETE |
| Stage 2: Production Infra | 4 | 29 CUDA + 13 dispatch + 20 edge + 18 serving + 18 distributed | COMPLETE |
| Stage 3: Extensions | 5 | 22 retention + 35 M3/variants + 26 Atlas + 22 dynamic freq = 105 | COMPLETE |

| Stage 4: MVP Build & Serve | 5 (4 done, 1 planned) | 27 Python (existing, no regressions) | IN PROGRESS |

**Current position**: S0–S3 complete. S4 MVP delivered (M1–M4): can build a model on text and serve it locally. S4-M5 (declared checkpoint format) planned for post-MVP. Total test count: 940 Rust + 27 Python = **967 tests**.
