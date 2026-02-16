# NL_Hecate Roadmap

This document provides the single authoritative view of project progress and upcoming work. It replaces the scattered "Phase N" naming that was reused across different workstreams.

## Naming Convention

The project is organized into **Stages**, each containing **Milestones**. No two milestones share a name.

```
Stage 0: Foundation       — Toolchain, spike, pipeline validation
Stage 1: Algorithm Core   — Memory rules, compositions, scheduling, parallelization
Stage 2: Production Infra — Multi-GPU, serving, compilation, deployment
Stage 3: Extensions       — Pluggable retention, M3 optimizer, CMS variants
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

### MIRAS Memory Rules (8/8)

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

### Parallelization Strategies (5/5 + 1 stub)

| Milestone | Strategy | Scope | Tests | Status |
|-----------|----------|-------|-------|--------|
| **S1-M17** | Chunkwise GD | Universal (all 8 rules) | 93 | COMPLETE |
| **S1-M18** | Associative Scan | Hebbian (exact) + Titans momentum | 22 | COMPLETE |
| **S1-M19** | TNT Hierarchical | Global + local memories | 42 | COMPLETE |
| **S1-M20** | Lattice GLA | Lattice OSR + Trellis specialized | 17 | COMPLETE |
| **S1-M21** | Atlas Parallel | Stub (awaiting Atlas Omega rule) | 6 | COMPLETE |

### PyO3 Bindings

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S1-M22: Python Bindings** | All 8 rules, 3 compositions, 10 kwargs. PyTorch baseline validation. | 27 Py | COMPLETE |

**Stage 1 totals**: 778 Rust + 27 Python = **805 tests, 0 failures**.

---

## Stage 2: Production Infrastructure (IN PROGRESS)

Move from validated algorithms to deployable system. Each milestone maps to a spec file.

### S2-M1: Compilation Strategy (PHASE 1-3 COMPLETE, PHASE 4 COMPLETE)

**Spec**: `specs/infrastructure/compilation/00_compilation.md`

Formalize the two-domain compilation boundary that already exists in practice:
- Enzyme handles Rust code (via `#[autodiff_reverse]`)
- Hand-written CUDA backward kernels handle opaque GPU operations
- No whole-model graph tracing (torch.compile cannot trace NL's inner loop)
- Architecture-specific `.cubin` + portable `.ptx` fallback

**What was delivered**: CUDA kernel pairs for all 3 matrix-based memory rules (DeltaRule, TitansLMM, HebbianRule). Each has forward + backward kernels with analytical gradients from the papers. All fp32. Single-block design (Grid=1, Block=min(d², 1024)) with shared memory for M recurrence. Projections and gates remain in Rust (Enzyme-differentiable); only the sequential inner loop is CUDA.

| Phase | Kernels | Tests | Status |
|-------|---------|-------|--------|
| Phase 1: DeltaRule | `delta_forward.cu`, `delta_backward.cu` | 11 | COMPLETE |
| Phase 2: TitansLMM | `titans_forward.cu`, `titans_backward.cu` | 6 | COMPLETE |
| Phase 3: HebbianRule | `hebbian_forward.cu`, `hebbian_backward.cu` | 7 | COMPLETE |
| Phase 4: Integration | Composition-level CUDA parity tests | 5 | COMPLETE |

**Remaining deliverables**:
- [x] CUDA kernel pairs for memory rule hot paths (Delta Rule, Titans LMM, Hebbian)
- [ ] Architecture dispatch (sm_86, sm_90 etc.)
- [ ] Compilation documentation (build matrix, toolchain requirements)
- [ ] `.cubin`/`.ptx` packaging

**Dependencies**: None (builds on existing kernel-pair pattern)
**Estimated scope**: Large (kernel pairs done, architecture dispatch and packaging remain)

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

### S2-M4: Edge Deployment

**Spec**: `specs/infrastructure/edge_deployment/00_edge_deployment.md`

Deploy micro models (d <= 128) on CPU with zero external dependencies. The inner loop enables on-device adaptation without retraining.

**Deliverables**:
- [ ] Profile 1: Inner-loop only (no Enzyme on target, pre-computed outer-loop weights)
- [ ] Profile 2: Full NL (Enzyme on target for fine-tuning)
- [ ] Profile 3: WASM (browser deployment)
- [ ] Target matrix validation: x86, aarch64, armv7, riscv64, wasm32
- [ ] Benchmark: 18k tok/s target on single CPU thread (d=64-128)

**Dependencies**: S2-M3 (serving primitives), Rust cross-compilation
**Estimated scope**: Small (primarily empirical validation, no new algorithms)

---

## Stage 3: Extensions (NOT STARTED)

Algorithm-level extensions that enrich the design space. None are blockers for Stage 2.

### S3-M1: Pluggable Retention Trait

**Specs**: `specs/algorithms/retention_mechanisms/00_interface.md`, `03_elastic_net.md`, `04_f_divergence.md`

Currently retention is baked into each memory rule. This milestone extracts it into a composable trait.

**Deliverables**:
- [ ] `RetentionMechanism` trait with `compute_retention()` and `apply_retention()`
- [ ] Elastic Net retention (L1+L2 with hard thresholding for sparsity)
- [ ] f-Divergence framework (proves L2, KL, TV are all special cases)
- [ ] Existing rules refactored to use pluggable trait (backward compatible)

**Dependencies**: None
**Estimated scope**: Medium (trait extraction + refactor of 8 rules)

---

### S3-M2: M3 Multi-Scale Optimizer

**Spec**: `specs/algorithms/optimization_machinery/02_m3.md`

Apply CMS to the optimizer itself — k momentum accumulators at k frequency levels. Frozen levels buffer gradient errors; active levels apply combined gradient.

**Deliverables**:
- [ ] Multi-scale gradient accumulation with error buffer reset
- [ ] M3 + Newton-Schulz integration
- [ ] Validation: M3 improves convergence over flat SGD on k=4

**Dependencies**: CMS, Conductor (both complete)
**Estimated scope**: Small

---

### S3-M3: CMS Deployment Variants

**Spec**: `specs/algorithms/frequency_scheduling/02_cms_variants.md`

Five deployment patterns beyond the basic single-CMS we have today.

**Deliverables**:
- [ ] Variant 1: Basic CMS (current — done)
- [ ] Variant 2: Nested CMS (CMS on both model and optimizer)
- [ ] Variant 3: Sequential CMS (frequency spectrum deepens per block)
- [ ] Variant 4: Independent CMS (each block has own schedule)
- [ ] Variant 5: Hybrid CMS (mix CMS and standard blocks)

**Dependencies**: S3-M2 (M3 needed for Variant 2)
**Estimated scope**: Small (mostly configuration, no new primitives)

---

### S3-M4: Atlas Omega Rule

**Spec**: `specs/algorithms/memory_update_rules/` (future, referenced in parallelization)

Implement Atlas Omega as 9th MIRAS variant, enabling the Atlas Parallel strategy (S1-M21 stub).

**Deliverables**:
- [ ] AtlasOmega struct implementing MemoryRule trait
- [ ] State-independent omega function (already stubbed in `atlas_parallel.rs`)
- [ ] Full parallelization via atlas_parallel_forward (currently `unimplemented!()`)
- [ ] FD gradient checks + convergence tests

**Dependencies**: S1-M21 (Atlas stub, complete)
**Estimated scope**: Medium

---

### S3-M5: Dynamic Frequency Scheduling

**Spec**: `specs/algorithms/frequency_scheduling/01_frequency_scheduler.md`

Currently CMS frequencies are hardcoded `[1, 8, 64, 512]`. This milestone adds data-dependent level activation.

**Deliverables**:
- [ ] Learned frequency gates (when does Level 2 fire?)
- [ ] Frequency adaptation during training (not just fixed schedule)
- [ ] Validation: dynamic scheduling outperforms fixed on variable-rate data

**Dependencies**: CMS k=4, Conductor (both complete)
**Estimated scope**: Medium

---

## Dependency Graph

```
Stage 0: Foundation ─────────────────────────── COMPLETE
    │
    ▼
Stage 1: Algorithm Core ─────────────────────── COMPLETE (805 tests)
    │
    ├─► S2-M1: Compilation Strategy ─────────── (kernel pairs COMPLETE)
    │       │
    │       ├─► S2-M2: Multi-GPU Distribution
    │       │
    │       └─► S2-M3: Serving
    │               │
    │               └─► S2-M4: Edge Deployment
    │
    ├─► S3-M1: Pluggable Retention ─────────── (independent)
    │
    ├─► S3-M2: M3 Optimizer ────────────────── (independent)
    │       │
    │       └─► S3-M3: CMS Variants
    │
    ├─► S3-M4: Atlas Omega Rule ────────────── (independent)
    │
    └─► S3-M5: Dynamic Frequency ───────────── (independent)
```

Stage 2 milestones are sequential (each builds on the last). Stage 3 milestones are independent of each other and can be done in any order. Stage 3 does not block Stage 2.

---

## Summary

| Stage | Milestones | Tests | Status |
|-------|-----------|-------|--------|
| Stage 0: Foundation | 3 | 57 + 47 + 98 | COMPLETE |
| Stage 1: Algorithm Core | 19 | 778 Rust + 27 Python | COMPLETE |
| Stage 2: Production Infra | 4 | 29 CUDA (S2-M1) | IN PROGRESS |
| Stage 3: Extensions | 5 | TBD | NOT STARTED |

**Current position**: Stage 2 in progress. S2-M1 kernel pairs complete (29 new CUDA tests: 11 Delta + 6 Titans + 7 Hebbian + 5 integration). Architecture dispatch and packaging remain. Total test count: 807 Rust + 27 Python + 29 CUDA = **863 tests**.
