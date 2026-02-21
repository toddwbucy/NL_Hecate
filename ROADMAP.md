# NL_Hecate Roadmap

This document provides the single authoritative view of project progress and upcoming work. It replaces the scattered "Phase N" naming that was reused across different workstreams.

## Naming Convention

The project is organized into **Stages**, each containing **Milestones**. No two milestones share a name.

```
Stage 0:  Foundation            — Toolchain, spike, pipeline validation
Stage 1:  Algorithm Core        — Memory rules, compositions, scheduling, parallelization
Stage 2:  Production Infra      — Multi-GPU, serving, compilation, deployment
Stage 3:  Extensions            — Pluggable retention, M3 optimizer, CMS variants
Stage 3b: Primitive Completeness — Remaining MIRAS knobs, self-referential architecture, Hope
Stage 4:  MVP                   — Build a model, serve it locally
```

---

## Stage 0: Foundation (COMPLETE)

Validate that the toolchain works before adding NL complexity.

| Milestone | Description | Tests | Status |
|-----------|-------------|-------|--------|
| **S0-M1: AD Spike** | Validated AD through Rust trait dispatch. Originally Enzyme, now archived to Acheron (Wengert tape superseded). | 57 | COMPLETE (archived) |
| **S0-M2: SWA Pipeline** | Pure SWA attention, no memory. Rust + CUDA + Python end-to-end. | 47 | COMPLETE |
| **S0-M3: Memory Intro** | Delta Rule + MAG. Gradient flow through memory validated. PyO3 bindings. | 98 | COMPLETE |

**Note**: Originally pinned to Enzyme toolchain (rustc d7daac06). Enzyme archived Feb 2026 — Wengert tape AD superseded it. No custom toolchain required.

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
- Wengert tape handles Rust code (operation recording + reverse replay)
- Hand-written CUDA backward kernels handle opaque GPU operations
- No whole-model graph tracing (torch.compile cannot trace NL's inner loop)
- Multi-architecture fat binary (sm_86/89/90 SASS + PTX fallback)

**What was delivered**: CUDA kernel pairs for all 3 matrix-based memory rules (DeltaRule, TitansLMM, HebbianRule). Each has forward + backward kernels with analytical gradients from the papers. All fp32. Single-block design (Grid=1, Block=min(d², 1024)) with shared memory for M recurrence. Projections and gates remain in Rust (tape-differentiable); only the sequential inner loop is CUDA. Multi-architecture fat binary via `-gencode` flags covers sm_86/89/90 with PTX fallback for future GPUs. Runtime `Backend` enum + `detect_gpu()` for diagnostics. `force_rust_reference()` override for testing.

| Phase | Kernels | Tests | Status |
|-------|---------|-------|--------|
| Phase 1: DeltaRule | `delta_forward.cu`, `delta_backward.cu` | 11 | COMPLETE |
| Phase 2: TitansLMM | `titans_forward.cu`, `titans_backward.cu` | 6 | COMPLETE |
| Phase 3: HebbianRule | `hebbian_forward.cu`, `hebbian_backward.cu` | 7 | COMPLETE |
| Phase 4: Integration | Composition-level CUDA parity tests | 5 | COMPLETE |
| Phase 5: Arch dispatch | `Backend` enum, `GpuInfo`, multi-arch build | ~13 | COMPLETE |
| **S2-M1a**: SWA head_dim fix | Replace warp shuffle with shared-mem tree reduction in `swa_forward.cu`/`swa_backward.cu`. Power-of-two guard in dispatch. | +4 (17 total SWA) | COMPLETE |
| **S2-M1b**: GPU-resident model | `GpuBuf<T>` RAII primitive, `GpuMAGParams`/`GpuContextState` on device, 3 new CUDA kernels (embedding, elementwise, cross_entropy), `_dd` dispatch variants, `gpu_cms_forward`/`gpu_cms_backward`/`gpu_weight_update`. PCIe: 8KB in + 4B out per step. | 6 | COMPLETE |

**Deliverables**:
- [x] CUDA kernel pairs for memory rule hot paths (Delta Rule, Titans LMM, Hebbian)
- [x] Architecture dispatch (sm_86, sm_89, sm_90 + PTX fallback)
- [x] `.cubin`/`.ptx` packaging (fat binary via multi-gencode)
- [x] Compilation documentation (`docs/build_matrix.md`)
- [x] SWA kernels fixed for head_dim > 32 (S2-M1a: shared-mem tree reduction, PR #40)
- [x] GPU-resident model: zero PCIe forward/backward/update (S2-M1b, PR #41)

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
- [x] Profile 1: Inner-loop only (no AD on target, pre-computed outer-loop weights)
- [x] Profile 2: Full NL (forward + backward + gradient apply on x86_64)
- [x] Profile 3: WASM (wasm32-unknown-unknown cross-compilation validated)
- [x] Target matrix validation: x86_64 native + wasm32 cross-compile
- [x] Benchmark: ~34k tok/s on x86_64 for d=64 (exceeds 18k target)
- [x] Edge/wasm builds require no special toolchain features

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

## Stage 3b: Primitive Completeness (NOT STARTED)

The NL paper suite defines ~50 primitives. Stages 1-3 implemented 26. This stage completes the primitive inventory — **specs first, then code**. Each primitive gets a spec sheet built from HADES graph traceability (paper equations, source fields) before any Rust touches a keyboard.

The Hope architecture (self-modifying Titans + CMS) is the castle. These are the lego pieces.

### Spec Work: Phase 1 — Inner-Loop Algorithm Knobs

The MIRAS Algorithm knob currently has {GD, GD+Momentum}. The papers define at least 5 more.

| Spec Task | Primitive | Paper Source | Spec Path (planned) | Status |
|-----------|-----------|-------------|---------------------|--------|
| **S3b-S1** | DGD (Delta Gradient Descent) | HOPE (2512.24695) §4.5 | `specs/algorithms/optimization_machinery/03_dgd.md` | NOT STARTED |
| **S3b-S2** | DMGD (Deep Momentum Gradient Descent) | HOPE (2512.24695) §4.5 Eq 33+ | `specs/algorithms/optimization_machinery/04_dmgd.md` | NOT STARTED |
| **S3b-S3** | FTRL (Follow the Regularized Leader) | MIRAS (2504.13173) §3.1 | `specs/algorithms/optimization_machinery/05_ftrl.md` | NOT STARTED |
| **S3b-S4** | Implicit GD (Longhorn-style) | Atlas (2505.23735) Table 1 | `specs/algorithms/optimization_machinery/06_implicit_gd.md` | NOT STARTED |
| **S3b-S5** | Newton-Schulz (inner-loop) | Atlas/HOPE (momentum nonlinearity) | `specs/algorithms/optimization_machinery/07_newton_schulz_inner.md` | NOT STARTED |

**DGD is the highest priority** — it's the core inner-loop optimizer for Hope. The update depends on both the current input AND the current memory state M, making it fundamentally more expressive than plain GD. The HOPE ablation shows ~1.2 ppl cost for removing it.

### Spec Work: Phase 2 — Retention & Bias Gaps

The pluggable retention system (S3-M1) provides the dispatch infrastructure. These specs fill the missing variants.

| Spec Task | Primitive | Paper Source | Spec Path (planned) | Status |
|-----------|-----------|-------------|---------------------|--------|
| **S3b-S6** | Bregman divergence (general) | MIRAS (2504.13173) §5.2 Variant 5 | `specs/algorithms/retention_mechanisms/06_bregman.md` | NOT STARTED |
| **S3b-S7** | L_q norm retention (general q) | MIRAS (2504.13173) §5.2 Variant 4 | `specs/algorithms/retention_mechanisms/07_lq_norm.md` | NOT STARTED |
| **S3b-S8** | Sigmoid-bounded retention | MIRAS (2504.13173) §5.2 (Bregman+logit) | `specs/algorithms/retention_mechanisms/08_sigmoid_bounded.md` | NOT STARTED |
| **S3b-S9** | l_1 attentional bias | MIRAS (2504.13173) §5.1 (p=1 sign-based) | `specs/algorithms/attentional_biases/01_l1_sign.md` | NOT STARTED |
| **S3b-S10** | KL attentional bias | MIRAS (2504.13173) §5.1 (theoretical) | `specs/algorithms/attentional_biases/02_kl_objective.md` | NOT STARTED |
| **S3b-S11** | Generic l_p bias dispatch | MIRAS (2504.13173) §5.1 Eq 11 | `specs/algorithms/attentional_biases/00_interface.md` | NOT STARTED |

### Spec Work: Phase 3 — Self-Referential Architecture

The central HOPE contribution. Requires Phase 1 (DGD) as a prerequisite — self-modifying without DGD is a roof without walls.

| Spec Task | Primitive | Paper Source | Spec Path (planned) | Status |
|-----------|-----------|-------------|---------------------|--------|
| **S3b-S12** | Self-referential projections (M_k, M_v, M_q) | HOPE (2512.24695) §8.1 Eq 79-85 | `specs/algorithms/self_referential/00_interface.md` | NOT STARTED |
| **S3b-S13** | Self-generated values (v̂ = M_□(v)) | HOPE (2512.24695) §8.1 Eq 84 | `specs/algorithms/self_referential/01_self_generated_values.md` | NOT STARTED |
| **S3b-S14** | Higher-order feature maps φ(k) | HOPE (2512.24695) §4.5 extension | `specs/algorithms/self_referential/02_feature_maps.md` | NOT STARTED |
| **S3b-S15** | Chunkwise training for self-ref Titans | HOPE (2512.24695) §8.2 | `specs/algorithms/self_referential/03_chunkwise_self_ref.md` | NOT STARTED |

### Spec Work: Phase 4 — Outer-Loop & Architectural Gaps

| Spec Task | Primitive | Paper Source | Spec Path (planned) | Status |
|-----------|-----------|-------------|---------------------|--------|
| **S3b-S16** | AdamW (outer-loop) | Standard / TNT experiments | `specs/algorithms/optimization_machinery/08_adamw_outer.md` | NOT STARTED |
| **S3b-S17** | AdaMuon | Atlas (2505.23735) | `specs/algorithms/optimization_machinery/09_adamuon.md` | NOT STARTED |
| **S3b-S18** | Atlas Omega rule spec | Atlas (2505.23735) | `specs/algorithms/memory_update_rules/titans_family/04_atlas_omega.md` | NOT STARTED |
| **S3b-S19** | Short Conv1D on keys/queries | Atlas (2505.23735), modern convention | `specs/infrastructure/attention/02_short_conv.md` | NOT STARTED |
| **S3b-S20** | Hope composition (self-mod Titans + CMS) | HOPE (2512.24695) §8.3 | `specs/algorithms/composition_patterns/04_hope.md` | NOT STARTED |

### Implementation Milestones (blocked by specs)

Implementation begins only after the corresponding spec is written and approved.

| Milestone | Implements Specs | Description | Dependencies | Status |
|-----------|-----------------|-------------|-------------|--------|
| **S3b-M1: Algorithm Knob Expansion** | S3b-S1 through S3b-S5 | DGD, DMGD, FTRL, implicit GD, NS inner. New `AlgorithmKind` enum or extend existing rule dispatch. | Specs S3b-S1–S5 | NOT STARTED |
| **S3b-M2: Retention & Bias Completion** | S3b-S6 through S3b-S11 | Bregman, L_q, sigmoid-bounded retention. l_1, KL bias. Generic l_p dispatch. | Specs S3b-S6–S11 | NOT STARTED |
| **S3b-M3: Self-Referential Primitives** | S3b-S12 through S3b-S15 | M_k/M_v/M_q memory modules replacing W projections. Feature maps. Chunkwise parallel. | S3b-M1 (DGD required), Specs S3b-S12–S15 | NOT STARTED |
| **S3b-M4: Outer-Loop & Architectural** | S3b-S16 through S3b-S20 | AdamW, AdaMuon, Atlas spec, short conv, Hope composition. | Specs S3b-S16–S20 | NOT STARTED |

### Priority Order

```
Phase 1 Specs (Algorithm Knobs)     ─── highest priority, foundation for everything
    │
    ├─► Phase 2 Specs (Retention/Bias)  ─── independent of Phase 1, can parallel
    │
    ▼
Phase 1 Implementation (S3b-M1)     ─── DGD needed before self-ref
    │
    ├─► Phase 2 Implementation (S3b-M2) ─── independent, can parallel
    │
    ▼
Phase 3 Specs (Self-Referential)    ─── needs DGD spec as input
    │
    ▼
Phase 3 Implementation (S3b-M3)     ─── the castle walls
    │
    ├─► Phase 4 Specs + Impl (S3b-M4)  ─── outer-loop + architectural gaps
    │
    ▼
Hope Architecture                   ─── the castle (S3b-S20 → future milestone)
```

### Spec Writing Process

Each spec follows the existing CONTRACT format and MUST include:
1. **CONTRACT header** — Purpose, Expects, Guarantees, Cost, Trade-off, Position, Source
2. **HADES traceability** — Specific equation numbers, paper IDs, collection references
3. **MIRAS knob mapping** — Which knobs this primitive fills
4. **Pseudocode** — Rust-like with trait bounds
5. **Gradient derivation** — Analytical backward for tape integration
6. **Interaction matrix** — Which existing primitives this composes with

---

## Stage 4: Build & Serve

Stage 4 has two phases with different scopes:

**Phase 1 (M1–M8)**: Pipeline infrastructure built alongside S1–S3 using whatever primitives existed at the time. Goal was to prove the build→serve loop works end-to-end. `toy_60m.json` is a pre-HOPE model: Titans LMM + MAG + k=2 CMS. It is a regression checkpoint, not the destination architecture.

**Phase 2 (M9–M14)**: Expanded after S3b delivers DGD, self-referential projections, and the HOPE composition. This is the real build — the HOPE architecture running end-to-end. The full scope of Phase 2 cannot be locked down until S3b specs are written, but the milestones below represent the known requirements.

**S4 Phase 1 and S3b run in parallel.** S4-M7 (primitive validation) can proceed now. S4 Phase 2 milestones are blocked on their corresponding S3b deliverables.

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

### S4-M5: Declared Checkpoint Format ✅

Versioned checkpoint format with schema migration, optional ContextState/Conductor state for build resume, and binary format option.

**PR**: #39

---

### S4-M6: Model Design — First Real Build ✅

Concrete model configuration proving the build→serve pipeline works end-to-end on real text data.

**What was delivered**: `configs/toy_60m.json` config (d=2048, heads=16, seq_len=512, k=2 CMS [1,8], Titans LMM + MAG). `data/download_fineweb.py` script for FineWeb byte-level data acquisition. `build.py` enhanced with `--config` JSON loading, binary `.bin` data support, and `--chunk_sizes` argument. Config file provides defaults, CLI args override.

**Architecture**: ~59.8M params, byte-level tokenizer (vocab=256), Titans LMM (matrix M + momentum S), MAG composition, k=2 CMS (Level 0 every step, Level 1 every 8th).

**Deliverables**:
- [x] `python/configs/toy_60m.json` — model config
- [x] `python/data/download_fineweb.py` — FineWeb data acquisition
- [x] `python/build.py` — `--config` flag, binary data, `--chunk_sizes`
- [x] Success criteria defined: loss convergence, no NaN, k=2 >= k=1, checkpoint roundtrip

**Dependencies**: S4-M5 (declared checkpoint format)

---

### S4-M7: Primitive Validation (PLANNED)

Run the toy_60m config end-to-end as a regression checkpoint for the pre-S3b primitive set. This validates that the build→serve pipeline works with existing primitives (Titans LMM + MAG + k=2 CMS + plain SGD outer-loop + byte tokenizer). It does NOT validate the HOPE architecture — that is S4 Phase 2.

Success criteria:
1. Loss converges (byte-level CE from ~5.5 to <4.0)
2. No NaN/Inf through 5000 steps
3. k=2 matches or beats k=1 baseline
4. Checkpoint save → load → continue with no loss spike
5. Serve generates coherent byte sequences

**Dependencies**: S4-M6 (model design + config)
**Note**: Can proceed in parallel with S3b spec work. Low-priority relative to S3b.
**Status**: NOT STARTED

---

## Stage 4 Phase 2: HOPE Build & Serve (BLOCKED ON S3b)

These milestones become active as S3b delivers its primitives. They cannot be fully specced until S3b specs are written — the scope below is the known shape.

### S4-M9: Multi-Block CMS Execution Engine (PLANNED)

S3-M3 (CMS Variants) built the configuration schema and validation for 5 deployment patterns (Basic, Nested, Sequential, Independent, Hybrid). What is missing is the actual execution engine that runs multiple blocks with independent or hierarchical CMS schedules.

**What this delivers**:
- Runtime that instantiates and steps multiple `BlockConfig` entries
- Block-level output aggregation (concat, sum, or attention-mix)
- Independent Conductor instances per block (for Independent/Hybrid variants)
- Integration with distributed_step for multi-block multi-GPU

**Dependencies**: S3-M3 (CMS Variant schemas, COMPLETE), S3b-M4 (HOPE composition spec for block wiring)
**Status**: NOT STARTED

---

### S4-M10: DGD Inner-Loop Build Path (PLANNED)

Replace plain gradient descent in the inner loop with DGD (Delta Gradient Descent). DGD's update depends on both the current input AND the current memory state M — making it fundamentally more expressive than plain GD. The HOPE ablation shows ~1.2 ppl cost for removing it.

**What this delivers**:
- `build.py` updated to configure `AlgorithmKind::DGD` (from S3b-M1)
- New `configs/hope_Nm_dgd.json` config with DGD enabled
- Validation: DGD build loop converges, loss is better than plain GD baseline

**Dependencies**: S3b-M1 (DGD implementation)
**Status**: NOT STARTED

---

### S4-M11: Self-Referential Build (PLANNED)

Replace fixed W_Q/W_K/W_V projection matrices with memory-derived projections (M_k, M_v, M_q from HOPE §8.1 Eq 79-85). The memory matrices project their own keys, values, and queries — the model's attention is computed from what the memory has learned, not from fixed outer-loop weights.

**What this delivers**:
- `build.py` updated to use self-referential projection path (from S3b-M3)
- Config: `configs/hope_Nm_selfref.json`
- Validation: gradient flows through M_k/M_v/M_q correctly, no gradient blocking
- Chunkwise parallel path for self-referential Titans (from S3b-S15)

**Dependencies**: S3b-M3 (self-referential primitives), S4-M10 (DGD required before self-ref per S3b ordering)
**Status**: NOT STARTED

---

### S4-M12: AdamW / AdaMuon Outer-Loop (PLANNED)

Replace plain SGD in `build.py` with AdamW or AdaMuon for the outer-loop weight updates. Plain SGD does not scale to real builds. The outer-loop optimizer choice does not violate IS/IS NOT — AdamW applies to the outer (slow) loop parameters, not the inner (fast) memory updates.

**What this delivers**:
- `apply_weight_gradients_adamw()` / `apply_weight_gradients_adamuon()` on MAGParams (from S3b-M4)
- `build.py` `--outer_optimizer` flag: `"sgd"` (default, backward compat) / `"adamw"` / `"adamuon"`
- Validation: AdamW outer-loop converges faster than SGD baseline at matched step count

**Dependencies**: S3b-M4 (AdamW + AdaMuon specs and implementation)
**Status**: NOT STARTED

---

### S4-M13: HOPE Model Config (PLANNED)

The full HOPE architecture config: self-referential Titans + DGD inner-loop + k=4 CMS + AdamW outer-loop. This replaces `toy_60m.json` as the primary build target.

**What this delivers**:
- `configs/hope_Nm.json` (size TBD once S3b primitives are costed)
- Documents the composition: which memory rules, which parallelization, which CMS variant
- Traces every config field back to a HOPE paper equation or S3b spec
- Defines the validation suite for the HOPE architecture

**Dependencies**: S4-M10 (DGD), S4-M11 (self-ref), S4-M12 (AdamW), S3b-S20 (HOPE composition spec)
**Status**: NOT STARTED — spec cannot be written until S3b-S20 exists

---

### S4-M14: HOPE End-to-End Validation (PLANNED)

Run the `hope_Nm.json` config end-to-end. This is the real build hardening milestone — the HOPE architecture working, not the toy_60m scaffolding.

Success criteria:
1. Loss converges on real text data (FineWeb or equivalent)
2. DGD inner-loop shows measurable ppl improvement over plain GD (ablation)
3. Self-referential projections: M_k/M_v/M_q gradients non-zero throughout build
4. k=4 CMS: all 4 levels contribute (gate diagnostics)
5. AdamW outer-loop: no divergence through full build
6. Checkpoint → restore → continue: identical loss trajectory
7. Serve: model generates coherent text, memory self-modifies during inference

**Dependencies**: S4-M13 (HOPE config), all S4 Phase 2 milestones
**Status**: NOT STARTED

---

### S4-M8: Wengert Tape Integration ✅

Replace the hand-written `cms_backward()` gradient path with the Wengert tape-based `tape_compute_gradients()`. This eliminates the maintenance burden of hand-written backward functions while preserving identical numerical output.

**What was delivered**: Five phases (P1–P5) over PRs #58–65:

| Phase | Description | PR | Tests | Status |
|-------|-------------|-----|-------|--------|
| P1–P1.9 | Wengert tape core (57 ops, arena allocator, opaque VJP blocks, backward replay) | #58 | 57 | COMPLETE |
| P2 | Traced forward wrappers (`traced_cms_forward()`, bitwise-equivalent to `cms_forward()`) | #55 | 23 | COMPLETE |
| P3.1 | `tape_compute_gradients()` — runs tape forward + backward, returns parameter gradients | #59 | — | COMPLETE |
| P3.2 | Error buffer routing for frozen CMS levels through tape path | #60 | — | COMPLETE |
| P3.3 | Dynamic frequency gate (`FrequencySchedule::Learned`) traced on tape | #61 | — | COMPLETE |
| P3.4 | Class 3 tests: tape vs hand-written backward for all 9 rules × k=1,2,4 | #62 | 39 | COMPLETE |
| P3.5 | Finite-difference gradient checks on tape path | #62 | — | COMPLETE |
| P4.1 | **Switchover**: `cms_compute_gradients()` now delegates to tape path. Hand-written path preserved as `_handwritten()` test oracle. | #63 | — | COMPLETE |
| P4.2 | 10-step build loop regression test (tape vs hand-written trajectory) | #64 | 1 | COMPLETE |
| P4.3 | PyO3 binding updated: Python `cms_compute_gradients()` routes through tape | #65 | — | COMPLETE |
| P5.1 | Enzyme feature flag removal (already done in PR #57) | — | — | COMPLETE |
| P5.3 | Documentation update (contract.md, ROADMAP.md) | — | — | COMPLETE |
| P5.4 | Final validation (full test suite) | — | — | COMPLETE |

**Key design decisions**:
- Tape records during `traced_cms_forward()`, replays in reverse via `tape.backward(loss_id)`
- `TracedParamIds` struct maps each parameter to its `BufId` for gradient extraction after backward
- Memory rules + CUDA kernels register as opaque VJP blocks via `OpaqueVjp` trait
- `cms_compute_gradients_handwritten()` preserved as test oracle (Class 3 tests verify tape ≡ hand-written)
- Build loop regression test catches accumulation errors that single-step tests miss

**Dependencies**: S0-M1 (tape core), S3-M5 (dynamic frequency for P3.3)

---

## Dependency Graph

```
Stage 0: Foundation ─────────────────────────── COMPLETE
    │
    ▼
Stage 1: Algorithm Core ─────────────────────── COMPLETE (805 tests)
    │
    ├─► S4-M8: Wengert Tape Integration ──────── COMPLETE (PRs #55–65)
    │
    ├─► S2-M1: Compilation Strategy ─────────── COMPLETE
    │       │
    │       ├─► S2-M1a: SWA head_dim fix ──────── COMPLETE (PR #40)
    │       ├─► S2-M1b: GPU-resident model ────── COMPLETE (PR #41)
    │       ├─► S2-M2: Multi-GPU Distribution ── COMPLETE
    │       └─► S2-M3: Serving ──────────────── COMPLETE
    │               └─► S2-M4: Edge Deployment ─ COMPLETE
    │
    ├─► S3-M1: Pluggable Retention ─────────── COMPLETE (PR #31)
    ├─► S3-M2: M3 Optimizer ────────────────── COMPLETE (PR #32)
    │       └─► S3-M3: CMS Variants (schemas) ── COMPLETE (PR #32)
    │               └─► S4-M9: Multi-Block Execution Engine ── PLANNED (needs S3b-M4)
    ├─► S3-M4: Atlas Omega Rule ────────────── COMPLETE (PR #35)
    ├─► S3-M5: Dynamic Frequency ───────────── COMPLETE (PR #36)
    │
    └─► S4-M1: Weight Serialization ────────── COMPLETE (PR #37)
            └─► S4-M2: Stateful PyO3 ────────── COMPLETE (PR #38)
                    ├─► S4-M3: Build Script ──── COMPLETE (PR #38)
                    ├─► S4-M4: Serve Script ──── COMPLETE (PR #38)
                    └─► S4-M5: Declared Ckpt ─── COMPLETE (PR #39)
                            └─► S4-M6: Model Design (toy_60m) ── COMPLETE
                                    └─► S4-M7: Primitive Validation ── PLANNED (parallel with S3b)

── S3b ─────────────────────────────────────────── NOT STARTED
    │
    ├─► S3b-M1: Algorithm Knobs (DGD, DMGD, FTRL…)
    │       └─► S4-M10: DGD Build Path ── PLANNED
    │               └─► S4-M11: Self-Referential Build ── PLANNED (needs S3b-M3)
    │
    ├─► S3b-M2: Retention & Bias Completion
    │
    ├─► S3b-M3: Self-Referential Primitives ── (needs S3b-M1)
    │
    └─► S3b-M4: Outer-Loop & Architectural
            └─► S4-M12: AdamW/AdaMuon Outer-Loop ── PLANNED
                    │
                    (S4-M10 + S4-M11 + S4-M12 + S3b-S20)
                    │
                    └─► S4-M13: HOPE Model Config ── PLANNED
                            └─► S4-M14: HOPE End-to-End Validation ── PLANNED
```

**Stage 2** milestones are sequential. **Stage 3** milestones are independent. **Stage 3b** is spec-first: primitives get spec sheets before implementation; S3b Phase 3 (self-referential) depends on S3b Phase 1 (DGD). **Stage 4 Phase 1** (M1–M8) is complete/planned with existing primitives and runs in parallel with S3b. **Stage 4 Phase 2** (M9–M14) is the real HOPE build — blocked on S3b deliverables; scope cannot be fully locked until S3b specs exist.

---

## Summary

| Stage | Milestones | Tests | Status |
|-------|-----------|-------|--------|
| Stage 0: Foundation | 3 | 57 + 47 + 98 | COMPLETE |
| Stage 1: Algorithm Core | 19 | 778 Rust + 27 Python | COMPLETE |
| Stage 2: Production Infra | 4 (+M1a, +M1b) | 33 CUDA + 13 dispatch + 6 GPU-resident + 20 edge + 18 serving + 18 distributed | COMPLETE |
| Stage 3: Extensions | 5 | 22 retention + 35 M3/variants + 26 Atlas + 22 dynamic freq = 105 | COMPLETE |
| Stage 3b: Primitive Completeness | 20 specs + 4 impl milestones | — | NOT STARTED |
| Stage 4 Phase 1: Pipeline Scaffolding | 8 (7 done, 1 planned) | 27 Python + 120 tape/traced/class3 Rust | IN PROGRESS |
| Stage 4 Phase 2: HOPE Build & Serve | 6 milestones (M9–M14) | — | BLOCKED ON S3b |

**Current position**: S0–S3 complete. S4 Phase 1 pipeline delivered (M1–M6, M8): can build a model on real text data and serve it locally using pre-S3b primitives. Wengert tape is the production gradient path (M8, PRs #55–65).

**Active fronts**:
1. **S3b spec work** (highest priority): 20 spec sheets for DGD, self-referential projections, retention gaps, and the HOPE composition. These are the lego pieces the castle is built from.
2. **S4-M7** (parallel, low priority): validate the toy_60m pipeline as a regression checkpoint on existing primitives.
3. **S4 Phase 2** (blocked): expands as S3b delivers. The real build — HOPE architecture end-to-end — cannot begin until DGD and self-referential projections are implemented.

Total test count: 834 Rust lib + 27 Python = **861 tests** (lib only; full `cargo test` is higher).
