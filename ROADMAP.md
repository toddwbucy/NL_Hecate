# NL_Hecate Roadmap

This document provides the single authoritative view of project progress and upcoming work. It replaces the scattered "Phase N" naming that was reused across different workstreams.

## Naming Convention

The project is organized into **Stages**, each containing **Milestones**. No two milestones share a name.

```
Stage 0:  Foundation              — Toolchain, spike, pipeline validation
Stage 1:  Algorithm Core          — Memory rules, compositions, scheduling, parallelization
Stage 2:  Production Infra        — Multi-GPU, serving, compilation, deployment
Stage 3:  Extensions              — Pluggable retention, M3 optimizer, CMS variants
Stage 3b: Primitive Completeness  — Split: Critical (HOPE path) + Deferred (Stage 5)
Stage 4:  MVP                     — Build a model, serve it locally
Stage 5:  MIRAS Completeness      — Deferred S3b primitives (post-HOPE)
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

## Stage 3b: Primitive Completeness (SPECS COMPLETE — IMPLEMENTATION NOT STARTED)

The NL paper suite defines ~50 primitives. Stages 1–3 implemented 26. This stage completes the primitive inventory. **All 20 spec sheets are written and stored in HADES (status: `v0.4.0`).** The spec-to-code pipeline is unblocked — implementation is the remaining work. No spec writing is needed — go straight to implementation.

The Hope architecture (self-modifying Titans + CMS) is the castle. These are the lego pieces. All specs confirmed present on disk in `specs/` and recorded in HADES `hecate_specs` collection.

**Important**: Following the 2026-02-22 committee review, S3b is split into two tracks. **S3b-Critical** contains only the primitives required for the HOPE architecture. **S3b-Deferred** contains MIRAS design-space completeness work that is valuable but does not block the primary path. See `docs/committee_response.md` for the full rationale.

### Spec Catalog (all 20 specs, unchanged)

<details>
<summary>Phase 1 — Inner-Loop Algorithm Knobs ✍️ SPECS COMPLETE</summary>

The MIRAS Algorithm knob currently has {GD, GD+Momentum}. The papers define at least 5 more.

| Spec | Primitive | Paper Source | Spec Path | Status |
|------|-----------|-------------|-----------|--------|
| **S3b-S1** | DGD (Delta Gradient Descent) | HOPE (2512.24695) §4.5 | `specs/algorithms/optimization_machinery/03_dgd.md` | SPEC COMPLETE |
| **S3b-S2** | DMGD (Deep Momentum Gradient Descent) | HOPE (2512.24695) §4.5 Eq 33+ | `specs/algorithms/optimization_machinery/04_dmgd.md` | SPEC COMPLETE |
| **S3b-S3** | FTRL (Follow the Regularized Leader) | MIRAS (2504.13173) §3.1 | `specs/algorithms/optimization_machinery/05_ftrl.md` | SPEC COMPLETE |
| **S3b-S4** | Implicit GD (Longhorn-style) | Atlas (2505.23735) Table 1 | `specs/algorithms/optimization_machinery/06_implicit_gd.md` | SPEC COMPLETE |
| **S3b-S5** | Newton-Schulz (inner-loop) | Atlas/HOPE (momentum nonlinearity) | `specs/algorithms/optimization_machinery/07_newton_schulz_inner.md` | SPEC COMPLETE |

**DGD is the highest priority** — it's the core inner-loop optimizer for Hope. The update depends on both the current input AND the current memory state M, making it fundamentally more expressive than plain GD. The HOPE ablation shows ~1.2 ppl cost for removing it.

</details>

<details>
<summary>Phase 2 — Retention & Bias Gaps ✍️ SPECS COMPLETE</summary>

The pluggable retention system (S3-M1) provides the dispatch infrastructure. These specs fill the missing variants.

| Spec | Primitive | Paper Source | Spec Path | Status |
|------|-----------|-------------|-----------|--------|
| **S3b-S6** | Bregman divergence (general) | MIRAS (2504.13173) §5.2 Variant 5 | `specs/algorithms/retention_mechanisms/06_bregman.md` | SPEC COMPLETE |
| **S3b-S7** | L_q norm retention (general q) | MIRAS (2504.13173) §5.2 Variant 4 | `specs/algorithms/retention_mechanisms/07_lq_norm.md` | SPEC COMPLETE |
| **S3b-S8** | Sigmoid-bounded retention | MIRAS (2504.13173) §5.2 (Bregman+logit) | `specs/algorithms/retention_mechanisms/08_sigmoid_bounded.md` | SPEC COMPLETE |
| **S3b-S9** | l_1 attentional bias | MIRAS (2504.13173) §5.1 (p=1 sign-based) | `specs/algorithms/attentional_biases/01_l1_sign.md` | SPEC COMPLETE |
| **S3b-S10** | KL attentional bias | MIRAS (2504.13173) §5.1 (theoretical) | `specs/algorithms/attentional_biases/02_kl_objective.md` | SPEC COMPLETE |
| **S3b-S11** | Generic l_p bias dispatch | MIRAS (2504.13173) §5.1 Eq 11 | `specs/algorithms/attentional_biases/03_lp_dispatch.md` | SPEC COMPLETE |

</details>

<details>
<summary>Phase 3 — Self-Referential Architecture ✍️ SPECS COMPLETE</summary>

The central HOPE contribution. Requires Phase 1 (DGD) **implementation** as a prerequisite — self-modifying without DGD is a roof without walls. Specs are ready; wait on S3b-M1 before starting S3b-M3.

| Spec | Primitive | Paper Source | Spec Path | Status |
|------|-----------|-------------|-----------|--------|
| **S3b-S12** | Self-referential projections (M_k, M_v, M_q) | HOPE (2512.24695) §8.1 Eq 79-85 | `specs/algorithms/self_referential/00_interface.md` | SPEC COMPLETE |
| **S3b-S13** | Self-generated values (v̂ = M_□(v)) | HOPE (2512.24695) §8.1 Eq 84 | `specs/algorithms/self_referential/01_self_generated_values.md` | SPEC COMPLETE |
| **S3b-S14** | Higher-order feature maps φ(k) | HOPE (2512.24695) §4.5 extension | `specs/algorithms/self_referential/02_feature_maps.md` | SPEC COMPLETE |
| **S3b-S15** | Chunkwise training for self-ref Titans | HOPE (2512.24695) §8.2 | `specs/algorithms/self_referential/03_chunkwise_self_ref.md` | SPEC COMPLETE |

</details>

<details>
<summary>Phase 4 — Outer-Loop & Architectural Gaps ✍️ SPECS COMPLETE</summary>

| Spec | Primitive | Paper Source | Spec Path | Status |
|------|-----------|-------------|-----------|--------|
| **S3b-S16** | AdamW (outer-loop) | Standard / TNT experiments | `specs/algorithms/optimization_machinery/08_adamw_outer.md` | SPEC COMPLETE |
| **S3b-S17** | AdaMuon | Atlas (2505.23735) | `specs/algorithms/optimization_machinery/09_adamuon.md` | SPEC COMPLETE |
| **S3b-S18** | Atlas Omega rule spec | Atlas (2505.23735) | `specs/algorithms/memory_update_rules/titans_family/04_atlas_omega.md` | SPEC COMPLETE + IMPL COMPLETE (S3-M4) |
| **S3b-S19** | Short Conv1D on keys/queries | Atlas (2505.23735), modern convention | `specs/infrastructure/attention/02_short_conv.md` | SPEC COMPLETE |
| **S3b-S20** | Hope composition (self-mod Titans + CMS) | HOPE (2512.24695) §8.3 | `specs/algorithms/composition_patterns/04_hope.md` | SPEC COMPLETE |

</details>

### S3b-Critical: HOPE Path (implementation milestones)

These are the primitives required to build the HOPE architecture. They run after S4-M7 validates the existing pipeline.

| Milestone | Implements Specs | HADES Task(s) | Dependencies | Status |
|-----------|-----------------|---------------|-------------|--------|
| **S3b-M1: DGD** | S3b-S1 only | `task_38485e` (GAP-I) | S4-M7 (validate pipeline first) | NOT STARTED |
| **S3b-M3: Self-Referential Primitives** | S3b-S12 through S3b-S15 | `task_929845` (GAP-L: projections), `task_eb31fd` (GAP-M: self-gen values), `task_79f2c5` (GAP-E: feature maps), `task_4a9b1d` (GAP-N: chunkwise) | S3b-M1 (DGD impl required first) | NOT STARTED |
| **S3b-M4-critical: Outer-Loop & HOPE** | S3b-S16, S3b-S19, S3b-S20 | **AdamW impl: MISSING**, **Conv1D impl: MISSING**, **HOPE composition impl: MISSING** (specs written, closed: `task_013b41`, `task_f903ab`) | S3b-S16/S19 parallel with M1; S3b-S20 blocked on M3 | NOT STARTED |

### S3b-Deferred: MIRAS Completeness (Stage 5 — post-HOPE)

These primitives complete the MIRAS design space but are not required for the HOPE architecture. They will be implemented after the HOPE model is building and serving successfully.

| Milestone | Implements Specs | HADES Task(s) | Status |
|-----------|-----------------|---------------|--------|
| **S3b-M1-deferred: Algorithm Knobs** | S3b-S2, S3b-S3, S3b-S4, S3b-S5 | `task_544d8d` (GAP-J: Implicit GD), `task_c48b71` (GAP-K: FTRL) | DEFERRED (Stage 5) |
| **S3b-M2: Retention & Bias** | S3b-S6 through S3b-S11 | `task_60e757` (GAP-F: Bregman), `task_eb6a4f` (GAP-G: L_q), `task_f18e65` (GAP-H: sigmoid), `task_667d92` (GAP-D: KL bias) | DEFERRED (Stage 5) |
| **S3b-M4-deferred: AdaMuon** | S3b-S17 | **MISSING** | DEFERRED (Stage 5) |

### Priority Order

```text
S4-M7: Primitive Validation (IMMEDIATE — run first)
    │
    ▼
S3b-M1: DGD (Critical Path)
    │
    ├─► S3b-S16: AdamW outer-loop (parallel)
    ├─► S3b-S19: Short Conv1D (parallel)
    │
    ▼
S3b-M3: Self-Referential Primitives (needs DGD)
    │
    ▼
S3b-S20: HOPE Composition (the castle, unlocks S4 Phase 2)

                    ┌──────────────────────────────┐
                    │ S3b-Deferred (Stage 5)       │
                    │ Post-HOPE: DMGD, FTRL,       │
                    │ Implicit GD, NS inner,       │
                    │ Bregman, L_q, sigmoid ret.,   │
                    │ KL bias, l_p dispatch,        │
                    │ AdaMuon                       │
                    └──────────────────────────────┘
```

---

## Resolved & Deferred Blockers (HADES `hope_blockers`)

Four blocking gaps were extracted from the NL paper suite during initial graph construction. Two have been empirically resolved through implementation work. Two affect only the **brain transplant** path (converting a pre-trained Llama into HOPE) and are deferred — the from-scratch HOPE training path is the primary path and is unblocked.

**Brain transplant is community-scope, not project-scope.** The from-scratch HOPE training path (S4 Phase 2) does not require brain transplant. If community contributors want to explore Llama→HOPE conversion, the two deferred blockers document the open questions.

| Blocker | Paper Section | Status | Resolution |
|---------|--------------|--------|------------|
| **How many CMS frequency levels (k)?** | §7.3 | **RESOLVED** | k=4 is the empirical default since S1-M3. `[1,8,64,512]` geometric spacing. S3-M5 adds learned gates for dynamic selection. |
| **How to assign f_i to each CMS level?** | §7.3 | **RESOLVED** | Fixed `[1,8,64,512]` from geometric spacing (S1-M3). S3-M5 (`dynamic_freq.rs`) adds learned sigmoid gates as alternative. |
| **Which Llama layers to use for brain transplant?** | §7.3 | **DEFERRED** | Brain transplant path only. Paper says "Given k pre-trained MLPs..." without specifying layer indices. Out of scope for primary training path. |
| **What happens to Llama attention layers in brain transplant?** | §7.3 | **DEFERRED** | Brain transplant path only. Paper discusses MLPs only. Out of scope for primary training path. |

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

### S4-M7: Primitive Validation (IMMEDIATE — run before S3b)

**HADES task**: `task_e4aab3`

**Elevated to immediate priority per 2026-02-22 committee review.** The build pipeline exists and is idle. Three specific risks need validation before adding DGD/self-referential complexity: (1) Wengert tape memory at 60M scale, (2) CMS 1/sqrt(k) normalization in a real network, (3) MLP inner-loop dominance at scale.

Run `curriculum_100k.json` (d=1024, k=4, 100K steps) or equivalent end-to-end. This validates that the build→serve pipeline works with existing primitives (Titans LMM + MAG + k=4 CMS + plain SGD outer-loop + byte tokenizer). It does NOT validate the HOPE architecture — that is S4 Phase 2.

Success criteria (hard thresholds — pre-committed, do not adjust post-hoc):

| Criterion | Hard Threshold | Stop-the-Line |
|-----------|---------------|---------------|
| **Loss convergence** | Validation loss must decrease by ≥15% between step 50K and step 100K | If <10% decrease, pipeline is broken |
| **No NaN/Inf** | Zero NaN/Inf values in any tensor through full run | Any NaN = immediate stop |
| **CMS Level 3 gate activity** | Level 3 (slowest, every 512 steps) must fire ≥50 times per 1,000 steps | If <25 fires/1K steps, CMS is degenerate |
| **Wengert tape memory scaling** | Memory per token must not exceed 1.1× increase across seq_len 512→2048 | If ≥1.2× increase, tape has quadratic leak |
| **Checkpoint roundtrip** | Loss after restore must be within 1e-6 of loss before save | Any deviation = serialization bug |
| **Serve output** | Log generated text after build completes (informational only — coherent text is NOT expected from pre-HOPE primitives) | N/A — this run validates pipeline engineering, not model quality |

Additional deliverables (from committee behavioral probe recommendations):
- **Per-level diagnostics**: gate activity, memory norms, gradient norms per CMS level logged to JSONL
- **Wengert tape profiling**: memory consumption and backward-vs-forward time ratio
- **Forget Gate Probe**: add to `test_cms.rs` — feed contradictory sequence, assert fast levels update faster than slow levels
- **Curriculum Integration Probe**: per-phase loss tracking at distribution boundaries

**Dependencies**: S4-M6 (model design + config)
**Blocks**: S3b-M1 (validate pipeline before adding DGD complexity)
**Status**: NOT STARTED — IMMEDIATE PRIORITY

---

## Stage 4 Phase 2: HOPE Build & Serve (BLOCKED ON S3b)

These milestones become active as S3b delivers its primitives. They cannot be fully specced until S3b specs are written — the scope below is the known shape.

### S4-M9: Multi-Block CMS Execution Engine (PLANNED)

**HADES task**: MISSING — create when S3b-M4 (HOPE composition) is delivered

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

**HADES task**: MISSING — create after S3b-M1 is delivered

Replace plain gradient descent in the inner loop with DGD (Delta Gradient Descent). DGD's update depends on both the current input AND the current memory state M — making it fundamentally more expressive than plain GD. The HOPE ablation shows ~1.2 ppl cost for removing it.

**What this delivers**:
- `build.py` updated to configure `AlgorithmKind::DGD` (from S3b-M1)
- New `configs/hope_Nm_dgd.json` config with DGD enabled
- Validation: DGD build loop converges, loss is better than plain GD baseline

**Dependencies**: S3b-M1 (DGD implementation)
**Status**: NOT STARTED

---

### S4-M11: Self-Referential Build (PLANNED)

**HADES task**: MISSING — create after S3b-M3 is delivered

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

**HADES task**: MISSING — create after S3b-S16 impl is delivered

Replace plain SGD in `build.py` with AdamW or AdaMuon for the outer-loop weight updates. Plain SGD does not scale to real builds. The outer-loop optimizer choice does not violate IS/IS NOT — AdamW applies to the outer (slow) loop parameters, not the inner (fast) memory updates.

**What this delivers**:
- `apply_weight_gradients_adamw()` / `apply_weight_gradients_adamuon()` on MAGParams (from S3b-M4)
- `build.py` `--outer_optimizer` flag: `"sgd"` (default, backward compat) / `"adamw"` / `"adamuon"`
- Validation: AdamW outer-loop converges faster than SGD baseline at matched step count

**Dependencies**: S3b-M4 (AdamW + AdaMuon specs and implementation)
**Status**: NOT STARTED

---

### S4-M13: HOPE Model Config (PLANNED)

**HADES task**: MISSING — create after S4-M10/M11/M12 + S3b-S20 delivered

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

**HADES task**: MISSING — create after S4-M13 delivered

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

## Behavioral Validation

967 tests validate code correctness (matrix math, gradient parity, serialization roundtrips). Only 1 test (`test_k2_beats_k1_multiscale`) validates learning physics. The following probes address this gap. Each is tied to a specific milestone and validates a distinct learning property.

| Probe | What it validates | When implemented | Location |
|-------|-------------------|------------------|----------|
| **Forget Gate Probe** | CMS frequency separation — fast levels update faster than slow levels on contradictory input | S4-M7 | `test_cms.rs` |
| **Loss Monotonicity Probe** | DGD inner-loop loss decreases monotonically across optimization steps within a single forward pass | S3b-M1 (DGD) | `test_dgd.rs` (new) |
| **Curriculum Integration Probe** | Model does not catastrophically forget previous distribution at curriculum phase boundaries | S4-M7 | `build.py` logging |

**Forget Gate Probe**: Feed a contradictory sequence (A=1 for N steps, then A=2 for N steps). After the switch, Level 0 (fires every step) should converge to A=2 faster than Level 1 (fires every 8th step). This validates the core CMS claim: higher-frequency levels adapt faster to new information.

**Loss Monotonicity Probe**: When DGD is implemented (S3b-M1), add a runtime assertion that the inner-loop loss decreases monotonically across DGD's optimization steps. If loss oscillates within the inner loop, the test fails even if the matrix math is correct. This directly validates HOPE Eq 88.

**Curriculum Integration Probe**: During S4-M7 validation, log eval loss on BOTH the outgoing and incoming distribution at every curriculum phase boundary. Verify the model does not catastrophically forget — some regression is expected, catastrophic collapse is a bug.

**Policy**: Probes are added going forward for new primitives and validation runs. We are NOT retroactively adding behavioral probes to existing Stage 1-3 primitives as CI gates.

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
                                    │
                                    ▼
                    ┌─► S4-M7: Primitive Validation ── ★ IMMEDIATE ★
                    │       + Forget Gate Probe
                    │       + Curriculum Integration Probe
                    │       + Tape profiling
                    │
                    ▼
── S3b-Critical ────────────────────── HOPE PATH (blocked on S4-M7)
    │
    ├─► S3b-M1: DGD ──────────────────── START HERE (after S4-M7)
    │       │   + Loss Monotonicity Probe
    │       └─► S4-M10: DGD Build Path ── PLANNED
    │               └─► S4-M11: Self-Referential Build ── PLANNED (needs S3b-M3)
    │
    ├─► S3b-S16: AdamW outer-loop ─────── (parallel with S3b-M1)
    │       └─► S4-M12: AdamW Outer-Loop ── PLANNED
    │
    ├─► S3b-S19: Short Conv1D ─────────── (parallel with S3b-M1)
    │
    ├─► S3b-M3: Self-Referential ──────── (needs S3b-M1 impl)
    │
    └─► S3b-S20: HOPE Composition ─────── (needs S3b-M3)
            │
            (S4-M10 + S4-M11 + S4-M12 + S3b-S20)
            │
            └─► S4-M13: HOPE Model Config ── PLANNED
                    └─► S4-M14: HOPE End-to-End Validation ── PLANNED

── S3b-Deferred (Stage 5) ─────────── POST-HOPE
    ├─► DMGD, FTRL, Implicit GD, NS inner
    ├─► Bregman, L_q, sigmoid-bounded retention
    ├─► KL bias, l_1 bias, l_p dispatch
    └─► AdaMuon
```

**Stage 2** milestones are sequential. **Stage 3** milestones are independent. **S4-M7** (primitive validation) is immediate priority — it runs BEFORE S3b implementation begins. **S3b-Critical** contains only HOPE-path primitives: DGD → self-ref → HOPE composition + AdamW/Short Conv parallel. **S3b-Deferred** (Stage 5) contains MIRAS design-space completeness work — implemented after the HOPE model is running. **Stage 4 Phase 2** (M9–M14) is the real HOPE build — blocked on S3b-Critical deliverables.

---

## HADES Task Cross-Reference (updated 2026-02-22)

Complete mapping between ROADMAP milestones and HADES Persephone tasks. Use this to avoid rebuilding context across sessions.

### HOPE Critical Path — Task Status

| Step | ROADMAP Milestone | HADES Task Key | HADES Title | Status |
|------|-------------------|---------------|-------------|--------|
| 1 | **S4-M7**: Primitive Validation | `task_e4aab3` | S4-M7: Primitive Validation Run | **open** |
| 2a | **S3b-M1**: DGD | `task_38485e` | GAP-I: Extract DGD as named composable primitive | **open** |
| 2b | **S3b-S16**: AdamW outer-loop (impl) | — | *MISSING — spec task closed (`task_013b41`)* | needs creation |
| 2c | **S3b-S19**: Short Conv1D (impl) | — | *MISSING — spec task closed (`task_f903ab`)* | needs creation |
| 3a | **S3b-M3/S12**: Self-ref projections | `task_929845` | GAP-L: Self-referential Phase 2 — adaptive projection memories | **open** |
| 3b | **S3b-M3/S13**: Self-generated values | `task_eb31fd` | GAP-M: Self-generated values — v_hat = M_square(v_t) | **open** |
| 3c | **S3b-M3/S14**: Feature maps | `task_79f2c5` | GAP-E: Feature maps — phi() hook and FeatureMapKind enum | **open** |
| 3d | **S3b-M3/S15**: Chunkwise self-ref | `task_4a9b1d` | GAP-N: Chunkwise training for self-referential memories | **open** |
| 4 | **S3b-S20**: HOPE Composition (impl) | — | *MISSING* | needs creation |
| 5a | **S4-M10**: DGD Build Path | — | *MISSING* | needs creation |
| 5b | **S4-M11**: Self-Referential Build | — | *MISSING* | needs creation |
| 5c | **S4-M12**: AdamW Outer-Loop | — | *MISSING* | needs creation |
| 6 | **S4-M13**: HOPE Model Config | — | *MISSING* | needs creation |
| 7 | **S4-M14**: HOPE End-to-End Validation | — | *MISSING* | needs creation |

### S3b-Deferred (Stage 5) — Task Status

| ROADMAP Milestone | HADES Task Key | HADES Title | Status |
|-------------------|---------------|-------------|--------|
| **S3b-S2**: DMGD | — | *MISSING* | deferred |
| **S3b-S3**: FTRL | `task_c48b71` | GAP-K: FTRL accumulator integration | **open** |
| **S3b-S4**: Implicit GD | `task_544d8d` | GAP-J: Implicit GD Cases 4 and 5 | **open** |
| **S3b-S5**: Newton-Schulz inner | — | *MISSING* | deferred |
| **S3b-S6**: Bregman retention | `task_60e757` | GAP-F: Bregman retention framework | **open** |
| **S3b-S7**: L_q norm retention | `task_eb6a4f` | GAP-G: L_q as pluggable RetentionKind variant | **open** |
| **S3b-S8**: Sigmoid-bounded retention | `task_f18e65` | GAP-H: Sigmoid bounded retention | **open** |
| **S3b-S9**: l_1 attentional bias | — | *MISSING* | deferred |
| **S3b-S10**: KL attentional bias | `task_667d92` | GAP-D: Implement KL attentional bias | **open** |
| **S3b-S11**: Generic l_p dispatch | — | *MISSING* | deferred |
| **S3b-S17**: AdaMuon | — | *MISSING* | deferred |

### Other Open Tasks (not on HOPE critical path)

| HADES Task Key | Title | Notes |
|---------------|-------|-------|
| `task_unimpl_specs` | Spec Audit: Unimplemented Specifications | Tracking task |
| `task_partial_specs` | Spec Audit: Partially Implemented Specifications | Tracking task |
| `task_d06657` | GAP-Q: contract.md final reconciliation | Housekeeping |
| `task_0ecca3` | GAP-C: Unblock TNT Q-K projection integration | Stage 5 |
| `task_08ca2e` | ShareGPT 90M build — 100K steps | Experimental run |
| `task_9f1281` | Skip all-masked chunks in loss logging | Build loop fix |
| `task_f9e744` | Baseline comparison: train matched transformer | Experimental |
| `task_41186a` | Generate external-notebook curriculum | Data pipeline |
| `task_97ffb6` | Generate HADES graph-reasoning curriculum | Data pipeline |

---

## Summary

| Stage | Milestones | Tests | Status |
|-------|-----------|-------|--------|
| Stage 0: Foundation | 3 | 57 + 47 + 98 | COMPLETE |
| Stage 1: Algorithm Core | 19 | 778 Rust + 27 Python | COMPLETE |
| Stage 2: Production Infra | 4 (+M1a, +M1b) | 33 CUDA + 13 dispatch + 6 GPU-resident + 20 edge + 18 serving + 18 distributed | COMPLETE |
| Stage 3: Extensions | 5 | 22 retention + 35 M3/variants + 26 Atlas + 22 dynamic freq = 105 | COMPLETE |
| Stage 3b-Critical: HOPE Path | 3 impl milestones (DGD, self-ref, outer-loop+HOPE) | — | NOT STARTED (blocked on S4-M7) |
| Stage 3b-Deferred: MIRAS Completeness | 3 impl milestones (algo knobs, retention/bias, AdaMuon) | — | DEFERRED (Stage 5) |
| Stage 4 Phase 1: Pipeline Scaffolding | 8 (7 done, 1 immediate) | 27 Python + 120 tape/traced/class3 Rust | S4-M7 IMMEDIATE |
| Stage 4 Phase 2: HOPE Build & Serve | 6 milestones (M9–M14) | — | BLOCKED ON S3b-Critical |

**Current position**: S0–S3 complete. S3b specs complete (all 20 written, `v0.4.0` in HADES `hecate_specs`). S4 Phase 1 pipeline delivered (M1–M6, M8): can build a model on real text data and serve it locally using pre-S3b primitives. Wengert tape is the production gradient path (M8, PRs #55–65). HADES `hope_blockers`: 2 resolved (k and f_i), 2 deferred (brain transplant only).

**Active fronts** (updated 2026-02-22 per committee review):
1. **S4-M7: Primitive Validation** (IMMEDIATE): Run curriculum_100k config to validate pipeline at scale. Profile Wengert tape memory. Add per-level diagnostics. Implement Forget Gate and Curriculum Integration behavioral probes. **This runs first.**
2. **S3b-Critical** (after S4-M7): DGD (S3b-M1) first — it unblocks self-referential (S3b-M3). AdamW (S3b-S16) and Short Conv (S3b-S19) run parallel with DGD. HOPE composition (S3b-S20) blocked on S3b-M3.
3. **S4 Phase 2** (blocked on S3b-Critical): the real HOPE build — DGD inner-loop + self-referential projections + AdamW outer-loop — cannot begin until S3b-Critical is complete.
4. **S3b-Deferred / Stage 5** (post-HOPE): MIRAS design-space completeness. Retention variants, bias variants, DMGD, FTRL, Implicit GD, NS inner, AdaMuon.

Total test count: 940 Rust + 27 Python = **967 tests** (36 PRs merged; full `cargo test`).
