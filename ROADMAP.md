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

## Stage 3b: Primitive Completeness (SPECS COMPLETE — IMPLEMENTATION IN PROGRESS)

The NL paper suite defines ~50 primitives. Stages 1–3 implemented 26. This stage completes the primitive inventory. **All 20 spec sheets are written and stored in HADES (status: `v0.4.0`).** The spec-to-code pipeline is unblocked — implementation is the remaining work. No spec writing is needed — go straight to implementation.

The Hope architecture (self-modifying Titans + CMS) is the castle. These are the lego pieces. All specs confirmed present on disk in `specs/` and recorded in HADES `hecate_specs` collection. **Implementation is underway**: DGD (S3b-M1) and DGD CUDA (S3b-M5) are complete. Self-referential projections (GAP-L, PR #115), self-generated values (GAP-M, PR #116), and chunkwise self-ref training (GAP-N, PR #117) delivered. Feature maps (GAP-E) remaining.

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
| **S3b-M1: DGD** | S3b-S1 only | `task_38485e` (GAP-I) | S4-M7 (validate pipeline first) | **COMPLETE** (PR #113) |
| **S3b-M5: DGD CUDA Kernel Pair** | S3b-S1 (CUDA tier) | `task_0380eb` | S3b-M1 (DGD impl required first) | **COMPLETE** (PR #114) |
| **S3b-M3: Self-Referential Primitives** | S3b-S12 through S3b-S15 | `task_929845` (GAP-L), `task_eb31fd` (GAP-M), `task_79f2c5` (GAP-E), `task_4a9b1d` (GAP-N) | S3b-M1 (DGD impl required first) | **IN PROGRESS** — GAP-L (PR #115), GAP-M (PR #116), GAP-N (PR #117) all delivered; GAP-E remaining |
| **S3b-M4-critical: Outer-Loop & HOPE** | S3b-S16, S3b-S19, S3b-S20 | `task_a27d3a` (AdamW impl), `task_d7e2dd` (Conv1D impl), `task_d55634` (HOPE composition impl) | S3b-S16/S19 parallel with M1; S3b-S20 blocked on M3 | **COMPLETE** — AdamW (PR #118), Conv1D (PR #108), HOPE composition (PR #96) |

#### S3b-M1: DGD — Delta Gradient Descent ✅

**Spec**: `specs/algorithms/optimization_machinery/03_dgd.md`

Extracted DGD as a standalone composable primitive from its inline implementation in the Delta Rule. DGD's update depends on both the current input AND the current memory state M (HOPE Eq 88: `M = (1-α)M - θ·outer(M@k - v, k)`), making it fundamentally more expressive than plain GD.

**What was delivered**: `core/src/dgd.rs` (~250 lines) with `dgd_step()` (in-place forward), `dgd_step_backward()` (analytical gradients for dM, dk, dv, dα, dθ), `DgdCache` for backward. Loss Monotonicity Probe validates inner-loop loss decreases monotonically. All existing rules refactored to call `dgd_step()` instead of inline DGD math. 16 tests (unit + FD gradient checks + monotonicity probe).

**PR**: #113

---

#### S3b-M5: DGD CUDA Kernel Pair ✅

DGD CUDA forward + backward kernels, following the established kernel-pair pattern from S2-M1. Projections and gates remain in Rust (tape-differentiable); only the sequential M recurrence loop is CUDA.

**What was delivered**: `core/kernels/dgd_forward.cu` + `core/kernels/dgd_backward.cu`. Forward: per-head blocks, shared memory for M recurrence, warp reduction for dot products. Backward: analytical gradients matching `dgd_step_backward()`. Multi-arch fat binary (sm_86/89/90 + PTX). Feature-gated `#[cfg(feature = "cuda")]`.

**PR**: #114

---

#### S3b-M3: Self-Referential Primitives (3/4 delivered, GAP-E remaining)

**Spec**: `specs/algorithms/self_referential/00_interface.md` through `03_chunkwise_self_ref.md`

Phase 2 of the HOPE architecture: all 6 memories (M_k, M_v, M_q, M_eta, M_alpha, M_mem) use DGD to produce adaptive projections instead of static W @ x (HOPE Eqs 79-82, 85, 88).

**GAP-L: Self-referential projections (PR #115)** — Delivered:
- `core/src/self_ref.rs` (~1050 lines): `SelfRefState`, `SelfRefCache`, `SelfRefParamGrads`, `self_ref_step()` (forward), `self_ref_step_backward()` (backward), `self_ref_read_only()` + backward for frozen CMS levels
- `ProjectionKind` enum (Static default, Adaptive) on `MAGConfig`
- Wired into `run_level_memory()` dispatch for MAG/MAC/MAL
- `MemoryCache::SelfRef` variant for backward dispatch
- 19 tests (smoke, static-identical, FD gradient checks, read-only backward)
- **PR 4 (PR #119, merged)**: Wire `SelfRefParamGrads` into outer-loop optimizer — 6 `m_*_init` fields on `MemoryLevelParams`, `SelfRefState::from_init()`, `ContextState::seed_self_ref()`, 4 backward sites wired, AdamW bufs 16→22, PyO3 flat weight serialization, checkpoint roundtrip. 1085 tests.

**GAP-M: Self-generated values (PR #116)** — Delivered:
- Phase 3 of HOPE architecture: each of 6 memories generates its own DGD target via `v̂_□ = M_{□,t-1}(v_t)` (HOPE Eq 84-85)
- DGD key alignment fix: all 6 component DGD calls now use `k_t` as key (matching paper Eq 88), not `x_t` for 5 components
- `self_generated_values: bool` on `MAGConfig` (default false = Phase 2 behavior)
- `v_hat_targets` cache field for backward (empty vec when Phase 2 = zero overhead)
- Forward: Step 3.5 computes v̂ per memory, writes to cache, passes slices to DGD
- Backward: chain rule through `v̂ = M @ v_t` → `dM += outer(dv̂, v_t)`, `dv_t += M^T @ dv̂`
- 7 new tests (identity init, target divergence, backward grads, 2 FD checks, phase2 equivalence, key fix validation)

**GAP-N: Chunkwise self-referential training (PR #117)** — Delivered:
- `core/src/chunkwise_self_ref.rs` (~1080 lines): `ChunkwiseSelfRefCache`, `chunkwise_self_ref_step()` (forward), `chunkwise_self_ref_step_backward()` (backward), `dgd_frozen_backward()` helper
- Frozen M snapshots at chunk boundaries (HOPE §8.2, Eqs 90-93): DGD error uses M_frozen, recurrence uses current M
- Gradient split: retention chain `(1-α)·d_m_out` → M_prev, error chain `−θ·outer(d_m_out@k, k)` → M_frozen with per-chunk accumulation and boundary merging
- `self_ref_chunk_size: usize` on `MAGConfig` (default 1 = sequential, >1 = chunkwise)
- `MemoryCache::ChunkwiseSelfRef` variant with full dispatch wiring (mag.rs, mal.rs, mac.rs, gradient.rs, chunkwise_gd.rs)
- Self-generated values support: `v̂ = M_frozen @ v_t` when enabled
- C=1 bit-identical to sequential `self_ref_step()` (verified by test)
- Python bindings: `self_ref_chunk_size` exposed as optional parameter (default 1)
- 15 tests (C=1 correctness, C=2/C=4 forward finite, backward shapes/finite/nonzero, FD gradient checks at C=1 and C=2, approximation bounds, convergence, self-gen frozen validation, remainder chunks)

**Remaining sub-tasks**:
- GAP-E (`task_79f2c5`): Feature maps — `φ(k)` hook and `FeatureMapKind` enum — NOT STARTED

---

### S3b-Deferred: MIRAS Completeness (Stage 5 — post-HOPE)

These primitives complete the MIRAS design space but are not required for the HOPE architecture. They will be implemented after the HOPE model is building and serving successfully.

| Milestone | Implements Specs | HADES Task(s) | Status |
|-----------|-----------------|---------------|--------|
| **S3b-M1-deferred: Algorithm Knobs** | S3b-S2, S3b-S3, S3b-S4, S3b-S5 | `task_544d8d` (GAP-J: Implicit GD), `task_c48b71` (GAP-K: FTRL) | DEFERRED (Stage 5) |
| **S3b-M2: Retention & Bias** | S3b-S6 through S3b-S11 | `task_60e757` (GAP-F: Bregman), `task_eb6a4f` (GAP-G: L_q), `task_f18e65` (GAP-H: sigmoid), `task_667d92` (GAP-D: KL bias) | DEFERRED (Stage 5) |
| **S3b-M4-deferred: AdaMuon** | S3b-S17 | **MISSING** | DEFERRED (Stage 5) |

### Priority Order

```text
S4-M7: Primitive Validation ──────────────── COMPLETE (PR #112)
    │
    ▼
S3b-M1: DGD ──────────────────────────────── COMPLETE (PR #113)
    │
    ├─► S3b-M5: DGD CUDA ────────────────── COMPLETE (PR #114)
    ├─► S3b-S16: AdamW outer-loop ────────── COMPLETE (PR #118)
    ├─► S3b-S19: Short Conv1D ────────────── COMPLETE (PR #108)
    │
    ▼
S3b-M3: Self-Referential Primitives ──────── GAP-L+M+N COMPLETE, GAP-E remaining
    │
    ▼
S3b-S20: HOPE Composition ────────────────── COMPLETE (PR #96)
    │
    ▼
S4 Phase 2: HOPE Build & Serve ───────────── IN PROGRESS (Phase 0 running)
    ├─► S4-M10: DGD Build Path ────────────── COMPLETE (PR #120)
    ├─► S4-M11: Self-Referential Build ────── COMPLETE (PRs #115-121)
    ├─► S4-M12: AdamW Outer-Loop ──────────── COMPLETE (PR #118)
    ├─► S4-M13: HOPE Model Config ─────────── COMPLETE (PR #120)
    └─► S4-M14: End-to-End Validation ─────── IN PROGRESS (Phase 0 build running)

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

**File**: Originally `python/build.py` (~150 lines), now `python/engine/loop.py` (~580 lines) via unified `python/hecate.py --build` entry point (PR #122).
**PR**: #38 (original), #122 (unification)

---

### S4-M4: Serve Script ✅

Load a checkpoint and interactively generate text. Autoregressive byte generation with temperature-controlled sampling. Chat mode, one-shot prompt, and interactive REPL.

**File**: Originally `python/serve.py` (~150 lines), now `python/engine/chat.py` (~150 lines) + `python/engine/generation.py` (~305 lines) via unified `python/hecate.py --chat/--prompt/--interactive` entry point (PR #122).
**PR**: #38 (original), #122 (unification)

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

### S4-M7: Primitive Validation ✅

**HADES task**: `task_e4aab3`

**Elevated to immediate priority per 2026-02-22 committee review.** The build pipeline exists and is idle. Three specific risks need validation before adding DGD/self-referential complexity: (1) Wengert tape memory at 60M scale, (2) CMS 1/sqrt(k) normalization in a real network, (3) MLP inner-loop dominance at scale.

Run `curriculum_100k.json` (d=1024, k=4, 100K steps) or equivalent end-to-end. This validates that the build→serve pipeline works with existing primitives (Titans LMM + MAG + k=4 CMS + plain SGD outer-loop + byte tokenizer). It does NOT validate the HOPE architecture — that is S4 Phase 2.

Success criteria (hard thresholds — pre-committed, do not adjust post-hoc):

| Criterion | Hard Threshold | Stop-the-Line | Implementation |
|-----------|---------------|---------------|----------------|
| **Loss convergence** | Validation loss must decrease by ≥15% between step 50K and step 100K | If <10% decrease, pipeline is broken | `check_loss_convergence()` — averages loss over 50K-55K and 95K-100K windows |
| **No NaN/Inf** | Zero NaN/Inf values in any tensor through full run | Any NaN = immediate stop | `check_no_nan_inf()` — scans all step events + abort events |
| **CMS Level 3 gate activity** | Level 3 must fire with non-degenerate gate (softplus(b_theta) > 0.001) ≥50 times over the full run | If <25 active fires over full run, CMS is degenerate | `check_level3_activity()` — reads `level3_summary` or sums `level3_activity` deltas |
| **Wengert tape memory scaling** | Per-token memory must not exceed 1.1× increase across seq_len 512→2048 | If ≥1.2× increase, tape has quadratic leak | `check_tape_scaling()` — reads `tape_memory_ratio` from `build_start` event |
| **Checkpoint roundtrip** | Loss after restore must be within 1e-6 of loss before save | Any deviation = serialization bug | `check_checkpoint_roundtrip()` — checks all `checkpoint_roundtrip` event deltas |
| **Curriculum probe** | Stories loss at step 55K+ must be < 2× minimum stories loss from steps 0-25K | Catastrophic forgetting at phase boundary | `check_curriculum_probe()` — compares `phase_boundary` event losses |

**Threshold notes**:
- The 1.1× tape scaling bound is a leak detector: the tape records per-token operations, so per-token memory should scale linearly (ratio ≈ 1.0). The bound catches quadratic tape growth, not model-specific behavior.
- The ≥50 active Level 3 fires threshold applies over the **entire 100K run** (not per-1000-steps). With `chunk_sizes=[1,8,64,512]`, Level 3 fires every 512 Conductor steps, yielding ~195 total fires over 100K steps. The threshold requires ~25% to have non-degenerate gates.
- The 15% loss decrease is a conservative pipeline-is-working gate, not a model quality target. Pre-HOPE primitives (Titans LMM + MAG + plain SGD) are not expected to produce competitive perplexity.

**Validation implementation**: `python/validate_run.py` — post-run script that checks all 6 thresholds against JSONL logs. Standalone profiling for tape memory via `python/profile_tape.py` (measures RSS at seq_len=512 vs 2048).

**What was delivered** (PR #112):
- **Forget Gate Probe**: `core/tests/test_cms.rs::test_forget_gate_probe` — feeds contradictory sequence, asserts Level 0 converges to new value faster than Level 1+. Validates CMS frequency separation.
- **Curriculum data pipeline**: `python/data/prepare_curriculum.py` — multi-phase curriculum generation with distribution boundaries for integration probe.
- **Curriculum config**: `python/configs/curriculum_100k.json` — d=1024, k=4, 100K steps, Titans LMM + MAG.
- **Build loop diagnostics**: `python/build.py` enhanced with per-level gate activity logging, memory norms, gradient norms per CMS level, and curriculum phase boundary tracking (JSONL output).
- **Validation harness**: `python/validate_run.py` — post-run script that checks all 6 hard thresholds against JSONL logs.
- **Tape profiling**: `python/profile_tape.py` — measures Wengert tape memory consumption and backward-vs-forward time ratio across seq_len sweep.
- **PyO3 bindings**: `python/src/lib.rs` additions for tape profiling entry points.

**Dependencies**: S4-M6 (model design + config)
**Blocks**: S3b-M1 (validate pipeline before adding DGD complexity)
**PR**: #112

---

## Stage 4 Phase 2: HOPE Build & Serve (IN PROGRESS)

S3b-Critical delivered the primitives. Phase 2 milestones are now active. Phase 0 build (TinyStories 100K steps, d=512, k=4, Titans LMM + MAG + adaptive projection + adamw_gpu) is running as of 2026-02-24.

### S4-M9: Multi-Block CMS Execution Engine (PLANNED)

S3-M3 (CMS Variants) built the configuration schema and validation for 5 deployment patterns (Basic, Nested, Sequential, Independent, Hybrid). What is missing is the actual execution engine that runs multiple blocks with independent or hierarchical CMS schedules.

**What this delivers**:
- Runtime that instantiates and steps multiple `BlockConfig` entries
- Block-level output aggregation (concat, sum, or attention-mix)
- Independent Conductor instances per block (for Independent/Hybrid variants)
- Integration with distributed_step for multi-block multi-GPU

**Dependencies**: S3-M3 (CMS Variant schemas, COMPLETE), S3b-M4 (HOPE composition spec, COMPLETE)
**Status**: NOT STARTED — deferred until single-block HOPE is validated (S4-M14)

---

### S4-M10: DGD Inner-Loop Build Path ✅

**HADES task**: `task_30e20a` (IMPL: build.py HOPE model configuration and build loop)

DGD is the default inner-loop optimizer for HOPE builds. The build config (`configs/phase0_warmup.json`) uses `algorithm: "dgd"` with DGD-specific parameters (`use_dgd: true`). DGD's update depends on both the current input AND the current memory state M (HOPE Eq 88), making it fundamentally more expressive than plain GD.

**What was delivered** (PR #120):
- `hecate.py --build` configures DGD inner-loop via config JSON
- `configs/phase0_warmup.json`: d=512, k=4, Titans LMM + MAG + adaptive projection + DGD
- DGD build loop converges (Phase 0 running, loss 10.37 → 4.3 by step 2000)

**Dependencies**: S3b-M1 (DGD implementation, COMPLETE)
**Status**: **COMPLETE** (PR #120)

---

### S4-M11: Self-Referential Build ✅

**HADES tasks**: `task_929845` (GAP-L), `task_eb31fd` (GAP-M), `task_4a9b1d` (GAP-N), `task_30e20a` (HOPE build config)

Self-referential projections are wired into the build path. All 6 memories (M_k, M_v, M_q, M_eta, M_alpha, M_mem) produce adaptive projections via DGD instead of static W @ x. The build config enables this via `projection_kind: "adaptive"` and `self_generated_values: true`.

**What was delivered** (PRs #115-117, #119, #120, #121):
- Self-referential projections (GAP-L, PR #115): `SelfRefState`, `SelfRefCache`, forward+backward
- Self-generated values (GAP-M, PR #116): `v_hat = M @ v_t`, DGD key alignment fix
- Chunkwise self-referential training (GAP-N, PR #117): frozen M snapshots at chunk boundaries
- Outer-loop wiring (PR #119): `SelfRefParamGrads` into AdamW, 6 `m_*_init` fields, AdamW bufs 16→22
- Build config (PR #120): HOPE fields wired through Python tier
- GPU grad shape fix (PR #121): backward grad shapes for adaptive projections

**Dependencies**: S3b-M3 (self-referential primitives, COMPLETE), S4-M10 (DGD build, COMPLETE)
**Status**: **COMPLETE** (PRs #115-121)

---

### S4-M12: AdamW Outer-Loop ✅

**HADES task**: `task_a27d3a` (IMPL: AdamW frequency-aware outer-loop optimizer)

Frequency-aware AdamW replaces plain SGD for outer-loop weight updates. Per-level bias correction counters account for CMS frequency gating — Level 3 parameters see fewer updates and need different momentum correction than Level 0. GPU path (`adamw_gpu`) is fully fused on device; CPU `adamw` auto-promotes to `adamw_gpu` when GPU is active.

**What was delivered** (PRs #118, #122):
- `FrequencyAwareAdamW` in Rust: per-level bias correction, Pulse-based gating (PR #118)
- `apply_weight_gradients_adamw()` on MAGParams + GPU variant
- `hecate.py --build` auto-promotes `adamw` → `adamw_gpu` when GPU detected (PR #122)
- `--optimizer` flag: `"sgd"` / `"adamw"` / `"adamw_gpu"`
- Phase 0 build running with adamw_gpu, loss converging

**Dependencies**: S3b-S16 (AdamW spec, COMPLETE)
**Status**: **COMPLETE** (PR #118). AdaMuon deferred to Stage 5.

---

### S4-M13: HOPE Model Config ✅

**HADES task**: `task_30e20a` (IMPL: build.py HOPE model configuration and build loop), `task_4a10eb` (SPEC: build.py HOPE model configuration)

The HOPE architecture config: self-referential Titans + DGD inner-loop + k=4 CMS + AdamW outer-loop. The Phase 0 config (`configs/phase0_warmup.json`) is the first concrete HOPE build target.

**What was delivered** (PR #120):
- `configs/phase0_warmup.json`: d=512, heads=8, seq_len=512, k=4 CMS [1,8,64,512], Titans LMM + MAG, adaptive projection, self-generated values, DGD inner-loop, adamw outer-loop, lr=0.0006
- All HOPE config fields wired through Python tier to Rust: `projection_kind`, `self_generated_values`, `self_ref_chunk_size`, `use_dgd`, `no_self_generated_values`
- Checkpoint resume copies HOPE fields from checkpoint config
- Input validation for self-ref parameter seeding

**Dependencies**: S4-M10 (DGD, COMPLETE), S4-M11 (self-ref, COMPLETE), S4-M12 (AdamW, COMPLETE), S3b-S20 (HOPE composition, COMPLETE)
**Status**: **COMPLETE** (PR #120)

---

### S4-M14: HOPE End-to-End Validation (IN PROGRESS)

**HADES task**: `task_08ca2e` (HOPE NLM Phase 0+1 Training)

Phase 0 build running on GPU0 as of 2026-02-24. Config: `phase0_warmup.json` (d=512, k=4, TinyStories, 100K steps). Loss trajectory: 10.37 (step 0) → 4.3 (step 2000). All HOPE primitives active: DGD inner-loop, adaptive projections, self-generated values, k=4 CMS, adamw_gpu outer-loop. Memory stable at 4081MB RSS.

Success criteria:
1. Loss converges on TinyStories data — ≥15% decrease between step 50K and 100K
2. No NaN/Inf through full run
3. CMS Level 3 gate activity: ≥50 non-degenerate fires over full run
4. Wengert tape memory scaling: ≤1.1× per-token increase (512→2048)
5. Checkpoint roundtrip: delta < 1e-6
6. Serve: generate text after build completes (informational — quality is Phase 1+ target)

**Validation**: Run `python validate_run.py runs/phase0_100k.jsonl` after build completes.

**Dependencies**: S4-M13 (HOPE config, COMPLETE), all S4 Phase 2 milestones (COMPLETE)
**Status**: **IN PROGRESS** — build running, ~2% complete

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

1053 tests validate code correctness (matrix math, gradient parity, serialization roundtrips). Only 1 test (`test_k2_beats_k1_multiscale`) validates learning physics. The following probes address this gap. Each is tied to a specific milestone and validates a distinct learning property.

| Probe | What it validates | When implemented | Location |
|-------|-------------------|------------------|----------|
| **Forget Gate Probe** | CMS frequency separation — fast levels update faster than slow levels on contradictory input | S4-M7 | `test_cms.rs` |
| **Loss Monotonicity Probe** | DGD inner-loop loss decreases monotonically across optimization steps within a single forward pass | S3b-M1 (DGD) | `test_dgd.rs` (new) |
| **Curriculum Integration Probe** | Model does not catastrophically forget previous distribution at curriculum phase boundaries | S4-M7 | `build.py` logging |

**Forget Gate Probe**: Feed a contradictory sequence (A=1 for N steps, then A=2 for N steps). After the switch, Level 0 (fires every step) should converge to A=2 faster than Level 1 (fires every 8th step). This validates the core CMS claim: higher-frequency levels adapt faster to new information.

**Loss Monotonicity Probe**: When DGD is implemented (S3b-M1), add a runtime assertion that the inner-loop loss decreases monotonically across DGD's optimization steps. If loss oscillates within the inner loop, the test fails even if the matrix math is correct. This directly validates HOPE Eq 88.

**Curriculum Integration Probe**: During S4-M7 validation, log eval loss on BOTH the outgoing and incoming distribution at every curriculum phase boundary. Verify the model does not catastrophically forget — some regression is expected, catastrophic collapse is a bug.

**HADES graph registration**: All probes are registered as formal nodes in the `hope_probes` collection with full schema (name, description, probe_type, paper_basis, checks, traced_to, failure_means). The CMS frequency separation invariant is registered as a behavioral axiom in `hope_axioms`. See `hades --db NL db aql "FOR doc IN hope_probes RETURN doc._key"` for the full list. Each probe links to the paper equation it validates via `paper_basis.equation` and to the spec it implements via `traced_to[]`.

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
                    ┌─► S4-M7: Primitive Validation ── COMPLETE (PR #112)
                    │       + Forget Gate Probe ✅
                    │       + Curriculum Integration Probe ✅
                    │       + Tape profiling ✅
                    │
                    ▼
── S3b-Critical ────────────────────── HOPE PATH (COMPLETE except GAP-E)
    │
    ├─► S3b-M1: DGD ──────────────────── COMPLETE (PR #113)
    │       │   + Loss Monotonicity Probe ✅
    │       ├─► S3b-M5: DGD CUDA ─────── COMPLETE (PR #114)
    │       └─► S4-M10: DGD Build Path ── COMPLETE (PR #120)
    │               └─► S4-M11: Self-Referential Build ── COMPLETE (PRs #115-121)
    │
    ├─► S3b-S16: AdamW outer-loop ─────── COMPLETE (PR #118)
    │       └─► S4-M12: AdamW Outer-Loop ── COMPLETE (PR #118)
    │
    ├─► S3b-S19: Short Conv1D ─────────── COMPLETE (PR #108)
    │
    ├─► S3b-M3: Self-Referential ──────── GAP-L #115 + GAP-M #116 + GAP-N #117 COMPLETE; GAP-E pending
    │
    └─► S3b-S20: HOPE Composition ─────── COMPLETE (PR #96)
            │
            └─► S4-M13: HOPE Model Config ── COMPLETE (PR #120)
                    └─► S4-M14: HOPE End-to-End Validation ── IN PROGRESS (Phase 0 running)

── S3b-Deferred (Stage 5) ─────────── POST-HOPE
    ├─► DMGD, FTRL, Implicit GD, NS inner
    ├─► Bregman, L_q, sigmoid-bounded retention
    ├─► KL bias, l_1 bias, l_p dispatch
    └─► AdaMuon
```

**Stage 2** milestones are sequential. **Stage 3** milestones are independent. **S3b-Critical** is complete (all HOPE-path primitives delivered) except GAP-E (feature maps). **Stage 4 Phase 2** (M10–M14) is active: M10-M13 COMPLETE, M14 (end-to-end validation) IN PROGRESS with Phase 0 build running. **S3b-Deferred** (Stage 5) contains MIRAS design-space completeness work — implemented after the HOPE model is validated.

---

## HADES Task Cross-Reference (updated 2026-02-24)

Complete mapping between ROADMAP milestones and HADES Persephone tasks. Use this to avoid rebuilding context across sessions.

### HOPE Critical Path — Task Status

| Step | ROADMAP Milestone | HADES Task Key | HADES Title | Status |
|------|-------------------|---------------|-------------|--------|
| 1 | **S4-M7**: Primitive Validation | `task_e4aab3` | S4-M7: Primitive Validation Run | **closed** (PR #112) |
| 2a | **S3b-M1**: DGD | `task_38485e` | GAP-I: Extract DGD as named composable primitive | **closed** (PR #113) |
| 2a+ | **S3b-M5**: DGD CUDA | `task_0380eb` | S3b-M5: DGD CUDA kernel pair implementation | **closed** (PR #114) |
| 2b | **S3b-S16**: AdamW outer-loop (impl) | `task_a27d3a` | IMPL: AdamW frequency-aware outer-loop optimizer | **closed** (PR #118) |
| 2c | **S3b-S19**: Short Conv1D (impl) | `task_d7e2dd` | PS-BLK-04: Conv1D preprocessing before memory module | **closed** (PR #108) |
| 3a | **S3b-M3/S12**: Self-ref projections | `task_929845` | GAP-L: Self-referential Phase 2 — adaptive projection memories | **closed** (PR #115, wiring PR #119) |
| 3b | **S3b-M3/S13**: Self-generated values | `task_eb31fd` | GAP-M: Self-generated values — v_hat = M_square(v_t) | **closed** (PR #116) |
| 3c | **S3b-M3/S14**: Feature maps | `task_79f2c5` | GAP-E: Feature maps — phi() hook and FeatureMapKind enum | **open** |
| 3d | **S3b-M3/S15**: Chunkwise self-ref | `task_4a9b1d` | GAP-N: Chunkwise training for self-referential memories | **closed** (PR #117) |
| 4 | **S3b-S20**: HOPE Composition (impl) | `task_d55634` | PS-TC-03: HOPE level-level composition Variants 1/3/4 | **closed** (PR #96) |
| 5a | **S4-M10**: DGD Build Path | `task_30e20a` | IMPL: build.py HOPE model configuration and build loop | **closed** (PR #120) |
| 5b | **S4-M11**: Self-Referential Build | `task_929845`+`task_eb31fd`+`task_4a9b1d` | GAP-L + GAP-M + GAP-N (aggregate) | **closed** (PRs #115-117, #119, #121) |
| 5c | **S4-M12**: AdamW Outer-Loop | `task_a27d3a` | IMPL: AdamW frequency-aware outer-loop optimizer | **closed** (PR #118) |
| 6 | **S4-M13**: HOPE Model Config | `task_30e20a` + `task_4a10eb` | IMPL + SPEC: build.py HOPE model configuration | **closed** (PR #120) |
| 7 | **S4-M14**: HOPE End-to-End Validation | `task_08ca2e` | HOPE NLM Phase 0+1 Training | **in progress** (Phase 0 build running) |

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
| `task_08ca2e` | HOPE NLM Phase 0+1 Training | **S4-M14** — Phase 0 build running |
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
| Stage 3b-Critical: HOPE Path | 4 impl milestones (DGD, self-ref, outer-loop+HOPE, DGD CUDA) | DGD: 16, DGD CUDA: ~10, self-ref: 41, AdamW: ~20, Conv1D: ~15, HOPE comp: ~12 | COMPLETE (except GAP-E feature maps) |
| Stage 3b-Deferred: MIRAS Completeness | 3 impl milestones (algo knobs, retention/bias, AdaMuon) | — | DEFERRED (Stage 5) |
| Stage 4 Phase 1: Pipeline Scaffolding | 8 | 27 Python + 120 tape/traced/class3 Rust + 1 forget gate probe | COMPLETE |
| Stage 4 Phase 2: HOPE Build & Serve | 6 milestones (M9–M14) | — | M10-M13 COMPLETE, M14 IN PROGRESS, M9 deferred |

**Current position** (updated 2026-02-24): S0–S3 complete. S3b-Critical complete (all HOPE-path primitives delivered, GAP-E feature maps remaining). S4 Phase 1 complete (M1–M8). **S4 Phase 2 active**: M10 (DGD build path, PR #120), M11 (self-referential build, PRs #115-121), M12 (AdamW outer-loop, PR #118), M13 (HOPE model config, PR #120) all COMPLETE. **M14 (end-to-end validation) IN PROGRESS**: Phase 0 build running on GPU0 (TinyStories 100K steps, d=512, k=4, loss 10.37→4.3 at step 2000). Unified `hecate.py` entry point with GPU-default paradigm (PR #122, draft).

**HADES graph state**: 7 probes registered in `hope_probes` (forget gate, loss monotonicity, slow-gradient-zero, fast-gradient-nonzero, self-modification-matches-eq88, frozen-state-structural, transplant-integrity). 3 axioms in `hope_axioms` (CMS frequency separation, container-is, container-is-not). 50 tasks closed, 16 open. `hope_blockers`: 2 resolved (k and f_i), 2 deferred (brain transplant only).

**Active fronts**:
1. **S4-M14 End-to-End Validation** (IN PROGRESS): Phase 0 build running. Post-run: `python validate_run.py runs/phase0_100k.jsonl` checks all 6 hard thresholds.
2. **PR #122** (DRAFT): Unify build.py + serve.py into hecate.py + engine/ package. GPU-default paradigm. In review.
3. **GAP-E** (open): Feature maps — `phi(k)` hook and `FeatureMapKind` enum. Last S3b-M3 sub-task.
4. **S3b-Deferred / Stage 5** (post-HOPE): MIRAS design-space completeness. Retention variants, bias variants, DMGD, FTRL, Implicit GD, NS inner, AdaMuon.

Total test count: 1379 Rust + 27 Python = **1,406 tests** (121 PRs merged; full `cargo test`).
