# NL_Hecate Progress Report

**Project**: NL_Hecate — Nested Learning implementation in Rust + Enzyme AD + CUDA
**Status**: Stages 0-2 COMPLETE. Stage 3 IN PROGRESS (S3-M1 Pluggable Retention complete).

---

## Executive Summary

NL_Hecate implements the Nested Learning research program (Mirrokni/Behrouz, Google Research) in Rust with Enzyme AD and CUDA kernels. The project built the essential forward/backward/optimize pipeline — 8 memory rules, 3 composition patterns, 5 parallelization strategies, CUDA kernel pairs, multi-GPU sync, serving, and edge deployment — that replaces PyTorch's training/inference stack with a unified self-modifying forward pass.

An integration spike (17 tests) validates the thesis end-to-end: the full VecStream -> Conductor -> cms_forward -> cms_backward -> apply pipeline learns a repeating token pattern, achieving 100% prediction accuracy across 3 representative configs. The serving path (Session::process_chunk) produces identical behavior to the raw loop.

**Total test count**: 894 Rust + 27 Python = **921 total**
**PRs merged**: 30
**Codebase**: ~25.3K lines Rust source + ~9.9K lines Rust tests + ~1.3K lines CUDA + ~1.2K lines Python

---

## Stage Summary

### Stage 0: Foundation (COMPLETE)

**Phase 0: Enzyme Spike** (57/57 tests)
- Proved Enzyme differentiates through Rust trait dispatch
- Manual chain-rule composition at kernel boundaries works
- Toolchain pinned at SHA d7daac06 (rustc 1.95.0-nightly)

**Track Zero-A: Pure SWA Attention** (67 tests)
- Rust core: forward/backward, 6 weight matrices gradient-checked vs FD
- CUDA kernels: SWA forward+backward pair, feature-gated dispatch, warp reduction
- PyO3 bindings: Maturin build, Python API
- PyTorch regression baseline: validates Rust matches PyTorch

**Track Zero-B: Delta Rule + MAG** (98 tests)
- MemoryRule trait, DeltaRule impl, MAG dual-branch composition, sigmoid gating
- 7 memory weight matrices gradient-checked
- Gate bias init: b_alpha=3.0 (sigmoid~0.95), b_theta=-4.6 (softplus~0.01)

### Stage 1: Algorithm Core (COMPLETE — 778 Rust + 27 Python = 805 tests)

All 22 milestones delivered. This is the mathematical heart of the system — everything that PyTorch's autograd + optimizer + DataLoader + model.train()/model_eval() does, reimplemented as a single self-modifying forward pass.

**Memory Rules (8/8)**:
- Delta Rule (GD, matrix structure, L2 bias)
- Titans LMM (GD+momentum, eta gate, momentum accumulator S)
- Hebbian Rule (direct correlation, no gradient descent)
- MONETA (2-layer MLP, l_p bias, elastic net retention)
- YAAD (Huber loss, decoupled local+global retention)
- MEMORA (KL-softmax bias, emergence-based retention)
- Lattice OSR (orthogonal state recurrence, slot-based compression)
- Trellis (two-pass KV compression, separate key/value decay)

**Composition Patterns (3/3)**:
- MAG: Memory gates attention output via sigmoid (parallel branches)
- MAL: Memory preprocesses input for attention (sequential, residual)
- MAC: Memory provides context, attention processes assembled input

**CMS Frequency Scheduling**:
- k=1 (single level), k=2 (two-frequency), k=4 (full hierarchy: 1/8/64/512)
- Conductor/Pulse timing, ErrorBuffer gradient accumulation, ContextState persistence
- 1/sqrt(k) output normalization for k>2

**Parallelization Strategies (5/5)**:
- Chunkwise GD (baseline)
- Associative Scan (parallel prefix)
- TNT Hierarchical (chunk+inter-chunk)
- Lattice GLA (gated linear attention)
- Atlas Parallel (memory-optimized)

**Infrastructure**:
- ContextStream: replaces DataLoader (no epochs, monotonic cursor, checkpoint-serializable)
- 100K stability sweep across all rule/composition/k combinations
- PyO3 bindings for all rules + compositions

### Stage 2: Production Infrastructure (COMPLETE — S2-M1 through S2-M4)

**S2-M1: CUDA Kernel Pairs + Compilation** (PRs #25, #29)
- Forward + backward kernels for SWA, Delta Rule, Titans LMM, Hebbian
- Composition dispatch kernels for MAG/MAL/MAC
- Multi-architecture fat binary (sm_86/89/90 SASS + PTX fallback)
- Backend enum + detect_gpu() + force_rust_reference() override
- Forward tolerance: 1e-5, backward: 1e-4 per-element vs Rust reference

**S2-M2: CMS-Aware Multi-GPU Gradient Sync** (PR #26)
- Replaces DDP: only active CMS levels synchronized (not all parameters every step)
- MockProcessGroup for testing without real multi-GPU hardware
- AllReduce averaging + ErrorBuffer integration for frozen levels

**S2-M3: Serving Non-Stationary Models** (PR #27)
- Session struct: per-user isolated ContextState + Conductor
- Two modes: Test (bounded) and Stream (unbounded via ContextStream) — no mode flag (CS-10)
- process_chunk calls cms_forward directly — same path as build (CS-18)
- LatencyTracker: average/worst/p99 for SLA validation
- Checkpoint/restore with pulse_id verification

**S2-M4: Edge Deployment** (PR #28)
- Zero-dependency micro models (d <= 128) on CPU
- Three profiles: inner-loop only, full NL, WASM (wasm32-unknown-unknown validated)
- ~34k tok/s on x86_64 for d=64 (exceeds 18k target)
- `#![feature(autodiff)]` gated behind `enzyme` feature for portability

### Stage 3: Extensions (IN PROGRESS — 1/5 milestones)

**S3-M1: Pluggable Retention** (PR #31)
- Extracted retention mechanisms from inline code in all 8 memory rules into `core/src/retention.rs`
- `RetentionKind` enum: L2WeightDecay, KLDivergence, ElasticNet (NEW), SphereNormalization
- 6 free functions + 2 in-place variants for zero-alloc hot paths
- `RetentionKind` field added to `MAGConfig` (24 constructors, all test files updated)
- Cross-rule retention swapping enabled (e.g. DeltaRule+ElasticNet) — CS-36 compliance
- PyO3 `retention` kwarg for Python-side configuration
- 22 dedicated retention tests + 17 inline unit tests, 847 base tests passing (0 failures)

### Integration Spike: End-to-End Validation

**17 tests** validating that the full pipeline actually learns a predictable pattern.

**Stage 1 tests (12)**: VecStream -> Conductor -> cms_forward -> cms_backward -> apply loop for 500 steps. Three configs (DeltaRule+MAG, TitansLMM+MAL, HebbianRule+MAG) all converge from random-chance loss (2.77) to near-zero, achieving 100% prediction accuracy on a repeating [0..8] token pattern.

**Stage 2 tests (5)**: Serving Session path produces identical behavior to raw loop. Checkpoint/restore yields identical loss trajectory. CUDA dispatch stub validates feature gating.

---

## Key Discoveries

### 1. CMS Stabilization via Nesting

The most significant finding: multi-level CMS nesting provides **implicit regularization** that extends the stable operating range of inner-loop learning rates.

| b_theta | softplus(b_theta) | k=1 | k=2 |
|---|---|---|---|
| 0.0 | 0.69 | converges (0.149) | converges (0.150, tie) |
| 1.0 | 1.31 | converges | converges (0.056) |
| **1.2** | **1.49** | **NaN at step ~9K** | **converges (0.055)** |
| 1.5 | 1.74 | NaN at step ~8.5K | NaN |

### 2. Inner Loop IS the Forward Pass

The serving module proves the central NL thesis: there is no train/eval distinction. Session::process_chunk calls cms_forward — the exact same function used during build. Memory self-modifies during inference. Per-token latency is O(1) with respect to context length because memory matrices are fixed-size (d times d).

### 3. MLP Rules: Inner Loop Dominates

For MONETA/YAAD/MEMORA (MLP-based memory structure), the outer-loop SGD barely affects loss. The MLP inner loop is expressive enough to fit without outer-loop weight changes. This suggests MLP memory rules may be more suitable for pure serving (no outer-loop needed) than matrix rules.

### 4. Learning Rate Scales as 1/sqrt(d)

The integration spike revealed that outer-loop learning rate must scale with model dimension. At d=8, lr=0.01 (used by d=64 unit tests) produces negligible weight changes — SWA gradient norm of 0.024, memory gradients ~1e-7. Scaling to lr=0.5 achieved convergence from 2.77 to 0.0007 in 500 steps. The relationship is approximately lr proportional to 1/sqrt(d).

### 5. Additive Level Composition Scaling

y_combined = SUM(level outputs) grows linearly with k. At k=4, the summed signal pushes sigmoid into saturation. Fixed with 1/sqrt(k) normalization (variance-preserving, applied only for k>2).

---

## Architecture

```text
core/src/                          (~25,300 lines, 32 modules)
  tensor.rs        — SIMD-friendly primitives, RNG, sigmoid, softplus
  swa.rs           — Sliding Window Attention forward/backward
  model.rs         — SWAConfig/Params, MAGConfig/Params, MemoryLevelParams
  forward.rs       — SWA forward pass
  backward.rs      — SWA backward pass (Enzyme AD)
  gradient.rs      — FD gradient checking, MAG/CMS/MAL/MAC gradient computation
  delta_rule.rs    — MemoryRule trait + DeltaRule (GD, matrix, L2)
  titans_lmm.rs    — TitansLMM (GD+momentum, eta gate)
  hebbian_rule.rs  — HebbianRule (direct correlation)
  moneta.rs        — MONETA (MLP, l_p, elastic net)
  yaad.rs          — YAAD (Huber, decoupled retention)
  memora.rs        — MEMORA (KL-softmax, emergence)
  lattice_osr.rs   — Lattice OSR (orthogonal state recurrence)
  trellis.rs       — Trellis (two-pass KV compression)
  retention.rs     — Pluggable retention (L2/KL/ElasticNet/Sphere) + in-place variants
  mag.rs           — MAG composition + CMS forward/backward
  mal.rs           — MAL composition
  mac.rs           — MAC composition
  dispatch.rs      — CPU/CUDA feature-gated dispatch
  conductor.rs     — Conductor/Pulse/ContextState/ErrorBuffer
  context_stream.rs — ContextStream trait + VecStream
  parallel.rs      — Parallelization strategy configs
  chunkwise_gd.rs  — Chunkwise GD parallelization
  associative_scan.rs — Parallel prefix scan
  tnt.rs           — TNT hierarchical parallelization
  lattice_gla.rs   — Gated linear attention
  atlas_parallel.rs — Atlas memory-optimized parallel
  serving.rs       — Session/LatencyTracker/Checkpoint (feature: serving)
  distributed.rs   — CMS-aware multi-GPU sync (feature: distributed)
  edge.rs          — Edge deployment profiles (feature: edge)
  cuda_ffi.rs      — CUDA FFI bindings (feature: cuda)

core/kernels/                      (~1,250 lines, 8 kernel files)
  swa_forward.cu / swa_backward.cu
  delta_forward.cu / delta_backward.cu
  titans_forward.cu / titans_backward.cu
  hebbian_forward.cu / hebbian_backward.cu

core/tests/                        (~9,500 lines, 28 test files)

python/                            (~1,200 lines)
  nl_hecate/       — PyO3 bindings (all rules + compositions)
  tests/           — PyTorch baseline + binding tests (27 tests)
```

---

## PR History

| PR | Title | Stage |
|---|---|---|
| #1 | Phase 0: Enzyme spike test suite | S0 |
| #2 | Phase 0 spike complete: 57/57 pass, OUTCOME 1 GO | S0 |
| #3 | Track Zero-A Phase 1: Rust core with SWA forward/backward | S0 |
| #4 | Track Zero-A Phase 2: CUDA SWA kernel pair | S0 |
| #5 | Track Zero-A Phase 3: PyO3 Python bindings | S0 |
| #6 | Track Zero-A Phase 4: PyTorch regression baseline | S0 |
| #7 | Track Zero-B Phase 1: Delta Rule + MAG composition | S0 |
| #8 | Track Zero-B Phase 2: PyO3 bindings + PyTorch MAG baseline | S0 |
| #9 | CMS k=2 — multi-level memory scheduling | S0 |
| #10 | CMS k=4 — full frequency hierarchy | S0 |
| #11 | Titans LMM: GD+momentum memory rule | S1 |
| #12 | CMS output normalization (1/sqrt(k) for k>2) | S1 |
| #13 | Stability boundary test | S1 |
| #14 | Hebbian rule implementation | S1 |
| #15 | MONETA: MLP memory rule | S1 |
| #16 | YAAD: Huber decoupled retention | S1 |
| #17 | MEMORA: KL-softmax emergence | S1 |
| #18 | Lattice OSR: orthogonal state recurrence | S1 |
| #19 | Trellis: two-pass KV compression | S1 |
| #20 | Update PyO3 bindings for all MIRAS rules | S1 |
| #21 | MAC/MAL composition patterns | S1 |
| #22 | ContextStream implementation | S1 |
| #23 | 100K stability sweep expansion | S1 |
| #24 | Parallelization strategies (5/5) | S1 |
| #25 | CUDA kernel pairs (S2-M1) | S2 |
| #26 | CMS-aware multi-GPU gradient sync (S2-M2) | S2 |
| #27 | Serving non-stationary models (S2-M3) | S2 |
| #28 | Edge deployment for zero-dependency micro models (S2-M4) | S2 |
| #29 | Multi-arch CUDA dispatch + build matrix (S2-M1 Phase 5) | S2 |
| #30 | Update progress report and integration spike tests | S2 |
| #31 | Pluggable retention trait — MIRAS Knob #3 (S3-M1) | S3 |

---

## Metrics

| Metric | Value |
|---|---|
| Total tests | 921 (894 Rust + 27 Python) |
| Rust tests (verified) | 847 base + 47 feature-gated = 894 passed, 0 failed, 2 ignored |
| Python tests | 27 |
| PRs merged | 31 |
| Spec files | 48 |
| Lines of Rust (core/src) | ~25,300 |
| Lines of Rust (core/tests) | ~9,900 |
| Lines of CUDA (kernels) | ~1,250 |
| Lines of Python (bindings+tests) | ~1,200 |
| Memory rules | 8/8 |
| Composition patterns | 3/3 |
| Parallelization strategies | 5/5 |
| CMS levels validated | k=1, k=2, k=4 |
| MIRAS knobs exercised | All 4 (Structure, Bias, Retention, Algorithm) |
| Edge throughput (d=64) | ~34k tok/s |
| Integration spike | 17/17 pass, Outcome 1 (GO) |

---

## What's Next: Stage 3 (Extensions) — 1/5 complete

S3-M1 (Pluggable Retention) is complete. Retention is now a first-class composable knob — any rule can use any retention mechanism, unlocking the full 4-knob MIRAS design space.

Remaining Stage 3 milestones are independent:

- ~~**S3-M1: Pluggable Retention**~~ — COMPLETE (PR #31)
- **S3-M2: M3 Multi-Scale Optimizer** — Apply CMS to the optimizer itself (k momentum accumulators)
- **S3-M3: CMS Deployment Variants** — Nested, sequential, independent, hybrid CMS patterns
- **S3-M4: Atlas Omega Rule** — 9th MIRAS variant, enables Atlas Parallel strategy
- **S3-M5: Dynamic Frequency Scheduling** — Data-dependent level activation (learned frequency gates)
