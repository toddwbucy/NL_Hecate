# nl.Module Contract

CONTRACT
Purpose: Top-level architectural specification for NL_Hecate — defines the three-tier structure, differentiation strategy, state lifetimes, CMS scheduling, and composition patterns.
Expects: Familiarity with the NL paper corpus (Titans, MIRAS, HOPE, Lattice, Atlas, TNT, Trellis).
Guarantees: Every component traces to a paper equation; every constraint traces to a code smell or axiom.
Cost: N/A (specification document, no runtime cost).
Trade-off: Breadth over depth — individual specs in specs/algorithms/ and specs/infrastructure/ provide detailed contracts per component.
Position: Root of the spec tree — all other specs refine sections of this contract.
Source: Mirrokni/Behrouz research group (Google Research): arxiv 2501.00663, 2504.13173, 2512.24695, 2504.05646, 2505.23735, 2511.07343, 2512.23852.

**Version**: 0.4.1
**Repository**: NL_Hecate
**Language Target**: Rust + CUDA (core), Python + PyO3 (orchestration bindings)
**Differentiation**: Wengert tape AD (Rust) + hand-written kernel pairs (CUDA)
**Source**: The complete output of the Mirrokni/Behrouz research group at Google Research

## What This Is

NL_Hecate is a specification-first implementation of the Nested Learning research program. Every component traces to a paper equation. Every constraint traces to an axiom or code smell. No concept exists here that doesn't exist in the papers.

The name: Hecate — goddess of crossroads, thresholds, and torchbearer. She stands at the boundary between conventional ML frameworks (static models, train/eval split, external optimizers) and what the NL papers describe (self-modifying models, no mode distinction, optimizers AS memory).

## The Three Layers

```
Layer 3: Python (orchestration)
  - PyO3 bindings to Layer 2
  - User-facing API: familiar Python module interface
  - No math here — just orchestration, configuration, data feeding
  - "So easy, even a developer can extend it"

Layer 2: Rust (mathematics + control flow + Wengert tape AD)
  - All 9 memory update rules, 3 composition patterns, CMS scheduling
  - Wengert tape provides automatic differentiation via operation recording
  - AD operates ONLY on Rust code — never raw CUDA
  - Tape chains through CUDA kernel pairs via OpaqueVjp trait
  - Trait system + marker traits enforce valid compositions at compile time
  - Ownership model enforces state lifecycle (outer vs inner loop)
  - Reference implementations of ALL kernels live here (portable, correct)
  - Conv1D key/query preprocessing, centralized momentum module

Layer 1: CUDA (kernel pairs — forward + backward)
  - Each kernel ships as a (forward, backward) pair
  - Backward kernels ARE the analytical gradients from the papers
  - AD does NOT look inside kernels — it chains between them
  - Hardware-specific: optimized per GPU architecture
  - Dispatched at runtime based on detected hardware
  - Optional: Rust reference implementation is the fallback
```

### The Kernel-Pair Pattern

No AD system in production differentiates through raw CUDA kernels.
Not PyTorch (torch.autograd.Function). Not JAX (custom_vjp). Not FlashAttention.
Not cuDNN. They all use the same pattern: paired forward + backward kernels,
composed by the AD system via the chain rule.

NL_Hecate follows this established pattern:

```
-- Each hot operation has THREE implementations:

1. Rust reference (portable, AD-compatible, correct-by-construction)
   - Used for: development, testing, correctness oracle, CPU fallback
   - The tape CAN differentiate through this directly

2. CUDA forward kernel (hardware-optimized)
   - Uses shared memory, tensor cores, warp-level primitives freely
   - Opaque to AD — compiled to SASS by nvcc

3. CUDA backward kernel (hand-derived from paper equations)
   - The analytical gradients from the papers become code here
   - Also opaque to AD — also hardware-optimized
   - Correctness verified against Rust reference backward

-- The tape sees:
   [Rust code] → [opaque VJP block] → [Rust code] → [opaque VJP block] → loss

   It differentiates the Rust parts directly.
   It chains through kernel pairs using their provided backward kernels.
   It NEVER needs to trace into CUDA.
```

This is how FlashAttention works — Tri Dao wrote both forward and backward
CUDA kernels by hand. PyTorch's autograd chains through them as black boxes.

### Hardware Dispatch

CUDA kernels are not portable across GPU architectures. The dispatch layer
handles this at runtime:

```
TRAIT: KernelDispatch
  fn select_backend(device: &Device) -> Backend

ENUM: Backend
  RustReference    -- portable, correct, slow (any hardware including CPU)
  CudaSM86         -- Ampere (A6000, A100, A5000)
  CudaSM89         -- Ada Lovelace (RTX 4090, L40)
  CudaSM90         -- Hopper (H100, H200)
  CudaPTX          -- JIT-compiled PTX (any NVIDIA GPU, first-launch cost)

-- Runtime flow:
fn chunkwise_memory_update(state, grads, chunk_size, device) {
    match select_backend(device) {
        RustReference => chunkwise_update_rust(state, grads, chunk_size),
        CudaSM86      => chunkwise_update_sm86(state, grads, chunk_size),
        CudaPTX       => chunkwise_update_ptx(state, grads, chunk_size),
        _             => chunkwise_update_rust(state, grads, chunk_size),  // fallback
    }
}
```

Portability strategy:

```
Phase 1 (development): Rust reference only. No CUDA. Correct, testable, portable.
Phase 2 (optimization): Add CUDA kernel pairs for target hardware (A6000/SM86).
         Rust reference becomes the correctness oracle for CUDA kernels.
Phase 3 (portability): Add kernel pairs for other architectures as needed.
         Dispatch layer selects best available. Rust reference is always the fallback.
```

CUDA portability within NVIDIA:

```
Compile with PTX (forward-compatible):
  -gencode arch=compute_80,code=compute_80
  → Generates PTX (NVIDIA intermediate representation)
  → Driver JIT-compiles to whatever card is installed
  → Runs on ANY SM80+ GPU (A100, A6000, RTX 3090, H100, ...)
  → First-launch cost (JIT), then cached

Compile with SASS (architecture-specific):
  -gencode arch=compute_86,code=sm_86
  → Binary for A6000 specifically
  → Fastest, no JIT cost
  → Only runs on SM86 GPUs

Ship both: SASS for known targets, PTX as fallback.
```

## The Contract

Every component in this system satisfies the following contract:

### 1. Paper Traceability

Every function traces to a paper equation or definition. No invented functions.

```
RULE: For every public function F in NL_Hecate:
  EXISTS equation E in {HOPE, Titans, MIRAS, Atlas, TNT, Lattice, Trellis}
  SUCH THAT F implements E
  AND the mapping is documented in F's contract header
```

### 2. Axiom Compliance

Every component complies with the NL IS/IS NOT containers. No exceptions.

```
IS Container (what every component must be):
  1. Part of a new learning paradigm (not a patched PyTorch module)
  2. Nested, multi-level, parallel optimization
  3. Each level with its own context flow
  4. Compressing its own context flow
  5. In-context learning naturally emerges
  6. Optimizers ARE associative memory modules
  7. Self-modifying learning module
  8. Continuum memory system

IS NOT Container (what no component may be):
  1. NOT single-level optimization
  2. NOT shared/global context flow
  3. NOT static/fixed update rules
  4. NOT discrete long/short-term memory
  5. NOT optimizers as just optimizers
```

### 3. State Lifecycle

Every piece of state has exactly one of three lifetimes:

```
outer_loop_param:
  - Persists across the entire build process
  - Modified by backprop (tape AD through outer loop)
  - Serialized in checkpoints
  - Examples: W_K, W_V, W_Q, gate parameters, persistent memory tokens
  - Rust: owned by the model struct, &mut access

inner_loop_state:
  - Created fresh each forward pass (or each chunk)
  - Modified by the inner optimization loop (GD, momentum, etc.)
  - NOT serialized (it's a "thought", not a "memory")
  - Examples: M (memory matrix during inner loop), S (momentum accumulator)
  - Rust: scoped lifetime, dropped after forward pass

context_memory:
  - Persists across forward passes but NOT across builds
  - The running memory state that accumulates across a context stream
  - Explicitly transferred between invocations
  - Examples: memory state at chunk boundaries, hierarchical memory (TNT global)
  - Rust: explicit ownership transfer via move semantics
```

### 4. Differentiation

> **Updated 2026-02-18**: Originally designed around Enzyme (LLVM-level AD).
> Now implemented via Wengert tape (`core/src/tape.rs`). The original Enzyme
> spec is preserved at `differentiation/00_enzyme_integration.md` (ARCHIVED).
> The current spec is `differentiation/01_wengert_tape.md`.

The system uses two differentiation mechanisms that compose via the chain rule:

1. **Wengert tape** — records Rust operations during forward, replays in reverse for gradients
2. **Hand-written backward kernels** — provide gradients for CUDA and memory rule operations

AD never differentiates through raw CUDA. This is a deliberate design choice
that matches every production ML framework (PyTorch, JAX, FlashAttention, cuDNN).

```text
RULE: The differentiation barrier is TRAIT-ENFORCED, not convention-based.

      Every implementation of MemoryUpdateRule MUST implement the OpaqueVjp trait.
      This trait:
        (a) Prevents the tape from tracing into the operation's internals
        (b) Forces the developer to provide an explicit backward adapter
        (c) Is a MANDATORY supertrait bound — code that omits it does not compile

RULE: Inner-loop gradients are ANALYTICAL (hand-derived from paper equations).
      These analytical gradients serve TWO purposes:
        (a) They ARE the inner-loop update (gradient descent in the forward pass)
        (b) They BECOME the backward functions (for outer-loop chain rule)

      The paper equations are not documentation — they are the backward pass.

RULE: The tape handles Rust-level composition and outer-loop gradient flow.
      When the tape encounters an opaque block (memory rule or CUDA kernel),
      it uses the registered backward function to continue the chain rule.

RULE: Differentiation is OPT-IN, not opt-out. (CS-40)
      Two participation levels:

      Tape-traced      — The tape records operations for this code path.
                         Used for: Rust code in the outer-loop gradient path.
                         (gate computations, projections, loss, composition logic)
                         Activated by with_tape().

      OpaqueVjp        — Opaque to the tape, provides its own backward adapter.
                         Used for: memory rules and CUDA kernel pairs.
                         The outer-loop gradient DOES flow through these —
                         via the registered backward function.

CRITICAL DISTINCTION:
      Inner-loop operations are OpaqueVjp, NOT severed from the gradient chain.
      They DO participate in the outer-loop gradient chain — through their
      registered backward adapters. Omitting OpaqueVjp would sever the chain
      and outer-loop parameters (W_K, W_V, W_Q) would receive zero gradient.
```

The gradient flow through the full system:

```text
Forward pass:
  x → [W_K projection]  → k     (Rust, tape-traced)
    → [gate computation] → gates (Rust, tape-traced)
    → [inner loop kernel] → y    (opaque VJP block)
    → [loss computation]  → loss (Rust, tape-traced)

Backward pass (outer-loop gradient for W_K):
  d(loss)/d(W_K) = d(loss)/d(y) * d(y)/d(k) * d(k)/d(W_K)
                   ^^^^^^^^^^^    ^^^^^^^^^    ^^^^^^^^^^^
                   tape (Rust)    backward     tape (Rust)
                                  adapter
                                  (opaque)

  The tape computes d(loss)/d(y) and d(k)/d(W_K) — these are Rust code.
  The backward adapter computes d(y)/d(k) — registered as opaque VJP.
  The tape chains them together via the chain rule.
```

### 5. Pulse (Timing Context)

Every operation receives a Pulse — a timing context that tells the component WHERE it is in the multi-scale schedule.

```
struct Pulse {
  global_step: u64,          // monotonic step counter
  active_levels: Vec<bool>,  // which CMS frequency levels are active NOW
  chunk_boundaries: Vec<u64>,// current chunk boundary per level
  phase: Phase,              // Build | Test | Stream (replaces train/eval)
}

-- Note: there is no "train" or "eval" phase. The forward pass runs the
-- SAME CODE in all phases. The ONLY difference: in Build phase, the tape's
-- AD graph is live downstream of the forward pass (computing outer-loop
-- gradients). In Test/Stream, the tape is inactive. The model itself never
-- checks what phase it's in — the Conductor owns that decision.
-- This is "same forward code, different computational context."

RULE: Every STEP, WRITE, READ, and FORWARD function takes a &Pulse.
      Components use it to determine:
        - Whether they should update (frequency scheduling)
        - What chunk boundary to use (parallelization)
        - Whether to accumulate error (frozen levels)

      The Pulse is READ-ONLY. Components never modify it.
      A central Conductor creates and advances the Pulse.
```

### 6. Composition Safety (Trait System)

Not all MIRAS knob combinations are valid. The Rust trait system enforces this at compile time
via marker traits (`core/src/composition_safety.rs`) and the CompositionPattern trait
(`core/src/composition_pattern.rs`).

```rust
// Marker traits (composition_safety.rs) — enforce mathematical constraints
trait ProbabilitySimplex { }      // KL retention requires this
trait UnitSphere { }              // Lattice OSR requires this
trait LinearRecurrence { }        // Associative scan requires this
trait StateIndependentMomentum { }// Atlas/Titans momentum constraint

// The MemoryRule trait (delta_rule.rs) — all 9 rules implement this
trait MemoryRule: OpaqueVjp {
  type Cache;
  fn level(&self) -> usize;
  fn supported_parallelization(&self) -> &'static [&'static str];
  fn step(level_params, embedded, seq_len, d, initial_m) -> (Vec<f32>, Cache);
  fn step_backward(level_params, cache, d_y, embedded) -> (grads, d_embedded);
}

// CompositionPattern trait (composition_pattern.rs)
trait CompositionPattern {
  fn prepend_persistent(cfg, level_params, embedded) -> Vec<f32>;
}

// Invalid combinations are compile errors:
// - KL retention requires softmax structure (ProbabilitySimplex)
// - Newton-Schulz algorithm requires matrix structure (not vector)
// - Lattice OSR requires sphere normalization (UnitSphere)
// - Associative scan parallelization requires LinearRecurrence
```

### 7. Composition Patterns

Three patterns for combining memory with attention. Each is a trait implementation.

```
trait CompositionPattern {
  fn forward(&mut self, x: &Tensor, memory: &mut dyn MemoryUpdateRule,
             attention: &dyn Attention, pulse: &Pulse) -> Tensor;
}

MAC: Memory reads -> concat with input -> attention processes -> memory writes
MAG: Memory and attention run parallel -> memory gates attention output
MAL: Memory preprocesses -> attention processes memory output
HOPE: End-to-end variant from HOPE §6 — persistent memory tokens + Conv1D

RULE: The composition pattern is orthogonal to the memory update rule.
      Any rule can plug into any pattern.
      But some patterns have additional requirements:
        - MAC uses full causal attention (not sliding window)
        - MAG/MAL use sliding window attention
        - MAG requires the memory output to be in [0,1] (sigmoid gate)
        - HOPE adds persistent memory tokens (n_persistent) and Conv1D
          preprocessing on keys/queries before the memory module
```

### 8. Parallelization

Sequential inner loops become chunk-wise parallel via three strategies:

```
Strategy 1: Freeze State to Chunk Boundary
  - Used by: Titans, Atlas, Lattice, Trellis
  - Approximate: gradients computed w.r.t. chunk-START state
  - Error bounded by chunk size

Strategy 2: Associative Scan (Blelloch)
  - Used by: momentum (linear recurrence)
  - Exact: O(log C) parallel steps
  - Only works for LINEAR recurrences

Strategy 3: Hierarchical Memory (TNT)
  - Used by: TNT
  - Global memory at coarse grain + independent local memories
  - Enables large effective chunk sizes without approximation error

Strategy 4: Lattice GLA
  - Used by: Lattice OSR
  - Gated Linear Attention with chunk_size=seq_len for C=1 exactness

Strategy 5: Atlas Parallel
  - Used by: Atlas Omega
  - State-independent omega precomputed in batch, M/S recurrence sequential

RULE: Every memory update rule specifies which parallelization
      strategies it supports. This is a trait bound.
```

### 9. Code Smell Enforcement

48 code smells (CS-01 through CS-48) define what code must NOT look like. Key categories
(full index: `specs/constraints/code_smells/00_index.md`):

```text
Ontological smells (CS-01, CS-04–09, CS-10, CS-11, CS-13, CS-19–21, CS-37, CS-38):
  - No MemoryModule class, no train/eval, no TrainingLoop, no "training" word
  - Use "levels" not "layers" for frequency hierarchy
  - Use "build" not "train", "test" not "eval"

Structural smells (CS-12, CS-18, CS-22, CS-23, CS-31):
  - Forward pass IS the only external API
  - Optimizer must be frequency-aware
  - NeuralLearningModule is indivisible

MIRAS smells (CS-33 through CS-36, CS-48):
  - Don't force same attentional bias across models
  - Don't restrict memory to matrix-valued
  - Don't assume GD is the only algorithm
  - Don't restrict retention to L2 only

Infrastructure smells (CS-32, CS-39–47):
  - Observe then advance (stateful counters mutate AFTER observers)
  - Learnable decay must be clamped
  - Autograd is opt-in not opt-out (CS-40)
  - GPU utilization != throughput
  - Gradient checkpointing hurts NL
  - DDP inflates reported throughput
  - In-place modification destroys reproducibility
  - Shared retention parameters across CMS levels
```

### 10. Committee Findings (v0.3.0)

Four findings from external review, integrated into this contract:

```
FINDING 1: Differentiation Barrier Enforcement
  Problem:  The #[custom_vjp] mechanism was "hope-based engineering."
            No compiler enforcement prevented AD from tracing into kernels.
  Fix:      OpaqueVjp trait is a required bound on MemoryUpdateRule.
            All inner-loop kernels (CUDA + 9 memory rules) register as opaque
            blocks on the Wengert tape via opaque_key() registry lookup.
            Barrier verification: Class 3 tests (tape vs hand-written backward)
            confirm identical gradients for all 9 rules × k=1,2,4.
  Updated:  Section 4 (Differentiation), differentiation/01_wengert_tape.md,
            memory_update_rules/00_interface.md

FINDING 2: Pulse Reconciliation Protocol (DEFERRED TO PHASE 2)
  Problem:  The conductor assumed perfect lockstep synchronization across GPUs.
            CMS asymmetric workloads COULD cause pulse drift → deadlocks.
  Status:   Phase 1 uses synchronous barrier sync (allreduce). Pulse skew
            cannot occur. Protocol deferred until asynchronous advancement
            is introduced. See scheduling/00_conductor.md for full protocol.
  Updated:  scheduling/00_conductor.md, distribution/00_multi_gpu.md

FINDING 3: Compile-Time Composition Safety
  Problem:  Interface claimed traits are "orthogonal" but some pairings are
            mathematically invalid. The constraint matrix was in a markdown
            file, not in the compiler. Invalid models could compile and run.
  Fix:      Marker traits (ProbabilitySimplex, UnitSphere, LinearRecurrence).
            Associated types on traits enforce compatibility at compile time.
            Builder pattern prevents direct instantiation of invalid combinations.
  Updated:  memory_update_rules/00_interface.md,
            constraints/trait_system/00_valid_compositions.md

FINDING 4: Stream-State Coupling
  Problem:  Model checkpoint and data stream position were treated as separate.
            On resume, a mismatched stream/model pair silently corrupts learning.
  Fix:      StreamCursor struct (dataset_index, pulse_id, rng_state) serialized
            INSIDE the model checkpoint — not alongside, not separately.
            Conductor OWNS the ContextStream (atomic checkpoint of both).
            Hash verification on restore rejects pulse_id mismatches.
  Updated:  context_stream/00_context_stream.md, serving/00_serving.md,
            state_lifecycle/00_state_ownership.md
```

### 11. Toolchain Constraints (v0.4.0)

```
> Updated 2026-02-19: Enzyme was archived (Acheron/enzyme-archive/) after the
> Wengert tape superseded it. The LLVM-pinning constraint chain below is HISTORICAL.
> Standard Rust stable/nightly compiles the full codebase. CUDA Toolkit 12.8+ is
> required only for the cuda feature gate.

HISTORICAL (Enzyme era):
  Enzyme pinned LLVM → pinned Rust nightly → constrained language features.
  This was an architectural constraint, not a build detail.
  Eliminated Feb 2026 by switching to Wengert tape AD (core/src/tape.rs).

CURRENT:
  - Rust: any stable or nightly (no custom toolchain)
  - CUDA Toolkit 12.8+: only for #[cfg(feature = "cuda")] kernel compilation
  - PyO3: compatible with standard Rust releases
  - No LLVM version coupling — tape is pure Rust
```

### 12. Implementation Sequence (v0.4.0)

```
Defined by committee review cycle. Non-negotiable ordering.

Phase 0: Enzyme Spike (2 weeks, time-boxed)
  → Can Enzyme differentiate through our Rust trait patterns?
  → Three outcomes: works / works with simplifications / doesn't work
  → Deliverable: spike notebook + pinned toolchain (or go/no-go on Enzyme)

Phase 1: Track Zero (2-4 weeks)
  Zero-A: Pure SWA attention, no memory. Match PyTorch Transformer baseline.
  Zero-B: Delta Rule + MAG + k=1. Match PyTorch reference implementation.
  → Deliverable: working pipeline, regression anchors passing

Phase 2: CMS Introduction (k=2)
  → Error buffer health monitoring, retention interference analysis
  → TRANSITION CRITERIA: k=2 must beat k=1 on at least one metric
  → If k=2 doesn't beat k=1, debug CMS before adding more levels

Phase 3: Full Design Space (k=4, multiple rules)
  → Combinatorial validation at 100/1K/10K/100K step horizons
  → Falsification: >20% degenerate combinations = orthogonality is wrong

See: infrastructure/track_zero/00_track_zero.md for full specification.
```

### 13. What This Contract Does NOT Specify

- **Hyperparameters**: Chunk sizes, frequency multipliers, learning rates — application-dependent
- **Model architecture**: How many blocks, what composition per block — that's model design
- **Data pipeline**: How data arrives is outside this contract (see context_stream infrastructure)
- **Evaluation metrics**: What to measure is task-dependent
- **Non-NVIDIA hardware**: AMD (ROCm), Apple (Metal), TPU — future bridges to cross if needed. The Rust reference implementations are hardware-agnostic. CUDA kernels are NVIDIA-specific optimizations.
- **Specific GPU architecture targets**: The dispatch layer handles this. Which architectures get optimized kernels is a resourcing decision, not an architectural one. The Rust reference is always the fallback.

### 14. Rust Module Inventory (v0.4.1)

> Added during partial-specs sweep to document all core/src/ modules.

```text
core/src/ module map:

Memory rules (9 total, each implements MemoryRule + OpaqueVjp):
  delta_rule.rs       — Delta Rule (Titans eta=0). Also defines MemoryRule trait.
  titans_lmm.rs       — Full Titans LMM with momentum
  hebbian_rule.rs     — Direct association (no gradient)
  moneta.rs           — 2-layer MLP + l_p attentional bias + L_q retention
  yaad.rs             — Huber loss + decoupled retention
  memora.rs           — KL divergence + softmax memory
  lattice_osr.rs      — Orthogonal State Recurrence (sphere normalization)
  trellis.rs          — Two-pass KV compression
  atlas_omega.rs      — 3-gate (alpha/theta/eta) with omega outer products

Composition & preprocessing:
  mag.rs              — MAG composition (memory gates attention)
  mal.rs              — MAL composition (memory preprocesses for attention)
  mac.rs              — MAC composition (memory-attention-memory)
  composition_pattern.rs — CompositionPattern trait + persistent memory tokens
  composition_safety.rs  — Marker traits (ProbabilitySimplex, UnitSphere, etc.)
  conv1d.rs           — Depthwise causal Conv1D with SiLU for k/q preprocessing
  momentum.rs         — Centralized momentum module (EMA, Delta, Deep)

Attention & parallelization:
  swa.rs              — Sliding window attention (reference implementation)
  dispatch.rs         — Backend dispatch (Rust reference / CUDA)
  atlas_parallel.rs   — Atlas Omega parallel forward (batch omega, sequential M/S)

CMS & scheduling:
  model.rs            — MAGConfig, MemoryLevelParams, enums (MemoryRuleKind,
                         AttentionalBias, MomentumKind, LatticeVariant, etc.)
  cms_variants.rs     — CMS variant implementations
  dynamic_freq.rs     — Learned frequency scheduling

Differentiation:
  tape.rs             — Wengert tape AD with opaque VJP blocks
  opaque_adapters.rs  — Backward adapters for all 9 rules on the tape
  traced_forward.rs   — Full CMS traced forward (active + frozen levels)
  gradient.rs         — End-to-end gradient computation (tape.backward)

Infrastructure:
  tensor.rs           — f32 tensor ops (matmul, sigmoid, softplus, etc.)
  retention.rs        — Retention mechanisms (L2 decay, KL, elastic net, etc.)
  adamw.rs            — Outer-loop AdamW optimizer
  bf16.rs             — bf16 utilities
  context_stream.rs   — ContextStream (DataLoader replacement)

Feature-gated (CUDA):
  gpu_buf.rs          — RAII device memory (DevBuf)
  gpu_forward.rs      — CUDA forward kernels
  gpu_backward.rs     — CUDA backward kernels
  gpu_optimizer.rs    — GPU-side optimizer
  gpu_params.rs       — GPU parameter management
  cuda_ffi.rs         — FFI bindings to nvcc-compiled kernels

Feature-gated (serving/distribution/edge):
  serving.rs          — Session, LatencyTracker, checkpoint/restore
  distributed.rs      — CMS-aware multi-GPU gradient sync
  edge.rs             — Edge deployment (d=64 target)
```

## Directory Structure as Graph

```text
specs/                              <- root node
  contract.md                       <- THIS FILE: top-level specification
  algorithms/                       <- algorithmic components (paper math)
    memory_update_rules/            <- MIRAS 4-knob framework (9 rules)
      00_interface.md               <- MemoryRule trait definition
      titans_family/                <- Titans-derived rules
        01_titans_lmm.md            <- Full LMM with momentum
        02_delta_rule.md            <- eta=0 special case
        03_hebbian_rule.md          <- No gradient, direct association
        04_atlas_omega.md           <- 3-gate (alpha/theta/eta) with omega (v0.4.1)
      miras_family/                 <- MIRAS design space exploration
        04_moneta.md                <- 2-layer MLP + l_p + L_q
        05_yaad.md                  <- Huber loss + decoupled retention
        06_memora.md                <- KL divergence + softmax
      compression_family/           <- Memory compression variants
        07_lattice_osr.md           <- Orthogonal State Recurrence
        08_trellis_twopass.md       <- Two-pass KV compression
    attentional_biases/             <- MIRAS Knob #2 extensions (v0.4.1)
      01_l1_sign.md                 <- Smooth tanh Sign approximation
      02_kl_objective.md            <- KL divergence objective
      03_lp_dispatch.md             <- Generalized l_p dispatch
    composition_patterns/           <- MAC/MAG/MAL/HOPE
      00_interface.md
      01_mac.md
      02_mag.md
      03_mal.md
      04_hope.md                    <- HOPE §6 end-to-end variant (v0.4.1)
    retention_mechanisms/           <- MIRAS Knob #3
      00_interface.md
      01_l2_weight_decay.md
      02_kl_divergence.md
      03_elastic_net.md
      04_f_divergence.md
      05_sphere_normalization.md
      06_bregman.md                 <- Bregman divergence (v0.4.1)
      07_lq_norm.md                 <- L_q norm for MONETA (v0.4.1)
      08_sigmoid_bounded.md         <- Sigmoid-bounded retention (v0.4.1)
    parallelization/                <- Chunk-wise strategies
      00_interface.md
      01_chunkwise_gd.md
      02_associative_scan.md
      03_tnt_hierarchical.md
      04_lattice_gla.md
      05_atlas_parallel.md
    optimization_machinery/         <- Inner/outer loop optimizers
      01_momentum.md                <- EMA + centralized momentum module
      02_m3.md                      <- M3 optimizer
      03_dgd.md                     <- Delta Gradient Descent (v0.4.1)
      04_dmgd.md                    <- Deep Momentum GD (v0.4.1)
      05_ftrl.md                    <- Follow-The-Regularized-Leader (v0.4.1)
      06_implicit_gd.md             <- Implicit gradient descent (v0.4.1)
      07_newton_schulz_inner.md     <- Inner-loop Newton-Schulz (v0.4.1)
      08_adamw_outer.md             <- Outer-loop AdamW (v0.4.1)
      09_adamuon.md                 <- AdamUON optimizer (v0.4.1)
    self_referential/               <- Self-referential mechanisms (v0.4.1)
      00_interface.md
      01_self_generated_values.md
      02_feature_maps.md
      03_chunkwise_self_ref.md
    frequency_scheduling/           <- CMS gating
      01_frequency_scheduler.md
      02_cms_variants.md
  infrastructure/                   <- PyTorch replacement tooling
    differentiation/                <- AD integration (Wengert tape + kernel pairs)
    state_lifecycle/                <- Outer/inner/context state management
    scheduling/                     <- Conductor + Pulse
    attention/                      <- SWA + Conv1D preprocessing
      00_attention.md               <- Sliding window attention
      02_short_conv.md              <- Causal Conv1D on k/q (v0.4.1)
    distribution/                   <- Multi-GPU without DDP assumptions
    compilation/                    <- Self-modifying graph compilation
    serving/                        <- Serving non-stationary models
    context_stream/                 <- DataLoader replacement
    precision/                      <- Numerical precision strategy (v0.4.0)
    track_zero/                     <- First implementation milestone (v0.4.0)
  constraints/                      <- Enforcement rules
    code_smells/                    <- CS-01 through CS-48
    trait_system/                   <- Valid composition pairings
```

Each file = graph node. Each directory = parent-child edge.
Contract headers on every file enable graph queries across the spec.
