# nl.Module Contract

**Version**: 0.4.0
**Repository**: NL_Hecate
**Language Target**: Rust + CUDA (core), Python + PyO3 (orchestration bindings)
**Differentiation**: Enzyme (LLVM-level AD on Rust) + hand-written kernel pairs (CUDA)
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

Layer 2: Rust (mathematics + control flow + Enzyme AD)
  - All memory update rules, composition patterns, scheduling
  - Enzyme provides automatic differentiation at LLVM IR level
  - Enzyme differentiates ONLY Rust code — never raw CUDA
  - Enzyme chains through CUDA kernel pairs via #[custom_vjp]
  - Trait system enforces valid compositions at compile time
  - Ownership model enforces state lifecycle (outer vs inner loop)
  - Reference implementations of ALL kernels live here (portable, correct)

Layer 1: CUDA (kernel pairs — forward + backward)
  - Each kernel ships as a (forward, backward) pair
  - Backward kernels ARE the analytical gradients from the papers
  - Enzyme does NOT look inside kernels — it chains between them
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

1. Rust reference (portable, Enzyme-compatible, correct-by-construction)
   - Used for: development, testing, correctness oracle, CPU fallback
   - Enzyme CAN differentiate through this directly

2. CUDA forward kernel (hardware-optimized)
   - Uses shared memory, tensor cores, warp-level primitives freely
   - Opaque to Enzyme — Enzyme never sees inside

3. CUDA backward kernel (hand-derived from paper equations)
   - The analytical gradients from the papers become code here
   - Also opaque to Enzyme — also hardware-optimized
   - Correctness verified against Rust reference backward

-- Enzyme sees:
   [Rust code] → [opaque kernel pair] → [Rust code] → [opaque kernel pair] → loss

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
  - Modified by backprop (Enzyme AD through outer loop)
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

The system uses two differentiation mechanisms that compose via the chain rule:

1. **Enzyme AD** — differentiates Rust code at the LLVM IR level
2. **Hand-written backward kernels** — provide gradients for CUDA operations

Enzyme NEVER differentiates through raw CUDA. This is a deliberate design choice
that matches every production ML framework (PyTorch, JAX, FlashAttention, cuDNN).

```
RULE: The differentiation barrier is COMPILER-ENFORCED, not convention-based.
      (Committee Finding 1: "Hope-based engineering" is not engineering.)

      Every implementation of MemoryUpdateRule MUST wrap its inner-loop
      kernel with the #[enzyme_opaque] attribute. This attribute:
        (a) Prevents Enzyme from tracing into the function at LLVM IR level
        (b) Forces the developer to provide an explicit backward via #[custom_vjp]
        (c) Is a MANDATORY trait bound — code that omits it does not compile

      Without this barrier, Enzyme may:
        - Segfault (best case: hard crash, obvious error)
        - Generate garbage gradients by tracing pointer operations (worst case:
          silent corruption, model appears to build but learns nothing)
        - Double-count by tracing the Rust wrapper AND the kernel backward

      The #[enzyme_opaque] + #[custom_vjp] pairing is the enforcement mechanism.
      It is not optional. It is not a convention. It is a compiler requirement.

RULE: The barrier must be TESTED, not just declared.
      Every kernel pair requires a barrier verification test:

      TEST: enzyme_barrier_holds(kernel_pair)
        1. Run forward pass through the kernel pair
        2. Run Enzyme backward — should produce ZERO autodiff contribution
           from the opaque kernel (only the #[custom_vjp] backward contributes)
        3. Assert: autodiff_gradient == 0 for the opaque region
        4. Assert: custom_vjp_gradient == analytical_gradient (from paper)
        5. Assert: total_gradient == custom_vjp_gradient (no double-counting)

      This test closes the loop: the compiler forbids the behavior,
      the test verifies the barrier is holding.

RULE: Inner-loop gradients are ANALYTICAL (hand-derived from paper equations).
      These analytical gradients serve TWO purposes:
        (a) They ARE the inner-loop update (gradient descent in the forward pass)
        (b) They BECOME the backward CUDA kernels (for outer-loop chain rule)

      The paper equations are not documentation — they are the backward pass.

RULE: Enzyme handles Rust-level composition and outer-loop gradient flow.
      When Enzyme encounters a CUDA kernel pair, it uses the provided
      backward kernel to continue the chain rule. It does not trace inside.

RULE: Differentiation is OPT-IN, not opt-out. (CS-40)
      Three annotation levels control what participates in AD:

      #[autodiff]      — Enzyme differentiates through this function directly.
                         Used for: Rust code in the outer-loop gradient path.
                         (gate computations, projections, loss, composition logic)

      #[custom_vjp]    — Opaque to Enzyme but provides its own backward.
                         Used for: CUDA kernel pairs. Enzyme chains through
                         these using the provided backward kernel.
                         The outer-loop gradient DOES flow through these —
                         via the hand-written backward, not via Enzyme tracing.

      #[no_autodiff]   — Completely severed from the gradient chain.
                         Used for: debug logging, metrics, visualization.
                         Nothing that affects the output should carry this.

CRITICAL DISTINCTION:
      Inner-loop operations are #[custom_vjp], NOT #[no_autodiff].
      They DO participate in the outer-loop gradient chain — through their
      hand-written backward kernels. Marking them #[no_autodiff] would sever
      the chain and outer-loop parameters (W_K, W_V, W_Q) would receive
      zero gradient. That is a bug, not a feature.
```

The gradient flow through the full system:

```
Forward pass:
  x → [W_K projection]  → k     (Rust, #[autodiff])
    → [gate computation] → gates (Rust, #[autodiff])
    → [inner loop kernel] → y    (CUDA, #[custom_vjp])
    → [loss computation]  → loss (Rust, #[autodiff])

Backward pass (outer-loop gradient for W_K):
  d(loss)/d(W_K) = d(loss)/d(y) * d(y)/d(k) * d(k)/d(W_K)
                   ^^^^^^^^^^^    ^^^^^^^^^    ^^^^^^^^^^^
                   Enzyme (Rust)  backward     Enzyme (Rust)
                                  kernel
                                  (CUDA)

  Enzyme computes d(loss)/d(y) and d(k)/d(W_K) — these are Rust code.
  The backward kernel computes d(y)/d(k) — this is the CUDA kernel pair.
  Enzyme chains them together via the chain rule.
  No part of this requires Enzyme to trace through CUDA.
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
-- SAME CODE in all phases. The ONLY difference: in Build phase, Enzyme's
-- AD graph is live downstream of the forward pass (computing outer-loop
-- gradients). In Test/Stream, Enzyme is inactive. The model itself never
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

Not all MIRAS knob combinations are valid. The Rust trait system enforces this at compile time.

```
trait MemoryStructure { }      // Knob 1: vector, matrix, MLP
trait AttentionalBias { }      // Knob 2: L2, dot-product, Huber, l_p
trait RetentionMechanism { }   // Knob 3: L2 decay, KL, elastic net, etc.
trait MemoryAlgorithm { }      // Knob 4: GD, GD+momentum, Newton, FTRL

trait MemoryUpdateRule:
  MemoryStructure + AttentionalBias + RetentionMechanism + MemoryAlgorithm
{
  fn write(&mut self, k: &Tensor, v: &Tensor, gates: &Gates, pulse: &Pulse);
  fn read(&self, q: &Tensor) -> Tensor;
  fn step(&mut self, x: &Tensor, pulse: &Pulse) -> (Tensor, ());
}

// Invalid combinations are compile errors:
// - KL retention requires softmax structure (probability simplex)
// - Newton-Schulz algorithm requires matrix structure (not vector)
// - Lattice OSR requires sphere normalization retention
// - Associative scan parallelization requires LINEAR recurrence
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

RULE: The composition pattern is orthogonal to the memory update rule.
      Any rule can plug into any pattern.
      But some patterns have additional requirements:
        - MAC uses full causal attention (not sliding window)
        - MAG/MAL use sliding window attention
        - MAG requires the memory output to be in [0,1] (sigmoid gate)
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

RULE: Every memory update rule specifies which parallelization
      strategies it supports. This is a trait bound.
```

### 9. Code Smell Enforcement

48 code smells (CS-01 through CS-48) define what code must NOT look like. Key categories:

```
Ontological smells (CS-01, CS-10, CS-11, CS-13, CS-37, CS-38):
  - No MemoryModule class, no train/eval, no TrainingLoop, no "training" word
  - Use "levels" not "layers" for frequency hierarchy
  - Use "build" not "train", "test" not "eval"

Structural smells (CS-18, CS-27, CS-28, CS-31, CS-32):
  - Forward pass IS the only external API
  - Optimizer must be frequency-aware
  - NeuralLearningModule is indivisible
  - Observe then advance (stateful counters mutate AFTER observers)

MIRAS smells (CS-33 through CS-36):
  - Don't force same attentional bias across models
  - Don't restrict memory to matrix-valued
  - Don't assume GD is the only algorithm
  - Don't restrict retention to L2 only

Infrastructure smells (CS-39 through CS-48):
  - Learnable decay must be clamped
  - Autograd is opt-in not opt-out
  - GPU utilization != throughput
  - Gradient checkpointing hurts NL
  - DDP inflates reported throughput
  - NL cannot fill high-end GPUs
  - torch.compile cannot trace NL inner loops
  - In-place modification destroys reproducibility
  - Shared retention parameters across CMS levels
```

### 10. Committee Findings (v0.3.0)

Four findings from external review, integrated into this contract:

```
FINDING 1: Differentiation Barrier Enforcement
  Problem:  The #[custom_vjp] mechanism was "hope-based engineering."
            No compiler enforcement prevented Enzyme from tracing into kernels.
  Fix:      #[enzyme_opaque] attribute is MANDATORY on all inner-loop kernels.
            EnzymeOpaque marker trait is a required bound on MemoryUpdateRule.
            Barrier verification tests assert zero autodiff contribution.
  Updated:  Section 4 (Differentiation), differentiation/00_enzyme_integration.md,
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
Enzyme pins the LLVM version. The LLVM version pins the Rust nightly.
The Rust nightly constrains which language features and libraries are available.
This is an ARCHITECTURAL constraint, not a build detail.

CONSTRAINT CHAIN:
  Enzyme requires LLVM version X
  → Rust nightly Y targets LLVM X (not all nightlies do)
  → Language features available in nightly Y constrain the trait system design
  → CUDA toolkit Z must link with the same LLVM version
  → PyO3 version must support nightly Y

RULE: The Enzyme→LLVM→Rust compatibility matrix MUST be resolved in Phase 0
      (the Enzyme spike), BEFORE any production code is written.
      If the trait system design (marker traits, associated types, builder
      pattern with generic bounds) requires features unavailable in the
      Enzyme-compatible nightly, the trait system adapts, not the toolchain.

RULE: Phase 0 produces a pinned toolchain document:
      - Exact Rust nightly version
      - Exact LLVM version (Enzyme requirement)
      - Exact CUDA toolkit version
      - List of Rust features AVAILABLE under this toolchain
      - List of Rust features UNAVAILABLE (and workarounds if needed)
      - Verified: trait system patterns compile under pinned toolchain

See: infrastructure/track_zero/00_track_zero.md for Phase 0 spike definition.
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

## Directory Structure as Graph

```
specs/                              <- root node
  contract.md                       <- THIS FILE: top-level specification
  algorithms/                       <- algorithmic components (paper math)
    memory_update_rules/            <- MIRAS 4-knob framework
      00_interface.md               <- MemoryUpdateRule trait definition
      titans_family/                <- Titans-derived rules
        01_titans_lmm.md            <- Full LMM with momentum
        02_delta_rule.md            <- eta=0 special case
        03_hebbian_rule.md          <- No gradient, direct association
      miras_family/                 <- MIRAS design space exploration
        04_moneta.md                <- 2-layer MLP + l_p + L_q
        05_yaad.md                  <- Huber loss + decoupled retention
        06_memora.md                <- KL divergence + softmax
      compression_family/           <- Memory compression variants
        07_lattice_osr.md           <- Orthogonal State Recurrence
        08_trellis_twopass.md       <- Two-pass KV compression
    composition_patterns/           <- MAC/MAG/MAL
      00_interface.md
      01_mac.md
      02_mag.md
      03_mal.md
    retention_mechanisms/           <- MIRAS Knob #3
      00_interface.md
      01_l2_weight_decay.md
      02_kl_divergence.md
      03_elastic_net.md
      04_f_divergence.md
      05_sphere_normalization.md
    parallelization/                <- Chunk-wise strategies
      00_interface.md
      01_chunkwise_gd.md
      02_associative_scan.md
      03_tnt_hierarchical.md
      04_lattice_gla.md
      05_atlas_parallel.md
    optimization_machinery/         <- Momentum hierarchy
      01_momentum.md
      02_m3.md
    frequency_scheduling/           <- CMS gating
      01_frequency_scheduler.md
      02_cms_variants.md
  infrastructure/                   <- PyTorch replacement tooling
    differentiation/                <- Enzyme AD integration
    state_lifecycle/                <- Outer/inner/context state management
    scheduling/                     <- Conductor + Pulse
    attention/                      <- SWA + full causal (non-NL component)
    distribution/                   <- Multi-GPU without DDP assumptions
    compilation/                    <- Self-modifying graph compilation
    serving/                        <- Serving non-stationary models
    context_stream/                 <- DataLoader replacement
    precision/                      <- Numerical precision strategy (v0.4.0)
    track_zero/                     <- First implementation milestone (v0.4.0)
  constraints/                      <- Enforcement rules
    code_smells/                    <- CS-01 through CS-47
    trait_system/                   <- Valid composition pairings
```

Each file = graph node. Each directory = parent-child edge.
Contract headers on every file enable graph queries across the spec.
