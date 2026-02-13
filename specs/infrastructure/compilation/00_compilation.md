# Compilation (Self-Modifying Graph Problem)

```
CONTRACT
  Purpose:    CS-45: "torch.compile cannot trace NL inner loops."
              NL models self-modify during the forward pass. The inner loop
              writes to memory, which changes the model's behavior for the
              next token. No static computation graph captures this.
              This spec defines what can and cannot be compiled, and how
              NL_Hecate handles the non-compilable parts.
  Expects:    A model with inner-loop self-modification.
              CUDA kernel pairs (forward + backward).
              Rust code compiled by rustc + Enzyme.
  Guarantees: Static parts (projections, gates, attention) CAN be compiled/optimized.
              Dynamic parts (inner loop state mutation) are handled by
              pre-compiled kernels, not JIT compilation.
              No graph tracing through the inner loop.
              No performance cliff from graph breaks.
  Cost:       No JIT overhead for the inner loop (pre-compiled kernels).
              Static parts benefit from standard compiler optimizations.
              The compilation boundary is explicit, not discovered at runtime.
  Trade-off:  We lose whole-model JIT optimization (like torch.compile).
              We gain predictability — no graph breaks, no recompilation,
              no "this model is 3x slower because the compiler gave up."
              The kernel-pair pattern means the hot paths are hand-optimized
              anyway — a JIT compiler wouldn't beat them.
  Position:   specs/infrastructure/compilation/00_compilation.md
              Addresses: CS-45, nl_toolchain tool-13
  Source:     CS-45 (torch.compile cannot trace); nl_toolchain tool-13
              (compilation for self-modifying graphs)
```

## Why torch.compile Fails on NL

```
torch.compile traces the computation graph by running the model once.
It records operations and generates optimized CUDA code.

PROBLEM: NL's inner loop MUTATES state during the forward pass.
  -- At token t, memory M_t depends on M_{t-1} (which was just modified).
  -- torch.compile cannot capture this: it sees a different graph
  -- at every token (because the memory state is different).
  -- Result: graph break at every inner-loop step → no compilation benefit.

PROBLEM: NL's forward pass is data-dependent.
  -- CMS frequency scheduling: which levels are active depends on global_step.
  -- Conditional branches (is_active()) break graph tracing.
  -- torch.compile's dynamic shapes don't help — it's dynamic CONTROL FLOW.

PROBLEM: NL's memory update rules have unbounded loop counts.
  -- The inner loop runs for chunk_size tokens.
  -- chunk_size varies across CMS levels.
  -- torch.compile cannot unroll variable-length loops efficiently.
```

## The NL_Hecate Compilation Strategy

```
PRINCIPLE: Compile what's static. Pre-optimize what's dynamic.

STATIC (can be compiled):
  -- Q, K, V projections: W_K @ x^T, W_V @ x^T, W_Q @ x^T
     Pure matrix multiplies. No state mutation. Standard BLAS.
  -- Gate computations: sigmoid(W_g @ x + b_g)
     Elementwise operations. No state mutation.
  -- Loss computation: next-token prediction loss
     Standard cross-entropy. No state mutation.
  -- Attention: multi-head causal / SWA
     Standard attention kernel. No state mutation during computation.

  These are Rust functions annotated #[autodiff].
  rustc + LLVM optimize them at compile time.
  Enzyme provides their backward passes automatically.

DYNAMIC (pre-compiled kernels):
  -- Inner-loop memory update: M_{t+1} = M_t - eta * grad + momentum
     State-mutating. Token-sequential (or chunk-parallel).
     Pre-compiled as CUDA kernel pairs (forward + backward).
     The kernel IS the optimized implementation — no JIT needed.
  -- Retention mechanism application: forgetting, normalization
     Applied within the inner loop. Same kernel.
  -- CMS level gating: conditional execution based on Pulse
     Simple boolean check — branching, not computation.

  These are CUDA kernel pairs annotated #[custom_vjp].
  Hand-optimized per GPU architecture.
  No compilation needed — they're already optimized.
```

## Compilation Boundaries

```
-- The boundary between compiled and pre-compiled is EXPLICIT:

[Rust, compiled by rustc+Enzyme]     [CUDA, pre-compiled kernel pairs]
  x → project to q,k,v          →     inner_loop(M, k, v, gates)      →
  x → compute gates              →     retention(M, lambda)            →
  output → compute loss           ←     attention(q, k, v, mask)       ←

-- Enzyme handles the Rust side (auto-differentiated, LLVM-optimized).
-- CUDA kernels handle the hot inner-loop side (hand-optimized).
-- The boundary is the #[custom_vjp] annotation.
-- There is no "graph break" because there is no single graph.
-- Instead: two compilation domains connected by explicit interfaces.
```

## What We Give Up

```
-- No whole-model fusion across the compilation boundary.
   torch.compile can sometimes fuse a projection + activation + inner loop
   into a single kernel. We can't — the inner loop is a separate kernel.
   But: the inner loop kernel is hand-optimized, so fusion wouldn't help much.

-- No dynamic optimization based on input shapes.
   torch.compile can specialize kernels for specific batch/sequence sizes.
   We use fixed kernel implementations dispatched at runtime.
   But: our kernels are already specialized per GPU architecture.

-- No automatic backward pass for CUDA kernels.
   torch.compile's autograd generates backward passes automatically.
   We hand-write backward kernels from paper equations.
   But: this gives us CONTROL over the backward — we know it's correct
   because it's the paper's analytical gradient, not a generated one.
```

## What We Gain

```
-- Predictable performance.
   No "this model is 3x slower because torch.compile gave up."
   No graph breaks to debug. No recompilation storms.
   The kernel-pair pattern gives constant, known performance.

-- Debuggable compilation.
   When something is slow, we know which kernel to look at.
   There's no opaque compiled graph to reverse-engineer.
   Rust code is profiled by standard tools. CUDA kernels by nsight.

-- Correctness by construction.
   The backward pass IS the paper equation, not a compiler's guess.
   Rust reference implementations are the correctness oracle.
   CUDA kernels are verified against them.

-- Hardware portability.
   No dependency on a specific compiler's CUDA code generation.
   Our kernels target specific architectures explicitly.
   The dispatch layer handles portability at runtime.
```

## Rust Compilation Pipeline

```
Source: .rs files (Rust code with #[autodiff] annotations)
    → rustc frontend (type checking, borrow checking, trait resolution)
    → MIR (Mid-level IR — Rust's internal representation)
    → LLVM IR (Enzyme operates here)
    → Enzyme pass: generate backward functions for #[autodiff] functions
    → LLVM optimization passes (inlining, vectorization, etc.)
    → Machine code (x86_64 for CPU, linked with CUDA kernels)

The Enzyme pass happens at LLVM IR level:
  -- It sees clean Rust code compiled to LLVM IR
  -- No raw pointers, no unsafe blocks in the AD-traced path
  -- #[custom_vjp] functions appear as opaque calls with known VJPs
  -- #[no_autodiff] functions are invisible to Enzyme

This is compile-time AD, not runtime tracing.
The backward pass is determined once, at compile time.
There is no runtime graph building.
```

## CUDA Kernel Compilation

```
CUDA kernels are compiled separately:
  nvcc → .cubin (architecture-specific binary)
  nvcc → .ptx (architecture-independent intermediate)

Each kernel pair produces:
  forward_sm86.cubin   -- optimized for A6000
  backward_sm86.cubin  -- optimized for A6000
  forward.ptx          -- portable (JIT at first launch)
  backward.ptx         -- portable (JIT at first launch)

The dispatch layer loads the appropriate binary at runtime.
If no architecture-specific binary exists, PTX is JIT-compiled.
The Rust reference implementation is always available as fallback.
```

## Axiom Compliance

- **CS-45** (torch.compile cannot trace): Addressed by not using graph tracing
- **NL IS #7** (self-modifying): Self-modification happens in pre-compiled kernels, not traced graphs
- **NL IS NOT #3** (not static): The inner loop IS dynamic — we don't pretend otherwise
