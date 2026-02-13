# Enzyme AD Integration + Kernel-Pair Pattern

```
CONTRACT
  Purpose:    Defines how differentiation works across the entire system.
              Two mechanisms compose via the chain rule:
                (1) Enzyme AD on Rust code (LLVM IR level)
                (2) Hand-written backward kernels for CUDA operations
              Committee Finding 1: "The specs say 'compute gradient' but
              don't define HOW without autograd." This is how.
  Expects:    Rust code compiled to LLVM IR.
              CUDA kernel pairs: (forward_kernel, backward_kernel).
  Guarantees: Outer-loop gradients computed correctly via composition of
              Enzyme (Rust) and hand-written backward kernels (CUDA).
              Enzyme NEVER traces through raw CUDA.
              Inner-loop operations participate in the chain via #[custom_vjp],
              not #[no_autodiff] — the gradient chain is preserved.
  Cost:       Enzyme: ~1.5-2x forward pass for reverse mode AD on Rust code.
              Backward kernels: hand-optimized, comparable to forward kernel cost.
  Trade-off:  Enzyme handles composition; kernel authors handle kernel math.
              Kernel backward correctness depends on the author, not AD.
              But: analytical gradients from papers are the correctness source.
  Position:   specs/infrastructure/differentiation/00_enzyme_integration.md
              Addresses: Committee Finding 1, CS-40, nl_toolchain tool-11
  Source:     Enzyme AD (enzyme.mit.edu), CS-40, PyTorch autograd.Function pattern,
              JAX custom_vjp pattern, FlashAttention forward/backward kernel pattern
```

## The Two Differentiation Mechanisms

```
MECHANISM 1: Enzyme AD (Rust code)
  -- Differentiates Rust functions compiled to LLVM IR
  -- Handles: projections (W_K, W_V, W_Q), gate computations (sigmoid, softplus),
  --          composition logic (MAC concat, MAG gate, MAL preprocess),
  --          loss function, CMS frequency branching
  -- This is Enzyme's home turf — clean Rust → clean LLVM IR → reliable AD

MECHANISM 2: Hand-Written Backward Kernels (CUDA)
  -- Each CUDA forward kernel ships with a paired backward kernel
  -- The backward kernel IS the analytical gradient from the paper
  -- Example: Titans Eq 12 gives d(||M@k-v||^2)/d(M) = 2*(M@k-v)@k^T
  --          This becomes the backward kernel for chunkwise memory update
  -- Enzyme sees the pair as an opaque box with a known VJP rule

COMPOSITION: Chain Rule
  -- Enzyme chains through CUDA kernel pairs using their provided backward
  -- Same pattern as PyTorch (autograd.Function), JAX (custom_vjp),
  --    FlashAttention (hand-written forward + backward CUDA kernels),
  --    cuDNN (separate forward/backward API calls)
  -- No production AD system differentiates through raw CUDA. Neither do we.
```

## The Four Annotation Levels (CS-40: Opt-In)

```
#[autodiff]
  -- Enzyme differentiates through this function directly.
  -- Used for: Rust code in the outer-loop gradient path.
  -- Examples: gate computations, projection matmuls, loss function,
  --           composition pattern forward logic

#[enzyme_opaque]
  -- HARD BARRIER. Enzyme cannot trace into this function under any circumstances.
  -- At the LLVM IR level, this function appears as an opaque external call.
  -- Enzyme will NOT attempt to differentiate through it — not even accidentally.
  -- MANDATORY for all inner-loop kernel implementations.
  -- Without this, Enzyme may trace pointer mutations and generate garbage gradients.
  --
  -- Committee Finding 1: This attribute is the ENFORCEMENT MECHANISM.
  -- It is not optional. It is not a convention. It is a compiler requirement.
  -- Any MemoryUpdateRule implementation that omits #[enzyme_opaque] on its
  -- inner-loop kernel will fail to compile (trait bound violation).

#[custom_vjp(backward_fn)]
  -- Opaque to Enzyme. Provides its own backward function.
  -- Enzyme uses the provided backward to continue the chain rule.
  -- MUST be paired with #[enzyme_opaque] — the opaque barrier prevents
  -- Enzyme from tracing inside, and the custom_vjp provides the alternative
  -- gradient path. Together they form the kernel-pair pattern.
  -- Used for: CUDA kernel pairs. The outer-loop gradient DOES flow
  --           through these — via the hand-written backward, not Enzyme.
  -- Examples: chunkwise_memory_update, fused_retention_gradient,
  --           associative_scan, lattice_gla_forward

#[no_autodiff]
  -- Completely severed from the gradient chain.
  -- Used for: operations whose output does NOT affect the loss.
  -- Examples: debug logging, metrics, visualization, probe checks
  -- WARNING: Do NOT use this for inner-loop operations that sit
  --          between outer-loop params and the loss. That severs the
  --          outer-loop gradient chain. Use #[enzyme_opaque] + #[custom_vjp] instead.
```

CRITICAL: The original contract (v0.1.0) marked inner-loop operations as
`#[no_autodiff]`. This was a bug. Inner-loop operations DO participate in
the outer-loop gradient chain — the gradient d(loss)/d(W_K) flows THROUGH
the inner loop. The correct annotation is `#[custom_vjp]` with a backward
kernel that computes the VJP analytically.

## Gradient Flow: Complete Picture

```
Forward pass:
  x → [W_K @ x^T] → k        #[autodiff]     (Enzyme traces this)
    → [gate_fn(k,v)] → gates  #[autodiff]     (Enzyme traces this)
    → [inner_loop(M,k,v,gates)] → y  #[custom_vjp]  (CUDA kernel pair)
    → [loss_fn(y, target)] → loss     #[autodiff]     (Enzyme traces this)

Backward pass (d(loss)/d(W_K)):
  d(loss)/d(loss) = 1                          (seed)
  d(loss)/d(y) = Enzyme computes               (Rust, #[autodiff])
  d(y)/d(k)    = backward kernel computes      (CUDA, #[custom_vjp])
  d(k)/d(W_K)  = Enzyme computes               (Rust, #[autodiff])
  d(loss)/d(W_K) = chain of all the above      (Enzyme chains them)

  Enzyme handles steps 2 and 4 (Rust code).
  The backward kernel handles step 3 (CUDA).
  Enzyme chains them together — standard reverse-mode AD composition.
```

## Kernel Pair Implementation Pattern

```
-- Every hot operation follows this pattern:

-- 1. Rust reference implementation (always available)
fn chunkwise_update_rust(state: &Tensor, grads: &Tensor, ...) -> Tensor {
    // Pure Rust. Enzyme CAN differentiate through this directly.
    // Used for: development, testing, CPU fallback.
    ...
}

-- 2. CUDA forward kernel (architecture-specific)
#[cuda_kernel]
fn chunkwise_update_forward_sm86(state: &Tensor, grads: &Tensor, ...) -> Tensor {
    // Optimized for A6000/Ampere. Uses shared memory, tensor cores, etc.
    // Enzyme does NOT trace through this.
    ...
}

-- 3. CUDA backward kernel (hand-derived from paper equations)
#[cuda_kernel]
fn chunkwise_update_backward_sm86(state: &Tensor, d_output: &Tensor, ...) -> Tensor {
    // The analytical gradient from the paper, implemented as CUDA.
    // Also optimized for A6000/Ampere.
    // This IS the backward pass — not generated, hand-written.
    ...
}

-- 4. Registration with Enzyme via #[custom_vjp]
#[custom_vjp(chunkwise_update_backward_dispatch)]
fn chunkwise_update(state: &Tensor, grads: &Tensor, ...) -> Tensor {
    // Dispatch layer selects best forward kernel
    match select_backend(device) {
        RustReference => chunkwise_update_rust(state, grads, ...),
        CudaSM86      => chunkwise_update_forward_sm86(state, grads, ...),
        CudaPTX       => chunkwise_update_forward_ptx(state, grads, ...),
        _             => chunkwise_update_rust(state, grads, ...),
    }
}

fn chunkwise_update_backward_dispatch(d_output: &Tensor, ...) -> Tensor {
    // Dispatch layer selects best backward kernel
    match select_backend(device) {
        RustReference => chunkwise_update_backward_rust(d_output, ...),
        CudaSM86      => chunkwise_update_backward_sm86(d_output, ...),
        CudaPTX       => chunkwise_update_backward_ptx(d_output, ...),
        _             => chunkwise_update_backward_rust(d_output, ...),
    }
}
```

## Kernel Correctness Verification

```
-- The Rust reference backward is the CORRECTNESS ORACLE.
-- Every CUDA backward kernel must satisfy:

FOR all valid inputs (state, d_output):
  cuda_backward(state, d_output) ≈ rust_backward(state, d_output)
  within tolerance (1e-5 relative for fp32, 1e-2 for bf16)

-- This is how FlashAttention is tested: the triton/CUDA backward
-- is verified against a naive PyTorch implementation.
-- Our Rust reference serves the same role.
```

## Barrier Verification Tests (Committee Finding 1)

```
-- Committee Finding: "Hope-based engineering is not engineering."
-- The #[enzyme_opaque] barrier must be TESTED, not just declared.
-- Every kernel pair requires TWO classes of tests:

TEST CLASS 1: Barrier Integrity
  -- Verify that Enzyme does NOT contribute gradients through the opaque region.

  fn test_enzyme_barrier_holds<K: KernelPair>() {
    let (state, d_output) = random_valid_inputs();

    // Run forward through the kernel pair
    let output = K::forward(state);

    // Run Enzyme backward on the WRAPPER (not the kernel)
    let enzyme_grad = enzyme_reverse(output, state);

    // The enzyme_grad for the opaque region must be ZERO.
    // Only the #[custom_vjp] backward should contribute.
    assert!(enzyme_grad.opaque_contribution == 0.0,
      "Enzyme traced through the opaque barrier — #[enzyme_opaque] is broken");

    // The total gradient must equal the custom_vjp backward exactly.
    let vjp_grad = K::backward(d_output);
    assert_close!(enzyme_grad.total, vjp_grad,
      "Double counting: Enzyme AND custom_vjp both contributed");
  }

TEST CLASS 2: Analytical Correctness
  -- Verify that the hand-written backward matches the paper equation.

  fn test_analytical_gradient_correct<K: KernelPair>() {
    let (state, d_output) = random_valid_inputs();

    // Finite differences (numerical ground truth)
    let numerical_grad = finite_differences(K::forward, state, epsilon=1e-5);

    // Analytical backward (from paper equation)
    let analytical_grad = K::backward(d_output);

    // Must match within tolerance
    assert_close!(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6,
      "Analytical gradient does not match finite differences");
  }

-- These tests are MANDATORY for every kernel pair.
-- They run in CI. A failing barrier test blocks the build.
-- The barrier test catches the silent corruption scenario:
--   Enzyme traces a pointer mutation, produces a valid-looking number,
--   but the gradient is mathematical garbage.

TEST CLASS 3: Integration Gradient Test (Track Zero gate)
  -- Compute d(loss)/d(W_K) through a COMPLETE small forward pass.
  -- Single block, 64x64 matrices, 3 chunks of 8 tokens.
  -- This catches chain-rule COMPOSITION bugs that kernel-level tests miss.

  fn test_integration_gradient() {
    let config = SmallConfig { d: 64, chunks: 3, chunk_size: 8 };
    let model = build_single_block(config);

    // (a) Enzyme: the production path (Rust + kernel pairs + chain rule)
    let enzyme_grad = enzyme_reverse(model.forward(data), model.w_k);

    // (b) Finite differences: numerical ground truth
    let numerical_grad = finite_differences(
      |w_k| model.forward_with(w_k, data), model.w_k, epsilon=1e-5
    );

    // Must match within tolerance
    assert_close!(enzyme_grad, numerical_grad, rtol=1e-4, atol=1e-6,
      "Integration gradient: Enzyme chain rule composition is broken");
  }

  -- WHY this matters: kernel-level tests verify each piece.
  -- Integration tests verify the COMPOSITION via chain rule.
  -- If Enzyme mis-chains two correct kernels, kernel tests pass
  -- but integration tests fail. This is the Track Zero-A pass criterion.
```

## Mandatory Trait Bound: EnzymeOpaque

```
-- Committee Finding: "The compiler must be the enforcer."

TRAIT: EnzymeOpaque
  -- Marker trait. No methods. Signals that this type's inner-loop
  -- operations are barrier-protected against Enzyme tracing.

RULE: MemoryUpdateRule requires EnzymeOpaque.
  trait MemoryUpdateRule: MemoryStructure + AttentionalBias
                        + RetentionMechanism + MemoryAlgorithm
                        + EnzymeOpaque    // <-- MANDATORY
  {
    // ... existing methods ...
  }

  -- If a developer implements MemoryUpdateRule but forgets to mark
  -- their inner-loop kernel as #[enzyme_opaque], the impl of
  -- EnzymeOpaque will fail (the attribute is what satisfies the trait).
  -- Result: compile error, not runtime corruption.

  -- This moves enforcement from "a convention in a markdown file"
  -- to "a compiler requirement that blocks the build."
```

## Gradient Flow Through CMS

```
-- At step t with active_levels = [true, false, true, false]:
-- Level 0 (active): forward + backward (Enzyme + kernel pair)
-- Level 1 (frozen): NO computation, NO gradient
-- Level 2 (active): forward + backward (Enzyme + kernel pair)
-- Level 3 (frozen): NO computation, NO gradient

-- Frozen levels have ZERO cost — forward path is skipped entirely.
-- Enzyme's adjoint naturally handles this: no forward = no backward.
```

## Relationship to nl_toolchain

```
tool-11 (inner-optimization): Multi-level optimization inside forward()
  → Inner loop is a #[custom_vjp] kernel pair.
    Enzyme chains through it but doesn't trace inside.
    The kernel backward IS the inner-loop gradient from the paper.

tool-13 (compilation): torch.compile for self-modifying graphs
  → Enzyme operates on LLVM IR of Rust code (statically compiled).
    CUDA kernels are opaque (pre-compiled, dispatched at runtime).
    No dynamic graph tracing needed — the kernel pair IS the interface.
```
