# Attention (The Non-NL Component)

```
CONTRACT
  Purpose:    Attention is the one component that is NOT novel in NL.
              It is standard causal attention (or sliding-window attention),
              borrowed wholesale from the Transformer literature.
              NL_Hecate wraps it, doesn't reinvent it.
              The composition patterns (MAC/MAG/MAL) determine how
              memory and attention interact — not attention itself.
  Expects:    Query, key, value tensors from input or from memory output.
              A causal mask (full or sliding window).
  Guarantees: Standard causal attention output.
              Two variants: full causal (for MAC) and sliding window (for MAG/MAL).
              Both use the kernel-pair pattern (Rust reference + optional CUDA).
  Cost:       Full causal: O(T^2 * d) per chunk. Sliding window: O(T * w * d).
              FlashAttention-style kernels reduce memory from O(T^2) to O(T).
  Trade-off:  Attention is a commodity. We use established implementations.
              No novel attention mechanism here — the novelty is in memory.
              Using existing attention kernels (FlashAttention pattern)
              reduces development risk for the most well-studied operation.
  Position:   specs/infrastructure/attention/00_attention.md
              Addresses: nl_toolchain tool-02, composition pattern requirements
  Source:     Titans (2501.00663) Section 3.2 (MAC/MAG/MAL attention requirements);
              FlashAttention (Tri Dao); MIRAS (2504.13173) SWA usage
```

## Two Attention Modes

```
MODE 1: Full Causal Attention
  -- Used by: MAC composition pattern
  -- Standard multi-head causal attention over the FULL context
  -- Input may include memory tokens prepended to the sequence (MAC)
  -- Q, K, V projections are OuterLoopParam (Enzyme differentiates)

  fn full_causal_attention(
    q: &Tensor,   // [batch, heads, seq, d_k]
    k: &Tensor,   // [batch, heads, seq + mem_tokens, d_k]
    v: &Tensor,   // [batch, heads, seq + mem_tokens, d_v]
    mask: &CausalMask,
    pulse: &Pulse,
  ) -> Tensor    // [batch, heads, seq, d_v]

  -- When used with MAC: memory reads produce tokens prepended to K, V
  -- Attention sees (memory_context, current_input) as its full context
  -- The reflective gate (Eq 25) operates on attention output, not inside it


MODE 2: Sliding Window Attention (SWA)
  -- Used by: MAG and MAL composition patterns
  -- Attends only to the nearest w tokens (window size w)
  -- Memory handles long-range dependencies; attention handles local
  -- This is the MIRAS design: memory for long-range, SWA for local

  fn sliding_window_attention(
    q: &Tensor,   // [batch, heads, seq, d_k]
    k: &Tensor,   // [batch, heads, seq, d_k]
    v: &Tensor,   // [batch, heads, seq, d_v]
    window_size: usize,
    pulse: &Pulse,
  ) -> Tensor    // [batch, heads, seq, d_v]

  -- Window size is a configuration parameter, not learned
  -- Typical values: 256, 512, 1024
  -- O(T * w * d) vs O(T^2 * d) for full causal
```

## Kernel-Pair Pattern for Attention

```
-- Attention follows the same kernel-pair pattern as everything else:

1. Rust reference: straightforward Q @ K^T / sqrt(d_k) → softmax → @ V
   - Enzyme CAN differentiate through this directly
   - Used for development and testing

2. CUDA forward kernel: FlashAttention-style tiled computation
   - O(T) memory instead of O(T^2)
   - Uses online softmax + recomputation
   - Hardware-specific optimizations (tensor cores, shared memory)

3. CUDA backward kernel: FlashAttention backward pass
   - Recomputes attention weights from saved Q, K, V (not stored)
   - Produces dQ, dK, dV for the chain rule
   - This is a well-understood kernel — Tri Dao's implementation is public

-- Enzyme chains through the attention kernel pair via #[custom_vjp]
-- The attention backward kernel provides dQ, dK, dV
-- Enzyme continues the chain from dQ, dK, dV back to the projection weights
```

## Attention IS NOT Memory

```
CRITICAL DISTINCTION (NL IS #6, IS NOT #5):
  -- Attention is NOT a memory mechanism in NL.
  -- Attention processes a FIXED context window.
  -- Memory persists and accumulates across windows.

  -- In conventional Transformers, attention IS the memory
  --   (KV cache grows with context).
  -- In NL, memory IS the memory; attention is just local processing.

  -- The composition patterns define the boundary:
  --   MAC: memory FEEDS attention (extends its context)
  --   MAG: memory GATES attention (controls what it outputs)
  --   MAL: memory PREPROCESSES for attention (structures its input)

  -- The attention mechanism itself is identical across all three patterns.
  -- Only the data flow changes.
```

## Integration with Composition Patterns

```
MAC (Memory As Context):
  attention receives: [memory_tokens; current_tokens]
  attention type: full causal (needs to attend to memory tokens)
  memory tokens are: read from persistent memory before attention

MAG (Memory As Gate):
  attention receives: current_tokens only
  attention type: sliding window (memory handles long-range)
  memory output is: sigmoid gate applied to attention output

MAL (Memory As Layer):
  attention receives: memory_output (not raw input)
  attention type: sliding window (memory already processed long-range)
  memory preprocesses: raw input → memory-processed representation
```

## What We Don't Reinvent

```
-- We do NOT implement novel attention mechanisms.
-- We do NOT implement sparse attention, linear attention, or attention alternatives.
-- Those are the memory update rules (Titans, MIRAS, Atlas, Lattice, Trellis).

-- Attention in NL_Hecate is:
--   (a) Multi-head causal attention (full or SWA)
--   (b) With standard learned Q, K, V projections
--   (c) Using the kernel-pair pattern for differentiation
--   (d) Wrapped in a composition pattern for memory interaction

-- This is intentional. The MIRAS paper shows that:
--   "attention-free models can match or beat hybrid models" (IS NOT #6)
--   Attention is the conventional part. Memory is the novel part.
--   We invest engineering effort in memory, not attention.
```

## Axiom Compliance

- **NL IS NOT #5** (not optimizers as just optimizers): Attention is the counterpoint — it IS just attention. Memory is where IS #6 lives.
- **MIRAS IS NOT #6** (not attention-dependent): The system works without attention entirely (pure memory variants). Attention is a convenience, not a requirement.
- **Titans IS #6** (compositional identity): MAC/MAG/MAL define how attention and memory compose — attention is one component in that composition.
