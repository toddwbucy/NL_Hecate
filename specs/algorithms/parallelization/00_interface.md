# Parallelization Interface

```
CONTRACT
  Purpose:    Defines how sequential inner-loop operations get transformed
              into chunk-wise parallel forms. Every memory update rule has a
              sequential form (token-by-token) and MAY have a parallel form
              (chunk-at-a-time). This interface describes what all parallel
              forms share.
  Expects:    A MemoryUpdateRule that declares support for a parallelization strategy.
  Guarantees: Chunk-wise processing with bounded approximation error.
              Inter-chunk: sequential (unavoidable).
              Intra-chunk: parallel (the win).
  Cost:       Zero — trait definition. Strategy costs vary.
  Trade-off:  Larger chunks = more parallelism = worse approximation.
              Smaller chunks = better approximation = less parallelism.
              This is the fundamental trade-off of ALL strategies.
  Position:   specs/algorithms/parallelization/00_interface.md
              Parent of all parallelization strategy implementations.
  Source:     Titans Eqs 16-18, Atlas Eqs 15-16/34-41, TNT Eqs 3-7/13-15, Lattice Eqs 15-17
```

## The Sawtooth Problem

Every memory update rule has the form:

```
FOR t = 0 to T-1:
  state = f(state, x_t)     -- sequential: each step depends on previous
```

This is inherently sequential on GPUs. The "sawtooth" pattern emerges:
each token launches a small GPU kernel, waits for completion, launches the next.
GPU utilization drops to 5-15%. (CS-41: GPU utilization != throughput.)

## The Chunk-Wise Solution

Split the sequence into chunks of size C. Within each chunk, approximate
the sequential dependency:

```
-- Inter-chunk: sequential (unavoidable)
FOR chunk = 0 to T/C - 1:
  state_boundary = process_chunk(state_boundary, chunk_tokens)

-- Intra-chunk: PARALLEL (the win)
FUNCTION: process_chunk(state_boundary, tokens[0..C-1])
  -- All C tokens processed simultaneously
  -- Gradients computed w.r.t. state_boundary (frozen within chunk)
  -- This approximation is what enables parallelism
```

## Trait Definition

```
TRAIT: ParallelizationStrategy

  CHUNK_FORWARD(state_boundary: &Tensor, chunk_tokens: &[Tensor],
                chunk_size: usize, pulse: &Pulse) -> (Vec<Tensor>, Tensor)
    Process one chunk in parallel. Returns outputs for all tokens
    and the new state at the chunk boundary.

  COMPUTE_DECAY_PRODUCTS(alpha_sequence: &[f32]) -> Tensor
    Compute cumulative decay products needed for parallel form.
    Used by chunkwise GD and associative scan.

  SUPPORTED_BY() -> Vec<MemoryUpdateRuleKind>
    Which memory update rules support this strategy.

  -- Requirements:
  --   1. state_boundary is the ONLY sequential dependency between chunks
  --   2. All operations within a chunk use state_boundary, not intermediate states
  --   3. Output for token t uses state_boundary (not state_{t-1})
```

## Three Strategies

| Strategy | Mechanism | Exact? | Requirements | File |
|---|---|---|---|---|
| Chunkwise GD | Freeze state at boundary | Approximate | Any differentiable rule | 01_chunkwise_gd.md |
| Associative Scan | Blelloch parallel prefix | Exact | Linear recurrence only | 02_associative_scan.md |
| TNT Hierarchical | Global + local memories | Exact (local) | Independent local shards | 03_tnt_hierarchical.md |
| Lattice GLA | Linearized OSR | Approximate | Lattice/Trellis rules | 04_lattice_gla.md |
| Atlas Parallel | Momentum independent of memory | Exact (momentum) | Atlas Omega rule | 05_atlas_parallel.md |

## Chunk Size Selection

```
-- The Pulse carries chunk_boundaries for each CMS level.
-- Different frequency levels may use different chunk sizes.
-- Typical ranges:
--   Titans: C = 64-512 (mid-range)
--   TNT: C = 128-2048 (hierarchical, larger effective chunks)
--   Atlas: sliding window (variable, overlapping)
--   Lattice: C = 16-256 (smaller = less approximation error)
```

## What This Interface Does NOT Specify

- **Chunk size**: Application-dependent. The Pulse provides it.
- **Memory rule**: Any rule that declares support for a strategy works.
- **Hardware**: GPU-agnostic. CUDA kernels are optimizations.
- **Composition pattern**: MAC/MAG/MAL are orthogonal. Memory branch runs parallel regardless.
