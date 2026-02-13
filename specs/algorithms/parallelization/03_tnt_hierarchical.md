# TNT Hierarchical Memory

```
CONTRACT
  Purpose:    Solve the sawtooth problem by splitting into independent shards.
              One global memory (coarse-grained, sequential) plus N local
              memories (fine-grained, fully parallel). 17.37x faster than
              Titans baseline with BETTER quality.
  Expects:    Input sequence. Global chunk size C_G. Local chunk size C_L.
              A MemoryUpdateRule for both global and local memories.
  Guarantees: Local memories are INDEPENDENT within shards (fully parallel).
              Global memory carries information across shards.
              Periodic reset of local memories prevents drift.
              Architecture-agnostic: works with Titans, Atlas, Mamba2, TTT.
  Cost:       Sequential depth: O(T / C_G) for global memory.
              Parallel width: N = C_G / C_L local memories per shard.
              Wall-clock dominated by global memory (coarse but sequential).
  Trade-off:  Massive speedup (17x) from parallelizing local memories.
              But: global memory must capture enough cross-shard context.
              If C_G is too large, global memory becomes the bottleneck.
              If C_G is too small, parallelism benefit disappears.
  Position:   specs/algorithms/parallelization/03_tnt_hierarchical.md
  Source:     TNT (2511.07343) Eqs 3-7, 13-15, Section 3
```

## Architecture

```
Sequence: [====================T tokens======================]
           |--- shard 0 ---|--- shard 1 ---|--- shard 2 ---|
           |  C_G tokens   |  C_G tokens   |  C_G tokens   |

Global Memory (sequential across shards):
  M_global updates at shard boundaries only.
  Sequential depth = T / C_G (very coarse).

Local Memories (parallel within each shard):
  Each shard has N = C_G / C_L independent local memories.
  All N local memories process simultaneously.
  Reset to M_global state at shard start.

  Shard 0: [local_0 | local_1 | ... | local_{N-1}]  -- all parallel
  Shard 1: [local_0 | local_1 | ... | local_{N-1}]  -- reset, then parallel
  Shard 2: [local_0 | local_1 | ... | local_{N-1}]  -- reset, then parallel
```

## The Three TNT Innovations

### 1. Hierarchical Memory

```
ALGORITHM: tnt_forward(x: &Tensor, M_global: &mut State, M_local_template: &State,
                       rule: &dyn MemoryUpdateRule, C_G: usize, C_L: usize,
                       pulse: &Pulse) -> Vec<Tensor>
  outputs = []
  N_shards = T / C_G

  FOR shard = 0 to N_shards - 1:
    shard_tokens = x[shard * C_G .. (shard + 1) * C_G]

    -- Reset local memories to global state
    local_memories = [M_global.clone() for _ in 0..C_G/C_L]

    -- Process all local chunks IN PARALLEL
    local_outputs = PARALLEL_FOR i = 0 to C_G/C_L - 1:
      chunk = shard_tokens[i * C_L .. (i + 1) * C_L]
      out = process_local_chunk(local_memories[i], chunk, rule, pulse)
      yield out

    outputs.extend(flatten(local_outputs))

    -- Update global memory at shard boundary (TNT Eq 5)
    -- Global sees a SUMMARY of what the shard processed
    shard_summary = compute_shard_summary(shard_tokens, local_outputs)
    M_global = rule.WRITE(M_global, shard_summary, pulse)

  return outputs
```

### 2. Q-K Projection (TNT Eq 13-14)

```
FUNCTION: qk_project(x: &Tensor, W_Q: &Tensor, W_K: &Tensor) -> Tensor
  -- TNT solves the compression-retrieval domain mismatch:
  -- Keys are optimized for WRITING (compression)
  -- Queries are optimized for READING (retrieval)
  -- These should be in the SAME domain but aren't by default.

  -- Q-K projection aligns them:
  q = x @ W_Q^T
  k = x @ W_K^T
  q_aligned = q @ W_QK^T    -- project query into key domain
  return q_aligned
```

### 3. Two-Stage Building

```
ALGORITHM: tnt_two_stage_build(data, model, outer_optimizer)
  -- Stage 1: Build with SMALL chunk size (better approximation)
  FOR step in 0..N_stage1:
    loss = model.forward(data, chunk_size=C_small)
    outer_optimizer.step(loss)   -- Enzyme AD for outer loop

  -- Stage 2: Build with LARGE chunk size (full throughput)
  FOR step in N_stage1..N_total:
    loss = model.forward(data, chunk_size=C_large)
    outer_optimizer.step(loss)

  -- Stage 1 establishes good initial conditions.
  -- Stage 2 runs at full speed with those conditions.
  -- TNT shows this matches or beats single-stage with C_large from start.
```

## Compatibility

```
SUPPORTED_BY:
  - TitansLMM:    YES  (original TNT paper demonstrates)
  - Atlas Omega:   YES  (TNT Appendix D)
  - Mamba2:        YES  (TNT Appendix D)
  - TTT:           YES  (TNT Appendix D)
  - Lattice:       YES  (compatible — local memories are small state)
  - Trellis:       YES  (same argument as Lattice)
  - MONETA/YAAD/MEMORA: YES  (local MLP memories are independent)
```

## Axiom Compliance

- **TNT IS #1** (hierarchical memory): Global + N local memories
- **TNT IS #2** (multi-resolution): Global = coarse, local = fine
- **TNT IS #5** (architecture-agnostic): Works with any MemoryUpdateRule
- **TNT IS #6** (periodic reset): Local memories reset at shard boundaries
- **NL IS #2** (parallel): Local memories are fully independent within shards
