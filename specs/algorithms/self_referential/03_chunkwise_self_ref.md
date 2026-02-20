# Chunkwise Training for Self-Referential Titans

<!-- HADES: hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90); hope_equations/eq-092-dot-product-recurrent (§8.2 Eq 92); hope_equations/eq-093-freq-transfer (§8.2 Eq 93) -->
```text
CONTRACT
  Purpose:    Chunkwise training adapts the existing parallelization strategies
              (see specs/algorithms/parallelization/) to self-referential Titans
              with 6 memory modules. The key insight: each component memory
              M_square uses the state at the START of the chunk for gradient
              computation, not the current state. This decouples tokens within
              a chunk, allowing parallel gradient computation. The update
              frequency f_square = L/C_square can vary per component — fast
              components (k, v) update every token, slow components (eta, alpha)
              update less frequently.
  Expects:    Chunk size C (tokens per chunk). 6 component memories
              M_k, M_v, M_q, M_eta, M_alpha, M_mem. Self-generated values
              v_hat per component (Phase 3) or shared v_t (Phase 2).
  Guarantees: Within each chunk, all tokens compute gradients w.r.t. the SAME
              memory state at the chunk boundary M_{square, C*ceil(t/C)} (paper
              notation, Eq 90). This enables parallel gradient computation
              across the C tokens in each chunk. At chunk boundaries, the
              memory state advances sequentially.
              The chunkwise approximation error is bounded by O(C * eta)
              and converges to the exact sequential update as C → 1.
  Cost:       O(C * d^2) per chunk per component (C parallel gradient evals).
              With 6 components: O(6 * C * d^2) per chunk total.
              The parallelism reduces wall-clock time from O(seq_len) to
              O(seq_len / C) sequential steps, each with O(C) parallel work.
  Trade-off:  Larger C gives more parallelism but worse approximation (gradients
              computed from stale state). Smaller C gives better approximation
              but less parallelism. C = 1 recovers exact sequential update.
              The self-referential setting amplifies the staleness effect because
              6 memories use stale states simultaneously.
  Position:   specs/algorithms/self_referential/03_chunkwise_self_ref.md
              Parent: 00_interface.md (self-referential framework)
              Sibling of: specs/algorithms/parallelization/01_chunkwise_gd.md
                          (standard chunkwise, single memory)
  Source:     HOPE (2512.24695) §8.2 Eqs 90-93
```

## Chunkwise Update for All Components (Eq 90)

The general chunkwise update applies to every component memory uniformly:

<!-- HADES: hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90) -->
```text
-- Chunkwise self-referential update (HOPE Eq 90):
FUNCTION: chunkwise_self_ref_step(M_square: &mut Tensor, k: &Tensor,
                                   v_hat: &Tensor, alpha_t: f32, eta_t: f32,
                                   M_chunk_start: &Tensor) -> ()
  -- M_square: component memory [d, d] (will be updated)
  -- M_chunk_start: memory state at chunk boundary C*ceil(t/C) (frozen for gradient)
  -- k: key vector [d, 1]
  -- v_hat: self-generated value [d, 1]

  -- Gradient computed w.r.t. CHUNK BOUNDARY state (not current state)
  grad = grad_L(M_chunk_start; k, v_hat)

  -- DGD-style retention + update (same as Eq 88)
  M_square = M_square @ (alpha_t * I - eta_t * k @ k^T) - eta_t * grad

  -- for square in {k, v, q, eta, alpha, memory}

  -- Key difference from sequential (Eq 88):
  --   Sequential: grad_L(M_{square,t-1}; ...)  — uses CURRENT state
  --   Chunkwise:  grad_L(M_{square,C*ceil(t/C)}; ...) — uses FROZEN state
  --   Within a chunk, all tokens see the same M for gradient computation.
  --   M still updates sequentially through the DGD recurrence,
  --   but the gradient is computed from a frozen snapshot.
```

## Dot-Product Objective (Eq 92)

For the dot-product (Hebbian) objective, the chunkwise recurrence has a
particularly simple form:

<!-- HADES: hope_equations/eq-092-dot-product-recurrent (§8.2 Eq 92) -->
```text
-- Dot-product chunkwise recurrence (HOPE Eq 92):
M_{square,t} = M_{square,t-1} @ (alpha_t * I - eta_t * k_t @ k_t^T)
               - eta_t * v_hat_{square,t} @ k_t^T

-- L(M; k, v) = -<M @ k, v>  (dot-product similarity)
-- grad_L = -v @ k^T           (independent of M!)
--
-- Because the gradient does NOT depend on M, the chunk-start approximation
-- introduces NO error for the dot-product objective.
-- Chunkwise = sequential for Hebbian/dot-product.
--
-- This recurrence accepts the "fast parallelizable dual form" from Titans:
--   The DGD recurrence M_t = M_{t-1} @ A_t + B_t (where A_t, B_t are
--   per-token matrices) can be parallelized via scan operations.
--   See specs/algorithms/parallelization/01_chunkwise_gd.md for the
--   scan-based parallel implementation.
```

## L2-Regression Objective (Eq 93)

For the L2 (Delta rule) objective, the gradient depends on M and requires
the chunk-start approximation:

<!-- HADES: hope_equations/eq-093-freq-transfer (§8.2 Eq 93) -->
```text
-- L2 chunkwise recurrence (HOPE Eq 93):
M_{square,t} = M_{square,t-1} @ (alpha_t * I - eta_t * k_t @ k_t^T)
               - eta_t * (M_{square,chunk_start} @ k_t - v_hat_{square,t}) @ k_t^T

-- L(M; k, v) = ||M @ k - v||^2  (L2 regression)
-- grad_L = (M @ k - v) @ k^T     (depends on M!)
--
-- The chunk-start approximation:
--   Uses M_{square,C*ceil(t/C)} instead of M_{square,t-1} in the gradient.
--   Within a chunk, M_{chunk_start} is CONSTANT across all tokens.
--   This means (M_{chunk_start} @ k_t - v_hat) @ k_t^T can be computed
--   in parallel for all tokens in the chunk.
--
-- Approximation error:
--   At token t within a chunk starting at t_0:
--   Exact gradient: (M_{t-1} @ k_t - v_hat) @ k_t^T
--   Approx gradient: (M_{t_0} @ k_t - v_hat) @ k_t^T
--   Error = (M_{t-1} - M_{t_0}) @ k_t @ k_t^T
--   ||M_{t-1} - M_{t_0}|| <= (t - t_0) * eta * max_grad = O(C * eta)
--   Smaller chunks or smaller learning rates reduce the error.
```

## Per-Component Update Frequencies

Different component memories can use different chunk sizes:

<!-- HADES: Derived from hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90), per-component frequency -->
```text
-- Update frequency f_square = L / C_square (HOPE §8.2)
--   L: context length (total tokens)
--   C_square: chunk size for component square
--   f_square: number of chunk-boundary updates per context

-- Example configuration:
--   M_k, M_v (projections): C = 1 (update every token, sequential)
--     These are the most performance-critical — stale keys/values
--     directly degrade output quality.
--
--   M_q (query projection): C = 4 (update every 4 tokens)
--     Query is less sensitive — a slightly stale query still retrieves
--     from the correct neighborhood.
--
--   M_eta, M_alpha (gate memories): C = 16 (update every 16 tokens)
--     Gates change slowly — the sigmoid saturation means small M changes
--     produce small gate changes. Larger chunks are safe.
--
--   M_mem (main memory): C = 1 (every token) or C = chunk_size (match SWA)
--     Must match the existing parallelization strategy for the main memory.
--     If using chunkwise GD (01_chunkwise_gd.md), C is already defined.

-- Mixed-frequency scheduling:
-- Different C values mean different components advance at different rates.
-- At any given token t:
--   If t is a chunk boundary for M_k (C_k divides t): M_k advances
--   If t is NOT a chunk boundary for M_alpha: M_alpha uses stale state
-- This is analogous to CMS frequency scheduling but applied WITHIN
-- the self-referential component set, not across CMS levels.
```

## Parallel Computation Within Chunks

Within a chunk, all tokens can compute their contributions in parallel:

<!-- HADES: Derived from hope_equations/eq-093-freq-transfer (§8.2 Eq 93), parallel computation pattern -->
```text
-- For L2 objective, within chunk [t_0, t_0 + C):
--   M_frozen = M_{square, t_0}                    -- snapshot at chunk start

--   FOR t in [t_0, t_0 + C) IN PARALLEL:
--     error_t = M_frozen @ k_t - v_hat_t           -- parallel (M_frozen shared)
--     grad_t = error_t @ k_t^T                     -- parallel (independent per t)

--   SEQUENTIAL recurrence (using precomputed grads):
--     FOR t in [t_0, t_0 + C):
--       M = M @ (alpha_t * I - eta_t * k_t @ k_t^T) - eta_t * grad_t

-- The parallel phase computes C gradients simultaneously.
-- The sequential phase applies the DGD recurrence using those gradients.
-- The DGD recurrence has the form M_t = M_{t-1} @ A_t + B_t
-- which admits parallelization via associative scan:
--   A_t = alpha_t * I - eta_t * k_t @ k_t^T       -- [d, d] per token
--   B_t = -eta_t * grad_t                          -- [d, d] per token
--   Scan: (A, B) ⊕ (A', B') = (A' @ A, A' @ B + B')
-- This reduces the sequential phase from O(C) steps to O(log C) steps.

-- For dot-product objective (Eq 92):
-- The gradient is M-independent, so BOTH phases are fully parallel.
-- No sequential recurrence needed beyond the scan.
```

## Interaction with CMS

Self-referential chunkwise training interacts with the CMS frequency
schedule in a layered manner:

<!-- HADES: Derived from hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90), CMS interaction -->
```text
-- Two frequency dimensions:
--   1. CMS level frequency: how often outer-loop gradients flow (Level 0: every
--      chunk, Level 3: every 512 chunks). Controls OUTER-LOOP learning cadence.
--   2. Component chunk size C_square: how often each self-ref component uses
--      fresh state for gradient computation. Controls INNER-LOOP parallelism.

-- These are orthogonal:
--   CMS Level 0, C_k = 1: M_k updates every token, outer-loop grads every chunk.
--   CMS Level 2, C_eta = 16: M_eta uses 16-token chunks, outer-loop grads every
--     64 chunks. The 16-token chunking is WITHIN each CMS step.

-- At frozen CMS levels:
--   Component memories still update via inner-loop (DGD recurrence).
--   No outer-loop gradients accumulate for frozen-level components.
--   The chunkwise approximation is irrelevant for frozen levels because
--   there is no outer-loop gradient to approximate — only the inner-loop
--   recurrence runs, which is always sequential/exact.

-- Per-level component configuration:
--   Each CMS level has its own set of 6 component memories.
--   Each level MAY use different chunk sizes for its components.
--   Fast CMS levels: small C (more accuracy, less parallelism).
--   Slow CMS levels: large C (more parallelism, acceptable staleness
--     because slow levels are already updating infrequently).
```

## Gradient Derivation (for tape integration)

<!-- HADES: Derived from hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90), VJP for chunkwise self-ref -->
```text
-- Forward (L2 objective, one chunk):
--   M_frozen = M_{square, t_0}                     -- snapshot
--   FOR t in chunk:
--     error_t = M_frozen @ k_t - v_hat_t
--     grad_t = error_t @ k_t^T
--     M_{square,t} = M_{square,t-1} @ A_t + B_t   -- A_t, B_t from DGD

-- Given: dL/dM_{square, t_end} (upstream from next chunk)
-- Need: dL/dM_{square, t_0} (for previous chunk), dL/dM_frozen, dL/dk_t, dL/dv_hat_t

-- Step 1: Backward through DGD recurrence (reverse scan)
--   Standard recurrence backward — same as 01_chunkwise_gd.md.
--   Produces dL/dgrad_t for each token, dL/dM_{square, t_0} (through recurrence).

-- Step 2: Backward through gradient computation
--   grad_t = error_t @ k_t^T
--   dL/derror_t = dL/dgrad_t @ k_t
--   dL/dk_t (through grad) = error_t^T @ dL/dgrad_t

-- Step 3: Backward through error = M_frozen @ k - v_hat
--   dL/dM_frozen += dL/derror_t @ k_t^T            -- accumulate across all t
--   dL/dk_t (through error) = M_frozen^T @ dL/derror_t
--   dL/dv_hat_t = -dL/derror_t

-- Combine:
--   dL/dM_{square, t_0} = dL/dM (through recurrence) + dL/dM_frozen
--   The chunk-start state receives gradient from TWO paths:
--     1. Through the sequential DGD recurrence (standard)
--     2. Through the frozen gradient computation (all C tokens accumulate)

-- Step 4: Backward through self-generated values (if Phase 3)
--   dL/dv_t (through v_hat) = M_{square,t_0}^T @ dL/dv_hat_t
--   dL/dM_square (through v_hat read) = dL/dv_hat_t @ v_t^T
--   This is the standard self-generated value backward (01_self_generated_values.md).
```

## Implementation Notes

1. **Reuses existing chunkwise infrastructure**: The DGD recurrence
   `M_t = M_{t-1} @ A_t + B_t` has the same form as the standard Titans
   chunkwise update in `01_chunkwise_gd.md`. The scan-based parallelization
   applies directly. The only difference is that self-referential Titans
   run 6 parallel instances of the scan (one per component).

2. **Memory snapshot**: At each chunk boundary, each component memory
   M_square must be snapshotted. For matrix memory [d, d], this is 6 × d^2
   floats per chunk boundary. The snapshot is read-only during the chunk and
   freed at chunk end. This is an `inner_loop_state` — not checkpointed.

3. **MLP architecture (Eq 91)**: For MLP memories (Eq 89/91), the chunkwise
   approximation freezes the MLP weights at chunk start. The gradient of
   the MLP loss w.r.t. frozen weights is computed for all tokens in
   parallel. This is more expensive than matrix memory because MLP
   backward involves two matmuls per token, but the parallelism still
   applies.

4. **Dot-product fast path**: When using the dot-product (Hebbian) objective
   (Eq 92), the gradient is M-independent. No chunk-start snapshot is
   needed — the recurrence can be fully parallelized without approximation.
   Detect this at configuration time and skip the snapshot allocation.

5. **Mixed-frequency implementation**: Different C_square values per component
   require a scheduling table that tracks chunk boundaries independently for
   each component. The simplest implementation: let all components share the
   same chunk size C (matching the main memory), and only optimize
   per-component frequencies when profiling shows it matters.

6. **Interaction with gradient checkpointing**: The existing gradient
   checkpointing (for the main memory's per-token M states) applies
   independently to each of the 6 component memories. With checkpointing,
   component memory states at chunk boundaries are recomputed during
   backward rather than stored — trading 6× compute for 6× memory savings.
   **Caution (CS-48)**: Gradient checkpointing can degrade NL convergence
   because CMS-aware gradient flow depends on exact intermediate states at
   chunk boundaries. Recomputation introduces floating-point non-determinism
   that compounds across CMS levels. Profile carefully before enabling —
   the memory savings may not justify the convergence cost.

## Axiom Compliance

- **NL IS #4** (compressing context): Chunkwise training compresses the per-token sequential dependency into chunk-level sequential steps, enabling parallel processing of the token stream without losing the memory's context-compression property.
- **NL IS #9** (principled not ad hoc): The chunkwise approximation derives from freezing the gradient evaluation point — a standard optimization technique (stale gradient methods). The approximation error is analytically bounded by O(C * eta).
- **MIRAS IS #1** (orthogonal design choices): Chunkwise parallelization is orthogonal to all four MIRAS knobs and to the self-referential extension. Any component memory using any rule can be chunked with any chunk size C.
