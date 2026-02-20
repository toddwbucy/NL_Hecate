# Newton-Schulz Iteration (Inner-Loop Preconditioner)

```text
CONTRACT
  Purpose:    Newton-Schulz (NS) iteration orthogonalizes the momentum or
              gradient signal before applying it to memory. This is the
              "locally optimal" upgrade — where GD uses raw gradients and DGD
              uses state-dependent gradients, Newton-Schulz maps them into an
              orthogonal space that preserves gradient direction information.
              In the outer loop this is Muon (Jordan et al. 2024); in the inner
              loop this is Atlas's memory optimizer.
  Expects:    Gradient or momentum tensor G (d x d matrix), number of iterations k.
  Guarantees: After k iterations, output O_k approximates the nearest
              semi-orthogonal matrix to G. As k → inf, O_k converges to the
              polar factor of G. k=5 is sufficient in practice (Atlas default).
              The iteration is purely matrix multiplications — no eigendecomposition.
  Cost:       Per iteration: O(d^3) — three matrix multiplications.
              k=5 iterations: 15 matmuls total. For d=64 (inner loop), this is
              ~0.8M FLOPs per NS call. For d=2048 (outer loop), ~130G FLOPs.
  Trade-off:  Better gradient geometry (locally optimal updates) at the cost of
              k extra matmul passes per step. The Atlas ablation shows this is
              worth it: Muon is the defining feature of Atlas vs OmegaNet.
  Position:   specs/algorithms/optimization_machinery/07_newton_schulz_inner.md
              Sibling of 06_implicit_gd.md (NS approximates the implicit solution)
  Source:     HOPE (2512.24695) §4.2 Eqs 42-44; Atlas (2505.23735) §5 Eq 32, §5.1 Eqs 39-41
```

## What Newton-Schulz Does

The core idea: raw gradients have arbitrary scale and correlation structure.
Newton-Schulz maps them to a space where the update preserves directional
information without distorting it through magnitude differences.

<!-- HADES: hope_equations/eq-043-orthogonalization-obj (§4.2 Eq 43) -->
```text
-- Orthogonalization objective (HOPE Eq 43):
L_orth(P(g); g) = ||P(g)^T P(g) - I||^2_F

-- Find P that makes P(g) as close to orthogonal as possible.
-- When P(g)^T P(g) = I, the columns of P(g) are orthonormal.
-- This means the gradient update preserves all directional info
-- without any dimension dominating others.
```

## The Newton-Schulz Iteration

Solve the orthogonalization objective by gradient descent:

<!-- HADES: hope_equations/eq-044-newton-schulz-iteration (§4.2 Eq 44) -->
```text
-- Newton-Schulz iteration (HOPE Eq 44):
FUNCTION: newton_schulz_k(G: &Tensor, k: usize) -> Tensor
  -- G: input gradient or momentum matrix [d, d]
  -- k: number of iterations (default: 5)
  -- Returns: O_k ≈ nearest semi-orthogonal matrix to G

  O_0 = G                              -- initialize with raw gradient
  for i in 0..k:
    O_{i+1} = O_i - zeta_{i+1} * (O_i - G + 2 * O_i @ (O_i^T @ O_i - I))
    -- Three matrix multiplications per iteration:
    --   1. O_i^T @ O_i        (d x d)
    --   2. subtract I, scale
    --   3. O_i @ (result)     (d x d)

  return O_k

-- The 3-degree polynomial form (with zeta = 1):
--   O_{i+1} = (3/2) * O_i - (1/2) * O_i @ O_i^T @ O_i
-- This is the classical Newton-Schulz iteration for matrix sign/polar.
```

The iteration converges quadratically — each step roughly doubles the number
of correct digits. k=5 gives ~32 digits of accuracy (far beyond f32 precision).

## Muon: Newton-Schulz as Outer-Loop Optimizer

The Muon optimizer (Jordan et al. 2024) applies NS to the momentum signal
before the weight update:

<!-- HADES: hope_equations/eq-042-muon-optimizer (§4.2 Eq 42) -->
```text
-- Muon optimizer (HOPE Eq 42):
FUNCTION: muon_step(W: &mut Tensor, m: &mut Tensor,
                     g: &Tensor, alpha_t: f32, eta_t: f32, k: usize) -> ()
  -- Standard momentum accumulation
  m = alpha_t * m - eta_t * g

  -- Newton-Schulz orthogonalization of momentum
  m_orth = newton_schulz_k(m, k)       -- k=5 default

  -- Apply orthogonalized momentum
  W = W + m_orth
```

This is the **outer-loop** application (updating W_K, W_V, W_Q projections).
The M3 optimizer (HOPE §7.2) extends this with CMS-aware multi-scale momentum.

## Atlas: Newton-Schulz as Inner-Loop Optimizer

Atlas applies Newton-Schulz to the inner-loop memory update — the momentum
accumulated over the sliding window is orthogonalized before being written
to memory:

<!-- HADES: atlas_equations/eq-032-atlas-memory-muon (§5 Eq 32) -->
```text
-- Atlas memory update (Eq 32):
M_t = alpha_t * M_{t-1} + newton_schulz_k(S_t)

-- S_t is the momentum from Omega-rule gradients:
S_t = theta_t * S_{t-1} + grad sum_{i=t-c+1}^{t} gamma_i * ||M(k_i) - v_i||^2

-- Newton-Schulz makes the memory update LOCALLY OPTIMAL:
-- it approximates the best possible single-step update direction.
```

## Parallelization: NS in Chunks

The key Atlas parallelization insight: within a chunk, all momentum values
S_t can be computed independently (gradients use the chunk-boundary M_{t'}):

<!-- HADES: atlas_equations/eq-039-fully-parallelizable-momentum (§5.1 Eq 39); atlas_equations/eq-040-newton-schulz5-parallel (§5.1 Eq 40) -->
```text
-- Atlas parallel training (Eqs 39-41):
-- Step 1: Compute all gradients in chunk w.r.t. M_{t'} (parallel)
-- Step 2: Accumulate into S_t via linear recurrence (parallel scan)
-- Step 3: Apply Newton-Schulz5 to each S_t (embarrassingly parallel)
-- Step 4: Sequential memory update: M_t = alpha * M_{t-1} + S'_t

-- NS is the most parallelizable step — each position's NS is independent.
-- For a chunk of size C, this is C independent NS5 calls.
```

## Nonlinear Output: NS as sigma(m)

In the DMGD framework (04_dmgd.md), Newton-Schulz is one choice for the
nonlinear output function sigma:

<!-- HADES: hope_equations/eq-052-nonlinear-outputs (§4.4 Eq 52) -->
```text
-- Nonlinear output momentum (HOPE Eq 52):
W_{t+1} = W_t + sigma(m_{t+1}(u_t))

-- sigma = identity          → standard DMGD
-- sigma = NewtonSchulz_k    → Muon
-- sigma = any nonlinearity  → general nonlinear output
```

## M3: CMS-Aware Multi-Scale Newton-Schulz

The M3 optimizer (Multi-scale Momentum Muon) applies CMS frequency scheduling
to the Newton-Schulz pipeline:

<!-- HADES: hope_equations/eq-075-arch-variant6 (§7.2 Eq 75) -->
```text
-- M3 (HOPE Eq 75):
-- Two-level momentum with CMS scheduling:
M^(1)_t = M^(1)_{t-1} + beta_1 * g_t        -- fast: every step
M^(2)_t = M^(2)_t - beta_2 * sum_{i=t-C}^t g_i   -- slow: every C steps

-- Both momenta pass through Newton-Schulz_T before combining:
-- m_t = Agg(NS_T(M^(1)_t), NS_T(M^(2)_t))
-- Agg = learnable weighted sum (alpha > 0)

-- This is CMS applied to the OPTIMIZER, not just the memory.
-- Fast momentum captures local gradient curvature.
-- Slow momentum captures global gradient trends.
```

## Gradient Derivation (for tape integration)

The backward pass through Newton-Schulz requires differentiating the
iterative process:

```text
-- Forward: O_k = NS_k(G)  (k iterations of the NS recurrence)
-- The iteration is a composition of differentiable matrix operations.

-- Option 1: Unroll through all k iterations (automatic via tape)
--   Pro: exact gradients. Con: k extra backward passes, memory for intermediates.

-- Option 2: Implicit differentiation at convergence
--   At convergence: O^T O = I, so d(O^T O) = 0
--   This gives: dO^T O + O^T dO = 0
--   Solving: dO = -(O O^T)^{-1} dL/dO_{output} = -O @ O^T @ dL/dO
--   Pro: O(1) backward regardless of k. Con: only exact at convergence.

-- Recommendation: Use Option 1 (unrolled) for k=5.
-- Five extra backward matmuls is cheap and avoids convergence assumptions.
-- The tape naturally handles this — each NS iteration is a standard
-- matrix operation that the Wengert tape records.
```

## Implementation Notes

1. **k=5 is the default**: Atlas uses k=5 (NS5). The HOPE M3 optimizer uses
   T iterations (tunable). For f32 inner loop, k=5 provides ~10 digits of
   accuracy — more than sufficient.

2. **Numerical conditioning**: The iteration diverges if the spectral radius
   of G exceeds 1. In practice, normalize G before NS: `G_norm = G / ||G||_F`.
   This ensures convergence and is what both Atlas and Muon do.

3. **Cost-benefit at inner-loop scale**: For inner-loop memory (d=64 typical),
   NS5 costs 15 matmuls of 64x64 matrices ≈ 4M FLOPs. Compare to the d^2
   outer product in the DGD update (4K FLOPs). NS dominates at small d —
   ~1000x more expensive than the base update. This is why Atlas's ablation
   shows NS matters most for quality, not speed.

4. **Batched NS for chunks**: Within a chunk of size C, all C positions need
   independent NS5 calls. These are embarrassingly parallel — batch as a
   single (C, d, d) tensor operation for GPU efficiency.

5. **Connection to 02_m3.md**: The M3 optimizer spec already covers the
   outer-loop multi-scale momentum with NS. This spec focuses on the NS
   iteration itself and its inner-loop application in Atlas.

6. **Opaque VJP vs tape unroll**: For the inner-loop (small d), unrolling
   through k=5 iterations is cheap enough to let the tape handle it directly.
   For the outer-loop (large d), an opaque VJP with implicit differentiation
   (Option 2) may be more memory-efficient. The choice is an implementation
   decision, not a spec constraint.

## Axiom Compliance

- **NL IS #6** (optimizers are associative memory): NS orthogonalizes the memory write, making the update geometrically optimal.
- **NL IS #7** (self-modifying): The NS output depends on the accumulated momentum state, adapting the update direction to gradient history.
- **NL IS #9** (principled not ad hoc): NS is the exact solution to a well-defined orthogonalization objective (Eq 43), not a heuristic.
