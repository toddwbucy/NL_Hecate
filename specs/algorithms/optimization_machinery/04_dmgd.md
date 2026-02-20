# Deep Momentum Gradient Descent (DMGD)

```text
CONTRACT
  Purpose:    DMGD upgrades the momentum accumulator from a linear EMA to a
              state-dependent memory. Standard momentum compresses gradients via
              a dot-product objective (Hebbian). DMGD replaces this with L2
              regression (Delta Momentum) or an MLP (Deep Momentum), making the
              momentum sensitive to WHAT it has already accumulated. This is to
              momentum what DGD is to the memory update — the same dot-product →
              L2 upgrade, applied one level up in the optimization hierarchy.
  Expects:    Gradient signal g_t, momentum state S (matrix or MLP), gates
              (alpha, eta), optional preconditioner P.
  Guarantees: State-dependent gradient compression. Gradient-dependent weight
              decay emerges naturally from the L2 objective in Delta Momentum.
              Deep Momentum (MLP) enables nonlinear gradient routing.
  Cost:       Delta Momentum: O(d^2) per step (one extra matmul S @ g for
              state-dependent term). Deep Momentum: O(d * h) where h = hidden
              dim of MLP. Both negligible next to the outer-loop backward pass.
  Trade-off:  More expressive momentum = better gradient compression across
              correlated sequences, but cannot use the linear scan trick for
              momentum parallelization (nonlinear in S).
  Position:   specs/algorithms/optimization_machinery/04_dmgd.md
              Sibling of 01_momentum.md (general framework), 03_dgd.md (DGD for memory)
  Source:     HOPE (2512.24695) §4.2–4.4, Eqs 33–53; Appendix B Eqs 100–111
```

## The Problem DMGD Solves

Standard EMA momentum uses a dot-product (Hebbian) objective:

<!-- HADES: hope_equations/eq-034-momentum-hebbian-obj (§4.2 Eq 34) -->
```text
-- Standard momentum objective (HOPE Eq 34):
L_momentum(m; g_t) = -<m, g_t>

-- Closed-form solution: EMA
m_{t+1} = alpha * m_t - eta * g_t
```

This is **independent of m_t** in the same way GD is independent of M_t.
The momentum adds the same gradient contribution regardless of what gradients
it has already accumulated. For correlated gradient sequences (common in
language modeling), this wastes momentum capacity by re-encoding redundant
gradient information.

## Delta Momentum: The L2 Upgrade

Replace the Hebbian objective with L2 regression for the momentum memory:

<!-- HADES: hope_equations/eq-046-expressive-association (§4.4 Eq 46) -->
```text
-- Delta Momentum objective (HOPE Eq 46):
L_momentum(m; g_t, P_t) = ||m @ g_t - P_t||^2_2

-- P_t is a preconditioner (maps gradients to global properties of past data)
-- When P_t = I, this reduces to: ||m @ g_t - I||^2
```

The gradient of this objective **depends on m_t**:

<!-- HADES: hope_equations/eq-049-l2-momentum-update (§4.4 Eq 49) -->
```text
-- Delta Momentum update (HOPE Eq 49):
m_{t+1} = m_t * (alpha_{t+1} - g_t^T @ g_t) - eta_t * P_t @ g_t

-- The (alpha - g^T g) term: gradient-dependent weight decay
-- When g_t aligns with accumulated momentum, decay is stronger
-- When g_t is orthogonal, momentum is preserved
```

Combined with the weight update:

<!-- HADES: hope_equations/eq-048-l2-momentum-weight (§4.4 Eq 48) -->
```text
-- Weight update with Delta Momentum (HOPE Eqs 48-49):
FUNCTION: delta_momentum_step(W: &mut Tensor, m: &mut Tensor,
                               g: &Tensor, P: &Tensor,
                               alpha_t: f32, eta_t: f32) -> ()
  -- g: gradient of outer loss w.r.t. W (= delta_ell @ h_hat^T in paper)
  -- P: preconditioner (task-to-global mapping)

  -- State-dependent momentum update
  decay = alpha_t - dot(g, g)           -- gradient-dependent forgetting
  m = m * decay - eta_t * P @ g         -- Delta Momentum recurrence

  -- Apply momentum to weights
  W = W + m
```

## Deep Momentum: MLP Memory

DMGD goes further — replace the linear momentum memory with an MLP:

<!-- HADES: hope_equations/eq-050-dmgd (§4.4 Eq 50) -->
```text
-- DMGD (HOPE Eq 50):
W_{t+1} = W_t + m_{t+1}(u_t)

m_{t+1} = alpha_{t+1} * m_t - eta_t * nabla L^(2)(m_t; u_t, 1)

-- m is now a 2-layer MLP, not a matrix
-- u_t = gradient input to the MLP
-- L^(2) = inner objective for the deep memory
-- m_{t+1}(u_t) means: evaluate the MLP on the gradient
```

The MLP momentum can learn nonlinear gradient routing — it decides which
gradient components to amplify, suppress, or combine based on learned
transformations, not just linear accumulation.

## Extensions: Feature Maps and Nonlinear Outputs

Two additional knobs enhance DMGD:

### Higher-Order Feature Maps

<!-- HADES: hope_equations/eq-051-higher-order-features (§4.4 Eq 51) -->
```text
-- Feature-mapped momentum (HOPE Eq 51):
m_{t+1} = alpha * m_t - eta * P_t @ phi(g_t)

-- phi: learned feature map applied to gradient keys
-- Enhances the capacity of momentum as associative memory
-- phi may itself be learned via its own internal objective
```

### Nonlinear Outputs (Newton-Schulz → Muon)

<!-- HADES: hope_equations/eq-052-nonlinear-outputs (§4.4 Eq 52) -->
```text
-- Nonlinear output momentum (HOPE Eq 52):
W_{t+1} = W_t + sigma(m_{t+1}(u_t))

m_{t+1} = alpha * m_t - eta * nabla L^(2)(m_t; u_t, I)

-- sigma = nonlinear output function
-- sigma = NewtonSchulz_k → Muon optimizer (Jordan et al. 2024)
-- sigma = identity → standard DMGD
```

## The Momentum Expressiveness Hierarchy

<!-- HADES: hope_equations/eq-033-ema-momentum through eq-052-nonlinear-outputs (§4.2–4.4) -->
```text
| Level | Momentum Type        | Objective     | State-Dep | Cost     | HOPE Eq |
|-------|---------------------|---------------|-----------|----------|---------|
| 0     | None (plain GD)     | —             | —         | O(1)     | Eq 32   |
| 1     | EMA                 | Dot-product   | No        | O(d)     | Eq 33   |
| 2     | Delta Momentum      | L2 regression | Yes       | O(d^2)   | Eq 49   |
| 3     | Deep Momentum (MLP) | L2 (deep)     | Yes       | O(d*h)   | Eq 50   |
| 3+    | + Feature maps      | L2 + phi(g)   | Yes       | O(d*h+f) | Eq 51   |
| 3++   | + Nonlinear output  | L2 + sigma(m) | Yes       | O(d*h+s) | Eq 52   |
```

Each level adds capacity for compressing gradient history. The upgrade path
mirrors the memory rule upgrade: Hebbian → Delta → Titans (MLP).

## Interaction with DGD

DMGD and DGD operate at different levels:

```text
-- DGD: upgrades the MEMORY update (inner loop)
--   M_{t+1} = (1-alpha) * M - theta * (M@k - v) @ k^T
--   The M@k term makes it state-dependent

-- DMGD: upgrades the MOMENTUM accumulator (between inner and outer loops)
--   S_{t+1} = S * (alpha - g^T g) - eta * P @ g
--   The S * (alpha - g^T g) term makes it state-dependent

-- Full Hope uses BOTH:
--   DGD for the memory rule + DMGD for the momentum
--   Double state-dependence at two levels of the optimization hierarchy
```

When DGD composes with Delta Momentum (the combination used in practice):

```text
FUNCTION: dgd_with_delta_momentum(M: &mut Tensor, S: &mut Tensor,
                                   k: &Tensor, v: &Tensor,
                                   alpha_t: f32, theta_t: f32,
                                   mu_t: f32, eta_t: f32) -> ()
  -- DGD gradient (state-dependent)
  error = M @ k - v
  grad = error @ k^T                    -- [d, d] outer product

  -- Delta Momentum update (state-dependent)
  S = S * (mu_t - dot(grad, grad)) - eta_t * grad

  -- Apply to memory with retention
  M = (1 - alpha_t) * M - S
```

## Gradient Derivation (for tape integration)

The backward pass through Delta Momentum requires gradients w.r.t. the
outer-loop parameters that produce g, P, alpha, eta.

<!-- HADES: Derived from hope_equations/eq-049-l2-momentum-update (§4.4 Eq 49), analytical VJP -->
```text
-- Forward: S_{t+1} = S_t * (alpha_t - g_t^T g_t) - eta_t * P_t @ g_t
-- Let D_t = alpha_t - g_t^T g_t (the gradient-dependent decay scalar)

-- Given: dL/dS_{t+1} (upstream gradient)
-- Need: dL/dS_t, dL/dg_t, dL/dalpha_t, dL/deta_t

dL/dS_t = D_t * dL/dS_{t+1}
         -- State flows through the decay term

dL/dg_t = -2 * (g_t^T @ S_t) * dL/dS_{t+1}
         - eta_t * P_t^T @ dL/dS_{t+1}
         -- Two terms: one through D_t (decay), one through the write

dL/dalpha_t = trace(S_t^T @ dL/dS_{t+1})
             -- Scalar: how retention gate affects loss

dL/deta_t = -trace((P_t @ g_t)^T @ dL/dS_{t+1})
           -- Scalar: how learning rate gate affects loss
```

For Deep Momentum (MLP), the backward pass uses standard backpropagation
through the MLP — the tape handles this automatically via its normal
chain rule mechanism. No opaque VJP needed for the MLP itself.

## Parallelization

Delta Momentum is nonlinear in S (the `S * (alpha - g^T g)` term), so
the linear scan trick from Titans Eq 18 is inapplicable. Use the same
chunkwise strategy as DGD:

<!-- HADES: hope_equations/eq-090-chunk-wise-update (§8.2 Eq 90, applied to momentum) -->
```text
-- Split sequence into chunks of size C
-- For each chunk:
--   1. Freeze S at chunk boundary (S_chunk_start)
--   2. Compute all g, P, gates in parallel
--   3. Sequential Delta Momentum recurrence within chunk
--   4. Pass final S to next chunk
```

When standard EMA momentum IS used (Level 1), the linear scan from
Titans Eq 18 remains valid — it's only the L2/Delta/Deep variants that
require chunkwise.

## Ablation Evidence

The HOPE paper (Table 6) shows momentum's impact:

<!-- HADES: HOPE (2512.24695) Table 6 ablation study -->
```text
| Configuration          | Perplexity | Delta vs Full |
|------------------------|-----------|---------------|
| Hope (full, with DMGD) | 12.24     | baseline      |
| w/o momentum           | 13.58     | +1.34 ppl     |
```

Removing momentum costs +1.34 ppl — the second largest single-component
impact after weight decay (+1.47 ppl) and larger than DGD (+1.17 ppl).

## Implementation Notes

1. **Delta Momentum vs Deep Momentum**: Start with Delta Momentum (Level 2).
   It's a matrix recurrence like DGD and reuses the same infrastructure.
   Deep Momentum (Level 3, MLP) requires the MLP memory infrastructure
   from MONETA/YAAD/MEMORA rules — implement after those are stable.

2. **Gradient-dependent decay**: The `(alpha - g^T g)` term in Eq 49 can
   go negative if `||g||^2 > alpha`. In practice this is rare with proper
   gate initialization (b_alpha=3.0 → sigmoid ≈ 0.95), but clamp the
   effective decay to [eps, 1-eps] as a safety measure (CS-39).

3. **Preconditioner P**: When P = I (identity), Delta Momentum simplifies
   to a direct L2 regression on gradients. When P captures second-order
   information, it becomes a preconditioned method. The choice of P is an
   orthogonal design decision — spec for AdaMuon (S3b-S17) covers this.

4. **Cost**: Delta Momentum adds one matrix-matrix product (`S * decay`)
   and one matrix-vector product (`P @ g`) per step. For d=2048, this is
   ~8M FLOPs — negligible next to the forward/backward pass.

5. **Existing code**: The Titans momentum in `core/src/titans.rs` already
   implements EMA momentum (Level 1). Delta Momentum extends this with
   the state-dependent decay term. The recurrence structure is identical
   to DGD — same loop, different operands.

## Axiom Compliance

- **NL IS #4** (compressing context): Compresses gradient history with state awareness.
- **NL IS #6** (optimizers are associative memory): Makes explicit that momentum IS memory.
- **NL IS #7** (self-modifying): The S-dependent decay means momentum adapts based on its own state.
- **NL IS #10** (nested optimization): A nested optimizer — optimizes within the optimizer.
