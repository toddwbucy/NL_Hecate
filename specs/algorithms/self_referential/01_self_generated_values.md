# Self-Generated Values (v_hat = M_square(v))

<!-- HADES: hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84); hope_equations/eq-085-phase3-optimization (§8.1 Eq 85) -->
```text
CONTRACT
  Purpose:    Self-generated values are the mechanism by which Phase 3
              self-referential memories become truly self-modifying. In Phase 2,
              all component memories share a single target value v_t — they all
              learn toward the same thing. In Phase 3, each memory generates its
              OWN target by passing v_t through itself: v_hat = M_{square,t-1}(v_t).
              This means the memory decides what to learn. The key memory M_k
              produces a key-space target, the gate memory M_eta produces a
              gate-space target, and the main memory M_mem produces its own
              learning target — each specializes v_t through its own learned
              transformation.
  Expects:    Value vector v_t [d, 1] (from M_v or static projection).
              Memory modules M_square each implementing MemoryRule trait.
              Each M_square must support a read operation: M.read(input) -> output.
  Guarantees: v_hat_{square,t} = M_{square,t-1}(v_t) for each component.
              Self-generated values are computed BEFORE the memory update
              (observe-then-advance, CS-32). The gradient flows back through
              the same memory that generated v_hat, creating a feedback loop
              that the Wengert tape handles via two separate recorded reads.
  Cost:       One extra M.read() per component per token. For matrix memory:
              O(d^2) per component (one matmul). For MLP memory: one forward
              pass per component. Total: 6 extra reads per token at Phase 3
              (vs 0 extra at Phase 2).
  Trade-off:  Self-generated values allow each memory to specialize its learning
              target, producing richer representations. But the recursive
              feedback (memory output determines memory target) can amplify
              initialization sensitivity — a poorly initialized memory may
              generate degenerate targets that reinforce themselves. The HOPE
              paper mitigates this with Phase 1 → Phase 2 → Phase 3 progressive
              training.
  Position:   specs/algorithms/self_referential/01_self_generated_values.md
              Parent: 00_interface.md (three-phase framework)
              Sibling of: 02_feature_maps.md (higher-order phi(k))
  Source:     HOPE (2512.24695) §8.1 Eqs 84-85, 87 (self-generated values)
```

## Phase 2 vs Phase 3: Shared vs Self-Generated

The transition from Phase 2 to Phase 3 is a single change in how target
values are constructed:

<!-- HADES: hope_equations/eq-080-phase2-component-optimization (§8.1 Eq 80, shared values) -->
```text
-- Phase 2: Shared values (HOPE Eq 80)
min_{M_square} L(M_square; square_t, v_t)
    for square in {k, v, q, eta, alpha}

-- All 5 component memories optimize toward the SAME target v_t.
-- v_t comes from the value projection (M_v(x_t) in Phase 2, or x_t @ W_v in Phase 1).
-- Every memory learns: "given my output as key, reconstruct v_t."
--
-- This is a reasonable default: v_t carries the input's semantic content,
-- and all projections should be aligned with that content.
-- But it forces all memories to live in the same value space.
```

<!-- HADES: hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84) -->
```text
-- Phase 3: Self-generated values (HOPE Eq 84)
v_hat_{square,t} = M_{square,t-1}(v_t)
    for square in {k, v, q, eta, alpha, memory}

-- Each memory applies ITSELF to v_t to produce its own target.
-- v_hat_{k,t} = M_{k,t-1}(v_t): the key memory transforms v_t into key-space.
-- v_hat_{eta,t} = M_{eta,t-1}(v_t): the gate memory transforms v_t into gate-space.
-- v_hat_{mem,t} = M_{mem,t-1}(v_t): the main memory transforms v_t into its own space.
--
-- Each memory now lives in its OWN value space.
-- The transformation M_{square}(v_t) is learned — it adapts in-context just
-- like the projection itself.
```

<!-- HADES: hope_equations/eq-085-phase3-optimization (§8.1 Eq 85) -->
```text
-- Phase 3: Optimization with self-generated values (HOPE Eq 85)
min_{M_square} L(M_square; k_t, v_hat_{square,t})
    for square in {k, v, q, eta, alpha, memory}

-- The optimization target changes from v_t to v_hat_{square,t}.
-- Everything else stays the same: same key k_t, same loss L, same algorithm.
-- The 4-knob MIRAS framework applies unchanged — only the target value differs.
--
-- Note: "memory" is now in the set (6 components, not 5).
-- Phase 2 treats the main memory separately (Eq 81).
-- Phase 3 unifies all components under the same update rule.
```

## The Self-Modification Feedback Loop

Self-generated values create a feedback loop where the memory's output
at time t-1 determines its learning target at time t:

<!-- HADES: Derived from hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84), feedback analysis -->
```text
-- The feedback loop for component M_square:
--
-- Time t:
--   1. Read: v_hat_{square,t} = M_{square,t-1}(v_t)     -- generate target
--   2. Loss: L_t = L(M_{square,t-1}; k_t, v_hat_{square,t})  -- compute error
--   3. Update: M_{square,t} = update(M_{square,t-1}, grad L_t)  -- modify memory
--
-- Time t+1:
--   1. Read: v_hat_{square,t+1} = M_{square,t}(v_{t+1})  -- NEW target from UPDATED memory
--   ...
--
-- The memory at time t influences its own target at time t (through Step 1),
-- which influences the gradient (Step 2), which modifies itself (Step 3),
-- which changes the next target (time t+1, Step 1).
--
-- This is a FIXED-POINT problem:
-- At convergence, M* satisfies: M* is a fixed point of the update rule
-- when using targets generated by M* itself.
--
-- Stability conditions:
-- The feedback is stable when the self-generated target v_hat moves slowly
-- relative to the memory update. This is naturally the case because:
--   1. alpha_t ≈ 0.95 (high retention): M changes slowly per step.
--   2. eta_t ≈ 0.01 (small learning rate): gradient steps are small.
--   3. The read M(v_t) is a smooth function of M's parameters.
-- Together, v_hat_{t+1} ≈ v_hat_t + O(eta_t), so the target drifts slowly.
```

## Gradient Through Self-Generated Values

The gradient must flow through both uses of M_square — the projection read
(Step 1 of the forward pass) and the self-generated value read:

<!-- HADES: Derived from hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84), VJP for self-generated values -->
```text
-- Forward operations for M_square per token:
--   Read 1 (projection): output_square_t = M_{square,t-1}(x_t)   -- Eq 83
--   Read 2 (self-gen):    v_hat_{square,t} = M_{square,t-1}(v_t)  -- Eq 84
--   Update:               M_{square,t} = update(M_{square,t-1}, k_t, v_hat_{square,t})

-- The tape records Read 1 and Read 2 as SEPARATE opaque VJP blocks.
-- Both share the same M_{square,t-1} parameters.

-- Backward (tape replays in reverse):
--   Through Update:
--     dL/dM_{square,t-1} (through update) = standard memory backward
--     dL/dv_hat_{square,t} = from loss gradient
--     dL/dk_t (through update) = from loss gradient

--   Through Read 2 (self-generated value):
--     dL/dM_{square,t-1} (through v_hat) = read_backward(dL/dv_hat_{square,t}, v_t)
--     dL/dv_t (through v_hat) = M_{square,t-1}^T @ dL/dv_hat_{square,t}

--   Through Read 1 (projection):
--     dL/dM_{square,t-1} (through projection) = read_backward(dL/doutput, x_t)
--     dL/dx_t (through projection) = M_{square,t-1}^T @ dL/doutput

-- Total gradient accumulation:
-- dL/dM_{square,t-1} = dL/dM (through update)
--                     + dL/dM (through Read 2, self-gen)
--                     + dL/dM (through Read 1, projection)
-- All three paths sum automatically via the tape's gradient accumulation.

-- For matrix memory M [d, d]:
--   Read backward: dL/dM = dL/doutput @ input^T  (outer product)
--   Read 2 contributes: dL/dv_hat @ v_t^T
--   Read 1 contributes: dL/doutput_square @ x_t^T
--   Both are rank-1 updates to the d×d gradient accumulator.
```

## Properties of Self-Generated Values

<!-- HADES: Derived from hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84), mathematical properties -->
```text
-- Property 1: Specialization
--   Each memory transforms v_t through its own learned mapping.
--   M_k(v_t) projects v_t into the space where keys are meaningful.
--   M_eta(v_t) projects v_t into the space where learning rates are meaningful.
--   M_mem(v_t) projects v_t into the space where the main memory operates.
--   This is AUTOMATIC — no explicit "value head" design is needed.
--   The memory's own parameters define the transformation.

-- Property 2: Phase 2 recovery
--   Phase 3 with identity self-generated values recovers Phase 2:
--   If M_{square,t-1}(v_t) = v_t (identity mapping), then v_hat = v_t.
--   For matrix memory: M = I (identity matrix) gives v_hat = v_t.
--   Phase 2 → Phase 3 transition can be initialized at identity,
--   allowing gradual specialization during fine-tuning.

-- Property 3: Information bottleneck
--   Self-generated values pass v_t through M_{square}'s representational
--   bottleneck. If M is a matrix [d, d] with rank r < d, then v_hat
--   lives in a rank-r subspace. The memory forces its target into
--   its own representational capacity — it cannot learn what it
--   cannot represent.

-- Property 4: Consistency pressure
--   The loss L(M; k_t, v_hat_t) where v_hat = M(v_t) creates a
--   consistency pressure: M should produce outputs that are easy for
--   M itself to reconstruct. This is similar to autoencoder objectives
--   but applied asymmetrically (read ≠ write in general).

-- Property 5: Bounded divergence from v_t
--   At initialization (M_0 from Phase 1 projection):
--     v_hat = M_0(v_t) = W_square @ v_t (linear transform of v_t).
--   As M adapts, v_hat drifts from this initial linear mapping.
--   The retention gate alpha_t bounds the drift rate per token:
--     ||v_hat_{t+1} - v_hat_t|| <= O(eta_t * ||grad||)
--   High retention (alpha ≈ 1) means v_hat changes slowly.
```

## Initialization Strategies

<!-- HADES: Derived from hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84), initialization for stability -->
```text
-- Strategy 1: Identity initialization (safe default)
--   Set M_{square,0} = I (identity matrix) for matrix memory.
--   v_hat = I @ v_t = v_t → recovers Phase 2 behavior at t=0.
--   The memory specializes its targets gradually during fine-tuning.
--   Pro: guaranteed stable start. Con: requires fine-tuning to see benefit.

-- Strategy 2: Phase 1 projection initialization (HOPE recommended)
--   Set M_{square,0} = W_square from Phase 1 pre-training.
--   v_hat = W_square @ v_t → targets are already meaningful projections.
--   Each memory starts by transforming v_t the way it transforms x_t.
--   Pro: warm start from pre-training. Con: may bias toward Phase 1 subspace.

-- Strategy 3: Scaled identity
--   Set M_{square,0} = (1 - eps) * W_square + eps * I.
--   Interpolates between projection and identity.
--   eps = 0.1: mostly Phase 1 behavior with slight identity regularization.
--   Pro: balances warm start with identity fallback.

-- For MLP memory (Eq 89):
--   M(x) = x + W_1 @ sigma(W_2 @ x)
--   Initialize W_1, W_2 small → M(x) ≈ x (near-identity).
--   Self-generated values start near v_t and specialize gradually.
--   The residual connection x + ... provides the identity fallback.
```

## Interaction with Attentional Bias

The self-generated value v_hat replaces v_t in the loss function. This
interacts with the attentional bias choice:

<!-- HADES: Derived from hope_equations/eq-085-phase3-optimization (§8.1 Eq 85), bias interaction -->
```text
-- L2 bias: L = ||M(k_t) - v_hat||^2
--   Memory minimizes squared error to its own self-generated target.
--   v_hat is a point in R^d — standard regression target.

-- L1 bias: L = ||M(k_t) - v_hat||_1
--   Memory stores Sign(v_hat) — the sign pattern of its own target.
--   Self-generated values determine WHICH signs the memory preserves.

-- KL bias: L = KL(p_hat || softmax(M(k_t)))
--   v_hat must be a probability distribution (or softmaxed).
--   p_hat = softmax(M_{square}(v_t) / tau): self-generated distribution target.
--   The memory generates its own predictive distribution to match.

-- General l_p bias: L = ||M(k_t) - v_hat||_p^p
--   Self-generated values determine the error landscape curvature.
--   The p exponent controls sensitivity to large vs small v_hat components.

-- In all cases, the gradient w.r.t. M has an EXTRA term through v_hat:
--   dL/dM = dL/dM (standard, through M(k_t))
--         + dL/dM (through v_hat = M(v_t), via chain rule)
-- The second term is the self-referential contribution.
```

## Implementation Notes

1. **Minimal code change from Phase 2**: The only difference between Phase 2
   and Phase 3 is replacing `v_target = v_t` with `v_target = M.read(v_t)` in
   the per-component update. The rest of the forward pass — projection reads,
   memory update, gradient flow — is identical. This is a one-line change per
   component, guarded by a `self_generated: bool` configuration flag.

2. **Read ordering**: Both reads (projection and self-gen) use M_{t-1} state.
   They MUST happen before the update to M_t. The forward pass pseudocode in
   00_interface.md enforces this via Steps 1-3 (reads) before Step 4 (updates).
   This is observe-then-advance (CS-32) applied to both read operations.

3. **Tape recording**: The tape records two separate opaque reads per component
   per token in Phase 3 (vs one in Phase 2). This doubles the tape entries for
   projection memories but does not change the asymptotic complexity — each
   entry is O(1) to record, and the backward is O(d^2) per read (same as
   the forward read cost).

4. **Degenerate target detection**: If a self-generated v_hat collapses to
   near-zero (||v_hat|| < eps), the loss gradient becomes uninformative.
   Monitor ||v_hat|| per component during training. If collapse occurs, it
   indicates the memory's learned transformation is projecting v_t into its
   null space — a sign of poor initialization or excessive retention decay.

5. **Configuration**: The `ProjectionKind::Adaptive(rule)` variant from
   00_interface.md gains a `self_generated: bool` field to distinguish
   Phase 2 (false) from Phase 3 (true). When false, all components use
   shared `v_t`. When true, each component calls `M.read(v_t)` for its
   own target.

6. **Interaction with CMS**: Self-generated values respect the CMS frequency
   schedule. At frozen CMS levels, projection memories do not update, so their
   self-generated values are static within the frozen window. This is consistent
   — a frozen memory produces the same v_hat for every token until it unfreezes.

## Axiom Compliance

- **NL IS #4** (compressing context): Self-generated values compress v_t through the memory's own bottleneck — each component extracts the aspect of v_t relevant to its function, discarding the rest.
- **NL IS #6** (optimizers are associative memory): The memory IS its own value generator — the optimization target comes from the same associative memory that is being optimized, unifying the "what to learn" and "how to learn" into a single module.
- **NL IS #9** (principled not ad hoc): Self-generated values emerge from the systematic application of "replace every static component with a memory" — Eq 84 is the natural consequence of applying Phase 2's adaptive principle to the target values themselves.
