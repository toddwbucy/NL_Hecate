# Self-Referential Projections (M_k, M_v, M_q)

<!-- HADES: hope_equations/eq-076-phase1-projections (§8.1 Eq 76, static); hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79, adaptive); hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84, self-modifying) -->
```text
CONTRACT
  Purpose:    Self-referential projections replace static linear projections
              (W_k, W_v, W_q) with adaptive memory modules (M_k, M_v, M_q)
              that themselves update in-context. In standard Titans (Phase 1),
              projections are fixed after outer-loop training — k_t = x_t W_k.
              In Phase 2, each projection becomes its own memory module that
              adapts to the token stream — k_t = M_{k,t-1}(x_t). In Phase 3,
              each memory generates its OWN target values by passing v_t through
              itself — v_hat = M_{square,t-1}(v_t) — making the system fully
              self-modifying. This spec documents the three-phase progression
              and the interface that projection memories must satisfy.
  Expects:    Input token embedding x_t [d, 1], value vector v_t [d, 1].
              For Phase 2+: memory modules M_k, M_v, M_q, M_eta, M_alpha,
              each implementing the same MemoryRule trait as the main memory.
              For Phase 3: self-generated value function per component.
  Guarantees: Phase 1 output identical to standard projections (W @ x).
              Phase 2: adaptive projections update per-token via inner loop.
              Phase 3: self-generated values — each memory decides what to learn.
              All phases compose with any (bias, retention, algorithm) combination.
              The main memory M and projection memories M_square are independent
              MemoryRule instances — they MAY use different rule types.
  Cost:       Phase 1: O(d^2) per projection (standard matmul).
              Phase 2: O(d^2) per projection per token (memory update + read).
              Phase 3: same as Phase 2 plus one extra read per projection
              (self-generated value). Total: 6 memory modules (k, v, q, eta,
              alpha, memory) × cost of one MemoryRule forward.
  Trade-off:  Phase 2+ projections adapt to context (more expressive) but cost
              6× more than static projections. Phase 3 self-modification enables
              the memory to control its own learning, but the recursive structure
              (M updates based on its own output) requires careful initialization
              to avoid degenerate attractors. The HOPE paper (§8.1) recommends
              Phase 1 pre-training followed by Phase 2/3 fine-tuning.
  Position:   specs/algorithms/self_referential/00_interface.md
              Children: 01_self_generated_values.md (Phase 3 detail),
                        02_feature_maps.md (higher-order φ(k)),
                        03_chunkwise_self_ref.md (parallel training)
  Source:     HOPE (2512.24695) §8.1 Eqs 76-89 (three-phase progression)
```

## Phase 1: Static Projections (Standard Titans)

Phase 1 is the conventional approach — projections are fixed outer-loop
parameters that do not adapt during inference:

<!-- HADES: hope_equations/eq-076-phase1-projections (§8.1 Eq 76) -->
```text
-- Phase 1: Static projections (HOPE Eq 76)
k_t = x_t @ W_k                    -- key: fixed linear projection
v_t = x_t @ W_v                    -- value: fixed linear projection
q_t = x_t @ W_q                    -- query: fixed linear projection
eta_t = x_t @ W_eta                -- learning rate: fixed projection
alpha_t = x_t @ W_alpha            -- retention gate: fixed projection

-- W_k, W_v, W_q, W_eta, W_alpha are outer_loop_params.
-- They persist across the forward pass, modified only by outer-loop AD.
-- This is the standard NL_Hecate design (Stages 0-2).
```

<!-- HADES: hope_equations/eq-077-phase1-optimization (§8.1 Eq 77) -->
```text
-- Phase 1: Memory optimization (HOPE Eq 77)
min_M L(M; k_t, v_t) with an optimization algorithm

-- Only the main memory M updates per-token.
-- Projections are frozen outer-loop parameters.
-- The choice of L (attentional bias) and optimizer (algorithm knob)
-- defines the memory update rule — same MIRAS 4-knob framework.
```

<!-- HADES: hope_equations/eq-078-phase1-read (§8.1 Eq 78) -->
```text
-- Phase 1: Memory read (HOPE Eq 78)
y_t = M_t @ q_t                    -- standard associative memory retrieval

-- Output is the main memory applied to the query.
-- In matrix memory: y = M @ q (matmul).
-- In MLP memory: y = MLP(q) (forward through stored weights).
```

## Phase 2: Adaptive Projections

Phase 2 replaces each static projection W with its own adaptive memory module
M_square that updates in-context:

<!-- HADES: hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79) -->
```text
-- Phase 2: Adaptive projections (HOPE Eq 79)
k_t = M_{k,t-1}(x_t)              -- key: adaptive memory module
v_t = M_{v,t-1}(x_t)              -- value: adaptive memory module
q_t = M_{q,t-1}(x_t)              -- query: adaptive memory module
eta_t = M_{eta,t-1}(x_t)          -- learning rate: adaptive memory module
alpha_t = M_{alpha,t-1}(x_t)      -- retention gate: adaptive memory module

-- Each M_{square} is a full MemoryRule instance with its own state.
-- They update per-token, just like the main memory.
-- The subscript t-1 means: read from state BEFORE this token's update.
-- This is observe-then-advance (CS-32) applied to projection memories.
```

<!-- HADES: hope_equations/eq-080-phase2-component-optimization (§8.1 Eq 80) -->
```text
-- Phase 2: Component memory optimization (HOPE Eq 80)
min_{M_square} L(M_square; square_t, v_t)
    for square in {k, v, q, eta, alpha}

-- Key design choice: all component memories share the SAME target value v_t.
-- The component's own output (e.g., k_t for M_k) serves as the key.
-- The shared value v_t serves as the target for all projection memories.
-- This means: "given the current input, every projection should learn to
-- produce something that helps reconstruct v_t."
```

<!-- HADES: hope_equations/eq-081-phase2-memory-optimization (§8.1 Eq 81) -->
```text
-- Phase 2: Main memory optimization (HOPE Eq 81)
min_{M_mem} L(M_mem; k_t, v_t)

-- The main memory uses the (now adaptive) k_t and v_t from Phase 2 projections.
-- k_t = M_{k,t-1}(x_t) instead of x_t @ W_k.
-- The adaptive projections feed richer, context-dependent keys and values
-- into the main memory update.
```

<!-- HADES: hope_equations/eq-082-phase2-read (§8.1 Eq 82) -->
```text
-- Phase 2: Adaptive memory read (HOPE Eq 82)
y_t = M_{mem,t}(q_t)               -- query is now adaptive: q_t = M_{q,t-1}(x_t)

-- Initial states M_{square,0} for ALL memories are meta-learned across sequences.
-- They are outer_loop_params, just like the current W_K_mem, W_V_mem, W_Q_mem.
-- The memory's initial state IS its learned projection matrix (at t=0, M_k
-- acts exactly like W_k before any in-context adaptation).
```

## Phase 3: Self-Modifying Memory

Phase 3 is the most expressive configuration — each memory generates its OWN
target values by passing v_t through itself:

<!-- HADES: hope_equations/eq-083-phase3-self-modifying (§8.1 Eq 83) -->
```text
-- Phase 3: Self-modifying forward pass (HOPE Eq 83)
y_t = M_{mem,t-1}(q_t)
k_t = M_{k,t-1}(x_t)
v_t = M_{v,t-1}(x_t)
eta_t = M_{eta,t-1}(x_t)
alpha_t = M_{alpha,t-1}(x_t)

-- Same structure as Phase 2 (Eq 79), but with a critical difference:
-- q_t = x_t @ W_q remains the ONLY non-adaptive projection.
-- HOPE §8.1 notes this is a design choice — the query projection
-- is kept static to anchor the read operation.
-- All other projections are adaptive memories.
```

<!-- HADES: hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84) -->
```text
-- Phase 3: Self-generated values (HOPE Eq 84)
v_hat_{square,t} = M_{square,t-1}(v_t)
    for square in {k, v, q, eta, alpha, memory}

-- Each memory generates its OWN target values.
-- v_hat_{k,t} = M_{k,t-1}(v_t): the key memory decides what "value" means for keys.
-- v_hat_{mem,t} = M_{mem,t-1}(v_t): the main memory decides its own learning target.
-- This is the KEY self-modification mechanism:
--   the memory decides what to learn.
-- Phase 2 shares a single v_t across all components.
-- Phase 3 gives each component its own perspective on v_t.
```

<!-- HADES: hope_equations/eq-085-phase3-optimization (§8.1 Eq 85) -->
```text
-- Phase 3: Self-modifying optimization (HOPE Eq 85)
min_{M_square} L(M_square; k_t, v_hat_{square,t})
    for square in {k, v, q, eta, alpha, memory}

-- Note: "memory" is now INCLUDED in the set of updatable components.
-- Phase 2 had 5 component memories (k, v, q, eta, alpha) + 1 main memory.
-- Phase 3 unifies them: all 6 are updated with the same rule, each using
-- its own self-generated value v_hat_{square,t}.

-- The optimization target for each component is:
--   key = k_t (shared input-derived key for all memories)
--   value = v_hat_{square,t} (self-generated, unique per component)
```

## Practical Form: DGD Update Rule

HOPE §8.1 gives a concrete instantiation using DGD (Delta Gradient Descent
with weight decay) for matrix memory:

<!-- HADES: hope_equations/eq-088-practical-dgd-update (§8.1 Eq 88) -->
```text
-- Practical: DGD update for all component memories (HOPE Eq 88)
FUNCTION: self_ref_update(M_square: &mut Tensor, k: &Tensor,
                           v_hat: &Tensor, alpha_t: f32, eta_t: f32) -> ()
  -- M_square: any component memory [d, d]
  -- k: key vector [d, 1]
  -- v_hat: self-generated value [d, 1] (Phase 3) or shared v_t (Phase 2)

  -- Retention + gradient update (DGD = Delta rule with decay)
  M_square = M_square @ (alpha_t * I - eta_t * k @ k^T)
             - eta_t * grad_L(M_square; k, v_hat)

  -- Expansion of M @ (alpha * I - eta * k k^T):
  --   alpha * M - eta * (M @ k) @ k^T
  --   = alpha * M - eta * retrieval @ k^T
  --   This is: retention (alpha * M) - learning rate × outer product of
  --   retrieval error with key.
  --
  -- This is Eq 88 applied to EVERY component memory, including "memory" itself.
  -- The update rule is the same for all 6 components — only v_hat differs.
```

<!-- HADES: hope_equations/eq-089-practical-mlp-architecture (§8.1 Eq 89) -->
```text
-- Practical: 2-layer MLP memory architecture (HOPE Eq 89)
M_square(x) = x + W_{square,1} @ sigma(W_{square,2} @ x)

-- Residual MLP: input + learned transformation.
-- sigma: activation function (ReLU, GELU, etc.)
-- W_{square,1}: [d, d_hidden], W_{square,2}: [d_hidden, d]
--
-- This is the default architecture for ALL memories in the practical form.
-- NOT forced: different components MAY use different architectures.
-- The MLP memory type already exists as a MIRAS memory structure choice
-- (MONETA/YAAD/MEMORA use 2-layer MLP).
```

## Per-Token Forward Pass (Phase 3)

The complete forward pass for one token at Phase 3:

<!-- HADES: Derived from hope_equations/eq-083-phase3-self-modifying through eq-088-practical-dgd-update (§8.1 Eqs 83-88), composed forward pass -->
```text
FUNCTION: self_ref_forward(x_t: &Tensor,
                            M_k: &mut Memory, M_v: &mut Memory,
                            M_q: &mut Memory, M_eta: &mut Memory,
                            M_alpha: &mut Memory, M_mem: &mut Memory,
                            W_q: &Tensor) -> Tensor
  -- Step 1: Adaptive projections (Eq 83)
  k_t = M_k.read(x_t)               -- M_{k,t-1}(x_t)
  v_t = M_v.read(x_t)               -- M_{v,t-1}(x_t)
  q_t = x_t @ W_q                   -- static query (design choice)
  eta_t = sigmoid(M_eta.read(x_t))  -- learning rate gate
  alpha_t = sigmoid(M_alpha.read(x_t)) -- retention gate

  -- Step 2: Main memory read (Eq 83)
  y_t = M_mem.read(q_t)             -- M_{mem,t-1}(q_t)

  -- Step 3: Self-generated values (Eq 84)
  v_hat_k = M_k.read(v_t)           -- M_{k,t-1}(v_t)
  v_hat_v = M_v.read(v_t)           -- M_{v,t-1}(v_t)
  v_hat_q = M_q.read(v_t)           -- M_{q,t-1}(v_t)
  v_hat_eta = M_eta.read(v_t)       -- M_{eta,t-1}(v_t)
  v_hat_alpha = M_alpha.read(v_t)   -- M_{alpha,t-1}(v_t)
  v_hat_mem = M_mem.read(v_t)       -- M_{mem,t-1}(v_t)

  -- Step 4: Update all memories (Eq 85 + 88)
  M_k.update(k_t, v_hat_k, alpha_t, eta_t)
  M_v.update(k_t, v_hat_v, alpha_t, eta_t)
  M_q.update(k_t, v_hat_q, alpha_t, eta_t)
  M_eta.update(k_t, v_hat_eta, alpha_t, eta_t)
  M_alpha.update(k_t, v_hat_alpha, alpha_t, eta_t)
  M_mem.update(k_t, v_hat_mem, alpha_t, eta_t)

  RETURN y_t

  -- Ordering: Steps 1-3 read from t-1 state (observe).
  --           Step 4 advances to t state (advance).
  --           This is observe-then-advance (CS-32).
  --
  -- Gate sharing: eta_t and alpha_t are shared across all 6 memories.
  -- Alternative: each memory has its own gates (12 gate memories total).
  -- The paper uses shared gates for simplicity.
```

## Relationship Between Phases

The three phases form a strict generalization hierarchy:

<!-- HADES: Derived from hope_equations/eq-076-phase1-projections through eq-085-phase3-optimization (§8.1 Eqs 76-85), phase relationship -->
```text
-- Phase 1 → Phase 2: Replace W_square with M_square
--   W_k → M_k (initialize M_k = W_k, then allow adaptation)
--   The initial state M_{k,0} IS the Phase 1 projection matrix W_k.
--   Before any tokens are processed, Phase 2 produces identical output.
--   After processing tokens, M_k has adapted to context.

-- Phase 2 → Phase 3: Replace shared v_t with self-generated v_hat
--   Phase 2: all memories learn toward shared v_t
--   Phase 3: each memory learns toward its own v_hat_{square,t}
--   Self-generated values emerge from: v_hat = M_{square,t-1}(v_t)
--   This is one extra M.read() call per component per token.

-- Phase 1 ⊂ Phase 2 ⊂ Phase 3:
--   Phase 1 is Phase 2 with eta_t = 0 for projection memories (no adaptation).
--   Phase 2 is Phase 3 with v_hat = v_t for all components (shared values).
--   Each phase strictly generalizes the previous one.

-- Training progression (HOPE §8.1 recommendation):
--   1. Pre-train at Phase 1 (standard Titans — cheap, stable)
--   2. Fine-tune at Phase 2 (adaptive projections — moderate cost)
--   3. Optionally advance to Phase 3 (self-modifying — full expressiveness)
--   The outer-loop params (M_{square,0}) transfer between phases.
```

## Gradient Flow Through Self-Referential Projections

The outer-loop gradient must flow through all 6 memory modules:

<!-- HADES: Derived from hope_equations/eq-085-phase3-optimization (§8.1 Eq 85), gradient flow analysis -->
```text
-- Given: dL/dy_t (upstream gradient from outer loop)

-- Through main memory read: y_t = M_{mem,t-1}(q_t)
dL/dq_t = M_mem.read_backward(dL/dy_t)
dL/dM_mem (through read) = outer_product(dL/dy_t, q_t)

-- Through static query: q_t = x_t @ W_q
dL/dx_t (through q) = dL/dq_t @ W_q^T
dL/dW_q = x_t^T @ dL/dq_t

-- Through adaptive projections: k_t = M_k(x_t), v_t = M_v(x_t)
-- These are opaque VJP blocks on the tape — each memory's backward
-- propagates gradients through its own update rule.
dL/dM_k = ... (through all downstream uses of k_t)
dL/dM_v = ... (through all downstream uses of v_t)

-- Through self-generated values: v_hat = M_square(v_t)
-- The gradient flows BACK through the same memory that generated v_hat.
-- This creates a feedback loop: M_square's parameters affect BOTH the
-- projection (Step 1) AND the target (Step 3).
-- The Wengert tape handles this naturally — it records both reads
-- as separate operations and replays them in reverse order.

-- Total gradient accumulation per memory:
-- dL/dM_k = dL/dM_k (through k_t projection)
--         + dL/dM_k (through v_hat_k self-generated value)
--         + dL/dM_k (through memory update step)
-- All three paths are summed by the tape's gradient accumulation.
```

## State Lifetime Analysis

<!-- HADES: Derived from hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79), state lifetime classification -->
```text
-- outer_loop_param (persists across build, modified by AD):
--   W_q: static query projection (only non-adaptive projection in Phase 3)
--   M_{square,0}: initial states for all 6 memory modules
--     These ARE the learned projection matrices at t=0.
--     At Phase 1, these are exactly W_k, W_v, W_q, W_eta, W_alpha.
--     At Phase 2+, they serve as initialization for adaptive memories.

-- inner_loop_state (scoped to forward pass, NOT serialized):
--   M_{square,t}: current state of each memory module at time t
--     Updated per-token by the memory's own update rule.
--     NOT checkpointed — reconstructed from M_{square,0} on each forward pass.
--     For matrix memory: M is [d, d].
--     For MLP memory: M is the pair (W_1, W_2).

-- context_memory (persists across forward calls):
--   In serving mode (ContextStream attached), M_{square,t} at chunk boundary
--   becomes M_{square,0} for the next chunk. This is the same context_memory
--   pattern used by the main memory — no special handling needed.
```

## Implementation Notes

1. **Phase 1 already implemented**: The current NL_Hecate codebase (Stages 0-2)
   is Phase 1. Projections W_K_mem, W_V_mem, W_Q_mem in `MemoryLevelParams`
   are static outer-loop parameters. This spec documents the Phase 2/3
   extensions that replace them with adaptive memory modules.

2. **MemoryRule reuse**: Each projection memory M_square is a standard
   MemoryRule instance. It can use any of the 8 named variants (Titans LMM,
   Delta Rule, Hebbian, MONETA, etc.). The projection memories do NOT need to
   use the same rule as the main memory — heterogeneous configurations are
   valid. The practical form (Eq 89) defaults to 2-layer MLP (MONETA-like),
   but this is a design choice, not a constraint.

3. **Cost multiplication**: Phase 2/3 multiplies memory cost by 6 (one main +
   five projection memories). For matrix memory [d, d], this is 6 × O(d^2) per
   token. For MLP memory, it is 6 × the MLP forward+backward cost. The HOPE
   paper acknowledges this cost and recommends Phase 1 pre-training to amortize
   it — Phase 2/3 fine-tuning uses fewer tokens.

4. **Initialization from Phase 1**: When transitioning from Phase 1 to Phase 2,
   set M_{k,0} = W_k (the Phase 1 projection matrix). This ensures continuity:
   at t=0, Phase 2 produces the same output as Phase 1. The outer-loop
   optimizer then fine-tunes M_{square,0} to support in-context adaptation.

5. **Static query design choice**: Phase 3 keeps q_t = x_t @ W_q as the only
   non-adaptive projection. This is a HOPE design choice, not a fundamental
   constraint. The rationale: the query anchors the read operation — if the
   query itself is self-modifying, the memory read becomes doubly recursive,
   which can be unstable. However, the interface allows making W_q adaptive
   if desired (it is already in the Phase 2 formulation, Eq 79).

6. **Interaction with CMS**: Each CMS level has its own set of 6 memory
   modules. At k=4 CMS levels, Phase 3 requires 24 memory modules total
   (6 per level × 4 levels). The Conductor's Pulse determines which levels
   are active, and only active levels update their projection memories.
   Frozen levels still read from their projection memories (observe) but
   do not update them (no advance).

7. **Pluggable dispatch**: Self-referential projections register as a
   `ProjectionKind::Adaptive(rule)` variant in the S3b-M3 infrastructure,
   alongside the existing `ProjectionKind::Static` (Phase 1). The rule
   parameter specifies which MemoryRule type the projection memory uses.

8. **Phase-2 adaptive CMS aggregation**: The same Phase-1 → Phase-2 progression
   applies to CMS level aggregation weights. Phase 1: learnable `alpha_l` scalars
   (outer_loop_param, softmax-normalized) replace the prior ad hoc `1/sqrt(k)`
   normalization. Phase 2: `alpha_l` becomes an adaptive memory `M_agg(x_t)` that
   produces context-dependent level weights — a sentence about long-range context
   weights Level 3 higher, local patterns weight Level 0 higher. This reuses the
   same `ProjectionKind::Adaptive(rule)` infrastructure. See
   `composition_patterns/04_hope.md` Variant 2/5 aggregation and task_44105a.

## Axiom Compliance

- **NL IS #4** (compressing context): Adaptive projections compress context into the projection itself — the key/value/query transformations adapt to what has been seen, not just what was pre-trained.
- **NL IS #6** (optimizers are associative memory): Projection memories ARE associative memories — they map input tokens to keys/values/queries via the same inner-loop optimization that drives the main memory.
- **NL IS #9** (principled not ad hoc): The three-phase progression derives from systematically replacing each static component with an adaptive one, following the HOPE paper's formal framework. Phase 2 is not "let's make projections learnable" — it is the natural next step when the memory framework is applied recursively to its own inputs.
- **MIRAS IS #1** (orthogonal design choices): Projection memories are independent MemoryRule instances. Their choice of structure, bias, retention, and algorithm is orthogonal to the main memory's choices. The self-referential extension adds a new dimension (projection adaptivity) without constraining the existing four knobs.
