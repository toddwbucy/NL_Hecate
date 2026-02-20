# HOPE Composition Pattern (Level-Level Composition)

<!-- HADES: hope_equations/eq-070-arch-variant1 (HOPE §6 Eq 70); hope_equations/eq-071-arch-variant2 (HOPE §6 Eq 71); hope_equations/eq-072-arch-variant3 (HOPE §6 Eq 72); hope_equations/eq-073-arch-variant4 (HOPE §6 Eq 73); hope_equations/eq-074-arch-variant5 (HOPE §6 Eq 74); hope_equations/eq-075-arch-variant6 (HOPE §6 Eq 75, M3 optimizer) -->
```text
CONTRACT
  Purpose:    MAC/MAG/MAL define how MEMORY composes with ATTENTION (one level).
              The HOPE composition variants define how CMS LEVELS compose with
              EACH OTHER (multiple levels). This is the next axis of composition:
              given k frequency levels, how does data flow between them?

              HOPE §6 identifies six level-composition variants:
                Variant 1 (Chained):      levels in series, output chains through all
                Variant 2 (Freq-Gated):   levels update at their own frequency, idle otherwise
                Variant 3 (Nested):       higher level meta-learns initial state of lower
                Variant 4 (Sequential):   output of level s feeds input of level s+1
                Variant 5 (Independent):  levels process input independently, outputs aggregated
                Variant 6 (M3):           CMS applied to the optimizer itself

              These are ORTHOGONAL to MAC/MAG/MAL. You pick one memory-attention
              composition (MAC, MAG, or MAL) AND one level-composition variant.
              The existing 02_cms_variants.md provides a brief overview; this spec
              gives the full HOPE equations and their implications.

  Expects:    k CMS frequency levels, each with its own memory module.
              Frequency scheduler (01_frequency_scheduler.md) producing Pulses.
              A memory-attention composition pattern (MAC/MAG/MAL) per level.
  Guarantees: When k=1: all variants reduce to a standard single-level block.
              When k>1: multi-scale behavior emerges from the chosen variant.
              All variants preserve the frequency-gating contract (CS-27/28):
              frozen levels do not receive outer-loop gradient updates.
  Cost:       Variant-dependent. Chained: O(k * block_cost). Independent: O(k * block_cost)
              but parallelizable. Nested: O(k * block_cost + meta-learning overhead).
              All variants amortize slow-level cost via frequency gating.
  Trade-off:  Chained/Sequential: richer inter-level information flow but sequential.
              Independent: parallelizable but no inter-level communication.
              Nested: most expressive but most complex (meta-learning between levels).
              The HOPE paper's own model uses Variant 2 (Freq-Gated) as the base,
              which is also what NL_Hecate implements as the default CMS configuration.
  Position:   specs/algorithms/composition_patterns/04_hope.md
              Sibling of: 01_mac.md, 02_mag.md, 03_mal.md (memory-attention composition)
              Cross-ref: specs/algorithms/frequency_scheduling/02_cms_variants.md (overview)
  Source:     HOPE (2512.24695) §6 Eqs 70-75; §8 Eqs 76-79 (self-referential phases);
              §9 Table 6 (ablation study)
```

## Two Axes of Composition

<!-- HADES: Derived from hope_equations/eq-070-arch-variant1 (HOPE §6); specs/algorithms/composition_patterns/00_interface.md -->
```text
-- NL has TWO independent composition axes:

-- Axis 1: Memory-Attention Composition (MAC/MAG/MAL)
--   Answers: "How does memory interact with attention WITHIN a single level?"
--   MAC: memory feeds attention context
--   MAG: memory gates attention output
--   MAL: memory preprocesses attention input

-- Axis 2: Level-Level Composition (HOPE Variants 1-6)
--   Answers: "How do CMS levels interact with EACH OTHER?"
--   Chained: output flows through all levels serially
--   Freq-Gated: each level updates at its frequency, idle otherwise
--   Nested: slow level meta-learns fast level's initial state
--   Sequential: level s output feeds level s+1
--   Independent: levels process input in parallel, outputs aggregated
--   M3 optimizer: CMS applied to the optimizer (orthogonal to architecture variants 1-5)

-- These are orthogonal. Example configurations:
--   MAG + Freq-Gated (k=4): default NL_Hecate — 4 levels, each MAG,
--     each fires at its own frequency. This is what the Conductor implements.
--   MAC + Independent (k=2): two independent memory modules, each MAC,
--     outputs aggregated via weighted sum.
--   MAL + Nested (k=3): slow level meta-learns initial M for fast level.
```

## Variant 1: Chained (Eq 70)

<!-- HADES: hope_equations/eq-070-arch-variant1 (HOPE §6 Eq 70) -->
```text
-- Chained CMS (HOPE Eq 70):
--   x_t passes through a chain of k frequency-indexed blocks.
--   Each block has chunk size C^(f) := max_t C_t^(f) / f_l.
--
-- FUNCTION: chained_cms_forward(x: &Tensor, levels: &[Level; k], pulse: &Pulse) -> Tensor
--   h = x
--   FOR l in 0..k:
--     IF pulse.is_active(l):
--       h = levels[l].forward(h, pulse)    -- each level transforms the running state
--     ELSE:
--       h = h                               -- frozen levels pass through
--   RETURN h
--
-- Properties:
--   - All levels see a TRANSFORMED input (level l sees level l-1's output)
--   - Information flows from fast → slow: Level 0 processes first, Level 3 last
--   - Sequential: cannot parallelize across levels
--   - The conventional Transformer with k=1, frequency=0 is a special case
--
-- When to use:
--   When slow levels should process ABSTRACT features (post-processed by fast levels).
--   Analogous to a deep network where each "layer" operates at a different timescale.
```

## Variant 2: Frequency-Gated (Eq 71) — DEFAULT

<!-- HADES: hope_equations/eq-071-arch-variant2 (HOPE §6 Eq 71) -->
```text
-- Frequency-Gated CMS (HOPE Eq 71):
--   theta^(f_l) updated every C^(f) steps by accumulating optimizer error f(theta; x).
--   Zero update otherwise. eta_t^(f_l) are per-level learning rates.
--   f(.) is the error component of an arbitrary optimizer.
--
-- FUNCTION: freq_gated_cms_forward(x: &Tensor, levels: &[Level; k], pulse: &Pulse) -> Tensor
--   outputs = []
--   FOR l in 0..k:
--     y_l = levels[l].memory_read(x)       -- ALL levels read every token
--     outputs.push(y_l)
--
--   -- Combine outputs via learnable weighted sum (HOPE Eq 74)
--   -- alpha_l: per-level aggregation weights (outer_loop_param, learned via tape)
--   -- Softmax ensures weights sum to 1, preventing output magnitude dependence on k.
--   weights = softmax(alpha)             -- alpha: [k] raw logits, outer_loop_param
--   y = sum_l weights[l] * outputs[l]
--
--   -- Parameter updates happen only at active frequencies:
--   FOR l in 0..k:
--     IF pulse.is_active(l):
--       levels[l].memory_write(x, pulse)    -- inner-loop update runs
--       -- Outer-loop gradients accumulate for this level
--     ELSE:
--       -- Frozen: no write, no outer-loop gradient, read-only M@q path
--       pass
--
--   RETURN y
--
-- Key insight: "When i != 0 (mod C^(f)), no sequential process — enables
-- parallel execution." (HOPE §6). Frozen levels contribute to output via
-- read-only memory access but require no sequential computation.
--
-- This is the DEFAULT NL_Hecate configuration:
--   - The Conductor generates Pulses that gate level activity
--   - All levels read every token (inner-loop M@q always runs)
--   - Only active levels get outer-loop gradient updates
--   - Output aggregation: learnable softmax weights per level (HOPE Eq 74)
--     Phase 1: static alpha_l (outer_loop_param, learned during build)
--     Phase 2: adaptive M_agg(x_t) produces context-dependent level weights
--     (see self_referential/00_interface.md for Phase 2 progression)
--
-- Why this is the default:
--   Simplest multi-scale behavior. No inter-level data dependencies.
--   Each level is independent — parallelizable across levels.
--   The Conductor already implements this via Pulse.is_active().
```

## Variant 3: Nested (Eq 72)

<!-- HADES: hope_equations/eq-072-arch-variant3 (HOPE §6 Eq 72) -->
```text
-- Nested CMS (HOPE Eq 72):
--   Initial state of block at level s+1 is meta-learned in level s.
--   Each level has its own context flow and is RE-INITIALIZED after
--   ceil(C^(s) / C^(s+1)) steps.
--
-- FUNCTION: nested_cms_forward(x: &Tensor, levels: &[Level; k], pulse: &Pulse) -> Tensor
--   -- Level 0 (fastest): standard forward, memory state persists
--   y_0 = levels[0].forward(x, pulse)
--
--   -- Level 1: re-initialized every C^(0)/C^(1) steps with state from Level 0
--   IF pulse.is_reinit_boundary(1):
--     levels[1].state = meta_learn(levels[0].state)   -- slow learns from fast
--   y_1 = levels[1].forward(x, pulse)
--
--   -- Level s+1: re-initialized every C^(s)/C^(s+1) steps with state from Level s
--   -- ...recursive pattern up to level k-1
--
--   RETURN aggregate(y_0, y_1, ..., y_{k-1})
--
-- Properties:
--   - Higher-order in-context learning: level s meta-learns level s+1's initialization
--   - Each level has its OWN context flow (separate inner-loop state)
--   - Re-initialization creates a hierarchy of learning horizons
--   - Update mechanism per block is unchanged (Eq 71) — only initialization differs
--
-- Cost: O(k * block_cost) + meta-learning overhead (one linear transform per reinit)
--
-- When to use:
--   When slow levels should ADAPT to what fast levels have learned.
--   Enables hierarchical in-context learning: fast level extracts patterns,
--   slow level uses those patterns as its starting point.
--   Most expressive variant but most complex to implement.
```

## Variant 4: Sequential (Eq 73)

<!-- HADES: hope_equations/eq-073-arch-variant4 (HOPE §6 Eq 73) -->
```text
-- Sequential CMS (HOPE Eq 73):
--   MLP blocks chained sequentially: level s output feeds level s+1 input.
--   ALL initial states meta-learned at the LOWEST frequency level (C^(1)).
--   The most persistent knowledge is the compression of all components'
--   shared context flow.
--
-- FUNCTION: sequential_cms_forward(x: &Tensor, levels: &[Level; k], pulse: &Pulse) -> Tensor
--   -- Initialize all levels from slowest level's meta-learned state
--   IF pulse.is_reinit_boundary(k - 1):
--     base_state = levels[k - 1].compress_context()
--     FOR l in 0..k:
--       levels[l].state = meta_learn_from(base_state, l)
--
--   h = x
--   FOR l in 0..k:
--     IF pulse.is_active(l):
--       h = levels[l].forward(h, pulse)     -- level l transforms level l-1's output
--     -- If frozen, h passes through unchanged (same as Chained)
--   RETURN h
--
-- Difference from Chained (Variant 1):
--   Chained: each level has independent initial state
--   Sequential: ALL levels initialized from the SLOWEST level's meta-learned state
--   The slowest level (highest frequency index) serves as the "context compressor"
--   — it holds the most persistent knowledge that seeds all other levels.
--
-- Difference from Nested (Variant 3):
--   Nested: level s meta-learns level s+1 (pairwise)
--   Sequential: slowest level meta-learns ALL levels (global)
--
-- When to use:
--   When all levels should share a common initialization derived from
--   the longest-horizon context. Good for tasks requiring strong coherence
--   across timescales (e.g., long document generation where all levels
--   should "agree" on the topic/style established early).
```

## Variant 5: Independent (Eq 74)

<!-- HADES: hope_equations/eq-074-arch-variant5 (HOPE §6 Eq 74) -->
```text
-- Independent CMS (HOPE Eq 74):
--   Each block processes x_t independently at its own frequency.
--   Outputs combined via aggregation function Agg().
--   Uses Eq 73 for knowledge transfer but changes output computation
--   from Eq 70 (chain) to parallel independent blocks.
--
-- FUNCTION: independent_cms_forward(x: &Tensor, levels: &[Level; k], pulse: &Pulse) -> Tensor
--   outputs = []
--   FOR l in 0..k IN PARALLEL:              -- fully parallelizable
--     IF pulse.is_active(l):
--       y_l = levels[l].forward(x, pulse)   -- each level sees RAW x
--     ELSE:
--       y_l = levels[l].memory_read(x)      -- frozen: read-only
--     outputs.push(y_l)
--
--   RETURN Agg(outputs)
--
-- Aggregation function Agg():
--   Simple: learnable weighted sum  y = sum_l alpha_l * y_l  (alpha > 0, outer_loop_param)
--   This is the "simple design choice" recommended by HOPE.
--   Alternatives: attention over level outputs, gated combination, concatenation + projection.
--   Default: learnable weighted sum with softmax normalization (HOPE Eq 74).
--   Phase 1: alpha_l are static outer_loop_params (learned during build).
--   Phase 2 extension: alpha_l replaced by adaptive M_agg(x_t) that weights
--   levels based on context (self-referential progression, see 00_interface.md).
--
-- Properties:
--   - Each level sees the SAME raw input x (no inter-level transformation)
--   - Fully parallelizable across levels — no data dependency between levels
--   - Aggregation weights alpha_l are outer_loop_params (learned via tape)
--   - No information flows BETWEEN levels during forward (only at aggregation)
--
-- Relationship to Variant 2 (Freq-Gated):
--   Variant 2 is essentially Variant 5 with memory_read as the level forward.
--   The NL_Hecate default (Variant 2) IS the Independent variant applied to
--   the memory-rule STEP function: each level independently processes x,
--   outputs aggregated via weighted sum. The difference is purely notational —
--   Variant 2 emphasizes frequency gating, Variant 5 emphasizes independence.
--
-- When to use:
--   When levels should learn INDEPENDENT representations at different timescales.
--   Best for GPU utilization (all levels run in parallel).
--   Trade-off: no inter-level communication means each level must independently
--   discover useful features at its timescale.
```

## Variant 6: M3 Optimizer (Eq 75)

<!-- HADES: hope_equations/eq-075-arch-variant6 (HOPE §6 Eq 75, M3 Algorithm 1) -->
```text
-- M3: Multi-scale Momentum Muon (HOPE Eq 75, Algorithm 1):
--   CMS applied to the OPTIMIZER, not the model architecture.
--   Two-level momentum system with Newton-Schulz normalization.
--
-- M3 = Adam (first+second moment) + Muon (Newton-Schulz) + CMS (chunked slow momentum)
--
-- ALGORITHM: M3 Optimizer (Algorithm 1 from HOPE)
--   Initialize: M^(1)_0 = 0, M^(2)_0 = 0, V_0 = 0    -- fast momentum, slow momentum, variance
--
--   FOR k = 0, 1, 2, ... (outer loop, every C_hat steps):
--     -- Slow momentum update (low frequency):
--     M^(2)_t = M^(2)_{t-1} + beta_3 * sum_{i=(k-1)*f}^{k*f} g_i
--     O^(2)_t = NewtonSchulz_T(M^(2)_t)                -- orthogonalize slow momentum
--
--     FOR t = kf+1, kf+2, ..., (k+1)f (inner loop, every step):
--       g_t = gradient of loss w.r.t. Theta_t
--       -- Fast momentum update (high frequency):
--       M^(1)_t = M^(1)_{t-1} + beta_1 * g_t
--       -- Second moment (Adam-style):
--       V_t = V_{t-1} + beta_2 * g_t^2
--       -- Orthogonalize fast momentum:
--       O^(1)_t = NewtonSchulz_T(M^(1)_t)
--       -- Combined update:
--       Theta_t = Theta_{t-1} - eta * (O^(1)_t + alpha * O^(2)_t) / (sqrt(V_t) + eps)
--
-- Key insight: CMS is not just an architecture pattern — it's a general principle
-- that applies to optimizers too. M3 proves this by achieving competitive results
-- with a two-frequency momentum system.
--
-- Connection to AdaMuon (09_adamuon.md):
--   AdaMuon: Adam + Muon (single frequency)
--   M3: Adam + Muon + CMS (two frequencies: fast momentum + slow momentum)
--   M3 generalizes AdaMuon by adding a slow momentum accumulator that updates
--   at a lower frequency, capturing long-horizon gradient structure.
--
-- Implementation status: Stage 3 extension (see ROADMAP.md).
-- Not yet implemented — AdaMuon (09_adamuon.md) is the prerequisite.
-- M3 adds CMS frequency scheduling to AdaMuon's existing infrastructure.
```

## The HOPE Model: Concrete Configuration

<!-- HADES: Derived from HOPE (2512.24695) §9 Table 6 (ablation study); hope_equations/eq-079-phase2-adaptive-projections (Phase 2) -->
```text
-- The "Hope" model from the paper's experiments uses this specific configuration:

-- Architecture:
--   Variant 2 (Freq-Gated) with k=4 CMS levels
--   MAG composition pattern (memory gates attention)
--   Sliding window attention (w=512)
--   DGD with momentum as the memory algorithm
--   L2 regression as the attentional bias
--   L2 decay as the retention mechanism
--   This is: Titans LMM + MAG + k=4 Freq-Gated CMS

-- Ablation results (Table 6, 760M params, 15B tokens):
--   Hope (full):        12.24 ppl, 58.1% reasoning
--   w/o DGD:            13.41 ppl, 56.5% reasoning
--   w/o Momentum:       13.58 ppl, 56.9% reasoning
--   w/o Multi-scale:    13.76 ppl, 55.2% reasoning  (k=1, no CMS)
--   w/o Memory:         14.68 ppl, 53.8% reasoning  (pure SWA Transformer)
--
-- Every component contributes positively. DGD, momentum, and multi-scale (CMS)
-- each independently improve over the base. The full configuration is the sum
-- of independently beneficial design choices — consistent with MIRAS IS #1
-- (orthogonal design choices).

-- NL_Hecate default matches this configuration:
--   Variant 2 (Freq-Gated) — Conductor + Pulse
--   k=4 — frequencies [1, 8, 64, 512]
--   MAG — memory gates attention
--   Titans LMM — matrix memory, L2, L2 decay, GD+momentum
--   This is not a coincidence — we built NL_Hecate around the HOPE results.
```

## Self-Referential Phases (Eqs 76-79)

The HOPE paper also defines a 3-phase progression for self-referential capabilities.
These are documented in `specs/algorithms/self_referential/` but connect here because
each phase changes what the composition pattern must support.

<!-- HADES: hope_equations/eq-076-phase1-projections (HOPE §8 Eq 76); hope_equations/eq-077-phase1-optimization (Eq 77); hope_equations/eq-078-phase1-read (Eq 78); hope_equations/eq-079-phase2-adaptive-projections (Eq 79) -->
```text
-- Phase 1 (Non-Adaptive, Eqs 76-78):
--   Projections (W_K, W_V, W_Q, gates) are static outer_loop_params.
--   Memory M is the only component that adapts in-context.
--   This is what NL_Hecate Stage 1-2 implements.
--   Composition patterns (MAC/MAG/MAL) work as designed.

-- Phase 2 (Adaptive Projections, Eq 79):
--   ALL projections become learnable associative memories (M_k, M_v, M_q, M_eta, M_alpha).
--   Each projection has its own memory module that updates over time.
--   Shared values across all component memories.
--   The composition pattern must now handle 6 memory modules per level
--   (main memory + 5 projection memories), not just one.
--   See: specs/algorithms/self_referential/01_self_generated_values.md

-- Phase 3 (Self-Generated Values):
--   Values are ALSO self-generated: v_hat = M_v @ k (not from input).
--   Full self-referential loop: the system generates its own training signal.
--   See: specs/algorithms/self_referential/02_self_referential_interface.md

-- Impact on level-composition variants:
--   Phase 1: any variant works — projections are static, only memory adapts
--   Phase 2: Independent (Variant 5) is simplest — each projection memory
--     runs independently at its own frequency. Nested (Variant 3) is most
--     expressive — slow projection memories meta-learn fast ones.
--   Phase 3: same as Phase 2 but with self-generated values adding a
--     feedback loop within each level. No change to inter-level composition.
```

## Inter-Level Knowledge Transfer

<!-- HADES: hope_equations/unnumbered-weight-generation (HOPE §6, weight generation); hope_equations/eq-025-fwp-direct-conn (HOPE §3 Eq 25, FWP direct connection) -->
```text
-- Beyond the six variants, HOPE describes weight generation between levels:
--   One level generates the PARAMETERS (weights) of another level.
--   Function g(.) takes a level's memory, objective, context, and parameters
--   to produce another level's parameters.
--
-- Two key examples:
--   1. Slow → Fast: lowest-frequency level generates initial weights for
--      higher-frequency levels. This is meta-learning across timescales.
--   2. Fast → Slow: fast level's output conditions slow level's input.
--      This is the FWP (Fast Weight Programmer) pattern (Eq 25).
--
-- Weight generation is OPTIONAL and independent of the variant choice.
-- It adds a cross-level dependency that makes parallelization harder
-- but enables richer inter-level communication.
--
-- NL_Hecate status: not yet implemented. The existing CMS implementation
-- (Variant 2, Freq-Gated) does not include weight generation. This is a
-- Stage 3 extension candidate.
```

## Choosing a Variant

<!-- HADES: Derived from hope_equations/eq-070-arch-variant1 through eq-075-arch-variant6 (HOPE §6) -->
```text
-- Decision tree for variant selection:

-- Q: Do you need inter-level communication during forward?
--   NO  → Variant 5 (Independent) or Variant 2 (Freq-Gated)
--         Best GPU utilization. Simplest implementation.
--   YES → Q: Should slow levels see fast levels' output?
--         YES → Variant 1 (Chained) or Variant 4 (Sequential)
--               Richer representations but sequential across levels.
--         NO  → Variant 3 (Nested)
--               Slow levels meta-learn fast levels' initialization.

-- Q: Do you want CMS in the optimizer too?
--   YES → Variant 6 (M3) — CMS applied to momentum accumulation.
--         Orthogonal to the architecture variant choice.

-- Default recommendation:
--   Start with Variant 2 (Freq-Gated). This is what the HOPE paper uses,
--   what NL_Hecate implements, and what the ablation study validates.
--   Only explore other variants if profiling shows a specific deficiency
--   (e.g., slow levels not learning → try Nested for meta-learned init).
```

## Implementation Notes

1. **Variant 2 is already implemented**: The Conductor + Pulse system in NL_Hecate
   implements Variant 2 (Freq-Gated). The `pulse.is_active(level)` check gates
   both inner-loop writes and outer-loop gradient accumulation. Output aggregation
   uses learnable softmax-normalized weights per level (HOPE Eq 74). **Note**: the
   prior `1/sqrt(k)` fixed normalization violated NL IS #9 (principled not ad hoc)
   and has been replaced with learnable weights — see task_44105a for implementation.

2. **Variant 5 is nearly identical to Variant 2**: In the NL_Hecate implementation,
   Variant 2 and Variant 5 are the same — each level independently processes the
   input and outputs are aggregated. The paper distinguishes them notationally
   (Variant 2 emphasizes frequency gating, Variant 5 emphasizes independence)
   but the implementation is identical.

3. **Variants 1/3/4 require architectural changes**: Chained, Nested, and Sequential
   variants add inter-level data dependencies that the current Conductor does not
   support. Implementing these requires extending the `cms_forward()` function to
   pass data between levels. These are Stage 3 extensions.

4. **M3 requires optimizer infrastructure**: Variant 6 (M3) requires adding a
   slow-momentum accumulator to the optimizer state. The AdaMuon spec
   (`09_adamuon.md`) provides the Newton-Schulz building block. M3 adds CMS
   frequency scheduling to the momentum update. Stage 3 extension.

5. **Composition is always two-axis**: When configuring a model, you specify BOTH:
   (a) memory-attention pattern (MAC/MAG/MAL) per level, and
   (b) level-composition variant (1-5) for the CMS.
   These choices are independent. The configuration struct in `02_cms_variants.md`
   already captures both via `BlockConfig.pattern` and `CMSConfig.variant`.

## Axiom Compliance

- **NL IS #2** (nested, multi-level): All variants (except k=1 degenerate case) implement multi-level optimization. The variant choice determines HOW levels nest.
- **NL IS #8** (continuum memory): The frequency spectrum [1, 8, 64, 512] approximates a continuum of timescales. Each variant preserves this continuum with different inter-level connectivity.
- **MIRAS IS #1** (orthogonal design choices): Level-composition variants are orthogonal to memory-attention patterns (MAC/MAG/MAL), to memory rule choice (4 MIRAS knobs), and to optimizer choice (AdamW/AdaMuon/M3).
- **CS-27/28** (frequency-aware optimizer): All variants respect frequency gating — frozen levels do not receive outer-loop gradient updates regardless of the variant.
- **CS-18** (forward pass IS the API): All variants are expressed as forward-pass computation. No separate "level-composition" phase — the variant determines the forward-pass data flow.
