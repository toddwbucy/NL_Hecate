# CMS Variants

```
CONTRACT
  Purpose:    Five configurations of the Continuum Memory System, each
              applying frequency scheduling to different parts of the model.
              From basic (one CMS block) to hybrid (CMS + non-CMS layers).
  Expects:    A model architecture with identifiable parameter groups.
              Frequency scheduler (01_frequency_scheduler.md).
              A Pulse flowing through the system.
  Guarantees: Each variant is a valid CMS configuration.
              The conventional Transformer is the k=1 degenerate case.
              All variants satisfy the NL IS axioms.
  Cost:       Varies. Basic CMS adds only the frequency gating overhead.
              Full nested CMS can be as complex as k * base_model_cost.
  Trade-off:  More CMS coverage = more multi-scale behavior = more complexity.
              Start with Basic CMS, evolve toward Nested or Hybrid as needed.
  Position:   specs/algorithms/frequency_scheduling/02_cms_variants.md
  Source:     HOPE (2512.24695) Section 7, Section 8
```

## Variant 1: Basic CMS

Apply CMS to a single component (typically MLP layers).

```
-- Standard Transformer block:
x -> [Attention] -> [MLP] -> output

-- Basic CMS: MLP is split into k frequency levels
x -> [Attention] -> [CMS_MLP(k levels)] -> output

-- Level 0: runs every step (fast features)
-- Level 1: runs every 8th step (medium features)
-- Level 2: runs every 64th step (slow features)

-- Implementation: k parallel MLP blocks, frequency-gated
-- This is Track A of the HOPE implementation
```

## Variant 2: Nested CMS

CMS within CMS — the optimizer itself uses CMS (M3).

```
-- Model parameters: CMS at frequencies [1, 8, 64, 512]
-- Optimizer momentum: CMS at frequencies [1, 4, 16, 64]
-- Two levels of nesting:
--   Level 1: model learns at multiple timescales
--   Level 2: optimizer's momentum accumulates at multiple timescales

-- This is NL IS #2 taken to its logical extreme.
-- Each "level" has its own optimization loop.
```

## Variant 3: Sequential CMS

Different CMS configurations per model depth.

```
-- Block 0-3:  k=2 CMS (fast + medium)
-- Block 4-7:  k=3 CMS (fast + medium + slow)
-- Block 8-11: k=4 CMS (all timescales)

-- Rationale: early blocks need fast feature extraction.
-- Later blocks need slow, persistent memory of context.
-- The frequency spectrum DEEPENS as you go through the model.
```

## Variant 4: Independent CMS

Each block has its OWN independent frequency schedule.

```
-- Block 0: frequencies [1, 4]
-- Block 1: frequencies [1, 8, 32]
-- Block 2: frequencies [1, 16, 256]

-- No shared schedule. Each block's Pulse is independent.
-- More flexible but harder to reason about.
-- MIRAS would call this "independent design choices per block."
```

## Variant 5: Hybrid CMS

Mix CMS blocks with non-CMS blocks.

```
-- Block 0: Standard MLP (no CMS)
-- Block 1: CMS MLP (k=4)
-- Block 2: Standard MLP
-- Block 3: CMS MLP (k=2)

-- Rationale: not every block needs multi-scale behavior.
-- Some blocks benefit from full-rate processing.
-- Hybrid lets you place CMS where it matters most.

-- Atlas uses this: some blocks have memory (MAG with Omega rule),
-- others are standard attention-only blocks.
```

## Output Aggregation

```
-- When k > 1, CMS produces one output per active level per step.
-- These must be combined into a single output tensor.
--
-- HADES: hope_equations/eq-074-frequency-gated (learnable combination weights)

AGGREGATION:
  alpha: [k] f32                 -- raw logits, outer_loop_param lifetime
  weights = softmax(alpha)       -- normalized per-level weights, always sum to 1
  y = sum_l weights[l] * y_l     -- weighted combination of level outputs

  -- alpha is initialized to zeros → uniform 1/k weighting at init
  -- The Wengert tape differentiates through softmax and the weighted sum
  -- (TapeOp::WeightedSum), so alpha learns from the training signal.
  --
  -- When k=1: softmax([alpha_0]) = [1.0], reduces to identity.
  -- No special-casing needed for any value of k.

WHY LEARNABLE (not fixed):
  -- Fixed 1/k or 1/sqrt(k) normalization violates NL IS #9 (principled, not ad hoc).
  -- Different levels contribute differently depending on the task.
  -- Learnable weights let the model discover the right balance.
  -- Cost: k additional f32 parameters per CMS block — negligible.

PHASE 2 EXTENSION (future, requires self-referential infrastructure S3b-S12):
  -- alpha_l becomes context-dependent: M_agg(x_t) produces per-token weights.
  -- Uses ProjectionKind::Adaptive(rule) from self_referential/00_interface.md.
  -- See composition_patterns/04_hope.md Variant 2 and task_44105a.
```

## Configuration Interface

```
STRUCT: CMSConfig
  n_levels: usize                -- k: number of frequency levels
  frequencies: Vec<u64>          -- [1, 8, 64, 512] typically
  level_dims: Vec<usize>         -- parameter count per level
  alpha: Vec<f32>                -- [k] aggregation logits (outer_loop_param, init zeros)
  variant: CMSVariant            -- Basic, Nested, Sequential, Independent, Hybrid
  blocks: Vec<BlockConfig>       -- per-block configuration

STRUCT: BlockConfig
  pattern: CompositionPatternKind   -- MAC, MAG, or MAL
  memory_rule: MemoryUpdateRuleKind -- which memory rule
  cms_enabled: bool                  -- whether CMS gating applies
  cms_config: Option<CMSConfig>      -- per-block CMS (for Independent variant)
```

## The k=1 Degenerate Case

When k=1 with frequency=[1], CMS reduces to a standard Transformer block:
- One frequency level, always active
- No error accumulation
- No multi-scale behavior
- Standard MLP with standard optimizer
- Aggregation: softmax([alpha_0]) = [1.0] — identity, no combination needed

This is important: CMS is a STRICT GENERALIZATION of the Transformer.
Every Transformer is a CMS model with k=1. The papers prove this formally.

## Axiom Compliance

- **NL IS #2** (nested, multi-level): CMS IS multi-level optimization
- **NL IS #8** (continuum memory): CMS approximates a continuum of timescales
- **NL IS NOT #1** (not single-level): Any k>1 CMS has multiple levels
- **CS-28** (frequency-aware): CMS makes the entire model frequency-aware
