# Composition Pattern Interface

```
CONTRACT
  Purpose:    Defines how a memory update rule connects to an attention module.
              Three patterns (MAC, MAG, MAL) answer the same question differently:
              given memory M and attention A, how do you combine them?
  Expects:    A MemoryUpdateRule implementation and an Attention implementation.
  Guarantees: Any MemoryUpdateRule can plug into any CompositionPattern.
              The pattern is orthogonal to the memory rule and to the
              parallelization strategy.
  Cost:       Zero — trait definition only.
  Trade-off:  MAC = safe but sequential. MAG = parallel but loses reflective gate.
              MAL = simplest but information bottleneck. No single best choice.
  Position:   specs/algorithms/composition_patterns/00_interface.md
              Parent of 01_mac.md, 02_mag.md, 03_mal.md
  Source:     Titans (2501.00663) Section 4, Figure 3
```

## The Three Patterns

| Pattern | Data Flow | Attention Type | Parallelism |
|---|---|---|---|
| **MAC** (Memory As Context) | M reads → concat → A processes → M writes | Full causal | Sequential |
| **MAG** (Memory As Gate) | M and A run parallel → M gates A output | Sliding window | Parallel |
| **MAL** (Memory As Layer) | M preprocesses → A processes M output | Sliding window | Sequential |

## Trait Definition

```
TRAIT: CompositionPattern
  REQUIRES: memory: &mut dyn MemoryUpdateRule
            attention: &dyn Attention

  FORWARD(x: &Tensor, memory: &mut dyn MemoryUpdateRule,
          attention: &dyn Attention, persistent: &Tensor,
          pulse: &Pulse) -> Tensor
    Process input through the composed system.
    Updates memory state. Returns output.

  ATTENTION_KIND() -> AttentionKind
    Returns FullCausal (MAC) or SlidingWindow (MAG, MAL).
    This determines which attention implementations are compatible.
```

## Shared Infrastructure

All three patterns use persistent memory tokens:

```
FUNCTION: prepend_persistent(x: &Tensor, persistent: &Tensor) -> Tensor
  -- persistent: (N_p, d) learnable tokens (outer_loop_param)
  -- x: (B, T, d) input sequence
  -- Returns: (B, T + N_p, d) with persistent tokens prepended
  persistent_expanded = broadcast(persistent, batch_dim=B)
  return concat(persistent_expanded, x, dim=sequence)
```

Persistent tokens store task-level knowledge. They are input-independent,
learned in the outer loop, and shared across all composition patterns.

## Trait Constraints

```
-- MAC requires full causal attention (attends over all assembled context)
-- MAG requires sliding window attention (local context only)
-- MAL requires sliding window attention (local context only)

-- MAC: memory output becomes attention INPUT (safe — attention doesn't know)
-- MAG: memory output GATES attention output (must be in [0,1])
-- MAL: memory output IS attention input (information bottleneck)

COMPILE_TIME_CHECK:
  MAC + SlidingWindowAttention = ERROR  (MAC needs full attention over assembled context)
  MAG + memory_output NOT in [0,1] = ERROR  (gating requires sigmoid)
```

## What This Interface Does NOT Specify

- **Which memory rule**: Any MemoryUpdateRule works. MAC/MAG/MAL are agnostic.
- **Frequency**: CMS scheduling happens at a higher level.
- **Stacking**: Multiple blocks can use different patterns per block.
- **The memory-specific behavior**: Composition patterns delegate to MemoryUpdateRule.
