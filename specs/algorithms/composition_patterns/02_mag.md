# MAG (Memory As Gate)

```
CONTRACT
  Purpose:    Memory and attention run in parallel. Memory output gates
              attention output. The default composition for Atlas. Enables
              parallel execution of memory and attention branches.
  Expects:    Input (B, T, d). Memory state. Persistent tokens.
              Sliding window attention (not full causal).
  Guarantees: Output (B, T, d). Memory state updated.
              Memory and attention are independent branches — can execute
              simultaneously on separate compute units.
              Memory output in [0,1] (sigmoid) to serve as valid gate.
  Cost:       max(attention_cost, memory_cost) — parallel execution.
              Attention: O(T * W * d) where W = window size (SWA).
              Memory: depends on MemoryUpdateRule.
  Trade-off:  Parallel execution (faster than MAC) but no reflective gate.
              Memory cannot see attention output — only the raw input.
              Gating can be lossy: if memory gate ~0, attention output is zeroed.
  Position:   specs/algorithms/composition_patterns/02_mag.md
              Child of 00_interface.md
  Source:     Titans (2501.00663) Section 4.2, Atlas (2505.23735) default
```

## Data Flow

```
Input x
    |
    +--> [Memory Branch] ---> m_t (memory output)
    |                              |
    |                              v
    |                         sigmoid(m_t) = gate
    |                              |
    +--> [SWA Branch] ------> a_t  |
                               |   |
                               v   v
                          output = a_t * gate
```

## Pseudocode

```
ALGORITHM: mag_forward(x: &Tensor, memory: &mut dyn MemoryUpdateRule,
                       attention: &dyn Attention, persistent: &Tensor,
                       pulse: &Pulse) -> Tensor
  -- x: (B, T, d)

  -- Branch 1: Memory (can execute independently)
  m_t = memory.STEP(x, pulse)                     -- (B, T, d)
  gate = sigmoid(m_t)                              -- (B, T, d), in [0,1]

  -- Branch 2: Sliding Window Attention (can execute independently)
  x_with_persistent = prepend_persistent(x, persistent)
  a_t = attention.forward(x_with_persistent)       -- (B, T, d)
  -- Extract original sequence portion (remove persistent prefix)
  a_t = a_t[:, persistent.len():, :]               -- (B, T, d)

  -- Combine: memory gates attention output
  output = a_t * gate                              -- element-wise

  return output
```

## Parallelism

The two branches are independent:
- Memory branch: uses only x and memory state
- Attention branch: uses only x and persistent tokens
- No data dependency between them until the final gate

In a Rust implementation, these branches can run on separate threads or
on separate CUDA streams. The only synchronization point is the multiply.

```
-- Parallel execution pattern (conceptual):
SPAWN memory_task:  m_t = memory.STEP(x, pulse)
SPAWN attn_task:    a_t = attention.forward(x_with_persistent)
AWAIT both
output = a_t * sigmoid(m_t)
```

## Why Sliding Window?

MAC uses full causal attention because it has memory context tokens to attend over.
MAG doesn't — memory and attention are separate branches.
Using full attention would be wasteful because:
1. There are no memory context tokens to attend over
2. The long-range context is captured by the memory branch, not by attention
3. SWA provides local context; memory provides global context

## Gating Semantics

The memory gate controls WHICH attention outputs to pass through:
- gate ≈ 1: memory says "this is important, let it through"
- gate ≈ 0: memory says "this is noise, suppress it"
- gate ≈ 0.5: memory is uncertain

This is different from MAC where memory provides ADDITIONAL context.
MAG provides a FILTER on attention's existing output.

## Trait Constraint

```
COMPILE_TIME_CHECK:
  MAG requires memory output to be sigmoid-compatible.
  If MemoryUpdateRule.READ returns values outside a range where
  sigmoid produces meaningful gradients, training may stall.
```

## Axiom Compliance

- **Atlas IS #6** (composable building blocks): MAG = memory + SWA + gating
- **NL IS #2** (parallel): Memory and attention run simultaneously at same level
- **MIRAS IS NOT #6** (attention optional): Memory branch works without attention.
  The attention branch could be removed, leaving pure memory.
