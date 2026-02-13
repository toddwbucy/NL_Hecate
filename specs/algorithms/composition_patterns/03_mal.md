# MAL (Memory As Layer)

```
CONTRACT
  Purpose:    Memory preprocesses input, attention processes memory output.
              Simplest composition — memory is a preprocessing layer.
              Creates an information bottleneck: attention only sees what
              memory decides to pass through.
  Expects:    Input (B, T, d). Memory state. Persistent tokens.
              Sliding window attention (not full causal).
  Guarantees: Output (B, T, d). Memory state updated.
              Information bottleneck: attention sees memory-filtered input.
              Sequential: memory first, then attention.
  Cost:       memory_cost + attention_cost (sequential, not parallel).
              But both are cheaper than MAC: memory doesn't need attention output,
              attention uses SWA not full causal.
  Trade-off:  Simplest data flow. But information bottleneck is severe:
              attention ONLY sees what memory passes through. If memory
              suppresses a token, attention cannot recover it.
              Works best when memory is a reliable compressor.
  Position:   specs/algorithms/composition_patterns/03_mal.md
              Child of 00_interface.md
  Source:     Titans (2501.00663) Section 4.3
```

## Data Flow

```
Input x
    |
    v
[Memory STEP] ---> m_t (memory-processed input)
    |
    v
[Prepend Persistent] ---> m_t_tilde
    |
    v
[SWA] ---> output
```

## Pseudocode

```
ALGORITHM: mal_forward(x: &Tensor, memory: &mut dyn MemoryUpdateRule,
                       attention: &dyn Attention, persistent: &Tensor,
                       pulse: &Pulse) -> Tensor
  -- x: (B, T, d)

  -- Step 1: Memory processes input
  m_t = memory.STEP(x, pulse)                         -- (B, T, d)

  -- Step 2: Prepend persistent tokens to memory output
  m_t_tilde = prepend_persistent(m_t, persistent)      -- (B, T + N_p, d)

  -- Step 3: Sliding window attention over memory output
  output_full = attention.forward(m_t_tilde)            -- (B, T + N_p, d)
  output = output_full[:, persistent.len():, :]         -- (B, T, d)

  return output
```

## The Information Bottleneck

MAL creates a strict bottleneck: attention can only work with what memory provides.

```
-- MAC:  attention sees [persistent, memory_context, raw_input]
-- MAG:  attention sees [persistent, raw_input], gated by memory
-- MAL:  attention sees [persistent, memory_output]

-- In MAC, attention can always "fall back" to the raw input.
-- In MAG, attention always processes raw input; memory only filters output.
-- In MAL, if memory drops information, it's gone. Attention cannot recover it.
```

This bottleneck is a feature when memory is a good compressor:
the attention module sees a cleaned, compressed representation
instead of raw noisy tokens. But it's a bug when memory makes mistakes.

## When to Use MAL

- When the memory module is well-trained and reliable
- When the input is noisy and needs preprocessing
- When you want to reduce the effective sequence length for attention
  (memory can compress multiple tokens into fewer, denser representations)
- As the inner layer in a stacked architecture (MAL at bottom, MAC at top)

## When NOT to Use MAL

- When memory is still learning (early in the build process)
- When faithfulness to raw input matters (e.g., verbatim recall tasks)
- When you need redundancy (MAC lets attention bypass memory if needed)

## Axiom Compliance

- **NL IS #3** (each level with own context flow): Memory creates its own context for attention
- **NL IS #4** (compressing context): Memory compresses before attention processes
- **Titans IS #6** (compositional): MAL = memory + SWA + persistent, simplest composition
