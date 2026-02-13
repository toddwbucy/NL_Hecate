# MAC (Memory As Context)

```
CONTRACT
  Purpose:    Memory provides additional context tokens for attention.
              Safest composition — attention's contract is fully preserved.
              Memory reads BEFORE attention, writes AFTER. Sequential but correct.
  Expects:    Input segment (B, L_seg, d). Memory state. Persistent tokens.
              Full causal attention module (not sliding window).
  Guarantees: Output (B, L_seg, d). Memory state updated.
              Attention sees memory output as additional input tokens — no
              modifications to attention internals required.
              Compatible with flash attention, xformers, any optimized kernel.
  Cost:       Attention: O((N_p + L_seg + L_seg)^2 * d) — full causal over assembled context.
              Memory: depends on underlying MemoryUpdateRule.
              Sequential: memory read, then attention, then memory write.
  Trade-off:  Correctness over speed. Attention is unmodified, but memory and
              attention cannot overlap. The reflective output gate (Eq 25) adds
              a second memory read — more expensive but enables "thinking about
              what you just thought."
  Position:   specs/algorithms/composition_patterns/01_mac.md
              Child of 00_interface.md
  Source:     Titans (2501.00663) Section 4.1, Eqs 21-25
```

## Data Flow

```
Input Segment S^(i)
    |
    v
[Memory READ] ---> h_t (historical context tokens)
    |
    v
[Persistent || h_t || S^(i)] = S_tilde  (assembled context)
    |
    v
[Full Causal Attention] ---> y_t
    |
    v
[Memory WRITE(y_t)] ---> update memory state
    |
    v
[Output Gate: y_t * Memory_READ(y_t)] ---> o_t   (reflective gate)
```

## Pseudocode

```
ALGORITHM: mac_forward(x: &Tensor, memory: &mut dyn MemoryUpdateRule,
                       attention: &dyn Attention, persistent: &Tensor,
                       pulse: &Pulse) -> Tensor
  -- x: (B, L_seg, d) — one segment of the input
  -- persistent: (N_p, d) — outer_loop_param

  -- Step 1: Read historical context from memory (Titans Eq 21)
  q_segment = project_to_query(x)                         -- (B, L_seg, d)
  h_t = memory.READ(q_segment)                            -- (B, L_seg, d)

  -- Step 2: Assemble three-branch input (Titans Eq 22)
  persistent_expanded = broadcast(persistent, batch=B)     -- (B, N_p, d)
  S_tilde = concat(persistent_expanded, h_t, x, dim=seq)  -- (B, N_p + 2*L_seg, d)

  -- Step 3: Full causal attention over assembled context (Titans Eq 23)
  y_full = attention.forward(S_tilde)                      -- (B, N_p + 2*L_seg, d)
  y_t = y_full[:, N_p + L_seg:, :]                        -- (B, L_seg, d) segment portion

  -- Step 4: Write attention output to memory (Titans Eq 24)
  -- Memory learns what attention found relevant
  FOR each token y in y_t along sequence dimension:
    memory.WRITE(y, pulse)
    -- WRITE respects Pulse: if level frozen, accumulates error

  -- Step 5: Reflective output gate (Titans Eq 25)
  -- Query UPDATED memory with y_t, then gate
  y_reflected = memory.READ(y_t)                           -- (B, L_seg, d)
  output = y_t * sigmoid(y_reflected)                      -- element-wise gate

  return output
```

## Segmentation

MAC processes input in segments (not individual tokens) because full attention
over the entire assembled context would be O(T^2):

```
FUNCTION: process_full_sequence(x: &Tensor, memory, attention, persistent,
                                 pulse, segment_length) -> Tensor
  outputs = []
  FOR segment in split(x, segment_length):
    out = mac_forward(segment, memory, attention, persistent, pulse)
    outputs.push(out)
  return concat(outputs, dim=seq)
```

## The Reflective Gate

Step 5 is MAC's unique feature. After writing y_t to memory (step 4),
it immediately reads using y_t as query (step 5). This creates a loop:
1. Attention produces y_t (what's relevant)
2. Memory digests y_t (incorporates new knowledge)
3. Memory is queried with y_t (what does UPDATED memory think?)
4. That opinion gates the output

## Why MAC is "Safe"

Attention sees h_t as additional input tokens. From attention's perspective,
nothing unusual is happening — just a longer sequence. Memory state updates
happen OUTSIDE the attention computation. This means:
- Flash attention works (no custom kernels needed)
- xformers works
- Any optimized attention implementation works
- The static-graph assumption of attention is preserved

The cost: memory and attention are sequential, not parallel.

## Axiom Compliance

- **Titans IS #6** (compositional): MAC = LMM + full attention + persistent + reflective gate
- **NL IS #5** (ICL emerges): Memory tokens become context that attention learns to use
- **NL IS #6** (optimizers are memory): Memory module IS the optimizer. Attention arbitrates.
