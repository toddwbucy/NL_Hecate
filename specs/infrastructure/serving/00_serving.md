# Serving Non-Stationary Models

```
CONTRACT
  Purpose:    NL models self-modify during inference. The inner loop runs
              at test time -- the model's behavior changes as it processes
              context. This means the model is NON-STATIONARY: the output
              for token 1000 depends on all 999 tokens before it.
              Serving a non-stationary model requires different infrastructure
              than serving a static model.
  Expects:    A built model (outer-loop params frozen after Build phase).
              A context stream (tokens arrive continuously).
              ContextMemory state that persists across requests.
  Guarantees: The model's inner loop runs during serving (self-modification).
              Context memory persists within a session (conversation/document).
              Context memory resets between sessions.
              No outer-loop param modification during serving (Test/Stream phase).
              Latency is bounded by chunk processing time, not context length.
  Cost:       Per-token cost: inner loop + attention (same as Build phase).
              Memory cost: ContextMemory grows with session length
              (bounded by retention mechanisms -- memory decays).
              No Enzyme AD cost (outer loop is frozen).
  Trade-off:  Stateful serving is more complex than stateless.
              Each session needs its own ContextMemory state.
              But: this is how the NL paradigm works. Serving without
              the inner loop would be serving a dead model.
  Position:   specs/infrastructure/serving/00_serving.md
              Addresses: nl_toolchain tool-14, NL IS #5, CS-10, CS-38
  Source:     NL IS #5 (ICL naturally emerges); CS-10 (no train/test mode);
              CS-38 (build not train); nl_toolchain tool-14 (serving)
```

## The Non-Stationary Problem

```
CONVENTIONAL SERVING:
  model = load_checkpoint("model.pt")
  model.set_mode(FROZEN)     # freeze everything
  for request in requests:
    output = model(request)  # same model for every request
    # model state is IDENTICAL before and after

NL SERVING:
  model = load_checkpoint("model.nl")
  // No mode switching -- there IS no separate mode (CS-10)
  for session in sessions:
    context = ContextState::new()  // fresh per session
    for chunk in session.chunks():
      (output, context) = model.process(chunk, context, pulse)
      // model's INNER state changed! Next chunk sees different model.
      // But OUTER params are unchanged (no Enzyme AD in serving).

KEY DIFFERENCE: In NL, the model ADAPTS to the context during serving.
This is NL IS #5: "in-context learning naturally emerges."
The inner loop IS the ICL mechanism.
Disabling it would disable the model's core capability.
```

## Session Management

```
STRUCT: Session
  id: SessionId,
  context: ContextState,          -- running memory state
  conductor: Conductor,           -- timing state for this session
  created_at: Timestamp,

-- Each session has its OWN ContextMemory and Conductor.
-- Sessions are isolated: session A's memory doesn't affect session B.
-- The model's outer-loop params are SHARED across sessions (read-only).

TRAIT: SessionManager
  fn create_session(&mut self) -> SessionId
  fn process(&mut self, session: SessionId, tokens: &[u32]) -> Vec<u32>
  fn close_session(&mut self, session: SessionId)
  fn checkpoint_session(&self, session: SessionId) -> SessionCheckpoint

-- Session creation: allocate fresh ContextState + Conductor
-- Processing: run model with session's context state
-- Close: drop ContextState (free memory)
-- Checkpoint: serialize ContextState for resumption
```

## Phases in Serving

```
Test phase (bounded assessment):
  -- Process a fixed-length context, measure performance.
  -- Inner loop runs. Memory accumulates.
  -- Outer-loop params frozen. No Enzyme AD.
  -- Session ends when context is fully processed.

Stream phase (unbounded serving):
  -- Process tokens indefinitely (conversation, document stream).
  -- Inner loop runs. Memory accumulates continuously.
  -- Outer-loop params frozen. No Enzyme AD.
  -- Session runs until explicitly closed.
  -- Memory is bounded by retention mechanisms (decay).

  NOTE: Stream phase IS serving. There is no separate "inference mode."
  The Conductor's phase flag controls whether Enzyme AD runs.
  Everything else -- inner loop, memory updates, CMS scheduling -- is identical.
```

## Latency Characteristics

```
Per-token latency:
  -- Inner loop computation: O(d_k * d_v) per memory update rule
  -- Attention computation: O(T * d) for full causal, O(w * d) for SWA
  -- CMS gating: O(k) boolean checks (negligible)
  -- Total: dominated by attention (for MAC) or memory (for MAG/MAL)

IMPORTANT: Per-token latency does NOT grow with context length.
  -- Unlike Transformer KV cache (O(T) per token as context grows),
  -- NL memory is fixed-size: d_k * d_v for matrix memory, d for vector.
  -- The memory CONTENT changes (inner loop), but SIZE is constant.
  -- Only attention KV cache grows with context (and SWA bounds it to O(w)).

Throughput in serving:
  -- No Enzyme AD -> roughly 2x faster than Build phase
  -- CMS still applies -> frozen levels still skip computation
  -- Average per-token cost: ~1.14x base cost (same as Build)
  -- Memory bandwidth is the bottleneck (loading model weights per token)
```

## Memory Pressure in Long Sessions

```
ContextMemory grows with session length:
  -- Memory matrix M: fixed size (d_k * d_v), content changes
  -- Momentum S: fixed size, content changes
  -- Attention KV cache: grows with context (bounded by window for SWA)

For MAC with full causal attention:
  -- KV cache grows O(T * d) per layer
  -- This IS the memory bottleneck (same as conventional Transformer serving)
  -- Mitigation: attention is over (memory_tokens + recent_tokens),
  --   NOT over the full context. Memory tokens are fixed count.

For MAG/MAL with sliding window attention:
  -- KV cache is bounded: O(w * d) per layer (w = window size)
  -- Total memory: O(w * d * num_layers) -- CONSTANT regardless of context
  -- This is a significant advantage for long-context serving.

Retention mechanisms prevent memory matrices from diverging:
  -- L2 decay: memory exponentially forgets old information
  -- KL divergence: memory stays close to a reference distribution
  -- Sphere normalization: memory stays on unit sphere (bounded)
  -- Without retention, memory could grow unbounded in magnitude
```

## Checkpoint and Resume (Updated per Committee Finding 4)

```
-- Committee Finding: "The stream and the model are one organism.
-- Serialize the stream cursor inside the model checkpoint. Always."

-- Long sessions can be checkpointed and resumed:

Checkpoint (ATOMIC — all saved together):
  save(session.context)         // ContextState (memory + momentum + step)
  save(session.conductor)       // Pulse state (global_step, phase)
  save(session.stream_cursor)   // StreamCursor (dataset index, RNG state, pulse_id)
  // Outer-loop params NOT saved per session (shared, immutable)

  CRITICAL: All three are saved in a SINGLE atomic operation.
  The Conductor owns both the model context and the stream cursor.
  There is no valid session checkpoint without a stream cursor.

Resume (with verification):
  checkpoint = load(checkpoint_path)

  // Committee Finding 4: Hash check — pulse IDs must match
  if checkpoint.stream_cursor.pulse_id != checkpoint.conductor.pulse_id {
    CRITICAL_ERROR("Stream/model pulse mismatch: stream at pulse {}, model at pulse {}. "
                   "The continuum of memory is broken. Cannot resume.",
                   checkpoint.stream_cursor.pulse_id,
                   checkpoint.conductor.pulse_id);
  }

  context = ContextState::from(checkpoint.context)
  conductor = Conductor::from_checkpoint(checkpoint.conductor)
  stream = ContextStream::from_cursor(checkpoint.stream_cursor)
  session = Session { context, conductor, stream, ... }
  // Continue processing from EXACTLY where we left off

-- This enables:
--   (a) Server restart without losing session state
--   (b) Session migration between servers
--   (c) Long-running document processing with interruption tolerance
--   (d) Deterministic resume: same data sequence, same memory state
```

## What Serving Is NOT

```
-- Serving is NOT "running the model without self-modification."
   The inner loop runs. The model adapts. This IS the model.

-- Serving is NOT "inference mode" or any other distinct mode.
   There is no mode distinction. (CS-10)
   Build phase = outer loop active. Test/Stream phase = outer loop frozen.
   The inner loop runs identically in all phases.

-- Serving is NOT "deploying a static model."
   The model is non-stationary. Its behavior changes with context.
   Two users with different conversation histories will get different outputs.
   This is a feature, not a bug -- it's in-context learning (NL IS #5).
```

## Axiom Compliance

- **NL IS #5** (ICL naturally emerges): Serving IS ICL -- the inner loop IS the mechanism
- **NL IS #7** (self-modifying): Model self-modifies during serving, not just building
- **CS-10** (no mode distinction): No separate mode. Stream phase IS serving.
- **CS-38** (build not train): Build phase builds. Stream phase serves. No "training" or "inference."
- **NL IS NOT #3** (not static): The served model is NOT static -- it adapts to context
