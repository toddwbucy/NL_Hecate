# State Ownership & Lifecycle

```
CONTRACT
  Purpose:    Committee Finding 3: "Who owns the memory state?" Defines
              the three lifetimes in the system, their ownership semantics,
              and how Rust's ownership model enforces them at compile time.
              Every tensor in the system belongs to exactly one lifetime.
  Expects:    Rust ownership and borrowing system.
              Three distinct lifetime categories (contract Section 3).
  Guarantees: No state ambiguity — every tensor is classified.
              Inner-loop state cannot escape its forward pass.
              Context memory transfers are explicit (move, not copy).
              Outer-loop params are the only things Enzyme AD touches.
  Cost:       Zero runtime cost — ownership is a compile-time property.
  Trade-off:  Strict ownership prevents accidental state leakage, but
              requires the programmer to think about lifetimes up front.
              This is a feature: ambiguous state ownership caused the
              v1 HOPE training bugs (leaking inner-loop state across steps).
  Position:   specs/infrastructure/state_lifecycle/00_state_ownership.md
              Addresses: Committee Finding 3, CS-46, contract Section 3
  Source:     Contract v0.2.0 Section 3; CS-46 (in-place modification
              destroys reproducibility); Rust ownership model
```

## The Three Lifetimes

```
LIFETIME 1: OuterLoopParam
  -- Persists across the entire build process (all steps, all chunks)
  -- Modified ONLY by outer-loop backprop (Enzyme AD)
  -- Serialized to checkpoints
  -- Examples: W_K, W_V, W_Q, gate parameters, CMS level weights,
  --           persistent memory tokens (M_init in Track B)

  Rust semantics:
    struct Model {
      w_k: Tensor,           // owned by model, persists across calls
      w_v: Tensor,
      gate_params: Tensor,
      persistent_memory: Tensor,
    }

  AD annotation: #[autodiff] (Enzyme differentiates through these)
  Serialization: YES — checkpoint includes all OuterLoopParam tensors
  Gradient: YES — d(loss)/d(param) via Enzyme reverse-mode AD


LIFETIME 2: InnerLoopState
  -- Created fresh each forward pass (or each chunk)
  -- Modified by the inner optimization loop (GD, momentum, etc.)
  -- Dropped when the forward pass (or chunk) completes
  -- NOT serialized — this is a "thought", not a "memory"
  -- NOT part of the outer-loop gradient graph (detached)

  Rust semantics:
    fn forward(&self, x: &Tensor, pulse: &Pulse) -> Tensor {
      let mut memory = self.persistent_memory.clone();  // clone from outer param
      let mut momentum = Tensor::zeros(...);             // fresh each call

      for token in chunk {
        // inner loop modifies memory and momentum IN PLACE
        inner_step(&mut memory, &mut momentum, token);
      }

      // memory and momentum are DROPPED here — cannot escape
      output
    }

  AD annotation: #[custom_vjp] (the inner loop provides its own backward)
  Serialization: NO — inner state is ephemeral
  Gradient: Inner-loop gradients are analytical (from paper equations).
            They are NOT computed by Enzyme.
            But the OUTER-loop gradient DOES flow through them via #[custom_vjp].


LIFETIME 3: ContextMemory
  -- Persists across forward passes within a context stream
  -- The running memory state that accumulates as tokens arrive
  -- Explicitly transferred between invocations (Rust move semantics)
  -- Reset between contexts (new document = fresh context memory)

  Rust semantics:
    struct ContextState {
      memory: Tensor,        // running memory at last chunk boundary
      step: u64,             // where we are in the context
    }

    // Transfer is explicit — the caller gives up ownership
    fn process_chunk(&self, x: &Tensor, state: ContextState,
                     pulse: &Pulse) -> (Tensor, ContextState) {
      // state is MOVED in, modified, and MOVED out
      // caller cannot use old state after this call
      ...
      (output, new_state)
    }

  AD annotation: Depends on phase.
    Build phase: context memory gradients flow through Enzyme
    Test/Stream phase: no outer-loop gradients computed
  Serialization: YES for long-context streaming (checkpoint the stream position)
  Gradient: In Build phase, d(loss)/d(context_memory_init) flows through.
            In Test/Stream phase, no gradient — just forward processing.
```

## Ownership Rules

```
RULE 1: No shared mutable state
  -- Two components CANNOT hold &mut references to the same tensor.
  -- The inner loop owns its memory copy. The model owns its params.
  -- This prevents the v1 bug where inner-loop modifications leaked
  -- into the outer-loop parameter state.

RULE 2: Inner state cannot escape
  -- InnerLoopState tensors have a lifetime bounded by the forward pass.
  -- Rust enforces this: the &mut memory reference is scoped to the
  -- for-loop body. After the loop, it's gone.
  -- There is no "return the inner state" — only the output escapes.

  Exception: ContextMemory IS explicitly returned (move semantics).
  But ContextMemory is a DIFFERENT lifetime than InnerLoopState.
  The inner-loop MODIFIES context memory, then context memory is
  returned. The inner-loop's own state (momentum, step count, etc.)
  is still dropped.

RULE 3: Outer params are immutable during forward pass
  -- In Rust terms: forward() takes &self, not &mut self.
  -- The model's outer-loop parameters are read-only during forward.
  -- Only Enzyme's backward pass (Build phase) modifies them.
  -- This is enforced by the borrow checker: &self means you
  -- can read self.w_k but not write to it.

RULE 4: Context memory transfer is move, not copy
  -- When you pass ContextState to process_chunk(), the caller
  -- loses ownership. The old state is CONSUMED.
  -- This prevents the bug: "process chunk N, then accidentally
  -- also use the pre-chunk-N state for something else."
  -- In Rust: process_chunk(state) moves state. Using state after
  -- this call is a compile error.
```

## Phase Transitions and State

```
Build phase:
  -- OuterLoopParam: Enzyme computes gradients, optimizer applies updates
  -- InnerLoopState: created, used, dropped (each chunk)
  -- ContextMemory: transferred between chunks, gradients flow through

Test phase:
  -- OuterLoopParam: FROZEN (no gradients, no updates)
  -- InnerLoopState: created, used, dropped (identical to Build)
  -- ContextMemory: transferred between chunks (no gradients)

  CRITICAL: The inner loop runs IDENTICALLY in Build and Test.
  The ONLY difference: Enzyme AD is disabled in Test phase.
  The model still self-modifies (inner loop still runs).
  It just doesn't learn from the self-modification.

Stream phase:
  -- OuterLoopParam: FROZEN
  -- InnerLoopState: created, used, dropped (identical)
  -- ContextMemory: transferred indefinitely (no context boundary)

  Stream is Test but without an end condition.
  Memory accumulates without bound until retention decays it.
```

## Anti-Patterns (Bugs We've Already Hit)

```
BUG 1: Inner state leaking into outer params (v1 HOPE)
  -- Caused by: storing inner-loop memory in the model struct
  -- Symptom: outer-loop gradient was stale (wrong memory state)
  -- Fix: inner loop operates on a CLONE, original untouched
  -- Enforcement: &self in forward() — can't mutate model struct

BUG 2: Accidental state sharing across ranks (DDP)
  -- Caused by: DDP syncing inner-loop state across GPUs
  -- Symptom: inner-loop gradients were averaged (wrong — they're local)
  -- Fix: inner-loop state is thread-local, never synced
  -- Enforcement: InnerLoopState is !Send (cannot cross thread boundary)

BUG 3: Context memory not properly reset between documents
  -- Caused by: forgetting to create fresh ContextState
  -- Symptom: new document "remembered" previous document's content
  -- Fix: explicit ContextState::new() at document boundaries
  -- Enforcement: ContextState has no Default impl — must be constructed

BUG 4: In-place modification of outer params (CS-46)
  -- Caused by: optimizer writing directly to param tensor
  -- Symptom: non-reproducible builds (order-dependent updates)
  -- Fix: optimizer produces a DELTA, applied atomically
  -- Enforcement: outer params are behind &self during forward

BUG 5: Stream/model mismatch on resume (Committee Finding 4)
  -- Caused by: checkpointing model state and stream position separately
  -- Symptom: model's memory reflects tokens 0-10000, but stream resumes
  --          from token 5000 or 15000. Memory is incoherent with context.
  -- Fix: Conductor owns the stream. Checkpoint is ATOMIC (model + stream).
  -- Enforcement: StreamCursor is mandatory in checkpoint struct.
  --              Pulse ID verification on restore rejects mismatches.
```

## Stream Ownership (Committee Finding 4)

```
-- Committee Finding: "The stream and the model are one organism."

-- The ContextStream is OWNED by the Conductor, not by external code.
-- This ensures that when the Conductor checkpoints, it atomically
-- freezes both the model state and the stream position.

STRUCT: Conductor
  config: CMSConfig,
  pulse: Pulse,
  step_observers: Vec<Box<dyn StepObserver>>,
  stream: Box<dyn ContextStream>,       // <-- OWNED by Conductor
  stream_cursor: StreamCursor,           // <-- serialized with checkpoint

-- When the Conductor pauses (for checkpoint, for phase transition),
-- the stream is frozen in the SAME atomic operation.
-- There is no race condition between model state and stream position.

-- This prevents BUG 5 below (stream/model mismatch on resume).
```

## Checkpoint Semantics (Updated per Committee Finding 4)

```
A BUILD checkpoint contains:
  1. All OuterLoopParam tensors (model weights, gate params, etc.)
  2. Optimizer state (momentum accumulators — one per CMS level)
  3. Conductor state (global_step, phase)
  4. CMS error buffers (accumulated gradients for frozen levels)
  5. StreamCursor (dataset index, chunk_id, pulse_id, RNG state)  // MANDATORY

A checkpoint does NOT contain:
  - InnerLoopState (ephemeral — recreated each forward pass)
  - Gradient tensors (recomputed each step)

A STREAM/SESSION checkpoint contains:
  1. ContextMemory (running memory state at current chunk boundary)
  2. Conductor state (global_step, phase)
  3. StreamCursor (dataset index, chunk_id, pulse_id, RNG state)

CRITICAL: StreamCursor is NEVER optional.
  -- Without it, restoring a checkpoint puts the model in a state
  -- where its memory reflects one data sequence but the stream
  -- is feeding it a different sequence. This breaks the continuum.
  -- Committee Finding 4: "You can't let the model run on a mismatched reality."

On restore, the system MUST verify:
  checkpoint.stream_cursor.pulse_id == checkpoint.conductor.pulse_id
  If mismatch → critical error, refuse to load.
```

## Axiom Compliance

- **NL IS #7** (self-modifying): InnerLoopState IS self-modification in action
- **NL IS NOT #3** (not static): Three lifetimes, all with different dynamics
- **CS-46** (in-place modification): Prevented by ownership rules
- **CS-10** (no train/eval): Phase changes what Enzyme does, not what the model does
