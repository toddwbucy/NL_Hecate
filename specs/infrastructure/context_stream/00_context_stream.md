# Context Stream (DataLoader Replacement)

```
CONTRACT
  Purpose:    Committee Finding 4: "How does data enter the system?"
              There is no DataLoader. There is no training loop.
              There is a context stream — an infinite sequence of tokens
              that the model processes continuously.
              CS-11: No TrainingLoop or DataLoader class.
              CS-13: The word "training" is a code smell.
  Expects:    A source of token sequences (text, tokenized).
              Chunk size configuration from CMS.
  Guarantees: Tokens arrive in chunks, sized to the CMS frequency schedule.
              The model never "knows" when a document starts or ends
              (unless told via special tokens — that's the data provider's job).
              No epochs. No shuffling. No train/val split.
              Context flows. The model processes.
  Cost:       I/O bound — the stream should never starve the GPU.
              Prefetch + async to hide latency.
  Trade-off:  No DataLoader means no off-the-shelf batching, shuffling, or
              curriculum learning. These must be implemented in the data
              SOURCE, not in the model's consumption pattern.
              This is simpler but less feature-rich than PyTorch DataLoader.
  Position:   specs/infrastructure/context_stream/00_context_stream.md
              Addresses: Committee Finding 4, CS-11, CS-13, nl_toolchain tool-04
  Source:     CS-11 (no training loop); CS-13 (no "training");
              nl_toolchain tool-04 (context-stream processor)
```

## The Stream Abstraction

```
TRAIT: ContextStream
  fn next_chunk(&mut self, chunk_size: usize) -> Option<TokenChunk>
  fn reset(&mut self)          -- for new context / new document
  fn position(&self) -> u64    -- current position in stream

STRUCT: TokenChunk
  tokens: Vec<u32>,            -- token IDs
  chunk_id: u64,               -- monotonic chunk counter
  document_boundary: bool,     -- true if this chunk crosses a document boundary

  -- Note: the model processes ALL chunks identically.
  -- document_boundary is metadata for the Conductor, not the model.
  -- The Conductor may choose to reset ContextMemory at boundaries.
```

## Why Not DataLoader

```
PROBLEM: DataLoader assumes training.
  -- DataLoader.shuffle = True assumes you're training
  -- DataLoader returns (input, target) pairs — supervised assumption
  -- DataLoader has epochs — the concept of "seeing all data once"
  -- DataLoader has batch_size — implies all samples are independent

  None of these concepts exist in NL.

INSTEAD: ContextStream assumes context.
  -- No shuffle — order matters (it's a CONTEXT, not a bag of samples)
  -- No (input, target) — the model predicts the next token, always
  -- No epochs — the stream is potentially infinite
  -- Chunk size is CMS-aligned, not batch-aligned

The model's forward pass IS next-token prediction.
The model's inner loop IS self-modification.
The model's outer loop (Build phase) IS parameter optimization.
None of this requires a DataLoader.
```

## Chunk Sizing

```
-- Chunks are sized to align with CMS frequency boundaries.
-- The fastest CMS level (C=1) processes every token.
-- The slowest CMS level (C=512) processes every 512th token.
-- Chunk sizes should be multiples of the largest CMS frequency.

Example with CMS frequencies [1, 8, 64, 512]:
  -- Chunk size = 512 tokens → all levels fire at least once per chunk
  -- Chunk size = 1024 tokens → all levels fire at least twice per chunk
  -- Chunk size = 64 tokens → level 3 (C=512) never fires within a chunk

  WARNING: If chunk_size < max(CMS_frequencies), some levels NEVER fire.
  This is valid (those levels fire across chunk boundaries via ContextMemory)
  but changes the system's behavior.

Chunk size selection is a hyperparameter, not part of this spec.
But the interaction between chunk size and CMS frequencies is a constraint:
  CONSTRAINT: chunk_size >= CMS_frequencies[0] (at least level 0 fires)
  RECOMMENDATION: chunk_size >= max(CMS_frequencies) (all levels fire)
```

## Data Sources

```
The ContextStream trait is intentionally minimal. Implementations can wrap:

  1. File-based: Read tokenized text files sequentially
     -- Simplest. Good for development and small-scale builds.

  2. Streaming: HuggingFace datasets with streaming=True
     -- No download required. Good for FineWeb-Edu scale data.
     -- Wraps the streaming iterator in ContextStream interface.

  3. Interleaved: Multiple sources (code, text, math) interleaved
     -- Curriculum learning happens HERE, not in the model.
     -- The interleaving strategy is the data source's responsibility.

  4. Interactive: Live user input for inference/Stream phase
     -- Single-user, real-time token stream.
     -- document_boundary = true at conversation breaks.

All of these produce the same TokenChunk interface.
The model doesn't know or care which one is behind the stream.
```

## Integration with Conductor

```
-- The Conductor drives the processing loop:

loop {
  let chunk = stream.next_chunk(chunk_size)?;

  // Process chunk through the model
  let (output, new_context) = model.process_chunk(
    &chunk, context_state, conductor.pulse()
  );

  // Update context memory
  context_state = new_context;

  // Advance the conductor (observe then advance — CS-32)
  conductor.advance(&model);

  // In Build phase: compute loss, run Enzyme backward, update params
  if conductor.pulse().phase == Phase::Build {
    let loss = compute_loss(&output, &chunk);  // next-token prediction
    enzyme_backward(loss);                      // outer-loop gradient
    optimizer.step(conductor.pulse());          // CMS-aware update
  }
}

-- There is no epoch boundary. The loop runs until:
--   (a) Build phase reaches step budget
--   (b) Test phase reaches evaluation budget
--   (c) Stream phase runs indefinitely
```

## Stream-State Coupling (Committee Finding 4)

```
-- Committee Finding: "The stream and the model are one organism."
-- In NL, the model's memory state is the DIRECT PRODUCT of the exact
-- sequence of tokens it has processed. If the system crashes and reloads
-- the model but restarts the data stream from a random point, the
-- continuum of memory is broken. The model has a brain that remembers
-- a past the data stream is no longer reflecting.
-- This violates NL IS #8 (continuum memory).

STRUCT: StreamCursor
  dataset_index: u64,          -- exact position in the data source
  chunk_id: u64,               -- which chunk we're on
  pulse_id: u64,               -- the Pulse that produced this position
  rng_state: RngState,         -- deterministic shuffler state (if any)
  content_hash: u64,           -- hash of the last N tokens processed

  -- The StreamCursor is serialized WITH the model checkpoint.
  -- Not alongside it. Not in a separate file. INSIDE it.
  -- The Conductor owns both the model state and the stream cursor.

RULE: Atomic checkpoint of model + stream.
  -- When the Conductor checkpoints, it freezes BOTH:
  --   (a) Model state (outer-loop params, CMS error buffers, optimizer)
  --   (b) StreamCursor (dataset position, RNG state, pulse_id)
  -- These are saved in a SINGLE atomic operation.
  -- There is no valid checkpoint without a StreamCursor.

RULE: Hash verification on restore.
  -- When loading a checkpoint, the system verifies:
  --   checkpoint.stream_cursor.pulse_id == checkpoint.model.pulse_id
  -- If they differ, the checkpoint is REJECTED with a critical error.
  -- A mismatched stream/model pair WILL produce garbage results.

  fn restore_checkpoint(path: &Path) -> Result<(Model, ContextStream), RestoreError> {
    let checkpoint = load(path)?;
    if checkpoint.stream_cursor.pulse_id != checkpoint.model_pulse_id {
      return Err(RestoreError::StreamModelMismatch {
        stream_pulse: checkpoint.stream_cursor.pulse_id,
        model_pulse: checkpoint.model_pulse_id,
        message: "Model and stream are out of sync. Cannot resume.",
      });
    }
    let model = Model::from_checkpoint(checkpoint.model_state);
    let stream = ContextStream::from_cursor(checkpoint.stream_cursor);
    Ok((model, stream))
  }

RULE: RNG state preservation.
  -- If the data source uses any randomization (shuffling, sampling),
  -- the StreamCursor MUST capture the RNG state.
  -- Without it, even resuming from the correct index may produce
  -- different token orderings, breaking determinism.
  -- If the data source is strictly sequential (no shuffling),
  -- rng_state may be None.
```

## Multi-GPU Considerations

```
-- In multi-GPU setups, each GPU processes DIFFERENT chunks in parallel.
-- This is data parallelism at the chunk level, not sample level.
-- Each GPU has its OWN ContextMemory (memories diverge across ranks).

-- The outer-loop gradients ARE synchronized across ranks.
-- The inner-loop states are NOT synchronized (thread-local).
-- This is the correct behavior: gradients are shared, context is local.

-- The ContextStream provides different chunks to different ranks:
fn next_chunk(&mut self, chunk_size: usize, rank: usize) -> Option<TokenChunk>
  -- rank 0 gets chunks [0, 2, 4, ...]
  -- rank 1 gets chunks [1, 3, 5, ...]
  -- Each rank maintains its own stream position
```

## Axiom Compliance

- **CS-11** (no TrainingLoop): There is no training loop — there is a processing loop
- **CS-13** (no "training"): The word "training" does not appear in any API
- **NL IS #5** (ICL naturally emerges): The stream IS the context for ICL
- **NL IS NOT #4** (not discrete long/short-term): The stream is continuous
