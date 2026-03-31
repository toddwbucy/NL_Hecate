# 68 — Unified Forward Path: One Code Path for Training and Generation

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Eliminate the train/eval code path split by establishing a single forward function that processes tokens identically whether building the model or generating from it |
| Expects   | Three separate forward functions: `gpu_stacked_forward` (training), `gpu_stacked_prefill` (prompt processing), `gpu_stacked_decode_token` (generation). Context resets between fixed-length chunks. Padding required for short inputs. |
| Guarantees | One forward entry point for all modes. No padding. No seq_len constraint on input. Sliding activation window for gradient computation. Continuous memory accumulation across arbitrary token streams. |
| Cost      | Refactor of forward/backward pipeline. New sliding activation cache. Per-token backward amortization strategy. |
| Trade-off | Token-by-token processing is slower than monolithic seq_len chunks. TNT chunking remains as a provably-equivalent throughput optimization, not a separate code path. |
| Position  | Fixes CS-10 violation (separate train/eval paths). Prerequisite for chat interface. Supersedes the prefill+decode split. |
| Source    | Session observation: model trained via `gpu_stacked_forward` produces NaN when evaluated via `gpu_stacked_prefill` due to padding; `gpu_stacked_decode_token` produces valid output. The paths diverged silently. |

## Problem

The model currently has three forward functions:

1. **`gpu_stacked_forward(input_ids[batch×seq_len], targets, pulse, context)`** — training path. Processes a full seq_len chunk atomically. Hand-written CUDA backward uses cached activations. Requires fixed-length padded input.

2. **`gpu_stacked_prefill(input_ids[seq_len], pulse, context)`** — generation prompt path. Calls `gpu_stacked_forward` internally with dummy targets. Inherits the padding requirement.

3. **`gpu_stacked_decode_token(token_id, pulse, context, kv_caches)`** — generation token path. Processes one token. Different kernel dispatch, different attention (single-token SWA vs full-sequence SWA).

### Consequences

1. **NaN on generation**: Padding 3 prompt tokens to seq_len=4096 with 4093 copies of a repeated token causes NaN in the forward pass. The model was never trained on padded input — it saw real text at every position.

2. **CS-10 violation**: The forward pass IS different between training and generation. Different code paths means different numerical behavior, different memory update patterns, different attention computation.

3. **No chat capability**: A chat interface sends variable-length messages. There is no valid way to process 12 tokens through a path that requires 4096.

4. **seq_len is an architectural constraint, not a hyperparameter**: Currently seq_len is baked into the forward pass shape. It should be a gradient window size — how far back we propagate, not how many tokens we can accept.

## Design: Unified Token-Stream Forward

### Core Principle

The model is: **process one token → update memory → produce output**. That is the atomic unit of computation. Everything else (batching, chunking, parallelization) is an optimization that must produce identical results.

### Single Entry Point

```rust
/// Process N tokens through the model. N can be 1 (generation) or any count (training).
/// Same function, same kernels, same memory update, always.
pub fn gpu_stacked_forward_tokens(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_ids: &[usize],       // N tokens, no padding, no seq_len constraint
    conductor: &mut Conductor,
    context: &mut GpuStackedContext,
    kv_caches: &mut Vec<GpuKVCache>,
    activation_window: &mut ActivationWindow,
) -> ForwardResult {
    for &token in token_ids {
        let pulse = conductor.pulse();

        // Process one token through all blocks (embed → per-block [LN → attn → memory → gate → residual] → LN → unembed)
        let step_cache = forward_single_token(params, cfg, token, &pulse, context, kv_caches);

        // Push activations into sliding window (oldest falls off when full)
        activation_window.push(step_cache);

        conductor.advance();
    }

    // Return logits from the last token + loss if targets provided
    activation_window.last_logits()
}
```

### Sliding Activation Window

The activation window replaces the current per-forward activation cache:

```rust
struct ActivationWindow {
    /// Ring buffer of per-token activation caches.
    /// Capacity = gradient_window_size (the old "seq_len").
    entries: VecDeque<TokenActivationCache>,

    /// Maximum number of entries before oldest is evicted.
    capacity: usize,
}

impl ActivationWindow {
    fn push(&mut self, cache: TokenActivationCache) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();  // oldest falls off — already in M
        }
        self.entries.push_back(cache);
    }
}
```

**What the window stores per token** (for backward):
- Embedding vector
- Per-block: LN inputs/outputs, Q/K/V projections, attention output, memory cache (for VJP), gate values, residuals
- Logits

**What happens when a token falls off the window**:
- Nothing needs to happen. The token's influence on M already occurred during forward.
- Gradients can no longer flow through that token's operations (it's outside the backward window).
- This is analogous to truncated BPTT in RNNs — we backprop through the last N steps, not the full history.

### Backward Over the Window

```rust
/// Compute gradients over the activation window (the last N tokens).
/// Called periodically during training — NOT after every single token.
pub fn gpu_stacked_backward_window(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    window: &ActivationWindow,
    targets: &[usize],           // targets aligned to window entries
) -> GpuStackedGrads {
    // Cross-entropy loss over window entries that have valid targets
    let loss = compute_loss(&window, targets);

    // Backward through window in reverse order (newest → oldest)
    // Same CUDA backward kernels, applied per-token
    let grads = backward_through_window(params, cfg, &window);

    grads
}
```

### Training Loop (New)

```rust
// Continuous token stream — no epochs, no resets
let mut activation_window = ActivationWindow::new(gradient_window_size);
let mut kv_caches = create_kv_caches(n_blocks, d, max_cache_len);

loop {
    // Get next batch of tokens from the stream (any count)
    let tokens = stream.next_chunk(gradient_window_size);
    let targets = stream.targets_for(&tokens);

    // Forward: process all tokens, window slides as they're added
    gpu_stacked_forward_tokens(
        &params, &cfg, &tokens, &mut conductor,
        &mut context, &mut kv_caches, &mut activation_window,
    );

    // Backward: gradients over the current window contents
    let grads = gpu_stacked_backward_window(
        &params, &cfg, &activation_window, &targets,
    );

    // Update outer-loop params
    gpu_stacked_adamw_update(&mut params, &mut grads, ...);

    // No reset. No boundary. The stream continues.
    // M carries forward. KV cache grows (or slides). Conductor advances.
}
```

### Generation (Same Code)

```rust
// User sends a message
let prompt_tokens = tokenize(user_message);

// Process prompt — SAME function as training
gpu_stacked_forward_tokens(
    &params, &cfg, &prompt_tokens, &mut conductor,
    &mut context, &mut kv_caches, &mut activation_window,
);

// Generate response — SAME function, one token at a time
loop {
    let logits = activation_window.last_logits();
    let next_token = sample(logits, temperature);

    gpu_stacked_forward_tokens(
        &params, &cfg, &[next_token], &mut conductor,
        &mut context, &mut kv_caches, &mut activation_window,
    );

    // Backward on each generated token too — the model learns from its own output
    // This is the NL signature: optimization IS the forward pass
    let grads = gpu_stacked_backward_window(&params, &cfg, &activation_window, ...);
    gpu_stacked_adamw_update(&mut params, &mut grads, ...);

    emit(next_token);
    if is_stop(next_token) { break; }
}
```

### seq_len Reframed

| Old meaning | New meaning |
|-------------|-------------|
| Fixed input size for forward pass | **gradient_window_size**: how many tokens of backward context we maintain |
| Architectural constraint | Training hyperparameter |
| Requires padding | No padding ever |
| Determines batch shape | Determines VRAM budget for activation cache |

### TNT Chunking as Optimization

TNT chunkwise processing (spec 47) is a **throughput optimization** that produces mathematically equivalent results to token-by-token processing within a chunk. It remains available:

```rust
// Token-by-token (ground truth, used for generation and small inputs):
for &token in tokens {
    forward_single_token(params, cfg, token, ...);
}

// TNT chunked (equivalent, used when processing many tokens for throughput):
gpu_tnt_forward(params, cfg, &tokens[chunk_start..chunk_end], ...);
```

The optimization is **internal to the forward function**. The caller always sees the same interface: tokens in, logits out. The function can choose token-by-token or TNT chunking based on input size.

### KV Cache Management

The SWA KV cache grows as tokens are processed. For long conversations:
- **Window-bounded**: Only the last `window_size` entries are used for attention (SWA is already windowed).
- **Eviction**: Entries older than `window_size` can be dropped from the KV cache.
- **No reset**: KV cache persists across the conversation, same as M.

## Migration Path

### Phase A: Immediate (this PR)
- Fix `generate` and `eval` to use `decode_token` path (no prefill, no padding) — **DONE**
- Remove the debug diagnostics from generate.rs
- Validate that decode_token produces coherent output at step 10K+ — **DONE**

### Phase B: Unified Forward
- Implement `gpu_stacked_forward_tokens` as the single entry point
- `forward_single_token` is the atomic operation (current `decode_token` + activation caching)
- Add `ActivationWindow` ring buffer for activation storage
- Implement `backward_window` that backprops through the window
- Remove `gpu_stacked_prefill` entirely
- `gpu_stacked_forward` becomes an internal optimization path (TNT chunked equivalent)

### Phase C: Sliding Window Training
- Modify training loop to use continuous token stream with sliding window
- Remove fixed-chunk batch boundaries
- Checkpoint: save M state, KV cache tail, conductor step, stream cursor, activation window
- Restart GPU0 with the unified path

### Phase D: Chat Integration
- Chat interface calls the same `gpu_stacked_forward_tokens` for user messages and response generation
- Backward runs on each token — the model learns during conversation
- Session state: M + KV cache + conductor + activation window

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | New `gpu_stacked_forward_tokens`, `forward_single_token`, `ActivationWindow`. Deprecate `gpu_stacked_prefill`. |
| `core/src/gpu_stacked_backward.rs` | New `gpu_stacked_backward_window` that operates on the activation ring buffer |
| `cli/src/run.rs` | Replace chunk-based training loop with continuous sliding window loop |
| `cli/src/generate.rs` | Use `gpu_stacked_forward_tokens` instead of prefill+decode |
| `cli/src/eval.rs` | Use `gpu_stacked_forward_tokens` for all probes |
| `cli/src/chat.rs` | Same function as generate, with persistent state |

## Validation

1. **Numerical equivalence**: `forward_single_token` called N times produces identical logits to `gpu_stacked_forward` on the same N tokens (at chunk_size=1, no TNT)
2. **No padding**: Generation works with any prompt length (1 token to 100K tokens)
3. **Same code path**: `generate` and training call the same function
4. **Gradient correctness**: `backward_window` produces identical gradients to current `gpu_stacked_backward` for the same token sequence
5. **Continuous context**: M state accumulates across gradient window boundaries — loss at window N+1 reflects learning from window N

## Success Criteria

1. One forward function for training, generation, chat, and evaluation
2. No padding tokens ever
3. seq_len is a gradient window parameter, not an input shape constraint
4. Model learns continuously — no boundary resets
5. Chat interface works with arbitrary message lengths

## Non-Goals

- Async backward (backward still synchronous with forward)
- Multi-GPU (single GPU first)
- Dynamic gradient window resizing during training
- Quantized activation cache (full f32 for now)
