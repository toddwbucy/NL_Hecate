# Spec 71: Chunked Forward Path — TNT Default for Build Mode

## CONTRACT

| Field | Value |
|-------|-------|
| **Purpose** | Restore build-mode throughput by processing full token sequences through existing chunkwise kernels, while keeping per-token path for generation |
| **Expects** | Spec 68D unified forward path (single composable entry point), existing chunkwise CUDA kernels (titans_chunkwise_forward_batched_dd, etc.) |
| **Guarantees** | Build throughput returns to pre-spec-68 levels (~1600+ tok/s at d=1024). Generation path unchanged. Single code entry point preserved (CS-18). No train/eval distinction (CS-10). |
| **Cost** | Chunkwise approximation in build mode (frozen M₀ within chunks). Exact per-token path still available for generation. |
| **Trade-off** | Speed vs exactness. chunk_size=1 gives exact but slow (217 tok/s). chunk_size=64 gives TNT approximation but fast (~1600 tok/s). The TNT paper proves the approximation error is bounded and shrinks with chunk_size. |
| **Position** | `core/src/gpu_stacked_forward.rs` — dual-mode dispatch in `gpu_stacked_forward_tokens` |
| **Source** | TNT (2511.07343) §3.2-3.3, Titans (2501.00663) §3.1 |

---

## Problem

Spec 68D made the unified forward path process tokens one-at-a-time via `forward_single_token`. This is correct but catastrophically slow for build mode:

- **Before spec 68**: seq_len=4096 processed in ~3 kernel launches per block/level (SWA + chunkwise memory + combine). ~1600 tok/s at d=1024.
- **After spec 68D**: 4096 individual `forward_single_token` calls. Each launches ~10 kernels. Total: ~40,000 kernel launches per step. 217 tok/s.

The per-token path is necessary for generation (can't know token N+1 until you sample token N). But build mode knows ALL tokens upfront — it should process them as a batch.

## Design

Two modes within the SAME entry point (`gpu_stacked_forward_tokens`):

```
if tokens.len() > 1:
    forward_sequence()    # build mode: full-sequence chunkwise forward
else:
    forward_single_token() # generate mode: per-token with ActivationWindow
```

### Build Mode: `forward_sequence`

Processes all tokens at once through the existing full-sequence infrastructure:

1. **SWA**: `swa_forward` on full sequence (one kernel launch per block)
2. **Memory levels**: `gpu_memory_forward` per level with chunkwise dispatch
   - Uses `TitansChunkwise` / `DeltaChunkwise` variants
   - chunk_size from config (e.g., 64 for TNT hierarchical)
   - Frozen M₀ within chunks, M updates at chunk boundaries
3. **Combine + project + layernorm**: standard full-sequence ops
4. **Returns**: `GpuStackedCache` directly (no ActivationWindow, no assembly)

### Generate Mode: `forward_single_token` (unchanged)

Per-token processing with ActivationWindow, exactly as today:
1. SWA single-token decode with KV cache
2. Memory forward s=1 per active level
3. Push `TokenActivationCache` to ActivationWindow
4. Assembly + deferred backward after generation completes

### Key: No ActivationWindow for Build Mode

The ActivationWindow was designed for per-token generation where tokens accumulate one by one. In build mode, the full-sequence forward produces a complete `GpuStackedCache` in one call — the same struct backward expects. No per-token caching, no assembly, no D2D copies.

## Changes

### `core/src/gpu_stacked_forward.rs`

**`gpu_stacked_forward_tokens`** — add sequence-mode dispatch:

```rust
pub fn gpu_stacked_forward_tokens(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_ids: &[usize],
    conductor: &mut Conductor,
    context: &mut GpuStackedContext,
    kv_caches: &mut [GpuKVCache],
    ws: &mut StackedDecodeWorkspace,
    activation_window: &mut ActivationWindow,
) -> Vec<f32> {
    if token_ids.len() > 1 {
        // Build mode: full-sequence forward, bypasses ActivationWindow
        return forward_sequence(
            params, cfg, token_ids, conductor, context, kv_caches,
        );
    }
    // Generate mode: per-token (existing path)
    // ... existing for &token in token_ids loop ...
}
```

**`forward_sequence`** — new function, processes full token sequence:

```rust
fn forward_sequence(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_ids: &[usize],
    conductor: &mut Conductor,
    context: &mut GpuStackedContext,
    kv_caches: &mut [GpuKVCache],
) -> Vec<f32> {
    let s = token_ids.len();
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let n_blocks = params.n_blocks();
    let bs = 1; // stacked path always batch_size=1

    // Embed all tokens at once
    let embedded = embed_tokens(params, token_ids, d); // [s, d]

    // Advance conductor once for the full sequence
    let pulse = conductor.pulse();
    for _ in 0..s {
        conductor.advance();
    }

    for block in 0..n_blocks {
        // SWA: full-sequence attention (one kernel launch)
        let (q, k, v_proj) = project_qkv(params.block(block), &block_input, s, d);
        let attn_out = swa_forward(q, k, v_proj, s, nh, hd, window_size);

        // Memory: per-level chunkwise forward
        for level in 0..cfg.k {
            if !pulse.active_levels[level] { continue; }
            let (y_level, mem_cache) = gpu_memory_forward(
                level_params, cfg, &level_input,
                context_m, s, d, bs, level,
            );
            // gpu_memory_forward internally dispatches to chunkwise when
            // tape_strategy is Proxy, or exact when Exact
        }

        // Combine + project + layernorm (full sequence ops)
    }

    // Cross-entropy logits
    let logits = compute_logits(params, &final_output, s, d, v);
    logits[last_token_offset..].to_vec()
}
```

The actual implementation reuses the existing per-level dispatch in `gpu_memory_forward` (gpu_forward.rs:1158), which already handles chunkwise vs exact based on `tape_strategy` and `checkpoint_interval`.

### `cli/src/step.rs`

**`step()`** — detect build mode, skip ActivationWindow:

```rust
pub fn step(
    gpu_params, mag_cfg, gpu_context, adamw_state,
    tokens, targets, conductor, ...
) -> StepResult {
    let n_blocks = gpu_params.n_blocks();
    let mut kv_caches: Vec<GpuKVCache> = ...;

    if tokens.len() > 1 {
        // Build mode: full-sequence forward returns cache directly
        let (last_logits, cache) = gpu_stacked_forward_sequence(
            gpu_params, mag_cfg, tokens, targets,
            conductor, gpu_context, &mut kv_caches,
        );

        let loss = host_cross_entropy_loss(&cache.logits, targets, v, tokens.len());
        let grads = gpu_stacked_backward(gpu_params, mag_cfg, &cache, ...);
        // ... optimizer, reset ...
    } else {
        // Single-token mode (shouldn't happen in step(), but handle gracefully)
        // ... existing ActivationWindow path ...
    }
}
```

**`generate()`** — unchanged. Continues using per-token path with ActivationWindow.

### `core/src/gpu_forward.rs`

**No changes.** `gpu_memory_forward` already dispatches to chunkwise kernels based on `is_proxy` and `chunk_size`. The exact path (tape_strategy=Exact) uses full-trajectory `titans_forward_dd`. The proxy path (tape_strategy=Proxy) uses `titans_chunkwise_forward_batched_dd`.

### `core/src/gpu_backward.rs`

**No changes.** Already handles both `GpuMemoryCache::Titans` (exact) and `GpuMemoryCache::TitansChunkwise` (chunkwise) variants.

### Config

The config's `tape_strategies` field controls exactness:
- `"tape_strategies": ["exact"]` → full M trajectory, exact backward (slower, more memory)
- `"tape_strategies": ["proxy"]` → chunkwise M boundaries, approximate backward (faster, less memory)

For build mode at scale, `"proxy"` is the standard choice (TNT paper default).

Current config has `"tape_strategies": ["exact"]` — this should change to `"proxy"` for throughput. With proxy + chunk_size from `tnt_global_chunk_size` or `chunk_sizes[0]`, the chunkwise kernels fire.

## Conductor Semantics

The conductor advances **once per step** (one optimizer update), not once per token. This matches pre-spec-68 behavior and the spec 57 reset semantics (fire_count increments once per step).

For k=1, this doesn't matter (level 0 always fires). For k>1, the conductor determines which levels are active for the entire step's forward pass — all tokens in the step see the same active levels.

The per-token `conductor.advance()` inside the current `gpu_stacked_forward_tokens` loop was a spec 68D artifact that's incorrect for k>1 (it would advance the conductor seq_len times per step instead of once).

## Memory Budget

At d=1024, hd=64, nh=16, seq_len=4096, chunk_size=64:

| Path | m_states size per block/level |
|------|------------------------------|
| Per-token exact (current) | 16 × 4097 × 64² × 4B = **1.07 GB** |
| Chunkwise (proxy, c=64) | 16 × 65 × 64² × 4B = **17.0 MB** |
| Per-token generate (s=1) | 16 × 2 × 64² × 4B = **524 KB** |

Chunkwise is **63x less memory** than per-token exact. This also means the forward can process the full seq_len without OOM.

## Acceptance Criteria

1. Build mode uses full-sequence forward (not per-token) when tokens.len() > 1
2. Throughput at d=1024 hd=64 batch=3 seq_len=4096: >1000 tok/s (vs 217 current)
3. Generation mode unchanged (per-token with ActivationWindow)
4. Loss matches pre-spec-68 runs within f32 tolerance for same config
5. `cargo check --features cuda`: zero errors
6. `cargo test`: all pass
7. Single entry point preserved (CS-18): `gpu_stacked_forward_tokens` handles both modes

## Traced Equations

- TNT (2511.07343) eq-006: local memory update within chunk (frozen M₀)
- TNT (2511.07343) eq-014: N local memories update at chunk boundaries
- Titans (2501.00663) eq-013: α_t gate for retention
- HOPE (2512.24695) eq-097: CMS chain composition

## Implementation Phases

### Phase A: Reintroduce `forward_sequence` in gpu_stacked_forward.rs
- New function that processes all tokens through existing block-level infrastructure
- Embed → (SWA + memory + combine + project + layernorm) per block → logits
- Returns `GpuStackedCache` directly
- Uses `gpu_memory_forward` for per-level dispatch (chunkwise vs exact)

### Phase B: Wire into `step()` and conductor
- `step()` calls `forward_sequence` when tokens.len() > 1
- Conductor advances once per step, not per token
- Skip ActivationWindow for build mode

### Phase C: Config update
- Change `titans_mag_d1024_hd64_gpu0.json`: `tape_strategies: ["proxy"]`
- Optionally add `chunk_sizes: [64]` for explicit TNT chunk control
- Verify throughput recovery

### Phase D: Clean up
- Remove per-token assembly code from build path (dead code)
- Update `concat_memory_caches` to handle TitansChunkwise (for generate's deferred backward)
