# 47 — Single-Token Decode for Stacked Models

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Implement prefill + single-token decode on `GpuStackedModel`, eliminating the 512-token full-forward replay that makes autoregressive generation 500x slower than necessary. The memory system M already holds compressed history — the decode path feeds one token through the existing forward math without replaying the sequence. No model change, no retraining, existing checkpoints work immediately. |
| Expects   | `GpuStackedModel` with N blocks × k CMS levels. Existing `GpuKVCache` struct (`core/src/gpu_forward.rs:2778`). Existing `swa_single_token_cuda` CUDA kernel (`core/src/cuda_ffi.rs:45`). Existing `gpu_memory_forward` that accepts arbitrary `s` parameter (including s=1). Per-head memory (spec 45) via `GpuStackedContext`. Conductor/Pulse system for CMS level scheduling. |
| Guarantees | 1. Prefill processes full seq_len prompt through all blocks, populating per-block KV caches and advancing M for all levels. 2. Decode processes exactly 1 token per call through all blocks: embed → per-block(LN → Q/K/V proj → SWA via cache → memory update/read → composition → residual) → final LN → unembed → logits. 3. Same M update math as training forward — no approximation, no mode flag (CS-10). 4. Conductor advances normally during decode — CMS levels fire at their frequencies. 5. Decode throughput: O(d²) per token per level instead of O(seq_len × d²). Target: >1000 tok/s decode on A6000 at d=768. |
| Cost      | Per-block KV cache: `window_size × 2 × d × sizeof(bf16)` = 512 × 2 × 768 × 2 = 1.5 MB per block. Total for 6 blocks: **9.4 MB**. Negligible vs M (which is already allocated at 12 × 64 × 64 × 4 = 192 KB per level per block for d=768, head_dim=64, num_heads=12). No new CUDA kernels — all primitives exist. |
| Trade-off | The SWA component still needs a KV cache (attention is inherently stateless over its window). But this is a small circular buffer, not the full-sequence KV cache a transformer needs. The memory system carries the long-range context. During decode, inactive CMS levels still do M@q readout (read-only path) at O(d²) per token — unavoidable for correct output but cheap. |
| Position  | `specs/infrastructure/47_single_token_decode.md` |
| Source    | CS-10 (no train/eval distinction). CS-18 (forward code identical in all phases). Titans (2501.00663) eq-034: M update is per-token. HOPE (2512.24695) eq-070/eq-097: CMS chain architecture. Existing `GpuKVCache` + `swa_single_token_cuda` infrastructure. |

## Problem Statement

`generate_stacked` (python/engine/generation.py:254) runs a **full 512-token forward pass through all 6 blocks** for every single generated token. 4 prompts × 20 tokens = 80 full forward passes. At ~500ms per forward, that's 40 seconds for 80 tokens — **37 tok/s** observed at step 1024.

This is equivalent to running a transformer without KV cache. But it's worse than that — NL models have M, which already accumulates the token history. A transformer *must* replay because attention is stateless. NL shouldn't replay because M is stateful.

The fix is not a "serving mode" or an "inference optimization" — it's completing the forward path. The model already processes tokens one at a time through M during training (the inner loop). The decode path does the same thing without the backward pass.

## Key Insight: What Needs a Cache vs What Doesn't

| Component | Needs cache? | Why |
|-----------|-------------|-----|
| Memory M | **No** — M IS the cache | M accumulates all token history via the recurrence. After prefill, M contains the compressed representation of all prompt tokens. Each decode step updates M with one new token. |
| Momentum S | **No** — same as M | S follows M's recurrence (Titans LMM). Updated alongside M. |
| SWA Attention | **Yes** — KV cache needed | Attention over the sliding window requires previous K/V vectors. Standard circular buffer, bounded by `window_size`. |
| Embedding/LN/Projections | **No** — stateless | Per-token computation, no history needed. |
| Gates (α, θ, η) | **No** — per-token | Computed fresh for each token from the current embedding. |

## Existing Infrastructure

Three key pieces already exist:

### 1. GpuKVCache (`core/src/gpu_forward.rs:2778`)
```rust
pub struct GpuKVCache {
    pub k_cache_bf16: GpuBuf<u16>,    // [max_len, d] bf16
    pub v_cache_bf16: GpuBuf<u16>,    // [max_len, d] bf16
    pub len: usize,                   // current filled positions
    pub max_len: usize,               // capacity
    pub d: usize,
    scratch_k_bf16: GpuBuf<u16>,      // persistent scratch
    scratch_v_bf16: GpuBuf<u16>,
}
```
Methods: `new(max_len, d, scratch_tokens)`, `append_f32(k, v, n_tokens)`, `reset()`.

### 2. swa_single_token_cuda (`core/src/cuda_ffi.rs:45`)
```rust
pub(crate) fn swa_single_token_cuda(
    q: *const u16,           // [1, total_dim] bf16
    k_cache: *const u16,     // [cache_len, total_dim] bf16
    v_cache: *const u16,     // [cache_len, total_dim] bf16
    out: *mut u16,           // [1, total_dim] bf16
    cache_len: i32,
    num_heads: i32,
    head_dim: i32,
    window_size: i32,
);
```
Already compiled and linked. Used by the single-block `GpuModel::decode_token`.

### 3. gpu_memory_forward with s=1
`gpu_memory_forward` takes `s: usize` as a parameter. Nothing prevents `s=1`. The per-token memory update (project → error → M update → readout) works for any sequence length.

## Architecture

### Phase 1: Prefill

Process the full prompt (up to `seq_len` tokens) through the existing `gpu_stacked_forward`. This populates:

- **M and S** for all levels in all blocks (via `GpuStackedContext`)
- **Per-block KV caches** (NEW) — one `GpuKVCache` per block, filled from the SWA Q/K/V projections during the forward pass

The prefill is the existing forward path with two additions:
1. Allocate and fill per-block KV caches from the K/V projections
2. Skip the loss computation (we only need logits from the last position)

```
Prefill output:
  - context.memory[block][level] = M state after processing full prompt
  - kv_caches[block] = last window_size K/V pairs from SWA
  - last_position_logits for the first sampling step
```

### Phase 2: Decode (per token)

For each generated token, process through all blocks sequentially:

```
Input: single token ID, Conductor pulse

For each block b in 0..n_blocks:
  1. Embed token → x_t [1, d]                          (reuse shared W_embed)
     (block 0 only — subsequent blocks use residual from previous block)

  2. LN_attn(x_t) → ln_out [1, d]

  3. QKV projection:
     q = ln_out @ W_Q^T    [1, d]
     k = ln_out @ W_K^T    [1, d]
     v = ln_out @ W_V^T    [1, d]

  4. Append (k, v) to kv_caches[b]                     (GpuKVCache::append_f32)

  5. SWA attention:
     Convert q to bf16
     swa_single_token_cuda(q_bf16, kv_caches[b], out)  (existing kernel)
     Convert out to f32

  6. Output projection: attn_out = swa_out @ W_O^T     [1, d]

  7. Residual skip 1: x_t = x_t + attn_out

  8. LN_mem(x_t) → ln_mem_out [1, d]

  9. CMS memory (for each level):
     Check pulse.active_levels[level]:
       If active:
         - In chain mode: pool ln_mem_out (or previous level output) to s_f tokens
           (at decode, s=1 and s_f=1 for all levels — no pooling needed)
         - gpu_memory_forward(level_params, cfg, &ln_mem_out, &mut context.memory[b][level],
                              s=1, d, level, batch_size=1)
         - Returns y_level [1, d]
       If inactive:
         - gpu_memory_read_only(level_params, &ln_mem_out, &context.memory[b][level],
                                s=1, d, num_heads, head_dim)
         - Returns y_level [1, d]

  10. Composition (MAG gate):
      y_combined = weighted_sum(y_per_level) or last chain output
      gate = sigmoid(y_combined)
      gated_out = attn_proj * gate
      (Same MAG sigmoid gating as training forward — NOT a blend)

  11. Residual skip 2: x_t = block_input + gated_out
      → This becomes the input to block b+1

After all blocks:
  12. Final LN → x_final [1, d]
  13. Logits = x_final @ W_unembed  [1, vocab_size]
  14. Conductor.advance()
```

### CMS Level Firing During Decode

The Conductor advances per decode token. CMS levels fire at their normal frequencies:
- L0 (chunk_size=1): fires every token → M updates every decode step
- L1 (chunk_size=8): fires every 8th decode token → M updates occasionally
- L2 (chunk_size=64): fires every 64th token → rare M update
- L3 (chunk_size=512): fires every 512th token → very rare

This is identical to training. No special handling needed.

**Critical: at decode time, s=1 for ALL levels regardless of token reduction.**
Token reduction (spec 46) reduces s_f = s / chunk_sizes[level]. When s=1, integer division 1/C = 0 for C > 1, so the implementation clamps s_f = max(1, s / chunk_sizes[level]). This is a decode-specific floor: every level always processes at least 1 token. No pooling/upsampling occurs during decode — each level processes the single token directly.

### Chain Mode at Decode

In chain mode, each level processes the previous level's output. At s=1:
- L0 input: ln_mem_out [1, d] → L0 output: y_0 [1, d]
- L1 input: y_0 [1, d] → L1 output: y_1 [1, d]
- L2 input: y_1 [1, d] → L2 output: y_2 [1, d]
- L3 input: y_2 [1, d] → L3 output: y_3 [1, d]

No pooling needed (h_seq_len == s_f == 1 at every level).

## Design Decisions

### D1: Prefill Reuses Existing Forward

**Decision**: Prefill calls the existing `gpu_stacked_forward` (or a thin wrapper) rather than implementing a separate prefill path.

**Rationale**: CS-10 — no mode distinction. The prefill IS a forward pass on seq_len tokens. The only additions are (a) extracting K/V into caches, and (b) skipping loss computation. This avoids duplicating 500+ lines of forward logic.

### D2: KV Cache Lifetime

**Decision**: KV caches are allocated at prefill, persist across decode tokens, and freed on `reset_cache()`.

**Rationale**: Same pattern as existing `GpuModel` (single-block). The cache represents the SWA window — it fills during prefill and rotates during decode (oldest entries evicted when full).

### D3: Conductor Continuity

**Decision**: The Conductor state from training (or from prefill) carries into decode. No reset.

**Rationale**: The model is continuous (CS-10). If the model was at step 50000 when the coherence probe runs, the Conductor should continue from step 50000. CMS level firing frequencies must be consistent.

### D4: No Backward During Decode (for coherence probes)

**Decision**: The coherence probe (eval_coherence_samples) uses decode without backward. The tape is not active.

**Rationale**: Coherence probes are observation-only. The model's M updates during decode (inner loop), but no outer-loop gradient computation occurs. This is the same as CS-10's "forward code identical in all phases" — the forward path runs, the backward doesn't. Self-modifying inference (generate_learning) would activate the tape, but that's a separate concern.

### D5: Per-Block KV Cache vs Shared

**Decision**: One KV cache per block. Each block has its own SWA attention with its own K/V projections.

**Rationale**: Block b's K/V projections come from block b's W_K, W_V — they're different weight matrices. The caches cannot be shared.

## Implementation

### Rust: `gpu_stacked_decode_token`

New function in `core/src/gpu_stacked_forward.rs`:

```rust
pub fn gpu_stacked_decode_token(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_id: usize,
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    kv_caches: &mut [GpuKVCache],     // one per block
) -> Vec<f32>  // logits [vocab_size]
```

### Rust: `gpu_stacked_prefill`

Wrapper around `gpu_stacked_forward` that also fills KV caches:

```rust
pub fn gpu_stacked_prefill(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    kv_caches: &mut [GpuKVCache],     // one per block, pre-allocated
    profiler: &mut Option<GpuProfiler>,
) -> Vec<f32>  // last-position logits [vocab_size]
```

### PyO3: GpuStackedModel methods

```python
# New methods on GpuStackedModel:
def prefill(self, input_ids: list[int], pulse) -> list[float]:
    """Process full prompt, populate KV caches + M state. Returns last-position logits."""

def decode_token(self, token_id: int, pulse) -> list[float]:
    """Process single token through all blocks. Returns logits [vocab_size]."""

def reset_cache(self) -> None:
    """Free KV caches and reset decode state."""
```

### Python: Update `generate_stacked`

```python
def generate_stacked(gpu_model, cfg, prompt_tokens, max_tokens, ...):
    # Prefill: process prompt, get initial logits
    pulse = conductor.pulse()
    last_logits = gpu_model.prefill(ctx, pulse)
    conductor.advance()

    for _ in range(max_tokens):
        next_tok = _sample_token(last_logits, vocab, temperature, top_k)
        seq.append(next_tok)

        pulse = conductor.pulse()
        last_logits = gpu_model.decode_token(next_tok, pulse)
        conductor.advance()
```

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | Add `gpu_stacked_prefill` + `gpu_stacked_decode_token` |
| `python/src/lib.rs` | Add `prefill()`, `decode_token()`, `reset_cache()` to `GpuStackedModel` |
| `python/engine/generation.py` | Update `generate_stacked` to use prefill/decode path |

### Files NOT Modified

| File | Why unchanged |
|------|---------------|
| `core/src/gpu_forward.rs` | `gpu_memory_forward` already works with s=1; `GpuKVCache` + `swa_single_token_cuda` already exist |
| `core/kernels/*.cu` | All CUDA primitives exist |
| `core/src/conductor.rs` | Pulse generation unchanged |
| `core/src/model.rs` | No config changes |
| `python/engine/config.py` | No config changes |

## Performance Model

### Current (generate_stacked with full replay)
- Per token: full forward (512 tokens × 6 blocks) = ~500ms
- 20 tokens: ~10 seconds
- Throughput: ~37 tok/s

### Proposed (prefill + decode)
- Prefill: one full forward = ~500ms (amortized over all decode tokens)
- Per decode token:
  - Embedding lookup: ~0.01ms
  - Per block (×6):
    - LN + QKV proj: ~0.1ms (single vector matmul)
    - SWA single-token attention: ~0.05ms (against 512 cached K/V)
    - Memory forward (s=1): ~0.08ms (single M update + readout)
    - Composition: ~0.01ms
  - Final LN + unembed: ~0.1ms
  - **Total per decode: ~1.5ms**
- 20 tokens: 500ms prefill + 20 × 1.5ms = ~530ms
- Throughput: **~13,000 tok/s decode** (after prefill)

Conservative estimate accounting for kernel launch overhead and memory latency: **>2,000 tok/s decode**.

## Acceptance Criteria

1. `gpu_model.prefill(prompt, pulse)` returns logits identical to `gpu_model.forward(prompt, target, pulse)[1]` for the last position
2. `gpu_model.decode_token(tok, pulse)` produces the same M state as processing the token via full forward
3. Coherence probe at step 1024 completes in <2 seconds (vs ~30 seconds currently)
4. Decode throughput >1000 tok/s on A6000 at d=768, 6 blocks, k=4
5. CMS levels fire correctly during decode (verified via HECATE_DEBUG_CMS)
6. Existing training throughput unaffected (decode code is separate path, not modifying forward)
7. `reset_cache()` frees KV caches and returns model to pre-prefill state
8. Chain mode (chained CMS) works correctly at s=1 (no pooling, direct chain)

## Ontological Compliance

- **CS-10**: No mode flag. `decode_token` runs the same M update as training. No `if inference:` branch. The only difference is s=1 and no backward.
- **CS-18**: All math in Rust tier. Python passes token ID and receives logits.
- **CS-32**: Observe-then-advance. M observes the token (M@k - v error), then advances (M update). Same as training.
- **CS-40**: Tape not activated during decode (no backward needed for coherence probes). Self-modifying inference (future) would activate tape — same opt-in contract.
- **CS-44**: fp32 unconditional in inner loop. M update at s=1 is the same fp32 math.

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-034-deltanet-update | titans_equations | Titans §3 (2501.00663) | implements |
| eq-070-arch-variant1 | hope_equations | HOPE §6 Eq 70 | implements |
| eq-097-hope-cms-chain | hope_equations | HOPE §7 Eq 97 | cites |

## Falsification Criteria

1. **Memory divergence**: If M state after decode differs from M state after full-forward replay of the same tokens, the decode path has a bug. Test by comparing M norms after N decode steps vs N full-forward steps on the same input.
2. **SWA cache correctness**: If attention output from the cache diverges from recomputed attention, the cache rotation has a bug. Test on sequences longer than window_size.
3. **CMS level mismatch**: If decode doesn't fire CMS levels at the same steps as training, the Conductor integration is wrong. Verify with HECATE_DEBUG_CMS.
4. **Performance regression**: If decode is not significantly faster than full replay (>10x), something is serializing unnecessarily. Profile with spec 38 heatmap.
