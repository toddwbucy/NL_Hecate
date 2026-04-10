# 79 — MAC Composition for Stacked Multi-Block Forward/Backward

```text
CONTRACT
  Purpose:    Replace MAG sigmoid gating with MAC (Memory As Context) composition
              in the stacked multi-block GPU forward/backward path. MAC lets memory
              contribute information as context tokens that attention reads alongside
              raw input, then writes attention output back to memory with a reflective
              gate. This addresses the MAG limitation where memory can only gate
              attention output (sigmoid [0,1]) but never inject its own knowledge into
              the residual stream.
  Expects:    Working stacked multi-block forward/backward (gpu_stacked_forward.rs,
              gpu_stacked_backward.rs). Existing MAC reference implementation (mac.rs)
              with forward + backward for single-block CPU path. Existing SWA CUDA
              kernel that supports window_size >= sequence_length (full causal).
              Existing sigmoid_cuda, elemwise_mul_cuda, gpu_memory_read_only,
              gpu_memory_forward functions.
  Guarantees: When composition="mac" in config:
                1. Memory READ produces context tokens h_t per block
                2. Assembled input [persistent || h_t || normed] fed to full causal attention
                3. Attention output y_t written back to memory (feedback loop)
                4. Reflective gate: y_t * sigmoid(memory.READ(y_t))
                5. Single residual: block_input + W_O(gated_out)
              When composition="mag": behavior unchanged (backward compatible).
              CPU reference (mac.rs) and GPU stacked path produce matching outputs
              within tolerance (forward: 1e-5, backward: 1e-4).
  Cost:       Per block: ~4x attention FLOPs vs MAG-SWA at L_seg=512 (assembled
              ~1024 tokens, full causal O(1024^2*d) vs SWA O(512*512*d)).
              Three memory operations per block (READ + WRITE + reflective READ)
              vs one memory forward in MAG. Total overhead: ~10-15% wall time.
              VRAM: ~1-2 GB additional for assembled context caches per block.
  Trade-off:  Memory becomes a first-class information contributor (context tokens)
              instead of a sigmoid dimmer switch. Attention can learn to read memory
              context OR ignore it (safe fallback to raw input). The feedback loop
              (write y_t → reflective READ) enables memory-attention co-evolution.
              Cost: full causal attention + 3 memory ops vs SWA + 1 memory op.
  Position:   specs/infrastructure/79_mac_stacked_composition.md
  Source:     Titans (2501.00663) Section 4.1, Eqs 21-25 (MAC composition);
              specs/algorithms/composition_patterns/01_mac.md (algorithm spec);
              core/src/mac.rs (CPU reference implementation)
```

## Motivation

MAG composition (current default) passes memory output through sigmoid to produce
a [0,1] gate on attention output. Memory never contributes information directly —
it can only pass or suppress attention. At 50% Chinchilla optimal (1.6B of 3.2B
tokens), the d=1024 model plateaued at loss 4.94. The hypothesis: memory needs a
richer information channel to attention for the loss curve to continue descending.

MAC provides this by making memory output into context tokens that attention reads
in its normal QKV mechanism — the same way Gemma 4's global attention layers
contribute directly to the residual stream through their global layers.

## Per-Block Data Flow (MAC)

```text
Input: residual [s, d] from previous block (or embedding for block 0)

1. block_input = residual                     -- save for skip connection
2. normed = LN(residual)                      -- single pre-norm (γ_b, β_b)

   ── Memory READ (context tokens) ──
3. h_t_per_level = []
   FOR each CMS level l:
     h_t_l = M_l @ W_Q_mem @ normed           -- read-only, [s, d]
   h_t = aggregate(h_t_per_level)             -- weighted sum, [s, d]

   ── Assemble ──
4. assembled = concat(persistent, h_t, normed) -- [n_p + 2s, d]

   ── Full Causal Attention ──
5. Q, K, V = assembled @ W_Q, W_K, W_V        -- [n_p + 2s, d]
6. attn_out = full_causal_attn(Q, K, V)        -- SWA with window >= n_p + 2s
7. y_t = attn_out[n_p + s ..]                  -- extract segment portion [s, d]

   ── Memory WRITE (feedback) ──
8. FOR each active CMS level l:
     reflective_y_l = memory_step(y_t, level=l)  -- updates M_l, returns M_l @ q(y_t)
   reflective_y = aggregate(reflective_y_per_level)

   ── Reflective Gate ──
9. gated_out = y_t * sigmoid(reflective_y)      -- [s, d]

   ── Output ──
10. projected = gated_out @ W_O                  -- [s, d]
11. residual = block_input + projected           -- single skip connection

Output: residual [s, d] → next block
```

## Comparison with Current MAG Flow

```text
MAG (current):                          MAC (proposed):
─────────────────────                   ─────────────────────
LN_attn(residual)                       LN(residual)
  ↓                                       ↓
QKV on [persistent || normed]           Memory READ → h_t
  ↓                                       ↓
SWA attention                           Assemble [persistent || h_t || normed]
  ↓                                       ↓
W_O → attn_proj                         Full causal attention (QKV on assembled)
  ↓                                       ↓
residual += attn_proj                   Extract y_t from segment portion
  ↓                                       ↓
LN_mem(residual)                        Memory WRITE(y_t) → reflective_y
  ↓                                       ↓
CMS memory → y_combined                y_t * sigmoid(reflective_y) → gated_out
  ↓                                       ↓
sigmoid(y_combined) → gate              W_O(gated_out) → projected
  ↓                                       ↓
attn_proj * gate → gated_out           residual += projected
  ↓
residual = block_input + gated_out

Key differences:
- MAG: 2 LayerNorms, 2 residual adds, memory only gates
- MAC: 1 LayerNorm, 1 residual add, memory provides context + reflective feedback
- MAG: memory sees LN_mem(residual_after_attn) — post-attention
- MAC: memory READ sees normed input (pre-attention), WRITE sees y_t (post-attention)
- MAG: SWA (local window)
- MAC: full causal over assembled (memory context + raw input)
```

## Attention: SWA Kernel as Full Causal

No new CUDA kernel required. The existing SWA kernel with `window_size >= assembled_length`
computes full causal attention. The MAC reference (`mac.rs:283`) already uses this:

```rust
assert!(ws >= s_total, "MAC requires window_size >= n_persistent + 2*seq_len");
```

For the stacked path, the effective attention window per block is computed as:

```text
assembled_len = n_persistent + 2 * s
mac_window = assembled_len    -- full causal = window covers entire assembled sequence
```

The config's `window_size` field retains its meaning for MAG (sliding window).
For MAC, the code overrides the effective window to `assembled_len` regardless
of the configured `window_size`. Config validation enforces:

```text
IF composition == "mac":
  effective_window = n_persistent + 2 * seq_len
  -- config.window_size is informational only (ignored)
```

## Memory Operations

### READ (Step 3 — context tokens)

Uses the existing `gpu_memory_read_only()` function. For each CMS level:

```text
q_mem = normed @ W_Q_mem_l         -- project to memory query space
h_t_l = M_l @ q_mem                -- matrix-vector, read-only (no M update)
```

All levels contribute h_t regardless of Pulse activity (frozen levels still read).
Aggregation: softmax-weighted sum (same as current level aggregation using alpha_mem).

### WRITE (Step 8 — feedback + reflective)

Uses the existing `gpu_memory_forward()` function on y_t (attention output).
This both updates M and produces the forward output for the reflective gate:

```text
-- For each active level l:
k_mem = y_t @ W_K_mem_l
v_mem = y_t @ W_V_mem_l
error = v_mem - M_l @ k_mem
M_l += alpha_l * error @ k_mem^T    -- Titans LMM update rule
S_l = beta * S_l + (1-beta) * error @ k_mem^T  -- momentum
reflective_y_l = M_l @ (y_t @ W_Q_mem_l)      -- read from UPDATED M
```

Frozen levels: no write, accumulate gradients in ErrorBuffer. Reflective read
uses read-only path on frozen M (same as step 3 but with y_t as query).

### Reflective Gate (Step 9)

```text
gate = sigmoid(reflective_y)        -- [0, 1] per element
gated_out = y_t * gate              -- element-wise
```

This gates attention output by UPDATED memory's opinion of it. Different from MAG
where memory gates attention by its opinion of the RAW input.

## Block Parameters

### Changed from MAG

```text
MAG block params:                    MAC block params:
  ln_attn_gamma, ln_attn_beta          ln_gamma, ln_beta        -- single LN
  ln_mem_gamma, ln_mem_beta            (removed)
  w_q, w_k, w_v, w_o                  w_q, w_k, w_v, w_o       -- same
  per-level memory params              per-level memory params   -- same
  alpha_mem (level aggregation)        alpha_mem                 -- same
  persistent_tokens                    persistent_tokens         -- same
```

Single LayerNorm per block instead of two. Total parameter reduction per block:
2 * d (one fewer gamma + beta). Negligible.

### New cache fields per block

```text
pub struct GpuStackedBlockCache {
    // ... existing fields ...
    // MAC-specific:
    pub h_t: GpuBuf<f32>,              // [s, d] — memory context tokens
    pub assembled: GpuBuf<f32>,        // [n_p + 2s, d] — for backward
    pub y_t: GpuBuf<f32>,             // [s, d] — extracted attention output
    pub reflective_y: GpuBuf<f32>,    // [s, d] — reflective memory output
    pub reflective_gate: GpuBuf<f32>, // [s, d] — sigmoid(reflective_y)
}
```

## Configuration

### JSON config change

```json
{
    "model": {
        "composition": "mac",
        "n_persistent": 8,
        "window_size": 512,
        ...
    }
}
```

- `"composition": "mac"` selects MAC path (default remains `"mag"` for backward compat)
- `"n_persistent"`: number of learnable persistent tokens (default 0)
- `"window_size"`: informational for MAC (effective window = n_p + 2*s)

### Config validation

```text
IF composition == "mac":
  ASSERT n_persistent >= 0
  -- window_size is overridden internally; no constraint needed
  -- seq_len (= segments * window_size) still determines total token count
```

### Segment length for MAC

Each call to `forward_sequence` receives one segment of `L_seg` tokens (where
`L_seg = window_size` from config). The outer loop in Python calls forward_sequence
once per segment, advancing the data cursor by L_seg tokens each time. Over `segments`
calls, the model processes `segments * L_seg` total tokens per training step.

The `window_size` config field serves different roles per composition:
- For MAG: the SWA sliding window size (local attention within L_seg tokens)
- For MAC: the segment size L_seg (how many raw tokens per forward call)

The assembled attention window **per call** is `n_p + 2 * L_seg`. For the initial experiment:
- `window_size: 512` → L_seg = 512, assembled per call = 8 + 2*512 = 1032 tokens
- `segments: 20` → 20 forward calls per step, 10240 total tokens
- Attention cost per call: O(1032^2 * d) ≈ 4x vs MAG's SWA O(512 * 512 * d)

## Backward Pass

Reverse of forward, same structure as `mac_backward()` in `mac.rs`:

```text
11. d_projected ← d_residual (skip connection: d_block_input += d_residual)
10. d_gated_out = d_projected @ W_O^T; d_W_O += gated_out^T @ d_projected
 9. d_y_t_gate = d_gated_out * reflective_gate
    d_reflective_gate = d_gated_out * y_t
    d_reflective_y = d_reflective_gate * sigmoid'(reflective_y)
 8. Memory WRITE backward: d_y_t_write, d_memory_params from reflective_y backward
 7. d_y_t = d_y_t_gate + d_y_t_write    -- accumulate from gate + write paths
 6. d_attn_out = scatter d_y_t into positions [n_p+s..n_p+2s] of zeros [n_p+2s, d]
 5. SWA backward on assembled attention
 4. d_assembled from QKV backward
 3. Split d_assembled → d_persistent, d_h_t, d_normed
    d_normed += d_normed_from_assembled
 2. Memory READ backward: d_h_t → d_normed_from_read, d_memory_read_params
 1. LN backward → d_residual, d_ln_gamma, d_ln_beta
    d_block_input_total = d_block_input + d_residual
```

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_forward.rs` | Add MAC branch in per-block loop: memory READ, assemble, full causal attn, extract, memory WRITE, reflective gate. Dispatch on `cfg.composition`. |
| `core/src/gpu_stacked_backward.rs` | Add MAC backward branch mirroring forward. |
| `core/src/stacked_model.rs` | Handle single-LN params for MAC blocks. Param initialization. |
| `core/src/model.rs` | Ensure `MAGConfig` routes `composition` to forward path. |
| `cli/src/config.rs` | Parse `"composition": "mac"` from JSON config. |
| `cli/src/feed.rs` | Set effective window = assembled_len when composition=MAC. |

## Acceptance Criteria

1. `composition: "mac"` in config selects MAC forward/backward path
2. `composition: "mag"` (or omitted) gives identical behavior to current (no regression)
3. Memory READ produces context tokens that attention sees in assembled input
4. Memory WRITE updates M with attention output (feedback loop)
5. Reflective gate: y_t * sigmoid(memory.READ_after_write(y_t))
6. Full causal attention via SWA kernel with window >= assembled_len
7. Backward gradients correct: FD check on d=8 tiny model passes
8. All existing tests pass (MAG path unchanged)
9. New tests: MAC stacked forward/backward on tiny model

## Experiment Configuration (first run)

```json
{
    "description": "MAC composition experiment: L_seg=512, d1024, k=1, 8 blocks. Memory as context instead of gate. Full causal attention over assembled ~1032 tokens.",
    "model": {
        "d_model": 1024,
        "num_heads": 16,
        "segments": 20,
        "window_size": 512,
        "vocab_size": 49152,
        "memory_rule": "titans",
        "composition": "mac",
        "k": 1,
        "chunk_sizes": [64],
        "n_persistent": 8,
        "m_norm_max": [50.0],
        "error_clip": [1.0],
        "residual": true,
        "n_blocks": 8,
        "parallel_strategy": "tnt_hierarchical",
        "tnt_global_chunk_size": 64,
        "tnt_local_chunk_size": 8,
        "memory_reset": "none",
        "tape_multiplier": 1,
        "tape_strategies": ["exact"]
    },
    "build": {
        "optimizer": "adamw_gpu_stacked",
        "lr": 0.0001,
        "steps": 65104,
        "warmup_steps": 6000,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_grad_norm": 1.0,
        "alpha_floor": [0.8],
        "theta_ceil": [1.0],
        "batch_size": 3,
        "log_every": 50,
        "save_every": 1000,
        "gpu": true,
        "seed": 42
    },
    "data": {
        "path": "/home/todd/olympus/NL_Hecate/data-cache/dolmino_10b_shuffled",
        "format": "dolmino"
    }
}
```

Target: ~500M tokens to compare loss trajectory against MAG baseline at same token count.
Success signal: loss curve clearly descending where MAG flattened at 4.94.

## Dependencies

- BUG-01 (spec 18, merged): W_O output projection
- BUG-02 (spec 20, merged): MAG sigmoid gating (MAC replaces this per-block)
- Existing: gpu_memory_read_only, gpu_memory_forward, sigmoid_cuda, elemwise_mul_cuda
- Existing: SWA CUDA kernel (supports window >= seq_len for full causal)

## Ontological Compliance

- **CS-10**: No mode flag — MAC applied identically in all phases.
- **CS-18**: Memory READ/assemble/attention/WRITE are math in the Rust tier.
- **CS-32**: Observe-then-advance — READ before WRITE, reflective READ after WRITE.
- **CS-40**: Opt-in AD — assembled context, attention caches recorded on tape only when active.
- **Titans Eqs 21-25**: MAC composition faithfully implements the paper's specification.
