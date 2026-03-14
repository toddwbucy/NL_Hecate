# Multi-Block Stacking — Execution Engine for Stacked SWA+CMS Blocks

```
CONTRACT
  Purpose:    Execution engine for N stacked HOPE blocks (SWA + CMS), enabling
              depth-through-blocks at scale. Each block has its own SWA
              projections, LayerNorms, and k CMS frequency levels. Embeddings
              and lm_head are shared. Residual stream connects all blocks.
  Expects:    Existing single-block GPU forward/backward (gpu_forward.rs,
              gpu_backward.rs). CMS variant config schema (cms_variants.rs).
              Conductor/Pulse frequency scheduling. CUDA kernels for SWA,
              LayerNorm, memory rules, cross-entropy.
  Guarantees: n_blocks=1 produces bit-identical output to existing single-block
              path (backward compat). n_blocks>1 chains blocks via residual
              stream with pre-LN normalization. All existing CUDA kernels
              reused — no new kernels required. Pulse applies uniformly to all
              blocks (same CMS firing schedule across depth).
  Cost:       N × single-block forward/backward compute. Activation memory:
              N × ~15 GPU buffers at [bs×s, d]. At d=512, k=4, s=512, bs=1,
              N=4: ~125MB activations. At d=1536, N=12: ~560MB.
  Trade-off:  Depth vs memory. More blocks = richer representations for slow
              CMS levels, but linear increase in compute and activation VRAM.
              Gradient checkpointing (future) can trade compute for memory.
  Position:   specs/infrastructure/14_multi_block_stacking.md
  Source:     HOPE (2512.24695) Section 6 — HOPE architecture with CMS.
              Section 5.1 — CMS as chain of MLP blocks.
              Eq. output-cms (sequential CMS output computation).
```

## Motivation

Single-block CMS (1 SWA + k=4 levels) caps at ~60-90M params regardless of
d_model. The HOPE paper trains at 760M and 1.3B by stacking N blocks — each
block is an independent SWA + CMS unit connected via residual stream. More
critically, single-block k=4 produces dead L3 (EXP-09, EXP-10, EXP-15a-c) —
all 4 CMS levels receive the same unrefined embedding, giving slow levels
insufficient signal. Multi-block stacking provides depth: block N's CMS levels
see input refined by blocks 0..N-1, making slow-frequency memory meaningful.

## Architecture

```text
input_ids [bs × s]
    │
    ▼ embedding_gather (shared w_embed)
x: [bs×s, d]
    │
    ┌─────────────────── Block 0 ───────────────────────┐
    │  x_ln = LN_attn(x)          pre-norm              │
    │  q,k,v = x_ln @ W_Q/K/V                           │
    │  attn = SWA(q, k, v)        bf16 sliding window    │
    │  x = x + attn               residual skip 1        │
    │  x_mem = LN_mem(x)          pre-norm for memory    │
    │  for level l in 0..k:                               │
    │    y_l = memory_forward(x_mem, M[0][l], pulse)     │
    │    M[0][l] ← updated (inner-loop state)            │
    │  y = aggregate(y_levels)    1/sqrt(k) or learned    │
    │  x = x + y                  residual skip 2         │
    └───────────────────────────────────────────────────┘
    │
    x flows to Block 1 (no inter-block LN — pre-LN handles it)
    │
    ┌─────────────────── Block 1 ───────────────────────┐
    │  (same structure, own params, own M states)        │
    └───────────────────────────────────────────────────┘
    │
    ... (blocks 2..N-1)
    │
    ▼ LN_final(x)                 final layer norm
    ▼ logits = x @ w_unembed      shared lm_head
    ▼ cross_entropy_loss
```

### What Is Per-Block

Each block has its own independent:
- SWA projections: W_Q, W_K, W_V, W_O (4 × d²)
- LayerNorm params: LN_attn (gamma, beta), LN_mem (gamma, beta)
- CMS levels (k per block): W_K_mem, W_V_mem, W_Q_mem, gates, w_omega per level
- CMS aggregation: alpha_mem[k], alpha_refl[k]
- Inner-loop M states: context.memory[block_idx][level] (d × d per level)

### What Is Shared

- Embedding table: w_embed [vocab × d]
- Output projection: w_unembed [d × vocab]
- Final LayerNorm: LN_final (gamma, beta) — applied once after last block
- Conductor/Pulse: one Pulse per step, uniform across all blocks

### Residual Stream

Pre-LN architecture (standard in modern transformers):
- LN before attention, LN before memory — NOT after
- Two residual skips per block: x += attn_out, x += mem_out
- Gradient flows through the residual stream unimpeded across all blocks
- `residual=true` is REQUIRED for n_blocks > 1 (enforced in config validation)

### Pulse/Conductor Interaction

The Conductor generates one Pulse per training step. This same Pulse applies
to ALL blocks uniformly:
- Level 0 fires in blocks 0..N-1 at every step
- Level 1 fires in blocks 0..N-1 every 8 steps
- Level 2 fires in blocks 0..N-1 every 64 steps
- Level 3 fires in blocks 0..N-1 every 512 steps

Each block maintains independent M states — block 0's Level 2 and block 5's
Level 2 evolve from different inputs (different depths in the residual stream).
The same frequency schedule means they fire simultaneously but update
differently because their inputs differ.

## Parameter Count

Per block at dimension d with k CMS levels:
```
SWA:      4 × d²                    (W_Q, W_K, W_V, W_O)
LN:       4 × d                     (attn gamma/beta + mem gamma/beta)
Per level: 3 × d² + 2 × d² + 6d + 3  (projections + omega + gates)
         ≈ 5 × d²
k levels: 5k × d²
Block total: (4 + 5k) × d² + 4d
```

Shared:
```
Embedding:  vocab × d
LM head:    d × vocab
Final LN:   2 × d
```

### Scale targets:

| Config | d | n_blocks | k | Vocab | Total params | Use |
|--------|---|----------|---|-------|-------------|-----|
| Shakedown | 512 | 4 | 4 | 50257 | ~103M | A6000 validation |
| Shakedown-6 | 512 | 6 | 4 | 50257 | ~128M | A6000 validation |
| Paper-match | 1536 | 12 | 4 | 32768 | ~780M | H100 training |
| Paper-large | 2048 | 12 | 4 | 32768 | ~1.35B | H100 training |

## Data Structures

### Rust Core

```rust
/// Per-block parameters. SWA projections (no embed/unembed) + CMS levels.
pub struct BlockParams {
    // SWA projections for this block
    pub w_q: Vec<f32>,          // [d, d]
    pub w_k: Vec<f32>,          // [d, d]
    pub w_v: Vec<f32>,          // [d, d]
    pub w_o: Vec<f32>,          // [d, d]
    // Pre-norm LayerNorm for attention branch
    pub ln_attn_gamma: Vec<f32>,  // [d]
    pub ln_attn_beta: Vec<f32>,   // [d]
    // Pre-norm LayerNorm for memory branch
    pub ln_mem_gamma: Vec<f32>,   // [d]
    pub ln_mem_beta: Vec<f32>,    // [d]
    // CMS memory levels (length k)
    pub levels: Vec<MemoryLevelParams>,
    // CMS aggregation logits
    pub alpha_mem: Vec<f32>,    // [k]
    pub alpha_refl: Vec<f32>,   // [k]
}

/// Stacked model: shared embed/unembed + N independent blocks + final LN.
pub struct StackedMAGParams {
    pub w_embed: Vec<f32>,      // [vocab, d]
    pub w_unembed: Vec<f32>,    // [d, vocab]
    // Final LayerNorm (after last block, before lm_head)
    pub ln_final_gamma: Vec<f32>,  // [d]
    pub ln_final_beta: Vec<f32>,   // [d]
    // N blocks
    pub blocks: Vec<BlockParams>,
}
```

### GPU Params

```rust
pub struct GpuBlockParams {
    pub w_q: GpuBuf<f32>,
    pub w_k: GpuBuf<f32>,
    pub w_v: GpuBuf<f32>,
    pub w_o: GpuBuf<f32>,
    pub ln_attn_gamma: GpuBuf<f32>,
    pub ln_attn_beta: GpuBuf<f32>,
    pub ln_mem_gamma: GpuBuf<f32>,
    pub ln_mem_beta: GpuBuf<f32>,
    pub levels: Vec<GpuMemoryLevelParams>,
    pub alpha_mem: GpuBuf<f32>,
    pub alpha_refl: GpuBuf<f32>,
}

pub struct GpuStackedParams {
    pub w_embed: GpuBuf<f32>,
    pub w_unembed: GpuBuf<f32>,
    pub ln_final_gamma: GpuBuf<f32>,
    pub ln_final_beta: GpuBuf<f32>,
    pub blocks: Vec<GpuBlockParams>,
}
```

### GPU Context State

```rust
/// Extended context state for multi-block models.
/// memory[block_idx][level] holds the d×d M matrix for that block+level.
pub struct GpuStackedContext {
    pub memory: Vec<Vec<GpuBuf<f32>>>,  // [n_blocks][k]
}
```

### GPU Cache (Forward Activations for Backward)

```rust
pub struct GpuBlockCache {
    // Attention branch activations
    pub x_pre_attn: GpuBuf<f32>,     // input to this block
    pub x_after_ln_attn: GpuBuf<f32>,
    pub q_f32: GpuBuf<f32>,
    pub k_f32: GpuBuf<f32>,
    pub v_f32: GpuBuf<f32>,
    pub attn_out: GpuBuf<f32>,
    pub x_after_skip1: GpuBuf<f32>,
    // Memory branch activations
    pub x_after_ln_mem: GpuBuf<f32>,
    pub level_caches: Vec<GpuMemoryCache>,
    pub mem_combined: GpuBuf<f32>,
    pub x_after_skip2: GpuBuf<f32>,  // = block output
}

pub struct GpuStackedCache {
    pub embedded: GpuBuf<f32>,        // post-embedding, pre-block-0
    pub x_final: GpuBuf<f32>,         // after last block + final LN
    pub block_caches: Vec<GpuBlockCache>,
    pub logits: GpuBuf<f32>,
    pub input_ids: GpuBuf<i32>,
    pub target_ids: GpuBuf<i32>,
}
```

## Checkpoint Format

Tensor keys gain a `block.{b}.` prefix:

```
embed.weight                    (shared)
lm_head.weight                  (shared)
ln_final.gamma                  (shared)
ln_final.beta                   (shared)
block.0.swa.w_q                 (per-block)
block.0.swa.w_k
block.0.swa.w_v
block.0.swa.w_o
block.0.ln_attn.gamma
block.0.ln_attn.beta
block.0.ln_mem.gamma
block.0.ln_mem.beta
block.0.level.0.w_k
block.0.level.0.w_v
block.0.level.0.w_q
block.0.level.0.gate.alpha
block.0.level.0.gate.theta
block.0.level.0.gate.eta
block.0.level.0.gate.b_alpha
block.0.level.0.gate.b_theta
block.0.level.0.gate.b_eta
block.0.level.0.w_omega
block.0.level.0.m_state.mem
block.0.alpha_mem
block.0.alpha_refl
block.1.swa.w_q                 (block 1...)
...
```

### Backward Compatibility

Auto-detection: if safetensors header contains `block.0.swa.w_q`, use stacked
loader. If it contains `swa.w_q` (no block prefix), use single-block loader.
Migration utility renames single-block keys to `block.0.*` prefix.

## Config

### JSON config format

```json
{
    "model": {
        "d_model": 512,
        "num_heads": 8,
        "seq_len": 512,
        "window_size": 512,
        "vocab_size": 50257,
        "memory_rule": "titans",
        "composition": "mag",
        "k": 4,
        "chunk_sizes": [1, 8, 64, 512],
        "n_blocks": 4,
        "residual": true,
        "m_norm_max": [100.0, 100.0, 100.0, 100.0]
    }
}
```

`n_blocks` defaults to 1 for backward compat. When n_blocks > 1:
- `residual` is forced true (error if explicitly set false)
- All blocks use the same config (uniform layout, simplest deployment variant)
- Non-uniform per-block configs are a future extension (Sequential/Hybrid variants)

### Python config.py

```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    n_blocks: int = 1  # Number of stacked SWA+CMS blocks
```

Validation:
- n_blocks >= 1
- if n_blocks > 1 and not residual: error
- if n_blocks > 1: log total param count at startup

## Backward Pass

Reverse block loop with residual gradient accumulation:

```text
d_logits = cross_entropy_backward(logits, targets)
d_x = d_logits @ w_unembed^T
d_w_unembed += x_final^T @ d_logits

d_x = LN_final_backward(d_x)  → d_ln_final_gamma, d_ln_final_beta

for block_idx in (0..n_blocks).rev():
    cache = block_caches[block_idx]

    // Residual skip 2 backward: gradient flows to both branches
    d_mem_combined = d_x        // memory branch gradient
    d_skip2 = d_x               // skip gradient (passes through)

    // Memory branch backward
    for level in 0..k:
        d_level_params, d_x_mem_l = memory_backward(d_mem_l, level_cache)
    d_x_mem = aggregate_backward(d_x_mem_levels)
    d_x_mem, d_ln_mem_gamma, d_ln_mem_beta = LN_mem_backward(d_x_mem)

    // Combine: add skip-2 gradient + memory branch gradient
    d_x = d_skip2 + d_x_mem

    // Residual skip 1 backward
    d_attn = d_x                // attention branch gradient
    d_skip1 = d_x               // skip gradient

    // SWA backward
    d_q, d_k, d_v = swa_backward(d_attn, attn_cache)
    d_x_ln = d_q @ W_Q^T + d_k @ W_K^T + d_v @ W_V^T
    d_w_q += x_ln^T @ d_q  (etc. for K, V, O)
    d_x_ln, d_ln_attn_gamma, d_ln_attn_beta = LN_attn_backward(d_x_ln)

    // Combine: add skip-1 gradient + attention branch gradient
    d_x = d_skip1 + d_x_ln

// After all blocks: d_x is the embedding gradient
d_w_embed = scatter_backward(d_x, input_ids)
```

## Build Phases

### Phase 1: Rust structs (no GPU)
- `BlockParams`, `StackedMAGParams` in stacked_model.rs
- init, zeros_like, num_params
- Stacked checkpoint save/load with `block.{b}.*` keys
- Migration utility for single-block checkpoints
- Tests: param count, checkpoint round-trip, migration

### Phase 2: GPU params + context
- `GpuBlockParams`, `GpuStackedParams` in gpu_params.rs
- `GpuStackedContext` with [n_blocks][k] memory buffers
- from_host / to_host round-trip tests

### Phase 3: GPU stacked forward
- `gpu_stacked_forward` — block loop calling existing kernels
- `GpuStackedCache` capturing all per-block activations
- Test: n_blocks=1 matches existing single-block forward (regression)
- Test: n_blocks=2 produces finite loss

### Phase 4: GPU stacked backward
- Reverse block loop, residual gradient accumulation
- Reuses existing memory_backward, swa_backward, LN_backward kernels
- Test: n_blocks=1 gradients match single-block backward
- FD gradient check at n_blocks=2, d=8, k=1

### Phase 5: PyO3 + Python integration
- StackedConfig, StackedParams wrappers
- gpu_stacked_forward/backward bindings
- config.py: n_blocks field
- loop.py: routing branch for stacked vs single-block
- Integration test: 10-step training with loss decrease

### Phase 6: Shakedown config
- `stacked_shakedown_4block.json`: d=512, n_blocks=4, k=4
- Validate on A6000: all 4 CMS levels active across all blocks
- Compare L3 activation vs single-block baseline

## Ontological Compliance

- **CS-04/05/06**: "blocks" for architectural depth, "levels" for CMS frequency
  hierarchy. NEVER "layers" for either.
- **CS-10**: No train/eval mode distinction — stacked forward is identical in
  all phases.
- **CS-18**: Orchestration (block loop, gradient accumulation) in Rust tier,
  NOT Python tier. Python only calls gpu_stacked_forward/backward.
- **CS-32**: Observe-then-advance — each block's memory update follows the
  same observe-then-advance pattern as single-block.
- **CS-40**: Opt-in AD — Wengert tape wraps the full stacked forward when
  tape diagnostics are requested.

## Axiom Compliance

- **HOPE Eq. output-cms**: y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))
  — sequential CMS. Our stacked architecture generalizes this: N blocks each
  with k levels, residual connections between blocks.
- **HOPE Section 6**: "we suggest initializing the parameters in each level
  independently" — each block initialized with independent seeds.
- **NL IS**: Multi-block stacking IS nested learning — each block is a
  nested optimization problem with its own parameters and inner-loop states.
