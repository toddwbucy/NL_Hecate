# CMS Intra-Segment Token Reduction

<!-- HADES: hope_equations/eq-070-arch-variant1 (HOPE §6 Eq 70); hope_equations/eq-071-arch-variant2 (HOPE §6 Eq 71); hope_equations/eq-074-arch-variant5 (HOPE §6 Eq 74); hope_equations/eq-097-hope-cms-chain (HOPE §7 Eq 97) -->

```text
CONTRACT
  Purpose:    Enforce the CMS frequency hierarchy WITHIN each segment by reducing
              the number of tokens each level processes. Currently all k CMS levels
              independently run the full memory recurrence over all seq_len tokens
              (e.g. 512 tokens × 4 levels = 2048 M updates). This spec makes each
              level process only seq_len / chunk_sizes[level] tokens:

                L0 (chunk_size=1):   512 tokens → 512 M updates
                L1 (chunk_size=8):    64 tokens →  64 M updates
                L2 (chunk_size=64):    8 tokens →   8 M updates
                L3 (chunk_size=512):   1 token  →   1 M update
                Total: 585 M updates (3.5× reduction from 2048)

              Higher levels receive mean-pooled summaries — groups of chunk_size
              consecutive input vectors averaged into a single vector. Each level
              operates at its natural timescale: L0 sees per-token detail, L1 sees
              8-token trends, L2 sees 64-token patterns, L3 sees segment context.

  Expects:    Stacked GPU forward/backward (gpu_stacked_forward.rs, gpu_stacked_backward.rs).
              Per-head memory kernels (spec 45). Existing CUDA forward/backward
              kernels that accept arbitrary seq_len. MAGConfig.chunk_sizes defining
              per-level frequencies. Conductor/Pulse scheduling (unchanged).
              Chain CMS (spec 35, HopeVariant::Chained) or Independent (FreqGated).

  Guarantees: - L0 (chunk_size=1): unchanged, processes all tokens.
              - Level f processes s_f = seq_len / chunk_sizes[f] tokens.
              - seq_len must be divisible by chunk_sizes[f] for all levels.
              - Mean pooling is the ONLY summarization method (no learned compression).
              - Existing CUDA kernels are reused with reduced seq_len — no kernel changes.
              - M state shape is unchanged: hd × hd per head (independent of seq_len).
              - context_m carry-forward is unchanged (M is seq_len-independent).
              - Output of each level is upsampled back to [seq_len, d] for aggregation.
              - When chunk_sizes = [1, 1, ..., 1], behavior is bit-identical to current.
              - Backward: gradients flow through pool (d_in = broadcast d_out / C)
                and upsample (d_in = sum of repeated d_out).

  Cost:       Forward: 585 M updates vs 2048 (3.5× reduction). Memory projections
              also reduced: L1 projects 64 tokens, not 512 (8× fewer FLOPs).
              Two new lightweight kernels (mean_pool, repeat_upsample) add ~0.1ms.
              Total memory_fwd time should drop ~3× from current baseline.
              Backward: proportional reduction. memory_bwd dominated by L0 (512 tokens);
              L1-L3 backward becomes negligible.
              VRAM: m_states buffers shrink (L1: 64×dd vs 512×dd, etc). Net VRAM
              savings: ~80% of L1/L2/L3 m_states allocation.

  Trade-off:  Higher levels see averaged representations, not individual tokens.
              Information is lost in the pooling step. This is BY DESIGN — slow
              levels should capture trends, not token-level detail. If a level
              needs finer resolution, its chunk_size should be reduced.
              Mean pooling is the simplest summarization. Learned compression
              (Lattice-style, per HOPE Eq 74 variants) is a future extension.

  Position:   specs/infrastructure/46_cms_token_reduction.md
  Source:     HOPE (2512.24695) §6 Eq 70 (chain CMS), Eq 71 (frequency gating),
              Eq 74 (independent CMS), §7 Eq 97 (HOPE CMS output).
              Diagnosed 2026-03-22 via H100 nvtop heatmap analysis (task_aa621b).
```

## Diagnosis: Why This Matters

### The Inverted Heatmap

Process heatmap at step 52224 (d=768, k=4, seq_len=512):

| Phase | Time (ms) | Expected |
|-------|-----------|----------|
| L0 backward | 49 | Should be MOST expensive (512 tokens) |
| L1 backward | 138 | Should be ~8× cheaper than L0 |
| L2 backward | 145 | Should be ~64× cheaper than L0 |
| L3 backward | 174 | Should be ~512× cheaper than L0 |

L0 is the cheapest because it uses exact backward (single kernel call). L1/L2/L3 use
chunkwise backward (multiple chunks, sequential GEMMs), which adds overhead. But all
four levels process the same 512 tokens — the chunk_sizes only control the chunkwise
approximation granularity, not the number of tokens processed.

**Root cause**: `gpu_memory_forward()` is called with `s=512` for ALL levels. The
chunk_sizes control when levels FIRE (Conductor/Pulse), not how many tokens they
process per fire. Within a segment, every level runs the full 512-token recurrence.

### Expected Behavior After This Spec

| Level | Tokens | M Updates | Backward Time (est.) |
|-------|--------|-----------|---------------------|
| L0 | 512 | 512 | ~49ms (unchanged) |
| L1 | 64 | 64 | ~6ms (8× reduction) |
| L2 | 8 | 8 | ~1ms (64× reduction) |
| L3 | 1 | 1 | ~0.1ms (512× reduction) |
| **Total** | **585** | **585** | **~56ms** (was ~506ms) |

memory_bwd drops from ~1625ms to ~200ms (estimated 8× reduction when including
projection matmuls, which also scale with token count).

## Architecture

### Per-Level Token Count

Each level f processes `s_f = seq_len / chunk_sizes[f]` tokens:

```text
seq_len = 512, chunk_sizes = [1, 8, 64, 512]

Level 0: s_0 = 512/1   = 512 tokens
Level 1: s_1 = 512/8   =  64 tokens
Level 2: s_2 = 512/64  =   8 tokens
Level 3: s_3 = 512/512 =   1 token
```

**Constraint**: `seq_len % chunk_sizes[f] == 0` for all levels. This is already
enforced by the Conductor (chunk_sizes are powers of 2, seq_len is a power of 2).

### Mean Pool Summarization

Given input `x: [bs×s, d]` and pool factor `C`, produce `x_pooled: [bs×(s/C), d]`:

```text
x_pooled[b, g, :] = mean(x[b, g*C : (g+1)*C, :])   for g = 0..s/C
```

This is a simple average over C consecutive d-dimensional vectors. No learnable
parameters. No activation function. Just arithmetic mean.

**Why mean, not sum**: Mean preserves magnitude scale across levels. If L0 output
vectors have norm ~10, L1 input vectors also have norm ~10. Sum would scale by C,
causing L1 to see 8× larger inputs than L0 — requiring per-level normalization.

### Repeat Upsample

Given `y_level: [bs×s_f, d]` at reduced resolution, produce `y_full: [bs×s, d]`:

```text
y_full[b, t, :] = y_level[b, t / C, :]   for t = 0..s
```

Each reduced-resolution output is repeated C times to fill the full sequence length.
This is nearest-neighbor upsampling — the simplest and cheapest option.

### Forward Path (Independent / FreqGated)

```text
input: ln_mem_out [bs×s, d]

for level in 0..k:
    C = chunk_sizes[level]
    s_f = s / C

    if C > 1:
        input_f = mean_pool(ln_mem_out, C)     // [bs×s_f, d]
    else:
        input_f = ln_mem_out                    // [bs×s, d] — no pooling for L0

    // Memory projections at REDUCED resolution
    k_mem = input_f @ W_K^T                    // [bs×s_f, d]
    v_mem = input_f @ W_V^T                    // [bs×s_f, d]
    q_mem = input_f @ W_Q^T                    // [bs×s_f, d]

    // Gates at reduced resolution
    alpha = sigmoid(k_mem_norms @ w_alpha + b_alpha)   // [bs×s_f]
    theta = softplus(q_mem_norms @ w_theta + b_theta)  // [bs×s_f]

    // Memory recurrence at reduced resolution
    for t in 0..s_f:
        M_{t+1} = (1 - alpha_t) * M_t + theta_t * v_t ⊗ k_t
        y_t = M_{t+1} @ q_t

    // y_level: [bs×s_f, d]
    if C > 1:
        y_level_full = repeat_upsample(y_level, C)   // [bs×s, d]
    else:
        y_level_full = y_level

    y_per_level.push(y_level_full)

y_combined = Σ softmax(alpha_mem)[l] * y_per_level[l]   // [bs×s, d]
```

### Forward Path (Chain / HopeVariant::Chained)

```text
input: ln_mem_out [bs×s, d]

h = ln_mem_out    // [bs×s, d] — full resolution

for level in 0..k:
    C = chunk_sizes[level]
    s_f = s / C

    if C > 1:
        h_f = mean_pool(h, C)    // [bs×s_f, d]
    else:
        h_f = h                  // [bs×s, d]

    // Memory forward at reduced resolution
    y_level = memory_forward(h_f, M[level], s_f, d)   // [bs×s_f, d]

    if level < k - 1:
        h = y_level   // Next level's input is at THIS level's resolution
        // Next level will pool further if its chunk_size > current level's

    y_per_level.push(y_level)   // Stored at native resolution

// Final output: y_per_level[k-1] at [bs×s_3, d] — broadcast to [bs×s, d]
y_combined = repeat_upsample(y_per_level[k-1], chunk_sizes[k-1])
```

**Chain subtlety**: In chain mode, L1 receives L0's output [bs×512, d] and pools it
to [bs×64, d]. L2 receives L1's output [bs×64, d] and pools it to [bs×8, d]. The
pool factor at each step is `chunk_sizes[level] / chunk_sizes[level-1]` (relative,
not absolute). For chunk_sizes=[1,8,64,512], the relative factors are [1, 8, 8, 8].

Wait — this needs clarification. L1's chunk_size is 8. If L0 outputs 512 tokens,
L1 should see 512/8 = 64 tokens. L2's chunk_size is 64. If L1 outputs 64 tokens,
L2 should see 64/8 = 8 tokens (pool factor = chunk_sizes[2]/chunk_sizes[1] = 8).
L3's chunk_size is 512. L2 outputs 8 tokens, L3 sees 8/8 = 1 token.

In chain mode, the pool factor between level f and f+1 is:
```
pool_factor = chunk_sizes[f+1] / chunk_sizes[f]
```

In independent mode, the pool factor is always relative to the raw input:
```
pool_factor = chunk_sizes[f]   (pool from s down to s/chunk_sizes[f])
```

### Backward Path (Independent)

Gradient flows backward through upsample, then through memory backward, then
through pool:

```text
d_y_combined: [bs×s, d]

for level in (0..k).rev():
    C = chunk_sizes[level]
    s_f = s / C

    // d_y_level_full = alpha_weight[level] * d_y_combined  (from softmax aggregation)
    d_y_level_full: [bs×s, d]

    if C > 1:
        // Backward of repeat_upsample: sum over C copies
        d_y_level = pool_upsample_backward(d_y_level_full, C)   // [bs×s_f, d]
        // d_y_level[g] = sum(d_y_level_full[g*C : (g+1)*C])
    else:
        d_y_level = d_y_level_full

    // Memory backward at reduced resolution
    d_input_f = memory_backward(d_y_level, cache[level], s_f, d)   // [bs×s_f, d]

    if C > 1:
        // Backward of mean_pool: broadcast gradient / C
        d_ln_mem_out += pool_backward(d_input_f, C)   // [bs×s, d]
        // d_ln_mem_out[t] += d_input_f[t/C] / C
    else:
        d_ln_mem_out += d_input_f

return d_ln_mem_out
```

### Backward Path (Chain)

Gradient flows backward through the chain in reverse order:

```text
d_upstream = repeat_upsample_backward(d_y_combined, chunk_sizes[k-1])
// d_upstream: [bs×s_{k-1}, d]

for level in (0..k).rev():
    C_rel = chunk_sizes[level] / chunk_sizes.get(level-1).unwrap_or(1)
    s_f = s / chunk_sizes[level]

    // Memory backward at this level's resolution
    d_input_f = memory_backward(d_upstream, cache[level], s_f, d)

    if level > 0:
        // Backward through pool: this level pooled the previous level's output
        d_upstream = pool_backward(d_input_f, C_rel)   // upsample gradient
    else:
        d_ln_mem_out = d_input_f   // L0 gradient flows to LN backward

return d_ln_mem_out
```

## New CUDA Kernels

### mean_pool_1d

```c
// Pool C consecutive d-dimensional vectors into their mean.
// Input:  x [n_tokens, d] where n_tokens = bs * s
// Output: out [n_tokens / C, d]
// Grid: (n_tokens / C), Block: (min(d, 1024))
__global__ void mean_pool_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_tokens, int d, int C)
{
    int group = blockIdx.x;    // which group of C tokens
    int col   = threadIdx.x;   // which dimension

    float sum = 0.0f;
    for (int i = 0; i < C; i++) {
        sum += x[(group * C + i) * d + col];
    }
    out[group * d + col] = sum / (float)C;
}
```

For d > 1024, the kernel uses strided access (same pattern as memory kernels).

### repeat_upsample_1d

```c
// Repeat each of s_f vectors C times to produce s_f * C vectors.
// Input:  x [n_groups, d] where n_groups = bs * s_f
// Output: out [n_groups * C, d]
// Grid: (n_groups * C), Block: (min(d, 1024))
__global__ void repeat_upsample_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n_groups, int d, int C)
{
    int token = blockIdx.x;    // output token index
    int col   = threadIdx.x;   // dimension
    int group = token / C;     // source group

    out[token * d + col] = x[group * d + col];
}
```

### Backward Kernels

**mean_pool_backward**: Given `d_out [n_groups, d]`, produce `d_x [n_tokens, d]`
where `d_x[t] = d_out[t/C] / C`. This is the same as `repeat_upsample` scaled by `1/C`.

**repeat_upsample_backward**: Given `d_out [n_tokens, d]`, produce `d_x [n_groups, d]`
where `d_x[g] = sum(d_out[g*C : (g+1)*C])`. This is the same as `mean_pool` scaled by `C`
(i.e., sum without division).

Both backward kernels can reuse the forward kernels with a scale factor, or be
implemented as trivial wrappers. No new kernel files needed — add to an existing
utility kernel file or create `core/kernels/pool_ops.cu`.

## Implementation Plan

### Phase 1: Pool/Upsample Primitives (~100 lines CUDA, ~40 lines Rust FFI)

**Files**:
- `core/kernels/pool_ops.cu` — mean_pool_1d, repeat_upsample_1d, and their backward variants
- `core/src/cuda_ffi.rs` — FFI declarations
- `core/src/dispatch.rs` — safe Rust wrappers
- `core/build.rs` — add pool_ops.cu to compilation

**Testing**: Rust unit tests comparing CUDA pool/upsample against CPU reference.
Gradient check via finite differences on pool → memory → upsample chain.

### Phase 2: Per-Level seq_len in Forward (~80 lines Rust)

**Files**:
- `core/src/gpu_stacked_forward.rs` — compute `s_f = s / chunk_sizes[level]`, pool
  input before `gpu_memory_forward`, upsample output after
- `core/src/gpu_forward.rs` — `gpu_memory_forward` already accepts `s: usize`;
  no changes needed. The reduced `s_f` flows through naturally.

**Key change in gpu_stacked_forward.rs** (Independent path):
```rust
for level in 0..cfg.k {
    let c = cfg.chunk_sizes[level];
    let s_f = s / c;

    let level_input = if c > 1 {
        mean_pool_1d(&ln_mem_out, s_f, d, c, bs)   // [bs*s_f, d]
    } else {
        ln_mem_out.clone_buf()
    };

    let (y_level, mem_cache) = gpu_memory_forward(
        &block.levels[level], cfg, &level_input,
        &mut block_ctx.memory[level],
        s_f,    // ← KEY CHANGE: was `s` for all levels
        d, level, bs,
    );

    let y_full = if c > 1 {
        repeat_upsample_1d(&y_level, s_f, d, c, bs)   // [bs*s, d]
    } else {
        y_level
    };

    y_per_level.push(y_full);
    memory_caches.push(Some(mem_cache));
}
```

**Chain path**: Same structure, but pool factor is relative (chunk_sizes[level] / chunk_sizes[level-1]).

### Phase 3: Per-Level seq_len in Backward (~100 lines Rust)

**Files**:
- `core/src/gpu_stacked_backward.rs` — match forward path: upsample_backward →
  memory_backward at s_f → pool_backward

The backward kernels in `gpu_backward.rs` already accept `s: usize`. Passing `s_f`
instead of `s` works transparently. The only new code is the pool/upsample backward
wrappers around the existing memory backward calls.

### Phase 4: m_states Buffer Sizing (~30 lines Rust)

Currently `m_states` is allocated as `[bs * (s + 1) * dd]` for every level. With
token reduction, level f needs only `[bs * (s_f + 1) * dd]`:

| Level | Current (s=512, dd=64²) | New | Savings |
|-------|------------------------|----|---------|
| L0 | 513 × 4096 = 2.1M floats | 513 × 4096 | 0% |
| L1 | 513 × 4096 = 2.1M | 65 × 4096 | 87% |
| L2 | 513 × 4096 = 2.1M | 9 × 4096 | 98% |
| L3 | 513 × 4096 = 2.1M | 2 × 4096 | 99.6% |

**Files**: `core/src/gpu_forward.rs` — allocate m_states with `s_f` instead of `s`
in `gpu_memory_forward`. Since `s` is already a parameter, this happens automatically
when Phase 2 passes `s_f`.

### Phase 5: Cache Shape Audit (~20 lines Rust)

`GpuMemoryCache` stores k_mem, v_mem, q_mem, alpha, theta, m_states, etc. These are
currently sized at `[bs * s * d]` or `[bs * s]`. With token reduction, they must be
sized at `[bs * s_f * d]` or `[bs * s_f]`.

Since `gpu_memory_forward` allocates these internally using its `s` parameter, passing
`s_f` automatically produces correctly-sized caches. The backward path reads from these
caches, so it must also use `s_f` — which it will, since the cache carries its own
buffer sizes.

**Verification**: Add debug_assert checks that cache buffer sizes match expected
`s_f * d` or `s_f` for each level.

### Phase 6: Python Config Validation (~5 lines Python)

**Files**: `python/engine/config.py` — add validation:
```python
assert cfg.seq_len % max(cfg.chunk_sizes) == 0, \
    f"seq_len ({cfg.seq_len}) must be divisible by max chunk_size ({max(cfg.chunk_sizes)})"
```

This is likely already satisfied (seq_len=512, max chunk_size=512), but should be
explicitly enforced.

### Phase 7: Testing (~150 lines Rust)

1. **Pool/upsample correctness**: CUDA vs CPU reference for various C, d, bs.
2. **Round-trip identity**: pool(upsample(x, C), C) ≈ x (exact for uniform x).
3. **Gradient check**: FD on pool → matmul → upsample chain.
4. **k=1 equivalence**: chunk_sizes=[1] produces identical output to current code.
5. **k=4 token counts**: verify L1 processes exactly 64 tokens, L2 exactly 8, L3 exactly 1.
6. **VRAM reduction**: verify m_states allocation is proportional to s_f, not s.
7. **Numerical parity**: for chunk_sizes=[1,1,1,1], output must be bit-identical to current.

## Interaction with Existing Specs

### Spec 25 (Segment-Scoped Tape)

The tape records operations per segment. With token reduction, L1's tape records 64
operations instead of 512. Tape memory drops proportionally. The cycle_length
(max chunk_size = 512) is unchanged — this is the segment boundary, not the per-level
token count.

### Spec 35 (Chain CMS GPU)

Chain CMS threads level outputs sequentially. Token reduction changes the resolution
at which each level operates. In chain mode, the pool factor between adjacent levels
is relative: `chunk_sizes[f+1] / chunk_sizes[f]`. For [1,8,64,512] this gives
relative factors [1, 8, 8, 8] — each level pools its input by 8× before the next.

### Spec 45 (Per-Head Memory Kernels)

Per-head mode splits d×d into num_heads × (head_dim × head_dim). Token reduction
is orthogonal — it changes seq_len, not d or head_dim. The pool/upsample operates
on the full d-dimensional vectors before/after the per-head split inside
`gpu_memory_forward`.

### Conductor/Pulse (CS-27, CS-28)

Pulse determines which levels FIRE at a given step. Token reduction determines how
many tokens a level processes WHEN it fires. These are orthogonal. In the current
segment-at-a-time execution (one segment per step), the Pulse check happens once
per segment, and all active levels process their reduced token counts within that
segment.

**Important**: The Pulse `active_levels` check in gpu_stacked_forward.rs is
unchanged. When L1 is inactive (step not divisible by 8), it still uses
`gpu_memory_read_only` — but now at s_f=64 resolution instead of s=512.

### Chunkwise Backward (TNT)

The chunkwise backward approximation uses its own notion of "chunks" (num_chunks =
seq_len / chunk_size_backward). With token reduction, L1's chunkwise backward
operates on s_f=64 tokens, which may need different chunk sizes for the backward
approximation. This is a detail to resolve during implementation — the chunkwise
backward chunk_size is distinct from the CMS frequency chunk_size.

## Design Decisions

### D1: Mean Pool vs Learned Compression

**Decision**: Mean pool (no learnable parameters).

**Rationale**: Simplest approach that preserves magnitude scale. The HOPE paper's CMS
hierarchy implies temporal averaging at each frequency level. Learned compression
(Lattice-style, or a small projection MLP) adds parameters and backward complexity
for uncertain gain. Mean pool can be replaced later without changing the architecture.

### D2: Upsample Method

**Decision**: Nearest-neighbor repeat (each pooled output repeated C times).

**Rationale**: Simplest. Linear interpolation would be more accurate but adds
complexity and isn't clearly better for discrete token sequences. The upsample
output is aggregated with other levels' outputs, so per-token precision at the
upsampled resolution is less critical.

### D3: Pool Before vs After Projection

**Decision**: Pool BEFORE projection (W_K, W_V, W_Q).

**Rationale**: Pooling before projection means L1 projects 64 vectors instead of 512,
saving 8× on the cuBLAS matmul. This is the dominant cost savings. Projecting first
and then pooling would preserve individual token information longer but defeats the
purpose of token reduction for compute savings.

### D4: Independent vs Chain Interaction

**Decision**: Token reduction applies to BOTH modes.

In **Independent** mode: each level pools directly from `ln_mem_out` by its absolute
chunk_size. L1 pools 512 → 64. L2 pools 512 → 8. L3 pools 512 → 1.

In **Chain** mode: each level pools from the previous level's output by the relative
chunk_size ratio. L0 outputs 512. L1 pools 512 → 64 (ratio 8). L2 pools 64 → 8
(ratio 8). L3 pools 8 → 1 (ratio 8).

### D5: Batch Dimension Handling

Pool and upsample treat the batch dimension as part of the token count:
`n_tokens = bs * s`. Pooling groups are within each batch element (not across batches).
The kernel must respect batch boundaries:

```c
int batch = blockIdx.x / (s / C);   // which batch element
int group = blockIdx.x % (s / C);   // which group within the batch
```

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-070-arch-variant1 | hope_equations | HOPE §6 Eq 70 | implements |
| eq-071-arch-variant2 | hope_equations | HOPE §6 Eq 71 | implements |
| eq-074-arch-variant5 | hope_equations | HOPE §6 Eq 74 | cites |
| eq-097-hope-cms-chain | hope_equations | HOPE §7 Eq 97 | cites |

## Acceptance Criteria

1. L0 processes 512 tokens, L1 processes 64, L2 processes 8, L3 processes 1
   (verified via profiler event timing ratios)
2. Total memory_bwd time drops ≥3× from 1625ms baseline
3. Sawtooth GPU utilization eliminated or substantially reduced
4. chunk_sizes=[1,1,1,1] produces bit-identical output to current code
5. All existing tests pass (may need cache size adjustments)
6. Loss trajectory matches or improves vs current baseline (requires end-to-end run)

## Estimated Effort

| Phase | Lines | Files |
|-------|-------|-------|
| 1. Pool/upsample CUDA + FFI | ~140 | pool_ops.cu, cuda_ffi.rs, dispatch.rs, build.rs |
| 2. Forward token reduction | ~80 | gpu_stacked_forward.rs |
| 3. Backward token reduction | ~100 | gpu_stacked_backward.rs |
| 4. Buffer sizing (auto) | ~30 | gpu_forward.rs audit |
| 5. Cache shape audit | ~20 | gpu_forward.rs, gpu_backward.rs |
| 6. Python validation | ~5 | config.py |
| 7. Testing | ~150 | tests in gpu_stacked_forward.rs, gpu_stacked_backward.rs |
| **Total** | **~525** | **~8 files** |

Estimated calendar time: 3-5 working sessions. Primary risk is the backward path
through pool → memory_backward → upsample, which must be validated via FD gradient
checking at each level independently.
