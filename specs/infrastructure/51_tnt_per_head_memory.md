# TNT Per-Head Memory Support

```
CONTRACT
  Purpose:    Extend TNT hierarchical parallelism (gpu_tnt_forward/backward) to
              support per-head memory (num_heads > 1). Removes the num_heads==1
              guard added in commit 4754703 that disabled TNT for all production
              configs, causing a 5x throughput regression.
  Expects:    Per-head memory infrastructure from spec cuda/08 (per_head_memory_kernels):
              M_h in R^{head_dim x head_dim}, context_m layout [bs*nh, hd*hd],
              reshape_to_per_head/reshape_from_per_head utilities, broadcast_gates.
              TNT sharding from spec algorithms/parallelization/03_tnt_hierarchical:
              global chunk C_G, local chunk C_L, N=C_G/C_L local memories per shard.
  Guarantees: (1) TNT dispatches for all num_heads values (1, 12, 16, 24).
              (2) Per-shard local memory kernels run with bs_mem=bs*nh batch dim,
                  giving nh independent recurrences in parallel per local chunk.
              (3) Global M update operates per-head: nh independent rank-1 updates.
              (4) Throughput restored to ~1200 tok/s for d=768/6b config (from 224).
              (5) VRAM reduced to pre-regression levels (~8-10 GB for d=768).
              (6) Backward produces correct gradients (FD-verified).
              (7) Single-head (num_heads=1) path produces bit-identical output.
  Cost:       Zero additional FLOP cost vs pre-regression TNT. The per-head split
              reduces per-kernel work (hd^2 vs d^2) while increasing kernel count
              (n_batch*nh vs n_batch). Net: same total FLOPs, better SM occupancy.
  Trade-off:  TNT global M update now operates per-head (nh x hd x hd) instead of
              monolithic (d x d). This matches the non-TNT path and is consistent
              with spec cuda/08's design. No cross-head interaction through M
              (same constraint as non-TNT per-head path).
  Position:   specs/infrastructure/51_tnt_per_head_memory.md
  Source:     TNT (2511.07343) Eqs 3-7, 13-15. Titans (2501.00663) Eq 32.
              MIRAS (2504.13173) memory structure knob. Spec cuda/08 per-head design.
```

## Problem Statement

### The Regression

Commit 4754703 added `cfg.swa.num_heads == 1` to the TNT dispatch guard in three
locations (`gpu_forward.rs`, `gpu_stacked_forward.rs` x2). The guard was added because
`gpu_tnt_forward` contained `assert!(cfg.swa.num_heads == 1)` — the function was
written before spec cuda/08 introduced per-head memory.

Since every production config has `num_heads > 1` (12, 16, or 24), TNT is **never
activated**. All forward passes fall through to `gpu_memory_forward`, which processes
the full sequence length sequentially with per-token M trajectory buffers:

```
Without TNT (current regression):
  m_states = GpuBuf::zeros(bs_mem * (s+1) * dd_mem)
  = 12 * 513 * 4096 = 25.2M floats PER LEVEL PER BLOCK
  → 6 blocks * 4 levels * 25.2M * 4 bytes = 2.4 GB just for M trajectories
  → Sequential: 512 timesteps per kernel call

With TNT (pre-regression):
  m_states = GpuBuf::zeros(n_batch * (cl+1) * dd)
  = 8 * 9 * dd per shard → chunked, parallel
  → Parallel: 8 local chunks per shard, shards sequential at C_G=64 boundaries
```

Result: 5x throughput regression (1220 → 224 tok/s) and ~3x VRAM increase.

### Root Cause

`gpu_tnt_forward` uses `d` (d_model) as the memory dimension throughout:
```rust
let dd = d * d;                                    // line 1810
let m_broadcast = GpuBuf::zeros(n_batch * dd);     // line 1839
titans_forward_dd(..., cl, d, n_batch, cl, dd, ..) // line 1961
```

The non-TNT path (`gpu_memory_forward`) was updated by spec cuda/08 to use:
```rust
let dd_mem = hd * hd;                              // line 1188
let bs_mem = bs * nh;                               // line 1189
let k_mem_ph = reshape_to_per_head(&k_mem, ...);   // line 1321
```

TNT was never updated to match.

## Solution

### Design Principle

TNT's sharding logic (broadcast M → local chunks → reduce → global update) is
**dimension-agnostic**. It doesn't care whether the "batch" dimension represents
literal batch or batch*heads. The fix is to thread per-head dimensions through
the existing TNT structure:

| TNT Variable | Before (num_heads=1) | After (per-head) |
|---|---|---|
| Memory dim | `dd = d * d` | `dd_mem = hd * hd` |
| Batch dim | `bs = 1` | `bs_mem = bs * nh` |
| Kernel input | `[shard_len, d]` | `[nh, shard_len, hd]` via reshape |
| Kernel batch | `n_batch` local chunks | `n_batch * nh` (heads folded into batch) |
| Broadcast M | `[dd]` → `[n_batch * dd]` | `[nh * dd_mem]` → `[n_batch * nh * dd_mem]` |
| Summary k/v | `[d]` per shard | `[nh * hd]` per shard (per-head pooling) |
| Global update | `M += alpha * k_sum outer v_sum` in d-space | Per-head: `M_h += alpha * k_sum_h outer v_sum_h` in hd-space |

### Key Insight: Global M Update Goes Per-Head

In the pre-regression code, global M was `d x d` and the global update used
d-dimensional shard summaries. With per-head memory, M is `nh x (hd x hd)`.
The global update must become per-head to match:

```
Per-head global update (per shard boundary):
  FOR h = 0 to nh-1:
    k_sum_h = mean(k_mem_h[shard_start..shard_end])    // [hd]
    v_sum_h = mean(v_mem_h[shard_start..shard_end])    // [hd]
    M_h += alpha * (k_sum_h outer v_sum_h)              // [hd x hd]
```

This is implemented by calling the existing `tnt_global_update_dd` dispatch
with `d=hd` and iterating over heads (or batching via `bs_mem`).

## Changes

### 1. `gpu_tnt_forward` (`core/src/gpu_forward.rs`)

**1a. Remove assertion, add per-head dimensions:**
```rust
// REMOVE: assert!(cfg.swa.num_heads == 1, ...);
let nh = cfg.swa.num_heads;
let hd = if nh > 1 { cfg.swa.head_dim } else { d };
let dd_mem = hd * hd;
let bs_mem = 1 * nh;  // bs is always 1 for TNT
```

**1b. Update broadcast (Step 1):**
```rust
// context_m is [bs_mem * dd_mem] = [nh * hd*hd]
// Broadcast each head's M to n_batch copies
let mut m_broadcast = GpuBuf::zeros(n_batch * bs_mem * dd_mem);
for h in 0..bs_mem {
    tnt_broadcast_m_dd(
        &context_m.slice(h * dd_mem, dd_mem),
        &mut m_broadcast.slice_mut(h * n_batch * dd_mem, n_batch * dd_mem),
        n_batch, hd,
    );
}
```
Or: refactor `tnt_broadcast_m_dd` to accept a batch of M matrices.

**1c. Per-head reshape after projection (Step 2):**

After computing k_mem, v_mem, q_mem in d-space via cuBLAS matmul, reshape:
```rust
let k_mem_ph = reshape_to_per_head(&k_mem, 1, shard_len, nh, hd);
let v_mem_ph = reshape_to_per_head(&v_mem, 1, shard_len, nh, hd);
let q_mem_ph = reshape_to_per_head(&q_mem, 1, shard_len, nh, hd);
let alpha_ph = broadcast_gates(&alpha, 1, shard_len, nh);
let theta_ph = broadcast_gates(&theta, 1, shard_len, nh);
```

**1d. Pad and dispatch with per-head batch (Steps 4-5):**

Padding operates on `[bs_mem * shard_len, hd]` layout. Kernel dispatch:
```rust
titans_forward_dd(
    &k_mem_b, &v_mem_b, &q_mem_b,
    &alpha_b, &theta_b, &eta_b,
    &m_initial_slice, &s_initial_slice,
    &mut m_states, &mut s_states, &mut y_local,
    cl, hd, n_batch * bs_mem,   // d=hd, batch=n_batch*nh
    cl, dd_mem, error_clip,
);
```

**1e. Per-head shard summary (Step 7):**

Reshape output back to d-space before summary:
```rust
let shard_y_dm = reshape_from_per_head(&shard_y_ph, 1, shard_len, nh, hd);
```
Or compute per-head summaries directly (preferred — avoids d-space detour):
```rust
// Per-head mean pooling: [nh, shard_len, hd] → [nh, hd]
for h in 0..nh {
    k_summaries[shard_idx].push(mean_pool(&k_mem_ph[h], shard_len, hd));
    v_summaries[shard_idx].push(mean_pool(&v_mem_ph[h], shard_len, hd));
}
```

**1f. Per-head global update (Step 8):**
```rust
for h in 0..bs_mem {
    tnt_global_update_dd(
        &mut context_m.slice_mut(h * dd_mem, dd_mem),
        &k_sums[h], &v_sums[h],
        hd, alpha,
    );
}
```

### 2. `gpu_tnt_backward` (`core/src/gpu_backward.rs`)

Mirror changes from forward:
- Remove batch_size=1 assumption where it conflicts with per-head
- Inner backward dispatch: `titans_backward_dd(..., cl, hd, n_batch * bs_mem, ...)`
- Per-head reshape of cached k/v/q/alpha/theta before backward kernel
- Per-head global M backward: iterate over heads

### 3. Dispatch guards (`gpu_stacked_forward.rs`, `gpu_forward.rs`)

Remove `&& cfg.swa.num_heads == 1` from all three TNT dispatch conditions:
- `gpu_stacked_forward.rs` line 376 (chained path)
- `gpu_stacked_forward.rs` line 440 (independent path)
- `gpu_forward.rs` line 948 (single-block path)

### 4. Dispatch functions (`dispatch.rs`)

**tnt_broadcast_m_dd**: Currently takes `(m_src, m_dst, n_local, d)`.
- Option A: Keep signature, call per-head in a loop (simplest).
- Option B: Add batch parameter: `(m_src, m_dst, n_local, d, n_heads)`.

**tnt_global_update_dd / backward**: Same options. Loop over heads or batch.

**tnt_shard_summary_mean_dd / backward**: Becomes per-head:
compute `[nh, hd]` summaries instead of `[d]`.

### 5. GpuMemoryCache::TNT variant

Update stored summaries from `Vec<GpuBuf<f32>>` of d-dim to `Vec<Vec<GpuBuf<f32>>>`
of `[nh][hd]`, or flatten to `Vec<GpuBuf<f32>>` of `[nh * hd]`.

## Verification

1. **Correctness**: Short k=2 CPU build with `num_heads=1` — bit-identical to pre-change
2. **Correctness**: Short k=2 CPU build with `num_heads=4` — compare TNT vs non-TNT output
3. **FD gradient check**: Small model (d=32, nh=4, hd=8) with TNT enabled
4. **Throughput**: d=768/6b config on GPU0 — target ~1200 tok/s (vs 224 current)
5. **VRAM**: d=768/6b should use ~8-10 GB (vs ~20 GB current)
6. **VRAM**: d=1024/8b should fit on 49 GB A6000 (currently OOMs)
7. **Cargo test**: All 778+ existing Rust tests pass
8. **Regression test**: Run k4_chain_smollm_d768_6b for 1024 steps, compare loss curve

## Risk Assessment

**Moderate** — this modifies the hot training path (forward + backward memory kernels
under TNT). However:
- The dimension-folding pattern (bs_mem, dd_mem) is proven in `gpu_memory_forward`
- The same reshape utilities are reused
- TNT's sharding logic is orthogonal to the head dimension
- `num_heads=1` path must produce identical output (regression gate)

## Files Modified

| File | Change |
|---|---|
| `core/src/gpu_forward.rs` | `gpu_tnt_forward`: per-head dimensions, remove assert |
| `core/src/gpu_backward.rs` | `gpu_tnt_backward`: mirror per-head changes |
| `core/src/gpu_stacked_forward.rs` | Remove `num_heads==1` from TNT dispatch (2 sites) |
| `core/src/gpu_stacked_backward.rs` | Remove `num_heads==1` if present |
| `core/src/gpu_forward.rs` | Remove `num_heads==1` from single-block TNT dispatch |
| `core/src/dispatch.rs` | Update TNT dispatch functions for per-head or loop-over-heads |

## Blocked Tasks

- **task_6527d3**: head_dim ablation (24 heads hd=32 vs 12 heads hd=64) — cannot
  produce valid results until TNT works with per-head memory.
