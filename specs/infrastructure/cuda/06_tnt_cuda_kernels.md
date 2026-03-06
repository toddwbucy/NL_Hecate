# TNT CUDA Kernels — Chunkwise Parallelism on GPU

```
CONTRACT
  Purpose   : GPU acceleration for TNT hierarchical memory via batched kernel reuse
  Expects   : Existing Titans/Delta CUDA kernels with batch_size parameter;
              TNT CPU reference (core/src/tnt.rs) producing correct results
  Guarantees: GPU TNT forward matches CPU reference within 1e-4 per-element;
              backward FD gradient check passes (eps=1e-2, tol=10%);
              N local memories within a shard execute as a single batched kernel call
  Cost      : 3 forward + 3 backward helper kernels (~200 lines CUDA total);
              Rust orchestration in gpu_forward.rs (~150 lines)
  Trade-off : Mean-pooling shard summary only (attention summary deferred);
              Sequential shard loop in Rust (shard count is small: seq_len/C_G)
  Position  : Tier 2a — GPU-capable, not yet training-validated
  Source    : TNT (2511.07343) §2-3
```

## Overview

TNT breaks the sequential bottleneck of Titans by processing tokens in parallel
within fixed-size shards. One global memory M_G evolves sequentially across shard
boundaries, while N = C_G / C_L local memories process sub-chunks independently
within each shard — all starting from the same M_G snapshot.

The key architectural insight: N independent local memories within a shard map
directly to `batch_size=N` in the existing Titans/Delta CUDA kernels. No new
sequential recurrence kernel is needed. Only lightweight helper kernels for:
1. Broadcasting M_G → N copies
2. Mean-pooling local outputs into shard summary
3. Updating M_G with shard summary outer product

## Algorithm (GPU Path)

```
for shard in shards:                           // Sequential (Rust loop)
    tnt_broadcast_m(M_G → [M_G; N])           // Helper kernel
    slice shard tokens into N local chunks
    titans_forward(batch_size=N)               // REUSE existing kernel
    tnt_shard_summary_mean(local_y → k,v)      // Helper kernel
    tnt_global_update(M_G, k, v, alpha)        // Helper kernel
```

## New CUDA Kernels

### Forward Helpers

**`tnt_broadcast_m_f32_cuda(m_src, m_dst, N, d)`**
- Copy single d×d matrix to N contiguous copies
- Grid=(1), Block=(min(d*d, 1024)), loop over N copies per thread

**`tnt_shard_summary_mean_f32_cuda(local_y, k_sum, v_sum, shard_len, d)`**
- Mean-pool local outputs: k_sum[j] = v_sum[j] = mean_t(local_y[t,j])
- Grid=(1), Block=(min(d, 1024)), parallel reduction over d dimensions

**`tnt_global_update_f32_cuda(global_m, k_sum, v_sum, d, alpha)`**
- M_G[i,j] = alpha * M_G[i,j] + v_sum[i] * k_sum[j]
- Grid=(1), Block=(min(d*d, 1024)), in-place update

### Backward Helpers

**`tnt_global_update_backward_f32_cuda(d_m_new, k_sum, v_sum, d_m_old, d_k_sum, d_v_sum, d, alpha)`**
- Reverse of global update outer product

**`tnt_shard_summary_mean_backward_f32_cuda(d_k_sum, d_v_sum, d_local_y, shard_len, d)`**
- Distribute summary gradient uniformly to all tokens

**`tnt_combine_gradients_f32_cuda(d_y_upstream, d_y_global, d_y_combined, n)`**
- Element-wise addition of upstream and global gradients

## GpuMemoryCache::TNT Variant

```rust
TNT {
    // Per-shard caches (Vec over shards)
    shard_caches: Vec<GpuMemoryCache>,  // Inner cache per shard (Titans/Delta)
    shard_y: Vec<GpuBuf<f32>>,          // [shard_len, d] per shard
    k_summaries: Vec<GpuBuf<f32>>,      // [d] per shard
    v_summaries: Vec<GpuBuf<f32>>,      // [d] per shard
    global_states: Vec<GpuBuf<f32>>,    // [d*d] M_G after each shard
    // Config
    n_locals: usize,                     // N = C_G / C_L
    global_chunk_size: usize,
    local_chunk_size: usize,
}
```

## Memory Budget

At d=1024, C_G=64, C_L=8, N=8, k=4 levels:
- Per shard M states: N × (C_L+1) × d² × 4 bytes = 8 × 9 × 1024² × 4 ≈ 288 MB
- × 4 levels = ~1.2 GB active (fits easily on H100 80GB)
- Global M + summaries: trivial (few KB)

## Equations Traced

| Equation | Collection | Relationship |
|----------|-----------|-------------|
| eq-003-chunkwise-compression | tnt_equations | implements |
| eq-005-global-memory-update | tnt_equations | implements |
| eq-006-hierarchical-memory | tnt_equations | implements |
| eq-014-n-local-memories-update | tnt_equations | implements |
| eq-013-general-hierarchical-memory | tnt_equations | cites |

## Backward Path (GPU)

```
for shard in shards.reverse():                     // Reverse shard loop (Rust)
    // Step 1: Reverse global update
    tnt_global_update_backward(d_m_carry, k, v → d_m_old, d_k_sum, d_v_sum)

    // Step 2: Reverse summary mean
    tnt_shard_summary_mean_backward(d_k_sum, d_v_sum → d_local_y_global)

    // Step 3: Combine upstream + global gradients
    tnt_combine_gradients(d_y_upstream_shard, d_local_y_global → d_y_combined)

    // Step 4: Batched inner backward (reuse existing kernel)
    titans_backward(batch_size=N, d_y_combined → d_embedded_shard, d_m_inner)

    // Step 5: Gate backward into temp buffers, saxpy-accumulate into level_grads
    gate_backward(temp_bufs); saxpy(1.0, temp → level_grads)

    // Step 6: d_m_carry = d_m_old (chain across shards)
```

Key implementation detail: `gate_backward_cuda` OVERWRITES its output buffers
(uses `=` not `+=`). Per-shard gate backward must write to temporary buffers,
then `saxpy_cuda(1.0, temp, dst)` accumulates into the level-wide gradient
accumulators. This matches the CPU reference where `total_grads.accumulate()`
adds shard contributions.

## Verification

1. Forward: GPU output matches CPU `tnt_forward()` within 1e-4 (d=64, small config)
2. Backward: FD gradient check (eps=1e-2, tol=10%) against GPU backward
3. Benchmark: H100 d=1024 target >1000 tok/s (vs 209 baseline with Titans)

## Deferred

- Attention-based shard summary (`use_attention_summary: true`) — requires softmax kernel
- Q-K projection (`use_qk_projection: true`) — requires chunkwise API changes
- CUDA graph capture for TNT — would need stable shard-loop graph structure
