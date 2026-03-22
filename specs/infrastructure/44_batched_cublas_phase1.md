# 44 — Batched cuBLAS Phase 1 in Chunkwise Kernels

## CONTRACT

| Field       | Value |
|-------------|-------|
| Purpose     | Replace the sequential per-token Phase 1 error computation in the chunkwise forward/backward kernels (spec 43) with cuBLAS GEMM calls. Currently, Phase 1 computes `error_t = M₀ @ k_t - v_t` as C sequential matrix-vector multiplies inside a single CUDA block (one SM). cuBLAS parallelizes this as a single `[C, d] = [C, d] @ [d, d]ᵀ` GEMM across all SMs, then a lightweight kernel subtracts V and clips. Phase 2 (sequential M recurrence + readout) is unchanged. |
| Expects     | Spec 43 chunkwise frozen-M₀ kernels deployed: `delta_chunkwise_forward.cu`, `titans_chunkwise_forward.cu`, `delta_chunkwise_backward.cu`, `titans_chunkwise_backward.cu`. Existing cuBLAS infrastructure in `dispatch.rs` (`cublas_matmul_transb_dd`, `cublas_handle()`). The chunk_size bug fix applied (chunk_size read from `cfg.chunk_sizes[level]`). |
| Guarantees  | 1. Numerical parity with monolithic kernels (1e-5 forward, 1e-4 backward). 2. Phase 1 computed via cuBLAS GEMM — spreads work across multiple SMs instead of confining to one block. 3. Phase 2 unchanged — sequential M recurrence + readout within a single kernel. 4. Fallback: chunk_size=1 (L0) uses monolithic kernel (cuBLAS overhead dominates for single matvec). 5. Backward error recompute also uses cuBLAS. 6. fp32 throughout (non-negotiable). |
| Cost        | Per-chunk kernel launch overhead: ~3 launches per chunk (boundary store, cuBLAS+clip, Phase 2) vs 1 monolithic launch. At 64 chunks (seq_len=512, chunk_size=8), ~192 launches vs 1. Mitigated by dramatically better SM utilization per launch. |
| Trade-off   | More kernel launches vs better parallelism. cuBLAS GEMM `[8, 768] @ [768, 768]ᵀ` uses all SMs; the monolithic kernel's per-token matvec uses 1 SM with 1024 threads doing 768 inner products sequentially across 8 tokens. The GEMM wins at d≥256 and chunk_size≥4. |
| Position    | `specs/infrastructure/44_batched_cublas_phase1.md` |
| Source      | Titans (2501.00663) eq-016, eq-017 (frozen-M₀ enables batching). TNT (2511.07343) eq-003 (chunkwise compression). |

## Background

Spec 43 implemented the paper-aligned chunkwise forward where Phase 1 errors are computed against frozen chunk-start M₀. This factorization was designed to enable parallelization (Titans eq-017 shows the chunked gradient as a matrix multiplication). However, the CUDA kernels implemented Phase 1 as a sequential loop of per-token matvecs inside a single block, missing the batching opportunity.

### Current bottleneck

The monolithic kernel structure:
```
Grid=(batch_size), Block=(min(d², 1024))
// Each block = 1 SM processing ALL chunks for 1 batch element

for chunk in 0..num_chunks:
    // Phase 1: C sequential matvecs (serialized within block)
    for t in chunk:
        prediction[d] = M₀[d,d] @ k_t[d]    // 1024 threads, d² FLOPs
        error[d] = prediction - v_t            // trivial
        clip(error)                             // L2 norm clip

    // Phase 2: sequential recurrence (must stay serial)
    for t in chunk:
        M_update + readout                     // inherently serial
```

At d=768, chunk_size=8: Phase 1 does 8 × 768² = 4.7M FLOPs on a single SM. cuBLAS would execute the same as a `[8, 768] @ [768, 768]` GEMM across all SMs (~40 SMs on A6000).

### Titans eq-017 justification

The paper explicitly shows this batching:
```
Σ θᵢ · (β_b/βᵢ) · ∇ℓ(M₀; xᵢ) = Θ_b · B_b · (M₀·X - X) · Xᵀ
```
All tokens in the chunk processed simultaneously against frozen M₀. Our Phase 1 is the `M₀·X` portion — a GEMM.

## Specification

### Forward: per-chunk orchestration from Rust

For each chunk c in [0, num_chunks):

1. **cuBLAS GEMM**: `predictions[C, d] = K_chunk[C, d] @ M₀ᵀ[d, d]`
   - Use `cublas_matmul_transb_dd` — C=chunk_size rows, K=d, N=d
   - M₀ pointer: `m_work` buffer (persists across chunks, updated by Phase 2)
   - K_chunk pointer: offset into k_mem at `chunk * chunk_size * d`

2. **Error kernel**: `errors[C, d] = predictions[C, d] - V_chunk[C, d]`, then L2 clip per row
   - New lightweight kernel: `error_subtract_clip_kernel`
   - Grid=(C), Block=(min(d, 256)) — one block per token in chunk
   - Per block: subtract V row, compute L2 norm, clip if > threshold

3. **Phase 2 kernel**: sequential M recurrence + readout, reads pre-computed errors
   - `delta_phase2_forward_kernel` / `titans_phase2_forward_kernel`
   - Grid=(batch_size), Block=(min(d², 1024)) — same as current Phase 2 section
   - Reads errors from global buffer, updates M in-place, writes y_t
   - Also stores M₀ to chunk_states before processing

4. **Update M₀**: Phase 2 kernel leaves M_final in `m_work` — becomes M₀ for next chunk

### Backward: same batching for error recompute

The backward kernel recomputes Phase 1 errors per chunk (lines 120-145 of `delta_chunkwise_backward.cu`). Same cuBLAS opportunity:

1. **cuBLAS GEMM**: recompute `predictions[C, d] = K_chunk[C, d] @ M₀ᵀ[d, d]`
2. **Error kernel**: subtract V, clip
3. **Phase 2 backward kernel**: sequential reverse loop, reads pre-computed errors

Additionally in backward, `d_k_t += M₀ᵀ @ d_error_t` across all C tokens is also batchable:
- `d_K_contrib[C, d] = d_errors[C, d] @ M₀[d, d]` — another cuBLAS GEMM

### Batch dimension handling

For batch_size > 1, use `cublasSgemmStridedBatched`:
- strideA = chunk_size * d (K_chunk stride between batch elements)
- strideB = d * d (M₀ stride between batch elements)
- strideC = chunk_size * d (predictions stride)
- batchCount = batch_size

### chunk_size=1 bypass

When chunk_size=1 (L0), Phase 1 is a single matvec — cuBLAS launch overhead exceeds the computation. Keep the monolithic kernel for this case. The dispatch check:
```rust
if chunk_size == 1 {
    delta_chunkwise_forward_dd(...)  // existing monolithic
} else {
    delta_chunkwise_forward_batched_dd(...)  // new orchestrated path
}
```

## Files

### New CUDA kernels
| File | Purpose |
|------|---------|
| `core/kernels/error_subtract_clip.cu` | Batch error: `pred[C,d] -= V[C,d]` + L2 clip per row |
| `core/kernels/delta_phase2_forward.cu` | Phase 2 recurrence + readout + boundary store (delta) |
| `core/kernels/titans_phase2_forward.cu` | Phase 2 recurrence + readout + boundary store (titans, +momentum) |
| `core/kernels/delta_phase2_backward.cu` | Phase 2 backward (reverse loop, reads pre-computed errors) |
| `core/kernels/titans_phase2_backward.cu` | Phase 2 backward with momentum gradients |

### Modified Rust
| File | Change |
|------|--------|
| `core/src/dispatch.rs` | New `delta_chunkwise_forward_batched_dd()`, `titans_chunkwise_forward_batched_dd()`, backward variants. Per-chunk loop calling cuBLAS + error kernel + Phase 2 kernel. |
| `core/src/cuda_ffi.rs` | FFI declarations for Phase 2 kernels + error_subtract_clip |
| `core/src/gpu_forward.rs` | Dispatch batched path when chunk_size > 1 |
| `core/src/gpu_backward.rs` | Dispatch batched backward when chunk_size > 1 |
| `core/build.rs` | Compile new .cu files |

### Unchanged
| File | Why |
|------|-----|
| `core/kernels/delta_chunkwise_forward.cu` | Preserved as fallback for chunk_size=1 |
| `core/kernels/titans_chunkwise_forward.cu` | Preserved as fallback |
| `core/kernels/delta_chunkwise_backward.cu` | Preserved as fallback |
| `core/kernels/titans_chunkwise_backward.cu` | Preserved as fallback |

## Ontological Compliance

- **CS-18**: Kernel selection is math dispatch (Rust tier). Python passes config only.
- **CS-32**: Observe-then-advance — Phase 1 (observe) completes before Phase 2 (advance).
- **CS-40**: Opt-in AD unchanged. Phase 2 backward is the opaque VJP block.
- **CS-44**: fp32 throughout. cuBLAS called with float pointers.

## Equations Traced

| Equation | Collection | Relationship |
|----------|-----------|-------------|
| eq-016-chunk-wise-gd | titans_equations | implements |
| eq-017-gradient-as-matmul | titans_equations | implements |
| eq-003-chunkwise-compression | tnt_equations | cites |
| eq-090-chunk-wise-update | hope_equations | cites |

## Acceptance Criteria

1. Forward parity: batched output matches monolithic kernel within 1e-5 per element
2. Backward parity: FD gradient check passes at d=8, chunk_size=4
3. Throughput: measurable tok/s improvement at d=768, k=2, chunk_sizes=[1, 8]
4. chunk_size=1 (L0) uses monolithic path — no regression
5. Existing tests pass (`cargo test --features cuda`)

## Falsification Criteria

1. If cuBLAS GEMM at [8, 768] × [768, 768] is slower than 8 sequential matvecs on one SM → cuBLAS launch overhead too high, try cublasSgemmStridedBatched across chunks
2. If 192 kernel launches per level per step add >5ms overhead → batch multiple chunks into fewer launches
3. If numerical parity fails beyond 1e-5 → cuBLAS uses different accumulation order, may need relaxed tolerance
