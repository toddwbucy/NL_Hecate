// SWA Forward CUDA Kernel — Phase 2 Track Zero-A
//
// Sliding Window Attention forward pass on GPU.
// Grid=(num_heads, seq_len), Block=(head_dim).
// One block per (head, query_position). Threads parallelize over head_dim.
//
// bf16 storage, f32 compute: Q/K/V/out/attn_weights stored as __nv_bfloat16,
// all arithmetic in float. This matches production transformer precision
// (FlashAttention-style) while keeping gradient-checkable accuracy.
//
// This file is compiled by nvcc into machine code (opaque to AD).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

// Maximum window size supported by shared memory allocation.
// Shared memory per block: head_dim + 2*aw_stride + head_dim floats,
// where aw_stride = n_persistent + window_size.
#define MAX_WINDOW 64

__global__ void swa_forward_kernel(
    const __nv_bfloat16* __restrict__ q,      // [batch_size, seq_len, total_dim]
    const __nv_bfloat16* __restrict__ k,      // [batch_size, seq_len, total_dim]
    const __nv_bfloat16* __restrict__ v,      // [batch_size, seq_len, total_dim]
    __nv_bfloat16* __restrict__ out,          // [batch_size, seq_len, total_dim]
    __nv_bfloat16* __restrict__ attn_weights, // [batch_size, num_heads, seq_len, aw_stride]
    int seq_len, int num_heads, int head_dim, int window_size,
    int n_persistent)                          // SWA*: always attend to positions [0, n_persistent)
{
    int b = blockIdx.x / num_heads;  // batch index
    int h = blockIdx.x % num_heads;  // head index
    int q_pos = blockIdx.y;          // query position
    int d = threadIdx.x;             // dimension within head

    int total_dim = num_heads * head_dim;
    int h_offset = h * head_dim;
    int aw_stride = n_persistent + window_size;  // total weight slots per query position

    // Offset Q/K/V/out buffers to this batch element
    q   += b * seq_len * total_dim;
    k   += b * seq_len * total_dim;
    v   += b * seq_len * total_dim;
    out += b * seq_len * total_dim;
    attn_weights += b * num_heads * seq_len * aw_stride;

    // SWA* two-range mask (Titans Eq 27):
    //   Range 1: [0, n_p_attend)         — persistent prefix (always visible, causal)
    //   Range 2: [local_start, q_pos+1)  — local sliding window
    int n_p_attend = (q_pos + 1 < n_persistent) ? (q_pos + 1) : n_persistent;
    int local_start, local_len;
    if (q_pos >= n_persistent) {
        local_start = (q_pos + 1 - window_size > n_persistent)
                    ? (q_pos + 1 - window_size) : n_persistent;
        local_len = q_pos + 1 - local_start;
    } else {
        local_start = 0;
        local_len = 0;
    }
    // Shared memory layout:
    //   q_row[head_dim]       — cached Q row for this position (f32)
    //   scores[aw_stride]     — raw attention scores (f32)
    //   weights[aw_stride]    — softmax weights (f32)
    //   reduce[head_dim]      — tree-reduction buffer for dot products
    extern __shared__ float smem[];
    float* q_row   = smem;                                      // [head_dim]
    float* scores  = smem + head_dim;                           // [aw_stride]
    float* weights = smem + head_dim + aw_stride;               // [aw_stride]
    float* reduce  = smem + head_dim + aw_stride + aw_stride;   // [head_dim]

    // Load Q row into shared memory (bf16 → f32)
    if (d < head_dim) {
        q_row[d] = __bfloat162float(q[q_pos * total_dim + h_offset + d]);
    }
    __syncthreads();

    // ── Phase 1: Compute scores ─────────────────────────────────────
    float scale = rsqrtf((float)head_dim);

    // Score persistent prefix positions [0, n_persistent)
    for (int w = 0; w < n_persistent; w++) {
        float partial = 0.0f;
        if (w < n_p_attend && d < head_dim) {
            partial = q_row[d] * __bfloat162float(k[w * total_dim + h_offset + d]);
        }

        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }

        if (d == 0) {
            scores[w] = (w < n_p_attend) ? (reduce[0] * scale) : -FLT_MAX;
        }
        __syncthreads();
    }

    // Score local window positions [local_start, local_start + local_len)
    for (int w = 0; w < window_size; w++) {
        float partial = 0.0f;
        if (w < local_len && d < head_dim) {
            int k_pos = local_start + w;
            partial = q_row[d] * __bfloat162float(k[k_pos * total_dim + h_offset + d]);
        }

        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }

        if (d == 0) {
            scores[n_persistent + w] = (w < local_len) ? (reduce[0] * scale) : -FLT_MAX;
        }
        __syncthreads();
    }
    __syncthreads();

    // ── Phase 2: Softmax (thread 0 only) ────────────────────────────
    if (d == 0) {
        float max_val = -FLT_MAX;
        for (int w = 0; w < aw_stride; w++) {
            if (scores[w] > max_val) max_val = scores[w];
        }

        float sum_exp = 0.0f;
        for (int w = 0; w < aw_stride; w++) {
            float e = expf(scores[w] - max_val);
            weights[w] = e;
            sum_exp += e;
        }

        int aw_base = (h * seq_len + q_pos) * aw_stride;
        for (int w = 0; w < aw_stride; w++) {
            weights[w] /= sum_exp;
            attn_weights[aw_base + w] = __float2bfloat16(weights[w]);
        }
    }
    __syncthreads();

    // ── Phase 3: Weighted sum of values ─────────────────────────────
    if (d < head_dim) {
        float acc = 0.0f;
        // Persistent prefix contributions
        for (int w = 0; w < n_p_attend; w++) {
            acc += weights[w] * __bfloat162float(v[w * total_dim + h_offset + d]);
        }
        // Local window contributions
        for (int w = 0; w < local_len; w++) {
            int k_pos = local_start + w;
            acc += weights[n_persistent + w] * __bfloat162float(v[k_pos * total_dim + h_offset + d]);
        }
        out[q_pos * total_dim + h_offset + d] = __float2bfloat16(acc);
    }
}

extern "C" void swa_forward_f32_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, __nv_bfloat16* attn_weights,
    int seq_len, int num_heads, int head_dim, int window_size, int batch_size,
    int n_persistent)
{
    dim3 grid(batch_size * num_heads, seq_len);
    dim3 block(head_dim);

    // Shared memory: q_row[hd] + scores[n_p+ws] + weights[n_p+ws] + reduce[hd]
    int aw_stride = n_persistent + window_size;
    int smem_bytes = (2 * head_dim + 2 * aw_stride) * sizeof(float);

    swa_forward_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, out, attn_weights,
        seq_len, num_heads, head_dim, window_size, n_persistent);
}

// ══════════════════════════════════════════════════════════════════════
// Single-token SWA kernel for KV cache decode
// ══════════════════════════════════════════════════════════════════════
//
// For autoregressive generation: 1 query position attending over a
// K/V cache of arbitrary length. Grid=(num_heads), Block=(head_dim).
// No attn_weights output (inference only, no backward needed).

__global__ void swa_single_token_kernel(
    const __nv_bfloat16* __restrict__ q,        // [1, total_dim]
    const __nv_bfloat16* __restrict__ k_cache,  // [cache_len, total_dim]
    const __nv_bfloat16* __restrict__ v_cache,  // [cache_len, total_dim]
    __nv_bfloat16* __restrict__ out,            // [1, total_dim]
    int cache_len, int num_heads, int head_dim, int window_size,
    int n_persistent)
{
    int h = blockIdx.x;       // head index
    int d = threadIdx.x;      // dimension within head

    int total_dim = num_heads * head_dim;
    int h_offset = h * head_dim;
    int aw_stride = n_persistent + window_size;

    // Query position is cache_len - 1 (the newest token).
    int q_pos = cache_len - 1;

    // SWA* two-range mask (same as full kernel)
    int n_p_attend = (q_pos + 1 < n_persistent) ? (q_pos + 1) : n_persistent;
    int local_start, local_len;
    if (q_pos >= n_persistent) {
        local_start = (q_pos + 1 - window_size > n_persistent)
                    ? (q_pos + 1 - window_size) : n_persistent;
        local_len = q_pos + 1 - local_start;
    } else {
        local_start = 0;
        local_len = 0;
    }

    extern __shared__ float smem[];
    float* q_row   = smem;
    float* scores  = smem + head_dim;
    float* weights = smem + head_dim + aw_stride;
    float* reduce  = smem + head_dim + aw_stride + aw_stride;

    if (d < head_dim) {
        q_row[d] = __bfloat162float(q[h_offset + d]);
    }
    __syncthreads();

    float scale = rsqrtf((float)head_dim);

    // Score persistent prefix
    for (int w = 0; w < n_persistent; w++) {
        float partial = 0.0f;
        if (w < n_p_attend && d < head_dim) {
            partial = q_row[d] * __bfloat162float(k_cache[w * total_dim + h_offset + d]);
        }
        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }
        if (d == 0) scores[w] = (w < n_p_attend) ? (reduce[0] * scale) : -FLT_MAX;
        __syncthreads();
    }

    // Score local window
    for (int w = 0; w < window_size; w++) {
        float partial = 0.0f;
        if (w < local_len && d < head_dim) {
            int k_pos = local_start + w;
            partial = q_row[d] * __bfloat162float(k_cache[k_pos * total_dim + h_offset + d]);
        }
        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }
        if (d == 0) scores[n_persistent + w] = (w < local_len) ? (reduce[0] * scale) : -FLT_MAX;
        __syncthreads();
    }
    __syncthreads();

    // Softmax
    if (d == 0) {
        float max_val = -FLT_MAX;
        for (int w = 0; w < aw_stride; w++)
            if (scores[w] > max_val) max_val = scores[w];
        float sum_exp = 0.0f;
        for (int w = 0; w < aw_stride; w++) {
            float e = expf(scores[w] - max_val);
            weights[w] = e;
            sum_exp += e;
        }
        for (int w = 0; w < aw_stride; w++)
            weights[w] /= sum_exp;
    }
    __syncthreads();

    // Weighted sum
    if (d < head_dim) {
        float acc = 0.0f;
        for (int w = 0; w < n_p_attend; w++)
            acc += weights[w] * __bfloat162float(v_cache[w * total_dim + h_offset + d]);
        for (int w = 0; w < local_len; w++) {
            int k_pos = local_start + w;
            acc += weights[n_persistent + w] * __bfloat162float(v_cache[k_pos * total_dim + h_offset + d]);
        }
        out[h_offset + d] = __float2bfloat16(acc);
    }
}

extern "C" void swa_single_token_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    __nv_bfloat16* out,
    int cache_len, int num_heads, int head_dim, int window_size,
    int n_persistent)
{
    dim3 grid(num_heads);
    dim3 block(head_dim);

    int aw_stride = n_persistent + window_size;
    int smem_bytes = (2 * head_dim + 2 * aw_stride) * sizeof(float);

    swa_single_token_kernel<<<grid, block, smem_bytes>>>(
        q, k_cache, v_cache, out,
        cache_len, num_heads, head_dim, window_size, n_persistent);
}
