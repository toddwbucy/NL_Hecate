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
// This file is compiled by nvcc into machine code (opaque to Enzyme).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

// Maximum window size supported by shared memory allocation.
// Shared memory per block: head_dim + 2*MAX_WINDOW + head_dim floats.
#define MAX_WINDOW 64

__global__ void swa_forward_kernel(
    const __nv_bfloat16* __restrict__ q,      // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ k,      // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ v,      // [seq_len, total_dim]
    __nv_bfloat16* __restrict__ out,          // [seq_len, total_dim]
    __nv_bfloat16* __restrict__ attn_weights, // [num_heads, seq_len, window_size]
    int seq_len, int num_heads, int head_dim, int window_size)
{
    int h = blockIdx.x;       // head index
    int q_pos = blockIdx.y;   // query position
    int d = threadIdx.x;      // dimension within head

    int total_dim = num_heads * head_dim;
    int h_offset = h * head_dim;

    // Causal window: [win_start, q_pos] inclusive
    int win_start = (q_pos + 1 >= window_size) ? (q_pos + 1 - window_size) : 0;
    int win_len = q_pos - win_start + 1;

    // Shared memory layout:
    //   q_row[head_dim]          — cached Q row for this position (f32)
    //   scores[window_size]      — raw attention scores (f32)
    //   weights[window_size]     — softmax weights (f32)
    //   reduce[head_dim]         — tree-reduction buffer for dot products
    extern __shared__ float smem[];
    float* q_row   = smem;                                         // [head_dim]
    float* scores  = smem + head_dim;                              // [window_size]
    float* weights = smem + head_dim + window_size;                // [window_size]
    float* reduce  = smem + head_dim + window_size + window_size;  // [head_dim]

    // Load Q row into shared memory (bf16 → f32)
    if (d < head_dim) {
        q_row[d] = __bfloat162float(q[q_pos * total_dim + h_offset + d]);
    }
    __syncthreads();

    // ── Phase 1: Compute scores ─────────────────────────────────────
    // Each window position needs a dot product Q[q_pos] · K[k_pos].
    // Each thread contributes its dimension, then warp-reduce.
    float scale = rsqrtf((float)head_dim);

    for (int w = 0; w < window_size; w++) {
        float partial = 0.0f;
        if (w < win_len && d < head_dim) {
            int k_pos = win_start + w;
            // Load K element: bf16 → f32
            partial = q_row[d] * __bfloat162float(k[k_pos * total_dim + h_offset + d]);
        }

        // Shared-memory tree reduction (works for any head_dim up to 1024)
        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (d < s) {
                reduce[d] += reduce[d + s];
            }
            __syncthreads();
        }

        // Thread 0 writes the score
        if (d == 0) {
            if (w < win_len) {
                scores[w] = reduce[0] * scale;
            } else {
                scores[w] = -FLT_MAX;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    // ── Phase 2: Softmax (thread 0 only, window_size is small) ──────
    if (d == 0) {
        // Find max for numerical stability
        float max_val = -FLT_MAX;
        for (int w = 0; w < window_size; w++) {
            if (scores[w] > max_val) max_val = scores[w];
        }

        // exp and sum
        float sum_exp = 0.0f;
        for (int w = 0; w < window_size; w++) {
            float e = expf(scores[w] - max_val);
            weights[w] = e;
            sum_exp += e;
        }

        // Normalize and write to global attn_weights (f32 → bf16)
        int aw_base = (h * seq_len + q_pos) * window_size;
        for (int w = 0; w < window_size; w++) {
            weights[w] /= sum_exp;
            attn_weights[aw_base + w] = __float2bfloat16(weights[w]);
        }
    }
    __syncthreads();

    // ── Phase 3: Weighted sum of values ─────────────────────────────
    if (d < head_dim) {
        float acc = 0.0f;
        for (int w = 0; w < win_len; w++) {
            int k_pos = win_start + w;
            // Load V element: bf16 → f32
            acc += weights[w] * __bfloat162float(v[k_pos * total_dim + h_offset + d]);
        }
        // Store output: f32 → bf16
        out[q_pos * total_dim + h_offset + d] = __float2bfloat16(acc);
    }
}

extern "C" void swa_forward_f32_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out, __nv_bfloat16* attn_weights,
    int seq_len, int num_heads, int head_dim, int window_size)
{
    dim3 grid(num_heads, seq_len);
    dim3 block(head_dim);

    // Shared memory: q_row[head_dim] + scores[window_size] + weights[window_size] + reduce[head_dim]
    int smem_bytes = (2 * head_dim + 2 * window_size) * sizeof(float);

    swa_forward_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, out, attn_weights,
        seq_len, num_heads, head_dim, window_size);
}
