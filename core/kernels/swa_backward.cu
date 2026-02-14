// SWA Backward CUDA Kernel — Phase 2 Track Zero-A
//
// Sliding Window Attention backward pass on GPU.
// Grid=(num_heads, seq_len), Block=(head_dim).
// One block per (head, query_position). Computes dQ, dK, dV.
//
// bf16 storage for Q/K/V/attn_weights, f32 for all gradients.
// All arithmetic in float. Matches forward kernel precision pattern.
//
// Matches backward.rs Stage 3 exactly:
//   1. d_attn_w[w] = sum_d d_attn_out[q,h,d] * V[k,h,d]
//   2. d_V[k,h,d] += weights[w] * d_attn_out[q,h,d]     (atomicAdd)
//   3. softmax backward: d_scores
//   4. d_Q[q,h,d] += d_scores[w] * K[k,h,d] * scale     (direct write)
//   5. d_K[k,h,d] += d_scores[w] * Q[q,h,d] * scale     (atomicAdd)
//
// This file is compiled by nvcc into machine code (opaque to Enzyme).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

__global__ void swa_backward_kernel(
    const __nv_bfloat16* __restrict__ q,           // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ k,           // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ v,           // [seq_len, total_dim]
    const __nv_bfloat16* __restrict__ attn_weights,// [num_heads, seq_len, window_size]
    const float* __restrict__ d_attn_out,          // [seq_len, total_dim]
    float* __restrict__ d_q,                       // [seq_len, total_dim]
    float* __restrict__ d_k,                       // [seq_len, total_dim]
    float* __restrict__ d_v,                       // [seq_len, total_dim]
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

    int aw_base = (h * seq_len + q_pos) * window_size;

    // Shared memory layout:
    //   d_attn_w[window_size]   — per-window-position gradient of attention weights
    //   d_scores[window_size]   — gradient of pre-softmax scores
    extern __shared__ float smem[];
    float* s_d_attn_w = smem;                     // [window_size]
    float* s_d_scores = smem + window_size;        // [window_size]

    float scale = rsqrtf((float)head_dim);

    // ── Phase 1: d_attn_w[w] = sum_d d_attn_out[q,h,d] * V[k,h,d] ────
    for (int w = 0; w < win_len; w++) {
        int k_pos = win_start + w;
        float partial = 0.0f;
        if (d < head_dim) {
            partial = d_attn_out[q_pos * total_dim + h_offset + d]
                    * __bfloat162float(v[k_pos * total_dim + h_offset + d]);
        }

        // Warp-level reduction (head_dim <= 32 fits in one warp)
        unsigned mask = __activemask();
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(mask, partial, offset);
        }

        if (d == 0) {
            s_d_attn_w[w] = partial;
        }
    }
    // Zero unused slots
    if (d == 0) {
        for (int w = win_len; w < window_size; w++) {
            s_d_attn_w[w] = 0.0f;
        }
    }
    __syncthreads();

    // ── Phase 2: d_V[k,h,d] += weights[w] * d_attn_out[q,h,d] ────────
    // atomicAdd because multiple q_pos blocks write to the same k_pos
    if (d < head_dim) {
        float dao_val = d_attn_out[q_pos * total_dim + h_offset + d];
        for (int w = 0; w < win_len; w++) {
            int k_pos = win_start + w;
            // Load attn_weights: bf16 → f32
            float aw = __bfloat162float(attn_weights[aw_base + w]);
            atomicAdd(&d_v[k_pos * total_dim + h_offset + d], aw * dao_val);
        }
    }

    // ── Phase 3: Softmax backward (thread 0 only) ─────────────────────
    // d_scores[w] = P[w] * (d_attn_w[w] - sum_j P[j] * d_attn_w[j])
    if (d == 0) {
        float dot_pw = 0.0f;
        for (int w = 0; w < win_len; w++) {
            dot_pw += __bfloat162float(attn_weights[aw_base + w]) * s_d_attn_w[w];
        }

        for (int w = 0; w < win_len; w++) {
            s_d_scores[w] = __bfloat162float(attn_weights[aw_base + w]) * (s_d_attn_w[w] - dot_pw);
        }
        for (int w = win_len; w < window_size; w++) {
            s_d_scores[w] = 0.0f;
        }
    }
    __syncthreads();

    // ── Phase 4: d_Q and d_K ──────────────────────────────────────────
    if (d < head_dim) {
        float dq_acc = 0.0f;
        for (int w = 0; w < win_len; w++) {
            int k_pos = win_start + w;
            float ds = s_d_scores[w] * scale;
            // d_Q[q,h,d] += d_scores[w] * K[k,h,d] * scale (load K: bf16 → f32)
            dq_acc += ds * __bfloat162float(k[k_pos * total_dim + h_offset + d]);
            // d_K[k,h,d] += d_scores[w] * Q[q,h,d] * scale (atomicAdd, load Q: bf16 → f32)
            atomicAdd(&d_k[k_pos * total_dim + h_offset + d],
                      ds * __bfloat162float(q[q_pos * total_dim + h_offset + d]));
        }
        // d_Q: exactly one block per (h, q_pos), so direct store is safe (no contention)
        d_q[q_pos * total_dim + h_offset + d] = dq_acc;
    }
}

extern "C" void swa_backward_f32_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* attn_weights, const float* d_attn_out,
    float* d_q, float* d_k, float* d_v,
    int seq_len, int num_heads, int head_dim, int window_size)
{
    dim3 grid(num_heads, seq_len);
    dim3 block(head_dim);

    // Shared memory: d_attn_w[window_size] + d_scores[window_size]
    int smem_bytes = 2 * window_size * sizeof(float);

    swa_backward_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, attn_weights, d_attn_out,
        d_q, d_k, d_v,
        seq_len, num_heads, head_dim, window_size);
}
