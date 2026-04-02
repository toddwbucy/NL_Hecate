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
// This file is compiled by nvcc into machine code (opaque to AD).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

__global__ void swa_backward_kernel(
    const __nv_bfloat16* __restrict__ q,           // [batch_size, seq_len, total_dim]
    const __nv_bfloat16* __restrict__ k,           // [batch_size, seq_len, total_dim]
    const __nv_bfloat16* __restrict__ v,           // [batch_size, seq_len, total_dim]
    const __nv_bfloat16* __restrict__ attn_weights,// [batch_size, num_heads, seq_len, aw_stride]
    const float* __restrict__ d_attn_out,          // [batch_size, seq_len, total_dim]
    float* __restrict__ d_q,                       // [batch_size, seq_len, total_dim]
    float* __restrict__ d_k,                       // [batch_size, seq_len, total_dim]
    float* __restrict__ d_v,                       // [batch_size, seq_len, total_dim]
    int seq_len, int num_heads, int head_dim, int window_size,
    int n_persistent)
{
    int b = blockIdx.x / num_heads;  // batch index
    int h = blockIdx.x % num_heads;  // head index
    int q_pos = blockIdx.y;          // query position
    int d = threadIdx.x;             // dimension within head

    int total_dim = num_heads * head_dim;
    int h_offset = h * head_dim;
    int aw_stride = n_persistent + window_size;

    // Offset all buffers to this batch element
    q            += b * seq_len * total_dim;
    k            += b * seq_len * total_dim;
    v            += b * seq_len * total_dim;
    attn_weights += b * num_heads * seq_len * aw_stride;
    d_attn_out   += b * seq_len * total_dim;
    d_q          += b * seq_len * total_dim;
    d_k          += b * seq_len * total_dim;
    d_v          += b * seq_len * total_dim;

    // SWA* two-range mask (must match forward kernel exactly)
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
    int aw_base = (h * seq_len + q_pos) * aw_stride;

    // Shared memory layout:
    //   d_attn_w[aw_stride]  — per-position gradient of attention weights
    //   d_scores[aw_stride]  — gradient of pre-softmax scores
    //   reduce[head_dim]     — tree-reduction buffer
    extern __shared__ float smem[];
    float* s_d_attn_w = smem;                     // [aw_stride]
    float* s_d_scores = smem + aw_stride;          // [aw_stride]
    float* reduce = smem + 2 * aw_stride;          // [head_dim]

    float scale = rsqrtf((float)head_dim);

    // ── Phase 1: d_attn_w = sum_d d_attn_out * V ──────────────────────
    // Persistent prefix positions
    for (int w = 0; w < n_persistent; w++) {
        float partial = 0.0f;
        if (w < n_p_attend && d < head_dim) {
            partial = d_attn_out[q_pos * total_dim + h_offset + d]
                    * __bfloat162float(v[w * total_dim + h_offset + d]);
        }
        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }
        if (d == 0) s_d_attn_w[w] = (w < n_p_attend) ? reduce[0] : 0.0f;
        __syncthreads();
    }
    // Local window positions
    for (int w = 0; w < window_size; w++) {
        float partial = 0.0f;
        if (w < local_len && d < head_dim) {
            int k_pos = local_start + w;
            partial = d_attn_out[q_pos * total_dim + h_offset + d]
                    * __bfloat162float(v[k_pos * total_dim + h_offset + d]);
        }
        reduce[d] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (d < s) reduce[d] += reduce[d + s];
            __syncthreads();
        }
        if (d == 0) s_d_attn_w[n_persistent + w] = (w < local_len) ? reduce[0] : 0.0f;
        __syncthreads();
    }

    // ── Phase 2: d_V += weights * d_attn_out ──────────────────────────
    if (d < head_dim) {
        float dao_val = d_attn_out[q_pos * total_dim + h_offset + d];
        // Persistent prefix
        for (int w = 0; w < n_p_attend; w++) {
            float aw = __bfloat162float(attn_weights[aw_base + w]);
            atomicAdd(&d_v[w * total_dim + h_offset + d], aw * dao_val);
        }
        // Local window
        for (int w = 0; w < local_len; w++) {
            int k_pos = local_start + w;
            float aw = __bfloat162float(attn_weights[aw_base + n_persistent + w]);
            atomicAdd(&d_v[k_pos * total_dim + h_offset + d], aw * dao_val);
        }
    }

    // ── Phase 3: Softmax backward ─────────────────────────────────────
    if (d == 0) {
        float dot_pw = 0.0f;
        for (int w = 0; w < aw_stride; w++) {
            dot_pw += __bfloat162float(attn_weights[aw_base + w]) * s_d_attn_w[w];
        }
        for (int w = 0; w < aw_stride; w++) {
            float aw = __bfloat162float(attn_weights[aw_base + w]);
            s_d_scores[w] = aw * (s_d_attn_w[w] - dot_pw);
        }
    }
    __syncthreads();

    // ── Phase 4: d_Q and d_K ──────────────────────────────────────────
    if (d < head_dim) {
        float dq_acc = 0.0f;
        float q_val = __bfloat162float(q[q_pos * total_dim + h_offset + d]);
        // Persistent prefix
        for (int w = 0; w < n_p_attend; w++) {
            float ds = s_d_scores[w] * scale;
            dq_acc += ds * __bfloat162float(k[w * total_dim + h_offset + d]);
            atomicAdd(&d_k[w * total_dim + h_offset + d], ds * q_val);
        }
        // Local window
        for (int w = 0; w < local_len; w++) {
            int k_pos = local_start + w;
            float ds = s_d_scores[n_persistent + w] * scale;
            dq_acc += ds * __bfloat162float(k[k_pos * total_dim + h_offset + d]);
            atomicAdd(&d_k[k_pos * total_dim + h_offset + d], ds * q_val);
        }
        d_q[q_pos * total_dim + h_offset + d] = dq_acc;
    }
}

extern "C" void swa_backward_f32_cuda(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* attn_weights, const float* d_attn_out,
    float* d_q, float* d_k, float* d_v,
    int seq_len, int num_heads, int head_dim, int window_size, int batch_size,
    int n_persistent)
{
    dim3 grid(batch_size * num_heads, seq_len);
    dim3 block(head_dim);

    // Shared memory: d_attn_w[n_p+ws] + d_scores[n_p+ws] + reduce[head_dim]
    int aw_stride = n_persistent + window_size;
    int smem_bytes = (2 * aw_stride + head_dim) * sizeof(float);

    swa_backward_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, attn_weights, d_attn_out,
        d_q, d_k, d_v,
        seq_len, num_heads, head_dim, window_size, n_persistent);
}
