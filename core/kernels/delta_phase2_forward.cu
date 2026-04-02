// Delta Phase 2 Forward — Spec 44 (Batched cuBLAS Phase 1)
//
// Sequential M recurrence + readout for ONE chunk, reading pre-computed errors.
// Called per-chunk from Rust after cuBLAS GEMM + error_subtract_clip.
//
// 1. Store current M (m_work) to m_chunk_states[chunk_idx]
// 2. For each token t in chunk:
//      M_{t+1} = (1-α) M_t - θ outer(error_t, k_t)
//      y_t = M_{t+1} @ q_t
// 3. m_work is updated in-place (M_final = M₀ for next chunk)
//
// Grid=(batch_size), Block=(min(d², 1024)).
// All fp32.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "m_norm_project.cuh"

static inline void check_cuda_launch(const char* kernel_name, int d, int smem_bytes) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s launch failed (d=%d, smem=%d): %s\n",
                kernel_name, d, smem_bytes, cudaGetErrorString(err));
        abort();
    }
}

__global__ void delta_phase2_forward_kernel(
    const float* __restrict__ k_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ alpha,          // [batch_size, seq_len]
    const float* __restrict__ theta,          // [batch_size, seq_len]
    const float* __restrict__ errors,         // [batch_size, chunk_size, d] — pre-computed
    float* __restrict__ m_work,               // [batch_size, d*d] — M state, updated in-place
    float* __restrict__ m_chunk_states,       // [batch_size, (num_chunks+1)*d*d]
    float* __restrict__ y,                    // [batch_size, seq_len, d]
    int seq_len, int d, int chunk_size, int chunk_idx,
    float m_norm_max)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int dd = d * d;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    int t_start = chunk_idx * chunk_size;
    int t_end   = t_start + chunk_size;
    if (t_end > seq_len) t_end = seq_len;
    int C = t_end - t_start;

    // Offset to batch element
    const float* k_b     = k_mem + b * seq_len * d;
    const float* q_b     = q_mem + b * seq_len * d;
    const float* alpha_b = alpha + b * seq_len;
    const float* theta_b = theta + b * seq_len;
    const float* err_b   = errors + b * chunk_size * d;
    float* m_b           = m_work + b * dd;
    float* mcs_b         = m_chunk_states + b * (num_chunks + 1) * dd;
    float* y_b           = y + b * seq_len * d;

    // Shared memory for error vector
    extern __shared__ float smem[];
    float* error_buf = smem;  // [d]

    // Store M₀ for this chunk
    int cs_off = chunk_idx * dd;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        mcs_b[cs_off + idx] = m_b[idx];
    }
    __syncthreads();

    // Sequential recurrence + readout
    for (int tl = 0; tl < C; tl++) {
        int t = t_start + tl;
        const float* k_t = k_b + t * d;
        const float* q_t = q_b + t * d;
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];

        // Load pre-computed error
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = err_b[tl * d + row];
        }
        __syncthreads();

        // M_{t+1} = (1-α) M_t - θ outer(error, k)
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_b[idx] = retention * m_b[idx]
                       - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // Per-token M-norm projection (spec 74, matches CPU reference)
        m_norm_project_inplace(m_b, error_buf, dd, tid, m_norm_max);

        // y_t = M_{t+1} @ q_t
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_b[row * d + j] * q_t[j];
            }
            y_b[t * d + row] = sum;
        }
        __syncthreads();
    }

    // If this is the last chunk, store M_final
    if (chunk_idx == num_chunks - 1) {
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            mcs_b[num_chunks * dd + idx] = m_b[idx];
        }
    }
}

extern "C" void delta_phase2_forward_f32_cuda(
    const float* k_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* errors, float* m_work,
    float* m_chunk_states, float* y,
    int seq_len, int d, int batch_size, int chunk_size, int chunk_idx,
    float m_norm_max)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(batch_size);
    dim3 block(block_size);

    int smem_bytes = d * sizeof(float);  // error_buf[d]

    delta_phase2_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, q_mem, alpha, theta, errors, m_work,
        m_chunk_states, y,
        seq_len, d, chunk_size, chunk_idx, m_norm_max);
    check_cuda_launch("delta_phase2_forward_kernel", d, smem_bytes);
}
