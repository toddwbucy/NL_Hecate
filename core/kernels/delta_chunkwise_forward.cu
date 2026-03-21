// DeltaRule Chunkwise Forward — Spec 43 (Paper-Aligned Frozen-M₀)
//
// Implements Titans eq-016 / TNT eq-003: errors computed against frozen
// chunk-start M₀, not the evolving M_t.
//
// Phase 1 (per chunk): error_t = M₀ @ k_t - v_t   (frozen M₀)
// Phase 2 (per chunk): M_t = (1-α) M_{t-1} - θ·outer(error_t, k_t)
//                      y_t = M_t @ q_t              (readout uses evolving M_t)
//
// At chunk_size=1 (L0): M₀ = M_{t-1}, exact (no approximation).
// At chunk_size>1:      errors use frozen chunk-start state.
//
// Memory layout:
//   m_chunk_states: [(num_chunks+1) * d²]  — M at each chunk boundary
//   m_work:         [d²]                    — evolving M in global memory
//   error_work:     [chunk_size * d]        — pre-computed errors for current chunk
//
// Grid=(batch_size), Block=(min(d², 1024)).
// All fp32.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_clip.cuh"

static inline void check_cuda_launch(const char* kernel_name, int d, int smem_bytes) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s launch failed (d=%d, smem=%d): %s\n",
                kernel_name, d, smem_bytes, cudaGetErrorString(err));
        abort();
    }
}

static inline void check_cuda_alloc(const char* tag, cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s: %s\n", tag, cudaGetErrorString(err));
        abort();
    }
}

__global__ void delta_chunkwise_forward_kernel(
    const float* __restrict__ k_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ alpha,          // [batch_size, seq_len]
    const float* __restrict__ theta,          // [batch_size, seq_len]
    const float* __restrict__ m_initial,      // [batch_size, d*d]
    float* __restrict__ m_chunk_states,       // [batch_size, (num_chunks+1)*d*d]
    float* __restrict__ y,                    // [batch_size, seq_len, d]
    float* __restrict__ m_work,               // [batch_size, d*d]
    float* __restrict__ error_work,           // [batch_size, chunk_size*d]
    int seq_len, int d, int chunk_size, float error_clip)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int dd = d * d;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // Offset to this batch element
    k_mem          += b * seq_len * d;
    v_mem          += b * seq_len * d;
    q_mem          += b * seq_len * d;
    alpha          += b * seq_len;
    theta          += b * seq_len;
    m_initial      += b * dd;
    m_chunk_states += b * (num_chunks + 1) * dd;
    y              += b * seq_len * d;
    m_work         += b * dd;
    error_work     += b * chunk_size * d;

    // Shared memory: prediction[d] + error_buf[d] = 2*d floats
    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf  = smem + d;

    // Load M₀ from m_initial into working M
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_work[idx] = m_initial[idx];
    }
    __syncthreads();

    for (int c = 0; c < num_chunks; c++) {
        int t_start = c * chunk_size;
        int t_end   = t_start + chunk_size;
        if (t_end > seq_len) t_end = seq_len;
        int C = t_end - t_start;

        // ── Store M₀ for this chunk ──
        int cs_off = c * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_chunk_states[cs_off + idx] = m_work[idx];
        }
        __syncthreads();

        // ── Phase 1: Pre-compute errors against frozen M₀ ──
        // error_t = M₀ @ k_t - v_t for all t in [t_start, t_end)
        for (int tl = 0; tl < C; tl++) {
            int t = t_start + tl;
            const float* k_t = k_mem + t * d;
            const float* v_t = v_mem + t * d;

            // prediction = M₀ @ k_t (strided: d > blockDim.x OK)
            for (int row = tid; row < d; row += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += m_work[row * d + j] * k_t[j];
                }
                prediction[row] = sum;
            }
            __syncthreads();

            // error = prediction - v, with clipping
            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = prediction[row] - v_t[row];
            }
            __syncthreads();
            error_clip_inplace(error_buf, prediction, d, tid, error_clip);

            // Store to global error buffer
            for (int row = tid; row < d; row += blockDim.x) {
                error_work[tl * d + row] = error_buf[row];
            }
            __syncthreads();
        }

        // ── Phase 2: Sequential M recurrence with pre-computed errors ──
        for (int tl = 0; tl < C; tl++) {
            int t = t_start + tl;
            const float* k_t = k_mem + t * d;
            const float* q_t = q_mem + t * d;
            float alpha_t = alpha[t];
            float theta_t = theta[t];

            // Load pre-computed error
            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = error_work[tl * d + row];
            }
            __syncthreads();

            // M_{t+1} = (1-α) * M_t - θ * outer(error, k)
            float retention = 1.0f - alpha_t;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                m_work[idx] = retention * m_work[idx]
                              - theta_t * error_buf[i] * k_t[j];
            }
            __syncthreads();

            // y_t = M_{t+1} @ q_t (readout uses evolving M)
            for (int row = tid; row < d; row += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += m_work[row * d + j] * q_t[j];
                }
                y[t * d + row] = sum;
            }
            __syncthreads();
        }
    }

    // Store M_final
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_chunk_states[num_chunks * dd + idx] = m_work[idx];
    }
}

extern "C" void delta_chunkwise_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_chunk_states, float* y,
    int seq_len, int d, int batch_size, int chunk_size, float error_clip)
{
    if (d <= 0 || 2 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "delta_chunkwise_forward_f32_cuda: d=%d out of range.\n", d);
        exit(1);
    }
    if (chunk_size <= 0) {
        fprintf(stderr, "delta_chunkwise_forward_f32_cuda: chunk_size=%d must be > 0.\n",
                chunk_size);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] = 2*d floats
    int smem_bytes = 2 * d * sizeof(float);

    // Allocate per-batch workspaces
    float* m_work = nullptr;
    float* error_work = nullptr;
    check_cuda_alloc("delta_chunkwise_fwd: cudaMalloc m_work",
                     cudaMalloc(&m_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("delta_chunkwise_fwd: cudaMalloc error_work",
                     cudaMalloc(&error_work, (size_t)batch_size * chunk_size * d * sizeof(float)));

    delta_chunkwise_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_chunk_states, y, m_work, error_work,
        seq_len, d, chunk_size, error_clip);
    check_cuda_launch("delta_chunkwise_forward_kernel", d, smem_bytes);

    check_cuda_alloc("delta_chunkwise_fwd: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(m_work);
    cudaFree(error_work);
}
