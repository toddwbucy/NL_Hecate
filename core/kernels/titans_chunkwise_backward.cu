// TitansLMM Chunkwise Backward — Spec 43 (Paper-Aligned Frozen-M₀)
//
// Backward for titans_chunkwise_forward. Two interacting accumulators (d_M, d_S)
// plus separate d_M₀ accumulator for the frozen-M₀ error gradient.
//
// Per chunk (reverse order):
//   1. Reload M₀/S₀ from chunk states, recompute forward trajectory
//   2. Reverse token loop:
//      - d_M += outer(d_y_t, q_t)
//      - d_q_t = M_{t+1}^T @ d_y_t
//      - d_S += d_M                          (S contributes additively to M)
//      - d_α_t = -sum(M_t * d_M)
//      - d_M = (1-α) * d_M                  (NO error chain — goes to d_M₀)
//      - d_η_t = sum(S_t * d_S)
//      - d_θ_t = -sum(outer(error_t, k_t) * d_S)
//      - d_error_t = -θ * (d_S @ k)
//      - d_k_t += -θ * error^T @ d_S
//      - d_k_t += M₀^T @ d_error_t          (from frozen M₀)
//      - d_v_t = -d_error_t
//      - d_S = η * d_S                      (propagate momentum)
//      - d_M₀ += outer(d_error_t, k_t)      (accumulated, not fed back)
//   3. d_M += d_M₀ for cross-chunk propagation
//
// Grid=(batch_size), Block=(min(d, 512)), __launch_bounds__(512).
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

__launch_bounds__(512)
__global__ void titans_chunkwise_backward_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ eta,
    const float* __restrict__ m_chunk_states,   // [(num_chunks+1)*d*d]
    const float* __restrict__ s_chunk_states,   // [(num_chunks+1)*d*d]
    const float* __restrict__ d_y,
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_theta,
    float* __restrict__ d_eta,
    float* __restrict__ d_m_initial,      // [d*d] — atomicAdd across batch
    float* __restrict__ d_s_initial,      // [d*d] — atomicAdd across batch
    float* __restrict__ d_M,              // [batch_size, d*d]
    float* __restrict__ d_S,              // [batch_size, d*d]
    float* __restrict__ d_M0,             // [batch_size, d*d]
    float* __restrict__ m_recompute,      // [batch_size, (chunk_size+1)*d*d]
    float* __restrict__ s_recompute,      // [batch_size, (chunk_size+1)*d*d]
    float* __restrict__ error_recompute,  // [batch_size, chunk_size*d]
    int seq_len, int d, int chunk_size, float error_clip)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int dd = d * d;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    k_mem          += b * seq_len * d;
    v_mem          += b * seq_len * d;
    q_mem          += b * seq_len * d;
    alpha          += b * seq_len;
    theta          += b * seq_len;
    eta            += b * seq_len;
    m_chunk_states += b * (num_chunks + 1) * dd;
    s_chunk_states += b * (num_chunks + 1) * dd;
    d_y            += b * seq_len * d;
    d_k_mem        += b * seq_len * d;
    d_v_mem        += b * seq_len * d;
    d_q_mem        += b * seq_len * d;
    d_alpha        += b * seq_len;
    d_theta        += b * seq_len;
    d_eta          += b * seq_len;
    d_M            += b * dd;
    d_S            += b * dd;
    d_M0           += b * dd;
    m_recompute    += b * (chunk_size + 1) * dd;
    s_recompute    += b * (chunk_size + 1) * dd;
    error_recompute += b * chunk_size * d;

    extern __shared__ float smem[];
    float* prediction  = smem;
    float* error_buf   = smem + d;
    float* d_error_buf = smem + 2 * d;
    float* reduce_buf  = smem + 3 * d;

    // Initialize d_M = 0, d_S = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
        d_S[idx] = 0.0f;
    }
    __syncthreads();

    for (int c = num_chunks - 1; c >= 0; c--) {
        int t_start = c * chunk_size;
        int t_end   = t_start + chunk_size;
        if (t_end > seq_len) t_end = seq_len;
        int C = t_end - t_start;

        // ── Recompute forward from chunk start ──
        int cs_off = c * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_recompute[idx] = m_chunk_states[cs_off + idx];
            s_recompute[idx] = s_chunk_states[cs_off + idx];
        }
        __syncthreads();

        // Phase 1: errors against frozen M₀
        for (int tl = 0; tl < C; tl++) {
            int t = t_start + tl;
            const float* k_t = k_mem + t * d;
            const float* v_t = v_mem + t * d;

            for (int row = tid; row < d; row += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += m_recompute[row * d + j] * k_t[j];
                }
                prediction[row] = sum;
            }
            __syncthreads();

            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = prediction[row] - v_t[row];
            }
            __syncthreads();
            error_clip_inplace(error_buf, prediction, d, tid, error_clip);

            for (int row = tid; row < d; row += blockDim.x) {
                error_recompute[tl * d + row] = error_buf[row];
            }
            __syncthreads();
        }

        // Phase 2: rebuild M+S trajectory
        for (int tl = 0; tl < C; tl++) {
            int t = t_start + tl;
            const float* k_t = k_mem + t * d;
            float alpha_t = alpha[t];
            float theta_t = theta[t];
            float eta_t   = eta[t];

            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = error_recompute[tl * d + row];
            }
            __syncthreads();

            float retention = 1.0f - alpha_t;
            int cur  = tl * dd;
            int next = (tl + 1) * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                float s_new = eta_t * s_recompute[cur + idx]
                              - theta_t * error_buf[i] * k_t[j];
                s_recompute[next + idx] = s_new;
                m_recompute[next + idx] = retention * m_recompute[cur + idx] + s_new;
            }
            __syncthreads();
        }

        // ── d_M₀ accumulator for this chunk ──
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M0[idx] = 0.0f;
        }
        __syncthreads();

        // ── Backward: reverse token loop ──
        for (int tl = C - 1; tl >= 0; tl--) {
            int t = t_start + tl;
            const float* k_t   = k_mem + t * d;
            const float* q_t   = q_mem + t * d;
            const float* d_y_t = d_y + t * d;
            const float* m_t   = m_recompute + tl * dd;
            const float* m_next = m_recompute + (tl + 1) * dd;
            const float* s_t   = s_recompute + tl * dd;
            const float* m_zero = m_recompute;  // M₀ for this chunk
            float alpha_t = alpha[t];
            float theta_t = theta[t];
            float eta_t   = eta[t];

            // d_M += outer(d_y_t, q_t)
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                d_M[idx] += d_y_t[i] * q_t[j];
            }
            __syncthreads();

            // d_q_t = M_{t+1}^T @ d_y_t
            for (int col = tid; col < d; col += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += m_next[i * d + col] * d_y_t[i];
                }
                d_q_mem[t * d + col] = sum;
            }
            __syncthreads();

            // d_S += d_M (S contributes additively to M)
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                d_S[idx] += d_M[idx];
            }
            __syncthreads();

            // d_alpha_t = -sum(M_t * d_M)
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < dd; idx += blockDim.x) {
                    local_sum += m_t[idx] * d_M[idx];
                }
                reduce_buf[tid] = local_sum;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                    __syncthreads();
                }
                if (tid == 0) d_alpha[t] = -reduce_buf[0];
                __syncthreads();
            }

            // d_M = (1-α) * d_M  (propagate through retention, NO error chain)
            float retention = 1.0f - alpha_t;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                d_M[idx] = retention * d_M[idx];
            }
            __syncthreads();

            // d_eta_t = sum(S_t * d_S)
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < dd; idx += blockDim.x) {
                    local_sum += s_t[idx] * d_S[idx];
                }
                reduce_buf[tid] = local_sum;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                    __syncthreads();
                }
                if (tid == 0) d_eta[t] = reduce_buf[0];
                __syncthreads();
            }

            // Load error for this token
            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = error_recompute[tl * d + row];
            }
            __syncthreads();

            // d_theta_t = -sum(outer(error, k) * d_S)
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < dd; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    local_sum += error_buf[i] * k_t[j] * d_S[idx];
                }
                reduce_buf[tid] = local_sum;
                __syncthreads();
                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                    __syncthreads();
                }
                if (tid == 0) d_theta[t] = -reduce_buf[0];
                __syncthreads();
            }

            // d_error[i] = sum_j (-θ * d_S[i,j]) * k_t[j]
            for (int row = tid; row < d; row += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += (-theta_t * d_S[row * d + j]) * k_t[j];
                }
                d_error_buf[row] = sum;
            }
            __syncthreads();

            // d_k[j] = sum_i (-θ * d_S[i,j]) * error[i]
            for (int col = tid; col < d; col += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += (-theta_t * d_S[i * d + col]) * error_buf[i];
                }
                d_k_mem[t * d + col] = sum;
            }
            __syncthreads();

            // d_k[j] += M₀^T @ d_error (from frozen M₀)
            for (int col = tid; col < d; col += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += m_zero[i * d + col] * d_error_buf[i];
                }
                d_k_mem[t * d + col] += sum;
            }

            // d_v = -d_error
            for (int row = tid; row < d; row += blockDim.x) {
                d_v_mem[t * d + row] = -d_error_buf[row];
            }
            __syncthreads();

            // d_S = η * d_S (propagate momentum)
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                d_S[idx] = eta_t * d_S[idx];
            }
            __syncthreads();

            // d_M₀ += outer(d_error, k)
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                d_M0[idx] += d_error_buf[i] * k_t[j];
            }
            __syncthreads();
        }

        // Propagate: d_M += d_M₀  (chunk-start gradient feeds previous chunk's d_M_final)
        // Also: d_S carries its accumulated value to the previous chunk.
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] += d_M0[idx];
        }
        __syncthreads();
    }

    // Accumulate into d_m_initial, d_s_initial (atomicAdd across batch)
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        atomicAdd(&d_m_initial[idx], d_M[idx]);
        atomicAdd(&d_s_initial[idx], d_S[idx]);
    }
}

extern "C" void titans_chunkwise_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_chunk_states, const float* s_chunk_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_eta,
    float* d_m_initial, float* d_s_initial,
    int seq_len, int d, int batch_size, int chunk_size, float error_clip)
{
    if (d <= 0) {
        fprintf(stderr, "titans_chunkwise_backward_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    if (chunk_size <= 0) {
        fprintf(stderr, "titans_chunkwise_backward_f32_cuda: chunk_size=%d must be > 0.\n",
                chunk_size);
        exit(1);
    }
    int dd = d * d;
    int block_size = (d < 512) ? d : 512;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 512) rounded = 512;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    int smem_bytes = (3 * d + block_size) * sizeof(float);
    if (smem_bytes > 163840) {
        fprintf(stderr, "titans_chunkwise_backward_f32_cuda: d=%d requires %d bytes smem.\n",
                d, smem_bytes);
        exit(1);
    }

    float* d_M_work = nullptr;
    float* d_S_work = nullptr;
    float* d_M0_work = nullptr;
    float* m_recompute = nullptr;
    float* s_recompute = nullptr;
    float* error_recompute = nullptr;

    check_cuda_alloc("titans_chunkwise_bwd: d_M",
                     cudaMalloc(&d_M_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_chunkwise_bwd: d_S",
                     cudaMalloc(&d_S_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_chunkwise_bwd: d_M0",
                     cudaMalloc(&d_M0_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_chunkwise_bwd: m_recompute",
                     cudaMalloc(&m_recompute, (size_t)batch_size * (chunk_size + 1) * dd * sizeof(float)));
    check_cuda_alloc("titans_chunkwise_bwd: s_recompute",
                     cudaMalloc(&s_recompute, (size_t)batch_size * (chunk_size + 1) * dd * sizeof(float)));
    check_cuda_alloc("titans_chunkwise_bwd: error_recompute",
                     cudaMalloc(&error_recompute, (size_t)batch_size * chunk_size * d * sizeof(float)));

    check_cuda_alloc("titans_chunkwise_bwd: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(titans_chunkwise_backward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    titans_chunkwise_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_chunk_states, s_chunk_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta,
        d_m_initial, d_s_initial,
        d_M_work, d_S_work, d_M0_work,
        m_recompute, s_recompute, error_recompute,
        seq_len, d, chunk_size, error_clip);
    check_cuda_launch("titans_chunkwise_backward_kernel", d, smem_bytes);

    check_cuda_alloc("titans_chunkwise_bwd: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(d_M_work);
    cudaFree(d_S_work);
    cudaFree(d_M0_work);
    cudaFree(m_recompute);
    cudaFree(s_recompute);
    cudaFree(error_recompute);
}
