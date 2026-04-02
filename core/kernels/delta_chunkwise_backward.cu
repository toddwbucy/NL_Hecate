// DeltaRule Chunkwise Backward — Spec 43 (Paper-Aligned Frozen-M₀)
//
// Backward pass for the chunkwise frozen-M₀ forward.
// Key simplification over full BPTT: error gradients accumulate into
// d_M₀ (per-chunk) instead of feeding back into the d_M recurrence chain.
//
// Per chunk (reverse order):
//   1. Reload M₀ from m_chunk_states
//   2. Recompute forward (Phase 1 + Phase 2) to rebuild M_t trajectory
//   3. Reverse token loop through the chunk:
//      - d_M += outer(d_y_t, q_t)           (from y = M @ q)
//      - d_q_t = M_t^T @ d_y_t
//      - d_α_t = -sum(M_{t-1} * d_M)        (retention gradient)
//      - d_θ_t = -sum(outer(error_t, k_t) * d_M)
//      - d_error_t = -θ * (d_M @ k)
//      - d_k_t += -θ * error^T @ d_M        (from outer product)
//      - d_k_t += M₀^T @ d_error_t          (from error = M₀@k - v)
//      - d_v_t = -d_error_t
//      - d_M = (1-α) * d_M                  (NO outer(d_error,k) — goes to d_M₀)
//      - d_M₀ += outer(d_error_t, k_t)      (accumulated, not fed back)
//   4. Propagate d_M₀ to previous chunk's d_M_final
//
// Grid=(batch_size), Block=(min(d, 512)), __launch_bounds__(512).
// All fp32.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_clip.cuh"
#include "m_norm_project.cuh"

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
__global__ void delta_chunkwise_backward_kernel(
    const float* __restrict__ k_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ alpha,          // [batch_size, seq_len]
    const float* __restrict__ theta,          // [batch_size, seq_len]
    const float* __restrict__ m_chunk_states, // [batch_size, (num_chunks+1)*d*d]
    const float* __restrict__ d_y,            // [batch_size, seq_len, d]
    float* __restrict__ d_k_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_v_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_q_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_alpha,              // [batch_size, seq_len]
    float* __restrict__ d_theta,              // [batch_size, seq_len]
    float* __restrict__ d_m_initial,          // [d*d] — summed across batch (atomicAdd)
    float* __restrict__ d_M,                  // [batch_size, d*d] — recurrence accumulator
    float* __restrict__ d_M0,                 // [batch_size, d*d] — frozen-M₀ accumulator
    float* __restrict__ m_recompute,          // [batch_size, (chunk_size+1)*d*d] — recomputed trajectory
    float* __restrict__ error_recompute,      // [batch_size, chunk_size*d] — recomputed errors
    int seq_len, int d, int chunk_size, float error_clip,
    float m_norm_max)
{
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int dd = d * d;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // Offset to batch element
    k_mem          += b * seq_len * d;
    v_mem          += b * seq_len * d;
    q_mem          += b * seq_len * d;
    alpha          += b * seq_len;
    theta          += b * seq_len;
    m_chunk_states += b * (num_chunks + 1) * dd;
    d_y            += b * seq_len * d;
    d_k_mem        += b * seq_len * d;
    d_v_mem        += b * seq_len * d;
    d_q_mem        += b * seq_len * d;
    d_alpha        += b * seq_len;
    d_theta        += b * seq_len;
    d_M            += b * dd;
    d_M0           += b * dd;
    m_recompute    += b * (chunk_size + 1) * dd;
    error_recompute += b * chunk_size * d;

    // Shared memory: prediction[d] + error_buf[d] + d_error[d] + reduce_buf[blockDim.x]
    extern __shared__ float smem[];
    float* prediction  = smem;
    float* error_buf   = smem + d;
    float* d_error_buf = smem + 2 * d;
    float* reduce_buf  = smem + 3 * d;

    // Initialize d_M = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
    }
    __syncthreads();

    // Process chunks in reverse order
    for (int c = num_chunks - 1; c >= 0; c--) {
        int t_start = c * chunk_size;
        int t_end   = t_start + chunk_size;
        if (t_end > seq_len) t_end = seq_len;
        int C = t_end - t_start;

        // ── Recompute forward: rebuild M trajectory from M₀ ──
        // Load M₀ for this chunk
        int cs_off = c * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_recompute[idx] = m_chunk_states[cs_off + idx];
        }
        __syncthreads();

        // Phase 1: recompute errors against frozen M₀
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

        // Phase 2: recompute M trajectory (M₀ already in m_recompute[0*dd])
        // We need M_t at each step for the backward, so store full trajectory
        for (int tl = 0; tl < C; tl++) {
            int t = t_start + tl;
            const float* k_t = k_mem + t * d;
            float alpha_t = alpha[t];
            float theta_t = theta[t];

            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = error_recompute[tl * d + row];
            }
            __syncthreads();

            // M_{t+1} = (1-α) M_t - θ outer(error, k)
            float retention = 1.0f - alpha_t;
            int m_cur  = tl * dd;
            int m_next = (tl + 1) * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                m_recompute[m_next + idx] = retention * m_recompute[m_cur + idx]
                                            - theta_t * error_buf[i] * k_t[j];
            }
            __syncthreads();

            // Per-token M-norm projection (spec 74, matches forward replay)
            m_norm_project_inplace(&m_recompute[m_next], error_buf, dd, tid, m_norm_max);
        }

        // ── Initialize d_M₀ accumulator for this chunk ──
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M0[idx] = 0.0f;
        }
        __syncthreads();

        // ── Backward: reverse token loop through chunk ──
        for (int tl = C - 1; tl >= 0; tl--) {
            int t = t_start + tl;
            const float* k_t   = k_mem + t * d;
            const float* q_t   = q_mem + t * d;
            const float* d_y_t = d_y + t * d;
            const float* m_t   = m_recompute + tl * dd;
            const float* m_next = m_recompute + (tl + 1) * dd;
            // M₀ for this chunk is always m_recompute[0]
            const float* m_zero = m_recompute;
            float alpha_t = alpha[t];
            float theta_t = theta[t];

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

            // Reload pre-computed error for this token
            for (int row = tid; row < d; row += blockDim.x) {
                error_buf[row] = error_recompute[tl * d + row];
            }
            __syncthreads();

            // d_theta_t = -sum(outer(error, k) * d_M)
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < dd; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    local_sum += error_buf[i] * k_t[j] * d_M[idx];
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

            // d_error[i] = sum_j (-θ * d_M[i,j]) * k_t[j]
            for (int row = tid; row < d; row += blockDim.x) {
                float sum = 0.0f;
                for (int j = 0; j < d; j++) {
                    sum += (-theta_t * d_M[row * d + j]) * k_t[j];
                }
                d_error_buf[row] = sum;
            }
            __syncthreads();

            // d_k_t[j] = sum_i (-θ * d_M[i,j]) * error[i]
            for (int col = tid; col < d; col += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += (-theta_t * d_M[i * d + col]) * error_buf[i];
                }
                d_k_mem[t * d + col] = sum;
            }
            __syncthreads();

            // d_k_t[j] += M₀^T @ d_error  (from error = M₀@k - v, chain to k)
            for (int col = tid; col < d; col += blockDim.x) {
                float sum = 0.0f;
                for (int i = 0; i < d; i++) {
                    sum += m_zero[i * d + col] * d_error_buf[i];
                }
                d_k_mem[t * d + col] += sum;
            }

            // d_v_t = -d_error
            for (int row = tid; row < d; row += blockDim.x) {
                d_v_mem[t * d + row] = -d_error_buf[row];
            }
            __syncthreads();

            // Propagate d_M backward through recurrence:
            //   d_M_{t-1} = (1-α) * d_M
            // KEY DIFFERENCE from full BPTT: NO outer(d_error, k) added here.
            // That gradient goes to d_M₀ instead (frozen-M₀ formulation).
            float retention = 1.0f - alpha_t;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                d_M[idx] = retention * d_M[idx];
            }
            __syncthreads();

            // d_M₀ += outer(d_error, k)  (accumulated across all tokens in chunk)
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                d_M0[idx] += d_error_buf[i] * k_t[j];
            }
            __syncthreads();
        }

        // ── Propagate: d_M for previous chunk = d_M (from recurrence) + d_M₀ ──
        // d_M₀ is the gradient of loss w.r.t. the chunk-start state M₀.
        // This M₀ was the M_final of the previous chunk, so d_M₀ feeds into
        // the previous chunk's d_M_final = d_M (propagated) + d_M₀ (accumulated).
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] += d_M0[idx];
        }
        __syncthreads();
    }

    // d_M now holds the gradient w.r.t. m_initial — accumulate across batch
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        atomicAdd(&d_m_initial[idx], d_M[idx]);
    }
}

extern "C" void delta_chunkwise_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* m_chunk_states, const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_m_initial,
    int seq_len, int d, int batch_size, int chunk_size, float error_clip,
    float m_norm_max)
{
    if (d <= 0) {
        fprintf(stderr, "delta_chunkwise_backward_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    if (chunk_size <= 0) {
        fprintf(stderr, "delta_chunkwise_backward_f32_cuda: chunk_size=%d must be > 0.\n",
                chunk_size);
        exit(1);
    }
    int dd = d * d;
    // Cap at min(d, 512) — __launch_bounds__(512) constrains to 128 regs/thread
    int block_size = (d < 512) ? d : 512;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 512) rounded = 512;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    int smem_bytes = (3 * d + block_size) * sizeof(float);
    if (smem_bytes > 163840) {
        fprintf(stderr, "delta_chunkwise_backward_f32_cuda: d=%d requires %d bytes smem (limit 163840).\n",
                d, smem_bytes);
        exit(1);
    }

    // Allocate workspaces
    float* d_M_work = nullptr;
    float* d_M0_work = nullptr;
    float* m_recompute = nullptr;
    float* error_recompute = nullptr;

    check_cuda_alloc("delta_chunkwise_bwd: cudaMalloc d_M",
                     cudaMalloc(&d_M_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("delta_chunkwise_bwd: cudaMalloc d_M0",
                     cudaMalloc(&d_M0_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("delta_chunkwise_bwd: cudaMalloc m_recompute",
                     cudaMalloc(&m_recompute, (size_t)batch_size * (chunk_size + 1) * dd * sizeof(float)));
    check_cuda_alloc("delta_chunkwise_bwd: cudaMalloc error_recompute",
                     cudaMalloc(&error_recompute, (size_t)batch_size * chunk_size * d * sizeof(float)));

    check_cuda_alloc("delta_chunkwise_bwd: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(delta_chunkwise_backward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    delta_chunkwise_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta,
        m_chunk_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
        d_M_work, d_M0_work, m_recompute, error_recompute,
        seq_len, d, chunk_size, error_clip, m_norm_max);
    check_cuda_launch("delta_chunkwise_backward_kernel", d, smem_bytes);

    check_cuda_alloc("delta_chunkwise_bwd: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(d_M_work);
    cudaFree(d_M0_work);
    cudaFree(m_recompute);
    cudaFree(error_recompute);
}
