// Delta Phase 2 Backward — Spec 44 (Batched cuBLAS Phase 1)
//
// Backward for ONE chunk, reading pre-computed errors and M trajectory.
// Called per-chunk (in reverse order) from Rust after cuBLAS error recompute.
//
// The forward recompute (Phase 1 errors + Phase 2 M trajectory) is done by:
//   1. cuBLAS GEMM for Phase 1 error recompute (Rust orchestration)
//   2. This kernel's internal Phase 2 recompute (M trajectory from M₀ + errors)
//   3. Reverse token loop for gradients
//
// Grid=(batch_size), Block=(min(d, 512)), __launch_bounds__(512).
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

__launch_bounds__(512)
__global__ void delta_phase2_backward_kernel(
    const float* __restrict__ k_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ alpha,          // [batch_size, seq_len]
    const float* __restrict__ theta,          // [batch_size, seq_len]
    const float* __restrict__ errors,         // [batch_size, chunk_size, d] — pre-computed
    const float* __restrict__ m_chunk_states, // [batch_size, (num_chunks+1)*d*d]
    const float* __restrict__ d_y,            // [batch_size, seq_len, d]
    float* __restrict__ d_k_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_v_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_q_mem,              // [batch_size, seq_len, d]
    float* __restrict__ d_alpha,              // [batch_size, seq_len]
    float* __restrict__ d_theta,              // [batch_size, seq_len]
    float* __restrict__ d_M,                  // [batch_size, d*d] — recurrence accumulator (persistent)
    float* __restrict__ d_M0,                 // [batch_size, d*d] — frozen-M₀ accumulator
    float* __restrict__ m_recompute,          // [batch_size, (chunk_size+1)*d*d]
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
    const float* mcs_b   = m_chunk_states + b * (num_chunks + 1) * dd;
    const float* dy_b    = d_y + b * seq_len * d;
    float* dk_b          = d_k_mem + b * seq_len * d;
    float* dv_b          = d_v_mem + b * seq_len * d;
    float* dq_b          = d_q_mem + b * seq_len * d;
    float* dalpha_b      = d_alpha + b * seq_len;
    float* dtheta_b      = d_theta + b * seq_len;
    float* dM_b          = d_M + b * dd;
    float* dM0_b         = d_M0 + b * dd;
    float* mrecomp_b     = m_recompute + b * (chunk_size + 1) * dd;

    // Shared memory: error_buf[d] + d_error[d] + reduce_buf[blockDim.x]
    extern __shared__ float smem[];
    float* error_buf   = smem;
    float* d_error_buf = smem + d;
    float* reduce_buf  = smem + 2 * d;

    // ── Recompute M trajectory from M₀ + pre-computed errors ──
    // Load M₀ for this chunk
    int cs_off = chunk_idx * dd;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        mrecomp_b[idx] = mcs_b[cs_off + idx];
    }
    __syncthreads();

    // Phase 2 recompute: build M trajectory from pre-computed errors
    for (int tl = 0; tl < C; tl++) {
        int t = t_start + tl;
        const float* k_t = k_b + t * d;
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];

        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = err_b[tl * d + row];
        }
        __syncthreads();

        float retention = 1.0f - alpha_t;
        int m_cur  = tl * dd;
        int m_next = (tl + 1) * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            mrecomp_b[m_next + idx] = retention * mrecomp_b[m_cur + idx]
                                      - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // Per-token M-norm projection (spec 74, matches forward replay)
        m_norm_project_inplace(&mrecomp_b[m_next], error_buf, dd, tid, m_norm_max);
    }

    // ── Initialize d_M₀ accumulator ──
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        dM0_b[idx] = 0.0f;
    }
    __syncthreads();

    // ── Backward: reverse token loop ──
    for (int tl = C - 1; tl >= 0; tl--) {
        int t = t_start + tl;
        const float* k_t   = k_b + t * d;
        const float* q_t   = q_b + t * d;
        const float* dy_t  = dy_b + t * d;
        const float* m_t   = mrecomp_b + tl * dd;
        const float* m_next = mrecomp_b + (tl + 1) * dd;
        const float* m_zero = mrecomp_b;  // M₀ for this chunk
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];

        // d_M += outer(d_y_t, q_t)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            dM_b[idx] += dy_t[i] * q_t[j];
        }
        __syncthreads();

        // d_q_t = M_{t+1}^T @ d_y_t
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + col] * dy_t[i];
            }
            dq_b[t * d + col] = sum;
        }
        __syncthreads();

        // d_alpha_t = -sum(M_t * d_M)
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                local_sum += m_t[idx] * dM_b[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) dalpha_b[t] = -reduce_buf[0];
            __syncthreads();
        }

        // Reload error
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = err_b[tl * d + row];
        }
        __syncthreads();

        // d_theta_t = -sum(outer(error, k) * d_M)
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                local_sum += error_buf[i] * k_t[j] * dM_b[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) dtheta_b[t] = -reduce_buf[0];
            __syncthreads();
        }

        // d_error = -θ (d_M @ k)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * dM_b[row * d + j]) * k_t[j];
            }
            d_error_buf[row] = sum;
        }
        __syncthreads();

        // d_k_t = -θ error^T @ d_M
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * dM_b[i * d + col]) * error_buf[i];
            }
            dk_b[t * d + col] = sum;
        }
        __syncthreads();

        // d_k_t += M₀^T @ d_error
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_zero[i * d + col] * d_error_buf[i];
            }
            dk_b[t * d + col] += sum;
        }

        // d_v_t = -d_error
        for (int row = tid; row < d; row += blockDim.x) {
            dv_b[t * d + row] = -d_error_buf[row];
        }
        __syncthreads();

        // d_M = (1-α) d_M  (no outer(d_error, k) — goes to d_M₀)
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            dM_b[idx] = retention * dM_b[idx];
        }
        __syncthreads();

        // d_M₀ += outer(d_error, k)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            dM0_b[idx] += d_error_buf[i] * k_t[j];
        }
        __syncthreads();
    }

    // Propagate: d_M += d_M₀  (d_M₀ feeds into previous chunk's M_final gradient)
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        dM_b[idx] += dM0_b[idx];
    }
    __syncthreads();
}

extern "C" void delta_phase2_backward_f32_cuda(
    const float* k_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* errors, const float* m_chunk_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta,
    float* d_M, float* d_M0, float* m_recompute,
    int seq_len, int d, int batch_size, int chunk_size, int chunk_idx,
    float m_norm_max)
{
    int block_size = (d < 512) ? d : 512;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 512) rounded = 512;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    int smem_bytes = (2 * d + block_size) * sizeof(float);

    delta_phase2_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, q_mem, alpha, theta, errors, m_chunk_states,
        d_y, d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta,
        d_M, d_M0, m_recompute,
        seq_len, d, chunk_size, chunk_idx, m_norm_max);
    check_cuda_launch("delta_phase2_backward_kernel", d, smem_bytes);
}
