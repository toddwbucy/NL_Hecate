// Titans Phase 2 Backward — Spec 44 (Batched cuBLAS Phase 1)
//
// Backward for ONE chunk, reading pre-computed errors.
// Called per-chunk (reverse order) from Rust after cuBLAS error recompute.
//
// Phase 2 recompute (M+S trajectory) + reverse token loop with
// d_M, d_S, d_M₀ accumulators.
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
__global__ void titans_phase2_backward_kernel(
    const float* __restrict__ k_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,          // [batch_size, seq_len, d]
    const float* __restrict__ alpha,          // [batch_size, seq_len]
    const float* __restrict__ theta,          // [batch_size, seq_len]
    const float* __restrict__ eta,            // [batch_size, seq_len]
    const float* __restrict__ errors,         // [batch_size, chunk_size, d] — pre-computed
    const float* __restrict__ m_chunk_states, // [batch_size, (num_chunks+1)*d*d]
    const float* __restrict__ s_chunk_states, // [batch_size, (num_chunks+1)*d*d]
    const float* __restrict__ d_y,            // [batch_size, seq_len, d]
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_theta,
    float* __restrict__ d_eta,
    float* __restrict__ d_M,                  // [batch_size, d*d] — persistent
    float* __restrict__ d_S,                  // [batch_size, d*d] — persistent
    float* __restrict__ d_M0,                 // [batch_size, d*d]
    float* __restrict__ m_recompute,          // [batch_size, (chunk_size+1)*d*d]
    float* __restrict__ s_recompute,          // [batch_size, (chunk_size+1)*d*d]
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
    const float* eta_b   = eta + b * seq_len;
    const float* err_b   = errors + b * chunk_size * d;
    const float* mcs_b   = m_chunk_states + b * (num_chunks + 1) * dd;
    const float* scs_b   = s_chunk_states + b * (num_chunks + 1) * dd;
    const float* dy_b    = d_y + b * seq_len * d;
    float* dk_b          = d_k_mem + b * seq_len * d;
    float* dv_b          = d_v_mem + b * seq_len * d;
    float* dq_b          = d_q_mem + b * seq_len * d;
    float* dalpha_b      = d_alpha + b * seq_len;
    float* dtheta_b      = d_theta + b * seq_len;
    float* deta_b        = d_eta + b * seq_len;
    float* dM_b          = d_M + b * dd;
    float* dS_b          = d_S + b * dd;
    float* dM0_b         = d_M0 + b * dd;
    float* mrecomp_b     = m_recompute + b * (chunk_size + 1) * dd;
    float* srecomp_b     = s_recompute + b * (chunk_size + 1) * dd;

    extern __shared__ float smem[];
    float* error_buf   = smem;
    float* d_error_buf = smem + d;
    float* reduce_buf  = smem + 2 * d;

    // ── Recompute M+S trajectory from chunk states + pre-computed errors ──
    int cs_off = chunk_idx * dd;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        mrecomp_b[idx] = mcs_b[cs_off + idx];
        srecomp_b[idx] = scs_b[cs_off + idx];
    }
    __syncthreads();

    for (int tl = 0; tl < C; tl++) {
        int t = t_start + tl;
        const float* k_t = k_b + t * d;
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];
        float eta_t   = eta_b[t];

        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = err_b[tl * d + row];
        }
        __syncthreads();

        float retention = 1.0f - alpha_t;
        int cur  = tl * dd;
        int next = (tl + 1) * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            float s_new = eta_t * srecomp_b[cur + idx]
                          - theta_t * error_buf[i] * k_t[j];
            srecomp_b[next + idx] = s_new;
            mrecomp_b[next + idx] = retention * mrecomp_b[cur + idx] + s_new;
        }
        __syncthreads();

        // Per-token M-norm projection (spec 74, matches forward replay)
        m_norm_project_inplace(&mrecomp_b[next], error_buf, dd, tid, m_norm_max);
    }

    // ── Initialize d_M₀ ──
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
        const float* s_t   = srecomp_b + tl * dd;
        const float* m_zero = mrecomp_b;
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];
        float eta_t   = eta_b[t];

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

        // d_S += d_M
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            dS_b[idx] += dM_b[idx];
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

        // d_M = (1-α) d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            dM_b[idx] = retention * dM_b[idx];
        }
        __syncthreads();

        // d_eta_t = sum(S_t * d_S)
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                local_sum += s_t[idx] * dS_b[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) deta_b[t] = reduce_buf[0];
            __syncthreads();
        }

        // Load error
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = err_b[tl * d + row];
        }
        __syncthreads();

        // d_theta_t = -sum(outer(error, k) * d_S)
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                local_sum += error_buf[i] * k_t[j] * dS_b[idx];
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

        // d_error = -θ (d_S @ k)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * dS_b[row * d + j]) * k_t[j];
            }
            d_error_buf[row] = sum;
        }
        __syncthreads();

        // d_k = -θ error^T @ d_S
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * dS_b[i * d + col]) * error_buf[i];
            }
            dk_b[t * d + col] = sum;
        }
        __syncthreads();

        // d_k += M₀^T @ d_error
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_zero[i * d + col] * d_error_buf[i];
            }
            dk_b[t * d + col] += sum;
        }

        // d_v = -d_error
        for (int row = tid; row < d; row += blockDim.x) {
            dv_b[t * d + row] = -d_error_buf[row];
        }
        __syncthreads();

        // d_S = η d_S
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            dS_b[idx] = eta_t * dS_b[idx];
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

    // d_M += d_M₀
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        dM_b[idx] += dM0_b[idx];
    }
    __syncthreads();
}

extern "C" void titans_phase2_backward_f32_cuda(
    const float* k_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* errors,
    const float* m_chunk_states, const float* s_chunk_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_eta,
    float* d_M, float* d_S, float* d_M0,
    float* m_recompute, float* s_recompute,
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

    titans_phase2_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, q_mem, alpha, theta, eta, errors,
        m_chunk_states, s_chunk_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta,
        d_M, d_S, d_M0, m_recompute, s_recompute,
        seq_len, d, chunk_size, chunk_idx, m_norm_max);
    check_cuda_launch("titans_phase2_backward_kernel", d, smem_bytes);
}
