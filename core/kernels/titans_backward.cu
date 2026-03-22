// TitansLMM Backward CUDA Kernel — S2-M1 Phase 2
//
// Backward pass with two interacting recurrences: d_M and d_S.
// Recomputes prediction/error from cached M_t states.
//
// Grid=(batch_size), Block=(min(d, 512)).
// All fp32.
//
// NOTE: d_M and d_S live in global memory (allocated via cudaMalloc in C wrapper),
// NOT shared memory. At d=512, d_M+d_S = 2MB — far exceeds GPU smem limits.
// Only small buffers (prediction[d], error[d], d_error[d], reduce_buf) in smem.
//
// Ampere+ (sm_80+) optimization:
//   When __CUDA_ARCH__ >= 800, the backward loop prefetches the PREVIOUS token's
//   k/v/q/d_y vectors via cp.async while computing on the current token.
//   The backward loop runs t = seq_len-1 down to 0, so "next to prefetch"
//   is token t-1. Double-buffered shared memory staging.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_clip.cuh"

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f32_bwd(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_bwd() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_bwd() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif // __CUDA_ARCH__ >= 800

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

// __launch_bounds__(512): nvcc caps register allocation to 65536/512 = 128 regs/thread,
// enabling launch at block_size=512 on sm_89/sm_90a where register file = 65536.
__launch_bounds__(512)
__global__ void titans_backward_kernel(
    const float* __restrict__ k_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ alpha,      // [batch_size, seq_len]
    const float* __restrict__ theta,      // [batch_size, seq_len]
    const float* __restrict__ eta,        // [batch_size, seq_len]
    const float* __restrict__ m_states,   // [batch_size, (seq_len+1)*d*d]
    const float* __restrict__ s_states,   // [batch_size, (seq_len+1)*d*d]
    const float* __restrict__ d_y,        // [batch_size, seq_len, d]
    float* __restrict__ d_k_mem,          // [batch_size, seq_len, d]
    float* __restrict__ d_v_mem,          // [batch_size, seq_len, d]
    float* __restrict__ d_q_mem,          // [batch_size, seq_len, d]
    float* __restrict__ d_alpha,          // [batch_size, seq_len]
    float* __restrict__ d_theta,          // [batch_size, seq_len]
    float* __restrict__ d_eta,            // [batch_size, seq_len]
    float* __restrict__ d_m_initial,      // [d*d] — summed across batch (atomicAdd)
    float* __restrict__ d_s_initial,      // [d*d] — summed across batch (atomicAdd)
    float* __restrict__ d_M,              // [batch_size, d*d] — per-batch accumulator
    float* __restrict__ d_S,              // [batch_size, d*d] — per-batch accumulator
    int seq_len, int d, float error_clip)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;
    int dd = d * d;

    // Offset per-batch pointers
    k_mem    += b * seq_len * d;
    v_mem    += b * seq_len * d;
    q_mem    += b * seq_len * d;
    alpha    += b * seq_len;
    theta    += b * seq_len;
    eta      += b * seq_len;
    m_states += b * (seq_len + 1) * dd;
    s_states += b * (seq_len + 1) * dd;
    d_y      += b * seq_len * d;
    d_k_mem  += b * seq_len * d;
    d_v_mem  += b * seq_len * d;
    d_q_mem  += b * seq_len * d;
    d_alpha  += b * seq_len;
    d_theta  += b * seq_len;
    d_eta    += b * seq_len;
    d_M      += b * dd;
    d_S      += b * dd;

    // Shared memory layout:
    //   Pre-Ampere: prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    //   Ampere+ (sm_80+):  prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    //                      + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;
    float* d_error = smem + 2 * d;
    float* reduce_buf = smem + 3 * d;

#if __CUDA_ARCH__ >= 800
    // Double-buffered vector staging (after reduce_buf)
    float* buf_k  = smem + 3 * d + blockDim.x;
    float* buf_v  = buf_k + 2 * d;
    float* buf_q  = buf_v + 2 * d;
    float* buf_dy = buf_q + 2 * d;
#endif

    // Init d_M = 0, d_S = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
        d_S[idx] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async prefetch for backward loop ──
    // Prefetch the last token (seq_len-1) into buffer 0
    int cur = 0;
    if (seq_len > 0) {
        int t_last = seq_len - 1;
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_bwd(&buf_k[0 * d + i],  &k_mem[t_last * d + i]);
            cp_async_f32_bwd(&buf_v[0 * d + i],  &v_mem[t_last * d + i]);
            cp_async_f32_bwd(&buf_q[0 * d + i],  &q_mem[t_last * d + i]);
            cp_async_f32_bwd(&buf_dy[0 * d + i], &d_y[t_last * d + i]);
        }
        cp_async_commit_bwd();
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int next = 1 - cur;

        // Prefetch token t-1 into alternate buffer
        if (t > 0) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_bwd(&buf_k[next * d + i],  &k_mem[(t - 1) * d + i]);
                cp_async_f32_bwd(&buf_v[next * d + i],  &v_mem[(t - 1) * d + i]);
                cp_async_f32_bwd(&buf_q[next * d + i],  &q_mem[(t - 1) * d + i]);
                cp_async_f32_bwd(&buf_dy[next * d + i], &d_y[(t - 1) * d + i]);
            }
            cp_async_commit_bwd();
        }

        // Wait for current buffer.
        // <1>: one prefetch still in flight (prev token). <0>: flush all on final iteration.
        if (t > 0) {
            cp_async_wait_bwd<1>();
        } else {
            cp_async_wait_bwd<0>();
        }
        __syncthreads();

        const float* k_t   = &buf_k[cur * d];
        const float* v_t   = &buf_v[cur * d];
        const float* q_t   = &buf_q[cur * d];
        const float* d_y_t = &buf_dy[cur * d];

#else
    // ── Pre-Ampere path: direct global memory access ──
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
#endif
        const float* m_t = m_states + t * dd;
        const float* m_next = m_states + (t + 1) * dd;
        const float* s_t = s_states + t * dd;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        float eta_t = eta[t];

        // d_M += outer(d_y_t, q_t)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] += d_y_t[i] * q_t[j];
        }
        __syncthreads();

        // d_q_t = M_{t+1}^T @ d_y_t (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + col] * d_y_t[i];
            }
            d_q_mem[t * d + col] = sum;
        }
        __syncthreads();

        // M_{t+1} = (1-alpha)*M_t + S_{t+1} backward
        // d_S += d_M (S contributes additively to M)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_S[idx] += d_M[idx];
        }
        __syncthreads();

        // d_alpha = -sum(M_t * d_M)
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

        // d_M_prev = (1-alpha) * d_M
        float retention = 1.0f - alpha_t;
        // We'll update d_M at the end after computing d_error

        // d_eta = sum(S_t * d_S)
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

        // Recompute prediction = M_t @ k, error = prediction - v (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) sum += m_t[row * d + j] * k_t[j];
            prediction[row] = sum;
        }
        __syncthreads();
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

        // d_theta = -sum(grad * d_S) where grad = outer(error, k)
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

        // d_grad = -theta * d_S
        // d_error[i] = sum_j d_grad[i,j] * k_t[j] (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_S[row * d + j]) * k_t[j];
            }
            d_error[row] = sum;
        }
        __syncthreads();

        // d_k[j] = sum_i d_grad[i,j] * error[i] (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_S[i * d + col]) * error_buf[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();

        // d_k[j] += sum_i M_t[i,j] * d_error[i] (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + col] * d_error[i];
            }
            d_k_mem[t * d + col] += sum;
        }

        // d_v[i] = -d_error[i] (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            d_v_mem[t * d + row] = -d_error[row];
        }
        __syncthreads();

        // d_S_prev = eta * d_S
        // d_M_prev = (1-alpha) * d_M + d_error[i] * k_t[j]
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_S[idx] = eta_t * d_S[idx];
            d_M[idx] = retention * d_M[idx] + d_error[i] * k_t[j];
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        cur = next;
#endif
    }

    // Accumulate d_m_initial and d_s_initial across batch elements (atomicAdd)
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        atomicAdd(&d_m_initial[idx], d_M[idx]);
        atomicAdd(&d_s_initial[idx], d_S[idx]);
    }
}

// ══════════════════════════════════════════════════════════════════════
// Segment backward: operates on [t_start, t_end) with d_m_seed/d_s_seed.
// m_states/s_states are segment-local: [(seg_len+1)*d*d].
// ══════════════════════════════════════════════════════════════════════

__launch_bounds__(512)
__global__ void titans_backward_segment_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ eta,
    const float* __restrict__ m_states,
    const float* __restrict__ s_states,
    const float* __restrict__ d_y,
    const float* __restrict__ d_m_seed,
    const float* __restrict__ d_s_seed,
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_theta,
    float* __restrict__ d_eta,
    float* __restrict__ d_m_out,
    float* __restrict__ d_s_out,
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    float* __restrict__ d_S,              // [d*d] — gradient accumulator in global memory
    int t_start, int t_end, int d, float error_clip)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;
    float* d_error = smem + 2 * d;
    float* reduce_buf = smem + 3 * d;

    // Init from seeds
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = d_m_seed[idx];
        d_S[idx] = d_s_seed[idx];
    }
    __syncthreads();

    for (int t = t_end - 1; t >= t_start; t--) {
        int seg_t = t - t_start;
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
        const float* m_t = m_states + seg_t * dd;
        const float* m_next = m_states + (seg_t + 1) * dd;
        const float* s_t = s_states + seg_t * dd;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        float eta_t = eta[t];

        // d_M += outer(d_y_t, q_t)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] += d_y_t[i] * q_t[j];
        }
        __syncthreads();

        // d_q_t (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + col] * d_y_t[i];
            }
            d_q_mem[t * d + col] = sum;
        }
        __syncthreads();

        // d_S += d_M
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_S[idx] += d_M[idx];
        }
        __syncthreads();

        // d_alpha
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

        float retention = 1.0f - alpha_t;

        // d_eta
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

        // Recompute prediction/error (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) sum += m_t[row * d + j] * k_t[j];
            prediction[row] = sum;
        }
        __syncthreads();
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

        // d_theta
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

        // d_error (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_S[row * d + j]) * k_t[j];
            }
            d_error[row] = sum;
        }
        __syncthreads();

        // d_k_mem (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_S[i * d + col]) * error_buf[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + col] * d_error[i];
            }
            d_k_mem[t * d + col] += sum;
        }

        // d_v_mem (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            d_v_mem[t * d + row] = -d_error[row];
        }
        __syncthreads();

        // Propagate d_S and d_M
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_S[idx] = eta_t * d_S[idx];
            d_M[idx] = retention * d_M[idx] + d_error[i] * k_t[j];
        }
        __syncthreads();
    }

    // Store outputs
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_out[idx] = d_M[idx];
        d_s_out[idx] = d_S[idx];
    }
}

extern "C" void titans_backward_segment_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_states, const float* s_states, const float* d_y,
    const float* d_m_seed, const float* d_s_seed,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_eta,
    float* d_m_out, float* d_s_out,
    int t_start, int t_end, int d, float error_clip)
{
    if (d <= 0) {
        fprintf(stderr, "titans_backward_segment_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    int dd = d * d;
    // Cap at min(d, 512): __launch_bounds__(512) constrains nvcc to 128 regs/thread.
    // Launching with >512 threads would need >65536 registers/block — exceeds sm_89/sm_90a
    // register file. For d > 512 the strided loop handles the extra elements.
    int block_size = (d < 512) ? d : 512;
    // Ceil to smallest power-of-2 >= block_size (required for tree reduction).
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 512) rounded = 512;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    // Segment kernel does NOT use cp.async — no double-buffer allocation needed.
    int smem_bytes = (3 * d + block_size) * sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "titans_backward_segment_f32_cuda: d=%d requires %d bytes shared memory (limit 163840).\n",
                d, smem_bytes);
        exit(1);
    }

    check_cuda_alloc("titans_backward_segment: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(titans_backward_segment_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // Allocate d_M and d_S workspaces
    float* d_M_work = nullptr;
    float* d_S_work = nullptr;
    check_cuda_alloc("titans_backward_segment: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));
    check_cuda_alloc("titans_backward_segment: cudaMalloc d_S_work",
                     cudaMalloc(&d_S_work, dd * sizeof(float)));

    titans_backward_segment_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_states, s_states, d_y,
        d_m_seed, d_s_seed,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta, d_eta,
        d_m_out, d_s_out,
        d_M_work, d_S_work, t_start, t_end, d, error_clip);
    check_cuda_launch("titans_backward_segment_kernel", d, smem_bytes);

    check_cuda_alloc("titans_backward_segment: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(d_M_work);
    cudaFree(d_S_work);
}

extern "C" void titans_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_states, const float* s_states, const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_eta,
    float* d_m_initial, float* d_s_initial,
    int seq_len, int d, int batch_size, float error_clip)
{
    if (d <= 0) {
        fprintf(stderr, "titans_backward_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    int dd = d * d;
    // Cap at min(d, 512): __launch_bounds__(512) constrains nvcc to 128 regs/thread.
    // Launching with >512 threads would need >65536 registers/block — exceeds sm_89/sm_90a
    // register file. For d > 512 the strided loop handles the extra elements.
    int block_size = (d < 512) ? d : 512;
    // Round up to next power of 2 for tree reduction correctness.
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 512) rounded = 512;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared memory layout:
    //   Pre-Ampere: prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    //   Ampere+ (sm_80+): + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    // Host allocates the maximum so the kernel works on any architecture.
    int smem_bytes = (3 * d + block_size + 8 * d) * sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "titans_backward_f32_cuda: d=%d requires %d bytes shared memory (limit 163840).\n",
                d, smem_bytes);
        exit(1);
    }

    // Ampere+ path may exceed the 48KB default dynamic shared memory limit at large d.
    check_cuda_alloc("titans_backward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(titans_backward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // Allocate per-batch d_M and d_S workspaces (batch_size * dd each).
    // d_m_initial and d_s_initial are zeroed by caller; accumulated via atomicAdd.
    float* d_M_work = nullptr;
    float* d_S_work = nullptr;
    check_cuda_alloc("titans_backward: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_backward: cudaMalloc d_S_work",
                     cudaMalloc(&d_S_work, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_backward: cudaMemset d_M_work",
                     cudaMemset(d_M_work, 0, (size_t)batch_size * dd * sizeof(float)));
    check_cuda_alloc("titans_backward: cudaMemset d_S_work",
                     cudaMemset(d_S_work, 0, (size_t)batch_size * dd * sizeof(float)));

    titans_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_states, s_states, d_y,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta, d_eta,
        d_m_initial, d_s_initial,
        d_M_work, d_S_work, seq_len, d, error_clip);
    check_cuda_launch("titans_backward_kernel", d, smem_bytes);

    check_cuda_alloc("titans_backward: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(d_M_work);
    cudaFree(d_S_work);
}
