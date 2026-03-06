// HebbianRule Backward CUDA Kernel — S2-M1 Phase 3
//
// Simplest backward: no error/prediction chain, no theta.
// d_M propagates through retention and outer product only.
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.
//
// NOTE: d_M lives in global memory (allocated via cudaMalloc in C wrapper),
// NOT shared memory. At d=512, d_M[d*d] = 1MB — exceeds GPU smem limits.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f32_hebb_bwd(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_hebb_bwd() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_hebb_bwd() {
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

__global__ void hebbian_backward_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ m_states,
    const float* __restrict__ d_y,
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_m_initial,
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // ── Shared memory layout ──
    // Pre-Ampere: reduce_buf[blockDim.x]
    // Ampere+ (sm_80+):   reduce_buf[blockDim.x] + k_buf[2*d] + v_buf[2*d]
    //                    + q_buf[2*d] + dy_buf[2*d]
    extern __shared__ float smem[];
    float* reduce_buf = smem;

#if __CUDA_ARCH__ >= 800
    // Double-buffered vector staging (after reduce_buf)
    float* buf_k  = smem + blockDim.x;
    float* buf_v  = buf_k + 2 * d;
    float* buf_q  = buf_v + 2 * d;
    float* buf_dy = buf_q + 2 * d;
#endif

    // Init d_M = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async prefetch for backward loop ──
    // Prefetch the last token (seq_len-1) into buffer 0
    int cur = 0;
    int t_last = seq_len - 1;
    if (seq_len > 0) {
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_hebb_bwd(&buf_k[0 * d + i],  &k_mem[t_last * d + i]);
            cp_async_f32_hebb_bwd(&buf_v[0 * d + i],  &v_mem[t_last * d + i]);
            cp_async_f32_hebb_bwd(&buf_q[0 * d + i],  &q_mem[t_last * d + i]);
            cp_async_f32_hebb_bwd(&buf_dy[0 * d + i], &d_y[t_last * d + i]);
        }
        cp_async_commit_hebb_bwd();
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        int next = 1 - cur;

        // Prefetch token t-1 into alternate buffer
        if (t > 0) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_hebb_bwd(&buf_k[next * d + i],  &k_mem[(t - 1) * d + i]);
                cp_async_f32_hebb_bwd(&buf_v[next * d + i],  &v_mem[(t - 1) * d + i]);
                cp_async_f32_hebb_bwd(&buf_q[next * d + i],  &q_mem[(t - 1) * d + i]);
                cp_async_f32_hebb_bwd(&buf_dy[next * d + i], &d_y[(t - 1) * d + i]);
            }
            cp_async_commit_hebb_bwd();
        }

        // Wait for current buffer
        if (t > 0) {
            cp_async_wait_hebb_bwd<1>();
        } else {
            cp_async_wait_hebb_bwd<0>();
        }
        __syncthreads();

        const float* k_t   = &buf_k[cur * d];
        const float* v_t   = &buf_v[cur * d];
        const float* q_t   = &buf_q[cur * d];
        const float* d_y_t = &buf_dy[cur * d];
        const float* m_t = m_states + t * dd;
        const float* m_next = m_states + (t + 1) * dd;
        float alpha_t = alpha[t];

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

        // d_v[i] = sum_j d_M[i,j] * k_t[j] (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_M[row * d + j] * k_t[j];
            }
            d_v_mem[t * d + row] = sum;
        }

        // d_k[j] = sum_i d_M[i,j] * v_t[i] (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_M[i * d + col] * v_t[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();

        // d_M_prev = (1-alpha) * d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] = retention * d_M[idx];
        }
        __syncthreads();

        cur = next;
    }

#else
    // ── Pre-Ampere path: direct global memory access ──
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
        const float* m_t = m_states + t * dd;
        const float* m_next = m_states + (t + 1) * dd;
        float alpha_t = alpha[t];

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

        // d_v[i] = sum_j d_M[i,j] * k_t[j] (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_M[row * d + j] * k_t[j];
            }
            d_v_mem[t * d + row] = sum;
        }

        // d_k[j] = sum_i d_M[i,j] * v_t[i] (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_M[i * d + col] * v_t[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();

        // d_M_prev = (1-alpha) * d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] = retention * d_M[idx];
        }
        __syncthreads();
    }
#endif // __CUDA_ARCH__ >= 800

    // Store d_m_initial
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_initial[idx] = d_M[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════
// Segment backward: operates on [t_start, t_end) with d_m_seed.
// m_states is segment-local: [(seg_len+1)*d*d].
// ══════════════════════════════════════════════════════════════════════

__global__ void hebbian_backward_segment_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ m_states,
    const float* __restrict__ d_y,
    const float* __restrict__ d_m_seed,
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_m_out,
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    int t_start, int t_end, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* reduce_buf = smem;

    // Init from seed
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = d_m_seed[idx];
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
        float alpha_t = alpha[t];

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

        // d_v, d_k from outer product (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_M[row * d + j] * k_t[j];
            }
            d_v_mem[t * d + row] = sum;
        }
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_M[i * d + col] * v_t[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();

        // d_M_prev = (1-alpha) * d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] = retention * d_M[idx];
        }
        __syncthreads();
    }

    // Store output
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_out[idx] = d_M[idx];
    }
}

extern "C" void hebbian_backward_segment_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_states, const float* d_y,
    const float* d_m_seed,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_m_out,
    int t_start, int t_end, int d)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "hebbian_backward_segment_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Segment kernel uses only reduce_buf[block_size], no cp.async.
    int smem_bytes = block_size * sizeof(float);

    float* d_M_work = nullptr;
    check_cuda_alloc("hebbian_backward_segment: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));

    hebbian_backward_segment_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_states, d_y,
        d_m_seed,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_m_out,
        d_M_work, t_start, t_end, d);
    check_cuda_launch("hebbian_backward_segment_kernel", d, smem_bytes);

    check_cuda_alloc("hebbian_backward_segment: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree d_M_work", cudaFree(d_M_work));
}

extern "C" void hebbian_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_states, const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_m_initial,
    int seq_len, int d)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "hebbian_backward_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Legacy: reduce_buf[block_size]
    //   Ampere+: + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    // Host allocates the maximum so the kernel works on any architecture.
    int smem_bytes = (block_size + 8 * d) * sizeof(float);

    float* d_M_work = nullptr;
    check_cuda_alloc("hebbian_backward: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));

    check_cuda_alloc("hebbian_backward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(hebbian_backward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    hebbian_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_m_initial,
        d_M_work, seq_len, d);
    check_cuda_launch("hebbian_backward_kernel", d, smem_bytes);

    check_cuda_alloc("hebbian_backward: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree d_M_work", cudaFree(d_M_work));
}
