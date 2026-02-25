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
#include <stdio.h>
#include <stdlib.h>

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

    // Shared: only reduce_buf[blockDim.x]
    extern __shared__ float smem[];
    float* reduce_buf = smem;

    // Init d_M = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
    }
    __syncthreads();

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

        // d_q_t = M_{t+1}^T @ d_y_t
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + tid] * d_y_t[i];
            }
            d_q_mem[t * d + tid] = sum;
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

        // d_v[i] = sum_j d_M[i,j] * k_t[j] (from outer product)
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_M[tid * d + j] * k_t[j];
            }
            d_v_mem[t * d + tid] = sum;
        }

        // d_k[j] = sum_i d_M[i,j] * v_t[i] (from outer product)
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_M[i * d + tid] * v_t[i];
            }
            d_k_mem[t * d + tid] = sum;
        }
        __syncthreads();

        // d_M_prev = (1-alpha) * d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            d_M[idx] = retention * d_M[idx];
        }
        __syncthreads();
    }

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

        // d_q_t
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + tid] * d_y_t[i];
            }
            d_q_mem[t * d + tid] = sum;
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

        // d_v, d_k from outer product
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_M[tid * d + j] * k_t[j];
            }
            d_v_mem[t * d + tid] = sum;
        }
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_M[i * d + tid] * v_t[i];
            }
            d_k_mem[t * d + tid] = sum;
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
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared: reduce_buf[block_size] only
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
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared: reduce_buf[block_size] only
    int smem_bytes = block_size * sizeof(float);

    float* d_M_work = nullptr;
    check_cuda_alloc("hebbian_backward: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));

    hebbian_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_m_initial,
        d_M_work, seq_len, d);
    check_cuda_launch("hebbian_backward_kernel", d, smem_bytes);

    check_cuda_alloc("hebbian_backward: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree d_M_work", cudaFree(d_M_work));
}
