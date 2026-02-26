// TitansLMM Backward CUDA Kernel — S2-M1 Phase 2
//
// Backward pass with two interacting recurrences: d_M and d_S.
// Recomputes prediction/error from cached M_t states.
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.
//
// NOTE: d_M and d_S live in global memory (allocated via cudaMalloc in C wrapper),
// NOT shared memory. At d=512, d_M+d_S = 2MB — far exceeds GPU smem limits.
// Only small buffers (prediction[d], error[d], d_error[d], reduce_buf) in smem.

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

__global__ void titans_backward_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ eta,
    const float* __restrict__ m_states,
    const float* __restrict__ s_states,
    const float* __restrict__ d_y,
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_theta,
    float* __restrict__ d_eta,
    float* __restrict__ d_m_initial,
    float* __restrict__ d_s_initial,
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    float* __restrict__ d_S,              // [d*d] — gradient accumulator in global memory
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared: prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;
    float* d_error = smem + 2 * d;
    float* reduce_buf = smem + 3 * d;

    // Init d_M = 0, d_S = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
        d_S[idx] = 0.0f;
    }
    __syncthreads();

    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
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

        // d_q_t = M_{t+1}^T @ d_y_t
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + tid] * d_y_t[i];
            }
            d_q_mem[t * d + tid] = sum;
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

        // Recompute prediction = M_t @ k, error = prediction - v
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) sum += m_t[tid * d + j] * k_t[j];
            prediction[tid] = sum;
        }
        __syncthreads();
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

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
        // d_error[i] = sum_j d_grad[i,j] * k_t[j]
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_S[tid * d + j]) * k_t[j];
            }
            d_error[tid] = sum;
        }
        __syncthreads();

        // d_k[j] = sum_i d_grad[i,j] * error[i]
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_S[i * d + tid]) * error_buf[i];
            }
            d_k_mem[t * d + tid] = sum;
        }
        __syncthreads();

        // d_k[j] += sum_i M_t[i,j] * d_error[i] (from prediction chain)
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + tid] * d_error[i];
            }
            d_k_mem[t * d + tid] += sum;
        }

        // d_v[i] = -d_error[i]
        if (tid < d) {
            d_v_mem[t * d + tid] = -d_error[tid];
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
    }

    // Store d_m_initial and d_s_initial
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_initial[idx] = d_M[idx];
        d_s_initial[idx] = d_S[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════
// Segment backward: operates on [t_start, t_end) with d_m_seed/d_s_seed.
// m_states/s_states are segment-local: [(seg_len+1)*d*d].
// ══════════════════════════════════════════════════════════════════════

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
    int t_start, int t_end, int d)
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

        // d_q_t
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + tid] * d_y_t[i];
            }
            d_q_mem[t * d + tid] = sum;
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

        // Recompute prediction/error
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) sum += m_t[tid * d + j] * k_t[j];
            prediction[tid] = sum;
        }
        __syncthreads();
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

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

        // d_error
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_S[tid * d + j]) * k_t[j];
            }
            d_error[tid] = sum;
        }
        __syncthreads();

        // d_k_mem
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_S[i * d + tid]) * error_buf[i];
            }
            d_k_mem[t * d + tid] = sum;
        }
        __syncthreads();
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + tid] * d_error[i];
            }
            d_k_mem[t * d + tid] += sum;
        }

        // d_v_mem
        if (tid < d) {
            d_v_mem[t * d + tid] = -d_error[tid];
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
    int t_start, int t_end, int d)
{
    int dd = d * d;
    // Cap at d (not dd): backward kernels require ~2× more registers than
    // forward due to prediction/error reconstruction. At d=512, block_size=1024
    // leaves only 64 regs/thread — too few. Using d=512 gives 128 regs/thread.
    int block_size = (d < 1024) ? d : 1024;
    // Round DOWN to largest power-of-2 ≤ block_size (floor, not ceil).
    // "while (rounded < block_size)" rounds UP: for non-power-of-2 d like
    // d=768 it overshoots to 1024 > d, wasting shared memory slots.
    int rounded = 1;
    while ((rounded << 1) <= block_size) rounded <<= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = (3 * d + block_size) * sizeof(float);

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
        d_M_work, d_S_work, t_start, t_end, d);
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
    int seq_len, int d)
{
    int dd = d * d;
    // Cap at d (not dd): backward kernels require ~2× more registers than
    // forward due to prediction/error reconstruction. At d=512, block_size=1024
    // leaves only 64 regs/thread — too few. Using d=512 gives 128 regs/thread.
    int block_size = (d < 1024) ? d : 1024;
    // Round DOWN to largest power-of-2 ≤ block_size (floor, not ceil).
    // "while (rounded < block_size)" rounds UP: for non-power-of-2 d like
    // d=768 it overshoots to 1024 > d, wasting shared memory slots.
    int rounded = 1;
    while ((rounded << 1) <= block_size) rounded <<= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = (3 * d + block_size) * sizeof(float);

    // Allocate d_M and d_S workspaces
    float* d_M_work = nullptr;
    float* d_S_work = nullptr;
    check_cuda_alloc("titans_backward: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));
    check_cuda_alloc("titans_backward: cudaMalloc d_S_work",
                     cudaMalloc(&d_S_work, dd * sizeof(float)));

    titans_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_states, s_states, d_y,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta, d_eta,
        d_m_initial, d_s_initial,
        d_M_work, d_S_work, seq_len, d);
    check_cuda_launch("titans_backward_kernel", d, smem_bytes);

    check_cuda_alloc("titans_backward: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(d_M_work);
    cudaFree(d_S_work);
}
