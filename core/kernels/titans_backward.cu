// TitansLMM Backward CUDA Kernel — S2-M1 Phase 2
//
// Backward pass with two interacting recurrences: d_M and d_S.
// Recomputes prediction/error from cached M_t states.
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.

#include <cuda_runtime.h>

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
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared: d_M[d*d] + d_S[d*d] + prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    extern __shared__ float smem[];
    float* d_M = smem;
    float* d_S = smem + dd;
    float* prediction = smem + 2 * dd;
    float* error_buf = smem + 2 * dd + d;
    float* d_error = smem + 2 * dd + 2 * d;
    float* reduce_buf = smem + 2 * dd + 3 * d;

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
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared: d_M[d*d] + d_S[d*d] + prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    int smem_bytes = (2 * dd + 3 * d + block_size) * sizeof(float);

    titans_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_states, s_states, d_y,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta, d_eta,
        d_m_initial, d_s_initial,
        seq_len, d);
}
