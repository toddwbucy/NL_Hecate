// TitansLMM Forward CUDA Kernel — S2-M1 Phase 2
//
// Extends Delta Rule with momentum accumulator S and gate eta.
// Sequential per-token: M and S evolve together.
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.
//
// Per token t:
//   prediction[i] = sum_j M[i,j] * k_t[j]
//   error[i] = prediction[i] - v_t[i]
//   S[i,j] = eta_t * S[i,j] - theta_t * error[i] * k_t[j]
//   M[i,j] = (1-alpha_t) * M[i,j] + S[i,j]
//   y_t[i] = sum_j M[i,j] * q_t[j]

#include <cuda_runtime.h>

__global__ void titans_forward_kernel(
    const float* __restrict__ k_mem,      // [seq_len, d]
    const float* __restrict__ v_mem,      // [seq_len, d]
    const float* __restrict__ q_mem,      // [seq_len, d]
    const float* __restrict__ alpha,      // [seq_len]
    const float* __restrict__ theta,      // [seq_len]
    const float* __restrict__ eta,        // [seq_len]
    const float* __restrict__ m_initial,  // [d*d]
    const float* __restrict__ s_initial,  // [d*d]
    float* __restrict__ m_states,         // [(seq_len+1)*d*d]
    float* __restrict__ s_states,         // [(seq_len+1)*d*d]
    float* __restrict__ y,                // [seq_len, d]
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared memory: M[d*d] + S[d*d] + prediction[d] + error[d]
    extern __shared__ float smem[];
    float* M = smem;
    float* S = smem + dd;
    float* prediction = smem + 2 * dd;
    float* error_buf = smem + 2 * dd + d;

    // Load M_0 and S_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        M[idx] = m_initial[idx];
        S[idx] = s_initial[idx];
    }
    __syncthreads();

    // Store initial states
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = M[idx];
        s_states[idx] = S[idx];
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        float eta_t = eta[t];

        // prediction = M @ k
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += M[tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        // error = prediction - v
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        // S = eta * S - theta * outer(error, k)
        // M = (1-alpha) * M + S
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            S[idx] = eta_t * S[idx] - theta_t * error_buf[i] * k_t[j];
            M[idx] = retention * M[idx] + S[idx];
        }
        __syncthreads();

        // Store M_{t+1} and S_{t+1}
        int off = (t + 1) * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_states[off + idx] = M[idx];
            s_states[off + idx] = S[idx];
        }

        // y = M @ q
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += M[tid * d + j] * q_t[j];
            }
            y[t * d + tid] = sum;
        }
        __syncthreads();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Checkpointed variant: stores M and S only every C steps + final state.
// ══════════════════════════════════════════════════════════════════════

__global__ void titans_forward_ckpt_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ eta,
    const float* __restrict__ m_initial,
    const float* __restrict__ s_initial,
    float* __restrict__ m_states,
    float* __restrict__ s_states,
    float* __restrict__ y,
    int seq_len, int d, int checkpoint_interval)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* M = smem;
    float* S = smem + dd;
    float* prediction = smem + 2 * dd;
    float* error_buf = smem + 2 * dd + d;

    // Load M_0 and S_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        M[idx] = m_initial[idx];
        S[idx] = s_initial[idx];
    }
    __syncthreads();

    // Store checkpoint 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = M[idx];
        s_states[idx] = S[idx];
    }
    __syncthreads();

    int ckpt_idx = 1;

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        float eta_t = eta[t];

        // prediction = M @ k
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += M[tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            S[idx] = eta_t * S[idx] - theta_t * error_buf[i] * k_t[j];
            M[idx] = retention * M[idx] + S[idx];
        }
        __syncthreads();

        // Store checkpoint if at interval boundary or final step
        if (((t + 1) % checkpoint_interval == 0) || (t + 1 == seq_len)) {
            int off = ckpt_idx * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                m_states[off + idx] = M[idx];
                s_states[off + idx] = S[idx];
            }
            ckpt_idx++;
        }

        // y = M @ q (always)
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += M[tid * d + j] * q_t[j];
            }
            y[t * d + tid] = sum;
        }
        __syncthreads();
    }
}

extern "C" void titans_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_initial, const float* s_initial,
    float* m_states, float* s_states, float* y,
    int seq_len, int d, int checkpoint_interval)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = (2 * dd + 2 * d) * sizeof(float);

    titans_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_initial, s_initial, m_states, s_states, y,
        seq_len, d, checkpoint_interval);
}

extern "C" void titans_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_initial, const float* s_initial,
    float* m_states, float* s_states, float* y,
    int seq_len, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared: M[d*d] + S[d*d] + prediction[d] + error[d]
    int smem_bytes = (2 * dd + 2 * d) * sizeof(float);

    titans_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_initial, s_initial, m_states, s_states, y,
        seq_len, d);
}
