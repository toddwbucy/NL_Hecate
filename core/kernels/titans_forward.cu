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
//
// NOTE: M and S live in global memory (m_states/s_states), NOT shared memory.
// At d=512, M+S = 2MB — far exceeds GPU shared memory limits (48-100KB).
// Only small working buffers (prediction[d], error[d]) use shared memory.

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

__global__ void titans_forward_kernel(
    const float* __restrict__ k_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ alpha,      // [batch_size, seq_len]
    const float* __restrict__ theta,      // [batch_size, seq_len]
    const float* __restrict__ eta,        // [batch_size, seq_len]
    const float* __restrict__ m_initial,  // [batch_size, d*d]
    const float* __restrict__ s_initial,  // [batch_size, d*d]
    float* __restrict__ m_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ s_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ y,                // [batch_size, seq_len, d]
    int seq_len, int d)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;
    int dd = d * d;

    // Offset all pointers to this batch element's slice
    k_mem     += b * seq_len * d;
    v_mem     += b * seq_len * d;
    q_mem     += b * seq_len * d;
    alpha     += b * seq_len;
    theta     += b * seq_len;
    eta       += b * seq_len;
    m_initial += b * dd;
    s_initial += b * dd;
    m_states  += b * (seq_len + 1) * dd;
    s_states  += b * (seq_len + 1) * dd;
    y         += b * seq_len * d;

    // Shared memory: only small working buffers
    extern __shared__ float smem[];
    float* prediction = smem;           // [d]
    float* error_buf = smem + d;        // [d]

    // Store M_0 and S_0 from initials
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_initial[idx];
        s_states[idx] = s_initial[idx];
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        float eta_t = eta[t];
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // prediction = M_t @ k
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_t_off + tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        // error = prediction - v
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        // S_{t+1} = eta * S_t - theta * outer(error, k)
        // M_{t+1} = (1-alpha) * M_t + S_{t+1}
        // Read from offset t, write to offset t+1 — non-overlapping
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            float s_new = eta_t * s_states[m_t_off + idx]
                         - theta_t * error_buf[i] * k_t[j];
            s_states[m_next_off + idx] = s_new;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx] + s_new;
        }
        __syncthreads();

        // y = M_{t+1} @ q
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_next_off + tid * d + j] * q_t[j];
            }
            y[t * d + tid] = sum;
        }
        __syncthreads();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Checkpointed variant: stores M and S only every C steps + final state.
// M and S workspaces allocated via cudaMalloc in C wrapper.
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
    float* __restrict__ m_work,           // [d*d] — working M
    float* __restrict__ s_work,           // [d*d] — working S
    int seq_len, int d, int checkpoint_interval)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;

    // Load M_0 and S_0 into workspaces
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_work[idx] = m_initial[idx];
        s_work[idx] = s_initial[idx];
    }
    __syncthreads();

    // Store checkpoint 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_work[idx];
        s_states[idx] = s_work[idx];
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
                sum += m_work[tid * d + j] * k_t[j];
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
            float s_new = eta_t * s_work[idx] - theta_t * error_buf[i] * k_t[j];
            s_work[idx] = s_new;
            m_work[idx] = retention * m_work[idx] + s_new;
        }
        __syncthreads();

        // Store checkpoint if at interval boundary or final step
        if (((t + 1) % checkpoint_interval == 0) || (t + 1 == seq_len)) {
            int off = ckpt_idx * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                m_states[off + idx] = m_work[idx];
                s_states[off + idx] = s_work[idx];
            }
            ckpt_idx++;
        }

        // y = M @ q (always)
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_work[tid * d + j] * q_t[j];
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

    int smem_bytes = 2 * d * sizeof(float);

    // Allocate M and S workspaces
    float* m_work = nullptr;
    float* s_work = nullptr;
    check_cuda_alloc("titans_forward_ckpt: cudaMalloc m_work",
                     cudaMalloc(&m_work, dd * sizeof(float)));
    check_cuda_alloc("titans_forward_ckpt: cudaMalloc s_work",
                     cudaMalloc(&s_work, dd * sizeof(float)));

    titans_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_initial, s_initial, m_states, s_states, y,
        m_work, s_work, seq_len, d, checkpoint_interval);
    check_cuda_launch("titans_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("titans_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    cudaFree(m_work);
    cudaFree(s_work);
}

extern "C" void titans_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_initial, const float* s_initial,
    float* m_states, float* s_states, float* y,
    int seq_len, int d, int batch_size)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared: prediction[d] + error[d] only
    int smem_bytes = 2 * d * sizeof(float);

    titans_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, eta,
        m_initial, s_initial, m_states, s_states, y,
        seq_len, d);
    check_cuda_launch("titans_forward_kernel", d, smem_bytes);
}
