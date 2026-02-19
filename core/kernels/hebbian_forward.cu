// HebbianRule Forward CUDA Kernel — S2-M1 Phase 3
//
// Simplest memory rule: no error correction, no theta gate.
// M = (1-alpha)*M + outer(v, k)
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.

#include <cuda_runtime.h>

__global__ void hebbian_forward_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ m_initial,
    float* __restrict__ m_states,
    float* __restrict__ y,
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared: M[d*d]
    extern __shared__ float smem[];
    float* M = smem;

    // Load M_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        M[idx] = m_initial[idx];
    }
    __syncthreads();

    // Store M_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = M[idx];
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float retention = 1.0f - alpha_t;

        // M = (1-alpha) * M + outer(v, k)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            M[idx] = retention * M[idx] + v_t[i] * k_t[j];
        }
        __syncthreads();

        // Store M_{t+1}
        int off = (t + 1) * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_states[off + idx] = M[idx];
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
// Checkpointed variant: stores M only every C steps + final state.
// ══════════════════════════════════════════════════════════════════════

__global__ void hebbian_forward_ckpt_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ m_initial,
    float* __restrict__ m_states,
    float* __restrict__ y,
    int seq_len, int d, int checkpoint_interval)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* M = smem;

    // Load M_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        M[idx] = m_initial[idx];
    }
    __syncthreads();

    // Store checkpoint 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = M[idx];
    }
    __syncthreads();

    int ckpt_idx = 1;

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float retention = 1.0f - alpha_t;

        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            M[idx] = retention * M[idx] + v_t[i] * k_t[j];
        }
        __syncthreads();

        // Store checkpoint if at interval boundary or final step
        if (((t + 1) % checkpoint_interval == 0) || (t + 1 == seq_len)) {
            int off = ckpt_idx * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                m_states[off + idx] = M[idx];
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

extern "C" void hebbian_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int checkpoint_interval)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = dd * sizeof(float);

    hebbian_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, seq_len, d, checkpoint_interval);
}

extern "C" void hebbian_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared: M[d*d]
    int smem_bytes = dd * sizeof(float);

    hebbian_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, seq_len, d);
}
