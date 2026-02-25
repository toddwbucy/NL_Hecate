// HebbianRule Forward CUDA Kernel — S2-M1 Phase 3
//
// Simplest memory rule: no error correction, no theta gate.
// M = (1-alpha)*M + outer(v, k)
//
// Grid=(1), Block=(min(d*d, 1024)).
// All fp32.
//
// NOTE: M lives in global memory (m_states), NOT shared memory.
// At d=512, M[d*d] = 1MB — far exceeds GPU shared memory limits (48-100KB).

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

    // No shared memory needed — M lives in m_states, no prediction/error buffers

    // Store M_0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_initial[idx];
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float retention = 1.0f - alpha_t;
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // M_{t+1} = (1-alpha) * M_t + outer(v, k)
        // Read from m_states[t*dd], write to m_states[(t+1)*dd] — non-overlapping
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx]
                                         + v_t[i] * k_t[j];
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
// Checkpointed variant: stores M only every C steps + final state.
// M workspace allocated via cudaMalloc in C wrapper.
// ══════════════════════════════════════════════════════════════════════

__global__ void hebbian_forward_ckpt_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ m_initial,
    float* __restrict__ m_states,
    float* __restrict__ y,
    float* __restrict__ m_work,           // [d*d] — working M in global memory
    int seq_len, int d, int checkpoint_interval)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // No shared memory needed for Hebbian

    // Load M_0 into workspace
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_work[idx] = m_initial[idx];
    }
    __syncthreads();

    // Store checkpoint 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_work[idx];
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
            m_work[idx] = retention * m_work[idx] + v_t[i] * k_t[j];
        }
        __syncthreads();

        // Store checkpoint if at interval boundary or final step
        if (((t + 1) % checkpoint_interval == 0) || (t + 1 == seq_len)) {
            int off = ckpt_idx * dd;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                m_states[off + idx] = m_work[idx];
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

    // No shared memory needed for Hebbian checkpointed
    int smem_bytes = 0;

    // Allocate M workspace in global memory
    float* m_work = nullptr;
    cudaMalloc(&m_work, dd * sizeof(float));

    hebbian_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, m_work, seq_len, d, checkpoint_interval);
    check_cuda_launch("hebbian_forward_ckpt_kernel", d, smem_bytes);

    cudaDeviceSynchronize();
    cudaFree(m_work);
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

    // No shared memory needed for Hebbian full-trajectory
    int smem_bytes = 0;

    hebbian_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, seq_len, d);
    check_cuda_launch("hebbian_forward_kernel", d, smem_bytes);
}
