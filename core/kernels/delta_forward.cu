// DeltaRule Forward CUDA Kernel — S2-M1 Phase 1
//
// Inner loop of Delta Rule memory: sequential M recurrence + readout.
// Projections (embedded → k/v/q) and gates (alpha, theta) computed in Rust.
// Only the sequential per-token M update is in CUDA.
//
// Grid=(1), Block=(min(d*d, 1024)).
// Single block: tokens processed sequentially, threads parallelize across M's d² elements.
//
// All fp32 (no bf16). Memory state M MUST be fp32 per NL spec.
//
// Per token t:
//   prediction[i] = sum_j M[i,j] * k_t[j]
//   error[i] = prediction[i] - v_t[i]
//   M[i,j] = (1-alpha_t) * M[i,j] - theta_t * error[i] * k_t[j]
//   store M to m_states[(t+1)*d*d..]
//   y_t[i] = sum_j M[i,j] * q_t[j]
//
// This file is compiled by nvcc into machine code (opaque to AD).
//
// NOTE: M lives in global memory (m_states), NOT shared memory.
// At d=512, M[d*d] = 1MB — far exceeds GPU shared memory limits (48-100KB).
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

__global__ void delta_forward_kernel(
    const float* __restrict__ k_mem,      // [seq_len, d]
    const float* __restrict__ v_mem,      // [seq_len, d]
    const float* __restrict__ q_mem,      // [seq_len, d]
    const float* __restrict__ alpha,      // [seq_len]
    const float* __restrict__ theta,      // [seq_len]
    const float* __restrict__ m_initial,  // [d*d]
    float* __restrict__ m_states,         // [(seq_len+1)*d*d]
    float* __restrict__ y,                // [seq_len, d]
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared memory: only small working buffers
    //   prediction[d]    — M @ k result
    //   error[d]         — prediction - v
    extern __shared__ float smem[];
    float* prediction = smem;           // [d]
    float* error_buf = smem + d;        // [d]

    // Store M_0 to m_states from m_initial
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_initial[idx];
    }
    __syncthreads();

    // Sequential token loop
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // ── prediction[i] = sum_j M_t[i,j] * k_t[j] ──
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_t_off + tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        // ── error[i] = prediction[i] - v_t[i] ──
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        // ── M_{t+1}[i,j] = (1-alpha_t) * M_t[i,j] - theta_t * error[i] * k_t[j] ──
        // Read from m_states[t*dd], write to m_states[(t+1)*dd] — non-overlapping
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx]
                                         - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // ── y_t[i] = sum_j M_{t+1}[i,j] * q_t[j] ──
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
// m_states sized [(num_checkpoints)*d*d], indexed by checkpoint index.
// M workspace allocated via cudaMalloc in C wrapper (can't reuse m_states).
// ══════════════════════════════════════════════════════════════════════

__global__ void delta_forward_ckpt_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ m_initial,
    float* __restrict__ m_states,         // [num_ckpt * d*d]
    float* __restrict__ y,
    float* __restrict__ m_work,           // [d*d] — working M in global memory
    int seq_len, int d, int checkpoint_interval)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;

    // Load M_0 into workspace
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_work[idx] = m_initial[idx];
    }
    __syncthreads();

    // Store M_0 as checkpoint 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_work[idx];
    }
    __syncthreads();

    int ckpt_idx = 1;  // next checkpoint slot

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

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
            m_work[idx] = retention * m_work[idx] - theta_t * error_buf[i] * k_t[j];
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

        // y = M @ q (always written)
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

extern "C" void delta_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int checkpoint_interval)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] only
    int smem_bytes = 2 * d * sizeof(float);

    // Allocate M workspace in global memory
    float* m_work = nullptr;
    check_cuda_alloc("delta_forward_ckpt: cudaMalloc m_work",
                     cudaMalloc(&m_work, dd * sizeof(float)));

    delta_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, m_work, seq_len, d, checkpoint_interval);
    check_cuda_launch("delta_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("delta_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree m_work", cudaFree(m_work));
}

extern "C" void delta_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] only
    int smem_bytes = 2 * d * sizeof(float);

    delta_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, seq_len, d);
    check_cuda_launch("delta_forward_kernel", d, smem_bytes);
}
