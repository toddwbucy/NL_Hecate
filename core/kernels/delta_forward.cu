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

#include <cuda_runtime.h>
#include <float.h>

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

    // Shared memory layout:
    //   M_current[d*d]   — current memory matrix
    //   prediction[d]    — M @ k result
    //   error[d]         — prediction - v
    extern __shared__ float smem[];
    float* M = smem;                    // [d*d]
    float* prediction = smem + dd;      // [d]
    float* error_buf = smem + dd + d;   // [d]

    // Load M_0 into shared memory
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        M[idx] = m_initial[idx];
    }
    __syncthreads();

    // Store M_0 to m_states
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = M[idx];
    }
    __syncthreads();

    // Sequential token loop
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        // ── prediction[i] = sum_j M[i,j] * k_t[j] ──
        // Each thread handles one row i (if tid < d)
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += M[tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        // ── error[i] = prediction[i] - v_t[i] ──
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        // ── M[i,j] = (1-alpha_t) * M[i,j] - theta_t * error[i] * k_t[j] ──
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            M[idx] = retention * M[idx] - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // ── Store M_{t+1} to m_states ──
        int m_off = (t + 1) * dd;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_states[m_off + idx] = M[idx];
        }

        // ── y_t[i] = sum_j M[i,j] * q_t[j] ──
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

extern "C" void delta_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    // Ensure block_size >= d for the matvec operations
    if (block_size < d) block_size = d;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: M[d*d] + prediction[d] + error[d]
    int smem_bytes = (dd + 2 * d) * sizeof(float);

    delta_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, seq_len, d);
}
