// DeltaRule Backward CUDA Kernel — S2-M1 Phase 1
//
// Backward pass for the inner loop of Delta Rule memory.
// Reverse token loop with accumulated d_M.
//
// Grid=(1), Block=(min(d*d, 1024)).
// Single block: tokens processed sequentially in reverse.
//
// All fp32. Recomputes forward intermediates (prediction, error) from cached
// M_t states rather than storing them — saves memory.
//
// Reverse loop (t = s-1 down to 0):
//   d_M += outer(d_y_t, q_t)                    (from readout y = M @ q)
//   d_q_t[j] = sum_i M_{t+1}[i,j] * d_y_t[i]   (matvec transpose)
//   Recompute: prediction = M_t @ k_t, error = prediction - v_t
//   d_theta_t = -sum_{i,j} error[i]*k_t[j] * d_M[i,j]
//   d_alpha_t = -sum_{i,j} M_t[i,j] * d_M[i,j]
//   d_grad = -theta_t * d_M
//   d_error[i] = sum_j d_grad[i,j] * k_t[j]
//   d_k_t[j] += sum_i d_grad[i,j] * error[i]
//   d_k_t[j] += sum_i M_t[i,j] * d_error[i]  (from prediction = M @ k chain)
//   d_v_t[i] = -d_error[i]
//   d_M = (1-alpha_t) * d_M   (propagate to M_t)
//
// This file is compiled by nvcc into machine code (opaque to Enzyme).

#include <cuda_runtime.h>
#include <float.h>

__global__ void delta_backward_kernel(
    const float* __restrict__ k_mem,      // [seq_len, d]
    const float* __restrict__ v_mem,      // [seq_len, d]
    const float* __restrict__ q_mem,      // [seq_len, d]
    const float* __restrict__ alpha,      // [seq_len]
    const float* __restrict__ theta,      // [seq_len]
    const float* __restrict__ m_states,   // [(seq_len+1)*d*d]
    const float* __restrict__ d_y,        // [seq_len, d]
    float* __restrict__ d_k_mem,          // [seq_len, d]
    float* __restrict__ d_v_mem,          // [seq_len, d]
    float* __restrict__ d_q_mem,          // [seq_len, d]
    float* __restrict__ d_alpha,          // [seq_len]
    float* __restrict__ d_theta,          // [seq_len]
    float* __restrict__ d_m_initial,      // [d*d]
    int seq_len, int d)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // Shared memory layout:
    //   d_M[d*d]        — accumulated gradient on memory state
    //   prediction[d]   — recomputed M_t @ k_t
    //   error[d]        — recomputed prediction - v
    //   d_error[d]      — gradient on error
    //   reduce_buf[blockDim.x] — for parallel reduction
    extern __shared__ float smem[];
    float* d_M = smem;                               // [d*d]
    float* prediction = smem + dd;                    // [d]
    float* error_buf = smem + dd + d;                 // [d]
    float* d_error = smem + dd + 2 * d;               // [d]
    float* reduce_buf = smem + dd + 3 * d;            // [blockDim.x]

    // Initialize d_M = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
    }
    __syncthreads();

    // Reverse token loop
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
        const float* m_t = m_states + t * dd;
        const float* m_next = m_states + (t + 1) * dd;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        // ── d_M += outer(d_y_t, q_t) ──
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] += d_y_t[i] * q_t[j];
        }
        __syncthreads();

        // ── d_q_t[j] = sum_i M_{t+1}[i,j] * d_y_t[i] ──
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + tid] * d_y_t[i];
            }
            d_q_mem[t * d + tid] = sum;
        }
        __syncthreads();

        // ── Recompute: prediction = M_t @ k_t ──
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_t[tid * d + j] * k_t[j];
            }
            prediction[tid] = sum;
        }
        __syncthreads();

        // ── error = prediction - v_t ──
        if (tid < d) {
            error_buf[tid] = prediction[tid] - v_t[tid];
        }
        __syncthreads();

        // ── d_alpha_t = -sum_{i,j} M_t[i,j] * d_M[i,j] (parallel reduction) ──
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                local_sum += m_t[idx] * d_M[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            // Tree reduction
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    reduce_buf[tid] += reduce_buf[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                d_alpha[t] = -reduce_buf[0];
            }
            __syncthreads();
        }

        // ── d_theta_t = -sum_{i,j} grad[i,j] * d_M[i,j] where grad=outer(error,k) ──
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                local_sum += error_buf[i] * k_t[j] * d_M[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    reduce_buf[tid] += reduce_buf[tid + s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                d_theta[t] = -reduce_buf[0];
            }
            __syncthreads();
        }

        // ── d_error[i] = sum_j (-theta_t * d_M[i,j]) * k_t[j] ──
        if (tid < d) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_M[tid * d + j]) * k_t[j];
            }
            d_error[tid] = sum;
        }
        __syncthreads();

        // ── d_k_mem_t[j] += sum_i d_grad[i,j] * error[i] ──
        // d_grad[i,j] = -theta_t * d_M[i,j]
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_M[i * d + tid]) * error_buf[i];
            }
            d_k_mem[t * d + tid] = sum;
        }
        __syncthreads();

        // ── d_k_mem_t[j] += sum_i M_t[i,j] * d_error[i]  (from prediction = M @ k chain) ──
        if (tid < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + tid] * d_error[i];
            }
            d_k_mem[t * d + tid] += sum;
        }

        // ── d_v_mem_t[i] = -d_error[i] ──
        if (tid < d) {
            d_v_mem[t * d + tid] = -d_error[tid];
        }
        __syncthreads();

        // ── d_M_prev = (1 - alpha_t) * d_M ──
        // Also add contribution from prediction backward:
        //   d_M_prev[i,j] += d_error[i] * k_t[j]
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] = retention * d_M[idx] + d_error[i] * k_t[j];
        }
        __syncthreads();
    }

    // ── Store d_m_initial ──
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_initial[idx] = d_M[idx];
    }
}

extern "C" void delta_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_m_initial,
    int seq_len, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    if (block_size < d) block_size = d;
    // Round up to next power of 2 for tree reduction correctness
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: d_M[d*d] + prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    int smem_bytes = (dd + 3 * d + block_size) * sizeof(float);

    delta_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
        seq_len, d);
}
