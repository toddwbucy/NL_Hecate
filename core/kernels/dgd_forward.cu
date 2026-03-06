// DGD Forward CUDA Kernel — S3b-M5
//
// Inner loop of Delta Gradient Descent (DGD): sequential M recurrence + readout.
// DGD is the core inner-loop optimizer for HOPE (2512.24695 §4.5).
//
// Math per token (identical to Delta Rule — DGD generalizes it):
//   prediction[i] = sum_j M[i,j] * k_t[j]
//   error[i] = prediction[i] - v_t[i]
//   M[i,j] = (1-alpha_t) * M[i,j] - theta_t * error[i] * k_t[j]
//   store M to m_states[(t+1)*d*d..]
//   y_t[i] = sum_j M[i,j] * q_t[j]
//
// Note on bias-agnosticism (CS-33): This kernel fuses the L2 error
// (M@k - v) with the update. Non-L2 attentional biases (Huber, L_p)
// would require splitting error computation from the update stage.
// This is the L2-only fast path; bias-agnostic dispatch is future work.
//
// Grid=(1), Block=(min(d*d, 1024)).
// Single block: tokens processed sequentially, threads parallelize across M's d² elements.
// All fp32 (no bf16). Memory state M MUST be fp32 per NL spec.
//
// NOTE: M lives in global memory (m_states), NOT shared memory.
// At d=512, M[d*d] = 1MB — far exceeds GPU shared memory limits (48-100KB).
// Only small working buffers (prediction[d], error[d]) use shared memory.
//
// Spec: specs/infrastructure/cuda/02_dgd_kernels.md
// Source: HOPE (2512.24695) Eq 88, Eq 121; core/src/dgd.rs

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// cp.async copies 4/8/16 bytes from global to shared memory asynchronously.
// The SM continues executing while the copy engine handles the transfer.
// Suffix _dgd avoids ODR conflicts with other translation units.
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f32_dgd(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_dgd() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_dgd() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

#endif // __CUDA_ARCH__ >= 800

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

__global__ void dgd_forward_kernel(
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

    // ── Shared memory layout ──
    // Pre-Ampere: prediction[d] + error[d] = 2*d floats
    // Ampere+ (sm_80+):   prediction[d] + error[d] + k_buf[2*d] + v_buf[2*d]
    //                    + q_buf[2*d] = 8*d floats
    extern __shared__ float smem[];
    float* prediction = smem;           // [d]
    float* error_buf = smem + d;        // [d]

#if __CUDA_ARCH__ >= 800
    // Double-buffered vector staging for cp.async prefetch.
    float* buf_k = smem + 2 * d;       // [2*d]
    float* buf_v = smem + 4 * d;       // [2*d]
    float* buf_q = smem + 6 * d;       // [2*d]
#endif

    // Store M_0 to m_states from m_initial
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_initial[idx];
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async double-buffered vector prefetch ──
    // Prefetch token 0 vectors into buffer 0
    int cur = 0;
    if (seq_len > 0) {
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_dgd(&buf_k[0 * d + i], &k_mem[0 * d + i]);
            cp_async_f32_dgd(&buf_v[0 * d + i], &v_mem[0 * d + i]);
            cp_async_f32_dgd(&buf_q[0 * d + i], &q_mem[0 * d + i]);
        }
        cp_async_commit_dgd();
    }

    // Sequential token loop
    for (int t = 0; t < seq_len; t++) {
        int next = 1 - cur;

        // Prefetch token t+1 into alternate buffer (overlaps with compute)
        if (t + 1 < seq_len) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_dgd(&buf_k[next * d + i], &k_mem[(t + 1) * d + i]);
                cp_async_f32_dgd(&buf_v[next * d + i], &v_mem[(t + 1) * d + i]);
                cp_async_f32_dgd(&buf_q[next * d + i], &q_mem[(t + 1) * d + i]);
            }
            cp_async_commit_dgd();
        }

        // Wait for current buffer to be ready.
        // On the last iteration no next prefetch was issued, so drain all groups.
        if (t + 1 < seq_len) {
            cp_async_wait_dgd<1>();
        } else {
            cp_async_wait_dgd<0>();
        }
        __syncthreads();

        // Pointers to current buffer's vectors
        const float* k_t = &buf_k[cur * d];
        const float* v_t = &buf_v[cur * d];
        const float* q_t = &buf_q[cur * d];
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // ── prediction[i] = sum_j M_t[i,j] * k_t[j] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_t_off + row * d + j] * k_t[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        // ── error[i] = prediction[i] - v_t[i] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();

        // ── M_{t+1}[i,j] = (1-alpha_t) * M_t[i,j] - theta_t * error[i] * k_t[j] ──
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx]
                                         - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // ── y_t[i] = sum_j M_{t+1}[i,j] * q_t[j] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_next_off + row * d + j] * q_t[j];
            }
            y[t * d + row] = sum;
        }
        __syncthreads();

        cur = next;
    }

#else
    // ── Pre-Ampere path: direct global memory access ──
    // Unchanged from original — byte-identical behavior.
    // Sequential token loop
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // ── prediction[i] = sum_j M_t[i,j] * k_t[j] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_t_off + row * d + j] * k_t[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        // ── error[i] = prediction[i] - v_t[i] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();

        // ── M_{t+1}[i,j] = (1-alpha_t) * M_t[i,j] - theta_t * error[i] * k_t[j] ──
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx]
                                         - theta_t * error_buf[i] * k_t[j];
        }
        __syncthreads();

        // ── y_t[i] = sum_j M_{t+1}[i,j] * q_t[j] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_next_off + row * d + j] * q_t[j];
            }
            y[t * d + row] = sum;
        }
        __syncthreads();
    }
#endif // __CUDA_ARCH__ >= 800
}

// ══════════════════════════════════════════════════════════════════════
// Checkpointed variant: stores M only every C steps + final state.
// m_states sized [(num_checkpoints)*d*d], indexed by checkpoint index.
// M workspace allocated via cudaMalloc in C wrapper.
// ══════════════════════════════════════════════════════════════════════

__global__ void dgd_forward_ckpt_kernel(
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

#if __CUDA_ARCH__ >= 800
    float* buf_k = smem + 2 * d;       // [2*d]
    float* buf_v = smem + 4 * d;       // [2*d]
    float* buf_q = smem + 6 * d;       // [2*d]
#endif

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

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async double-buffered vector prefetch ──
    int cur = 0;
    if (seq_len > 0) {
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_dgd(&buf_k[0 * d + i], &k_mem[0 * d + i]);
            cp_async_f32_dgd(&buf_v[0 * d + i], &v_mem[0 * d + i]);
            cp_async_f32_dgd(&buf_q[0 * d + i], &q_mem[0 * d + i]);
        }
        cp_async_commit_dgd();
    }

    for (int t = 0; t < seq_len; t++) {
        int next = 1 - cur;

        if (t + 1 < seq_len) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_dgd(&buf_k[next * d + i], &k_mem[(t + 1) * d + i]);
                cp_async_f32_dgd(&buf_v[next * d + i], &v_mem[(t + 1) * d + i]);
                cp_async_f32_dgd(&buf_q[next * d + i], &q_mem[(t + 1) * d + i]);
            }
            cp_async_commit_dgd();
        }

        if (t + 1 < seq_len) {
            cp_async_wait_dgd<1>();
        } else {
            cp_async_wait_dgd<0>();
        }
        __syncthreads();

        const float* k_t = &buf_k[cur * d];
        const float* v_t = &buf_v[cur * d];
        const float* q_t = &buf_q[cur * d];
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        // prediction = M @ k (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_work[row * d + j] * k_t[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
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

        // y = M @ q (always written, strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_work[row * d + j] * q_t[j];
            }
            y[t * d + row] = sum;
        }
        __syncthreads();

        cur = next;
    }

#else
    // ── Pre-Ampere path: direct global memory access ──
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        // prediction = M @ k (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_work[row * d + j] * k_t[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
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

        // y = M @ q (always written, strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_work[row * d + j] * q_t[j];
            }
            y[t * d + row] = sum;
        }
        __syncthreads();
    }
#endif // __CUDA_ARCH__ >= 800
}

extern "C" void dgd_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int checkpoint_interval)
{
    if (d <= 0 || 2 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "dgd_forward_ckpt_f32_cuda: d=%d out of range.\n", d);
        exit(1);
    }
    if (checkpoint_interval <= 0) {
        fprintf(stderr, "dgd_forward_ckpt_f32_cuda: checkpoint_interval=%d must be > 0.\n",
                checkpoint_interval);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Pre-Ampere: prediction[d] + error[d] = 2*d floats
    //   Ampere+ (sm_80+):   prediction[d] + error[d] + k_buf[2*d] + v_buf[2*d]
    //                      + q_buf[2*d] = 8*d floats
    // Host allocates the maximum (8*d) so the kernel works on any architecture.
    int smem_bytes = 8 * d * sizeof(float);

    float* m_work = nullptr;
    check_cuda_alloc("dgd_forward_ckpt: cudaMalloc m_work",
                     cudaMalloc(&m_work, dd * sizeof(float)));

    check_cuda_alloc("dgd_forward_ckpt: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(dgd_forward_ckpt_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dgd_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, m_work, seq_len, d, checkpoint_interval);
    check_cuda_launch("dgd_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("dgd_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree m_work", cudaFree(m_work));
}

extern "C" void dgd_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "dgd_forward_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Pre-Ampere: prediction[d] + error[d] = 2*d floats
    //   Ampere+ (sm_80+):   prediction[d] + error[d] + k_buf[2*d] + v_buf[2*d]
    //                      + q_buf[2*d] = 8*d floats
    // Host allocates the maximum (8*d) so the kernel works on any architecture.
    int smem_bytes = 8 * d * sizeof(float);

    check_cuda_alloc("dgd_forward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(dgd_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dgd_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, seq_len, d);
    check_cuda_launch("dgd_forward_kernel", d, smem_bytes);
}
