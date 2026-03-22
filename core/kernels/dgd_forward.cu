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
#include "error_clip.cuh"

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
    const float* __restrict__ k_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ alpha,      // [batch_size, seq_len]
    const float* __restrict__ theta,      // [batch_size, seq_len]
    const float* __restrict__ m_initial,  // [batch_size, d*d]
    float* __restrict__ m_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ y,                // [batch_size, seq_len, d]
    int seq_len, int d, int input_stride, int m_stride,
    float error_clip)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;
    int dd = d * d;

    // Offset all pointers to this batch element's slice
    k_mem     += b * input_stride * d;
    v_mem     += b * input_stride * d;
    q_mem     += b * input_stride * d;
    alpha     += b * input_stride;
    theta     += b * input_stride;
    m_initial += b * m_stride;
    m_states  += b * (seq_len + 1) * dd;
    y         += b * seq_len * d;

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
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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
    int seq_len, int d, int checkpoint_interval, float error_clip)
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
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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
    int seq_len, int d, int checkpoint_interval, float error_clip)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "dgd_forward_ckpt_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
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
        m_states, y, m_work, seq_len, d, checkpoint_interval, error_clip);
    check_cuda_launch("dgd_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("dgd_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree m_work", cudaFree(m_work));
}

extern "C" void dgd_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int batch_size,
    int input_stride, int m_stride, float error_clip)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "dgd_forward_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(batch_size);
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
        m_states, y, seq_len, d, input_stride, m_stride, error_clip);
    check_cuda_launch("dgd_forward_kernel", d, smem_bytes);
}

// ══════════════════════════════════════════════════════════════════════
// Fused DGD Forward — Spec 39
//
// Fuses L2 normalization, gate compute (alpha/theta), gate clamping,
// and the DGD recurrence into a single kernel.
// cuBLAS K/V/Q projections remain separate (called before this).
//
// Eliminates:
//   - 2× l2_normalize kernel launches (k, q)
//   - 2× gate_compute kernel launches (alpha, theta)
//   - 1× clamp kernel launch (optional)
//   - Intermediate global memory buffers (k_norms, q_norms, alpha, theta)
//
// Grid=(batch_size), Block=(min(d*d, 1024)).
// Shared memory: prediction[d] + error[d] + w_alpha[2d] + w_theta[2d]
//                + warp_scratch[32] + k_buf[d] + v_buf[d] + q_buf[d]
//                = 9*d + 32 floats
//
// Gate weights are loaded into shared memory once at kernel start,
// then reused for every token (constant across the sequence).
// ══════════════════════════════════════════════════════════════════════

__global__ void dgd_fused_forward_kernel(
    float* __restrict__ k_mem,              // [bs*s, d] — raw projections, normalized in-place
    const float* __restrict__ v_mem,        // [bs*s, d]
    float* __restrict__ q_mem,              // [bs*s, d] — raw projections, normalized in-place
    const float* __restrict__ w_alpha,      // [2*d] gate weights
    const float* __restrict__ b_alpha_ptr,  // [1] gate bias (device pointer)
    const float* __restrict__ w_theta,      // [2*d] gate weights
    const float* __restrict__ b_theta_ptr,  // [1] gate bias
    float alpha_floor, float alpha_ceil,
    float theta_floor, float theta_ceil,
    const float* __restrict__ m_initial,    // [bs, d*d]
    float* __restrict__ m_states,           // [bs, (s+1)*d*d]
    float* __restrict__ y,                  // [bs, s, d]
    float* __restrict__ alpha_out,          // [bs*s] — gate values for backward
    float* __restrict__ theta_out,          // [bs*s] — gate values for backward
    float* __restrict__ k_norms_out,        // [bs*s] — L2 norms for backward
    float* __restrict__ q_norms_out,        // [bs*s] — L2 norms for backward
    int seq_len, int d, float error_clip)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;
    int dd = d * d;
    int n_warps = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
    int warp_id = tid / WARP_SZ;
    int lane = tid % WARP_SZ;

    // Offset pointers to this batch element's data
    k_mem       += b * seq_len * d;
    v_mem       += b * seq_len * d;
    q_mem       += b * seq_len * d;
    m_initial   += b * dd;
    m_states    += b * (seq_len + 1) * dd;
    y           += b * seq_len * d;
    alpha_out   += b * seq_len;
    theta_out   += b * seq_len;
    k_norms_out += b * seq_len;
    q_norms_out += b * seq_len;

    // ── Shared memory layout ──
    // prediction[d] + error[d] + s_w_alpha[2*d] + s_w_theta[2*d]
    // + warp_scratch[32] + k_buf[d] + v_buf[d] + q_buf[d]
    extern __shared__ float smem[];
    float* prediction  = smem;                  // [d]
    float* error_buf   = smem + d;              // [d]
    float* s_w_alpha   = smem + 2 * d;          // [2*d]
    float* s_w_theta   = smem + 4 * d;          // [2*d]
    float* warp_scratch = smem + 6 * d;         // [32]
    float* k_buf       = smem + 6 * d + 32;     // [d]
    float* v_buf       = smem + 7 * d + 32;     // [d]
    float* q_buf       = smem + 8 * d + 32;     // [d]

    // Load gate weights into shared memory (constant across all tokens)
    for (int i = tid; i < 2 * d; i += blockDim.x) {
        s_w_alpha[i] = w_alpha[i];
        s_w_theta[i] = w_theta[i];
    }
    float b_alpha_val = *b_alpha_ptr;
    float b_theta_val = *b_theta_ptr;

    // Store M_0 from m_initial
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        m_states[idx] = m_initial[idx];
    }
    __syncthreads();

    // Sequential token loop
    for (int t = 0; t < seq_len; t++) {

        // ── Phase 1: Load raw k, v, q into shared memory ──
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = k_mem[t * d + i];
            v_buf[i] = v_mem[t * d + i];
            q_buf[i] = q_mem[t * d + i];
        }
        __syncthreads();

        // ── Phase 2a: L2-normalize k ──
        float k_sq = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            k_sq += k_buf[j] * k_buf[j];
        }
        for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
            k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, off);
        }
        if (lane == 0 && warp_id < 32) warp_scratch[warp_id] = k_sq;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane < n_warps) ? warp_scratch[lane] : 0.0f;
            for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, off);
            }
            if (lane == 0) warp_scratch[0] = val;
        }
        __syncthreads();
        float k_norm = sqrtf(warp_scratch[0]);
        float k_inv = 1.0f / fmaxf(k_norm, 1e-8f);
        if (tid == 0) k_norms_out[t] = k_norm;
        for (int j = tid; j < d; j += blockDim.x) {
            float nk = k_buf[j] * k_inv;
            k_buf[j] = nk;
            k_mem[t * d + j] = nk;   // write normalized k back for backward
        }
        __syncthreads();

        // ── Phase 2b: L2-normalize q ──
        float q_sq = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            q_sq += q_buf[j] * q_buf[j];
        }
        for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
            q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, off);
        }
        if (lane == 0 && warp_id < 32) warp_scratch[warp_id] = q_sq;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane < n_warps) ? warp_scratch[lane] : 0.0f;
            for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, off);
            }
            if (lane == 0) warp_scratch[0] = val;
        }
        __syncthreads();
        float q_norm = sqrtf(warp_scratch[0]);
        float q_inv = 1.0f / fmaxf(q_norm, 1e-8f);
        if (tid == 0) q_norms_out[t] = q_norm;
        for (int j = tid; j < d; j += blockDim.x) {
            float nq = q_buf[j] * q_inv;
            q_buf[j] = nq;
            q_mem[t * d + j] = nq;   // write normalized q back for backward
        }
        __syncthreads();

        // ── Phase 3a: Alpha gate = sigmoid(dot(concat(k,v), w_alpha) + b_alpha) ──
        float dot_a = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            dot_a += k_buf[j] * s_w_alpha[j];
        }
        for (int j = tid; j < d; j += blockDim.x) {
            dot_a += v_buf[j] * s_w_alpha[d + j];
        }
        for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
            dot_a += __shfl_down_sync(0xFFFFFFFF, dot_a, off);
        }
        if (lane == 0 && warp_id < 32) warp_scratch[warp_id] = dot_a;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane < n_warps) ? warp_scratch[lane] : 0.0f;
            for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, off);
            }
            val += b_alpha_val;
            val = 1.0f / (1.0f + expf(-val));  // sigmoid
            val = fmaxf(val, alpha_floor);
            val = fminf(val, alpha_ceil);
            if (lane == 0) warp_scratch[0] = val;
        }
        __syncthreads();
        float alpha_t = warp_scratch[0];
        if (tid == 0) alpha_out[t] = alpha_t;

        // ── Phase 3b: Theta gate = softplus(dot(concat(k,v), w_theta) + b_theta) ──
        float dot_t = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            dot_t += k_buf[j] * s_w_theta[j];
        }
        for (int j = tid; j < d; j += blockDim.x) {
            dot_t += v_buf[j] * s_w_theta[d + j];
        }
        for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
            dot_t += __shfl_down_sync(0xFFFFFFFF, dot_t, off);
        }
        if (lane == 0 && warp_id < 32) warp_scratch[warp_id] = dot_t;
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane < n_warps) ? warp_scratch[lane] : 0.0f;
            for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xFFFFFFFF, val, off);
            }
            val += b_theta_val;
            val = (val > 20.0f) ? val : logf(1.0f + expf(val));  // softplus
            val = fmaxf(val, theta_floor);
            val = fminf(val, theta_ceil);
            if (lane == 0) warp_scratch[0] = val;
        }
        __syncthreads();
        float theta_t = warp_scratch[0];
        if (tid == 0) theta_out[t] = theta_t;

        // ── Phase 4: DGD recurrence (unchanged math) ──
        int m_t_off = t * dd;
        int m_next_off = (t + 1) * dd;

        // prediction[i] = sum_j M_t[i,j] * k_t[j]
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_t_off + row * d + j] * k_buf[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        // error[i] = prediction[i] - v_t[i]
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_buf[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

        // M_{t+1} = (1-alpha_t) * M_t - theta_t * outer(error, k)
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            m_states[m_next_off + idx] = retention * m_states[m_t_off + idx]
                                         - theta_t * error_buf[i] * k_buf[j];
        }
        __syncthreads();

        // y_t[i] = sum_j M_{t+1}[i,j] * q_t[j]
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_states[m_next_off + row * d + j] * q_buf[j];
            }
            y[t * d + row] = sum;
        }
        __syncthreads();
    }
}

extern "C" void dgd_fused_forward_f32_cuda(
    float* k_mem, const float* v_mem, float* q_mem,
    const float* w_alpha, const float* b_alpha_ptr,
    const float* w_theta, const float* b_theta_ptr,
    float alpha_floor, float alpha_ceil,
    float theta_floor, float theta_ceil,
    const float* m_initial, float* m_states, float* y,
    float* alpha_out, float* theta_out,
    float* k_norms_out, float* q_norms_out,
    int seq_len, int d, int batch_size, float error_clip)
{
    // Shared memory: 9*d + 32 floats
    int smem_floats = 9 * d + 32;
    int smem_bytes = smem_floats * (int)sizeof(float);
    if (d <= 0 || smem_bytes > 163840) {
        fprintf(stderr, "dgd_fused_forward_f32_cuda: d=%d out of range.\n", d);
        exit(1);
    }
    int dd = d * d;
    // Round up to warp boundary so all warps are full (no partial-warp UB in __shfl_down_sync)
    int block_size = (dd < 1024) ? ((dd + 31) & ~31) : 1024;

    dim3 grid(batch_size);
    dim3 block(block_size);

    check_cuda_alloc("dgd_fused_forward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(dgd_fused_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dgd_fused_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem,
        w_alpha, b_alpha_ptr, w_theta, b_theta_ptr,
        alpha_floor, alpha_ceil, theta_floor, theta_ceil,
        m_initial, m_states, y,
        alpha_out, theta_out, k_norms_out, q_norms_out,
        seq_len, d, error_clip);
    check_cuda_launch("dgd_fused_forward_kernel", d, smem_bytes);
}
