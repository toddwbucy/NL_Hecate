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
//
// Ampere+ (sm_80+) optimization:
//   When __CUDA_ARCH__ >= 800, the per-token loop uses cp.async to prefetch
//   the next token's k/v/q vectors into shared memory while computing on the
//   current token. Double-buffered: two sets of k/v/q buffers, alternating.
//   This hides global memory latency for vector loads.
//   The sm_86/89 legacy path is unchanged (direct global memory access).

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_clip.cuh"
#include "m_norm_project.cuh"

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// cp.async copies 4/8/16 bytes from global to shared memory asynchronously.
// The SM continues executing while the copy engine handles the transfer.
// _delta suffix avoids ODR conflicts with titans_forward.cu helpers.
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f32_delta(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_delta() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_delta() {
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

__global__ void delta_forward_kernel(
    const float* __restrict__ k_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,      // [batch_size, seq_len, d]
    const float* __restrict__ alpha,      // [batch_size, seq_len]
    const float* __restrict__ theta,      // [batch_size, seq_len]
    const float* __restrict__ m_initial,  // [batch_size, d*d]
    float* __restrict__ m_states,         // [batch_size, (seq_len+1)*d*d]
    float* __restrict__ y,                // [batch_size, seq_len, d]
    int seq_len, int d, int input_stride, int m_stride,
    float error_clip, float m_norm_max)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;
    int dd = d * d;

    // Offset all pointers to this batch element's slice
    // input_stride separates heads in input buffers (= seq_len for normal, full_s for replay)
    // m_stride separates heads in m_initial (= dd for normal, num_ckpt*dd for replay)
    k_mem    += b * input_stride * d;
    v_mem    += b * input_stride * d;
    q_mem    += b * input_stride * d;
    alpha    += b * input_stride;
    theta    += b * input_stride;
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
            cp_async_f32_delta(&buf_k[0 * d + i], &k_mem[0 * d + i]);
            cp_async_f32_delta(&buf_v[0 * d + i], &v_mem[0 * d + i]);
            cp_async_f32_delta(&buf_q[0 * d + i], &q_mem[0 * d + i]);
        }
        cp_async_commit_delta();
    }

    for (int t = 0; t < seq_len; t++) {
        int next = 1 - cur;

        // Prefetch token t+1 into alternate buffer (overlaps with compute)
        if (t + 1 < seq_len) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_delta(&buf_k[next * d + i], &k_mem[(t + 1) * d + i]);
                cp_async_f32_delta(&buf_v[next * d + i], &v_mem[(t + 1) * d + i]);
                cp_async_f32_delta(&buf_q[next * d + i], &q_mem[(t + 1) * d + i]);
            }
            cp_async_commit_delta();
        }

        // Wait for current buffer to be ready
        if (t + 1 < seq_len) {
            cp_async_wait_delta<1>();
        } else {
            cp_async_wait_delta<0>();
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

        // ── error[i] = prediction[i] - v_t[i] (strided) ──
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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

        // Per-token M-norm projection (spec 74, matches CPU reference)
        m_norm_project_inplace(&m_states[m_next_off], prediction, dd, tid, m_norm_max);

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

        // ── error[i] = prediction[i] - v_t[i] (strided) ──
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

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

        // Per-token M-norm projection (spec 74, matches CPU reference)
        m_norm_project_inplace(&m_states[m_next_off], prediction, dd, tid, m_norm_max);

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
    int seq_len, int d, int checkpoint_interval, float error_clip,
    float m_norm_max)
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

        // Per-token M-norm projection (spec 74, matches CPU reference)
        m_norm_project_inplace(m_work, prediction, dd, tid, m_norm_max);

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
}

extern "C" void delta_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int checkpoint_interval, float error_clip,
    float m_norm_max)
{
    if (d <= 0 || 2 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "delta_forward_ckpt_f32_cuda: d=%d out of range.\n", d);
        exit(1);
    }
    if (checkpoint_interval <= 0) {
        fprintf(stderr, "delta_forward_ckpt_f32_cuda: checkpoint_interval=%d must be > 0.\n",
                checkpoint_interval);
        exit(1);
    }
    if (seq_len < 0) {
        fprintf(stderr, "delta_forward_ckpt_f32_cuda: seq_len=%d must be >= 0.\n", seq_len);
        exit(1);
    }
    long long dd64 = (long long)d * (long long)d;
    long long num_ckpts64 = (seq_len == 0)
        ? 1LL
        : ((long long)seq_len + checkpoint_interval - 1) / checkpoint_interval + 1;
    if (dd64 > INT_MAX || num_ckpts64 * dd64 > INT_MAX || (long long)seq_len * d > INT_MAX) {
        fprintf(stderr, "delta_forward_ckpt_f32_cuda: d=%d seq_len=%d ckpt_interval=%d would overflow int32 indices.\n",
                d, seq_len, checkpoint_interval);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: prediction[d] + error[d] = 2*d floats.
    // Checkpointed kernel does not use cp.async (single-block, no batch),
    // so only the two working buffers are needed.
    int smem_bytes = 2 * d * sizeof(float);

    // Allocate M workspace in global memory
    float* m_work = nullptr;
    check_cuda_alloc("delta_forward_ckpt: cudaMalloc m_work",
                     cudaMalloc(&m_work, dd * sizeof(float)));

    delta_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, m_work, seq_len, d, checkpoint_interval, error_clip,
        m_norm_max);
    check_cuda_launch("delta_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("delta_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree m_work", cudaFree(m_work));
}

extern "C" void delta_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int batch_size,
    int input_stride, int m_stride, float error_clip,
    float m_norm_max)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "delta_forward_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    if (seq_len < 0) {
        fprintf(stderr, "delta_forward_f32_cuda: seq_len=%d must be >= 0.\n", seq_len);
        exit(1);
    }
    long long dd64 = (long long)d * d;
    if (dd64 > INT_MAX || (long long)(seq_len + 1) * dd64 > INT_MAX || (long long)seq_len * d > INT_MAX) {
        fprintf(stderr, "delta_forward_f32_cuda: d=%d seq_len=%d would overflow int32 indices.\n", d, seq_len);
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
    // On sm_86/89 the extra shared memory is allocated but unused — no cost.
    int smem_bytes = 8 * d * sizeof(float);

    check_cuda_alloc("delta_forward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(delta_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    delta_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_initial,
        m_states, y, seq_len, d, input_stride, m_stride, error_clip,
        m_norm_max);
    check_cuda_launch("delta_forward_kernel", d, smem_bytes);
}
