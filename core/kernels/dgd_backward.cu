// DGD Backward CUDA Kernel — S3b-M5
//
// Backward pass for the DGD inner loop.
// Reverse token loop with accumulated d_M.
//
// Grid=(1), Block=(min(d*d, 1024)).
// Single block: tokens processed sequentially in reverse.
//
// All fp32. Recomputes forward intermediates (prediction, error) from cached
// M_t states rather than storing them — saves memory.
//
// NOTE: d_M lives in global memory (allocated via cudaMalloc in C wrapper),
// NOT shared memory. At d=512, d_M[d*d] = 1MB — exceeds GPU smem limits.
// Only small buffers (prediction[d], error[d], d_error[d], reduce_buf) in smem.
//
// Analytical gradients from HOPE Appendix C (core/src/dgd.rs lines 101-188):
//   dL/dM_t     = (1-alpha) * dM_out - theta * dM_out @ (k @ k^T)
//   dL/dk_t     = -theta * (M_t^T @ dM_out @ k + E_t^T @ dM_out)
//   dL/dv_t     = theta * dM_out @ k
//   dL/dalpha_t = -trace(M_t^T @ dM_out)
//   dL/dtheta_t = -trace((E @ k^T)^T @ dM_out)
//
// Spec: specs/infrastructure/cuda/02_dgd_kernels.md
// Source: HOPE (2512.24695) Eq 88, Appendix C; core/src/dgd.rs

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_clip.cuh"

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// Suffix _dgd_bwd avoids ODR conflicts with forward TU and other kernels.
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_f32_dgd_bwd(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit_dgd_bwd() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_dgd_bwd() {
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

__global__ void dgd_backward_kernel(
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
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    int seq_len, int d, float error_clip)
{
    int tid = threadIdx.x;
    int dd = d * d;

    // ── Shared memory layout ──
    // Pre-Ampere: prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    // Ampere+ (sm_80+):   prediction[d] + error[d] + d_error[d] + reduce_buf[blockDim.x]
    //                    + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    extern __shared__ float smem[];
    float* prediction = smem;                    // [d]
    float* error_buf = smem + d;                 // [d]
    float* d_error = smem + 2 * d;               // [d]
    float* reduce_buf = smem + 3 * d;            // [blockDim.x]

#if __CUDA_ARCH__ >= 800
    // Double-buffered vector staging (after reduce_buf)
    float* buf_k  = smem + 3 * d + blockDim.x;
    float* buf_v  = buf_k + 2 * d;
    float* buf_q  = buf_v + 2 * d;
    float* buf_dy = buf_q + 2 * d;
#endif

    // Initialize d_M = 0
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async prefetch for backward loop ──
    // Prefetch the last token (seq_len-1) into buffer 0
    int cur = 0;
    int t_last = seq_len - 1;
    if (seq_len > 0) {
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_dgd_bwd(&buf_k[0 * d + i],  &k_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_v[0 * d + i],  &v_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_q[0 * d + i],  &q_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_dy[0 * d + i], &d_y[t_last * d + i]);
        }
        cp_async_commit_dgd_bwd();
    }

    // Reverse token loop
    for (int t = seq_len - 1; t >= 0; t--) {
        int next = 1 - cur;

        // Prefetch token t-1 into alternate buffer
        if (t > 0) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_dgd_bwd(&buf_k[next * d + i],  &k_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_v[next * d + i],  &v_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_q[next * d + i],  &q_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_dy[next * d + i], &d_y[(t - 1) * d + i]);
            }
            cp_async_commit_dgd_bwd();
        }

        // Wait for current buffer.
        // On the last iteration (t==0) no next prefetch was issued, so drain all groups.
        if (t > 0) {
            cp_async_wait_dgd_bwd<1>();
        } else {
            cp_async_wait_dgd_bwd<0>();
        }
        __syncthreads();

        const float* k_t   = &buf_k[cur * d];
        const float* v_t   = &buf_v[cur * d];
        const float* q_t   = &buf_q[cur * d];
        const float* d_y_t = &buf_dy[cur * d];

#else
    // ── Pre-Ampere path: direct global memory access ──
    // Reverse token loop
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
#endif
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

        // ── d_q_t[j] = sum_i M_{t+1}[i,j] * d_y_t[i] (strided: supports d > blockDim.x) ──
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + col] * d_y_t[i];
            }
            d_q_mem[t * d + col] = sum;
        }
        __syncthreads();

        // ── Recompute: prediction = M_t @ k_t (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += m_t[row * d + j] * k_t[j];
            }
            prediction[row] = sum;
        }
        __syncthreads();

        // ── error = prediction - v_t (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

        // ── d_alpha_t = -sum_{i,j} M_t[i,j] * d_M[i,j] (parallel reduction) ──
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                local_sum += m_t[idx] * d_M[idx];
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

        // ── d_error[i] = sum_j (-theta_t * d_M[i,j]) * k_t[j] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_M[row * d + j]) * k_t[j];
            }
            d_error[row] = sum;
        }
        __syncthreads();

        // ── d_k_mem_t[j] += sum_i d_grad[i,j] * error[i] (strided: supports d > blockDim.x) ──
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_M[i * d + col]) * error_buf[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();

        // ── d_k_mem_t[j] += sum_i M_t[i,j] * d_error[i] (strided: supports d > blockDim.x) ──
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + col] * d_error[i];
            }
            d_k_mem[t * d + col] += sum;
        }

        // ── d_v_mem_t[i] = -d_error[i] (strided: supports d > blockDim.x) ──
        for (int row = tid; row < d; row += blockDim.x) {
            d_v_mem[t * d + row] = -d_error[row];
        }
        __syncthreads();

        // ── d_M_prev = (1 - alpha_t) * d_M + d_error[i] * k_t[j] ──
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] = retention * d_M[idx] + d_error[i] * k_t[j];
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        cur = next;
#endif
    }

    // ── Store d_m_initial ──
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_initial[idx] = d_M[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════
// Segment backward: operates on a segment [t_start, t_end) with
// d_m_seed from the subsequent segment instead of initializing d_M=0.
// ══════════════════════════════════════════════════════════════════════

__global__ void dgd_backward_segment_kernel(
    const float* __restrict__ k_mem,
    const float* __restrict__ v_mem,
    const float* __restrict__ q_mem,
    const float* __restrict__ alpha,
    const float* __restrict__ theta,
    const float* __restrict__ m_states,   // segment-local: [(seg_len+1)*d*d]
    const float* __restrict__ d_y,
    const float* __restrict__ d_m_seed,   // [d*d]
    float* __restrict__ d_k_mem,
    float* __restrict__ d_v_mem,
    float* __restrict__ d_q_mem,
    float* __restrict__ d_alpha,
    float* __restrict__ d_theta,
    float* __restrict__ d_m_out,          // [d*d]
    float* __restrict__ d_M,              // [d*d] — gradient accumulator in global memory
    int t_start, int t_end, int d, float error_clip)
{
    int tid = threadIdx.x;
    int dd = d * d;

    extern __shared__ float smem[];
    float* prediction = smem;
    float* error_buf = smem + d;
    float* d_error = smem + 2 * d;
    float* reduce_buf = smem + 3 * d;

#if __CUDA_ARCH__ >= 800
    float* buf_k  = smem + 3 * d + blockDim.x;
    float* buf_v  = buf_k + 2 * d;
    float* buf_q  = buf_v + 2 * d;
    float* buf_dy = buf_q + 2 * d;
#endif

    // Initialize d_M from seed (not zeros)
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_M[idx] = d_m_seed[idx];
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    // ── Ampere+ path: cp.async prefetch for backward segment loop ──
    int cur = 0;
    int t_last = t_end - 1;
    if (t_end > t_start) {
        for (int i = tid; i < d; i += blockDim.x) {
            cp_async_f32_dgd_bwd(&buf_k[0 * d + i],  &k_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_v[0 * d + i],  &v_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_q[0 * d + i],  &q_mem[t_last * d + i]);
            cp_async_f32_dgd_bwd(&buf_dy[0 * d + i], &d_y[t_last * d + i]);
        }
        cp_async_commit_dgd_bwd();
    }

    // Reverse loop over segment tokens
    for (int t = t_end - 1; t >= t_start; t--) {
        int next = 1 - cur;

        // Prefetch token t-1 into alternate buffer
        if (t > t_start) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_dgd_bwd(&buf_k[next * d + i],  &k_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_v[next * d + i],  &v_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_q[next * d + i],  &q_mem[(t - 1) * d + i]);
                cp_async_f32_dgd_bwd(&buf_dy[next * d + i], &d_y[(t - 1) * d + i]);
            }
            cp_async_commit_dgd_bwd();
        }

        // On the last iteration (t==t_start) no next prefetch was issued, drain all groups.
        if (t > t_start) {
            cp_async_wait_dgd_bwd<1>();
        } else {
            cp_async_wait_dgd_bwd<0>();
        }
        __syncthreads();

        int seg_t = t - t_start;
        const float* k_t   = &buf_k[cur * d];
        const float* v_t   = &buf_v[cur * d];
        const float* q_t   = &buf_q[cur * d];
        const float* d_y_t = &buf_dy[cur * d];

#else
    // ── Pre-Ampere path: direct global memory access ──
    // Reverse loop over segment tokens
    for (int t = t_end - 1; t >= t_start; t--) {
        int seg_t = t - t_start;
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        const float* d_y_t = d_y + t * d;
#endif
        const float* m_t = m_states + seg_t * dd;
        const float* m_next = m_states + (seg_t + 1) * dd;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        // d_M += outer(d_y_t, q_t)
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] += d_y_t[i] * q_t[j];
        }
        __syncthreads();

        // d_q_t (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_next[i * d + col] * d_y_t[i];
            }
            d_q_mem[t * d + col] = sum;
        }
        __syncthreads();

        // Recompute prediction/error from M_t (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) sum += m_t[row * d + j] * k_t[j];
            prediction[row] = sum;
        }
        __syncthreads();
        for (int row = tid; row < d; row += blockDim.x) {
            error_buf[row] = prediction[row] - v_t[row];
        }
        __syncthreads();
        error_clip_inplace(error_buf, prediction, d, tid, error_clip);

        // d_alpha
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < dd; idx += blockDim.x) {
                local_sum += m_t[idx] * d_M[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) d_alpha[t] = -reduce_buf[0];
            __syncthreads();
        }

        // d_theta
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
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) d_theta[t] = -reduce_buf[0];
            __syncthreads();
        }

        // d_error (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += (-theta_t * d_M[row * d + j]) * k_t[j];
            }
            d_error[row] = sum;
        }
        __syncthreads();

        // d_k_mem (strided: supports d > blockDim.x)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += (-theta_t * d_M[i * d + col]) * error_buf[i];
            }
            d_k_mem[t * d + col] = sum;
        }
        __syncthreads();
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += m_t[i * d + col] * d_error[i];
            }
            d_k_mem[t * d + col] += sum;
        }

        // d_v_mem (strided: supports d > blockDim.x)
        for (int row = tid; row < d; row += blockDim.x) {
            d_v_mem[t * d + row] = -d_error[row];
        }
        __syncthreads();

        // Propagate d_M
        float retention = 1.0f - alpha_t;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_M[idx] = retention * d_M[idx] + d_error[i] * k_t[j];
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        cur = next;
#endif
    }

    // Store d_M output for earlier segment
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        d_m_out[idx] = d_M[idx];
    }
}

extern "C" void dgd_backward_segment_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* m_states, const float* d_y,
    const float* d_m_seed,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_m_out,
    int t_start, int t_end, int d, float error_clip)
{
    if (d <= 0) {
        fprintf(stderr, "dgd_backward_segment_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    int dd = d * d;
    // Cap at d (not dd): backward kernels require ~2× more registers than
    // forward due to prediction/error reconstruction. At d=512, block_size=1024
    // leaves only 64 regs/thread — too few. Using d=512 gives 128 regs/thread.
    int block_size = (d < 1024) ? d : 1024;
    // Ceil to smallest power-of-2 >= block_size, then cap at 1024.
    // Must round UP: see titans_backward_segment comment for rationale.
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded >>= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Legacy: prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    //   Ampere+: + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    // Host allocates the maximum so the kernel works on any architecture.
    int smem_bytes = (3 * d + block_size + 8 * d) * sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "dgd_backward_segment_f32_cuda: d=%d requires %d bytes shared memory (limit 163840).\n",
                d, smem_bytes);
        exit(1);
    }

    float* d_M_work = nullptr;
    check_cuda_alloc("dgd_backward_segment: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));

    check_cuda_alloc("dgd_backward_segment: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(dgd_backward_segment_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dgd_backward_segment_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
        d_m_seed,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_out,
        d_M_work, t_start, t_end, d, error_clip);
    check_cuda_launch("dgd_backward_segment_kernel", d, smem_bytes);

    check_cuda_alloc("dgd_backward_segment: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree d_M_work", cudaFree(d_M_work));
}

extern "C" void dgd_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_m_initial,
    int seq_len, int d, float error_clip)
{
    if (d <= 0) {
        fprintf(stderr, "dgd_backward_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    int dd = d * d;
    // Cap at d (not dd): backward kernels require ~2× more registers than
    // forward due to prediction/error reconstruction. At d=512, block_size=1024
    // leaves only 64 regs/thread — too few. Using d=512 gives 128 regs/thread.
    int block_size = (d < 1024) ? d : 1024;
    // Ceil to smallest power-of-2 >= block_size, then cap at 1024.
    // Must round UP: see titans_backward_segment comment for rationale.
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded >>= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Legacy: prediction[d] + error[d] + d_error[d] + reduce_buf[block_size]
    //   Ampere+: + k_buf[2*d] + v_buf[2*d] + q_buf[2*d] + dy_buf[2*d]
    // Host allocates the maximum so the kernel works on any architecture.
    int smem_bytes = (3 * d + block_size + 8 * d) * sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "dgd_backward_f32_cuda: d=%d requires %d bytes shared memory (limit 163840).\n",
                d, smem_bytes);
        exit(1);
    }

    float* d_M_work = nullptr;
    check_cuda_alloc("dgd_backward: cudaMalloc d_M_work",
                     cudaMalloc(&d_M_work, dd * sizeof(float)));

    check_cuda_alloc("dgd_backward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(dgd_backward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dgd_backward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
        d_M_work, seq_len, d, error_clip);
    check_cuda_launch("dgd_backward_kernel", d, smem_bytes);

    check_cuda_alloc("dgd_backward: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree d_M_work", cudaFree(d_M_work));
}
