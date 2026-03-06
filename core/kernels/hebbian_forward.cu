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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Ampere+ cp.async helpers (sm_80+)
// cp.async copies 4/8/16 bytes from global to shared memory asynchronously.
// The SM continues executing while the copy engine handles the transfer.
// ══════════════════════════════════════════════════════════════════════
#if __CUDA_ARCH__ >= 800

// Copy a single 4-byte float from global to shared memory asynchronously.
// Uses inline PTX: cp.async.ca.shared.global [dst], [src], 4;
__device__ __forceinline__ void cp_async_f32_hebb(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src)
    );
}

// Commit all prior cp.async instructions into a group.
__device__ __forceinline__ void cp_async_commit_hebb() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N groups are still in flight.
// cp_async_wait_hebb<0>() waits for ALL groups to complete.
// cp_async_wait_hebb<1>() waits until at most 1 group remains (pipeline depth=1).
template <int N>
__device__ __forceinline__ void cp_async_wait_hebb() {
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

    // ── Shared memory layout ──
    // Pre-Ampere: no shared memory needed
    // Ampere+ (sm_80+):   k_buf[2*d] + v_buf[2*d] + q_buf[2*d] = 6*d floats
    extern __shared__ float smem[];

#if __CUDA_ARCH__ >= 800
    // Double-buffered vector staging for cp.async prefetch.
    float* buf_k = smem;               // [2*d]
    float* buf_v = smem + 2 * d;       // [2*d]
    float* buf_q = smem + 4 * d;       // [2*d]
#endif

    // Store M_0
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
            cp_async_f32_hebb(&buf_k[0 * d + i], &k_mem[0 * d + i]);
            cp_async_f32_hebb(&buf_v[0 * d + i], &v_mem[0 * d + i]);
            cp_async_f32_hebb(&buf_q[0 * d + i], &q_mem[0 * d + i]);
        }
        cp_async_commit_hebb();
    }

    for (int t = 0; t < seq_len; t++) {
        int next = 1 - cur;

        // Prefetch token t+1 into alternate buffer (overlaps with compute)
        if (t + 1 < seq_len) {
            for (int i = tid; i < d; i += blockDim.x) {
                cp_async_f32_hebb(&buf_k[next * d + i], &k_mem[(t + 1) * d + i]);
                cp_async_f32_hebb(&buf_v[next * d + i], &v_mem[(t + 1) * d + i]);
                cp_async_f32_hebb(&buf_q[next * d + i], &q_mem[(t + 1) * d + i]);
            }
            cp_async_commit_hebb();
        }

        // Wait for current buffer to be ready
        if (t + 1 < seq_len) {
            cp_async_wait_hebb<1>();
        } else {
            cp_async_wait_hebb<0>();
        }
        __syncthreads();

        // Pointers to current buffer's vectors
        const float* k_t = &buf_k[cur * d];
        const float* v_t = &buf_v[cur * d];
        const float* q_t = &buf_q[cur * d];
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

        // y = M_{t+1} @ q (strided: supports d > blockDim.x)
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

        // y = M_{t+1} @ q (strided: supports d > blockDim.x)
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

        // y = M @ q (always, strided: supports d > blockDim.x)
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

extern "C" void hebbian_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d, int checkpoint_interval)
{
    if (d <= 0) {
        fprintf(stderr, "hebbian_forward_ckpt_f32_cuda: d=%d must be > 0.\n", d);
        exit(1);
    }
    if (checkpoint_interval <= 0) {
        fprintf(stderr, "hebbian_forward_ckpt_f32_cuda: checkpoint_interval=%d must be > 0.\n",
                checkpoint_interval);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(1);
    dim3 block(block_size);

    // Checkpoint kernel uses no shared memory.
    int smem_bytes = 0;

    // Allocate M workspace in global memory
    float* m_work = nullptr;
    check_cuda_alloc("hebbian_forward_ckpt: cudaMalloc m_work",
                     cudaMalloc(&m_work, dd * sizeof(float)));

    hebbian_forward_ckpt_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, m_work, seq_len, d, checkpoint_interval);
    check_cuda_launch("hebbian_forward_ckpt_kernel", d, smem_bytes);

    check_cuda_alloc("hebbian_forward_ckpt: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
    check_cuda_alloc("cudaFree m_work", cudaFree(m_work));
}

extern "C" void hebbian_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* m_initial,
    float* m_states, float* y,
    int seq_len, int d)
{
    if (d <= 0 || 8 * d * (int)sizeof(float) > 163840) {
        fprintf(stderr, "hebbian_forward_f32_cuda: d=%d out of range (must be 1..=5120).\n", d);
        exit(1);
    }
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory layout:
    //   Pre-Ampere: no shared memory needed
    //   Ampere+ (sm_80+):   k_buf[2*d] + v_buf[2*d] + q_buf[2*d] = 6*d floats
    // Host allocates the maximum (6*d) so the kernel works on any architecture.
    // On sm_86/89 the extra shared memory is allocated but unused — no cost.
    int smem_bytes = 6 * d * sizeof(float);

    check_cuda_alloc("hebbian_forward: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(hebbian_forward_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    hebbian_forward_kernel<<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, m_initial,
        m_states, y, seq_len, d);
    check_cuda_launch("hebbian_forward_kernel", d, smem_bytes);
}
