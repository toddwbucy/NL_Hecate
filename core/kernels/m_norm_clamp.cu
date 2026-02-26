/// m_norm_clamp.cu — Standalone Frobenius norm clamp for M state.
///
/// Called ONCE per level per training step (not per timestep) after
/// copy_final_m writes the final M into context_m. This prevents
/// gradual norm divergence without polluting the forward/backward
/// kernel hot paths with per-timestep reduction overhead.
///
/// Straight-through in backward: the clamp is a stability tool, not
/// a differentiable objective. Gradient flows through as-if identity.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/// Frobenius norm clamp: if ||m||_F > m_norm_max, rescale m in-place.
/// block: (block_size,), grid: (1,)
/// smem: block_size floats for partial-sum reduction
__global__ void m_norm_clamp_kernel(float* m, int dd, float m_norm_max) {
    extern __shared__ float s_norm[];
    int tid = threadIdx.x;

    // Each thread accumulates sum-of-squares for its stripe
    float local = 0.0f;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        float v = m[idx];
        local += v * v;
    }
    s_norm[tid] = local;
    __syncthreads();

    // Parallel reduction (block_size must be power-of-2)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm[tid] += s_norm[tid + s];
        __syncthreads();
    }

    float fnorm = sqrtf(s_norm[0]);
    if (fnorm > m_norm_max) {
        float scale = m_norm_max / fnorm;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m[idx] *= scale;
        }
    }
}

/// Clamp the Frobenius norm of a d×d matrix stored in device memory.
/// No-op if m_norm_max <= 0 or >= 1e30f (disabled).
/// d must be a power of 2 and <= 1024.
extern "C" void m_norm_clamp_f32_cuda(float* m, int d, float m_norm_max) {
    if (m_norm_max <= 0.0f || m_norm_max >= 1e30f) return;

    int dd = d * d;

    // block_size = largest power-of-2 <= d, capped at 1024
    int block_size = 1;
    while (block_size * 2 <= d && block_size < 1024) block_size <<= 1;

    int smem_bytes = block_size * sizeof(float);
    m_norm_clamp_kernel<<<1, block_size, smem_bytes>>>(m, dd, m_norm_max);
    // No cudaDeviceSynchronize — rely on stream-ordered semantics.
    // gpu_forward.rs calls cuda_sync() after copy_final_m; the clamp is
    // enqueued on the same (default) stream, so it runs before any subsequent
    // host-visible read. A global device sync here would serialize across all
    // levels and hurt throughput unnecessarily.
}
