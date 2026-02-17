// AdamW Weight Update CUDA Kernel — outer-loop optimizer
//
// Fused AdamW: updates weights, first moment (m), and second moment (v)
// in a single pass. Zero PCIe traffic — all buffers stay on device.
//
// AdamW (Loshchilov & Hutter, 2019): decoupled weight decay.
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w_{t-1})
//
// Grid=ceil(n/256), Block=256. One thread per parameter.
// Bias correction factors are precomputed on host and passed as args
// to avoid per-thread powf() calls.

#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void adamw_kernel(
    float* __restrict__ w,          // weights [n] (updated in-place)
    const float* __restrict__ g,    // gradients [n]
    float* __restrict__ m,          // first moment [n] (updated in-place)
    float* __restrict__ v,          // second moment [n] (updated in-place)
    int n,
    float lr,
    float beta1, float beta2,
    float eps,
    float bc1_inv,                  // 1 / (1 - beta1^t), precomputed
    float bc2_inv,                  // 1 / (1 - beta2^t), precomputed
    float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = g[i];

        // Update biased moments
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;

        // Store updated moments
        m[i] = mi;
        v[i] = vi;

        // Bias-corrected estimates
        float m_hat = mi * bc1_inv;
        float v_hat = vi * bc2_inv;

        // AdamW update: decoupled weight decay
        w[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w[i]);
    }
}

void adamw_update_cuda(
    float* w, const float* g, float* m, float* v,
    int n,
    float lr, float beta1, float beta2,
    float eps, float bc1_inv, float bc2_inv,
    float weight_decay)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    adamw_kernel<<<grid, block>>>(w, g, m, v, n,
        lr, beta1, beta2, eps, bc1_inv, bc2_inv, weight_decay);
}

// ── Gradient norm squared: partial reduction per block ─────────────
// Computes sum of g[i]^2 per block, stores in partial_sums[blockIdx.x].
// Host sums the partials (few hundred values) to get total norm^2.

__global__ void grad_norm_sq_kernel(
    const float* __restrict__ g,
    float* __restrict__ partial_sums,
    int n)
{
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and square
    smem[tid] = (i < n) ? g[i] * g[i] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = smem[0];
    }
}

void grad_norm_sq_cuda(
    const float* g, float* partial_sums,
    int n, int* out_num_blocks)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    *out_num_blocks = grid;
    grad_norm_sq_kernel<<<grid, block, block * sizeof(float)>>>(g, partial_sums, n);
}

// ── Scale gradient buffer: g[i] *= scale ───────────────────────────
// Used for gradient clipping: scale = max_norm / actual_norm.

__global__ void grad_scale_kernel(float* __restrict__ g, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        g[i] *= scale;
    }
}

void grad_scale_cuda(float* g, float scale, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    grad_scale_kernel<<<grid, block>>>(g, scale, n);
}

} // extern "C"
