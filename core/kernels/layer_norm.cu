/*
 * LayerNorm forward + backward CUDA kernels.
 *
 * Forward: one block per position (token). Each block computes mean, variance,
 *          then normalizes and applies affine (gamma * x_hat + beta).
 *
 * Backward: one block per position. Three-term formula for d_x, atomicAdd for
 *           d_gamma and d_beta across positions.
 *
 * Grid:  (n,)        — one block per position
 * Block: (block_dim,) — threads cooperate on the d dimension
 *
 * All fp32 (inner-loop precision requirement).
 */

#include <cuda_runtime.h>
#include <math.h>

/* ── Forward kernel ─────────────────────────────────────────────── */

__global__ void layer_norm_forward_kernel(
    const float* __restrict__ x,          // [n, d]
    const float* __restrict__ gamma,      // [d]
    const float* __restrict__ beta,       // [d]
    float* __restrict__ out,              // [n, d]
    float* __restrict__ mean_cache,       // [n]
    float* __restrict__ rstd_cache,       // [n]
    int d, float eps)
{
    extern __shared__ float smem[];       // [2 * blockDim.x]
    float* s_sum  = smem;
    float* s_sum2 = smem + blockDim.x;

    int row = blockIdx.x;
    const float* x_row = x + row * d;
    float* out_row = out + row * d;

    // Phase 1: partial sums for mean and variance
    float local_sum  = 0.0f;
    float local_sum2 = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float val = x_row[i];
        local_sum  += val;
        local_sum2 += val * val;
    }
    s_sum[threadIdx.x]  = local_sum;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Block-level tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x]  += s_sum[threadIdx.x + stride];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)d;
    float var  = s_sum2[0] / (float)d - mean * mean;
    float rstd = rsqrtf(var + eps);

    // Cache for backward
    if (threadIdx.x == 0) {
        mean_cache[row] = mean;
        rstd_cache[row] = rstd;
    }

    // Phase 2: normalize and apply affine
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        out_row[i] = gamma[i] * x_hat + beta[i];
    }
}

/* ── Backward kernel ────────────────────────────────────────────── */

__global__ void layer_norm_backward_kernel(
    const float* __restrict__ d_out,      // [n, d]
    const float* __restrict__ x,          // [n, d]
    const float* __restrict__ gamma,      // [d]
    const float* __restrict__ mean_cache, // [n]
    const float* __restrict__ rstd_cache, // [n]
    float* __restrict__ d_x,              // [n, d]
    float* __restrict__ d_gamma,          // [d] (atomicAdd across rows)
    float* __restrict__ d_beta,           // [d] (atomicAdd across rows)
    int d)
{
    extern __shared__ float smem[];       // [2 * blockDim.x]
    float* s_dot1 = smem;                 // sum(d_out * gamma)
    float* s_dot2 = smem + blockDim.x;   // sum(d_out * gamma * x_hat)

    int row = blockIdx.x;
    const float* d_out_row = d_out + row * d;
    const float* x_row = x + row * d;
    float* d_x_row = d_x + row * d;

    float mean = mean_cache[row];
    float rstd = rstd_cache[row];

    // Phase 1: compute partial sums for the three-term formula
    float local_dot1 = 0.0f;
    float local_dot2 = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        float dg = d_out_row[i] * gamma[i];
        local_dot1 += dg;
        local_dot2 += dg * x_hat;
    }
    s_dot1[threadIdx.x] = local_dot1;
    s_dot2[threadIdx.x] = local_dot2;
    __syncthreads();

    // Block-level tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_dot1[threadIdx.x] += s_dot1[threadIdx.x + stride];
            s_dot2[threadIdx.x] += s_dot2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum_dg     = s_dot1[0];
    float sum_dg_xh  = s_dot2[0];
    float inv_d = 1.0f / (float)d;

    // Phase 2: compute d_x using three-term formula, accumulate d_gamma and d_beta
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float x_hat = (x_row[i] - mean) * rstd;
        float dg = d_out_row[i] * gamma[i];

        // d_x = rstd/d * (d * dg - sum_dg - x_hat * sum_dg_xh)
        d_x_row[i] = rstd * inv_d * ((float)d * dg - sum_dg - x_hat * sum_dg_xh);

        // d_gamma, d_beta: accumulated across positions
        atomicAdd(&d_gamma[i], d_out_row[i] * x_hat);
        atomicAdd(&d_beta[i],  d_out_row[i]);
    }
}

/* ── C entry points for Rust FFI ────────────────────────────────── */

extern "C" void layer_norm_forward_cuda(
    const float* x, const float* gamma, const float* beta,
    float* out, float* mean_cache, float* rstd_cache,
    int n, int d, float eps)
{
    // One block per position. Block size = min(d, 1024), must be power of 2.
    int block_size = 1;
    while (block_size < d && block_size < 1024) block_size <<= 1;
    size_t smem = 2 * block_size * sizeof(float);
    layer_norm_forward_kernel<<<n, block_size, smem>>>(
        x, gamma, beta, out, mean_cache, rstd_cache, d, eps);
}

extern "C" void layer_norm_backward_cuda(
    const float* d_out, const float* x,
    const float* gamma, const float* mean_cache, const float* rstd_cache,
    float* d_x, float* d_gamma, float* d_beta,
    int n, int d)
{
    // Same grid/block as forward. d_gamma/d_beta must be zeroed before call.
    int block_size = 1;
    while (block_size < d && block_size < 1024) block_size <<= 1;
    size_t smem = 2 * block_size * sizeof(float);
    layer_norm_backward_kernel<<<n, block_size, smem>>>(
        d_out, x, gamma, mean_cache, rstd_cache,
        d_x, d_gamma, d_beta, d);
}
