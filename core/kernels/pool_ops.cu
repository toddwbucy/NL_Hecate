// Pool Operations CUDA Kernels — CMS Token Reduction (Spec 46)
//
// Mean-pool and repeat-upsample for CMS frequency hierarchy.
// Each CMS level processes seq_len / chunk_size tokens:
//   L0 (chunk_size=1):   512 tokens
//   L1 (chunk_size=8):    64 tokens (mean of 8 consecutive)
//   L2 (chunk_size=64):    8 tokens (mean of 64 consecutive)
//   L3 (chunk_size=512):   1 token  (mean of all 512)
//
// All fp32. Standard grid/block patterns.

#include <cuda_runtime.h>
#include <cstdio>

// ── Mean Pool 1D ──────────────────────────────────────────────────────
// Pool C consecutive d-dimensional vectors into their arithmetic mean.
// Input:  x [bs * s, d]
// Output: out [bs * s_f, d]  where s_f = s / C
//
// Grid: (bs * s_f), Block: (min(d, 1024))
// For d > 1024, each thread strides across dimensions.

__global__ void mean_pool_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int s, int d, int C)
{
    // blockIdx.x = batch * s_f + group
    int s_f = s / C;
    int batch = blockIdx.x / s_f;
    int group = blockIdx.x % s_f;

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        float sum = 0.0f;
        int base = (batch * s + group * C) * d + col;
        for (int i = 0; i < C; i++) {
            sum += x[base + i * d];
        }
        out[blockIdx.x * d + col] = sum / (float)C;
    }
}

extern "C" void mean_pool_1d_f32_cuda(
    const float* x, float* out,
    int bs, int s, int d, int C)
{
    int s_f = s / C;
    int n_groups = bs * s_f;
    if (n_groups == 0) return;
    int block = (d < 1024) ? d : 1024;
    mean_pool_1d_kernel<<<n_groups, block>>>(x, out, s, d, C);
}

// ── Repeat Upsample 1D ───────────────────────────────────────────────
// Repeat each of s_f vectors C times to produce s_f * C = s vectors.
// Input:  x [bs * s_f, d]
// Output: out [bs * s, d]
//
// Grid: (bs * s), Block: (min(d, 1024))

__global__ void repeat_upsample_1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int s, int d, int C)
{
    // blockIdx.x = batch * s + token
    int s_f = s / C;
    int batch = blockIdx.x / s;
    int token = blockIdx.x % s;
    int group = token / C;

    int src_idx = (batch * s_f + group) * d;
    int dst_idx = blockIdx.x * d;

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        out[dst_idx + col] = x[src_idx + col];
    }
}

extern "C" void repeat_upsample_1d_f32_cuda(
    const float* x, float* out,
    int bs, int s, int d, int C)
{
    int n_tokens = bs * s;
    if (n_tokens == 0) return;
    int block = (d < 1024) ? d : 1024;
    repeat_upsample_1d_kernel<<<n_tokens, block>>>(x, out, s, d, C);
}

// ── Mean Pool Backward ───────────────────────────────────────────────
// Backward of mean_pool_1d: broadcast gradient and divide by C.
// d_out: [bs * s_f, d]   (gradient from downstream)
// d_x:   [bs * s, d]     (gradient to upstream — ACCUMULATED, not overwritten)
//
// d_x[batch, g*C + i, col] += d_out[batch, g, col] / C
//
// Grid: (bs * s), Block: (min(d, 1024))

__global__ void mean_pool_1d_backward_kernel(
    const float* __restrict__ d_out,
    float* __restrict__ d_x,
    int s, int d, int C)
{
    int s_f = s / C;
    int batch = blockIdx.x / s;
    int token = blockIdx.x % s;
    int group = token / C;

    float scale = 1.0f / (float)C;
    int src_idx = (batch * s_f + group) * d;
    int dst_idx = blockIdx.x * d;

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        d_x[dst_idx + col] += d_out[src_idx + col] * scale;
    }
}

extern "C" void mean_pool_1d_backward_f32_cuda(
    const float* d_out, float* d_x,
    int bs, int s, int d, int C)
{
    int n_tokens = bs * s;
    if (n_tokens == 0) return;
    int block = (d < 1024) ? d : 1024;
    mean_pool_1d_backward_kernel<<<n_tokens, block>>>(d_out, d_x, s, d, C);
}

// ── Repeat Upsample Backward ─────────────────────────────────────────
// Backward of repeat_upsample_1d: sum gradients within each group of C.
// d_out: [bs * s, d]     (gradient from downstream)
// d_x:   [bs * s_f, d]   (gradient to upstream)
//
// d_x[batch, g, col] = sum_{i=0..C-1} d_out[batch, g*C + i, col]
//
// Grid: (bs * s_f), Block: (min(d, 1024))

__global__ void repeat_upsample_1d_backward_kernel(
    const float* __restrict__ d_out,
    float* __restrict__ d_x,
    int s, int d, int C)
{
    int s_f = s / C;
    int batch = blockIdx.x / s_f;
    int group = blockIdx.x % s_f;

    for (int col = threadIdx.x; col < d; col += blockDim.x) {
        float sum = 0.0f;
        int base = (batch * s + group * C) * d + col;
        for (int i = 0; i < C; i++) {
            sum += d_out[base + i * d];
        }
        d_x[blockIdx.x * d + col] = sum;
    }
}

extern "C" void repeat_upsample_1d_backward_f32_cuda(
    const float* d_out, float* d_x,
    int bs, int s, int d, int C)
{
    int s_f = s / C;
    int n_groups = bs * s_f;
    if (n_groups == 0) return;
    int block = (d < 1024) ? d : 1024;
    repeat_upsample_1d_backward_kernel<<<n_groups, block>>>(d_out, d_x, s, d, C);
}
