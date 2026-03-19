// M3 Optimizer CUDA Kernels — multi-scale momentum with Muon
//
// Three kernels matching the 2D/1D update split from HOPE Eq 75:
//
// 1. m3_ema_update: Fused M1 + V + conditional M2 EMA update
// 2. m3_apply_1d:   Adam-style param update (biases, LN)
// 3. m3_apply_2d:   Muon-style param update after NS orthogonalization
//
// Grid=ceil(n/256), Block=256. One thread per parameter element.
//
// Source: HOPE (2512.24695) Eq 42, 44, 75; spec 02_m3.md, spec 34.
// Constraint: CS-27, CS-28 — frequency-aware optimizer for CMS k>=2.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

extern "C" {

// ── Kernel 1: Fused EMA update for M1, V, and conditional M2 ─────────
//
// Every step:  M1 = beta1 * M1 + (1-beta1) * g
//              V  = beta2 * V  + (1-beta2) * g^2
// At chunk boundaries (update_m2 == 1):
//              M2 = beta3 * M2 + (1-beta3) * g
//
// update_m2 is precomputed on host: (step % chunk_size == 0) ? 1 : 0

__global__ void m3_ema_kernel(
    float* __restrict__ m1,         // fast momentum [n]
    float* __restrict__ m2,         // slow momentum [n]
    float* __restrict__ v,          // second moment [n]
    const float* __restrict__ g,    // gradient [n]
    int n,
    float beta1, float beta2, float beta3,
    int update_m2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gi = g[i];
        m1[i] = beta1 * m1[i] + (1.0f - beta1) * gi;
        v[i]  = beta2 * v[i]  + (1.0f - beta2) * gi * gi;
        if (update_m2) {
            m2[i] = beta3 * m2[i] + (1.0f - beta3) * gi;
        }
    }
}

cudaError_t m3_ema_update_cuda(
    float* m1, float* m2, float* v, const float* g,
    int n,
    float beta1, float beta2, float beta3,
    int update_m2)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    m3_ema_kernel<<<grid, block>>>(m1, m2, v, g, n, beta1, beta2, beta3, update_m2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "m3_ema_update_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// ── Kernel 2: 1D param update (Adam-style with V division) ───────────
//
// update = (m1 + alpha * m2) / (sqrt(v / bc2) + eps)
// param -= lr * update
//
// bc2 = 1 - beta2^step (bias correction factor, precomputed on host)

__global__ void m3_apply_1d_kernel(
    float* __restrict__ w,          // param [n]
    const float* __restrict__ m1,   // fast momentum [n]
    const float* __restrict__ m2,   // slow momentum [n]
    const float* __restrict__ v,    // second moment [n]
    int n,
    float lr, float alpha, float eps, float bc2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float combined = m1[i] + alpha * m2[i];
        float v_hat = v[i] / bc2;
        w[i] -= lr * combined / (sqrtf(v_hat) + eps);
    }
}

cudaError_t m3_apply_1d_cuda(
    float* w, const float* m1, const float* m2, const float* v,
    int n,
    float lr, float alpha, float eps, float bc2)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    m3_apply_1d_kernel<<<grid, block>>>(w, m1, m2, v, n, lr, alpha, eps, bc2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "m3_apply_1d_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// ── Kernel 3: 2D param update (Muon-style, post-NS) ─────────────────
//
// NS orthogonalization is done by cuBLAS matmul in Rust before calling this.
// o1 and o2 are already NS-orthogonalized and norm-scaled.
//
// param -= lr * (o1 + alpha * o2)

__global__ void m3_apply_2d_kernel(
    float* __restrict__ w,          // param [n]
    const float* __restrict__ o1,   // NS(M1) * ||M1|| [n]
    const float* __restrict__ o2,   // NS(M2) * ||M2|| [n]
    int n,
    float lr, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] -= lr * (o1[i] + alpha * o2[i]);
    }
}

cudaError_t m3_apply_2d_cuda(
    float* w, const float* o1, const float* o2,
    int n,
    float lr, float alpha)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    m3_apply_2d_kernel<<<grid, block>>>(w, o1, o2, n, lr, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "m3_apply_2d_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// ── Frobenius norm squared: partial reduction ─────────────────────────
// Same pattern as grad_norm_sq but for a single buffer (used for ||M||).

__global__ void frob_norm_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ partial_sums,
    int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (i < n) ? x[i] * x[i] : 0.0f;
    __syncthreads();

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

cudaError_t frob_norm_sq_cuda(
    const float* x, float* partial_sums,
    int n, int* out_num_blocks)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    *out_num_blocks = grid;
    frob_norm_sq_kernel<<<grid, block, block * sizeof(float)>>>(x, partial_sums, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "frob_norm_sq_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// ── Scale buffer: x[i] *= scale ──────────────────────────────────────
// Used to normalize by 1/||M|| before NS and rescale by ||M|| after.

__global__ void scale_buf_kernel(float* __restrict__ x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= scale;
    }
}

cudaError_t scale_buf_cuda(float* x, float scale, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_buf_kernel<<<grid, block>>>(x, scale, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "scale_buf_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// ── NS polynomial combination: x[i] = a*x[i] + b*y[i] + c*z[i] ─────
// Used in Newton-Schulz iteration: X_new = a*X + b*(A@X) + c*(A@(A@X))

__global__ void m3_ns_poly_kernel(
    float* __restrict__ x,          // in/out: current X iterate
    const float* __restrict__ y,    // A @ X
    const float* __restrict__ z,    // A @ (A @ X)
    int n,
    float a, float b, float c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = a * x[i] + b * y[i] + c * z[i];
    }
}

cudaError_t m3_ns_poly_cuda(
    float* x, const float* y, const float* z,
    int n, float a, float b, float c)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    m3_ns_poly_kernel<<<grid, block>>>(x, y, z, n, a, b, c);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "m3_ns_poly_cuda: %s\n", cudaGetErrorString(err));
    }
    return err;
}

} // extern "C"
