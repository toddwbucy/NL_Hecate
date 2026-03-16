#define WARP_SZ 32
// Elementwise CUDA Kernels — GPU-Resident Model
//
// Operations that don't need their own kernel file but must run on device
// without round-tripping through CPU:
//   - sigmoid forward: gate[i] = 1/(1+exp(-x[i]))
//   - softplus: theta[i] = log(1+exp(x[i]))
//   - gating forward: out[i] = a[i] * b[i]
//   - gating backward: d_a[i] = d_out[i] * b[i], d_b[i] = d_out[i] * a[i]
//   - sigmoid backward: d_x[i] = d_gate[i] * gate[i] * (1-gate[i])
//   - f32 → bf16 conversion (for SWA input)
//   - bf16 → f32 conversion (for SWA output)
//   - gate computation: dot(concat(k,v), w) + bias → sigmoid/softplus per token
//
// Standard grid=(ceil(n/256)), block=(256).
// All fp32 unless noted.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <climits>
#include <cstdio>

// ── Sigmoid ───────────────────────────────────────────────────────────

__global__ void sigmoid_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

extern "C" void sigmoid_cuda(const float* x, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sigmoid_kernel<<<grid, block>>>(x, out, n);
}

// ── Softplus: log(1 + exp(x)) ────────────────────────────────────────

__global__ void softplus_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i];
        // Numerically stable: for large x, softplus(x) ≈ x
        out[i] = (val > 20.0f) ? val : logf(1.0f + expf(val));
    }
}

extern "C" void softplus_cuda(const float* x, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    softplus_kernel<<<grid, block>>>(x, out, n);
}

// ── Element-wise multiply (gating forward) ────────────────────────────

__global__ void elemwise_mul_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * b[i];
    }
}

extern "C" void elemwise_mul_cuda(const float* a, const float* b, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    elemwise_mul_kernel<<<grid, block>>>(a, b, out, n);
}

// ── Gating backward: d_a = d_out * b, d_b = d_out * a ────────────────

__global__ void gating_backward_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ d_a,
    float* __restrict__ d_b,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_a[i] = d_out[i] * b[i];
        d_b[i] = d_out[i] * a[i];
    }
}

extern "C" void gating_backward_cuda(
    const float* d_out, const float* a, const float* b,
    float* d_a, float* d_b, int n)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    gating_backward_kernel<<<grid, block>>>(d_out, a, b, d_a, d_b, n);
}

// ── Sigmoid backward: d_x = d_gate * gate * (1 - gate) ───────────────

__global__ void sigmoid_backward_kernel(
    const float* __restrict__ d_gate,
    const float* __restrict__ gate,
    float* __restrict__ d_x,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        d_x[i] = d_gate[i] * g * (1.0f - g);
    }
}

extern "C" void sigmoid_backward_cuda(
    const float* d_gate, const float* gate, float* d_x, int n)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    sigmoid_backward_kernel<<<grid, block>>>(d_gate, gate, d_x, n);
}

// ── f32 → bf16 conversion ─────────────────────────────────────────────

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ src, unsigned short* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 val = __float2bfloat16(src[i]);
        dst[i] = *reinterpret_cast<unsigned short*>(&val);
    }
}

extern "C" void f32_to_bf16_cuda(const float* src, unsigned short* dst, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    f32_to_bf16_kernel<<<grid, block>>>(src, dst, n);
}

// ── bf16 → f32 conversion ─────────────────────────────────────────────

__global__ void bf16_to_f32_kernel(
    const unsigned short* __restrict__ src, float* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 val = *reinterpret_cast<const __nv_bfloat16*>(&src[i]);
        dst[i] = __bfloat162float(val);
    }
}

extern "C" void bf16_to_f32_cuda(const unsigned short* src, float* dst, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    bf16_to_f32_kernel<<<grid, block>>>(src, dst, n);
}

// ── Per-token gate computation ────────────────────────────────────────
// For each token t: gate_out[t] = activation(dot(concat(k_mem_t, v_mem_t), w_gate) + bias)
// k_mem_t and v_mem_t are each [d], w_gate is [2*d], bias_ptr points to a scalar on device.
// Using a device pointer for bias makes this kernel CUDA-graph-capture-safe: the graph
// captures the pointer (which is stable across optimizer updates), not the value.
// activation: 0=sigmoid, 1=softplus
// Grid=(seq_len), Block=(min(d, 512)) with warp reduction.

__global__ void gate_compute_kernel(
    const float* __restrict__ k_mem,     // [seq_len, d]
    const float* __restrict__ v_mem,     // [seq_len, d]
    const float* __restrict__ w_gate,    // [2*d]
    const float* __restrict__ bias_ptr,  // [1] — device pointer, read once per block
    float* __restrict__ gate_out,        // [seq_len]
    int seq_len, int d, int activation)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    // Partial dot product
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        sum += k_mem[t * d + i] * w_gate[i];
        sum += v_mem[t * d + i] * w_gate[d + i];
    }

    // Warp reduction
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / WARP_SZ;
    int lane = threadIdx.x % WARP_SZ;

    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
        for (int w = 0; w < num_warps; w++) {
            total += warp_sums[w];
        }
        total += bias_ptr[0];  // read bias from device memory (stable pointer)

        if (activation == 0) {
            // sigmoid
            gate_out[t] = 1.0f / (1.0f + expf(-total));
        } else {
            // softplus
            gate_out[t] = (total > 20.0f) ? total : logf(1.0f + expf(total));
        }
    }
}

extern "C" void gate_compute_cuda(
    const float* k_mem, const float* v_mem, const float* w_gate,
    const float* bias_ptr, float* gate_out,
    int seq_len, int d, int activation)
{
    int block = ((d + 31) / 32) * 32;  // round up to warp boundary
    if (block < 32) block = 32;
    if (block > 1024) block = 1024;
    gate_compute_kernel<<<seq_len, block>>>(
        k_mem, v_mem, w_gate, bias_ptr, gate_out, seq_len, d, activation);
}

// ── SAXPY: y[i] += alpha * x[i] ──────────────────────────────────────
// Simpler than cuBLAS for small buffers (gates, biases).

__global__ void saxpy_kernel(float alpha, const float* __restrict__ x,
                             float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] += alpha * x[i];
    }
}

extern "C" void saxpy_cuda(float alpha, const float* x, float* y, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    saxpy_kernel<<<grid, block>>>(alpha, x, y, n);
}

// ── CS-39 theta clamp (forward): clamp each element in-place to [lo, hi] ─
// Applied after gate_compute_cuda softplus output for theta.

__global__ void clamp_f32_kernel(float* __restrict__ inout, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = inout[i];
        inout[i] = fmaxf(lo, fminf(hi, v));
    }
}

extern "C" void clamp_f32_cuda(float* inout, int n, float lo, float hi) {
    int block = 256;
    int grid = (n + block - 1) / block;
    clamp_f32_kernel<<<grid, block>>>(inout, n, lo, hi);
}

// ── CS-39 theta clamp (backward): straight-through mask ──────────────────
// Zeros d_theta[t] when theta[t] is at the clamp boundary (lo or hi),
// preserving gradient only when the clamp is inactive (lo < theta < hi).
// Mirrors CPU: clamp_mask = 0 if theta <= floor || theta >= ceil, else 1.

__global__ void theta_clamp_mask_kernel(
    const float* __restrict__ theta, float* __restrict__ d_theta,
    int n, float lo, float hi)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = theta[i];
        if (t <= lo || t >= hi) {
            d_theta[i] = 0.0f;
        }
    }
}

extern "C" void theta_clamp_mask_cuda(
    const float* theta, float* d_theta, int n, float lo, float hi)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    theta_clamp_mask_kernel<<<grid, block>>>(theta, d_theta, n, lo, hi);
}

// ── DGD Delta Norm ──────────────────────────────────────────────────
//
// Computes ‖M @ k - v‖₂ where M is [d,d], k is [d], v is [d].
// Used as a diagnostic side-channel for the DGD prediction error.
// Single block, d threads (strided for d > blockDim.x).
//
// Source: HOPE (2512.24695) Eq 88 — error = M@k - v
// Spec:   specs/infrastructure/16_dgd_delta_norm_gpu.md

__global__ void dgd_delta_norm_kernel(
    const float* __restrict__ M,    // [d, d]
    const float* __restrict__ k,    // [d]
    const float* __restrict__ v,    // [d]
    float* __restrict__ norm_out,   // [1] — scalar output
    int d)
{
    int tid = threadIdx.x;

    // Shared memory: prediction[d] (reused for warp scratch)
    extern __shared__ float smem[];
    float* prediction = smem;  // [d]

    // Step 1: prediction = M @ k (matvec, strided)
    for (int row = tid; row < d; row += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[row * d + j] * k[j];
        }
        prediction[row] = sum;
    }
    __syncthreads();

    // Step 2: error = prediction - v, accumulate sum-of-squares
    float local_sq = 0.0f;
    for (int row = tid; row < d; row += blockDim.x) {
        float err = prediction[row] - v[row];
        local_sq += err * err;
    }

    // Step 3: Warp reduction
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
    }

    // Step 4: Inter-warp reduction via shared memory
    int warp_id = tid / WARP_SZ;
    int lane = tid % WARP_SZ;
    if (lane == 0) prediction[warp_id] = local_sq;  // reuse prediction as scratch
    __syncthreads();

    if (warp_id == 0) {
        int n_warps = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
        float val = (lane < n_warps) ? prediction[lane] : 0.0f;
        for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0) {
            norm_out[0] = sqrtf(val);
        }
    }
}

// ── Broadcast fill (spec 27) ────────────────────────────────────────
// Fill a [n_batch * n_slots * dd] buffer by broadcasting [n_batch * dd]
// source into every slot.  Layout: dst[b * n_slots * dd + t * dd + i] = src[b * dd + i].
// Used in proxy backward to construct full trajectory from M_final.
// 64-bit index arithmetic: d=1024 → dd=1M, n_batch*n_slots*dd can exceed INT_MAX.

__global__ void broadcast_fill_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int dd, int n_slots)
{
    long long b = blockIdx.y;          // batch element
    long long t = blockIdx.x;          // time slot
    long long src_off = b * dd;
    long long dst_off = (b * n_slots + t) * dd;
    for (long long i = threadIdx.x; i < dd; i += blockDim.x) {
        dst[dst_off + i] = src[src_off + i];
    }
}

extern "C" void broadcast_fill_f32_cuda(
    float* dst, const float* src,
    int dd, int n_slots, int n_batch)
{
    // Overflow guard: dd and source size must fit in int (kernel param type)
    long long dd64 = (long long)dd;
    if (dd64 > INT_MAX || (long long)n_batch * dd64 > INT_MAX) {
        fprintf(stderr, "broadcast_fill_f32_cuda: overflow (n_batch=%d, n_slots=%d, dd=%d)\n",
                n_batch, n_slots, dd);
        return;
    }
    int block = (dd < 256) ? dd : 256;
    dim3 grid(n_slots, n_batch);
    broadcast_fill_f32_kernel<<<grid, block>>>(dst, src, dd, n_slots);
}

extern "C" void dgd_delta_norm_cuda(
    const float* M, const float* k, const float* v,
    float* norm_out, int d)
{
    int dd = d * d;
    int block_size = (dd < 1024) ? dd : 1024;
    // Round up to warp boundary — __shfl_down_sync requires full warps
    block_size = ((block_size + WARP_SZ - 1) / WARP_SZ) * WARP_SZ;
    if (block_size > 1024) block_size = 1024;
    // Need at least d floats of smem for prediction + warp scratch
    int smem_bytes = d * sizeof(float);
    dgd_delta_norm_kernel<<<1, block_size, smem_bytes>>>(M, k, v, norm_out, d);
}
