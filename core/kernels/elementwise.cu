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
// k_mem_t and v_mem_t are each [d], w_gate is [2*d], bias is scalar.
// activation: 0=sigmoid, 1=softplus
// Grid=(seq_len), Block=(min(d, 512)) with warp reduction.

__global__ void gate_compute_kernel(
    const float* __restrict__ k_mem,     // [seq_len, d]
    const float* __restrict__ v_mem,     // [seq_len, d]
    const float* __restrict__ w_gate,    // [2*d]
    float bias,
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
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[32];
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;

    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; w++) {
            total += warp_sums[w];
        }
        total += bias;

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
    float bias, float* gate_out,
    int seq_len, int d, int activation)
{
    int block = (d < 512) ? d : 512;
    if (block < 32) block = 32;  // minimum warp
    gate_compute_kernel<<<seq_len, block>>>(
        k_mem, v_mem, w_gate, bias, gate_out, seq_len, d, activation);
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
