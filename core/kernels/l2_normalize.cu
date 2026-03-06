// L2 Row Normalization — Forward + Backward CUDA Kernels
//
// Per-row L2 normalization for key/query vectors before memory operations.
// Titans paper (2501.00663) Section "Architectural Details":
//   "normalize queries and keys using l_2-norm"
//
// Forward: k_norm[i,:] = k_raw[i,:] / max(||k_raw[i,:]||_2, eps)
// Backward: d_k_raw = (d_k_norm - k_norm * dot(d_k_norm, k_norm)) / max(norm, eps)
//
// Grid=(n_rows), Block=(min(d, 1024)).

#include <cuda_runtime.h>
#include <math.h>

// ── Forward: normalize each row to unit L2 norm ─────────────────────

__global__ void l2_normalize_rows_kernel(
    float* __restrict__ x,           // [n_rows, d] — normalized in-place
    float* __restrict__ norms,       // [n_rows] — output: pre-normalization L2 norms
    int d, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* row_ptr = x + row * d;

    // Phase 1: compute squared norm via parallel reduction
    float local_sum = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        float val = row_ptr[j];
        local_sum += val * val;
    }

    // Warp-level reduction
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Cross-warp reduction via shared memory
    extern __shared__ float smem[];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float sq_norm = smem[0];
    float norm = sqrtf(sq_norm);
    float inv_norm = 1.0f / fmaxf(norm, eps);

    // Store the norm
    if (tid == 0) norms[row] = norm;

    // Phase 2: normalize in-place
    for (int j = tid; j < d; j += blockDim.x) {
        row_ptr[j] *= inv_norm;
    }
}

extern "C" void l2_normalize_rows_f32_cuda(
    float* x, float* norms,
    int n_rows, int d, float eps)
{
    if (n_rows <= 0 || d <= 0) return;

    int block_size = (d < 1024) ? d : 1024;

    dim3 grid(n_rows);
    dim3 block(block_size);
    int n_warps = (block_size + 31) / 32;
    int smem_bytes = n_warps * sizeof(float);

    l2_normalize_rows_kernel<<<grid, block, smem_bytes>>>(x, norms, d, eps);
}

// ── Backward: Jacobian of L2 normalization ──────────────────────────
//
// Given: d_out = gradient w.r.t. normalized output
//        x_norm = normalized vector (||x_norm|| = 1)
//        norms = pre-normalization L2 norms
//
// Compute: d_in = (d_out - x_norm * dot(d_out, x_norm)) / max(norm, eps)
//
// This is the standard Jacobian: d(x/||x||)/dx = (I - x_hat * x_hat^T) / ||x||

__global__ void l2_normalize_backward_kernel(
    const float* __restrict__ d_out,     // [n_rows, d] — gradient w.r.t. normalized
    const float* __restrict__ x_norm,    // [n_rows, d] — normalized vectors
    const float* __restrict__ norms,     // [n_rows] — pre-normalization norms
    float* __restrict__ d_in,            // [n_rows, d] — gradient w.r.t. raw input
    int d, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* d_out_row = d_out + row * d;
    const float* x_norm_row = x_norm + row * d;
    float* d_in_row = d_in + row * d;
    float norm = norms[row];
    float inv_norm = 1.0f / fmaxf(norm, eps);

    // Phase 1: compute dot(d_out, x_norm) via parallel reduction
    float local_dot = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        local_dot += d_out_row[j] * x_norm_row[j];
    }

    // Warp-level reduction
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(mask, local_dot, offset);
    }

    // Cross-warp reduction
    extern __shared__ float smem[];
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) smem[warp_id] = local_dot;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    float dot_val = smem[0];

    // Phase 2: d_in depends on whether forward used the norm or the eps clamp.
    // If norm >= eps: d_in = (d_out - x_norm * dot(d_out, x_norm)) / norm  (sphere Jacobian)
    // If norm < eps: forward was x/eps (linear scaling), so d_in = d_out / eps
    if (norm >= eps) {
        for (int j = tid; j < d; j += blockDim.x) {
            d_in_row[j] = (d_out_row[j] - x_norm_row[j] * dot_val) * inv_norm;
        }
    } else {
        float inv_eps = 1.0f / eps;
        for (int j = tid; j < d; j += blockDim.x) {
            d_in_row[j] = d_out_row[j] * inv_eps;
        }
    }
}

extern "C" void l2_normalize_backward_f32_cuda(
    const float* d_out, const float* x_norm, const float* norms,
    float* d_in,
    int n_rows, int d, float eps)
{
    if (n_rows <= 0 || d <= 0) return;

    int block_size = (d < 1024) ? d : 1024;

    dim3 grid(n_rows);
    dim3 block(block_size);
    int n_warps = (block_size + 31) / 32;
    int smem_bytes = n_warps * sizeof(float);

    l2_normalize_backward_kernel<<<grid, block, smem_bytes>>>(
        d_out, x_norm, norms, d_in, d, eps);
}
