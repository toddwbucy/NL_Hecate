// Batch error subtract + L2 clip — Spec 44
//
// Given predictions = M₀ @ K (computed by cuBLAS), subtract V and clip per row.
//   errors[t, d] = predictions[t, d] - V[t, d]
//   if ‖errors[t]‖₂ > error_clip: errors[t] *= error_clip / ‖errors[t]‖₂
//
// Grid=(batch_size * C), Block=(min(d, 256)).
// C = number of tokens in this chunk.
// All fp32.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SZ 32

__global__ void error_subtract_clip_kernel(
    float* __restrict__ predictions,  // [total_rows, d] — modified in-place to become errors
    const float* __restrict__ v,      // [total_rows, d]
    int d, float error_clip)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* pred_row = predictions + row * d;
    const float* v_row = v + row * d;

    // Subtract: pred[i] -= v[i]
    for (int i = tid; i < d; i += blockDim.x) {
        pred_row[i] -= v_row[i];
    }

    if (error_clip <= 0.0f) return;

    __syncthreads();

    // Compute ‖error‖²
    float local_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        float val = pred_row[i];
        local_sq += val * val;
    }

    // Warp reduction
    for (int off = WARP_SZ / 2; off > 0; off >>= 1)
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);

    // Inter-warp reduction via shared memory
    extern __shared__ float smem[];
    int warp_id = tid / WARP_SZ, lane = tid % WARP_SZ;
    if (lane == 0) smem[warp_id] = local_sq;
    __syncthreads();

    if (tid == 0) {
        int nw = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += smem[w];
        smem[0] = total;
    }
    __syncthreads();

    float err_norm = sqrtf(smem[0]);
    if (err_norm > error_clip) {
        float scale = error_clip / err_norm;
        for (int i = tid; i < d; i += blockDim.x) {
            pred_row[i] *= scale;
        }
    }
}

extern "C" void error_subtract_clip_f32_cuda(
    float* predictions, const float* v,
    int total_rows, int d, float error_clip)
{
    if (total_rows <= 0 || d <= 0) return;

    int block_size = (d < 256) ? d : 256;
    // Round up to power of 2
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 256) rounded = 256;
    block_size = rounded;

    dim3 grid(total_rows);
    dim3 block(block_size);
    int smem_bytes = ((block_size + WARP_SZ - 1) / WARP_SZ) * sizeof(float);

    error_subtract_clip_kernel<<<grid, block, smem_bytes>>>(
        predictions, v, d, error_clip);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] error_subtract_clip_kernel launch failed: %s\n",
                cudaGetErrorString(err));
        abort();
    }
}
