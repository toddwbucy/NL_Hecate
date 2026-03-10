// Per-token error clipping — shared device function (spec 17)
//
// Clips ‖error_buf‖₂ to error_clip when error_clip > 0.
// Reuses prediction[] as warp reduction scratch (dead after error computation).
// Straight-through estimator: identity Jacobian through the clamp in backward.
//
// Must be called after error_buf is populated and __syncthreads()'d,
// and before the M-update outer product.
//
// Source: HOPE (2512.24695) Eq 88 — bounds M-update magnitude
// Spec:   specs/infrastructure/17_error_clip.md

#pragma once

#define WARP_SZ 32  // NVIDIA warp size (avoids __device_builtin_variable linker issue)

// Clip error_buf in-place if ‖error_buf‖₂ > error_clip.
// prediction[] is used as scratch for inter-warp reduction.
// Caller must __syncthreads() before and after this block.
__device__ __forceinline__ void error_clip_inplace(
    float* error_buf, float* prediction, int d, int tid, float error_clip)
{
    if (error_clip <= 0.0f) return;

    // Step 1: partial ‖error‖²
    float local_sq = 0.0f;
    for (int row = tid; row < d; row += blockDim.x)
        local_sq += error_buf[row] * error_buf[row];

    // Warp reduction
    for (int off = WARP_SZ / 2; off > 0; off >>= 1)
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, off);

    // Inter-warp via prediction[] scratch (dead until next token)
    int warp_id = tid / WARP_SZ, lane = tid % WARP_SZ;
    if (lane == 0) prediction[warp_id] = local_sq;
    __syncthreads();

    if (tid == 0) {
        int nw = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += prediction[w];
        prediction[0] = total;  // ‖error‖² in prediction[0]
    }
    __syncthreads();

    float err_norm = sqrtf(prediction[0]);
    if (err_norm > error_clip) {
        float scale = error_clip / err_norm;
        for (int row = tid; row < d; row += blockDim.x)
            error_buf[row] *= scale;
    }
    __syncthreads();
}
