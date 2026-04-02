// Per-token M-norm projection — shared device function (spec 74)
//
// Projects M onto the L2 ball of radius m_norm_max:
//   if ||M||_F > m_norm_max: M <- M * (m_norm_max / ||M||_F)
//
// Matches CPU reference (titans_lmm.rs:377-386).
// Straight-through backward: identity Jacobian (same as gradient clipping).
//
// Must be called after M update and __syncthreads(), before y = M @ q.
// scratch[] must have at least ceil(blockDim.x / 32) floats of shared memory.
//
// Source: CPU reference titans_lmm.rs:377-386; spec 74
// Traced: titans_equations/eq-003-memory-update

#pragma once

#ifndef WARP_SZ
#define WARP_SZ 32
#endif

// Project M onto L2 ball in-place if ||M||_F > m_norm_max.
// scratch[] is shared memory reused for inter-warp reduction.
// Caller must __syncthreads() before calling. Function syncthreads internally.
__device__ __forceinline__ void m_norm_project_inplace(
    float* m_ptr, float* scratch, int dd, int tid, float m_norm_max)
{
    if (m_norm_max >= 1e30f) return;

    // Partial sum of squares across M elements
    float local_sq = 0.0f;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        float v = m_ptr[idx];
        local_sq += v * v;
    }

    // Warp-level reduction via shuffle (handles partial final warp)
    int warp_id = tid / WARP_SZ, lane = tid % WARP_SZ;
    int lanes_in_warp = min((int)blockDim.x - warp_id * WARP_SZ, WARP_SZ);
    unsigned wmask = (lanes_in_warp >= WARP_SZ)
                         ? 0xFFFFFFFFu
                         : ((1u << lanes_in_warp) - 1u);
    for (int off = WARP_SZ / 2; off > 0; off >>= 1) {
        float other = __shfl_down_sync(wmask, local_sq, off);
        if (lane + off < lanes_in_warp)
            local_sq += other;
    }

    if (lane == 0) scratch[warp_id] = local_sq;
    __syncthreads();

    if (tid == 0) {
        int nw = (blockDim.x + WARP_SZ - 1) / WARP_SZ;
        float total = 0.0f;
        for (int w = 0; w < nw; w++) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();

    float fnorm = sqrtf(scratch[0]);
    if (fnorm > m_norm_max) {
        float scale = m_norm_max / fnorm;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_ptr[idx] *= scale;
        }
    }
    __syncthreads();
}
