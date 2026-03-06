// TNT forward helper kernels — lightweight glue for chunkwise parallelism.
//
// These kernels do NOT contain the sequential M recurrence (that's in
// titans_forward.cu / delta_forward.cu, reused with batch_size=N).
// They handle only:
//   1. Broadcasting global M → N copies for parallel local memories
//   2. Mean-pooling local outputs into shard summary (k_sum, v_sum)
//   3. Updating global M via outer product with summary vectors
//
// Source: TNT (2511.07343) §2-3, eqs 3, 5, 6, 14.
// All fp32 (memory operations are unconditionally fp32 per spec).

#include <cstdio>
#include <cstdlib>

// ── Helper: CUDA launch error check (aborts on failure) ──────────────

static inline void check_cuda_launch_tnt(const char* name, int d) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[TNT] %s launch failed (d=%d): %s\n",
                name, d, cudaGetErrorString(err));
        abort();
    }
}

// ── Kernel 1: Broadcast global M → N copies ─────────────────────────
//
// m_src: [d*d]        — single global memory matrix
// m_dst: [N * d*d]    — N contiguous copies
// Grid=(1), Block=(min(d*d, 1024))

__global__ void tnt_broadcast_m_kernel(
    const float* __restrict__ m_src,   // [d*d]
    float* __restrict__ m_dst,         // [N*d*d]
    int N, int dd)
{
    int tid = threadIdx.x;
    // Each thread handles multiple elements via striding
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        float val = m_src[idx];
        for (int n = 0; n < N; n++) {
            m_dst[n * dd + idx] = val;
        }
    }
}

extern "C"
void tnt_broadcast_m_f32_cuda(
    const float* m_src, float* m_dst,
    int N, int d)
{
    if (d <= 0 || N <= 0) {
        fprintf(stderr, "tnt_broadcast_m_f32_cuda: d=%d and N=%d must be > 0\n", d, N);
        abort();
    }
    int dd = d * d;
    int block = (dd < 1024) ? dd : 1024;
    tnt_broadcast_m_kernel<<<1, block>>>(m_src, m_dst, N, dd);
    check_cuda_launch_tnt("tnt_broadcast_m", d);
}

// ── Kernel 2: Shard summary via mean-pooling ──────────────────────────
//
// local_y: [shard_len, d]  — outputs from all local memories concatenated
// k_sum:   [d]             — mean over tokens
// v_sum:   [d]             — same as k_sum (mean-pooling uses same vector for both)
// Grid=(1), Block=(min(d, 1024))

__global__ void tnt_shard_summary_mean_kernel(
    const float* __restrict__ local_y,  // [shard_len, d]
    float* __restrict__ k_sum,          // [d]
    float* __restrict__ v_sum,          // [d]
    int shard_len, int d)
{
    int j = threadIdx.x;
    // Each thread computes one element of the summary vector
    for (int jj = j; jj < d; jj += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < shard_len; t++) {
            sum += local_y[t * d + jj];
        }
        float mean = sum / (float)shard_len;
        k_sum[jj] = mean;
        v_sum[jj] = mean;
    }
}

extern "C"
void tnt_shard_summary_mean_f32_cuda(
    const float* local_y, float* k_sum, float* v_sum,
    int shard_len, int d)
{
    if (d <= 0 || shard_len <= 0) {
        fprintf(stderr, "tnt_shard_summary_mean_f32_cuda: d=%d and shard_len=%d must be > 0\n", d, shard_len);
        abort();
    }
    int block = (d < 1024) ? d : 1024;
    tnt_shard_summary_mean_kernel<<<1, block>>>(local_y, k_sum, v_sum, shard_len, d);
    check_cuda_launch_tnt("tnt_shard_summary_mean", d);
}

// ── Kernel 3: Global memory update via outer product ────────────────
//
// global_m[i,j] = alpha * global_m[i,j] + v_sum[i] * k_sum[j]
// Grid=(1), Block=(min(d*d, 1024))

__global__ void tnt_global_update_kernel(
    float* __restrict__ global_m,      // [d*d] — updated in-place
    const float* __restrict__ k_sum,   // [d]
    const float* __restrict__ v_sum,   // [d]
    int d, float alpha)
{
    int tid = threadIdx.x;
    int dd = d * d;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        int i = idx / d;
        int j = idx % d;
        global_m[idx] = alpha * global_m[idx] + v_sum[i] * k_sum[j];
    }
}

extern "C"
void tnt_global_update_f32_cuda(
    float* global_m, const float* k_sum, const float* v_sum,
    int d, float alpha)
{
    if (d <= 0) {
        fprintf(stderr, "tnt_global_update_f32_cuda: d=%d must be > 0\n", d);
        abort();
    }
    int dd = d * d;
    int block = (dd < 1024) ? dd : 1024;
    tnt_global_update_kernel<<<1, block>>>(global_m, k_sum, v_sum, d, alpha);
    check_cuda_launch_tnt("tnt_global_update", d);
}
