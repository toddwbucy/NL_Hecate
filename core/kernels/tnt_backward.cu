// TNT backward helper kernels — gradients for the lightweight glue operations.
//
// The heavy backward math (sequential M recurrence gradients) is in
// titans_backward.cu / delta_backward.cu, reused with batch_size=N.
// These kernels handle only:
//   1. Backward through global M update (outer product)
//   2. Backward through mean-pooling shard summary
//   3. Combining upstream + global gradient contributions
//
// Source: TNT (2511.07343) §2-3.
// All fp32.

#include <cstdio>

static inline void check_cuda_launch_tnt_bw(const char* name, int d) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[TNT backward] %s launch failed (d=%d): %s\n",
                name, d, cudaGetErrorString(err));
    }
}

// ── Kernel 1: Backward through global memory update ─────────────────
//
// Forward: m_new[i,j] = alpha * m_old[i,j] + v_sum[i] * k_sum[j]
// Backward:
//   d_m_old[i,j] = alpha * d_m_new[i,j]
//   d_v_sum[i]  += sum_j d_m_new[i,j] * k_sum[j]
//   d_k_sum[j]  += sum_i d_m_new[i,j] * v_sum[i]
//
// Grid=(1), Block=(min(d, 1024))
// Each thread handles one row of the d×d matrix.

__global__ void tnt_global_update_backward_kernel(
    const float* __restrict__ d_m_new,  // [d*d]
    const float* __restrict__ k_sum,    // [d]
    const float* __restrict__ v_sum,    // [d]
    float* __restrict__ d_m_old,        // [d*d]
    float* __restrict__ d_k_sum,        // [d] — must be pre-zeroed
    float* __restrict__ d_v_sum,        // [d] — must be pre-zeroed
    int d, float alpha)
{
    int i = threadIdx.x;
    // Each thread handles one row (index i) across all columns j
    for (int ii = i; ii < d; ii += blockDim.x) {
        float dv = 0.0f;
        for (int j = 0; j < d; j++) {
            float dm = d_m_new[ii * d + j];
            d_m_old[ii * d + j] = alpha * dm;
            dv += dm * k_sum[j];
            // d_k_sum[j] needs atomic since multiple rows contribute
            atomicAdd(&d_k_sum[j], dm * v_sum[ii]);
        }
        atomicAdd(&d_v_sum[ii], dv);
    }
}

extern "C"
void tnt_global_update_backward_f32_cuda(
    const float* d_m_new, const float* k_sum, const float* v_sum,
    float* d_m_old, float* d_k_sum, float* d_v_sum,
    int d, float alpha)
{
    int block = (d < 1024) ? d : 1024;
    tnt_global_update_backward_kernel<<<1, block>>>(
        d_m_new, k_sum, v_sum, d_m_old, d_k_sum, d_v_sum, d, alpha);
    check_cuda_launch_tnt_bw("tnt_global_update_backward", d);
}

// ── Kernel 2: Backward through mean-pooling shard summary ───────────
//
// Forward: k_sum[j] = v_sum[j] = (1/shard_len) * sum_t local_y[t,j]
// Backward: d_local_y[t,j] = (d_k_sum[j] + d_v_sum[j]) / shard_len
//
// Grid=(1), Block=(min(d, 1024))

__global__ void tnt_shard_summary_mean_backward_kernel(
    const float* __restrict__ d_k_sum,   // [d]
    const float* __restrict__ d_v_sum,   // [d]
    float* __restrict__ d_local_y,       // [shard_len, d] — accumulated (+=)
    int shard_len, int d)
{
    int j = threadIdx.x;
    for (int jj = j; jj < d; jj += blockDim.x) {
        float grad = (d_k_sum[jj] + d_v_sum[jj]) / (float)shard_len;
        for (int t = 0; t < shard_len; t++) {
            d_local_y[t * d + jj] += grad;
        }
    }
}

extern "C"
void tnt_shard_summary_mean_backward_f32_cuda(
    const float* d_k_sum, const float* d_v_sum,
    float* d_local_y, int shard_len, int d)
{
    int block = (d < 1024) ? d : 1024;
    tnt_shard_summary_mean_backward_kernel<<<1, block>>>(
        d_k_sum, d_v_sum, d_local_y, shard_len, d);
    check_cuda_launch_tnt_bw("tnt_shard_summary_mean_backward", d);
}

// ── Kernel 3: Combine gradient contributions ────────────────────────
//
// d_y_combined[i] = d_y_upstream[i] + d_y_global[i]
// Simple element-wise addition. Used to merge the direct upstream gradient
// with the gradient flowing back through the global M update.
//
// Grid=(ceil(n/1024)), Block=(1024)

__global__ void tnt_combine_gradients_kernel(
    const float* __restrict__ d_y_upstream,
    const float* __restrict__ d_y_global,
    float* __restrict__ d_y_combined,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_y_combined[idx] = d_y_upstream[idx] + d_y_global[idx];
    }
}

extern "C"
void tnt_combine_gradients_f32_cuda(
    const float* d_y_upstream, const float* d_y_global,
    float* d_y_combined, int n)
{
    int block = 1024;
    int grid = (n + block - 1) / block;
    tnt_combine_gradients_kernel<<<grid, block>>>(d_y_upstream, d_y_global, d_y_combined, n);
    check_cuda_launch_tnt_bw("tnt_combine_gradients", n);
}
