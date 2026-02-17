// Cross-Entropy Loss CUDA Kernel — GPU-Resident Model
//
// Fused softmax + negative log-likelihood forward and backward.
// Only kernel that produces a scalar output (loss) — 4 bytes D2H.
//
// Forward: softmax(logits) → -log(p[target]) → mean over valid tokens → loss
// Backward: d_logits[t,j] = (softmax[t,j] - 1{j==target}) / count
//
// Grid=(seq_len), Block=(min(vocab, 1024)).
// Uses shared memory for:
//   - max reduction (numerical stability)
//   - sum_exp reduction
//   - per-token NLL (atomicAdd to global loss)

#include <cuda_runtime.h>
#include <float.h>

// ── Forward: fused softmax + NLL loss ────────────────────────────────

__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,     // [seq_len, vocab]
    const int*   __restrict__ target_ids, // [seq_len]
    float*       __restrict__ loss_out,   // [1] — atomic accumulate
    int seq_len, int vocab)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int target = target_ids[t];
    if (target < 0 || target >= vocab) return;  // skip invalid targets

    const float* row = logits + t * vocab;

    // Step 1: Find max for numerical stability (parallel reduction)
    extern __shared__ float sdata[];
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        float val = row[j];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Tree reduction for max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Step 2: Compute sum_exp (parallel reduction)
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        local_sum += expf(row[j] - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Step 3: NLL = -(logit[target] - max_val - log(sum_exp))
    if (threadIdx.x == 0) {
        float nll = -(row[target] - max_val - logf(sum_exp));
        atomicAdd(loss_out, nll);
    }
}

extern "C" void cross_entropy_forward_cuda(
    const float* logits, const int* target_ids, float* loss_out,
    int seq_len, int vocab)
{
    // Zero the output first
    cudaMemset(loss_out, 0, sizeof(float));

    int block = (vocab < 1024) ? vocab : 1024;
    // Ensure block is power of 2 for reduction (round down)
    int b = 1;
    while (b * 2 <= block) b *= 2;
    block = b;
    if (block < 32) block = 32;

    int smem = block * sizeof(float);
    cross_entropy_forward_kernel<<<seq_len, block, smem>>>(
        logits, target_ids, loss_out, seq_len, vocab);
}

// ── Backward: softmax gradient ───────────────────────────────────────
// d_logits[t,j] = (softmax(logits)[t,j] - 1{j==target}) / count

__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ logits,     // [seq_len, vocab]
    const int*   __restrict__ target_ids, // [seq_len]
    float*       __restrict__ d_logits,   // [seq_len, vocab]
    int seq_len, int vocab, float inv_count)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int target = target_ids[t];
    if (target < 0 || target >= vocab) {
        // Zero out gradient for invalid targets
        for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
            d_logits[t * vocab + j] = 0.0f;
        }
        return;
    }

    const float* row = logits + t * vocab;

    // Compute max (shared memory reduction)
    extern __shared__ float sdata[];
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        float val = row[j];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float max_val = sdata[0];

    // Compute sum_exp
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        local_sum += expf(row[j] - max_val);
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0];

    // Write gradient: softmax - one_hot, scaled by 1/count
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        float softmax_j = expf(row[j] - max_val) / sum_exp;
        float grad = softmax_j;
        if (j == target) grad -= 1.0f;
        d_logits[t * vocab + j] = grad * inv_count;
    }
}

extern "C" void cross_entropy_backward_cuda(
    const float* logits, const int* target_ids, float* d_logits,
    int seq_len, int vocab, float inv_count)
{
    int block = (vocab < 1024) ? vocab : 1024;
    int b = 1;
    while (b * 2 <= block) b *= 2;
    block = b;
    if (block < 32) block = 32;

    int smem = block * sizeof(float);
    cross_entropy_backward_kernel<<<seq_len, block, smem>>>(
        logits, target_ids, d_logits, seq_len, vocab, inv_count);
}
