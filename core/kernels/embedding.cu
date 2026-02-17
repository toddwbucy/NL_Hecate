// Embedding Gather/Scatter CUDA Kernels — GPU-Resident Model
//
// Forward: gather rows from embedding table by token ID.
//   Grid=(seq_len), Block=(min(d, 1024)).
//   output[t*d + i] = w_embed[input_ids[t]*d + i]
//
// Backward: scatter-add gradients back to embedding table rows.
//   Grid=(seq_len), Block=(min(d, 1024)).
//   d_embed[target_tok*d + i] += d_embedded[t*d + i]  (atomicAdd for multi-token)
//
// All fp32. input_ids passed as int*.

#include <cuda_runtime.h>

__global__ void embedding_gather_kernel(
    const float* __restrict__ w_embed,    // [vocab, d]
    const int*   __restrict__ input_ids,  // [seq_len]
    float*       __restrict__ output,     // [seq_len, d]
    int seq_len, int d)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int tok = input_ids[t];
    int base_in  = tok * d;
    int base_out = t * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        output[base_out + i] = w_embed[base_in + i];
    }
}

__global__ void embedding_scatter_add_kernel(
    const float* __restrict__ d_embedded,  // [seq_len, d]
    const int*   __restrict__ input_ids,   // [seq_len]
    float*       __restrict__ d_embed,     // [vocab, d] — gradient accumulator
    int seq_len, int d)
{
    int t = blockIdx.x;
    if (t >= seq_len) return;

    int tok = input_ids[t];
    int base_grad = tok * d;
    int base_in   = t * d;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        atomicAdd(&d_embed[base_grad + i], d_embedded[base_in + i]);
    }
}

// C entry points for FFI
extern "C" {

void embedding_gather_cuda(
    const float* w_embed, const int* input_ids, float* output,
    int seq_len, int d)
{
    int block = (d < 1024) ? d : 1024;
    embedding_gather_kernel<<<seq_len, block>>>(w_embed, input_ids, output, seq_len, d);
}

void embedding_scatter_add_cuda(
    const float* d_embedded, const int* input_ids, float* d_embed,
    int seq_len, int d)
{
    int block = (d < 1024) ? d : 1024;
    embedding_scatter_add_kernel<<<seq_len, block>>>(d_embedded, input_ids, d_embed, seq_len, d);
}

} // extern "C"
