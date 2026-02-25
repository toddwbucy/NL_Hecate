// SwiGluMlp Forward CUDA Kernel — task_71731d
//
// Implements HOPE §7.3 ad-hoc level stacking: Llama-style SwiGLU MLP as a
// stateless CMS memory rule. No inner-loop M state. Three weight matrices
// (gate_proj, up_proj, down_proj) are outer-loop parameters updated by AdamW.
//
// Math (all tokens batched):
//   gate_out = X @ gate_proj.T             [seq_len × intermediate]
//   up_out   = X @ up_proj.T               [seq_len × intermediate]
//   sig      = sigmoid(gate_out)           [seq_len × intermediate]
//   fused    = gate_out * sig * up_out     [seq_len × intermediate]   (SwiGLU)
//   Y        = fused @ down_proj.T         [seq_len × d_model]
//
// Interface: ALL pointers are HOST (CPU) pointers. The kernel manages its own
// device allocations internally. This matches the existing pattern in dispatch.rs
// where device memory is caller-managed, but here we consolidate it for simplicity.
//
// Saved buffers (gate_buf, up_buf, fused_buf, cache_buf) are populated on the
// host side so Rust can store them in SwiGluMlpCache for the backward pass.
//
// cuBLAS GEMM note (column-major vs row-major):
// All weight matrices stored row-major in Rust: gate_proj[inter × d_model].
// cuBLAS operates column-major. We use CUBLAS_OP_T on weight matrices so
// cuBLAS treats them as [d_model × inter] column-major (== [inter × d_model] row-major).
// This gives us: gate_buf = gate_proj @ X.T correctly.
//
// Spec: specs/infrastructure/build/02_llama_level_stacking.md
// Source: HOPE (2512.24695) §7.3, Eq 71-72

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cublasHandle_t g_cublas_handle_fwd = nullptr;

static cublasHandle_t get_cublas_handle_fwd(void) {
    if (g_cublas_handle_fwd == nullptr) {
        cublasStatus_t st = cublasCreate(&g_cublas_handle_fwd);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[NL_Hecate FATAL] cublasCreate (swiglu_fwd) failed: %d\n", (int)st);
            abort();
        }
    }
    return g_cublas_handle_fwd;
}

static inline void check_cublas(cublasStatus_t st, const char* label) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NL_Hecate FATAL] cuBLAS error in swiglu_fwd/%s: %d\n", label, (int)st);
        abort();
    }
}

static inline void check_cuda(cudaError_t err, const char* label) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] CUDA error in swiglu_fwd/%s: %s\n", label, cudaGetErrorString(err));
        abort();
    }
}

static void* dev_alloc(size_t bytes, const char* label) {
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, bytes), label);
    return ptr;
}

static void host_to_dev(void* dev, const void* host, size_t bytes, const char* label) {
    check_cuda(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice), label);
}

static void dev_to_host(void* host, const void* dev, size_t bytes, const char* label) {
    check_cuda(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost), label);
}

// ── SiLU gate fusion kernel ───────────────────────────────────────────────
//
// Computes:
//   sig[i] = sigmoid(gate[i]) = 1 / (1 + exp(-gate[i]))
//   fused[i] = gate[i] * sig[i] * up[i]
//
// gate_cache stores sig values for reuse in backward.
// N = seq_len * intermediate.
__global__ void swiglu_fuse_kernel(
    const float* __restrict__ gate,   // [N] gate_out (device)
    const float* __restrict__ up,     // [N] up_out (device)
    float* __restrict__ fused,        // [N] output (device)
    float* __restrict__ gate_cache,   // [N] sigmoid(gate), saved for backward (device)
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    gate_cache[i] = sig;
    fused[i] = g * sig * up[i];
}

// ── Public C interface ─────────────────────────────────────────────────────
//
// ALL pointers are HOST pointers. Kernel allocates device memory, runs, copies back.
// gate_buf, up_buf, fused_buf, cache_buf are output host buffers populated here
// so Rust can cache them for the backward pass.
extern "C" void swiglu_forward_f32_cuda(
    const float* X,           // host: [seq_len × d_model]
    const float* gate_proj,   // host: [intermediate × d_model]  (row-major)
    const float* up_proj,     // host: [intermediate × d_model]  (row-major)
    const float* down_proj,   // host: [d_model × intermediate]  (row-major)
    float* Y,                 // host: [seq_len × d_model]  (output)
    float* gate_buf,          // host: [seq_len × intermediate]  (saved for bwd)
    float* up_buf,            // host: [seq_len × intermediate]  (saved for bwd)
    float* fused_buf,         // host: [seq_len × intermediate]  (saved for bwd)
    float* cache_buf,         // host: [seq_len × intermediate]  (sigmoid, saved for bwd)
    int seq_len,
    int d_model,
    int intermediate)
{
    cublasHandle_t h = get_cublas_handle_fwd();
    const float alpha1 = 1.0f, beta0 = 0.0f;

    size_t szX    = (size_t)seq_len * d_model * sizeof(float);
    size_t szGate = (size_t)intermediate * d_model * sizeof(float);
    size_t szDown = (size_t)d_model * intermediate * sizeof(float);
    size_t szBuf  = (size_t)seq_len * intermediate * sizeof(float);
    int N         = seq_len * intermediate;

    // Allocate device memory
    float* dX         = (float*)dev_alloc(szX,    "dX");
    float* dGateProj  = (float*)dev_alloc(szGate, "dGateProj");
    float* dUpProj    = (float*)dev_alloc(szGate, "dUpProj");
    float* dDownProj  = (float*)dev_alloc(szDown, "dDownProj");
    float* dY         = (float*)dev_alloc(szX,    "dY");
    float* dGateBuf   = (float*)dev_alloc(szBuf,  "dGateBuf");
    float* dUpBuf     = (float*)dev_alloc(szBuf,  "dUpBuf");
    float* dFusedBuf  = (float*)dev_alloc(szBuf,  "dFusedBuf");
    float* dCacheBuf  = (float*)dev_alloc(szBuf,  "dCacheBuf");

    // Upload inputs
    host_to_dev(dX, X, szX, "H2D X");
    host_to_dev(dGateProj, gate_proj, szGate, "H2D gate_proj");
    host_to_dev(dUpProj, up_proj, szGate, "H2D up_proj");
    host_to_dev(dDownProj, down_proj, szDown, "H2D down_proj");

    // gate_buf = X @ gate_proj.T
    // cuBLAS col-major: result(inter × seq_len) = gate_proj(inter × d) @ X.T(d × seq_len)
    // = sgemm(OP_N, OP_T, inter, seq_len, d_model, alpha, gate_proj, inter, X, d_model, 0, gate_buf, inter)
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            intermediate, seq_len, d_model,
            &alpha1,
            dGateProj, intermediate,
            dX, d_model,
            &beta0,
            dGateBuf, intermediate),
        "gate gemm");

    // up_buf = X @ up_proj.T  (same layout)
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            intermediate, seq_len, d_model,
            &alpha1,
            dUpProj, intermediate,
            dX, d_model,
            &beta0,
            dUpBuf, intermediate),
        "up gemm");

    // SiLU gate fusion: fused = silu(gate) * up, save sigmoid in cache
    int block = 256;
    int grid  = (N + block - 1) / block;
    swiglu_fuse_kernel<<<grid, block>>>(dGateBuf, dUpBuf, dFusedBuf, dCacheBuf, N);
    check_cuda(cudaGetLastError(), "swiglu_fuse_kernel launch");

    // Y = fused @ down_proj.T
    // fused[seq_len × inter], down_proj[d_model × inter]
    // result Y[seq_len × d_model]
    // col-major: Y(d_model × seq_len) = down_proj(d_model × inter) @ fused.T(inter × seq_len)
    // = sgemm(OP_N, OP_T, d_model, seq_len, inter, alpha, down_proj, d_model, fused, inter, 0, Y, d_model)
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d_model, seq_len, intermediate,
            &alpha1,
            dDownProj, d_model,
            dFusedBuf, intermediate,
            &beta0,
            dY, d_model),
        "down gemm");

    check_cuda(cudaDeviceSynchronize(), "sync");

    // Copy outputs to host
    dev_to_host(Y, dY, szX, "D2H Y");
    dev_to_host(gate_buf, dGateBuf, szBuf, "D2H gate_buf");
    dev_to_host(up_buf, dUpBuf, szBuf, "D2H up_buf");
    dev_to_host(fused_buf, dFusedBuf, szBuf, "D2H fused_buf");
    dev_to_host(cache_buf, dCacheBuf, szBuf, "D2H cache_buf");

    // Free device memory
    cudaFree(dX); cudaFree(dGateProj); cudaFree(dUpProj); cudaFree(dDownProj);
    cudaFree(dY); cudaFree(dGateBuf); cudaFree(dUpBuf);
    cudaFree(dFusedBuf); cudaFree(dCacheBuf);
}
