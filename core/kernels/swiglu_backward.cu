// SwiGluMlp Backward CUDA Kernel — task_71731d
//
// Backward pass for HOPE §7.3 SwiGLU MLP memory rule.
//
// Forward recap:
//   gate_out = X @ gate_proj.T               (saved as gate_buf)
//   up_out   = X @ up_proj.T                 (saved as up_buf)
//   sig      = sigmoid(gate_out)             (saved as cache_buf)
//   fused    = gate_out * sig * up_out        (saved as fused_buf)
//   Y        = fused @ down_proj.T
//
// Backward (standard backprop through SwiGLU):
//   d_fused      = d_Y @ down_proj
//   d_down_proj  = fused.T @ d_Y
//   dsilu[i]     = sig[i] * (1 + gate_out[i] * (1 - sig[i]))
//   d_up[i]      = d_fused[i] * gate_out[i] * sig[i]   = d_fused[i] * silu(gate[i])
//   d_gate[i]    = d_fused[i] * up_out[i] * dsilu[i]
//   d_gate_proj  = d_gate.T @ X
//   d_up_proj    = d_up.T @ X
//   d_X          = d_gate @ gate_proj + d_up @ up_proj
//
// Interface: ALL pointers are HOST (CPU) pointers. Kernel manages device allocation.
//
// Spec: specs/infrastructure/build/02_llama_level_stacking.md

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static cublasHandle_t g_cublas_handle_bwd = nullptr;

static cublasHandle_t get_cublas_handle_bwd(void) {
    if (g_cublas_handle_bwd == nullptr) {
        cublasStatus_t st = cublasCreate(&g_cublas_handle_bwd);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[NL_Hecate FATAL] cublasCreate (swiglu_bwd) failed: %d\n", (int)st);
            abort();
        }
    }
    return g_cublas_handle_bwd;
}

static inline void check_cublas_bwd(cublasStatus_t st, const char* label) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NL_Hecate FATAL] cuBLAS error in swiglu_bwd/%s: %d\n", label, (int)st);
        abort();
    }
}

static inline void check_cuda_bwd(cudaError_t err, const char* label) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] CUDA error in swiglu_bwd/%s: %s\n", label, cudaGetErrorString(err));
        abort();
    }
}

static void* dev_alloc_bwd(size_t bytes, const char* label) {
    void* ptr = nullptr;
    check_cuda_bwd((cudaError_t)cudaMalloc(&ptr, bytes), label);
    return ptr;
}

static void h2d(void* dev, const void* host, size_t bytes) {
    check_cuda_bwd(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice), "H2D");
}

static void d2h(void* host, const void* dev, size_t bytes) {
    check_cuda_bwd(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost), "D2H");
}

// ── SiLU gate backward kernel ──────────────────────────────────────────────
//
// d_gate[i]  = d_fused[i] * up[i] * sig[i] * (1 + gate[i] * (1 - sig[i]))
// d_up[i]    = d_fused[i] * gate[i] * sig[i]
__global__ void swiglu_fuse_backward_kernel(
    const float* __restrict__ d_fused_in,  // [N] upstream gradient through fused
    const float* __restrict__ gate,        // [N] gate_out from forward
    const float* __restrict__ up,          // [N] up_out from forward
    const float* __restrict__ gate_cache,  // [N] sigmoid(gate) from forward
    float* __restrict__ d_gate_out,        // [N] output: grad w.r.t. gate_out
    float* __restrict__ d_up_out,          // [N] output: grad w.r.t. up_out
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float df  = d_fused_in[i];
    float g   = gate[i];
    float sig = gate_cache[i];
    float u   = up[i];

    // d/d_gate [ gate * sig * up ] = up * sig * (1 + gate * (1 - sig))
    float dsilu     = sig * (1.0f + g * (1.0f - sig));
    d_gate_out[i]   = df * u * dsilu;

    // d/d_up [ gate * sig * up ] = gate * sig = silu(gate)
    d_up_out[i]     = df * g * sig;
}

// ── Public C interface ─────────────────────────────────────────────────────
// ALL pointers are HOST pointers.
extern "C" void swiglu_backward_f32_cuda(
    const float* d_Y,          // host: [seq_len × d_model]   upstream gradient
    const float* X,            // host: [seq_len × d_model]   input from forward
    const float* gate_proj,    // host: [intermediate × d_model]
    const float* up_proj,      // host: [intermediate × d_model]
    const float* down_proj,    // host: [d_model × intermediate]
    const float* fused_buf,    // host: [seq_len × intermediate] fused from forward
    const float* gate_buf,     // host: [seq_len × intermediate] gate_out from forward
    const float* up_buf,       // host: [seq_len × intermediate] up_out from forward
    const float* cache_buf,    // host: [seq_len × intermediate] sigmoid from forward
    float* d_X,                // host: [seq_len × d_model]   output grad
    float* d_gate_proj,        // host: [intermediate × d_model] output grad
    float* d_up_proj,          // host: [intermediate × d_model] output grad
    float* d_down_proj,        // host: [d_model × intermediate] output grad
    int seq_len,
    int d_model,
    int intermediate)
{
    cublasHandle_t h = get_cublas_handle_bwd();
    const float alpha1 = 1.0f, beta0 = 0.0f, beta1 = 1.0f;

    size_t szX    = (size_t)seq_len * d_model * sizeof(float);
    size_t szGate = (size_t)intermediate * d_model * sizeof(float);
    size_t szDown = (size_t)d_model * intermediate * sizeof(float);
    size_t szBuf  = (size_t)seq_len * intermediate * sizeof(float);
    int N         = seq_len * intermediate;

    // Allocate device memory — inputs
    float* ddY        = (float*)dev_alloc_bwd(szX,    "ddY");
    float* dX_dev     = (float*)dev_alloc_bwd(szX,    "dX");
    float* dGateProj  = (float*)dev_alloc_bwd(szGate, "dGateProj");
    float* dUpProj    = (float*)dev_alloc_bwd(szGate, "dUpProj");
    float* dDownProj  = (float*)dev_alloc_bwd(szDown, "dDownProj");
    float* dFused     = (float*)dev_alloc_bwd(szBuf,  "dFused");
    float* dGateBuf   = (float*)dev_alloc_bwd(szBuf,  "dGateBuf");
    float* dUpBuf     = (float*)dev_alloc_bwd(szBuf,  "dUpBuf");
    float* dCacheBuf  = (float*)dev_alloc_bwd(szBuf,  "dCacheBuf");
    // Scratch buffers
    float* dDFused    = (float*)dev_alloc_bwd(szBuf,  "dDFused");
    float* dDGateOut  = (float*)dev_alloc_bwd(szBuf,  "dDGateOut");
    float* dDUpOut    = (float*)dev_alloc_bwd(szBuf,  "dDUpOut");
    // Output grads
    float* dDGateProj = (float*)dev_alloc_bwd(szGate, "dDGateProj");
    float* dDUpProj   = (float*)dev_alloc_bwd(szGate, "dDUpProj");
    float* dDDownProj = (float*)dev_alloc_bwd(szDown, "dDDownProj");

    // Upload
    h2d(ddY,       d_Y,       szX);
    h2d(dX_dev,    X,         szX);
    h2d(dGateProj, gate_proj, szGate);
    h2d(dUpProj,   up_proj,   szGate);
    h2d(dDownProj, down_proj, szDown);
    h2d(dFused,    fused_buf, szBuf);
    h2d(dGateBuf,  gate_buf,  szBuf);
    h2d(dUpBuf,    up_buf,    szBuf);
    h2d(dCacheBuf, cache_buf, szBuf);

    // Step 1: d_fused = d_Y @ down_proj
    // d_fused[seq_len × inter] = d_Y[seq_len × d_model] @ down_proj[d_model × inter]
    // col-major: (inter × seq_len) = down_proj.T(inter × d_model) @ d_Y.T(d_model × seq_len)
    // = sgemm(OP_T, OP_N, inter, seq_len, d_model, alpha, down_proj, d_model, d_Y, d_model, 0, d_fused, inter)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            dDownProj, d_model,
            ddY, d_model,
            &beta0,
            dDFused, intermediate),
        "d_fused = d_Y @ down_proj");

    // Step 2: d_down_proj = fused.T @ d_Y  → stored as [d_model × inter]
    // d_down_proj[d_model × inter] row-major = d_Y.T[d_model × seq_len] @ fused[seq_len × inter]
    // col-major: (inter × d_model) = fused.T(inter × seq_len) @ d_Y(seq_len × d_model) ...
    // Actually want: d_down_proj(d_model × inter) = d_Y.T @ fused
    // = sgemm(OP_N, OP_T, d_model, inter, seq_len, alpha, d_Y, d_model, fused, inter, 0, d_down_proj, d_model)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d_model, intermediate, seq_len,
            &alpha1,
            ddY, d_model,
            dFused, intermediate,
            &beta0,
            dDDownProj, d_model),
        "d_down_proj");

    // Step 3: elementwise backward through SiLU gate fusion
    int block = 256;
    int grid  = (N + block - 1) / block;
    swiglu_fuse_backward_kernel<<<grid, block>>>(
        dDFused, dGateBuf, dUpBuf, dCacheBuf,
        dDGateOut, dDUpOut, N);
    check_cuda_bwd(cudaGetLastError(), "swiglu_fuse_backward_kernel");

    // Step 4: d_gate_proj = d_gate_out.T @ X → stored as [inter × d_model]
    // = sgemm(OP_N, OP_T, inter, d_model, seq_len, alpha, d_gate_out, inter, X, d_model, 0, d_gate_proj, inter)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            intermediate, d_model, seq_len,
            &alpha1,
            dDGateOut, intermediate,
            dX_dev, d_model,
            &beta0,
            dDGateProj, intermediate),
        "d_gate_proj");

    // Step 5: d_up_proj = d_up_out.T @ X  (same layout)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            intermediate, d_model, seq_len,
            &alpha1,
            dDUpOut, intermediate,
            dX_dev, d_model,
            &beta0,
            dDUpProj, intermediate),
        "d_up_proj");

    // Step 6a: d_X = d_gate_out @ gate_proj
    // d_X[seq_len × d_model] = d_gate_out[seq_len × inter] @ gate_proj[inter × d_model]
    // col-major: (d_model × seq_len) = gate_proj.T(d_model × inter) @ d_gate_out.T(inter × seq_len)
    // = sgemm(OP_T, OP_N, d_model, seq_len, inter, alpha, gate_proj, inter, d_gate_out, inter, 0, d_X, d_model)
    float* dDX = (float*)dev_alloc_bwd(szX, "dDX");
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            dGateProj, intermediate,
            dDGateOut, intermediate,
            &beta0,
            dDX, d_model),
        "d_X = d_gate @ gate_proj");

    // Step 6b: d_X += d_up_out @ up_proj  (beta=1 to accumulate)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            dUpProj, intermediate,
            dDUpOut, intermediate,
            &beta1,
            dDX, d_model),
        "d_X += d_up @ up_proj");

    check_cuda_bwd(cudaDeviceSynchronize(), "sync");

    // Copy results to host
    d2h(d_X,        dDX,        szX);
    d2h(d_gate_proj, dDGateProj, szGate);
    d2h(d_up_proj,  dDUpProj,   szGate);
    d2h(d_down_proj, dDDownProj, szDown);

    // Free device memory
    cudaFree(ddY); cudaFree(dX_dev); cudaFree(dGateProj); cudaFree(dUpProj);
    cudaFree(dDownProj); cudaFree(dFused); cudaFree(dGateBuf); cudaFree(dUpBuf);
    cudaFree(dCacheBuf); cudaFree(dDFused); cudaFree(dDGateOut); cudaFree(dDUpOut);
    cudaFree(dDGateProj); cudaFree(dDUpProj); cudaFree(dDDownProj); cudaFree(dDX);
}
