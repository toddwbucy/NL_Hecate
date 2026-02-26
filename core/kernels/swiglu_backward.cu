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
// Device buffer strategy: buffers are allocated ONCE on first call and reused
// across all subsequent calls. This eliminates per-step cudaMalloc/cudaFree
// overhead which dominates at large d (e.g. 67MB weight matrices for d=2048).
// A single process trains with a single (d, inter, seq_len) config, so one
// shared buffer pool covering all CMS levels is sufficient (levels execute
// sequentially through the Wengert tape, never overlapping).
//
// Interface: ALL pointers are HOST (CPU) pointers.
//
// Spec: specs/infrastructure/build/02_llama_level_stacking.md

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ── cuBLAS handle ─────────────────────────────────────────────────────────

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

// ── Error checking helpers ────────────────────────────────────────────────

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

// ── Persistent device buffer pool ────────────────────────────────────────
//
// Allocated once on first call with a given (d, inter, seq_len) triple.
// Re-allocated if dimensions change (unusual in training). Never freed during
// training — process lifetime is the resource scope.
//
// Buffer grouping by size:
//   szX    = seq × d   : ddY, dXFwd, dDX
//   szGate = inter × d : dGateProj, dUpProj, dDGateProj, dDUpProj
//   szDown = d × inter : dDownProj, dDDownProj
//   szBuf  = seq × inter: dFused, dGateBuf, dUpBuf, dCacheBuf,
//                         dDFused, dDGateOut, dDUpOut

static struct {
    // Input buffers — H2D every call
    float* ddY;         // [seq × d]    upstream gradient d_Y
    float* dXFwd;       // [seq × d]    forward input X
    float* dGateProj;   // [inter × d]  weight gate_proj
    float* dUpProj;     // [inter × d]  weight up_proj
    float* dDownProj;   // [d × inter]  weight down_proj
    float* dFused;      // [seq × inter] fused_buf from forward
    float* dGateBuf;    // [seq × inter] gate_buf from forward (gate_out)
    float* dUpBuf;      // [seq × inter] up_buf from forward (up_out)
    float* dCacheBuf;   // [seq × inter] cache_buf from forward (sigmoid)
    // Scratch buffers — computed on device
    float* dDFused;     // [seq × inter] d_fused = d_Y @ down_proj
    float* dDGateOut;   // [seq × inter] d_gate_out from SiLU backward
    float* dDUpOut;     // [seq × inter] d_up_out from SiLU backward
    // Output gradient buffers — D2H every call
    float* dDX;         // [seq × d]    output d_X
    float* dDGateProj;  // [inter × d]  output d_gate_proj
    float* dDUpProj;    // [inter × d]  output d_up_proj
    float* dDDownProj;  // [d × inter]  output d_down_proj
    int d, inter, seq_len;  // allocated dimensions (0 = unallocated)
} g_bwd_pool = {nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr,
                0, 0, 0};

// Ensure the persistent buffer pool is allocated for the given dimensions.
// Frees and re-allocates if dimensions changed (only happens during config changes).
static void ensure_bwd_pool(int d, int inter, int seq_len) {
    if (g_bwd_pool.ddY != nullptr
            && g_bwd_pool.d == d
            && g_bwd_pool.inter == inter
            && g_bwd_pool.seq_len == seq_len) {
        return; // Already allocated with matching dimensions
    }

    // Free stale buffers if any
    if (g_bwd_pool.ddY != nullptr) {
        cudaFree(g_bwd_pool.ddY);
        cudaFree(g_bwd_pool.dXFwd);
        cudaFree(g_bwd_pool.dGateProj);
        cudaFree(g_bwd_pool.dUpProj);
        cudaFree(g_bwd_pool.dDownProj);
        cudaFree(g_bwd_pool.dFused);
        cudaFree(g_bwd_pool.dGateBuf);
        cudaFree(g_bwd_pool.dUpBuf);
        cudaFree(g_bwd_pool.dCacheBuf);
        cudaFree(g_bwd_pool.dDFused);
        cudaFree(g_bwd_pool.dDGateOut);
        cudaFree(g_bwd_pool.dDUpOut);
        cudaFree(g_bwd_pool.dDX);
        cudaFree(g_bwd_pool.dDGateProj);
        cudaFree(g_bwd_pool.dDUpProj);
        cudaFree(g_bwd_pool.dDDownProj);
        g_bwd_pool.ddY = nullptr;
    }

    size_t szX    = (size_t)seq_len * d * sizeof(float);
    size_t szGate = (size_t)inter * d * sizeof(float);
    size_t szDown = (size_t)d * inter * sizeof(float);
    size_t szBuf  = (size_t)seq_len * inter * sizeof(float);

    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.ddY,       szX),    "alloc ddY");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dXFwd,     szX),    "alloc dXFwd");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dGateProj, szGate), "alloc dGateProj");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dUpProj,   szGate), "alloc dUpProj");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDownProj, szDown),  "alloc dDownProj");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dFused,    szBuf),  "alloc dFused");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dGateBuf,  szBuf),  "alloc dGateBuf");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dUpBuf,    szBuf),  "alloc dUpBuf");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dCacheBuf, szBuf),  "alloc dCacheBuf");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDFused,   szBuf),  "alloc dDFused");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDGateOut, szBuf),  "alloc dDGateOut");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDUpOut,   szBuf),  "alloc dDUpOut");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDX,       szX),    "alloc dDX");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDGateProj,szGate), "alloc dDGateProj");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDUpProj,  szGate), "alloc dDUpProj");
    check_cuda_bwd(cudaMalloc((void**)&g_bwd_pool.dDDownProj,szDown),  "alloc dDDownProj");

    g_bwd_pool.d = d; g_bwd_pool.inter = inter; g_bwd_pool.seq_len = seq_len;
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
//
// ALL pointers are HOST pointers. Kernel uses persistent device buffers
// (allocated once on first call per config, never freed). Weights and saved
// activations are H2D every call. Output grads d_X, d_gate_proj, d_up_proj,
// d_down_proj are D2H after computation.
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

    // Ensure persistent device buffers are allocated (first call only)
    ensure_bwd_pool(d_model, intermediate, seq_len);

    // Upload inputs (saved activations + weights change each step)
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.ddY,       d_Y,       szX,    cudaMemcpyHostToDevice), "H2D ddY");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dXFwd,     X,         szX,    cudaMemcpyHostToDevice), "H2D X");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dGateProj, gate_proj, szGate, cudaMemcpyHostToDevice), "H2D gate_proj");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dUpProj,   up_proj,   szGate, cudaMemcpyHostToDevice), "H2D up_proj");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dDownProj, down_proj, szDown, cudaMemcpyHostToDevice), "H2D down_proj");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dFused,    fused_buf, szBuf,  cudaMemcpyHostToDevice), "H2D fused_buf");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dGateBuf,  gate_buf,  szBuf,  cudaMemcpyHostToDevice), "H2D gate_buf");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dUpBuf,    up_buf,    szBuf,  cudaMemcpyHostToDevice), "H2D up_buf");
    check_cuda_bwd(cudaMemcpy(g_bwd_pool.dCacheBuf, cache_buf, szBuf,  cudaMemcpyHostToDevice), "H2D cache_buf");

    // Step 1: d_fused = d_Y @ down_proj
    // d_fused[seq_len × inter] = d_Y[seq_len × d_model] @ down_proj[d_model × inter]
    // Row-major trick: down_proj[d×inter] with lda=inter → cuBLAS sees down_proj.T (col-major [inter×d]).
    // transN uses down_proj.T. d_Y[seq×d] with lda=d → cuBLAS sees d_Y.T; transN uses d_Y.T.
    // Result d_fused.T[inter×seq] = down_proj.T[inter×d] @ d_Y.T[d×seq]. Written as d_fused_rm[seq×inter]. ✓
    // lda (transN): lda >= m=inter ✓. ldb (transN): ldb >= k=d_model ✓.
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            g_bwd_pool.dDownProj, intermediate,
            g_bwd_pool.ddY, d_model,
            &beta0,
            g_bwd_pool.dDFused, intermediate),
        "d_fused = d_Y @ down_proj");

    // Step 2: d_down_proj = d_Y.T @ fused  → stored as [d_model × inter]
    // d_down_proj.T[inter × d] = fused.T[inter×seq] @ d_Y[seq×d]
    // Row-major trick: fused[seq×inter] with lda=inter → cuBLAS sees fused.T (col-major [inter×seq]); transN.
    // d_Y[seq×d] with lda=d → cuBLAS sees d_Y.T; transT gives d_Y itself.
    // Result d_down_proj.T[inter×d] (m_c=inter, n_c=d, ldc=inter). Written to d_down_proj_rm[d×inter] ✓
    // lda (transN): lda >= m=inter ✓. ldb (transT): ldb >= n=d_model ✓.
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            intermediate, d_model, seq_len,
            &alpha1,
            g_bwd_pool.dFused, intermediate,
            g_bwd_pool.ddY, d_model,
            &beta0,
            g_bwd_pool.dDDownProj, intermediate),
        "d_down_proj");

    // Step 3: elementwise backward through SiLU gate fusion
    int block = 256;
    int grid  = (N + block - 1) / block;
    swiglu_fuse_backward_kernel<<<grid, block>>>(
        g_bwd_pool.dDFused, g_bwd_pool.dGateBuf, g_bwd_pool.dUpBuf, g_bwd_pool.dCacheBuf,
        g_bwd_pool.dDGateOut, g_bwd_pool.dDUpOut, N);
    check_cuda_bwd(cudaGetLastError(), "swiglu_fuse_backward_kernel");

    // Step 4: d_gate_proj = d_gate_out.T @ X → stored as [inter × d_model]
    // d_gate_proj.T[d×inter] = X.T[d×seq] @ d_gate_out[seq×inter]
    // X[seq×d] with lda=d → cuBLAS sees X.T (col-major [d×seq]); transN uses X.T.
    // d_gate_out[seq×inter] with lda=inter → cuBLAS sees d_gate_out.T; transT gives d_gate_out.
    // Result d_gate_proj.T[d×inter] (m_c=d, n_c=inter, ldc=d). Written to d_gate_proj_rm[inter×d] ✓
    // lda (transN): lda >= m=d_model ✓. ldb (transT): ldb >= n=inter ✓.
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d_model, intermediate, seq_len,
            &alpha1,
            g_bwd_pool.dXFwd, d_model,
            g_bwd_pool.dDGateOut, intermediate,
            &beta0,
            g_bwd_pool.dDGateProj, d_model),
        "d_gate_proj");

    // Step 5: d_up_proj = d_up_out.T @ X  (same layout as d_gate_proj)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_T,
            d_model, intermediate, seq_len,
            &alpha1,
            g_bwd_pool.dXFwd, d_model,
            g_bwd_pool.dDUpOut, intermediate,
            &beta0,
            g_bwd_pool.dDUpProj, d_model),
        "d_up_proj");

    // Step 6a: d_X = d_gate_out @ gate_proj
    // d_X[seq_len × d_model] = d_gate_out[seq_len × inter] @ gate_proj[inter × d_model]
    // d_X.T[d×seq] = gate_proj.T[d×inter] @ d_gate_out.T[inter×seq]
    // gate_proj[inter×d] with lda=d_model → cuBLAS sees gate_proj.T (col-major [d×inter]); transN uses gate_proj.T.
    // d_gate_out[seq×inter] with lda=inter → cuBLAS sees d_gate_out.T; transN uses d_gate_out.T.
    // Result d_X.T[d×seq] (m_c=d, n_c=seq, ldc=d_model). Written to d_X_rm[seq×d] ✓
    // lda (transN): lda >= m=d_model ✓. ldb (transN): ldb >= k=inter ✓.
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            g_bwd_pool.dGateProj, d_model,
            g_bwd_pool.dDGateOut, intermediate,
            &beta0,
            g_bwd_pool.dDX, d_model),
        "d_X = d_gate @ gate_proj");

    // Step 6b: d_X += d_up_out @ up_proj  (beta=1 to accumulate)
    check_cublas_bwd(
        cublasSgemm(h,
            CUBLAS_OP_N, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            g_bwd_pool.dUpProj, d_model,
            g_bwd_pool.dDUpOut, intermediate,
            &beta1,
            g_bwd_pool.dDX, d_model),
        "d_X += d_up @ up_proj");

    check_cuda_bwd(cudaDeviceSynchronize(), "sync");

    // Copy output gradients to host
    check_cuda_bwd(cudaMemcpy(d_X,        g_bwd_pool.dDX,       szX,    cudaMemcpyDeviceToHost), "D2H d_X");
    check_cuda_bwd(cudaMemcpy(d_gate_proj, g_bwd_pool.dDGateProj, szGate, cudaMemcpyDeviceToHost), "D2H d_gate_proj");
    check_cuda_bwd(cudaMemcpy(d_up_proj,  g_bwd_pool.dDUpProj,  szGate, cudaMemcpyDeviceToHost), "D2H d_up_proj");
    check_cuda_bwd(cudaMemcpy(d_down_proj, g_bwd_pool.dDDownProj, szDown, cudaMemcpyDeviceToHost), "D2H d_down_proj");
    // Note: dGateProj, dUpProj, dDownProj remain on device (reused on next call)
}
