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
// Device buffer strategy: buffers are allocated ONCE on first call and reused
// across all subsequent calls. This eliminates per-step cudaMalloc/cudaFree
// overhead which dominates at large d (e.g. 67MB weight matrices for d=2048).
// A single process trains with a single (d, inter, seq_len) config, so one
// shared buffer pool covering all CMS levels is sufficient (levels execute
// sequentially through the Wengert tape, never overlapping).
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
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── cuBLAS handle ─────────────────────────────────────────────────────────

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

// ── Error checking helpers ────────────────────────────────────────────────

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

// ── Persistent device buffer pool ────────────────────────────────────────
//
// Allocated once on first call with a given (d, inter, seq_len) triple.
// Re-allocated if dimensions change (unusual in training). Never freed during
// training — process lifetime is the resource scope.

static struct {
    float* dGateProj;   // [inter × d] — weight, H2D every call
    float* dUpProj;     // [inter × d] — weight, H2D every call
    float* dDownProj;   // [d × inter] — weight, H2D every call
    float* dX;          // [seq × d] — input, H2D every call
    float* dY;          // [seq × d] — output, D2H every call
    float* dGateBuf;    // [seq × inter] — intermediate, D2H every call (saved for bwd)
    float* dUpBuf;      // [seq × inter] — intermediate, D2H every call (saved for bwd)
    float* dFusedBuf;   // [seq × inter] — intermediate, D2H every call (saved for bwd)
    float* dCacheBuf;   // [seq × inter] — sigmoid cache, D2H every call (saved for bwd)
    int d, inter, seq_len;  // allocated dimensions (0 = unallocated)
} g_fwd_pool = {nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0};

static std::mutex g_fwd_pool_mutex;

// Ensure the persistent buffer pool is allocated for the given dimensions.
// Frees and re-allocates if dimensions changed (only happens during config changes).
static void ensure_fwd_pool(int d, int inter, int seq_len) {
    std::lock_guard<std::mutex> lock(g_fwd_pool_mutex);
    if (g_fwd_pool.dGateProj != nullptr
            && g_fwd_pool.d == d
            && g_fwd_pool.inter == inter
            && g_fwd_pool.seq_len == seq_len) {
        return; // Already allocated with matching dimensions
    }

    // Free stale buffers if any
    if (g_fwd_pool.dGateProj != nullptr) {
        check_cuda(cudaFree(g_fwd_pool.dGateProj), "free gate_proj");
        check_cuda(cudaFree(g_fwd_pool.dUpProj),   "free up_proj");
        check_cuda(cudaFree(g_fwd_pool.dDownProj), "free down_proj");
        check_cuda(cudaFree(g_fwd_pool.dX),        "free X");
        check_cuda(cudaFree(g_fwd_pool.dY),        "free Y");
        check_cuda(cudaFree(g_fwd_pool.dGateBuf),  "free gate_buf");
        check_cuda(cudaFree(g_fwd_pool.dUpBuf),    "free up_buf");
        check_cuda(cudaFree(g_fwd_pool.dFusedBuf), "free fused_buf");
        check_cuda(cudaFree(g_fwd_pool.dCacheBuf), "free cache_buf");
        g_fwd_pool.dGateProj = nullptr;
    }

    size_t szGate = (size_t)inter * d * sizeof(float);       // gate_proj, up_proj
    size_t szDown = (size_t)d * inter * sizeof(float);       // down_proj
    size_t szX    = (size_t)seq_len * d * sizeof(float);     // X, Y
    size_t szBuf  = (size_t)seq_len * inter * sizeof(float); // gate/up/fused/cache bufs

    check_cuda(cudaMalloc((void**)&g_fwd_pool.dGateProj, szGate), "alloc gate_proj");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dUpProj,   szGate), "alloc up_proj");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dDownProj, szDown), "alloc down_proj");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dX,        szX),   "alloc X");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dY,        szX),   "alloc Y");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dGateBuf,  szBuf), "alloc gate_buf");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dUpBuf,    szBuf), "alloc up_buf");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dFusedBuf, szBuf), "alloc fused_buf");
    check_cuda(cudaMalloc((void**)&g_fwd_pool.dCacheBuf, szBuf), "alloc cache_buf");

    g_fwd_pool.d = d; g_fwd_pool.inter = inter; g_fwd_pool.seq_len = seq_len;
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
// ALL pointers are HOST pointers. Kernel uses persistent device buffers
// (allocated once on first call per config, never freed). Weights are
// H2D every call (they change after each AdamW step). gate_buf, up_buf,
// fused_buf, cache_buf are output host buffers populated for the backward pass.
// NOT thread-safe for concurrent calls — Wengert tape guarantees sequential
// execution of all CMS levels so buffer reuse is safe during training.
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
    if (seq_len <= 0 || d_model <= 0 || intermediate <= 0) {
        fprintf(stderr,
                "[NL_Hecate FATAL] swiglu_fwd invalid dims: seq_len=%d d_model=%d intermediate=%d\n",
                seq_len, d_model, intermediate);
        abort();
    }
    cublasHandle_t h = get_cublas_handle_fwd();
    const float alpha1 = 1.0f, beta0 = 0.0f;

    size_t szGate = (size_t)intermediate * d_model * sizeof(float);
    size_t szDown = (size_t)d_model * intermediate * sizeof(float);
    size_t szX    = (size_t)seq_len * d_model * sizeof(float);
    size_t szBuf  = (size_t)seq_len * intermediate * sizeof(float);
    size_t N      = (size_t)seq_len * intermediate;

    // Ensure persistent device buffers are allocated (first call only)
    ensure_fwd_pool(d_model, intermediate, seq_len);

    // Upload inputs (weights change every AdamW step; X changes every chunk)
    check_cuda(cudaMemcpy(g_fwd_pool.dGateProj, gate_proj, szGate, cudaMemcpyHostToDevice), "H2D gate_proj");
    check_cuda(cudaMemcpy(g_fwd_pool.dUpProj,   up_proj,   szGate, cudaMemcpyHostToDevice), "H2D up_proj");
    check_cuda(cudaMemcpy(g_fwd_pool.dDownProj, down_proj, szDown, cudaMemcpyHostToDevice), "H2D down_proj");
    check_cuda(cudaMemcpy(g_fwd_pool.dX, X, szX, cudaMemcpyHostToDevice), "H2D X");

    // gate_buf = X @ gate_proj.T
    // Row-major trick: gate_proj[inter×d] with lda=d_model → cuBLAS sees gate_proj.T (col-major [d×inter]).
    // transa=T transposes it back to gate_proj. X[seq×d] with lda=d_model → cuBLAS sees X.T; transb=N uses X.T.
    // Result gate_buf.T[inter×seq] = gate_proj[inter×d] @ X.T[d×seq]. Written as gate_buf_rm[seq×inter]. ✓
    // lda constraint (transa=T): lda >= k=d_model ✓. ldb (transb=N): ldb >= k=d_model ✓.
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            g_fwd_pool.dGateProj, d_model,
            g_fwd_pool.dX, d_model,
            &beta0,
            g_fwd_pool.dGateBuf, intermediate),
        "gate gemm");

    // up_buf = X @ up_proj.T  (same layout)
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            g_fwd_pool.dUpProj, d_model,
            g_fwd_pool.dX, d_model,
            &beta0,
            g_fwd_pool.dUpBuf, intermediate),
        "up gemm");

    // SiLU gate fusion: fused = silu(gate) * up, save sigmoid in cache
    int block = 256;
    int grid  = (int)((N + block - 1) / block);
    swiglu_fuse_kernel<<<grid, block>>>(
        g_fwd_pool.dGateBuf, g_fwd_pool.dUpBuf,
        g_fwd_pool.dFusedBuf, g_fwd_pool.dCacheBuf, N);
    check_cuda(cudaGetLastError(), "swiglu_fuse_kernel launch");

    // Y = fused @ down_proj.T
    // down_proj[d×inter] with lda=inter → cuBLAS sees down_proj.T (col-major [inter×d]).
    // transa=T transposes back to down_proj[d×inter]. fused[seq×inter] with lda=inter → cuBLAS sees fused.T; transb=N uses fused.T.
    // Result Y.T[d×seq] = down_proj[d×inter] @ fused.T[inter×seq]. Written as Y_rm[seq×d]. ✓
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            g_fwd_pool.dDownProj, intermediate,
            g_fwd_pool.dFusedBuf, intermediate,
            &beta0,
            g_fwd_pool.dY, d_model),
        "down gemm");

    check_cuda(cudaDeviceSynchronize(), "sync");

    // Copy outputs to host (all saved for backward pass)
    check_cuda(cudaMemcpy(Y,        g_fwd_pool.dY,        szX,   cudaMemcpyDeviceToHost), "D2H Y");
    check_cuda(cudaMemcpy(gate_buf, g_fwd_pool.dGateBuf,  szBuf, cudaMemcpyDeviceToHost), "D2H gate_buf");
    check_cuda(cudaMemcpy(up_buf,   g_fwd_pool.dUpBuf,    szBuf, cudaMemcpyDeviceToHost), "D2H up_buf");
    check_cuda(cudaMemcpy(fused_buf,g_fwd_pool.dFusedBuf, szBuf, cudaMemcpyDeviceToHost), "D2H fused_buf");
    check_cuda(cudaMemcpy(cache_buf,g_fwd_pool.dCacheBuf, szBuf, cudaMemcpyDeviceToHost), "D2H cache_buf");
    // Note: weights remain in g_fwd_pool on device; backward has its own pool (swiglu_backward.cu)
}

// ── Device-to-device variant ───────────────────────────────────────────
//
// ALL pointers are DEVICE pointers (pre-allocated GpuBuf<f32> from Rust).
// No pool allocation, no H2D/D2H copies, no cudaDeviceSynchronize.
// Caller (gpu_forward.rs) provides caller-managed device buffers and syncs
// via dispatch::cuda_sync() after the full CMS level loop.
// Reuses get_cublas_handle_fwd() — same shared handle, no new allocations.
extern "C" void swiglu_forward_f32_cuda_dd(
    const float* X,           // device: [seq_len × d_model]
    const float* gate_proj,   // device: [intermediate × d_model]  (row-major)
    const float* up_proj,     // device: [intermediate × d_model]  (row-major)
    const float* down_proj,   // device: [d_model × intermediate]  (row-major)
    float* Y,                 // device: [seq_len × d_model]  (output)
    float* gate_buf,          // device: [seq_len × intermediate]  (saved for bwd)
    float* up_buf,            // device: [seq_len × intermediate]  (saved for bwd)
    float* fused_buf,         // device: [seq_len × intermediate]  (saved for bwd)
    float* cache_buf,         // device: [seq_len × intermediate]  (sigmoid, saved for bwd)
    int seq_len,
    int d_model,
    int intermediate)
{
    if (seq_len <= 0 || d_model <= 0 || intermediate <= 0) {
        fprintf(stderr,
                "[NL_Hecate FATAL] swiglu_fwd_dd invalid dims: seq_len=%d d_model=%d intermediate=%d\n",
                seq_len, d_model, intermediate);
        abort();
    }
    cublasHandle_t h = get_cublas_handle_fwd();
    const float alpha1 = 1.0f, beta0 = 0.0f;

    size_t N = (size_t)seq_len * intermediate;

    // gate_buf = X @ gate_proj.T
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            gate_proj, d_model,
            X, d_model,
            &beta0,
            gate_buf, intermediate),
        "gate gemm dd");

    // up_buf = X @ up_proj.T
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate, seq_len, d_model,
            &alpha1,
            up_proj, d_model,
            X, d_model,
            &beta0,
            up_buf, intermediate),
        "up gemm dd");

    // SiLU gate fusion: fused = silu(gate) * up, save sigmoid in cache
    int block = 256;
    size_t grid_sz = (N + (size_t)block - 1) / (size_t)block;
    if (grid_sz > (size_t)INT_MAX) {
        fprintf(stderr, "[NL_Hecate FATAL] swiglu_fwd_dd grid overflow: %zu\n", grid_sz);
        abort();
    }
    int grid = (int)grid_sz;
    swiglu_fuse_kernel<<<grid, block>>>(
        gate_buf, up_buf, fused_buf, cache_buf, N);
    check_cuda(cudaGetLastError(), "swiglu_fuse_kernel dd launch");

    // Y = fused @ down_proj.T
    check_cublas(
        cublasSgemm(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d_model, seq_len, intermediate,
            &alpha1,
            down_proj, intermediate,
            fused_buf, intermediate,
            &beta0,
            Y, d_model),
        "down gemm dd");
    // No cudaDeviceSynchronize — caller syncs via dispatch::cuda_sync()
}
