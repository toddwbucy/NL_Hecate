/// Feature-gated dispatch: Rust reference vs CUDA kernels.
///
/// Without `--features cuda` → Rust reference path.
/// With `--features cuda` → CUDA kernels via FFI (cuda_ffi.rs).
///
/// Both paths produce comparable results (verified by tests).
/// Primary dispatch is compile-time (`#[cfg(feature = "cuda")]`).
/// Runtime override via `force_rust_reference()` for testing/debugging.
///
/// SWA: bf16 storage, f32 compute (FlashAttention-style).
/// Memory rules (Delta, Titans, Hebbian): all fp32 (M must be fp32 per spec).

use std::sync::atomic::{AtomicU8, Ordering};

// ══════════════════════════════════════════════════════════════════════
// Backend enum + GPU detection
// ══════════════════════════════════════════════════════════════════════

/// GPU compute backend.
///
/// Describes which code path actually runs kernels:
/// - `RustReference`: pure Rust (always available, AD-compatible)
/// - `CudaNative`: architecture-specific SASS (sm_86/89/90a)
/// - `CudaPtx`: JIT-compiled PTX for architectures without explicit SASS
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Pure Rust reference implementation (always available, AD-compatible).
    RustReference,
    /// CUDA with architecture-specific SASS (native performance).
    CudaNative,
    /// CUDA with JIT-compiled PTX (forward-compatible fallback).
    CudaPtx,
}

/// Known SM versions with embedded SASS in the fat binary.
/// sm_86: Ampere (A6000, RTX 3090)
/// sm_89: Ada Lovelace (RTX 4090)
/// sm_90: Hopper (H100, H200) — sm_90a SASS with TMA/cp.async support
///        Note: H100/H200 report sm_version=90 at runtime; the CUDA runtime
///        selects sm_90a SASS from the fat binary automatically.
/// sm_100: Blackwell (B100, B200) — covered by compute_90a PTX fallback
const NATIVE_SM_VERSIONS: &[i32] = &[86, 89, 90];

/// Minimum supported SM version (PTX fallback baseline).
const MIN_SM_VERSION: i32 = 86;

/// GPU device information queried from the CUDA runtime.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Device name (e.g., "NVIDIA RTX A6000").
    pub name: String,
    /// Major compute capability (e.g., 8 for sm_86).
    pub compute_major: i32,
    /// Minor compute capability (e.g., 6 for sm_86).
    pub compute_minor: i32,
    /// Combined SM version (major * 10 + minor, e.g., 86).
    pub sm_version: i32,
}

/// Query the current CUDA device properties.
///
/// Returns `Some(GpuInfo)` when CUDA is available and a GPU is present.
/// Returns `None` when compiled without `cuda` feature or no GPU found.
#[cfg(feature = "cuda")]
pub fn detect_gpu() -> Option<GpuInfo> {
    #[repr(C)]
    struct CudaDeviceProp {
        name: [u8; 256],
        // cudaDeviceProp grows across CUDA toolkit versions (900+ bytes as of 12.x).
        // Over-allocate to 4096 bytes for forward-compatibility. We only read the
        // `name` field; compute capability comes from cudaDeviceGetAttribute below.
        _padding: [u8; 4096],
    }

    extern "C" {
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaGetDeviceProperties(prop: *mut CudaDeviceProp, device: i32) -> i32;
    }

    let mut count = 0i32;
    let rc = unsafe { cudaGetDeviceCount(&mut count) };
    if rc != 0 || count == 0 {
        return None;
    }

    let mut prop = CudaDeviceProp {
        name: [0u8; 256],
        _padding: [0u8; 4096],
    };
    let rc = unsafe { cudaGetDeviceProperties(&mut prop, 0) };
    if rc != 0 {
        return None;
    }

    // Extract null-terminated name string.
    let name_end = prop.name.iter().position(|&b| b == 0).unwrap_or(256);
    let name = String::from_utf8_lossy(&prop.name[..name_end]).to_string();

    // Compute capability is at byte offsets matching cudaDeviceProp layout.
    // After `name[256]`, the next fields in cudaDeviceProp are:
    //   totalGlobalMem (size_t = 8 bytes), then various fields...
    //   major is at a known offset. We use a different FFI approach:
    // Actually, cudaDeviceGetAttribute is simpler and more reliable.
    extern "C" {
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    }
    // cudaDevAttrComputeCapabilityMajor = 75
    // cudaDevAttrComputeCapabilityMinor = 76
    let mut major = 0i32;
    let mut minor = 0i32;
    let rc_major = unsafe { cudaDeviceGetAttribute(&mut major, 75, 0) };
    let rc_minor = unsafe { cudaDeviceGetAttribute(&mut minor, 76, 0) };
    if rc_major != 0 || rc_minor != 0 {
        return None;
    }

    Some(GpuInfo {
        name,
        compute_major: major,
        compute_minor: minor,
        sm_version: major * 10 + minor,
    })
}

/// Query the current CUDA device properties.
///
/// Always returns `None` when compiled without `cuda` feature.
#[cfg(not(feature = "cuda"))]
pub fn detect_gpu() -> Option<GpuInfo> {
    None
}

/// Select the appropriate backend based on detected GPU.
///
/// - No GPU → `RustReference`
/// - GPU with native SASS (sm_86/89/90) → `CudaNative`
/// - GPU sm >= 86 but no explicit SASS → `CudaPtx` (JIT fallback)
/// - GPU sm < 86 → `RustReference` (unsupported)
pub fn select_backend(gpu: &Option<GpuInfo>) -> Backend {
    match gpu {
        None => Backend::RustReference,
        Some(info) => {
            if info.sm_version < MIN_SM_VERSION {
                Backend::RustReference
            } else if NATIVE_SM_VERSIONS.contains(&info.sm_version) {
                Backend::CudaNative
            } else {
                Backend::CudaPtx
            }
        }
    }
}

// ── Runtime backend override ────────────────────────────────────────

/// 0 = auto (use CUDA if available), 1 = force Rust reference.
static FORCE_BACKEND: AtomicU8 = AtomicU8::new(0);

/// Force all dispatch functions to use the Rust reference path,
/// even when CUDA is compiled in. Useful for testing and debugging.
pub fn force_rust_reference(force: bool) {
    FORCE_BACKEND.store(if force { 1 } else { 0 }, Ordering::SeqCst);
}

/// Check if the Rust reference path is forced.
pub fn is_rust_forced() -> bool {
    FORCE_BACKEND.load(Ordering::SeqCst) != 0
}

// ══════════════════════════════════════════════════════════════════════
// cuBLAS matmul dispatch
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
const CUBLAS_OP_N: i32 = 0;
#[cfg(feature = "cuda")]
const CUBLAS_OP_T: i32 = 1;

#[cfg(feature = "cuda")]
extern "C" {
    fn cublasCreate_v2(handle: *mut *mut std::ffi::c_void) -> i32;
    #[allow(dead_code)]
    fn cublasDestroy_v2(handle: *mut std::ffi::c_void) -> i32;
    fn cublasSgemm_v2(
        handle: *mut std::ffi::c_void,
        transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: *const f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: *const f32,
        c: *mut f32, ldc: i32,
    ) -> i32;
    fn cublasSaxpy_v2(
        handle: *mut std::ffi::c_void,
        n: i32,
        alpha: *const f32,
        x: *const f32, incx: i32,
        y: *mut f32, incy: i32,
    ) -> i32;
    fn cublasSgemmStridedBatched(
        handle: *mut std::ffi::c_void,
        transa: i32, transb: i32,
        m: i32, n: i32, k: i32,
        alpha: *const f32,
        a: *const f32, lda: i32, stride_a: i64,
        b: *const f32, ldb: i32, stride_b: i64,
        beta: *const f32,
        c: *mut f32, ldc: i32, stride_c: i64,
        batch_count: i32,
    ) -> i32;
}

/// Thread-safe wrapper for cuBLAS handle (raw pointer).
///
/// cuBLAS handles are safe for concurrent sgemm calls — the CUDA runtime
/// serializes kernel launches on the default stream.
#[cfg(feature = "cuda")]
struct SafeCublasHandle(*mut std::ffi::c_void);

#[cfg(feature = "cuda")]
unsafe impl Send for SafeCublasHandle {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SafeCublasHandle {}

#[cfg(feature = "cuda")]
static CUBLAS_HANDLE: std::sync::OnceLock<SafeCublasHandle> = std::sync::OnceLock::new();

/// Get the global cuBLAS handle, creating it on first use (public accessor).
#[cfg(feature = "cuda")]
pub fn cublas_handle_pub() -> *mut std::ffi::c_void {
    cublas_handle()
}

/// Get the global cuBLAS handle, creating it on first use.
#[cfg(feature = "cuda")]
fn cublas_handle() -> *mut std::ffi::c_void {
    CUBLAS_HANDLE.get_or_init(|| {
        let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cublasCreate_v2(&mut handle) };
        assert_eq!(rc, 0, "cublasCreate_v2 failed with error code {rc}");
        SafeCublasHandle(handle)
    }).0
}

/// cuBLAS sgemm: C = alpha * A[m,k] @ B[k,n] + beta * C.
///
/// Uses the row-major→column-major trick: to compute C = A @ B in row-major,
/// call sgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n), which computes
/// C^T = B^T @ A^T in column-major — reading back as row-major gives C = A @ B.
#[cfg(feature = "cuda")]
fn cublas_matmul(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    let d_a = DevBuf::new(m * k);
    let d_b = DevBuf::new(k * n);
    let d_c = DevBuf::new(m * n);

    d_a.copy_from_host(a);
    d_b.copy_from_host(b);
    if beta != 0.0 {
        d_c.copy_from_host(out);
    }

    let alpha_val: f32 = 1.0;
    let beta_val: f32 = beta;

    let rc = unsafe {
        cublasSgemm_v2(
            cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha_val,
            d_b.ptr as *const f32, n as i32,
            d_a.ptr as *const f32, k as i32,
            &beta_val,
            d_c.ptr, n as i32,
        )
    };
    assert_eq!(rc, 0, "cublasSgemm_v2 failed with error code {rc}");

    unsafe {
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after cuBLAS sgemm (error {rc})");
    }

    d_c.copy_to_host(out);
}

/// cuBLAS fused transpose-B sgemm: C = A[m,k] @ B^T where B is stored as [n,k].
///
/// Row-major trick with OP_T on first argument: sgemm(T, N, n, m, k, ..., B, k, A, k, ..., C, n)
/// computes C^T = B @ A^T in column-major, which read as row-major gives C = A @ B^T.
#[cfg(feature = "cuda")]
fn cublas_matmul_transb(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    let d_a = DevBuf::new(m * k);
    let d_b = DevBuf::new(n * k);
    let d_c = DevBuf::new(m * n);

    d_a.copy_from_host(a);
    d_b.copy_from_host(b);
    if beta != 0.0 {
        d_c.copy_from_host(out);
    }

    let alpha_val: f32 = 1.0;
    let beta_val: f32 = beta;

    let rc = unsafe {
        cublasSgemm_v2(
            cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha_val,
            d_b.ptr as *const f32, k as i32,
            d_a.ptr as *const f32, k as i32,
            &beta_val,
            d_c.ptr, n as i32,
        )
    };
    assert_eq!(rc, 0, "cublasSgemm_v2 (transB) failed with error code {rc}");

    unsafe {
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after cuBLAS transB sgemm (error {rc})");
    }

    d_c.copy_to_host(out);
}

/// C[m,n] = A[m,k] @ B[k,n]. cuBLAS on GPU if available, Rust fallback otherwise.
pub fn matmul_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cublas_matmul(a, b, out, m, k, n, 0.0);
            return;
        }
    }
    crate::tensor::matmul_f32(a, b, out, m, k, n);
}

/// C[m,n] += A[m,k] @ B[k,n]. cuBLAS on GPU if available, Rust fallback otherwise.
pub fn matmul_acc_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cublas_matmul(a, b, out, m, k, n, 1.0);
            return;
        }
    }
    crate::tensor::matmul_acc_f32(a, b, out, m, k, n);
}

/// C[m,n] = A[m,k] @ B^T where B is stored as [n,k].
/// Fused transpose-matmul: cuBLAS uses OP_T to avoid a separate transpose allocation.
/// Eliminates the need for `transpose_f32(W) + matmul_f32(X, W_t)` on weight matrices.
pub fn matmul_transb_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cublas_matmul_transb(a, b, out, m, k, n, 0.0);
            return;
        }
    }
    // Fallback: explicit transpose + matmul
    let mut bt = vec![0.0f32; k * n];
    crate::tensor::transpose_f32(b, &mut bt, n, k);
    crate::tensor::matmul_f32(a, &bt, out, m, k, n);
}

// ══════════════════════════════════════════════════════════════════════
// Dispatch functions — SWA + Memory rule inner loops
// ══════════════════════════════════════════════════════════════════════

/// SWA forward dispatch.
///
/// Calls either the Rust reference or CUDA kernel depending on feature gate.
/// Respects `force_rust_reference()` override.
pub fn swa_forward_dispatch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    attn_weights: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            assert!(
                head_dim <= 1024 && head_dim.is_power_of_two(),
                "CUDA SWA kernels require head_dim <= 1024 and power of two, got head_dim={head_dim}"
            );
            cuda_forward(q, k, v, out, attn_weights, seq_len, num_heads, head_dim, window_size);
            return;
        }
    }
    crate::swa::swa_forward(q, k, v, out, attn_weights, seq_len, num_heads, head_dim, window_size);
}

/// SWA backward dispatch.
///
/// Calls either the Rust reference or CUDA kernel depending on feature gate.
/// Respects `force_rust_reference()` override.
/// dQ, dK, dV are NOT zeroed by this function — caller must pre-zero.
pub fn swa_backward_dispatch(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_weights: &[f32],
    d_attn_out: &[f32],
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            assert!(
                head_dim <= 1024 && head_dim.is_power_of_two(),
                "CUDA SWA kernels require head_dim <= 1024 and power of two, got head_dim={head_dim}"
            );
            cuda_backward(q, k, v, attn_weights, d_attn_out, d_q, d_k, d_v,
                          seq_len, num_heads, head_dim, window_size);
            return;
        }
    }
    crate::swa::swa_backward_rust(q, k, v, attn_weights, d_attn_out, d_q, d_k, d_v,
                                   seq_len, num_heads, head_dim, window_size);
}

// ── CUDA dispatch helpers (device memory management) ────────────────

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

#[cfg(feature = "cuda")]
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
#[cfg(feature = "cuda")]
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// ── bf16 conversion helpers ─────────────────────────────────────────
//
// bf16 = bfloat16: 1 sign + 8 exponent + 7 mantissa bits.
// Same exponent range as f32, ~2 decimal digits of precision.
// Stored as u16 on the Rust side, __nv_bfloat16 on the CUDA side.

/// Convert f32 to bf16 (round to nearest even), returning the u16 bits.
#[cfg(feature = "cuda")]
#[inline]
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

/// Convert bf16 bits (u16) back to f32.
#[cfg(feature = "cuda")]
#[inline]
fn bf16_bits_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Convert an f32 slice to a bf16 (u16) vec.
#[cfg(feature = "cuda")]
fn f32_slice_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&x| f32_to_bf16_bits(x)).collect()
}

/// Convert a bf16 (u16) vec to f32, writing into dst.
#[cfg(feature = "cuda")]
fn bf16_vec_to_f32(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = bf16_bits_to_f32(s);
    }
}

// ── RAII device buffer for f32 ──────────────────────────────────────

/// RAII wrapper for device memory allocation (f32 elements).
#[cfg(feature = "cuda")]
struct DevBuf {
    ptr: *mut f32,
    len: usize,
}

#[cfg(feature = "cuda")]
impl DevBuf {
    /// Allocate device memory for `len` f32 elements.
    fn new(len: usize) -> Self {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let bytes = len * std::mem::size_of::<f32>();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        assert_eq!(rc, 0, "cudaMalloc failed with error code {rc}");
        DevBuf { ptr: ptr as *mut f32, len }
    }

    /// Copy host data to device.
    fn copy_from_host(&self, src: &[f32]) {
        assert_eq!(src.len(), self.len);
        let bytes = self.len * std::mem::size_of::<f32>();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy H2D failed with error code {rc}");
    }

    /// Copy device data back to host.
    fn copy_to_host(&self, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.len);
        let bytes = self.len * std::mem::size_of::<f32>();
        let rc = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy D2H failed with error code {rc}");
    }

    /// Zero-initialize device memory.
    #[allow(dead_code)]
    fn zero(&self) {
        let zeros = vec![0.0f32; self.len];
        self.copy_from_host(&zeros);
    }
}

#[cfg(feature = "cuda")]
impl Drop for DevBuf {
    fn drop(&mut self) {
        let rc = unsafe { cudaFree(self.ptr as *mut std::ffi::c_void) };
        debug_assert_eq!(rc, 0, "cudaFree failed with error code {rc}");
    }
}

// ── RAII device buffer for bf16 (u16) ───────────────────────────────

/// RAII wrapper for device memory allocation (bf16/u16 elements).
#[cfg(feature = "cuda")]
struct DevBuf16 {
    ptr: *mut u16,
    len: usize,
}

#[cfg(feature = "cuda")]
impl DevBuf16 {
    /// Allocate device memory for `len` bf16 elements.
    fn new(len: usize) -> Self {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let bytes = len * std::mem::size_of::<u16>();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        assert_eq!(rc, 0, "cudaMalloc failed with error code {rc}");
        DevBuf16 { ptr: ptr as *mut u16, len }
    }

    /// Convert f32 host data to bf16 and copy to device.
    fn copy_from_host_f32(&self, src: &[f32]) {
        assert_eq!(src.len(), self.len);
        let bf16_data = f32_slice_to_bf16(src);
        let bytes = self.len * std::mem::size_of::<u16>();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                bf16_data.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy H2D (bf16) failed with error code {rc}");
    }

    /// Copy bf16 device data back to host as f32.
    fn copy_to_host_f32(&self, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.len);
        let mut bf16_data = vec![0u16; self.len];
        let bytes = self.len * std::mem::size_of::<u16>();
        let rc = unsafe {
            cudaMemcpy(
                bf16_data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy D2H (bf16) failed with error code {rc}");
        bf16_vec_to_f32(&bf16_data, dst);
    }

    /// Zero-initialize device memory.
    fn zero(&self) {
        let zeros = vec![0u16; self.len];
        let bytes = self.len * std::mem::size_of::<u16>();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                zeros.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy H2D zero (bf16) failed with error code {rc}");
    }
}

#[cfg(feature = "cuda")]
impl Drop for DevBuf16 {
    fn drop(&mut self) {
        let rc = unsafe { cudaFree(self.ptr as *mut std::ffi::c_void) };
        debug_assert_eq!(rc, 0, "cudaFree failed with error code {rc}");
    }
}

// ── CUDA forward dispatch ───────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cuda_forward(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32], attn_weights: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    let total = seq_len * num_heads * head_dim;
    let aw_total = num_heads * seq_len * window_size;

    // bf16 device buffers for Q/K/V/out/attn_weights
    let d_q = DevBuf16::new(total);
    let d_k = DevBuf16::new(total);
    let d_v = DevBuf16::new(total);
    let d_out = DevBuf16::new(total);
    let d_aw = DevBuf16::new(aw_total);

    // Convert f32 host → bf16 device
    d_q.copy_from_host_f32(q);
    d_k.copy_from_host_f32(k);
    d_v.copy_from_host_f32(v);
    d_out.zero();
    d_aw.zero();

    unsafe {
        crate::cuda_ffi::swa_forward_f32_cuda(
            d_q.ptr, d_k.ptr, d_v.ptr,
            d_out.ptr, d_aw.ptr,
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32, 1,
            0, // n_persistent=0 for raw-slice test helper
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after SWA forward kernel (error code {rc})");
    }

    // Convert bf16 device → f32 host
    d_out.copy_to_host_f32(out);
    d_aw.copy_to_host_f32(attn_weights);
}

// ── CUDA backward dispatch ──────────────────────────────────────────

#[cfg(feature = "cuda")]
fn cuda_backward(
    q: &[f32], k: &[f32], v: &[f32],
    attn_weights: &[f32], d_attn_out: &[f32],
    dq_host: &mut [f32], dk_host: &mut [f32], dv_host: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    let total = seq_len * num_heads * head_dim;
    let aw_total = num_heads * seq_len * window_size;

    // bf16 device buffers for Q/K/V/attn_weights
    let d_q = DevBuf16::new(total);
    let d_k = DevBuf16::new(total);
    let d_v = DevBuf16::new(total);
    let d_aw = DevBuf16::new(aw_total);

    // f32 device buffers for gradients
    let d_dao = DevBuf::new(total);
    let d_dq = DevBuf::new(total);
    let d_dk = DevBuf::new(total);
    let d_dv = DevBuf::new(total);

    // Convert f32 host → bf16 device for inputs
    d_q.copy_from_host_f32(q);
    d_k.copy_from_host_f32(k);
    d_v.copy_from_host_f32(v);
    d_aw.copy_from_host_f32(attn_weights);

    // f32 copies for gradients
    d_dao.copy_from_host(d_attn_out);
    d_dq.copy_from_host(dq_host);
    d_dk.copy_from_host(dk_host);
    d_dv.copy_from_host(dv_host);

    unsafe {
        crate::cuda_ffi::swa_backward_f32_cuda(
            d_q.ptr, d_k.ptr, d_v.ptr,
            d_aw.ptr, d_dao.ptr as *const f32,
            d_dq.ptr, d_dk.ptr, d_dv.ptr,
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32, 1,
            0, // n_persistent=0 for raw-slice test helper
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after SWA backward kernel (error code {rc})");
    }

    d_dq.copy_to_host(dq_host);
    d_dk.copy_to_host(dk_host);
    d_dv.copy_to_host(dv_host);
}

// ══════════════════════════════════════════════════════════════════════
// Memory rule CUDA dispatch — all fp32 (no bf16 conversion needed)
// ══════════════════════════════════════════════════════════════════════

// ── Delta Rule dispatch ─────────────────────────────────────────────

/// Delta Rule forward inner loop dispatch.
///
/// Takes pre-computed projections and gates (from Rust), runs the sequential
/// M recurrence in either Rust or CUDA depending on feature gate.
///
/// NOTE: This dispatch path uses L2 attentional bias only. Non-L2 biases
/// (L1, Lp) are handled by the MemoryRule trait path (DeltaRule::step).
/// The dispatch path is used for GPU-resident forward/backward and gradient
/// checkpointing; extending it to support configurable l_p requires adding
/// bias parameters through the entire dispatch chain including CUDA kernels.
///
/// Returns (m_states, y) where m_states is [(seq_len+1)*d*d] and y is [seq_len*d].
pub fn delta_forward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    m_initial: &[f32],
    m_states: &mut [f32],
    y: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
    m_norm_max: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_delta_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                               m_states, y, seq_len, d, error_clip, m_norm_max);
            return;
        }
    }
    rust_delta_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                       m_states, y, seq_len, d, error_clip, m_norm_max);
}

/// Delta Rule backward inner loop dispatch.
///
/// Returns gradients on k_mem, v_mem, q_mem, alpha, theta, m_initial.
pub fn delta_backward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    m_states: &[f32],
    d_y: &[f32],
    d_k_mem: &mut [f32],
    d_v_mem: &mut [f32],
    d_q_mem: &mut [f32],
    d_alpha: &mut [f32],
    d_theta: &mut [f32],
    d_m_initial: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_delta_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                                d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                                seq_len, d, error_clip);
            return;
        }
    }
    rust_delta_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                        seq_len, d, error_clip);
}

// ── Titans LMM dispatch ─────────────────────────────────────────────

/// Titans LMM forward inner loop dispatch.
pub fn titans_forward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_initial: &[f32],
    s_initial: &[f32],
    m_states: &mut [f32],
    s_states: &mut [f32],
    y: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
    m_norm_max: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_titans_forward(k_mem, v_mem, q_mem, alpha, theta, eta,
                                m_initial, s_initial, m_states, s_states, y,
                                seq_len, d, error_clip, m_norm_max);
            return;
        }
    }
    rust_titans_forward(k_mem, v_mem, q_mem, alpha, theta, eta,
                        m_initial, s_initial, m_states, s_states, y,
                        seq_len, d, error_clip, m_norm_max);
}

/// Titans LMM backward inner loop dispatch.
pub fn titans_backward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_states: &[f32],
    s_states: &[f32],
    d_y: &[f32],
    d_k_mem: &mut [f32],
    d_v_mem: &mut [f32],
    d_q_mem: &mut [f32],
    d_alpha: &mut [f32],
    d_theta: &mut [f32],
    d_eta: &mut [f32],
    d_m_initial: &mut [f32],
    d_s_initial: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_titans_backward(k_mem, v_mem, q_mem, alpha, theta, eta,
                                 m_states, s_states, d_y,
                                 d_k_mem, d_v_mem, d_q_mem,
                                 d_alpha, d_theta, d_eta,
                                 d_m_initial, d_s_initial,
                                 seq_len, d, error_clip);
            return;
        }
    }
    rust_titans_backward(k_mem, v_mem, q_mem, alpha, theta, eta,
                         m_states, s_states, d_y,
                         d_k_mem, d_v_mem, d_q_mem,
                         d_alpha, d_theta, d_eta,
                         d_m_initial, d_s_initial,
                         seq_len, d, error_clip);
}

// ── TitansLMM MLP Memory forward dispatch (Spec 75) ────────────────

/// TitansLMM MLP memory forward inner loop — **test-only host↔device helper**.
///
/// Copies host buffers to device, launches `titans_mlp_forward_f32_cuda`,
/// synchronizes, and copies results back. A backward helper
/// (`titans_mlp_backward_cuda`) also exists below. The production GPU path
/// is wired through `gpu_forward.rs` / `gpu_backward.rs` using the `_dd`
/// device-to-device variants.
///
/// Packed L_M=2 buffer: W1[d_h,d], b1[d_h], W2[d,d_h], b2[d].
/// activation: 0=GELU, 1=SiLU, 2=ReLU.
#[cfg(feature = "cuda")]
pub fn titans_mlp_forward_cuda(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_initial: &[f32],
    s_initial: &[f32],
    m_states: &mut [f32],
    s_states: &mut [f32],
    y: &mut [f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
    batch_size: usize,
    activation: i32,
    m_norm_max: f32,
) {
    let state_size = d_hidden * d + d_hidden + d * d_hidden + d;
    let input_total = batch_size * seq_len * d;
    let gate_total = batch_size * seq_len;
    let init_total = batch_size * state_size;
    let states_total = batch_size * (seq_len + 1) * state_size;

    let dev_km = DevBuf::new(input_total);
    let dev_vm = DevBuf::new(input_total);
    let dev_qm = DevBuf::new(input_total);
    let dev_alpha = DevBuf::new(gate_total);
    let dev_theta = DevBuf::new(gate_total);
    let dev_eta = DevBuf::new(gate_total);
    let dev_minit = DevBuf::new(init_total);
    let dev_sinit = DevBuf::new(init_total);
    let dev_mstates = DevBuf::new(states_total);
    let dev_sstates = DevBuf::new(states_total);
    let dev_y = DevBuf::new(input_total);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_eta.copy_from_host(eta);
    dev_minit.copy_from_host(m_initial);
    dev_sinit.copy_from_host(s_initial);
    dev_mstates.zero();
    dev_sstates.zero();
    dev_y.zero();

    let input_stride = seq_len * d;
    let m_stride = (seq_len + 1) * state_size;

    unsafe {
        crate::cuda_ffi::titans_mlp_forward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_eta.ptr,
            dev_minit.ptr, dev_sinit.ptr,
            dev_mstates.ptr, dev_sstates.ptr, dev_y.ptr,
            seq_len as i32, d as i32, d_hidden as i32,
            batch_size as i32,
            input_stride as i32, m_stride as i32,
            activation, m_norm_max,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after titans_mlp forward (error {rc})");
    }

    dev_mstates.copy_to_host(m_states);
    dev_sstates.copy_to_host(s_states);
    dev_y.copy_to_host(y);
}

/// TitansLMM MLP memory backward inner loop — **test-only host↔device helper**.
///
/// Copies host buffers to device, launches `titans_mlp_backward_f32_cuda`,
/// synchronizes, and copies results back. The production GPU path will call
/// the FFI directly from `gpu_backward.rs` with device buffers.
///
/// Requires m_states/s_states from forward pass (full trajectory).
/// activation: 0=GELU, 1=SiLU, 2=ReLU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_mlp_backward_cuda(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_states: &[f32],
    s_states: &[f32],
    d_y: &[f32],
    d_k_mem: &mut [f32],
    d_v_mem: &mut [f32],
    d_q_mem: &mut [f32],
    d_alpha: &mut [f32],
    d_theta: &mut [f32],
    d_eta: &mut [f32],
    d_m_initial: &mut [f32],
    d_s_initial: &mut [f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
    batch_size: usize,
    activation: i32,
    m_norm_max: f32,
) {
    let state_size = d_hidden * d + d_hidden + d * d_hidden + d;
    let input_total = batch_size * seq_len * d;
    let gate_total = batch_size * seq_len;
    let init_total = batch_size * state_size;
    let states_total = batch_size * (seq_len + 1) * state_size;

    // Upload inputs
    let dev_km = DevBuf::new(input_total);
    let dev_vm = DevBuf::new(input_total);
    let dev_qm = DevBuf::new(input_total);
    let dev_alpha = DevBuf::new(gate_total);
    let dev_theta = DevBuf::new(gate_total);
    let dev_eta = DevBuf::new(gate_total);
    let dev_mstates = DevBuf::new(states_total);
    let dev_sstates = DevBuf::new(states_total);
    let dev_dy = DevBuf::new(input_total);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_eta.copy_from_host(eta);
    dev_mstates.copy_from_host(m_states);
    dev_sstates.copy_from_host(s_states);
    dev_dy.copy_from_host(d_y);

    // Allocate outputs (zeroed by kernel or atomicAdd)
    let dev_dkm = DevBuf::new(input_total);
    let dev_dvm = DevBuf::new(input_total);
    let dev_dqm = DevBuf::new(input_total);
    let dev_dalpha = DevBuf::new(gate_total);
    let dev_dtheta = DevBuf::new(gate_total);
    let dev_deta = DevBuf::new(gate_total);
    let dev_dm_init = DevBuf::new(init_total);
    let dev_ds_init = DevBuf::new(init_total);

    dev_dkm.zero();
    dev_dvm.zero();
    dev_dqm.zero();
    dev_dalpha.zero();
    dev_dtheta.zero();
    dev_deta.zero();
    dev_dm_init.zero();
    dev_ds_init.zero();

    let input_stride = seq_len * d;
    let m_stride = (seq_len + 1) * state_size;

    unsafe {
        crate::cuda_ffi::titans_mlp_backward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_eta.ptr,
            dev_mstates.ptr, dev_sstates.ptr, dev_dy.ptr,
            dev_dkm.ptr, dev_dvm.ptr, dev_dqm.ptr,
            dev_dalpha.ptr, dev_dtheta.ptr, dev_deta.ptr,
            dev_dm_init.ptr, dev_ds_init.ptr,
            seq_len as i32, d as i32, d_hidden as i32,
            batch_size as i32,
            input_stride as i32, m_stride as i32,
            activation, m_norm_max,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after titans_mlp backward (error {rc})");
    }

    dev_dkm.copy_to_host(d_k_mem);
    dev_dvm.copy_to_host(d_v_mem);
    dev_dqm.copy_to_host(d_q_mem);
    dev_dalpha.copy_to_host(d_alpha);
    dev_dtheta.copy_to_host(d_theta);
    dev_deta.copy_to_host(d_eta);
    dev_dm_init.copy_to_host(d_m_initial);
    dev_ds_init.copy_to_host(d_s_initial);
}

// ── Hebbian Rule dispatch ───────────────────────────────────────────

/// Hebbian Rule forward inner loop dispatch.
pub fn hebbian_forward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    m_initial: &[f32],
    m_states: &mut [f32],
    y: &mut [f32],
    seq_len: usize,
    d: usize,
    m_norm_max: f32,
) {
    #[cfg(feature = "cuda")]
    {
        // Gate CUDA path when active clamp requested — see delta_forward_dispatch.
        let clamp_enabled = m_norm_max > 0.0 && m_norm_max < f32::MAX;
        if !is_rust_forced() && !clamp_enabled {
            cuda_hebbian_forward(k_mem, v_mem, q_mem, alpha, m_initial,
                                 m_states, y, seq_len, d);
            return;
        }
    }
    rust_hebbian_forward(k_mem, v_mem, q_mem, alpha, m_initial,
                         m_states, y, seq_len, d, m_norm_max);
}

/// Hebbian Rule backward inner loop dispatch.
pub fn hebbian_backward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    m_states: &[f32],
    d_y: &[f32],
    d_k_mem: &mut [f32],
    d_v_mem: &mut [f32],
    d_q_mem: &mut [f32],
    d_alpha: &mut [f32],
    d_m_initial: &mut [f32],
    seq_len: usize,
    d: usize,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_hebbian_backward(k_mem, v_mem, q_mem, alpha, m_states, d_y,
                                  d_k_mem, d_v_mem, d_q_mem, d_alpha, d_m_initial,
                                  seq_len, d);
            return;
        }
    }
    rust_hebbian_backward(k_mem, v_mem, q_mem, alpha, m_states, d_y,
                          d_k_mem, d_v_mem, d_q_mem, d_alpha, d_m_initial,
                          seq_len, d);
}

// ── DGD (Delta Gradient Descent) dispatch ─────────────────────────

/// DGD forward inner loop dispatch.
///
/// Identical recurrence to Delta Rule (L2 attentional bias only), but
/// dispatches through separate DGD CUDA kernels to allow future
/// bias-agnostic divergence (CS-33).
///
/// Note: `m_norm_max` is intentionally omitted. DGD shares recurrence with
/// Delta Rule at L2 bias; if DGD behavior diverges in the future, revisit
/// this API at that point. For now, clamping is handled by the standalone
/// `m_norm_clamp_f32_cuda` called once per level after the forward pass.
///
/// Returns (m_states, y) where m_states is [(seq_len+1)*d*d] and y is [seq_len*d].
pub fn dgd_forward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    m_initial: &[f32],
    m_states: &mut [f32],
    y: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_dgd_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                             m_states, y, seq_len, d, error_clip);
            return;
        }
    }
    // DGD math is identical to Delta Rule at L2 bias
    rust_delta_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                       m_states, y, seq_len, d, error_clip, f32::MAX);
}

/// DGD backward inner loop dispatch.
///
/// Returns gradients on k_mem, v_mem, q_mem, alpha, theta, m_initial.
pub fn dgd_backward_dispatch(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    m_states: &[f32],
    d_y: &[f32],
    d_k_mem: &mut [f32],
    d_v_mem: &mut [f32],
    d_q_mem: &mut [f32],
    d_alpha: &mut [f32],
    d_theta: &mut [f32],
    d_m_initial: &mut [f32],
    seq_len: usize,
    d: usize,
    error_clip: f32,
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_dgd_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                              d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                              seq_len, d, error_clip);
            return;
        }
    }
    // DGD math is identical to Delta Rule at L2 bias
    rust_delta_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                        seq_len, d, error_clip);
}

// ── Rust reference inner loops ──────────────────────────────────────
// Always compiled. Used as fallback when CUDA is absent or force_rust_reference() is set.

/// Clamp M-state slice to Frobenius norm ceiling (straight-through).
///
/// Straight-through: the clamp Jacobian is treated as identity during backward,
/// which is the standard practice for gradient clipping (same as CS-39/CS-44).
/// No-op when m_norm_max is 0.0 or f32::MAX (disabled).
#[inline]
/// Clip error vector in-place: if ‖error‖₂ > clip, rescale to clip.
/// Matches CUDA error_clip_inplace. No-op when clip <= 0.
fn clip_error_l2(error: &mut [f32], clip: f32) {
    if clip <= 0.0 { return; }
    let norm_sq: f32 = error.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > clip {
        let scale = clip / norm;
        for x in error.iter_mut() { *x *= scale; }
    }
}

fn clamp_m_norm(slice: &mut [f32], m_norm_max: f32) {
    if m_norm_max > 0.0 && m_norm_max < f32::MAX {
        let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > m_norm_max {
            let scale = m_norm_max / norm;
            for x in slice.iter_mut() { *x *= scale; }
        }
    }
}

/// Rust reference Delta Rule forward inner loop.
fn rust_delta_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32, m_norm_max: f32,
) {
    let dd = d * d;
    m_states[..dd].copy_from_slice(m_initial);
    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        // prediction = M_t @ k_t
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        // error = prediction - v
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }
        clip_error_l2(&mut error, error_clip);

        // M update
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] - theta_t * error[i] * k_t[j];
            }
        }

        clamp_m_norm(&mut m_states[m_next..m_next + dd], m_norm_max);

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
}

/// Rust reference Delta Rule backward inner loop.
fn rust_delta_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_theta: &mut [f32], d_m_initial: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let mut d_m = vec![0.0f32; dd];

    for t in (0..seq_len).rev() {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];
        let m_t = &m_states[t * dd..(t + 1) * dd];
        let m_next = &m_states[(t + 1) * dd..(t + 2) * dd];
        let alpha_t = alpha[t];
        let theta_t = theta[t];

        // d_M += outer(d_y_t, q_t)
        for i in 0..d {
            for j in 0..d { d_m[i * d + j] += d_y_t[i] * q_t[j]; }
        }

        // d_q_t = M_{t+1}^T @ d_y_t
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_next[i * d + j] * d_y_t[i]; }
            d_q_mem[t * d + j] = sum;
        }

        // Recompute prediction and error (with same clip as forward)
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_t[i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }
        clip_error_l2(&mut error, error_clip);

        // d_alpha, d_theta (reductions)
        let mut d_alpha_sum = 0.0f32;
        let mut d_theta_sum = 0.0f32;
        for i in 0..d {
            for j in 0..d {
                d_alpha_sum += m_t[i * d + j] * d_m[i * d + j];
                d_theta_sum += error[i] * k_t[j] * d_m[i * d + j];
            }
        }
        d_alpha[t] = -d_alpha_sum;
        d_theta[t] = -d_theta_sum;

        // d_error, d_k contributions
        let mut d_err = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += (-theta_t * d_m[i * d + j]) * k_t[j]; }
            d_err[i] = sum;
        }

        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += (-theta_t * d_m[i * d + j]) * error[i]; }
            d_k_mem[t * d + j] = sum;
        }

        // prediction = M @ k backward → d_k, d_M
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_t[i * d + j] * d_err[i]; }
            d_k_mem[t * d + j] += sum;
        }

        // d_v = -d_error
        for i in 0..d { d_v_mem[t * d + i] = -d_err[i]; }

        // Propagate d_M backward
        let retention = 1.0 - alpha_t;
        let mut d_m_prev = vec![0.0f32; dd];
        for i in 0..d {
            for j in 0..d {
                d_m_prev[i * d + j] = retention * d_m[i * d + j] + d_err[i] * k_t[j];
            }
        }
        d_m = d_m_prev;
    }

    d_m_initial.copy_from_slice(&d_m);
}

/// Rust reference Titans forward inner loop.
fn rust_titans_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], eta: &[f32],
    m_initial: &[f32], s_initial: &[f32],
    m_states: &mut [f32], s_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32, m_norm_max: f32,
) {
    let dd = d * d;
    m_states[..dd].copy_from_slice(m_initial);
    s_states[..dd].copy_from_slice(s_initial);

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let eta_t = eta[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;
        let s_t = t * dd;
        let s_next = (t + 1) * dd;

        // prediction = M_t @ k_t; error = prediction - v_t
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }
        clip_error_l2(&mut error, error_clip);

        // S_{t+1} = eta_t * S_t - theta_t * outer(error, k)
        for i in 0..d {
            for j in 0..d {
                s_states[s_next + i * d + j] =
                    eta_t * s_states[s_t + i * d + j] - theta_t * error[i] * k_t[j];
            }
        }

        // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}
        let retention = 1.0 - alpha_t;
        for i in 0..dd {
            m_states[m_next + i] = retention * m_states[m_t + i] + s_states[s_next + i];
        }

        clamp_m_norm(&mut m_states[m_next..m_next + dd], m_norm_max);

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
}

/// Rust reference Titans backward inner loop.
#[allow(clippy::too_many_arguments)]
fn rust_titans_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], eta: &[f32],
    m_states: &[f32], s_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_theta: &mut [f32], d_eta: &mut [f32],
    d_m_initial: &mut [f32], d_s_initial: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let mut d_m = vec![0.0f32; dd];
    let mut d_s = vec![0.0f32; dd];

    for t in (0..seq_len).rev() {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];
        let m_t_off = t * dd;
        let m_next_off = (t + 1) * dd;
        let s_t_off = t * dd;
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let eta_t = eta[t];

        // d_M += outer(d_y_t, q_t)
        for i in 0..d {
            for j in 0..d { d_m[i * d + j] += d_y_t[i] * q_t[j]; }
        }

        // d_q_t = M_{t+1}^T @ d_y_t
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_states[m_next_off + i * d + j] * d_y_t[i]; }
            d_q_mem[t * d + j] = sum;
        }

        // M_{t+1} = (1-alpha) * M_t + S_{t+1} backward
        for i in 0..dd { d_s[i] += d_m[i]; }

        let mut d_alpha_sum = 0.0f32;
        for i in 0..dd { d_alpha_sum += m_states[m_t_off + i] * d_m[i]; }
        d_alpha[t] = -d_alpha_sum;

        let retention = 1.0 - alpha_t;
        let mut d_m_prev = vec![0.0f32; dd];
        for i in 0..dd { d_m_prev[i] = retention * d_m[i]; }

        // S_{t+1} = eta * S_t - theta * grad backward
        let mut d_eta_sum = 0.0f32;
        for i in 0..dd { d_eta_sum += s_states[s_t_off + i] * d_s[i]; }
        d_eta[t] = d_eta_sum;

        // Recompute prediction/error (with same clip as forward)
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t_off + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }
        clip_error_l2(&mut error, error_clip);

        let mut d_theta_sum = 0.0f32;
        for i in 0..d {
            for j in 0..d { d_theta_sum += error[i] * k_t[j] * d_s[i * d + j]; }
        }
        d_theta[t] = -d_theta_sum;

        // d_grad = -theta * d_S; d_error, d_k
        let mut d_err = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += (-theta_t * d_s[i * d + j]) * k_t[j]; }
            d_err[i] = sum;
        }

        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += (-theta_t * d_s[i * d + j]) * error[i]; }
            d_k_mem[t * d + j] = sum;
        }

        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_states[m_t_off + i * d + j] * d_err[i]; }
            d_k_mem[t * d + j] += sum;
        }

        for i in 0..d { d_v_mem[t * d + i] = -d_err[i]; }

        // d_S_prev = eta * d_S
        let mut d_s_prev = vec![0.0f32; dd];
        for i in 0..dd { d_s_prev[i] = eta_t * d_s[i]; }

        // Propagate d_M backward: add prediction chain
        for i in 0..d {
            for j in 0..d {
                d_m_prev[i * d + j] += d_err[i] * k_t[j];
            }
        }

        d_m = d_m_prev;
        d_s = d_s_prev;
    }

    d_m_initial.copy_from_slice(&d_m);
    d_s_initial.copy_from_slice(&d_s);
}

/// Rust reference Hebbian forward inner loop.
fn rust_hebbian_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, m_norm_max: f32,
) {
    let dd = d * d;
    m_states[..dd].copy_from_slice(m_initial);

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let alpha_t = alpha[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        // M_{t+1} = (1-alpha) * M_t + outer(v, k)
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] + v_t[i] * k_t[j];
            }
        }

        clamp_m_norm(&mut m_states[m_next..m_next + dd], m_norm_max);

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
}

/// Rust reference Hebbian backward inner loop.
fn rust_hebbian_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], m_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_m_initial: &mut [f32],
    seq_len: usize, d: usize,
) {
    let dd = d * d;
    let mut d_m = vec![0.0f32; dd];

    for t in (0..seq_len).rev() {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];
        let m_t_off = t * dd;
        let m_next_off = (t + 1) * dd;
        let alpha_t = alpha[t];

        // d_M += outer(d_y_t, q_t)
        for i in 0..d {
            for j in 0..d { d_m[i * d + j] += d_y_t[i] * q_t[j]; }
        }

        // d_q_t = M_{t+1}^T @ d_y_t
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_states[m_next_off + i * d + j] * d_y_t[i]; }
            d_q_mem[t * d + j] = sum;
        }

        // d_alpha = -frobenius_dot(d_M, M_t)
        let mut d_alpha_sum = 0.0f32;
        for i in 0..dd { d_alpha_sum += m_states[m_t_off + i] * d_m[i]; }
        d_alpha[t] = -d_alpha_sum;

        // d_v from outer product: d_v[i] = sum_j d_M[i,j] * k_t[j]
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += d_m[i * d + j] * k_t[j]; }
            d_v_mem[t * d + i] = sum;
        }

        // d_k from outer product: d_k[j] = sum_i d_M[i,j] * v_t[i]
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += d_m[i * d + j] * v_t[i]; }
            d_k_mem[t * d + j] = sum;
        }

        // d_M_prev = (1-alpha) * d_M
        let retention = 1.0 - alpha_t;
        for i in 0..dd { d_m[i] = retention * d_m[i]; }
    }

    d_m_initial.copy_from_slice(&d_m);
}

// ── CUDA memory rule dispatch helpers ────────────────────────────────

#[cfg(feature = "cuda")]
fn cuda_delta_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_minit = DevBuf::new(dd);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_y = DevBuf::new(seq_len * d);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_minit.copy_from_host(m_initial);
    dev_mstates.zero();
    dev_y.zero();

    unsafe {
        crate::cuda_ffi::delta_forward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_minit.ptr,
            dev_mstates.ptr, dev_y.ptr,
            seq_len as i32, d as i32, 1,
            seq_len as i32, (d * d) as i32, error_clip,
            m_norm_max,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after delta forward (error {rc})");
    }

    dev_mstates.copy_to_host(m_states);
    dev_y.copy_to_host(y);
}

#[cfg(feature = "cuda")]
fn cuda_delta_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_theta: &mut [f32], d_m_initial: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_dy = DevBuf::new(seq_len * d);
    let dev_dkm = DevBuf::new(seq_len * d);
    let dev_dvm = DevBuf::new(seq_len * d);
    let dev_dqm = DevBuf::new(seq_len * d);
    let dev_dalpha = DevBuf::new(seq_len);
    let dev_dtheta = DevBuf::new(seq_len);
    let dev_dm_init = DevBuf::new(dd);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_mstates.copy_from_host(m_states);
    dev_dy.copy_from_host(d_y);
    dev_dkm.zero();
    dev_dvm.zero();
    dev_dqm.zero();
    dev_dalpha.zero();
    dev_dtheta.zero();
    dev_dm_init.zero();

    unsafe {
        crate::cuda_ffi::delta_backward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_mstates.ptr,
            dev_dy.ptr as *const f32,
            dev_dkm.ptr, dev_dvm.ptr, dev_dqm.ptr,
            dev_dalpha.ptr, dev_dtheta.ptr, dev_dm_init.ptr,
            seq_len as i32, d as i32, 1, error_clip,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after delta backward (error {rc})");
    }

    dev_dkm.copy_to_host(d_k_mem);
    dev_dvm.copy_to_host(d_v_mem);
    dev_dqm.copy_to_host(d_q_mem);
    dev_dalpha.copy_to_host(d_alpha);
    dev_dtheta.copy_to_host(d_theta);
    dev_dm_init.copy_to_host(d_m_initial);
}

#[cfg(feature = "cuda")]
fn cuda_titans_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], eta: &[f32],
    m_initial: &[f32], s_initial: &[f32],
    m_states: &mut [f32], s_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_eta = DevBuf::new(seq_len);
    let dev_minit = DevBuf::new(dd);
    let dev_sinit = DevBuf::new(dd);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_sstates = DevBuf::new((seq_len + 1) * dd);
    let dev_y = DevBuf::new(seq_len * d);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_eta.copy_from_host(eta);
    dev_minit.copy_from_host(m_initial);
    dev_sinit.copy_from_host(s_initial);
    dev_mstates.zero();
    dev_sstates.zero();
    dev_y.zero();

    unsafe {
        crate::cuda_ffi::titans_forward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_eta.ptr,
            dev_minit.ptr, dev_sinit.ptr,
            dev_mstates.ptr, dev_sstates.ptr, dev_y.ptr,
            seq_len as i32, d as i32, 1,
            seq_len as i32, (d * d) as i32, error_clip,
            m_norm_max,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after titans forward (error {rc})");
    }

    dev_mstates.copy_to_host(m_states);
    dev_sstates.copy_to_host(s_states);
    dev_y.copy_to_host(y);
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn cuda_titans_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], eta: &[f32],
    m_states: &[f32], s_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_theta: &mut [f32], d_eta: &mut [f32],
    d_m_initial: &mut [f32], d_s_initial: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_eta = DevBuf::new(seq_len);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_sstates = DevBuf::new((seq_len + 1) * dd);
    let dev_dy = DevBuf::new(seq_len * d);
    let dev_dkm = DevBuf::new(seq_len * d);
    let dev_dvm = DevBuf::new(seq_len * d);
    let dev_dqm = DevBuf::new(seq_len * d);
    let dev_dalpha = DevBuf::new(seq_len);
    let dev_dtheta = DevBuf::new(seq_len);
    let dev_deta = DevBuf::new(seq_len);
    let dev_dm_init = DevBuf::new(dd);
    let dev_ds_init = DevBuf::new(dd);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_eta.copy_from_host(eta);
    dev_mstates.copy_from_host(m_states);
    dev_sstates.copy_from_host(s_states);
    dev_dy.copy_from_host(d_y);
    dev_dkm.zero();
    dev_dvm.zero();
    dev_dqm.zero();
    dev_dalpha.zero();
    dev_dtheta.zero();
    dev_deta.zero();
    dev_dm_init.zero();
    dev_ds_init.zero();

    unsafe {
        crate::cuda_ffi::titans_backward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_eta.ptr,
            dev_mstates.ptr, dev_sstates.ptr,
            dev_dy.ptr as *const f32,
            dev_dkm.ptr, dev_dvm.ptr, dev_dqm.ptr,
            dev_dalpha.ptr, dev_dtheta.ptr, dev_deta.ptr,
            dev_dm_init.ptr, dev_ds_init.ptr,
            seq_len as i32, d as i32, 1, error_clip,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after titans backward (error {rc})");
    }

    dev_dkm.copy_to_host(d_k_mem);
    dev_dvm.copy_to_host(d_v_mem);
    dev_dqm.copy_to_host(d_q_mem);
    dev_dalpha.copy_to_host(d_alpha);
    dev_dtheta.copy_to_host(d_theta);
    dev_deta.copy_to_host(d_eta);
    dev_dm_init.copy_to_host(d_m_initial);
    dev_ds_init.copy_to_host(d_s_initial);
}

#[cfg(feature = "cuda")]
fn cuda_hebbian_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_minit = DevBuf::new(dd);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_y = DevBuf::new(seq_len * d);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_minit.copy_from_host(m_initial);
    dev_mstates.zero();
    dev_y.zero();

    unsafe {
        crate::cuda_ffi::hebbian_forward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_minit.ptr,
            dev_mstates.ptr, dev_y.ptr,
            seq_len as i32, d as i32, 1,
            seq_len as i32, (d * d) as i32,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after hebbian forward (error {rc})");
    }

    dev_mstates.copy_to_host(m_states);
    dev_y.copy_to_host(y);
}

#[cfg(feature = "cuda")]
fn cuda_hebbian_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], m_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_m_initial: &mut [f32],
    seq_len: usize, d: usize,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_dy = DevBuf::new(seq_len * d);
    let dev_dkm = DevBuf::new(seq_len * d);
    let dev_dvm = DevBuf::new(seq_len * d);
    let dev_dqm = DevBuf::new(seq_len * d);
    let dev_dalpha = DevBuf::new(seq_len);
    let dev_dm_init = DevBuf::new(dd);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_mstates.copy_from_host(m_states);
    dev_dy.copy_from_host(d_y);
    dev_dkm.zero();
    dev_dvm.zero();
    dev_dqm.zero();
    dev_dalpha.zero();
    dev_dm_init.zero();

    unsafe {
        crate::cuda_ffi::hebbian_backward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_mstates.ptr,
            dev_dy.ptr as *const f32,
            dev_dkm.ptr, dev_dvm.ptr, dev_dqm.ptr,
            dev_dalpha.ptr, dev_dm_init.ptr,
            seq_len as i32, d as i32,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after hebbian backward (error {rc})");
    }

    dev_dkm.copy_to_host(d_k_mem);
    dev_dvm.copy_to_host(d_v_mem);
    dev_dqm.copy_to_host(d_q_mem);
    dev_dalpha.copy_to_host(d_alpha);
    dev_dm_init.copy_to_host(d_m_initial);
}

#[cfg(feature = "cuda")]
fn cuda_dgd_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_minit = DevBuf::new(dd);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_y = DevBuf::new(seq_len * d);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_minit.copy_from_host(m_initial);
    dev_mstates.zero();
    dev_y.zero();

    unsafe {
        crate::cuda_ffi::dgd_forward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_minit.ptr,
            dev_mstates.ptr, dev_y.ptr,
            seq_len as i32, d as i32, 1,
            seq_len as i32, (d * d) as i32, error_clip,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after dgd forward (error {rc})");
    }

    dev_mstates.copy_to_host(m_states);
    dev_y.copy_to_host(y);
}

#[cfg(feature = "cuda")]
fn cuda_dgd_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_states: &[f32], d_y: &[f32],
    d_k_mem: &mut [f32], d_v_mem: &mut [f32], d_q_mem: &mut [f32],
    d_alpha: &mut [f32], d_theta: &mut [f32], d_m_initial: &mut [f32],
    seq_len: usize, d: usize, error_clip: f32,
) {
    let dd = d * d;
    let dev_km = DevBuf::new(seq_len * d);
    let dev_vm = DevBuf::new(seq_len * d);
    let dev_qm = DevBuf::new(seq_len * d);
    let dev_alpha = DevBuf::new(seq_len);
    let dev_theta = DevBuf::new(seq_len);
    let dev_mstates = DevBuf::new((seq_len + 1) * dd);
    let dev_dy = DevBuf::new(seq_len * d);
    let dev_dkm = DevBuf::new(seq_len * d);
    let dev_dvm = DevBuf::new(seq_len * d);
    let dev_dqm = DevBuf::new(seq_len * d);
    let dev_dalpha = DevBuf::new(seq_len);
    let dev_dtheta = DevBuf::new(seq_len);
    let dev_dm_init = DevBuf::new(dd);

    dev_km.copy_from_host(k_mem);
    dev_vm.copy_from_host(v_mem);
    dev_qm.copy_from_host(q_mem);
    dev_alpha.copy_from_host(alpha);
    dev_theta.copy_from_host(theta);
    dev_mstates.copy_from_host(m_states);
    dev_dy.copy_from_host(d_y);
    dev_dkm.zero();
    dev_dvm.zero();
    dev_dqm.zero();
    dev_dalpha.zero();
    dev_dtheta.zero();
    dev_dm_init.zero();

    unsafe {
        crate::cuda_ffi::dgd_backward_f32_cuda(
            dev_km.ptr, dev_vm.ptr, dev_qm.ptr,
            dev_alpha.ptr, dev_theta.ptr, dev_mstates.ptr,
            dev_dy.ptr as *const f32,
            dev_dkm.ptr, dev_dvm.ptr, dev_dqm.ptr,
            dev_dalpha.ptr, dev_dtheta.ptr, dev_dm_init.ptr,
            seq_len as i32, d as i32, error_clip,
        );
        let rc = cudaDeviceSynchronize();
        assert_eq!(rc, 0, "cudaDeviceSynchronize failed after dgd backward (error {rc})");
    }

    dev_dkm.copy_to_host(d_k_mem);
    dev_dvm.copy_to_host(d_v_mem);
    dev_dqm.copy_to_host(d_q_mem);
    dev_dalpha.copy_to_host(d_alpha);
    dev_dtheta.copy_to_host(d_theta);
    dev_dm_init.copy_to_host(d_m_initial);
}

// ══════════════════════════════════════════════════════════════════════
// Device-to-device dispatch variants (_dd)
//
// These accept GpuBuf<f32>/GpuBuf<u16> pointers and call the same CUDA
// kernels WITHOUT any H2D/D2H copies. Used by the GPU-resident forward
// and backward passes (gpu_forward.rs, gpu_backward.rs).
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
use crate::gpu_buf::{GpuBuf, GpuSlice};

/// cuBLAS sgemm on device buffers: C = alpha * A[m,k] @ B[k,n] + beta * C.
/// Row-major trick: call sgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n).
#[cfg(feature = "cuda")]
pub fn cublas_matmul_dd(
    a: &GpuBuf<f32>, b: &GpuBuf<f32>, out: &mut GpuBuf<f32>,
    m: usize, k: usize, n: usize, beta: f32,
) {
    let alpha_val: f32 = 1.0;
    let beta_val: f32 = beta;
    let rc = unsafe {
        cublasSgemm_v2(
            cublas_handle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha_val,
            b.as_ptr(), n as i32,
            a.as_ptr(), k as i32,
            &beta_val,
            out.ptr(), n as i32,
        )
    };
    assert_eq!(rc, 0, "cublasSgemm_v2 (dd) failed: error code {rc}");
}

/// cuBLAS fused transpose-B on device buffers: C = A[m,k] @ B^T where B is [n,k].
#[cfg(feature = "cuda")]
pub fn cublas_matmul_transb_dd(
    a: &GpuBuf<f32>, b: &GpuBuf<f32>, out: &mut GpuBuf<f32>,
    m: usize, k: usize, n: usize, beta: f32,
) {
    let alpha_val: f32 = 1.0;
    let beta_val: f32 = beta;
    let rc = unsafe {
        cublasSgemm_v2(
            cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            n as i32, m as i32, k as i32,
            &alpha_val,
            b.as_ptr(), k as i32,
            a.as_ptr(), k as i32,
            &beta_val,
            out.ptr(), n as i32,
        )
    };
    assert_eq!(rc, 0, "cublasSgemm_v2 transB (dd) failed: error code {rc}");
}

/// cuBLAS accumulate on device buffers: C[m,n] += A[m,k] @ B[k,n].
#[cfg(feature = "cuda")]
pub fn cublas_matmul_acc_dd(
    a: &GpuBuf<f32>, b: &GpuBuf<f32>, out: &mut GpuBuf<f32>,
    m: usize, k: usize, n: usize,
) {
    cublas_matmul_dd(a, b, out, m, k, n, 1.0);
}

/// SWA forward on device bf16 buffers. No H2D/D2H.
/// `n_persistent`: number of persistent prefix tokens (SWA* from Titans Eq 27).
/// When 0, behaves as standard sliding window attention.
#[cfg(feature = "cuda")]
pub fn swa_forward_dd(
    q: &GpuBuf<u16>, k: &GpuBuf<u16>, v: &GpuBuf<u16>,
    out: &mut GpuBuf<u16>, attn_weights: &mut GpuBuf<u16>,
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
    batch_size: usize, n_persistent: usize,
) {
    let total_dim = num_heads * head_dim;
    let qkv_len = batch_size * seq_len * total_dim;
    let aw_stride = n_persistent + window_size;
    let aw_len = batch_size * num_heads * seq_len * aw_stride;
    assert!(n_persistent <= seq_len, "swa_forward_dd: n_persistent ({n_persistent}) > seq_len ({seq_len})");
    assert!(q.len() >= qkv_len, "swa_forward_dd: q too small ({} < {qkv_len})", q.len());
    assert!(k.len() >= qkv_len, "swa_forward_dd: k too small ({} < {qkv_len})", k.len());
    assert!(v.len() >= qkv_len, "swa_forward_dd: v too small ({} < {qkv_len})", v.len());
    assert!(out.len() >= qkv_len, "swa_forward_dd: out too small ({} < {qkv_len})", out.len());
    assert!(attn_weights.len() >= aw_len, "swa_forward_dd: attn_weights too small ({} < {aw_len})", attn_weights.len());
    unsafe {
        crate::cuda_ffi::swa_forward_f32_cuda(
            q.as_ptr(), k.as_ptr(), v.as_ptr(),
            out.ptr(), attn_weights.ptr(),
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
            batch_size as i32, n_persistent as i32,
        );
    }
}

/// SWA single-token attention on device bf16 buffers (KV cache decode).
/// Q is [1, d], K/V cache are [cache_len, d], out is [1, d]. No attn_weights.
#[cfg(feature = "cuda")]
pub fn swa_single_token_dd(
    q: &GpuBuf<u16>, k_cache: &GpuBuf<u16>, v_cache: &GpuBuf<u16>,
    out: &mut GpuBuf<u16>,
    cache_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
    n_persistent: usize,
) {
    let d = num_heads * head_dim;
    assert!(cache_len >= n_persistent, "swa_single_token_dd: cache_len={cache_len} < n_persistent={n_persistent}");
    assert!(q.len() >= d, "swa_single_token_dd: q too small ({} < {d})", q.len());
    assert!(out.len() >= d, "swa_single_token_dd: out too small ({} < {d})", out.len());
    let cache_elems = cache_len * d;
    assert!(k_cache.len() >= cache_elems, "swa_single_token_dd: k_cache too small ({} < {cache_elems})", k_cache.len());
    assert!(v_cache.len() >= cache_elems, "swa_single_token_dd: v_cache too small ({} < {cache_elems})", v_cache.len());
    unsafe {
        crate::cuda_ffi::swa_single_token_cuda(
            q.as_ptr(), k_cache.as_ptr(), v_cache.as_ptr(),
            out.ptr(),
            cache_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
            n_persistent as i32,
        );
    }
}

/// SWA backward on device buffers. Q/K/V/aw are bf16, gradients are f32.
#[cfg(feature = "cuda")]
pub fn swa_backward_dd(
    q: &GpuBuf<u16>, k: &GpuBuf<u16>, v: &GpuBuf<u16>,
    attn_weights: &GpuBuf<u16>, d_attn_out: &GpuBuf<f32>,
    d_q: &mut GpuBuf<f32>, d_k: &mut GpuBuf<f32>, d_v: &mut GpuBuf<f32>,
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
    batch_size: usize, n_persistent: usize,
) {
    let total_dim = num_heads * head_dim;
    let qkv_len = batch_size * seq_len * total_dim;
    let aw_stride = n_persistent + window_size;
    let aw_len = batch_size * num_heads * seq_len * aw_stride;
    assert!(n_persistent <= seq_len, "swa_backward_dd: n_persistent ({n_persistent}) > seq_len ({seq_len})");
    assert!(q.len() >= qkv_len, "swa_backward_dd: q too small ({} < {qkv_len})", q.len());
    assert!(k.len() >= qkv_len, "swa_backward_dd: k too small ({} < {qkv_len})", k.len());
    assert!(v.len() >= qkv_len, "swa_backward_dd: v too small ({} < {qkv_len})", v.len());
    assert!(attn_weights.len() >= aw_len, "swa_backward_dd: attn_weights too small ({} < {aw_len})", attn_weights.len());
    assert!(d_attn_out.len() >= qkv_len, "swa_backward_dd: d_attn_out too small ({} < {qkv_len})", d_attn_out.len());
    assert!(d_q.len() >= qkv_len, "swa_backward_dd: d_q too small ({} < {qkv_len})", d_q.len());
    assert!(d_k.len() >= qkv_len, "swa_backward_dd: d_k too small ({} < {qkv_len})", d_k.len());
    assert!(d_v.len() >= qkv_len, "swa_backward_dd: d_v too small ({} < {qkv_len})", d_v.len());
    unsafe {
        crate::cuda_ffi::swa_backward_f32_cuda(
            q.as_ptr(), k.as_ptr(), v.as_ptr(),
            attn_weights.as_ptr(), d_attn_out.as_ptr(),
            d_q.ptr(), d_k.ptr(), d_v.ptr(),
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
            batch_size as i32, n_persistent as i32,
        );
    }
}

/// Validate stride parameters for batched forward kernels.
/// Uses debug_assert (internal API — callers are all in this crate).
#[cfg(feature = "cuda")]
#[inline]
fn check_forward_strides(
    seq_len: usize, d: usize, batch_size: usize,
    input_stride: usize, m_stride: usize,
) {
    debug_assert!(batch_size > 0, "batch_size must be > 0");
    debug_assert!(input_stride >= seq_len,
        "input_stride ({input_stride}) must be >= seq_len ({seq_len})");
    let dd = d * d;
    debug_assert!(m_stride >= dd,
        "m_stride ({m_stride}) must be >= d*d ({dd})");
    debug_assert!(seq_len <= i32::MAX as usize, "seq_len exceeds i32");
    debug_assert!(d <= i32::MAX as usize, "d exceeds i32");
    debug_assert!(batch_size <= i32::MAX as usize, "batch_size exceeds i32");
    debug_assert!(input_stride <= i32::MAX as usize, "input_stride exceeds i32");
    debug_assert!(m_stride <= i32::MAX as usize, "m_stride exceeds i32");
}

/// Delta forward on device buffers.
#[cfg(feature = "cuda")]
pub fn delta_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize,
    input_stride: usize, m_stride: usize, error_clip: f32,
    m_norm_max: f32,
) {
    check_forward_strides(seq_len, d, batch_size, input_stride, m_stride);
    unsafe {
        crate::cuda_ffi::delta_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32,
            input_stride as i32, m_stride as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Delta backward on device buffers.
#[cfg(feature = "cuda")]
pub fn delta_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, error_clip: f32,
) {
    unsafe {
        crate::cuda_ffi::delta_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), m_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_initial.ptr(),
            seq_len as i32, d as i32, batch_size as i32, error_clip,
        );
    }
}

/// Titans forward on device buffers.
#[cfg(feature = "cuda")]
pub fn titans_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, s_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize,
    input_stride: usize, m_stride: usize, error_clip: f32,
    m_norm_max: f32,
) {
    check_forward_strides(seq_len, d, batch_size, input_stride, m_stride);
    unsafe {
        crate::cuda_ffi::titans_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32,
            input_stride as i32, m_stride as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Titans backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, s_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_eta: &mut GpuBuf<f32>,
    d_m_initial: &mut GpuBuf<f32>, d_s_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, error_clip: f32,
) {
    unsafe {
        crate::cuda_ffi::titans_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_states.as_ptr(), s_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_eta.ptr(),
            d_m_initial.ptr(), d_s_initial.ptr(),
            seq_len as i32, d as i32, batch_size as i32, error_clip,
        );
    }
}

// ── TitansLMM MLP memory dispatch (spec 75) ───────────────────────

/// TitansLMM MLP memory forward on device buffers.
/// state_size = 2*d*d_hidden + d_hidden + d per head.
/// activation: 0=GELU, 1=SiLU, 2=ReLU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_mlp_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, s_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, d_hidden: usize, batch_size: usize,
    activation: i32, m_norm_max: f32,
) {
    let state_size = d_hidden * d + d_hidden + d * d_hidden + d;
    let input_stride = seq_len * d;
    let m_stride = (seq_len + 1) * state_size;
    unsafe {
        crate::cuda_ffi::titans_mlp_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, d_hidden as i32,
            batch_size as i32,
            input_stride as i32, m_stride as i32,
            activation, m_norm_max,
        );
    }
}

/// TitansLMM MLP memory backward on device buffers.
/// Requires m_states/s_states from forward pass (full trajectory).
/// activation: 0=GELU, 1=SiLU, 2=ReLU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_mlp_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, s_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_eta: &mut GpuBuf<f32>,
    d_m_initial: &mut GpuBuf<f32>, d_s_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, d_hidden: usize, batch_size: usize,
    activation: i32, m_norm_max: f32,
) {
    let state_size = d_hidden * d + d_hidden + d * d_hidden + d;
    let input_stride = seq_len * d;
    let m_stride = (seq_len + 1) * state_size;
    unsafe {
        crate::cuda_ffi::titans_mlp_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_states.as_ptr(), s_states.as_ptr(), d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_eta.ptr(),
            d_m_initial.ptr(), d_s_initial.ptr(),
            seq_len as i32, d as i32, d_hidden as i32,
            batch_size as i32,
            input_stride as i32, m_stride as i32,
            activation, m_norm_max,
        );
    }
}

// ── Chunkwise dispatch wrappers (spec 43 — frozen-M₀) ──────────────

/// Delta chunkwise forward on device buffers.
/// m_chunk_states: [bs * (num_chunks+1) * d*d] — M at each chunk boundary + final.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_chunkwise_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_chunk_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::delta_chunkwise_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_chunk_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32, chunk_size as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Delta chunkwise backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_chunkwise_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_chunk_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::delta_chunkwise_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), m_chunk_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_initial.ptr(),
            seq_len as i32, d as i32, batch_size as i32, chunk_size as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Titans chunkwise forward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_chunkwise_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_chunk_states: &mut GpuBuf<f32>, s_chunk_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::titans_chunkwise_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_chunk_states.ptr(), s_chunk_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32, chunk_size as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Titans chunkwise backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_chunkwise_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_chunk_states: &GpuBuf<f32>, s_chunk_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_eta: &mut GpuBuf<f32>,
    d_m_initial: &mut GpuBuf<f32>, d_s_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::titans_chunkwise_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_chunk_states.as_ptr(), s_chunk_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_eta.ptr(),
            d_m_initial.ptr(), d_s_initial.ptr(),
            seq_len as i32, d as i32, batch_size as i32, chunk_size as i32, error_clip,
            m_norm_max,
        );
    }
}

// ── Spec 44: Batched cuBLAS Phase 1 orchestration ─────────────────────

/// Delta chunkwise forward with batched cuBLAS Phase 1.
/// Per-chunk loop: cuBLAS GEMM for predictions, error_subtract_clip, Phase 2 kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_chunkwise_forward_batched_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_chunk_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // Allocate persistent workspaces
    let m_work = GpuBuf::<f32>::zeros(batch_size * dd);
    let predictions = GpuBuf::<f32>::zeros(batch_size * chunk_size * d);

    // Initialize m_work from m_initial
    unsafe { m_work.copy_from_device_ptr(m_initial.as_ptr(), batch_size * dd); }

    for c in 0..num_chunks {
        let t_start = c * chunk_size;
        let t_end = std::cmp::min(t_start + chunk_size, seq_len);
        let c_len = t_end - t_start;

        // Phase 1: batched cuBLAS GEMM — all batch elements in one call
        {
            let alpha_val: f32 = 1.0;
            let beta_val: f32 = 0.0;
            let rc = unsafe {
                cublasSgemmStridedBatched(
                    cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    d as i32, c_len as i32, d as i32,
                    &alpha_val,
                    m_work.as_ptr(), d as i32, dd as i64,
                    k_mem.as_ptr().add(t_start * d), d as i32, (seq_len * d) as i64,
                    &beta_val,
                    predictions.ptr(), d as i32, (chunk_size * d) as i64,
                    batch_size as i32,
                )
            };
            assert_eq!(rc, 0, "cublasSgemmStridedBatched (delta fwd phase1) failed: {rc}");
        }

        // Error subtract + clip: predictions -= V_chunk, then L2 clip
        for b in 0..batch_size {
            let v_offset = b * seq_len * d + t_start * d;
            let pred_offset = b * chunk_size * d;
            unsafe {
                crate::cuda_ffi::error_subtract_clip_f32_cuda(
                    predictions.ptr().add(pred_offset),
                    v_mem.as_ptr().add(v_offset),
                    c_len as i32, d as i32, error_clip,
                );
            }
        }

        // Phase 2: sequential recurrence + readout + boundary store
        unsafe {
            crate::cuda_ffi::delta_phase2_forward_f32_cuda(
                k_mem.as_ptr(), q_mem.as_ptr(),
                alpha.as_ptr(), theta.as_ptr(),
                predictions.as_ptr(), m_work.ptr(),
                m_chunk_states.ptr(), y.ptr(),
                seq_len as i32, d as i32, batch_size as i32,
                chunk_size as i32, c as i32,
                m_norm_max,
            );
        }
    }
}

/// Titans chunkwise forward with batched cuBLAS Phase 1.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_chunkwise_forward_batched_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_chunk_states: &mut GpuBuf<f32>, s_chunk_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    let m_work = GpuBuf::<f32>::zeros(batch_size * dd);
    let s_work = GpuBuf::<f32>::zeros(batch_size * dd);
    let predictions = GpuBuf::<f32>::zeros(batch_size * chunk_size * d);

    // Initialize m_work from m_initial, s_work from s_initial
    unsafe {
        m_work.copy_from_device_ptr(m_initial.as_ptr(), batch_size * dd);
        s_work.copy_from_device_ptr(s_initial.as_ptr(), batch_size * dd);
    }

    for c in 0..num_chunks {
        let t_start = c * chunk_size;
        let t_end = std::cmp::min(t_start + chunk_size, seq_len);
        let c_len = t_end - t_start;

        // Phase 1: batched cuBLAS GEMM — all batch elements in one call
        {
            let alpha_val: f32 = 1.0;
            let beta_val: f32 = 0.0;
            let rc = unsafe {
                cublasSgemmStridedBatched(
                    cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    d as i32, c_len as i32, d as i32,
                    &alpha_val,
                    m_work.as_ptr(), d as i32, dd as i64,
                    k_mem.as_ptr().add(t_start * d), d as i32, (seq_len * d) as i64,
                    &beta_val,
                    predictions.ptr(), d as i32, (chunk_size * d) as i64,
                    batch_size as i32,
                )
            };
            assert_eq!(rc, 0, "cublasSgemmStridedBatched (titans fwd phase1) failed: {rc}");
        }

        // Error subtract + clip (per-batch: V stride differs from predictions stride)
        for b in 0..batch_size {
            let v_offset = b * seq_len * d + t_start * d;
            let pred_offset = b * chunk_size * d;
            unsafe {
                crate::cuda_ffi::error_subtract_clip_f32_cuda(
                    predictions.ptr().add(pred_offset),
                    v_mem.as_ptr().add(v_offset),
                    c_len as i32, d as i32, error_clip,
                );
            }
        }

        // Phase 2: sequential recurrence + readout + boundary store
        unsafe {
            crate::cuda_ffi::titans_phase2_forward_f32_cuda(
                k_mem.as_ptr(), q_mem.as_ptr(),
                alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
                predictions.as_ptr(), m_work.ptr(), s_work.ptr(),
                m_chunk_states.ptr(), s_chunk_states.ptr(), y.ptr(),
                seq_len as i32, d as i32, batch_size as i32,
                chunk_size as i32, c as i32,
                m_norm_max,
            );
        }
    }
}

/// Delta chunkwise backward with batched cuBLAS Phase 1 error recompute.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_chunkwise_backward_batched_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_chunk_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // Allocate workspaces
    let d_M = GpuBuf::<f32>::zeros(batch_size * dd);
    let d_M0 = GpuBuf::<f32>::zeros(batch_size * dd);
    let m_recompute = GpuBuf::<f32>::zeros(batch_size * (chunk_size + 1) * dd);
    let errors = GpuBuf::<f32>::zeros(batch_size * chunk_size * d);

    // Process chunks in reverse
    for c in (0..num_chunks).rev() {
        let t_start = c * chunk_size;
        let t_end = std::cmp::min(t_start + chunk_size, seq_len);
        let c_len = t_end - t_start;

        // Phase 1: batched cuBLAS GEMM — recompute errors for all batch elements
        {
            let alpha_val: f32 = 1.0;
            let beta_val: f32 = 0.0;
            let rc = unsafe {
                cublasSgemmStridedBatched(
                    cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    d as i32, c_len as i32, d as i32,
                    &alpha_val,
                    m_chunk_states.as_ptr().add(c * dd), d as i32, ((num_chunks + 1) * dd) as i64,
                    k_mem.as_ptr().add(t_start * d), d as i32, (seq_len * d) as i64,
                    &beta_val,
                    errors.ptr(), d as i32, (chunk_size * d) as i64,
                    batch_size as i32,
                )
            };
            assert_eq!(rc, 0, "cublasSgemmStridedBatched (delta bwd phase1) failed: {rc}");
        }

        // Error subtract + clip (per-batch: V stride differs from errors stride)
        for b in 0..batch_size {
            let v_offset = b * seq_len * d + t_start * d;
            let err_offset = b * chunk_size * d;
            unsafe {
                crate::cuda_ffi::error_subtract_clip_f32_cuda(
                    errors.ptr().add(err_offset),
                    v_mem.as_ptr().add(v_offset),
                    c_len as i32, d as i32, error_clip,
                );
            }
        }

        // Phase 2 backward
        unsafe {
            crate::cuda_ffi::delta_phase2_backward_f32_cuda(
                k_mem.as_ptr(), q_mem.as_ptr(),
                alpha.as_ptr(), theta.as_ptr(),
                errors.as_ptr(), m_chunk_states.as_ptr(),
                d_y.as_ptr(),
                d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
                d_alpha.ptr(), d_theta.ptr(),
                d_M.ptr(), d_M0.ptr(), m_recompute.ptr(),
                seq_len as i32, d as i32, batch_size as i32,
                chunk_size as i32, c as i32,
                m_norm_max,
            );
        }
    }

    // d_M holds per-batch gradients w.r.t. m_initial — need to sum across batch.
    // The monolithic kernel does this via atomicAdd. We need the same behavior.
    // Zero d_m_initial first, then accumulate each batch element.
    d_m_initial.zero();
    for b in 0..batch_size {
        // d_m_initial[i] += d_M[b*dd + i]
        let d_m_slice = d_M.slice(b * dd, dd);
        // Accumulate using cuBLAS saxpy: y = alpha*x + y
        let alpha_one: f32 = 1.0;
        unsafe {
            cublasSaxpy_v2(
                cublas_handle(),
                dd as i32,
                &alpha_one,
                d_m_slice.as_ptr(), 1,
                d_m_initial.ptr(), 1,
            );
        }
    }
}

/// Titans chunkwise backward with batched cuBLAS Phase 1 error recompute.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_chunkwise_backward_batched_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_chunk_states: &GpuBuf<f32>, s_chunk_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_eta: &mut GpuBuf<f32>,
    d_m_initial: &mut GpuBuf<f32>, d_s_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, chunk_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    let dd = d * d;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    let d_M = GpuBuf::<f32>::zeros(batch_size * dd);
    let d_S = GpuBuf::<f32>::zeros(batch_size * dd);
    let d_M0 = GpuBuf::<f32>::zeros(batch_size * dd);
    let m_recompute = GpuBuf::<f32>::zeros(batch_size * (chunk_size + 1) * dd);
    let s_recompute = GpuBuf::<f32>::zeros(batch_size * (chunk_size + 1) * dd);
    let errors = GpuBuf::<f32>::zeros(batch_size * chunk_size * d);

    for c in (0..num_chunks).rev() {
        let t_start = c * chunk_size;
        let t_end = std::cmp::min(t_start + chunk_size, seq_len);
        let c_len = t_end - t_start;

        // Phase 1: batched cuBLAS GEMM — recompute errors for all batch elements
        {
            let alpha_val: f32 = 1.0;
            let beta_val: f32 = 0.0;
            let rc = unsafe {
                cublasSgemmStridedBatched(
                    cublas_handle(),
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    d as i32, c_len as i32, d as i32,
                    &alpha_val,
                    m_chunk_states.as_ptr().add(c * dd), d as i32, ((num_chunks + 1) * dd) as i64,
                    k_mem.as_ptr().add(t_start * d), d as i32, (seq_len * d) as i64,
                    &beta_val,
                    errors.ptr(), d as i32, (chunk_size * d) as i64,
                    batch_size as i32,
                )
            };
            assert_eq!(rc, 0, "cublasSgemmStridedBatched (titans bwd phase1) failed: {rc}");
        }

        // Error subtract + clip (per-batch: V stride differs from errors stride)
        for b in 0..batch_size {
            let v_offset = b * seq_len * d + t_start * d;
            let err_offset = b * chunk_size * d;
            unsafe {
                crate::cuda_ffi::error_subtract_clip_f32_cuda(
                    errors.ptr().add(err_offset),
                    v_mem.as_ptr().add(v_offset),
                    c_len as i32, d as i32, error_clip,
                );
            }
        }

        // Phase 2 backward
        unsafe {
            crate::cuda_ffi::titans_phase2_backward_f32_cuda(
                k_mem.as_ptr(), q_mem.as_ptr(),
                alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
                errors.as_ptr(),
                m_chunk_states.as_ptr(), s_chunk_states.as_ptr(),
                d_y.as_ptr(),
                d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
                d_alpha.ptr(), d_theta.ptr(), d_eta.ptr(),
                d_M.ptr(), d_S.ptr(), d_M0.ptr(),
                m_recompute.ptr(), s_recompute.ptr(),
                seq_len as i32, d as i32, batch_size as i32,
                chunk_size as i32, c as i32,
                m_norm_max,
            );
        }
    }

    // Accumulate per-batch gradients into d_m_initial, d_s_initial
    d_m_initial.zero();
    d_s_initial.zero();
    for b in 0..batch_size {
        let alpha_one: f32 = 1.0;
        unsafe {
            cublasSaxpy_v2(
                cublas_handle(), dd as i32, &alpha_one,
                d_M.as_ptr().add(b * dd), 1,
                d_m_initial.ptr(), 1,
            );
            cublasSaxpy_v2(
                cublas_handle(), dd as i32, &alpha_one,
                d_S.as_ptr().add(b * dd), 1,
                d_s_initial.ptr(), 1,
            );
        }
    }
}

/// Fused DGD forward: L2-normalize + gate compute + DGD recurrence in one kernel (spec 39).
/// k_mem and q_mem are normalized in-place. alpha/theta/norms written to output buffers for backward.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_fused_forward_dd(
    k_mem: &mut GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &mut GpuBuf<f32>,
    w_alpha: &GpuBuf<f32>, b_alpha: &GpuBuf<f32>,
    w_theta: &GpuBuf<f32>, b_theta: &GpuBuf<f32>,
    alpha_floor: f32, alpha_ceil: f32,
    theta_floor: f32, theta_ceil: f32,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    alpha_out: &mut GpuBuf<f32>, theta_out: &mut GpuBuf<f32>,
    k_norms_out: &mut GpuBuf<f32>, q_norms_out: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, error_clip: f32,
) {
    unsafe {
        crate::cuda_ffi::dgd_fused_forward_f32_cuda(
            k_mem.ptr(), v_mem.as_ptr(), q_mem.ptr(),
            w_alpha.as_ptr(), b_alpha.as_ptr(),
            w_theta.as_ptr(), b_theta.as_ptr(),
            alpha_floor, alpha_ceil, theta_floor, theta_ceil,
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            alpha_out.ptr(), theta_out.ptr(),
            k_norms_out.ptr(), q_norms_out.ptr(),
            seq_len as i32, d as i32, batch_size as i32, error_clip,
        );
    }
}

/// Fused Titans forward: L2-normalize + gate compute (alpha/theta/eta) + Titans recurrence (spec 39).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_fused_forward_dd(
    k_mem: &mut GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &mut GpuBuf<f32>,
    w_alpha: &GpuBuf<f32>, b_alpha: &GpuBuf<f32>,
    w_theta: &GpuBuf<f32>, b_theta: &GpuBuf<f32>,
    w_eta: &GpuBuf<f32>, b_eta: &GpuBuf<f32>,
    alpha_floor: f32, alpha_ceil: f32,
    theta_floor: f32, theta_ceil: f32,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, s_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    alpha_out: &mut GpuBuf<f32>, theta_out: &mut GpuBuf<f32>, eta_out: &mut GpuBuf<f32>,
    k_norms_out: &mut GpuBuf<f32>, q_norms_out: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::titans_fused_forward_f32_cuda(
            k_mem.ptr(), v_mem.as_ptr(), q_mem.ptr(),
            w_alpha.as_ptr(), b_alpha.as_ptr(),
            w_theta.as_ptr(), b_theta.as_ptr(),
            w_eta.as_ptr(), b_eta.as_ptr(),
            alpha_floor, alpha_ceil, theta_floor, theta_ceil,
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            alpha_out.ptr(), theta_out.ptr(), eta_out.ptr(),
            k_norms_out.ptr(), q_norms_out.ptr(),
            seq_len as i32, d as i32, batch_size as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Hebbian forward on device buffers.
#[cfg(feature = "cuda")]
pub fn hebbian_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize,
    input_stride: usize, m_stride: usize,
) {
    check_forward_strides(seq_len, d, batch_size, input_stride, m_stride);
    unsafe {
        crate::cuda_ffi::hebbian_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32,
            input_stride as i32, m_stride as i32,
        );
    }
}

/// Hebbian backward on device buffers.
#[cfg(feature = "cuda")]
pub fn hebbian_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_m_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::hebbian_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_m_initial.ptr(),
            seq_len as i32, d as i32,
        );
    }
}

/// DGD forward on device buffers.
#[cfg(feature = "cuda")]
pub fn dgd_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, batch_size: usize,
    input_stride: usize, m_stride: usize, error_clip: f32,
) {
    check_forward_strides(seq_len, d, batch_size, input_stride, m_stride);
    unsafe {
        crate::cuda_ffi::dgd_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, batch_size as i32,
            input_stride as i32, m_stride as i32, error_clip,
        );
    }
}

/// DGD backward on device buffers.
#[cfg(feature = "cuda")]
pub fn dgd_backward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_initial: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, error_clip: f32,
) {
    unsafe {
        crate::cuda_ffi::dgd_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), m_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_initial.ptr(),
            seq_len as i32, d as i32, error_clip,
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// Gradient checkpointing dispatch wrappers
// ══════════════════════════════════════════════════════════════════════

/// Delta checkpointed forward on device buffers.
#[cfg(feature = "cuda")]
pub fn delta_forward_dd_ckpt(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, checkpoint_interval: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::delta_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Titans checkpointed forward on device buffers.
#[cfg(feature = "cuda")]
pub fn titans_forward_dd_ckpt(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>, s_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, s_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, checkpoint_interval: usize, error_clip: f32,
    m_norm_max: f32,
) {
    unsafe {
        crate::cuda_ffi::titans_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32, error_clip,
            m_norm_max,
        );
    }
}

/// Hebbian checkpointed forward on device buffers.
#[cfg(feature = "cuda")]
pub fn hebbian_forward_dd_ckpt(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, checkpoint_interval: usize,
) {
    unsafe {
        crate::cuda_ffi::hebbian_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32,
        );
    }
}

/// DGD checkpointed forward on device buffers.
#[cfg(feature = "cuda")]
pub fn dgd_forward_dd_ckpt(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize, checkpoint_interval: usize, error_clip: f32,
) {
    unsafe {
        crate::cuda_ffi::dgd_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32, error_clip,
        );
    }
}

/// Delta segment backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_backward_dd_segment(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_m_seed: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_out: &mut GpuBuf<f32>,
    t_start: usize, t_end: usize, d: usize, batch_size: usize, seq_len: usize,
    error_clip: f32,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(t_end <= seq_len, "segment t_end={t_end} must be <= seq_len={seq_len}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= batch_size * d * d, "d_m_seed too small");
    debug_assert!(d_m_out.len() >= batch_size * d * d, "d_m_out too small");
    unsafe {
        crate::cuda_ffi::delta_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
            batch_size as i32, seq_len as i32, error_clip,
        );
    }
}

/// Titans segment backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn titans_backward_dd_segment(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, s_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_m_seed: &GpuBuf<f32>, d_s_seed: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_eta: &mut GpuBuf<f32>,
    d_m_out: &mut GpuBuf<f32>, d_s_out: &mut GpuBuf<f32>,
    t_start: usize, t_end: usize, d: usize, batch_size: usize, seq_len: usize,
    error_clip: f32,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(t_end <= seq_len, "segment t_end={t_end} must be <= seq_len={seq_len}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= batch_size * d * d, "d_m_seed too small");
    debug_assert!(d_m_out.len() >= batch_size * d * d, "d_m_out too small");
    debug_assert!(d_s_seed.len() >= batch_size * d * d, "d_s_seed too small");
    debug_assert!(d_s_out.len() >= batch_size * d * d, "d_s_out too small");
    unsafe {
        crate::cuda_ffi::titans_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_states.as_ptr(), s_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(), d_s_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_eta.ptr(),
            d_m_out.ptr(), d_s_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
            batch_size as i32, seq_len as i32, error_clip,
        );
    }
}

/// Hebbian segment backward on device buffers.
#[cfg(feature = "cuda")]
pub fn hebbian_backward_dd_segment(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_m_seed: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_m_out: &mut GpuBuf<f32>,
    t_start: usize, t_end: usize, d: usize, batch_size: usize, seq_len: usize,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(t_end <= seq_len, "segment t_end={t_end} must be <= seq_len={seq_len}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= batch_size * d * d, "d_m_seed too small");
    debug_assert!(d_m_out.len() >= batch_size * d * d, "d_m_out too small");
    unsafe {
        crate::cuda_ffi::hebbian_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_m_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
            batch_size as i32, seq_len as i32,
        );
    }
}

/// DGD segment backward on device buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn dgd_backward_dd_segment(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_states: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    d_m_seed: &GpuBuf<f32>,
    d_k_mem: &mut GpuBuf<f32>, d_v_mem: &mut GpuBuf<f32>, d_q_mem: &mut GpuBuf<f32>,
    d_alpha: &mut GpuBuf<f32>, d_theta: &mut GpuBuf<f32>, d_m_out: &mut GpuBuf<f32>,
    t_start: usize, t_end: usize, d: usize, batch_size: usize, seq_len: usize,
    error_clip: f32,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(t_end <= seq_len, "segment t_end={t_end} must be <= seq_len={seq_len}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= batch_size * d * d, "d_m_seed too small");
    debug_assert!(d_m_out.len() >= batch_size * d * d, "d_m_out too small");
    unsafe {
        crate::cuda_ffi::dgd_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
            batch_size as i32, seq_len as i32, error_clip,
        );
    }
}

/// Gate backward: accumulate d_w_alpha/theta/eta and d_b_alpha/theta/eta on device.
///
/// `d_theta`/`theta` and `d_eta`/`eta` are Option — pass None for rules without that gate.
/// When has_theta=0/has_eta=0, kernel skips the corresponding computation.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gate_backward_dd(
    d_alpha: &GpuBuf<f32>, alpha: &GpuBuf<f32>,
    d_theta: Option<&GpuBuf<f32>>, theta: Option<&GpuBuf<f32>>,
    d_eta:   Option<&GpuBuf<f32>>, eta:   Option<&GpuBuf<f32>>,
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>,
    d_w_alpha: &mut GpuBuf<f32>, d_b_alpha: &mut GpuBuf<f32>,
    d_w_theta: &mut GpuBuf<f32>, d_b_theta: &mut GpuBuf<f32>,
    d_w_eta:   &mut GpuBuf<f32>, d_b_eta:   &mut GpuBuf<f32>,
    seq_len: usize, d: usize,
) {
    let has_theta = d_theta.is_some() as i32;
    let has_eta   = d_eta.is_some() as i32;
    let null: *const f32 = std::ptr::null();
    let null_mut: *mut f32 = std::ptr::null_mut();
    unsafe {
        crate::cuda_ffi::gate_backward_cuda(
            d_alpha.as_ptr(), alpha.as_ptr(),
            d_theta.map(|b| b.as_ptr()).unwrap_or(null),
            theta.map(|b| b.as_ptr()).unwrap_or(null),
            d_eta.map(|b| b.as_ptr()).unwrap_or(null),
            eta.map(|b| b.as_ptr()).unwrap_or(null),
            k_mem.as_ptr(), v_mem.as_ptr(),
            d_w_alpha.ptr(), d_b_alpha.ptr(),
            if has_theta == 1 { d_w_theta.ptr() } else { null_mut },
            if has_theta == 1 { d_b_theta.ptr() } else { null_mut },
            if has_eta   == 1 { d_w_eta.ptr() }   else { null_mut },
            if has_eta   == 1 { d_b_eta.ptr() }    else { null_mut },
            seq_len as i32, d as i32, has_theta, has_eta,
        );
    }
}

/// Synchronize the CUDA device (wait for all pending kernel launches).
#[cfg(feature = "cuda")]
pub fn cuda_sync() {
    let rc = unsafe { cudaDeviceSynchronize() };
    assert_eq!(rc, 0, "cudaDeviceSynchronize failed: error code {rc}");
}

// ══════════════════════════════════════════════════════════════════════
// TNT helper kernel dispatch (device-to-device)
// ══════════════════════════════════════════════════════════════════════

/// Broadcast global M to N contiguous copies on device.
#[cfg(feature = "cuda")]
pub fn tnt_broadcast_m_dd(
    m_src: &GpuBuf<f32>, m_dst: &mut GpuBuf<f32>,
    n_local: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::tnt_broadcast_m_f32_cuda(
            m_src.as_ptr(), m_dst.ptr(),
            n_local as i32, d as i32,
        );
    }
}

/// Mean-pool local outputs into shard summary on device.
#[cfg(feature = "cuda")]
pub fn tnt_shard_summary_mean_dd(
    local_y: &GpuBuf<f32>, k_sum: &mut GpuBuf<f32>, v_sum: &mut GpuBuf<f32>,
    shard_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::tnt_shard_summary_mean_f32_cuda(
            local_y.as_ptr(), k_sum.ptr(), v_sum.ptr(),
            shard_len as i32, d as i32,
        );
    }
}

/// Update global M via outer product on device.
#[cfg(feature = "cuda")]
pub fn tnt_global_update_dd(
    global_m: &mut GpuBuf<f32>, k_sum: &GpuBuf<f32>, v_sum: &GpuBuf<f32>,
    d: usize, alpha: f32,
) {
    unsafe {
        crate::cuda_ffi::tnt_global_update_f32_cuda(
            global_m.ptr(), k_sum.as_ptr(), v_sum.as_ptr(),
            d as i32, alpha,
        );
    }
}

/// Backward through global M update on device.
#[cfg(feature = "cuda")]
pub fn tnt_global_update_backward_dd(
    d_m_new: &GpuBuf<f32>, k_sum: &GpuBuf<f32>, v_sum: &GpuBuf<f32>,
    d_m_old: &mut GpuBuf<f32>, d_k_sum: &mut GpuBuf<f32>, d_v_sum: &mut GpuBuf<f32>,
    d: usize, alpha: f32,
) {
    unsafe {
        crate::cuda_ffi::tnt_global_update_backward_f32_cuda(
            d_m_new.as_ptr(), k_sum.as_ptr(), v_sum.as_ptr(),
            d_m_old.ptr(), d_k_sum.ptr(), d_v_sum.ptr(),
            d as i32, alpha,
        );
    }
}

/// Backward through mean-pooling shard summary on device.
#[cfg(feature = "cuda")]
pub fn tnt_shard_summary_mean_backward_dd(
    d_k_sum: &GpuBuf<f32>, d_v_sum: &GpuBuf<f32>,
    d_local_y: &mut GpuBuf<f32>, shard_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::tnt_shard_summary_mean_backward_f32_cuda(
            d_k_sum.as_ptr(), d_v_sum.as_ptr(),
            d_local_y.ptr(), shard_len as i32, d as i32,
        );
    }
}

/// Combine upstream + global gradient contributions on device.
#[cfg(feature = "cuda")]
pub fn tnt_combine_gradients_dd(
    d_y_upstream: &GpuBuf<f32>, d_y_global: &GpuBuf<f32>,
    d_y_combined: &mut GpuBuf<f32>, n: usize,
) {
    unsafe {
        crate::cuda_ffi::tnt_combine_gradients_f32_cuda(
            d_y_upstream.as_ptr(), d_y_global.as_ptr(),
            d_y_combined.ptr(), n as i32,
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// M-norm clamp dispatch (spec 65)
// ══════════════════════════════════════════════════════════════════════

/// Clamp the Frobenius norm of a single d×d matrix on device.
/// No-op if m_norm_max <= 0 or >= 1e30.
#[cfg(feature = "cuda")]
pub fn m_norm_clamp(m: &mut GpuBuf<f32>, d: i32, m_norm_max: f32) {
    unsafe {
        crate::cuda_ffi::m_norm_clamp_f32_cuda(m.ptr(), d, m_norm_max);
    }
}

/// Batched M-norm clamp: clamp batch_size independent d×d matrices in one launch.
/// m points to contiguous [batch_size, d*d] buffer on device.
/// No-op if m_norm_max <= 0 or >= 1e30.
#[cfg(feature = "cuda")]
pub fn m_norm_clamp_batch(m: &mut GpuBuf<f32>, d: i32, batch_size: i32, m_norm_max: f32) {
    unsafe {
        crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(m.ptr(), d, batch_size, m_norm_max);
    }
}
