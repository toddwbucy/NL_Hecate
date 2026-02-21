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
/// - `CudaNative`: architecture-specific SASS (sm_86/89/90)
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
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
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
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
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
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_delta_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                               m_states, y, seq_len, d);
            return;
        }
    }
    rust_delta_forward(k_mem, v_mem, q_mem, alpha, theta, m_initial,
                       m_states, y, seq_len, d);
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
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_delta_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                                d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                                seq_len, d);
            return;
        }
    }
    rust_delta_backward(k_mem, v_mem, q_mem, alpha, theta, m_states, d_y,
                        d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_m_initial,
                        seq_len, d);
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
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_titans_forward(k_mem, v_mem, q_mem, alpha, theta, eta,
                                m_initial, s_initial, m_states, s_states, y,
                                seq_len, d);
            return;
        }
    }
    rust_titans_forward(k_mem, v_mem, q_mem, alpha, theta, eta,
                        m_initial, s_initial, m_states, s_states, y,
                        seq_len, d);
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
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_titans_backward(k_mem, v_mem, q_mem, alpha, theta, eta,
                                 m_states, s_states, d_y,
                                 d_k_mem, d_v_mem, d_q_mem,
                                 d_alpha, d_theta, d_eta,
                                 d_m_initial, d_s_initial,
                                 seq_len, d);
            return;
        }
    }
    rust_titans_backward(k_mem, v_mem, q_mem, alpha, theta, eta,
                         m_states, s_states, d_y,
                         d_k_mem, d_v_mem, d_q_mem,
                         d_alpha, d_theta, d_eta,
                         d_m_initial, d_s_initial,
                         seq_len, d);
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
) {
    #[cfg(feature = "cuda")]
    {
        if !is_rust_forced() {
            cuda_hebbian_forward(k_mem, v_mem, q_mem, alpha, m_initial,
                                 m_states, y, seq_len, d);
            return;
        }
    }
    rust_hebbian_forward(k_mem, v_mem, q_mem, alpha, m_initial,
                         m_states, y, seq_len, d);
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

// ── Rust reference inner loops ──────────────────────────────────────
// Always compiled. Used as fallback when CUDA is absent or force_rust_reference() is set.

/// Rust reference Delta Rule forward inner loop.
fn rust_delta_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_initial: &[f32],
    m_states: &mut [f32], y: &mut [f32],
    seq_len: usize, d: usize,
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

        // error = prediction - v; M update
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            let err_i = prediction[i] - v_t[i];
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] - theta_t * err_i * k_t[j];
            }
        }

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
    seq_len: usize, d: usize,
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

        // Recompute prediction and error
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_t[i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }

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
    seq_len: usize, d: usize,
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

        // S_{t+1} = eta_t * S_t - theta_t * outer(error, k)
        for i in 0..d {
            let err_i = prediction[i] - v_t[i];
            for j in 0..d {
                s_states[s_next + i * d + j] =
                    eta_t * s_states[s_t + i * d + j] - theta_t * err_i * k_t[j];
            }
        }

        // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}
        let retention = 1.0 - alpha_t;
        for i in 0..dd {
            m_states[m_next + i] = retention * m_states[m_t + i] + s_states[s_next + i];
        }

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
    seq_len: usize, d: usize,
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

        // Recompute prediction/error
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t_off + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }

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
    seq_len: usize, d: usize,
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
    seq_len: usize, d: usize,
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
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
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
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
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
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
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
            seq_len as i32, d as i32,
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
            seq_len as i32, d as i32,
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

// ══════════════════════════════════════════════════════════════════════
// Device-to-device dispatch variants (_dd)
//
// These accept GpuBuf<f32>/GpuBuf<u16> pointers and call the same CUDA
// kernels WITHOUT any H2D/D2H copies. Used by the GPU-resident forward
// and backward passes (gpu_forward.rs, gpu_backward.rs).
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
use crate::gpu_buf::{GpuBuf, GpuSlice, GpuSliceMut};

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
#[cfg(feature = "cuda")]
pub fn swa_forward_dd(
    q: &GpuBuf<u16>, k: &GpuBuf<u16>, v: &GpuBuf<u16>,
    out: &mut GpuBuf<u16>, attn_weights: &mut GpuBuf<u16>,
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    unsafe {
        crate::cuda_ffi::swa_forward_f32_cuda(
            q.as_ptr(), k.as_ptr(), v.as_ptr(),
            out.ptr(), attn_weights.ptr(),
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
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
) {
    unsafe {
        crate::cuda_ffi::swa_single_token_cuda(
            q.as_ptr(), k_cache.as_ptr(), v_cache.as_ptr(),
            out.ptr(),
            cache_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
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
) {
    unsafe {
        crate::cuda_ffi::swa_backward_f32_cuda(
            q.as_ptr(), k.as_ptr(), v.as_ptr(),
            attn_weights.as_ptr(), d_attn_out.as_ptr(),
            d_q.ptr(), d_k.ptr(), d_v.ptr(),
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
        );
    }
}

/// Delta forward on device buffers.
#[cfg(feature = "cuda")]
pub fn delta_forward_dd(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_initial: &GpuSlice<f32>,
    m_states: &mut GpuBuf<f32>, y: &mut GpuBuf<f32>,
    seq_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::delta_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::delta_backward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), m_states.as_ptr(),
            d_y.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_initial.ptr(),
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::titans_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
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
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize,
) {
    unsafe {
        crate::cuda_ffi::hebbian_forward_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32,
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
    seq_len: usize, d: usize, checkpoint_interval: usize,
) {
    unsafe {
        crate::cuda_ffi::delta_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_initial.as_ptr(),
            m_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32,
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
    seq_len: usize, d: usize, checkpoint_interval: usize,
) {
    unsafe {
        crate::cuda_ffi::titans_forward_ckpt_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(), eta.as_ptr(),
            m_initial.as_ptr(), s_initial.as_ptr(),
            m_states.ptr(), s_states.ptr(), y.ptr(),
            seq_len as i32, d as i32, checkpoint_interval as i32,
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
    t_start: usize, t_end: usize, d: usize,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= d * d, "d_m_seed too small");
    unsafe {
        crate::cuda_ffi::delta_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), theta.as_ptr(),
            m_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_theta.ptr(), d_m_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
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
    t_start: usize, t_end: usize, d: usize,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= d * d, "d_m_seed too small");
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
    t_start: usize, t_end: usize, d: usize,
) {
    debug_assert!(t_start < t_end, "segment t_start={t_start} must be < t_end={t_end}");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(d_m_seed.len() >= d * d, "d_m_seed too small");
    unsafe {
        crate::cuda_ffi::hebbian_backward_segment_f32_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
            alpha.as_ptr(), m_states.as_ptr(), d_y.as_ptr(),
            d_m_seed.as_ptr(),
            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
            d_alpha.ptr(), d_m_out.ptr(),
            t_start as i32, t_end as i32, d as i32,
        );
    }
}

/// Synchronize the CUDA device (wait for all pending kernel launches).
#[cfg(feature = "cuda")]
pub fn cuda_sync() {
    let rc = unsafe { cudaDeviceSynchronize() };
    assert_eq!(rc, 0, "cudaDeviceSynchronize failed: error code {rc}");
}
