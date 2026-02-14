/// Feature-gated dispatch: Rust reference vs CUDA kernels.
///
/// Without `--features cuda` → Rust reference path (swa.rs).
/// With `--features cuda` → CUDA kernels via FFI (cuda_ffi.rs).
///
/// Both paths produce comparable results (verified by tests).
/// Dispatch is compile-time only — no runtime GPU detection overhead.
///
/// The CUDA path uses bf16 storage for Q/K/V/out/attn_weights with f32
/// compute. The dispatch layer converts between the f32 Rust world and
/// the bf16 CUDA world transparently.

/// SWA forward dispatch.
///
/// Calls either the Rust reference or CUDA kernel depending on feature gate.
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
        assert!(
            head_dim <= 32,
            "CUDA SWA kernels require head_dim <= 32 (one warp), got head_dim={head_dim}"
        );
        cuda_forward(q, k, v, out, attn_weights, seq_len, num_heads, head_dim, window_size);
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        crate::swa::swa_forward(q, k, v, out, attn_weights, seq_len, num_heads, head_dim, window_size);
    }
}

/// SWA backward dispatch.
///
/// Calls either the Rust reference or CUDA kernel depending on feature gate.
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
        assert!(
            head_dim <= 32,
            "CUDA SWA kernels require head_dim <= 32 (one warp), got head_dim={head_dim}"
        );
        cuda_backward(q, k, v, attn_weights, d_attn_out, d_q, d_k, d_v,
                      seq_len, num_heads, head_dim, window_size);
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        crate::swa::swa_backward_rust(q, k, v, attn_weights, d_attn_out, d_q, d_k, d_v,
                                       seq_len, num_heads, head_dim, window_size);
    }
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
