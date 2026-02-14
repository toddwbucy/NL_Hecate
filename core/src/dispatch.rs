/// Feature-gated dispatch: Rust reference vs CUDA kernels.
///
/// Without `--features cuda` → Rust reference path (swa.rs).
/// With `--features cuda` → CUDA kernels via FFI (cuda_ffi.rs).
///
/// Both paths produce identical results (verified by tests).
/// Dispatch is compile-time only — no runtime GPU detection overhead.

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

/// RAII wrapper for device memory allocation.
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
    fn zero(&self) {
        let zeros = vec![0.0f32; self.len];
        self.copy_from_host(&zeros);
    }
}

#[cfg(feature = "cuda")]
impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr as *mut std::ffi::c_void); }
    }
}

#[cfg(feature = "cuda")]
fn cuda_forward(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32], attn_weights: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    let total = seq_len * num_heads * head_dim;
    let aw_total = num_heads * seq_len * window_size;

    let d_q = DevBuf::new(total);
    let d_k = DevBuf::new(total);
    let d_v = DevBuf::new(total);
    let d_out = DevBuf::new(total);
    let d_aw = DevBuf::new(aw_total);

    d_q.copy_from_host(q);
    d_k.copy_from_host(k);
    d_v.copy_from_host(v);
    d_out.zero();
    d_aw.zero();

    unsafe {
        crate::cuda_ffi::swa_forward_f32_cuda(
            d_q.ptr, d_k.ptr, d_v.ptr,
            d_out.ptr, d_aw.ptr,
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
        );
        cudaDeviceSynchronize();
    }

    d_out.copy_to_host(out);
    d_aw.copy_to_host(attn_weights);
}

#[cfg(feature = "cuda")]
fn cuda_backward(
    q: &[f32], k: &[f32], v: &[f32],
    attn_weights: &[f32], d_attn_out: &[f32],
    dq_host: &mut [f32], dk_host: &mut [f32], dv_host: &mut [f32],
    seq_len: usize, num_heads: usize, head_dim: usize, window_size: usize,
) {
    let total = seq_len * num_heads * head_dim;
    let aw_total = num_heads * seq_len * window_size;

    let d_q = DevBuf::new(total);
    let d_k = DevBuf::new(total);
    let d_v = DevBuf::new(total);
    let d_aw = DevBuf::new(aw_total);
    let d_dao = DevBuf::new(total);
    let d_dq = DevBuf::new(total);
    let d_dk = DevBuf::new(total);
    let d_dv = DevBuf::new(total);

    d_q.copy_from_host(q);
    d_k.copy_from_host(k);
    d_v.copy_from_host(v);
    d_aw.copy_from_host(attn_weights);
    d_dao.copy_from_host(d_attn_out);
    // Zero the output gradient buffers
    d_dq.copy_from_host(dq_host);
    d_dk.copy_from_host(dk_host);
    d_dv.copy_from_host(dv_host);

    unsafe {
        crate::cuda_ffi::swa_backward_f32_cuda(
            d_q.ptr, d_k.ptr, d_v.ptr,
            d_aw.ptr, d_dao.ptr,
            d_dq.ptr, d_dk.ptr, d_dv.ptr,
            seq_len as i32, num_heads as i32, head_dim as i32, window_size as i32,
        );
        cudaDeviceSynchronize();
    }

    d_dq.copy_to_host(dq_host);
    d_dk.copy_to_host(dk_host);
    d_dv.copy_to_host(dv_host);
}
