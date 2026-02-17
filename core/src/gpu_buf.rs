/// GPU-resident buffer: typed RAII wrapper for device memory.
///
/// `GpuBuf<T>` owns a contiguous allocation on the CUDA device. T must implement
/// `GpuElement` (sealed to f32 and u16/bf16). Drop calls cudaFree.
///
/// `GpuSlice<T>` provides a non-owning view into a GpuBuf — used for indexing
/// into large allocations like m_states[(seq_len+1)*d*d] by level offset.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                  count: usize, kind: i32) -> i32;
    fn cudaMemset(devPtr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
}

#[cfg(feature = "cuda")]
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
#[cfg(feature = "cuda")]
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
#[cfg(feature = "cuda")]
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// ══════════════════════════════════════════════════════════════════════
// GpuElement sealed trait
// ══════════════════════════════════════════════════════════════════════

/// Marker trait for types storable in GPU buffers. Sealed to f32 and u16.
#[cfg(feature = "cuda")]
pub trait GpuElement: Copy + Default + private::Sealed {
    /// Size in bytes of one element.
    fn byte_size() -> usize;
}

#[cfg(feature = "cuda")]
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

#[cfg(feature = "cuda")]
impl GpuElement for f32 {
    fn byte_size() -> usize { 4 }
}

#[cfg(feature = "cuda")]
impl GpuElement for u16 {
    fn byte_size() -> usize { 2 }
}

// ══════════════════════════════════════════════════════════════════════
// GpuBuf<T> — owning RAII device buffer
// ══════════════════════════════════════════════════════════════════════

/// Owning RAII wrapper for a device memory allocation of `len` elements of type T.
///
/// Allocated via cudaMalloc, freed via cudaFree on drop.
/// NOT Send/Sync — GPU pointers are bound to the CUDA context of the allocating thread.
#[cfg(feature = "cuda")]
pub struct GpuBuf<T: GpuElement> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: GpuElement> GpuBuf<T> {
    /// Allocate `len` elements of uninitialized device memory.
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "GpuBuf::new: len must be > 0");
        let bytes = len * T::byte_size();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, bytes) };
        assert_eq!(rc, 0, "cudaMalloc failed: error code {rc} (requested {bytes} bytes)");
        GpuBuf { ptr: ptr as *mut T, len }
    }

    /// Allocate and zero-initialize `len` elements.
    pub fn zeros(len: usize) -> Self {
        let buf = Self::new(len);
        buf.zero();
        buf
    }

    /// Allocate and copy host data to device.
    pub fn from_host(data: &[T]) -> Self {
        let buf = Self::new(data.len());
        buf.copy_from_host(data);
        buf
    }

    /// Raw device pointer (for passing to CUDA kernels / cuBLAS).
    pub fn ptr(&self) -> *mut T { self.ptr }

    /// Raw const device pointer.
    pub fn as_ptr(&self) -> *const T { self.ptr as *const T }

    /// Number of elements.
    pub fn len(&self) -> usize { self.len }

    /// Copy host slice to device. Lengths must match.
    pub fn copy_from_host(&self, src: &[T]) {
        assert_eq!(src.len(), self.len, "GpuBuf::copy_from_host: length mismatch ({} vs {})", src.len(), self.len);
        let bytes = self.len * T::byte_size();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy H2D failed: error code {rc}");
    }

    /// Copy device data to host slice. Lengths must match.
    pub fn copy_to_host(&self, dst: &mut [T]) {
        assert_eq!(dst.len(), self.len, "GpuBuf::copy_to_host: length mismatch ({} vs {})", dst.len(), self.len);
        let bytes = self.len * T::byte_size();
        let rc = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy D2H failed: error code {rc}");
    }

    /// Zero all bytes in the allocation (cudaMemset).
    pub fn zero(&self) {
        let bytes = self.len * T::byte_size();
        let rc = unsafe { cudaMemset(self.ptr as *mut std::ffi::c_void, 0, bytes) };
        assert_eq!(rc, 0, "cudaMemset failed: error code {rc}");
    }

    /// Non-owning slice view into a sub-range [offset..offset+len].
    /// Panics if out of bounds.
    pub fn slice(&self, offset: usize, len: usize) -> GpuSlice<T> {
        assert!(offset + len <= self.len,
            "GpuBuf::slice: offset({offset})+len({len}) > buf.len({})", self.len);
        GpuSlice {
            ptr: unsafe { self.ptr.add(offset) },
            len,
        }
    }

    /// Mutable slice view into a sub-range.
    pub fn slice_mut(&mut self, offset: usize, len: usize) -> GpuSliceMut<T> {
        assert!(offset + len <= self.len,
            "GpuBuf::slice_mut: offset({offset})+len({len}) > buf.len({})", self.len);
        GpuSliceMut {
            ptr: unsafe { self.ptr.add(offset) },
            len,
        }
    }

    /// Copy from another GpuBuf (device-to-device). Lengths must match.
    pub fn copy_from_device(&self, src: &GpuBuf<T>) {
        assert_eq!(src.len, self.len, "GpuBuf::copy_from_device: length mismatch");
        let bytes = self.len * T::byte_size();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.ptr as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy D2D failed: error code {rc}");
    }
}

#[cfg(feature = "cuda")]
impl<T: GpuElement> Drop for GpuBuf<T> {
    fn drop(&mut self) {
        let rc = unsafe { cudaFree(self.ptr as *mut std::ffi::c_void) };
        debug_assert_eq!(rc, 0, "cudaFree failed: error code {rc}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuSlice<T> — non-owning immutable view
// ══════════════════════════════════════════════════════════════════════

/// Non-owning immutable view into device memory. Does NOT free on drop.
/// Used for indexing into m_states by level offset without copying.
#[cfg(feature = "cuda")]
pub struct GpuSlice<T: GpuElement> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: GpuElement> GpuSlice<T> {
    pub fn as_ptr(&self) -> *const T { self.ptr as *const T }
    pub fn ptr(&self) -> *mut T { self.ptr }
    pub fn len(&self) -> usize { self.len }

    /// Copy this slice's device data to a host buffer.
    pub fn copy_to_host(&self, dst: &mut [T]) {
        assert_eq!(dst.len(), self.len);
        let bytes = self.len * T::byte_size();
        let rc = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy D2H (slice) failed: error code {rc}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuSliceMut<T> — non-owning mutable view
// ══════════════════════════════════════════════════════════════════════

/// Non-owning mutable view into device memory. Does NOT free on drop.
#[cfg(feature = "cuda")]
pub struct GpuSliceMut<T: GpuElement> {
    ptr: *mut T,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: GpuElement> GpuSliceMut<T> {
    pub fn as_ptr(&self) -> *const T { self.ptr as *const T }
    pub fn ptr(&self) -> *mut T { self.ptr }
    pub fn len(&self) -> usize { self.len }

    /// Copy host data into this slice.
    pub fn copy_from_host(&self, src: &[T]) {
        assert_eq!(src.len(), self.len);
        let bytes = self.len * T::byte_size();
        let rc = unsafe {
            cudaMemcpy(
                self.ptr as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(rc, 0, "cudaMemcpy H2D (slice_mut) failed: error code {rc}");
    }

    /// Zero all bytes in this slice.
    pub fn zero(&self) {
        let bytes = self.len * T::byte_size();
        let rc = unsafe { cudaMemset(self.ptr as *mut std::ffi::c_void, 0, bytes) };
        assert_eq!(rc, 0, "cudaMemset (slice_mut) failed: error code {rc}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// bf16 conversion helpers for GpuBuf<u16>
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
impl GpuBuf<u16> {
    /// Allocate bf16 buffer and upload f32 data (converting to bf16 on CPU).
    pub fn from_host_f32(data: &[f32]) -> Self {
        let bf16_data: Vec<u16> = data.iter().map(|&x| f32_to_bf16(x)).collect();
        Self::from_host(&bf16_data)
    }

    /// Download bf16 device data to f32 host slice.
    pub fn copy_to_host_f32(&self, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.len);
        let mut bf16_data = vec![0u16; self.len];
        self.copy_to_host(&mut bf16_data);
        for (d, &s) in dst.iter_mut().zip(bf16_data.iter()) {
            *d = bf16_to_f32(s);
        }
    }
}

/// Convert f32 → bf16 (round to nearest even), returning u16 bits.
#[cfg(feature = "cuda")]
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

/// Convert bf16 bits (u16) → f32.
#[cfg(feature = "cuda")]
#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

// ══════════════════════════════════════════════════════════════════════
// cublasSaxpy helper (for weight updates: W -= lr * grad)
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
extern "C" {
    fn cublasSaxpy_v2(
        handle: *mut std::ffi::c_void,
        n: i32,
        alpha: *const f32,
        x: *const f32, incx: i32,
        y: *mut f32, incy: i32,
    ) -> i32;
}

/// In-place GPU SAXPY: y += alpha * x. Used for weight update (W -= lr * grad → alpha = -lr).
#[cfg(feature = "cuda")]
pub fn gpu_saxpy(handle: *mut std::ffi::c_void, x: &GpuBuf<f32>, y: &mut GpuBuf<f32>, alpha: f32) {
    assert_eq!(x.len(), y.len(), "gpu_saxpy: length mismatch");
    let rc = unsafe {
        cublasSaxpy_v2(
            handle,
            x.len() as i32,
            &alpha,
            x.as_ptr(), 1,
            y.ptr(), 1,
        )
    };
    assert_eq!(rc, 0, "cublasSaxpy_v2 failed: error code {rc}");
}
