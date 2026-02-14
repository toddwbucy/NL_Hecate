// FFI declarations for CUDA SWA kernels.
//
// These functions are compiled by nvcc into machine code (.o files),
// linked by the `cc` crate at build time. They are opaque to Enzyme
// because nvcc produces SASS/PTX, not LLVM IR.
//
// Only available when compiled with `--features cuda`.

extern "C" {
    /// CUDA SWA forward kernel.
    ///
    /// All pointers must be device memory (allocated via cudaMalloc).
    /// Layout matches the Rust reference `swa::swa_forward()`:
    ///   q, k, v:       [seq_len, num_heads * head_dim] row-major
    ///   out:            [seq_len, num_heads * head_dim] row-major
    ///   attn_weights:   [num_heads, seq_len, window_size]
    pub(crate) fn swa_forward_f32_cuda(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        attn_weights: *mut f32,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        window_size: i32,
    );

    /// CUDA SWA backward kernel.
    ///
    /// Computes dQ, dK, dV from upstream d_attn_out and cached attn_weights.
    /// Layout matches `swa::swa_backward_rust()`.
    /// dQ, dK, dV must be pre-zeroed by the caller.
    pub(crate) fn swa_backward_f32_cuda(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        attn_weights: *const f32,
        d_attn_out: *const f32,
        d_q: *mut f32,
        d_k: *mut f32,
        d_v: *mut f32,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        window_size: i32,
    );
}
