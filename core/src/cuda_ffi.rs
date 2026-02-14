// FFI declarations for CUDA SWA kernels.
//
// These functions are compiled by nvcc into machine code (.o files),
// linked by the `cc` crate at build time. They are opaque to Enzyme
// because nvcc produces SASS/PTX, not LLVM IR.
//
// bf16 pointers use *const u16 / *mut u16 in Rust FFI, corresponding
// to __nv_bfloat16* on the CUDA side (both are 16-bit types).
// Gradient pointers (d_attn_out, d_q, d_k, d_v) remain f32.
//
// Only available when compiled with `--features cuda`.

extern "C" {
    /// CUDA SWA forward kernel (bf16 storage, f32 compute).
    ///
    /// All pointers must be device memory (allocated via cudaMalloc).
    /// Layout matches the Rust reference `swa::swa_forward()`:
    ///   q, k, v:       [seq_len, num_heads * head_dim] row-major (bf16)
    ///   out:            [seq_len, num_heads * head_dim] row-major (bf16)
    ///   attn_weights:   [num_heads, seq_len, window_size] (bf16)
    pub(crate) fn swa_forward_f32_cuda(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        out: *mut u16,
        attn_weights: *mut u16,
        seq_len: i32,
        num_heads: i32,
        head_dim: i32,
        window_size: i32,
    );

    /// CUDA SWA backward kernel (bf16 inputs, f32 gradients).
    ///
    /// Computes dQ, dK, dV from upstream d_attn_out and cached attn_weights.
    /// Layout matches `swa::swa_backward_rust()`.
    /// Q/K/V/attn_weights are bf16; d_attn_out/dQ/dK/dV are f32.
    /// dQ, dK, dV must be pre-zeroed by the caller.
    pub(crate) fn swa_backward_f32_cuda(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        attn_weights: *const u16,
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
