// FFI declarations for CUDA kernels.
//
// These functions are compiled by nvcc into machine code (.o files),
// linked by the `cc` crate at build time. They are opaque to Enzyme
// because nvcc produces SASS/PTX, not LLVM IR.
//
// SWA kernels use bf16 storage (*const u16 / *mut u16).
// Memory rule kernels (Delta, Titans, Hebbian) use all-f32.
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

    // ── Delta Rule memory kernels (all f32) ─────────────────────────

    /// CUDA DeltaRule forward inner loop (all f32).
    ///
    /// Takes pre-computed projections (k_mem, v_mem, q_mem) and gates (alpha, theta).
    /// Runs the sequential M recurrence + readout in CUDA.
    /// All pointers must be device memory.
    pub(crate) fn delta_forward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        theta: *const f32,
        m_initial: *const f32,
        m_states: *mut f32,
        y: *mut f32,
        seq_len: i32,
        d: i32,
    );

    /// CUDA DeltaRule backward inner loop (all f32).
    ///
    /// Takes cached m_states and upstream d_y, produces gradients on
    /// k_mem, v_mem, q_mem, alpha, theta, and m_initial.
    /// d_k_mem, d_v_mem, d_q_mem must be pre-zeroed.
    pub(crate) fn delta_backward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        theta: *const f32,
        m_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32,
        d_v_mem: *mut f32,
        d_q_mem: *mut f32,
        d_alpha: *mut f32,
        d_theta: *mut f32,
        d_m_initial: *mut f32,
        seq_len: i32,
        d: i32,
    );

    // ── Titans LMM memory kernels (all f32) ─────────────────────────

    /// CUDA TitansLMM forward inner loop (all f32).
    ///
    /// Extends Delta with momentum accumulator S and gate eta.
    pub(crate) fn titans_forward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        theta: *const f32,
        eta: *const f32,
        m_initial: *const f32,
        s_initial: *const f32,
        m_states: *mut f32,
        s_states: *mut f32,
        y: *mut f32,
        seq_len: i32,
        d: i32,
    );

    /// CUDA TitansLMM backward inner loop (all f32).
    pub(crate) fn titans_backward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        theta: *const f32,
        eta: *const f32,
        m_states: *const f32,
        s_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32,
        d_v_mem: *mut f32,
        d_q_mem: *mut f32,
        d_alpha: *mut f32,
        d_theta: *mut f32,
        d_eta: *mut f32,
        d_m_initial: *mut f32,
        d_s_initial: *mut f32,
        seq_len: i32,
        d: i32,
    );

    // ── Hebbian Rule memory kernels (all f32) ───────────────────────

    /// CUDA HebbianRule forward inner loop (all f32).
    ///
    /// Simplest rule: no error term, no theta gate.
    /// M = (1-alpha)*M + outer(v,k).
    pub(crate) fn hebbian_forward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        m_initial: *const f32,
        m_states: *mut f32,
        y: *mut f32,
        seq_len: i32,
        d: i32,
    );

    /// CUDA HebbianRule backward inner loop (all f32).
    pub(crate) fn hebbian_backward_f32_cuda(
        k_mem: *const f32,
        v_mem: *const f32,
        q_mem: *const f32,
        alpha: *const f32,
        m_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32,
        d_v_mem: *mut f32,
        d_q_mem: *mut f32,
        d_alpha: *mut f32,
        d_m_initial: *mut f32,
        seq_len: i32,
        d: i32,
    );
}
