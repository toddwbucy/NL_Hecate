// FFI declarations for CUDA kernels.
//
// These functions are compiled by nvcc into machine code (.o files),
// linked by the `cc` crate at build time. They are opaque to AD
// because nvcc produces SASS/PTX (compiled machine code).
//
// SWA kernels use bf16 storage (*const u16 / *mut u16).
// Memory rule kernels (Delta, Titans, Hebbian) use all-f32.
//
// Kernel-pair pattern: every kernel declared here has a corresponding
// CPU reference in Rust (swa.rs, backward.rs, delta_rule.rs, etc.).
// The CPU references are kept for verification — tests run both paths
// and assert element-wise agreement. See individual Rust files for
// which functions serve as the CPU counterpart.
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
        batch_size: i32,
        n_persistent: i32,
    );

    /// CUDA SWA backward kernel (bf16 inputs, f32 gradients).
    ///
    /// Computes dQ, dK, dV from upstream d_attn_out and cached attn_weights.
    /// Layout matches `swa::swa_backward_rust()`.
    /// Q/K/V/attn_weights are bf16; d_attn_out/dQ/dK/dV are f32.
    /// dQ, dK, dV must be pre-zeroed by the caller.
    /// CUDA SWA single-token attention for KV cache decode (bf16 storage, f32 compute).
    ///
    /// Q is [1, total_dim], K/V cache are [cache_len, total_dim], out is [1, total_dim].
    /// All bf16. No attn_weights output (inference only).
    pub(crate) fn swa_single_token_cuda(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        out: *mut u16,
        cache_len: i32,
        num_heads: i32,
        head_dim: i32,
        window_size: i32,
        n_persistent: i32,
    );

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
        batch_size: i32,
        n_persistent: i32,
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
        batch_size: i32,
        input_stride: i32,
        m_stride: i32,
        error_clip: f32,
        m_norm_max: f32,
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
        batch_size: i32,
        error_clip: f32,
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
        batch_size: i32,
        input_stride: i32,
        m_stride: i32,
        error_clip: f32,
        m_norm_max: f32,
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
        batch_size: i32,
        error_clip: f32,
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
        batch_size: i32,
        input_stride: i32,
        m_stride: i32,
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

    // ── Checkpointed forward kernels (gradient checkpointing) ─────────

    /// DeltaRule forward with checkpoint_interval — stores M every C steps.
    pub(crate) fn delta_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32, error_clip: f32,
        m_norm_max: f32,
    );

    /// TitansLMM forward with checkpoint_interval — stores M/S every C steps.
    pub(crate) fn titans_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_states: *mut f32, s_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32, error_clip: f32,
        m_norm_max: f32,
    );

    /// HebbianRule forward with checkpoint_interval — stores M every C steps.
    pub(crate) fn hebbian_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32,
    );

    // ── Segment backward kernels (gradient checkpointing) ───────────

    /// DeltaRule segment backward — operates on [t_start, t_end) with d_m_seed.
    pub(crate) fn delta_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32, batch_size: i32, seq_len: i32,
        error_clip: f32,
    );

    /// TitansLMM segment backward — operates on [t_start, t_end) with d_m_seed/d_s_seed.
    pub(crate) fn titans_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_states: *const f32, s_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32, d_s_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_eta: *mut f32,
        d_m_out: *mut f32, d_s_out: *mut f32,
        t_start: i32, t_end: i32, d: i32, batch_size: i32, seq_len: i32,
        error_clip: f32,
    );

    /// HebbianRule segment backward — operates on [t_start, t_end) with d_m_seed.
    pub(crate) fn hebbian_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32, batch_size: i32, seq_len: i32,
    );

    // ── DGD (Delta Gradient Descent) memory kernels (all f32) ────────

    /// CUDA DGD forward inner loop (all f32).
    ///
    /// Same math as Delta Rule (DGD generalizes it), but in a separate
    /// kernel file for future bias-agnostic extension (CS-33).
    /// Source: HOPE (2512.24695) Eq 88; core/src/dgd.rs
    pub(crate) fn dgd_forward_f32_cuda(
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
        batch_size: i32,
        input_stride: i32,
        m_stride: i32,
        error_clip: f32,
    );

    /// CUDA DGD backward inner loop (all f32).
    pub(crate) fn dgd_backward_f32_cuda(
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
        error_clip: f32,
    );

    /// DGD forward with checkpoint_interval — stores M every C steps.
    pub(crate) fn dgd_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32, error_clip: f32,
    );

    /// DGD segment backward — operates on [t_start, t_end) with d_m_seed.
    pub(crate) fn dgd_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32, batch_size: i32, seq_len: i32,
        error_clip: f32,
    );

    // ── Chunkwise kernels (spec 43 — frozen-M₀) ──────────────────────
    //
    // Paper-aligned formulation: errors computed against frozen chunk-start M₀.
    // Phase 1: error_t = M₀ @ k_t - v_t (frozen M₀, parallelizable)
    // Phase 2: M_t = (1-α)M_{t-1} - θ·outer(error_t, k_t), y_t = M_t @ q_t
    // Source: Titans eq-016/017, TNT eq-003/004, HOPE eq-090.

    /// Delta chunkwise forward (frozen-M₀). Stores (num_chunks+1) M states.
    pub(crate) fn delta_chunkwise_forward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_chunk_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, error_clip: f32,
        m_norm_max: f32,
    );

    /// Delta chunkwise backward (frozen-M₀). Key: d_M = (1-α)d_M only, no error chain.
    pub(crate) fn delta_chunkwise_backward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_chunk_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_initial: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, error_clip: f32,
        m_norm_max: f32,
    );

    /// Titans chunkwise forward (frozen-M₀). Stores (num_chunks+1) M and S states.
    pub(crate) fn titans_chunkwise_forward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_chunk_states: *mut f32, s_chunk_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, error_clip: f32,
        m_norm_max: f32,
    );

    /// Titans chunkwise backward (frozen-M₀). Three accumulators: d_M, d_S, d_M₀.
    pub(crate) fn titans_chunkwise_backward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_chunk_states: *const f32, s_chunk_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_eta: *mut f32,
        d_m_initial: *mut f32, d_s_initial: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, error_clip: f32,
        m_norm_max: f32,
    );

    // ── Spec 44: Phase 2 kernels + error_subtract_clip ─────────────────

    /// Batch error subtract + L2 clip: pred[i] -= v[i], then clip per row.
    pub(crate) fn error_subtract_clip_f32_cuda(
        predictions: *mut f32, v: *const f32,
        total_rows: i32, d: i32, error_clip: f32,
    );

    /// Delta Phase 2 forward: sequential M recurrence + readout for one chunk.
    pub(crate) fn delta_phase2_forward_f32_cuda(
        k_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        errors: *const f32, m_work: *mut f32,
        m_chunk_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, chunk_idx: i32,
        m_norm_max: f32,
    );

    /// Titans Phase 2 forward: sequential M+S recurrence + readout for one chunk.
    pub(crate) fn titans_phase2_forward_f32_cuda(
        k_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        errors: *const f32, m_work: *mut f32, s_work: *mut f32,
        m_chunk_states: *mut f32, s_chunk_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, chunk_idx: i32,
        m_norm_max: f32,
    );

    /// Delta Phase 2 backward: reverse token loop for one chunk.
    pub(crate) fn delta_phase2_backward_f32_cuda(
        k_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        errors: *const f32, m_chunk_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32,
        d_M: *mut f32, d_M0: *mut f32, m_recompute: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, chunk_idx: i32,
        m_norm_max: f32,
    );

    /// Titans Phase 2 backward: reverse token loop for one chunk.
    pub(crate) fn titans_phase2_backward_f32_cuda(
        k_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        errors: *const f32,
        m_chunk_states: *const f32, s_chunk_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_eta: *mut f32,
        d_M: *mut f32, d_S: *mut f32, d_M0: *mut f32,
        m_recompute: *mut f32, s_recompute: *mut f32,
        seq_len: i32, d: i32, batch_size: i32, chunk_size: i32, chunk_idx: i32,
        m_norm_max: f32,
    );

    // ── Broadcast fill (spec 27) ───────────────────────────────────────

    /// Fill dst[b*n_slots*dd + t*dd + i] = src[b*dd + i] for all (b,t,i).
    /// Used to broadcast M_final/S_final into full trajectory for proxy backward.
    pub(crate) fn broadcast_fill_f32_cuda(
        dst: *mut f32, src: *const f32,
        dd: i32, n_slots: i32, n_batch: i32,
    ) -> i32;

    // ── M-norm clamp ──────────────────────────────────────────────────

    /// Frobenius-norm clamp for M state (called once per training step).
    /// No-op if m_norm_max <= 0 or >= 1e30.
    pub(crate) fn m_norm_clamp_f32_cuda(m: *mut f32, d: i32, m_norm_max: f32);

    /// Spec 65: batched M-norm clamp — one launch for batch_size independent d×d matrices.
    /// m points to contiguous [batch_size, d*d] buffer.
    pub(crate) fn m_norm_clamp_batch_f32_cuda(m: *mut f32, d: i32, batch_size: i32, m_norm_max: f32);

    // ── DGD delta norm ───────────────────────────────────────────────

    /// Compute ‖M @ k - v‖₂ — the DGD prediction error norm.
    /// M is [d,d], k is [d], v is [d]. Writes scalar norm to norm_out[0].
    /// Source: HOPE (2512.24695) Eq 88 — error = M@k - v
    pub(crate) fn dgd_delta_norm_cuda(
        m: *const f32, k: *const f32, v: *const f32,
        norm_out: *mut f32, d: i32,
    );

    // ── LayerNorm ────────────────────────────────────────────────────

    /// LayerNorm forward: x_hat = (x - mean) / sqrt(var + eps), out = gamma * x_hat + beta.
    /// One block per position. Caches mean and rstd for backward.
    pub(crate) fn layer_norm_forward_cuda(
        x: *const f32, gamma: *const f32, beta: *const f32,
        out: *mut f32, mean_cache: *mut f32, rstd_cache: *mut f32,
        n: i32, d: i32, eps: f32,
    );

    /// LayerNorm backward: three-term formula for d_x, atomicAdd for d_gamma/d_beta.
    /// d_gamma and d_beta must be zeroed before call.
    pub(crate) fn layer_norm_backward_cuda(
        d_out: *const f32, x: *const f32,
        gamma: *const f32, mean_cache: *const f32, rstd_cache: *const f32,
        d_x: *mut f32, d_gamma: *mut f32, d_beta: *mut f32,
        n: i32, d: i32,
    );

    // ── L2 key/query normalization ──────────────────────────────────

    /// Normalize each row of x to unit L2 norm in-place.
    /// Stores pre-normalization norms in `norms` buffer.
    /// Titans paper (2501.00663): "normalize queries and keys using l_2-norm"
    pub(crate) fn l2_normalize_rows_f32_cuda(
        x: *mut f32, norms: *mut f32,
        n_rows: i32, d: i32, eps: f32,
    );

    /// Backward through L2 row normalization.
    /// d_in = (d_out - x_norm * dot(d_out, x_norm)) / max(norm, eps)
    pub(crate) fn l2_normalize_backward_f32_cuda(
        d_out: *const f32, x_norm: *const f32, norms: *const f32,
        d_in: *mut f32,
        n_rows: i32, d: i32, eps: f32,
    );

    // ── Embedding kernels ─────────────────────────────────────────────

    /// Gather rows from embedding table by token ID.
    pub(crate) fn embedding_gather_cuda(
        w_embed: *const f32,
        input_ids: *const i32,
        output: *mut f32,
        seq_len: i32,
        d: i32,
    );

    /// Scatter-add gradients back to embedding table rows (atomicAdd).
    pub(crate) fn embedding_scatter_add_cuda(
        d_embedded: *const f32,
        input_ids: *const i32,
        d_embed: *mut f32,
        seq_len: i32,
        d: i32,
    );

    /// Transpose copy: dst[col * rows + row] = src[row * cols + col].
    /// Used for weight tying: w_embed = w_unembed^T.
    pub(crate) fn transpose_copy_cuda(
        src: *const f32,   // [rows, cols] = w_unembed [d, vocab]
        dst: *mut f32,     // [cols, rows] = w_embed [vocab, d]
        rows: i32,         // d_model
        cols: i32,         // vocab_size
    );

    // ── Elementwise kernels ───────────────────────────────────────────

    /// Element-wise sigmoid: out[i] = 1/(1+exp(-x[i])).
    pub(crate) fn sigmoid_cuda(x: *const f32, out: *mut f32, n: i32);

    /// Element-wise multiply: out[i] = a[i] * b[i].
    pub(crate) fn elemwise_mul_cuda(a: *const f32, b: *const f32, out: *mut f32, n: i32);

    /// Gating backward: d_a[i] = d_out[i]*b[i], d_b[i] = d_out[i]*a[i].
    pub(crate) fn gating_backward_cuda(
        d_out: *const f32, a: *const f32, b: *const f32,
        d_a: *mut f32, d_b: *mut f32, n: i32,
    );

    /// Sigmoid backward: d_x[i] = d_gate[i] * gate[i] * (1-gate[i]).
    pub(crate) fn sigmoid_backward_cuda(
        d_gate: *const f32, gate: *const f32, d_x: *mut f32, n: i32,
    );

    /// f32 → bf16 conversion on GPU.
    pub(crate) fn f32_to_bf16_cuda(src: *const f32, dst: *mut u16, n: i32);

    /// bf16 → f32 conversion on GPU.
    pub(crate) fn bf16_to_f32_cuda(src: *const u16, dst: *mut f32, n: i32);

    /// Per-token gate computation: dot(concat(k,v), w) + bias → activation.
    /// activation: 0=sigmoid, 1=softplus.
    /// bias_ptr: device pointer to a [1]-element f32 buffer — read by kernel at launch.
    /// Using a device pointer instead of a scalar makes this kernel CUDA-graph-capture-safe:
    /// the graph captures the stable pointer; optimizer updates the value in-place.
    pub(crate) fn gate_compute_cuda(
        k_mem: *const f32, v_mem: *const f32, w_gate: *const f32,
        bias_ptr: *const f32, gate_out: *mut f32,
        seq_len: i32, d: i32, activation: i32,
    );

    /// CS-39 theta clamp (forward): clamp each element in-place to [lo, hi].
    /// No-op when lo == 0 and hi >= f32::MAX.
    pub(crate) fn clamp_f32_cuda(inout: *mut f32, n: i32, lo: f32, hi: f32);

    /// CS-39 theta clamp (backward): straight-through mask.
    /// Zeroes d_theta[i] when theta[i] <= lo or theta[i] >= hi.
    pub(crate) fn theta_clamp_mask_cuda(
        theta: *const f32, d_theta: *mut f32, n: i32, lo: f32, hi: f32,
    );

    /// Gate backward: accumulate d_w_alpha/theta/eta and d_b_alpha/theta/eta.
    /// has_theta=1 enables theta computation; has_eta=1 enables eta (Titans).
    /// When has_theta=0 or has_eta=0, corresponding pointer args are ignored.
    /// sigmoid(logit_theta) recovered as 1 - exp(-theta) — no logit cache needed.
    pub(crate) fn gate_backward_cuda(
        d_alpha: *const f32, alpha: *const f32,
        d_theta: *const f32, theta: *const f32,
        d_eta: *const f32,   eta: *const f32,
        k_mem: *const f32, v_mem: *const f32,
        d_w_alpha: *mut f32, d_b_alpha: *mut f32,
        d_w_theta: *mut f32, d_b_theta: *mut f32,
        d_w_eta: *mut f32,   d_b_eta: *mut f32,
        T: i32, D: i32, has_theta: i32, has_eta: i32,
    );

    /// Simple SAXPY: y[i] += alpha * x[i]. For small buffers.
    pub(crate) fn saxpy_cuda(alpha: f32, x: *const f32, y: *mut f32, n: i32);

    // ── TNT helper kernels (chunkwise parallelism glue) ─────────────────

    /// Broadcast global M to N contiguous copies for parallel local memories.
    pub(crate) fn tnt_broadcast_m_f32_cuda(
        m_src: *const f32,   // [d*d]
        m_dst: *mut f32,     // [N*d*d]
        n_local: i32,
        d: i32,
    );

    /// Mean-pool local outputs into shard summary vectors.
    pub(crate) fn tnt_shard_summary_mean_f32_cuda(
        local_y: *const f32,  // [shard_len, d]
        k_sum: *mut f32,      // [d]
        v_sum: *mut f32,      // [d]
        shard_len: i32,
        d: i32,
    );

    /// Update global M via outer product: M[i,j] = alpha*M[i,j] + v[i]*k[j].
    pub(crate) fn tnt_global_update_f32_cuda(
        global_m: *mut f32,    // [d*d]
        k_sum: *const f32,     // [d]
        v_sum: *const f32,     // [d]
        d: i32,
        alpha: f32,
    );

    /// Backward through global M update outer product.
    pub(crate) fn tnt_global_update_backward_f32_cuda(
        d_m_new: *const f32,   // [d*d]
        k_sum: *const f32,     // [d]
        v_sum: *const f32,     // [d]
        d_m_old: *mut f32,     // [d*d]
        d_k_sum: *mut f32,     // [d] — pre-zeroed
        d_v_sum: *mut f32,     // [d] — pre-zeroed
        d: i32,
        alpha: f32,
    );

    /// Backward through mean-pooling shard summary.
    pub(crate) fn tnt_shard_summary_mean_backward_f32_cuda(
        d_k_sum: *const f32,   // [d]
        d_v_sum: *const f32,   // [d]
        d_local_y: *mut f32,   // [shard_len, d] — accumulated
        shard_len: i32,
        d: i32,
    );

    /// Combine upstream + global gradient contributions (element-wise add).
    pub(crate) fn tnt_combine_gradients_f32_cuda(
        d_y_upstream: *const f32,
        d_y_global: *const f32,
        d_y_combined: *mut f32,
        n: i32,
    );

    // ── AdamW optimizer kernels ─────────────────────────────────────────

    /// Fused AdamW weight update. Updates w, m, v in-place on device.
    /// bc1_inv = 1/(1-beta1^t), bc2_inv = 1/(1-beta2^t) precomputed on host.
    pub(crate) fn adamw_update_cuda(
        w: *mut f32, g: *const f32,
        m: *mut f32, v: *mut f32,
        n: i32,
        lr: f32, beta1: f32, beta2: f32,
        eps: f32, bc1_inv: f32, bc2_inv: f32,
        weight_decay: f32,
    ) -> u32;  // cudaError_t: 0 = cudaSuccess

    /// Partial reduction for gradient L2 norm squared.
    /// Writes ceil(n/256) partial sums to partial_sums buffer.
    /// out_num_blocks receives the number of partials written.
    pub(crate) fn grad_norm_sq_cuda(
        g: *const f32, partial_sums: *mut f32,
        n: i32, out_num_blocks: *mut i32,
    ) -> u32;  // cudaError_t: 0 = cudaSuccess

    /// Partial reduction for dot product: sum of a[i]*b[i] per block (spec 53).
    /// Same pattern as grad_norm_sq_cuda. Writes ceil(n/256) partial sums.
    pub(crate) fn dot_product_partial_f32_cuda(
        a: *const f32, b: *const f32,
        partial_sums: *mut f32,
        n: i32, out_num_blocks: *mut i32,
    ) -> u32;  // cudaError_t: 0 = cudaSuccess

    /// Scale gradient buffer in-place: g[i] *= scale.
    pub(crate) fn grad_scale_cuda(g: *mut f32, scale: f32, n: i32) -> u32;  // cudaError_t: 0 = cudaSuccess

    /// Reduce partial norm sums on GPU + compute clip scale (spec 62).
    /// Single-block kernel: sums partial_sums[0..total_partials], writes
    /// L2 norm to out_norm and clip scale to out_scale.
    /// Scale = min(1.0, max_grad_norm / norm).
    pub(crate) fn reduce_partials_clip_cuda(
        partial_sums: *const f32, total_partials: i32,
        max_grad_norm: f32,
        out_norm: *mut f32, out_scale: *mut f32,
    ) -> u32;

    /// Conditional gradient scaling from device pointer (spec 62).
    /// Reads *scale_ptr; if >= 1.0, exits immediately. Otherwise g[i] *= scale.
    pub(crate) fn grad_scale_conditional_cuda(
        g: *mut f32, scale_ptr: *const f32, n: i32,
    ) -> u32;

    // ── M3 optimizer kernels (spec 34) ────────────────────────────────

    /// Fused M1 + V + conditional M2 EMA update.
    /// update_m2: 1 if step % chunk_size == 0, else 0.
    pub(crate) fn m3_ema_update_cuda(
        m1: *mut f32, m2: *mut f32, v: *mut f32, g: *const f32,
        n: i32,
        beta1: f32, beta2: f32, beta3: f32,
        update_m2: i32,
    ) -> u32;

    /// 1D param update: w -= lr * (m1 + alpha*m2) / (sqrt(v/bc2) + eps)
    pub(crate) fn m3_apply_1d_cuda(
        w: *mut f32, m1: *const f32, m2: *const f32, v: *const f32,
        n: i32,
        lr: f32, alpha: f32, eps: f32, bc2: f32,
    ) -> u32;

    /// 2D param update: w -= lr * (o1 + alpha*o2)  (o1, o2 already NS-orthogonalized)
    /// Currently unused — M3 uses split saxpy for M1/M2, but this fused kernel
    /// is available for future optimization (single launch instead of two saxpy).
    #[allow(dead_code)]
    pub(crate) fn m3_apply_2d_cuda(
        w: *mut f32, o1: *const f32, o2: *const f32,
        n: i32,
        lr: f32, alpha: f32,
    ) -> u32;

    /// Frobenius norm squared with partial reduction (same as grad_norm_sq but generic).
    pub(crate) fn frob_norm_sq_cuda(
        x: *const f32, partial_sums: *mut f32,
        n: i32, out_num_blocks: *mut i32,
    ) -> u32;

    /// Scale buffer in-place: x[i] *= scale.
    pub(crate) fn scale_buf_cuda(x: *mut f32, scale: f32, n: i32) -> u32;

    /// NS polynomial combination: x[i] = a*x[i] + b*y[i] + c*z[i].
    /// Used in Newton-Schulz iteration for Muon orthogonalization.
    pub(crate) fn m3_ns_poly_cuda(
        x: *mut f32, y: *const f32, z: *const f32,
        n: i32, a: f32, b: f32, c: f32,
    ) -> u32;

    // ── Cross-entropy kernels ─────────────────────────────────────────

    /// Fused softmax + NLL forward. Produces scalar loss via atomicAdd.
    /// loss_out must be pre-zeroed (kernel zeros it internally).
    pub(crate) fn cross_entropy_forward_cuda(
        logits: *const f32, target_ids: *const i32, loss_out: *mut f32,
        seq_len: i32, vocab: i32,
    );

    /// Cross-entropy backward: d_logits = (softmax - one_hot) / count.
    pub(crate) fn cross_entropy_backward_cuda(
        logits: *const f32, target_ids: *const i32, d_logits: *mut f32,
        seq_len: i32, vocab: i32, inv_count: f32,
    );

    // ── MLP memory kernels (MONETA + YAAD, all f32) ─────────────────────

    /// CUDA MONETA (l_p bias) forward inner loop (all f32).
    ///
    /// 2-layer MLP memory: W1[d_hidden, d], W2[d, d_hidden].
    /// Runs sequential recurrence with l_p attentional bias and L2/Lq retention.
    /// w1_states/w2_states: [(s+1)*w_size] trajectory storage.
    /// Source: MIRAS (2504.13173); core/src/moneta.rs
    pub(crate) fn mlp_forward_lp_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        w1_initial: *const f32, w2_initial: *const f32,
        w1_states: *mut f32, w2_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, d_hidden: i32,
        lp_p: f32, sign_sharpness: f32, lambda_2: f32, lq_q: f32,
    );

    /// CUDA YAAD (Huber bias) forward inner loop (all f32).
    ///
    /// Same MLP structure as MONETA but with Huber attentional bias
    /// and decoupled L2 retention (requires boundary snapshots).
    /// Source: MIRAS (2504.13173); core/src/yaad.rs
    pub(crate) fn mlp_forward_huber_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        w1_initial: *const f32, w2_initial: *const f32,
        w1_boundary: *const f32, w2_boundary: *const f32,
        w1_states: *mut f32, w2_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, d_hidden: i32,
        huber_delta: f32, lambda_local: f32, lambda_2: f32,
    );

    /// CUDA MONETA (l_p bias) backward inner loop (all f32, q=2 only).
    ///
    /// Produces gradients on k_mem, v_mem, q_mem, alpha, theta, w1_initial, w2_initial.
    /// LQ backward (q > 2) is deferred — current kernel only supports L2 retention.
    pub(crate) fn mlp_backward_lp_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        w1_states: *const f32, w2_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32,
        d_w1_initial: *mut f32, d_w2_initial: *mut f32,
        seq_len: i32, d: i32, d_hidden: i32,
        lp_p: f32, sign_sharpness: f32, lambda_2: f32, lq_q: f32,
    );

    /// CUDA YAAD (Huber bias) backward inner loop (all f32).
    ///
    /// Same structure as MONETA backward but with Huber gradient
    /// and decoupled L2 retention (uses boundary snapshots).
    pub(crate) fn mlp_backward_huber_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        w1_states: *const f32, w2_states: *const f32,
        w1_boundary: *const f32, w2_boundary: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32,
        d_w1_initial: *mut f32, d_w2_initial: *mut f32,
        seq_len: i32, d: i32, d_hidden: i32,
        huber_delta: f32, lambda_local: f32, lambda_2: f32,
    );

    // ── TitansLMM MLP Memory forward (Spec 75 Phase B) ────────────────
    // Deep neural memory: M = {W₁, b₁, W₂, b₂} packed flat buffer.
    // Fused forward+update kernel with EMA momentum, L2 bias, L2 retention.
    // Batch support via grid dimension (batch_size parallel heads/levels).
    // No cudaDeviceSynchronize inside — caller syncs.

    /// TitansLMM MLP memory forward inner loop (all f32).
    ///
    /// Packed L_M=2 buffer: W1[d_h,d], b1[d_h], W2[d,d_h], b2[d].
    /// Sequential token recurrence with fused gradient + EMA momentum + retention.
    /// activation: 0=GELU, 1=SiLU, 2=ReLU.
    /// Source: Titans (2501.00663) Eqs 12-15; core/src/titans_lmm.rs (step_mlp())
    pub(crate) fn titans_mlp_forward_f32_cuda(
        k_mem: *const f32,       // [batch_size, seq_len, d]
        v_mem: *const f32,       // [batch_size, seq_len, d]
        q_mem: *const f32,       // [batch_size, seq_len, d]
        alpha: *const f32,       // [batch_size, seq_len]
        theta: *const f32,       // [batch_size, seq_len]
        eta: *const f32,         // [batch_size, seq_len]
        m_initial: *const f32,   // [batch_size, state_size]
        s_initial: *const f32,   // [batch_size, state_size]
        m_states: *mut f32,      // [batch_size, (seq_len+1)*state_size]
        s_states: *mut f32,      // [batch_size, (seq_len+1)*state_size]
        y: *mut f32,             // [batch_size, seq_len, d]
        seq_len: i32,
        d: i32,
        d_hidden: i32,
        batch_size: i32,
        input_stride: i32,       // seq_len * d
        m_stride: i32,           // (seq_len+1) * state_size
        activation: i32,         // 0=GELU, 1=SiLU, 2=ReLU
        m_norm_max: f32,
    );

    // ── TitansLMM MLP Memory backward (Spec 75 Phase C) ─────────────
    // Analytical backward through deep neural memory with EMA momentum.
    // Reverse token loop computing d_k, d_v, d_q, d_alpha, d_theta,
    // d_eta, d_m_initial, d_s_initial.
    // No cudaDeviceSynchronize inside — caller syncs.

    /// TitansLMM MLP memory backward inner loop (all f32).
    ///
    /// Requires m_states/s_states from forward pass (full trajectory).
    /// activation: 0=GELU, 1=SiLU, 2=ReLU.
    /// Source: Titans (2501.00663) Eqs 12-15; core/src/titans_lmm.rs (mlp_inner_backward)
    pub(crate) fn titans_mlp_backward_f32_cuda(
        k_mem: *const f32,       // [batch_size, seq_len, d]
        v_mem: *const f32,       // [batch_size, seq_len, d]
        q_mem: *const f32,       // [batch_size, seq_len, d]
        alpha: *const f32,       // [batch_size, seq_len]
        theta: *const f32,       // [batch_size, seq_len]
        eta: *const f32,         // [batch_size, seq_len]
        m_states: *const f32,    // [batch_size, (seq_len+1)*state_size]
        s_states: *const f32,    // [batch_size, (seq_len+1)*state_size]
        d_y: *const f32,         // [batch_size, seq_len, d]
        d_k_mem: *mut f32,       // [batch_size, seq_len, d]
        d_v_mem: *mut f32,       // [batch_size, seq_len, d]
        d_q_mem: *mut f32,       // [batch_size, seq_len, d]
        d_alpha: *mut f32,       // [batch_size, seq_len]
        d_theta: *mut f32,       // [batch_size, seq_len]
        d_eta: *mut f32,         // [batch_size, seq_len]
        d_m_initial: *mut f32,   // [batch_size, state_size]
        d_s_initial: *mut f32,   // [batch_size, state_size]
        seq_len: i32,
        d: i32,
        d_hidden: i32,
        batch_size: i32,
        input_stride: i32,       // seq_len * d
        m_stride: i32,           // (seq_len+1) * state_size
        activation: i32,         // 0=GELU, 1=SiLU, 2=ReLU
        m_norm_max: f32,
    );

    // ── SwiGLU MLP device-to-device kernels ────────────────────────────
    // Zero-PCIe variants used in the GPU training path. All pointers are
    // device memory. Caller provides pre-allocated GpuBuf<f32> buffers.
    // No cudaDeviceSynchronize inside — caller syncs via dispatch::cuda_sync().

    /// SwiGLU forward (device-to-device, no H2D/D2H, no persistent pool).
    ///
    /// X, gate_proj, up_proj, down_proj: device inputs.
    /// Y: device output [seq_len × d_model].
    /// gate_buf, up_buf, fused_buf, cache_buf: device scratch saved for backward.
    pub(crate) fn swiglu_forward_f32_cuda_dd(
        x: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        y: *mut f32,
        gate_buf: *mut f32,
        up_buf: *mut f32,
        fused_buf: *mut f32,
        cache_buf: *mut f32,
        seq_len: i32,
        d_model: i32,
        intermediate: i32,
    );

    /// SwiGLU backward (device-to-device, no H2D/D2H).
    ///
    /// All forward activations (fused_buf, gate_buf, up_buf, cache_buf) are
    /// device pointers from GpuMemoryCache::SwiGlu. Output grads
    /// d_x, d_gate_proj, d_up_proj, d_down_proj written with beta=0 (fresh).
    pub(crate) fn swiglu_backward_f32_cuda_dd(
        d_y: *const f32,
        x: *const f32,
        gate_proj: *const f32,
        up_proj: *const f32,
        down_proj: *const f32,
        fused_buf: *const f32,
        gate_buf: *const f32,
        up_buf: *const f32,
        cache_buf: *const f32,
        d_x: *mut f32,
        d_gate_proj: *mut f32,
        d_up_proj: *mut f32,
        d_down_proj: *mut f32,
        seq_len: i32,
        d_model: i32,
        intermediate: i32,
    );

    // ── Fused memory forward kernels (spec 39) ───────────────────────
    //
    // Fuse L2 normalization + gate compute + gate clamping into the
    // main DGD/Titans recurrence kernel. Eliminates 4-5 separate launches
    // and intermediate buffer round-trips per level.

    /// Fused DGD forward: normalize + gates + DGD recurrence in one kernel.
    /// k_mem and q_mem are normalized IN-PLACE (backward needs normalized values).
    /// alpha_out, theta_out, k_norms_out, q_norms_out written for backward.
    pub(crate) fn dgd_fused_forward_f32_cuda(
        k_mem: *mut f32,           // [bs*s, d] — normalized in-place
        v_mem: *const f32,         // [bs*s, d]
        q_mem: *mut f32,           // [bs*s, d] — normalized in-place
        w_alpha: *const f32,       // [2*d]
        b_alpha_ptr: *const f32,   // [1]
        w_theta: *const f32,       // [2*d]
        b_theta_ptr: *const f32,   // [1]
        alpha_floor: f32,
        alpha_ceil: f32,
        theta_floor: f32,
        theta_ceil: f32,
        m_initial: *const f32,     // [bs, d*d]
        m_states: *mut f32,        // [bs, (s+1)*d*d]
        y: *mut f32,               // [bs, s, d]
        alpha_out: *mut f32,       // [bs*s]
        theta_out: *mut f32,       // [bs*s]
        k_norms_out: *mut f32,     // [bs*s]
        q_norms_out: *mut f32,     // [bs*s]
        seq_len: i32,
        d: i32,
        batch_size: i32,
        error_clip: f32,
    );

    /// Fused Titans forward: normalize + gates (alpha/theta/eta) + Titans recurrence.
    /// Same as DGD fused but adds eta gate and momentum S.
    pub(crate) fn titans_fused_forward_f32_cuda(
        k_mem: *mut f32,           // [bs*s, d] — normalized in-place
        v_mem: *const f32,         // [bs*s, d]
        q_mem: *mut f32,           // [bs*s, d] — normalized in-place
        w_alpha: *const f32,       // [2*d]
        b_alpha_ptr: *const f32,   // [1]
        w_theta: *const f32,       // [2*d]
        b_theta_ptr: *const f32,   // [1]
        w_eta: *const f32,         // [2*d]
        b_eta_ptr: *const f32,     // [1]
        alpha_floor: f32,
        alpha_ceil: f32,
        theta_floor: f32,
        theta_ceil: f32,
        m_initial: *const f32,     // [bs, d*d]
        s_initial: *const f32,     // [bs, d*d]
        m_states: *mut f32,        // [bs, (s+1)*d*d]
        s_states: *mut f32,        // [bs, (s+1)*d*d]
        y: *mut f32,               // [bs, s, d]
        alpha_out: *mut f32,       // [bs*s]
        theta_out: *mut f32,       // [bs*s]
        eta_out: *mut f32,         // [bs*s]
        k_norms_out: *mut f32,     // [bs*s]
        q_norms_out: *mut f32,     // [bs*s]
        seq_len: i32,
        d: i32,
        batch_size: i32,
        error_clip: f32,
        m_norm_max: f32,
    );

    // ── Per-head memory transpose/broadcast (Spec 45) ────────────────

    /// Transpose [bs, s, nh*hd] → [bs*nh, s, hd] (forward=1) or reverse (forward=0).
    /// Zero-copy layout conversion for per-head memory kernels.
    pub(crate) fn transpose_heads_f32_cuda(
        input: *const f32,
        output: *mut f32,
        bs: i32, s: i32, nh: i32, hd: i32,
        forward: i32,
    );

    /// Broadcast [bs, s] → [bs*nh, s] by repeating each batch's gate values nh times.
    pub(crate) fn broadcast_heads_f32_cuda(
        input: *const f32,
        output: *mut f32,
        bs: i32, s: i32, nh: i32,
    );

    /// Sum [bs*nh, s] → [bs, s] across heads (backward of broadcast_heads).
    /// Spec 45: reduces per-head gate gradients back to position-level.
    pub(crate) fn sum_heads_f32_cuda(
        input: *const f32,
        output: *mut f32,
        bs: i32, s: i32, nh: i32,
    );

    // ── Pool/upsample operations (Spec 46: CMS token reduction) ─────────

    /// Mean-pool C consecutive d-dimensional vectors into their average.
    /// x: [bs*s, d] → out: [bs*(s/C), d]
    pub(crate) fn mean_pool_1d_f32_cuda(
        x: *const f32, out: *mut f32,
        bs: i32, s: i32, d: i32, C: i32,
    );

    /// Repeat each vector C times: x: [bs*(s/C), d] → out: [bs*s, d]
    pub(crate) fn repeat_upsample_1d_f32_cuda(
        x: *const f32, out: *mut f32,
        bs: i32, s: i32, d: i32, C: i32,
    );

    /// Backward of mean_pool: broadcast gradient / C.
    /// d_out: [bs*(s/C), d] → d_x: [bs*s, d] (accumulated)
    pub(crate) fn mean_pool_1d_backward_f32_cuda(
        d_out: *const f32, d_x: *mut f32,
        bs: i32, s: i32, d: i32, C: i32,
    );

    /// Backward of repeat_upsample: sum groups of C.
    /// d_out: [bs*s, d] → d_x: [bs*(s/C), d]
    pub(crate) fn repeat_upsample_1d_backward_f32_cuda(
        d_out: *const f32, d_x: *mut f32,
        bs: i32, s: i32, d: i32, C: i32,
    );
}
