// FFI declarations for CUDA kernels.
//
// These functions are compiled by nvcc into machine code (.o files),
// linked by the `cc` crate at build time. They are opaque to AD
// because nvcc produces SASS/PTX (compiled machine code).
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

    /// Element-wise softplus: out[i] = log(1+exp(x[i])).
    pub(crate) fn softplus_cuda(x: *const f32, out: *mut f32, n: i32);

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
    pub(crate) fn gate_compute_cuda(
        k_mem: *const f32, v_mem: *const f32, w_gate: *const f32,
        bias: f32, gate_out: *mut f32,
        seq_len: i32, d: i32, activation: i32,
    );

    /// Simple SAXPY: y[i] += alpha * x[i]. For small buffers.
    pub(crate) fn saxpy_cuda(alpha: f32, x: *const f32, y: *mut f32, n: i32);

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
    );

    /// Partial reduction for gradient L2 norm squared.
    /// Writes ceil(n/256) partial sums to partial_sums buffer.
    /// out_num_blocks receives the number of partials written.
    pub(crate) fn grad_norm_sq_cuda(
        g: *const f32, partial_sums: *mut f32,
        n: i32, out_num_blocks: *mut i32,
    );

    /// Iterative GPU-side reduction of partial sums to a single scalar.
    /// Uses ping-pong between buf_a and buf_b. Result in buf_a[0].
    /// buf_b must be at least ceil(n/256) elements.
    pub(crate) fn reduce_sum_cuda(buf_a: *mut f32, buf_b: *mut f32, n: i32);

    /// Scale gradient buffer in-place: g[i] *= scale.
    pub(crate) fn grad_scale_cuda(g: *mut f32, scale: f32, n: i32);

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
}
