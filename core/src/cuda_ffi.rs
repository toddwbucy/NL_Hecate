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
        batch_size: i32,
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
        batch_size: i32,
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

    // ── Checkpointed forward kernels (gradient checkpointing) ─────────

    /// DeltaRule forward with checkpoint_interval — stores M every C steps.
    pub(crate) fn delta_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32,
    );

    /// TitansLMM forward with checkpoint_interval — stores M/S every C steps.
    pub(crate) fn titans_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_states: *mut f32, s_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32,
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
        t_start: i32, t_end: i32, d: i32,
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
        t_start: i32, t_end: i32, d: i32,
    );

    /// HebbianRule segment backward — operates on [t_start, t_end) with d_m_seed.
    pub(crate) fn hebbian_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32,
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
    );

    /// DGD forward with checkpoint_interval — stores M every C steps.
    pub(crate) fn dgd_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32,
    );

    /// DGD segment backward — operates on [t_start, t_end) with d_m_seed.
    pub(crate) fn dgd_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32,
    );

    // ── M-norm clamp ──────────────────────────────────────────────────

    /// Frobenius-norm clamp for M state (called once per training step).
    /// No-op if m_norm_max <= 0 or >= 1e30.
    pub(crate) fn m_norm_clamp_f32_cuda(m: *mut f32, d: i32, m_norm_max: f32);

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

    /// Iterative GPU-side reduction of partial sums to a single scalar.
    /// Uses ping-pong between buf_a and buf_b. Result in buf_a[0].
    /// buf_b must be at least ceil(n/256) elements.
    pub(crate) fn reduce_sum_cuda(buf_a: *mut f32, buf_b: *mut f32, n: i32);

    /// Scale gradient buffer in-place: g[i] *= scale.
    pub(crate) fn grad_scale_cuda(g: *mut f32, scale: f32, n: i32) -> u32;  // cudaError_t: 0 = cudaSuccess

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
}
