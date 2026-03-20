/// GPU-resident stacked multi-block backward pass.
///
/// Reverses through blocks in order n_blocks-1..0, computing gradients
/// per block and accumulating through the residual stream.
///
/// Reuses existing CUDA backward kernels from gpu_backward.rs.
///
/// Spec: specs/infrastructure/14_multi_block_stacking.md
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::GpuStackedParams;
#[cfg(feature = "cuda")]
use crate::gpu_stacked_forward::{GpuStackedCache, GpuBufClone};
#[cfg(feature = "cuda")]
use crate::gpu_backward::{GpuLevelGrads, gpu_memory_backward, gpu_memory_read_only_backward, gpu_matmul_transa_dd};
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind, HopeVariant};

// ══════════════════════════════════════════════════════════════════════
// GpuStackedBlockGrads — per-block gradient buffers
// ══════════════════════════════════════════════════════════════════════

/// Gradient buffers for one block.
#[cfg(feature = "cuda")]
pub struct GpuStackedBlockGrads {
    pub d_w_q: GpuBuf<f32>,
    pub d_w_k: GpuBuf<f32>,
    pub d_w_v: GpuBuf<f32>,
    pub d_w_o: GpuBuf<f32>,
    pub d_ln_attn_gamma: GpuBuf<f32>,
    pub d_ln_attn_beta: GpuBuf<f32>,
    pub d_ln_mem_gamma: GpuBuf<f32>,
    pub d_ln_mem_beta: GpuBuf<f32>,
    pub levels: Vec<GpuLevelGrads>,
    /// Gradient for learnable level aggregation weights. Length k.
    pub d_alpha_mem: Vec<f32>,
    /// Per-level L2 norm of d_y_combined (output gradient entering each level's
    /// backward). Length k. 0.0 for inactive levels. Used by tape diagnostics.
    pub level_output_gnorms: Vec<f32>,
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedGrads — full stacked gradient buffers
// ══════════════════════════════════════════════════════════════════════

/// Gradient buffers for the full stacked model.
#[cfg(feature = "cuda")]
pub struct GpuStackedGrads {
    // Shared parameter gradients
    pub d_w_embed: GpuBuf<f32>,
    pub d_w_unembed: GpuBuf<f32>,
    pub d_ln_final_gamma: GpuBuf<f32>,
    pub d_ln_final_beta: GpuBuf<f32>,
    // Per-block gradients
    pub blocks: Vec<GpuStackedBlockGrads>,
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_backward — main entry point
// ══════════════════════════════════════════════════════════════════════

/// Stacked multi-block backward pass.
///
/// Flow (reverse of forward):
///   1. Cross-entropy backward → d_logits
///   2. Unembed backward → d_ln_final_out + d_w_unembed
///   3. Final LN backward → d_residual_stream
///   4. For each block b in (n_blocks-1)..0:
///      a. Residual skip 2 backward → d_residual_after_attn + d_y_combined
///      b. Memory backward per level → d_ln_mem_out
///      c. LN_mem backward → d_residual_after_attn update
///      d. Residual skip 1 backward → d_block_input + d_attn_out
///      e. SWA backward → d_q, d_k, d_v
///      f. QKV backward → d_ln_attn_out + d_w_q/k/v
///      g. LN_attn backward → d_block_input update
///   5. Embedding scatter-add → d_w_embed
#[cfg(feature = "cuda")]
pub fn gpu_stacked_backward(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    cache: &GpuStackedCache,
) -> GpuStackedGrads {
    let s = cache.s;
    let d = cache.d;
    let v = cache.v;
    let nh = cache.nh;
    let hd = cache.hd;
    let ws = cache.ws;
    let bs = cache.batch_size;
    let n_tokens = bs * s;
    let bsd = n_tokens * d;
    let bsv = n_tokens * v;
    let bsd_i32 = bsd as i32;
    let n_blocks = params.n_blocks();

    let inter = if cfg.memory_rule == MemoryRuleKind::SwiGluMlp { cfg.intermediate_size } else { 0 };

    // Initialize shared gradient buffers
    let d_w_embed = GpuBuf::zeros(v * d);
    let mut d_w_unembed = GpuBuf::zeros(d * v);
    let d_ln_final_gamma = GpuBuf::zeros(d);
    let d_ln_final_beta = GpuBuf::zeros(d);

    // ── Cross-entropy backward ─────────────────────────────────────────
    let d_logits = GpuBuf::zeros(bsv);
    let valid_count = cache.target_ids_i32.iter()
        .filter(|&&t| t >= 0 && (t as usize) < v)
        .count() as f32;
    let count = if valid_count > 0.0 { valid_count } else { 1.0 };
    unsafe {
        crate::cuda_ffi::cross_entropy_backward_cuda(
            cache.logits.as_ptr(),
            cache.target_ids_gpu.ptr() as *const i32,
            d_logits.ptr(),
            n_tokens as i32, v as i32,
            1.0 / count,
        );
    }

    // ── Unembed backward ───────────────────────────────────────────────
    let mut d_ln_final_out = GpuBuf::zeros(bsd);
    crate::dispatch::cublas_matmul_transb_dd(
        &d_logits, &params.w_unembed, &mut d_ln_final_out, n_tokens, v, d, 0.0,
    );
    gpu_matmul_transa_dd(
        &cache.ln_final_out, &d_logits, &mut d_w_unembed,
        d, n_tokens, v,
    );

    // ── Final LN backward ──────────────────────────────────────────────
    // The last block's output (residual_stream) went through ln_final.
    // Reconstruct: residual_out = block_input + gated_out
    //   where gated_out = attn_proj * gate (MAG sigmoid gating)
    let last_block_cache = &cache.block_caches[n_blocks - 1];
    let residual_stream_final = GpuBuf::zeros(bsd);
    let gated_out_last = GpuBuf::zeros(bsd);
    unsafe {
        crate::cuda_ffi::elemwise_mul_cuda(
            last_block_cache.attn_proj.as_ptr(), last_block_cache.gate.as_ptr(),
            gated_out_last.ptr(), bsd_i32,
        );
        crate::cuda_ffi::saxpy_cuda(1.0, last_block_cache.block_input.as_ptr(), residual_stream_final.ptr(), bsd_i32);
        crate::cuda_ffi::saxpy_cuda(1.0, gated_out_last.as_ptr(), residual_stream_final.ptr(), bsd_i32);
    }

    let mut d_residual_stream = GpuBuf::zeros(bsd);
    unsafe {
        crate::cuda_ffi::layer_norm_backward_cuda(
            d_ln_final_out.as_ptr(),
            residual_stream_final.as_ptr(),
            params.ln_final_gamma.as_ptr(),
            cache.ln_final_mean.as_ptr(),
            cache.ln_final_rstd.as_ptr(),
            d_residual_stream.ptr(),
            d_ln_final_gamma.ptr(),
            d_ln_final_beta.ptr(),
            n_tokens as i32, d as i32,
        );
    }

    // ── Per-block backward (reverse order) ─────────────────────────────
    let mut block_grads = Vec::with_capacity(n_blocks);
    // Pre-allocate in forward order, fill in reverse
    for _ in 0..n_blocks {
        block_grads.push(None);
    }

    for b in (0..n_blocks).rev() {
        let block = &params.blocks[b];
        let bc = &cache.block_caches[b];

        // Initialize per-block gradients
        let mut d_w_q = GpuBuf::zeros(d * d);
        let mut d_w_k = GpuBuf::zeros(d * d);
        let mut d_w_v = GpuBuf::zeros(d * d);
        let mut d_w_o = GpuBuf::zeros(d * d);
        let d_ln_attn_gamma = GpuBuf::zeros(d);
        let d_ln_attn_beta = GpuBuf::zeros(d);
        let d_ln_mem_gamma = GpuBuf::zeros(d);
        let d_ln_mem_beta = GpuBuf::zeros(d);
        let mut level_grads: Vec<GpuLevelGrads> = (0..cfg.k)
            .map(|_| GpuLevelGrads::zeros_mlp(d, inter))
            .collect();

        // ── MAG gating backward ──────────────────────────────────────
        // Forward: residual_out = block_input + gated_out
        //          gated_out = attn_proj * gate
        //          gate = sigmoid(y_combined)
        // Spec: specs/infrastructure/20_stacked_mag_sigmoid_gating.md

        // d_gated_out = d_residual_stream (from output residual skip)
        // Gating backward: gated_out = attn_proj * gate
        let d_attn_proj = GpuBuf::zeros(bsd);
        let d_gate = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::gating_backward_cuda(
                d_residual_stream.as_ptr(), bc.attn_proj.as_ptr(), bc.gate.as_ptr(),
                d_attn_proj.ptr(), d_gate.ptr(), bsd_i32,
            );
        }

        // Sigmoid backward: gate = sigmoid(y_combined)
        let d_y_combined = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::sigmoid_backward_cuda(
                d_gate.as_ptr(), bc.gate.as_ptr(), d_y_combined.ptr(), bsd_i32,
            );
        }

        let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

        // ── Per-level output gradient norms (tape diagnostics) ─────
        let mut block_level_gnorms = vec![0.0f32; cfg.k];
        // Shared scratch buffer for grad_norm_sq_cuda calls
        let max_norm_blocks = (bsd + 255) / 256;
        let gnorm_scratch = GpuBuf::zeros(max_norm_blocks);

        // Helper: compute L2 norm of a GPU buffer using grad_norm_sq_cuda
        let compute_gnorm = |buf: &GpuBuf<f32>, scratch: &GpuBuf<f32>| -> f32 {
            let mut num_blocks_out: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::grad_norm_sq_cuda(
                    buf.as_ptr(), scratch.ptr(), bsd as i32, &mut num_blocks_out,
                )
            };
            assert_eq!(err, 0, "grad_norm_sq_cuda failed");
            crate::dispatch::cuda_sync();
            let nb = num_blocks_out as usize;
            let mut host = vec![0.0f32; nb];
            scratch.slice(0, nb).copy_to_host(&mut host);
            let sq_sum: f64 = host.iter().map(|x| *x as f64).sum();
            sq_sum.sqrt() as f32
        };

        let d_alpha_mem: Vec<f32>;
        let d_mem_input;

        if is_chained {
            // ── Chain CMS backward (HOPE Eq 70/97) ──────────────────
            // Spec: specs/infrastructure/35_chain_cms_gpu.md
            // No alpha_mem in chain mode — gradient flows serially through levels.
            d_alpha_mem = vec![0.0f32; cfg.k];

            // d_y_combined IS d_y for level k-1 (no weighted sum)
            let mut d_upstream = d_y_combined.clone_buf();

            // Reverse chain: k-1 → 0
            for level in (0..cfg.k).rev() {
                // In chain mode, each level's input is the previous level's output
                // Level 0's input is ln_mem_out
                let level_input = if level == 0 {
                    &bc.ln_mem_out
                } else {
                    &bc.y_per_level[level - 1]
                };

                // Record per-level gnorm of the actual gradient entering this level
                if bc.memory_caches[level].is_some() {
                    block_level_gnorms[level] = compute_gnorm(&d_upstream, &gnorm_scratch);
                }

                // Dispatch based on forward's cached mode (memory_caches[level].is_some()),
                // not pulse.active_levels, to match SwiGluMlp promotion logic.
                if let Some(ref mem_cache) = bc.memory_caches[level] {
                    let d_emb_level = gpu_memory_backward(
                        &block.levels[level], cfg, mem_cache,
                        &d_upstream, level_input,
                        &mut level_grads[level],
                        s, d, level, bs,
                    );
                    d_upstream = d_emb_level;
                } else {
                    // Frozen level: read-only backward, gradient flows through
                    let d_emb_level = gpu_memory_read_only_backward(
                        &block.levels[level], &bc.y_per_level[level],
                        &d_upstream, level_input,
                        &mut level_grads[level],
                        s, d, bs,
                    );
                    d_upstream = d_emb_level;
                }
            }

            // d_upstream is now d_ln_mem_out
            d_mem_input = d_upstream;
        } else {
            // ── Independent/FreqGated aggregation backward ──────────
            // Forward: y_combined = Σ_l w[l] * y_level[l], w = softmax(alpha_mem)
            // Spec: specs/infrastructure/21_stacked_alpha_aggregation.md

            // Compute d_y_combined norm once (same gradient for all levels in this mode)
            let d_y_norm = compute_gnorm(&d_y_combined, &gnorm_scratch);
            for level in 0..cfg.k {
                if bc.memory_caches[level].is_some() {
                    block_level_gnorms[level] = d_y_norm;
                }
            }

            let w = &bc.alpha_weights;
            let mut d_y_host = vec![0.0f32; bsd];
            crate::dispatch::cuda_sync();
            d_y_combined.slice(0, bsd).copy_to_host(&mut d_y_host);
            let mut dots = vec![0.0f64; cfg.k];
            for l in 0..cfg.k {
                let mut y_l_host = vec![0.0f32; bsd];
                bc.y_per_level[l].slice(0, bsd).copy_to_host(&mut y_l_host);
                dots[l] = d_y_host.iter().zip(y_l_host.iter())
                    .map(|(&dy, &y)| dy as f64 * y as f64).sum();
            }
            let weighted_dot_sum: f64 = (0..cfg.k).map(|j| w[j] as f64 * dots[j]).sum();
            d_alpha_mem = (0..cfg.k)
                .map(|l| (w[l] as f64 * (dots[l] - weighted_dot_sum)) as f32)
                .collect();

            // Each level receives d_y_level[l] = w[l] * d_y_combined
            d_mem_input = GpuBuf::zeros(bsd);
            for level in 0..cfg.k {
                let d_y_level = GpuBuf::zeros(bsd);
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(w[level], d_y_combined.as_ptr(), d_y_level.ptr(), bsd_i32);
                }

                // Dispatch based on forward's cached mode (memory_caches.is_some())
                if let Some(ref mem_cache) = bc.memory_caches[level] {
                    let d_emb_level = gpu_memory_backward(
                        &block.levels[level], cfg, mem_cache,
                        &d_y_level, &bc.ln_mem_out,
                        &mut level_grads[level],
                        s, d, level, bs,
                    );
                    unsafe {
                        crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd_i32);
                    }
                } else {
                    let d_emb_level = gpu_memory_read_only_backward(
                        &block.levels[level], &bc.y_per_level[level],
                        &d_y_level, &bc.ln_mem_out,
                        &mut level_grads[level],
                        s, d, bs,
                    );
                    unsafe {
                        crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd_i32);
                    }
                }
            }
        }

        // ── LN_mem backward ────────────────────────────────────────
        let d_residual_after_attn = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::layer_norm_backward_cuda(
                d_mem_input.as_ptr(),
                bc.residual_after_attn.as_ptr(),
                block.ln_mem_gamma.as_ptr(),
                bc.ln_mem_mean.as_ptr(),
                bc.ln_mem_rstd.as_ptr(),
                d_residual_after_attn.ptr(),
                d_ln_mem_gamma.ptr(),
                d_ln_mem_beta.ptr(),
                n_tokens as i32, d as i32,
            );
        }

        // ── Residual skip 1 backward ───────────────────────────────
        // residual_after_attn = block_input + attn_proj
        // d_attn_proj accumulates from two paths:
        //   1. d_attn_proj (from gating backward above)
        //   2. d_residual_after_attn (from skip 1: block_input + attn_proj)
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_residual_after_attn.as_ptr(), d_attn_proj.ptr(), bsd_i32);
        }

        // W_O backward: attn_proj = attn_out @ W_O^T
        // d_attn_out = d_attn_proj @ W_O
        // d_w_o = d_attn_proj^T @ attn_out
        // Spec: specs/infrastructure/18_stacked_w_o_output_projection.md
        let mut d_attn_out = GpuBuf::zeros(bsd);
        crate::dispatch::cublas_matmul_dd(
            &d_attn_proj, &block.w_o, &mut d_attn_out,
            n_tokens, d, d, 0.0,
        );
        gpu_matmul_transa_dd(
            &d_attn_proj, &bc.attn_out, &mut d_w_o,
            d, n_tokens, d,
        );

        // ── SWA backward ───────────────────────────────────────────
        let mut d_q = GpuBuf::zeros(bsd);
        let mut d_k = GpuBuf::zeros(bsd);
        let mut d_v = GpuBuf::zeros(bsd);

        crate::dispatch::swa_backward_dd(
            &bc.q_bf16, &bc.k_bf16, &bc.v_bf16,
            &bc.attn_weights_bf16, &d_attn_out,
            &mut d_q, &mut d_k, &mut d_v,
            s, nh, hd, ws, bs,
        );

        // ── QKV projection backward ───────────────────────────────
        let mut d_qkv_source = GpuBuf::zeros(bsd);
        crate::dispatch::cublas_matmul_acc_dd(&d_q, &block.w_q, &mut d_qkv_source, n_tokens, d, d);
        crate::dispatch::cublas_matmul_acc_dd(&d_k, &block.w_k, &mut d_qkv_source, n_tokens, d, d);
        crate::dispatch::cublas_matmul_acc_dd(&d_v, &block.w_v, &mut d_qkv_source, n_tokens, d, d);

        gpu_matmul_transa_dd(&d_q, &bc.ln_attn_out, &mut d_w_q, d, n_tokens, d);
        gpu_matmul_transa_dd(&d_k, &bc.ln_attn_out, &mut d_w_k, d, n_tokens, d);
        gpu_matmul_transa_dd(&d_v, &bc.ln_attn_out, &mut d_w_v, d, n_tokens, d);

        // ── LN_attn backward ──────────────────────────────────────
        let d_block_input = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::layer_norm_backward_cuda(
                d_qkv_source.as_ptr(),
                bc.block_input.as_ptr(),
                block.ln_attn_gamma.as_ptr(),
                bc.ln_attn_mean.as_ptr(),
                bc.ln_attn_rstd.as_ptr(),
                d_block_input.ptr(),
                d_ln_attn_gamma.ptr(),
                d_ln_attn_beta.ptr(),
                n_tokens as i32, d as i32,
            );
        }
        // block_input is used in two places:
        //   1. residual_out = block_input + gated_out  → d_block_input += d_residual_stream
        //   2. residual_after_attn = block_input + attn_proj → d_block_input += d_residual_after_attn
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_residual_stream.as_ptr(), d_block_input.ptr(), bsd_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, d_residual_after_attn.as_ptr(), d_block_input.ptr(), bsd_i32);
        }

        // d_block_input becomes d_residual_stream for the previous block
        d_residual_stream = d_block_input;

        block_grads[b] = Some(GpuStackedBlockGrads {
            d_w_q, d_w_k, d_w_v, d_w_o,
            d_ln_attn_gamma, d_ln_attn_beta,
            d_ln_mem_gamma, d_ln_mem_beta,
            levels: level_grads,
            d_alpha_mem,
            level_output_gnorms: block_level_gnorms,
        });
    }

    // ── Embedding scatter-add ──────────────────────────────────────────
    // d_residual_stream is now d_embedded (gradient w.r.t. embedding output)
    unsafe {
        crate::cuda_ffi::embedding_scatter_add_cuda(
            d_residual_stream.as_ptr(),
            cache.input_ids_gpu.ptr() as *const i32,
            d_w_embed.ptr(),
            n_tokens as i32, d as i32,
        );
    }

    crate::dispatch::cuda_sync();

    GpuStackedGrads {
        d_w_embed,
        d_w_unembed,
        d_ln_final_gamma,
        d_ln_final_beta,
        blocks: block_grads.into_iter().map(|bg| bg.unwrap()).collect(),
    }
}
