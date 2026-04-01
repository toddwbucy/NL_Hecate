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
#[cfg(feature = "cuda")]
use crate::gpu_profiler::GpuProfiler;
#[cfg(feature = "cuda")]
use crate::{prof_start, prof_stop};

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
    profiler: &mut Option<GpuProfiler>,
    need_gnorms: bool,
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
    prof_start!(profiler, "cross_entropy_bwd", Loss, None, None);
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
    prof_stop!(profiler);

    // ── Unembed backward ───────────────────────────────────────────────
    prof_start!(profiler, "unembed_bwd", Projection, None, None);
    let mut d_ln_final_out = GpuBuf::zeros(bsd);
    crate::dispatch::cublas_matmul_transb_dd(
        &d_logits, &params.w_unembed, &mut d_ln_final_out, n_tokens, v, d, 0.0,
    );
    gpu_matmul_transa_dd(
        &cache.ln_final_out, &d_logits, &mut d_w_unembed,
        d, n_tokens, v,
    );
    prof_stop!(profiler);

    // ── Final LN backward ──────────────────────────────────────────────
    // The last block's output (residual_stream) went through ln_final.
    // Reconstruct: residual_out = block_input + gated_out
    //   where gated_out = attn_proj * gate (MAG sigmoid gating)
    prof_start!(profiler, "ln_final_bwd", LayerNorm, None, None);
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
    prof_stop!(profiler);

    // ── Per-block backward (reverse order) ─────────────────────────────
    let mut block_grads = Vec::with_capacity(n_blocks);
    // Pre-allocate in forward order, fill in reverse
    for _ in 0..n_blocks {
        block_grads.push(None);
    }

    // Spec 54/63: scratch buffer for batched gnorm + dot product launches.
    // Spec 63: expanded to n_blocks regions so MAG blocks can write at non-overlapping
    // offsets, enabling ONE sync after the entire block loop instead of per-block.
    let max_norm_blocks = (bsd + 255) / 256;
    let per_block_slots = max_norm_blocks * (1 + cfg.k);
    let gnorm_scratch: GpuBuf<f32> = GpuBuf::zeros(n_blocks * per_block_slots);
    let mut gnorm_host = vec![0.0f32; n_blocks * per_block_slots];

    // Spec 63: deferred MAG readback metadata — (block_idx, gnorm_nb, dot_segs, alpha_weights)
    struct DeferredMagBlock {
        block_idx: usize,
        base_offset: usize,
        gnorm_nb: usize,
        dot_segs: Vec<(usize, usize)>, // (start, end) relative to base_offset
        alpha_weights: Vec<f32>,
    }
    let mut deferred_mag_blocks: Vec<DeferredMagBlock> = Vec::new();

    // Spec 63: deferred chain gnorm metadata — (block_idx, level, seg_start, seg_end)
    // Accumulated across ALL blocks so we can do ONE sync + ONE D2H after the loop.
    let mut deferred_chain_gnorms: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut chain_gnorm_max_offset: usize = 0;

    // Spec 63: hoist keep-alive buffers outside the block loop so their cudaFree
    // calls don't create per-block sync points. cudaFree is synchronous (waits for
    // all pending GPU work), so dropping per-block was causing pipeline stalls.
    let mut all_keep_alive: Vec<GpuBuf<f32>> = Vec::new();

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
        prof_start!(profiler, "mag_gate_bwd", Composition, Some(b), None);
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
        prof_stop!(profiler);

        let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

        // ── Per-level output gradient norms (tape diagnostics) ─────
        let mut block_level_gnorms = vec![0.0f32; cfg.k];

        let d_alpha_mem: Vec<f32>;
        let d_mem_input;

        if is_chained {
            // ── Chain CMS backward (HOPE Eq 70/97) ──────────────────
            // Spec: specs/infrastructure/35_chain_cms_gpu.md
            // Spec 46: per-level token reduction — backward through pool/upsample.
            // No alpha_mem in chain mode — gradient flows serially through levels.
            d_alpha_mem = vec![0.0f32; cfg.k];

            // d_y_combined is at full resolution [bs*s, d].
            // Spec 46: backward of the final upsample (chain output → full res)
            let last_level = cfg.k - 1;
            let last_c = cfg.chunk_sizes.get(last_level).copied().unwrap_or(1);
            let last_s_f = bc.level_seq_lens[last_level];
            let mut d_upstream = if last_c > 1 {
                // Backward of repeat_upsample: sum groups of C
                let mut d_reduced = GpuBuf::zeros(bs * last_s_f * d);
                unsafe {
                    crate::cuda_ffi::repeat_upsample_1d_backward_f32_cuda(
                        d_y_combined.as_ptr(), d_reduced.ptr(),
                        bs as i32, s as i32, d as i32, last_c as i32,
                    );
                }
                d_reduced
            } else {
                d_y_combined.clone_buf()
            };

            // Spec 54/63: batched gnorm launches for chain mode.
            // Spec 63: gnorms are purely diagnostic in chain mode — skip kernel
            // launch entirely on non-log steps. On log steps, each block writes
            // to its own region of the expanded scratch buffer (base = b * per_block_slots)
            // so ONE sync after the block loop replaces per-block syncs.
            let block_gnorm_base = b * per_block_slots;
            let mut gnorm_offset = block_gnorm_base;

            // Reverse chain: k-1 → 0
            for level in (0..cfg.k).rev() {
                let s_f = bc.level_seq_lens[level];

                // In chain mode, each level's input is the previous level's output,
                // pooled down to this level's resolution. Level 0's input is ln_mem_out.
                // Spec 46: reconstruct the pooled input for projection gradient computation.
                let level_input_buf;
                let level_input = if level == 0 {
                    // L0 input was ln_mem_out [bs*s, d] (no pooling for L0 with chunk_size=1)
                    &bc.ln_mem_out
                } else {
                    let prev_s_f = bc.level_seq_lens[level - 1];
                    if prev_s_f > s_f {
                        // Reconstruct pooled input: pool y_per_level[level-1] to s_f
                        let pool_factor = prev_s_f / s_f;
                        level_input_buf = GpuBuf::zeros(bs * s_f * d);
                        unsafe {
                            crate::cuda_ffi::mean_pool_1d_f32_cuda(
                                bc.y_per_level[level - 1].as_ptr(), level_input_buf.ptr(),
                                bs as i32, prev_s_f as i32, d as i32, pool_factor as i32,
                            );
                        }
                        &level_input_buf
                    } else {
                        &bc.y_per_level[level - 1]
                    }
                };

                // Spec 54/63: launch gnorm kernel at block-specific offset (no sync yet).
                // Spec 63: skip entirely on non-log steps — gnorms are diagnostic only
                // in chain mode. On log steps, results accumulate across blocks for
                // a single post-loop sync.
                if need_gnorms && bc.memory_caches[level].is_some() {
                    let buf_len = d_upstream.len() as i32;
                    let mut nb: i32 = 0;
                    let err = unsafe {
                        crate::cuda_ffi::grad_norm_sq_cuda(
                            d_upstream.as_ptr(),
                            gnorm_scratch.ptr().add(gnorm_offset),
                            buf_len, &mut nb,
                        )
                    };
                    assert_eq!(err, 0, "grad_norm_sq_cuda failed");
                    let seg_end = gnorm_offset + nb as usize;
                    deferred_chain_gnorms.push((b, level, gnorm_offset, seg_end));
                    gnorm_offset = seg_end;
                    if seg_end > chain_gnorm_max_offset {
                        chain_gnorm_max_offset = seg_end;
                    }
                }

                // Dispatch based on forward's cached mode (memory_caches[level].is_some()),
                // not pulse.active_levels, to match SwiGluMlp promotion logic.
                prof_start!(profiler, "memory_bwd", MemoryBackward, Some(b), Some(level));
                if let Some(ref mem_cache) = bc.memory_caches[level] {
                    let d_emb_level = gpu_memory_backward(
                        &block.levels[level], cfg, mem_cache,
                        &d_upstream, level_input,
                        &mut level_grads[level],
                        s_f, d, level, bs,
                    );
                    // Keep old d_upstream alive so gnorm kernel can read it
                    let old = std::mem::replace(&mut d_upstream, d_emb_level);
                    all_keep_alive.push(old);
                } else {
                    // Frozen level: read-only backward, gradient flows through
                    let d_emb_level = gpu_memory_read_only_backward(
                        &block.levels[level], &bc.y_per_level[level],
                        &d_upstream, level_input,
                        &mut level_grads[level],
                        s_f, d, bs,
                    );
                    let old = std::mem::replace(&mut d_upstream, d_emb_level);
                    all_keep_alive.push(old);
                }
                prof_stop!(profiler);

                // Spec 46: backward through pool between this level and previous.
                // In chain mode, forward pooled the prev level's output to get this level's input.
                if level > 0 {
                    let prev_s_f = bc.level_seq_lens[level - 1];
                    if prev_s_f > s_f {
                        // Backward of mean_pool: broadcast gradient / pool_factor
                        let pool_factor = prev_s_f / s_f;
                        let mut d_expanded = GpuBuf::zeros(bs * prev_s_f * d);
                        unsafe {
                            crate::cuda_ffi::mean_pool_1d_backward_f32_cuda(
                                d_upstream.as_ptr(), d_expanded.ptr(),
                                bs as i32, prev_s_f as i32, d as i32, pool_factor as i32,
                            );
                        }
                        // Keep old buffer alive — cudaFree is synchronous and would
                        // fence the batched gnorm kernels still in the stream queue.
                        let old = std::mem::replace(&mut d_upstream, d_expanded);
                        all_keep_alive.push(old);
                    }
                }
            }

            // Spec 63: chain gnorm sync deferred to post-loop (deferred_chain_gnorms).
            // keep-alive buffers live in all_keep_alive — freed after block loop.

            // Backward through level-0 pool: forward pooled ln_mem_out [bs*s, d] → [bs*s_f, d].
            // The inter-level pool backward (level > 0 guard above) doesn't handle this.
            let c0 = cfg.chunk_sizes.get(0).copied().unwrap_or(1);
            if c0 > 1 {
                let mut d_full = GpuBuf::zeros(bsd);
                unsafe {
                    crate::cuda_ffi::mean_pool_1d_backward_f32_cuda(
                        d_upstream.as_ptr(), d_full.ptr(),
                        bs as i32, s as i32, d as i32, c0 as i32,
                    );
                }
                let old = std::mem::replace(&mut d_upstream, d_full);
                all_keep_alive.push(old);
            }

            // d_upstream is now at full resolution [bs*s, d]
            d_mem_input = d_upstream;
        } else {
            // ── Independent/FreqGated aggregation backward ──────────
            // Forward: y_combined = Σ_l w[l] * y_level[l], w = softmax(alpha_mem)
            // Spec: specs/infrastructure/21_stacked_alpha_aggregation.md

            // Spec 54/63: batch gnorm + k dot products into per-block scratch region.
            // Spec 63: kernels launch at block-specific offsets so ALL blocks' partials
            // can be read in ONE sync after the loop (instead of per-block syncs).
            let w = &bc.alpha_weights;
            let block_base = b * per_block_slots;
            let mut offset = block_base;

            // Slot 0: gnorm of d_y_combined (same gradient for all levels)
            let mut gnorm_nb: i32 = 0;
            {
                let err = unsafe {
                    crate::cuda_ffi::grad_norm_sq_cuda(
                        d_y_combined.as_ptr(),
                        gnorm_scratch.ptr().add(offset),
                        bsd as i32, &mut gnorm_nb,
                    )
                };
                assert_eq!(err, 0, "grad_norm_sq_cuda failed");
                offset += gnorm_nb as usize;
            }

            // Slots 1..k: dot products for softmax Jacobian (spec 53)
            let mut dot_segs: Vec<(usize, usize)> = Vec::new();
            for l in 0..cfg.k {
                let seg_start = offset;
                let mut nb: i32 = 0;
                let err = unsafe {
                    crate::cuda_ffi::dot_product_partial_f32_cuda(
                        d_y_combined.as_ptr(),
                        bc.y_per_level[l].as_ptr(),
                        gnorm_scratch.ptr().add(offset),
                        bsd as i32,
                        &mut nb,
                    )
                };
                assert_eq!(err, 0, "dot_product_partial_f32_cuda failed");
                offset += nb as usize;
                dot_segs.push((seg_start, offset));
            }

            // Spec 63: defer readback — record metadata for post-loop processing.
            // d_alpha_mem placeholder will be filled after the block loop.
            deferred_mag_blocks.push(DeferredMagBlock {
                block_idx: b,
                base_offset: block_base,
                gnorm_nb: gnorm_nb as usize,
                dot_segs,
                alpha_weights: w.clone(),
            });
            d_alpha_mem = vec![0.0f32; cfg.k]; // placeholder, patched post-loop

            // Each level receives d_y_level[l] = w[l] * d_y_combined
            // Spec 46: d_y_combined is at full resolution [bs*s, d].
            // For levels with chunk_size > 1, backward goes through upsample then pool.
            d_mem_input = GpuBuf::zeros(bsd);
            for level in 0..cfg.k {
                let c = cfg.chunk_sizes.get(level).copied().unwrap_or(1);
                let s_f = bc.level_seq_lens[level];

                let d_y_level_full = GpuBuf::zeros(bsd);
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(w[level], d_y_combined.as_ptr(), d_y_level_full.ptr(), bsd_i32);
                }

                // Spec 46: backward through upsample (sum groups of C)
                let d_y_level = if c > 1 {
                    let mut d_reduced = GpuBuf::zeros(bs * s_f * d);
                    unsafe {
                        crate::cuda_ffi::repeat_upsample_1d_backward_f32_cuda(
                            d_y_level_full.as_ptr(), d_reduced.ptr(),
                            bs as i32, s as i32, d as i32, c as i32,
                        );
                    }
                    d_reduced
                } else {
                    d_y_level_full
                };

                // Spec 46: the level_input for backward was the pooled ln_mem_out.
                // Forward pooled ln_mem_out from [bs*s, d] to [bs*s_f, d] for this level.
                // We reconstruct the pooled input here for the backward projection gradients.
                let level_input = if c > 1 {
                    let mut pooled = GpuBuf::zeros(bs * s_f * d);
                    unsafe {
                        crate::cuda_ffi::mean_pool_1d_f32_cuda(
                            bc.ln_mem_out.as_ptr(), pooled.ptr(),
                            bs as i32, s as i32, d as i32, c as i32,
                        );
                    }
                    pooled
                } else {
                    bc.ln_mem_out.clone_buf()
                };

                // Dispatch based on forward's cached mode (memory_caches.is_some())
                prof_start!(profiler, "memory_bwd", MemoryBackward, Some(b), Some(level));
                if let Some(ref mem_cache) = bc.memory_caches[level] {
                    let d_emb_level = gpu_memory_backward(
                        &block.levels[level], cfg, mem_cache,
                        &d_y_level, &level_input,
                        &mut level_grads[level],
                        s_f, d, level, bs,
                    );
                    // Spec 46: backward through mean_pool — broadcast gradient / C
                    if c > 1 {
                        unsafe {
                            crate::cuda_ffi::mean_pool_1d_backward_f32_cuda(
                                d_emb_level.as_ptr(), d_mem_input.ptr(),
                                bs as i32, s as i32, d as i32, c as i32,
                            );
                        }
                    } else {
                        unsafe {
                            crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd_i32);
                        }
                    }
                } else {
                    let d_emb_level = gpu_memory_read_only_backward(
                        &block.levels[level], &bc.y_per_level[level],
                        &d_y_level, &level_input,
                        &mut level_grads[level],
                        s_f, d, bs,
                    );
                    if c > 1 {
                        unsafe {
                            crate::cuda_ffi::mean_pool_1d_backward_f32_cuda(
                                d_emb_level.as_ptr(), d_mem_input.ptr(),
                                bs as i32, s as i32, d as i32, c as i32,
                            );
                        }
                    } else {
                        unsafe {
                            crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd_i32);
                        }
                    }
                }
                prof_stop!(profiler);
            }
        }

        // ── LN_mem backward ────────────────────────────────────────
        prof_start!(profiler, "ln_mem_bwd", LayerNorm, Some(b), None);
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
        prof_stop!(profiler);

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
        prof_start!(profiler, "out_proj_bwd", Projection, Some(b), None);
        let mut d_attn_out = GpuBuf::zeros(bsd);
        crate::dispatch::cublas_matmul_dd(
            &d_attn_proj, &block.w_o, &mut d_attn_out,
            n_tokens, d, d, 0.0,
        );
        gpu_matmul_transa_dd(
            &d_attn_proj, &bc.attn_out, &mut d_w_o,
            d, n_tokens, d,
        );
        prof_stop!(profiler);

        // ── SWA backward ───────────────────────────────────────────
        prof_start!(profiler, "swa_bwd", Attention, Some(b), None);
        let mut d_q = GpuBuf::zeros(bsd);
        let mut d_k = GpuBuf::zeros(bsd);
        let mut d_v = GpuBuf::zeros(bsd);

        crate::dispatch::swa_backward_dd(
            &bc.q_bf16, &bc.k_bf16, &bc.v_bf16,
            &bc.attn_weights_bf16, &d_attn_out,
            &mut d_q, &mut d_k, &mut d_v,
            s, nh, hd, ws, bs,
        );
        prof_stop!(profiler);

        // ── QKV projection backward ───────────────────────────────
        prof_start!(profiler, "qkv_proj_bwd", Projection, Some(b), None);
        let mut d_qkv_source = GpuBuf::zeros(bsd);
        crate::dispatch::cublas_matmul_acc_dd(&d_q, &block.w_q, &mut d_qkv_source, n_tokens, d, d);
        crate::dispatch::cublas_matmul_acc_dd(&d_k, &block.w_k, &mut d_qkv_source, n_tokens, d, d);
        crate::dispatch::cublas_matmul_acc_dd(&d_v, &block.w_v, &mut d_qkv_source, n_tokens, d, d);

        gpu_matmul_transa_dd(&d_q, &bc.ln_attn_out, &mut d_w_q, d, n_tokens, d);
        gpu_matmul_transa_dd(&d_k, &bc.ln_attn_out, &mut d_w_k, d, n_tokens, d);
        gpu_matmul_transa_dd(&d_v, &bc.ln_attn_out, &mut d_w_v, d, n_tokens, d);
        prof_stop!(profiler);

        // ── LN_attn backward ──────────────────────────────────────
        prof_start!(profiler, "ln_attn_bwd", LayerNorm, Some(b), None);
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
        prof_stop!(profiler);
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

    // ── Spec 63: deferred readback — ONE sync for all blocks ───────────
    // Both chain gnorms and MAG gnorm+dots write to non-overlapping regions
    // of gnorm_scratch. ONE sync + ONE D2H covers all blocks.
    let mag_max_offset = deferred_mag_blocks.iter()
        .flat_map(|dm| dm.dot_segs.last().map(|&(_, end)| end))
        .max()
        .unwrap_or(0);
    let total_max_offset = chain_gnorm_max_offset.max(mag_max_offset);

    if total_max_offset > 0 {
        crate::dispatch::cuda_sync();
        gnorm_scratch.slice(0, total_max_offset).copy_to_host(&mut gnorm_host[..total_max_offset]);
    }

    // Process deferred chain gnorms → block_level_gnorms (diagnostic only)
    for &(block_idx, level, start, end) in &deferred_chain_gnorms {
        let bg = block_grads[block_idx].as_mut().unwrap();
        let sq_sum: f64 = (start..end).map(|i| gnorm_host[i] as f64).sum();
        bg.level_output_gnorms[level] = sq_sum.sqrt() as f32;
    }

    // Process deferred MAG blocks → d_alpha_mem + block_level_gnorms
    for dm in &deferred_mag_blocks {
        let bg = block_grads[dm.block_idx].as_mut().unwrap();

        // Reduce gnorm partials → block_level_gnorms
        if need_gnorms {
            let gnorm_sq: f64 = (dm.base_offset..dm.base_offset + dm.gnorm_nb)
                .map(|i| gnorm_host[i] as f64).sum();
            let d_y_norm = gnorm_sq.sqrt() as f32;
            for level in 0..cfg.k {
                bg.level_output_gnorms[level] = d_y_norm;
            }
        }

        // Reduce dot partials → d_alpha_mem (needed every step for optimizer)
        let w = &dm.alpha_weights;
        let mut dots = vec![0.0f64; cfg.k];
        for (l, &(start, end)) in dm.dot_segs.iter().enumerate() {
            dots[l] = (start..end).map(|i| gnorm_host[i] as f64).sum();
        }
        let weighted_dot_sum: f64 = (0..cfg.k).map(|j| w[j] as f64 * dots[j]).sum();
        bg.d_alpha_mem = (0..cfg.k)
            .map(|l| (w[l] as f64 * (dots[l] - weighted_dot_sum)) as f32)
            .collect();
    }

    // Spec 63: drop all keep-alive buffers here (after block loop + deferred MAG).
    // cudaFree is synchronous but at this point all backward kernels are enqueued,
    // and the embedding scatter-add below doesn't read these buffers.
    drop(all_keep_alive);

    // ── Embedding scatter-add ──────────────────────────────────────────
    // d_residual_stream is now d_embedded (gradient w.r.t. embedding output)
    prof_start!(profiler, "embed_scatter", Embedding, None, None);
    unsafe {
        crate::cuda_ffi::embedding_scatter_add_cuda(
            d_residual_stream.as_ptr(),
            cache.input_ids_gpu.ptr() as *const i32,
            d_w_embed.ptr(),
            n_tokens as i32, d as i32,
        );
    }
    prof_stop!(profiler);
    // Spec 54: removed cuda_sync() — grads stay on GPU, consumed by optimizer
    // kernels on the same default stream (ordering guaranteed).

    GpuStackedGrads {
        d_w_embed,
        d_w_unembed,
        d_ln_final_gamma,
        d_ln_final_beta,
        blocks: block_grads.into_iter().map(|bg| bg.unwrap()).collect(),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Spec 53: Verify GPU partial-reduction dot product matches CPU reference.
    #[test]
    fn test_gpu_dot_product_matches_cpu() {
        let sizes = [128, 256, 1000, 4096, 12_582_912 / 64]; // last = scaled-down d=2048 case
        for n in sizes {
            // Generate deterministic test data
            let a_host: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.7).sin())).collect();
            let b_host: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.3).cos())).collect();

            // CPU reference (f64 accumulation)
            let cpu_dot: f64 = a_host.iter().zip(b_host.iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();

            // GPU path
            let a_gpu = GpuBuf::from_host(&a_host);
            let b_gpu = GpuBuf::from_host(&b_host);
            let max_blocks = (n + 255) / 256;
            let scratch = GpuBuf::<f32>::zeros(max_blocks);

            let mut num_blocks: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::dot_product_partial_f32_cuda(
                    a_gpu.as_ptr(), b_gpu.as_ptr(), scratch.ptr(),
                    n as i32, &mut num_blocks,
                )
            };
            assert_eq!(err, 0);
            crate::dispatch::cuda_sync();

            let nb = num_blocks as usize;
            let mut partials = vec![0.0f32; nb];
            scratch.slice(0, nb).copy_to_host(&mut partials);
            let gpu_dot: f64 = partials.iter().map(|x| *x as f64).sum();

            // Tolerance: f32 products accumulated in shared memory lose precision
            // relative to f64 CPU reference. Allow 1e-3 relative or 1e-6 absolute.
            let rel_err = if cpu_dot.abs() > 1e-10 {
                (gpu_dot - cpu_dot).abs() / cpu_dot.abs()
            } else {
                (gpu_dot - cpu_dot).abs()
            };
            assert!(
                rel_err < 1e-3,
                "n={n}: gpu={gpu_dot:.8}, cpu={cpu_dot:.8}, rel_err={rel_err:.2e}"
            );
        }
    }
}
