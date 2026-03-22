/// GPU-resident stacked multi-block forward pass.
///
/// N blocks of [SWA + CMS(k levels)] connected via residual stream.
/// Shared embedding/unembedding + final LayerNorm across all blocks.
///
/// Reuses existing CUDA kernels from gpu_forward.rs — no new CUDA code.
///
/// Spec: specs/infrastructure/14_multi_block_stacking.md
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind, HopeVariant};
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;
#[cfg(feature = "cuda")]
use crate::parallel::ParallelStrategy;
#[cfg(feature = "cuda")]
use crate::gpu_forward::{GpuMemoryCache, gpu_memory_forward, gpu_memory_read_only, gpu_tnt_forward};
#[cfg(feature = "cuda")]
use crate::gpu_profiler::GpuProfiler;
#[cfg(feature = "cuda")]
use crate::{prof_start, prof_stop};

// ══════════════════════════════════════════════════════════════════════
// GpuStackedBlockCache — per-block forward activations
// ══════════════════════════════════════════════════════════════════════

/// Forward activations for one block. Consumed by stacked backward.
#[cfg(feature = "cuda")]
pub struct GpuStackedBlockCache {
    // Block input (residual stream at this block's entry)
    pub block_input: GpuBuf<f32>,     // [bs*s, d]
    // Attention branch
    pub q_f32: GpuBuf<f32>,           // [bs*s, d]
    pub k_f32: GpuBuf<f32>,           // [bs*s, d]
    pub v_f32: GpuBuf<f32>,           // [bs*s, d]
    pub q_bf16: GpuBuf<u16>,          // [bs*s, d]
    pub k_bf16: GpuBuf<u16>,          // [bs*s, d]
    pub v_bf16: GpuBuf<u16>,          // [bs*s, d]
    pub attn_out_bf16: GpuBuf<u16>,   // [bs*s, d]
    pub attn_weights_bf16: GpuBuf<u16>, // [bs*nh, s, ws]
    pub attn_out: GpuBuf<f32>,        // [bs*s, d]
    // LayerNorm caches
    pub ln_attn_out: GpuBuf<f32>,     // [bs*s, d]
    pub ln_attn_mean: GpuBuf<f32>,    // [bs*s]
    pub ln_attn_rstd: GpuBuf<f32>,    // [bs*s]
    pub ln_mem_out: GpuBuf<f32>,      // [bs*s, d]
    pub ln_mem_mean: GpuBuf<f32>,     // [bs*s]
    pub ln_mem_rstd: GpuBuf<f32>,     // [bs*s]
    // Memory branch
    pub memory_caches: Vec<Option<GpuMemoryCache>>,
    pub y_per_level: Vec<GpuBuf<f32>>,
    pub y_combined: GpuBuf<f32>,      // [bs*s, d]
    // MAG gating
    pub attn_proj: GpuBuf<f32>,       // [bs*s, d] — attn_out @ W_O^T
    pub gate: GpuBuf<f32>,            // [bs*s, d] — sigmoid(y_combined)
    // Learnable level aggregation
    pub alpha_weights: Vec<f32>,      // [k] — softmax(alpha_mem), for backward
    // Residual connections
    pub residual_after_attn: GpuBuf<f32>, // [bs*s, d] = block_input + attn_proj
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedCache — full stacked forward cache
// ══════════════════════════════════════════════════════════════════════

/// Full stacked forward cache. Contains per-block caches + shared activations.
#[cfg(feature = "cuda")]
pub struct GpuStackedCache {
    pub block_caches: Vec<GpuStackedBlockCache>,
    // Shared activations
    pub embedded: GpuBuf<f32>,            // [bs*s, d]
    pub input_ids_i32: Vec<i32>,
    pub target_ids_i32: Vec<i32>,
    pub input_ids_gpu: GpuBuf<f32>,
    pub target_ids_gpu: GpuBuf<f32>,
    // Final LN + output
    pub ln_final_out: GpuBuf<f32>,        // [bs*s, d]
    pub ln_final_mean: GpuBuf<f32>,       // [bs*s]
    pub ln_final_rstd: GpuBuf<f32>,       // [bs*s]
    pub logits: GpuBuf<f32>,              // [bs*s, v]
    // Pulse snapshot
    pub pulse: Pulse,
    // Dimensions
    pub s: usize,
    pub d: usize,
    pub v: usize,
    pub nh: usize,
    pub hd: usize,
    pub ws: usize,
    pub batch_size: usize,
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_forward — main entry point
// ══════════════════════════════════════════════════════════════════════

/// Stacked multi-block forward pass on GPU.
///
/// Flow:
///   1. Embed tokens (shared w_embed)
///   2. For each block b in 0..n_blocks:
///      a. LN_attn → QKV → SWA → residual skip 1
///      b. LN_mem → per-level memory → combine → residual skip 2
///   3. Final LN (shared ln_final)
///   4. Unembed (shared w_unembed) → cross-entropy loss
///
/// Returns (loss, cache) where cache is consumed by gpu_stacked_backward.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_forward(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    profiler: &mut Option<GpuProfiler>,
) -> (f32, GpuStackedCache) {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;
    let n_blocks = params.n_blocks();

    assert!(s > 0, "seq_len must be > 0");
    assert!(input_ids.len() >= s, "input_ids too short");
    let batch_size = input_ids.len() / s;
    assert!(batch_size >= 1);
    assert_eq!(input_ids.len(), batch_size * s);
    assert_eq!(target_ids.len(), batch_size * s);
    assert_eq!(d, nh * hd);
    assert_eq!(context.n_blocks, n_blocks, "context n_blocks mismatch");

    let bs = batch_size;
    let n_tokens = bs * s;
    let total = n_tokens * d;
    let tokens_i32 = i32::try_from(n_tokens).expect("n_tokens exceeds i32::MAX");
    let d_i32 = i32::try_from(d).expect("d_model exceeds i32::MAX");
    let v_i32 = i32::try_from(v).expect("vocab_size exceeds i32::MAX");
    let total_i32 = i32::try_from(total).expect("total exceeds i32::MAX");

    // Convert IDs to i32 for CUDA kernels
    let input_ids_i32: Vec<i32> = input_ids.iter()
        .map(|&x| i32::try_from(x).expect("input token id overflows i32"))
        .collect();
    let target_ids_i32: Vec<i32> = target_ids.iter()
        .map(|&x| i32::try_from(x).expect("target token id overflows i32"))
        .collect();

    // Upload IDs to GPU
    let d_input_ids = GpuBuf::<f32>::new(n_tokens);
    let d_target_ids = GpuBuf::<f32>::new(n_tokens);
    unsafe {
        let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
            d_input_ids.ptr() as *mut std::ffi::c_void,
            input_ids_i32.as_ptr() as *const std::ffi::c_void,
            n_tokens * 4,
        );
        assert_eq!(rc, 0);
        let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
            d_target_ids.ptr() as *mut std::ffi::c_void,
            target_ids_i32.as_ptr() as *const std::ffi::c_void,
            n_tokens * 4,
        );
        assert_eq!(rc, 0);
    }

    // ── Stage 1: Embedding gather (shared) ─────────────────────────────
    prof_start!(profiler, "embed_gather", Embedding, None, None);
    let embedded = GpuBuf::<f32>::zeros(total);
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.w_embed.as_ptr(),
            d_input_ids.ptr() as *const i32,
            embedded.ptr(),
            tokens_i32, d_i32,
        );
    }
    prof_stop!(profiler);

    // ── Stage 2: Per-block forward ─────────────────────────────────────
    // The residual stream starts as the embedding output.
    // Each block reads it, processes SWA + CMS, and writes back.
    let mut residual_stream = embedded.clone_buf();
    let mut block_caches = Vec::with_capacity(n_blocks);

    let is_tnt = cfg.parallel.as_ref()
        .map(|p| p.strategy == ParallelStrategy::TNTHierarchical)
        .unwrap_or(false);

    for b in 0..n_blocks {
        let block = &params.blocks[b];
        let block_ctx = &mut context.blocks[b];

        // Save block input for backward
        let block_input = residual_stream.clone_buf();

        // ── LN_attn on residual stream ─────────────────────────────
        prof_start!(profiler, "ln_attn_fwd", LayerNorm, Some(b), None);
        let ln_attn_out = GpuBuf::<f32>::zeros(total);
        let ln_attn_mean = GpuBuf::<f32>::zeros(n_tokens);
        let ln_attn_rstd = GpuBuf::<f32>::zeros(n_tokens);
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                residual_stream.as_ptr(),
                block.ln_attn_gamma.as_ptr(),
                block.ln_attn_beta.as_ptr(),
                ln_attn_out.ptr(), ln_attn_mean.ptr(), ln_attn_rstd.ptr(),
                n_tokens as i32, d_i32, 1e-5,
            );
        }
        prof_stop!(profiler);

        // ── QKV projections ────────────────────────────────────────
        prof_start!(profiler, "qkv_proj_fwd", Projection, Some(b), None);
        let mut q_f32 = GpuBuf::zeros(total);
        let mut k_f32 = GpuBuf::zeros(total);
        let mut v_f32 = GpuBuf::zeros(total);
        crate::dispatch::cublas_matmul_transb_dd(&ln_attn_out, &block.w_q, &mut q_f32, n_tokens, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&ln_attn_out, &block.w_k, &mut k_f32, n_tokens, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&ln_attn_out, &block.w_v, &mut v_f32, n_tokens, d, d, 0.0);
        prof_stop!(profiler);

        // ── SWA attention (bf16) ───────────────────────────────────
        prof_start!(profiler, "f32_to_bf16", Precision, Some(b), None);
        let aw_total = bs * nh * s * ws;
        let q_bf16 = GpuBuf::<u16>::zeros(total);
        let k_bf16 = GpuBuf::<u16>::zeros(total);
        let v_bf16 = GpuBuf::<u16>::zeros(total);
        let mut attn_out_bf16 = GpuBuf::<u16>::zeros(total);
        let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);

        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(q_f32.as_ptr(), q_bf16.ptr(), total_i32);
            crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), k_bf16.ptr(), total_i32);
            crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), v_bf16.ptr(), total_i32);
        }
        prof_stop!(profiler);

        prof_start!(profiler, "swa_fwd", Attention, Some(b), None);
        crate::dispatch::swa_forward_dd(
            &q_bf16, &k_bf16, &v_bf16,
            &mut attn_out_bf16, &mut attn_weights_bf16,
            s, nh, hd, ws, bs,
        );
        prof_stop!(profiler);

        prof_start!(profiler, "bf16_to_f32", Precision, Some(b), None);
        let attn_out = GpuBuf::<f32>::zeros(total);
        unsafe {
            crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), total_i32);
        }
        prof_stop!(profiler);

        // ── Output projection: attn_proj = attn_out @ W_O^T ─────────
        // Spec: specs/infrastructure/18_stacked_w_o_output_projection.md
        prof_start!(profiler, "out_proj_fwd", Projection, Some(b), None);
        let mut attn_proj = GpuBuf::<f32>::zeros(total);
        crate::dispatch::cublas_matmul_transb_dd(&attn_out, &block.w_o, &mut attn_proj, n_tokens, d, d, 0.0);
        prof_stop!(profiler);

        // ── Residual skip 1: residual_after_attn = block_input + attn_proj ──
        prof_start!(profiler, "residual_1", Residual, Some(b), None);
        let residual_after_attn = GpuBuf::<f32>::zeros(total);
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input.as_ptr(), residual_after_attn.ptr(), total_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, attn_proj.as_ptr(), residual_after_attn.ptr(), total_i32);
        }
        prof_stop!(profiler);

        // ── LN_mem on residual_after_attn ──────────────────────────
        prof_start!(profiler, "ln_mem_fwd", LayerNorm, Some(b), None);
        let ln_mem_out = GpuBuf::<f32>::zeros(total);
        let ln_mem_mean = GpuBuf::<f32>::zeros(n_tokens);
        let ln_mem_rstd = GpuBuf::<f32>::zeros(n_tokens);
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                residual_after_attn.as_ptr(),
                block.ln_mem_gamma.as_ptr(),
                block.ln_mem_beta.as_ptr(),
                ln_mem_out.ptr(), ln_mem_mean.ptr(), ln_mem_rstd.ptr(),
                n_tokens as i32, d_i32, 1e-5,
            );
        }
        prof_stop!(profiler);

        // ── Memory branch per level ────────────────────────────────
        let mut memory_caches = Vec::with_capacity(cfg.k);
        let mut y_per_level = Vec::with_capacity(cfg.k);

        let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

        if is_chained {
            // HOPE Eq 70 / Eq 97: Chain CMS — each level processes previous level's output.
            // Spec: specs/infrastructure/35_chain_cms_gpu.md
            let mut h = ln_mem_out.clone_buf();
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                prof_start!(profiler, "memory_fwd", MemoryForward, Some(b), Some(level));
                if effective_active {
                    if is_tnt
                        && bs == 1
                        && matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule)
                    {
                        let parallel_cfg = cfg.parallel.as_ref().unwrap();
                        let (y_level, mem_cache) = gpu_tnt_forward(
                            &block.levels[level], cfg, &h,
                            &mut block_ctx.memory[level],
                            s, d, level, bs, parallel_cfg,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    } else {
                        let (y_level, mem_cache) = gpu_memory_forward(
                            &block.levels[level], cfg, &h,
                            &mut block_ctx.memory[level],
                            s, d, level, bs,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    }
                } else {
                    // Frozen: identity pass-through, read-only M@q
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &h,
                        &block_ctx.memory[level],
                        n_tokens, d,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                prof_stop!(profiler);
                // Chain: next level's input is this level's output
                if level < cfg.k - 1 {
                    h = y_per_level[level].clone_buf();
                }
            }
        } else {
            // FreqGated/Independent: all levels process same ln_mem_out
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                prof_start!(profiler, "memory_fwd", MemoryForward, Some(b), Some(level));
                if effective_active {
                    if is_tnt
                        && bs == 1
                        && matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule)
                    {
                        let parallel_cfg = cfg.parallel.as_ref().unwrap();
                        let (y_level, mem_cache) = gpu_tnt_forward(
                            &block.levels[level], cfg, &ln_mem_out,
                            &mut block_ctx.memory[level],
                            s, d, level, bs, parallel_cfg,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    } else {
                        let (y_level, mem_cache) = gpu_memory_forward(
                            &block.levels[level], cfg, &ln_mem_out,
                            &mut block_ctx.memory[level],
                            s, d, level, bs,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    }
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &ln_mem_out,
                        &block_ctx.memory[level],
                        n_tokens, d,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                prof_stop!(profiler);
            }
        }

        // ── Level aggregation ────────────────────────────────────────
        prof_start!(profiler, "level_agg", Composition, Some(b), None);
        let alpha_weights;
        let y_combined;

        if is_chained {
            // Chain CMS: y_combined = y_per_level[k-1] directly (no weighted sum)
            // Spec 35: alpha_mem aggregation weights are NOT used in chain mode
            alpha_weights = vec![0.0f32; cfg.k];
            y_combined = y_per_level.last().unwrap().clone_buf();
        } else {
            // Learnable level aggregation: weights = softmax(alpha_mem)
            // Spec: specs/infrastructure/21_stacked_alpha_aggregation.md
            // HOPE eq-074: y_t = Agg(...), "learnable weighted sum"
            let mut alpha_host = vec![0.0f32; cfg.k];
            block.alpha_mem.slice(0, cfg.k).copy_to_host(&mut alpha_host);
            alpha_weights = crate::stacked_model::host_softmax(&alpha_host);
            y_combined = GpuBuf::<f32>::zeros(total);
            for (l, y_level) in y_per_level.iter().enumerate() {
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(alpha_weights[l], y_level.as_ptr(), y_combined.ptr(), total_i32);
                }
            }
        }

        prof_stop!(profiler);

        // ── MAG sigmoid gating: gate = σ(y_combined), gated_out = attn_proj * gate ──
        // Spec: specs/infrastructure/20_stacked_mag_sigmoid_gating.md
        // Titans eq-028: o = y ⊙ σ(M(x̃))
        prof_start!(profiler, "mag_gate", Composition, Some(b), None);
        let gate = GpuBuf::<f32>::zeros(total);
        let gated_out = GpuBuf::<f32>::zeros(total);
        unsafe {
            crate::cuda_ffi::sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), total_i32);
            crate::cuda_ffi::elemwise_mul_cuda(attn_proj.as_ptr(), gate.as_ptr(), gated_out.ptr(), total_i32);
        }
        prof_stop!(profiler);

        // ── Residual: residual_out = block_input + gated_out ──
        prof_start!(profiler, "residual_2", Residual, Some(b), None);
        let new_residual = GpuBuf::<f32>::zeros(total);
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input.as_ptr(), new_residual.ptr(), total_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, gated_out.as_ptr(), new_residual.ptr(), total_i32);
        }
        residual_stream = new_residual;
        prof_stop!(profiler);

        block_caches.push(GpuStackedBlockCache {
            block_input,
            q_f32, k_f32, v_f32,
            q_bf16, k_bf16, v_bf16,
            attn_out_bf16, attn_weights_bf16,
            attn_out,
            ln_attn_out, ln_attn_mean, ln_attn_rstd,
            ln_mem_out, ln_mem_mean, ln_mem_rstd,
            memory_caches,
            y_per_level,
            y_combined,
            attn_proj,
            gate,
            alpha_weights,
            residual_after_attn,
        });
    }

    // ── Stage 3: Final LayerNorm (shared) ──────────────────────────────
    prof_start!(profiler, "ln_final_fwd", LayerNorm, None, None);
    let ln_final_out = GpuBuf::<f32>::zeros(total);
    let ln_final_mean = GpuBuf::<f32>::zeros(n_tokens);
    let ln_final_rstd = GpuBuf::<f32>::zeros(n_tokens);
    unsafe {
        crate::cuda_ffi::layer_norm_forward_cuda(
            residual_stream.as_ptr(),
            params.ln_final_gamma.as_ptr(),
            params.ln_final_beta.as_ptr(),
            ln_final_out.ptr(), ln_final_mean.ptr(), ln_final_rstd.ptr(),
            n_tokens as i32, d_i32, 1e-5,
        );
    }
    prof_stop!(profiler);

    // ── Stage 4: Unembed (shared w_unembed) ───────────────────────────
    prof_start!(profiler, "unembed_fwd", Projection, None, None);
    let mut logits = GpuBuf::<f32>::zeros(n_tokens * v);
    crate::dispatch::cublas_matmul_dd(&ln_final_out, &params.w_unembed, &mut logits, n_tokens, d, v, 0.0);
    prof_stop!(profiler);

    // ── Stage 5: Cross-entropy loss ───────────────────────────────────
    prof_start!(profiler, "cross_entropy_fwd", Loss, None, None);
    let loss_gpu = GpuBuf::<f32>::zeros(1);
    unsafe {
        crate::cuda_ffi::cross_entropy_forward_cuda(
            logits.as_ptr(),
            d_target_ids.ptr() as *const i32,
            loss_gpu.ptr(),
            tokens_i32, v_i32,
        );
    }
    prof_stop!(profiler);
    crate::dispatch::cuda_sync();

    let mut loss_host = [0.0f32; 1];
    loss_gpu.copy_to_host(&mut loss_host);
    let valid_count = target_ids_i32.iter()
        .filter(|&&t| t >= 0 && (t as usize) < v)
        .count() as f32;
    let loss = if valid_count > 0.0 { loss_host[0] / valid_count } else { 0.0 };

    let cache = GpuStackedCache {
        block_caches,
        embedded,
        input_ids_i32,
        target_ids_i32,
        input_ids_gpu: d_input_ids,
        target_ids_gpu: d_target_ids,
        ln_final_out,
        ln_final_mean,
        ln_final_rstd,
        logits,
        pulse: pulse.clone(),
        s, d, v, nh, hd, ws,
        batch_size: bs,
    };

    (loss, cache)
}

// ══════════════════════════════════════════════════════════════════════
// GpuBuf clone helper
// ══════════════════════════════════════════════════════════════════════

/// Clone a GpuBuf by allocating a new buffer and doing D2D copy.
#[cfg(feature = "cuda")]
pub(crate) trait GpuBufClone {
    fn clone_buf(&self) -> Self;
}

#[cfg(feature = "cuda")]
impl GpuBufClone for GpuBuf<f32> {
    fn clone_buf(&self) -> Self {
        let new = GpuBuf::<f32>::zeros(self.len());
        let bytes = self.len() * std::mem::size_of::<f32>();
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                new.ptr() as *mut std::ffi::c_void,
                self.as_ptr() as *const std::ffi::c_void,
                bytes,
            );
            assert_eq!(rc, 0, "GpuBuf clone D2D copy failed");
        }
        new
    }
}
