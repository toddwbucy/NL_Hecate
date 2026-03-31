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
// Debug: set HECATE_DEBUG_CMS=1 to log CMS dispatch on pattern changes
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
static DEBUG_CMS: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    std::env::var("HECATE_DEBUG_CMS").map(|v| v == "1").unwrap_or(false)
});

#[cfg(feature = "cuda")]
use std::sync::Mutex;
#[cfg(feature = "cuda")]
static LAST_CMS_PATTERN: std::sync::LazyLock<Mutex<(Vec<bool>, usize)>> =
    std::sync::LazyLock::new(|| Mutex::new((Vec::new(), 0)));

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
    /// Per-level seq_len after CMS token reduction (Spec 46).
    /// s_f[level] = s / chunk_sizes[level]. For chunk_sizes=[1,8,64,512]: [512,64,8,1].
    pub level_seq_lens: Vec<usize>,
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

        // Spec 46: per-level token reduction via CMS frequency hierarchy.
        // Level f processes s_f = s / chunk_sizes[f] tokens.
        let level_seq_lens: Vec<usize> = (0..cfg.k)
            .map(|level| {
                let c = cfg.chunk_sizes.get(level).copied().unwrap_or(1);
                debug_assert!(c >= 1 && s % c == 0,
                    "seq_len ({s}) must be divisible by chunk_sizes[{level}] ({c})");
                s / c
            })
            .collect();
        let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

        // Debug: log CMS dispatch only when the active-level pattern changes.
        // Suppress output when global_step regresses (e.g. coherence probe resets
        // the Conductor to step 0 — that's a separate inference context, not training).
        if *DEBUG_CMS && b == 0 {
            let current: Vec<bool> = (0..cfg.k).map(|l| {
                pulse.active_levels[l] || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp)
            }).collect();
            let step = pulse.global_step;
            let changed = {
                let guard = LAST_CMS_PATTERN.lock().unwrap();
                let (ref prev_pattern, prev_step) = *guard;
                // Only print if step advances AND pattern changed
                step >= prev_step && *prev_pattern != current
            };
            if changed {
                eprintln!("[CMS] step={} k={} level_seq_lens={:?} active={:?}",
                    step, cfg.k, level_seq_lens,
                    current.iter().enumerate()
                        .map(|(l, &a)| format!("L{}:{}", l, if a { "fwd" } else { "ro" }))
                        .collect::<Vec<_>>().join(" "));
                *LAST_CMS_PATTERN.lock().unwrap() = (current, step);
            }
        }

        if is_chained {
            // HOPE Eq 70 / Eq 97: Chain CMS — each level processes previous level's output.
            // Spec: specs/infrastructure/35_chain_cms_gpu.md
            // Spec 46: each level also reduces token count via mean pooling.
            let mut h = ln_mem_out.clone_buf();
            let mut h_seq_len = s;  // current resolution of h
            for level in 0..cfg.k {
                let s_f = level_seq_lens[level];
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                // Spec 46: pool h down to this level's resolution if needed.
                // In chain mode, pool factor is relative to current h resolution.
                let level_input = if h_seq_len > s_f {
                    let pool_factor = h_seq_len / s_f;
                    let mut pooled = GpuBuf::zeros(bs * s_f * d);
                    unsafe {
                        crate::cuda_ffi::mean_pool_1d_f32_cuda(
                            h.as_ptr(), pooled.ptr(),
                            bs as i32, h_seq_len as i32, d as i32, pool_factor as i32,
                        );
                    }
                    pooled
                } else {
                    h.clone_buf()
                };

                prof_start!(profiler, "memory_fwd", MemoryForward, Some(b), Some(level));
                if effective_active {
                    if is_tnt
                        && bs == 1
                        // Spec 51: TNT now supports per-head memory (num_heads > 1)
                        && matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule)
                    {
                        let parallel_cfg = cfg.parallel.as_ref().unwrap();
                        let (y_level, mem_cache) = gpu_tnt_forward(
                            &block.levels[level], cfg, &level_input,
                            &mut block_ctx.memory[level],
                            s_f, d, level, bs, parallel_cfg,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    } else {
                        let (y_level, mem_cache) = gpu_memory_forward(
                            &block.levels[level], cfg, &level_input,
                            &mut block_ctx.memory[level],
                            s_f, d, level, bs,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    }
                } else {
                    // Frozen: identity pass-through, read-only M@q
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &level_input,
                        &block_ctx.memory[level],
                        bs * s_f, d, cfg.swa.num_heads, cfg.swa.head_dim,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                prof_stop!(profiler);
                // Chain: next level's input is this level's output (at reduced resolution)
                if level < cfg.k - 1 {
                    h = y_per_level[level].clone_buf();
                    h_seq_len = s_f;
                }
            }
        } else {
            // FreqGated/Independent: all levels process same ln_mem_out
            // Spec 46: each level pools ln_mem_out to its own resolution
            for level in 0..cfg.k {
                let s_f = level_seq_lens[level];
                let c = cfg.chunk_sizes.get(level).copied().unwrap_or(1);
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                // Spec 46: pool input down to this level's resolution
                let level_input = if c > 1 {
                    let mut pooled = GpuBuf::zeros(bs * s_f * d);
                    unsafe {
                        crate::cuda_ffi::mean_pool_1d_f32_cuda(
                            ln_mem_out.as_ptr(), pooled.ptr(),
                            bs as i32, s as i32, d as i32, c as i32,
                        );
                    }
                    pooled
                } else {
                    ln_mem_out.clone_buf()
                };

                prof_start!(profiler, "memory_fwd", MemoryForward, Some(b), Some(level));
                if effective_active {
                    if is_tnt
                        && bs == 1
                        // Spec 51: TNT now supports per-head memory (num_heads > 1)
                        && matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule)
                    {
                        let parallel_cfg = cfg.parallel.as_ref().unwrap();
                        let (y_level, mem_cache) = gpu_tnt_forward(
                            &block.levels[level], cfg, &level_input,
                            &mut block_ctx.memory[level],
                            s_f, d, level, bs, parallel_cfg,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    } else {
                        let (y_level, mem_cache) = gpu_memory_forward(
                            &block.levels[level], cfg, &level_input,
                            &mut block_ctx.memory[level],
                            s_f, d, level, bs,
                        );
                        y_per_level.push(y_level);
                        memory_caches.push(Some(mem_cache));
                    }
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &level_input,
                        &block_ctx.memory[level],
                        bs * s_f, d, cfg.swa.num_heads, cfg.swa.head_dim,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                prof_stop!(profiler);
            }

            // Spec 46: upsample reduced-resolution outputs back to full seq_len
            // for aggregation. y_per_level[l] is [bs*s_f, d]; needs [bs*s, d].
            for level in 0..cfg.k {
                let c = cfg.chunk_sizes.get(level).copied().unwrap_or(1);
                if c > 1 {
                    let s_f = level_seq_lens[level];
                    let mut upsampled = GpuBuf::zeros(bs * s * d);
                    unsafe {
                        crate::cuda_ffi::repeat_upsample_1d_f32_cuda(
                            y_per_level[level].as_ptr(), upsampled.ptr(),
                            bs as i32, s as i32, d as i32, c as i32,
                        );
                    }
                    y_per_level[level] = upsampled;
                }
            }
        }

        // ── Level aggregation ────────────────────────────────────────
        prof_start!(profiler, "level_agg", Composition, Some(b), None);
        let alpha_weights;
        let y_combined;

        if is_chained {
            // Chain CMS: y_combined = y_per_level[k-1] directly (no weighted sum)
            // Spec 35: alpha_mem aggregation weights are NOT used in chain mode
            // Spec 46: last level output may be at reduced resolution — upsample to [bs*s, d]
            alpha_weights = vec![0.0f32; cfg.k];
            let last_level = cfg.k - 1;
            let last_c = cfg.chunk_sizes.get(last_level).copied().unwrap_or(1);
            let last_y = y_per_level.last().unwrap();
            if last_c > 1 {
                let mut upsampled = GpuBuf::zeros(bs * s * d);
                unsafe {
                    crate::cuda_ffi::repeat_upsample_1d_f32_cuda(
                        last_y.as_ptr(), upsampled.ptr(),
                        bs as i32, s as i32, d as i32, last_c as i32,
                    );
                }
                y_combined = upsampled;
            } else {
                y_combined = last_y.clone_buf();
            };
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
            level_seq_lens,
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
// Stacked decode workspace — pre-allocated GPU buffers for single-token decode
// ══════════════════════════════════════════════════════════════════════

/// Pre-allocated GPU workspace for `gpu_stacked_decode_token`.
/// Created once at prefill, reused for every decode call (zero cudaMalloc per token).
///
/// Spec: specs/infrastructure/47_single_token_decode.md
#[cfg(feature = "cuda")]
pub struct StackedDecodeWorkspace {
    /// Token ID upload buffer [1].
    pub d_input: GpuBuf<f32>,
    /// Per-block decode scratch (one per block). Reused each call.
    pub per_block: Vec<BlockDecodeBuffers>,
    /// Final LN output [1, d].
    pub ln_final_out: GpuBuf<f32>,
    pub ln_final_mean: GpuBuf<f32>,
    pub ln_final_rstd: GpuBuf<f32>,
    /// Logits [vocab_size] on GPU.
    pub logits_gpu: GpuBuf<f32>,
    /// Inter-block residual stream [d] — avoids per-token cudaMalloc.
    pub residual: GpuBuf<f32>,
}

/// Per-block buffers for single-token decode.
#[cfg(feature = "cuda")]
pub struct BlockDecodeBuffers {
    pub ln_attn_out: GpuBuf<f32>,     // [d]
    pub ln_attn_mean: GpuBuf<f32>,    // [1]
    pub ln_attn_rstd: GpuBuf<f32>,    // [1]
    pub q_f32: GpuBuf<f32>,           // [d]
    pub k_f32: GpuBuf<f32>,           // [d]
    pub v_f32: GpuBuf<f32>,           // [d]
    pub q_bf16: GpuBuf<u16>,          // [d]
    pub attn_out_bf16: GpuBuf<u16>,   // [d]
    pub attn_out_f32: GpuBuf<f32>,    // [d]
    pub attn_proj: GpuBuf<f32>,       // [d]
    pub residual_after_attn: GpuBuf<f32>, // [d]
    pub ln_mem_out: GpuBuf<f32>,      // [d]
    pub ln_mem_mean: GpuBuf<f32>,     // [1]
    pub ln_mem_rstd: GpuBuf<f32>,     // [1]
    pub y_combined: GpuBuf<f32>,      // [d]
    pub gate: GpuBuf<f32>,            // [d]
    pub gated_out: GpuBuf<f32>,       // [d]
    pub residual_out: GpuBuf<f32>,    // [d] — output residual (avoids per-token cudaMalloc)
    pub chain_h: GpuBuf<f32>,         // [d] — scratch for chained CMS level input
}

#[cfg(feature = "cuda")]
impl StackedDecodeWorkspace {
    /// Allocate all workspace buffers once. `n_blocks` determines per-block count.
    pub fn new(n_blocks: usize, d: usize, v: usize) -> Self {
        let per_block = (0..n_blocks).map(|_| BlockDecodeBuffers {
            ln_attn_out: GpuBuf::zeros(d),
            ln_attn_mean: GpuBuf::zeros(1),
            ln_attn_rstd: GpuBuf::zeros(1),
            q_f32: GpuBuf::zeros(d),
            k_f32: GpuBuf::zeros(d),
            v_f32: GpuBuf::zeros(d),
            q_bf16: GpuBuf::<u16>::zeros(d),
            attn_out_bf16: GpuBuf::<u16>::zeros(d),
            attn_out_f32: GpuBuf::zeros(d),
            attn_proj: GpuBuf::zeros(d),
            residual_after_attn: GpuBuf::zeros(d),
            ln_mem_out: GpuBuf::zeros(d),
            ln_mem_mean: GpuBuf::zeros(1),
            ln_mem_rstd: GpuBuf::zeros(1),
            y_combined: GpuBuf::zeros(d),
            gate: GpuBuf::zeros(d),
            gated_out: GpuBuf::zeros(d),
            residual_out: GpuBuf::zeros(d),
            chain_h: GpuBuf::zeros(d),
        }).collect();

        StackedDecodeWorkspace {
            d_input: GpuBuf::<f32>::new(1),
            per_block,
            ln_final_out: GpuBuf::zeros(d),
            ln_final_mean: GpuBuf::zeros(1),
            ln_final_rstd: GpuBuf::zeros(1),
            logits_gpu: GpuBuf::zeros(v),
            residual: GpuBuf::zeros(d),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_prefill — process full prompt, populate KV caches
// ══════════════════════════════════════════════════════════════════════

/// Prefill: run full stacked forward on prompt tokens, populate per-block KV caches,
/// return last-position logits [vocab_size].
///
/// This reuses `gpu_stacked_forward` for the math, then extracts K/V projections
/// from the cache to fill per-block KV caches for decode.
///
/// Spec: specs/infrastructure/47_single_token_decode.md
#[cfg(feature = "cuda")]
pub fn gpu_stacked_prefill(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    kv_caches: &mut Vec<crate::gpu_forward::GpuKVCache>,
    profiler: &mut Option<GpuProfiler>,
) -> Vec<f32> {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let n_blocks = params.n_blocks();

    // Use dummy targets (we don't need loss, just logits + KV caches)
    let dummy_targets: Vec<usize> = input_ids.iter().skip(1).chain(std::iter::once(&0)).cloned().collect();

    let (_loss, cache) = gpu_stacked_forward(
        params, cfg, input_ids, &dummy_targets, pulse, context, profiler,
    );

    // Extract K/V projections from each block's cache into KV caches.
    // block_caches[b].k_f32 is [seq_len, d] — exactly what the cache needs.
    kv_caches.clear();
    let window = cfg.swa.window_size;
    let kv_capacity = std::cmp::max(s + 1024, window + 1024);
    for bc in &cache.block_caches {
        let mut kv = crate::gpu_forward::GpuKVCache::new(
            kv_capacity,  // capacity: max(seq_len, window_size) + room for decode tokens
            d,
            s,  // scratch for prefill-sized append
        );
        kv.append_f32(&bc.k_f32, &bc.v_f32, s);
        kv_caches.push(kv);
    }

    // Extract last-position logits
    let last_offset = (s - 1) * v;
    let mut logits = vec![0.0f32; v];
    let mut all_logits = vec![0.0f32; s * v];
    cache.logits.copy_to_host(&mut all_logits);
    logits.copy_from_slice(&all_logits[last_offset..last_offset + v]);
    logits
}

// ══════════════════════════════════════════════════════════════════════
// gpu_stacked_decode_token — decode one token using KV caches + M state
// ══════════════════════════════════════════════════════════════════════

/// Decode one token through all stacked blocks. Returns logits [vocab_size].
///
/// Per-token path:
/// 1. Embed 1 token
/// 2. Per block: LN → QKV → append KV cache → SWA single-token → memory(s=1) → MAG gate → residual
/// 3. Final LN → unembed → logits
///
/// Spec: specs/infrastructure/47_single_token_decode.md
#[cfg(feature = "cuda")]
pub fn gpu_stacked_decode_token(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_id: usize,
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    kv_caches: &mut [crate::gpu_forward::GpuKVCache],
    ws: &mut StackedDecodeWorkspace,
) -> Vec<f32> {
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let window_size = cfg.swa.window_size;
    let n_blocks = params.n_blocks();
    let d_i32 = d as i32;

    assert!(token_id < v, "token_id {} >= vocab_size {}", token_id, v);
    assert_eq!(kv_caches.len(), n_blocks, "need one KV cache per block");

    // ── Embed 1 token ──────────────────────────────────────────────
    let input_i32 = [token_id as i32];
    unsafe {
        let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
            ws.d_input.ptr() as *mut std::ffi::c_void,
            input_i32.as_ptr() as *const std::ffi::c_void,
            4,
        );
        assert_eq!(rc, 0);
    }

    // Gather embedding for 1 token into workspace residual (no allocation)
    ws.residual.zero();
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.w_embed.as_ptr(),
            ws.d_input.ptr() as *const i32,
            ws.residual.ptr(),
            1, d_i32,
        );
    }

    let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

    // ── Per-block forward ──────────────────────────────────────────
    for b in 0..n_blocks {
        let block = &params.blocks[b];
        let block_ctx = &mut context.blocks[b];
        let kv = &mut kv_caches[b];
        let bws = &mut ws.per_block[b];

        // Save block input for residual
        let block_input_ptr = ws.residual.as_ptr();

        // ── LN_attn ────────────────────────────────────────────────
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                ws.residual.as_ptr(),
                block.ln_attn_gamma.as_ptr(),
                block.ln_attn_beta.as_ptr(),
                bws.ln_attn_out.ptr(), bws.ln_attn_mean.ptr(), bws.ln_attn_rstd.ptr(),
                1, d_i32, 1e-5,
            );
        }

        // ── QKV projections [1,d] @ [d,d]^T ───────────────────────
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_q, &mut bws.q_f32, 1, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_k, &mut bws.k_f32, 1, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_v, &mut bws.v_f32, 1, d, d, 0.0);

        // ── Append K,V to cache ────────────────────────────────────
        kv.append_f32(&bws.k_f32, &bws.v_f32, 1);

        // ── SWA single-token attention ─────────────────────────────
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(bws.q_f32.as_ptr(), bws.q_bf16.ptr(), d_i32);
        }
        crate::dispatch::swa_single_token_dd(
            &bws.q_bf16, &kv.k_cache_bf16, &kv.v_cache_bf16,
            &mut bws.attn_out_bf16,
            kv.len, nh, hd, window_size,
        );
        unsafe {
            crate::cuda_ffi::bf16_to_f32_cuda(bws.attn_out_bf16.as_ptr(), bws.attn_out_f32.ptr(), d_i32);
        }

        // ── Output projection: attn_proj = attn_out @ W_O^T ───────
        crate::dispatch::cublas_matmul_transb_dd(&bws.attn_out_f32, &block.w_o, &mut bws.attn_proj, 1, d, d, 0.0);

        // ── Residual skip 1 ────────────────────────────────────────
        bws.residual_after_attn.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input_ptr, bws.residual_after_attn.ptr(), d_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, bws.attn_proj.as_ptr(), bws.residual_after_attn.ptr(), d_i32);
        }

        // ── LN_mem ─────────────────────────────────────────────────
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                bws.residual_after_attn.as_ptr(),
                block.ln_mem_gamma.as_ptr(),
                block.ln_mem_beta.as_ptr(),
                bws.ln_mem_out.ptr(), bws.ln_mem_mean.ptr(), bws.ln_mem_rstd.ptr(),
                1, d_i32, 1e-5,
            );
        }

        // ── CMS memory per level (s=1, no pooling) ─────────────────
        // Note: at s=1, gpu_memory_forward and gpu_tnt_forward produce identical
        // results (TNT chunkwise with C=1 chunk of 1 token = standard memory update).
        // Using gpu_memory_forward unconditionally is CS-10 compliant.
        let mut y_per_level: Vec<GpuBuf<f32>> = Vec::with_capacity(cfg.k);

        if is_chained {
            // Chain mode: each level processes previous level's output.
            // Copy ln_mem_out into workspace chain_h to avoid clone allocation.
            bws.chain_h.zero();
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, bws.ln_mem_out.as_ptr(), bws.chain_h.ptr(), d_i32);
            }
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                if effective_active {
                    let (y_level, _mem_cache) = gpu_memory_forward(
                        &block.levels[level], cfg, &bws.chain_h,
                        &mut block_ctx.memory[level],
                        1, d, level, 1,
                    );
                    y_per_level.push(y_level);
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &bws.chain_h,
                        &block_ctx.memory[level],
                        1, d, nh, hd,
                    );
                    y_per_level.push(y_level);
                }
                // Chain: next level's input is this level's output
                if level < cfg.k - 1 {
                    bws.chain_h.zero();
                    unsafe {
                        crate::cuda_ffi::saxpy_cuda(1.0, y_per_level[level].as_ptr(), bws.chain_h.ptr(), d_i32);
                    }
                }
            }
        } else {
            // Independent mode: all levels process ln_mem_out
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                if effective_active {
                    let (y_level, _mem_cache) = gpu_memory_forward(
                        &block.levels[level], cfg, &bws.ln_mem_out,
                        &mut block_ctx.memory[level],
                        1, d, level, 1,
                    );
                    y_per_level.push(y_level);
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &bws.ln_mem_out,
                        &block_ctx.memory[level],
                        1, d, nh, hd,
                    );
                    y_per_level.push(y_level);
                }
            }
        }

        // ── Level aggregation ──────────────────────────────────────
        if is_chained {
            // Chain: use last level output directly
            bws.y_combined.zero();
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, y_per_level.last().unwrap().as_ptr(), bws.y_combined.ptr(), d_i32);
            }
        } else {
            // Learnable weighted sum
            let mut alpha_host = vec![0.0f32; cfg.k];
            block.alpha_mem.slice(0, cfg.k).copy_to_host(&mut alpha_host);
            let alpha_weights = crate::stacked_model::host_softmax(&alpha_host);
            bws.y_combined.zero();
            for (l, y_level) in y_per_level.iter().enumerate() {
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(alpha_weights[l], y_level.as_ptr(), bws.y_combined.ptr(), d_i32);
                }
            }
        }

        // ── MAG sigmoid gating ─────────────────────────────────────
        unsafe {
            crate::cuda_ffi::sigmoid_cuda(bws.y_combined.as_ptr(), bws.gate.ptr(), d_i32);
            crate::cuda_ffi::elemwise_mul_cuda(bws.attn_proj.as_ptr(), bws.gate.as_ptr(), bws.gated_out.ptr(), d_i32);
        }

        // ── Residual skip 2 (write into workspace buffer, then swap) ─
        bws.residual_out.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input_ptr, bws.residual_out.ptr(), d_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, bws.gated_out.as_ptr(), bws.residual_out.ptr(), d_i32);
        }
        // Copy block output into ws.residual for next block's input
        ws.residual.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, bws.residual_out.as_ptr(), ws.residual.ptr(), d_i32);
        }
    }

    // ── Final LN ───────────────────────────────────────────────────
    unsafe {
        crate::cuda_ffi::layer_norm_forward_cuda(
            ws.residual.as_ptr(),
            params.ln_final_gamma.as_ptr(),
            params.ln_final_beta.as_ptr(),
            ws.ln_final_out.ptr(), ws.ln_final_mean.ptr(), ws.ln_final_rstd.ptr(),
            1, d_i32, 1e-5,
        );
    }

    // ── Unembed ────────────────────────────────────────────────────
    crate::dispatch::cublas_matmul_dd(&ws.ln_final_out, &params.w_unembed, &mut ws.logits_gpu, 1, d, v, 0.0);
    crate::dispatch::cuda_sync();

    // Download logits
    let mut logits = vec![0.0f32; v];
    ws.logits_gpu.copy_to_host(&mut logits);
    logits
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

#[cfg(feature = "cuda")]
impl GpuBufClone for GpuBuf<u16> {
    fn clone_buf(&self) -> Self {
        let new = GpuBuf::<u16>::zeros(self.len());
        let bytes = self.len() * std::mem::size_of::<u16>();
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                new.ptr() as *mut std::ffi::c_void,
                self.as_ptr() as *const std::ffi::c_void,
                bytes,
            );
            assert_eq!(rc, 0, "GpuBuf<u16> clone D2D copy failed");
        }
        new
    }
}

// ══════════════════════════════════════════════════════════════════════
// Unified Forward Path — Spec 68, Phase B
// ══════════════════════════════════════════════════════════════════════
//
// Single composable code path for training AND generation.
// `forward_single_token` is the atomic operation.
// `ActivationWindow` stores per-token activations for backward.
// `gpu_stacked_forward_tokens` is the one entry point.

#[cfg(feature = "cuda")]
use std::collections::VecDeque;
#[cfg(feature = "cuda")]
use crate::conductor::Conductor;

// ── Per-token activation cache ──────────────────────────────────────

/// Activations for one token through one block. Saved for backward.
#[cfg(feature = "cuda")]
pub struct TokenBlockCache {
    pub block_input: GpuBuf<f32>,         // [d]
    pub q_f32: GpuBuf<f32>,              // [d]
    pub k_f32: GpuBuf<f32>,              // [d]
    pub v_f32: GpuBuf<f32>,              // [d]
    pub q_bf16: GpuBuf<u16>,             // [d]
    pub k_bf16: GpuBuf<u16>,             // [d]
    pub v_bf16: GpuBuf<u16>,             // [d]
    pub attn_out_bf16: GpuBuf<u16>,      // [d]
    pub attn_out: GpuBuf<f32>,           // [d]
    pub attn_proj: GpuBuf<f32>,          // [d]
    pub ln_attn_out: GpuBuf<f32>,        // [d]
    pub ln_attn_mean: GpuBuf<f32>,       // [1]
    pub ln_attn_rstd: GpuBuf<f32>,       // [1]
    pub ln_mem_out: GpuBuf<f32>,         // [d]
    pub ln_mem_mean: GpuBuf<f32>,        // [1]
    pub ln_mem_rstd: GpuBuf<f32>,        // [1]
    pub memory_caches: Vec<Option<GpuMemoryCache>>,
    pub y_per_level: Vec<GpuBuf<f32>>,   // [k] buffers, each [d] (s=1)
    pub level_seq_lens: Vec<usize>,       // [k] — always 1 for single-token
    pub y_combined: GpuBuf<f32>,         // [d]
    pub gate: GpuBuf<f32>,              // [d]
    pub alpha_weights: Vec<f32>,         // [k]
    pub residual_after_attn: GpuBuf<f32>, // [d]
}

/// Activations for one token through the entire model.
#[cfg(feature = "cuda")]
pub struct TokenActivationCache {
    pub block_caches: Vec<TokenBlockCache>,
    pub embedded: GpuBuf<f32>,           // [d]
    pub token_id: usize,
    pub token_id_i32: i32,
    pub ln_final_out: GpuBuf<f32>,       // [d]
    pub ln_final_mean: GpuBuf<f32>,      // [1]
    pub ln_final_rstd: GpuBuf<f32>,      // [1]
    pub logits: GpuBuf<f32>,             // [v]
    pub pulse: Pulse,
}

// ── Activation window (ring buffer) ─────────────────────────────────

/// Sliding window of per-token activation caches for backward.
/// Capacity = gradient_window_size (the old "seq_len").
///
/// Spec 68: "When a token falls off the window, nothing needs to happen.
/// The token's influence on M already occurred during forward.
/// Gradients can no longer flow through that token's operations."
#[cfg(feature = "cuda")]
pub struct ActivationWindow {
    pub entries: VecDeque<TokenActivationCache>,
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl ActivationWindow {
    pub fn new(capacity: usize) -> Self {
        ActivationWindow {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, cache: TokenActivationCache) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front(); // oldest falls off — already in M
        }
        self.entries.push_back(cache);
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn capacity(&self) -> usize { self.capacity }
    pub fn is_full(&self) -> bool { self.entries.len() == self.capacity }

    /// Last token's logits (for sampling during generation).
    pub fn last_logits(&self) -> Option<Vec<f32>> {
        self.entries.back().map(|entry| {
            let v = entry.logits.len();
            let mut logits = vec![0.0f32; v];
            entry.logits.copy_to_host(&mut logits);
            logits
        })
    }

    /// Assemble the window contents into a `GpuStackedCache` for backward.
    ///
    /// Concatenates per-token [d] buffers into contiguous [window_len, d] buffers.
    /// Recomputes batched SWA to get attention weights for backward.
    pub fn assemble_cache(
        &self,
        cfg: &MAGConfig,
        target_ids: &[usize],
    ) -> GpuStackedCache {
        let s = self.entries.len();
        assert!(s > 0, "cannot assemble empty window");
        let d = cfg.swa.d_model;
        let v = cfg.swa.vocab_size;
        let nh = cfg.swa.num_heads;
        let hd = cfg.swa.head_dim;
        let ws = cfg.swa.window_size;
        let n_blocks = self.entries[0].block_caches.len();
        let bs: usize = 1; // unified path is always batch_size=1

        // Use the pulse from the last token in the window
        let pulse = self.entries.back().unwrap().pulse.clone();

        // ── Shared activations ──────────────────────────────────────
        let embedded = concat_f32_bufs(self.entries.iter().map(|e| &e.embedded), d);

        // Token IDs
        let input_ids_i32: Vec<i32> = self.entries.iter().map(|e| e.token_id_i32).collect();
        let target_ids_i32: Vec<i32> = target_ids.iter()
            .map(|&x| i32::try_from(x).expect("target token id overflows i32"))
            .collect();
        assert_eq!(target_ids_i32.len(), s, "target_ids length must match window length");

        let input_ids_gpu = GpuBuf::<f32>::new(s);
        let target_ids_gpu = GpuBuf::<f32>::new(s);
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
                input_ids_gpu.ptr() as *mut std::ffi::c_void,
                input_ids_i32.as_ptr() as *const std::ffi::c_void,
                s * 4,
            );
            assert_eq!(rc, 0);
            let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
                target_ids_gpu.ptr() as *mut std::ffi::c_void,
                target_ids_i32.as_ptr() as *const std::ffi::c_void,
                s * 4,
            );
            assert_eq!(rc, 0);
        }

        // Final LN
        let ln_final_out = concat_f32_bufs(self.entries.iter().map(|e| &e.ln_final_out), d);
        let ln_final_mean = concat_f32_bufs(self.entries.iter().map(|e| &e.ln_final_mean), 1);
        let ln_final_rstd = concat_f32_bufs(self.entries.iter().map(|e| &e.ln_final_rstd), 1);

        // Logits
        let logits = concat_f32_bufs(self.entries.iter().map(|e| &e.logits), v);

        // ── Per-block caches ────────────────────────────────────────
        let mut block_caches = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let block_input = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].block_input), d);
            let q_f32 = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].q_f32), d);
            let k_f32 = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].k_f32), d);
            let v_f32 = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].v_f32), d);

            // bf16 QKV for SWA recomputation
            let q_bf16 = concat_u16_bufs(self.entries.iter().map(|e| &e.block_caches[b].q_bf16), d);
            let k_bf16 = concat_u16_bufs(self.entries.iter().map(|e| &e.block_caches[b].k_bf16), d);
            let v_bf16 = concat_u16_bufs(self.entries.iter().map(|e| &e.block_caches[b].v_bf16), d);

            // Recompute batched SWA to get attention weights for backward
            let aw_total = bs * nh * s * ws;
            let mut attn_out_bf16 = GpuBuf::<u16>::zeros(s * d);
            let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);
            crate::dispatch::swa_forward_dd(
                &q_bf16, &k_bf16, &v_bf16,
                &mut attn_out_bf16, &mut attn_weights_bf16,
                s, nh, hd, ws, bs,
            );

            // Convert recomputed attn_out to f32 (matches what single-token produced)
            let attn_out = GpuBuf::<f32>::zeros(s * d);
            unsafe {
                crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), (s * d) as i32);
            }

            let attn_proj = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].attn_proj), d);
            let ln_attn_out = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_attn_out), d);
            let ln_attn_mean = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_attn_mean), 1);
            let ln_attn_rstd = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_attn_rstd), 1);
            let ln_mem_out = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_mem_out), d);
            let ln_mem_mean = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_mem_mean), 1);
            let ln_mem_rstd = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].ln_mem_rstd), 1);
            let y_combined = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].y_combined), d);
            let gate = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].gate), d);
            let residual_after_attn = concat_f32_bufs(self.entries.iter().map(|e| &e.block_caches[b].residual_after_attn), d);

            // Memory caches: concatenate per-level across tokens
            let k_levels = self.entries[0].block_caches[b].memory_caches.len();
            let mut memory_caches: Vec<Option<GpuMemoryCache>> = Vec::with_capacity(k_levels);
            let mut y_per_level: Vec<GpuBuf<f32>> = Vec::with_capacity(k_levels);
            let mut level_seq_lens: Vec<usize> = Vec::with_capacity(k_levels);

            for level in 0..k_levels {
                // y_per_level: concat per-token [d] → [s, d]
                y_per_level.push(concat_f32_bufs(
                    self.entries.iter().map(|e| &e.block_caches[b].y_per_level[level]), d,
                ));
                level_seq_lens.push(s); // all tokens, no CMS pooling at s=1

                // Memory caches: concatenate the per-token memory caches
                let first = &self.entries[0].block_caches[b].memory_caches[level];
                if first.is_some() {
                    memory_caches.push(Some(concat_memory_caches(
                        self.entries.iter().map(|e| e.block_caches[b].memory_caches[level].as_ref().unwrap()),
                        d, s, nh, hd, cfg,
                    )));
                } else {
                    memory_caches.push(None);
                }
            }

            // Alpha weights: use the most recent token's alpha (params may have been
            // updated between tokens during within-gen learning).
            let alpha_weights = self.entries.back().unwrap().block_caches[b].alpha_weights.clone();

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
                level_seq_lens,
                y_combined,
                gate, attn_proj,
                alpha_weights,
                residual_after_attn,
            });
        }

        GpuStackedCache {
            block_caches,
            embedded,
            input_ids_i32,
            target_ids_i32,
            input_ids_gpu,
            target_ids_gpu,
            ln_final_out,
            ln_final_mean,
            ln_final_rstd,
            logits,
            pulse,
            s,
            d,
            v,
            nh,
            hd,
            ws,
            batch_size: bs,
        }
    }
}

// ── Buffer concatenation helpers ────────────────────────────────────

/// Concatenate per-token f32 GPU buffers [elem_size] into one contiguous [n * elem_size] buffer.
#[cfg(feature = "cuda")]
fn concat_f32_bufs<'a>(bufs: impl Iterator<Item = &'a GpuBuf<f32>>, elem_size: usize) -> GpuBuf<f32> {
    let bufs: Vec<&GpuBuf<f32>> = bufs.collect();
    let n = bufs.len();
    let total = n * elem_size;
    let out = GpuBuf::<f32>::zeros(total);
    for (i, buf) in bufs.iter().enumerate() {
        assert_eq!(buf.len(), elem_size, "concat_f32_bufs: element {i} has len {} != {elem_size}", buf.len());
        let offset_bytes = i * elem_size * 4;
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                (out.ptr() as *mut u8).add(offset_bytes) as *mut std::ffi::c_void,
                buf.as_ptr() as *const std::ffi::c_void,
                elem_size * 4,
            );
            assert_eq!(rc, 0, "concat_f32_bufs D2D copy failed");
        }
    }
    out
}

/// Concatenate per-token u16 GPU buffers [elem_size] into one contiguous [n * elem_size] buffer.
#[cfg(feature = "cuda")]
fn concat_u16_bufs<'a>(bufs: impl Iterator<Item = &'a GpuBuf<u16>>, elem_size: usize) -> GpuBuf<u16> {
    let bufs: Vec<&GpuBuf<u16>> = bufs.collect();
    let n = bufs.len();
    let total = n * elem_size;
    let out = GpuBuf::<u16>::zeros(total);
    for (i, buf) in bufs.iter().enumerate() {
        assert_eq!(buf.len(), elem_size, "concat_u16_bufs: element {i} has len {} != {elem_size}", buf.len());
        let offset_bytes = i * elem_size * 2;
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                (out.ptr() as *mut u8).add(offset_bytes) as *mut std::ffi::c_void,
                buf.as_ptr() as *const std::ffi::c_void,
                elem_size * 2,
            );
            assert_eq!(rc, 0, "concat_u16_bufs D2D copy failed");
        }
    }
    out
}

/// Concatenate per-token GpuMemoryCache entries into one cache with s=window_len.
#[cfg(feature = "cuda")]
fn concat_memory_caches<'a>(
    caches: impl Iterator<Item = &'a GpuMemoryCache>,
    d: usize,
    s: usize,
    nh: usize,
    hd: usize,
    _cfg: &MAGConfig,
) -> GpuMemoryCache {
    let caches: Vec<&GpuMemoryCache> = caches.collect();
    assert_eq!(caches.len(), s);

    // All caches must be the same variant
    match &caches[0] {
        GpuMemoryCache::Delta { proxy, .. } => {
            let proxy = *proxy;
            let bs_mem = nh;  // batch_size=1, fold heads into batch
            GpuMemoryCache::Delta {
                k_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { k_mem, .. } => k_mem, _ => unreachable!() }), bs_mem * hd),
                v_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { v_mem, .. } => v_mem, _ => unreachable!() }), bs_mem * hd),
                q_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { q_mem, .. } => q_mem, _ => unreachable!() }), bs_mem * hd),
                alpha: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { alpha, .. } => alpha, _ => unreachable!() }), bs_mem),
                theta: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { theta, .. } => theta, _ => unreachable!() }), bs_mem),
                k_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { k_norms, .. } => k_norms, _ => unreachable!() }), bs_mem),
                q_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Delta { q_norms, .. } => q_norms, _ => unreachable!() }), bs_mem),
                m_states: assemble_m_states(&caches, d, s, nh, hd, proxy, false),
                proxy,
            }
        }
        GpuMemoryCache::Titans { proxy, .. } => {
            let proxy = *proxy;
            let bs_mem = nh;
            GpuMemoryCache::Titans {
                k_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { k_mem, .. } => k_mem, _ => unreachable!() }), bs_mem * hd),
                v_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { v_mem, .. } => v_mem, _ => unreachable!() }), bs_mem * hd),
                q_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { q_mem, .. } => q_mem, _ => unreachable!() }), bs_mem * hd),
                alpha: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { alpha, .. } => alpha, _ => unreachable!() }), bs_mem),
                theta: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { theta, .. } => theta, _ => unreachable!() }), bs_mem),
                eta: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { eta, .. } => eta, _ => unreachable!() }), bs_mem),
                k_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { k_norms, .. } => k_norms, _ => unreachable!() }), bs_mem),
                q_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Titans { q_norms, .. } => q_norms, _ => unreachable!() }), bs_mem),
                m_states: assemble_m_states(&caches, d, s, nh, hd, proxy, false),
                s_states: assemble_m_states(&caches, d, s, nh, hd, proxy, true),
                proxy,
            }
        }
        GpuMemoryCache::Hebbian { .. } => {
            let bs_mem = nh;
            GpuMemoryCache::Hebbian {
                k_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { k_mem, .. } => k_mem, _ => unreachable!() }), bs_mem * hd),
                v_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { v_mem, .. } => v_mem, _ => unreachable!() }), bs_mem * hd),
                q_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { q_mem, .. } => q_mem, _ => unreachable!() }), bs_mem * hd),
                alpha: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { alpha, .. } => alpha, _ => unreachable!() }), bs_mem),
                k_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { k_norms, .. } => k_norms, _ => unreachable!() }), bs_mem),
                q_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::Hebbian { q_norms, .. } => q_norms, _ => unreachable!() }), bs_mem),
                m_states: assemble_m_states(&caches, d, s, nh, hd, false, false),
            }
        }
        GpuMemoryCache::DGD { .. } => {
            let bs_mem = nh;
            GpuMemoryCache::DGD {
                k_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { k_mem, .. } => k_mem, _ => unreachable!() }), bs_mem * hd),
                v_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { v_mem, .. } => v_mem, _ => unreachable!() }), bs_mem * hd),
                q_mem: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { q_mem, .. } => q_mem, _ => unreachable!() }), bs_mem * hd),
                alpha: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { alpha, .. } => alpha, _ => unreachable!() }), bs_mem),
                theta: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { theta, .. } => theta, _ => unreachable!() }), bs_mem),
                k_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { k_norms, .. } => k_norms, _ => unreachable!() }), bs_mem),
                q_norms: concat_f32_bufs(caches.iter().map(|c| match c { GpuMemoryCache::DGD { q_norms, .. } => q_norms, _ => unreachable!() }), bs_mem),
                m_states: assemble_m_states(&caches, d, s, nh, hd, false, false),
            }
        }
        _ => panic!(
            "concat_memory_caches: unsupported memory cache variant — \
             unified path currently supports Delta, Titans, Hebbian, DGD. \
             SwiGlu/Mlp/Ckpt/Chunkwise variants require separate assembly logic.",
        ),
    }
}

/// Assemble m_states (or s_states) from per-token caches into the format backward expects.
///
/// When proxy=true: backward expects [(s+1) * bs_mem * dd] with M snapshots at each timestep.
/// When proxy=false: backward expects [bs_mem * dd] — just the final M (already in context).
///
/// For the unified path at s=1 per token: each token's memory cache has either
/// [2 * bs_mem * dd] (proxy, s=1: M_0 and M_1) or [bs_mem * dd] (non-proxy).
/// We concatenate them into the expected layout.
#[cfg(feature = "cuda")]
fn assemble_m_states(
    caches: &[&GpuMemoryCache],
    _d: usize,
    s: usize,
    nh: usize,
    hd: usize,
    proxy: bool,
    is_s_states: bool,
) -> GpuBuf<f32> {
    let dd = hd * hd;
    let bs_mem = nh;
    let chunk = bs_mem * dd;

    if !proxy {
        // Non-proxy: backward only needs final M. Take from last token.
        let last = caches.last().unwrap();
        let src = if is_s_states {
            match last { GpuMemoryCache::Titans { s_states, .. } => s_states, _ => unreachable!() }
        } else {
            match last {
                GpuMemoryCache::Delta { m_states, .. } => m_states,
                GpuMemoryCache::Titans { m_states, .. } => m_states,
                GpuMemoryCache::Hebbian { m_states, .. } => m_states,
                _ => unreachable!(),
            }
        };
        src.clone_buf()
    } else {
        // Proxy: backward expects M_0, M_1, ..., M_s layout.
        // Each per-token cache at s=1 with proxy has [2 * chunk]: M_t and M_{t+1}.
        // Assemble: take M_t from each token, then M_s from the last.
        let total = (s + 1) * chunk;
        let out = GpuBuf::<f32>::zeros(total);

        for (t, cache) in caches.iter().enumerate() {
            let src = if is_s_states {
                match cache { GpuMemoryCache::Titans { s_states, .. } => s_states, _ => unreachable!() }
            } else {
                match cache {
                    GpuMemoryCache::Delta { m_states, .. } => m_states,
                    GpuMemoryCache::Titans { m_states, .. } => m_states,
                    _ => unreachable!(),
                }
            };
            // Copy M_t (first chunk of each per-token cache)
            let dst_offset = t * chunk * 4;
            unsafe {
                let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                    (out.ptr() as *mut u8).add(dst_offset) as *mut std::ffi::c_void,
                    src.as_ptr() as *const std::ffi::c_void,
                    chunk * 4,
                );
                assert_eq!(rc, 0);
            }
        }
        // Copy M_s from the last token's second chunk
        let last_src = if is_s_states {
            match caches.last().unwrap() { GpuMemoryCache::Titans { s_states, .. } => s_states, _ => unreachable!() }
        } else {
            match caches.last().unwrap() {
                GpuMemoryCache::Delta { m_states, .. } => m_states,
                GpuMemoryCache::Titans { m_states, .. } => m_states,
                _ => unreachable!(),
            }
        };
        let dst_offset = s * chunk * 4;
        let src_offset = chunk * 4; // second chunk of last token
        unsafe {
            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                (out.ptr() as *mut u8).add(dst_offset) as *mut std::ffi::c_void,
                (last_src.as_ptr() as *const u8).add(src_offset) as *const std::ffi::c_void,
                chunk * 4,
            );
            assert_eq!(rc, 0);
        }
        out
    }
}

// ── Single-token forward with activation caching ────────────────────

/// Process one token through all blocks, saving activations for backward.
///
/// This is the atomic operation of the unified forward path. Same computation
/// as `gpu_stacked_decode_token`, but snapshots workspace buffers into a
/// `TokenActivationCache` that can be assembled for backward.
///
/// Spec 68: "The model is: process one token → update memory → produce output.
/// That is the atomic unit of computation."
#[cfg(feature = "cuda")]
pub fn forward_single_token(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_id: usize,
    pulse: &Pulse,
    context: &mut GpuStackedContext,
    kv_caches: &mut [crate::gpu_forward::GpuKVCache],
    ws: &mut StackedDecodeWorkspace,
) -> TokenActivationCache {
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let window_size = cfg.swa.window_size;
    let n_blocks = params.n_blocks();
    let d_i32 = d as i32;

    assert!(token_id < v, "token_id {} >= vocab_size {}", token_id, v);
    assert_eq!(kv_caches.len(), n_blocks, "need one KV cache per block");

    // ── Embed 1 token ──────────────────────────────────────────────
    let input_i32 = [token_id as i32];
    unsafe {
        let rc = crate::gpu_forward::gpu_buf_memcpy_h2d(
            ws.d_input.ptr() as *mut std::ffi::c_void,
            input_i32.as_ptr() as *const std::ffi::c_void,
            4,
        );
        assert_eq!(rc, 0);
    }

    ws.residual.zero();
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.w_embed.as_ptr(),
            ws.d_input.ptr() as *const i32,
            ws.residual.ptr(),
            1, d_i32,
        );
    }

    // Save embedding
    let embedded = ws.residual.clone_buf();

    let is_chained = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);

    // ── Per-block forward ──────────────────────────────────────────
    let mut block_caches_out = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let block = &params.blocks[b];
        let block_ctx = &mut context.blocks[b];
        let kv = &mut kv_caches[b];
        let bws = &mut ws.per_block[b];

        // Save block input
        let block_input = ws.residual.clone_buf();

        // ── LN_attn ────────────────────────────────────────────────
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                ws.residual.as_ptr(),
                block.ln_attn_gamma.as_ptr(),
                block.ln_attn_beta.as_ptr(),
                bws.ln_attn_out.ptr(), bws.ln_attn_mean.ptr(), bws.ln_attn_rstd.ptr(),
                1, d_i32, 1e-5,
            );
        }

        // ── QKV projections ────────────────────────────────────────
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_q, &mut bws.q_f32, 1, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_k, &mut bws.k_f32, 1, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&bws.ln_attn_out, &block.w_v, &mut bws.v_f32, 1, d, d, 0.0);

        // ── Append K,V to cache ────────────────────────────────────
        kv.append_f32(&bws.k_f32, &bws.v_f32, 1);

        // ── SWA single-token attention ─────────────────────────────
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(bws.q_f32.as_ptr(), bws.q_bf16.ptr(), d_i32);
        }
        crate::dispatch::swa_single_token_dd(
            &bws.q_bf16, &kv.k_cache_bf16, &kv.v_cache_bf16,
            &mut bws.attn_out_bf16,
            kv.len, nh, hd, window_size,
        );
        unsafe {
            crate::cuda_ffi::bf16_to_f32_cuda(bws.attn_out_bf16.as_ptr(), bws.attn_out_f32.ptr(), d_i32);
        }

        // ── Output projection ──────────────────────────────────────
        crate::dispatch::cublas_matmul_transb_dd(&bws.attn_out_f32, &block.w_o, &mut bws.attn_proj, 1, d, d, 0.0);

        // ── Residual skip 1 ────────────────────────────────────────
        bws.residual_after_attn.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input.as_ptr(), bws.residual_after_attn.ptr(), d_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, bws.attn_proj.as_ptr(), bws.residual_after_attn.ptr(), d_i32);
        }

        // ── LN_mem ─────────────────────────────────────────────────
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                bws.residual_after_attn.as_ptr(),
                block.ln_mem_gamma.as_ptr(),
                block.ln_mem_beta.as_ptr(),
                bws.ln_mem_out.ptr(), bws.ln_mem_mean.ptr(), bws.ln_mem_rstd.ptr(),
                1, d_i32, 1e-5,
            );
        }

        // ── CMS memory per level (s=1) ─────────────────────────────
        let mut y_per_level: Vec<GpuBuf<f32>> = Vec::with_capacity(cfg.k);
        let mut memory_caches: Vec<Option<GpuMemoryCache>> = Vec::with_capacity(cfg.k);
        let mut level_seq_lens: Vec<usize> = Vec::with_capacity(cfg.k);

        if is_chained {
            bws.chain_h.zero();
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, bws.ln_mem_out.as_ptr(), bws.chain_h.ptr(), d_i32);
            }
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                if effective_active {
                    let (y_level, mem_cache) = gpu_memory_forward(
                        &block.levels[level], cfg, &bws.chain_h,
                        &mut block_ctx.memory[level],
                        1, d, level, 1,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(Some(mem_cache));
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &bws.chain_h,
                        &block_ctx.memory[level],
                        1, d, nh, hd,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                level_seq_lens.push(1);

                if level < cfg.k - 1 {
                    bws.chain_h.zero();
                    unsafe {
                        crate::cuda_ffi::saxpy_cuda(1.0, y_per_level[level].as_ptr(), bws.chain_h.ptr(), d_i32);
                    }
                }
            }
        } else {
            for level in 0..cfg.k {
                let effective_active = pulse.active_levels[level]
                    || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

                if effective_active {
                    let (y_level, mem_cache) = gpu_memory_forward(
                        &block.levels[level], cfg, &bws.ln_mem_out,
                        &mut block_ctx.memory[level],
                        1, d, level, 1,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(Some(mem_cache));
                } else {
                    let y_level = gpu_memory_read_only(
                        &block.levels[level], &bws.ln_mem_out,
                        &block_ctx.memory[level],
                        1, d, nh, hd,
                    );
                    y_per_level.push(y_level);
                    memory_caches.push(None);
                }
                level_seq_lens.push(1);
            }
        }

        // ── Level aggregation ──────────────────────────────────────
        let alpha_weights;
        if is_chained {
            bws.y_combined.zero();
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, y_per_level.last().unwrap().as_ptr(), bws.y_combined.ptr(), d_i32);
            }
            alpha_weights = vec![1.0]; // chain uses last level directly
        } else {
            let mut alpha_host = vec![0.0f32; cfg.k];
            block.alpha_mem.slice(0, cfg.k).copy_to_host(&mut alpha_host);
            alpha_weights = crate::stacked_model::host_softmax(&alpha_host);
            bws.y_combined.zero();
            for (l, y_level) in y_per_level.iter().enumerate() {
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(alpha_weights[l], y_level.as_ptr(), bws.y_combined.ptr(), d_i32);
                }
            }
        }

        // ── MAG sigmoid gating ─────────────────────────────────────
        unsafe {
            crate::cuda_ffi::sigmoid_cuda(bws.y_combined.as_ptr(), bws.gate.ptr(), d_i32);
            crate::cuda_ffi::elemwise_mul_cuda(bws.attn_proj.as_ptr(), bws.gate.as_ptr(), bws.gated_out.ptr(), d_i32);
        }

        // ── Residual skip 2 ───────────────────────────────────────
        bws.residual_out.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, block_input.as_ptr(), bws.residual_out.ptr(), d_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, bws.gated_out.as_ptr(), bws.residual_out.ptr(), d_i32);
        }
        ws.residual.zero();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, bws.residual_out.as_ptr(), ws.residual.ptr(), d_i32);
        }

        // ── Snapshot workspace into activation cache ───────────────
        // Clone bf16 buffers for Q (K/V already saved as f32 above)
        let k_bf16_save = GpuBuf::<u16>::zeros(d);
        let v_bf16_save = GpuBuf::<u16>::zeros(d);
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(bws.k_f32.as_ptr(), k_bf16_save.ptr(), d_i32);
            crate::cuda_ffi::f32_to_bf16_cuda(bws.v_f32.as_ptr(), v_bf16_save.ptr(), d_i32);
        }

        block_caches_out.push(TokenBlockCache {
            block_input,
            q_f32: bws.q_f32.clone_buf(),
            k_f32: bws.k_f32.clone_buf(),
            v_f32: bws.v_f32.clone_buf(),
            q_bf16: bws.q_bf16.clone_buf(),
            k_bf16: k_bf16_save,
            v_bf16: v_bf16_save,
            attn_out_bf16: bws.attn_out_bf16.clone_buf(),
            attn_out: bws.attn_out_f32.clone_buf(),
            attn_proj: bws.attn_proj.clone_buf(),
            ln_attn_out: bws.ln_attn_out.clone_buf(),
            ln_attn_mean: bws.ln_attn_mean.clone_buf(),
            ln_attn_rstd: bws.ln_attn_rstd.clone_buf(),
            ln_mem_out: bws.ln_mem_out.clone_buf(),
            ln_mem_mean: bws.ln_mem_mean.clone_buf(),
            ln_mem_rstd: bws.ln_mem_rstd.clone_buf(),
            memory_caches,
            y_per_level,
            level_seq_lens,
            y_combined: bws.y_combined.clone_buf(),
            gate: bws.gate.clone_buf(),
            alpha_weights,
            residual_after_attn: bws.residual_after_attn.clone_buf(),
        });
    }

    // ── Final LN ───────────────────────────────────────────────────
    unsafe {
        crate::cuda_ffi::layer_norm_forward_cuda(
            ws.residual.as_ptr(),
            params.ln_final_gamma.as_ptr(),
            params.ln_final_beta.as_ptr(),
            ws.ln_final_out.ptr(), ws.ln_final_mean.ptr(), ws.ln_final_rstd.ptr(),
            1, d_i32, 1e-5,
        );
    }

    // ── Unembed ────────────────────────────────────────────────────
    crate::dispatch::cublas_matmul_dd(&ws.ln_final_out, &params.w_unembed, &mut ws.logits_gpu, 1, d, v, 0.0);
    crate::dispatch::cuda_sync();

    TokenActivationCache {
        block_caches: block_caches_out,
        embedded,
        token_id,
        token_id_i32: token_id as i32,
        ln_final_out: ws.ln_final_out.clone_buf(),
        ln_final_mean: ws.ln_final_mean.clone_buf(),
        ln_final_rstd: ws.ln_final_rstd.clone_buf(),
        logits: ws.logits_gpu.clone_buf(),
        pulse: pulse.clone(),
    }
}

// ── Unified entry point ─────────────────────────────────────────────

/// Process N tokens through the model. N can be 1 (generation) or any count (training).
/// Same function, same kernels, same memory update, always.
///
/// Returns logits for the last token (for sampling during generation).
///
/// When `save_activations` is true, each token's activations are pushed into the
/// activation window for later backward. When false (pure inference without learning),
/// tokens are processed but only logits are returned.
///
/// Spec 68: "The model is: process one token → update memory → produce output.
/// That is the atomic unit of computation. Everything else is an optimization."
#[cfg(feature = "cuda")]
pub fn gpu_stacked_forward_tokens(
    params: &GpuStackedParams,
    cfg: &MAGConfig,
    token_ids: &[usize],
    conductor: &mut Conductor,
    context: &mut GpuStackedContext,
    kv_caches: &mut [crate::gpu_forward::GpuKVCache],
    ws: &mut StackedDecodeWorkspace,
    mut activation_window: Option<&mut ActivationWindow>,
) -> Vec<f32> {
    let v = cfg.swa.vocab_size;
    let mut last_logits = vec![0.0f32; v];

    let save = activation_window.is_some();

    for &token in token_ids {
        let pulse = conductor.pulse();

        if save {
            let token_cache = forward_single_token(
                params, cfg, token, &pulse, context, kv_caches, ws,
            );
            token_cache.logits.copy_to_host(&mut last_logits);
            activation_window.as_mut().unwrap().push(token_cache);
        } else {
            last_logits = gpu_stacked_decode_token(
                params, cfg, token, &pulse, context, kv_caches, ws,
            );
        }

        conductor.advance();
    }

    last_logits
}
