/// GPU-resident CMS backward pass and weight update.
///
/// Mirrors `cms_backward` in mag.rs but operates entirely on device pointers.
/// Produces `GpuMAGGrads` (gradient buffers on GPU), consumed by `gpu_weight_update`.
///
/// Only supports matrix-based rules: DeltaRule, TitansLMM, HebbianRule.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuMAGParams, GpuMemoryLevelParams};
#[cfg(feature = "cuda")]
use crate::gpu_forward::{GpuCMSCache, GpuMemoryCache, gpu_buf_memcpy_d2d};
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind};

// ══════════════════════════════════════════════════════════════════════
// GpuMAGGrads — gradient buffers on GPU
// ══════════════════════════════════════════════════════════════════════

/// Gradient buffers on GPU. Same shape as GpuMAGParams.
/// Created by backward, consumed by weight update, then dropped.
#[cfg(feature = "cuda")]
pub struct GpuMAGGrads {
    // SWA gradients
    pub d_w_embed: GpuBuf<f32>,
    pub d_w_q: GpuBuf<f32>,
    pub d_w_k: GpuBuf<f32>,
    pub d_w_v: GpuBuf<f32>,
    pub d_w_o: GpuBuf<f32>,
    pub d_w_unembed: GpuBuf<f32>,
    // LayerNorm gradients (residual path only; zeros when residual=false)
    pub d_ln_attn_gamma: GpuBuf<f32>,
    pub d_ln_attn_beta: GpuBuf<f32>,
    pub d_ln_mem_gamma: GpuBuf<f32>,
    pub d_ln_mem_beta: GpuBuf<f32>,
    // Per-level memory gradients
    pub levels: Vec<GpuLevelGrads>,
    /// Per-level L2 norm of d_y_combined (output gradient entering each level's
    /// backward). Length k. 0.0 for inactive levels. Populated by gpu_cms_backward().
    pub level_output_gnorms: Vec<f32>,
}

/// Per-level gradient buffers on GPU.
#[cfg(feature = "cuda")]
pub struct GpuLevelGrads {
    pub d_w_k_mem: GpuBuf<f32>,
    pub d_w_v_mem: GpuBuf<f32>,
    pub d_w_q_mem: GpuBuf<f32>,
    pub d_w_alpha: GpuBuf<f32>,
    pub d_b_alpha: GpuBuf<f32>,
    pub d_w_theta: GpuBuf<f32>,
    pub d_b_theta: GpuBuf<f32>,
    pub d_w_eta: GpuBuf<f32>,
    pub d_b_eta: GpuBuf<f32>,
    // SwiGluMlp weight grads. zeros(1) for non-SwiGLU levels.
    pub d_gate_proj: GpuBuf<f32>,
    pub d_up_proj:   GpuBuf<f32>,
    pub d_down_proj: GpuBuf<f32>,
    pub has_mlp: bool,
}

#[cfg(feature = "cuda")]
impl GpuMAGGrads {
    /// Download GPU gradients to host as a MAGParams (same struct, gradient values).
    pub fn to_host(&self, cfg: &crate::model::MAGConfig) -> crate::model::MAGParams {
        // Use config-aware constructor so all fields (including m_*_init for
        // adaptive projections, w_freq for learned schedules, etc.) are properly
        // sized. Fields not computed on GPU remain zero — the CPU optimizer
        // applies zero gradients harmlessly.
        let mut result = crate::model::MAGParams::zeros_like(cfg);

        self.d_w_embed.copy_to_host(&mut result.swa.w_embed);
        self.d_w_q.copy_to_host(&mut result.swa.w_q);
        self.d_w_k.copy_to_host(&mut result.swa.w_k);
        self.d_w_v.copy_to_host(&mut result.swa.w_v);
        self.d_w_o.copy_to_host(&mut result.swa.w_o);
        self.d_w_unembed.copy_to_host(&mut result.swa.w_unembed);
        self.d_ln_attn_gamma.copy_to_host(&mut result.swa.ln_attn_gamma);
        self.d_ln_attn_beta.copy_to_host(&mut result.swa.ln_attn_beta);
        self.d_ln_mem_gamma.copy_to_host(&mut result.swa.ln_mem_gamma);
        self.d_ln_mem_beta.copy_to_host(&mut result.swa.ln_mem_beta);

        for (i, lg) in self.levels.iter().enumerate() {
            let lp = &mut result.levels[i];
            lg.d_w_k_mem.copy_to_host(lp.w_k_mem.master_mut()); lp.w_k_mem.sync_from_master();
            lg.d_w_v_mem.copy_to_host(lp.w_v_mem.master_mut()); lp.w_v_mem.sync_from_master();
            lg.d_w_q_mem.copy_to_host(lp.w_q_mem.master_mut()); lp.w_q_mem.sync_from_master();
            lg.d_w_alpha.copy_to_host(&mut lp.w_alpha);
            lg.d_b_alpha.copy_to_host(&mut lp.b_alpha);
            lg.d_w_theta.copy_to_host(&mut lp.w_theta);
            lg.d_b_theta.copy_to_host(&mut lp.b_theta);
            lg.d_w_eta.copy_to_host(&mut lp.w_eta);
            lg.d_b_eta.copy_to_host(&mut lp.b_eta);
            if lg.has_mlp {
                lg.d_gate_proj.copy_to_host(&mut lp.gate_proj);
                lg.d_up_proj.copy_to_host(&mut lp.up_proj);
                lg.d_down_proj.copy_to_host(&mut lp.down_proj);
            }
            // w_omega, m_*_init, w_freq, w_*_conv: not in GPU grads yet, stay zero
        }

        result
    }
}

#[cfg(feature = "cuda")]
impl GpuLevelGrads {
    /// Allocate zero-initialized gradient buffers for one CMS level.
    /// For SwiGLU levels, pass inter = cfg.intermediate_size.
    /// For matrix rules (Delta/Titans/Hebbian/DGD), pass inter = 0.
    pub(crate) fn zeros_mlp(d: usize, inter: usize) -> Self {
        GpuLevelGrads {
            d_w_k_mem: GpuBuf::zeros(d * d),
            d_w_v_mem: GpuBuf::zeros(d * d),
            d_w_q_mem: GpuBuf::zeros(d * d),
            d_w_alpha: GpuBuf::zeros(2 * d),
            d_b_alpha: GpuBuf::zeros(1),
            d_w_theta: GpuBuf::zeros(2 * d),
            d_b_theta: GpuBuf::zeros(1),
            d_w_eta: GpuBuf::zeros(2 * d),
            d_b_eta: GpuBuf::zeros(1),
            d_gate_proj: GpuBuf::zeros(if inter > 0 { inter * d } else { 1 }),
            d_up_proj:   GpuBuf::zeros(if inter > 0 { inter * d } else { 1 }),
            d_down_proj: GpuBuf::zeros(if inter > 0 { d * inter } else { 1 }),
            has_mlp: inter > 0,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GPU-resident CMS backward
// ══════════════════════════════════════════════════════════════════════

/// GPU-resident CMS backward pass. All on device.
///
/// Consumes the forward cache and produces gradient buffers.
/// Frozen levels: gradients for read-only path are simpler (just d_W_q_mem, d_embedded).
#[cfg(feature = "cuda")]
pub fn gpu_cms_backward(
    params: &GpuMAGParams,
    cfg: &MAGConfig,
    cache: &GpuCMSCache,
    collect_output_gnorms: bool,
    // Note: error_buffers not supported in GPU path — frozen levels
    // only get q_mem projection gradient which is accumulated in d_embedded.
) -> GpuMAGGrads {
    let s = cache.s;
    let d = cache.d;
    let v = cache.v;
    let nh = cache.nh;
    let hd = cache.hd;
    let ws = cache.ws;
    let bs = cache.batch_size;
    let sd = s * d;
    let bsd = bs * sd;
    let bsv = bs * s * v;

    // Initialize gradient buffers (all zeros on GPU)
    let mut grads = GpuMAGGrads {
        d_w_embed: GpuBuf::zeros(v * d),
        d_w_q: GpuBuf::zeros(d * d),
        d_w_k: GpuBuf::zeros(d * d),
        d_w_v: GpuBuf::zeros(d * d),
        d_w_o: GpuBuf::zeros(d * d),
        d_w_unembed: GpuBuf::zeros(d * v),
        d_ln_attn_gamma: GpuBuf::zeros(d),
        d_ln_attn_beta: GpuBuf::zeros(d),
        d_ln_mem_gamma: GpuBuf::zeros(d),
        d_ln_mem_beta: GpuBuf::zeros(d),
        levels: {
            let inter = if cfg.memory_rule == MemoryRuleKind::SwiGluMlp { cfg.intermediate_size } else { 0 };
            (0..cfg.k).map(|_| GpuLevelGrads::zeros_mlp(d, inter)).collect()
        },
        level_output_gnorms: vec![0.0f32; cfg.k],
    };

    // ── Stage 7: Cross-entropy backward ──────────────────────────────
    let mut d_logits = GpuBuf::zeros(bsv);
    // Count valid targets (masked targets with id < 0 or >= vocab are skipped
    // by the kernel — we must normalize by the same count as forward).
    let valid_count = cache.target_ids_i32.iter()
        .filter(|&&t| t >= 0 && (t as usize) < v)
        .count() as f32;
    let count = if valid_count > 0.0 { valid_count } else { 1.0 };
    unsafe {
        crate::cuda_ffi::cross_entropy_backward_cuda(
            cache.logits.as_ptr(),
            cache.target_ids_gpu.ptr() as *const i32,
            d_logits.ptr(),
            (bs * s) as i32, v as i32,
            1.0 / count,
        );
    }

    // ── Stage 6: Unembed backward ────────────────────────────────────
    // d_projected = d_logits @ W_unembed^T (transB: W_unembed is [d, v], so C = d_logits[bs*s,v] @ W_unembed^T[v,d])
    let mut d_projected = GpuBuf::zeros(bsd);
    crate::dispatch::cublas_matmul_transb_dd(
        &d_logits, &params.swa.w_unembed, &mut d_projected, bs * s, v, d, 0.0,
    );

    // d_w_unembed = projected^T @ d_logits → [d, bs*s] @ [bs*s, v] = [d, v]
    gpu_matmul_transa_dd(
        &cache.projected, &d_logits, &mut grads.d_w_unembed,
        d, bs * s, v,
    );

    // ── Stage 5: Output projection backward ──────────────────────────
    let mut d_gated_out = GpuBuf::zeros(bsd);

    if cfg.residual {
        // Residual path: projected = residual_final @ W_O^T
        // d_residual_final = d_projected @ W_O
        let residual_final = cache.residual_final.as_ref()
            .expect("residual_final must be Some when cfg.residual=true");
        crate::dispatch::cublas_matmul_dd(
            &d_projected, &params.swa.w_o, &mut d_gated_out, bs * s, d, d, 0.0,
        );
        // d_w_o = d_projected^T @ residual_final
        gpu_matmul_transa_dd(
            &d_projected, residual_final, &mut grads.d_w_o,
            d, bs * s, d,
        );
    } else {
        // Legacy path: projected = gated_out @ W_O^T
        crate::dispatch::cublas_matmul_dd(
            &d_projected, &params.swa.w_o, &mut d_gated_out, bs * s, d, d, 0.0,
        );
        gpu_matmul_transa_dd(
            &d_projected, &cache.gated_out, &mut grads.d_w_o,
            d, bs * s, d,
        );
    }

    // ── Stage 4: Gating / residual gradient routing ───────────────────
    let mut d_attn_out;
    let d_y_combined;

    if cfg.residual {
        // Residual path: d_gated_out is d_residual_final
        // residual_final = residual_after_attn + y_combined (additions)
        // d_y_combined = d_residual_final (gradient = 1.0 through addition)
        // d_residual_after_attn = d_residual_final (gradient = 1.0 through addition)
        d_y_combined = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_gated_out.as_ptr(), d_y_combined.ptr(), bsd as i32);
        }
        let mut d_residual_after_attn = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_gated_out.as_ptr(), d_residual_after_attn.ptr(), bsd as i32);
        }
        // We'll use d_residual_after_attn later as d_attn_out (it splits into d_embedded + d_attn_out)
        d_attn_out = d_residual_after_attn;
    } else {
        // Legacy: gated_out = attn_out * gate → gating_backward + sigmoid_backward
        d_attn_out = GpuBuf::zeros(bsd);
        let mut d_gate = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::gating_backward_cuda(
                d_gated_out.as_ptr(), cache.attn_out.as_ptr(), cache.gate.as_ptr(),
                d_attn_out.ptr(), d_gate.ptr(), bsd as i32,
            );
        }
        d_y_combined = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::sigmoid_backward_cuda(
                d_gate.as_ptr(), cache.gate.as_ptr(), d_y_combined.ptr(), bsd as i32,
            );
        }
    }

    // Scale for 1/sqrt(k) normalization (k>2)
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(
                scale - 1.0, d_y_combined.as_ptr(), d_y_combined.ptr(), bsd as i32,
            );
        }
    }

    // ── Capture d_y_combined L2 norm for GPU tape summary (opt-in) ──
    if collect_output_gnorms {
        let max_blocks = (bsd + 255) / 256;
        let mut scratch = GpuBuf::zeros(max_blocks);
        let mut num_blocks: i32 = 0;
        let err = unsafe {
            crate::cuda_ffi::grad_norm_sq_cuda(
                d_y_combined.as_ptr(), scratch.ptr(), bsd as i32, &mut num_blocks,
            )
        };
        assert_eq!(err, 0, "grad_norm_sq_cuda for d_y_combined failed");
        crate::dispatch::cuda_sync();
        let nb = num_blocks as usize;
        let mut host = vec![0.0f32; nb];
        scratch.slice(0, nb).copy_to_host(&mut host);
        let sq_sum: f64 = host.iter().map(|x| *x as f64).sum();
        let d_y_norm = sq_sum.sqrt() as f32;
        for level in 0..cfg.k {
            if cache.pulse.active_levels[level] {
                grads.level_output_gnorms[level] = d_y_norm;
            }
        }
    }

    // ── Stage 3b: Per-level memory backward ──────────────────────────
    // Memory input source: LN(residual) for residual, embedded for legacy
    let mem_input_ref = if cfg.residual {
        cache.ln_mem_out.as_ref().expect("ln_mem_out must be Some when cfg.residual=true")
    } else {
        &cache.embedded
    };
    let mut d_mem_input = GpuBuf::zeros(bsd);

    for level in 0..cfg.k {
        if cache.pulse.active_levels[level] {
            if let Some(ref mem_cache) = cache.memory_caches[level] {
                let d_emb_level = gpu_memory_backward(
                    &params.levels[level], cfg, mem_cache,
                    &d_y_combined, mem_input_ref,
                    &mut grads.levels[level],
                    s, d, level, bs,
                );
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd as i32);
                }
            }
        } else {
            let d_emb_level = gpu_memory_read_only_backward(
                &params.levels[level], &cache.y_per_level[level],
                &d_y_combined, mem_input_ref,
                &mut grads.levels[level],
                s, d, bs,
            );
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_mem_input.ptr(), bsd as i32);
            }
        }
    }

    // ── LN_mem backward (residual path) ──────────────────────────────
    if cfg.residual {
        let residual_after_attn = cache.residual_after_attn.as_ref()
            .expect("residual_after_attn must be Some");
        let ln_mem_mean = cache.ln_mem_mean.as_ref().expect("ln_mem_mean");
        let ln_mem_rstd = cache.ln_mem_rstd.as_ref().expect("ln_mem_rstd");
        let mut d_residual_from_mem = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::layer_norm_backward_cuda(
                d_mem_input.as_ptr(),
                residual_after_attn.as_ptr(),
                params.swa.ln_mem_gamma.as_ptr(),
                ln_mem_mean.as_ptr(),
                ln_mem_rstd.as_ptr(),
                d_residual_from_mem.ptr(),
                grads.d_ln_mem_gamma.ptr(),
                grads.d_ln_mem_beta.ptr(),
                (bs * s) as i32, d as i32,
            );
        }
        // d_attn_out already holds d_residual_after_attn from the addition backward.
        // Add the LN_mem backward contribution to it.
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_residual_from_mem.as_ptr(), d_attn_out.ptr(), bsd as i32);
        }
    }

    // ── Stage 3a: SWA backward ───────────────────────────────────────
    let mut d_q = GpuBuf::zeros(bsd);
    let mut d_k = GpuBuf::zeros(bsd);
    let mut d_v = GpuBuf::zeros(bsd);

    crate::dispatch::swa_backward_dd(
        &cache.q_bf16, &cache.k_bf16, &cache.v_bf16,
        &cache.attn_weights_bf16, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws, bs,
    );

    // ── Stage 2a: QKV projection backward ────────────────────────────
    // QKV source was LN(embedded) for residual, raw embedded for legacy
    let qkv_source = if cfg.residual {
        cache.ln_attn_out.as_ref().expect("ln_attn_out must be Some")
    } else {
        &cache.embedded
    };
    let mut d_qkv_source = GpuBuf::zeros(bsd);
    crate::dispatch::cublas_matmul_acc_dd(&d_q, &params.swa.w_q, &mut d_qkv_source, bs * s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(&d_k, &params.swa.w_k, &mut d_qkv_source, bs * s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(&d_v, &params.swa.w_v, &mut d_qkv_source, bs * s, d, d);

    // d_w_q = d_q^T @ qkv_source
    gpu_matmul_transa_dd(&d_q, qkv_source, &mut grads.d_w_q, d, bs * s, d);
    gpu_matmul_transa_dd(&d_k, qkv_source, &mut grads.d_w_k, d, bs * s, d);
    gpu_matmul_transa_dd(&d_v, qkv_source, &mut grads.d_w_v, d, bs * s, d);

    // ── LN_attn backward (residual path) ─────────────────────────────
    let mut d_embedded = GpuBuf::zeros(bsd);
    if cfg.residual {
        // d_qkv_source flows through LN_attn backward to d_embedded
        let ln_attn_mean = cache.ln_attn_mean.as_ref().expect("ln_attn_mean");
        let ln_attn_rstd = cache.ln_attn_rstd.as_ref().expect("ln_attn_rstd");
        unsafe {
            crate::cuda_ffi::layer_norm_backward_cuda(
                d_qkv_source.as_ptr(),
                cache.embedded.as_ptr(),
                params.swa.ln_attn_gamma.as_ptr(),
                ln_attn_mean.as_ptr(),
                ln_attn_rstd.as_ptr(),
                d_embedded.ptr(),
                grads.d_ln_attn_gamma.ptr(),
                grads.d_ln_attn_beta.ptr(),
                (bs * s) as i32, d as i32,
            );
        }
        // Also add d_attn_out (which is d_residual_after_attn) to d_embedded
        // because residual_after_attn = embedded + attn_out → d_embedded += d_residual_after_attn
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_attn_out.as_ptr(), d_embedded.ptr(), bsd as i32);
        }
    } else {
        // Legacy: d_embedded from QKV projections
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_qkv_source.as_ptr(), d_embedded.ptr(), bsd as i32);
        }
        // Combine d_embedded from memory branch (legacy only — residual handles this through LN)
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_mem_input.as_ptr(), d_embedded.ptr(), bsd as i32);
        }
    }

    // ── Stage 1: Embedding scatter-add ───────────────────────────────
    unsafe {
        crate::cuda_ffi::embedding_scatter_add_cuda(
            d_embedded.as_ptr(),
            cache.input_ids_gpu.ptr() as *const i32,
            grads.d_w_embed.ptr(),
            (bs * s) as i32, d as i32,
        );
    }

    crate::dispatch::cuda_sync();
    grads
}

// ══════════════════════════════════════════════════════════════════════
// Memory backward helpers (GPU-resident)
// ══════════════════════════════════════════════════════════════════════

/// Active level backward: full gradient through memory rule inner loop.
/// Returns d_embedded contribution from memory projections.
/// `batch_size` is the number of sequences processed in parallel this step.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_backward(
    level_params: &GpuMemoryLevelParams,
    cfg: &MAGConfig,
    mem_cache: &GpuMemoryCache,
    d_y: &GpuBuf<f32>,
    embedded: &GpuBuf<f32>,
    level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
    level: usize,
    batch_size: usize,
) -> GpuBuf<f32> {
    let dd = d * d;
    let bsd = batch_size * s * d;
    let bs_s = batch_size * s;

    match mem_cache {
        GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms } => {
            let mut d_k_mem = GpuBuf::zeros(bsd);
            let mut d_v_mem = GpuBuf::zeros(bsd);
            let mut d_q_mem = GpuBuf::zeros(bsd);
            let mut d_alpha = GpuBuf::zeros(bs_s);
            let mut d_theta = GpuBuf::zeros(bs_s);
            let mut d_m_initial = GpuBuf::zeros(dd);

            crate::dispatch::delta_backward_dd(
                k_mem, v_mem, q_mem, alpha, theta,
                m_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_theta, &mut d_m_initial,
                s, d, batch_size,
                cfg.error_clip_for_level(level),
            );

            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }

            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, k_norms, q_norms } => {
            let mut d_k_mem = GpuBuf::zeros(bsd);
            let mut d_v_mem = GpuBuf::zeros(bsd);
            let mut d_q_mem = GpuBuf::zeros(bsd);
            let mut d_alpha = GpuBuf::zeros(bs_s);
            let mut d_theta = GpuBuf::zeros(bs_s);
            let mut d_eta = GpuBuf::zeros(bs_s);
            let mut d_m_initial = GpuBuf::zeros(dd);
            let mut d_s_initial = GpuBuf::zeros(dd);

            crate::dispatch::titans_backward_dd(
                k_mem, v_mem, q_mem, alpha, theta, eta,
                m_states, s_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_theta, &mut d_eta,
                &mut d_m_initial, &mut d_s_initial,
                s, d, batch_size,
                cfg.error_clip_for_level(level),
            );

            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }

            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), Some(eta),
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), Some(&d_eta),
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states, k_norms, q_norms } => {
            // Hebbian kernels don't yet support batch_size > 1
            assert_eq!(batch_size, 1, "Hebbian batch_size > 1 not yet supported");
            let sd = s * d;
            let mut d_k_mem = GpuBuf::zeros(sd);
            let mut d_v_mem = GpuBuf::zeros(sd);
            let mut d_q_mem = GpuBuf::zeros(sd);
            let mut d_alpha = GpuBuf::zeros(s);
            let mut d_m_initial = GpuBuf::zeros(dd);

            crate::dispatch::hebbian_backward_dd(
                k_mem, v_mem, q_mem, alpha, m_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_m_initial,
                s, d,
            );

            // Hebbian has no theta or eta — pass None
            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, None, None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, None, None,
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        // ── Checkpointed variants: segment-based backward ──────────
        GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval, k_norms, q_norms } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, mut d_alpha, mut d_theta) =
                delta_backward_checkpointed(k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, d_y, s, d, c, cfg.error_clip_for_level(level));
            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }
            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        GpuMemoryCache::TitansCkpt { k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, checkpoint_interval, k_norms, q_norms } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, mut d_alpha, mut d_theta, d_eta) =
                titans_backward_checkpointed(k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, d_y, s, d, c, cfg.error_clip_for_level(level));
            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }
            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), Some(eta),
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), Some(&d_eta),
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        GpuMemoryCache::HebbianCkpt { k_mem, v_mem, q_mem, alpha, m_checkpoints, checkpoint_interval, k_norms, q_norms } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, d_alpha) =
                hebbian_backward_checkpointed(k_mem, v_mem, q_mem, alpha, m_checkpoints, d_y, s, d, c);
            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, None, None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, None, None,
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        // ── DGD: same structure as Delta (uses delta_backward kernels) ──
        GpuMemoryCache::DGD { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms } => {
            let mut d_k_mem = GpuBuf::zeros(bsd);
            let mut d_v_mem = GpuBuf::zeros(bsd);
            let mut d_q_mem = GpuBuf::zeros(bsd);
            let mut d_alpha = GpuBuf::zeros(bs_s);
            let mut d_theta = GpuBuf::zeros(bs_s);
            let mut d_m_initial = GpuBuf::zeros(dd);

            crate::dispatch::delta_backward_dd(
                k_mem, v_mem, q_mem, alpha, theta,
                m_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_theta, &mut d_m_initial,
                s, d, batch_size,
                cfg.error_clip_for_level(level),
            );

            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }

            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        GpuMemoryCache::DGDCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval, k_norms, q_norms } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, mut d_alpha, mut d_theta) =
                delta_backward_checkpointed(k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, d_y, s, d, c, cfg.error_clip_for_level(level));
            // CS-39 straight-through: zero d_alpha where alpha was clamped.
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            // CS-39 straight-through: zero d_theta where theta was clamped.
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }
            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        // ── TNT: reverse shard loop with batched inner backward ──────
        GpuMemoryCache::TNT {
            shard_inner_caches, k_summaries, v_summaries,
            global_chunk_size, local_chunk_size,
            total_shards, first_retained_shard,
        } => {
            assert_eq!(batch_size, 1, "TNT backward currently supports batch_size=1 only");
            let cg = *global_chunk_size;
            let cl = *local_chunk_size;
            let n_total = *total_shards;
            let first_ret = *first_retained_shard;
            let n_retained = shard_inner_caches.len();
            // Spec 25 invariants: retained window must be consistent.
            assert!(n_total > 0, "TNT backward: total_shards must be > 0");
            assert!(first_ret < n_total, "TNT backward: first_retained_shard ({first_ret}) >= total_shards ({n_total})");
            assert_eq!(n_retained, n_total - first_ret, "TNT backward: cache count ({n_retained}) != total_shards - first_retained ({} - {first_ret})", n_total);

            // Accumulators for projection weight grads across all shards
            let mut d_k_mem_total = GpuBuf::<f32>::zeros(s * d);
            let mut d_v_mem_total = GpuBuf::<f32>::zeros(s * d);
            let mut d_q_mem_total = GpuBuf::<f32>::zeros(s * d);

            // Reverse shard iteration: propagate d_m through global updates.
            // Global M backward runs over ALL shards (summaries are retained for all).
            // Inner backward only runs for retained shards (gradient truncation for evicted ones).
            let mut d_m_carry = GpuBuf::zeros(dd);

            for shard_idx in (0..n_total).rev() {
                let shard_start = shard_idx * cg;
                let shard_end = (shard_start + cg).min(s);
                let shard_len = shard_end - shard_start;
                let n_batch = (shard_len + cl - 1) / cl;

                // Step 1: Backward through global M update (runs for ALL shards —
                // summaries are retained even for evicted shards).
                // Forward: m_new = alpha * m_old + v_sum ⊗ k_sum
                // d_m_new = d_m_carry (gradient from subsequent shards)
                let mut d_m_old = GpuBuf::zeros(dd);
                let mut d_k_sum = GpuBuf::zeros(d);
                let mut d_v_sum = GpuBuf::zeros(d);
                crate::dispatch::tnt_global_update_backward_dd(
                    &d_m_carry, &k_summaries[shard_idx], &v_summaries[shard_idx],
                    &mut d_m_old, &mut d_k_sum, &mut d_v_sum, d, 0.95,
                );

                // Spec 25: check if this shard's inner cache was retained.
                // Evicted shards (shard_idx < first_ret) only contribute global M gradients.
                // Inner backward (projection/gate grads, local M grads) is truncated.
                if shard_idx < first_ret {
                    // Gradient truncation: only global M backward for evicted shards.
                    // Still propagate d_m_carry so the global chain rule is complete.
                    d_m_carry = d_m_old;
                    continue;
                }

                // Map absolute shard_idx to the retained cache index.
                let cache_idx = shard_idx - first_ret;

                // Step 2: Backward through shard summary mean
                let mut d_local_y_global = GpuBuf::zeros(shard_len * d);
                crate::dispatch::tnt_shard_summary_mean_backward_dd(
                    &d_k_sum, &d_v_sum, &mut d_local_y_global, shard_len, d,
                );

                // Step 3: Combine upstream d_y with d_local_y from global path
                let d_y_shard_slice = d_y.slice(shard_start * d, shard_len * d);
                let mut d_y_upstream_shard = GpuBuf::zeros(shard_len * d);
                unsafe {
                    let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                        d_y_upstream_shard.ptr() as *mut std::ffi::c_void,
                        d_y_shard_slice.as_ptr() as *const std::ffi::c_void,
                        shard_len * d * 4,
                    );
                    assert_eq!(rc, 0, "TNT backward: d_y upstream copy failed (rc={rc})");
                }
                let mut d_y_combined = GpuBuf::zeros(shard_len * d);
                crate::dispatch::tnt_combine_gradients_dd(
                    &d_y_upstream_shard, &d_local_y_global,
                    &mut d_y_combined, shard_len * d,
                );

                // Step 4: Pad d_y_combined to [n_batch, cl, d] layout if needed
                let padded_len = n_batch * cl;
                let d_y_padded = if shard_len == padded_len {
                    d_y_combined
                } else {
                    let mut dp = GpuBuf::zeros(padded_len * d);
                    unsafe {
                        let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                            dp.ptr() as *mut std::ffi::c_void,
                            d_y_combined.as_ptr() as *const std::ffi::c_void,
                            shard_len * d * 4,
                        );
                        assert_eq!(rc, 0, "TNT backward: d_y padding copy failed (rc={rc})");
                    }
                    dp
                };

                // Step 5: Run inner backward kernel (Titans/Delta with batch_size=n_batch)
                let inner_cache = &shard_inner_caches[cache_idx];
                let shard_tokens = padded_len;
                let mut d_k_shard = GpuBuf::zeros(shard_tokens * d);
                let mut d_v_shard = GpuBuf::zeros(shard_tokens * d);
                let mut d_q_shard = GpuBuf::zeros(shard_tokens * d);
                let mut d_alpha_shard = GpuBuf::zeros(shard_tokens);
                let mut d_theta_shard = GpuBuf::zeros(shard_tokens);
                let mut d_eta_shard = GpuBuf::zeros(shard_tokens);
                let has_eta = matches!(inner_cache, GpuMemoryCache::Titans { .. });

                let (shard_k_norms, shard_q_norms, shard_k_mem, shard_q_mem) = match inner_cache {
                    GpuMemoryCache::Titans { k_mem, q_mem, k_norms, q_norms, .. } => (k_norms, q_norms, k_mem, q_mem),
                    GpuMemoryCache::Delta { k_mem, q_mem, k_norms, q_norms, .. } => (k_norms, q_norms, k_mem, q_mem),
                    _ => unreachable!("TNT inner cache must be Titans or Delta"),
                };

                match inner_cache {
                    GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, .. } => {
                        let mut d_m_initial = GpuBuf::zeros(n_batch * dd);
                        let mut d_s_initial = GpuBuf::zeros(n_batch * dd);

                        crate::dispatch::titans_backward_dd(
                            k_mem, v_mem, q_mem, alpha, theta, eta,
                            m_states, s_states, &d_y_padded,
                            &mut d_k_shard, &mut d_v_shard, &mut d_q_shard,
                            &mut d_alpha_shard, &mut d_theta_shard, &mut d_eta_shard,
                            &mut d_m_initial, &mut d_s_initial,
                            cl, d, n_batch,
                            cfg.error_clip_for_level(level),
                        );

                        for b in 0..n_batch {
                            unsafe {
                                crate::cuda_ffi::saxpy_cuda(
                                    1.0, d_m_initial.as_ptr().add(b * dd), d_m_old.ptr(), dd as i32,
                                );
                            }
                        }
                    }
                    GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, .. } => {
                        let mut d_m_initial = GpuBuf::zeros(n_batch * dd);

                        crate::dispatch::delta_backward_dd(
                            k_mem, v_mem, q_mem, alpha, theta,
                            m_states, &d_y_padded,
                            &mut d_k_shard, &mut d_v_shard, &mut d_q_shard,
                            &mut d_alpha_shard, &mut d_theta_shard, &mut d_m_initial,
                            cl, d, n_batch,
                            cfg.error_clip_for_level(level),
                        );

                        for b in 0..n_batch {
                            unsafe {
                                crate::cuda_ffi::saxpy_cuda(
                                    1.0, d_m_initial.as_ptr().add(b * dd), d_m_old.ptr(), dd as i32,
                                );
                            }
                        }
                    }
                    _ => unreachable!("TNT inner cache must be Titans or Delta"),
                }

                // CS-39 straight-through: zero d_alpha where alpha was clamped.
                let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
                let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
                if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                    if let GpuMemoryCache::Titans { alpha, .. } | GpuMemoryCache::Delta { alpha, .. } = inner_cache {
                        unsafe {
                            crate::cuda_ffi::theta_clamp_mask_cuda(
                                alpha.as_ptr(), d_alpha_shard.ptr(), shard_tokens as i32, alpha_floor, alpha_ceil,
                            );
                        }
                    }
                }

                // CS-39 straight-through: zero d_theta where theta was clamped
                let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
                let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
                if theta_floor > 0.0 || theta_ceil < f32::MAX {
                    if let GpuMemoryCache::Titans { theta, .. } | GpuMemoryCache::Delta { theta, .. } = inner_cache {
                        unsafe {
                            crate::cuda_ffi::theta_clamp_mask_cuda(
                                theta.as_ptr(), d_theta_shard.ptr(), shard_tokens as i32, theta_floor, theta_ceil,
                            );
                        }
                    }
                }

                // Step 5b: L2 normalization backward for k/q per shard.
                {
                    let mut d_k_raw = GpuBuf::zeros(shard_tokens * d);
                    let mut d_q_raw = GpuBuf::zeros(shard_tokens * d);
                    unsafe {
                        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
                            d_k_shard.as_ptr(), shard_k_mem.as_ptr(), shard_k_norms.as_ptr(),
                            d_k_raw.ptr(), shard_tokens as i32, d as i32, 1e-8,
                        );
                        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
                            d_q_shard.as_ptr(), shard_q_mem.as_ptr(), shard_q_norms.as_ptr(),
                            d_q_raw.ptr(), shard_tokens as i32, d as i32, 1e-8,
                        );
                    }
                    d_k_shard = d_k_raw;
                    d_q_shard = d_q_raw;
                }

                // Step 6: Accumulate unpadded shard gradients into full-sequence totals.
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_k_shard.as_ptr(), d_k_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_v_shard.as_ptr(), d_v_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_q_shard.as_ptr(), d_q_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                }

                // Step 7: Gate backward per shard → temp buffers → accumulate into level_grads.
                {
                    let mut tmp_dw_alpha = GpuBuf::zeros(2 * d);
                    let mut tmp_db_alpha = GpuBuf::zeros(1);
                    let mut tmp_dw_theta = GpuBuf::zeros(2 * d);
                    let mut tmp_db_theta = GpuBuf::zeros(1);
                    let mut tmp_dw_eta   = GpuBuf::zeros(2 * d);
                    let mut tmp_db_eta   = GpuBuf::zeros(1);

                    match inner_cache {
                        GpuMemoryCache::Titans { k_mem, v_mem, alpha, theta, eta, .. } => {
                            crate::dispatch::gate_backward_dd(
                                &d_alpha_shard, alpha,
                                Some(&d_theta_shard), Some(theta),
                                Some(&d_eta_shard), Some(eta),
                                k_mem, v_mem,
                                &mut tmp_dw_alpha, &mut tmp_db_alpha,
                                &mut tmp_dw_theta, &mut tmp_db_theta,
                                &mut tmp_dw_eta,   &mut tmp_db_eta,
                                shard_tokens, d,
                            );
                        }
                        GpuMemoryCache::Delta { k_mem, v_mem, alpha, theta, .. } => {
                            crate::dispatch::gate_backward_dd(
                                &d_alpha_shard, alpha,
                                Some(&d_theta_shard), Some(theta),
                                None, None,
                                k_mem, v_mem,
                                &mut tmp_dw_alpha, &mut tmp_db_alpha,
                                &mut tmp_dw_theta, &mut tmp_db_theta,
                                &mut tmp_dw_eta,   &mut tmp_db_eta,
                                shard_tokens, d,
                            );
                        }
                        _ => unreachable!(),
                    }

                    unsafe {
                        crate::cuda_ffi::saxpy_cuda(1.0, tmp_dw_alpha.as_ptr(), level_grads.d_w_alpha.ptr(), (2 * d) as i32);
                        crate::cuda_ffi::saxpy_cuda(1.0, tmp_db_alpha.as_ptr(), level_grads.d_b_alpha.ptr(), 1);
                        crate::cuda_ffi::saxpy_cuda(1.0, tmp_dw_theta.as_ptr(), level_grads.d_w_theta.ptr(), (2 * d) as i32);
                        crate::cuda_ffi::saxpy_cuda(1.0, tmp_db_theta.as_ptr(), level_grads.d_b_theta.ptr(), 1);
                        if has_eta {
                            crate::cuda_ffi::saxpy_cuda(1.0, tmp_dw_eta.as_ptr(), level_grads.d_w_eta.ptr(), (2 * d) as i32);
                            crate::cuda_ffi::saxpy_cuda(1.0, tmp_db_eta.as_ptr(), level_grads.d_b_eta.ptr(), 1);
                        }
                    }
                }

                // Step 8: Propagate d_m_carry backward
                d_m_carry = d_m_old;
            }

            // Projection weight grads: d_W[d,d] = d_proj^T[d, s] @ embedded[s, d]
            gpu_matmul_transa_dd(&d_k_mem_total, embedded, &mut level_grads.d_w_k_mem, d, s, d);
            gpu_matmul_transa_dd(&d_v_mem_total, embedded, &mut level_grads.d_w_v_mem, d, s, d);
            gpu_matmul_transa_dd(&d_q_mem_total, embedded, &mut level_grads.d_w_q_mem, d, s, d);

            // d_embedded from memory projections: d_emb = d_k @ W_k + d_v @ W_v + d_q @ W_q
            let mut d_embedded = GpuBuf::zeros(s * d);
            crate::dispatch::cublas_matmul_acc_dd(&d_k_mem_total, &level_params.w_k_mem, &mut d_embedded, s, d, d);
            crate::dispatch::cublas_matmul_acc_dd(&d_v_mem_total, &level_params.w_v_mem, &mut d_embedded, s, d, d);
            crate::dispatch::cublas_matmul_acc_dd(&d_q_mem_total, &level_params.w_q_mem, &mut d_embedded, s, d, d);

            d_embedded
        }
        // ── SwiGLU: stateless MLP, direct weight grads ───────────────
        GpuMemoryCache::SwiGlu { gate_buf, up_buf, fused_buf, cache_buf } => {
            let inter = cfg.intermediate_size;
            let mut d_x = GpuBuf::zeros(batch_size * s * d);
            unsafe {
                crate::cuda_ffi::swiglu_backward_f32_cuda_dd(
                    d_y.as_ptr(),
                    embedded.as_ptr(),
                    level_params.gate_proj.as_ptr(),
                    level_params.up_proj.as_ptr(),
                    level_params.down_proj.as_ptr(),
                    fused_buf.as_ptr(),
                    gate_buf.as_ptr(),
                    up_buf.as_ptr(),
                    cache_buf.as_ptr(),
                    d_x.ptr(),
                    level_grads.d_gate_proj.ptr(),
                    level_grads.d_up_proj.ptr(),
                    level_grads.d_down_proj.ptr(),
                    (batch_size * s) as i32, d as i32, inter as i32,
                );
            }
            d_x  // d_embedded contribution (accumulated into d_embedded_mem by caller)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Checkpointed backward: segment-based replay + backward
// ══════════════════════════════════════════════════════════════════════

/// Build a list of segment boundaries from checkpoint_interval and seq_len.
/// Returns Vec<(t_start, t_end, ckpt_idx)> in forward order.
/// ckpt_idx is the index into m_checkpoints for the segment's initial state.
#[cfg(feature = "cuda")]
fn segment_boundaries(seq_len: usize, c: usize) -> Vec<(usize, usize, usize)> {
    assert!(c > 0, "checkpoint_interval must be > 0");
    let mut segments = Vec::new();
    let mut t = 0;
    let mut ckpt_idx = 0;
    while t < seq_len {
        let t_end = (t + c).min(seq_len);
        segments.push((t, t_end, ckpt_idx));
        ckpt_idx += 1;
        t = t_end;
    }
    segments
}

/// Delta Rule checkpointed backward.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn delta_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize, error_clip: f32,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let sd = s * d;
    let segments = segment_boundaries(s, c);

    // Accumulation buffers (full sequence)
    let mut d_k_mem = GpuBuf::zeros(sd);
    let mut d_v_mem = GpuBuf::zeros(sd);
    let mut d_q_mem = GpuBuf::zeros(sd);
    let mut d_alpha = GpuBuf::zeros(s);
    let mut d_theta = GpuBuf::zeros(s);

    // d_M seed: starts as zeros for the last segment
    let mut d_m_seed = GpuBuf::zeros(dd);

    // Pre-allocate scratch buffers sized for max segment (CS-42: no per-segment cudaMalloc)
    let max_seg = c.min(s);
    let mut local_m_states = GpuBuf::zeros((max_seg + 1) * dd);
    let mut local_y = GpuBuf::zeros(max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(dd);

    // Process segments in reverse
    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // 1. Replay forward from checkpoint to reconstruct local m_states
        let ckpt_m = m_checkpoints.slice(ckpt_idx * dd, dd);
        local_m_states.zero();
        local_y.zero();

        // Replay forward: pass offset pointers into full-sequence buffers.
        // Works because the forward kernel is purely sequential.
        unsafe {
            crate::cuda_ffi::delta_forward_f32_cuda(
                (k_mem.as_ptr()).add(t_start * d),
                (v_mem.as_ptr()).add(t_start * d),
                (q_mem.as_ptr()).add(t_start * d),
                (alpha.as_ptr()).add(t_start),
                (theta.as_ptr()).add(t_start),
                ckpt_m.as_ptr(),
                local_m_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32, 1, // batch_size=1 (checkpointed paths are always bs=1)
                error_clip,
            );
        }
        crate::dispatch::cuda_sync();

        // 2. Segment backward with d_m_seed
        seg_d_m_out.zero();
        crate::dispatch::delta_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha, theta,
            &local_m_states, d_y,
            &d_m_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut d_theta, &mut seg_d_m_out,
            t_start, t_end, d, error_clip,
        );
        crate::dispatch::cuda_sync();

        // 3. Propagate d_M seed to earlier segment
        d_m_seed.copy_from_device(&seg_d_m_out);
    }

    (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta)
}

/// Titans checkpointed backward.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn titans_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, s_checkpoints: &GpuBuf<f32>,
    d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize, error_clip: f32,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let sd = s * d;
    let segments = segment_boundaries(s, c);

    let mut d_k_mem = GpuBuf::zeros(sd);
    let mut d_v_mem = GpuBuf::zeros(sd);
    let mut d_q_mem = GpuBuf::zeros(sd);
    let mut d_alpha = GpuBuf::zeros(s);
    let mut d_theta = GpuBuf::zeros(s);
    let mut d_eta = GpuBuf::zeros(s);
    let mut d_m_seed = GpuBuf::zeros(dd);
    let mut d_s_seed = GpuBuf::zeros(dd);

    // Pre-allocate scratch buffers sized for max segment (CS-42: no per-segment cudaMalloc)
    let max_seg = c.min(s);
    let mut local_m_states = GpuBuf::zeros((max_seg + 1) * dd);
    let mut local_s_states = GpuBuf::zeros((max_seg + 1) * dd);
    let mut local_y = GpuBuf::zeros(max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(dd);
    let mut seg_d_s_out = GpuBuf::zeros(dd);

    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // Replay forward
        let ckpt_m = m_checkpoints.slice(ckpt_idx * dd, dd);
        let ckpt_s = s_checkpoints.slice(ckpt_idx * dd, dd);
        local_m_states.zero();
        local_s_states.zero();
        local_y.zero();

        unsafe {
            crate::cuda_ffi::titans_forward_f32_cuda(
                (k_mem.as_ptr()).add(t_start * d),
                (v_mem.as_ptr()).add(t_start * d),
                (q_mem.as_ptr()).add(t_start * d),
                (alpha.as_ptr()).add(t_start),
                (theta.as_ptr()).add(t_start),
                (eta.as_ptr()).add(t_start),
                ckpt_m.as_ptr(),
                ckpt_s.as_ptr(),
                local_m_states.ptr(),
                local_s_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32, 1, // batch_size=1 (checkpointed paths are always bs=1)
                error_clip,
            );
        }
        crate::dispatch::cuda_sync();

        // Segment backward
        seg_d_m_out.zero();
        seg_d_s_out.zero();
        crate::dispatch::titans_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha, theta, eta,
            &local_m_states, &local_s_states, d_y,
            &d_m_seed, &d_s_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut d_theta, &mut d_eta,
            &mut seg_d_m_out, &mut seg_d_s_out,
            t_start, t_end, d, error_clip,
        );
        crate::dispatch::cuda_sync();

        d_m_seed.copy_from_device(&seg_d_m_out);
        d_s_seed.copy_from_device(&seg_d_s_out);
    }

    (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta)
}

/// Hebbian checkpointed backward.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn hebbian_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let sd = s * d;
    let segments = segment_boundaries(s, c);

    let mut d_k_mem = GpuBuf::zeros(sd);
    let mut d_v_mem = GpuBuf::zeros(sd);
    let mut d_q_mem = GpuBuf::zeros(sd);
    let mut d_alpha = GpuBuf::zeros(s);
    let mut d_m_seed = GpuBuf::zeros(dd);

    // Pre-allocate scratch buffers sized for max segment (CS-42: no per-segment cudaMalloc)
    let max_seg = c.min(s);
    let mut local_m_states = GpuBuf::zeros((max_seg + 1) * dd);
    let mut local_y = GpuBuf::zeros(max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(dd);

    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // Replay forward
        let ckpt_m = m_checkpoints.slice(ckpt_idx * dd, dd);
        local_m_states.zero();
        local_y.zero();

        unsafe {
            crate::cuda_ffi::hebbian_forward_f32_cuda(
                (k_mem.as_ptr()).add(t_start * d),
                (v_mem.as_ptr()).add(t_start * d),
                (q_mem.as_ptr()).add(t_start * d),
                (alpha.as_ptr()).add(t_start),
                ckpt_m.as_ptr(),
                local_m_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32,
            );
        }
        crate::dispatch::cuda_sync();

        // Segment backward
        seg_d_m_out.zero();
        crate::dispatch::hebbian_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha,
            &local_m_states, d_y,
            &d_m_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut seg_d_m_out,
            t_start, t_end, d,
        );
        crate::dispatch::cuda_sync();

        d_m_seed.copy_from_device(&seg_d_m_out);
    }

    (d_k_mem, d_v_mem, d_q_mem, d_alpha)
}

/// Accumulate projection weight gradients from inner loop gradients.
/// Returns d_embedded contribution.
///
/// This mirrors the Rust code in step_backward() across all rules:
///   d_w_k_mem = d_k_mem^T @ embedded
///   d_w_v_mem = d_v_mem^T @ embedded
///   d_w_q_mem = d_q_mem^T @ embedded
///   d_embedded += d_k_mem @ w_k_mem + d_v_mem @ w_v_mem + d_q_mem @ w_q_mem
///   d_w_alpha/d_b_alpha, d_w_theta/d_b_theta, d_w_eta/d_b_eta from gate_backward_dd
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn accumulate_projection_grads(
    level_params: &GpuMemoryLevelParams,
    embedded: &GpuBuf<f32>,
    k_mem: &GpuBuf<f32>,               // gate inputs (forward cache, normalized)
    v_mem: &GpuBuf<f32>,
    q_mem: &GpuBuf<f32>,               // normalized q projection (for L2 backward)
    alpha: &GpuBuf<f32>,               // gate outputs (forward cache)
    theta: Option<&GpuBuf<f32>>,       // softplus output; None for Hebbian
    eta: Option<&GpuBuf<f32>>,         // sigmoid output; None for Delta/Hebbian/DGD
    d_k_mem: &GpuBuf<f32>,
    d_v_mem: &GpuBuf<f32>,
    d_q_mem: &GpuBuf<f32>,
    d_alpha: &GpuBuf<f32>,             // per-token upstream gate gradients
    d_theta: Option<&GpuBuf<f32>>,
    d_eta: Option<&GpuBuf<f32>>,
    k_norms: &GpuBuf<f32>,            // L2 norms from forward normalization
    q_norms: &GpuBuf<f32>,
    level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
    batch_size: usize,
) -> GpuBuf<f32> {
    let bs_s = batch_size * s;
    let bsd = bs_s * d;
    let d_i32 = d as i32;
    let n_rows = bs_s as i32;

    // L2 normalization backward: d_k_mem and d_q_mem are gradients w.r.t. normalized k/q.
    // Transform to gradients w.r.t. raw (pre-normalization) projections.
    let mut d_k_raw = GpuBuf::zeros(bsd);
    let mut d_q_raw = GpuBuf::zeros(bsd);
    unsafe {
        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
            d_k_mem.as_ptr(), k_mem.as_ptr(), k_norms.as_ptr(),
            d_k_raw.ptr(), n_rows, d_i32, 1e-8,
        );
        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
            d_q_mem.as_ptr(), q_mem.as_ptr(), q_norms.as_ptr(),
            d_q_raw.ptr(), n_rows, d_i32, 1e-8,
        );
    }

    // Projection weight grads: d_W[d,d] = d_proj^T[d, bs*s] @ embedded[bs*s, d]
    gpu_matmul_transa_dd(&d_k_raw, embedded, &mut level_grads.d_w_k_mem, d, bs_s, d);
    gpu_matmul_transa_dd(d_v_mem, embedded, &mut level_grads.d_w_v_mem, d, bs_s, d);
    gpu_matmul_transa_dd(&d_q_raw, embedded, &mut level_grads.d_w_q_mem, d, bs_s, d);

    // d_embedded from memory projections
    let mut d_embedded = GpuBuf::zeros(bsd);
    crate::dispatch::cublas_matmul_acc_dd(&d_k_raw, &level_params.w_k_mem, &mut d_embedded, bs_s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(d_v_mem, &level_params.w_v_mem, &mut d_embedded, bs_s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(&d_q_raw, &level_params.w_q_mem, &mut d_embedded, bs_s, d, d);

    // Gate backward: accumulate d_w_alpha/theta/eta and d_b_alpha/theta/eta.
    // sigmoid(logit_theta) recovered from softplus output as 1 - exp(-theta).
    // No forward changes required — cached gate outputs suffice.
    crate::dispatch::gate_backward_dd(
        d_alpha, alpha,
        d_theta, theta,
        d_eta,   eta,
        k_mem, v_mem,
        &mut level_grads.d_w_alpha, &mut level_grads.d_b_alpha,
        &mut level_grads.d_w_theta, &mut level_grads.d_b_theta,
        &mut level_grads.d_w_eta,   &mut level_grads.d_b_eta,
        bs_s, d,
    );

    d_embedded
}

/// Frozen level read-only backward (simplified).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_read_only_backward(
    level_params: &GpuMemoryLevelParams,
    _y_level: &GpuBuf<f32>,
    _d_y: &GpuBuf<f32>,
    _embedded: &GpuBuf<f32>,
    _level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
    batch_size: usize,
) -> GpuBuf<f32> {
    // Frozen level: y = q_mem @ M^T where M is frozen context.
    // d_q_mem = d_y @ M → needs M, which we don't store in cache.
    // d_embedded = d_q_mem @ W_q_mem
    //
    // For the first iteration of GPU-resident model, return zeros.
    // Frozen level gradients go to error buffers (not direct weight updates),
    // so this only affects d_embedded propagation from frozen levels.
    // The dominant gradient signal comes from active levels.
    let _ = level_params;
    GpuBuf::zeros(batch_size * s * d)
}

// ══════════════════════════════════════════════════════════════════════
// Weight update (cublasSaxpy: W -= lr * grad)
// ══════════════════════════════════════════════════════════════════════

/// In-place weight update on GPU: W -= lr * grad for all parameters.
/// Uses cublasSaxpy (y += alpha*x with alpha = -lr).
#[cfg(feature = "cuda")]
pub fn gpu_weight_update(params: &mut GpuMAGParams, grads: &GpuMAGGrads, lr: f32) {
    let neg_lr = -lr;

    // SWA weights
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_embed, &mut params.swa.w_embed, neg_lr);
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_q, &mut params.swa.w_q, neg_lr);
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_k, &mut params.swa.w_k, neg_lr);
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_v, &mut params.swa.w_v, neg_lr);
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_o, &mut params.swa.w_o, neg_lr);
    crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &grads.d_w_unembed, &mut params.swa.w_unembed, neg_lr);

    // Per-level memory weights
    for (level, lg) in grads.levels.iter().enumerate() {
        let lp = &mut params.levels[level];
        crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &lg.d_w_k_mem, &mut lp.w_k_mem, neg_lr);
        crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &lg.d_w_v_mem, &mut lp.w_v_mem, neg_lr);
        crate::gpu_buf::gpu_saxpy(crate::dispatch::cublas_handle_pub(), &lg.d_w_q_mem, &mut lp.w_q_mem, neg_lr);
        // Gate weights: alpha, theta, eta biases
        // Use saxpy_cuda for small buffers (cuBLAS overhead not worth it for 1-element)
        unsafe {
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_w_alpha.as_ptr(), lp.w_alpha.ptr(), lp.w_alpha.len() as i32);
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_b_alpha.as_ptr(), lp.b_alpha.ptr(), 1);
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_w_theta.as_ptr(), lp.w_theta.ptr(), lp.w_theta.len() as i32);
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_b_theta.as_ptr(), lp.b_theta.ptr(), 1);
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_w_eta.as_ptr(), lp.w_eta.ptr(), lp.w_eta.len() as i32);
            crate::cuda_ffi::saxpy_cuda(neg_lr, lg.d_b_eta.as_ptr(), lp.b_eta.ptr(), 1);
        }
    }

    crate::dispatch::cuda_sync();
}

/// Weight tying: copy w_unembed^T → w_embed on GPU.
/// This ensures the embedding table tracks the unembedding after each weight update,
/// compensating for the vanishing gradient through the deep chain.
#[cfg(feature = "cuda")]
pub fn gpu_sync_embed_weights(params: &mut GpuMAGParams, d: usize, vocab: usize) {
    unsafe {
        crate::cuda_ffi::transpose_copy_cuda(
            params.swa.w_unembed.as_ptr(),  // src: [d, vocab]
            params.swa.w_embed.ptr(),        // dst: [vocab, d]
            d as i32,
            vocab as i32,
        );
    }
    crate::dispatch::cuda_sync();
}

// ══════════════════════════════════════════════════════════════════════
// Helper: matmul with transposed A (C = A^T @ B)
// ══════════════════════════════════════════════════════════════════════

/// C[m,n] = A^T[m,k] @ B[k,n] where A is stored as [k,m].
/// Used for gradient weight computation (d_W = d_out^T @ input).
///
/// Row-major trick: sgemm(N, T, n, m, k, alpha, B, n, A, m, beta, C, n)
#[cfg(feature = "cuda")]
pub(crate) fn gpu_matmul_transa_dd(
    a: &GpuBuf<f32>,   // [k, m] stored row-major
    b: &GpuBuf<f32>,   // [k, n] stored row-major
    c: &mut GpuBuf<f32>, // [m, n] output
    m: usize, k: usize, n: usize,
) {
    extern "C" {
        fn cublasSgemm_v2(
            handle: *mut std::ffi::c_void,
            transa: i32, transb: i32,
            m: i32, n: i32, k: i32,
            alpha: *const f32,
            a: *const f32, lda: i32,
            b: *const f32, ldb: i32,
            beta: *const f32,
            c: *mut f32, ldc: i32,
        ) -> i32;
    }

    let alpha_val: f32 = 1.0;
    let beta_val: f32 = 0.0;

    // We want C = A^T @ B in row-major.
    // Column-major equivalent: C^T = B^T @ A
    // sgemm computes col-major: C_col = alpha * op(A_col) @ op(B_col) + beta * C_col
    // With the row-major→col-major trick:
    //   C^T[n,m] = B^T[n,k] @ A[k,m]
    //   sgemm(N, T, n=n, m_=m, k_=k, alpha, B_ptr, lda=n, A_ptr, ldb=m, beta, C_ptr, ldc=n)
    // Wait, let me think again. We have:
    //   A is [k, m] row-major → column-major is A_col with shape [m, k]
    //   B is [k, n] row-major → column-major is B_col with shape [n, k]
    //   C is [m, n] row-major → column-major is C_col with shape [n, m]
    //
    //   We want C = A^T @ B. In row-major:
    //     C[m,n] = A^T[m,k] @ B[k,n]
    //
    //   In column-major, this is:
    //     C_col[n,m] = B_col[n,k] @ A_col^T[k,m]
    //     sgemm(T, N, n, m, k, alpha, A_col, m, B_col, n, beta, C_col, n)
    //
    //   But A_col (column-major view of row-major A) has A_col[i,j] = A[j,i],
    //   and the physical layout is A_ptr with leading dimension m.
    //   Similarly B_col has leading dimension n.
    //
    //   So: sgemm(N, T, n, m, k, alpha, B_ptr, n, A_ptr, m, beta, C_ptr, n)

    let rc = unsafe {
        cublasSgemm_v2(
            crate::dispatch::cublas_handle_pub(),
            0, 1, // CUBLAS_OP_N, CUBLAS_OP_T
            n as i32, m as i32, k as i32,
            &alpha_val,
            b.as_ptr(), n as i32,
            a.as_ptr(), m as i32,
            &beta_val,
            c.ptr(), n as i32,
        )
    };
    assert_eq!(rc, 0, "cublasSgemm_v2 (transa_dd) failed: error code {rc}");
}
