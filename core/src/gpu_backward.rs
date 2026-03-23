/// GPU-resident CMS backward pass and weight update.
///
/// Mirrors `cms_backward` in mag.rs but operates entirely on device pointers.
/// Produces `GpuMAGGrads` (gradient buffers on GPU), consumed by `gpu_weight_update`.
///
/// Supports matrix-based rules: DeltaRule, TitansLMM, HebbianRule, DGD.
/// Supports MLP-based rules: Moneta, YAAD.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuMAGParams, GpuMemoryLevelParams};
#[cfg(feature = "cuda")]
use crate::gpu_forward::{GpuCMSCache, GpuMemoryCache};
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
    let d_logits = GpuBuf::zeros(bsv);
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
    let d_attn_out;
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
        let d_residual_after_attn = GpuBuf::zeros(bsd);
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, d_gated_out.as_ptr(), d_residual_after_attn.ptr(), bsd as i32);
        }
        // We'll use d_residual_after_attn later as d_attn_out (it splits into d_embedded + d_attn_out)
        d_attn_out = d_residual_after_attn;
    } else {
        // Legacy: gated_out = attn_out * gate → gating_backward + sigmoid_backward
        d_attn_out = GpuBuf::zeros(bsd);
        let d_gate = GpuBuf::zeros(bsd);
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
        let scratch = GpuBuf::zeros(max_blocks);
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
    let d_mem_input = GpuBuf::zeros(bsd);

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
        let d_residual_from_mem = GpuBuf::zeros(bsd);
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
    let d_embedded = GpuBuf::zeros(bsd);
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
    let _dd = d * d;
    let _bsd = batch_size * s * d;
    let bs_s = batch_size * s;

    // Spec 45: per-head memory dimensions
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let dd_mem = hd * hd;
    let bs_mem = batch_size * nh;
    let bs_mem_s = bs_mem * s;
    let bs_mem_d = bs_mem * s * hd;

    match mem_cache {
        GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms, proxy, .. } => {
            // Spec 45: per-head backward
            // Transpose cached d_model k/v/q and d_y to per-head layout
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, batch_size, s, nh);

            let mut d_k_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_v_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_q_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_alpha_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_theta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_m_initial = GpuBuf::zeros(dd_mem);

            // m_states are already per-head layout from forward
            let m_for_bw = if *proxy {
                let m_bcast = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
                unsafe {
                    let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                        m_bcast.ptr(), m_states.as_ptr(),
                        i32::try_from(dd_mem).expect("dd_mem overflows i32"),
                        i32::try_from(s + 1).expect("s+1 overflows i32"),
                        i32::try_from(bs_mem).expect("bs_mem overflows i32"),
                    );
                    assert_eq!(rc, 0, "broadcast_fill_f32_cuda failed (rc={rc})");
                }
                m_bcast
            } else {
                m_states.dup()
            };

            crate::dispatch::delta_backward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                &m_for_bw, &d_y_ph,
                &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                &mut d_alpha_ph, &mut d_theta_ph, &mut d_m_initial,
                s, hd, bs_mem,
                cfg.error_clip_for_level(level),
            );

            // Sum per-head gate grads → d_model resolution
            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, batch_size, s, nh);

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

            // Transpose d_k/d_v/d_q back to d_model for projection grads
            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, k_norms, q_norms, proxy, .. } => {
            // Spec 45: per-head backward
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, batch_size, s, nh);
            let eta_ph = crate::gpu_forward::broadcast_gates(eta, batch_size, s, nh);

            let mut d_k_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_v_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_q_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_alpha_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_theta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_eta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_m_initial = GpuBuf::zeros(dd_mem);
            let mut d_s_initial = GpuBuf::zeros(dd_mem);

            // m_states/s_states are already per-head layout from forward
            let (m_for_bw, s_for_bw) = if *proxy {
                let m_bcast = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
                let s_bcast = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
                unsafe {
                    let dd_i32 = i32::try_from(dd_mem).expect("dd_mem overflows i32");
                    let slots_i32 = i32::try_from(s + 1).expect("s+1 overflows i32");
                    let bs_i32 = i32::try_from(bs_mem).expect("bs_mem overflows i32");
                    let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                        m_bcast.ptr(), m_states.as_ptr(),
                        dd_i32, slots_i32, bs_i32,
                    );
                    assert_eq!(rc, 0, "broadcast_fill M failed (rc={rc})");
                    let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                        s_bcast.ptr(), s_states.as_ptr(),
                        dd_i32, slots_i32, bs_i32,
                    );
                    assert_eq!(rc, 0, "broadcast_fill S failed (rc={rc})");
                }
                (m_bcast, s_bcast)
            } else {
                (m_states.dup(), s_states.dup())
            };

            crate::dispatch::titans_backward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                &m_for_bw, &s_for_bw, &d_y_ph,
                &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                &mut d_alpha_ph, &mut d_theta_ph, &mut d_eta_ph,
                &mut d_m_initial, &mut d_s_initial,
                s, hd, bs_mem,
                cfg.error_clip_for_level(level),
            );

            // Sum per-head gate grads → d_model resolution
            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, batch_size, s, nh);
            let d_eta = crate::gpu_forward::sum_gates_across_heads(&d_eta_ph, batch_size, s, nh);

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

            // Transpose d_k/d_v/d_q back to d_model for projection grads
            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), Some(eta),
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), Some(&d_eta),
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        // ── Chunkwise variants (spec 43 — frozen-M₀ backward) ──────────
        GpuMemoryCache::DeltaChunkwise { k_mem, v_mem, q_mem, alpha, theta, m_chunk_states, k_norms, q_norms, chunk_size, .. } => {
            // Spec 45: per-head backward for chunkwise Delta
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, batch_size, s, nh);

            let mut d_k_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_v_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_q_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_alpha_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_theta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_m_initial = GpuBuf::zeros(dd_mem);

            if *chunk_size > 1 {
                crate::dispatch::delta_chunkwise_backward_batched_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                    m_chunk_states, &d_y_ph,
                    &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                    &mut d_alpha_ph, &mut d_theta_ph, &mut d_m_initial,
                    s, hd, bs_mem, *chunk_size,
                    cfg.error_clip_for_level(level),
                );
            } else {
                crate::dispatch::delta_chunkwise_backward_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                    m_chunk_states, &d_y_ph,
                    &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                    &mut d_alpha_ph, &mut d_theta_ph, &mut d_m_initial,
                    s, hd, bs_mem, *chunk_size,
                    cfg.error_clip_for_level(level),
                );
            }

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, batch_size, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, batch_size,
            )
        }
        GpuMemoryCache::TitansChunkwise { k_mem, v_mem, q_mem, alpha, theta, eta, m_chunk_states, s_chunk_states, k_norms, q_norms, chunk_size, .. } => {
            // Spec 45: per-head backward for chunkwise Titans
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, batch_size, s, nh);
            let eta_ph = crate::gpu_forward::broadcast_gates(eta, batch_size, s, nh);

            let mut d_k_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_v_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_q_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_alpha_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_theta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_eta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_m_initial = GpuBuf::zeros(dd_mem);
            let mut d_s_initial = GpuBuf::zeros(dd_mem);

            if *chunk_size > 1 {
                crate::dispatch::titans_chunkwise_backward_batched_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                    m_chunk_states, s_chunk_states, &d_y_ph,
                    &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                    &mut d_alpha_ph, &mut d_theta_ph, &mut d_eta_ph,
                    &mut d_m_initial, &mut d_s_initial,
                    s, hd, bs_mem, *chunk_size,
                    cfg.error_clip_for_level(level),
                );
            } else {
                crate::dispatch::titans_chunkwise_backward_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                    m_chunk_states, s_chunk_states, &d_y_ph,
                    &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                    &mut d_alpha_ph, &mut d_theta_ph, &mut d_eta_ph,
                    &mut d_m_initial, &mut d_s_initial,
                    s, hd, bs_mem, *chunk_size,
                    cfg.error_clip_for_level(level),
                );
            }

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, batch_size, s, nh);
            let d_eta = crate::gpu_forward::sum_gates_across_heads(&d_eta_ph, batch_size, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

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
            assert_eq!(batch_size, 1, "Hebbian batch_size > 1 not yet supported");
            // Spec 45: per-head backward — loop over nh heads (hebbian is single-batch)
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);

            let d_k_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            let d_v_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            let d_q_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            let d_alpha_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s);
            let d_m_initial: GpuBuf<f32> = GpuBuf::zeros(dd_mem);

            for h in 0..nh {
                unsafe {
                    crate::cuda_ffi::hebbian_backward_f32_cuda(
                        k_mem_ph.as_ptr().add(h * s * hd),
                        v_mem_ph.as_ptr().add(h * s * hd),
                        q_mem_ph.as_ptr().add(h * s * hd),
                        alpha_ph.as_ptr().add(h * s),
                        m_states.as_ptr().add(h * (s + 1) * dd_mem),
                        d_y_ph.as_ptr().add(h * s * hd),
                        d_k_ph.ptr().add(h * s * hd),
                        d_v_ph.ptr().add(h * s * hd),
                        d_q_ph.ptr().add(h * s * hd),
                        d_alpha_ph.ptr().add(h * s),
                        d_m_initial.ptr(),  // accumulate across heads (overwritten, not summed)
                        s as i32, hd as i32,
                    );
                }
            }

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

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
        // Spec 45: per-head — loop over nh heads, call single-batch helper per head
        GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval, k_norms, q_norms } => {
            let c = *checkpoint_interval;
            // Reshape to per-head layout: [nh, s, hd]
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, 1, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, 1, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, 1, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, 1, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, 1, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, 1, s, nh);

            // Single batched call — all heads processed together
            let (d_k_ph, d_v_ph, d_q_ph, d_alpha_ph, d_theta_ph) = delta_backward_checkpointed(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                m_checkpoints, &d_y_ph,
                s, hd, c, nh, cfg.error_clip_for_level(level),
            );

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, 1, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, 1, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, 1, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, 1, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, 1, s, nh, hd);

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
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, 1, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, 1, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, 1, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, 1, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, 1, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, 1, s, nh);
            let eta_ph = crate::gpu_forward::broadcast_gates(eta, 1, s, nh);

            // Single batched call — all heads processed together
            let (d_k_ph, d_v_ph, d_q_ph, d_alpha_ph, d_theta_ph, d_eta_ph) = titans_backward_checkpointed(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                m_checkpoints, s_checkpoints, &d_y_ph,
                s, hd, c, nh, cfg.error_clip_for_level(level),
            );

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, 1, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, 1, s, nh);
            let d_eta = crate::gpu_forward::sum_gates_across_heads(&d_eta_ph, 1, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, 1, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, 1, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, 1, s, nh, hd);

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
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, 1, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, 1, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, 1, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, 1, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, 1, s, nh);

            // Single batched call — all heads processed together
            let (d_k_ph, d_v_ph, d_q_ph, d_alpha_ph) = hebbian_backward_checkpointed(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph,
                m_checkpoints, &d_y_ph,
                s, hd, c, nh,
            );

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, 1, s, nh);
            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, 1, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, 1, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, 1, s, nh, hd);

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
            // Spec 45: per-head backward (DGD uses delta backward kernels)
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, batch_size, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, batch_size, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, batch_size, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, batch_size, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, batch_size, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, batch_size, s, nh);

            let mut d_k_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_v_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_q_ph = GpuBuf::zeros(bs_mem_d);
            let mut d_alpha_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_theta_ph = GpuBuf::zeros(bs_mem_s);
            let mut d_m_initial = GpuBuf::zeros(dd_mem);

            crate::dispatch::delta_backward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                m_states, &d_y_ph,
                &mut d_k_ph, &mut d_v_ph, &mut d_q_ph,
                &mut d_alpha_ph, &mut d_theta_ph, &mut d_m_initial,
                s, hd, bs_mem,
                cfg.error_clip_for_level(level),
            );

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, batch_size, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, batch_size, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), bs_s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), bs_s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, batch_size, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, batch_size, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, batch_size, s, nh, hd);

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
            let d_y_ph = crate::gpu_forward::reshape_to_per_head(d_y, 1, s, nh, hd);
            let k_mem_ph = crate::gpu_forward::reshape_to_per_head(k_mem, 1, s, nh, hd);
            let v_mem_ph = crate::gpu_forward::reshape_to_per_head(v_mem, 1, s, nh, hd);
            let q_mem_ph = crate::gpu_forward::reshape_to_per_head(q_mem, 1, s, nh, hd);
            let alpha_ph = crate::gpu_forward::broadcast_gates(alpha, 1, s, nh);
            let theta_ph = crate::gpu_forward::broadcast_gates(theta, 1, s, nh);

            // Single batched call — DGD reuses delta backward kernels
            let (d_k_ph, d_v_ph, d_q_ph, d_alpha_ph, d_theta_ph) = delta_backward_checkpointed(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                m_checkpoints, &d_y_ph,
                s, hd, c, nh, cfg.error_clip_for_level(level),
            );

            let d_alpha = crate::gpu_forward::sum_gates_across_heads(&d_alpha_ph, 1, s, nh);
            let d_theta = crate::gpu_forward::sum_gates_across_heads(&d_theta_ph, 1, s, nh);

            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        alpha.as_ptr(), d_alpha.ptr(), s as i32, alpha_floor, alpha_ceil,
                    );
                }
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                unsafe {
                    crate::cuda_ffi::theta_clamp_mask_cuda(
                        theta.as_ptr(), d_theta.ptr(), s as i32, theta_floor, theta_ceil,
                    );
                }
            }

            let d_k_mem = crate::gpu_forward::reshape_from_per_head(&d_k_ph, 1, s, nh, hd);
            let d_v_mem = crate::gpu_forward::reshape_from_per_head(&d_v_ph, 1, s, nh, hd);
            let d_q_mem = crate::gpu_forward::reshape_from_per_head(&d_q_ph, 1, s, nh, hd);

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
        // Spec 51: per-head memory support — inner kernels use hd/kernel_batch,
        // then reshape back to d-space for L2 backward, gate backward, and projection grads.
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
            assert!(n_total > 0, "TNT backward: total_shards must be > 0");
            assert!(first_ret < n_total, "TNT backward: first_retained_shard ({first_ret}) >= total_shards ({n_total})");
            assert_eq!(n_retained, n_total - first_ret, "TNT backward: cache count ({n_retained}) != total_shards - first_retained ({} - {first_ret})", n_total);
            assert_eq!(k_summaries.len(), n_total, "TNT backward: k_summaries.len() ({}) != total_shards ({n_total})", k_summaries.len());
            assert_eq!(v_summaries.len(), n_total, "TNT backward: v_summaries.len() ({}) != total_shards ({n_total})", v_summaries.len());

            let _hd_i32 = i32::try_from(hd).expect("head_dim exceeds i32::MAX");

            // Accumulators for projection weight grads across all shards (d-space)
            let d_k_mem_total = GpuBuf::<f32>::zeros(s * d);
            let d_v_mem_total = GpuBuf::<f32>::zeros(s * d);
            let d_q_mem_total = GpuBuf::<f32>::zeros(s * d);

            // Spec 51: d_m_carry is per-head [nh * dd_mem]
            let mut d_m_carry = GpuBuf::<f32>::zeros(bs_mem * dd_mem);

            for shard_idx in (0..n_total).rev() {
                let shard_start = shard_idx * cg;
                let shard_end = (shard_start + cg).min(s);
                let shard_len = shard_end - shard_start;
                let n_batch = (shard_len + cl - 1) / cl;
                let kernel_batch = n_batch * bs_mem;

                // Step 1: Per-head global M update backward
                // k_summaries/v_summaries are [d] = [nh*hd] (per-head concatenated)
                let d_m_old = GpuBuf::<f32>::zeros(bs_mem * dd_mem);
                let d_k_sum = GpuBuf::<f32>::zeros(d);  // [nh*hd]
                let d_v_sum = GpuBuf::<f32>::zeros(d);
                for h in 0..bs_mem {
                    let mut d_m_old_h = GpuBuf::<f32>::zeros(dd_mem);
                    let mut d_k_sum_h = GpuBuf::<f32>::zeros(hd);
                    let mut d_v_sum_h = GpuBuf::<f32>::zeros(hd);
                    // Extract head h's carry gradient and summaries
                    let d_m_carry_h = GpuBuf::<f32>::zeros(dd_mem);
                    let k_sum_h = GpuBuf::<f32>::zeros(hd);
                    let v_sum_h = GpuBuf::<f32>::zeros(hd);
                    unsafe {
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            d_m_carry_h.ptr() as *mut _, (d_m_carry.as_ptr() as *const u8).add(h * dd_mem * 4) as *const _, dd_mem * 4);
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            k_sum_h.ptr() as *mut _, (k_summaries[shard_idx].as_ptr() as *const u8).add(h * hd * 4) as *const _, hd * 4);
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            v_sum_h.ptr() as *mut _, (v_summaries[shard_idx].as_ptr() as *const u8).add(h * hd * 4) as *const _, hd * 4);
                    }
                    crate::dispatch::tnt_global_update_backward_dd(
                        &d_m_carry_h, &k_sum_h, &v_sum_h,
                        &mut d_m_old_h, &mut d_k_sum_h, &mut d_v_sum_h, hd, 0.95,
                    );
                    unsafe {
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            (d_m_old.ptr() as *mut u8).add(h * dd_mem * 4) as *mut _, d_m_old_h.as_ptr() as *const _, dd_mem * 4);
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            (d_k_sum.ptr() as *mut u8).add(h * hd * 4) as *mut _, d_k_sum_h.as_ptr() as *const _, hd * 4);
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            (d_v_sum.ptr() as *mut u8).add(h * hd * 4) as *mut _, d_v_sum_h.as_ptr() as *const _, hd * 4);
                    }
                }

                if shard_idx < first_ret {
                    d_m_carry = d_m_old;
                    continue;
                }

                let cache_idx = shard_idx - first_ret;

                // Step 2: Per-head shard summary backward → d_local_y per-head [nh, shard_len, hd]
                let d_local_y_ph = GpuBuf::<f32>::zeros(bs_mem * shard_len * hd);
                for h in 0..bs_mem {
                    let d_k_sum_h = GpuBuf::<f32>::zeros(hd);
                    let d_v_sum_h = GpuBuf::<f32>::zeros(hd);
                    unsafe {
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            d_k_sum_h.ptr() as *mut _, (d_k_sum.as_ptr() as *const u8).add(h * hd * 4) as *const _, hd * 4);
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            d_v_sum_h.ptr() as *mut _, (d_v_sum.as_ptr() as *const u8).add(h * hd * 4) as *const _, hd * 4);
                    }
                    let mut d_local_y_h = GpuBuf::<f32>::zeros(shard_len * hd);
                    crate::dispatch::tnt_shard_summary_mean_backward_dd(
                        &d_k_sum_h, &d_v_sum_h, &mut d_local_y_h, shard_len, hd,
                    );
                    unsafe {
                        crate::gpu_forward::gpu_buf_memcpy_d2d(
                            (d_local_y_ph.ptr() as *mut u8).add(h * shard_len * hd * 4) as *mut _,
                            d_local_y_h.as_ptr() as *const _, shard_len * hd * 4);
                    }
                }
                // Reshape per-head d_local_y to d-space for combining with upstream d_y
                let d_local_y_global = crate::gpu_forward::reshape_from_per_head(&d_local_y_ph, 1, shard_len, nh, hd);

                // Step 3: Combine upstream d_y (d-space) with d_local_y (d-space)
                let d_y_shard_slice = d_y.slice(shard_start * d, shard_len * d);
                let d_y_upstream_shard = GpuBuf::<f32>::zeros(shard_len * d);
                unsafe {
                    let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                        d_y_upstream_shard.ptr() as *mut std::ffi::c_void,
                        d_y_shard_slice.as_ptr() as *const std::ffi::c_void,
                        shard_len * d * 4,
                    );
                    assert_eq!(rc, 0, "TNT backward: d_y upstream copy failed (rc={rc})");
                }
                let mut d_y_combined = GpuBuf::<f32>::zeros(shard_len * d);
                crate::dispatch::tnt_combine_gradients_dd(
                    &d_y_upstream_shard, &d_local_y_global,
                    &mut d_y_combined, shard_len * d,
                );

                // Spec 51: reshape combined d_y to per-head for inner backward
                let d_y_combined_ph = crate::gpu_forward::reshape_to_per_head(&d_y_combined, 1, shard_len, nh, hd);

                // Step 4: Pad d_y_combined per-head to [nh*n_batch, cl, hd]
                let padded_len = n_batch * cl;
                let ph_padded_elems = bs_mem * padded_len;
                let d_y_padded = if shard_len == padded_len {
                    d_y_combined_ph
                } else {
                    let dp = GpuBuf::<f32>::zeros(ph_padded_elems * hd);
                    unsafe {
                        for h in 0..bs_mem {
                            let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                                (dp.ptr() as *mut u8).add(h * padded_len * hd * 4) as *mut _,
                                (d_y_combined_ph.as_ptr() as *const u8).add(h * shard_len * hd * 4) as *const _,
                                shard_len * hd * 4,
                            );
                            assert_eq!(rc, 0, "TNT backward: d_y padding copy failed (rc={rc})");
                        }
                    }
                    dp
                };

                // Step 5: Inner backward kernel (per-head: kernel_batch = n_batch*nh, d = hd)
                let inner_cache = &shard_inner_caches[cache_idx];
                let shard_tokens = ph_padded_elems;  // nh * padded_len
                let mut d_k_shard = GpuBuf::<f32>::zeros(shard_tokens * hd);
                let mut d_v_shard = GpuBuf::<f32>::zeros(shard_tokens * hd);
                let mut d_q_shard = GpuBuf::<f32>::zeros(shard_tokens * hd);
                let mut d_alpha_shard = GpuBuf::<f32>::zeros(shard_tokens);
                let mut d_theta_shard = GpuBuf::<f32>::zeros(shard_tokens);
                let mut d_eta_shard = GpuBuf::<f32>::zeros(shard_tokens);
                let has_eta = matches!(inner_cache, GpuMemoryCache::Titans { .. });

                let (shard_k_norms, shard_q_norms) = match inner_cache {
                    GpuMemoryCache::Titans { k_norms, q_norms, .. } => (k_norms, q_norms),
                    GpuMemoryCache::Delta { k_norms, q_norms, .. } => (k_norms, q_norms),
                    _ => unreachable!("TNT inner cache must be Titans or Delta"),
                };

                let is_proxy = match inner_cache {
                    GpuMemoryCache::Titans { proxy, .. } | GpuMemoryCache::Delta { proxy, .. } => *proxy,
                    _ => false,
                };

                match inner_cache {
                    GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, .. } => {
                        let mut d_m_initial = GpuBuf::<f32>::zeros(kernel_batch * dd_mem);
                        let mut d_s_initial = GpuBuf::<f32>::zeros(kernel_batch * dd_mem);

                        let (m_for_bw, s_for_bw) = if is_proxy {
                            let m_bcast = GpuBuf::<f32>::zeros(kernel_batch * (cl + 1) * dd_mem);
                            let s_bcast = GpuBuf::<f32>::zeros(kernel_batch * (cl + 1) * dd_mem);
                            unsafe {
                                let dd_i32 = i32::try_from(dd_mem).expect("dd_mem overflows i32");
                                let slots_i32 = i32::try_from(cl + 1).expect("cl+1 overflows i32");
                                let nb_i32 = i32::try_from(kernel_batch).expect("kernel_batch overflows i32");
                                let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                                    m_bcast.ptr(), m_states.as_ptr(), dd_i32, slots_i32, nb_i32,
                                );
                                assert_eq!(rc, 0, "broadcast_fill M failed (rc={rc})");
                                let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                                    s_bcast.ptr(), s_states.as_ptr(), dd_i32, slots_i32, nb_i32,
                                );
                                assert_eq!(rc, 0, "broadcast_fill S failed (rc={rc})");
                            }
                            (m_bcast, s_bcast)
                        } else {
                            (m_states.dup(), s_states.dup())
                        };

                        crate::dispatch::titans_backward_dd(
                            k_mem, v_mem, q_mem, alpha, theta, eta,
                            &m_for_bw, &s_for_bw, &d_y_padded,
                            &mut d_k_shard, &mut d_v_shard, &mut d_q_shard,
                            &mut d_alpha_shard, &mut d_theta_shard, &mut d_eta_shard,
                            &mut d_m_initial, &mut d_s_initial,
                            cl, hd, kernel_batch,
                            cfg.error_clip_for_level(level),
                        );

                        if !is_proxy {
                            for b in 0..kernel_batch {
                                unsafe {
                                    crate::cuda_ffi::saxpy_cuda(
                                        1.0, d_m_initial.as_ptr().add(b * dd_mem),
                                        (d_m_old.ptr() as *mut u8).add((b / n_batch) * dd_mem * 4) as *mut f32,
                                        dd_mem as i32,
                                    );
                                }
                            }
                        }
                    }
                    GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, .. } => {
                        let mut d_m_initial = GpuBuf::<f32>::zeros(kernel_batch * dd_mem);

                        let m_for_bw = if is_proxy {
                            let m_bcast = GpuBuf::<f32>::zeros(kernel_batch * (cl + 1) * dd_mem);
                            unsafe {
                                let rc = crate::cuda_ffi::broadcast_fill_f32_cuda(
                                    m_bcast.ptr(), m_states.as_ptr(),
                                    i32::try_from(dd_mem).expect("dd_mem overflows i32"),
                                    i32::try_from(cl + 1).expect("cl+1 overflows i32"),
                                    i32::try_from(kernel_batch).expect("kernel_batch overflows i32"),
                                );
                                assert_eq!(rc, 0, "broadcast_fill M failed (rc={rc})");
                            }
                            m_bcast
                        } else {
                            m_states.dup()
                        };

                        crate::dispatch::delta_backward_dd(
                            k_mem, v_mem, q_mem, alpha, theta,
                            &m_for_bw, &d_y_padded,
                            &mut d_k_shard, &mut d_v_shard, &mut d_q_shard,
                            &mut d_alpha_shard, &mut d_theta_shard, &mut d_m_initial,
                            cl, hd, kernel_batch,
                            cfg.error_clip_for_level(level),
                        );

                        if !is_proxy {
                            for b in 0..kernel_batch {
                                unsafe {
                                    crate::cuda_ffi::saxpy_cuda(
                                        1.0, d_m_initial.as_ptr().add(b * dd_mem),
                                        (d_m_old.ptr() as *mut u8).add((b / n_batch) * dd_mem * 4) as *mut f32,
                                        dd_mem as i32,
                                    );
                                }
                            }
                        }
                    }
                    _ => unreachable!("TNT inner cache must be Titans or Delta"),
                }

                // CS-39 straight-through: clamp masks on per-head gate data (element-wise, works)
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

                // Spec 51: reshape per-head grads back to d-space for L2 backward + gate backward
                let d_k_shard_dm = crate::gpu_forward::reshape_from_per_head(&d_k_shard, 1, padded_len, nh, hd);
                let d_v_shard_dm = crate::gpu_forward::reshape_from_per_head(&d_v_shard, 1, padded_len, nh, hd);
                let d_q_shard_dm = crate::gpu_forward::reshape_from_per_head(&d_q_shard, 1, padded_len, nh, hd);
                let d_alpha_shard_dm = crate::gpu_forward::sum_gates_across_heads(&d_alpha_shard, 1, padded_len, nh);
                let d_theta_shard_dm = crate::gpu_forward::sum_gates_across_heads(&d_theta_shard, 1, padded_len, nh);
                let d_eta_shard_dm = if has_eta {
                    crate::gpu_forward::sum_gates_across_heads(&d_eta_shard, 1, padded_len, nh)
                } else {
                    GpuBuf::<f32>::zeros(padded_len)
                };

                // Reshape cached per-head k_mem/v_mem/q_mem back to d-space for gate backward + L2 backward
                let (cache_k_mem_dm, cache_v_mem_dm, cache_q_mem_dm, cache_alpha_dm, cache_theta_dm, cache_eta_dm) = match inner_cache {
                    GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, .. } => {
                        let k_dm = crate::gpu_forward::reshape_from_per_head(k_mem, 1, padded_len, nh, hd);
                        let v_dm = crate::gpu_forward::reshape_from_per_head(v_mem, 1, padded_len, nh, hd);
                        let q_dm = crate::gpu_forward::reshape_from_per_head(q_mem, 1, padded_len, nh, hd);
                        // Extract head 0's gates (all heads have same value per position)
                        let a_dm = GpuBuf::<f32>::zeros(padded_len);
                        let t_dm = GpuBuf::<f32>::zeros(padded_len);
                        let e_dm = GpuBuf::<f32>::zeros(padded_len);
                        unsafe {
                            crate::gpu_forward::gpu_buf_memcpy_d2d(a_dm.ptr() as *mut _, alpha.as_ptr() as *const _, padded_len * 4);
                            crate::gpu_forward::gpu_buf_memcpy_d2d(t_dm.ptr() as *mut _, theta.as_ptr() as *const _, padded_len * 4);
                            crate::gpu_forward::gpu_buf_memcpy_d2d(e_dm.ptr() as *mut _, eta.as_ptr() as *const _, padded_len * 4);
                        }
                        (k_dm, v_dm, q_dm, a_dm, t_dm, Some(e_dm))
                    }
                    GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, .. } => {
                        let k_dm = crate::gpu_forward::reshape_from_per_head(k_mem, 1, padded_len, nh, hd);
                        let v_dm = crate::gpu_forward::reshape_from_per_head(v_mem, 1, padded_len, nh, hd);
                        let q_dm = crate::gpu_forward::reshape_from_per_head(q_mem, 1, padded_len, nh, hd);
                        let a_dm = GpuBuf::<f32>::zeros(padded_len);
                        let t_dm = GpuBuf::<f32>::zeros(padded_len);
                        unsafe {
                            crate::gpu_forward::gpu_buf_memcpy_d2d(a_dm.ptr() as *mut _, alpha.as_ptr() as *const _, padded_len * 4);
                            crate::gpu_forward::gpu_buf_memcpy_d2d(t_dm.ptr() as *mut _, theta.as_ptr() as *const _, padded_len * 4);
                        }
                        (k_dm, v_dm, q_dm, a_dm, t_dm, None)
                    }
                    _ => unreachable!(),
                };

                // Step 5b: L2 normalization backward for k/q (d-space, using d-space norms)
                let d_k_shard_final;
                let d_q_shard_final;
                {
                    let d_k_raw = GpuBuf::<f32>::zeros(padded_len * d);
                    let d_q_raw = GpuBuf::<f32>::zeros(padded_len * d);
                    unsafe {
                        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
                            d_k_shard_dm.as_ptr(), cache_k_mem_dm.as_ptr(), shard_k_norms.as_ptr(),
                            d_k_raw.ptr(), padded_len as i32, d as i32, 1e-8,
                        );
                        crate::cuda_ffi::l2_normalize_backward_f32_cuda(
                            d_q_shard_dm.as_ptr(), cache_q_mem_dm.as_ptr(), shard_q_norms.as_ptr(),
                            d_q_raw.ptr(), padded_len as i32, d as i32, 1e-8,
                        );
                    }
                    d_k_shard_final = d_k_raw;
                    d_q_shard_final = d_q_raw;
                }

                // Step 6: Accumulate unpadded shard gradients into full-sequence totals (d-space)
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_k_shard_final.as_ptr(), d_k_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_v_shard_dm.as_ptr(), d_v_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                    crate::cuda_ffi::saxpy_cuda(
                        1.0, d_q_shard_final.as_ptr(), d_q_mem_total.ptr().add(shard_start * d),
                        (shard_len * d) as i32,
                    );
                }

                // Step 7: Gate backward (d-space)
                {
                    let mut tmp_dw_alpha = GpuBuf::<f32>::zeros(2 * d);
                    let mut tmp_db_alpha = GpuBuf::<f32>::zeros(1);
                    let mut tmp_dw_theta = GpuBuf::<f32>::zeros(2 * d);
                    let mut tmp_db_theta = GpuBuf::<f32>::zeros(1);
                    let mut tmp_dw_eta   = GpuBuf::<f32>::zeros(2 * d);
                    let mut tmp_db_eta   = GpuBuf::<f32>::zeros(1);

                    if let Some(ref eta_dm) = cache_eta_dm {
                        crate::dispatch::gate_backward_dd(
                            &d_alpha_shard_dm, &cache_alpha_dm,
                            Some(&d_theta_shard_dm), Some(&cache_theta_dm),
                            Some(&d_eta_shard_dm), Some(eta_dm),
                            &cache_k_mem_dm, &cache_v_mem_dm,
                            &mut tmp_dw_alpha, &mut tmp_db_alpha,
                            &mut tmp_dw_theta, &mut tmp_db_theta,
                            &mut tmp_dw_eta,   &mut tmp_db_eta,
                            padded_len, d,
                        );
                    } else {
                        crate::dispatch::gate_backward_dd(
                            &d_alpha_shard_dm, &cache_alpha_dm,
                            Some(&d_theta_shard_dm), Some(&cache_theta_dm),
                            None, None,
                            &cache_k_mem_dm, &cache_v_mem_dm,
                            &mut tmp_dw_alpha, &mut tmp_db_alpha,
                            &mut tmp_dw_theta, &mut tmp_db_theta,
                            &mut tmp_dw_eta,   &mut tmp_db_eta,
                            padded_len, d,
                        );
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
            let mut d_embedded = GpuBuf::<f32>::zeros(s * d);
            crate::dispatch::cublas_matmul_acc_dd(&d_k_mem_total, &level_params.w_k_mem, &mut d_embedded, s, d, d);
            crate::dispatch::cublas_matmul_acc_dd(&d_v_mem_total, &level_params.w_v_mem, &mut d_embedded, s, d, d);
            crate::dispatch::cublas_matmul_acc_dd(&d_q_mem_total, &level_params.w_q_mem, &mut d_embedded, s, d, d);

            d_embedded
        }
        // ── MLP memory: MONETA (l_p) / YAAD (Huber) backward ───────────
        GpuMemoryCache::Mlp {
            k_mem, v_mem, q_mem, alpha, theta,
            w1_states, w2_states, k_norms, q_norms,
            w1_boundary, w2_boundary,
        } => {
            assert_eq!(batch_size, 1, "MLP memory backward with batch_size > 1 is not supported");
            let dh = cfg.d_hidden;
            let dh_i32 = i32::try_from(dh).expect("d_hidden exceeds i32::MAX");
            let d_k_mem = GpuBuf::zeros(s * d);
            let d_v_mem = GpuBuf::zeros(s * d);
            let d_q_mem = GpuBuf::zeros(s * d);
            let d_alpha = GpuBuf::zeros(s);
            let d_theta = GpuBuf::zeros(s);
            let w1_size = dh * d;
            let w2_size = d * dh;
            let d_w1_initial = GpuBuf::zeros(w1_size);
            let d_w2_initial = GpuBuf::zeros(w2_size);

            match (w1_boundary, w2_boundary) {
                // YAAD: Huber bias + decoupled L2 retention
                (Some(w1b), Some(w2b)) => {
                    unsafe {
                        crate::cuda_ffi::mlp_backward_huber_f32_cuda(
                            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
                            alpha.as_ptr(), theta.as_ptr(),
                            w1_states.as_ptr(), w2_states.as_ptr(),
                            w1b.as_ptr(), w2b.as_ptr(),
                            d_y.as_ptr(),
                            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
                            d_alpha.ptr(), d_theta.ptr(),
                            d_w1_initial.ptr(), d_w2_initial.ptr(),
                            s as i32, d as i32, dh_i32,
                            cfg.delta, cfg.lambda_local, cfg.lambda_2,
                        );
                    }
                }
                // MONETA: l_p bias + L2 retention (LQ backward deferred — q must be 2)
                (None, None) => {
                    assert!(
                        (cfg.lq_q - 2.0).abs() < 1e-6,
                        "MONETA GPU backward only supports lq_q=2.0 (L2 retention). \
                         Got lq_q={:.4}. LQ backward (q > 2) is deferred.",
                        cfg.lq_q,
                    );
                    unsafe {
                        crate::cuda_ffi::mlp_backward_lp_f32_cuda(
                            k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
                            alpha.as_ptr(), theta.as_ptr(),
                            w1_states.as_ptr(), w2_states.as_ptr(),
                            d_y.as_ptr(),
                            d_k_mem.ptr(), d_v_mem.ptr(), d_q_mem.ptr(),
                            d_alpha.ptr(), d_theta.ptr(),
                            d_w1_initial.ptr(), d_w2_initial.ptr(),
                            s as i32, d as i32, dh_i32,
                            cfg.lp_p, cfg.sign_sharpness, cfg.lambda_2, cfg.lq_q,
                        );
                    }
                }
                // Mismatched boundary snapshots — invariant violation
                (Some(_), None) | (None, Some(_)) => {
                    panic!(
                        "MLP backward: mismatched boundary snapshots — \
                         W1 and W2 boundaries must both be present (YAAD) or both absent (MONETA)"
                    );
                }
            }

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

            // TODO: accumulate d_w1_initial/d_w2_initial into level_grads for outer-loop update
            // For now, these gradients are dropped — MLP inner-loop dominates (verified in Rust ref)

            accumulate_projection_grads(
                level_params, embedded,
                k_mem, v_mem, q_mem, alpha, Some(theta), None,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, Some(&d_theta), None,
                k_norms, q_norms,
                level_grads, s, d, 1,
            )
        }
        // ── SwiGLU: stateless MLP, direct weight grads ───────────────
        GpuMemoryCache::SwiGlu { gate_buf, up_buf, fused_buf, cache_buf } => {
            let inter = cfg.intermediate_size;
            let d_x = GpuBuf::zeros(batch_size * s * d);
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

/// Delta Rule checkpointed backward — batched across heads.
///
/// All inputs are in per-head layout: `[bs, s, d]` for k/v/q/d_y,
/// `[bs, s]` for alpha/theta, `[bs, num_ckpt, dd]` for m_checkpoints.
/// Forward replay runs sequentially per head; segment backward is batched.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn delta_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize, bs: usize, error_clip: f32,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let segments = segment_boundaries(s, c);
    let num_ckpt = crate::gpu_forward::checkpoint_count(s, c);

    // Validate input buffer sizes against expected flattened layout
    debug_assert!(k_mem.len() >= bs * s * d, "k_mem too small: {} < {}", k_mem.len(), bs * s * d);
    debug_assert!(v_mem.len() >= bs * s * d, "v_mem too small: {} < {}", v_mem.len(), bs * s * d);
    debug_assert!(q_mem.len() >= bs * s * d, "q_mem too small: {} < {}", q_mem.len(), bs * s * d);
    debug_assert!(alpha.len() >= bs * s, "alpha too small: {} < {}", alpha.len(), bs * s);
    debug_assert!(theta.len() >= bs * s, "theta too small: {} < {}", theta.len(), bs * s);
    debug_assert!(d_y.len() >= bs * s * d, "d_y too small: {} < {}", d_y.len(), bs * s * d);
    debug_assert!(m_checkpoints.len() >= bs * num_ckpt * dd, "m_checkpoints too small: {} < {}", m_checkpoints.len(), bs * num_ckpt * dd);

    // Accumulation buffers — batched [bs, s, d] / [bs, s]
    let mut d_k_mem = GpuBuf::zeros(bs * s * d);
    let mut d_v_mem = GpuBuf::zeros(bs * s * d);
    let mut d_q_mem = GpuBuf::zeros(bs * s * d);
    let mut d_alpha = GpuBuf::zeros(bs * s);
    let mut d_theta = GpuBuf::zeros(bs * s);

    // d_M seed: [bs, dd] — starts as zeros for the last segment
    let d_m_seed = GpuBuf::zeros(bs * dd);

    // Pre-allocate scratch buffers — batched [bs, (max_seg+1), dd]
    let max_seg = c.min(s);
    let local_m_states: GpuBuf<f32> = GpuBuf::zeros(bs * (max_seg + 1) * dd);
    let local_y: GpuBuf<f32> = GpuBuf::zeros(bs * max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(bs * dd);

    // Process segments in reverse
    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // 1. Replay forward — batched across all heads in one kernel launch.
        //    input_stride=s so the kernel skips s tokens between heads (the full
        //    sequence stride), m_stride=num_ckpt*dd so each head picks its
        //    checkpoint from the interleaved [bs, num_ckpt, dd] layout.
        local_m_states.zero();
        local_y.zero();
        unsafe {
            crate::cuda_ffi::delta_forward_f32_cuda(
                k_mem.as_ptr().add(t_start * d),
                v_mem.as_ptr().add(t_start * d),
                q_mem.as_ptr().add(t_start * d),
                alpha.as_ptr().add(t_start),
                theta.as_ptr().add(t_start),
                m_checkpoints.as_ptr().add(ckpt_idx * dd),
                local_m_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32, bs as i32,
                s as i32, (num_ckpt * dd) as i32, error_clip,
            );
        }
        crate::dispatch::cuda_sync();

        // 2. Segment backward — batched across all heads
        seg_d_m_out.zero();
        crate::dispatch::delta_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha, theta,
            &local_m_states, d_y,
            &d_m_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut d_theta, &mut seg_d_m_out,
            t_start, t_end, d, bs, s, error_clip,
        );
        crate::dispatch::cuda_sync();

        // 3. Propagate d_M seed to earlier segment (contiguous copy for all heads)
        d_m_seed.copy_from_device(&seg_d_m_out);
    }

    (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta)
}

/// Titans checkpointed backward — batched across heads.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn titans_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>, theta: &GpuBuf<f32>, eta: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, s_checkpoints: &GpuBuf<f32>,
    d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize, bs: usize, error_clip: f32,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let segments = segment_boundaries(s, c);
    let num_ckpt = crate::gpu_forward::checkpoint_count(s, c);

    // Validate input buffer sizes
    debug_assert!(k_mem.len() >= bs * s * d, "k_mem too small: {} < {}", k_mem.len(), bs * s * d);
    debug_assert!(v_mem.len() >= bs * s * d, "v_mem too small: {} < {}", v_mem.len(), bs * s * d);
    debug_assert!(q_mem.len() >= bs * s * d, "q_mem too small: {} < {}", q_mem.len(), bs * s * d);
    debug_assert!(alpha.len() >= bs * s, "alpha too small: {} < {}", alpha.len(), bs * s);
    debug_assert!(theta.len() >= bs * s, "theta too small: {} < {}", theta.len(), bs * s);
    debug_assert!(eta.len() >= bs * s, "eta too small: {} < {}", eta.len(), bs * s);
    debug_assert!(d_y.len() >= bs * s * d, "d_y too small: {} < {}", d_y.len(), bs * s * d);
    debug_assert!(m_checkpoints.len() >= bs * num_ckpt * dd, "m_checkpoints too small: {} < {}", m_checkpoints.len(), bs * num_ckpt * dd);
    debug_assert!(s_checkpoints.len() >= bs * num_ckpt * dd, "s_checkpoints too small: {} < {}", s_checkpoints.len(), bs * num_ckpt * dd);

    let mut d_k_mem = GpuBuf::zeros(bs * s * d);
    let mut d_v_mem = GpuBuf::zeros(bs * s * d);
    let mut d_q_mem = GpuBuf::zeros(bs * s * d);
    let mut d_alpha = GpuBuf::zeros(bs * s);
    let mut d_theta = GpuBuf::zeros(bs * s);
    let mut d_eta = GpuBuf::zeros(bs * s);
    let d_m_seed = GpuBuf::zeros(bs * dd);
    let d_s_seed = GpuBuf::zeros(bs * dd);

    let max_seg = c.min(s);
    let local_m_states: GpuBuf<f32> = GpuBuf::zeros(bs * (max_seg + 1) * dd);
    let local_s_states: GpuBuf<f32> = GpuBuf::zeros(bs * (max_seg + 1) * dd);
    let local_y: GpuBuf<f32> = GpuBuf::zeros(bs * max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(bs * dd);
    let mut seg_d_s_out = GpuBuf::zeros(bs * dd);

    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // 1. Replay forward — batched across all heads in one kernel launch.
        //    input_stride=s, m_stride=num_ckpt*dd (same stride separation as delta).
        local_m_states.zero();
        local_s_states.zero();
        local_y.zero();
        unsafe {
            crate::cuda_ffi::titans_forward_f32_cuda(
                k_mem.as_ptr().add(t_start * d),
                v_mem.as_ptr().add(t_start * d),
                q_mem.as_ptr().add(t_start * d),
                alpha.as_ptr().add(t_start),
                theta.as_ptr().add(t_start),
                eta.as_ptr().add(t_start),
                m_checkpoints.as_ptr().add(ckpt_idx * dd),
                s_checkpoints.as_ptr().add(ckpt_idx * dd),
                local_m_states.ptr(),
                local_s_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32, bs as i32,
                s as i32, (num_ckpt * dd) as i32, error_clip,
            );
        }
        crate::dispatch::cuda_sync();

        // 2. Segment backward — batched across all heads
        seg_d_m_out.zero();
        seg_d_s_out.zero();
        crate::dispatch::titans_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha, theta, eta,
            &local_m_states, &local_s_states, d_y,
            &d_m_seed, &d_s_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut d_theta, &mut d_eta,
            &mut seg_d_m_out, &mut seg_d_s_out,
            t_start, t_end, d, bs, s, error_clip,
        );
        crate::dispatch::cuda_sync();

        d_m_seed.copy_from_device(&seg_d_m_out);
        d_s_seed.copy_from_device(&seg_d_s_out);
    }

    (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta)
}

/// Hebbian checkpointed backward — batched across heads.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn hebbian_backward_checkpointed(
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>, q_mem: &GpuBuf<f32>,
    alpha: &GpuBuf<f32>,
    m_checkpoints: &GpuBuf<f32>, d_y: &GpuBuf<f32>,
    s: usize, d: usize, c: usize, bs: usize,
) -> (GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>, GpuBuf<f32>) {
    let dd = d * d;
    let segments = segment_boundaries(s, c);
    let num_ckpt = crate::gpu_forward::checkpoint_count(s, c);

    // Validate input buffer sizes
    debug_assert!(k_mem.len() >= bs * s * d, "k_mem too small: {} < {}", k_mem.len(), bs * s * d);
    debug_assert!(v_mem.len() >= bs * s * d, "v_mem too small: {} < {}", v_mem.len(), bs * s * d);
    debug_assert!(q_mem.len() >= bs * s * d, "q_mem too small: {} < {}", q_mem.len(), bs * s * d);
    debug_assert!(alpha.len() >= bs * s, "alpha too small: {} < {}", alpha.len(), bs * s);
    debug_assert!(d_y.len() >= bs * s * d, "d_y too small: {} < {}", d_y.len(), bs * s * d);
    debug_assert!(m_checkpoints.len() >= bs * num_ckpt * dd, "m_checkpoints too small: {} < {}", m_checkpoints.len(), bs * num_ckpt * dd);

    let mut d_k_mem = GpuBuf::zeros(bs * s * d);
    let mut d_v_mem = GpuBuf::zeros(bs * s * d);
    let mut d_q_mem = GpuBuf::zeros(bs * s * d);
    let mut d_alpha = GpuBuf::zeros(bs * s);
    let d_m_seed = GpuBuf::zeros(bs * dd);

    let max_seg = c.min(s);
    let local_m_states: GpuBuf<f32> = GpuBuf::zeros(bs * (max_seg + 1) * dd);
    let local_y: GpuBuf<f32> = GpuBuf::zeros(bs * max_seg * d);
    let mut seg_d_m_out = GpuBuf::zeros(bs * dd);

    for &(t_start, t_end, ckpt_idx) in segments.iter().rev() {
        let seg_len = t_end - t_start;

        // 1. Replay forward — batched across all heads in one kernel launch.
        //    input_stride=s, m_stride=num_ckpt*dd (same stride separation as delta).
        local_m_states.zero();
        local_y.zero();
        unsafe {
            crate::cuda_ffi::hebbian_forward_f32_cuda(
                k_mem.as_ptr().add(t_start * d),
                v_mem.as_ptr().add(t_start * d),
                q_mem.as_ptr().add(t_start * d),
                alpha.as_ptr().add(t_start),
                m_checkpoints.as_ptr().add(ckpt_idx * dd),
                local_m_states.ptr(),
                local_y.ptr(),
                seg_len as i32, d as i32, bs as i32,
                s as i32, (num_ckpt * dd) as i32,
            );
        }
        crate::dispatch::cuda_sync();

        // 2. Segment backward — batched across all heads
        seg_d_m_out.zero();
        crate::dispatch::hebbian_backward_dd_segment(
            k_mem, v_mem, q_mem, alpha,
            &local_m_states, d_y,
            &d_m_seed,
            &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
            &mut d_alpha, &mut seg_d_m_out,
            t_start, t_end, d, bs, s,
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
    let d_k_raw = GpuBuf::zeros(bsd);
    let d_q_raw = GpuBuf::zeros(bsd);
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
