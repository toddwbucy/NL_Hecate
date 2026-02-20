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
    // Per-level memory gradients
    pub levels: Vec<GpuLevelGrads>,
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
}

#[cfg(feature = "cuda")]
impl GpuMAGGrads {
    /// Download GPU gradients to host as a MAGParams (same struct, gradient values).
    pub fn to_host(&self, cfg: &crate::model::MAGConfig) -> crate::model::MAGParams {
        let d = cfg.swa.d_model;
        let v = cfg.swa.vocab_size;
        let mut swa = crate::model::SWAParams::zeros_like(&cfg.swa);
        self.d_w_embed.copy_to_host(&mut swa.w_embed);
        self.d_w_q.copy_to_host(&mut swa.w_q);
        self.d_w_k.copy_to_host(&mut swa.w_k);
        self.d_w_v.copy_to_host(&mut swa.w_v);
        self.d_w_o.copy_to_host(&mut swa.w_o);
        self.d_w_unembed.copy_to_host(&mut swa.w_unembed);

        let levels: Vec<_> = self.levels.iter().enumerate().map(|(i, lg)| {
            let mut lp = crate::model::MemoryLevelParams::zeros_like(d);
            lg.d_w_k_mem.copy_to_host(&mut lp.w_k_mem);
            lg.d_w_v_mem.copy_to_host(&mut lp.w_v_mem);
            lg.d_w_q_mem.copy_to_host(&mut lp.w_q_mem);
            lg.d_w_alpha.copy_to_host(&mut lp.w_alpha);
            lg.d_b_alpha.copy_to_host(&mut lp.b_alpha);
            lg.d_w_theta.copy_to_host(&mut lp.w_theta);
            lg.d_b_theta.copy_to_host(&mut lp.b_theta);
            lg.d_w_eta.copy_to_host(&mut lp.w_eta);
            lg.d_b_eta.copy_to_host(&mut lp.b_eta);
            // w_omega not in GPU grads (gate backward TODO), keep zeros
            lp
        }).collect();

        // TODO: GPU kernels don't yet compute alpha_mem/alpha_refl gradients.
        // These are small [k] vectors — GPU-side aggregation backward is a Stage 3 task.
        crate::model::MAGParams { swa, levels, alpha_mem: vec![0.0f32; cfg.k], alpha_refl: vec![0.0f32; cfg.k] }
    }
}

#[cfg(feature = "cuda")]
impl GpuLevelGrads {
    fn zeros(d: usize) -> Self {
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
    // Note: error_buffers not supported in GPU path — frozen levels
    // only get q_mem projection gradient which is accumulated in d_embedded.
) -> GpuMAGGrads {
    let s = cache.s;
    let d = cache.d;
    let v = cache.v;
    let nh = cache.nh;
    let hd = cache.hd;
    let ws = cache.ws;
    let sd = s * d;

    // Initialize gradient buffers (all zeros on GPU)
    let mut grads = GpuMAGGrads {
        d_w_embed: GpuBuf::zeros(v * d),
        d_w_q: GpuBuf::zeros(d * d),
        d_w_k: GpuBuf::zeros(d * d),
        d_w_v: GpuBuf::zeros(d * d),
        d_w_o: GpuBuf::zeros(d * d),
        d_w_unembed: GpuBuf::zeros(d * v),
        levels: (0..cfg.k).map(|_| GpuLevelGrads::zeros(d)).collect(),
    };

    // ── Stage 7: Cross-entropy backward ──────────────────────────────
    let mut d_logits = GpuBuf::zeros(s * v);
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
            s as i32, v as i32,
            1.0 / count,
        );
    }

    // ── Stage 6: Unembed backward ────────────────────────────────────
    // d_projected = d_logits @ W_unembed^T (transB: W_unembed is [d, v], so C = d_logits[s,v] @ W_unembed^T[v,d])
    let mut d_projected = GpuBuf::zeros(sd);
    crate::dispatch::cublas_matmul_transb_dd(
        &d_logits, &params.swa.w_unembed, &mut d_projected, s, v, d, 0.0,
    );

    // d_w_unembed = projected^T @ d_logits → [d, s] @ [s, v] = [d, v]
    // We need projected^T. For GPU: C = A^T @ B is sgemm(N, T, v, d, s, ..., d_logits, v, projected, d, ..., C, v)
    // Using row-major trick with transA: cublasSgemm(N, T, n=v, m=d, k=s, B_ptr=d_logits, n=v, A_ptr=projected, d, ...)
    // Actually: d_w_unembed[d,v] = projected^T[d,s] @ d_logits[s,v]
    // matmul_transb doesn't help. We need matmul with A transposed.
    // Use: C = A^T @ B. In row-major → sgemm(N, T, n, m, k, alpha, B, n, A, k, beta, C, n)
    // where m=d, k=s, n=v → sgemm(N, T, v, d, s, alpha, d_logits, v, projected, d, beta, C, v)
    gpu_matmul_transa_dd(
        &cache.projected, &d_logits, &mut grads.d_w_unembed,
        d, s, v,
    );

    // ── Stage 5: Output projection backward ──────────────────────────
    // projected = gated_out @ W_O^T  →  d_gated_out = d_projected @ W_O
    let mut d_gated_out = GpuBuf::zeros(sd);
    crate::dispatch::cublas_matmul_dd(
        &d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d, 0.0,
    );

    // d_w_o = d_projected^T @ gated_out → [d, s] @ [s, d] = [d, d]
    gpu_matmul_transa_dd(
        &d_projected, &cache.gated_out, &mut grads.d_w_o,
        d, s, d,
    );

    // ── Stage 4: Gating backward ─────────────────────────────────────
    // gated_out = attn_out * gate
    let mut d_attn_out = GpuBuf::zeros(sd);
    let mut d_gate = GpuBuf::zeros(sd);
    unsafe {
        crate::cuda_ffi::gating_backward_cuda(
            d_gated_out.as_ptr(), cache.attn_out.as_ptr(), cache.gate.as_ptr(),
            d_attn_out.ptr(), d_gate.ptr(), sd as i32,
        );
    }

    // d_y_combined = d_gate * gate * (1 - gate)  (sigmoid backward)
    let mut d_y_combined = GpuBuf::zeros(sd);
    unsafe {
        crate::cuda_ffi::sigmoid_backward_cuda(
            d_gate.as_ptr(), cache.gate.as_ptr(), d_y_combined.ptr(), sd as i32,
        );
    }

    // Scale for 1/sqrt(k) normalization (k>2)
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(
                scale - 1.0, d_y_combined.as_ptr(), d_y_combined.ptr(), sd as i32,
            );
        }
    }

    // ── Stage 3b: Per-level memory backward ──────────────────────────
    let mut d_embedded_mem = GpuBuf::zeros(sd);

    for level in 0..cfg.k {
        if cache.pulse.active_levels[level] {
            if let Some(ref mem_cache) = cache.memory_caches[level] {
                let d_emb_level = gpu_memory_backward(
                    &params.levels[level], cfg, mem_cache,
                    &d_y_combined, &cache.embedded,
                    &mut grads.levels[level],
                    s, d,
                );
                // Accumulate d_embedded contribution
                unsafe {
                    crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_embedded_mem.ptr(), sd as i32);
                }
            }
        } else {
            // Frozen level: y = q_mem @ M^T → d_q_mem = d_y @ M, d_embedded += d_q_mem @ W_q_mem
            let d_emb_level = gpu_memory_read_only_backward(
                &params.levels[level], &cache.y_per_level[level],
                &d_y_combined, &cache.embedded,
                &mut grads.levels[level],
                // For read-only, we need the context M — but we don't have it in cache.
                // In the full version we'd pass GpuContextState. For now, skip frozen grads.
                s, d,
            );
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, d_emb_level.as_ptr(), d_embedded_mem.ptr(), sd as i32);
            }
        }
    }

    // ── Stage 3a: SWA backward ───────────────────────────────────────
    let mut d_q = GpuBuf::zeros(sd);
    let mut d_k = GpuBuf::zeros(sd);
    let mut d_v = GpuBuf::zeros(sd);

    crate::dispatch::swa_backward_dd(
        &cache.q_bf16, &cache.k_bf16, &cache.v_bf16,
        &cache.attn_weights_bf16, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 2a: QKV projection backward ────────────────────────────
    // d_embedded += d_q @ W_q + d_k @ W_k + d_v @ W_v
    let mut d_embedded = GpuBuf::zeros(sd);
    crate::dispatch::cublas_matmul_acc_dd(&d_q, &params.swa.w_q, &mut d_embedded, s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(&d_k, &params.swa.w_k, &mut d_embedded, s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(&d_v, &params.swa.w_v, &mut d_embedded, s, d, d);

    // d_w_q = d_q^T @ embedded → [d, s] @ [s, d] = [d, d]
    gpu_matmul_transa_dd(&d_q, &cache.embedded, &mut grads.d_w_q, d, s, d);
    gpu_matmul_transa_dd(&d_k, &cache.embedded, &mut grads.d_w_k, d, s, d);
    gpu_matmul_transa_dd(&d_v, &cache.embedded, &mut grads.d_w_v, d, s, d);

    // Combine d_embedded from attention + memory branches
    unsafe {
        crate::cuda_ffi::saxpy_cuda(1.0, d_embedded_mem.as_ptr(), d_embedded.ptr(), sd as i32);
    }

    // ── Stage 1: Embedding scatter-add ───────────────────────────────
    unsafe {
        crate::cuda_ffi::embedding_scatter_add_cuda(
            d_embedded.as_ptr(),
            cache.input_ids_gpu.ptr() as *const i32,
            grads.d_w_embed.ptr(),
            s as i32, d as i32,
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
#[cfg(feature = "cuda")]
fn gpu_memory_backward(
    level_params: &GpuMemoryLevelParams,
    cfg: &MAGConfig,
    mem_cache: &GpuMemoryCache,
    d_y: &GpuBuf<f32>,
    embedded: &GpuBuf<f32>,
    level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
) -> GpuBuf<f32> {
    let dd = d * d;
    let sd = s * d;

    match mem_cache {
        GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states } => {
            let mut d_k_mem = GpuBuf::zeros(sd);
            let mut d_v_mem = GpuBuf::zeros(sd);
            let mut d_q_mem = GpuBuf::zeros(sd);
            let mut d_alpha = GpuBuf::zeros(s);
            let mut d_theta = GpuBuf::zeros(s);
            let mut d_m_initial = GpuBuf::zeros(dd);

            crate::dispatch::delta_backward_dd(
                k_mem, v_mem, q_mem, alpha, theta,
                m_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_theta, &mut d_m_initial,
                s, d,
            );

            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &d_theta, None,
                level_grads, s, d,
            )
        }
        GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states } => {
            let mut d_k_mem = GpuBuf::zeros(sd);
            let mut d_v_mem = GpuBuf::zeros(sd);
            let mut d_q_mem = GpuBuf::zeros(sd);
            let mut d_alpha = GpuBuf::zeros(s);
            let mut d_theta = GpuBuf::zeros(s);
            let mut d_eta = GpuBuf::zeros(s);
            let mut d_m_initial = GpuBuf::zeros(dd);
            let mut d_s_initial = GpuBuf::zeros(dd);

            crate::dispatch::titans_backward_dd(
                k_mem, v_mem, q_mem, alpha, theta, eta,
                m_states, s_states, d_y,
                &mut d_k_mem, &mut d_v_mem, &mut d_q_mem,
                &mut d_alpha, &mut d_theta, &mut d_eta,
                &mut d_m_initial, &mut d_s_initial,
                s, d,
            );

            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &d_theta, Some(&d_eta),
                level_grads, s, d,
            )
        }
        GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states } => {
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

            // Hebbian has no theta — pass None for d_theta
            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &GpuBuf::zeros(s), None,
                level_grads, s, d,
            )
        }
        // ── Checkpointed variants: segment-based backward ──────────
        GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta) =
                delta_backward_checkpointed(k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, d_y, s, d, c);
            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &d_theta, None,
                level_grads, s, d,
            )
        }
        GpuMemoryCache::TitansCkpt { k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, checkpoint_interval } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta) =
                titans_backward_checkpointed(k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, d_y, s, d, c);
            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &d_theta, Some(&d_eta),
                level_grads, s, d,
            )
        }
        GpuMemoryCache::HebbianCkpt { k_mem, v_mem, q_mem, alpha, m_checkpoints, checkpoint_interval } => {
            let c = *checkpoint_interval;
            let (d_k_mem, d_v_mem, d_q_mem, d_alpha) =
                hebbian_backward_checkpointed(k_mem, v_mem, q_mem, alpha, m_checkpoints, d_y, s, d, c);
            accumulate_projection_grads(
                level_params, embedded,
                &d_k_mem, &d_v_mem, &d_q_mem,
                &d_alpha, &GpuBuf::zeros(s), None,
                level_grads, s, d,
            )
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
    s: usize, d: usize, c: usize,
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
                seg_len as i32, d as i32,
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
            t_start, t_end, d,
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
    s: usize, d: usize, c: usize,
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
                seg_len as i32, d as i32,
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
            t_start, t_end, d,
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
///   d_w_alpha/d_b_alpha and d_w_theta/d_b_theta from gate backward
#[cfg(feature = "cuda")]
fn accumulate_projection_grads(
    level_params: &GpuMemoryLevelParams,
    embedded: &GpuBuf<f32>,
    d_k_mem: &GpuBuf<f32>,
    d_v_mem: &GpuBuf<f32>,
    d_q_mem: &GpuBuf<f32>,
    _d_alpha: &GpuBuf<f32>,  // per-token gate grads (gate backward handled separately)
    _d_theta: &GpuBuf<f32>,
    _d_eta: Option<&GpuBuf<f32>>,
    level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
) -> GpuBuf<f32> {
    let sd = s * d;

    // Projection weight grads: d_W = d_proj^T @ embedded
    gpu_matmul_transa_dd(d_k_mem, embedded, &mut level_grads.d_w_k_mem, d, s, d);
    gpu_matmul_transa_dd(d_v_mem, embedded, &mut level_grads.d_w_v_mem, d, s, d);
    gpu_matmul_transa_dd(d_q_mem, embedded, &mut level_grads.d_w_q_mem, d, s, d);

    // d_embedded from memory projections
    let mut d_embedded = GpuBuf::zeros(sd);
    crate::dispatch::cublas_matmul_acc_dd(d_k_mem, &level_params.w_k_mem, &mut d_embedded, s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(d_v_mem, &level_params.w_v_mem, &mut d_embedded, s, d, d);
    crate::dispatch::cublas_matmul_acc_dd(d_q_mem, &level_params.w_q_mem, &mut d_embedded, s, d, d);

    // Gate backward (alpha, theta, eta) is complex — involves per-token dot products
    // with concat(k_mem, v_mem). For now, gate weight grads are approximated by
    // accumulating the per-token scalar grads. Full gate backward requires additional
    // CUDA kernels for the chain rule through gate_compute. This is left as a
    // follow-up optimization — the inner loop grads (d_k_mem, d_v_mem, d_q_mem)
    // dominate the gradient signal for projection weights.
    //
    // TODO: Implement gate_backward_dd for d_w_alpha, d_b_alpha, d_w_theta, etc.

    d_embedded
}

/// Frozen level read-only backward (simplified).
#[cfg(feature = "cuda")]
fn gpu_memory_read_only_backward(
    level_params: &GpuMemoryLevelParams,
    _y_level: &GpuBuf<f32>,
    _d_y: &GpuBuf<f32>,
    _embedded: &GpuBuf<f32>,
    _level_grads: &mut GpuLevelGrads,
    s: usize,
    d: usize,
) -> GpuBuf<f32> {
    // Frozen level: y = q_mem @ M^T where M is frozen context.
    // d_q_mem = d_y @ M → needs M, which we don't store in cache.
    // d_embedded = d_q_mem @ W_q_mem
    //
    // For the first iteration of GPU-resident model, return zeros.
    // Frozen level gradients go to error buffers (not direct weight updates),
    // so this only affects d_embedded propagation from frozen levels.
    // The dominant gradient signal comes from active levels.
    GpuBuf::zeros(s * d)
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
fn gpu_matmul_transa_dd(
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
