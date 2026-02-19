/// GPU-resident CMS forward pass.
///
/// Mirrors `cms_forward` in mag.rs but operates entirely on device pointers.
/// Only input_ids (4KB), target_ids (4KB), and loss (4 bytes) cross PCIe.
///
/// Supports matrix-based memory rules: DeltaRule, TitansLMM, HebbianRule.
/// MLP/slot-based rules (Moneta, YAAD, MEMORA, Lattice, Trellis, Atlas) fall
/// back to the CPU reference path since they lack CUDA kernels.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuMAGParams, GpuContextState};
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind};
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;

// ══════════════════════════════════════════════════════════════════════
// GpuCMSCache — forward activations on GPU (consumed by backward)
// ══════════════════════════════════════════════════════════════════════

/// Forward activation cache — all buffers on GPU.
/// Consumed by gpu_cms_backward(), then dropped (frees VRAM).
#[cfg(feature = "cuda")]
pub struct GpuCMSCache {
    // Input IDs on GPU (i32 for CUDA kernel compatibility)
    pub input_ids_gpu: GpuBuf<f32>,   // repurposed: actually stores i32 reinterpreted
    pub target_ids_gpu: GpuBuf<f32>,  // same

    // Actually use separate i32 buffers for embedding/cross-entropy kernels
    pub input_ids_i32: Vec<i32>,
    pub target_ids_i32: Vec<i32>,

    // Shared: embedded input
    pub embedded: GpuBuf<f32>,        // [s, d]

    // Attention branch (bf16 for SWA)
    pub q_f32: GpuBuf<f32>,           // [s, d] — f32 version (needed for backward projections)
    pub k_f32: GpuBuf<f32>,           // [s, d]
    pub v_f32: GpuBuf<f32>,           // [s, d]
    pub q_bf16: GpuBuf<u16>,          // [s, d] bf16
    pub k_bf16: GpuBuf<u16>,          // [s, d] bf16
    pub v_bf16: GpuBuf<u16>,          // [s, d] bf16
    pub attn_out_bf16: GpuBuf<u16>,   // [s, d] bf16
    pub attn_weights_bf16: GpuBuf<u16>, // [nh, s, ws] bf16
    pub attn_out: GpuBuf<f32>,        // [s, d] f32 (converted back)

    // Memory branch per level
    pub memory_caches: Vec<Option<GpuMemoryCache>>,
    pub y_per_level: Vec<GpuBuf<f32>>, // [s, d] per level

    // Combined + gating
    pub y_combined: GpuBuf<f32>,      // [s, d]
    pub gate: GpuBuf<f32>,            // [s, d] sigmoid(y_combined)
    pub gated_out: GpuBuf<f32>,       // [s, d] attn_out * gate

    // Post-gating
    pub projected: GpuBuf<f32>,       // [s, d]
    pub logits: GpuBuf<f32>,          // [s, v]

    // Pulse snapshot (needed by backward for level dispatch)
    pub pulse: Pulse,

    // Dimensions
    pub s: usize,
    pub d: usize,
    pub v: usize,
    pub nh: usize,
    pub hd: usize,
    pub ws: usize,
}

/// Per-level memory cache on GPU.
#[cfg(feature = "cuda")]
pub enum GpuMemoryCache {
    Delta {
        k_mem: GpuBuf<f32>,     // [s, d]
        v_mem: GpuBuf<f32>,     // [s, d]
        q_mem: GpuBuf<f32>,     // [s, d]
        alpha: GpuBuf<f32>,     // [s]
        theta: GpuBuf<f32>,     // [s]
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
    },
    Titans {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        eta: GpuBuf<f32>,
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
        s_states: GpuBuf<f32>,  // [(s+1)*d*d]
    },
    Hebbian {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        m_states: GpuBuf<f32>,
    },
    // ── Checkpointed variants (gradient checkpointing) ──────────────
    DeltaCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,  // [num_ckpt * d*d]
        checkpoint_interval: usize,
    },
    TitansCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        eta: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,
        s_checkpoints: GpuBuf<f32>,
        checkpoint_interval: usize,
    },
    HebbianCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,
        checkpoint_interval: usize,
    },
}

// ══════════════════════════════════════════════════════════════════════
// GPU-resident CMS forward
// ══════════════════════════════════════════════════════════════════════

/// GPU-resident CMS forward pass.
///
/// Only `input_ids` and `target_ids` are uploaded; loss (f32) is the only download.
/// All intermediate activations remain on GPU in the returned cache.
#[cfg(feature = "cuda")]
pub fn gpu_cms_forward(
    params: &GpuMAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut GpuContextState,
) -> (f32, GpuCMSCache) {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);
    assert!(target_ids.len() >= s);

    // Convert input_ids to i32 for CUDA kernels
    let input_ids_i32: Vec<i32> = input_ids[..s].iter().map(|&x| x as i32).collect();
    let target_ids_i32: Vec<i32> = target_ids[..s].iter().map(|&x| x as i32).collect();
    let d_input_ids = GpuBuf::<f32>::new(s);
    let d_target_ids = GpuBuf::<f32>::new(s);
    // Upload i32 data through the f32 buffer (same size)
    unsafe {
        let rc = gpu_buf_memcpy_h2d(
            d_input_ids.ptr() as *mut std::ffi::c_void,
            input_ids_i32.as_ptr() as *const std::ffi::c_void,
            s * 4,
        );
        assert_eq!(rc, 0);
        let rc = gpu_buf_memcpy_h2d(
            d_target_ids.ptr() as *mut std::ffi::c_void,
            target_ids_i32.as_ptr() as *const std::ffi::c_void,
            s * 4,
        );
        assert_eq!(rc, 0);
    }

    // ── Stage 1: Embedding gather on GPU ──────────────────────────────
    let mut embedded = GpuBuf::<f32>::zeros(s * d);
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.swa.w_embed.as_ptr(),
            d_input_ids.ptr() as *const i32,
            embedded.ptr(),
            s as i32, d as i32,
        );
    }

    // ── Stage 2a: QKV projections (cuBLAS on GPU) ─────────────────────
    let mut q_f32 = GpuBuf::zeros(s * d);
    let mut k_f32 = GpuBuf::zeros(s * d);
    let mut v_f32 = GpuBuf::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_q, &mut q_f32, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_k, &mut k_f32, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_v, &mut v_f32, s, d, d, 0.0);

    // ── Stage 3a: SWA attention (bf16 on GPU) ─────────────────────────
    let total = s * d;
    let aw_total = nh * s * ws;
    let mut q_bf16 = GpuBuf::<u16>::zeros(total);
    let mut k_bf16 = GpuBuf::<u16>::zeros(total);
    let mut v_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_out_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);

    // f32 → bf16 conversion on GPU
    unsafe {
        crate::cuda_ffi::f32_to_bf16_cuda(q_f32.as_ptr(), q_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), k_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), v_bf16.ptr(), total as i32);
    }

    // SWA forward kernel (bf16)
    crate::dispatch::swa_forward_dd(
        &q_bf16, &k_bf16, &v_bf16,
        &mut attn_out_bf16, &mut attn_weights_bf16,
        s, nh, hd, ws,
    );

    // bf16 → f32 for attn_out (needed for gating)
    let mut attn_out = GpuBuf::<f32>::zeros(total);
    unsafe {
        crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), total as i32);
    }

    // ── Stage 2b+3b: Memory branch per level ──────────────────────────
    let mut memory_caches = Vec::with_capacity(cfg.k);
    let mut y_per_level = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            // Active level: compute projections, gates, and memory update on GPU.
            let (y_level, mem_cache) = gpu_memory_forward(
                &params.levels[level], cfg, &embedded,
                &mut context.memory[level],
                s, d,
            );
            y_per_level.push(y_level);
            memory_caches.push(Some(mem_cache));
        } else {
            // Frozen level: read-only M @ q_mem on GPU.
            let y_level = gpu_memory_read_only(
                &params.levels[level], &embedded,
                &context.memory[level],
                s, d,
            );
            y_per_level.push(y_level);
            memory_caches.push(None);
        }
    }

    // ── Combine levels: y_combined = sum with 1/sqrt(k) for k>2 ───────
    let mut y_combined = GpuBuf::<f32>::zeros(s * d);
    for y_level in &y_per_level {
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, y_level.as_ptr(), y_combined.ptr(), (s * d) as i32);
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        // Scale in place: y_combined *= scale. Use saxpy trick: y = scale*y + 0 is not right.
        // Instead: zero a tmp, then saxpy(scale, y, tmp), copy back. Or just use elemwise.
        // Simpler: allocate temp, saxpy, swap. Actually simplest: download, scale, upload.
        // Best: use a scale kernel. We have saxpy: y += alpha*x. So: tmp=0, saxpy(scale, y, tmp), copy.
        // Actually: y = (scale-1)*y + y = scale*y. So: saxpy(scale-1, y, y). y += (scale-1)*y = scale*y. Yes!
        unsafe {
            crate::cuda_ffi::saxpy_cuda(scale - 1.0, y_combined.as_ptr(), y_combined.ptr(), (s * d) as i32);
        }
    }

    // ── Stage 4: Gating on GPU ────────────────────────────────────────
    let mut gate = GpuBuf::<f32>::zeros(s * d);
    let mut gated_out = GpuBuf::<f32>::zeros(s * d);
    unsafe {
        crate::cuda_ffi::sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), (s * d) as i32);
        crate::cuda_ffi::elemwise_mul_cuda(attn_out.as_ptr(), gate.as_ptr(), gated_out.ptr(), (s * d) as i32);
    }

    // ── Stage 5: Output projection (cuBLAS on GPU) ────────────────────
    let mut projected = GpuBuf::<f32>::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(&gated_out, &params.swa.w_o, &mut projected, s, d, d, 0.0);

    // ── Stage 6: Unembed (cuBLAS on GPU) ──────────────────────────────
    let mut logits = GpuBuf::<f32>::zeros(s * v);
    crate::dispatch::cublas_matmul_dd(&projected, &params.swa.w_unembed, &mut logits, s, d, v, 0.0);

    // ── Stage 7: Cross-entropy loss (GPU → scalar D2H) ────────────────
    let mut loss_gpu = GpuBuf::<f32>::zeros(1);
    unsafe {
        crate::cuda_ffi::cross_entropy_forward_cuda(
            logits.as_ptr(),
            d_target_ids.ptr() as *const i32,
            loss_gpu.ptr(),
            s as i32, v as i32,
        );
    }
    crate::dispatch::cuda_sync();

    // Download scalar loss (4 bytes D2H)
    let mut loss_host = [0.0f32; 1];
    loss_gpu.copy_to_host(&mut loss_host);
    // Mean over VALID tokens only (masked targets with id < 0 or >= vocab are
    // skipped by the cross-entropy kernel, so we must divide by the actual count).
    let valid_count = target_ids_i32.iter()
        .filter(|&&t| t >= 0 && (t as usize) < v)
        .count() as f32;
    let loss = if valid_count > 0.0 { loss_host[0] / valid_count } else { 0.0 };

    let cache = GpuCMSCache {
        input_ids_gpu: d_input_ids,
        target_ids_gpu: d_target_ids,
        input_ids_i32,
        target_ids_i32,
        embedded,
        q_f32, k_f32, v_f32,
        q_bf16, k_bf16, v_bf16,
        attn_out_bf16, attn_weights_bf16,
        attn_out,
        memory_caches,
        y_per_level,
        y_combined,
        gate, gated_out,
        projected, logits,
        pulse: pulse.clone(),
        s, d, v, nh, hd, ws,
    };

    (loss, cache)
}

// ══════════════════════════════════════════════════════════════════════
// Memory forward helpers (GPU-resident)
// ══════════════════════════════════════════════════════════════════════

/// Compute memory projections + gates + inner loop for an active level, all on GPU.
#[cfg(feature = "cuda")]
fn gpu_memory_forward(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    cfg: &MAGConfig,
    embedded: &GpuBuf<f32>,
    context_m: &mut GpuBuf<f32>,   // [d*d] — updated with final M
    s: usize,
    d: usize,
) -> (GpuBuf<f32>, GpuMemoryCache) {
    let dd = d * d;

    // Memory projections: k_mem, v_mem, q_mem = embedded @ W^T
    let mut k_mem = GpuBuf::zeros(s * d);
    let mut v_mem = GpuBuf::zeros(s * d);
    let mut q_mem = GpuBuf::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_k_mem, &mut k_mem, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_v_mem, &mut v_mem, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut q_mem, s, d, d, 0.0);

    // Compute per-token gates: alpha = sigmoid(dot(concat(k,v), w_alpha) + b_alpha)
    let mut alpha = GpuBuf::zeros(s);
    let mut theta = GpuBuf::zeros(s);
    let mut b_alpha_host = [0.0f32];
    let mut b_theta_host = [0.0f32];
    level_params.b_alpha.copy_to_host(&mut b_alpha_host);
    level_params.b_theta.copy_to_host(&mut b_theta_host);

    unsafe {
        // alpha = sigmoid(gate_compute(k_mem, v_mem, w_alpha, b_alpha))
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_alpha.as_ptr(),
            b_alpha_host[0], alpha.ptr(),
            s as i32, d as i32, 0, // 0=sigmoid
        );
        // theta = softplus(gate_compute(k_mem, v_mem, w_theta, b_theta))
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_theta.as_ptr(),
            b_theta_host[0], theta.ptr(),
            s as i32, d as i32, 1, // 1=softplus
        );
    }

    let m_initial = context_m.slice(0, dd);

    match (cfg.checkpoint_interval, cfg.memory_rule) {
        // ── Full-trajectory paths (checkpoint_interval=None, current behavior) ──
        (None, MemoryRuleKind::DeltaRule) => {
            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::delta_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta,
                &m_initial, &mut m_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_states, context_m, s * dd, dd);
            (y, GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states })
        }
        (None, MemoryRuleKind::TitansLMM) => {
            let eta = compute_eta(level_params, &k_mem, &v_mem, s, d);
            let s_initial_buf = GpuBuf::zeros(dd);
            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut s_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::titans_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
                &m_initial, &s_initial_buf.slice(0, dd),
                &mut m_states, &mut s_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_states, context_m, s * dd, dd);
            (y, GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states })
        }
        (None, MemoryRuleKind::HebbianRule) => {
            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::hebbian_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha,
                &m_initial, &mut m_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_states, context_m, s * dd, dd);
            (y, GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states })
        }
        // ── Checkpointed paths (checkpoint_interval=Some(c)) ──
        (Some(c), MemoryRuleKind::DeltaRule) => {
            let num_ckpt = checkpoint_count(s, c);
            let mut m_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::delta_forward_dd_ckpt(
                &k_mem, &v_mem, &q_mem, &alpha, &theta,
                &m_initial, &mut m_checkpoints, &mut y, s, d, c,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            (y, GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval: c })
        }
        (Some(c), MemoryRuleKind::TitansLMM) => {
            let eta = compute_eta(level_params, &k_mem, &v_mem, s, d);
            let s_initial_buf = GpuBuf::zeros(dd);
            let num_ckpt = checkpoint_count(s, c);
            let mut m_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut s_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::titans_forward_dd_ckpt(
                &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
                &m_initial, &s_initial_buf.slice(0, dd),
                &mut m_checkpoints, &mut s_checkpoints, &mut y, s, d, c,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            (y, GpuMemoryCache::TitansCkpt { k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, checkpoint_interval: c })
        }
        (Some(c), MemoryRuleKind::HebbianRule) => {
            let num_ckpt = checkpoint_count(s, c);
            let mut m_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::hebbian_forward_dd_ckpt(
                &k_mem, &v_mem, &q_mem, &alpha,
                &m_initial, &mut m_checkpoints, &mut y, s, d, c,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            (y, GpuMemoryCache::HebbianCkpt { k_mem, v_mem, q_mem, alpha, m_checkpoints, checkpoint_interval: c })
        }
        _ => panic!("GPU-resident forward only supports DeltaRule, TitansLMM, HebbianRule. Got {:?}", cfg.memory_rule),
    }
}

/// Copy final M state from states buffer to context (D2D).
#[cfg(feature = "cuda")]
#[inline]
fn copy_final_m(states: &GpuBuf<f32>, context_m: &mut GpuBuf<f32>, offset: usize, dd: usize) {
    let m_final = states.slice(offset, dd);
    unsafe {
        let rc = gpu_buf_memcpy_d2d(
            context_m.ptr() as *mut std::ffi::c_void,
            m_final.as_ptr() as *const std::ffi::c_void,
            dd * 4,
        );
        assert_eq!(rc, 0);
    }
}

/// Compute eta gate for Titans (shared between full and checkpointed paths).
#[cfg(feature = "cuda")]
fn compute_eta(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>,
    s: usize, d: usize,
) -> GpuBuf<f32> {
    let mut eta = GpuBuf::zeros(s);
    let mut b_eta_host = [0.0f32];
    level_params.b_eta.copy_to_host(&mut b_eta_host);
    unsafe {
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_eta.as_ptr(),
            b_eta_host[0], eta.ptr(),
            s as i32, d as i32, 0,
        );
    }
    eta
}

/// Number of checkpoints stored: M_0, then one per C-step boundary, plus final state.
/// Not gated by `cuda` feature — pure arithmetic, usable in tests.
pub fn checkpoint_count(seq_len: usize, c: usize) -> usize {
    // Checkpoints at t=0 (initial), then at each C boundary or final step.
    // The kernel increments ckpt_idx after storing initial, then for each
    // t where (t+1) % C == 0 OR t+1 == seq_len.
    let mut count = 1; // initial M_0
    for t in 0..seq_len {
        if ((t + 1) % c == 0) || (t + 1 == seq_len) {
            count += 1;
        }
    }
    count
}


/// Frozen level: read-only y = M @ q_mem (all on GPU).
#[cfg(feature = "cuda")]
fn gpu_memory_read_only(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    embedded: &GpuBuf<f32>,
    context_m: &GpuBuf<f32>,   // [d*d] — read only
    s: usize,
    d: usize,
) -> GpuBuf<f32> {
    // q_mem = embedded @ W_q_mem^T
    let mut q_mem = GpuBuf::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut q_mem, s, d, d, 0.0);

    // y = M @ q_mem for each token (M is [d, d], q_mem[t] is [d])
    // Batch: y[s, d] = q_mem[s, d] @ M^T (since y[t] = M @ q[t])
    // Actually: y_t = M @ q_t. In matrix form: Y^T = M @ Q^T, so Y = Q @ M^T
    let mut y = GpuBuf::zeros(s * d);
    // M is [d,d], Q is [s,d]. Y[s,d] = Q[s,d] @ M^T[d,d]
    // Use matmul_transb_dd: C = A @ B^T where B is [n,k] = M is [d,d]
    crate::dispatch::cublas_matmul_transb_dd(&q_mem, context_m, &mut y, s, d, d, 0.0);
    y
}

// ══════════════════════════════════════════════════════════════════════
// Helper: raw memcpy wrappers (used for D2D and H2D of i32 data)
// ══════════════════════════════════════════════════════════════════════

/// Raw cudaMemcpy H2D wrapper for gpu_forward internal use.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn gpu_buf_memcpy_h2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
) -> i32 {
    extern "C" {
        fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                      count: usize, kind: i32) -> i32;
    }
    cudaMemcpy(dst, src, bytes, 1) // 1 = H2D
}

// ══════════════════════════════════════════════════════════════════════
// GpuKVCache — KV cache for autoregressive decode
// ══════════════════════════════════════════════════════════════════════

/// GPU-resident KV cache for autoregressive generation.
///
/// Stores K and V projections in bf16 on device. Filled during prefill,
/// extended one token at a time during decode. Avoids re-projecting
/// the entire sequence each step.
#[cfg(feature = "cuda")]
pub struct GpuKVCache {
    /// K cache: [max_len, d] bf16 on device.
    pub k_cache_bf16: GpuBuf<u16>,
    /// V cache: [max_len, d] bf16 on device.
    pub v_cache_bf16: GpuBuf<u16>,
    /// Current number of filled positions.
    pub len: usize,
    /// Maximum cache capacity.
    pub max_len: usize,
    /// Model dimension (num_heads * head_dim).
    pub d: usize,
    /// Persistent scratch buffer for f32→bf16 conversion (avoids per-token cudaMalloc).
    scratch_k_bf16: GpuBuf<u16>,
    scratch_v_bf16: GpuBuf<u16>,
}

#[cfg(feature = "cuda")]
impl GpuKVCache {
    /// Allocate a new empty KV cache with persistent scratch buffers.
    ///
    /// `scratch_tokens` is the max number of tokens appended in a single call
    /// (seq_len for prefill, 1 for decode). Scratch buffers are allocated once
    /// and reused to avoid per-token cudaMalloc overhead.
    pub fn new(max_len: usize, d: usize, scratch_tokens: usize) -> Self {
        let scratch_size = scratch_tokens * d;
        GpuKVCache {
            k_cache_bf16: GpuBuf::<u16>::zeros(max_len * d),
            v_cache_bf16: GpuBuf::<u16>::zeros(max_len * d),
            len: 0,
            max_len,
            d,
            scratch_k_bf16: GpuBuf::<u16>::zeros(scratch_size),
            scratch_v_bf16: GpuBuf::<u16>::zeros(scratch_size),
        }
    }

    /// Append n_tokens of K/V data (f32 on GPU) to the cache.
    /// Converts f32 → bf16 using persistent scratch buffers and copies into cache.
    pub fn append_f32(&mut self, k_f32: &GpuBuf<f32>, v_f32: &GpuBuf<f32>, n_tokens: usize) {
        assert!(self.len + n_tokens <= self.max_len,
            "KV cache overflow: {} + {} > {}", self.len, n_tokens, self.max_len);
        assert!(n_tokens * self.d <= self.scratch_k_bf16.len(),
            "append_f32: n_tokens*d={} exceeds scratch buffer size {}", n_tokens * self.d, self.scratch_k_bf16.len());

        let total = n_tokens * self.d;

        // Convert f32 → bf16 into pre-allocated scratch buffers (no cudaMalloc)
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), self.scratch_k_bf16.ptr(), total as i32);
            crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), self.scratch_v_bf16.ptr(), total as i32);
        }

        // D2D copy from scratch into cache at offset
        let offset_bytes = self.len * self.d * 2; // u16 = 2 bytes
        let copy_bytes = total * 2;
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                (self.k_cache_bf16.ptr() as *mut u8).add(offset_bytes) as *mut std::ffi::c_void,
                self.scratch_k_bf16.as_ptr() as *const std::ffi::c_void,
                copy_bytes,
            );
            assert_eq!(rc, 0);
            let rc = gpu_buf_memcpy_d2d(
                (self.v_cache_bf16.ptr() as *mut u8).add(offset_bytes) as *mut std::ffi::c_void,
                self.scratch_v_bf16.as_ptr() as *const std::ffi::c_void,
                copy_bytes,
            );
            assert_eq!(rc, 0);
        }

        self.len += n_tokens;
    }

    /// Reset the cache (set len=0, no dealloc).
    pub fn reset(&mut self) {
        self.len = 0;
    }
}

// ══════════════════════════════════════════════════════════════════════
// GPU prefill forward — process full prompt, populate KV cache
// ══════════════════════════════════════════════════════════════════════

/// Prefill: run full forward on prompt, populate KV cache, return last-position logits.
///
/// This is the "prompt processing" phase of cached generation.
/// Runs the same stages as gpu_cms_forward but:
/// - Populates a KV cache with the K/V projections
/// - Skips cross-entropy loss
/// - Downloads only last-position logits (vocab-sized, not seq_len*vocab)
#[cfg(feature = "cuda")]
pub fn gpu_prefill_forward(
    params: &GpuMAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    pulse: &Pulse,
    context: &mut GpuContextState,
) -> (Vec<f32>, GpuKVCache) {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);

    // Upload input_ids
    let input_ids_i32: Vec<i32> = input_ids[..s].iter().map(|&x| x as i32).collect();
    let d_input_ids = GpuBuf::<f32>::new(s);
    unsafe {
        let rc = gpu_buf_memcpy_h2d(
            d_input_ids.ptr() as *mut std::ffi::c_void,
            input_ids_i32.as_ptr() as *const std::ffi::c_void,
            s * 4,
        );
        assert_eq!(rc, 0);
    }

    // ── Stage 1: Embedding gather ──────────────────────────────────
    let mut embedded = GpuBuf::<f32>::zeros(s * d);
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.swa.w_embed.as_ptr(),
            d_input_ids.ptr() as *const i32,
            embedded.ptr(),
            s as i32, d as i32,
        );
    }

    // ── Stage 2a: QKV projections ──────────────────────────────────
    let mut q_f32 = GpuBuf::zeros(s * d);
    let mut k_f32 = GpuBuf::zeros(s * d);
    let mut v_f32 = GpuBuf::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_q, &mut q_f32, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_k, &mut k_f32, s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&embedded, &params.swa.w_v, &mut v_f32, s, d, d, 0.0);

    // ── Populate KV cache ──────────────────────────────────────────
    let max_cache_len = cfg.swa.seq_len + 2048; // prompt + up to 2048 decode tokens
    let mut kv_cache = GpuKVCache::new(max_cache_len, d, s); // scratch sized for prefill (s tokens)
    kv_cache.append_f32(&k_f32, &v_f32, s);

    // ── Stage 3a: SWA attention (bf16) ─────────────────────────────
    let total = s * d;
    let aw_total = nh * s * ws;
    let mut q_bf16 = GpuBuf::<u16>::zeros(total);
    let mut k_bf16 = GpuBuf::<u16>::zeros(total);
    let mut v_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_out_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);

    unsafe {
        crate::cuda_ffi::f32_to_bf16_cuda(q_f32.as_ptr(), q_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), k_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), v_bf16.ptr(), total as i32);
    }

    crate::dispatch::swa_forward_dd(
        &q_bf16, &k_bf16, &v_bf16,
        &mut attn_out_bf16, &mut attn_weights_bf16,
        s, nh, hd, ws,
    );

    let mut attn_out = GpuBuf::<f32>::zeros(total);
    unsafe {
        crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), total as i32);
    }

    // ── Stage 2b+3b: Memory branch per level ───────────────────────
    let mut y_per_level = Vec::with_capacity(cfg.k);
    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            let (y_level, _mem_cache) = gpu_memory_forward(
                &params.levels[level], cfg, &embedded,
                &mut context.memory[level],
                s, d,
            );
            y_per_level.push(y_level);
        } else {
            let y_level = gpu_memory_read_only(
                &params.levels[level], &embedded,
                &context.memory[level],
                s, d,
            );
            y_per_level.push(y_level);
        }
    }

    // ── Combine levels ─────────────────────────────────────────────
    let mut y_combined = GpuBuf::<f32>::zeros(s * d);
    for y_level in &y_per_level {
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, y_level.as_ptr(), y_combined.ptr(), (s * d) as i32);
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(scale - 1.0, y_combined.as_ptr(), y_combined.ptr(), (s * d) as i32);
        }
    }

    // ── Stage 4: Gating ────────────────────────────────────────────
    let mut gate = GpuBuf::<f32>::zeros(s * d);
    let mut gated_out = GpuBuf::<f32>::zeros(s * d);
    unsafe {
        crate::cuda_ffi::sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), (s * d) as i32);
        crate::cuda_ffi::elemwise_mul_cuda(attn_out.as_ptr(), gate.as_ptr(), gated_out.ptr(), (s * d) as i32);
    }

    // ── Stage 5: Output projection ─────────────────────────────────
    let mut projected = GpuBuf::<f32>::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(&gated_out, &params.swa.w_o, &mut projected, s, d, d, 0.0);

    // ── Stage 6: Unembed (only last position) ──────────────────────
    // Extract last position projected[s-1] as [1, d] and unembed to [1, v]
    let mut last_projected = GpuBuf::<f32>::zeros(d);
    unsafe {
        let rc = gpu_buf_memcpy_d2d(
            last_projected.ptr() as *mut std::ffi::c_void,
            (projected.as_ptr() as *const u8).add((s - 1) * d * 4) as *const std::ffi::c_void,
            d * 4,
        );
        assert_eq!(rc, 0);
    }
    let mut last_logits_gpu = GpuBuf::<f32>::zeros(v);
    crate::dispatch::cublas_matmul_dd(&last_projected, &params.swa.w_unembed, &mut last_logits_gpu, 1, d, v, 0.0);
    crate::dispatch::cuda_sync();

    // Download last-position logits
    let mut last_logits = vec![0.0f32; v];
    last_logits_gpu.copy_to_host(&mut last_logits);

    (last_logits, kv_cache)
}

// ══════════════════════════════════════════════════════════════════════
// DecodeWorkspace — pre-allocated GPU buffers for single-token decode
// ══════════════════════════════════════════════════════════════════════

/// Pre-allocated GPU buffers for single-token decode, avoiding per-token cudaMalloc.
/// Created once during prefill, reused for every decode step.
#[cfg(feature = "cuda")]
pub struct DecodeWorkspace {
    pub d_input: GpuBuf<f32>,       // [1] — token ID upload
    pub embedded: GpuBuf<f32>,      // [d]
    pub q_f32: GpuBuf<f32>,         // [d]
    pub k_f32: GpuBuf<f32>,         // [d]
    pub v_f32: GpuBuf<f32>,         // [d]
    pub q_bf16: GpuBuf<u16>,        // [d]
    pub attn_out_bf16: GpuBuf<u16>, // [d]
    pub attn_out: GpuBuf<f32>,      // [d]
    pub y_combined: GpuBuf<f32>,    // [d]
    pub gate: GpuBuf<f32>,          // [d]
    pub gated_out: GpuBuf<f32>,     // [d]
    pub projected: GpuBuf<f32>,     // [d]
    pub logits_gpu: GpuBuf<f32>,    // [v]
}

#[cfg(feature = "cuda")]
impl DecodeWorkspace {
    /// Allocate all workspace buffers once.
    pub fn new(d: usize, v: usize) -> Self {
        DecodeWorkspace {
            d_input: GpuBuf::<f32>::new(1),
            embedded: GpuBuf::<f32>::zeros(d),
            q_f32: GpuBuf::zeros(d),
            k_f32: GpuBuf::zeros(d),
            v_f32: GpuBuf::zeros(d),
            q_bf16: GpuBuf::<u16>::zeros(d),
            attn_out_bf16: GpuBuf::<u16>::zeros(d),
            attn_out: GpuBuf::<f32>::zeros(d),
            y_combined: GpuBuf::<f32>::zeros(d),
            gate: GpuBuf::<f32>::zeros(d),
            gated_out: GpuBuf::<f32>::zeros(d),
            projected: GpuBuf::<f32>::zeros(d),
            logits_gpu: GpuBuf::<f32>::zeros(v),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GPU single-token forward — decode one token using KV cache
// ══════════════════════════════════════════════════════════════════════

/// Decode one token using KV cache and pre-allocated workspace. Returns logits [vocab].
///
/// Per-token path (all s=1):
/// 1. Embed 1 token
/// 2. QKV project (3 cuBLAS: [1,d] @ [d,d]^T)
/// 3. Append K, V to cache
/// 4. SWA single-token attention (new kernel)
/// 5. Memory branch per level (existing kernels with s=1)
/// 6. Combine levels, gate, output project, unembed
/// 7. Download logits [vocab]
#[cfg(feature = "cuda")]
pub fn gpu_single_token_forward(
    params: &GpuMAGParams,
    cfg: &MAGConfig,
    token_id: usize,
    pulse: &Pulse,
    context: &mut GpuContextState,
    kv_cache: &mut GpuKVCache,
    ws: &mut DecodeWorkspace,
) -> Vec<f32> {
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let window_size = cfg.swa.window_size;

    assert!(token_id < v, "token_id {} >= vocab_size {}", token_id, v);
    assert!(kv_cache.len > 0, "KV cache must be populated via prefill first");
    assert!(kv_cache.len < kv_cache.max_len, "KV cache full: {} >= {}", kv_cache.len, kv_cache.max_len);

    // ── Stage 1: Embed 1 token ─────────────────────────────────────
    let input_i32 = [token_id as i32];
    unsafe {
        let rc = gpu_buf_memcpy_h2d(
            ws.d_input.ptr() as *mut std::ffi::c_void,
            input_i32.as_ptr() as *const std::ffi::c_void,
            4,
        );
        assert_eq!(rc, 0);
        crate::cuda_ffi::embedding_gather_cuda(
            params.swa.w_embed.as_ptr(),
            ws.d_input.ptr() as *const i32,
            ws.embedded.ptr(),
            1, d as i32,
        );
    }

    // ── Stage 2a: QKV projections [1,d] @ [d,d]^T ─────────────────
    crate::dispatch::cublas_matmul_transb_dd(&ws.embedded, &params.swa.w_q, &mut ws.q_f32, 1, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&ws.embedded, &params.swa.w_k, &mut ws.k_f32, 1, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(&ws.embedded, &params.swa.w_v, &mut ws.v_f32, 1, d, d, 0.0);

    // ── Stage 3a-prep: Append K, V to cache ────────────────────────
    kv_cache.append_f32(&ws.k_f32, &ws.v_f32, 1);

    // ── Stage 3a: SWA single-token attention ───────────────────────
    unsafe {
        crate::cuda_ffi::f32_to_bf16_cuda(ws.q_f32.as_ptr(), ws.q_bf16.ptr(), d as i32);
    }

    crate::dispatch::swa_single_token_dd(
        &ws.q_bf16, &kv_cache.k_cache_bf16, &kv_cache.v_cache_bf16,
        &mut ws.attn_out_bf16,
        kv_cache.len, nh, hd, window_size,
    );

    unsafe {
        crate::cuda_ffi::bf16_to_f32_cuda(ws.attn_out_bf16.as_ptr(), ws.attn_out.ptr(), d as i32);
    }

    // ── Stage 2b+3b: Memory branch per level (s=1) ────────────────
    // Note: memory buffers are allocated per-level by gpu_memory_forward (s=1 is small).
    let mut y_per_level = Vec::with_capacity(cfg.k);
    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            let (y_level, _mem_cache) = gpu_memory_forward(
                &params.levels[level], cfg, &ws.embedded,
                &mut context.memory[level],
                1, d,
            );
            y_per_level.push(y_level);
        } else {
            let y_level = gpu_memory_read_only(
                &params.levels[level], &ws.embedded,
                &context.memory[level],
                1, d,
            );
            y_per_level.push(y_level);
        }
    }

    // ── Combine levels ─────────────────────────────────────────────
    ws.y_combined.zero();
    for y_level in &y_per_level {
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, y_level.as_ptr(), ws.y_combined.ptr(), d as i32);
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(scale - 1.0, ws.y_combined.as_ptr(), ws.y_combined.ptr(), d as i32);
        }
    }

    // ── Stage 4: Gating ────────────────────────────────────────────
    unsafe {
        crate::cuda_ffi::sigmoid_cuda(ws.y_combined.as_ptr(), ws.gate.ptr(), d as i32);
        crate::cuda_ffi::elemwise_mul_cuda(ws.attn_out.as_ptr(), ws.gate.as_ptr(), ws.gated_out.ptr(), d as i32);
    }

    // ── Stage 5: Output projection ─────────────────────────────────
    crate::dispatch::cublas_matmul_transb_dd(&ws.gated_out, &params.swa.w_o, &mut ws.projected, 1, d, d, 0.0);

    // ── Stage 6: Unembed ───────────────────────────────────────────
    crate::dispatch::cublas_matmul_dd(&ws.projected, &params.swa.w_unembed, &mut ws.logits_gpu, 1, d, v, 0.0);
    crate::dispatch::cuda_sync();

    // Download logits
    let mut logits = vec![0.0f32; v];
    ws.logits_gpu.copy_to_host(&mut logits);
    logits
}

// ══════════════════════════════════════════════════════════════════════
// Helper: raw memcpy wrappers (used for D2D and H2D of i32 data)
// ══════════════════════════════════════════════════════════════════════

/// Raw cudaMemcpy D2D wrapper.
#[cfg(feature = "cuda")]
pub(crate) unsafe fn gpu_buf_memcpy_d2d(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
) -> i32 {
    extern "C" {
        fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                      count: usize, kind: i32) -> i32;
    }
    cudaMemcpy(dst, src, bytes, 3) // 3 = D2D
}
