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
    let loss = loss_host[0] / s as f32; // mean over tokens

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

    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::delta_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta,
                &m_initial, &mut m_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();

            // Update context: copy final M back to context buffer
            let m_final_slice = m_states.slice(s * dd, dd);
            unsafe {
                let bytes = dd * 4;
                let rc = gpu_buf_memcpy_d2d(
                    context_m.ptr() as *mut std::ffi::c_void,
                    m_final_slice.as_ptr() as *const std::ffi::c_void,
                    bytes,
                );
                assert_eq!(rc, 0);
            }

            let cache = GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states };
            (y, cache)
        }
        MemoryRuleKind::TitansLMM => {
            let mut eta = GpuBuf::zeros(s);
            let mut b_eta_host = [0.0f32];
            level_params.b_eta.copy_to_host(&mut b_eta_host);
            unsafe {
                crate::cuda_ffi::gate_compute_cuda(
                    k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_eta.as_ptr(),
                    b_eta_host[0], eta.ptr(),
                    s as i32, d as i32, 0, // sigmoid for eta
                );
            }

            // Titans needs both M and S initial states. For simplicity,
            // S_initial is zero (per-chunk reset as noted in memory).
            let s_initial_buf = GpuBuf::zeros(dd);
            let s_initial_slice = s_initial_buf.slice(0, dd);

            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut s_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);

            crate::dispatch::titans_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
                &m_initial, &s_initial_slice,
                &mut m_states, &mut s_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();

            // Update context with final M
            let m_final_slice = m_states.slice(s * dd, dd);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    context_m.ptr() as *mut std::ffi::c_void,
                    m_final_slice.as_ptr() as *const std::ffi::c_void,
                    dd * 4,
                );
                assert_eq!(rc, 0);
            }

            let cache = GpuMemoryCache::Titans {
                k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states,
            };
            (y, cache)
        }
        MemoryRuleKind::HebbianRule => {
            let mut m_states = GpuBuf::zeros((s + 1) * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::hebbian_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha,
                &m_initial, &mut m_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();

            // Update context with final M
            let m_final_slice = m_states.slice(s * dd, dd);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    context_m.ptr() as *mut std::ffi::c_void,
                    m_final_slice.as_ptr() as *const std::ffi::c_void,
                    dd * 4,
                );
                assert_eq!(rc, 0);
            }

            let cache = GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states };
            (y, cache)
        }
        _ => panic!("GPU-resident forward only supports DeltaRule, TitansLMM, HebbianRule. Got {:?}", cfg.memory_rule),
    }
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
