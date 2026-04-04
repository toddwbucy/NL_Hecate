/// GPU-resident model parameters and context state.
///
/// All weight matrices live on GPU. Created once at init via `from_host()`,
/// downloaded for checkpoints via `to_host()`. No PCIe traffic during
/// forward/backward/update.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::model::{SWAParams, MAGParams, MAGConfig, MemoryLevelParams};

// ══════════════════════════════════════════════════════════════════════
// GpuSWAParams — attention branch weights on GPU
// ══════════════════════════════════════════════════════════════════════

/// SWA projection weights resident on GPU.
#[cfg(feature = "cuda")]
pub struct GpuSWAParams {
    pub w_embed: GpuBuf<f32>,   // [vocab, d]
    pub w_q: GpuBuf<f32>,       // [d, d]
    pub w_k: GpuBuf<f32>,       // [d, d]
    pub w_v: GpuBuf<f32>,       // [d, d]
    pub w_o: GpuBuf<f32>,       // [d, d]
    pub w_unembed: GpuBuf<f32>, // [d, vocab]
    pub ln_attn_gamma: GpuBuf<f32>,  // [d]
    pub ln_attn_beta: GpuBuf<f32>,   // [d]
    pub ln_mem_gamma: GpuBuf<f32>,   // [d]
    pub ln_mem_beta: GpuBuf<f32>,    // [d]
}

#[cfg(feature = "cuda")]
impl GpuSWAParams {
    pub fn from_host(host: &SWAParams) -> Self {
        GpuSWAParams {
            w_embed: GpuBuf::from_host(&host.w_embed),
            w_q: GpuBuf::from_host(&host.w_q),
            w_k: GpuBuf::from_host(&host.w_k),
            w_v: GpuBuf::from_host(&host.w_v),
            w_o: GpuBuf::from_host(&host.w_o),
            w_unembed: GpuBuf::from_host(&host.w_unembed),
            ln_attn_gamma: GpuBuf::from_host(&host.ln_attn_gamma),
            ln_attn_beta: GpuBuf::from_host(&host.ln_attn_beta),
            ln_mem_gamma: GpuBuf::from_host(&host.ln_mem_gamma),
            ln_mem_beta: GpuBuf::from_host(&host.ln_mem_beta),
        }
    }

    pub fn to_host(&self, cfg_d: usize, cfg_v: usize) -> SWAParams {
        let mut p = SWAParams {
            w_embed: vec![0.0f32; cfg_v * cfg_d],
            w_q: vec![0.0f32; cfg_d * cfg_d],
            w_k: vec![0.0f32; cfg_d * cfg_d],
            w_v: vec![0.0f32; cfg_d * cfg_d],
            w_o: vec![0.0f32; cfg_d * cfg_d],
            w_unembed: vec![0.0f32; cfg_d * cfg_v],
            ln_attn_gamma: vec![0.0f32; cfg_d],
            ln_attn_beta: vec![0.0f32; cfg_d],
            ln_mem_gamma: vec![0.0f32; cfg_d],
            ln_mem_beta: vec![0.0f32; cfg_d],
        };
        self.w_embed.copy_to_host(&mut p.w_embed);
        self.w_q.copy_to_host(&mut p.w_q);
        self.w_k.copy_to_host(&mut p.w_k);
        self.w_v.copy_to_host(&mut p.w_v);
        self.w_o.copy_to_host(&mut p.w_o);
        self.w_unembed.copy_to_host(&mut p.w_unembed);
        self.ln_attn_gamma.copy_to_host(&mut p.ln_attn_gamma);
        self.ln_attn_beta.copy_to_host(&mut p.ln_attn_beta);
        self.ln_mem_gamma.copy_to_host(&mut p.ln_mem_gamma);
        self.ln_mem_beta.copy_to_host(&mut p.ln_mem_beta);
        p
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuMemoryLevelParams — per-level memory weights on GPU
// ══════════════════════════════════════════════════════════════════════

/// Per-level memory projections and gates resident on GPU.
#[cfg(feature = "cuda")]
pub struct GpuMemoryLevelParams {
    pub w_k_mem: GpuBuf<f32>,   // [d, d]
    pub w_v_mem: GpuBuf<f32>,   // [d, d]
    pub w_q_mem: GpuBuf<f32>,   // [d, d]
    pub w_alpha: GpuBuf<f32>,   // [2*d]
    pub b_alpha: GpuBuf<f32>,   // [1]
    pub w_theta: GpuBuf<f32>,   // [2*d]
    pub b_theta: GpuBuf<f32>,   // [1]
    pub w_eta: GpuBuf<f32>,     // [2*d]
    pub b_eta: GpuBuf<f32>,     // [1]
    pub w_omega: GpuBuf<f32>,   // [d, 2*d] for Atlas, dummy(1) for others
    pub has_omega: bool,
    // Frequency gate (empty for Fixed schedule — use len=1 dummy)
    pub w_freq: GpuBuf<f32>,
    pub b_freq: GpuBuf<f32>,
    pub has_freq: bool,
    // Conv1D weights (empty when kernel_size=0 — use len=1 dummy)
    pub w_k_conv: GpuBuf<f32>,
    pub b_k_conv: GpuBuf<f32>,
    pub w_q_conv: GpuBuf<f32>,
    pub b_q_conv: GpuBuf<f32>,
    pub has_conv: bool,
    // SwiGluMlp projections. zeros(1) dummy for all non-SwiGLU levels.
    pub gate_proj: GpuBuf<f32>,   // [inter × d]
    pub up_proj:   GpuBuf<f32>,   // [inter × d]
    pub down_proj: GpuBuf<f32>,   // [d × inter]
    pub has_mlp: bool,
    // RandomFourier feature map weights (CPU-side; phi computed before CUDA kernel launch).
    // Empty Vec when feature_map == Identity.
    pub w_rand_cpu: Vec<f32>,     // [d * d]
    pub b_rand_cpu: Vec<f32>,     // [d]
    pub has_fm: bool,
}

#[cfg(feature = "cuda")]
impl GpuMemoryLevelParams {
    pub fn from_host(host: &MemoryLevelParams) -> Self {
        let has_freq = !host.w_freq.is_empty();
        // Allocate at least 1 element for freq bufs to avoid zero-length cudaMalloc
        let w_freq = if has_freq {
            GpuBuf::from_host(&host.w_freq)
        } else {
            GpuBuf::zeros(1)
        };
        let b_freq = if has_freq {
            GpuBuf::from_host(&host.b_freq)
        } else {
            GpuBuf::zeros(1)
        };

        let has_conv = !host.w_k_conv.is_empty();
        let w_k_conv = if has_conv { GpuBuf::from_host(&host.w_k_conv) } else { GpuBuf::zeros(1) };
        let b_k_conv = if has_conv { GpuBuf::from_host(&host.b_k_conv) } else { GpuBuf::zeros(1) };
        let w_q_conv = if has_conv { GpuBuf::from_host(&host.w_q_conv) } else { GpuBuf::zeros(1) };
        let b_q_conv = if has_conv { GpuBuf::from_host(&host.b_q_conv) } else { GpuBuf::zeros(1) };

        let has_mlp   = !host.gate_proj.is_empty();
        let gate_proj = if has_mlp { GpuBuf::from_host(&host.gate_proj) } else { GpuBuf::zeros(1) };
        let up_proj   = if has_mlp { GpuBuf::from_host(&host.up_proj)   } else { GpuBuf::zeros(1) };
        let down_proj = if has_mlp { GpuBuf::from_host(&host.down_proj) } else { GpuBuf::zeros(1) };

        let has_w_rand = !host.w_rand.is_empty();
        let has_b_rand = !host.b_rand.is_empty();
        assert_eq!(
            has_w_rand, has_b_rand,
            "GpuMemoryLevelParams::from_host: w_rand (len={}) and b_rand (len={}) must both be \
             non-empty or both be empty — mismatched FM pair indicates a corrupted host params",
            host.w_rand.len(), host.b_rand.len()
        );
        let has_fm = has_w_rand && has_b_rand;

        GpuMemoryLevelParams {
            w_k_mem: GpuBuf::from_host(host.w_k_mem.master()),
            w_v_mem: GpuBuf::from_host(host.w_v_mem.master()),
            w_q_mem: GpuBuf::from_host(host.w_q_mem.master()),
            w_alpha: GpuBuf::from_host(&host.w_alpha),
            b_alpha: GpuBuf::from_host(&host.b_alpha),
            w_theta: GpuBuf::from_host(&host.w_theta),
            b_theta: GpuBuf::from_host(&host.b_theta),
            w_eta: GpuBuf::from_host(&host.w_eta),
            b_eta: GpuBuf::from_host(&host.b_eta),
            w_omega: if !host.w_omega.is_empty() { GpuBuf::from_host(&host.w_omega) } else { GpuBuf::zeros(1) },
            has_omega: !host.w_omega.is_empty(),
            w_freq,
            b_freq,
            has_freq,
            w_k_conv, b_k_conv, w_q_conv, b_q_conv,
            has_conv,
            gate_proj, up_proj, down_proj,
            has_mlp,
            w_rand_cpu: host.w_rand.clone(),
            b_rand_cpu: host.b_rand.clone(),
            has_fm,
        }
    }

    pub fn to_host(&self, d: usize) -> MemoryLevelParams {
        let mut p = MemoryLevelParams::zeros_like(d);
        self.w_k_mem.copy_to_host(p.w_k_mem.master_mut()); p.w_k_mem.sync_from_master();
        self.w_v_mem.copy_to_host(p.w_v_mem.master_mut()); p.w_v_mem.sync_from_master();
        self.w_q_mem.copy_to_host(p.w_q_mem.master_mut()); p.w_q_mem.sync_from_master();
        self.w_alpha.copy_to_host(&mut p.w_alpha);
        self.b_alpha.copy_to_host(&mut p.b_alpha);
        self.w_theta.copy_to_host(&mut p.w_theta);
        self.b_theta.copy_to_host(&mut p.b_theta);
        self.w_eta.copy_to_host(&mut p.w_eta);
        self.b_eta.copy_to_host(&mut p.b_eta);
        if self.has_omega {
            p.w_omega = vec![0.0f32; self.w_omega.len()];
            self.w_omega.copy_to_host(&mut p.w_omega);
        }
        if self.has_freq {
            p.w_freq = vec![0.0f32; self.w_freq.len()];
            p.b_freq = vec![0.0f32; self.b_freq.len()];
            self.w_freq.copy_to_host(&mut p.w_freq);
            self.b_freq.copy_to_host(&mut p.b_freq);
        }
        if self.has_conv {
            p.w_k_conv = vec![0.0f32; self.w_k_conv.len()];
            p.b_k_conv = vec![0.0f32; self.b_k_conv.len()];
            p.w_q_conv = vec![0.0f32; self.w_q_conv.len()];
            p.b_q_conv = vec![0.0f32; self.b_q_conv.len()];
            self.w_k_conv.copy_to_host(&mut p.w_k_conv);
            self.b_k_conv.copy_to_host(&mut p.b_k_conv);
            self.w_q_conv.copy_to_host(&mut p.w_q_conv);
            self.b_q_conv.copy_to_host(&mut p.b_q_conv);
        }
        if self.has_mlp {
            p.gate_proj = vec![0.0f32; self.gate_proj.len()];
            p.up_proj   = vec![0.0f32; self.up_proj.len()];
            p.down_proj = vec![0.0f32; self.down_proj.len()];
            self.gate_proj.copy_to_host(&mut p.gate_proj);
            self.up_proj.copy_to_host(&mut p.up_proj);
            self.down_proj.copy_to_host(&mut p.down_proj);
        }
        if self.has_fm {
            p.w_rand = self.w_rand_cpu.clone();
            p.b_rand = self.b_rand_cpu.clone();
        }
        p
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuMAGParams — complete model parameters on GPU
// ══════════════════════════════════════════════════════════════════════

/// All learnable parameters resident on GPU.
/// Created once from host MAGParams, persists for the lifetime of the model.
#[cfg(feature = "cuda")]
pub struct GpuMAGParams {
    pub swa: GpuSWAParams,
    pub levels: Vec<GpuMemoryLevelParams>,
    /// Persistent tokens on GPU. Empty (len=0) when n_persistent == 0.
    pub persistent_tokens: GpuBuf<f32>,
}

#[cfg(feature = "cuda")]
impl GpuMAGParams {
    /// Upload all parameters from host to GPU. Called once at model init.
    pub fn from_host(host: &MAGParams) -> Self {
        let persistent_tokens = if host.persistent_tokens.is_empty() {
            GpuBuf::zeros(1) // Avoid zero-length cudaMalloc
        } else {
            GpuBuf::from_host(&host.persistent_tokens)
        };
        GpuMAGParams {
            swa: GpuSWAParams::from_host(&host.swa),
            levels: host.levels.iter().map(GpuMemoryLevelParams::from_host).collect(),
            persistent_tokens,
        }
    }

    /// Download all parameters from GPU to host. Called for checkpointing.
    pub fn to_host(&self, cfg: &MAGConfig) -> MAGParams {
        let d = cfg.swa.d_model;
        let v = cfg.swa.vocab_size;
        let n_pt = cfg.n_persistent * d;
        let mut persistent_tokens = vec![0.0f32; n_pt];
        if n_pt > 0 {
            self.persistent_tokens.copy_to_host(&mut persistent_tokens);
        }
        MAGParams {
            swa: self.swa.to_host(d, v),
            levels: self.levels.iter().map(|l| l.to_host(d)).collect(),
            alpha_mem: vec![0.0f32; cfg.k],
            alpha_refl: vec![0.0f32; cfg.k],
            persistent_tokens,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuContextState — per-level memory matrices on GPU
// ══════════════════════════════════════════════════════════════════════

/// Per-level M matrices resident on GPU. Persists across steps.
///
/// With batch_size > 1, each level buffer holds [batch_size * mem_dd] floats where
/// mem_dd = num_heads * head_dim * head_dim (linear M) or num_heads * state_size
/// (MLP memory, spec 75). For per-head memory (num_heads > 1), the buffer is tightly
/// packed: batch b, head h at offset (b*nh + h) * dd_per_head.
/// Slot 0 is the "primary" context used for checkpointing and inference.
///
/// When `cuda_graph_warmup > 0`: also holds pre-allocated `ForwardScratch` and per-level
/// `GpuLevelScratch` with fixed GPU addresses for CUDA graph capture/replay.
#[cfg(feature = "cuda")]
pub struct GpuContextState {
    /// Per-level M matrices on GPU. Each is [batch_size * mem_dd].
    pub memory: Vec<GpuBuf<f32>>,
    pub d: usize,
    pub batch_size: usize,
    /// Number of memory heads (1 = monolithic d×d, >1 = per-head hd×hd).
    pub num_heads: usize,
    /// Per-head dimension (d / num_heads).
    pub head_dim: usize,
    /// Total M elements across all heads: nh * hd² (linear) or nh * state_size (MLP).
    /// Stored field — avoids recomputation and handles MLP memory sizing (spec 75).
    stored_mem_dd: usize,
    /// Pre-allocated forward buffers for CUDA graph capture (None when warmup_steps=0).
    pub forward_scratch: Option<crate::cuda_graph::ForwardScratch>,
    /// Pre-allocated per-level activation buffers for CUDA graph capture (empty when warmup_steps=0).
    pub level_scratch: Vec<crate::cuda_graph::GpuLevelScratch>,
    /// CUDA graph store — captures/replays kernel dispatch per pulse bitmask.
    pub cuda_graph: crate::cuda_graph::CudaGraphStore,
}

#[cfg(feature = "cuda")]
impl GpuContextState {
    /// Total M element count across all heads.
    /// Linear M: num_heads * head_dim². MLP memory (spec 75): num_heads * state_size.
    #[inline]
    pub fn mem_dd(&self) -> usize {
        self.stored_mem_dd
    }

    /// Initialize zero M matrices for k levels, each [batch_size * mem_dd].
    ///
    /// Extracts num_heads/head_dim from cfg (defaults to nh=1, hd=d when cfg is None).
    /// When `memory_layers >= 2` (TitansLMM MLP memory, spec 75), the per-head state
    /// size is larger: state_size = 2*hd*d_h + d_h + hd instead of hd*hd.
    /// When `cuda_graph_warmup > 0`, also pre-allocates scratch buffers for all k levels
    /// and the forward pass. This adds persistent VRAM equal to ~one forward pass's
    /// intermediates — the same memory that would be dynamically allocated per step.
    pub fn new(
        k: usize, d: usize, batch_size: usize,
        cfg: Option<&crate::model::MAGConfig>,
        cuda_graph_warmup: usize,
    ) -> Self {
        let (nh, hd) = cfg.map_or((1, d), |c| (c.swa.num_heads, c.swa.head_dim));
        // Spec 75: MLP memory needs larger per-head state buffer
        let mem_dd = if let Some(c) = cfg {
            if c.memory_layers >= 2
                && matches!(c.memory_rule, crate::model::MemoryRuleKind::TitansLMM)
            {
                assert_eq!(c.memory_layers, 2,
                    "TitansLMM CUDA MLP currently supports exactly 2 memory layers, got {}",
                    c.memory_layers);
                let d_h = c.memory_expansion_factor * hd;
                // packed: W1[d_h,hd] + b1[d_h] + W2[hd,d_h] + b2[hd]
                nh * (2 * hd * d_h + d_h + hd)
            } else {
                nh * hd * hd
            }
        } else {
            nh * hd * hd
        };
        let memory = (0..k).map(|_| GpuBuf::zeros(batch_size * mem_dd)).collect();
        let (forward_scratch, level_scratch) = if cuda_graph_warmup > 0 {
            if let Some(cfg) = cfg {
                let fwd = crate::cuda_graph::ForwardScratch::from_cfg(cfg, batch_size);
                let lvl: Vec<_> = (0..k)
                    .map(|_| crate::cuda_graph::GpuLevelScratch::from_cfg(cfg, batch_size))
                    .collect();
                (Some(fwd), lvl)
            } else {
                (None, Vec::new())
            }
        } else {
            (None, Vec::new())
        };
        GpuContextState {
            memory,
            d,
            batch_size,
            num_heads: nh,
            head_dim: hd,
            stored_mem_dd: mem_dd,
            forward_scratch,
            level_scratch,
            cuda_graph: crate::cuda_graph::CudaGraphStore::new(cuda_graph_warmup),
        }
    }

    /// Download slot-0 M to host ContextState (for checkpoint / inference).
    /// Only the primary context (slot 0) is exported — sufficient for restore.
    /// Copies mem_dd = num_heads * head_dim^2 floats (all heads of batch 0).
    pub fn to_host(&self, k: usize) -> crate::conductor::ContextState {
        let mem_dd = self.mem_dd();
        let mut ctx = crate::conductor::ContextState::new_with_memory_size(k, self.d, mem_dd);
        for (i, gpu_mem) in self.memory.iter().enumerate() {
            // Copy only slot-0: first mem_dd floats (all heads of batch element 0).
            gpu_mem.slice(0, mem_dd).copy_to_host(&mut ctx.memory[i]);
        }
        ctx
    }

    /// Zero all memory matrices on GPU in-place (cudaMemset).
    /// Zeros all batch_size * mem_dd floats per level.
    /// Used at document boundaries — same semantics as ContextState::reset().
    pub fn reset(&mut self) {
        for buf in &self.memory {
            buf.zero();
        }
    }

    /// TNT periodic reset: zero context.memory[level] (all batch slots).
    /// Called after each step where pulse.active_levels[level] is true,
    /// when memory_reset = "periodic". Resets M to zeros (the initial prior).
    /// CS-32: called AFTER the step's forward/backward/update — the previous
    /// shard's final M was already observed; the next shard starts from zero.
    ///
    /// Note: when M_init becomes a learnable outer_loop_param (follow-up task),
    /// this method should D2D-copy m_init[level] instead of zeroing.
    pub fn periodic_reset_level(&self, level: usize) {
        if let Some(buf) = self.memory.get(level) {
            buf.zero();
        }
    }

    /// Upload host M matrices into existing GPU buffers (in-place).
    /// Broadcasts slot-0 to all batch slots at mem_dd stride. Preserves scratch/cuda_graph state.
    /// Used by gpu_tape_forward_summary to restore context after diagnostic.
    pub fn upload_memory(&mut self, host: &crate::conductor::ContextState) {
        let mem_dd = self.mem_dd();
        let bytes = mem_dd * 4;
        for (i, m) in host.memory.iter().enumerate() {
            let slot0 = GpuBuf::<f32>::from_host(m);
            for b in 0..self.batch_size {
                unsafe {
                    let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                        (self.memory[i].ptr() as *mut u8).add(b * bytes) as *mut std::ffi::c_void,
                        slot0.as_ptr() as *const std::ffi::c_void,
                        bytes,
                    );
                    assert_eq!(rc, 0, "upload_memory D2D copy failed for level {i} slot {b}");
                }
            }
        }
    }

    /// Upload from host ContextState and broadcast to all batch_size slots.
    /// Used for checkpoint restore: host M is written to every slot so all
    /// batch elements start from the same saved context.
    ///
    /// `num_heads` and `head_dim` determine the per-slot M size (mem_dd = nh * hd * hd).
    /// For monolithic models pass (1, d); for per-head pass the actual head config.
    pub fn from_host_context(
        host: &crate::conductor::ContextState,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let d = host.d;
        // Infer mem_dd from host memory size (supports both linear hd*hd and MLP state_size).
        let mem_dd = host.memory.first()
            .map(|m| m.len())
            .unwrap_or(num_heads * head_dim * head_dim);
        assert!(
            host.memory.iter().all(|m| m.len() == mem_dd),
            "GpuContextState::from_host_context: all levels must have uniform memory size"
        );
        let bytes = mem_dd * std::mem::size_of::<f32>();
        let memory = host.memory.iter().map(|m| {
            let buf = GpuBuf::<f32>::zeros(batch_size * mem_dd);
            // Upload host M once, then D2D-copy to all slots.
            let slot0 = GpuBuf::<f32>::from_host(m);
            for b in 0..batch_size {
                unsafe {
                    let rc = crate::gpu_forward::gpu_buf_memcpy_d2d(
                        (buf.ptr() as *mut u8).add(b * bytes) as *mut std::ffi::c_void,
                        slot0.as_ptr() as *const std::ffi::c_void,
                        bytes,
                    );
                    assert_eq!(rc, 0, "from_host_context D2D copy failed for slot {b}");
                }
            }
            buf
        }).collect();
        GpuContextState {
            memory,
            d,
            batch_size,
            num_heads,
            head_dim,
            stored_mem_dd: mem_dd,
            forward_scratch: None,
            level_scratch: Vec::new(),
            cuda_graph: crate::cuda_graph::CudaGraphStore::new(0),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuBlockParams — per-block SWA + LN + CMS levels on GPU
// ══════════════════════════════════════════════════════════════════════

/// One block's learnable parameters resident on GPU.
/// Contains SWA projections (no embed/unembed), LayerNorms, and CMS levels.
#[cfg(feature = "cuda")]
pub struct GpuBlockParams {
    // SWA attention projections
    pub w_q: GpuBuf<f32>,       // [d, d]
    pub w_k: GpuBuf<f32>,       // [d, d]
    pub w_v: GpuBuf<f32>,       // [d, d]
    pub w_o: GpuBuf<f32>,       // [d, d]
    // Pre-norm LayerNorm for attention branch
    pub ln_attn_gamma: GpuBuf<f32>,  // [d]
    pub ln_attn_beta: GpuBuf<f32>,   // [d]
    // Pre-norm LayerNorm for memory branch
    pub ln_mem_gamma: GpuBuf<f32>,   // [d]
    pub ln_mem_beta: GpuBuf<f32>,    // [d]
    // CMS memory levels (length k)
    pub levels: Vec<GpuMemoryLevelParams>,
    // CMS aggregation logits (kept on GPU for future learnable aggregation)
    pub alpha_mem: GpuBuf<f32>,  // [k]
    pub alpha_refl: GpuBuf<f32>, // [k]
}

#[cfg(feature = "cuda")]
impl GpuBlockParams {
    /// Upload one block's parameters from host BlockParams.
    pub fn from_host(host: &crate::stacked_model::BlockParams) -> Self {
        GpuBlockParams {
            w_q: GpuBuf::from_host(&host.w_q),
            w_k: GpuBuf::from_host(&host.w_k),
            w_v: GpuBuf::from_host(&host.w_v),
            w_o: GpuBuf::from_host(&host.w_o),
            ln_attn_gamma: GpuBuf::from_host(&host.ln_attn_gamma),
            ln_attn_beta: GpuBuf::from_host(&host.ln_attn_beta),
            ln_mem_gamma: GpuBuf::from_host(&host.ln_mem_gamma),
            ln_mem_beta: GpuBuf::from_host(&host.ln_mem_beta),
            levels: host.levels.iter().map(GpuMemoryLevelParams::from_host).collect(),
            alpha_mem: GpuBuf::from_host(&host.alpha_mem),
            alpha_refl: GpuBuf::from_host(&host.alpha_refl),
        }
    }

    /// Download block parameters from GPU to host.
    pub fn to_host(&self, d: usize, k: usize) -> crate::stacked_model::BlockParams {
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        let mut w_o = vec![0.0f32; d * d];
        let mut ln_attn_gamma = vec![0.0f32; d];
        let mut ln_attn_beta = vec![0.0f32; d];
        let mut ln_mem_gamma = vec![0.0f32; d];
        let mut ln_mem_beta = vec![0.0f32; d];
        let mut alpha_mem = vec![0.0f32; k];
        let mut alpha_refl = vec![0.0f32; k];

        self.w_q.copy_to_host(&mut w_q);
        self.w_k.copy_to_host(&mut w_k);
        self.w_v.copy_to_host(&mut w_v);
        self.w_o.copy_to_host(&mut w_o);
        self.ln_attn_gamma.copy_to_host(&mut ln_attn_gamma);
        self.ln_attn_beta.copy_to_host(&mut ln_attn_beta);
        self.ln_mem_gamma.copy_to_host(&mut ln_mem_gamma);
        self.ln_mem_beta.copy_to_host(&mut ln_mem_beta);
        self.alpha_mem.copy_to_host(&mut alpha_mem);
        self.alpha_refl.copy_to_host(&mut alpha_refl);

        crate::stacked_model::BlockParams {
            w_q, w_k, w_v, w_o,
            ln_attn_gamma, ln_attn_beta,
            ln_mem_gamma, ln_mem_beta,
            levels: self.levels.iter().map(|l| l.to_host(d)).collect(),
            alpha_mem, alpha_refl,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedParams — shared embed/unembed + N blocks on GPU
// ══════════════════════════════════════════════════════════════════════

/// Full stacked model parameters on GPU.
/// Shared embedding/unembedding + final LayerNorm, plus N independent blocks.
#[cfg(feature = "cuda")]
pub struct GpuStackedParams {
    pub w_embed: GpuBuf<f32>,        // [vocab, d]
    pub w_unembed: GpuBuf<f32>,      // [d, vocab]
    pub ln_final_gamma: GpuBuf<f32>, // [d]
    pub ln_final_beta: GpuBuf<f32>,  // [d]
    pub blocks: Vec<GpuBlockParams>,
    /// Persistent tokens on GPU: [n_persistent, d]. Empty when n_persistent == 0.
    pub persistent_tokens: GpuBuf<f32>,
}

#[cfg(feature = "cuda")]
impl GpuStackedParams {
    /// Upload all stacked parameters from host to GPU.
    pub fn from_host(host: &crate::stacked_model::StackedMAGParams) -> Self {
        let persistent_tokens = if host.persistent_tokens.is_empty() {
            GpuBuf::zeros(1) // placeholder for n_persistent=0
        } else {
            GpuBuf::from_host(&host.persistent_tokens)
        };
        GpuStackedParams {
            w_embed: GpuBuf::from_host(&host.w_embed),
            w_unembed: GpuBuf::from_host(&host.w_unembed),
            ln_final_gamma: GpuBuf::from_host(&host.ln_final_gamma),
            ln_final_beta: GpuBuf::from_host(&host.ln_final_beta),
            blocks: host.blocks.iter().map(GpuBlockParams::from_host).collect(),
            persistent_tokens,
        }
    }

    /// Download all parameters from GPU to host.
    pub fn to_host(&self, cfg: &crate::model::MAGConfig) -> crate::stacked_model::StackedMAGParams {
        let d = cfg.swa.d_model;
        let vocab = cfg.swa.vocab_size;
        let k = cfg.k;
        let mut w_embed = vec![0.0f32; vocab * d];
        let mut w_unembed = vec![0.0f32; d * vocab];
        let mut ln_final_gamma = vec![0.0f32; d];
        let mut ln_final_beta = vec![0.0f32; d];

        self.w_embed.copy_to_host(&mut w_embed);
        self.w_unembed.copy_to_host(&mut w_unembed);
        self.ln_final_gamma.copy_to_host(&mut ln_final_gamma);
        self.ln_final_beta.copy_to_host(&mut ln_final_beta);

        let n_pt = cfg.n_persistent * d;
        let mut persistent_tokens = vec![0.0f32; n_pt];
        if n_pt > 0 {
            self.persistent_tokens.copy_to_host(&mut persistent_tokens);
        }

        crate::stacked_model::StackedMAGParams {
            w_embed, w_unembed,
            ln_final_gamma, ln_final_beta,
            blocks: self.blocks.iter().map(|b| b.to_host(d, k)).collect(),
            persistent_tokens,
        }
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuStackedContext — per-block per-level M matrices on GPU
// ══════════════════════════════════════════════════════════════════════

/// Memory state for a stacked model: [n_blocks][k] M matrices on GPU.
/// Each block maintains its own set of k memory matrices.
#[cfg(feature = "cuda")]
pub struct GpuStackedContext {
    /// Per-block context states. Each has k memory buffers.
    pub blocks: Vec<GpuContextState>,
    pub d: usize,
    pub batch_size: usize,
    pub n_blocks: usize,
    // ── Dormancy tracking (spec 28) ──────────────────────────────────
    /// Previous step's per-(block, level) M Frobenius norms.
    pub prev_m_norms: Vec<Vec<f32>>,
    /// Per-(block, level) absolute M norm delta from last step.
    pub m_norm_deltas: Vec<Vec<f32>>,
    /// Per-(block, level) consecutive steps with M-diff below dormancy floor.
    pub dormancy_below_count: Vec<Vec<usize>>,
    /// Per-level dormancy floor thresholds (length = k). Empty = disabled.
    pub dormancy_floors: Vec<f32>,
    /// Consecutive below-floor steps to trigger "dormant" status.
    pub dormancy_consecutive: usize,
    /// Cached per-(block, level, head) M norms from pre-reset snapshot.
    /// Populated by update_m_norm_tracking() before periodic_reset zeros the buffers.
    pub cached_per_head_norms: Vec<Vec<Vec<f32>>>,
}

#[cfg(feature = "cuda")]
impl GpuStackedContext {
    /// Initialize zero M matrices for all blocks x levels.
    /// No CUDA graph support for stacked forward (standard dispatch only).
    ///
    /// When `cfg` is provided, per-head memory layout is inherited
    /// (num_heads × head_dim² tiles). Pass `None` for monolithic d×d.
    pub fn new(
        n_blocks: usize, k: usize, d: usize, batch_size: usize,
        cfg: Option<&crate::model::MAGConfig>,
    ) -> Self {
        let blocks = (0..n_blocks)
            .map(|_| GpuContextState::new(k, d, batch_size, cfg, 0))
            .collect();
        GpuStackedContext {
            blocks, d, batch_size, n_blocks,
            prev_m_norms: vec![vec![0.0; k]; n_blocks],
            m_norm_deltas: vec![vec![0.0; k]; n_blocks],
            dormancy_below_count: vec![vec![0; k]; n_blocks],
            dormancy_floors: Vec::new(),
            dormancy_consecutive: 5,
            cached_per_head_norms: Vec::new(),
        }
    }

    /// Zero all memory across all blocks.
    pub fn reset(&mut self) {
        for ctx in &mut self.blocks {
            ctx.reset();
        }
    }

    /// TNT periodic reset for a specific level across all blocks.
    /// CS-32: reset happens after the step's advance, before the next step's observe.
    pub fn periodic_reset_level(&self, level: usize) {
        for ctx in &self.blocks {
            ctx.periodic_reset_level(level);
        }
    }

    /// Compute per-(block, level) Frobenius norms of M matrices on GPU.
    /// Returns `Vec<Vec<f32>>` -- outer len = n_blocks, inner len = k.
    pub fn memory_norms(&self) -> Vec<Vec<f32>> {
        let mut result = Vec::with_capacity(self.n_blocks);
        for ctx in &self.blocks {
            let slot_size = ctx.mem_dd(); // per-head: nh * hd * hd; monolithic: d * d
            let mut block_norms = Vec::with_capacity(ctx.memory.len());
            for buf in &ctx.memory {
                if buf.len() == 0 || slot_size == 0 {
                    block_norms.push(0.0);
                    continue;
                }
                // Norm slot 0 only (matches single-block metric, independent of batch_size)
                let n = slot_size.min(buf.len()) as i32;
                let mut num_blocks_out: i32 = 0;
                let max_norm_blocks = (n as usize + 255) / 256;
                let scratch = GpuBuf::zeros(max_norm_blocks);
                let err = unsafe {
                    crate::cuda_ffi::grad_norm_sq_cuda(
                        buf.as_ptr(), scratch.ptr(), n, &mut num_blocks_out,
                    )
                };
                assert_eq!(err, 0, "grad_norm_sq_cuda for M state norm failed");
                crate::dispatch::cuda_sync();
                let nb = num_blocks_out as usize;
                let mut host = vec![0.0f32; nb];
                scratch.slice(0, nb).copy_to_host(&mut host);
                let sq_sum: f64 = host.iter().map(|x| *x as f64).sum();
                block_norms.push(sq_sum.sqrt() as f32);
            }
            result.push(block_norms);
        }
        result
    }

    /// Compute per-(block, level, head) Frobenius norms of M sub-matrices.
    /// Returns `Vec<Vec<Vec<f32>>>` — [n_blocks][k][num_heads].
    /// Empty inner Vec for levels with non-square M (MLP rules) or num_heads <= 1.
    ///
    /// GPU memory layout: packed per-head tiles, head h at offset `h * hd * hd`,
    /// each tile is `hd × hd` contiguous. D2H copy of slot 0, then per-tile norm.
    pub fn memory_norms_per_head(&self) -> Vec<Vec<Vec<f32>>> {
        let mut result = Vec::with_capacity(self.blocks.len());
        for ctx in &self.blocks {
            let nh = ctx.num_heads;
            let hd = ctx.head_dim;
            let mem_dd = ctx.mem_dd(); // nh * hd² (linear) or nh * state_size (MLP)
            let mut block_head_norms = Vec::with_capacity(ctx.memory.len());
            for buf in &ctx.memory {
                // Skip per-head decomposition for MLP memory (non-square tiles)
                if nh <= 1 || mem_dd == 0 || buf.len() < mem_dd || mem_dd != nh * hd * hd {
                    block_head_norms.push(Vec::new());
                    continue;
                }
                // D2H copy of slot 0 (first batch element): mem_dd floats
                let mut host = vec![0.0f32; mem_dd];
                buf.slice(0, mem_dd).copy_to_host(&mut host);
                // Packed layout: head h is at host[h*hd*hd .. (h+1)*hd*hd]
                let tile_size = hd * hd;
                let norms: Vec<f32> = (0..nh).map(|h| {
                    let start = h * tile_size;
                    host[start..start + tile_size].iter()
                        .map(|v| v * v).sum::<f32>().sqrt()
                }).collect();
                block_head_norms.push(norms);
            }
            result.push(block_head_norms);
        }
        result
    }

    /// Update M-norm tracking after a forward pass (spec 28 dormancy sentinel).
    /// Computes current M norms, diffs against prev, updates dormancy counters.
    pub fn update_m_norm_tracking(&mut self) {
        let current_norms = self.memory_norms();
        let _k = if let Some(first) = current_norms.first() { first.len() } else { 0 };

        for (bi, block_norms) in current_norms.iter().enumerate() {
            if bi >= self.prev_m_norms.len() { continue; }
            for (li, &norm) in block_norms.iter().enumerate() {
                if li >= self.prev_m_norms[bi].len() { continue; }
                let delta = (norm - self.prev_m_norms[bi][li]).abs();
                self.m_norm_deltas[bi][li] = delta;

                // Dormancy counter: check against per-level floor
                if !self.dormancy_floors.is_empty() && li < self.dormancy_floors.len() {
                    if delta < self.dormancy_floors[li] {
                        self.dormancy_below_count[bi][li] += 1;
                    } else {
                        self.dormancy_below_count[bi][li] = 0;
                    }
                }
            }
        }

        self.prev_m_norms = current_norms;
        // Cache per-head norms before periodic reset zeros the buffers.
        // PyO3 memory_norms_per_head() reads this cached snapshot.
        self.cached_per_head_norms = self.memory_norms_per_head();
    }

    /// Per-(block, level) dormancy status: "active", "warning", "dormant".
    /// Warning = above half of dormancy_consecutive threshold.
    /// Dormant = at or above dormancy_consecutive.
    /// Returns all "active" when dormancy_consecutive == 0 (disabled).
    pub fn dormancy_status(&self) -> Vec<Vec<String>> {
        if self.dormancy_consecutive == 0 {
            return self.dormancy_below_count.iter().map(|block| {
                vec!["active".to_string(); block.len()]
            }).collect();
        }
        let half = self.dormancy_consecutive / 2;
        self.dormancy_below_count.iter().map(|block| {
            block.iter().map(|&count| {
                if count >= self.dormancy_consecutive {
                    "dormant".to_string()
                } else if count > half {
                    "warning".to_string()
                } else {
                    "active".to_string()
                }
            }).collect()
        }).collect()
    }

    /// Configure dormancy detection thresholds.
    /// Resets dormancy counters to avoid stale state from a previous configuration.
    pub fn set_dormancy_config(&mut self, floors: Vec<f32>, consecutive: usize) {
        self.dormancy_floors = floors;
        self.dormancy_consecutive = consecutive;
        // Reset counters to match current block/level structure
        let k = if let Some(first) = self.dormancy_below_count.first() {
            first.len()
        } else {
            0
        };
        self.dormancy_below_count = vec![vec![0; k]; self.n_blocks];
    }

    /// Deep clone all GPU memory buffers (D2D copy). Used by tape diagnostics
    /// to save/restore context around a non-destructive forward+backward.
    pub fn deep_clone(&self) -> Self {
        let blocks = self.blocks.iter().map(|ctx| {
            let memory = ctx.memory.iter().map(|buf| {
                let copy = GpuBuf::zeros(buf.len());
                let err = unsafe {
                    crate::gpu_forward::gpu_buf_memcpy_d2d(
                        copy.ptr() as *mut std::ffi::c_void,
                        buf.as_ptr() as *const std::ffi::c_void,
                        buf.len() * std::mem::size_of::<f32>(),
                    )
                };
                assert_eq!(err, 0, "D2D memcpy failed in GpuStackedContext::deep_clone");
                copy
            }).collect();
            // No CUDA graph scratch for stacked models, so create a minimal context
            GpuContextState {
                memory,
                d: ctx.d,
                batch_size: ctx.batch_size,
                num_heads: ctx.num_heads,
                head_dim: ctx.head_dim,
                stored_mem_dd: ctx.stored_mem_dd,
                forward_scratch: None,
                level_scratch: Vec::new(),
                cuda_graph: crate::cuda_graph::CudaGraphStore::new(0),
            }
        }).collect();
        GpuStackedContext {
            blocks,
            d: self.d,
            batch_size: self.batch_size,
            n_blocks: self.n_blocks,
            prev_m_norms: self.prev_m_norms.clone(),
            m_norm_deltas: self.m_norm_deltas.clone(),
            dormancy_below_count: self.dormancy_below_count.clone(),
            dormancy_floors: self.dormancy_floors.clone(),
            dormancy_consecutive: self.dormancy_consecutive,
            cached_per_head_norms: self.cached_per_head_norms.clone(),
        }
    }
}
