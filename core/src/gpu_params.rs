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
        };
        self.w_embed.copy_to_host(&mut p.w_embed);
        self.w_q.copy_to_host(&mut p.w_q);
        self.w_k.copy_to_host(&mut p.w_k);
        self.w_v.copy_to_host(&mut p.w_v);
        self.w_o.copy_to_host(&mut p.w_o);
        self.w_unembed.copy_to_host(&mut p.w_unembed);
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
    pub w_omega: GpuBuf<f32>,   // [d, 2*d]
    // Frequency gate (empty for Fixed schedule — use len=1 dummy)
    pub w_freq: GpuBuf<f32>,
    pub b_freq: GpuBuf<f32>,
    pub has_freq: bool,
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

        GpuMemoryLevelParams {
            w_k_mem: GpuBuf::from_host(&host.w_k_mem),
            w_v_mem: GpuBuf::from_host(&host.w_v_mem),
            w_q_mem: GpuBuf::from_host(&host.w_q_mem),
            w_alpha: GpuBuf::from_host(&host.w_alpha),
            b_alpha: GpuBuf::from_host(&host.b_alpha),
            w_theta: GpuBuf::from_host(&host.w_theta),
            b_theta: GpuBuf::from_host(&host.b_theta),
            w_eta: GpuBuf::from_host(&host.w_eta),
            b_eta: GpuBuf::from_host(&host.b_eta),
            w_omega: GpuBuf::from_host(&host.w_omega),
            w_freq,
            b_freq,
            has_freq,
        }
    }

    pub fn to_host(&self, d: usize) -> MemoryLevelParams {
        let mut p = MemoryLevelParams::zeros_like(d);
        self.w_k_mem.copy_to_host(&mut p.w_k_mem);
        self.w_v_mem.copy_to_host(&mut p.w_v_mem);
        self.w_q_mem.copy_to_host(&mut p.w_q_mem);
        self.w_alpha.copy_to_host(&mut p.w_alpha);
        self.b_alpha.copy_to_host(&mut p.b_alpha);
        self.w_theta.copy_to_host(&mut p.w_theta);
        self.b_theta.copy_to_host(&mut p.b_theta);
        self.w_eta.copy_to_host(&mut p.w_eta);
        self.b_eta.copy_to_host(&mut p.b_eta);
        self.w_omega.copy_to_host(&mut p.w_omega);
        if self.has_freq {
            p.w_freq = vec![0.0f32; self.w_freq.len()];
            p.b_freq = vec![0.0f32; self.b_freq.len()];
            self.w_freq.copy_to_host(&mut p.w_freq);
            self.b_freq.copy_to_host(&mut p.b_freq);
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
}

#[cfg(feature = "cuda")]
impl GpuMAGParams {
    /// Upload all parameters from host to GPU. Called once at model init.
    pub fn from_host(host: &MAGParams) -> Self {
        GpuMAGParams {
            swa: GpuSWAParams::from_host(&host.swa),
            levels: host.levels.iter().map(GpuMemoryLevelParams::from_host).collect(),
        }
    }

    /// Download all parameters from GPU to host. Called for checkpointing.
    pub fn to_host(&self, cfg: &MAGConfig) -> MAGParams {
        let d = cfg.swa.d_model;
        let v = cfg.swa.vocab_size;
        MAGParams {
            swa: self.swa.to_host(d, v),
            levels: self.levels.iter().map(|l| l.to_host(d)).collect(),
            alpha_mem: vec![0.0f32; cfg.k],
            alpha_refl: vec![0.0f32; cfg.k],
            persistent_tokens: vec![0.0f32; cfg.n_persistent * cfg.swa.d_model],
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuContextState — per-level memory matrices on GPU
// ══════════════════════════════════════════════════════════════════════

/// Per-level M matrices resident on GPU. Persists across steps.
/// For matrix rules (Delta/Titans/Hebbian): each level has d*d floats.
#[cfg(feature = "cuda")]
pub struct GpuContextState {
    /// Per-level M matrices on GPU. Each is [d*d] (or mem_size for MLP rules).
    pub memory: Vec<GpuBuf<f32>>,
    pub d: usize,
}

#[cfg(feature = "cuda")]
impl GpuContextState {
    /// Initialize zero M matrices for k levels, each d*d.
    pub fn new(k: usize, d: usize) -> Self {
        let memory = (0..k).map(|_| GpuBuf::zeros(d * d)).collect();
        GpuContextState { memory, d }
    }

    /// Download to host ContextState (for checkpoint).
    pub fn to_host(&self, k: usize) -> crate::conductor::ContextState {
        let mut ctx = crate::conductor::ContextState::new(k, self.d);
        for (i, gpu_mem) in self.memory.iter().enumerate() {
            gpu_mem.copy_to_host(&mut ctx.memory[i]);
        }
        ctx
    }

    /// Zero all memory matrices on GPU in-place (cudaMemset).
    /// Used at document boundaries — same semantics as ContextState::reset().
    pub fn reset(&mut self) {
        for buf in &self.memory {
            buf.zero();
        }
    }

    /// Upload from host ContextState (for restore).
    pub fn from_host_context(host: &crate::conductor::ContextState) -> Self {
        let memory = host.memory.iter()
            .map(|m| GpuBuf::from_host(m))
            .collect();
        GpuContextState { memory, d: host.d }
    }
}
