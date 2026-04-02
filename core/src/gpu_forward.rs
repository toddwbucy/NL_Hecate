/// GPU-resident CMS forward pass.
///
/// Mirrors `cms_forward` in mag.rs but operates entirely on device pointers.
/// Only input_ids (4KB), target_ids (4KB), and loss (4 bytes) cross PCIe.
///
/// Supports matrix-based memory rules: DeltaRule, TitansLMM, HebbianRule, DGD.
/// Supports MLP-based memory rules: Moneta, YAAD.
/// Remaining MLP/slot-based rules (MEMORA, Lattice, Trellis, Atlas) fall
/// back to the CPU reference path since they lack CUDA kernels.
///
/// Feature-gated: only available with `--features cuda`.

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::gpu_params::{GpuMAGParams, GpuContextState};
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind, LevelTapeStrategy};
#[cfg(feature = "cuda")]
use crate::conductor::Pulse;

/// One-shot warning: frozen MLP level skipped (no gpu_mlp_read_only yet).
#[cfg(feature = "cuda")]
static WARNED_FROZEN_MLP: AtomicBool = AtomicBool::new(false);

/// Returns a zeroed GPU buffer for a frozen MLP level, with a one-time warning.
/// MLP rules (Moneta/YAAD) have context_m = [W1|W2], incompatible with
/// gpu_memory_read_only's [d,d] matrix assumption.
#[cfg(feature = "cuda")]
fn frozen_mlp_fallback(path: &str, level: usize, rule: MemoryRuleKind, n_elements: usize) -> GpuBuf<f32> {
    if !WARNED_FROZEN_MLP.swap(true, Ordering::Relaxed) {
        eprintln!(
            "WARNING: {}: frozen MLP level {} ({:?}) contributing zeros to y_combined — \
             gpu_mlp_read_only (y = W2 @ silu(W1 @ q)) not yet implemented. \
             Moneta/YAAD are recommended with k=1; in k>1 configs, frozen levels \
             produce zero contribution. Use matrix-based rules (TitansLMM, DeltaRule) \
             if frozen-level read-only M @ q_mem is required. This message prints once.",
            path, level, rule,
        );
    }
    GpuBuf::<f32>::zeros(n_elements)
}
#[cfg(feature = "cuda")]
use crate::parallel::ParallelStrategy;

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
    pub embedded: GpuBuf<f32>,        // [bs*s, d]

    // Attention branch (bf16 for SWA)
    pub q_f32: GpuBuf<f32>,           // [bs*s, d] — f32 version (needed for backward projections)
    pub k_f32: GpuBuf<f32>,           // [bs*s, d]
    pub v_f32: GpuBuf<f32>,           // [bs*s, d]
    pub q_bf16: GpuBuf<u16>,          // [bs*s, d] bf16
    pub k_bf16: GpuBuf<u16>,          // [bs*s, d] bf16
    pub v_bf16: GpuBuf<u16>,          // [bs*s, d] bf16
    pub attn_out_bf16: GpuBuf<u16>,   // [bs*s, d] bf16
    pub attn_weights_bf16: GpuBuf<u16>, // [bs*nh, s, n_persistent+ws] bf16
    pub attn_out: GpuBuf<f32>,        // [bs*s, d] f32 (converted back)

    // Memory branch per level
    pub memory_caches: Vec<Option<GpuMemoryCache>>,
    pub y_per_level: Vec<GpuBuf<f32>>, // [bs*s, d] per level

    // Combined + gating (non-residual path)
    pub y_combined: GpuBuf<f32>,      // [bs*s, d]
    pub gate: GpuBuf<f32>,            // [bs*s, d] sigmoid(y_combined)
    pub gated_out: GpuBuf<f32>,       // [bs*s, d] attn_out * gate

    // Residual path caches (only populated when cfg.residual=true)
    pub ln_attn_out: Option<GpuBuf<f32>>,     // [bs*s, d]
    pub ln_attn_mean: Option<GpuBuf<f32>>,    // [bs*s]
    pub ln_attn_rstd: Option<GpuBuf<f32>>,    // [bs*s]
    pub ln_mem_out: Option<GpuBuf<f32>>,      // [bs*s, d]
    pub ln_mem_mean: Option<GpuBuf<f32>>,     // [bs*s]
    pub ln_mem_rstd: Option<GpuBuf<f32>>,     // [bs*s]
    pub residual_after_attn: Option<GpuBuf<f32>>, // [bs*s, d]
    pub residual_final: Option<GpuBuf<f32>>,      // [bs*s, d]

    // Post-gating
    pub projected: GpuBuf<f32>,       // [bs*s, d]
    pub logits: GpuBuf<f32>,          // [bs*s, v]

    // Pulse snapshot (needed by backward for level dispatch)
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

/// Per-level memory cache on GPU.
#[cfg(feature = "cuda")]
pub enum GpuMemoryCache {
    Delta {
        k_mem: GpuBuf<f32>,     // [s, d]
        v_mem: GpuBuf<f32>,     // [s, d]
        q_mem: GpuBuf<f32>,     // [s, d]
        alpha: GpuBuf<f32>,     // [s]
        theta: GpuBuf<f32>,     // [s]
        m_states: GpuBuf<f32>,  // [(s+1)*d*d] exact, [d*d] proxy
        k_norms: GpuBuf<f32>,  // [s] — L2 norms before normalization
        q_norms: GpuBuf<f32>,  // [s]
        proxy: bool,            // spec 27: true = m_states is M_final only
    },
    Titans {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        eta: GpuBuf<f32>,
        m_states: GpuBuf<f32>,  // [(s+1)*d*d] exact, [d*d] proxy
        s_states: GpuBuf<f32>,  // [(s+1)*d*d] exact, [d*d] proxy
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
        proxy: bool,            // spec 27: true = m_states/s_states are finals only
    },
    Hebbian {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        m_states: GpuBuf<f32>,
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
    },
    // ── DGD (Delta Gradient Descent) — HOPE inner-loop optimizer ─────
    DGD {
        k_mem: GpuBuf<f32>,     // [s, d]
        v_mem: GpuBuf<f32>,     // [s, d]
        q_mem: GpuBuf<f32>,     // [s, d]
        alpha: GpuBuf<f32>,     // [s]
        theta: GpuBuf<f32>,     // [s]
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
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
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
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
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
    },
    HebbianCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,
        checkpoint_interval: usize,
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
    },
    DGDCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,  // [num_ckpt * d*d]
        checkpoint_interval: usize,
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
    },
    /// SwiGLU MLP — no M state. All activations saved for weight grad computation.
    SwiGlu {
        gate_buf:  GpuBuf<f32>,   // [s × inter]
        up_buf:    GpuBuf<f32>,   // [s × inter]
        fused_buf: GpuBuf<f32>,   // [s × inter]
        cache_buf: GpuBuf<f32>,   // [s × inter] sigmoid(gate_out)
    },
    /// MLP memory (MONETA/YAAD) — full W1/W2 trajectory.
    /// W1[d_hidden, d], W2[d, d_hidden] per timestep.
    Mlp {
        k_mem: GpuBuf<f32>,       // [s, d]
        v_mem: GpuBuf<f32>,       // [s, d]
        q_mem: GpuBuf<f32>,       // [s, d]
        alpha: GpuBuf<f32>,       // [s]
        theta: GpuBuf<f32>,       // [s]
        w1_states: GpuBuf<f32>,   // [(s+1) * d_hidden * d]
        w2_states: GpuBuf<f32>,   // [(s+1) * d * d_hidden]
        k_norms: GpuBuf<f32>,     // [s] — L2 norms before normalization
        q_norms: GpuBuf<f32>,     // [s]
        // YAAD-only: boundary snapshots for decoupled L2 retention (None for MONETA)
        w1_boundary: Option<GpuBuf<f32>>,  // [d_hidden * d]
        w2_boundary: Option<GpuBuf<f32>>,  // [d * d_hidden]
    },
    /// Delta chunkwise (spec 43 — frozen-M₀). Stores chunk boundary M states.
    /// m_chunk_states: [bs * (num_chunks+1) * d*d].
    DeltaChunkwise {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        m_chunk_states: GpuBuf<f32>,  // [(num_chunks+1)*d*d]
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
        chunk_size: usize,
        num_chunks: usize,
    },
    /// Titans chunkwise (spec 43 — frozen-M₀). Stores chunk boundary M and S states.
    TitansChunkwise {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        eta: GpuBuf<f32>,
        m_chunk_states: GpuBuf<f32>,  // [(num_chunks+1)*d*d]
        s_chunk_states: GpuBuf<f32>,  // [(num_chunks+1)*d*d]
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
        chunk_size: usize,
        num_chunks: usize,
    },
    /// TNT hierarchical — per-shard inner caches + global state trajectory.
    /// The inner caches are Titans/Delta/etc. caches, one per shard.
    TNT {
        /// Inner memory cache per retained shard (Titans/Delta/Hebbian variant).
        /// Length = min(total_shards, retained_shards). Older shards are evicted
        /// during forward when the cycle-scoped retention window is exceeded.
        shard_inner_caches: Vec<GpuMemoryCache>,
        /// Summary key vectors [d] per shard (ALL shards — needed for global M backward).
        k_summaries: Vec<GpuBuf<f32>>,
        /// Summary value vectors [d] per shard (ALL shards — needed for global M backward).
        v_summaries: Vec<GpuBuf<f32>>,
        /// TNT config: global_chunk_size, local_chunk_size.
        global_chunk_size: usize,
        local_chunk_size: usize,
        /// Total number of shards in the sequence (may exceed retained count).
        total_shards: usize,
        /// Index of the first retained shard (0-based). Shards before this were evicted.
        first_retained_shard: usize,
    },
}

// ══════════════════════════════════════════════════════════════════════
// DGD delta norm extraction from cache
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
impl GpuMemoryCache {
    /// Compute ‖M_{s-1} @ k_{s-1} - v_{s-1}‖₂ from the cached forward data.
    ///
    /// Uses the PRE-update M state (M_{s-1}) with the last token's k/v to match
    /// the error that actually drove the final M-update: e_{s-1} = M_{s-1}@k - v.
    /// For matrix-memory rules (Delta, Titans, DGD): reads from m_states cache.
    /// For Hebbian/SwiGlu: returns 0.0 (no error vector).
    /// For TNT: drills into shard inner caches, returns max across shards.
    ///
    /// Checkpointed variants use the second-to-last checkpoint as an
    /// approximation of the pre-update M (exact when s is a checkpoint boundary).
    ///
    /// Source: HOPE (2512.24695) Eq 88 — error = M@k - v
    /// Spec:   specs/infrastructure/16_dgd_delta_norm_gpu.md
    pub fn dgd_delta_norm(&self, s: usize, d: usize, batch_size: usize, num_heads: usize) -> f32 {
        if num_heads > 1 {
            // Diagnostic assumes monolithic d×d M state; per-head layout (nh × hd × hd)
            // is not supported. Return 0.0 so callers see a safe sentinel.
            return 0.0;
        }
        if s == 0 {
            return 0.0; // No tokens processed, no error to measure
        }
        // Diagnostic reads batch slot 0 only. Callers must ensure bs==1
        // (gpu_tape_forward_summary guards this at entry).
        debug_assert!(
            batch_size <= 1,
            "dgd_delta_norm only reads batch slot 0; caller must ensure batch_size <= 1 (got {})",
            batch_size
        );
        let dd = d * d;

        // Helper: compute ‖M @ k - v‖₂ from GPU buffers.
        let compute_norm = |m_ptr: *const f32, k_ptr: *const f32, v_ptr: *const f32| -> f32 {
            let norm_out = GpuBuf::<f32>::zeros(1);
            unsafe {
                crate::cuda_ffi::dgd_delta_norm_cuda(m_ptr, k_ptr, v_ptr, norm_out.ptr(), d as i32);
            }
            crate::dispatch::cuda_sync();
            let mut host = [0.0f32; 1];
            norm_out.copy_to_host(&mut host);
            host[0]
        };

        match self {
            GpuMemoryCache::Delta { proxy: true, .. }
            | GpuMemoryCache::Titans { proxy: true, .. } => {
                // Proxy caches (pre-spec-43 legacy) don't store M states.
                return 0.0;
            }
            GpuMemoryCache::DeltaChunkwise { k_mem, v_mem, m_chunk_states, num_chunks, .. }
            | GpuMemoryCache::TitansChunkwise { k_mem, v_mem, m_chunk_states, num_chunks, .. } => {
                // Spec 43: use last chunk's M₀ as approximate pre-update M.
                // m_chunk_states layout: [(num_chunks+1)*dd]. Index (num_chunks-1) = last chunk start.
                if *num_chunks == 0 || s == 0 {
                    return 0.0;
                }
                let m_last_chunk_start = m_chunk_states.slice((*num_chunks - 1) * dd, dd);
                let k_last = k_mem.slice((s - 1) * d, d);
                let v_last = v_mem.slice((s - 1) * d, d);
                return compute_norm(m_last_chunk_start.as_ptr(), k_last.as_ptr(), v_last.as_ptr());
            }
            GpuMemoryCache::Delta { k_mem, v_mem, m_states, .. }
            | GpuMemoryCache::DGD { k_mem, v_mem, m_states, .. }
            | GpuMemoryCache::Titans { k_mem, v_mem, m_states, .. } => {
                // m_states layout: [bs*(s+1)*dd]. M_t is at offset t*dd.
                // Error e_{s-1} = M_{s-1} @ k_{s-1} - v_{s-1} uses the PRE-update M.
                let m_pre = m_states.slice((s - 1) * dd, dd);
                let k_last = k_mem.slice((s - 1) * d, d);
                let v_last = v_mem.slice((s - 1) * d, d);
                compute_norm(m_pre.as_ptr(), k_last.as_ptr(), v_last.as_ptr())
            }
            GpuMemoryCache::DeltaCkpt { k_mem, v_mem, m_checkpoints, checkpoint_interval, .. }
            | GpuMemoryCache::DGDCkpt { k_mem, v_mem, m_checkpoints, checkpoint_interval, .. }
            | GpuMemoryCache::TitansCkpt { k_mem, v_mem, m_checkpoints, checkpoint_interval, .. } => {
                // Checkpoints store M at boundaries: M_0, M_c, M_2c, ..., M_s.
                // Best available pre-update approximation: second-to-last checkpoint.
                // For most configs (s % c == 0), the last checkpoint IS M_s (post-update).
                // Use the second-to-last when available, else fall back to last.
                let num_ckpt = (s + checkpoint_interval - 1) / checkpoint_interval + 1;
                let ckpt_idx = if num_ckpt >= 2 { num_ckpt - 2 } else { 0 };
                let m_approx = m_checkpoints.slice(ckpt_idx * dd, dd);
                let k_last = k_mem.slice((s - 1) * d, d);
                let v_last = v_mem.slice((s - 1) * d, d);
                compute_norm(m_approx.as_ptr(), k_last.as_ptr(), v_last.as_ptr())
            }
            // TNT hierarchical — drill into shard inner caches and take max delta norm.
            // Each shard's inner cache is a Titans/Delta GpuMemoryCache with its own
            // M state. The inner cache batch dimension is n_batch (number of local
            // chunks per shard), but dgd_delta_norm reads batch slot 0 only.
            // We iterate all shards and take max — the last shard has the most context.
            GpuMemoryCache::TNT { shard_inner_caches, local_chunk_size, .. } => {
                if shard_inner_caches.is_empty() {
                    return 0.0;
                }
                let shard_s = *local_chunk_size;
                let mut max_delta = 0.0f32;
                for shard_cache in shard_inner_caches.iter() {
                    // Pass batch_size=1 — inner caches use n_batch (local chunk count)
                    // as their batch dim, but we only need slot 0's diagnostic.
                    let delta = shard_cache.dgd_delta_norm(shard_s, d, 1, num_heads);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                }
                max_delta
            }
            // MLP memory (MONETA/YAAD): delta norm not supported yet (needs MLP-specific norm)
            GpuMemoryCache::Mlp { .. } => 0.0,
            // Hebbian has no error vector; SwiGlu has no M state
            GpuMemoryCache::Hebbian { .. }
            | GpuMemoryCache::HebbianCkpt { .. }
            | GpuMemoryCache::SwiGlu { .. } => 0.0,
        }
    }

    /// Extract alpha (retention/forgetting gate) statistics from the forward cache.
    ///
    /// `alpha_floor` is the CS-39 configured floor — used to compute `frac_at_floor`.
    /// Returns `Some(GateStats)` for rules that have a learned alpha (Delta, Titans,
    /// Hebbian, DGD + Ckpt variants). Returns `None` for SwiGlu.
    /// For TNT: aggregates alpha values across all shard inner caches, filtering
    /// out zero-padded elements (sigmoid output is always > 0 for real tokens).
    ///
    /// Source: Titans (2501.00663) eq-013 — α_t = σ(w_α · [k,v] + b_α)
    pub fn alpha_stats(&self, alpha_floor: f32) -> Option<GateStats> {
        let alpha_buf: Option<&GpuBuf<f32>> = match self {
            GpuMemoryCache::Delta { alpha, .. }
            | GpuMemoryCache::Titans { alpha, .. }
            | GpuMemoryCache::Hebbian { alpha, .. }
            | GpuMemoryCache::DGD { alpha, .. }
            | GpuMemoryCache::DeltaCkpt { alpha, .. }
            | GpuMemoryCache::TitansCkpt { alpha, .. }
            | GpuMemoryCache::HebbianCkpt { alpha, .. }
            | GpuMemoryCache::DGDCkpt { alpha, .. }
            | GpuMemoryCache::Mlp { alpha, .. }
            | GpuMemoryCache::DeltaChunkwise { alpha, .. }
            | GpuMemoryCache::TitansChunkwise { alpha, .. } => Some(alpha),
            GpuMemoryCache::SwiGlu { .. } => None,
            GpuMemoryCache::TNT { .. } => None, // handled separately below
        };

        if let Some(buf) = alpha_buf {
            let n = buf.len();
            if n == 0 { return None; }
            let mut host = vec![0.0f32; n];
            buf.copy_to_host(&mut host);
            return Some(GateStats::from_slice(&host, alpha_floor, true));
        }

        // TNT: gather alpha values from all shard inner caches.
        if let GpuMemoryCache::TNT { shard_inner_caches, .. } = self {
            let mut all_alpha = Vec::new();
            for shard_cache in shard_inner_caches.iter() {
                let inner_buf: Option<&GpuBuf<f32>> = match shard_cache {
                    GpuMemoryCache::Delta { alpha, .. }
                    | GpuMemoryCache::Titans { alpha, .. }
                    | GpuMemoryCache::Hebbian { alpha, .. }
                    | GpuMemoryCache::DGD { alpha, .. }
                    | GpuMemoryCache::DeltaCkpt { alpha, .. }
                    | GpuMemoryCache::TitansCkpt { alpha, .. }
                    | GpuMemoryCache::HebbianCkpt { alpha, .. }
                    | GpuMemoryCache::DGDCkpt { alpha, .. } => Some(alpha),
                    _ => None,
                };
                if let Some(buf) = inner_buf {
                    let n = buf.len();
                    if n > 0 {
                        let mut host = vec![0.0f32; n];
                        buf.copy_to_host(&mut host);
                        // Filter padding zeros (sigmoid output is always > 0)
                        all_alpha.extend(host.iter().copied().filter(|&v| v > 0.0));
                    }
                }
            }
            if all_alpha.is_empty() { return None; }
            return Some(GateStats::from_slice(&all_alpha, alpha_floor, true));
        }

        None
    }

    /// Extract eta (momentum gate) statistics from the forward cache.
    ///
    /// Returns `Some(GateStats)` for Titans and TitansCkpt only (the only rules
    /// with a momentum accumulator S). Returns `None` for all other rules.
    /// For TNT: aggregates eta values across shard inner caches if they are Titans.
    ///
    /// Source: Titans (2501.00663) eq-014 — η_t = σ(w_η · [k,v] + b_η)
    pub fn eta_stats(&self) -> Option<GateStats> {
        let eta_buf: Option<&GpuBuf<f32>> = match self {
            GpuMemoryCache::Titans { eta, .. }
            | GpuMemoryCache::TitansCkpt { eta, .. }
            | GpuMemoryCache::TitansChunkwise { eta, .. } => Some(eta),
            GpuMemoryCache::Delta { .. }
            | GpuMemoryCache::DeltaChunkwise { .. }
            | GpuMemoryCache::Hebbian { .. }
            | GpuMemoryCache::DGD { .. }
            | GpuMemoryCache::DeltaCkpt { .. }
            | GpuMemoryCache::HebbianCkpt { .. }
            | GpuMemoryCache::DGDCkpt { .. }
            | GpuMemoryCache::Mlp { .. }
            | GpuMemoryCache::SwiGlu { .. } => None,
            GpuMemoryCache::TNT { .. } => None, // handled separately below
        };

        if let Some(buf) = eta_buf {
            let n = buf.len();
            if n == 0 { return None; }
            let mut host = vec![0.0f32; n];
            buf.copy_to_host(&mut host);
            // No configured bound for eta — pass 0.0 as floor (will yield frac_at_bound=0)
            return Some(GateStats::from_slice(&host, 0.0, true));
        }

        // TNT: gather eta values from Titans shard inner caches only.
        if let GpuMemoryCache::TNT { shard_inner_caches, .. } = self {
            let mut all_eta = Vec::new();
            for shard_cache in shard_inner_caches.iter() {
                let inner_buf: Option<&GpuBuf<f32>> = match shard_cache {
                    GpuMemoryCache::Titans { eta, .. }
                    | GpuMemoryCache::TitansCkpt { eta, .. } => Some(eta),
                    _ => None,
                };
                if let Some(buf) = inner_buf {
                    let n = buf.len();
                    if n > 0 {
                        let mut host = vec![0.0f32; n];
                        buf.copy_to_host(&mut host);
                        all_eta.extend(host.iter().copied().filter(|&v| v > 0.0));
                    }
                }
            }
            if all_eta.is_empty() { return None; }
            return Some(GateStats::from_slice(&all_eta, 0.0, true));
        }

        None
    }

    /// Extract theta (inner-loop learning rate) statistics from the forward cache.
    ///
    /// `theta_ceil` is the configured ceiling — used to compute `frac_at_ceil`.
    /// Returns `Some(ThetaStats)` for rules that have a learned theta (Delta, Titans, DGD),
    /// `None` for Hebbian/SwiGlu (which have no theta).
    /// For TNT: aggregates theta values across all shard inner caches, filtering
    /// out zero-padded elements (softplus output is always > 0 for real tokens).
    ///
    /// Source: HOPE (2512.24695) — theta = softplus(w_theta · [k,v] + b_theta)
    pub fn theta_stats(&self, theta_ceil: f32) -> Option<ThetaStats> {
        // Helper: extract theta buffer ref from any variant that has one
        let theta_buf: Option<&GpuBuf<f32>> = match self {
            GpuMemoryCache::Delta { theta, .. }
            | GpuMemoryCache::Titans { theta, .. }
            | GpuMemoryCache::DGD { theta, .. }
            | GpuMemoryCache::DeltaCkpt { theta, .. }
            | GpuMemoryCache::TitansCkpt { theta, .. }
            | GpuMemoryCache::DGDCkpt { theta, .. }
            | GpuMemoryCache::Mlp { theta, .. }
            | GpuMemoryCache::DeltaChunkwise { theta, .. }
            | GpuMemoryCache::TitansChunkwise { theta, .. } => Some(theta),
            GpuMemoryCache::Hebbian { .. }
            | GpuMemoryCache::HebbianCkpt { .. }
            | GpuMemoryCache::SwiGlu { .. } => None,
            GpuMemoryCache::TNT { .. } => None, // handled separately below
        };

        if let Some(buf) = theta_buf {
            let n = buf.len();
            if n == 0 { return None; }
            let mut host = vec![0.0f32; n];
            buf.copy_to_host(&mut host);
            return Some(ThetaStats::from_slice(&host, theta_ceil));
        }

        // TNT: gather theta values from all shard inner caches.
        // Filter out zeros from padding — softplus(x) > 0 for all real inputs,
        // so theta == 0.0 means the element is zero-padding from partial shards.
        if let GpuMemoryCache::TNT { shard_inner_caches, .. } = self {
            let mut all_theta = Vec::new();
            for shard_cache in shard_inner_caches.iter() {
                let inner_buf: Option<&GpuBuf<f32>> = match shard_cache {
                    GpuMemoryCache::Delta { theta, .. }
                    | GpuMemoryCache::Titans { theta, .. }
                    | GpuMemoryCache::DGD { theta, .. }
                    | GpuMemoryCache::DeltaCkpt { theta, .. }
                    | GpuMemoryCache::TitansCkpt { theta, .. }
                    | GpuMemoryCache::DGDCkpt { theta, .. } => Some(theta),
                    _ => None,
                };
                if let Some(buf) = inner_buf {
                    let n = buf.len();
                    if n > 0 {
                        let mut host = vec![0.0f32; n];
                        buf.copy_to_host(&mut host);
                        // Filter padding zeros (softplus output is always > 0)
                        all_theta.extend(host.iter().copied().filter(|&v| v > 0.0));
                    }
                }
            }
            if all_theta.is_empty() { return None; }
            return Some(ThetaStats::from_slice(&all_theta, theta_ceil));
        }

        None
    }
}

/// Statistics about the theta (inner-loop learning rate) distribution for one (block, level).
///
/// Theta is per-token, per-level: theta_t = softplus(w_theta · [k_t, v_t] + b_theta).
/// This struct summarizes the distribution across all tokens in one forward pass,
/// revealing whether theta_ceil is constraining the learned rate.
#[derive(Debug, Clone)]
pub struct ThetaStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
    /// Fraction of tokens where theta >= theta_ceil (hitting the configured ceiling)
    pub frac_at_ceil: f32,
}

impl ThetaStats {
    /// Compute stats from a sorted slice of theta values.
    /// `theta_ceil` is the configured ceiling for computing `frac_at_ceil`.
    pub fn from_slice(values: &[f32], theta_ceil: f32) -> Self {
        let n = values.len();
        assert!(n > 0);
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let sum: f32 = sorted.iter().sum();
        let mean = sum / n as f32;

        // Interpolated quantiles: rank = p * (n - 1), interpolate between neighbors
        let quantile = |p: f64| -> f32 {
            let rank = p * (n as f64 - 1.0);
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = rank - lo as f64;
            (sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac) as f32
        };
        let median = quantile(0.5);
        let p95 = quantile(0.95);
        let p99 = quantile(0.99);

        // Fraction at configured ceiling (with small epsilon for float comparison)
        let ceil_threshold = theta_ceil * (1.0 - 1e-4);
        let at_ceil = if theta_ceil < f32::MAX {
            sorted.iter().filter(|&&v| v >= ceil_threshold).count()
        } else {
            0 // no ceiling configured
        };
        let frac_at_ceil = at_ceil as f32 / n as f32;

        ThetaStats { count: n, min, max, mean, median, p95, p99, frac_at_ceil }
    }
}

/// Generic gate statistics for alpha (retention) and eta (momentum) gates.
///
/// Same interpolated-quantile approach as ThetaStats, but with directional
/// bound checking: floor for alpha (CS-39 alpha_floor), ceiling for theta.
/// Eta has no configured bound.
///
/// Source: Titans (2501.00663) eq-013 (alpha), eq-014 (eta)
///         Spec 26: specs/infrastructure/26_gate_diagnostics.md
#[derive(Debug, Clone)]
pub struct GateStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
    /// Fraction of tokens at the configured bound (floor or ceil).
    pub frac_at_bound: f32,
}

impl GateStats {
    /// Compute stats from a slice of gate values.
    ///
    /// `bound` is the configured floor or ceiling value.
    /// `bound_is_floor` controls direction:
    ///   - true (alpha): counts values ≤ bound × (1 + ε)
    ///   - false (theta-style): counts values ≥ bound × (1 - ε)
    /// Pass `bound=0.0` with `bound_is_floor=true` to get frac_at_bound=0 (no bound).
    pub fn from_slice(values: &[f32], bound: f32, bound_is_floor: bool) -> Self {
        let n = values.len();
        assert!(n > 0);
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let sum: f32 = sorted.iter().sum();
        let mean = sum / n as f32;

        let quantile = |p: f64| -> f32 {
            let rank = p * (n as f64 - 1.0);
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = rank - lo as f64;
            (sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac) as f32
        };
        let median = quantile(0.5);
        let p95 = quantile(0.95);
        let p99 = quantile(0.99);

        let at_bound = if bound_is_floor && bound > 0.0 {
            // Floor: count values ≤ bound × (1 + ε)
            let threshold = bound * (1.0 + 1e-4);
            sorted.iter().filter(|&&v| v <= threshold).count()
        } else if !bound_is_floor && bound < f32::MAX {
            // Ceiling: count values ≥ bound × (1 - ε)
            let threshold = bound * (1.0 - 1e-4);
            sorted.iter().filter(|&&v| v >= threshold).count()
        } else {
            0
        };
        let frac_at_bound = at_bound as f32 / n as f32;

        GateStats { count: n, min, max, mean, median, p95, p99, frac_at_bound }
    }
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

    // residual=true is now supported via CUDA LayerNorm kernels.
    assert!(s > 0, "seq_len must be > 0");
    // Derive batch_size from input length; default to 1 for single-sequence calls
    assert!(input_ids.len() >= s, "input_ids too short");
    let batch_size = input_ids.len() / s;
    assert!(batch_size >= 1);
    assert_eq!(input_ids.len(), batch_size * s, "input_ids must be multiple of seq_len");
    assert_eq!(target_ids.len(), batch_size * s, "target_ids must match input_ids length");
    assert_eq!(d, nh * hd);

    let bs = batch_size;  // shorthand
    // Pre-checked i32 conversions for CUDA kernel parameters (used throughout)
    let tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let d_i32      = i32::try_from(d).expect("d_model exceeds i32::MAX");
    let v_i32      = i32::try_from(v).expect("vocab_size exceeds i32::MAX");

    // Convert input_ids to i32 for CUDA kernels (flat: [batch_size * s])
    let input_ids_i32: Vec<i32> = input_ids.iter()
        .map(|&x| i32::try_from(x).expect("input token id overflows i32 — vocab_size exceeds i32::MAX"))
        .collect();
    let target_ids_i32: Vec<i32> = target_ids.iter()
        .map(|&x| i32::try_from(x).expect("target token id overflows i32 — vocab_size exceeds i32::MAX"))
        .collect();

    // ── CUDA Graph step accounting ─────────────────────────────────────
    // Advance the step counter and check for capture/replay opportunities.
    // Capture and replay only trigger when scratch buffers are allocated
    // (forward_scratch.is_some() requires cuda_graph_warmup > 0).
    context.cuda_graph.step();
    let bitmask = pulse_to_bitmask(pulse, cfg.k);

    // ── CUDA Graph capture phase (one-time at warmup_steps) ────────────
    if context.cuda_graph.should_capture() && context.forward_scratch.is_some() && cfg.swa.num_heads == 1 {
        // Check that the memory rule is supported for graph capture.
        let is_tnt_mode = cfg.parallel.as_ref()
            .map(|p| p.strategy == ParallelStrategy::TNTHierarchical)
            .unwrap_or(false);
        // Cycle-scoped eviction (spec 25) is TNT-only — already excluded by is_tnt_mode.
        // CUDA graph capture is only disabled here for explicit checkpoint_interval.
        // Use effective_checkpoint_interval() for consistency with the rest of the codebase.
        let has_ckpt = cfg.effective_checkpoint_interval(0).is_some();
        // Spec 27: CUDA graph scratch always allocates full trajectory —
        // proxy levels must use the standard dispatch path for VRAM savings.
        let has_proxy = (0..cfg.k).any(|l| cfg.tape_strategy_for_level(l) == LevelTapeStrategy::Proxy);
        let can_capture = !cfg.residual
            && matches!(cfg.memory_rule, MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM)
            && !has_ckpt
            && !is_tnt_mode
            && !has_proxy;

        if can_capture {
            gpu_cms_capture_all_patterns(params, cfg, &input_ids_i32, &target_ids_i32, pulse, context, bs);
        } else {
            // Rule not supported for graph capture — disable permanently.
            eprintln!("[cuda_graph] Capture skipped: rule {:?} not supported. Falling through to standard dispatch.", cfg.memory_rule);
            context.cuda_graph.disable();
        }
    }

    // ── CUDA Graph replay path ─────────────────────────────────────────
    if context.cuda_graph.should_replay(bitmask) && cfg.swa.num_heads == 1 {
        if let Some(loss) = gpu_cms_replay(params, cfg, &input_ids_i32, &target_ids_i32,
                                            pulse, context, bitmask, bs) {
            return loss;
        }
        // If replay returns None, fall through to standard dispatch.
    }

    // ── Standard dispatch path (warmup or graph disabled) ─────────────
    let d_input_ids = GpuBuf::<f32>::new(bs * s);
    let d_target_ids = GpuBuf::<f32>::new(bs * s);
    unsafe {
        let rc = gpu_buf_memcpy_h2d(
            d_input_ids.ptr() as *mut std::ffi::c_void,
            input_ids_i32.as_ptr() as *const std::ffi::c_void,
            bs * s * 4,
        );
        assert_eq!(rc, 0);
        let rc = gpu_buf_memcpy_h2d(
            d_target_ids.ptr() as *mut std::ffi::c_void,
            target_ids_i32.as_ptr() as *const std::ffi::c_void,
            bs * s * 4,
        );
        assert_eq!(rc, 0);
    }

    // ── Stage 1: Embedding gather on GPU ──────────────────────────────
    let embedded = GpuBuf::<f32>::zeros(bs * s * d);
    unsafe {
        crate::cuda_ffi::embedding_gather_cuda(
            params.swa.w_embed.as_ptr(),
            d_input_ids.ptr() as *const i32,
            embedded.ptr(),
            tokens_i32, d_i32,
        );
    }

    // ── Stage 2a: Pre-LN for attention (residual path) + QKV projections ──
    let n_tokens = bs * s;
    let (ln_attn_out, ln_attn_mean, ln_attn_rstd) = if cfg.residual {
        let out = GpuBuf::<f32>::zeros(n_tokens * d);
        let mean = GpuBuf::<f32>::zeros(n_tokens);
        let rstd = GpuBuf::<f32>::zeros(n_tokens);
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                embedded.as_ptr(),
                params.swa.ln_attn_gamma.as_ptr(),
                params.swa.ln_attn_beta.as_ptr(),
                out.ptr(), mean.ptr(), rstd.ptr(),
                n_tokens as i32, d_i32, 1e-5,
            );
        }
        (Some(out), Some(mean), Some(rstd))
    } else {
        (None, None, None)
    };

    // QKV source: LN output for residual path, raw embedded otherwise
    let qkv_source = if cfg.residual { ln_attn_out.as_ref().unwrap() } else { &embedded };
    let mut q_f32 = GpuBuf::zeros(bs * s * d);
    let mut k_f32 = GpuBuf::zeros(bs * s * d);
    let mut v_f32 = GpuBuf::zeros(bs * s * d);
    crate::dispatch::cublas_matmul_transb_dd(qkv_source, &params.swa.w_q, &mut q_f32, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(qkv_source, &params.swa.w_k, &mut k_f32, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(qkv_source, &params.swa.w_v, &mut v_f32, bs * s, d, d, 0.0);

    // ── Stage 3a: SWA attention (bf16 on GPU) ─────────────────────────
    let total = bs * s * d;
    let n_p = cfg.n_persistent;
    let aw_stride = n_p + ws;
    let aw_total = bs * nh * s * aw_stride;
    let total_i32 = i32::try_from(total).expect("bs*s*d exceeds i32::MAX");
    let q_bf16 = GpuBuf::<u16>::zeros(total);
    let k_bf16 = GpuBuf::<u16>::zeros(total);
    let v_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_out_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);

    // f32 → bf16 conversion on GPU
    unsafe {
        crate::cuda_ffi::f32_to_bf16_cuda(q_f32.as_ptr(), q_bf16.ptr(), total_i32);
        crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), k_bf16.ptr(), total_i32);
        crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), v_bf16.ptr(), total_i32);
    }

    // SWA forward kernel (bf16) — batch_size sequences in parallel
    crate::dispatch::swa_forward_dd(
        &q_bf16, &k_bf16, &v_bf16,
        &mut attn_out_bf16, &mut attn_weights_bf16,
        s, nh, hd, ws, bs, n_p,
    );

    // bf16 → f32 for attn_out (needed for gating)
    let attn_out = GpuBuf::<f32>::zeros(total);
    unsafe {
        crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), total_i32);
    }

    // ── Residual skip 1: embedded + attn_out ────────────────────────
    let (residual_after_attn, ln_mem_out, ln_mem_mean, ln_mem_rstd) = if cfg.residual {
        // residual = embedded + attn_out
        let residual = GpuBuf::<f32>::zeros(n_tokens * d);
        unsafe {
            // Copy embedded, then add attn_out
            crate::cuda_ffi::saxpy_cuda(0.0, embedded.as_ptr(), residual.ptr(), total_i32); // zero residual
        }
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, embedded.as_ptr(), residual.ptr(), total_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, attn_out.as_ptr(), residual.ptr(), total_i32);
        }
        // LN_mem on residual
        let ln_out = GpuBuf::<f32>::zeros(n_tokens * d);
        let mean = GpuBuf::<f32>::zeros(n_tokens);
        let rstd = GpuBuf::<f32>::zeros(n_tokens);
        unsafe {
            crate::cuda_ffi::layer_norm_forward_cuda(
                residual.as_ptr(),
                params.swa.ln_mem_gamma.as_ptr(),
                params.swa.ln_mem_beta.as_ptr(),
                ln_out.ptr(), mean.ptr(), rstd.ptr(),
                n_tokens as i32, d_i32, 1e-5,
            );
        }
        (Some(residual), Some(ln_out), Some(mean), Some(rstd))
    } else {
        (None, None, None, None)
    };

    // Memory branch input: LN(residual) for residual path, raw embedded otherwise
    let mem_input = if cfg.residual { ln_mem_out.as_ref().unwrap() } else { &embedded };

    // ── Stage 2b+3b: Memory branch per level ──────────────────────────
    let mut memory_caches = Vec::with_capacity(cfg.k);
    let mut y_per_level = Vec::with_capacity(cfg.k);

    // Check if TNT parallelization is active
    let is_tnt = cfg.parallel.as_ref()
        .map(|p| p.strategy == ParallelStrategy::TNTHierarchical)
        .unwrap_or(false);

    for level in 0..cfg.k {
        // SwiGLU is always active regardless of pulse (no M state to update).
        // Mirrors CPU path in mag.rs: let active = active || matches!(cfg.memory_rule, SwiGluMlp).
        let effective_active = pulse.active_levels[level]
            || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);
        if effective_active {
            if is_tnt
                && bs == 1
                && matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule)
            {
                // TNT path: shard-parallel memory processing via gpu_tnt_forward.
                let parallel_cfg = cfg.parallel.as_ref().unwrap();
                let (y_level, mem_cache) = gpu_tnt_forward(
                    &params.levels[level], cfg, mem_input,
                    &mut context.memory[level],
                    s, d, level, bs, parallel_cfg,
                );
                y_per_level.push(y_level);
                memory_caches.push(Some(mem_cache));
            } else {
            // Active level: compute projections, gates, and memory update on GPU.
            let (y_level, mem_cache) = gpu_memory_forward(
                &params.levels[level], cfg, mem_input,
                &mut context.memory[level],
                s, d, level, bs,
            );
            y_per_level.push(y_level);
            memory_caches.push(Some(mem_cache));
            }
        } else {
            // Frozen level: read-only M @ q_mem on GPU.
            if matches!(cfg.memory_rule, MemoryRuleKind::Moneta | MemoryRuleKind::YAAD) {
                y_per_level.push(frozen_mlp_fallback("gpu_cms_forward", level, cfg.memory_rule, bs * s * d));
                memory_caches.push(None);
            } else {
            // Each batch element has distinct embeddings, so compute Y = Q @ M^T
            // for all bs*s tokens simultaneously. Same frozen M for all batch elements.
            let y_level = gpu_memory_read_only(
                &params.levels[level], mem_input,
                &context.memory[level],
                bs * s, d, nh, hd,
            );
            y_per_level.push(y_level);
            memory_caches.push(None);
            }
        }
    }

    // ── Combine levels: y_combined = sum with 1/sqrt(k) for k>2 ───────
    let y_combined = GpuBuf::<f32>::zeros(bs * s * d);
    for y_level in &y_per_level {
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, y_level.as_ptr(), y_combined.ptr(), total_i32);
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        unsafe {
            crate::cuda_ffi::saxpy_cuda(scale - 1.0, y_combined.as_ptr(), y_combined.ptr(), total_i32);
        }
    }

    // ── Stage 4+5: Gating/residual → output projection ────────────────
    let gate = GpuBuf::<f32>::zeros(bs * s * d);
    let gated_out = GpuBuf::<f32>::zeros(bs * s * d);
    let mut projected = GpuBuf::<f32>::zeros(bs * s * d);

    let residual_final = if cfg.residual {
        // Residual path: residual_final = residual_after_attn + y_combined (additive, no sigmoid)
        let residual_attn = residual_after_attn.as_ref().unwrap();
        let res_final = GpuBuf::<f32>::zeros(n_tokens * d);
        unsafe {
            // res_final = residual_after_attn + y_combined
            crate::cuda_ffi::saxpy_cuda(1.0, residual_attn.as_ptr(), res_final.ptr(), total_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, y_combined.as_ptr(), res_final.ptr(), total_i32);
        }
        // Output projection on residual
        crate::dispatch::cublas_matmul_transb_dd(&res_final, &params.swa.w_o, &mut projected, bs * s, d, d, 0.0);
        // Residual backward never reads gate (additive gradient routing bypasses
        // sigmoid_backward entirely). Gate stays zeros for cache shape stability.
        // gated_out = attn_out for cache shape stability (also unused by residual backward).
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, attn_out.as_ptr(), gated_out.ptr(), total_i32);
        }
        Some(res_final)
    } else {
        // Legacy path: sigmoid gating
        unsafe {
            crate::cuda_ffi::sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), total_i32);
            crate::cuda_ffi::elemwise_mul_cuda(attn_out.as_ptr(), gate.as_ptr(), gated_out.ptr(), total_i32);
        }
        crate::dispatch::cublas_matmul_transb_dd(&gated_out, &params.swa.w_o, &mut projected, bs * s, d, d, 0.0);
        None
    };

    // ── Stage 6: Unembed (cuBLAS on GPU) ──────────────────────────────
    let mut logits = GpuBuf::<f32>::zeros(bs * s * v);
    crate::dispatch::cublas_matmul_dd(&projected, &params.swa.w_unembed, &mut logits, bs * s, d, v, 0.0);

    // ── Stage 7: Cross-entropy loss (GPU → scalar D2H) ────────────────
    let loss_gpu = GpuBuf::<f32>::zeros(1);
    unsafe {
        crate::cuda_ffi::cross_entropy_forward_cuda(
            logits.as_ptr(),
            d_target_ids.ptr() as *const i32,
            loss_gpu.ptr(),
            tokens_i32, v_i32,
        );
    }
    crate::dispatch::cuda_sync();

    // Download scalar loss (4 bytes D2H)
    let mut loss_host = [0.0f32; 1];
    loss_gpu.copy_to_host(&mut loss_host);
    // Mean over VALID tokens across all batch elements
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
        ln_attn_out, ln_attn_mean, ln_attn_rstd,
        ln_mem_out, ln_mem_mean, ln_mem_rstd,
        residual_after_attn,
        residual_final,
        projected, logits,
        pulse: pulse.clone(),
        s, d, v, nh, hd, ws,
        batch_size: bs,
    };

    (loss, cache)
}

// ══════════════════════════════════════════════════════════════════════
// Memory forward helpers (GPU-resident)
// ══════════════════════════════════════════════════════════════════════

/// Transpose [bs, s, nh*hd] → [bs*nh, s, hd] on GPU.
/// Spec 45: converts d_model-layout projections to per-head layout for memory kernels.
#[cfg(feature = "cuda")]
pub(crate) fn reshape_to_per_head(buf: &GpuBuf<f32>, bs: usize, s: usize, nh: usize, hd: usize) -> GpuBuf<f32> {
    let out = GpuBuf::zeros(bs * nh * s * hd);
    unsafe {
        crate::cuda_ffi::transpose_heads_f32_cuda(
            buf.as_ptr(), out.ptr(),
            bs as i32, s as i32, nh as i32, hd as i32, 1,
        );
    }
    out
}

/// Transpose [bs*nh, s, hd] → [bs, s, nh*hd] on GPU.
/// Spec 45: converts per-head memory output back to d_model layout.
#[cfg(feature = "cuda")]
pub(crate) fn reshape_from_per_head(buf: &GpuBuf<f32>, bs: usize, s: usize, nh: usize, hd: usize) -> GpuBuf<f32> {
    let out = GpuBuf::zeros(bs * s * nh * hd);
    unsafe {
        crate::cuda_ffi::transpose_heads_f32_cuda(
            buf.as_ptr(), out.ptr(),
            bs as i32, s as i32, nh as i32, hd as i32, 0,
        );
    }
    out
}

/// Broadcast [bs, s] → [bs*nh, s] by repeating gate values across heads.
/// Spec 45: gates are position-level signals, same across all heads.
#[cfg(feature = "cuda")]
pub(crate) fn broadcast_gates(buf: &GpuBuf<f32>, bs: usize, s: usize, nh: usize) -> GpuBuf<f32> {
    let out = GpuBuf::zeros(bs * nh * s);
    unsafe {
        crate::cuda_ffi::broadcast_heads_f32_cuda(
            buf.as_ptr(), out.ptr(),
            bs as i32, s as i32, nh as i32,
        );
    }
    out
}

/// Sum [bs*nh, s] → [bs, s] across heads (backward of broadcast_gates).
/// Spec 45: reduces per-head gate gradients back to position-level.
#[cfg(feature = "cuda")]
pub(crate) fn sum_gates_across_heads(buf: &GpuBuf<f32>, bs: usize, s: usize, nh: usize) -> GpuBuf<f32> {
    let out = GpuBuf::zeros(bs * s);
    unsafe {
        crate::cuda_ffi::sum_heads_f32_cuda(
            buf.as_ptr(), out.ptr(),
            bs as i32, s as i32, nh as i32,
        );
    }
    out
}

/// Compute memory projections + gates + inner loop for an active level, all on GPU.
///
/// `embedded` has shape [batch_size * s, d] (flat batch).
/// `context_m` is the carry-forward M state — broadcast to all batch elements,
/// then element-0's final M is written back after the forward pass.
///
/// Spec 45: When num_heads > 1, memory operates in per-head mode:
///   - M is num_heads × (head_dim × head_dim) instead of 1 × (d × d)
///   - Kernel grid = batch_size × num_heads (one SM per head)
///   - 12× less computation, 12× more SM utilization at d=768/12h
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_forward(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    cfg: &MAGConfig,
    embedded: &GpuBuf<f32>,
    context_m: &mut GpuBuf<f32>,   // carry-forward M state, updated with final M
    s: usize,
    d: usize,
    level: usize,
    batch_size: usize,
) -> (GpuBuf<f32>, GpuMemoryCache) {
    let bs = batch_size;
    let dd = d * d;  // d_model × d_model (used by fused path + DGD)
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let d_i32      = i32::try_from(d).expect("d_model exceeds i32::MAX");

    // Memory projections: k_mem, v_mem, q_mem = embedded @ W^T
    // embedded is [bs*s, d]; cuBLAS treats bs*s as M dimension
    let mut k_mem = GpuBuf::zeros(bs * s * d);
    let mut v_mem = GpuBuf::zeros(bs * s * d);
    let mut q_mem = GpuBuf::zeros(bs * s * d);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_k_mem, &mut k_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_v_mem, &mut v_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut q_mem, bs * s, d, d, 0.0);

    // ── Spec 45: Per-head dimensions for memory kernels ──────────────
    // Memory M is nh × (hd × hd) instead of 1 × (d × d).
    // dd_mem = per-head matrix size, bs_mem = batch folded with heads.
    let dd_mem = hd * hd;
    let bs_mem = bs * nh;
    let hd_i32 = i32::try_from(hd).expect("head_dim exceeds i32::MAX");

    // context_m is [bs_mem * dd_mem] — slot (b*nh+h)'s initial M_h at offset (b*nh+h)*dd_mem.
    let m_initial_slice = context_m.slice(0, bs_mem * dd_mem);
    let m_norm_max = cfg.max_m_norm(level);
    let eff_ckpt = cfg.effective_checkpoint_interval(level);
    let is_proxy = cfg.tape_strategy_for_level(level) == LevelTapeStrategy::Proxy;

    // Per-level gate clamp bounds
    let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
    let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
    let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
    let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);

    // ── Spec 39: Fused forward path ──────────────────────────────────
    // Fused kernels compute gates internally at d_model resolution —
    // incompatible with per-head memory (Spec 45). Disable fused path
    // when num_heads > 1; the unfused path handles per-head transpose.
    // TODO: re-fuse with per-head gate computation for nh > 1.
    let use_fused = eff_ckpt.is_none()
        && !is_proxy
        && nh == 1  // Spec 45: skip fused when per-head is active
        && matches!(cfg.memory_rule, MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM)
        // Spec 74: DGD fused kernel lacks per-token M-norm projection.
        // When m_norm_max is finite, DeltaRule must use the unfused path.
        && !(cfg.memory_rule == MemoryRuleKind::DeltaRule && m_norm_max < 1e30);

    if use_fused {
        // Allocate output buffers for gates and norms (produced by fused kernel)
        let mut alpha = GpuBuf::zeros(bs * s);
        let mut theta = GpuBuf::zeros(bs * s);
        let mut k_norms = GpuBuf::zeros(bs * s);
        let mut q_norms = GpuBuf::zeros(bs * s);

        match cfg.memory_rule {
            MemoryRuleKind::DeltaRule => {
                let mut m_states = GpuBuf::zeros(bs * (s + 1) * dd);
                let mut y = GpuBuf::zeros(bs * s * d);
                crate::dispatch::delta_fused_forward_dd(
                    &mut k_mem, &v_mem, &mut q_mem,
                    &level_params.w_alpha, &level_params.b_alpha,
                    &level_params.w_theta, &level_params.b_theta,
                    alpha_floor, alpha_ceil, theta_floor, theta_ceil,
                    &m_initial_slice,
                    &mut m_states, &mut y,
                    &mut alpha, &mut theta,
                    &mut k_norms, &mut q_norms,
                    s, d, bs, cfg.error_clip_for_level(level),
                );
                crate::dispatch::cuda_sync();
                copy_final_m_batch(&m_states, context_m, s, dd, bs);
                // Spec 65: batched clamp — one launch for all batch elements
                unsafe {
                    crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                        context_m.ptr(), d_i32, bs as i32, m_norm_max,
                    );
                }
                // Fused path is exact only (spec 43: proxy uses chunkwise, not fused)
                return (y, GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms, proxy: false });
            }
            MemoryRuleKind::TitansLMM => {
                let mut eta = GpuBuf::zeros(bs * s);
                let batch_s_initial = GpuBuf::zeros(bs * dd);
                let s_initial_slice = batch_s_initial.slice(0, bs * dd);
                let mut m_states = GpuBuf::zeros(bs * (s + 1) * dd);
                let mut s_states = GpuBuf::zeros(bs * (s + 1) * dd);
                let mut y = GpuBuf::zeros(bs * s * d);
                crate::dispatch::titans_fused_forward_dd(
                    &mut k_mem, &v_mem, &mut q_mem,
                    &level_params.w_alpha, &level_params.b_alpha,
                    &level_params.w_theta, &level_params.b_theta,
                    &level_params.w_eta, &level_params.b_eta,
                    alpha_floor, alpha_ceil, theta_floor, theta_ceil,
                    &m_initial_slice, &s_initial_slice,
                    &mut m_states, &mut s_states, &mut y,
                    &mut alpha, &mut theta, &mut eta,
                    &mut k_norms, &mut q_norms,
                    s, d, bs, cfg.error_clip_for_level(level), m_norm_max,
                );
                crate::dispatch::cuda_sync();
                copy_final_m_batch(&m_states, context_m, s, dd, bs);
                // Spec 65: batched clamp — one launch for all batch elements
                unsafe {
                    crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                        context_m.ptr(), d_i32, bs as i32, m_norm_max,
                    );
                }
                // Fused path is exact only (spec 43: proxy uses chunkwise, not fused)
                return (y, GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, k_norms, q_norms, proxy: false });
            }
            _ => unreachable!(),
        }
    }

    // ── Unfused path (checkpointed variants, Hebbian, chunkwise, SwiGLU MLP) ──
    // L2-normalize keys and queries (Titans paper: "normalize queries and keys using l_2-norm")
    let k_norms = GpuBuf::zeros(bs * s);
    let q_norms = GpuBuf::zeros(bs * s);
    unsafe {
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(k_mem.ptr(), k_norms.ptr(), tokens_i32, d_i32, 1e-8);
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(q_mem.ptr(), q_norms.ptr(), tokens_i32, d_i32, 1e-8);
    }

    // Compute per-token gates: alpha[bs*s], theta[bs*s]
    // b_alpha and b_theta are passed as device pointers (CUDA-graph-capture-safe):
    // the graph captures the stable pointer; optimizer updates the value in-place.
    let alpha = GpuBuf::zeros(bs * s);
    let theta = GpuBuf::zeros(bs * s);

    unsafe {
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_alpha.as_ptr(),
            level_params.b_alpha.as_ptr(), alpha.ptr(),
            tokens_i32, d_i32, 0, // 0=sigmoid
        );
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_theta.as_ptr(),
            level_params.b_theta.as_ptr(), theta.ptr(),
            tokens_i32, d_i32, 1, // 1=softplus
        );
        // CS-39: clamp post-sigmoid alpha to [floor, ceil] per level.
        if alpha_floor > 0.0 || alpha_ceil < 1.0 {
            crate::cuda_ffi::clamp_f32_cuda(alpha.ptr(), tokens_i32, alpha_floor, alpha_ceil);
        }
        // CS-39: clamp post-softplus theta to [floor, ceil] per level.
        if theta_floor > 0.0 || theta_ceil < f32::MAX {
            crate::cuda_ffi::clamp_f32_cuda(theta.ptr(), tokens_i32, theta_floor, theta_ceil);
        }
    }

    // ── Spec 45: Transpose to per-head layout ────────────────────────
    // k/v/q: [bs, s, d] → [bs*nh, s, hd]  (per-head recurrence)
    // gates:  [bs, s]   → [bs*nh, s]       (broadcast across heads)
    // k_norms/q_norms stay in d_model layout for backward L2 norm undo.
    let k_mem_ph = reshape_to_per_head(&k_mem, bs, s, nh, hd);
    let v_mem_ph = reshape_to_per_head(&v_mem, bs, s, nh, hd);
    let q_mem_ph = reshape_to_per_head(&q_mem, bs, s, nh, hd);
    let alpha_ph = broadcast_gates(&alpha, bs, s, nh);
    let theta_ph = broadcast_gates(&theta, bs, s, nh);

    match (eff_ckpt, cfg.memory_rule) {
        // ── Full-trajectory / chunkwise paths (checkpoint_interval=None) ──
        (None, MemoryRuleKind::DeltaRule) if is_proxy => {
            // Spec 43/44: chunkwise forward (frozen-M₀). Spec 45: per-head.
            let chunk_size = cfg.chunk_sizes.get(level).copied().unwrap_or(s);
            let num_chunks = (s + chunk_size - 1) / chunk_size;
            let mut m_chunk_states = GpuBuf::zeros(bs_mem * (num_chunks + 1) * dd_mem);
            let mut y_ph = GpuBuf::zeros(bs_mem * s * hd);
            if chunk_size > 1 {
                crate::dispatch::delta_chunkwise_forward_batched_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                    &m_initial_slice, &mut m_chunk_states, &mut y_ph,
                    s, hd, bs_mem, chunk_size, cfg.error_clip_for_level(level), m_norm_max,
                );
            } else {
                crate::dispatch::delta_chunkwise_forward_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                    &m_initial_slice, &mut m_chunk_states, &mut y_ph,
                    s, hd, bs_mem, chunk_size, cfg.error_clip_for_level(level), m_norm_max,
                );
            }
            crate::dispatch::cuda_sync();
            copy_final_m_batch(&m_chunk_states, context_m, num_chunks, dd_mem, bs_mem);
            // Spec 65: batched clamp — one launch for all batch-head elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::DeltaChunkwise { k_mem, v_mem, q_mem, alpha, theta, m_chunk_states, k_norms, q_norms, chunk_size, num_chunks })
        }
        (None, MemoryRuleKind::DeltaRule) => {
            // Exact: full per-token trajectory (Spec 45: per-head)
            let mut m_states = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
            let mut y_ph = GpuBuf::zeros(bs_mem * s * hd);
            crate::dispatch::delta_forward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph,
                &m_initial_slice, &mut m_states, &mut y_ph, s, hd, bs_mem,
                s, dd_mem, cfg.error_clip_for_level(level), m_norm_max,
            );
            crate::dispatch::cuda_sync();
            copy_final_m_batch(&m_states, context_m, s, dd_mem, bs_mem);
            // Spec 65: batched clamp — one launch for all batch-head elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms, proxy: false })
        }
        (None, MemoryRuleKind::TitansLMM) if is_proxy => {
            // Spec 43/44: chunkwise forward (frozen-M₀). Spec 45: per-head.
            let eta_dm = compute_eta(level_params, &k_mem, &v_mem, bs * s, d);
            let eta_ph = broadcast_gates(&eta_dm, bs, s, nh);
            let batch_s_initial = GpuBuf::zeros(bs_mem * dd_mem);
            let s_initial_slice = batch_s_initial.slice(0, bs_mem * dd_mem);
            let chunk_size = cfg.chunk_sizes.get(level).copied().unwrap_or(s);
            let num_chunks = (s + chunk_size - 1) / chunk_size;
            let mut m_chunk_states = GpuBuf::zeros(bs_mem * (num_chunks + 1) * dd_mem);
            let mut s_chunk_states = GpuBuf::zeros(bs_mem * (num_chunks + 1) * dd_mem);
            let mut y_ph = GpuBuf::zeros(bs_mem * s * hd);
            if chunk_size > 1 {
                crate::dispatch::titans_chunkwise_forward_batched_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                    &m_initial_slice, &s_initial_slice,
                    &mut m_chunk_states, &mut s_chunk_states, &mut y_ph,
                    s, hd, bs_mem, chunk_size, cfg.error_clip_for_level(level), m_norm_max,
                );
            } else {
                crate::dispatch::titans_chunkwise_forward_dd(
                    &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                    &m_initial_slice, &s_initial_slice,
                    &mut m_chunk_states, &mut s_chunk_states, &mut y_ph,
                    s, hd, bs_mem, chunk_size, cfg.error_clip_for_level(level), m_norm_max,
                );
            }
            crate::dispatch::cuda_sync();
            copy_final_m_batch(&m_chunk_states, context_m, num_chunks, dd_mem, bs_mem);
            // Spec 65: batched clamp — one launch for all batch-head elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::TitansChunkwise { k_mem, v_mem, q_mem, alpha, theta, eta: eta_dm, m_chunk_states, s_chunk_states, k_norms, q_norms, chunk_size, num_chunks })
        }
        (None, MemoryRuleKind::TitansLMM) => {
            // Exact: full per-token trajectory
            // Spec 45: eta computed at d_model resolution, then broadcast to per-head
            let eta_dm = compute_eta(level_params, &k_mem, &v_mem, bs * s, d);
            let eta_ph = broadcast_gates(&eta_dm, bs, s, nh);
            // Per-head: allocate with hd×hd matrices, bs*nh batch elements
            let batch_s_initial = GpuBuf::zeros(bs_mem * dd_mem);
            let s_initial_slice = batch_s_initial.slice(0, bs_mem * dd_mem);
            let mut m_states = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
            let mut s_states = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
            let mut y_ph = GpuBuf::zeros(bs_mem * s * hd);
            crate::dispatch::titans_forward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph, &theta_ph, &eta_ph,
                &m_initial_slice, &s_initial_slice,
                &mut m_states, &mut s_states, &mut y_ph, s, hd, bs_mem,
                s, dd_mem, cfg.error_clip_for_level(level), m_norm_max,
            );
            crate::dispatch::cuda_sync();
            // Copy per-head final M back to context_m
            copy_final_m_batch(&m_states, context_m, s, dd_mem, bs_mem);
            // Spec 65: batched clamp — one launch for all batch-head elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
                );
            }
            // Transpose y from [bs*nh, s, hd] → [bs, s, d]
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            // Cache: store d_model k/v/q (for projection grads) + per-head m/s_states
            (y, GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta: eta_dm, m_states, s_states, k_norms, q_norms, proxy: false })
        }
        (None, MemoryRuleKind::HebbianRule) => {
            assert_eq!(bs, 1, "Hebbian GPU forward with batch_size > 1 is not supported");
            // Spec 45: per-head memory, Spec 09: batched forward dispatch
            let mut m_states = GpuBuf::zeros(bs_mem * (s + 1) * dd_mem);
            let mut y_ph = GpuBuf::zeros(bs_mem * s * hd);
            crate::dispatch::hebbian_forward_dd(
                &k_mem_ph, &v_mem_ph, &q_mem_ph, &alpha_ph,
                &m_initial_slice, &mut m_states, &mut y_ph,
                s, hd, bs_mem, s, dd_mem,
            );
            crate::dispatch::cuda_sync();
            copy_final_m_batch(&m_states, context_m, s, dd_mem, bs_mem);
            // Spec 65: batched clamp — one launch for all batch-head elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states, k_norms, q_norms })
        }
        // ── Checkpointed paths (checkpoint_interval=Some(c)) ──
        // Gradient checkpointing with batch_size>1 is not supported — ablation configs
        // do not use checkpoint_interval, so this combination never occurs in practice.
        (Some(c), MemoryRuleKind::DeltaRule) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            // Spec 45: per-head — loop over nh heads (ckpt dispatch is single-batch)
            let num_ckpt = checkpoint_count(s, c);
            let m_checkpoints: GpuBuf<f32> = GpuBuf::zeros(nh * num_ckpt * dd_mem);
            let y_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            let error_clip = cfg.error_clip_for_level(level);
            for h in 0..nh {
                unsafe {
                    crate::cuda_ffi::delta_forward_ckpt_f32_cuda(
                        k_mem_ph.as_ptr().add(h * s * hd),
                        v_mem_ph.as_ptr().add(h * s * hd),
                        q_mem_ph.as_ptr().add(h * s * hd),
                        alpha_ph.as_ptr().add(h * s),
                        theta_ph.as_ptr().add(h * s),
                        m_initial_slice.as_ptr().add(h * dd_mem),
                        m_checkpoints.ptr().add(h * num_ckpt * dd_mem),
                        y_ph.ptr().add(h * s * hd),
                        s as i32, hd_i32, c as i32, error_clip,
                        m_norm_max,
                    );
                }
            }
            crate::dispatch::cuda_sync();
            // Copy final checkpoint for each head to context_m
            for h in 0..nh {
                let src_offset = h * num_ckpt * dd_mem + (num_ckpt - 1) * dd_mem;
                let dst_offset = h * dd_mem;
                unsafe {
                    let rc = gpu_buf_memcpy_d2d(
                        (context_m.ptr() as *mut u8).add(dst_offset * 4) as *mut std::ffi::c_void,
                        m_checkpoints.as_ptr().add(src_offset) as *const std::ffi::c_void,
                        dd_mem * 4,
                    );
                    assert_eq!(rc, 0, "copy final ckpt failed for head {h}");
                }
            }
            // Spec 65: batched clamp after all copies
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, nh as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        (Some(c), MemoryRuleKind::TitansLMM) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            // Spec 45: per-head — loop over nh heads (ckpt dispatch is single-batch)
            let eta_dm = compute_eta(level_params, &k_mem, &v_mem, s, d);
            let eta_ph = broadcast_gates(&eta_dm, bs, s, nh);
            let num_ckpt = checkpoint_count(s, c);
            let m_checkpoints: GpuBuf<f32> = GpuBuf::zeros(nh * num_ckpt * dd_mem);
            let s_checkpoints: GpuBuf<f32> = GpuBuf::zeros(nh * num_ckpt * dd_mem);
            let y_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            let s_initial_buf: GpuBuf<f32> = GpuBuf::zeros(dd_mem);
            let error_clip = cfg.error_clip_for_level(level);
            for h in 0..nh {
                unsafe {
                    crate::cuda_ffi::titans_forward_ckpt_f32_cuda(
                        k_mem_ph.as_ptr().add(h * s * hd),
                        v_mem_ph.as_ptr().add(h * s * hd),
                        q_mem_ph.as_ptr().add(h * s * hd),
                        alpha_ph.as_ptr().add(h * s),
                        theta_ph.as_ptr().add(h * s),
                        eta_ph.as_ptr().add(h * s),
                        m_initial_slice.as_ptr().add(h * dd_mem),
                        s_initial_buf.as_ptr(),  // zeros for each head
                        m_checkpoints.ptr().add(h * num_ckpt * dd_mem),
                        s_checkpoints.ptr().add(h * num_ckpt * dd_mem),
                        y_ph.ptr().add(h * s * hd),
                        s as i32, hd_i32, c as i32, error_clip,
                        m_norm_max,
                    );
                }
            }
            crate::dispatch::cuda_sync();
            for h in 0..nh {
                let src_offset = h * num_ckpt * dd_mem + (num_ckpt - 1) * dd_mem;
                let dst_offset = h * dd_mem;
                unsafe {
                    let rc = gpu_buf_memcpy_d2d(
                        (context_m.ptr() as *mut u8).add(dst_offset * 4) as *mut std::ffi::c_void,
                        m_checkpoints.as_ptr().add(src_offset) as *const std::ffi::c_void,
                        dd_mem * 4,
                    );
                    assert_eq!(rc, 0, "copy final ckpt failed for head {h}");
                }
            }
            // Spec 65: batched clamp after all copies
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, nh as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::TitansCkpt { k_mem, v_mem, q_mem, alpha, theta, eta: eta_dm, m_checkpoints, s_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        (Some(c), MemoryRuleKind::HebbianRule) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            // Spec 45: per-head — loop over nh heads (ckpt dispatch is single-batch)
            let num_ckpt = checkpoint_count(s, c);
            let m_checkpoints: GpuBuf<f32> = GpuBuf::zeros(nh * num_ckpt * dd_mem);
            let y_ph: GpuBuf<f32> = GpuBuf::zeros(nh * s * hd);
            for h in 0..nh {
                unsafe {
                    crate::cuda_ffi::hebbian_forward_ckpt_f32_cuda(
                        k_mem_ph.as_ptr().add(h * s * hd),
                        v_mem_ph.as_ptr().add(h * s * hd),
                        q_mem_ph.as_ptr().add(h * s * hd),
                        alpha_ph.as_ptr().add(h * s),
                        m_initial_slice.as_ptr().add(h * dd_mem),
                        m_checkpoints.ptr().add(h * num_ckpt * dd_mem),
                        y_ph.ptr().add(h * s * hd),
                        s as i32, hd_i32, c as i32,
                    );
                }
            }
            crate::dispatch::cuda_sync();
            for h in 0..nh {
                let src_offset = h * num_ckpt * dd_mem + (num_ckpt - 1) * dd_mem;
                let dst_offset = h * dd_mem;
                unsafe {
                    let rc = gpu_buf_memcpy_d2d(
                        (context_m.ptr() as *mut u8).add(dst_offset * 4) as *mut std::ffi::c_void,
                        m_checkpoints.as_ptr().add(src_offset) as *const std::ffi::c_void,
                        dd_mem * 4,
                    );
                    assert_eq!(rc, 0, "copy final ckpt failed for head {h}");
                }
            }
            // Spec 65: batched clamp after all copies
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context_m.ptr(), hd_i32, nh as i32, m_norm_max,
                );
            }
            let y = reshape_from_per_head(&y_ph, batch_size, s, nh, hd);
            (y, GpuMemoryCache::HebbianCkpt { k_mem, v_mem, q_mem, alpha, m_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        // ── SwiGLU: stateless MLP, no M state, no m_norm_clamp ──────────
        (_, MemoryRuleKind::SwiGluMlp) => {
            let inter = cfg.intermediate_size;
            let gate_buf  = GpuBuf::zeros(bs * s * inter);
            let up_buf    = GpuBuf::zeros(bs * s * inter);
            let fused_buf = GpuBuf::zeros(bs * s * inter);
            let cache_buf = GpuBuf::zeros(bs * s * inter);
            let y = GpuBuf::zeros(bs * s * d);
            unsafe {
                crate::cuda_ffi::swiglu_forward_f32_cuda_dd(
                    embedded.as_ptr(),
                    level_params.gate_proj.as_ptr(),
                    level_params.up_proj.as_ptr(),
                    level_params.down_proj.as_ptr(),
                    y.ptr(),
                    gate_buf.ptr(), up_buf.ptr(), fused_buf.ptr(), cache_buf.ptr(),
                    tokens_i32, d_i32,
                    i32::try_from(inter).expect("inter_dim exceeds i32::MAX"),
                );
            }
            // context_m is unused for SwiGLU (no M state) — not updated.
            (y, GpuMemoryCache::SwiGlu { gate_buf, up_buf, fused_buf, cache_buf })
        }
        // ── MLP memory: MONETA (l_p bias, L2/Lq retention) ─────────────
        (None, MemoryRuleKind::Moneta) => {
            assert_eq!(bs, 1, "MONETA GPU forward with batch_size > 1 is not supported");
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let required = w1_size + w2_size;
            assert!(
                context_m.len() >= required,
                "MONETA: context_m too small ({} elements) for MLP layout [W1|W2] = {} \
                 (d_hidden={}, d={}). GpuContextState must allocate d_hidden*d + d*d_hidden.",
                context_m.len(), required, dh, d,
            );
            // context_m layout: [w1_size + w2_size] = [W1 | W2]
            let w1_initial = context_m.slice(0, w1_size);
            let w2_initial = context_m.slice(w1_size, w2_size);
            let w1_states = GpuBuf::zeros((s + 1) * w1_size);
            let w2_states = GpuBuf::zeros((s + 1) * w2_size);
            let y = GpuBuf::zeros(s * d);
            let dh_i32 = i32::try_from(dh).expect("d_hidden exceeds i32::MAX");
            unsafe {
                crate::cuda_ffi::mlp_forward_lp_f32_cuda(
                    k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
                    alpha.as_ptr(), theta.as_ptr(),
                    w1_initial.as_ptr(), w2_initial.as_ptr(),
                    w1_states.ptr(), w2_states.ptr(), y.ptr(),
                    s as i32, d_i32, dh_i32,
                    cfg.lp_p, cfg.sign_sharpness, cfg.lambda_2, cfg.lq_q,
                );
            }
            crate::dispatch::cuda_sync();
            // Copy final W1/W2 back to context_m: [W1_final | W2_final]
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    context_m.ptr() as *mut std::ffi::c_void,
                    (w1_states.as_ptr() as *const u8).add(s * w1_size * 4) as *const std::ffi::c_void,
                    w1_size * 4,
                );
                assert_eq!(rc, 0, "copy final W1 failed");
                let rc = gpu_buf_memcpy_d2d(
                    (context_m.ptr() as *mut u8).add(w1_size * 4) as *mut std::ffi::c_void,
                    (w2_states.as_ptr() as *const u8).add(s * w2_size * 4) as *const std::ffi::c_void,
                    w2_size * 4,
                );
                assert_eq!(rc, 0, "copy final W2 failed");
            }
            // m_norm_clamp not applicable to MLP memory (different topology)
            (y, GpuMemoryCache::Mlp {
                k_mem, v_mem, q_mem, alpha, theta,
                w1_states, w2_states, k_norms, q_norms,
                w1_boundary: None, w2_boundary: None,
            })
        }
        // ── MLP memory: YAAD (Huber bias, decoupled L2 retention) ─────
        (None, MemoryRuleKind::YAAD) => {
            assert_eq!(bs, 1, "YAAD GPU forward with batch_size > 1 is not supported");
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let required = w1_size + w2_size;
            assert!(
                context_m.len() >= required,
                "YAAD: context_m too small ({} elements) for MLP layout [W1|W2] = {} \
                 (d_hidden={}, d={}). GpuContextState must allocate d_hidden*d + d*d_hidden.",
                context_m.len(), required, dh, d,
            );
            // context_m layout: [w1_size + w2_size] = [W1 | W2]
            let w1_initial = context_m.slice(0, w1_size);
            let w2_initial = context_m.slice(w1_size, w2_size);
            // YAAD boundary snapshots: chunk-start W1/W2 for decoupled retention
            let w1_boundary = GpuBuf::zeros(w1_size);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    w1_boundary.ptr() as *mut std::ffi::c_void,
                    context_m.as_ptr() as *const std::ffi::c_void,
                    w1_size * 4,
                );
                assert_eq!(rc, 0, "copy W1 boundary failed");
            }
            let w2_boundary = GpuBuf::zeros(w2_size);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    w2_boundary.ptr() as *mut std::ffi::c_void,
                    (context_m.as_ptr() as *const u8).add(w1_size * 4) as *const std::ffi::c_void,
                    w2_size * 4,
                );
                assert_eq!(rc, 0, "copy W2 boundary failed");
            }
            let w1_states = GpuBuf::zeros((s + 1) * w1_size);
            let w2_states = GpuBuf::zeros((s + 1) * w2_size);
            let y = GpuBuf::zeros(s * d);
            let dh_i32 = i32::try_from(dh).expect("d_hidden exceeds i32::MAX");
            unsafe {
                crate::cuda_ffi::mlp_forward_huber_f32_cuda(
                    k_mem.as_ptr(), v_mem.as_ptr(), q_mem.as_ptr(),
                    alpha.as_ptr(), theta.as_ptr(),
                    w1_initial.as_ptr(), w2_initial.as_ptr(),
                    w1_boundary.as_ptr(), w2_boundary.as_ptr(),
                    w1_states.ptr(), w2_states.ptr(), y.ptr(),
                    s as i32, d_i32, dh_i32,
                    cfg.delta, cfg.lambda_local, cfg.lambda_2,
                );
            }
            crate::dispatch::cuda_sync();
            // Copy final W1/W2 back to context_m: [W1_final | W2_final]
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    context_m.ptr() as *mut std::ffi::c_void,
                    (w1_states.as_ptr() as *const u8).add(s * w1_size * 4) as *const std::ffi::c_void,
                    w1_size * 4,
                );
                assert_eq!(rc, 0, "copy final W1 failed");
                let rc = gpu_buf_memcpy_d2d(
                    (context_m.ptr() as *mut u8).add(w1_size * 4) as *mut std::ffi::c_void,
                    (w2_states.as_ptr() as *const u8).add(s * w2_size * 4) as *const std::ffi::c_void,
                    w2_size * 4,
                );
                assert_eq!(rc, 0, "copy final W2 failed");
            }
            (y, GpuMemoryCache::Mlp {
                k_mem, v_mem, q_mem, alpha, theta,
                w1_states, w2_states, k_norms, q_norms,
                w1_boundary: Some(w1_boundary), w2_boundary: Some(w2_boundary),
            })
        }
        (Some(_), MemoryRuleKind::Moneta) | (Some(_), MemoryRuleKind::YAAD) => {
            panic!(
                "Checkpointed (gradient-checkpoint) MLP forward is not supported for {:?}. \
                 Use checkpoint_interval=None (full trajectory) for Moneta/YAAD levels.",
                cfg.memory_rule,
            );
        }
        _ => panic!("GPU-resident forward: unsupported (checkpoint, rule) combination. Got {:?}", cfg.memory_rule),
    }
}

// ══════════════════════════════════════════════════════════════════════
// TNT GPU forward — chunkwise parallelism via shard loop
// ══════════════════════════════════════════════════════════════════════

/// TNT GPU forward: split sequence into shards, run N local memories in parallel
/// per shard using existing Titans/Delta kernels with batch_size=N.
///
/// Only supports matrix-based rules (DeltaRule, TitansLMM) —
/// these are the rules with CUDA kernels that support batch_size.
///
/// Shard loop is sequential in Rust (O(seq_len/C_G) iterations, typically 8-16).
/// Within each shard, N local memories run in parallel via the batched kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_tnt_forward(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    cfg: &MAGConfig,
    embedded: &GpuBuf<f32>,       // [bs*s, d] — full sequence embeddings
    context_m: &mut GpuBuf<f32>,  // [d*d] — carry-forward global M (bs=1 for TNT)
    s: usize,                      // seq_len
    d: usize,
    level: usize,
    batch_size: usize,
    parallel_cfg: &crate::parallel::ParallelConfig,
) -> (GpuBuf<f32>, GpuMemoryCache) {
    // TNT operates on single sequences (batch_size=1 at the outer level).
    // The inner batch dimension is n_locals (N parallel local memories).
    assert_eq!(batch_size, 1, "TNT GPU forward requires batch_size=1 (TNT uses inner batch for N local memories)");
    assert!(
        matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM | MemoryRuleKind::DeltaRule),
        "TNT GPU forward only supports TitansLMM and DeltaRule, got {:?}", cfg.memory_rule,
    );

    // ── Spec 51: Per-head dimensions for TNT memory kernels ──────────
    // Same pattern as gpu_memory_forward (Spec 45): fold heads into batch.
    let nh = cfg.swa.num_heads;
    let hd = d / nh;
    let dd_mem = hd * hd;
    let bs_mem = batch_size * nh;  // bs=1 for TNT, so bs_mem = nh
    let hd_i32 = i32::try_from(hd).expect("head_dim exceeds i32::MAX");
    assert!(
        context_m.len() >= bs_mem * dd_mem,
        "context_m layout mismatch: expected >= {} (bs_mem={} * dd_mem={}), got {}",
        bs_mem * dd_mem, bs_mem, dd_mem, context_m.len(),
    );

    let cg = parallel_cfg.tnt_global_chunk_size;
    let cl = parallel_cfg.tnt_local_chunk_size;
    assert!(cl <= cg && cl >= 1 && cg >= 1);

    let num_shards = (s + cg - 1) / cg;
    let d_i32 = i32::try_from(d).expect("d exceeds i32::MAX");
    let m_norm_max = cfg.max_m_norm(level);

    // Full output buffer: [s, d]
    let y_full = GpuBuf::<f32>::zeros(s * d);

    // Spec 25: cycle-scoped cache retention — only keep the last N shards' inner caches.
    // Summaries are kept for ALL shards (O(d) each, needed for global backward).
    let max_retained = cfg.retained_shards(cg);
    let mut shard_inner_caches = Vec::with_capacity(max_retained.min(num_shards));
    let mut k_summaries = Vec::with_capacity(num_shards);
    let mut v_summaries = Vec::with_capacity(num_shards);
    let mut n_shards_dropped: usize = 0;

    for shard_idx in 0..num_shards {
        let shard_start = shard_idx * cg;
        let shard_end = (shard_start + cg).min(s);
        let shard_len = shard_end - shard_start;

        // Number of local chunks in this shard
        let n_batch = (shard_len + cl - 1) / cl;

        // Step 1: Broadcast global M → N copies for local memories
        // Spec 51: per-head broadcast — context_m is [nh * dd_mem], broadcast each
        // head's M to n_batch copies → [nh * n_batch * dd_mem] = [nh*n_batch, dd_mem]
        let mut m_broadcast = GpuBuf::<f32>::zeros(bs_mem * n_batch * dd_mem);
        for h in 0..bs_mem {
            for b in 0..n_batch {
                unsafe {
                    let rc = gpu_buf_memcpy_d2d(
                        (m_broadcast.ptr() as *mut u8).add((h * n_batch + b) * dd_mem * 4) as *mut std::ffi::c_void,
                        (context_m.as_ptr() as *const u8).add(h * dd_mem * 4) as *const std::ffi::c_void,
                        dd_mem * 4,
                    );
                    assert_eq!(rc, 0, "TNT per-head M broadcast D2D memcpy failed (rc={rc})");
                }
            }
        }

        // Step 2: Compute memory projections for the shard's tokens
        let shard_embedded_slice = embedded.slice(shard_start * d, shard_len * d);
        let shard_tokens_i32 = i32::try_from(shard_len).expect("shard_len exceeds i32::MAX");

        let mut k_mem = GpuBuf::zeros(shard_len * d);
        let mut v_mem = GpuBuf::zeros(shard_len * d);
        let mut q_mem = GpuBuf::zeros(shard_len * d);

        // Copy shard embeddings to owned buffer for cuBLAS
        let shard_embedded = GpuBuf::<f32>::zeros(shard_len * d);
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                shard_embedded.ptr() as *mut std::ffi::c_void,
                shard_embedded_slice.as_ptr() as *const std::ffi::c_void,
                shard_len * d * 4,
            );
            assert_eq!(rc, 0, "TNT shard D2D memcpy failed (rc={rc})");
        }

        crate::dispatch::cublas_matmul_transb_dd(&shard_embedded, &level_params.w_k_mem, &mut k_mem, shard_len, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&shard_embedded, &level_params.w_v_mem, &mut v_mem, shard_len, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&shard_embedded, &level_params.w_q_mem, &mut q_mem, shard_len, d, d, 0.0);

        // L2-normalize keys and queries (Titans paper: "normalize queries and keys using l_2-norm")
        let shard_k_norms = GpuBuf::zeros(shard_len);
        let shard_q_norms = GpuBuf::zeros(shard_len);
        unsafe {
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(k_mem.ptr(), shard_k_norms.ptr(), shard_tokens_i32, d_i32, 1e-8);
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(q_mem.ptr(), shard_q_norms.ptr(), shard_tokens_i32, d_i32, 1e-8);
        }

        // Step 3: Compute ALL gates for shard tokens in d-space (before per-head reshape)
        let alpha = GpuBuf::zeros(shard_len);
        let theta = GpuBuf::zeros(shard_len);
        // Pre-compute eta for Titans (needs d-space k/v, must happen before reshape)
        let eta_opt = if matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM) {
            let eta = GpuBuf::zeros(shard_len);
            unsafe {
                crate::cuda_ffi::gate_compute_cuda(
                    k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_eta.as_ptr(),
                    level_params.b_eta.as_ptr(), eta.ptr(),
                    shard_tokens_i32, d_i32, 0,
                );
            }
            Some(eta)
        } else {
            None
        };
        unsafe {
            crate::cuda_ffi::gate_compute_cuda(
                k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_alpha.as_ptr(),
                level_params.b_alpha.as_ptr(), alpha.ptr(),
                shard_tokens_i32, d_i32, 0,
            );
            crate::cuda_ffi::gate_compute_cuda(
                k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_theta.as_ptr(),
                level_params.b_theta.as_ptr(), theta.ptr(),
                shard_tokens_i32, d_i32, 1,
            );
            let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
            let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
            if alpha_floor > 0.0 || alpha_ceil < 1.0 {
                crate::cuda_ffi::clamp_f32_cuda(alpha.ptr(), shard_tokens_i32, alpha_floor, alpha_ceil);
            }
            let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
            let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
            if theta_floor > 0.0 || theta_ceil < f32::MAX {
                crate::cuda_ffi::clamp_f32_cuda(theta.ptr(), shard_tokens_i32, theta_floor, theta_ceil);
            }
        }

        // ── Spec 51: Per-head reshape (same pattern as gpu_memory_forward) ──
        // k/v/q: [1, shard_len, d] → [nh, shard_len, hd]
        // gates:  [1, shard_len]   → [nh, shard_len]
        let k_mem_ph = reshape_to_per_head(&k_mem, 1, shard_len, nh, hd);
        let v_mem_ph = reshape_to_per_head(&v_mem, 1, shard_len, nh, hd);
        let q_mem_ph = reshape_to_per_head(&q_mem, 1, shard_len, nh, hd);
        let alpha_ph = broadcast_gates(&alpha, 1, shard_len, nh);
        let theta_ph = broadcast_gates(&theta, 1, shard_len, nh);
        let eta_ph = eta_opt.map(|e| broadcast_gates(&e, 1, shard_len, nh));

        // Step 4: Pad to [bs_mem * n_batch, cl, hd] layout for batched kernel
        // Per-head data: [nh, shard_len, hd] padded to [nh, padded_len, hd] = [nh*n_batch, cl, hd]
        let padded_len = n_batch * cl;
        let ph_padded_elems = bs_mem * padded_len; // nh * padded_len
        let (k_mem_b, v_mem_b, q_mem_b, alpha_b, theta_b, eta_b_opt) = if shard_len == padded_len {
            (k_mem_ph, v_mem_ph, q_mem_ph, alpha_ph, theta_ph, eta_ph)
        } else {
            let kp = GpuBuf::zeros(ph_padded_elems * hd);
            let vp = GpuBuf::zeros(ph_padded_elems * hd);
            let qp = GpuBuf::zeros(ph_padded_elems * hd);
            let ap = GpuBuf::zeros(ph_padded_elems);
            let tp = GpuBuf::zeros(ph_padded_elems);
            let ep = eta_ph.as_ref().map(|_| GpuBuf::zeros(ph_padded_elems));
            unsafe {
                // Per-head layout: each head's shard_len tokens are contiguous.
                // Pad each head's block from shard_len to padded_len.
                for h in 0..bs_mem {
                    let src_off = h * shard_len;
                    let dst_off = h * padded_len;
                    let rc = gpu_buf_memcpy_d2d(
                        (kp.ptr() as *mut u8).add(dst_off * hd * 4) as *mut _,
                        (k_mem_ph.as_ptr() as *const u8).add(src_off * hd * 4) as *const _,
                        shard_len * hd * 4,
                    );
                    assert_eq!(rc, 0, "TNT pad copy kp failed (rc={rc})");
                    let rc = gpu_buf_memcpy_d2d(
                        (vp.ptr() as *mut u8).add(dst_off * hd * 4) as *mut _,
                        (v_mem_ph.as_ptr() as *const u8).add(src_off * hd * 4) as *const _,
                        shard_len * hd * 4,
                    );
                    assert_eq!(rc, 0, "TNT pad copy vp failed (rc={rc})");
                    let rc = gpu_buf_memcpy_d2d(
                        (qp.ptr() as *mut u8).add(dst_off * hd * 4) as *mut _,
                        (q_mem_ph.as_ptr() as *const u8).add(src_off * hd * 4) as *const _,
                        shard_len * hd * 4,
                    );
                    assert_eq!(rc, 0, "TNT pad copy qp failed (rc={rc})");
                    let rc = gpu_buf_memcpy_d2d(
                        (ap.ptr() as *mut u8).add(dst_off * 4) as *mut _,
                        (alpha_ph.as_ptr() as *const u8).add(src_off * 4) as *const _,
                        shard_len * 4,
                    );
                    assert_eq!(rc, 0, "TNT pad copy alpha failed (rc={rc})");
                    let rc = gpu_buf_memcpy_d2d(
                        (tp.ptr() as *mut u8).add(dst_off * 4) as *mut _,
                        (theta_ph.as_ptr() as *const u8).add(src_off * 4) as *const _,
                        shard_len * 4,
                    );
                    assert_eq!(rc, 0, "TNT pad copy theta failed (rc={rc})");
                    if let Some(ref eta_src) = eta_ph {
                        let ep_ref = ep.as_ref().unwrap();
                        let rc = gpu_buf_memcpy_d2d(
                            (ep_ref.ptr() as *mut u8).add(dst_off * 4) as *mut _,
                            (eta_src.as_ptr() as *const u8).add(src_off * 4) as *const _,
                            shard_len * 4,
                        );
                        assert_eq!(rc, 0, "TNT pad copy eta failed (rc={rc})");
                    }
                }
            }
            (kp, vp, qp, ap, tp, ep)
        };

        // Step 5: Run the batched memory kernel
        // Spec 51: batch = n_batch * nh (heads folded into batch), dim = hd, dd = dd_mem
        let kernel_batch = n_batch * bs_mem;  // n_batch * nh
        let m_initial_slice = m_broadcast.slice(0, kernel_batch * dd_mem);
        let mut y_local = GpuBuf::zeros(ph_padded_elems * hd);  // [nh*padded_len, hd]

        let is_proxy = cfg.tape_strategy_for_level(level) == LevelTapeStrategy::Proxy;

        let inner_cache = match cfg.memory_rule {
            MemoryRuleKind::TitansLMM => {
                let eta_b = eta_b_opt.expect("eta must be pre-computed for TitansLMM");

                let s_initial_buf = GpuBuf::zeros(kernel_batch * dd_mem);
                let s_initial_slice = s_initial_buf.slice(0, kernel_batch * dd_mem);
                let mut m_states = GpuBuf::zeros(kernel_batch * (cl + 1) * dd_mem);
                let mut s_states = GpuBuf::zeros(kernel_batch * (cl + 1) * dd_mem);
                crate::dispatch::titans_forward_dd(
                    &k_mem_b, &v_mem_b, &q_mem_b,
                    &alpha_b, &theta_b, &eta_b,
                    &m_initial_slice, &s_initial_slice,
                    &mut m_states, &mut s_states, &mut y_local, cl, hd, kernel_batch,
                    cl, dd_mem, cfg.error_clip_for_level(level), m_norm_max,
                );

                if is_proxy {
                    let m_final = GpuBuf::zeros(kernel_batch * dd_mem);
                    let s_final = GpuBuf::zeros(kernel_batch * dd_mem);
                    for b in 0..kernel_batch {
                        unsafe {
                            let rc = gpu_buf_memcpy_d2d(
                                (m_final.ptr() as *mut u8).add(b * dd_mem * 4) as *mut _,
                                (m_states.as_ptr() as *const u8).add((b * (cl + 1) + cl) * dd_mem * 4) as *const _,
                                dd_mem * 4,
                            );
                            assert_eq!(rc, 0, "TNT proxy M_final copy failed");
                            let rc = gpu_buf_memcpy_d2d(
                                (s_final.ptr() as *mut u8).add(b * dd_mem * 4) as *mut _,
                                (s_states.as_ptr() as *const u8).add((b * (cl + 1) + cl) * dd_mem * 4) as *const _,
                                dd_mem * 4,
                            );
                            assert_eq!(rc, 0, "TNT proxy S_final copy failed");
                        }
                    }
                    GpuMemoryCache::Titans {
                        k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                        alpha: alpha_b, theta: theta_b, eta: eta_b,
                        m_states: m_final, s_states: s_final,
                        k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                        proxy: true,
                    }
                } else {
                    GpuMemoryCache::Titans {
                        k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                        alpha: alpha_b, theta: theta_b, eta: eta_b,
                        m_states, s_states,
                        k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                        proxy: false,
                    }
                }
            }
            MemoryRuleKind::DeltaRule => {
                let mut m_states = GpuBuf::zeros(kernel_batch * (cl + 1) * dd_mem);
                crate::dispatch::delta_forward_dd(
                    &k_mem_b, &v_mem_b, &q_mem_b,
                    &alpha_b, &theta_b,
                    &m_initial_slice, &mut m_states, &mut y_local, cl, hd, kernel_batch,
                    cl, dd_mem, cfg.error_clip_for_level(level), m_norm_max,
                );

                if is_proxy {
                    let m_final = GpuBuf::zeros(kernel_batch * dd_mem);
                    for b in 0..kernel_batch {
                        unsafe {
                            let rc = gpu_buf_memcpy_d2d(
                                (m_final.ptr() as *mut u8).add(b * dd_mem * 4) as *mut _,
                                (m_states.as_ptr() as *const u8).add((b * (cl + 1) + cl) * dd_mem * 4) as *const _,
                                dd_mem * 4,
                            );
                            assert_eq!(rc, 0, "TNT proxy M_final copy failed");
                        }
                    }
                    GpuMemoryCache::Delta {
                        k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                        alpha: alpha_b, theta: theta_b, m_states: m_final,
                        k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                        proxy: true,
                    }
                } else {
                    GpuMemoryCache::Delta {
                        k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                        alpha: alpha_b, theta: theta_b, m_states,
                        k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                        proxy: false,
                    }
                }
            }
            _ => unreachable!(), // asserted above
        };

        crate::dispatch::cuda_sync();

        // Step 6: Reshape per-head output back to d-space and copy to y_full
        // y_local is [nh*n_batch, cl, hd] → reshape to [padded_len, d] then copy shard_len*d
        let y_dm = reshape_from_per_head(&y_local, 1, padded_len, nh, hd);
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                (y_full.ptr() as *mut u8).add(shard_start * d * 4) as *mut std::ffi::c_void,
                y_dm.as_ptr() as *const std::ffi::c_void,
                shard_len * d * 4,
            );
            assert_eq!(rc, 0, "TNT y_full D2D memcpy failed (rc={rc})");
        }

        // Step 7: Per-head shard summary (mean-pooling per head on GPU)
        // Spec 51: compute [nh, hd] summaries, stored as [nh*hd] = [d] for backward compat.
        // Each head h's unpadded output is at y_local[h*padded_len*hd .. h*padded_len*hd + shard_len*hd].
        #[allow(unused_mut)]
        let mut k_sum = GpuBuf::<f32>::zeros(d);  // [nh * hd] = [d]
        #[allow(unused_mut)]
        let mut v_sum = GpuBuf::<f32>::zeros(d);
        for h in 0..bs_mem {
            let h_y = GpuBuf::<f32>::zeros(shard_len * hd);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    h_y.ptr() as *mut std::ffi::c_void,
                    (y_local.as_ptr() as *const u8).add(h * padded_len * hd * 4) as *const std::ffi::c_void,
                    shard_len * hd * 4,
                );
                assert_eq!(rc, 0, "TNT per-head summary copy failed (rc={rc})");
            }
            // Mean-pool head h's output into k_sum[h*hd..(h+1)*hd] and v_sum[h*hd..(h+1)*hd]
            let mut k_h = GpuBuf::<f32>::zeros(hd);
            let mut v_h = GpuBuf::<f32>::zeros(hd);
            crate::dispatch::tnt_shard_summary_mean_dd(&h_y, &mut k_h, &mut v_h, shard_len, hd);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(
                    (k_sum.ptr() as *mut u8).add(h * hd * 4) as *mut _,
                    k_h.as_ptr() as *const _,
                    hd * 4,
                );
                assert_eq!(rc, 0, "TNT k_sum copy failed");
                let rc = gpu_buf_memcpy_d2d(
                    (v_sum.ptr() as *mut u8).add(h * hd * 4) as *mut _,
                    v_h.as_ptr() as *const _,
                    hd * 4,
                );
                assert_eq!(rc, 0, "TNT v_sum copy failed");
            }
        }

        // Step 8: Per-head global M update via outer product
        // Spec 51: each head's M_h updated independently with head-specific k_sum_h, v_sum_h
        for h in 0..bs_mem {
            unsafe {
                crate::cuda_ffi::tnt_global_update_f32_cuda(
                    (context_m.ptr() as *mut u8).add(h * dd_mem * 4) as *mut f32,
                    (k_sum.as_ptr() as *const u8).add(h * hd * 4) as *const f32,
                    (v_sum.as_ptr() as *const u8).add(h * hd * 4) as *const f32,
                    hd_i32, 0.95,
                );
            }
        }
        crate::dispatch::cuda_sync();

        // Spec 65: batched clamp — one launch for all batch-head elements
        unsafe {
            crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                context_m.ptr(), hd_i32, bs_mem as i32, m_norm_max,
            );
        }

        // Save caches for backward — inner caches are cycle-scoped,
        // summaries are kept for all shards (global backward needs them).
        shard_inner_caches.push(inner_cache);
        k_summaries.push(k_sum);
        v_summaries.push(v_sum);

        // Spec 25: rolling eviction — drop oldest inner cache when retention window exceeded.
        // Summaries are NOT evicted (O(d) each, needed for global M backward).
        while shard_inner_caches.len() > max_retained {
            shard_inner_caches.remove(0);
            n_shards_dropped += 1;
        }
    }

    let first_retained = n_shards_dropped;
    (y_full, GpuMemoryCache::TNT {
        shard_inner_caches, k_summaries, v_summaries,
        global_chunk_size: cg, local_chunk_size: cl,
        total_shards: num_shards, first_retained_shard: first_retained,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Scratch-based forward helpers (CUDA graph capture/replay)
// ══════════════════════════════════════════════════════════════════════

/// Run memory kernel dispatch for one active level, writing into pre-allocated scratch.
///
/// Unlike `gpu_memory_forward`, this function:
///   - Uses pre-allocated `GpuLevelScratch` buffers with FIXED device addresses
///   - Does NOT call `copy_final_m_batch` (caller does that outside the captured graph)
///   - Returns nothing; all outputs are in scratch fields
///
/// The caller wraps this in `begin_capture` / `end_capture` to build a CUDA graph.
/// During CUDA graph replay, all kernel launches recorded here are re-executed
/// using the same (stable) scratch buffer pointers.
///
/// Supported rules: DeltaRule, TitansLMM. Other rules fall back to standard dispatch.
#[cfg(feature = "cuda")]
/// Returns `false` for unsupported rule/checkpoint combos — caller must abort capture and
/// fall back to standard dispatch. Never panics so a misconfigured rule does not
/// hard-abort during a GPU capture window.
fn gpu_memory_forward_into_scratch(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    cfg: &MAGConfig,
    embedded: &GpuBuf<f32>,
    context_m: &GpuBuf<f32>,   // [bs*d*d] — carry-forward (read-only here; written by caller)
    scratch: &mut crate::cuda_graph::GpuLevelScratch,
    s: usize,
    d: usize,
    level: usize,
    batch_size: usize,
) -> bool {
    let bs = batch_size;
    let dd = d * d;
    // Memory projections into pre-allocated scratch.k_mem, v_mem, q_mem
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_k_mem, &mut scratch.k_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_v_mem, &mut scratch.v_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut scratch.q_mem, bs * s, d, d, 0.0);

    let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
    let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
    let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
    let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
    let m_initial_slice = context_m.slice(0, bs * dd);

    let eff_ckpt = cfg.effective_checkpoint_interval(level);

    match (eff_ckpt, cfg.memory_rule) {
        (None, MemoryRuleKind::DeltaRule) => {
            // Spec 74: DGD fused kernel lacks per-token M-norm projection.
            // Fall back to standard dispatch when m_norm_max is finite.
            if cfg.max_m_norm(level) < 1e30 {
                return false;
            }
            // Spec 39: Fused kernel — L2-normalize + gate compute + clamp + DGD recurrence
            // in a single launch. Writes normalized k/q back to scratch.k_mem/q_mem,
            // gates to scratch.alpha/theta, norms to scratch.k_norms/q_norms.
            scratch.m_states.zero();
            crate::dispatch::delta_fused_forward_dd(
                &mut scratch.k_mem, &scratch.v_mem, &mut scratch.q_mem,
                &level_params.w_alpha, &level_params.b_alpha,
                &level_params.w_theta, &level_params.b_theta,
                alpha_floor, alpha_ceil, theta_floor, theta_ceil,
                &m_initial_slice,
                &mut scratch.m_states, &mut scratch.y,
                &mut scratch.alpha, &mut scratch.theta,
                &mut scratch.k_norms, &mut scratch.q_norms,
                s, d, bs, cfg.error_clip_for_level(level),
            );
            // NOTE: copy_final_m_batch is NOT called here — caller does it outside the graph.
            true
        }
        (None, MemoryRuleKind::TitansLMM) => {
            // Spec 39: Fused kernel — L2-normalize + gate compute (alpha/theta/eta) + clamp
            // + Titans recurrence in a single launch.
            scratch.s_initial.zero();
            let s_initial_slice = scratch.s_initial.slice(0, bs * dd);
            if !scratch.has_s_states {
                eprintln!("[cuda_graph] gpu_memory_forward_into_scratch: TitansLMM requires has_s_states=true — falling back");
                return false;
            }
            scratch.m_states.zero();
            scratch.s_states.zero();
            crate::dispatch::titans_fused_forward_dd(
                &mut scratch.k_mem, &scratch.v_mem, &mut scratch.q_mem,
                &level_params.w_alpha, &level_params.b_alpha,
                &level_params.w_theta, &level_params.b_theta,
                &level_params.w_eta, &level_params.b_eta,
                alpha_floor, alpha_ceil, theta_floor, theta_ceil,
                &m_initial_slice, &s_initial_slice,
                &mut scratch.m_states, &mut scratch.s_states, &mut scratch.y,
                &mut scratch.alpha, &mut scratch.theta, &mut scratch.eta,
                &mut scratch.k_norms, &mut scratch.q_norms,
                s, d, bs, cfg.error_clip_for_level(level), cfg.max_m_norm(level),
            );
            // NOTE: copy_final_m_batch NOT called here — caller does it outside the graph.
            true
        }
        _ => {
            // Unsupported rule/checkpoint combo for scratch path — signal caller to fall back.
            eprintln!("[cuda_graph] gpu_memory_forward_into_scratch: unsupported rule {:?} — disabling graph capture", cfg.memory_rule);
            false
        }
    }
}

/// Build a non-owning `GpuMemoryCache` from scratch buffer pointers.
///
/// The returned cache holds `from_raw_non_owning` GpuBuf views into the scratch.
/// These are valid until the next call to gpu_cms_forward (which overwrites scratch).
/// Safety invariant: backward is called before the next forward — no aliasing hazard.
///
/// NOTE: Scratch buffers always hold full trajectory (proxy: false) because they
/// are pre-allocated at graph creation time. Proxy VRAM savings only apply to
/// the per-step allocation path (gpu_memory_forward), not the CUDA-graph path.
#[cfg(feature = "cuda")]
unsafe fn memory_cache_from_scratch(
    scratch: &crate::cuda_graph::GpuLevelScratch,
    cfg: &MAGConfig,
    bs: usize,
    s: usize,
    d: usize,
) -> Option<GpuMemoryCache> {
    let dd = d * d;
    match (cfg.checkpoint_interval, cfg.memory_rule) {
        (None, MemoryRuleKind::DeltaRule) => Some(GpuMemoryCache::Delta {
            k_mem:    GpuBuf::from_raw_non_owning(scratch.k_mem.ptr(), bs * s * d),
            v_mem:    GpuBuf::from_raw_non_owning(scratch.v_mem.ptr(), bs * s * d),
            q_mem:    GpuBuf::from_raw_non_owning(scratch.q_mem.ptr(), bs * s * d),
            alpha:    GpuBuf::from_raw_non_owning(scratch.alpha.ptr(), bs * s),
            theta:    GpuBuf::from_raw_non_owning(scratch.theta.ptr(), bs * s),
            m_states: GpuBuf::from_raw_non_owning(scratch.m_states.ptr(), bs * (s + 1) * dd),
            k_norms:  GpuBuf::from_raw_non_owning(scratch.k_norms.ptr(), bs * s),
            q_norms:  GpuBuf::from_raw_non_owning(scratch.q_norms.ptr(), bs * s),
            proxy: false,
        }),
        (None, MemoryRuleKind::TitansLMM) => Some(GpuMemoryCache::Titans {
            k_mem:    GpuBuf::from_raw_non_owning(scratch.k_mem.ptr(), bs * s * d),
            v_mem:    GpuBuf::from_raw_non_owning(scratch.v_mem.ptr(), bs * s * d),
            q_mem:    GpuBuf::from_raw_non_owning(scratch.q_mem.ptr(), bs * s * d),
            alpha:    GpuBuf::from_raw_non_owning(scratch.alpha.ptr(), bs * s),
            theta:    GpuBuf::from_raw_non_owning(scratch.theta.ptr(), bs * s),
            eta:      GpuBuf::from_raw_non_owning(scratch.eta.ptr(), bs * s),
            m_states: GpuBuf::from_raw_non_owning(scratch.m_states.ptr(), bs * (s + 1) * dd),
            s_states: GpuBuf::from_raw_non_owning(scratch.s_states.ptr(), bs * (s + 1) * dd),
            k_norms:  GpuBuf::from_raw_non_owning(scratch.k_norms.ptr(), bs * s),
            q_norms:  GpuBuf::from_raw_non_owning(scratch.q_norms.ptr(), bs * s),
            proxy: false,
        }),
        _ => None,  // Unsupported rule — caller (gpu_cms_replay) returns None → standard dispatch
    }
}

// ══════════════════════════════════════════════════════════════════════
// CUDA Graph capture/replay helpers
// ══════════════════════════════════════════════════════════════════════

/// Compute the CMS pulse bitmask (bit i = level i is active). L0 always fires.
#[cfg(feature = "cuda")]
pub fn pulse_to_bitmask(pulse: &Pulse, k: usize) -> u8 {
    let mut mask: u8 = 0;
    for i in 0..k.min(8) {
        if pulse.active_levels[i] {
            mask |= 1u8 << i;
        }
    }
    mask
}

/// All 8 reachable CMS bitmasks for k=4 (L0 always fires → only odd bitmasks 1..15).
#[cfg(feature = "cuda")]
const REACHABLE_BITMASKS_K4: [u8; 8] = [
    0b0001, 0b0011, 0b0101, 0b0111,
    0b1001, 0b1011, 0b1101, 0b1111,
];

/// Capture all reachable pulse patterns into the CudaGraphStore.
///
/// For each bitmask: save context_m → begin_capture → run kernel-only forward into scratch
/// → end_capture → restore context_m. Does NOT update context_m (caller does actual step).
/// Called once at step == warmup_steps.
#[cfg(feature = "cuda")]
fn gpu_cms_capture_all_patterns(
    params: &GpuMAGParams,
    cfg: &MAGConfig,
    input_ids_i32: &[i32],
    target_ids_i32: &[i32],
    _pulse: &Pulse,
    context: &mut GpuContextState,
    bs: usize,
) {
    let s  = cfg.swa.seq_len;
    let d  = cfg.swa.d_model;
    let v  = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;
    let k  = cfg.k;

    // Build reachable bitmask list for this k value
    let reachable: Vec<u8> = REACHABLE_BITMASKS_K4.iter()
        .copied()
        .filter(|&b| (b as usize) < (1 << k))  // drop bitmasks that exceed k levels
        .collect();

    // Save context_m state for each level (D2D into host-side Vec<Vec<f32>>)
    let dd = d * d;
    let saved_m: Vec<Vec<f32>> = context.memory.iter()
        .map(|buf| {
            let mut v = vec![0.0f32; bs * dd];
            buf.copy_to_host(&mut v);
            v
        })
        .collect();

    // Upload input_ids into forward_scratch.d_input_ids (captured as device pointer)
    {
        let fwd = context.forward_scratch.as_ref().expect("forward_scratch must be Some");
        unsafe {
            let rc = gpu_buf_memcpy_h2d(
                fwd.d_input_ids.ptr() as *mut std::ffi::c_void,
                input_ids_i32.as_ptr() as *const std::ffi::c_void,
                bs * s * 4,
            );
            assert_eq!(rc, 0, "capture: H2D input_ids failed");
            let rc = gpu_buf_memcpy_h2d(
                fwd.d_target_ids.ptr() as *mut std::ffi::c_void,
                target_ids_i32.as_ptr() as *const std::ffi::c_void,
                bs * s * 4,
            );
            assert_eq!(rc, 0, "capture: H2D target_ids failed");
        }
    }

    let tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let d_i32      = i32::try_from(d).expect("d_model exceeds i32::MAX");
    let v_i32      = i32::try_from(v).expect("vocab_size exceeds i32::MAX");
    let total      = bs * s * d;
    let total_i32  = i32::try_from(total).expect("bs*s*d exceeds i32::MAX");

    for &bitmask in &reachable {
        // Restore context_m state before each capture
        for (level, buf) in context.memory.iter().enumerate() {
            buf.copy_from_host(&saved_m[level]);
        }

        // Build a synthetic pulse from this bitmask
        let mut active_levels = vec![false; k];
        for i in 0..k {
            active_levels[i] = (bitmask >> i) & 1 != 0;
        }

        // Begin CUDA graph capture (captures all work on default stream)
        if !context.cuda_graph.begin_capture() {
            eprintln!("[cuda_graph] begin_capture failed for bitmask {bitmask:#04b}. Disabling.");
            context.cuda_graph.disable();
            return;
        }

        // Run the full forward pass into scratch buffers (no copy_final_m, no sync, no D2H)
        // All kernel launches go into the capture stream.
        // SAFETY: forward_scratch and level_scratch have fixed addresses valid for the lifetime of context.
        let fwd_ptr = context.forward_scratch.as_mut().expect("forward_scratch must be Some") as *mut crate::cuda_graph::ForwardScratch;
        let lvl_ptr = context.level_scratch.as_mut_ptr();
        let fwd = unsafe { &mut *fwd_ptr };

        // Stage 1: Embedding
        unsafe {
            crate::cuda_ffi::embedding_gather_cuda(
                params.swa.w_embed.as_ptr(),
                fwd.d_input_ids.ptr() as *const i32,
                fwd.embedded.ptr(),
                tokens_i32, d_i32,
            );
        }

        // Stage 2a: QKV projections
        crate::dispatch::cublas_matmul_transb_dd(&fwd.embedded, &params.swa.w_q, &mut fwd.q_f32, bs * s, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&fwd.embedded, &params.swa.w_k, &mut fwd.k_f32, bs * s, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(&fwd.embedded, &params.swa.w_v, &mut fwd.v_f32, bs * s, d, d, 0.0);

        // Stage 3a: f32→bf16, SWA, bf16→f32
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(fwd.q_f32.as_ptr(), fwd.q_bf16.ptr(), total_i32);
            crate::cuda_ffi::f32_to_bf16_cuda(fwd.k_f32.as_ptr(), fwd.k_bf16.ptr(), total_i32);
            crate::cuda_ffi::f32_to_bf16_cuda(fwd.v_f32.as_ptr(), fwd.v_bf16.ptr(), total_i32);
        }
        crate::dispatch::swa_forward_dd(&fwd.q_bf16, &fwd.k_bf16, &fwd.v_bf16,
                                         &mut fwd.attn_out_bf16, &mut fwd.attn_weights_bf16,
                                         s, nh, hd, ws, bs, cfg.n_persistent);
        unsafe { crate::cuda_ffi::bf16_to_f32_cuda(fwd.attn_out_bf16.as_ptr(), fwd.attn_out.ptr(), total_i32); }

        // Stage 2b+3b: Memory per level
        for level in 0..k {
            let is_active = active_levels[level]
                || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

            if is_active {
                let lvl_scratch = unsafe { &mut *lvl_ptr.add(level) };
                let context_m_ptr = &context.memory[level] as *const GpuBuf<f32>;
                let context_m = unsafe { &*context_m_ptr };
                if !gpu_memory_forward_into_scratch(
                    &params.levels[level], cfg, &fwd.embedded,
                    context_m, lvl_scratch, s, d, level, bs,
                ) {
                    // Unsupported rule: end the capture cleanly to release the stream,
                    // then disable graph capture entirely and fall back to standard dispatch.
                    let _ = context.cuda_graph.end_capture(bitmask);
                    context.cuda_graph.disable();
                    return;
                }
                // y_per_level[level] ← scratch.y (copy into forward_scratch.y_per_level)
                unsafe {
                    let rc = gpu_buf_memcpy_d2d(
                        fwd.y_per_level[level].ptr() as *mut std::ffi::c_void,
                        lvl_scratch.y.as_ptr() as *const std::ffi::c_void,
                        bs * s * d * 4,
                    );
                    assert_eq!(rc, 0);
                }
            } else {
                // Frozen: q_mem @ M^T — use persistent scratch to keep device pointers
                // stable across CUDA graph capture/replay (transient GpuBuf would dangle).
                crate::dispatch::cublas_matmul_transb_dd(&fwd.embedded, &params.levels[level].w_q_mem,
                                                          &mut fwd.q_tmp_per_level[level], bs * s, d, d, 0.0);
                crate::dispatch::cublas_matmul_transb_dd(&fwd.q_tmp_per_level[level], &context.memory[level],
                                                          &mut fwd.y_per_level[level], bs * s, d, d, 0.0);
            }
        }

        // Stage: Combine
        fwd.y_combined.zero();
        for level in 0..k {
            unsafe {
                crate::cuda_ffi::saxpy_cuda(1.0, fwd.y_per_level[level].as_ptr(), fwd.y_combined.ptr(), total_i32);
            }
        }
        if k > 2 {
            let scale = 1.0 / (k as f32).sqrt();
            unsafe { crate::cuda_ffi::saxpy_cuda(scale - 1.0, fwd.y_combined.as_ptr(), fwd.y_combined.ptr(), total_i32); }
        }

        // Stage: Gate
        unsafe {
            crate::cuda_ffi::sigmoid_cuda(fwd.y_combined.as_ptr(), fwd.gate.ptr(), total_i32);
            crate::cuda_ffi::elemwise_mul_cuda(fwd.attn_out.as_ptr(), fwd.gate.as_ptr(), fwd.gated_out.ptr(), total_i32);
        }

        // Stage: Output projection + unembed
        crate::dispatch::cublas_matmul_transb_dd(&fwd.gated_out, &params.swa.w_o, &mut fwd.projected, bs * s, d, d, 0.0);
        crate::dispatch::cublas_matmul_dd(&fwd.projected, &params.swa.w_unembed, &mut fwd.logits, bs * s, d, v, 0.0);

        // Stage: Cross-entropy (result in fwd.loss_gpu)
        unsafe {
            crate::cuda_ffi::cross_entropy_forward_cuda(
                fwd.logits.as_ptr(),
                fwd.d_target_ids.ptr() as *const i32,
                fwd.loss_gpu.ptr(),
                tokens_i32, v_i32,
            );
        }

        // End capture → instantiate graph
        if !context.cuda_graph.end_capture(bitmask) {
            eprintln!("[cuda_graph] end_capture failed for bitmask {bitmask:#04b}. Disabling.");
            context.cuda_graph.disable();
            // Restore context_m
            for (level, buf) in context.memory.iter().enumerate() {
                buf.copy_from_host(&saved_m[level]);
            }
            return;
        }
    }

    // Restore context_m to pre-capture state (caller will run the actual step via standard dispatch)
    for (level, buf) in context.memory.iter().enumerate() {
        buf.copy_from_host(&saved_m[level]);
    }

    eprintln!("[cuda_graph] Captured {} patterns at step {}.",
              reachable.len(), context.cuda_graph.steps_seen);
    let _ = (nh, hd, ws, total_i32, v_i32, d_i32, tokens_i32); // suppress unused warnings
}

/// Run one GPU forward pass via CUDA graph replay.
///
/// Uploads current input/target token ids into pre-allocated scratch (outside graph),
/// then launches the captured graph, runs manual copy_final_m + m_norm_clamp (outside graph),
/// syncs, downloads loss, and returns `(f32, GpuCMSCache)` with non-owning cache.
///
/// Returns `None` if replay fails (caller falls through to standard dispatch).
#[cfg(feature = "cuda")]
fn gpu_cms_replay(
    _params: &GpuMAGParams,
    cfg: &MAGConfig,
    input_ids_i32: &[i32],
    target_ids_i32: &[i32],
    pulse: &Pulse,
    context: &mut GpuContextState,
    bitmask: u8,
    bs: usize,
) -> Option<(f32, GpuCMSCache)> {
    let s  = cfg.swa.seq_len;
    let d  = cfg.swa.d_model;
    let v  = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;
    let k  = cfg.k;
    // NOTE: CUDA graph scratch (GpuLevelScratch) uses monolithic dd = d*d layout,
    // not per-head dd_mem = nh*hd*hd. The entire CUDA graph path
    // (gpu_memory_forward_into_scratch + gpu_cms_replay) needs a per-head audit
    // if cuda_graph_warmup is ever enabled with num_heads > 1.
    // Currently disabled in all configs (warmup=0).
    let dd = d * d;

    let fwd = context.forward_scratch.as_ref()?;
    let d_i32       = i32::try_from(d).expect("d_model exceeds i32::MAX");
    let _tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let _v_i32      = i32::try_from(v).expect("vocab_size exceeds i32::MAX");

    // H2D upload BEFORE graph launch (outside captured region, on default stream)
    unsafe {
        let rc = gpu_buf_memcpy_h2d(
            fwd.d_input_ids.ptr() as *mut std::ffi::c_void,
            input_ids_i32.as_ptr() as *const std::ffi::c_void,
            bs * s * 4,
        );
        if rc != 0 { return None; }
        let rc = gpu_buf_memcpy_h2d(
            fwd.d_target_ids.ptr() as *mut std::ffi::c_void,
            target_ids_i32.as_ptr() as *const std::ffi::c_void,
            bs * s * 4,
        );
        if rc != 0 { return None; }
    }

    // Launch the captured graph (replays all kernel launches with fixed scratch pointers)
    if !context.cuda_graph.replay(bitmask) {
        return None;
    }

    // Manual copy_final_m + m_norm_clamp OUTSIDE graph, for each active level
    for level in 0..k {
        let is_active = pulse.active_levels[level]
            || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);
        if is_active {
            let m_states = &context.level_scratch[level].m_states;
            copy_final_m_batch(m_states, &mut context.memory[level], s, dd, bs);
            let m_norm_max = cfg.max_m_norm(level);
            // Spec 65: batched clamp — one launch for all batch elements
            unsafe {
                crate::cuda_ffi::m_norm_clamp_batch_f32_cuda(
                    context.memory[level].ptr(), d_i32, bs as i32, m_norm_max,
                );
            }
        }
    }

    // Sync + D2H loss
    crate::dispatch::cuda_sync();
    let fwd = context.forward_scratch.as_ref().unwrap();
    let mut loss_host = [0.0f32; 1];
    fwd.loss_gpu.copy_to_host(&mut loss_host);
    let valid_count = target_ids_i32.iter()
        .filter(|&&t| t >= 0 && (t as usize) < v)
        .count() as f32;
    let loss = if valid_count > 0.0 { loss_host[0] / valid_count } else { 0.0 };

    // Build GpuCMSCache with non-owning views into scratch
    // SAFETY: scratch lives in GpuContextState which outlives this cache.
    // Backward is called before the next forward, so non-owning views are valid.
    let fwd = context.forward_scratch.as_ref().unwrap();
    let mut memory_caches = Vec::with_capacity(k);
    let mut y_per_level: Vec<GpuBuf<f32>> = Vec::with_capacity(k);

    for level in 0..k {
        let is_active = pulse.active_levels[level]
            || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);
        if is_active {
            let lvl = &context.level_scratch[level];
            let cache = unsafe { memory_cache_from_scratch(lvl, cfg, bs, s, d)? };
            memory_caches.push(Some(cache));
            let y = unsafe { GpuBuf::from_raw_non_owning(lvl.y.ptr(), bs * s * d) };
            y_per_level.push(y);
        } else {
            memory_caches.push(None);
            let y = unsafe { GpuBuf::from_raw_non_owning(fwd.y_per_level[level].ptr(), bs * s * d) };
            y_per_level.push(y);
        }
    }

    let cache = GpuCMSCache {
        input_ids_gpu:       unsafe { GpuBuf::from_raw_non_owning(fwd.d_input_ids.ptr(), bs * s) },
        target_ids_gpu:      unsafe { GpuBuf::from_raw_non_owning(fwd.d_target_ids.ptr(), bs * s) },
        input_ids_i32:       input_ids_i32.to_vec(),
        target_ids_i32:      target_ids_i32.to_vec(),
        embedded:            unsafe { GpuBuf::from_raw_non_owning(fwd.embedded.ptr(), bs * s * d) },
        q_f32:               unsafe { GpuBuf::from_raw_non_owning(fwd.q_f32.ptr(), bs * s * d) },
        k_f32:               unsafe { GpuBuf::from_raw_non_owning(fwd.k_f32.ptr(), bs * s * d) },
        v_f32:               unsafe { GpuBuf::from_raw_non_owning(fwd.v_f32.ptr(), bs * s * d) },
        q_bf16:              unsafe { GpuBuf::from_raw_non_owning(fwd.q_bf16.ptr(), bs * s * d) },
        k_bf16:              unsafe { GpuBuf::from_raw_non_owning(fwd.k_bf16.ptr(), bs * s * d) },
        v_bf16:              unsafe { GpuBuf::from_raw_non_owning(fwd.v_bf16.ptr(), bs * s * d) },
        attn_out_bf16:       unsafe { GpuBuf::from_raw_non_owning(fwd.attn_out_bf16.ptr(), bs * s * d) },
        attn_weights_bf16:   unsafe { GpuBuf::from_raw_non_owning(fwd.attn_weights_bf16.ptr(), bs * nh * s * (cfg.n_persistent + ws)) },
        attn_out:            unsafe { GpuBuf::from_raw_non_owning(fwd.attn_out.ptr(), bs * s * d) },
        memory_caches,
        y_per_level,
        y_combined:          unsafe { GpuBuf::from_raw_non_owning(fwd.y_combined.ptr(), bs * s * d) },
        gate:                unsafe { GpuBuf::from_raw_non_owning(fwd.gate.ptr(), bs * s * d) },
        gated_out:           unsafe { GpuBuf::from_raw_non_owning(fwd.gated_out.ptr(), bs * s * d) },
        // CUDA graph path is only used for non-residual configs (capture check excludes residual).
        ln_attn_out: None, ln_attn_mean: None, ln_attn_rstd: None,
        ln_mem_out: None, ln_mem_mean: None, ln_mem_rstd: None,
        residual_after_attn: None, residual_final: None,
        projected:           unsafe { GpuBuf::from_raw_non_owning(fwd.projected.ptr(), bs * s * d) },
        logits:              unsafe { GpuBuf::from_raw_non_owning(fwd.logits.ptr(), bs * s * v) },
        pulse: pulse.clone(),
        s, d, v, nh, hd, ws,
        batch_size: bs,
    };

    Some((loss, cache))
}

/// Copy all batch elements' final M back to their respective context slots.
///
/// Element b's final M is at m_states offset: `b * (seq_len+1) * dd + seq_len * dd`
/// It is written to context_m at offset: `b * dd`
///
/// This replaces the old `copy_final_m` (element-0 only) with a full batch copy.
/// Preserves sequential context continuity: each slot sees its own stream across steps.
#[cfg(feature = "cuda")]
fn copy_final_m_batch(
    states: &GpuBuf<f32>,
    context_m: &mut GpuBuf<f32>,
    seq_len: usize,
    dd: usize,
    batch_size: usize,
) {
    let bytes = dd * 4;
    for b in 0..batch_size {
        let src_offset = (b * (seq_len + 1) + seq_len) * dd;
        let dst_offset = b * dd;
        let m_final = states.slice(src_offset, dd);
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                (context_m.ptr() as *mut u8).add(dst_offset * 4) as *mut std::ffi::c_void,
                m_final.as_ptr() as *const std::ffi::c_void,
                bytes,
            );
            assert_eq!(rc, 0, "copy_final_m_batch D2D failed for slot {b}");
        }
    }
}

/// Copy final M state from a checkpointed states buffer to context (D2D).
/// Used only for checkpointed paths (bs=1 only).
#[cfg(feature = "cuda")]
#[inline]
#[allow(dead_code)] // Retained as single-slot variant of copy_final_m_batch
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
    let eta = GpuBuf::zeros(s);
    // Pass b_eta as device pointer (CUDA-graph-capture-safe: stable pointer, value updated in-place).
    unsafe {
        crate::cuda_ffi::gate_compute_cuda(
            k_mem.as_ptr(), v_mem.as_ptr(), level_params.w_eta.as_ptr(),
            level_params.b_eta.as_ptr(), eta.ptr(),
            s as i32, d as i32, 0,
        );
    }
    eta
}

/// Number of checkpoints stored: M_0, then one per C-step boundary, plus final state.
/// Not gated by `cuda` feature — pure arithmetic, usable in tests.
pub fn checkpoint_count(seq_len: usize, c: usize) -> usize {
    assert!(c > 0, "checkpoint_interval must be > 0");
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
///
/// For monolithic (nh=1): single cuBLAS matmul Y = Q @ M^T.
/// For per-head (nh>1): transpose Q to per-head layout, matmul per head, transpose back.
/// `s` is total tokens (batch_size * seq_len from caller).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_read_only(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    embedded: &GpuBuf<f32>,
    context_m: &GpuBuf<f32>,   // [mem_dd] — read only (d*d for nh=1, nh*hd*hd for nh>1)
    s: usize,
    d: usize,
    num_heads: usize,
    head_dim: usize,
) -> GpuBuf<f32> {
    // q_mem = embedded @ W_q_mem^T  →  [s, d]
    let mut q_mem = GpuBuf::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut q_mem, s, d, d, 0.0);

    if num_heads == 1 {
        // Monolithic: Y[s,d] = Q[s,d] @ M[d,d]^T
        let mut y = GpuBuf::zeros(s * d);
        crate::dispatch::cublas_matmul_transb_dd(&q_mem, context_m, &mut y, s, d, d, 0.0);
        y
    } else {
        // Per-head: reshape Q → [nh, s, hd], matmul per head, reshape back.
        // Use batch_size=1 since s is already total_tokens and frozen M is shared.
        let q_ph = reshape_to_per_head(&q_mem, 1, s, num_heads, head_dim);
        let y_ph = GpuBuf::<f32>::zeros(num_heads * s * head_dim);
        let dd_mem = head_dim * head_dim;
        for h in 0..num_heads {
            let off_q = h * s * head_dim;
            let off_m = h * dd_mem;
            let off_y = h * s * head_dim;
            unsafe {
                let q_h: GpuBuf<f32> = GpuBuf::from_raw_non_owning(
                    q_ph.as_ptr().add(off_q) as *mut f32, s * head_dim);
                let m_h: GpuBuf<f32> = GpuBuf::from_raw_non_owning(
                    context_m.as_ptr().add(off_m) as *mut f32, dd_mem);
                let mut y_h: GpuBuf<f32> = GpuBuf::from_raw_non_owning(
                    y_ph.ptr().add(off_y) as *mut f32, s * head_dim);
                // Y_h[s, hd] = Q_h[s, hd] @ M_h[hd, hd]^T
                crate::dispatch::cublas_matmul_transb_dd(
                    &q_h, &m_h, &mut y_h, s, head_dim, head_dim, 0.0);
            }
        }
        let y = reshape_from_per_head(&y_ph, 1, s, num_heads, head_dim);
        y
    }
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

    /// Pre-populate positions [0, n_persistent) with projected persistent token K/V.
    /// Must be called once per block before any tokens are processed.
    /// `persistent_tokens` is [n_p, d] f32, `w_k`/`w_v` are [d, d] f32.
    pub fn prepopulate_persistent(
        &mut self,
        persistent_tokens: &GpuBuf<f32>,
        w_k: &GpuBuf<f32>,
        w_v: &GpuBuf<f32>,
        n_persistent: usize,
        d: usize,
    ) {
        if n_persistent == 0 { return; }
        assert!(n_persistent <= self.max_len,
            "KV cache too small for {} persistent tokens", n_persistent);
        assert_eq!(self.len, 0, "prepopulate_persistent must be called on empty cache");

        let npd = n_persistent * d;
        let mut pk = GpuBuf::<f32>::zeros(npd);
        let mut pv = GpuBuf::<f32>::zeros(npd);
        crate::dispatch::cublas_matmul_transb_dd(persistent_tokens, w_k, &mut pk, n_persistent, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(persistent_tokens, w_v, &mut pv, n_persistent, d, d, 0.0);

        // Convert to bf16 and copy into cache positions [0, n_p)
        let pk_bf16 = GpuBuf::<u16>::zeros(npd);
        let pv_bf16 = GpuBuf::<u16>::zeros(npd);
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(pk.as_ptr(), pk_bf16.ptr(), npd as i32);
            crate::cuda_ffi::f32_to_bf16_cuda(pv.as_ptr(), pv_bf16.ptr(), npd as i32);
            gpu_buf_memcpy_d2d(
                self.k_cache_bf16.ptr() as *mut _,
                pk_bf16.as_ptr() as *const _,
                npd * 2,
            );
            gpu_buf_memcpy_d2d(
                self.v_cache_bf16.ptr() as *mut _,
                pv_bf16.as_ptr() as *const _,
                npd * 2,
            );
        }
        self.len = n_persistent;
    }

    /// Re-project persistent tokens into positions [0, n_p) after optimizer updates.
    /// Same as prepopulate_persistent but works on a non-empty cache (no len change).
    pub fn refresh_persistent(
        &mut self,
        persistent_tokens: &GpuBuf<f32>,
        w_k: &GpuBuf<f32>,
        w_v: &GpuBuf<f32>,
        n_persistent: usize,
        d: usize,
    ) {
        if n_persistent == 0 { return; }
        assert!(self.len >= n_persistent,
            "cache len {} < n_persistent {}", self.len, n_persistent);

        let npd = n_persistent * d;
        let mut pk = GpuBuf::<f32>::zeros(npd);
        let mut pv = GpuBuf::<f32>::zeros(npd);
        crate::dispatch::cublas_matmul_transb_dd(persistent_tokens, w_k, &mut pk, n_persistent, d, d, 0.0);
        crate::dispatch::cublas_matmul_transb_dd(persistent_tokens, w_v, &mut pv, n_persistent, d, d, 0.0);

        let pk_bf16 = GpuBuf::<u16>::zeros(npd);
        let pv_bf16 = GpuBuf::<u16>::zeros(npd);
        unsafe {
            crate::cuda_ffi::f32_to_bf16_cuda(pk.as_ptr(), pk_bf16.ptr(), npd as i32);
            crate::cuda_ffi::f32_to_bf16_cuda(pv.as_ptr(), pv_bf16.ptr(), npd as i32);
            gpu_buf_memcpy_d2d(
                self.k_cache_bf16.ptr() as *mut _,
                pk_bf16.as_ptr() as *const _,
                npd * 2,
            );
            gpu_buf_memcpy_d2d(
                self.v_cache_bf16.ptr() as *mut _,
                pv_bf16.as_ptr() as *const _,
                npd * 2,
            );
        }
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
    assert!(!cfg.residual, "gpu_prefill_forward: residual=true requires CUDA LayerNorm kernels (not yet implemented). Use CPU path.");
    assert!(input_ids.len() >= s);

    // Upload input_ids
    let input_ids_i32: Vec<i32> = input_ids[..s].iter()
        .map(|&x| i32::try_from(x).expect("input token id overflows i32 — vocab_size exceeds i32::MAX"))
        .collect();
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
    let embedded = GpuBuf::<f32>::zeros(s * d);
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
    let aw_stride = cfg.n_persistent + ws;
    let aw_total = nh * s * aw_stride;
    let q_bf16 = GpuBuf::<u16>::zeros(total);
    let k_bf16 = GpuBuf::<u16>::zeros(total);
    let v_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_out_bf16 = GpuBuf::<u16>::zeros(total);
    let mut attn_weights_bf16 = GpuBuf::<u16>::zeros(aw_total);

    unsafe {
        crate::cuda_ffi::f32_to_bf16_cuda(q_f32.as_ptr(), q_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(k_f32.as_ptr(), k_bf16.ptr(), total as i32);
        crate::cuda_ffi::f32_to_bf16_cuda(v_f32.as_ptr(), v_bf16.ptr(), total as i32);
    }

    // prefill is always batch_size=1 (single prompt)
    crate::dispatch::swa_forward_dd(
        &q_bf16, &k_bf16, &v_bf16,
        &mut attn_out_bf16, &mut attn_weights_bf16,
        s, nh, hd, ws, 1, cfg.n_persistent,
    );

    let attn_out = GpuBuf::<f32>::zeros(total);
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
                s, d, level, 1,
            );
            y_per_level.push(y_level);
        } else {
            if matches!(cfg.memory_rule, MemoryRuleKind::Moneta | MemoryRuleKind::YAAD) {
                y_per_level.push(frozen_mlp_fallback("gpu_prefill_forward", level, cfg.memory_rule, s * d));
            } else {
            let y_level = gpu_memory_read_only(
                &params.levels[level], &embedded,
                &context.memory[level],
                s, d, cfg.swa.num_heads, cfg.swa.head_dim,
            );
            y_per_level.push(y_level);
            }
        }
    }

    // ── Combine levels ─────────────────────────────────────────────
    let y_combined = GpuBuf::<f32>::zeros(s * d);
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
    let gate = GpuBuf::<f32>::zeros(s * d);
    let gated_out = GpuBuf::<f32>::zeros(s * d);
    unsafe {
        crate::cuda_ffi::sigmoid_cuda(y_combined.as_ptr(), gate.ptr(), (s * d) as i32);
        crate::cuda_ffi::elemwise_mul_cuda(attn_out.as_ptr(), gate.as_ptr(), gated_out.ptr(), (s * d) as i32);
    }

    // ── Stage 5: Output projection ─────────────────────────────────
    let mut projected = GpuBuf::<f32>::zeros(s * d);
    crate::dispatch::cublas_matmul_transb_dd(&gated_out, &params.swa.w_o, &mut projected, s, d, d, 0.0);

    // ── Stage 6: Unembed (only last position) ──────────────────────
    // Extract last position projected[s-1] as [1, d] and unembed to [1, v]
    let last_projected = GpuBuf::<f32>::zeros(d);
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

    assert!(!cfg.residual, "gpu_single_token_forward: residual=true requires CUDA LayerNorm kernels (not yet implemented). Use CPU path.");
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
        kv_cache.len, nh, hd, window_size, cfg.n_persistent,
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
                1, d, level, 1,
            );
            y_per_level.push(y_level);
        } else {
            if matches!(cfg.memory_rule, MemoryRuleKind::Moneta | MemoryRuleKind::YAAD) {
                y_per_level.push(frozen_mlp_fallback("gpu_single_token_forward", level, cfg.memory_rule, d));
            } else {
            let y_level = gpu_memory_read_only(
                &params.levels[level], &ws.embedded,
                &context.memory[level],
                1, d, cfg.swa.num_heads, cfg.swa.head_dim,
            );
            y_per_level.push(y_level);
            }
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

// ══════════════════════════════════════════════════════════════════════
// Shared test helpers (spec 39 CUDA tests)
// ══════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod test_helpers {
    pub fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            v.push(f * 0.1);
        }
        v
    }

    pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests: Fused vs unfused kernel bit-identity (spec 39)
// ══════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod fused_tests {
    use crate::gpu_buf::GpuBuf;
    use super::test_helpers::{rand_vec, max_abs_diff};

    #[test]
    fn test_dgd_fused_matches_unfused() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        // Random inputs
        let k_raw = rand_vec(bs * s * d, 42);
        let v_raw = rand_vec(bs * s * d, 43);
        let q_raw = rand_vec(bs * s * d, 44);
        let m_init = rand_vec(bs * dd, 45);
        let w_alpha = rand_vec(2 * d, 46);
        let b_alpha_v = vec![3.0f32]; // sigmoid(3) ≈ 0.95
        let w_theta = rand_vec(2 * d, 47);
        let b_theta_v = vec![-4.6f32]; // softplus(-4.6) ≈ 0.01
        let alpha_floor = 0.0f32;
        let alpha_ceil = 1.0f32;
        let theta_floor = 0.0f32;
        let theta_ceil = 1.0f32;
        let error_clip = 0.0f32;

        // ── Unfused path: l2_normalize → gate_compute → dgd_forward ──
        let k_unf = GpuBuf::from_host(&k_raw);
        let q_unf = GpuBuf::from_host(&q_raw);
        let v_unf = GpuBuf::from_host(&v_raw);
        let k_norms_unf = GpuBuf::zeros(bs * s);
        let q_norms_unf = GpuBuf::zeros(bs * s);
        let alpha_unf = GpuBuf::zeros(bs * s);
        let theta_unf = GpuBuf::zeros(bs * s);
        let d_w_alpha = GpuBuf::from_host(&w_alpha);
        let d_b_alpha = GpuBuf::from_host(&b_alpha_v);
        let d_w_theta = GpuBuf::from_host(&w_theta);
        let d_b_theta = GpuBuf::from_host(&b_theta_v);

        unsafe {
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(
                k_unf.as_ptr() as *mut f32, k_norms_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1e-8,
            );
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(
                q_unf.as_ptr() as *mut f32, q_norms_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1e-8,
            );
            crate::cuda_ffi::gate_compute_cuda(
                k_unf.as_ptr(), v_unf.as_ptr(), d_w_alpha.as_ptr(),
                d_b_alpha.as_ptr(), alpha_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 0,
            );
            crate::cuda_ffi::gate_compute_cuda(
                k_unf.as_ptr(), v_unf.as_ptr(), d_w_theta.as_ptr(),
                d_b_theta.as_ptr(), theta_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1,
            );
        }
        let m_init_buf = GpuBuf::from_host(&m_init);
        let mut m_states_unf = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_unf = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_forward_dd(
            &k_unf, &v_unf, &q_unf, &alpha_unf, &theta_unf,
            &m_init_buf.slice(0, bs * dd),
            &mut m_states_unf, &mut y_unf, s, d, bs, s, d * d, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Read results
        let mut y_unf_h = vec![0.0f32; bs * s * d];
        let mut alpha_unf_h = vec![0.0f32; bs * s];
        let mut theta_unf_h = vec![0.0f32; bs * s];
        let mut k_norms_unf_h = vec![0.0f32; bs * s];
        let mut q_norms_unf_h = vec![0.0f32; bs * s];
        let mut k_norm_h = vec![0.0f32; bs * s * d];
        y_unf.copy_to_host(&mut y_unf_h);
        alpha_unf.copy_to_host(&mut alpha_unf_h);
        theta_unf.copy_to_host(&mut theta_unf_h);
        k_norms_unf.copy_to_host(&mut k_norms_unf_h);
        q_norms_unf.copy_to_host(&mut q_norms_unf_h);
        k_unf.copy_to_host(&mut k_norm_h);

        // ── Fused path ──
        let mut k_fused = GpuBuf::from_host(&k_raw);
        let mut q_fused = GpuBuf::from_host(&q_raw);
        let v_fused = GpuBuf::from_host(&v_raw);
        let m_init_fused = GpuBuf::from_host(&m_init);
        let mut m_states_fused = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_fused = GpuBuf::zeros(bs * s * d);
        let mut alpha_fused = GpuBuf::zeros(bs * s);
        let mut theta_fused = GpuBuf::zeros(bs * s);
        let mut k_norms_fused = GpuBuf::zeros(bs * s);
        let mut q_norms_fused = GpuBuf::zeros(bs * s);

        crate::dispatch::delta_fused_forward_dd(
            &mut k_fused, &v_fused, &mut q_fused,
            &d_w_alpha, &d_b_alpha,
            &d_w_theta, &d_b_theta,
            alpha_floor, alpha_ceil, theta_floor, theta_ceil,
            &m_init_fused.slice(0, bs * dd),
            &mut m_states_fused, &mut y_fused,
            &mut alpha_fused, &mut theta_fused,
            &mut k_norms_fused, &mut q_norms_fused,
            s, d, bs, error_clip,
        );
        crate::dispatch::cuda_sync();

        // Read fused results
        let mut y_fused_h = vec![0.0f32; bs * s * d];
        let mut alpha_fused_h = vec![0.0f32; bs * s];
        let mut theta_fused_h = vec![0.0f32; bs * s];
        let mut k_norms_fused_h = vec![0.0f32; bs * s];
        let mut q_norms_fused_h = vec![0.0f32; bs * s];
        let mut k_norm_fused_h = vec![0.0f32; bs * s * d];
        y_fused.copy_to_host(&mut y_fused_h);
        alpha_fused.copy_to_host(&mut alpha_fused_h);
        theta_fused.copy_to_host(&mut theta_fused_h);
        k_norms_fused.copy_to_host(&mut k_norms_fused_h);
        q_norms_fused.copy_to_host(&mut q_norms_fused_h);
        k_fused.copy_to_host(&mut k_norm_fused_h);

        // Compare — should be identical (same math, same precision)
        let y_diff = max_abs_diff(&y_unf_h, &y_fused_h);
        let alpha_diff = max_abs_diff(&alpha_unf_h, &alpha_fused_h);
        let theta_diff = max_abs_diff(&theta_unf_h, &theta_fused_h);
        let k_norms_diff = max_abs_diff(&k_norms_unf_h, &k_norms_fused_h);
        let q_norms_diff = max_abs_diff(&q_norms_unf_h, &q_norms_fused_h);
        let k_diff = max_abs_diff(&k_norm_h, &k_norm_fused_h);

        eprintln!("DGD fused vs unfused — y_diff={y_diff:.2e}, alpha_diff={alpha_diff:.2e}, theta_diff={theta_diff:.2e}, k_norms_diff={k_norms_diff:.2e}, q_norms_diff={q_norms_diff:.2e}, k_diff={k_diff:.2e}");

        // Tolerance: 1e-5 per spec (CUDA vs CUDA, same precision)
        assert!(y_diff < 1e-5, "y output mismatch: {y_diff}");
        assert!(alpha_diff < 1e-5, "alpha mismatch: {alpha_diff}");
        assert!(theta_diff < 1e-5, "theta mismatch: {theta_diff}");
        assert!(k_norms_diff < 1e-5, "k_norms mismatch: {k_norms_diff}");
        assert!(q_norms_diff < 1e-5, "q_norms mismatch: {q_norms_diff}");
        assert!(k_diff < 1e-5, "normalized k mismatch: {k_diff}");
    }

    #[test]
    fn test_titans_fused_matches_unfused() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        // Random inputs
        let k_raw = rand_vec(bs * s * d, 52);
        let v_raw = rand_vec(bs * s * d, 53);
        let q_raw = rand_vec(bs * s * d, 54);
        let m_init = rand_vec(bs * dd, 55);
        let s_init = vec![0.0f32; bs * dd]; // momentum starts at zero
        let w_alpha = rand_vec(2 * d, 56);
        let b_alpha_v = vec![3.0f32];
        let w_theta = rand_vec(2 * d, 57);
        let b_theta_v = vec![-4.6f32];
        let w_eta = rand_vec(2 * d, 58);
        let b_eta_v = vec![2.0f32]; // sigmoid(2) ≈ 0.88
        let alpha_floor = 0.0f32;
        let alpha_ceil = 1.0f32;
        let theta_floor = 0.0f32;
        let theta_ceil = 1.0f32;
        let error_clip = 0.0f32;

        // ── Unfused path: l2_normalize → gate_compute × 3 → titans_forward ──
        let k_unf = GpuBuf::from_host(&k_raw);
        let q_unf = GpuBuf::from_host(&q_raw);
        let v_unf = GpuBuf::from_host(&v_raw);
        let k_norms_unf = GpuBuf::zeros(bs * s);
        let q_norms_unf = GpuBuf::zeros(bs * s);
        let alpha_unf = GpuBuf::zeros(bs * s);
        let theta_unf = GpuBuf::zeros(bs * s);
        let eta_unf = GpuBuf::zeros(bs * s);
        let d_w_alpha = GpuBuf::from_host(&w_alpha);
        let d_b_alpha = GpuBuf::from_host(&b_alpha_v);
        let d_w_theta = GpuBuf::from_host(&w_theta);
        let d_b_theta = GpuBuf::from_host(&b_theta_v);
        let d_w_eta = GpuBuf::from_host(&w_eta);
        let d_b_eta = GpuBuf::from_host(&b_eta_v);

        unsafe {
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(
                k_unf.as_ptr() as *mut f32, k_norms_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1e-8,
            );
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(
                q_unf.as_ptr() as *mut f32, q_norms_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1e-8,
            );
            // alpha: mode=0 (sigmoid)
            crate::cuda_ffi::gate_compute_cuda(
                k_unf.as_ptr(), v_unf.as_ptr(), d_w_alpha.as_ptr(),
                d_b_alpha.as_ptr(), alpha_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 0,
            );
            // theta: mode=1 (softplus)
            crate::cuda_ffi::gate_compute_cuda(
                k_unf.as_ptr(), v_unf.as_ptr(), d_w_theta.as_ptr(),
                d_b_theta.as_ptr(), theta_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 1,
            );
            // eta: mode=0 (sigmoid)
            crate::cuda_ffi::gate_compute_cuda(
                k_unf.as_ptr(), v_unf.as_ptr(), d_w_eta.as_ptr(),
                d_b_eta.as_ptr(), eta_unf.as_ptr() as *mut f32,
                (bs * s) as i32, d as i32, 0,
            );
        }
        let m_init_buf = GpuBuf::from_host(&m_init);
        let s_init_buf = GpuBuf::from_host(&s_init);
        let mut m_states_unf = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut s_states_unf = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_unf = GpuBuf::zeros(bs * s * d);
        crate::dispatch::titans_forward_dd(
            &k_unf, &v_unf, &q_unf, &alpha_unf, &theta_unf, &eta_unf,
            &m_init_buf.slice(0, bs * dd), &s_init_buf.slice(0, bs * dd),
            &mut m_states_unf, &mut s_states_unf, &mut y_unf,
            s, d, bs, s, d * d, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Read unfused results
        let mut y_unf_h = vec![0.0f32; bs * s * d];
        let mut alpha_unf_h = vec![0.0f32; bs * s];
        let mut theta_unf_h = vec![0.0f32; bs * s];
        let mut eta_unf_h = vec![0.0f32; bs * s];
        let mut k_norms_unf_h = vec![0.0f32; bs * s];
        let mut q_norms_unf_h = vec![0.0f32; bs * s];
        y_unf.copy_to_host(&mut y_unf_h);
        alpha_unf.copy_to_host(&mut alpha_unf_h);
        theta_unf.copy_to_host(&mut theta_unf_h);
        eta_unf.copy_to_host(&mut eta_unf_h);
        k_norms_unf.copy_to_host(&mut k_norms_unf_h);
        q_norms_unf.copy_to_host(&mut q_norms_unf_h);

        // ── Fused path ──
        let mut k_fused = GpuBuf::from_host(&k_raw);
        let mut q_fused = GpuBuf::from_host(&q_raw);
        let v_fused = GpuBuf::from_host(&v_raw);
        let m_init_fused = GpuBuf::from_host(&m_init);
        let s_init_fused = GpuBuf::from_host(&s_init);
        let mut m_states_fused = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut s_states_fused = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_fused = GpuBuf::zeros(bs * s * d);
        let mut alpha_fused = GpuBuf::zeros(bs * s);
        let mut theta_fused = GpuBuf::zeros(bs * s);
        let mut eta_fused = GpuBuf::zeros(bs * s);
        let mut k_norms_fused = GpuBuf::zeros(bs * s);
        let mut q_norms_fused = GpuBuf::zeros(bs * s);

        crate::dispatch::titans_fused_forward_dd(
            &mut k_fused, &v_fused, &mut q_fused,
            &d_w_alpha, &d_b_alpha,
            &d_w_theta, &d_b_theta,
            &d_w_eta, &d_b_eta,
            alpha_floor, alpha_ceil, theta_floor, theta_ceil,
            &m_init_fused.slice(0, bs * dd), &s_init_fused.slice(0, bs * dd),
            &mut m_states_fused, &mut s_states_fused, &mut y_fused,
            &mut alpha_fused, &mut theta_fused, &mut eta_fused,
            &mut k_norms_fused, &mut q_norms_fused,
            s, d, bs, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Read fused results
        let mut y_fused_h = vec![0.0f32; bs * s * d];
        let mut alpha_fused_h = vec![0.0f32; bs * s];
        let mut theta_fused_h = vec![0.0f32; bs * s];
        let mut eta_fused_h = vec![0.0f32; bs * s];
        let mut k_norms_fused_h = vec![0.0f32; bs * s];
        let mut q_norms_fused_h = vec![0.0f32; bs * s];
        y_fused.copy_to_host(&mut y_fused_h);
        alpha_fused.copy_to_host(&mut alpha_fused_h);
        theta_fused.copy_to_host(&mut theta_fused_h);
        eta_fused.copy_to_host(&mut eta_fused_h);
        k_norms_fused.copy_to_host(&mut k_norms_fused_h);
        q_norms_fused.copy_to_host(&mut q_norms_fused_h);

        // Compare
        let y_diff = max_abs_diff(&y_unf_h, &y_fused_h);
        let alpha_diff = max_abs_diff(&alpha_unf_h, &alpha_fused_h);
        let theta_diff = max_abs_diff(&theta_unf_h, &theta_fused_h);
        let eta_diff = max_abs_diff(&eta_unf_h, &eta_fused_h);
        let k_norms_diff = max_abs_diff(&k_norms_unf_h, &k_norms_fused_h);
        let q_norms_diff = max_abs_diff(&q_norms_unf_h, &q_norms_fused_h);

        eprintln!("Titans fused vs unfused — y_diff={y_diff:.2e}, alpha_diff={alpha_diff:.2e}, theta_diff={theta_diff:.2e}, eta_diff={eta_diff:.2e}, k_norms_diff={k_norms_diff:.2e}, q_norms_diff={q_norms_diff:.2e}");

        assert!(y_diff < 1e-5, "y output mismatch: {y_diff}");
        assert!(alpha_diff < 1e-5, "alpha mismatch: {alpha_diff}");
        assert!(theta_diff < 1e-5, "theta mismatch: {theta_diff}");
        assert!(eta_diff < 1e-5, "eta mismatch: {eta_diff}");
        assert!(k_norms_diff < 1e-5, "k_norms mismatch: {k_norms_diff}");
        assert!(q_norms_diff < 1e-5, "q_norms mismatch: {q_norms_diff}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests: Scratch path (CUDA-graph-compatible) matches standard forward
// ══════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod scratch_tests {
    use crate::gpu_buf::GpuBuf;
    use crate::model::{MAGConfig, MemoryRuleKind};
    use crate::cuda_graph::GpuLevelScratch;
    use super::test_helpers::{rand_vec, max_abs_diff};

    /// Build a minimal GpuMemoryLevelParams from random weights for testing.
    fn make_level_params(d: usize, seed: u64) -> crate::gpu_params::GpuMemoryLevelParams {
        crate::gpu_params::GpuMemoryLevelParams {
            w_k_mem: GpuBuf::from_host(&rand_vec(d * d, seed)),
            w_v_mem: GpuBuf::from_host(&rand_vec(d * d, seed + 1)),
            w_q_mem: GpuBuf::from_host(&rand_vec(d * d, seed + 2)),
            w_alpha: GpuBuf::from_host(&rand_vec(2 * d, seed + 3)),
            b_alpha: GpuBuf::from_host(&[3.0f32]),
            w_theta: GpuBuf::from_host(&rand_vec(2 * d, seed + 4)),
            b_theta: GpuBuf::from_host(&[-4.6f32]),
            w_eta:   GpuBuf::from_host(&rand_vec(2 * d, seed + 5)),
            b_eta:   GpuBuf::from_host(&[2.0f32]),
            w_omega: GpuBuf::from_host(&rand_vec(d * 2 * d, seed + 6)),
            w_freq:  GpuBuf::zeros(1),
            b_freq:  GpuBuf::zeros(1),
            has_freq: false,
            w_k_conv: GpuBuf::zeros(1),
            b_k_conv: GpuBuf::zeros(1),
            w_q_conv: GpuBuf::zeros(1),
            b_q_conv: GpuBuf::zeros(1),
            has_conv: false,
            gate_proj: GpuBuf::zeros(1),
            up_proj:   GpuBuf::zeros(1),
            down_proj: GpuBuf::zeros(1),
            has_mlp: false,
            w_rand_cpu: vec![],
            b_rand_cpu: vec![],
            has_fm: false,
        }
    }

    /// Build a minimal MAGConfig for scratch path testing.
    fn make_cfg(d: usize, s: usize, rule: MemoryRuleKind) -> MAGConfig {
        let mut cfg = MAGConfig::test_config();
        cfg.swa.d_model = d;
        cfg.swa.num_heads = 2;
        cfg.swa.head_dim = d / 2;
        cfg.swa.seq_len = s;
        cfg.swa.window_size = s;
        cfg.memory_rule = rule;
        cfg
    }

    /// DeltaRule: scratch path matches gpu_memory_forward within 1e-5.
    #[test]
    fn test_dgd_scratch_matches_standard() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        let level_params = make_level_params(d, 100);
        let cfg = make_cfg(d, s, MemoryRuleKind::DeltaRule);
        let embedded = GpuBuf::from_host(&rand_vec(bs * s * d, 200));
        let m_init = rand_vec(bs * dd, 201);

        // ── Standard path: gpu_memory_forward ──
        let mut context_m_std = GpuBuf::from_host(&m_init);
        let (y_std, cache_std) = super::gpu_memory_forward(
            &level_params, &cfg, &embedded, &mut context_m_std,
            s, d, 0, bs,
        );
        crate::dispatch::cuda_sync();

        // Extract cache fields
        let (alpha_std, theta_std, k_norms_std, q_norms_std, k_std, q_std) = match &cache_std {
            super::GpuMemoryCache::Delta { alpha, theta, k_norms, q_norms, k_mem, q_mem, .. } =>
                (alpha, theta, k_norms, q_norms, k_mem, q_mem),
            _ => panic!("Expected Delta cache"),
        };

        // ── Scratch path: gpu_memory_forward_into_scratch ──
        let context_m_scratch = GpuBuf::from_host(&m_init);
        let mut scratch = GpuLevelScratch::new(bs, s, d, false);
        let ok = super::gpu_memory_forward_into_scratch(
            &level_params, &cfg, &embedded, &context_m_scratch,
            &mut scratch, s, d, 0, bs,
        );
        assert!(ok, "Scratch path should return true for DeltaRule");
        crate::dispatch::cuda_sync();

        // ── Compare y ──
        let mut y_std_h = vec![0.0f32; bs * s * d];
        let mut y_scr_h = vec![0.0f32; bs * s * d];
        y_std.copy_to_host(&mut y_std_h);
        scratch.y.copy_to_host(&mut y_scr_h);
        let y_diff = max_abs_diff(&y_std_h, &y_scr_h);

        // ── Compare alpha/theta ──
        let mut alpha_std_h = vec![0.0f32; bs * s];
        let mut alpha_scr_h = vec![0.0f32; bs * s];
        alpha_std.copy_to_host(&mut alpha_std_h);
        scratch.alpha.copy_to_host(&mut alpha_scr_h);
        let alpha_diff = max_abs_diff(&alpha_std_h, &alpha_scr_h);

        let mut theta_std_h = vec![0.0f32; bs * s];
        let mut theta_scr_h = vec![0.0f32; bs * s];
        theta_std.copy_to_host(&mut theta_std_h);
        scratch.theta.copy_to_host(&mut theta_scr_h);
        let theta_diff = max_abs_diff(&theta_std_h, &theta_scr_h);

        // ── Compare k_norms/q_norms ──
        let mut kn_std_h = vec![0.0f32; bs * s];
        let mut kn_scr_h = vec![0.0f32; bs * s];
        k_norms_std.copy_to_host(&mut kn_std_h);
        scratch.k_norms.copy_to_host(&mut kn_scr_h);
        let kn_diff = max_abs_diff(&kn_std_h, &kn_scr_h);

        let mut qn_std_h = vec![0.0f32; bs * s];
        let mut qn_scr_h = vec![0.0f32; bs * s];
        q_norms_std.copy_to_host(&mut qn_std_h);
        scratch.q_norms.copy_to_host(&mut qn_scr_h);
        let qn_diff = max_abs_diff(&qn_std_h, &qn_scr_h);

        // ── Compare normalized k/q ──
        let mut k_std_h = vec![0.0f32; bs * s * d];
        let mut k_scr_h = vec![0.0f32; bs * s * d];
        k_std.copy_to_host(&mut k_std_h);
        scratch.k_mem.copy_to_host(&mut k_scr_h);
        let k_diff = max_abs_diff(&k_std_h, &k_scr_h);

        let mut q_std_h = vec![0.0f32; bs * s * d];
        let mut q_scr_h = vec![0.0f32; bs * s * d];
        q_std.copy_to_host(&mut q_std_h);
        scratch.q_mem.copy_to_host(&mut q_scr_h);
        let q_diff = max_abs_diff(&q_std_h, &q_scr_h);

        eprintln!("DGD scratch vs standard — y={y_diff:.2e}, alpha={alpha_diff:.2e}, theta={theta_diff:.2e}, kn={kn_diff:.2e}, qn={qn_diff:.2e}, k={k_diff:.2e}, q={q_diff:.2e}");

        assert!(y_diff < 1e-5, "y mismatch: {y_diff}");
        assert!(alpha_diff < 1e-5, "alpha mismatch: {alpha_diff}");
        assert!(theta_diff < 1e-5, "theta mismatch: {theta_diff}");
        assert!(kn_diff < 1e-5, "k_norms mismatch: {kn_diff}");
        assert!(qn_diff < 1e-5, "q_norms mismatch: {qn_diff}");
        assert!(k_diff < 1e-5, "normalized k mismatch: {k_diff}");
        assert!(q_diff < 1e-5, "normalized q mismatch: {q_diff}");
    }

    /// TitansLMM: scratch path matches gpu_memory_forward within 1e-5.
    #[test]
    fn test_titans_scratch_matches_standard() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        let level_params = make_level_params(d, 300);
        let cfg = make_cfg(d, s, MemoryRuleKind::TitansLMM);
        let embedded = GpuBuf::from_host(&rand_vec(bs * s * d, 400));
        let m_init = rand_vec(bs * dd, 401);

        // ── Standard path: gpu_memory_forward ──
        let mut context_m_std = GpuBuf::from_host(&m_init);
        let (y_std, cache_std) = super::gpu_memory_forward(
            &level_params, &cfg, &embedded, &mut context_m_std,
            s, d, 0, bs,
        );
        crate::dispatch::cuda_sync();

        // Extract cache fields
        let (alpha_std, theta_std, eta_std, k_norms_std, q_norms_std) = match &cache_std {
            super::GpuMemoryCache::Titans { alpha, theta, eta, k_norms, q_norms, .. } =>
                (alpha, theta, eta, k_norms, q_norms),
            _ => panic!("Expected Titans cache"),
        };

        // ── Scratch path: gpu_memory_forward_into_scratch ──
        let context_m_scratch = GpuBuf::from_host(&m_init);
        let mut scratch = GpuLevelScratch::new(bs, s, d, true);
        let ok = super::gpu_memory_forward_into_scratch(
            &level_params, &cfg, &embedded, &context_m_scratch,
            &mut scratch, s, d, 0, bs,
        );
        assert!(ok, "Scratch path should return true for TitansLMM");
        crate::dispatch::cuda_sync();

        // ── Compare y ──
        let mut y_std_h = vec![0.0f32; bs * s * d];
        let mut y_scr_h = vec![0.0f32; bs * s * d];
        y_std.copy_to_host(&mut y_std_h);
        scratch.y.copy_to_host(&mut y_scr_h);
        let y_diff = max_abs_diff(&y_std_h, &y_scr_h);

        // ── Compare alpha/theta/eta ──
        let mut alpha_std_h = vec![0.0f32; bs * s];
        let mut alpha_scr_h = vec![0.0f32; bs * s];
        alpha_std.copy_to_host(&mut alpha_std_h);
        scratch.alpha.copy_to_host(&mut alpha_scr_h);
        let alpha_diff = max_abs_diff(&alpha_std_h, &alpha_scr_h);

        let mut theta_std_h = vec![0.0f32; bs * s];
        let mut theta_scr_h = vec![0.0f32; bs * s];
        theta_std.copy_to_host(&mut theta_std_h);
        scratch.theta.copy_to_host(&mut theta_scr_h);
        let theta_diff = max_abs_diff(&theta_std_h, &theta_scr_h);

        let mut eta_std_h = vec![0.0f32; bs * s];
        let mut eta_scr_h = vec![0.0f32; bs * s];
        eta_std.copy_to_host(&mut eta_std_h);
        scratch.eta.copy_to_host(&mut eta_scr_h);
        let eta_diff = max_abs_diff(&eta_std_h, &eta_scr_h);

        // ── Compare k_norms/q_norms ──
        let mut kn_std_h = vec![0.0f32; bs * s];
        let mut kn_scr_h = vec![0.0f32; bs * s];
        k_norms_std.copy_to_host(&mut kn_std_h);
        scratch.k_norms.copy_to_host(&mut kn_scr_h);
        let kn_diff = max_abs_diff(&kn_std_h, &kn_scr_h);

        let mut qn_std_h = vec![0.0f32; bs * s];
        let mut qn_scr_h = vec![0.0f32; bs * s];
        q_norms_std.copy_to_host(&mut qn_std_h);
        scratch.q_norms.copy_to_host(&mut qn_scr_h);
        let qn_diff = max_abs_diff(&qn_std_h, &qn_scr_h);

        eprintln!("Titans scratch vs standard — y={y_diff:.2e}, alpha={alpha_diff:.2e}, theta={theta_diff:.2e}, eta={eta_diff:.2e}, kn={kn_diff:.2e}, qn={qn_diff:.2e}");

        assert!(y_diff < 1e-5, "y mismatch: {y_diff}");
        assert!(alpha_diff < 1e-5, "alpha mismatch: {alpha_diff}");
        assert!(theta_diff < 1e-5, "theta mismatch: {theta_diff}");
        assert!(eta_diff < 1e-5, "eta mismatch: {eta_diff}");
        assert!(kn_diff < 1e-5, "k_norms mismatch: {kn_diff}");
        assert!(qn_diff < 1e-5, "q_norms mismatch: {qn_diff}");
    }

    /// Unsupported rule returns false (graceful fallback, no panic).
    #[test]
    fn test_scratch_unsupported_rule_returns_false() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        let level_params = make_level_params(d, 500);
        let cfg = make_cfg(d, s, MemoryRuleKind::HebbianRule);
        let embedded = GpuBuf::from_host(&rand_vec(bs * s * d, 600));
        let context_m = GpuBuf::from_host(&rand_vec(bs * dd, 601));
        let mut scratch = GpuLevelScratch::new(bs, s, d, false);

        let ok = super::gpu_memory_forward_into_scratch(
            &level_params, &cfg, &embedded, &context_m,
            &mut scratch, s, d, 0, bs,
        );
        assert!(!ok, "HebbianRule should return false (unsupported for scratch path)");
    }

    /// DeltaRule scratch path with non-default clamp bounds exercises the fused kernel's
    /// gate clamping logic (alpha_floor/ceil, theta_ceil).
    #[test]
    fn test_dgd_scratch_with_clamp_bounds() {
        let d = 32;
        let s = 8;
        let dd = d * d;
        let bs = 1;

        let level_params = make_level_params(d, 700);
        let mut cfg = make_cfg(d, s, MemoryRuleKind::DeltaRule);
        cfg.alpha_floor = vec![0.1];
        cfg.alpha_ceil = vec![0.9];
        cfg.theta_floor = vec![0.01];
        cfg.theta_ceil = vec![0.5];
        let embedded = GpuBuf::from_host(&rand_vec(bs * s * d, 800));
        let m_init = rand_vec(bs * dd, 801);

        // ── Standard path ──
        let mut context_m_std = GpuBuf::from_host(&m_init);
        let (y_std, cache_std) = super::gpu_memory_forward(
            &level_params, &cfg, &embedded, &mut context_m_std,
            s, d, 0, bs,
        );
        crate::dispatch::cuda_sync();

        let (alpha_std, theta_std) = match &cache_std {
            super::GpuMemoryCache::Delta { alpha, theta, .. } => (alpha, theta),
            _ => panic!("Expected Delta cache"),
        };

        // ── Scratch path ──
        let context_m_scratch = GpuBuf::from_host(&m_init);
        let mut scratch = GpuLevelScratch::new(bs, s, d, false);
        let ok = super::gpu_memory_forward_into_scratch(
            &level_params, &cfg, &embedded, &context_m_scratch,
            &mut scratch, s, d, 0, bs,
        );
        assert!(ok);
        crate::dispatch::cuda_sync();

        // ── Compare y ──
        let mut y_std_h = vec![0.0f32; bs * s * d];
        let mut y_scr_h = vec![0.0f32; bs * s * d];
        y_std.copy_to_host(&mut y_std_h);
        scratch.y.copy_to_host(&mut y_scr_h);
        let y_diff = max_abs_diff(&y_std_h, &y_scr_h);

        // ── Compare alpha (should be clamped to [0.1, 0.9]) ──
        let mut alpha_std_h = vec![0.0f32; bs * s];
        let mut alpha_scr_h = vec![0.0f32; bs * s];
        alpha_std.copy_to_host(&mut alpha_std_h);
        scratch.alpha.copy_to_host(&mut alpha_scr_h);
        let alpha_diff = max_abs_diff(&alpha_std_h, &alpha_scr_h);

        // Verify clamping is active: all alpha values within [0.1, 0.9]
        for (i, &a) in alpha_scr_h.iter().enumerate() {
            assert!(a >= 0.1 - 1e-6 && a <= 0.9 + 1e-6,
                "alpha[{i}]={a} outside clamped range [0.1, 0.9]");
        }

        // ── Compare theta (should be clamped to [0.01, 0.5]) ──
        let mut theta_std_h = vec![0.0f32; bs * s];
        let mut theta_scr_h = vec![0.0f32; bs * s];
        theta_std.copy_to_host(&mut theta_std_h);
        scratch.theta.copy_to_host(&mut theta_scr_h);
        let theta_diff = max_abs_diff(&theta_std_h, &theta_scr_h);

        for (i, &t) in theta_scr_h.iter().enumerate() {
            assert!(t >= 0.01 - 1e-6 && t <= 0.5 + 1e-6,
                "theta[{i}]={t} outside clamped range [0.01, 0.5]");
        }

        eprintln!("DGD scratch with clamps — y={y_diff:.2e}, alpha={alpha_diff:.2e}, theta={theta_diff:.2e}");

        assert!(y_diff < 1e-5, "y mismatch: {y_diff}");
        assert!(alpha_diff < 1e-5, "alpha mismatch: {alpha_diff}");
        assert!(theta_diff < 1e-5, "theta mismatch: {theta_diff}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests: Chunkwise frozen-M₀ kernels (spec 43)
// ══════════════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod chunkwise_tests {
    use crate::gpu_buf::GpuBuf;
    use super::test_helpers::{rand_vec, max_abs_diff};

    /// Delta chunkwise forward with chunk_size=1 must match per-token forward exactly.
    /// At C=1, frozen M₀ = M_{t-1} — identical to evolving-M_t per-token formulation.
    #[test]
    fn test_delta_chunkwise_c1_matches_per_token() {
        let d = 32;
        let s = 16;
        let dd = d * d;
        let bs = 1;
        let error_clip = 1.0f32;

        let k_mem = rand_vec(bs * s * d, 4300);
        let v_mem = rand_vec(bs * s * d, 4301);
        let q_mem = rand_vec(bs * s * d, 4302);
        // Small alpha (retention ≈ 0.95) and theta (learning rate ≈ 0.01)
        let alpha: Vec<f32> = (0..bs * s).map(|i| 0.05 + 0.001 * (i as f32)).collect();
        let theta: Vec<f32> = (0..bs * s).map(|i| 0.01 + 0.001 * (i as f32)).collect();
        let m_init = rand_vec(bs * dd, 4303);

        // GPU buffers
        let k_gpu = GpuBuf::from_host(&k_mem);
        let v_gpu = GpuBuf::from_host(&v_mem);
        let q_gpu = GpuBuf::from_host(&q_mem);
        let alpha_gpu = GpuBuf::from_host(&alpha);
        let theta_gpu = GpuBuf::from_host(&theta);
        let m_init_gpu = GpuBuf::from_host(&m_init);
        let m_init_slice = m_init_gpu.slice(0, bs * dd);

        // ── Per-token forward ──
        let mut m_states_pt = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_pt = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_init_slice, &mut m_states_pt, &mut y_pt, s, d, bs, s, d * d, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // ── Chunkwise forward with chunk_size=1 ──
        let chunk_size = 1;
        let num_chunks = s; // s/1 = s chunks
        let mut m_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut y_cw = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_chunkwise_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_init_slice, &mut m_chunk_states, &mut y_cw,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Compare y outputs
        let mut y_pt_host = vec![0.0f32; bs * s * d];
        let mut y_cw_host = vec![0.0f32; bs * s * d];
        y_pt.copy_to_host(&mut y_pt_host);
        y_cw.copy_to_host(&mut y_cw_host);
        let y_diff = max_abs_diff(&y_pt_host, &y_cw_host);
        eprintln!("Delta C=1 parity: y_diff = {y_diff:.2e}");
        assert!(y_diff < 1e-5, "Delta chunkwise C=1 y mismatch: {y_diff}");

        // Compare M_final (last chunk state vs last per-token state)
        let mut m_final_pt = vec![0.0f32; dd];
        let mut m_final_cw = vec![0.0f32; dd];
        let m_pt_slice = m_states_pt.slice(s * dd, dd);
        let m_cw_slice = m_chunk_states.slice(num_chunks * dd, dd);
        m_pt_slice.copy_to_host(&mut m_final_pt);
        m_cw_slice.copy_to_host(&mut m_final_cw);
        let m_diff = max_abs_diff(&m_final_pt, &m_final_cw);
        eprintln!("Delta C=1 parity: M_final_diff = {m_diff:.2e}");
        assert!(m_diff < 1e-5, "Delta chunkwise C=1 M_final mismatch: {m_diff}");
    }

    /// Titans chunkwise forward with chunk_size=1 must match per-token forward exactly.
    #[test]
    fn test_titans_chunkwise_c1_matches_per_token() {
        let d = 32;
        let s = 16;
        let dd = d * d;
        let bs = 1;
        let error_clip = 1.0f32;

        let k_mem = rand_vec(bs * s * d, 4310);
        let v_mem = rand_vec(bs * s * d, 4311);
        let q_mem = rand_vec(bs * s * d, 4312);
        let alpha: Vec<f32> = (0..bs * s).map(|i| 0.05 + 0.001 * (i as f32)).collect();
        let theta: Vec<f32> = (0..bs * s).map(|i| 0.01 + 0.001 * (i as f32)).collect();
        let eta: Vec<f32> = (0..bs * s).map(|i| 0.9 + 0.001 * (i as f32)).collect();
        let m_init = rand_vec(bs * dd, 4313);
        let s_init = vec![0.0f32; bs * dd];

        let k_gpu = GpuBuf::from_host(&k_mem);
        let v_gpu = GpuBuf::from_host(&v_mem);
        let q_gpu = GpuBuf::from_host(&q_mem);
        let alpha_gpu = GpuBuf::from_host(&alpha);
        let theta_gpu = GpuBuf::from_host(&theta);
        let eta_gpu = GpuBuf::from_host(&eta);
        let m_init_gpu = GpuBuf::from_host(&m_init);
        let s_init_gpu = GpuBuf::from_host(&s_init);
        let m_init_slice = m_init_gpu.slice(0, bs * dd);
        let s_init_slice = s_init_gpu.slice(0, bs * dd);

        // ── Per-token forward ──
        let mut m_states_pt = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut s_states_pt = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_pt = GpuBuf::zeros(bs * s * d);
        crate::dispatch::titans_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu, &eta_gpu,
            &m_init_slice, &s_init_slice,
            &mut m_states_pt, &mut s_states_pt, &mut y_pt, s, d, bs,
            s, d * d, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // ── Chunkwise forward with chunk_size=1 ──
        let chunk_size = 1;
        let num_chunks = s;
        let mut m_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut s_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut y_cw = GpuBuf::zeros(bs * s * d);
        crate::dispatch::titans_chunkwise_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu, &eta_gpu,
            &m_init_slice, &s_init_slice,
            &mut m_chunk_states, &mut s_chunk_states, &mut y_cw,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        let mut y_pt_host = vec![0.0f32; bs * s * d];
        let mut y_cw_host = vec![0.0f32; bs * s * d];
        y_pt.copy_to_host(&mut y_pt_host);
        y_cw.copy_to_host(&mut y_cw_host);
        let y_diff = max_abs_diff(&y_pt_host, &y_cw_host);
        eprintln!("Titans C=1 parity: y_diff = {y_diff:.2e}");
        assert!(y_diff < 1e-5, "Titans chunkwise C=1 y mismatch: {y_diff}");

        let mut m_final_pt = vec![0.0f32; dd];
        let mut m_final_cw = vec![0.0f32; dd];
        m_states_pt.slice(s * dd, dd).copy_to_host(&mut m_final_pt);
        m_chunk_states.slice(num_chunks * dd, dd).copy_to_host(&mut m_final_cw);
        let m_diff = max_abs_diff(&m_final_pt, &m_final_cw);
        eprintln!("Titans C=1 parity: M_final_diff = {m_diff:.2e}");
        assert!(m_diff < 1e-5, "Titans chunkwise C=1 M_final mismatch: {m_diff}");

        // S_final parity: momentum state must also match at C=1.
        let mut s_final_pt = vec![0.0f32; dd];
        let mut s_final_cw = vec![0.0f32; dd];
        s_states_pt.slice(s * dd, dd).copy_to_host(&mut s_final_pt);
        s_chunk_states.slice(num_chunks * dd, dd).copy_to_host(&mut s_final_cw);
        let s_diff = max_abs_diff(&s_final_pt, &s_final_cw);
        eprintln!("Titans C=1 parity: S_final_diff = {s_diff:.2e}");
        assert!(s_diff < 1e-5, "Titans chunkwise C=1 S_final mismatch: {s_diff}");
    }

    /// Delta chunkwise backward FD gradient check (d_k only).
    /// Uses chunk_size=s (single chunk) — the frozen-M₀ formulation.
    /// Validates d_k via central differences; other grads (d_v, d_q, d_alpha,
    /// d_theta) are exercised but not FD-checked here.
    #[test]
    fn test_delta_chunkwise_backward_fd() {
        let d = 8;
        let s = 4;
        let dd = d * d;
        let bs = 1;
        let chunk_size = s; // single chunk — full frozen-M₀ approximation
        let num_chunks = 1;
        let error_clip = 0.0f32; // no clip for cleaner FD
        let eps = 1e-2f32;
        let fd_tol = 0.10; // 10% relative tolerance

        let k_mem = rand_vec(bs * s * d, 4320);
        let v_mem = rand_vec(bs * s * d, 4321);
        let q_mem = rand_vec(bs * s * d, 4322);
        let alpha: Vec<f32> = (0..bs * s).map(|_| 0.05).collect();
        let theta: Vec<f32> = (0..bs * s).map(|_| 0.01).collect();
        let m_init = rand_vec(bs * dd, 4323);
        // Upstream gradient: uniform 1s for simple loss = sum(y)
        let d_y = vec![1.0f32; bs * s * d];

        // ── Analytical backward ──
        let k_gpu = GpuBuf::from_host(&k_mem);
        let v_gpu = GpuBuf::from_host(&v_mem);
        let q_gpu = GpuBuf::from_host(&q_mem);
        let alpha_gpu = GpuBuf::from_host(&alpha);
        let theta_gpu = GpuBuf::from_host(&theta);
        let m_init_gpu = GpuBuf::from_host(&m_init);
        let m_init_slice = m_init_gpu.slice(0, bs * dd);
        let d_y_gpu = GpuBuf::from_host(&d_y);

        // Forward
        let mut m_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut y = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_chunkwise_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_init_slice, &mut m_chunk_states, &mut y,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Backward
        let mut d_k = GpuBuf::zeros(bs * s * d);
        let mut d_v = GpuBuf::zeros(bs * s * d);
        let mut d_q = GpuBuf::zeros(bs * s * d);
        let mut d_alpha_buf = GpuBuf::zeros(bs * s);
        let mut d_theta_buf = GpuBuf::zeros(bs * s);
        let mut d_m_init = GpuBuf::zeros(dd);
        crate::dispatch::delta_chunkwise_backward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_chunk_states, &d_y_gpu,
            &mut d_k, &mut d_v, &mut d_q,
            &mut d_alpha_buf, &mut d_theta_buf, &mut d_m_init,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        let mut d_k_host = vec![0.0f32; bs * s * d];
        let mut d_v_host = vec![0.0f32; bs * s * d];
        d_k.copy_to_host(&mut d_k_host);
        d_v.copy_to_host(&mut d_v_host);

        // ── FD check on k_mem ──
        let loss_fn = |k_perturbed: &[f32]| -> f32 {
            let k_g = GpuBuf::from_host(k_perturbed);
            let v_g = GpuBuf::from_host(&v_mem);
            let q_g = GpuBuf::from_host(&q_mem);
            let a_g = GpuBuf::from_host(&alpha);
            let t_g = GpuBuf::from_host(&theta);
            let mi_g = GpuBuf::from_host(&m_init);
            let mi_s = mi_g.slice(0, bs * dd);
            let mut mc = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
            let mut yy = GpuBuf::zeros(bs * s * d);
            crate::dispatch::delta_chunkwise_forward_dd(
                &k_g, &v_g, &q_g, &a_g, &t_g,
                &mi_s, &mut mc, &mut yy,
                s, d, bs, chunk_size, error_clip, f32::MAX,
            );
            crate::dispatch::cuda_sync();
            let mut y_host = vec![0.0f32; bs * s * d];
            yy.copy_to_host(&mut y_host);
            y_host.iter().sum::<f32>()
        };

        let mut n_checked = 0;
        let mut n_passed = 0;
        let abs_threshold = 5e-4;
        for i in 0..k_mem.len().min(32) {
            let mut k_plus = k_mem.clone();
            let mut k_minus = k_mem.clone();
            k_plus[i] += eps;
            k_minus[i] -= eps;
            let fd_grad = (loss_fn(&k_plus) - loss_fn(&k_minus)) / (2.0 * eps);
            let anal_grad = d_k_host[i];
            if anal_grad.abs() < abs_threshold && fd_grad.abs() < abs_threshold {
                n_passed += 1;
                n_checked += 1;
                continue;
            }
            let rel_err = if anal_grad.abs() > 1e-8 {
                ((fd_grad - anal_grad) / anal_grad).abs()
            } else {
                (fd_grad - anal_grad).abs()
            };
            n_checked += 1;
            if rel_err < fd_tol { n_passed += 1; }
            if rel_err >= fd_tol {
                eprintln!("  d_k[{i}]: anal={anal_grad:.6e} fd={fd_grad:.6e} rel_err={rel_err:.4}");
            }
        }
        let pass_rate = n_passed as f64 / n_checked as f64;
        eprintln!("Delta chunkwise FD (d_k): {n_passed}/{n_checked} passed ({:.1}%)", pass_rate * 100.0);
        assert!(pass_rate >= 0.90, "Delta chunkwise d_k FD check failed: {pass_rate:.2}");
    }

    /// Titans chunkwise backward FD gradient check (d_k only).
    #[test]
    fn test_titans_chunkwise_backward_fd() {
        let d = 8;
        let s = 4;
        let dd = d * d;
        let bs = 1;
        let chunk_size = s;
        let num_chunks = 1;
        let error_clip = 0.0f32;
        let eps = 1e-2f32;
        let fd_tol = 0.10;

        let k_mem = rand_vec(bs * s * d, 4330);
        let v_mem = rand_vec(bs * s * d, 4331);
        let q_mem = rand_vec(bs * s * d, 4332);
        let alpha: Vec<f32> = (0..bs * s).map(|_| 0.05).collect();
        let theta: Vec<f32> = (0..bs * s).map(|_| 0.01).collect();
        let eta: Vec<f32> = (0..bs * s).map(|_| 0.9).collect();
        let m_init = rand_vec(bs * dd, 4333);
        let s_init = vec![0.0f32; bs * dd];
        let d_y = vec![1.0f32; bs * s * d];

        // ── Analytical backward ──
        let k_gpu = GpuBuf::from_host(&k_mem);
        let v_gpu = GpuBuf::from_host(&v_mem);
        let q_gpu = GpuBuf::from_host(&q_mem);
        let alpha_gpu = GpuBuf::from_host(&alpha);
        let theta_gpu = GpuBuf::from_host(&theta);
        let eta_gpu = GpuBuf::from_host(&eta);
        let m_init_gpu = GpuBuf::from_host(&m_init);
        let s_init_gpu = GpuBuf::from_host(&s_init);
        let m_init_slice = m_init_gpu.slice(0, bs * dd);
        let s_init_slice = s_init_gpu.slice(0, bs * dd);
        let d_y_gpu = GpuBuf::from_host(&d_y);

        // Forward
        let mut m_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut s_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut y = GpuBuf::zeros(bs * s * d);
        crate::dispatch::titans_chunkwise_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu, &eta_gpu,
            &m_init_slice, &s_init_slice,
            &mut m_chunk_states, &mut s_chunk_states, &mut y,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Backward
        let mut d_k = GpuBuf::zeros(bs * s * d);
        let mut d_v = GpuBuf::zeros(bs * s * d);
        let mut d_q = GpuBuf::zeros(bs * s * d);
        let mut d_alpha_buf = GpuBuf::zeros(bs * s);
        let mut d_theta_buf = GpuBuf::zeros(bs * s);
        let mut d_eta_buf = GpuBuf::zeros(bs * s);
        let mut d_m_init_buf = GpuBuf::zeros(dd);
        let mut d_s_init_buf = GpuBuf::zeros(dd);
        crate::dispatch::titans_chunkwise_backward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu, &eta_gpu,
            &m_chunk_states, &s_chunk_states, &d_y_gpu,
            &mut d_k, &mut d_v, &mut d_q,
            &mut d_alpha_buf, &mut d_theta_buf, &mut d_eta_buf,
            &mut d_m_init_buf, &mut d_s_init_buf,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        let mut d_k_host = vec![0.0f32; bs * s * d];
        d_k.copy_to_host(&mut d_k_host);

        // ── FD check on k_mem ──
        let loss_fn = |k_perturbed: &[f32]| -> f32 {
            let k_g = GpuBuf::from_host(k_perturbed);
            let v_g = GpuBuf::from_host(&v_mem);
            let q_g = GpuBuf::from_host(&q_mem);
            let a_g = GpuBuf::from_host(&alpha);
            let t_g = GpuBuf::from_host(&theta);
            let e_g = GpuBuf::from_host(&eta);
            let mi_g = GpuBuf::from_host(&m_init);
            let si_g = GpuBuf::from_host(&s_init);
            let mi_s = mi_g.slice(0, bs * dd);
            let si_s = si_g.slice(0, bs * dd);
            let mut mc = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
            let mut sc = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
            let mut yy = GpuBuf::zeros(bs * s * d);
            crate::dispatch::titans_chunkwise_forward_dd(
                &k_g, &v_g, &q_g, &a_g, &t_g, &e_g,
                &mi_s, &si_s,
                &mut mc, &mut sc, &mut yy,
                s, d, bs, chunk_size, error_clip, f32::MAX,
            );
            crate::dispatch::cuda_sync();
            let mut y_host = vec![0.0f32; bs * s * d];
            yy.copy_to_host(&mut y_host);
            y_host.iter().sum::<f32>()
        };

        let mut n_checked = 0;
        let mut n_passed = 0;
        let abs_threshold = 5e-4;
        for i in 0..k_mem.len().min(32) {
            let mut k_plus = k_mem.clone();
            let mut k_minus = k_mem.clone();
            k_plus[i] += eps;
            k_minus[i] -= eps;
            let fd_grad = (loss_fn(&k_plus) - loss_fn(&k_minus)) / (2.0 * eps);
            let anal_grad = d_k_host[i];
            if anal_grad.abs() < abs_threshold && fd_grad.abs() < abs_threshold {
                n_passed += 1;
                n_checked += 1;
                continue;
            }
            let rel_err = if anal_grad.abs() > 1e-8 {
                ((fd_grad - anal_grad) / anal_grad).abs()
            } else {
                (fd_grad - anal_grad).abs()
            };
            n_checked += 1;
            if rel_err < fd_tol { n_passed += 1; }
            if rel_err >= fd_tol {
                eprintln!("  d_k[{i}]: anal={anal_grad:.6e} fd={fd_grad:.6e} rel_err={rel_err:.4}");
            }
        }
        let pass_rate = n_passed as f64 / n_checked as f64;
        eprintln!("Titans chunkwise FD (d_k): {n_passed}/{n_checked} passed ({:.1}%)", pass_rate * 100.0);
        assert!(pass_rate >= 0.90, "Titans chunkwise d_k FD check failed: {pass_rate:.2}");
    }

    /// Delta chunkwise with chunk_size>1 should produce different y than per-token
    /// (validating that frozen-M₀ IS a different formulation, not just the same math).
    #[test]
    fn test_delta_chunkwise_c4_differs_from_per_token() {
        let d = 16;
        let s = 8;
        let dd = d * d;
        let bs = 1;
        let error_clip = 1.0f32;

        // Use larger values so the difference is visible
        let k_mem = rand_vec(bs * s * d, 4340);
        let v_mem = rand_vec(bs * s * d, 4341);
        let q_mem = rand_vec(bs * s * d, 4342);
        let alpha: Vec<f32> = (0..bs * s).map(|_| 0.05).collect();
        let theta: Vec<f32> = (0..bs * s).map(|_| 0.1).collect(); // larger theta to amplify difference
        let m_init = rand_vec(bs * dd, 4343);

        let k_gpu = GpuBuf::from_host(&k_mem);
        let v_gpu = GpuBuf::from_host(&v_mem);
        let q_gpu = GpuBuf::from_host(&q_mem);
        let alpha_gpu = GpuBuf::from_host(&alpha);
        let theta_gpu = GpuBuf::from_host(&theta);
        let m_init_gpu = GpuBuf::from_host(&m_init);
        let m_init_slice = m_init_gpu.slice(0, bs * dd);

        // Per-token
        let mut m_states_pt = GpuBuf::zeros(bs * (s + 1) * dd);
        let mut y_pt = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_init_slice, &mut m_states_pt, &mut y_pt, s, d, bs, s, d * d, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        // Chunkwise with chunk_size=4 (2 chunks of 4 tokens each)
        let chunk_size = 4;
        let num_chunks = 2;
        let mut m_chunk_states = GpuBuf::zeros(bs * (num_chunks + 1) * dd);
        let mut y_cw = GpuBuf::zeros(bs * s * d);
        crate::dispatch::delta_chunkwise_forward_dd(
            &k_gpu, &v_gpu, &q_gpu, &alpha_gpu, &theta_gpu,
            &m_init_slice, &mut m_chunk_states, &mut y_cw,
            s, d, bs, chunk_size, error_clip, f32::MAX,
        );
        crate::dispatch::cuda_sync();

        let mut y_pt_host = vec![0.0f32; bs * s * d];
        let mut y_cw_host = vec![0.0f32; bs * s * d];
        y_pt.copy_to_host(&mut y_pt_host);
        y_cw.copy_to_host(&mut y_cw_host);
        let y_diff = max_abs_diff(&y_pt_host, &y_cw_host);
        eprintln!("Delta C=4 vs per-token: y_diff = {y_diff:.2e}");

        // Should differ — frozen-M₀ uses chunk-start M, not evolving M_t
        // First token in each chunk should match (t=0 sees same M₀), but subsequent
        // tokens within a chunk will diverge since errors differ.
        assert!(y_diff > 1e-6, "Expected difference between C=4 and per-token, got {y_diff}");
    }
}
