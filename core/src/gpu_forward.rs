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
    pub attn_weights_bf16: GpuBuf<u16>, // [bs*nh, s, ws] bf16
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
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
        k_norms: GpuBuf<f32>,  // [s] — L2 norms before normalization
        q_norms: GpuBuf<f32>,  // [s]
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
        k_norms: GpuBuf<f32>,
        q_norms: GpuBuf<f32>,
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
    /// TNT hierarchical — per-shard inner caches + global state trajectory.
    /// The inner caches are Titans/Delta/etc. caches, one per shard.
    TNT {
        /// Inner memory cache per shard (Titans/Delta/Hebbian variant).
        shard_inner_caches: Vec<GpuMemoryCache>,
        /// Local outputs per shard [shard_len, d] — needed for summary backward.
        shard_y_bufs: Vec<GpuBuf<f32>>,
        /// Summary key vectors [d] per shard.
        k_summaries: Vec<GpuBuf<f32>>,
        /// Summary value vectors [d] per shard.
        v_summaries: Vec<GpuBuf<f32>>,
        /// Global M state BEFORE each shard update [d*d] per shard (num_shards entries).
        global_m_before: Vec<GpuBuf<f32>>,
        /// TNT config: global_chunk_size, local_chunk_size.
        global_chunk_size: usize,
        local_chunk_size: usize,
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
    pub fn dgd_delta_norm(&self, s: usize, d: usize, batch_size: usize) -> f32 {
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
            let mut norm_out = GpuBuf::<f32>::zeros(1);
            unsafe {
                crate::cuda_ffi::dgd_delta_norm_cuda(m_ptr, k_ptr, v_ptr, norm_out.ptr(), d as i32);
            }
            crate::dispatch::cuda_sync();
            let mut host = [0.0f32; 1];
            norm_out.copy_to_host(&mut host);
            host[0]
        };

        match self {
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
                    let delta = shard_cache.dgd_delta_norm(shard_s, d, 1);
                    if delta > max_delta {
                        max_delta = delta;
                    }
                }
                max_delta
            }
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
            | GpuMemoryCache::DGDCkpt { alpha, .. } => Some(alpha),
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
            | GpuMemoryCache::TitansCkpt { eta, .. } => Some(eta),
            GpuMemoryCache::Delta { .. }
            | GpuMemoryCache::Hebbian { .. }
            | GpuMemoryCache::DGD { .. }
            | GpuMemoryCache::DeltaCkpt { .. }
            | GpuMemoryCache::HebbianCkpt { .. }
            | GpuMemoryCache::DGDCkpt { .. }
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
            | GpuMemoryCache::DGDCkpt { theta, .. } => Some(theta),
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
    if context.cuda_graph.should_capture() && context.forward_scratch.is_some() {
        // Check that the memory rule is supported for graph capture.
        let is_tnt_mode = cfg.parallel.as_ref()
            .map(|p| p.strategy == ParallelStrategy::TNTHierarchical)
            .unwrap_or(false);
        // Spec 25: tape_multiplier routes levels to Ckpt path, which is incompatible
        // with CUDA graph capture. Disable capture if any level would use Ckpt.
        let has_ckpt = cfg.checkpoint_interval.is_some()
            || cfg.tape_multiplier.map_or(false, |m| m > 0);
        let can_capture = !cfg.residual
            && matches!(cfg.memory_rule, MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM)
            && !has_ckpt
            && !is_tnt_mode;

        if can_capture {
            gpu_cms_capture_all_patterns(params, cfg, &input_ids_i32, &target_ids_i32, pulse, context, bs);
        } else {
            // Rule not supported for graph capture — disable permanently.
            eprintln!("[cuda_graph] Capture skipped: rule {:?} not supported. Falling through to standard dispatch.", cfg.memory_rule);
            context.cuda_graph.disable();
        }
    }

    // ── CUDA Graph replay path ─────────────────────────────────────────
    if context.cuda_graph.should_replay(bitmask) {
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
    let mut embedded = GpuBuf::<f32>::zeros(bs * s * d);
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
        let mut out = GpuBuf::<f32>::zeros(n_tokens * d);
        let mut mean = GpuBuf::<f32>::zeros(n_tokens);
        let mut rstd = GpuBuf::<f32>::zeros(n_tokens);
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
    let aw_total = bs * nh * s * ws;
    let total_i32 = i32::try_from(total).expect("bs*s*d exceeds i32::MAX");
    let mut q_bf16 = GpuBuf::<u16>::zeros(total);
    let mut k_bf16 = GpuBuf::<u16>::zeros(total);
    let mut v_bf16 = GpuBuf::<u16>::zeros(total);
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
        s, nh, hd, ws, bs,
    );

    // bf16 → f32 for attn_out (needed for gating)
    let mut attn_out = GpuBuf::<f32>::zeros(total);
    unsafe {
        crate::cuda_ffi::bf16_to_f32_cuda(attn_out_bf16.as_ptr(), attn_out.ptr(), total_i32);
    }

    // ── Residual skip 1: embedded + attn_out ────────────────────────
    let (residual_after_attn, ln_mem_out, ln_mem_mean, ln_mem_rstd) = if cfg.residual {
        // residual = embedded + attn_out
        let mut residual = GpuBuf::<f32>::zeros(n_tokens * d);
        unsafe {
            // Copy embedded, then add attn_out
            crate::cuda_ffi::saxpy_cuda(0.0, embedded.as_ptr(), residual.ptr(), total_i32); // zero residual
        }
        unsafe {
            crate::cuda_ffi::saxpy_cuda(1.0, embedded.as_ptr(), residual.ptr(), total_i32);
            crate::cuda_ffi::saxpy_cuda(1.0, attn_out.as_ptr(), residual.ptr(), total_i32);
        }
        // LN_mem on residual
        let mut ln_out = GpuBuf::<f32>::zeros(n_tokens * d);
        let mut mean = GpuBuf::<f32>::zeros(n_tokens);
        let mut rstd = GpuBuf::<f32>::zeros(n_tokens);
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
            // Each batch element has distinct embeddings, so compute Y = Q @ M^T
            // for all bs*s tokens simultaneously. Same frozen M for all batch elements.
            let y_level = gpu_memory_read_only(
                &params.levels[level], mem_input,
                &context.memory[level],
                bs * s, d,
            );
            y_per_level.push(y_level);
            memory_caches.push(None);
        }
    }

    // ── Combine levels: y_combined = sum with 1/sqrt(k) for k>2 ───────
    let mut y_combined = GpuBuf::<f32>::zeros(bs * s * d);
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
    let mut gate = GpuBuf::<f32>::zeros(bs * s * d);
    let mut gated_out = GpuBuf::<f32>::zeros(bs * s * d);
    let mut projected = GpuBuf::<f32>::zeros(bs * s * d);

    let residual_final = if cfg.residual {
        // Residual path: residual_final = residual_after_attn + y_combined (additive, no sigmoid)
        let residual_attn = residual_after_attn.as_ref().unwrap();
        let mut res_final = GpuBuf::<f32>::zeros(n_tokens * d);
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
    let mut loss_gpu = GpuBuf::<f32>::zeros(1);
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

/// Compute memory projections + gates + inner loop for an active level, all on GPU.
///
/// `embedded` has shape [batch_size * s, d] (flat batch).
/// `context_m` is the carry-forward M state [d*d] — broadcast to all batch elements,
/// then element-0's final M is written back after the forward pass.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_forward(
    level_params: &crate::gpu_params::GpuMemoryLevelParams,
    cfg: &MAGConfig,
    embedded: &GpuBuf<f32>,
    context_m: &mut GpuBuf<f32>,   // [d*d] — carry-forward, updated with elem-0's final M
    s: usize,
    d: usize,
    level: usize,
    batch_size: usize,
) -> (GpuBuf<f32>, GpuMemoryCache) {
    let bs = batch_size;
    let dd = d * d;
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

    // L2-normalize keys and queries (Titans paper: "normalize queries and keys using l_2-norm")
    let mut k_norms = GpuBuf::zeros(bs * s);
    let mut q_norms = GpuBuf::zeros(bs * s);
    unsafe {
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(k_mem.ptr(), k_norms.ptr(), tokens_i32, d_i32, 1e-8);
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(q_mem.ptr(), q_norms.ptr(), tokens_i32, d_i32, 1e-8);
    }

    // Compute per-token gates: alpha[bs*s], theta[bs*s]
    // b_alpha and b_theta are passed as device pointers (CUDA-graph-capture-safe):
    // the graph captures the stable pointer; optimizer updates the value in-place.
    let mut alpha = GpuBuf::zeros(bs * s);
    let mut theta = GpuBuf::zeros(bs * s);

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
        let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
        let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
        if alpha_floor > 0.0 || alpha_ceil < 1.0 {
            crate::cuda_ffi::clamp_f32_cuda(alpha.ptr(), tokens_i32, alpha_floor, alpha_ceil);
        }
        // CS-39: clamp post-softplus theta to [floor, ceil] per level.
        let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
        let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
        if theta_floor > 0.0 || theta_ceil < f32::MAX {
            crate::cuda_ffi::clamp_f32_cuda(theta.ptr(), tokens_i32, theta_floor, theta_ceil);
        }
    }

    // context_m is [bs * dd] — slot b's initial M occupies offset b*dd.
    // No broadcast: pass context_m directly; the kernel reads m_initial + b*dd per element.
    let m_initial_slice = context_m.slice(0, bs * dd);
    let m_norm_max = cfg.max_m_norm(level);

    // Spec 25: boundary-scoped Wengert tape. Use effective_checkpoint_interval
    // which accounts for tape_multiplier — derives per-level intervals from chunk_sizes.
    // When tape_multiplier is None/0, falls back to cfg.checkpoint_interval (legacy).
    let eff_ckpt = cfg.effective_checkpoint_interval(level);

    match (eff_ckpt, cfg.memory_rule) {
        // ── Full-trajectory paths (checkpoint_interval=None, current behavior) ──
        (None, MemoryRuleKind::DeltaRule) => {
            let mut m_states = GpuBuf::zeros(bs * (s + 1) * dd);
            let mut y = GpuBuf::zeros(bs * s * d);
            crate::dispatch::delta_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta,
                &m_initial_slice, &mut m_states, &mut y, s, d, bs,
                cfg.error_clip_for_level(level),
            );
            crate::dispatch::cuda_sync();
            // Copy all bs slots' final M back: element b's final M at m_states offset b*(s+1)*dd + s*dd.
            copy_final_m_batch(&m_states, context_m, s, dd, bs);
            for b in 0..bs {
                unsafe {
                    crate::cuda_ffi::m_norm_clamp_f32_cuda(
                        (context_m.ptr() as *mut u8).add(b * dd * 4) as *mut f32,
                        d_i32, m_norm_max,
                    );
                }
            }
            (y, GpuMemoryCache::Delta { k_mem, v_mem, q_mem, alpha, theta, m_states, k_norms, q_norms })
        }
        (None, MemoryRuleKind::TitansLMM) => {
            // Compute eta gate for all bs*s tokens (Titans uses 3 gates: alpha, theta, eta)
            let eta = compute_eta(level_params, &k_mem, &v_mem, bs * s, d);
            // s_initial: all-zeros per batch element (bs * dd floats)
            let batch_s_initial = GpuBuf::zeros(bs * dd);
            let s_initial_slice = batch_s_initial.slice(0, bs * dd);
            let mut m_states = GpuBuf::zeros(bs * (s + 1) * dd);
            let mut s_states = GpuBuf::zeros(bs * (s + 1) * dd);
            let mut y = GpuBuf::zeros(bs * s * d);
            crate::dispatch::titans_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
                &m_initial_slice, &s_initial_slice,
                &mut m_states, &mut s_states, &mut y, s, d, bs,
                cfg.error_clip_for_level(level),
            );
            crate::dispatch::cuda_sync();
            copy_final_m_batch(&m_states, context_m, s, dd, bs);
            for b in 0..bs {
                unsafe {
                    crate::cuda_ffi::m_norm_clamp_f32_cuda(
                        (context_m.ptr() as *mut u8).add(b * dd * 4) as *mut f32,
                        d_i32, m_norm_max,
                    );
                }
            }
            (y, GpuMemoryCache::Titans { k_mem, v_mem, q_mem, alpha, theta, eta, m_states, s_states, k_norms, q_norms })
        }
        (None, MemoryRuleKind::HebbianRule) => {
            assert_eq!(bs, 1, "Hebbian GPU forward with batch_size > 1 is not supported");
            let mut m_states = GpuBuf::zeros(bs * (s + 1) * dd);
            let mut y = GpuBuf::zeros(bs * s * d);
            crate::dispatch::hebbian_forward_dd(
                &k_mem, &v_mem, &q_mem, &alpha,
                &m_initial_slice, &mut m_states, &mut y, s, d,
            );
            crate::dispatch::cuda_sync();
            // Hebbian is bs=1 only (asserted above), single slot copy.
            copy_final_m_batch(&m_states, context_m, s, dd, 1);
            unsafe { crate::cuda_ffi::m_norm_clamp_f32_cuda(context_m.ptr(), d_i32, m_norm_max); }
            (y, GpuMemoryCache::Hebbian { k_mem, v_mem, q_mem, alpha, m_states, k_norms, q_norms })
        }
        // ── Checkpointed paths (checkpoint_interval=Some(c)) ──
        // Gradient checkpointing with batch_size>1 is not supported — ablation configs
        // do not use checkpoint_interval, so this combination never occurs in practice.
        (Some(c), MemoryRuleKind::DeltaRule) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            let m_initial = context_m.slice(0, dd);
            let num_ckpt = checkpoint_count(s, c);
            let mut m_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::delta_forward_dd_ckpt(
                &k_mem, &v_mem, &q_mem, &alpha, &theta,
                &m_initial, &mut m_checkpoints, &mut y, s, d, c,
                cfg.error_clip_for_level(level),
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            unsafe { crate::cuda_ffi::m_norm_clamp_f32_cuda(context_m.ptr(), d_i32, m_norm_max); }
            (y, GpuMemoryCache::DeltaCkpt { k_mem, v_mem, q_mem, alpha, theta, m_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        (Some(c), MemoryRuleKind::TitansLMM) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            let m_initial = context_m.slice(0, dd);
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
                cfg.error_clip_for_level(level),
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            unsafe { crate::cuda_ffi::m_norm_clamp_f32_cuda(context_m.ptr(), d_i32, m_norm_max); }
            (y, GpuMemoryCache::TitansCkpt { k_mem, v_mem, q_mem, alpha, theta, eta, m_checkpoints, s_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        (Some(c), MemoryRuleKind::HebbianRule) => {
            assert_eq!(bs, 1, "checkpoint_interval with batch_size>1 not supported");
            let m_initial = context_m.slice(0, dd);
            let num_ckpt = checkpoint_count(s, c);
            let mut m_checkpoints = GpuBuf::zeros(num_ckpt * dd);
            let mut y = GpuBuf::zeros(s * d);
            crate::dispatch::hebbian_forward_dd_ckpt(
                &k_mem, &v_mem, &q_mem, &alpha,
                &m_initial, &mut m_checkpoints, &mut y, s, d, c,
            );
            crate::dispatch::cuda_sync();
            copy_final_m(&m_checkpoints, context_m, (num_ckpt - 1) * dd, dd);
            unsafe { crate::cuda_ffi::m_norm_clamp_f32_cuda(context_m.ptr(), d_i32, m_norm_max); }
            (y, GpuMemoryCache::HebbianCkpt { k_mem, v_mem, q_mem, alpha, m_checkpoints, checkpoint_interval: c, k_norms, q_norms })
        }
        // ── SwiGLU: stateless MLP, no M state, no m_norm_clamp ──────────
        (_, MemoryRuleKind::SwiGluMlp) => {
            let inter = cfg.intermediate_size;
            let mut gate_buf  = GpuBuf::zeros(bs * s * inter);
            let mut up_buf    = GpuBuf::zeros(bs * s * inter);
            let mut fused_buf = GpuBuf::zeros(bs * s * inter);
            let mut cache_buf = GpuBuf::zeros(bs * s * inter);
            let mut y = GpuBuf::zeros(bs * s * d);
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
        _ => panic!("GPU-resident forward only supports DeltaRule, TitansLMM, HebbianRule, DGD, SwiGluMlp. Got {:?}", cfg.memory_rule),
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

    let dd = d * d;
    let cg = parallel_cfg.tnt_global_chunk_size;
    let cl = parallel_cfg.tnt_local_chunk_size;
    assert!(cl <= cg && cl >= 1 && cg >= 1);

    let num_shards = (s + cg - 1) / cg;
    let d_i32 = i32::try_from(d).expect("d exceeds i32::MAX");
    let m_norm_max = cfg.max_m_norm(level);

    // Full output buffer: [s, d]
    let mut y_full = GpuBuf::<f32>::zeros(s * d);

    // Per-shard caches for backward
    let mut shard_inner_caches = Vec::with_capacity(num_shards);
    let mut shard_y_bufs = Vec::with_capacity(num_shards);
    let mut k_summaries = Vec::with_capacity(num_shards);
    let mut v_summaries = Vec::with_capacity(num_shards);
    let mut global_m_before = Vec::with_capacity(num_shards);

    for shard_idx in 0..num_shards {
        let shard_start = shard_idx * cg;
        let shard_end = (shard_start + cg).min(s);
        let shard_len = shard_end - shard_start;

        // Number of local chunks in this shard
        let n_batch = (shard_len + cl - 1) / cl;

        // Save global M state before this shard's update (for backward)
        let mut m_snapshot = GpuBuf::<f32>::zeros(dd);
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                m_snapshot.ptr() as *mut std::ffi::c_void,
                context_m.as_ptr() as *const std::ffi::c_void,
                dd * 4,
            );
            assert_eq!(rc, 0);
        }
        global_m_before.push(m_snapshot);

        // Step 1: Broadcast global M → N copies for local memories
        let mut m_broadcast = GpuBuf::<f32>::zeros(n_batch * dd);
        crate::dispatch::tnt_broadcast_m_dd(context_m, &mut m_broadcast, n_batch, d);

        // Step 2: Compute memory projections for the shard's tokens
        let shard_embedded_slice = embedded.slice(shard_start * d, shard_len * d);
        let shard_tokens_i32 = i32::try_from(shard_len).expect("shard_len exceeds i32::MAX");

        let mut k_mem = GpuBuf::zeros(shard_len * d);
        let mut v_mem = GpuBuf::zeros(shard_len * d);
        let mut q_mem = GpuBuf::zeros(shard_len * d);

        // Copy shard embeddings to owned buffer for cuBLAS
        let mut shard_embedded = GpuBuf::<f32>::zeros(shard_len * d);
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
        let mut shard_k_norms = GpuBuf::zeros(shard_len);
        let mut shard_q_norms = GpuBuf::zeros(shard_len);
        unsafe {
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(k_mem.ptr(), shard_k_norms.ptr(), shard_tokens_i32, d_i32, 1e-8);
            crate::cuda_ffi::l2_normalize_rows_f32_cuda(q_mem.ptr(), shard_q_norms.ptr(), shard_tokens_i32, d_i32, 1e-8);
        }

        // Step 3: Compute gates for shard tokens
        let mut alpha = GpuBuf::zeros(shard_len);
        let mut theta = GpuBuf::zeros(shard_len);
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

        // Step 4: Pad to [n_batch, cl, d] layout for batched kernel
        // Tokens within a shard are already in sequential local-chunk order.
        // Pad the last local chunk to cl tokens with zeros if needed.
        let padded_len = n_batch * cl;
        let (k_mem_b, v_mem_b, q_mem_b, alpha_b, theta_b) = if shard_len == padded_len {
            // No padding needed — exact fit
            (k_mem, v_mem, q_mem, alpha, theta)
        } else {
            // Pad with zeros (allocate padded buffers, copy actual data)
            let mut kp = GpuBuf::zeros(padded_len * d);
            let mut vp = GpuBuf::zeros(padded_len * d);
            let mut qp = GpuBuf::zeros(padded_len * d);
            let mut ap = GpuBuf::zeros(padded_len);
            let mut tp = GpuBuf::zeros(padded_len);
            unsafe {
                let rc = gpu_buf_memcpy_d2d(kp.ptr() as *mut _, k_mem.as_ptr() as *const _, shard_len * d * 4);
                assert_eq!(rc, 0, "TNT pad copy kp failed (rc={rc})");
                let rc = gpu_buf_memcpy_d2d(vp.ptr() as *mut _, v_mem.as_ptr() as *const _, shard_len * d * 4);
                assert_eq!(rc, 0, "TNT pad copy vp failed (rc={rc})");
                let rc = gpu_buf_memcpy_d2d(qp.ptr() as *mut _, q_mem.as_ptr() as *const _, shard_len * d * 4);
                assert_eq!(rc, 0, "TNT pad copy qp failed (rc={rc})");
                let rc = gpu_buf_memcpy_d2d(ap.ptr() as *mut _, alpha.as_ptr() as *const _, shard_len * 4);
                assert_eq!(rc, 0, "TNT pad copy alpha failed (rc={rc})");
                let rc = gpu_buf_memcpy_d2d(tp.ptr() as *mut _, theta.as_ptr() as *const _, shard_len * 4);
                assert_eq!(rc, 0, "TNT pad copy theta failed (rc={rc})");
            }
            (kp, vp, qp, ap, tp)
        };

        // Step 5: Run the batched memory kernel with batch_size=n_batch, seq_len=cl
        let m_initial_slice = m_broadcast.slice(0, n_batch * dd);
        let mut y_local = GpuBuf::zeros(padded_len * d);

        let inner_cache = match cfg.memory_rule {
            MemoryRuleKind::TitansLMM => {
                // Compute eta gate for shard tokens (unpadded first, then pad)
                let mut eta = GpuBuf::zeros(shard_len);
                unsafe {
                    crate::cuda_ffi::gate_compute_cuda(
                        k_mem_b.as_ptr(), v_mem_b.as_ptr(), level_params.w_eta.as_ptr(),
                        level_params.b_eta.as_ptr(), eta.ptr(),
                        shard_tokens_i32, d_i32, 0,
                    );
                }
                let eta_b = if shard_len == padded_len {
                    eta
                } else {
                    let mut ep = GpuBuf::zeros(padded_len);
                    unsafe {
                        let rc = gpu_buf_memcpy_d2d(ep.ptr() as *mut _, eta.as_ptr() as *const _, shard_len * 4);
                        assert_eq!(rc, 0, "TNT pad copy eta failed (rc={rc})");
                    }
                    ep
                };

                let s_initial_buf = GpuBuf::zeros(n_batch * dd);
                let s_initial_slice = s_initial_buf.slice(0, n_batch * dd);
                let mut m_states = GpuBuf::zeros(n_batch * (cl + 1) * dd);
                let mut s_states = GpuBuf::zeros(n_batch * (cl + 1) * dd);
                crate::dispatch::titans_forward_dd(
                    &k_mem_b, &v_mem_b, &q_mem_b,
                    &alpha_b, &theta_b, &eta_b,
                    &m_initial_slice, &s_initial_slice,
                    &mut m_states, &mut s_states, &mut y_local, cl, d, n_batch,
                    cfg.error_clip_for_level(level),
                );
                GpuMemoryCache::Titans {
                    k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                    alpha: alpha_b, theta: theta_b, eta: eta_b,
                    m_states, s_states,
                    k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                }
            }
            MemoryRuleKind::DeltaRule => {
                let mut m_states = GpuBuf::zeros(n_batch * (cl + 1) * dd);
                crate::dispatch::delta_forward_dd(
                    &k_mem_b, &v_mem_b, &q_mem_b,
                    &alpha_b, &theta_b,
                    &m_initial_slice, &mut m_states, &mut y_local, cl, d, n_batch,
                    cfg.error_clip_for_level(level),
                );
                GpuMemoryCache::Delta {
                    k_mem: k_mem_b, v_mem: v_mem_b, q_mem: q_mem_b,
                    alpha: alpha_b, theta: theta_b, m_states,
                    k_norms: shard_k_norms.dup(), q_norms: shard_q_norms.dup(),
                }
            }
            _ => unreachable!(), // asserted above
        };

        crate::dispatch::cuda_sync();

        // Step 6: Copy unpadded local outputs to full output buffer
        // y_local is [n_batch, cl, d] batched layout — we need [shard_len, d]
        // Since batched layout is contiguous and matches the sequential order,
        // just copy the first shard_len*d elements.
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                (y_full.ptr() as *mut u8).add(shard_start * d * 4) as *mut std::ffi::c_void,
                y_local.as_ptr() as *const std::ffi::c_void,
                shard_len * d * 4,
            );
            assert_eq!(rc, 0, "TNT y_full D2D memcpy failed (rc={rc})");
        }

        // Step 7: Compute shard summary (mean-pooling on GPU)
        // Use the unpadded shard output for summary
        let mut shard_y = GpuBuf::<f32>::zeros(shard_len * d);
        unsafe {
            let rc = gpu_buf_memcpy_d2d(
                shard_y.ptr() as *mut std::ffi::c_void,
                y_local.as_ptr() as *const std::ffi::c_void,
                shard_len * d * 4,
            );
            assert_eq!(rc, 0, "TNT shard_y D2D memcpy failed (rc={rc})");
        }
        let mut k_sum = GpuBuf::<f32>::zeros(d);
        let mut v_sum = GpuBuf::<f32>::zeros(d);
        crate::dispatch::tnt_shard_summary_mean_dd(&shard_y, &mut k_sum, &mut v_sum, shard_len, d);

        // Step 8: Update global M via outer product
        crate::dispatch::tnt_global_update_dd(context_m, &k_sum, &v_sum, d, 0.95);
        crate::dispatch::cuda_sync();

        // Apply m_norm_clamp to global M after update
        unsafe {
            crate::cuda_ffi::m_norm_clamp_f32_cuda(context_m.ptr(), d_i32, m_norm_max);
        }

        // Save caches for backward
        shard_inner_caches.push(inner_cache);
        shard_y_bufs.push(shard_y);
        k_summaries.push(k_sum);
        v_summaries.push(v_sum);
    }

    (y_full, GpuMemoryCache::TNT {
        shard_inner_caches, shard_y_bufs, k_summaries, v_summaries,
        global_m_before, global_chunk_size: cg, local_chunk_size: cl,
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
    let tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let d_i32      = i32::try_from(d).expect("d_model exceeds i32::MAX");

    // Memory projections into pre-allocated scratch.k_mem, v_mem, q_mem
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_k_mem, &mut scratch.k_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_v_mem, &mut scratch.v_mem, bs * s, d, d, 0.0);
    crate::dispatch::cublas_matmul_transb_dd(embedded, &level_params.w_q_mem, &mut scratch.q_mem, bs * s, d, d, 0.0);

    // L2-normalize keys and queries (Titans paper: "normalize queries and keys using l_2-norm")
    unsafe {
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(scratch.k_mem.ptr(), scratch.k_norms.ptr(), tokens_i32, d_i32, 1e-8);
        crate::cuda_ffi::l2_normalize_rows_f32_cuda(scratch.q_mem.ptr(), scratch.q_norms.ptr(), tokens_i32, d_i32, 1e-8);
    }

    // Gates into pre-allocated scratch.alpha, theta (device pointers — graph-capture-safe)
    unsafe {
        crate::cuda_ffi::gate_compute_cuda(
            scratch.k_mem.as_ptr(), scratch.v_mem.as_ptr(), level_params.w_alpha.as_ptr(),
            level_params.b_alpha.as_ptr(), scratch.alpha.ptr(),
            tokens_i32, d_i32, 0, // 0=sigmoid
        );
        crate::cuda_ffi::gate_compute_cuda(
            scratch.k_mem.as_ptr(), scratch.v_mem.as_ptr(), level_params.w_theta.as_ptr(),
            level_params.b_theta.as_ptr(), scratch.theta.ptr(),
            tokens_i32, d_i32, 1, // 1=softplus
        );
        let alpha_floor = cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
        let alpha_ceil  = cfg.alpha_ceil.get(level).copied().unwrap_or(1.0);
        if alpha_floor > 0.0 || alpha_ceil < 1.0 {
            crate::cuda_ffi::clamp_f32_cuda(scratch.alpha.ptr(), tokens_i32, alpha_floor, alpha_ceil);
        }
        let theta_floor = cfg.theta_floor.get(level).copied().unwrap_or(0.0);
        let theta_ceil  = cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
        if theta_floor > 0.0 || theta_ceil < f32::MAX {
            crate::cuda_ffi::clamp_f32_cuda(scratch.theta.ptr(), tokens_i32, theta_floor, theta_ceil);
        }
    }

    let m_initial_slice = context_m.slice(0, bs * dd);

    match (cfg.checkpoint_interval, cfg.memory_rule) {
        (None, MemoryRuleKind::DeltaRule) => {
            // Zero scratch.m_states at start of each step (initial M from context_m is copied in)
            scratch.m_states.zero();
            crate::dispatch::delta_forward_dd(
                &scratch.k_mem, &scratch.v_mem, &scratch.q_mem,
                &scratch.alpha, &scratch.theta,
                &m_initial_slice, &mut scratch.m_states, &mut scratch.y, s, d, bs,
                cfg.error_clip_for_level(level),
            );
            // NOTE: copy_final_m_batch is NOT called here — caller does it outside the graph.
            true
        }
        (None, MemoryRuleKind::TitansLMM) => {
            // Eta gate for Titans
            unsafe {
                crate::cuda_ffi::gate_compute_cuda(
                    scratch.k_mem.as_ptr(), scratch.v_mem.as_ptr(), level_params.w_eta.as_ptr(),
                    level_params.b_eta.as_ptr(), scratch.eta.ptr(),
                    tokens_i32, d_i32, 0,
                );
            }
            // Use the persistent scratch.s_initial buffer (stable device pointer, safe for CUDA graph capture).
            // Titans: s_initial is always zero per chunk; re-zero on each step.
            scratch.s_initial.zero();
            let s_initial_slice = scratch.s_initial.slice(0, bs * dd);
            if scratch.has_s_states {
                scratch.m_states.zero();
                scratch.s_states.zero();
                crate::dispatch::titans_forward_dd(
                    &scratch.k_mem, &scratch.v_mem, &scratch.q_mem,
                    &scratch.alpha, &scratch.theta, &scratch.eta,
                    &m_initial_slice, &s_initial_slice,
                    &mut scratch.m_states, &mut scratch.s_states, &mut scratch.y, s, d, bs,
                    cfg.error_clip_for_level(level),
                );
                // NOTE: copy_final_m_batch NOT called here — caller does it outside the graph.
            }
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
    let mut saved_m: Vec<Vec<f32>> = context.memory.iter()
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
                                         s, nh, hd, ws, bs);
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
    params: &GpuMAGParams,
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
    let dd = d * d;

    let fwd = context.forward_scratch.as_ref()?;
    let d_i32      = i32::try_from(d).expect("d_model exceeds i32::MAX");
    let tokens_i32 = i32::try_from(bs * s).expect("bs*s exceeds i32::MAX");
    let v_i32      = i32::try_from(v).expect("vocab_size exceeds i32::MAX");

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
            for b in 0..bs {
                unsafe {
                    crate::cuda_ffi::m_norm_clamp_f32_cuda(
                        (context.memory[level].ptr() as *mut u8).add(b * dd * 4) as *mut f32,
                        d_i32, m_norm_max,
                    );
                }
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
        attn_weights_bf16:   unsafe { GpuBuf::from_raw_non_owning(fwd.attn_weights_bf16.ptr(), bs * nh * s * ws) },
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
#[cfg(feature = "cuda")]
pub(crate) fn gpu_memory_read_only(
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

    // prefill is always batch_size=1 (single prompt)
    crate::dispatch::swa_forward_dd(
        &q_bf16, &k_bf16, &v_bf16,
        &mut attn_out_bf16, &mut attn_weights_bf16,
        s, nh, hd, ws, 1,
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
                s, d, level, 1,
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
                1, d, level, 1,
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
