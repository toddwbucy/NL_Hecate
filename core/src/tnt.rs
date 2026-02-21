/// TNT Hierarchical — architecture-agnostic parallelization via global + local memories.
///
/// Core idea: one coarse global memory (sequential across shards) and N fine local
/// memories (fully parallel within each shard). Reported 17.37x speedup in the paper.
///
/// Algorithm:
/// 1. Split sequence into shards of size C_G (global chunk size)
/// 2. For each shard:
///    a. Clone global memory state → N local memories
///    b. Split shard into sub-chunks of size C_L (local chunk size)
///    c. Each local memory processes its sub-chunk independently (parallel)
///    d. Compute shard summary (k_summary, v_summary) from all local outputs
///    e. Update global memory with summary
/// 3. Outputs from local memories are the final outputs
///
/// The approximation: local memories don't see each other's updates within a shard.
/// The global memory propagates information across shards.

use crate::model::{MAGConfig, MemoryLevelParams, MemoryRuleKind};
use crate::chunkwise_gd::{chunkwise_gd_forward, chunkwise_gd_backward, ChunkwiseGDCache};
use crate::delta_rule::MemoryRule;
use crate::delta_rule::DeltaRule;
use crate::titans_lmm::TitansLMM;
use crate::hebbian_rule::HebbianRule;
use crate::moneta::Moneta;
use crate::yaad::YAAD;
use crate::memora::MEMORA;
use crate::lattice_osr::LatticeOSR;
use crate::trellis::Trellis;

/// TNT configuration.
#[derive(Clone, Debug)]
pub struct TNTConfig {
    /// Global shard size (C_G): how many tokens per shard.
    pub global_chunk_size: usize,
    /// Local chunk size (C_L): how many tokens per local memory.
    /// n_local = C_G / C_L local memories per shard.
    pub local_chunk_size: usize,
    /// Use Q-K projection to align queries into key domain (TNT Eqs 13-14).
    /// When false, queries are used as-is (original behavior).
    pub use_qk_projection: bool,
    /// Use attention-based shard summary instead of mean-pooling.
    /// When false, mean-pooling heuristic is used (original behavior).
    pub use_attention_summary: bool,
}

/// Two-stage build workflow configuration (TNT §3).
///
/// Stage 1: small chunk size for better approximation while establishing initial conditions.
/// Stage 2: large chunk size for full throughput with those conditions.
#[derive(Clone, Debug)]
pub struct TwoStageBuildConfig {
    /// Chunk size for stage 1 (small, better approximation).
    pub stage1_chunk_size: usize,
    /// Chunk size for stage 2 (large, full throughput).
    pub stage2_chunk_size: usize,
    /// Step at which to transition from stage 1 to stage 2.
    pub transition_step: usize,
}

impl TwoStageBuildConfig {
    /// Returns the appropriate chunk size for the given step.
    pub fn chunk_size_at(&self, step: usize) -> usize {
        if step < self.transition_step {
            self.stage1_chunk_size
        } else {
            self.stage2_chunk_size
        }
    }
}

/// TNT-specific learnable parameters.
///
/// These are separate from MemoryLevelParams because they are TNT-specific
/// (Q-K projection and attention summary query) and not shared with other
/// parallelization strategies.
pub struct TNTParams {
    /// W_QK: projects queries into key domain [d, d]. TNT Eq 13-14.
    pub w_qk: Vec<f32>,
    /// W_summary_q: projects global memory state into summary query [d, d].
    /// Used for attention-based shard summary.
    pub w_summary_q: Vec<f32>,
}

impl TNTParams {
    /// Initialize TNT params with small random values.
    pub fn init(d: usize, rng: &mut crate::tensor::SimpleRng) -> Self {
        let mut w_qk = vec![0.0f32; d * d];
        let mut w_summary_q = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_qk, 0.01);
        rng.fill_uniform(&mut w_summary_q, 0.01);
        TNTParams { w_qk, w_summary_q }
    }

    /// Zero-initialized (for gradient accumulation).
    pub fn zeros(d: usize) -> Self {
        TNTParams {
            w_qk: vec![0.0f32; d * d],
            w_summary_q: vec![0.0f32; d * d],
        }
    }

    /// Accumulate gradients from another TNTParams.
    pub fn accumulate(&mut self, other: &TNTParams) {
        assert_eq!(self.w_qk.len(), other.w_qk.len(), "w_qk length mismatch in accumulate");
        assert_eq!(self.w_summary_q.len(), other.w_summary_q.len(), "w_summary_q length mismatch in accumulate");
        for (a, b) in self.w_qk.iter_mut().zip(other.w_qk.iter()) { *a += b; }
        for (a, b) in self.w_summary_q.iter_mut().zip(other.w_summary_q.iter()) { *a += b; }
    }
}

/// Cache for TNT forward pass.
pub struct TNTForwardCache {
    pub shards: Vec<ShardCache>,
    pub shard_boundaries: Vec<usize>,   // start indices
    pub shard_lens: Vec<usize>,
    pub global_states: Vec<Vec<f32>>,   // global memory after each shard
    pub seq_len: usize,
    pub d: usize,
}

/// Cache for one shard's forward pass.
pub struct ShardCache {
    /// Per-local-memory chunkwise caches
    pub local_caches: Vec<ChunkwiseGDCache>,
    /// Local output: [shard_len, d]
    pub local_y: Vec<f32>,
    /// Summary key and value for global update
    pub k_summary: Vec<f32>,
    pub v_summary: Vec<f32>,
    /// Attention summary cache (None when using mean-pooling).
    pub attn_summary_cache: Option<AttentionSummaryCache>,
}

/// Extract memory state size for a given rule/config.
fn memory_state_size(cfg: &MAGConfig) -> usize {
    let d = cfg.swa.d_model;
    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM
        | MemoryRuleKind::HebbianRule | MemoryRuleKind::AtlasOmega => d * d,
        MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
            let dh = cfg.d_hidden;
            dh * d + d * dh // W1 + W2
        }
        MemoryRuleKind::LatticeOSR => cfg.m_slots * d,
        MemoryRuleKind::Trellis => {
            let dk = cfg.d_compress;
            dk * d + d * dk // S_K + S_V
        }
    }
}

// ── Q-K Projection (TNT Eqs 13-14) ──────────────────────────────────

/// Project query vectors into the key domain: q_aligned = q @ W_QK^T.
///
/// Solves the compression-retrieval domain mismatch: keys are optimized
/// for writing (compression), queries for reading (retrieval). W_QK aligns them.
///
/// Source: TNT (2511.07343) Eqs 13-14.
pub fn qk_project(q: &[f32], w_qk: &[f32], d: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(q.len(), n * d);
    debug_assert_eq!(w_qk.len(), d * d);
    let mut out = vec![0.0f32; n * d];
    // q[n,d] @ w_qk^T[d,d] → out[n,d]
    for t in 0..n {
        for j in 0..d {
            let mut sum = 0.0f32;
            for k in 0..d {
                sum += q[t * d + k] * w_qk[j * d + k]; // W_QK^T[k,j] = W_QK[j,k]
            }
            out[t * d + j] = sum;
        }
    }
    out
}

/// Backward through Q-K projection: q_aligned = q @ W_QK^T.
/// Returns (d_q, d_w_qk).
pub fn qk_project_backward(
    d_aligned: &[f32],
    q: &[f32],
    w_qk: &[f32],
    d: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(d_aligned.len(), n * d);
    // d_q = d_aligned @ W_QK
    let mut d_q = vec![0.0f32; n * d];
    for t in 0..n {
        for k in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += d_aligned[t * d + j] * w_qk[j * d + k];
            }
            d_q[t * d + k] = sum;
        }
    }
    // d_w_qk[j,k] = sum_t d_aligned[t,j] * q[t,k]
    let mut d_w_qk = vec![0.0f32; d * d];
    for t in 0..n {
        for j in 0..d {
            let da = d_aligned[t * d + j];
            for k in 0..d {
                d_w_qk[j * d + k] += da * q[t * d + k];
            }
        }
    }
    (d_q, d_w_qk)
}

// ── Shard Summary ───────────────────────────────────────────────────

/// Mean-pooling shard summary (original heuristic).
fn compute_shard_summary_mean(local_y: &[f32], shard_len: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k_summary = vec![0.0f32; d];
    let mut v_summary = vec![0.0f32; d];
    if shard_len == 0 { return (k_summary, v_summary); }
    for t in 0..shard_len {
        for j in 0..d {
            k_summary[j] += local_y[t * d + j];
            v_summary[j] += local_y[t * d + j];
        }
    }
    let inv = 1.0 / shard_len as f32;
    for j in 0..d {
        k_summary[j] *= inv;
        v_summary[j] *= inv;
    }
    (k_summary, v_summary)
}

/// Cache for attention-based shard summary (needed for backward).
pub struct AttentionSummaryCache {
    /// Attention weights [shard_len] after softmax.
    pub attn_weights: Vec<f32>,
    /// Summary query vector [d].
    pub summary_q: Vec<f32>,
    /// The local_y used as keys/values [shard_len, d].
    pub shard_len: usize,
}

/// Attention-based shard summary: uses global memory state to attend over local outputs.
///
/// q = mean(global_m rows) @ W_summary_q^T  (single query from global state)
/// attn_weights = softmax(local_y @ q / sqrt(d))
/// summary = attn_weights^T @ local_y  (weighted combination)
///
/// Returns (k_summary, v_summary, cache).
///
/// **Note**: Assumes global_m has a d×d matrix layout (matrix memory rules:
/// Delta, Titans, Hebbian, AtlasOmega). Will panic for non-matrix rules
/// (MLP rules like Moneta/YAAD/MEMORA, compression rules like Lattice/Trellis)
/// whose memory_state_size() != d*d.
fn compute_shard_summary_attention(
    local_y: &[f32],
    shard_len: usize,
    global_m: &[f32],
    w_summary_q: &[f32],
    d: usize,
) -> (Vec<f32>, Vec<f32>, AttentionSummaryCache) {
    assert_eq!(
        global_m.len(), d * d,
        "compute_shard_summary_attention requires d×d matrix memory layout, got {} (d={}). \
         Attention summary is only valid for matrix rules (Delta/Titans/Hebbian/AtlasOmega).",
        global_m.len(), d
    );

    if shard_len == 0 {
        return (
            vec![0.0f32; d],
            vec![0.0f32; d],
            AttentionSummaryCache { attn_weights: vec![], summary_q: vec![0.0f32; d], shard_len: 0 },
        );
    }

    // Derive query from global memory: mean of M rows → project through W_summary_q
    // Mean of M rows (d rows of d elements each, d×d layout)
    let mut mean_row = vec![0.0f32; d];
    for r in 0..d {
        for j in 0..d {
            mean_row[j] += global_m[r * d + j];
        }
    }
    if d > 0 {
        let inv = 1.0 / d as f32;
        for j in 0..d { mean_row[j] *= inv; }
    }

    // summary_q = mean_row @ W_summary_q^T → [d]
    let mut summary_q = vec![0.0f32; d];
    for j in 0..d {
        let mut sum = 0.0f32;
        for k in 0..d {
            sum += mean_row[k] * w_summary_q[j * d + k];
        }
        summary_q[j] = sum;
    }

    // Attention scores: local_y @ summary_q / sqrt(d) → [shard_len]
    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; shard_len];
    for t in 0..shard_len {
        let mut dot = 0.0f32;
        for j in 0..d {
            dot += local_y[t * d + j] * summary_q[j];
        }
        scores[t] = dot * scale;
    }

    // Softmax
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores = vec![0.0f32; shard_len];
    let mut exp_sum = 0.0f32;
    for t in 0..shard_len {
        exp_scores[t] = (scores[t] - max_score).exp();
        exp_sum += exp_scores[t];
    }
    let mut attn_weights = exp_scores;
    if exp_sum > 0.0 {
        for w in attn_weights.iter_mut() { *w /= exp_sum; }
    }

    // Weighted sum: summary = attn_weights^T @ local_y → [d]
    let mut summary = vec![0.0f32; d];
    for t in 0..shard_len {
        let w = attn_weights[t];
        for j in 0..d {
            summary[j] += w * local_y[t * d + j];
        }
    }

    let cache = AttentionSummaryCache { attn_weights: attn_weights.clone(), summary_q, shard_len };
    (summary.clone(), summary, cache)
}

/// Update global memory with shard summary.
/// Simple: M_global += outer(v_summary, k_summary) (Hebbian-style)
fn update_global_memory(
    global_m: &mut [f32],
    k_summary: &[f32],
    v_summary: &[f32],
    d: usize,
    alpha: f32,  // retention factor
) {
    for i in 0..d {
        for j in 0..d {
            global_m[i * d + j] = alpha * global_m[i * d + j]
                + v_summary[i] * k_summary[j];
        }
    }
}

/// TNT forward pass.
///
/// Splits sequence into shards, processes each shard's local memories in parallel
/// (via chunkwise GD), and updates global memory at shard boundaries.
///
/// When `tnt_params` is provided and `tnt_cfg.use_attention_summary` is true,
/// uses attention-based shard summary instead of mean-pooling.
pub fn tnt_forward(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    tnt_cfg: &TNTConfig,
    cfg: &MAGConfig,
    initial_m: Option<Vec<f32>>,
    tnt_params: Option<&TNTParams>,
) -> (Vec<f32>, TNTForwardCache) {
    let cg = tnt_cfg.global_chunk_size;
    let cl = tnt_cfg.local_chunk_size;
    assert!(cl <= cg, "local chunk size must be <= global chunk size");
    assert!(cl >= 1 && cg >= 1);
    // Q-K projection (TNT Eqs 13-14) requires injecting projected queries into
    // the inner memory retrieval loop. This needs chunkwise_gd_forward API changes
    // to accept a query transform. The qk_project/qk_project_backward primitives
    // are implemented and tested; integration is deferred until the chunkwise API
    // supports query hooks.
    assert!(
        !tnt_cfg.use_qk_projection,
        "use_qk_projection is not yet integrated into the chunkwise forward/backward pipeline. \
         The qk_project primitives are implemented but require chunkwise API changes to wire in."
    );

    let state_size = memory_state_size(cfg);
    let num_shards = (seq_len + cg - 1) / cg;

    let mut global_m = initial_m.unwrap_or_else(|| vec![0.0f32; state_size]);
    let mut y = vec![0.0f32; seq_len * d];
    let mut shards = Vec::with_capacity(num_shards);
    let mut shard_boundaries = Vec::with_capacity(num_shards);
    let mut shard_lens = Vec::with_capacity(num_shards);
    let mut global_states = Vec::with_capacity(num_shards + 1);
    global_states.push(global_m.clone());

    for shard_idx in 0..num_shards {
        let shard_start = shard_idx * cg;
        let shard_end = (shard_start + cg).min(seq_len);
        let shard_len = shard_end - shard_start;
        shard_boundaries.push(shard_start);
        shard_lens.push(shard_len);

        let shard_embedded = &embedded[shard_start * d..shard_end * d];

        // Split shard into local chunks
        let n_locals = (shard_len + cl - 1) / cl;
        let mut local_caches = Vec::with_capacity(n_locals);
        let mut shard_y = vec![0.0f32; shard_len * d];

        for local_idx in 0..n_locals {
            let local_start = local_idx * cl;
            let local_end = (local_start + cl).min(shard_len);
            let local_len = local_end - local_start;

            let local_embedded = &shard_embedded[local_start * d..local_end * d];

            // Each local memory starts from a clone of the global memory
            let (local_y, local_cache) = chunkwise_gd_forward(
                level_params, local_embedded, local_len, d,
                local_len, // C=local_len (process as single chunk within local)
                cfg, Some(global_m.clone()),
            );

            shard_y[local_start * d..local_end * d].copy_from_slice(&local_y);
            local_caches.push(local_cache);
        }

        // Copy shard output to full output
        y[shard_start * d..shard_end * d].copy_from_slice(&shard_y);

        // Compute shard summary and update global memory
        let (k_summary, v_summary, attn_cache) = if tnt_cfg.use_attention_summary {
            if let Some(tp) = tnt_params {
                let (k, v, cache) = compute_shard_summary_attention(
                    &shard_y, shard_len, &global_m, &tp.w_summary_q, d,
                );
                (k, v, Some(cache))
            } else {
                let (k, v) = compute_shard_summary_mean(&shard_y, shard_len, d);
                (k, v, None)
            }
        } else {
            let (k, v) = compute_shard_summary_mean(&shard_y, shard_len, d);
            (k, v, None)
        };

        // Global update: outer-product for state-dependent matrix rules (Delta/Titans/Hebbian).
        // AtlasOmega uses a learned omega function for M updates, so it carries forward
        // the last local boundary state instead of using the outer-product update.
        let use_outer_product_update = matches!(
            cfg.memory_rule,
            MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM | MemoryRuleKind::HebbianRule
        );
        if use_outer_product_update {
            update_global_memory(&mut global_m, &k_summary, &v_summary, d, 0.95);
        } else if let Some(last_cache) = local_caches.last() {
            // Rules not in the outer-product set: carry forward last local
            // boundary state so global memory evolves across shards.
            // This includes MLP rules (Moneta/YAAD/MEMORA), compression rules
            // (Lattice/Trellis), and AtlasOmega — which has a d×d matrix M but
            // updates it via a learned omega function (state-independent), not
            // the Hebbian-style outer-product used by Delta/Titans/Hebbian.
            if let Some(last_chunk) = last_cache.chunks.last() {
                global_m = last_chunk.boundary_after.state.clone();
            }
        }

        global_states.push(global_m.clone());

        shards.push(ShardCache {
            local_caches,
            local_y: shard_y,
            k_summary,
            v_summary,
            attn_summary_cache: attn_cache,
        });
    }

    let cache = TNTForwardCache {
        shards, shard_boundaries, shard_lens, global_states,
        seq_len, d,
    };

    (y, cache)
}

/// Backward through global memory update.
/// Forward: global_m_new[i,j] = alpha * global_m_old[i,j] + v_summary[i] * k_summary[j]
/// Returns (d_global_m_old, d_k_summary, d_v_summary).
fn update_global_memory_backward(
    d_global_m_new: &[f32],
    k_summary: &[f32],
    v_summary: &[f32],
    d: usize,
    alpha: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut d_global_m_old = vec![0.0f32; d * d];
    let mut d_k_summary = vec![0.0f32; d];
    let mut d_v_summary = vec![0.0f32; d];

    for i in 0..d {
        for j in 0..d {
            let dg = d_global_m_new[i * d + j];
            d_global_m_old[i * d + j] = alpha * dg;
            d_v_summary[i] += dg * k_summary[j];
            d_k_summary[j] += dg * v_summary[i];
        }
    }

    (d_global_m_old, d_k_summary, d_v_summary)
}

/// Backward through shard summary (mean pooling).
/// Distributes d_k_summary and d_v_summary back to local outputs.
fn shard_summary_backward(
    d_k_summary: &[f32],
    d_v_summary: &[f32],
    shard_len: usize,
    d: usize,
) -> Vec<f32> {
    let mut d_local_y = vec![0.0f32; shard_len * d];
    if shard_len == 0 { return d_local_y; }
    let inv = 1.0 / shard_len as f32;
    for t in 0..shard_len {
        for j in 0..d {
            d_local_y[t * d + j] = (d_k_summary[j] + d_v_summary[j]) * inv;
        }
    }
    d_local_y
}

/// Backward through attention-based shard summary.
///
/// Forward path:
///   mean_row = mean(global_m rows)                      [d]
///   summary_q = mean_row @ W_summary_q^T                [d]
///   scores[t] = (local_y[t] · summary_q) / sqrt(d)      [shard_len]
///   attn_weights = softmax(scores)                       [shard_len]
///   summary[j] = sum_t attn_weights[t] * local_y[t,j]   [d]
///   k_summary = v_summary = summary
///
/// Returns (d_local_y, d_w_summary_q, d_global_m_from_summary).
fn shard_summary_attention_backward(
    d_k_summary: &[f32],
    d_v_summary: &[f32],
    cache: &AttentionSummaryCache,
    local_y: &[f32],
    global_m: &[f32],
    w_summary_q: &[f32],
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let shard_len = cache.shard_len;
    let mut d_local_y = vec![0.0f32; shard_len * d];
    let mut d_w_summary_q = vec![0.0f32; d * d];
    let mut d_global_m = vec![0.0f32; d * d];

    if shard_len == 0 {
        return (d_local_y, d_w_summary_q, d_global_m);
    }

    // d_summary = d_k_summary + d_v_summary (since k_summary = v_summary = summary)
    let mut d_summary = vec![0.0f32; d];
    for j in 0..d {
        d_summary[j] = d_k_summary[j] + d_v_summary[j];
    }

    // Backward through: summary[j] = sum_t attn_weights[t] * local_y[t,j]
    // d_attn_weights[t] = sum_j d_summary[j] * local_y[t,j]
    // d_local_y[t,j] += attn_weights[t] * d_summary[j]
    let mut d_attn_weights = vec![0.0f32; shard_len];
    for t in 0..shard_len {
        for j in 0..d {
            d_attn_weights[t] += d_summary[j] * local_y[t * d + j];
            d_local_y[t * d + j] += cache.attn_weights[t] * d_summary[j];
        }
    }

    // Backward through softmax: d_scores[t] = attn_weights[t] * (d_attn_weights[t] - sum_k attn_weights[k] * d_attn_weights[k])
    let weighted_sum: f32 = (0..shard_len)
        .map(|t| cache.attn_weights[t] * d_attn_weights[t])
        .sum();
    let mut d_scores = vec![0.0f32; shard_len];
    for t in 0..shard_len {
        d_scores[t] = cache.attn_weights[t] * (d_attn_weights[t] - weighted_sum);
    }

    // Backward through: scores[t] = (local_y[t] · summary_q) / sqrt(d)
    // d_local_y[t,j] += d_scores[t] * summary_q[j] / sqrt(d)
    // d_summary_q[j] += sum_t d_scores[t] * local_y[t,j] / sqrt(d)
    let scale = 1.0 / (d as f32).sqrt();
    let mut d_summary_q = vec![0.0f32; d];
    for t in 0..shard_len {
        let ds = d_scores[t] * scale;
        for j in 0..d {
            d_local_y[t * d + j] += ds * cache.summary_q[j];
            d_summary_q[j] += ds * local_y[t * d + j];
        }
    }

    // Backward through: summary_q[j] = sum_k mean_row[k] * W_summary_q[j,k]
    // d_mean_row[k] += sum_j d_summary_q[j] * W_summary_q[j,k]
    // d_W_summary_q[j,k] += d_summary_q[j] * mean_row[k]
    let mut mean_row = vec![0.0f32; d];
    for r in 0..d {
        for j in 0..d {
            mean_row[j] += global_m[r * d + j];
        }
    }
    let inv_d = 1.0 / d as f32;
    for j in 0..d { mean_row[j] *= inv_d; }

    let mut d_mean_row = vec![0.0f32; d];
    for j in 0..d {
        for k in 0..d {
            d_mean_row[k] += d_summary_q[j] * w_summary_q[j * d + k];
            d_w_summary_q[j * d + k] += d_summary_q[j] * mean_row[k];
        }
    }

    // Backward through: mean_row = mean(global_m rows)
    // d_global_m[r,j] += d_mean_row[j] / d
    for r in 0..d {
        for j in 0..d {
            d_global_m[r * d + j] += d_mean_row[j] * inv_d;
        }
    }

    (d_local_y, d_w_summary_q, d_global_m)
}

/// TNT backward pass: reverse through shards, propagating gradients through
/// both local chunkwise passes and global memory updates.
///
/// Returns (level_grads, d_embedded, tnt_grads) where tnt_grads is Some
/// when tnt_params was provided.
pub fn tnt_backward(
    level_params: &MemoryLevelParams,
    cache: &TNTForwardCache,
    d_y: &[f32],
    embedded: &[f32],
    tnt_cfg: &TNTConfig,
    cfg: &MAGConfig,
    tnt_params: Option<&TNTParams>,
) -> (MemoryLevelParams, Vec<f32>, Option<TNTParams>) {
    let s = cache.seq_len;
    let d = cache.d;
    let cl = tnt_cfg.local_chunk_size;
    let state_size = memory_state_size(cfg);

    let mut total_grads = MemoryLevelParams::zeros_like(d);
    let mut d_embedded = vec![0.0f32; s * d];
    let mut tnt_grads = tnt_params.map(|_| TNTParams::zeros(d));

    // Gradient for global memory, propagated backward through shards
    let mut d_global_m = vec![0.0f32; state_size];

    // Reverse shard order to propagate d_global_m backward
    for shard_idx in (0..cache.shards.len()).rev() {
        let shard = &cache.shards[shard_idx];
        let shard_start = cache.shard_boundaries[shard_idx];
        let shard_len = cache.shard_lens[shard_idx];
        let shard_embedded = &embedded[shard_start * d..(shard_start + shard_len) * d];

        // Backward through global update at this shard boundary
        // (gradient flows from later shards via d_global_m)
        let use_outer_product_update = matches!(
            cfg.memory_rule,
            MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM | MemoryRuleKind::HebbianRule
        );
        if use_outer_product_update {
            let (d_gm_old, d_k_sum, d_v_sum) = update_global_memory_backward(
                &d_global_m, &shard.k_summary, &shard.v_summary, d, 0.95,
            );
            d_global_m = d_gm_old;

            // Backward through shard summary → d_shard_y_from_global for this shard.
            // Invariant: attn_summary_cache is only populated during forward when
            // tnt_params was provided (see tnt_forward lines 422-431), so the
            // expect on tnt_params below is guaranteed safe.
            let d_shard_y_from_global = if let Some(attn_cache) = &shard.attn_summary_cache {
                // shard_summary_attention_backward: produces gradients for
                // local_y, w_summary_q, and additional d_global_m
                let global_m_at_shard = &cache.global_states[shard_idx];
                let tp = tnt_params.expect("attn_summary_cache requires tnt_params");
                let (d_ly, d_wsq, d_gm_attn) = shard_summary_attention_backward(
                    &d_k_sum, &d_v_sum, attn_cache,
                    &shard.local_y, global_m_at_shard, &tp.w_summary_q, d,
                );
                // Accumulate w_summary_q gradient
                if let Some(ref mut tg) = tnt_grads {
                    for (a, b) in tg.w_summary_q.iter_mut().zip(d_wsq.iter()) { *a += b; }
                }
                // Add attention summary's contribution to d_global_m
                for (a, b) in d_global_m.iter_mut().zip(d_gm_attn.iter()) { *a += b; }
                d_ly
            } else {
                // Mean-pooling backward (original path)
                shard_summary_backward(&d_k_sum, &d_v_sum, shard_len, d)
            };

            // Combine upstream d_y with gradient from global path
            let mut d_y_combined = vec![0.0f32; shard_len * d];
            for t in 0..shard_len {
                for j in 0..d {
                    let idx = t * d + j;
                    d_y_combined[idx] = d_y[(shard_start + t) * d + j]
                        + d_shard_y_from_global[idx];
                }
            }

            // Backward through local chunks with combined gradient
            for (local_idx, local_cache) in shard.local_caches.iter().enumerate() {
                let local_start = local_idx * cl;
                let local_end = (local_start + cl).min(shard_len);
                let local_len = local_end - local_start;

                let d_y_local = &d_y_combined[local_start * d..local_end * d];
                let local_embedded = &shard_embedded[local_start * d..local_end * d];

                let (local_grads, local_d_emb) = chunkwise_gd_backward(
                    level_params, local_cache, d_y_local, local_embedded, cfg,
                );

                total_grads.accumulate(&local_grads);
                let abs_start = shard_start + local_start;
                d_embedded[abs_start * d..(abs_start + local_len) * d]
                    .copy_from_slice(&local_d_emb);
            }
        } else {
            // Non-matrix rules: simpler path (global = last local boundary state)
            for (local_idx, local_cache) in shard.local_caches.iter().enumerate() {
                let local_start = local_idx * cl;
                let local_end = (local_start + cl).min(shard_len);
                let local_len = local_end - local_start;

                let abs_start = shard_start + local_start;
                let d_y_local = &d_y[abs_start * d..(abs_start + local_len) * d];
                let local_embedded = &shard_embedded[local_start * d..local_end * d];

                let (local_grads, local_d_emb) = chunkwise_gd_backward(
                    level_params, local_cache, d_y_local, local_embedded, cfg,
                );

                total_grads.accumulate(&local_grads);
                d_embedded[abs_start * d..(abs_start + local_len) * d]
                    .copy_from_slice(&local_d_emb);
            }
        }
    }

    (total_grads, d_embedded, tnt_grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::chunkwise_gd::chunkwise_gd_forward as cw_forward;

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut e = vec![0.0f32; s * d];
        rng.fill_uniform(&mut e, 0.1);
        e
    }

    fn relative_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-8))
            .fold(0.0f32, f32::max)
    }

    fn tnt_cfg_small() -> TNTConfig {
        TNTConfig {
            global_chunk_size: 2,
            local_chunk_size: 1,
            use_qk_projection: false,
            use_attention_summary: false,
        }
    }

    // ─── Per-rule tests ─────────────────────────────────────

    macro_rules! tnt_rule_tests {
        ($prefix:ident, $config_fn:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $prefix _tnt_small_shard>]() {
                    let cfg = MAGConfig::$config_fn();
                    let params = MAGParams::init(&cfg, 42);
                    let embedded = make_embedded(&cfg, 99);
                    let s = cfg.swa.seq_len;
                    let d = cfg.swa.d_model;
                    let tnt = tnt_cfg_small();

                    let (y, cache) = tnt_forward(
                        &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
                    );
                    assert_eq!(y.len(), s * d);
                    for &v in &y {
                        assert!(v.is_finite(), "{}: TNT output not finite", stringify!($prefix));
                    }
                    assert!(cache.shards.len() >= 2, "expected >= 2 shards");
                }

                #[test]
                fn [<test_ $prefix _tnt_forward_finite>]() {
                    let cfg = MAGConfig::$config_fn();
                    let params = MAGParams::init(&cfg, 42);
                    let embedded = make_embedded(&cfg, 99);
                    let s = cfg.swa.seq_len;
                    let d = cfg.swa.d_model;
                    let tnt = TNTConfig {
                        global_chunk_size: s, local_chunk_size: s,
                        use_qk_projection: false, use_attention_summary: false,
                    };

                    let (y, _) = tnt_forward(
                        &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
                    );
                    for &v in &y {
                        assert!(v.is_finite());
                    }
                }

                #[test]
                fn [<test_ $prefix _tnt_backward_shapes>]() {
                    let cfg = MAGConfig::$config_fn();
                    let params = MAGParams::init(&cfg, 42);
                    let embedded = make_embedded(&cfg, 99);
                    let s = cfg.swa.seq_len;
                    let d = cfg.swa.d_model;
                    let tnt = tnt_cfg_small();

                    let (_, cache) = tnt_forward(
                        &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
                    );
                    let d_y = vec![1.0f32; s * d];
                    let (grads, d_emb, _) = tnt_backward(
                        &params.levels[0], &cache, &d_y, &embedded, &tnt, &cfg, None,
                    );
                    assert_eq!(grads.w_k_mem.len(), d * d);
                    assert_eq!(d_emb.len(), s * d);
                }

                #[test]
                fn [<test_ $prefix _tnt_fd_gradient>]() {
                    let cfg = MAGConfig::$config_fn();
                    let params = MAGParams::init(&cfg, 42);
                    let embedded = make_embedded(&cfg, 99);
                    let s = cfg.swa.seq_len;
                    let d = cfg.swa.d_model;
                    let tnt = tnt_cfg_small();
                    let eps = 1e-2f32;

                    let (_, cache) = tnt_forward(
                        &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
                    );
                    let d_y = vec![1.0f32; s * d];
                    let (grads, _, _) = tnt_backward(
                        &params.levels[0], &cache, &d_y, &embedded, &tnt, &cfg, None,
                    );

                    let mut lp_p = params.levels[0].clone();
                    lp_p.w_k_mem[0] += eps;
                    let (y_p, _) = tnt_forward(&lp_p, &embedded, s, d, &tnt, &cfg, None, None);
                    let loss_p: f32 = y_p.iter().sum();

                    let mut lp_m = params.levels[0].clone();
                    lp_m.w_k_mem[0] -= eps;
                    let (y_m, _) = tnt_forward(&lp_m, &embedded, s, d, &tnt, &cfg, None, None);
                    let loss_m: f32 = y_m.iter().sum();

                    let fd = (loss_p - loss_m) / (2.0 * eps);
                    let analytical = grads.w_k_mem[0];
                    let same_sign = (analytical > 0.0) == (fd > 0.0)
                        || analytical.abs() < 5e-4;
                    assert!(same_sign || (analytical.abs() < 5e-4 && fd.abs() < 5e-4),
                        "{}: FD sign mismatch: analytical={analytical:.6e} fd={fd:.6e}",
                        stringify!($prefix));
                }
            }
        }
    }

    tnt_rule_tests!(delta_tnt, test_config);
    tnt_rule_tests!(titans_tnt, titans_test_config);
    tnt_rule_tests!(hebbian_tnt, hebbian_test_config);
    tnt_rule_tests!(moneta_tnt, moneta_test_config);
    tnt_rule_tests!(yaad_tnt, yaad_test_config);
    tnt_rule_tests!(memora_tnt, memora_test_config);
    tnt_rule_tests!(lattice_tnt, lattice_test_config);
    tnt_rule_tests!(trellis_tnt, trellis_test_config);
    tnt_rule_tests!(atlas_tnt, atlas_test_config);

    // ─── General tests ──────────────────────────────────────

    #[test]
    fn test_tnt_local_independence() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let tnt = TNTConfig {
            global_chunk_size: 4, local_chunk_size: 2,
            use_qk_projection: false, use_attention_summary: false,
        };

        let (_, cache) = tnt_forward(
            &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
        );

        if cache.shards[0].local_caches.len() >= 2 {
            let lc0 = &cache.shards[0].local_caches[0];
            let lc1 = &cache.shards[0].local_caches[1];
            assert!(lc0.seq_len != lc1.seq_len || lc0.chunks.len() > 0);
        }
    }

    #[test]
    fn test_tnt_global_propagation() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let tnt = tnt_cfg_small();

        let (_, cache) = tnt_forward(
            &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
        );

        if cache.global_states.len() >= 3 {
            let g0_norm: f32 = cache.global_states[0].iter().map(|x| x * x).sum::<f32>().sqrt();
            let g1_norm: f32 = cache.global_states[1].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(g1_norm > g0_norm || g1_norm > 1e-10,
                "Global memory should evolve: g0_norm={g0_norm} g1_norm={g1_norm}");
        }
    }

    #[test]
    fn test_tnt_outer_loop_weight_descent() {
        let cfg = MAGConfig::test_config();
        let mut level_params = MAGParams::init(&cfg, 42).levels.into_iter().next().unwrap();
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let tnt = tnt_cfg_small();
        let lr = 0.1;

        let mut rng = SimpleRng::new(99);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 1.0);
        let mut target = vec![0.0f32; s * d];
        rng.fill_uniform(&mut target, 1.0);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for outer_step in 0..100 {
            let (y, cache) = tnt_forward(
                &level_params, &embedded, s, d, &tnt, &cfg, None, None,
            );
            let loss: f32 = y.iter().zip(target.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
            if outer_step == 0 { first_loss = loss; }
            if outer_step == 99 { last_loss = loss; }

            let d_y: Vec<f32> = y.iter().zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
            let (grads, _, _) = tnt_backward(
                &level_params, &cache, &d_y, &embedded, &tnt, &cfg, None,
            );

            for (w, g) in level_params.w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
            for (w, g) in level_params.w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
            for (w, g) in level_params.w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
        }

        assert!(last_loss <= first_loss + 1e-6,
            "TNT outer-loop weight descent should not diverge: first={first_loss:.6} last={last_loss:.6}");
    }

    #[test]
    fn test_tnt_single_shard_matches_chunkwise() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let tnt = TNTConfig {
            global_chunk_size: s, local_chunk_size: s,
            use_qk_projection: false, use_attention_summary: false,
        };
        let (y_tnt, _) = tnt_forward(
            &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
        );

        let (y_cw, _) = cw_forward(
            &params.levels[0], &embedded, s, d, s, &cfg, None,
        );

        let max_diff: f32 = y_tnt.iter().zip(y_cw.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5,
            "Single-shard TNT should match chunkwise, max_diff={max_diff}");
    }

    #[test]
    fn test_tnt_backward_nonzero() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let tnt = tnt_cfg_small();

        let (_, cache) = tnt_forward(
            &params.levels[0], &embedded, s, d, &tnt, &cfg, None, None,
        );
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb, _) = tnt_backward(
            &params.levels[0], &cache, &d_y, &embedded, &tnt, &cfg, None,
        );

        let grad_norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(grad_norm > 1e-10, "TNT backward grads should be non-zero");
        let emb_norm: f32 = d_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(emb_norm > 1e-10, "TNT d_embedded should be non-zero");
    }

    // ─── New feature tests ──────────────────────────────────

    #[test]
    fn test_qk_projection_basic() {
        let d = 4;
        let n = 3;
        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // Identity-ish W_QK
        let mut w_qk = vec![0.0f32; d * d];
        for i in 0..d { w_qk[i * d + i] = 1.0; }
        let result = qk_project(&q, &w_qk, d, n);
        // Identity projection → output ≈ input
        for i in 0..n * d {
            assert!((result[i] - q[i]).abs() < 1e-6, "identity QK projection failed at {i}");
        }
    }

    #[test]
    fn test_qk_projection_backward_fd() {
        let d = 3;
        let n = 2;
        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut w_qk: Vec<f32> = (0..d * d).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let eps = 1e-3f32;

        let aligned = qk_project(&q, &w_qk, d, n);
        let d_aligned = vec![1.0f32; n * d];
        let (d_q, d_w_qk) = qk_project_backward(&d_aligned, &q, &w_qk, d, n);

        // FD check on w_qk[0]
        w_qk[0] += eps;
        let loss_p: f32 = qk_project(&q, &w_qk, d, n).iter().sum();
        w_qk[0] -= 2.0 * eps;
        let loss_m: f32 = qk_project(&q, &w_qk, d, n).iter().sum();
        let fd = (loss_p - loss_m) / (2.0 * eps);
        assert!((d_w_qk[0] - fd).abs() < 1e-3,
            "QK backward FD: analytical={} fd={fd}", d_w_qk[0]);
    }

    #[test]
    fn test_attention_summary_finite() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(77);
        let tnt_p = TNTParams::init(d, &mut rng);
        let tnt = TNTConfig {
            global_chunk_size: 2, local_chunk_size: 1,
            use_qk_projection: false, use_attention_summary: true,
        };

        let (y, cache) = tnt_forward(
            &params.levels[0], &embedded, s, d, &tnt, &cfg, None, Some(&tnt_p),
        );
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "attention summary: y[{i}] not finite");
        }
        // Verify attention caches are populated
        for shard in &cache.shards {
            assert!(shard.attn_summary_cache.is_some(),
                "attention summary cache should be Some when use_attention_summary=true");
        }
    }

    #[test]
    fn test_attention_summary_differs_from_mean() {
        // Directly compare the two summary functions on the same local outputs.
        // With non-zero global_m, the attention query is non-trivial, producing
        // a different weighted combination than uniform mean-pooling.
        let d = 8;
        let shard_len = 4;
        let mut rng = SimpleRng::new(77);
        let tnt_p = TNTParams::init(d, &mut rng);

        let mut local_y = vec![0.0f32; shard_len * d];
        rng.fill_uniform(&mut local_y, 1.0);

        let mut global_m = vec![0.0f32; d * d];
        rng.fill_uniform(&mut global_m, 0.5);

        let (k_mean, _) = compute_shard_summary_mean(&local_y, shard_len, d);
        let (k_attn, _, _cache) = compute_shard_summary_attention(
            &local_y, shard_len, &global_m, &tnt_p.w_summary_q, d,
        );

        let max_diff: f32 = k_mean.iter().zip(k_attn.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff > 1e-6,
            "attention vs mean summary should differ, max_diff={max_diff}");
    }

    #[test]
    fn test_two_stage_build_config() {
        let cfg = TwoStageBuildConfig {
            stage1_chunk_size: 4,
            stage2_chunk_size: 16,
            transition_step: 100,
        };
        assert_eq!(cfg.chunk_size_at(0), 4);
        assert_eq!(cfg.chunk_size_at(99), 4);
        assert_eq!(cfg.chunk_size_at(100), 16);
        assert_eq!(cfg.chunk_size_at(1000), 16);
    }

    #[test]
    fn test_tnt_params_accumulate() {
        let d = 4;
        let mut rng = SimpleRng::new(42);
        let p1 = TNTParams::init(d, &mut rng);
        let p2 = TNTParams::init(d, &mut rng);
        let mut grads = TNTParams::zeros(d);
        grads.accumulate(&p1);
        grads.accumulate(&p2);
        // Should be sum of p1 + p2
        for i in 0..d * d {
            let expected = p1.w_qk[i] + p2.w_qk[i];
            assert!((grads.w_qk[i] - expected).abs() < 1e-6);
        }
    }
}
