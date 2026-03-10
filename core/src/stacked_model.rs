/// Multi-block stacked HOPE architecture — N blocks of SWA + CMS(k levels).
///
/// Each block has its own SWA projections, LayerNorms, and CMS levels.
/// Embeddings, lm_head, and final LayerNorm are shared across all blocks.
/// Residual stream connects blocks: x_out_block_i → x_in_block_{i+1}.
///
/// Spec: specs/infrastructure/14_multi_block_stacking.md

use serde::{Serialize, Deserialize};
use crate::model::{
    MAGConfig, MAGParams, SWAConfig, MemoryLevelParams,
    default_b_alpha, default_b_theta, default_b_eta,
};
use crate::tensor::SimpleRng;

// ── Per-block parameters ─────────────────────────────────────────────

/// Parameters for one block in the stacked architecture.
/// Contains SWA projections (no embed/unembed) + LN + CMS levels.
#[derive(Clone, Serialize, Deserialize)]
pub struct BlockParams {
    // SWA attention projections for this block
    pub w_q: Vec<f32>,              // [d, d]
    pub w_k: Vec<f32>,              // [d, d]
    pub w_v: Vec<f32>,              // [d, d]
    pub w_o: Vec<f32>,              // [d, d]
    // Pre-norm LayerNorm for attention branch
    pub ln_attn_gamma: Vec<f32>,    // [d]
    pub ln_attn_beta: Vec<f32>,     // [d]
    // Pre-norm LayerNorm for memory branch
    pub ln_mem_gamma: Vec<f32>,     // [d]
    pub ln_mem_beta: Vec<f32>,      // [d]
    // CMS memory levels (length k)
    pub levels: Vec<MemoryLevelParams>,
    // CMS aggregation logits
    pub alpha_mem: Vec<f32>,        // [k]
    pub alpha_refl: Vec<f32>,       // [k]
}

impl BlockParams {
    /// Initialize one block with Xavier-scaled projections and level-specific gates.
    ///
    /// NOTE: This covers the Titans MAG path used by shakedown configs. Features
    /// not yet wired: AtlasOmega atlas_init, learned frequency gates (omega/freq
    /// params in MemoryLevelParams), Conv1D buffers, MAC persistent tokens.
    /// These will be added when their respective compositions are stacked.
    pub fn init(cfg: &MAGConfig, seed: u64) -> Self {
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);

        let proj_scale = (2.0 / (d + d) as f32).sqrt();
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        let mut w_o = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_q, proj_scale);
        rng.fill_uniform(&mut w_k, proj_scale);
        rng.fill_uniform(&mut w_v, proj_scale);
        rng.fill_uniform(&mut w_o, proj_scale);

        let ln_attn_gamma = vec![1.0f32; d];
        let ln_attn_beta = vec![0.0f32; d];
        let ln_mem_gamma = vec![1.0f32; d];
        let ln_mem_beta = vec![0.0f32; d];

        let mut levels = Vec::with_capacity(cfg.k);
        for level in 0..cfg.k {
            let mut level_rng = SimpleRng::new(seed.wrapping_add(1000 + level as u64 * 500));
            let level_params = MemoryLevelParams::init(
                d, &mut level_rng,
                default_b_alpha(level),
                default_b_theta(level),
                default_b_eta(level),
            );
            levels.push(level_params);
        }

        let alpha_mem = vec![0.0f32; cfg.k];
        let alpha_refl = vec![0.0f32; cfg.k];

        BlockParams {
            w_q, w_k, w_v, w_o,
            ln_attn_gamma, ln_attn_beta,
            ln_mem_gamma, ln_mem_beta,
            levels, alpha_mem, alpha_refl,
        }
    }

    /// Zero-initialized block (for gradient accumulation).
    pub fn zeros(d: usize, k: usize) -> Self {
        let levels = (0..k).map(|_| MemoryLevelParams::zeros_like(d)).collect();
        BlockParams {
            w_q: vec![0.0; d * d],
            w_k: vec![0.0; d * d],
            w_v: vec![0.0; d * d],
            w_o: vec![0.0; d * d],
            ln_attn_gamma: vec![0.0; d],
            ln_attn_beta: vec![0.0; d],
            ln_mem_gamma: vec![0.0; d],
            ln_mem_beta: vec![0.0; d],
            levels,
            alpha_mem: vec![0.0; k],
            alpha_refl: vec![0.0; k],
        }
    }

    /// Total parameter count for this block.
    pub fn num_params(&self) -> usize {
        let d2 = self.w_q.len(); // d*d
        let d = self.ln_attn_gamma.len();
        let swa = 4 * d2;
        let ln = 4 * d;
        let levels: usize = self.levels.iter().map(|l| l.num_params()).collect::<Vec<_>>().iter().sum();
        let agg = self.alpha_mem.len() + self.alpha_refl.len();
        swa + ln + levels + agg
    }
}

// ── Stacked model parameters ────────────────────────────────────────

/// Full stacked model: shared embed/unembed + N blocks + final LN.
#[derive(Clone, Serialize, Deserialize)]
pub struct StackedMAGParams {
    pub w_embed: Vec<f32>,          // [vocab, d]
    pub w_unembed: Vec<f32>,        // [d, vocab]
    // Final LayerNorm (after last block, before lm_head)
    pub ln_final_gamma: Vec<f32>,   // [d]
    pub ln_final_beta: Vec<f32>,    // [d]
    // N blocks
    pub blocks: Vec<BlockParams>,
}

impl StackedMAGParams {
    /// Initialize stacked model with n_blocks independent blocks.
    /// Each block gets a distinct seed offset to avoid weight correlation.
    pub fn init(cfg: &MAGConfig, n_blocks: usize, seed: u64) -> Self {
        let d = cfg.swa.d_model;
        let v = cfg.swa.vocab_size;
        let mut rng = SimpleRng::new(seed);

        // Shared embedding
        let embed_scale = (1.0 / d as f32).sqrt();
        let mut w_embed = vec![0.0f32; v * d];
        rng.fill_uniform(&mut w_embed, embed_scale);

        // Shared unembedding
        let unembed_scale = (1.0 / d as f32).sqrt();
        let mut w_unembed = vec![0.0f32; d * v];
        rng.fill_uniform(&mut w_unembed, unembed_scale);

        // Final LayerNorm
        let ln_final_gamma = vec![1.0f32; d];
        let ln_final_beta = vec![0.0f32; d];

        // Per-block init with distinct seeds
        let blocks = (0..n_blocks)
            .map(|b| BlockParams::init(cfg, seed.wrapping_add(10_000 + b as u64 * 10_000)))
            .collect();

        StackedMAGParams {
            w_embed, w_unembed,
            ln_final_gamma, ln_final_beta,
            blocks,
        }
    }

    /// Zero-initialized stacked params (for gradient accumulation).
    pub fn zeros(d: usize, vocab: usize, k: usize, n_blocks: usize) -> Self {
        let blocks = (0..n_blocks).map(|_| BlockParams::zeros(d, k)).collect();
        StackedMAGParams {
            w_embed: vec![0.0; vocab * d],
            w_unembed: vec![0.0; d * vocab],
            ln_final_gamma: vec![0.0; d],
            ln_final_beta: vec![0.0; d],
            blocks,
        }
    }

    /// Total parameter count across all blocks + shared params.
    pub fn num_params(&self) -> usize {
        let shared = self.w_embed.len() + self.w_unembed.len()
            + self.ln_final_gamma.len() + self.ln_final_beta.len();
        let blocks: usize = self.blocks.iter().map(|b| b.num_params()).sum();
        shared + blocks
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Convert from existing single-block MAGParams (backward compat).
    /// Wraps the single block's SWA + levels into block 0.
    ///
    /// LIMITATION: The final LayerNorm (ln_final_gamma/beta) is initialized to
    /// identity (gamma=1, beta=0) because single-block MAGParams has no equivalent.
    /// This means the wrapped model is NOT behavior-preserving on the first forward
    /// pass — the extra LN will slightly change logits. This is acceptable for
    /// checkpoint conversion where training continues, but not for inference parity.
    /// Convert a single-block MAGParams into a 1-block StackedMAGParams.
    ///
    /// WARNING: This adds a fresh final LayerNorm (gamma=1, beta=0) that has no
    /// counterpart in the single-block forward path. A converted checkpoint will NOT
    /// reproduce the source model's logits exactly. Use only for structural tests,
    /// not for inference continuity.
    pub fn from_single_block(params: &MAGParams) -> Self {
        let d = params.swa.ln_attn_gamma.len();
        let block = BlockParams {
            w_q: params.swa.w_q.clone(),
            w_k: params.swa.w_k.clone(),
            w_v: params.swa.w_v.clone(),
            w_o: params.swa.w_o.clone(),
            ln_attn_gamma: params.swa.ln_attn_gamma.clone(),
            ln_attn_beta: params.swa.ln_attn_beta.clone(),
            ln_mem_gamma: params.swa.ln_mem_gamma.clone(),
            ln_mem_beta: params.swa.ln_mem_beta.clone(),
            levels: params.levels.clone(),
            alpha_mem: params.alpha_mem.clone(),
            alpha_refl: params.alpha_refl.clone(),
        };

        StackedMAGParams {
            w_embed: params.swa.w_embed.clone(),
            w_unembed: params.swa.w_unembed.clone(),
            ln_final_gamma: vec![1.0; d],
            ln_final_beta: vec![0.0; d],
            blocks: vec![block],
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> MAGConfig {
        use crate::model::{CompositionKind, MemoryRuleKind, AttentionalBias,
                           HopeVariant, LatticeVariant, MomentumKind,
                           ProjectionKind, FeatureMapKind};
        use crate::dynamic_freq::FrequencySchedule;
        use crate::retention::RetentionKind;
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 16,
                window_size: 16,
                vocab_size: 32,
            },
            memory_enabled: true,
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::TitansLMM,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0,
            lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0,
            d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
            feature_map: FeatureMapKind::Identity,
            residual: true,
        }
    }

    #[test]
    fn test_stacked_init_param_count() {
        let cfg = tiny_config();
        let d = 8;
        let v = 32;
        let k = 2;
        let n = 4;
        let params = StackedMAGParams::init(&cfg, n, 42);

        assert_eq!(params.n_blocks(), n);
        assert_eq!(params.w_embed.len(), v * d);
        assert_eq!(params.w_unembed.len(), d * v);
        assert_eq!(params.ln_final_gamma.len(), d);
        assert_eq!(params.blocks.len(), n);

        // Each block: 4*d*d SWA + 4*d LN + k*(level params) + 2*k aggregation
        for block in &params.blocks {
            assert_eq!(block.w_q.len(), d * d);
            assert_eq!(block.levels.len(), k);
            assert_eq!(block.alpha_mem.len(), k);
        }

        // Param count should be positive and > shared-only
        let total = params.num_params();
        let shared = 2 * v * d + 2 * d;
        assert!(total > shared, "total={total} should exceed shared={shared}");
    }

    #[test]
    fn test_distinct_block_seeds() {
        let cfg = tiny_config();
        let params = StackedMAGParams::init(&cfg, 3, 42);

        // Blocks should have different weights (distinct seeds)
        assert_ne!(params.blocks[0].w_q, params.blocks[1].w_q);
        assert_ne!(params.blocks[1].w_q, params.blocks[2].w_q);
    }

    #[test]
    fn test_zeros() {
        let z = StackedMAGParams::zeros(8, 32, 2, 3);
        assert!(z.w_embed.iter().all(|&x| x == 0.0));
        assert!(z.blocks[0].w_q.iter().all(|&x| x == 0.0));
        assert_eq!(z.blocks.len(), 3);
    }

    #[test]
    fn test_from_single_block() {
        let cfg = tiny_config();
        let single = MAGParams::init(&cfg, 42);
        let stacked = StackedMAGParams::from_single_block(&single);

        assert_eq!(stacked.n_blocks(), 1);
        assert_eq!(stacked.w_embed, single.swa.w_embed);
        assert_eq!(stacked.w_unembed, single.swa.w_unembed);
        assert_eq!(stacked.blocks[0].w_q, single.swa.w_q);
        assert_eq!(stacked.blocks[0].levels.len(), single.levels.len());
    }
}
