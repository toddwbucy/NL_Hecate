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

// ── Host-side softmax for alpha_mem ──────────────────────────────────

/// Numerically stable softmax over a small host-side vector (k elements).
/// Used for learnable level aggregation weights (HOPE eq-074).
pub fn host_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0f32; logits.len()];
    let mut sum_exp = 0.0f32;
    for (i, &a) in logits.iter().enumerate() {
        let e = (a - max_val).exp();
        out[i] = e;
        sum_exp += e;
    }
    for w in &mut out {
        *w /= sum_exp;
    }
    out
}

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

    /// Push-up level extension for stacked models: per-block level shift.
    ///
    /// Each block independently shifts level[i] → level[i+1] and gets a fresh L0
    /// with distinct Xavier init (different seed per block). Shared params
    /// (embed, unembed, ln_final) are preserved exactly.
    ///
    /// Mirrors `MAGParams::extend_push_up` (model.rs) but operates per-block.
    /// Spec: specs/infrastructure/22_stacked_extend_k_per_block.md
    pub fn extend_push_up(&self, new_cfg: &MAGConfig, seed: u64) -> StackedMAGParams {
        let n_blocks = self.blocks.len();
        assert!(n_blocks > 0, "extend_push_up: stacked model must have at least 1 block");

        // Validate structural compatibility with new_cfg
        let old_d = self.w_embed.len() / new_cfg.swa.vocab_size;
        assert_eq!(
            old_d, new_cfg.swa.d_model,
            "extend_push_up: d_model mismatch (checkpoint has {old_d}, new_cfg has {})",
            new_cfg.swa.d_model,
        );
        assert_eq!(
            self.w_embed.len(), new_cfg.swa.vocab_size * new_cfg.swa.d_model,
            "extend_push_up: vocab_size mismatch",
        );

        let old_k = self.blocks[0].levels.len();
        assert_eq!(
            new_cfg.k, old_k + 1,
            "extend_push_up requires new_cfg.k ({}) == old_k + 1 ({})",
            new_cfg.k, old_k + 1,
        );
        // Validate all blocks have the same k
        for (b, block) in self.blocks.iter().enumerate() {
            assert_eq!(
                block.levels.len(), old_k,
                "extend_push_up: block {} has {} levels, expected {}",
                b, block.levels.len(), old_k,
            );
        }

        // Per-block level shift with distinct seeds
        let new_blocks: Vec<BlockParams> = self.blocks.iter().enumerate().map(|(b, old_block)| {
            let block_seed = seed.wrapping_add(b as u64 * 10_000);
            let mut new_block = BlockParams::init(new_cfg, block_seed);

            // Preserve SWA projections and LayerNorms for this block
            new_block.w_q = old_block.w_q.clone();
            new_block.w_k = old_block.w_k.clone();
            new_block.w_v = old_block.w_v.clone();
            new_block.w_o = old_block.w_o.clone();
            new_block.ln_attn_gamma = old_block.ln_attn_gamma.clone();
            new_block.ln_attn_beta = old_block.ln_attn_beta.clone();
            new_block.ln_mem_gamma = old_block.ln_mem_gamma.clone();
            new_block.ln_mem_beta = old_block.ln_mem_beta.clone();

            // Shift levels: old level[i] → new level[i+1]
            for i in 0..old_k {
                new_block.levels[i + 1] = old_block.levels[i].clone();
            }

            // Clone old L0 projections into fresh L0 for scale-matched init.
            // Gate biases kept at level-0 defaults from BlockParams::init.
            let donor = &old_block.levels[0];
            new_block.levels[0].w_k_mem = donor.w_k_mem.clone();
            new_block.levels[0].w_v_mem = donor.w_v_mem.clone();
            new_block.levels[0].w_q_mem = donor.w_q_mem.clone();
            new_block.levels[0].w_alpha = donor.w_alpha.clone();
            new_block.levels[0].w_theta = donor.w_theta.clone();
            new_block.levels[0].w_eta = donor.w_eta.clone();
            new_block.levels[0].w_omega = donor.w_omega.clone();

            // Shift alpha logits per block
            for i in 0..old_k {
                new_block.alpha_mem[i + 1] = old_block.alpha_mem[i];
                new_block.alpha_refl[i + 1] = old_block.alpha_refl[i];
            }
            // new alpha[0] = 0.0 from init (uniform initial weight for new L0)

            new_block
        }).collect();

        StackedMAGParams {
            w_embed: self.w_embed.clone(),
            w_unembed: self.w_unembed.clone(),
            ln_final_gamma: self.ln_final_gamma.clone(),
            ln_final_beta: self.ln_final_beta.clone(),
            blocks: new_blocks,
        }
    }

    /// Clone expansion: duplicate existing levels to fill a larger k.
    ///
    /// Unlike push-up (which shifts levels to slower frequencies and adds a fresh L0),
    /// clone copies existing level weights directly into the new slots.
    /// k=1→k=4: level[0] is cloned into levels[0..4].
    /// k=2→k=4: levels[0,1] are cloned into levels[0..4] (round-robin).
    ///
    /// SWA projections, LayerNorms, and embeddings are preserved.
    /// Gate biases for cloned levels use the new config's defaults (from BlockParams::init)
    /// to match the target frequency's expected retention/learning-rate regime.
    pub fn extend_clone(&self, new_cfg: &MAGConfig, seed: u64) -> StackedMAGParams {
        let n_blocks = self.blocks.len();
        assert!(n_blocks > 0, "extend_clone: stacked model must have at least 1 block");

        let old_d = self.w_embed.len() / new_cfg.swa.vocab_size;
        assert_eq!(
            old_d, new_cfg.swa.d_model,
            "extend_clone: d_model mismatch (checkpoint has {old_d}, new_cfg has {})",
            new_cfg.swa.d_model,
        );
        assert_eq!(
            self.w_embed.len(), new_cfg.swa.vocab_size * new_cfg.swa.d_model,
            "extend_clone: vocab_size mismatch",
        );

        let old_k = self.blocks[0].levels.len();
        let new_k = new_cfg.k;
        assert!(
            new_k > old_k,
            "extend_clone requires new_cfg.k ({}) > old_k ({})",
            new_k, old_k,
        );
        for (b, block) in self.blocks.iter().enumerate() {
            assert_eq!(
                block.levels.len(), old_k,
                "extend_clone: block {} has {} levels, expected {}",
                b, block.levels.len(), old_k,
            );
        }

        let new_blocks: Vec<BlockParams> = self.blocks.iter().enumerate().map(|(b, old_block)| {
            let block_seed = seed.wrapping_add(b as u64 * 10_000);
            let mut new_block = BlockParams::init(new_cfg, block_seed);

            // Preserve SWA projections and LayerNorms
            new_block.w_q = old_block.w_q.clone();
            new_block.w_k = old_block.w_k.clone();
            new_block.w_v = old_block.w_v.clone();
            new_block.w_o = old_block.w_o.clone();
            new_block.ln_attn_gamma = old_block.ln_attn_gamma.clone();
            new_block.ln_attn_beta = old_block.ln_attn_beta.clone();
            new_block.ln_mem_gamma = old_block.ln_mem_gamma.clone();
            new_block.ln_mem_beta = old_block.ln_mem_beta.clone();

            // Clone levels round-robin: new level[i] gets ALL fields from old level[i % old_k].
            // Gate biases (b_alpha, b_theta, b_eta) are then restored to new config defaults
            // so each cloned level gets frequency-appropriate gate initialization.
            for i in 0..new_k {
                let init_b_alpha = new_block.levels[i].b_alpha.clone();
                let init_b_theta = new_block.levels[i].b_theta.clone();
                let init_b_eta = new_block.levels[i].b_eta.clone();

                let donor = &old_block.levels[i % old_k];
                new_block.levels[i] = donor.clone();

                // Restore frequency-appropriate gate biases from init
                new_block.levels[i].b_alpha = init_b_alpha;
                new_block.levels[i].b_theta = init_b_theta;
                new_block.levels[i].b_eta = init_b_eta;
            }

            // Clone alpha_mem/alpha_refl round-robin
            for i in 0..new_k {
                new_block.alpha_mem[i] = old_block.alpha_mem[i % old_k];
                new_block.alpha_refl[i] = old_block.alpha_refl[i % old_k];
            }

            new_block
        }).collect();

        StackedMAGParams {
            w_embed: self.w_embed.clone(),
            w_unembed: self.w_unembed.clone(),
            ln_final_gamma: self.ln_final_gamma.clone(),
            ln_final_beta: self.ln_final_beta.clone(),
            blocks: new_blocks,
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
            error_clip: vec![],
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

    fn tiny_config_k1() -> MAGConfig {
        let mut cfg = tiny_config();
        cfg.k = 1;
        cfg.chunk_sizes = vec![1];
        cfg
    }

    fn tiny_config_k2() -> MAGConfig {
        let mut cfg = tiny_config();
        cfg.k = 2;
        cfg.chunk_sizes = vec![1, 8];
        cfg
    }

    #[test]
    fn test_extend_push_up_basic() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let n_blocks = 3;
        let old = StackedMAGParams::init(&cfg_k1, n_blocks, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Structure
        assert_eq!(extended.n_blocks(), n_blocks);
        for block in &extended.blocks {
            assert_eq!(block.levels.len(), 2);
            assert_eq!(block.alpha_mem.len(), 2);
            assert_eq!(block.alpha_refl.len(), 2);
        }
    }

    #[test]
    fn test_extend_push_up_shared_preserved() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 4, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Shared params preserved exactly
        assert_eq!(extended.w_embed, old.w_embed);
        assert_eq!(extended.w_unembed, old.w_unembed);
        assert_eq!(extended.ln_final_gamma, old.ln_final_gamma);
        assert_eq!(extended.ln_final_beta, old.ln_final_beta);
    }

    #[test]
    fn test_extend_push_up_swa_preserved() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 4, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Per-block SWA preserved
        for b in 0..4 {
            assert_eq!(extended.blocks[b].w_q, old.blocks[b].w_q);
            assert_eq!(extended.blocks[b].w_k, old.blocks[b].w_k);
            assert_eq!(extended.blocks[b].w_v, old.blocks[b].w_v);
            assert_eq!(extended.blocks[b].w_o, old.blocks[b].w_o);
            assert_eq!(extended.blocks[b].ln_attn_gamma, old.blocks[b].ln_attn_gamma);
            assert_eq!(extended.blocks[b].ln_mem_gamma, old.blocks[b].ln_mem_gamma);
        }
    }

    #[test]
    fn test_extend_push_up_levels_shifted() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 3, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Old level 0 → new level 1 (shifted up)
        for b in 0..3 {
            assert_eq!(
                extended.blocks[b].levels[1].w_k_mem.master(),
                old.blocks[b].levels[0].w_k_mem.master(),
                "block {b}: old L0.w_k_mem should be at new L1"
            );
            assert_eq!(
                extended.blocks[b].levels[1].w_v_mem.master(),
                old.blocks[b].levels[0].w_v_mem.master(),
            );
        }
    }

    #[test]
    fn test_extend_push_up_fresh_l0_scale_matched() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 3, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Fresh L0 has old L0's projection weights (scale-matched)
        for b in 0..3 {
            assert_eq!(
                extended.blocks[b].levels[0].w_k_mem.master(),
                old.blocks[b].levels[0].w_k_mem.master(),
                "block {b}: fresh L0 should clone old L0 projections"
            );
        }
    }

    #[test]
    fn test_extend_push_up_fresh_l0_gate_biases_default() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 2, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Fresh L0 gate biases should be level-0 defaults, not cloned from old L0
        for b in 0..2 {
            // Level-0 default b_alpha is default_b_alpha(0)
            let expected_b_alpha = vec![default_b_alpha(0)];
            assert_eq!(
                extended.blocks[b].levels[0].b_alpha, expected_b_alpha,
                "block {b}: fresh L0 b_alpha should be level-0 default"
            );
        }
    }

    #[test]
    fn test_extend_push_up_alpha_shifted() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let mut old = StackedMAGParams::init(&cfg_k1, 2, 42);
        // Set non-zero alpha to verify shift
        old.blocks[0].alpha_mem[0] = 1.5;
        old.blocks[1].alpha_mem[0] = -0.7;
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Alpha[0] should be 0.0 (fresh), alpha[1] should be old alpha[0]
        assert_eq!(extended.blocks[0].alpha_mem[0], 0.0);
        assert_eq!(extended.blocks[0].alpha_mem[1], 1.5);
        assert_eq!(extended.blocks[1].alpha_mem[0], 0.0);
        assert_eq!(extended.blocks[1].alpha_mem[1], -0.7);
    }

    #[test]
    fn test_extend_push_up_per_block_l0_donor_diversified() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k2 = tiny_config_k2();
        let old = StackedMAGParams::init(&cfg_k1, 3, 42);
        let extended = old.extend_push_up(&cfg_k2, 99);

        // Fresh L0 projections are donor-cloned from each block's old L0.
        // Since old blocks were initialized with distinct seeds, each block's
        // donor is different — so the extended L0 projections differ per block.
        assert_ne!(
            extended.blocks[0].levels[0].w_k_mem,
            extended.blocks[1].levels[0].w_k_mem,
            "block 0 and 1 fresh L0 w_k_mem should differ (per-block donor diversification)"
        );
        // Gate biases are fixed defaults (not seed-dependent), so they match.
        assert_eq!(
            extended.blocks[0].levels[0].b_theta,
            extended.blocks[1].levels[0].b_theta,
            "gate biases are level-0 defaults, identical across blocks"
        );
    }

    fn tiny_config_k4() -> MAGConfig {
        let mut cfg = tiny_config();
        cfg.k = 4;
        cfg.chunk_sizes = vec![1, 8, 64, 512];
        cfg
    }

    #[test]
    fn test_extend_clone_k1_to_k4_structure() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k4 = tiny_config_k4();
        let n_blocks = 3;
        let old = StackedMAGParams::init(&cfg_k1, n_blocks, 42);
        let cloned = old.extend_clone(&cfg_k4, 99);

        assert_eq!(cloned.n_blocks(), n_blocks);
        for block in &cloned.blocks {
            assert_eq!(block.levels.len(), 4);
            assert_eq!(block.alpha_mem.len(), 4);
            assert_eq!(block.alpha_refl.len(), 4);
        }
    }

    #[test]
    fn test_extend_clone_shared_preserved() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k4 = tiny_config_k4();
        let old = StackedMAGParams::init(&cfg_k1, 4, 42);
        let cloned = old.extend_clone(&cfg_k4, 99);

        assert_eq!(cloned.w_embed, old.w_embed);
        assert_eq!(cloned.w_unembed, old.w_unembed);
        assert_eq!(cloned.ln_final_gamma, old.ln_final_gamma);
        assert_eq!(cloned.ln_final_beta, old.ln_final_beta);
    }

    #[test]
    fn test_extend_clone_round_robin_donor() {
        let cfg_k2 = tiny_config_k2();
        let cfg_k4 = tiny_config_k4();
        let old = StackedMAGParams::init(&cfg_k2, 2, 42);
        let cloned = old.extend_clone(&cfg_k4, 99);

        // k=2→k=4: level[0] from donor[0], level[1] from donor[1],
        //          level[2] from donor[0], level[3] from donor[1]
        for b in 0..2 {
            assert_eq!(
                cloned.blocks[b].levels[0].w_k_mem.master(),
                old.blocks[b].levels[0].w_k_mem.master(),
                "block {b}: cloned L0 should come from donor L0"
            );
            assert_eq!(
                cloned.blocks[b].levels[1].w_k_mem.master(),
                old.blocks[b].levels[1].w_k_mem.master(),
                "block {b}: cloned L1 should come from donor L1"
            );
            assert_eq!(
                cloned.blocks[b].levels[2].w_k_mem.master(),
                old.blocks[b].levels[0].w_k_mem.master(),
                "block {b}: cloned L2 should come from donor L0 (round-robin)"
            );
            assert_eq!(
                cloned.blocks[b].levels[3].w_k_mem.master(),
                old.blocks[b].levels[1].w_k_mem.master(),
                "block {b}: cloned L3 should come from donor L1 (round-robin)"
            );
        }
    }

    #[test]
    fn test_extend_clone_gate_biases_from_init() {
        let cfg_k1 = tiny_config_k1();
        let cfg_k4 = tiny_config_k4();
        let old = StackedMAGParams::init(&cfg_k1, 2, 42);
        let cloned = old.extend_clone(&cfg_k4, 99);

        // Gate biases should come from init (frequency-appropriate), not from donor
        let fresh = StackedMAGParams::init(&cfg_k4, 2, 99);
        for b in 0..2 {
            for lev in 0..4 {
                assert_eq!(
                    cloned.blocks[b].levels[lev].b_alpha,
                    fresh.blocks[b].levels[lev].b_alpha,
                    "block {b} level {lev}: b_alpha should match init defaults"
                );
                assert_eq!(
                    cloned.blocks[b].levels[lev].b_theta,
                    fresh.blocks[b].levels[lev].b_theta,
                    "block {b} level {lev}: b_theta should match init defaults"
                );
                assert_eq!(
                    cloned.blocks[b].levels[lev].b_eta,
                    fresh.blocks[b].levels[lev].b_eta,
                    "block {b} level {lev}: b_eta should match init defaults"
                );
            }
        }
    }

    #[test]
    fn test_extend_clone_alpha_mem_round_robin() {
        let cfg_k2 = tiny_config_k2();
        let cfg_k4 = tiny_config_k4();
        let old = StackedMAGParams::init(&cfg_k2, 2, 42);
        let cloned = old.extend_clone(&cfg_k4, 99);

        for b in 0..2 {
            assert_eq!(cloned.blocks[b].alpha_mem[0], old.blocks[b].alpha_mem[0]);
            assert_eq!(cloned.blocks[b].alpha_mem[1], old.blocks[b].alpha_mem[1]);
            assert_eq!(cloned.blocks[b].alpha_mem[2], old.blocks[b].alpha_mem[0],
                "alpha_mem[2] should round-robin from alpha_mem[0]");
            assert_eq!(cloned.blocks[b].alpha_mem[3], old.blocks[b].alpha_mem[1],
                "alpha_mem[3] should round-robin from alpha_mem[1]");
        }
    }
}
