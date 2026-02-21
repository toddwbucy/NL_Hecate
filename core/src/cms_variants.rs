/// CMS Deployment Variants — configuration schema for multi-block CMS layouts.
///
/// Five deployment patterns from basic (single block) to hybrid (CMS + non-CMS).
/// This module is configuration + validation only — the multi-block execution
/// engine is Stage 4 work.
///
/// Spec: specs/algorithms/frequency_scheduling/02_cms_variants.md

use crate::model::{CompositionKind, MemoryRuleKind, MAGConfig};
use crate::retention::{RetentionKind, default_retention};
use crate::m3::M3Config;

// ── Variant enum ──────────────────────────────────────────────────────

/// Five CMS deployment patterns.
#[derive(Clone, Debug, PartialEq)]
pub enum DeploymentVariant {
    /// Single block, single CMS config (current system).
    Basic,
    /// CMS + M3 optimizer (requires S3-M2). Each block has its own M3 config.
    Nested,
    /// k increases with depth (non-decreasing across blocks).
    Sequential,
    /// Per-block independent frequency schedules.
    Independent,
    /// Mix of CMS and non-CMS blocks.
    Hybrid,
}

impl std::fmt::Display for DeploymentVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeploymentVariant::Basic => write!(f, "Basic"),
            DeploymentVariant::Nested => write!(f, "Nested"),
            DeploymentVariant::Sequential => write!(f, "Sequential"),
            DeploymentVariant::Independent => write!(f, "Independent"),
            DeploymentVariant::Hybrid => write!(f, "Hybrid"),
        }
    }
}

// ── Block config ──────────────────────────────────────────────────────

/// Per-block configuration within a multi-block model.
#[derive(Clone, Debug)]
pub struct BlockConfig {
    pub composition: CompositionKind,
    pub memory_rule: MemoryRuleKind,
    pub retention: RetentionKind,
    /// Whether CMS frequency gating is enabled for this block.
    /// false = standard attention-only block (non-CMS).
    pub cms_enabled: bool,
    /// CMS levels for this block (ignored if !cms_enabled).
    pub k: usize,
    /// Per-level frequencies (ignored if !cms_enabled).
    pub frequencies: Vec<usize>,
    /// M3 optimizer config for this block (Nested variant requires this).
    pub m3: Option<M3Config>,
}

impl BlockConfig {
    /// Helper: default CMS block with given parameters.
    pub fn default_cms(k: usize, rule: MemoryRuleKind, comp: CompositionKind) -> Self {
        let freqs = match k {
            1 => vec![1],
            2 => vec![1, 8],
            3 => vec![1, 8, 64],
            4 => vec![1, 8, 64, 512],
            _ => (0..k).map(|i| 8usize.pow(i as u32).max(1)).collect(),
        };
        BlockConfig {
            composition: comp,
            memory_rule: rule,
            retention: default_retention(rule),
            cms_enabled: true,
            k,
            frequencies: freqs,
            m3: None,
        }
    }

    /// Helper: non-CMS standard attention block.
    pub fn default_standard(comp: CompositionKind) -> Self {
        BlockConfig {
            composition: comp,
            memory_rule: MemoryRuleKind::DeltaRule,
            retention: RetentionKind::L2WeightDecay,
            cms_enabled: false,
            k: 1,
            frequencies: vec![1],
            m3: None,
        }
    }
}

// ── Multi-block config ────────────────────────────────────────────────

/// Multi-block CMS configuration: variant + per-block configs + global params.
#[derive(Clone, Debug)]
pub struct MultiBlockConfig {
    pub variant: DeploymentVariant,
    pub blocks: Vec<BlockConfig>,
    pub d_model: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
}

impl MultiBlockConfig {
    /// Basic variant: single block from existing MAGConfig (backward compatible).
    pub fn basic(cfg: &MAGConfig) -> Result<Self, String> {
        let block = BlockConfig {
            composition: cfg.composition,
            memory_rule: cfg.memory_rule,
            retention: cfg.retention.clone(),
            cms_enabled: cfg.memory_enabled,
            k: cfg.k,
            frequencies: cfg.chunk_sizes.clone(),
            m3: cfg.m3.clone(),
        };
        let mbc = MultiBlockConfig {
            variant: DeploymentVariant::Basic,
            blocks: vec![block],
            d_model: cfg.swa.d_model,
            num_heads: cfg.swa.num_heads,
            seq_len: cfg.swa.seq_len,
            vocab_size: cfg.swa.vocab_size,
        };
        validate(&mbc)?;
        Ok(mbc)
    }

    /// Nested variant: every CMS block must have M3 config.
    pub fn nested(blocks: Vec<BlockConfig>, d_model: usize, num_heads: usize,
                  seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        let mbc = MultiBlockConfig {
            variant: DeploymentVariant::Nested,
            blocks,
            d_model,
            num_heads,
            seq_len,
            vocab_size,
        };
        validate(&mbc)?;
        Ok(mbc)
    }

    /// Sequential variant: k must be non-decreasing across blocks.
    pub fn sequential(blocks: Vec<BlockConfig>, d_model: usize, num_heads: usize,
                      seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        let mbc = MultiBlockConfig {
            variant: DeploymentVariant::Sequential,
            blocks,
            d_model,
            num_heads,
            seq_len,
            vocab_size,
        };
        validate(&mbc)?;
        Ok(mbc)
    }

    /// Independent variant: per-block independent schedules.
    pub fn independent(blocks: Vec<BlockConfig>, d_model: usize, num_heads: usize,
                       seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        let mbc = MultiBlockConfig {
            variant: DeploymentVariant::Independent,
            blocks,
            d_model,
            num_heads,
            seq_len,
            vocab_size,
        };
        validate(&mbc)?;
        Ok(mbc)
    }

    /// Hybrid variant: mix of CMS and non-CMS blocks.
    pub fn hybrid(blocks: Vec<BlockConfig>, d_model: usize, num_heads: usize,
                  seq_len: usize, vocab_size: usize) -> Result<Self, String> {
        let mbc = MultiBlockConfig {
            variant: DeploymentVariant::Hybrid,
            blocks,
            d_model,
            num_heads,
            seq_len,
            vocab_size,
        };
        validate(&mbc)?;
        Ok(mbc)
    }

    /// Total parameter estimate across all blocks.
    ///
    /// Approximate: SWA (shared) + per-block memory params.
    /// Uses d_model to estimate — actual count depends on rule-specific params.
    pub fn total_params_estimate(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        // Shared SWA: embed(v*d) + q/k/v/o(4*d*d) + unembed(d*v)
        let swa = 2 * v * d + 4 * d * d;
        // Per-block memory: k levels, each with 3 projections(d*d) + 3 gates(2*d+1 each)
        let per_level = 3 * d * d + 3 * (2 * d + 1);
        let memory: usize = self.blocks.iter().map(|b| {
            if b.cms_enabled { b.k * per_level } else { 0 }
        }).sum();
        swa + memory
    }
}

// ── Validation ────────────────────────────────────────────────────────

/// Validate a MultiBlockConfig for variant-specific constraints.
pub fn validate(cfg: &MultiBlockConfig) -> Result<(), String> {
    if cfg.blocks.is_empty() {
        return Err("blocks must not be empty".into());
    }

    // Common validation: each CMS block must have k >= 1 and matching freq length
    for (i, block) in cfg.blocks.iter().enumerate() {
        if block.cms_enabled {
            if block.k < 1 {
                return Err(format!("block[{i}]: k must be >= 1, got {}", block.k));
            }
            if block.frequencies.len() != block.k {
                return Err(format!(
                    "block[{i}]: frequencies length {} != k {}",
                    block.frequencies.len(), block.k
                ));
            }
            for (j, &f) in block.frequencies.iter().enumerate() {
                if f == 0 {
                    return Err(format!("block[{i}].frequencies[{j}] must be >= 1"));
                }
            }
        }
    }

    // Variant-specific validation
    match cfg.variant {
        DeploymentVariant::Basic => {
            if cfg.blocks.len() != 1 {
                return Err(format!(
                    "Basic variant requires exactly 1 block, got {}",
                    cfg.blocks.len()
                ));
            }
        }
        DeploymentVariant::Nested => {
            // Every CMS block must have M3 config
            for (i, block) in cfg.blocks.iter().enumerate() {
                if block.cms_enabled && block.m3.is_none() {
                    return Err(format!(
                        "Nested variant: block[{i}] is CMS-enabled but has no M3 config"
                    ));
                }
                if let Some(ref m3) = block.m3 {
                    m3.validate().map_err(|e| format!("block[{i}].m3: {e}"))?;
                }
            }
        }
        DeploymentVariant::Sequential => {
            // k must be non-decreasing across CMS blocks
            let mut prev_k = 0;
            for (i, block) in cfg.blocks.iter().enumerate() {
                if block.cms_enabled {
                    if block.k < prev_k {
                        return Err(format!(
                            "Sequential variant: block[{i}].k={} < previous k={prev_k} (must be non-decreasing)",
                            block.k
                        ));
                    }
                    prev_k = block.k;
                }
            }
        }
        DeploymentVariant::Independent => {
            // No additional constraints beyond common validation
        }
        DeploymentVariant::Hybrid => {
            // Must have at least one CMS and one non-CMS block
            let has_cms = cfg.blocks.iter().any(|b| b.cms_enabled);
            let has_non_cms = cfg.blocks.iter().any(|b| !b.cms_enabled);
            if !has_cms {
                return Err("Hybrid variant requires at least one CMS block".into());
            }
            if !has_non_cms {
                return Err("Hybrid variant requires at least one non-CMS block".into());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_display() {
        assert_eq!(format!("{}", DeploymentVariant::Basic), "Basic");
        assert_eq!(format!("{}", DeploymentVariant::Nested), "Nested");
        assert_eq!(format!("{}", DeploymentVariant::Sequential), "Sequential");
        assert_eq!(format!("{}", DeploymentVariant::Independent), "Independent");
        assert_eq!(format!("{}", DeploymentVariant::Hybrid), "Hybrid");
    }

    #[test]
    fn test_block_config_defaults() {
        let cms = BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG);
        assert!(cms.cms_enabled);
        assert_eq!(cms.k, 2);
        assert_eq!(cms.frequencies, vec![1, 8]);
        assert!(cms.m3.is_none());

        let std = BlockConfig::default_standard(CompositionKind::MAL);
        assert!(!std.cms_enabled);
        assert_eq!(std.k, 1);
        assert_eq!(std.composition, CompositionKind::MAL);
    }
}
