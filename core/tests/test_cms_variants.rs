//! CMS Deployment Variants tests.
//!
//! Categories: construction (~5), validation (~5), helpers (~5).

use nl_hecate_core::cms_variants::{
    CMSVariant, BlockConfig, MultiBlockConfig, validate,
};
use nl_hecate_core::model::{MAGConfig, CompositionKind, MemoryRuleKind};
use nl_hecate_core::m3::M3Config;

// ── Construction tests ───────────────────────────────────────────────

#[test]
fn test_basic_from_mag_config() {
    let mag_cfg = MAGConfig::test_config();
    let mbc = MultiBlockConfig::basic(&mag_cfg).unwrap();
    assert_eq!(mbc.variant, CMSVariant::Basic);
    assert_eq!(mbc.blocks.len(), 1);
    assert!(mbc.blocks[0].cms_enabled);
    assert_eq!(mbc.blocks[0].k, 1);
    assert_eq!(mbc.d_model, 8);
    assert_eq!(mbc.vocab_size, 16);
}

#[test]
fn test_sequential_k_nondecreasing() {
    let blocks = vec![
        BlockConfig::default_cms(1, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_cms(4, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let mbc = MultiBlockConfig::sequential(blocks, 64, 4, 32, 256).unwrap();
    assert_eq!(mbc.variant, CMSVariant::Sequential);
    assert_eq!(mbc.blocks.len(), 3);
}

#[test]
fn test_nested_requires_m3() {
    let mut blocks = vec![
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    blocks[0].m3 = Some(M3Config::default_k2());
    let mbc = MultiBlockConfig::nested(blocks, 64, 4, 32, 256).unwrap();
    assert_eq!(mbc.variant, CMSVariant::Nested);
}

#[test]
fn test_hybrid_requires_mix() {
    let blocks = vec![
        BlockConfig::default_standard(CompositionKind::MAG),
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let mbc = MultiBlockConfig::hybrid(blocks, 64, 4, 32, 256).unwrap();
    assert_eq!(mbc.variant, CMSVariant::Hybrid);
    assert_eq!(mbc.blocks.len(), 2);
}

#[test]
fn test_independent_any_config() {
    let blocks = vec![
        BlockConfig::default_cms(1, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_cms(4, MemoryRuleKind::TitansLMM, CompositionKind::MAL),
        BlockConfig::default_cms(2, MemoryRuleKind::HebbianRule, CompositionKind::MAC),
    ];
    let mbc = MultiBlockConfig::independent(blocks, 64, 4, 32, 256).unwrap();
    assert_eq!(mbc.variant, CMSVariant::Independent);
    assert_eq!(mbc.blocks.len(), 3);
}

// ── Validation rejection tests ───────────────────────────────────────

#[test]
fn test_sequential_rejects_decreasing_k() {
    let blocks = vec![
        BlockConfig::default_cms(4, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let result = MultiBlockConfig::sequential(blocks, 64, 4, 32, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("non-decreasing"));
}

#[test]
fn test_nested_rejects_no_m3() {
    // CMS block without M3 config → rejection
    let blocks = vec![
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let result = MultiBlockConfig::nested(blocks, 64, 4, 32, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no M3 config"));
}

#[test]
fn test_hybrid_rejects_all_cms() {
    let blocks = vec![
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_cms(4, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let result = MultiBlockConfig::hybrid(blocks, 64, 4, 32, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("non-CMS"));
}

#[test]
fn test_hybrid_rejects_all_noncms() {
    let blocks = vec![
        BlockConfig::default_standard(CompositionKind::MAG),
        BlockConfig::default_standard(CompositionKind::MAL),
    ];
    let result = MultiBlockConfig::hybrid(blocks, 64, 4, 32, 256);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("CMS block"));
}

#[test]
fn test_empty_blocks_rejected() {
    let cfg = MultiBlockConfig {
        variant: CMSVariant::Basic,
        blocks: vec![],
        d_model: 64,
        num_heads: 4,
        seq_len: 32,
        vocab_size: 256,
    };
    assert!(validate(&cfg).is_err());
}

// ── Helper tests ─────────────────────────────────────────────────────

#[test]
fn test_basic_roundtrip_from_mag() {
    // Construct from MAGConfig, verify fields preserved
    let mag_cfg = MAGConfig::test_config_k2();
    let mbc = MultiBlockConfig::basic(&mag_cfg).unwrap();
    assert_eq!(mbc.blocks[0].k, 2);
    assert_eq!(mbc.blocks[0].frequencies, vec![1, 8]);
    assert_eq!(mbc.blocks[0].composition, CompositionKind::MAG);
    assert_eq!(mbc.blocks[0].memory_rule, MemoryRuleKind::DeltaRule);
}

#[test]
fn test_total_params_estimate() {
    let blocks = vec![
        BlockConfig::default_cms(1, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
        BlockConfig::default_standard(CompositionKind::MAG),
        BlockConfig::default_cms(2, MemoryRuleKind::DeltaRule, CompositionKind::MAG),
    ];
    let mbc = MultiBlockConfig::independent(blocks, 64, 4, 32, 256).unwrap();
    let estimate = mbc.total_params_estimate();
    // SWA = 2*256*64 + 4*64*64 = 32768 + 16384 = 49152
    // Block 0: 1 level * (3*64*64 + 3*(128+1)) = 12288 + 387 = 12675
    // Block 1: non-CMS = 0
    // Block 2: 2 levels * 12675 = 25350
    // Total ≈ 49152 + 12675 + 25350 = 87177
    assert!(estimate > 50000, "estimate {} should be > 50k", estimate);
    assert!(estimate < 200000, "estimate {} should be < 200k", estimate);
}

#[test]
fn test_variant_display() {
    assert_eq!(CMSVariant::Basic.to_string(), "Basic");
    assert_eq!(CMSVariant::Nested.to_string(), "Nested");
    assert_eq!(CMSVariant::Sequential.to_string(), "Sequential");
    assert_eq!(CMSVariant::Independent.to_string(), "Independent");
    assert_eq!(CMSVariant::Hybrid.to_string(), "Hybrid");
}

#[test]
fn test_k1_degenerates_to_transformer() {
    // k=1 with freq=[1] is a standard transformer block — no multi-scale behavior
    let block = BlockConfig::default_cms(1, MemoryRuleKind::DeltaRule, CompositionKind::MAG);
    assert_eq!(block.k, 1);
    assert_eq!(block.frequencies, vec![1]);
    // Should be valid as Basic
    let mbc = MultiBlockConfig {
        variant: CMSVariant::Basic,
        blocks: vec![block],
        d_model: 64,
        num_heads: 4,
        seq_len: 32,
        vocab_size: 256,
    };
    assert!(validate(&mbc).is_ok());
}

#[test]
fn test_block_config_default_cms_k4() {
    let block = BlockConfig::default_cms(4, MemoryRuleKind::TitansLMM, CompositionKind::MAL);
    assert_eq!(block.k, 4);
    assert_eq!(block.frequencies, vec![1, 8, 64, 512]);
    assert_eq!(block.composition, CompositionKind::MAL);
    assert_eq!(block.memory_rule, MemoryRuleKind::TitansLMM);
    assert!(block.cms_enabled);
}
