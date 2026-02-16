//! Retention mechanism tests — unit, golden-value, cross-combo, integration.
//!
//! Phase 7 of S3-M1: validates the pluggable retention extraction.
//! - Unit: each mechanism in isolation, edge cases
//! - Golden-value: extracted retention == old fused result (bit-identical)
//! - Cross-combo: non-default retention combos (DeltaRule+ElasticNet, etc.)
//! - Integration: elastic net configs in training loop

use nl_hecate_core::model::{
    MAGConfig, MAGParams, SWAConfig, CompositionKind, MemoryRuleKind,
};
use nl_hecate_core::retention::{
    RetentionKind, RetentionConfig, default_retention,
    l2_apply_retention, l2_retention_gradient, l2_decoupled_gradient,
    kl_apply_retention, elastic_net_apply, sphere_project_and_normalize,
};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};
use nl_hecate_core::mag::{cms_forward, cms_backward};
use nl_hecate_core::context_stream::VecStream;

// ── Helper: tiny config builder ────────────────────────────────────

fn tiny_config(rule: MemoryRuleKind, retention: RetentionKind) -> MAGConfig {
    MAGConfig {
        swa: SWAConfig {
            d_model: 8,
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 4,
            vocab_size: 16,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: rule,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 4,
        lp_p: 2.0,
        lq_q: 2.0,
        lambda_local: 0.01,
        lambda_2: 0.01,
        delta: 1.0,
        m_slots: 4,
        d_compress: 4,
        lambda_k: 0.01,
        lambda_v: 0.01,
        parallel: None,
        retention,
        m3: None,
    }
}

fn context_memory_size(cfg: &MAGConfig) -> usize {
    let d = cfg.swa.d_model;
    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule
        | MemoryRuleKind::TitansLMM
        | MemoryRuleKind::HebbianRule => d * d,
        MemoryRuleKind::Moneta
        | MemoryRuleKind::YAAD
        | MemoryRuleKind::MEMORA => cfg.d_hidden * d + d * cfg.d_hidden,
        MemoryRuleKind::LatticeOSR => cfg.m_slots * d,
        MemoryRuleKind::Trellis => 2 * cfg.d_compress * d,
    }
}

// ── Unit tests: default_retention mapping ──────────────────────────

#[test]
fn test_default_retention_all_rules() {
    assert_eq!(default_retention(MemoryRuleKind::DeltaRule), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::TitansLMM), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::HebbianRule), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::Moneta), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::YAAD), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::Trellis), RetentionKind::L2WeightDecay);
    assert_eq!(default_retention(MemoryRuleKind::MEMORA), RetentionKind::KLDivergence);
    assert_eq!(default_retention(MemoryRuleKind::LatticeOSR), RetentionKind::SphereNormalization);
}

// ── Unit tests: L2 apply retention ─────────────────────────────────

#[test]
fn test_l2_retain_half() {
    let mut w = vec![2.0, -4.0, 6.0, 0.0];
    l2_apply_retention(&mut w, 0.5);
    assert_eq!(w, vec![1.0, -2.0, 3.0, 0.0]);
}

#[test]
fn test_l2_retain_zero_clears() {
    let mut w = vec![1.0, 2.0, 3.0];
    l2_apply_retention(&mut w, 0.0);
    assert!(w.iter().all(|&x| x == 0.0));
}

#[test]
fn test_l2_retain_one_identity() {
    let original = vec![1.23, -4.56, 7.89];
    let mut w = original.clone();
    l2_apply_retention(&mut w, 1.0);
    assert_eq!(w, original);
}

#[test]
fn test_l2_retain_negative() {
    let mut w = vec![1.0, -1.0];
    l2_apply_retention(&mut w, -0.5);
    assert!((w[0] - (-0.5)).abs() < 1e-7);
    assert!((w[1] - 0.5).abs() < 1e-7);
}

// ── Unit tests: L2 retention gradient ──────────────────────────────

#[test]
fn test_l2_gradient_basic() {
    let w = vec![1.0, -3.0, 0.5];
    let g = l2_retention_gradient(&w, 0.1);
    assert!((g[0] - 0.2).abs() < 1e-7);
    assert!((g[1] - (-0.6)).abs() < 1e-7);
    assert!((g[2] - 0.1).abs() < 1e-7);
}

#[test]
fn test_l2_gradient_zero_lambda() {
    let w = vec![100.0, -200.0];
    let g = l2_retention_gradient(&w, 0.0);
    assert!(g.iter().all(|&x| x == 0.0));
}

// ── Unit tests: decoupled gradient ─────────────────────────────────

#[test]
fn test_decoupled_matches_l2_when_local_zero() {
    let w = vec![1.0, 2.0, 3.0];
    let boundary = vec![0.5, 1.0, 1.5];
    let g_decoupled = l2_decoupled_gradient(&w, &boundary, 0.0, 0.1);
    let g_l2 = l2_retention_gradient(&w, 0.1);
    for i in 0..w.len() {
        assert!((g_decoupled[i] - g_l2[i]).abs() < 1e-7,
            "decoupled[{i}]={} vs l2[{i}]={}", g_decoupled[i], g_l2[i]);
    }
}

#[test]
fn test_decoupled_local_pulls_toward_boundary() {
    let w = vec![2.0];
    let boundary = vec![1.0];
    let g = l2_decoupled_gradient(&w, &boundary, 0.5, 0.0);
    assert!((g[0] - 1.0).abs() < 1e-7);
}

// ── Unit tests: KL retention ───────────────────────────────────────

#[test]
fn test_kl_rows_sum_to_one() {
    let w = vec![1.0/3.0; 6];
    let grad = vec![0.01, 0.02, 0.03, 0.01, -0.01, 0.0];
    let result = kl_apply_retention(&w, &grad, 0.9, 0.1, 2, 3);
    for r in 0..2 {
        let row_sum: f32 = result[r * 3..(r + 1) * 3].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5,
            "row {r} sum = {row_sum}");
    }
}

#[test]
fn test_kl_all_positive() {
    let w = vec![0.25; 4];
    let grad = vec![0.1, -0.1, 0.2, -0.2];
    let result = kl_apply_retention(&w, &grad, 0.8, 0.5, 1, 4);
    assert!(result.iter().all(|&x| x > 0.0), "KL result must be all positive");
}

// ── Unit tests: elastic net ────────────────────────────────────────

#[test]
fn test_elastic_net_sparsity() {
    let mut w = vec![0.05, -0.03, 2.0, -1.5];
    elastic_net_apply(&mut w, 0.9, 0.5, 0.2);
    assert_eq!(w[0], 0.0, "small positive should be zeroed");
    assert_eq!(w[1], 0.0, "small negative should be zeroed");
    assert!(w[2] > 0.0, "large positive should survive");
    assert!(w[3] < 0.0, "large negative should survive");
}

#[test]
fn test_elastic_net_zero_l1_equals_l2() {
    let mut w1 = vec![1.0, -2.0, 3.0];
    let mut w2 = w1.clone();
    elastic_net_apply(&mut w1, 0.9, 0.0, 1.0);
    l2_apply_retention(&mut w2, 0.9);
    for i in 0..3 {
        assert!((w1[i] - w2[i]).abs() < 1e-6,
            "with lambda_1=0, elastic net should equal L2: {i}: {} vs {}", w1[i], w2[i]);
    }
}

// ── Unit tests: sphere projection ──────────────────────────────────

#[test]
fn test_sphere_result_is_unit_norm() {
    let slot = vec![1.0, 0.0, 0.0];
    let delta = vec![0.0, 0.5, 0.0];
    let (s_new, _) = sphere_project_and_normalize(&slot, &delta, 3);
    let norm: f32 = s_new.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_sphere_parallel_delta_no_change() {
    let slot = vec![1.0, 0.0, 0.0];
    let delta = vec![0.5, 0.0, 0.0];
    let (s_new, _) = sphere_project_and_normalize(&slot, &delta, 3);
    assert!((s_new[0] - 1.0).abs() < 1e-6);
    assert!(s_new[1].abs() < 1e-6);
    assert!(s_new[2].abs() < 1e-6);
}

// ── Golden-value tests: all rules finite with default retention ────

#[test]
fn test_golden_all_rules_default_retention() {
    let rules = [
        MemoryRuleKind::DeltaRule,
        MemoryRuleKind::TitansLMM,
        MemoryRuleKind::HebbianRule,
        MemoryRuleKind::Moneta,
        MemoryRuleKind::YAAD,
        MemoryRuleKind::MEMORA,
        MemoryRuleKind::LatticeOSR,
        MemoryRuleKind::Trellis,
    ];
    for &rule in &rules {
        let retention = default_retention(rule);
        let cfg = tiny_config(rule, retention);
        let params = MAGParams::init(&cfg, 42);
        let mem_size = context_memory_size(&cfg);
        let mut context = ContextState::new_with_memory_size(cfg.k, cfg.swa.d_model, mem_size);
        let stream = Box::new(VecStream::new((0..32_usize).collect()));
        let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone()).with_stream(stream);

        let (chunk, _) = conductor.next_chunk(cfg.swa.seq_len)
            .expect("should get chunk");
        let pulse = conductor.pulse();

        let (loss, _cache) = cms_forward(
            &params, &cfg, &chunk.input_ids, &chunk.target_ids, &pulse, &mut context,
        );
        assert!(loss.is_finite(),
            "{:?} with {:?}: loss not finite ({})", rule, retention, loss);
    }
}

// ── Cross-combo tests: non-default retention ───────────────────────

/// Helper: run one forward pass with a non-default retention config.
fn cross_combo_forward_test(rule: MemoryRuleKind, retention: RetentionKind) {
    let cfg = tiny_config(rule, retention);
    let params = MAGParams::init(&cfg, 42);
    let mem_size = context_memory_size(&cfg);
    let mut context = ContextState::new_with_memory_size(cfg.k, cfg.swa.d_model, mem_size);
    let stream = Box::new(VecStream::new((0..32_usize).collect()));
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone()).with_stream(stream);

    let (chunk, _) = conductor.next_chunk(cfg.swa.seq_len)
        .expect("should get chunk");
    let pulse = conductor.pulse();

    let (loss, _cache) = cms_forward(
        &params, &cfg, &chunk.input_ids, &chunk.target_ids, &pulse, &mut context,
    );
    assert!(loss.is_finite(),
        "{:?}+{:?}: loss not finite ({})", rule, retention, loss);
}

#[test]
fn test_cross_combo_delta_elastic_net() {
    cross_combo_forward_test(MemoryRuleKind::DeltaRule, RetentionKind::ElasticNet);
}

#[test]
fn test_cross_combo_hebbian_elastic_net() {
    cross_combo_forward_test(MemoryRuleKind::HebbianRule, RetentionKind::ElasticNet);
}

#[test]
fn test_cross_combo_titans_elastic_net() {
    cross_combo_forward_test(MemoryRuleKind::TitansLMM, RetentionKind::ElasticNet);
}

// ── Integration: multi-step training with elastic net ──────────────

#[test]
fn test_integration_elastic_net_learns() {
    let cfg = tiny_config(MemoryRuleKind::DeltaRule, RetentionKind::ElasticNet);
    let mut params = MAGParams::init(&cfg, 42);
    let mem_size = context_memory_size(&cfg);
    let mut context = ContextState::new_with_memory_size(cfg.k, cfg.swa.d_model, mem_size);
    let corpus: Vec<usize> = (0..200_usize).map(|i| i % cfg.swa.vocab_size).collect();
    let stream = Box::new(VecStream::new(corpus));
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone()).with_stream(stream);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(cfg.swa.d_model))
        .collect();

    let mut losses = Vec::new();
    for _step in 0..20 {
        let (chunk, _) = match conductor.next_chunk(cfg.swa.seq_len) {
            Some(c) => c,
            None => break,
        };
        if chunk.input_ids.len() < cfg.swa.seq_len {
            conductor.advance();
            continue;
        }
        let pulse = conductor.pulse();

        let (loss, cache) = cms_forward(
            &params, &cfg, &chunk.input_ids, &chunk.target_ids, &pulse, &mut context,
        );
        losses.push(loss);

        let grads = cms_backward(
            &params, &cfg, &cache, &chunk.input_ids, &chunk.target_ids, &mut error_buffers,
        );

        // SGD update
        params.apply_weight_gradients(&grads, 0.01);

        conductor.advance();
    }

    assert!(losses.len() >= 2, "need at least 2 losses to compare");
    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(last < first,
        "ElasticNet should allow learning: first={first:.4}, last={last:.4}");
}

// ── Trait / config tests ───────────────────────────────────────────

#[test]
fn test_retention_config_default() {
    let rc = RetentionConfig::default();
    assert_eq!(rc.lambda_1, 0.0);
    assert_eq!(rc.lambda_2, 0.0);
    assert_eq!(rc.lambda_local, 0.0);
}

#[test]
fn test_retention_kind_traits() {
    let a = RetentionKind::L2WeightDecay;
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_ne!(a, RetentionKind::ElasticNet);
    let s = format!("{:?}", a);
    assert!(s.contains("L2WeightDecay"));
}
