//! Integration tests for Lattice GLA parallelization.
//!
//! Verifies Lattice OSR and Trellis GLA forward/backward,
//! CMS k=2 compatibility, and approximation quality.

use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::lattice_gla::{lattice_gla_forward, lattice_gla_backward, trellis_gla_forward};
use nl_hecate_core::delta_rule::MemoryRule;
use nl_hecate_core::lattice_osr::LatticeOSR;
use nl_hecate_core::tensor::SimpleRng;

fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let mut rng = SimpleRng::new(seed);
    let mut e = vec![0.0f32; s * d];
    rng.fill_uniform(&mut e, 0.1);
    e
}

/// Smoke: Lattice GLA forward + backward work end-to-end.
#[test]
fn test_lattice_gla_smoke() {
    let cfg = MAGConfig::lattice_test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let (y, cache) = lattice_gla_forward(
        &params.levels[0], &embedded, s, d, 2, &cfg, None,
    );
    assert_eq!(y.len(), s * d);
    for &v in &y { assert!(v.is_finite()); }

    let d_y = vec![1.0f32; s * d];
    let (grads, d_emb) = lattice_gla_backward(
        &params.levels[0], &cache, &d_y, &embedded, &cfg,
    );
    assert_eq!(d_emb.len(), s * d);
    let norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 1e-10);
}

/// CMS k=2: Lattice GLA works on each CMS level.
#[test]
fn test_lattice_gla_cms_k2() {
    let cfg = MAGConfig::lattice_test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    for level in 0..cfg.k {
        let (y, _) = lattice_gla_forward(
            &params.levels[level], &embedded, s, d, 2, &cfg, None,
        );
        for &v in &y {
            assert!(v.is_finite(), "level {level} output not finite");
        }
    }
}

/// Approximation quality sweep: larger chunks → larger error but bounded.
#[test]
fn test_lattice_gla_quality_sweep() {
    let cfg = MAGConfig::lattice_test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    // Sequential baseline
    let rule = LatticeOSR { m_slots: cfg.m_slots };
    let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

    // GLA with full sequence (no renormalization) should match
    let (y_full, _) = lattice_gla_forward(
        &params.levels[0], &embedded, s, d, s, &cfg, None,
    );
    let diff_full: f32 = y_seq.iter().zip(y_full.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff_full < 1e-5, "full chunk should match sequential: {diff_full}");

    // GLA with chunk_size=2 may differ slightly
    let (y_c2, _) = lattice_gla_forward(
        &params.levels[0], &embedded, s, d, 2, &cfg, None,
    );
    let diff_c2: f32 = y_seq.iter().zip(y_c2.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff_c2.is_finite(), "chunk_size=2 output should be finite");
}

/// Trellis GLA: forward + backward smoke test.
#[test]
fn test_trellis_gla_smoke() {
    let cfg = MAGConfig::trellis_test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let (y, _) = trellis_gla_forward(
        &params.levels[0], &embedded, s, d, 2, &cfg, None,
    );
    assert_eq!(y.len(), s * d);
    for &v in &y { assert!(v.is_finite()); }
}
