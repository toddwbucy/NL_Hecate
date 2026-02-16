//! Integration tests for associative scan parallelization.
//!
//! Verifies Hebbian scan exactness, CMS k=2 compatibility,
//! and Titans momentum scan accuracy.

use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::associative_scan::{hebbian_scan_forward, hebbian_scan_backward, titans_momentum_scan};
use nl_hecate_core::delta_rule::MemoryRule;
use nl_hecate_core::hebbian_rule::HebbianRule;
use nl_hecate_core::tensor::SimpleRng;

fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let mut rng = SimpleRng::new(seed);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 0.1);
    embedded
}

/// Hebbian scan exactly matches sequential Hebbian for full pipeline.
#[test]
fn test_hebbian_scan_exact_vs_sequential() {
    let cfg = MAGConfig::hebbian_test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let (y_scan, _) = hebbian_scan_forward(&params.levels[0], &embedded, s, d, None);
    let (y_seq, _) = HebbianRule.step(&params.levels[0], &embedded, s, d, None);

    let max_diff: f32 = y_scan.iter().zip(y_seq.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4,
        "Hebbian scan should be exact: max_diff={max_diff}");
}

/// CMS k=2: Hebbian scan works at both levels.
#[test]
fn test_hebbian_scan_cms_k2() {
    let cfg = MAGConfig::hebbian_test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    for level in 0..cfg.k {
        let (y, _) = hebbian_scan_forward(&params.levels[level], &embedded, s, d, None);
        assert_eq!(y.len(), s * d);
        for &v in &y {
            assert!(v.is_finite(), "level {level} output not finite");
        }
    }
}

/// Hebbian scan backward produces correct gradient shapes.
#[test]
fn test_hebbian_scan_backward_integration() {
    let cfg = MAGConfig::hebbian_test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let (_, cache) = hebbian_scan_forward(&params.levels[0], &embedded, s, d, None);
    let d_y = vec![1.0f32; s * d];
    let (grads, d_emb) = hebbian_scan_backward(&params.levels[0], &cache, &d_y, &embedded);

    let norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 1e-10, "gradients should be non-zero");
    assert_eq!(d_emb.len(), s * d);
}

/// Titans momentum scan: final state matches sequential for moderate sequences.
#[test]
fn test_titans_momentum_scan_integration() {
    let n = 16;
    let state_size = 64; // d*d for d=8
    let mut rng = SimpleRng::new(42);

    let mut etas = vec![0.0f32; n];
    rng.fill_uniform(&mut etas, 1.0);
    for e in &mut etas { *e = e.abs().min(0.99); }
    let mut thetas = vec![0.0f32; n];
    rng.fill_uniform(&mut thetas, 0.05);
    let mut grads = vec![0.0f32; n * state_size];
    rng.fill_uniform(&mut grads, 1.0);
    let s_init = vec![0.0f32; state_size];

    let s_scan = titans_momentum_scan(&etas, &thetas, &grads, &s_init, state_size);

    // Sequential check of final state
    let mut s = s_init.clone();
    for t in 0..n {
        let mut s_new = vec![0.0f32; state_size];
        for j in 0..state_size {
            s_new[j] = etas[t] * s[j] - thetas[t] * grads[t * state_size + j];
        }
        s = s_new;
    }

    // Final state should match
    let max_diff: f32 = s_scan[(n - 1) * state_size..n * state_size].iter()
        .zip(s.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-3,
        "Titans momentum scan final state mismatch: max_diff={max_diff}");
}
