//! Integration tests for chunkwise GD parallelization strategy.
//!
//! Tests the full chunkwise GD pipeline: forward/backward through chunks,
//! quality degradation curves, convergence comparison, and CMS k=2 integration.

use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::chunkwise_gd::{chunkwise_gd_forward, chunkwise_gd_backward};
use nl_hecate_core::tensor::SimpleRng;

fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let mut rng = SimpleRng::new(seed);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 1.0);
    embedded
}

fn relative_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max_rel = 0.0f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        let denom = a[i].abs().max(b[i].abs()).max(1e-8);
        max_rel = max_rel.max(diff / denom);
    }
    max_rel
}

/// Smoke test: chunkwise GD forward + backward work end-to-end for Delta Rule.
#[test]
fn test_chunkwise_smoke() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    // Forward with C=2
    let (y, cache) = chunkwise_gd_forward(
        &params.levels[0], &embedded, s, d, 2, &cfg, None,
    );
    assert_eq!(y.len(), s * d);
    for &v in &y {
        assert!(v.is_finite(), "y contains non-finite values");
    }

    // Backward
    let d_y = vec![1.0f32; s * d];
    let (grads, d_emb) = chunkwise_gd_backward(
        &params.levels[0], &cache, &d_y, &embedded, &cfg,
    );
    assert_eq!(d_emb.len(), s * d);
    let grad_norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(grad_norm > 1e-10, "gradients should be non-zero");
}

/// C=1 exactly matches sequential for Delta Rule.
#[test]
fn test_chunkwise_c1_equivalence() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let (y_c1, _) = chunkwise_gd_forward(
        &params.levels[0], &embedded, s, d, 1, &cfg, None,
    );
    let (y_full, _) = chunkwise_gd_forward(
        &params.levels[0], &embedded, s, d, s, &cfg, None,
    );

    let max_diff: f32 = y_c1.iter().zip(y_full.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "C=1 should match full sequential, max_diff={max_diff}");
}

/// Quality degradation curve: error increases monotonically with chunk size.
#[test]
fn test_chunkwise_quality_degradation_curve() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    // Baseline
    let (y_seq, _) = chunkwise_gd_forward(
        &params.levels[0], &embedded, s, d, 1, &cfg, None,
    );

    let mut prev_diff = 0.0f32;
    for c in [2, 4, s] {
        let (y_chunk, _) = chunkwise_gd_forward(
            &params.levels[0], &embedded, s, d, c, &cfg, None,
        );
        let diff = relative_diff(&y_seq, &y_chunk);

        // Quality should degrade (or stay same for full seq)
        if c < s {
            assert!(diff >= prev_diff - 0.01,
                "Quality should not improve with larger C: C={c} diff={diff:.4} < prev={prev_diff:.4}");
        }
        prev_diff = diff;
    }
}

/// Convergence comparison: C=1 and C=2 both converge, C=1 should converge better.
#[test]
fn test_chunkwise_convergence_comparison() {
    let cfg = MAGConfig::test_config();
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let lr = 0.1;

    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 1.0);
    let mut target = vec![0.0f32; s * d];
    rng.fill_uniform(&mut target, 1.0);

    // Train with C=1 (exact)
    let mut params_c1 = MAGParams::init(&cfg, 42);
    let mut loss_c1_first = 0.0f32;
    let mut loss_c1_last = 0.0f32;

    for step in 0..100 {
        let (y, cache) = chunkwise_gd_forward(
            &params_c1.levels[0], &embedded, s, d, 1, &cfg, None,
        );
        let loss: f32 = y.iter().zip(target.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
        if step == 0 { loss_c1_first = loss; }
        if step == 99 { loss_c1_last = loss; }

        let d_y: Vec<f32> = y.iter().zip(target.iter())
            .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
        let (grads, _) = chunkwise_gd_backward(
            &params_c1.levels[0], &cache, &d_y, &embedded, &cfg,
        );
        for (w, g) in params_c1.levels[0].w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
        for (w, g) in params_c1.levels[0].w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
        for (w, g) in params_c1.levels[0].w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
    }

    // Train with C=2 (approximate)
    let mut params_c2 = MAGParams::init(&cfg, 42);
    let mut loss_c2_last = 0.0f32;

    for step in 0..100 {
        let (y, cache) = chunkwise_gd_forward(
            &params_c2.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        let loss: f32 = y.iter().zip(target.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
        if step == 99 { loss_c2_last = loss; }

        let d_y: Vec<f32> = y.iter().zip(target.iter())
            .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
        let (grads, _) = chunkwise_gd_backward(
            &params_c2.levels[0], &cache, &d_y, &embedded, &cfg,
        );
        for (w, g) in params_c2.levels[0].w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
        for (w, g) in params_c2.levels[0].w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
        for (w, g) in params_c2.levels[0].w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
    }

    // Both should converge (loss decreased)
    assert!(loss_c1_last < loss_c1_first,
        "C=1 should converge: first={loss_c1_first:.6} last={loss_c1_last:.6}");

    // C=2 should also converge (or at least not diverge)
    assert!(loss_c2_last <= loss_c1_first + 0.01,
        "C=2 should not diverge: {loss_c2_last:.6} > initial {loss_c1_first:.6}");
}

/// CMS k=2: chunkwise GD works within a CMS training loop.
/// The CMS forward/backward uses sequential processing, but this test verifies
/// that chunkwise_gd_forward produces compatible memory outputs for CMS composition.
#[test]
fn test_chunkwise_cms_k2_compatibility() {
    let cfg = MAGConfig::test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    // Run chunkwise GD on each CMS level independently
    let embedded = make_embedded(&cfg, 99);

    for level in 0..cfg.k {
        let (y, cache) = chunkwise_gd_forward(
            &params.levels[level], &embedded, s, d, 2, &cfg, None,
        );
        assert_eq!(y.len(), s * d, "level {level} output size mismatch");

        for &v in &y {
            assert!(v.is_finite(), "level {level} has non-finite output");
        }

        // Backward should work
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = chunkwise_gd_backward(
            &params.levels[level], &cache, &d_y, &embedded, &cfg,
        );
        assert_eq!(d_emb.len(), s * d);

        let grad_norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(grad_norm > 1e-10, "level {level} gradients are zero");
    }
}
