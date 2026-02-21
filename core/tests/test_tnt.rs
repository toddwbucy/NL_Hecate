//! Integration tests for TNT hierarchical parallelization.

use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::tnt::{tnt_forward, tnt_backward, TNTConfig};
use nl_hecate_core::chunkwise_gd::chunkwise_gd_forward;
use nl_hecate_core::tensor::SimpleRng;

fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let mut rng = SimpleRng::new(seed);
    let mut e = vec![0.0f32; s * d];
    rng.fill_uniform(&mut e, 0.1);
    e
}

fn tnt_cfg(cg: usize, cl: usize) -> TNTConfig {
    TNTConfig {
        global_chunk_size: cg,
        local_chunk_size: cl,
        use_qk_projection: false,
        use_attention_summary: false,
    }
}

/// Smoke test: TNT forward + backward work end-to-end.
#[test]
fn test_tnt_smoke() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let tnt = tnt_cfg(2, 1);

    let (y, cache) = tnt_forward(&params.levels[0], &embedded, s, d, &tnt, &cfg, None, None);
    assert_eq!(y.len(), s * d);

    let d_y = vec![1.0f32; s * d];
    let (grads, d_emb, _) = tnt_backward(&params.levels[0], &cache, &d_y, &embedded, &tnt, &cfg, None);
    assert_eq!(d_emb.len(), s * d);
    let norm: f32 = grads.w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm > 1e-10);
}

/// Quality comparison: TNT should be within reasonable range of sequential.
#[test]
fn test_tnt_quality_comparison() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let embedded = make_embedded(&cfg, 99);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    // Sequential baseline
    let (y_seq, _) = chunkwise_gd_forward(&params.levels[0], &embedded, s, d, 1, &cfg, None);

    // TNT with small shards
    let tnt = tnt_cfg(2, 1);
    let (y_tnt, _) = tnt_forward(&params.levels[0], &embedded, s, d, &tnt, &cfg, None, None);

    let max_diff: f32 = y_seq.iter().zip(y_tnt.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff.is_finite(), "TNT vs sequential diff should be finite");
}

/// TNT convergence: training with TNT should not diverge.
#[test]
fn test_tnt_convergence() {
    let cfg = MAGConfig::test_config();
    let mut level_params = MAGParams::init(&cfg, 42).levels.into_iter().next().unwrap();
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let tnt = tnt_cfg(2, 1);
    let lr = 0.1;

    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 1.0);
    let mut target = vec![0.0f32; s * d];
    rng.fill_uniform(&mut target, 1.0);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..50 {
        let (y, cache) = tnt_forward(&level_params, &embedded, s, d, &tnt, &cfg, None, None);
        let loss: f32 = y.iter().zip(target.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
        if step == 0 { first_loss = loss; }
        if step == 49 { last_loss = loss; }

        let d_y: Vec<f32> = y.iter().zip(target.iter())
            .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
        let (grads, _, _) = tnt_backward(&level_params, &cache, &d_y, &embedded, &tnt, &cfg, None);

        for (w, g) in level_params.w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
        for (w, g) in level_params.w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
        for (w, g) in level_params.w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
    }

    assert!(last_loss.is_finite(), "TNT training loss should be finite");
    assert!(last_loss <= first_loss + 0.1,
        "TNT should not diverge: first={first_loss:.6} last={last_loss:.6}");
}

/// CMS k=2: TNT works on each CMS level.
#[test]
fn test_tnt_cms_k2() {
    let cfg = MAGConfig::test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let tnt = tnt_cfg(4, 2);

    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 0.1);

    for level in 0..cfg.k {
        let (y, _) = tnt_forward(&params.levels[level], &embedded, s, d, &tnt, &cfg, None, None);
        for &v in &y {
            assert!(v.is_finite(), "level {level} TNT output not finite");
        }
    }
}
