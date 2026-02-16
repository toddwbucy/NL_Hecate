//! MEMORA integration tests: multi-step training, simplex preservation, CMS k=2, comparison vs Delta Rule.

use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::mag::{cms_forward, cms_backward, mag_forward, mag_backward};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};

fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

/// Run MAG training loop for N steps. Returns (initial_loss, final_loss).
fn mag_train(
    params: &mut MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    steps: usize,
    lr: f32,
) -> (f32, f32) {
    let (initial_loss, _) = mag_forward(params, cfg, input_ids, target_ids);
    for _ in 0..steps {
        let (_, cache) = mag_forward(params, cfg, input_ids, target_ids);
        let grads = mag_backward(params, cfg, &cache, input_ids, target_ids);
        params.apply_weight_gradients(&grads, lr);
    }
    let (final_loss, _) = mag_forward(params, cfg, input_ids, target_ids);
    (initial_loss, final_loss)
}

/// Run CMS training loop for N steps. Returns (initial_loss, final_loss).
fn cms_train(
    params: &mut MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    steps: usize,
    lr: f32,
) -> (f32, f32) {
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let d = cfg.swa.d_model;
    let dh = cfg.d_hidden;
    let mem_size = dh * d + d * dh;
    let mut context = ContextState::new_with_memory_size(cfg.k, d, mem_size);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();

    let mut initial_loss = 0.0f32;
    for step in 0..steps {
        let pulse = conductor.pulse();
        let (loss, cache) = cms_forward(params, cfg, input_ids, target_ids, &pulse, &mut context);
        if step == 0 { initial_loss = loss; }
        let grads = cms_backward(params, cfg, &cache, input_ids, target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, lr);
        // Apply error buffer grads when levels become active
        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
            }
        }
        conductor.advance();
    }
    // Final loss
    let pulse = conductor.pulse();
    let (final_loss, _) = cms_forward(params, cfg, input_ids, target_ids, &pulse, &mut context);
    (initial_loss, final_loss)
}

#[test]
fn test_memora_k1_smoke() {
    let cfg = MAGConfig::memora_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (initial_loss, final_loss) = mag_train(&mut params, &cfg, &input_ids, &target_ids, 100, 0.01);
    eprintln!("MEMORA k=1 smoke: initial={initial_loss:.4}, final={final_loss:.4}");
    assert!(initial_loss.is_finite(), "Initial loss not finite: {initial_loss}");
    assert!(final_loss.is_finite(), "Final loss not finite: {final_loss}");
    // No NaN after 100 steps
}

#[test]
fn test_memora_k1_convergence() {
    let mut cfg = MAGConfig::memora_test_config();
    cfg.swa.seq_len = 8;
    cfg.swa.window_size = 8;
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (initial_loss, final_loss) = mag_train(&mut params, &cfg, &input_ids, &target_ids, 1000, 0.01);
    eprintln!("MEMORA k=1 convergence: initial={initial_loss:.4}, final={final_loss:.4}");
    assert!(final_loss < initial_loss,
        "Loss should decrease: initial={initial_loss:.4}, final={final_loss:.4}");
}

#[test]
fn test_memora_simplex_preserved_after_training() {
    let cfg = MAGConfig::memora_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    // Train for 50 steps
    for _ in 0..50 {
        let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    // Run one more forward to check simplex on the cached W states
    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    match &cache.memory_cache {
        nl_hecate_core::mag::MemoryCache::MEMORA(mc) => {
            let d = mc.d;
            let dh = mc.d_hidden;
            let s = mc.seq_len;
            let w1_size = dh * d;
            let w2_size = d * dh;

            // Check final W1
            let w1_final = &mc.w1_states[s * w1_size..(s + 1) * w1_size];
            for r in 0..dh {
                let row = &w1_final[r * d..(r + 1) * d];
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4,
                    "After training, W1 row {r}: sum={sum}, expected ~1.0");
                for &v in row {
                    assert!(v >= 0.0, "W1 entry negative after training: {v}");
                }
            }
            // Check final W2
            let w2_final = &mc.w2_states[s * w2_size..(s + 1) * w2_size];
            for r in 0..d {
                let row = &w2_final[r * dh..(r + 1) * dh];
                let sum: f32 = row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4,
                    "After training, W2 row {r}: sum={sum}, expected ~1.0");
            }
        }
        _ => panic!("Expected MEMORACache"),
    }
}

#[test]
fn test_memora_k2_multiscale() {
    let cfg = MAGConfig::memora_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (initial_loss, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 500, 0.01);
    eprintln!("MEMORA k=2: initial={initial_loss:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite: {final_loss}");
    assert!(final_loss < initial_loss,
        "CMS k=2 loss should decrease: initial={initial_loss:.4}, final={final_loss:.4}");
}

#[test]
fn test_memora_vs_delta() {
    // Both should converge — MEMORA within 5x of Delta Rule
    let mut delta_cfg = MAGConfig::test_config();
    delta_cfg.swa.seq_len = 8;
    delta_cfg.swa.window_size = 8;
    let mut delta_params = MAGParams::init(&delta_cfg, 42);
    let (input_ids, target_ids) = make_data(&delta_cfg);

    let (_, delta_final) = mag_train(&mut delta_params, &delta_cfg, &input_ids, &target_ids, 500, 0.01);

    let mut memora_cfg = MAGConfig::memora_test_config();
    memora_cfg.swa.seq_len = 8;
    memora_cfg.swa.window_size = 8;
    let mut memora_params = MAGParams::init(&memora_cfg, 42);
    let (input_ids2, target_ids2) = make_data(&memora_cfg);

    let (_, memora_final) = mag_train(&mut memora_params, &memora_cfg, &input_ids2, &target_ids2, 500, 0.01);

    eprintln!("Delta final: {delta_final:.4}, MEMORA final: {memora_final:.4}");
    // MEMORA should converge (within 5x of Delta Rule)
    assert!(memora_final < delta_final * 5.0,
        "MEMORA ({memora_final:.4}) should be within 5x of Delta ({delta_final:.4})");
}
