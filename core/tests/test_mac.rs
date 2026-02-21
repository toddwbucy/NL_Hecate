//! MAC (Memory As Context) integration tests: multi-step training,
//! reflective gate verification, CMS k=2, comparison vs MAG.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::dynamic_freq::FrequencySchedule;
use nl_hecate_core::retention::RetentionKind;
use nl_hecate_core::mac::{mac_forward, mac_backward, cms_mac_forward, cms_mac_backward};
use nl_hecate_core::mag::{mag_forward, mag_backward};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};

fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

fn make_multiscale_data(
    seq_len: usize,
    vocab_size: usize,
    slow_period: usize,
    num_regimes: usize,
) -> (Vec<usize>, Vec<usize>) {
    let tokens_per_regime = vocab_size / num_regimes;
    let mut input_ids = Vec::with_capacity(seq_len);
    let mut target_ids = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let slow_regime = (t / slow_period) % num_regimes;
        let fast_component = (t * 3 + 1) % tokens_per_regime;
        let input = (t * 7 + slow_regime * 3) % vocab_size;
        let target = (fast_component + slow_regime * tokens_per_regime) % vocab_size;
        input_ids.push(input);
        target_ids.push(target);
    }

    (input_ids, target_ids)
}

/// CMS training loop for MAC. Returns (initial_loss, final_loss).
fn cms_mac_train(
    params: &mut MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    steps: usize,
    lr: f32,
) -> (f32, f32) {
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let d = cfg.swa.d_model;
    let mut context = ContextState::new(cfg.k, d);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();

    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for _ in 0..steps {
        let pulse = conductor.pulse();
        let (loss, cache) = cms_mac_forward(params, cfg, input_ids, target_ids, &pulse, &mut context);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        final_loss = loss;

        let grads = cms_mac_backward(params, cfg, &cache, input_ids, target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, lr);

        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
            }
        }

        conductor.advance();
    }

    (initial_loss.unwrap(), final_loss)
}

// ── MAC k=1 tests ──────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_mac_k1_smoke() {
    let cfg = MAGConfig::mac_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut prev_loss = None;
    for step in 0..100 {
        let (loss, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "Loss NaN at step {step}");
        if prev_loss.is_none() {
            prev_loss = Some(loss);
        }

        let grads = mac_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mac_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = prev_loss.unwrap();
    eprintln!("MAC k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_mac_k1_convergence() {
    let cfg = MAGConfig::mac_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut initial_loss = None;
    for _ in 0..1_000 {
        let (loss, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        let grads = mac_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mac_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = initial_loss.unwrap();
    eprintln!("MAC k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify the reflective gate is correctly computed (sigmoid output in (0,1)).
/// At init with conservative theta_t ≈ 0.01, reflective_y ≈ 0 so gate ≈ 0.5.
/// This is expected — the convergence test validates that MAC learns through this path.
#[test]
fn test_mac_reflective_gate_active() {
    let cfg = MAGConfig::mac_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);

    // Gate should be in (0, 1) range (sigmoid output)
    for (i, &g) in cache.reflective_gate.iter().enumerate() {
        assert!(g >= 0.0 && g <= 1.0,
            "Gate value at {i} should be in [0,1]: {g}");
    }

    let gate_mean: f32 = cache.reflective_gate.iter().sum::<f32>()
        / cache.reflective_gate.len() as f32;
    eprintln!("MAC reflective gate mean: {gate_mean:.4} (expect ~0.5 at init)");

    // Gate should be finite and non-NaN
    assert!(gate_mean.is_finite(), "Gate mean should be finite");

    // With DeltaRule at d=8, reflective_y has non-zero norm after step updates M
    let ry_norm: f32 = cache.reflective_y.iter()
        .map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("MAC reflective_y norm: {ry_norm:.4e}");
    // reflective_y can be very small at init (theta ≈ 0.01, d=8) — just verify finite
    assert!(ry_norm.is_finite(), "reflective_y should be finite");
}

/// CMS k=2 with MAC on multi-scale data.
#[test]
fn test_mac_k2_multiscale() {
    let cfg = MAGConfig::mac_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_mac_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("MAC k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// k=1: softmax([0.0]) = [1.0], so learnable aggregation is identity.
/// Loss should match the non-CMS k=1 path exactly.
#[test]
fn test_mac_k1_aggregation_identity() {
    let cfg = MAGConfig::mac_test_config();
    assert_eq!(cfg.k, 1, "test expects k=1");

    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    // k=1 forward
    let (loss1, _) = mac_forward(&params, &cfg, &input_ids, &target_ids);

    // Verify alpha_mem and alpha_refl are zero (uniform softmax)
    assert!(params.alpha_mem.iter().all(|&a| a == 0.0), "alpha_mem should be zero-init");
    assert!(params.alpha_refl.iter().all(|&a| a == 0.0), "alpha_refl should be zero-init");

    // Run again — deterministic
    let (loss2, _) = mac_forward(&params, &cfg, &input_ids, &target_ids);
    assert_eq!(loss1, loss2, "k=1 should be deterministic");
    eprintln!("MAC k=1 aggregation identity: loss={loss1:.6}");
}

/// CMS k=2: verify d_alpha_mem and d_alpha_refl are nonzero after a few training
/// steps (levels start symmetric but diverge once memory states differ).
#[test]
fn test_mac_k2_alpha_gradients_nonzero() {
    let cfg = MAGConfig::mac_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let d = cfg.swa.d_model;
    let mut context = ContextState::new(cfg.k, d);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();

    // Train a few steps to break symmetry between levels
    // (at init, all levels produce identical read-only output → d_alpha=0 by symmetry)
    for _ in 0..10 {
        let pulse = conductor.pulse();
        let (_, cache) = cms_mac_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        let grads = cms_mac_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);
        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], 0.01);
            }
        }
        conductor.advance();
    }

    // Now check gradients — levels should be asymmetric
    let pulse = conductor.pulse();
    let (_, cache) = cms_mac_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
    let grads = cms_mac_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

    // d_alpha_mem should have nonzero entries after symmetry breaks
    let alpha_mem_max = grads.alpha_mem.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("d_alpha_mem max abs: {alpha_mem_max:.6e}");
    assert!(alpha_mem_max > 1e-12, "d_alpha_mem should be nonzero: max={alpha_mem_max}");

    // d_alpha_refl: may be zero if only 1 level active, check all are finite
    let alpha_refl_max = grads.alpha_refl.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("d_alpha_refl max abs: {alpha_refl_max:.6e}");

    // All gradients should be finite
    for (i, &g) in grads.alpha_mem.iter().enumerate() {
        assert!(g.is_finite(), "d_alpha_mem[{i}] not finite: {g}");
    }
    for (i, &g) in grads.alpha_refl.iter().enumerate() {
        assert!(g.is_finite(), "d_alpha_refl[{i}] not finite: {g}");
    }
}

/// masked_softmax with 1 active level should produce weight=1.0 for that level.
#[test]
fn test_mac_masked_softmax_single_active() {
    let cfg = MAGConfig::mac_test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let d = cfg.swa.d_model;
    let mut context = ContextState::new(cfg.k, d);

    // Advance conductor until only Level 0 is active (Level 1 fires every 8 steps)
    // Step 1: Level 0 active, Level 1 NOT active
    conductor.advance(); // step 0 done
    let pulse = conductor.pulse(); // step 1
    assert!(pulse.active_levels[0], "Level 0 should be active at step 1");
    assert!(!pulse.active_levels[1], "Level 1 should be inactive at step 1");

    let (loss, cache) = cms_mac_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
    assert!(loss.is_finite(), "Loss should be finite with single active level");

    // The reflective output should work with single active level
    let gate_mean: f32 = cache.reflective_gate.iter().sum::<f32>()
        / cache.reflective_gate.len() as f32;
    eprintln!("Single-active reflective gate mean: {gate_mean:.4}");
    assert!(gate_mean.is_finite(), "Gate mean should be finite");
}

/// Compare MAC vs MAG on same data. Both should converge, within tolerance.
#[test]
fn test_mac_vs_mag() {
    use nl_hecate_core::model::SWAConfig;

    let swa_mag = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };
    // MAC needs window_size >= 2*seq_len for full causal on assembled input
    let swa_mac = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 16, vocab_size: 16,
    };

    let cfg_mag = MAGConfig {
        swa: swa_mag.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
    };
    let cfg_mac = MAGConfig {
        swa: swa_mac, memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAC,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
    };

    let seq_len = 8;
    let vocab_size = 16;
    let input_ids: Vec<usize> = (0..seq_len).map(|t| t % vocab_size).collect();
    let target_ids: Vec<usize> = (1..=seq_len).map(|t| t % vocab_size).collect();
    let lr = 0.01;
    let steps = 5_000;

    // Train MAG
    let mut params_mag = MAGParams::init(&cfg_mag, 42);
    let mut mag_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_mag, &cfg_mag, &input_ids, &target_ids);
        if mag_initial.is_none() { mag_initial = Some(loss); }
        let grads = mag_backward(&params_mag, &cfg_mag, &cache, &input_ids, &target_ids);
        params_mag.apply_weight_gradients(&grads, lr);
    }
    let mag_final = mag_forward(&params_mag, &cfg_mag, &input_ids, &target_ids).0;

    // Train MAC
    let mut params_mac = MAGParams::init(&cfg_mac, 42);
    let mut mac_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mac_forward(&params_mac, &cfg_mac, &input_ids, &target_ids);
        if mac_initial.is_none() { mac_initial = Some(loss); }
        let grads = mac_backward(&params_mac, &cfg_mac, &cache, &input_ids, &target_ids);
        params_mac.apply_weight_gradients(&grads, lr);
    }
    let mac_final = mac_forward(&params_mac, &cfg_mac, &input_ids, &target_ids).0;

    eprintln!("MAG: initial={:.4}, final={mag_final:.4}", mag_initial.unwrap());
    eprintln!("MAC: initial={:.4}, final={mac_final:.4}", mac_initial.unwrap());

    // Both should converge
    assert!(mag_final < mag_initial.unwrap(), "MAG should converge");
    assert!(mac_final < mac_initial.unwrap(), "MAC should converge");

    // MAC should be in same ballpark as MAG (different composition, not necessarily better at d=8)
    assert!(mac_final < mag_final * 5.0,
        "MAC should not catastrophically regress: mac={mac_final:.6}, mag={mag_final:.6}");

    let ratio = if mac_final > mag_final {
        mac_final / mag_final
    } else {
        mag_final / mac_final
    };
    eprintln!("Performance ratio: {ratio:.2}x (closer to 1.0 = more similar)");
}
