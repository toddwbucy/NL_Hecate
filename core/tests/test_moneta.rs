//! MONETA integration tests: 2-layer MLP memory with l_p bias and L2 retention.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::dynamic_freq::FrequencySchedule;
use nl_hecate_core::retention::RetentionKind;
use nl_hecate_core::mag::{cms_forward, cms_backward, mag_forward, mag_backward, MemoryCache};
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

/// Create ContextState with correct memory size for MONETA (W1+W2 instead of d*d).
fn make_context_state(cfg: &MAGConfig) -> ContextState {
    let dh = cfg.d_hidden;
    let d = cfg.swa.d_model;
    let mem_size = dh * d + d * dh;
    ContextState::new_with_memory_size(cfg.k, d, mem_size)
}

/// CMS training loop for MONETA. Uses correct memory size for ContextState.
fn cms_train(
    params: &mut MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    steps: usize,
    lr: f32,
) -> (f32, f32) {
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let mut context = make_context_state(cfg);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(cfg.swa.d_model))
        .collect();

    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for _ in 0..steps {
        let pulse = conductor.pulse();
        let (loss, cache) = cms_forward(params, cfg, input_ids, target_ids, &pulse, &mut context);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        final_loss = loss;

        let grads = cms_backward(params, cfg, &cache, input_ids, target_ids, &mut error_buffers);
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

// ── MONETA k=1 tests ─────────────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_moneta_k1_smoke() {
    let cfg = MAGConfig::moneta_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut prev_loss = None;
    for step in 0..100 {
        let (loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "Loss NaN at step {step}");
        if prev_loss.is_none() {
            prev_loss = Some(loss);
        }

        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mag_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = prev_loss.unwrap();
    eprintln!("MONETA k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_moneta_k1_convergence() {
    let cfg = MAGConfig::moneta_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut initial_loss = None;
    for _ in 0..1_000 {
        let (loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mag_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = initial_loss.unwrap();
    eprintln!("MONETA k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify MLP W1/W2 are non-trivial after forward pass.
#[test]
fn test_moneta_mlp_nonzero() {
    let cfg = MAGConfig::moneta_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    match &cache.memory_cache {
        MemoryCache::Moneta(mc) => {
            let d = mc.d;
            let dh = mc.d_hidden;
            let s = mc.seq_len;
            let w1_size = dh * d;
            let w2_size = d * dh;

            let w1_final = &mc.w1_states[s * w1_size..(s + 1) * w1_size];
            let w1_norm: f32 = w1_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final W1 norm: {w1_norm:.6e}");
            assert!(w1_norm > 1e-6, "MLP W1 should be non-trivial, got {w1_norm:.4e}");

            let w2_final = &mc.w2_states[s * w2_size..(s + 1) * w2_size];
            let w2_norm: f32 = w2_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final W2 norm: {w2_norm:.6e}");
            assert!(w2_norm > 1e-6, "MLP W2 should be non-trivial, got {w2_norm:.4e}");
        }
        _ => panic!("Expected MonetaCache"),
    }
}

/// CMS k=2 with MONETA on multi-scale data.
#[test]
fn test_moneta_k2_multiscale() {
    let cfg = MAGConfig::moneta_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("MONETA k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Compare MONETA vs Delta Rule on same data.
/// MONETA should converge — MLP memory has higher capacity per parameter.
#[test]
fn test_moneta_vs_delta() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };

    let cfg_delta = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
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
    let cfg_moneta = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::Moneta,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 4, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.01, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
    };

    let input_ids: Vec<usize> = (0..swa.seq_len).map(|t| t % swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=swa.seq_len).map(|t| t % swa.vocab_size).collect();
    let lr = 0.01;
    let steps = 5_000;

    // Train Delta Rule
    let mut params_delta = MAGParams::init(&cfg_delta, 42);
    let mut delta_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_delta, &cfg_delta, &input_ids, &target_ids);
        if delta_initial.is_none() { delta_initial = Some(loss); }
        let grads = mag_backward(&params_delta, &cfg_delta, &cache, &input_ids, &target_ids);
        params_delta.apply_weight_gradients(&grads, lr);
    }
    let delta_final = mag_forward(&params_delta, &cfg_delta, &input_ids, &target_ids).0;

    // Train MONETA
    let mut params_moneta = MAGParams::init(&cfg_moneta, 42);
    let mut moneta_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_moneta, &cfg_moneta, &input_ids, &target_ids);
        if moneta_initial.is_none() { moneta_initial = Some(loss); }
        let grads = mag_backward(&params_moneta, &cfg_moneta, &cache, &input_ids, &target_ids);
        params_moneta.apply_weight_gradients(&grads, lr);
    }
    let moneta_final = mag_forward(&params_moneta, &cfg_moneta, &input_ids, &target_ids).0;

    eprintln!("Delta: initial={:.4}, final={delta_final:.4}", delta_initial.unwrap());
    eprintln!("MONETA: initial={:.4}, final={moneta_final:.4}", moneta_initial.unwrap());

    // Both should converge
    assert!(delta_final < delta_initial.unwrap(), "Delta should converge");
    assert!(moneta_final < moneta_initial.unwrap(), "MONETA should converge");

    // MONETA should be in same ballpark as Delta (MLP memory is different, not necessarily better at d=8)
    assert!(moneta_final < delta_final * 10.0,
        "MONETA should not catastrophically regress: moneta={moneta_final:.6}, delta={delta_final:.6}");

    let ratio = if moneta_final > delta_final {
        moneta_final / delta_final
    } else {
        delta_final / moneta_final
    };
    eprintln!("Performance ratio: {ratio:.2}x (closer to 1.0 = more similar)");
}
