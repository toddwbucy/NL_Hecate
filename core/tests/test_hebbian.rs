//! Hebbian Rule integration tests: multi-step training, comparison vs Delta Rule.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind};
use nl_hecate_core::mag::{cms_forward, cms_backward, mag_forward, mag_backward, MemoryCache};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};

fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

/// Generate multi-scale data with fast + slow temporal patterns.
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
    let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
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
        params.sgd_step(&grads, lr);

        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
            }
        }

        conductor.advance();
    }

    (initial_loss.unwrap(), final_loss)
}

// ── Hebbian k=1 tests ────────────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_hebbian_k1_smoke() {
    let cfg = MAGConfig::hebbian_test_config();
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
        params.sgd_step(&grads, 0.01);
    }

    let final_loss = {
        let (loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        loss
    };
    let initial = prev_loss.unwrap();
    eprintln!("Hebbian k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_hebbian_k1_convergence() {
    let cfg = MAGConfig::hebbian_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut initial_loss = None;
    for _ in 0..1_000 {
        let (loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.sgd_step(&grads, 0.01);
    }

    let final_loss = {
        let (loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        loss
    };
    let initial = initial_loss.unwrap();
    eprintln!("Hebbian k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify memory M is non-trivial after forward pass.
#[test]
fn test_hebbian_memory_nonzero() {
    let cfg = MAGConfig::hebbian_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    match &cache.memory_cache {
        MemoryCache::Hebbian(hc) => {
            let d = hc.d;
            let seq = hc.seq_len;
            let m_final = &hc.m_states[seq * d * d..(seq + 1) * d * d];
            let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final M norm: {m_norm:.6e}");
            assert!(m_norm > 1e-6, "Memory M should be non-trivial, got {m_norm:.4e}");
        }
        _ => {
            panic!("Expected HebbianCache");
        }
    }
}

/// CMS k=2 with Hebbian on multi-scale data.
#[test]
fn test_hebbian_k2_multiscale() {
    let cfg = MAGConfig::hebbian_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("Hebbian k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Compare Hebbian vs Delta Rule on same data.
/// Hebbian should converge but is expected to be less precise (no error correction).
#[test]
fn test_hebbian_vs_delta() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };

    let cfg_delta = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
    };
    let cfg_hebbian = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::HebbianRule,
        k: 1, chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
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
        params_delta.sgd_step(&grads, lr);
    }
    let delta_final = mag_forward(&params_delta, &cfg_delta, &input_ids, &target_ids).0;

    // Train Hebbian
    let mut params_hebbian = MAGParams::init(&cfg_hebbian, 42);
    let mut hebbian_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_hebbian, &cfg_hebbian, &input_ids, &target_ids);
        if hebbian_initial.is_none() { hebbian_initial = Some(loss); }
        let grads = mag_backward(&params_hebbian, &cfg_hebbian, &cache, &input_ids, &target_ids);
        params_hebbian.sgd_step(&grads, lr);
    }
    let hebbian_final = mag_forward(&params_hebbian, &cfg_hebbian, &input_ids, &target_ids).0;

    eprintln!("Delta: initial={:.4}, final={delta_final:.4}", delta_initial.unwrap());
    eprintln!("Hebbian: initial={:.4}, final={hebbian_final:.4}", hebbian_initial.unwrap());

    // Both should converge
    assert!(delta_final < delta_initial.unwrap(), "Delta should converge");
    assert!(hebbian_final < hebbian_initial.unwrap(), "Hebbian should converge");

    // Hebbian is expected to be less precise, but should still be in same ballpark
    // (within 5x of Delta — generous margin for no error correction)
    assert!(hebbian_final < delta_final * 5.0,
        "Hebbian should not catastrophically regress: hebbian={hebbian_final:.6}, delta={delta_final:.6}");

    if hebbian_final > delta_final {
        let ratio = hebbian_final / delta_final;
        eprintln!("NOTE: Delta beats Hebbian by {ratio:.2}x — expected (error correction helps)");
    } else {
        eprintln!("Hebbian matches/beats Delta at d=8 scale — surprising but possible");
    }
}
