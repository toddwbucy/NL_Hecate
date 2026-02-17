//! MAL (Memory As Layer) integration tests: multi-step training,
//! memory-is-attention-input verification, CMS k=2, comparison vs MAG.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::dynamic_freq::FrequencySchedule;
use nl_hecate_core::retention::RetentionKind;
use nl_hecate_core::mal::{mal_forward, mal_backward, cms_mal_forward, cms_mal_backward};
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

/// CMS training loop for MAL. Returns (initial_loss, final_loss).
fn cms_mal_train(
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
        let (loss, cache) = cms_mal_forward(params, cfg, input_ids, target_ids, &pulse, &mut context);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        final_loss = loss;

        let grads = cms_mal_backward(params, cfg, &cache, input_ids, target_ids, &mut error_buffers);
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

// ── MAL k=1 tests ──────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_mal_k1_smoke() {
    let cfg = MAGConfig::mal_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut prev_loss = None;
    for step in 0..100 {
        let (loss, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "Loss NaN at step {step}");
        if prev_loss.is_none() {
            prev_loss = Some(loss);
        }

        let grads = mal_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mal_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = prev_loss.unwrap();
    eprintln!("MAL k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_mal_k1_convergence() {
    let cfg = MAGConfig::mal_test_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut initial_loss = None;
    for _ in 0..1_000 {
        let (loss, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        let grads = mal_backward(&params, &cfg, &cache, &input_ids, &target_ids);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let final_loss = mal_forward(&params, &cfg, &input_ids, &target_ids).0;
    let initial = initial_loss.unwrap();
    eprintln!("MAL k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify that attention Q,K,V are computed from memory output m_t, not raw embedded.
/// In MAL, memory preprocesses the input — the cache should show m_t differs from embedded.
#[test]
fn test_mal_memory_is_attention_input() {
    let cfg = MAGConfig::mal_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);

    // m_t (memory output) should differ from embedded (raw input)
    let diff: f32 = cache.embedded.iter().zip(cache.m_t.iter())
        .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();

    eprintln!("MAL memory-is-input: ||embedded - m_t|| = {diff:.4e}");
    assert!(diff > 1e-6,
        "m_t should differ from embedded (memory should transform input): diff={diff:.4e}");

    // Also verify m_t is non-zero (memory is producing output)
    let m_t_norm: f32 = cache.m_t.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(m_t_norm > 1e-6, "m_t should be non-zero: norm={m_t_norm:.4e}");
}

/// CMS k=2 with MAL on multi-scale data.
#[test]
fn test_mal_k2_multiscale() {
    let cfg = MAGConfig::mal_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_mal_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("MAL k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Compare MAL vs MAG on same data. Both should converge, within tolerance.
#[test]
fn test_mal_vs_mag() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };

    let cfg_mag = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
    };
    let cfg_mal = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAL,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
    };

    let input_ids: Vec<usize> = (0..swa.seq_len).map(|t| t % swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=swa.seq_len).map(|t| t % swa.vocab_size).collect();
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

    // Train MAL
    let mut params_mal = MAGParams::init(&cfg_mal, 42);
    let mut mal_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mal_forward(&params_mal, &cfg_mal, &input_ids, &target_ids);
        if mal_initial.is_none() { mal_initial = Some(loss); }
        let grads = mal_backward(&params_mal, &cfg_mal, &cache, &input_ids, &target_ids);
        params_mal.apply_weight_gradients(&grads, lr);
    }
    let mal_final = mal_forward(&params_mal, &cfg_mal, &input_ids, &target_ids).0;

    eprintln!("MAG: initial={:.4}, final={mag_final:.4}", mag_initial.unwrap());
    eprintln!("MAL: initial={:.4}, final={mal_final:.4}", mal_initial.unwrap());

    // Both should converge
    assert!(mag_final < mag_initial.unwrap(), "MAG should converge");
    assert!(mal_final < mal_initial.unwrap(), "MAL should converge");

    // MAL should be in same ballpark as MAG (different composition, not necessarily better at d=8)
    assert!(mal_final < mag_final * 5.0,
        "MAL should not catastrophically regress: mal={mal_final:.6}, mag={mag_final:.6}");

    let ratio = if mal_final > mag_final {
        mal_final / mag_final
    } else {
        mag_final / mal_final
    };
    eprintln!("Performance ratio: {ratio:.2}x (closer to 1.0 = more similar)");
}
