//! Trellis two-pass KV compression integration tests: multi-step training,
//! two-pass state evolution, CMS k=2, comparison vs Delta Rule.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::retention::RetentionKind;
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
    assert!(num_regimes > 0, "num_regimes must be > 0");
    let tokens_per_regime = vocab_size / num_regimes;
    assert!(tokens_per_regime > 0, "vocab_size must be >= num_regimes");
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
    assert!(steps > 0, "cms_train requires steps > 0");
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let d = cfg.swa.d_model;
    let mem_size = 2 * cfg.d_compress * d;
    let mut context = ContextState::new_with_memory_size(cfg.k, d, mem_size);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
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

// ── Trellis k=1 tests ──────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_trellis_k1_smoke() {
    let cfg = MAGConfig::trellis_test_config();
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

    let final_loss = {
        let (loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        loss
    };
    let initial = prev_loss.unwrap();
    eprintln!("Trellis k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_trellis_k1_convergence() {
    let cfg = MAGConfig::trellis_test_config();
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

    let final_loss = {
        let (loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        loss
    };
    let initial = initial_loss.unwrap();
    eprintln!("Trellis k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify both S_K and S_V state matrices evolve during forward pass.
/// This is the core property of two-pass compression — both passes must be active.
#[test]
fn test_trellis_two_pass_states_evolve() {
    let cfg = MAGConfig::trellis_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    match &cache.memory_cache {
        MemoryCache::Trellis(tc) => {
            let d = tc.d;
            let d_k = tc.d_k;
            let s = tc.seq_len;

            // Check S_K evolved: compare initial (t=0) vs final (t=s)
            let sk_initial = &tc.sk_states[0..d_k * d];
            let sk_final = &tc.sk_states[s * d_k * d..(s + 1) * d_k * d];
            let sk_diff: f32 = sk_initial.iter().zip(sk_final.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            assert!(sk_diff > 1e-6,
                "S_K should evolve: ||S_K_0 - S_K_final|| = {sk_diff:.4e}");

            // Check S_V evolved: compare initial (t=0) vs final (t=s)
            let sv_initial = &tc.sv_states[0..d * d_k];
            let sv_final = &tc.sv_states[s * d * d_k..(s + 1) * d * d_k];
            let sv_diff: f32 = sv_initial.iter().zip(sv_final.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            assert!(sv_diff > 1e-6,
                "S_V should evolve: ||S_V_0 - S_V_final|| = {sv_diff:.4e}");

            eprintln!("Two-pass states evolve: ||dS_K||={sk_diff:.4e}, ||dS_V||={sv_diff:.4e}");
        }
        _ => panic!("Expected TrellisCache"),
    }
}

/// CMS k=2 with Trellis on multi-scale data.
#[test]
fn test_trellis_k2_multiscale() {
    let cfg = MAGConfig::trellis_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("Trellis k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Compare Trellis vs Delta Rule on same data.
/// Trellis should converge — within tolerance of Delta Rule.
#[test]
fn test_trellis_vs_delta() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };

    let cfg_delta = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0,
        d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
    };
    let cfg_trellis = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::Trellis,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0,
        d_compress: 8, lambda_k: 0.01, lambda_v: 0.01,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
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

    // Train Trellis
    let mut params_trellis = MAGParams::init(&cfg_trellis, 42);
    let mut trellis_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_trellis, &cfg_trellis, &input_ids, &target_ids);
        if trellis_initial.is_none() { trellis_initial = Some(loss); }
        let grads = mag_backward(&params_trellis, &cfg_trellis, &cache, &input_ids, &target_ids);
        params_trellis.apply_weight_gradients(&grads, lr);
    }
    let trellis_final = mag_forward(&params_trellis, &cfg_trellis, &input_ids, &target_ids).0;

    eprintln!("Delta: initial={:.4}, final={delta_final:.4}", delta_initial.unwrap());
    eprintln!("Trellis: initial={:.4}, final={trellis_final:.4}", trellis_initial.unwrap());

    // Both should converge
    assert!(delta_final < delta_initial.unwrap(), "Delta should converge");
    assert!(trellis_final < trellis_initial.unwrap(), "Trellis should converge");

    // Trellis has two d_k×d matrices at d_k=d=8, so similar capacity — within 5x of Delta
    assert!(trellis_final < delta_final * 5.0,
        "Trellis should not catastrophically regress: trellis={trellis_final:.6}, delta={delta_final:.6}");

    if trellis_final > delta_final {
        let ratio = trellis_final / delta_final;
        eprintln!("NOTE: Delta beats Trellis by {ratio:.2}x — two-pass overhead at tiny scale");
    } else {
        eprintln!("Trellis matches/beats Delta at d=8 — compression advantage even at small scale");
    }
}
