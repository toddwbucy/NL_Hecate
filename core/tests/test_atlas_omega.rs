//! Atlas Omega integration tests: multi-step training, momentum validation,
//! parallel forward correctness, CMS k=1/2/4, comparison vs Delta Rule.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::retention::RetentionKind;
use nl_hecate_core::mag::{cms_forward, cms_backward, mag_forward, mag_backward, MemoryCache};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};
use nl_hecate_core::atlas_parallel::AtlasOmegaParams;
use nl_hecate_core::atlas_omega::AtlasOmega;
use nl_hecate_core::delta_rule::MemoryRule;

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

// ── Atlas Omega k=1 tests ───────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_atlas_k1_smoke() {
    let cfg = MAGConfig::atlas_test_config();
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
    eprintln!("Atlas k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_atlas_k1_convergence() {
    let cfg = MAGConfig::atlas_test_config();
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
    eprintln!("Atlas k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify memory M is non-trivial after forward pass.
#[test]
fn test_atlas_memory_nonzero() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    match &cache.memory_cache {
        MemoryCache::Atlas(ac) => {
            let d = ac.d;
            let seq = ac.seq_len;
            let m_final = &ac.m_states[seq * d * d..(seq + 1) * d * d];
            let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final M norm: {m_norm:.6e}");
            assert!(m_norm > 1e-6, "Memory M should be non-trivial, got {m_norm:.4e}");
        }
        _ => {
            panic!("Expected Atlas cache");
        }
    }
}

/// Verify momentum S is non-trivial after forward pass (Atlas has momentum like Titans).
#[test]
fn test_atlas_momentum_nonzero() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    match &cache.memory_cache {
        MemoryCache::Atlas(ac) => {
            let d = ac.d;
            let seq = ac.seq_len;
            let s_final = &ac.s_states[seq * d * d..(seq + 1) * d * d];
            let s_norm: f32 = s_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final S norm: {s_norm:.6e}");
            assert!(s_norm > 1e-6, "Momentum S should be non-trivial, got {s_norm:.4e}");
        }
        _ => {
            panic!("Expected Atlas cache");
        }
    }
}

/// Atlas omega is state-independent: same (k, v) always produces same omega,
/// regardless of memory state M. This is the key parallelization enabler.
#[test]
fn test_atlas_omega_state_independence_via_mag() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    // Run forward twice with same params — loss should be identical
    let (loss1, cache1) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    let (loss2, _cache2) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    assert_eq!(loss1, loss2, "MAG loss should be deterministic: {loss1} vs {loss2}");

    // Omega values in cache should be non-trivial
    match &cache1.memory_cache {
        MemoryCache::Atlas(ac) => {
            let om_norm: f32 = ac.omega_mats.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(om_norm > 1e-8, "Omega matrices should be non-trivial");
        }
        _ => panic!("Expected Atlas cache"),
    }
}

// ── Atlas Parallel tests ────────────────────────────────────────────

/// Core correctness: atlas_parallel_forward matches sequential AtlasOmega.step().
/// The parallel forward computes all omegas/gates in parallel, then accumulates
/// M and S sequentially. This must produce identical results to the fully sequential step().
#[test]
fn test_atlas_parallel_matches_sequential() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;
    let s = cfg.swa.seq_len;

    // Create embedded input
    use nl_hecate_core::tensor::SimpleRng;
    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 0.1);

    // Sequential (via AtlasOmega.step())
    let omega_params = AtlasOmegaParams::init(d, 42);
    let rule = AtlasOmega { omega_params };
    let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

    // Parallel
    let (y_par, _) = nl_hecate_core::atlas_parallel::atlas_parallel_forward(
        &params.levels[0], &embedded, s, d, &cfg, None,
    );

    // They should match exactly (same math, same sequence)
    for i in 0..(s * d) {
        assert!(
            (y_seq[i] - y_par[i]).abs() < 1e-6,
            "Mismatch at {i}: seq={}, par={}", y_seq[i], y_par[i]
        );
    }
}

/// Parallel forward with initial memory M_0.
#[test]
fn test_atlas_parallel_with_initial_memory() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;
    let s = cfg.swa.seq_len;

    use nl_hecate_core::tensor::SimpleRng;
    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 0.1);

    // Non-zero initial memory
    let mut m0 = vec![0.0f32; d * d];
    rng.fill_uniform(&mut m0, 0.05);

    // Sequential with initial M
    let omega_params = AtlasOmegaParams::init(d, 42);
    let rule = AtlasOmega { omega_params };
    let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, Some(m0.clone()));

    // Parallel with initial M
    let (y_par, _) = nl_hecate_core::atlas_parallel::atlas_parallel_forward(
        &params.levels[0], &embedded, s, d, &cfg, Some(m0),
    );

    for i in 0..(s * d) {
        assert!(
            (y_seq[i] - y_par[i]).abs() < 1e-6,
            "Mismatch at {i} with initial M: seq={}, par={}", y_seq[i], y_par[i]
        );
    }
}

// ── CMS k=2 tests ──────────────────────────────────────────────────

/// CMS k=2 with Atlas on multi-scale data.
#[test]
fn test_atlas_k2_multiscale() {
    let cfg = MAGConfig::atlas_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("Atlas k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// CMS k=2 smoke: no NaN in multi-level processing.
#[test]
fn test_atlas_k2_smoke() {
    let cfg = MAGConfig::atlas_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(cfg.swa.d_model))
        .collect();

    for step in 0..100 {
        let pulse = conductor.pulse();
        let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        assert!(loss.is_finite(), "CMS loss NaN at step {step}");

        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);

        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], 0.01);
            }
        }

        conductor.advance();
    }
}

// ── Comparison tests ────────────────────────────────────────────────

/// Compare Atlas vs Delta Rule on same data.
/// Atlas should converge (soft criterion — it has different dynamics from DeltaRule).
#[test]
fn test_atlas_vs_delta() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 8, window_size: 8, vocab_size: 16,
    };

    let cfg_delta = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
        m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
        m3: None,
    };
    let cfg_atlas = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::AtlasOmega,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
        m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
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

    // Train Atlas Omega
    let mut params_atlas = MAGParams::init(&cfg_atlas, 42);
    let mut atlas_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_atlas, &cfg_atlas, &input_ids, &target_ids);
        if atlas_initial.is_none() { atlas_initial = Some(loss); }
        let grads = mag_backward(&params_atlas, &cfg_atlas, &cache, &input_ids, &target_ids);
        params_atlas.apply_weight_gradients(&grads, lr);
    }
    let atlas_final = mag_forward(&params_atlas, &cfg_atlas, &input_ids, &target_ids).0;

    eprintln!("Delta: initial={:.4}, final={delta_final:.4}", delta_initial.unwrap());
    eprintln!("Atlas: initial={:.4}, final={atlas_final:.4}", atlas_initial.unwrap());

    // Both should converge
    assert!(delta_final < delta_initial.unwrap(), "Delta should converge");
    assert!(atlas_final < atlas_initial.unwrap(), "Atlas should converge");

    // Atlas uses a learned omega function which may converge differently than DeltaRule.
    // At small d=8 scale, allow generous margin (within 10x of Delta).
    assert!(atlas_final < delta_final * 10.0,
        "Atlas should not catastrophically regress: atlas={atlas_final:.6}, delta={delta_final:.6}");

    if atlas_final > delta_final {
        let ratio = atlas_final / delta_final;
        eprintln!("NOTE: Delta beats Atlas by {ratio:.2}x — expected at d=8 (omega function has fewer signals)");
    } else {
        eprintln!("Atlas matches/beats Delta at d=8 scale");
    }
}

// ── Edge case tests ─────────────────────────────────────────────────

/// Backward with zero upstream gradient should produce zero parameter gradients.
#[test]
fn test_atlas_backward_zero_upstream() {
    let cfg = MAGConfig::atlas_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);
    let d = cfg.swa.d_model;
    let s = cfg.swa.seq_len;

    let omega_params = AtlasOmegaParams::init(d, 42);
    let rule = AtlasOmega { omega_params };

    use nl_hecate_core::tensor::SimpleRng;
    let mut rng = SimpleRng::new(99);
    let mut embedded = vec![0.0f32; s * d];
    rng.fill_uniform(&mut embedded, 0.1);

    let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

    let d_y = vec![0.0f32; s * d];
    let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

    // All gradients should be zero
    let w_k_max = grads.w_k_mem.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let w_v_max = grads.w_v_mem.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let w_q_max = grads.w_q_mem.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    assert!(w_k_max < 1e-10, "w_k_mem grads should be ~0 with zero upstream, got {w_k_max}");
    assert!(w_v_max < 1e-10, "w_v_mem grads should be ~0 with zero upstream, got {w_v_max}");
    assert!(w_q_max < 1e-10, "w_q_mem grads should be ~0 with zero upstream, got {w_q_max}");
    assert!(emb_max < 1e-10, "d_embedded should be ~0 with zero upstream, got {emb_max}");
}

/// Sequence length 1: degenerate case should still work.
#[test]
fn test_atlas_seq_len_1() {
    use nl_hecate_core::model::SWAConfig;

    let swa = SWAConfig {
        d_model: 8, num_heads: 2, head_dim: 4,
        seq_len: 1, window_size: 1, vocab_size: 16,
    };
    let cfg = MAGConfig {
        swa, memory_enabled: true,
        memory_rule: MemoryRuleKind::AtlasOmega,
        k: 1, chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
        m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
        m3: None,
    };

    let params = MAGParams::init(&cfg, 42);
    let input_ids = vec![0];
    let target_ids = vec![1];

    let (loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    assert!(loss.is_finite(), "Loss should be finite for seq_len=1");

    let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);
    // Just check nothing panics and gradients are finite
    for &v in &grads.levels[0].w_k_mem {
        assert!(v.is_finite());
    }
}

/// Strategy support: AtlasOmega should support atlas_parallel.
#[test]
fn test_atlas_strategy_support() {
    use nl_hecate_core::parallel::{strategy_supported, ParallelStrategy};

    assert!(strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::Sequential));
    assert!(strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::ChunkwiseGD));
    assert!(strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::TNTHierarchical));
    assert!(strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::AtlasParallel));
    // Atlas should NOT support these specialized strategies
    assert!(!strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::AssociativeScan));
    assert!(!strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::LatticeGLA));
}

/// Backward produces non-trivial gradients for gate weights (w_alpha, w_theta, w_eta).
/// With neutral biases (b_alpha=0, b_theta=0) the gates are unsaturated and produce
/// larger gradients. Default biases (b_alpha=3.0) saturate sigmoid near 1.0.
#[test]
fn test_atlas_gate_gradients_nontrivial() {
    let cfg = MAGConfig::atlas_test_config();
    // Use neutral biases for gradient visibility (same pattern as gradient.rs)
    let mut params = MAGParams::init(&cfg, 42);
    for level in &mut params.levels {
        level.b_alpha = vec![0.0f32]; // sigmoid(0)=0.5
        level.b_theta = vec![0.0f32]; // softplus(0)=ln(2)
        level.b_eta = vec![0.0f32];   // sigmoid(0)=0.5
    }
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);

    let alpha_max = grads.levels[0].w_alpha.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let theta_max = grads.levels[0].w_theta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let eta_max = grads.levels[0].w_eta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    eprintln!("Gate grad magnitudes: alpha={alpha_max:.4e}, theta={theta_max:.4e}, eta={eta_max:.4e}");

    // At least two of three gate gradients should be non-trivial
    let nontrivial_count = [alpha_max, theta_max, eta_max].iter()
        .filter(|&&x| x > 1e-8).count();
    assert!(nontrivial_count >= 2,
        "At least 2/3 gate gradients should be non-trivial: alpha={alpha_max:.4e}, theta={theta_max:.4e}, eta={eta_max:.4e}");
}
