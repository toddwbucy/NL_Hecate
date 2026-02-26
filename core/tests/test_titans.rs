//! Titans LMM integration tests: multi-step training, momentum validation, comparison vs Delta Rule.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind, HopeVariant, LatticeVariant, MomentumKind, ProjectionKind};
use nl_hecate_core::dynamic_freq::FrequencySchedule;
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

// ── Titans k=1 tests ─────────────────────────────────────────────────

/// 100-step smoke test: no NaN, no divergence, loss finite.
#[test]
fn test_titans_k1_smoke() {
    let cfg = MAGConfig::titans_test_config();
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
    eprintln!("Titans k=1 smoke: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < 100.0, "Loss diverged: {final_loss}");
}

/// 1K-step convergence: loss decreases.
#[test]
fn test_titans_k1_convergence() {
    let cfg = MAGConfig::titans_test_config();
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
    eprintln!("Titans k=1 convergence: initial={initial:.4}, final={final_loss:.4}");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify momentum state S is non-trivial after forward pass.
#[test]
fn test_titans_momentum_nonzero() {
    let cfg = MAGConfig::titans_test_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let (_, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);

    // Extract the TitansLMMCache from MemoryCache
    match &cache.memory_cache {
        MemoryCache::Titans(tc) => {
            let d = tc.d;
            let seq = tc.seq_len;
            // Check final S state (at index seq_len * d * d)
            let s_final = &tc.s_states[seq * d * d..(seq + 1) * d * d];
            let s_norm: f32 = s_final.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!("Final S norm: {s_norm:.6e}");
            assert!(s_norm > 1e-6, "Momentum S should be non-trivial, got {s_norm:.4e}");
        }
        MemoryCache::Delta(_) => {
            panic!("Expected TitansLMMCache, got DeltaRuleCache");
        }
        MemoryCache::Hebbian(_) => {
            panic!("Expected TitansLMMCache, got HebbianCache");
        }
        MemoryCache::Moneta(_) => {
            panic!("Expected TitansLMMCache, got MonetaCache");
        }
        MemoryCache::YAAD(_) => {
            panic!("Expected TitansLMMCache, got YAADCache");
        }
        MemoryCache::MEMORA(_) => {
            panic!("Expected TitansLMMCache, got MEMORACache");
        }
        MemoryCache::Lattice(_) => {
            panic!("Expected TitansLMMCache, got LatticeCache");
        }
        MemoryCache::Trellis(_) => {
            panic!("Expected TitansLMMCache, got TrellisCache");
        }
        MemoryCache::Atlas(_) => {
            panic!("Expected TitansLMMCache, got AtlasOmegaCache");
        }
        MemoryCache::SwiGlu(_) => {
            panic!("Expected TitansLMMCache, got SwiGluMlpCache");
        }
        MemoryCache::SelfRef(_) => {
            panic!("Expected TitansLMMCache, got SelfRefCache");
        }
        MemoryCache::ChunkwiseSelfRef(_) => {
            panic!("Expected TitansLMMCache, got ChunkwiseSelfRefCache");
        }
    }
}

/// Compare Titans vs Delta Rule on same data.
/// Titans should match or beat Delta Rule (soft criterion — same scale, same lr).
#[test]
fn test_titans_vs_delta() {
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
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: Default::default(),
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
        intermediate_size: 0,
        m_norm_max: vec![],
    };
    let cfg_titans = MAGConfig {
        swa: swa.clone(), memory_enabled: true,
        memory_rule: MemoryRuleKind::TitansLMM,
        k: 1, chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            composition: CompositionKind::MAG,
        parallel: None,
        retention: RetentionKind::L2WeightDecay,
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: Default::default(),
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
        intermediate_size: 0,
        m_norm_max: vec![],
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

    // Train Titans LMM
    let mut params_titans = MAGParams::init(&cfg_titans, 42);
    let mut titans_initial = None;
    for _ in 0..steps {
        let (loss, cache) = mag_forward(&params_titans, &cfg_titans, &input_ids, &target_ids);
        if titans_initial.is_none() { titans_initial = Some(loss); }
        let grads = mag_backward(&params_titans, &cfg_titans, &cache, &input_ids, &target_ids);
        params_titans.apply_weight_gradients(&grads, lr);
    }
    let titans_final = mag_forward(&params_titans, &cfg_titans, &input_ids, &target_ids).0;

    eprintln!("Delta: initial={:.4}, final={delta_final:.4}", delta_initial.unwrap());
    eprintln!("Titans: initial={:.4}, final={titans_final:.4}", titans_initial.unwrap());

    // Both should converge
    assert!(delta_final < delta_initial.unwrap(), "Delta should converge");
    assert!(titans_final < titans_initial.unwrap(), "Titans should converge");

    // Soft criterion: Titans should match or beat Delta (within 10% margin)
    let margin = 0.10;
    assert!(titans_final < delta_final * (1.0 + margin),
        "Titans should not regress vs Delta: titans={titans_final:.6}, delta={delta_final:.6}");

    if titans_final < delta_final {
        let improvement = (delta_final - titans_final) / delta_final * 100.0;
        eprintln!("Titans beats Delta by {improvement:.2}%");
    } else {
        eprintln!("NOTE: Titans matches Delta at d=8 scale — expected for tiny model");
    }
}

/// CMS k=2 with Titans on multi-scale data.
#[test]
fn test_titans_k2_multiscale() {
    let cfg = MAGConfig::titans_test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_multiscale_data(
        cfg.swa.seq_len, cfg.swa.vocab_size, 4, 4,
    );

    let (initial, final_loss) = cms_train(&mut params, &cfg, &input_ids, &target_ids, 5_000, 0.01);
    eprintln!("Titans k=2 multiscale: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}");
}

/// Verify M-norm clamping: adversarial large-magnitude inputs must not push ‖M‖_F above m_norm_max.
///
/// Uses a config with m_norm_max=5.0 on a d=16 model. Every stored M_t slice (t≥1)
/// must satisfy ‖M_t‖_F ≤ 5.0 + ε. Verifies the straight-through CPU path in TitansLMM.
#[test]
fn test_titans_m_norm_clamp_cpu() {
    use nl_hecate_core::model::{SWAConfig, AttentionalBias};
    use nl_hecate_core::retention::default_retention;

    let d: usize = 16;
    let seq_len: usize = 32;
    // Cap must be below the unclamped run's max ‖M‖ so the constraint is live.
    // A preliminary control run (below) confirms the unclamped max is ~0.0026 at
    // this d/seq_len, so we set the cap to 0.001 (< unclamped) to guarantee
    // the clamp fires and is not vacuous.
    let m_norm_max_val = 0.001_f32;

    let cfg = MAGConfig {
        swa: SWAConfig {
            d_model: d, num_heads: 2, head_dim: d / 2,
            seq_len, window_size: seq_len, vocab_size: 256,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: MemoryRuleKind::TitansLMM,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0,
        lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0,
        lambda_k: 0.0, lambda_v: 0.0,
        parallel: None,
        retention: default_retention(MemoryRuleKind::TitansLMM),
        m3: None,
        frequency_schedule: FrequencySchedule::Fixed,
        checkpoint_interval: None,
        hope_variant: HopeVariant::FreqGated,
        lattice_variant: LatticeVariant::Decode,
        n_persistent: 0,
        attentional_bias: AttentionalBias::L2,
        kernel_size: 0,
        momentum_kind: MomentumKind::None,
        momentum_d_hidden: 0,
        projection_kind: ProjectionKind::Static,
        self_generated_values: false,
        self_ref_chunk_size: 1,
        theta_floor: vec![],
        theta_ceil: vec![],
        intermediate_size: 0,
        m_norm_max: vec![m_norm_max_val],
    };

    let params = MAGParams::init(&cfg, 42);
    let input_ids: Vec<usize> = (0..seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=seq_len).map(|t| t % cfg.swa.vocab_size).collect();

    let (loss, cache_outer) = mag_forward(&params, &cfg, &input_ids, &target_ids);
    assert!(loss.is_finite(), "Loss should be finite with m_norm_max clamping");

    // Extract TitansLMM cache and verify per-step M-state norms
    let titans_cache = match &cache_outer.memory_cache {
        MemoryCache::Titans(c) => c,
        _ => panic!("Expected Titans cache"),
    };

    let dd = d * d;
    assert_eq!(titans_cache.m_states.len(), (seq_len + 1) * dd);

    let eps = 1e-4_f32;
    for t in 1..=seq_len {  // M_0 is the initial state (before any clamping)
        let slice = &titans_cache.m_states[t * dd..(t + 1) * dd];
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            norm <= m_norm_max_val + eps,
            "‖M_{t}‖_F = {norm:.6} exceeds m_norm_max={m_norm_max_val} at t={t}"
        );
    }

    eprintln!("test_titans_m_norm_clamp_cpu: loss={loss:.4}, all M norms ≤ {m_norm_max_val} ✓");

    // Control: same inputs with clamp disabled (m_norm_max=[]) must push ‖M‖_F
    // above the cap, proving the earlier assertion tested a live constraint.
    let cfg_no_clamp = MAGConfig {
        m_norm_max: vec![],  // disabled
        ..cfg.clone()
    };
    let params_nc = MAGParams::init(&cfg_no_clamp, 42);
    let (_, cache_nc) = mag_forward(&params_nc, &cfg_no_clamp, &input_ids, &target_ids);
    let titans_nc = match &cache_nc.memory_cache {
        MemoryCache::Titans(c) => c,
        _ => panic!("Expected Titans cache in control run"),
    };
    let max_unclamped = (1..=seq_len)
        .map(|t| {
            let slice = &titans_nc.m_states[t * dd..(t + 1) * dd];
            slice.iter().map(|x| x * x).sum::<f32>().sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_unclamped > m_norm_max_val,
        "Control run (no clamp) should exceed m_norm_max={m_norm_max_val}, \
         but max ‖M‖={max_unclamped:.6}. The clamp test may be vacuous."
    );
    eprintln!("Control (no clamp): max ‖M‖={max_unclamped:.6} > {m_norm_max_val} ✓ (clamp was live)");
}
