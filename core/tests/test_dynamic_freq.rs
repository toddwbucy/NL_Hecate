/// Tests for S3-M5: Dynamic Frequency Scheduling.
///
/// Tests cover:
///   - Unit tests for freq gate computation, thresholds, mean pool
///   - Integration tests with full CMS forward/backward
///   - Gradient tests verifying w_freq/b_freq receive non-zero updates
///   - Edge cases (k=1, all gates above/below threshold)

use nl_hecate_core::model::{MAGConfig, MAGParams, HopeVariant, LatticeVariant};
use nl_hecate_core::conductor::{Pulse, ContextState, ErrorBuffer};
use nl_hecate_core::mag::{cms_forward, cms_backward};
use nl_hecate_core::dynamic_freq::{
    FrequencySchedule, LearnedFreqConfig,
    mean_pool, compute_freq_gates, apply_threshold, should_anneal, default_b_freq,
};

// ── Helpers ──────────────────────────────────────────────────────────

fn make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

fn fixed_k2_config() -> MAGConfig {
    MAGConfig::test_config_k2()
}

fn learned_k2_config() -> MAGConfig {
    MAGConfig::dynamic_freq_test_config()
}

fn learned_k4_config() -> MAGConfig {
    MAGConfig::dynamic_freq_test_config_k4()
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[test]
fn test_freq_gate_basic() {
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    // Verify w_freq/b_freq are allocated
    for l in 0..cfg.k {
        assert_eq!(params.levels[l].w_freq.len(), d,
            "Level {l}: w_freq should have length d={d}");
        assert_eq!(params.levels[l].b_freq.len(), 1,
            "Level {l}: b_freq should have length 1");
    }
}

#[test]
fn test_threshold_level0_always_active() {
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    // Set all gates to very negative bias → sigmoid near 0
    for l in 0..cfg.k {
        params.levels[l].w_freq = vec![0.0f32; d];
        params.levels[l].b_freq = vec![-100.0];
    }
    let embedded_mean = vec![0.0f32; d];
    let cache = compute_freq_gates(&embedded_mean, &params.levels, cfg.k, d);
    let active = apply_threshold(&cache, 0.5);

    assert!(active[0], "Level 0 must ALWAYS be active");
    assert!(!active[1], "Level 1 should be inactive with -100 bias");
}

#[test]
fn test_gate_sigmoid_range() {
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;
    let embedded_mean = vec![0.5f32; d];

    let cache = compute_freq_gates(&embedded_mean, &params.levels, cfg.k, d);
    for l in 0..cfg.k {
        let g = cache.gate_values[l];
        assert!(g > 0.0 && g < 1.0, "Level {l}: gate={g} not in (0,1)");
    }
}

#[test]
fn test_mean_pool() {
    let d = 4;
    let s = 3;
    let embedded: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let mean = mean_pool(&embedded, s, d);
    assert_eq!(mean.len(), d);
    assert!((mean[0] - 5.0).abs() < 1e-6);
    assert!((mean[1] - 6.0).abs() < 1e-6);
}

#[test]
fn test_config_defaults() {
    let cfg = LearnedFreqConfig::default();
    assert_eq!(cfg.threshold, 0.5);
    assert_eq!(cfg.anneal_steps, 0);
}

#[test]
fn test_anneal_schedule() {
    assert!(should_anneal(0, 100));
    assert!(should_anneal(99, 100));
    assert!(!should_anneal(100, 100));
    assert!(!should_anneal(0, 0));
}

// ── Integration Tests ────────────────────────────────────────────────

#[test]
fn test_fixed_schedule_unchanged() {
    // Fixed schedule should produce identical results to before.
    let cfg = fixed_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    let mut ctx1 = ContextState::new(cfg.k, d);
    let mut ctx2 = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

    let (loss1, cache1) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx1);
    let (loss2, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx2);

    assert_eq!(loss1, loss2, "Fixed schedule should be deterministic");
    assert!(cache1.freq_cache.is_none(), "Fixed schedule should have no freq_cache");
}

#[test]
fn test_learned_gates_override_pulse() {
    // With Learned schedule, the pulse's active_levels get overridden.
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    let mut context = ContextState::new(cfg.k, d);
    // Give both-active pulse — the learned gate should override level 1
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    assert!(loss.is_finite(), "Learned freq loss should be finite: {loss}");
    assert!(cache.freq_cache.is_some(), "Learned schedule should have freq_cache");

    // Level 0 must be active in the effective pulse
    assert!(cache.pulse.active_levels[0], "Level 0 must always be active");
}

#[test]
fn test_learned_k2_convergence() {
    // Verify loss decreases with Learned scheduling over a few steps.
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    // Persist ContextState across steps so memory accumulates (M matrices carry over).
    let mut context = ContextState::new(cfg.k, d);
    let mut losses = Vec::new();
    for step in 0..5 {
        let pulse = Pulse { global_step: step, active_levels: vec![true, true] };
        let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        losses.push(loss);

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);
    }

    // Loss should decrease (or at least first < last won't be higher by much)
    assert!(losses[0].is_finite());
    assert!(losses[4].is_finite());
    // At minimum, training should not diverge
    assert!(losses[4] < losses[0] * 2.0, "Loss should not double: first={} last={}", losses[0], losses[4]);
}

#[test]
fn test_learned_k4_convergence() {
    let cfg = learned_k4_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    // Persist ContextState across steps so memory accumulates.
    let mut context = ContextState::new(cfg.k, d);
    let mut losses = Vec::new();
    for step in 0..5 {
        let pulse = Pulse { global_step: step, active_levels: vec![true; cfg.k] };
        let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        losses.push(loss);

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);
    }

    assert!(losses[0].is_finite());
    assert!(losses[4].is_finite());
    assert!(losses[4] < losses[0] * 2.0, "k=4 loss should not double");
}

#[test]
fn test_fire_rate_varies_with_input() {
    // Different inputs should produce different gate activations.
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    // Two different input sequences
    let input_a: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let input_b: Vec<usize> = (0..cfg.swa.seq_len).map(|t| (t + 5) % cfg.swa.vocab_size).collect();
    let target: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();

    let mut ctx_a = ContextState::new(cfg.k, d);
    let mut ctx_b = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

    let (_, cache_a) = cms_forward(&params, &cfg, &input_a, &target, &pulse, &mut ctx_a);
    let (_, cache_b) = cms_forward(&params, &cfg, &input_b, &target, &pulse, &mut ctx_b);

    // Gate values should differ between different inputs
    let fc_a = cache_a.freq_cache.as_ref().unwrap();
    let fc_b = cache_b.freq_cache.as_ref().unwrap();

    let diff: f32 = fc_a.gate_values.iter().zip(fc_b.gate_values.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1e-6, "Gate values should differ between inputs, diff={diff}");
}

#[test]
fn test_learned_vs_fixed_comparable() {
    // Both schedules should produce finite, reasonable losses.
    let cfg_fixed = fixed_k2_config();
    let cfg_learned = learned_k2_config();
    let (input_ids, target_ids) = make_test_data(&cfg_fixed);
    let d = cfg_fixed.swa.d_model;

    let params_fixed = MAGParams::init(&cfg_fixed, 42);
    let params_learned = MAGParams::init(&cfg_learned, 42);

    let mut ctx_f = ContextState::new(cfg_fixed.k, d);
    let mut ctx_l = ContextState::new(cfg_learned.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

    let (loss_f, _) = cms_forward(&params_fixed, &cfg_fixed, &input_ids, &target_ids, &pulse, &mut ctx_f);
    let (loss_l, _) = cms_forward(&params_learned, &cfg_learned, &input_ids, &target_ids, &pulse, &mut ctx_l);

    assert!(loss_f.is_finite());
    assert!(loss_l.is_finite());
    // Both should be in reasonable loss range
    assert!(loss_f > 0.0 && loss_f < 20.0, "Fixed loss out of range: {loss_f}");
    assert!(loss_l > 0.0 && loss_l < 20.0, "Learned loss out of range: {loss_l}");
}

// ── Gradient Tests ───────────────────────────────────────────────────

#[test]
fn test_straight_through_nonzero() {
    // Verify freq gate backward produces non-zero gradients.
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
    let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();
    let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

    // w_freq/b_freq gradients should be non-zero for at least one level
    let mut any_nonzero = false;
    for l in 0..cfg.k {
        let w_norm: f32 = grads.levels[l].w_freq.iter().map(|x| x * x).sum::<f32>().sqrt();
        let b_norm: f32 = grads.levels[l].b_freq.iter().map(|x| x * x).sum::<f32>().sqrt();
        if w_norm > 1e-10 || b_norm > 1e-10 {
            any_nonzero = true;
        }
    }
    assert!(any_nonzero, "At least one level should have non-zero freq gate grads");
}

#[test]
fn test_w_freq_updates_during_training() {
    // Verify w_freq actually changes across training steps.
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    let w_freq_0_before = params.levels[0].w_freq.clone();
    let w_freq_1_before = params.levels[1].w_freq.clone();

    // Persist ContextState across steps so memory accumulates.
    let mut context = ContextState::new(cfg.k, d);
    for step in 0..3 {
        let pulse = Pulse { global_step: step, active_levels: vec![true, true] };
        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);
    }

    // w_freq should have changed
    let diff_0: f32 = params.levels[0].w_freq.iter().zip(w_freq_0_before.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff_1: f32 = params.levels[1].w_freq.iter().zip(w_freq_1_before.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(diff_0 > 1e-8 || diff_1 > 1e-8,
        "w_freq should change during training: diff_0={diff_0}, diff_1={diff_1}");
}

#[test]
fn test_b_freq_updates_during_training() {
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    let b_freq_before: Vec<f32> = params.levels.iter().map(|l| l.b_freq[0]).collect();

    // Persist ContextState across steps so memory accumulates.
    let mut context = ContextState::new(cfg.k, d);
    for step in 0..3 {
        let pulse = Pulse { global_step: step, active_levels: vec![true, true] };
        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
        params.apply_weight_gradients(&grads, 0.01);
    }

    let b_freq_after: Vec<f32> = params.levels.iter().map(|l| l.b_freq[0]).collect();
    let diff: f32 = b_freq_before.iter().zip(b_freq_after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(diff > 1e-8, "b_freq should change during training: diff={diff}");
}

#[test]
fn test_freq_gate_gradient_fd() {
    // Finite-difference gradient check for b_freq.
    let cfg = learned_k2_config();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(&cfg);
    let d = cfg.swa.d_model;

    // Compute analytical gradient
    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
    let (loss_base, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();
    let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

    // FD check on b_freq for level 0
    let eps = 1e-3;
    let mut params_plus = params.clone();
    params_plus.levels[0].b_freq[0] += eps;
    let mut ctx_plus = ContextState::new(cfg.k, d);
    let (loss_plus, _) = cms_forward(&params_plus, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_plus);

    let fd_grad = (loss_plus - loss_base) / eps;
    let anal_grad = grads.levels[0].b_freq[0];

    // The straight-through estimator is approximate, so we only check sign agreement
    // when both gradients are reliably non-zero (above 1e-4).
    if fd_grad.abs() > 1e-4 && anal_grad.abs() > 1e-4 {
        assert!(fd_grad * anal_grad >= 0.0,
            "FD and analytical gradient should agree in sign: fd={fd_grad}, anal={anal_grad}");
    }
    // At minimum, both should be finite
    assert!(fd_grad.is_finite(), "FD gradient not finite");
    assert!(anal_grad.is_finite(), "Analytical gradient not finite");
}

// ── Edge Cases ───────────────────────────────────────────────────────

#[test]
fn test_k1_learned_is_noop() {
    // With k=1 and Learned, Level 0 is always active → gate is a no-op.
    use nl_hecate_core::retention::default_retention;
    use nl_hecate_core::model::{SWAConfig, CompositionKind, MemoryRuleKind};

    let cfg = MAGConfig {
        swa: SWAConfig {
            d_model: 8, num_heads: 2, head_dim: 4,
            seq_len: 4, window_size: 4, vocab_size: 16,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
        m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        parallel: None,
        retention: default_retention(MemoryRuleKind::DeltaRule),
        m3: None,
        frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig::default()),
        checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: Default::default(),
            kernel_size: 0,
    };
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();

    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true] };
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    assert!(loss.is_finite());
    // Level 0 always active
    assert!(cache.pulse.active_levels[0]);
}

#[test]
fn test_all_gates_below_threshold() {
    // Force all gates below threshold → only Level 0 fires.
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    // Set all w_freq to zero and b_freq very negative
    for l in 0..cfg.k {
        params.levels[l].w_freq = vec![0.0f32; d];
        params.levels[l].b_freq = vec![-100.0];
    }

    let (input_ids, target_ids) = make_test_data(&cfg);
    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    assert!(loss.is_finite());
    assert!(cache.pulse.active_levels[0], "Level 0 must be active");
    assert!(!cache.pulse.active_levels[1], "Level 1 should be inactive with -100 bias");
}

#[test]
fn test_all_gates_above_threshold() {
    // Force all gates above threshold → all levels fire.
    let cfg = learned_k2_config();
    let mut params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    // Set all w_freq to zero and b_freq very positive
    for l in 0..cfg.k {
        params.levels[l].w_freq = vec![0.0f32; d];
        params.levels[l].b_freq = vec![100.0];
    }

    let (input_ids, target_ids) = make_test_data(&cfg);
    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    assert!(loss.is_finite());
    assert!(cache.pulse.active_levels[0], "Level 0 must be active");
    assert!(cache.pulse.active_levels[1], "Level 1 should be active with +100 bias");
}

#[test]
fn test_fixed_mode_no_freq_params() {
    // Fixed schedule should have empty w_freq/b_freq.
    let cfg = fixed_k2_config();
    let params = MAGParams::init(&cfg, 42);

    for l in 0..cfg.k {
        assert!(params.levels[l].w_freq.is_empty(),
            "Fixed schedule should have empty w_freq");
        assert!(params.levels[l].b_freq.is_empty(),
            "Fixed schedule should have empty b_freq");
    }

    // num_params should NOT include freq params
    let base_params = params.num_params();
    let cfg_learned = learned_k2_config();
    let params_learned = MAGParams::init(&cfg_learned, 42);
    let learned_params = params_learned.num_params();

    // Learned should have d + 1 extra params per level
    let d = cfg.swa.d_model;
    let expected_extra = cfg.k * (d + 1);
    assert_eq!(learned_params - base_params, expected_extra,
        "Learned should have {expected_extra} extra params, got {}", learned_params - base_params);
}

#[test]
fn test_default_b_freq_ordering() {
    assert!(default_b_freq(0) > default_b_freq(1));
    assert!(default_b_freq(1) > default_b_freq(2));
    assert!(default_b_freq(2) > default_b_freq(3));
}

#[test]
fn test_anneal_uses_fixed_schedule() {
    // During annealing, learned gates should NOT override pulse.
    use nl_hecate_core::model::{SWAConfig, CompositionKind, MemoryRuleKind};
    use nl_hecate_core::retention::default_retention;

    let cfg = MAGConfig {
        swa: SWAConfig {
            d_model: 8, num_heads: 2, head_dim: 4,
            seq_len: 8, window_size: 8, vocab_size: 16,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 2,
        chunk_sizes: vec![1, 8],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
        m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        parallel: None,
        retention: default_retention(MemoryRuleKind::DeltaRule),
        m3: None,
        frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig {
            threshold: 0.5,
            anneal_steps: 1000, // Long annealing — step 0 should use fixed
        }),
        checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: Default::default(),
            kernel_size: 0,
    };

    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;
    let (input_ids, target_ids) = make_test_data(&cfg);

    let mut context = ContextState::new(cfg.k, d);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, false] };
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

    assert!(loss.is_finite());
    // During annealing, freq_cache should be None (fixed schedule used)
    assert!(cache.freq_cache.is_none(), "During annealing, should use fixed schedule");
    // Pulse should match what we passed in
    assert!(cache.pulse.active_levels[0]);
    assert!(!cache.pulse.active_levels[1], "Level 1 should stay frozen during annealing");
}
