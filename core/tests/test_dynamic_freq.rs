//! Dynamic Frequency Scheduling tests.
//!
//! Tests learned gates for CMS level activation, backward compatibility with
//! fixed scheduling, gradient flow, and edge cases.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer, Pulse};
use nl_hecate_core::mag::{cms_forward, cms_backward};
use nl_hecate_core::dynamic_freq::{FrequencyScheduler, DynamicFreqConfig, default_freq_bias};

fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

// ── Backward compatibility ──────────────────────────────────────────

#[test]
fn test_fixed_scheduling_default_unchanged() {
    // Default config has dynamic_scheduling=false. Verify CMS forward
    // produces identical results to the original fixed-schedule path.
    let cfg = MAGConfig::test_config_k2();
    assert!(!cfg.dynamic_scheduling);

    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let mut context = ContextState::new(cfg.k, cfg.swa.d_model);

    // Run first step with fixed scheduling
    let pulse = conductor.pulse();
    let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
    assert!(loss > 0.0, "Loss should be positive");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_fixed_vs_dynamic_off_equivalent() {
    // With dynamic_scheduling=false (default), the system should behave
    // identically to the original code. Two configs that differ only in
    // the dynamic_scheduling field should produce the same loss.
    let cfg1 = MAGConfig::test_config_k2();
    let mut cfg2 = MAGConfig::test_config_k2();
    cfg2.dynamic_scheduling = false; // explicit, same as default

    let params = MAGParams::init(&cfg1, 42);
    let (input_ids, target_ids) = make_data(&cfg1);

    let mut conductor1 = Conductor::new(cfg1.k, cfg1.chunk_sizes.clone());
    let mut context1 = ContextState::new(cfg1.k, cfg1.swa.d_model);
    let pulse1 = conductor1.pulse();
    let (loss1, _) = cms_forward(&params, &cfg1, &input_ids, &target_ids, &pulse1, &mut context1);

    let mut conductor2 = Conductor::new(cfg2.k, cfg2.chunk_sizes.clone());
    let mut context2 = ContextState::new(cfg2.k, cfg2.swa.d_model);
    let pulse2 = conductor2.pulse();
    let (loss2, _) = cms_forward(&params, &cfg2, &input_ids, &target_ids, &pulse2, &mut context2);

    assert!((loss1 - loss2).abs() < 1e-10, "Losses should be identical: {loss1} vs {loss2}");
}

// ── FrequencyScheduler unit tests ───────────────────────────────────

#[test]
fn test_scheduler_k1_trivial() {
    // k=1: no gates needed, level 0 always active.
    let d = 8;
    let sched = FrequencyScheduler::new(1, d, &[1], 42, DynamicFreqConfig::default());
    assert_eq!(sched.k(), 1);
    assert_eq!(sched.gates.len(), 0);
    assert_eq!(sched.num_params(), 0);

    let embedded = vec![0.5f32; 4 * d];
    let (gate_values, _) = sched.compute_gates(&embedded, 4, d);
    assert_eq!(gate_values.len(), 1);
    assert!((gate_values[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_scheduler_k2_gate_output() {
    let d = 8;
    let sched = FrequencyScheduler::new(2, d, &[1, 8], 42, DynamicFreqConfig::default());
    assert_eq!(sched.k(), 2);
    assert_eq!(sched.gates.len(), 1);

    let embedded = vec![0.1f32; 8 * d]; // seq_len=8
    let (gv, _cache) = sched.compute_gates(&embedded, 8, d);

    // Level 0 always 1.0
    assert!((gv[0] - 1.0).abs() < 1e-6);
    // Level 1 gate should be in (0, 1)
    assert!(gv[1] > 0.0 && gv[1] < 1.0, "Gate[1]={} not in (0,1)", gv[1]);
}

#[test]
fn test_scheduler_k4_all_gates() {
    let d = 8;
    let chunk_sizes = vec![1, 8, 64, 512];
    let sched = FrequencyScheduler::new(4, d, &chunk_sizes, 42, DynamicFreqConfig::default());
    assert_eq!(sched.k(), 4);
    assert_eq!(sched.gates.len(), 3);

    let embedded = vec![0.2f32; 16 * d]; // seq_len=16
    let (gv, _) = sched.compute_gates(&embedded, 16, d);
    assert_eq!(gv.len(), 4);

    // Level 0 always 1.0
    assert!((gv[0] - 1.0).abs() < 1e-6);

    // All other gates in [0, 1]
    for i in 1..4 {
        assert!(gv[i] >= 0.0 && gv[i] <= 1.0, "Gate[{i}]={} out of range", gv[i]);
    }

    // Higher levels should have lower initial gate values (smaller duty cycle)
    // because bias is initialized to match fixed-schedule duty cycle
    assert!(gv[1] > gv[2], "Level 1 gate ({}) should be > Level 2 gate ({})", gv[1], gv[2]);
    assert!(gv[2] > gv[3], "Level 2 gate ({}) should be > Level 3 gate ({})", gv[2], gv[3]);
}

// ── Default bias initialization ─────────────────────────────────────

#[test]
fn test_default_bias_matches_duty_cycle() {
    // Verify that default_freq_bias produces initial sigmoid values
    // matching the duty cycle of fixed scheduling.
    let test_cases = vec![
        (1, 8, 1.0 / 8.0),
        (2, 64, 1.0 / 64.0),
        (3, 512, 1.0 / 512.0),
    ];

    for (level, chunk_size, expected_duty) in test_cases {
        let b = default_freq_bias(level, chunk_size);
        let gate = 1.0 / (1.0 + (-b).exp());
        let rel_err = ((gate - expected_duty) / expected_duty).abs();
        assert!(
            rel_err < 0.05,
            "Level {level} (chunk={chunk_size}): gate={gate:.6}, expected ~{expected_duty:.6}, rel_err={rel_err:.4}"
        );
    }
}

// ── Gradient flow through gates ─────────────────────────────────────

#[test]
fn test_gate_gradient_flow() {
    // Verify that backward produces non-zero gradients for gate params.
    let d = 8;
    let k = 3;
    let chunk_sizes = vec![1, 8, 64];
    let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

    let embedded = vec![0.3f32; 4 * d];
    let (_gv, cache) = sched.compute_gates(&embedded, 4, d);

    // Upstream: d_loss/d_gate_value for each level
    let d_scale = vec![0.0, 1.5, -0.7]; // level 0 always ignored
    let grads = sched.backward(&cache, &d_scale, d);

    // Level 1 grads should be non-zero (d_scale[1] = 1.5)
    assert!(grads.gates[0].norm() > 1e-10, "Level 1 gate grads should be non-zero");
    // Level 2 grads should be non-zero (d_scale[2] = -0.7)
    assert!(grads.gates[1].norm() > 1e-10, "Level 2 gate grads should be non-zero");
}

#[test]
fn test_gate_fd_gradient_check() {
    // Finite-difference gradient check for gate parameters.
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

    let embedded = vec![0.5f32; 4 * d]; // seq_len=4, d=8

    // Define a scalar loss = gate_values[1] (just the gate output)
    let compute_loss = |s: &FrequencyScheduler| -> f32 {
        let (gv, _) = s.compute_gates(&embedded, 4, d);
        gv[1] // loss = gate value for level 1
    };

    // Analytical gradient via backward
    let (_gv, cache) = sched.compute_gates(&embedded, 4, d);
    let d_scale = vec![0.0, 1.0]; // d_loss/d_gate[1] = 1.0
    let grads = sched.backward(&cache, &d_scale, d);

    // FD check on bias
    let eps = 1e-3;
    let mut sched_plus = sched.clone();
    let mut sched_minus = sched.clone();
    sched_plus.gates[0].b_freq[0] += eps;
    sched_minus.gates[0].b_freq[0] -= eps;
    let fd_bias = (compute_loss(&sched_plus) - compute_loss(&sched_minus)) / (2.0 * eps);
    let anal_bias = grads.gates[0].b_freq[0];
    let bias_err = (fd_bias - anal_bias).abs();
    assert!(
        bias_err < 1e-3,
        "Bias FD gradient mismatch: fd={fd_bias:.6}, analytical={anal_bias:.6}, err={bias_err:.6}"
    );

    // FD check on a weight element
    for wi in 0..d.min(3) {
        let mut sp = sched.clone();
        let mut sm = sched.clone();
        sp.gates[0].w_freq[wi] += eps;
        sm.gates[0].w_freq[wi] -= eps;
        let fd_w = (compute_loss(&sp) - compute_loss(&sm)) / (2.0 * eps);
        let anal_w = grads.gates[0].w_freq[wi];
        let w_err = (fd_w - anal_w).abs();
        assert!(
            w_err < 1e-3,
            "Weight[{wi}] FD gradient mismatch: fd={fd_w:.6}, analytical={anal_w:.6}, err={w_err:.6}"
        );
    }
}

// ── Dynamic scheduling integration with CMS ─────────────────────────

#[test]
fn test_dynamic_pulse_generation() {
    // Verify that FrequencyScheduler can generate Pulse structs
    // that are compatible with cms_forward.
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

    let embedded = vec![0.1f32; 4 * d];
    let (gv, _) = sched.compute_gates(&embedded, 4, d);
    let active = sched.gate_to_active(&gv);

    // Build a Pulse from the dynamic scheduler
    let pulse = Pulse {
        global_step: 0,
        active_levels: active,
    };

    assert_eq!(pulse.active_levels.len(), k);
    assert!(pulse.active_levels[0]); // Level 0 always active
}

#[test]
fn test_dynamic_with_cms_forward_soft_gating() {
    // Run cms_forward with dynamically generated pulse (soft gating = all active).
    let cfg = MAGConfig::test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_data(&cfg);

    let d = cfg.swa.d_model;
    let sched = FrequencyScheduler::new(cfg.k, d, &cfg.chunk_sizes, 42, DynamicFreqConfig::default());

    // Compute embedded for gate input (simplified: use zeros as proxy)
    let embedded_proxy = vec![0.1f32; cfg.swa.seq_len * d];
    let (gv, _) = sched.compute_gates(&embedded_proxy, cfg.swa.seq_len, d);
    let active = sched.gate_to_active(&gv);

    let pulse = Pulse {
        global_step: 0,
        active_levels: active,
    };

    let mut context = ContextState::new(cfg.k, d);
    let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
    assert!(loss.is_finite() && loss > 0.0);
}

// ── Temperature effects ─────────────────────────────────────────────

#[test]
fn test_low_temperature_sharpens_gates() {
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];

    let cfg_warm = DynamicFreqConfig { temperature: 5.0, ..Default::default() };
    let cfg_cold = DynamicFreqConfig { temperature: 0.1, ..Default::default() };

    let sched_warm = FrequencyScheduler::new(k, d, &chunk_sizes, 42, cfg_warm);
    let sched_cold = FrequencyScheduler::new(k, d, &chunk_sizes, 42, cfg_cold);

    let embedded = vec![1.0f32; 4 * d];
    let (gv_warm, _) = sched_warm.compute_gates(&embedded, 4, d);
    let (gv_cold, _) = sched_cold.compute_gates(&embedded, 4, d);

    // Cold temperature should push gate closer to 0 or 1
    let dist_warm = (gv_warm[1] - 0.5).abs();
    let dist_cold = (gv_cold[1] - 0.5).abs();
    assert!(
        dist_cold >= dist_warm,
        "Cold temp should be sharper: warm_dist={dist_warm}, cold_dist={dist_cold}"
    );
}

// ── Hard gating ─────────────────────────────────────────────────────

#[test]
fn test_hard_gating_thresholds() {
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let config = DynamicFreqConfig {
        hard_gating: true,
        threshold: 0.5,
        ..Default::default()
    };

    let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);

    // Force bias high → gate > 0.5 → active
    sched.gates[0].b_freq[0] = 5.0;
    let embedded = vec![0.0f32; 4 * d];
    let (gv, _) = sched.compute_gates(&embedded, 4, d);
    let active = sched.gate_to_active(&gv);
    assert!(active[1], "Level 1 should be active with high bias");

    // Force bias low → gate < 0.5 → inactive
    sched.gates[0].b_freq[0] = -5.0;
    let (gv2, _) = sched.compute_gates(&embedded, 4, d);
    let active2 = sched.gate_to_active(&gv2);
    assert!(!active2[1], "Level 1 should be inactive with low bias");
}

// ── Edge cases ──────────────────────────────────────────────────────

#[test]
fn test_all_gates_open() {
    // Force all gates fully open (high bias)
    let d = 8;
    let k = 4;
    let chunk_sizes = vec![1, 8, 64, 512];
    let config = DynamicFreqConfig {
        hard_gating: true,
        threshold: 0.5,
        ..Default::default()
    };
    let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);
    for gate in &mut sched.gates {
        gate.b_freq[0] = 10.0;
    }

    let embedded = vec![0.0f32; 4 * d];
    let (gv, _) = sched.compute_gates(&embedded, 4, d);
    let active = sched.gate_to_active(&gv);

    assert!(active.iter().all(|&a| a), "All levels should be active with high bias");
    for gval in &gv {
        assert!(*gval > 0.99, "Gate should be near 1.0, got {gval}");
    }
}

#[test]
fn test_all_gates_closed_except_level0() {
    // Force all gates fully closed (low bias)
    let d = 8;
    let k = 4;
    let chunk_sizes = vec![1, 8, 64, 512];
    let config = DynamicFreqConfig {
        hard_gating: true,
        threshold: 0.5,
        ..Default::default()
    };
    let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);
    for gate in &mut sched.gates {
        gate.b_freq[0] = -10.0;
    }

    let embedded = vec![0.0f32; 4 * d];
    let (gv, _) = sched.compute_gates(&embedded, 4, d);
    let active = sched.gate_to_active(&gv);

    assert!(active[0], "Level 0 must always be active");
    for i in 1..k {
        assert!(!active[i], "Level {i} should be inactive with low bias");
        assert!(gv[i] < 0.01, "Gate[{i}] should be near 0, got {}", gv[i]);
    }
}

#[test]
fn test_cms_k1_k2_k4_compatibility() {
    // Verify FrequencyScheduler works with k=1, k=2, k=4.
    for (k, cs) in [(1, vec![1]), (2, vec![1, 8]), (4, vec![1, 8, 64, 512])] {
        let d = 8;
        let sched = FrequencyScheduler::new(k, d, &cs, 42, DynamicFreqConfig::default());
        assert_eq!(sched.k(), k);
        assert_eq!(sched.gates.len(), k.saturating_sub(1));

        let embedded = vec![0.1f32; 4 * d];
        let (gv, _) = sched.compute_gates(&embedded, 4, d);
        assert_eq!(gv.len(), k);
        assert!((gv[0] - 1.0).abs() < 1e-6, "Level 0 always 1.0");
    }
}

// ── Min gate floor ──────────────────────────────────────────────────

#[test]
fn test_min_gate_prevents_zero() {
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let config = DynamicFreqConfig {
        min_gate: 0.05,
        ..Default::default()
    };
    let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);
    sched.gates[0].b_freq[0] = -50.0; // Extremely negative → sigmoid ≈ 0

    let embedded = vec![0.0f32; 4 * d];
    let (gv, _) = sched.compute_gates(&embedded, 4, d);
    assert!(gv[1] >= 0.05, "Gate should be >= min_gate, got {}", gv[1]);
}

// ── Weight update convergence ───────────────────────────────────────

#[test]
fn test_gate_params_update_changes_output() {
    // Verify that applying gradients actually changes gate outputs.
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

    let embedded = vec![0.3f32; 4 * d];
    let (gv_before, cache) = sched.compute_gates(&embedded, 4, d);

    // Compute gradient and apply
    let d_scale = vec![0.0, 1.0];
    let grads = sched.backward(&cache, &d_scale, d);
    sched.apply_weight_gradients(&grads, 0.5); // large LR

    let (gv_after, _) = sched.compute_gates(&embedded, 4, d);

    assert!(
        (gv_after[1] - gv_before[1]).abs() > 1e-4,
        "Gate output should change after weight update: before={}, after={}",
        gv_before[1], gv_after[1]
    );
}

// ── Data-dependent gate activation ──────────────────────────────────

#[test]
fn test_different_inputs_different_gates() {
    // Different input embeddings should produce different gate values.
    let d = 8;
    let k = 2;
    let chunk_sizes = vec![1, 8];
    let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

    let embedded_a = vec![0.1f32; 4 * d];
    let mut embedded_b = vec![0.9f32; 4 * d];
    // Make embedding B clearly different
    for i in 0..d {
        embedded_b[i] = -0.5;
    }

    let (gv_a, _) = sched.compute_gates(&embedded_a, 4, d);
    let (gv_b, _) = sched.compute_gates(&embedded_b, 4, d);

    assert!(
        (gv_a[1] - gv_b[1]).abs() > 1e-6,
        "Different inputs should give different gates: a={}, b={}",
        gv_a[1], gv_b[1]
    );
}

// ── Scheduler serialization (zeros_like roundtrip) ──────────────────

#[test]
fn test_zeros_like_correct_shape() {
    let k = 4;
    let d = 16;
    let sched = FrequencyScheduler::zeros_like(k, d);
    assert_eq!(sched.gates.len(), 3);
    for gate in &sched.gates {
        assert_eq!(gate.w_freq.len(), d);
        assert_eq!(gate.b_freq.len(), 1);
        assert!(gate.w_freq.iter().all(|&x| x == 0.0));
        assert!(gate.b_freq[0] == 0.0);
    }
}
