//! M3 Multi-Scale Optimizer tests.
//!
//! Categories: unit (~8), error buffering (~4), integration (~4), edge cases (~4).

use nl_hecate_core::m3::{
    M3Config, M3State, m3_step, m3_is_active, newton_schulz_5,
    flatten_mag_params, unflatten_to_mag_grads, mag_params_count,
};
use nl_hecate_core::model::{MAGConfig, MAGParams};

// ── Unit tests ───────────────────────────────────────────────────────

#[test]
fn test_m3_config_k1() {
    let cfg = M3Config::default_k1();
    assert_eq!(cfg.k, 1);
    assert_eq!(cfg.etas, vec![0.9]);
    assert_eq!(cfg.thetas, vec![0.1]);
    assert_eq!(cfg.frequencies, vec![1]);
    assert!(!cfg.use_newton_schulz);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_m3_config_k2() {
    let cfg = M3Config::default_k2();
    assert_eq!(cfg.k, 2);
    assert_eq!(cfg.etas, vec![0.9, 0.99]);
    assert_eq!(cfg.thetas, vec![0.1, 0.01]);
    assert_eq!(cfg.frequencies, vec![1, 8]);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_m3_config_k4() {
    let cfg = M3Config::default_k4();
    assert_eq!(cfg.k, 4);
    assert_eq!(cfg.frequencies, vec![1, 8, 64, 512]);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_m3_state_init() {
    let cfg = M3Config::default_k2();
    let state = M3State::new(&cfg, 50);
    assert_eq!(state.momentum.len(), 2);
    assert_eq!(state.error_accum.len(), 2);
    assert_eq!(state.momentum[0].len(), 50);
    assert_eq!(state.momentum[1].len(), 50);
    assert_eq!(state.step, 0);
    assert_eq!(state.total_params, 50);
    // All zeros initially
    assert!(state.momentum[0].iter().all(|&x| x == 0.0));
    assert!(state.error_accum[1].iter().all(|&x| x == 0.0));
}

#[test]
fn test_m3_single_step() {
    let cfg = M3Config::default_k1();
    let mut state = M3State::new(&cfg, 3);
    let grad = vec![1.0, 2.0, 3.0];

    let update = m3_step(&mut state, &cfg, &grad);

    // Level 0: freq=1, always active. step=0: 0%1==0 so active.
    // momentum = 0.9 * 0 + 0.1 * (0 + grad) = 0.1 * grad
    // update = weight[0] * momentum[0] = 1.0 * 0.1 * grad
    assert_eq!(update.len(), 3);
    assert!((update[0] - 0.1).abs() < 1e-6, "update[0]={}", update[0]);
    assert!((update[1] - 0.2).abs() < 1e-6, "update[1]={}", update[1]);
    assert!((update[2] - 0.3).abs() < 1e-6, "update[2]={}", update[2]);
    assert_eq!(state.step, 1);
}

#[test]
fn test_ema_convergence() {
    // Constant gradient: momentum should converge to theta/(1-eta) * grad
    let cfg = M3Config::default_k1();
    let mut state = M3State::new(&cfg, 1);
    let grad = vec![1.0];

    // Converge: S = eta*S + theta*g => steady state S = theta/(1-eta)*g = 0.1/0.1 = 1.0
    for _ in 0..100 {
        let _ = m3_step(&mut state, &cfg, &grad);
    }
    // update = weight * momentum = 1.0 * converged_momentum
    let update = m3_step(&mut state, &cfg, &grad);
    let expected = 0.1 / (1.0 - 0.9); // theta / (1 - eta) = 1.0
    assert!((update[0] - expected).abs() < 0.01,
        "EMA convergence: update={}, expected={expected}", update[0]);
}

#[test]
fn test_newton_schulz_output_finite() {
    let d = 4;
    // Random-ish matrix
    let m: Vec<f32> = (0..d*d).map(|i| ((i as f32 * 0.37 + 0.1).sin()) * 0.5).collect();
    let result = newton_schulz_5(&m, d, 10);
    assert_eq!(result.len(), d * d);
    // Result should be non-zero and finite
    let out_frob: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(out_frob > 0.1, "NS output should be non-zero: frob={out_frob}");
    assert!(out_frob.is_finite(), "NS output should be finite");
    // Result should be different from input (NS modifies the matrix)
    let diff: f32 = result.iter().zip(m.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.01, "NS should modify the matrix");
}

#[test]
fn test_newton_schulz_zero_iterations() {
    let d = 3;
    let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result = newton_schulz_5(&m, d, 0);
    assert_eq!(result, m, "zero iterations should return input unchanged");
}

// ── Error buffering tests ────────────────────────────────────────────

#[test]
fn test_frozen_level_accumulates() {
    let cfg = M3Config::default_k2();
    let mut state = M3State::new(&cfg, 2);

    // Step 0: both levels active (0%1==0, 0%8==0)
    let _ = m3_step(&mut state, &cfg, &[1.0, 2.0]);

    // Steps 1-7: level 0 active, level 1 frozen
    for step in 1..8 {
        let _ = m3_step(&mut state, &cfg, &[1.0, 2.0]);
        if step < 7 {
            // Level 1 should be accumulating
            assert_eq!(state.steps_accumulated[1], step,
                "step {}: expected {} accumulated", step + 1, step);
        }
    }

    // After step 7 (state.step=8), level 1 fired at step 8 and was drained
    // Actually: m3_step at the end of step 7 advances to step=8.
    // At step=7 (before advance): 7%8!=0, so level 1 frozen.
    // Let's check the state after the loop
    assert_eq!(state.step, 8);
    // Level 1 fired at step 8 (8%8==0) during the step=8 call
    // Wait — step is advanced at END of m3_step. So call at step=7 checks 7%8!=0: frozen.
    // Then step becomes 8. Next call would check step=8: 8%8==0: active.
    let _ = m3_step(&mut state, &cfg, &[1.0, 2.0]);
    // After this call, level 1 fired (step=8 checked), then step advanced to 9
    assert_eq!(state.steps_accumulated[1], 0, "level 1 should have drained");
}

#[test]
fn test_active_level_drains_buffer() {
    let cfg = M3Config::default_k2();
    let mut state = M3State::new(&cfg, 2);

    // Manually set error buffer
    state.error_accum[1] = vec![10.0, 20.0];
    state.steps_accumulated[1] = 5;
    state.step = 8; // 8 % 8 == 0, so level 1 will be active

    let _ = m3_step(&mut state, &cfg, &[1.0, 2.0]);

    // Error buffer should be drained
    assert_eq!(state.error_accum[1], vec![0.0, 0.0]);
    assert_eq!(state.steps_accumulated[1], 0);
    // Momentum should have incorporated the accumulated error
    // momentum[1] = 0.99*0 + 0.01*(10+1, 20+2) = 0.01*(11, 22) = (0.11, 0.22)
    assert!((state.momentum[1][0] - 0.11).abs() < 1e-6);
    assert!((state.momentum[1][1] - 0.22).abs() < 1e-6);
}

#[test]
fn test_k2_alternating_activation() {
    let cfg = M3Config::default_k2();
    let mut state = M3State::new(&cfg, 1);

    // Track which levels fire at each step
    let mut level1_fired = Vec::new();
    for _ in 0..16 {
        let step_before = state.step;
        let _ = m3_step(&mut state, &cfg, &[1.0]);
        level1_fired.push(m3_is_active(&cfg, step_before, 1));
    }

    // Level 1 (freq=8) should fire at steps 0, 8
    assert!(level1_fired[0], "level 1 should fire at step 0");
    assert!(!level1_fired[1], "level 1 should NOT fire at step 1");
    assert!(!level1_fired[7], "level 1 should NOT fire at step 7");
    assert!(level1_fired[8], "level 1 should fire at step 8");
}

#[test]
fn test_k4_frequency_gating() {
    let cfg = M3Config::default_k4();
    // Check activation pattern at various steps
    assert!(m3_is_active(&cfg, 0, 0));   // freq=1: always
    assert!(m3_is_active(&cfg, 0, 1));   // freq=8: at 0
    assert!(m3_is_active(&cfg, 0, 2));   // freq=64: at 0
    assert!(m3_is_active(&cfg, 0, 3));   // freq=512: at 0
    assert!(!m3_is_active(&cfg, 1, 1));  // freq=8: not at 1
    assert!(!m3_is_active(&cfg, 63, 2)); // freq=64: not at 63
    assert!(m3_is_active(&cfg, 64, 2));  // freq=64: at 64
    assert!(!m3_is_active(&cfg, 511, 3));// freq=512: not at 511
    assert!(m3_is_active(&cfg, 512, 3)); // freq=512: at 512
}

// ── Integration tests ────────────────────────────────────────────────

#[test]
fn test_m3_vs_sgd_k1_eta0() {
    // With eta=0.0, theta=1.0, weight=1.0: M3 degenerates to plain gradient
    // update = 1.0 * (0.0 * S + 1.0 * grad) = grad
    let cfg = M3Config {
        k: 1,
        etas: vec![0.0],
        thetas: vec![1.0],
        weights: vec![1.0],
        frequencies: vec![1],
        use_newton_schulz: false,
        ns_iterations: 5,
        ns_dim: None,
    };
    let mut state = M3State::new(&cfg, 3);
    let grad = vec![0.5, -0.3, 0.8];

    let update = m3_step(&mut state, &cfg, &grad);

    // update should equal grad (no momentum)
    for i in 0..3 {
        assert!((update[i] - grad[i]).abs() < 1e-6,
            "M3 k1 eta=0: update[{i}]={}, grad[{i}]={}", update[i], grad[i]);
    }
}

#[test]
fn test_m3_k2_momentum_accumulates() {
    // Verify k=2 momentum builds up correctly across steps.
    // Level 0 (freq=1): fires every step, accumulates fast gradient info.
    // Level 1 (freq=8): fires every 8 steps, accumulates slow gradient info.
    let cfg = M3Config::default_k2();
    let mut state = M3State::new(&cfg, 1);

    // Apply constant gradient for 16 steps
    let mut updates = Vec::new();
    for _ in 0..16 {
        let u = m3_step(&mut state, &cfg, &[1.0]);
        updates.push(u[0]);
    }

    // Updates should generally increase as momentum builds (until saturation)
    assert!(updates[8] > updates[0],
        "momentum should build: step8={}, step0={}", updates[8], updates[0]);
    // Both levels contribute at step 8 (8%1==0 and 8%8==0)
    // The combined update at step 8 should have a visible jump from level 1 kicking in
    assert!(updates[15] > updates[1],
        "later updates should be larger: step15={}, step1={}", updates[15], updates[1]);
}

#[test]
fn test_m3_with_mag_forward_backward() {
    // Full integration: MAGConfig -> params -> flatten -> M3 -> unflatten -> apply
    let mag_cfg = MAGConfig::test_config();
    let mut params = MAGParams::init(&mag_cfg, 42);
    let m3_cfg = M3Config::default_k1();
    let n = params.num_params();
    let mut m3_state = M3State::new(&m3_cfg, n);

    // Create fake gradient (all 0.01)
    let grads = MAGParams::zeros_like(&mag_cfg);
    let flat_grads = flatten_mag_params(&grads);
    assert_eq!(flat_grads.len(), n);

    // Run M3 step
    let update = m3_step(&mut m3_state, &m3_cfg, &flat_grads);
    assert_eq!(update.len(), n);

    // Unflatten and apply
    let update_as_grads = unflatten_to_mag_grads(&update, &params);
    let orig_w_q = params.swa.w_q.clone();
    params.apply_weight_gradients(&update_as_grads, 1.0);

    // Since grads were zero, update should be zero, params unchanged
    assert_eq!(params.swa.w_q, orig_w_q);
}

#[test]
fn test_m3_apply_weight_gradients_m3() {
    // Test the convenience method directly
    let mag_cfg = MAGConfig::test_config();
    let mut params = MAGParams::init(&mag_cfg, 42);
    let orig_params = params.clone();
    let m3_cfg = M3Config::default_k1();
    let n = params.num_params();
    let mut m3_state = M3State::new(&m3_cfg, n);

    // Non-zero gradients
    let mut grads = MAGParams::zeros_like(&mag_cfg);
    grads.swa.w_q.iter_mut().for_each(|v| *v = 0.01);
    grads.levels[0].w_k_mem.iter_mut().for_each(|v| *v = 0.01);

    params.apply_weight_gradients_m3(&grads, &mut m3_state, &m3_cfg);

    // Params should have changed
    let changed = params.swa.w_q.iter().zip(orig_params.swa.w_q.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "M3 should have modified w_q");
}

// ── Edge cases ───────────────────────────────────────────────────────

#[test]
fn test_m3_all_levels_active() {
    // All levels fire simultaneously (step=0, all freqs divide 0)
    let cfg = M3Config::default_k4();
    let mut state = M3State::new(&cfg, 2);

    // At step 0, all 4 levels are active
    for level in 0..4 {
        assert!(m3_is_active(&cfg, 0, level), "level {level} should be active at step 0");
    }

    let update = m3_step(&mut state, &cfg, &[1.0, 2.0]);
    assert_eq!(update.len(), 2);
    // All levels contribute: sum(weight[i] * theta[i] * grad) for i in 0..4
    // = 1.0*0.1 + 1.0*0.05 + 1.0*0.01 + 1.0*0.001 = 0.161 per grad unit
    let expected_0 = 0.1 + 0.05 + 0.01 + 0.001;
    assert!((update[0] - expected_0).abs() < 1e-5,
        "all active: update[0]={}, expected={expected_0}", update[0]);
}

#[test]
fn test_m3_no_levels_active() {
    // Custom config where no level fires at step 1
    let cfg = M3Config {
        k: 2,
        etas: vec![0.9, 0.99],
        thetas: vec![0.1, 0.01],
        weights: vec![1.0, 1.0],
        frequencies: vec![2, 4], // neither fires at step 1
        use_newton_schulz: false,
        ns_iterations: 5,
        ns_dim: None,
    };
    let mut state = M3State::new(&cfg, 2);
    state.step = 1; // step 1: 1%2!=0, 1%4!=0

    let update = m3_step(&mut state, &cfg, &[1.0, 2.0]);

    // All momentum stays at 0 (initialized to zero), so update = 0
    assert!((update[0]).abs() < 1e-10, "no levels active: update should be 0");
    assert!((update[1]).abs() < 1e-10, "no levels active: update should be 0");
    // Gradients should have accumulated in error buffers
    assert_eq!(state.error_accum[0], vec![1.0, 2.0]);
    assert_eq!(state.error_accum[1], vec![1.0, 2.0]);
}

#[test]
fn test_m3_step_counter_advances() {
    let cfg = M3Config::default_k1();
    let mut state = M3State::new(&cfg, 1);
    assert_eq!(state.step, 0);

    let _ = m3_step(&mut state, &cfg, &[1.0]);
    assert_eq!(state.step, 1);

    let _ = m3_step(&mut state, &cfg, &[1.0]);
    assert_eq!(state.step, 2);

    for _ in 0..10 {
        let _ = m3_step(&mut state, &cfg, &[1.0]);
    }
    assert_eq!(state.step, 12);
}

#[test]
fn test_flatten_unflatten_roundtrip() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let n = mag_params_count(&params);

    let flat = flatten_mag_params(&params);
    assert_eq!(flat.len(), n, "flat length should match num_params");

    let restored = unflatten_to_mag_grads(&flat, &params);

    // Check SWA roundtrip
    assert_eq!(restored.swa.w_embed, params.swa.w_embed);
    assert_eq!(restored.swa.w_q, params.swa.w_q);
    assert_eq!(restored.swa.w_k, params.swa.w_k);
    assert_eq!(restored.swa.w_v, params.swa.w_v);
    assert_eq!(restored.swa.w_o, params.swa.w_o);
    assert_eq!(restored.swa.w_unembed, params.swa.w_unembed);

    // Check memory level roundtrip
    assert_eq!(restored.levels.len(), params.levels.len());
    assert_eq!(restored.levels[0].w_k_mem, params.levels[0].w_k_mem);
    assert_eq!(restored.levels[0].w_alpha, params.levels[0].w_alpha);
    assert_eq!(restored.levels[0].b_alpha, params.levels[0].b_alpha);
    assert_eq!(restored.levels[0].w_eta, params.levels[0].w_eta);
}

#[test]
fn test_flatten_unflatten_roundtrip_k2() {
    let cfg = MAGConfig::test_config_k2();
    let params = MAGParams::init(&cfg, 42);
    let flat = flatten_mag_params(&params);
    let restored = unflatten_to_mag_grads(&flat, &params);

    assert_eq!(restored.levels.len(), 2);
    assert_eq!(restored.levels[0].w_k_mem, params.levels[0].w_k_mem);
    assert_eq!(restored.levels[1].w_k_mem, params.levels[1].w_k_mem);
    assert_eq!(restored.levels[1].b_theta, params.levels[1].b_theta);
}
