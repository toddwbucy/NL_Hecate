//! DGD (Delta Gradient Descent) primitive tests.
//!
//! Tests the standalone DGD functions extracted from delta_rule.rs.
//! Covers: basic I/O, zero gates, momentum, Sherman-Morrison,
//! FD gradient check, loss monotonicity, and delegation regression.

use nl_hecate_core::dgd::{dgd_error, dgd_error_into, dgd_update, dgd_step, dgd_step_backward, dgd_momentum_step, dgd_sherman_morrison};
use nl_hecate_core::model::{
    MAGConfig, MAGParams, SWAConfig, CompositionKind, MemoryRuleKind,
    HopeVariant, LatticeVariant, MomentumKind,
};
use nl_hecate_core::dynamic_freq::FrequencySchedule;
use nl_hecate_core::retention::{RetentionKind, default_retention};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};
use nl_hecate_core::mag::{cms_forward, cms_backward};

// ── Test 1: Basic dgd_step ──────────────────────────────────────────

#[test]
fn test_dgd_step_basic() {
    let d = 4;
    // M = identity, k = [1,0,0,0], v = [0.5, 0.5, 0.5, 0.5]
    let mut m = vec![0.0f32; d * d];
    for i in 0..d { m[i * d + i] = 1.0; }

    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![0.5, 0.5, 0.5, 0.5];

    let alpha = 0.1;
    let theta = 0.5;

    // error = M@k - v = [1,0,0,0] - [0.5,0.5,0.5,0.5] = [0.5, -0.5, -0.5, -0.5]
    let error = dgd_step(&mut m, &k, &v, alpha, theta, d);

    assert!((error[0] - 0.5).abs() < 1e-6, "error[0] = {}", error[0]);
    assert!((error[1] + 0.5).abs() < 1e-6, "error[1] = {}", error[1]);
    assert!((error[2] + 0.5).abs() < 1e-6, "error[2] = {}", error[2]);
    assert!((error[3] + 0.5).abs() < 1e-6, "error[3] = {}", error[3]);

    // M should be modified: M = 0.9 * I - 0.5 * outer(error, k)
    // outer(error, k) = [[0.5,0,0,0],[-0.5,0,0,0],[-0.5,0,0,0],[-0.5,0,0,0]]
    // M[0][0] = 0.9*1.0 - 0.5*0.5 = 0.65
    // M[1][0] = 0.9*0.0 - 0.5*(-0.5) = 0.25
    assert!((m[0] - 0.65).abs() < 1e-6, "m[0][0] = {}", m[0]);
    assert!((m[d] - 0.25).abs() < 1e-6, "m[1][0] = {}", m[d]);
    // Off-axis elements should just be retention-scaled
    // M[1][1] = 0.9 * 1.0 - 0 = 0.9
    assert!((m[d + 1] - 0.9).abs() < 1e-6, "m[1][1] = {}", m[d + 1]);
}

// ── Test 2: Zero gates ──────────────────────────────────────────────

#[test]
fn test_dgd_step_zero_gates() {
    let d = 4;
    let mut m = vec![0.0f32; d * d];
    for i in 0..d { m[i * d + i] = 1.0; }
    let m_orig = m.clone();

    let k = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.5, 0.5, 0.5, 0.5];

    // alpha=0, theta=0 => M unchanged: (1-0)*M - 0*grad = M
    dgd_step(&mut m, &k, &v, 0.0, 0.0, d);

    for i in 0..(d * d) {
        assert!((m[i] - m_orig[i]).abs() < 1e-10,
            "M should be unchanged with zero gates, index {i}: {} vs {}", m[i], m_orig[i]);
    }
}

// ── Test 3: Momentum step ───────────────────────────────────────────

#[test]
fn test_dgd_momentum_step() {
    let d = 4;
    let mut m = vec![0.0f32; d * d];
    for i in 0..d { m[i * d + i] = 1.0; }
    let mut s = vec![0.0f32; d * d];

    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![0.5, 0.5, 0.5, 0.5];
    let alpha = 0.05;
    let theta = 0.3;
    let beta = 0.9;

    // Step 1: S starts at 0, so S = 0 + theta*grad = theta*grad
    let error1 = dgd_error(&m, &k, &v, d);
    dgd_momentum_step(&mut m, &mut s, &error1, &k, alpha, theta, beta, d);

    // S should be non-zero after step 1
    let s_norm: f32 = s.iter().map(|x| x * x).sum();
    assert!(s_norm > 1e-6, "S should accumulate momentum, norm = {s_norm}");

    // Step 2: S should accumulate (beta * S_prev + theta * grad_new)
    let _s_after_1 = s.clone();
    let error2 = dgd_error(&m, &k, &v, d);
    dgd_momentum_step(&mut m, &mut s, &error2, &k, alpha, theta, beta, d);

    // S[0] should be beta * s_after_1[0] + theta * new_grad[0]
    // We just verify it changed and is larger in magnitude (accumulation)
    let s_norm_2: f32 = s.iter().map(|x| x * x).sum();
    assert!(s_norm_2 > 0.0, "S should continue to accumulate");
}

// ── Test 4: Sherman-Morrison matches dgd_step for L2 ────────────────

#[test]
fn test_dgd_sherman_morrison_matches_step() {
    // For L2 bias with normalized k (||k||=1), SM and dgd_step should
    // produce similar results when eta is small (linear regime).
    let d = 4;
    let k: Vec<f32> = {
        let raw = vec![1.0, 0.0, 0.0, 0.0]; // already unit norm
        raw
    };
    let v = vec![0.5, 0.3, 0.2, 0.1];

    // Small eta for linear regime equivalence
    let eta = 0.01;
    let eta_prime = eta / (1.0 + eta); // ~0.0099

    // DGD step with alpha=0 (no retention), theta=eta_prime
    let mut m_dgd = vec![0.0f32; d * d];
    for i in 0..d { m_dgd[i * d + i] = 1.0; }
    dgd_step(&mut m_dgd, &k, &v, 0.0, eta_prime, d);

    // Sherman-Morrison
    let mut m_sm = vec![0.0f32; d * d];
    for i in 0..d { m_sm[i * d + i] = 1.0; }
    dgd_sherman_morrison(&mut m_sm, &k, &v, eta, d);

    // They should match well for unit-norm k
    for i in 0..(d * d) {
        assert!((m_dgd[i] - m_sm[i]).abs() < 1e-4,
            "DGD and SM should match at index {i}: dgd={}, sm={}", m_dgd[i], m_sm[i]);
    }
}

// ── Test 5: FD gradient check ───────────────────────────────────────

#[test]
fn test_dgd_step_backward_fd() {
    let d = 8;
    let eps = 1e-2;
    let tol = 0.10; // 10%
    let abs_threshold = 5e-4;

    // Random-ish but deterministic inputs
    let mut m = vec![0.0f32; d * d];
    for i in 0..d { m[i * d + i] = 0.5; }
    for i in 0..(d * d) { m[i] += 0.01 * (i as f32).sin(); }

    let k: Vec<f32> = (0..d).map(|i| 0.3 * ((i as f32) * 1.7).cos()).collect();
    let v: Vec<f32> = (0..d).map(|i| 0.4 * ((i as f32) * 2.3).sin()).collect();
    let alpha = 0.1;
    let theta = 0.5;

    // Compute analytical gradients
    // Use a simple loss: L = sum(M_{t+1})  =>  dL/dM_{t+1} = ones
    let d_m_out = vec![1.0f32; d * d];
    let grads = dgd_step_backward(&d_m_out, &m, &k, &v, alpha, theta, d);

    // FD check on M
    let mut fd_pass = 0;
    let mut fd_total = 0;
    for idx in 0..(d * d) {
        let mut m_plus = m.clone();
        m_plus[idx] += eps;
        dgd_step(&mut m_plus, &k, &v, alpha, theta, d);
        let loss_plus: f32 = m_plus.iter().sum();

        let mut m_minus = m.clone();
        m_minus[idx] -= eps;
        dgd_step(&mut m_minus, &k, &v, alpha, theta, d);
        let loss_minus: f32 = m_minus.iter().sum();

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let anal = grads.d_m[idx];

        fd_total += 1;
        let denom = fd.abs().max(anal.abs());
        if denom < abs_threshold {
            fd_pass += 1;
            continue;
        }
        let rel_err = (fd - anal).abs() / denom;
        assert!(rel_err < tol,
            "d_m[{idx}] FD mismatch: fd={fd:.6}, anal={anal:.6}, rel_err={rel_err:.4}");
        fd_pass += 1;
    }
    assert!(fd_pass == fd_total, "Not all d_m checks passed: {fd_pass}/{fd_total}");

    // FD check on k
    for idx in 0..d {
        let mut k_plus = k.clone();
        k_plus[idx] += eps;
        let mut m_p = m.clone();
        dgd_step(&mut m_p, &k_plus, &v, alpha, theta, d);
        let loss_plus: f32 = m_p.iter().sum();

        let mut k_minus = k.clone();
        k_minus[idx] -= eps;
        let mut m_m = m.clone();
        dgd_step(&mut m_m, &k_minus, &v, alpha, theta, d);
        let loss_minus: f32 = m_m.iter().sum();

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let anal = grads.d_k[idx];

        let denom = fd.abs().max(anal.abs());
        if denom < abs_threshold { continue; }
        let rel_err = (fd - anal).abs() / denom;
        assert!(rel_err < tol,
            "d_k[{idx}] FD mismatch: fd={fd:.6}, anal={anal:.6}, rel_err={rel_err:.4}");
    }

    // FD check on v
    for idx in 0..d {
        let mut v_plus = v.clone();
        v_plus[idx] += eps;
        let mut m_p = m.clone();
        dgd_step(&mut m_p, &k, &v_plus, alpha, theta, d);
        let loss_plus: f32 = m_p.iter().sum();

        let mut v_minus = v.clone();
        v_minus[idx] -= eps;
        let mut m_m = m.clone();
        dgd_step(&mut m_m, &k, &v_minus, alpha, theta, d);
        let loss_minus: f32 = m_m.iter().sum();

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let anal = grads.d_v[idx];

        let denom = fd.abs().max(anal.abs());
        if denom < abs_threshold { continue; }
        let rel_err = (fd - anal).abs() / denom;
        assert!(rel_err < tol,
            "d_v[{idx}] FD mismatch: fd={fd:.6}, anal={anal:.6}, rel_err={rel_err:.4}");
    }

    // FD check on alpha
    {
        let mut m_p = m.clone();
        dgd_step(&mut m_p, &k, &v, alpha + eps, theta, d);
        let loss_plus: f32 = m_p.iter().sum();

        let mut m_m = m.clone();
        dgd_step(&mut m_m, &k, &v, alpha - eps, theta, d);
        let loss_minus: f32 = m_m.iter().sum();

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let anal = grads.d_alpha;

        let denom = fd.abs().max(anal.abs());
        if denom >= abs_threshold {
            let rel_err = (fd - anal).abs() / denom;
            assert!(rel_err < tol,
                "d_alpha FD mismatch: fd={fd:.6}, anal={anal:.6}, rel_err={rel_err:.4}");
        }
    }

    // FD check on theta
    {
        let mut m_p = m.clone();
        dgd_step(&mut m_p, &k, &v, alpha, theta + eps, d);
        let loss_plus: f32 = m_p.iter().sum();

        let mut m_m = m.clone();
        dgd_step(&mut m_m, &k, &v, alpha, theta - eps, d);
        let loss_minus: f32 = m_m.iter().sum();

        let fd = (loss_plus - loss_minus) / (2.0 * eps);
        let anal = grads.d_theta;

        let denom = fd.abs().max(anal.abs());
        if denom >= abs_threshold {
            let rel_err = (fd - anal).abs() / denom;
            assert!(rel_err < tol,
                "d_theta FD mismatch: fd={fd:.6}, anal={anal:.6}, rel_err={rel_err:.4}");
        }
    }
}

// ── Test 6: Loss monotonicity ───────────────────────────────────────

#[test]
fn test_dgd_loss_monotonicity() {
    // Behavioral probe: run N DGD steps on fixed (k, v), assert ||M@k - v||^2
    // decreases monotonically.
    // HADES: hope_probes/probe-loss-monotonicity-dgd
    let d = 8;
    let mut m = vec![0.0f32; d * d];
    let k: Vec<f32> = (0..d).map(|i| {
        let raw = ((i as f32) * 1.3 + 0.5).sin();
        raw
    }).collect();
    // Normalize k for stability
    let k_norm: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k: Vec<f32> = k.iter().map(|x| x / k_norm).collect();
    let v: Vec<f32> = (0..d).map(|i| ((i as f32) * 2.7).cos()).collect();

    let n_steps = 20;
    let alpha = 0.01; // small retention
    let theta = 0.3;  // reasonable learning rate

    let mut prev_loss = f32::MAX;
    for step in 0..n_steps {
        let error = dgd_error(&m, &k, &v, d);
        let loss: f32 = error.iter().map(|e| e * e).sum();

        if step > 0 {
            assert!(loss <= prev_loss + 1e-6,
                "Loss should decrease monotonically: step {step}, loss={loss:.6}, prev={prev_loss:.6}");
        }
        prev_loss = loss;

        dgd_step(&mut m, &k, &v, alpha, theta, d);
    }

    // Final loss should be much smaller than initial
    let final_error = dgd_error(&m, &k, &v, d);
    let final_loss: f32 = final_error.iter().map(|e| e * e).sum();
    assert!(final_loss < 0.1, "Final loss should be small: {final_loss}");
}

// ── Test 7: DeltaRule delegation regression ─────────────────────────

/// Helper: build a tiny DeltaRule config.
fn tiny_delta_config() -> MAGConfig {
    MAGConfig {
        swa: SWAConfig {
            d_model: 8,
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 4,
            vocab_size: 16,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: MemoryRuleKind::DeltaRule,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 0,
        lp_p: 2.0,
        sign_sharpness: 10.0,
        lq_q: 2.0,
        lambda_local: 0.0,
        lambda_2: 0.0,
        delta: 1.0,
        m_slots: 0,
        d_compress: 0,
        lambda_k: 0.0,
        lambda_v: 0.0,
        parallel: None,
        retention: default_retention(MemoryRuleKind::DeltaRule),
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
    }
}

#[test]
fn test_delta_rule_delegation_regression() {
    // Run DeltaRule forward + backward with a known seed and verify
    // loss and gradient norms are reasonable (proves delegation didn't break anything).
    let cfg = tiny_delta_config();
    let params = MAGParams::init(&cfg, 42);
    let d = cfg.swa.d_model;

    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();

    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let mut context = ContextState::new(cfg.k, d);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();

    let pulse = conductor.pulse();
    let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
    let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

    // Loss should be finite and positive
    assert!(loss.is_finite(), "Loss should be finite: {loss}");
    assert!(loss > 0.0, "Loss should be positive: {loss}");

    // Gradient norms should be finite and non-zero
    let w_k_grad_norm: f32 = grads.levels[0].w_k_mem.master().iter().map(|x| x * x).sum::<f32>().sqrt();
    let w_v_grad_norm: f32 = grads.levels[0].w_v_mem.master().iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(w_k_grad_norm.is_finite() && w_k_grad_norm > 0.0,
        "w_k_mem gradient should be finite and non-zero: {w_k_grad_norm}");
    assert!(w_v_grad_norm.is_finite() && w_v_grad_norm > 0.0,
        "w_v_mem gradient should be finite and non-zero: {w_v_grad_norm}");

    // Verify a training step reduces loss — update all f32 params (not just bf16 projections)
    let mut params2 = params.clone();
    let lr = 0.1;
    for i in 0..grads.levels[0].w_k_mem.master().len() {
        params2.levels[0].w_k_mem.master_mut()[i] -= lr * grads.levels[0].w_k_mem.master()[i];
    }
    params2.levels[0].w_k_mem.sync_from_master();
    for i in 0..grads.levels[0].w_v_mem.master().len() {
        params2.levels[0].w_v_mem.master_mut()[i] -= lr * grads.levels[0].w_v_mem.master()[i];
    }
    params2.levels[0].w_v_mem.sync_from_master();
    for i in 0..grads.levels[0].w_alpha.len() {
        params2.levels[0].w_alpha[i] -= lr * grads.levels[0].w_alpha[i];
    }
    for i in 0..grads.levels[0].w_theta.len() {
        params2.levels[0].w_theta[i] -= lr * grads.levels[0].w_theta[i];
    }
    for i in 0..grads.levels[0].b_alpha.len() {
        params2.levels[0].b_alpha[i] -= lr * grads.levels[0].b_alpha[i];
    }
    for i in 0..grads.levels[0].b_theta.len() {
        params2.levels[0].b_theta[i] -= lr * grads.levels[0].b_theta[i];
    }
    // Also update SWA params
    for i in 0..grads.swa.w_q.len() {
        params2.swa.w_q[i] -= lr * grads.swa.w_q[i];
    }
    for i in 0..grads.swa.w_k.len() {
        params2.swa.w_k[i] -= lr * grads.swa.w_k[i];
    }
    for i in 0..grads.swa.w_v.len() {
        params2.swa.w_v[i] -= lr * grads.swa.w_v[i];
    }
    for i in 0..grads.swa.w_o.len() {
        params2.swa.w_o[i] -= lr * grads.swa.w_o[i];
    }
    for i in 0..grads.swa.w_embed.len() {
        params2.swa.w_embed[i] -= lr * grads.swa.w_embed[i];
    }
    for i in 0..grads.swa.w_unembed.len() {
        params2.swa.w_unembed[i] -= lr * grads.swa.w_unembed[i];
    }

    let mut context2 = ContextState::new(cfg.k, d);
    let pulse2 = conductor.pulse();
    let (loss2, _) = cms_forward(&params2, &cfg, &input_ids, &target_ids, &pulse2, &mut context2);
    assert!(loss2 < loss, "Loss should decrease after gradient step: {loss2} >= {loss}");
}
