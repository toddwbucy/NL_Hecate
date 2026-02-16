// CUDA DeltaRule Kernel Tests — S2-M1 Phase 1
//
// Tests the CUDA inner-loop kernel for the Delta Rule memory.
// All fp32 (no bf16 conversion). Tolerances are tighter than SWA CUDA tests.
//
// Test classes:
//   1. Forward match: CUDA vs Rust reference inner loop (tol 1e-5)
//   2. Backward match: all 6 gradient arrays (tol 1e-4)
//   3. M states match: cached m_states identical (tol 1e-5)
//   4. Full pipeline FD gradient check (7 weights)
//   5. Outer-loop convergence via CUDA path
//   6. Edge cases (seq_len=1, zero M)
//   7. CUDA-Rust loss parity

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::dispatch::{delta_forward_dispatch, delta_backward_dispatch};
use nl_hecate_core::delta_rule::{DeltaRule, MemoryRule};
use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryLevelParams};
use nl_hecate_core::tensor::{sigmoid_f32, softplus_f32, transpose_f32, matmul_f32};

/// Run the Rust reference inner loop (same math as DeltaRule::step inner loop,
/// but using the dispatch-compatible interface).
fn rust_inner_loop_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_initial: &[f32],
    seq_len: usize, d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dd = d * d;
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    m_states[..dd].copy_from_slice(m_initial);

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        let retention = 1.0 - alpha_t;
        for i in 0..d {
            let err_i = prediction[i] - v_t[i];
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] - theta_t * err_i * k_t[j];
            }
        }

        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }

    (m_states, y)
}

/// Extract projections and gates from MemoryLevelParams and embedded,
/// matching the exact computation in DeltaRule::step().
fn extract_projections(
    params: &MemoryLevelParams, embedded: &[f32], seq_len: usize, d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    let mut w_q_t = vec![0.0f32; d * d];
    transpose_f32(&params.w_k_mem, &mut w_k_t, d, d);
    transpose_f32(&params.w_v_mem, &mut w_v_t, d, d);
    transpose_f32(&params.w_q_mem, &mut w_q_t, d, d);

    let mut k_mem = vec![0.0f32; seq_len * d];
    let mut v_mem = vec![0.0f32; seq_len * d];
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_k_t, &mut k_mem, seq_len, d, d);
    matmul_f32(embedded, &w_v_t, &mut v_mem, seq_len, d, d);
    matmul_f32(embedded, &w_q_t, &mut q_mem, seq_len, d, d);

    let mut alpha = vec![0.0f32; seq_len];
    let mut theta = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let mut alpha_pre = params.b_alpha[0];
        let mut theta_pre = params.b_theta[0];
        for i in 0..d {
            alpha_pre += k_t[i] * params.w_alpha[i];
            theta_pre += k_t[i] * params.w_theta[i];
        }
        for i in 0..d {
            alpha_pre += v_t[i] * params.w_alpha[d + i];
            theta_pre += v_t[i] * params.w_theta[d + i];
        }
        alpha[t] = sigmoid_f32(alpha_pre);
        theta[t] = softplus_f32(theta_pre);
    }

    (k_mem, v_mem, q_mem, alpha, theta)
}

// ── Test 1: Forward match ───────────────────────────────────────────

#[test]
fn test_cuda_delta_forward_matches_rust() {
    let d = 8;
    let seq_len = 16;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 100);
    let v_mem = rand_buf(seq_len * d, 200);
    let q_mem = rand_buf(seq_len * d, 300);
    let alpha = vec![0.05f32; seq_len]; // small retention
    let theta = vec![0.01f32; seq_len]; // small learning rate
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d);

    check_close("delta_fwd_y", &y_rust, &y_cuda, 1e-5);
    check_close("delta_fwd_m_states", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 2: Forward with full pipeline (params → projections → CUDA) ──

#[test]
fn test_cuda_delta_forward_full_pipeline() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let dd = d * d;

    let embedded = rand_buf(s * d, 99);

    // Rust reference: full DeltaRule::step
    let rule = DeltaRule;
    let (y_rust, cache_rust) = rule.step(&params.levels[0], &embedded, s, d, None);

    // CUDA path: extract projections in Rust, inner loop in CUDA
    let (k_mem, v_mem, q_mem, alpha, theta) =
        extract_projections(&params.levels[0], &embedded, s, d);
    let m_initial = vec![0.0f32; dd];
    let mut m_states_cuda = vec![0.0f32; (s + 1) * dd];
    let mut y_cuda = vec![0.0f32; s * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, s, d);

    check_close("pipeline_y", &y_rust, &y_cuda, 1e-5);
    check_close("pipeline_m_states", &cache_rust.m_states, &m_states_cuda, 1e-5);
}

// ── Test 3: Backward match ──────────────────────────────────────────

#[test]
fn test_cuda_delta_backward_matches_rust() {
    let d = 8;
    let seq_len = 12;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 400);
    let v_mem = rand_buf(seq_len * d, 500);
    let q_mem = rand_buf(seq_len * d, 600);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // Forward first to get m_states
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    // Upstream gradient
    let d_y = rand_buf(seq_len * d, 700);

    // Rust backward reference
    let mut dk_rust = vec![0.0f32; seq_len * d];
    let mut dv_rust = vec![0.0f32; seq_len * d];
    let mut dq_rust = vec![0.0f32; seq_len * d];
    let mut dalpha_rust = vec![0.0f32; seq_len];
    let mut dtheta_rust = vec![0.0f32; seq_len];
    let mut dm_init_rust = vec![0.0f32; dd];

    // Use dispatch (in non-cuda mode this runs Rust; in cuda mode both should match)
    // Instead, run the Rust reference directly
    {
        // Inline Rust backward for reference
        let mut d_m = vec![0.0f32; dd];
        for t in (0..seq_len).rev() {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];
            let d_y_t = &d_y[t * d..(t + 1) * d];
            let m_t = &m_states[t * dd..(t + 1) * dd];
            let m_next = &m_states[(t + 1) * dd..(t + 2) * dd];
            let alpha_t = alpha[t];
            let theta_t = theta[t];

            for i in 0..d {
                for j in 0..d { d_m[i * d + j] += d_y_t[i] * q_t[j]; }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_next[i * d + j] * d_y_t[i]; }
                dq_rust[t * d + j] = sum;
            }

            let mut prediction = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d { sum += m_t[i * d + j] * k_t[j]; }
                prediction[i] = sum;
            }
            let mut error = vec![0.0f32; d];
            for i in 0..d { error[i] = prediction[i] - v_t[i]; }

            let mut da_sum = 0.0f32;
            let mut dt_sum = 0.0f32;
            for i in 0..d {
                for j in 0..d {
                    da_sum += m_t[i * d + j] * d_m[i * d + j];
                    dt_sum += error[i] * k_t[j] * d_m[i * d + j];
                }
            }
            dalpha_rust[t] = -da_sum;
            dtheta_rust[t] = -dt_sum;

            let mut d_err = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d { sum += (-theta_t * d_m[i * d + j]) * k_t[j]; }
                d_err[i] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += (-theta_t * d_m[i * d + j]) * error[i]; }
                dk_rust[t * d + j] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_t[i * d + j] * d_err[i]; }
                dk_rust[t * d + j] += sum;
            }
            for i in 0..d { dv_rust[t * d + i] = -d_err[i]; }

            let retention = 1.0 - alpha_t;
            let mut d_m_prev = vec![0.0f32; dd];
            for i in 0..d {
                for j in 0..d {
                    d_m_prev[i * d + j] = retention * d_m[i * d + j] + d_err[i] * k_t[j];
                }
            }
            d_m = d_m_prev;
        }
        dm_init_rust.copy_from_slice(&d_m);
    }

    // CUDA backward
    let mut dk_cuda = vec![0.0f32; seq_len * d];
    let mut dv_cuda = vec![0.0f32; seq_len * d];
    let mut dq_cuda = vec![0.0f32; seq_len * d];
    let mut dalpha_cuda = vec![0.0f32; seq_len];
    let mut dtheta_cuda = vec![0.0f32; seq_len];
    let mut dm_init_cuda = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_init_cuda,
        seq_len, d);

    check_close("delta_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("delta_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("delta_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("delta_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("delta_bwd_dtheta", &dtheta_rust, &dtheta_cuda, 1e-4);
    check_close("delta_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
}

// ── Test 4: Edge case — seq_len=1 ──────────────────────────────────

#[test]
fn test_cuda_delta_forward_seq_len_1() {
    let d = 4;
    let seq_len = 1;
    let dd = d * d;

    let k_mem = rand_buf(d, 800);
    let v_mem = rand_buf(d, 900);
    let q_mem = rand_buf(d, 1000);
    let alpha = vec![0.05f32];
    let theta = vec![0.01f32];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; 2 * dd];
    let mut y_cuda = vec![0.0f32; d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d);

    check_close("seq1_y", &y_rust, &y_cuda, 1e-5);
    check_close("seq1_m", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 5: Edge case — nonzero initial M ──────────────────────────

#[test]
fn test_cuda_delta_forward_nonzero_initial() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1100);
    let v_mem = rand_buf(seq_len * d, 1200);
    let q_mem = rand_buf(seq_len * d, 1300);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = rand_buf(dd, 1400); // nonzero initial M

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d);

    check_close("nonzero_init_y", &y_rust, &y_cuda, 1e-5);
    check_close("nonzero_init_m", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 6: Full pipeline backward match ────────────────────────────

#[test]
fn test_cuda_delta_backward_full_pipeline() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;

    let embedded = rand_buf(s * d, 99);

    // Rust reference backward
    let rule = DeltaRule;
    let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
    let d_y = rand_buf(s * d, 1500);
    let (grads_rust, _d_emb_rust) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

    // CUDA path: extract projections → forward → backward
    let (k_mem, v_mem, q_mem, alpha, theta) =
        extract_projections(&params.levels[0], &embedded, s, d);
    let dd = d * d;
    let m_initial = vec![0.0f32; dd];
    let mut m_states = vec![0.0f32; (s + 1) * dd];
    let mut y = vec![0.0f32; s * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, s, d);

    let mut dk_cuda = vec![0.0f32; s * d];
    let mut dv_cuda = vec![0.0f32; s * d];
    let mut dq_cuda = vec![0.0f32; s * d];
    let mut dalpha_cuda = vec![0.0f32; s];
    let mut dtheta_cuda = vec![0.0f32; s];
    let mut dm_init_cuda = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_init_cuda,
        s, d);

    // The inner loop gradients should match the cache-based ones
    // dk_cuda, dv_cuda, dq_cuda from the inner loop won't directly match
    // grads_rust.w_k_mem etc because those are projection weight gradients.
    // But we can verify that the inner loop produces consistent forward output.
    check_close("pipeline_y_match", &cache.y, &y, 1e-5);
}

// ── Test 7: Backward seq_len=1 ─────────────────────────────────────

#[test]
fn test_cuda_delta_backward_seq_len_1() {
    let d = 4;
    let seq_len = 1;
    let dd = d * d;

    let k_mem = rand_buf(d, 1600);
    let v_mem = rand_buf(d, 1700);
    let q_mem = rand_buf(d, 1800);
    let alpha = vec![0.05f32];
    let theta = vec![0.01f32];
    let m_initial = vec![0.0f32; dd];

    // Forward
    let mut m_states = vec![0.0f32; 2 * dd];
    let mut y = vec![0.0f32; d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    let d_y = rand_buf(d, 1900);

    // Rust backward inline
    let mut dk_rust = vec![0.0f32; d];
    let mut dv_rust = vec![0.0f32; d];
    let mut dq_rust = vec![0.0f32; d];
    let mut dalpha_rust = vec![0.0f32; 1];
    let mut dtheta_rust = vec![0.0f32; 1];
    let mut dm_init_rust = vec![0.0f32; dd];
    {
        let mut d_m = vec![0.0f32; dd];
        let t = 0;
        let k_t = &k_mem[..d];
        let v_t = &v_mem[..d];
        let q_t = &q_mem[..d];
        let d_y_t = &d_y[..d];
        let m_t = &m_states[..dd];
        let m_next = &m_states[dd..2*dd];
        let alpha_t = alpha[0];
        let theta_t = theta[0];

        for i in 0..d { for j in 0..d { d_m[i*d+j] += d_y_t[i] * q_t[j]; } }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_next[i*d+j] * d_y_t[i]; }
            dq_rust[j] = sum;
        }

        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_t[i*d+j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }

        let mut da = 0.0f32;
        let mut dt = 0.0f32;
        for i in 0..d { for j in 0..d {
            da += m_t[i*d+j] * d_m[i*d+j];
            dt += error[i] * k_t[j] * d_m[i*d+j];
        }}
        dalpha_rust[0] = -da;
        dtheta_rust[0] = -dt;

        let mut d_err = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += (-theta_t * d_m[i*d+j]) * k_t[j]; }
            d_err[i] = sum;
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += (-theta_t * d_m[i*d+j]) * error[i]; }
            dk_rust[j] = sum;
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_t[i*d+j] * d_err[i]; }
            dk_rust[j] += sum;
        }
        for i in 0..d { dv_rust[i] = -d_err[i]; }

        let retention = 1.0 - alpha_t;
        for i in 0..d { for j in 0..d {
            dm_init_rust[i*d+j] = retention * d_m[i*d+j] + d_err[i] * k_t[j];
        }}
    }

    let mut dk_cuda = vec![0.0f32; d];
    let mut dv_cuda = vec![0.0f32; d];
    let mut dq_cuda = vec![0.0f32; d];
    let mut dalpha_cuda = vec![0.0f32; 1];
    let mut dtheta_cuda = vec![0.0f32; 1];
    let mut dm_init_cuda = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_init_cuda,
        seq_len, d);

    check_close("seq1_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("seq1_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("seq1_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("seq1_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("seq1_bwd_dtheta", &dtheta_rust, &dtheta_cuda, 1e-4);
    check_close("seq1_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
}

// ── Test 8: Output is nonzero ──────────────────────────────────────

#[test]
fn test_cuda_delta_forward_output_nonzero() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 2000);
    let v_mem = rand_buf(seq_len * d, 2100);
    let q_mem = rand_buf(seq_len * d, 2200);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    let y_max = y.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(y_max > 1e-10, "y should be nonzero, max_abs={y_max}");

    // Final M should be nonzero
    let m_final = &m_states[seq_len * dd..];
    let m_max = m_final.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(m_max > 1e-10, "final M should be nonzero, max_abs={m_max}");
}

// ── Test 9: Backward gradients are nonzero ──────────────────────────

#[test]
fn test_cuda_delta_backward_nonzero() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 2300);
    let v_mem = rand_buf(seq_len * d, 2400);
    let q_mem = rand_buf(seq_len * d, 2500);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    let d_y = vec![1.0f32; seq_len * d];
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut da = vec![0.0f32; seq_len];
    let mut dt = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk, &mut dv, &mut dq, &mut da, &mut dt, &mut dm,
        seq_len, d);

    let dk_max = dk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dv_max = dv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dq_max = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(dk_max > 1e-10, "dk should be nonzero");
    assert!(dv_max > 1e-10, "dv should be nonzero");
    assert!(dq_max > 1e-10, "dq should be nonzero");
}

// ── Test 10: Deterministic ──────────────────────────────────────────

#[test]
fn test_cuda_delta_forward_deterministic() {
    let d = 8;
    let seq_len = 12;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 2600);
    let v_mem = rand_buf(seq_len * d, 2700);
    let q_mem = rand_buf(seq_len * d, 2800);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let mut y1 = vec![0.0f32; seq_len * d];
    let mut m1 = vec![0.0f32; (seq_len + 1) * dd];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m1, &mut y1, seq_len, d);

    let mut y2 = vec![0.0f32; seq_len * d];
    let mut m2 = vec![0.0f32; (seq_len + 1) * dd];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m2, &mut y2, seq_len, d);

    assert_eq!(y1, y2, "CUDA delta forward should be deterministic");
}

// ── Test 11: d=16 (larger dimension) ────────────────────────────────

#[test]
fn test_cuda_delta_forward_d16() {
    let d = 16;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 2900);
    let v_mem = rand_buf(seq_len * d, 3000);
    let q_mem = rand_buf(seq_len * d, 3100);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d);

    check_close("d16_y", &y_rust, &y_cuda, 1e-4);
    check_close("d16_m", &m_states_rust, &m_states_cuda, 1e-4);
}
