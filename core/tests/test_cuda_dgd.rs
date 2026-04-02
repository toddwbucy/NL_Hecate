// CUDA DGD (Delta Gradient Descent) Kernel Tests — S3b-M5
//
// Tests the CUDA inner-loop kernel for DGD (HOPE inner-loop optimizer).
// Math is identical to Delta Rule at L2 bias, but uses separate kernel
// files to allow future bias-agnostic divergence (CS-33).
//
// Test classes:
//   1. Forward match: CUDA vs Rust reference inner loop (tol 1e-5)
//   2. Backward match: all 6 gradient arrays (tol 1e-4)
//   3. Edge cases (seq_len=1, nonzero initial M)
//   4. Output nonzero sanity checks
//   5. Deterministic forward
//   6. d=16 larger dimension

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::dispatch::{dgd_forward_dispatch, dgd_backward_dispatch};

/// Run the Rust reference inner loop (identical to Delta Rule L2 inner loop).
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

// ── Test 1: Forward match ───────────────────────────────────────────

#[test]
fn test_cuda_dgd_forward_matches_rust() {
    let d = 8;
    let seq_len = 16;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 100);
    let v_mem = rand_buf(seq_len * d, 200);
    let q_mem = rand_buf(seq_len * d, 300);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, 0.0);

    check_close("dgd_fwd_y", &y_rust, &y_cuda, 1e-5);
    check_close("dgd_fwd_m_states", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 2: Backward match ──────────────────────────────────────────

#[test]
fn test_cuda_dgd_backward_matches_rust() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d, 0.0);

    // Upstream gradient
    let d_y = rand_buf(seq_len * d, 700);

    // Rust backward reference (inline)
    let mut dk_rust = vec![0.0f32; seq_len * d];
    let mut dv_rust = vec![0.0f32; seq_len * d];
    let mut dq_rust = vec![0.0f32; seq_len * d];
    let mut dalpha_rust = vec![0.0f32; seq_len];
    let mut dtheta_rust = vec![0.0f32; seq_len];
    let mut dm_init_rust = vec![0.0f32; dd];

    {
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

    dgd_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_init_cuda,
        seq_len, d, 0.0);

    check_close("dgd_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("dgd_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("dgd_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("dgd_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("dgd_bwd_dtheta", &dtheta_rust, &dtheta_cuda, 1e-4);
    check_close("dgd_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
}

// ── Test 3: Edge case — seq_len=1 ──────────────────────────────────

#[test]
fn test_cuda_dgd_forward_seq_len_1() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, 0.0);

    check_close("seq1_y", &y_rust, &y_cuda, 1e-5);
    check_close("seq1_m", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 4: Edge case — nonzero initial M ──────────────────────────

#[test]
fn test_cuda_dgd_forward_nonzero_initial() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1100);
    let v_mem = rand_buf(seq_len * d, 1200);
    let q_mem = rand_buf(seq_len * d, 1300);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = rand_buf(dd, 1400);

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, 0.0);

    check_close("nonzero_init_y", &y_rust, &y_cuda, 1e-5);
    check_close("nonzero_init_m", &m_states_rust, &m_states_cuda, 1e-5);
}

// ── Test 5: Backward seq_len=1 ─────────────────────────────────────

#[test]
fn test_cuda_dgd_backward_seq_len_1() {
    let d = 4;
    let seq_len = 1;
    let dd = d * d;

    let k_mem = rand_buf(d, 1600);
    let v_mem = rand_buf(d, 1700);
    let q_mem = rand_buf(d, 1800);
    let alpha = vec![0.05f32];
    let theta = vec![0.01f32];
    let m_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; 2 * dd];
    let mut y = vec![0.0f32; d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d, 0.0);

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

    dgd_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_init_cuda,
        seq_len, d, 0.0);

    check_close("seq1_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("seq1_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("seq1_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("seq1_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("seq1_bwd_dtheta", &dtheta_rust, &dtheta_cuda, 1e-4);
    check_close("seq1_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
}

// ── Test 6: Output is nonzero ──────────────────────────────────────

#[test]
fn test_cuda_dgd_forward_output_nonzero() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d, 0.0);

    let y_max = y.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(y_max > 1e-10, "y should be nonzero, max_abs={y_max}");

    let m_final = &m_states[seq_len * dd..];
    let m_max = m_final.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(m_max > 1e-10, "final M should be nonzero, max_abs={m_max}");
}

// ── Test 7: Backward gradients are nonzero ──────────────────────────

#[test]
fn test_cuda_dgd_backward_nonzero() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d, 0.0);

    let d_y = vec![1.0f32; seq_len * d];
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut da = vec![0.0f32; seq_len];
    let mut dt = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];

    dgd_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk, &mut dv, &mut dq, &mut da, &mut dt, &mut dm,
        seq_len, d, 0.0);

    let dk_max = dk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dv_max = dv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dq_max = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(dk_max > 1e-10, "dk should be nonzero");
    assert!(dv_max > 1e-10, "dv should be nonzero");
    assert!(dq_max > 1e-10, "dq should be nonzero");
}

// ── Test 8: Deterministic ──────────────────────────────────────────

#[test]
fn test_cuda_dgd_forward_deterministic() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m1, &mut y1, seq_len, d, 0.0);

    let mut y2 = vec![0.0f32; seq_len * d];
    let mut m2 = vec![0.0f32; (seq_len + 1) * dd];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m2, &mut y2, seq_len, d, 0.0);

    assert_eq!(y1, y2, "CUDA DGD forward should be deterministic");
}

// ── Test 9: d=16 (larger dimension) ────────────────────────────────

#[test]
fn test_cuda_dgd_forward_d16() {
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
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, 0.0);

    check_close("d16_y", &y_rust, &y_cuda, 1e-4);
    check_close("d16_m", &m_states_rust, &m_states_cuda, 1e-4);
}

// ── Test 10: Cross-validate DGD dispatch against Delta dispatch ─────

#[test]
fn test_cuda_dgd_matches_delta_dispatch() {
    use nl_hecate_core::dispatch::{delta_forward_dispatch, delta_backward_dispatch};

    let d = 8;
    let seq_len = 10;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 3200);
    let v_mem = rand_buf(seq_len * d, 3300);
    let q_mem = rand_buf(seq_len * d, 3400);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // Forward: DGD vs Delta
    let mut m_dgd = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_dgd = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_dgd, &mut y_dgd, seq_len, d, 0.0);

    let mut m_delta = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_delta = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_delta, &mut y_delta, seq_len, d, 0.0, f32::MAX);

    check_close("dgd_vs_delta_y", &y_dgd, &y_delta, 1e-5);
    check_close("dgd_vs_delta_m", &m_dgd, &m_delta, 1e-5);

    // Backward: DGD vs Delta
    let d_y = rand_buf(seq_len * d, 3500);

    let mut dk_dgd = vec![0.0f32; seq_len * d];
    let mut dv_dgd = vec![0.0f32; seq_len * d];
    let mut dq_dgd = vec![0.0f32; seq_len * d];
    let mut da_dgd = vec![0.0f32; seq_len];
    let mut dt_dgd = vec![0.0f32; seq_len];
    let mut dm_dgd = vec![0.0f32; dd];

    dgd_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_dgd, &d_y,
        &mut dk_dgd, &mut dv_dgd, &mut dq_dgd,
        &mut da_dgd, &mut dt_dgd, &mut dm_dgd,
        seq_len, d, 0.0);

    let mut dk_delta = vec![0.0f32; seq_len * d];
    let mut dv_delta = vec![0.0f32; seq_len * d];
    let mut dq_delta = vec![0.0f32; seq_len * d];
    let mut da_delta = vec![0.0f32; seq_len];
    let mut dt_delta = vec![0.0f32; seq_len];
    let mut dm_delta = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_delta, &d_y,
        &mut dk_delta, &mut dv_delta, &mut dq_delta,
        &mut da_delta, &mut dt_delta, &mut dm_delta,
        seq_len, d, 0.0);

    check_close("dgd_vs_delta_dk", &dk_dgd, &dk_delta, 1e-4);
    check_close("dgd_vs_delta_dv", &dv_dgd, &dv_delta, 1e-4);
    check_close("dgd_vs_delta_dq", &dq_dgd, &dq_delta, 1e-4);
    check_close("dgd_vs_delta_da", &da_dgd, &da_delta, 1e-4);
    check_close("dgd_vs_delta_dt", &dt_dgd, &dt_delta, 1e-4);
    check_close("dgd_vs_delta_dm", &dm_dgd, &dm_delta, 1e-4);
}

// ── Large-d test (validates shared memory overflow fix) ───────────

#[test]
fn test_cuda_dgd_forward_large_d() {
    // d=128 → M[128*128] = 64KB, would have overflowed shared memory (limit 48KB)
    let d = 128;
    let seq_len = 4;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 5000);
    let v_mem = rand_buf(seq_len * d, 5100);
    let q_mem = rand_buf(seq_len * d, 5200);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_inner_loop_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, 0.0);

    check_close("dgd_large_d_y", &y_rust, &y_cuda, 1e-4);
    check_close("dgd_large_d_m", &m_states_rust, &m_states_cuda, 1e-4);

    // Verify M is non-zero
    let m_final = &m_states_cuda[seq_len * dd..];
    let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(m_norm > 1e-6, "M_final should be non-zero at d=128, got ‖M‖={m_norm}");
}
