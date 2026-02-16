// CUDA TitansLMM Kernel Tests — S2-M1 Phase 2
//
// Tests the CUDA inner-loop kernel for Titans LMM (GD + momentum).
// All fp32. Forward tol 1e-5, backward tol 1e-4.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::dispatch::{titans_forward_dispatch, titans_backward_dispatch};

/// Rust reference Titans forward inner loop.
fn rust_titans_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], eta: &[f32],
    m_initial: &[f32], s_initial: &[f32],
    seq_len: usize, d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dd = d * d;
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    m_states[..dd].copy_from_slice(m_initial);
    s_states[..dd].copy_from_slice(s_initial);

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;
        let s_t = t * dd;
        let s_next = (t + 1) * dd;

        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        for i in 0..d {
            let err_i = prediction[i] - v_t[i];
            for j in 0..d {
                s_states[s_next + i * d + j] =
                    eta[t] * s_states[s_t + i * d + j] - theta[t] * err_i * k_t[j];
            }
        }

        let retention = 1.0 - alpha[t];
        for i in 0..dd {
            m_states[m_next + i] = retention * m_states[m_t + i] + s_states[s_next + i];
        }

        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }

    (m_states, s_states, y)
}

// ── Forward tests ──────────────────────────────────────────────────

#[test]
fn test_cuda_titans_forward_matches_rust() {
    let d = 8;
    let seq_len = 16;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 100);
    let v_mem = rand_buf(seq_len * d, 200);
    let q_mem = rand_buf(seq_len * d, 300);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    let (m_rust, s_rust, y_rust) = rust_titans_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d);

    let mut m_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m_cuda, &mut s_cuda, &mut y_cuda,
        seq_len, d);

    check_close("titans_fwd_y", &y_rust, &y_cuda, 1e-5);
    check_close("titans_fwd_m", &m_rust, &m_cuda, 1e-5);
    check_close("titans_fwd_s", &s_rust, &s_cuda, 1e-5);
}

#[test]
fn test_cuda_titans_forward_seq_len_1() {
    let d = 4;
    let dd = d * d;

    let k_mem = rand_buf(d, 400);
    let v_mem = rand_buf(d, 500);
    let q_mem = rand_buf(d, 600);
    let alpha = vec![0.05f32];
    let theta = vec![0.01f32];
    let eta = vec![0.9f32];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    let (m_rust, s_rust, y_rust) = rust_titans_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, 1, d);

    let mut m_cuda = vec![0.0f32; 2 * dd];
    let mut s_cuda = vec![0.0f32; 2 * dd];
    let mut y_cuda = vec![0.0f32; d];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m_cuda, &mut s_cuda, &mut y_cuda,
        1, d);

    check_close("titans_seq1_y", &y_rust, &y_cuda, 1e-5);
}

#[test]
fn test_cuda_titans_forward_momentum_nonzero() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 700);
    let v_mem = rand_buf(seq_len * d, 800);
    let q_mem = rand_buf(seq_len * d, 900);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m_states, &mut s_states, &mut y,
        seq_len, d);

    let s_final = &s_states[seq_len * dd..];
    let s_max = s_final.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(s_max > 1e-10, "final S should be nonzero (momentum), max_abs={s_max}");
}

// ── Backward tests ─────────────────────────────────────────────────

#[test]
fn test_cuda_titans_backward_matches_rust() {
    let d = 8;
    let seq_len = 12;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1000);
    let v_mem = rand_buf(seq_len * d, 1100);
    let q_mem = rand_buf(seq_len * d, 1200);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    // Forward
    let (m_states, s_states, _y) = rust_titans_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d);

    let d_y = rand_buf(seq_len * d, 1300);

    // Rust backward reference (inline)
    let mut dk_rust = vec![0.0f32; seq_len * d];
    let mut dv_rust = vec![0.0f32; seq_len * d];
    let mut dq_rust = vec![0.0f32; seq_len * d];
    let mut dalpha_rust = vec![0.0f32; seq_len];
    let mut dtheta_rust = vec![0.0f32; seq_len];
    let mut deta_rust = vec![0.0f32; seq_len];
    let mut dm_init_rust = vec![0.0f32; dd];
    let mut ds_init_rust = vec![0.0f32; dd];

    {
        let mut d_m = vec![0.0f32; dd];
        let mut d_s = vec![0.0f32; dd];

        for t in (0..seq_len).rev() {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];
            let d_y_t = &d_y[t * d..(t + 1) * d];
            let m_t = &m_states[t * dd..(t + 1) * dd];
            let m_next = &m_states[(t + 1) * dd..(t + 2) * dd];
            let s_t = &s_states[t * dd..(t + 1) * dd];

            for i in 0..d { for j in 0..d { d_m[i*d+j] += d_y_t[i] * q_t[j]; } }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_next[i*d+j] * d_y_t[i]; }
                dq_rust[t*d+j] = sum;
            }

            for i in 0..dd { d_s[i] += d_m[i]; }

            let mut da = 0.0f32;
            for i in 0..dd { da += m_t[i] * d_m[i]; }
            dalpha_rust[t] = -da;

            let retention = 1.0 - alpha[t];
            let mut d_m_prev = vec![0.0f32; dd];
            for i in 0..dd { d_m_prev[i] = retention * d_m[i]; }

            let mut de = 0.0f32;
            for i in 0..dd { de += s_t[i] * d_s[i]; }
            deta_rust[t] = de;

            let mut prediction = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d { sum += m_t[i*d+j] * k_t[j]; }
                prediction[i] = sum;
            }
            let mut error = vec![0.0f32; d];
            for i in 0..d { error[i] = prediction[i] - v_t[i]; }

            let mut dth = 0.0f32;
            for i in 0..d { for j in 0..d { dth += error[i] * k_t[j] * d_s[i*d+j]; } }
            dtheta_rust[t] = -dth;

            let mut d_err = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d { sum += (-theta[t] * d_s[i*d+j]) * k_t[j]; }
                d_err[i] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += (-theta[t] * d_s[i*d+j]) * error[i]; }
                dk_rust[t*d+j] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_t[i*d+j] * d_err[i]; }
                dk_rust[t*d+j] += sum;
            }
            for i in 0..d { dv_rust[t*d+i] = -d_err[i]; }

            let mut d_s_prev = vec![0.0f32; dd];
            for i in 0..dd { d_s_prev[i] = eta[t] * d_s[i]; }

            for i in 0..d { for j in 0..d {
                d_m_prev[i*d+j] += d_err[i] * k_t[j];
            }}

            d_m = d_m_prev;
            d_s = d_s_prev;
        }
        dm_init_rust.copy_from_slice(&d_m);
        ds_init_rust.copy_from_slice(&d_s);
    }

    // CUDA backward
    let mut dk_cuda = vec![0.0f32; seq_len * d];
    let mut dv_cuda = vec![0.0f32; seq_len * d];
    let mut dq_cuda = vec![0.0f32; seq_len * d];
    let mut dalpha_cuda = vec![0.0f32; seq_len];
    let mut dtheta_cuda = vec![0.0f32; seq_len];
    let mut deta_cuda = vec![0.0f32; seq_len];
    let mut dm_init_cuda = vec![0.0f32; dd];
    let mut ds_init_cuda = vec![0.0f32; dd];

    titans_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_states, &s_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut deta_cuda,
        &mut dm_init_cuda, &mut ds_init_cuda,
        seq_len, d);

    check_close("titans_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("titans_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("titans_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("titans_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("titans_bwd_dtheta", &dtheta_rust, &dtheta_cuda, 1e-4);
    check_close("titans_bwd_deta", &deta_rust, &deta_cuda, 1e-4);
    check_close("titans_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
    check_close("titans_bwd_ds_init", &ds_init_rust, &ds_init_cuda, 1e-4);
}

#[test]
fn test_cuda_titans_backward_nonzero() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1400);
    let v_mem = rand_buf(seq_len * d, 1500);
    let q_mem = rand_buf(seq_len * d, 1600);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m_states, &mut s_states, &mut y,
        seq_len, d);

    let d_y = vec![1.0f32; seq_len * d];
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut da = vec![0.0f32; seq_len];
    let mut dt = vec![0.0f32; seq_len];
    let mut de = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];
    let mut ds = vec![0.0f32; dd];

    titans_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_states, &s_states, &d_y,
        &mut dk, &mut dv, &mut dq, &mut da, &mut dt, &mut de,
        &mut dm, &mut ds, seq_len, d);

    let dk_max = dk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dv_max = dv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dq_max = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(dk_max > 1e-10, "titans dk should be nonzero");
    assert!(dv_max > 1e-10, "titans dv should be nonzero");
    assert!(dq_max > 1e-10, "titans dq should be nonzero");
}

#[test]
fn test_cuda_titans_forward_deterministic() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1700);
    let v_mem = rand_buf(seq_len * d, 1800);
    let q_mem = rand_buf(seq_len * d, 1900);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    let mut y1 = vec![0.0f32; seq_len * d];
    let mut m1 = vec![0.0f32; (seq_len + 1) * dd];
    let mut s1 = vec![0.0f32; (seq_len + 1) * dd];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m1, &mut s1, &mut y1,
        seq_len, d);

    let mut y2 = vec![0.0f32; seq_len * d];
    let mut m2 = vec![0.0f32; (seq_len + 1) * dd];
    let mut s2 = vec![0.0f32; (seq_len + 1) * dd];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, &mut m2, &mut s2, &mut y2,
        seq_len, d);

    assert_eq!(y1, y2, "CUDA titans forward should be deterministic");
}
