// CUDA HebbianRule Kernel Tests — S2-M1 Phase 3
//
// Tests the CUDA inner-loop kernel for the Hebbian memory rule.
// Simplest rule: no error correction, no theta.
// All fp32. Forward tol 1e-5, backward tol 1e-4.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::dispatch::{hebbian_forward_dispatch, hebbian_backward_dispatch};

/// Rust reference Hebbian forward inner loop.
fn rust_hebbian_forward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], m_initial: &[f32],
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
        let retention = 1.0 - alpha[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        for i in 0..d {
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] + v_t[i] * k_t[j];
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

// ── Forward tests ──────────────────────────────────────────────────

#[test]
fn test_cuda_hebbian_forward_matches_rust() {
    let d = 8;
    let seq_len = 16;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 100);
    let v_mem = rand_buf(seq_len * d, 200);
    let q_mem = rand_buf(seq_len * d, 300);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_rust, y_rust) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, seq_len, d);

    let mut m_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_cuda, &mut y_cuda, seq_len, d);

    check_close("hebbian_fwd_y", &y_rust, &y_cuda, 1e-5);
    check_close("hebbian_fwd_m", &m_rust, &m_cuda, 1e-5);
}

#[test]
fn test_cuda_hebbian_forward_seq_len_1() {
    let d = 4;
    let dd = d * d;

    let k_mem = rand_buf(d, 400);
    let v_mem = rand_buf(d, 500);
    let q_mem = rand_buf(d, 600);
    let alpha = vec![0.05f32];
    let m_initial = vec![0.0f32; dd];

    let (_m_rust, y_rust) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, 1, d);

    let mut m_cuda = vec![0.0f32; 2 * dd];
    let mut y_cuda = vec![0.0f32; d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_cuda, &mut y_cuda, 1, d);

    check_close("hebbian_seq1_y", &y_rust, &y_cuda, 1e-5);
}

#[test]
fn test_cuda_hebbian_forward_nonzero_initial() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 700);
    let v_mem = rand_buf(seq_len * d, 800);
    let q_mem = rand_buf(seq_len * d, 900);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = rand_buf(dd, 1000);

    let (m_rust, y_rust) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, seq_len, d);

    let mut m_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_cuda, &mut y_cuda, seq_len, d);

    check_close("hebbian_nonzero_y", &y_rust, &y_cuda, 1e-5);
    check_close("hebbian_nonzero_m", &m_rust, &m_cuda, 1e-5);
}

// ── Backward tests ─────────────────────────────────────────────────

#[test]
fn test_cuda_hebbian_backward_matches_rust() {
    let d = 8;
    let seq_len = 12;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1100);
    let v_mem = rand_buf(seq_len * d, 1200);
    let q_mem = rand_buf(seq_len * d, 1300);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states, _y) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, seq_len, d);

    let d_y = rand_buf(seq_len * d, 1400);

    // Rust backward reference
    let mut dk_rust = vec![0.0f32; seq_len * d];
    let mut dv_rust = vec![0.0f32; seq_len * d];
    let mut dq_rust = vec![0.0f32; seq_len * d];
    let mut dalpha_rust = vec![0.0f32; seq_len];
    let mut dm_init_rust = vec![0.0f32; dd];
    {
        let mut d_m = vec![0.0f32; dd];
        for t in (0..seq_len).rev() {
            let k_t = &k_mem[t*d..(t+1)*d];
            let v_t = &v_mem[t*d..(t+1)*d];
            let q_t = &q_mem[t*d..(t+1)*d];
            let d_y_t = &d_y[t*d..(t+1)*d];
            let m_t = &m_states[t*dd..(t+1)*dd];
            let m_next = &m_states[(t+1)*dd..(t+2)*dd];

            for i in 0..d { for j in 0..d { d_m[i*d+j] += d_y_t[i] * q_t[j]; } }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_next[i*d+j] * d_y_t[i]; }
                dq_rust[t*d+j] = sum;
            }

            let mut da = 0.0f32;
            for i in 0..dd { da += m_t[i] * d_m[i]; }
            dalpha_rust[t] = -da;

            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d { sum += d_m[i*d+j] * k_t[j]; }
                dv_rust[t*d+i] = sum;
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += d_m[i*d+j] * v_t[i]; }
                dk_rust[t*d+j] = sum;
            }

            let retention = 1.0 - alpha[t];
            for i in 0..dd { d_m[i] = retention * d_m[i]; }
        }
        dm_init_rust.copy_from_slice(&d_m);
    }

    // CUDA backward
    let mut dk_cuda = vec![0.0f32; seq_len * d];
    let mut dv_cuda = vec![0.0f32; seq_len * d];
    let mut dq_cuda = vec![0.0f32; seq_len * d];
    let mut dalpha_cuda = vec![0.0f32; seq_len];
    let mut dm_init_cuda = vec![0.0f32; dd];

    hebbian_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dm_init_cuda,
        seq_len, d);

    check_close("hebbian_bwd_dk", &dk_rust, &dk_cuda, 1e-4);
    check_close("hebbian_bwd_dv", &dv_rust, &dv_cuda, 1e-4);
    check_close("hebbian_bwd_dq", &dq_rust, &dq_cuda, 1e-4);
    check_close("hebbian_bwd_dalpha", &dalpha_rust, &dalpha_cuda, 1e-4);
    check_close("hebbian_bwd_dm_init", &dm_init_rust, &dm_init_cuda, 1e-4);
}

#[test]
fn test_cuda_hebbian_backward_nonzero() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1500);
    let v_mem = rand_buf(seq_len * d, 1600);
    let q_mem = rand_buf(seq_len * d, 1700);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    let d_y = vec![1.0f32; seq_len * d];
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut da = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];

    hebbian_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_states, &d_y,
        &mut dk, &mut dv, &mut dq, &mut da, &mut dm,
        seq_len, d);

    let dk_max = dk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dv_max = dv.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let dq_max = dq.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(dk_max > 1e-10, "hebbian dk should be nonzero");
    assert!(dv_max > 1e-10, "hebbian dv should be nonzero");
    assert!(dq_max > 1e-10, "hebbian dq should be nonzero");
}

#[test]
fn test_cuda_hebbian_forward_deterministic() {
    let d = 8;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 1800);
    let v_mem = rand_buf(seq_len * d, 1900);
    let q_mem = rand_buf(seq_len * d, 2000);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let mut y1 = vec![0.0f32; seq_len * d];
    let mut m1 = vec![0.0f32; (seq_len + 1) * dd];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m1, &mut y1, seq_len, d);

    let mut y2 = vec![0.0f32; seq_len * d];
    let mut m2 = vec![0.0f32; (seq_len + 1) * dd];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m2, &mut y2, seq_len, d);

    assert_eq!(y1, y2, "CUDA hebbian forward should be deterministic");
}

#[test]
fn test_cuda_hebbian_forward_d16() {
    let d = 16;
    let seq_len = 8;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 2100);
    let v_mem = rand_buf(seq_len * d, 2200);
    let q_mem = rand_buf(seq_len * d, 2300);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_rust, y_rust) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, seq_len, d);

    let mut m_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_cuda, &mut y_cuda, seq_len, d);

    check_close("hebbian_d16_y", &y_rust, &y_cuda, 1e-4);
    check_close("hebbian_d16_m", &m_rust, &m_cuda, 1e-4);
}
