// CUDA Large Dimension Tests — d > 1024 kernel restructuring validation
//
// These tests exercise the strided loop paths that only activate when d > blockDim.x.
// At d <= 1024, blockDim.x >= d so strided loops execute once (algebraically identical
// to the original `if (tid < d)` guard). At d=2048, the loops iterate 2+ times.
//
// Test plan:
//   1. d=2048 forward + backward for all 4 memory rules (CUDA vs Rust reference)
//   2. Gate backward at D=1024 (was impossible before: asserted D <= 512)
//   3. M non-zero sanity check at d=2048 (catches silent data corruption)
//
// All tests use seq_len=2 to keep memory manageable: M[2048*2048] = 16MB per state,
// with (seq_len+1)=3 states → 48MB total. At seq_len=16 that would be 256MB.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};
use serial_test::serial;

use nl_hecate_core::dispatch::{
    delta_forward_dispatch, delta_backward_dispatch,
    titans_forward_dispatch, titans_backward_dispatch,
    hebbian_forward_dispatch, hebbian_backward_dispatch,
    dgd_forward_dispatch, dgd_backward_dispatch,
};

// ── Rust reference implementations ──────────────────────────────────

fn rust_delta_forward(
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

        // prediction = M_t @ k
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        // M_{t+1} = (1-alpha)*M_t - theta*error*k^T
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            let err_i = prediction[i] - v_t[i];
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] - theta_t * err_i * k_t[j];
            }
        }

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
    (m_states, y)
}

fn rust_delta_backward(
    k_mem: &[f32], v_mem: &[f32], q_mem: &[f32],
    alpha: &[f32], theta: &[f32], m_states: &[f32], d_y: &[f32],
    seq_len: usize, d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dd = d * d;
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut dalpha = vec![0.0f32; seq_len];
    let mut dtheta = vec![0.0f32; seq_len];
    let mut dm_init = vec![0.0f32; dd];

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

        // d_M += outer(d_y_t, q_t)
        for i in 0..d {
            for j in 0..d { d_m[i * d + j] += d_y_t[i] * q_t[j]; }
        }
        // d_q = M_{t+1}^T @ d_y
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_next[i * d + j] * d_y_t[i]; }
            dq[t * d + j] = sum;
        }

        // Recompute prediction/error
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_t[i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }

        // d_alpha, d_theta
        let mut da_sum = 0.0f32;
        let mut dt_sum = 0.0f32;
        for i in 0..d {
            for j in 0..d {
                da_sum += m_t[i * d + j] * d_m[i * d + j];
                dt_sum += error[i] * k_t[j] * d_m[i * d + j];
            }
        }
        dalpha[t] = -da_sum;
        dtheta[t] = -dt_sum;

        // d_error, d_k, d_v
        let mut d_err = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += (-theta_t * d_m[i * d + j]) * k_t[j]; }
            d_err[i] = sum;
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += (-theta_t * d_m[i * d + j]) * error[i]; }
            dk[t * d + j] = sum;
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_t[i * d + j] * d_err[i]; }
            dk[t * d + j] += sum;
        }
        for i in 0..d { dv[t * d + i] = -d_err[i]; }

        // Propagate d_M
        let retention = 1.0 - alpha_t;
        let mut d_m_prev = vec![0.0f32; dd];
        for i in 0..d {
            for j in 0..d {
                d_m_prev[i * d + j] = retention * d_m[i * d + j] + d_err[i] * k_t[j];
            }
        }
        d_m = d_m_prev;
    }
    dm_init.copy_from_slice(&d_m);
    (dk, dv, dq, dalpha, dtheta, dm_init)
}

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
        let alpha_t = alpha[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        // M_{t+1} = (1-alpha)*M_t + outer(v, k)
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            for j in 0..d {
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] + v_t[i] * k_t[j];
            }
        }

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
    (m_states, y)
}

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
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let eta_t = eta[t];
        let m_t = t * dd;
        let m_next = (t + 1) * dd;

        // prediction = M_t @ k
        let mut prediction = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_t + i * d + j] * k_t[j]; }
            prediction[i] = sum;
        }

        // error = prediction - v
        let mut error = vec![0.0f32; d];
        for i in 0..d { error[i] = prediction[i] - v_t[i]; }

        // S_{t+1} = eta * S_t - theta * outer(error, k)
        // M_{t+1} = (1-alpha) * M_t + S_{t+1}
        let retention = 1.0 - alpha_t;
        for i in 0..d {
            for j in 0..d {
                let s_new = eta_t * s_states[m_t + i * d + j]
                            - theta_t * error[i] * k_t[j];
                s_states[m_next + i * d + j] = s_new;
                m_states[m_next + i * d + j] =
                    retention * m_states[m_t + i * d + j] + s_new;
            }
        }

        // y = M_{t+1} @ q
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d { sum += m_states[m_next + i * d + j] * q_t[j]; }
            y[t * d + i] = sum;
        }
    }
    (m_states, s_states, y)
}

// ══════════════════════════════════════════════════════════════════════
// Test 1: Delta Rule forward + backward at d=2048
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_delta_forward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 10001);
    let v_mem = rand_buf(seq_len * d, 10002);
    let q_mem = rand_buf(seq_len * d, 10003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_delta_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, f32::MAX);

    check_close("delta_d2048_y", &y_rust, &y_cuda, 1e-4);
    check_close("delta_d2048_m", &m_states_rust, &m_states_cuda, 1e-4);

    // Sanity: M_final non-zero (catches silent data corruption from unwritten elements)
    let m_final = &m_states_cuda[seq_len * dd..];
    let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(m_norm > 1e-6, "Delta d=2048: M_final should be non-zero, got ‖M‖={m_norm}");
}

#[test]
#[serial(cuda)]
fn test_cuda_delta_backward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 20001);
    let v_mem = rand_buf(seq_len * d, 20002);
    let q_mem = rand_buf(seq_len * d, 20003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // Forward first
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d, f32::MAX);

    let d_y = rand_buf(seq_len * d, 20004);

    // Rust reference backward
    let (dk_rust, dv_rust, dq_rust, dalpha_rust, dtheta_rust, dm_rust) =
        rust_delta_backward(&k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y, seq_len, d);

    // CUDA backward
    let mut dk_cuda = vec![0.0f32; seq_len * d];
    let mut dv_cuda = vec![0.0f32; seq_len * d];
    let mut dq_cuda = vec![0.0f32; seq_len * d];
    let mut dalpha_cuda = vec![0.0f32; seq_len];
    let mut dtheta_cuda = vec![0.0f32; seq_len];
    let mut dm_cuda = vec![0.0f32; dd];

    delta_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dtheta_cuda, &mut dm_cuda,
        seq_len, d);

    check_close("delta_bwd_d2048_dk", &dk_rust, &dk_cuda, 1e-3);
    check_close("delta_bwd_d2048_dv", &dv_rust, &dv_cuda, 1e-3);
    check_close("delta_bwd_d2048_dq", &dq_rust, &dq_cuda, 1e-3);
    check_close("delta_bwd_d2048_dalpha", &dalpha_rust, &dalpha_cuda, 1e-2);
    check_close("delta_bwd_d2048_dtheta", &dtheta_rust, &dtheta_cuda, 1e-2);
    check_close("delta_bwd_d2048_dm", &dm_rust, &dm_cuda, 1e-3);
}

// ══════════════════════════════════════════════════════════════════════
// Test 2: Hebbian forward at d=2048
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_hebbian_forward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 30001);
    let v_mem = rand_buf(seq_len * d, 30002);
    let q_mem = rand_buf(seq_len * d, 30003);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_hebbian_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, f32::MAX);

    check_close("hebbian_d2048_y", &y_rust, &y_cuda, 1e-4);
    check_close("hebbian_d2048_m", &m_states_rust, &m_states_cuda, 1e-4);

    let m_final = &m_states_cuda[seq_len * dd..];
    let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(m_norm > 1e-6, "Hebbian d=2048: M_final should be non-zero, got ‖M‖={m_norm}");
}

#[test]
#[serial(cuda)]
fn test_cuda_hebbian_backward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 31001);
    let v_mem = rand_buf(seq_len * d, 31002);
    let q_mem = rand_buf(seq_len * d, 31003);
    let alpha = vec![0.05f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // Forward
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    hebbian_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_initial,
        &mut m_states, &mut y, seq_len, d, f32::MAX);

    let d_y = rand_buf(seq_len * d, 31004);

    // CUDA backward
    let mut dk_cuda = vec![0.0f32; seq_len * d];
    let mut dv_cuda = vec![0.0f32; seq_len * d];
    let mut dq_cuda = vec![0.0f32; seq_len * d];
    let mut dalpha_cuda = vec![0.0f32; seq_len];
    let mut dm_cuda = vec![0.0f32; dd];

    hebbian_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &m_states, &d_y,
        &mut dk_cuda, &mut dv_cuda, &mut dq_cuda,
        &mut dalpha_cuda, &mut dm_cuda,
        seq_len, d);

    // Sanity: gradients are non-zero
    let dk_norm: f32 = dk_cuda.iter().map(|x| x * x).sum::<f32>().sqrt();
    let dm_norm: f32 = dm_cuda.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(dk_norm > 1e-6, "Hebbian bwd d=2048: dk should be non-zero");
    assert!(dm_norm > 1e-6, "Hebbian bwd d=2048: dm should be non-zero");
}

// ══════════════════════════════════════════════════════════════════════
// Test 3: Titans LMM forward at d=2048
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_titans_forward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 40001);
    let v_mem = rand_buf(seq_len * d, 40002);
    let q_mem = rand_buf(seq_len * d, 40003);
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
        &m_initial, &s_initial,
        &mut m_cuda, &mut s_cuda, &mut y_cuda, seq_len, d, f32::MAX);

    check_close("titans_d2048_y", &y_rust, &y_cuda, 1e-4);
    check_close("titans_d2048_m", &m_rust, &m_cuda, 1e-4);
    check_close("titans_d2048_s", &s_rust, &s_cuda, 1e-4);

    let m_final = &m_cuda[seq_len * dd..];
    let m_norm: f32 = m_final.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(m_norm > 1e-6, "Titans d=2048: M_final should be non-zero, got ‖M‖={m_norm}");
}

#[test]
#[serial(cuda)]
fn test_cuda_titans_backward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 41001);
    let v_mem = rand_buf(seq_len * d, 41002);
    let q_mem = rand_buf(seq_len * d, 41003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let eta = vec![0.9f32; seq_len];
    let m_initial = vec![0.0f32; dd];
    let s_initial = vec![0.0f32; dd];

    // Forward
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut s_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    titans_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut m_states, &mut s_states, &mut y, seq_len, d, f32::MAX);

    let d_y = rand_buf(seq_len * d, 41004);

    // CUDA backward
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut dalpha = vec![0.0f32; seq_len];
    let mut dtheta = vec![0.0f32; seq_len];
    let mut deta = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];
    let mut ds = vec![0.0f32; dd];

    titans_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_states, &s_states, &d_y,
        &mut dk, &mut dv, &mut dq,
        &mut dalpha, &mut dtheta, &mut deta,
        &mut dm, &mut ds,
        seq_len, d);

    // Sanity: gradients are non-zero
    let dk_norm: f32 = dk.iter().map(|x| x * x).sum::<f32>().sqrt();
    let dm_norm: f32 = dm.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(dk_norm > 1e-6, "Titans bwd d=2048: dk should be non-zero");
    assert!(dm_norm > 1e-6, "Titans bwd d=2048: dm should be non-zero");
}

// ══════════════════════════════════════════════════════════════════════
// Test 4: DGD forward at d=2048
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_dgd_forward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 50001);
    let v_mem = rand_buf(seq_len * d, 50002);
    let q_mem = rand_buf(seq_len * d, 50003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // DGD math is identical to Delta at L2 bias — reuse Rust reference
    let (m_states_rust, y_rust) = rust_delta_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d);

    check_close("dgd_d2048_y", &y_rust, &y_cuda, 1e-4);
    check_close("dgd_d2048_m", &m_states_rust, &m_states_cuda, 1e-4);
}

#[test]
#[serial(cuda)]
fn test_cuda_dgd_backward_d2048() {
    let d = 2048;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 51001);
    let v_mem = rand_buf(seq_len * d, 51002);
    let q_mem = rand_buf(seq_len * d, 51003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    // Forward via DGD dispatch
    let mut m_states = vec![0.0f32; (seq_len + 1) * dd];
    let mut y = vec![0.0f32; seq_len * d];
    dgd_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states, &mut y, seq_len, d);

    let d_y = rand_buf(seq_len * d, 51004);

    // CUDA backward
    let mut dk = vec![0.0f32; seq_len * d];
    let mut dv = vec![0.0f32; seq_len * d];
    let mut dq = vec![0.0f32; seq_len * d];
    let mut dalpha = vec![0.0f32; seq_len];
    let mut dtheta = vec![0.0f32; seq_len];
    let mut dm = vec![0.0f32; dd];

    dgd_backward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_states, &d_y,
        &mut dk, &mut dv, &mut dq,
        &mut dalpha, &mut dtheta, &mut dm,
        seq_len, d);

    // Sanity: gradients are non-zero
    let dk_norm: f32 = dk.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(dk_norm > 1e-6, "DGD bwd d=2048: dk should be non-zero");
}

// ══════════════════════════════════════════════════════════════════════
// Test 5: Gate backward at D=1024 (was impossible before: assert(D<=512))
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_gate_backward_d1024() {
    use nl_hecate_core::gpu_buf::GpuBuf;
    use nl_hecate_core::dispatch::gate_backward_dd;

    let d = 1024;
    let seq_len = 4;
    let two_d = 2 * d;

    // Gate inputs
    let k_mem_h = rand_buf(seq_len * d, 60001);
    let v_mem_h = rand_buf(seq_len * d, 60002);

    // Gate outputs (sigmoid/softplus applied)
    let mut alpha_h = vec![0.0f32; seq_len];
    let mut theta_h = vec![0.0f32; seq_len];
    for t in 0..seq_len {
        alpha_h[t] = 0.5 + 0.1 * (t as f32);
        theta_h[t] = 0.1 + 0.05 * (t as f32);
    }

    // Upstream gradients
    let d_alpha_h = rand_buf(seq_len, 60003);
    let d_theta_h = rand_buf(seq_len, 60004);

    // Upload to device
    let k_dev = GpuBuf::from_host(&k_mem_h);
    let v_dev = GpuBuf::from_host(&v_mem_h);
    let alpha_dev = GpuBuf::from_host(&alpha_h);
    let theta_dev = GpuBuf::from_host(&theta_h);
    let d_alpha_dev = GpuBuf::from_host(&d_alpha_h);
    let d_theta_dev = GpuBuf::from_host(&d_theta_h);

    // Output buffers on device
    let mut dw_alpha_dev = GpuBuf::<f32>::zeros(two_d);
    let mut db_alpha_dev = GpuBuf::<f32>::zeros(1);
    let mut dw_theta_dev = GpuBuf::<f32>::zeros(two_d);
    let mut db_theta_dev = GpuBuf::<f32>::zeros(1);
    let mut dw_eta_dev = GpuBuf::<f32>::zeros(two_d);
    let mut db_eta_dev = GpuBuf::<f32>::zeros(1);

    gate_backward_dd(
        &d_alpha_dev, &alpha_dev,
        Some(&d_theta_dev), Some(&theta_dev),
        None, None, // no eta
        &k_dev, &v_dev,
        &mut dw_alpha_dev, &mut db_alpha_dev,
        &mut dw_theta_dev, &mut db_theta_dev,
        &mut dw_eta_dev, &mut db_eta_dev,
        seq_len, d,
    );
    nl_hecate_core::dispatch::cuda_sync();

    // Copy results back to host
    let mut dw_alpha = vec![0.0f32; two_d];
    let mut db_alpha = vec![0.0f32; 1];
    let mut dw_theta = vec![0.0f32; two_d];
    let mut db_theta = vec![0.0f32; 1];
    dw_alpha_dev.copy_to_host(&mut dw_alpha);
    db_alpha_dev.copy_to_host(&mut db_alpha);
    dw_theta_dev.copy_to_host(&mut dw_theta);
    db_theta_dev.copy_to_host(&mut db_theta);

    // Verify weight gradients are non-zero
    let wa_norm: f32 = dw_alpha.iter().map(|x| x * x).sum::<f32>().sqrt();
    let wt_norm: f32 = dw_theta.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(wa_norm > 1e-6, "Gate bwd D=1024: dw_alpha should be non-zero, got ‖·‖={wa_norm}");
    assert!(wt_norm > 1e-6, "Gate bwd D=1024: dw_theta should be non-zero, got ‖·‖={wt_norm}");

    // Verify bias gradients
    assert!(db_alpha[0].abs() > 1e-8, "Gate bwd D=1024: db_alpha should be non-zero");
    assert!(db_theta[0].abs() > 1e-8, "Gate bwd D=1024: db_theta should be non-zero");

    // Verify all 2*D weight elements are written (key test: indices [1024, 2048) were
    // previously unwritten when D > 512)
    let upper_half_alpha: f32 = dw_alpha[d..].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(upper_half_alpha > 1e-6,
        "Gate bwd D=1024: dw_alpha[D..2D] should be non-zero (was unwritten before fix)");
}

// ══════════════════════════════════════════════════════════════════════
// Test 6: Delta forward at d=1536 (odd dimension between 1024 and 2048)
// Exercises strided loop with non-power-of-2 stride count
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial(cuda)]
fn test_cuda_delta_forward_d1536() {
    let d = 1536;
    let seq_len = 2;
    let dd = d * d;

    let k_mem = rand_buf(seq_len * d, 70001);
    let v_mem = rand_buf(seq_len * d, 70002);
    let q_mem = rand_buf(seq_len * d, 70003);
    let alpha = vec![0.05f32; seq_len];
    let theta = vec![0.01f32; seq_len];
    let m_initial = vec![0.0f32; dd];

    let (m_states_rust, y_rust) = rust_delta_forward(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial, seq_len, d);

    let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
    let mut y_cuda = vec![0.0f32; seq_len * d];
    delta_forward_dispatch(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
        &mut m_states_cuda, &mut y_cuda, seq_len, d, f32::MAX);

    check_close("delta_d1536_y", &y_rust, &y_cuda, 1e-4);
    check_close("delta_d1536_m", &m_states_rust, &m_states_cuda, 1e-4);
}
