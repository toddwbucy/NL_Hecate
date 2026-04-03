// CUDA parity tests for TitansLMM MLP Memory forward kernel (Spec 75 Phase B).
//
// Verifies that `titans_mlp_forward_f32_cuda` matches the CPU reference
// inner loop from `step_mlp()` in core/src/titans_lmm.rs.
//
// The CUDA kernel receives pre-projected k/v/q and pre-computed gates
// (alpha, theta, eta), so these tests replicate only the inner loop
// portion of step_mlp (lines 487-584), not the full function.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};
use serial_test::serial;

use nl_hecate_core::titans_lmm::{MLPMemoryLayout, mlp_forward, mlp_inner_backward};
use nl_hecate_core::moneta::apply_attentional_bias;
use nl_hecate_core::model::{MemoryActivation, AttentionalBias};
use nl_hecate_core::retention::l2_apply_retention;
use nl_hecate_core::dispatch::titans_mlp_forward_cuda;

/// CPU reference inner loop matching step_mlp() — processes pre-projected
/// k_mem, v_mem, q_mem with pre-computed gates.
fn cpu_mlp_inner_loop(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_initial: &[f32],
    s_initial: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
    activation: MemoryActivation,
    m_norm_max: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let layout = MLPMemoryLayout::new(2, d, d_hidden / d);
    let state_size = layout.total_params;

    let mut m_states = vec![0.0f32; (seq_len + 1) * state_size];
    let mut s_states = vec![0.0f32; (seq_len + 1) * state_size];
    let mut y = vec![0.0f32; seq_len * d];

    // Initialize M_0 and S_0
    m_states[..state_size].copy_from_slice(m_initial);
    s_states[..state_size].copy_from_slice(s_initial);

    let mut grad_buf = vec![0.0f32; state_size];

    for t in 0..seq_len {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let eta_t = eta[t];

        let m_base = t * state_size;
        let m_next = (t + 1) * state_size;
        let s_base = t * state_size;
        let s_next = (t + 1) * state_size;

        // 1. MLP forward on k_t
        let (prediction, pre_acts, activations) = mlp_forward(
            &m_states, m_base, k_t, &layout, activation);

        // 2. Error = prediction - v_t
        let mut error = vec![0.0f32; d];
        for i in 0..d {
            error[i] = prediction[i] - v_t[i];
        }

        // 3. L2 attentional bias → d_out
        let biased = apply_attentional_bias(&error, AttentionalBias::L2, 10.0);

        // 4. Analytical backward → grad_buf
        grad_buf.fill(0.0);
        mlp_inner_backward(
            &biased, &pre_acts, &activations,
            &m_states, m_base, &layout, activation,
            &mut grad_buf, 0);

        // 5. EMA momentum: S_{t+1} = eta*S_t - theta*grad
        for i in 0..state_size {
            s_states[s_next + i] = eta_t * s_states[s_base + i]
                                 - theta_t * grad_buf[i];
        }

        // 6. Retention: M_{t+1} = (1-alpha)*M_t + S_{t+1}
        m_states.copy_within(m_base..m_base + state_size, m_next);
        l2_apply_retention(&mut m_states[m_next..m_next + state_size], 1.0 - alpha_t);
        for i in 0..state_size {
            m_states[m_next + i] += s_states[s_next + i];
        }

        // 7. M-norm clamp
        if m_norm_max < f32::MAX {
            let slice = &mut m_states[m_next..m_next + state_size];
            let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            if norm > m_norm_max {
                let scale = m_norm_max / norm;
                for x in slice.iter_mut() { *x *= scale; }
            }
        }

        // 8. Readout: y_t = M_{t+1}(q_t)
        let (y_t, _, _) = mlp_forward(
            &m_states, m_next, q_t, &layout, activation);
        y[t * d..(t + 1) * d].copy_from_slice(&y_t);
    }

    (m_states, s_states, y)
}

/// Single-token diagnostic: isolates per-token divergence.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_single_token_debug() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 1;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 42);
    let v_mem = rand_buf(seq_len * d, 43);
    let q_mem = rand_buf(seq_len * d, 44);
    let alpha = vec![0.05f32];
    let theta = vec![0.01f32];
    let eta = vec![0.9f32];
    let m_initial = rand_buf(state_size, 45);
    let s_initial = vec![0.0f32; state_size];

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, f32::MAX);

    // Check initial M copy
    check_close("M_0 copy", &cuda_m[..state_size], &cpu_m[..state_size], 1e-7);

    // Check S_0 copy
    check_close("S_0 copy", &cuda_s[..state_size], &cpu_s[..state_size], 1e-7);

    // Check S after 1 token (gradient accuracy)
    check_close("S_1", &cuda_s[state_size..2*state_size], &cpu_s[state_size..2*state_size], 1e-4);

    // Check M after 1 token
    check_close("M_1", &cuda_m[state_size..2*state_size], &cpu_m[state_size..2*state_size], 1e-4);

    // Check output
    check_close("y_0", &cuda_y, &cpu_y, 1e-3);
}

/// Basic CUDA vs CPU parity test: small config (d=8, expansion=4, d_h=32).
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_parity_gelu() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;
    let m_norm_max = 100.0f32;

    // Generate deterministic test data
    let k_mem = rand_buf(seq_len * d, 42);
    let v_mem = rand_buf(seq_len * d, 43);
    let q_mem = rand_buf(seq_len * d, 44);

    // Gates: alpha ∈ (0, 1), theta > 0, eta ∈ (0, 1)
    let alpha: Vec<f32> = (0..seq_len).map(|t| 0.05 + 0.02 * t as f32).collect();
    let theta: Vec<f32> = (0..seq_len).map(|t| 0.01 + 0.005 * t as f32).collect();
    let eta: Vec<f32> = (0..seq_len).map(|t| 0.9 - 0.05 * t as f32).collect();

    let m_initial = rand_buf(state_size, 45);
    let s_initial = vec![0.0f32; state_size]; // S starts at zero

    // CPU reference
    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, m_norm_max);

    // CUDA path
    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, // activation=0 (GELU)
        m_norm_max);

    // Compare with tolerance (f32 accumulation + transcendental functions)
    check_close("titans_mlp y (GELU)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (GELU)", &cuda_m, &cpu_m, 1e-4);
    check_close("titans_mlp s_states (GELU)", &cuda_s, &cpu_s, 1e-4);
}

/// SiLU activation parity test.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_parity_silu() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 50);
    let v_mem = rand_buf(seq_len * d, 51);
    let q_mem = rand_buf(seq_len * d, 52);
    let alpha: Vec<f32> = vec![0.05; seq_len];
    let theta: Vec<f32> = vec![0.01; seq_len];
    let eta: Vec<f32> = vec![0.9; seq_len];
    let m_initial = rand_buf(state_size, 53);
    let s_initial = vec![0.0f32; state_size];

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::SiLU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 1, // activation=1 (SiLU)
        f32::MAX);

    check_close("titans_mlp y (SiLU)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (SiLU)", &cuda_m, &cpu_m, 1e-4);
    check_close("titans_mlp s_states (SiLU)", &cuda_s, &cpu_s, 1e-4);
}

/// ReLU activation parity test.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_parity_relu() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 60);
    let v_mem = rand_buf(seq_len * d, 61);
    let q_mem = rand_buf(seq_len * d, 62);
    let alpha: Vec<f32> = vec![0.05; seq_len];
    let theta: Vec<f32> = vec![0.01; seq_len];
    let eta: Vec<f32> = vec![0.9; seq_len];
    let m_initial = rand_buf(state_size, 63);
    let s_initial = vec![0.0f32; state_size];

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::ReLU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 2, // activation=2 (ReLU)
        f32::MAX);

    check_close("titans_mlp y (ReLU)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (ReLU)", &cuda_m, &cpu_m, 1e-4);
    check_close("titans_mlp s_states (ReLU)", &cuda_s, &cpu_s, 1e-4);
}

/// Test M-norm projection fires and matches CPU.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_m_norm_clamp() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    // Use a very aggressive learning rate and low M-norm to force clamping.
    let k_mem = rand_buf(seq_len * d, 70);
    let v_mem = rand_buf(seq_len * d, 71);
    let q_mem = rand_buf(seq_len * d, 72);
    let alpha: Vec<f32> = vec![0.01; seq_len]; // low retention → big M updates
    let theta: Vec<f32> = vec![0.5; seq_len];  // aggressive learning rate
    let eta: Vec<f32> = vec![0.9; seq_len];

    // Large initial M to be close to norm limit
    let mut m_initial = rand_buf(state_size, 73);
    for x in m_initial.iter_mut() { *x *= 10.0; }
    let s_initial = vec![0.0f32; state_size];
    let m_norm_max = 5.0f32; // will definitely fire

    let (cpu_m, _cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, m_norm_max);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, m_norm_max);

    check_close("titans_mlp y (M-norm)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (M-norm)", &cuda_m, &cpu_m, 1e-4);

    // Verify M-norm clamping actually fired: all M_{t>0} should be ≤ m_norm_max
    for t in 1..=seq_len {
        let base = t * state_size;
        let norm_sq: f32 = cuda_m[base..base + state_size].iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            norm <= m_norm_max + 1e-3,
            "M-norm at t={t}: {norm:.4} exceeds max {m_norm_max}"
        );
    }
}

/// Longer sequence test (seq_len=16) to exercise more recurrence steps.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_longer_sequence() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 16;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 80);
    let v_mem = rand_buf(seq_len * d, 81);
    let q_mem = rand_buf(seq_len * d, 82);

    // Varying gates per token
    let alpha: Vec<f32> = (0..seq_len).map(|t| 0.02 + 0.005 * (t as f32)).collect();
    let theta: Vec<f32> = (0..seq_len).map(|t| 0.005 + 0.002 * (t as f32)).collect();
    let eta: Vec<f32> = (0..seq_len).map(|t| 0.95 - 0.01 * (t as f32)).collect();

    let m_initial = rand_buf(state_size, 83);
    let s_initial = vec![0.0f32; state_size];

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, f32::MAX);

    // Relaxed tolerance for 16-step accumulation
    check_close("titans_mlp y (seq=16)", &cuda_y, &cpu_y, 5e-4);
    check_close("titans_mlp m_states (seq=16)", &cuda_m, &cpu_m, 5e-4);
    check_close("titans_mlp s_states (seq=16)", &cuda_s, &cpu_s, 5e-4);
}

/// Non-zero initial S (momentum carry from previous segment).
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_nonzero_s_initial() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 90);
    let v_mem = rand_buf(seq_len * d, 91);
    let q_mem = rand_buf(seq_len * d, 92);
    let alpha: Vec<f32> = vec![0.05; seq_len];
    let theta: Vec<f32> = vec![0.01; seq_len];
    let eta: Vec<f32> = vec![0.9; seq_len];

    let m_initial = rand_buf(state_size, 93);
    let s_initial = rand_buf(state_size, 94); // non-zero momentum carry

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, f32::MAX);

    check_close("titans_mlp y (S_0≠0)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (S_0≠0)", &cuda_m, &cpu_m, 1e-4);
    check_close("titans_mlp s_states (S_0≠0)", &cuda_s, &cpu_s, 1e-4);
}

/// Batched test: batch_size=4, each batch processed independently.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_batched() {
    let d = 8;
    let expansion = 4;
    let d_h = d * expansion;
    let seq_len = 4;
    let batch_size = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    // Generate per-batch data by concatenating with different seeds.
    let mut k_all = Vec::new();
    let mut v_all = Vec::new();
    let mut q_all = Vec::new();
    let mut alpha_all = Vec::new();
    let mut theta_all = Vec::new();
    let mut eta_all = Vec::new();
    let mut m_init_all = Vec::new();
    let mut s_init_all = Vec::new();

    for b in 0..batch_size {
        let seed_base = 100 + b as u64 * 10;
        k_all.extend_from_slice(&rand_buf(seq_len * d, seed_base));
        v_all.extend_from_slice(&rand_buf(seq_len * d, seed_base + 1));
        q_all.extend_from_slice(&rand_buf(seq_len * d, seed_base + 2));
        alpha_all.extend(vec![0.05 + 0.01 * b as f32; seq_len]);
        theta_all.extend(vec![0.01; seq_len]);
        eta_all.extend(vec![0.9; seq_len]);
        m_init_all.extend_from_slice(&rand_buf(state_size, seed_base + 3));
        s_init_all.extend(vec![0.0f32; state_size]);
    }

    // CUDA batched call
    let mut cuda_m = vec![0.0f32; batch_size * (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; batch_size * (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; batch_size * seq_len * d];

    titans_mlp_forward_cuda(
        &k_all, &v_all, &q_all, &alpha_all, &theta_all, &eta_all,
        &m_init_all, &s_init_all,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, batch_size, 0, f32::MAX);

    // Compare each batch element against independent CPU reference
    let input_stride = seq_len * d;
    let gate_stride = seq_len;
    let init_stride = state_size;
    let states_stride = (seq_len + 1) * state_size;
    let y_stride = seq_len * d;

    for b in 0..batch_size {
        let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
            &k_all[b * input_stride..(b + 1) * input_stride],
            &v_all[b * input_stride..(b + 1) * input_stride],
            &q_all[b * input_stride..(b + 1) * input_stride],
            &alpha_all[b * gate_stride..(b + 1) * gate_stride],
            &theta_all[b * gate_stride..(b + 1) * gate_stride],
            &eta_all[b * gate_stride..(b + 1) * gate_stride],
            &m_init_all[b * init_stride..(b + 1) * init_stride],
            &s_init_all[b * init_stride..(b + 1) * init_stride],
            seq_len, d, d_h,
            MemoryActivation::GELU, f32::MAX);

        let label = format!("batch {b}");
        check_close(&format!("{label} y"), &cuda_y[b * y_stride..(b + 1) * y_stride], &cpu_y, 1e-4);
        check_close(&format!("{label} m"), &cuda_m[b * states_stride..(b + 1) * states_stride], &cpu_m, 1e-4);
        check_close(&format!("{label} s"), &cuda_s[b * states_stride..(b + 1) * states_stride], &cpu_s, 1e-4);
    }
}

/// Non-power-of-two d_hidden: d=6, expansion=3, d_h=18.
/// Verifies the kernel handles non-aligned dimensions correctly.
#[test]
#[serial(cuda)]
fn test_titans_mlp_cuda_non_pow2_d_hidden() {
    let d = 6;
    let expansion = 3;
    let d_h = d * expansion; // 18 — not a power of 2
    let seq_len = 4;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    let k_mem = rand_buf(seq_len * d, 200);
    let v_mem = rand_buf(seq_len * d, 201);
    let q_mem = rand_buf(seq_len * d, 202);
    let alpha: Vec<f32> = vec![0.05; seq_len];
    let theta: Vec<f32> = vec![0.01; seq_len];
    let eta: Vec<f32> = vec![0.9; seq_len];
    let m_initial = rand_buf(state_size, 203);
    let s_initial = vec![0.0f32; state_size];

    let (cpu_m, cpu_s, cpu_y) = cpu_mlp_inner_loop(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial, seq_len, d, d_h,
        MemoryActivation::GELU, f32::MAX);

    let mut cuda_m = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; seq_len * d];

    titans_mlp_forward_cuda(
        &k_mem, &v_mem, &q_mem, &alpha, &theta, &eta,
        &m_initial, &s_initial,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, 1, 0, f32::MAX);

    check_close("titans_mlp y (non-pow2 d_h)", &cuda_y, &cpu_y, 1e-4);
    check_close("titans_mlp m_states (non-pow2 d_h)", &cuda_m, &cpu_m, 1e-4);
    check_close("titans_mlp s_states (non-pow2 d_h)", &cuda_s, &cpu_s, 1e-4);
}
