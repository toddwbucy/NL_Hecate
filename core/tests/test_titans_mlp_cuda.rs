// CUDA parity tests for TitansLMM MLP Memory kernels (Spec 75 Phase B+C).
//
// Verifies that `titans_mlp_forward_f32_cuda` and `titans_mlp_backward_f32_cuda`
// match the CPU reference inner loop from `step_mlp()` in core/src/titans_lmm.rs.
//
// The CUDA kernels receive pre-projected k/v/q and pre-computed gates
// (alpha, theta, eta), so these tests replicate only the inner loop
// portion of step_mlp, not the full function.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};
use serial_test::serial;

use nl_hecate_core::titans_lmm::{MLPMemoryLayout, mlp_forward, mlp_inner_backward};
use nl_hecate_core::moneta::apply_attentional_bias;
use nl_hecate_core::model::{MemoryActivation, AttentionalBias};
use nl_hecate_core::retention::l2_apply_retention;
use nl_hecate_core::dispatch::{titans_mlp_forward_cuda, titans_mlp_backward_cuda};

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

// ═══════════════════════════════════════════════════════════════════════
// Backward kernel parity tests (Spec 75 Phase C)
// ═══════════════════════════════════════════════════════════════════════

/// Activation function and derivative for CPU backward reference.
fn act_fn(x: f32, activation: MemoryActivation) -> f32 {
    match activation {
        MemoryActivation::GELU => {
            let c = 0.7978845608028654f32;
            let inner = c * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
        MemoryActivation::SiLU => {
            let s = 1.0 / (1.0 + (-x).exp());
            x * s
        }
        MemoryActivation::ReLU => x.max(0.0),
    }
}

fn act_prime(x: f32, activation: MemoryActivation) -> f32 {
    match activation {
        MemoryActivation::GELU => {
            let c = 0.7978845608028654f32;
            let a = 0.044715f32;
            let inner = c * (x + a * x * x * x);
            let t = inner.tanh();
            let sech2 = 1.0 - t * t;
            let d_inner = c * (1.0 + 3.0 * a * x * x);
            0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
        }
        MemoryActivation::SiLU => {
            let s = 1.0 / (1.0 + (-x).exp());
            s + x * s * (1.0 - s)
        }
        MemoryActivation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
    }
}

/// CPU backward reference matching the CUDA kernel's 6-phase structure exactly.
///
/// Takes forward trajectory (m_states, s_states) and upstream d_y.
/// Returns (d_k_mem, d_v_mem, d_q_mem, d_alpha, d_theta, d_eta, d_m_initial, d_s_initial).
#[allow(clippy::too_many_arguments)]
fn cpu_mlp_backward(
    k_mem: &[f32],
    v_mem: &[f32],
    q_mem: &[f32],
    alpha: &[f32],
    theta: &[f32],
    eta: &[f32],
    m_states: &[f32],
    s_states: &[f32],
    d_y: &[f32],
    seq_len: usize,
    d: usize,
    d_hidden: usize,
    activation: MemoryActivation,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let w1_size = d_hidden * d;
    let b1_size = d_hidden;
    let w2_size = d * d_hidden;
    let b2_size = d;
    let w1_off = 0;
    let b1_off = w1_size;
    let w2_off = w1_size + b1_size;
    let b2_off = w2_off + w2_size;
    let state_size = w1_size + b1_size + w2_size + b2_size;

    let mut d_k_out = vec![0.0f32; seq_len * d];
    let mut d_v_out = vec![0.0f32; seq_len * d];
    let mut d_q_out = vec![0.0f32; seq_len * d];
    let mut d_alpha_out = vec![0.0f32; seq_len];
    let mut d_theta_out = vec![0.0f32; seq_len];
    let mut d_eta_out = vec![0.0f32; seq_len];

    let mut d_m = vec![0.0f32; state_size];
    let mut d_s = vec![0.0f32; state_size];

    for t in (0..seq_len).rev() {
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        let q_t = &q_mem[t * d..(t + 1) * d];
        let dy_t = &d_y[t * d..(t + 1) * d];
        let m_t = &m_states[t * state_size..(t + 1) * state_size];
        let m_next = &m_states[(t + 1) * state_size..(t + 2) * state_size];
        let s_t = &s_states[t * state_size..(t + 1) * state_size];
        let alpha_t = alpha[t];
        let theta_t = theta[t];
        let eta_t = eta[t];
        let retention = 1.0 - alpha_t;

        // ── PHASE 1: Readout backward — y_t = MLP(M_{t+1}, q_t) ──

        // Recompute readout: q_pre = W1_next @ q_t + b1_next
        let mut q_pre = vec![0.0f32; d_hidden];
        for row in 0..d_hidden {
            let mut sum = m_next[b1_off + row];
            for j in 0..d {
                sum += m_next[w1_off + row * d + j] * q_t[j];
            }
            q_pre[row] = sum;
        }
        // q_hid = σ(q_pre)
        let q_hid: Vec<f32> = q_pre.iter().map(|&x| act_fn(x, activation)).collect();

        // d_M[W2] += outer(d_y_t, q_hid)
        for i in 0..d {
            for j in 0..d_hidden {
                d_m[w2_off + i * d_hidden + j] += dy_t[i] * q_hid[j];
            }
        }
        // d_M[b2] += d_y_t
        for i in 0..b2_size {
            d_m[b2_off + i] += dy_t[i];
        }

        // d_q_hid = W2_next^T @ d_y_t, then d_q_pre = d_q_hid * σ'(q_pre)
        let mut d_q_pre = vec![0.0f32; d_hidden];
        for j in 0..d_hidden {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_next[w2_off + i * d_hidden + j] * dy_t[i];
            }
            d_q_pre[j] = sum * act_prime(q_pre[j], activation);
        }

        // d_M[W1] += outer(d_q_pre, q_t)
        for i in 0..d_hidden {
            for j in 0..d {
                d_m[w1_off + i * d + j] += d_q_pre[i] * q_t[j];
            }
        }
        // d_M[b1] += d_q_pre
        for i in 0..b1_size {
            d_m[b1_off + i] += d_q_pre[i];
        }

        // d_q_mem[t] = W1_next^T @ d_q_pre
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d_hidden {
                sum += m_next[w1_off + i * d + j] * d_q_pre[i];
            }
            d_q_out[t * d + j] = sum;
        }

        // ── PHASE 2: Retention backward ──
        // d_S += d_M
        for i in 0..state_size {
            d_s[i] += d_m[i];
        }

        // d_alpha = -<M_t, d_M>
        let mut dot_alpha = 0.0f32;
        for i in 0..state_size {
            dot_alpha += m_t[i] * d_m[i];
        }
        d_alpha_out[t] = -dot_alpha;

        // d_eta = <S_t, d_S>
        let mut dot_eta = 0.0f32;
        for i in 0..state_size {
            dot_eta += s_t[i] * d_s[i];
        }
        d_eta_out[t] = dot_eta;

        // ── PHASE 3: Recompute forward intermediates from M_t ──
        let mut pre_act = vec![0.0f32; d_hidden];
        for row in 0..d_hidden {
            let mut sum = m_t[b1_off + row];
            for j in 0..d {
                sum += m_t[w1_off + row * d + j] * k_t[j];
            }
            pre_act[row] = sum;
        }
        let hidden: Vec<f32> = pre_act.iter().map(|&x| act_fn(x, activation)).collect();

        let mut error = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = m_t[b2_off + i];
            for j in 0..d_hidden {
                sum += m_t[w2_off + i * d_hidden + j] * hidden[j];
            }
            error[i] = sum - v_t[i];
        }

        // grad_h = W2_t^T @ error, grad_pre = grad_h * σ'(pre_act)
        let mut grad_pre = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            let mut gh = 0.0f32;
            for j in 0..d {
                gh += m_t[w2_off + j * d_hidden + i] * error[j];
            }
            grad_pre[i] = gh * act_prime(pre_act[i], activation);
        }

        // ── PHASE 4: d_theta = -<grad, d_S> (virtual tensor dot product) ──
        let mut dot_theta = 0.0f32;
        // W1 contribution
        for idx in 0..w1_size {
            let row = idx / d;
            let col = idx % d;
            dot_theta += grad_pre[row] * k_t[col] * d_s[w1_off + idx];
        }
        // b1 contribution
        for i in 0..b1_size {
            dot_theta += grad_pre[i] * d_s[b1_off + i];
        }
        // W2 contribution
        for idx in 0..w2_size {
            let row = idx / d_hidden;
            let col = idx % d_hidden;
            dot_theta += error[row] * hidden[col] * d_s[w2_off + idx];
        }
        // b2 contribution
        for i in 0..b2_size {
            dot_theta += error[i] * d_s[b2_off + i];
        }
        d_theta_out[t] = -dot_theta;

        // ── PHASE 5: MLP backward (second-order) ──
        // Step A: d_err[i] = -θ·(Σ_j d_S[w2+i*dh+j]*hidden[j] + d_S[b2+i])
        let mut d_err = vec![0.0f32; d];
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d_hidden {
                sum += d_s[w2_off + i * d_hidden + j] * hidden[j];
            }
            sum += d_s[b2_off + i];
            d_err[i] = -theta_t * sum;
        }

        // Step B: d_h_buf[j] = -θ·Σ_i d_S[w2+i*dh+j]*error[i]
        let mut d_h_buf = vec![0.0f32; d_hidden];
        for j in 0..d_hidden {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += d_s[w2_off + i * d_hidden + j] * error[i];
            }
            d_h_buf[j] = -theta_t * sum;
        }

        // Step C: d_k[j] = -θ·Σ_i d_S[w1+i*d+j]*grad_pre[i]
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d_hidden {
                sum += d_s[w1_off + i * d + j] * grad_pre[i];
            }
            d_k_out[t * d + j] = -theta_t * sum;
        }

        // Step D: d_grad_pre[i] = -θ·(Σ_j d_S[w1+i*d+j]*k[j] + d_S[b1+i])
        let mut d_grad_pre = vec![0.0f32; d_hidden];
        for i in 0..d_hidden {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += d_s[w1_off + i * d + j] * k_t[j];
            }
            sum += d_s[b1_off + i];
            d_grad_pre[i] = -theta_t * sum;
        }

        // Step E: d_grad_h = d_grad_pre * σ'(pre_act)
        for i in 0..d_hidden {
            d_grad_pre[i] *= act_prime(pre_act[i], activation);
        }
        // d_grad_pre is now d_grad_h

        // Step F: d_err += W2_t @ d_grad_h
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d_hidden {
                sum += m_t[w2_off + i * d_hidden + j] * d_grad_pre[j];
            }
            d_err[i] += sum;
        }

        // Step G: d_v = -d_err
        for i in 0..d {
            d_v_out[t * d + i] = -d_err[i];
        }

        // Step H: d_h_total = d_h_buf + W2_t^T @ d_err
        for j in 0..d_hidden {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_t[w2_off + i * d_hidden + j] * d_err[i];
            }
            d_h_buf[j] += sum;
        }

        // Step I: d_pre_act = d_h_total * σ'(pre_act)
        for i in 0..d_hidden {
            d_h_buf[i] *= act_prime(pre_act[i], activation);
        }
        // d_h_buf is now d_pre_act

        // Step J: d_k += W1_t^T @ d_pre_act
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d_hidden {
                sum += m_t[w1_off + i * d + j] * d_h_buf[i];
            }
            d_k_out[t * d + j] += sum;
        }

        // ── PHASE 6: Propagation ──
        // W1: d_M = retention*d_M + outer(d_pre_act, k)
        for idx in 0..w1_size {
            let row = idx / d;
            let col = idx % d;
            d_m[w1_off + idx] = retention * d_m[w1_off + idx]
                + d_h_buf[row] * k_t[col];
        }
        // b1: d_M = retention*d_M + d_pre_act
        for i in 0..b1_size {
            d_m[b1_off + i] = retention * d_m[b1_off + i] + d_h_buf[i];
        }
        // W2: d_M = retention*d_M + outer(d_err, hidden) + outer(error, d_grad_h)
        for idx in 0..w2_size {
            let row = idx / d_hidden;
            let col = idx % d_hidden;
            d_m[w2_off + idx] = retention * d_m[w2_off + idx]
                + d_err[row] * hidden[col]
                + error[row] * d_grad_pre[col];
        }
        // b2: d_M = retention*d_M + d_err
        for i in 0..b2_size {
            d_m[b2_off + i] = retention * d_m[b2_off + i] + d_err[i];
        }
        // d_S *= η
        for i in 0..state_size {
            d_s[i] = eta_t * d_s[i];
        }
    }

    (d_k_out, d_v_out, d_q_out, d_alpha_out, d_theta_out, d_eta_out, d_m, d_s)
}

/// Helper: run forward + backward through both CPU and CUDA, compare all gradients.
#[allow(clippy::too_many_arguments)]
fn run_backward_parity(
    seq_len: usize, d: usize, expansion: usize,
    batch_size: usize, activation: MemoryActivation, act_id: i32,
    m_norm_max: f32, label: &str, tol: f32,
) {
    let d_h = d * expansion;
    let layout = MLPMemoryLayout::new(2, d, expansion);
    let state_size = layout.total_params;

    // Per-batch data
    let mut k_all = Vec::new();
    let mut v_all = Vec::new();
    let mut q_all = Vec::new();
    let mut alpha_all = Vec::new();
    let mut theta_all = Vec::new();
    let mut eta_all = Vec::new();
    let mut m_init_all = Vec::new();
    let mut s_init_all = Vec::new();
    let mut dy_all = Vec::new();

    for b in 0..batch_size {
        let seed_base = 300 + b as u64 * 10;
        // Scale inputs small to keep gradients reasonable
        let mut k = rand_buf(seq_len * d, seed_base);
        let mut v = rand_buf(seq_len * d, seed_base + 1);
        let mut q = rand_buf(seq_len * d, seed_base + 2);
        let mut dy = rand_buf(seq_len * d, seed_base + 6);
        for x in k.iter_mut() { *x *= 0.3; }
        for x in v.iter_mut() { *x *= 0.3; }
        for x in q.iter_mut() { *x *= 0.3; }
        for x in dy.iter_mut() { *x *= 0.3; }

        k_all.extend_from_slice(&k);
        v_all.extend_from_slice(&v);
        q_all.extend_from_slice(&q);
        alpha_all.extend(vec![0.05 + 0.01 * b as f32; seq_len]);
        theta_all.extend(vec![0.01; seq_len]);
        eta_all.extend(vec![0.9; seq_len]);

        let mut m_init = rand_buf(state_size, seed_base + 3);
        for x in m_init.iter_mut() { *x *= 0.1; }
        m_init_all.extend_from_slice(&m_init);
        s_init_all.extend(vec![0.0f32; state_size]);
        dy_all.extend_from_slice(&dy);
    }

    // ── Run forward (CPU) to get m_states, s_states ──
    // For CUDA backward, we also need the forward trajectory from the CUDA forward
    // (the backward kernel takes m_states/s_states as input).
    let mut cuda_m = vec![0.0f32; batch_size * (seq_len + 1) * state_size];
    let mut cuda_s = vec![0.0f32; batch_size * (seq_len + 1) * state_size];
    let mut cuda_y = vec![0.0f32; batch_size * seq_len * d];

    titans_mlp_forward_cuda(
        &k_all, &v_all, &q_all, &alpha_all, &theta_all, &eta_all,
        &m_init_all, &s_init_all,
        &mut cuda_m, &mut cuda_s, &mut cuda_y,
        seq_len, d, d_h, batch_size, act_id, m_norm_max);

    // ── Run CUDA backward ──
    let total_input = batch_size * seq_len * d;
    let total_gate = batch_size * seq_len;
    let total_init = batch_size * state_size;

    let mut cuda_dk = vec![0.0f32; total_input];
    let mut cuda_dv = vec![0.0f32; total_input];
    let mut cuda_dq = vec![0.0f32; total_input];
    let mut cuda_dalpha = vec![0.0f32; total_gate];
    let mut cuda_dtheta = vec![0.0f32; total_gate];
    let mut cuda_deta = vec![0.0f32; total_gate];
    let mut cuda_dm_init = vec![0.0f32; total_init];
    let mut cuda_ds_init = vec![0.0f32; total_init];

    titans_mlp_backward_cuda(
        &k_all, &v_all, &q_all, &alpha_all, &theta_all, &eta_all,
        &cuda_m, &cuda_s, &dy_all,
        &mut cuda_dk, &mut cuda_dv, &mut cuda_dq,
        &mut cuda_dalpha, &mut cuda_dtheta, &mut cuda_deta,
        &mut cuda_dm_init, &mut cuda_ds_init,
        seq_len, d, d_h, batch_size, act_id, m_norm_max);

    // ── Run CPU backward per batch and compare ──
    let input_stride = seq_len * d;
    let gate_stride = seq_len;
    let states_stride = (seq_len + 1) * state_size;

    // For d_m_initial/d_s_initial: CUDA sums across batch via atomicAdd,
    // CPU computes per-batch separately. Sum CPU results for comparison.
    let mut cpu_dm_init_sum = vec![0.0f32; state_size];
    let mut cpu_ds_init_sum = vec![0.0f32; state_size];

    for b in 0..batch_size {
        // CPU backward uses the CUDA forward's m_states/s_states (same trajectory)
        let m_slice = &cuda_m[b * states_stride..(b + 1) * states_stride];
        let s_slice = &cuda_s[b * states_stride..(b + 1) * states_stride];

        let (cpu_dk, cpu_dv, cpu_dq, cpu_da, cpu_dt, cpu_de, cpu_dm, cpu_ds) =
            cpu_mlp_backward(
                &k_all[b * input_stride..(b + 1) * input_stride],
                &v_all[b * input_stride..(b + 1) * input_stride],
                &q_all[b * input_stride..(b + 1) * input_stride],
                &alpha_all[b * gate_stride..(b + 1) * gate_stride],
                &theta_all[b * gate_stride..(b + 1) * gate_stride],
                &eta_all[b * gate_stride..(b + 1) * gate_stride],
                m_slice, s_slice,
                &dy_all[b * input_stride..(b + 1) * input_stride],
                seq_len, d, d_h, activation);

        let tag = format!("{label} batch {b}");
        check_close(&format!("{tag} d_k"), &cuda_dk[b * input_stride..(b + 1) * input_stride], &cpu_dk, tol);
        check_close(&format!("{tag} d_v"), &cuda_dv[b * input_stride..(b + 1) * input_stride], &cpu_dv, tol);
        check_close(&format!("{tag} d_q"), &cuda_dq[b * input_stride..(b + 1) * input_stride], &cpu_dq, tol);
        check_close(&format!("{tag} d_alpha"), &cuda_dalpha[b * gate_stride..(b + 1) * gate_stride], &cpu_da, tol);
        check_close(&format!("{tag} d_theta"), &cuda_dtheta[b * gate_stride..(b + 1) * gate_stride], &cpu_dt, tol);
        check_close(&format!("{tag} d_eta"), &cuda_deta[b * gate_stride..(b + 1) * gate_stride], &cpu_de, tol);

        for i in 0..state_size {
            cpu_dm_init_sum[i] += cpu_dm[i];
            cpu_ds_init_sum[i] += cpu_ds[i];
        }
    }

    // Compare summed d_m_initial / d_s_initial
    // For batch_size>1, atomicAdd ordering differs — use relaxed tolerance
    let init_tol = if batch_size > 1 { tol * 2.0 } else { tol };
    check_close(&format!("{label} d_m_initial"), &cuda_dm_init[..state_size], &cpu_dm_init_sum, init_tol);
    check_close(&format!("{label} d_s_initial"), &cuda_ds_init[..state_size], &cpu_ds_init_sum, init_tol);
}

// ── Backward parity tests ──────────────────────────────────────────

/// Backward parity: GELU, single batch, small config.
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_gelu() {
    run_backward_parity(4, 8, 4, 1, MemoryActivation::GELU, 0, f32::MAX, "bw_gelu", 1e-3);
}

/// Backward parity: SiLU activation.
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_silu() {
    run_backward_parity(4, 8, 4, 1, MemoryActivation::SiLU, 1, f32::MAX, "bw_silu", 1e-3);
}

/// Backward parity: ReLU activation.
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_relu() {
    run_backward_parity(4, 8, 4, 1, MemoryActivation::ReLU, 2, f32::MAX, "bw_relu", 1e-3);
}

/// Backward parity: batched (batch_size=4).
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_batched() {
    run_backward_parity(4, 8, 4, 4, MemoryActivation::GELU, 0, f32::MAX, "bw_batched", 1e-3);
}

/// Backward parity: longer sequence (16 tokens).
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_longer() {
    run_backward_parity(16, 8, 4, 1, MemoryActivation::GELU, 0, f32::MAX, "bw_long", 2e-3);
}

/// Backward parity: single token (boundary case).
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_single_token() {
    run_backward_parity(1, 8, 4, 1, MemoryActivation::GELU, 0, f32::MAX, "bw_1tok", 1e-3);
}

/// Backward parity: non-power-of-2 d with finite m_norm_max.
/// Exercises rounded block_size > d shared memory layout and clamped state path.
#[test]
#[serial(cuda)]
fn test_titans_mlp_backward_non_pow2_clamped() {
    run_backward_parity(4, 6, 3, 1, MemoryActivation::GELU, 0, 1.0, "bw_np2_clamp", 2e-3);
}
