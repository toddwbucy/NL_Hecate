// Test 07: Mini NL Forward Pass — Integration Test
//
// PURPOSE: Exercise the COMPLETE chain that NL_Hecate needs:
//   input → project(W_K, W_V) → memory_update(state, k, v) → gate → loss
//
// This is the integration test that composes:
//   - Enzyme-differentiated projection (struct with W_K, W_V)
//   - Hand-written memory update backward (kernel-pair simulation)
//   - Enzyme-differentiated gating and loss
//
// DESIGN NOTE: We split projections into separate scalar-returning functions
// because Enzyme's Rust wrapper has unclear support for tuple returns in
// #[autodiff]. Each projection (key, value) gets its own #[autodiff] wrapper.
// This is actually how NL_Hecate would structure it anyway — separate W_K and
// W_V projections are independent operations.
//
// If this test passes, we have strong evidence that the full NL_Hecate
// architecture can work with manual chain-rule composition at the
// kernel boundaries.

use std::autodiff::autodiff_reverse;

// ============================================================
// Mini NL Model
// ============================================================

/// Key projection weights (outer-loop parameter)
#[derive(Clone)]
struct KeyParams {
    w: [f32; 4],
}

/// Value projection weights (outer-loop parameter)
#[derive(Clone)]
struct ValParams {
    w: [f32; 4],
}

/// Inner-loop state (scoped to forward pass, NOT serialized).
#[derive(Clone, Copy)]
struct MemoryState {
    m: [f32; 4],
}

// ============================================================
// Forward pass components
// ============================================================

/// Dot product of two 4-vectors
fn dot4(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

/// Key projection: k_score = W_K · input
fn key_project(params: &KeyParams, input: &[f32; 4]) -> f32 {
    dot4(&params.w, input)
}

/// Value projection: v_score = W_V · input
fn val_project(params: &ValParams, input: &[f32; 4]) -> f32 {
    dot4(&params.w, input)
}

/// Memory update (the "kernel" — backward is hand-written)
/// Delta rule: M' = M + lr * (v - M) where lr = sigmoid(k_score)
fn memory_update(state: &MemoryState, k_score: f32, v_score: f32) -> f32 {
    let lr = 1.0 / (1.0 + (-k_score).exp());
    let m_mean = (state.m[0] + state.m[1] + state.m[2] + state.m[3]) / 4.0;
    m_mean + lr * (v_score - m_mean)
}

/// Hand-written backward for memory_update
/// Returns: (d_k_score, d_v_score, d_m_mean)
fn memory_update_backward(
    state: &MemoryState, k_score: f32, v_score: f32, d_out: f32,
) -> (f32, f32, f32) {
    let lr = 1.0 / (1.0 + (-k_score).exp());
    let m_mean = (state.m[0] + state.m[1] + state.m[2] + state.m[3]) / 4.0;

    let d_m_mean = d_out * (1.0 - lr);
    let d_v_score = d_out * lr;
    let d_lr = d_out * (v_score - m_mean);
    let d_k_score = d_lr * lr * (1.0 - lr);  // sigmoid derivative

    (d_k_score, d_v_score, d_m_mean)
}

/// Gate and loss: output = gate * memory_out + (1 - gate) * bypass
/// loss = (output - target)^2
fn gate_and_loss(gate: f32, memory_out: f32, bypass: f32, target: f32) -> f32 {
    let output = gate * memory_out + (1.0 - gate) * bypass;
    let diff = output - target;
    diff * diff
}

// ============================================================
// Enzyme-differentiable wrappers (scalar returns only)
// ============================================================

/// Key projection — Enzyme computes d/d(w_k) given upstream gradient as seed
#[autodiff_reverse(d_key_project, Duplicated, Const, Active)]
fn key_project_ad(params: &KeyParams, input: &[f32; 4]) -> f32 {
    key_project(params, input)
}

/// Value projection — Enzyme computes d/d(w_v) given upstream gradient as seed
#[autodiff_reverse(d_val_project, Duplicated, Const, Active)]
fn val_project_ad(params: &ValParams, input: &[f32; 4]) -> f32 {
    val_project(params, input)
}

/// Gate+loss — Enzyme computes d/d(gate), d/d(memory_out), d/d(bypass)
#[autodiff_reverse(d_gate_loss, Active, Active, Active, Const, Active)]
fn gate_and_loss_ad(gate: f32, memory_out: f32, bypass: f32, target: f32) -> f32 {
    gate_and_loss(gate, memory_out, bypass, target)
}

// ============================================================
// Full forward pass (for finite difference reference)
// ============================================================

fn full_forward(
    k_params: &KeyParams, v_params: &ValParams, gate: f32,
    input: &[f32; 4], state: &MemoryState, target: f32,
) -> f32 {
    let k_score = key_project(k_params, input);
    let v_score = val_project(v_params, input);
    let mem_out = memory_update(state, k_score, v_score);
    let bypass = v_score;
    gate_and_loss(gate, mem_out, bypass, target)
}

// ============================================================
// Manual chain rule: full d(loss)/d(params) using Enzyme + hand backward
// ============================================================

/// Compute d(loss)/d(w_k), d(loss)/d(w_v), d(loss)/d(gate)
/// using the three-stage composition:
///   Enzyme(gate+loss) × hand_backward(memory) × Enzyme(projection)
fn compute_gradients(
    k_params: &KeyParams, v_params: &ValParams, gate: f32,
    input: &[f32; 4], state: &MemoryState, target: f32,
) -> (f32, KeyParams, ValParams, f32) {
    // --- Forward pass ---
    let k_score = key_project(k_params, input);
    let v_score = val_project(v_params, input);
    let mem_out = memory_update(state, k_score, v_score);
    let bypass = v_score;

    // --- Stage 3 backward: Enzyme on gate+loss ---
    // Returns d(loss)/d(gate), d(loss)/d(mem_out), d(loss)/d(bypass)
    let (loss, d_gate, d_mem_out, d_bypass) = d_gate_loss(
        gate, mem_out, bypass, target, 1.0
    );

    // --- Stage 2 backward: hand-written kernel backward ---
    // Returns d(mem_out)/d(k_score), d(mem_out)/d(v_score)
    let (d_k_from_mem, d_v_from_mem, _d_m_mean) = memory_update_backward(
        state, k_score, v_score, d_mem_out
    );

    // Total upstream gradient on k_score and v_score:
    // k_score only flows through memory_update
    // v_score flows through both memory_update AND bypass
    let d_k_score_total = d_k_from_mem;
    let d_v_score_total = d_v_from_mem + d_bypass;

    // --- Stage 1 backward: Enzyme on projections ---
    // FINDING: For Duplicated parameters, Enzyme's seed does NOT scale the
    // shadow accumulation. The shadow always gets the unit-seed (seed=1.0)
    // Jacobian, regardless of the actual seed passed. We must manually
    // multiply by the upstream gradient afterward.
    //
    // Call with seed=1.0 to get d(score)/d(w[i]) = input[i],
    // then scale by upstream gradient.
    let mut d_k_params = KeyParams { w: [0.0; 4] };
    let _k_val = d_key_project(k_params, &mut d_k_params, input, 1.0);
    // Manual chain rule: d(loss)/d(w_k[i]) = d_k_score_total * d(k_score)/d(w_k[i])
    for i in 0..4 {
        d_k_params.w[i] *= d_k_score_total;
    }

    let mut d_v_params = ValParams { w: [0.0; 4] };
    let _v_val = d_val_project(v_params, &mut d_v_params, input, 1.0);
    for i in 0..4 {
        d_v_params.w[i] *= d_v_score_total;
    }

    (loss, d_k_params, d_v_params, d_gate)
}

// ============================================================
// Verification
// ============================================================

fn check_gradient(name: &str, computed: f32, reference: f32, tol: f32) -> bool {
    let abs_diff = (computed - reference).abs();
    let rel_diff = if reference.abs() > 1e-10 {
        abs_diff / reference.abs()
    } else {
        abs_diff
    };

    if rel_diff < tol {
        println!("  [PASS] {}: computed={:.6}, ref={:.6}, rel_err={:.2e}",
                 name, computed, reference, rel_diff);
        true
    } else {
        println!("  [FAIL] {}: computed={:.6}, ref={:.6}, rel_err={:.2e} (tol={:.2e})",
                 name, computed, reference, rel_diff, tol);
        false
    }
}

pub fn run() -> (usize, usize) {
    println!("\n=== Test 07: Mini NL Forward Pass (Integration) ===\n");
    println!("  Composition: Enzyme(projection) → hand(memory) → Enzyme(gate+loss)\n");

    let eps = 1e-4_f32;
    let tol = 1.5e-2_f32;  // Relaxed: composed chain rule compounds FD error across 3 stages
    let mut pass = 0;
    let mut fail = 0;

    let k_params = KeyParams { w: [0.5, -0.3, 0.8, 0.1] };
    let v_params = ValParams { w: [0.2, 0.6, -0.4, 0.7] };
    let gate = 0.7_f32;
    let input = [1.0_f32, -0.5, 0.3, 0.8];
    let state = MemoryState { m: [0.1, 0.2, 0.15, 0.25] };
    let target = 1.0_f32;

    // Compute gradients via Enzyme + hand backward composition
    let (loss, d_k, d_v, d_gate_val) = compute_gradients(
        &k_params, &v_params, gate, &input, &state, target
    );
    println!("  Forward loss = {:.6}", loss);

    // Verify against full finite differences (ground truth)
    println!("\n  Checking d_loss/d_w_k (Enzyme composed):");
    for i in 0..4 {
        let mut kp = k_params.clone();
        kp.w[i] += eps;
        let f_plus = full_forward(&kp, &v_params, gate, &input, &state, target);
        kp.w[i] -= 2.0 * eps;
        let f_minus = full_forward(&kp, &v_params, gate, &input, &state, target);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient(&format!("d_loss/d_w_k[{}]", i), d_k.w[i], fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!("\n  Checking d_loss/d_w_v (Enzyme composed):");
    for i in 0..4 {
        let mut vp = v_params.clone();
        vp.w[i] += eps;
        let f_plus = full_forward(&k_params, &vp, gate, &input, &state, target);
        vp.w[i] -= 2.0 * eps;
        let f_minus = full_forward(&k_params, &vp, gate, &input, &state, target);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient(&format!("d_loss/d_w_v[{}]", i), d_v.w[i], fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!("\n  Checking d_loss/d_gate (Enzyme direct):");
    {
        let f_plus = full_forward(&k_params, &v_params, gate + eps, &input, &state, target);
        let f_minus = full_forward(&k_params, &v_params, gate - eps, &input, &state, target);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient("d_loss/d_gate", d_gate_val, fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // --- Second test point to avoid symmetry accidents ---
    println!("\n  --- Second test point ---");
    let k2 = KeyParams { w: [-0.2, 0.9, 0.1, -0.5] };
    let v2 = ValParams { w: [0.4, -0.1, 0.6, 0.3] };
    let gate2 = 0.4_f32;
    let input2 = [0.7_f32, 0.2, -0.9, 0.4];
    let state2 = MemoryState { m: [0.5, -0.3, 0.1, 0.4] };
    let target2 = 0.5_f32;

    let (_loss2, d_k2, d_v2, d_gate2) = compute_gradients(
        &k2, &v2, gate2, &input2, &state2, target2
    );

    println!("\n  Checking d_loss/d_w_k (point 2):");
    for i in 0..4 {
        let mut kp = k2.clone();
        kp.w[i] += eps;
        let f_plus = full_forward(&kp, &v2, gate2, &input2, &state2, target2);
        kp.w[i] -= 2.0 * eps;
        let f_minus = full_forward(&kp, &v2, gate2, &input2, &state2, target2);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient(&format!("pt2: d_loss/d_w_k[{}]", i), d_k2.w[i], fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!("\n  Checking d_loss/d_w_v (point 2):");
    for i in 0..4 {
        let mut vp = v2.clone();
        vp.w[i] += eps;
        let f_plus = full_forward(&k2, &vp, gate2, &input2, &state2, target2);
        vp.w[i] -= 2.0 * eps;
        let f_minus = full_forward(&k2, &vp, gate2, &input2, &state2, target2);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient(&format!("pt2: d_loss/d_w_v[{}]", i), d_v2.w[i], fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // Gate check for point 2
    {
        let f_plus = full_forward(&k2, &v2, gate2 + eps, &input2, &state2, target2);
        let f_minus = full_forward(&k2, &v2, gate2 - eps, &input2, &state2, target2);
        let fd = (f_plus - f_minus) / (2.0 * eps);
        if check_gradient("pt2: d_loss/d_gate", d_gate2, fd, tol) {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    println!("\n  Result: {}/{} passed\n", pass, pass + fail);
    (pass, fail)
}
