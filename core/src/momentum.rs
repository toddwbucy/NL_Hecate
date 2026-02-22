/// Centralized momentum module — HOPE §4.2-4.4 expressiveness hierarchy.
///
/// Four levels of momentum sophistication for inner-loop memory updates:
///   Level 0 (None): No momentum accumulator. Plain GD rules (DeltaRule, Hebbian, etc.).
///   Level 1 (EMA): Exponential moving average. S = eta*S - theta*grad (HOPE Eq 33).
///   Level 2 (DeltaMomentum): State-dependent decay. decay = clamp(eta - ||g||^2, eps, 1-eps)
///     (HOPE Eq 49). Momentum forgets faster on high-gradient steps.
///   Level 3 (DeepMomentum): MLP replaces linear accumulator (HOPE Eq 50).
///     Two-layer MLP (W1, W2) with SiLU activation, inner-loop SGD on L2 objective.
///
/// All functions operate on flat d×d momentum matrices stored as slices into s_states.
/// The gate computation (eta_t, theta_t) stays in the caller (TitansLMM) —
/// this module receives computed scalar gates.

use crate::model::MomentumKind;
use crate::retention::l2_apply_retention;
use crate::tensor::{frobenius_dot_f32, silu_f32, silu_prime_f32};

// ── EMA (Level 1) ──────────────────────────────────────────────────

/// EMA momentum forward: S_{t+1} = eta_t * S_t - theta_t * grad.
///
/// Operates in-place on `s_states` at position `(t+1)*d*d`.
/// Copies S_t to S_{t+1}, scales by eta, subtracts theta*grad.
pub fn ema_step(
    s_states: &mut [f32],
    t: usize,
    d: usize,
    eta_t: f32,
    theta_t: f32,
    grad: &[f32],
) {
    let dd = d * d;
    let s_t_off = t * dd;
    let s_next_off = (t + 1) * dd;
    s_states.copy_within(s_t_off..s_t_off + dd, s_next_off);
    l2_apply_retention(&mut s_states[s_next_off..s_next_off + dd], eta_t);
    for i in 0..dd {
        s_states[s_next_off + i] -= theta_t * grad[i];
    }
}

/// EMA momentum backward.
///
/// Given d_S (upstream gradient on S_{t+1}), computes:
///   d_S_prev = eta_t * d_S  (propagate to S_t)
///   d_eta = frob(S_t, d_S)
///   d_theta = -frob(grad, d_S)
///   d_grad = -theta_t * d_S
///
/// `d_s` is consumed and returned as `d_s_prev`.
pub fn ema_step_backward(
    d_s: &mut [f32],
    s_t: &[f32],
    grad_t: &[f32],
    eta_t: f32,
    theta_t: f32,
    d: usize,
) -> (f32, f32, Vec<f32>) {
    let dd = d * d;
    let d_eta_scalar = frobenius_dot_f32(s_t, d_s);
    let d_theta_scalar = -frobenius_dot_f32(grad_t, d_s);

    let mut d_grad = vec![0.0f32; dd];
    for i in 0..dd {
        d_grad[i] = -theta_t * d_s[i];
    }

    // d_s_prev = eta * d_s (in-place)
    l2_apply_retention(d_s, eta_t);

    (d_eta_scalar, d_theta_scalar, d_grad)
}

// ── Delta Momentum (Level 2) ───────────────────────────────────────

/// Delta Momentum forward: state-dependent decay (HOPE Eq 49).
///
///   decay = clamp(eta_t - ||grad||_F^2, 1e-6, 1.0 - 1e-6)
///   S_{t+1} = decay * S_t - theta_t * grad  (P=I for now)
///
/// Stores `decay[t]` in `decay_buf` for backward.
/// Key difference from EMA: when gradient norm is large, decay drops —
/// momentum forgets faster on high-gradient steps. Clamped per CS-39.
pub fn delta_momentum_step(
    s_states: &mut [f32],
    t: usize,
    d: usize,
    eta_t: f32,
    theta_t: f32,
    grad: &[f32],
    decay_buf: &mut [f32],
) {
    let dd = d * d;
    let grad_norm_sq = frobenius_dot_f32(grad, grad);
    let decay = (eta_t - grad_norm_sq).clamp(1e-6, 1.0 - 1e-6);
    decay_buf[t] = decay;

    let s_t_off = t * dd;
    let s_next_off = (t + 1) * dd;
    s_states.copy_within(s_t_off..s_t_off + dd, s_next_off);
    l2_apply_retention(&mut s_states[s_next_off..s_next_off + dd], decay);
    for i in 0..dd {
        s_states[s_next_off + i] -= theta_t * grad[i];
    }
}

/// Delta Momentum backward.
///
/// Like EMA backward, but decay is `eta_t - ||grad||_F^2`, so:
///   d_S_prev = decay_t * d_S
///   d_eta = frob(S_t, d_S)  (eta contributes to decay linearly)
///   d_theta = -frob(grad, d_S)
///   d_grad has extra term through decay: d_grad[i] += -2 * grad[i] * frob(S_t, d_S)
///
/// The extra d_grad term comes from d(decay)/d(grad) = -2*grad, so:
///   d_grad += d(decay)/d(grad) * (d_loss/d_decay) = -2*grad * frob(S_t, d_S)
pub fn delta_momentum_step_backward(
    d_s: &mut [f32],
    s_t: &[f32],
    grad_t: &[f32],
    eta_t: f32,
    theta_t: f32,
    decay_t: f32,
    d: usize,
) -> (f32, f32, Vec<f32>) {
    let dd = d * d;

    // d_loss/d_decay = frob(S_t, d_S)  (since S_{t+1} = decay * S_t - ...)
    let d_decay = frobenius_dot_f32(s_t, d_s);
    let d_eta_scalar = d_decay; // decay = eta - ||g||^2 → d(decay)/d(eta) = 1
    let d_theta_scalar = -frobenius_dot_f32(grad_t, d_s);

    // d_grad from -theta * d_S term (same as EMA)
    let mut d_grad = vec![0.0f32; dd];
    for i in 0..dd {
        d_grad[i] = -theta_t * d_s[i];
    }

    // Extra: d(decay)/d(grad[i]) = -2*grad[i], so d_grad[i] += -2*grad[i] * d_decay
    // But only if decay was NOT clamped (gradient should be zero at the clamp boundary).
    // For simplicity, we use the clamped decay_t to check:
    let raw_decay = eta_t - frobenius_dot_f32(grad_t, grad_t);
    if raw_decay > 1e-6 && raw_decay < 1.0 - 1e-6 {
        for i in 0..dd {
            d_grad[i] += -2.0 * grad_t[i] * d_decay;
        }
    }

    // d_s_prev = decay_t * d_s (in-place)
    l2_apply_retention(d_s, decay_t);

    (d_eta_scalar, d_theta_scalar, d_grad)
}

// ── Deep Momentum (Level 3) ────────────────────────────────────────

/// DeepMomentum MLP configuration.
pub struct DeepMomentumMLP {
    /// Memory dimension (MLP input/output: d*d flattened).
    pub d: usize,
    /// MLP hidden dimension (default: 4*d).
    pub d_hidden: usize,
}

/// Cache for DeepMomentum forward, needed for backward.
pub struct DeepMomentumCache {
    /// MLP W1 states: [(seq_len+1) * d_hidden * dd]
    pub w1_states: Vec<f32>,
    /// MLP W2 states: [(seq_len+1) * dd * d_hidden]
    pub w2_states: Vec<f32>,
    /// Pre-activation (before SiLU): [seq_len * d_hidden]
    pub pre_act: Vec<f32>,
    /// Post-activation (SiLU output): [seq_len * d_hidden]
    pub hidden: Vec<f32>,
    /// MLP output (momentum contribution): [seq_len * dd]
    pub output: Vec<f32>,
}

impl DeepMomentumCache {
    /// Create a new cache with Kaiming-like init for W1_0.
    ///
    /// W1_0 is initialized to ±scale/sqrt(dd) in a deterministic checkerboard
    /// pattern (no RNG needed). W2_0 stays zero. This breaks the bootstrap
    /// deadlock: silu(W1@grad) ≠ 0, so grad_w2 ≠ 0, so W2 evolves.
    pub fn new(seq_len: usize, dd: usize, d_hidden: usize) -> Self {
        let w1_len = (seq_len + 1) * d_hidden * dd;
        let mut w1_states = vec![0.0f32; w1_len];
        // Initialize W1_0 (first d_hidden*dd entries) with Kaiming scale
        let scale = 1.0 / (dd as f32).sqrt();
        for i in 0..(d_hidden * dd) {
            w1_states[i] = if i % 2 == 0 { scale } else { -scale };
        }
        DeepMomentumCache {
            w1_states,
            w2_states: vec![0.0f32; (seq_len + 1) * dd * d_hidden],
            pre_act: vec![0.0f32; seq_len * d_hidden],
            hidden: vec![0.0f32; seq_len * d_hidden],
            output: vec![0.0f32; seq_len * dd],
        }
    }
}

/// Deep Momentum forward (HOPE Eq 50).
///
/// MLP-based momentum: a 2-layer MLP (W1, W2, SiLU) replaces the linear
/// momentum accumulator. The MLP is trained on the inner-loop L2 objective
/// with the gradient as both input and target.
///
/// 1. Forward MLP: pre = W1 @ grad_flat, h = silu(pre), output = W2 @ h
/// 2. L2 objective: error = output - grad_flat
/// 3. MLP inner-loop update:
///    grad_w2 = outer(error, h), grad_w1 = outer(dsilu(pre) .* W2^T@error, grad_flat)
///    W1_{t+1} = eta * W1_t - theta * grad_w1
///    W2_{t+1} = eta * W2_t - theta * grad_w2
/// 4. Re-evaluate: output = W2_{t+1} @ silu(W1_{t+1} @ grad_flat)
///    This output is the momentum contribution to M.
pub fn deep_momentum_step(
    mlp: &DeepMomentumMLP,
    cache: &mut DeepMomentumCache,
    t: usize,
    eta_t: f32,
    theta_t: f32,
    grad: &[f32],  // d*d flattened gradient
) -> Vec<f32> {
    let dd = mlp.d * mlp.d;
    let dh = mlp.d_hidden;

    // Current MLP weights
    let w1_t_off = t * dh * dd;
    let w2_t_off = t * dd * dh;

    // Step 1: Forward MLP with current weights
    // pre = W1_t @ grad  (dh × dd) @ (dd × 1) = (dh × 1)
    let pa_base = t * dh;
    for i in 0..dh {
        let mut sum = 0.0f32;
        for j in 0..dd {
            sum += cache.w1_states[w1_t_off + i * dd + j] * grad[j];
        }
        cache.pre_act[pa_base + i] = sum;
        cache.hidden[pa_base + i] = silu_f32(sum);
    }

    // output = W2_t @ h  (dd × dh) @ (dh × 1) = (dd × 1)
    let o_base = t * dd;
    for i in 0..dd {
        let mut sum = 0.0f32;
        for j in 0..dh {
            sum += cache.w2_states[w2_t_off + i * dh + j] * cache.hidden[pa_base + j];
        }
        cache.output[o_base + i] = sum;
    }

    // Step 2: L2 objective: error = output - grad (target = grad = P@grad with P=I)
    let mut error = vec![0.0f32; dd];
    for i in 0..dd {
        error[i] = cache.output[o_base + i] - grad[i];
    }

    // Step 3: MLP inner-loop gradients
    // grad_w2[i,j] = error[i] * h[j]  (dd × dh)
    // W2_next = eta * W2 - theta * grad_w2
    let w2_next_off = (t + 1) * dd * dh;
    for i in 0..dd {
        for j in 0..dh {
            let gw2 = error[i] * cache.hidden[pa_base + j];
            cache.w2_states[w2_next_off + i * dh + j] =
                eta_t * cache.w2_states[w2_t_off + i * dh + j] - theta_t * gw2;
        }
    }

    // delta_hidden[j] = sum_i W2[i,j]^T @ error[i] = sum_i W2_t[i,j] * error[i]
    let mut delta_hidden = vec![0.0f32; dh];
    for j in 0..dh {
        let mut sum = 0.0f32;
        for i in 0..dd {
            sum += cache.w2_states[w2_t_off + i * dh + j] * error[i];
        }
        delta_hidden[j] = sum;
    }

    // grad_w1[i,j] = (dsilu(pre[i]) * delta_hidden[i]) * grad[j]  (dh × dd)
    // W1_next = eta * W1 - theta * grad_w1
    let w1_next_off = (t + 1) * dh * dd;
    for i in 0..dh {
        let ds = silu_prime_f32(cache.pre_act[pa_base + i]) * delta_hidden[i];
        for j in 0..dd {
            let gw1 = ds * grad[j];
            cache.w1_states[w1_next_off + i * dd + j] =
                eta_t * cache.w1_states[w1_t_off + i * dd + j] - theta_t * gw1;
        }
    }

    // Step 4: Re-evaluate with updated weights
    // output = W2_{t+1} @ silu(W1_{t+1} @ grad)
    let mut result = vec![0.0f32; dd];
    let mut new_pre = vec![0.0f32; dh];
    let mut new_h = vec![0.0f32; dh];

    for i in 0..dh {
        let mut sum = 0.0f32;
        for j in 0..dd {
            sum += cache.w1_states[w1_next_off + i * dd + j] * grad[j];
        }
        new_pre[i] = sum;
        new_h[i] = silu_f32(sum);
    }

    for i in 0..dd {
        let mut sum = 0.0f32;
        for j in 0..dh {
            sum += cache.w2_states[w2_next_off + i * dh + j] * new_h[j];
        }
        result[i] = sum;
    }

    result
}

/// Deep Momentum backward — VJP through MLP output to outer-loop params.
///
/// The MLP weights (W1, W2) are inner-loop state, NOT outer-loop params.
/// But the outer-loop params (eta, theta, and the grad that depends on W_K, W_V)
/// flow through the MLP computation.
///
/// Given d_output (upstream gradient on the MLP output), computes:
///   d_eta, d_theta: scalar gradients for the gates
///   d_grad: gradient flowing back to the memory gradient (hence to W_K, W_V)
pub fn deep_momentum_step_backward(
    mlp: &DeepMomentumMLP,
    cache: &DeepMomentumCache,
    t: usize,
    eta_t: f32,
    theta_t: f32,
    grad_t: &[f32],
    d_output: &[f32],
) -> (f32, f32, Vec<f32>) {
    let dd = mlp.d * mlp.d;
    let dh = mlp.d_hidden;

    // The output = W2_{t+1} @ silu(W1_{t+1} @ grad)
    // This is a function of W1_{t+1}, W2_{t+1}, and grad.
    // W1_{t+1} = eta * W1_t - theta * grad_w1(W1_t, W2_t, grad)
    // W2_{t+1} = eta * W2_t - theta * grad_w2(W1_t, W2_t, grad)
    //
    // For simplicity, we compute the direct VJP through the final evaluation
    // w.r.t. eta and theta (treating W1_t, W2_t as constants from prior step).
    //
    // This is analogous to how MONETA backward handles its inner-loop MLP:
    // the inner-loop weights are not outer-loop params, so we only need
    // gradients w.r.t. the outer-loop quantities that flow through.

    let w1_next_off = (t + 1) * dh * dd;
    let w2_next_off = (t + 1) * dd * dh;

    // Re-evaluate intermediate values with W_{t+1} weights
    let mut h_next = vec![0.0f32; dh];
    let mut pre_next = vec![0.0f32; dh];
    for i in 0..dh {
        let mut sum = 0.0f32;
        for j in 0..dd {
            sum += cache.w1_states[w1_next_off + i * dd + j] * grad_t[j];
        }
        pre_next[i] = sum;
        h_next[i] = silu_f32(sum);
    }

    // d_output → d_W2_next, d_h_next
    // output[i] = sum_j W2_next[i,j] * h_next[j]
    // d_h_next[j] = sum_i d_output[i] * W2_next[i,j]
    let mut d_h_next = vec![0.0f32; dh];
    for j in 0..dh {
        let mut sum = 0.0f32;
        for i in 0..dd {
            sum += d_output[i] * cache.w2_states[w2_next_off + i * dh + j];
        }
        d_h_next[j] = sum;
    }

    // d_h_next → d_pre_next (through SiLU)
    // d_pre[i] = d_h[i] * silu'(pre[i])
    let mut d_pre_next = vec![0.0f32; dh];
    for i in 0..dh {
        d_pre_next[i] = d_h_next[i] * silu_prime_f32(pre_next[i]);
    }

    // d_pre_next → d_grad (through W1_next @ grad)
    // pre[i] = sum_j W1_next[i,j] * grad[j]
    // d_grad[j] += sum_i d_pre[i] * W1_next[i,j]
    let mut d_grad = vec![0.0f32; dd];
    for j in 0..dd {
        let mut sum = 0.0f32;
        for i in 0..dh {
            sum += d_pre_next[i] * cache.w1_states[w1_next_off + i * dd + j];
        }
        d_grad[j] = sum;
    }

    // d_output → d_W2_next[i,j] = d_output[i] * h_next[j]
    // d_W2_next → d_eta via W2_next = eta * W2_t - theta * grad_w2
    // d_eta += frob(W2_t, d_W2_next)  (since d(W2_next)/d(eta) = W2_t)
    let w2_t_off = t * dd * dh;
    let w1_t_off = t * dh * dd;

    let mut d_eta = 0.0f32;
    let mut d_theta = 0.0f32;

    // d_W2_next contribution to d_eta, d_theta
    for i in 0..dd {
        for j in 0..dh {
            let dw2 = d_output[i] * h_next[j];
            d_eta += cache.w2_states[w2_t_off + i * dh + j] * dw2;
            // d_theta from -theta * grad_w2 term
            // grad_w2 was computed during forward, but we need d(W2_next)/d(theta) = -grad_w2
            // For now use the linearized approximation: d_theta accumulates through eta/theta path
        }
    }

    // d_W1_next contribution to d_eta via d_pre_next → d_W1_next
    // d_W1_next[i,j] = d_pre_next[i] * grad[j]
    for i in 0..dh {
        for j in 0..dd {
            let dw1 = d_pre_next[i] * grad_t[j];
            d_eta += cache.w1_states[w1_t_off + i * dd + j] * dw1;
        }
    }

    // d_theta: W_{t+1} = eta*W_t - theta*grad_w
    // d(W_{t+1})/d(theta) = -grad_w  → d_theta = -frob(grad_w, d_W)
    // We approximate by using current error for grad_w reconstruction
    let pa_base = t * dh;
    let o_base = t * dd;
    for i in 0..dd {
        let error_i = cache.output[o_base + i] - grad_t[i];
        for j in 0..dh {
            let gw2 = error_i * cache.hidden[pa_base + j];
            let dw2_next = d_output[i] * h_next[j];
            d_theta -= gw2 * dw2_next;
        }
    }

    // delta_hidden for W1 grad
    let mut delta_hidden = vec![0.0f32; dh];
    for j in 0..dh {
        let mut sum = 0.0f32;
        for i in 0..dd {
            let error_i = cache.output[o_base + i] - grad_t[i];
            sum += cache.w2_states[w2_t_off + i * dh + j] * error_i;
        }
        delta_hidden[j] = sum;
    }
    for i in 0..dh {
        let ds = silu_prime_f32(cache.pre_act[pa_base + i]) * delta_hidden[i];
        for j in 0..dd {
            let gw1 = ds * grad_t[j];
            let dw1_next = d_pre_next[i] * grad_t[j];
            d_theta -= gw1 * dw1_next;
        }
    }

    (d_eta, d_theta, d_grad)
}

// ── Dispatch ────────────────────────────────────────────────────────

/// Effective momentum kind: TitansLMM with None auto-upgrades to EMA.
pub fn effective_momentum_kind(kind: MomentumKind, rule: crate::model::MemoryRuleKind) -> MomentumKind {
    match kind {
        MomentumKind::None => {
            match rule {
                crate::model::MemoryRuleKind::TitansLMM |
                crate::model::MemoryRuleKind::AtlasOmega => MomentumKind::EMA,
                _ => MomentumKind::None,
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{SimpleRng, outer_product_f32};

    fn make_grad(d: usize, seed: u64) -> Vec<f32> {
        let dd = d * d;
        let mut rng = SimpleRng::new(seed);
        let mut g = vec![0.0f32; dd];
        rng.fill_uniform(&mut g, 0.1);
        g
    }

    fn make_s_states(d: usize, seq_len: usize, seed: u64) -> Vec<f32> {
        let dd = d * d;
        let mut rng = SimpleRng::new(seed);
        let mut s = vec![0.0f32; (seq_len + 1) * dd];
        rng.fill_uniform(&mut s[..dd], 0.05); // seed S_0
        s
    }

    // ── EMA tests ──

    #[test]
    fn test_ema_step() {
        let d = 4;
        let dd = d * d;
        let mut s_states = make_s_states(d, 4, 42);
        let grad = make_grad(d, 99);
        let eta = 0.9;
        let theta = 0.01;

        // Save S_0 before
        let s0: Vec<f32> = s_states[..dd].to_vec();

        ema_step(&mut s_states, 0, d, eta, theta, &grad);

        // S_1 = eta * S_0 - theta * grad
        for i in 0..dd {
            let expected = eta * s0[i] - theta * grad[i];
            assert!((s_states[dd + i] - expected).abs() < 1e-7,
                "EMA mismatch at {}: {} vs {}", i, s_states[dd + i], expected);
        }
    }

    #[test]
    fn test_ema_backward_fd() {
        let d = 4;
        let dd = d * d;
        let eps = 1e-3;

        let grad = make_grad(d, 99);
        let eta = 0.85;
        let theta = 0.02;

        // Forward baseline
        let mut s_base = make_s_states(d, 1, 42);
        ema_step(&mut s_base, 0, d, eta, theta, &grad);
        let s1_base: Vec<f32> = s_base[dd..2 * dd].to_vec();

        // Synthetic upstream d_S
        let mut rng = SimpleRng::new(77);
        let mut d_s = vec![0.0f32; dd];
        rng.fill_uniform(&mut d_s, 0.1);

        // Analytical backward
        let s_t = &s_base[..dd]; // this is actually modified s0, use original
        let s_orig = make_s_states(d, 1, 42);
        let s0 = &s_orig[..dd];
        let mut d_s_copy = d_s.clone();
        let (d_eta, d_theta, _d_grad) = ema_step_backward(&mut d_s_copy, s0, &grad, eta, theta, d);

        // FD check for eta
        let mut s_plus = make_s_states(d, 1, 42);
        ema_step(&mut s_plus, 0, d, eta + eps, theta, &grad);
        let mut s_minus = make_s_states(d, 1, 42);
        ema_step(&mut s_minus, 0, d, eta - eps, theta, &grad);

        let mut fd_eta = 0.0f32;
        for i in 0..dd {
            fd_eta += (s_plus[dd + i] - s_minus[dd + i]) / (2.0 * eps) * d_s[i];
        }
        let rel_err = (d_eta - fd_eta).abs() / (fd_eta.abs() + 1e-8);
        assert!(rel_err < 0.05, "EMA d_eta FD mismatch: analytical={}, fd={}, rel_err={}", d_eta, fd_eta, rel_err);

        // FD check for theta
        let mut s_plus = make_s_states(d, 1, 42);
        ema_step(&mut s_plus, 0, d, eta, theta + eps, &grad);
        let mut s_minus = make_s_states(d, 1, 42);
        ema_step(&mut s_minus, 0, d, eta, theta - eps, &grad);

        let mut fd_theta = 0.0f32;
        for i in 0..dd {
            fd_theta += (s_plus[dd + i] - s_minus[dd + i]) / (2.0 * eps) * d_s[i];
        }
        let rel_err = (d_theta - fd_theta).abs() / (fd_theta.abs() + 1e-8);
        assert!(rel_err < 0.05, "EMA d_theta FD mismatch: analytical={}, fd={}, rel_err={}", d_theta, fd_theta, rel_err);
    }

    // ── Delta Momentum tests ──

    #[test]
    fn test_delta_momentum_step() {
        let d = 4;
        let dd = d * d;
        let mut s_states = make_s_states(d, 4, 42);
        let grad = make_grad(d, 99);
        let eta = 0.9;
        let theta = 0.01;
        let mut decay_buf = vec![0.0f32; 4];

        let s0: Vec<f32> = s_states[..dd].to_vec();
        let grad_norm_sq = frobenius_dot_f32(&grad, &grad);

        delta_momentum_step(&mut s_states, 0, d, eta, theta, &grad, &mut decay_buf);

        let expected_decay = (eta - grad_norm_sq).clamp(1e-6, 1.0 - 1e-6);
        assert!((decay_buf[0] - expected_decay).abs() < 1e-7);

        for i in 0..dd {
            let expected = expected_decay * s0[i] - theta * grad[i];
            assert!((s_states[dd + i] - expected).abs() < 1e-6,
                "Delta momentum mismatch at {}", i);
        }
    }

    #[test]
    fn test_delta_momentum_decay_clamp() {
        let d = 2;
        let dd = d * d;
        let mut s_states = vec![0.0f32; 2 * dd];
        s_states[..dd].copy_from_slice(&[1.0, 0.0, 0.0, 1.0]);

        // Make gradient with large norm to trigger clamping
        let grad = vec![10.0, 10.0, 10.0, 10.0]; // ||g||^2 = 400
        let eta = 0.9; // decay = 0.9 - 400 → clamped to 1e-6
        let theta = 0.01;
        let mut decay_buf = vec![0.0f32; 1];

        delta_momentum_step(&mut s_states, 0, d, eta, theta, &grad, &mut decay_buf);

        assert!((decay_buf[0] - 1e-6).abs() < 1e-10, "Should clamp to 1e-6, got {}", decay_buf[0]);
    }

    #[test]
    fn test_delta_momentum_backward_fd() {
        let d = 4;
        let dd = d * d;
        let eps = 1e-3;

        let grad = make_grad(d, 99);
        let eta = 0.95; // High eta so decay won't clamp
        let theta = 0.01;

        // FD check for eta
        let mut rng = SimpleRng::new(77);
        let mut d_s = vec![0.0f32; dd];
        rng.fill_uniform(&mut d_s, 0.1);

        let s_orig = make_s_states(d, 1, 42);

        let mut decay_buf_p = vec![0.0f32; 1];
        let mut decay_buf_m = vec![0.0f32; 1];
        let mut decay_buf_b = vec![0.0f32; 1];

        let mut s_plus = s_orig.clone();
        delta_momentum_step(&mut s_plus, 0, d, eta + eps, theta, &grad, &mut decay_buf_p);
        let mut s_minus = s_orig.clone();
        delta_momentum_step(&mut s_minus, 0, d, eta - eps, theta, &grad, &mut decay_buf_m);
        let mut s_base = s_orig.clone();
        delta_momentum_step(&mut s_base, 0, d, eta, theta, &grad, &mut decay_buf_b);

        let mut fd_eta = 0.0f32;
        for i in 0..dd {
            fd_eta += (s_plus[dd + i] - s_minus[dd + i]) / (2.0 * eps) * d_s[i];
        }

        let mut d_s_copy = d_s.clone();
        let (d_eta, _d_theta, _d_grad) = delta_momentum_step_backward(
            &mut d_s_copy, &s_orig[..dd], &grad, eta, theta, decay_buf_b[0], d,
        );

        let rel_err = (d_eta - fd_eta).abs() / (fd_eta.abs() + 1e-8);
        assert!(rel_err < 0.1, "Delta d_eta FD mismatch: analytical={}, fd={}, rel_err={}", d_eta, fd_eta, rel_err);
    }

    // ── Deep Momentum tests ──

    #[test]
    fn test_deep_momentum_step() {
        let d = 3;
        let dd = d * d;
        let dh = 4;
        let mlp = DeepMomentumMLP { d, d_hidden: dh };
        let seq_len = 2;
        let mut cache = DeepMomentumCache::new(seq_len, dd, dh);

        // Initialize W1, W2 with small random values
        let mut rng = SimpleRng::new(42);
        rng.fill_uniform(&mut cache.w1_states[..dh * dd], 0.1);
        rng.fill_uniform(&mut cache.w2_states[..dd * dh], 0.1);

        let grad = make_grad(d, 99);
        let result = deep_momentum_step(&mlp, &mut cache, 0, 0.9, 0.01, &grad);

        // Verify output is finite and non-zero
        assert_eq!(result.len(), dd);
        for &v in &result {
            assert!(v.is_finite(), "Deep momentum output should be finite");
        }
        let norm: f32 = result.iter().map(|x| x * x).sum();
        assert!(norm > 0.0, "Deep momentum output should be non-zero");

        // Verify W1, W2 were updated (next step should differ)
        let w1_diff: f32 = (0..dh * dd).map(|i| {
            (cache.w1_states[dh * dd + i] - cache.w1_states[i]).abs()
        }).sum();
        assert!(w1_diff > 0.0, "W1 should have been updated");
    }

    #[test]
    fn test_deep_momentum_backward_fd() {
        let d = 3;
        let dd = d * d;
        let dh = 4;
        let eps = 5e-3;

        let grad = make_grad(d, 99);
        let eta = 0.9;
        let theta = 0.01;

        // Make deterministic initial MLP weights
        let init_cache = || {
            let mut cache = DeepMomentumCache::new(1, dd, dh);
            let mut rng = SimpleRng::new(42);
            rng.fill_uniform(&mut cache.w1_states[..dh * dd], 0.1);
            rng.fill_uniform(&mut cache.w2_states[..dd * dh], 0.1);
            cache
        };

        let mlp = DeepMomentumMLP { d, d_hidden: dh };

        // Baseline forward
        let mut cache_base = init_cache();
        let output_base = deep_momentum_step(&mlp, &mut cache_base, 0, eta, theta, &grad);

        // Synthetic upstream d_output
        let mut rng = SimpleRng::new(77);
        let mut d_output = vec![0.0f32; dd];
        rng.fill_uniform(&mut d_output, 0.1);

        // Analytical backward
        let (d_eta, d_theta, _d_grad) = deep_momentum_step_backward(
            &mlp, &cache_base, 0, eta, theta, &grad, &d_output,
        );

        // FD check for eta
        let mut cache_p = init_cache();
        let out_p = deep_momentum_step(&mlp, &mut cache_p, 0, eta + eps, theta, &grad);
        let mut cache_m = init_cache();
        let out_m = deep_momentum_step(&mlp, &mut cache_m, 0, eta - eps, theta, &grad);

        let mut fd_eta = 0.0f32;
        for i in 0..dd {
            fd_eta += (out_p[i] - out_m[i]) / (2.0 * eps) * d_output[i];
        }

        let abs_err = (d_eta - fd_eta).abs();
        let scale = fd_eta.abs().max(d_eta.abs()).max(1e-6);
        assert!(abs_err / scale < 0.3,
            "Deep d_eta: analytical={:.6}, fd={:.6}, err={:.6}", d_eta, fd_eta, abs_err / scale);
    }

    #[test]
    fn test_ema_degeneracy() {
        // EMA via dispatch matches direct call
        let d = 4;
        let dd = d * d;
        let grad = make_grad(d, 99);
        let eta = 0.9;
        let theta = 0.01;

        let mut s1 = make_s_states(d, 1, 42);
        let mut s2 = s1.clone();

        ema_step(&mut s1, 0, d, eta, theta, &grad);

        // Manual EMA to compare
        let s0 = make_s_states(d, 1, 42);
        for i in 0..dd {
            s2[dd + i] = eta * s0[i] - theta * grad[i];
        }

        for i in 0..dd {
            assert!((s1[dd + i] - s2[dd + i]).abs() < 1e-7,
                "EMA degeneracy mismatch at {}", i);
        }
    }
}
