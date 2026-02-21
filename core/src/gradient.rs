/// Gradient orchestration and verification.
///
/// Provides:
/// - `compute_gradients`: main API for computing SWA parameter gradients
/// - `mag_compute_gradients`: main API for computing MAG parameter gradients
/// - `finite_diff_gradient`: central finite differences for verification
/// - Gradient checking utilities

use crate::model::{SWAConfig, SWAParams, MAGConfig, MAGParams, MemoryRuleKind, MemoryLevelParams};
use crate::forward::forward;
use crate::backward::backward_full;
use crate::mag::{mag_forward, mag_backward, cms_forward, cms_backward};
use crate::mal::{mal_forward, mal_backward, cms_mal_forward, cms_mal_backward};
use crate::mac::{mac_forward, mac_backward, cms_mac_forward, cms_mac_backward};
use crate::conductor::{Pulse, ContextState, ErrorBuffer};
use crate::tape::with_tape;
use crate::traced_forward::traced_cms_forward;
use crate::opaque_adapters::{register_opaque_vjps, level_params_from_flat};
use crate::dynamic_freq::{compute_gate_surrogate, freq_gate_backward};

/// Create a ContextState with the correct memory size per level for the given config.
/// MONETA uses W1+W2 (d_hidden*d + d*d_hidden), other rules use d*d.
fn make_context_state(cfg: &MAGConfig) -> ContextState {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
            let dh = cfg.d_hidden;
            let d = cfg.swa.d_model;
            let mem_size = dh * d + d * dh;
            ContextState::new_with_memory_size(cfg.k, d, mem_size)
        }
        MemoryRuleKind::LatticeOSR => {
            let d = cfg.swa.d_model;
            let mem_size = cfg.m_slots * d;
            ContextState::new_with_memory_size(cfg.k, d, mem_size)
        }
        MemoryRuleKind::Trellis => {
            let d = cfg.swa.d_model;
            let mem_size = 2 * cfg.d_compress * d;
            ContextState::new_with_memory_size(cfg.k, d, mem_size)
        }
        _ => ContextState::new(cfg.k, cfg.swa.d_model),
    }
}

/// Compute gradients of loss with respect to all SWA parameters.
/// This is the main training API — used by Python bindings in Phase 3.
#[allow(dead_code)]
pub fn compute_gradients(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, SWAParams) {
    let (loss, cache) = forward(params, cfg, input_ids, target_ids);
    let grads = backward_full(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Compute gradients of loss with respect to all MAG parameters.
#[allow(dead_code)]
pub fn mag_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MAGParams) {
    let (loss, cache) = mag_forward(params, cfg, input_ids, target_ids);
    let grads = mag_backward(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Compute finite-difference gradient for a single weight element (SWA).
/// Uses central differences: (f(x+eps) - f(x-eps)) / (2*eps).
#[allow(dead_code)]
fn fd_single(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&SWAParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut SWAParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Compute finite-difference gradient for a single weight element (MAG).
#[allow(dead_code)]
fn mag_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = mag_forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = mag_forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (SWA).
/// Returns (num_checked, num_passed, max_relative_error).
///
/// Uses relative error with denominator = max(|a|, |b|, abs_threshold) to
/// avoid division by near-zero values. Gradients where both analytical and
/// numerical are below `abs_threshold` are auto-passed (below FD resolution).
#[allow(dead_code)]
pub(crate) fn check_weight_gradient(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &SWAParams,
    name: &str,
    get_weight: impl Fn(&SWAParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut SWAParams, usize, f32),
    get_grad: impl Fn(&SWAParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();

    #[cfg(feature = "cuda")]
    let abs_threshold = 5e-3;
    #[cfg(not(feature = "cuda"))]
    let abs_threshold = 5e-4;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );

        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());

        checked += 1;

        if denom < abs_threshold {
            passed += 1;
            continue;
        }

        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }

        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!(
                "  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, \
                 rel_err={rel_err:.4e}"
            );
        }
    }

    (checked, passed, max_rel_err)
}

/// Check gradient for a specific weight matrix (MAG).
/// Returns (num_checked, num_passed, max_relative_error).
#[allow(dead_code)]
pub(crate) fn mag_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();

    #[cfg(feature = "cuda")]
    let abs_threshold = 5e-3;
    #[cfg(not(feature = "cuda"))]
    let abs_threshold = 5e-4;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = mag_fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );

        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());

        checked += 1;

        if denom < abs_threshold {
            passed += 1;
            continue;
        }

        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }

        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!(
                "  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, \
                 rel_err={rel_err:.4e}"
            );
        }
    }

    (checked, passed, max_rel_err)
}

/// Compute gradients of loss with respect to all CMS parameters.
/// Delegates to `tape_compute_gradients()` (Wengert tape path).
///
/// The hand-written backward path is preserved as `cms_compute_gradients_handwritten()`
/// for use as a test oracle.
#[allow(dead_code)]
pub fn cms_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> (f32, MAGParams) {
    tape_compute_gradients(params, cfg, input_ids, target_ids, pulse, context, error_buffers)
}

/// Hand-written backward path (cms_forward + cms_backward).
/// Preserved as test oracle for Class 3 comparisons.
#[allow(dead_code)]
pub fn cms_compute_gradients_handwritten(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> (f32, MAGParams) {
    debug_assert_eq!(error_buffers.len(), cfg.k,
        "error_buffers length ({}) must equal cfg.k ({})", error_buffers.len(), cfg.k);
    let (loss, cache) = cms_forward(params, cfg, input_ids, target_ids, pulse, context);
    let grads = cms_backward(params, cfg, &cache, input_ids, target_ids, error_buffers);
    (loss, grads)
}

/// Compute gradients via the Wengert tape (traced forward + automatic backward).
///
/// Runs `traced_cms_forward()` to record every operation on the tape,
/// then calls `tape.backward()` to replay in reverse and accumulate gradients.
///
/// Frozen-level gradients are routed into `error_buffers` (not returned in
/// the gradient struct). Active-level gradients go directly into the returned
/// `MAGParams`. This matches the hand-written backward semantics.
pub fn tape_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> (f32, MAGParams) {
    debug_assert_eq!(error_buffers.len(), cfg.k,
        "error_buffers length ({}) must equal cfg.k ({})", error_buffers.len(), cfg.k);
    let d = cfg.swa.d_model;

    let registry = register_opaque_vjps();
    with_tape(registry, |tape| {
        // ── Forward: record everything on tape ──────────────────
        let (loss, cache, loss_id, param_ids) =
            traced_cms_forward(tape, params, cfg, input_ids, target_ids, pulse, context);

        // ── Backward: replay tape in reverse ────────────────────
        tape.backward(loss_id);

        // ── Extract SWA parameter gradients ─────────────────────
        let swa_grads = SWAParams {
            w_embed: tape.get_param_grad(param_ids.w_embed),
            w_q: tape.get_param_grad(param_ids.w_q),
            w_k: tape.get_param_grad(param_ids.w_k),
            w_v: tape.get_param_grad(param_ids.w_v),
            w_o: tape.get_param_grad(param_ids.w_o),
            w_unembed: tape.get_param_grad(param_ids.w_unembed),
        };

        // ── Extract per-level parameter gradients ───────────────
        let mut level_grads = Vec::with_capacity(cfg.k);
        for level in 0..cfg.k {
            let lp_grad_flat = tape.get_param_grad(param_ids.level_params[level]);
            let mut lp_grad = level_params_from_flat(&lp_grad_flat, d);

            // For frozen levels, the w_q_mem was registered as a separate param.
            // Merge its gradient into the level's w_q_mem field.
            if let Some(w_q_mem_id) = param_ids.frozen_w_q_mem[level] {
                let w_q_mem_grad = tape.get_param_grad(w_q_mem_id);
                // The lp_flat already contains w_q_mem at its offset, but the
                // frozen path registered w_q_mem separately. The lp_flat's
                // w_q_mem slice received no gradient (the tape routed through
                // the separate w_q_mem_id). So replace rather than add.
                lp_grad.w_q_mem = w_q_mem_grad;
            }

            if cache.pulse.active_levels[level] {
                // Active level: return gradient directly.
                level_grads.push(lp_grad);
            } else {
                // Frozen level: route gradient into error buffer, return zeros.
                // Use zeros_like_from to match lp_grad's shape (includes w_freq/b_freq
                // when FrequencySchedule::Learned is active).
                error_buffers[level].accumulate(&lp_grad);
                level_grads.push(MemoryLevelParams::zeros_like_from(&lp_grad, d));
            }
        }

        // ── Frequency gate surrogate gradient ──────────────────────
        // The frequency gate controls a discrete decision (active vs frozen),
        // so the tape can't differentiate through it via chain rule. Use the
        // same surrogate mechanism as cms_backward: compute how much each
        // level's output affected the loss, then backprop through sigmoid.
        if let Some(ref fc) = cache.freq_cache {
            let s = cfg.swa.seq_len;
            let combined_y_id = param_ids.combined_y.unwrap();
            let zero_fallback;
            let d_y_combined: &[f32] = match tape.get_grad(combined_y_id) {
                Some(g) => g,
                None => { zero_fallback = vec![0.0f32; s * d]; &zero_fallback },
            };
            let d_gate_values = compute_gate_surrogate(
                &cache.y_per_level, d_y_combined, &cache.pulse.active_levels, cfg.k, s * d,
            );
            // d_embedded_mean is intentionally discarded: the tape already recorded
            // the mean-pool matmul (ones_row @ embedded), so backward through
            // that op accumulates d_embedded via the chain rule automatically.
            let (freq_grads, _d_embedded_mean) = freq_gate_backward(
                &d_gate_values, fc, &params.levels, cfg.k, d,
            );
            for (l, fg) in freq_grads.into_iter().enumerate() {
                if !level_grads[l].w_freq.is_empty() {
                    for j in 0..d {
                        level_grads[l].w_freq[j] += fg.d_w_freq[j];
                    }
                    level_grads[l].b_freq[0] += fg.d_b_freq[0];
                }
            }
        }

        // TODO: Tape path doesn't yet extract alpha_mem/alpha_refl gradients.
        // Hand-written backward (cms_mac_backward) computes these via softmax Jacobian.
        // Tape-based alpha grads are a Stage 3 task (TapeOp::WeightedSum registration).
        (loss, MAGParams { swa: swa_grads, levels: level_grads, alpha_mem: vec![0.0f32; cfg.k], alpha_refl: vec![0.0f32; cfg.k], persistent_tokens: vec![0.0f32; cfg.n_persistent * cfg.swa.d_model] })
    })
}

/// Finite-difference gradient for a single weight element using CMS forward.
/// Uses fresh ContextState each call to ensure deterministic FD evaluation.
#[allow(dead_code)]
fn cms_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let mut ctx_plus = make_context_state(cfg);
    let (loss_plus, _) = cms_forward(&p_plus, cfg, input_ids, target_ids, pulse, &mut ctx_plus);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let mut ctx_minus = make_context_state(cfg);
    let (loss_minus, _) = cms_forward(&p_minus, cfg, input_ids, target_ids, pulse, &mut ctx_minus);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (CMS).
/// Returns (num_checked, num_passed, max_relative_error).
#[allow(dead_code)]
pub(crate) fn cms_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();

    let abs_threshold = 5e-4;

    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = cms_fd_single(
            params, cfg, input_ids, target_ids, pulse,
            &get_weight, &set_weight, idx, eps,
        );

        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());

        checked += 1;

        if denom < abs_threshold {
            passed += 1;
            continue;
        }

        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }

        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!(
                "  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, \
                 rel_err={rel_err:.4e}"
            );
        }
    }

    (checked, passed, max_rel_err)
}

// ── MAL gradient infrastructure ─────────────────────────────────────

/// Compute gradients of loss with respect to all MAL parameters.
#[allow(dead_code)]
pub fn mal_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MAGParams) {
    let (loss, cache) = mal_forward(params, cfg, input_ids, target_ids);
    let grads = mal_backward(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Finite-difference gradient for a single weight element using MAL forward.
#[allow(dead_code)]
fn mal_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = mal_forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = mal_forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (MAL).
#[allow(dead_code)]
pub(crate) fn mal_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();
    let abs_threshold = 5e-4;
    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = mal_fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );
        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());
        checked += 1;
        if denom < abs_threshold { passed += 1; continue; }
        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!("  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, rel_err={rel_err:.4e}");
        }
    }
    (checked, passed, max_rel_err)
}

/// Compute CMS MAL gradients.
#[allow(dead_code)]
pub fn cms_mal_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> (f32, MAGParams) {
    let (loss, cache) = cms_mal_forward(params, cfg, input_ids, target_ids, pulse, context);
    let grads = cms_mal_backward(params, cfg, &cache, input_ids, target_ids, error_buffers);
    (loss, grads)
}

/// Finite-difference gradient for a single weight element using CMS MAL forward.
#[allow(dead_code)]
fn cms_mal_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let mut ctx_plus = make_context_state(cfg);
    let (loss_plus, _) = cms_mal_forward(&p_plus, cfg, input_ids, target_ids, pulse, &mut ctx_plus);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let mut ctx_minus = make_context_state(cfg);
    let (loss_minus, _) = cms_mal_forward(&p_minus, cfg, input_ids, target_ids, pulse, &mut ctx_minus);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (CMS MAL).
#[allow(dead_code)]
pub(crate) fn cms_mal_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();
    let abs_threshold = 5e-4;
    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = cms_mal_fd_single(
            params, cfg, input_ids, target_ids, pulse,
            &get_weight, &set_weight, idx, eps,
        );
        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());
        checked += 1;
        if denom < abs_threshold { passed += 1; continue; }
        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!("  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, rel_err={rel_err:.4e}");
        }
    }
    (checked, passed, max_rel_err)
}

// ── MAC gradient infrastructure ─────────────────────────────────────

/// Compute gradients of loss with respect to all MAC parameters.
#[allow(dead_code)]
pub fn mac_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MAGParams) {
    let (loss, cache) = mac_forward(params, cfg, input_ids, target_ids);
    let grads = mac_backward(params, cfg, &cache, input_ids, target_ids);
    (loss, grads)
}

/// Finite-difference gradient for a single weight element using MAC forward.
#[allow(dead_code)]
fn mac_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let (loss_plus, _) = mac_forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let (loss_minus, _) = mac_forward(&p_minus, cfg, input_ids, target_ids);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (MAC).
#[allow(dead_code)]
pub(crate) fn mac_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();
    let abs_threshold = 5e-4;
    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = mac_fd_single(
            params, cfg, input_ids, target_ids,
            &get_weight, &set_weight, idx, eps,
        );
        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());
        checked += 1;
        if denom < abs_threshold { passed += 1; continue; }
        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!("  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, rel_err={rel_err:.4e}");
        }
    }
    (checked, passed, max_rel_err)
}

/// Compute CMS MAC gradients.
#[allow(dead_code)]
pub fn cms_mac_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> (f32, MAGParams) {
    let (loss, cache) = cms_mac_forward(params, cfg, input_ids, target_ids, pulse, context);
    let grads = cms_mac_backward(params, cfg, &cache, input_ids, target_ids, error_buffers);
    (loss, grads)
}

/// Finite-difference gradient for a single weight element using CMS MAC forward.
#[allow(dead_code)]
fn cms_mac_fd_single(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let mut ctx_plus = make_context_state(cfg);
    let (loss_plus, _) = cms_mac_forward(&p_plus, cfg, input_ids, target_ids, pulse, &mut ctx_plus);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let mut ctx_minus = make_context_state(cfg);
    let (loss_minus, _) = cms_mac_forward(&p_minus, cfg, input_ids, target_ids, pulse, &mut ctx_minus);

    (loss_plus - loss_minus) / (2.0 * eps)
}

/// Check gradient for a specific weight matrix (CMS MAC).
#[allow(dead_code)]
pub(crate) fn cms_mac_check_weight_gradient(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    grads: &MAGParams,
    name: &str,
    pulse: &Pulse,
    get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    num_samples: usize,
    eps: f32,
    tol: f32,
) -> (usize, usize, f32) {
    let grad_vec = get_grad(grads);
    let weight_vec = get_weight(params);
    let n = weight_vec.len();
    let abs_threshold = 5e-4;
    let step = if n > num_samples { n / num_samples } else { 1 };
    let mut checked = 0;
    let mut passed = 0;
    let mut max_rel_err = 0.0f32;

    for idx in (0..n).step_by(step).take(num_samples) {
        let analytical = grad_vec[idx];
        let numerical = cms_mac_fd_single(
            params, cfg, input_ids, target_ids, pulse,
            &get_weight, &set_weight, idx, eps,
        );
        let abs_diff = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs());
        checked += 1;
        if denom < abs_threshold { passed += 1; continue; }
        let rel_err = abs_diff / denom;
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err < tol {
            passed += 1;
        } else {
            eprintln!("  FAIL {name}[{idx}]: analytical={analytical:.6e}, numerical={numerical:.6e}, rel_err={rel_err:.4e}");
        }
    }
    (checked, passed, max_rel_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Force Rust reference for MAL gradient tests when CUDA is available.
    /// Called at the start of specific tests (not globally for all gradient tests).
    /// FD gradient checking requires both analytical and numerical paths to use
    /// identical arithmetic. cuBLAS rounding differs from Rust, which can flip
    /// marginal gradient signs (especially in MAL's residual connection path).
    /// Only MAL tests need this — other rules' gradients are large enough to
    /// tolerate cuBLAS rounding differences.
    #[cfg(feature = "cuda")]
    fn ensure_rust_reference() {
        crate::dispatch::force_rust_reference(true);
    }
    #[cfg(not(feature = "cuda"))]
    fn ensure_rust_reference() {}

    /// Tiny config for gradient checking. Smaller model = larger gradients per
    /// parameter = better FD resolution at f32 precision.
    fn grad_check_config() -> SWAConfig {
        SWAConfig {
            d_model: 8,
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 4,
            vocab_size: 16,
        }
    }

    fn make_test_data(cfg: &SWAConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len)
            .map(|t| t % cfg.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    /// FD eps: large enough for f32 to resolve loss differences.
    const FD_EPS: f32 = 1e-2;
    /// Tolerance: accounts for both FD truncation and f32 rounding.
    #[cfg(not(feature = "cuda"))]
    const FD_TOL: f32 = 0.10;
    #[cfg(feature = "cuda")]
    const FD_TOL: f32 = 0.20;

    // ── SWA gradient checks (unchanged from Zero-A) ──────────────────

    #[test]
    fn test_gradient_w_unembed() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_unembed",
            |p| &p.w_unembed, |p, i, v| p.w_unembed[i] = v, |g| &g.w_unembed,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_unembed: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_unembed: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_o() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_o",
            |p| &p.w_o, |p, i, v| p.w_o[i] = v, |g| &g.w_o,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_o: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_o: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_q() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_q",
            |p| &p.w_q, |p, i, v| p.w_q[i] = v, |g| &g.w_q,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_q: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_q: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_k() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_k",
            |p| &p.w_k, |p, i, v| p.w_k[i] = v, |g| &g.w_k,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_k: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_k: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_v() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_v",
            |p| &p.w_v, |p, i, v| p.w_v[i] = v, |g| &g.w_v,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_v: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_v: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_gradient_w_embed() {
        let cfg = grad_check_config();
        let params = SWAParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, grads) = compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_embed",
            |p| &p.w_embed, |p, i, v| p.w_embed[i] = v, |g| &g.w_embed,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_embed: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_embed: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── MAG gradient checks ──────────────────────────────────────────

    fn mag_grad_check_config() -> MAGConfig {
        MAGConfig::test_config()
    }

    fn mag_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    /// Init MAG params with neutral gate biases for gradient checking.
    /// b_alpha=0 → sigmoid(0)=0.5 (50/50 retention) gives larger memory gradients.
    /// b_theta=0 → softplus(0)=ln(2)≈0.69 (larger learning rate) gives larger signal.
    fn mag_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
            level.b_theta = vec![0.0f32];  // softplus(0)=ln(2)≈0.69
        }
        params
    }

    /// Crown jewel: W_K_mem gradient flows THROUGH memory recurrence.
    #[test]
    fn test_mag_gradient_w_k_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_v_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_q_mem() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_alpha() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_w_theta() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_b_alpha() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mag_gradient_b_theta() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    /// Regression: SWA weights still get correct gradients when embedded in MAG.
    #[test]
    fn test_mag_gradient_swa_w_o() {
        let cfg = mag_grad_check_config();
        let params = mag_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "swa_w_o",
            |p| &p.swa.w_o, |p, i, v| p.swa.w_o[i] = v, |g| &g.swa.w_o,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("swa_w_o: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "swa_w_o: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    /// Outer-loop weight descent: loss decreases as projection weights are updated
    /// via tape-computed gradients. This validates the outer loop — the inner loop
    /// (memory self-modification inside the forward pass) runs without any external
    /// optimizer. See CS-10 through CS-17 for the no-epochs/no-external-optimizer
    /// constraints, which apply to the inner loop.
    #[test]
    fn test_mag_outer_loop_weight_descent() {
        let cfg = mag_grad_check_config();
        let mut params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);

        let (initial_loss, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        eprintln!("MAG initial loss: {initial_loss:.4}");

        let lr = 0.01;
        let outer_steps = 1000;
        for _ in 0..outer_steps {
            let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);
            params.apply_weight_gradients(&grads, lr);
        }

        let (final_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        eprintln!("MAG final loss after {outer_steps} outer-loop steps: {final_loss:.4}");

        assert!(final_loss < initial_loss,
            "Loss should decrease: initial={initial_loss:.4}, final={final_loss:.4}");

        // Verify memory evolved meaningfully
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        // Verify memory evolved — MONETA uses W1/W2 instead of d×d matrix
        let memory_evolved = match &cache.memory_cache {
            crate::mag::MemoryCache::Delta(c) => {
                let m_t = &c.m_states[s * d * d..(s + 1) * d * d];
                m_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Titans(c) => {
                let m_t = &c.m_states[s * d * d..(s + 1) * d * d];
                m_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Hebbian(c) => {
                let m_t = &c.m_states[s * d * d..(s + 1) * d * d];
                m_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Moneta(c) => {
                let w1_size = c.d_hidden * d;
                let w1_t = &c.w1_states[s * w1_size..(s + 1) * w1_size];
                w1_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::YAAD(c) => {
                let w1_size = c.d_hidden * d;
                let w1_t = &c.w1_states[s * w1_size..(s + 1) * w1_size];
                w1_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::MEMORA(c) => {
                let w1_size = c.d_hidden * d;
                let w1_t = &c.w1_states[s * w1_size..(s + 1) * w1_size];
                w1_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Lattice(c) => {
                let m = c.m;
                let s_t = &c.s_states[s * m * d..(s + 1) * m * d];
                s_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Trellis(c) => {
                let sk_size = c.d_k * d;
                let sk_t = &c.sk_states[s * sk_size..(s + 1) * sk_size];
                sk_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
            crate::mag::MemoryCache::Atlas(c) => {
                let m_t = &c.m_states[s * d * d..(s + 1) * d * d];
                m_t.iter().map(|x| x * x).sum::<f32>().sqrt()
            }
        };
        eprintln!("MAG final memory norm: {memory_evolved:.4e}");
        assert!(memory_evolved > 1e-6, "Memory should evolve during training, norm={memory_evolved}");
    }

    // ── CMS gradient checks (k=2, both levels active) ──────────────

    fn cms_grad_check_config() -> MAGConfig {
        MAGConfig::test_config_k2()
    }

    fn cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    fn cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    /// Both-active pulse: step 0 with chunk_sizes [1, 8] → both levels fire.
    fn both_active_pulse(k: usize) -> Pulse {
        Pulse {
            global_step: 0,
            active_levels: vec![true; k],
        }
    }

    // ── Level 0 FD checks ──────────────────────────────────────────

    #[test]
    fn test_cms_gradient_l0_w_k_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_w_v_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_w_q_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_w_alpha() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_b_alpha() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_w_theta() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l0_b_theta() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l0_b_theta: {passed}/{checked} passed");
    }

    // ── Level 1 FD checks ──────────────────────────────────────────

    #[test]
    fn test_cms_gradient_l1_w_k_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_w_v_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_w_q_mem() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_w_alpha() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_b_alpha() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_w_theta() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l1_b_theta() {
        let cfg = cms_grad_check_config();
        let params = cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l1_b_theta: {passed}/{checked} passed");
    }

    // ── k=4 FD gradient checks (Levels 2 and 3) ─────────────────────

    fn cms_grad_check_config_k4() -> MAGConfig {
        MAGConfig::test_config_k4()
    }

    fn cms_params_for_grad_check_k4(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    fn cms_make_test_data_k4(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── Level 2 FD checks ──────────────────────────────────────────

    #[test]
    fn test_cms_gradient_l2_w_k_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_w_k_mem", &pulse,
            |p| &p.levels[2].w_k_mem, |p, i, v| p.levels[2].w_k_mem[i] = v, |g| &g.levels[2].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_w_v_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_w_v_mem", &pulse,
            |p| &p.levels[2].w_v_mem, |p, i, v| p.levels[2].w_v_mem[i] = v, |g| &g.levels[2].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_w_q_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_w_q_mem", &pulse,
            |p| &p.levels[2].w_q_mem, |p, i, v| p.levels[2].w_q_mem[i] = v, |g| &g.levels[2].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_w_alpha() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_w_alpha", &pulse,
            |p| &p.levels[2].w_alpha, |p, i, v| p.levels[2].w_alpha[i] = v, |g| &g.levels[2].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_b_alpha() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_b_alpha", &pulse,
            |p| &p.levels[2].b_alpha, |p, i, v| p.levels[2].b_alpha[i] = v, |g| &g.levels[2].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_w_theta() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_w_theta", &pulse,
            |p| &p.levels[2].w_theta, |p, i, v| p.levels[2].w_theta[i] = v, |g| &g.levels[2].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l2_b_theta() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l2_b_theta", &pulse,
            |p| &p.levels[2].b_theta, |p, i, v| p.levels[2].b_theta[i] = v, |g| &g.levels[2].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l2_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l2_b_theta: {passed}/{checked} passed");
    }

    // ── Level 3 FD checks ──────────────────────────────────────────

    #[test]
    fn test_cms_gradient_l3_w_k_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_w_k_mem", &pulse,
            |p| &p.levels[3].w_k_mem, |p, i, v| p.levels[3].w_k_mem[i] = v, |g| &g.levels[3].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_w_v_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_w_v_mem", &pulse,
            |p| &p.levels[3].w_v_mem, |p, i, v| p.levels[3].w_v_mem[i] = v, |g| &g.levels[3].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_w_q_mem() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_w_q_mem", &pulse,
            |p| &p.levels[3].w_q_mem, |p, i, v| p.levels[3].w_q_mem[i] = v, |g| &g.levels[3].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_w_alpha() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_w_alpha", &pulse,
            |p| &p.levels[3].w_alpha, |p, i, v| p.levels[3].w_alpha[i] = v, |g| &g.levels[3].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_b_alpha() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_b_alpha", &pulse,
            |p| &p.levels[3].b_alpha, |p, i, v| p.levels[3].b_alpha[i] = v, |g| &g.levels[3].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_w_theta() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_w_theta", &pulse,
            |p| &p.levels[3].w_theta, |p, i, v| p.levels[3].w_theta[i] = v, |g| &g.levels[3].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_cms_gradient_l3_b_theta() {
        let cfg = cms_grad_check_config_k4();
        let params = cms_params_for_grad_check_k4(&cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data_k4(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "l3_b_theta", &pulse,
            |p| &p.levels[3].b_theta, |p, i, v| p.levels[3].b_theta[i] = v, |g| &g.levels[3].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("CMS l3_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "l3_b_theta: {passed}/{checked} passed");
    }

    // ── Titans LMM gradient checks (k=1) ────────────────────────────

    fn titans_grad_check_config() -> MAGConfig {
        MAGConfig::titans_test_config()
    }

    fn titans_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
            level.b_theta = vec![0.0f32];  // softplus(0)=ln(2)≈0.69
            level.b_eta = vec![0.0f32];    // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_titans_gradient_w_k_mem() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_w_v_mem() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_w_q_mem() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_w_alpha() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_b_alpha() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_w_theta() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_b_theta() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_w_eta() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_w_eta",
            |p| &p.levels[0].w_eta, |p, i, v| p.levels[0].w_eta[i] = v, |g| &g.levels[0].w_eta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans_w_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_w_eta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_titans_gradient_b_eta() {
        let cfg = titans_grad_check_config();
        let params = titans_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "titans_b_eta",
            |p| &p.levels[0].b_eta, |p, i, v| p.levels[0].b_eta[i] = v, |g| &g.levels[0].b_eta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans_b_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans_b_eta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── Titans CMS gradient checks (k=2, both levels active) ────────

    fn titans_cms_grad_check_config() -> MAGConfig {
        MAGConfig::titans_test_config_k2()
    }

    fn titans_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
            level.b_eta = vec![0.0f32];
        }
        params
    }

    fn titans_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── Titans CMS Level 0 FD checks ────────────────────────────────

    #[test]
    fn test_titans_cms_gradient_l0_w_k_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_w_v_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_w_q_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_w_alpha() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_b_alpha() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_w_theta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_b_theta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_b_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_w_eta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_w_eta", &pulse,
            |p| &p.levels[0].w_eta, |p, i, v| p.levels[0].w_eta[i] = v, |g| &g.levels[0].w_eta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_w_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_w_eta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l0_b_eta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l0_b_eta", &pulse,
            |p| &p.levels[0].b_eta, |p, i, v| p.levels[0].b_eta[i] = v, |g| &g.levels[0].b_eta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l0_b_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l0_b_eta: {passed}/{checked} passed");
    }

    // ── Titans CMS Level 1 FD checks ────────────────────────────────

    #[test]
    fn test_titans_cms_gradient_l1_w_k_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_w_v_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_w_q_mem() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_w_alpha() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_b_alpha() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_w_theta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_b_theta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_b_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_w_eta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_w_eta", &pulse,
            |p| &p.levels[1].w_eta, |p, i, v| p.levels[1].w_eta[i] = v, |g| &g.levels[1].w_eta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_w_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_w_eta: {passed}/{checked} passed");
    }

    #[test]
    fn test_titans_cms_gradient_l1_b_eta() {
        let cfg = titans_cms_grad_check_config();
        let params = titans_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = titans_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "titans_l1_b_eta", &pulse,
            |p| &p.levels[1].b_eta, |p, i, v| p.levels[1].b_eta[i] = v, |g| &g.levels[1].b_eta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("titans CMS l1_b_eta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "titans l1_b_eta: {passed}/{checked} passed");
    }

    // ── Hebbian Rule gradient checks (k=1) ────────────────────────────

    fn hebbian_grad_check_config() -> MAGConfig {
        MAGConfig::hebbian_test_config()
    }

    fn hebbian_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_hebbian_gradient_w_k_mem() {
        let cfg = hebbian_grad_check_config();
        let params = hebbian_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "hebbian_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_hebbian_gradient_w_v_mem() {
        let cfg = hebbian_grad_check_config();
        let params = hebbian_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "hebbian_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_hebbian_gradient_w_q_mem() {
        let cfg = hebbian_grad_check_config();
        let params = hebbian_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "hebbian_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_hebbian_gradient_w_alpha() {
        let cfg = hebbian_grad_check_config();
        let params = hebbian_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "hebbian_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_hebbian_gradient_b_alpha() {
        let cfg = hebbian_grad_check_config();
        let params = hebbian_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "hebbian_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── Hebbian CMS gradient checks (k=2, both levels active) ────────

    fn hebbian_cms_grad_check_config() -> MAGConfig {
        MAGConfig::hebbian_test_config_k2()
    }

    fn hebbian_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
        }
        params
    }

    fn hebbian_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── Hebbian CMS Level 0 FD checks ────────────────────────────────

    #[test]
    fn test_hebbian_cms_gradient_l0_w_k_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l0_w_v_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l0_w_q_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l0_w_alpha() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l0_b_alpha() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l0_b_alpha: {passed}/{checked} passed");
    }

    // ── Hebbian CMS Level 1 FD checks ────────────────────────────────

    #[test]
    fn test_hebbian_cms_gradient_l1_w_k_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l1_w_v_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l1_w_q_mem() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l1_w_alpha() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_hebbian_cms_gradient_l1_b_alpha() {
        let cfg = hebbian_cms_grad_check_config();
        let params = hebbian_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = hebbian_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "hebbian_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("hebbian CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "hebbian l1_b_alpha: {passed}/{checked} passed");
    }

    // ── MONETA gradient checks (k=1, MAG) ────────────────────────────

    fn moneta_grad_check_config() -> MAGConfig {
        MAGConfig::moneta_test_config()
    }

    fn moneta_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_moneta_gradient_w_k_mem() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_w_v_mem() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_w_q_mem() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_w_alpha() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_b_alpha() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_w_theta() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_moneta_gradient_b_theta() {
        let cfg = moneta_grad_check_config();
        let params = moneta_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "moneta_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta_b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── MONETA CMS gradient checks (k=2, both levels active) ─────────

    fn moneta_cms_grad_check_config() -> MAGConfig {
        MAGConfig::moneta_test_config_k2()
    }

    fn moneta_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
        }
        params
    }

    fn moneta_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── MONETA CMS Level 0 FD checks ─────────────────────────────────

    #[test]
    fn test_moneta_cms_gradient_l0_w_k_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_w_v_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_w_q_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_w_alpha() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_b_alpha() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_w_theta() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l0_b_theta() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l0_b_theta: {passed}/{checked} passed");
    }

    // ── MONETA CMS Level 1 FD checks ─────────────────────────────────

    #[test]
    fn test_moneta_cms_gradient_l1_w_k_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_w_v_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_w_q_mem() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_w_alpha() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_b_alpha() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_w_theta() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_moneta_cms_gradient_l1_b_theta() {
        let cfg = moneta_cms_grad_check_config();
        let params = moneta_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = moneta_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "moneta_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("moneta CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "moneta l1_b_theta: {passed}/{checked} passed");
    }

    // ── YAAD gradient checks (k=1, MAG) ──────────────────────────────

    fn yaad_grad_check_config() -> MAGConfig {
        MAGConfig::yaad_test_config()
    }

    fn yaad_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_yaad_gradient_w_k_mem() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_w_v_mem() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_w_q_mem() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_w_alpha() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_b_alpha() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_w_theta() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_yaad_gradient_b_theta() {
        let cfg = yaad_grad_check_config();
        let params = yaad_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "yaad_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad_b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── YAAD CMS gradient checks (k=2, both levels active) ────────────

    fn yaad_cms_grad_check_config() -> MAGConfig {
        MAGConfig::yaad_test_config_k2()
    }

    fn yaad_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
        }
        params
    }

    fn yaad_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── YAAD CMS Level 0 FD checks ──────────────────────────────────

    #[test]
    fn test_yaad_cms_gradient_l0_w_k_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_w_v_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_w_q_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_w_alpha() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_b_alpha() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_w_theta() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l0_b_theta() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l0_b_theta: {passed}/{checked} passed");
    }

    // ── YAAD CMS Level 1 FD checks ──────────────────────────────────

    #[test]
    fn test_yaad_cms_gradient_l1_w_k_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_w_v_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_w_q_mem() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_w_alpha() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_b_alpha() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_w_theta() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_yaad_cms_gradient_l1_b_theta() {
        let cfg = yaad_cms_grad_check_config();
        let params = yaad_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = yaad_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "yaad_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("yaad CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "yaad l1_b_theta: {passed}/{checked} passed");
    }

    // ── MEMORA gradient checks (k=1, MAG) ──────────────────────────────

    fn memora_grad_check_config() -> MAGConfig {
        MAGConfig::memora_test_config()
    }

    fn memora_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_memora_gradient_w_k_mem() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_w_v_mem() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_w_q_mem() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_w_alpha() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_b_alpha() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_w_theta() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_memora_gradient_b_theta() {
        let cfg = memora_grad_check_config();
        let params = memora_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "memora_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora_b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── MEMORA CMS gradient checks (k=2, both levels active) ────────────

    fn memora_cms_grad_check_config() -> MAGConfig {
        MAGConfig::memora_test_config_k2()
    }

    fn memora_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
        }
        params
    }

    fn memora_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── MEMORA CMS Level 0 FD checks ──────────────────────────────────

    #[test]
    fn test_memora_cms_gradient_l0_w_k_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_w_v_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_w_q_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_w_alpha() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_b_alpha() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_w_theta() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l0_b_theta() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l0_b_theta: {passed}/{checked} passed");
    }

    // ── MEMORA CMS Level 1 FD checks ──────────────────────────────────

    #[test]
    fn test_memora_cms_gradient_l1_w_k_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_w_v_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_w_q_mem() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_w_alpha() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_b_alpha() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_w_theta() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_memora_cms_gradient_l1_b_theta() {
        let cfg = memora_cms_grad_check_config();
        let params = memora_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = memora_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "memora_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("memora CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "memora l1_b_theta: {passed}/{checked} passed");
    }

    // ── Lattice OSR gradient checks (k=1, MAG) ──────────────────────────

    fn lattice_grad_check_config() -> MAGConfig {
        MAGConfig::lattice_test_config()
    }

    fn lattice_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
        }
        params
    }

    #[test]
    fn test_lattice_gradient_w_k_mem() {
        let cfg = lattice_grad_check_config();
        let params = lattice_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "lattice_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_lattice_gradient_w_v_mem() {
        let cfg = lattice_grad_check_config();
        let params = lattice_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "lattice_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_lattice_gradient_w_q_mem() {
        let cfg = lattice_grad_check_config();
        let params = lattice_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "lattice_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_lattice_gradient_w_alpha() {
        let cfg = lattice_grad_check_config();
        let params = lattice_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "lattice_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("lattice_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_lattice_gradient_b_alpha() {
        let cfg = lattice_grad_check_config();
        let params = lattice_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "lattice_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("lattice_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── Lattice OSR CMS gradient checks (k=2, both levels active) ────────

    fn lattice_cms_grad_check_config() -> MAGConfig {
        MAGConfig::lattice_test_config_k2()
    }

    fn lattice_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
        }
        params
    }

    fn lattice_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── Lattice CMS Level 0 FD checks ──────────────────────────────────

    #[test]
    fn test_lattice_cms_gradient_l0_w_k_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l0_w_v_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l0_w_q_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l0_w_alpha() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l0_b_alpha() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l0_b_alpha: {passed}/{checked} passed");
    }

    // ── Lattice CMS Level 1 FD checks ──────────────────────────────────

    #[test]
    fn test_lattice_cms_gradient_l1_w_k_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l1_w_v_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l1_w_q_mem() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l1_w_alpha() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_lattice_cms_gradient_l1_b_alpha() {
        let cfg = lattice_cms_grad_check_config();
        let params = lattice_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = lattice_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "lattice_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("lattice CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "lattice l1_b_alpha: {passed}/{checked} passed");
    }

    // ── Trellis gradient checks (k=1, MAG) ──────────────────────────────

    fn trellis_grad_check_config() -> MAGConfig {
        MAGConfig::trellis_test_config()
    }

    fn trellis_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];  // sigmoid(0)=0.5
            level.b_theta = vec![0.0f32];  // softplus(0)≈0.69
        }
        params
    }

    #[test]
    fn test_trellis_gradient_w_k_mem() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_w_v_mem() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_w_q_mem() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_w_alpha() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_w_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_b_alpha() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_b_alpha: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_w_theta() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_w_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_trellis_gradient_b_theta() {
        let cfg = trellis_grad_check_config();
        let params = trellis_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mag_make_test_data(&cfg);
        let (_loss, grads) = mag_compute_gradients(&params, &cfg, &input_ids, &target_ids);

        let (checked, passed, max_err) = mag_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads,
            "trellis_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis_b_theta: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── Trellis CMS gradient checks (k=2, both levels active) ────────────

    fn trellis_cms_grad_check_config() -> MAGConfig {
        MAGConfig::trellis_test_config_k2()
    }

    fn trellis_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    fn trellis_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    // ── Trellis CMS Level 0 FD checks ──────────────────────────────────

    #[test]
    fn test_trellis_cms_gradient_l0_w_k_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_w_v_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_w_q_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_w_alpha() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_b_alpha() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_w_theta() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l0_b_theta() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l0_b_theta: {passed}/{checked} passed");
    }

    // ── Trellis CMS Level 1 FD checks ──────────────────────────────────

    #[test]
    fn test_trellis_cms_gradient_l1_w_k_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_w_v_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_w_q_mem() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_w_alpha() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_b_alpha() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_w_theta() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_trellis_cms_gradient_l1_b_theta() {
        let cfg = trellis_cms_grad_check_config();
        let params = trellis_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = trellis_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "trellis_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("trellis CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "trellis l1_b_theta: {passed}/{checked} passed");
    }

    // ── MAL gradient checks (DeltaRule, k=1) ────────────────────────

    fn mal_grad_check_config() -> MAGConfig {
        MAGConfig::mal_test_config()
    }

    fn mal_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    fn mal_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    #[test]
    fn test_mal_gradient_w_k_mem() {
        ensure_rust_reference();
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_w_v_mem() {
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_w_q_mem() {
        ensure_rust_reference();
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_w_alpha() {
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_b_alpha() {
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_w_theta() {
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_gradient_b_theta() {
        let cfg = mal_grad_check_config();
        let params = mal_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_make_test_data(&cfg);
        let (_loss, grads) = mal_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL b_theta: {passed}/{checked} passed");
    }

    // ── MAL CMS gradient checks (DeltaRule, k=2) ────────────────────

    fn mal_cms_grad_check_config_k2() -> MAGConfig {
        MAGConfig::mal_test_config_k2()
    }

    fn mal_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    fn mal_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_mal_cms_gradient_l0_w_k_mem() {
        ensure_rust_reference();
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_w_v_mem() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_w_q_mem() {
        ensure_rust_reference();
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_w_alpha() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_b_alpha() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_w_theta() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l0_b_theta() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l0_b_theta: {passed}/{checked} passed");
    }

    // ── MAL CMS Level 1 ──

    #[test]
    fn test_mal_cms_gradient_l1_w_k_mem() {
        ensure_rust_reference();
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_w_v_mem() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_w_q_mem() {
        ensure_rust_reference();
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_w_alpha() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_b_alpha() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_w_theta() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mal_cms_gradient_l1_b_theta() {
        let cfg = mal_cms_grad_check_config_k2();
        let params = mal_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mal_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mal_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mal_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mal_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAL CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAL l1_b_theta: {passed}/{checked} passed");
    }

    // ── MAC gradient checks (DeltaRule, k=1) ────────────────────────

    fn mac_grad_check_config() -> MAGConfig {
        MAGConfig::mac_test_config()
    }

    fn mac_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    fn mac_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    #[test]
    fn test_mac_gradient_w_k_mem() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_w_v_mem() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_w_q_mem() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_w_alpha() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_b_alpha() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_w_theta() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_w_theta",
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_gradient_b_theta() {
        let cfg = mac_grad_check_config();
        let params = mac_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_make_test_data(&cfg);
        let (_loss, grads) = mac_compute_gradients(&params, &cfg, &input_ids, &target_ids);
        let (checked, passed, max_err) = mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_b_theta",
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC b_theta: {passed}/{checked} passed");
    }

    // ── MAC CMS gradient checks (DeltaRule, k=2) ────────────────────

    fn mac_cms_grad_check_config_k2() -> MAGConfig {
        MAGConfig::mac_test_config_k2()
    }

    fn mac_cms_params_for_grad_check(cfg: &MAGConfig, seed: u64) -> MAGParams {
        let mut params = MAGParams::init(cfg, seed);
        for level in &mut params.levels {
            level.b_alpha = vec![0.0f32];
            level.b_theta = vec![0.0f32];
        }
        params
    }

    fn mac_cms_make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_mac_cms_gradient_l0_w_k_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_w_k_mem", &pulse,
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_w_v_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_w_v_mem", &pulse,
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_w_q_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_w_q_mem", &pulse,
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_w_alpha() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_w_alpha", &pulse,
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_b_alpha() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_b_alpha", &pulse,
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_w_theta() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_w_theta", &pulse,
            |p| &p.levels[0].w_theta, |p, i, v| p.levels[0].w_theta[i] = v, |g| &g.levels[0].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l0_b_theta() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l0_b_theta", &pulse,
            |p| &p.levels[0].b_theta, |p, i, v| p.levels[0].b_theta[i] = v, |g| &g.levels[0].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l0_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l0_b_theta: {passed}/{checked} passed");
    }

    // ── MAC CMS Level 1 ──

    #[test]
    fn test_mac_cms_gradient_l1_w_k_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_w_k_mem", &pulse,
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_w_k_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_w_v_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_w_v_mem", &pulse,
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_w_v_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_w_q_mem() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_w_q_mem", &pulse,
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_w_q_mem: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_w_alpha() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_w_alpha", &pulse,
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_w_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_w_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_b_alpha() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_b_alpha", &pulse,
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_b_alpha: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_b_alpha: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_w_theta() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_w_theta", &pulse,
            |p| &p.levels[1].w_theta, |p, i, v| p.levels[1].w_theta[i] = v, |g| &g.levels[1].w_theta,
            16, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_w_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_w_theta: {passed}/{checked} passed");
    }

    #[test]
    fn test_mac_cms_gradient_l1_b_theta() {
        let cfg = mac_cms_grad_check_config_k2();
        let params = mac_cms_params_for_grad_check(&cfg, 42);
        let (input_ids, target_ids) = mac_cms_make_test_data(&cfg);
        let pulse = both_active_pulse(cfg.k);
        let mut ctx = make_context_state(&cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let (_loss, grads) = cms_mac_compute_gradients(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs);
        let (checked, passed, max_err) = cms_mac_check_weight_gradient(
            &params, &cfg, &input_ids, &target_ids, &grads, "mac_l1_b_theta", &pulse,
            |p| &p.levels[1].b_theta, |p, i, v| p.levels[1].b_theta[i] = v, |g| &g.levels[1].b_theta,
            1, FD_EPS, FD_TOL,
        );
        eprintln!("MAC CMS l1_b_theta: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked, "MAC l1_b_theta: {passed}/{checked} passed");
    }

    // ── P3.1: tape_compute_gradients smoke test ─────────────────────

    #[test]
    fn test_tape_compute_gradients_delta_k1_loss_match() {
        // Verify tape path produces identical loss and finite nonzero gradients.
        let cfg = MAGConfig::test_config(); // k=1, DeltaRule
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
        let pulse = Pulse { global_step: 0, active_levels: vec![true] };

        // Hand-written backward path
        let mut ctx_ref = make_context_state(&cfg);
        let mut ebufs_ref: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_ref, _grads_ref) = cms_compute_gradients_handwritten(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref, &mut ebufs_ref,
        );

        // Tape backward path
        let mut ctx_tape = make_context_state(&cfg);
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_tape, grads_tape) = tape_compute_gradients(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_tape, &mut ebufs_tape,
        );

        // Loss must be bitwise identical (tape forward is bitwise equiv).
        assert_eq!(loss_ref.to_bits(), loss_tape.to_bits(),
            "loss mismatch: ref={loss_ref} tape={loss_tape}");

        // SWA gradients: finite and nonzero
        let swa_fields: Vec<(&str, &[f32])> = vec![
            ("w_embed", &grads_tape.swa.w_embed),
            ("w_q", &grads_tape.swa.w_q),
            ("w_k", &grads_tape.swa.w_k),
            ("w_v", &grads_tape.swa.w_v),
            ("w_o", &grads_tape.swa.w_o),
            ("w_unembed", &grads_tape.swa.w_unembed),
        ];
        for (name, grad) in &swa_fields {
            assert!(grad.iter().all(|x| x.is_finite()),
                "tape grad swa.{name} has non-finite values");
            let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.0, "tape grad swa.{name} is all zeros");
            eprintln!("tape swa.{name}: norm={norm:.4e}, len={}", grad.len());
        }

        // Level 0 gradients: finite and nonzero for key fields
        let lp = &grads_tape.levels[0];
        let level_fields: Vec<(&str, &[f32])> = vec![
            ("w_k_mem", &lp.w_k_mem),
            ("w_v_mem", &lp.w_v_mem),
            ("w_q_mem", &lp.w_q_mem),
            ("w_alpha", &lp.w_alpha),
            ("b_alpha", &lp.b_alpha),
        ];
        for (name, grad) in &level_fields {
            assert!(grad.iter().all(|x| x.is_finite()),
                "tape grad level[0].{name} has non-finite values");
            let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 0.0, "tape grad level[0].{name} is all zeros");
            eprintln!("tape level[0].{name}: norm={norm:.4e}, len={}", grad.len());
        }
    }

    // ── P3.2: Frozen-level error buffer routing ─────────────────────

    #[test]
    fn test_tape_frozen_level_routes_to_error_buffer() {
        // k=2: level 0 active, level 1 frozen.
        // Tape path should route level 1 grads into error_buffers[1],
        // and return zeros for level 1 in the gradient struct.
        //
        // Critical: must warm up memory with an all-active forward pass first,
        // otherwise M=0 → d_q = M^T @ d_y = 0 (mathematically correct but trivial).
        let cfg = MAGConfig::test_config_k2(); // k=2, DeltaRule
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();

        // Step 0: warm up with both levels active to populate memory.
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let mut ctx_ref = make_context_state(&cfg);
        let mut ctx_tape = make_context_state(&cfg);
        cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut ctx_ref);
        cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut ctx_tape);

        // Step 1: level 1 frozen (memory is now non-zero).
        let pulse1 = Pulse { global_step: 1, active_levels: vec![true, false] };

        // Hand-written backward path (reference)
        let mut ebufs_ref: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_ref, grads_ref) = cms_compute_gradients_handwritten(
            &params, &cfg, &input_ids, &target_ids, &pulse1, &mut ctx_ref, &mut ebufs_ref,
        );

        // Tape backward path
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_tape, grads_tape) = tape_compute_gradients(
            &params, &cfg, &input_ids, &target_ids, &pulse1, &mut ctx_tape, &mut ebufs_tape,
        );

        // Loss must be bitwise identical.
        assert_eq!(loss_ref.to_bits(), loss_tape.to_bits(),
            "loss mismatch: ref={loss_ref} tape={loss_tape}");

        // Level 0 (active): grads should be nonzero.
        let l0_norm: f32 = grads_tape.levels[0].w_k_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert!(l0_norm > 0.0, "Active level 0 grads should be nonzero");

        // Level 1 (frozen): returned grads should be all zeros.
        let l1_fields: Vec<(&str, &[f32])> = vec![
            ("w_k_mem", &grads_tape.levels[1].w_k_mem),
            ("w_v_mem", &grads_tape.levels[1].w_v_mem),
            ("w_q_mem", &grads_tape.levels[1].w_q_mem),
            ("w_alpha", &grads_tape.levels[1].w_alpha),
            ("b_alpha", &grads_tape.levels[1].b_alpha),
        ];
        for (name, grad) in &l1_fields {
            let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_eq!(norm, 0.0,
                "Frozen level 1 grad {name} should be zero, got norm={norm:.4e}");
        }

        // Error buffer[1] should have accumulated frozen grads (nonzero).
        // The read-only backward flows d_y through M^T to get d_q_mem,
        // then through the matmul to get d_W_Q_mem.
        assert_eq!(ebufs_tape[1].steps_accumulated, 1,
            "error_buffers[1] should have 1 step accumulated");
        let ebuf_wqm_norm: f32 = ebufs_tape[1].grads.w_q_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert!(ebuf_wqm_norm > 0.0,
            "error_buffers[1].grads.w_q_mem should be nonzero");

        // Error buffer[0] should be untouched (active level).
        assert_eq!(ebufs_tape[0].steps_accumulated, 0,
            "error_buffers[0] should be untouched for active level");

        // Reference error buffer should match tape error buffer.
        assert_eq!(ebufs_ref[1].steps_accumulated, ebufs_tape[1].steps_accumulated,
            "steps_accumulated mismatch");
        let ref_wqm_norm: f32 = ebufs_ref[1].grads.w_q_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert!(ref_wqm_norm > 0.0,
            "Reference error_buffers[1].grads.w_q_mem should also be nonzero");

        // Reference grads for level 1 should also be zero (cms_backward routes to ebuf).
        let ref_l1_norm: f32 = grads_ref.levels[1].w_q_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(ref_l1_norm, 0.0,
            "Reference level 1 grads should also be zero (routed to ebuf)");
    }

    // ── P3.3: Learned frequency gate gradients ──────────────────────

    #[test]
    fn test_tape_learned_freq_gate_gradients() {
        use crate::dynamic_freq::{FrequencySchedule, LearnedFreqConfig};

        // k=2 DeltaRule with learned frequency gates.
        let mut cfg = MAGConfig::test_config_k2();
        cfg.frequency_schedule = FrequencySchedule::Learned(LearnedFreqConfig::default());

        let params = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        // Hand-written backward path (reference)
        let mut ctx_ref = make_context_state(&cfg);
        let mut ebufs_ref: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_ref, grads_ref) = cms_compute_gradients_handwritten(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref, &mut ebufs_ref,
        );

        // Tape path
        let mut ctx_tape = make_context_state(&cfg);
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(d))
            .collect();
        let (loss_tape, grads_tape) = tape_compute_gradients(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_tape, &mut ebufs_tape,
        );

        // Loss must match.
        assert_eq!(loss_ref.to_bits(), loss_tape.to_bits(),
            "Learned freq gate: loss mismatch ref={loss_ref} tape={loss_tape}");

        // w_freq/b_freq gradients should be present and nonzero for at least one level.
        let mut any_freq_nonzero = false;
        for l in 0..cfg.k {
            assert_eq!(grads_tape.levels[l].w_freq.len(), d,
                "level {l}: w_freq grad should have length d={d}");
            assert_eq!(grads_tape.levels[l].b_freq.len(), 1,
                "level {l}: b_freq grad should have length 1");
            let wf_norm: f32 = grads_tape.levels[l].w_freq.iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            let bf_norm: f32 = grads_tape.levels[l].b_freq.iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            if wf_norm > 0.0 || bf_norm > 0.0 {
                any_freq_nonzero = true;
            }
        }
        assert!(any_freq_nonzero,
            "At least one level should have nonzero w_freq/b_freq gradient");

        // Reference path should also have freq gradients (from cms_backward).
        for l in 0..cfg.k {
            let ref_wf_norm: f32 = grads_ref.levels[l].w_freq.iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            let tape_wf_norm: f32 = grads_tape.levels[l].w_freq.iter()
                .map(|x| x * x).sum::<f32>().sqrt();
            // Both should be nonzero or both zero.
            if ref_wf_norm > 1e-10 || tape_wf_norm > 1e-10 {
                assert!(ref_wf_norm > 1e-10 && tape_wf_norm > 1e-10,
                    "level {l}: ref w_freq norm={ref_wf_norm:.4e} tape={tape_wf_norm:.4e} — one is zero");
            }
        }
    }

    // ── P3.4: Class 3 tests — tape vs hand-written backward ─────────

    /// Compare two gradient slices element-wise. Returns (max_rel_err, num_mismatches).
    /// Uses relative tolerance for large values, absolute tolerance for small values.
    fn compare_grad_slices(
        name: &str,
        ref_grad: &[f32],
        tape_grad: &[f32],
        rtol: f32,
        atol: f32,
    ) -> (f32, usize) {
        assert_eq!(ref_grad.len(), tape_grad.len(),
            "{name}: length mismatch ref={} tape={}", ref_grad.len(), tape_grad.len());
        let mut max_rel_err = 0.0f32;
        let mut mismatches = 0;
        for (i, (&r, &t)) in ref_grad.iter().zip(tape_grad.iter()).enumerate() {
            assert!(r.is_finite(), "{name}[{i}]: ref grad is not finite ({r})");
            assert!(t.is_finite(), "{name}[{i}]: tape grad is not finite ({t})");
            let abs_diff = (r - t).abs();
            if abs_diff <= atol {
                continue; // absolute difference within tolerance
            }
            let denom = r.abs().max(t.abs());
            let rel_err = abs_diff / denom;
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
            if rel_err > rtol {
                if mismatches < 5 {
                    eprintln!("  {name}[{i}]: ref={r:.6e} tape={t:.6e} rel_err={rel_err:.4e}");
                }
                mismatches += 1;
            }
        }
        (max_rel_err, mismatches)
    }

    /// Run tape_compute_gradients and cms_compute_gradients on the same config,
    /// assert loss is bitwise equal and all parameter gradients match within tolerance.
    fn class3_tape_vs_handwritten(cfg: &MAGConfig, label: &str) {
        ensure_rust_reference();
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(cfg, 42);
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
        let pulse = Pulse { global_step: 0, active_levels: vec![true; cfg.k] };

        // Hand-written backward
        let mut ctx_ref = make_context_state(cfg);
        let mut ebufs_ref: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let (loss_ref, grads_ref) = cms_compute_gradients_handwritten(
            &params, cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref, &mut ebufs_ref,
        );

        // Tape backward
        let mut ctx_tape = make_context_state(cfg);
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let (loss_tape, grads_tape) = tape_compute_gradients(
            &params, cfg, &input_ids, &target_ids, &pulse, &mut ctx_tape, &mut ebufs_tape,
        );

        // Loss: bitwise identical.
        assert_eq!(loss_ref.to_bits(), loss_tape.to_bits(),
            "{label}: loss mismatch ref={loss_ref} tape={loss_tape}");

        let rtol = 1e-5;
        let atol = 1e-6;
        let mut total_mismatches = 0;

        // SWA gradients.
        let swa_fields: Vec<(&str, &[f32], &[f32])> = vec![
            ("w_embed", &grads_ref.swa.w_embed, &grads_tape.swa.w_embed),
            ("w_q", &grads_ref.swa.w_q, &grads_tape.swa.w_q),
            ("w_k", &grads_ref.swa.w_k, &grads_tape.swa.w_k),
            ("w_v", &grads_ref.swa.w_v, &grads_tape.swa.w_v),
            ("w_o", &grads_ref.swa.w_o, &grads_tape.swa.w_o),
            ("w_unembed", &grads_ref.swa.w_unembed, &grads_tape.swa.w_unembed),
        ];
        for (name, ref_g, tape_g) in &swa_fields {
            let full_name = format!("{label}/swa.{name}");
            let (max_err, mm) = compare_grad_slices(&full_name, ref_g, tape_g, rtol, atol);
            if mm > 0 {
                eprintln!("{full_name}: {mm} mismatches, max_rel_err={max_err:.4e}");
            }
            total_mismatches += mm;
        }

        // Per-level gradients.
        for level in 0..cfg.k {
            let rl = &grads_ref.levels[level];
            let tl = &grads_tape.levels[level];
            let level_fields: Vec<(&str, &[f32], &[f32])> = vec![
                ("w_k_mem", &rl.w_k_mem, &tl.w_k_mem),
                ("w_v_mem", &rl.w_v_mem, &tl.w_v_mem),
                ("w_q_mem", &rl.w_q_mem, &tl.w_q_mem),
                ("w_alpha", &rl.w_alpha, &tl.w_alpha),
                ("b_alpha", &rl.b_alpha, &tl.b_alpha),
                ("w_theta", &rl.w_theta, &tl.w_theta),
                ("b_theta", &rl.b_theta, &tl.b_theta),
                ("w_eta", &rl.w_eta, &tl.w_eta),
                ("b_eta", &rl.b_eta, &tl.b_eta),
                ("w_omega", &rl.w_omega, &tl.w_omega),
            ];
            for (name, ref_g, tape_g) in &level_fields {
                if ref_g.is_empty() && tape_g.is_empty() {
                    continue; // field not used by this rule
                }
                let full_name = format!("{label}/level[{level}].{name}");
                let (max_err, mm) = compare_grad_slices(&full_name, ref_g, tape_g, rtol, atol);
                if mm > 0 {
                    eprintln!("{full_name}: {mm} mismatches, max_rel_err={max_err:.4e}");
                }
                total_mismatches += mm;
            }
        }

        assert_eq!(total_mismatches, 0,
            "{label}: {total_mismatches} gradient element mismatches (rtol={rtol}, atol={atol})");
        eprintln!("{label}: all gradients match (loss={loss_ref:.4e})");
    }

    // ── DeltaRule ────────────────────────────────────────────────────

    #[test]
    fn test_class3_delta_k1() {
        class3_tape_vs_handwritten(&MAGConfig::test_config(), "delta_k1");
    }

    #[test]
    fn test_class3_delta_k2() {
        class3_tape_vs_handwritten(&MAGConfig::test_config_k2(), "delta_k2");
    }

    // ── Titans LMM ──────────────────────────────────────────────────

    #[test]
    fn test_class3_titans_k1() {
        class3_tape_vs_handwritten(&MAGConfig::titans_test_config(), "titans_k1");
    }

    #[test]
    fn test_class3_titans_k2() {
        class3_tape_vs_handwritten(&MAGConfig::titans_test_config_k2(), "titans_k2");
    }

    // ── Hebbian ─────────────────────────────────────────────────────

    #[test]
    fn test_class3_hebbian_k1() {
        class3_tape_vs_handwritten(&MAGConfig::hebbian_test_config(), "hebbian_k1");
    }

    #[test]
    fn test_class3_hebbian_k2() {
        class3_tape_vs_handwritten(&MAGConfig::hebbian_test_config_k2(), "hebbian_k2");
    }

    // ── MONETA ──────────────────────────────────────────────────────

    #[test]
    fn test_class3_moneta_k1() {
        class3_tape_vs_handwritten(&MAGConfig::moneta_test_config(), "moneta_k1");
    }

    #[test]
    fn test_class3_moneta_k2() {
        class3_tape_vs_handwritten(&MAGConfig::moneta_test_config_k2(), "moneta_k2");
    }

    // ── YAAD ────────────────────────────────────────────────────────

    #[test]
    fn test_class3_yaad_k1() {
        class3_tape_vs_handwritten(&MAGConfig::yaad_test_config(), "yaad_k1");
    }

    #[test]
    fn test_class3_yaad_k2() {
        class3_tape_vs_handwritten(&MAGConfig::yaad_test_config_k2(), "yaad_k2");
    }

    // ── MEMORA ──────────────────────────────────────────────────────

    #[test]
    fn test_class3_memora_k1() {
        class3_tape_vs_handwritten(&MAGConfig::memora_test_config(), "memora_k1");
    }

    #[test]
    fn test_class3_memora_k2() {
        class3_tape_vs_handwritten(&MAGConfig::memora_test_config_k2(), "memora_k2");
    }

    // ── Lattice OSR ─────────────────────────────────────────────────

    #[test]
    fn test_class3_lattice_k1() {
        class3_tape_vs_handwritten(&MAGConfig::lattice_test_config(), "lattice_k1");
    }

    #[test]
    fn test_class3_lattice_k2() {
        class3_tape_vs_handwritten(&MAGConfig::lattice_test_config_k2(), "lattice_k2");
    }

    // ── Trellis ─────────────────────────────────────────────────────

    #[test]
    fn test_class3_trellis_k1() {
        class3_tape_vs_handwritten(&MAGConfig::trellis_test_config(), "trellis_k1");
    }

    #[test]
    fn test_class3_trellis_k2() {
        class3_tape_vs_handwritten(&MAGConfig::trellis_test_config_k2(), "trellis_k2");
    }

    // ── Atlas Omega ─────────────────────────────────────────────────

    #[test]
    fn test_class3_atlas_k1() {
        class3_tape_vs_handwritten(&MAGConfig::atlas_test_config(), "atlas_k1");
    }

    #[test]
    fn test_class3_atlas_k2() {
        class3_tape_vs_handwritten(&MAGConfig::atlas_test_config_k2(), "atlas_k2");
    }

    // ── k=4 DeltaRule ───────────────────────────────────────────────

    #[test]
    fn test_class3_delta_k4() {
        class3_tape_vs_handwritten(&MAGConfig::test_config_k4(), "delta_k4");
    }

    // ── Mixed active/frozen (k=2, level 1 frozen) ───────────────────

    #[test]
    fn test_class3_delta_k2_frozen() {
        ensure_rust_reference();
        let cfg = MAGConfig::test_config_k2();
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();

        // Warm up: all-active forward to populate memory.
        let warm_pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let mut ctx_ref = make_context_state(&cfg);
        let mut ebufs_ref: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let _ = cms_compute_gradients_handwritten(
            &params, &cfg, &input_ids, &target_ids, &warm_pulse, &mut ctx_ref, &mut ebufs_ref,
        );
        let mut ctx_tape = ctx_ref.clone();
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();

        // Reset error buffers for the actual test.
        ebufs_ref = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();

        // Frozen pulse: level 0 active, level 1 frozen.
        let frozen_pulse = Pulse { global_step: 1, active_levels: vec![true, false] };

        let (loss_ref, grads_ref) = cms_compute_gradients_handwritten(
            &params, &cfg, &input_ids, &target_ids, &frozen_pulse, &mut ctx_ref, &mut ebufs_ref,
        );
        let (loss_tape, grads_tape) = tape_compute_gradients(
            &params, &cfg, &input_ids, &target_ids, &frozen_pulse, &mut ctx_tape, &mut ebufs_tape,
        );

        assert_eq!(loss_ref.to_bits(), loss_tape.to_bits(),
            "frozen: loss mismatch ref={loss_ref} tape={loss_tape}");

        // Level 0 (active): gradients should match.
        let rtol = 1e-5;
        let atol = 1e-6;
        let (_, mm) = compare_grad_slices(
            "frozen/level[0].w_k_mem",
            &grads_ref.levels[0].w_k_mem, &grads_tape.levels[0].w_k_mem,
            rtol, atol,
        );
        assert_eq!(mm, 0, "frozen/level[0] gradient mismatch");

        // Level 1 (frozen): returned grads should be zero in both paths.
        let ref_l1_norm: f32 = grads_ref.levels[1].w_k_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        let tape_l1_norm: f32 = grads_tape.levels[1].w_k_mem.iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        assert!(ref_l1_norm < 1e-10,
            "frozen/level[1]: ref grads should be zero (routed to ebuf), norm={ref_l1_norm}");
        assert!(tape_l1_norm < 1e-10,
            "frozen/level[1]: tape grads should be zero (routed to ebuf), norm={tape_l1_norm}");

        // Error buffers should match: both routed level 1 grads there.
        let ref_ebuf_norm = ebufs_ref[1].grads.norm();
        let tape_ebuf_norm = ebufs_tape[1].grads.norm();
        assert!(ref_ebuf_norm > 1e-10,
            "frozen: ref error_buffer[1] should be nonzero, norm={ref_ebuf_norm}");
        assert!((ref_ebuf_norm - tape_ebuf_norm).abs() / ref_ebuf_norm < 1e-4,
            "frozen: error_buffer[1] norm mismatch ref={ref_ebuf_norm:.6e} tape={tape_ebuf_norm:.6e}");

        eprintln!("delta_k2_frozen: all checks pass (loss={loss_ref:.4e}, ebuf_norm={ref_ebuf_norm:.4e})");
    }

    // ── P3.5: Class 2 FD tests — tape gradients vs finite differences ──

    /// Compute tape gradients and check a specific weight against FD.
    /// Reuses cms_check_weight_gradient (FD via cms_forward) but compares
    /// against gradients from tape_compute_gradients.
    fn tape_fd_check(
        cfg: &MAGConfig,
        name: &str,
        get_weight: impl Fn(&MAGParams) -> &Vec<f32>,
        set_weight: impl Fn(&mut MAGParams, usize, f32),
        get_grad: impl Fn(&MAGParams) -> &Vec<f32>,
    ) {
        ensure_rust_reference();
        let params = cms_params_for_grad_check(cfg, 42);
        let (input_ids, target_ids) = cms_make_test_data(cfg);
        let pulse = both_active_pulse(cfg.k);
        let d = cfg.swa.d_model;

        let mut ctx = make_context_state(cfg);
        let mut ebufs: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let (_loss, grads) = tape_compute_gradients(
            &params, cfg, &input_ids, &target_ids, &pulse, &mut ctx, &mut ebufs,
        );

        let (checked, passed, max_err) = cms_check_weight_gradient(
            &params, cfg, &input_ids, &target_ids, &grads, name, &pulse,
            get_weight, set_weight, get_grad,
            20, FD_EPS, FD_TOL,
        );
        eprintln!("tape FD {name}: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(passed == checked,
            "tape FD {name}: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    // ── DeltaRule k=1: SWA params ───────────────────────────────────

    #[test]
    fn test_tape_fd_delta_k1_w_embed() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_embed",
            |p| &p.swa.w_embed, |p, i, v| p.swa.w_embed[i] = v, |g| &g.swa.w_embed);
    }

    #[test]
    fn test_tape_fd_delta_k1_w_q() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_q",
            |p| &p.swa.w_q, |p, i, v| p.swa.w_q[i] = v, |g| &g.swa.w_q);
    }

    #[test]
    fn test_tape_fd_delta_k1_w_k() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_k",
            |p| &p.swa.w_k, |p, i, v| p.swa.w_k[i] = v, |g| &g.swa.w_k);
    }

    #[test]
    fn test_tape_fd_delta_k1_w_v() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_v",
            |p| &p.swa.w_v, |p, i, v| p.swa.w_v[i] = v, |g| &g.swa.w_v);
    }

    #[test]
    fn test_tape_fd_delta_k1_w_o() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_o",
            |p| &p.swa.w_o, |p, i, v| p.swa.w_o[i] = v, |g| &g.swa.w_o);
    }

    #[test]
    fn test_tape_fd_delta_k1_w_unembed() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/w_unembed",
            |p| &p.swa.w_unembed, |p, i, v| p.swa.w_unembed[i] = v, |g| &g.swa.w_unembed);
    }

    // ── DeltaRule k=1: level 0 params ───────────────────────────────

    #[test]
    fn test_tape_fd_delta_k1_l0_w_k_mem() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/l0_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem);
    }

    #[test]
    fn test_tape_fd_delta_k1_l0_w_v_mem() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/l0_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem);
    }

    #[test]
    fn test_tape_fd_delta_k1_l0_w_q_mem() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/l0_w_q_mem",
            |p| &p.levels[0].w_q_mem, |p, i, v| p.levels[0].w_q_mem[i] = v, |g| &g.levels[0].w_q_mem);
    }

    #[test]
    fn test_tape_fd_delta_k1_l0_w_alpha() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/l0_w_alpha",
            |p| &p.levels[0].w_alpha, |p, i, v| p.levels[0].w_alpha[i] = v, |g| &g.levels[0].w_alpha);
    }

    #[test]
    fn test_tape_fd_delta_k1_l0_b_alpha() {
        let cfg = MAGConfig::test_config();
        tape_fd_check(&cfg, "delta_k1/l0_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha);
    }

    // ── DeltaRule k=2: SWA params ───────────────────────────────────

    #[test]
    fn test_tape_fd_delta_k2_w_embed() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/w_embed",
            |p| &p.swa.w_embed, |p, i, v| p.swa.w_embed[i] = v, |g| &g.swa.w_embed);
    }

    #[test]
    fn test_tape_fd_delta_k2_w_unembed() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/w_unembed",
            |p| &p.swa.w_unembed, |p, i, v| p.swa.w_unembed[i] = v, |g| &g.swa.w_unembed);
    }

    // ── DeltaRule k=2: level 0 + level 1 params ────────────────────

    #[test]
    fn test_tape_fd_delta_k2_l0_w_k_mem() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l0_w_k_mem",
            |p| &p.levels[0].w_k_mem, |p, i, v| p.levels[0].w_k_mem[i] = v, |g| &g.levels[0].w_k_mem);
    }

    #[test]
    fn test_tape_fd_delta_k2_l0_w_v_mem() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l0_w_v_mem",
            |p| &p.levels[0].w_v_mem, |p, i, v| p.levels[0].w_v_mem[i] = v, |g| &g.levels[0].w_v_mem);
    }

    #[test]
    fn test_tape_fd_delta_k2_l0_b_alpha() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l0_b_alpha",
            |p| &p.levels[0].b_alpha, |p, i, v| p.levels[0].b_alpha[i] = v, |g| &g.levels[0].b_alpha);
    }

    #[test]
    fn test_tape_fd_delta_k2_l1_w_k_mem() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l1_w_k_mem",
            |p| &p.levels[1].w_k_mem, |p, i, v| p.levels[1].w_k_mem[i] = v, |g| &g.levels[1].w_k_mem);
    }

    #[test]
    fn test_tape_fd_delta_k2_l1_w_v_mem() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l1_w_v_mem",
            |p| &p.levels[1].w_v_mem, |p, i, v| p.levels[1].w_v_mem[i] = v, |g| &g.levels[1].w_v_mem);
    }

    #[test]
    fn test_tape_fd_delta_k2_l1_w_q_mem() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l1_w_q_mem",
            |p| &p.levels[1].w_q_mem, |p, i, v| p.levels[1].w_q_mem[i] = v, |g| &g.levels[1].w_q_mem);
    }

    #[test]
    fn test_tape_fd_delta_k2_l1_w_alpha() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l1_w_alpha",
            |p| &p.levels[1].w_alpha, |p, i, v| p.levels[1].w_alpha[i] = v, |g| &g.levels[1].w_alpha);
    }

    #[test]
    fn test_tape_fd_delta_k2_l1_b_alpha() {
        let cfg = MAGConfig::test_config_k2();
        tape_fd_check(&cfg, "delta_k2/l1_b_alpha",
            |p| &p.levels[1].b_alpha, |p, i, v| p.levels[1].b_alpha[i] = v, |g| &g.levels[1].b_alpha);
    }

    // ── P4.2: Build loop regression — 10-step trajectory ──────────

    /// Run a 10-step build loop comparing tape path vs hand-written path.
    ///
    /// Both paths start from identical params and context. A Conductor
    /// generates pulses so CMS frequency scheduling varies across steps
    /// (Level 1 fires at step 0, 8, etc.). Loss must match within f32
    /// tolerance at every step. Gradient norms must also match, catching
    /// accumulation errors that single-step tests miss.
    #[test]
    fn test_build_loop_regression_delta_k2() {
        use crate::conductor::Conductor;
        ensure_rust_reference();

        let cfg = MAGConfig::test_config_k2();
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let lr = 0.01f32;
        let num_steps = 10;

        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();

        // Identical starting state for both paths
        let params_init = MAGParams::init(&cfg, 42);

        // ── Tape path ──
        let mut params_tape = params_init.clone();
        let mut ctx_tape = make_context_state(&cfg);
        let mut ebufs_tape: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let mut conductor_tape = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut losses_tape = Vec::with_capacity(num_steps);
        let mut grad_norms_tape = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            let pulse = conductor_tape.pulse();
            let (loss, grads) = tape_compute_gradients(
                &params_tape, &cfg, &input_ids, &target_ids,
                &pulse, &mut ctx_tape, &mut ebufs_tape,
            );
            let gnorm = grad_l2_norm(&grads);
            losses_tape.push(loss);
            grad_norms_tape.push(gnorm);
            params_tape.apply_weight_gradients(&grads, lr);
            conductor_tape.advance();
            eprintln!("tape   step {step:2}: loss={loss:.6e}  gnorm={gnorm:.6e}");
        }

        // ── Hand-written path ──
        let mut params_hw = params_init.clone();
        let mut ctx_hw = make_context_state(&cfg);
        let mut ebufs_hw: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(d)).collect();
        let mut conductor_hw = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut losses_hw = Vec::with_capacity(num_steps);
        let mut grad_norms_hw = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            let pulse = conductor_hw.pulse();
            let (loss, grads) = cms_compute_gradients_handwritten(
                &params_hw, &cfg, &input_ids, &target_ids,
                &pulse, &mut ctx_hw, &mut ebufs_hw,
            );
            let gnorm = grad_l2_norm(&grads);
            losses_hw.push(loss);
            grad_norms_hw.push(gnorm);
            params_hw.apply_weight_gradients(&grads, lr);
            conductor_hw.advance();
            eprintln!("hw     step {step:2}: loss={loss:.6e}  gnorm={gnorm:.6e}");
        }

        // ── Compare trajectories ──
        let rtol = 1e-5f32;
        for step in 0..num_steps {
            let lt = losses_tape[step];
            let lh = losses_hw[step];
            // Loss: bitwise identical at step 0 (same code path for forward).
            // After weight updates accumulate, f32 rounding may differ slightly.
            if step == 0 {
                assert_eq!(lt.to_bits(), lh.to_bits(),
                    "step 0: loss must be bitwise identical, tape={lt} hw={lh}");
            } else {
                let abs_diff = (lt - lh).abs();
                let denom = lt.abs().max(lh.abs()).max(1e-8);
                let rel_err = abs_diff / denom;
                assert!(rel_err <= rtol,
                    "step {step}: loss diverged rel_err={rel_err:.4e} (tape={lt:.6e} hw={lh:.6e})");
            }

            // Gradient norms: should track closely
            let gt = grad_norms_tape[step];
            let gh = grad_norms_hw[step];
            let gabs = (gt - gh).abs();
            let gdenom = gt.abs().max(gh.abs()).max(1e-8);
            let grel = gabs / gdenom;
            assert!(grel <= rtol,
                "step {step}: grad norm diverged rel_err={grel:.4e} (tape={gt:.6e} hw={gh:.6e})");
        }

        // Loss should decrease over the trajectory (outer loop is working)
        assert!(losses_tape[num_steps - 1] < losses_tape[0],
            "tape: loss should decrease, initial={:.4e} final={:.4e}",
            losses_tape[0], losses_tape[num_steps - 1]);

        eprintln!("build_loop_regression: {num_steps} steps, all losses and grad norms match");
    }

    /// L2 norm of all gradient parameters (flat).
    fn grad_l2_norm(grads: &MAGParams) -> f32 {
        let mut sum = 0.0f32;
        // SWA
        for &g in grads.swa.w_embed.iter()
            .chain(grads.swa.w_q.iter())
            .chain(grads.swa.w_k.iter())
            .chain(grads.swa.w_v.iter())
            .chain(grads.swa.w_o.iter())
            .chain(grads.swa.w_unembed.iter())
        {
            sum += g * g;
        }
        // Levels
        for level in &grads.levels {
            for &g in level.w_k_mem.iter()
                .chain(level.w_v_mem.iter())
                .chain(level.w_q_mem.iter())
                .chain(level.w_alpha.iter())
                .chain(level.b_alpha.iter())
                .chain(level.w_theta.iter())
                .chain(level.b_theta.iter())
                .chain(level.w_eta.iter())
                .chain(level.b_eta.iter())
                .chain(level.w_omega.iter())
                .chain(level.w_freq.iter())
                .chain(level.b_freq.iter())
            {
                sum += g * g;
            }
        }
        sum.sqrt()
    }
}
