/// Gradient orchestration and verification.
///
/// Provides:
/// - `compute_gradients`: main API for computing SWA parameter gradients
/// - `mag_compute_gradients`: main API for computing MAG parameter gradients
/// - `finite_diff_gradient`: central finite differences for verification
/// - Gradient checking utilities

use crate::bf16::Bf16Storage;
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
    get_weight: impl Fn(&SWAParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    // Read back effective value after bf16 truncation (if applicable).
    let eff_plus = get_weight(&p_plus)[idx];
    let (loss_plus, _) = mag_forward(&p_plus, cfg, input_ids, target_ids);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let eff_minus = get_weight(&p_minus)[idx];
    let (loss_minus, _) = mag_forward(&p_minus, cfg, input_ids, target_ids);

    // Use effective perturbation as denominator — accounts for bf16 quantization.
    let effective_delta = eff_plus - eff_minus;
    if effective_delta.abs() < 1e-30 {
        0.0 // perturbation collapsed to zero (shouldn't happen with eps=1e-2)
    } else {
        (loss_plus - loss_minus) / effective_delta
    }
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
    get_weight: impl Fn(&SWAParams) -> &[f32],
    set_weight: impl Fn(&mut SWAParams, usize, f32),
    get_grad: impl Fn(&SWAParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
            // SwiGluMlp: lp_flat layout is standard_fields ++ gate_proj ++ up_proj ++ down_proj.
            // The standard auto-detection in level_params_from_flat misidentifies the
            // MLP projection data as conv1d weights (49152 = 2*d*(ks+1) for ks=383).
            // Slice off the standard prefix and extract MLP grads explicitly.
            let mut lp_grad = if cfg.memory_rule == crate::model::MemoryRuleKind::SwiGluMlp {
                let inter = cfg.intermediate_size;
                let std_size = 5 * d * d + 6 * d + 3; // standard fields, no freq/conv
                let required = std_size + 3 * inter * d;
                assert!(
                    lp_grad_flat.len() >= required,
                    "tape_compute_gradients: SwiGluMlp lp_grad_flat too short: \
                     got {} expected >= {} (d={}, inter={})",
                    lp_grad_flat.len(), required, d, inter
                );
                let mut lp = level_params_from_flat(&lp_grad_flat[..std_size], d, 0);
                lp.gate_proj = lp_grad_flat[std_size..std_size + inter * d].to_vec();
                lp.up_proj   = lp_grad_flat[std_size + inter * d..std_size + 2 * inter * d].to_vec();
                lp.down_proj = lp_grad_flat[std_size + 2 * inter * d..std_size + 3 * inter * d].to_vec();
                lp
            } else {
                level_params_from_flat(&lp_grad_flat, d, cfg.kernel_size)
            };

            // For frozen levels, the w_q_mem was registered as a separate param.
            // Merge its gradient into the level's w_q_mem field.
            if let Some(w_q_mem_id) = param_ids.frozen_w_q_mem[level] {
                let w_q_mem_grad = tape.get_param_grad(w_q_mem_id);
                // The lp_flat already contains w_q_mem at its offset, but the
                // frozen path registered w_q_mem separately. The lp_flat's
                // w_q_mem slice received no gradient (the tape routed through
                // the separate w_q_mem_id). So replace rather than add.
                lp_grad.w_q_mem = Bf16Storage::from_f32_vec(w_q_mem_grad);
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    idx: usize,
    eps: f32,
) -> f32 {
    let orig = get_weight(params)[idx];

    let mut p_plus = params.clone();
    set_weight(&mut p_plus, idx, orig + eps);
    let eff_plus = get_weight(&p_plus)[idx];
    let mut ctx_plus = make_context_state(cfg);
    let (loss_plus, _) = cms_forward(&p_plus, cfg, input_ids, target_ids, pulse, &mut ctx_plus);

    let mut p_minus = params.clone();
    set_weight(&mut p_minus, idx, orig - eps);
    let eff_minus = get_weight(&p_minus)[idx];
    let mut ctx_minus = make_context_state(cfg);
    let (loss_minus, _) = cms_forward(&p_minus, cfg, input_ids, target_ids, pulse, &mut ctx_minus);

    // Use effective perturbation as denominator — accounts for bf16 quantization.
    let effective_delta = eff_plus - eff_minus;
    if effective_delta.abs() < 1e-30 {
        0.0
    } else {
        (loss_plus - loss_minus) / effective_delta
    }
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
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
    get_weight: impl Fn(&MAGParams) -> &[f32],
    set_weight: impl Fn(&mut MAGParams, usize, f32),
    get_grad: impl Fn(&MAGParams) -> &[f32],
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
#[path = "gradient_tests.rs"]
mod tests;
