// Opaque VJP adapters: bridge between tape's flat-buffer interface
// and each memory rule's typed step_backward() method.
//
// Spec: specs/infrastructure/differentiation/01_wengert_tape.md §Opaque VJP Registration
//
// Each adapter follows the same pattern:
//   1. Reconstruct MemoryLevelParams + rule-specific Cache from saved flat buffers
//   2. Call the existing step_backward()
//   3. Flatten (MemoryLevelParams grads, d_embedded) back into d_inputs
//
// The saved buffer layout is defined by record_on_tape() implementations below.
// Convention: saved[0] = metadata, saved[1] = level_params_flat, saved[2] = embedded, saved[3..] = cache fields.

use std::collections::HashMap;
use crate::tape::{OpaqueKey, OpaqueBackwardFn, OpaqueVjp, Tape, BufId};
use crate::model::MemoryLevelParams;
use crate::delta_rule::{DeltaRule, DeltaRuleCache, MemoryRule};
use crate::titans_lmm::{TitansLMM, TitansLMMCache};
use crate::hebbian_rule::{HebbianRule, HebbianCache};
use crate::moneta::{Moneta, MonetaCache};
use crate::yaad::{YAAD, YAADCache};
use crate::memora::{MEMORA, MEMORACache};
use crate::lattice_osr::{LatticeOSR, LatticeCache};
use crate::trellis::{Trellis, TrellisCache};
use crate::atlas_omega::{AtlasOmega, AtlasOmegaCache};
use crate::tensor;

// ── MemoryLevelParams serialization ───────────────────────────────────

/// Number of f32 elements in a flattened MemoryLevelParams.
/// Computed from the actual struct contents (handles variable-length w_freq/b_freq).
pub fn level_params_flat_len(p: &MemoryLevelParams) -> usize {
    p.w_k_mem.len() + p.w_v_mem.len() + p.w_q_mem.len()
        + p.w_alpha.len() + p.b_alpha.len()
        + p.w_theta.len() + p.b_theta.len()
        + p.w_eta.len() + p.b_eta.len()
        + p.w_omega.len()
        + p.w_freq.len() + p.b_freq.len()
        + p.w_k_conv.len() + p.b_k_conv.len()
        + p.w_q_conv.len() + p.b_q_conv.len()
}

/// Flatten MemoryLevelParams into a contiguous f32 slice.
pub fn level_params_to_flat(p: &MemoryLevelParams, out: &mut Vec<f32>) {
    out.clear();
    out.extend_from_slice(&p.w_k_mem);
    out.extend_from_slice(&p.w_v_mem);
    out.extend_from_slice(&p.w_q_mem);
    out.extend_from_slice(&p.w_alpha);
    out.extend_from_slice(&p.b_alpha);
    out.extend_from_slice(&p.w_theta);
    out.extend_from_slice(&p.b_theta);
    out.extend_from_slice(&p.w_eta);
    out.extend_from_slice(&p.b_eta);
    out.extend_from_slice(&p.w_omega);
    out.extend_from_slice(&p.w_freq);
    out.extend_from_slice(&p.b_freq);
    out.extend_from_slice(&p.w_k_conv);
    out.extend_from_slice(&p.b_k_conv);
    out.extend_from_slice(&p.w_q_conv);
    out.extend_from_slice(&p.b_q_conv);
}

/// Reconstruct MemoryLevelParams from a flat slice. Requires knowing d.
/// w_freq and b_freq are variable-length (empty when FrequencySchedule::Fixed,
/// d and 1 respectively when Learned). Determined from remaining slice length.
/// `kernel_size`: Conv1D kernel size (0 = no conv fields). When > 0, expects
/// 2*d*kernel_size + 2*d trailing elements for w_k_conv/b_k_conv/w_q_conv/b_q_conv.
pub fn level_params_from_flat(flat: &[f32], d: usize, kernel_size: usize) -> MemoryLevelParams {
    let mut offset = 0;
    let take = |off: &mut usize, n: usize| -> Vec<f32> {
        let slice = flat[*off..*off + n].to_vec();
        *off += n;
        slice
    };
    let w_k_mem = take(&mut offset, d * d);
    let w_v_mem = take(&mut offset, d * d);
    let w_q_mem = take(&mut offset, d * d);
    let w_alpha = take(&mut offset, 2 * d);
    let b_alpha = take(&mut offset, 1);
    let w_theta = take(&mut offset, 2 * d);
    let b_theta = take(&mut offset, 1);
    let w_eta = take(&mut offset, 2 * d);
    let b_eta = take(&mut offset, 1);
    let w_omega = take(&mut offset, d * 2 * d);
    // Variable-length optional fields: freq then conv.
    // Both are detected from the remaining buffer length when kernel_size == 0
    // (opaque backward adapters don't carry kernel_size in metadata).
    let remaining = flat.len() - offset;
    // Determine actual kernel_size: if caller passed 0, infer from buffer.
    // Conv fields occupy 2*d*ks + 2*d = 2*d*(ks+1) elements.
    // Freq fields occupy 0 or d+1 elements.
    // Note: auto-detection is unambiguous for d >= 2 (since d+1 is odd when d is even,
    // it cannot equal 2*d*(ks+1) which is always even). For d=1, pass kernel_size explicitly.
    assert!(kernel_size > 0 || remaining == 0 || d >= 2,
        "auto-detect requires d >= 2; pass kernel_size explicitly for d=1");
    let effective_ks = if kernel_size > 0 {
        kernel_size
    } else if remaining > 0 {
        // Try freq = d+1 first, then check if leftover is valid conv
        let leftover = if remaining >= d + 1 && (remaining == d + 1 || (remaining > d + 1 && (remaining - (d + 1)) % (2 * d) == 0)) {
            remaining - (d + 1)
        } else {
            remaining
        };
        if leftover > 0 && leftover % (2 * d) == 0 {
            leftover / (2 * d) - 1
        } else {
            0
        }
    } else {
        0
    };
    // Now parse freq: present if remaining > conv_size
    let conv_size = if effective_ks > 0 { 2 * d * effective_ks + 2 * d } else { 0 };
    let freq_size = remaining - conv_size;
    if freq_size > 0 {
        assert!(freq_size == d + 1,
            "malformed level_params buffer: expected freq size 0 or {}, got {}",
            d + 1, freq_size);
    }
    let (w_freq, b_freq) = if freq_size > 0 {
        (take(&mut offset, d), take(&mut offset, 1))
    } else {
        (vec![], vec![])
    };
    let (w_k_conv, b_k_conv, w_q_conv, b_q_conv) = if effective_ks > 0 {
        (
            take(&mut offset, d * effective_ks),
            take(&mut offset, d),
            take(&mut offset, d * effective_ks),
            take(&mut offset, d),
        )
    } else {
        (vec![], vec![], vec![], vec![])
    };
    MemoryLevelParams {
        w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta,
        w_eta, b_eta, w_omega, w_freq, b_freq,
        w_k_conv, b_k_conv, w_q_conv, b_q_conv,
    }
}

/// Flatten MemoryLevelParams gradient into a Vec<f32>.
pub fn level_params_grads_to_flat(g: &MemoryLevelParams) -> Vec<f32> {
    let mut out = Vec::new();
    level_params_to_flat(g, &mut out);
    out
}

/// Flatten MemoryLevelParams gradient, padded to `target_len` with zeros.
/// Used when the lp_flat buffer includes w_freq/b_freq (Learned frequency
/// schedule) but the memory rule backward only produces core-field gradients.
pub fn level_params_grads_to_flat_padded(g: &MemoryLevelParams, target_len: usize) -> Vec<f32> {
    let mut out = level_params_grads_to_flat(g);
    out.resize(target_len, 0.0);
    out
}

// ── Metadata encoding ─────────────────────────────────────────────────

// The first saved buffer (saved[0]) encodes metadata as a small f32 vec:
//   [seq_len as f32, d as f32, ...rule-specific fields]
// This avoids needing separate size parameters in the opaque interface.

fn read_meta_2(saved: &[f32]) -> (usize, usize) {
    (saved[0] as usize, saved[1] as usize)
}

fn read_meta_3(saved: &[f32]) -> (usize, usize, usize) {
    (saved[0] as usize, saved[1] as usize, saved[2] as usize)
}

/// Read metadata with attentional bias encoding: [seq_len, d, bias_f32, sign_sharpness].
/// Used by DeltaRule and TitansLMM opaque backward adapters.
fn read_meta_with_bias(saved: &[f32]) -> (usize, usize, crate::model::AttentionalBias, f32) {
    let seq_len = saved[0] as usize;
    let d = saved[1] as usize;
    let (bias, sign_sharpness) = if saved.len() > 3 {
        (crate::moneta::f32_to_bias(saved[2]), saved[3])
    } else {
        (crate::model::AttentionalBias::L2, 10.0)
    };
    (seq_len, d, bias, sign_sharpness)
}

// ── Active rule adapters ──────────────────────────────────────────────
//
// Saved buffer layout for matrix-memory rules (Delta, Titans, Hebbian, Atlas):
//   saved[0] = metadata: [seq_len, d]
//   saved[1] = level_params_flat
//   saved[2] = embedded: [seq_len * d]
//   saved[3..] = cache fields in struct order (flattened contiguously)
//
// d_inputs layout:
//   d_inputs[0] = d_embedded: [seq_len * d]
//   d_inputs[1] = d_level_params: flattened MemoryLevelParams gradients

/// Delta Rule opaque backward adapter.
pub fn delta_rule_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, bias, sign_sharpness) = read_meta_with_bias(saved[0]);
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = DeltaRuleCache {
        seq_len, d,
        m_states: saved[3].to_vec(),
        k_mem: saved[4].to_vec(),
        v_mem: saved[5].to_vec(),
        q_mem: saved[6].to_vec(),
        concat_kv: saved[7].to_vec(),
        alpha_pre: saved[8].to_vec(),
        alpha: saved[9].to_vec(),
        theta_pre: saved[10].to_vec(),
        theta: saved[11].to_vec(),
        error: saved[12].to_vec(),
        grad_outer: saved[13].to_vec(),
        y: saved[14].to_vec(),
    };

    let rule = DeltaRule { bias, sign_sharpness };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Titans LMM opaque backward adapter.
pub fn titans_lmm_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, bias, sign_sharpness) = read_meta_with_bias(saved[0]);
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    // Read momentum_kind from extra_meta if present (new format: meta[4] = momentum_kind as f32).
    // DeepMomentum requires deep_cache to be restored for backward — if those saved entries
    // aren't present, downgrade to EMA to avoid panic in step_backward().
    let momentum_kind = if saved[0].len() > 4 {
        match saved[0][4] as u8 {
            0 => crate::model::MomentumKind::None,
            1 => crate::model::MomentumKind::EMA,
            2 => crate::model::MomentumKind::DeltaMomentum,
            3 => {
                // DeepMomentum backward needs deep_cache; without saved MLP
                // state we can't reconstruct it, so fall back to EMA.
                // Full deep momentum opaque backward is future work.
                crate::model::MomentumKind::EMA
            }
            _ => crate::model::MomentumKind::EMA,
        }
    } else {
        crate::model::MomentumKind::EMA // backward compat: old recordings used EMA
    };

    let cache = TitansLMMCache {
        seq_len, d,
        m_states: saved[3].to_vec(),
        s_states: saved[4].to_vec(),
        k_mem: saved[5].to_vec(),
        v_mem: saved[6].to_vec(),
        q_mem: saved[7].to_vec(),
        concat_kv: saved[8].to_vec(),
        alpha_pre: saved[9].to_vec(),
        alpha: saved[10].to_vec(),
        theta_pre: saved[11].to_vec(),
        theta: saved[12].to_vec(),
        eta_pre: saved[13].to_vec(),
        eta: saved[14].to_vec(),
        error: saved[15].to_vec(),
        grad_outer: saved[16].to_vec(),
        y: saved[17].to_vec(),
        momentum_kind,
        decay: if momentum_kind == crate::model::MomentumKind::DeltaMomentum && saved.len() > 18 {
            saved[18].to_vec()
        } else { vec![] },
        deep_cache: None, // Deep momentum backward through opaque adapter is future work
        deep_d_hidden: 0,
    };

    let rule = TitansLMM {
        bias, sign_sharpness,
        momentum_kind,
        momentum_d_hidden: 0,
    };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Hebbian rule opaque backward adapter.
pub fn hebbian_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = HebbianCache {
        seq_len, d,
        m_states: saved[3].to_vec(),
        k_mem: saved[4].to_vec(),
        v_mem: saved[5].to_vec(),
        q_mem: saved[6].to_vec(),
        concat_kv: saved[7].to_vec(),
        alpha_pre: saved[8].to_vec(),
        alpha: saved[9].to_vec(),
        y: saved[10].to_vec(),
    };

    let rule = HebbianRule;
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Moneta opaque backward adapter.
pub fn moneta_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, d_hidden) = read_meta_3(saved[0]);
    let lp_p = saved[0][3];
    let lambda_2 = saved[0][4];
    let sign_sharpness = if saved[0].len() > 5 { saved[0][5] } else { 10.0 };
    let lq_q = if saved[0].len() > 6 { saved[0][6] } else { 2.0 };
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    // When lq_q > 2, a1_states and a2_states are saved as extra buffers after y
    let (a1_states, a2_states) = if (lq_q - 2.0).abs() >= 1e-6 && saved.len() > 19 {
        (saved[18].to_vec(), saved[19].to_vec())
    } else {
        (vec![], vec![])
    };

    let cache = MonetaCache {
        seq_len, d, d_hidden,
        w1_states: saved[3].to_vec(),
        w2_states: saved[4].to_vec(),
        k_mem: saved[5].to_vec(),
        v_mem: saved[6].to_vec(),
        q_mem: saved[7].to_vec(),
        concat_kv: saved[8].to_vec(),
        alpha_pre: saved[9].to_vec(),
        alpha: saved[10].to_vec(),
        theta_pre: saved[11].to_vec(),
        theta: saved[12].to_vec(),
        pre_act: saved[13].to_vec(),
        hidden: saved[14].to_vec(),
        prediction: saved[15].to_vec(),
        error: saved[16].to_vec(),
        y: saved[17].to_vec(),
        lp_p,
        lambda_2,
        sign_sharpness,
        lq_q,
        a1_states,
        a2_states,
    };

    let rule = Moneta { d_hidden, lp_p, lambda_2, sign_sharpness, lq_q };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// YAAD opaque backward adapter.
pub fn yaad_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, d_hidden) = read_meta_3(saved[0]);
    let delta = saved[0][3];
    let lambda_local = saved[0][4];
    let lambda_2 = saved[0][5];
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = YAADCache {
        seq_len, d, d_hidden,
        w1_states: saved[3].to_vec(),
        w2_states: saved[4].to_vec(),
        w1_boundary: saved[5].to_vec(),
        w2_boundary: saved[6].to_vec(),
        k_mem: saved[7].to_vec(),
        v_mem: saved[8].to_vec(),
        q_mem: saved[9].to_vec(),
        concat_kv: saved[10].to_vec(),
        alpha_pre: saved[11].to_vec(),
        alpha: saved[12].to_vec(),
        theta_pre: saved[13].to_vec(),
        theta: saved[14].to_vec(),
        pre_act: saved[15].to_vec(),
        hidden: saved[16].to_vec(),
        prediction: saved[17].to_vec(),
        error: saved[18].to_vec(),
        y: saved[19].to_vec(),
        delta,
        lambda_local,
        lambda_2,
    };

    let rule = YAAD { d_hidden, delta, lambda_local, lambda_2 };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// MEMORA opaque backward adapter.
pub fn memora_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, d_hidden) = read_meta_3(saved[0]);
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = MEMORACache {
        seq_len, d, d_hidden,
        w1_states: saved[3].to_vec(),
        w2_states: saved[4].to_vec(),
        k_mem: saved[5].to_vec(),
        v_mem: saved[6].to_vec(),
        q_mem: saved[7].to_vec(),
        concat_kv: saved[8].to_vec(),
        alpha_pre: saved[9].to_vec(),
        alpha: saved[10].to_vec(),
        theta_pre: saved[11].to_vec(),
        theta: saved[12].to_vec(),
        pre_act: saved[13].to_vec(),
        hidden: saved[14].to_vec(),
        prediction: saved[15].to_vec(),
        error: saved[16].to_vec(),
        y: saved[17].to_vec(),
        log_w1_prev: saved[18].to_vec(),
        log_w2_prev: saved[19].to_vec(),
    };

    let rule = MEMORA { d_hidden };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Lattice OSR opaque backward adapter.
pub fn lattice_osr_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, m_slots) = read_meta_3(saved[0]);
    let variant = if saved[0].len() > 3 {
        match saved[0][3] as usize {
            1 => crate::model::LatticeVariant::Encode,
            2 => crate::model::LatticeVariant::Similarity,
            _ => crate::model::LatticeVariant::Decode,
        }
    } else {
        crate::model::LatticeVariant::Decode
    };
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = LatticeCache {
        seq_len, d,
        m: m_slots,
        variant,
        s_states: saved[3].to_vec(),
        k_mem: saved[4].to_vec(),
        v_mem: saved[5].to_vec(),
        q_mem: saved[6].to_vec(),
        concat_kv: saved[7].to_vec(),
        alpha_pre: saved[8].to_vec(),
        alpha: saved[9].to_vec(),
        scores: saved[10].to_vec(),
        slot_gates: saved[11].to_vec(),
        read_weights: saved[12].to_vec(),
        s_unnorm_norms: saved[13].to_vec(),
    };

    let rule = LatticeOSR { m_slots, variant };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Trellis opaque backward adapter.
pub fn trellis_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, d_k) = read_meta_3(saved[0]);
    let lambda_k = saved[0][3];
    let lambda_v = saved[0][4];
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = TrellisCache {
        seq_len, d, d_k,
        sk_states: saved[3].to_vec(),
        sv_states: saved[4].to_vec(),
        k_mem: saved[5].to_vec(),
        v_mem: saved[6].to_vec(),
        q_mem: saved[7].to_vec(),
        concat_kv: saved[8].to_vec(),
        alpha_pre: saved[9].to_vec(),
        alpha: saved[10].to_vec(),
        theta_pre: saved[11].to_vec(),
        theta: saved[12].to_vec(),
        pred_k: saved[13].to_vec(),
        error_k: saved[14].to_vec(),
        compressed_k_pre: saved[15].to_vec(),
        compressed_k: saved[16].to_vec(),
        compressed_k_silu: saved[17].to_vec(),
        compressed_k_silu_norm: saved[18].to_vec(),
        read_compressed_q_pre: saved[19].to_vec(),
        read_compressed_q: saved[20].to_vec(),
        read_compressed_q_silu: saved[21].to_vec(),
        read_compressed_q_silu_norm: saved[22].to_vec(),
        pred_v: saved[23].to_vec(),
        error_v: saved[24].to_vec(),
    };

    let rule = Trellis { d_k, lambda_k, lambda_v };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

/// Atlas Omega opaque backward adapter.
pub fn atlas_omega_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d, 0);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = AtlasOmegaCache {
        seq_len, d,
        m_states: saved[3].to_vec(),
        s_states: saved[4].to_vec(),
        k_mem: saved[5].to_vec(),
        v_mem: saved[6].to_vec(),
        q_mem: saved[7].to_vec(),
        concat_kv: saved[8].to_vec(),
        alpha_pre: saved[9].to_vec(),
        alpha: saved[10].to_vec(),
        theta_pre: saved[11].to_vec(),
        theta: saved[12].to_vec(),
        eta_pre: saved[13].to_vec(),
        eta: saved[14].to_vec(),
        silu_kv: saved[15].to_vec(),
        omega_vecs: saved[16].to_vec(),
        omega_mats: saved[17].to_vec(),
        y: saved[18].to_vec(),
    };

    let rule = AtlasOmega;
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat_padded(&param_grads, d_inputs[1].len());
}

// ── SWA adapter ───────────────────────────────────────────────────────
//
// Saved buffer layout:
//   saved[0] = metadata: [seq_len, num_heads, head_dim, window_size]
//   saved[1] = q: [seq_len * num_heads * head_dim]
//   saved[2] = k: [seq_len * num_heads * head_dim]
//   saved[3] = v: [seq_len * num_heads * head_dim]
//   saved[4] = attn_weights: [num_heads * seq_len * window_size]
//
// d_inputs layout:
//   d_inputs[0] = d_q
//   d_inputs[1] = d_k
//   d_inputs[2] = d_v

pub fn swa_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let meta = saved[0];
    let seq_len = meta[0] as usize;
    let num_heads = meta[1] as usize;
    let head_dim = meta[2] as usize;
    let window_size = meta[3] as usize;

    let q = saved[1];
    let k = saved[2];
    let v = saved[3];
    let attn_weights = saved[4];
    let d_attn_out = d_outputs[0];

    let full_dim = seq_len * num_heads * head_dim;
    let mut d_q = vec![0.0f32; full_dim];
    let mut d_k = vec![0.0f32; full_dim];
    let mut d_v = vec![0.0f32; full_dim];

    crate::swa::swa_backward_rust(
        q, k, v, attn_weights, d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        seq_len, num_heads, head_dim, window_size,
    );

    d_inputs[0] = d_q;
    d_inputs[1] = d_k;
    d_inputs[2] = d_v;
}

// ── Frozen read-only adapters ─────────────────────────────────────────
//
// Frozen levels do M @ q_t only (no memory write).
// Gradient: d_q = M^T @ d_y. No gradient to M (frozen = not updated this step).
//
// Saved buffer layout:
//   saved[0] = metadata: [seq_len, d]
//   saved[1] = m_frozen: [d * d]  (the frozen memory matrix)
//
// d_inputs layout:
//   d_inputs[0] = d_q: [seq_len * d]

fn frozen_read_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let m_frozen = saved[1]; // d × d
    let d_y = d_outputs[0]; // seq_len × d

    // d_q[t] = M^T @ d_y[t] for each token
    let mut d_q = vec![0.0f32; seq_len * d];
    let mut m_t = vec![0.0f32; d * d];
    tensor::transpose_f32(m_frozen, &mut m_t, d, d);
    for t in 0..seq_len {
        let dy_t = &d_y[t * d..(t + 1) * d];
        let dq_t = &mut d_q[t * d..(t + 1) * d];
        // dq_t = M^T @ dy_t
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += m_t[i * d + j] * dy_t[j];
            }
            dq_t[i] = sum;
        }
    }

    d_inputs[0] = d_q;
}

// All frozen variants share the same backward (M^T @ d_y).
pub fn frozen_delta_rule_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_titans_lmm_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_hebbian_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_moneta_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_yaad_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_memora_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_lattice_osr_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_trellis_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }
pub fn frozen_atlas_omega_backward(d: &[&[f32]], s: &[&[f32]], di: &mut [Vec<f32>]) { frozen_read_backward(d, s, di) }

// ── OpaqueVjp implementations ────────────────────────────────────────
//
// Each impl produces saved buffers in the exact order the corresponding
// backward adapter expects to reconstruct them. The layout is:
//   inputs[0] = embedded_buf, inputs[1] = level_params_buf
//   outputs[0] = y_buf
//   saved[0] = metadata, saved[1] = level_params_flat, saved[2] = embedded,
//   saved[3..] = cache fields in adapter-expected order

/// Helper: register metadata, level_params, and embedded as saved BufIds.
/// Returns (embedded_input_id, level_params_input_id, metadata_saved_id,
///          level_params_saved_id, embedded_saved_id).
fn record_common_inputs(
    tape: &mut Tape,
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    extra_meta: &[f32],
) -> (BufId, BufId, BufId, BufId, BufId) {
    // Input BufIds (for gradient flow)
    let embedded_input = tape.register_input(embedded, vec![seq_len, d]);
    let lp_flat = level_params_grads_to_flat(level_params);
    let lp_input = tape.register_param(&lp_flat, vec![lp_flat.len()]);

    // Saved BufIds (for backward reconstruction)
    let mut meta = vec![seq_len as f32, d as f32];
    meta.extend_from_slice(extra_meta);
    let meta_id = tape.alloc(meta, vec![]);
    let lp_saved = tape.alloc(lp_flat, vec![]);
    let emb_saved = tape.alloc(embedded.to_vec(), vec![seq_len, d]);

    (embedded_input, lp_input, meta_id, lp_saved, emb_saved)
}

impl OpaqueVjp for DeltaRule {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::DeltaRule }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let extra_meta = [crate::moneta::bias_to_f32(self.bias), self.sign_sharpness];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        // Saved cache fields: same order as delta_rule_opaque_backward reads them
        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.m_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.error, vec![]),
            tape.alloc(cache.grad_outer, vec![]),
            tape.alloc(cache.y, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::DeltaRule,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for TitansLMM {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::TitansLMM }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let mk_f32 = match self.momentum_kind {
            crate::model::MomentumKind::None => 0.0f32,
            crate::model::MomentumKind::EMA => 1.0,
            crate::model::MomentumKind::DeltaMomentum => 2.0,
            crate::model::MomentumKind::DeepMomentum => 3.0,
        };
        let extra_meta = [crate::moneta::bias_to_f32(self.bias), self.sign_sharpness, mk_f32];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let mut cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.m_states, vec![]),
            tape.alloc(cache.s_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.eta_pre, vec![]),
            tape.alloc(cache.eta, vec![]),
            tape.alloc(cache.error, vec![]),
            tape.alloc(cache.grad_outer, vec![]),
            tape.alloc(cache.y, vec![]),
        ];

        // Save DeltaMomentum decay buffer
        if !cache.decay.is_empty() {
            cache_ids.push(tape.alloc(cache.decay, vec![]));
        }

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::TitansLMM,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for HebbianRule {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::HebbianRule }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &[]);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.m_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.y, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::HebbianRule,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for Moneta {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::Moneta }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let extra_meta = [self.d_hidden as f32, self.lp_p, self.lambda_2, self.sign_sharpness, self.lq_q];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let mut cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.w1_states, vec![]),
            tape.alloc(cache.w2_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.pre_act, vec![]),
            tape.alloc(cache.hidden, vec![]),
            tape.alloc(cache.prediction, vec![]),
            tape.alloc(cache.error, vec![]),
            tape.alloc(cache.y, vec![]),
        ];
        // Save a1/a2 accumulator states when L_q > 2 (needed for backward)
        if !cache.a1_states.is_empty() {
            cache_ids.push(tape.alloc(cache.a1_states, vec![]));
            cache_ids.push(tape.alloc(cache.a2_states, vec![]));
        }

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::Moneta,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for YAAD {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::YAAD }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let extra_meta = [self.d_hidden as f32, self.delta, self.lambda_local, self.lambda_2];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.w1_states, vec![]),
            tape.alloc(cache.w2_states, vec![]),
            tape.alloc(cache.w1_boundary, vec![]),
            tape.alloc(cache.w2_boundary, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.pre_act, vec![]),
            tape.alloc(cache.hidden, vec![]),
            tape.alloc(cache.prediction, vec![]),
            tape.alloc(cache.error, vec![]),
            tape.alloc(cache.y, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::YAAD,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for MEMORA {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::MEMORA }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let extra_meta = [self.d_hidden as f32];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.w1_states, vec![]),
            tape.alloc(cache.w2_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.pre_act, vec![]),
            tape.alloc(cache.hidden, vec![]),
            tape.alloc(cache.prediction, vec![]),
            tape.alloc(cache.error, vec![]),
            tape.alloc(cache.y, vec![]),
            tape.alloc(cache.log_w1_prev, vec![]),
            tape.alloc(cache.log_w2_prev, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::MEMORA,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for LatticeOSR {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::LatticeOSR }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let variant_code = match self.variant {
            crate::model::LatticeVariant::Decode => 0.0f32,
            crate::model::LatticeVariant::Encode => 1.0f32,
            crate::model::LatticeVariant::Similarity => 2.0f32,
        };
        let extra_meta = [self.m_slots as f32, variant_code];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.s_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.scores, vec![]),
            tape.alloc(cache.slot_gates, vec![]),
            tape.alloc(cache.read_weights, vec![]),
            tape.alloc(cache.s_unnorm_norms, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::LatticeOSR,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for Trellis {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::Trellis }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let extra_meta = [self.d_k as f32, self.lambda_k, self.lambda_v];
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &extra_meta);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.sk_states, vec![]),
            tape.alloc(cache.sv_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.pred_k, vec![]),
            tape.alloc(cache.error_k, vec![]),
            tape.alloc(cache.compressed_k_pre, vec![]),
            tape.alloc(cache.compressed_k, vec![]),
            tape.alloc(cache.compressed_k_silu, vec![]),
            tape.alloc(cache.compressed_k_silu_norm, vec![]),
            tape.alloc(cache.read_compressed_q_pre, vec![]),
            tape.alloc(cache.read_compressed_q, vec![]),
            tape.alloc(cache.read_compressed_q_silu, vec![]),
            tape.alloc(cache.read_compressed_q_silu_norm, vec![]),
            tape.alloc(cache.pred_v, vec![]),
            tape.alloc(cache.error_v, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::Trellis,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

impl OpaqueVjp for AtlasOmega {
    fn opaque_key(&self) -> OpaqueKey { OpaqueKey::AtlasOmega }

    fn record_on_tape(
        &self, tape: &mut Tape, level_params: &MemoryLevelParams,
        embedded: &[f32], seq_len: usize, d: usize, initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId, BufId, BufId) {
        let (emb_in, lp_in, meta_id, lp_saved, emb_saved) =
            record_common_inputs(tape, level_params, embedded, seq_len, d, &[]);

        let (y, cache) = self.step(level_params, embedded, seq_len, d, initial_m);

        let cache_ids: Vec<BufId> = vec![
            tape.alloc(cache.m_states, vec![]),
            tape.alloc(cache.s_states, vec![]),
            tape.alloc(cache.k_mem, vec![]),
            tape.alloc(cache.v_mem, vec![]),
            tape.alloc(cache.q_mem, vec![]),
            tape.alloc(cache.concat_kv, vec![]),
            tape.alloc(cache.alpha_pre, vec![]),
            tape.alloc(cache.alpha, vec![]),
            tape.alloc(cache.theta_pre, vec![]),
            tape.alloc(cache.theta, vec![]),
            tape.alloc(cache.eta_pre, vec![]),
            tape.alloc(cache.eta, vec![]),
            tape.alloc(cache.silu_kv, vec![]),
            tape.alloc(cache.omega_vecs, vec![]),
            tape.alloc(cache.omega_mats, vec![]),
            tape.alloc(cache.y, vec![]),
        ];

        let y_id = tape.alloc(y.clone(), vec![seq_len, d]);
        let mut saved = vec![meta_id, lp_saved, emb_saved];
        saved.extend(cache_ids);
        tape.record_opaque(OpaqueKey::AtlasOmega,
            vec![emb_in, lp_in], vec![y_id], saved);
        (y, y_id, emb_in, lp_in)
    }
}

// ── Registry builder ──────────────────────────────────────────────────

/// Build the complete opaque VJP registry mapping every OpaqueKey to its
/// backward adapter function. Called once at initialization.
pub fn register_opaque_vjps() -> HashMap<OpaqueKey, OpaqueBackwardFn> {
    let mut registry = HashMap::new();

    // Active rules
    registry.insert(OpaqueKey::DeltaRule, delta_rule_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::TitansLMM, titans_lmm_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::HebbianRule, hebbian_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::Moneta, moneta_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::YAAD, yaad_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::MEMORA, memora_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::LatticeOSR, lattice_osr_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::Trellis, trellis_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::AtlasOmega, atlas_omega_opaque_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::SWA, swa_opaque_backward as OpaqueBackwardFn);

    // Frozen variants
    registry.insert(OpaqueKey::FrozenDeltaRule, frozen_delta_rule_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenTitansLMM, frozen_titans_lmm_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenHebbianRule, frozen_hebbian_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenMoneta, frozen_moneta_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenYAAD, frozen_yaad_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenMEMORA, frozen_memora_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenLatticeOSR, frozen_lattice_osr_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenTrellis, frozen_trellis_backward as OpaqueBackwardFn);
    registry.insert(OpaqueKey::FrozenAtlasOmega, frozen_atlas_omega_backward as OpaqueBackwardFn);

    registry
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_params_roundtrip() {
        use crate::tensor::SimpleRng;
        let d = 4;
        let mut rng = SimpleRng::new(42);
        let params = MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0);

        let flat = level_params_grads_to_flat(&params);
        let reconstructed = level_params_from_flat(&flat, d, 0);

        assert_eq!(params.w_k_mem, reconstructed.w_k_mem);
        assert_eq!(params.w_v_mem, reconstructed.w_v_mem);
        assert_eq!(params.w_q_mem, reconstructed.w_q_mem);
        assert_eq!(params.w_alpha, reconstructed.w_alpha);
        assert_eq!(params.b_alpha, reconstructed.b_alpha);
        assert_eq!(params.w_theta, reconstructed.w_theta);
        assert_eq!(params.b_theta, reconstructed.b_theta);
        assert_eq!(params.w_eta, reconstructed.w_eta);
        assert_eq!(params.b_eta, reconstructed.b_eta);
        assert_eq!(params.w_omega, reconstructed.w_omega);
        assert_eq!(params.w_freq, reconstructed.w_freq);
        assert_eq!(params.b_freq, reconstructed.b_freq);
    }

    #[test]
    fn test_conv_auto_detect_roundtrip() {
        use crate::tensor::SimpleRng;
        let d = 8;
        let kernel_size = 4;
        let mut rng = SimpleRng::new(99);
        let mut params = MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0);
        // Simulate conv fields (as init_conv would produce)
        params.w_k_conv = vec![0.1; d * kernel_size];
        params.b_k_conv = vec![0.0; d];
        params.w_q_conv = vec![0.2; d * kernel_size];
        params.b_q_conv = vec![0.0; d];

        // Serialize with explicit kernel_size
        let flat = level_params_grads_to_flat(&params);

        // Reconstruct with kernel_size=0 (auto-detect)
        let recon = level_params_from_flat(&flat, d, 0);
        assert_eq!(recon.w_k_conv, params.w_k_conv);
        assert_eq!(recon.b_k_conv, params.b_k_conv);
        assert_eq!(recon.w_q_conv, params.w_q_conv);
        assert_eq!(recon.b_q_conv, params.b_q_conv);
        assert_eq!(recon.w_k_mem, params.w_k_mem);
        assert_eq!(recon.w_freq, params.w_freq);
        assert_eq!(recon.b_freq, params.b_freq);
    }

    #[test]
    fn test_register_opaque_vjps_all_keys() {
        let registry = register_opaque_vjps();
        // All 19 keys should be registered
        assert_eq!(registry.len(), 19);
        assert!(registry.contains_key(&OpaqueKey::DeltaRule));
        assert!(registry.contains_key(&OpaqueKey::TitansLMM));
        assert!(registry.contains_key(&OpaqueKey::HebbianRule));
        assert!(registry.contains_key(&OpaqueKey::Moneta));
        assert!(registry.contains_key(&OpaqueKey::YAAD));
        assert!(registry.contains_key(&OpaqueKey::MEMORA));
        assert!(registry.contains_key(&OpaqueKey::LatticeOSR));
        assert!(registry.contains_key(&OpaqueKey::Trellis));
        assert!(registry.contains_key(&OpaqueKey::AtlasOmega));
        assert!(registry.contains_key(&OpaqueKey::SWA));
        assert!(registry.contains_key(&OpaqueKey::FrozenDeltaRule));
        assert!(registry.contains_key(&OpaqueKey::FrozenTitansLMM));
        assert!(registry.contains_key(&OpaqueKey::FrozenHebbianRule));
        assert!(registry.contains_key(&OpaqueKey::FrozenMoneta));
        assert!(registry.contains_key(&OpaqueKey::FrozenYAAD));
        assert!(registry.contains_key(&OpaqueKey::FrozenMEMORA));
        assert!(registry.contains_key(&OpaqueKey::FrozenLatticeOSR));
        assert!(registry.contains_key(&OpaqueKey::FrozenTrellis));
        assert!(registry.contains_key(&OpaqueKey::FrozenAtlasOmega));
    }

    #[test]
    fn test_frozen_read_backward() {
        // 2 tokens, d=2. M = [[1,0],[0,2]], q = [[1,1],[2,3]]
        // y[t] = M @ q[t]: y[0] = [1,2], y[1] = [2,6]
        // d_y = [[1,0],[0,1]]
        // d_q[t] = M^T @ d_y[t]: d_q[0] = M^T @ [1,0] = [1,0], d_q[1] = M^T @ [0,1] = [0,2]
        let meta = vec![2.0, 2.0]; // seq_len=2, d=2
        let m_frozen = vec![1.0, 0.0, 0.0, 2.0]; // [[1,0],[0,2]]
        let d_y = vec![1.0, 0.0, 0.0, 1.0];

        let saved: Vec<&[f32]> = vec![&meta, &m_frozen];
        let d_outputs: Vec<&[f32]> = vec![&d_y];
        let mut d_inputs = vec![vec![0.0f32; 4]]; // d_q

        frozen_read_backward(&d_outputs, &saved, &mut d_inputs);

        // M is symmetric here, so M^T = M
        assert!((d_inputs[0][0] - 1.0).abs() < 1e-6); // d_q[0][0]
        assert!((d_inputs[0][1] - 0.0).abs() < 1e-6); // d_q[0][1]
        assert!((d_inputs[0][2] - 0.0).abs() < 1e-6); // d_q[1][0]
        assert!((d_inputs[0][3] - 2.0).abs() < 1e-6); // d_q[1][1]
    }

    // ── OpaqueVjp round-trip tests ──────────────────────────────────
    //
    // Pattern: record forward on tape → seed unit gradient → backward →
    // compare tape-computed gradients against direct step_backward().

    fn make_test_params(d: usize) -> MemoryLevelParams {
        use crate::tensor::SimpleRng;
        let mut rng = SimpleRng::new(42);
        MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, -1.0)
    }

    fn make_test_embedded(seq_len: usize, d: usize) -> Vec<f32> {
        use crate::tensor::SimpleRng;
        let mut rng = SimpleRng::new(99);
        let mut embedded = vec![0.0f32; seq_len * d];
        rng.fill_uniform(&mut embedded, 0.5);
        embedded
    }

    /// Test that record_on_tape + tape.backward produces the same
    /// d_embedded and d_level_params as calling step + step_backward directly.
    fn assert_opaque_roundtrip<R: MemoryRule>(
        rule: &R, d: usize, seq_len: usize,
    ) {
        let params = make_test_params(d);
        let embedded = make_test_embedded(seq_len, d);

        // --- Direct path: step + step_backward ---
        let (y_direct, cache) = rule.step(&params, &embedded, seq_len, d, None);
        let d_y = vec![1.0f32; seq_len * d]; // unit upstream gradient
        let (param_grads_direct, d_embedded_direct) =
            rule.step_backward(&params, &cache, &d_y, &embedded);

        // --- Tape path: record_on_tape + backward ---
        let registry = register_opaque_vjps();
        let y_tape = crate::tape::with_tape(registry, |tape| {
            let (y, y_id, emb_in, lp_in) =
                rule.record_on_tape(tape, &params, &embedded, seq_len, d, None);

            // Seed y_id with unit upstream gradient and run backward.
            // backward() processes all ops in reverse; loss_id only controls auto-seeding.
            tape.seed_grad(y_id, d_y.clone());
            tape.backward(y_id);

            let d_emb_tape = tape.get_grad(emb_in)
                .expect("embedded should have gradient");
            let d_lp_tape = tape.get_grad(lp_in)
                .expect("level_params should have gradient");

            // Compare d_embedded
            assert_eq!(d_emb_tape.len(), d_embedded_direct.len(),
                "d_embedded length mismatch");
            for (i, (&tape_v, &direct_v)) in d_emb_tape.iter().zip(d_embedded_direct.iter()).enumerate() {
                assert!((tape_v - direct_v).abs() < 1e-5,
                    "d_embedded[{}]: tape={} direct={}", i, tape_v, direct_v);
            }

            // Compare d_level_params
            let lp_grads_direct = level_params_grads_to_flat(&param_grads_direct);
            assert_eq!(d_lp_tape.len(), lp_grads_direct.len(),
                "d_level_params length mismatch");
            for (i, (&tape_v, &direct_v)) in d_lp_tape.iter().zip(lp_grads_direct.iter()).enumerate() {
                assert!((tape_v - direct_v).abs() < 1e-5,
                    "d_level_params[{}]: tape={} direct={}", i, tape_v, direct_v);
            }

            y
        });

        // Verify forward output is identical
        assert_eq!(y_tape.len(), y_direct.len());
        for (i, (&a, &b)) in y_tape.iter().zip(y_direct.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "y[{}]: tape={} direct={}", i, a, b);
        }
    }

    #[test]
    fn test_opaque_vjp_delta_rule() {
        assert_opaque_roundtrip(&DeltaRule::l2(), 4, 3);
    }

    #[test]
    fn test_opaque_vjp_titans_lmm() {
        assert_opaque_roundtrip(&TitansLMM::l2(), 4, 3);
    }

    #[test]
    fn test_opaque_vjp_hebbian() {
        assert_opaque_roundtrip(&HebbianRule, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_moneta() {
        assert_opaque_roundtrip(&Moneta { d_hidden: 8, lp_p: 2.0, lambda_2: 0.01, sign_sharpness: 10.0, lq_q: 2.0 }, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_yaad() {
        assert_opaque_roundtrip(&YAAD { d_hidden: 8, delta: 0.9, lambda_local: 0.1, lambda_2: 0.01 }, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_memora() {
        assert_opaque_roundtrip(&MEMORA { d_hidden: 8 }, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_lattice_osr() {
        assert_opaque_roundtrip(&LatticeOSR { m_slots: 3, variant: crate::model::LatticeVariant::Decode }, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_trellis() {
        assert_opaque_roundtrip(&Trellis { d_k: 3, lambda_k: 0.01, lambda_v: 0.01 }, 4, 3);
    }

    #[test]
    fn test_opaque_vjp_atlas_omega() {
        assert_opaque_roundtrip(&AtlasOmega, 4, 3);
    }

    #[test]
    fn test_opaque_key_correct() {
        assert_eq!(DeltaRule::l2().opaque_key(), OpaqueKey::DeltaRule);
        assert_eq!(TitansLMM::l2().opaque_key(), OpaqueKey::TitansLMM);
        assert_eq!(HebbianRule.opaque_key(), OpaqueKey::HebbianRule);
        assert_eq!((Moneta { d_hidden: 8, lp_p: 2.0, lambda_2: 0.01, sign_sharpness: 10.0, lq_q: 2.0 }).opaque_key(), OpaqueKey::Moneta);
        assert_eq!((YAAD { d_hidden: 8, delta: 0.9, lambda_local: 0.1, lambda_2: 0.01 }).opaque_key(), OpaqueKey::YAAD);
        assert_eq!((MEMORA { d_hidden: 8 }).opaque_key(), OpaqueKey::MEMORA);
        assert_eq!((LatticeOSR { m_slots: 3, variant: crate::model::LatticeVariant::Decode }).opaque_key(), OpaqueKey::LatticeOSR);
        assert_eq!((Trellis { d_k: 3, lambda_k: 0.01, lambda_v: 0.01 }).opaque_key(), OpaqueKey::Trellis);
        assert_eq!(AtlasOmega.opaque_key(), OpaqueKey::AtlasOmega);
    }

    // ── Class 1: Tape Isolation Tests (P1.9) ────────────────────────
    //
    // Spec §Testing Strategy, Class 1: "Opaque block is sole gradient source."
    // For each opaque key: forward with tape, seed backward, verify tape
    // gradient matches the analytical backward EXACTLY (tight tolerance).
    // Any mismatch = tape leaked through barrier or adapter is wrong.
    //
    // Tighter tolerance than P1.8 round-trip tests: rtol=1e-6, atol=1e-8.

    fn assert_close(tape_v: f32, direct_v: f32, label: &str, idx: usize) {
        let diff = (tape_v - direct_v).abs();
        let rtol = 1e-6 * direct_v.abs();
        let atol = 1e-8;
        assert!(diff < rtol.max(atol),
            "Class 1 isolation failure at {}[{}]: tape={:.8e} direct={:.8e} diff={:.8e}",
            label, idx, tape_v, direct_v, diff);
    }

    /// Class 1: tight-tolerance isolation test for active memory rules.
    fn assert_class1_isolation<R: MemoryRule>(
        rule: &R, d: usize, seq_len: usize,
    ) {
        let params = make_test_params(d);
        let embedded = make_test_embedded(seq_len, d);

        // Direct path
        let (y_direct, cache) = rule.step(&params, &embedded, seq_len, d, None);
        let d_y = vec![1.0f32; seq_len * d];
        let (param_grads_direct, d_embedded_direct) =
            rule.step_backward(&params, &cache, &d_y, &embedded);

        // Tape path
        let registry = register_opaque_vjps();
        let y_tape = crate::tape::with_tape(registry, |tape| {
            let (y, y_id, emb_in, lp_in) =
                rule.record_on_tape(tape, &params, &embedded, seq_len, d, None);

            tape.seed_grad(y_id, d_y.clone());
            tape.backward(y_id);

            let d_emb_tape = tape.get_grad(emb_in)
                .expect("embedded should have gradient");
            let d_lp_tape = tape.get_grad(lp_in)
                .expect("level_params should have gradient");

            // Tight tolerance comparison
            assert_eq!(d_emb_tape.len(), d_embedded_direct.len());
            for (i, (&tv, &dv)) in d_emb_tape.iter().zip(d_embedded_direct.iter()).enumerate() {
                assert_close(tv, dv, "d_embedded", i);
            }

            let lp_grads_direct = level_params_grads_to_flat(&param_grads_direct);
            assert_eq!(d_lp_tape.len(), lp_grads_direct.len());
            for (i, (&tv, &dv)) in d_lp_tape.iter().zip(lp_grads_direct.iter()).enumerate() {
                assert_close(tv, dv, "d_level_params", i);
            }

            // Isolation check: only input buffers (emb_in, lp_in) should have gradients.
            // All other non-output buffers should have None (no leakage).
            for buf_id in 0..tape.num_bufs() {
                if buf_id == emb_in || buf_id == lp_in || buf_id == y_id {
                    continue;
                }
                // Saved buffers and metadata should NOT accumulate gradient
                if let Some(grad) = tape.get_grad(buf_id) {
                    let nonzero = grad.iter().any(|&g| g.abs() > 1e-30);
                    assert!(!nonzero,
                        "Gradient leaked to non-input buf {}: norm={}",
                        buf_id, grad.iter().map(|g| g * g).sum::<f32>().sqrt());
                }
            }

            y
        });

        // Forward identical
        for (i, (&a, &b)) in y_tape.iter().zip(y_direct.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "y[{}]: tape={} direct={}", i, a, b);
        }
    }

    #[test]
    fn test_class1_delta_rule() { assert_class1_isolation(&DeltaRule::l2(), 4, 3); }
    #[test]
    fn test_class1_titans_lmm() { assert_class1_isolation(&TitansLMM::l2(), 4, 3); }
    #[test]
    fn test_class1_hebbian() { assert_class1_isolation(&HebbianRule, 4, 3); }
    #[test]
    fn test_class1_moneta() { assert_class1_isolation(&Moneta { d_hidden: 8, lp_p: 2.0, lambda_2: 0.01, sign_sharpness: 10.0, lq_q: 2.0 }, 4, 3); }
    #[test]
    fn test_class1_yaad() { assert_class1_isolation(&YAAD { d_hidden: 8, delta: 0.9, lambda_local: 0.1, lambda_2: 0.01 }, 4, 3); }
    #[test]
    fn test_class1_memora() { assert_class1_isolation(&MEMORA { d_hidden: 8 }, 4, 3); }
    #[test]
    fn test_class1_lattice_osr() { assert_class1_isolation(&LatticeOSR { m_slots: 3, variant: crate::model::LatticeVariant::Decode }, 4, 3); }
    #[test]
    fn test_class1_trellis() { assert_class1_isolation(&Trellis { d_k: 3, lambda_k: 0.01, lambda_v: 0.01 }, 4, 3); }
    #[test]
    fn test_class1_atlas_omega() { assert_class1_isolation(&AtlasOmega, 4, 3); }

    // ── Class 1: SWA isolation ──────────────────────────────────────

    #[test]
    fn test_class1_swa() {
        use crate::tensor::SimpleRng;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 3;
        let window_size = 3;
        let full_dim = seq_len * num_heads * head_dim;

        let mut rng = SimpleRng::new(77);
        let mut q = vec![0.0f32; full_dim];
        let mut k = vec![0.0f32; full_dim];
        let mut v = vec![0.0f32; full_dim];
        rng.fill_uniform(&mut q, 0.5);
        rng.fill_uniform(&mut k, 0.5);
        rng.fill_uniform(&mut v, 0.5);

        // Direct forward + backward
        let mut attn_out = vec![0.0f32; full_dim];
        let mut attn_weights = vec![0.0f32; num_heads * seq_len * window_size];
        crate::swa::swa_forward(
            &q, &k, &v, &mut attn_out, &mut attn_weights,
            seq_len, num_heads, head_dim, window_size,
        );

        let d_attn_out = vec![1.0f32; full_dim];
        let mut d_q_direct = vec![0.0f32; full_dim];
        let mut d_k_direct = vec![0.0f32; full_dim];
        let mut d_v_direct = vec![0.0f32; full_dim];
        crate::swa::swa_backward_rust(
            &q, &k, &v, &attn_weights, &d_attn_out,
            &mut d_q_direct, &mut d_k_direct, &mut d_v_direct,
            seq_len, num_heads, head_dim, window_size,
        );

        // Tape path: manually record SWA opaque op
        let registry = register_opaque_vjps();
        crate::tape::with_tape(registry, |tape| {
            let q_in = tape.register_input(&q, vec![seq_len, num_heads * head_dim]);
            let k_in = tape.register_input(&k, vec![seq_len, num_heads * head_dim]);
            let v_in = tape.register_input(&v, vec![seq_len, num_heads * head_dim]);

            let meta = vec![seq_len as f32, num_heads as f32, head_dim as f32, window_size as f32];
            let meta_id = tape.alloc(meta, vec![]);
            let q_saved = tape.alloc(q.clone(), vec![]);
            let k_saved = tape.alloc(k.clone(), vec![]);
            let v_saved = tape.alloc(v.clone(), vec![]);
            let aw_saved = tape.alloc(attn_weights.clone(), vec![]);
            let out_id = tape.alloc(attn_out.clone(), vec![seq_len, num_heads * head_dim]);

            tape.record_opaque(OpaqueKey::SWA,
                vec![q_in, k_in, v_in],
                vec![out_id],
                vec![meta_id, q_saved, k_saved, v_saved, aw_saved]);

            tape.seed_grad(out_id, d_attn_out.clone());
            tape.backward(out_id);

            let d_q_tape = tape.get_grad(q_in).expect("q should have gradient");
            let d_k_tape = tape.get_grad(k_in).expect("k should have gradient");
            let d_v_tape = tape.get_grad(v_in).expect("v should have gradient");

            for (i, (&tv, &dv)) in d_q_tape.iter().zip(d_q_direct.iter()).enumerate() {
                assert_close(tv, dv, "d_q", i);
            }
            for (i, (&tv, &dv)) in d_k_tape.iter().zip(d_k_direct.iter()).enumerate() {
                assert_close(tv, dv, "d_k", i);
            }
            for (i, (&tv, &dv)) in d_v_tape.iter().zip(d_v_direct.iter()).enumerate() {
                assert_close(tv, dv, "d_v", i);
            }
        });
    }

    // ── Class 1: Frozen variant isolation ───────────────────────────
    //
    // Frozen read: y[t] = M @ q[t]. Backward: d_q[t] = M^T @ d_y[t].
    // Verify tape produces same d_q as direct M^T @ d_y computation.

    fn assert_frozen_isolation(key: OpaqueKey) {
        let seq_len = 3;
        let d = 4;
        let mut rng = crate::tensor::SimpleRng::new(123);
        let mut m_frozen = vec![0.0f32; d * d];
        rng.fill_uniform(&mut m_frozen, 1.0);

        let mut q = vec![0.0f32; seq_len * d];
        rng.fill_uniform(&mut q, 0.5);

        // Direct: y[t] = M @ q[t]
        let mut y = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_frozen[i * d + j] * q[t * d + j];
                }
                y[t * d + i] = sum;
            }
        }

        // Direct backward: d_q[t] = M^T @ d_y[t]
        let d_y = vec![1.0f32; seq_len * d];
        let mut d_q_direct = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_frozen[j * d + i] * d_y[t * d + j]; // M^T
                }
                d_q_direct[t * d + i] = sum;
            }
        }

        // Tape path
        let registry = register_opaque_vjps();
        crate::tape::with_tape(registry, |tape| {
            let q_in = tape.register_input(&q, vec![seq_len, d]);

            let meta = vec![seq_len as f32, d as f32];
            let meta_id = tape.alloc(meta, vec![]);
            let m_saved = tape.alloc(m_frozen.clone(), vec![]);
            let y_id = tape.alloc(y.clone(), vec![seq_len, d]);

            tape.record_opaque(key,
                vec![q_in], vec![y_id], vec![meta_id, m_saved]);

            tape.seed_grad(y_id, d_y.clone());
            tape.backward(y_id);

            let d_q_tape = tape.get_grad(q_in).expect("q should have gradient");
            for (i, (&tv, &dv)) in d_q_tape.iter().zip(d_q_direct.iter()).enumerate() {
                assert_close(tv, dv, "frozen_d_q", i);
            }
        });
    }

    #[test]
    fn test_class1_frozen_delta_rule() { assert_frozen_isolation(OpaqueKey::FrozenDeltaRule); }
    #[test]
    fn test_class1_frozen_titans_lmm() { assert_frozen_isolation(OpaqueKey::FrozenTitansLMM); }
    #[test]
    fn test_class1_frozen_hebbian() { assert_frozen_isolation(OpaqueKey::FrozenHebbianRule); }
    #[test]
    fn test_class1_frozen_moneta() { assert_frozen_isolation(OpaqueKey::FrozenMoneta); }
    #[test]
    fn test_class1_frozen_yaad() { assert_frozen_isolation(OpaqueKey::FrozenYAAD); }
    #[test]
    fn test_class1_frozen_memora() { assert_frozen_isolation(OpaqueKey::FrozenMEMORA); }
    #[test]
    fn test_class1_frozen_lattice_osr() { assert_frozen_isolation(OpaqueKey::FrozenLatticeOSR); }
    #[test]
    fn test_class1_frozen_trellis() { assert_frozen_isolation(OpaqueKey::FrozenTrellis); }
    #[test]
    fn test_class1_frozen_atlas_omega() { assert_frozen_isolation(OpaqueKey::FrozenAtlasOmega); }
}
