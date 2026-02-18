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
// The saved buffer layout is defined by record_on_tape() (Phase 2).
// Convention: saved[0] = embedded, saved[1] = level_params_flat, saved[2..] = cache fields.

use std::collections::HashMap;
use crate::tape::{OpaqueKey, OpaqueBackwardFn};
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
}

/// Reconstruct MemoryLevelParams from a flat slice. Requires knowing d.
/// w_freq and b_freq are variable-length (empty when FrequencySchedule::Fixed,
/// d and 1 respectively when Learned). Determined from remaining slice length.
pub fn level_params_from_flat(flat: &[f32], d: usize) -> MemoryLevelParams {
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
    // w_freq/b_freq: consume remaining (0 if Fixed schedule, d+1 if Learned)
    let remaining = flat.len() - offset;
    let (w_freq, b_freq) = if remaining > 0 {
        debug_assert_eq!(remaining, d + 1,
            "malformed level_params buffer: expected 0 or {} trailing elements, got {}",
            d + 1, remaining);
        (take(&mut offset, d), take(&mut offset, 1))
    } else {
        (vec![], vec![])
    };
    MemoryLevelParams {
        w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta,
        w_eta, b_eta, w_omega, w_freq, b_freq,
    }
}

/// Flatten MemoryLevelParams gradient into a Vec<f32>.
pub fn level_params_grads_to_flat(g: &MemoryLevelParams) -> Vec<f32> {
    let mut out = Vec::new();
    level_params_to_flat(g, &mut out);
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
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
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

    let rule = DeltaRule;
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
}

/// Titans LMM opaque backward adapter.
pub fn titans_lmm_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
    let embedded = saved[2];
    let d_y = d_outputs[0];

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
    };

    let rule = TitansLMM;
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
}

/// Hebbian rule opaque backward adapter.
pub fn hebbian_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
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
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
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
    let level_params = level_params_from_flat(saved[1], d);
    let embedded = saved[2];
    let d_y = d_outputs[0];

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
    };

    let rule = Moneta { d_hidden, lp_p, lambda_2 };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
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
    let level_params = level_params_from_flat(saved[1], d);
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
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
}

/// MEMORA opaque backward adapter.
pub fn memora_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, d_hidden) = read_meta_3(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
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
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
}

/// Lattice OSR opaque backward adapter.
pub fn lattice_osr_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d, m_slots) = read_meta_3(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
    let embedded = saved[2];
    let d_y = d_outputs[0];

    let cache = LatticeCache {
        seq_len, d,
        m: m_slots,
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

    let rule = LatticeOSR { m_slots };
    let (param_grads, d_embedded) = rule.step_backward(&level_params, &cache, d_y, embedded);

    d_inputs[0] = d_embedded;
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
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
    let level_params = level_params_from_flat(saved[1], d);
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
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
}

/// Atlas Omega opaque backward adapter.
pub fn atlas_omega_opaque_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
) {
    let (seq_len, d) = read_meta_2(saved[0]);
    let level_params = level_params_from_flat(saved[1], d);
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
    d_inputs[1] = level_params_grads_to_flat(&param_grads);
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
        let reconstructed = level_params_from_flat(&flat, d);

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
}
