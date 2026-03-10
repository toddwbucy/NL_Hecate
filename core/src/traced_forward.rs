/// Traced forward wrappers: tape-aware computation that records ops for backward.
///
/// Phase 2 of Wengert Tape AD. Each wrapper:
///   1. Reads inputs from tape arena (clones to release borrow)
///   2. Calls the real dispatch/tensor function
///   3. Allocates output in tape arena
///   4. Records the TapeOp
///   5. Returns the output BufId
///
/// `traced_cms_forward()` mirrors `cms_forward()` stage-by-stage,
/// producing bitwise-identical loss while building a gradient-ready tape.
///
/// Gate: `traced_cms_forward()` produces `loss.to_bits() == cms_forward().loss.to_bits()`.

use crate::tape::{Tape, BufId, TapeOp, OpaqueKey};
use crate::opaque_adapters::level_params_grads_to_flat;
use crate::model::{MAGConfig, MAGParams, MemoryRuleKind, MemoryLevelParams};
use crate::mag::{CMSForwardCache, MemoryCache};
use crate::conductor::{Pulse, ContextState};
use crate::delta_rule::{MemoryRule, DeltaRule, delta_rule_read_only};
use crate::titans_lmm::TitansLMM;
use crate::hebbian_rule::HebbianRule;
use crate::moneta::{Moneta, moneta_read_only};
use crate::yaad::{YAAD, yaad_read_only};
use crate::memora::{MEMORA, memora_read_only};
use crate::lattice_osr::{LatticeOSR, lattice_read_only};
use crate::trellis::{Trellis, trellis_read_only};
use crate::atlas_omega::AtlasOmega;
use crate::swiglu_mlp::SwiGluMlp;
use crate::dynamic_freq::{FrequencySchedule, FreqGateCache, should_anneal};
use crate::self_ref::ProjectionKind;

// ── TracedParamIds: map tape BufIds back to MAGParams fields ────────

/// Maps each parameter registered on the tape back to its BufId.
/// Returned by `traced_cms_forward()` so callers can extract gradients
/// via `tape.get_param_grad(id)` after `tape.backward()`.
#[derive(Debug, Clone)]
pub struct TracedParamIds {
    pub w_embed: BufId,
    pub w_q: BufId,
    pub w_k: BufId,
    pub w_v: BufId,
    pub w_o: BufId,
    pub w_unembed: BufId,
    /// lp_flat BufId per level (all levels, active and frozen).
    pub level_params: Vec<BufId>,
    /// Extra w_q_mem param BufId for frozen levels (None for active levels).
    pub frozen_w_q_mem: Vec<Option<BufId>>,
    /// Per-level w_freq BufId for learned frequency gates (None if Fixed schedule).
    pub freq_w_freq: Vec<Option<BufId>>,
    /// Per-level b_freq BufId for learned frequency gates (None if Fixed schedule).
    pub freq_b_freq: Vec<Option<BufId>>,
    /// Combined y buffer ID (pre-sigmoid). Used for freq gate surrogate gradient.
    pub combined_y: Option<BufId>,
}

// ── P2.1: Traced Standard Op Wrappers ───────────────────────────────

/// Embedding lookup: out[t] = table[indices[t]].
pub fn traced_embed_lookup(
    tape: &mut Tape,
    table: BufId,
    indices: &[usize],
    vocab: usize,
    d: usize,
) -> BufId {
    let table_data = tape.buf_data(table).to_vec();
    let seq_len = indices.len();
    let mut out = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let tok = indices[t];
        assert!(tok < vocab, "traced_embed_lookup: indices[{t}]={tok} >= vocab {vocab}");
        out[t * d..(t + 1) * d].copy_from_slice(&table_data[tok * d..(tok + 1) * d]);
    }
    tape.record_with_output(out, vec![seq_len, d], |out_id| {
        TapeOp::EmbedLookup { table, indices: indices.to_vec(), out: out_id, vocab_size: vocab, d }
    })
}

/// Matrix multiply: out = A @ B, A: [m, k], B: [k, n].
pub fn traced_matmul(
    tape: &mut Tape,
    a: BufId,
    b: BufId,
    m: usize,
    k: usize,
    n: usize,
) -> BufId {
    let a_data = tape.buf_data(a).to_vec();
    let b_data = tape.buf_data(b).to_vec();
    let mut out = vec![0.0f32; m * n];
    crate::dispatch::matmul_dispatch(&a_data, &b_data, &mut out, m, k, n);
    tape.record_with_output(out, vec![m, n], |out_id| {
        TapeOp::Matmul { a, b, out: out_id, m, k, n }
    })
}

/// Matrix multiply with transposed B: out = A @ B^T, A: [m, k], B: [n, k].
pub fn traced_matmul_transb(
    tape: &mut Tape,
    a: BufId,
    b: BufId,
    m: usize,
    k: usize,
    n: usize,
) -> BufId {
    let a_data = tape.buf_data(a).to_vec();
    let b_data = tape.buf_data(b).to_vec();
    let mut out = vec![0.0f32; m * n];
    crate::dispatch::matmul_transb_dispatch(&a_data, &b_data, &mut out, m, k, n);
    tape.record_with_output(out, vec![m, n], |out_id| {
        TapeOp::MatmulTransposeB { a, b, out: out_id, m, k, n }
    })
}

/// Element-wise sigmoid: out = sigmoid(input).
pub fn traced_sigmoid(tape: &mut Tape, input: BufId) -> BufId {
    let data = tape.buf_data(input).to_vec();
    let out: Vec<f32> = data.iter().map(|&x| crate::tensor::sigmoid_f32(x)).collect();
    let shape = tape.buf_shape(input).to_vec();
    tape.record_with_output(out, shape, |out_id| {
        TapeOp::Sigmoid { input, out: out_id }
    })
}

/// Element-wise multiply: out = a * b.
pub fn traced_mul(tape: &mut Tape, a: BufId, b: BufId) -> BufId {
    let a_data = tape.buf_data(a).to_vec();
    let b_data = tape.buf_data(b).to_vec();
    debug_assert_eq!(a_data.len(), b_data.len());
    let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(&x, &y)| x * y).collect();
    let shape = tape.buf_shape(a).to_vec();
    tape.record_with_output(out, shape, |out_id| {
        TapeOp::Mul { a, b, out: out_id }
    })
}

/// Element-wise add: out = a + b.
pub fn traced_add(tape: &mut Tape, a: BufId, b: BufId) -> BufId {
    let a_data = tape.buf_data(a).to_vec();
    let b_data = tape.buf_data(b).to_vec();
    debug_assert_eq!(a_data.len(), b_data.len());
    let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(&x, &y)| x + y).collect();
    let shape = tape.buf_shape(a).to_vec();
    tape.record_with_output(out, shape, |out_id| {
        TapeOp::Add { a, b, out: out_id }
    })
}

/// Scalar multiply: out = scalar * input.
pub fn traced_scale(tape: &mut Tape, input: BufId, scalar: f32) -> BufId {
    let data = tape.buf_data(input).to_vec();
    let out: Vec<f32> = data.iter().map(|&x| scalar * x).collect();
    let shape = tape.buf_shape(input).to_vec();
    tape.record_with_output(out, shape, |out_id| {
        TapeOp::Scale { input, scalar, out: out_id }
    })
}

/// Cross-entropy loss: out = -mean(log(softmax(logits)[target])).
pub fn traced_cross_entropy(
    tape: &mut Tape,
    logits: BufId,
    targets: &[usize],
    vocab: usize,
) -> BufId {
    let logits_data = tape.buf_data(logits).to_vec();
    let seq_len = targets.len();
    let loss = crate::tensor::cross_entropy_loss(&logits_data, targets, seq_len, vocab);
    tape.record_with_output(vec![loss], vec![1], |out_id| {
        TapeOp::CrossEntropy { logits, targets: targets.to_vec(), out: out_id, vocab_size: vocab }
    })
}

// ── P2.2: Traced SWA (Opaque Block) ────────────────────────────────

/// Traced SWA forward. Records as opaque block with [q, k, v] as inputs.
/// Returns (attn_out BufId, raw attn_weights Vec for CMSForwardCache).
pub fn traced_swa_forward(
    tape: &mut Tape,
    q: BufId,
    k: BufId,
    v: BufId,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> (BufId, Vec<f32>) {
    let d = num_heads * head_dim;
    let q_data = tape.buf_data(q).to_vec();
    let k_data = tape.buf_data(k).to_vec();
    let v_data = tape.buf_data(v).to_vec();

    let mut attn_out = vec![0.0f32; seq_len * d];
    let mut attn_weights = vec![0.0f32; num_heads * seq_len * window_size];
    crate::dispatch::swa_forward_dispatch(
        &q_data, &k_data, &v_data,
        &mut attn_out, &mut attn_weights,
        seq_len, num_heads, head_dim, window_size,
    );

    // Saved buffers: meta, q, k, v, attn_weights (matches opaque_adapters SWA layout)
    let meta = vec![
        seq_len as f32, num_heads as f32, head_dim as f32, window_size as f32,
    ];
    let meta_id = tape.alloc(meta, vec![]);
    let q_saved = tape.alloc(q_data, vec![]);
    let k_saved = tape.alloc(k_data, vec![]);
    let v_saved = tape.alloc(v_data, vec![]);
    let aw_saved = tape.alloc(attn_weights.clone(), vec![]);

    let out_id = tape.alloc(attn_out, vec![seq_len, d]);

    tape.record_opaque(
        OpaqueKey::SWA,
        vec![q, k, v],
        vec![out_id],
        vec![meta_id, q_saved, k_saved, v_saved, aw_saved],
        None,
    );

    (out_id, attn_weights)
}

// ── P2.3: Active memory rule recording helpers ──────────────────────

/// Record common saved buffers for an active rule opaque block.
/// Returns (meta_id, lp_saved_id, emb_saved_id).
fn alloc_common_saved(
    tape: &mut Tape,
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    extra_meta: &[f32],
) -> (BufId, BufId, BufId) {
    let kernel_size = if level_params.w_k_conv.is_empty() {
        assert!(level_params.w_q_conv.is_empty(),
            "alloc_common_saved: w_k_conv is empty but w_q_conv has {} elements",
            level_params.w_q_conv.len());
        0
    } else {
        assert!(level_params.w_k_conv.len() % d == 0,
            "alloc_common_saved: w_k_conv length {} not divisible by d={}", level_params.w_k_conv.len(), d);
        let ks = level_params.w_k_conv.len() / d;
        assert!(level_params.w_q_conv.len() == d * ks,
            "alloc_common_saved: w_q_conv length {} != w_k_conv-derived d*ks={}*{}={}",
            level_params.w_q_conv.len(), d, ks, d * ks);
        ks
    };
    let mut meta = vec![seq_len as f32, d as f32];
    meta.extend_from_slice(extra_meta);
    meta.push(kernel_size as f32); // always last element — matches record_common_inputs
    let meta_id = tape.alloc(meta, vec![]);
    let lp_flat = level_params_grads_to_flat(level_params);
    let lp_saved = tape.alloc(lp_flat, vec![]);
    let emb_saved = tape.alloc(embedded.to_vec(), vec![seq_len, d]);
    (meta_id, lp_saved, emb_saved)
}

/// Map MemoryRuleKind to active OpaqueKey.
fn active_opaque_key(rule: MemoryRuleKind) -> OpaqueKey {
    match rule {
        MemoryRuleKind::DeltaRule => OpaqueKey::DeltaRule,
        MemoryRuleKind::TitansLMM => OpaqueKey::TitansLMM,
        MemoryRuleKind::HebbianRule => OpaqueKey::HebbianRule,
        MemoryRuleKind::Moneta => OpaqueKey::Moneta,
        MemoryRuleKind::YAAD => OpaqueKey::YAAD,
        MemoryRuleKind::MEMORA => OpaqueKey::MEMORA,
        MemoryRuleKind::LatticeOSR => OpaqueKey::LatticeOSR,
        MemoryRuleKind::Trellis => OpaqueKey::Trellis,
        MemoryRuleKind::AtlasOmega => OpaqueKey::AtlasOmega,
        // SwiGluMlp has no inner-loop M state — active and frozen are identical
        MemoryRuleKind::SwiGluMlp => OpaqueKey::SwiGluMlp,
    }
}

/// Map MemoryRuleKind to frozen OpaqueKey.
fn frozen_opaque_key(rule: MemoryRuleKind) -> OpaqueKey {
    match rule {
        MemoryRuleKind::DeltaRule => OpaqueKey::FrozenDeltaRule,
        MemoryRuleKind::TitansLMM => OpaqueKey::FrozenTitansLMM,
        MemoryRuleKind::HebbianRule => OpaqueKey::FrozenHebbianRule,
        MemoryRuleKind::Moneta => OpaqueKey::FrozenMoneta,
        MemoryRuleKind::YAAD => OpaqueKey::FrozenYAAD,
        MemoryRuleKind::MEMORA => OpaqueKey::FrozenMEMORA,
        MemoryRuleKind::LatticeOSR => OpaqueKey::FrozenLatticeOSR,
        MemoryRuleKind::Trellis => OpaqueKey::FrozenTrellis,
        MemoryRuleKind::AtlasOmega => OpaqueKey::FrozenAtlasOmega,
        // SwiGluMlp has no M state — frozen path reuses the active backward
        MemoryRuleKind::SwiGluMlp => OpaqueKey::SwiGluMlp,
    }
}

// ── P2.4: traced_cms_forward() ──────────────────────────────────────

/// Traced CMS forward pass. Mirrors `cms_forward()` stage-by-stage,
/// recording every operation on the tape for backward differentiation.
///
/// Returns (loss, CMSForwardCache, loss_buf_id, TracedParamIds).
/// The loss is bitwise-identical to `cms_forward()`.
pub fn traced_cms_forward(
    tape: &mut Tape,
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
) -> (f32, CMSForwardCache, BufId, TracedParamIds) {
    let swa_cfg = &cfg.swa;
    let s = swa_cfg.seq_len;
    let d = swa_cfg.d_model;
    let v = swa_cfg.vocab_size;
    let nh = swa_cfg.num_heads;
    let hd = swa_cfg.head_dim;
    let ws = swa_cfg.window_size;

    // TODO(PR-4): Wire OpaqueKey::SelfRef into tape recording + opaque_adapters.
    // Until then, Adaptive projections are not supported in the traced (tape) path.
    assert!(
        cfg.projection_kind != ProjectionKind::Adaptive,
        "traced_cms_forward: ProjectionKind::Adaptive not yet supported in tape path. \
         Use cms_forward + cms_backward directly, or wait for PR 4."
    );

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);
    assert!(target_ids.len() >= s);
    assert_eq!(pulse.active_levels.len(), cfg.k);
    assert_eq!(context.memory.len(), cfg.k);

    // Validate Trellis context memory size.
    if cfg.memory_rule == MemoryRuleKind::Trellis {
        let expected = 2 * cfg.d_compress * d;
        for level in 0..cfg.k {
            assert_eq!(context.memory[level].len(), expected,
                "Trellis context memory[{level}] has wrong size: got {}, expected {expected}.",
                context.memory[level].len());
        }
    }

    // ── Stage 1: Embedding lookup ──────────────────────────────────
    let w_embed_id = tape.register_param(&params.swa.w_embed, vec![v, d]);
    let emb_id = traced_embed_lookup(tape, w_embed_id, &input_ids[..s], v, d);
    let embedded = tape.buf_data(emb_id).to_vec();

    // ── Dynamic frequency gate ─────────────────────────────────────
    // Learned schedule: tape-recorded gate ops so gradients flow to w_freq/b_freq.
    // Fixed schedule: pass through unchanged.
    let mut freq_w_freq_ids: Vec<Option<BufId>> = vec![None; cfg.k];
    let mut freq_b_freq_ids: Vec<Option<BufId>> = vec![None; cfg.k];
    let (effective_pulse, freq_cache) = match &cfg.frequency_schedule {
        FrequencySchedule::Learned(learned_cfg)
            if !should_anneal(pulse.global_step, learned_cfg.anneal_steps) =>
        {
            // Mean-pool embedded tokens: ones_row [1, s] @ embedded [s, d] → [1, d]
            let ones_data: Vec<f32> = vec![1.0 / s as f32; s];
            let ones_id = tape.alloc(ones_data, vec![1, s]);
            let mean_id = traced_matmul(tape, ones_id, emb_id, 1, s, d);
            let embedded_mean = tape.buf_data(mean_id).to_vec();

            // Per-level: pre = dot(mean, w_freq) + b_freq, gate = sigmoid(pre)
            let mut gate_values = Vec::with_capacity(cfg.k);
            let mut gate_pre = Vec::with_capacity(cfg.k);
            let mut active_levels = Vec::with_capacity(cfg.k);

            for l in 0..cfg.k {
                // Allocate (not register_param) — w_freq/b_freq are already inside
                // lp_flat as params. The surrogate gradient in tape_compute_gradients
                // handles their grads via freq_gate_backward(), not get_param_grad().
                let w_freq_id = tape.alloc(params.levels[l].w_freq.clone(), vec![1, d]);
                let b_freq_id = tape.alloc(params.levels[l].b_freq.clone(), vec![1]);

                // pre = mean @ w_freq^T → [1, 1]
                let dot_id = traced_matmul_transb(tape, mean_id, w_freq_id, 1, d, 1);
                // pre + b_freq → [1]
                let pre_id = traced_add(tape, dot_id, b_freq_id);
                let pre_val = tape.buf_data(pre_id)[0];
                gate_pre.push(pre_val);

                // gate = sigmoid(pre)
                let gate_id = traced_sigmoid(tape, pre_id);
                let gate_val = tape.buf_data(gate_id)[0];
                gate_values.push(gate_val);

                // Hard threshold with straight-through estimator
                if l == 0 {
                    // Level 0 always active (spec invariant), but still record
                    // the gate for gradient flow.
                    active_levels.push(true);
                } else {
                    let out_id = tape.alloc(
                        vec![if gate_val > learned_cfg.threshold { 1.0 } else { 0.0 }],
                        vec![1],
                    );
                    tape.record(TapeOp::StraightThroughBool {
                        input: gate_id,
                        threshold: learned_cfg.threshold,
                        out: out_id,
                    });
                    active_levels.push(gate_val > learned_cfg.threshold);
                }
            }

            let new_pulse = Pulse {
                global_step: pulse.global_step,
                active_levels,
            };
            let fc = FreqGateCache {
                gate_values,
                gate_pre,
                embedded_mean,
            };
            (new_pulse, Some(fc))
        }
        _ => (pulse.clone(), None),
    };
    let pulse = &effective_pulse;

    // ── Stage 2a: QKV projections ──────────────────────────────────
    let w_q_id = tape.register_param(&params.swa.w_q, vec![d, d]);
    let w_k_id = tape.register_param(&params.swa.w_k, vec![d, d]);
    let w_v_id = tape.register_param(&params.swa.w_v, vec![d, d]);
    let q_id = traced_matmul_transb(tape, emb_id, w_q_id, s, d, d);
    let k_id = traced_matmul_transb(tape, emb_id, w_k_id, s, d, d);
    let v_id = traced_matmul_transb(tape, emb_id, w_v_id, s, d, d);

    let q_data = tape.buf_data(q_id).to_vec();
    let k_data = tape.buf_data(k_id).to_vec();
    let v_data = tape.buf_data(v_id).to_vec();

    // ── Stage 3a: SWA (opaque) ─────────────────────────────────────
    let (attn_out_id, attn_weights_raw) = traced_swa_forward(
        tape, q_id, k_id, v_id, s, nh, hd, ws,
    );
    let attn_out = tape.buf_data(attn_out_id).to_vec();

    // ── Stage 2b+3b: Per-level memory ──────────────────────────────
    let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut y_per_level_data: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);
    let mut y_ids: Vec<BufId> = Vec::with_capacity(cfg.k);
    let mut level_param_ids: Vec<BufId> = Vec::with_capacity(cfg.k);
    let mut frozen_w_q_mem_ids: Vec<Option<BufId>> = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        // SwiGluMlp: combine standard fields (b_alpha, b_theta, b_eta gate biases are outer-loop
        // params that need gradients) with the MLP weight matrices (gate_proj, up_proj, down_proj).
        // Standard fields (w_k_mem etc.) are empty for SwiGluMlp — no allocation overhead.
        let lp_flat = if cfg.memory_rule == MemoryRuleKind::SwiGluMlp {
            let lp = &params.levels[level];
            let mut flat = level_params_grads_to_flat(lp); // standard fields: gate biases + empty vecs
            flat.extend_from_slice(&lp.gate_proj);
            flat.extend_from_slice(&lp.up_proj);
            flat.extend_from_slice(&lp.down_proj);
            flat
        } else {
            level_params_grads_to_flat(&params.levels[level])
        };
        let lp_id = tape.register_param(&lp_flat, vec![lp_flat.len()]);

        level_param_ids.push(lp_id);

        // SwiGluMlp is stateless (no inner-loop M) — always use the active path.
        let effective_active = pulse.active_levels[level]
            || matches!(cfg.memory_rule, MemoryRuleKind::SwiGluMlp);

        if effective_active {
            // Active level: run rule.step(), record opaque, extract final M.
            let initial_m = Some(std::mem::take(&mut context.memory[level]));

            let (y_data, mem_cache, final_m, y_id) = traced_active_level(
                tape, cfg, level, &params.levels[level], &embedded, s, d,
                initial_m, emb_id, lp_id,
            );

            context.memory[level] = final_m;
            y_ids.push(y_id);
            y_per_level_data.push(y_data);
            memory_caches.push(Some(mem_cache));
            q_mem_per_level.push(None);
            frozen_memories.push(None);
            frozen_w_q_mem_ids.push(None);
        } else {
            // Frozen level: compute q_mem as traced op (for gradient flow),
            // then call rule-specific read_only (for bitwise-identical forward).
            let frozen_ref = &context.memory[level];

            // Traced q_mem projection: connects gradient back to emb_id via tape
            let w_q_f32 = params.levels[level].w_q_mem.as_f32();
            let w_q_mem_id = tape.register_param(&w_q_f32, vec![d, d]);
            let q_mem_id = traced_matmul_transb(tape, emb_id, w_q_mem_id, s, d, d);

            // Call the same read_only function as cms_forward for exact output
            let (y_data, q_mem_data) = match cfg.memory_rule {
                MemoryRuleKind::Moneta => moneta_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::YAAD => yaad_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::MEMORA => memora_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::LatticeOSR => lattice_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, cfg.m_slots,
                ),
                MemoryRuleKind::Trellis => trellis_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, cfg.d_compress,
                ),
                // SwiGluMlp has no M state — must always run via effective_active path above.
                MemoryRuleKind::SwiGluMlp => unreachable!(
                    "SwiGluMlp has no frozen read-only path; \
                     inactive levels must run via the active path (effective_active=true)"
                ),
                // HebbianRule and AtlasOmega: no FM in active step() yet (deferred PR).
                MemoryRuleKind::HebbianRule | MemoryRuleKind::AtlasOmega => delta_rule_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, &crate::feature_map::FeatureMapKind::Identity,
                ),
                // Delta, Titans — support configured feature map.
                _ => delta_rule_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d, &cfg.feature_map,
                ),
            };

            // Verify traced q_mem matches read_only q_mem (guards against
            // CUDA/CPU dispatch divergence in matmul_transb).
            debug_assert_eq!(
                tape.buf_data(q_mem_id),
                q_mem_data.as_slice(),
                "Frozen level {level} ({:?}): traced q_mem diverged from read_only q_mem",
                cfg.memory_rule,
            );

            // Determine which feature map was applied in this frozen level.
            // Only DeltaRule/TitansLMM support cfg.feature_map; all others use Identity.
            let effective_fm = match cfg.memory_rule {
                MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM => &cfg.feature_map,
                _ => &crate::feature_map::FeatureMapKind::Identity,
            };
            let has_fm = !matches!(effective_fm, crate::feature_map::FeatureMapKind::Identity);

            // Compute z_q_mem (phi pre-activations) so backward can apply the phi VJP.
            // Re-running apply_batch on q_mem is cheap: w_rand is frozen, same inputs.
            let fm_z_q_save = if has_fm {
                let (_, z) = crate::feature_map::apply_batch(
                    &q_mem_data, effective_fm,
                    &params.levels[level].w_rand, &params.levels[level].b_rand, s, d,
                );
                z
            } else {
                vec![]
            };
            let (fm_kind_f32, fm_sigma) = match effective_fm {
                crate::feature_map::FeatureMapKind::Identity => (0.0f32, 1.0f32),
                crate::feature_map::FeatureMapKind::RandomFourier { sigma } => (1.0f32, *sigma),
                crate::feature_map::FeatureMapKind::ELU => (2.0f32, 1.0f32),
            };

            // Record frozen opaque block.
            // Meta: [s, d, fm_kind_f32, fm_sigma] (Identity: fm_kind=0, old tapes had len=2).
            // Extra saves (if non-Identity): fm_z_q_mem, w_rand.
            let fk = frozen_opaque_key(cfg.memory_rule);
            let meta = vec![s as f32, d as f32, fm_kind_f32, fm_sigma];
            let meta_id = tape.alloc(meta, vec![]);
            let m_saved = tape.alloc(frozen_ref.to_vec(), vec![]);
            let y_id = tape.alloc(y_data.clone(), vec![s, d]);
            let mut tape_saved = vec![meta_id, m_saved];
            if has_fm {
                tape_saved.push(tape.alloc(fm_z_q_save, vec![]));
                tape_saved.push(tape.alloc(params.levels[level].w_rand.clone(), vec![]));
            }
            tape.record_opaque(fk, vec![q_mem_id], vec![y_id], tape_saved, Some(level));

            y_ids.push(y_id);
            y_per_level_data.push(y_data);
            memory_caches.push(None);
            q_mem_per_level.push(Some(q_mem_data));
            frozen_memories.push(Some(frozen_ref.clone()));
            frozen_w_q_mem_ids.push(Some(w_q_mem_id));
        }
    }

    // ── Stage 4: Combine level outputs ─────────────────────────────
    // y_combined = sum of all level outputs, with 1/sqrt(k) for k>2
    let mut combined_id = y_ids[0];
    for i in 1..cfg.k {
        combined_id = traced_add(tape, combined_id, y_ids[i]);
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        combined_id = traced_scale(tape, combined_id, scale);
    }
    let y_combined = tape.buf_data(combined_id).to_vec();

    // ── Stage 5: Gating — gate = sigmoid(y_combined), gated = attn_out * gate
    let gate_id = traced_sigmoid(tape, combined_id);
    let gated_id = traced_mul(tape, attn_out_id, gate_id);
    let gate = tape.buf_data(gate_id).to_vec();
    let gated_out = tape.buf_data(gated_id).to_vec();

    // ── Stage 6: Output projection
    let w_o_id = tape.register_param(&params.swa.w_o, vec![d, d]);
    let proj_id = traced_matmul_transb(tape, gated_id, w_o_id, s, d, d);
    let projected = tape.buf_data(proj_id).to_vec();

    // ── Stage 7: Unembed
    let w_unembed_id = tape.register_param(&params.swa.w_unembed, vec![d, v]);
    let logits_id = traced_matmul(tape, proj_id, w_unembed_id, s, d, v);
    let logits = tape.buf_data(logits_id).to_vec();

    // ── Stage 8: Cross-entropy loss
    let loss_id = traced_cross_entropy(tape, logits_id, &target_ids[..s], v);
    let loss = tape.buf_data(loss_id)[0];

    let cache = CMSForwardCache {
        embedded,
        q: q_data,
        k: k_data,
        v: v_data,
        attn_out,
        attn_weights: attn_weights_raw,
        memory_caches,
        q_mem_per_level,
        frozen_memories,
        y_per_level: y_per_level_data,
        y_combined,
        gate,
        gated_out,
        projected,
        logits,
        // Traced forward doesn't support residual path yet (tape records legacy path).
        ln_attn_mean: None, ln_attn_rstd: None, ln_attn_out: None,
        ln_mem_mean: None, ln_mem_rstd: None, ln_mem_out: None,
        residual_after_attn: None, residual_final: None,
        pulse: pulse.clone(),
        freq_cache,
    };

    let param_ids = TracedParamIds {
        w_embed: w_embed_id,
        w_q: w_q_id,
        w_k: w_k_id,
        w_v: w_v_id,
        w_o: w_o_id,
        w_unembed: w_unembed_id,
        level_params: level_param_ids,
        frozen_w_q_mem: frozen_w_q_mem_ids,
        freq_w_freq: freq_w_freq_ids,
        freq_b_freq: freq_b_freq_ids,
        combined_y: Some(combined_id),
    };

    (loss, cache, loss_id, param_ids)
}

/// Run an active memory rule, record on tape, return (y_data, MemoryCache, final_m, y_buf_id).
///
/// Cache fields are cloned into the tape arena for backward, while the original
/// cache struct is preserved for CMSForwardCache.
fn traced_active_level(
    tape: &mut Tape,
    cfg: &MAGConfig,
    level: usize,
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    s: usize,
    d: usize,
    initial_m: Option<Vec<f32>>,
    emb_id: BufId,
    lp_id: BufId,
) -> (Vec<f32>, MemoryCache, Vec<f32>, BufId) {
    let key = active_opaque_key(cfg.memory_rule);

    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let rule = DeltaRule::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let fm_kind_f32 = match rule.feature_map {
                crate::feature_map::FeatureMapKind::Identity => 0.0f32,
                crate::feature_map::FeatureMapKind::RandomFourier { .. } => 1.0,
                crate::feature_map::FeatureMapKind::ELU => 2.0,
            };
            let fm_sigma = match rule.feature_map {
                crate::feature_map::FeatureMapKind::RandomFourier { sigma } => sigma,
                _ => 1.0,
            };
            let extra_meta = [crate::moneta::bias_to_f32(rule.bias), rule.sign_sharpness, rule.theta_floor, rule.theta_ceil, fm_kind_f32, fm_sigma];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save feature map z caches and frozen FM weights before conv caches (if non-Identity).
            // Backward adapter reads fm_z_k at saved[15], fm_z_q at saved[16],
            // w_rand at saved[17], b_rand at saved[18].
            if !cache.fm_z_k_mem.is_empty() {
                saved.push(tape.alloc(cache.fm_z_k_mem.clone(), vec![]));
                saved.push(tape.alloc(cache.fm_z_q_mem.clone(), vec![]));
                saved.push(tape.alloc(level_params.w_rand.clone(), vec![]));
                saved.push(tape.alloc(level_params.b_rand.clone(), vec![]));
            }
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Delta(cache), final_m, y_id)
        }
        MemoryRuleKind::TitansLMM => {
            let rule = TitansLMM::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let mk_f32 = match rule.momentum_kind {
                crate::model::MomentumKind::None => 0.0f32,
                crate::model::MomentumKind::EMA => 1.0,
                crate::model::MomentumKind::DeltaMomentum => 2.0,
                crate::model::MomentumKind::DeepMomentum => 3.0,
            };
            let fm_kind_f32 = match rule.feature_map {
                crate::feature_map::FeatureMapKind::Identity => 0.0f32,
                crate::feature_map::FeatureMapKind::RandomFourier { .. } => 1.0,
                crate::feature_map::FeatureMapKind::ELU => 2.0,
            };
            let fm_sigma = match rule.feature_map {
                crate::feature_map::FeatureMapKind::RandomFourier { sigma } => sigma,
                _ => 1.0,
            };
            let extra_meta = [crate::moneta::bias_to_f32(rule.bias), rule.sign_sharpness, mk_f32, rule.theta_floor, rule.theta_ceil, fm_kind_f32, fm_sigma];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let mut cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.s_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.eta_pre.clone(), vec![]),
                tape.alloc(cache.eta.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            // Save DeltaMomentum decay buffer at saved[18] — backward adapter reads it there.
            if rule.momentum_kind == crate::model::MomentumKind::DeltaMomentum {
                assert!(!cache.decay.is_empty(),
                    "traced_forward TitansLMM: DeltaMomentum produced empty decay buffer");
                cache_ids.push(tape.alloc(cache.decay.clone(), vec![]));
            }
            // Save feature map z caches and frozen FM weights before conv caches (if non-Identity).
            // Backward adapter: fm_base = 18 (EMA/None) or 19 (DeltaMomentum).
            // fm_z_k at fm_base, fm_z_q at fm_base+1, w_rand at fm_base+2, b_rand at fm_base+3.
            if !cache.fm_z_k_mem.is_empty() {
                cache_ids.push(tape.alloc(cache.fm_z_k_mem.clone(), vec![]));
                cache_ids.push(tape.alloc(cache.fm_z_q_mem.clone(), vec![]));
                cache_ids.push(tape.alloc(level_params.w_rand.clone(), vec![]));
                cache_ids.push(tape.alloc(level_params.b_rand.clone(), vec![]));
            }
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Titans(cache), final_m, y_id)
        }
        MemoryRuleKind::HebbianRule => {
            let (y, cache) = HebbianRule.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &[]);
            let cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Hebbian(cache), final_m, y_id)
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut final_m = Vec::with_capacity(w1_size + w2_size);
            final_m.extend_from_slice(w1_final);
            final_m.extend_from_slice(w2_final);

            let extra_meta = [dh as f32, cfg.lp_p, cfg.lambda_2, cfg.sign_sharpness, cfg.lq_q, rule.theta_floor, rule.theta_ceil];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let mut cache_ids = vec![
                tape.alloc(cache.w1_states.clone(), vec![]),
                tape.alloc(cache.w2_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.pre_act.clone(), vec![]),
                tape.alloc(cache.hidden.clone(), vec![]),
                tape.alloc(cache.prediction.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            // Save a1/a2 accumulator states when L_q > 2 (needed for backward)
            if !cache.a1_states.is_empty() {
                cache_ids.push(tape.alloc(cache.a1_states.clone(), vec![]));
                cache_ids.push(tape.alloc(cache.a2_states.clone(), vec![]));
            }
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Moneta(cache), final_m, y_id)
        }
        MemoryRuleKind::YAAD => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut final_m = Vec::with_capacity(w1_size + w2_size);
            final_m.extend_from_slice(w1_final);
            final_m.extend_from_slice(w2_final);

            let extra_meta = [dh as f32, cfg.delta, cfg.lambda_local, cfg.lambda_2];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.w1_states.clone(), vec![]),
                tape.alloc(cache.w2_states.clone(), vec![]),
                tape.alloc(cache.w1_boundary.clone(), vec![]),
                tape.alloc(cache.w2_boundary.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.pre_act.clone(), vec![]),
                tape.alloc(cache.hidden.clone(), vec![]),
                tape.alloc(cache.prediction.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::YAAD(cache), final_m, y_id)
        }
        MemoryRuleKind::MEMORA => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut final_m = Vec::with_capacity(w1_size + w2_size);
            final_m.extend_from_slice(w1_final);
            final_m.extend_from_slice(w2_final);

            let extra_meta = [dh as f32];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.w1_states.clone(), vec![]),
                tape.alloc(cache.w2_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.pre_act.clone(), vec![]),
                tape.alloc(cache.hidden.clone(), vec![]),
                tape.alloc(cache.prediction.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
                tape.alloc(cache.log_w1_prev.clone(), vec![]),
                tape.alloc(cache.log_w2_prev.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::MEMORA(cache), final_m, y_id)
        }
        MemoryRuleKind::LatticeOSR => {
            let rule = LatticeOSR { m_slots: cfg.m_slots, variant: cfg.lattice_variant };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let m = cfg.m_slots;
            let s_final = &cache.s_states[s * m * d..(s + 1) * m * d];
            let final_m = s_final.to_vec();

            let extra_meta = [m as f32];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.s_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.scores.clone(), vec![]),
                tape.alloc(cache.slot_gates.clone(), vec![]),
                tape.alloc(cache.read_weights.clone(), vec![]),
                tape.alloc(cache.s_unnorm_norms.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Lattice(cache), final_m, y_id)
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let d_k = cfg.d_compress;
            let sk_size = d_k * d;
            let sv_size = d * d_k;
            let sk_final = &cache.sk_states[s * sk_size..(s + 1) * sk_size];
            let sv_final = &cache.sv_states[s * sv_size..(s + 1) * sv_size];
            let mut final_m = Vec::with_capacity(sk_size + sv_size);
            final_m.extend_from_slice(sk_final);
            final_m.extend_from_slice(sv_final);

            let extra_meta = [d_k as f32, cfg.lambda_k, cfg.lambda_v, rule.theta_floor, rule.theta_ceil];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.sk_states.clone(), vec![]),
                tape.alloc(cache.sv_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.pred_k.clone(), vec![]),
                tape.alloc(cache.error_k.clone(), vec![]),
                tape.alloc(cache.compressed_k_pre.clone(), vec![]),
                tape.alloc(cache.compressed_k.clone(), vec![]),
                tape.alloc(cache.compressed_k_silu.clone(), vec![]),
                tape.alloc(cache.compressed_k_silu_norm.clone(), vec![]),
                tape.alloc(cache.read_compressed_q_pre.clone(), vec![]),
                tape.alloc(cache.read_compressed_q.clone(), vec![]),
                tape.alloc(cache.read_compressed_q_silu.clone(), vec![]),
                tape.alloc(cache.read_compressed_q_silu_norm.clone(), vec![]),
                tape.alloc(cache.pred_v.clone(), vec![]),
                tape.alloc(cache.error_v.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Trellis(cache), final_m, y_id)
        }
        MemoryRuleKind::AtlasOmega => {
            let (y, cache) = AtlasOmega.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &[]);
            let cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.s_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.eta_pre.clone(), vec![]),
                tape.alloc(cache.eta.clone(), vec![]),
                tape.alloc(cache.silu_kv.clone(), vec![]),
                tape.alloc(cache.omega_vecs.clone(), vec![]),
                tape.alloc(cache.omega_mats.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            // Save conv1d cache if active
            assert!(cache.k_conv_cache.is_some() == cache.q_conv_cache.is_some(),
                "traced_forward: partial Conv1D cache — k={}, q={}",
                cache.k_conv_cache.is_some(), cache.q_conv_cache.is_some());
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            (y, MemoryCache::Atlas(cache), final_m, y_id)
        }
        MemoryRuleKind::SwiGluMlp => {
            // SwiGluMlp: no inner-loop M state. Weights are gate/up/down_proj only.
            // saved layout matches swiglu_opaque_backward in swiglu_mlp.rs:
            //   saved[0] = meta [seq_len, d, inter]
            //   saved[1] = lp_id (already contains gate ++ up ++ down flat)
            //   saved[2] = emb_in (embedded input)
            //   saved[3] = x_id  (cache.x — copy of embedded)
            //   saved[4] = gate_out_id
            //   saved[5] = up_out_id
            //   saved[6] = fused_id
            //   saved[7] = gate_cache_id
            let inter = cfg.intermediate_size;
            let rule = SwiGluMlp { intermediate_size: inter };
            // Use trait step() which dispatches to CUDA when built with the cuda feature.
            // Calling step_cpu() directly is fatal at d=2048 (17B ops per naive matmul).
            let (y, cache) = rule.step(level_params, embedded, s, d, None::<Vec<f32>>);

            let meta_id = tape.alloc(vec![s as f32, d as f32, inter as f32], vec![]);
            let emb_saved = tape.alloc(embedded.to_vec(), vec![s, d]);
            let x_id = tape.alloc(cache.x.clone(), vec![]);
            let gate_out_id = tape.alloc(cache.gate_out.clone(), vec![]);
            let up_out_id = tape.alloc(cache.up_out.clone(), vec![]);
            let fused_id = tape.alloc(cache.fused.clone(), vec![]);
            let gc_id = tape.alloc(cache.gate_cache.clone(), vec![]);

            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let saved = vec![meta_id, lp_id, emb_saved, x_id, gate_out_id, up_out_id, fused_id, gc_id];
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level));

            // No M state — return empty vec
            (y, MemoryCache::SwiGlu(cache), vec![], y_id)
        }
    }
}

// ── P2.9: Traced LayerNorm ────────────────────────────────────────────

/// Traced LayerNorm: records on tape for backward differentiation.
/// Returns output BufId. eps = 1e-5 (same as CUDA forward).
pub fn traced_layernorm(
    tape: &mut Tape,
    input: BufId,
    gamma: BufId,
    beta: BufId,
    n: usize,
    d: usize,
) -> BufId {
    let x_data = tape.buf_data(input).to_vec();
    let g_data = tape.buf_data(gamma).to_vec();
    let b_data = tape.buf_data(beta).to_vec();

    let (out_data, mean_cache, rstd_cache) = crate::mag::layer_norm(
        &x_data, &g_data, &b_data, n, d, 1e-5,
    );

    let mean_id = tape.alloc(mean_cache, vec![n]);
    let rstd_id = tape.alloc(rstd_cache, vec![n]);
    tape.record_with_output(out_data, vec![n, d], |out_id| {
        TapeOp::LayerNorm {
            input, gamma, beta, out: out_id,
            mean_cache: mean_id, rstd_cache: rstd_id,
            n, d,
        }
    })
}

// ── P2.10: Traced Stacked Forward ────────────────────────────────────

use crate::stacked_model::{StackedMAGParams, BlockParams};

/// BufId map for stacked model parameters — needed to extract gradients after backward.
#[derive(Debug, Clone)]
pub struct TracedStackedParamIds {
    pub w_embed: BufId,
    pub w_unembed: BufId,
    pub ln_final_gamma: BufId,
    pub ln_final_beta: BufId,
    pub blocks: Vec<TracedBlockParamIds>,
}

/// BufId map for one block's parameters.
#[derive(Debug, Clone)]
pub struct TracedBlockParamIds {
    pub w_q: BufId,
    pub w_k: BufId,
    pub w_v: BufId,
    pub w_o: BufId,
    pub ln_attn_gamma: BufId,
    pub ln_attn_beta: BufId,
    pub ln_mem_gamma: BufId,
    pub ln_mem_beta: BufId,
    /// lp_flat BufId per level
    pub level_params: Vec<BufId>,
}

/// Traced stacked multi-block forward pass.
///
/// Mirrors `gpu_stacked_forward` stage-by-stage on CPU, recording every
/// operation on the tape for backward differentiation.
///
/// Flow:
///   1. Embed tokens (shared w_embed)
///   2. For each block b in 0..n_blocks:
///      a. LN_attn → QKV → SWA → residual skip 1
///      b. LN_mem → per-level memory → combine → residual skip 2
///   3. Final LN (shared ln_final)
///   4. Unembed (shared w_unembed) → cross-entropy loss
///
/// Returns (loss, loss_buf_id, TracedStackedParamIds).
pub fn traced_stacked_forward(
    tape: &mut Tape,
    params: &StackedMAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut Vec<Vec<Vec<f32>>>,   // [n_blocks][k][d*d]
) -> (f32, BufId, TracedStackedParamIds) {
    let s = cfg.swa.seq_len;
    let d = cfg.swa.d_model;
    let v = cfg.swa.vocab_size;
    let nh = cfg.swa.num_heads;
    let hd = cfg.swa.head_dim;
    let ws = cfg.swa.window_size;
    let n_blocks = params.blocks.len();

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);
    assert!(target_ids.len() >= s);
    assert_eq!(pulse.active_levels.len(), cfg.k);
    assert_eq!(context.len(), n_blocks, "context must have one entry per block");
    for (b, bc) in context.iter().enumerate() {
        assert_eq!(bc.len(), cfg.k, "block {b} context must have k={} levels", cfg.k);
    }

    // ── Stage 1: Embedding lookup (shared) ─────────────────────────
    let w_embed_id = tape.register_param(&params.w_embed, vec![v, d]);
    let emb_id = traced_embed_lookup(tape, w_embed_id, &input_ids[..s], v, d);

    // residual_id tracks the residual stream BufId through blocks
    let mut residual_id = emb_id;

    // ── Stage 2: Per-block forward ─────────────────────────────────
    let mut block_param_ids = Vec::with_capacity(n_blocks);

    for b in 0..n_blocks {
        let block = &params.blocks[b];

        // ── LN_attn on residual stream ─────────────────────────
        let ln_attn_gamma_id = tape.register_param(&block.ln_attn_gamma, vec![d]);
        let ln_attn_beta_id = tape.register_param(&block.ln_attn_beta, vec![d]);
        let ln_attn_out_id = traced_layernorm(tape, residual_id, ln_attn_gamma_id, ln_attn_beta_id, s, d);

        // ── QKV projections ────────────────────────────────────
        let w_q_id = tape.register_param(&block.w_q, vec![d, d]);
        let w_k_id = tape.register_param(&block.w_k, vec![d, d]);
        let w_v_id = tape.register_param(&block.w_v, vec![d, d]);
        let q_id = traced_matmul_transb(tape, ln_attn_out_id, w_q_id, s, d, d);
        let k_id = traced_matmul_transb(tape, ln_attn_out_id, w_k_id, s, d, d);
        let v_id = traced_matmul_transb(tape, ln_attn_out_id, w_v_id, s, d, d);

        // ── SWA attention ──────────────────────────────────────
        let (attn_out_id, _attn_weights) = traced_swa_forward(
            tape, q_id, k_id, v_id, s, nh, hd, ws,
        );

        // ── Residual skip 1: residual + attn_out ───────────────
        let residual_after_attn_id = traced_add(tape, residual_id, attn_out_id);

        // ── LN_mem on residual_after_attn ──────────────────────
        let ln_mem_gamma_id = tape.register_param(&block.ln_mem_gamma, vec![d]);
        let ln_mem_beta_id = tape.register_param(&block.ln_mem_beta, vec![d]);
        let ln_mem_out_id = traced_layernorm(tape, residual_after_attn_id, ln_mem_gamma_id, ln_mem_beta_id, s, d);

        // ── Per-level memory ───────────────────────────────────
        let ln_mem_data = tape.buf_data(ln_mem_out_id).to_vec();
        let mut y_ids: Vec<BufId> = Vec::with_capacity(cfg.k);
        let mut level_param_ids = Vec::with_capacity(cfg.k);

        for level in 0..cfg.k {
            let lp = &block.levels[level];
            let lp_flat = crate::opaque_adapters::level_params_grads_to_flat(lp);
            let lp_id = tape.register_param(&lp_flat, vec![lp_flat.len()]);
            level_param_ids.push(lp_id);

            let effective_active = pulse.active_levels[level];

            if effective_active {
                // Active level: reuse traced_active_level with block_index tagging
                let initial_m = Some(std::mem::take(&mut context[b][level]));

                let (y_data, _mem_cache, final_m, y_id) = traced_active_level_stacked(
                    tape, cfg, level, lp, &ln_mem_data, s, d,
                    initial_m, ln_mem_out_id, lp_id, b,
                );

                context[b][level] = final_m;
                y_ids.push(y_id);
                let _ = y_data;
            } else {
                // Frozen level: read-only path
                let frozen_ref = &context[b][level];
                let w_q_f32 = lp.w_q_mem.as_f32();
                let w_q_mem_id = tape.register_param(&w_q_f32, vec![d, d]);
                let q_mem_id = traced_matmul_transb(tape, ln_mem_out_id, w_q_mem_id, s, d, d);

                let (y_data, _q_mem_data) = delta_rule_read_only(
                    lp, &ln_mem_data, frozen_ref, s, d, &cfg.feature_map,
                );

                let fk = frozen_opaque_key(cfg.memory_rule);
                let (fm_kind_f32, fm_sigma) = match &cfg.feature_map {
                    crate::feature_map::FeatureMapKind::Identity => (0.0f32, 1.0f32),
                    crate::feature_map::FeatureMapKind::RandomFourier { sigma } => (1.0f32, *sigma),
                    crate::feature_map::FeatureMapKind::ELU => (2.0f32, 1.0f32),
                };
                let meta = vec![s as f32, d as f32, fm_kind_f32, fm_sigma];
                let meta_id = tape.alloc(meta, vec![]);
                let m_saved = tape.alloc(frozen_ref.to_vec(), vec![]);
                let y_id = tape.alloc(y_data, vec![s, d]);
                let tape_saved = vec![meta_id, m_saved];
                tape.record_opaque_stacked(fk, vec![q_mem_id], vec![y_id], tape_saved, Some(level), Some(b));
                y_ids.push(y_id);
            }
        }

        // ── Combine level outputs ──────────────────────────────
        let mut combined_id = y_ids[0];
        for i in 1..cfg.k {
            combined_id = traced_add(tape, combined_id, y_ids[i]);
        }
        if cfg.k > 2 {
            let scale = 1.0 / (cfg.k as f32).sqrt();
            combined_id = traced_scale(tape, combined_id, scale);
        }

        // ── Residual skip 2: residual_after_attn + y_combined ──
        residual_id = traced_add(tape, residual_after_attn_id, combined_id);

        // Register w_o for gradient tracking. NOT applied as output projection
        // in stacked path — matches gpu_stacked_forward.rs lines 239-241 where
        // w_o is explicitly skipped. The residual stream carries raw attn_out.
        let w_o_id = tape.register_param(&block.w_o, vec![d, d]);

        block_param_ids.push(TracedBlockParamIds {
            w_q: w_q_id,
            w_k: w_k_id,
            w_v: w_v_id,
            w_o: w_o_id,
            ln_attn_gamma: ln_attn_gamma_id,
            ln_attn_beta: ln_attn_beta_id,
            ln_mem_gamma: ln_mem_gamma_id,
            ln_mem_beta: ln_mem_beta_id,
            level_params: level_param_ids,
        });
    }

    // ── Stage 3: Final LayerNorm (shared) ──────────────────────────
    let ln_final_gamma_id = tape.register_param(&params.ln_final_gamma, vec![d]);
    let ln_final_beta_id = tape.register_param(&params.ln_final_beta, vec![d]);
    let ln_final_out_id = traced_layernorm(tape, residual_id, ln_final_gamma_id, ln_final_beta_id, s, d);

    // ── Stage 4: Unembed (shared) ──────────────────────────────────
    let w_unembed_id = tape.register_param(&params.w_unembed, vec![d, v]);
    let logits_id = traced_matmul(tape, ln_final_out_id, w_unembed_id, s, d, v);

    // ── Stage 5: Cross-entropy loss ────────────────────────────────
    let loss_id = traced_cross_entropy(tape, logits_id, &target_ids[..s], v);
    let loss = tape.buf_data(loss_id)[0];

    let param_ids = TracedStackedParamIds {
        w_embed: w_embed_id,
        w_unembed: w_unembed_id,
        ln_final_gamma: ln_final_gamma_id,
        ln_final_beta: ln_final_beta_id,
        blocks: block_param_ids,
    };

    (loss, loss_id, param_ids)
}

/// Run an active memory rule for a stacked block, recording with block_index.
///
/// Same as `traced_active_level` but uses `record_opaque_stacked` to tag
/// the opaque block with its block index for per-block diagnostic extraction.
fn traced_active_level_stacked(
    tape: &mut Tape,
    cfg: &MAGConfig,
    level: usize,
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    s: usize,
    d: usize,
    initial_m: Option<Vec<f32>>,
    emb_id: BufId,
    lp_id: BufId,
    block_index: usize,
) -> (Vec<f32>, MemoryCache, Vec<f32>, BufId) {
    let key = active_opaque_key(cfg.memory_rule);

    // For now, only TitansLMM and DeltaRule are used in stacked configs.
    // This dispatches to the same rule logic as traced_active_level, but
    // records with block_index via record_opaque_stacked.
    match cfg.memory_rule {
        MemoryRuleKind::TitansLMM => {
            let rule = TitansLMM::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let mk_f32 = match rule.momentum_kind {
                crate::model::MomentumKind::None => 0.0f32,
                crate::model::MomentumKind::EMA => 1.0,
                crate::model::MomentumKind::DeltaMomentum => 2.0,
                crate::model::MomentumKind::DeepMomentum => 3.0,
            };
            let fm_kind_f32 = match rule.feature_map {
                crate::feature_map::FeatureMapKind::Identity => 0.0f32,
                crate::feature_map::FeatureMapKind::RandomFourier { .. } => 1.0,
                crate::feature_map::FeatureMapKind::ELU => 2.0,
            };
            let fm_sigma = match rule.feature_map {
                crate::feature_map::FeatureMapKind::RandomFourier { sigma } => sigma,
                _ => 1.0,
            };
            let extra_meta = [crate::moneta::bias_to_f32(rule.bias), rule.sign_sharpness, mk_f32, rule.theta_floor, rule.theta_ceil, fm_kind_f32, fm_sigma];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let mut cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.s_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.eta_pre.clone(), vec![]),
                tape.alloc(cache.eta.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            if rule.momentum_kind == crate::model::MomentumKind::DeltaMomentum {
                cache_ids.push(tape.alloc(cache.decay.clone(), vec![]));
            }
            if !cache.fm_z_k_mem.is_empty() {
                cache_ids.push(tape.alloc(cache.fm_z_k_mem.clone(), vec![]));
                cache_ids.push(tape.alloc(cache.fm_z_q_mem.clone(), vec![]));
                cache_ids.push(tape.alloc(level_params.w_rand.clone(), vec![]));
                cache_ids.push(tape.alloc(level_params.b_rand.clone(), vec![]));
            }
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque_stacked(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level), Some(block_index));

            (y, MemoryCache::Titans(cache), final_m, y_id)
        }
        MemoryRuleKind::DeltaRule => {
            let rule = DeltaRule::from_cfg_level(cfg, level);
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let m_final_start = s * d * d;
            let final_m = cache.m_states[m_final_start..m_final_start + d * d].to_vec();

            let fm_kind_f32 = match rule.feature_map {
                crate::feature_map::FeatureMapKind::Identity => 0.0f32,
                crate::feature_map::FeatureMapKind::RandomFourier { .. } => 1.0,
                crate::feature_map::FeatureMapKind::ELU => 2.0,
            };
            let fm_sigma = match rule.feature_map {
                crate::feature_map::FeatureMapKind::RandomFourier { sigma } => sigma,
                _ => 1.0,
            };
            let extra_meta = [crate::moneta::bias_to_f32(rule.bias), rule.sign_sharpness, rule.theta_floor, rule.theta_ceil, fm_kind_f32, fm_sigma];
            let (meta_id, lp_saved, emb_saved) =
                alloc_common_saved(tape, level_params, embedded, s, d, &extra_meta);
            let cache_ids = vec![
                tape.alloc(cache.m_states.clone(), vec![]),
                tape.alloc(cache.k_mem.clone(), vec![]),
                tape.alloc(cache.v_mem.clone(), vec![]),
                tape.alloc(cache.q_mem.clone(), vec![]),
                tape.alloc(cache.concat_kv.clone(), vec![]),
                tape.alloc(cache.alpha_pre.clone(), vec![]),
                tape.alloc(cache.alpha.clone(), vec![]),
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            if !cache.fm_z_k_mem.is_empty() {
                saved.push(tape.alloc(cache.fm_z_k_mem.clone(), vec![]));
                saved.push(tape.alloc(cache.fm_z_q_mem.clone(), vec![]));
                saved.push(tape.alloc(level_params.w_rand.clone(), vec![]));
                saved.push(tape.alloc(level_params.b_rand.clone(), vec![]));
            }
            if let (Some(kc), Some(qc)) = (&cache.k_conv_cache, &cache.q_conv_cache) {
                saved.push(tape.alloc(kc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(kc.pre_silu.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_conv.clone(), vec![]));
                saved.push(tape.alloc(qc.pre_silu.clone(), vec![]));
            }
            tape.record_opaque_stacked(key, vec![emb_id, lp_id], vec![y_id], saved, Some(level), Some(block_index));

            (y, MemoryCache::Delta(cache), final_m, y_id)
        }
        _ => panic!("traced_stacked_forward: only TitansLMM and DeltaRule are supported for stacked models, got {:?}", cfg.memory_rule),
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::with_tape;
    use crate::opaque_adapters::register_opaque_vjps;
    use crate::model::{MAGConfig, MAGParams, MemoryRuleKind};
    use crate::conductor::{Pulse, ContextState};
    use crate::mag::cms_forward;
    use crate::retention::default_retention;
    use crate::dynamic_freq::FrequencySchedule;

    // ── Helper: build config for a given rule (k=1) ─────────────────

    fn make_config_k1(rule: MemoryRuleKind) -> MAGConfig {
        let mut cfg = MAGConfig::test_config();
        cfg.memory_rule = rule;
        cfg.retention = default_retention(rule);
        match rule {
            MemoryRuleKind::Moneta => { cfg.d_hidden = 4; }
            MemoryRuleKind::YAAD => { cfg.d_hidden = 4; cfg.delta = 1.0; cfg.lambda_local = 0.01; cfg.lambda_2 = 0.01; }
            MemoryRuleKind::MEMORA => { cfg.d_hidden = 4; }
            MemoryRuleKind::LatticeOSR => { cfg.m_slots = 3; }
            MemoryRuleKind::Trellis => { cfg.d_compress = 4; cfg.lambda_k = 0.01; cfg.lambda_v = 0.01; }
            _ => {}
        }
        cfg
    }

    /// Build config for a given rule (k=2, level 0 active, level 1 frozen).
    fn make_config_k2(rule: MemoryRuleKind) -> MAGConfig {
        let mut cfg = MAGConfig::test_config_k2();
        cfg.memory_rule = rule;
        cfg.retention = default_retention(rule);
        match rule {
            MemoryRuleKind::Moneta => { cfg.d_hidden = 4; }
            MemoryRuleKind::YAAD => { cfg.d_hidden = 4; cfg.delta = 1.0; cfg.lambda_local = 0.01; cfg.lambda_2 = 0.01; }
            MemoryRuleKind::MEMORA => { cfg.d_hidden = 4; }
            MemoryRuleKind::LatticeOSR => { cfg.m_slots = 3; }
            MemoryRuleKind::Trellis => { cfg.d_compress = 4; cfg.lambda_k = 0.01; cfg.lambda_v = 0.01; }
            _ => {}
        }
        cfg
    }

    fn make_input(s: usize, v: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut rng = crate::tensor::SimpleRng::new(seed);
        let input_ids: Vec<usize> = (0..s).map(|_| (rng.next_u64() % v as u64) as usize).collect();
        let target_ids: Vec<usize> = (0..s).map(|_| (rng.next_u64() % v as u64) as usize).collect();
        (input_ids, target_ids)
    }

    fn context_for_rule(cfg: &MAGConfig) -> ContextState {
        let d = cfg.swa.d_model;
        match cfg.memory_rule {
            MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
                let w1_size = cfg.d_hidden * d;
                let w2_size = d * cfg.d_hidden;
                ContextState::new_with_memory_size(cfg.k, d, w1_size + w2_size)
            }
            MemoryRuleKind::LatticeOSR => {
                ContextState::new_with_memory_size(cfg.k, d, cfg.m_slots * d)
            }
            MemoryRuleKind::Trellis => {
                ContextState::new_with_memory_size(cfg.k, d, 2 * cfg.d_compress * d)
            }
            _ => ContextState::new(cfg.k, d),
        }
    }

    // ── P2.5a: Individual wrapper sanity tests ──────────────────────

    #[test]
    fn test_traced_matmul_transb_output() {
        with_tape(std::collections::HashMap::new(), |tape| {
            let m = 3; let k = 4; let n = 5;
            let mut rng = crate::tensor::SimpleRng::new(42);
            let mut a = vec![0.0f32; m * k]; rng.fill_uniform(&mut a, 1.0);
            let mut b = vec![0.0f32; n * k]; rng.fill_uniform(&mut b, 1.0);

            let a_id = tape.alloc(a.clone(), vec![m, k]);
            let b_id = tape.alloc(b.clone(), vec![n, k]);
            let out_id = traced_matmul_transb(tape, a_id, b_id, m, k, n);

            let mut expected = vec![0.0f32; m * n];
            crate::dispatch::matmul_transb_dispatch(&a, &b, &mut expected, m, k, n);
            assert_eq!(tape.buf_data(out_id), expected.as_slice());
        });
    }

    #[test]
    fn test_traced_embed_lookup_output() {
        with_tape(std::collections::HashMap::new(), |tape| {
            let v = 8; let d = 4;
            let mut rng = crate::tensor::SimpleRng::new(42);
            let mut table = vec![0.0f32; v * d]; rng.fill_uniform(&mut table, 1.0);
            let indices = vec![2, 5, 0, 7];

            let table_id = tape.alloc(table.clone(), vec![v, d]);
            let out_id = traced_embed_lookup(tape, table_id, &indices, v, d);

            let result = tape.buf_data(out_id);
            for (t, &idx) in indices.iter().enumerate() {
                assert_eq!(&result[t * d..(t + 1) * d], &table[idx * d..(idx + 1) * d]);
            }
        });
    }

    #[test]
    fn test_traced_cross_entropy_output() {
        with_tape(std::collections::HashMap::new(), |tape| {
            let s = 3; let v = 8;
            let mut rng = crate::tensor::SimpleRng::new(42);
            let mut logits = vec![0.0f32; s * v]; rng.fill_uniform(&mut logits, 2.0);
            let targets = vec![1, 3, 5];

            let logits_id = tape.alloc(logits.clone(), vec![s, v]);
            let loss_id = traced_cross_entropy(tape, logits_id, &targets, v);

            let expected = crate::tensor::cross_entropy_loss(&logits, &targets, s, v);
            assert_eq!(tape.buf_data(loss_id)[0].to_bits(), expected.to_bits());
        });
    }

    #[test]
    fn test_traced_swa_output() {
        let registry = register_opaque_vjps();
        with_tape(registry, |tape| {
            let s = 4; let nh = 2; let hd = 4; let ws = 4;
            let d = nh * hd;
            let mut rng = crate::tensor::SimpleRng::new(42);
            let mut q = vec![0.0f32; s * d]; rng.fill_uniform(&mut q, 0.5);
            let mut k = vec![0.0f32; s * d]; rng.fill_uniform(&mut k, 0.5);
            let mut v = vec![0.0f32; s * d]; rng.fill_uniform(&mut v, 0.5);

            let q_id = tape.alloc(q.clone(), vec![s, d]);
            let k_id = tape.alloc(k.clone(), vec![s, d]);
            let v_id = tape.alloc(v.clone(), vec![s, d]);

            let (out_id, aw_traced) = traced_swa_forward(tape, q_id, k_id, v_id, s, nh, hd, ws);

            let mut expected_out = vec![0.0f32; s * d];
            let mut expected_aw = vec![0.0f32; nh * s * ws];
            crate::dispatch::swa_forward_dispatch(&q, &k, &v, &mut expected_out, &mut expected_aw, s, nh, hd, ws);

            assert_eq!(tape.buf_data(out_id), expected_out.as_slice());
            assert_eq!(aw_traced, expected_aw);
        });
    }

    #[test]
    fn test_traced_sigmoid_mul_chain() {
        with_tape(std::collections::HashMap::new(), |tape| {
            let n = 8;
            let mut rng = crate::tensor::SimpleRng::new(42);
            let mut a = vec![0.0f32; n]; rng.fill_uniform(&mut a, 2.0);
            let mut b = vec![0.0f32; n]; rng.fill_uniform(&mut b, 2.0);

            let a_id = tape.alloc(a.clone(), vec![n]);
            let b_id = tape.alloc(b.clone(), vec![n]);

            let sig_id = traced_sigmoid(tape, a_id);
            let mul_id = traced_mul(tape, sig_id, b_id);

            let result = tape.buf_data(mul_id);
            for i in 0..n {
                let expected = crate::tensor::sigmoid_f32(a[i]) * b[i];
                assert_eq!(result[i].to_bits(), expected.to_bits());
            }
        });
    }

    // ── P2.5b: Bitwise identity tests ───────────────────────────────

    /// Test that traced_cms_forward produces bitwise-identical loss to cms_forward.
    fn assert_bitwise_identity_k1(rule: MemoryRuleKind) {
        let cfg = make_config_k1(rule);
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_input(s, v, 123);
        let pulse = Pulse { global_step: 0, active_levels: vec![true] };

        // Reference path
        let mut ctx_ref = context_for_rule(&cfg);
        let (loss_ref, _cache_ref) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref);

        // Traced path
        let registry = register_opaque_vjps();
        let mut ctx_traced = context_for_rule(&cfg);
        let (loss_traced, _cache_traced, _loss_id, _param_ids) = with_tape(registry, |tape| {
            traced_cms_forward(tape, &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_traced)
        });

        assert_eq!(loss_ref.to_bits(), loss_traced.to_bits(),
            "k=1 {:?}: loss_ref={loss_ref} loss_traced={loss_traced}", rule);

        // Context memory must match
        for level in 0..cfg.k {
            assert_eq!(ctx_ref.memory[level], ctx_traced.memory[level],
                "k=1 {:?}: context.memory[{level}] mismatch", rule);
        }
    }

    fn assert_bitwise_identity_k2(rule: MemoryRuleKind) {
        let cfg = make_config_k2(rule);
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_input(s, v, 123);
        // Level 0 active, level 1 frozen
        let pulse = Pulse { global_step: 0, active_levels: vec![true, false] };

        // Reference path
        let mut ctx_ref = context_for_rule(&cfg);
        let (loss_ref, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref);

        // Traced path
        let registry = register_opaque_vjps();
        let mut ctx_traced = context_for_rule(&cfg);
        let (loss_traced, _, _, _) = with_tape(registry, |tape| {
            traced_cms_forward(tape, &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_traced)
        });

        assert_eq!(loss_ref.to_bits(), loss_traced.to_bits(),
            "k=2 {:?}: loss_ref={loss_ref} loss_traced={loss_traced}", rule);

        for level in 0..cfg.k {
            assert_eq!(ctx_ref.memory[level], ctx_traced.memory[level],
                "k=2 {:?}: context.memory[{level}] mismatch", rule);
        }
    }

    // k=1 bitwise identity tests (all active)
    #[test] fn test_bitwise_k1_delta_rule() { assert_bitwise_identity_k1(MemoryRuleKind::DeltaRule); }
    #[test] fn test_bitwise_k1_titans_lmm() { assert_bitwise_identity_k1(MemoryRuleKind::TitansLMM); }
    #[test] fn test_bitwise_k1_hebbian() { assert_bitwise_identity_k1(MemoryRuleKind::HebbianRule); }
    #[test] fn test_bitwise_k1_moneta() { assert_bitwise_identity_k1(MemoryRuleKind::Moneta); }
    #[test] fn test_bitwise_k1_yaad() { assert_bitwise_identity_k1(MemoryRuleKind::YAAD); }
    #[test] fn test_bitwise_k1_memora() { assert_bitwise_identity_k1(MemoryRuleKind::MEMORA); }
    #[test] fn test_bitwise_k1_lattice_osr() { assert_bitwise_identity_k1(MemoryRuleKind::LatticeOSR); }
    #[test] fn test_bitwise_k1_trellis() { assert_bitwise_identity_k1(MemoryRuleKind::Trellis); }
    #[test] fn test_bitwise_k1_atlas_omega() { assert_bitwise_identity_k1(MemoryRuleKind::AtlasOmega); }

    // k=2 bitwise identity tests (level 0 active, level 1 frozen)
    #[test] fn test_bitwise_k2_delta_rule() { assert_bitwise_identity_k2(MemoryRuleKind::DeltaRule); }
    #[test] fn test_bitwise_k2_titans_lmm() { assert_bitwise_identity_k2(MemoryRuleKind::TitansLMM); }
    #[test] fn test_bitwise_k2_hebbian() { assert_bitwise_identity_k2(MemoryRuleKind::HebbianRule); }
    #[test] fn test_bitwise_k2_moneta() { assert_bitwise_identity_k2(MemoryRuleKind::Moneta); }
    #[test] fn test_bitwise_k2_yaad() { assert_bitwise_identity_k2(MemoryRuleKind::YAAD); }
    #[test] fn test_bitwise_k2_memora() { assert_bitwise_identity_k2(MemoryRuleKind::MEMORA); }
    #[test] fn test_bitwise_k2_lattice_osr() { assert_bitwise_identity_k2(MemoryRuleKind::LatticeOSR); }
    #[test] fn test_bitwise_k2_trellis() { assert_bitwise_identity_k2(MemoryRuleKind::Trellis); }
    #[test] fn test_bitwise_k2_atlas_omega() { assert_bitwise_identity_k2(MemoryRuleKind::AtlasOmega); }

    // ── L_q > 2 traced path: bitwise identity ─────────────────────

    /// Verify traced path with lq_q > 2 produces bit-identical results to
    /// the reference cms_forward, exercising the a1/a2 accumulator save/restore
    /// through the opaque adapter.
    #[test]
    fn test_bitwise_k1_moneta_lq3() {
        let mut cfg = make_config_k1(MemoryRuleKind::Moneta);
        cfg.lq_q = 3.0;
        cfg.lambda_2 = 0.1;
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_input(s, v, 123);
        let pulse = Pulse { global_step: 0, active_levels: vec![true] };

        let mut ctx_ref = context_for_rule(&cfg);
        let (loss_ref, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref);

        let registry = register_opaque_vjps();
        let mut ctx_traced = context_for_rule(&cfg);
        let (loss_traced, _, _, _) = with_tape(registry, |tape| {
            traced_cms_forward(tape, &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_traced)
        });

        assert_eq!(loss_ref.to_bits(), loss_traced.to_bits(),
            "k=1 Moneta lq_q=3: loss_ref={loss_ref} loss_traced={loss_traced}");
        for level in 0..cfg.k {
            assert_eq!(ctx_ref.memory[level], ctx_traced.memory[level],
                "k=1 Moneta lq_q=3: context.memory[{level}] mismatch");
        }
    }

    // ── Learned frequency gate: bitwise identity ────────────────────

    #[test]
    fn test_bitwise_learned_freq_gate_k2() {
        use crate::dynamic_freq::LearnedFreqConfig;

        // k=2 DeltaRule with learned frequency gates.
        let mut cfg = MAGConfig::test_config_k2();
        cfg.frequency_schedule = FrequencySchedule::Learned(LearnedFreqConfig::default());

        let params = MAGParams::init(&cfg, 42);
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let (input_ids, target_ids) = make_input(s, v, 123);

        // Both levels active (the gate decides dynamically).
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        // Reference path
        let mut ctx_ref = ContextState::new(cfg.k, cfg.swa.d_model);
        let (loss_ref, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_ref);

        // Traced path
        let registry = register_opaque_vjps();
        let mut ctx_traced = ContextState::new(cfg.k, cfg.swa.d_model);
        let (loss_traced, _, _loss_id, param_ids) = with_tape(registry, |tape| {
            traced_cms_forward(tape, &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_traced)
        });

        assert_eq!(loss_ref.to_bits(), loss_traced.to_bits(),
            "Learned freq gate k=2: loss_ref={loss_ref} loss_traced={loss_traced}");

        // freq_w_freq/freq_b_freq are None — w_freq/b_freq are allocated (not
        // registered as params) because their gradients come from the surrogate
        // mechanism in tape_compute_gradients, not from get_param_grad. The
        // actual param data lives inside lp_flat.
        for l in 0..cfg.k {
            assert!(param_ids.freq_w_freq[l].is_none(),
                "freq_w_freq[{l}] should be None (allocated, not registered)");
            assert!(param_ids.freq_b_freq[l].is_none(),
                "freq_b_freq[{l}] should be None (allocated, not registered)");
        }

        // Context memory must match.
        for level in 0..cfg.k {
            assert_eq!(ctx_ref.memory[level], ctx_traced.memory[level],
                "Learned freq gate: context.memory[{level}] mismatch");
        }
    }
}
