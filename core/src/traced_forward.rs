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
use crate::dynamic_freq::FrequencySchedule;

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
    let mut meta = vec![seq_len as f32, d as f32];
    meta.extend_from_slice(extra_meta);
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
    // Learned frequency schedule requires traced gate ops (Stage 3 scope).
    // Fixed schedules pass through unchanged — no gate computation needed.
    let (effective_pulse, freq_cache) = match &cfg.frequency_schedule {
        FrequencySchedule::Learned(_) => {
            unimplemented!(
                "Traced path for Learned frequency schedule \
                 is Stage 3 scope (pluggable retention / CMS variants)"
            );
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
        let lp_flat = level_params_grads_to_flat(&params.levels[level]);
        let lp_id = tape.register_param(&lp_flat, vec![lp_flat.len()]);

        level_param_ids.push(lp_id);

        if pulse.active_levels[level] {
            // Active level: run rule.step(), record opaque, extract final M.
            let initial_m = Some(std::mem::take(&mut context.memory[level]));

            let (y_data, mem_cache, final_m, y_id) = traced_active_level(
                tape, cfg, &params.levels[level], &embedded, s, d,
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
            let w_q_mem_id = tape.register_param(&params.levels[level].w_q_mem, vec![d, d]);
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
                // Delta, Titans, Hebbian, Atlas — all use matrix M: y = M @ q
                _ => delta_rule_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d,
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

            // Record frozen opaque block: input = q_mem_id, saved = [meta, m_frozen]
            let fk = frozen_opaque_key(cfg.memory_rule);
            let meta = vec![s as f32, d as f32];
            let meta_id = tape.alloc(meta, vec![]);
            let m_saved = tape.alloc(frozen_ref.to_vec(), vec![]);
            let y_id = tape.alloc(y_data.clone(), vec![s, d]);
            tape.record_opaque(fk, vec![q_mem_id], vec![y_id], vec![meta_id, m_saved]);

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
            let (y, cache) = DeltaRule.step(level_params, embedded, s, d, initial_m);
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
                tape.alloc(cache.theta_pre.clone(), vec![]),
                tape.alloc(cache.theta.clone(), vec![]),
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

            (y, MemoryCache::Delta(cache), final_m, y_id)
        }
        MemoryRuleKind::TitansLMM => {
            let (y, cache) = TitansLMM.step(level_params, embedded, s, d, initial_m);
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
                tape.alloc(cache.error.clone(), vec![]),
                tape.alloc(cache.grad_outer.clone(), vec![]),
                tape.alloc(cache.y.clone(), vec![]),
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

            (y, MemoryCache::Hebbian(cache), final_m, y_id)
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let dh = cfg.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut final_m = Vec::with_capacity(w1_size + w2_size);
            final_m.extend_from_slice(w1_final);
            final_m.extend_from_slice(w2_final);

            let extra_meta = [dh as f32, cfg.lp_p, cfg.lambda_2];
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
            ];
            let y_id = tape.alloc(y.clone(), vec![s, d]);
            let mut saved = vec![meta_id, lp_saved, emb_saved];
            saved.extend(cache_ids);
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

            (y, MemoryCache::MEMORA(cache), final_m, y_id)
        }
        MemoryRuleKind::LatticeOSR => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

            (y, MemoryCache::Lattice(cache), final_m, y_id)
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            let d_k = cfg.d_compress;
            let sk_size = d_k * d;
            let sv_size = d * d_k;
            let sk_final = &cache.sk_states[s * sk_size..(s + 1) * sk_size];
            let sv_final = &cache.sv_states[s * sv_size..(s + 1) * sv_size];
            let mut final_m = Vec::with_capacity(sk_size + sv_size);
            final_m.extend_from_slice(sk_final);
            final_m.extend_from_slice(sv_final);

            let extra_meta = [d_k as f32, cfg.lambda_k, cfg.lambda_v];
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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

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
            tape.record_opaque(key, vec![emb_id, lp_id], vec![y_id], saved);

            (y, MemoryCache::Atlas(cache), final_m, y_id)
        }
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
}
