/// MAG (Memory-Attention-Gate) composition.
///
/// Architecture:
///   embed → QKV (attn) → SWA ─────────────→ attn_out ──┐
///        \→ KVQ (mem) → Delta Rule → sigmoid → gate ──→ * → output proj → unembed → loss
///
/// Two branches share `embedded` input. Memory output gates attention output
/// via element-wise multiply with sigmoid activation.

use crate::tensor::{matmul_f32, transpose_f32, cross_entropy_loss, sigmoid_f32};
use crate::model::{MAGConfig, MAGParams, MemoryRuleKind};
use crate::delta_rule::{MemoryRule, DeltaRule, DeltaRuleCache, delta_rule_read_only, delta_rule_read_only_backward};
use crate::titans_lmm::{TitansLMM, TitansLMMCache};
use crate::hebbian_rule::{HebbianRule, HebbianCache};
use crate::moneta::{Moneta, MonetaCache, moneta_read_only, moneta_read_only_backward};
use crate::yaad::{YAAD, YAADCache, yaad_read_only, yaad_read_only_backward};
use crate::memora::{MEMORA, MEMORACache, memora_read_only, memora_read_only_backward};
use crate::lattice_osr::{LatticeOSR, LatticeCache, lattice_read_only, lattice_read_only_backward};
use crate::trellis::{Trellis, TrellisCache, trellis_read_only, trellis_read_only_backward};
use crate::conductor::{Pulse, ContextState, ErrorBuffer};

/// Memory cache enum for static dispatch across memory rule variants.
/// Preserves monomorphization (Enzyme requires no vtable indirection).
pub enum MemoryCache {
    Delta(DeltaRuleCache),
    Titans(TitansLMMCache),
    Hebbian(HebbianCache),
    Moneta(MonetaCache),
    YAAD(YAADCache),
    MEMORA(MEMORACache),
    Lattice(LatticeCache),
    Trellis(TrellisCache),
}

/// Cache for MAG forward pass — holds both branches' intermediates.
pub struct MAGForwardCache {
    // Shared
    pub embedded: Vec<f32>,
    // Attention branch (from SWA forward)
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_weights: Vec<f32>,
    // Memory branch
    pub memory_cache: MemoryCache,
    // Gating
    pub gate: Vec<f32>,        // sigmoid(y_t): [seq_len, d]
    pub gated_out: Vec<f32>,   // attn_out * gate: [seq_len, d]
    // Post-gating
    pub projected: Vec<f32>,   // gated_out @ W_O^T: [seq_len, d]
    pub logits: Vec<f32>,      // projected @ W_unembed: [seq_len, vocab]
}

/// MAG forward pass. Returns (loss, cache).
pub fn mag_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MAGForwardCache) {
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

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < v, "mag_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        let src = &params.swa.w_embed[tok * d..(tok + 1) * d];
        embedded[t * d..(t + 1) * d].copy_from_slice(src);
    }

    // Stage 2a: Attention branch — QKV projections
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    matmul_f32(&embedded, &w_q_t, &mut q, s, d, d);
    matmul_f32(&embedded, &w_k_t, &mut k, s, d, d);
    matmul_f32(&embedded, &w_v_t, &mut vv, s, d, d);

    // Stage 3a: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 2b+3b: Memory branch — dispatch based on memory rule
    let (y, memory_cache) = match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let (y, cache) = DeltaRule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Delta(cache))
        }
        MemoryRuleKind::TitansLMM => {
            let (y, cache) = TitansLMM.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Titans(cache))
        }
        MemoryRuleKind::HebbianRule => {
            let (y, cache) = HebbianRule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Hebbian(cache))
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Moneta(cache))
        }
        MemoryRuleKind::YAAD => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::YAAD(cache))
        }
        MemoryRuleKind::MEMORA => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::MEMORA(cache))
        }
        MemoryRuleKind::LatticeOSR => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Lattice(cache))
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Trellis(cache))
        }
    };

    // Stage 4: Gating — gate = sigmoid(y), gated_out = attn_out * gate
    let mut gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        gate[i] = sigmoid_f32(y[i]);
        gated_out[i] = attn_out[i] * gate[i];
    }

    // Stage 5: Output projection — projected = gated_out @ W_O^T
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&gated_out, &w_o_t, &mut projected, s, d, d);

    // Stage 6: Unembed — logits = projected @ W_unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 7: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = MAGForwardCache {
        embedded, q, k, v: vv, attn_out, attn_weights,
        memory_cache, gate, gated_out, projected, logits,
    };

    (loss, cache)
}

/// MAG full backward pass. Returns parameter gradients.
pub fn mag_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &MAGForwardCache,
    input_ids: &[usize],
    target_ids: &[usize],
) -> MAGParams {
    let swa_cfg = &cfg.swa;
    let s = swa_cfg.seq_len;
    let d = swa_cfg.d_model;
    let v = swa_cfg.vocab_size;
    let nh = swa_cfg.num_heads;
    let hd = swa_cfg.head_dim;
    let ws = swa_cfg.window_size;

    let mut grads = MAGParams::zeros_like(cfg);

    // ── Stage 7: Cross-entropy gradient ──────────────────────────────
    let mut d_logits = vec![0.0f32; s * v];
    let count = (0..s)
        .filter(|&t| target_ids.get(t).map_or(false, |&tok| tok < v))
        .count() as f32;
    if count > 0.0 {
        for t in 0..s {
            let target = match target_ids.get(t) {
                Some(&tok) if tok < v => tok,
                _ => continue,
            };
            let base = t * v;
            let row = &cache.logits[base..base + v];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for j in 0..v {
                let e = (row[j] - max_val).exp();
                d_logits[base + j] = e;
                sum_exp += e;
            }
            for j in 0..v {
                d_logits[base + j] /= sum_exp;
            }
            d_logits[base + target] -= 1.0;
            for j in 0..v {
                d_logits[base + j] /= count;
            }
        }
    }

    // ── Stage 6: Unembed backward ────────────────────────────────────
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.swa.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    matmul_f32(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 5: Output projection backward ──────────────────────────
    // projected = gated_out @ W_O^T
    let mut d_gated_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 4: Gating backward ─────────────────────────────────────
    // gated_out = attn_out * gate
    // d_attn_out = d_gated_out * gate
    // d_gate = d_gated_out * attn_out
    let mut d_attn_out = vec![0.0f32; s * d];
    let mut d_gate = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_attn_out[i] = d_gated_out[i] * cache.gate[i];
        d_gate[i] = d_gated_out[i] * cache.attn_out[i];
    }

    // gate = sigmoid(y) → d_y = d_gate * gate * (1 - gate)
    let mut d_y = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y[i] = d_gate[i] * cache.gate[i] * (1.0 - cache.gate[i]);
    }

    // ── Stage 3b: Memory backward — dispatch on cache variant ──────────
    let (mem_grads, d_embedded_mem) = match &cache.memory_cache {
        MemoryCache::Delta(delta_cache) => {
            DeltaRule.step_backward(&params.levels[0], delta_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Titans(titans_cache) => {
            TitansLMM.step_backward(&params.levels[0], titans_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Hebbian(hebbian_cache) => {
            HebbianRule.step_backward(&params.levels[0], hebbian_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Moneta(moneta_cache) => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            rule.step_backward(&params.levels[0], moneta_cache, &d_y, &cache.embedded)
        }
        MemoryCache::YAAD(yaad_cache) => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            rule.step_backward(&params.levels[0], yaad_cache, &d_y, &cache.embedded)
        }
        MemoryCache::MEMORA(memora_cache) => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            rule.step_backward(&params.levels[0], memora_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Lattice(lattice_cache) => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            rule.step_backward(&params.levels[0], lattice_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Trellis(trellis_cache) => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            rule.step_backward(&params.levels[0], trellis_cache, &d_y, &cache.embedded)
        }
    };

    // Accumulate memory parameter gradients into level 0
    grads.levels[0].accumulate(&mem_grads);

    // ── Stage 3a: SWA Attention backward ─────────────────────────────
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 2a: QKV projection backward ────────────────────────────
    let mut d_embedded = vec![0.0f32; s * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_embedded, s, d, d);

    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    matmul_f32(&d_q_t, &cache.embedded, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    matmul_f32(&d_k_t, &cache.embedded, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    matmul_f32(&d_v_t, &cache.embedded, &mut grads.swa.w_v, d, s, d);

    // ── Combine d_embedded from both branches ────────────────────────
    for i in 0..(s * d) {
        d_embedded[i] += d_embedded_mem[i];
    }

    // ── Stage 1: Embedding scatter-add ───────────────────────────────
    for t in 0..s {
        let tok = input_ids[t];
        for dd in 0..d {
            grads.swa.w_embed[tok * d + dd] += d_embedded[t * d + dd];
        }
    }

    grads
}

// ── CMS (Continuous Memory Systems) ─────────────────────────────────

/// Cache for CMS forward pass — multiple frequency levels.
pub struct CMSForwardCache {
    pub embedded: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_weights: Vec<f32>,
    /// Per-level memory cache. None if level was frozen (read-only).
    pub memory_caches: Vec<Option<MemoryCache>>,
    /// Per-level q_mem for frozen levels (needed for backward).
    pub q_mem_per_level: Vec<Option<Vec<f32>>>,
    /// Per-level frozen M matrix (for frozen level backward). None if active.
    pub frozen_memories: Vec<Option<Vec<f32>>>,
    /// Per-level memory output.
    pub y_per_level: Vec<Vec<f32>>,
    /// Combined memory output (sum of all levels).
    pub y_combined: Vec<f32>,
    pub gate: Vec<f32>,
    pub gated_out: Vec<f32>,
    pub projected: Vec<f32>,
    pub logits: Vec<f32>,
    /// Which levels were active this step.
    pub pulse: Pulse,
}

/// CMS forward pass. Like mag_forward but with k frequency levels.
///
/// Active levels run full DeltaRule (write + read), updating context.memory[level].
/// Frozen levels do read-only: y_t = M @ q_t using persisted memory.
pub fn cms_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
) -> (f32, CMSForwardCache) {
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

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < v, "cms_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Stage 2a: Attention branch — QKV projections
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    matmul_f32(&embedded, &w_q_t, &mut q, s, d, d);
    matmul_f32(&embedded, &w_k_t, &mut k, s, d, d);
    matmul_f32(&embedded, &w_v_t, &mut vv, s, d, d);

    // Stage 3a: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 2b+3b: Memory branch — per-level dispatch
    let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut y_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            // Active level: full memory write + read, seeded from persisted memory.
            // Take ownership — context.memory[level] will be replaced after step().
            let initial_m = Some(std::mem::take(&mut context.memory[level]));
            let (y_level, mem_cache) = match cfg.memory_rule {
                MemoryRuleKind::DeltaRule => {
                    let (y, cache) = DeltaRule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let m_final_start = s * d * d;
                    context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                    (y, MemoryCache::Delta(cache))
                }
                MemoryRuleKind::TitansLMM => {
                    let (y, cache) = TitansLMM.step(&params.levels[level], &embedded, s, d, initial_m);
                    let m_final_start = s * d * d;
                    context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                    (y, MemoryCache::Titans(cache))
                }
                MemoryRuleKind::HebbianRule => {
                    let (y, cache) = HebbianRule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let m_final_start = s * d * d;
                    context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                    (y, MemoryCache::Hebbian(cache))
                }
                MemoryRuleKind::Moneta => {
                    let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
                    let (y, cache) = rule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let dh = cfg.d_hidden;
                    let w1_size = dh * d;
                    let w2_size = d * dh;
                    let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
                    let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
                    let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
                    ctx_mem.extend_from_slice(w1_final);
                    ctx_mem.extend_from_slice(w2_final);
                    context.memory[level] = ctx_mem;
                    (y, MemoryCache::Moneta(cache))
                }
                MemoryRuleKind::YAAD => {
                    let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
                    let (y, cache) = rule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let dh = cfg.d_hidden;
                    let w1_size = dh * d;
                    let w2_size = d * dh;
                    let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
                    let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
                    let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
                    ctx_mem.extend_from_slice(w1_final);
                    ctx_mem.extend_from_slice(w2_final);
                    context.memory[level] = ctx_mem;
                    (y, MemoryCache::YAAD(cache))
                }
                MemoryRuleKind::MEMORA => {
                    let rule = MEMORA { d_hidden: cfg.d_hidden };
                    let (y, cache) = rule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let dh = cfg.d_hidden;
                    let w1_size = dh * d;
                    let w2_size = d * dh;
                    let w1_final = &cache.w1_states[s * w1_size..(s + 1) * w1_size];
                    let w2_final = &cache.w2_states[s * w2_size..(s + 1) * w2_size];
                    let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
                    ctx_mem.extend_from_slice(w1_final);
                    ctx_mem.extend_from_slice(w2_final);
                    context.memory[level] = ctx_mem;
                    (y, MemoryCache::MEMORA(cache))
                }
                MemoryRuleKind::LatticeOSR => {
                    let rule = LatticeOSR { m_slots: cfg.m_slots };
                    let (y, cache) = rule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let m = cfg.m_slots;
                    let s_final = &cache.s_states[s * m * d..(s + 1) * m * d];
                    context.memory[level] = s_final.to_vec();
                    (y, MemoryCache::Lattice(cache))
                }
                MemoryRuleKind::Trellis => {
                    let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
                    let (y, cache) = rule.step(&params.levels[level], &embedded, s, d, initial_m);
                    let d_k = cfg.d_compress;
                    let sk_size = d_k * d;
                    let sv_size = d * d_k;
                    let sk_final = &cache.sk_states[s * sk_size..(s + 1) * sk_size];
                    let sv_final = &cache.sv_states[s * sv_size..(s + 1) * sv_size];
                    let mut ctx_mem = Vec::with_capacity(sk_size + sv_size);
                    ctx_mem.extend_from_slice(sk_final);
                    ctx_mem.extend_from_slice(sv_final);
                    context.memory[level] = ctx_mem;
                    (y, MemoryCache::Trellis(cache))
                }
            };

            y_per_level.push(y_level);
            memory_caches.push(Some(mem_cache));
            q_mem_per_level.push(None);
            frozen_memories.push(None);
        } else {
            // Frozen level: read-only from persisted M (rule-aware dispatch).
            // Borrow context memory for the forward call, then clone for cache storage
            // (backward needs owned copy since context may be mutated between calls).
            let frozen_ref = &context.memory[level];
            let (y_level, q_mem) = match cfg.memory_rule {
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
                _ => delta_rule_read_only(
                    &params.levels[level], &embedded, frozen_ref, s, d,
                ),
            };
            y_per_level.push(y_level);
            memory_caches.push(None);
            q_mem_per_level.push(Some(q_mem));
            frozen_memories.push(Some(frozen_ref.clone()));
        }
    }

    // Combine: y_combined = sum of level outputs, with 1/sqrt(k) normalization for k>2.
    //
    // At k=2, additive composition works fine (signal doubles, sigmoid handles it).
    // At k=4+, additive sum grows linearly with k, pushing sigmoid into saturation
    // where gradients vanish. 1/sqrt(k) normalization keeps signal variance constant
    // (analogous to attention's 1/sqrt(d) scaling) while preserving gradient magnitude.
    //
    // Why not normalize k=2: the 1/sqrt(k) factor also scales the backward gradient
    // to all memory parameters, slowing outer-loop learning of gate biases (b_theta,
    // b_alpha). At k=2, this cost outweighs the benefit since the signal isn't large
    // enough to cause saturation.
    let mut y_combined = vec![0.0f32; s * d];
    for y_level in &y_per_level {
        for i in 0..(s * d) {
            y_combined[i] += y_level[i];
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            y_combined[i] *= scale;
        }
    }

    // Stage 4: Gating — gate = sigmoid(y_combined), gated_out = attn_out * gate
    let mut gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        gate[i] = sigmoid_f32(y_combined[i]);
        gated_out[i] = attn_out[i] * gate[i];
    }

    // Stage 5: Output projection
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&gated_out, &w_o_t, &mut projected, s, d, d);

    // Stage 6: Unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 7: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = CMSForwardCache {
        embedded, q, k, v: vv, attn_out, attn_weights,
        memory_caches, q_mem_per_level, frozen_memories,
        y_per_level, y_combined,
        gate, gated_out, projected, logits,
        pulse: pulse.clone(),
    };

    (loss, cache)
}

/// CMS backward pass. Returns parameter gradients.
///
/// Active levels get full backward through DeltaRule.
/// Frozen levels get read-only backward; their gradients accumulate in error_buffers.
pub fn cms_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &CMSForwardCache,
    input_ids: &[usize],
    target_ids: &[usize],
    error_buffers: &mut [ErrorBuffer],
) -> MAGParams {
    let swa_cfg = &cfg.swa;
    let s = swa_cfg.seq_len;
    let d = swa_cfg.d_model;
    let v = swa_cfg.vocab_size;
    let nh = swa_cfg.num_heads;
    let hd = swa_cfg.head_dim;
    let ws = swa_cfg.window_size;

    let mut grads = MAGParams::zeros_like(cfg);

    // ── Stage 7: Cross-entropy gradient ──────────────────────────────
    let mut d_logits = vec![0.0f32; s * v];
    let count = (0..s)
        .filter(|&t| target_ids.get(t).map_or(false, |&tok| tok < v))
        .count() as f32;
    if count > 0.0 {
        for t in 0..s {
            let target = match target_ids.get(t) {
                Some(&tok) if tok < v => tok,
                _ => continue,
            };
            let base = t * v;
            let row = &cache.logits[base..base + v];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for j in 0..v {
                let e = (row[j] - max_val).exp();
                d_logits[base + j] = e;
                sum_exp += e;
            }
            for j in 0..v {
                d_logits[base + j] /= sum_exp;
            }
            d_logits[base + target] -= 1.0;
            for j in 0..v {
                d_logits[base + j] /= count;
            }
        }
    }

    // ── Stage 6: Unembed backward ────────────────────────────────────
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.swa.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    matmul_f32(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 5: Output projection backward ──────────────────────────
    let mut d_gated_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 4: Gating backward ─────────────────────────────────────
    let mut d_attn_out = vec![0.0f32; s * d];
    let mut d_gate = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_attn_out[i] = d_gated_out[i] * cache.gate[i];
        d_gate[i] = d_gated_out[i] * cache.attn_out[i];
    }

    // gate = sigmoid(y_combined) → d_y_combined = d_gate * gate * (1 - gate)
    let mut d_y_combined = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y_combined[i] = d_gate[i] * cache.gate[i] * (1.0 - cache.gate[i]);
    }

    // Chain rule for 1/sqrt(k) normalization: d_y_level = (1/sqrt(k)) * d_y_combined
    // Scale d_y_combined once before distributing to per-level backward passes.
    // Only applies for k>2 (matching forward normalization).
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            d_y_combined[i] *= scale;
        }
    }

    // ── Stage 3b: Per-level memory backward ──────────────────────────
    // d_y_combined (now scaled by 1/sqrt(k) for k>2) distributes to each level
    let mut d_embedded_mem_total = vec![0.0f32; s * d];

    for level in 0..cfg.k {
        if cache.pulse.active_levels[level] {
            // Active level: dispatch backward based on cache variant
            let mem_cache = cache.memory_caches[level].as_ref().unwrap();
            let (mem_grads, d_embedded_mem) = match mem_cache {
                MemoryCache::Delta(delta_cache) => {
                    DeltaRule.step_backward(&params.levels[level], delta_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Titans(titans_cache) => {
                    TitansLMM.step_backward(&params.levels[level], titans_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Hebbian(hebbian_cache) => {
                    HebbianRule.step_backward(&params.levels[level], hebbian_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Moneta(moneta_cache) => {
                    let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
                    rule.step_backward(&params.levels[level], moneta_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::YAAD(yaad_cache) => {
                    let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
                    rule.step_backward(&params.levels[level], yaad_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::MEMORA(memora_cache) => {
                    let rule = MEMORA { d_hidden: cfg.d_hidden };
                    rule.step_backward(&params.levels[level], memora_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Lattice(lattice_cache) => {
                    let rule = LatticeOSR { m_slots: cfg.m_slots };
                    rule.step_backward(&params.levels[level], lattice_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Trellis(trellis_cache) => {
                    let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
                    rule.step_backward(&params.levels[level], trellis_cache, &d_y_combined, &cache.embedded)
                }
            };
            grads.levels[level].accumulate(&mem_grads);
            for i in 0..(s * d) {
                d_embedded_mem_total[i] += d_embedded_mem[i];
            }
        } else {
            // Frozen level: read-only backward (rule-aware dispatch)
            let q_mem = cache.q_mem_per_level[level].as_ref().unwrap();
            let frozen_m = cache.frozen_memories[level].as_ref().unwrap();
            let (mem_grads, d_embedded_mem) = match cfg.memory_rule {
                MemoryRuleKind::Moneta => moneta_read_only_backward(
                    &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::YAAD => yaad_read_only_backward(
                    &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::MEMORA => memora_read_only_backward(
                    &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d, cfg.d_hidden,
                ),
                MemoryRuleKind::LatticeOSR => lattice_read_only_backward(
                    &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d, cfg.m_slots,
                ),
                MemoryRuleKind::Trellis => trellis_read_only_backward(
                    &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d, cfg.d_compress,
                ),
                _ => delta_rule_read_only_backward(
                    &params.levels[level],
                    frozen_m,
                    q_mem,
                    &d_y_combined,
                    &cache.embedded,
                    s,
                    d,
                ),
            };
            // Frozen level grads go to error buffer, not direct grads
            error_buffers[level].accumulate(&mem_grads);
            for i in 0..(s * d) {
                d_embedded_mem_total[i] += d_embedded_mem[i];
            }
        }
    }

    // ── Stage 3a: SWA Attention backward ─────────────────────────────
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 2a: QKV projection backward ────────────────────────────
    let mut d_embedded = vec![0.0f32; s * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_embedded, s, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_embedded, s, d, d);

    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    matmul_f32(&d_q_t, &cache.embedded, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    matmul_f32(&d_k_t, &cache.embedded, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    matmul_f32(&d_v_t, &cache.embedded, &mut grads.swa.w_v, d, s, d);

    // ── Combine d_embedded from both branches ────────────────────────
    for i in 0..(s * d) {
        d_embedded[i] += d_embedded_mem_total[i];
    }

    // ── Stage 1: Embedding scatter-add ───────────────────────────────
    for t in 0..s {
        let tok = input_ids[t];
        for dd in 0..d {
            grads.swa.w_embed[tok * d + dd] += d_embedded[t * d + dd];
        }
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};

    fn test_config() -> MAGConfig {
        MAGConfig::test_config()
    }

    fn make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_mag_forward_finite_loss() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss, _cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "MAG loss not finite: {loss}");
        assert!(loss > 0.0, "MAG loss should be positive: {loss}");
        assert!(loss < 20.0, "MAG loss too high: {loss}");
    }

    #[test]
    fn test_mag_forward_gate_values() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        for (i, &g) in cache.gate.iter().enumerate() {
            assert!(g > 0.0 && g < 1.0, "gate[{i}]={g} not in (0,1)");
        }
    }

    #[test]
    fn test_mag_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss1, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        let (loss2, _) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(loss1, loss2, "MAG forward should be deterministic");
    }

    #[test]
    fn test_mag_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        for (name, g) in [
            ("w_q", &grads.swa.w_q), ("w_k", &grads.swa.w_k),
            ("w_v", &grads.swa.w_v), ("w_o", &grads.swa.w_o),
            ("w_unembed", &grads.swa.w_unembed), ("w_embed", &grads.swa.w_embed),
            ("w_k_mem", &grads.levels[0].w_k_mem), ("w_v_mem", &grads.levels[0].w_v_mem),
            ("w_q_mem", &grads.levels[0].w_q_mem), ("w_alpha", &grads.levels[0].w_alpha),
            ("b_alpha", &grads.levels[0].b_alpha), ("w_theta", &grads.levels[0].w_theta),
            ("b_theta", &grads.levels[0].b_theta),
        ] {
            for (i, &val) in g.iter().enumerate() {
                assert!(val.is_finite(), "mag grad_{name}[{i}] not finite: {val}");
            }
        }
    }

    #[test]
    fn test_mag_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        for (name, g) in [
            ("w_q", &grads.swa.w_q), ("w_o", &grads.swa.w_o),
            ("w_k_mem", &grads.levels[0].w_k_mem), ("w_v_mem", &grads.levels[0].w_v_mem),
            ("w_q_mem", &grads.levels[0].w_q_mem),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "mag grad_{name} all zeros (max_abs={max_abs})");
        }
    }

    #[test]
    fn test_mag_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mag_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mag_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        let d = cfg.swa.d_model;
        assert_eq!(grads.levels[0].w_k_mem.len(), d * d);
        assert_eq!(grads.levels[0].w_alpha.len(), 2 * d);
        assert_eq!(grads.levels[0].b_alpha.len(), 1);
    }

    // ── CMS tests ───────────────────────────────────────────────────

    fn test_config_k2() -> MAGConfig {
        MAGConfig::test_config_k2()
    }

    fn make_test_data_k2(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_cms_forward_finite_loss() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        // Step 0: both levels active
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        assert!(loss.is_finite(), "CMS loss not finite: {loss}");
        assert!(loss > 0.0, "CMS loss should be positive: {loss}");
    }

    #[test]
    fn test_cms_forward_level1_frozen_output() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);

        // Step 0: both active (initializes memory)
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (_, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut context);

        // Verify context.memory[1] is non-zero after active step
        let m1_norm: f32 = context.memory[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(m1_norm > 1e-8, "Level 1 memory should be non-zero after active step");

        // Step 1: Level 1 frozen, still contributes via read-only M
        let pulse1 = Pulse { global_step: 1, active_levels: vec![true, false] };
        let (loss, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse1, &mut context);
        assert!(loss.is_finite(), "CMS frozen loss not finite: {loss}");

        // Level 1 y should be non-zero (reading from non-zero M)
        let y1_norm: f32 = cache.y_per_level[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(y1_norm > 1e-8, "Frozen level 1 output should be non-zero");
    }

    #[test]
    fn test_cms_forward_deterministic() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);

        let mut ctx1 = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut ctx2 = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        let (loss1, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx1);
        let (loss2, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx2);
        assert_eq!(loss1, loss2, "CMS forward should be deterministic");
    }

    #[test]
    fn test_cms_backward_finite() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

        // All gradients should be finite
        for level in 0..cfg.k {
            for &val in grads.levels[level].w_k_mem.iter()
                .chain(grads.levels[level].w_v_mem.iter())
                .chain(grads.levels[level].w_q_mem.iter())
            {
                assert!(val.is_finite(), "CMS level {level} gradient not finite");
            }
        }
        for &val in grads.swa.w_q.iter().chain(grads.swa.w_o.iter()) {
            assert!(val.is_finite(), "CMS SWA gradient not finite");
        }
    }

    #[test]
    fn test_cms_backward_nonzero() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        // Both active so both have gradients
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

        // Both active levels should have non-zero memory gradients
        for level in 0..cfg.k {
            let norm: f32 = grads.levels[level].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm > 1e-10, "CMS level {level} w_k_mem grads all zeros");
        }
    }

    #[test]
    fn test_cms_backward_frozen_error_buffer() {
        let cfg = test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);

        // Step 0: both active (initialize memory)
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (_, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut context);

        // Step 1: Level 1 frozen
        let pulse1 = Pulse { global_step: 1, active_levels: vec![true, false] };
        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse1, &mut context);

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers);

        // Level 0 (active) should have direct grads
        let l0_norm: f32 = grads.levels[0].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l0_norm > 1e-10, "Active level 0 should have non-zero grads");

        // Level 1 (frozen) should have grads in error buffer, not in direct grads
        let l1_direct_norm: f32 = grads.levels[1].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l1_direct_norm < 1e-12, "Frozen level 1 should have zero direct grads, got {l1_direct_norm}");

        // Error buffer should have accumulated the frozen level's grads
        assert_eq!(error_buffers[1].steps_accumulated, 1);
        let eb_norm = error_buffers[1].grads.norm();
        assert!(eb_norm > 1e-10, "Error buffer should have non-zero grads for frozen level");
    }
}
