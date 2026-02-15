/// MAL (Memory As Layer) composition.
///
/// Architecture:
///   embed → memory.step(embedded) → m_t → QKV(m_t) → SWA(m_t) → output proj → unembed → loss
///
/// Simplest composition: memory preprocesses input, attention processes memory output.
/// No sigmoid gate — m_t IS the attention input. Information bottleneck by design.

use crate::tensor::{matmul_f32, transpose_f32, cross_entropy_loss};
use crate::model::{MAGConfig, MAGParams, MemoryRuleKind};
use crate::delta_rule::{MemoryRule, DeltaRule, delta_rule_read_only, delta_rule_read_only_backward};
use crate::titans_lmm::TitansLMM;
use crate::hebbian_rule::HebbianRule;
use crate::moneta::{Moneta, moneta_read_only, moneta_read_only_backward};
use crate::yaad::{YAAD, yaad_read_only, yaad_read_only_backward};
use crate::memora::{MEMORA, memora_read_only, memora_read_only_backward};
use crate::lattice_osr::{LatticeOSR, lattice_read_only, lattice_read_only_backward};
use crate::trellis::{Trellis, trellis_read_only, trellis_read_only_backward};
use crate::conductor::{Pulse, ContextState, ErrorBuffer};
use crate::mag::MemoryCache;

/// Cache for MAL forward pass.
pub struct MALForwardCache {
    pub embedded: Vec<f32>,
    pub m_t: Vec<f32>,          // memory output: [seq_len, d]
    pub attn_input: Vec<f32>,   // m_t + embedded (residual for bootstrapping)
    pub memory_cache: MemoryCache,
    pub q: Vec<f32>,            // computed from attn_input
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub projected: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Dispatch memory step for MAL. Returns (y, MemoryCache).
fn dispatch_memory_step(
    cfg: &MAGConfig,
    level_params: &crate::model::MemoryLevelParams,
    embedded: &[f32],
    s: usize,
    d: usize,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, MemoryCache) {
    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let (y, cache) = DeltaRule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Delta(cache))
        }
        MemoryRuleKind::TitansLMM => {
            let (y, cache) = TitansLMM.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Titans(cache))
        }
        MemoryRuleKind::HebbianRule => {
            let (y, cache) = HebbianRule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Hebbian(cache))
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Moneta(cache))
        }
        MemoryRuleKind::YAAD => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::YAAD(cache))
        }
        MemoryRuleKind::MEMORA => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::MEMORA(cache))
        }
        MemoryRuleKind::LatticeOSR => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Lattice(cache))
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            let (y, cache) = rule.step(level_params, embedded, s, d, initial_m);
            (y, MemoryCache::Trellis(cache))
        }
    }
}

/// Dispatch memory step_backward. Returns (level param grads, d_embedded).
fn dispatch_memory_backward(
    cfg: &MAGConfig,
    level_params: &crate::model::MemoryLevelParams,
    cache: &MemoryCache,
    d_y: &[f32],
    embedded: &[f32],
) -> (crate::model::MemoryLevelParams, Vec<f32>) {
    match cache {
        MemoryCache::Delta(c) => DeltaRule.step_backward(level_params, c, d_y, embedded),
        MemoryCache::Titans(c) => TitansLMM.step_backward(level_params, c, d_y, embedded),
        MemoryCache::Hebbian(c) => HebbianRule.step_backward(level_params, c, d_y, embedded),
        MemoryCache::Moneta(c) => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            rule.step_backward(level_params, c, d_y, embedded)
        }
        MemoryCache::YAAD(c) => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            rule.step_backward(level_params, c, d_y, embedded)
        }
        MemoryCache::MEMORA(c) => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            rule.step_backward(level_params, c, d_y, embedded)
        }
        MemoryCache::Lattice(c) => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            rule.step_backward(level_params, c, d_y, embedded)
        }
        MemoryCache::Trellis(c) => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            rule.step_backward(level_params, c, d_y, embedded)
        }
    }
}

/// MAL forward pass. Returns (loss, cache).
pub fn mal_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MALForwardCache) {
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
        assert!(tok < v, "mal_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Stage 2: Memory step on embedded → m_t
    let (m_t, memory_cache) = dispatch_memory_step(cfg, &params.levels[0], &embedded, s, d, None);

    // Stage 2.5: Residual — attn_input = m_t + embedded
    // This breaks the bootstrapping deadlock: at init m_t ≈ 0, so attention sees embedded.
    // As memory learns, m_t adds useful context to the raw input.
    let mut attn_input = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        attn_input[i] = m_t[i] + embedded[i];
    }

    // Stage 3: QKV projections on attn_input (m_t + embedded)
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    matmul_f32(&attn_input, &w_q_t, &mut q, s, d, d);
    matmul_f32(&attn_input, &w_k_t, &mut k, s, d, d);
    matmul_f32(&attn_input, &w_v_t, &mut vv, s, d, d);

    // Stage 4: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 5: Output projection
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&attn_out, &w_o_t, &mut projected, s, d, d);

    // Stage 6: Unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 7: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = MALForwardCache {
        embedded, m_t, attn_input, memory_cache,
        q, k, v: vv, attn_out, attn_weights,
        projected, logits,
    };

    (loss, cache)
}

/// MAL backward pass. Returns parameter gradients.
pub fn mal_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &MALForwardCache,
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
    let mut d_attn_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_attn_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.attn_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 4: SWA Attention backward ─────────────────────────────
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 3: QKV projection backward (inputs are attn_input = m_t + embedded) ─
    let mut d_attn_input = vec![0.0f32; s * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_attn_input, s, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_attn_input, s, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_attn_input, s, d, d);

    // Weight gradients: d_W = d_QKV^T @ attn_input
    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    matmul_f32(&d_q_t, &cache.attn_input, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    matmul_f32(&d_k_t, &cache.attn_input, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    matmul_f32(&d_v_t, &cache.attn_input, &mut grads.swa.w_v, d, s, d);

    // ── Stage 2.5: Residual backward (attn_input = m_t + embedded) ──
    // d_m_t = d_attn_input (gradient flows to memory)
    // d_embedded_res = d_attn_input (gradient flows to embedding via residual)
    let d_m_t = d_attn_input.clone();

    // ── Stage 2: Memory backward ────────────────────────────────────
    let (mem_grads, d_embedded_mem) = dispatch_memory_backward(
        cfg, &params.levels[0], &cache.memory_cache, &d_m_t, &cache.embedded,
    );
    grads.levels[0].accumulate(&mem_grads);

    // ── Stage 1: Embedding scatter-add (residual + memory paths) ────
    for t in 0..s {
        let tok = input_ids[t];
        for dd in 0..d {
            // Residual path: d_attn_input flows directly to embedding
            // Memory path: d_embedded_mem from memory backward
            grads.swa.w_embed[tok * d + dd] += d_attn_input[t * d + dd] + d_embedded_mem[t * d + dd];
        }
    }

    grads
}

// ── CMS MAL ─────────────────────────────────────────────────────────

/// Cache for CMS MAL forward pass.
pub struct CMSMALForwardCache {
    pub embedded: Vec<f32>,
    pub memory_caches: Vec<Option<MemoryCache>>,
    pub q_mem_per_level: Vec<Option<Vec<f32>>>,
    pub frozen_memories: Vec<Option<Vec<f32>>>,
    pub y_per_level: Vec<Vec<f32>>,
    pub m_t_combined: Vec<f32>,
    pub attn_input: Vec<f32>,   // m_t_combined + embedded (residual)
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub projected: Vec<f32>,
    pub logits: Vec<f32>,
    pub pulse: Pulse,
}

/// Extract final memory state from cache and persist to context.
pub fn persist_memory_state(
    _cfg: &MAGConfig,
    cache: &MemoryCache,
    s: usize,
    d: usize,
    context_mem: &mut Vec<f32>,
) {
    match cache {
        MemoryCache::Delta(c) => {
            let start = s * d * d;
            *context_mem = c.m_states[start..start + d * d].to_vec();
        }
        MemoryCache::Titans(c) => {
            let start = s * d * d;
            *context_mem = c.m_states[start..start + d * d].to_vec();
        }
        MemoryCache::Hebbian(c) => {
            let start = s * d * d;
            *context_mem = c.m_states[start..start + d * d].to_vec();
        }
        MemoryCache::Moneta(c) => {
            let dh = c.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &c.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &c.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
            ctx_mem.extend_from_slice(w1_final);
            ctx_mem.extend_from_slice(w2_final);
            *context_mem = ctx_mem;
        }
        MemoryCache::YAAD(c) => {
            let dh = c.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &c.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &c.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
            ctx_mem.extend_from_slice(w1_final);
            ctx_mem.extend_from_slice(w2_final);
            *context_mem = ctx_mem;
        }
        MemoryCache::MEMORA(c) => {
            let dh = c.d_hidden;
            let w1_size = dh * d;
            let w2_size = d * dh;
            let w1_final = &c.w1_states[s * w1_size..(s + 1) * w1_size];
            let w2_final = &c.w2_states[s * w2_size..(s + 1) * w2_size];
            let mut ctx_mem = Vec::with_capacity(w1_size + w2_size);
            ctx_mem.extend_from_slice(w1_final);
            ctx_mem.extend_from_slice(w2_final);
            *context_mem = ctx_mem;
        }
        MemoryCache::Lattice(c) => {
            let m = c.m;
            let s_final = &c.s_states[s * m * d..(s + 1) * m * d];
            *context_mem = s_final.to_vec();
        }
        MemoryCache::Trellis(c) => {
            let d_k = c.d_k;
            let sk_size = d_k * d;
            let sv_size = d * d_k;
            let sk_final = &c.sk_states[s * sk_size..(s + 1) * sk_size];
            let sv_final = &c.sv_states[s * sv_size..(s + 1) * sv_size];
            let mut ctx_mem = Vec::with_capacity(sk_size + sv_size);
            ctx_mem.extend_from_slice(sk_final);
            ctx_mem.extend_from_slice(sv_final);
            *context_mem = ctx_mem;
        }
    }
}

/// Dispatch frozen read-only forward. Returns (y, q_mem).
fn dispatch_read_only(
    cfg: &MAGConfig,
    level_params: &crate::model::MemoryLevelParams,
    embedded: &[f32],
    frozen_ref: &[f32],
    s: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta => moneta_read_only(level_params, embedded, frozen_ref, s, d, cfg.d_hidden),
        MemoryRuleKind::YAAD => yaad_read_only(level_params, embedded, frozen_ref, s, d, cfg.d_hidden),
        MemoryRuleKind::MEMORA => memora_read_only(level_params, embedded, frozen_ref, s, d, cfg.d_hidden),
        MemoryRuleKind::LatticeOSR => lattice_read_only(level_params, embedded, frozen_ref, s, d, cfg.m_slots),
        MemoryRuleKind::Trellis => trellis_read_only(level_params, embedded, frozen_ref, s, d, cfg.d_compress),
        _ => delta_rule_read_only(level_params, embedded, frozen_ref, s, d),
    }
}

/// Dispatch frozen read-only backward. Returns (level grads, d_embedded).
fn dispatch_read_only_backward(
    cfg: &MAGConfig,
    level_params: &crate::model::MemoryLevelParams,
    frozen_m: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    s: usize,
    d: usize,
) -> (crate::model::MemoryLevelParams, Vec<f32>) {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta => moneta_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::YAAD => yaad_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::MEMORA => memora_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::LatticeOSR => lattice_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.m_slots),
        MemoryRuleKind::Trellis => trellis_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_compress),
        _ => delta_rule_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d),
    }
}

/// CMS MAL forward pass. Returns (loss, cache).
pub fn cms_mal_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
) -> (f32, CMSMALForwardCache) {
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
        assert!(tok < v, "cms_mal_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Stage 2: Per-level memory dispatch
    let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut y_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            let initial_m = Some(std::mem::take(&mut context.memory[level]));
            let (y_level, mem_cache) = dispatch_memory_step(
                cfg, &params.levels[level], &embedded, s, d, initial_m,
            );
            persist_memory_state(cfg, &mem_cache, s, d, &mut context.memory[level]);
            y_per_level.push(y_level);
            memory_caches.push(Some(mem_cache));
            q_mem_per_level.push(None);
            frozen_memories.push(None);
        } else {
            let frozen_ref = &context.memory[level];
            let (y_level, q_mem) = dispatch_read_only(
                cfg, &params.levels[level], &embedded, frozen_ref, s, d,
            );
            y_per_level.push(y_level);
            memory_caches.push(None);
            q_mem_per_level.push(Some(q_mem));
            frozen_memories.push(Some(frozen_ref.clone()));
        }
    }

    // Combine: m_t_combined = sum of level outputs, with 1/sqrt(k) for k>2
    let mut m_t_combined = vec![0.0f32; s * d];
    for y_level in &y_per_level {
        for i in 0..(s * d) {
            m_t_combined[i] += y_level[i];
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            m_t_combined[i] *= scale;
        }
    }

    // Stage 2.5: Residual — attn_input = m_t_combined + embedded
    let mut attn_input = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        attn_input[i] = m_t_combined[i] + embedded[i];
    }

    // Stage 3: QKV projections on attn_input
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    matmul_f32(&attn_input, &w_q_t, &mut q, s, d, d);
    matmul_f32(&attn_input, &w_k_t, &mut k, s, d, d);
    matmul_f32(&attn_input, &w_v_t, &mut vv, s, d, d);

    // Stage 4: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 5: Output projection
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&attn_out, &w_o_t, &mut projected, s, d, d);

    // Stage 6: Unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 7: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = CMSMALForwardCache {
        embedded, memory_caches, q_mem_per_level, frozen_memories,
        y_per_level, m_t_combined, attn_input,
        q, k, v: vv, attn_out, attn_weights,
        projected, logits,
        pulse: pulse.clone(),
    };

    (loss, cache)
}

/// CMS MAL backward pass. Returns parameter gradients.
pub fn cms_mal_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &CMSMALForwardCache,
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
    let mut d_attn_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_attn_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.attn_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 4: SWA Attention backward ─────────────────────────────
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 3: QKV projection backward (inputs are attn_input) ──
    let mut d_attn_input = vec![0.0f32; s * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_attn_input, s, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_attn_input, s, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_attn_input, s, d, d);

    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    matmul_f32(&d_q_t, &cache.attn_input, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    matmul_f32(&d_k_t, &cache.attn_input, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    matmul_f32(&d_v_t, &cache.attn_input, &mut grads.swa.w_v, d, s, d);

    // ── Stage 2.5: Residual backward (attn_input = m_t_combined + embedded) ──
    let mut d_m_t = d_attn_input.clone();

    // 1/sqrt(k) normalization chain rule for k>2
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            d_m_t[i] *= scale;
        }
    }

    // ── Stage 2: Per-level memory backward ──────────────────────────
    let mut d_embedded_total = vec![0.0f32; s * d];

    // Residual path: d_attn_input flows directly to embedding
    for i in 0..(s * d) {
        d_embedded_total[i] = d_attn_input[i];
    }

    for level in 0..cfg.k {
        if cache.pulse.active_levels[level] {
            let mem_cache = cache.memory_caches[level].as_ref().unwrap();
            let (mem_grads, d_embedded_mem) = dispatch_memory_backward(
                cfg, &params.levels[level], mem_cache, &d_m_t, &cache.embedded,
            );
            grads.levels[level].accumulate(&mem_grads);
            for i in 0..(s * d) {
                d_embedded_total[i] += d_embedded_mem[i];
            }
        } else {
            let q_mem = cache.q_mem_per_level[level].as_ref().unwrap();
            let frozen_m = cache.frozen_memories[level].as_ref().unwrap();
            let (mem_grads, d_embedded_mem) = dispatch_read_only_backward(
                cfg, &params.levels[level], frozen_m, q_mem, &d_m_t, &cache.embedded, s, d,
            );
            error_buffers[level].accumulate(&mem_grads);
            for i in 0..(s * d) {
                d_embedded_total[i] += d_embedded_mem[i];
            }
        }
    }

    // ── Stage 1: Embedding scatter-add ───────────────────────────────
    for t in 0..s {
        let tok = input_ids[t];
        for dd in 0..d {
            grads.swa.w_embed[tok * d + dd] += d_embedded_total[t * d + dd];
        }
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};

    fn test_config() -> MAGConfig {
        MAGConfig::mal_test_config()
    }

    fn make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_mal_forward_finite_loss() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss, _cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "MAL loss not finite: {loss}");
        assert!(loss > 0.0, "MAL loss should be positive: {loss}");
        assert!(loss < 20.0, "MAL loss too high: {loss}");
    }

    #[test]
    fn test_mal_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss1, _) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        let (loss2, _) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(loss1, loss2, "MAL forward should be deterministic");
    }

    #[test]
    fn test_mal_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mal_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        for (name, g) in [
            ("w_q", &grads.swa.w_q), ("w_k", &grads.swa.w_k),
            ("w_v", &grads.swa.w_v), ("w_o", &grads.swa.w_o),
            ("w_unembed", &grads.swa.w_unembed), ("w_embed", &grads.swa.w_embed),
            ("w_k_mem", &grads.levels[0].w_k_mem),
        ] {
            for (i, &val) in g.iter().enumerate() {
                assert!(val.is_finite(), "mal grad_{name}[{i}] not finite: {val}");
            }
        }
    }

    #[test]
    fn test_mal_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mal_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        // w_o always has gradient (directly connects to loss).
        // w_q gradient can be tiny at init because Q = w_q @ m_t and m_t ≈ 0
        // when memory starts empty and theta_t ≈ 0.01.
        for (name, g) in [
            ("w_o", &grads.swa.w_o),
            ("w_k_mem", &grads.levels[0].w_k_mem),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "mal grad_{name} all zeros (max_abs={max_abs})");
        }
    }

    #[test]
    fn test_mal_m_t_is_attention_input() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mal_forward(&params, &cfg, &input_ids, &target_ids);

        // Verify m_t ≠ embedded (memory actually transforms the input)
        let diff: f32 = cache.m_t.iter().zip(cache.embedded.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "m_t should differ from embedded, diff={diff}");
    }
}
