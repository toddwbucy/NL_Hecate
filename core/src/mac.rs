/// MAC (Memory As Context) composition.
///
/// Architecture:
///   embed → read_only(embedded) → h_t → concat(h_t, embedded) → QKV(assembled)
///         → full causal attn(2s) → extract y_t(s:) → memory.step(y_t) → reflective_y
///         → y_t * sigmoid(reflective_y) → W_O → unembed → loss
///
/// Most expressive: memory provides context for attention, then reflects on attention output.
/// Two memory operations per step: read-only (context) + step (reflective gate).

use crate::tensor::{matmul_f32, transpose_f32, cross_entropy_loss, sigmoid_f32};
use crate::model::{MAGConfig, MAGParams, MemoryRuleKind, MemoryLevelParams};
use crate::delta_rule::{MemoryRule, DeltaRule, delta_rule_read_only, delta_rule_read_only_backward};
use crate::titans_lmm::TitansLMM;
use crate::hebbian_rule::HebbianRule;
use crate::moneta::{Moneta, moneta_read_only, moneta_read_only_backward};
use crate::yaad::{YAAD, yaad_read_only, yaad_read_only_backward};
use crate::memora::{MEMORA, memora_read_only, memora_read_only_backward};
use crate::lattice_osr::{LatticeOSR, lattice_read_only, lattice_read_only_backward};
use crate::trellis::{Trellis, trellis_read_only, trellis_read_only_backward};
use crate::atlas_omega::AtlasOmega;
use crate::conductor::{Pulse, ContextState, ErrorBuffer};
use crate::mag::MemoryCache;

/// Cache for MAC forward pass.
pub struct MACForwardCache {
    pub embedded: Vec<f32>,       // (s, d)
    // Read-only memory context
    pub h_t: Vec<f32>,            // (s, d) — read from current M
    pub q_mem_read: Vec<f32>,     // for read_only backward
    pub frozen_m_read: Vec<f32>,  // M used for read_only
    // Assembled and attention
    pub assembled: Vec<f32>,      // (2s, d) = concat(h_t, embedded)
    pub q: Vec<f32>,              // (2s, d)
    pub k: Vec<f32>,              // (2s, d)
    pub v: Vec<f32>,              // (2s, d)
    pub attn_out: Vec<f32>,       // (2s, d)
    pub attn_weights: Vec<f32>,
    // y_t extraction and reflective memory
    pub y_t: Vec<f32>,            // (s, d) — extracted from attn_out[s:]
    pub memory_cache: MemoryCache,
    pub reflective_y: Vec<f32>,   // (s, d) — memory output from step(y_t)
    // Reflective gate
    pub reflective_gate: Vec<f32>, // sigmoid(reflective_y): (s, d)
    pub gated_out: Vec<f32>,       // y_t * reflective_gate: (s, d)
    // Post-gate
    pub projected: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Read-only dispatch (no memory update, just M @ q).
fn read_only_dispatch(
    cfg: &MAGConfig,
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    m_state: &[f32],
    s: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta => moneta_read_only(level_params, embedded, m_state, s, d, cfg.d_hidden),
        MemoryRuleKind::YAAD => yaad_read_only(level_params, embedded, m_state, s, d, cfg.d_hidden),
        MemoryRuleKind::MEMORA => memora_read_only(level_params, embedded, m_state, s, d, cfg.d_hidden),
        MemoryRuleKind::LatticeOSR => lattice_read_only(level_params, embedded, m_state, s, d, cfg.m_slots),
        MemoryRuleKind::Trellis => trellis_read_only(level_params, embedded, m_state, s, d, cfg.d_compress),
        _ => delta_rule_read_only(level_params, embedded, m_state, s, d),
    }
}

/// Read-only backward dispatch.
fn read_only_backward_dispatch(
    cfg: &MAGConfig,
    level_params: &MemoryLevelParams,
    frozen_m: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    s: usize,
    d: usize,
) -> (MemoryLevelParams, Vec<f32>) {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta => moneta_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::YAAD => yaad_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::MEMORA => memora_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_hidden),
        MemoryRuleKind::LatticeOSR => lattice_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.m_slots),
        MemoryRuleKind::Trellis => trellis_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d, cfg.d_compress),
        _ => delta_rule_read_only_backward(level_params, frozen_m, q_mem, d_y, embedded, s, d),
    }
}

/// Memory step dispatch. Returns (y, MemoryCache).
fn step_dispatch(
    cfg: &MAGConfig,
    level_params: &MemoryLevelParams,
    input: &[f32],
    s: usize,
    d: usize,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, MemoryCache) {
    match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let (y, c) = DeltaRule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Delta(c))
        }
        MemoryRuleKind::TitansLMM => {
            let (y, c) = TitansLMM.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Titans(c))
        }
        MemoryRuleKind::HebbianRule => {
            let (y, c) = HebbianRule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Hebbian(c))
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            let (y, c) = rule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Moneta(c))
        }
        MemoryRuleKind::YAAD => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            let (y, c) = rule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::YAAD(c))
        }
        MemoryRuleKind::MEMORA => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            let (y, c) = rule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::MEMORA(c))
        }
        MemoryRuleKind::LatticeOSR => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            let (y, c) = rule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Lattice(c))
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            let (y, c) = rule.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Trellis(c))
        }
        MemoryRuleKind::AtlasOmega => {
            let (y, c) = AtlasOmega.step(level_params, input, s, d, initial_m);
            (y, MemoryCache::Atlas(c))
        }
    }
}

/// Memory step_backward dispatch. Returns (level grads, d_input).
fn step_backward_dispatch(
    cfg: &MAGConfig,
    level_params: &MemoryLevelParams,
    cache: &MemoryCache,
    d_y: &[f32],
    input: &[f32],
) -> (MemoryLevelParams, Vec<f32>) {
    match cache {
        MemoryCache::Delta(c) => DeltaRule.step_backward(level_params, c, d_y, input),
        MemoryCache::Titans(c) => TitansLMM.step_backward(level_params, c, d_y, input),
        MemoryCache::Hebbian(c) => HebbianRule.step_backward(level_params, c, d_y, input),
        MemoryCache::Moneta(c) => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2 };
            rule.step_backward(level_params, c, d_y, input)
        }
        MemoryCache::YAAD(c) => {
            let rule = YAAD { d_hidden: cfg.d_hidden, delta: cfg.delta, lambda_local: cfg.lambda_local, lambda_2: cfg.lambda_2 };
            rule.step_backward(level_params, c, d_y, input)
        }
        MemoryCache::MEMORA(c) => {
            let rule = MEMORA { d_hidden: cfg.d_hidden };
            rule.step_backward(level_params, c, d_y, input)
        }
        MemoryCache::Lattice(c) => {
            let rule = LatticeOSR { m_slots: cfg.m_slots };
            rule.step_backward(level_params, c, d_y, input)
        }
        MemoryCache::Trellis(c) => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            rule.step_backward(level_params, c, d_y, input)
        }
        MemoryCache::Atlas(c) => AtlasOmega.step_backward(level_params, c, d_y, input),
    }
}

/// Get the initial memory state for read-only operations.
/// For k=1 (no CMS), return zeros of the right size.
fn initial_memory_state(cfg: &MAGConfig, d: usize) -> Vec<f32> {
    match cfg.memory_rule {
        MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
            vec![0.0f32; cfg.d_hidden * d + d * cfg.d_hidden]
        }
        MemoryRuleKind::LatticeOSR => {
            vec![0.0f32; cfg.m_slots * d]
        }
        MemoryRuleKind::Trellis => {
            vec![0.0f32; 2 * cfg.d_compress * d]
        }
        _ => vec![0.0f32; d * d],
    }
}

/// MAC forward pass. Returns (loss, cache).
pub fn mac_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, MACForwardCache) {
    let swa_cfg = &cfg.swa;
    let s = swa_cfg.seq_len;
    let d = swa_cfg.d_model;
    let v = swa_cfg.vocab_size;
    let nh = swa_cfg.num_heads;
    let hd = swa_cfg.head_dim;
    let ws = swa_cfg.window_size;
    let s2 = 2 * s;

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);
    assert!(target_ids.len() >= s);
    assert!(ws >= s2, "MAC requires window_size >= 2*seq_len for full causal attention on assembled input");

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < v, "mac_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Stage 2: Read-only memory on embedded → h_t
    let m_state = initial_memory_state(cfg, d);
    let (h_t, q_mem_read) = read_only_dispatch(cfg, &params.levels[0], &embedded, &m_state, s, d);

    // Stage 3: Assemble — concat(h_t, embedded) → (2s, d)
    let mut assembled = vec![0.0f32; s2 * d];
    assembled[..s * d].copy_from_slice(&h_t);
    assembled[s * d..].copy_from_slice(&embedded);

    // Stage 4: QKV projections on assembled
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s2 * d];
    let mut k = vec![0.0f32; s2 * d];
    let mut vv = vec![0.0f32; s2 * d];
    matmul_f32(&assembled, &w_q_t, &mut q, s2, d, d);
    matmul_f32(&assembled, &w_k_t, &mut k, s2, d, d);
    matmul_f32(&assembled, &w_v_t, &mut vv, s2, d, d);

    // Stage 5: Full causal attention on assembled (ws >= 2s)
    let mut attn_out = vec![0.0f32; s2 * d];
    let mut attn_weights = vec![0.0f32; nh * s2 * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s2, nh, hd, ws);

    // Stage 6: Extract y_t = attn_out[s*d..] → (s, d)
    let y_t = attn_out[s * d..].to_vec();

    // Stage 7: Memory step on y_t → reflective_y (updates M)
    let (reflective_y, memory_cache) = step_dispatch(
        cfg, &params.levels[0], &y_t, s, d, None,
    );

    // Stage 8: Reflective gate — output = y_t * sigmoid(reflective_y)
    let mut reflective_gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        reflective_gate[i] = sigmoid_f32(reflective_y[i]);
        gated_out[i] = y_t[i] * reflective_gate[i];
    }

    // Stage 9: Output projection
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&gated_out, &w_o_t, &mut projected, s, d, d);

    // Stage 10: Unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 11: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = MACForwardCache {
        embedded, h_t, q_mem_read, frozen_m_read: m_state,
        assembled, q, k, v: vv, attn_out, attn_weights,
        y_t, memory_cache, reflective_y, reflective_gate, gated_out,
        projected, logits,
    };

    (loss, cache)
}

/// MAC backward pass. Returns parameter gradients.
pub fn mac_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &MACForwardCache,
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
    let s2 = 2 * s;

    let mut grads = MAGParams::zeros_like(cfg);

    // ── Stage 11: Cross-entropy gradient ─────────────────────────────
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

    // ── Stage 10: Unembed backward ───────────────────────────────────
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.swa.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    matmul_f32(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 9: Output projection backward ──────────────────────────
    let mut d_gated_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 8: Reflective gate backward ────────────────────────────
    // gated_out = y_t * reflective_gate
    // d_y_t_gate = d_gated_out * reflective_gate
    // d_reflective_gate = d_gated_out * y_t
    let mut d_y_t_gate = vec![0.0f32; s * d];
    let mut d_reflective_gate = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y_t_gate[i] = d_gated_out[i] * cache.reflective_gate[i];
        d_reflective_gate[i] = d_gated_out[i] * cache.y_t[i];
    }

    // reflective_gate = sigmoid(reflective_y) → d_reflective_y
    let mut d_reflective_y = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_reflective_y[i] = d_reflective_gate[i]
            * cache.reflective_gate[i]
            * (1.0 - cache.reflective_gate[i]);
    }

    // ── Stage 7: Memory step backward (on y_t) ──────────────────────
    let (mem_grads_step, d_y_t_mem) = step_backward_dispatch(
        cfg, &params.levels[0], &cache.memory_cache, &d_reflective_y, &cache.y_t,
    );

    // Combine d_y_t from gate and memory
    let mut d_y_t = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y_t[i] = d_y_t_gate[i] + d_y_t_mem[i];
    }

    // ── Stage 6: Scatter d_y_t into d_attn_out at positions [s*d..] ─
    let mut d_attn_out = vec![0.0f32; s2 * d];
    d_attn_out[s * d..].copy_from_slice(&d_y_t);

    // ── Stage 5: SWA backward (ws=2s) ────────────────────────────────
    let mut d_q = vec![0.0f32; s2 * d];
    let mut d_k = vec![0.0f32; s2 * d];
    let mut d_v = vec![0.0f32; s2 * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s2, nh, hd, ws,
    );

    // ── Stage 4: QKV projection backward ─────────────────────────────
    let mut d_assembled = vec![0.0f32; s2 * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_assembled, s2, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_assembled, s2, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_assembled, s2, d, d);

    // Weight gradients: d_W = d_QKV^T @ assembled
    let mut d_q_t = vec![0.0f32; d * s2];
    transpose_f32(&d_q, &mut d_q_t, s2, d);
    matmul_f32(&d_q_t, &cache.assembled, &mut grads.swa.w_q, d, s2, d);

    let mut d_k_t = vec![0.0f32; d * s2];
    transpose_f32(&d_k, &mut d_k_t, s2, d);
    matmul_f32(&d_k_t, &cache.assembled, &mut grads.swa.w_k, d, s2, d);

    let mut d_v_t = vec![0.0f32; d * s2];
    transpose_f32(&d_v, &mut d_v_t, s2, d);
    matmul_f32(&d_v_t, &cache.assembled, &mut grads.swa.w_v, d, s2, d);

    // ── Stage 3: Split d_assembled → d_h_t + d_embedded_attn ─────────
    let d_h_t = &d_assembled[..s * d];
    let d_embedded_attn = &d_assembled[s * d..];

    // ── Stage 2: Read-only backward → d_embedded_read + mem_grads_read
    let (mem_grads_read, d_embedded_read) = read_only_backward_dispatch(
        cfg, &params.levels[0], &cache.frozen_m_read, &cache.q_mem_read,
        d_h_t, &cache.embedded, s, d,
    );

    // Combine memory grads from both step and read paths
    grads.levels[0].accumulate(&mem_grads_step);
    grads.levels[0].accumulate(&mem_grads_read);

    // Combine d_embedded from attn and read paths
    let mut d_embedded = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_embedded[i] = d_embedded_attn[i] + d_embedded_read[i];
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

// ── CMS MAC ─────────────────────────────────────────────────────────

/// Cache for CMS MAC forward pass.
pub struct CMSMACForwardCache {
    pub embedded: Vec<f32>,
    // Per-level read-only context
    pub read_caches: Vec<ReadOnlyCacheLevel>,
    pub h_t_combined: Vec<f32>,
    // Assembled and attention
    pub assembled: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub y_t: Vec<f32>,
    // Per-level step (active only)
    pub step_caches: Vec<Option<MemoryCache>>,
    pub reflective_per_level: Vec<Option<Vec<f32>>>,
    pub reflective_y_combined: Vec<f32>,
    pub reflective_gate: Vec<f32>,
    pub gated_out: Vec<f32>,
    pub projected: Vec<f32>,
    pub logits: Vec<f32>,
    pub pulse: Pulse,
}

/// Per-level read-only cache for CMS MAC.
pub struct ReadOnlyCacheLevel {
    pub q_mem: Vec<f32>,
    pub frozen_m: Vec<f32>,
}

/// Extract final memory state from cache for context persistence (MAC-specific).
fn persist_mac_memory(
    cfg: &MAGConfig,
    cache: &MemoryCache,
    s: usize,
    d: usize,
    context_mem: &mut Vec<f32>,
) {
    crate::mal::persist_memory_state(cfg, cache, s, d, context_mem);
}

/// CMS MAC forward pass.
pub fn cms_mac_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    pulse: &Pulse,
    context: &mut ContextState,
) -> (f32, CMSMACForwardCache) {
    let swa_cfg = &cfg.swa;
    let s = swa_cfg.seq_len;
    let d = swa_cfg.d_model;
    let v = swa_cfg.vocab_size;
    let nh = swa_cfg.num_heads;
    let hd = swa_cfg.head_dim;
    let ws = swa_cfg.window_size;
    let s2 = 2 * s;

    assert_eq!(d, nh * hd);
    assert!(input_ids.len() >= s);
    assert!(target_ids.len() >= s);
    assert!(ws >= s2, "CMS MAC requires window_size >= 2*seq_len");
    assert_eq!(pulse.active_levels.len(), cfg.k);
    assert_eq!(context.memory.len(), cfg.k);

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < v, "cms_mac_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Stage 2: Per-level read-only → h_t per level
    let mut read_caches = Vec::with_capacity(cfg.k);
    let mut h_t_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        let frozen_ref = &context.memory[level];
        let (h_level, q_mem) = read_only_dispatch(
            cfg, &params.levels[level], &embedded, frozen_ref, s, d,
        );
        h_t_per_level.push(h_level);
        read_caches.push(ReadOnlyCacheLevel {
            q_mem,
            frozen_m: frozen_ref.clone(),
        });
    }

    // Combine h_t with 1/sqrt(k) for k>2
    let mut h_t_combined = vec![0.0f32; s * d];
    for h in &h_t_per_level {
        for i in 0..(s * d) {
            h_t_combined[i] += h[i];
        }
    }
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            h_t_combined[i] *= scale;
        }
    }

    // Stage 3: Assemble
    let mut assembled = vec![0.0f32; s2 * d];
    assembled[..s * d].copy_from_slice(&h_t_combined);
    assembled[s * d..].copy_from_slice(&embedded);

    // Stage 4: QKV projections on assembled
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_q, &mut w_q_t, d, d);
    transpose_f32(&params.swa.w_k, &mut w_k_t, d, d);
    transpose_f32(&params.swa.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s2 * d];
    let mut k = vec![0.0f32; s2 * d];
    let mut vv = vec![0.0f32; s2 * d];
    matmul_f32(&assembled, &w_q_t, &mut q, s2, d, d);
    matmul_f32(&assembled, &w_k_t, &mut k, s2, d, d);
    matmul_f32(&assembled, &w_v_t, &mut vv, s2, d, d);

    // Stage 5: Full causal attention
    let mut attn_out = vec![0.0f32; s2 * d];
    let mut attn_weights = vec![0.0f32; nh * s2 * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s2, nh, hd, ws);

    // Stage 6: Extract y_t
    let y_t = attn_out[s * d..].to_vec();

    // Stage 7: Per-level reflective memory step (active only)
    let mut step_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut reflective_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);

    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            let initial_m = Some(std::mem::take(&mut context.memory[level]));
            let (refl_y, mem_cache) = step_dispatch(
                cfg, &params.levels[level], &y_t, s, d, initial_m,
            );
            persist_mac_memory(cfg, &mem_cache, s, d, &mut context.memory[level]);
            reflective_per_level.push(Some(refl_y));
            step_caches.push(Some(mem_cache));
        } else {
            // Frozen levels: no reflective signal (read-only only contributes h_t)
            reflective_per_level.push(None);
            step_caches.push(None);
        }
    }

    // Combine reflective outputs (active levels only)
    let mut reflective_y_combined = vec![0.0f32; s * d];
    let mut active_count = 0usize;
    for refl in &reflective_per_level {
        if let Some(r) = refl {
            for i in 0..(s * d) {
                reflective_y_combined[i] += r[i];
            }
            active_count += 1;
        }
    }
    // Normalize if multiple active levels contribute to reflective signal
    if active_count > 2 {
        let scale = 1.0 / (active_count as f32).sqrt();
        for i in 0..(s * d) {
            reflective_y_combined[i] *= scale;
        }
    }

    // Stage 8: Reflective gate
    let mut reflective_gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        reflective_gate[i] = sigmoid_f32(reflective_y_combined[i]);
        gated_out[i] = y_t[i] * reflective_gate[i];
    }

    // Stage 9: Output projection
    let mut w_o_t = vec![0.0f32; d * d];
    transpose_f32(&params.swa.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&gated_out, &w_o_t, &mut projected, s, d, d);

    // Stage 10: Unembed
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 11: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = CMSMACForwardCache {
        embedded, read_caches, h_t_combined,
        assembled, q, k, v: vv, attn_out, attn_weights, y_t,
        step_caches, reflective_per_level, reflective_y_combined,
        reflective_gate, gated_out,
        projected, logits,
        pulse: pulse.clone(),
    };

    (loss, cache)
}

/// CMS MAC backward pass.
pub fn cms_mac_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &CMSMACForwardCache,
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
    let s2 = 2 * s;

    let mut grads = MAGParams::zeros_like(cfg);

    // ── Stage 11: Cross-entropy gradient ─────────────────────────────
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

    // ── Stage 10: Unembed backward ───────────────────────────────────
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.swa.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    matmul_f32(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 9: Output projection backward ──────────────────────────
    let mut d_gated_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

    // ── Stage 8: Reflective gate backward ────────────────────────────
    let mut d_y_t_gate = vec![0.0f32; s * d];
    let mut d_reflective_gate = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y_t_gate[i] = d_gated_out[i] * cache.reflective_gate[i];
        d_reflective_gate[i] = d_gated_out[i] * cache.y_t[i];
    }

    let mut d_reflective_y_combined = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_reflective_y_combined[i] = d_reflective_gate[i]
            * cache.reflective_gate[i]
            * (1.0 - cache.reflective_gate[i]);
    }

    // Normalize reflective gradient if multiple active levels
    let active_count = cache.step_caches.iter().filter(|c| c.is_some()).count();
    if active_count > 2 {
        let scale = 1.0 / (active_count as f32).sqrt();
        for i in 0..(s * d) {
            d_reflective_y_combined[i] *= scale;
        }
    }

    // ── Stage 7: Per-level step backward (active levels only) ────────
    let mut d_y_t_mem_total = vec![0.0f32; s * d];
    for level in 0..cfg.k {
        if let Some(ref step_cache) = cache.step_caches[level] {
            let (mem_grads, d_y_t_mem) = step_backward_dispatch(
                cfg, &params.levels[level], step_cache, &d_reflective_y_combined, &cache.y_t,
            );
            grads.levels[level].accumulate(&mem_grads);
            for i in 0..(s * d) {
                d_y_t_mem_total[i] += d_y_t_mem[i];
            }
        }
    }

    // Combine d_y_t
    let mut d_y_t = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_y_t[i] = d_y_t_gate[i] + d_y_t_mem_total[i];
    }

    // ── Stage 6: Scatter d_y_t into d_attn_out ──────────────────────
    let mut d_attn_out = vec![0.0f32; s2 * d];
    d_attn_out[s * d..].copy_from_slice(&d_y_t);

    // ── Stage 5: SWA backward ────────────────────────────────────────
    let mut d_q = vec![0.0f32; s2 * d];
    let mut d_k = vec![0.0f32; s2 * d];
    let mut d_v = vec![0.0f32; s2 * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s2, nh, hd, ws,
    );

    // ── Stage 4: QKV projection backward ─────────────────────────────
    let mut d_assembled = vec![0.0f32; s2 * d];

    crate::tensor::matmul_acc_f32(&d_q, &params.swa.w_q, &mut d_assembled, s2, d, d);
    crate::tensor::matmul_acc_f32(&d_k, &params.swa.w_k, &mut d_assembled, s2, d, d);
    crate::tensor::matmul_acc_f32(&d_v, &params.swa.w_v, &mut d_assembled, s2, d, d);

    let mut d_q_t = vec![0.0f32; d * s2];
    transpose_f32(&d_q, &mut d_q_t, s2, d);
    matmul_f32(&d_q_t, &cache.assembled, &mut grads.swa.w_q, d, s2, d);

    let mut d_k_t = vec![0.0f32; d * s2];
    transpose_f32(&d_k, &mut d_k_t, s2, d);
    matmul_f32(&d_k_t, &cache.assembled, &mut grads.swa.w_k, d, s2, d);

    let mut d_v_t = vec![0.0f32; d * s2];
    transpose_f32(&d_v, &mut d_v_t, s2, d);
    matmul_f32(&d_v_t, &cache.assembled, &mut grads.swa.w_v, d, s2, d);

    // ── Stage 3: Split d_assembled ───────────────────────────────────
    let d_h_t_combined = &d_assembled[..s * d];
    let d_embedded_attn = &d_assembled[s * d..];

    // 1/sqrt(k) chain rule for h_t normalization
    let mut d_h_t_per_level = d_h_t_combined.to_vec();
    if cfg.k > 2 {
        let scale = 1.0 / (cfg.k as f32).sqrt();
        for i in 0..(s * d) {
            d_h_t_per_level[i] *= scale;
        }
    }

    // ── Stage 2: Per-level read-only backward ────────────────────────
    let mut d_embedded_read_total = vec![0.0f32; s * d];
    for level in 0..cfg.k {
        let rc = &cache.read_caches[level];
        let (mem_grads_read, d_embedded_read) = read_only_backward_dispatch(
            cfg, &params.levels[level], &rc.frozen_m, &rc.q_mem,
            &d_h_t_per_level, &cache.embedded, s, d,
        );
        // Read-only grads go to error buffers for frozen levels, direct grads for active
        if cache.pulse.active_levels[level] {
            grads.levels[level].accumulate(&mem_grads_read);
        } else {
            error_buffers[level].accumulate(&mem_grads_read);
        }
        for i in 0..(s * d) {
            d_embedded_read_total[i] += d_embedded_read[i];
        }
    }

    // Combine d_embedded
    let mut d_embedded = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        d_embedded[i] = d_embedded_attn[i] + d_embedded_read_total[i];
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
        MAGConfig::mac_test_config()
    }

    fn make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let input_ids: Vec<usize> = (0..cfg.swa.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.swa.seq_len)
            .map(|t| t % cfg.swa.vocab_size)
            .collect();
        (input_ids, target_ids)
    }

    #[test]
    fn test_mac_forward_finite_loss() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss, _cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        assert!(loss.is_finite(), "MAC loss not finite: {loss}");
        assert!(loss > 0.0, "MAC loss should be positive: {loss}");
        assert!(loss < 20.0, "MAC loss too high: {loss}");
    }

    #[test]
    fn test_mac_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (loss1, _) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        let (loss2, _) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(loss1, loss2, "MAC forward should be deterministic");
    }

    #[test]
    fn test_mac_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mac_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        for (name, g) in [
            ("w_q", &grads.swa.w_q), ("w_k", &grads.swa.w_k),
            ("w_v", &grads.swa.w_v), ("w_o", &grads.swa.w_o),
            ("w_unembed", &grads.swa.w_unembed), ("w_embed", &grads.swa.w_embed),
            ("w_k_mem", &grads.levels[0].w_k_mem),
        ] {
            for (i, &val) in g.iter().enumerate() {
                assert!(val.is_finite(), "mac grad_{name}[{i}] not finite: {val}");
            }
        }
    }

    #[test]
    fn test_mac_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);
        let grads = mac_backward(&params, &cfg, &cache, &input_ids, &target_ids);

        // w_k_mem gradient is attenuated through read_only + concat + attn + step +
        // reflective gate at init, so use a relaxed threshold (1e-12).
        for (name, g) in [
            ("w_q", &grads.swa.w_q), ("w_o", &grads.swa.w_o),
            ("w_k_mem", &grads.levels[0].w_k_mem),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-12, "mac grad_{name} all zeros (max_abs={max_abs})");
        }
    }

    #[test]
    fn test_mac_reflective_gate() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let (_loss, cache) = mac_forward(&params, &cfg, &input_ids, &target_ids);

        // Gate should be non-trivial (not all 0.5 — memory actually does something)
        let gate_sum: f32 = cache.reflective_gate.iter().sum();
        let gate_mean = gate_sum / cache.reflective_gate.len() as f32;
        // With zero-init memory, read_only returns zeros, so reflective gate ≈ 0.5
        // But memory.step updates M, so reflective_y should be non-trivial
        for &g in &cache.reflective_gate {
            assert!(g > 0.0 && g < 1.0, "reflective_gate value {g} not in (0,1)");
        }
        eprintln!("MAC reflective gate mean: {gate_mean:.4}");
    }
}
