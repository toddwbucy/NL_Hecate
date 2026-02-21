/// MAG (Memory-Attention-Gate) composition.
///
/// Architecture:
///   embed → QKV (attn) → SWA ─────────────→ attn_out ──┐
///        \→ KVQ (mem) → Delta Rule → sigmoid → gate ──→ * → output proj → unembed → loss
///
/// Two branches share `embedded` input. Memory output gates attention output
/// via element-wise multiply with sigmoid activation.

use crate::tensor::{transpose_f32, cross_entropy_loss, sigmoid_f32};
use crate::model::{MAGConfig, MAGParams, MemoryRuleKind, HopeVariant};
use crate::delta_rule::{MemoryRule, DeltaRule, DeltaRuleCache, delta_rule_read_only, delta_rule_read_only_backward};
use crate::titans_lmm::{TitansLMM, TitansLMMCache};
use crate::hebbian_rule::{HebbianRule, HebbianCache};
use crate::moneta::{Moneta, MonetaCache, moneta_read_only, moneta_read_only_backward};
use crate::yaad::{YAAD, YAADCache, yaad_read_only, yaad_read_only_backward};
use crate::memora::{MEMORA, MEMORACache, memora_read_only, memora_read_only_backward};
use crate::lattice_osr::{LatticeOSR, LatticeCache, lattice_read_only, lattice_read_only_backward};
use crate::trellis::{Trellis, TrellisCache, trellis_read_only, trellis_read_only_backward};
use crate::atlas_omega::{AtlasOmega, AtlasOmegaCache};
use crate::conductor::{Pulse, ContextState, ErrorBuffer};
use crate::dynamic_freq::{
    FrequencySchedule, FreqGateCache,
    mean_pool, compute_freq_gates, apply_threshold, should_anneal,
    freq_gate_backward, compute_gate_surrogate,
};

/// Memory cache enum for static dispatch across memory rule variants.
/// Preserves monomorphization (AD requires no vtable indirection).
pub enum MemoryCache {
    Delta(DeltaRuleCache),
    Titans(TitansLMMCache),
    Hebbian(HebbianCache),
    Moneta(MonetaCache),
    YAAD(YAADCache),
    MEMORA(MEMORACache),
    Lattice(LatticeCache),
    Trellis(TrellisCache),
    Atlas(AtlasOmegaCache),
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

    // Stage 2a: Attention branch — QKV projections (fused transpose-matmul via cuBLAS)
    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_q, &mut q, s, d, d);
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_k, &mut k, s, d, d);
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_v, &mut vv, s, d, d);

    // Stage 3a: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 2b+3b: Memory branch — dispatch based on memory rule
    let (y, memory_cache) = match cfg.memory_rule {
        MemoryRuleKind::DeltaRule => {
            let (y, cache) = DeltaRule::from_cfg(cfg).step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Delta(cache))
        }
        MemoryRuleKind::TitansLMM => {
            let (y, cache) = TitansLMM::from_cfg(cfg).step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Titans(cache))
        }
        MemoryRuleKind::HebbianRule => {
            let (y, cache) = HebbianRule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Hebbian(cache))
        }
        MemoryRuleKind::Moneta => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2, sign_sharpness: cfg.sign_sharpness };
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
            let rule = LatticeOSR { m_slots: cfg.m_slots, variant: cfg.lattice_variant };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Lattice(cache))
        }
        MemoryRuleKind::Trellis => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Trellis(cache))
        }
        MemoryRuleKind::AtlasOmega => {
            let (y, cache) = AtlasOmega.step(&params.levels[0], &embedded, s, d, None);
            (y, MemoryCache::Atlas(cache))
        }
    };

    // Stage 4: Gating — gate = sigmoid(y), gated_out = attn_out * gate
    let mut gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        gate[i] = sigmoid_f32(y[i]);
        gated_out[i] = attn_out[i] * gate[i];
    }

    // Stage 5: Output projection — projected = gated_out @ W_O^T (fused transpose-matmul)
    let mut projected = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&gated_out, &params.swa.w_o, &mut projected, s, d, d);

    // Stage 6: Unembed — logits = projected @ W_unembed
    let mut logits = vec![0.0f32; s * v];
    crate::dispatch::matmul_dispatch(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

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
    let mut d_projected = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&d_logits, &params.swa.w_unembed, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    crate::dispatch::matmul_dispatch(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 5: Output projection backward ──────────────────────────
    // projected = gated_out @ W_O^T
    let mut d_gated_out = vec![0.0f32; s * d];
    crate::dispatch::matmul_dispatch(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    crate::dispatch::matmul_dispatch(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

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
            DeltaRule::from_cfg(cfg).step_backward(&params.levels[0], delta_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Titans(titans_cache) => {
            TitansLMM::from_cfg(cfg).step_backward(&params.levels[0], titans_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Hebbian(hebbian_cache) => {
            HebbianRule.step_backward(&params.levels[0], hebbian_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Moneta(moneta_cache) => {
            let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2, sign_sharpness: cfg.sign_sharpness };
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
            let rule = LatticeOSR { m_slots: cfg.m_slots, variant: cfg.lattice_variant };
            rule.step_backward(&params.levels[0], lattice_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Trellis(trellis_cache) => {
            let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
            rule.step_backward(&params.levels[0], trellis_cache, &d_y, &cache.embedded)
        }
        MemoryCache::Atlas(atlas_cache) => {
            AtlasOmega.step_backward(&params.levels[0], atlas_cache, &d_y, &cache.embedded)
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

    crate::dispatch::matmul_acc_dispatch(&d_q, &params.swa.w_q, &mut d_embedded, s, d, d);
    crate::dispatch::matmul_acc_dispatch(&d_k, &params.swa.w_k, &mut d_embedded, s, d, d);
    crate::dispatch::matmul_acc_dispatch(&d_v, &params.swa.w_v, &mut d_embedded, s, d, d);

    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    crate::dispatch::matmul_dispatch(&d_q_t, &cache.embedded, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    crate::dispatch::matmul_dispatch(&d_k_t, &cache.embedded, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    crate::dispatch::matmul_dispatch(&d_v_t, &cache.embedded, &mut grads.swa.w_v, d, s, d);

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
    /// Frequency gate cache for backward (None if Fixed schedule).
    pub freq_cache: Option<FreqGateCache>,
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

    // Validate context memory size for Trellis (needs 2*d_k*d, not default d*d).
    if cfg.memory_rule == MemoryRuleKind::Trellis {
        let expected = 2 * cfg.d_compress * d;
        for level in 0..cfg.k {
            assert_eq!(context.memory[level].len(), expected,
                "Trellis context memory[{level}] has wrong size: got {}, expected {expected}. \
                 Use ContextState::new_with_memory_size(k, d, 2 * d_compress * d).",
                context.memory[level].len());
        }
    }

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < v, "cms_forward: input_ids[{t}]={tok} >= vocab_size {v}");
        embedded[t * d..(t + 1) * d].copy_from_slice(&params.swa.w_embed[tok * d..(tok + 1) * d]);
    }

    // Dynamic frequency gate: override pulse active_levels if Learned schedule.
    let (effective_pulse, freq_cache) = match &cfg.frequency_schedule {
        FrequencySchedule::Learned(learned_cfg)
            if !should_anneal(pulse.global_step, learned_cfg.anneal_steps) =>
        {
            let embedded_mean = mean_pool(&embedded, s, d);
            let fc = compute_freq_gates(&embedded_mean, &params.levels, cfg.k, d);
            let active = apply_threshold(&fc, learned_cfg.threshold);
            let new_pulse = Pulse {
                global_step: pulse.global_step,
                active_levels: active,
            };
            (new_pulse, Some(fc))
        }
        _ => (pulse.clone(), None),
    };
    let pulse = &effective_pulse;

    // Stage 2a: Attention branch — QKV projections (fused transpose-matmul via cuBLAS)
    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_q, &mut q, s, d, d);
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_k, &mut k, s, d, d);
    crate::dispatch::matmul_transb_dispatch(&embedded, &params.swa.w_v, &mut vv, s, d, d);

    // Stage 3a: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    crate::dispatch::swa_forward_dispatch(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 2b+3b: Memory branch — HOPE variant dispatch
    //
    // FreqGated/Independent: each level independently processes embedded (default)
    // Chained: levels in series, output chains through all
    // Nested: levels independent but with meta-learned memory re-initialization
    // Sequential: chained with global re-init from slowest level
    let (y_per_level, memory_caches, q_mem_per_level, frozen_memories) = match cfg.hope_variant {
        HopeVariant::Chained => {
            chained_level_outputs(params, cfg, &embedded, pulse, context, s, d)
        }
        HopeVariant::Nested => {
            nested_level_outputs(params, cfg, &embedded, pulse, context, s, d)
        }
        HopeVariant::Sequential => {
            sequential_level_outputs(params, cfg, &embedded, pulse, context, s, d)
        }
        HopeVariant::FreqGated | HopeVariant::Independent => {
            // Default: each level independently processes embedded
            let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
            let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
            let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
            let mut y_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

            for level in 0..cfg.k {
                let active = pulse.active_levels[level];
                let (y_level, mc, qm, fm) = run_level_memory(
                    params, cfg, level, &embedded, s, d, active, context,
                );
                y_per_level.push(y_level);
                memory_caches.push(mc);
                q_mem_per_level.push(qm);
                frozen_memories.push(fm);
            }

            (y_per_level, memory_caches, q_mem_per_level, frozen_memories)
        }
    };

    // Combine level outputs into y_combined.
    //
    // Chained/Sequential (Eqs 70, 73): levels process in series — the final level's
    // output IS y_combined (RETURN h). No aggregation sum.
    //
    // FreqGated/Independent/Nested (Eqs 71, 72, 74): levels process independently —
    // outputs are summed with 1/sqrt(k) normalization for k>2.
    //
    // Why 1/sqrt(k) for k>2: additive sum grows linearly with k, pushing sigmoid
    // into saturation where gradients vanish. 1/sqrt(k) keeps signal variance
    // constant (analogous to attention's 1/sqrt(d) scaling).
    //
    // Why not normalize k=2: the 1/sqrt(k) factor also scales the backward gradient
    // to all memory parameters, slowing outer-loop learning of gate biases (b_theta,
    // b_alpha). At k=2, this cost outweighs the benefit since the signal isn't large
    // enough to cause saturation.
    let y_combined = match cfg.hope_variant {
        HopeVariant::Chained | HopeVariant::Sequential => {
            // Serial pipeline: final level's output is the result (spec: RETURN h)
            y_per_level.last().unwrap().clone()
        }
        _ => {
            // Parallel/independent: aggregate via sum + optional 1/sqrt(k) scaling
            let mut combined = vec![0.0f32; s * d];
            for y_level in &y_per_level {
                for i in 0..(s * d) {
                    combined[i] += y_level[i];
                }
            }
            if cfg.k > 2 {
                let scale = 1.0 / (cfg.k as f32).sqrt();
                for i in 0..(s * d) {
                    combined[i] *= scale;
                }
            }
            combined
        }
    };

    // Stage 4: Gating — gate = sigmoid(y_combined), gated_out = attn_out * gate
    let mut gate = vec![0.0f32; s * d];
    let mut gated_out = vec![0.0f32; s * d];
    for i in 0..(s * d) {
        gate[i] = sigmoid_f32(y_combined[i]);
        gated_out[i] = attn_out[i] * gate[i];
    }

    // Stage 5: Output projection (fused transpose-matmul via cuBLAS)
    let mut projected = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&gated_out, &params.swa.w_o, &mut projected, s, d, d);

    // Stage 6: Unembed
    let mut logits = vec![0.0f32; s * v];
    crate::dispatch::matmul_dispatch(&projected, &params.swa.w_unembed, &mut logits, s, d, v);

    // Stage 7: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = CMSForwardCache {
        embedded, q, k, v: vv, attn_out, attn_weights,
        memory_caches, q_mem_per_level, frozen_memories,
        y_per_level, y_combined,
        gate, gated_out, projected, logits,
        pulse: pulse.clone(),
        freq_cache,
    };

    (loss, cache)
}

// ── HOPE Variant Forwards ────────────────────────────────────────────

/// Run a single level's memory step on the given input.
///
/// Returns (y_level, Option<MemoryCache>, Option<q_mem>, Option<frozen_mem>).
/// This is the level-processing logic extracted from cms_forward's loop body
/// to allow HOPE variants to feed different inputs to each level.
fn run_level_memory(
    params: &MAGParams,
    cfg: &MAGConfig,
    level: usize,
    input: &[f32],
    s: usize,
    d: usize,
    active: bool,
    context: &mut ContextState,
) -> (Vec<f32>, Option<MemoryCache>, Option<Vec<f32>>, Option<Vec<f32>>) {
    if active {
        let initial_m = Some(std::mem::take(&mut context.memory[level]));
        let (y_level, mem_cache) = match cfg.memory_rule {
            MemoryRuleKind::DeltaRule => {
                let (y, cache) = DeltaRule::from_cfg(cfg).step(&params.levels[level], input, s, d, initial_m);
                let m_final_start = s * d * d;
                context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                (y, MemoryCache::Delta(cache))
            }
            MemoryRuleKind::TitansLMM => {
                let (y, cache) = TitansLMM::from_cfg(cfg).step(&params.levels[level], input, s, d, initial_m);
                let m_final_start = s * d * d;
                context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                (y, MemoryCache::Titans(cache))
            }
            MemoryRuleKind::HebbianRule => {
                let (y, cache) = HebbianRule.step(&params.levels[level], input, s, d, initial_m);
                let m_final_start = s * d * d;
                context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                (y, MemoryCache::Hebbian(cache))
            }
            MemoryRuleKind::Moneta => {
                let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2, sign_sharpness: cfg.sign_sharpness };
                let (y, cache) = rule.step(&params.levels[level], input, s, d, initial_m);
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
                let (y, cache) = rule.step(&params.levels[level], input, s, d, initial_m);
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
                let (y, cache) = rule.step(&params.levels[level], input, s, d, initial_m);
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
                let rule = LatticeOSR { m_slots: cfg.m_slots, variant: cfg.lattice_variant };
                let (y, cache) = rule.step(&params.levels[level], input, s, d, initial_m);
                let m = cfg.m_slots;
                let s_final = &cache.s_states[s * m * d..(s + 1) * m * d];
                context.memory[level] = s_final.to_vec();
                (y, MemoryCache::Lattice(cache))
            }
            MemoryRuleKind::Trellis => {
                let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
                let (y, cache) = rule.step(&params.levels[level], input, s, d, initial_m);
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
            MemoryRuleKind::AtlasOmega => {
                let (y, cache) = AtlasOmega.step(&params.levels[level], input, s, d, initial_m);
                let m_final_start = s * d * d;
                context.memory[level] = cache.m_states[m_final_start..m_final_start + d * d].to_vec();
                (y, MemoryCache::Atlas(cache))
            }
        };
        (y_level, Some(mem_cache), None, None)
    } else {
        let frozen_ref = &context.memory[level];
        let (y_level, q_mem) = match cfg.memory_rule {
            MemoryRuleKind::Moneta => moneta_read_only(
                &params.levels[level], input, frozen_ref, s, d, cfg.d_hidden,
            ),
            MemoryRuleKind::YAAD => yaad_read_only(
                &params.levels[level], input, frozen_ref, s, d, cfg.d_hidden,
            ),
            MemoryRuleKind::MEMORA => memora_read_only(
                &params.levels[level], input, frozen_ref, s, d, cfg.d_hidden,
            ),
            MemoryRuleKind::LatticeOSR => lattice_read_only(
                &params.levels[level], input, frozen_ref, s, d, cfg.m_slots,
            ),
            MemoryRuleKind::Trellis => trellis_read_only(
                &params.levels[level], input, frozen_ref, s, d, cfg.d_compress,
            ),
            _ => delta_rule_read_only(
                &params.levels[level], input, frozen_ref, s, d,
            ),
        };
        (y_level, None, Some(q_mem), Some(frozen_ref.clone()))
    }
}

/// HOPE Variant 1: Chained CMS forward (HOPE §6 Eq 70).
///
/// Levels process data serially: level 0 sees `embedded`, level 1 sees level 0's
/// output, etc. Frozen levels pass through unchanged. The final level's output
/// becomes y_combined (no aggregation sum needed — it's already a single output).
///
/// This enables slow levels to process ABSTRACT features (post-processed by fast levels).
pub fn chained_level_outputs(
    params: &MAGParams,
    cfg: &MAGConfig,
    embedded: &[f32],
    pulse: &Pulse,
    context: &mut ContextState,
    s: usize,
    d: usize,
) -> (Vec<Vec<f32>>, Vec<Option<MemoryCache>>, Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) {
    let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut y_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

    // Running state: starts as embedded, each level transforms it
    let mut h = embedded.to_vec();

    for level in 0..cfg.k {
        let active = pulse.active_levels[level];
        if active {
            let (y_level, mc, qm, fm) = run_level_memory(params, cfg, level, &h, s, d, true, context);
            // Next level sees this level's output
            h = y_level.clone();
            y_per_level.push(y_level);
            memory_caches.push(mc);
            q_mem_per_level.push(qm);
            frozen_memories.push(fm);
        } else {
            // Frozen: read-only, but still advances h so downstream levels
            // see this level's read-only transform (preserving serial pipeline).
            let (y_level, mc, qm, fm) = run_level_memory(params, cfg, level, &h, s, d, false, context);
            h = y_level.clone();
            y_per_level.push(y_level);
            memory_caches.push(mc);
            q_mem_per_level.push(qm);
            frozen_memories.push(fm);
        }
    }

    (y_per_level, memory_caches, q_mem_per_level, frozen_memories)
}

/// HOPE Variant 3: Nested CMS forward (HOPE §6 Eq 72).
///
/// Higher level meta-learns initial state of lower level. Level s re-initializes
/// level s+1's memory via linear projection of its own memory state.
///
/// Re-initialization happens when the slower level fires (pulse.is_active).
/// The linear projection M_{s+1} = W_meta @ M_s is a simple d*d matmul
/// using the level's existing w_k_mem as the projection (reusing params).
pub fn nested_level_outputs(
    params: &MAGParams,
    cfg: &MAGConfig,
    embedded: &[f32],
    pulse: &Pulse,
    context: &mut ContextState,
    s: usize,
    d: usize,
) -> (Vec<Vec<f32>>, Vec<Option<MemoryCache>>, Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) {
    let mut memory_caches: Vec<Option<MemoryCache>> = Vec::with_capacity(cfg.k);
    let mut q_mem_per_level: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut frozen_memories: Vec<Option<Vec<f32>>> = Vec::with_capacity(cfg.k);
    let mut y_per_level: Vec<Vec<f32>> = Vec::with_capacity(cfg.k);

    // Nested re-initialization: when level s fires, it re-initializes level s+1.
    // We process levels from slowest (highest index) to fastest (lowest index)
    // so that re-initialization propagates downward.
    for level in (0..cfg.k).rev() {
        if level + 1 < cfg.k && pulse.active_levels[level] {
            // Level `level` is active → re-init level `level+1`'s memory
            // Using w_k_mem of level+1 as the meta-learning projection.
            // M_{level+1} = w_k_mem_{level+1} @ M_{level} (d*d matmul)
            let mem_size = context.memory[level].len();
            if mem_size == d * d && context.memory[level + 1].len() == d * d {
                let mut new_m = vec![0.0f32; d * d];
                // Simple linear projection: new_m = W @ old_m (row-major d×d matmul)
                let w = &params.levels[level + 1].w_k_mem;
                let m = &context.memory[level];
                for i in 0..d {
                    for j in 0..d {
                        let mut sum = 0.0f32;
                        for k_idx in 0..d {
                            sum += w[i * d + k_idx] * m[k_idx * d + j];
                        }
                        new_m[i * d + j] = sum;
                    }
                }
                context.memory[level + 1] = new_m;
            }
        }
    }

    // After re-initialization, all levels process embedded independently (like FreqGated)
    for level in 0..cfg.k {
        let active = pulse.active_levels[level];
        let (y_level, mc, qm, fm) = run_level_memory(params, cfg, level, embedded, s, d, active, context);
        y_per_level.push(y_level);
        memory_caches.push(mc);
        q_mem_per_level.push(qm);
        frozen_memories.push(fm);
    }

    (y_per_level, memory_caches, q_mem_per_level, frozen_memories)
}

/// HOPE Variant 4: Sequential CMS forward (HOPE §6 Eq 73).
///
/// Like Chained (levels in series), but all levels are re-initialized from
/// the slowest level's compressed state when the slowest level fires.
/// The slowest level serves as the "context compressor" — it holds the most
/// persistent knowledge that seeds all other levels.
pub fn sequential_level_outputs(
    params: &MAGParams,
    cfg: &MAGConfig,
    embedded: &[f32],
    pulse: &Pulse,
    context: &mut ContextState,
    s: usize,
    d: usize,
) -> (Vec<Vec<f32>>, Vec<Option<MemoryCache>>, Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) {
    // Re-init all levels from slowest level's state when slowest fires
    let slowest = cfg.k - 1;
    if pulse.active_levels[slowest] {
        let base_mem = context.memory[slowest].clone();
        let mem_size = base_mem.len();
        for level in 0..slowest {
            if context.memory[level].len() == mem_size && mem_size == d * d {
                // Project base state through level's own w_k_mem
                let w = &params.levels[level].w_k_mem;
                let mut new_m = vec![0.0f32; d * d];
                for i in 0..d {
                    for j in 0..d {
                        let mut sum = 0.0f32;
                        for k_idx in 0..d {
                            sum += w[i * d + k_idx] * base_mem[k_idx * d + j];
                        }
                        new_m[i * d + j] = sum;
                    }
                }
                context.memory[level] = new_m;
            }
        }
    }

    // Then chain levels serially (same as Chained variant)
    chained_level_outputs(params, cfg, embedded, pulse, context, s, d)
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
    let mut d_projected = vec![0.0f32; s * d];
    crate::dispatch::matmul_transb_dispatch(&d_logits, &params.swa.w_unembed, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    crate::dispatch::matmul_dispatch(&projected_t, &d_logits, &mut grads.swa.w_unembed, d, s, v);

    // ── Stage 5: Output projection backward ──────────────────────────
    let mut d_gated_out = vec![0.0f32; s * d];
    crate::dispatch::matmul_dispatch(&d_projected, &params.swa.w_o, &mut d_gated_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    crate::dispatch::matmul_dispatch(&d_projected_t, &cache.gated_out, &mut grads.swa.w_o, d, s, d);

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

    // Chain rule for aggregation backward.
    //
    // Chained/Sequential: y_combined = y_last, so d_y_combined flows only to the
    // last level. Full chain-rule through the serial pipeline (d_y_k-1 → d_y_k-2
    // → ...) is a Stage 3 backward extension — for now, only the last level gets
    // gradients. Earlier levels in the chain still get frozen-path gradients via
    // error buffers when applicable.
    //
    // FreqGated/Independent/Nested: y_combined = sum, so d_y_combined distributes
    // to all levels equally (with 1/sqrt(k) scaling for k>2).
    match cfg.hope_variant {
        HopeVariant::Chained | HopeVariant::Sequential => {
            // No scaling needed — d_y_combined goes only to last level (handled below)
        }
        _ => {
            if cfg.k > 2 {
                let scale = 1.0 / (cfg.k as f32).sqrt();
                for i in 0..(s * d) {
                    d_y_combined[i] *= scale;
                }
            }
        }
    }

    // ── Stage 3b: Per-level memory backward ──────────────────────────
    // For aggregated variants: d_y_combined distributes to each level.
    // For Chained/Sequential: only the last level receives d_y_combined.
    let chained_backward = matches!(cfg.hope_variant, HopeVariant::Chained | HopeVariant::Sequential);
    let mut d_embedded_mem_total = vec![0.0f32; s * d];

    for level in 0..cfg.k {
        // Chained/Sequential: only last level gets gradient from y_combined
        if chained_backward && level != cfg.k - 1 {
            // Earlier levels in the serial chain: no direct gradient from y_combined.
            // Their error buffers still accumulate frozen-path gradients when applicable.
            if !cache.pulse.active_levels[level] {
                let q_mem = cache.q_mem_per_level[level].as_ref().unwrap();
                let frozen_m = cache.frozen_memories[level].as_ref().unwrap();
                let (mem_grads, _d_embedded_mem) = match cfg.memory_rule {
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
                        &params.levels[level], frozen_m, q_mem, &d_y_combined, &cache.embedded, s, d,
                    ),
                };
                error_buffers[level].accumulate(&mem_grads);
            }
            continue;
        }
        if cache.pulse.active_levels[level] {
            // Active level: dispatch backward based on cache variant
            let mem_cache = cache.memory_caches[level].as_ref().unwrap();
            let (mem_grads, d_embedded_mem) = match mem_cache {
                MemoryCache::Delta(delta_cache) => {
                    DeltaRule::from_cfg(cfg).step_backward(&params.levels[level], delta_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Titans(titans_cache) => {
                    TitansLMM::from_cfg(cfg).step_backward(&params.levels[level], titans_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Hebbian(hebbian_cache) => {
                    HebbianRule.step_backward(&params.levels[level], hebbian_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Moneta(moneta_cache) => {
                    let rule = Moneta { d_hidden: cfg.d_hidden, lp_p: cfg.lp_p, lambda_2: cfg.lambda_2, sign_sharpness: cfg.sign_sharpness };
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
                    let rule = LatticeOSR { m_slots: cfg.m_slots, variant: cfg.lattice_variant };
                    rule.step_backward(&params.levels[level], lattice_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Trellis(trellis_cache) => {
                    let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
                    rule.step_backward(&params.levels[level], trellis_cache, &d_y_combined, &cache.embedded)
                }
                MemoryCache::Atlas(atlas_cache) => {
                    AtlasOmega.step_backward(&params.levels[level], atlas_cache, &d_y_combined, &cache.embedded)
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

    // ── Frequency gate backward (straight-through estimator) ────────
    if let Some(ref fc) = cache.freq_cache {
        // Compute surrogate gradient signal for frequency gates
        let d_gate_values = compute_gate_surrogate(
            &cache.y_per_level, &d_y_combined, &cache.pulse.active_levels, cfg.k, s * d,
        );
        let (freq_grads, d_embedded_mean) = freq_gate_backward(
            &d_gate_values, fc, &params.levels, cfg.k, d,
        );

        // Accumulate w_freq/b_freq gradients into level grads
        for (l, fg) in freq_grads.into_iter().enumerate() {
            if !grads.levels[l].w_freq.is_empty() {
                for j in 0..d {
                    grads.levels[l].w_freq[j] += fg.d_w_freq[j];
                }
                grads.levels[l].b_freq[0] += fg.d_b_freq[0];
            }
        }

        // d_embedded_mean contributes to d_embedded via mean-pool backward:
        // d_embedded[t, j] += d_embedded_mean[j] / seq_len for all t
        let inv_s = 1.0 / s as f32;
        for t in 0..s {
            let base = t * d;
            for j in 0..d {
                d_embedded_mem_total[base + j] += d_embedded_mean[j] * inv_s;
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

    crate::dispatch::matmul_acc_dispatch(&d_q, &params.swa.w_q, &mut d_embedded, s, d, d);
    crate::dispatch::matmul_acc_dispatch(&d_k, &params.swa.w_k, &mut d_embedded, s, d, d);
    crate::dispatch::matmul_acc_dispatch(&d_v, &params.swa.w_v, &mut d_embedded, s, d, d);

    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    crate::dispatch::matmul_dispatch(&d_q_t, &cache.embedded, &mut grads.swa.w_q, d, s, d);

    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    crate::dispatch::matmul_dispatch(&d_k_t, &cache.embedded, &mut grads.swa.w_k, d, s, d);

    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    crate::dispatch::matmul_dispatch(&d_v_t, &cache.embedded, &mut grads.swa.w_v, d, s, d);

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

    // ── HOPE variant tests ─────────────────────────────────────────

    fn test_config_k2_variant(variant: HopeVariant) -> MAGConfig {
        let mut cfg = MAGConfig::test_config_k2();
        cfg.hope_variant = variant;
        cfg
    }

    #[test]
    fn test_hope_chained_finite_loss() {
        let cfg = test_config_k2_variant(HopeVariant::Chained);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        assert!(loss.is_finite(), "Chained loss not finite: {loss}");
        assert!(loss > 0.0, "Chained loss should be positive: {loss}");
    }

    #[test]
    fn test_hope_chained_levels_in_series() {
        // In Chained mode, level 1 sees level 0's output, not raw embedded.
        // Verify both levels produce different outputs (they process different inputs).
        let cfg = test_config_k2_variant(HopeVariant::Chained);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (_, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);

        // Compare Chained vs FreqGated: they should produce different losses
        // because the input to level 1 differs.
        let cfg_fg = test_config_k2_variant(HopeVariant::FreqGated);
        let mut ctx_fg = ContextState::new(cfg_fg.k, cfg_fg.swa.d_model);
        let (_, cache_fg) = cms_forward(&params, &cfg_fg, &input_ids, &target_ids, &pulse, &mut ctx_fg);
        // Level 0 should be the same (both see embedded), but level 1 differs
        let y0_diff: f32 = cache.y_per_level[0].iter()
            .zip(cache_fg.y_per_level[0].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(y0_diff < 1e-6, "Level 0 should be same for Chained vs FreqGated, diff={y0_diff}");

        let y1_diff: f32 = cache.y_per_level[1].iter()
            .zip(cache_fg.y_per_level[1].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(y1_diff > 1e-6, "Level 1 should differ for Chained vs FreqGated, diff={y1_diff}");
    }

    #[test]
    fn test_hope_nested_finite_loss() {
        let cfg = test_config_k2_variant(HopeVariant::Nested);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        assert!(loss.is_finite(), "Nested loss not finite: {loss}");
        assert!(loss > 0.0, "Nested loss should be positive: {loss}");
    }

    #[test]
    fn test_hope_nested_reinit_modifies_memory() {
        // Verify that nested variant actually re-initializes level 1's memory
        // from level 0's state via projection when level 0 fires.
        let cfg = test_config_k2_variant(HopeVariant::Nested);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let d = cfg.swa.d_model;

        // Step 0: both active — initializes memories from zeros
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let mut context = ContextState::new(cfg.k, d);
        let _ = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut context);

        // After step 0, both levels have non-zero memory
        let m0_norm: f32 = context.memory[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let m1_norm: f32 = context.memory[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(m0_norm > 1e-8, "Level 0 memory should be non-zero, norm={m0_norm}");
        assert!(m1_norm > 1e-8, "Level 1 memory should be non-zero, norm={m1_norm}");

        // Manually compute what nested re-init would produce for level 1:
        // new_m1 = w_k_mem[1] @ memory[0]
        let w = &params.levels[1].w_k_mem;
        let m = &context.memory[0];
        let mut expected_reinit = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut sum = 0.0f32;
                for k_idx in 0..d {
                    sum += w[i * d + k_idx] * m[k_idx * d + j];
                }
                expected_reinit[i * d + j] = sum;
            }
        }
        let reinit_norm: f32 = expected_reinit.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(reinit_norm > 1e-8, "Projected re-init should be non-zero, norm={reinit_norm}");

        // The re-init value should differ from the current memory[1]
        let diff: f32 = expected_reinit.iter().zip(context.memory[1].iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-8, "Re-init value should differ from current memory[1], diff={diff}");
    }

    #[test]
    fn test_hope_sequential_finite_loss() {
        let cfg = test_config_k2_variant(HopeVariant::Sequential);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (loss, _cache) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut context);
        assert!(loss.is_finite(), "Sequential loss not finite: {loss}");
        assert!(loss > 0.0, "Sequential loss should be positive: {loss}");
    }

    #[test]
    fn test_hope_sequential_delegates_to_chained() {
        // Sequential = re-init from slowest + chained. When slowest is NOT firing,
        // sequential should behave identically to chained (no re-init).
        let cfg_seq = test_config_k2_variant(HopeVariant::Sequential);
        let cfg_ch = test_config_k2_variant(HopeVariant::Chained);
        let params = MAGParams::init(&cfg_seq, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg_seq);

        // When slowest level (1) is NOT firing, sequential delegates to chained directly
        let pulse = Pulse { global_step: 1, active_levels: vec![true, false] };
        let mut ctx_seq = ContextState::new(cfg_seq.k, cfg_seq.swa.d_model);
        let mut ctx_ch = ContextState::new(cfg_ch.k, cfg_ch.swa.d_model);
        let (loss_seq, _) = cms_forward(&params, &cfg_seq, &input_ids, &target_ids, &pulse, &mut ctx_seq);
        let (loss_ch, _) = cms_forward(&params, &cfg_ch, &input_ids, &target_ids, &pulse, &mut ctx_ch);

        // Without slowest firing, no re-init occurs → identical to Chained
        assert_eq!(loss_seq, loss_ch, "Sequential without slowest firing should equal Chained");
    }

    #[test]
    fn test_hope_sequential_multi_step() {
        // Verify sequential can run multiple steps including slowest firing.
        let cfg = test_config_k2_variant(HopeVariant::Sequential);
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg);
        let d = cfg.swa.d_model;
        let mut context = ContextState::new(cfg.k, d);

        // Step 0: both active (initializes memory via chained processing)
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (loss0, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse0, &mut context);
        assert!(loss0.is_finite(), "Step 0 loss should be finite");

        // Step 1: only level 0 active (no re-init since slowest is frozen)
        let pulse1 = Pulse { global_step: 1, active_levels: vec![true, false] };
        let (loss1, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse1, &mut context);
        assert!(loss1.is_finite(), "Step 1 loss should be finite");

        // Step 2: both active again (slowest fires → re-init occurs)
        let pulse2 = Pulse { global_step: 2, active_levels: vec![true, true] };
        let (loss2, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse2, &mut context);
        assert!(loss2.is_finite(), "Step 2 loss should be finite (after re-init)");

        // Level 0 memory should be non-zero after processing
        let m0_norm: f32 = context.memory[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(m0_norm > 1e-8, "Level 0 memory should be non-zero, norm={m0_norm}");
    }

    #[test]
    fn test_hope_freqgated_is_default() {
        // FreqGated is the default variant; verify test_config uses it
        let cfg = MAGConfig::test_config_k2();
        assert_eq!(cfg.hope_variant, HopeVariant::FreqGated,
            "Default hope_variant should be FreqGated");
    }

    #[test]
    fn test_hope_independent_same_as_freqgated() {
        // Independent and FreqGated should produce identical results
        let cfg_fg = test_config_k2_variant(HopeVariant::FreqGated);
        let cfg_ind = test_config_k2_variant(HopeVariant::Independent);
        let params = MAGParams::init(&cfg_fg, 42);
        let (input_ids, target_ids) = make_test_data_k2(&cfg_fg);
        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

        let mut ctx_fg = ContextState::new(cfg_fg.k, cfg_fg.swa.d_model);
        let mut ctx_ind = ContextState::new(cfg_ind.k, cfg_ind.swa.d_model);

        let (loss_fg, _) = cms_forward(&params, &cfg_fg, &input_ids, &target_ids, &pulse, &mut ctx_fg);
        let (loss_ind, _) = cms_forward(&params, &cfg_ind, &input_ids, &target_ids, &pulse, &mut ctx_ind);
        assert_eq!(loss_fg, loss_ind, "Independent should equal FreqGated");
    }

    #[test]
    fn test_hope_all_variants_deterministic() {
        for variant in [HopeVariant::Chained, HopeVariant::Nested,
                        HopeVariant::Sequential, HopeVariant::FreqGated] {
            let cfg = test_config_k2_variant(variant);
            let params = MAGParams::init(&cfg, 42);
            let (input_ids, target_ids) = make_test_data_k2(&cfg);
            let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

            let mut ctx1 = ContextState::new(cfg.k, cfg.swa.d_model);
            let mut ctx2 = ContextState::new(cfg.k, cfg.swa.d_model);
            let (loss1, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx1);
            let (loss2, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx2);
            assert_eq!(loss1, loss2, "{variant:?} forward should be deterministic");
        }
    }
}
