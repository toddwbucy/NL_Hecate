/// MAG (Memory-Attention-Gate) composition.
///
/// Architecture:
///   embed → QKV (attn) → SWA ─────────────→ attn_out ──┐
///        \→ KVQ (mem) → Delta Rule → sigmoid → gate ──→ * → output proj → unembed → loss
///
/// Two branches share `embedded` input. Memory output gates attention output
/// via element-wise multiply with sigmoid activation.

use crate::tensor::{matmul_f32, transpose_f32, cross_entropy_loss, sigmoid_f32};
use crate::model::{MAGConfig, MAGParams};
use crate::delta_rule::{MemoryRule, DeltaRule, DeltaRuleCache};

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
    pub delta_cache: DeltaRuleCache,
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

    // Stage 2b+3b: Memory branch — Delta Rule (via MemoryRule trait)
    let memory = DeltaRule;
    let (y, delta_cache) = memory.step(params, &embedded, s, d);

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
        delta_cache, gate, gated_out, projected, logits,
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

    // ── Stage 3b: Delta Rule backward (via MemoryRule trait) ──────────
    let memory = DeltaRule;
    let (mem_grads, d_embedded_mem) = memory.step_backward(
        params, &cache.delta_cache, &d_y, &cache.embedded,
    );

    // Accumulate memory parameter gradients
    for i in 0..grads.w_k_mem.len() { grads.w_k_mem[i] += mem_grads.w_k_mem[i]; }
    for i in 0..grads.w_v_mem.len() { grads.w_v_mem[i] += mem_grads.w_v_mem[i]; }
    for i in 0..grads.w_q_mem.len() { grads.w_q_mem[i] += mem_grads.w_q_mem[i]; }
    for i in 0..grads.w_alpha.len() { grads.w_alpha[i] += mem_grads.w_alpha[i]; }
    for i in 0..grads.b_alpha.len() { grads.b_alpha[i] += mem_grads.b_alpha[i]; }
    for i in 0..grads.w_theta.len() { grads.w_theta[i] += mem_grads.w_theta[i]; }
    for i in 0..grads.b_theta.len() { grads.b_theta[i] += mem_grads.b_theta[i]; }

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
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta),
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
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem),
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
        assert_eq!(grads.w_k_mem.len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
    }
}
