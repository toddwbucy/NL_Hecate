/// Hand-written backward pass for all stages of the SWA Transformer.
///
/// Each function computes analytical gradients matching its forward counterpart.
/// These compose via chain rule in gradient.rs.

use crate::tensor::{matmul_f32, matmul_acc_f32, transpose_f32};
use crate::model::{SWAConfig, SWAParams};
use crate::forward::ForwardCache;

/// Gradients for all intermediate activations.
#[allow(dead_code)]
pub(crate) struct BackwardGrads {
    pub(crate) d_logits: Vec<f32>,
    pub(crate) d_projected: Vec<f32>,
    pub(crate) d_attn_out: Vec<f32>,
    pub(crate) d_q: Vec<f32>,
    pub(crate) d_k: Vec<f32>,
    pub(crate) d_v: Vec<f32>,
    pub(crate) d_embedded: Vec<f32>,
}

/// Internal backward pass returning parameter gradients AND d_embedded.
/// Both `backward` and `backward_full` delegate here.
fn backward_internal(
    params: &SWAParams,
    cfg: &SWAConfig,
    cache: &ForwardCache,
    target_ids: &[usize],
) -> (SWAParams, Vec<f32>) {
    let s = cfg.seq_len;
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let ws = cfg.window_size;

    let mut grads = SWAParams::zeros_like(cfg);

    // ── Stage 6: Cross-entropy gradient ──────────────────────────────
    // d_loss/d_logits = softmax(logits) - one_hot(target) / seq_len
    // Guard: target_ids may be shorter than seq_len; only count positions
    // that are both in-bounds and have a valid vocab index.
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

            // softmax of logits row
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
            // subtract 1 at target position
            d_logits[base + target] -= 1.0;
            // scale by 1/count (average over positions)
            for j in 0..v {
                d_logits[base + j] /= count;
            }
        }
    }

    // ── Stage 5: Unembed backward ────────────────────────────────────
    // logits = projected[s,d] @ W_unembed[d,v]
    // d_projected = d_logits[s,v] @ W_unembed^T[v,d]
    // d_W_unembed = projected^T[d,s] @ d_logits[s,v]
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    let mut projected_t = vec![0.0f32; d * s];
    transpose_f32(&cache.projected, &mut projected_t, s, d);
    matmul_f32(&projected_t, &d_logits, &mut grads.w_unembed, d, s, v);

    // ── Stage 4: Output projection backward ──────────────────────────
    // projected = attn_out[s,d] @ W_O^T[d,d]
    // d_attn_out = d_projected[s,d] @ W_O[d,d] (since (W_O^T)^T = W_O)
    // d_W_O = d_projected^T @ attn_out
    let mut d_attn_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.w_o, &mut d_attn_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.attn_out, &mut grads.w_o, d, s, d);

    // ── Stage 3: SWA Attention backward ──────────────────────────────
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    crate::dispatch::swa_backward_dispatch(
        &cache.q, &cache.k, &cache.v,
        &cache.attn_weights, &d_attn_out,
        &mut d_q, &mut d_k, &mut d_v,
        s, nh, hd, ws,
    );

    // ── Stage 2: QKV projection backward ─────────────────────────────
    // Q = embedded[s,d] @ W_Q^T[d,d]
    // d_X = dQ @ W, d_W = dQ^T @ X
    let mut d_embedded = vec![0.0f32; s * d];

    // d_embedded += d_Q @ W_Q
    matmul_acc_f32(&d_q, &params.w_q, &mut d_embedded, s, d, d);
    // d_embedded += d_K @ W_K
    matmul_acc_f32(&d_k, &params.w_k, &mut d_embedded, s, d, d);
    // d_embedded += d_V @ W_V
    matmul_acc_f32(&d_v, &params.w_v, &mut d_embedded, s, d, d);

    // d_W_Q = d_Q^T @ embedded
    let mut d_q_t = vec![0.0f32; d * s];
    transpose_f32(&d_q, &mut d_q_t, s, d);
    matmul_f32(&d_q_t, &cache.embedded, &mut grads.w_q, d, s, d);

    // d_W_K = d_K^T @ embedded
    let mut d_k_t = vec![0.0f32; d * s];
    transpose_f32(&d_k, &mut d_k_t, s, d);
    matmul_f32(&d_k_t, &cache.embedded, &mut grads.w_k, d, s, d);

    // d_W_V = d_V^T @ embedded
    let mut d_v_t = vec![0.0f32; d * s];
    transpose_f32(&d_v, &mut d_v_t, s, d);
    matmul_f32(&d_v_t, &cache.embedded, &mut grads.w_v, d, s, d);

    (grads, d_embedded)
}

/// Backward pass returning parameter gradients (without embedding gradient).
/// Used by Phase 2 CUDA dispatch where some callers don't need embedding gradients.
#[allow(dead_code)]
pub fn backward(
    params: &SWAParams,
    cfg: &SWAConfig,
    cache: &ForwardCache,
    target_ids: &[usize],
) -> SWAParams {
    let (grads, _d_embedded) = backward_internal(params, cfg, cache, target_ids);
    grads
}

/// Full backward including embedding gradient. Needs input_ids for scatter-add.
pub fn backward_full(
    params: &SWAParams,
    cfg: &SWAConfig,
    cache: &ForwardCache,
    input_ids: &[usize],
    target_ids: &[usize],
) -> SWAParams {
    let s = cfg.seq_len;
    let d = cfg.d_model;

    let (mut grads, d_embedded) = backward_internal(params, cfg, cache, target_ids);

    // Embedding scatter-add: d_W_embed[tok] += d_embedded[t]
    for t in 0..s {
        let tok = input_ids[t];
        assert!(tok < cfg.vocab_size, "backward_full: input_ids[{t}]={tok} >= vocab_size {}", cfg.vocab_size);
        for dd in 0..d {
            grads.w_embed[tok * d + dd] += d_embedded[t * d + dd];
        }
    }

    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::forward;

    #[test]
    fn test_backward_produces_finite_grads() {
        let cfg = SWAConfig::test_config();
        let params = SWAParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();

        let (_loss, cache) = forward(&params, &cfg, &input_ids, &target_ids);
        let grads = backward_full(&params, &cfg, &cache, &input_ids, &target_ids);

        // All gradients should be finite
        for (name, g) in [
            ("w_q", &grads.w_q), ("w_k", &grads.w_k), ("w_v", &grads.w_v),
            ("w_o", &grads.w_o), ("w_unembed", &grads.w_unembed), ("w_embed", &grads.w_embed),
        ] {
            for (i, &val) in g.iter().enumerate() {
                assert!(val.is_finite(), "grad_{name}[{i}] is not finite: {val}");
            }
        }
    }

    #[test]
    fn test_backward_grads_not_all_zero() {
        let cfg = SWAConfig::test_config();
        let params = SWAParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();

        let (_loss, cache) = forward(&params, &cfg, &input_ids, &target_ids);
        let grads = backward_full(&params, &cfg, &cache, &input_ids, &target_ids);

        // At least some gradients should be non-zero
        for (name, g) in [
            ("w_q", &grads.w_q), ("w_k", &grads.w_k), ("w_v", &grads.w_v),
            ("w_o", &grads.w_o), ("w_unembed", &grads.w_unembed),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "grad_{name} is all zeros (max_abs={max_abs})");
        }
    }
}
