/// Hand-written backward pass for all stages of the SWA Transformer.
///
/// Each function computes analytical gradients matching its forward counterpart.
/// These compose via chain rule in gradient.rs.

use crate::tensor::{matmul_f32, matmul_acc_f32, transpose_f32};
use crate::model::{SWAConfig, SWAParams};
use crate::forward::ForwardCache;

/// Gradients for all intermediate activations.
pub struct BackwardGrads {
    /// d_loss/d_logits: [seq_len, vocab_size]
    pub d_logits: Vec<f32>,
    /// d_loss/d_projected: [seq_len, d_model]
    pub d_projected: Vec<f32>,
    /// d_loss/d_attn_out: [seq_len, d_model]
    pub d_attn_out: Vec<f32>,
    /// d_loss/d_q: [seq_len, d_model]
    pub d_q: Vec<f32>,
    /// d_loss/d_k: [seq_len, d_model]
    pub d_k: Vec<f32>,
    /// d_loss/d_v: [seq_len, d_model]
    pub d_v: Vec<f32>,
    /// d_loss/d_embedded: [seq_len, d_model]
    pub d_embedded: Vec<f32>,
}

/// Full backward pass. Returns parameter gradients.
///
/// Stages (reverse order):
/// 6. d_loss/d_logits (cross-entropy softmax gradient)
/// 5. d_loss/d_projected and d_loss/d_W_unembed
/// 4. d_loss/d_attn_out and d_loss/d_W_O
/// 3. d_loss/d_Q, d_loss/d_K, d_loss/d_V (SWA attention backward)
/// 2. d_loss/d_embedded and d_loss/d_W_Q, d_loss/d_W_K, d_loss/d_W_V
/// 1. d_loss/d_W_embed (scatter-add)
pub fn backward(
    params: &SWAParams,
    cfg: &SWAConfig,
    cache: &ForwardCache,
    target_ids: &[usize],
) -> SWAParams {
    let s = cfg.seq_len;
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let ws = cfg.window_size;

    let mut grads = SWAParams::zeros_like(cfg);

    // ── Stage 6: Cross-entropy gradient ──────────────────────────────
    // d_loss/d_logits = softmax(logits) - one_hot(target) / seq_len
    let mut d_logits = vec![0.0f32; s * v];
    let count = target_ids.iter().filter(|&&t| t < v).count() as f32;
    if count > 0.0 {
        for t in 0..s {
            let base = t * v;
            let row = &cache.logits[base..base + v];
            let target = target_ids[t];
            if target >= v { continue; }

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
    // d_W_O = (d_projected)^T[d,s] @ attn_out^T ... wait.
    // projected = attn_out @ W_O^T. Let A = attn_out, B = W_O^T.
    // d_A = dC @ B^T = d_projected @ (W_O^T)^T = d_projected @ W_O
    // d_B = A^T @ dC = attn_out^T @ d_projected
    // d_(W_O^T) = attn_out^T @ d_projected, so d_W_O = (attn_out^T @ d_projected)^T
    // = d_projected^T @ attn_out
    let mut d_attn_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.w_o, &mut d_attn_out, s, d, d);

    let mut d_projected_t = vec![0.0f32; d * s];
    transpose_f32(&d_projected, &mut d_projected_t, s, d);
    matmul_f32(&d_projected_t, &cache.attn_out, &mut grads.w_o, d, s, d);

    // ── Stage 3: SWA Attention backward ──────────────────────────────
    // Forward: for each head h, query position q_pos:
    //   scores[w] = sum_d Q[q,h,d] * K[k,h,d] * scale
    //   attn_weights = softmax(scores)
    //   out[q,h,d] = sum_w attn_weights[w] * V[k,h,d]
    //
    // Backward:
    //   d_V[k,h,d] += attn_weights[w] * d_out[q,h,d]
    //   d_attn_w[w] = sum_d d_out[q,h,d] * V[k,h,d]
    //   d_scores = softmax_backward(d_attn_w, attn_weights)
    //   d_Q[q,h,d] += d_scores[w] * K[k,h,d] * scale
    //   d_K[k,h,d] += d_scores[w] * Q[q,h,d] * scale
    let total_dim = nh * hd;
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v = vec![0.0f32; s * d];

    let scale = 1.0 / (hd as f32).sqrt();

    for h in 0..nh {
        let h_offset = h * hd;

        for q_pos in 0..s {
            let win_start = if q_pos + 1 >= ws { q_pos + 1 - ws } else { 0 };
            let win_len = q_pos - win_start + 1;

            let aw_base = (h * s + q_pos) * ws;

            // d_attn_w[w] = sum_d d_attn_out[q,h,d] * V[k,h,d]
            let mut d_attn_w = vec![0.0f32; ws];
            for w in 0..win_len {
                let k_pos = win_start + w;
                let mut sum = 0.0f32;
                for dd in 0..hd {
                    sum += d_attn_out[q_pos * total_dim + h_offset + dd]
                         * cache.v[k_pos * total_dim + h_offset + dd];
                }
                d_attn_w[w] = sum;
            }

            // d_V[k,h,d] += attn_weights[w] * d_attn_out[q,h,d]
            for w in 0..win_len {
                let k_pos = win_start + w;
                let aw = cache.attn_weights[aw_base + w];
                for dd in 0..hd {
                    d_v[k_pos * total_dim + h_offset + dd] +=
                        aw * d_attn_out[q_pos * total_dim + h_offset + dd];
                }
            }

            // Softmax backward: d_scores[i] = P[i] * (d_attn_w[i] - sum_j(P[j] * d_attn_w[j]))
            let mut dot_pw = 0.0f32;
            for w in 0..win_len {
                dot_pw += cache.attn_weights[aw_base + w] * d_attn_w[w];
            }
            let mut d_scores = vec![0.0f32; ws];
            for w in 0..win_len {
                d_scores[w] = cache.attn_weights[aw_base + w] * (d_attn_w[w] - dot_pw);
            }

            // d_Q[q,h,d] += d_scores[w] * K[k,h,d] * scale
            // d_K[k,h,d] += d_scores[w] * Q[q,h,d] * scale
            for w in 0..win_len {
                let k_pos = win_start + w;
                let ds = d_scores[w] * scale;
                for dd in 0..hd {
                    d_q[q_pos * total_dim + h_offset + dd] +=
                        ds * cache.k[k_pos * total_dim + h_offset + dd];
                    d_k[k_pos * total_dim + h_offset + dd] +=
                        ds * cache.q[q_pos * total_dim + h_offset + dd];
                }
            }
        }
    }

    // ── Stage 2: QKV projection backward ─────────────────────────────
    // Q = embedded[s,d] @ W_Q^T[d,d]
    // d_embedded_from_Q = d_Q[s,d] @ W_Q[d,d]
    // d_W_Q = d_Q^T[d,s] @ embedded[s,d] ... then transpose.
    //   Actually: Q = X @ W^T → d_X = dQ @ W, d_W^T = X^T @ dQ, d_W = dQ^T @ X
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

    // ── Stage 1: Embedding backward ──────────────────────────────────
    // embedded[t] = W_embed[input_ids[t]]
    // d_W_embed[tok] += d_embedded[t] for all t where input_ids[t] == tok
    // This is scatter-add.
    // Note: input_ids are needed but not stored in cache. We get them from
    // the caller context via the target_ids hack — actually we need input_ids.
    // For now, we'll accept input_ids as a parameter.

    // (Embedding gradient is handled in the public backward_full function)

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
    let v = cfg.vocab_size;

    // Get all other gradients
    let mut grads = backward(params, cfg, cache, target_ids);

    // Recompute d_embedded for the scatter-add (backward already computed it
    // but doesn't return it separately — let's recompute the embedding gradient
    // part). Actually, backward() computes and uses d_embedded internally but
    // doesn't include the W_embed gradient. Let's fix this by recomputing
    // d_embedded here.

    // Recompute d_logits
    let mut d_logits = vec![0.0f32; s * v];
    let count = target_ids.iter().filter(|&&t| t < v).count() as f32;
    if count > 0.0 {
        for t in 0..s {
            let base = t * v;
            let row = &cache.logits[base..base + v];
            let target = target_ids[t];
            if target >= v { continue; }
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

    // d_projected
    let mut w_unembed_t = vec![0.0f32; v * d];
    transpose_f32(&params.w_unembed, &mut w_unembed_t, d, v);
    let mut d_projected = vec![0.0f32; s * d];
    matmul_f32(&d_logits, &w_unembed_t, &mut d_projected, s, v, d);

    // d_attn_out
    let mut d_attn_out = vec![0.0f32; s * d];
    matmul_f32(&d_projected, &params.w_o, &mut d_attn_out, s, d, d);

    // d_Q, d_K, d_V from SWA backward (same as in backward())
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let ws = cfg.window_size;
    let total_dim = nh * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut d_q = vec![0.0f32; s * d];
    let mut d_k = vec![0.0f32; s * d];
    let mut d_v_vec = vec![0.0f32; s * d];

    for h in 0..nh {
        let h_offset = h * hd;
        for q_pos in 0..s {
            let win_start = if q_pos + 1 >= ws { q_pos + 1 - ws } else { 0 };
            let win_len = q_pos - win_start + 1;
            let aw_base = (h * s + q_pos) * ws;

            let mut d_attn_w = vec![0.0f32; ws];
            for w in 0..win_len {
                let k_pos = win_start + w;
                let mut sum = 0.0f32;
                for dd in 0..hd {
                    sum += d_attn_out[q_pos * total_dim + h_offset + dd]
                         * cache.v[k_pos * total_dim + h_offset + dd];
                }
                d_attn_w[w] = sum;
            }

            for w in 0..win_len {
                let k_pos = win_start + w;
                let aw = cache.attn_weights[aw_base + w];
                for dd in 0..hd {
                    d_v_vec[k_pos * total_dim + h_offset + dd] +=
                        aw * d_attn_out[q_pos * total_dim + h_offset + dd];
                }
            }

            let mut dot_pw = 0.0f32;
            for w in 0..win_len {
                dot_pw += cache.attn_weights[aw_base + w] * d_attn_w[w];
            }
            let mut d_scores = vec![0.0f32; ws];
            for w in 0..win_len {
                d_scores[w] = cache.attn_weights[aw_base + w] * (d_attn_w[w] - dot_pw);
            }

            for w in 0..win_len {
                let k_pos = win_start + w;
                let ds = d_scores[w] * scale;
                for dd in 0..hd {
                    d_q[q_pos * total_dim + h_offset + dd] +=
                        ds * cache.k[k_pos * total_dim + h_offset + dd];
                    d_k[k_pos * total_dim + h_offset + dd] +=
                        ds * cache.q[q_pos * total_dim + h_offset + dd];
                }
            }
        }
    }

    // d_embedded from QKV
    let mut d_embedded = vec![0.0f32; s * d];
    matmul_acc_f32(&d_q, &params.w_q, &mut d_embedded, s, d, d);
    matmul_acc_f32(&d_k, &params.w_k, &mut d_embedded, s, d, d);
    matmul_acc_f32(&d_v_vec, &params.w_v, &mut d_embedded, s, d, d);

    // Embedding scatter-add
    for t in 0..s {
        let tok = input_ids[t];
        if tok < cfg.vocab_size {
            for dd in 0..d {
                grads.w_embed[tok * d + dd] += d_embedded[t * d + dd];
            }
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
