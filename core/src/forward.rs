/// Full forward pass: embed → QKV project → SWA → output project → unembed → loss.
///
/// Track Zero-A: single block, no residual, no LayerNorm, no MLP.
/// All intermediate buffers are allocated and returned for backward pass use.

use crate::tensor::{matmul_f32, cross_entropy_loss};
use crate::swa::swa_forward;
use crate::model::{SWAConfig, SWAParams};

/// All intermediate activations from a forward pass, needed for backward.
pub struct ForwardCache {
    /// Embedded input: [seq_len, d_model]
    pub embedded: Vec<f32>,
    /// Q = embedded @ W_Q: [seq_len, d_model]
    pub q: Vec<f32>,
    /// K = embedded @ W_K: [seq_len, d_model]
    pub k: Vec<f32>,
    /// V = embedded @ W_V: [seq_len, d_model]
    pub v: Vec<f32>,
    /// SWA output: [seq_len, d_model]
    pub attn_out: Vec<f32>,
    /// Attention weights: [num_heads, seq_len, window_size]
    pub attn_weights: Vec<f32>,
    /// Output projection: [seq_len, d_model]
    pub projected: Vec<f32>,
    /// Logits: [seq_len, vocab_size]
    pub logits: Vec<f32>,
}

/// Run the full forward pass. Returns (loss, cache).
///
/// input_ids: [seq_len] — token indices for input
/// target_ids: [seq_len] — token indices for loss computation (next-token targets)
pub fn forward(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: &[usize],
    target_ids: &[usize],
) -> (f32, ForwardCache) {
    let s = cfg.seq_len;
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let nh = cfg.num_heads;
    let hd = cfg.head_dim;
    let ws = cfg.window_size;

    // Stage 1: Embedding lookup
    let mut embedded = vec![0.0f32; s * d];
    for t in 0..s {
        let tok = input_ids[t];
        if tok < v {
            let src = &params.w_embed[tok * d..(tok + 1) * d];
            embedded[t * d..(t + 1) * d].copy_from_slice(src);
        }
    }

    // Stage 2: QKV projections — Q = X @ W_Q^T, etc.
    // W_Q is [d_model, d_model], stored row-major.
    // X is [seq_len, d_model]. We want X @ W_Q^T = [seq_len, d_model].
    // Equivalently: for each position, q[t] = embedded[t] @ W_Q^T
    // But matmul_f32(A, B, out, M, K, N) computes A[M,K] @ B[K,N].
    // So we need W_Q transposed, or we can transpose the operation.
    // Simpler: X[s,d] @ W_Q^T[d,d] where W_Q^T[i,j] = W_Q[j,i].
    // Let's just do the matmul directly: Q = X @ W_Q^T.
    // Since W_Q is [d,d] row-major and we want W_Q^T, we can use
    // matmul with B=W_Q and swap the last two dims... actually,
    // for a square matrix, X @ W^T can be computed as:
    // out[i,j] = sum_k X[i,k] * W[j,k] = sum_k X[i,k] * W^T[k,j]
    // matmul_f32(X, W^T, out, s, d, d) where W^T needs to be materialized.
    //
    // Actually, let's use a different convention: store W as [d_out, d_in]
    // and compute Y = X @ W^T. This means Y[t,j] = sum_k X[t,k] * W[j,k].
    //
    // For simplicity, let's just transpose W first.
    let mut w_q_t = vec![0.0f32; d * d];
    let mut w_k_t = vec![0.0f32; d * d];
    let mut w_v_t = vec![0.0f32; d * d];
    crate::tensor::transpose_f32(&params.w_q, &mut w_q_t, d, d);
    crate::tensor::transpose_f32(&params.w_k, &mut w_k_t, d, d);
    crate::tensor::transpose_f32(&params.w_v, &mut w_v_t, d, d);

    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut vv = vec![0.0f32; s * d];
    matmul_f32(&embedded, &w_q_t, &mut q, s, d, d);
    matmul_f32(&embedded, &w_k_t, &mut k, s, d, d);
    matmul_f32(&embedded, &w_v_t, &mut vv, s, d, d);

    // Stage 3: SWA Attention
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_weights = vec![0.0f32; nh * s * ws];
    swa_forward(&q, &k, &vv, &mut attn_out, &mut attn_weights, s, nh, hd, ws);

    // Stage 4: Output projection — projected = attn_out @ W_O^T
    let mut w_o_t = vec![0.0f32; d * d];
    crate::tensor::transpose_f32(&params.w_o, &mut w_o_t, d, d);
    let mut projected = vec![0.0f32; s * d];
    matmul_f32(&attn_out, &w_o_t, &mut projected, s, d, d);

    // Stage 5: Unembed — logits = projected @ W_unembed^T
    // W_unembed is [d_model, vocab_size], so W_unembed^T is [vocab_size, d_model].
    // logits[t,v] = sum_k projected[t,k] * W_unembed[k,v]
    // Actually W_unembed[d, vocab] means: logits = projected[s,d] @ W_unembed[d,v] = [s,v]
    // No transpose needed if we store W_unembed as [d_model, vocab_size]!
    let mut logits = vec![0.0f32; s * v];
    matmul_f32(&projected, &params.w_unembed, &mut logits, s, d, v);

    // Stage 6: Cross-entropy loss
    let loss = cross_entropy_loss(&logits, target_ids, s, v);

    let cache = ForwardCache {
        embedded, q, k, v: vv, attn_out, attn_weights, projected, logits,
    };

    (loss, cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_produces_finite_loss() {
        let cfg = SWAConfig::test_config();
        let params = SWAParams::init(&cfg, 42);
        // Simple input: tokens 0..seq_len, targets shifted by 1
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();

        let (loss, _cache) = forward(&params, &cfg, &input_ids, &target_ids);

        assert!(loss.is_finite(), "Loss should be finite, got {}", loss);
        assert!(loss > 0.0, "Loss should be positive, got {}", loss);
        // Random init loss should be close to ln(vocab_size) ≈ 5.55
        assert!(loss < 20.0, "Loss {} seems too high for random init", loss);
    }

    #[test]
    fn test_forward_cache_shapes() {
        let cfg = SWAConfig::test_config();
        let params = SWAParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();

        let (_loss, cache) = forward(&params, &cfg, &input_ids, &target_ids);

        assert_eq!(cache.embedded.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.q.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.k.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.v.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.attn_out.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.projected.len(), cfg.seq_len * cfg.d_model);
        assert_eq!(cache.logits.len(), cfg.seq_len * cfg.vocab_size);
    }

    #[test]
    fn test_forward_deterministic() {
        let cfg = SWAConfig::test_config();
        let params = SWAParams::init(&cfg, 42);
        let input_ids: Vec<usize> = (0..cfg.seq_len).collect();
        let target_ids: Vec<usize> = (1..=cfg.seq_len).map(|t| t % cfg.vocab_size).collect();

        let (loss1, _) = forward(&params, &cfg, &input_ids, &target_ids);
        let (loss2, _) = forward(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(loss1, loss2, "Forward pass should be deterministic");
    }
}
