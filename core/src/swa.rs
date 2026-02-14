/// Sliding Window Attention (SWA) — Rust reference implementation.
///
/// Per-query-position attention over a causal sliding window.
/// Multi-head: Q/K/V are [seq_len, num_heads * head_dim], reshaped internally.
///
/// This is the portable reference. CUDA kernel (Phase 2) will match this exactly.

use crate::tensor::softmax_f32;

/// SWA forward pass.
///
/// Inputs (all row-major flat slices):
///   q: [seq_len, num_heads * head_dim]  — queries
///   k: [seq_len, num_heads * head_dim]  — keys
///   v: [seq_len, num_heads * head_dim]  — values
///
/// Output:
///   out: [seq_len, num_heads * head_dim] — attention output
///
/// For each query position q_pos, attends to key positions in
/// [max(0, q_pos - window_size + 1) .. q_pos] (inclusive, causal).
///
/// Also returns:
///   attn_weights: [num_heads, seq_len, window_size] — softmax weights (for backward)
///
/// The scale factor is 1/sqrt(head_dim).
///
/// bf16 boundary notes (Phase 2 CUDA):
///   Q/K/V inputs and attn_weights will be stored in bf16 for memory bandwidth.
///   All accumulation (dot products, softmax, weighted sums) stays f32.
///   This Rust reference uses f32 throughout for FD-checkable gradients.
///   See `tensor::truncate_to_bf16` for the conversion helper.
pub fn swa_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    attn_weights: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) {
    let total_dim = num_heads * head_dim;
    debug_assert_eq!(q.len(), seq_len * total_dim);
    debug_assert_eq!(k.len(), seq_len * total_dim);
    debug_assert_eq!(v.len(), seq_len * total_dim);
    debug_assert_eq!(out.len(), seq_len * total_dim);
    debug_assert_eq!(attn_weights.len(), num_heads * seq_len * window_size);

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Zero output
    for x in out.iter_mut() { *x = 0.0; }
    for x in attn_weights.iter_mut() { *x = 0.0; }

    for h in 0..num_heads {
        let h_offset = h * head_dim;

        for q_pos in 0..seq_len {
            // Causal window: [win_start, q_pos] inclusive
            let win_start = if q_pos + 1 >= window_size { q_pos + 1 - window_size } else { 0 };
            let win_len = q_pos - win_start + 1; // always >= 1

            // Compute attention scores for this query position
            let mut scores = vec![f32::NEG_INFINITY; window_size];
            for w in 0..win_len {
                let k_pos = win_start + w;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_pos * total_dim + h_offset + d]
                         * k[k_pos * total_dim + h_offset + d];
                }
                scores[w] = dot * scale;
            }

            // Softmax over valid positions
            let aw_base = (h * seq_len + q_pos) * window_size;
            softmax_f32(&scores, &mut attn_weights[aw_base..aw_base + window_size], 1, window_size);

            // Weighted sum of values
            for w in 0..win_len {
                let k_pos = win_start + w;
                let weight = attn_weights[aw_base + w];
                for d in 0..head_dim {
                    out[q_pos * total_dim + h_offset + d] +=
                        weight * v[k_pos * total_dim + h_offset + d];
                }
            }
        }
    }
}

/// SWA backward pass — Rust reference implementation.
///
/// Computes dQ, dK, dV from d_attn_out and cached attn_weights/Q/K/V.
/// This is the exact logic from backward.rs Stage 3, extracted for dispatch.
///
/// All slices are [seq_len, num_heads * head_dim] row-major flat,
/// except attn_weights which is [num_heads, seq_len, window_size].
/// SWA backward — public for CUDA comparison tests.
#[allow(dead_code)]
pub fn swa_backward_rust(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn_weights: &[f32],
    d_attn_out: &[f32],
    d_q: &mut [f32],
    d_k: &mut [f32],
    d_v: &mut [f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) {
    let total_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let h_offset = h * head_dim;

        for q_pos in 0..seq_len {
            let win_start = if q_pos + 1 >= window_size { q_pos + 1 - window_size } else { 0 };
            let win_len = q_pos - win_start + 1;

            let aw_base = (h * seq_len + q_pos) * window_size;

            // d_attn_w[w] = sum_d d_attn_out[q,h,d] * V[k,h,d]
            let mut d_attn_w = vec![0.0f32; window_size];
            for w in 0..win_len {
                let k_pos = win_start + w;
                let mut sum = 0.0f32;
                for dd in 0..head_dim {
                    sum += d_attn_out[q_pos * total_dim + h_offset + dd]
                         * v[k_pos * total_dim + h_offset + dd];
                }
                d_attn_w[w] = sum;
            }

            // d_V[k,h,d] += attn_weights[w] * d_attn_out[q,h,d]
            for w in 0..win_len {
                let k_pos = win_start + w;
                let aw = attn_weights[aw_base + w];
                for dd in 0..head_dim {
                    d_v[k_pos * total_dim + h_offset + dd] +=
                        aw * d_attn_out[q_pos * total_dim + h_offset + dd];
                }
            }

            // Softmax backward: d_scores[i] = P[i] * (d_attn_w[i] - sum_j(P[j] * d_attn_w[j]))
            let mut dot_pw = 0.0f32;
            for w in 0..win_len {
                dot_pw += attn_weights[aw_base + w] * d_attn_w[w];
            }
            let mut d_scores = vec![0.0f32; window_size];
            for w in 0..win_len {
                d_scores[w] = attn_weights[aw_base + w] * (d_attn_w[w] - dot_pw);
            }

            // d_Q[q,h,d] += d_scores[w] * K[k,h,d] * scale
            // d_K[k,h,d] += d_scores[w] * Q[q,h,d] * scale
            for w in 0..win_len {
                let k_pos = win_start + w;
                let ds = d_scores[w] * scale;
                for dd in 0..head_dim {
                    d_q[q_pos * total_dim + h_offset + dd] +=
                        ds * k[k_pos * total_dim + h_offset + dd];
                    d_k[k_pos * total_dim + h_offset + dd] +=
                        ds * q[q_pos * total_dim + h_offset + dd];
                }
            }
        }
    }
}

/// Naive reference SWA for testing: single head, no multi-head reshaping.
/// q, k, v: [seq_len, dim]. Returns [seq_len, dim].
pub fn swa_forward_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    dim: usize,
    window_size: usize,
) {
    let scale = 1.0 / (dim as f32).sqrt();

    for x in out.iter_mut() { *x = 0.0; }

    for q_pos in 0..seq_len {
        let win_start = if q_pos + 1 >= window_size { q_pos + 1 - window_size } else { 0 };
        let win_len = q_pos - win_start + 1;

        // Compute scores
        let mut scores = vec![0.0f32; win_len];
        for w in 0..win_len {
            let k_pos = win_start + w;
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[q_pos * dim + d] * k[k_pos * dim + d];
            }
            scores[w] = dot * scale;
        }

        // Softmax
        let mut weights = vec![0.0f32; win_len];
        softmax_f32(&scores, &mut weights, 1, win_len);

        // Weighted sum
        for w in 0..win_len {
            let k_pos = win_start + w;
            for d in 0..dim {
                out[q_pos * dim + d] += weights[w] * v[k_pos * dim + d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swa_single_position() {
        // seq_len=1, window=1 → just copies V (attention is trivially 1.0)
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let k = vec![1.0f32, 0.0, 0.0, 1.0];
        let v = vec![0.5f32, 0.3, 0.1, 0.7];
        let mut out = vec![0.0f32; 4];

        swa_forward_single_head(&q, &k, &v, &mut out, 1, 4, 4);

        for i in 0..4 {
            assert!((out[i] - v[i]).abs() < 1e-6,
                "Position 0 should copy V, got out[{}]={}", i, out[i]);
        }
    }

    #[test]
    fn test_swa_causal_masking() {
        // seq_len=3, window=3. Position 0 can only see itself.
        // Position 1 sees [0,1]. Position 2 sees [0,1,2].
        let dim = 2;
        let seq_len = 3;
        // Make Q=K so attention scores depend on self-similarity
        let q = vec![1.0, 0.0,  0.0, 1.0,  1.0, 1.0f32];
        let k = q.clone();
        let v = vec![1.0, 0.0,  0.0, 1.0,  0.5, 0.5f32];
        let mut out = vec![0.0f32; 6];

        swa_forward_single_head(&q, &k, &v, &mut out, seq_len, dim, 3);

        // Position 0: only sees V[0] → out should be V[0]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);

        // Position 2: sees all 3 → weighted combination of V[0..3]
        // Just check it's not NaN and is a valid combination
        assert!(out[4].is_finite());
        assert!(out[5].is_finite());
    }

    #[test]
    fn test_swa_window_limiting() {
        // seq_len=4, window=2. Position 3 can only see [2, 3], NOT [0, 1].
        let dim = 2;
        let seq_len = 4;
        // Make all Q and K identical so attention is uniform within window
        let q = vec![1.0, 0.0,  1.0, 0.0,  1.0, 0.0,  1.0, 0.0f32];
        let k = q.clone();
        // V values: distinct per position
        let v = vec![10.0, 0.0,  20.0, 0.0,  30.0, 0.0,  40.0, 0.0f32];
        let mut out = vec![0.0f32; 8];

        swa_forward_single_head(&q, &k, &v, &mut out, seq_len, dim, 2);

        // Position 3: window=[2,3], uniform attention → avg(V[2], V[3])
        // With identical Q=K, attention weights should be uniform
        let expected_pos3 = (30.0 + 40.0) / 2.0;
        assert!((out[6] - expected_pos3).abs() < 1e-4,
            "Position 3 should average V[2:4], got {}, expected {}", out[6], expected_pos3);
    }

    #[test]
    fn test_swa_multi_head() {
        let num_heads = 2;
        let head_dim = 2;
        let seq_len = 2;
        let window_size = 2;
        let total_dim = num_heads * head_dim;

        // Q, K, V: [2, 4] (seq_len=2, total_dim=4)
        let q = vec![1.0, 0.0, 0.0, 1.0,
                     0.0, 1.0, 1.0, 0.0f32];
        let k = q.clone();
        let v = vec![0.5, 0.3, 0.1, 0.7,
                     0.2, 0.8, 0.4, 0.6f32];
        let mut out = vec![0.0f32; seq_len * total_dim];
        let mut attn_w = vec![0.0f32; num_heads * seq_len * window_size];

        swa_forward(&q, &k, &v, &mut out, &mut attn_w, seq_len, num_heads, head_dim, window_size);

        // Check output dimensions are filled and finite
        for i in 0..(seq_len * total_dim) {
            assert!(out[i].is_finite(), "out[{}] is not finite: {}", i, out[i]);
        }
        // Check attention weights sum to 1 for each (head, position)
        for h in 0..num_heads {
            for pos in 0..seq_len {
                let base = (h * seq_len + pos) * window_size;
                let sum: f32 = attn_w[base..base + window_size].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5,
                    "Attn weights for head {} pos {} sum to {}", h, pos, sum);
            }
        }
    }
}
