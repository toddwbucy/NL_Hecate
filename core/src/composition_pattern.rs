/// CompositionPattern trait and persistent token infrastructure.
///
/// Formalizes the interface from specs/algorithms/composition_patterns/00_interface.md:
/// three patterns (MAC, MAG, MAL) answer the same question differently —
/// given memory M and attention A, how do you combine them?
///
/// This module provides:
/// - `AttentionKind` enum distinguishing full causal vs sliding window
/// - `CompositionPattern` trait with `attention_kind()` method
/// - `prepend_persistent()` function for learnable persistent tokens
/// - Compile-time checks: MAC requires FullCausal, MAG/MAL require SlidingWindow

use crate::model::CompositionKind;

// ── Attention Kind ───────────────────────────────────────────────────

/// Which attention type a composition pattern requires.
///
/// MAC assembles memory context into the attention input and needs
/// full causal attention over the entire assembled sequence.
/// MAG/MAL operate on local windows and require sliding window attention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Full causal attention over the entire assembled sequence (MAC).
    FullCausal,
    /// Sliding window attention over local context (MAG, MAL).
    SlidingWindow,
}

// ── CompositionPattern Trait ─────────────────────────────────────────

/// Trait formalizing the composition pattern interface.
///
/// All three patterns (MAC, MAG, MAL) implement this trait.
/// The actual forward/backward dispatch continues through enum-based
/// static dispatch (CompositionKind → mag/mal/mac modules), but this
/// trait formalizes the contract and enables compile-time checks.
pub trait CompositionPattern {
    /// Which attention type this pattern requires.
    fn attention_kind(&self) -> AttentionKind;
}

/// MAC: Memory As Context — memory provides context, attention processes
/// the assembled (memory + input) sequence with full causal attention.
pub struct MAC;

/// MAG: Memory As Gate — memory and attention run in parallel,
/// memory gates the attention output with sliding window attention.
pub struct MAG;

/// MAL: Memory As Layer — memory preprocesses input, attention processes
/// memory output with sliding window attention.
pub struct MAL;

impl CompositionPattern for MAC {
    fn attention_kind(&self) -> AttentionKind {
        AttentionKind::FullCausal
    }
}

impl CompositionPattern for MAG {
    fn attention_kind(&self) -> AttentionKind {
        AttentionKind::SlidingWindow
    }
}

impl CompositionPattern for MAL {
    fn attention_kind(&self) -> AttentionKind {
        AttentionKind::SlidingWindow
    }
}

/// Map CompositionKind enum to AttentionKind at runtime.
pub fn attention_kind_for(composition: CompositionKind) -> AttentionKind {
    match composition {
        CompositionKind::MAC => AttentionKind::FullCausal,
        CompositionKind::MAG => AttentionKind::SlidingWindow,
        CompositionKind::MAL => AttentionKind::SlidingWindow,
    }
}

// ── Persistent Tokens ────────────────────────────────────────────────

/// Prepend persistent tokens to the input sequence.
///
/// Persistent tokens are learnable, input-independent tokens (outer_loop_param)
/// that store task-level knowledge. They are shared across all composition
/// patterns and prepended to the input before processing.
///
/// # Arguments
/// * `x` — input tensor, shape `[seq_len, d_model]` (single-sequence, no batch dim)
/// * `persistent` — persistent token weights, shape `[n_persistent, d_model]`
///
/// # Returns
/// Concatenated tensor of shape `[n_persistent + seq_len, d_model]`
pub fn prepend_persistent(
    x: &[f32],
    seq_len: usize,
    persistent: &[f32],
    n_persistent: usize,
    d_model: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), seq_len * d_model, "x shape mismatch");
    assert_eq!(
        persistent.len(),
        n_persistent * d_model,
        "persistent shape mismatch"
    );

    let total_len = (n_persistent + seq_len) * d_model;
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(persistent);
    out.extend_from_slice(x);
    out
}

/// Backward pass for prepend_persistent: split gradients.
///
/// Given gradient w.r.t. the concatenated output `[n_persistent + seq_len, d_model]`,
/// returns (d_persistent, d_x).
pub fn prepend_persistent_backward(
    d_out: &[f32],
    n_persistent: usize,
    seq_len: usize,
    d_model: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(
        d_out.len(),
        (n_persistent + seq_len) * d_model,
        "d_out shape mismatch"
    );

    let split = n_persistent * d_model;
    let d_persistent = d_out[..split].to_vec();
    let d_x = d_out[split..].to_vec();
    (d_persistent, d_x)
}

// ── Validation ───────────────────────────────────────────────────────

/// Validate that a composition kind is compatible with the given window size.
///
/// MAC requires full causal attention: window_size >= 2 * seq_len
/// (so the assembled sequence including memory context is fully attended to).
/// MAG/MAL use sliding window and have no minimum beyond seq_len.
pub fn validate_attention_compatibility(
    composition: CompositionKind,
    window_size: usize,
    seq_len: usize,
) -> Result<(), String> {
    match composition {
        CompositionKind::MAC => {
            // MAC assembles [h_t | x] of length 2*seq_len, needs full causal.
            let required = seq_len
                .checked_mul(2)
                .ok_or_else(|| "seq_len too large for 2*seq_len window check".to_string())?;
            if window_size < required {
                return Err(format!(
                    "MAC requires window_size >= 2*seq_len for full causal attention \
                     over assembled context. Got window_size={}, seq_len={} (need >= {}).",
                    window_size, seq_len, required
                ));
            }
        }
        CompositionKind::MAG | CompositionKind::MAL => {
            // Sliding window — no special requirement beyond seq_len
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionKind mapping ────────────────────────────────────────

    #[test]
    fn test_mac_requires_full_causal() {
        assert_eq!(attention_kind_for(CompositionKind::MAC), AttentionKind::FullCausal);
    }

    #[test]
    fn test_mag_requires_sliding_window() {
        assert_eq!(attention_kind_for(CompositionKind::MAG), AttentionKind::SlidingWindow);
    }

    #[test]
    fn test_mal_requires_sliding_window() {
        assert_eq!(attention_kind_for(CompositionKind::MAL), AttentionKind::SlidingWindow);
    }

    // ── Trait implementations ────────────────────────────────────────

    #[test]
    fn test_mac_trait() {
        let mac = MAC;
        assert_eq!(mac.attention_kind(), AttentionKind::FullCausal);
    }

    #[test]
    fn test_mag_trait() {
        let mag = MAG;
        assert_eq!(mag.attention_kind(), AttentionKind::SlidingWindow);
    }

    #[test]
    fn test_mal_trait() {
        let mal = MAL;
        assert_eq!(mal.attention_kind(), AttentionKind::SlidingWindow);
    }

    // ── prepend_persistent ───────────────────────────────────────────

    #[test]
    fn test_prepend_persistent_basic() {
        let d = 4;
        let n_p = 2;
        let seq = 3;
        let persistent = vec![1.0; n_p * d]; // 2 tokens of all-1s
        let x = vec![2.0; seq * d]; // 3 tokens of all-2s

        let result = prepend_persistent(&x, seq, &persistent, n_p, d);
        assert_eq!(result.len(), (n_p + seq) * d);
        // First n_p*d elements are persistent
        assert!(result[..n_p * d].iter().all(|&v| v == 1.0));
        // Remaining are x
        assert!(result[n_p * d..].iter().all(|&v| v == 2.0));
    }

    #[test]
    fn test_prepend_persistent_zero_tokens() {
        let d = 4;
        let x = vec![3.0; 5 * d];
        let result = prepend_persistent(&x, 5, &[], 0, d);
        assert_eq!(result.len(), 5 * d);
        assert_eq!(result, x);
    }

    #[test]
    fn test_prepend_persistent_preserves_order() {
        let d = 2;
        let persistent = vec![10.0, 20.0, 30.0, 40.0]; // 2 tokens
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens

        let result = prepend_persistent(&x, 3, &persistent, 2, d);
        assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // ── prepend_persistent_backward ──────────────────────────────────

    #[test]
    fn test_backward_splits_correctly() {
        let d = 3;
        let n_p = 2;
        let seq = 4;
        let d_out: Vec<f32> = (0..((n_p + seq) * d)).map(|i| i as f32).collect();

        let (d_persistent, d_x) = prepend_persistent_backward(&d_out, n_p, seq, d);
        assert_eq!(d_persistent.len(), n_p * d);
        assert_eq!(d_x.len(), seq * d);
        // d_persistent = first 6 elements
        assert_eq!(d_persistent, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // d_x = remaining 12 elements
        assert_eq!(d_x, (6..18).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn test_backward_zero_persistent() {
        let d = 4;
        let seq = 3;
        let d_out = vec![1.0; seq * d];
        let (d_persistent, d_x) = prepend_persistent_backward(&d_out, 0, seq, d);
        assert!(d_persistent.is_empty());
        assert_eq!(d_x, d_out);
    }

    // ── Attention compatibility validation ────────────────────────────

    #[test]
    fn test_mac_valid_window() {
        // MAC needs window >= 2*seq_len
        assert!(validate_attention_compatibility(CompositionKind::MAC, 48, 24).is_ok());
    }

    #[test]
    fn test_mac_exact_window() {
        assert!(validate_attention_compatibility(CompositionKind::MAC, 48, 24).is_ok());
    }

    #[test]
    fn test_mac_insufficient_window() {
        let result = validate_attention_compatibility(CompositionKind::MAC, 16, 24);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("window_size >= 2*seq_len"));
    }

    #[test]
    fn test_mag_any_window() {
        // MAG uses sliding window, no special requirement
        assert!(validate_attention_compatibility(CompositionKind::MAG, 8, 24).is_ok());
    }

    #[test]
    fn test_mal_any_window() {
        assert!(validate_attention_compatibility(CompositionKind::MAL, 8, 24).is_ok());
    }

    // ── Round-trip: prepend then backward ────────────────────────────

    #[test]
    fn test_round_trip_gradient() {
        let d = 4;
        let n_p = 3;
        let seq = 5;
        let persistent = vec![1.0; n_p * d];
        let x = vec![2.0; seq * d];

        let combined = prepend_persistent(&x, seq, &persistent, n_p, d);
        // Gradient of 1 everywhere
        let d_out = vec![1.0; combined.len()];
        let (d_persistent, d_x) = prepend_persistent_backward(&d_out, n_p, seq, d);

        // All gradients should be 1.0 (identity-like)
        assert!(d_persistent.iter().all(|&v| v == 1.0));
        assert!(d_x.iter().all(|&v| v == 1.0));
        assert_eq!(d_persistent.len(), persistent.len());
        assert_eq!(d_x.len(), x.len());
    }
}
