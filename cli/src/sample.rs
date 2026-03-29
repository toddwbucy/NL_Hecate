/// Token sampling from logits: temperature scaling, top-k filtering, argmax.
/// Rust port of Python _sample_token() from engine/generation.py.

/// Sample a single token from logits with temperature and optional top-k.
///
/// - `temperature <= 0.0` → deterministic argmax
/// - `top_k > 0` → filter to top-k logits before sampling
/// - Otherwise → temperature-scaled softmax over full vocab
pub fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> usize {
    let vocab = logits.len();
    if vocab == 0 {
        return 0;
    }

    // Greedy: argmax
    if temperature <= 0.0 {
        return argmax(logits);
    }

    // Build (index, logit) pairs
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // Top-k filtering
    if top_k > 0 && top_k < vocab {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
    }

    // Temperature-scaled softmax
    let max_logit = indexed.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let weighted: Vec<(usize, f32)> = indexed
        .iter()
        .map(|&(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
        .collect();
    let total: f32 = weighted.iter().map(|(_, w)| w).sum();

    if total <= 0.0 {
        return weighted.last().map(|(idx, _)| *idx).unwrap_or(0);
    }

    // Weighted random sampling
    let r = fastrand::f32() * total;
    let mut cumsum = 0.0f32;
    for &(idx, w) in &weighted {
        cumsum += w;
        if r < cumsum {
            return idx;
        }
    }
    weighted.last().map(|(idx, _)| *idx).unwrap_or(0)
}

/// Deterministic argmax over logits.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Choose a padding token that avoids special-token memory instability (CS-50).
/// Special tokens (id 0-2) cause inner loop divergence when 29+ identical tokens appear.
/// Uses the prompt's first regular token, or fallback to token 3.
pub fn safe_pad_token(prompt_tokens: &[usize], vocab_size: usize) -> usize {
    if let Some(&first) = prompt_tokens.first() {
        if first >= 3 {
            return first;
        }
    }
    3.min(vocab_size.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 0.5]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 0.0, 7.0]), 3);
    }

    #[test]
    fn argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn sample_greedy() {
        // temperature=0 should always return argmax
        let logits = vec![1.0, 5.0, 3.0, 2.0];
        for _ in 0..100 {
            assert_eq!(sample_token(&logits, 0.0, 0), 1);
            assert_eq!(sample_token(&logits, -1.0, 0), 1);
        }
    }

    #[test]
    fn sample_top_k_1_is_argmax() {
        // top_k=1 should always return the highest logit
        let logits = vec![1.0, 2.0, 10.0, 3.0];
        for _ in 0..100 {
            assert_eq!(sample_token(&logits, 1.0, 1), 2);
        }
    }

    #[test]
    fn sample_returns_valid_index() {
        let logits = vec![0.1; 100];
        for _ in 0..1000 {
            let idx = sample_token(&logits, 1.0, 0);
            assert!(idx < 100);
        }
    }

    #[test]
    fn sample_with_top_k() {
        // With top_k=2 on these logits, only indices 2 and 3 should be sampled
        let logits = vec![0.0, 0.0, 10.0, 9.0];
        for _ in 0..100 {
            let idx = sample_token(&logits, 1.0, 2);
            assert!(idx == 2 || idx == 3, "got index {idx}, expected 2 or 3");
        }
    }

    #[test]
    fn sample_high_temperature_spreads() {
        // Very high temperature should produce diverse outputs
        let logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut seen = [false; 5];
        for _ in 0..1000 {
            let idx = sample_token(&logits, 10.0, 0);
            seen[idx] = true;
        }
        // With uniform logits and high temp, we should hit most indices
        let count = seen.iter().filter(|&&s| s).count();
        assert!(count >= 4, "expected diverse sampling, only hit {count}/5");
    }

    #[test]
    fn safe_pad_basic() {
        // First token >= 3 should be used
        assert_eq!(safe_pad_token(&[42, 10, 5], 100), 42);
        // First token < 3 should fallback to 3
        assert_eq!(safe_pad_token(&[1, 10, 5], 100), 3);
        // Empty prompt should fallback to 3
        assert_eq!(safe_pad_token(&[], 100), 3);
        // Tiny vocab should clamp
        assert_eq!(safe_pad_token(&[], 3), 2);
    }
}
