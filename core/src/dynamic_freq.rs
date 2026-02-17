/// Dynamic Frequency Scheduling — learned gates for CMS level activation.
///
/// Fixed schedule: step % C == 0 (modular arithmetic, current default).
/// Learned schedule: sigmoid gate per level decides when to fire based on
/// the current input embedding, not just the step counter.
///
/// The gate follows the existing w_alpha/w_theta/w_eta pattern:
///   freq_gate_l = sigmoid(embedded_mean @ w_freq_l + b_freq_l)
///   active = freq_gate_l > threshold  (hard decision, straight-through backward)
///   Level 0 is ALWAYS forced active regardless of gate value.
///
/// Source: NL_Hecate extension (no paper equation). Design follows existing
/// gate patterns from Titans (Eqs 12-15) and HOPE.

use crate::model::MemoryLevelParams;

// ── Configuration ─────────────────────────────────────────────────────

/// Which frequency scheduling strategy to use.
#[derive(Clone, Debug, PartialEq)]
pub enum FrequencySchedule {
    /// Current default: level fires when step % chunk_size == 0.
    Fixed,
    /// Learned sigmoid gate per level decides firing based on input.
    Learned(LearnedFreqConfig),
}

/// Configuration for learned frequency gates.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedFreqConfig {
    /// Hard gate threshold: level fires when sigmoid output > threshold.
    /// Default: 0.5.
    pub threshold: f32,
    /// Number of initial steps using fixed schedule before switching to learned.
    /// 0 = immediate learned scheduling (default).
    pub anneal_steps: usize,
}

impl Default for LearnedFreqConfig {
    fn default() -> Self {
        LearnedFreqConfig {
            threshold: 0.5,
            anneal_steps: 0,
        }
    }
}

// ── Forward Cache ─────────────────────────────────────────────────────

/// Cache for backward through frequency gates.
pub struct FreqGateCache {
    /// Per-level sigmoid outputs: [k].
    pub gate_values: Vec<f32>,
    /// Per-level pre-sigmoid values (for backward): [k].
    pub gate_pre: Vec<f32>,
    /// Mean-pooled input embedding: [d].
    pub embedded_mean: Vec<f32>,
}

// ── Forward Functions ─────────────────────────────────────────────────

/// Mean-pool embedded tokens across the sequence dimension.
/// Input: [seq_len * d], Output: [d].
pub fn mean_pool(embedded: &[f32], seq_len: usize, d: usize) -> Vec<f32> {
    assert_eq!(embedded.len(), seq_len * d);
    let mut mean = vec![0.0f32; d];
    let inv_s = 1.0 / seq_len as f32;
    for t in 0..seq_len {
        let base = t * d;
        for j in 0..d {
            mean[j] += embedded[base + j];
        }
    }
    for j in 0..d {
        mean[j] *= inv_s;
    }
    mean
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute frequency gates for all levels.
///
/// For each level l:
///   pre_l = dot(embedded_mean, w_freq_l) + b_freq_l
///   gate_l = sigmoid(pre_l)
///
/// w_freq for level l is level_params[l].w_freq (length d).
/// b_freq for level l is level_params[l].b_freq (length 1).
pub fn compute_freq_gates(
    embedded_mean: &[f32],
    level_params: &[MemoryLevelParams],
    k: usize,
    d: usize,
) -> FreqGateCache {
    assert_eq!(embedded_mean.len(), d);
    assert_eq!(level_params.len(), k);

    let mut gate_values = Vec::with_capacity(k);
    let mut gate_pre = Vec::with_capacity(k);

    for l in 0..k {
        let w = &level_params[l].w_freq;
        let b = &level_params[l].b_freq;
        assert_eq!(w.len(), d, "w_freq[{l}] must have length d={d}, got {}", w.len());
        assert_eq!(b.len(), 1, "b_freq[{l}] must have length 1, got {}", b.len());

        // pre = embedded_mean @ w_freq + b_freq
        let mut pre = b[0];
        for j in 0..d {
            pre += embedded_mean[j] * w[j];
        }
        gate_pre.push(pre);
        gate_values.push(sigmoid(pre));
    }

    FreqGateCache {
        gate_values,
        gate_pre,
        embedded_mean: embedded_mean.to_vec(),
    }
}

/// Apply hard threshold to gate values. Level 0 is always forced active.
/// Returns Vec<bool> of length k.
pub fn apply_threshold(cache: &FreqGateCache, threshold: f32) -> Vec<bool> {
    let k = cache.gate_values.len();
    let mut active = Vec::with_capacity(k);
    for l in 0..k {
        if l == 0 {
            active.push(true); // Level 0 always active (spec invariant)
        } else {
            active.push(cache.gate_values[l] > threshold);
        }
    }
    active
}

/// Whether to use fixed schedule at this step (annealing period).
pub fn should_anneal(step: usize, anneal_steps: usize) -> bool {
    step < anneal_steps
}

// ── Backward Functions ────────────────────────────────────────────────

/// Gradients for frequency gate parameters at one level.
pub struct FreqGrads {
    /// Gradient for w_freq: [d].
    pub d_w_freq: Vec<f32>,
    /// Gradient for b_freq: [1].
    pub d_b_freq: Vec<f32>,
}

/// Backward through frequency gates using straight-through estimator.
///
/// The hard threshold is non-differentiable. We use the straight-through
/// estimator: forward uses hard threshold, backward passes gradient through
/// sigmoid as if the threshold weren't there.
///
/// For each level l:
///   d_pre_l = d_gate_values[l] * sigmoid'(pre_l)
///           = d_gate_values[l] * gate_values[l] * (1 - gate_values[l])
///   d_w_freq_l = d_pre_l * embedded_mean   (outer product, but pre is scalar)
///   d_b_freq_l = d_pre_l
///   d_embedded_mean += d_pre_l * w_freq_l
///
/// `d_gate_values`: upstream gradient w.r.t. each level's gate activation.
/// This is the surrogate signal — how much the loss changes per unit gate change.
pub fn freq_gate_backward(
    d_gate_values: &[f32],
    cache: &FreqGateCache,
    level_params: &[MemoryLevelParams],
    k: usize,
    d: usize,
) -> (Vec<FreqGrads>, Vec<f32>) {
    assert_eq!(d_gate_values.len(), k);
    assert_eq!(cache.gate_values.len(), k);
    assert_eq!(cache.embedded_mean.len(), d);

    let mut grads = Vec::with_capacity(k);
    let mut d_embedded_mean = vec![0.0f32; d];

    for l in 0..k {
        let g = cache.gate_values[l];
        // Straight-through: d_pre = d_gate * sigmoid_deriv
        let d_pre = d_gate_values[l] * g * (1.0 - g);

        // d_w_freq = d_pre * embedded_mean
        let mut d_w_freq = vec![0.0f32; d];
        for j in 0..d {
            d_w_freq[j] = d_pre * cache.embedded_mean[j];
        }

        // d_b_freq = d_pre
        let d_b_freq = vec![d_pre];

        // d_embedded_mean += d_pre * w_freq
        let w = &level_params[l].w_freq;
        for j in 0..d {
            d_embedded_mean[j] += d_pre * w[j];
        }

        grads.push(FreqGrads { d_w_freq, d_b_freq });
    }

    (grads, d_embedded_mean)
}

/// Compute surrogate gradient signal for frequency gates.
///
/// For each level, we need: "how much did this level's activation affect loss?"
/// We use the norm of the level's memory output (y_per_level) weighted by
/// the upstream gradient (d_y_combined) as a proxy signal.
///
/// If a level was active and contributed a lot to the output, the gradient
/// encourages the gate to stay open. If inactive (gate < threshold), the
/// gradient encourages the gate to open if it would have helped.
///
/// For active levels: d_gate = +||d_y ⊙ y_level|| (positive = keep firing)
/// For inactive levels: d_gate = +small_positive (encourage exploration)
pub fn compute_gate_surrogate(
    y_per_level: &[Vec<f32>],
    d_y_combined: &[f32],
    active_levels: &[bool],
    k: usize,
    s_d: usize,
) -> Vec<f32> {
    let mut d_gate = vec![0.0f32; k];
    for l in 0..k {
        if active_levels[l] {
            // Dot product of d_y_combined and y_per_level[l] — measures how much
            // this level's output aligns with the loss gradient direction.
            let mut dot = 0.0f32;
            for i in 0..s_d {
                dot += d_y_combined[i] * y_per_level[l][i];
            }
            // Positive dot means this level helped reduce loss → keep gate open.
            // We want gradient that pushes gate toward 1 when helpful.
            d_gate[l] = -dot; // negative because d_loss/d_gate: more gate = less loss → negative
        } else {
            // Inactive level: small positive gradient to encourage exploration.
            // This is deliberately small — we don't want to force levels to fire
            // without evidence that it would help.
            d_gate[l] = 0.01;
        }
    }
    d_gate
}

// ── Bias Initialization ───────────────────────────────────────────────

/// Default b_freq bias per level index.
/// Higher levels start with lower bias (less likely to fire), matching
/// the fixed schedule's geometric spacing.
pub fn default_b_freq(level: usize) -> f32 {
    match level {
        0 => 0.0,   // sigmoid(0) = 0.5, but Level 0 is forced active anyway
        1 => -1.0,  // sigmoid(-1) ≈ 0.27 — fires ~27% of the time initially
        2 => -2.0,  // sigmoid(-2) ≈ 0.12
        3 => -3.0,  // sigmoid(-3) ≈ 0.05
        _ => -2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MemoryLevelParams;
    use crate::tensor::SimpleRng;

    fn make_level_params_with_freq(d: usize, k: usize) -> Vec<MemoryLevelParams> {
        let mut rng = SimpleRng::new(42);
        let mut levels = Vec::with_capacity(k);
        for l in 0..k {
            let mut p = MemoryLevelParams::init(d, &mut rng, 3.0, -4.6, 1.5);
            // Initialize w_freq with small random values
            p.w_freq = vec![0.0f32; d];
            rng.fill_uniform(&mut p.w_freq, 0.1);
            p.b_freq = vec![default_b_freq(l)];
            levels.push(p);
        }
        levels
    }

    #[test]
    fn test_mean_pool() {
        let d = 4;
        let s = 3;
        // 3 tokens, d=4: [1,2,3,4], [5,6,7,8], [9,10,11,12]
        let embedded: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mean = mean_pool(&embedded, s, d);
        assert_eq!(mean.len(), d);
        assert!((mean[0] - 5.0).abs() < 1e-6); // (1+5+9)/3
        assert!((mean[1] - 6.0).abs() < 1e-6); // (2+6+10)/3
        assert!((mean[2] - 7.0).abs() < 1e-6);
        assert!((mean[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_freq_gate_basic() {
        let d = 4;
        let k = 2;
        let levels = make_level_params_with_freq(d, k);
        let embedded_mean = vec![0.5f32; d];

        let cache = compute_freq_gates(&embedded_mean, &levels, k, d);
        assert_eq!(cache.gate_values.len(), k);
        assert_eq!(cache.gate_pre.len(), k);
        // All gate values should be in (0, 1)
        for &g in &cache.gate_values {
            assert!(g > 0.0 && g < 1.0, "gate value {g} not in (0,1)");
        }
    }

    #[test]
    fn test_gate_sigmoid_range() {
        let d = 8;
        let k = 4;
        let levels = make_level_params_with_freq(d, k);
        let embedded_mean = vec![1.0f32; d];

        let cache = compute_freq_gates(&embedded_mean, &levels, k, d);
        for l in 0..k {
            let g = cache.gate_values[l];
            assert!(g > 0.0 && g < 1.0, "Level {l}: gate={g} out of (0,1)");
        }
    }

    #[test]
    fn test_threshold_level0_always_active() {
        let d = 4;
        let k = 3;
        let mut levels = make_level_params_with_freq(d, k);
        // Set all biases very negative so all gates output near 0
        for l in 0..k {
            levels[l].b_freq = vec![-100.0];
            levels[l].w_freq = vec![0.0f32; d];
        }
        let embedded_mean = vec![0.0f32; d];

        let cache = compute_freq_gates(&embedded_mean, &levels, k, d);
        let active = apply_threshold(&cache, 0.5);

        // Level 0 must always be true
        assert!(active[0], "Level 0 must always be active");
        // Other levels should be false with -100 bias
        assert!(!active[1], "Level 1 should be inactive");
        assert!(!active[2], "Level 2 should be inactive");
    }

    #[test]
    fn test_config_defaults() {
        let cfg = LearnedFreqConfig::default();
        assert_eq!(cfg.threshold, 0.5);
        assert_eq!(cfg.anneal_steps, 0);
    }

    #[test]
    fn test_anneal_schedule() {
        assert!(should_anneal(0, 100));
        assert!(should_anneal(99, 100));
        assert!(!should_anneal(100, 100));
        assert!(!should_anneal(101, 100));
        assert!(!should_anneal(0, 0)); // anneal_steps=0 means never anneal
    }

    #[test]
    fn test_freq_gate_backward_shapes() {
        let d = 4;
        let k = 2;
        let levels = make_level_params_with_freq(d, k);
        let embedded_mean = vec![0.5f32; d];

        let cache = compute_freq_gates(&embedded_mean, &levels, k, d);
        let d_gate_values = vec![1.0f32; k];

        let (grads, d_emb) = freq_gate_backward(&d_gate_values, &cache, &levels, k, d);
        assert_eq!(grads.len(), k);
        for g in &grads {
            assert_eq!(g.d_w_freq.len(), d);
            assert_eq!(g.d_b_freq.len(), 1);
        }
        assert_eq!(d_emb.len(), d);
    }

    #[test]
    fn test_straight_through_nonzero() {
        let d = 4;
        let k = 2;
        let levels = make_level_params_with_freq(d, k);
        let embedded_mean = vec![0.5f32; d];

        let cache = compute_freq_gates(&embedded_mean, &levels, k, d);
        let d_gate_values = vec![1.0f32; k];

        let (grads, d_emb) = freq_gate_backward(&d_gate_values, &cache, &levels, k, d);

        // At least some gradients should be non-zero
        let w_norm: f32 = grads.iter()
            .flat_map(|g| g.d_w_freq.iter())
            .map(|x| x * x)
            .sum();
        assert!(w_norm > 1e-10, "w_freq gradients should be non-zero");

        let emb_norm: f32 = d_emb.iter().map(|x| x * x).sum();
        assert!(emb_norm > 1e-10, "d_embedded_mean should be non-zero");
    }

    #[test]
    fn test_default_b_freq_ordering() {
        // Higher levels should have more negative bias (fire less often)
        assert!(default_b_freq(0) > default_b_freq(1));
        assert!(default_b_freq(1) > default_b_freq(2));
        assert!(default_b_freq(2) > default_b_freq(3));
    }
}
