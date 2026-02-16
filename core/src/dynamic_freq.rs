/// Dynamic Frequency Scheduling for CMS levels.
///
/// Instead of fixed periodic scheduling (step % chunk_size == 0),
/// learned gates decide which CMS levels fire based on input data.
/// Gate parameters are outer-loop params trained via Enzyme AD.
///
/// Level 0 is ALWAYS active (non-negotiable, per spec).
/// Higher levels have learned gates: sigmoid(w_freq @ mean_embedding + b_freq).
/// The binary fire/no-fire decision uses a temperature-controlled sigmoid
/// with optional hard thresholding for inference.
///
/// Source: HOPE (2512.24695) Section 7.1, extended with learned gating.

use crate::tensor::SimpleRng;

/// Per-level frequency gate parameters (outer_loop_param lifetime).
///
/// Each level > 0 has a linear projection from d_model to a scalar gate value.
/// Layout:
///   w_freq: [d_model]  — projection weights
///   b_freq: [1]        — bias (initialized to match fixed-schedule duty cycle)
#[derive(Clone, Debug)]
pub struct FreqGateParams {
    /// Projection weights: [d_model]
    pub w_freq: Vec<f32>,
    /// Bias scalar: [1]
    pub b_freq: Vec<f32>,
}

impl FreqGateParams {
    /// Initialize gate params for a given level.
    ///
    /// `b_init` controls the initial duty cycle via sigmoid(b_init):
    ///   Level 1 (chunk_size=8):   b_init ≈ -2.2  → sigmoid ≈ 0.10 (fire ~10% ≈ 1/8-ish)
    ///   Level 2 (chunk_size=64):  b_init ≈ -4.1  → sigmoid ≈ 0.016 (fire ~1.6% ≈ 1/64)
    ///   Level 3 (chunk_size=512): b_init ≈ -6.2  → sigmoid ≈ 0.002 (fire ~0.2% ≈ 1/512)
    pub fn init(d: usize, rng: &mut SimpleRng, b_init: f32) -> Self {
        let scale = (1.0 / d as f32).sqrt() * 0.01; // Small init so gates start near bias
        let mut w_freq = vec![0.0f32; d];
        rng.fill_uniform(&mut w_freq, scale);
        FreqGateParams {
            w_freq,
            b_freq: vec![b_init],
        }
    }

    /// Zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(d: usize) -> Self {
        FreqGateParams {
            w_freq: vec![0.0f32; d],
            b_freq: vec![0.0f32; 1],
        }
    }

    /// Total parameter count.
    pub fn num_params(&self) -> usize {
        self.w_freq.len() + 1
    }

    /// Apply outer-loop weight update: param -= lr * grad.
    pub fn apply_weight_gradients(&mut self, grads: &FreqGateParams, lr: f32) {
        for i in 0..self.w_freq.len() {
            self.w_freq[i] -= lr * grads.w_freq[i];
        }
        self.b_freq[0] -= lr * grads.b_freq[0];
    }

    /// Element-wise accumulate: self += other.
    pub fn accumulate(&mut self, other: &FreqGateParams) {
        for i in 0..self.w_freq.len() {
            self.w_freq[i] += other.w_freq[i];
        }
        self.b_freq[0] += other.b_freq[0];
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for &x in &self.w_freq {
            sum += x * x;
        }
        sum += self.b_freq[0] * self.b_freq[0];
        sum.sqrt()
    }
}

/// Configuration for dynamic frequency scheduling.
#[derive(Clone, Debug)]
pub struct DynamicFreqConfig {
    /// Temperature for sigmoid gating. Lower = sharper decisions.
    /// Default: 1.0. At inference, can be set very low for hard gating.
    pub temperature: f32,
    /// Threshold for hard gating (gate > threshold → fire).
    /// Used when `hard_gating` is true. Default: 0.5.
    pub threshold: f32,
    /// Whether to use hard thresholding (for inference/testing).
    /// Default: false (soft gating for differentiable training).
    pub hard_gating: bool,
    /// Minimum gate value floor. Prevents complete shutdown of a level.
    /// Default: 0.0 (no floor). Set to e.g. 0.01 to ensure some gradient flow.
    pub min_gate: f32,
}

impl Default for DynamicFreqConfig {
    fn default() -> Self {
        DynamicFreqConfig {
            temperature: 1.0,
            threshold: 0.5,
            hard_gating: false,
            min_gate: 0.0,
        }
    }
}

/// Frequency Scheduler: holds per-level gate parameters.
///
/// Level 0 has no gate (always active). Levels 1..k each have FreqGateParams.
/// So `gates` has length k-1, where gates[i] corresponds to level i+1.
#[derive(Clone, Debug)]
pub struct FrequencyScheduler {
    /// Per-level gate params. Length = k - 1 (level 0 has no gate).
    pub gates: Vec<FreqGateParams>,
    pub config: DynamicFreqConfig,
}

/// Cache for dynamic frequency scheduling forward pass.
/// Stores intermediate values needed for backward.
#[derive(Clone, Debug)]
pub struct FreqSchedulerCache {
    /// Mean embedding used as gate input: [d_model]
    pub mean_embedding: Vec<f32>,
    /// Pre-sigmoid logits per level (levels 1..k): [k-1]
    pub gate_logits: Vec<f32>,
    /// Post-sigmoid gate values per level (levels 1..k): [k-1]
    pub gate_values: Vec<f32>,
}

/// Default bias init per level to match fixed-schedule duty cycle.
/// sigmoid(b) ≈ 1/chunk_size[level].
pub fn default_freq_bias(_level: usize, chunk_size: usize) -> f32 {
    // We want sigmoid(b) ≈ 1/chunk_size
    // sigmoid(b) = 1 / (1 + exp(-b))
    // So b = ln(1/chunk_size / (1 - 1/chunk_size)) = ln(1/(chunk_size - 1))
    // = -ln(chunk_size - 1)
    if chunk_size <= 1 {
        0.0 // Level 0, always active
    } else {
        -((chunk_size - 1) as f32).ln()
    }
}

impl FrequencyScheduler {
    /// Create a new scheduler for k CMS levels.
    /// `chunk_sizes` is used to compute initial bias values.
    pub fn new(k: usize, d: usize, chunk_sizes: &[usize], seed: u64, config: DynamicFreqConfig) -> Self {
        assert!(k >= 1);
        assert_eq!(chunk_sizes.len(), k);

        let mut gates = Vec::with_capacity(k.saturating_sub(1));
        for level in 1..k {
            let mut rng = SimpleRng::new(seed.wrapping_add(2000 + level as u64 * 300));
            let b_init = default_freq_bias(level, chunk_sizes[level]);
            gates.push(FreqGateParams::init(d, &mut rng, b_init));
        }

        FrequencyScheduler { gates, config }
    }

    /// Number of levels this scheduler handles (k).
    pub fn k(&self) -> usize {
        self.gates.len() + 1
    }

    /// Create zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(k: usize, d: usize) -> Self {
        let gates = (0..k.saturating_sub(1)).map(|_| FreqGateParams::zeros_like(d)).collect();
        FrequencyScheduler {
            gates,
            config: DynamicFreqConfig::default(),
        }
    }

    /// Total parameter count across all gates.
    pub fn num_params(&self) -> usize {
        self.gates.iter().map(|g| g.num_params()).sum()
    }

    /// Apply outer-loop weight update.
    pub fn apply_weight_gradients(&mut self, grads: &FrequencyScheduler, lr: f32) {
        for (gate, grad) in self.gates.iter_mut().zip(grads.gates.iter()) {
            gate.apply_weight_gradients(grad, lr);
        }
    }

    /// Compute gate values for all levels given input embeddings.
    ///
    /// `embedded`: [seq_len, d_model] — current input embeddings.
    /// Returns (active_levels: Vec<bool>, gate_values: Vec<f32>, cache: FreqSchedulerCache).
    ///
    /// Level 0 is always active (gate_value = 1.0).
    /// Levels 1..k use learned sigmoid gates on the mean embedding.
    pub fn compute_gates(
        &self,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
    ) -> (Vec<f32>, FreqSchedulerCache) {
        let k = self.k();

        // Compute mean embedding across sequence positions
        let mut mean_emb = vec![0.0f32; d];
        for t in 0..seq_len {
            for dd in 0..d {
                mean_emb[dd] += embedded[t * d + dd];
            }
        }
        if seq_len > 0 {
            let inv_s = 1.0 / seq_len as f32;
            for dd in 0..d {
                mean_emb[dd] *= inv_s;
            }
        }

        // Compute gate values for levels 1..k
        let mut gate_logits = Vec::with_capacity(k - 1);
        let mut gate_values = Vec::with_capacity(k);

        // Level 0: always 1.0
        gate_values.push(1.0);

        for level_idx in 0..self.gates.len() {
            let gate = &self.gates[level_idx];

            // logit = w_freq . mean_emb + b_freq
            let mut logit = gate.b_freq[0];
            for dd in 0..d {
                logit += gate.w_freq[dd] * mean_emb[dd];
            }

            // Apply temperature scaling
            let scaled_logit = logit / self.config.temperature;

            // Sigmoid gate
            let gate_val = 1.0 / (1.0 + (-scaled_logit).exp());

            // Apply min_gate floor
            let gate_val = gate_val.max(self.config.min_gate);

            gate_logits.push(logit);
            gate_values.push(gate_val);
        }

        let cache = FreqSchedulerCache {
            mean_embedding: mean_emb,
            gate_logits,
            gate_values: gate_values.clone(),
        };

        (gate_values, cache)
    }

    /// Convert soft gate values to boolean active_levels for Pulse.
    /// With hard_gating: threshold comparison.
    /// Without hard_gating: always active (soft gating applied to outputs instead).
    pub fn gate_to_active(&self, gate_values: &[f32]) -> Vec<bool> {
        let mut active = Vec::with_capacity(gate_values.len());
        for (i, &gv) in gate_values.iter().enumerate() {
            if i == 0 {
                active.push(true); // Level 0 always active
            } else if self.config.hard_gating {
                active.push(gv > self.config.threshold);
            } else {
                // Soft gating: all levels "active" but output scaled by gate
                active.push(true);
            }
        }
        active
    }

    /// Backward pass for frequency gates.
    ///
    /// Given upstream gradients d_y_per_level (from the CMS forward/backward),
    /// compute gradients for the gate parameters.
    ///
    /// `d_scale_per_level[i]` = derivative of loss w.r.t. gate_value[i].
    /// For soft gating: d_scale = sum over (s*d) of d_y_combined[j] * y_level[j].
    /// (Because y_combined = sum of gate[level] * y_level for dynamic, vs just sum for fixed.)
    pub fn backward(
        &self,
        cache: &FreqSchedulerCache,
        d_scale_per_level: &[f32],
        d: usize,
    ) -> FrequencyScheduler {
        let k = self.k();
        let mut grads = FrequencyScheduler::zeros_like(k, d);

        for level_idx in 0..self.gates.len() {
            let gate_val = cache.gate_values[level_idx + 1]; // +1 because gate_values includes level 0

            // d_gate / d_logit = gate_val * (1 - gate_val) / temperature
            let d_sigmoid = gate_val * (1.0 - gate_val) / self.config.temperature;

            // d_loss / d_logit = d_loss/d_gate * d_gate/d_logit
            let d_logit = d_scale_per_level[level_idx + 1] * d_sigmoid;

            // d_logit / d_w_freq = mean_embedding
            // d_logit / d_b_freq = 1.0
            for dd in 0..d {
                grads.gates[level_idx].w_freq[dd] = d_logit * cache.mean_embedding[dd];
            }
            grads.gates[level_idx].b_freq[0] = d_logit;
        }

        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_freq_bias() {
        // chunk_size=1 → always active, bias=0
        assert_eq!(default_freq_bias(0, 1), 0.0);

        // chunk_size=8 → sigmoid(b) ≈ 1/8
        let b = default_freq_bias(1, 8);
        let gate = 1.0 / (1.0 + (-b).exp());
        assert!((gate - 1.0/8.0).abs() < 0.01, "gate={gate}, expected ~0.125");

        // chunk_size=64
        let b = default_freq_bias(2, 64);
        let gate = 1.0 / (1.0 + (-b).exp());
        assert!((gate - 1.0/64.0).abs() < 0.005, "gate={gate}, expected ~0.0156");

        // chunk_size=512
        let b = default_freq_bias(3, 512);
        let gate = 1.0 / (1.0 + (-b).exp());
        assert!((gate - 1.0/512.0).abs() < 0.001, "gate={gate}, expected ~0.00195");
    }

    #[test]
    fn test_freq_gate_params_init() {
        let mut rng = SimpleRng::new(42);
        let d = 8;
        let params = FreqGateParams::init(d, &mut rng, -2.0);
        assert_eq!(params.w_freq.len(), d);
        assert_eq!(params.b_freq.len(), 1);
        assert!((params.b_freq[0] - (-2.0)).abs() < 1e-6);
        // Weights should be small
        for &w in &params.w_freq {
            assert!(w.abs() < 0.1, "Weight {w} too large");
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let d = 8;
        let k = 4;
        let chunk_sizes = vec![1, 8, 64, 512];
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());
        assert_eq!(sched.k(), 4);
        assert_eq!(sched.gates.len(), 3); // levels 1, 2, 3
    }

    #[test]
    fn test_scheduler_k1_no_gates() {
        let d = 8;
        let k = 1;
        let chunk_sizes = vec![1];
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());
        assert_eq!(sched.k(), 1);
        assert_eq!(sched.gates.len(), 0);
    }

    #[test]
    fn test_compute_gates_level0_always_one() {
        let d = 8;
        let k = 2;
        let chunk_sizes = vec![1, 8];
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

        let embedded = vec![0.1f32; 4 * d]; // seq_len=4
        let (gate_values, _cache) = sched.compute_gates(&embedded, 4, d);

        assert_eq!(gate_values.len(), 2);
        assert!((gate_values[0] - 1.0).abs() < 1e-6, "Level 0 gate must be 1.0");
    }

    #[test]
    fn test_gate_values_in_range() {
        let d = 8;
        let k = 4;
        let chunk_sizes = vec![1, 8, 64, 512];
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

        let embedded = vec![0.5f32; 8 * d]; // seq_len=8
        let (gate_values, _) = sched.compute_gates(&embedded, 8, d);

        for (i, &gv) in gate_values.iter().enumerate() {
            assert!(gv >= 0.0 && gv <= 1.0, "Gate[{i}]={gv} out of [0,1]");
        }
    }

    #[test]
    fn test_hard_gating() {
        let d = 8;
        let k = 2;
        let chunk_sizes = vec![1, 8];
        let config = DynamicFreqConfig {
            hard_gating: true,
            threshold: 0.5,
            ..Default::default()
        };
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);

        let embedded = vec![0.0f32; 4 * d]; // Zero input → gate ≈ sigmoid(bias)
        let (gate_values, _) = sched.compute_gates(&embedded, 4, d);
        let active = sched.gate_to_active(&gate_values);

        assert!(active[0]); // Level 0 always active
        // Level 1: bias ≈ -ln(7) ≈ -1.95 → sigmoid ≈ 0.125 < 0.5 → inactive
        assert!(!active[1], "Level 1 should be inactive with default bias for chunk_size=8");
    }

    #[test]
    fn test_zeros_like() {
        let sched = FrequencyScheduler::zeros_like(4, 8);
        assert_eq!(sched.gates.len(), 3);
        for gate in &sched.gates {
            assert!(gate.w_freq.iter().all(|&x| x == 0.0));
            assert!(gate.b_freq.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_apply_weight_gradients() {
        let d = 4;
        let k = 2;
        let chunk_sizes = vec![1, 8];
        let mut sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());
        let orig_b = sched.gates[0].b_freq[0];

        let mut grads = FrequencyScheduler::zeros_like(k, d);
        grads.gates[0].b_freq[0] = 1.0;
        grads.gates[0].w_freq[0] = 2.0;

        sched.apply_weight_gradients(&grads, 0.1);
        assert!((sched.gates[0].b_freq[0] - (orig_b - 0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_backward_produces_gradients() {
        let d = 4;
        let k = 3;
        let chunk_sizes = vec![1, 8, 64];
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, DynamicFreqConfig::default());

        // Forward
        let embedded = vec![0.3f32; 4 * d];
        let (gate_values, cache) = sched.compute_gates(&embedded, 4, d);
        assert_eq!(gate_values.len(), 3);

        // Backward: loss is more sensitive to level 1 than level 2
        let d_scale = vec![0.0, 1.0, 0.5]; // level 0 ignored, levels 1 and 2
        let grads = sched.backward(&cache, &d_scale, d);

        // Check that gradients are non-zero for levels with non-zero d_scale
        let g1_norm = grads.gates[0].norm();
        let g2_norm = grads.gates[1].norm();
        assert!(g1_norm > 0.0, "Level 1 gate grads should be non-zero");
        assert!(g2_norm > 0.0, "Level 2 gate grads should be non-zero");
    }

    #[test]
    fn test_temperature_effect() {
        let d = 8;
        let k = 2;
        let chunk_sizes = vec![1, 8];

        // Low temperature → sharper gate
        let config_low = DynamicFreqConfig { temperature: 0.1, ..Default::default() };
        let sched_low = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config_low);

        // High temperature → softer gate
        let config_high = DynamicFreqConfig { temperature: 10.0, ..Default::default() };
        let sched_high = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config_high);

        let embedded = vec![1.0f32; 4 * d];
        let (gv_low, _) = sched_low.compute_gates(&embedded, 4, d);
        let (gv_high, _) = sched_high.compute_gates(&embedded, 4, d);

        // Low temperature should be closer to 0 or 1 than high temperature
        let dist_low = (gv_low[1] - 0.5).abs();
        let dist_high = (gv_high[1] - 0.5).abs();
        assert!(dist_low >= dist_high, "Low temp should give sharper gates");
    }

    #[test]
    fn test_min_gate_floor() {
        let d = 8;
        let k = 2;
        let chunk_sizes = vec![1, 8];
        let config = DynamicFreqConfig { min_gate: 0.05, ..Default::default() };
        let sched = FrequencyScheduler::new(k, d, &chunk_sizes, 42, config);

        // With zero input, gate ≈ sigmoid(bias) ≈ 0.125
        // But if we force bias very negative...
        let mut sched_neg = sched.clone();
        sched_neg.gates[0].b_freq[0] = -20.0; // sigmoid(-20) ≈ 0

        let embedded = vec![0.0f32; 4 * d];
        let (gv, _) = sched_neg.compute_gates(&embedded, 4, d);
        assert!(gv[1] >= 0.05, "Gate value {} should be >= min_gate 0.05", gv[1]);
    }
}
