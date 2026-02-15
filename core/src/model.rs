/// SWA Transformer model configuration and parameters.
///
/// Track Zero-A: single-block SWA with no memory, no CMS, no inner loop.
/// All weight matrices are flat Vec<f32> in row-major layout.

use crate::tensor::SimpleRng;

/// Which memory update rule to use for the inner loop.
///
/// MIRAS Algorithm knob: selects the optimizer for memory updates.
/// - DeltaRule: GD without momentum (Titans Eq 34)
/// - TitansLMM: GD + momentum accumulator S (Titans Eqs 12-15, 32-33)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryRuleKind {
    DeltaRule,
    TitansLMM,
}

/// Model configuration — immutable after construction.
#[derive(Clone, Debug)]
pub struct SWAConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub window_size: usize,
    pub vocab_size: usize,
}

impl SWAConfig {
    /// Test configuration: tiny model for fast iteration.
    pub fn test_config() -> Self {
        SWAConfig {
            d_model: 64,
            num_heads: 4,
            head_dim: 16,  // d_model / num_heads
            seq_len: 24,
            window_size: 16,
            vocab_size: 256,
        }
    }
}

/// All learnable parameters — flat Vec<f32> for Enzyme compatibility.
///
/// Layout (row-major):
///   w_embed:  [vocab_size, d_model]
///   w_q:      [d_model, d_model]
///   w_k:      [d_model, d_model]
///   w_v:      [d_model, d_model]
///   w_o:      [d_model, d_model]
///   w_unembed:[d_model, vocab_size]
#[derive(Clone)]
pub struct SWAParams {
    pub w_embed: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_unembed: Vec<f32>,
}

impl SWAParams {
    /// Initialize with small random values using Xavier-like scaling.
    pub fn init(cfg: &SWAConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let v = cfg.vocab_size;

        let embed_scale = (1.0 / d as f32).sqrt();
        let proj_scale = (2.0 / (d + d) as f32).sqrt(); // Xavier for d→d
        let unembed_scale = (1.0 / d as f32).sqrt();

        let mut w_embed = vec![0.0f32; v * d];
        rng.fill_uniform(&mut w_embed, embed_scale);

        let mut w_q = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_q, proj_scale);

        let mut w_k = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_k, proj_scale);

        let mut w_v = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_v, proj_scale);

        let mut w_o = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_o, proj_scale);

        let mut w_unembed = vec![0.0f32; d * v];
        rng.fill_uniform(&mut w_unembed, unembed_scale);

        SWAParams { w_embed, w_q, w_k, w_v, w_o, w_unembed }
    }

    /// Create a zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(cfg: &SWAConfig) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        SWAParams {
            w_embed: vec![0.0f32; v * d],
            w_q: vec![0.0f32; d * d],
            w_k: vec![0.0f32; d * d],
            w_v: vec![0.0f32; d * d],
            w_o: vec![0.0f32; d * d],
            w_unembed: vec![0.0f32; d * v],
        }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.w_embed.len() + self.w_q.len() + self.w_k.len()
            + self.w_v.len() + self.w_o.len() + self.w_unembed.len()
    }

    /// Apply SGD: param -= lr * grad for all weight matrices.
    pub fn sgd_step(&mut self, grads: &SWAParams, lr: f32) {
        fn step(param: &mut [f32], grad: &[f32], lr: f32) {
            for i in 0..param.len() {
                param[i] -= lr * grad[i];
            }
        }
        step(&mut self.w_embed, &grads.w_embed, lr);
        step(&mut self.w_q, &grads.w_q, lr);
        step(&mut self.w_k, &grads.w_k, lr);
        step(&mut self.w_v, &grads.w_v, lr);
        step(&mut self.w_o, &grads.w_o, lr);
        step(&mut self.w_unembed, &grads.w_unembed, lr);
    }
}

// ── Memory Level Parameters ──────────────────────────────────────────

/// Per-level memory weights — the primitive for CMS frequency levels.
///
/// Each CMS level (0..k) has its own independent set of projections and gates.
/// For k=1 (Zero-B), MAGParams.levels has length 1.
/// For k=2 (Phase 2), MAGParams.levels has length 2.
///
/// Layout (row-major):
///   w_k_mem:  [d, d] memory key projection
///   w_v_mem:  [d, d] memory value projection
///   w_q_mem:  [d, d] memory query projection
///   w_alpha:  [2*d]  retention gate weights (input: concat(k,v))
///   b_alpha:  [1]    retention gate bias
///   w_theta:  [2*d]  learning rate gate weights
///   b_theta:  [1]    learning rate gate bias
///   w_eta:    [2*d]  momentum gate weights (TitansLMM only; zero-init for DeltaRule)
///   b_eta:    [1]    momentum gate bias (TitansLMM only)
#[derive(Clone)]
pub struct MemoryLevelParams {
    pub w_k_mem: Vec<f32>,
    pub w_v_mem: Vec<f32>,
    pub w_q_mem: Vec<f32>,
    pub w_alpha: Vec<f32>,
    pub b_alpha: Vec<f32>,
    pub w_theta: Vec<f32>,
    pub b_theta: Vec<f32>,
    pub w_eta: Vec<f32>,
    pub b_eta: Vec<f32>,
}

impl MemoryLevelParams {
    /// Initialize one level's memory weights.
    /// `b_alpha_init` and `b_theta_init` control gate bias initialization:
    ///   Level 0: b_alpha=3.0 (sigmoid≈0.95), b_theta=-4.6 (softplus≈0.01)
    ///   Level 1: b_alpha=4.0 (sigmoid≈0.98), b_theta=-5.6 (softplus≈0.004)
    /// `b_eta_init`: momentum gate bias (only used by TitansLMM; still allocated for uniform struct).
    pub fn init(d: usize, rng: &mut SimpleRng, b_alpha_init: f32, b_theta_init: f32, b_eta_init: f32) -> Self {
        let proj_scale = (2.0 / (d + d) as f32).sqrt();
        let gate_scale = (1.0 / (2 * d) as f32).sqrt();

        let mut w_k_mem = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_k_mem, proj_scale);

        let mut w_v_mem = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_v_mem, proj_scale);

        let mut w_q_mem = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_q_mem, proj_scale);

        let mut w_alpha = vec![0.0f32; 2 * d];
        rng.fill_uniform(&mut w_alpha, gate_scale);

        let mut w_theta = vec![0.0f32; 2 * d];
        rng.fill_uniform(&mut w_theta, gate_scale);

        let mut w_eta = vec![0.0f32; 2 * d];
        rng.fill_uniform(&mut w_eta, gate_scale);

        let b_alpha = vec![b_alpha_init];
        let b_theta = vec![b_theta_init];
        let b_eta = vec![b_eta_init];

        MemoryLevelParams { w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta, w_eta, b_eta }
    }

    /// Create zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(d: usize) -> Self {
        MemoryLevelParams {
            w_k_mem: vec![0.0f32; d * d],
            w_v_mem: vec![0.0f32; d * d],
            w_q_mem: vec![0.0f32; d * d],
            w_alpha: vec![0.0f32; 2 * d],
            b_alpha: vec![0.0f32; 1],
            w_theta: vec![0.0f32; 2 * d],
            b_theta: vec![0.0f32; 1],
            w_eta: vec![0.0f32; 2 * d],
            b_eta: vec![0.0f32; 1],
        }
    }

    /// Total number of parameters in this level.
    pub fn num_params(&self) -> usize {
        self.w_k_mem.len() + self.w_v_mem.len() + self.w_q_mem.len()
            + self.w_alpha.len() + self.b_alpha.len()
            + self.w_theta.len() + self.b_theta.len()
            + self.w_eta.len() + self.b_eta.len()
    }

    /// Apply SGD: param -= lr * grad for all weight matrices.
    pub fn sgd_step(&mut self, grads: &MemoryLevelParams, lr: f32) {
        fn step(param: &mut [f32], grad: &[f32], lr: f32) {
            for i in 0..param.len() {
                param[i] -= lr * grad[i];
            }
        }
        step(&mut self.w_k_mem, &grads.w_k_mem, lr);
        step(&mut self.w_v_mem, &grads.w_v_mem, lr);
        step(&mut self.w_q_mem, &grads.w_q_mem, lr);
        step(&mut self.w_alpha, &grads.w_alpha, lr);
        step(&mut self.b_alpha, &grads.b_alpha, lr);
        step(&mut self.w_theta, &grads.w_theta, lr);
        step(&mut self.b_theta, &grads.b_theta, lr);
        step(&mut self.w_eta, &grads.w_eta, lr);
        step(&mut self.b_eta, &grads.b_eta, lr);
    }

    /// Element-wise accumulate: self += other.
    pub fn accumulate(&mut self, other: &MemoryLevelParams) {
        fn acc(dst: &mut [f32], src: &[f32]) {
            for i in 0..dst.len() {
                dst[i] += src[i];
            }
        }
        acc(&mut self.w_k_mem, &other.w_k_mem);
        acc(&mut self.w_v_mem, &other.w_v_mem);
        acc(&mut self.w_q_mem, &other.w_q_mem);
        acc(&mut self.w_alpha, &other.w_alpha);
        acc(&mut self.b_alpha, &other.b_alpha);
        acc(&mut self.w_theta, &other.w_theta);
        acc(&mut self.b_theta, &other.b_theta);
        acc(&mut self.w_eta, &other.w_eta);
        acc(&mut self.b_eta, &other.b_eta);
    }

    /// Frobenius norm across all weight matrices.
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for v in [&self.w_k_mem, &self.w_v_mem, &self.w_q_mem,
                   &self.w_alpha, &self.b_alpha, &self.w_theta, &self.b_theta,
                   &self.w_eta, &self.b_eta] {
            for &x in v.iter() {
                sum += x * x;
            }
        }
        sum.sqrt()
    }
}

// ── MAG Configuration ────────────────────────────────────────────────

/// MAG (Memory-Attention-Gate) configuration.
/// Wraps SWAConfig and adds CMS parameters.
#[derive(Clone, Debug)]
pub struct MAGConfig {
    pub swa: SWAConfig,
    pub memory_enabled: bool,
    /// Which memory update rule to use.
    pub memory_rule: MemoryRuleKind,
    /// Number of CMS frequency levels (1 for Zero-B, 2 for Phase 2).
    pub k: usize,
    /// Chunk sizes per level: [1] for k=1, [1, 8] for k=2.
    /// Level i fires every chunk_sizes[i] steps.
    pub chunk_sizes: Vec<usize>,
}

/// Default gate bias init values per level index.
fn default_b_alpha(level: usize) -> f32 {
    // Higher levels need higher retention (closer to 1.0) because they fire
    // less frequently and must preserve information across longer intervals.
    match level {
        0 => 3.0,    // sigmoid(3.0) ≈ 0.95
        1 => 4.0,    // sigmoid(4.0) ≈ 0.98
        2 => 4.5,    // sigmoid(4.5) ≈ 0.989
        3 => 5.0,    // sigmoid(5.0) ≈ 0.993
        _ => 3.0,
    }
}

fn default_b_theta(level: usize) -> f32 {
    // Higher levels need smaller inner-loop lr because they accumulate M
    // over more steps (64/512). Even with 1/sqrt(k) output normalization,
    // aggressive inner-loop lr causes M to blow up for infrequently-firing levels.
    match level {
        0 => -4.6,   // softplus(-4.6) ≈ 0.01
        1 => -5.6,   // softplus(-5.6) ≈ 0.004
        2 => -6.6,   // softplus(-6.6) ≈ 0.0014
        3 => -7.6,   // softplus(-7.6) ≈ 0.0005
        _ => -4.6,
    }
}

/// Default momentum gate bias per level index.
/// Higher levels have more persistent momentum (higher eta → closer to 1).
pub fn default_b_eta(level: usize) -> f32 {
    match level {
        0 => 1.5,    // sigmoid(1.5) ≈ 0.82 (moderate momentum)
        1 => 2.0,    // sigmoid(2.0) ≈ 0.88
        2 => 2.5,    // sigmoid(2.5) ≈ 0.92
        3 => 3.0,    // sigmoid(3.0) ≈ 0.95 (very persistent)
        _ => 2.0,
    }
}

impl MAGConfig {
    /// Test configuration: tiny model for gradient checking (k=1).
    pub fn test_config() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 4,
                window_size: 4,
                vocab_size: 16,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
        }
    }

    /// Test configuration for CMS k=2 testing.
    pub fn test_config_k2() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 8,       // Must be >= chunk_sizes[1]=8
                window_size: 8,
                vocab_size: 16,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
        }
    }

    /// Validation config k=1: d=32 for multi-scale data experiments.
    pub fn validation_config_k1() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 32,
                num_heads: 4,
                head_dim: 8,
                seq_len: 32,
                window_size: 32,
                vocab_size: 64,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
        }
    }

    /// Validation config k=2: d=32 for multi-scale data experiments.
    /// Level 0 fires every step, Level 1 fires every 8th step.
    pub fn validation_config_k2() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 32,
                num_heads: 4,
                head_dim: 8,
                seq_len: 32,
                window_size: 32,
                vocab_size: 64,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
        }
    }

    /// Test configuration for CMS k=4 FD gradient checking.
    /// Small d=8 for large gradients, seq_len=16 for fast FD tests.
    pub fn test_config_k4() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 16,
                window_size: 16,
                vocab_size: 16,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 4,
            chunk_sizes: vec![1, 8, 64, 512],
        }
    }

    /// Validation config k=4: d=32, seq=512 for full frequency hierarchy.
    /// Level 0 fires every step, Level 1 every 8th, Level 2 every 64th, Level 3 every 512th.
    pub fn validation_config_k4() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 32,
                num_heads: 4,
                head_dim: 8,
                seq_len: 512,
                window_size: 512,
                vocab_size: 64,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 4,
            chunk_sizes: vec![1, 8, 64, 512],
        }
    }

    /// Titans LMM test configuration: tiny model for gradient checking (k=1).
    pub fn titans_test_config() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 4,
                window_size: 4,
                vocab_size: 16,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::TitansLMM,
            k: 1,
            chunk_sizes: vec![1],
        }
    }

    /// Titans LMM test configuration for CMS k=2 testing.
    pub fn titans_test_config_k2() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 8,
                window_size: 8,
                vocab_size: 16,
            },
            memory_enabled: true,
            memory_rule: MemoryRuleKind::TitansLMM,
            k: 2,
            chunk_sizes: vec![1, 8],
        }
    }
}

// ── MAG Parameters ───────────────────────────────────────────────────

/// MAG parameters: attention branch (SWAParams) + per-level memory weights.
///
/// `levels` has length `k` — one MemoryLevelParams per CMS frequency level.
/// For backward compatibility, existing k=1 code accesses `params.levels[0]`.
#[derive(Clone)]
pub struct MAGParams {
    pub swa: SWAParams,
    pub levels: Vec<MemoryLevelParams>,
}

impl MAGParams {
    /// Initialize with Xavier scaling for projections and level-specific gate bias init.
    pub fn init(cfg: &MAGConfig, seed: u64) -> Self {
        let swa = SWAParams::init(&cfg.swa, seed);
        let d = cfg.swa.d_model;

        let mut levels = Vec::with_capacity(cfg.k);
        for level in 0..cfg.k {
            // Different seed offset per level to avoid correlation
            let mut rng = SimpleRng::new(seed.wrapping_add(1000 + level as u64 * 500));
            levels.push(MemoryLevelParams::init(
                d, &mut rng,
                default_b_alpha(level),
                default_b_theta(level),
                default_b_eta(level),
            ));
        }

        MAGParams { swa, levels }
    }

    /// Create zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(cfg: &MAGConfig) -> Self {
        let d = cfg.swa.d_model;
        let levels = (0..cfg.k).map(|_| MemoryLevelParams::zeros_like(d)).collect();
        MAGParams {
            swa: SWAParams::zeros_like(&cfg.swa),
            levels,
        }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.swa.num_params()
            + self.levels.iter().map(|l| l.num_params()).sum::<usize>()
    }

    /// Apply SGD: param -= lr * grad for all weight matrices across all levels.
    pub fn sgd_step(&mut self, grads: &MAGParams, lr: f32) {
        self.swa.sgd_step(&grads.swa, lr);
        for (level, level_grads) in self.levels.iter_mut().zip(grads.levels.iter()) {
            level.sgd_step(level_grads, lr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_consistency() {
        let cfg = SWAConfig::test_config();
        assert_eq!(cfg.d_model, cfg.num_heads * cfg.head_dim);
    }

    #[test]
    fn test_init_deterministic() {
        let cfg = SWAConfig::test_config();
        let p1 = SWAParams::init(&cfg, 42);
        let p2 = SWAParams::init(&cfg, 42);
        assert_eq!(p1.w_q, p2.w_q);
        assert_eq!(p1.w_k, p2.w_k);
        assert_eq!(p1.w_embed, p2.w_embed);
    }

    #[test]
    fn test_param_shapes() {
        let cfg = SWAConfig::test_config();
        let p = SWAParams::init(&cfg, 42);
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        assert_eq!(p.w_embed.len(), v * d);
        assert_eq!(p.w_q.len(), d * d);
        assert_eq!(p.w_k.len(), d * d);
        assert_eq!(p.w_v.len(), d * d);
        assert_eq!(p.w_o.len(), d * d);
        assert_eq!(p.w_unembed.len(), d * v);
    }

    #[test]
    fn test_zeros_like() {
        let cfg = SWAConfig::test_config();
        let z = SWAParams::zeros_like(&cfg);
        assert!(z.w_q.iter().all(|&x| x == 0.0));
        assert!(z.w_embed.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_init_range() {
        let cfg = SWAConfig::test_config();
        let p = SWAParams::init(&cfg, 42);
        // Xavier scale for d=64: sqrt(2/128) ≈ 0.125
        for &v in &p.w_q {
            assert!(v.abs() < 0.2, "Weight {} out of expected range", v);
        }
    }

    // ── MAG tests ────────────────────────────────────────────────────

    #[test]
    fn test_mag_config() {
        let cfg = MAGConfig::test_config();
        assert!(cfg.memory_enabled);
        assert_eq!(cfg.k, 1);
        assert_eq!(cfg.chunk_sizes, vec![1]);
        assert_eq!(cfg.swa.d_model, cfg.swa.num_heads * cfg.swa.head_dim);
    }

    #[test]
    fn test_mag_param_shapes() {
        let cfg = MAGConfig::test_config();
        let p = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        assert_eq!(p.levels.len(), 1);
        assert_eq!(p.levels[0].w_k_mem.len(), d * d);
        assert_eq!(p.levels[0].w_v_mem.len(), d * d);
        assert_eq!(p.levels[0].w_q_mem.len(), d * d);
        assert_eq!(p.levels[0].w_alpha.len(), 2 * d);
        assert_eq!(p.levels[0].b_alpha.len(), 1);
        assert_eq!(p.levels[0].w_theta.len(), 2 * d);
        assert_eq!(p.levels[0].b_theta.len(), 1);
    }

    #[test]
    fn test_mag_init_deterministic() {
        let cfg = MAGConfig::test_config();
        let p1 = MAGParams::init(&cfg, 42);
        let p2 = MAGParams::init(&cfg, 42);
        assert_eq!(p1.levels[0].w_k_mem, p2.levels[0].w_k_mem);
        assert_eq!(p1.levels[0].w_alpha, p2.levels[0].w_alpha);
        assert_eq!(p1.levels[0].b_alpha, p2.levels[0].b_alpha);
    }

    #[test]
    fn test_mag_gate_bias_init() {
        let cfg = MAGConfig::test_config();
        let p = MAGParams::init(&cfg, 42);
        // Level 0: b_alpha=3.0 → sigmoid≈0.95 (high retention)
        assert!((p.levels[0].b_alpha[0] - 3.0).abs() < 1e-6);
        // Level 0: b_theta=-4.6 → softplus≈0.01 (small lr)
        assert!((p.levels[0].b_theta[0] - (-4.6)).abs() < 1e-6);
    }

    #[test]
    fn test_mag_zeros_like() {
        let cfg = MAGConfig::test_config();
        let z = MAGParams::zeros_like(&cfg);
        assert_eq!(z.levels.len(), 1);
        assert!(z.levels[0].w_k_mem.iter().all(|&x| x == 0.0));
        assert!(z.levels[0].w_alpha.iter().all(|&x| x == 0.0));
        assert!(z.levels[0].b_alpha.iter().all(|&x| x == 0.0));
        assert!(z.swa.w_q.iter().all(|&x| x == 0.0));
    }

    // ── k=2 specific tests ──────────────────────────────────────────

    #[test]
    fn test_mag_k2_config() {
        let cfg = MAGConfig::test_config_k2();
        assert_eq!(cfg.k, 2);
        assert_eq!(cfg.chunk_sizes, vec![1, 8]);
    }

    #[test]
    fn test_mag_k2_param_shapes() {
        let cfg = MAGConfig::test_config_k2();
        let p = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        assert_eq!(p.levels.len(), 2);
        for level in &p.levels {
            assert_eq!(level.w_k_mem.len(), d * d);
            assert_eq!(level.w_alpha.len(), 2 * d);
            assert_eq!(level.b_alpha.len(), 1);
        }
    }

    #[test]
    fn test_mag_k2_gate_bias_init() {
        let cfg = MAGConfig::test_config_k2();
        let p = MAGParams::init(&cfg, 42);
        // Level 0: b_alpha=3.0, b_theta=-4.6
        assert!((p.levels[0].b_alpha[0] - 3.0).abs() < 1e-6);
        assert!((p.levels[0].b_theta[0] - (-4.6)).abs() < 1e-6);
        // Level 1: b_alpha=4.0 (higher retention), b_theta=-5.6 (smaller lr)
        assert!((p.levels[1].b_alpha[0] - 4.0).abs() < 1e-6);
        assert!((p.levels[1].b_theta[0] - (-5.6)).abs() < 1e-6);
    }

    #[test]
    fn test_mag_k2_different_seeds() {
        let cfg = MAGConfig::test_config_k2();
        let p = MAGParams::init(&cfg, 42);
        // Level 0 and Level 1 should have different weights
        assert_ne!(p.levels[0].w_k_mem, p.levels[1].w_k_mem);
    }

    #[test]
    fn test_memory_level_params_accumulate() {
        let d = 4;
        let mut a = MemoryLevelParams::zeros_like(d);
        let mut b = MemoryLevelParams::zeros_like(d);
        a.w_k_mem[0] = 1.0;
        b.w_k_mem[0] = 2.0;
        a.b_alpha[0] = 0.5;
        b.b_alpha[0] = 0.3;
        a.accumulate(&b);
        assert!((a.w_k_mem[0] - 3.0).abs() < 1e-6);
        assert!((a.b_alpha[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_memory_level_params_norm() {
        let d = 2;
        let mut p = MemoryLevelParams::zeros_like(d);
        p.w_k_mem[0] = 3.0;
        p.w_k_mem[1] = 4.0;
        // norm = sqrt(9+16) = 5.0
        assert!((p.norm() - 5.0).abs() < 1e-6);
    }

    // ── k=4 specific tests ──────────────────────────────────────────

    #[test]
    fn test_mag_k4_config() {
        let cfg = MAGConfig::test_config_k4();
        assert_eq!(cfg.k, 4);
        assert_eq!(cfg.chunk_sizes, vec![1, 8, 64, 512]);
        assert_eq!(cfg.swa.d_model, cfg.swa.num_heads * cfg.swa.head_dim);

        let val = MAGConfig::validation_config_k4();
        assert_eq!(val.k, 4);
        assert_eq!(val.swa.seq_len, 512);
        assert_eq!(val.chunk_sizes, vec![1, 8, 64, 512]);
    }

    #[test]
    fn test_mag_k4_param_shapes() {
        let cfg = MAGConfig::test_config_k4();
        let p = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        assert_eq!(p.levels.len(), 4);
        for level in &p.levels {
            assert_eq!(level.w_k_mem.len(), d * d);
            assert_eq!(level.w_v_mem.len(), d * d);
            assert_eq!(level.w_q_mem.len(), d * d);
            assert_eq!(level.w_alpha.len(), 2 * d);
            assert_eq!(level.b_alpha.len(), 1);
            assert_eq!(level.w_theta.len(), 2 * d);
            assert_eq!(level.b_theta.len(), 1);
        }
    }

    #[test]
    fn test_mag_k4_gate_bias_init() {
        let cfg = MAGConfig::test_config_k4();
        let p = MAGParams::init(&cfg, 42);
        // Level 0: b_alpha=3.0, b_theta=-4.6
        assert!((p.levels[0].b_alpha[0] - 3.0).abs() < 1e-6);
        assert!((p.levels[0].b_theta[0] - (-4.6)).abs() < 1e-6);
        // Level 1: b_alpha=4.0, b_theta=-5.6
        assert!((p.levels[1].b_alpha[0] - 4.0).abs() < 1e-6);
        assert!((p.levels[1].b_theta[0] - (-5.6)).abs() < 1e-6);
        // Level 2: b_alpha=4.5, b_theta=-6.6
        assert!((p.levels[2].b_alpha[0] - 4.5).abs() < 1e-6);
        assert!((p.levels[2].b_theta[0] - (-6.6)).abs() < 1e-6);
        // Level 3: b_alpha=5.0, b_theta=-7.6
        assert!((p.levels[3].b_alpha[0] - 5.0).abs() < 1e-6);
        assert!((p.levels[3].b_theta[0] - (-7.6)).abs() < 1e-6);
    }

    #[test]
    fn test_mag_k4_different_seeds() {
        let cfg = MAGConfig::test_config_k4();
        let p = MAGParams::init(&cfg, 42);
        // All 4 levels should have different weights (different RNG seeds)
        assert_ne!(p.levels[0].w_k_mem, p.levels[1].w_k_mem);
        assert_ne!(p.levels[1].w_k_mem, p.levels[2].w_k_mem);
        assert_ne!(p.levels[2].w_k_mem, p.levels[3].w_k_mem);
    }

    // ── Titans LMM tests ────────────────────────────────────────────

    #[test]
    fn test_memory_level_params_with_eta() {
        let cfg = MAGConfig::titans_test_config();
        let p = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        assert_eq!(p.levels[0].w_eta.len(), 2 * d);
        assert_eq!(p.levels[0].b_eta.len(), 1);
        // eta params included in num_params
        let expected = 3 * d * d + 2 * (2 * d) + 2 + (2 * d) + 1; // 3 proj + 2 gates + eta
        assert_eq!(p.levels[0].num_params(), expected);
    }

    #[test]
    fn test_mag_config_titans() {
        let cfg = MAGConfig::titans_test_config();
        assert_eq!(cfg.memory_rule, MemoryRuleKind::TitansLMM);
        assert!(cfg.memory_enabled);
        assert_eq!(cfg.k, 1);

        let cfg2 = MAGConfig::titans_test_config_k2();
        assert_eq!(cfg2.memory_rule, MemoryRuleKind::TitansLMM);
        assert_eq!(cfg2.k, 2);
    }

    #[test]
    fn test_titans_default_b_eta() {
        assert!((default_b_eta(0) - 1.5).abs() < 1e-6);
        assert!((default_b_eta(1) - 2.0).abs() < 1e-6);
        assert!((default_b_eta(2) - 2.5).abs() < 1e-6);
        assert!((default_b_eta(3) - 3.0).abs() < 1e-6);
        assert!((default_b_eta(4) - 2.0).abs() < 1e-6); // fallback
    }
}
