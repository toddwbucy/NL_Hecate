/// SWA Transformer model configuration and parameters.
///
/// Track Zero-A: single-block SWA with no memory, no CMS, no inner loop.
/// All weight matrices are flat Vec<f32> in row-major layout.

use serde::{Serialize, Deserialize};
use crate::tensor::SimpleRng;
use crate::parallel::ParallelConfig;
use crate::retention::{RetentionKind, default_retention};
use crate::m3::{M3Config, M3State, m3_step, flatten_mag_params, unflatten_to_mag_grads};
use crate::dynamic_freq::{FrequencySchedule, default_b_freq};

/// Which composition pattern to use (Titans Section 4).
///
/// Three ways memory connects to attention — orthogonal to memory rule choice.
/// - MAG: Memory gates attention output via sigmoid (parallel branches)
/// - MAL: Memory preprocesses input, attention processes memory output (simplest)
/// - MAC: Memory provides context, attention processes assembled input (most expressive)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum CompositionKind {
    MAG,
    MAL,
    MAC,
}

/// Which memory update rule to use for the inner loop.
///
/// MIRAS Algorithm knob: selects the optimizer for memory updates.
/// - DeltaRule: GD without momentum (Titans Eq 34)
/// - TitansLMM: GD + momentum accumulator S (Titans Eqs 12-15, 32-33)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MemoryRuleKind {
    DeltaRule,
    TitansLMM,
    HebbianRule,
    Moneta,
    YAAD,
    MEMORA,
    LatticeOSR,
    Trellis,
    AtlasOmega,
}

/// Model configuration — immutable after construction.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
#[derive(Clone, Serialize, Deserialize)]
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

    /// Outer-loop weight update: param -= lr * grad for all projection weights.
    pub fn apply_weight_gradients(&mut self, grads: &SWAParams, lr: f32) {
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
///   w_omega:  [d, 2*d] Atlas Omega projection (AtlasOmega only; zero-init for others)
#[derive(Clone, Serialize, Deserialize)]
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
    /// Atlas Omega projection: [d, 2*d]. omega(k,v) = W_omega @ silu(concat(k_mem, v_mem)).
    /// Zero-initialized for non-Atlas rules (adds d*2*d params per level).
    pub w_omega: Vec<f32>,
    /// Dynamic frequency gate weights: [d]. Empty for Fixed schedule.
    pub w_freq: Vec<f32>,
    /// Dynamic frequency gate bias: [1]. Empty for Fixed schedule.
    pub b_freq: Vec<f32>,
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

        // Atlas Omega projection: zero-initialized by default.
        // atlas_init() below provides Xavier-initialized w_omega.
        let w_omega = vec![0.0f32; d * 2 * d];

        // Dynamic frequency gate: empty by default (Fixed schedule).
        // Populated by MAGParams::init() when FrequencySchedule::Learned.
        let w_freq = vec![];
        let b_freq = vec![];

        MemoryLevelParams { w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta, w_eta, b_eta, w_omega, w_freq, b_freq }
    }

    /// Initialize with Xavier-initialized w_omega for Atlas Omega rule.
    pub fn atlas_init(d: usize, rng: &mut SimpleRng, b_alpha_init: f32, b_theta_init: f32, b_eta_init: f32) -> Self {
        let mut params = Self::init(d, rng, b_alpha_init, b_theta_init, b_eta_init);
        let omega_scale = (1.0 / (2 * d) as f32).sqrt();
        rng.fill_uniform(&mut params.w_omega, omega_scale);
        params
    }

    /// Initialize frequency gate weights for learned scheduling.
    /// Called by MAGParams::init() when FrequencySchedule::Learned.
    pub fn init_freq_gate(&mut self, d: usize, rng: &mut SimpleRng, level: usize) {
        let freq_scale = (1.0 / d as f32).sqrt();
        self.w_freq = vec![0.0f32; d];
        rng.fill_uniform(&mut self.w_freq, freq_scale);
        self.b_freq = vec![default_b_freq(level)];
    }

    /// Create zero-initialized shadow for gradient accumulation.
    /// If `freq_d` > 0, allocates w_freq/b_freq for Learned schedule.
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
            w_omega: vec![0.0f32; d * 2 * d],
            w_freq: vec![],
            b_freq: vec![],
        }
    }

    /// Create zero-initialized shadow matching the shape of `template`.
    pub fn zeros_like_from(template: &MemoryLevelParams, d: usize) -> Self {
        let mut z = Self::zeros_like(d);
        if !template.w_freq.is_empty() {
            z.w_freq = vec![0.0f32; template.w_freq.len()];
            z.b_freq = vec![0.0f32; template.b_freq.len()];
        }
        z
    }

    /// Total number of parameters in this level.
    pub fn num_params(&self) -> usize {
        self.w_k_mem.len() + self.w_v_mem.len() + self.w_q_mem.len()
            + self.w_alpha.len() + self.b_alpha.len()
            + self.w_theta.len() + self.b_theta.len()
            + self.w_eta.len() + self.b_eta.len()
            + self.w_omega.len()
            + self.w_freq.len() + self.b_freq.len()
    }

    /// Outer-loop weight update: param -= lr * grad for all projection weights.
    pub fn apply_weight_gradients(&mut self, grads: &MemoryLevelParams, lr: f32) {
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
        step(&mut self.w_omega, &grads.w_omega, lr);
        if !self.w_freq.is_empty() && !grads.w_freq.is_empty() {
            step(&mut self.w_freq, &grads.w_freq, lr);
            step(&mut self.b_freq, &grads.b_freq, lr);
        }
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
        acc(&mut self.w_omega, &other.w_omega);
        if !self.w_freq.is_empty() && !other.w_freq.is_empty() {
            acc(&mut self.w_freq, &other.w_freq);
            acc(&mut self.b_freq, &other.b_freq);
        }
    }

    /// Frobenius norm across all weight matrices.
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for v in [&self.w_k_mem, &self.w_v_mem, &self.w_q_mem,
                   &self.w_alpha, &self.b_alpha, &self.w_theta, &self.b_theta,
                   &self.w_eta, &self.b_eta, &self.w_omega,
                   &self.w_freq, &self.b_freq] {
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MAGConfig {
    pub swa: SWAConfig,
    pub memory_enabled: bool,
    /// Which composition pattern to use (MAG, MAL, MAC).
    pub composition: CompositionKind,
    /// Which memory update rule to use.
    pub memory_rule: MemoryRuleKind,
    /// Number of CMS frequency levels (1 for Zero-B, 2 for Phase 2).
    pub k: usize,
    /// Chunk sizes per level: [1] for k=1, [1, 8] for k=2.
    /// Level i fires every chunk_sizes[i] steps.
    pub chunk_sizes: Vec<usize>,
    /// MLP hidden dimension for MONETA (default: 0, unused by matrix rules).
    pub d_hidden: usize,
    /// l_p norm exponent for MONETA attentional bias (default: 2.0).
    pub lp_p: f32,
    /// L_q retention exponent for MONETA (default: 2.0).
    pub lq_q: f32,
    /// Local retention strength: L2 penalty toward chunk-boundary snapshot.
    /// Used by YAAD (default: 0.01). MONETA sets to 0.0 (disabled).
    pub lambda_local: f32,
    /// Global L2 retention strength for MONETA/YAAD (default: 0.01).
    pub lambda_2: f32,
    /// Huber loss threshold for YAAD (default: 1.0).
    /// Errors below delta get L2 gradient; above get bounded L1 gradient.
    pub delta: f32,
    /// Number of memory slots for Lattice OSR (default: 0, unused by other rules).
    pub m_slots: usize,
    /// Key compression dimension for Trellis (default: 0, unused by other rules).
    pub d_compress: usize,
    /// Key state L2 decay rate for Trellis (default: 0.0).
    pub lambda_k: f32,
    /// Value state L2 decay rate for Trellis (default: 0.0).
    pub lambda_v: f32,
    /// Parallelization config. None = sequential (backward compatible).
    pub parallel: Option<ParallelConfig>,
    /// Which retention mechanism to use. Default: derived from memory_rule.
    /// Override to use a non-default mechanism (e.g., ElasticNet with DeltaRule).
    pub retention: RetentionKind,
    /// M3 multi-scale optimizer config. None = plain SGD (default).
    pub m3: Option<M3Config>,
    /// Frequency scheduling strategy. Fixed = modular arithmetic (default).
    /// Learned = sigmoid gate per level based on input embeddings.
    pub frequency_schedule: FrequencySchedule,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 4,
            chunk_sizes: vec![1, 8, 64, 512],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 4,
            chunk_sizes: vec![1, 8, 64, 512],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::TitansLMM,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Hebbian Rule test configuration: tiny model for gradient checking (k=1).
    pub fn hebbian_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::HebbianRule,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Hebbian Rule test configuration for CMS k=2 testing.
    pub fn hebbian_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::HebbianRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::TitansLMM,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MONETA test configuration: d=8, d_hidden=4 (k=1).
    pub fn moneta_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::Moneta,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.0,
            lambda_2: 0.01,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MONETA test configuration for CMS k=2 testing.
    pub fn moneta_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::Moneta,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.0,
            lambda_2: 0.01,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// YAAD test configuration: d=8, d_hidden=4, Huber delta=1.0 (k=1).
    pub fn yaad_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::YAAD,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.01,
            lambda_2: 0.01,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// YAAD test configuration for CMS k=2 testing.
    pub fn yaad_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::YAAD,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.01,
            lambda_2: 0.01,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MEMORA test configuration: d=8, d_hidden=4, KL retention (k=1).
    pub fn memora_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::MEMORA,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.0,
            lambda_2: 0.0,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::MEMORA),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MEMORA test configuration for CMS k=2 testing.
    pub fn memora_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::MEMORA,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 4,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.0,
            lambda_2: 0.0,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::MEMORA),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Lattice OSR test configuration: d=8, m_slots=4 (k=1).
    pub fn lattice_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::LatticeOSR,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 4, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::LatticeOSR),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Lattice OSR test configuration for CMS k=2 testing.
    pub fn lattice_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::LatticeOSR,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 4, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::LatticeOSR),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Trellis test configuration: d=8, d_compress=8, lambda_k/v=0.01 (k=1).
    pub fn trellis_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::Trellis,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 0,
            d_compress: 8,
            lambda_k: 0.01,
            lambda_v: 0.01,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Trellis test configuration for CMS k=2 testing.
    pub fn trellis_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::Trellis,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 0,
            d_compress: 8,
            lambda_k: 0.01,
            lambda_v: 0.01,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Atlas Omega test configuration: d=8, k=1.
    pub fn atlas_test_config() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::AtlasOmega,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::AtlasOmega),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Atlas Omega test configuration for CMS k=2 testing.
    pub fn atlas_test_config_k2() -> Self {
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::AtlasOmega,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::AtlasOmega),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MAL test configuration: d=8, k=1, DeltaRule.
    pub fn mal_test_config() -> Self {
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
            composition: CompositionKind::MAL,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MAL test configuration for CMS k=2 testing.
    pub fn mal_test_config_k2() -> Self {
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
            composition: CompositionKind::MAL,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MAC test configuration: d=8, k=1, DeltaRule.
    /// window_size=16 (2×seq_len) for full causal attention on assembled (2s, d).
    pub fn mac_test_config() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 4,
                window_size: 8,   // 2 * seq_len for assembled input
                vocab_size: 16,
            },
            memory_enabled: true,
            composition: CompositionKind::MAC,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 1,
            chunk_sizes: vec![1],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// MAC test configuration for CMS k=2 testing.
    pub fn mac_test_config_k2() -> Self {
        MAGConfig {
            swa: SWAConfig {
                d_model: 8,
                num_heads: 2,
                head_dim: 4,
                seq_len: 8,
                window_size: 16,  // 2 * seq_len for assembled input
                vocab_size: 16,
            },
            memory_enabled: true,
            composition: CompositionKind::MAC,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
        }
    }

    /// Dynamic frequency test config: k=2, Learned schedule (DeltaRule).
    pub fn dynamic_freq_test_config() -> Self {
        use crate::dynamic_freq::LearnedFreqConfig;
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 2,
            chunk_sizes: vec![1, 8],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig::default()),
        }
    }

    /// Dynamic frequency test config: k=4, Learned schedule.
    pub fn dynamic_freq_test_config_k4() -> Self {
        use crate::dynamic_freq::LearnedFreqConfig;
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
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
            k: 4,
            chunk_sizes: vec![1, 8, 64, 512],
            d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig::default()),
        }
    }
}

// ── MAG Parameters ───────────────────────────────────────────────────

/// MAG parameters: attention branch (SWAParams) + per-level memory weights.
///
/// `levels` has length `k` — one MemoryLevelParams per CMS frequency level.
/// For backward compatibility, existing k=1 code accesses `params.levels[0]`.
#[derive(Clone, Serialize, Deserialize)]
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
            let mut level_params = if cfg.memory_rule == MemoryRuleKind::AtlasOmega {
                MemoryLevelParams::atlas_init(
                    d, &mut rng,
                    default_b_alpha(level),
                    default_b_theta(level),
                    default_b_eta(level),
                )
            } else {
                MemoryLevelParams::init(
                    d, &mut rng,
                    default_b_alpha(level),
                    default_b_theta(level),
                    default_b_eta(level),
                )
            };
            // Initialize frequency gate if using learned scheduling
            if matches!(cfg.frequency_schedule, FrequencySchedule::Learned(_)) {
                let mut freq_rng = SimpleRng::new(seed.wrapping_add(5000 + level as u64 * 100));
                level_params.init_freq_gate(d, &mut freq_rng, level);
            }
            levels.push(level_params);
        }

        MAGParams { swa, levels }
    }

    /// Create zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(cfg: &MAGConfig) -> Self {
        let d = cfg.swa.d_model;
        let has_freq = matches!(cfg.frequency_schedule, FrequencySchedule::Learned(_));
        let levels = (0..cfg.k).map(|_| {
            let mut z = MemoryLevelParams::zeros_like(d);
            if has_freq {
                z.w_freq = vec![0.0f32; d];
                z.b_freq = vec![0.0f32; 1];
            }
            z
        }).collect();
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

    /// Outer-loop weight update: param -= lr * grad for all projection weights across all levels.
    pub fn apply_weight_gradients(&mut self, grads: &MAGParams, lr: f32) {
        self.swa.apply_weight_gradients(&grads.swa, lr);
        for (level, level_grads) in self.levels.iter_mut().zip(grads.levels.iter()) {
            level.apply_weight_gradients(level_grads, lr);
        }
    }

    /// Outer-loop weight update using M3 multi-scale optimizer.
    ///
    /// Flattens gradients → runs M3 step → unflattens → applies update.
    /// The M3 step transforms gradients through multi-scale momentum before application.
    pub fn apply_weight_gradients_m3(
        &mut self, grads: &MAGParams, m3_state: &mut M3State, m3_cfg: &M3Config,
    ) {
        let flat_grads = flatten_mag_params(grads);
        let update = m3_step(m3_state, m3_cfg, &flat_grads);
        let update_as_grads = unflatten_to_mag_grads(&update, self);
        // Apply: param -= update (m3_step already scaled by theta)
        self.apply_weight_gradients(&update_as_grads, 1.0);
    }
}

// ── Checkpoint Serialization ─────────────────────────────────────────

/// Internal wrapper for JSON checkpoint format.
#[derive(Serialize, Deserialize)]
struct ParamCheckpoint {
    config: MAGConfig,
    params: MAGParams,
}

/// Save MAGParams + MAGConfig to a JSON file.
pub fn save_checkpoint(path: &std::path::Path, params: &MAGParams, config: &MAGConfig) -> std::io::Result<()> {
    let checkpoint = ParamCheckpoint {
        config: config.clone(),
        params: params.clone(),
    };
    let json = serde_json::to_string(&checkpoint)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load MAGParams + MAGConfig from a JSON file.
pub fn load_checkpoint(path: &std::path::Path) -> std::io::Result<(MAGParams, MAGConfig)> {
    let json = std::fs::read_to_string(path)?;
    let checkpoint: ParamCheckpoint = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok((checkpoint.params, checkpoint.config))
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
        // 3 proj (d*d each) + alpha gate (2*d + 1) + theta gate (2*d + 1) + eta gate (2*d + 1) + w_omega (d * 2*d)
        let expected = 3 * d * d + 3 * (2 * d + 1) + d * 2 * d;
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
