/// SWA Transformer model configuration and parameters.
///
/// Track Zero-A: single-block SWA with no memory, no CMS, no inner loop.
/// All weight matrices are flat Vec<f32> in row-major layout.

use serde::{Serialize, Deserialize};
use crate::bf16::Bf16Storage;
use crate::tensor::SimpleRng;
use crate::parallel::ParallelConfig;
use crate::retention::{RetentionKind, default_retention};
use crate::m3::{M3Config, M3State, m3_step, flatten_mag_params, unflatten_to_mag_grads};
use crate::dynamic_freq::{FrequencySchedule, default_b_freq};
pub use crate::self_ref::ProjectionKind;

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

/// Momentum expressiveness hierarchy (HOPE §4.2-4.4).
///
/// Four levels of momentum sophistication for inner-loop memory updates.
/// Orthogonal to memory rule choice — any momentum kind can pair with any rule
/// that uses a momentum accumulator (currently TitansLMM, AtlasOmega).
///
/// - None (Level 0): No momentum accumulator. Used by DeltaRule, Hebbian, etc.
/// - EMA (Level 1): S = eta*S - theta*grad (HOPE Eq 33, Titans Eqs 32-33).
/// - DeltaMomentum (Level 2): State-dependent decay S*(eta - g^Tg) - theta*P@g (HOPE Eq 49).
/// - DeepMomentum (Level 3): MLP replaces linear accumulator (HOPE Eq 50).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MomentumKind {
    /// Level 0: no momentum accumulator.
    None,
    /// Level 1: exponential moving average. S_{t+1} = eta_t * S_t - theta_t * grad.
    EMA,
    /// Level 2: gradient-dependent decay. decay = clamp(eta - ||g||^2, eps, 1-eps).
    DeltaMomentum,
    /// Level 3: MLP-based momentum. W1,W2 inner-loop weights replace linear S.
    DeepMomentum,
}

impl Default for MomentumKind {
    fn default() -> Self {
        MomentumKind::None
    }
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
    /// HOPE §7.3 ad-hoc level stacking: SwiGLU MLP with outer-loop AdamW.
    /// No inner-loop M state — gate_proj/up_proj/down_proj are the memory.
    SwiGluMlp,
}

/// MIRAS Knob 2: Attentional bias (loss function for memory updates).
///
/// Selects the objective that drives the inner-loop memory gradient.
/// L2 is the default (squared error). L1 uses smooth tanh approximation.
/// Lp generalizes to arbitrary p-norms. KL uses cross-entropy on softmax.
/// See specs/algorithms/attentional_biases/ for details.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttentionalBias {
    /// L2 squared error: grad = 2 * (M@k - v) @ k^T. Default for most rules.
    L2,
    /// L1 value-less: grad = tanh(a * (M@k - v)) @ k^T. Robust to outliers.
    L1,
    /// General l_p norm: grad = p * tanh(a*e) * |e|^(p-1) @ k^T.
    Lp(f32),
    /// KL cross-entropy: grad = (softmax(M@k) - softmax(v)) @ k^T.
    KL,
    /// Huber: L2 for |e| < delta, L1 for |e| >= delta. Bridges L2 and L1.
    Huber,
}

impl Default for AttentionalBias {
    fn default() -> Self {
        AttentionalBias::L2
    }
}

/// HOPE §6 level-level composition variants (Eqs 70-75).
///
/// Defines how CMS levels interact with EACH OTHER — orthogonal to
/// memory-attention composition (MAC/MAG/MAL) which defines how memory
/// interacts with attention WITHIN a single level.
///
/// Source: HOPE (2512.24695) §6 Eqs 70-75.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HopeVariant {
    /// Variant 1 (Eq 70): levels in series, output chains through all.
    /// Level 0 processes first, each subsequent level sees transformed input.
    Chained,
    /// Variant 2 (Eq 71): each level updates at its own frequency, idle otherwise.
    /// All levels independently process raw input, outputs aggregated.
    /// This is the DEFAULT — what the Conductor + Pulse system implements.
    FreqGated,
    /// Variant 3 (Eq 72): higher level meta-learns initial state of lower.
    /// Level s re-initializes level s+1 via linear projection of its memory state.
    Nested,
    /// Variant 4 (Eq 73): output of level s feeds level s+1, all initialized
    /// from slowest level's meta-learned state.
    Sequential,
    /// Variant 5 (Eq 74): levels process input independently, outputs aggregated.
    /// Identical to FreqGated in NL_Hecate (notational distinction in paper).
    Independent,
}

impl Default for HopeVariant {
    fn default() -> Self {
        HopeVariant::FreqGated
    }
}

fn default_hope_variant() -> HopeVariant { HopeVariant::FreqGated }

/// Lattice OSR update variants (Lattice Eqs 5-8, 24-26).
///
/// Three ways to compute delta_s in the slot update:
/// - Decode: delta_s = gate * v_t (store the value — default, Eqs 5-6)
/// - Encode: delta_s = gate * k_t (store the key — Eqs 24-25)
/// - Similarity: delta_s = gate * (v_t - dot(S[i], v_t) * S[i]) (pre-project — Eqs 7-8)
///
/// All share the same orthogonal_update + normalize step afterward.
/// Source: Lattice (2504.05646) Section 3-4, unified under Eq 26.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LatticeVariant {
    /// Eqs 5-6: delta_s = gate_i * v_t. Store the value.
    Decode,
    /// Eqs 24-25: delta_s = gate_i * k_t. Store the key.
    Encode,
    /// Eqs 7-8: delta_s = gate_i * (v_t - dot(S[i], v_t) * S[i]).
    /// Explicit orthogonal projection of v_t before the standard orthogonal_update.
    Similarity,
}

impl Default for LatticeVariant {
    fn default() -> Self {
        LatticeVariant::Decode
    }
}

fn default_lattice_variant() -> LatticeVariant { LatticeVariant::Decode }

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

/// All learnable parameters — flat Vec<f32> for AD compatibility.
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
    pub w_k_mem: Bf16Storage,
    pub w_v_mem: Bf16Storage,
    pub w_q_mem: Bf16Storage,
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
    /// Conv1D key weights: [d, kernel_size]. Empty when kernel_size=0.
    pub w_k_conv: Vec<f32>,
    /// Conv1D key bias: [d]. Empty when kernel_size=0.
    pub b_k_conv: Vec<f32>,
    /// Conv1D query weights: [d, kernel_size]. Empty when kernel_size=0.
    pub w_q_conv: Vec<f32>,
    /// Conv1D query bias: [d]. Empty when kernel_size=0.
    pub b_q_conv: Vec<f32>,
    /// Self-referential key projection memory initial state: [d*d].
    /// Empty when ProjectionKind::Static. Seeds M_k at sequence start.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_k_init: Vec<f32>,
    /// Self-referential value projection memory initial state: [d*d].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_v_init: Vec<f32>,
    /// Self-referential query projection memory initial state: [d*d].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_q_init: Vec<f32>,
    /// Self-referential momentum gate memory initial state: [d*d].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_eta_init: Vec<f32>,
    /// Self-referential retention memory initial state: [d*d].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_alpha_init: Vec<f32>,
    /// Self-referential main projection memory initial state: [d*d].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub m_mem_init: Vec<f32>,
    /// SwiGluMlp gate projection: [intermediate x d_model]. Empty for all other rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gate_proj: Vec<f32>,
    /// SwiGluMlp up projection: [intermediate x d_model]. Empty for all other rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub up_proj: Vec<f32>,
    /// SwiGluMlp down projection: [d_model x intermediate]. Empty for all other rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub down_proj: Vec<f32>,
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

        let mut w_k_raw = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_k_raw, proj_scale);
        let w_k_mem = Bf16Storage::from_f32_vec(w_k_raw);

        let mut w_v_raw = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_v_raw, proj_scale);
        let w_v_mem = Bf16Storage::from_f32_vec(w_v_raw);

        let mut w_q_raw = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_q_raw, proj_scale);
        let w_q_mem = Bf16Storage::from_f32_vec(w_q_raw);

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

        MemoryLevelParams { w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta, w_eta, b_eta, w_omega, w_freq, b_freq, w_k_conv: vec![], b_k_conv: vec![], w_q_conv: vec![], b_q_conv: vec![], m_k_init: vec![], m_v_init: vec![], m_q_init: vec![], m_eta_init: vec![], m_alpha_init: vec![], m_mem_init: vec![], gate_proj: vec![], up_proj: vec![], down_proj: vec![] }
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

    /// Initialize Conv1D weights for key/query preprocessing.
    /// Called by MAGParams::init() when kernel_size > 0.
    /// Kaiming uniform init: fan_in = kernel_size.
    pub fn init_conv(&mut self, d: usize, kernel_size: usize, rng: &mut SimpleRng) {
        let conv_scale = (1.0 / kernel_size as f32).sqrt();
        self.w_k_conv = vec![0.0f32; d * kernel_size];
        rng.fill_uniform(&mut self.w_k_conv, conv_scale);
        self.b_k_conv = vec![0.0f32; d]; // zeros
        self.w_q_conv = vec![0.0f32; d * kernel_size];
        rng.fill_uniform(&mut self.w_q_conv, conv_scale);
        self.b_q_conv = vec![0.0f32; d]; // zeros
    }

    /// Create zero-initialized shadow for gradient accumulation.
    /// If `freq_d` > 0, allocates w_freq/b_freq for Learned schedule.
    pub fn zeros_like(d: usize) -> Self {
        MemoryLevelParams {
            w_k_mem: Bf16Storage::zeros(d * d),
            w_v_mem: Bf16Storage::zeros(d * d),
            w_q_mem: Bf16Storage::zeros(d * d),
            w_alpha: vec![0.0f32; 2 * d],
            b_alpha: vec![0.0f32; 1],
            w_theta: vec![0.0f32; 2 * d],
            b_theta: vec![0.0f32; 1],
            w_eta: vec![0.0f32; 2 * d],
            b_eta: vec![0.0f32; 1],
            w_omega: vec![0.0f32; d * 2 * d],
            w_freq: vec![],
            b_freq: vec![],
            w_k_conv: vec![],
            b_k_conv: vec![],
            w_q_conv: vec![],
            b_q_conv: vec![],
            m_k_init: vec![],
            m_v_init: vec![],
            m_q_init: vec![],
            m_eta_init: vec![],
            m_alpha_init: vec![],
            m_mem_init: vec![],
            gate_proj: vec![],
            up_proj: vec![],
            down_proj: vec![],
        }
    }

    /// Create zero-initialized shadow matching the shape of `template`.
    pub fn zeros_like_from(template: &MemoryLevelParams, d: usize) -> Self {
        let mut z = Self::zeros_like(d);
        if !template.w_freq.is_empty() {
            z.w_freq = vec![0.0f32; template.w_freq.len()];
            z.b_freq = vec![0.0f32; template.b_freq.len()];
        }
        if !template.w_k_conv.is_empty() {
            z.w_k_conv = vec![0.0f32; template.w_k_conv.len()];
            z.b_k_conv = vec![0.0f32; template.b_k_conv.len()];
            z.w_q_conv = vec![0.0f32; template.w_q_conv.len()];
            z.b_q_conv = vec![0.0f32; template.b_q_conv.len()];
        }
        if !template.m_k_init.is_empty() { z.m_k_init = vec![0.0f32; template.m_k_init.len()]; }
        if !template.m_v_init.is_empty() { z.m_v_init = vec![0.0f32; template.m_v_init.len()]; }
        if !template.m_q_init.is_empty() { z.m_q_init = vec![0.0f32; template.m_q_init.len()]; }
        if !template.m_eta_init.is_empty() { z.m_eta_init = vec![0.0f32; template.m_eta_init.len()]; }
        if !template.m_alpha_init.is_empty() { z.m_alpha_init = vec![0.0f32; template.m_alpha_init.len()]; }
        if !template.m_mem_init.is_empty() { z.m_mem_init = vec![0.0f32; template.m_mem_init.len()]; }
        if !template.gate_proj.is_empty() { z.gate_proj = vec![0.0f32; template.gate_proj.len()]; }
        if !template.up_proj.is_empty() { z.up_proj = vec![0.0f32; template.up_proj.len()]; }
        if !template.down_proj.is_empty() { z.down_proj = vec![0.0f32; template.down_proj.len()]; }
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
            + self.w_k_conv.len() + self.b_k_conv.len()
            + self.w_q_conv.len() + self.b_q_conv.len()
            + self.m_k_init.len() + self.m_v_init.len() + self.m_q_init.len()
            + self.m_eta_init.len() + self.m_alpha_init.len() + self.m_mem_init.len()
            + self.gate_proj.len() + self.up_proj.len() + self.down_proj.len()
    }

    /// Outer-loop weight update: param -= lr * grad for all projection weights.
    /// For Bf16Storage fields: update master copy, then sync bf16 stored copy.
    pub fn apply_weight_gradients(&mut self, grads: &MemoryLevelParams, lr: f32) {
        fn step(param: &mut [f32], grad: &[f32], lr: f32) {
            for i in 0..param.len() {
                param[i] -= lr * grad[i];
            }
        }
        step(self.w_k_mem.master_mut(), grads.w_k_mem.master(), lr);
        self.w_k_mem.sync_from_master();
        step(self.w_v_mem.master_mut(), grads.w_v_mem.master(), lr);
        self.w_v_mem.sync_from_master();
        step(self.w_q_mem.master_mut(), grads.w_q_mem.master(), lr);
        self.w_q_mem.sync_from_master();
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
        if !self.w_k_conv.is_empty() && !grads.w_k_conv.is_empty() {
            step(&mut self.w_k_conv, &grads.w_k_conv, lr);
            step(&mut self.b_k_conv, &grads.b_k_conv, lr);
            step(&mut self.w_q_conv, &grads.w_q_conv, lr);
            step(&mut self.b_q_conv, &grads.b_q_conv, lr);
        }
        if !self.m_k_init.is_empty() && !grads.m_k_init.is_empty() { debug_assert_eq!(self.m_k_init.len(), grads.m_k_init.len()); step(&mut self.m_k_init, &grads.m_k_init, lr); }
        if !self.m_v_init.is_empty() && !grads.m_v_init.is_empty() { debug_assert_eq!(self.m_v_init.len(), grads.m_v_init.len()); step(&mut self.m_v_init, &grads.m_v_init, lr); }
        if !self.m_q_init.is_empty() && !grads.m_q_init.is_empty() { debug_assert_eq!(self.m_q_init.len(), grads.m_q_init.len()); step(&mut self.m_q_init, &grads.m_q_init, lr); }
        if !self.m_eta_init.is_empty() && !grads.m_eta_init.is_empty() { debug_assert_eq!(self.m_eta_init.len(), grads.m_eta_init.len()); step(&mut self.m_eta_init, &grads.m_eta_init, lr); }
        if !self.m_alpha_init.is_empty() && !grads.m_alpha_init.is_empty() { debug_assert_eq!(self.m_alpha_init.len(), grads.m_alpha_init.len()); step(&mut self.m_alpha_init, &grads.m_alpha_init, lr); }
        if !self.m_mem_init.is_empty() && !grads.m_mem_init.is_empty() { debug_assert_eq!(self.m_mem_init.len(), grads.m_mem_init.len()); step(&mut self.m_mem_init, &grads.m_mem_init, lr); }
        if !self.gate_proj.is_empty() && !grads.gate_proj.is_empty() { debug_assert_eq!(self.gate_proj.len(), grads.gate_proj.len()); step(&mut self.gate_proj, &grads.gate_proj, lr); }
        if !self.up_proj.is_empty() && !grads.up_proj.is_empty() { debug_assert_eq!(self.up_proj.len(), grads.up_proj.len()); step(&mut self.up_proj, &grads.up_proj, lr); }
        if !self.down_proj.is_empty() && !grads.down_proj.is_empty() { debug_assert_eq!(self.down_proj.len(), grads.down_proj.len()); step(&mut self.down_proj, &grads.down_proj, lr); }
    }

    /// Element-wise accumulate: self += other.
    /// For Bf16Storage fields: accumulate into master copy (no sync needed —
    /// accumulate is used for gradient aggregation, not parameter update).
    pub fn accumulate(&mut self, other: &MemoryLevelParams) {
        fn acc(dst: &mut [f32], src: &[f32]) {
            for i in 0..dst.len() {
                dst[i] += src[i];
            }
        }
        acc(self.w_k_mem.master_mut(), other.w_k_mem.master());
        acc(self.w_v_mem.master_mut(), other.w_v_mem.master());
        acc(self.w_q_mem.master_mut(), other.w_q_mem.master());
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
        if !self.w_k_conv.is_empty() && !other.w_k_conv.is_empty() {
            acc(&mut self.w_k_conv, &other.w_k_conv);
            acc(&mut self.b_k_conv, &other.b_k_conv);
            acc(&mut self.w_q_conv, &other.w_q_conv);
            acc(&mut self.b_q_conv, &other.b_q_conv);
        }
        if !self.m_k_init.is_empty() && !other.m_k_init.is_empty() { debug_assert_eq!(self.m_k_init.len(), other.m_k_init.len()); acc(&mut self.m_k_init, &other.m_k_init); }
        if !self.m_v_init.is_empty() && !other.m_v_init.is_empty() { debug_assert_eq!(self.m_v_init.len(), other.m_v_init.len()); acc(&mut self.m_v_init, &other.m_v_init); }
        if !self.m_q_init.is_empty() && !other.m_q_init.is_empty() { debug_assert_eq!(self.m_q_init.len(), other.m_q_init.len()); acc(&mut self.m_q_init, &other.m_q_init); }
        if !self.m_eta_init.is_empty() && !other.m_eta_init.is_empty() { debug_assert_eq!(self.m_eta_init.len(), other.m_eta_init.len()); acc(&mut self.m_eta_init, &other.m_eta_init); }
        if !self.m_alpha_init.is_empty() && !other.m_alpha_init.is_empty() { debug_assert_eq!(self.m_alpha_init.len(), other.m_alpha_init.len()); acc(&mut self.m_alpha_init, &other.m_alpha_init); }
        if !self.m_mem_init.is_empty() && !other.m_mem_init.is_empty() { debug_assert_eq!(self.m_mem_init.len(), other.m_mem_init.len()); acc(&mut self.m_mem_init, &other.m_mem_init); }
        if !self.gate_proj.is_empty() && !other.gate_proj.is_empty() { debug_assert_eq!(self.gate_proj.len(), other.gate_proj.len()); acc(&mut self.gate_proj, &other.gate_proj); }
        if !self.up_proj.is_empty() && !other.up_proj.is_empty() { debug_assert_eq!(self.up_proj.len(), other.up_proj.len()); acc(&mut self.up_proj, &other.up_proj); }
        if !self.down_proj.is_empty() && !other.down_proj.is_empty() { debug_assert_eq!(self.down_proj.len(), other.down_proj.len()); acc(&mut self.down_proj, &other.down_proj); }
    }

    /// Frobenius norm across all weight matrices.
    /// For Bf16Storage fields: uses master (fp32) copy for norm calculation.
    pub fn norm(&self) -> f32 {
        let mut sum = 0.0f32;
        // Bf16Storage fields — iterate master copy
        for v in [self.w_k_mem.master(), self.w_v_mem.master(), self.w_q_mem.master()] {
            for &x in v.iter() {
                sum += x * x;
            }
        }
        // Plain Vec<f32> fields
        for v in [&self.w_alpha, &self.b_alpha, &self.w_theta, &self.b_theta,
                   &self.w_eta, &self.b_eta, &self.w_omega,
                   &self.w_freq, &self.b_freq,
                   &self.w_k_conv, &self.b_k_conv, &self.w_q_conv, &self.b_q_conv,
                   &self.m_k_init, &self.m_v_init, &self.m_q_init,
                   &self.m_eta_init, &self.m_alpha_init, &self.m_mem_init,
                   &self.gate_proj, &self.up_proj, &self.down_proj] {
            for &x in v.iter() {
                sum += x * x;
            }
        }
        sum.sqrt()
    }

    /// Snap bf16 master copies to match their stored bf16 values.
    /// After this, `w_k_mem.master()[i] == w_k_mem.as_f32()[i]` for all i.
    /// Used before FD gradient checking so get_weight reads what forward sees.
    pub fn snap_bf16_masters(&mut self) {
        self.w_k_mem.snap_master_to_stored();
        self.w_v_mem.snap_master_to_stored();
        self.w_q_mem.snap_master_to_stored();
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
    /// Smooth Sign sharpness for l_p gradient: tanh(a * e) approximates signum(e).
    /// Default: 10.0. See specs/algorithms/attentional_biases/01_l1_sign.md.
    #[serde(default = "default_sign_sharpness")]
    pub sign_sharpness: f32,
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
    /// Gradient checkpointing interval for memory rules.
    /// None = store full M trajectory (no overhead, current behavior).
    /// Some(C) = store M every C steps, recompute during backward.
    #[serde(default)]
    pub checkpoint_interval: Option<usize>,
    /// HOPE §6 level-level composition variant. Determines how CMS levels
    /// interact with each other. Default: FreqGated (Variant 2).
    #[serde(default = "default_hope_variant")]
    pub hope_variant: HopeVariant,
    /// Lattice OSR update variant. Only used when memory_rule == LatticeOSR.
    /// Default: Decode (Eqs 5-6). See Lattice (2504.05646) Eqs 5-8, 24-26.
    #[serde(default = "default_lattice_variant")]
    pub lattice_variant: LatticeVariant,
    /// Number of persistent tokens for MAC composition (Titans Eq 22).
    /// These are learnable, input-independent tokens (outer_loop_param) prepended
    /// to the assembled context. Default: 0 (backward compatible, 2-branch assemble).
    #[serde(default)]
    pub n_persistent: usize,
    /// Which attentional bias (inner-loop loss) to use. Default: L2 (standard Delta rule).
    /// L1 uses smooth tanh approximation. Lp(p) generalizes to arbitrary p-norms.
    /// See specs/algorithms/attentional_biases/03_lp_dispatch.md.
    #[serde(default)]
    pub attentional_bias: AttentionalBias,
    /// Conv1D kernel size for key/query preprocessing. Default: 0 (disabled).
    /// When > 0, depthwise causal Conv1D is applied to keys and queries before
    /// the memory module. See specs/infrastructure/attention/02_short_conv.md.
    #[serde(default)]
    pub kernel_size: usize,
    /// Which momentum variant to use (HOPE §4.2-4.4). Default: None.
    /// Only meaningful for rules with momentum accumulators (TitansLMM, AtlasOmega).
    /// TitansLMM with None auto-upgrades to EMA for backward compat.
    #[serde(default)]
    pub momentum_kind: MomentumKind,
    /// MLP hidden dimension for DeepMomentum. Default: 0 (auto = 4*d).
    /// Ignored for MomentumKind::None/EMA/DeltaMomentum.
    #[serde(default)]
    pub momentum_d_hidden: usize,
    /// Projection style for memory key/value/query generation (HOPE §5).
    /// Static (default) = Phase 1 W @ x. Adaptive = Phase 2 DGD projection memories.
    #[serde(default)]
    pub projection_kind: ProjectionKind,
    /// Phase 3: self-generated value targets (HOPE Eq 84-85).
    /// When true, each memory generates its own DGD target: v̂_□ = M_{□,t-1}(v_t).
    /// When false (default), all memories share v_t as target (Phase 2 behavior).
    /// Only meaningful when projection_kind == Adaptive.
    #[serde(default)]
    pub self_generated_values: bool,
    /// Chunkwise self-referential chunk size (HOPE §8.2, Eqs 90-93).
    /// 1 = sequential (default, bit-identical to self_ref_step).
    /// C > 1 = freeze M at chunk boundaries for gradient computation.
    /// Only meaningful when projection_kind == Adaptive.
    #[serde(default = "default_one")]
    pub self_ref_chunk_size: usize,
    /// Per-level theta (inner-loop lr) floor after softplus activation (CS-39).
    /// Prevents higher CMS levels from collapsing to near-zero learning rate.
    /// Length must match `k`. Default: all zeros (no floor).
    #[serde(default)]
    pub theta_floor: Vec<f32>,
    /// Per-level theta (inner-loop lr) ceiling after softplus activation (CS-39).
    /// Prevents any level from overshooting. Default: empty (no ceiling).
    #[serde(default)]
    pub theta_ceil: Vec<f32>,
    /// Per-level M Frobenius norm ceiling (straight-through in backward).
    /// When ‖M‖_F exceeds this value after the M update, M is rescaled to m_norm_max.
    /// Empty = disabled (legacy behavior). Recommended: 100.0 for d=512 titans.
    #[serde(default)]
    pub m_norm_max: Vec<f32>,
    /// SwiGluMlp intermediate (hidden) dimension. 0 for all matrix-memory rules.
    /// Typically 4*d_model (e.g., 8192 for d=2048 — Llama-3.2-1B MLP size).
    #[serde(default)]
    pub intermediate_size: usize,
}

fn default_one() -> usize { 1 }

fn default_sign_sharpness() -> f32 { 10.0 }

impl MAGConfig {
    /// Clamp a post-softplus theta value using per-level floor/ceil (CS-39).
    /// Returns theta unchanged if no floor/ceil is configured for this level.
    #[inline]
    pub fn clamp_theta(&self, level: usize, theta: f32) -> f32 {
        let floor = self.theta_floor.get(level).copied().unwrap_or(0.0);
        let ceil = self.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
        theta.clamp(floor, ceil)
    }

    /// Returns the M Frobenius norm ceiling for `level`, or f32::MAX if unset/zero.
    /// When the ceiling is f32::MAX, the clamp is effectively disabled.
    #[inline]
    pub fn max_m_norm(&self, level: usize) -> f32 {
        let v = self.m_norm_max.get(level).copied().unwrap_or(0.0);
        if v > 0.0 { v } else { f32::MAX }
    }
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
        4 => 5.5,    // sigmoid(5.5) ≈ 0.996
        5 => 6.0,    // sigmoid(6.0) ≈ 0.998
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
        4 => -8.6,   // softplus(-8.6) ≈ 0.00018
        5 => -9.6,   // softplus(-9.6) ≈ 0.00007
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
        4 => 3.5,    // sigmoid(3.5) ≈ 0.97
        5 => 4.0,    // sigmoid(4.0) ≈ 0.98
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            sign_sharpness: 10.0,
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
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 4, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::LatticeOSR),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 4, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::LatticeOSR),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 0,
            d_compress: 8,
            lambda_k: 0.01,
            lambda_v: 0.01,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0,
            m_slots: 0,
            d_compress: 8,
            lambda_k: 0.01,
            lambda_v: 0.01,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::AtlasOmega),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::AtlasOmega),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Fixed,
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig::default()),
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
            d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0, delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
            parallel: None,
            retention: default_retention(MemoryRuleKind::DeltaRule),
            m3: None,
            frequency_schedule: FrequencySchedule::Learned(LearnedFreqConfig::default()),
            checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
            n_persistent: 0,
            attentional_bias: AttentionalBias::L2,
            kernel_size: 0,
            momentum_kind: MomentumKind::None,
            momentum_d_hidden: 0,
            projection_kind: ProjectionKind::Static,
            self_generated_values: false,
            self_ref_chunk_size: 1,
            theta_floor: vec![],
            theta_ceil: vec![],
            intermediate_size: 0,
            m_norm_max: vec![],
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
    /// Learnable CMS aggregation logits for memory output combination.
    /// [k] f32, init zeros → softmax produces uniform 1/k at init.
    pub alpha_mem: Vec<f32>,
    /// Learnable CMS aggregation logits for reflective gate signal combination.
    /// [k] f32, init zeros → softmax over active subset at init.
    pub alpha_refl: Vec<f32>,
    /// Learnable persistent tokens for MAC composition (Titans Eq 22).
    /// Shape: [n_persistent * d_model]. outer_loop_param — updated by AD.
    /// Empty when n_persistent == 0.
    #[serde(default)]
    pub persistent_tokens: Vec<f32>,
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
            // Initialize Conv1D weights if kernel_size > 0
            if cfg.kernel_size > 0 {
                let mut conv_rng = SimpleRng::new(seed.wrapping_add(7000 + level as u64 * 100));
                level_params.init_conv(d, cfg.kernel_size, &mut conv_rng);
            }
            // Initialize self-referential projection memory inits for Adaptive mode
            if cfg.projection_kind == ProjectionKind::Adaptive {
                level_params.m_k_init = vec![0.0f32; d * d];
                level_params.m_v_init = vec![0.0f32; d * d];
                level_params.m_q_init = vec![0.0f32; d * d];
                level_params.m_eta_init = vec![0.0f32; d * d];
                level_params.m_alpha_init = vec![0.0f32; d * d];
                level_params.m_mem_init = vec![0.0f32; d * d];
            }
            // Initialize SwiGluMlp projections with Xavier scaling
            if cfg.memory_rule == MemoryRuleKind::SwiGluMlp {
                let inter = cfg.intermediate_size;
                assert!(inter > 0, "intermediate_size must be > 0 for SwiGluMlp rule");
                let gate_scale = (2.0 / (d + inter) as f32).sqrt();
                let down_scale = (2.0 / (inter + d) as f32).sqrt();
                let mut mlp_rng = SimpleRng::new(seed.wrapping_add(11000 + level as u64 * 300));
                let mut gp = vec![0.0f32; inter * d];
                mlp_rng.fill_uniform(&mut gp, gate_scale);
                level_params.gate_proj = gp;
                let mut up = vec![0.0f32; inter * d];
                mlp_rng.fill_uniform(&mut up, gate_scale);
                level_params.up_proj = up;
                let mut dp = vec![0.0f32; d * inter];
                mlp_rng.fill_uniform(&mut dp, down_scale);
                level_params.down_proj = dp;
            }
            levels.push(level_params);
        }

        let alpha_mem = vec![0.0f32; cfg.k];
        let alpha_refl = vec![0.0f32; cfg.k];

        // Persistent tokens: Xavier uniform init scaled by 1/sqrt(d)
        let n_p = cfg.n_persistent;
        let persistent_tokens = if n_p > 0 {
            let mut pt_rng = SimpleRng::new(seed.wrapping_add(9000));
            let scale = 1.0 / (d as f32).sqrt();
            let mut pt = vec![0.0f32; n_p * d];
            pt_rng.fill_uniform(&mut pt, scale);
            pt
        } else {
            vec![]
        };

        MAGParams { swa, levels, alpha_mem, alpha_refl, persistent_tokens }
        // NOTE: No weight tying at init — both w_embed and w_unembed keep their
        // independent Kaiming initialization. Weight tying (sync_embed_from_unembed)
        // is applied after each weight update during training, in the Python tier
        // (per CS-18: orchestration belongs in Python).
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
            if cfg.kernel_size > 0 {
                z.w_k_conv = vec![0.0f32; d * cfg.kernel_size];
                z.b_k_conv = vec![0.0f32; d];
                z.w_q_conv = vec![0.0f32; d * cfg.kernel_size];
                z.b_q_conv = vec![0.0f32; d];
            }
            if cfg.projection_kind == ProjectionKind::Adaptive {
                z.m_k_init = vec![0.0f32; d * d];
                z.m_v_init = vec![0.0f32; d * d];
                z.m_q_init = vec![0.0f32; d * d];
                z.m_eta_init = vec![0.0f32; d * d];
                z.m_alpha_init = vec![0.0f32; d * d];
                z.m_mem_init = vec![0.0f32; d * d];
            }
            if cfg.memory_rule == MemoryRuleKind::SwiGluMlp {
                let inter = cfg.intermediate_size;
                z.gate_proj = vec![0.0f32; inter * d];
                z.up_proj = vec![0.0f32; inter * d];
                z.down_proj = vec![0.0f32; d * inter];
            }
            z
        }).collect();
        MAGParams {
            swa: SWAParams::zeros_like(&cfg.swa),
            levels,
            alpha_mem: vec![0.0f32; cfg.k],
            alpha_refl: vec![0.0f32; cfg.k],
            persistent_tokens: vec![0.0f32; cfg.n_persistent * cfg.swa.d_model],
        }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.swa.num_params()
            + self.levels.iter().map(|l| l.num_params()).sum::<usize>()
            + self.alpha_mem.len()
            + self.alpha_refl.len()
            + self.persistent_tokens.len()
    }

    /// Outer-loop weight update: param -= lr * grad for all projection weights across all levels.
    pub fn apply_weight_gradients(&mut self, grads: &MAGParams, lr: f32) {
        self.swa.apply_weight_gradients(&grads.swa, lr);
        for (level, level_grads) in self.levels.iter_mut().zip(grads.levels.iter()) {
            level.apply_weight_gradients(level_grads, lr);
        }
        for (a, &da) in self.alpha_mem.iter_mut().zip(grads.alpha_mem.iter()) {
            *a -= lr * da;
        }
        for (a, &da) in self.alpha_refl.iter_mut().zip(grads.alpha_refl.iter()) {
            *a -= lr * da;
        }
        for (p, &dp) in self.persistent_tokens.iter_mut().zip(grads.persistent_tokens.iter()) {
            *p -= lr * dp;
        }
    }

    /// Weight tying at init: copy w_embed^T → w_unembed.
    /// At initialization, w_embed has proper variance — make w_unembed match.
    /// During training, sync goes the other direction (unembed→embed) since
    /// unembedding gets stronger gradients through the loss.
    pub fn sync_unembed_from_embed(&mut self) {
        let dd = self.swa.w_q.len();
        let d = (dd as f64).sqrt() as usize;
        if d * d != dd || self.swa.w_embed.len() % d != 0 { return; }
        let vocab = self.swa.w_embed.len() / d;
        // w_unembed[i*vocab + v] = w_embed[v*d + i]
        for v in 0..vocab {
            for i in 0..d {
                self.swa.w_unembed[i * vocab + v] = self.swa.w_embed[v * d + i];
            }
        }
    }

    /// Weight tying: copy w_unembed^T → w_embed (CPU path).
    /// Called after each weight update to keep embeddings in sync.
    /// Infers d from w_q.len() = d*d, vocab from w_embed.len() / d.
    pub fn sync_embed_from_unembed(&mut self) {
        let dd = self.swa.w_q.len();
        let d = (dd as f64).sqrt() as usize;
        if d * d != dd || self.swa.w_embed.len() % d != 0 { return; }
        let vocab = self.swa.w_embed.len() / d;
        // w_embed[v*d + i] = w_unembed[i*vocab + v]
        for v in 0..vocab {
            for i in 0..d {
                self.swa.w_embed[v * d + i] = self.swa.w_unembed[i * vocab + v];
            }
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

use crate::conductor::{ConductorState, ContextState};
use crate::context_stream::StreamCursor;

/// Legacy v0 checkpoint format (no version field). Kept for backward compat.
#[derive(Serialize, Deserialize)]
struct ParamCheckpoint {
    config: MAGConfig,
    params: MAGParams,
}

/// Optional build-resume state. Omitted for serving checkpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildResumeState {
    pub conductor: ConductorState,
    pub stream_cursor: StreamCursor,
    pub context: ContextState,
    pub global_step: usize,
}

/// Declared checkpoint format (v1+). Schema-versioned with optional build-resume state.
#[derive(Serialize, Deserialize)]
pub struct DeclaredCheckpoint {
    pub version: u32,
    pub created_at: String,
    pub description: Option<String>,
    pub config: MAGConfig,
    pub params: MAGParams,
    pub build_state: Option<BuildResumeState>,
}

/// Generate ISO 8601 timestamp without chrono dependency.
fn iso8601_now() -> String {
    // Use UNIX_EPOCH elapsed seconds; format as readable timestamp.
    // For a proper ISO 8601, we'd need chrono — but the spec says "no chrono dep".
    // We format epoch seconds as a string; consumers who need human-readable can parse.
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("epoch:{}", d.as_secs()),
        Err(_) => "epoch:0".to_string(),
    }
}

/// Save MAGParams + MAGConfig as a v1 serving checkpoint (no build state).
pub fn save_checkpoint(path: &std::path::Path, params: &MAGParams, config: &MAGConfig) -> std::io::Result<()> {
    let checkpoint = DeclaredCheckpoint {
        version: 1,
        created_at: iso8601_now(),
        description: None,
        config: config.clone(),
        params: params.clone(),
        build_state: None,
    };
    let json = serde_json::to_string(&checkpoint)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Save MAGParams + MAGConfig + build-resume state as a v1 checkpoint.
pub fn save_build_checkpoint(
    path: &std::path::Path,
    params: &MAGParams,
    config: &MAGConfig,
    build_state: BuildResumeState,
) -> std::io::Result<()> {
    let checkpoint = DeclaredCheckpoint {
        version: 1,
        created_at: iso8601_now(),
        description: Some("build checkpoint (resumable)".to_string()),
        config: config.clone(),
        params: params.clone(),
        build_state: Some(build_state),
    };
    let json = serde_json::to_string(&checkpoint)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load checkpoint. Handles both v1 (DeclaredCheckpoint) and legacy v0 (ParamCheckpoint).
/// Returns (params, config, optional build state).
pub fn load_checkpoint(path: &std::path::Path) -> std::io::Result<(MAGParams, MAGConfig, Option<BuildResumeState>)> {
    let json = std::fs::read_to_string(path)?;
    // Try v1 first
    if let Ok(declared) = serde_json::from_str::<DeclaredCheckpoint>(&json) {
        return Ok((declared.params, declared.config, declared.build_state));
    }
    // Fall back to legacy v0
    let legacy: ParamCheckpoint = serde_json::from_str(&json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok((legacy.params, legacy.config, None))
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
        assert!(z.levels[0].w_k_mem.master().iter().all(|&x| x == 0.0));
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
        a.w_k_mem.master_mut()[0] = 1.0;
        b.w_k_mem.master_mut()[0] = 2.0;
        a.b_alpha[0] = 0.5;
        b.b_alpha[0] = 0.3;
        a.accumulate(&b);
        assert!((a.w_k_mem.master()[0] - 3.0).abs() < 1e-6);
        assert!((a.b_alpha[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_memory_level_params_norm() {
        let d = 2;
        let mut p = MemoryLevelParams::zeros_like(d);
        p.w_k_mem.master_mut()[0] = 3.0;
        p.w_k_mem.master_mut()[1] = 4.0;
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
        assert!((default_b_eta(4) - 3.5).abs() < 1e-6);
        assert!((default_b_eta(5) - 4.0).abs() < 1e-6);
        assert!((default_b_eta(6) - 2.0).abs() < 1e-6); // fallback
    }

    // ── Declared Checkpoint tests ────────────────────────────────────

    #[test]
    fn test_checkpoint_v1_roundtrip() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let dir = std::env::temp_dir().join("hecate_test_ckpt_v1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("serving.json");
        save_checkpoint(&path, &params, &cfg).unwrap();
        let (loaded_params, loaded_cfg, build_state) = load_checkpoint(&path).unwrap();
        assert_eq!(loaded_params.swa.w_q, params.swa.w_q);
        assert_eq!(loaded_cfg.swa.d_model, cfg.swa.d_model);
        assert_eq!(loaded_cfg.k, cfg.k);
        assert!(build_state.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_v1_with_build_state() {
        use crate::conductor::{ConductorState, ContextState};
        use crate::context_stream::StreamCursor;

        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;

        let build_state = BuildResumeState {
            conductor: ConductorState { k: 1, chunk_sizes: vec![1], step: 42 },
            stream_cursor: StreamCursor { position: 100, chunk_id: 42, pulse_id: 42, rng_state: None, content_hash: 0 },
            context: ContextState::new(1, d),
            global_step: 42,
        };

        let dir = std::env::temp_dir().join("hecate_test_ckpt_build");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("build.json");
        save_build_checkpoint(&path, &params, &cfg, build_state.clone()).unwrap();

        let (loaded_params, loaded_cfg, loaded_bs) = load_checkpoint(&path).unwrap();
        assert_eq!(loaded_params.swa.w_q, params.swa.w_q);
        assert_eq!(loaded_cfg.k, cfg.k);
        let bs = loaded_bs.expect("build_state should be Some");
        assert_eq!(bs.global_step, 42);
        assert_eq!(bs.conductor.step, 42);
        assert_eq!(bs.stream_cursor.position, 100);
        assert_eq!(bs.context.memory.len(), 1);
        assert_eq!(bs.context.d, d);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_legacy_compat() {
        // Write a raw v0 JSON (no version field) and verify load still works
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let legacy = ParamCheckpoint { config: cfg.clone(), params: params.clone() };
        let json = serde_json::to_string(&legacy).unwrap();

        let dir = std::env::temp_dir().join("hecate_test_ckpt_legacy");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("legacy.json");
        std::fs::write(&path, &json).unwrap();

        let (loaded_params, loaded_cfg, build_state) = load_checkpoint(&path).unwrap();
        assert_eq!(loaded_params.swa.w_q, params.swa.w_q);
        assert_eq!(loaded_cfg.swa.d_model, cfg.swa.d_model);
        assert!(build_state.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_version_field() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);

        let dir = std::env::temp_dir().join("hecate_test_ckpt_version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("versioned.json");
        save_checkpoint(&path, &params, &cfg).unwrap();

        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap()
        ).unwrap();
        assert_eq!(raw["version"], 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_checkpoint_timestamp() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);

        let dir = std::env::temp_dir().join("hecate_test_ckpt_ts");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ts.json");
        save_checkpoint(&path, &params, &cfg).unwrap();

        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap()
        ).unwrap();
        let ts = raw["created_at"].as_str().unwrap();
        assert!(!ts.is_empty());
        assert!(ts.starts_with("epoch:"));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Conv1D field tests ────────────────────────────────────────────

    #[test]
    fn test_gates_eta_field() {
        let g = crate::delta_rule::Gates { alpha: 0.5, theta: 0.1, eta: 0.9 };
        assert_eq!(g.alpha, 0.5);
        assert_eq!(g.theta, 0.1);
        assert_eq!(g.eta, 0.9);
    }

    #[test]
    fn test_conv_fields_disabled() {
        let cfg = MAGConfig::test_config(); // kernel_size: 0
        assert_eq!(cfg.kernel_size, 0);
        let params = MAGParams::init(&cfg, 42);
        for level in &params.levels {
            assert!(level.w_k_conv.is_empty());
            assert!(level.b_k_conv.is_empty());
            assert!(level.w_q_conv.is_empty());
            assert!(level.b_q_conv.is_empty());
        }
    }

    #[test]
    fn test_conv_fields_enabled() {
        let mut cfg = MAGConfig::test_config();
        cfg.kernel_size = 4;
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        for level in &params.levels {
            assert_eq!(level.w_k_conv.len(), d * 4);
            assert_eq!(level.b_k_conv.len(), d);
            assert_eq!(level.w_q_conv.len(), d * 4);
            assert_eq!(level.b_q_conv.len(), d);
            // Bias should be zeros
            assert!(level.b_k_conv.iter().all(|&x| x == 0.0));
            assert!(level.b_q_conv.iter().all(|&x| x == 0.0));
            // Weights should be non-zero (random init)
            assert!(level.w_k_conv.iter().any(|&x| x != 0.0));
            assert!(level.w_q_conv.iter().any(|&x| x != 0.0));
        }
    }

    #[test]
    fn test_conv_num_params() {
        let mut cfg = MAGConfig::test_config();
        let d = cfg.swa.d_model;
        let params_no_conv = MAGParams::init(&cfg, 42);
        let base_count = params_no_conv.num_params();

        cfg.kernel_size = 4;
        let params_conv = MAGParams::init(&cfg, 42);
        let conv_count = params_conv.num_params();
        // Each level adds 2*d*ks + 2*d = 2*d*(ks+1) params for conv
        let expected_extra = cfg.k * (2 * d * 4 + 2 * d);
        assert_eq!(conv_count, base_count + expected_extra);
    }

    #[test]
    fn test_conv_flat_roundtrip() {
        use crate::opaque_adapters::*;
        let mut cfg = MAGConfig::test_config();
        cfg.kernel_size = 4;
        let params = MAGParams::init(&cfg, 42);
        let level = &params.levels[0];

        let mut flat = Vec::new();
        level_params_to_flat(level, &mut flat);
        let reconstructed = level_params_from_flat(&flat, cfg.swa.d_model, cfg.kernel_size);

        assert_eq!(level.w_k_conv, reconstructed.w_k_conv);
        assert_eq!(level.b_k_conv, reconstructed.b_k_conv);
        assert_eq!(level.w_q_conv, reconstructed.w_q_conv);
        assert_eq!(level.b_q_conv, reconstructed.b_q_conv);
    }

    #[test]
    fn test_conv_accumulate() {
        let mut cfg = MAGConfig::test_config();
        cfg.kernel_size = 4;
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        let mut acc = MemoryLevelParams::zeros_like_from(&params.levels[0], d);
        acc.accumulate(&params.levels[0]);
        assert_eq!(acc.w_k_conv, params.levels[0].w_k_conv);
    }

    #[test]
    fn test_conv_zeros_like_from() {
        let mut cfg = MAGConfig::test_config();
        cfg.kernel_size = 4;
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        let z = MemoryLevelParams::zeros_like_from(&params.levels[0], d);
        assert_eq!(z.w_k_conv.len(), params.levels[0].w_k_conv.len());
        assert_eq!(z.b_k_conv.len(), params.levels[0].b_k_conv.len());
        assert!(z.w_k_conv.iter().all(|&x| x == 0.0));
    }

    // ── Self-referential init field tests ─────────────────────────

    fn adaptive_test_config() -> MAGConfig {
        let mut cfg = MAGConfig::test_config();
        cfg.projection_kind = ProjectionKind::Adaptive;
        cfg
    }

    #[test]
    fn test_sr_init_fields_adaptive() {
        let cfg = adaptive_test_config();
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        assert_eq!(params.levels[0].m_k_init.len(), d * d);
        assert_eq!(params.levels[0].m_v_init.len(), d * d);
        assert_eq!(params.levels[0].m_q_init.len(), d * d);
        assert_eq!(params.levels[0].m_eta_init.len(), d * d);
        assert_eq!(params.levels[0].m_alpha_init.len(), d * d);
        assert_eq!(params.levels[0].m_mem_init.len(), d * d);
        // Static config should have empty fields
        let static_params = MAGParams::init(&MAGConfig::test_config(), 42);
        assert!(static_params.levels[0].m_k_init.is_empty());
    }

    #[test]
    fn test_num_params_includes_init() {
        let cfg = adaptive_test_config();
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        let static_params = MAGParams::init(&MAGConfig::test_config(), 42);
        // Adaptive should have 6*d*d more params per level
        assert_eq!(params.num_params(), static_params.num_params() + 6 * d * d);
    }

    #[test]
    fn test_checkpoint_roundtrip_init() {
        let cfg = adaptive_test_config();
        let mut params = MAGParams::init(&cfg, 42);
        // Set some nonzero values
        params.levels[0].m_k_init[0] = 1.5;
        params.levels[0].m_mem_init[3] = -0.7;
        let json = serde_json::to_string(&params).unwrap();
        let back: MAGParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.levels[0].m_k_init[0], 1.5);
        assert_eq!(back.levels[0].m_mem_init[3], -0.7);
        assert_eq!(back.levels[0].m_k_init.len(), params.levels[0].m_k_init.len());
    }

    #[test]
    fn test_checkpoint_file_roundtrip_adaptive() {
        let cfg = adaptive_test_config();
        let mut params = MAGParams::init(&cfg, 42);
        params.levels[0].m_k_init[0] = 1.5;
        params.levels[0].m_v_init[1] = -2.3;
        params.levels[0].m_mem_init[3] = 0.7;

        let dir = std::env::temp_dir().join("hecate_test_ckpt_adaptive");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adaptive.json");
        save_checkpoint(&path, &params, &cfg).unwrap();

        let (loaded, loaded_cfg, _) = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.levels[0].m_k_init[0], 1.5);
        assert_eq!(loaded.levels[0].m_v_init[1], -2.3);
        assert_eq!(loaded.levels[0].m_mem_init[3], 0.7);
        assert_eq!(loaded.levels[0].m_k_init.len(), cfg.swa.d_model * cfg.swa.d_model);
        assert_eq!(loaded_cfg.projection_kind, ProjectionKind::Adaptive);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_old_checkpoint_compat() {
        // Simulate an old checkpoint that has Adaptive-sized params but is missing
        // the m_*_init keys entirely (as would happen with a pre-wiring checkpoint).
        let cfg = adaptive_test_config();
        let params = MAGParams::init(&cfg, 42);
        let json = serde_json::to_string(&params).unwrap();

        // Parse into a mutable JSON value and strip all m_*_init keys from levels
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        if let Some(levels) = val.get_mut("levels").and_then(|v| v.as_array_mut()) {
            for level in levels.iter_mut() {
                if let Some(obj) = level.as_object_mut() {
                    obj.remove("m_k_init");
                    obj.remove("m_v_init");
                    obj.remove("m_q_init");
                    obj.remove("m_eta_init");
                    obj.remove("m_alpha_init");
                    obj.remove("m_mem_init");
                }
            }
        }

        // Deserialize the stripped JSON — #[serde(default)] should give empty vecs
        let edited_json = serde_json::to_string(&val).unwrap();
        let back: MAGParams = serde_json::from_str(&edited_json).unwrap();
        assert!(back.levels[0].m_k_init.is_empty(), "m_k_init should default to empty");
        assert!(back.levels[0].m_v_init.is_empty(), "m_v_init should default to empty");
        assert!(back.levels[0].m_q_init.is_empty(), "m_q_init should default to empty");
        assert!(back.levels[0].m_eta_init.is_empty(), "m_eta_init should default to empty");
        assert!(back.levels[0].m_alpha_init.is_empty(), "m_alpha_init should default to empty");
        assert!(back.levels[0].m_mem_init.is_empty(), "m_mem_init should default to empty");
    }

    #[test]
    fn test_accumulate_with_init() {
        let cfg = adaptive_test_config();
        let grads = MAGParams::zeros_like(&cfg);
        // Create two gradient sets
        let mut g1 = grads.clone();
        g1.levels[0].m_k_init[0] = 1.0;
        g1.levels[0].m_mem_init[5] = 2.0;
        let mut g2 = MAGParams::zeros_like(&cfg);
        g2.levels[0].m_k_init[0] = 3.0;
        g2.levels[0].m_mem_init[5] = 4.0;
        // Accumulate
        let mut acc = MAGParams::zeros_like(&cfg);
        acc.levels[0].accumulate(&g1.levels[0]);
        acc.levels[0].accumulate(&g2.levels[0]);
        assert_eq!(acc.levels[0].m_k_init[0], 4.0);
        assert_eq!(acc.levels[0].m_mem_init[5], 6.0);
    }

    #[test]
    fn test_apply_weight_gradients_init() {
        let cfg = adaptive_test_config();
        let mut params = MAGParams::init(&cfg, 42);
        params.levels[0].m_k_init[0] = 1.0;
        params.levels[0].m_mem_init[0] = 2.0;
        let mut grads = MAGParams::zeros_like(&cfg);
        grads.levels[0].m_k_init[0] = 0.5;
        grads.levels[0].m_mem_init[0] = 3.0;
        params.levels[0].apply_weight_gradients(&grads.levels[0], 0.1);
        // param -= lr * grad → 1.0 - 0.1 * 0.5 = 0.95
        assert!((params.levels[0].m_k_init[0] - 0.95).abs() < 1e-7);
        // 2.0 - 0.1 * 3.0 = 1.7
        assert!((params.levels[0].m_mem_init[0] - 1.7).abs() < 1e-7);
    }

    #[test]
    fn test_zeros_like_from_with_init() {
        let cfg = adaptive_test_config();
        let d = cfg.swa.d_model;
        let params = MAGParams::init(&cfg, 42);
        let z = MemoryLevelParams::zeros_like_from(&params.levels[0], d);
        assert_eq!(z.m_k_init.len(), d * d);
        assert_eq!(z.m_mem_init.len(), d * d);
        assert!(z.m_k_init.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_norm_includes_init() {
        let cfg = adaptive_test_config();
        let mut p = MAGParams::zeros_like(&cfg);
        // Zero everywhere — norm should be 0
        assert_eq!(p.levels[0].norm(), 0.0);
        // Set one init field — norm should be nonzero
        p.levels[0].m_k_init[0] = 1.0;
        assert!(p.levels[0].norm() > 0.0);
        assert!((p.levels[0].norm() - 1.0).abs() < 1e-7);
    }
}
