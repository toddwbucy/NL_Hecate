/// Titans LMM (Long-term Memory Module) — GD + Momentum memory rule.
///
/// Extends Delta Rule with a momentum accumulator S (Titans Eqs 12-15, 32-33).
/// MIRAS knobs: matrix structure, L2 bias, L2 decay retention, GD+momentum algorithm.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   theta_t = softplus(concat(k_t, v_t) @ w_theta + b_theta)
///   eta_t   = sigmoid(concat(k_t, v_t) @ w_eta + b_eta)        // NEW: momentum gate
///   prediction = M_t @ k_t
///   error = prediction - v_t
///   grad = outer(error, k_t)
///   S_{t+1} = eta_t * S_t - theta_t * grad                     // NEW: momentum accumulator
///   M_{t+1} = (1-alpha_t) * M_t + S_{t+1}                      // Memory uses momentum
///   y_t = M_{t+1} @ q_t
///
/// Backward: reverse token loop with accumulated d_M AND d_S (two recurrences).

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softplus_f32,
    outer_product_f32, frobenius_dot_f32,
};
use crate::retention::l2_apply_retention;
use crate::model::{MemoryLevelParams, MemoryActivation};
use crate::moneta::{apply_attentional_bias, apply_attentional_bias_backward};
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};
use crate::feature_map::{self, FeatureMapKind};

// ══════════════════════════════════════════════════════════════════════
// MLP Memory — Deep Neural Memory (spec 75, Titans §3.1 Eqs 8-15)
// ══════════════════════════════════════════════════════════════════════

/// Describes the flat-buffer layout for an L_M-layer MLP memory.
///
/// All weight and bias parameters are packed into a single contiguous
/// `Vec<f32>` so that momentum/retention/M-norm can operate element-wise
/// on the entire buffer, identical to the linear d×d case.
#[derive(Clone, Debug)]
pub struct MLPMemoryLayout {
    pub n_layers: usize,
    pub d: usize,
    pub d_h: usize,
    pub total_params: usize,
    pub layers: Vec<MLPLayerDesc>,
}

/// Per-layer descriptor within the flat buffer.
#[derive(Clone, Debug)]
pub struct MLPLayerDesc {
    pub w_offset: usize,
    pub w_rows: usize,
    pub w_cols: usize,
    pub b_offset: usize,
    pub b_size: usize,
}

impl MLPMemoryLayout {
    /// Build layout for L_M layers with given d (head_dim) and expansion factor.
    ///
    /// Layer shapes for L_M=2: W₁[d_h, d], b₁[d_h], W₂[d, d_h], b₂[d].
    /// General: layer 1 expands d→d_h, layers 2..L-1 are d_h→d_h, layer L projects d_h→d.
    pub fn new(n_layers: usize, d: usize, expansion_factor: usize) -> Self {
        assert!(n_layers >= 2, "MLP memory requires at least 2 layers, got {n_layers}");
        let d_h = expansion_factor * d;
        let mut layers = Vec::with_capacity(n_layers);
        let mut offset = 0;

        for l in 0..n_layers {
            let (rows, cols) = if l == 0 {
                (d_h, d) // expand: d → d_h
            } else if l == n_layers - 1 {
                (d, d_h) // project: d_h → d
            } else {
                (d_h, d_h) // hidden: d_h → d_h
            };
            let w_size = rows * cols;
            let w_offset = offset;
            offset += w_size;
            let b_offset = offset;
            let b_size = rows;
            offset += b_size;
            layers.push(MLPLayerDesc { w_offset, w_rows: rows, w_cols: cols, b_offset, b_size });
        }

        MLPMemoryLayout { n_layers, d, d_h, total_params: offset, layers }
    }

    /// Weight slice for layer l within a state buffer at the given base offset.
    #[inline]
    pub fn w_slice<'a>(&self, buf: &'a [f32], base: usize, l: usize) -> &'a [f32] {
        let desc = &self.layers[l];
        &buf[base + desc.w_offset..base + desc.w_offset + desc.w_rows * desc.w_cols]
    }

    /// Mutable weight slice.
    #[inline]
    pub fn w_slice_mut<'a>(&self, buf: &'a mut [f32], base: usize, l: usize) -> &'a mut [f32] {
        let desc = &self.layers[l];
        &mut buf[base + desc.w_offset..base + desc.w_offset + desc.w_rows * desc.w_cols]
    }

    /// Bias slice for layer l.
    #[inline]
    pub fn b_slice<'a>(&self, buf: &'a [f32], base: usize, l: usize) -> &'a [f32] {
        let desc = &self.layers[l];
        &buf[base + desc.b_offset..base + desc.b_offset + desc.b_size]
    }

    /// Mutable bias slice.
    #[inline]
    pub fn b_slice_mut<'a>(&self, buf: &'a mut [f32], base: usize, l: usize) -> &'a mut [f32] {
        let desc = &self.layers[l];
        &mut buf[base + desc.b_offset..base + desc.b_offset + desc.b_size]
    }
}

// ── Activation functions and derivatives ────────────────────────────

/// GELU approximation matching PyTorch: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
#[inline]
fn gelu_f32(x: f32) -> f32 {
    let c = 0.7978845608028654f32; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GELU derivative: Φ(x) + x·φ(x), approximated via the tanh form.
#[inline]
fn gelu_derivative_f32(x: f32) -> f32 {
    let c = 0.7978845608028654f32;
    let a = 0.044715f32;
    let inner = c * (x + a * x * x * x);
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let d_inner = c * (1.0 + 3.0 * a * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
}

/// SiLU (Swish): x * sigmoid(x)
#[inline]
fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

/// SiLU derivative: σ(x) + x·σ(x)·(1-σ(x)) = σ(x)(1 + x(1-σ(x)))
#[inline]
fn silu_derivative_f32(x: f32) -> f32 {
    let s = sigmoid_f32(x);
    s * (1.0 + x * (1.0 - s))
}

/// ReLU: max(0, x)
#[inline]
fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

/// ReLU derivative: 1 if x > 0, else 0
#[inline]
fn relu_derivative_f32(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Apply activation function element-wise.
fn apply_activation(dst: &mut [f32], activation: MemoryActivation) {
    match activation {
        MemoryActivation::GELU => {
            for x in dst.iter_mut() { *x = gelu_f32(*x); }
        }
        MemoryActivation::SiLU => {
            for x in dst.iter_mut() { *x = silu_f32(*x); }
        }
        MemoryActivation::ReLU => {
            for x in dst.iter_mut() { *x = relu_f32(*x); }
        }
    }
}

/// Apply activation derivative element-wise, given pre-activation values.
fn apply_activation_derivative(dst: &mut [f32], pre_act: &[f32], activation: MemoryActivation) {
    match activation {
        MemoryActivation::GELU => {
            for (d, &p) in dst.iter_mut().zip(pre_act.iter()) {
                *d *= gelu_derivative_f32(p);
            }
        }
        MemoryActivation::SiLU => {
            for (d, &p) in dst.iter_mut().zip(pre_act.iter()) {
                *d *= silu_derivative_f32(p);
            }
        }
        MemoryActivation::ReLU => {
            for (d, &p) in dst.iter_mut().zip(pre_act.iter()) {
                *d *= relu_derivative_f32(p);
            }
        }
    }
}

// ── MLP forward / inner-loop backward ───────────────────────────────

/// MLP forward pass: evaluate M(input) given weights in flat buffer.
///
/// Returns the output vector (d-dim) and per-layer pre-activation values
/// needed for the analytical backward.
///
/// `state_buf`: flat buffer of MLP params at base offset `base`.
/// `input`: d-dim input vector (k_t or q_t).
/// `layout`: MLP buffer layout.
/// `activation`: which nonlinearity.
///
/// Returns (output[d], pre_activations[n_layers], activations[n_layers+1])
/// where activations[0] = input and activations[l] = σ(W_l @ activations[l-1] + b_l).
pub fn mlp_forward(
    state_buf: &[f32],
    base: usize,
    input: &[f32],
    layout: &MLPMemoryLayout,
    activation: MemoryActivation,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = layout.n_layers;
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n + 1);
    let mut pre_acts: Vec<Vec<f32>> = Vec::with_capacity(n);
    activations.push(input.to_vec());

    for l in 0..n {
        let desc = &layout.layers[l];
        let w = layout.w_slice(state_buf, base, l);
        let b = layout.b_slice(state_buf, base, l);
        let h_prev = &activations[l];

        // pre_act = W @ h_prev + b
        let out_dim = desc.w_rows;
        let in_dim = desc.w_cols;
        let mut pre_act = vec![0.0f32; out_dim];
        matmul_f32(w, h_prev, &mut pre_act, out_dim, in_dim, 1);
        for i in 0..out_dim {
            pre_act[i] += b[i];
        }

        pre_acts.push(pre_act.clone());

        // Apply activation (all layers except last)
        if l < n - 1 {
            apply_activation(&mut pre_act, activation);
        }
        activations.push(pre_act);
    }

    let output = activations[n].clone();
    (output, pre_acts, activations)
}

/// MLP inner-loop analytical backward: compute gradients of L2 loss w.r.t.
/// all MLP weight/bias parameters.
///
/// Given `d_out` (gradient w.r.t. MLP output, e.g. `2 * error` for L2 loss),
/// computes grad_W_l and grad_b_l for each layer and writes them into `grad_buf`
/// using the same flat layout as the state buffer.
///
/// `pre_acts[l]`: pre-activation values from forward pass.
/// `activations[l]`: post-activation values (activations[0] = input).
pub fn mlp_inner_backward(
    d_out_init: &[f32],
    pre_acts: &[Vec<f32>],
    activations: &[Vec<f32>],
    state_buf: &[f32],
    base: usize,
    layout: &MLPMemoryLayout,
    activation: MemoryActivation,
    grad_buf: &mut [f32],
    grad_base: usize,
) {
    let n = layout.n_layers;
    let mut d_out = d_out_init.to_vec();

    for l in (0..n).rev() {
        let desc = &layout.layers[l];
        let h_prev = &activations[l];

        // Apply activation derivative (all layers except last)
        if l < n - 1 {
            apply_activation_derivative(&mut d_out, &pre_acts[l], activation);
        }

        // grad_W_l = outer(d_out, h_prev)
        let gw = layout.w_slice_mut(grad_buf, grad_base, l);
        let out_dim = desc.w_rows;
        let in_dim = desc.w_cols;
        for i in 0..out_dim {
            for j in 0..in_dim {
                gw[i * in_dim + j] = d_out[i] * h_prev[j];
            }
        }

        // grad_b_l = d_out
        let gb = layout.b_slice_mut(grad_buf, grad_base, l);
        gb.copy_from_slice(&d_out[..desc.b_size]);

        // Propagate to previous layer: d_out = W_l^T @ d_out
        if l > 0 {
            let w = layout.w_slice(state_buf, base, l);
            let mut d_prev = vec![0.0f32; in_dim];
            // W is [out_dim, in_dim], we need W^T @ d_out = [in_dim] vector
            for j in 0..in_dim {
                let mut sum = 0.0f32;
                for i in 0..out_dim {
                    sum += w[i * in_dim + j] * d_out[i];
                }
                d_prev[j] = sum;
            }
            d_out = d_prev;
        }
    }
}

/// MLP backward w.r.t. both weights AND input (outer-loop VJP helper).
///
/// Unlike `mlp_inner_backward` which overwrites weight gradients in `grad_buf`,
/// this function ACCUMULATES weight gradients into `d_weights` and also returns
/// `d_input`. Needed for the outer-loop backward through the readout path
/// `y = MLP(M, q)` and the prediction path `pred = MLP(M, k)`.
///
/// `d_output`: gradient w.r.t. MLP output.
/// `pre_acts`, `activations`: from `mlp_forward`.
/// `state_buf` at `base`: MLP weights (used for W^T propagation).
/// `d_weights` at `d_base`: buffer for weight gradient accumulation.
///
/// Returns `d_input`.
pub fn mlp_backward_full(
    d_output: &[f32],
    pre_acts: &[Vec<f32>],
    activations: &[Vec<f32>],
    state_buf: &[f32],
    base: usize,
    layout: &MLPMemoryLayout,
    activation: MemoryActivation,
    d_weights: &mut [f32],
    d_base: usize,
) -> Vec<f32> {
    let n = layout.n_layers;
    let mut d_out = d_output.to_vec();

    for l in (0..n).rev() {
        let desc = &layout.layers[l];
        let h_prev = &activations[l];

        if l < n - 1 {
            apply_activation_derivative(&mut d_out, &pre_acts[l], activation);
        }

        // grad_W_l += outer(d_out, h_prev)
        let out_dim = desc.w_rows;
        let in_dim = desc.w_cols;
        let gw = layout.w_slice_mut(d_weights, d_base, l);
        for i in 0..out_dim {
            for j in 0..in_dim {
                gw[i * in_dim + j] += d_out[i] * h_prev[j];
            }
        }

        // grad_b_l += d_out
        let gb = layout.b_slice_mut(d_weights, d_base, l);
        for i in 0..desc.b_size {
            gb[i] += d_out[i];
        }

        // Propagate: d_out = W_l^T @ d_out (always, including l=0 for d_input)
        let w = layout.w_slice(state_buf, base, l);
        let mut d_prev = vec![0.0f32; in_dim];
        for j in 0..in_dim {
            let mut sum = 0.0f32;
            for i in 0..out_dim {
                sum += w[i * in_dim + j] * d_out[i];
            }
            d_prev[j] = sum;
        }
        d_out = d_prev;
    }

    d_out // d_input
}

/// Frozen MLP read-only forward: y_t = M_mlp(q_t) for each token.
/// Used for frozen CMS levels where M is not updated.
/// Returns (y, q_mem) — same contract as delta_rule_read_only.
pub fn titans_mlp_read_only(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    frozen_m: &[f32],
    seq_len: usize,
    d: usize,
    memory_layers: usize,
    expansion_factor: usize,
    activation: MemoryActivation,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(embedded.len(), seq_len * d);
    let layout = MLPMemoryLayout::new(memory_layers, d, expansion_factor);
    debug_assert_eq!(frozen_m.len(), layout.total_params);

    let mut q_mem = vec![0.0f32; seq_len * d];
    let w_q_f32 = level_params.w_q_mem.as_f32();
    crate::dispatch::matmul_transb_dispatch(embedded, &w_q_f32, &mut q_mem, seq_len, d, d);

    let mut y = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let (out, _, _) = mlp_forward(frozen_m, 0, q_t, &layout, activation);
        y[t * d..(t + 1) * d].copy_from_slice(&out);
    }

    (y, q_mem)
}

/// Frozen MLP read-only backward: d_q via MLP Jacobian, then d_W_Q_mem and d_embedded.
pub fn titans_mlp_read_only_backward(
    level_params: &MemoryLevelParams,
    frozen_m: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    memory_layers: usize,
    expansion_factor: usize,
    activation: MemoryActivation,
) -> (MemoryLevelParams, Vec<f32>) {
    let layout = MLPMemoryLayout::new(memory_layers, d, expansion_factor);
    let mut grads = MemoryLevelParams::zeros_like(d);
    let mut d_q_mem = vec![0.0f32; seq_len * d];

    // Allocate once outside loop — accumulated values are discarded (frozen M has no weight grads)
    let mut dummy_dw = vec![0.0f32; layout.total_params];
    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];
        // Recompute forward to get pre_acts/activations for backward
        let (_, pre_acts, activations) = mlp_forward(frozen_m, 0, q_t, &layout, activation);
        // MLP backward: d_input only (no weight gradients for frozen M)
        let d_q_t = mlp_backward_full(
            d_y_t, &pre_acts, &activations, frozen_m, 0, &layout, activation,
            &mut dummy_dw, 0,
        );
        d_q_mem[t * d..(t + 1) * d].copy_from_slice(&d_q_t);
    }

    // q_mem = embedded @ W_Q_mem^T → d_W_Q_mem, d_embedded
    let mut d_q_mem_t = vec![0.0f32; d * seq_len];
    transpose_f32(&d_q_mem, &mut d_q_mem_t, seq_len, d);
    crate::dispatch::matmul_dispatch(&d_q_mem_t, embedded, grads.w_q_mem.master_mut(), d, seq_len, d);

    let mut d_embedded = vec![0.0f32; seq_len * d];
    let w_q_f32 = level_params.w_q_mem.as_f32();
    crate::dispatch::matmul_acc_dispatch(&d_q_mem, &w_q_f32, &mut d_embedded, seq_len, d, d);

    (grads, d_embedded)
}

// ── Titans LMM implementation ───────────────────────────────────────

/// Titans LMM: GD + momentum memory rule (Titans Eqs 12-15, 32-33).
/// `bias` controls the inner-loop loss: L2 (default), L1, or Lp(p).
/// `sign_sharpness` controls the tanh approximation steepness for non-L2 biases.
/// `momentum_kind` selects the momentum variant (EMA/Delta/Deep).
///
/// When `memory_layers >= 2` (spec 75), memory M becomes a multi-layer MLP:
///   M(x) = W₂ @ σ(W₁ @ x + b₁) + b₂
/// The inner loop computes analytical gradients through the MLP layers,
/// and momentum/retention/M-norm operate element-wise on the flat parameter buffer.
pub struct TitansLMM {
    pub bias: crate::model::AttentionalBias,
    pub sign_sharpness: f32,
    pub momentum_kind: crate::model::MomentumKind,
    pub momentum_d_hidden: usize,
    /// Per-level alpha floor/ceil (CS-39 retention gate clamp). 0.0/1.0 = no clamp.
    pub alpha_floor: f32,
    pub alpha_ceil: f32,
    /// Per-level theta floor/ceil (CS-39 training wheels). 0.0/MAX = no clamp.
    pub theta_floor: f32,
    pub theta_ceil: f32,
    /// M Frobenius norm ceiling (straight-through). f32::MAX = disabled.
    pub m_norm_max: f32,
    /// Feature map applied to k_t and q_t before memory operations (φ).
    pub feature_map: FeatureMapKind,
    /// Number of MLP layers (1 = linear d×d matrix, 2+ = deep MLP). Spec 75.
    pub memory_layers: usize,
    /// Expansion factor for MLP hidden dim: d_h = factor × d. Default: 4.
    pub memory_expansion_factor: usize,
    /// Activation function for MLP layers. Default: GELU.
    pub memory_activation: MemoryActivation,
}

impl TitansLMM {
    /// L2 bias (backward compatible default — EMA momentum, linear memory).
    pub fn l2() -> Self {
        TitansLMM {
            bias: crate::model::AttentionalBias::L2,
            sign_sharpness: 10.0,
            momentum_kind: crate::model::MomentumKind::EMA,
            momentum_d_hidden: 0,
            alpha_floor: 0.0,
            alpha_ceil: 1.0,
            theta_floor: 0.0,
            theta_ceil: f32::MAX,
            m_norm_max: f32::MAX,
            feature_map: FeatureMapKind::Identity,
            memory_layers: 1,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
        }
    }
    /// Construct from MAGConfig fields for a specific CMS level.
    pub fn from_cfg_level(cfg: &crate::model::MAGConfig, level: usize) -> Self {
        let mk = crate::momentum::effective_momentum_kind(cfg.momentum_kind, cfg.memory_rule);
        TitansLMM {
            bias: cfg.attentional_bias,
            sign_sharpness: cfg.sign_sharpness,
            momentum_kind: mk,
            momentum_d_hidden: cfg.momentum_d_hidden,
            alpha_floor: cfg.alpha_floor.get(level).copied().unwrap_or(0.0),
            alpha_ceil: cfg.alpha_ceil.get(level).copied().unwrap_or(1.0),
            theta_floor: cfg.theta_floor.get(level).copied().unwrap_or(0.0),
            theta_ceil: cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX),
            m_norm_max: cfg.max_m_norm(level),
            feature_map: cfg.feature_map.clone(),
            memory_layers: cfg.memory_layers,
            memory_expansion_factor: cfg.memory_expansion_factor,
            memory_activation: cfg.memory_activation,
        }
    }
    /// Construct from MAGConfig fields (backward compat — no level clamp).
    pub fn from_cfg(cfg: &crate::model::MAGConfig) -> Self {
        Self::from_cfg_level(cfg, 0)
    }

    // ── MLP memory forward path (spec 75) ───────────────────────────

    /// Full sequence forward using MLP memory (memory_layers >= 2).
    ///
    /// The memory M is an L_M-layer MLP whose parameters are updated
    /// per-token by analytical gradient descent. Momentum and retention
    /// operate element-wise on the flat parameter buffer.
    fn step_mlp(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, TitansLMMCache) {
        debug_assert_eq!(embedded.len(), seq_len * d);
        assert!(
            matches!(self.momentum_kind,
                crate::model::MomentumKind::EMA | crate::model::MomentumKind::None),
            "MLP memory (memory_layers >= 2) only supports EMA momentum; \
             got {:?}. DeltaMomentum/DeepMomentum for MLP deferred to Phase B.",
            self.momentum_kind
        );

        let layout = MLPMemoryLayout::new(self.memory_layers, d, self.memory_expansion_factor);
        let state_size = layout.total_params;

        // ── Shared preamble: project embedded → k_mem, v_mem, q_mem ──
        let mut w_k_mem_t = vec![0.0f32; d * d];
        let mut w_v_mem_t = vec![0.0f32; d * d];
        let mut w_q_mem_t = vec![0.0f32; d * d];
        let w_k_f32 = level_params.w_k_mem.as_f32();
        let w_v_f32 = level_params.w_v_mem.as_f32();
        let w_q_f32 = level_params.w_q_mem.as_f32();
        transpose_f32(&w_k_f32, &mut w_k_mem_t, d, d);
        transpose_f32(&w_v_f32, &mut w_v_mem_t, d, d);
        transpose_f32(&w_q_f32, &mut w_q_mem_t, d, d);

        let mut k_mem = vec![0.0f32; seq_len * d];
        let mut v_mem = vec![0.0f32; seq_len * d];
        let mut q_mem = vec![0.0f32; seq_len * d];
        matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
        matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
        matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

        // Conv1D key/query preprocessing
        let (k_conv_cache, q_conv_cache) = crate::conv1d::apply_conv1d_to_kq(
            &mut k_mem, &mut q_mem, level_params, seq_len, d);

        // L2-normalize keys and queries
        let k_mem_norms = crate::tensor::l2_normalize_rows(&mut k_mem, seq_len, d);
        let q_mem_norms = crate::tensor::l2_normalize_rows(&mut q_mem, seq_len, d);

        // Feature map setup
        let has_fm = !matches!(self.feature_map, FeatureMapKind::Identity);
        let mut fm_z_k_mem = if has_fm { vec![0.0f32; seq_len * d] } else { vec![] };
        let mut fm_z_q_mem = if has_fm { vec![0.0f32; seq_len * d] } else { vec![] };
        let mut phi_k_buf = vec![0.0f32; d];
        let mut phi_q_buf = vec![0.0f32; d];
        let mut z_k_buf = if has_fm { vec![0.0f32; d] } else { vec![] };
        let mut z_q_buf = if has_fm { vec![0.0f32; d] } else { vec![] };

        // ── MLP-specific allocations ─────────────────────────────────
        let mut m_states = vec![0.0f32; (seq_len + 1) * state_size];
        let mut s_states = vec![0.0f32; (seq_len + 1) * state_size];
        if let Some(m0) = initial_m {
            assert_eq!(m0.len(), state_size,
                "MLP initial_m size mismatch: expected {state_size}, got {}", m0.len());
            m_states[..state_size].copy_from_slice(&m0);
        }

        // Gate buffers (same as linear path)
        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut theta_pre = vec![0.0f32; seq_len];
        let mut theta = vec![0.0f32; seq_len];
        let mut eta_pre = vec![0.0f32; seq_len];
        let mut eta = vec![0.0f32; seq_len];
        let mut error = vec![0.0f32; seq_len * d];
        let mut y = vec![0.0f32; seq_len * d];

        // Reusable gradient buffer (cleared each token)
        let mut grad_buf = vec![0.0f32; state_size];

        // Per-token MLP activation caches (for outer-loop backward)
        let mut mlp_k_pre_acts: Vec<Vec<Vec<f32>>> = Vec::with_capacity(seq_len);
        let mut mlp_k_activations: Vec<Vec<Vec<f32>>> = Vec::with_capacity(seq_len);

        // ── Sequential token loop ────────────────────────────────────
        for t in 0..seq_len {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];

            // Feature map
            let (phi_k_t, phi_q_t): (&[f32], &[f32]) = if has_fm {
                feature_map::apply_into(k_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, &mut phi_k_buf, &mut z_k_buf, d);
                feature_map::apply_into(q_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, &mut phi_q_buf, &mut z_q_buf, d);
                fm_z_k_mem[t * d..(t + 1) * d].copy_from_slice(&z_k_buf);
                fm_z_q_mem[t * d..(t + 1) * d].copy_from_slice(&z_q_buf);
                (&phi_k_buf, &phi_q_buf)
            } else {
                (k_t, q_t)
            };

            // ── Gate computation (shared with linear) ────────────────
            let c_base = t * 2 * d;
            concat_kv[c_base..c_base + d].copy_from_slice(k_t);
            concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
            let concat_t = &concat_kv[c_base..c_base + 2 * d];

            let mut alpha_pre_t = level_params.b_alpha[0];
            for i in 0..(2 * d) { alpha_pre_t += concat_t[i] * level_params.w_alpha[i]; }
            alpha_pre[t] = alpha_pre_t;
            alpha[t] = sigmoid_f32(alpha_pre_t).clamp(self.alpha_floor, self.alpha_ceil);

            let mut theta_pre_t = level_params.b_theta[0];
            for i in 0..(2 * d) { theta_pre_t += concat_t[i] * level_params.w_theta[i]; }
            theta_pre[t] = theta_pre_t;
            theta[t] = softplus_f32(theta_pre_t).clamp(self.theta_floor, self.theta_ceil);

            let mut eta_pre_t = level_params.b_eta[0];
            for i in 0..(2 * d) { eta_pre_t += concat_t[i] * level_params.w_eta[i]; }
            eta_pre[t] = eta_pre_t;
            eta[t] = sigmoid_f32(eta_pre_t);

            let eta_t = eta[t];
            let theta_t = theta[t];

            // ── 1. MLP forward on φ(k_t) → prediction ───────────────
            let m_base = t * state_size;
            let (prediction, pre_acts, activations) = mlp_forward(
                &m_states, m_base, phi_k_t, &layout, self.memory_activation);

            // ── 2. Error = prediction - v_t ──────────────────────────
            let e_base = t * d;
            for i in 0..d {
                error[e_base + i] = prediction[i] - v_t[i];
            }

            // ── 3. Attentional bias → d_out for MLP backward ────────
            let biased = apply_attentional_bias(
                &error[e_base..e_base + d], self.bias, self.sign_sharpness);

            // ── 4. Analytical backward through MLP → grad_buf ────────
            grad_buf.fill(0.0);
            mlp_inner_backward(
                &biased, &pre_acts, &activations,
                &m_states, m_base, &layout, self.memory_activation,
                &mut grad_buf, 0);

            // Cache activations for outer-loop backward
            mlp_k_pre_acts.push(pre_acts);
            mlp_k_activations.push(activations);

            // ── 5. EMA momentum: S_{t+1} = η·S_t - θ·grad ──────────
            let s_base = t * state_size;
            let s_next = (t + 1) * state_size;
            for i in 0..state_size {
                s_states[s_next + i] = eta_t * s_states[s_base + i]
                                     - theta_t * grad_buf[i];
            }

            // ── 6. Retention + momentum: M_{t+1} = (1-α)·M_t + S_{t+1}
            let m_next = (t + 1) * state_size;
            m_states.copy_within(m_base..m_base + state_size, m_next);
            l2_apply_retention(&mut m_states[m_next..m_next + state_size], 1.0 - alpha[t]);
            for i in 0..state_size {
                m_states[m_next + i] += s_states[s_next + i];
            }

            // ── 7. M-norm clamp (Frobenius norm over entire MLP state)
            if self.m_norm_max < f32::MAX {
                let slice = &mut m_states[m_next..m_next + state_size];
                let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > self.m_norm_max {
                    let scale = self.m_norm_max / norm;
                    for x in slice.iter_mut() { *x *= scale; }
                }
            }

            // ── 8. Read: y_t = M_{t+1}(φ(q_t)) ─────────────────────
            let (y_t, _, _) = mlp_forward(
                &m_states, m_next, phi_q_t, &layout, self.memory_activation);
            y[t * d..(t + 1) * d].copy_from_slice(&y_t);
        }

        // ── Build cache ──────────────────────────────────────────────
        let cache = TitansLMMCache {
            seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
            error,
            grad_outer: Vec::new(), // MLP path doesn't use outer product
            y: y.clone(),
            fm_z_k_mem, fm_z_q_mem,
            momentum_kind: self.momentum_kind,
            decay: Vec::new(),
            deep_cache: None,
            deep_d_hidden: 0,
            k_conv_cache, q_conv_cache,
            k_mem_norms, q_mem_norms,
            mlp_layout: Some(layout),
            mlp_activation: self.memory_activation,
            mlp_k_pre_acts,
            mlp_k_activations,
        };

        (y, cache)
    }

    /// Full sequence backward through MLP memory (memory_layers >= 2).
    ///
    /// Follows the algorithm from the CUDA kernel (titans_mlp_backward.cu):
    ///   Phase 1: Readout backward — y_t = MLP(M_{t+1}, q_t)
    ///   Phase 2: Retention backward — d_S += d_M, d_alpha, d_M *= (1-α)
    ///   Phase 3: Recompute inner gradient from cached activations
    ///   Phase 4: Momentum backward (EMA) → d_eta, d_theta, d_grad
    ///   Phase 5: Second-order inner gradient VJP → d_biased, cross-terms
    ///   Phase 6: Combined prediction + activation path backward → d_k, d_v, d_M
    fn step_backward_mlp(
        &self,
        level_params: &MemoryLevelParams,
        cache: &TitansLMMCache,
        d_y: &[f32],
        embedded: &[f32],
        layout: &MLPMemoryLayout,
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        let n = layout.n_layers;
        let state_size = layout.total_params;
        let activation = cache.mlp_activation;

        assert!(matches!(cache.momentum_kind,
            crate::model::MomentumKind::EMA | crate::model::MomentumKind::None),
            "MLP memory backward only supports EMA momentum; got {:?}", cache.momentum_kind);

        let mut grads = MemoryLevelParams::zeros_like_from(level_params, d);

        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        let mut d_m = vec![0.0f32; state_size];
        let mut d_s = vec![0.0f32; state_size];

        let has_fm = !cache.fm_z_k_mem.is_empty();

        // Reverse token loop
        for t in (0..s).rev() {
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let m_t_base = t * state_size;
            let m_next_base = (t + 1) * state_size;
            let s_t = &cache.s_states[t * state_size..(t + 1) * state_size];
            let err_t = &cache.error[t * d..(t + 1) * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let theta_pre_t = cache.theta_pre[t];
            let eta_t = cache.eta[t];
            let pre_acts_k = &cache.mlp_k_pre_acts[t];
            let activations_k = &cache.mlp_k_activations[t];

            // Feature maps
            let z_k_t = if has_fm { &cache.fm_z_k_mem[t * d..(t + 1) * d] } else { &[] as &[f32] };
            let z_q_t = if has_fm { &cache.fm_z_q_mem[t * d..(t + 1) * d] } else { &[] as &[f32] };
            let phi_q_t = if has_fm {
                feature_map::apply(q_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, d).0
            } else {
                q_t.to_vec()
            };

            let d_y_t = &d_y[t * d..(t + 1) * d];

            // ── Phase 1: Readout backward — y_t = MLP(M_{t+1}, φ(q_t)) ──
            let (_, q_pre_acts, q_activations) = mlp_forward(
                &cache.m_states, m_next_base, &phi_q_t, layout, activation);
            // d_M += weight grads from readout, d_phi_q = input grad
            let d_phi_q_t = mlp_backward_full(
                d_y_t, &q_pre_acts, &q_activations,
                &cache.m_states, m_next_base, layout, activation,
                &mut d_m, 0);
            let d_q_t = feature_map::vjp(&d_phi_q_t, z_q_t, &self.feature_map, &level_params.w_rand, d);
            d_q_mem[t * d..(t + 1) * d].copy_from_slice(&d_q_t);

            // ── Phase 2: Retention backward ──
            // d_S += d_M
            for i in 0..state_size { d_s[i] += d_m[i]; }

            // d_alpha = -frob(M_t, d_M)
            let d_alpha_scalar = -frobenius_dot_f32(
                &cache.m_states[m_t_base..m_t_base + state_size], &d_m);

            // d_M_prev = (1-α) * d_M
            let mut d_m_prev = d_m.clone();
            l2_apply_retention(&mut d_m_prev, 1.0 - alpha_t);

            // ── Phase 3: Recompute inner gradient ──
            let biased = apply_attentional_bias(err_t, self.bias, self.sign_sharpness);
            let mut grad_t = vec![0.0f32; state_size];
            mlp_inner_backward(&biased, pre_acts_k, activations_k,
                &cache.m_states, m_t_base, layout, activation, &mut grad_t, 0);

            // ── Phase 4: Momentum backward (EMA) ──
            let (d_eta_scalar, d_theta_scalar, d_grad) =
                crate::momentum::ema_step_backward(&mut d_s, s_t, &grad_t, eta_t, theta_t, d);

            // ── Phase 5: Second-order inner gradient VJP ──

            // Step 5a: Recompute inner backward d_out at each layer
            // inner_d_outs[l] = the d_out vector at layer l (after activation derivative,
            // before W^T propagation). Needed for cross-terms and inject values.
            let mut inner_d_outs: Vec<Vec<f32>> = Vec::with_capacity(n);
            {
                let mut d_out_ib = biased.clone();
                for l in (0..n).rev() {
                    let desc = &layout.layers[l];
                    if l < n - 1 {
                        apply_activation_derivative(&mut d_out_ib, &pre_acts_k[l], activation);
                    }
                    inner_d_outs.push(d_out_ib.clone());
                    if l > 0 {
                        let w = layout.w_slice(&cache.m_states, m_t_base, l);
                        let out_dim = desc.w_rows;
                        let in_dim = desc.w_cols;
                        let mut d_prev = vec![0.0f32; in_dim];
                        for j in 0..in_dim {
                            let mut sum = 0.0f32;
                            for ii in 0..out_dim {
                                sum += w[ii * in_dim + j] * d_out_ib[ii];
                            }
                            d_prev[j] = sum;
                        }
                        d_out_ib = d_prev;
                    }
                }
            }
            inner_d_outs.reverse(); // inner_d_outs[l] = d_out at layer l

            // Step 5b: Forward adjoint → d_biased + cross-terms for d_M
            //
            // The inner gradient is LINEAR in biased (the error signal). The adjoint
            // of this linear map gives d_biased from d_grad. Cross-terms arise from
            // the inner backward's use of W_l^T for propagation — the VJP of W^T
            // w.r.t. W produces outer(inner_d_out, adj) contributions to d_M.
            let mut cross_terms = vec![0.0f32; state_size];
            let d_biased: Vec<f32>;
            {
                let desc0 = &layout.layers[0];
                let d_gw0 = layout.w_slice(&d_grad, 0, 0);
                let d_gb0 = layout.b_slice(&d_grad, 0, 0);
                let h0 = &activations_k[0]; // phi_k

                // adj = d_grad_W[0] @ h[0] + d_grad_b[0]
                let mut adj = vec![0.0f32; desc0.w_rows];
                for i in 0..desc0.w_rows {
                    let mut sum = 0.0f32;
                    for j in 0..desc0.w_cols {
                        sum += d_gw0[i * desc0.w_cols + j] * h0[j];
                    }
                    adj[i] = sum + d_gb0[i];
                }

                for l in 0..(n - 1) {
                    // Apply activation derivative at layer l
                    apply_activation_derivative(&mut adj, &pre_acts_k[l], activation);

                    // Cross-term for W_{l+1}: outer(inner_d_outs[l+1], adj)
                    // This is the VJP of W_{l+1}^T @ d_out w.r.t. W_{l+1}
                    let desc_next = &layout.layers[l + 1];
                    let cw = layout.w_slice_mut(&mut cross_terms, 0, l + 1);
                    for i in 0..desc_next.w_rows {
                        for j in 0..desc_next.w_cols {
                            cw[i * desc_next.w_cols + j] += inner_d_outs[l + 1][i] * adj[j];
                        }
                    }

                    // Propagate forward: adj = W_{l+1} @ adj
                    let w_next = layout.w_slice(&cache.m_states, m_t_base, l + 1);
                    let mut new_adj = vec![0.0f32; desc_next.w_rows];
                    for i in 0..desc_next.w_rows {
                        let mut sum = 0.0f32;
                        for j in 0..desc_next.w_cols {
                            sum += w_next[i * desc_next.w_cols + j] * adj[j];
                        }
                        new_adj[i] = sum;
                    }
                    adj = new_adj;

                    // Add direct contribution from layer l+1
                    let d_gw = layout.w_slice(&d_grad, 0, l + 1);
                    let d_gb = layout.b_slice(&d_grad, 0, l + 1);
                    let h = &activations_k[l + 1];
                    for i in 0..desc_next.w_rows {
                        let mut sum = 0.0f32;
                        for j in 0..desc_next.w_cols {
                            sum += d_gw[i * desc_next.w_cols + j] * h[j];
                        }
                        adj[i] += sum + d_gb[i];
                    }
                }

                d_biased = adj; // [d]-dimensional
            }

            // Step 5c: Compute inject values for activation path
            // inject[l] = d_grad_W[l]^T @ inner_d_outs[l] — gradient contribution
            // from the inner gradient's dependence on activations[l] (which depend on phi_k).
            let mut injects: Vec<Vec<f32>> = Vec::with_capacity(n);
            for l in 0..n {
                let desc = &layout.layers[l];
                let d_gw = layout.w_slice(&d_grad, 0, l);
                let d_out_l = &inner_d_outs[l];
                let in_dim = desc.w_cols;
                let out_dim = desc.w_rows;
                let mut inject = vec![0.0f32; in_dim];
                for j in 0..in_dim {
                    let mut sum = 0.0f32;
                    for i in 0..out_dim {
                        sum += d_gw[i * in_dim + j] * d_out_l[i];
                    }
                    inject[j] = sum;
                }
                injects.push(inject);
            }

            // Step 5d: VJP through attentional bias
            let d_err = apply_attentional_bias_backward(&d_biased, err_t, self.bias, self.sign_sharpness);

            // ── Phase 6: Combined prediction + activation backward ──
            // d_v from error = prediction - v
            for i in 0..d { d_v_mem[t * d + i] -= d_err[i]; }

            // Standard MLP backward from d_err (prediction path) fused with
            // inject values (activation path). At each layer, the inject adds
            // the inner gradient's contribution through phi_k → activations.
            // Also accumulates d_M from prediction backward into d_m_prev.
            let d_phi_k_t: Vec<f32>;
            {
                let mut d_out_combined = d_err.clone();
                for l in (0..n).rev() {
                    let desc = &layout.layers[l];
                    let h_prev = &activations_k[l];

                    if l < n - 1 {
                        apply_activation_derivative(&mut d_out_combined, &pre_acts_k[l], activation);
                    }

                    // d_M_prev[W_l] += outer(d_out, h_prev)
                    let out_dim = desc.w_rows;
                    let in_dim = desc.w_cols;
                    let gw = layout.w_slice_mut(&mut d_m_prev, 0, l);
                    for i in 0..out_dim {
                        for j in 0..in_dim {
                            gw[i * in_dim + j] += d_out_combined[i] * h_prev[j];
                        }
                    }
                    // d_M_prev[b_l] += d_out
                    let gb = layout.b_slice_mut(&mut d_m_prev, 0, l);
                    for i in 0..desc.b_size {
                        gb[i] += d_out_combined[i];
                    }

                    // Propagate: d_out = W_l^T @ d_out
                    let w = layout.w_slice(&cache.m_states, m_t_base, l);
                    let mut d_prev = vec![0.0f32; in_dim];
                    for j in 0..in_dim {
                        let mut sum = 0.0f32;
                        for ii in 0..out_dim {
                            sum += w[ii * in_dim + j] * d_out_combined[ii];
                        }
                        d_prev[j] = sum;
                    }
                    d_out_combined = d_prev;

                    // Add inject from activation path
                    for j in 0..in_dim {
                        d_out_combined[j] += injects[l][j];
                    }
                }
                d_phi_k_t = d_out_combined;
            }

            // Add cross-terms from Phase 5b into d_M_prev
            for i in 0..state_size { d_m_prev[i] += cross_terms[i]; }

            // Feature map VJP for k
            let d_k_t = feature_map::vjp(&d_phi_k_t, z_k_t, &self.feature_map, &level_params.w_rand, d);
            for j in 0..d {
                d_k_mem[t * d + j] += d_k_t[j];
            }

            // ── Gate backward ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_scalar * sig_deriv;

            let theta_raw = softplus_f32(theta_pre_t);
            let clamp_mask = if theta_raw <= self.theta_floor || theta_raw >= self.theta_ceil { 0.0 } else { 1.0 };
            let softplus_deriv = sigmoid_f32(theta_pre_t);
            let d_theta_pre = d_theta_scalar * softplus_deriv * clamp_mask;

            let eta_sig_deriv = eta_t * (1.0 - eta_t);
            let d_eta_pre = d_eta_scalar * eta_sig_deriv;

            // Gate weight gradients
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;
            for i in 0..(2 * d) {
                grads.w_theta[i] += d_theta_pre * concat_t[i];
            }
            grads.b_theta[0] += d_theta_pre;
            for i in 0..(2 * d) {
                grads.w_eta[i] += d_eta_pre * concat_t[i];
            }
            grads.b_eta[0] += d_eta_pre;

            // Concat backward → d_k_mem, d_v_mem
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i]
                                    + d_theta_pre * level_params.w_theta[i]
                                    + d_eta_pre * level_params.w_eta[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i]
                                    + d_theta_pre * level_params.w_theta[d + i]
                                    + d_eta_pre * level_params.w_eta[d + i];
            }

            // Carry d_m and d_s to next (earlier) token
            d_m = d_m_prev;
            // d_s was already updated in-place by ema_step_backward
        }

        // ── Post-loop: L2 norm, conv1d, projection backward (shared with linear) ──
        if !cache.k_mem_norms.is_empty() {
            let mut d_k_raw = vec![0.0f32; s * d];
            crate::tensor::l2_normalize_rows_backward(
                &d_k_mem, &cache.k_mem, &cache.k_mem_norms,
                &mut d_k_raw, s, d);
            d_k_mem = d_k_raw;

            let mut d_q_raw = vec![0.0f32; s * d];
            crate::tensor::l2_normalize_rows_backward(
                &d_q_mem, &cache.q_mem, &cache.q_mem_norms,
                &mut d_q_raw, s, d);
            d_q_mem = d_q_raw;
        }

        crate::conv1d::backward_conv1d_kq(
            &mut d_k_mem, &mut d_q_mem,
            &cache.k_conv_cache, &cache.q_conv_cache,
            level_params, &mut grads, s, d);

        let mut d_embedded = vec![0.0f32; s * d];

        let mut d_k_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_k_mem, &mut d_k_mem_t, s, d);
        matmul_f32(&d_k_mem_t, embedded, grads.w_k_mem.master_mut(), d, s, d);

        let mut d_v_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_v_mem, &mut d_v_mem_t, s, d);
        matmul_f32(&d_v_mem_t, embedded, grads.w_v_mem.master_mut(), d, s, d);

        let mut d_q_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_q_mem, &mut d_q_mem_t, s, d);
        matmul_f32(&d_q_mem_t, embedded, grads.w_q_mem.master_mut(), d, s, d);

        let w_k_f32 = level_params.w_k_mem.as_f32();
        let w_v_f32 = level_params.w_v_mem.as_f32();
        let w_q_f32 = level_params.w_q_mem.as_f32();
        crate::tensor::matmul_acc_f32(&d_k_mem, &w_k_f32, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_v_mem, &w_v_f32, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_q_mem, &w_q_f32, &mut d_embedded, s, d, d);

        (grads, d_embedded)
    }
}

impl Default for TitansLMM {
    fn default() -> Self { Self::l2() }
}

/// All intermediate values from a Titans LMM forward pass, needed for backward.
pub struct TitansLMMCache {
    pub seq_len: usize,
    pub d: usize,
    /// Memory matrices M_t for t=0..seq_len: [(seq_len+1) * d * d]
    pub m_states: Vec<f32>,
    /// Momentum matrices S_t for t=0..seq_len: [(seq_len+1) * d * d]
    pub s_states: Vec<f32>,
    /// Per-token projected keys: [seq_len, d]
    pub k_mem: Vec<f32>,
    /// Per-token projected values: [seq_len, d]
    pub v_mem: Vec<f32>,
    /// Per-token projected queries: [seq_len, d]
    pub q_mem: Vec<f32>,
    /// Per-token concatenated (k,v): [seq_len, 2*d]
    pub concat_kv: Vec<f32>,
    /// Pre-sigmoid alpha values: [seq_len]
    pub alpha_pre: Vec<f32>,
    /// Sigmoid alpha values: [seq_len]
    pub alpha: Vec<f32>,
    /// Pre-softplus theta values: [seq_len]
    pub theta_pre: Vec<f32>,
    /// Softplus theta values: [seq_len]
    pub theta: Vec<f32>,
    /// Pre-sigmoid eta values: [seq_len]
    pub eta_pre: Vec<f32>,
    /// Sigmoid eta values: [seq_len]
    pub eta: Vec<f32>,
    /// Prediction errors: [seq_len, d]
    pub error: Vec<f32>,
    /// Gradient outer products: [seq_len, d*d]
    pub grad_outer: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
    /// Which momentum variant was used (needed for backward dispatch).
    pub momentum_kind: crate::model::MomentumKind,
    /// Per-token decay values for DeltaMomentum: [seq_len]. Empty for EMA/Deep.
    pub decay: Vec<f32>,
    /// Deep Momentum MLP cache. None for EMA/DeltaMomentum.
    pub deep_cache: Option<crate::momentum::DeepMomentumCache>,
    /// Deep Momentum MLP config dimension. 0 when not DeepMomentum.
    pub deep_d_hidden: usize,
    /// Conv1D cache for key preprocessing (None when kernel_size=0)
    pub k_conv_cache: Option<crate::conv1d::Conv1DCache>,
    /// Conv1D cache for query preprocessing (None when kernel_size=0)
    pub q_conv_cache: Option<crate::conv1d::Conv1DCache>,
    /// Pre-activation z for feature map on keys: [seq_len * d]. Empty for Identity.
    pub fm_z_k_mem: Vec<f32>,
    /// Pre-activation z for feature map on queries: [seq_len * d]. Empty for Identity.
    pub fm_z_q_mem: Vec<f32>,
    /// Pre-normalization L2 norms of k_mem rows: [seq_len].
    pub k_mem_norms: Vec<f32>,
    /// Pre-normalization L2 norms of q_mem rows: [seq_len].
    pub q_mem_norms: Vec<f32>,
    /// MLP layout (None for linear memory, Some for memory_layers >= 2).
    pub mlp_layout: Option<MLPMemoryLayout>,
    /// MLP activation (stored for backward). Only meaningful when mlp_layout.is_some().
    pub mlp_activation: MemoryActivation,
    /// Per-token MLP pre-activations for k-path (for backward through inner loop).
    /// mlp_k_pre_acts[t][l] = pre-activation vector at layer l for token t.
    /// Empty when linear memory.
    pub mlp_k_pre_acts: Vec<Vec<Vec<f32>>>,
    /// Per-token MLP activations for k-path (including input as [0]).
    /// mlp_k_activations[t][l] = activation vector at layer l for token t.
    /// Empty when linear memory.
    pub mlp_k_activations: Vec<Vec<Vec<f32>>>,
}

impl MemoryRule for TitansLMM {
    type Cache = TitansLMMCache;
    type State = MemoryState;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        crate::parallel::supported_strategies(crate::model::MemoryRuleKind::TitansLMM)
    }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError> {
        // Simplified write (no momentum tracking — use step() for full path)
        let d = state.d;
        let mut prediction = vec![0.0f32; d];
        matmul_f32(&state.m, k, &mut prediction, d, d, 1);

        // error = prediction - v
        let mut error = vec![0.0f32; d];
        for i in 0..d {
            error[i] = prediction[i] - v[i];
        }

        // Apply attentional bias (L2 = identity, L1/Lp = nonlinear)
        let biased = apply_attentional_bias(&error, self.bias, self.sign_sharpness);

        let retention = 1.0 - gates.alpha;
        let lr = gates.theta;
        for i in 0..d {
            for j in 0..d {
                state.m[i * d + j] = retention * state.m[i * d + j] - lr * biased[i] * k[j];
            }
        }
        Ok(())
    }

    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) -> Result<(), MemoryError> {
        let d = state.d;
        matmul_f32(&state.m, q, out, d, d, 1);
        Ok(())
    }

    /// Full sequence forward with momentum accumulator S.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, TitansLMMCache) {
        // MLP memory dispatch (spec 75, Titans §3.1)
        if self.memory_layers >= 2 {
            return self.step_mlp(level_params, embedded, seq_len, d, initial_m);
        }

        debug_assert_eq!(embedded.len(), seq_len * d);

        // Project embedded → k_mem, v_mem, q_mem via W^T
        let mut w_k_mem_t = vec![0.0f32; d * d];
        let mut w_v_mem_t = vec![0.0f32; d * d];
        let mut w_q_mem_t = vec![0.0f32; d * d];
        let w_k_f32 = level_params.w_k_mem.as_f32();
        let w_v_f32 = level_params.w_v_mem.as_f32();
        let w_q_f32 = level_params.w_q_mem.as_f32();
        transpose_f32(&w_k_f32, &mut w_k_mem_t, d, d);
        transpose_f32(&w_v_f32, &mut w_v_mem_t, d, d);
        transpose_f32(&w_q_f32, &mut w_q_mem_t, d, d);

        let mut k_mem = vec![0.0f32; seq_len * d];
        let mut v_mem = vec![0.0f32; seq_len * d];
        let mut q_mem = vec![0.0f32; seq_len * d];
        matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
        matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
        matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

        // Conv1D key/query preprocessing (after projection, before memory loop)
        let (k_conv_cache, q_conv_cache) = crate::conv1d::apply_conv1d_to_kq(
            &mut k_mem, &mut q_mem, level_params, seq_len, d);

        // L2-normalize keys and queries (Titans paper: "normalize queries and keys using l_2-norm")
        // This bounds ||k_t|| = 1, making memory updates d-invariant.
        let k_mem_norms = crate::tensor::l2_normalize_rows(&mut k_mem, seq_len, d);
        let q_mem_norms = crate::tensor::l2_normalize_rows(&mut q_mem, seq_len, d);

        // Resolve momentum kind
        let mk = self.momentum_kind;
        let dd = d * d;
        let deep_dh = if mk == crate::model::MomentumKind::DeepMomentum {
            if self.momentum_d_hidden > 0 { self.momentum_d_hidden } else { 4 * d }
        } else { 0 };

        // Allocate cache — seed M_0 from initial_m if provided, else zeros
        // S_0 is always zeros (no initial momentum)
        let mut m_states = vec![0.0f32; (seq_len + 1) * d * d];
        let mut s_states = vec![0.0f32; (seq_len + 1) * d * d];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), d * d);
            m_states[..d * d].copy_from_slice(&m0);
        }
        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut theta_pre = vec![0.0f32; seq_len];
        let mut theta = vec![0.0f32; seq_len];
        let mut eta_pre = vec![0.0f32; seq_len];
        let mut eta = vec![0.0f32; seq_len];
        let mut error = vec![0.0f32; seq_len * d];
        let mut grad_outer = vec![0.0f32; seq_len * d * d];
        let mut y = vec![0.0f32; seq_len * d];
        let mut decay_buf = if mk == crate::model::MomentumKind::DeltaMomentum {
            vec![0.0f32; seq_len]
        } else { vec![] };
        let mut deep_cache = if mk == crate::model::MomentumKind::DeepMomentum {
            Some(crate::momentum::DeepMomentumCache::new(seq_len, dd, deep_dh))
        } else { None };

        let has_fm = !matches!(self.feature_map, FeatureMapKind::Identity);
        let mut fm_z_k_mem = if has_fm { vec![0.0f32; seq_len * d] } else { vec![] };
        let mut fm_z_q_mem = if has_fm { vec![0.0f32; seq_len * d] } else { vec![] };

        // Pre-allocated phi/z buffers — reused each token to avoid per-iter Vec alloc.
        let mut phi_k_buf = vec![0.0f32; d];
        let mut phi_q_buf = vec![0.0f32; d];
        let mut z_k_buf = if has_fm { vec![0.0f32; d] } else { vec![] };
        let mut z_q_buf = if has_fm { vec![0.0f32; d] } else { vec![] };

        // Sequential token loop
        for t in 0..seq_len {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];

            // Feature map: phi_k_t = φ(k_t), phi_q_t = φ(q_t). Gates use raw k_t.
            // For Identity: phi slices alias k_t/q_t (zero allocation).
            // For non-Identity: phi_k_buf/phi_q_buf are pre-allocated and reused.
            let (phi_k_t, phi_q_t): (&[f32], &[f32]) = if has_fm {
                feature_map::apply_into(k_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, &mut phi_k_buf, &mut z_k_buf, d);
                feature_map::apply_into(q_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, &mut phi_q_buf, &mut z_q_buf, d);
                fm_z_k_mem[t * d..(t + 1) * d].copy_from_slice(&z_k_buf);
                fm_z_q_mem[t * d..(t + 1) * d].copy_from_slice(&z_q_buf);
                (&phi_k_buf, &phi_q_buf)
            } else {
                (k_t, q_t)
            };

            // Concatenate (k_t, v_t) — raw k_t for gate computation
            let c_base = t * 2 * d;
            concat_kv[c_base..c_base + d].copy_from_slice(k_t);
            concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
            let concat_t = &concat_kv[c_base..c_base + 2 * d];

            // alpha_t = sigmoid(concat @ w_alpha + b_alpha)
            let mut alpha_pre_t = level_params.b_alpha[0];
            for i in 0..(2 * d) {
                alpha_pre_t += concat_t[i] * level_params.w_alpha[i];
            }
            alpha_pre[t] = alpha_pre_t;
            alpha[t] = sigmoid_f32(alpha_pre_t).clamp(self.alpha_floor, self.alpha_ceil);

            // theta_t = softplus(concat @ w_theta + b_theta)
            let mut theta_pre_t = level_params.b_theta[0];
            for i in 0..(2 * d) {
                theta_pre_t += concat_t[i] * level_params.w_theta[i];
            }
            theta_pre[t] = theta_pre_t;
            theta[t] = softplus_f32(theta_pre_t).clamp(self.theta_floor, self.theta_ceil);

            // eta_t = sigmoid(concat @ w_eta + b_eta) — NEW: momentum gate
            let mut eta_pre_t = level_params.b_eta[0];
            for i in 0..(2 * d) {
                eta_pre_t += concat_t[i] * level_params.w_eta[i];
            }
            eta_pre[t] = eta_pre_t;
            eta[t] = sigmoid_f32(eta_pre_t);

            // prediction = M_t @ φ(k_t)
            let m_t = &m_states[t * d * d..(t + 1) * d * d];
            let mut prediction = vec![0.0f32; d];
            matmul_f32(m_t, &phi_k_t, &mut prediction, d, d, 1);

            // error = prediction - v_t (raw error, stored for backward VJP)
            let e_base = t * d;
            for i in 0..d {
                error[e_base + i] = prediction[i] - v_t[i];
            }

            // Apply attentional bias: L2 → identity, L1 → tanh(a*e), Lp → general
            let biased = apply_attentional_bias(&error[e_base..e_base + d], self.bias, self.sign_sharpness);

            // grad = outer(biased_error, φ(k_t))
            let g_base = t * d * d;
            outer_product_f32(&biased, &phi_k_t, &mut grad_outer[g_base..g_base + d * d]);

            // Momentum update — dispatched to momentum.rs
            let eta_t = eta[t];
            let theta_t = theta[t];
            let s_next_off = (t + 1) * dd;
            match mk {
                crate::model::MomentumKind::EMA | crate::model::MomentumKind::None => {
                    crate::momentum::ema_step(&mut s_states, t, d, eta_t, theta_t,
                        &grad_outer[g_base..g_base + dd]);
                }
                crate::model::MomentumKind::DeltaMomentum => {
                    crate::momentum::delta_momentum_step(&mut s_states, t, d, eta_t, theta_t,
                        &grad_outer[g_base..g_base + dd], &mut decay_buf);
                }
                crate::model::MomentumKind::DeepMomentum => {
                    // For deep momentum, the MLP output replaces S_{t+1}
                    let dc = deep_cache.as_mut().unwrap();
                    let mlp = crate::momentum::DeepMomentumMLP { d, d_hidden: deep_dh };
                    let mom_out = crate::momentum::deep_momentum_step(
                        &mlp, dc, t, eta_t, theta_t, &grad_outer[g_base..g_base + dd]);
                    s_states[s_next_off..s_next_off + dd].copy_from_slice(&mom_out);
                }
            }

            // M_{t+1} = (1-alpha_t) * M_t + S_{t+1}  (memory update with momentum)
            let m_next_off = (t + 1) * d * d;
            let m_t_off = t * d * d;
            m_states.copy_within(m_t_off..m_t_off + d * d, m_next_off);
            l2_apply_retention(&mut m_states[m_next_off..m_next_off + d * d], 1.0 - alpha[t]);
            for i in 0..(d * d) {
                m_states[m_next_off + i] += s_states[s_next_off + i];
            }

            // M-norm clamp (straight-through) — prevents memory divergence
            if self.m_norm_max < f32::MAX {
                let slice = &mut m_states[m_next_off..m_next_off + d * d];
                let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
                let norm = norm_sq.sqrt();
                if norm > self.m_norm_max {
                    let scale = self.m_norm_max / norm;
                    for x in slice.iter_mut() { *x *= scale; }
                }
            }

            // y_t = M_{t+1} @ φ(q_t)
            let m_next = &m_states[m_next_off..m_next_off + d * d];
            matmul_f32(m_next, &phi_q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = TitansLMMCache {
            seq_len, d, m_states, s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, theta_pre, theta, eta_pre, eta,
            error, grad_outer, y: y.clone(), fm_z_k_mem, fm_z_q_mem,
            momentum_kind: mk,
            decay: decay_buf,
            deep_cache,
            deep_d_hidden: deep_dh,
            k_conv_cache, q_conv_cache,
            k_mem_norms, q_mem_norms,
            mlp_layout: None,
            mlp_activation: MemoryActivation::GELU,
            mlp_k_pre_acts: Vec::new(),
            mlp_k_activations: Vec::new(),
        };

        (y, cache)
    }

    /// Full sequence backward through the Titans LMM memory.
    ///
    /// Two interacting recurrences: d_M and d_S propagate backward.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &TitansLMMCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        // Dispatch to MLP backward when memory_layers >= 2
        if let Some(ref layout) = cache.mlp_layout {
            return self.step_backward_mlp(level_params, cache, d_y, embedded, layout);
        }

        let mut grads = MemoryLevelParams::zeros_like_from(level_params, d);

        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // d_M and d_S: accumulated gradients on memory and momentum state
        let mut d_m = vec![0.0f32; d * d];
        let mut d_s = vec![0.0f32; d * d];

        let has_fm = !cache.fm_z_k_mem.is_empty();

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let m_t = &cache.m_states[t * d * d..(t + 1) * d * d];
            let m_next = &cache.m_states[(t + 1) * d * d..(t + 2) * d * d];
            let s_t = &cache.s_states[t * d * d..(t + 1) * d * d];
            let err_t = &cache.error[t * d..(t + 1) * d];
            let grad_t = &cache.grad_outer[t * d * d..(t + 1) * d * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];
            let theta_t = cache.theta[t];
            let theta_pre_t = cache.theta_pre[t];
            let eta_t = cache.eta[t];

            // Reconstruct phi_k_t and phi_q_t (for backward through memory ops)
            let z_k_t = if has_fm { &cache.fm_z_k_mem[t * d..(t + 1) * d] } else { &[] as &[f32] };
            let z_q_t = if has_fm { &cache.fm_z_q_mem[t * d..(t + 1) * d] } else { &[] as &[f32] };
            let phi_k_t = if has_fm {
                feature_map::apply(k_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, d).0
            } else {
                k_t.to_vec()
            };
            let phi_q_t = if has_fm {
                feature_map::apply(q_t, &self.feature_map, &level_params.w_rand, &level_params.b_rand, d).0
            } else {
                q_t.to_vec()
            };

            // ── y_t = M_{t+1} @ φ(q_t) backward ──
            let d_y_t = &d_y[t * d..(t + 1) * d];

            // d_M += outer(d_y_t, φ(q_t))
            for i in 0..d {
                for j in 0..d {
                    d_m[i * d + j] += d_y_t[i] * phi_q_t[j];
                }
            }

            // d_phi_q_t = M_{t+1}^T @ d_y_t, then VJP → d_q_mem
            let mut d_phi_q_t = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_next[j * d + i] * d_y_t[j];
                }
                d_phi_q_t[i] = sum;
            }
            let d_q_t = feature_map::vjp(&d_phi_q_t, z_q_t, &self.feature_map, &level_params.w_rand, d);
            d_q_mem[t * d..(t + 1) * d].copy_from_slice(&d_q_t);

            // ── M_{t+1} = (1-alpha) * M_t + S_{t+1} backward ──
            // d_S_{t+1} = d_M  (S contributes additively to M)
            // Note: d_s already accumulates from future — add current d_m contribution
            for i in 0..(d * d) {
                d_s[i] += d_m[i];
            }

            let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);

            let mut d_m_prev = d_m.clone();
            l2_apply_retention(&mut d_m_prev, 1.0 - alpha_t);

            // ── Momentum backward — dispatched to momentum.rs ──
            let (d_eta_scalar, d_theta_scalar, d_grad) = match cache.momentum_kind {
                crate::model::MomentumKind::EMA | crate::model::MomentumKind::None => {
                    crate::momentum::ema_step_backward(&mut d_s, s_t, grad_t, eta_t, theta_t, d)
                }
                crate::model::MomentumKind::DeltaMomentum => {
                    let decay_t = cache.decay[t];
                    crate::momentum::delta_momentum_step_backward(
                        &mut d_s, s_t, grad_t, eta_t, theta_t, decay_t, d)
                }
                crate::model::MomentumKind::DeepMomentum => {
                    let dc = cache.deep_cache.as_ref().unwrap();
                    let mlp = crate::momentum::DeepMomentumMLP {
                        d, d_hidden: cache.deep_d_hidden,
                    };
                    // For deep momentum, d_s is the d_output for the MLP
                    let result = crate::momentum::deep_momentum_step_backward(
                        &mlp, dc, t, eta_t, theta_t, grad_t, &d_s);
                    // d_s_prev = 0 for deep (no linear S recurrence)
                    for i in 0..(d * d) { d_s[i] = 0.0; }
                    result
                }
            };
            // After dispatch, d_s has been updated to d_s_prev in-place (for EMA/Delta).
            // For Deep, d_s was zeroed (no linear S recurrence).
            // Clone for the d_m/d_s swap at end of loop iteration.
            let d_s_prev = d_s.clone();

            // ── grad = outer(biased_error, φ(k)) backward ──
            // Recompute biased error for d_phi_k (not stored in cache)
            let biased = apply_attentional_bias(err_t, self.bias, self.sign_sharpness);

            // d_biased[i] = sum_j d_grad[i,j] * φ(k)[j]
            let mut d_biased = vec![0.0f32; d];
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_grad[i * d + j] * phi_k_t[j];
                }
                d_biased[i] = sum;
            }
            // d_phi_k[j] += sum_i d_grad[i,j] * biased[i]  (accumulate, add prediction contrib below)
            let mut d_phi_k_t = vec![0.0f32; d];
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_grad[i * d + j] * biased[i];
                }
                d_phi_k_t[j] = sum;
            }

            // ── VJP through attentional bias: biased = f(error) ──
            let d_err = apply_attentional_bias_backward(&d_biased, err_t, self.bias, self.sign_sharpness);

            // ── error = prediction - v backward ──
            for i in 0..d {
                d_v_mem[t * d + i] -= d_err[i];
            }

            // ── prediction = M_t @ φ(k_t) backward ──
            for i in 0..d {
                for j in 0..d {
                    d_m_prev[i * d + j] += d_err[i] * phi_k_t[j];
                }
            }
            // d_phi_k[j] += sum_i M_t[i,j] * d_err[i]
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += m_t[i * d + j] * d_err[i];
                }
                d_phi_k_t[j] += sum;
            }
            // VJP through feature map: d_phi_k_t → d_k_mem
            let d_k_t = feature_map::vjp(&d_phi_k_t, z_k_t, &self.feature_map, &level_params.w_rand, d);
            for j in 0..d {
                d_k_mem[t * d + j] += d_k_t[j];
            }

            // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_scalar * sig_deriv;

            // ── Gate backward: theta_t = softplus(theta_pre_t) ──
            // Clamp gradient mask: zero gradient when theta was at floor or ceil (CS-39)
            let theta_raw = softplus_f32(theta_pre_t);
            let clamp_mask = if theta_raw <= self.theta_floor || theta_raw >= self.theta_ceil { 0.0 } else { 1.0 };
            let softplus_deriv = sigmoid_f32(theta_pre_t);
            let d_theta_pre = d_theta_scalar * softplus_deriv * clamp_mask;

            // ── Gate backward: eta_t = sigmoid(eta_pre_t) ──
            let eta_sig_deriv = eta_t * (1.0 - eta_t);
            let d_eta_pre = d_eta_scalar * eta_sig_deriv;

            // ── w_alpha, b_alpha gradient ──
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;

            // ── w_theta, b_theta gradient ──
            for i in 0..(2 * d) {
                grads.w_theta[i] += d_theta_pre * concat_t[i];
            }
            grads.b_theta[0] += d_theta_pre;

            // ── w_eta, b_eta gradient ──
            for i in 0..(2 * d) {
                grads.w_eta[i] += d_eta_pre * concat_t[i];
            }
            grads.b_eta[0] += d_eta_pre;

            // ── concat backward → d_k_mem, d_v_mem ──
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i]
                                    + d_theta_pre * level_params.w_theta[i]
                                    + d_eta_pre * level_params.w_eta[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i]
                                    + d_theta_pre * level_params.w_theta[d + i]
                                    + d_eta_pre * level_params.w_eta[d + i];
            }

            // Update d_m and d_s for next (earlier) token
            d_m = d_m_prev;
            d_s = d_s_prev;
        }

        // ── L2 normalization backward (before conv1d backward) ──
        // d_k_mem and d_q_mem are w.r.t. normalized k/q. Chain the normalization Jacobian.
        // Guard: old tape recordings may have empty norms (backward compat).
        if !cache.k_mem_norms.is_empty() {
            let mut d_k_raw = vec![0.0f32; s * d];
            crate::tensor::l2_normalize_rows_backward(
                &d_k_mem, &cache.k_mem, &cache.k_mem_norms,
                &mut d_k_raw, s, d);
            d_k_mem = d_k_raw;

            let mut d_q_raw = vec![0.0f32; s * d];
            crate::tensor::l2_normalize_rows_backward(
                &d_q_mem, &cache.q_mem, &cache.q_mem_norms,
                &mut d_q_raw, s, d);
            d_q_mem = d_q_raw;
        }

        // ── Conv1D backward (before projection backward) ──
        crate::conv1d::backward_conv1d_kq(
            &mut d_k_mem, &mut d_q_mem,
            &cache.k_conv_cache, &cache.q_conv_cache,
            level_params, &mut grads, s, d);

        // ── Projection backward: k_mem = embedded @ W_K_mem^T ──
        let mut d_embedded = vec![0.0f32; s * d];

        let mut d_k_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_k_mem, &mut d_k_mem_t, s, d);
        matmul_f32(&d_k_mem_t, embedded, grads.w_k_mem.master_mut(), d, s, d);

        let mut d_v_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_v_mem, &mut d_v_mem_t, s, d);
        matmul_f32(&d_v_mem_t, embedded, grads.w_v_mem.master_mut(), d, s, d);

        let mut d_q_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_q_mem, &mut d_q_mem_t, s, d);
        matmul_f32(&d_q_mem_t, embedded, grads.w_q_mem.master_mut(), d, s, d);

        let w_k_f32 = level_params.w_k_mem.as_f32();
        let w_v_f32 = level_params.w_v_mem.as_f32();
        let w_q_f32 = level_params.w_q_mem.as_f32();
        crate::tensor::matmul_acc_f32(&d_k_mem, &w_k_f32, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_v_mem, &w_v_f32, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_q_mem, &w_q_f32, &mut d_embedded, s, d, d);

        (grads, d_embedded)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

    fn test_config() -> MAGConfig {
        MAGConfig::titans_test_config()
    }

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    #[test]
    fn test_titans_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_titans_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let m_0 = &cache.m_states[0..d * d];
        let m_t = &cache.m_states[s * d * d..(s + 1) * d * d];
        let m0_norm: f32 = m_0.iter().map(|x| x * x).sum();
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_titans_forward_momentum_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // S_0 should be zero
        let s_0 = &cache.s_states[0..d * d];
        let s0_norm: f32 = s_0.iter().map(|x| x * x).sum();
        assert!(s0_norm < 1e-12, "S_0 should be zero");

        // S_T should be non-zero (momentum accumulated)
        let s_t = &cache.s_states[s * d * d..(s + 1) * d * d];
        let st_norm: f32 = s_t.iter().map(|x| x * x).sum();
        assert!(st_norm > 1e-12, "S_T should have evolved, norm={st_norm}");
    }

    #[test]
    fn test_titans_forward_eta_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            let th = cache.theta[t];
            assert!(th >= 0.0, "theta[{t}]={th} should be non-negative");
            let e = cache.eta[t];
            assert!(e > 0.0 && e < 1.0, "eta[{t}]={e} not in (0,1)");
        }
    }

    #[test]
    fn test_titans_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Titans LMM forward should be deterministic");
    }

    #[test]
    fn test_titans_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM::l2();
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
        assert_eq!(cache.s_states.len(), (s + 1) * d * d);
        assert_eq!(cache.eta.len(), s);
    }

    #[test]
    fn test_titans_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", grads.w_k_mem.master()), ("w_v_mem", grads.w_v_mem.master()),
            ("w_q_mem", grads.w_q_mem.master()), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta), ("w_eta", &grads.w_eta),
            ("b_eta", &grads.b_eta),
        ] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "d_embedded[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_titans_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", grads.w_k_mem.master()), ("w_v_mem", grads.w_v_mem.master()),
            ("w_q_mem", grads.w_q_mem.master()),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "grad_{name} is all zeros (max_abs={max_abs})");
        }
        // eta grads should also be non-zero
        let eta_max = grads.w_eta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(eta_max > 1e-10, "w_eta grads should be non-zero");

        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "d_embedded is all zeros");
    }

    #[test]
    fn test_titans_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = TitansLMM::l2();
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        assert_eq!(grads.w_k_mem.master().len(), d * d);
        assert_eq!(grads.w_v_mem.master().len(), d * d);
        assert_eq!(grads.w_q_mem.master().len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
        assert_eq!(grads.w_theta.len(), 2 * d);
        assert_eq!(grads.b_theta.len(), 1);
        assert_eq!(grads.w_eta.len(), 2 * d);
        assert_eq!(grads.b_eta.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_titans_read_only_zero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let frozen_m = vec![0.0f32; d * d];
        let (y, _q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d, &crate::feature_map::FeatureMapKind::Identity);
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_titans_read_only_nonzero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let mut frozen_m = vec![0.0f32; d * d];
        for i in 0..d { frozen_m[i * d + i] = 1.0; }
        let (y, q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d, &crate::feature_map::FeatureMapKind::Identity);
        for i in 0..(s * d) {
            assert!((y[i] - q_mem[i]).abs() < 1e-6, "y[{i}]={} != q_mem[{i}]={}", y[i], q_mem[i]);
        }
    }

    #[test]
    fn test_titans_level_and_parallelization() {
        let rule = TitansLMM::l2();
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"chunkwise_gd"));
        assert!(strategies.contains(&"associative_scan_partial"));
        assert!(strategies.contains(&"tnt"));
    }

    // ── Attentional bias dispatch tests ──────────────────────────────

    #[test]
    fn test_titans_l1_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM { bias: crate::model::AttentionalBias::L1, sign_sharpness: 10.0, momentum_kind: crate::model::MomentumKind::EMA, momentum_d_hidden: 0, alpha_floor: 0.0, alpha_ceil: 1.0, theta_floor: 0.0, theta_ceil: f32::MAX, m_norm_max: f32::MAX, feature_map: FeatureMapKind::Identity, memory_layers: 1, memory_expansion_factor: 4, memory_activation: MemoryActivation::GELU };
        let (y, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "L1 y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_titans_lp3_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM { bias: crate::model::AttentionalBias::Lp(3.0), sign_sharpness: 10.0, momentum_kind: crate::model::MomentumKind::EMA, momentum_d_hidden: 0, alpha_floor: 0.0, alpha_ceil: 1.0, theta_floor: 0.0, theta_ceil: f32::MAX, m_norm_max: f32::MAX, feature_map: FeatureMapKind::Identity, memory_layers: 1, memory_expansion_factor: 4, memory_activation: MemoryActivation::GELU };
        let (y, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "Lp(3) y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_titans_l1_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = TitansLMM { bias: crate::model::AttentionalBias::L1, sign_sharpness: 10.0, momentum_kind: crate::model::MomentumKind::EMA, momentum_d_hidden: 0, alpha_floor: 0.0, alpha_ceil: 1.0, theta_floor: 0.0, theta_ceil: f32::MAX, m_norm_max: f32::MAX, feature_map: FeatureMapKind::Identity, memory_layers: 1, memory_expansion_factor: 4, memory_activation: MemoryActivation::GELU };
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d_y = vec![1.0f32; y.len()];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "L1 d_emb[{i}] not finite: {v}");
        }
        for (i, &v) in grads.w_k_mem.master().iter().enumerate() {
            assert!(v.is_finite(), "L1 d_w_k_mem[{i}] not finite: {v}");
        }
    }

    // ── Momentum variant integration tests ─────────────────────────

    #[test]
    fn test_titans_ema_unchanged() {
        // Refactored TitansLMM with EMA produces bit-identical results across runs.
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);

        let rule = TitansLMM::l2(); // EMA by default
        let (y1, cache1) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, cache2) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);

        for i in 0..y1.len() {
            assert!((y1[i] - y2[i]).abs() == 0.0,
                "EMA should be deterministic, y[{}]: {} vs {}", i, y1[i], y2[i]);
        }
        for i in 0..cache1.s_states.len() {
            assert!((cache1.s_states[i] - cache2.s_states[i]).abs() == 0.0,
                "S_states should be identical at {}", i);
        }
    }

    #[test]
    fn test_titans_delta_momentum() {
        // TitansLMM with DeltaMomentum produces different output than EMA.
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let ema_rule = TitansLMM::l2();
        let (y_ema, _) = ema_rule.step(&params.levels[0], &embedded, s, d, None);

        let delta_rule = TitansLMM {
            bias: crate::model::AttentionalBias::L2,
            sign_sharpness: 10.0,
            momentum_kind: crate::model::MomentumKind::DeltaMomentum,
            momentum_d_hidden: 0,
            alpha_floor: 0.0,
            alpha_ceil: 1.0,
            theta_floor: 0.0,
            theta_ceil: f32::MAX,
            m_norm_max: f32::MAX,
            feature_map: FeatureMapKind::Identity,
            memory_layers: 1, memory_expansion_factor: 4, memory_activation: MemoryActivation::GELU,
        };
        let (y_delta, cache_delta) = delta_rule.step(&params.levels[0], &embedded, s, d, None);

        for (i, &v) in y_delta.iter().enumerate() {
            assert!(v.is_finite(), "Delta y[{i}] not finite: {v}");
        }

        let diff: f32 = y_ema.iter().zip(y_delta.iter()).map(|(a, b)| (a - b).abs()).sum();
        // With tiny test dims (d=8, seq_len=4), ||g||^2 is small so decay ≈ eta.
        // The diff is non-zero but tiny — just verify it's not bit-identical.
        assert!(diff > 0.0, "DeltaMomentum should differ from EMA, diff={diff}");

        assert_eq!(cache_delta.decay.len(), s);
        for &dc in &cache_delta.decay {
            assert!(dc >= 1e-6 && dc <= 1.0 - 1e-6, "Decay should be clamped, got {dc}");
        }
    }

    #[test]
    fn test_titans_deep_momentum() {
        // TitansLMM with DeepMomentum produces valid, distinct output.
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let deep_rule = TitansLMM {
            bias: crate::model::AttentionalBias::L2,
            sign_sharpness: 10.0,
            momentum_kind: crate::model::MomentumKind::DeepMomentum,
            momentum_d_hidden: 4 * d,
            alpha_floor: 0.0,
            alpha_ceil: 1.0,
            theta_floor: 0.0,
            theta_ceil: f32::MAX,
            m_norm_max: f32::MAX,
            feature_map: FeatureMapKind::Identity,
            memory_layers: 1,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
        };
        let (y_deep, cache_deep) = deep_rule.step(&params.levels[0], &embedded, s, d, None);

        for (i, &v) in y_deep.iter().enumerate() {
            assert!(v.is_finite(), "Deep y[{i}] not finite: {v}");
        }

        let m_final = &cache_deep.m_states[s * d * d..(s + 1) * d * d];
        let m_norm: f32 = m_final.iter().map(|x| x * x).sum();
        // With tiny test dims and W2_0=0, the MLP output is very small at first.
        // W2 only gets one inner-loop update before re-evaluation, so the output
        // is O(theta * ||grad||^2 * scale) — tiny but non-zero.
        assert!(m_norm > 0.0, "Deep M_T should have evolved, norm={m_norm}");

        assert!(cache_deep.deep_cache.is_some());
        let dc = cache_deep.deep_cache.as_ref().unwrap();
        assert!(!dc.output.is_empty());

        let ema_rule = TitansLMM::l2();
        let (y_ema, _) = ema_rule.step(&params.levels[0], &embedded, s, d, None);
        let diff: f32 = y_ema.iter().zip(y_deep.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "DeepMomentum should differ from EMA, diff={diff}");
    }

    // ══════════════════════════════════════════════════════════════════
    // MLP Memory tests (spec 75)
    // ══════════════════════════════════════════════════════════════════

    fn mlp_config(layers: usize, expansion: usize) -> MAGConfig {
        let mut cfg = test_config();
        cfg.memory_layers = layers;
        cfg.memory_expansion_factor = expansion;
        cfg
    }

    #[test]
    fn test_mlp_layout_2layer() {
        let layout = MLPMemoryLayout::new(2, 8, 4);
        assert_eq!(layout.n_layers, 2);
        assert_eq!(layout.d, 8);
        assert_eq!(layout.d_h, 32);
        // W1[32,8] + b1[32] + W2[8,32] + b2[8] = 256 + 32 + 256 + 8 = 552
        assert_eq!(layout.total_params, 552);

        assert_eq!(layout.layers[0].w_rows, 32);
        assert_eq!(layout.layers[0].w_cols, 8);
        assert_eq!(layout.layers[0].b_size, 32);

        assert_eq!(layout.layers[1].w_rows, 8);
        assert_eq!(layout.layers[1].w_cols, 32);
        assert_eq!(layout.layers[1].b_size, 8);
    }

    #[test]
    fn test_mlp_layout_3layer() {
        let layout = MLPMemoryLayout::new(3, 4, 2);
        // W1[8,4] + b1[8] + W2[8,8] + b2[8] + W3[4,8] + b3[4] = 32+8+64+8+32+4 = 148
        assert_eq!(layout.d_h, 8);
        assert_eq!(layout.total_params, 148);
        assert_eq!(layout.layers[1].w_rows, 8); // hidden layer
        assert_eq!(layout.layers[1].w_cols, 8);
    }

    #[test]
    fn test_mlp_forward_zero_weights() {
        // MLP with all-zero weights → output should be all-zero (bias=0 too)
        let layout = MLPMemoryLayout::new(2, 4, 2);
        let state = vec![0.0f32; layout.total_params];
        let input = vec![1.0f32; 4];
        let (output, pre_acts, activations) = mlp_forward(&state, 0, &input, &layout, MemoryActivation::GELU);
        assert_eq!(output.len(), 4);
        for &v in &output {
            assert_eq!(v, 0.0, "Zero-weight MLP should produce zero output");
        }
        assert_eq!(pre_acts.len(), 2);
        assert_eq!(activations.len(), 3); // input + 2 layers
    }

    #[test]
    fn test_mlp_forward_identity_like() {
        // Set W1 = I (via expansion), W2 so that M acts like identity on first d elements
        let d = 4;
        let layout = MLPMemoryLayout::new(2, d, 1); // expansion=1 → d_h=d
        let mut state = vec![0.0f32; layout.total_params];

        // W1 = I_d (d×d identity)
        for i in 0..d {
            state[layout.layers[0].w_offset + i * d + i] = 1.0;
        }
        // W2 = I_d
        for i in 0..d {
            state[layout.layers[1].w_offset + i * d + i] = 1.0;
        }

        let input = vec![0.5f32; d];
        // With ReLU: ReLU(I@x + 0) = ReLU(x) = x (since x > 0), then I@x + 0 = x
        let (output, _, _) = mlp_forward(&state, 0, &input, &layout, MemoryActivation::ReLU);
        for i in 0..d {
            let diff = (output[i] - input[i]).abs();
            assert!(diff < 1e-6, "Identity-like MLP output[{i}]={} expected {}", output[i], input[i]);
        }
    }

    #[test]
    fn test_mlp_backward_gradient_finite() {
        let d = 4;
        let layout = MLPMemoryLayout::new(2, d, 2);
        let mut state = vec![0.0f32; layout.total_params];
        // Give weights small non-zero values
        let mut rng = SimpleRng::new(42);
        rng.fill_uniform(&mut state, 0.1);

        let input = vec![0.5f32; d];
        let (output, pre_acts, activations) = mlp_forward(&state, 0, &input, &layout, MemoryActivation::GELU);

        // d_out = 2 * output (L2 loss gradient)
        let d_out: Vec<f32> = output.iter().map(|x| 2.0 * x).collect();
        let mut grad = vec![0.0f32; layout.total_params];
        mlp_inner_backward(&d_out, &pre_acts, &activations, &state, 0, &layout, MemoryActivation::GELU, &mut grad, 0);

        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "MLP backward grad[{i}] not finite: {g}");
        }
        let grad_norm: f32 = grad.iter().map(|x| x * x).sum();
        assert!(grad_norm > 0.0, "MLP backward should produce non-zero gradients");
    }

    #[test]
    fn test_memory_layers_1_uses_linear_path() {
        // memory_layers=1 → takes the original linear d×d path, NOT step_mlp
        let cfg = mlp_config(1, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        // linear-path rule (memory_layers=1)
        let rule_linear = TitansLMM {
            memory_layers: 1,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y1, cache1) = rule_linear.step(&params.levels[0], &embedded, s, d, None);

        // baseline l2() constructor also has memory_layers=1
        let rule_base = TitansLMM::l2();
        let (y2, cache2) = rule_base.step(&params.levels[0], &embedded, s, d, None);

        // Must be bit-identical: same code path, same inputs
        assert_eq!(y1.len(), y2.len());
        for i in 0..y1.len() {
            assert_eq!(y1[i], y2[i],
                "memory_layers=1 output should be bit-identical to l2(), y[{i}] {} vs {}", y1[i], y2[i]);
        }
        assert_eq!(cache1.m_states.len(), cache2.m_states.len());
        // m_states should be d*d per step (linear), NOT MLP expanded
        assert_eq!(cache1.m_states.len(), (s + 1) * d * d);
        // MLP layout should be None for linear path
        assert!(cache1.mlp_layout.is_none());
    }

    #[test]
    fn test_mlp_memory_2layer_forward_finite() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "MLP y[{i}] not finite: {v}");
        }
        assert_eq!(y.len(), s * d);

        // Cache should have MLP layout
        assert!(cache.mlp_layout.is_some());
        let layout = cache.mlp_layout.as_ref().unwrap();
        assert_eq!(layout.n_layers, 2);
        // m_states should use MLP-sized buffers
        assert_eq!(cache.m_states.len(), (s + 1) * layout.total_params);
    }

    #[test]
    fn test_mlp_memory_evolves() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (_, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let layout = cache.mlp_layout.as_ref().unwrap();
        let sp = layout.total_params;

        let m0_norm: f32 = cache.m_states[..sp].iter().map(|x| x * x).sum();
        let mt_norm: f32 = cache.m_states[s * sp..(s + 1) * sp].iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero, norm={m0_norm}");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_mlp_output_differs_from_linear() {
        // MLP memory should produce different output than linear memory
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let linear_rule = TitansLMM::l2();
        let (y_linear, _) = linear_rule.step(&params.levels[0], &embedded, s, d, None);

        let mlp_rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y_mlp, _) = mlp_rule.step(&params.levels[0], &embedded, s, d, None);

        // Both produce finite output, but they should differ
        let diff: f32 = y_linear.iter().zip(y_mlp.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6,
            "MLP and linear memory should produce different output, diff={diff}");
    }

    #[test]
    fn test_mlp_m_norm_clamp() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 2,
            memory_activation: MemoryActivation::ReLU,
            m_norm_max: 1.0, // very tight clamp
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let layout = cache.mlp_layout.as_ref().unwrap();
        let sp = layout.total_params;

        // Check that every M_t respects the norm bound
        for t in 1..=s {
            let m_t = &cache.m_states[t * sp..(t + 1) * sp];
            let norm: f32 = m_t.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm <= 1.0 + 1e-5,
                "M[{t}] norm {norm} exceeds m_norm_max=1.0");
        }

        // Output should still be finite
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "Clamped MLP y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_mlp_activation_variants() {
        // All three activation functions produce finite, non-zero output
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        for act in [MemoryActivation::GELU, MemoryActivation::SiLU, MemoryActivation::ReLU] {
            let rule = TitansLMM {
                memory_layers: 2,
                memory_expansion_factor: 2,
                memory_activation: act,
                ..TitansLMM::l2()
            };
            let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

            for (i, &v) in y.iter().enumerate() {
                assert!(v.is_finite(), "{act:?} MLP y[{i}] not finite: {v}");
            }
            assert!(cache.mlp_layout.is_some());
            assert_eq!(cache.mlp_activation, act);
        }
    }

    #[test]
    fn test_mlp_initial_m_carry() {
        // Passing initial_m should produce different output than starting from zero
        let cfg = mlp_config(2, 2);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 2,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };

        // Run once from zero to get M_T
        let (y_first, cache_first) = rule.step(&params.levels[0], &embedded, s, d, None);
        let layout = cache_first.mlp_layout.as_ref().unwrap();
        let sp = layout.total_params;
        let m_final = cache_first.m_states[s * sp..(s + 1) * sp].to_vec();

        // Run again with M_T as initial_m
        let (y_carry, _) = rule.step(&params.levels[0], &embedded, s, d, Some(m_final));

        // Output should differ because initial state is different
        let diff: f32 = y_first.iter().zip(y_carry.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6,
            "Carrying M state should produce different output, diff={diff}");
    }

    #[test]
    fn test_mlp_from_cfg_level() {
        // from_cfg_level correctly plumbs memory_layers from MAGConfig
        let cfg = mlp_config(3, 2);
        let rule = TitansLMM::from_cfg_level(&cfg, 0);
        assert_eq!(rule.memory_layers, 3);
        assert_eq!(rule.memory_expansion_factor, 2);
        assert_eq!(rule.memory_activation, MemoryActivation::GELU);
    }

    // ── MLP backward: finite-difference gradient verification ──────

    /// Scalar loss for FD checking: L = sum(y^2), so dL/dy = 2*y.
    fn mlp_fd_loss(
        rule: &TitansLMM,
        params: &MemoryLevelParams,
        embedded: &[f32],
        s: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> f32 {
        let (y, _cache) = rule.step(params, embedded, s, d, initial_m);
        y.iter().map(|v| v * v).sum()
    }

    /// Central-difference FD check for a single parameter slice.
    /// Returns (checked, passed, max_rel_err).
    fn mlp_fd_check_param(
        rule: &TitansLMM,
        params: &MemoryLevelParams,
        embedded: &[f32],
        s: usize,
        d: usize,
        initial_m: &Option<Vec<f32>>,
        name: &str,
        get: impl Fn(&MemoryLevelParams) -> &[f32],
        set: impl Fn(&mut MemoryLevelParams, usize, f32),
        get_grad: impl Fn(&MemoryLevelParams) -> &[f32],
        grads: &MemoryLevelParams,
        n_check: usize,
        eps: f32,
        tol: f32,
        abs_thresh: f32,
    ) -> (usize, usize, f32) {
        let analytical = get_grad(grads);
        let param_len = get(params).len();
        let step = if param_len <= n_check { 1 } else { param_len / n_check };
        let mut checked = 0;
        let mut passed = 0;
        let mut max_err = 0.0f32;

        for idx in (0..param_len).step_by(step) {
            let a_grad = analytical[idx];
            if a_grad.abs() < abs_thresh { continue; }

            let orig = get(params)[idx];

            let mut p_plus = params.clone();
            set(&mut p_plus, idx, orig + eps);
            let l_plus = mlp_fd_loss(rule, &p_plus, embedded, s, d, initial_m.clone());

            let mut p_minus = params.clone();
            set(&mut p_minus, idx, orig - eps);
            let l_minus = mlp_fd_loss(rule, &p_minus, embedded, s, d, initial_m.clone());

            let fd_grad = (l_plus - l_minus) / (2.0 * eps);
            let denom = a_grad.abs().max(fd_grad.abs()).max(1e-8);
            let rel_err = (a_grad - fd_grad).abs() / denom;

            checked += 1;
            if rel_err < tol {
                passed += 1;
            } else {
                eprintln!("  FAIL {name}[{idx}]: analytical={a_grad:.6e} fd={fd_grad:.6e} rel_err={rel_err:.4e}");
            }
            max_err = max_err.max(rel_err);
        }
        (checked, passed, max_err)
    }

    #[test]
    fn test_mlp_backward_fd_w_k_mem() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m.clone());
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (grads, _d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        let (checked, passed, max_err) = mlp_fd_check_param(
            &rule, &params.levels[0], &embedded, s, d, &initial_m, "w_k_mem",
            |p| p.w_k_mem.master(), |p, i, v| p.w_k_mem.set(i, v),
            |g| g.w_k_mem.master(), &grads,
            20, 1e-2, 0.10, 1e-6,
        );
        eprintln!("MLP FD w_k_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(checked > 0, "w_k_mem: no gradients above threshold");
        assert!(passed == checked, "MLP FD w_k_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mlp_backward_fd_w_v_mem() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m.clone());
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (grads, _d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        let (checked, passed, max_err) = mlp_fd_check_param(
            &rule, &params.levels[0], &embedded, s, d, &initial_m, "w_v_mem",
            |p| p.w_v_mem.master(), |p, i, v| p.w_v_mem.set(i, v),
            |g| g.w_v_mem.master(), &grads,
            20, 1e-2, 0.10, 1e-6,
        );
        eprintln!("MLP FD w_v_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(checked > 0, "w_v_mem: no gradients above threshold");
        assert!(passed == checked, "MLP FD w_v_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mlp_backward_fd_w_q_mem() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m.clone());
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (grads, _d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        let (checked, passed, max_err) = mlp_fd_check_param(
            &rule, &params.levels[0], &embedded, s, d, &initial_m, "w_q_mem",
            |p| p.w_q_mem.master(), |p, i, v| p.w_q_mem.set(i, v),
            |g| g.w_q_mem.master(), &grads,
            20, 1e-2, 0.10, 1e-6,
        );
        eprintln!("MLP FD w_q_mem: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(checked > 0, "w_q_mem: no gradients above threshold");
        assert!(passed == checked, "MLP FD w_q_mem: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_mlp_backward_fd_gates() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m.clone());
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (grads, _d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, get, set, get_grad) in [
            ("w_alpha",
             (|p: &MemoryLevelParams| p.w_alpha.as_slice()) as fn(&MemoryLevelParams) -> &[f32],
             (|p: &mut MemoryLevelParams, i: usize, v: f32| p.w_alpha[i] = v) as fn(&mut MemoryLevelParams, usize, f32),
             (|g: &MemoryLevelParams| g.w_alpha.as_slice()) as fn(&MemoryLevelParams) -> &[f32]),
            ("w_theta",
             |p: &MemoryLevelParams| p.w_theta.as_slice(),
             |p: &mut MemoryLevelParams, i: usize, v: f32| p.w_theta[i] = v,
             |g: &MemoryLevelParams| g.w_theta.as_slice()),
            ("w_eta",
             |p: &MemoryLevelParams| p.w_eta.as_slice(),
             |p: &mut MemoryLevelParams, i: usize, v: f32| p.w_eta[i] = v,
             |g: &MemoryLevelParams| g.w_eta.as_slice()),
            ("b_alpha",
             |p: &MemoryLevelParams| p.b_alpha.as_slice(),
             |p: &mut MemoryLevelParams, i: usize, v: f32| p.b_alpha[i] = v,
             |g: &MemoryLevelParams| g.b_alpha.as_slice()),
            ("b_theta",
             |p: &MemoryLevelParams| p.b_theta.as_slice(),
             |p: &mut MemoryLevelParams, i: usize, v: f32| p.b_theta[i] = v,
             |g: &MemoryLevelParams| g.b_theta.as_slice()),
            ("b_eta",
             |p: &MemoryLevelParams| p.b_eta.as_slice(),
             |p: &mut MemoryLevelParams, i: usize, v: f32| p.b_eta[i] = v,
             |g: &MemoryLevelParams| g.b_eta.as_slice()),
        ] {
            let (checked, passed, max_err) = mlp_fd_check_param(
                &rule, &params.levels[0], &embedded, s, d, &initial_m, name,
                get, set, get_grad, &grads,
                20, 1e-2, 0.10, 1e-6,
            );
            eprintln!("MLP FD {name}: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
            // Some gate grads may all be below threshold (e.g. b_eta with 1 element) — skip those
            if checked > 0 {
                assert!(passed == checked,
                    "MLP FD {name}: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
            }
        }
    }

    #[test]
    fn test_mlp_backward_fd_d_embedded() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m.clone());
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (_grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        let eps = 1e-2f32;
        let tol = 0.10;
        let abs_thresh = 1e-6;
        let n_check = 20;
        let step_sz = if embedded.len() <= n_check { 1 } else { embedded.len() / n_check };
        let mut checked = 0;
        let mut passed = 0;
        let mut max_err = 0.0f32;

        for idx in (0..embedded.len()).step_by(step_sz) {
            let a_grad = d_emb[idx];
            if a_grad.abs() < abs_thresh { continue; }

            let mut emb_plus = embedded.to_vec();
            emb_plus[idx] += eps;
            let l_plus = mlp_fd_loss(&rule, &params.levels[0], &emb_plus, s, d, initial_m.clone());

            let mut emb_minus = embedded.to_vec();
            emb_minus[idx] -= eps;
            let l_minus = mlp_fd_loss(&rule, &params.levels[0], &emb_minus, s, d, initial_m.clone());

            let fd_grad = (l_plus - l_minus) / (2.0 * eps);
            let denom = a_grad.abs().max(fd_grad.abs()).max(1e-8);
            let rel_err = (a_grad - fd_grad).abs() / denom;

            checked += 1;
            if rel_err < tol {
                passed += 1;
            } else {
                eprintln!("  FAIL d_embedded[{idx}]: analytical={a_grad:.6e} fd={fd_grad:.6e} rel_err={rel_err:.4e}");
            }
            max_err = max_err.max(rel_err);
        }
        eprintln!("MLP FD d_embedded: {passed}/{checked} pass, max_rel_err={max_err:.4e}");
        assert!(checked > 0, "d_embedded: no gradients above threshold");
        assert!(passed == checked, "MLP FD d_embedded: {passed}/{checked} passed, max_rel_err={max_err:.4e}");
    }

    /// Create a random initial MLP state with small non-zero weights.
    /// This breaks the dead-neuron cycle that occurs with zero-initialized MLP memory
    /// (W layers stay zero forever when M starts at zero because grad_W = outer(d_out, 0) = 0).
    fn make_mlp_initial_m(layout: &MLPMemoryLayout, seed: u64) -> Vec<f32> {
        let mut rng = SimpleRng::new(seed);
        let mut m = vec![0.0f32; layout.total_params];
        rng.fill_uniform(&mut m, 0.3);
        // Scale weight matrices by Xavier factor
        for l in 0..layout.n_layers {
            let desc = &layout.layers[l];
            let fan_in = desc.w_cols;
            let fan_out = desc.w_rows;
            let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
            let w = layout.w_slice_mut(&mut m, 0, l);
            for x in w.iter_mut() { *x *= scale / 0.3; }
        }
        m
    }

    #[test]
    fn test_mlp_backward_finite_and_nonzero() {
        let cfg = mlp_config(2, 4);
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let layout = MLPMemoryLayout::new(2, d, 4);
        let initial_m = Some(make_mlp_initial_m(&layout, 77));

        let rule = TitansLMM {
            memory_layers: 2,
            memory_expansion_factor: 4,
            memory_activation: MemoryActivation::GELU,
            ..TitansLMM::l2()
        };
        let (y, cache) = rule.step(&params.levels[0], &embedded, s, d, initial_m);
        let d_y: Vec<f32> = y.iter().map(|v| 2.0 * v).collect();
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);


        // All gradients must be finite
        for (name, g) in [
            ("w_k_mem", grads.w_k_mem.master()), ("w_v_mem", grads.w_v_mem.master()),
            ("w_q_mem", grads.w_q_mem.master()), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha), ("w_theta", &grads.w_theta),
            ("b_theta", &grads.b_theta), ("w_eta", &grads.w_eta),
            ("b_eta", &grads.b_eta),
        ] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "MLP grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "MLP d_embedded[{i}] not finite: {v}");
        }

        // Key projection gradients must be non-zero
        for (name, g) in [
            ("w_k_mem", grads.w_k_mem.master()), ("w_v_mem", grads.w_v_mem.master()),
            ("w_q_mem", grads.w_q_mem.master()),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "MLP grad_{name} is all zeros (max_abs={max_abs})");
        }
        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "MLP d_embedded is all zeros");
    }
}
