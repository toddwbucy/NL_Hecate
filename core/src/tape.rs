// Wengert tape: reverse-mode AD via operation recording.
//
// Spec: specs/infrastructure/differentiation/01_wengert_tape.md
//
// Records operations during the forward pass into a linear tape,
// then replays them in reverse to compute gradients via the chain rule.
// Composes with kernel pairs (memory rules, SWA) via registered opaque VJP blocks.
//
// CS-40: Opt-in — nothing recorded unless with_tape() is called.
// CS-42: All intermediates stored in arena — no recomputation.
// CS-47: Parameters are snapshotted at registration — immune to later mutation.

use std::cell::Cell;
use std::collections::HashMap;

// ── Buffer management ────────────────────────────────────────────────

/// Arena index for tensor buffers. Immutable after creation.
pub type BufId = usize;

/// A flat tensor buffer in the tape arena.
#[derive(Clone, Debug)]
pub struct TapeBuf {
    /// Flat storage (row-major).
    pub data: Vec<f32>,
    /// Shape metadata, e.g., [seq_len, d_model].
    pub shape: Vec<usize>,
    /// True for outer-loop parameters (W_K, W_V, etc.) — these get gradient output.
    pub is_param: bool,
}

impl TapeBuf {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        TapeBuf { data, shape, is_param: false }
    }

    pub fn param(data: Vec<f32>, shape: Vec<usize>) -> Self {
        TapeBuf { data, shape, is_param: true }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

// ── Opaque VJP system ────────────────────────────────────────────────

/// Key identifying which opaque backward function to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpaqueKey {
    // Active memory rules (full write + read cycle)
    DeltaRule,
    TitansLMM,
    HebbianRule,
    Moneta,
    YAAD,
    MEMORA,
    LatticeOSR,
    Trellis,
    AtlasOmega,

    // Sliding window attention
    SWA,

    // Frozen read-only variants (memory read without write)
    FrozenDeltaRule,
    FrozenTitansLMM,
    FrozenHebbianRule,
    FrozenMoneta,
    FrozenYAAD,
    FrozenMEMORA,
    FrozenLatticeOSR,
    FrozenTrellis,
    FrozenAtlasOmega,
}

/// Backward function for an opaque block.
///
/// - `d_outputs`: upstream gradient for each output buffer
/// - `saved`: tensors saved during forward (cache data)
/// - `d_inputs`: output — gradient for each input buffer (caller-allocated)
pub type OpaqueBackwardFn = fn(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],
    d_inputs: &mut [Vec<f32>],
);

// ── OpaqueVjp trait ──────────────────────────────────────────────────

/// Marker + recording trait for types whose forward pass is opaque to the tape.
///
/// Replaces the spec's `EnzymeOpaque` marker. Any type implementing this trait
/// can register its forward execution on the tape as an opaque block with a
/// known backward function (looked up by `opaque_key()` in the registry).
///
/// Implementations live in `opaque_adapters.rs` alongside the backward adapters.
pub trait OpaqueVjp {
    /// Which registry key maps to this type's backward adapter.
    fn opaque_key(&self) -> OpaqueKey;

    /// Execute the forward pass and record it on the tape.
    ///
    /// Contract:
    /// - Snapshots `level_params` and `embedded` into tape arena (CS-47)
    /// - Calls `step()` to execute the forward pass
    /// - Flattens the rule-specific cache into saved buffers matching the
    ///   adapter's expected layout in `opaque_adapters.rs`
    /// - Pushes `TapeOp::Opaque` with the correct key, inputs, outputs, saved
    /// - Returns (output Vec<f32>, output BufId) for downstream tape ops
    ///
    /// `initial_m`: Optional initial memory state (from CMS context_memory).
    fn record_on_tape(
        &self,
        tape: &mut Tape,
        level_params: &crate::model::MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, BufId);
}

// ── Tape operations ──────────────────────────────────────────────────

/// A single recorded operation on the tape.
#[derive(Debug, Clone)]
pub enum TapeOp {
    // ── Linear algebra ──────────────────────────────────────────
    /// out = A @ B where A: [m, k], B: [k, n], out: [m, n]
    Matmul { a: BufId, b: BufId, out: BufId, m: usize, k: usize, n: usize },
    /// out = A @ B^T where A: [m, k], B: [n, k], out: [m, n]
    MatmulTransposeB { a: BufId, b: BufId, out: BufId, m: usize, k: usize, n: usize },
    /// out = A^T where A: [rows, cols], out: [cols, rows]
    Transpose { input: BufId, out: BufId, rows: usize, cols: usize },
    /// out = a ⊗ b where a: [d1], b: [d2], out: [d1, d2]
    OuterProduct { a: BufId, b: BufId, out: BufId },
    /// out = sum_ij(A[i,j] * B[i,j])  (scalar)
    FrobeniusDot { a: BufId, b: BufId, out: BufId },

    // ── Element-wise ────────────────────────────────────────────
    /// out = A + B
    Add { a: BufId, b: BufId, out: BufId },
    /// out = A - B
    Sub { a: BufId, b: BufId, out: BufId },
    /// out = A * B  (element-wise)
    Mul { a: BufId, b: BufId, out: BufId },
    /// out = scalar * A
    Scale { input: BufId, scalar: f32, out: BufId },
    /// out = -A
    Negate { input: BufId, out: BufId },

    // ── Activations ─────────────────────────────────────────────
    /// out = sigmoid(x); saves output for backward
    Sigmoid { input: BufId, out: BufId },
    /// out = log(1 + exp(x)); numerically stable
    Softplus { input: BufId, out: BufId },
    /// out = x * sigmoid(x); saves input for backward
    SiLU { input: BufId, out: BufId },

    // ── Reductions / structured ─────────────────────────────────
    /// out = softmax(x) per-row; x: [rows, cols]
    Softmax { input: BufId, out: BufId, rows: usize, cols: usize },
    /// out = -mean(log(softmax(logits)[target])); combined for stability
    CrossEntropy { logits: BufId, targets: Vec<usize>, out: BufId, vocab_size: usize },
    /// out[t] = table[indices[t]]; scatter-add backward
    EmbedLookup { table: BufId, indices: Vec<usize>, out: BufId, vocab_size: usize, d: usize },
    /// out = ||x||_2
    L2Norm { input: BufId, out: BufId },

    // ── Retention ───────────────────────────────────────────────
    /// out = lambda * input
    L2Retention { input: BufId, lambda: f32, out: BufId },

    // ── Concat / reshape ────────────────────────────────────────
    /// out = concat(inputs, axis)
    Concat { inputs: Vec<BufId>, out: BufId, axis: usize,
             /// Per-input sizes along the concat axis, for backward slicing.
             sizes: Vec<usize> },
    /// out = input[offset..offset+len]
    Slice { input: BufId, out: BufId, offset: usize, len: usize, input_len: usize },

    // ── NL-specific ops ─────────────────────────────────────────
    /// Trellis: y = normalize(x * sigmoid(x))
    NormalizedSiLU { input: BufId, out: BufId },
    /// Lattice OSR: project each of m slots to unit sphere
    SphereProjectNormalize { input: BufId, out: BufId, d: usize, m_slots: usize },
    /// MEMORA: softmax(alpha * log(prior) - theta * grad)
    KLRetention { input: BufId, prior: BufId, alpha: f32, theta: f32, out: BufId },
    /// Frequency gate: hard threshold with straight-through estimator
    StraightThroughBool { input: BufId, threshold: f32, out: BufId },

    // ── Opaque blocks ───────────────────────────────────────────
    /// Registered VJP block (memory rules, SWA, frozen variants).
    Opaque {
        key: OpaqueKey,
        inputs: Vec<BufId>,
        outputs: Vec<BufId>,
        saved: Vec<BufId>,
    },
}

// ── The Tape ─────────────────────────────────────────────────────────

/// Wengert tape for reverse-mode AD.
///
/// Records operations during forward pass, replays in reverse for gradients.
/// Created via `with_tape()`, dropped after backward pass completes.
pub struct Tape {
    /// Operations in forward order. Replayed in reverse during backward.
    ops: Vec<TapeOp>,
    /// Arena of tensor buffers. Indexed by BufId.
    bufs: Vec<TapeBuf>,
    /// Gradient accumulators, indexed by BufId. None until backward seeds them.
    grad_accum: Vec<Option<Vec<f32>>>,
    /// Whether we are currently recording (always true between creation and backward).
    recording: bool,
    /// Registry of opaque backward functions.
    opaque_registry: HashMap<OpaqueKey, OpaqueBackwardFn>,
}

impl Tape {
    /// Create a new empty tape with the given opaque VJP registry.
    pub fn new(registry: HashMap<OpaqueKey, OpaqueBackwardFn>) -> Self {
        Tape {
            ops: Vec::new(),
            bufs: Vec::new(),
            grad_accum: Vec::new(),
            recording: true,
            opaque_registry: registry,
        }
    }

    /// Create a tape with an empty registry (for testing standard ops only).
    pub fn new_empty() -> Self {
        Tape::new(HashMap::new())
    }

    // ── Buffer management ────────────────────────────────────────

    /// Allocate a new buffer in the arena. Returns its BufId.
    pub fn alloc(&mut self, data: Vec<f32>, shape: Vec<usize>) -> BufId {
        let id = self.bufs.len();
        self.bufs.push(TapeBuf::new(data, shape));
        self.grad_accum.push(None);
        id
    }

    /// Register an outer-loop parameter. CLONES the data (CS-47 snapshot).
    /// The tape holds its own copy, immune to later mutation of the original.
    pub fn register_param(&mut self, data: &[f32], shape: Vec<usize>) -> BufId {
        let id = self.bufs.len();
        self.bufs.push(TapeBuf::param(data.to_vec(), shape));
        self.grad_accum.push(None);
        id
    }

    /// Register an input (non-parameter) buffer. Clones the data.
    pub fn register_input(&mut self, data: &[f32], shape: Vec<usize>) -> BufId {
        self.alloc(data.to_vec(), shape)
    }

    /// Get the data for a buffer.
    pub fn buf_data(&self, id: BufId) -> &[f32] {
        &self.bufs[id].data
    }

    /// Get the shape for a buffer.
    pub fn buf_shape(&self, id: BufId) -> &[usize] {
        &self.bufs[id].shape
    }

    /// Get the number of elements in a buffer.
    pub fn buf_numel(&self, id: BufId) -> usize {
        self.bufs[id].numel()
    }

    /// Check if a buffer is a parameter.
    pub fn is_param(&self, id: BufId) -> bool {
        self.bufs[id].is_param
    }

    /// Number of buffers in the arena.
    pub fn num_bufs(&self) -> usize {
        self.bufs.len()
    }

    /// Number of ops recorded.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    // ── Recording ────────────────────────────────────────────────

    /// Whether the tape is currently recording.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record an operation on the tape. Also allocates the output buffer.
    /// Returns the output BufId.
    pub fn record(&mut self, op: TapeOp) {
        assert!(self.recording, "Tape::record called but tape is not recording");
        self.ops.push(op);
    }

    /// Record an operation and allocate its output buffer in one step.
    /// Returns the output BufId.
    pub fn record_with_output(&mut self, data: Vec<f32>, shape: Vec<usize>,
                               op_fn: impl FnOnce(BufId) -> TapeOp) -> BufId {
        let out_id = self.alloc(data, shape);
        let op = op_fn(out_id);
        self.record(op);
        out_id
    }

    /// Record an opaque block. Inputs and saved must already be allocated.
    /// Outputs are allocated by the caller before this call.
    pub fn record_opaque(&mut self, key: OpaqueKey, inputs: Vec<BufId>,
                          outputs: Vec<BufId>, saved: Vec<BufId>) {
        assert!(self.opaque_registry.contains_key(&key),
                "No opaque backward registered for {:?}", key);
        self.record(TapeOp::Opaque { key, inputs, outputs, saved });
    }

    // ── Gradient seeding and access ──────────────────────────────

    /// Seed the gradient for a buffer (typically the scalar loss).
    pub fn seed_grad(&mut self, id: BufId, grad: Vec<f32>) {
        assert_eq!(grad.len(), self.bufs[id].numel(),
                   "Gradient size mismatch: grad={} buf={}", grad.len(), self.bufs[id].numel());
        self.grad_accum[id] = Some(grad);
    }

    /// Accumulate gradient into a buffer's accumulator.
    fn accumulate_grad(&mut self, id: BufId, grad: &[f32]) {
        let n = self.bufs[id].numel();
        assert_eq!(grad.len(), n, "accumulate_grad size mismatch: grad={} buf={}", grad.len(), n);
        match &mut self.grad_accum[id] {
            Some(existing) => {
                for (e, g) in existing.iter_mut().zip(grad.iter()) {
                    *e += g;
                }
            }
            None => {
                self.grad_accum[id] = Some(grad.to_vec());
            }
        }
    }

    /// Get the accumulated gradient for a buffer. Returns None if no gradient flowed to it.
    pub fn get_grad(&self, id: BufId) -> Option<&[f32]> {
        self.grad_accum[id].as_deref()
    }

    /// Get gradient for a parameter buffer, returning zeros if no gradient flowed.
    pub fn get_param_grad(&self, id: BufId) -> Vec<f32> {
        assert!(self.bufs[id].is_param, "get_param_grad called on non-param buffer {}", id);
        match &self.grad_accum[id] {
            Some(g) => g.clone(),
            None => vec![0.0; self.bufs[id].numel()],
        }
    }

    // ── Backward pass ────────────────────────────────────────────

    /// Run the backward pass: replay ops in reverse, computing VJPs.
    /// Must call seed_grad() on the loss BufId before calling this.
    pub fn backward(&mut self, loss_id: BufId) {
        self.recording = false;

        // Seed the loss with 1.0 if not already seeded.
        if self.grad_accum[loss_id].is_none() {
            assert_eq!(self.bufs[loss_id].numel(), 1,
                       "Auto-seeding only works for scalar loss (got {} elements)",
                       self.bufs[loss_id].numel());
            self.grad_accum[loss_id] = Some(vec![1.0]);
        }

        // Process ops in reverse order.
        for op_idx in (0..self.ops.len()).rev() {
            // Clone the op to avoid borrow conflict with self.
            let op = self.ops[op_idx].clone();
            self.backward_op(&op);
        }
    }

    /// Compute VJP for a single operation.
    fn backward_op(&mut self, op: &TapeOp) {
        use crate::tensor;

        match op {
            // ── Matmul: out = A @ B ──────────────────────────────
            TapeOp::Matmul { a, b, out, m, k, n } => {
                let (m, k, n) = (*m, *k, *n);
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // d_A = d_out @ B^T
                    let b_data = self.bufs[*b].data.clone();
                    let mut d_a = vec![0.0f32; m * k];
                    let mut b_t = vec![0.0f32; n * k];
                    tensor::transpose_f32(&b_data, &mut b_t, k, n);
                    tensor::matmul_f32(&d_out, &b_t, &mut d_a, m, n, k);
                    self.accumulate_grad(*a, &d_a);

                    // d_B = A^T @ d_out
                    let a_data = self.bufs[*a].data.clone();
                    let mut d_b = vec![0.0f32; k * n];
                    let mut a_t = vec![0.0f32; k * m];
                    tensor::transpose_f32(&a_data, &mut a_t, m, k);
                    tensor::matmul_f32(&a_t, &d_out, &mut d_b, k, m, n);
                    self.accumulate_grad(*b, &d_b);
                }
            }

            // ── MatmulTransposeB: out = A @ B^T ─────────────────
            TapeOp::MatmulTransposeB { a, b, out, m, k, n } => {
                let (m, k, n) = (*m, *k, *n);
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // d_A = d_out @ B  (d_out: m×n, B: n×k → d_A: m×k)
                    let b_data = self.bufs[*b].data.clone();
                    let mut d_a = vec![0.0f32; m * k];
                    tensor::matmul_f32(&d_out, &b_data, &mut d_a, m, n, k);
                    self.accumulate_grad(*a, &d_a);

                    // d_B = d_out^T @ A  (d_out^T: n×m, A: m×k → d_B: n×k)
                    let a_data = self.bufs[*a].data.clone();
                    let mut d_out_t = vec![0.0f32; n * m];
                    tensor::transpose_f32(&d_out, &mut d_out_t, m, n);
                    let mut d_b = vec![0.0f32; n * k];
                    tensor::matmul_f32(&d_out_t, &a_data, &mut d_b, n, m, k);
                    self.accumulate_grad(*b, &d_b);
                }
            }

            // ── Transpose: out = A^T ─────────────────────────────
            TapeOp::Transpose { input, out, rows, cols } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // d_A = d_out^T (transpose back)
                    let mut d_input = vec![0.0f32; *rows * *cols];
                    tensor::transpose_f32(&d_out, &mut d_input, *cols, *rows);
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── OuterProduct: out = a ⊗ b ────────────────────────
            TapeOp::OuterProduct { a, b, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let a_data = self.bufs[*a].data.clone();
                    let b_data = self.bufs[*b].data.clone();
                    let d1 = a_data.len();
                    let d2 = b_data.len();

                    // d_a[i] = sum_j(d_out[i,j] * b[j])
                    let mut d_a = vec![0.0f32; d1];
                    for i in 0..d1 {
                        for j in 0..d2 {
                            d_a[i] += d_out[i * d2 + j] * b_data[j];
                        }
                    }
                    self.accumulate_grad(*a, &d_a);

                    // d_b[j] = sum_i(d_out[i,j] * a[i])
                    let mut d_b = vec![0.0f32; d2];
                    for j in 0..d2 {
                        for i in 0..d1 {
                            d_b[j] += d_out[i * d2 + j] * a_data[i];
                        }
                    }
                    self.accumulate_grad(*b, &d_b);
                }
            }

            // ── FrobeniusDot: out = <A, B>_F (scalar) ────────────
            TapeOp::FrobeniusDot { a, b, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let scalar = d_out[0];
                    // d_A = d_out * B
                    let d_a: Vec<f32> = self.bufs[*b].data.iter().map(|x| scalar * x).collect();
                    self.accumulate_grad(*a, &d_a);
                    // d_B = d_out * A
                    let d_b: Vec<f32> = self.bufs[*a].data.iter().map(|x| scalar * x).collect();
                    self.accumulate_grad(*b, &d_b);
                }
            }

            // ── Add: out = A + B ─────────────────────────────────
            TapeOp::Add { a, b, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    self.accumulate_grad(*a, &d_out);
                    self.accumulate_grad(*b, &d_out);
                }
            }

            // ── Sub: out = A - B ─────────────────────────────────
            TapeOp::Sub { a, b, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    self.accumulate_grad(*a, &d_out);
                    let neg: Vec<f32> = d_out.iter().map(|x| -x).collect();
                    self.accumulate_grad(*b, &neg);
                }
            }

            // ── Mul: out = A * B (element-wise) ──────────────────
            TapeOp::Mul { a, b, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // d_A = d_out * B
                    let d_a: Vec<f32> = d_out.iter().zip(self.bufs[*b].data.iter())
                        .map(|(d, b)| d * b).collect();
                    self.accumulate_grad(*a, &d_a);
                    // d_B = d_out * A
                    let d_b: Vec<f32> = d_out.iter().zip(self.bufs[*a].data.iter())
                        .map(|(d, a)| d * a).collect();
                    self.accumulate_grad(*b, &d_b);
                }
            }

            // ── Scale: out = scalar * A ──────────────────────────
            TapeOp::Scale { input, scalar, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let d_input: Vec<f32> = d_out.iter().map(|d| *scalar * d).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── Negate: out = -A ─────────────────────────────────
            TapeOp::Negate { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let d_input: Vec<f32> = d_out.iter().map(|d| -d).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── Sigmoid: out = σ(x) ──────────────────────────────
            TapeOp::Sigmoid { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let out_data = &self.bufs[*out].data;
                    // d_x = d_out * out * (1 - out)
                    let d_input: Vec<f32> = d_out.iter().zip(out_data.iter())
                        .map(|(d, o)| d * o * (1.0 - o)).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── Softplus: out = log(1 + exp(x)) ─────────────────
            TapeOp::Softplus { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let out_data = &self.bufs[*out].data;
                    // d_x = d_out * σ(x) where σ(x) = 1 - exp(-out)
                    let d_input: Vec<f32> = d_out.iter().zip(out_data.iter())
                        .map(|(d, o)| d * (1.0 - (-o).exp())).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── SiLU: out = x * σ(x) ────────────────────────────
            TapeOp::SiLU { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let x = &self.bufs[*input].data;
                    // d_x = d_out * (σ(x) + x * σ(x) * (1 - σ(x)))
                    let d_input: Vec<f32> = d_out.iter().zip(x.iter())
                        .map(|(d, &xi)| {
                            let sig = tensor::sigmoid_f32(xi);
                            d * (sig + xi * sig * (1.0 - sig))
                        }).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── Softmax: out = softmax(x) per-row ───────────────
            TapeOp::Softmax { input, out, rows, cols } => {
                self.backward_softmax(*input, *out, *rows, *cols);
            }

            // ── CrossEntropy: out = -mean(log(softmax(logits)[target])) ──
            TapeOp::CrossEntropy { logits, targets, out, vocab_size } => {
                self.backward_cross_entropy(*logits, targets, *out, *vocab_size);
            }

            // ── EmbedLookup: out[t] = table[indices[t]] ─────────
            TapeOp::EmbedLookup { table, indices, out, vocab_size, d } => {
                self.backward_embed_lookup(*table, indices, *out, *vocab_size, *d);
            }

            // ── L2Norm: out = ||x||_2 ────────────────────────────
            TapeOp::L2Norm { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let x = &self.bufs[*input].data;
                    let norm = self.bufs[*out].data[0];
                    let inv_norm = if norm > 1e-8 { 1.0 / norm } else { 0.0 };
                    let scalar = d_out[0];
                    let d_input: Vec<f32> = x.iter().map(|xi| scalar * xi * inv_norm).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── L2Retention: out = lambda * input ────────────────
            TapeOp::L2Retention { input, lambda, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let d_input: Vec<f32> = d_out.iter().map(|d| *lambda * d).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── Concat ──────────────────────────────────────────
            TapeOp::Concat { inputs, out, axis: _, sizes } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // Slice d_out back to each input.
                    let mut offset = 0;
                    for (inp, &sz) in inputs.iter().zip(sizes.iter()) {
                        let d_inp = d_out[offset..offset + sz].to_vec();
                        self.accumulate_grad(*inp, &d_inp);
                        offset += sz;
                    }
                }
            }

            // ── Slice ───────────────────────────────────────────
            TapeOp::Slice { input, out, offset, len, input_len } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let mut d_input = vec![0.0f32; *input_len];
                    d_input[*offset..*offset + *len].copy_from_slice(&d_out);
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── NormalizedSiLU (Trellis) ─────────────────────────
            TapeOp::NormalizedSiLU { input, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let x = &self.bufs[*input].data;
                    let out_data = &self.bufs[*out].data;
                    let n = x.len();

                    // Compute y = silu(x) and norm = ||y||
                    let y: Vec<f32> = x.iter().map(|&xi| xi * tensor::sigmoid_f32(xi)).collect();
                    let norm = y.iter().map(|yi| yi * yi).sum::<f32>().sqrt().max(1e-8);

                    // d_y = (d_out - out * dot(d_out, out)) / norm  (normalize VJP)
                    let dot: f32 = d_out.iter().zip(out_data.iter()).map(|(a, b)| a * b).sum();
                    let mut d_y = vec![0.0f32; n];
                    for i in 0..n {
                        d_y[i] = (d_out[i] - out_data[i] * dot) / norm;
                    }

                    // d_x = d_y * silu'(x)
                    let d_input: Vec<f32> = d_y.iter().zip(x.iter())
                        .map(|(&dy, &xi)| {
                            let sig = tensor::sigmoid_f32(xi);
                            dy * (sig + xi * sig * (1.0 - sig))
                        }).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── SphereProjectNormalize (Lattice OSR) ─────────────
            TapeOp::SphereProjectNormalize { input, out, d, m_slots } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let x = &self.bufs[*input].data;
                    let d_dim = *d;
                    let mut d_input = vec![0.0f32; *m_slots * d_dim];
                    for slot in 0..*m_slots {
                        let base = slot * d_dim;
                        let s = &x[base..base + d_dim];
                        let norm = s.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                        let s_norm: Vec<f32> = s.iter().map(|v| v / norm).collect();
                        let dot: f32 = d_out[base..base + d_dim].iter()
                            .zip(s_norm.iter()).map(|(a, b)| a * b).sum();
                        for i in 0..d_dim {
                            d_input[base + i] = (d_out[base + i] - s_norm[i] * dot) / norm;
                        }
                    }
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── KLRetention (MEMORA) ─────────────────────────────
            TapeOp::KLRetention { input, prior, alpha, theta, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    let out_data = &self.bufs[*out].data;
                    let prior_data = &self.bufs[*prior].data;
                    let n = out_data.len();

                    // This operates per-row via softmax, but for simplicity treat as flat.
                    // d_z = softmax_vjp(d_out, out)
                    // For a single-row softmax:
                    let dot: f32 = d_out.iter().zip(out_data.iter()).map(|(a, b)| a * b).sum();
                    let d_z: Vec<f32> = (0..n).map(|i| out_data[i] * (d_out[i] - dot)).collect();

                    // d_prior[j] = d_z[j] * alpha / max(prior[j], 1e-8)
                    let d_prior: Vec<f32> = d_z.iter().zip(prior_data.iter())
                        .map(|(&dz, &p)| dz * *alpha / p.max(1e-8)).collect();
                    self.accumulate_grad(*prior, &d_prior);

                    // d_input[j] = -d_z[j] * theta
                    let d_input: Vec<f32> = d_z.iter().map(|&dz| -dz * *theta).collect();
                    self.accumulate_grad(*input, &d_input);
                }
            }

            // ── StraightThroughBool ──────────────────────────────
            TapeOp::StraightThroughBool { input, threshold: _, out } => {
                if let Some(d_out) = self.grad_accum[*out].clone() {
                    // Straight-through: gradient passes through unchanged.
                    self.accumulate_grad(*input, &d_out);
                }
            }

            // ── Opaque: registered VJP block ─────────────────────
            TapeOp::Opaque { key, inputs, outputs, saved } => {
                // Collect upstream gradients for all outputs.
                let d_outputs: Vec<Vec<f32>> = outputs.iter().map(|&oid| {
                    self.grad_accum[oid].clone().unwrap_or_else(|| vec![0.0; self.bufs[oid].numel()])
                }).collect();
                let d_out_refs: Vec<&[f32]> = d_outputs.iter().map(|v| v.as_slice()).collect();

                // Collect saved tensors.
                let saved_data: Vec<Vec<f32>> = saved.iter().map(|&sid| {
                    self.bufs[sid].data.clone()
                }).collect();
                let saved_refs: Vec<&[f32]> = saved_data.iter().map(|v| v.as_slice()).collect();

                // Allocate output gradient buffers.
                let mut d_inputs: Vec<Vec<f32>> = inputs.iter().map(|&iid| {
                    vec![0.0f32; self.bufs[iid].numel()]
                }).collect();

                // Call registered backward function.
                let backward_fn = self.opaque_registry[key];
                backward_fn(&d_out_refs, &saved_refs, &mut d_inputs);

                // Accumulate gradients into input buffers.
                for (iid, d_inp) in inputs.iter().zip(d_inputs.iter()) {
                    self.accumulate_grad(*iid, d_inp);
                }
            }
        }
    }
}

impl Tape {
    /// Corrected VJP for Softmax.
    fn backward_softmax(&mut self, input: BufId, out: BufId, rows: usize, cols: usize) {
        if let Some(d_out) = self.grad_accum[out].clone() {
            let out_data = &self.bufs[out].data;
            let mut d_input = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let base = r * cols;
                let mut dot = 0.0f32;
                for c in 0..cols {
                    dot += d_out[base + c] * out_data[base + c];
                }
                for c in 0..cols {
                    d_input[base + c] = out_data[base + c] * (d_out[base + c] - dot);
                }
            }
            self.accumulate_grad(input, &d_input);
        }
    }

    /// Corrected VJP for CrossEntropy (combined softmax + NLL).
    fn backward_cross_entropy(&mut self, logits: BufId, targets: &[usize],
                               out: BufId, vocab_size: usize) {
        if let Some(d_out) = self.grad_accum[out].clone() {
            let logit_data = &self.bufs[logits].data;
            let seq_len = targets.len();
            let v = vocab_size;
            let scalar = d_out[0]; // loss is scalar

            let n_valid = targets.iter().filter(|&&t| t < v).count() as f32;
            if n_valid == 0.0 { return; }

            let mut d_logits = vec![0.0f32; seq_len * v];
            for t in 0..seq_len {
                let target = targets[t];
                if target >= v { continue; }
                let base = t * v;
                let row = &logit_data[base..base + v];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for j in 0..v {
                    let e = (row[j] - max_val).exp();
                    d_logits[base + j] = e;
                    sum_exp += e;
                }
                for j in 0..v {
                    d_logits[base + j] /= sum_exp;
                }
                d_logits[base + target] -= 1.0;
                for j in 0..v {
                    d_logits[base + j] *= scalar / n_valid;
                }
            }
            self.accumulate_grad(logits, &d_logits);
        }
    }

    /// Corrected VJP for EmbedLookup (scatter-add).
    fn backward_embed_lookup(&mut self, table: BufId, indices: &[usize],
                              out: BufId, vocab_size: usize, d: usize) {
        if let Some(d_out) = self.grad_accum[out].clone() {
            let mut d_table = vec![0.0f32; vocab_size * d];
            for (t, &tok) in indices.iter().enumerate() {
                if tok >= vocab_size { continue; }
                for i in 0..d {
                    d_table[tok * d + i] += d_out[t * d + i];
                }
            }
            self.accumulate_grad(table, &d_table);
        }
    }
}

// ── Thread-local tape access (CS-40: opt-in) ─────────────────────────

thread_local! {
    static TAPE_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Drop guard that clears the TAPE_ACTIVE flag when scope exits, including
/// on panic. Ensures `is_tape_active()` is never left stale.
struct TapeGuard;

impl Drop for TapeGuard {
    fn drop(&mut self) {
        TAPE_ACTIVE.with(|flag| flag.set(false));
    }
}

/// Execute a closure with an active tape. Sets the thread-local active flag
/// for the duration of `f`, making `is_tape_active()` return true. After `f`
/// returns (or panics) the flag is cleared via a drop guard. This is the sole
/// entry point for AD — nothing is recorded unless this is called (CS-40).
pub fn with_tape<F, R>(registry: HashMap<OpaqueKey, OpaqueBackwardFn>, f: F) -> R
where
    F: FnOnce(&mut Tape) -> R,
{
    TAPE_ACTIVE.with(|flag| {
        debug_assert!(!flag.get(), "nested with_tape() calls are not supported");
        flag.set(true);
    });

    let _guard = TapeGuard;
    let mut tape = Tape::new(registry);
    f(&mut tape)
    // _guard dropped here, resetting flag even on panic.
}

/// Execute with an empty opaque registry (for testing standard ops only).
pub fn with_tape_empty<F, R>(f: F) -> R
where
    F: FnOnce(&mut Tape) -> R,
{
    with_tape(HashMap::new(), f)
}

/// Check if a tape is currently active on this thread. Returns true only
/// while inside a `with_tape()` closure. Traced wrappers can use this to
/// decide whether to record operations.
pub fn is_tape_active() -> bool {
    TAPE_ACTIVE.with(|flag| flag.get())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    #[test]
    fn test_tape_alloc_and_access() {
        let mut tape = Tape::new_empty();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let id = tape.alloc(data.clone(), vec![2, 2]);
        assert_eq!(tape.buf_data(id), &data[..]);
        assert_eq!(tape.buf_shape(id), &[2, 2]);
        assert_eq!(tape.buf_numel(id), 4);
        assert!(!tape.is_param(id));
    }

    #[test]
    fn test_register_param_clones() {
        let mut tape = Tape::new_empty();
        let mut original = vec![1.0, 2.0, 3.0];
        let id = tape.register_param(&original, vec![3]);
        // Mutate original — tape should be unaffected (CS-47).
        original[0] = 999.0;
        assert_eq!(tape.buf_data(id)[0], 1.0);
        assert!(tape.is_param(id));
    }

    #[test]
    fn test_backward_add() {
        let mut tape = Tape::new_empty();
        let a_id = tape.alloc(vec![1.0, 2.0], vec![2]);
        let b_id = tape.alloc(vec![3.0, 4.0], vec![2]);
        let out_id = tape.alloc(vec![4.0, 6.0], vec![2]);
        tape.record(TapeOp::Add { a: a_id, b: b_id, out: out_id });
        tape.seed_grad(out_id, vec![1.0, 1.0]);
        tape.backward(out_id);
        assert_eq!(tape.get_grad(a_id).unwrap(), &[1.0, 1.0]);
        assert_eq!(tape.get_grad(b_id).unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn test_backward_mul() {
        let mut tape = Tape::new_empty();
        let a_id = tape.alloc(vec![2.0, 3.0], vec![2]);
        let b_id = tape.alloc(vec![4.0, 5.0], vec![2]);
        let out_id = tape.alloc(vec![8.0, 15.0], vec![2]);
        tape.record(TapeOp::Mul { a: a_id, b: b_id, out: out_id });
        tape.seed_grad(out_id, vec![1.0, 1.0]);
        tape.backward(out_id);
        // d_a = d_out * b = [4, 5], d_b = d_out * a = [2, 3]
        assert_eq!(tape.get_grad(a_id).unwrap(), &[4.0, 5.0]);
        assert_eq!(tape.get_grad(b_id).unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn test_backward_scale() {
        let mut tape = Tape::new_empty();
        let inp = tape.alloc(vec![1.0, 2.0, 3.0], vec![3]);
        let out = tape.alloc(vec![2.0, 4.0, 6.0], vec![3]);
        tape.record(TapeOp::Scale { input: inp, scalar: 2.0, out });
        tape.seed_grad(out, vec![1.0, 1.0, 1.0]);
        tape.backward(out);
        assert_eq!(tape.get_grad(inp).unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_backward_sigmoid() {
        let mut tape = Tape::new_empty();
        let x = vec![0.0f32]; // sigmoid(0) = 0.5
        let sig_x = vec![0.5f32];
        let inp = tape.alloc(x, vec![1]);
        let out = tape.alloc(sig_x, vec![1]);
        tape.record(TapeOp::Sigmoid { input: inp, out });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        // d_x = d_out * out * (1 - out) = 1 * 0.5 * 0.5 = 0.25
        let grad = tape.get_grad(inp).unwrap();
        assert!((grad[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_backward_matmul() {
        // out = A @ B where A: [1,2], B: [2,1] → out: [1,1]
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0, 2.0], vec![1, 2]);       // [1,2]
        let b = tape.alloc(vec![3.0, 4.0], vec![2, 1]);       // [2,1]
        let out = tape.alloc(vec![11.0], vec![1, 1]);          // 1*3 + 2*4 = 11
        tape.record(TapeOp::Matmul { a, b, out, m: 1, k: 2, n: 1 });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        // d_A = d_out @ B^T = [1] @ [3, 4] = [3, 4]
        assert_eq!(tape.get_grad(a).unwrap(), &[3.0, 4.0]);
        // d_B = A^T @ d_out = [[1],[2]] @ [1] = [[1],[2]]
        assert_eq!(tape.get_grad(b).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_backward_chain_mul_add() {
        // c = a * b, out = c + a  →  d_a = d_out * b + d_out = b + 1
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![3.0], vec![1]);
        let b = tape.alloc(vec![5.0], vec![1]);
        let c = tape.alloc(vec![15.0], vec![1]);  // a * b = 15
        let out = tape.alloc(vec![18.0], vec![1]); // c + a = 18
        tape.record(TapeOp::Mul { a, b, out: c });
        tape.record(TapeOp::Add { a: c, b: a, out });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        // d_out = 1
        // Add VJP: d_c = 1, d_a += 1
        // Mul VJP: d_a += d_c * b = 5, d_b = d_c * a = 3
        // Total: d_a = 1 + 5 = 6, d_b = 3
        let grad_a = tape.get_grad(a).unwrap();
        let grad_b = tape.get_grad(b).unwrap();
        assert!((grad_a[0] - 6.0).abs() < 1e-6, "d_a={} expected 6", grad_a[0]);
        assert!((grad_b[0] - 3.0).abs() < 1e-6, "d_b={} expected 3", grad_b[0]);
    }

    #[test]
    fn test_backward_matmul_transpose_b() {
        // out = A @ B^T where A: [1,2], B: [1,2] → out: [1,1]
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0, 2.0], vec![1, 2]);
        let b = tape.alloc(vec![3.0, 4.0], vec![1, 2]);
        // A @ B^T = [1,2] @ [[3],[4]] = [11]
        let out = tape.alloc(vec![11.0], vec![1, 1]);
        tape.record(TapeOp::MatmulTransposeB { a, b, out, m: 1, k: 2, n: 1 });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        // d_A = d_out @ B = [1] @ [3,4] = [3,4]
        assert_eq!(tape.get_grad(a).unwrap(), &[3.0, 4.0]);
        // d_B = d_out^T @ A = [1] @ [1,2] = [1,2]
        assert_eq!(tape.get_grad(b).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_opaque_backward_called() {
        // Register a simple opaque backward that doubles the upstream gradient.
        fn double_backward(
            d_outputs: &[&[f32]],
            _saved: &[&[f32]],
            d_inputs: &mut [Vec<f32>],
        ) {
            for (d_inp, d_out) in d_inputs.iter_mut().zip(d_outputs.iter()) {
                for (di, &do_) in d_inp.iter_mut().zip(d_out.iter()) {
                    *di = 2.0 * do_;
                }
            }
        }

        let mut registry = HashMap::new();
        registry.insert(OpaqueKey::DeltaRule, double_backward as OpaqueBackwardFn);

        let mut tape = Tape::new(registry);
        let inp = tape.alloc(vec![1.0, 2.0], vec![2]);
        let out = tape.alloc(vec![3.0, 4.0], vec![2]);
        tape.record_opaque(OpaqueKey::DeltaRule, vec![inp], vec![out], vec![]);
        tape.seed_grad(out, vec![1.0, 1.0]);
        tape.backward(out);
        // Opaque backward doubles: d_inp = 2 * d_out = [2, 2]
        assert_eq!(tape.get_grad(inp).unwrap(), &[2.0, 2.0]);
    }

    #[test]
    fn test_with_tape_empty() {
        // is_tape_active() should be false before with_tape.
        assert!(!is_tape_active());

        let result = with_tape_empty(|tape| {
            // is_tape_active() should be true inside with_tape.
            assert!(is_tape_active());

            let a = tape.alloc(vec![1.0, 2.0], vec![2]);
            let b = tape.alloc(vec![3.0, 4.0], vec![2]);
            let out = tape.alloc(vec![4.0, 6.0], vec![2]);
            tape.record(TapeOp::Add { a, b, out });
            tape.seed_grad(out, vec![1.0, 1.0]);
            tape.backward(out);
            tape.get_grad(a).unwrap().to_vec()
        });
        assert_eq!(result, vec![1.0, 1.0]);

        // is_tape_active() should be false after with_tape.
        assert!(!is_tape_active());
    }

    #[test]
    fn test_param_grad_returns_zeros_if_no_gradient() {
        let mut tape = Tape::new_empty();
        let p = tape.register_param(&[1.0, 2.0, 3.0], vec![3]);
        // No ops, no backward — gradient should be zeros.
        let grad = tape.get_param_grad(p);
        assert_eq!(grad, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gradient_accumulation_multiple_consumers() {
        // a is consumed by two ops: out1 = a * 2, out2 = a * 3
        // final = out1 + out2
        // d_a = d_final * 2 + d_final * 3 = 5
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0], vec![1]);
        let out1 = tape.alloc(vec![2.0], vec![1]);
        let out2 = tape.alloc(vec![3.0], vec![1]);
        let final_out = tape.alloc(vec![5.0], vec![1]);
        tape.record(TapeOp::Scale { input: a, scalar: 2.0, out: out1 });
        tape.record(TapeOp::Scale { input: a, scalar: 3.0, out: out2 });
        tape.record(TapeOp::Add { a: out1, b: out2, out: final_out });
        tape.seed_grad(final_out, vec![1.0]);
        tape.backward(final_out);
        let grad = tape.get_grad(a).unwrap();
        assert!((grad[0] - 5.0).abs() < 1e-6, "d_a={} expected 5", grad[0]);
    }

    // ── P1.5: Standard op VJP tests ──────────────────────────────────

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
        assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < tol,
                    "{msg}[{i}]: actual={a} expected={e} diff={}",
                    (a - e).abs());
        }
    }

    #[test]
    fn test_backward_sub() {
        // out = a - b, d_a = d_out, d_b = -d_out
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![5.0, 3.0], vec![2]);
        let b = tape.alloc(vec![2.0, 1.0], vec![2]);
        let out = tape.alloc(vec![3.0, 2.0], vec![2]);
        tape.record(TapeOp::Sub { a, b, out });
        tape.seed_grad(out, vec![1.0, 2.0]);
        tape.backward(out);
        assert_close(tape.get_grad(a).unwrap(), &[1.0, 2.0], 1e-6, "d_a");
        assert_close(tape.get_grad(b).unwrap(), &[-1.0, -2.0], 1e-6, "d_b");
    }

    #[test]
    fn test_backward_negate() {
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![3.0, -2.0], vec![2]);
        let out = tape.alloc(vec![-3.0, 2.0], vec![2]);
        tape.record(TapeOp::Negate { input, out });
        tape.seed_grad(out, vec![1.0, 1.0]);
        tape.backward(out);
        assert_close(tape.get_grad(input).unwrap(), &[-1.0, -1.0], 1e-6, "d_input");
    }

    #[test]
    fn test_backward_softplus() {
        // softplus(x) = log(1 + exp(x)), d_x = d_out * sigmoid(x)
        let x_vals = vec![0.0, 1.0, -1.0];
        let out_vals: Vec<f32> = x_vals.iter().map(|&x: &f32| (1.0 + x.exp()).ln()).collect();
        let mut tape = Tape::new_empty();
        let input = tape.alloc(x_vals.clone(), vec![3]);
        let out = tape.alloc(out_vals, vec![3]);
        tape.record(TapeOp::Softplus { input, out });
        tape.seed_grad(out, vec![1.0, 1.0, 1.0]);
        tape.backward(out);
        let expected: Vec<f32> = x_vals.iter()
            .map(|&x| tensor::sigmoid_f32(x)).collect();
        assert_close(tape.get_grad(input).unwrap(), &expected, 1e-5, "softplus_grad");
    }

    #[test]
    fn test_backward_silu() {
        // silu(x) = x * sigmoid(x), d_x = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let x_vals = vec![0.0, 1.0, -1.0, 2.0];
        let out_vals: Vec<f32> = x_vals.iter()
            .map(|&x| x * tensor::sigmoid_f32(x)).collect();
        let mut tape = Tape::new_empty();
        let input = tape.alloc(x_vals.clone(), vec![4]);
        let out = tape.alloc(out_vals, vec![4]);
        tape.record(TapeOp::SiLU { input, out });
        tape.seed_grad(out, vec![1.0; 4]);
        tape.backward(out);
        let expected: Vec<f32> = x_vals.iter().map(|&x| {
            let s = tensor::sigmoid_f32(x);
            s + x * s * (1.0 - s)
        }).collect();
        assert_close(tape.get_grad(input).unwrap(), &expected, 1e-5, "silu_grad");
    }

    #[test]
    fn test_backward_softmax() {
        // 1 row, 3 cols. softmax([1, 2, 3])
        let x = vec![1.0, 2.0, 3.0];
        let max_x = 3.0f32;
        let exps: Vec<f32> = x.iter().map(|v| (v - max_x).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let sm: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let mut tape = Tape::new_empty();
        let input = tape.alloc(x, vec![1, 3]);
        let out = tape.alloc(sm.clone(), vec![1, 3]);
        tape.record(TapeOp::Softmax { input, out, rows: 1, cols: 3 });
        // Seed with [1, 0, 0] → asking d_loss/d_input when loss = softmax[0]
        tape.seed_grad(out, vec![1.0, 0.0, 0.0]);
        tape.backward(out);
        let grad = tape.get_grad(input).unwrap();
        // d_x[i] = sm[i] * (d_out[i] - sum(d_out * sm))
        let dot: f32 = sm[0]; // d_out=[1,0,0], sm[0]*1
        let expected: Vec<f32> = sm.iter().enumerate()
            .map(|(i, &s)| s * (if i == 0 { 1.0 } else { 0.0 } - dot)).collect();
        assert_close(grad, &expected, 1e-5, "softmax_grad");
    }

    #[test]
    fn test_backward_cross_entropy() {
        // 2 tokens, vocab=3. logits=[[1,2,3],[2,1,0]], targets=[2,0]
        let logits = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let targets = vec![2usize, 0];
        // Compute loss manually
        let mut loss = 0.0f32;
        for t in 0..2 {
            let base = t * 3;
            let row = &logits[base..base + 3];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_e: f32 = row.iter().map(|v| (v - max_v).exp()).sum();
            loss -= ((row[targets[t]] - max_v).exp() / sum_e).ln();
        }
        loss /= 2.0; // mean over 2 tokens

        let mut tape = Tape::new_empty();
        let logit_buf = tape.alloc(logits, vec![2, 3]);
        let out = tape.alloc(vec![loss], vec![1]);
        tape.record(TapeOp::CrossEntropy {
            logits: logit_buf, targets: targets.clone(), out, vocab_size: 3,
        });
        tape.backward(out);
        let grad = tape.get_grad(logit_buf).unwrap();

        // Verify via FD
        let logits_orig = vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let eps = 1e-3;
        for idx in 0..6 {
            let mut logits_p = logits_orig.clone();
            logits_p[idx] += eps;
            let mut logits_m = logits_orig.clone();
            logits_m[idx] -= eps;
            let loss_fn = |lg: &[f32]| -> f32 {
                let mut l = 0.0f32;
                for t in 0..2 {
                    let base = t * 3;
                    let row = &lg[base..base + 3];
                    let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum_e: f32 = row.iter().map(|v| (v - max_v).exp()).sum();
                    l -= ((row[targets[t]] - max_v).exp() / sum_e).ln();
                }
                l / 2.0
            };
            let fd = (loss_fn(&logits_p) - loss_fn(&logits_m)) / (2.0 * eps);
            assert!((grad[idx] - fd).abs() < 1e-3,
                    "cross_entropy_grad[{idx}]: tape={} fd={}", grad[idx], fd);
        }
    }

    #[test]
    fn test_backward_embed_lookup() {
        // vocab=3, d=2. indices=[0, 2, 0]. table=[[1,2],[3,4],[5,6]]
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let indices = vec![0usize, 2, 0];
        // out = [[1,2],[5,6],[1,2]]
        let out_data = vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0];
        let mut tape = Tape::new_empty();
        let table_buf = tape.alloc(table, vec![3, 2]);
        let out = tape.alloc(out_data, vec![3, 2]);
        tape.record(TapeOp::EmbedLookup {
            table: table_buf, indices, out, vocab_size: 3, d: 2,
        });
        // d_out = ones → scatter-add
        tape.seed_grad(out, vec![1.0; 6]);
        tape.backward(out);
        let grad = tape.get_grad(table_buf).unwrap();
        // tok 0 appears at t=0 and t=2 → d_table[0] = [2, 2]
        // tok 1 never appears → d_table[1] = [0, 0]
        // tok 2 appears at t=1 → d_table[2] = [1, 1]
        assert_close(grad, &[2.0, 2.0, 0.0, 0.0, 1.0, 1.0], 1e-6, "embed_grad");
    }

    #[test]
    fn test_backward_l2_norm() {
        // x = [3, 4], norm = 5, d_x = d_out * x / norm
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![3.0, 4.0], vec![2]);
        let out = tape.alloc(vec![5.0], vec![1]);
        tape.record(TapeOp::L2Norm { input, out });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        assert_close(tape.get_grad(input).unwrap(), &[0.6, 0.8], 1e-6, "l2norm_grad");
    }

    #[test]
    fn test_backward_l2_norm_zero() {
        // Near-zero input: gradient should be zero (not NaN/Inf)
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![0.0, 0.0], vec![2]);
        let out = tape.alloc(vec![0.0], vec![1]);
        tape.record(TapeOp::L2Norm { input, out });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        let grad = tape.get_grad(input).unwrap();
        assert_close(grad, &[0.0, 0.0], 1e-6, "l2norm_zero_grad");
    }

    #[test]
    fn test_backward_l2_retention() {
        // out = lambda * input, d_input = lambda * d_out
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![1.0, 2.0, 3.0], vec![3]);
        let out = tape.alloc(vec![0.9, 1.8, 2.7], vec![3]);
        tape.record(TapeOp::L2Retention { input, lambda: 0.9, out });
        tape.seed_grad(out, vec![1.0, 1.0, 1.0]);
        tape.backward(out);
        assert_close(tape.get_grad(input).unwrap(), &[0.9, 0.9, 0.9], 1e-6, "l2ret_grad");
    }

    #[test]
    fn test_backward_transpose() {
        // A = [[1,2,3],[4,5,6]] (2x3), out = A^T (3x2)
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = tape.alloc(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]);
        tape.record(TapeOp::Transpose { input, out, rows: 2, cols: 3 });
        // d_out (3x2) = [[1,2],[3,4],[5,6]]
        tape.seed_grad(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        tape.backward(out);
        // d_input = d_out^T (2x3) = [[1,3,5],[2,4,6]]
        assert_close(tape.get_grad(input).unwrap(),
                     &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 1e-6, "transpose_grad");
    }

    #[test]
    fn test_backward_outer_product() {
        // a = [1, 2], b = [3, 4, 5], out = a ⊗ b (2x3)
        // out = [[3,4,5],[6,8,10]]
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0, 2.0], vec![2]);
        let b = tape.alloc(vec![3.0, 4.0, 5.0], vec![3]);
        let out = tape.alloc(vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0], vec![2, 3]);
        tape.record(TapeOp::OuterProduct { a, b, out });
        // d_out = ones(2x3)
        tape.seed_grad(out, vec![1.0; 6]);
        tape.backward(out);
        // d_a[i] = sum_j(b[j]) = 12
        // d_b[j] = sum_i(a[i]) = 3
        assert_close(tape.get_grad(a).unwrap(), &[12.0, 12.0], 1e-6, "d_a");
        assert_close(tape.get_grad(b).unwrap(), &[3.0, 3.0, 3.0], 1e-6, "d_b");
    }

    #[test]
    fn test_backward_frobenius_dot() {
        // out = sum(A * B) (scalar). A=[1,2,3], B=[4,5,6]. out=32
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tape.alloc(vec![4.0, 5.0, 6.0], vec![3]);
        let out = tape.alloc(vec![32.0], vec![1]);
        tape.record(TapeOp::FrobeniusDot { a, b, out });
        tape.seed_grad(out, vec![1.0]);
        tape.backward(out);
        // d_A = d_out * B = B, d_B = d_out * A = A
        assert_close(tape.get_grad(a).unwrap(), &[4.0, 5.0, 6.0], 1e-6, "d_a");
        assert_close(tape.get_grad(b).unwrap(), &[1.0, 2.0, 3.0], 1e-6, "d_b");
    }

    #[test]
    fn test_backward_concat() {
        // Concat [a(2), b(3)] → out(5)
        let mut tape = Tape::new_empty();
        let a = tape.alloc(vec![1.0, 2.0], vec![2]);
        let b = tape.alloc(vec![3.0, 4.0, 5.0], vec![3]);
        let out = tape.alloc(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        tape.record(TapeOp::Concat { inputs: vec![a, b], out, axis: 0, sizes: vec![2, 3] });
        tape.seed_grad(out, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        tape.backward(out);
        assert_close(tape.get_grad(a).unwrap(), &[10.0, 20.0], 1e-6, "d_a");
        assert_close(tape.get_grad(b).unwrap(), &[30.0, 40.0, 50.0], 1e-6, "d_b");
    }

    #[test]
    fn test_backward_slice() {
        // input(5) → slice at offset=1, len=3 → out(3)
        let mut tape = Tape::new_empty();
        let input = tape.alloc(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let out = tape.alloc(vec![2.0, 3.0, 4.0], vec![3]);
        tape.record(TapeOp::Slice { input, out, offset: 1, len: 3, input_len: 5 });
        tape.seed_grad(out, vec![10.0, 20.0, 30.0]);
        tape.backward(out);
        assert_close(tape.get_grad(input).unwrap(),
                     &[0.0, 10.0, 20.0, 30.0, 0.0], 1e-6, "slice_grad");
    }

    // ── P1.6: NL-specific op VJP tests ───────────────────────────────

    #[test]
    fn test_backward_normalized_silu() {
        // NormalizedSiLU: y = silu(x), out = y / ||y||
        // Verify via FD
        let x_vals = vec![1.0, -0.5, 2.0];
        let silu = |x: f32| -> f32 { x * tensor::sigmoid_f32(x) };
        let normalized_silu = |x: &[f32]| -> Vec<f32> {
            let y: Vec<f32> = x.iter().map(|&v| silu(v)).collect();
            let norm = y.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            y.iter().map(|v| v / norm).collect()
        };
        let out_vals = normalized_silu(&x_vals);

        let mut tape = Tape::new_empty();
        let input = tape.alloc(x_vals.clone(), vec![3]);
        let out = tape.alloc(out_vals, vec![3]);
        tape.record(TapeOp::NormalizedSiLU { input, out });
        tape.seed_grad(out, vec![1.0, 0.0, 0.0]); // gradient only on first element
        tape.backward(out);
        let grad = tape.get_grad(input).unwrap();

        // FD check
        let eps = 1e-3;
        // Loss = normalized_silu(x)[0]
        let loss = |x: &[f32]| -> f32 { normalized_silu(x)[0] };
        for i in 0..3 {
            let mut x_p = x_vals.clone();
            x_p[i] += eps;
            let mut x_m = x_vals.clone();
            x_m[i] -= eps;
            let fd = (loss(&x_p) - loss(&x_m)) / (2.0 * eps);
            assert!((grad[i] - fd).abs() < 1e-3,
                    "normalized_silu_grad[{i}]: tape={} fd={}", grad[i], fd);
        }
    }

    #[test]
    fn test_backward_sphere_project_normalize() {
        // 2 slots of dim 2: input = [3, 4, 0, 5]
        // slot0: [3,4] → norm=5 → [0.6, 0.8]
        // slot1: [0,5] → norm=5 → [0.0, 1.0]
        let x = vec![3.0, 4.0, 0.0, 5.0];
        let out_vals = vec![0.6, 0.8, 0.0, 1.0];
        let mut tape = Tape::new_empty();
        let input = tape.alloc(x.clone(), vec![4]);
        let out = tape.alloc(out_vals, vec![4]);
        tape.record(TapeOp::SphereProjectNormalize { input, out, d: 2, m_slots: 2 });
        tape.seed_grad(out, vec![1.0, 0.0, 0.0, 1.0]);
        tape.backward(out);
        let grad = tape.get_grad(input).unwrap();

        // FD check
        let sphere_proj = |x: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; 4];
            for slot in 0..2 {
                let base = slot * 2;
                let norm = (x[base] * x[base] + x[base + 1] * x[base + 1]).sqrt().max(1e-8);
                out[base] = x[base] / norm;
                out[base + 1] = x[base + 1] / norm;
            }
            out
        };
        // loss = sphere_proj(x)[0] + sphere_proj(x)[3]
        let loss = |x: &[f32]| -> f32 {
            let sp = sphere_proj(x);
            sp[0] + sp[3]
        };
        let eps = 1e-3;
        for i in 0..4 {
            let mut x_p = x.clone();
            x_p[i] += eps;
            let mut x_m = x.clone();
            x_m[i] -= eps;
            let fd = (loss(&x_p) - loss(&x_m)) / (2.0 * eps);
            assert!((grad[i] - fd).abs() < 1e-3,
                    "sphere_proj_grad[{i}]: tape={} fd={}", grad[i], fd);
        }
    }

    #[test]
    fn test_backward_kl_retention() {
        // Single-row: z = alpha * log(prior) - theta * input, out = softmax(z)
        let input = vec![0.1, 0.2, 0.3];
        let prior = vec![0.5, 0.3, 0.2];
        let alpha = 1.0f32;
        let theta = 0.5f32;
        // z_j = alpha * ln(prior_j) - theta * input_j
        let z: Vec<f32> = prior.iter().zip(input.iter())
            .map(|(&p, &g): (&f32, &f32)| alpha * p.max(1e-8).ln() - theta * g).collect();
        let max_z = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = z.iter().map(|v| (v - max_z).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out_vals: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let mut tape = Tape::new_empty();
        let input_buf = tape.alloc(input.clone(), vec![3]);
        let prior_buf = tape.alloc(prior.clone(), vec![3]);
        let out_buf = tape.alloc(out_vals, vec![3]);
        tape.record(TapeOp::KLRetention {
            input: input_buf, prior: prior_buf, alpha, theta, out: out_buf,
        });
        tape.seed_grad(out_buf, vec![1.0, 0.0, 0.0]);
        tape.backward(out_buf);
        let grad_input = tape.get_grad(input_buf).unwrap();
        let grad_prior = tape.get_grad(prior_buf).unwrap();

        // FD check
        let kl_ret = |inp: &[f32], pr: &[f32]| -> Vec<f32> {
            let z: Vec<f32> = pr.iter().zip(inp.iter())
                .map(|(&p, &g): (&f32, &f32)| alpha * p.max(1e-8).ln() - theta * g).collect();
            let max_z = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = z.iter().map(|v| (v - max_z).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        };
        let loss = |inp: &[f32], pr: &[f32]| -> f32 { kl_ret(inp, pr)[0] };
        let eps = 1e-3;
        for i in 0..3 {
            let mut inp_p = input.clone(); inp_p[i] += eps;
            let mut inp_m = input.clone(); inp_m[i] -= eps;
            let fd = (loss(&inp_p, &prior) - loss(&inp_m, &prior)) / (2.0 * eps);
            assert!((grad_input[i] - fd).abs() < 1e-3,
                    "kl_ret_d_input[{i}]: tape={} fd={}", grad_input[i], fd);

            let mut pr_p = prior.clone(); pr_p[i] += eps;
            let mut pr_m = prior.clone(); pr_m[i] -= eps;
            let fd = (loss(&input, &pr_p) - loss(&input, &pr_m)) / (2.0 * eps);
            assert!((grad_prior[i] - fd).abs() < 1e-3,
                    "kl_ret_d_prior[{i}]: tape={} fd={}", grad_prior[i], fd);
        }
    }

    #[test]
    fn test_backward_straight_through_bool() {
        // Forward: out = (x > 0.5) as f32. Backward: d_input = d_out (straight-through).
        let x = vec![0.2, 0.7, 0.4, 0.9];
        let out_vals = vec![0.0, 1.0, 0.0, 1.0];
        let mut tape = Tape::new_empty();
        let input = tape.alloc(x, vec![4]);
        let out = tape.alloc(out_vals, vec![4]);
        tape.record(TapeOp::StraightThroughBool { input, threshold: 0.5, out });
        tape.seed_grad(out, vec![1.0, 2.0, 3.0, 4.0]);
        tape.backward(out);
        // Straight-through: gradient passes unchanged
        assert_close(tape.get_grad(input).unwrap(), &[1.0, 2.0, 3.0, 4.0],
                     1e-6, "st_bool_grad");
    }
}
