# Wengert Tape: Reverse-Mode AD via Operation Recording

```
CONTRACT
  Purpose:    Defines Mechanism 1 (automatic differentiation) as a Rust-native
              Wengert tape that records operations during the forward pass and
              replays them in reverse to compute gradients. Replaces Enzyme AD
              (00_enzyme_integration.md) while preserving Mechanism 2 (hand-written
              backward kernels for CUDA/opaque operations). The tape composes with
              kernel pairs via registered opaque VJP blocks — same architectural
              pattern as Enzyme's #[custom_vjp], implemented as a Rust data structure
              instead of an LLVM-level pass.
  Expects:    Forward pass producing a scalar loss.
              Registered opaque VJP functions for all memory rules.
              All tensor operations routed through tape-aware wrappers.
  Guarantees: Correct outer-loop gradients via the chain rule through the actual
              execution path — including through inner-loop memory dynamics.
              Opaque blocks (memory rules, SWA) provide their own VJPs.
              No LLVM/Enzyme toolchain dependency. No train/eval mode distinction (CS-10).
              Opt-in recording (CS-40). No in-place mutation of tracked tensors (CS-47).
              All intermediates stored, no recomputation (CS-42).
  Cost:       Memory: O(ops * buffer_size) for tape storage — bounded by chunk size.
              Time: ~2x forward pass (standard reverse-mode AD overhead).
              Tape recording adds per-op overhead (~50ns/op for arena allocation).
  Trade-off:  Full control over AD semantics vs. Enzyme's zero-overhead LLVM-level AD.
              Tape overhead is measurable but acceptable for the reference Rust path.
              GPU path remains unchanged — hand-written gpu_cms_backward is production.
              Tape serves as correctness oracle for GPU backward, not replacement.
  Position:   specs/infrastructure/differentiation/01_wengert_tape.md
              Supersedes Mechanism 1 in 00_enzyme_integration.md.
              Mechanism 2 (kernel pairs) and the four-annotation-level concept
              from 00_enzyme_integration.md remain valid — adapted to tape registration.
  Source:     Wengert (1964) "A simple automatic derivative evaluation program",
              Griewank & Walther "Evaluating Derivatives" Ch. 3-4,
              PyTorch autograd.Function pattern, JAX custom_vjp pattern,
              CS-40, CS-42, CS-47, CS-10, CS-32
```

## What Changes vs What Stays

```
CHANGES:
  Mechanism 1: Enzyme AD on Rust LLVM IR  →  Wengert tape recording in Rust
    - #[autodiff]      → tape records ops during forward, replays in reverse
    - #[enzyme_opaque] → OpaqueVjp marker trait + registered backward fn
    - #[custom_vjp]    → Tape::record_opaque(key, inputs, outputs, saved)
    - #[no_autodiff]   → simply don't record (opt-in by default, CS-40)
    - enzyme feature flag → removed entirely
    - Enzyme toolchain dependency → removed entirely

STAYS:
  Mechanism 2: Hand-written backward kernels (CUDA)
    - All CUDA kernel pairs unchanged
    - gpu_cms_backward unchanged — remains the production GPU path
    - Kernel correctness verification unchanged (Rust reference = oracle)

  GPU path:
    - GpuMAGParams, GpuMAGGrads, gpu_cms_backward — all unchanged
    - CUDA dispatch layer — unchanged
    - The tape is Rust-reference-path only

  Existing backward code:
    - backward.rs, cms_backward() — DEMOTED to test oracle
    - Class 3 tests verify tape gradients match hand-written exactly
    - After tape is validated, hand-written backward becomes test-only
    - Eventually removed once tape is proven correct across all variants
```

## Tape Architecture

```
struct Tape {
    ops: Vec<TapeOp>,           // Recorded operations in forward order
    bufs: Vec<TapeBuf>,         // Arena of tensor buffers
    grad_accum: Vec<Vec<f32>>,  // Gradient accumulators, indexed by BufId
    recording: bool,            // Whether ops are being recorded (CS-40: opt-in)
}

-- BufId: index into the bufs arena. Immutable after creation.
type BufId = usize;

struct TapeBuf {
    data: Vec<f32>,         // Flat storage (row-major)
    shape: Vec<usize>,      // e.g., [seq_len, d_model]
    is_param: bool,         // True for outer-loop parameters (W_K, W_V, etc.)
}

-- Thread-local tape access. Only one tape per thread.
-- with_tape() is the sole entry point — enforces CS-40 opt-in.
thread_local! {
    static TAPE: RefCell<Option<Tape>> = RefCell::new(None);
}

fn with_tape<F, R>(f: F) -> R
where F: FnOnce(&mut Tape) -> R
{
    // Creates tape, runs f, returns result.
    // Tape is dropped after f completes — no persistent state.
}

-- CS-40 enforcement: if no tape is active, operations execute without recording.
-- This is opt-in by construction — you must call with_tape() to get AD.

-- CS-47 enforcement: registering a parameter SNAPSHOTS its data.
-- The tape clones the buffer at registration time.
-- Subsequent in-place mutations to the original do not affect the tape's copy.
-- All tape ops produce NEW BufIds — no op writes to an existing BufId.

-- CS-42 compliance: all intermediates are stored in the arena.
-- No recomputation during backward. Memory cost is explicit and bounded.
```

## TapeOp Enum

```
enum TapeOp {
    // ── Standard ops (~20) ──────────────────────────────────────────

    // Linear algebra
    Matmul { a: BufId, b: BufId, out: BufId, m: usize, k: usize, n: usize },
    MatmulTransposeB { a: BufId, b: BufId, out: BufId, m: usize, k: usize, n: usize },
    Transpose { input: BufId, out: BufId, rows: usize, cols: usize },
    OuterProduct { a: BufId, b: BufId, out: BufId },
    FrobeniusDot { a: BufId, b: BufId, out: BufId },

    // Element-wise
    Add { a: BufId, b: BufId, out: BufId },
    Sub { a: BufId, b: BufId, out: BufId },
    Mul { a: BufId, b: BufId, out: BufId },
    Scale { input: BufId, scalar: f32, out: BufId },
    Negate { input: BufId, out: BufId },

    // Activations
    Sigmoid { input: BufId, out: BufId },
    Softplus { input: BufId, out: BufId },
    SiLU { input: BufId, out: BufId },

    // Reductions / structured
    Softmax { input: BufId, out: BufId, rows: usize, cols: usize },
    CrossEntropy { logits: BufId, targets: Vec<usize>, out: BufId,
                   vocab_size: usize },
    EmbedLookup { table: BufId, indices: Vec<usize>, out: BufId,
                  vocab_size: usize, d: usize },
    L2Norm { input: BufId, out: BufId },

    // Retention ops
    L2Retention { input: BufId, lambda: f32, out: BufId },

    // Concat / reshape
    Concat { inputs: Vec<BufId>, out: BufId, axis: usize },
    Slice { input: BufId, out: BufId, offset: usize, len: usize },

    // ── NL-specific ops ─────────────────────────────────────────────

    NormalizedSiLU { input: BufId, out: BufId },
    SphereProjectNormalize { input: BufId, out: BufId, d: usize, m_slots: usize },
    KLRetention { input: BufId, prior: BufId, alpha: f32, out: BufId },
    StraightThroughBool { input: BufId, threshold: f32, out: BufId },

    // ── Opaque blocks ───────────────────────────────────────────────

    Opaque {
        key: OpaqueKey,
        inputs: Vec<BufId>,
        outputs: Vec<BufId>,
        saved: Vec<BufId>,       // Tensors saved for backward (e.g., cache)
    },
}
```

## VJP Rules for Standard Ops

Each operation records enough information during the forward pass to compute
the vector-Jacobian product (VJP) during the backward pass. The backward pass
processes ops in reverse order, propagating `d_out` (the upstream gradient of
the loss with respect to this op's output) back to `d_input` (the gradient
with respect to this op's inputs).

```
-- Notation:
--   d_X means d(loss)/d(X)  (the gradient of the scalar loss w.r.t. tensor X)
--   @ means matrix multiply
--   * means element-wise multiply
--   ^T means transpose
--   ⊗ means outer product

MATMUL:  out = A @ B  (A: m×k, B: k×n, out: m×n)
  d_A = d_out @ B^T        -- (m×n) @ (n×k) = (m×k)
  d_B = A^T @ d_out        -- (k×m) @ (m×n) = (k×n)

MATMUL_TRANSPOSE_B:  out = A @ B^T  (A: m×k, B: n×k, out: m×n)
  d_A = d_out @ B           -- (m×n) @ (n×k) = (m×k)
  d_B = d_out^T @ A         -- (n×m) @ (m×k) = (n×k)

TRANSPOSE:  out = A^T  (A: m×k, out: k×m)
  d_A = d_out^T

OUTER_PRODUCT:  out = a ⊗ b  (a: d1, b: d2, out: d1×d2)
  d_a[i] = sum_j(d_out[i,j] * b[j])     -- d_out @ b
  d_b[j] = sum_i(d_out[i,j] * a[i])     -- d_out^T @ a

FROBENIUS_DOT:  out = sum_ij(A[i,j] * B[i,j])  (scalar)
  d_A = d_out * B           -- d_out is scalar, broadcasts
  d_B = d_out * A

ADD:  out = A + B
  d_A = d_out
  d_B = d_out

SUB:  out = A - B
  d_A = d_out
  d_B = -d_out

MUL:  out = A * B  (element-wise)
  d_A = d_out * B
  d_B = d_out * A

SCALE:  out = scalar * A
  d_A = scalar * d_out
  -- scalar is a constant, no gradient

NEGATE:  out = -A
  d_A = -d_out

SIGMOID:  out = σ(x) = 1 / (1 + exp(-x))
  d_x = d_out * out * (1 - out)
  -- Uses saved output, not input (numerically stable)

SOFTPLUS:  out = log(1 + exp(x))
  d_x = d_out * σ(x)
  -- σ(x) = 1 / (1 + exp(-x)), recoverable from out: σ(x) = 1 - exp(-out)

SILU:  out = x * σ(x)
  d_x = d_out * (σ(x) + x * σ(x) * (1 - σ(x)))
  -- Equivalent: d_x = d_out * (out + σ(x) * (1 - out))
  -- Requires saving both x and σ(x)

SOFTMAX:  out[i] = exp(x[i]) / sum_j(exp(x[j]))  (per-row)
  d_x[i] = out[i] * (d_out[i] - sum_j(d_out[j] * out[j]))
  -- Standard softmax VJP. Per-row, applied to each row independently.

CROSS_ENTROPY:  out = -mean_t(log(softmax(logits)[target_t]))
  d_logits[t, j] = (softmax(logits[t])[j] - 1_{j == target_t}) / n_valid
  -- Combined softmax + cross-entropy for numerical stability.

EMBED_LOOKUP:  out[t] = table[indices[t]]
  d_table[tok] += d_out[t]  for each t where indices[t] == tok
  -- Scatter-add: multiple tokens may map to the same embedding row.

L2_NORM:  out = ||x||_2 = sqrt(sum(x[i]^2))
  d_x[i] = d_out * x[i] / out
  -- Undefined at x=0; clamp denominator to eps=1e-8.

L2_RETENTION:  out = lambda * input
  d_input = lambda * d_out
  -- lambda is a constant gate parameter, not a tracked variable here.
  -- When lambda itself is learnable, it flows through the gate op that produces it.

CONCAT:  out = concat(A, B, ..., axis=a)
  d_A = d_out[slice for A along axis a]
  d_B = d_out[slice for B along axis a]
  -- Each input gets its corresponding slice of the upstream gradient.

SLICE:  out = input[offset..offset+len]
  d_input = zeros; d_input[offset..offset+len] = d_out
  -- Gradient is zero-padded to input size, nonzero only in the sliced region.
```

## NL-Specific Op VJPs

```
NORMALIZED_SILU (Trellis):
  -- Forward: y = x * σ(x), norm = ||y||, out = y / max(norm, eps)
  -- VJP: Chain through normalization, then through SiLU.
  --   Let s = σ(x), y = x * s
  --   d_y = (d_out - out * dot(d_out, out)) / max(norm, eps)   -- normalize VJP
  --   d_x = d_y * (s + x * s * (1 - s))                       -- SiLU VJP
  -- Source: Trellis (2512.23852) Eq. 7

SPHERE_PROJECT_NORMALIZE (Lattice OSR):
  -- Forward: For each of m slots of dimension d:
  --   S[i] = S[i] / max(||S[i]||, eps)   (project to unit sphere)
  -- VJP: Standard unit normalization Jacobian per slot:
  --   d_S[i] = (d_out[i] - S_norm[i] * dot(d_out[i], S_norm[i])) / max(||S[i]||, eps)
  --   where S_norm[i] = S[i] / ||S[i]||
  -- Source: Lattice (2504.05646) orthogonal projection constraint

KL_RETENTION (MEMORA):
  -- Forward: out[row] = softmax(alpha * log(prior[row]) - theta * grad[row])
  --   where log is element-wise, clamped to log(max(x, 1e-8))
  -- VJP: Chain through softmax, then through the linear combination.
  --   Let z[row] = alpha * log(prior[row]) - theta * grad[row]
  --   d_z = softmax_vjp(d_out, out)      -- standard softmax VJP
  --   d_prior[row, j] = d_z[row, j] * alpha / max(prior[row, j], 1e-8)
  --   d_grad[row, j] = -d_z[row, j] * theta
  -- Source: MEMORA rule (MIRAS 2504.13173) Eq. for KL-regularized update

STRAIGHT_THROUGH_BOOL (frequency gate):
  -- Forward: out = (input > threshold) as f32   (hard threshold, 0 or 1)
  -- VJP: d_input = d_out                        (straight-through estimator)
  -- Gradient passes through unchanged as if the threshold were identity.
  -- Source: Bengio et al. (2013) straight-through estimator;
  --         used for learned frequency gating in CMS
```

## Opaque VJP Registration

Opaque blocks wrap existing `step_backward()` methods from each memory rule.
The tape records inputs/outputs/saved tensors during forward and calls the
registered backward function during reverse-mode replay.

```
-- All memory rules + SWA + frozen read-only variants.
enum OpaqueKey {
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

-- Type signature for opaque backward functions.
-- Takes: upstream gradient (d_outputs), saved tensors from forward.
-- Returns: gradients w.r.t. each input (d_inputs).
type OpaqueBackwardFn = fn(
    d_outputs: &[&[f32]],       // upstream gradient for each output
    saved: &[&[f32]],           // tensors saved during forward
    d_inputs: &mut [Vec<f32>],  // output: gradient for each input
);

-- Registry: maps OpaqueKey → backward function.
-- Populated at initialization from existing step_backward() methods.
fn register_opaque_vjps() -> HashMap<OpaqueKey, OpaqueBackwardFn> {
    let mut registry = HashMap::new();

    registry.insert(OpaqueKey::DeltaRule, delta_rule_opaque_backward);
    registry.insert(OpaqueKey::TitansLMM, titans_lmm_opaque_backward);
    registry.insert(OpaqueKey::HebbianRule, hebbian_opaque_backward);
    registry.insert(OpaqueKey::Moneta, moneta_opaque_backward);
    registry.insert(OpaqueKey::YAAD, yaad_opaque_backward);
    registry.insert(OpaqueKey::MEMORA, memora_opaque_backward);
    registry.insert(OpaqueKey::LatticeOSR, lattice_osr_opaque_backward);
    registry.insert(OpaqueKey::Trellis, trellis_opaque_backward);
    registry.insert(OpaqueKey::AtlasOmega, atlas_omega_opaque_backward);
    registry.insert(OpaqueKey::SWA, swa_opaque_backward);

    // Frozen variants: read-only backward (no write gradients)
    registry.insert(OpaqueKey::FrozenDeltaRule, frozen_delta_rule_backward);
    // ... (one per rule)

    registry
}
```

### Example Adapter: DeltaRule

```
-- Wraps DeltaRule::step_backward() into the opaque VJP interface.
-- The adapter translates between tape buffer layout and the existing
-- step_backward() signature.

fn delta_rule_opaque_backward(
    d_outputs: &[&[f32]],       // [d_y: &[f32]]
    saved: &[&[f32]],           // [embedded, level_params_flat, cache_flat]
    d_inputs: &mut [Vec<f32>],  // [d_embedded, d_level_params]
) {
    // Reconstruct the MemoryLevelParams and Cache from saved flat buffers.
    let level_params = MemoryLevelParams::from_flat(saved[1]);
    let cache = DeltaRuleCache::from_flat(saved[2]);
    let d_y = d_outputs[0];
    let embedded = saved[0];

    // Call the existing hand-written backward.
    let (param_grads, d_embedded) = DeltaRule::new(/* config */)
        .step_backward(&level_params, &cache, d_y, embedded);

    // Write results into the tape's gradient accumulators.
    d_inputs[0] = d_embedded;
    d_inputs[1] = param_grads.to_flat();
}

-- Every other rule follows the same pattern:
--   1. Reconstruct typed structures from flat saved buffers
--   2. Call existing step_backward()
--   3. Flatten results back into d_inputs
-- No new gradient math — the existing backward code IS the VJP.
```

### Frozen Read-Only Backward

```
-- Frozen levels do M @ q_t only (no memory write).
-- Gradient flows through the read path: d_q = M^T @ d_out.
-- No gradient to M itself (frozen = not updated this step).
-- But d_q accumulates into the ErrorBuffer for deferred application
-- when the level becomes active.

fn frozen_delta_rule_backward(
    d_outputs: &[&[f32]],
    saved: &[&[f32]],           // [M_frozen, q_t]
    d_inputs: &mut [Vec<f32>],
) {
    let m = saved[0];  // d×d frozen memory matrix
    let d_y = d_outputs[0];

    // d_q = M^T @ d_y (gradient through read path)
    let d_q = matmul_transpose_a(m, d_y, d, d);
    d_inputs[0] = d_q;

    // No gradient to M — frozen levels don't update memory this step.
    // The upstream gradient (d_y) is accumulated into the ErrorBuffer
    // by the caller (cms_backward), not by this VJP function.
}
```

## Trait System Changes

```
-- RENAME: EnzymeOpaque → OpaqueVjp
-- The marker trait now signals that the type provides an opaque VJP
-- via the tape registry, not via Enzyme's #[enzyme_opaque] attribute.

trait OpaqueVjp {
    /// The key used to look up this rule's backward function in the registry.
    fn opaque_key(&self) -> OpaqueKey;

    /// Record this rule's forward pass on the tape.
    /// Called during traced forward when tape is active.
    /// Must:
    ///   1. Snapshot all inputs as TapeBufs (CS-47)
    ///   2. Execute the forward pass (same code as untraced)
    ///   3. Record outputs as TapeBufs
    ///   4. Save any tensors needed by step_backward() (the cache)
    ///   5. Push a TapeOp::Opaque onto the tape
    fn record_on_tape(
        &self,
        tape: &mut Tape,
        input_bufs: &[BufId],
        level_params: &MemoryLevelParams,
        state: &mut MemoryState,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
    ) -> Vec<BufId>;  // output BufIds
}

-- MemoryRule trait gains OpaqueVjp as a supertrait.
trait MemoryRule: OpaqueVjp {
    type Cache;
    fn level(&self) -> usize;
    fn supported_parallelization(&self) -> &'static [&'static str];
    fn init(&self, d: usize) -> MemoryState;
    fn write(&self, ...) -> Result<(), MemoryError>;
    fn read(&self, ...) -> Vec<f32>;
    fn step(&self, ...) -> (Vec<f32>, Self::Cache);
    fn step_backward(&self, ...) -> (MemoryLevelParams, Vec<f32>);
}

-- REMOVED: enzyme feature flag.
-- The tape is always available. It activates only when with_tape() is called.
-- No compile-time feature gate needed — the tape is opt-in at runtime (CS-40).
```

## Gradient Flow: Updated Diagrams

### MAG Composition (Single CMS Level, Active)

```
Forward (tape recording):

  input_ids
      |
      v
  [EmbedLookup] ─────────────────────────── TapeOp::EmbedLookup
      |
      v
  embedded (BufId:0)
      |
      ├──> [Matmul W_Q^T] → q ────────────── TapeOp::MatmulTransposeB
      ├──> [Matmul W_K^T] → k ────────────── TapeOp::MatmulTransposeB
      ├──> [Matmul W_V^T] → v ────────────── TapeOp::MatmulTransposeB
      |
      ├──> [Memory Branch: Opaque] ─────────── TapeOp::Opaque(DeltaRule)
      |         |
      |         v
      |     mem_out (BufId:4)
      |         |
      |         v
      |     [Sigmoid] → gate ──────────────── TapeOp::Sigmoid
      |
      └──> [SWA Branch: Opaque] ────────────── TapeOp::Opaque(SWA)
                |
                v
            attn_out (BufId:6)
                |
                v
           [Mul gate * attn_out] → combined ── TapeOp::Mul
                |
                v
           [Matmul W_O^T] → projected ─────── TapeOp::MatmulTransposeB
                |
                v
           [Matmul W_unembed^T] → logits ──── TapeOp::MatmulTransposeB
                |
                v
           [CrossEntropy] → loss ──────────── TapeOp::CrossEntropy


Backward (tape reverse replay):

  d_loss = 1.0
      |
      v
  CrossEntropy VJP → d_logits
      |
      v
  MatmulTransposeB VJP → d_projected, d_W_unembed
      |
      v
  MatmulTransposeB VJP → d_combined, d_W_O
      |
      v
  Mul VJP → d_attn_out = d_combined * gate
          → d_gate = d_combined * attn_out
      |
      ├──> SWA Opaque VJP → d_q, d_k, d_v          (swa_backward_dispatch)
      |
      ├──> Sigmoid VJP → d_mem_out = d_gate * gate * (1 - gate)
      |         |
      |         v
      |    DeltaRule Opaque VJP → d_embedded_mem, d_level_params
      |                                                (step_backward)
      v
  MatmulTransposeB VJPs → d_embedded += d_q @ W_Q + d_k @ W_K + d_v @ W_V
                        → d_W_Q, d_W_K, d_W_V
      |
      v
  d_embedded += d_embedded_mem      (accumulate memory branch gradient)
      |
      v
  EmbedLookup VJP → d_W_embed (scatter-add)
```

### CMS Multi-Level (k=4, Mixed Active/Frozen)

```
-- At step t, pulse.active_levels = [true, false, true, false]
-- Level 0: ACTIVE  → full forward + backward (TapeOp::Opaque with active key)
-- Level 1: FROZEN  → read-only M @ q_t (TapeOp::Opaque with frozen key)
-- Level 2: ACTIVE  → full forward + backward
-- Level 3: FROZEN  → read-only M @ q_t

Forward:
  embedded
      |
      ├──> Level 0: Opaque(DeltaRule)     → y_0   (write + read)
      ├──> Level 1: Opaque(FrozenDelta)   → y_1   (read-only from persisted M)
      ├──> Level 2: Opaque(DeltaRule)     → y_2   (write + read)
      ├──> Level 3: Opaque(FrozenDelta)   → y_3   (read-only from persisted M)
      |
      v
  [Add y_0 + y_1 + y_2 + y_3] → combined_mem ──── TapeOp::Add (4-way)
      |
      v
  [Scale 1/sqrt(k)] → mem_out ─────────────────── TapeOp::Scale (k>2 normalization)
      |
      v
  [Sigmoid] → gate ────────────────────────────── TapeOp::Sigmoid
      ...continues as above...

Backward:
  d_mem_out flows backward through Scale, Add, into each level:
  - Level 0: DeltaRule backward → d_embedded_0, d_level_params_0
  - Level 1: FrozenDelta backward → d_q_1 (accumulated into ErrorBuffer)
  - Level 2: DeltaRule backward → d_embedded_2, d_level_params_2
  - Level 3: FrozenDelta backward → d_q_3 (accumulated into ErrorBuffer)

  Error buffers accumulate frozen-level gradients.
  When level 1 next becomes active (step t+8), its error buffer is applied.
```

### Dynamic Frequency Gate

```
-- When cfg.frequency_schedule == Learned, the active_levels are
-- determined by a learnable gate, not the static pulse.

  embedded
      |
      v
  [Matmul W_freq] → freq_logits ──────────── TapeOp::MatmulTransposeB
      |
      v
  [StraightThroughBool] → active_mask ────── TapeOp::StraightThroughBool
      |
      v
  -- active_mask is {0, 1} per level
  -- Forward: use hard threshold
  -- Backward: straight-through estimator passes d_out unchanged to d_freq_logits
  -- This allows the outer loop to learn which levels should be active
```

## CMS Integration

```
-- The Conductor creates the tape at the start of each chunk during Build phase.
-- During Test/Stream phases, no tape is created — forward-only (CS-10).
-- The tape lives for exactly one chunk: create → forward → backward → drop.

fn build_chunk(
    conductor: &mut Conductor,
    params: &MAGParams,
    cfg: &MAGConfig,
    chunk: &TokenChunk,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
) -> MAGParams {
    let pulse = conductor.pulse();  // CS-32: observe before advance

    // Create tape (Build phase only)
    let grads = with_tape(|tape| {
        // Register outer-loop params on tape (snapshots, CS-47)
        let param_bufs = tape.register_params(params);

        // Traced forward: same dispatch, but records ops
        let loss_buf = traced_cms_forward(
            tape, params, cfg, &chunk.input_ids, &chunk.target_ids,
            &pulse, context, error_buffers,
        );

        // Seed backward with d_loss = 1.0
        tape.backward(loss_buf);

        // Extract gradients from tape accumulators
        tape.extract_param_grads(param_bufs)
    });

    conductor.advance();  // CS-32: advance AFTER
    grads
}

-- Key invariant: traced_cms_forward() calls the SAME dispatch layer as
-- cms_forward(). It wraps each operation in tape recording, but the
-- actual computation is identical. If the tape is not active (e.g.,
-- test/stream phase), the wrappers are no-ops.
```

## Testing Strategy

### Test Class 1: Tape Isolation (Opaque Block Integrity)

```
-- Verifies that the opaque block is the SOLE gradient source through its region.
-- No gradient leaks from the tape through the opaque boundary.
-- Replaces the Enzyme barrier tests from 00_enzyme_integration.md.

fn test_tape_isolation<R: MemoryRule>() {
    let rule = R::default();
    let (params, embedded, d) = random_test_inputs();

    // Forward with tape
    let grads = with_tape(|tape| {
        let embedded_buf = tape.register_input(&embedded);

        // Record memory rule as opaque block
        let out_bufs = rule.record_on_tape(tape, &[embedded_buf], &params, ...);

        // Seed backward
        tape.set_grad(out_bufs[0], &ones_like(out_bufs[0]));
        tape.backward(out_bufs[0]);

        tape.get_grad(embedded_buf)
    });

    // Compare: gradient must come ONLY from the registered backward fn.
    // Run step_backward() directly to get the expected gradient.
    let (_, expected_d_embedded) = rule.step_backward(&params, &cache, &d_y, &embedded);

    assert_close!(grads, expected_d_embedded, rtol=1e-6, atol=1e-8,
        "Tape produced different gradient than registered backward — \
         either the tape leaked through the opaque boundary or the \
         adapter is wrong");
}

-- Run for: DeltaRule, TitansLMM, HebbianRule, Moneta, YAAD, MEMORA,
--          LatticeOSR, Trellis, AtlasOmega, SWA
-- Plus frozen variants: FrozenDeltaRule, etc.
```

### Test Class 2: Analytical Correctness (Tape vs Finite Differences)

```
-- Verifies that the tape's end-to-end gradient matches numerical ground truth.
-- Uses the EXISTING finite-difference infrastructure from gradient.rs.

fn test_tape_fd_correctness() {
    let cfg = SmallConfig { d: 8, heads: 2, vocab: 16, seq_len: 4, k: 1 };
    let params = MAGParams::random(&cfg);
    let (input_ids, target_ids) = random_token_pair(cfg.seq_len, cfg.vocab);

    // (a) Tape gradient
    let tape_grads = with_tape(|tape| {
        let loss_buf = traced_cms_forward(tape, &params, &cfg, ...);
        tape.backward(loss_buf);
        tape.extract_param_grads(...)
    });

    // (b) Finite differences (numerical ground truth)
    let fd_grads = fd_compute_gradients(&params, &cfg, &input_ids, &target_ids,
                                         eps=1e-2);

    // Must match within FD tolerance
    assert_gradient_match!(tape_grads, fd_grads,
        rel_tol=0.10, abs_threshold=5e-4,
        "Tape gradient does not match finite differences");
}

-- Uses existing FD infrastructure: central differences, eps=1e-2,
-- abs_threshold=5e-4 for f32 precision (see memory notes).
-- Tests: W_K, W_V, W_Q, W_O, W_embed, W_unembed, per-level params.
```

### Test Class 3: Integration (Tape vs Hand-Written Backward)

```
-- Verifies that the tape produces IDENTICAL gradients to the existing
-- hand-written cms_backward(). This is the critical validation that
-- the tape correctly composes all VJPs through the actual execution path.

fn test_tape_matches_handwritten() {
    let cfg = MAGConfig { k: 2, ..test_config() };
    let params = MAGParams::random(&cfg);
    let (input_ids, target_ids) = random_tokens(&cfg);

    // (a) Hand-written backward (existing code)
    let (loss_hw, cache) = cms_forward(&params, &cfg, &input_ids, &target_ids,
                                        &pulse, &mut context);
    let hw_grads = cms_backward(&params, &cfg, &cache, &input_ids, &target_ids,
                                 &mut error_buffers);

    // (b) Tape backward (new code)
    let tape_grads = with_tape(|tape| {
        let loss_buf = traced_cms_forward(tape, &params, &cfg, &input_ids,
                                           &target_ids, &pulse, &mut context2,
                                           &mut error_buffers2);
        tape.backward(loss_buf);
        tape.extract_param_grads(...)
    });

    // Must match EXACTLY (same computation, same VJP functions)
    assert_close!(tape_grads, hw_grads, rtol=1e-6, atol=1e-8,
        "Tape gradient differs from hand-written backward — \
         chain-rule composition bug in tape replay");
}

-- Run for: k=1 (single level), k=2 (two levels, mixed active/frozen),
--          k=4 (full CMS with all level combinations).
-- Run for: all 8 memory rule variants.
-- Run for: MAG composition (default) — MAL and MAC are future.
```

## Migration Path

```
Phase 1: tape.rs core + VJP rules + opaque registrations + unit tests
  -- New files: core/src/tape.rs
  -- Implements: Tape, TapeOp, TapeBuf, BufId, with_tape()
  -- Implements: VJP rules for all ~20 standard ops
  -- Implements: OpaqueKey, OpaqueBackwardFn, register_opaque_vjps()
  -- Implements: Adapter wrappers for all 8 rules + SWA + frozen variants
  -- Tests: Class 1 (isolation) for all opaque blocks
  -- Tests: Class 2 (FD correctness) for individual standard ops
  -- Gate: All Class 1 + Class 2 tests pass

Phase 2: traced_forward.rs wrappers (tape-aware, record + execute)
  -- New files: core/src/traced_forward.rs
  -- Implements: tape-aware wrappers around each forward operation
  --   traced_matmul(), traced_sigmoid(), traced_softmax(), etc.
  -- Implements: traced_embed_lookup(), traced_cross_entropy()
  -- Implements: traced_cms_forward() — mirrors cms_forward() structure,
  --   calls traced_* wrappers that record on tape when active
  -- Tests: traced_cms_forward() produces same loss as cms_forward()
  -- Gate: Forward outputs are bitwise identical with and without tape

Phase 3: traced_cms_forward full integration + Class 3 tests pass
  -- Modify: core/src/gradient.rs — add tape_compute_gradients()
  -- Implements: full backward through traced_cms_forward via tape
  -- Tests: Class 3 (tape vs hand-written) for all 8 rules × k=1,2,4
  -- Tests: Error buffer accumulation matches for frozen levels
  -- Tests: Dynamic frequency gate gradient flows correctly
  -- Gate: All Class 3 tests pass for every variant

Phase 4: Switchover
  -- Modify: core/src/gradient.rs — mag_compute_gradients() calls tape path
  -- The tape becomes the PRODUCTION gradient computation path
  -- cms_backward() is still called in test oracle mode
  -- Tests: Full build loop (forward + tape backward + optimizer step)
  --         produces same loss trajectory as current code
  -- Gate: 10-step build on tiny model matches current code within f32 tolerance

Phase 5: Cleanup
  -- Remove: enzyme feature flag from Cargo.toml and all #[cfg(feature = "enzyme")]
  -- Demote: backward.rs → test infrastructure only (not called in production path)
  -- Demote: cms_backward() → called only by Class 3 tests
  -- Update: CLAUDE.md toolchain notes (Enzyme dependency removed)
  -- Update: contract.md "Differentiation" line
  -- Gate: cargo test --all-features passes, no enzyme references in production code
```

## Relationship to Existing Specs

```
00_enzyme_integration.md:
  -- Mechanism 1 SUPERSEDED by this spec (Enzyme → tape)
  -- Mechanism 2 UNCHANGED (kernel pairs, hand-written backward)
  -- Four annotation levels ADAPTED:
       #[autodiff]      → tape recording
       #[enzyme_opaque] → OpaqueVjp trait
       #[custom_vjp]    → Tape::record_opaque()
       #[no_autodiff]   → no recording (default)
  -- Barrier verification tests ADAPTED to tape isolation tests (Class 1)
  -- Integration gradient test PRESERVED as Class 3

00_state_ownership.md:
  -- outer_loop_param: tape snapshots at registration (CS-47)
  -- inner_loop_state: lives inside opaque blocks, not on tape
  -- context_memory: passed through to opaque blocks, not tracked by tape

00_conductor.md:
  -- Conductor creates tape for Build phase chunks only
  -- CS-32 observe-then-advance unchanged
  -- Pulse drives active/frozen level decisions → opaque key selection

00_numerical_precision.md:
  -- Tape operates in fp32 unconditionally (inner loop fp32 requirement)
  -- No bf16 in tape buffers — all accumulators are fp32
```
