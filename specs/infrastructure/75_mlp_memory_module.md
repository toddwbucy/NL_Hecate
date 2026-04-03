# 75 — MLP Memory Module (Deep Neural Memory)

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Replace linear matrix memory M ∈ ℝ^{d×d} with a configurable multi-layer MLP memory M = {W₁, b₁, …, Wₗ, bₗ}, enabling nonlinear associative storage as described in the Titans and MIRAS papers |
| Expects   | Working linear memory path (Titans LMM, Delta Rule) with per-token M-norm projection (spec 74); cuBLAS batched GEMM infrastructure from Phase 1 backward (spec 44). Note: MIRAS MLP rules (MONETA/YAAD/MEMORA) live in their own files and have separate GPU completion tracked by spec 77 / task_edfe4c. This spec focuses exclusively on extending TitansLMM from linear M to MLP M for MAG/MAC/MAL compositions. |
| Guarantees | `memory_layers=1` reproduces current linear M behavior bit-exactly; `memory_layers≥2` implements the deep neural memory from Titans §3.1 and MIRAS §4; all CUDA kernels match CPU reference within 1e-4; backward gradients verified via finite-difference; momentum and retention gates apply per-weight-matrix |
| Cost      | Per token: O(d · d_h + d_h · d) per MLP layer for forward + analytical backward (vs O(d²) for linear). Memory: O(L_M · d · d_h) for weights + momentum per level per head |
| Trade-off | Higher capacity per parameter (MLP can learn nonlinear associations — MIRAS §4.2) vs higher per-token compute. Nonlinearity breaks associative scan parallelization (chunkwise GD still works). 8× more memory state at expansion_factor=4 |
| Position  | Implements the "Memory Structure" dimension of the MIRAS 4-knob framework (CS-34). Supersedes the matrix-only assumption in `01_titans_lmm.md` for L_M ≥ 2. Child of `memory_update_rules/00_interface.md` |
| Source    | Titans (2501.00663) §3.1 Eqs 8-15, §5.5 deep memory ablation; TNT (2511.07343) §2.1 Eqs 1-2 (general f(W,k) formulation); MIRAS (2504.13173) Eqs 24-27, Table 1 (MONETA/YAAD/MEMORA MLP specs) |
| Traced equations | titans_equations/eq-012-associative-memory-loss, titans_equations/eq-013-forgetting-gate, titans_equations/eq-015-memory-retrieval, tnt_equations/eq-001-memory-compression, tnt_equations/eq-002-memory-retrieval, miras_equations/eq-024-025-moneta-spec |

---

## Problem

### What the Papers Say

The Titans paper (2501.00663) defines the neural memory module as:

> "A deep neural network whose parameters ARE the memory, trained at test time via gradient descent on associative memory loss."
> — titans_abstractions/neural-memory-module-complete

All equations use **function notation** `M(k_t)`, not matrix notation `M @ k_t`:

- **Eq 12** (Loss): `ℓ(M; x_t) = ||M(k_t) - v_t||₂²`
- **Eq 15** (Read): `y_t = M*(q_t)` — forward pass through M without weight update
- **Eq 8** (Update): `M_t = M_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)` — gradient of loss w.r.t. M's parameters

TNT (2511.07343) makes this explicit with sub-network notation:

- **Eq 1**: `W_t ← W_{t-1} - η_t ∇_W L(f(W_{t-1}, k_t), v_t)` — W are weights of sub-network `f`
- **Eq 2**: `o_t = f(W_t, q_t)` — retrieval is a forward pass through `f`

MIRAS (2504.13173) defines three named variants that all use 2-layer MLP memory:

- **MONETA** (Eq 24-25): 2-layer MLP, expansion factor 4, GELU
- **YAAD** (Eq 26): Same MLP as MONETA, Huber loss
- **MEMORA** (Eq 27): Same MLP, KL retention

The MIRAS framework explicitly lists Memory Structure options as: **vector, linear, MLP, complex** (miras_definitions/def-four-design-choices). Our current implementation only supports "linear" (matrix M).

Titans §5.5 ablation shows L_M=2 outperforms L_M=1 at the same parameter budget. The paper's production model uses L_M ≥ 2.

### What We Have

Our current implementation (`titans_lmm.rs`, `gpu_forward.rs`) uses a single d×d matrix:

```
M ∈ ℝ^{d×d}
Forward:    y = M @ q           (matrix-vector multiply)
Error:      e = M @ k - v       (matrix-vector multiply)
Gradient:   ∇M = outer(e, k)    (rank-1 outer product)
Momentum:   S = η·S - θ·∇M     (element-wise on d×d matrix)
Retention:  M = (1-α)·M + S    (element-wise on d×d matrix)
```

This is Titans Eq 32 — the **linear specialization** of the general framework. It cannot learn nonlinear key-value associations, and its capacity scales as O(min(d_in, d_out)) (MIRAS §4.2).

### The Gap

Without MLP memory, our model implements only the simplest case of the Titans memory module. The papers explicitly position the linear matrix as a baseline, not the target architecture. As stated in the Titans paper: the neural memory is a "deep MLP" — without this, the model is not architecturally a Titans model.

---

## Architecture

### MLP Memory Structure

For `memory_layers = L_M` (default 2), the memory M is an MLP with L_M layers:

```
M = {W₁, b₁, W₂, b₂, …, W_L, b_L}

Layer dimensions (expansion_factor = 4, matching MONETA):
  W₁ ∈ ℝ^{d_h × d}      b₁ ∈ ℝ^{d_h}       (expand)
  W₂ ∈ ℝ^{d × d_h}      b₂ ∈ ℝ^{d}         (project back)

  where d_h = expansion_factor × d = 4d
  and d = head_dim (d_model / num_heads)
```

For `L_M = 2` (the standard case):
```
M(x) = W₂ @ σ(W₁ @ x + b₁) + b₂
```

For `L_M = 3`:
```
M(x) = W₃ @ σ(W₂ @ σ(W₁ @ x + b₁) + b₂) + b₃

  W₁ ∈ ℝ^{d_h × d}      (expand)
  W₂ ∈ ℝ^{d_h × d_h}    (hidden-to-hidden)
  W₃ ∈ ℝ^{d × d_h}      (project back)
```

General pattern for L_M ≥ 2:
- Layer 1: d → d_h (expand)
- Layers 2 through L_M−1: d_h → d_h (hidden)
- Layer L_M: d_h → d (project back)
- Activation σ after every layer except the last

### Activation Function

**GELU** (default, matching MONETA/YAAD/MEMORA in MIRAS):
```
σ(x) = x · Φ(x)    where Φ is the standard normal CDF
σ'(x) = Φ(x) + x · φ(x)   where φ is the standard normal PDF
```

Approximation (matching PyTorch):
```
σ(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Config option `memory_activation`: `gelu` (default) | `silu` | `relu`

### Initialization

MLP weights are **outer_loop_params** (learned via AdamW, serialized in checkpoints):
- `W_init` values: Xavier uniform, matching the paper's learnable `W_init` that local memories reset to (TNT Eq 6)
- Biases initialized to zero
- At the start of each forward pass, MLP weights are copied to inner_loop_state and then updated per-token by the inner loop
- When `memory_reset = "learned"` (TNT §4.1.1), memories reset to `W_init` at shard boundaries

### Parameter Count

For d=128 (hd=128), L_M=2, expansion_factor=4:

| Component | Shape | Count |
|-----------|-------|-------|
| W₁ | (512, 128) | 65,536 |
| b₁ | (512,) | 512 |
| W₂ | (128, 512) | 65,536 |
| b₂ | (128,) | 128 |
| **Total per head per level** | | **131,712** |
| Linear M (current) | (128, 128) | 16,384 |
| **Ratio** | | **8.04×** |

For 12 heads, 12 blocks, k=1: 131,712 × 12 × 12 = **18.97M** additional params (vs 2.36M for linear M).

### Backward Compatibility: `memory_layers = 1`

When `memory_layers = 1`, the MLP degenerates to a single linear layer:
```
M(x) = W₁ @ x + b₁     where W₁ ∈ ℝ^{d×d}, b₁ ∈ ℝ^{d}
```

With b₁ initialized to zero, this is equivalent to the current `M @ x` behavior. The bias adds d parameters per head per level — negligible. The forward/backward paths must produce bit-identical results to the current linear implementation when `memory_layers = 1`.

---

## Forward Pass

### Per-Token Forward (Sequential Form)

Source: Titans Eq 12 + 15 + MONETA spec (MIRAS Eq 24-25)

```
ALGORITHM: mlp_memory_step(mlp: &mut MLPMemory, k_t, v_t, q_t, gates) -> y_t
  -- gates: alpha_t (retention), theta_t (learning rate), eta_t (momentum)

  -- 1. Compute error: forward k through MLP, compare to v
  h_k = [k_t]                              -- activations list
  FOR l in 1..L_M:
    pre_act = W_l @ h_k[l-1] + b_l         -- linear transform
    IF l < L_M:
      h_k.push(σ(pre_act))                 -- activation (except last layer)
    ELSE:
      h_k.push(pre_act)                    -- last layer: no activation
  prediction = h_k[L_M]                     -- M(k_t)
  error = prediction - v_t                  -- surprise signal (Titans Eq 12)

  -- 2. Analytical backward through MLP (gradient of ||M(k) - v||²)
  --    Chain rule: ∂L/∂W_l = ∂L/∂h_l · ∂h_l/∂W_l
  d_out = 2.0 * error                       -- ∂L/∂prediction for L2 loss
  FOR l in L_M..1 (reverse):
    IF l < L_M:
      d_out = d_out ⊙ σ'(W_l @ h_k[l-1] + b_l)   -- activation derivative
    grad_W_l = outer(d_out, h_k[l-1])       -- ∂L/∂W_l
    grad_b_l = d_out                         -- ∂L/∂b_l
    IF l > 1:
      d_out = W_l^T @ d_out                 -- propagate to previous layer

  -- 3. Momentum update (Titans Eq 10, per weight matrix)
  FOR l in 1..L_M:
    S_W_l = eta_t · S_W_l - theta_t · grad_W_l
    S_b_l = eta_t · S_b_l - theta_t · grad_b_l

  -- 4. Retention + momentum (Titans Eq 13, per weight matrix)
  FOR l in 1..L_M:
    W_l = (1 - alpha_t) · W_l + S_W_l
    b_l = (1 - alpha_t) · b_l + S_b_l

  -- 5. Read: forward q through UPDATED MLP (Titans Eq 15)
  h_q = [q_t]
  FOR l in 1..L_M:
    pre_act = W_l @ h_q[l-1] + b_l
    IF l < L_M:
      h_q.push(σ(pre_act))
    ELSE:
      h_q.push(pre_act)
  y_t = h_q[L_M]                            -- M*(q_t)

  return y_t
```

### Attentional Bias Compatibility

The error signal `e = M(k) - v` feeds into the attentional bias before gradient computation:

| Bias | Gradient modification | Source |
|------|----------------------|--------|
| L2 (default) | `d_out = 2 · error` | Titans Eq 12 |
| L1 | `d_out = tanh(a · error)` | MIRAS Eq 14 |
| l_p | `d_out = p · sign(error) · \|error\|^{p-1}` | MIRAS Eq 24 |
| Huber | L2 if \|e\| ≤ δ, else δ·L1 | MIRAS Eq 26 |

The attentional bias modifies `d_out` (the initial gradient w.r.t. prediction) before backprop through the MLP layers. The rest of the chain rule is unchanged.

### M-Norm Adaptation

For linear M, spec 74 projects M onto the L2 ball: `if ||M||_F > max: M *= max/||M||_F`.

For MLP memory, the natural analog is to constrain the **output norm** rather than parameter norms:

```
-- After MLP weight update, check output magnitude on a probe
-- Option A: Parameter-level norm (per weight matrix)
FOR l in 1..L_M:
  norm_W = ||W_l||_F
  IF norm_W > m_norm_max_layer:
    W_l *= m_norm_max_layer / norm_W

-- Option B: No M-norm (rely on retention gate for regularization)
-- The (1-α) decay already prevents unbounded growth. MLP memory
-- with L2 retention does not exhibit the same divergence as linear M
-- because the nonlinearity bounds the output range.
```

**Decision**: Use **Option A** (per-weight-matrix Frobenius norm projection) for consistency with spec 74. The `m_norm_max` config applies to each weight matrix independently. This is the straight-through analog — project each weight matrix onto its L2 ball after the inner-loop update.

---

## Outer-Loop Backward Pass

The outer loop (AdamW) differentiates through the entire forward pass including the MLP memory operations. The Wengert tape records MLP memory operations as **opaque VJP blocks**.

### What the Outer Loop Differentiates Through

```
Outer-loop params:  W_K, W_V, W_Q, gate_params, W_init (MLP initial weights)
                    ↓
Forward pass:       For each token t:
                      k_t = x_t @ W_K^T
                      v_t = x_t @ W_V^T
                      q_t = x_t @ W_Q^T
                      gates = compute_gates(k_t, v_t, gate_params)
                      [MLP memory step — opaque VJP block]
                      y_t = output
                    ↓
Loss:               L(y, target)
```

The MLP memory step is an opaque VJP block on the Wengert tape. Its backward provides:
- `d_k_t`, `d_v_t`, `d_q_t` — gradients w.r.t. projected inputs
- `d_alpha_t`, `d_theta_t`, `d_eta_t` — gradients w.r.t. gate outputs
- `d_W_init` — gradient w.r.t. initial MLP weights (accumulated across tokens)

### MLP Memory Backward (Outer-Loop VJP)

Given `d_y_t` (gradient from downstream), compute gradients w.r.t. inputs:

```
ALGORITHM: mlp_memory_backward_token(
    d_y_t,              -- gradient from loss w.r.t. y_t
    mlp_state_t,        -- MLP weights AFTER token t's update
    mlp_state_t_prev,   -- MLP weights BEFORE token t's update
    h_k_t, h_q_t,       -- cached activations from forward
    k_t, v_t, q_t,      -- cached projections
    gates_t,             -- cached gate values
    d_M_accum,           -- accumulated gradient w.r.t. MLP weights (carried backward)
    d_S_accum            -- accumulated gradient w.r.t. momentum (carried backward)
) -> (d_k_t, d_v_t, d_q_t, d_alpha_t, d_theta_t, d_eta_t)

  -- Gradient from read: y_t = M_{t+1}(q_t)
  -- d_M from output: backprop d_y through MLP at state t+1, w.r.t. weights
  -- d_q: backprop d_y through MLP at state t+1, w.r.t. input q
  d_q_t, d_M_from_read = mlp_backward_wrt_input_and_weights(
      d_y_t, mlp_state_t, h_q_t)

  -- Accumulate into persistent d_M
  d_M_accum += d_M_from_read

  -- d_S += d_M (momentum feeds into memory: M = (1-α)M + S)
  d_S_accum += d_M_accum

  -- d_alpha_t = -sum(M_{t-1} ⊙ d_M_accum)  (retention gate gradient)
  d_alpha_t = 0.0
  FOR l in 1..L_M:
    d_alpha_t -= sum(W_l_prev ⊙ d_M_accum_W_l)
    d_alpha_t -= sum(b_l_prev ⊙ d_M_accum_b_l)

  -- d_M_accum = (1-α) · d_M_accum  (propagate through retention)
  d_M_accum *= (1 - alpha_t)

  -- d_eta_t = sum(S_{t-1} ⊙ d_S_accum)  (momentum gate gradient)
  d_eta_t = 0.0
  FOR l in 1..L_M:
    d_eta_t += sum(S_W_l_prev ⊙ d_S_accum_W_l)
    d_eta_t += sum(S_b_l_prev ⊙ d_S_accum_b_l)

  -- d_theta_t = -sum(grad_l ⊙ d_S_accum)  (learning rate gradient)
  -- grad_l is the inner-loop gradient computed during forward
  d_theta_t = 0.0
  FOR l in 1..L_M:
    d_theta_t -= sum(grad_W_l ⊙ d_S_accum_W_l)
    d_theta_t -= sum(grad_b_l ⊙ d_S_accum_b_l)

  -- d_k, d_v from the inner-loop gradient computation
  -- The inner gradient depends on k (forward through MLP + outer product)
  -- and v (subtracted from prediction). Chain rule through these.
  d_error = ...  -- backprop d_S through inner-loop gradient chain
  d_k_t = ...    -- backprop through MLP forward w.r.t. input k
  d_v_t = -d_error  -- error = prediction - v, so d_v = -d_error

  -- d_S_accum = η · d_S_accum  (propagate through momentum decay)
  d_S_accum *= eta_t

  return (d_k_t, d_v_t, d_q_t, d_alpha_t, d_theta_t, d_eta_t)
```

**Key difference from linear M backward**: Each "d_M += outer(d_y, q)" becomes a full MLP backward pass w.r.t. weights. Each "d_q = M^T @ d_y" becomes an MLP backward pass w.r.t. input. The computational cost of the backward roughly triples vs linear M.

---

## State Layout

### Inner-Loop State (per head, per level)

```
MLPMemoryState {
  -- MLP weights (inner_loop_state, mutated per-token)
  weights: Vec<{W: Tensor, b: Tensor}>  -- L_M layers
  -- Momentum accumulators (inner_loop_state)
  momentum: Vec<{S_W: Tensor, S_b: Tensor}>  -- L_M layers
}
```

Total state size per head per level (L_M=2, expansion_factor=4, d=128):
- Weights: (512×128 + 512) + (128×512 + 128) = 131,712 floats
- Momentum: same = 131,712 floats
- **Total: 263,424 floats = 1.006 MB** (vs 32,768 floats = 128 KB for linear M+S)

### Trajectory Storage (for backward)

Per-token MLP state must be stored for outer-loop backward. For seq_len=4096:

| Storage | Linear M | MLP (L_M=2) |
|---------|----------|-------------|
| M trajectory | (4097 × d²) = 67.1M floats | (4097 × 131,712) = 539.6M floats |
| S trajectory | same | same |
| Per head | 512 MB | 4.12 GB |

**This is prohibitive.** Trajectory storage must use **checkpointing** (recompute M trajectory from chunk boundaries during backward). This is the same strategy already used for linear M with `tape_strategies: ["proxy"]`:

- Store M state at chunk boundaries only (every C tokens)
- During backward, recompute the per-token M trajectory from the chunk boundary state
- Storage: O(num_chunks × state_size) instead of O(seq_len × state_size)

With chunk_size=8 (TNT local): 512 chunks × 131,712 × 2 = 134.9M floats = 515 MB per head — still large but feasible with our existing recompute infrastructure.

### Cached Activations (for analytical backward)

The inner-loop analytical gradient needs cached activations from the MLP forward pass:

```
MLPForwardCache {
  h: Vec<Tensor>     -- L_M+1 activation vectors per token (input + each layer output)
  pre_act: Vec<Tensor>  -- L_M pre-activation vectors (before σ)
}
```

Per token: (L_M+1) × d_h + L_M × d_h vectors. For L_M=2, d_h=512: 3×512 + 2×512 = 2,560 floats.
Per chunk of 8 tokens: 20,480 floats = 80 KB. Negligible.

---

## CUDA Strategy

### Why New Kernels Are Needed

The current CUDA forward kernels (`titans_forward.cu`, `titans_chunkwise_forward.cu`) implement M as a single d×d matrix with hand-written thread-level matmuls. MLP memory requires:

1. **Matrix multiplications of different shapes** (d×d_h, d_h×d) — too large for thread-level computation at expansion_factor=4
2. **Element-wise activation** (GELU) between matmuls
3. **Multi-pass backward** through MLP layers

### Approach: cuBLAS Batched GEMM + Element-Wise Kernels

Follow the existing Phase 1/Phase 2 pattern from spec 44 (batched cuBLAS backward):

```
Phase 1: cuBLAS batched GEMM for MLP forward passes (all tokens in chunk)
Phase 2: Sequential per-token loop for state updates (momentum, retention, M-norm)
```

#### Kernel Inventory

| Kernel | Purpose | Launch |
|--------|---------|--------|
| `mlp_memory_forward_cu` | Batched MLP forward: h = σ(W @ x + b) for all tokens in chunk | cuBLAS GEMM + element-wise |
| `mlp_memory_error_cu` | Compute error = prediction - v, apply attentional bias | Element-wise, 1 block/token |
| `mlp_memory_inner_backward_cu` | Analytical gradient through MLP layers (per-token) | cuBLAS GEMM + element-wise |
| `mlp_memory_update_cu` | Momentum + retention + M-norm per token (sequential) | 1 block per batch×head |
| `mlp_memory_read_cu` | Forward q through updated MLP | cuBLAS GEMM + element-wise |

#### Chunkwise Forward (TNT-Compatible)

```
FOR each chunk c of size C:
  -- Phase 1: Batch all MLP forwards in chunk (cuBLAS)
  --   Compute M(k_t) for all t in chunk using frozen M_c (chunk boundary state)
  --   This gives errors for all tokens in the chunk

  -- Phase 2: Sequential per-token update loop
  FOR t in chunk:
    1. Apply attentional bias to error
    2. Analytical backward through MLP (get grad_W_l, grad_b_l)
    3. Momentum update: S_W_l = η·S_W_l - θ·grad_W_l
    4. Retention + update: W_l = (1-α)·W_l + S_W_l
    5. Per-weight M-norm projection
    6. Read: y_t = M_updated(q_t)
```

This matches the existing Titans chunkwise pattern: Phase 1 computes errors using cuBLAS (parallelizable across tokens), Phase 2 does the sequential state update.

#### cuBLAS Workspace

Batched GEMM needs workspace for:
- Input matrices: [batch×heads, chunk_size, d] for k/q — already allocated
- Weight matrices: [batch×heads, d_h, d] for W₁ — broadcast across chunk
- Output matrices: [batch×heads, chunk_size, d_h] — new allocation

Estimated additional GPU memory per forward call: chunk_size × d_h × batch × heads × sizeof(f32).
For C=8, d_h=512, batch=3, heads=12: 8 × 512 × 3 × 12 × 4 = 589 KB — negligible.

---

## Configuration

### New Config Fields

```json
{
  "model": {
    "memory_layers": 2,              // L_M: number of MLP layers (1 = linear, 2+ = MLP)
    "memory_expansion_factor": 4,    // d_h = expansion_factor × head_dim
    "memory_activation": "gelu"      // activation: "gelu" | "silu" | "relu"
  }
}
```

Note: Bias terms are always included in MLP layers (standard MLP convention). No separate `memory_bias` config field — biases are part of the flat buffer layout unconditionally.

### Defaults and Backward Compatibility

| Field | Default | Notes |
|-------|---------|-------|
| `memory_layers` | 1 | Preserves current behavior by default |
| `memory_expansion_factor` | 4 | Matches MONETA (MIRAS Eq 24-25) |
| `memory_activation` | `"gelu"` | Matches MONETA/YAAD/MEMORA |

Existing configs with no `memory_layers` field default to 1 (linear M). No config migration needed.

### Paper-Aligned Config Example

```json
{
  "model": {
    "d_model": 1536,
    "num_heads": 12,
    "memory_layers": 2,
    "memory_expansion_factor": 4,
    "memory_activation": "gelu",
    "memory_rule": "titans",
    "composition": "mag",
    "k": 1
  }
}
```

---

## Affected Files

### Phase A: CPU Reference (Rust) — Implemented

| File | Change |
|------|--------|
| `core/src/titans_lmm.rs` | `MLPMemoryLayout` struct (flat buffer descriptor). `mlp_forward()` / `mlp_inner_backward()` helpers. `TitansLMM::step_mlp()` — full MLP forward path with gates, error, analytical backward, EMA momentum, retention, M-norm clamp. Dispatched from `step()` when `memory_layers >= 2`. |
| `core/src/model.rs` | `MemoryActivation` enum (GELU/SiLU/ReLU). `memory_layers`, `memory_expansion_factor`, `memory_activation` fields on `MAGConfig`. `tape_budget_bytes()` updated for MLP state size. |
| `core/src/opaque_adapters.rs` | Default MLP fields in backward cache/rule construction (MLP outer-loop backward deferred to Phase B). |

### Phase A: Deferred to Phase B/C

| Item | Notes |
|------|-------|
| `core/src/momentum.rs` | DeltaMomentum/DeepMomentum for MLP. Phase A only supports EMA (asserted). |
| `core/src/checkpoint.rs` | MLP weight serialization. Phase A operates in-memory only. |
| Outer-loop backward (`step_backward`) | Linear backward unchanged; MLP backward asserted-unreachable until Phase B. |

### CUDA Kernels

| File | Change |
|------|--------|
| `core/kernels/mlp_memory_forward.cu` | **NEW**. Batched MLP forward (cuBLAS GEMM + activation kernel). |
| `core/kernels/mlp_memory_backward.cu` | **NEW**. Analytical inner-loop backward through MLP layers. |
| `core/kernels/mlp_memory_update.cu` | **NEW**. Per-token momentum + retention + M-norm on MLP weights. |
| `core/kernels/mlp_memory_outer_backward.cu` | **NEW**. Outer-loop VJP: d_M, d_S coupled recurrence for MLP weights. |

### GPU Integration (Rust)

| File | Change |
|------|--------|
| `core/src/cuda_ffi.rs` | FFI declarations for 4 new kernel entry points. |
| `core/src/dispatch.rs` | Dispatch wrappers routing linear vs MLP based on `memory_layers`. |
| `core/src/gpu_forward.rs` | MLP memory path in `gpu_memory_forward`. Route to cuBLAS path when `memory_layers >= 2`. |
| `core/src/gpu_backward.rs` | MLP memory backward path. Phase 1 (cuBLAS error recompute) + Phase 2 (MLP trajectory recompute + reverse token loop). |
| `core/src/gpu_stacked_forward.rs` | Allocate MLP weight buffers. Pass MLP state through the forward pipeline. |
| `core/src/gpu_stacked_optimizer.rs` | AdamW on MLP initial weights (W_init). Grad norm includes MLP weight gradients. |

### Build

| File | Change |
|------|--------|
| `core/build.rs` | Compile new `.cu` files. Link cuBLAS (already linked for Phase 1 backward). |

---

## Parallelization Compatibility

| Strategy | Compatible? | Notes |
|----------|------------|-------|
| ChunkwiseGD | **YES** | Freeze MLP weights at chunk boundary, compute all errors, then sequential update. Same pattern as linear M. |
| AssociativeScan | **NO** | Nonlinear activation breaks the linear recurrence required for scan. MLP memory forces chunkwise GD. |
| TNT Hierarchical | **YES** | MLP memory can serve as global or local memory. Local memories reset MLP weights to W_init at shard boundaries (TNT Eq 6). |

When `memory_layers >= 2`, the system must disable associative scan and fall back to chunkwise GD. This is enforced at config validation time.

---

## Phased Implementation

### Phase A: CPU Reference (Rust)

1. Add `MLPMemory` struct to `titans_lmm.rs`
2. Implement MLP forward, analytical backward, momentum update, retention
3. Wire into `MemoryRule::step()` dispatch based on `memory_layers`
4. Verify: `memory_layers=1` produces bit-identical output to current code
5. Add finite-difference gradient checks for `memory_layers=2`
6. Add unit tests for MLP forward, backward, update, checkpoint round-trip

### Phase B: CUDA Kernels

1. Implement `mlp_memory_forward.cu` (cuBLAS GEMM + GELU kernel)
2. Implement `mlp_memory_backward.cu` (analytical inner-loop backward)
3. Implement `mlp_memory_update.cu` (momentum + retention + M-norm)
4. Verify CUDA matches CPU reference within 1e-4

### Phase C: GPU Integration

1. Wire MLP memory into `gpu_forward.rs` / `gpu_backward.rs`
2. Implement outer-loop VJP for MLP memory (Wengert tape opaque block)
3. Allocate MLP weight buffers in `gpu_stacked_forward.rs`
4. End-to-end training test: 100 steps, compare loss curve to CPU reference

---

## Acceptance Criteria

1. **Backward compatibility**: `memory_layers=1` produces bit-identical output to current linear M implementation on both CPU and GPU
2. **CPU correctness**: MLP memory (L_M=2) forward output matches a reference NumPy implementation within 1e-6
3. **CUDA correctness**: GPU MLP memory matches CPU reference within 1e-4 per element
4. **Gradient verification**: Finite-difference checks pass for all MLP weight gradients (inner-loop analytical + outer-loop VJP) with FD_TOL=10%, abs_threshold=5e-4
5. **Existing tests**: All 778+ Rust tests pass unchanged
6. **New tests**: MLP forward, backward, update, momentum, retention, M-norm, checkpoint round-trip, chunkwise compatibility
7. **Config validation**: `memory_layers >= 2` with `parallel_strategy: "associative_scan"` rejected at config load time
8. **Training**: 100-step GPU training run with `memory_layers=2` converges (loss decreasing) and matches CPU loss curve within f32 tolerance

---

## Code Smell Compliance

| Smell | Compliance |
|-------|-----------|
| CS-10 (no mode distinction) | MLP memory updates identically in all contexts — no train/eval split |
| CS-18 (forward IS the API) | MLP memory step is part of the forward pass, not a separate training API |
| CS-32 (observe-then-advance) | Read M(q) after update, matching observe-then-advance |
| CS-34 (don't restrict to matrix) | **This spec directly addresses CS-34** — MLP memory is the non-matrix option |
| CS-35 (don't assume GD only) | MLP memory uses GD by default but the algorithm knob is independent |
| CS-40 (opt-in AD) | Inner-loop gradients are analytical. Outer-loop uses Wengert tape with opt-in. |
| CS-42 (arena intermediates) | MLP activations stored in arena during forward, freed after backward |
