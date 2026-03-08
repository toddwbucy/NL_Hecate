# Residual Stream + LayerNorm for MAG Forward Pass

```text
CONTRACT
  Purpose  : Add a residual stream and pre-LayerNorm to the MAG forward pass,
             providing an unattenuated gradient path from loss to all CMS levels
  Expects  : Single-block MAG architecture with SWA + CMS memory (k≥1)
  Guarantees: (1) Every CMS level's output is ADDED to a shared residual stream
             (gradient = 1.0, no sigmoid attenuation);
             (2) Pre-LN stabilizes activation magnitudes before attention and memory;
             (3) Backward pass preserves gradient magnitude through the residual
             skip path regardless of gate saturation;
             (4) Existing k=1 behavior recoverable via config flag (residual=false)
  Cost     : +2d learnable parameters (two LayerNorm γ,β vectors);
             ~5% forward-pass FLOP increase (two LayerNorm + one vector add)
  Trade-off: Introduces a component not in the original Titans paper's MAG
             definition. Justified by: (a) every Titans experiment used models
             with depth≥6 where inter-block residuals provide the same gradient
             highway, (b) our single-block architecture has no other skip path,
             (c) empirical evidence of dead higher levels across 5+ experiments
  Position : Infrastructure — gradient flow fix, prerequisite to multi-block stacking
  Source   : Standard pre-LN transformer practice (Xiong et al., 2020);
             HOPE (2512.24695) §5 structural equivalence (AdaTransformer = Transformer
             with adaptive MLP → residual stream implied);
             Titans (2501.00663) §4 all experimental models have inter-block residuals
```

## Problem Statement

The current MAG forward pass has no residual stream:

```
embedded → [SWA attention] → attn_out
embedded → [CMS memory]    → y_combined → sigmoid → gate
output = attn_out * gate → w_o → w_unembed → logits
```

The ONLY gradient path from loss to CMS memory parameters passes through:
1. `d(loss)/d(gate)` — depends on `attn_out` magnitude
2. `d(gate)/d(y_combined) = sigmoid'(y_combined)` — **≈0.05 when gate saturates**
3. `d(y_combined)/d(level_output)` — divided by `1/sqrt(k)` for k>2

For higher CMS levels that fire infrequently (L2 every 64 steps, L3 every 512),
this compounding attenuation produces effectively zero gradients. Across 5+
experiments (cold-start k=4, manual pyramid, auto-promotion on dolmino, reasoning,
shakespeare), higher levels consistently show zero gradient post-promotion.

## Solution: Residual Stream

Add a residual stream that CMS memory levels contribute to via **addition**,
not via sigmoid gating:

```
embedded → LN_attn → [SWA attention] → attn_out
                                            ↓
residual = embedded + attn_out              ← skip connection 1
                                            ↓
residual → LN_mem → [CMS memory] → y_combined
                                            ↓
residual = residual + y_combined            ← skip connection 2 (THE FIX)
                                            ↓
residual → w_o → w_unembed → logits
```

### Why This Fixes Dead Levels

1. **Addition has gradient = 1.0**: `d(residual)/d(y_combined) = 1.0` regardless
   of magnitude. No sigmoid attenuation. No gate saturation. Every CMS level's
   gradient is the SAME magnitude as the loss gradient on the residual.

2. **Slow levels read enriched context**: When L3 fires, it reads
   `LN_mem(residual)` which contains L0's accumulated contributions from hundreds
   of prior steps. L3 gets meaningful input to work with.

3. **Levels contribute corrections**: Each level adds a correction to the residual
   rather than having to carry the entire signal through a gate. Even a small
   correction produces gradient signal.

### MAG Gating: Preserved Within the Residual

The sigmoid gating that defines MAG composition is NOT removed. It moves INSIDE
the memory branch's contribution to the residual:

```
y_combined = sigmoid(memory_out) * attn_out_prenorm  # MAG gating preserved
residual = residual + y_combined                      # added to residual
```

This preserves the MAG property (memory gates attention) while giving the gated
output an unattenuated path to the loss. The key difference: previously `gated_out`
WAS the output; now `gated_out` is ADDED to the residual. The gradient through
the addition is 1.0; the gradient through the gate is still sigmoid' — but both
paths exist, and the addition path dominates for slow levels.

**Alternative (simpler, recommended for first implementation)**: Drop the internal
MAG gating entirely and let memory contribute directly:

```
residual = embedded + attn_out + memory_out           # pure additive
```

This is closer to standard transformer architecture (attention + MLP in parallel,
both added to residual). The MAG gating can be re-added as an ablation variant
once we confirm that the residual stream fixes dead levels.

## LayerNorm Specification

### Pre-LN (before attention and before memory)

```
LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```

Where:
- `x ∈ R^d` — input vector (per-position)
- `μ = mean(x)`, `σ² = var(x)` — computed per-position across d dimensions
- `γ, β ∈ R^d` — learnable scale and shift (outer-loop params)
- `ε = 1e-5` — numerical stability

### Two LayerNorm instances

| Name | Input | Output feeds | Learnable params |
|------|-------|-------------|-----------------|
| `ln_attn` | `embedded` (or `residual` in multi-block) | SWA attention Q,K,V | γ_attn, β_attn ∈ R^d |
| `ln_mem` | `residual` (after attention skip) | CMS memory input | γ_mem, β_mem ∈ R^d |

### Parameter count impact

+4d parameters total (2 × γ + 2 × β at d dimensions each).
At d=512: +2,048 params (0.006% of 36M model). Negligible.

### Initialization

- `γ` initialized to all-ones (identity scaling)
- `β` initialized to all-zeros (no shift)
- This makes LayerNorm a near-identity at initialization — the model starts
  equivalent to the current architecture, then learns to normalize.

## Forward Pass: New Data Flow

### CPU path (`mag.rs::cms_forward`)

```rust
// Stage 1: Embedding
let embedded = embed(input_ids, &params.swa.w_embed);  // [s, d]

// Stage 2: Pre-LN for attention
let ln_attn_out = layer_norm(&embedded, &params.ln_attn_gamma,
                              &params.ln_attn_beta, s, d);  // [s, d]

// Stage 3: SWA Attention
let q = ln_attn_out @ w_q^T;
let k = ln_attn_out @ w_k^T;
let v = ln_attn_out @ w_v^T;
let attn_out = swa_forward(q, k, v);  // [s, d]

// Stage 4: Residual skip 1 — attention
let residual = embedded + attn_out;  // [s, d]

// Stage 5: Pre-LN for memory
let ln_mem_out = layer_norm(&residual, &params.ln_mem_gamma,
                             &params.ln_mem_beta, s, d);  // [s, d]

// Stage 6: CMS Memory (all levels read ln_mem_out)
let memory_out = cms_memory_levels(ln_mem_out, ...);  // [s, d]

// Stage 7: Residual skip 2 — memory (THE KEY CHANGE)
let residual = residual + memory_out;  // [s, d]

// Stage 8: Output projection
let projected = residual @ w_o^T;  // [s, d]

// Stage 9: Unembed + loss
let logits = projected @ w_unembed;  // [s, v]
let loss = cross_entropy(logits, targets);
```

### GPU path (`gpu_forward.rs::gpu_cms_forward`)

Same stages, using CUDA kernels:
- `layer_norm_cuda` — new kernel (forward + backward)
- `vector_add_cuda` — new kernel (or reuse `saxpy_cuda` with α=1.0)
- All existing attention, memory, unembed kernels unchanged

## Backward Pass Changes

### Residual gradient flow

```
d_residual = d_projected @ w_o          // from unembed+projection backward
d_memory_out = d_residual               // ADDITION: gradient = 1.0 (no attenuation!)
d_residual_pre_mem = d_residual         // ADDITION: gradient also flows to skip path
d_ln_mem_input = d_memory_backward(d_memory_out) @ LN_backward  // through memory levels
d_residual += d_ln_mem_input            // accumulate LN+memory backward into residual
d_attn_out = d_residual                 // from skip connection 1
d_embedded = d_residual                 // from skip connection 1
d_ln_attn_input = d_attn_backward(d_attn_out) @ LN_backward
d_embedded += d_ln_attn_input           // accumulate LN+attention backward
```

The critical point: `d_memory_out = d_residual` — the loss gradient reaches every
CMS level at full strength. No sigmoid'. No 1/sqrt(k). The memory backward pass
then distributes this gradient across levels proportional to their contribution.

### LayerNorm backward

Standard (well-known, numerically stable formulation):

```
d_gamma = sum_over_positions(d_out * x_hat)
d_beta  = sum_over_positions(d_out)
d_x     = (1/d) * (1/sigma) * (d * d_out * gamma
          - sum(d_out * gamma) - x_hat * sum(d_out * gamma * x_hat))
```

Where `x_hat = (x - μ) / sigma` is the normalized input. This is a pure function
of the forward quantities — no special handling needed for the Wengert tape.

## Config Changes

### New fields in MAGConfig / BuildConfig

```python
# In model section of config JSON:
"residual": true,       # Enable residual stream (default: true for new runs)

# Internally in MAGConfig (Rust):
pub residual: bool,     // default true
```

When `residual=false`, the forward pass reverts to the current behavior
(no skip connections, no LayerNorm, gated_out directly to w_o). This allows
A/B comparison and backward compatibility with existing checkpoints.

### New learnable parameters

```rust
// Added to SWAParams (outer-loop, serialized):
pub ln_attn_gamma: Vec<f32>,  // [d], init: ones
pub ln_attn_beta: Vec<f32>,   // [d], init: zeros
pub ln_mem_gamma: Vec<f32>,   // [d], init: ones
pub ln_mem_beta: Vec<f32>,    // [d], init: zeros
```

These are outer-loop parameters: updated by AdamW during build, serialized
in checkpoints, restored on resume. They follow the same lifecycle as
`w_q`, `w_k`, `w_v`, `w_o`.

## Checkpoint Compatibility

### New tensor names in safetensors

```
ln_attn.gamma    [d]
ln_attn.beta     [d]
ln_mem.gamma     [d]
ln_mem.beta      [d]
```

### Loading old checkpoints (backward compat)

When loading a checkpoint that lacks `ln_attn.*` / `ln_mem.*` tensors:
- If `residual=true`: initialize LN params to identity (γ=1, β=0)
- If `residual=false`: LN params not needed, skip
- Print: `"  Note: checkpoint missing LN params, initialized to identity"`

This allows loading any existing k=1/k=2/k=4 checkpoint and enabling
the residual stream without retraining from scratch.

## Files to Modify

| File | Change | Scope |
|------|--------|-------|
| `core/src/model.rs` | Add `ln_attn_gamma/beta`, `ln_mem_gamma/beta` to `SWAParams`; add `residual: bool` to `MAGConfig`; update `init()`, `num_params()`, `extend_push_up()`, `extend_stack_up()` | Struct + init |
| `core/src/mag.rs` | Rewrite `cms_forward` stages 2-7 with LN + residual; update `CMSForwardCache` | CPU forward |
| `core/src/backward.rs` | Add LN backward; rewrite residual gradient flow in `cms_backward` | CPU backward |
| `core/src/gpu_forward.rs` | Add `layer_norm_cuda` calls; rewrite stages 4-5 with residual add (4 call sites) | GPU forward |
| `core/src/gpu_backward.rs` | Add LN backward kernels; rewrite residual gradient routing | GPU backward |
| `core/kernels/layer_norm.cu` | New file: `layer_norm_forward_cuda`, `layer_norm_backward_cuda` kernels | CUDA |
| `core/src/checkpoint.rs` | Add `ln_attn.*`, `ln_mem.*` tensor names; backward-compat loading | Serialization |
| `python/src/lib.rs` | Add `residual` field to PyO3 `MAGConfig`; expose LN params | Bindings |
| `python/engine/config.py` | Add `residual: bool = True` to model config | Config |

## CUDA Kernel: LayerNorm

### Forward

```cuda
// One block per position (token), blockDim.x = min(d, 1024)
__global__ void layer_norm_forward(
    const float* x,      // [n, d] input
    const float* gamma,  // [d] scale
    const float* beta,   // [d] shift
    float* out,          // [n, d] normalized output
    float* mean_cache,   // [n] cached means (for backward)
    float* rstd_cache,   // [n] cached 1/sqrt(var+eps) (for backward)
    int n, int d, float eps
);
```

### Backward

```cuda
__global__ void layer_norm_backward(
    const float* d_out,      // [n, d] upstream gradient
    const float* x,          // [n, d] original input
    const float* gamma,      // [d] scale
    const float* mean_cache, // [n]
    const float* rstd_cache, // [n]
    float* d_x,              // [n, d] gradient w.r.t. input
    float* d_gamma,          // [d] gradient w.r.t. scale (atomicAdd across positions)
    float* d_beta,           // [d] gradient w.r.t. shift (atomicAdd across positions)
    int n, int d
);
```

Standard implementation: warp-level reduction for mean/variance in forward,
three-term formula for d_x in backward. Well-understood, no novel math.

## Verification Plan

### Unit tests (Rust)

1. **LayerNorm forward**: random input → verify output has mean≈0, var≈1 per position
2. **LayerNorm backward**: finite-difference gradient check at d=8
3. **Residual forward equivalence**: `residual=false` produces identical output
   to current code (bitwise)
4. **Residual forward correctness**: `residual=true` produces
   `output = embed + attn(LN(embed)) + memory(LN(embed + attn(LN(embed))))`
5. **End-to-end gradient check**: FD at d=8, k=2, verify all level gradients nonzero

### GPU tests

6. **CUDA LN forward**: matches CPU LN within 1e-5
7. **CUDA LN backward**: matches CPU LN backward within 1e-4
8. **GPU residual forward**: matches CPU residual forward within 1e-5
9. **GPU residual backward**: matches CPU residual backward within 1e-4

### Integration test (Python)

10. **Level gradient signal**: Train k=2 for 100 steps with `residual=true`.
    Assert L1 gnorm > 0.01 at step 100. (Current architecture: L1 gnorm ≈ 0)

## Falsification Criterion

If, after implementing the residual stream and running the auto-promotion
curriculum at d=512 k=1→k=4 on the Shakespeare dataset (50K steps), higher
levels STILL show zero gradients post-promotion, then the dead-level problem
is NOT caused by gradient attenuation through the sigmoid gate. The problem
would instead be:
- Model scale (36M too small for CMS)
- Single-block architecture (need depth for multiple gradient paths)
- Or a fundamental issue with the CMS frequency schedule

This experiment is designed to isolate the gradient-path hypothesis before
investing in the larger multi-block stacking effort.
