# 74 — Per-Token M-Norm Projection in CUDA Forward Kernels

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Add per-token M-norm projection inside CUDA forward kernels, matching the CPU reference behavior |
| Expects   | Working `m_norm_clamp_batch_f32_cuda` post-forward clamp (spec 65); CPU reference per-token clamp in `titans_lmm.rs:377-386` |
| Guarantees | M stays bounded throughout the forward pass (not just after it); eliminates the M-growth → clamp → freeze → gradient mismatch cycle; CPU/GPU behavioral parity |
| Cost      | One shared-memory reduction + conditional scale per token per kernel invocation; ~2% overhead at d=64 (hd) |
| Trade-off | Slightly higher per-token kernel cost vs dramatically improved training stability; M can no longer grow unchecked within a kernel invocation |
| Position  | Supersedes spec 65's post-forward clamp as the primary M-norm control; spec 65 batched clamp retained as safety net |
| Source    | CPU reference: `titans_lmm.rs:377-386`; training instability analysis (d=1024 build, 2026-04-01) |
| Traced equations | titans_equations/eq-003-memory-update (Titans Eq. 3: M_{t+1} update rule) |

## Problem

### CPU/GPU Behavioral Divergence

The CPU reference (`core/src/titans_lmm.rs:377-386`) applies M-norm clamping **per-token** — after each token's M update and before computing the output `y_t = M_{t+1} @ q_t`:

```rust
// CPU reference (titans_lmm.rs:377-386)
if self.m_norm_max < f32::MAX {
    let slice = &mut m_states[m_next_off..m_next_off + d * d];
    let norm_sq: f32 = slice.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > self.m_norm_max {
        let scale = self.m_norm_max / norm;
        for x in slice.iter_mut() { *x *= scale; }
    }
}
```

The GPU kernels (`titans_forward.cu`, `titans_chunkwise_forward.cu`) have **no per-token M-norm control**. M grows unchecked throughout the entire kernel invocation (8 tokens for TNT local chunks, up to seq_len for non-TNT paths). The post-forward clamp (spec 65) only fires after the kernel returns.

### Consequences of the Divergence

During a kernel invocation processing N tokens:
1. M starts at the clamped value (e.g., ||M|| = 10.0)
2. M grows per-token as the update rule accumulates: `M_{t+1} = (1-alpha) * M_t + S_{t+1}`
3. After N tokens, ||M|| may reach 50+ (5x the clamp target)
4. All intermediate computations (`prediction = M @ k`, `y = M @ q`) used the inflated M
5. Post-forward clamp projects M back to 10.0 for the next step
6. Outer-loop gradients are computed based on inflated-M dynamics but applied to clamped-M weights

This creates the observed instability pattern:
- `m_delta = 0.000` for thousands of steps (M at clamp ceiling, post-forward clamp erases all updates)
- Gradient norm spikes (10K-34K) from inflated-M intermediate computations
- Catastrophic loss spikes when the feedback loop between M growth and weight updates destabilizes

### Evidence

d=1024 build (2026-04-01), `m_norm_max = [10.0]`:
```text
step=  393  m_norm=40.00  m_delta=0.000   (M frozen at clamp ceiling)
step= 1097  m_norm=40.00  m_delta=0.000   gnorm=14851
step= 6553  loss=11.32    ppl=82298       (catastrophic spike)
```

The model converged beautifully when M was well below the clamp (steps 3000-5000, loss ~4.0, gnorm ~2-5) but destabilized when M saturated at the ceiling.

## Design: Per-Token M-Norm Projection

### Algorithm

After each token's M update and before computing the output, project M onto the L2 ball of radius `m_norm_max`:

```text
M_{t+1} = (1-alpha_t) * M_t + S_{t+1}          // existing update
if ||M_{t+1}||_F > m_norm_max:                   // NEW: per-token projection
    M_{t+1} <- M_{t+1} * (m_norm_max / ||M_{t+1}||_F)
y_t = M_{t+1} @ q_t                              // output uses bounded M
```

This is identical to the CPU reference behavior. The projection is:
- **Idempotent**: applying twice gives the same result
- **Direction-preserving**: M's direction is unchanged, only magnitude is bounded
- **Straight-through in backward**: gradient flows as identity (same as spec 65)

### CUDA Implementation

Inside the per-token loop, after M/S update and before y computation:

```c
// Per-token M-norm projection (matches CPU reference)
// Reuse prediction[] shared memory for partial sums
if (m_norm_max < 1e30f) {
    float local_sq = 0.0f;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        float v = m_states[m_next_off + idx];
        local_sq += v * v;
    }
    prediction[tid] = local_sq;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) prediction[tid] += prediction[tid + s];
        __syncthreads();
    }

    float fnorm = sqrtf(prediction[0]);
    if (fnorm > m_norm_max) {
        float scale = m_norm_max / fnorm;
        for (int idx = tid; idx < dd; idx += blockDim.x) {
            m_states[m_next_off + idx] *= scale;
        }
    }
    __syncthreads();
}
```

The `prediction[]` shared memory buffer is safe to reuse here — it was written in the prediction step and is not needed again until the next token's prediction.

### Kernel Signature Changes

All affected kernels gain one parameter: `float m_norm_max`

```c
// titans_forward.cu
extern "C" void titans_forward_f32_cuda(
    ..., float error_clip, float m_norm_max);  // NEW parameter

// titans_chunkwise_forward.cu (Phase 2)
extern "C" void titans_chunkwise_forward_f32_cuda(
    ..., float error_clip, float m_norm_max);  // NEW parameter
```

Disabled when `m_norm_max >= 1e30` (same convention as spec 65).

### Files Modified

| File | Change |
|------|--------|
| `core/kernels/titans_forward.cu` | Add per-token M-norm projection after M update, pass `m_norm_max` |
| `core/kernels/titans_chunkwise_forward.cu` | Same for Phase 2 sequential path |
| `core/kernels/delta_forward.cu` | Same pattern for Delta rule |
| `core/kernels/delta_chunkwise_forward.cu` | Same |
| `core/src/cuda_ffi.rs` | Update FFI signatures with `m_norm_max: f32` parameter |
| `core/src/dispatch.rs` | Update dispatch wrappers to pass `m_norm_max` |
| `core/src/gpu_forward.rs` | Pass `m_norm_max` to kernel dispatch calls |

### Interaction with Spec 65 (Post-Forward Batched Clamp)

Spec 65's post-forward clamp is **retained as a safety net**. With per-token projection:
- The post-forward clamp should be a no-op in normal operation (M is already bounded)
- It catches edge cases: TNT global M update (outer product in Rust code) can push M above the target between shards
- Zero additional cost when it's a no-op (norm check is fast)

## Performance Impact

Per-token overhead for d=64 (head_dim in production configs):
- Sum-of-squares: 64 FP multiplies + adds per thread (`dd/blockDim.x = 4096/64`)
- Tree reduction: 6 steps (`log2(64)`)
- Conditional scale: 64 FP multiplies per thread (when triggered)

Compared to existing per-token work:
- `M @ k` matrix-vector product: `2 * 64 * 64 = 8192` FP ops
- `M @ q` matrix-vector product: `8192` FP ops
- M/S update: `5 * 4096 = 20480` FP ops

The projection adds ~4160 FP ops per token (sum-of-squares + scale), which is ~11% of a single matvec. With the conditional (only fires when `||M|| > target`), it's often just the sum-of-squares check (~2048 ops, ~5%).

## Verification

1. **CPU/GPU parity**: Run CPU reference and GPU kernel on identical inputs with `m_norm_max=5.0`. Per-element tolerance 1e-5 (existing forward tolerance).
2. **Existing tests**: `cargo test --features cuda` — all pass (the clamp is additive, not modifying existing behavior when M is below threshold).
3. **Training equivalence**: The d=1024 build should show:
   - `m_delta > 0` consistently (M no longer frozen)
   - `grad_norm` staying in single digits through warmup and beyond
   - No catastrophic loss spikes
   - Monotonic loss descent without phase transitions
4. **No-op when disabled**: `m_norm_max = f32::MAX` (or 0.0 in config) produces bit-identical output to current kernels.
