# 66 — Multi-Head Fused Memory Kernel

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Extend the fused memory kernels (spec 39) to work with `num_heads > 1`, eliminating the `nh == 1` guard that renders the fused path dead code for all production configs |
| Expects   | Working fused forward kernels (`dgd_fused_forward_kernel`, `titans_fused_forward_kernel`) for nh=1; per-head memory layout (spec 45) with heads folded into batch dim |
| Guarantees | Fused path active for all num_heads configs; bit-identical output vs unfused path; measurable tok/s improvement at d=768 hd=64 |
| Cost      | Gate pre-computation adds 2 small kernel launches (l2_normalize + gate_compute) that remain outside the fused kernel; net launch count still drops |
| Trade-off | Pre-computed gates vs gate-internal-to-kernel. Pre-computed is simpler (no kernel changes to gate math), at cost of 2 extra kernel launches vs a fully fused single-kernel approach |
| Position  | Extends spec 39 (single-head fusion); prerequisite for d=1024+ scaling |
| Source    | gpu_forward.rs line 1208: `nh == 1` guard; task_ed9b3d |

## Problem

The fused forward kernels compute gates **internally** using `dot(concat(k,v), w_alpha) + b_alpha`
at d_model resolution. This is incompatible with per-head memory (spec 45) where:

1. Gates are computed at d_model resolution (before head split)
2. Gate scalars are broadcast from `[bs, s]` to `[bs*nh, s]`
3. The memory recurrence operates at `hd` resolution with `bs*nh` batch elements

The fused kernels expect `w_alpha[2*d]` and internally do the dot product, but when the
kernel operates at `d=hd` the gate weight vectors would need to be `[2*hd]` — which is
wrong because gates are a function of the full d_model-resolution k and v.

## Design: Split-Fuse Strategy

Rather than rewriting gate computation inside the fused kernel for multi-head, **pre-compute
gates at d_model resolution and pass them in**. The fused kernel then skips its internal gate
computation and uses the pre-computed alpha/theta scalars directly.

This is the cleanest approach because:
- No changes to gate weight handling or shapes
- The fused kernel's recurrence loop is identical — just reads alpha/theta from a buffer instead of computing them
- The backward path for gates is already separate (no fused backward exists)
- Net kernel launch reduction is still substantial

### Current unfused path (per level, nh > 1)

```text
Launch 1-3: cuBLAS K/V/Q projections          [bs*s, d_model]
Launch 4-5: l2_normalize k, q                  [bs*s, d_model]
Launch 6-7: gate_compute alpha, theta           [bs*s] (d_model resolution)
Launch 8-9: clamp alpha, theta (optional)
Launch 10-11: transpose_heads k/v/q            [bs*nh, s, hd]
Launch 12-13: broadcast_heads alpha/theta      [bs*nh, s]
Launch 14: delta/titans_forward_dd              bs_mem=bs*nh, d=hd
Launch 15: cuda_sync
Launch 16+: m_norm_clamp (loop over bs_mem)
Launch 17: reshape_from_per_head y
```

**17+ launches per level.**

### New fused path (per level, nh > 1)

```text
Launch 1-3: cuBLAS K/V/Q projections           [bs*s, d_model]
Launch 4-5: l2_normalize k, q                  [bs*s, d_model]  — KEEP (d_model res)
Launch 6-7: gate_compute alpha, theta           [bs*s]           — KEEP (d_model res)
Launch 8-9: clamp alpha, theta (optional)                        — KEEP
Launch 10-11: transpose_heads k/v/q             [bs*nh, s, hd]  — KEEP
Launch 12-13: broadcast_heads alpha/theta       [bs*nh, s]       — KEEP
Launch 14: fused_forward_dd (GATES PRE-COMPUTED) bs_mem=bs*nh, d=hd  — FUSED (recurrence only)
Launch 15: cuda_sync
```

**15 launches per level, BUT the fused kernel replaces the unfused recurrence AND inlines
the M-norm clamp.** The fused kernel also eliminates intermediate buffer writes for
k_norms/q_norms within the recurrence (these are already computed at d_model resolution
and not needed at hd resolution).

Wait — that's wrong. The real savings come from a different angle. Let me reconsider.

### Actual analysis: what the fused kernel buys for multi-head

The unfused `delta_forward_dd` / `titans_forward_dd` kernel already handles `bs*nh` batch
elements efficiently. The fused kernel's advantage over the unfused kernel is:

1. **Internal L2 normalization** of k/q per-token (eliminates 2 launches)
2. **Internal gate computation** (eliminates 2 launches + 2 clamp launches)
3. **Internal M-norm clamp** at end (eliminates per-batch-head loop of clamp launches)
4. **k/v/q stay in registers** across norm→gate→recurrence (no global round-trip)

For multi-head, items 1-2 cannot fire inside the fused kernel because they need d_model
resolution. But items 3-4 still apply. More importantly, if we restructure slightly:

**Pre-compute gates and norms at d_model, then pass to a "recurrence-only" fused kernel
that inlines M-norm clamp.** The net savings:

- Eliminated: per-batch-head `m_norm_clamp` loop (currently `bs*nh` separate launches)
- Eliminated: k_norms/q_norms intermediate buffers (not needed — norms done at d_model)
- Reduced: the unfused recurrence kernel re-reads alpha/theta from global; fused can
  preload into shared memory once per sequence

At d=768, nh=12, bs=2: that's 24 separate `m_norm_clamp` launches eliminated per level,
× 4 levels × 6 blocks = **576 eliminated clamp launches per step**. That alone is significant.

### Revised approach: Multi-Head Recurrence-Fused Kernel

New CUDA kernel variant: `dgd_fused_forward_multihead_kernel` / `titans_fused_forward_multihead_kernel`

**Differences from nh=1 fused kernel:**
- Does NOT compute gates internally (gates pre-computed, passed as `alpha[bs_mem*s]`, `theta[bs_mem*s]`)
- Does NOT normalize k/q internally (already normalized at d_model resolution before head split)
- DOES inline M-norm clamp at the end of the sequence loop
- `d` parameter = `hd` (head_dim), not d_model
- `batch_size` parameter = `bs * nh`
- Gate weight vectors (`w_alpha`, `w_theta`) NOT passed — not needed

**Kernel signature:**

```c
__global__ void dgd_fused_forward_multihead_kernel(
    const float* k_mem,      // [bs_mem * s, hd] — L2-normalized, head-transposed
    const float* v_mem,      // [bs_mem * s, hd]
    const float* q_mem,      // [bs_mem * s, hd] — L2-normalized, head-transposed
    const float* alpha,      // [bs_mem * s] — pre-computed, broadcast to per-head
    const float* theta,      // [bs_mem * s] — pre-computed, broadcast to per-head
    const float* m_initial,  // [bs_mem, hd*hd] — initial M per head
    float* m_states,         // [bs_mem, (s+1), hd*hd] — full trajectory
    float* y,                // [bs_mem, s, hd] — output
    float* alpha_out,        // [bs_mem * s] — pass-through for backward
    float* theta_out,        // [bs_mem * s] — pass-through for backward
    float m_norm_max,        // Frobenius norm clamp
    int seq_len, int d, int batch_size,
    float error_clip
);
```

For Titans, adds: `const float* eta`, `const float* s_initial`, `float* s_states`, `float* eta_out`.

**Grid/Block:** Same as existing fused kernel — `dim3(bs_mem)`, block = `min(hd*hd, 1024)`.
At hd=64: dd=4096, block=1024, 4 iterations per thread to cover all M elements. This is
well within register budget (hd=64 is 16× smaller M than d_model=768).

### Shared memory layout (multihead fused)

```c
float prediction[hd];    // hd floats
float error_buf[hd];     // hd floats
float k_buf[hd];         // hd floats — loaded from per-head k_mem
float v_buf[hd];         // hd floats
float q_buf[hd];         // hd floats
float warp_scratch[32];  // warp reduction
// Total: 5*hd + 32 = 5*64 + 32 = 352 floats = 1.4 KB
```

NO gate weight vectors in shared memory (gates are pre-computed scalars read from global).
This is much less shared memory than the nh=1 fused kernel (which stores `w_alpha[2*d]` +
`w_theta[2*d]` = 4*768 = 12 KB at d=768).

### Per-token loop (inside kernel)

```c
for (int t = 0; t < seq_len; t++) {
    // Load pre-computed gate values (2 floats from global — trivial)
    float a = alpha[batch_idx * seq_len + t];
    float th = theta[batch_idx * seq_len + t];

    // Load k[t], v[t], q[t] into shared memory (already normalized, per-head)
    load_vector(k_buf, k_mem + (batch_idx * seq_len + t) * d);
    load_vector(v_buf, v_mem + (batch_idx * seq_len + t) * d);

    // prediction = M @ k (cooperative across threads)
    cooperative_matvec(M_ptr, k_buf, prediction, d);

    // error = prediction - v
    error_buf[tid] = prediction[tid] - v_buf[tid];

    // M update: M = (1-a)*M - th * error ⊗ k
    float decay = 1.0f - a;
    for (int i = tid; i < dd; i += blockDim.x) {
        int row = i / d, col = i % d;
        M_ptr[i] = decay * M_ptr[i] - th * error_buf[row] * k_buf[col];
    }

    // Store M[t+1] in trajectory
    store_m(m_states, M_ptr, t+1, batch_idx, dd, seq_len);

    // Load q, compute y[t] = M @ q
    load_vector(q_buf, q_mem + (batch_idx * seq_len + t) * d);
    cooperative_matvec(M_ptr, q_buf, y_out, d);
    store_vector(y + (batch_idx * seq_len + t) * d, y_out, d);
}

// Inline M-norm clamp (replaces separate kernel launch)
float norm_sq = 0.0f;
for (int i = tid; i < dd; i += blockDim.x)
    norm_sq += M_ptr[i] * M_ptr[i];
// warp reduce norm_sq → total
float norm = sqrtf(total);
if (norm > m_norm_max) {
    float scale = m_norm_max / norm;
    for (int i = tid; i < dd; i += blockDim.x)
        M_ptr[i] *= scale;
}
```

## Dispatch Changes (Rust side)

### gpu_forward.rs

Remove the `nh == 1` guard. New dispatch logic:

```rust
// Spec 65: fused path works for all num_heads
let use_fused = eff_ckpt.is_none()
    && !is_proxy
    && matches!(cfg.memory_rule, MemoryRuleKind::DeltaRule | MemoryRuleKind::TitansLMM);

if use_fused {
    if nh == 1 {
        // Original spec 39 path: gates computed inside kernel
        // (preserved for backward compat, eventually removable)
        ...existing code...
    } else {
        // Spec 65: multi-head fused path
        // Step 1: L2 normalize k, q at d_model resolution
        l2_normalize_rows_f32_cuda(k_mem, bs * s, d);
        l2_normalize_rows_f32_cuda(q_mem, bs * s, d);

        // Step 2: Compute gates at d_model resolution
        gate_compute_cuda(k_mem, v_mem, w_alpha, b_alpha, alpha, ...);
        gate_compute_cuda(k_mem, v_mem, w_theta, b_theta, theta, ...);
        // + optional clamp

        // Step 3: Transpose to per-head layout
        transpose_heads(k_mem, ...);  // [bs, s, d] → [bs*nh, s, hd]
        transpose_heads(v_mem, ...);
        transpose_heads(q_mem, ...);
        broadcast_heads(alpha, ...);  // [bs, s] → [bs*nh, s]
        broadcast_heads(theta, ...);

        // Step 4: Fused recurrence + M-norm clamp
        dgd_fused_forward_multihead(
            k_mem_ph, v_mem_ph, q_mem_ph,
            alpha_ph, theta_ph,
            m_initial, m_states, y_ph,
            alpha_out, theta_out,
            m_norm_max,
            s, hd, bs_mem, error_clip,
        );

        // Step 5: Reshape y back to d_model
        reshape_from_per_head(y_ph, ...);
    }
}
```

### cuda_ffi.rs

New FFI functions:

```rust
extern "C" {
    pub(crate) fn dgd_fused_forward_multihead_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        m_initial: *const f32, m_states: *mut f32,
        y: *mut f32,
        alpha_out: *mut f32, theta_out: *mut f32,
        m_norm_max: f32,
        seq_len: i32, d: i32, batch_size: i32,
        error_clip: f32,
    );

    pub(crate) fn titans_fused_forward_multihead_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, eta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_states: *mut f32, s_states: *mut f32,
        y: *mut f32,
        alpha_out: *mut f32, theta_out: *mut f32, eta_out: *mut f32,
        m_norm_max: f32,
        seq_len: i32, d: i32, batch_size: i32,
        error_clip: f32,
    );
}
```

### dispatch.rs

New dispatch functions wrapping the FFI calls, following existing patterns.

## Backward Path

**No fused backward kernel changes needed.** The backward path already handles multi-head
correctly (heads folded into batch dim). The forward cache (`GpuMemoryCache::Delta` /
`::Titans`) stores the same m_states/k_mem/v_mem/alpha/theta regardless of fused vs unfused.
The existing unfused backward kernels consume the cache identically.

## Files to Modify

| File | Change |
|------|--------|
| `core/kernels/dgd_forward.cu` | Add `dgd_fused_forward_multihead_kernel` + C wrapper |
| `core/kernels/titans_forward.cu` | Add `titans_fused_forward_multihead_kernel` + C wrapper |
| `core/src/cuda_ffi.rs` | Add 2 FFI declarations |
| `core/src/dispatch.rs` | Add 2 dispatch functions |
| `core/src/gpu_forward.rs` | Remove `nh == 1` guard, add multi-head fused dispatch |

## Validation

1. `cargo test --features cuda` — all existing tests pass
2. New test: `test_multihead_fused_matches_unfused` — run both paths on d=128 nh=2 s=32,
   compare y, m_states, alpha, theta element-wise (tolerance 1e-5)
3. New test: `test_multihead_fused_d768_nh12` — production config dimensions
4. Training equivalence: run 50 steps on same config, compare loss/gnorm to unfused baseline
5. Profile: measure tok/s before/after on A6000 at d=768 hd=64 s=4096

## Success Criteria

1. Fused forward path fires for all `num_heads > 1` configs (the `nh == 1` guard is gone)
2. Bit-identical output vs unfused path for hd=64 (12 heads) and hd=32 (24 heads)
3. M-norm clamp inlined — no per-batch-head clamp loop
4. `cargo test --features cuda` passes
5. Measurable tok/s improvement at d=768 hd=64 seq_len=4096

## Non-Goals

- No changes to gate computation (stays at d_model resolution, separate kernels)
- No fused backward kernel (backward already works for multi-head)
- No removal of nh=1 fused path (preserved for backward compat)
- No register tiling of M (future work, spec 39 Phase 4)
