# DGD CUDA Kernel Pair

```text
CONTRACT
  Purpose:    GPU-accelerated DGD inner-loop: dgd_forward.cu and dgd_backward.cu.
              DGD is the hottest path in HOPE — every token at every active CMS level
              calls it. The kernel pair accelerates dgd_step (non-momentum) and
              dgd_momentum_step on CUDA, matching the Rust reference within tolerance.
  Expects:    Rust dgd_step() and dgd_step_backward() signatures verified (GAP-I).
              Existing kernel infrastructure: cc build, GpuBuf, Backend dispatch,
              GpuMemoryCache enum, OpaqueVjp trait.
  Guarantees: Forward parity ≤ 1e-5, backward parity ≤ 1e-4 vs Rust reference.
              fp32 throughout (non-negotiable for inner-loop state).
              Fat binary: sm_86/89/90 SASS + compute_86 PTX fallback.
  Cost:       Per token: O(d²) — same as Rust reference. Single-block sequential
              over tokens (inherent data dependency: M_{t+1} depends on M_t).
  Trade-off:  GPU parallelism is WITHIN each token (d² threads across M elements),
              not across tokens. Shared memory holds M (d²), so max d ≈ 32 for
              48 KB smem. Larger d requires the checkpointed variant.
  Position:   specs/infrastructure/cuda/02_dgd_kernels.md
  Source:     HOPE (2512.24695) §4.5, Appendix C, Eq 88, Eq 121;
              core/src/dgd.rs (Rust reference, verified by GAP-I)
```

## 1. Kernel Variants

Four kernel entry points across two files:

| File | Kernel | Purpose |
|---|---|---|
| `dgd_forward.cu` | `dgd_forward_kernel` | Non-momentum DGD recurrence + readout |
| `dgd_forward.cu` | `dgd_forward_ckpt_kernel` | Checkpointed variant (stores M every C steps) |
| `dgd_backward.cu` | `dgd_backward_kernel` | Full backward (all M states cached) |
| `dgd_backward.cu` | `dgd_backward_segment_kernel` | Segment backward for checkpointed forward |

**Momentum variant (deferred)**: Separate kernels `dgd_momentum_forward_kernel` and
`dgd_momentum_backward_kernel` in the same files. These add the S accumulator
(same pattern as Titans adds S over Delta). Planned for a future PR.

**Sherman-Morrison**: NOT a separate kernel. Deferred to a future spec. Rationale:
SM requires keys to be normalized (||k|| ≈ 1) and is L2-only. The iterative DGD kernel
handles all attentional biases uniformly. SM can be added as an optimization later
without changing the kernel interface — it would be a runtime branch based on a
`use_sherman_morrison` flag, or a separate kernel dispatched from Rust.

## 2. Forward Kernel Signatures

### 2.1 Non-Momentum Forward

```cpp
extern "C" void dgd_forward_f32_cuda(
    const float* k_mem,      // [seq_len, d] — projected keys
    const float* v_mem,      // [seq_len, d] — projected values
    const float* q_mem,      // [seq_len, d] — projected queries (for readout)
    const float* alpha,      // [seq_len]    — retention gate (sigmoid output)
    const float* theta,      // [seq_len]    — learning rate gate (softplus output)
    const float* m_initial,  // [d*d]        — M_0 (row-major)
    float* m_states,         // [(seq_len+1)*d*d] — all M states for backward
    float* y,                // [seq_len, d] — readout output
    int seq_len, int d
);
```

**Math per token** (identical to Delta Rule — DGD generalizes it):
```text
prediction[i] = sum_j M[i,j] * k_t[j]
error[i]      = prediction[i] - v_t[i]
M[i,j]        = (1 - alpha_t) * M[i,j] - theta_t * error[i] * k_t[j]
store M to m_states[(t+1)*d*d ..]
y_t[i]        = sum_j M[i,j] * q_t[j]
```

### 2.2 Momentum Forward

```cpp
extern "C" void dgd_momentum_forward_f32_cuda(
    const float* k_mem,      // [seq_len, d]
    const float* v_mem,      // [seq_len, d]
    const float* q_mem,      // [seq_len, d]
    const float* alpha,      // [seq_len]    — retention gate
    const float* theta,      // [seq_len]    — learning rate gate
    const float* beta,       // [seq_len]    — momentum gate
    const float* m_initial,  // [d*d]
    const float* s_initial,  // [d*d]        — momentum accumulator S_0
    float* m_states,         // [(seq_len+1)*d*d]
    float* s_states,         // [(seq_len+1)*d*d] — all S states for backward
    float* y,                // [seq_len, d]
    int seq_len, int d
);
```

**Math per token** (from `dgd_momentum_step` in `core/src/dgd.rs`):
```text
prediction[i] = sum_j M[i,j] * k_t[j]
error[i]      = prediction[i] - v_t[i]
S[i,j]        = beta_t * S[i,j] + theta_t * error[i] * k_t[j]
M[i,j]        = (1 - alpha_t) * M[i,j] - S[i,j]
store M to m_states[(t+1)*d*d ..]
store S to s_states[(t+1)*d*d ..]
y_t[i]        = sum_j M[i,j] * q_t[j]
```

**Sign convention difference vs Titans**: Titans uses `S = eta*S - theta*outer(error,k)`
then `M = (1-alpha)*M + S`. DGD momentum uses `S = beta*S + theta*outer(error,k)` then
`M = (1-alpha)*M - S`. The signs differ — these are NOT interchangeable with the Titans
kernel.

### 2.3 Checkpointed Forward (Non-Momentum)

```cpp
extern "C" void dgd_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* m_initial,
    float* m_states,         // [num_ckpt * d*d]
    float* y,
    int seq_len, int d, int checkpoint_interval
);
```

Stores M only at checkpoint boundaries (every `checkpoint_interval` steps) + final.
Backward uses segment kernel to recompute M from checkpoints.

### 2.4 Checkpointed Forward (Momentum)

```cpp
extern "C" void dgd_momentum_forward_ckpt_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* beta,
    const float* m_initial, const float* s_initial,
    float* m_states, float* s_states,    // [num_ckpt * d*d] each
    float* y,
    int seq_len, int d, int checkpoint_interval
);
```

## 3. Backward Kernel Signatures

### 3.1 Non-Momentum Backward

```cpp
extern "C" void dgd_backward_f32_cuda(
    const float* k_mem,      // [seq_len, d]
    const float* v_mem,      // [seq_len, d]
    const float* q_mem,      // [seq_len, d]
    const float* alpha,      // [seq_len]
    const float* theta,      // [seq_len]
    const float* m_states,   // [(seq_len+1)*d*d] — from forward cache
    const float* d_y,        // [seq_len, d]      — upstream gradient
    float* d_k_mem,          // [seq_len, d]      — output gradient
    float* d_v_mem,          // [seq_len, d]
    float* d_q_mem,          // [seq_len, d]
    float* d_alpha,          // [seq_len]
    float* d_theta,          // [seq_len]
    float* d_m_initial,      // [d*d]             — gradient on M_0
    int seq_len, int d
);
```

**Reverse loop per token** (from `dgd_step_backward` in `core/src/dgd.rs`):
```text
d_M += outer(d_y_t, q_t)                      // readout contribution
d_q_t[j] = sum_i M_{t+1}[i,j] * d_y_t[i]     // query gradient

// Recompute forward intermediates from cached M_t
prediction = M_t @ k_t
error = prediction - v_t

// Gate gradients (parallel reduction)
d_alpha_t = -frobenius_dot(M_t, d_M)
d_theta_t = -frobenius_dot(outer(error, k_t), d_M)

// Input gradients
d_error[i] = sum_j (-theta_t * d_M[i,j]) * k_t[j]
d_k_t[j]   = sum_i (-theta_t * d_M[i,j]) * error[i]
            + sum_i M_t[i,j] * d_error[i]
d_v_t[i]   = -d_error[i]

// Propagate d_M backward
d_M = (1-alpha_t) * d_M + outer(d_error, k_t)
```

This is structurally identical to `delta_backward_kernel` — same gradient equations
because DGD and Delta Rule share the same update math. The analytical derivation
matches `dgd_step_backward` in `core/src/dgd.rs` lines 101–188.

### 3.2 Momentum Backward

```cpp
extern "C" void dgd_momentum_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* beta,
    const float* m_states,   // [(seq_len+1)*d*d]
    const float* s_states,   // [(seq_len+1)*d*d]
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_beta,
    float* d_m_initial, float* d_s_initial,
    int seq_len, int d
);
```

Adds d_beta output and d_s_initial. d_S propagates through the reverse loop
analogously to Titans backward (but with DGD sign convention).

### 3.3 Segment Backward (Non-Momentum)

```cpp
extern "C" void dgd_backward_segment_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* m_states,   // segment-local: [(seg_len+1)*d*d]
    const float* d_y,
    const float* d_m_seed,   // [d*d] — seed from subsequent segment
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta,
    float* d_m_out,          // [d*d] — propagated to earlier segment
    int t_start, int t_end, int d
);
```

### 3.4 Segment Backward (Momentum)

```cpp
extern "C" void dgd_momentum_backward_segment_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* beta,
    const float* m_states, const float* s_states,
    const float* d_y,
    const float* d_m_seed, const float* d_s_seed,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_beta,
    float* d_m_out, float* d_s_out,
    int t_start, int t_end, int d
);
```

## 4. Memory Layout

All tensors are row-major, fp32, in device global memory (via `GpuBuf<f32>`).

| Tensor | Shape | Layout | Notes |
|---|---|---|---|
| M | [d, d] | Row-major, contiguous | Same as existing `GpuContextState.memory` buffers |
| S | [d, d] | Row-major, contiguous | Momentum only. Same shape/lifetime as M |
| k, v, q | [seq_len, d] | Token-major | Token t at offset `t * d` |
| alpha, theta, beta | [seq_len] | Per-token scalars | Gate outputs, computed by Rust projection code |
| m_states | [(seq_len+1), d, d] | Token-major | M_0 at offset 0, M_t at offset `t * d * d` |
| s_states | [(seq_len+1), d, d] | Token-major | Momentum only |
| y | [seq_len, d] | Token-major | Readout output |

**Relationship to GpuContextState**: The `memory` field holds per-level M matrices.
For DGD, `GpuContextState.memory[level_idx]` is a `GpuBuf<f32>` of length `d*d`.
This is copied to `m_initial` at kernel launch and updated from the final M state
after the kernel completes.

## 5. Grid, Block, and Shared Memory Strategy

### 5.1 Launch Configuration

Same pattern as all existing matrix-memory kernels:

```cpp
int dd = d * d;
int block_size = (dd < 1024) ? dd : 1024;
if (block_size < d) block_size = d;  // ensure ≥ d for matvec

// Round to power of 2 for tree reduction (backward only)
int rounded = 1;
while (rounded < block_size) rounded <<= 1;
if (rounded > 1024) rounded = 1024;
block_size = rounded;  // backward needs this; forward can use non-rounded

dim3 grid(1);           // single block — sequential over tokens
dim3 block(block_size);
```

**Why single block**: The M recurrence `M_{t+1} = f(M_t, k_t, v_t)` is inherently
sequential across tokens. Intra-token parallelism across d² matrix elements is the
only available parallelism. Grid(1) ensures shared memory M is coherent.

**Max d**: Shared memory ≤ 48 KB (sm_86 default). For non-momentum:
`(d² + 2d) * 4 bytes ≤ 48K` → d ≤ 108. For momentum: `(2d² + 2d) * 4 ≤ 48K` → d ≤ 76.
Production models use d_head ∈ {32, 64, 128}. d=128 exceeds smem for momentum;
use checkpointed variant or request extended shared memory (100 KB on sm_90).

### 5.2 Shared Memory Layout — Non-Momentum Forward

```cpp
extern __shared__ float smem[];
float* M         = smem;              // [d*d]
float* prediction = smem + dd;        // [d]
float* error_buf  = smem + dd + d;    // [d]
// Total: (d*d + 2*d) * sizeof(float)
```

### 5.3 Shared Memory Layout — Momentum Forward

```cpp
extern __shared__ float smem[];
float* M         = smem;              // [d*d]
float* S         = smem + dd;         // [d*d]
float* prediction = smem + 2*dd;      // [d]
float* error_buf  = smem + 2*dd + d;  // [d]
// Total: (2*d*d + 2*d) * sizeof(float)
```

### 5.4 Shared Memory Layout — Non-Momentum Backward

```cpp
extern __shared__ float smem[];
float* d_M        = smem;                    // [d*d]
float* prediction = smem + dd;               // [d]
float* error_buf  = smem + dd + d;           // [d]
float* d_error    = smem + dd + 2*d;         // [d]
float* reduce_buf = smem + dd + 3*d;         // [blockDim.x]
// Total: (d*d + 3*d + blockDim.x) * sizeof(float)
```

### 5.5 Shared Memory Layout — Momentum Backward

```cpp
extern __shared__ float smem[];
float* d_M        = smem;                    // [d*d]
float* d_S        = smem + dd;               // [d*d]
float* prediction = smem + 2*dd;             // [d]
float* error_buf  = smem + 2*dd + d;         // [d]
float* d_error    = smem + 2*dd + 2*d;       // [d]
float* reduce_buf = smem + 2*dd + 3*d;       // [blockDim.x]
// Total: (2*d*d + 3*d + blockDim.x) * sizeof(float)
```

## 6. Parallel Reduction for Gate Gradients

Gate gradients (d_alpha, d_theta, d_beta) are scalar sums over d² elements.
Use the existing tree-reduction pattern from `delta_backward_kernel`:

```cpp
// d_alpha_t = -frobenius_dot(M_t, d_M)
{
    float local_sum = 0.0f;
    for (int idx = tid; idx < dd; idx += blockDim.x) {
        local_sum += m_t[idx] * d_M[idx];
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_alpha[t] = -reduce_buf[0];
    __syncthreads();
}
```

Requires `blockDim.x` to be a power of 2 (hence the rounding in launch config).

## 7. atomicAdd Requirements

**None for DGD**. The single-block design means all threads share the same d_M in
shared memory. Gradient scatter for d_k, d_v is computed per-token and written
directly to global memory at known offsets `[t * d + j]`. No cross-token accumulation
races exist because the reverse loop processes one token at a time.

This matches the existing Delta/Titans/Hebbian backward kernels — none of them
use atomicAdd either. The atomicAdd pattern documented in MEMORY.md applies to
the multi-head SWA attention backward, not memory rule kernels.

## 8. Numerical Precision

| Data | Precision | Rationale |
|---|---|---|
| M, S states | fp32 | Non-negotiable. bf16 drift corrupts memory after ~100 steps |
| k, v, q inputs | fp32 | Projected in Rust from bf16 SWA outputs. Already fp32 at kernel boundary |
| alpha, theta, beta | fp32 | Gate outputs (sigmoid/softplus), always fp32 |
| m_states, s_states | fp32 | Cache for backward |
| y output | fp32 | Readout, mixed back to bf16 by caller if needed |
| All backward | fp32 | Gradient accumulation must be fp32 |

**No bf16 anywhere in these kernels**. The bf16↔fp32 boundary is at the SWA
attention level (handled by existing `swa_forward.cu`), not at the memory rule level.

## 9. Cache Struct (GpuMemoryCache Extension)

Add two variants to the existing `GpuMemoryCache` enum in `core/src/gpu_forward.rs`:

```rust
pub enum GpuMemoryCache {
    // ... existing Delta, Titans, Hebbian, DeltaCkpt, TitansCkpt, HebbianCkpt ...

    DGD {
        k_mem: GpuBuf<f32>,     // [s, d]
        v_mem: GpuBuf<f32>,     // [s, d]
        q_mem: GpuBuf<f32>,     // [s, d]
        alpha: GpuBuf<f32>,     // [s]
        theta: GpuBuf<f32>,     // [s]
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
    },
    DGDMomentum {
        k_mem: GpuBuf<f32>,     // [s, d]
        v_mem: GpuBuf<f32>,     // [s, d]
        q_mem: GpuBuf<f32>,     // [s, d]
        alpha: GpuBuf<f32>,     // [s]
        theta: GpuBuf<f32>,     // [s]
        beta: GpuBuf<f32>,      // [s]
        m_states: GpuBuf<f32>,  // [(s+1)*d*d]
        s_states: GpuBuf<f32>,  // [(s+1)*d*d]
    },
    DGDCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,  // [num_ckpt * d*d]
        checkpoint_interval: usize,
    },
    DGDMomentumCkpt {
        k_mem: GpuBuf<f32>,
        v_mem: GpuBuf<f32>,
        q_mem: GpuBuf<f32>,
        alpha: GpuBuf<f32>,
        theta: GpuBuf<f32>,
        beta: GpuBuf<f32>,
        m_checkpoints: GpuBuf<f32>,
        s_checkpoints: GpuBuf<f32>,
        checkpoint_interval: usize,
    },
}
```

## 10. Wengert Tape Integration

### 10.1 OpaqueKey Extension

Add `DGD` variant to `OpaqueKey` enum in `core/src/tape.rs`:

```rust
pub enum OpaqueKey {
    // ... existing ...
    DGD,          // non-momentum DGD
    DGDMomentum,  // DGD with momentum accumulator
    FrozenDGD,    // read-only DGD (frozen CMS levels)
}
```

### 10.2 Dispatch in gpu_cms_forward / gpu_cms_backward

The GPU forward path dispatches based on rule kind. DGD adds a new branch:

```rust
// In gpu_cms_forward (per-level dispatch)
match rule_kind {
    MemoryRuleKind::DeltaRule => {
        // existing delta_forward_f32_cuda call
    }
    MemoryRuleKind::DGD => {
        // dgd_forward_f32_cuda — same signature as delta
        unsafe { dgd_forward_f32_cuda(...) }
    }
    MemoryRuleKind::DGDMomentum => {
        // dgd_momentum_forward_f32_cuda — adds beta, s_initial, s_states
        unsafe { dgd_momentum_forward_f32_cuda(...) }
    }
    // ...
}
```

### 10.3 OpaqueVjp Implementation

DGD implements the `OpaqueVjp` trait. The `record_on_tape` method:
1. Runs the GPU forward kernel
2. Records inputs (k, v, q, alpha, theta, m_initial) and outputs (y, m_final) as tape buffers
3. Stores the `GpuMemoryCache::DGD` as saved state for backward

The registered backward function calls `dgd_backward_f32_cuda` with the saved cache.

## 11. Architecture Targets

Match existing fat binary pattern in `core/build.rs`:

```rust
// In build.rs, add to the cc::Build chain:
.file("kernels/dgd_forward.cu")
.file("kernels/dgd_backward.cu")
```

The existing `-gencode` flags compile for:
- `sm_86` (SASS): RTX 3090, A6000
- `sm_89` (SASS): RTX 4090, Ada Lovelace
- `sm_90` (SASS): H100, Hopper
- `compute_86` (PTX): Forward-compatible fallback for sm ≥ 86

No DGD-specific arch considerations. The kernels use standard fp32 ops, shared memory,
and `__syncthreads()` — all available since sm_30.

## 12. Self-Referential / Batched Mode

### 12.1 Current State

HOPE Phase 2 (GAP-L/M/N, currently open) runs 6 component memories, each calling DGD
independently. With the current single-block design, this means 6 sequential kernel
launches per token position.

### 12.2 Decision: Sequential First, Batched Later

**Do NOT implement batched mode in this spec**. Rationale:

1. **GAP-L/M/N are not yet implemented**. The exact number of components, their
   interdependencies, and whether they share projections is still being designed.
2. **Sequential launches are correct**. 6 kernel launches add ~6 × launch overhead
   (typically 5-15 μs each on modern GPUs), totaling ~30-90 μs per position.
   For d=64 with ~4K FLOPs per kernel, compute time dominates launch overhead.
3. **Batched mode is straightforward** when needed: change `grid(1)` to `grid(n_components)`,
   add component index to all pointer offsets. This is a mechanical change that
   does not affect the kernel math.

### 12.3 Future Batched Kernel (Sketch)

When GAP-L/M/N land, a batched variant can be added:

```cpp
extern "C" void dgd_forward_batched_f32_cuda(
    const float* k_mem,      // [n_components, seq_len, d]
    const float* v_mem,      // [n_components, seq_len, d]
    const float* q_mem,      // [n_components, seq_len, d]
    const float* alpha,      // [n_components, seq_len]
    const float* theta,      // [n_components, seq_len]
    const float* m_initial,  // [n_components, d*d]
    float* m_states,         // [n_components, (seq_len+1)*d*d]
    float* y,                // [n_components, seq_len, d]
    int n_components, int seq_len, int d
);
// grid(n_components), block(min(d*d, 1024))
// Each block handles one component independently
```

This is a spec placeholder, not a deliverable for S3b-M5.

## 13. Relationship to Existing Delta Rule Kernel

DGD non-momentum and Delta Rule perform **identical math**:

```text
M = (1 - alpha) * M - theta * (M @ k - v) @ k^T
```

The Rust code separates them because:
- DGD is bias-agnostic: the caller can transform the error before passing to `dgd_update`
- DGD has the Sherman-Morrison fast path
- DGD has the momentum variant with different sign conventions than Titans

For the CUDA kernel, the non-momentum DGD kernel will be structurally identical
to `delta_forward_kernel`. Two implementation options:

**Option A**: Reuse `delta_forward_f32_cuda` / `delta_backward_f32_cuda` for
non-momentum DGD. Add separate kernels only for momentum.

**Option B**: Create separate `dgd_forward.cu` / `dgd_backward.cu` files with
their own kernel functions, even though the non-momentum math is identical.

**Recommendation**: Option B. Separate files maintain 1:1 correspondence between
Rust primitives and CUDA kernels. The code duplication is mechanical and small.
When non-L2 attentional biases are added later (requiring a bias-transformed error),
the DGD kernel will diverge from Delta. Starting with separate files avoids a
future refactor.

## 14. FFI Declarations (cuda_ffi.rs)

Add to `core/src/cuda_ffi.rs`:

```rust
#[cfg(feature = "cuda")]
extern "C" {
    // Non-momentum
    pub(crate) fn dgd_forward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32
    );

    pub(crate) fn dgd_backward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_initial: *mut f32,
        seq_len: i32, d: i32
    );

    // Momentum
    pub(crate) fn dgd_momentum_forward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, beta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_states: *mut f32, s_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32
    );

    pub(crate) fn dgd_momentum_backward_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, beta: *const f32,
        m_states: *const f32, s_states: *const f32,
        d_y: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_beta: *mut f32,
        d_m_initial: *mut f32, d_s_initial: *mut f32,
        seq_len: i32, d: i32
    );

    // Checkpointed
    pub(crate) fn dgd_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, m_initial: *const f32,
        m_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32
    );

    pub(crate) fn dgd_momentum_forward_ckpt_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, beta: *const f32,
        m_initial: *const f32, s_initial: *const f32,
        m_states: *mut f32, s_states: *mut f32, y: *mut f32,
        seq_len: i32, d: i32, checkpoint_interval: i32
    );

    // Segment backward
    pub(crate) fn dgd_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32,
        m_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_m_out: *mut f32,
        t_start: i32, t_end: i32, d: i32
    );

    pub(crate) fn dgd_momentum_backward_segment_f32_cuda(
        k_mem: *const f32, v_mem: *const f32, q_mem: *const f32,
        alpha: *const f32, theta: *const f32, beta: *const f32,
        m_states: *const f32, s_states: *const f32, d_y: *const f32,
        d_m_seed: *const f32, d_s_seed: *const f32,
        d_k_mem: *mut f32, d_v_mem: *mut f32, d_q_mem: *mut f32,
        d_alpha: *mut f32, d_theta: *mut f32, d_beta: *mut f32,
        d_m_out: *mut f32, d_s_out: *mut f32,
        t_start: i32, t_end: i32, d: i32
    );
}
```

## 15. Testing Requirements

| Test | Tolerance | Method |
|---|---|---|
| Forward parity (non-momentum) | 1e-5 per-element | CUDA vs Rust `dgd_step` |
| Forward parity (momentum) | 1e-5 per-element | CUDA vs Rust `dgd_momentum_step` |
| Backward parity (non-momentum) | 1e-4 per-element | CUDA vs Rust `dgd_step_backward` |
| Backward parity (momentum) | 1e-4 per-element | CUDA vs Rust backward (not yet impl) |
| End-to-end loss | 1e-5 | gpu_cms_forward with DGD vs CPU reference |
| Checkpointed forward | exact match | ckpt variant vs full-cache variant |
| Segment backward | 1e-4 | segment variant vs full backward |

## 16. Acceptance Criteria

- [ ] `dgd_forward.cu` + `dgd_backward.cu` exist with all kernel variants
- [ ] Forward + backward kernel signatures match this spec exactly
- [ ] Grid/block: `grid(1), block(min(d*d, 1024))` with power-of-2 rounding for backward
- [ ] Shared memory layouts match Section 5
- [ ] Non-momentum forward matches Rust `dgd_step()` within 1e-5
- [ ] Non-momentum backward matches Rust `dgd_step_backward()` within 1e-4
- [ ] Momentum forward matches Rust `dgd_momentum_step()` within 1e-5
- [ ] Gate gradients (d_alpha, d_theta, d_beta) use tree reduction
- [ ] `GpuMemoryCache::DGD` and `DGDMomentum` variants added
- [ ] `OpaqueKey::DGD` registered in tape
- [ ] FFI declarations in `cuda_ffi.rs`
- [ ] Build.rs updated to compile new .cu files
- [ ] Fat binary: sm_86/89/90 + PTX
- [ ] Sherman-Morrison deferred (documented, not implemented)
- [ ] Batched mode deferred (documented with future sketch)
