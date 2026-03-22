# Per-Head Memory Kernels

## CONTRACT
- **Purpose**: Split monolithic d×d memory matrices into num_heads independent (head_dim × head_dim) matrices. This converts the backward kernel from 1 SM (grid=batch_size=1) to num_heads SMs (grid=num_heads), giving ~12× speedup at d=768/12 heads, and reduces total memory computation by 12× (num_heads × head_dim² vs d_model²).
- **Expects**: Existing Titans/Delta/Hebbian/DGD forward+backward CUDA kernels operating on [batch_size, seq_len, d] inputs with d×d memory matrix M. Rust dispatch in `dispatch.rs`, GPU forward/backward in `gpu_forward.rs`/`gpu_backward.rs`. All kernels currently launch with `dim3 grid(batch_size)` where batch_size=1.
- **Guarantees**: (1) All memory kernels launch with `dim3 grid(batch_size * num_heads)` — each head's recurrence runs on a separate SM. (2) Memory M is num_heads × (head_dim × head_dim) instead of 1 × (d × d). (3) Total computation per token drops from d² to num_heads × head_dim² (a factor of num_heads reduction). (4) Forward output is mathematically equivalent to independent per-head recurrences (no cross-head interaction through M — same as current since W_k_mem and W_v_mem already project into a shared d-space with no head mixing in the memory rule itself). (5) Backward produces correct gradients verified by FD checking against per-head Rust reference. (6) Existing single-head (num_heads=1) path produces bit-identical output.
- **Cost**: At d=768, nh=12: M state memory drops from 1154 MB to 96 MB per level (12× reduction). Backward per-token global memory access drops from 6.8 MB to 48 KB per SM. Forward computation drops from 589K to 49K FLOPs per token.
- **Trade-off**: Per-head memory means heads cannot share information through the memory matrix. This is not a regression — the current d×d matrix receives pre-projected k_mem/v_mem/q_mem that are already linear projections from d-space. The memory rule itself (M update, prediction, error) operates element-wise on these projections. Splitting into per-head is mathematically equivalent to the current setup when projection weights are block-diagonal — the full-rank W_k_mem ∈ R^{d×d} becomes num_heads independent W_k_mem^h ∈ R^{head_dim×head_dim}. The outer-loop optimizer can still learn cross-head interactions through W_k_mem/W_v_mem/W_q_mem.
- **Position**: `specs/infrastructure/cuda/08_per_head_memory_kernels.md`
- **Source**: Titans (2501.00663) Eq 32 defines M_t ∈ R^{d_in × d_in} — the paper does not prescribe d_in = d_model. MIRAS (2504.13173) framework treats memory structure as an independent knob. Multi-head attention universally splits Q/K/V into per-head subspaces — extending this to memory is standard practice. Cross-ref: `specs/infrastructure/cuda/05_large_dimension_kernel_restructuring.md` (strided loops for large d).

## Problem Statement

### Current Architecture
```
Input: embedded [batch_size, seq_len, d_model]
  → W_k_mem [d_model, d_model] → k_mem [bs, seq_len, d_model]
  → W_v_mem [d_model, d_model] → v_mem [bs, seq_len, d_model]
  → W_q_mem [d_model, d_model] → q_mem [bs, seq_len, d_model]

Memory: M ∈ R^{d_model × d_model}  (one matrix, 768×768 = 589,824 params)

CUDA kernel:
  grid  = (batch_size)      → batch_size=1 → 1 SM active
  block = (min(d_model, 1024)) → 1024 threads
  Each thread loops over d²/1024 = 576 elements per timestep
  Sequential recurrence: t = seq_len-1 down to 0
  Result: 1 SM processes 512 timesteps × 589K elements = serial bottleneck
```

### Why H100 Shows No Speedup
The H100 has 132 SMs with 3.35 TB/s aggregate bandwidth. With grid=(1), only 1 SM is active, achieving ~400 GB/s effective bandwidth. The backward kernel reads 3 × d² floats per timestep (M_t, S_t, d_M) = 6.8 MB. Over 512 timesteps = 3.5 GB of sequential global memory access on a single SM. This takes ~200ms regardless of GPU — the H100's advantage is parallelism across SMs, not per-SM speed.

### Per-Head Architecture
```
Input: embedded [batch_size, seq_len, d_model]
  → W_k_mem [d_model, d_model] → k_mem [bs, seq_len, d_model]
  → reshape to [bs * num_heads, seq_len, head_dim]

Memory: M_h ∈ R^{head_dim × head_dim}  (12 matrices, each 64×64 = 4,096 params)

CUDA kernel:
  grid  = (batch_size * num_heads)  → 1*12 = 12 SMs active
  block = (head_dim)                → 64 threads
  Each thread loops over head_dim²/64 = 64 elements per timestep
  Sequential recurrence: same, but 12 independent recurrences in parallel
  Result: 12 SMs each process 512 timesteps × 4K elements = 12× speedup
```

## Detailed Design

### Phase 1: Kernel Modifications (CUDA)

All four memory rule kernel pairs (forward + backward) follow the same transformation:

#### Kernel Signature Change
```cuda
// BEFORE:
__global__ void titans_backward_kernel(
    const float* k_mem,       // [batch_size, seq_len, d]
    ...
    float* d_M,               // [batch_size, d*d]
    int seq_len, int d, float error_clip)
{
    int b = blockIdx.x;       // batch index
    int dd = d * d;
    k_mem += b * seq_len * d;
    ...

// AFTER:
__global__ void titans_backward_kernel(
    const float* k_mem,       // [batch_size * num_heads, seq_len, head_dim]
    ...
    float* d_M,               // [batch_size * num_heads, head_dim * head_dim]
    int seq_len, int d, float error_clip)  // d = head_dim now
{
    int bh = blockIdx.x;      // batch_head index (0..batch_size*num_heads-1)
    int dd = d * d;
    k_mem += bh * seq_len * d;
    ...
```

The kernel body is **unchanged** — it already operates on generic `d`. We just pass `d = head_dim` instead of `d = d_model`, and the grid covers `batch_size * num_heads` independent recurrences.

#### Launch Wrapper Change
```cuda
// BEFORE:
extern "C" void titans_backward_f32_cuda(..., int seq_len, int d, int batch_size, ...) {
    int block_size = (d < 1024) ? d : 1024;
    // ... power-of-2 rounding ...
    dim3 grid(batch_size);
    dim3 block(block_size);

// AFTER:
extern "C" void titans_backward_f32_cuda(..., int seq_len, int d, int batch_size,
                                          int num_heads, ...) {
    // d is now head_dim (e.g., 64). block_size = head_dim.
    int block_size = (d < 1024) ? d : 1024;
    // ... power-of-2 rounding ...
    dim3 grid(batch_size * num_heads);
    dim3 block(block_size);
```

#### Files Modified
| File | Forward | Backward |
|------|---------|----------|
| `core/kernels/titans_forward.cu` | grid → bs*nh | — |
| `core/kernels/titans_backward.cu` | — | grid → bs*nh |
| `core/kernels/delta_forward.cu` | grid → bs*nh | — |
| `core/kernels/delta_backward.cu` | — | grid → bs*nh |
| `core/kernels/hebbian_forward.cu` | grid → bs*nh | — |
| `core/kernels/hebbian_backward.cu` | — | grid → bs*nh |
| `core/kernels/dgd_forward.cu` | grid → bs*nh | — |
| `core/kernels/dgd_backward.cu` | — | grid → bs*nh |
| `core/kernels/titans_chunkwise_forward.cu` | grid → bs*nh | — |
| `core/kernels/titans_chunkwise_backward.cu` | — | grid → bs*nh |
| `core/kernels/delta_chunkwise_forward.cu` | grid → bs*nh | — |
| `core/kernels/delta_chunkwise_backward.cu` | — | grid → bs*nh |
| `core/kernels/titans_phase2_forward.cu` | grid → bs*nh | — |
| `core/kernels/titans_phase2_backward.cu` | — | grid → bs*nh |
| `core/kernels/delta_phase2_forward.cu` | grid → bs*nh | — |
| `core/kernels/delta_phase2_backward.cu` | — | grid → bs*nh |

### Phase 2: Rust Dispatch Layer

#### Projection Weight Reshape
Currently W_k_mem ∈ R^{d×d}. Two options:

**Option A (reshape, no weight change)**: Keep W_k_mem as [d, d]. After matmul `k_mem = embedded @ W_k_mem^T → [bs*s, d]`, reshape to `[bs*nh, s, hd]` before passing to kernel. The matmul is unchanged. Only the view changes.

**Option B (block-diagonal weights)**: Store W_k_mem as [nh, hd, hd]. Use batched matmul. More efficient but breaks checkpoint compatibility.

**Decision: Option A** for this spec. The projection matmul is not the bottleneck (cuBLAS handles it efficiently). The reshape is a zero-copy pointer arithmetic operation. Checkpoint compatibility is preserved.

#### Memory State Reshape
M states change from `[batch_size, (seq_len+1), d*d]` to `[batch_size * num_heads, (seq_len+1), head_dim * head_dim]`.

Total memory: 1 × 513 × 589,824 × 4 bytes = 1,154 MB → 12 × 513 × 4,096 × 4 bytes = 96 MB per level.

#### dispatch.rs Changes
```rust
// Add num_heads parameter to all backward dispatch functions:
pub fn titans_backward_dd(
    k_mem: &GpuBuf<f32>,    // now [bs*nh, s, hd]
    ...,
    s: usize, d: usize,     // d = head_dim
    batch_size: usize,       // batch_size = original_bs * num_heads
    error_clip: f32,
) { ... }
```

The key insight: we multiply `batch_size * num_heads` before calling the kernel, and pass `head_dim` as `d`. The kernel code itself doesn't change.

#### gpu_forward.rs / gpu_backward.rs Changes
In `gpu_memory_forward` and `gpu_memory_backward`:
1. After computing k_mem/v_mem/q_mem via matmul (these remain [bs*s, d_model]), reshape to [bs*nh, s, hd]
2. Allocate M states as [bs*nh, (s+1), hd*hd] instead of [bs, (s+1), d*d]
3. Pass `d=hd` and `batch_size=bs*nh` to dispatch functions
4. After kernel returns, reshape gradient outputs back to [bs*s, d_model] for projection weight gradient accumulation

### Phase 3: Context Memory (CMS State)

Context memory (`context_m`, `context_s`) persists across forward calls:
- Currently: `[d*d]` per level = 2.2 MB
- Per-head: `[nh * hd * hd]` per level = 0.19 MB (same total as nh × hd²)

The storage size is identical: `d*d = nh * hd * hd` when `d = nh * hd`. The layout changes from one contiguous d×d block to nh contiguous hd×hd blocks. Since the kernel now indexes by `bh * dd` where `dd = hd*hd`, this just works with the flat buffer.

Wait — `d*d = 589,824` but `nh * hd*hd = 49,152`. These are NOT equal. `d² = (nh·hd)² = nh²·hd²`, but per-head uses `nh·hd²`. The ratio is `nh = 12`. So context_m shrinks by 12×.

Context memory allocation must change: `GpuBuf::zeros(nh * hd * hd)` instead of `GpuBuf::zeros(d * d)`.

The `copy_final_m` and `copy_initial_m` operations must be updated to copy nh × (hd × hd) elements.

### Phase 4: Checkpoint Compatibility

Existing checkpoints store M as d×d. New checkpoints store as nh × (hd × hd).

**Migration**: On load, if checkpoint has d×d context_m and config has num_heads > 1, extract the block-diagonal: for each head h, copy the hd×hd subblock at rows [h*hd..(h+1)*hd], cols [h*hd..(h+1)*hd]. Off-diagonal blocks are discarded (they represented cross-head memory interaction that per-head cannot express).

**Forward compatibility**: New checkpoints include a `memory_layout: "per_head"` field. Old checkpoints without this field are assumed `"monolithic"`.

## Performance Projections

### At d=768, nh=12, hd=64, s=512, k=4

| Metric | Current (d×d) | Per-Head (hd×hd) | Speedup |
|--------|--------------|-------------------|---------|
| SMs active (backward) | 1 | 12 | 12× |
| Elements per timestep | 589,824 | 4,096 × 12 = 49,152 | 12× less |
| Global memory per token | 6.8 MB | 48 KB × 12 = 576 KB | 12× less |
| M state memory per level | 1,154 MB | 96 MB | 12× less |
| Backward time per level (est.) | 200 ms | ~17 ms | ~12× |
| Total step time (est.) | 7,169 ms | ~1,000 ms | ~7× |

### At d=768, nh=12, on H100
With 12 SMs active, the H100 can deliver 12× the per-SM bandwidth. Plus the 12× reduction in total work means each SM finishes 12× faster. Combined: backward kernel should be **~144× faster** in raw compute, but will be limited by kernel launch overhead and other pipeline stages. Realistic estimate: **10-15× total step speedup**.

## Code Smells

- **CS-18**: Kernel changes are CUDA tier. Dispatch/reshape changes are Rust tier. Orchestration unchanged (Python tier). Correct tier separation.
- **CS-32**: Observe-then-advance preserved — reshape happens before kernel call, results reshaped after.
- **CS-40**: Opt-in AD unchanged — tape records the same opaque VJP blocks.
- **CS-33/34**: MIRAS 4-knob framework unaffected — memory structure knob remains "matrix", just smaller matrices per head.

## Falsification Criteria

1. Per-head backward kernel is NOT faster than monolithic → head_dim is too small for GPU efficiency (unlikely at hd=64, but possible at hd=32)
2. Per-head model produces worse loss than monolithic at matched steps → cross-head memory interaction was load-bearing (would indicate d×d was learning off-diagonal cross-head patterns)
3. FD gradient check fails → reshape or index arithmetic error in the dispatch layer
4. Checkpoint migration loses information that matters → off-diagonal blocks of d×d M were non-negligible

## Implementation Order

1. Write CUDA kernel changes (trivial — add num_heads param to launch wrapper, change grid dim)
2. Update `cuda_ffi.rs` FFI signatures
3. Update `dispatch.rs` dispatch functions
4. Update `gpu_forward.rs` — reshape k/v/q_mem, allocate per-head M states
5. Update `gpu_backward.rs` — same reshape, gradient accumulation
6. Update context memory allocation and copy_final_m/copy_initial_m
7. Update checkpoint save/load for per-head layout
8. FD gradient verification at small scale (d=16, nh=2, hd=8)
9. Build and profile on A6000 and H100
