# Hopper (sm_90a) Kernel Optimization — H100/H200 Native Performance

```text
CONTRACT
  Purpose:    Add sm_90a native SASS compilation and Hopper-specific kernel
              optimizations (TMA prefetch, cp.async vector staging, double-buffered
              M row loads) to the existing CUDA kernel infrastructure. Target:
              Titans LMM forward and backward kernels first, then propagate to
              Delta, Hebbian, DGD. Ensures H100 and H200 GPUs run native code
              instead of JIT-compiled PTX fallback.
  Expects:    17 existing CUDA kernel files (core/kernels/*.cu) compiled via cc
              crate in build.rs. Fat binary currently targets sm_86/89/90 + PTX.
              Dispatch.rs selects Backend::{CudaNative, CudaPtx, RustReference}.
              All memory update kernels use global memory for M/S matrices with
              shared memory only for small working buffers (prediction[d], error[d]).
              CUDA Toolkit 12.8+.
  Guarantees: Numerical equivalence: sm_90a kernels produce identical output to
              sm_86/89/90 kernels within existing tolerances (forward ≤ 1e-5,
              backward ≤ 1e-4). No behavioral change — only latency reduction.
              Fat binary includes sm_90a SASS alongside existing architectures.
              Backward compatibility: sm_86/89 kernels are not modified.
              fp32 throughout for inner-loop state (non-negotiable).
  Cost:       Build time: ~10-15% longer (one additional gencode target).
              Binary size: ~15% larger (additional SASS in fat binary).
              No runtime cost — sm_90a kernels replace PTX JIT on Hopper GPUs.
  Trade-off:  TMA and cp.async add code complexity to kernel source. The
              non-Hopper code paths (sm_86/89) remain unchanged — Hopper-specific
              code is #ifdef-guarded by __CUDA_ARCH__ >= 800 (cp.async available
              on Ampere sm_80+, not just Hopper). This means two code
              paths per kernel to maintain. The alternative (single code path) leaves
              significant H100 performance on the table. At d=2048 (the 1B model
              target), memory latency hiding is critical.
  Position:   specs/infrastructure/cuda/04_hopper_kernel_optimization.md
  Source:     NVIDIA Hopper Tuning Guide (CUDA 12.8); NVIDIA CUDA Programming Guide
              §9 (CUDA Graphs), §B.28 (TMA); H100 Architecture Whitepaper.
              NL-Hecate: core/build.rs, core/src/dispatch.rs, core/kernels/titans_forward.cu.
              HADES: hecate_specs/hopper-kernel-optimization (this spec);
              extends hecate_specs/dgd_kernels, hecate_specs/swiglu_kernels,
              hecate_specs/cuda-graph-capture, hecate_specs/variant-tier-policy.
```

---

## 1. Motivation

Profiling at d=512 (scripts/profile_composition.py) shows the Titans memory update
kernel consumes >95% of per-step compute. At the current sm_90 generic target, H100
GPUs JIT-compile from PTX — losing 5-10% vs native SASS. More importantly, the
kernels use none of the Hopper-specific memory hierarchy features:

| Feature | Current (sm_86 idiom) | Hopper (sm_90a) | Impact |
|---|---|---|---|
| Global→shared loads | Explicit `smem[i] = global[i]` | TMA async transfer | Hides latency |
| Vector prefetch | None | `cp.async` pipeline | Overlaps compute+load |
| Shared memory | 48-100 KB | 228 KB (227 KB usable) | Larger tile sizes |
| Warp scheduling | All warps homogeneous | Producer/consumer specialization | Better occupancy |
| Register file | 64K regs/SM | 64K regs/SM (same) | No change |

At d=512, L1 cache may absorb most latency. At **d=2048** (the 1B HOPE target),
M is 16 MB per batch element and the L1/L2 hierarchy cannot help — explicit
prefetching and latency hiding become essential.

---

## 2. Scope

### In Scope

1. **sm_90a gencode target** in build.rs (native SASS for H100/H200)
2. **dispatch.rs update** — recognize sm_90/90a as CudaNative
3. **cp.async prefetch** for k/v/q vectors in Titans forward/backward kernels
4. **Double-buffered M row loads** via TMA in Titans forward kernel
5. **Blackwell (sm_100) PTX fallback** — add compute_90a PTX for future Hopper+
6. **Verification** on rented H100 pod (numerical equivalence + throughput)

### Out of Scope

- Tensor core (WMMA/MMA) usage — memory update kernels are matrix-vector, not GEMM
- Thread block clusters — not beneficial for single-batch sequential recurrence
- FP8 attention kernels — separate spec, orthogonal concern
- Multi-GPU / NCCL — explicitly deferred
- Backward graph capture — separate spec (cuda_graph_capture.md)

---

## 3. Build System Changes

### 3.1 build.rs — Add sm_90a

```rust
// Architecture-specific SASS (native performance)
.flag("-gencode").flag("arch=compute_86,code=sm_86")    // A6000, RTX 3090
.flag("-gencode").flag("arch=compute_89,code=sm_89")    // RTX 4090
.flag("-gencode").flag("arch=compute_90a,code=sm_90a")  // H100, H200
// PTX fallback (JIT for future GPUs)
.flag("-gencode").flag("arch=compute_90a,code=compute_90a")  // Hopper+ PTX
.flag("-gencode").flag("arch=compute_86,code=compute_86")    // Ampere+ PTX (legacy)
```

**Why sm_90a not sm_90**: The `a` suffix enables TMA, thread block clusters, and
other Hopper-specific features. Without it, nvcc generates generic Hopper code that
doesn't use TMA instructions. The `a` is required for TMA intrinsics. cp.async uses
`__CUDA_ARCH__ >= 800` (available on Ampere+), while future TMA phases would
use `>= 900`.

**Binary size**: Adding one gencode target adds ~1-2 MB to the fat binary (17 kernels
× ~100 KB SASS each). Acceptable.

### 3.2 dispatch.rs — Recognize sm_90a

```rust
const NATIVE_SM_VERSIONS: &[i32] = &[86, 89, 90];
```

sm_90a reports as sm_version=90 at runtime (major=9, minor=0). The existing dispatch
already selects `CudaNative` for sm_version=90. **No change needed** in dispatch.rs —
the CUDA runtime selects sm_90a SASS from the fat binary automatically when it's
available.

However, add sm_100 to the comment and Blackwell awareness:

```rust
/// Known SM versions with embedded SASS in the fat binary.
/// sm_86: Ampere (A6000, RTX 3090)
/// sm_89: Ada Lovelace (RTX 4090)
/// sm_90a: Hopper (H100, H200) — includes TMA support
/// sm_100: Blackwell (B100, B200) — via PTX fallback until native SASS added
const NATIVE_SM_VERSIONS: &[i32] = &[86, 89, 90];
```

---

## 4. Kernel Optimization Strategy

### 4.1 Architecture Guard Pattern

All cp.async code is guarded by `__CUDA_ARCH__ >= 800` (Ampere and later):

```cuda
#if __CUDA_ARCH__ >= 800
    // Ampere+ path: cp.async double-buffered vector prefetch
    // cp.async available on sm_80+ (Ampere, Ada, Hopper, Blackwell)
    cp_async_prefetch_k_v_q(/* ... */);
#else
    // Pre-Ampere path: direct global loads (unchanged)
#endif
```

This ensures:
- Pre-Ampere kernels (< sm_80) are **byte-identical** to current — no regression risk
- sm_86/89/90a kernels all use cp.async prefetch — Ampere, Ada, Hopper benefit
- A single .cu source file produces both variants in the fat binary

### 4.2 cp.async Prefetch for k/v/q Vectors (Phase 1 — all kernels)

**Target**: titans_forward.cu lines 83-89, the per-token vector loads.

**Current pattern** (synchronous):
```rust
// Per-token loop: loads stall if vectors not in L1 cache
for t in 0..seq_len {
    let k_t: &[f32; D] = &k_mem[t * d..(t + 1) * d];
    let v_t: &[f32; D] = &v_mem[t * d..(t + 1) * d];
    let q_t: &[f32; D] = &q_mem[t * d..(t + 1) * d];
    // ... compute with k_t, v_t, q_t ...
}
```

**Ampere+ pattern** (async prefetch, double-buffered):
```rust
// Shared memory: two vector buffers for double-buffering
// where D: Dim, T: DeviceFloat
let mut k_buf: [[T; D]; 2];
let mut v_buf: [[T; D]; 2];
let mut q_buf: [[T; D]; 2];
let mut cur: usize = 0;

// Prefetch token 0 into buffer 0 (guard against seq_len==0)
if seq_len > 0 {
    cp_async_prefetch(&mut k_buf[0], &k_mem[0..d]);
    cp_async_prefetch(&mut v_buf[0], &v_mem[0..d]);
    cp_async_prefetch(&mut q_buf[0], &q_mem[0..d]);
    cp_async_commit_group();
}

for t in 0..seq_len {
    let next = 1 - cur;
    // Prefetch token t+1 into alternate buffer (overlaps with compute)
    if t + 1 < seq_len {
        cp_async_prefetch(&mut k_buf[next], &k_mem[(t + 1) * d..(t + 2) * d]);
        cp_async_prefetch(&mut v_buf[next], &v_mem[(t + 1) * d..(t + 2) * d]);
        cp_async_prefetch(&mut q_buf[next], &q_mem[(t + 1) * d..(t + 2) * d]);
        cp_async_commit_group();
    }
    // Wait for current buffer to be ready.
    // wait(1): one prefetch still in flight. wait(0): flush all on final iteration.
    if t + 1 < seq_len {
        cp_async_wait_group(1);
    } else {
        cp_async_wait_group(0);
    }
    sync_threads();

    // Compute with k_buf[cur], v_buf[cur], q_buf[cur]
    // ... prediction, error, S/M update, readout ...

    cur = next;
}
```

**Shared memory cost**: 2 buffers × 3 vectors × d × 4 bytes.
- d=512: 12 KB (trivial)
- d=2048: 48 KB (fits easily in 227 KB limit)

**Expected benefit**: Eliminates vector load stalls. At d=2048, each vector is 8 KB —
prefetching hides the ~400 cycle global memory latency.

### 4.3 Double-Buffered M Row Loads via TMA (Phase 2 — titans_forward only)

**Target**: The matrix-vector products `prediction = M @ k` and `y = M @ q`.

**Current pattern**: Each thread reads M row-by-row from global memory:
```rust
// prediction[tid] = dot(M[tid, :], k)
let mut sum: f32 = 0.0;
for j in 0..d {
    sum += m_states[m_t_off + tid * d + j] * k_t[j];
}
```

At d=2048, this is 8 KB per row × d rows = 32 MB of reads per matvec. The L2 cache
(50 MB on H100) can hold most of M, but the access pattern (strided by d elements
between threads) causes cache line waste.

**TMA pattern** (tile-based async load):
```rust
// TMA descriptor: set up once at kernel launch (host-side)
// Describes M as a 2D tensor: [d rows × d cols], row-major, fp32

// In kernel: load M rows in tiles of TILE_ROWS
// where D: Dim, T: DeviceFloat
const TILE_ROWS: usize = 32;
let mut m_tile: [[[T; D]; TILE_ROWS]; 2];  // double-buffered
let mut cur_tile: usize = 0;

// Load first tile
tma_load_2d(&mut m_tile[0], &tma_desc_m, 0, 0, TILE_ROWS, d);
tma_commit();

let mut row_start = 0;
while row_start < d {
    let next_tile = 1 - cur_tile;
    // Prefetch next tile
    if row_start + TILE_ROWS < d {
        tma_load_2d(&mut m_tile[next_tile], &tma_desc_m,
                    row_start + TILE_ROWS, 0, TILE_ROWS, d);
        tma_commit();
    }
    tma_wait(1);
    sync_threads();

    // Compute partial predictions from this tile
    for r in 0..TILE_ROWS {
        let row = row_start + r;
        if row >= d { break; }
        if tid == row {
            let mut sum: f32 = 0.0;
            for j in 0..d {
                sum += m_tile[cur_tile][r][j] * k_shared[j];
            }
            prediction[row] = sum;
        }
    }
    cur_tile = next_tile;
    row_start += TILE_ROWS;
}
```

**Shared memory cost**: 2 × TILE_ROWS × d × 4 bytes.
- d=512, TILE=32: 128 KB (fits in 227 KB)
- d=2048, TILE=8: 128 KB (fits, but smaller tiles)
- TILE_ROWS is tuned per d to fit shared memory budget

**Expected benefit**: TMA loads bypass the register file entirely and arrive in
shared memory without SM involvement. The SM continues computing the previous tile's
dot products while the next tile loads. At d=2048, this should reduce the matvec
portion of the kernel by 30-50%.

**Caveat**: The M *update* (lines 113-119) writes back to global memory. TMA
store-back is possible but adds complexity. Phase 2 focuses on read-side only.

### 4.4 M Update Write Coalescing (Phase 3 — optional)

The S/M update loop (titans_forward.cu lines 113-119) writes d² elements back to
global memory with thread-strided access:

```rust
for idx in (tid..dd).step_by(block_dim_x) {
    s_states[m_next_off + idx] = s_new;
    m_states[m_next_off + idx] = retention * m_states[m_t_off + idx] + s_new;
}
```

This is already well-coalesced (consecutive threads write consecutive addresses).
The main optimization opportunity is **async store** — writing m_states[t+1] via TMA
store while beginning the readout (`y = M_{t+1} @ q`) from the shared-memory copy
that was just computed. Deferred to Phase 3 pending profiling data from Phase 2.

---

## 5. Kernel-by-Kernel Priority

| Priority | Kernel | Phase | Rationale |
|---|---|---|---|
| **P0** | `titans_forward.cu` | 1+2 | Hottest path, >50% of step time |
| **P0** | `titans_backward.cu` | 1 | Second hottest, recomputes prediction/error |
| **P1** | `dgd_forward.cu` | 1 | HOPE staircase Step 4 will use DGD |
| **P1** | `dgd_backward.cu` | 1 | Same |
| **P2** | `delta_forward.cu` | 1 | Share pattern with Titans (subset) |
| **P2** | `delta_backward.cu` | 1 | Same |
| **P2** | `hebbian_forward.cu` | 1 | Lower priority, simpler kernel |
| **P2** | `hebbian_backward.cu` | 1 | Same |
| **P3** | `swa_forward.cu` | 1 | Attention — likely not bottleneck |
| **P3** | `swa_backward.cu` | 1 | Same |
| **P3** | `adamw.cu` | 1 | Simple element-wise, already fast |
| **P3** | Others | 1 | embedding, cross_entropy, m_norm_clamp |

Phase 1 (cp.async prefetch) applies to all kernels uniformly — same pattern.
Phase 2 (TMA double-buffer M) applies only to memory update kernels (Titans, DGD,
Delta, Hebbian).

---

## 6. Verification Plan

### 6.1 Local (sm_86, no H100 needed)

- **Compilation test**: `cargo build -p nl_hecate_core --features cuda` succeeds
  with sm_90a gencode flag. nvcc may warn about sm_90a features in #ifdef-guarded
  code — this is expected and harmless.
- **sm_86 regression**: Run full test suite. sm_86 SASS must be byte-identical
  to pre-change binary (verified via `cuobjdump` diffing the fat binary).
- **Code path test**: Verify `#if __CUDA_ARCH__ >= 800` code compiles without
  errors when targeted at sm_86, sm_89, and sm_90a.

### 6.2 H100 Pod (rented, 2-4 hours)

- **Numerical equivalence**: Run 200 steps on H100 with sm_90a kernels.
  Compare loss trace against sm_90 PTX fallback run. Must be within fp32 ULP
  tolerance (identical loss to 6 decimal places).
- **Throughput benchmark**: `scripts/profile_composition.py` on H100 at d=512.
  Measure baseline (sm_90 PTX) vs optimized (sm_90a native) throughput.
- **d=2048 scaling**: Profile at d=2048 to validate TMA benefit at scale.
  This is the configuration that will matter for the 1B model.
- **Nsight Compute**: Profile memory throughput, SM occupancy, and stall reasons
  for the Titans forward kernel. Confirm TMA loads are active and cp.async
  pipelines are utilized.

---

## 7. Code Smell Compliance

| Smell | Relevance | Compliance |
|---|---|---|
| CS-10 | No train/eval distinction | sm_90a kernels run identically in all phases |
| CS-22 | Forward pass is the only API | No new API surface — same FFI entry points |
| CS-40 | Opt-in AD | Hopper optimizations are transparent to the Wengert tape |
| CS-18 | Orchestration in Python tier | No change — kernel dispatch unchanged |

The Hopper optimizations are strictly performance — they do not change the kernel
interfaces, the tape recording, or the mathematical output. The same extern "C"
function signatures are called from Rust; the CUDA runtime selects the appropriate
SASS variant from the fat binary based on the detected GPU architecture.

---

## 8. Files Modified

| File | Change |
|---|---|
| `core/build.rs` | Add sm_90a gencode + compute_90a PTX |
| `core/src/dispatch.rs` | Update comments (no logic change needed) |
| `core/kernels/titans_forward.cu` | Phase 1: cp.async. Phase 2: TMA double-buffer |
| `core/kernels/titans_backward.cu` | Phase 1: cp.async for recomputed vectors |
| `core/kernels/dgd_forward.cu` | Phase 1: cp.async |
| `core/kernels/dgd_backward.cu` | Phase 1: cp.async |
| `core/kernels/delta_forward.cu` | Phase 1: cp.async |
| `core/kernels/delta_backward.cu` | Phase 1: cp.async |
| `core/kernels/hebbian_forward.cu` | Phase 1: cp.async |
| `core/kernels/hebbian_backward.cu` | Phase 1: cp.async |

Files NOT modified: `swa_forward.cu`, `swa_backward.cu`, `embedding.cu`,
`elementwise.cu`, `cross_entropy.cu`, `adamw.cu`, `swiglu_forward.cu`,
`swiglu_backward.cu`, `m_norm_clamp.cu`, `gate_backward.cu`.

---

## 9. Blackwell Forward Compatibility

sm_100 (B100/B200) is covered by the `compute_90a` PTX fallback — the Blackwell
driver JIT-compiles Hopper PTX for Blackwell at first launch. When Blackwell pods
become available for testing, a follow-on spec will add:

```rust
.flag("-gencode").flag("arch=compute_100a,code=sm_100a")
```

Blackwell's key new feature for NL-Hecate is FP4 tensor cores — but these apply to
attention (bf16 path), not memory update kernels (fp32 path). A separate spec will
address FP4/FP8 attention when Blackwell hardware is accessible.

---

## 10. Success Criteria

1. `cargo build --features cuda` succeeds with sm_90a in the fat binary
2. All existing tests pass (sm_86 regression — no numerical change)
3. H100 pod: sm_90a native throughput ≥ sm_90 PTX throughput (any improvement)
4. H100 pod at d=2048: TMA-enabled kernel shows measurable improvement (≥10%)
   over non-TMA path in Nsight Compute memory throughput metric
5. Fat binary contains SASS for sm_86, sm_89, sm_90a + PTX for compute_90a, compute_86
