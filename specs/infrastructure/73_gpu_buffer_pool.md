# Spec 73: GPU Buffer Pool — Eliminate Per-Step cudaMalloc/cudaFree Overhead

## CONTRACT
- **Purpose**: Eliminate ~572 `cudaMalloc`/`cudaFree` calls per build step via a thread-local recycling pool in `GpuBuf`.
- **Expects**: `GpuBuf<T>` RAII wrapper in `core/src/gpu_buf.rs` with direct `cudaMalloc` on construction, `cudaFree` on drop.
- **Guarantees**: After one warm-up step, zero `cudaMalloc`/`cudaFree` calls per step. Pool recycles buffers by exact byte size. Functionally identical behavior (same `cudaMemset` zeroing, same data flow, same numerical results). Pool drains cleanly on shutdown.
- **Cost**: ~80 lines new Rust in `gpu_buf.rs`. No new CUDA kernels. No changes to forward/backward math.
- **Trade-off**: VRAM stays at high-water mark between steps (buffers recycled, not freed). This is desired for a build loop — the same buffers are needed every step.
- **Position**: Infrastructure optimization. Eliminates the dominant source of GPU pipeline serialization visible in nvtop.
- **Source**: nvtop profiling showing sawtooth GPU utilization at ~3100 tok/s constant across seq_len=4096/16384/32768. GPU computes in short bursts then idles waiting for `cudaMalloc`/`cudaFree` between kernel launches.

## Problem

Every build step allocates and frees ~572 GPU buffers:

| Function | Allocations | Frees (on drop) |
|----------|-------------|-----------------|
| `forward_sequence` | ~313 | ~313 (cache dropped after backward) |
| `gpu_cross_entropy_loss` | 1 | 1 |
| `gpu_stacked_backward` | ~258 | ~258 (grads dropped after optimizer) |
| **Total per step** | **~572** | **~572** |

`cudaMalloc` is a synchronous CUDA runtime call — it serializes the GPU pipeline. Each call takes 5–50 µs, but more critically, it prevents overlapping kernel execution with memory management. With 572 alloc/free pairs interspersed with kernel launches, the GPU repeatedly stalls.

This is visible in nvtop as a sawtooth pattern: compute spike → idle gap → compute spike. The pattern repeats identically regardless of `seq_len` because the allocation count is fixed per step (it's the number of *buffers*, not their size, that matters).

### Why tok/s is constant across seq_len

At `seq_len=4096`: GPU work ∝ 4096, alloc overhead ∝ 572 calls → ratio R
At `seq_len=32768`: GPU work ∝ 32768, alloc overhead ∝ 572 calls → but each `cudaMalloc` for a larger buffer takes longer (internal page table operations scale with size). The ratio stays approximately constant, yielding ~3100 tok/s at all seq_lens.

### Existing precedent: StackedDecodeWorkspace

The single-token decode path (`forward_single_token`) already solves this with `StackedDecodeWorkspace` — a struct that pre-allocates all per-token buffers at model init. The comment says:

> "Pre-allocated GPU workspace for single-token forward pass. Created once at prefill, reused for every decode call (zero cudaMalloc per token)."

The build-mode `forward_sequence` path never got this treatment.

## Solution

Thread-local buffer pool in `GpuBuf`. When a `GpuBuf` is dropped, its device pointer is returned to a pool keyed by byte size instead of calling `cudaFree`. When a new `GpuBuf` is allocated, the pool is checked first — if a buffer of the exact byte size exists, it's reused (with `cudaMemset` to zero if needed). Only on the first step (cold start) does `cudaMalloc` actually run.

### Design: thread-local pool (no API changes to forward/backward)

```rust
use std::cell::RefCell;
use std::collections::HashMap;

struct GpuPool {
    free_lists: HashMap<usize, Vec<*mut u8>>,
    stats: PoolStats,
}

struct PoolStats {
    hits: u64,
    misses: u64,
    returns: u64,
}

thread_local! {
    static GPU_POOL: RefCell<Option<GpuPool>> = RefCell::new(None);
}
```

### GpuBuf changes

```rust
impl<T: GpuElement> GpuBuf<T> {
    pub fn new(len: usize) -> Self {
        let bytes = len * T::byte_size();
        // Try pool first
        let ptr = GPU_POOL.with(|pool| {
            pool.borrow_mut().as_mut().and_then(|p| p.alloc(bytes))
        });
        let ptr = match ptr {
            Some(p) => p as *mut T,
            None => {
                // Cold path: actual cudaMalloc
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let rc = unsafe { cudaMalloc(&mut p, bytes) };
                assert_eq!(rc, 0, "cudaMalloc failed: error {rc} ({bytes} bytes)");
                p as *mut T
            }
        };
        GpuBuf { ptr, len, owned: true, _not_send_sync: PhantomData }
    }
}

impl<T: GpuElement> Drop for GpuBuf<T> {
    fn drop(&mut self) {
        if !self.owned { return; }
        let bytes = self.len * T::byte_size();
        let returned = GPU_POOL.with(|pool| {
            pool.borrow_mut().as_mut().map(|p| {
                p.free(self.ptr as *mut u8, bytes);
                true
            }).unwrap_or(false)
        });
        if !returned {
            unsafe { cudaFree(self.ptr as *mut std::ffi::c_void); }
        }
        self.ptr = std::ptr::null_mut();
    }
}
```

### Pool lifecycle

```rust
/// Enable the buffer pool. Call once before the build loop.
pub fn gpu_pool_enable() {
    GPU_POOL.with(|pool| {
        *pool.borrow_mut() = Some(GpuPool::new());
    });
}

/// Drain the pool, freeing all cached buffers. Call on shutdown.
pub fn gpu_pool_drain() -> PoolStats {
    GPU_POOL.with(|pool| {
        pool.borrow_mut().take()
            .map(|p| p.drain())
            .unwrap_or_default()
    })
}

/// Log pool statistics (hits, misses, unique sizes). Call on log steps.
pub fn gpu_pool_stats() -> Option<PoolStats> {
    GPU_POOL.with(|pool| {
        pool.borrow().as_ref().map(|p| p.stats.clone())
    })
}
```

### Changes

| File | Change |
|------|--------|
| `core/src/gpu_buf.rs` | Add `GpuPool`, `PoolStats`, thread-local `GPU_POOL`. Modify `GpuBuf::new()` to try pool first. Modify `Drop` to return to pool. Add `gpu_pool_enable()`, `gpu_pool_drain()`, `gpu_pool_stats()`. |
| `cli/src/feed.rs` | Call `gpu_pool_enable()` before build loop. Call `gpu_pool_drain()` on clean exit, log stats. |
| `cli/src/probe.rs` | Call `gpu_pool_enable()` before generation. Call `gpu_pool_drain()` on exit. |

### What does NOT change

- `forward_sequence` — zero code changes. `GpuBuf::zeros()` transparently recycles.
- `gpu_stacked_backward` — zero code changes.
- `gpu_stacked_adamw_update` — zero code changes.
- `gpu_cross_entropy_loss` — zero code changes.
- `GpuBuf::from_raw_non_owning` — unchanged (not owned, not pooled).
- `clone_buf()` — allocates via `new()` (gets pool benefit), copies via `cudaMemcpy` (unchanged).

### Pool key: exact byte size

Buffers are pooled by exact byte count. For a typical d=1024, k=4, n_blocks=4 build:

| Size | Count per step | Examples |
|------|---------------|----------|
| `s × d × 4` | ~80 | residual, ln_out, q/k/v projections |
| `s × d × 2` | ~24 | bf16 attention buffers |
| `s × v × 4` | 2 | logits (forward + backward) |
| `s × 4` | ~20 | LN mean/rstd, gate scalars |
| `d × 4` | ~40 | gradient accumulators |

After step 1, all ~572 buffers are in the pool. Step 2 finds exact-size matches for every allocation — zero `cudaMalloc` calls.

### Memory overhead

The pool holds ~572 buffers between steps. These are the same buffers that would be allocated anyway — no additional VRAM is consumed. The pool just prevents the free→realloc cycle.

Peak VRAM usage is identical to the current implementation (all buffers exist simultaneously during forward+backward). The only difference is they persist across steps instead of being freed and reallocated.

### Thread safety

The pool is thread-local (`thread_local!`). This is correct because:
1. CUDA device pointers are bound to the creating thread's CUDA context.
2. NL_Hecate's build loop runs on a single thread.
3. No `Send`/`Sync` on `GpuBuf` (already enforced via `PhantomData<Rc<()>>`).

## Backward path — no changes needed

The backward already uses the pattern from spec 54: `all_keep_alive` buffers prevent premature `cudaFree` during the block loop. With the pool, these buffers return to the pool on drop instead of calling `cudaFree` — same deferred lifetime, zero overhead.

## zeros() still calls cudaMemset

`GpuBuf::zeros(n)` calls `GpuBuf::new(n)` (pool-backed) then `cudaMemset(ptr, 0, bytes)`. The memset is an async GPU operation on the default stream — it takes ~1 µs to enqueue regardless of buffer size. This is negligible compared to the eliminated `cudaMalloc` overhead.

Recycled buffers contain stale data from the previous step. The `cudaMemset` ensures correctness. Buffers obtained via `GpuBuf::new()` (no zeroing) skip memset — they'll be overwritten by kernel output anyway.

## Verification

1. `cargo test --features cuda --lib` — all existing tests pass (pool disabled in tests by default)
2. New test: enable pool, run 3 forward+backward cycles, verify `pool_stats.misses == step1_count` and `pool_stats.hits == 2 * step1_count` (steps 2 and 3 are all hits)
3. Loss/gnorm equivalence: 100 steps with pool vs without — must match within f32 tolerance
4. nvtop: sawtooth pattern should flatten after step 1
5. tok/s improvement measurable on d=1024 build
6. `gpu_pool_drain()` on shutdown: verify all pointers freed, no CUDA leaks
