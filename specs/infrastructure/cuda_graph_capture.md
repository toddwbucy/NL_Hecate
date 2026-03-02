# CUDA Graph Capture for Wengert Tape Kernel Dispatch

```text
CONTRACT
  Purpose:    After a configurable warmup period, capture the CUDA kernel launch
              sequence for each distinct CMS pulse pattern as a CUDA Graph. From
              that point, replay the pre-captured graph instead of dispatching
              kernels individually from the CPU host thread — eliminating per-step
              CPU→GPU kernel dispatch overhead entirely for steady-state training.
  Expects:    Primary model (TitansLMM or DeltaRule + MAG + k=4) training-stable
              with a validated loss curve. CUDA Toolkit 12.8+, driver ≥ 13.0,
              target sm_86 (A6000 Ampere). GpuBuf<f32> arena sizes fixed at capture
              time (they are: seq_len, d, batch_size are config-determined).
  Guarantees: Numerically identical output to non-captured dispatch on every step
              (CUDA graphs replay the same kernel code with the same pointer args).
              Tape metadata recording (CPU-side) is unchanged — the tape still sees
              every op. OpaqueVjp backward dispatch is unchanged. No change to
              gradient correctness. Zero overhead when feature flag is absent.
  Cost:       Capture: ~5–20 ms per pattern (one-time per run, after warmup).
              Replay: ~0 CPU overhead per step (GPU-resident launch queue).
              Memory: one cudaGraph_t handle per pattern (8 valid for k=4), negligible.
  Trade-off:  Static graphs require fixed buffer pointers and sizes. Any change
              to buffer layout (checkpoint resume to different d or seq_len) must
              invalidate and re-capture. Fallback to standard dispatch handles this.
              Conditional branching in kernels (e.g. level_active check) is fine —
              the graph captures whatever the kernel code does; it does not flatten it.
  Position:   specs/infrastructure/cuda_graph_capture.md
  Source:     CUDA Programming Guide §9 "CUDA Graphs",
              NVIDIA Tech Blog "Optimizing NLP Models with CUDA Graphs",
              CUDA 10.0+ API: cudaGraphCreate, cudaStreamBeginCapture,
              cudaStreamEndCapture, cudaGraphInstantiate, cudaGraphLaunch.
              NL-Hecate: core/src/gpu_forward.rs, core/src/conductor.rs (Pulse).
              CS-10, CS-22, CS-40.
```

---

## 1. Motivation

The Wengert tape's control flow (op list, backward traversal) is correctly
CPU-side and stays there. The bottleneck is CPU→GPU kernel dispatch overhead:
the host thread launching each CUDA kernel individually adds latency between
kernel calls (typically 5–20 µs per kernel on modern hardware).

For NL-Hecate specifically:
- At d=512, k=4: ~10 CUDA kernels per forward step (gate_compute, delta_forward ×4,
  m_norm_clamp ×4, softmax, etc.)
- At 128 tok/s baseline: ~8 ms per step budget → 50–200 µs dispatch overhead
  is a meaningful fraction (0.6%–2.5% of step time)
- The CMS pulse pattern (which levels fire) is fully deterministic from the step
  counter and chunk_sizes — no runtime data dependency

CUDA Graphs capture the kernel launch sequence once and replay it at GPU speed
with zero CPU involvement per step once captured.

---

## 2. Pulse Patterns and Graph Count

For k=4 with chunk_sizes=[1,8,64,512]:

```text
Bitmask  active_levels       Period (LCM step mod)
0b0001   [T,F,F,F]           every step where L0 only fires
0b0011   [T,T,F,F]           every 8 steps
0b0101   [T,F,T,F]           every 64 steps (when L2 fires but L1 doesn't yet)
0b0111   [T,T,T,F]           when L0+L1+L2 fire
0b1001   [T,F,F,T]           every 512 steps (L3 fires alone with L0)
0b1011   [T,T,F,T]           L0+L1+L3
0b1101   [T,F,T,T]           L0+L2+L3
0b1111   [T,T,T,T]           all levels fire (step 0, step LCM(1,8,64,512)=512, ...)
```

L0 always fires (chunk_sizes[0]=1 → fires every step). Only 8 of the 16 possible
4-bit bitmasks are reachable in steady state. The graph store allocates 16 slots
(indexed by bitmask u8) but populates only the 8 reachable ones.

---

## 3. Capture Lifecycle

```text
warmup phase  (steps 0..WARMUP_STEPS-1):
  - Standard dispatch (current behavior)
  - GPU buffer layout stabilizes (alloc from fixed config, no realloc after step 0)
  - No graph capture

capture phase (step == WARMUP_STEPS):
  - For each of the 8 reachable pulse patterns:
    1. Reset GpuContextState to known-good state
    2. cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
    3. Run full gpu_memory_forward() for this pulse pattern
    4. cudaStreamEndCapture(stream, &graph_raw)
    5. cudaGraphInstantiate(&exec, graph_raw, ...)
    6. Store exec in CudaGraphStore[bitmask]
    7. cudaGraphDestroy(graph_raw)
  - Fallback: if any capture fails, set store to Disabled; log warning

replay phase  (steps > WARMUP_STEPS):
  - Compute bitmask from pulse.active_levels
  - If store.has(bitmask): cudaGraphLaunch(store[bitmask], stream)
  - Else: standard dispatch (pattern not captured, shouldn't happen for k=4)
  - Tape metadata recording proceeds on CPU as usual (unchanged)

invalidation:
  - On checkpoint resume: if buffer layout changes (d or seq_len differ),
    call CudaGraphStore::invalidate() → drops all cudaGraphExec_t handles
    and re-enters warmup phase
```

---

## 4. Struct Design

```rust
// core/src/cuda_graph.rs  (new module, feature-gated)
#[cfg(feature = "cuda")]
pub struct CudaGraphStore {
    // Indexed by pulse bitmask (u8, 0..=0b1111).
    // None = not yet captured or disabled for this pattern.
    graphs: [Option<CudaGraphExec>; 16],
    warmup_steps: usize,
    steps_seen: usize,
    enabled: bool,         // false if any capture failed
}

#[cfg(feature = "cuda")]
struct CudaGraphExec {
    handle: cudaGraphExec_t,  // CUDA opaque handle
}

// Drop impl calls cudaGraphExecDestroy
impl Drop for CudaGraphExec { ... }

impl CudaGraphStore {
    pub fn new(warmup_steps: usize) -> Self { ... }
    pub fn step(&mut self) { self.steps_seen += 1; }
    pub fn should_capture(&self) -> bool {
        self.enabled && self.steps_seen == self.warmup_steps
    }
    pub fn replay(&self, bitmask: u8, stream: cudaStream_t) -> bool { ... }
    pub fn invalidate(&mut self) { ... }
}
```

---

## 5. Integration Points (What Changes)

### `core/src/gpu_forward.rs`
- `GpuModel` gains an optional `CudaGraphStore` field (None when feature absent
  or warmup_steps=0 which disables graphs entirely)
- `gpu_cms_forward()`: after warmup, wraps the kernel dispatch region in
  `cudaStreamBeginCapture` / `cudaStreamEndCapture` on first occurrence of each
  pattern; calls `cudaGraphLaunch` on subsequent occurrences
- Capture/replay is transparent to the return value: outputs are the same GPU
  buffers, same addresses

### `core/src/model.rs` / `python/src/lib.rs`
- `GpuModel::new()` accepts `warmup_steps: usize` (default 100, 0 = disabled)
- Python: `GpuModel.from_params(..., cuda_graph_warmup=100)`

### `core/src/gpu_backward.rs`
- Backward graphs are captured identically if `backward_capture=True` (deferred:
  backward kernel sequence is less deterministic due to conditional arms)
- Phase 1: forward capture only. Backward capture is a follow-on.

### `core/kernels/` (CUDA kernels)
- Unchanged. Any kernel that works under standard dispatch also works under CUDA
  Graph capture (CUDA spec guarantee: graphs replay the same PTX with same args).

---

## 6. What Does NOT Change

| Component | Change |
|---|---|
| Wengert tape architecture | None — tape records ops on CPU as before |
| OpaqueVjp mechanism | None — backward adapters registered identically |
| GpuBuf<f32> arena | None — allocations are all config-determined, stable after step 0 |
| Python-tier orchestration | None — loop.py unchanged |
| Pulse / Conductor | None — bitmask computed from existing active_levels |
| Numerical output | Identical — graphs replay same kernel code |
| fp32 inner-loop constraint | None — graphs do not change dtype |

---

## 7. Code Smell Constraints

- **CS-10** (no train/eval distinction): CUDA graph replay happens on every
  step regardless of training or eval context. No mode flag.
- **CS-22** (forward pass is the only API): graphs accelerate the forward
  pass; they do not introduce a new API surface.
- **CS-40** (opt-in AD): graph capture wraps the GPU kernel launch region only.
  The CPU-side tape recording is unaffected — the tape still sees every op.
  CUDA graph capture is transparent to the Wengert tape.
- **CS-47** (no in-place mutation of tracked tensors): GPU buffers captured in
  the graph are the same GpuBuf handles the tape already tracks. No aliasing
  hazard introduced.

---

## 8. Out of Scope (Follow-On PRs)

- Backward graph capture — requires deterministic backward kernel sequence
- Conditional CUDA Graph nodes (CUDA 12.4+) — single-graph approach with
  conditional nodes; more complex validation
- Multi-stream execution — orthogonal optimization
- Async memory prefetch — separate concern

---

## 9. Verification

```bash
# Feature flag is "cuda"; graph capture is behind a build flag too
cargo build -p nl_hecate_core --features cuda

# Numerical equivalence: run 200 steps with and without graph capture,
# verify loss trace is bitwise-identical (or within f32 ulp tolerance)
python hecate.py build configs/fineweb_edu_k4_v2.json --steps 200 \
  --cuda-graph-warmup 100  # 100 warmup steps → capture at step 100

# Throughput: measure tok/s before and after
python scripts/profile_step.py --cuda-graph-warmup 100

# Expected gain: 50-200µs/step eliminated → ~1-3% throughput improvement
# at d=512, k=4, seq_len=512 on A6000
```
