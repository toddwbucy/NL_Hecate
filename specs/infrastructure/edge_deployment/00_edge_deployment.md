# Edge Deployment: Zero-Dependency Micro Models

```
CONTRACT
  Purpose:    Track Zero-A Phase 4 revealed an unplanned capability: the Rust
              pipeline achieves 18k tok/s on CPU for a 49K-parameter model with
              ZERO external dependencies — no Python, no CUDA, no tensor library.
              The entire model (192 KB) fits in L1/L2 cache. Combined with NL's
              self-modifying forward pass, this enables on-device deployment of
              task-specific micro models that continue learning after deployment.
              This spec defines the edge deployment profile and its implications.
  Expects:    A built micro model (outer-loop params frozen or unfrozen).
              A Rust binary compiled for the target architecture.
              No GPU. No Python runtime. No framework.
  Guarantees: The model runs on any target rustc supports (ARM, RISC-V, x86,
              WASM, bare-metal with alloc).
              The inner loop runs during serving (on-device adaptation).
              The outer loop CAN run on-device if Enzyme is available for the
              target — otherwise inner-loop-only adaptation.
              Binary size is dominated by the model weights, not the runtime.
              Latency is bounded by CPU cache performance, not memory bandwidth.
  Cost:       At micro scale (d<=128, vocab<=256): all computation in L1/L2 cache.
              No kernel launch overhead. No PCIe transfers. No GIL.
              Single-threaded is sufficient — parallelism overhead exceeds compute.
              Power draw: milliwatts (CPU core only, no GPU).
  Trade-off:  Micro models have limited capacity. This is NOT a path to GPT-scale.
              This IS a path to deploying thousands of specialized models on edge
              devices where each model does one thing well and adapts to its
              local context continuously.
  Position:   specs/infrastructure/edge_deployment/00_edge_deployment.md
              Extends: serving (00_serving.md), compilation (00_compilation.md)
  Source:     Track Zero-A Phase 4 empirical results; NL IS #5 (ICL naturally
              emerges); NL IS #7 (self-modifying); CS-10 (no mode distinction)
```

## The Observation

```
Track Zero-A Phase 4 measured both pipelines on identical hardware:

  Model: d=64, heads=4, seq_len=32, vocab=256 (49K params, 192 KB)
  Rust pipeline:    18,425 tok/s  (Enzyme AD, single-threaded CPU)
  PyTorch pipeline: 37,106 tok/s  (autograd, BLAS multi-threaded)

PyTorch is 2x faster here because BLAS parallelizes 64x64 matmuls across
cores. But the Rust pipeline has ZERO dependencies — the binary is the
runtime. PyTorch requires:
  - Python 3.x runtime          (~30 MB)
  - libtorch + CUDA bindings    (~2 GB)
  - numpy, etc.                 (~50 MB)
  Total framework weight:       >2 GB

The Rust binary for a micro model:
  - Static binary (no .so)      ~500 KB (release, stripped)
  - Model weights                ~192 KB
  - Total deployment artifact:  <1 MB

This is a 2000:1 ratio in deployment size for a 2:1 ratio in throughput.
```

## Why CPU Wins at Micro Scale

```
At d=64, the entire computation graph fits in CPU cache:

  Weight matrix (64x64):  16 KB   → fits in L1 (32-64 KB)
  All 6 matrices:         96 KB   → fits in L2 (256 KB-1 MB)
  Full model + activations: ~250 KB → fits in L2

  L1 latency:   ~1 ns
  L2 latency:   ~5 ns
  DRAM latency: ~100 ns
  GPU global:   ~400 ns (plus kernel launch overhead)

For a 64x64 matmul:
  Operations: 64 * 64 = 4,096 multiply-adds
  CPU time at 4 GHz with FMA: ~1 microsecond
  GPU kernel launch overhead: ~5-10 microseconds

The GPU LAUNCH alone is 5-10x slower than the CPU COMPUTATION.
At micro scale, the overhead IS the bottleneck, not the math.
```

## What NL Adds: On-Device Adaptation

```
CONVENTIONAL EDGE ML:
  model = load_frozen_weights("model.tflite")
  for input in sensor_stream:
    output = model.infer(input)
    // Model is DEAD. It cannot adapt. It cannot learn.
    // If the distribution shifts, accuracy degrades silently.
    // Recovery requires: collect data → ship to cloud → retrain → redeploy.

NL EDGE ML:
  model = load_weights("model.nl")
  for input in sensor_stream:
    (output, context) = model.process(input, context, pulse)
    // Model's inner loop ADAPTED to this input.
    // Memory M was updated. Next prediction reflects new context.
    // Distribution shift? The model adjusts in real time.
    // No cloud. No retraining. No redeployment.

This is the fundamental difference. Other edge frameworks (TFLite, ONNX
Runtime, TensorRT) STRIP the backward pass to reduce binary size. They
are inference-only by design.

NL_Hecate KEEPS the backward pass — and at micro scale it costs nearly
nothing. Enzyme compiles it into native code at build time. There is no
runtime graph, no tape, no autograd overhead. The backward pass is just
more machine code alongside the forward pass.
```

## Deployment Profiles

```
PROFILE 1: Inner-loop only (no Enzyme on target)
  Outer-loop params: frozen (loaded from checkpoint)
  Inner-loop state: active (memory updates during forward pass)
  Enzyme required: NO (inner loop uses pre-compiled update rules)
  On-device learning: context adaptation only (NL IS #5)
  Use case: sensor fusion, anomaly detection, predictive maintenance

  The inner loop gives the model ICL (in-context learning). It adapts
  to local patterns without modifying the base weights. This is what
  conventional models cannot do at all.

PROFILE 2: Full NL (Enzyme on target)
  Outer-loop params: trainable (Enzyme provides gradients)
  Inner-loop state: active
  Enzyme required: YES (target must have LLVM backend)
  On-device learning: full self-modification
  Use case: personalization, federated learning, continuous adaptation

  If the target supports LLVM (x86, ARM, RISC-V — but NOT WASM, NOT
  bare-metal without LLVM), Enzyme can generate backward passes at
  compile time. The model can update its own outer-loop weights on-device.
  This is full on-device training with no framework.

PROFILE 3: WASM (browser / serverless)
  Target: wasm32-unknown-unknown
  Enzyme: NOT available (Enzyme targets native LLVM, not WASM)
  Inner loop: active (compiled to WASM, runs in V8/SpiderMonkey)
  On-device learning: context adaptation only
  Use case: browser-based personalization, edge serverless functions

  The Rust model compiles to WASM. Inner-loop adaptation works.
  Outer-loop training requires a native build (desktop/server).
  Deploy the built model as WASM; it adapts via inner loop in the browser.
```

## Target Architecture Matrix

```
Target               Enzyme?   Profile   Binary Size   Expected tok/s
─────────────────────────────────────────────────────────────────────
x86_64 (desktop)     YES       2         ~500 KB       15-20k
aarch64 (RPi/phone)  YES       2         ~500 KB       3-8k
armv7 (MCU/IoT)      NO*       1         ~200 KB       1-3k
riscv64              YES       2         ~500 KB       1-5k
wasm32 (browser)     NO        3         ~300 KB       5-10k
thumbv7 (bare-metal) NO        1         ~150 KB       500-2k

* armv7 may gain Enzyme support as LLVM backend matures.

IMPORTANT: These are estimates for d=64, vocab=256 micro models.
Scaling to d=256 or d=512 changes the picture significantly —
weights alone grow to 3 MB or 12 MB, potentially exceeding cache.
```

## Model Size Regimes

```
The edge deployment story holds for models that fit in CPU cache.
Beyond that, memory bandwidth dominates and GPU becomes necessary.

Regime           d      Params    Weights    Fits in    GPU needed?
──────────────────────────────────────────────────────────────────
Micro            64     49K       192 KB     L2 cache   NO
Small            128    180K      720 KB     L2 cache   NO
Medium           256    660K      2.6 MB     L3 cache   maybe
Large            512    2.5M      10 MB      L3 cache   YES*
Standard         4096   >100M     >400 MB    DRAM only  YES

* "YES" means GPU throughput exceeds CPU by >10x.
  Below 256, the overhead of GPU dispatch often exceeds the compute.

The sweet spot for edge deployment: d=64 to d=128.
This gives enough capacity for task-specific models while staying
entirely within CPU cache for both weights and activations.
```

## Deployment Scenario: Adaptive Sensor Model

```
SCENARIO: Anomaly detection on an industrial vibration sensor.

Hardware: ARM Cortex-A53 (Raspberry Pi class), no GPU, 1 GB RAM.

Model: d=64, 2 heads, vocab=64 (quantized sensor readings).
  Parameters: ~12K (48 KB)
  Binary: ~200 KB total

Deployment:
  1. Build phase (server): Train on historical sensor data.
     Outer-loop params capture "normal behavior" distribution.
  2. Deploy (edge): Ship binary + checkpoint to device.
  3. Serve (edge): Process sensor stream continuously.
     Inner loop adapts memory to LOCAL equipment characteristics.
     Each sensor gets a slightly different model over time.
  4. Detect: When loss spikes (new input doesn't match memory),
     flag as anomaly. The model's own loss IS the anomaly score.

What NL buys you here:
  -- The model adapts to each specific machine's vibration signature.
  -- Seasonal drift (temperature, wear) is absorbed by the inner loop.
  -- No need to retrain centrally when conditions change.
  -- The model IS the detector — no separate anomaly scoring layer.
  -- Latency: <1 ms per reading at 3k tok/s on ARM.
  -- Power: <100 mW (single CPU core, partial utilization).
```

## Implications for Track Zero-B and Beyond

```
MEMORY SYSTEMS ON EDGE:
  Track Zero-B adds Delta Rule + MAG composition — memory that persists
  and accumulates context across the forward pass. This is EXACTLY what
  edge deployment needs: a model that remembers its local context.

  The memory matrix M for d=64 is 64*64 = 16 KB (vector memory is 256 B).
  This is negligible. The memory system adds almost no overhead at micro
  scale but gives the model its adaptive capability.

CMS ON EDGE:
  CMS frequency scheduling (k=4 levels) is a natural fit for edge power
  management. Level 0 fires every token (fast path). Levels 1-3 fire
  less frequently (slower, more expensive operations). On a power-
  constrained device, the Conductor can adjust CMS frequencies to trade
  adaptation speed for power consumption.

KERNEL-PAIR PATTERN ON EDGE:
  At micro scale, CUDA kernels are irrelevant — there is no GPU.
  The Rust reference implementations ARE the production kernels.
  This validates the three-implementation pattern: the Rust reference
  is not just a testing oracle, it's the edge deployment target.
```

## What This Is NOT

```
-- This is NOT "shrink a big model until it fits on edge."
   Micro models are purpose-built for one task from the start.
   They are not distilled, pruned, or quantized large models.
   They are small because the task is small.

-- This is NOT competitive with large language models.
   A 49K-parameter model cannot do general-purpose language modeling.
   It CAN do: pattern prediction, anomaly detection, signal classification,
   control loops, byte-level sequence modeling for specific domains.

-- This is NOT "Rust is faster than Python."
   At d=4096, PyTorch + CUDA dominates. The framework overhead is
   amortized by enormous matmuls that saturate GPU memory bandwidth.
   The edge story works BECAUSE the model is small enough that
   framework overhead is the dominant cost.

-- This IS "the right tool for the right scale."
   Large models need frameworks, GPUs, and infrastructure.
   Micro models need a compiler and a CPU.
   NL_Hecate handles both with the same codebase:
   Layer 2 (Rust) for edge, Layer 1 (CUDA) for scale.
```

## Axiom Compliance

- **NL IS #5** (ICL naturally emerges): Inner-loop adaptation on edge IS in-context learning
- **NL IS #7** (self-modifying): The model self-modifies on-device during serving
- **CS-10** (no mode distinction): Same code runs on server (Build) and edge (Stream)
- **CS-45** (no torch.compile): No compilation framework needed — Rust IS the compiler target
- **NL IS NOT #3** (not static): Edge models are non-stationary, adapting to local context
