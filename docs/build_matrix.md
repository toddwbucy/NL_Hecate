# NL_Hecate Build Matrix

## Toolchain Requirements

| Component | Version | Notes |
|---|---|---|
| Rust | stable or nightly | No custom toolchain needed (Enzyme archived) |
| CUDA Toolkit | 12.8+ | Required for `cuda` feature only |
| nvcc | Matches CUDA Toolkit version | Compiles `.cu` to fat binary |
| cc crate | 1.0+ | Drives nvcc from `build.rs` |

## Feature Combinations

| Features | Target | CUDA | WASM | Use Case |
|---|---|---|---|---|
| (none) | x86_64 | No | No | Rust reference only |
| `cuda` | x86_64 + sm_86/89/90 | Yes | No | GPU-accelerated kernels |
| `edge` | x86_64, wasm32 | No | Yes | Micro model deployment |
| `cuda,distributed` | x86_64 + sm_86/89/90 | Yes | No | Multi-GPU build |
| `cuda,serving` | x86_64 + sm_86/89/90 | Yes | No | Production inference |
| `cuda,distributed,serving` | x86_64 + sm_86/89/90 | Yes | No | Full production |

## GPU Architecture Support

The CUDA build produces a **fat binary** containing multiple architecture variants. The CUDA runtime automatically selects the correct one at kernel launch time.

| Architecture | GPU Examples | Binary Type | Performance |
|---|---|---|---|
| sm_86 | RTX A6000, RTX 3090, RTX 3080 | SASS (native) | Optimal |
| sm_89 | RTX 4090, RTX 2000 Ada, L40 | SASS (native) | Optimal |
| sm_90 | H100, H200 | SASS (native) | Optimal |
| sm_86+ (other) | Future GPUs | PTX (JIT-compiled) | Near-optimal (first-launch JIT cost) |
| < sm_86 | V100, A100, RTX 2080 | Not supported | Falls back to Rust reference |

### How Fat Binary Works

`build.rs` passes multiple `-gencode` flags to nvcc:

```
-gencode arch=compute_86,code=sm_86     # SASS for sm_86
-gencode arch=compute_89,code=sm_89     # SASS for sm_89
-gencode arch=compute_90,code=sm_90     # SASS for sm_90
-gencode arch=compute_86,code=compute_86  # PTX for JIT fallback
```

The `cc` crate packages all variants into a single `.o` file linked into the Rust binary. No separate `.cubin` or `.ptx` file management needed.

## Build Commands

### Rust Reference Only (no GPU)

```bash
cargo build --release
cargo test --release --lib --tests
```

### CUDA Build (requires GPU + CUDA Toolkit)

```bash
cargo build --release --features cuda
cargo test --release --features cuda --lib --tests
```

### Full Build (all features)

```bash
cargo test --release \
  --features "cuda,distributed,serving,edge" \
  --lib --tests
```

### Edge / WASM Build

```bash
# x86_64 edge
cargo build --release --features edge

# wasm32 cross-compile validation
cargo build --release --features edge --target wasm32-unknown-unknown --lib
```

## Runtime Backend Selection

The dispatch system uses **compile-time** feature gates as the primary mechanism (zero overhead). An optional **runtime** override exists for testing:

```rust
use nl_hecate_core::dispatch::{detect_gpu, select_backend, force_rust_reference};

// Query GPU at startup (diagnostics/logging)
let gpu = detect_gpu();
let backend = select_backend(&gpu);
println!("Backend: {:?}, GPU: {:?}", backend, gpu);

// Force Rust path for comparison testing
force_rust_reference(true);
// ... all dispatch functions now use Rust reference ...
force_rust_reference(false);
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_PATH` | `/usr/local/cuda-12.8` | CUDA Toolkit installation path |
| `RUSTFLAGS` | (none) | Standard Rust flags (no Enzyme-specific flags needed) |

## Known Limitations

1. **sm < 86 unsupported**: GPUs older than Ampere fall back to Rust reference. No PTX is emitted for compute < 86.
2. **head_dim <= 32 for SWA CUDA**: The SWA kernel uses one warp per (head, position). Larger head_dim requires multi-warp reduction (not implemented).
3. **Single-block memory kernels**: Delta/Titans/Hebbian kernels use Grid=1, Block=min(d^2, 1024). This is optimal for small d but doesn't parallelize across the sequence dimension.
4. **CUDA kernels opaque to AD**: CUDA kernels are compiled to SASS by nvcc — opaque to the Wengert tape. Gradients for CUDA operations use hand-written backward kernels registered as opaque VJP blocks.

## Kernel Inventory

| Kernel File | Operation | Precision | Grid | Block |
|---|---|---|---|---|
| `swa_forward.cu` | SWA attention forward | bf16 storage, f32 compute | (num_heads, seq_len) | (head_dim) |
| `swa_backward.cu` | SWA attention backward | bf16 inputs, f32 grads | (num_heads, seq_len) | (head_dim) |
| `delta_forward.cu` | Delta Rule M recurrence | all f32 | (1) | min(d*d, 1024) |
| `delta_backward.cu` | Delta Rule gradients | all f32 | (1) | min(d*d, 1024) |
| `titans_forward.cu` | Titans LMM M+S recurrence | all f32 | (1) | min(d*d, 1024) |
| `titans_backward.cu` | Titans LMM gradients | all f32 | (1) | min(d*d, 1024) |
| `hebbian_forward.cu` | Hebbian M recurrence | all f32 | (1) | min(d*d, 1024) |
| `hebbian_backward.cu` | Hebbian gradients | all f32 | (1) | min(d*d, 1024) |

## Test Counts

| Test File | Tests | Feature Gate |
|---|---|---|
| `test_cuda_swa.rs` | 5 | `cuda` |
| `test_cuda_delta.rs` | 11 | `cuda` |
| `test_cuda_titans.rs` | 6 | `cuda` |
| `test_cuda_hebbian.rs` | 7 | `cuda` |
| `test_cuda_compositions.rs` | 5 | `cuda` |
| `test_dispatch.rs` | ~13 | partial `cuda` |
