# Enzyme Toolchain Build Log

## Target Configuration

- **Host**: Linux x86_64
- **GPUs**: 2x RTX A6000 (SM 8.6), 1x RTX 2000 Ada (SM 8.9)
- **CUDA Driver**: 580.126.09
- **Spike scope**: CPU-only AD testing (GPU only if testing CUDA kernel dispatch)

## Build Instructions

### Prerequisites

```bash
# System packages
sudo apt install -y clang lld ninja-build cmake python3 git curl

# Rustup (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Building rustc with Enzyme

```bash
# Clone Rust source
git clone https://github.com/rust-lang/rust.git rust-enzyme
cd rust-enzyme

# Pin to specific commit (fill in after successful build)
# git checkout <SHA>

# Configure
./configure \
  --release-channel=nightly \
  --enable-llvm-enzyme \
  --enable-llvm-link-shared \
  --enable-llvm-assertions \
  --enable-ninja \
  --enable-option-checking \
  --disable-docs \
  --set llvm.download-ci-llvm=false

# Build stage 1 (expect 1-3 hours)
./x build --stage 1 library

# Link as rustup toolchain
rustup toolchain link enzyme build/host/stage1
```

### Verification

```bash
# Run Enzyme's own test suite
./x test --stage 1 tests/ui/autodiff

# Verify in spike project
cd /path/to/NL_Hecate/spike
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme build --release
```

## Pinned Versions

Fill in after successful build:

| Component | Version / SHA |
|-----------|--------------|
| rustc source | `git checkout ________` |
| LLVM version | ________ |
| Enzyme version | ________ |
| Build date | ________ |
| Build time | ________ minutes |
| Build flags | See configure above |
| Patches needed | ________ |

## RUSTFLAGS Reference

```bash
# Minimum required
RUSTFLAGS="-Zautodiff=Enable"

# With diagnostics
RUSTFLAGS="-Zautodiff=Enable,PrintTA,PrintAA,PrintPerf"

# Full debug (very verbose)
RUSTFLAGS="-Zautodiff=Enable,PrintSteps,PrintModBefore,PrintModAfter"
```

## Known Issues

Document any workarounds needed here:

1. _____
2. _____
