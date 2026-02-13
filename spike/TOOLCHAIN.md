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

# Pin to tested commit
git checkout d7daac06

# Create bootstrap.toml (replaces ./configure)
cat > bootstrap.toml << 'EOF'
[rust]
channel = "nightly"

[llvm]
enzyme = true
download-ci-llvm = false
assertions = true
ninja = true
link-shared = true

[build]
docs = false
EOF

# Build stage 1 (~11 minutes with 48 cores, ~1-3 hours with fewer)
./x build --stage 1 library

# Link as rustup toolchain
rustup toolchain link enzyme build/x86_64-unknown-linux-gnu/stage1
```

### Verification

```bash
# Run Enzyme's own test suite (expect 10 pass, 4 ignored)
./x test --stage 1 tests/ui/autodiff

# Verify in spike project
cd /path/to/NL_Hecate/spike
RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme build --release

# Run the spike
./target/release/nl-hecate-enzyme-spike
```

## Pinned Versions

| Component | Version / SHA |
|-----------|--------------|
| rustc source | `git checkout d7daac06` |
| rustc version | `1.95.0-nightly (d7daac06d 2026-02-13)` |
| Enzyme test suite | 10 pass, 0 fail, 4 ignored |
| Build date | 2026-02-13 |
| Build time | 11 minutes 24 seconds (48 cores, 247GB RAM) |
| Build method | `bootstrap.toml` with `llvm.enzyme = true` |
| Patches needed | None |

## Correct Autodiff API

**CRITICAL**: Web sources often show the wrong API. The correct syntax (verified from
`tests/ui/autodiff/` in the rust-enzyme repo):

```rust
#![feature(autodiff)]
use std::autodiff::autodiff_reverse;

// Scalar params: Active
#[autodiff_reverse(d_square, Active, Active)]
fn square(x: f32) -> f32 { x * x }
// Generated: d_square(x: f32, seed: f32) -> (f32, f32)

// Reference params: Duplicated (requires &mut shadow)
#[autodiff_reverse(d_project, Duplicated, Active, Active)]
fn project(params: &Params, x: f32) -> f32 { ... }
// Generated: d_project(params: &Params, d_params: &mut Params, x: f32, seed: f32) -> (f32, f32)

// Params not differentiated: Const
#[autodiff_reverse(d_loss, Active, Const, Active)]
fn loss(output: f32, target: f32) -> f32 { ... }
```

**WRONG** (do NOT use):
- `use std::autodiff::autodiff;` — module exists but `autodiff` macro doesn't
- `#[autodiff(name, Reverse, ...)]` — `Reverse` keyword is NOT part of the syntax
- `#[autodiff_reverse]` does NOT have a `Reverse` mode argument

## RUSTFLAGS Reference

```bash
# Minimum required
RUSTFLAGS="-Zautodiff=Enable"

# With diagnostics
RUSTFLAGS="-Zautodiff=Enable,PrintTA,PrintAA,PrintPerf"

# Full debug (very verbose — dumps LLVM IR)
RUSTFLAGS="-Zautodiff=Enable,PrintSteps,PrintModBefore,PrintModAfter"
```

## Known Issues

1. **Duplicated shadow params ignore seed**: When calling an `#[autodiff_reverse]`
   function with `Duplicated` parameters, the shadow struct always receives the
   unit-seed (seed=1.0) Jacobian regardless of the actual seed value passed. For
   chain rule composition, call with seed=1.0 and manually multiply the shadow
   values by the upstream gradient afterward.

2. **`black_box` crashes Enzyme**: `std::hint::black_box` compiles to inline asm
   that Enzyme cannot differentiate through. Assertion failure in
   `GradientUtils.cpp: "cannot find deal with ptr that isnt arg"`. Do NOT use
   `#[autodiff_reverse]` on any function containing `black_box`.

3. **`extern "C"` in same crate is NOT opaque**: With fat LTO, Enzyme sees through
   `extern "C"` functions in the same compilation unit and differentiates them
   normally. For true opaque boundaries, the function must be in a separate
   compilation unit (separate crate or linked object).

4. **Fat LTO is mandatory**: `Cargo.toml` must have `lto = "fat"` in both
   `[profile.dev]` and `[profile.release]` for Enzyme to work.
