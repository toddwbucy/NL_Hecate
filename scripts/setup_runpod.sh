#!/bin/bash
set -euo pipefail

echo "=== NL_Hecate RunPod Setup ==="

# ── System deps ──────────────────────────────────────
apt-get update -qq && apt-get install -y -qq python3-venv 2>/dev/null || true

# ── Rust toolchain ───────────────────────────────────
if ! command -v rustc &>/dev/null; then
    echo "[1/4] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "[1/4] Rust already installed: $(rustc --version)"
fi
source "$HOME/.cargo/env" 2>/dev/null || true

# ── Python venv + deps ──────────────────────────────
cd /workspace/NL_Hecate/python
echo "[2/4] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install maturin numpy zstandard tokenizers

# ── Build nl_hecate (Rust → .so with CUDA) ──────────
echo "[3/4] Building nl_hecate (maturin develop --release)..."
echo "       This compiles CUDA kernels for sm_86/89/90a — takes 2-5 min."
maturin develop --release --features cuda

# ── Verify ──────────────────────────────────────────
echo "[4/4] Verifying..."
python3 -c "import nl_hecate; print(f'nl_hecate loaded OK')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "=== Setup complete ==="
echo "Activate with: source /workspace/NL_Hecate/python/.venv/bin/activate"
echo "Run with:      cd /workspace/NL_Hecate/python && python3 hecate.py --build --config <config>"
