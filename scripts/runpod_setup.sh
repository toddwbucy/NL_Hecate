#!/usr/bin/env bash
# runpod_setup.sh — Clone, build, and configure NL_Hecate on a RunPod H100 instance.
#
# Assumes:
#   - RunPod pytorch template (CUDA drivers + toolkit pre-installed)
#   - Python 3.10+ available
#   - SSH key or token for GitHub access (for private repo)
#
# Usage (from anywhere on RunPod):
#   curl -sL https://raw.githubusercontent.com/toddwbucy/NL_Hecate/main/scripts/runpod_setup.sh | bash
# Or if already cloned:
#   cd /workspace/NL_Hecate && bash scripts/runpod_setup.sh
#
# Idempotent: safe to re-run after failures or updates.

set -euo pipefail

WORKSPACE="/workspace/NL_Hecate"
REPO_URL="git@github.com:toddwbucy/NL_Hecate.git"

# ── Helpers ──────────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════"
}

check_ok() {
    echo "  ✓ $1"
}

die() {
    echo ""
    echo "  ✗ FAILED: $1" >&2
    exit 1
}

# ── Step 0: Clone or update repo ────────────────────────────────────────────

banner "Step 0/7 — Clone or update repository"

if [[ -d "$WORKSPACE/.git" ]]; then
    cd "$WORKSPACE"
    git pull --ff-only origin main 2>&1 || true
    check_ok "Repository updated (git pull)"
else
    git clone "$REPO_URL" "$WORKSPACE"
    check_ok "Repository cloned to $WORKSPACE"
fi

cd "$WORKSPACE"

# ── Step 1: System packages ─────────────────────────────────────────────────

banner "Step 1/7 — System packages"

apt-get update -qq
apt-get install -y -qq build-essential pkg-config libssl-dev git curl wget > /dev/null 2>&1

check_ok "System packages installed"

# ── Step 2: Verify CUDA ─────────────────────────────────────────────────────

banner "Step 2/7 — Verify CUDA"

if ! command -v nvcc &>/dev/null; then
    # Try common RunPod locations
    for candidate in /usr/local/cuda /usr/local/cuda-12; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            export PATH="${candidate}/bin:$PATH"
            export CUDA_PATH="$candidate"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    die "nvcc not found. Is this a RunPod pytorch template with CUDA?"
fi

export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
check_ok "nvcc found: CUDA ${CUDA_VER}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
    check_ok "GPU: $line"
done

# Ensure CUDA_PATH persists for cargo build
if ! grep -q 'CUDA_PATH' "$HOME/.bashrc" 2>/dev/null; then
    echo "export CUDA_PATH=${CUDA_PATH}" >> "$HOME/.bashrc"
    echo "export PATH=${CUDA_PATH}/bin:\$PATH" >> "$HOME/.bashrc"
fi

# ── Step 3: Rust toolchain ──────────────────────────────────────────────────

banner "Step 3/7 — Rust toolchain"

if ! command -v rustc &>/dev/null; then
    echo "  Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    check_ok "Rust installed: $(rustc --version)"
else
    check_ok "Rust already installed: $(rustc --version)"
fi

# Ensure cargo is on PATH for the rest of this script
source "$HOME/.cargo/env" 2>/dev/null || true

# ── Step 4: Python venv + pip packages ──────────────────────────────────────

banner "Step 4/7 — Python venv + packages"

if [[ ! -d "$WORKSPACE/.venv" ]]; then
    python3 -m venv "$WORKSPACE/.venv"
    check_ok "Created venv at .venv/"
else
    check_ok "Venv already exists at .venv/"
fi

source "$WORKSPACE/.venv/bin/activate"
check_ok "Activated venv (python: $(python --version))"

pip install --upgrade pip -q
pip install maturin numpy zstandard tokenizers -q

check_ok "Python packages installed"

# ── Step 5: Build Rust + CUDA ───────────────────────────────────────────────

banner "Step 5/7 — Build Rust core (with CUDA kernels)"

cd "$WORKSPACE/core"
cargo build --release --features cuda 2>&1

check_ok "cargo build --release --features cuda succeeded"

# ── Step 6: Build Python bindings ───────────────────────────────────────────

banner "Step 6/7 — Build Python bindings (maturin)"

cd "$WORKSPACE/python"
source "$WORKSPACE/.venv/bin/activate"
maturin develop --release --features cuda 2>&1

check_ok "maturin develop --release --features cuda succeeded"

# ── Step 7: Smoke test ──────────────────────────────────────────────────────

banner "Step 7/7 — Smoke test"

cd "$WORKSPACE/python"
source "$WORKSPACE/.venv/bin/activate"

python -c "import nl_hecate; print('  ✓ PyO3 import OK')"

if [[ -d "$WORKSPACE/python/data/dolmino_smoke" ]]; then
    echo "  Running smoke build (10 steps)..."
    python -u hecate.py --build --config configs/llama_smoke_test.json --max-steps 10
    check_ok "Smoke build passed (10 steps)"
else
    echo "  Skipping smoke build (no dolmino_smoke data — use --data-dir smoke with rsync)"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

banner "Setup complete"

echo ""
echo "  To activate the environment in a new shell:"
echo "    cd $WORKSPACE"
echo "    source .venv/bin/activate"
echo ""
echo "  To run a build:"
echo "    python -u hecate.py --build --config configs/<your_config>.json"
echo ""
