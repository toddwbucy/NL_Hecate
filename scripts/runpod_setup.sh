#!/usr/bin/env bash
# runpod_setup.sh — Clone, build, and configure NL_Hecate on a RunPod H100/H200 instance.
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
# After setup, prepare data and launch:
#   cd /workspace/NL_Hecate/python
#   source /workspace/NL_Hecate/.venv/bin/activate
#   python scripts/prepare_dolmino.py --target_tokens 500_000_000 --output data/dolmino_100b
#   CUDA_VISIBLE_DEVICES=0 python -u hecate.py --build --config configs/k4_chain_dolmino_d2048_64h_h200.json
#
# Idempotent: safe to re-run after failures or updates.

set -euo pipefail

WORKSPACE="/workspace/NL_Hecate"
REPO_URL="git@github.com:toddwbucy/NL_Hecate.git"

# ── Helpers ──────────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

check_ok() {
    echo "  [ok] $1"
}

die() {
    echo ""
    echo "  [FAILED] $1" >&2
    exit 1
}

# ── Step 0: Clone or update repo ────────────────────────────────────────────

banner "Step 0/8 — Clone or update repository"

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

banner "Step 1/8 — System packages"

apt-get update -qq
apt-get install -y -qq build-essential pkg-config libssl-dev git curl wget > /dev/null 2>&1

check_ok "System packages installed"

# ── Step 2: Verify CUDA ─────────────────────────────────────────────────────

banner "Step 2/8 — Verify CUDA"

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

banner "Step 3/8 — Rust toolchain"

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

banner "Step 4/8 — Python venv + packages"

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

banner "Step 5/8 — Build Rust core (with CUDA kernels)"

cd "$WORKSPACE/core"
cargo build --release --features cuda 2>&1

check_ok "cargo build --release --features cuda succeeded"

# ── Step 6: Build Python bindings ───────────────────────────────────────────

banner "Step 6/8 — Build Python bindings (maturin)"

cd "$WORKSPACE/python"
source "$WORKSPACE/.venv/bin/activate"
maturin develop --release --features cuda 2>&1

check_ok "maturin develop --release --features cuda succeeded"

# ── Step 7: Prepare Dolmino v1 data ────────────────────────────────────────

banner "Step 7/8 — Prepare Dolmino v1 training data"

cd "$WORKSPACE/python"
source "$WORKSPACE/.venv/bin/activate"

if [[ -f "$WORKSPACE/python/data/dolmino_100b/meta.json" ]]; then
    check_ok "Dolmino v1 data already exists"
else
    echo "  Downloading Dolmino-Mix shards and tokenizing (475M tokens)..."
    echo "  This streams from HuggingFace — requires internet access."
    echo ""
    echo "  If the source data is already on a mounted volume, you can skip this"
    echo "  and rsync pre-tokenized data instead:"
    echo "    rsync -av <source>:data/dolmino_100b/ data/dolmino_100b/"
    echo ""
    echo "  To prepare from raw shards on a local mount:"
    echo "    python scripts/prepare_dolmino.py \\"
    echo "      --source /path/to/dolmino_mix_100B \\"
    echo "      --target_tokens 500_000_000 \\"
    echo "      --output data/dolmino_100b"
    echo ""
    echo "  Skipping automatic preparation — prepare data manually before training."
fi

# ── Step 8: Smoke test ──────────────────────────────────────────────────────

banner "Step 8/8 — Smoke test"

cd "$WORKSPACE/python"
source "$WORKSPACE/.venv/bin/activate"

python -c "import nl_hecate; print('  [ok] PyO3 import OK')"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

# ── Done ─────────────────────────────────────────────────────────────────────

banner "Setup complete — $GPU_NAME ($GPU_MEM)"

echo ""
echo "  Environment:"
echo "    source $WORKSPACE/.venv/bin/activate"
echo "    cd $WORKSPACE/python"
echo ""
echo "  Prepare data (if not already done):"
echo "    Option A — rsync pre-tokenized data from home server:"
echo "      rsync -avz <home>:olympus/NL_Hecate/python/data/dolmino_100b/ data/dolmino_100b/"
echo ""
echo "    Option B — tokenize from raw Dolmino shards (if mounted):"
echo "      python scripts/prepare_dolmino.py --source /path/to/dolmino_mix_100B \\"
echo "        --target_tokens 500_000_000 --output data/dolmino_100b"
echo ""
echo "  Launch d=2048 scale-up build:"
echo "    CUDA_VISIBLE_DEVICES=0 nohup python -u hecate.py --build \\"
echo "      --config configs/k4_chain_dolmino_d2048_64h_h200.json \\"
echo "      > runs/k4_chain_dolmino_d2048_64h/build.log 2>&1 &"
echo ""
echo "  Monitor:"
echo "    tail -f runs/k4_chain_dolmino_d2048_64h/build.log"
echo "    nvidia-smi -l 5"
echo ""
echo "  Config: k=4 chain, d=2048, 64 heads (hd=32), 12 blocks, batch=4"
echo "  If OOM at batch=4, edit config and set batch_size to 2."
echo ""
