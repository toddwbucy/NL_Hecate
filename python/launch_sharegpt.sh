#!/bin/bash
# Launch end-to-end ShareGPT training: prepare data (if missing) → build → log summary.
# Usage: bash launch_sharegpt.sh [--steps N] [--smoke]
#
# --smoke: Run a quick 100-step smoke test instead of the full 100K run.

set -euo pipefail
cd "$(dirname "$0")"

STEPS=""
SMOKE=false

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=true ;;
        --steps=*) STEPS="${arg#*=}" ;;
        --steps) shift; STEPS="$1" ;;
    esac
done

echo "============================================================"
echo "  NL-Hecate ShareGPT Training Pipeline"
echo "============================================================"

# Step 1: Prepare data if missing
if [ ! -f "data/sharegpt/meta.json" ]; then
    echo ""
    echo "Step 1: Preparing ShareGPT data..."
    echo "------------------------------------------------------------"
    python3 data/prepare_sharegpt.py
    echo ""
else
    echo ""
    echo "Step 1: ShareGPT data already prepared (data/sharegpt/meta.json exists)"
    echo ""
fi

# Step 2: Build
echo "Step 2: Starting build..."
echo "------------------------------------------------------------"

if $SMOKE; then
    echo "  Mode: SMOKE TEST (100 steps)"
    python3 build.py \
        --config configs/sharegpt_32k.json \
        --steps 100 \
        --log_every 10 \
        --eval_every 50 \
        --eval_max_chunks 5 \
        --save_every 50 \
        --gpu \
        --log_file runs/sharegpt_smoke.jsonl
elif [ -n "$STEPS" ]; then
    echo "  Mode: CUSTOM ($STEPS steps)"
    python3 build.py \
        --config configs/sharegpt_32k.json \
        --steps "$STEPS" \
        --gpu \
        --log_file runs/sharegpt_32k.jsonl
else
    echo "  Mode: FULL (100K steps)"
    nohup python3 -u build.py \
        --config configs/sharegpt_32k.json \
        --gpu \
        --log_file runs/sharegpt_32k.jsonl \
        > runs/sharegpt_32k.log 2>&1 &
    PID=$!
    echo "  Background PID: $PID"
    echo "  Log: runs/sharegpt_32k.log"
    echo "  JSONL: runs/sharegpt_32k.jsonl"
    echo ""
    echo "  Monitor with: tail -f runs/sharegpt_32k.log"
    echo "  Or: python3 -c \"import json; [print(json.loads(l)['loss']) for l in open('runs/sharegpt_32k.jsonl') if '\\\"step\\\"' in l]\""
fi

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
