#!/bin/bash
# Launch curriculum training on GPU1: prepare data (if missing) → build → watchdog.
# Usage: bash launch_curriculum.sh [--smoke]
#
# --smoke: Run a quick 100-step smoke test instead of the full 100K run.
# Runs on GPU1 (CUDA_VISIBLE_DEVICES=1) so it can run alongside ShareGPT on GPU0.

set -euo pipefail
cd "$(dirname "$0")"

SMOKE=false
for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=true ;;
    esac
done

echo "============================================================"
echo "  NL-Hecate Curriculum Training Pipeline (GPU1)"
echo "============================================================"

# Step 1: Prepare curriculum data if missing
if [ ! -f "data/curriculum/meta.json" ]; then
    echo ""
    echo "Step 1: Preparing curriculum data..."
    echo "------------------------------------------------------------"
    python3 data/prepare_curriculum.py \
        --tokenizer data/sharegpt/tokenizer.json \
        --output data/curriculum
    echo ""
else
    echo ""
    echo "Step 1: Curriculum data already prepared (data/curriculum/meta.json exists)"
    echo ""
fi

# Step 2: Build on GPU1
echo "Step 2: Starting build on GPU1..."
echo "------------------------------------------------------------"

mkdir -p runs

if $SMOKE; then
    echo "  Mode: SMOKE TEST (100 steps on GPU1)"
    CUDA_VISIBLE_DEVICES=1 python3 hecate.py --build \
        --config configs/curriculum_100k.json \
        --steps 100 \
        --log_every 10 \
        --eval_every 50 \
        --eval_max_chunks 5 \
        --save_every 50 \
        --log_file runs/curriculum_smoke.jsonl
else
    echo "  Mode: FULL (100K steps on GPU1)"
    CUDA_VISIBLE_DEVICES=1 nohup python3 -u hecate.py --build \
        --config configs/curriculum_100k.json \
        --log_file runs/curriculum_100k.jsonl \
        > runs/curriculum_100k.log 2>&1 &
    PID=$!

    echo "$PID" > runs/curriculum_100k.pid.sh

    echo "  Background PID: $PID"
    echo "  Log: runs/curriculum_100k.log"
    echo "  JSONL: runs/curriculum_100k.jsonl"
    echo "  PID file: runs/curriculum_100k.pid.sh"
    echo ""
    echo "  Monitor:"
    echo "    tail -f runs/curriculum_100k.log"
    echo "    watch -n5 cat runs/curriculum_100k.heartbeat"
    echo ""
    echo "  Check if alive:"
    echo "    kill -0 \$(cat runs/curriculum_100k.pid.sh) 2>/dev/null && echo ALIVE || echo DEAD"

    # Mini watchdog: check every 5 min if process is still alive
    (
        while true; do
            sleep 300
            if [ -f runs/curriculum_100k.pid.sh ]; then
                WPID=$(cat runs/curriculum_100k.pid.sh)
                if ! kill -0 "$WPID" 2>/dev/null; then
                    echo "[$(date)] CRASH DETECTED: PID $WPID no longer running" >> runs/curriculum_100k.log
                    rm -f runs/curriculum_100k.pid.sh
                    exit 0
                fi
            else
                exit 0
            fi
        done
    ) &
    disown
fi

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
