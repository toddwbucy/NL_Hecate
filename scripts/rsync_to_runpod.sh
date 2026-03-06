#!/usr/bin/env bash
# rsync_to_runpod.sh — Transfer datasets to a RunPod instance.
#
# Code is deployed via git clone (runpod_setup.sh handles that).
# This script ONLY transfers data that isn't in the git repo.
#
# Usage:
#   ./scripts/rsync_to_runpod.sh <user@host> --data-dir smoke|full [--port N] [--key PATH] [--dry-run]
#
# Examples:
#   ./scripts/rsync_to_runpod.sh root@216.243.220.220 --port 10775 --key ~/.ssh/id_ed25519 --data-dir smoke
#   ./scripts/rsync_to_runpod.sh root@216.243.220.220 --port 10775 --key ~/.ssh/id_ed25519 --data-dir full

set -uo pipefail

REMOTE_BASE="/workspace/NL_Hecate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments ──────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 <user@host> --data-dir smoke|full [--port N] [--key PATH] [--dry-run]"
    echo ""
    echo "  <user@host>      SSH destination (e.g., root@209.20.158.7)"
    echo "  --data-dir MODE  Data to transfer: smoke (8.9MB), full (6.5GB)"
    echo "  --port N         SSH port (default: 22)"
    echo "  --key PATH       SSH private key path (e.g., ~/.ssh/id_ed25519)"
    echo "  --dry-run        Show what would be transferred without actually doing it"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

SSH_DEST="$1"
shift

DATA_DIR=""
DRY_RUN=""
SSH_PORT="22"
SSH_KEY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            [[ $# -lt 2 ]] && { echo "Error: --port requires a number"; exit 1; }
            SSH_PORT="$2"
            shift 2
            ;;
        --key)
            [[ $# -lt 2 ]] && { echo "Error: --key requires a path"; exit 1; }
            SSH_KEY="$2"
            shift 2
            ;;
        --data-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: --data-dir requires an argument (smoke|full|none)"
                exit 1
            fi
            DATA_DIR="$2"
            if [[ "$DATA_DIR" != "smoke" && "$DATA_DIR" != "full" ]]; then
                echo "Error: --data-dir must be smoke or full (got: $DATA_DIR)"
                exit 1
            fi
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Error: unknown argument '$1'"
            usage
            ;;
    esac
done

# Build SSH command for rsync -e
SSH_CMD="ssh -p ${SSH_PORT}"
[[ -n "$SSH_KEY" ]] && SSH_CMD="$SSH_CMD -i $SSH_KEY"

# Common rsync flags: no-owner/no-group avoids chown failures on containers
RSYNC_OPTS="-avz --progress --no-owner --no-group"

# rsync exit code 23 = "some files could not be transferred" (e.g., permission attrs)
# Files themselves transferred fine — treat as success
run_rsync() {
    rsync $RSYNC_OPTS -e "$SSH_CMD" $DRY_RUN "$@"
    local rc=$?
    if [[ $rc -ne 0 && $rc -ne 23 ]]; then
        echo "Error: rsync failed with exit code $rc"
        exit $rc
    fi
}

# ── Sync code (explicit file list — only what's needed) ──────────────────────

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data-dir is required (smoke or full)"
    usage
fi

echo "═══════════════════════════════════════════════════════════"
echo "  NL_Hecate → RunPod data transfer"
echo "  Destination: ${SSH_DEST}:${REMOTE_BASE}"
echo "  SSH:         ${SSH_CMD}"
echo "  Data mode:   ${DATA_DIR}"
[[ -n "$DRY_RUN" ]] && echo "  *** DRY RUN — no files will be transferred ***"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Ensure remote data directory exists
$SSH_CMD "${SSH_DEST}" "mkdir -p ${REMOTE_BASE}/python/data"

case "$DATA_DIR" in
    smoke)
        echo "── Syncing smoke data (dolmino_smoke, ~8.9MB) ──────────"
        if [[ ! -d "${REPO_ROOT}/python/data/dolmino_smoke" ]]; then
            echo "Error: ${REPO_ROOT}/python/data/dolmino_smoke not found"
            exit 1
        fi
        run_rsync \
            "${REPO_ROOT}/python/data/dolmino_smoke/" \
            "${SSH_DEST}:${REMOTE_BASE}/python/data/dolmino_smoke/"
        ;;
    full)
        echo "── Syncing full dataset (dolmino_100b, ~6.5GB) ─────────"
        if [[ ! -d "${REPO_ROOT}/python/data/dolmino_100b" ]]; then
            echo "Error: ${REPO_ROOT}/python/data/dolmino_100b not found"
            exit 1
        fi
        run_rsync \
            "${REPO_ROOT}/python/data/dolmino_100b/" \
            "${SSH_DEST}:${REMOTE_BASE}/python/data/dolmino_100b/"
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Data transfer complete."
echo ""
echo "  Code is deployed via git clone (runpod_setup.sh)."
echo "  Data is now at ${REMOTE_BASE}/python/data/"
echo "═══════════════════════════════════════════════════════════"
