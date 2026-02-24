#!/usr/bin/env python3
"""DEPRECATED: Use hecate.py --chat / --prompt / --interactive instead.

This script is a backward-compatibility stub. All functionality has moved
to the unified entry point: python hecate.py

Examples:
    python hecate.py --chat --checkpoint checkpoints/model.json
    python hecate.py --prompt "Once upon a time" --checkpoint checkpoints/model.json
    python hecate.py --interactive --checkpoint checkpoints/model.json
    python hecate.py --chat --checkpoint checkpoints/model.json --learn --lr 0.0006
"""
import sys

print(
    "\n"
    "  serve.py is deprecated. Use the unified entry point:\n"
    "\n"
    "    python hecate.py --chat --checkpoint checkpoints/model.json\n"
    "    python hecate.py --prompt \"Once upon a time\" --checkpoint checkpoints/model.json\n"
    "    python hecate.py --interactive --checkpoint checkpoints/model.json\n"
    "\n"
    "  All serve.py flags are supported.\n",
    file=sys.stderr,
)
sys.exit(1)
