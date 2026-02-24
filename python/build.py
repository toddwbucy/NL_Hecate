#!/usr/bin/env python3
"""DEPRECATED: Use hecate.py --build instead.

This script is a backward-compatibility stub. All functionality has moved
to the unified entry point: python hecate.py --build

Examples:
    python hecate.py --build --config configs/hope_60m.json
    python hecate.py --build --config configs/hope_60m.json --load checkpoints/model_step5000.json
    python hecate.py --build --steps 200  # uses built-in demo text
"""
import sys

print(
    "\n"
    "  build.py is deprecated. Use the unified entry point:\n"
    "\n"
    "    python hecate.py --build --config configs/hope_60m.json\n"
    "    python hecate.py --build --steps 200\n"
    "\n"
    "  All build.py flags are supported. Just add --build.\n",
    file=sys.stderr,
)
sys.exit(1)
