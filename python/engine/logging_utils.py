"""Structured logging and system utilities."""

import json
import os
import resource
import time


def rss_mb() -> float:
    """Current process RSS in MB (from /proc for accuracy)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])  # resident pages
        return pages * 4096 / (1024 * 1024)
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class JSONLLogger:
    """Append-only structured logger. One JSON object per line."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")

    def log(self, **fields):
        fields["timestamp"] = time.time()
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
