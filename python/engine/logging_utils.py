"""Structured logging and system utilities."""

import json
import os
import resource
import sys
import time


def rss_mb() -> float:
    """Current process RSS in MB (from /proc for accuracy)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])  # resident pages
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 * 1024)
    except Exception:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return ru_maxrss / (1024 * 1024)  # bytes on macOS
        return ru_maxrss / 1024.0  # kilobytes on Linux


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
