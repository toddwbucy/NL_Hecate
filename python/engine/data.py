"""Data loading: mmap byte stream, BPE ShareGPT loader, demo text."""

import json
import mmap
from pathlib import Path


DEMO_TEXT = (
    "the cat sat on the mat. "
    "the dog ran in the park. "
    "birds fly high in the sky. "
    "fish swim deep in the sea. "
) * 10


def load_binary_tokens(path: str) -> list[int]:
    """Load a binary file where each byte IS a token ID."""
    with open(path, "rb") as f:
        return list(f.read())


class MmapTokenStream:
    """Memory-mapped token stream for datasets that exceed RAM.

    Maps a binary file (one byte = one token) via mmap. Random access
    without loading the full file. Implements the same interface as
    a list[int] for indexing.
    """

    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._len = self._mm.size()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self._mm[idx])
        return self._mm[idx]

    def close(self):
        self._mm.close()
        self._f.close()


class BpeDataLoader:
    """Load pre-tokenized ShareGPT data (numpy arrays) and serve chunks.

    Manages a position cursor into the flat token/target arrays.
    Returns (input_ids, target_ids) per chunk, wrapping at corpus end.
    Masked targets: -1 in numpy -> vocab_size as Python int (triggers
    kernel skip via target >= vocab).
    """

    def __init__(self, data_dir: str, split: str = "train"):
        import numpy as np
        data_path = Path(data_dir)
        self.tokens = np.load(data_path / f"{split}_tokens.npy")
        self.targets = np.load(data_path / f"{split}_targets.npy")
        assert len(self.tokens) == len(self.targets), \
            f"tokens ({len(self.tokens)}) != targets ({len(self.targets)})"

        with open(data_path / "meta.json") as f:
            self.meta = json.load(f)
        self.vocab_size = self.meta["vocab_size"]
        self.position = 0
        self.total_tokens = len(self.tokens)

    def next_chunk(self, seq_len: int) -> tuple[list[int], list[int]] | None:
        """Get next chunk of (input_ids, target_ids).

        Returns None if remaining tokens < seq_len (wraps on next call).
        Masked targets (-1) are converted to vocab_size for the kernel.
        """
        if self.position + seq_len > self.total_tokens:
            self.position = 0  # wrap
        if self.total_tokens < seq_len:
            return None

        end = self.position + seq_len
        input_ids = self.tokens[self.position:end].tolist()
        raw_targets = self.targets[self.position:end]

        # Convert -1 (masked) -> vocab_size (kernel skip sentinel)
        target_ids = []
        for t in raw_targets:
            target_ids.append(int(t) if t >= 0 else self.vocab_size)

        self.position = end
        return input_ids, target_ids

    def __len__(self) -> int:
        return self.total_tokens
