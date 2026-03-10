"""Data loading: mmap byte stream, BPE token stream, demo text."""

import json
import mmap
from pathlib import Path


class CursorMismatchError(Exception):
    """Raised when a cursor sidecar does not match the current dataset."""
    pass


class CursorOutOfBounds(Exception):
    """Raised when cursor position exceeds the current dataset size."""
    pass


def _fnv1a_32(tokens: list[int]) -> int:
    """FNV-1a 32-bit hash over a list of token IDs. Integrity canary only."""
    h = 0x811c9dc5
    for tok in tokens:
        for byte in tok.to_bytes(4, "little"):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
    return h


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


class BpeTokenStream:
    """Load pre-tokenized ShareGPT data (numpy arrays) and serve chunks.

    Manages a position cursor into the flat token/target arrays.
    Returns (input_ids, target_ids) per chunk, wrapping at corpus end.
    Masked targets: -1 in numpy -> vocab_size as Python int (triggers
    kernel skip via target >= vocab).
    """

    def __init__(self, data_dir: str, split: str = "train"):
        import numpy as np
        data_path = Path(data_dir)
        self._path = (data_path / f"{split}_tokens.npy").resolve()
        self.tokens = np.load(self._path)
        self.targets = np.load(data_path / f"{split}_targets.npy")
        if len(self.tokens) != len(self.targets):
            raise ValueError(
                f"{split} tokens ({len(self.tokens)}) != targets ({len(self.targets)})")

        with open(data_path / "meta.json") as f:
            self.meta = json.load(f)
        self.vocab_size = self.meta["vocab_size"]
        self.position = 0
        self.total_tokens = len(self.tokens)
        self._chunk_id = 0
        self._last_hash = 0
        self._seq_len = 0   # last seq_len passed to next_chunk; stored in cursor

    def next_chunk(self, seq_len: int) -> tuple[list[int], list[int]] | None:
        """Get next chunk of (input_ids, target_ids).

        Wraps position to 0 when remaining tokens < seq_len.
        Returns None only when total_tokens < seq_len (corpus too short).
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

        self._last_hash = _fnv1a_32(input_ids)
        self._chunk_id += 1
        self._seq_len = seq_len
        self.position = end
        return input_ids, target_ids

    def cursor(self) -> dict:
        """Return a serializable cursor capturing exact stream position.

        The content_hash is a FNV-1a hash of the last chunk served.
        On restore, this is checked to detect wrong dataset or corruption.
        seq_len is stored explicitly — recovery from pos//chunk_id breaks
        after position wraps.
        """
        return {
            "position":     self.position,
            "total_tokens": self.total_tokens,
            "content_hash": self._last_hash,
            "chunk_id":     self._chunk_id,
            "seq_len":      self._seq_len,
            "dataset_path": str(self._path),
        }

    def restore(self, cursor: dict) -> None:
        """Seek to saved cursor position and validate dataset integrity.

        Raises:
            CursorMismatchError: total_tokens mismatch (wrong dataset)
            CursorMismatchError: content_hash mismatch (corruption / wrong file)
            CursorMismatchError: content_hash non-zero but unverifiable (malformed cursor)
            CursorOutOfBounds:   position < 0 or position > total_tokens
        """
        saved_total = cursor.get("total_tokens", 0)
        if saved_total != self.total_tokens:
            raise CursorMismatchError(
                f"Dataset size mismatch: checkpoint has {saved_total:,} tokens, "
                f"current dataset has {self.total_tokens:,}. Wrong dataset?"
            )

        pos = cursor.get("position", 0)
        if pos < 0 or pos > self.total_tokens:
            raise CursorOutOfBounds(
                f"Cursor position {pos:,} is out of bounds for dataset size {self.total_tokens:,}."
            )

        # Validate content hash immediately using the stored seq_len.
        # seq_len is stored explicitly in the cursor (not derived from pos//chunk_id,
        # which breaks after position wraps).
        saved_hash = cursor.get("content_hash", 0)
        seq_len    = cursor.get("seq_len", 0)
        chunk_id   = cursor.get("chunk_id", 0)
        if saved_hash != 0:
            if seq_len <= 0 or pos < seq_len:
                raise CursorMismatchError(
                    f"Malformed cursor: content_hash is non-zero ({saved_hash:#010x}) "
                    f"but pos={pos:,} < seq_len={seq_len:,} — cannot verify integrity."
                )
            last_chunk = self.tokens[pos - seq_len : pos].tolist()
            actual_hash = _fnv1a_32(last_chunk)
            if actual_hash != saved_hash:
                raise CursorMismatchError(
                    f"Content hash mismatch at position {pos:,}: "
                    f"expected {saved_hash:#010x}, got {actual_hash:#010x}. "
                    "Wrong dataset or corrupted file?"
                )

        self.position   = pos
        self._chunk_id  = chunk_id
        self._seq_len   = seq_len
        self._last_hash = saved_hash

    def __len__(self) -> int:
        return self.total_tokens
