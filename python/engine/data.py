"""Data loading: mmap byte stream, BPE token stream, shard stream, demo text."""

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


class ShardTokenStream:
    """Load pre-tokenized uint16 memmap shards and serve chunks.

    Designed for the SmolLM corpus format: a directory of shard_NNN.bin files
    (flat uint16), a manifest.json with vocab_size, and a tokenizers/ subdirectory.
    Targets are derived from input shifted by 1 position (standard causal LM).

    Each shard is memory-mapped individually. Offset arithmetic maps a global
    position to the correct shard + local offset, keeping RAM usage near zero
    (OS page cache handles the rest).

    Implements the same interface as BpeTokenStream: next_chunk(), cursor(),
    restore(), __len__(), plus .tokens and .targets properties for val carving.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        import numpy as np
        data_path = Path(data_dir)
        shard_dir = data_path / split

        # Load manifest
        manifest_path = data_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json found in {data_path}. "
                "Expected SmolLM corpus format.")
        with open(manifest_path) as f:
            self._manifest = json.load(f)
        self.vocab_size = self._manifest["tokenizer"]["vocab_size"]

        # Discover and memory-map shards in sorted order
        shard_files = sorted(shard_dir.glob("shard_*.bin"))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard_*.bin files found in {shard_dir}")

        self._shards: list = []       # list of np.memmap
        self._shard_offsets: list[int] = []  # cumulative start position per shard
        cumulative = 0
        for sf in shard_files:
            mm = np.memmap(str(sf), dtype=np.uint16, mode='r')
            self._shards.append(mm)
            self._shard_offsets.append(cumulative)
            cumulative += len(mm)

        # Total usable tokens: last token has no target, so total_tokens = cumulative - 1
        self.total_tokens = cumulative - 1
        self._total_raw = cumulative  # raw token count across all shards

        self._path = str(shard_dir.resolve())
        self._split = split
        self.position = 0
        self._chunk_id = 0
        self._last_hash = 0
        self._seq_len = 0

        # Expose meta dict for compatibility with code that reads loader.meta
        self.meta = {"vocab_size": self.vocab_size}

    def _read_range(self, start: int, end: int) -> list[int]:
        """Read token IDs from global position [start, end) across shard boundaries."""
        import numpy as np
        result = []
        pos = start
        while pos < end:
            # Binary search for the shard containing pos
            shard_idx = self._find_shard(pos)
            local_start = pos - self._shard_offsets[shard_idx]
            shard_len = len(self._shards[shard_idx])
            local_end = min(local_start + (end - pos), shard_len)
            result.extend(self._shards[shard_idx][local_start:local_end].tolist())
            pos += (local_end - local_start)
        return result

    def _find_shard(self, global_pos: int) -> int:
        """Find which shard contains global_pos via binary search."""
        import bisect
        # _shard_offsets[i] is the start of shard i. We want the last shard
        # whose offset <= global_pos.
        idx = bisect.bisect_right(self._shard_offsets, global_pos) - 1
        return max(0, idx)

    def next_chunk(self, seq_len: int) -> tuple[list[int], list[int]] | None:
        """Get next chunk of (input_ids, target_ids).

        Wraps position to 0 when remaining tokens < seq_len.
        Returns None only when total_tokens < seq_len (corpus too short).
        Targets are input shifted by 1 position (standard causal LM).
        """
        if self.position + seq_len > self.total_tokens:
            self.position = 0  # wrap
        if self.total_tokens < seq_len:
            return None

        # Read seq_len + 1 tokens: first seq_len are input, last seq_len are target
        raw = self._read_range(self.position, self.position + seq_len + 1)
        input_ids = raw[:seq_len]
        target_ids = raw[1:seq_len + 1]

        self._last_hash = _fnv1a_32(input_ids)
        self._chunk_id += 1
        self._seq_len = seq_len
        self.position += seq_len
        return input_ids, target_ids

    def cursor(self) -> dict:
        """Return a serializable cursor capturing exact stream position."""
        return {
            "position":     self.position,
            "total_tokens": self.total_tokens,
            "content_hash": self._last_hash,
            "chunk_id":     self._chunk_id,
            "seq_len":      self._seq_len,
            "dataset_path": self._path,
        }

    def restore(self, cursor: dict) -> None:
        """Seek to saved cursor position and validate dataset integrity."""
        saved_total = cursor.get("total_tokens", 0)
        if saved_total != self.total_tokens:
            raise CursorMismatchError(
                f"Dataset size mismatch: checkpoint has {saved_total:,} tokens, "
                f"current dataset has {self.total_tokens:,}. Wrong dataset?"
            )

        pos = cursor.get("position", 0)
        if pos < 0 or pos > self.total_tokens:
            raise CursorOutOfBounds(
                f"Cursor position {pos:,} is out of bounds for dataset "
                f"size {self.total_tokens:,}."
            )

        saved_hash = cursor.get("content_hash", 0)
        seq_len    = cursor.get("seq_len", 0)
        chunk_id   = cursor.get("chunk_id", 0)
        if saved_hash != 0:
            if seq_len <= 0 or pos < seq_len:
                raise CursorMismatchError(
                    f"Malformed cursor: content_hash is non-zero "
                    f"({saved_hash:#010x}) but pos={pos:,} < "
                    f"seq_len={seq_len:,} — cannot verify integrity."
                )
            last_chunk = self._read_range(pos - seq_len, pos)
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

    @property
    def tokens(self):
        """Lazy numpy view over all shard tokens (for val carving compatibility).

        Returns a concatenated numpy array. This loads all shards into a single
        array — only call this for val carving (small slices), not for training.
        """
        import numpy as np
        if not hasattr(self, '_tokens_concat'):
            self._tokens_concat = np.concatenate(
                [s[:] for s in self._shards])
        return self._tokens_concat

    @property
    def targets(self):
        """Shifted-by-1 view of tokens (for val carving compatibility)."""
        return self.tokens[1:]

    def __len__(self) -> int:
        return self.total_tokens
