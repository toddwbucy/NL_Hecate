"""Tests for BpeDataLoader cursor serialization (task_c99edc).

Covers: cursor(), restore(), CursorMismatchError, CursorOutOfBounds,
        sidecar round-trip, hash validation, warm-start behaviour.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from engine.data import BpeDataLoader, CursorMismatchError, CursorOutOfBounds, _fnv1a_32


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_dataset(tmp_path: Path, n_tokens: int = 1024, vocab_size: int = 256,
                  seed: int = 42, split: str = "train") -> Path:
    """Write minimal train_tokens.npy / train_targets.npy / meta.json."""
    rng = np.random.RandomState(seed)
    tokens  = rng.randint(4, vocab_size, size=n_tokens, dtype=np.uint32)
    targets = np.roll(tokens, -1).astype(np.int32)
    np.save(tmp_path / f"{split}_tokens.npy",  tokens)
    np.save(tmp_path / f"{split}_targets.npy", targets)
    meta = {
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
        "special_tokens": {"<|endoftext|>": 3},
        "train": {"split": split, "documents": 1, "total_tokens": n_tokens,
                  "valid_targets": n_tokens, "masked_targets": 0, "mask_ratio": 0.0},
        "val":   {"split": "val",   "documents": 0, "total_tokens": 0,
                  "valid_targets": 0, "masked_targets": 0, "mask_ratio": 0.0},
        "seed": seed, "val_ratio": 0.0, "source": "test",
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


# ── cursor() tests ────────────────────────────────────────────────────────────

def test_cursor_initial_state(tmp_path):
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    c = loader.cursor()
    assert c["position"]     == 0
    assert c["total_tokens"] == 1024
    assert c["content_hash"] == 0       # no chunk served yet
    assert c["chunk_id"]     == 0


def test_cursor_after_chunks(tmp_path):
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    seq_len = 64
    loader.next_chunk(seq_len)
    loader.next_chunk(seq_len)
    c = loader.cursor()
    assert c["position"]  == seq_len * 2
    assert c["chunk_id"]  == 2
    assert c["content_hash"] != 0


# ── restore() round-trip ─────────────────────────────────────────────────────

def test_restore_position_correct(tmp_path):
    """restore() seeks to saved position; next chunk starts there."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    seq_len = 64

    loader.next_chunk(seq_len)
    cursor = loader.cursor()

    loader2 = BpeDataLoader(str(tmp_path))
    loader2.restore(cursor)

    assert loader2.position  == seq_len
    assert loader2._chunk_id == 1

    # Both loaders should serve the same next chunk
    chunk_orig = loader.next_chunk(seq_len)
    chunk_rest = loader2.next_chunk(seq_len)
    assert chunk_orig == chunk_rest


def test_restore_zero_position(tmp_path):
    """restore() with position=0 (warm-start) succeeds silently."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    cursor = loader.cursor()          # position=0, content_hash=0, chunk_id=0
    loader.restore(cursor)            # must not raise
    assert loader.position == 0


# ── CursorMismatchError: wrong dataset ───────────────────────────────────────

def test_restore_wrong_dataset_size(tmp_path):
    """total_tokens mismatch raises CursorMismatchError."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    loader.next_chunk(64)
    cursor = loader.cursor()
    cursor["total_tokens"] = 9999     # wrong size

    loader2 = BpeDataLoader(str(tmp_path))
    with pytest.raises(CursorMismatchError, match="Dataset size mismatch"):
        loader2.restore(cursor)


def test_restore_hash_mismatch(tmp_path):
    """Corrupted content_hash raises CursorMismatchError immediately on restore."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    loader.next_chunk(64)
    cursor = loader.cursor()
    cursor["content_hash"] = 0xDEADBEEF   # corrupt hash

    loader2 = BpeDataLoader(str(tmp_path))
    with pytest.raises(CursorMismatchError, match="Content hash mismatch"):
        loader2.restore(cursor)


# ── CursorOutOfBounds ─────────────────────────────────────────────────────────

def test_restore_out_of_bounds(tmp_path):
    """position > total_tokens raises CursorOutOfBounds."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    cursor = loader.cursor()
    cursor["position"] = 99999        # beyond dataset

    loader2 = BpeDataLoader(str(tmp_path))
    with pytest.raises(CursorOutOfBounds):
        loader2.restore(cursor)


# ── Sidecar file round-trip ───────────────────────────────────────────────────

def test_sidecar_written_and_read(tmp_path):
    """Sidecar JSON is written alongside checkpoint and can be round-tripped."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    seq_len = 64
    for _ in range(3):
        loader.next_chunk(seq_len)

    ckpt_path = tmp_path / "model_step3.safetensors"
    sidecar   = Path(str(ckpt_path) + ".cursor.json")

    sidecar.write_text(json.dumps(loader.cursor(), indent=2))
    assert sidecar.exists()

    saved = json.loads(sidecar.read_text())
    loader2 = BpeDataLoader(str(tmp_path))
    loader2.restore(saved)            # must not raise

    assert loader2.position  == seq_len * 3
    assert loader2._chunk_id == 3


# ── fnv1a hash consistency ────────────────────────────────────────────────────

def test_fnv1a_deterministic():
    """Same token list always produces the same hash."""
    tokens = [1, 2, 3, 4, 5]
    assert _fnv1a_32(tokens) == _fnv1a_32(tokens)


def test_fnv1a_different_inputs_differ():
    """Different token lists produce different hashes (probabilistic)."""
    assert _fnv1a_32([1, 2, 3]) != _fnv1a_32([3, 2, 1])
