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
    """Same token list always produces the same hash (two independent calls)."""
    tokens = [1, 2, 3, 4, 5]
    h1 = _fnv1a_32(tokens)
    h2 = _fnv1a_32(list(tokens))   # new list object
    assert h1 == h2


def test_fnv1a_different_inputs_differ():
    """Different token lists produce different hashes (probabilistic)."""
    assert _fnv1a_32([1, 2, 3]) != _fnv1a_32([3, 2, 1])


# ── Wrap-around correctness ───────────────────────────────────────────────────

def test_restore_after_wrap(tmp_path):
    """restore() is correct after position wraps to 0.

    seq_len = pos // chunk_id breaks after a wrap (e.g. pos=64, chunk_id=3
    gives seq_len=21 instead of 64). Storing seq_len explicitly in the cursor
    avoids this. This test would fail with the old derivation.
    """
    _make_dataset(tmp_path, n_tokens=128)
    loader = BpeDataLoader(str(tmp_path))
    seq_len = 64

    loader.next_chunk(seq_len)   # pos=64,  chunk_id=1
    loader.next_chunk(seq_len)   # pos=128, chunk_id=2
    loader.next_chunk(seq_len)   # wraps: pos=64, chunk_id=3

    cursor = loader.cursor()
    assert cursor["position"] == 64
    assert cursor["chunk_id"] == 3
    assert cursor["seq_len"]  == 64   # stored, not derived

    loader2 = BpeDataLoader(str(tmp_path))
    loader2.restore(cursor)           # must not raise

    assert loader2.position  == 64
    assert loader2._chunk_id == 3
    assert loader2._seq_len  == 64


# ── Multi-slot sidecar (task_969aa6) ─────────────────────────────────────────

def test_multi_slot_cursor_roundtrip(tmp_path):
    """Per-slot cursors are saved and restored correctly for batch_size > 1.

    Simulates the loop.py multi-slot path:
      - Two slot loaders, each advanced a different number of chunks.
      - Sidecar written as {"slots": [cursor0, cursor1]}.
      - Fresh loaders reconstructed and restored from sidecar.
      - Positions and chunk_ids must match exactly.
    """
    _make_dataset(tmp_path, n_tokens=512)
    seq_len = 64
    n_slots = 2
    slot_size = 512 // n_slots

    # Build and advance slot loaders
    slots = []
    for b in range(n_slots):
        loader_b = BpeDataLoader(str(tmp_path))
        loader_b.position = b * slot_size
        slots.append(loader_b)

    slots[0].next_chunk(seq_len)   # slot 0: position = slot_size*0 + seq_len
    slots[0].next_chunk(seq_len)
    slots[1].next_chunk(seq_len)   # slot 1: position = slot_size*1 + seq_len

    saved_positions = [loader.position  for loader in slots]
    saved_chunk_ids = [loader._chunk_id for loader in slots]

    # Write sidecar ({"slots": [...]})
    ckpt_path = tmp_path / "model_step10.safetensors"
    sidecar   = Path(str(ckpt_path) + ".cursor.json")
    sidecar.write_text(json.dumps({"slots": [loader.cursor() for loader in slots]}, indent=2))

    # Reconstruct fresh slot loaders and restore
    saved = json.loads(sidecar.read_text())
    slot_cursors = saved["slots"]
    assert len(slot_cursors) == n_slots

    restored = []
    for b in range(n_slots):
        loader_b = BpeDataLoader(str(tmp_path))
        loader_b.position = b * slot_size      # initial partition start
        loader_b.restore(slot_cursors[b])      # overwrite with saved position
        restored.append(loader_b)

    for b in range(n_slots):
        assert restored[b].position  == saved_positions[b], \
            f"slot {b} position mismatch: {restored[b].position} != {saved_positions[b]}"
        assert restored[b]._chunk_id == saved_chunk_ids[b], \
            f"slot {b} chunk_id mismatch"


def test_single_slot_sidecar_unchanged(tmp_path):
    """batch_size=1 path writes a flat cursor dict, not a {'slots': [...]} wrapper."""
    _make_dataset(tmp_path)
    loader = BpeDataLoader(str(tmp_path))
    loader.next_chunk(64)

    ckpt_path = tmp_path / "model_step1.safetensors"
    sidecar   = Path(str(ckpt_path) + ".cursor.json")

    # Simulate loop.py single-slot save: bpe_loaders is empty
    bpe_loaders: list = []
    if bpe_loaders:
        sidecar.write_text(json.dumps({"slots": [loader.cursor() for loader in bpe_loaders]}, indent=2))
    else:
        sidecar.write_text(json.dumps(loader.cursor(), indent=2))

    saved = json.loads(sidecar.read_text())
    assert "slots" not in saved,         "single-slot sidecar must be a flat cursor dict"
    assert "position" in saved,          "single-slot sidecar must contain 'position'"
    assert saved["position"] == 64


def test_slot_count_mismatch_resets(tmp_path):
    """Sidecar slot count != bpe_loaders count falls back to partition start, no abort."""
    _make_dataset(tmp_path, n_tokens=512)
    seq_len = 64
    n_slots = 2
    slot_size = 512 // n_slots

    # Write a sidecar with 3 slots
    three_slot_cursors = [{"position": i * 50, "total_tokens": 512,
                           "content_hash": 0, "chunk_id": i,
                           "seq_len": seq_len, "dataset_path": str(tmp_path)}
                          for i in range(3)]
    ckpt_path = tmp_path / "model_old.safetensors"
    sidecar   = Path(str(ckpt_path) + ".cursor.json")
    sidecar.write_text(json.dumps({"slots": three_slot_cursors}, indent=2))

    # Now reconstruct with only 2 slots (batch_size changed)
    bpe_loaders = []
    for b in range(n_slots):
        loader_b = BpeDataLoader(str(tmp_path))
        loader_b.position = b * slot_size
        bpe_loaders.append(loader_b)

    saved = json.loads(sidecar.read_text())
    slot_cursors = saved.get("slots") if isinstance(saved, dict) else None

    # Mismatch: slot_cursors has 3 entries, bpe_loaders has 2 → do NOT restore
    if slot_cursors and len(slot_cursors) == len(bpe_loaders):
        for loader_b, cur in zip(bpe_loaders, slot_cursors, strict=True):
            loader_b.restore(cur)
    # else: fall through — loaders stay at partition-start positions

    # Loaders must be at partition-start (not the 3-slot sidecar positions)
    assert bpe_loaders[0].position == 0,          "slot 0 must reset to partition start"
    assert bpe_loaders[1].position == slot_size,  "slot 1 must reset to partition start"
