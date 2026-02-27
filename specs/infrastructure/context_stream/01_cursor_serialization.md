# ContextStream Cursor Serialization

```text
CONTRACT
  Purpose:    A checkpoint must save three things: params + M state + cursor.
              Currently checkpoints save params + M state. The cursor is missing.
              This spec defines the cursor schema and the save/restore protocol.

              A ContextStream is a cursor into a sequence of tokens.
              The source is irrelevant — file, chat turn, tool output, pipe.
              The Rust primitive sees tokens. Nothing else.
              Creating mode-specific stream subtypes (BuildStream, ChatStream)
              would be CS-10 (train/eval distinction) in disguise.

  Expects:    BpeDataLoader managing a flat token array with a position cursor.
              Checkpoints written by engine/loop.py at configurable intervals.
              A content hash over the last chunk for dataset integrity checking.

  Guarantees: Every checkpoint write produces a matching sidecar cursor file.
              Resume seeks the data loader to the exact saved position.
              Wrong dataset on resume raises a clear error (hash mismatch).
              No new stream type abstractions are introduced.
              Correct resume for both build-mode (flat numpy array) and
              chat-mode (same stream, human-paced tokens) — identical primitive.

  Cost:       One small JSON file per checkpoint. Negligible I/O.

  Trade-off:  Sidecar file (not inline in checkpoint JSON) — zero Rust changes,
              zero PyO3 changes, zero recompile. The Rust StreamCursor schema
              already exists and is correct; this spec populates it from Python.
              Future: inline cursor into BuildResumeState.stream_cursor (v0.5.0).

  Position:   specs/infrastructure/context_stream/01_cursor_serialization.md
  Source:     CS-10 (no mode distinction); CS-11 (no training loop);
              00_context_stream.md §Cursor Invariant
```

---

## Why The Cursor Matters For NLMs (Not Transformers)

A transformer checkpoint is self-contained: restore params, re-feed any data
from any position, same result. Data is seen many times (epochs). Order is
shuffled. Restarting from position 0 is fine.

An NLM checkpoint is NOT self-contained without the cursor. Data is ordered
and seen exactly once. M at step N encodes the causal history of tokens 0..N.
If you restore params+M at step N but reset the data cursor to 0, the model
re-processes tokens 0..N with an M that already absorbed them. The outer-loop
weights and inner-loop memory state become contradictory.

**The cursor IS the session.** M + cursor together define a resumable state.

This also means chat inference needs no resend of previous turns — M already
holds them. Each new turn sends only new tokens. The cursor advances.

---

## Cursor Schema

```json
{
  "position":      <uint64>,   // byte offset into flat token array
  "total_tokens":  <uint64>,   // length of the array (integrity check)
  "content_hash":  <uint64>,   // FNV-1a hash of last chunk's input_ids
  "chunk_id":      <uint64>,   // monotonic chunk counter (matches Conductor pulse)
  "dataset_path":  <string>    // absolute path of the numpy file (human readable)
}
```

**content_hash** is a FNV-1a hash over the last chunk's token IDs. On restore,
if the hash of the chunk at `position - seq_len` does not match, resume raises
`CursorMismatchError`. This catches resuming on the wrong dataset or a
corrupted file without reading the full array.

---

## Interface: BpeDataLoader

```python
class BpeDataLoader:

    def cursor(self) -> dict:
        """Return a serializable cursor capturing current stream position.

        Returns:
            {
              "position":     self.position,
              "total_tokens": self.total_tokens,
              "content_hash": <fnv1a of last chunk>,   # 0 if no chunk served yet
              "chunk_id":     self._chunk_id,
              "dataset_path": str(self._path.resolve()),
            }
        """

    def restore(self, cursor: dict) -> None:
        """Seek to saved position and validate integrity.

        Raises:
            CursorMismatchError  if total_tokens mismatch (wrong dataset)
            CursorMismatchError  if content_hash mismatch (corruption or
                                 wrong file at same path)
            CursorOutOfBounds    if position > total_tokens
        """
```

---

## Sidecar File Protocol

Sidecar path: `<checkpoint_path>.cursor.json`

Examples:
```text
checkpoints/model_step30000.json         ← params + M state
checkpoints/model_step30000.json.cursor.json   ← cursor
```

**On save** (every checkpoint write, including the final one):
```python
cursor = bpe_loader.cursor()
sidecar = Path(ckpt_path).with_suffix('.json.cursor.json')
sidecar.write_text(json.dumps(cursor, indent=2))
```

**On resume** (before first step, after params/M restore):
```python
sidecar = Path(bcfg.load).with_suffix('.json.cursor.json')
if sidecar.exists():
    cursor = json.loads(sidecar.read_text())
    bpe_loader.restore(cursor)
    print(f"  Stream position: {cursor['position']:,} / {cursor['total_tokens']:,} tokens")
else:
    print("  WARNING: no cursor sidecar found — starting from position 0")
    # Acceptable for warm-start (donor checkpoint from different run)
```

**Missing sidecar is not an error** — warm-start checkpoints (e.g. SwiGLU
donor loaded for Titans Phase 2) intentionally reset the data cursor to 0.
The warning is informational only.

---

## FNV-1a Hash (Python implementation)

```python
def _fnv1a_32(tokens: list[int]) -> int:
    """FNV-1a 32-bit hash over a list of token IDs."""
    h = 0x811c9dc5
    for tok in tokens:
        for byte in tok.to_bytes(4, 'little'):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
    return h
```

Fast, no dependencies, sufficient for an integrity canary. Not cryptographic.

---

## Files To Modify

| File | Change |
|------|--------|
| `engine/data.py` | Add `cursor()`, `restore()`, `_fnv1a_32()`, `CursorMismatchError`, `_chunk_id`, `_last_hash`, `_path` tracking |
| `engine/loop.py` | Write sidecar on every checkpoint save; restore from sidecar on resume |

No Rust changes. No PyO3 changes. No recompile required.

---

## Verification

```bash
# 1. Build for 200 steps, kill
CUDA_VISIBLE_DEVICES=1 python hecate.py --build --config configs/llama_smoke_test.json \
  --save_every 100 2>&1 | head -50
# Confirm: checkpoints/smoke_step100.json.cursor.json exists

# 2. Resume and verify position printed
python hecate.py --build --config configs/llama_smoke_test.json --load checkpoints/smoke_step100.json
# Expect: "Stream position: 51,200 / <total> tokens" (100 steps × 512 seq_len)

# 3. Wrong dataset smoke test
cp checkpoints/smoke_step100.json.cursor.json /tmp/bad.cursor.json
# Edit position to be valid but content_hash wrong
# Expect: CursorMismatchError on resume
```
