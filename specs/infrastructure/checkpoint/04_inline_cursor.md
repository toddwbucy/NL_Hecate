# 04 — Inline Cursor: State File as Authoritative Cursor Source

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Make the state file the single authoritative source for stream cursor state on resume, eliminating the need for `.cursor.json` sidecar files |
| Expects   | Spec 02 state file already stores cursor in `cursor.slots[]`; `save_checkpoint` already writes cursors to both safetensors metadata and state file; resume reads cursors only from safetensors `BuildResumeState` |
| Guarantees | Resume reads cursor from state file first; falls back to safetensors metadata if state file has no cursor; falls back to legacy `.cursor.json` sidecar if both are empty; no new sidecar files are created |
| Cost      | ~20 lines in feed.rs resume path; one migration helper for legacy sidecars |
| Trade-off | State file becomes a resume dependency (acceptable — it already is for identity/history). Safetensors metadata still carries cursors for standalone checkpoint portability |
| Position  | SFL-3 in the State File Lifecycle epic. Depends on spec 02 (state file schema). Supersedes `context_stream/01_cursor_serialization.md` sidecar protocol |
| Source    | IaC model lifecycle — cursor is model state, belongs in the lifecycle record |

## Current State

Cursors flow through three locations:

1. **Safetensors metadata** (`BuildResumeState.stream_cursors`): Written at every checkpoint. Read on resume. Authoritative today.
2. **State file** (`cursor.slots[]`): Written at every checkpoint (SFL-1). Never read on resume. Redundant today.
3. **Legacy sidecar** (`.cursor.json`): Defined in `context_stream/01_cursor_serialization.md`. Not written by Rust CLI. May exist from older Python runs.

## Design: Cursor Priority on Resume

When loading a checkpoint with `--resume` or `build.load`:

```
1. If state file exists AND cursor.slots is non-empty → use state file cursors
2. Else if BuildResumeState.stream_cursors is non-empty → use safetensors cursors
3. Else if BuildResumeState.stream_cursor.position > 0 → use single legacy cursor
4. Else if <checkpoint>.cursor.json exists → read legacy sidecar
5. Else → no cursor restoration (fresh start)
```

Steps 2-3 are the existing behavior. Step 1 is new. Step 4 is migration support.

## Legacy Sidecar Migration

If a `.cursor.json` sidecar exists alongside a checkpoint but no state file cursor is available:

1. Read the sidecar JSON
2. Map fields to `StreamCursor`: `position` → `position`, `chunk_id` → `chunk_id` (or 0), `content_hash` → `content_hash` (or 0)
3. Use the cursor for resume
4. On next checkpoint, cursor is written to the state file — sidecar is never updated

This is a one-way migration. The sidecar file is not deleted (user may want it for debugging).

## Implementation

1. After state file load in `feed.rs`, check `model_state.cursor.slots` — if non-empty AND resuming, override `resume_cursors`
2. Add `load_legacy_cursor(checkpoint_path)` helper in `state_file.rs` for `.cursor.json` fallback
3. Wire fallback chain: state file → safetensors → legacy sidecar → empty
