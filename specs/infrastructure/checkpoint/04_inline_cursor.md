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

```rust
enum CursorSource {
    StateFile,            // state file cursor matches loaded checkpoint
    SafetensorsSlots,     // BuildResumeState.stream_cursors (batch>1)
    SafetensorsLegacy,    // BuildResumeState.stream_cursor (batch=1 compat)
    LegacySidecar,        // <checkpoint>.cursor.json migration
    FreshStart,           // no cursor — start from position 0
}

fn select_resume_cursor_source(
    state_slots: &[StreamCursor],
    current_checkpoint: Option<&CurrentCheckpoint>,
    checkpoint_path: &Path,
    build_state: Option<&BuildResumeState>,
) -> CursorSource {
    // Guard: state file cursor only valid if it matches the loaded checkpoint
    if !state_slots.is_empty()
        && current_checkpoint.map(|cc| cc.path == checkpoint_path).unwrap_or(false)
    {
        return CursorSource::StateFile;
    }
    if let Some(bs) = build_state {
        if !bs.stream_cursors.is_empty() {
            return CursorSource::SafetensorsSlots;
        }
        if bs.stream_cursor.position > 0 {
            return CursorSource::SafetensorsLegacy;
        }
    }
    if checkpoint_path.with_extension("cursor.json").exists() {
        return CursorSource::LegacySidecar;
    }
    CursorSource::FreshStart
}
```

Steps 2-3 (SafetensorsSlots/Legacy) are the existing behavior. Step 1 (StateFile) is new with checkpoint identity guard. Step 4 (LegacySidecar) is migration support.

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
