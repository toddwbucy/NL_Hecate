# BPE Checkpoint Context Persistence

```text
CONTRACT
  Purpose:    Persist M_l memory matrices (ContextState) in BPE-path training
              checkpoints so that post-hoc diagnostic tools (spec 10: memory
              manifold analysis) can read M_l without access to a live model.

  Expects:    - A BPE training run using BpeTokenStream (sharegpt, dolmino
                formats) with a Conductor that has no attached VecStream
              - A live ContextState containing k levels of M_l: [d×d] each
              - Sidecar .cursor.json managing data position (unchanged)

  Guarantees: - save_checkpoint_with_context serialises M_l into the
                safetensors __metadata__ build_state field, identical format
                to save_build_checkpoint
              - StreamCursor in the serialised build_state is zeroed (BPE
                position is managed by sidecar, not by conductor stream)
              - load_build_checkpoint on a BPE checkpoint returns
                build_state.context_memory correctly; the caller ignores
                stream_cursor and reads position from the sidecar as before
              - Old BPE checkpoints (no build_state) continue to work:
                context starts fresh, backward compatible

  Cost:       Adds k*d*d*4 bytes to each checkpoint (k=4, d=256: +1MB).
              Zero overhead on the hot training path; serialisation only at
              save_every cadence.

  Trade-off:  StreamCursor in the BPE build_state is semantically empty
              (all zeros). Callers must not use it for position resume;
              sidecar .cursor.json remains the authoritative BPE cursor.
              This is a deliberate split: checkpoint = model state;
              sidecar = data-loader position.

  Position:   Python tier (PyO3 binding) + loop.py. No Rust core change.
              The new binding reuses the existing safetensors BuildResumeState
              serialisation path with a dummy StreamCursor.

  Source:     No paper equation — pure infrastructure. Motivated by spec 10
              (memory_manifold_analysis) Module 2 and Module 4, which require
              M_l from checkpoints to compute effective rank and subspace
              alignment respectively.
              HADES: hecate_specs/memory-manifold-analysis (consumer spec)
```

---

## Problem

The BPE training path (`use_bpe = True`, formats: sharegpt, dolmino) manages
data position via a per-slot sidecar `.cursor.json` rather than a `VecStream`
attached to the `Conductor`. The existing `save_build_checkpoint` PyO3 binding
enforces `conductor.has_stream()` before serialising, returning an error for
streamless BPE conductors:

```text
"save_build_checkpoint requires an attached stream on the Conductor"
```

As a result, BPE periodic checkpoints call `save_checkpoint(path, params, cfg)`
which drops the `ContextState` on the floor. The M_l matrices — which accumulate
continuously across all training steps when `memory_reset = "carry_forward"` —
are alive in GPU memory but never serialised.

This blocks Module 2 (effective rank) and Module 4 (subspace alignment) of the
memory manifold analysis tool from running on any BPE-path checkpoint.

## Solution

### New PyO3 binding: `save_checkpoint_with_context`

Add a new binding in `python/src/lib.rs` that:

1. Does NOT require an attached stream on the conductor
2. Builds `BuildResumeState` with:
   - `conductor`: `ConductorState` from `conductor.inner.state()` (step, k, chunk_sizes)
   - `stream_cursor`: `StreamCursor::zeroed()` — dummy value, BPE position lives in sidecar
   - `context`: clone of `context.inner` — the real M_l matrices
   - `global_step`: `conductor.inner.step()`
3. Calls the existing `rust_save_build_checkpoint` — no Rust core change needed

```rust
// python/src/lib.rs (pseudocode)
fn save_checkpoint_with_context(
    path: &str,
    params: &MAGParams,
    cfg: &MAGConfig,
    conductor: &Conductor,
    context: &ContextState,
) -> PyResult<()> {
    // Guard: reject stream-backed conductors — this API is BPE/streamless only.
    if conductor.inner.has_stream() {
        return Err(PyValueError::new_err(
            "save_checkpoint_with_context is for streamless BPE conductors; \
             use save_build_checkpoint for stream-backed conductors"
        ));
    }
    let build_state = RustBuildResumeState {
        conductor: conductor.inner.state(),        // step, k, chunk_sizes
        stream_cursor: StreamCursor::zeroed(),      // BPE: position in sidecar
        context: context.inner.clone(),             // M_l matrices
        global_step: conductor.inner.step(),
    };
    rust_save_build_checkpoint(
        std::path::Path::new(path), &params.inner, &cfg.inner, build_state,
    ).map_err(|e| PyValueError::new_err(format!("save_checkpoint_with_context failed: {e}")))
}
```

### `loop.py` save path

Replace the BPE checkpoint branch:

```python
# Before
if use_bpe:
    nl_hecate.save_checkpoint(ckpt_path, params, cfg)
else:
    nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)

# After
if use_bpe:
    nl_hecate.save_checkpoint_with_context(ckpt_path, params, cfg, conductor, context)
else:
    nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)
```

This applies to both the periodic save (`save_every`) and the final save at end
of training.

### `loop.py` load path

The BPE load path currently always creates a fresh `ContextState`:

```python
# Before (line ~307)
if use_bpe:
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
    context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
```

After this fix, when resuming from a new-format BPE checkpoint, restore M_l:

```python
# After
if use_bpe:
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
    if bcfg.load and build_state is not None and "context_memory" in build_state:
        context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
        context.set_memory(build_state["context_memory"])
        target_step = int(build_state.get("conductor_step", 0))
        while conductor.step < target_step:
            conductor.advance()
    else:
        context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
```

`build_state` is populated in the BPE resume block by attempting
`nl_hecate.load_build_checkpoint(path)` first and falling back to
`nl_hecate.load_checkpoint(path)` on exception. The fallback produces
`build_state = None`, which the context restore block handles via the
backward-compatible fresh-init path.

---

## Cursor Invariant

The `.cursor.json` sidecar remains the authoritative BPE data-loader position.
The zeroed `StreamCursor` in the build_state is never used for position resume
on the BPE path. The invariant is:

```text
BPE checkpoint resume:
  model state  ← build_state.context (M_l)
  data position ← .cursor.json sidecar (unchanged)
  conductor step ← build_state.conductor.step

Non-BPE (ContextStream) resume:
  model state  ← build_state.context (M_l)
  data position ← build_state.stream_cursor.position
  conductor step ← build_state.conductor.step
```

---

## Backward Compatibility

Old BPE checkpoints (saved by `save_checkpoint`, no `build_state`):
- `load_build_checkpoint` returns `build_state = None`
- BPE load path detects `build_state is None` → creates fresh `ContextState`
- Training continues from fresh M_l (same behaviour as today)

New BPE checkpoints (saved by `save_checkpoint_with_context`):
- `load_build_checkpoint` returns `build_state` with `context_memory`
- BPE load path restores M_l from `build_state["context_memory"]`

---

## Falsification

This spec is falsified (implementation incorrect) if any of:

1. A BPE checkpoint saved by `save_checkpoint_with_context` is loaded and
   `build_state["context_memory"]` is not a list of k flat `Vec<f32>` of
   length `d*d` each.

2. A BPE run resumed from a new-format checkpoint produces a different loss
   trajectory than a fresh run at the same step — M_l restore is idempotent
   (resuming should preserve continuity).

3. Loading a new-format BPE checkpoint raises an exception in
   `load_build_checkpoint`.

4. The sidecar `.cursor.json` is modified or invalidated by using
   `save_checkpoint_with_context`.

---

## Files Modified

| File | Change |
|------|--------|
| `python/src/lib.rs` | New `save_checkpoint_with_context` PyO3 binding (~20 lines) |
| `python/engine/loop.py` | BPE save: `save_checkpoint` → `save_checkpoint_with_context` (2 callsites) |
| `python/engine/loop.py` | BPE load: restore `context_memory` from `build_state` when present |
