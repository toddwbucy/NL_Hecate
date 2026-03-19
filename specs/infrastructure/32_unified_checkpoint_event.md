# Spec 32: Unified Checkpoint Event

## CONTRACT

| Field | Value |
|-------|-------|
| **Purpose** | Consolidate three independent timing loops (save_every, tape_every, eval_every) into a single checkpoint event that fires atomically |
| **Expects** | BuildConfig with save_every > 0; CMS chunk_sizes for alignment validation |
| **Guarantees** | One checkpoint cadence controls save + tape + coherence sample; no sawtooth GPU dips between checkpoints |
| **Cost** | Zero runtime cost change — same operations, different scheduling |
| **Trade-off** | Loses independent tape/eval frequency; gains deterministic, predictable checkpoint behavior |
| **Position** | Python tier only (config.py, loop.py, evaluation.py, hecate.py). No Rust changes |
| **Source** | Empirical: GPU0 k=3 SmolLM 3.9% tok/s CV from independent timer dips |

## Motivation

Three independent timers create unpredictable GPU utilization dips:

```
step 1000: eval fires    → context save/restore + val forward
step 1024: tape fires    → CPU/GPU tape replay
step 5000: save fires    → host param download + safetensors write
step 5120: eval fires    → another context save/restore
```

The `eval_every` field is ontologically wrong — it implies a train/eval distinction
(CS-10 through CS-17). Using a separate validation set with `val_path` imports
conventional ML framing that NL explicitly rejects.

## Design

### Single knob: `save_every`

At every `save_every` step, the checkpoint event fires. It performs, in order:

1. **Save** — model state (safetensors) + cursor sidecar
2. **Tape diagnostic** — level health (alpha, theta, M-norm, dormancy)
3. **Coherence sample** — forward pass on next tokens from the *same build stream*, decode output

### Config changes

**Removed fields** (backward-compatible: accepted in JSON, logged as warnings):
- `eval_every` → mapped to `save_every` if `save_every` not set
- `eval_max_chunks` → replaced by single-chunk coherence sample
- `val_path`, `val_doc_starts_path` → no separate validation corpus
- `tape_every` → fires at checkpoint cadence, no independent frequency

**New field**:
- `coher_sample: bool = True` — whether to decode a coherence sample at checkpoint

**Retained fields**:
- `save_every: int` — the one knob (0 = disabled)
- `save_path: str` — checkpoint output path
- `tape_device: str` — "off", "cpu", "gpu" (controls tape method, not frequency)
- `probe_max_tokens: int` — tokens per coherence sample
- `probe_prompts: int` — number of prompts to probe

### Validation

- `save_every` must be > 0 for any checkpoint behavior
- `save_every` should be a multiple of `max(chunk_sizes)` for CMS alignment (warn if not)

### Backward compatibility

When loading JSON configs with old fields:
```python
if "eval_every" in build_cfg and "save_every" not in build_cfg:
    build_cfg["save_every"] = build_cfg["eval_every"]
    warn("eval_every is deprecated, using as save_every")
```

`tape_every`, `val_path`, `eval_max_chunks` are silently ignored with a deprecation warning.

### Coherence sample vs eval

The coherence sample is NOT evaluation. It's a qualitative health check:
- Uses the next chunk from the *build stream* (no separate val corpus)
- Decodes a short sample (probe_max_tokens tokens)
- No loss computation on a held-out set
- Fires at checkpoint cadence — not independently

This preserves CS-10 (no train/eval distinction): the model processes the same
continuous stream. We just peek at its output quality periodically.

## Scope

### Files to modify
- `python/engine/config.py` — remove deprecated fields, add coher_sample, backward compat
- `python/engine/loop.py` — merge eval/tape/save blocks into single checkpoint event
- `python/hecate.py` — remove --eval_every, --eval_max_chunks CLI args
- `python/configs/*.json` — update active configs to use save_every only

### Files NOT modified
- No Rust changes
- No CUDA changes
- `python/engine/evaluation.py` — functions retained (coherence sample still calls generate/probes)

## What gets removed from loop.py

1. Independent `eval_every` condition block (lines ~1428-1585)
2. Independent `tape_every` condition within eval block
3. Independent `save_every` condition block (lines ~1664-1707)
4. Window-local val re-carving logic (lines ~1986-1991)
5. `_window_val_tokens`, `_window_local_val` infrastructure
6. Separate `val_stream` initialization

## What the unified block looks like

```python
# ── Checkpoint event: save + tape + coherence ────────────────
if bcfg.save_every > 0 and step > 0 and step % bcfg.save_every == 0:
    # 1. Save checkpoint (safetensors + cursor sidecar)
    _save_checkpoint(...)

    # 2. Tape diagnostic (if tape_device != "off")
    _run_tape_diagnostic(...)

    # 3. Coherence sample (if coher_sample=True)
    if bcfg.coher_sample:
        _run_coherence_sample(...)

    # 4. Log everything together
    jsonl.log(event="checkpoint", step=step, ...)
```
