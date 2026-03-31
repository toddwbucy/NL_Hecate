# 67 — Composable Checkpoint Pipeline

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Replace hardcoded checkpoint logic with a config-driven pipeline of composable actions, enabling per-run control over what happens at checkpoint boundaries |
| Expects   | Current monolithic checkpoint block in `cli/src/run.rs` (save → probe); scattered config fields (`save_every`, `profile_every`, `tokenizer_path`, `probe_max_tokens`, `probe_temperature`) |
| Guarantees | Checkpoint actions are independently togglable from config JSON; existing configs continue to work via `#[serde(default)]` backward compat; no action runs unless explicitly listed |
| Cost      | New `CheckpointConfig` struct + action dispatch enum; migration of ~5 scattered fields into one object |
| Trade-off | Slightly more complex config schema vs full composability. Enum-based dispatch vs trait-object plugins — enum is simpler and sufficient for the known action set |
| Position  | Fixes the probe-cannot-be-disabled bug; prerequisite for future checkpoint actions (heatmaps, eval sweeps, ONNX export) |
| Source    | task_966309; d=2048 run blocked by 30-min probe overhead at every checkpoint |

## Problem

The current checkpoint path in `run.rs` is a hardcoded sequence:

```rust
if do_checkpoint {
    save_checkpoint(...);           // always runs
    if let Some(ref tok_path) = cfg.build.tokenizer_path {
        run_inline_probes(...);     // runs if tokenizer_path is set
    }
}
```

Issues:
1. **Probes can't be disabled** without removing `tokenizer_path` from config — `probe_max_tokens: 0` is ignored
2. **No ordering control** — save always runs first, probes always second
3. **No extensibility** — adding a new checkpoint action means editing the run loop and adding more scattered fields to `BuildConfig`
4. **Profile is decoupled** — `profile_every` runs on its own cadence, not composable with checkpoint
5. **Config pollution** — `BuildConfig` has 6+ fields for checkpoint-adjacent behavior (`save_every`, `profile_every`, `tokenizer_path`, `probe_max_tokens`, `probe_temperature`, `cms_sidecar`)

## Design: Checkpoint Action Pipeline

### Config schema

New `checkpoint` object in the build config. The save itself is a fixed anchor point.
Actions compose around it in two phases: `before_save` and `after_save`.

```json
{
    "build": {
        "checkpoint": {
            "every": 500,
            "before_save": [],
            "after_save": ["cms_sidecar"]
        }
    }
}
```

With per-action config when needed:

```json
{
    "build": {
        "checkpoint": {
            "every": 500,
            "before_save": ["profile"],
            "after_save": [
                "cms_sidecar",
                { "type": "probe", "max_tokens": 64, "temperature": 0.8, "tokenizer": "path/to/tokenizer.json" }
            ]
        }
    }
}
```

**Why before/after matters:**
- `before_save`: actions that should reflect the exact model state being checkpointed
  (e.g., profile timing, M-norm snapshot). If they fail, the checkpoint hasn't been
  written yet — the run can continue from the previous checkpoint.
- `after_save`: actions that are safe to run after the checkpoint is on disk
  (e.g., probes, eval sweeps). If they crash or hang, the checkpoint is already safe.
  Expensive/risky actions belong here.

The save itself always runs — it is not an action, it is the anchor. Actions in each
phase execute **in order** as listed in the config.

### Action registry

```rust
enum CheckpointAction {
    Probe { max_tokens: usize, temperature: f32, tokenizer: String },
    Profile,
    CmsSidecar,
}

struct CheckpointConfig {
    every: usize,
    before_save: Vec<CheckpointAction>,
    after_save: Vec<CheckpointAction>,
}
```

Each variant maps to an existing standalone function. The run loop becomes:

```rust
if do_checkpoint {
    // Pre-save actions
    for action in &checkpoint_cfg.before_save {
        run_checkpoint_action(action, ...);
    }

    // The anchor — always runs
    save_checkpoint(...);

    // Post-save actions
    for action in &checkpoint_cfg.after_save {
        run_checkpoint_action(action, ...);
    }
}

fn run_checkpoint_action(action: &CheckpointAction, ...) {
    match action {
        CheckpointAction::Probe { max_tokens, temperature, tokenizer } => {
            let snapshot = gpu_params.to_host(d, v, k);
            run_inline_probes(&snapshot, ..., *max_tokens, *temperature, tokenizer, ...);
        }
        CheckpointAction::Profile => write_step_profile(...),
        CheckpointAction::CmsSidecar => write_cms_sidecar(...),
    }
}
```

### Deserialization

`CheckpointAction` deserializes from either a string or an object:

```rust
// "save" → CheckpointAction::Save
// { "type": "probe", "max_tokens": 64 } → CheckpointAction::Probe { max_tokens: 64, ... }
```

This uses `#[serde(untagged)]` with two variants: a string enum and a tagged struct.

### Backward compatibility

When the `checkpoint` object is absent, synthesize defaults from the legacy fields:

```rust
#[serde(default)]
pub checkpoint: Option<CheckpointConfig>,
```

In `run.rs`, if `checkpoint` is `None`, build the legacy pipeline:

```rust
let checkpoint_cfg = cfg.build.checkpoint.unwrap_or_else(|| {
    let mut before_save = Vec::new();
    let mut after_save = Vec::new();

    if cfg.build.cms_sidecar {
        after_save.push(CheckpointAction::CmsSidecar);
    }
    if let Some(ref tok_path) = cfg.build.tokenizer_path {
        if cfg.build.probe_max_tokens > 0 {
            after_save.push(CheckpointAction::Probe {
                max_tokens: cfg.build.probe_max_tokens,
                temperature: cfg.build.probe_temperature,
                tokenizer: tok_path.clone(),
            });
        }
    }
    CheckpointConfig {
        every: cfg.build.save_every,
        before_save,
        after_save,
    }
});
```

Old configs work unchanged. New configs use the `checkpoint` object and ignore the legacy fields.

Note: the legacy fallback respects `probe_max_tokens == 0` as "skip probes" — this
is the immediate fix that unblocks existing configs.

### Phase boundary behavior

The phase-boundary checkpoint (end of each phase) uses the same action pipeline. Currently it duplicates the save + probe logic — this collapses to a single call.

## Immediate Fix (ship with this PR)

Add an early exit to `run_inline_probes`:

```rust
pub fn run_inline_probes(..., max_tokens: usize, ...) -> serde_json::Value {
    if max_tokens == 0 {
        return json!({"skipped": true});
    }
    // ... existing code
}
```

This unblocks the d=2048 run immediately while the full pipeline is implemented.

## Files to Modify

| File | Change |
|------|--------|
| `cli/src/eval.rs` | Add `max_tokens == 0` early exit (immediate fix) |
| `cli/src/config.rs` | Add `CheckpointConfig`, `CheckpointAction` structs with serde |
| `cli/src/run.rs` | Replace hardcoded checkpoint block with action loop; add legacy fallback |

## Migration Path

1. **Phase A** (this PR): Add `max_tokens == 0` guard + `CheckpointConfig` struct + action dispatch. Legacy fields remain, `checkpoint` object is optional.
2. **Phase B** (future): Deprecation warnings for legacy fields. New configs should use `checkpoint` object exclusively.
3. **Phase C** (future): Remove legacy fields. All configs use `checkpoint` object.

## Validation

1. `cargo test --features cuda` — existing tests pass
2. Old configs without `checkpoint` object produce identical behavior
3. New config with empty `before_save`/`after_save` runs only the save
4. New config with probe in `after_save` runs probes after save
5. New config with profile in `before_save` runs profile before save
6. `probe_max_tokens: 0` in legacy config skips probes (immediate fix)
7. Action ordering respected within each phase (`before_save`, `after_save`)

## Success Criteria

1. Probes can be disabled from config without removing `tokenizer_path`
2. Checkpoint actions composable and orderable from JSON
3. Existing configs work without modification
4. d=2048 Gear 1 run launches without probe overhead

## Non-Goals

- No trait-object plugin system (enum is sufficient for known actions)
- No async/parallel action execution (sequential is fine)
- No per-action `every` cadence (all actions fire at the same `checkpoint.every` interval)
- `profile_every` stays separate for now (it's a per-step diagnostic, not a checkpoint action)
