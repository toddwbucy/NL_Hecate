# Checkpoint Format: safetensors Binary

```
CONTRACT
  Purpose:   Replace serde_json text serialization with safetensors binary format
             for model checkpoints. Eliminates 3-7x size bloat and 30-60s JSON
             parse overhead. Enables memory-mapped loading (~1s vs ~60s).
  Expects:   MAGParams, MAGConfig accessible for serialization. safetensors crate
             available. Existing .json checkpoints on disk for backward compat.
  Guarantees: New checkpoints written as .safetensors (binary, compact, mmappable).
              load_checkpoint detects format by extension and routes accordingly.
              All existing .json checkpoints remain loadable (no migration required).
              Round-trip exact: load(save(params)) == params (bitwise f32).
  Cost:      One new crate dependency (safetensors 0.3). Recompile required.
             Existing .json checkpoints become legacy (still readable, not written).
  Trade-off: safetensors is slightly less human-debuggable than JSON. Acceptable:
             the config metadata is still stored as JSON in the safetensors header,
             and weights can be inspected with Python `safetensors` library.
  Position:  core/src/checkpoint.rs (new file) + core/src/model.rs (route calls)
  Source:    HuggingFace safetensors spec: https://github.com/huggingface/safetensors
```

## Motivation

Current JSON checkpoint for a 433M param SwiGLU model:
- File size: ~5 GB (f32 serialized as text, ~7 chars/float × 433M = 3+ GB weights alone)
- Load time: 30-60 seconds (serde_json parsing all float strings)
- Precision: potential rounding on JSON float round-trip

safetensors equivalent:
- File size: ~1.7 GB (4 bytes/f32, raw binary)
- Load time: <1 second (memory-mapped, OS handles paging)
- Precision: exact bitwise f32 round-trip (raw bytes, no text conversion)

## File Format

```
<filename>.safetensors
  ├── 8-byte LE u64: metadata JSON length (N)
  ├── N bytes: metadata JSON
  │     {
  │       "__metadata__": {
  │         "version":    "2",
  │         "format":     "nl_hecate_v2",
  │         "created_at": "<ISO8601>",
  │         "config":     "<MAGConfig as JSON string>",
  │         "build_state": "<BuildResumeState as JSON string | null>"
  │       },
  │       "<tensor_name>": { "dtype": "F32", "shape": [...], "data_offsets": [s, e] },
  │       ...
  │     }
  └── Raw tensor bytes (concatenated, page-aligned)
```

Config is stored as a JSON string under `__metadata__.config`. This keeps it
human-readable while the weights stay binary. The safetensors crate handles
the header automatically when tensors are registered with named keys.

## Tensor Naming Scheme

Each parameter array gets a flat namespaced key:

```
embed.weight                          [vocab × d_model]
lm_head.weight                        [vocab × d_model]
level.{i}.w_k                         [d × d]
level.{i}.w_v                         [d × d]
level.{i}.w_q                         [d × d]
level.{i}.gate.alpha                  [2d]        (bias concat)
level.{i}.gate.theta                  [2d]
level.{i}.gate.eta                    [2d]
level.{i}.mlp.gate_proj               [inter × d]   (SwiGLU only)
level.{i}.mlp.up_proj                 [inter × d]   (SwiGLU only)
level.{i}.mlp.down_proj               [d × inter]   (SwiGLU only)
level.{i}.m_state                     [d × d]       (Titans/Delta only)
level.{i}.s_state                     [d × d]       (momentum, if present)
```

## Implementation Plan

### 1. `core/Cargo.toml`
```toml
safetensors = "0.3"
```

### 2. `core/src/checkpoint.rs` (new file)

Exposes two public functions:

```rust
pub fn save_safetensors(
    path: &Path,
    params: &MAGParams,
    config: &MAGConfig,
    build_state: Option<&BuildResumeState>,
) -> std::io::Result<()>

pub fn load_safetensors(
    path: &Path,
) -> std::io::Result<(MAGParams, MAGConfig, Option<BuildResumeState>)>
```

Implementation:
- Collect all f32 vecs from MAGParams into a `HashMap<String, (Vec<f32>, Vec<usize>)>`
- Serialize config + build_state as JSON strings into `__metadata__`
- Call `safetensors::serialize_to_file(tensors, &metadata, path)`
- Load: `safetensors::SafeTensors::deserialize(&mmap)`, extract tensors by name,
  reconstruct MAGParams, parse config from `__metadata__["config"]`

### 3. `core/src/model.rs`

Update `save_checkpoint` and `save_build_checkpoint` to call `save_safetensors`
and write a `.safetensors` extension path.

Update `load_checkpoint` to detect format:
```rust
pub fn load_checkpoint(path: &Path) -> std::io::Result<(MAGParams, MAGConfig, Option<BuildResumeState>)> {
    if path.extension().and_then(|e| e.to_str()) == Some("json") {
        load_checkpoint_json(path)   // existing legacy path
    } else {
        load_safetensors(path)       // new binary path
    }
}
```

### 4. PyO3 bindings (`python/nl_hecate/`)

`save_checkpoint(path: &str, ...)` — path from Python will already have `.safetensors`
extension when called from loop.py. No binding changes needed IF the extension
detection works transparently.

Update `engine/loop.py` save paths:
```python
# Before: "checkpoints/model_step{step}.json"
# After:  "checkpoints/model_step{step}.safetensors"
```

The cursor sidecar stays `.json` (it's small metadata, JSON is fine there).

## Backward Compatibility

- `.json` files: loaded via existing `serde_json` path, unchanged
- `.safetensors` files: loaded via new binary path
- No automatic migration — existing checkpoints remain readable indefinitely
- Training logs, sidecar files, config files: unchanged (still JSON, small)

## Verification

```bash
# After implementation:
cargo test --features cuda          # all 778+ tests must pass
cargo test checkpoint               # round-trip tests specifically

# Python round-trip:
python3 -c "
import nl_hecate
params, cfg = nl_hecate.load_checkpoint('checkpoints/model_step50k.safetensors')
nl_hecate.save_checkpoint('checkpoints/test_rt.safetensors', params, cfg)
p2, c2 = nl_hecate.load_checkpoint('checkpoints/test_rt.safetensors')
print('Round-trip OK')
"

# Size comparison:
ls -lh checkpoints/model.json checkpoints/model_step50k.safetensors
# Expect: ~5GB → ~1.7GB

# Load time comparison (informational):
time python3 -c "import nl_hecate; nl_hecate.load_checkpoint('checkpoints/model.json')"
time python3 -c "import nl_hecate; nl_hecate.load_checkpoint('checkpoints/model_step50k.safetensors')"
```

## Files to Modify

| File | Change |
|------|--------|
| `core/Cargo.toml` | Add `safetensors = "0.3"` |
| `core/src/checkpoint.rs` | New file: `save_safetensors`, `load_safetensors` |
| `core/src/model.rs` | Route `save_checkpoint`/`load_checkpoint` by extension |
| `core/src/lib.rs` | `mod checkpoint;` |
| `python/engine/loop.py` | Change save path extensions `.json` → `.safetensors` |
| `python/scripts/convert_checkpoint.py` | New: migrate existing .json → .safetensors |
