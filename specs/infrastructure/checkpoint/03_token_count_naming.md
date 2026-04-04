# 03 — Token-Count Checkpoint Naming

## CONTRACT

| Field     | Value |
|-----------|-------|
| Purpose   | Replace raw-integer checkpoint filenames with human-readable token-count suffixes so that file listings convey scale at a glance |
| Expects   | Spec 78 `CheckpointNaming::Tokens` variant currently emits `model_{N}tok.safetensors` with raw integer N; state file (spec 02) stores the exact token count |
| Guarantees | Filenames use SI-style suffixes (`K`, `M`, `B`); the exact integer is authoritative in the state file — filenames are labels, not data; `CheckpointNaming::Steps` is unaffected; backward-compatible load (any `.safetensors` loads regardless of name) |
| Cost      | One formatting function; updated tests for checkpoint_filename |
| Trade-off | Lossy filename (150M could be 150,000,000 or 150,999,999) vs readable. Exact count in state file resolves ambiguity |
| Position  | Part of the State File Lifecycle (SFL) epic. Depends on spec 78 (checkpoint trigger policy) and spec 02 (state file schema) |
| Source    | IaC model lifecycle vision — filenames should be human-readable at `ls` time |

## Formatting Rules

Given `total_tokens: u64`, produce a human-friendly string:

| Range | Format | Examples |
|-------|--------|----------|
| < 1,000 | `{N}` | `512` |
| 1,000 .. 999,999 | `{N/1000}K` | `5K`, `750K` |
| 1,000,000 .. 999,999,999 | `{N/1e6}M` | `1M`, `150M`, `999M` |
| >= 1,000,000,000 | `{N/1e9:.1}B` | `1.0B`, `1.2B`, `12.5B` |

Rules:
- K and M tiers use integer division (floor). No decimal places.
- B tier uses one decimal place (always, even `.0`).
- Sub-1000 range is raw integer (early training checkpoints).
- The suffix `_tok` is always appended after the SI label.

### Filename pattern

```
{stem}_{formatted_tokens}_tok.safetensors
```

Where `stem` is derived from the base save_path (e.g. `model`, `legal_v1`).

### Examples

| total_tokens | Filename |
|-------------|----------|
| 512 | `model_512_tok.safetensors` |
| 5,120 | `model_5K_tok.safetensors` |
| 512,000 | `model_512K_tok.safetensors` |
| 5,120,000 | `model_5M_tok.safetensors` |
| 150,000,000 | `model_150M_tok.safetensors` |
| 1,200,000,000 | `model_1.2B_tok.safetensors` |

## Backward Compatibility

- `CheckpointNaming::Steps` continues to produce `model_step{N}.safetensors` unchanged.
- `load_checkpoint` / `load_stacked_safetensors` loads by path — it does not parse the filename. Any `.safetensors` file loads regardless of naming convention.
- Old `model_{N}tok.safetensors` files remain loadable (they're just safetensors files).

## State File Interaction

The state file's `checkpoints[].path` records the actual filename produced. The state file's `checkpoints[].tokens` stores the exact integer. The filename is a human convenience — the state file is the source of truth for token counts.

## Implementation

1. Add `format_tokens(total_tokens: u64) -> String` to `checkpoint_policy.rs`
2. Update `checkpoint_filename()` to call `format_tokens()` when `CheckpointNaming::Tokens`
3. Update tests for new formatting
