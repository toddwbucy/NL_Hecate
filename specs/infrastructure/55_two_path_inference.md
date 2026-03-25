# Two-Path NLM Inference

```
CONTRACT
  Purpose:    Enforce the NLM's two legitimate execution paths and remove
              eval-mode artifacts. An NLM has no train/eval distinction (CS-10).
              The model either LEARNS (step_adamw) or SPEAKS (prefill + decode_token).
              forward() and step_generate() are eval-mode ghosts that violate this.
  Expects:    GpuModel, GpuStackedModel with step_adamw(), prefill(), decode_token().
  Guarantees: After this cleanup, exactly TWO paths through the model exist:
              1. step_adamw()  — the model learns (forward + backward + optimizer)
              2. prefill() + decode_token() — the model speaks (autoregressive output)
              All callers of forward() and step_generate() are migrated or removed.
  Cost:       Zero runtime cost. This is a deletion + migration, not a feature.
  Trade-off:  Evaluation becomes more expensive (step_adamw computes gradients even
              when we only want loss). This is correct — the model learns from val
              data too. If pure loss measurement is needed without weight updates,
              a future freeze_weights flag on step_adamw is the right solution,
              NOT a separate forward-only path.
  Position:   specs/infrastructure/55_two_path_inference.md
  Source:     CS-10 (no train/eval), CS-18 (forward pass is the only API),
              Titans (2501.00663) §3 — test-time learning IS the forward pass
```

## The Two Paths

### Path 1: Learn — `step_adamw()`

The model processes a sequence and learns from it. This is the ONLY way the model
processes training data, validation data, user input in chat, or any other tokens.

```
step_adamw(input_ids, target_ids, pulse, lr, ...) -> (loss, grad_norm)

  1. gpu_cms_forward()   — forward pass, M updates, loss computation
  2. gpu_cms_backward()  — backward pass, gradient computation
  3. gpu_adamw_update()  — weight updates (Pulse-gated per CMS level)
```

There is no "eval mode" variant. When evaluating on validation data, step_adamw()
runs at the same learning rate. The model learns from everything it sees — this is
what makes it a Nested Learning Machine.

### Path 2: Speak — `prefill()` + `decode_token()`

The model generates output tokens autoregressively. This is mechanical decoding
of the model's current state into text. It does NOT learn — it reads from the
KV cache built during prefill.

```
prefill(input_ids, pulse) -> last_logits     # build KV cache from context
decode_token(token_id, pulse) -> logits      # extend one token, return logits
```

This path exists because autoregressive generation requires single-token steps
with cached attention state. It is not an "inference mode" — it is the output
interface. The model's M matrices are read-only during decode (they were updated
during the learn path that preceded generation).

## What Gets Removed

### `forward()` — eval mode violation (CS-10)

**Location**: `GpuModel.forward()` (lib.rs:2276), `GpuStackedModel.forward()` (lib.rs:3075)

**What it does**: `gpu_cms_forward()` only — no backward, no optimizer. Returns
(loss, logits_flat). This is textbook eval mode: same forward pass, gradients
disabled.

**Why it exists**: Legacy assumption that evaluation needs a cheaper forward-only
path. This assumption is wrong for NLMs — the model should learn from val data.

**Callers to migrate** (14 total):

| File | Line | Current Call | Migration |
|------|------|-------------|-----------|
| `engine/evaluation.py` | 223 | `gpu_model.forward(...)` | → `step_adamw()` with eval LR |
| `engine/evaluation.py` | 248 | `gpu_model.forward(...)` | → `step_adamw()` with eval LR |
| `engine/evaluation.py` | 299 | `gpu_model.forward(...)` | → `step_adamw()` with eval LR |
| `engine/loop.py` | 1528-1529 | checkpoint roundtrip verify | → `step_adamw()` (both models) |
| `tools/niah_verify.py` | 212 | `gpu_model.forward(...)` | → `step_adamw()` or prefill-based |
| `tests/test_bindings.py` | 57 | `nl_hecate.forward(...)` | Remove (CPU-path, non-GPU) |
| `tests/test_bindings.py` | 69-70 | `nl_hecate.forward(...)` | Remove (CPU-path) |
| `tests/test_bindings.py` | 80 | `nl_hecate.forward(...)` | Remove (CPU-path) |
| `tests/test_bindings.py` | 91 | `nl_hecate.forward(...)` | Remove (CPU-path) |
| `tests/test_baseline.py` | 57 | `nl_hecate.forward(...)` | → `step_adamw()` |
| `tests/test_training.py` | 22, 29 | `nl_hecate.forward(...)` | → `step_adamw()` (read loss only) |
| `training/compliance_predictor.py` | 70 | `self.forward(...)` | Internal PyTorch (not NL — exempt) |

**Note**: `training/compliance_predictor.py` calls `self.forward()` on a standard
PyTorch MLP for HADES compliance prediction. This is NOT an NL model and is exempt.

### `step_generate()` — confused hybrid

**Location**: `GpuModel.step_generate()` (lib.rs:2386)

**What it does**: `gpu_cms_forward()` + `gpu_cms_backward()` + `gpu_adamw_update()`
+ extracts last-position logits from the forward cache before backward runs.
Returns (loss, grad_norm, last_logits).

**Why it exists**: Attempted to combine "learn" and "speak" into one call for
`generate_learning()`. This is architecturally confused — learning and speaking
are separate operations with different purposes.

**Callers to migrate** (2 total):

| File | Line | Current Call | Migration |
|------|------|-------------|-----------|
| `engine/generation.py` | 185 | `gpu_model.step_generate(...)` | → `step_adamw()` for learning phase |
| `engine/generation.py` | 227 | `gpu_model.step_generate(...)` | → `step_adamw()` then prefill+decode for speaking |

**Migration for `generate_learning()`**: Split into two phases:
1. `step_adamw()` on the input chunk → model learns
2. `prefill()` + `decode_token()` loop → model generates next tokens

This is exactly what `chat.py` already does correctly (learn_from_tokens +
generate_response as separate operations).

## The CPU-Path `forward()` (lib.rs:190)

There is also a standalone `forward()` function (not a method) at lib.rs:190 for
the CPU SWA path. This predates the GPU stack entirely. It should be removed as
part of this cleanup — the CPU path is unused in production and all active builds
use the GPU stack.

## Evaluation Migration Detail

Current `evaluate()` and `evaluate_numpy()` call `forward()` to get loss without
learning. The correct NLM behavior:

```python
# BEFORE (eval mode — CS-10 violation):
loss, _ = gpu_model.forward(input_ids, target_ids, pulse)

# AFTER (the model learns from val data too):
loss, _gnorm = gpu_model.step_adamw(
    input_ids, target_ids, pulse, lr=eval_lr,
    beta1=0.9, beta2=0.999, eps=1e-8,
    weight_decay=0.1, max_grad_norm=1.0,
    freeze_embed=False,
)
```

**eval_lr**: Use the same LR as training. The model learns from validation data.
If the caller wants to measure loss without permanent weight changes, it should
save/restore the model state (same as checkpoint roundtrip verification does now).

**Context management**: `evaluate()` already creates a fresh Conductor for eval.
After migration, the model's M and weights will be modified by val data. Callers
that need pristine state for subsequent training must save/restore context and
weights around the eval call. This is already the pattern in loop.py's checkpoint
roundtrip verification.

## Checkpoint Roundtrip Verification

The checkpoint verification in `loop.py:1528-1529` uses `forward()` to compare
loss between the training model and a freshly-loaded verification model. After
migration:

```python
# Both models run step_adamw — they should produce identical loss AND gradient
train_loss, train_gnorm = gpu_model.step_adamw(rt_input, rt_target, pulse, lr, ...)
verify_loss, verify_gnorm = v_model.step_adamw(rt_input, rt_target, pulse, lr, ...)
delta_loss = abs(verify_loss - train_loss)
delta_gnorm = abs(verify_gnorm - train_gnorm)
```

This is actually a STRONGER verification — it confirms the full forward+backward
pipeline roundtrips correctly, not just the forward pass.

## NIAH Verification Tool

`tools/niah_verify.py` uses `forward()` to get logits at the answer position.
Two migration options:

1. **step_adamw() + ignore gradients**: Same as eval migration. The model learns
   from the NIAH context, which is fine — that's what it would do in production.
2. **prefill() + decode**: If we only need the logit at the answer position, this
   is more natural. Prefill the context, then check what the model would predict.

Option 2 is cleaner for NIAH because it mirrors the actual use case (predicting
a specific token after reading context).

## Implementation Sequence

1. **Migrate `evaluation.py`** — replace 3x `forward()` with `step_adamw()`
2. **Migrate `loop.py` checkpoint verify** — replace 2x `forward()` with `step_adamw()`
3. **Migrate `generation.py`** — replace `generate_learning()` to use `step_adamw()` + `prefill()`/`decode_token()`
4. **Migrate `niah_verify.py`** — replace `forward()` with prefill-based approach
5. **Migrate tests** — update `test_bindings.py`, `test_baseline.py`, `test_training.py`
6. **Remove `forward()` from `GpuModel`** (lib.rs:2276)
7. **Remove `forward()` from `GpuStackedModel`** (lib.rs:3075)
8. **Remove `step_generate()` from `GpuModel`** (lib.rs:2386)
9. **Remove CPU-path `forward()`** (lib.rs:190)
10. **Verify**: `grep -r '\.forward(' python/` returns only `compliance_predictor.py`

## Constraint Compliance

- **CS-10**: Enforced — no train/eval mode distinction. step_adamw() IS the eval path.
- **CS-18**: Enforced — forward pass is the only external API (step_adamw wraps it).
- **CS-11**: Enforced — no gradient tape on/off switch. step_adamw always computes gradients.
- **CS-22**: Preserved — step_adamw is the single entry point for processing sequences.

## Future: freeze_weights Flag

If profiling shows that val evaluation is too expensive with full backward+optimizer,
the correct solution is a `freeze_weights=True` parameter on `step_adamw()` that
skips the optimizer step but still runs forward+backward. This preserves the
single-path architecture while avoiding weight mutation during eval. This is NOT
part of this spec — it is noted here as the sanctioned future direction. A separate
`forward()` method is NEVER the answer.
