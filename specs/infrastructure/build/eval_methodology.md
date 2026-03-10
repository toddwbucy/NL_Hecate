# Eval Methodology for Non-Stationary Models

```text
CONTRACT
  Purpose:    Define how to measure a model that learns during inference.
              Conventional eval (freeze weights, run forward, measure loss)
              tests transformer-like behavior. NL models learn during every
              forward pass — eval must measure THAT, not a frozen snapshot.
  Expects:    GpuModel with generate_learning(), to_host_params(),
              to_host_context(), upload_params(), upload_context().
  Guarantees: Eval probes measure the NL mechanism (inner+outer loop learning),
              not just static weight quality.
              Training state is fully restored after eval (no contamination).
  Cost:       ~3x slower than frozen eval (forward+backward+optimizer per token).
              Full state snapshot/restore adds ~200ms per eval block.
  Trade-off:  Heavier eval vs actually measuring the right thing.
  Position:   specs/infrastructure/build/eval_methodology.md
  Source:     CS-10 (no mode distinction), FineWeb-Edu k=1/k=4 runs (Feb 2026)
```

## Problem Statement

The current eval pipeline (`eval_coherence_samples`, `generate_samples`) runs
generation with learning OFF:

```python
gpu_model.reset_context()           # fresh M (no accumulated memory)
generate(gpu_model, prompt, ...)    # forward-only, no backward, no optimizer
```

This measures whether the outer-loop parameters (W_K, W_V, gates) are good
static weights — which is what a transformer test measures. It tells us nothing
about whether the NL mechanism works. Specifically:

- **Inner-loop learning**: M updates during forward are active, but starting
  from zero (reset_context) means we only test cold-start M accumulation
  over 30 tokens. Not representative.
- **Outer-loop learning**: Completely absent. The model cannot adapt its
  projections to the prompt. This is the core NL capability we need to measure.

Evidence: At 25K steps with eval loss 6.4 (PPL ~600), both k=1 and k=4
produce degenerate output (`<|im_start|>` repetition) during frozen eval.
The model IS learning (loss drops every 5K steps) but the eval doesn't
show it because eval tests a mode the model was never designed for.

## Design: Three Eval Probes

### Probe 1: Within-Generation Learning Curve

**Question**: Does the model get better at predicting tokens as it generates?

**Method**:
```text
1. Save full state (params + context + adamw_state)
2. reset_context()
3. Run generate_learning(prompt, max_tokens=60)
   → returns (tokens, per_token_losses, per_token_grad_norms)
4. Restore full state
5. Report: loss trajectory over 60 tokens
```

**Expected signal**: Loss should decrease over the generation span. Token 50's
loss should be lower than token 5's loss because the model has been learning
from its own output for 45 steps.

**Diagnostic value**: If loss is FLAT, the outer-loop learning rate is too low
or the backward pass isn't reaching the right parameters. If loss INCREASES,
the model is learning garbage from its own degenerate output (positive feedback
loop — important failure mode to detect).

**Output format** (JSONL):
```json
{
  "event": "learning_probe",
  "probe": "within_generation",
  "step": 25000,
  "prompt": "Once upon a time",
  "token_losses": [7.2, 7.1, 6.9, ...],
  "token_grad_norms": [0.8, 0.7, ...],
  "generated_text": "Once upon a time...",
  "loss_slope": -0.012,
  "loss_first10_avg": 7.15,
  "loss_last10_avg": 6.43
}
```

### Probe 2: Cross-Exposure Adaptation

**Question**: Does the model produce better output the second time it sees
the same prompt, because it learned from the first exposure?

**Method**:
```text
1. Save full state
2. reset_context()
3. Run generate_learning(prompt, max_tokens=30) → output_1, losses_1
4. DO NOT RESTORE — model has now learned from its generation
5. reset_context()  (clear M, keep updated params)
6. Run generate_learning(prompt, max_tokens=30) → output_2, losses_2
7. Restore full state
8. Report: compare losses_1 vs losses_2
```

**Expected signal**: losses_2 should be lower than losses_1 on average.
The model's outer-loop parameters were updated during run 1, so run 2
benefits from that learning. This is something NO transformer can do —
it's the definitive NL test.

**Key detail**: We reset_context() between runs but do NOT restore params.
This isolates outer-loop learning (params changed) from inner-loop learning
(M accumulation). The model must demonstrate that its PARAMETER updates
from run 1 transfer to run 2.

**Variant — accumulated context**: Same as above but WITHOUT reset_context()
between runs. This tests whether inner-loop M accumulation across exposures
also contributes. Compare:
- Fresh M each time (outer-loop only)
- Accumulated M (inner + outer loop)

**Output format** (JSONL):
```json
{
  "event": "learning_probe",
  "probe": "cross_exposure",
  "step": 25000,
  "prompt": "Once upon a time",
  "run1_avg_loss": 7.15,
  "run2_avg_loss": 6.89,
  "improvement": 0.26,
  "improvement_pct": 3.6,
  "run1_text": "...",
  "run2_text": "..."
}
```

### Probe 3: Accumulated Context vs Cold Start

**Question**: Does memory accumulated during training help generation?

**Method**:
```text
1. Save full state (which includes accumulated training context)
2. Cold start: reset_context() → generate_learning(prompt) → losses_cold
3. Restore full state
4. Warm start: DON'T reset_context() → generate_learning(prompt) → losses_warm
5. Restore full state
6. Report: compare losses_cold vs losses_warm
```

**Expected signal**: Warm start (accumulated M from training) should produce
lower initial loss than cold start (M=0). If there's no difference, the
accumulated M from training chunks isn't contributing to generation quality.

**Output format** (JSONL):
```json
{
  "event": "learning_probe",
  "probe": "context_value",
  "step": 25000,
  "prompt": "Once upon a time",
  "cold_avg_loss": 7.15,
  "warm_avg_loss": 6.72,
  "context_benefit": 0.43,
  "cold_text": "...",
  "warm_text": "..."
}
```

## Orchestration: Full State Snapshot

`generate_learning()` modifies three things:

| State | What Changes | Save Method | Restore Method |
|---|---|---|---|
| Context (M matrices) | Inner-loop memory updates | `to_host_context()` | `upload_context()` |
| Params (W_K, W_V, gates, embeddings) | Outer-loop gradient updates | `to_host_params()` | `upload_params()` |
| AdamW moments (m, v, step) | Optimizer running averages | **NOT YET EXPOSED** | **NOT YET EXPOSED** |

**Gap**: AdamW state save/restore is not exposed via PyO3. Two options:

**Option A (recommended)**: Add `to_host_adamw()` / `upload_adamw()` to GpuModel.
This is the correct fix — full state snapshot with zero information loss.
The AdamW state is just two Vec<f32> (m and v buffers) plus a step counter.

**Option B (implemented)**: Call `gpu_model.reset_optimizer()` after probes to
clear the corrupted AdamW moments entirely. This zeroes both m/v buffers and
the step counter, so the next training step lazy-reinitializes them. At eval
frequency (every 5K steps), the moments reconverge within ~100 steps.
Option A remains recommended for future implementation.

### Snapshot Pseudocode

```python
def full_snapshot(gpu_model):
    """Save complete model state for later restoration."""
    return {
        "params": gpu_model.to_host_params(),
        "context": gpu_model.to_host_context(),
        # "adamw": gpu_model.to_host_adamw(),  # Option A (needs implementation)
    }

def full_restore(gpu_model, snapshot):
    """Restore complete model state from snapshot."""
    gpu_model.upload_params(snapshot["params"])
    gpu_model.upload_context(snapshot["context"])
    # gpu_model.upload_adamw(snapshot["adamw"])  # Option A (future)
    # For now: call gpu_model.reset_optimizer() after probes (Option B)
```

## Integration into Build Loop

Replace the current eval block's coherence samples with learning probes.
The eval block at each `eval_every` step becomes:

```python
# ── Eval loss (unchanged) ────────────────────────────────────
eval_ctx = gpu_model.to_host_context()
gpu_model.reset_context()
eval_loss, eval_ppl = evaluate(gpu_model, bcfg, val_stream, ...)
gpu_model.upload_context(eval_ctx)

# ── Gate diagnostics (unchanged) ──────────────────────────────
print_level_metrics(gpu_model, bcfg.k)

# ── Learning probes (NEW — replaces frozen coherence samples) ─
snapshot = full_snapshot(gpu_model)
try:
    # Probe 1: within-generation learning curve
    gpu_model.reset_context()
    for prompt in EVAL_PROMPTS:
        tokens, losses, gnorms = generate_learning(
            gpu_model, cfg, tokenizer.encode(prompt),
            max_tokens=60, temperature=0.7, lr=bcfg.lr, ...
        )
        log_within_generation(jsonl, step, prompt, tokens, losses, gnorms)

    # Probe 2: cross-exposure adaptation (one prompt, two runs)
    full_restore(gpu_model, snapshot)  # clean start for Probe 2
    gpu_model.reset_optimizer()
    result = probe_cross_exposure(gpu_model, cfg, prompt_ids, ...)
finally:
    full_restore(gpu_model, snapshot)
    gpu_model.reset_optimizer()  # probes corrupt AdamW moments
```

**Probe frequency**: Probes 1 and 2 at every `eval_every`. Probe 3 at every
`save_every` (checkpoint steps only — less frequent, more expensive).

## Eval Prompts

The current EVAL_PROMPTS ("Once upon a time", "The meaning of life is",
"In the beginning") are from the old ShareGPT config and may not be
representative of FineWeb-Edu content. Consider prompts that:

1. **Match the training domain**: FineWeb-Edu is educational text, so
   prompts like "The process of photosynthesis" or "In mathematics, a
   prime number" would be more natural starting points.

2. **Vary in specificity**: Generic ("The") vs specific ("The capital of
   France is") to test whether the model's learning is content-dependent.

3. **Repeat across runs**: Same prompts for k=1 and k=4 for comparability.

Proposed EVAL_PROMPTS for FineWeb-Edu:
```python
EVAL_PROMPTS = [
    "The process of",
    "In mathematics,",
    "Scientists discovered that",
    "The history of",
]
```

## Metrics Summary

| Metric | Probe | What It Tells Us |
|---|---|---|
| loss_slope (Probe 1) | Within-gen | Is the model learning during generation? |
| loss_last10 - loss_first10 (Probe 1) | Within-gen | Magnitude of within-generation improvement |
| run2_loss - run1_loss (Probe 2) | Cross-exposure | Do parameter updates transfer across generations? |
| warm_loss - cold_loss (Probe 3) | Context value | Does accumulated M help generation? |
| All above, k=1 vs k=4 | Comparison | Does frequency hierarchy improve any of these? |

## Falsifiable Predictions

1. **Probe 1 loss_slope < 0** after 25K steps: The model learns during generation.
   If loss_slope >= 0, the outer-loop learning during generation is ineffective.

2. **Probe 2 improvement > 0** after 25K steps: The model adapts to repeated
   prompts. If improvement <= 0, outer-loop parameter updates don't transfer.

3. **Probe 3 context_benefit > 0** after 25K steps: Accumulated M helps.
   If context_benefit <= 0, the inner-loop memory isn't contributing.

4. **k=4 shows larger Probe 2 improvement than k=1**: Frequency hierarchy
   enables faster cross-exposure adaptation (L0 adapts quickly, L3 retains).
   If k=1 adapts equally well, CMS hierarchy doesn't help adaptation.

## Implementation Sequence

1. **Implement full_snapshot / full_restore** (Python only, Option B — no Rust changes)
2. **Add Probe 1** to eval block (replace frozen coherence samples)
3. **Add Probe 2** to eval block (after Probe 1)
4. **Add Probe 3** to checkpoint block (less frequent)
5. **Update EVAL_PROMPTS** for FineWeb-Edu domain
6. **Later**: Implement Option A (AdamW state save/restore in Rust+PyO3)

## Interaction with CS-49 (Context Corruption)

All probes use full_snapshot/full_restore, which is a superset of the CS-49
context-only save/restore. The existing CS-49 fix (coherence samples wrapped
in context save/restore) will be replaced by full_snapshot/full_restore that
also covers params. This is strictly more protective.

## Interaction with CS-10 (No Mode Distinction)

This spec makes the eval pipeline CS-10 compliant. The model is always in
one of two states:
- **Learning**: generate_learning() — outer loop active, params update
- **Not learning**: eval loss computation — forward only, no backward

There is no third "eval generation" mode with frozen weights. The frozen
coherence samples are removed.
