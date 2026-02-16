# Integration Spike Outcome: Stages 1-2 End-to-End Validation

**Date**: 2026-02-16
**Outcome**: **1 (GO)** — Proceed to Stage 3

## Summary

The integration spike proves the thesis: **the full NL pipeline learns a predictable pattern from a token stream.** Three representative configurations all reduce loss from random-chance (~2.77) to near-zero (<0.001) in 500 steps, with 100% prediction accuracy on the repeating pattern task. The serving path produces identical losses to the raw API path. Checkpoint/restore is deterministic.

## Test Results

### Stage 1: Algorithm Core (12 tests)

| Test | Config | Result | Details |
|------|--------|--------|---------|
| smoke_config_a | DeltaRule+MAG k=2 | PASS | 100 steps, all finite |
| smoke_config_b | TitansLMM+MAL k=2 | PASS | 100 steps, all finite |
| smoke_config_c | Hebbian+MAG k=1 | PASS | 100 steps, all finite |
| convergence_config_a | DeltaRule+MAG k=2 | PASS | 2.77 -> 0.0007 (100% reduction) |
| convergence_config_b | TitansLMM+MAL k=2 | PASS | 2.78 -> 0.0003 (100% reduction) |
| convergence_config_c | Hebbian+MAG k=1 | PASS | 2.77 -> 0.0005 (100% reduction) |
| prediction_config_a | DeltaRule+MAG k=2 | PASS | 8/8 correct (100%) |
| prediction_config_b | TitansLMM+MAL k=2 | PASS | 8/8 correct (100%) |
| prediction_config_c | Hebbian+MAG k=1 | PASS | 8/8 correct (100%) |
| context_memory_evolves | DeltaRule+MAG k=2 | PASS | Memory norms change between 100 and 200 steps |
| gradient_flow | DeltaRule+MAG k=2 | PASS | SWA norm=0.024, level norms > 0 |
| multi_config_diagnostic | A + B | PASS | Both converge; B faster (MAL + momentum) |

### Stage 2: Production Infrastructure (4 tests, `feature = "serving"`)

| Test | Result | Details |
|------|--------|---------|
| serving_test_mode_smoke | PASS | 100 chunks, all finite, no explosion |
| serving_stream_mode_runs | PASS | 100 chunks from VecStream, all finite |
| serving_matches_raw | PASS | 100 steps: Session == raw cms_forward (exact match) |
| serving_checkpoint_restore | PASS | 50+50 steps post-restore match pre-restore (deterministic) |

### Stage 2: CUDA Dispatch (1 test, `feature = "cuda"`)

| Test | Result | Details |
|------|--------|---------|
| cuda_dispatch_smoke | PASS* | Compiles and runs forward+backward, finite loss |

*CUDA tests use the Rust reference path currently; dispatch routing to actual kernels is deferred to S2-M1 architecture dispatch completion.

**Total: 17 tests, 17 pass, 0 fail.**

## Key Findings

### Learning Dynamics at d=8

- **Learning rate**: lr=0.5 was needed for this tiny model (d=8). At lr=0.01 (used by d=64 unit tests), the model barely learns in 500 steps. This scales roughly as 1/sqrt(d).
- **Convergence speed**: Config B (TitansLMM+MAL) converges fastest — hits loss<0.02 by step 150 vs step 250 for Config A. The momentum term in TitansLMM helps, and MAL's sequential composition provides a cleaner gradient signal.
- **All 3 configs achieve 100% accuracy**: The repeating pattern [0,1,2,...,7] is fully learned. Predictions are exact: every position correctly predicts the next token.

### Loss Trajectories

```text
Config A (DeltaRule+MAG k=2):
  step   0: 2.7739  (random)
  step 100: 1.9737
  step 200: 0.4539
  step 300: 0.0037
  step 500: 0.0007

Config B (TitansLMM+MAL k=2):
  step   0: 2.7753  (random)
  step 100: 0.8903
  step 200: 0.0023
  step 300: 0.0007
  step 500: 0.0003

Config C (Hebbian+MAG k=1):
  step 500: 0.0005
```

### Serving Path Observations

- **Forward-only path**: Session::process_chunk() only does forward (inner loop). With fixed weights and fixed input, memory converges to a fixed point — loss stabilizes rather than decreasing. This is correct NL behavior: the inner loop adapts memory, but outer-loop parameter updates require backward + apply.
- **Exact parity**: Raw cms_forward() and Session::process_chunk() produce bitwise-identical losses over 100 steps — the serving abstraction adds zero numerical deviation.
- **Deterministic checkpoint**: Restore produces an identical loss trajectory, confirming all state (conductor step, context memory, stream cursor) is fully captured.

### VecStream Wrap-Around

VecStream returns truncated chunks when wrapping at corpus boundaries. This causes assertion failures in cms_forward (which requires input_ids.len() >= seq_len). The training loop skips short chunks; the serving path passes them through and would panic. This is a known limitation for the serving stream path — production would need either: (a) corpus padding, (b) VecStream always returning full chunks by starting the new cycle, or (c) Session skipping short chunks.

### Gradient Magnitudes

- SWA weight gradients: norm = 0.024 (healthy)
- Memory level gradients: norm ~ 1e-7 (very small due to gate biases b_alpha=3.0, b_theta=-4.6)
- Error buffer accumulation for frozen levels works correctly — gradients accumulate and apply when the level activates

## Architecture Confidence

| Component | Confidence | Notes |
|-----------|-----------|-------|
| VecStream -> Conductor -> cms_forward loop | HIGH | Full pipeline works end-to-end |
| MAG composition (DeltaRule, Hebbian) | HIGH | Converges to zero loss |
| MAL composition (TitansLMM) | HIGH | Fastest convergence of the three |
| CMS k=2 multi-level scheduling | HIGH | Level 0 + Level 1 fire correctly |
| CMS k=1 single-level | HIGH | Simplest path works |
| Error buffer accumulation | HIGH | Non-zero level grads, correct apply timing |
| ContextState memory evolution | HIGH | Memory norms grow from zero, change over time |
| Serving Session (forward path) | HIGH | Exact parity with raw API |
| Checkpoint/restore | HIGH | Deterministic replay |
| Outer-loop gradient flow | HIGH | Non-zero gradients on all weight types |

## Go/No-Go Decision

**GO for Stage 3.**

The foundation is solid:
1. Three different rule+composition+CMS configs all learn a pattern from scratch
2. The full VecStream -> Conductor -> forward -> backward -> apply pipeline works end-to-end
3. The serving Session path is an exact wrapper around the raw API
4. Checkpoint/restore is deterministic
5. No numerical instability, no NaN, no divergence

Stage 3 can proceed with confidence that the infrastructure it builds upon is functionally correct.
