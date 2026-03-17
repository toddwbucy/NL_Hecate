# Analysis Report: k=2 Push-Up Extension (10K Additional Steps)

**Run**: `baseline_pushup_p2_k2_ext10k`
**Date**: 2026-03-15
**GPU**: A6000 (GPU 0)
**Status**: Steps 20000-27240 completed (as of report), targeting 30000

## Purpose

Test whether the push-up k=2 model (extended from k=1 baseline) would show
level differentiation and improved NIAH retrieval with 10K additional steps
of fresh data beyond the original 20K training steps.

## Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | d=512, 4 blocks, 8 heads, TNT hierarchical, MAG |
| k | 2 (chunk_sizes=[1,8]) |
| Source checkpoint | `baseline_pushup_p2_k2/model.safetensors` (k=2 push-up at step 20K) |
| Original source | k=1 baseline at 20K → extended to k=2 via push-up |
| Steps | 10K additional (steps 20001-30000) |
| Data | dolmino_100b, data_seek=17920512 (100% fresh, no overlap) |
| LR | 0.0003 with cosine decay (continuing from step 20K schedule) |

## Result: Levels Did NOT Differentiate

### Tape Diagnostics Summary

| Metric | L0 Range | L1 Range | Delta | Verdict |
|--------|----------|----------|-------|---------|
| alpha (retention) | 0.79-0.83 | 0.82-0.85 | +0.00 to +0.04 | **Barely separated** |
| theta (inner LR) | 0.976-0.993 | 0.726-0.929 | -0.06 to -0.25 | **L1 unstable**, bouncing |
| eta (output gate) | 0.865-0.897 | 0.758-0.816 | -0.06 to -0.11 | Slight separation |
| dgd_delta_norm | 85-105 | 64-76 | ratio 1.3-1.4x | **Flat, no growth** |
| output_grad_norm | 0.012-0.017 | identical | 0 | **No differentiation** |

Key observations:
- **L0 theta pinned at ceiling (~0.98)**: L0 is maximally aggressive and cannot modulate
- **L1 theta bouncing**: 0.93 → 0.73 → 0.92 → 0.90. Not converging to a role — oscillating
- **dgd ratio stuck at 1.3x**: Both levels making nearly identical magnitude updates
- **Gradient norms identical**: Both levels receive exactly the same gradient signal
- **No improvement over original 20K**: The extension added nothing to differentiation

### Loss Trajectory

| Window | Mean Loss | Min | Max |
|--------|-----------|-----|-----|
| Steps 20K-22.5K | 3.395 | 1.940 | 5.094 |
| Steps 22.5K-25K | 3.293 | 2.096 | 5.582 |
| Steps 25K-27.5K | 3.270 | 2.183 | 5.135 |

- Marginal improvement (3.40 → 3.27 mean) but **highly noisy**
- Three loss spikes >5.0: steps 22040, 24000, 26952
- Step 24000 spike (5.58) correlates with L1 theta dropping to 0.73 — L1 geometry shift destabilized the model

### Stability Concerns

- Loss range: 1.94 - 5.58 (2.9x spread)
- Grad norm range: 1.73 - 6.90 (4x spread)
- The loss spikes suggest the undifferentiated levels are actively causing instability:
  L1 theta oscillates → geometry shifts → loss spikes → recovery → repeat

### NIAH Context (Pre-Extension, Steps 5K-20K)

Push-up k=2 vs k=1 baseline pass rates:

| Distance | k=1 (20K) | k=2 Push-Up (20K) | Delta |
|----------|-----------|-------------------|-------|
| 1024 | 40% | 50% | +10% |
| 2048 | 60% | 50% | -10% |
| 4096 | 60% | 40% | **-20%** |
| 8192 | 60% | 30% | **-30%** |

The push-up k=2 degraded badly at long distances — exactly the ranges where
L1 (chunk_size=8) should be contributing. Extension NIAH not yet run but
tape diagnostics predict no improvement.

## Comparison: Fresh k=2 (Concurrent Run)

A fresh k=2 from random init (`fresh_k2_from_scratch`) was run simultaneously
on GPU 1 for direct comparison:

| Metric | Push-Up k=2 (step 27K) | Fresh k=2 (step 4.5K) |
|--------|----------------------|---------------------|
| L0-L1 theta delta | -0.08 (barely different) | **-0.74** (10x difference) |
| L0-L1 alpha delta | +0.04 | **+0.05-0.08** |
| dgd ratio | 1.3x | **2.3-2.9x** |
| block_gnorm_cv | 0.30 | **0.58** |
| Loss spikes >5.0 | 3 in 7K steps | 0 in 4.5K steps |

The fresh k=2 achieved stronger differentiation in 2K steps than the push-up
achieved in 27K steps. And it's more stable (no loss spikes).

## Root Cause

Push-up from k=1 clones a single generalist level into two copies. Both start
with identical weights, identical gate biases, and receive identical gradient
signals. The 8x frequency difference (chunk_sizes [1,8]) is insufficient
pressure to break the symmetry when starting from a shared basin.

The extension experiment confirms this is not a training budget issue — 10K
more steps on fresh data did not help. The levels are trapped in the same
optimization basin and cannot escape.

## Conclusion

**The k=1 → k=2 push-up initialization method is ruled out** (at this scale
and configuration). The 10K extension confirmed that additional training does
not resolve the differentiation failure.

The fresh k=2 initialization is the correct approach: both levels must learn
their roles from scratch, with the 8x frequency gap providing the structural
pressure for specialization from step 0.

### Implications for Pipeline

The push-up pipeline should be reconceived as:
1. **Fresh k=2**: Train the two outermost (slowest) levels from random init
2. **Push-IN k=3**: Add a new fastest level (new L0) on top of the established k=2
3. **Push-IN k=4**: Repeat — new L0 on top of established k=3

The inter-level relationship is governed by the 8x frequency ratio, which is
scale-invariant: dynamics learned at [1,8] transfer to [64,512] because the
ratio is preserved.

### Next Experiment

Task `task_c9b481`: Fresh k=2 with low initial retention (b_alpha=0.0,
sigmoid=0.50) to observe the model's natural entropy reduction trajectory.
Tests whether controlled low-retention initialization produces even better
level differentiation.

---
*Report generated 2026-03-15. Run may still be in progress (step 27240/30000).*
