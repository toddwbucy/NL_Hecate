# Auto-Promotion Pyramid on Dolmino 100B (GPU1)

**Date**: 2026-03-07
**Status**: IN PROGRESS
**GPU**: 1 (NVIDIA RTX A6000, 48GB)

## Hypothesis

Convergence-driven auto-promotion can replace manual multi-phase pyramid training. Rather than pre-defining phase boundaries and running separate k=1, k=2, k=3, k=4 jobs with hand-tuned step counts, the system detects when L0's learning signal has plateaued and automatically extends the model via push-up stacking — all within a single continuous run.

This addresses two failure modes observed in the manual pyramid approach (documented in HADES: `pyramid_manual_stacking_failure_2026_03_07`):

1. **Data re-exposure confound**: Manual phases 1 and 2 trained on overlapping token ranges [0, ~5M), so promoted levels carried redundant representations from shared training data. Auto-promotion with a monotonically advancing data cursor eliminates this by design.
2. **Missing m_norm_max**: The manual pyramid omitted the M-norm clamp, leading to Frobenius norm divergence and NaN at step 4805. The auto-promotion config includes `m_norm_max: [100.0]` from the start, extended automatically at each promotion.

## Saturation Detection: Ratio Stability via Trimmed Standard Deviation

The promotion signal is NOT a flat threshold on gnorm or loss. Instead, it detects when L0's learning has entered a "bouncing corridor" — still active, but not trending downward.

**Mechanism:**
1. Track EMA of L0 gnorm (α=0.1) and running peak
2. Compute saturation ratio: `EMA / peak` (0 = dead, 1 = at peak, <1 = declining)
3. Maintain a rolling window (50 samples) of ratio values
4. Compute **trimmed standard deviation**: drop top/bottom 10% of samples, then stdev of remainder
5. If trimmed stdev < 0.025 for 50 consecutive samples → L0 has plateaued → promote

**Why trimmed stdev?** The Dolmino 100B corpus has mixed difficulty — easy boilerplate passages interleaved with hard technical content. Raw stdev is inflated by gnorm spikes on hard passages. Trimming the outliers filters data-difficulty variance from the stability signal.

**Why not a flat threshold?** L0 gnorm never drops to a fixed "saturated" value on mixed data. It bounces between 0.2 and 8.4 throughout training. The ratio captures the trend relative to the model's own peak, and the stdev captures whether that trend has flattened.

## Configuration

| Parameter | Value |
|---|---|
| Composition | MAG (parallel gating) |
| Memory rule | Titans LMM |
| d_model | 512 |
| num_heads | 8 |
| seq_len | 512 |
| window_size | 512 |
| k (initial) | 1 |
| target_k | 4 |
| chunk_sizes | [1] → [1,8] → [1,8,64] → [1,8,64,512] |
| Momentum | EMA |
| M-norm clamp | 100.0 per level (extended at promotion) |
| Data | Dolmino 100B (sharegpt BPE, 950M tokens) |
| LR | 0.0006 (linear warmup 500 steps, no decay) |
| Optimizer | AdamW (b1=0.9, b2=0.999, wd=0.1) |
| Grad clip | max_norm=1.0 |
| Steps | 100,000 max |
| Promotion cooldown | 2,000 steps between promotions |
| Stability window | 50 ratio samples |
| Stability streak | 50 consecutive low-stdev samples |
| Stability threshold | 0.025 trimmed stdev |
| log_every | 10 |
| eval_every | 500 |
| save_every | 5,000 |

## Promotion Events

### Promotion 1: k=1 → k=2 (step 4580)

- **Trigger**: ratio_stdev = 0.024287, ratio_mean = 0.1613
- **L0 state**: EMA gnorm = 1.45, peak gnorm = 8.10 (ratio 17.8%)
- **Data cursor**: 2,345,472 tokens consumed
- **Note**: Initial run crashed at this point due to `tnt_global_chunk_size=None` bug in MAGConfig construction. Fixed by using `bcfg.tnt_global_chunk_size` (default 64). Resumed from pre-promotion checkpoint with manual `extend_k=2, push_up=true`, auto-promote enabled for subsequent promotions.

Post-promotion k=2 behavior:
- Loss started at 4.63 (not 10+ — SWA weights preserved)
- L0 gnorm: 0.3-7.0 (active, learning)
- L1 gnorm: 0.2-1.0 when firing (every 8th step), healthy
- L0 ‖M‖ hit 100.0 clamp by step 1500 (pushing hard)
- L1 ‖M‖ = 56.2 at step 1500 (growing, unclamped)

### Promotion 2: k=2 → k=3 (step 5690)

- **Trigger**: Auto-promotion fired cleanly (no crash)
- **Data cursor**: 5,259,264 tokens consumed
- **Checkpoint**: `auto_pyramid_gpu1_pre_k3_step5690.safetensors`
- **Post-promotion**: Loss stable 4.1-5.8, single gnorm spike to 25.4 at step 5770 (recovered immediately)
- L0 gnorm: 0.8-4.1 (active)
- L1/L2: Not yet observed in logs due to `log_every=10` not aligning with chunk boundaries (8, 64)

### Promotion 3: k=3 → k=4

- **Status**: PENDING. Cooldown expires at step 7690. Saturation detector active.

## Observations (in progress)

_To be completed when run finishes._

### Learning probes (step 1500, k=2)
- Within-generation: loss 3.51 → 1.09, slope = -0.245 (strong in-context learning)
- Cross-exposure: 71.9% improvement on second exposure (memory retention functional)

## Known Issues

1. **log_every=10 misaligns with CMS chunk boundaries**: L1 (every 8) and L2 (every 64) fire events rarely coincide with logged steps. Fixed in subsequent configs by using `log_every=8`.
2. **eval_every=500 misaligns with higher levels**: L3 (every 512) never fires during eval. Fixed in subsequent configs with `eval_every=4096`.
3. **Dead-level detector fires during warmup**: Per-level gnorms are near-zero when lr < 0.001. False positive. Not fatal (warning only).

## Results

_To be completed when run finishes. Will include:_
- Final loss and perplexity at each k level
- All promotion timestamps and trigger values
- Level activity heatmaps at k=4
- Learning probe trajectories
- Comparison with manual pyramid (failed) and cold-start k=4 (dead L3)
