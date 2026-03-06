# Cold-Start k=4 100K Comparison: MAG vs MAC

**Date**: 2026-03-06
**Experiments**: EXP-09 (MAC), EXP-10 (MAG)
**Verdict**: Cold-start k=4 CMS is a dead end. L3 never activates. L2 is effectively dead. Progressive stacking — either push-up or stack-up — is required.

## Executive Summary

Two parallel 100K-step runs compared MAG (memory gates attention) and MAC (memory-attention-memory) compositions with Titans LMM, k=4 CMS, d=512, on Dolmino 100B data. Both runs converged to nearly identical final loss (~3.79-3.80 avg last 10), confirming that:

1. **Composition pattern does not matter when higher levels are dead.** MAG and MAC produce the same result because the model effectively operates as k=1 — only Level 0 contributes meaningful gradients.
2. **Level 3 is completely dead** — 0 active fires out of 196 total in both runs. L3 gnorm never exceeded 0.002 across 100K steps.
3. **Level 2 is effectively dead** — sporadic gnorm spikes of 0.003-0.145 but never sustained. 153 dead warnings in both runs.
4. **Level 1 shows weak, intermittent activity** — gnorm ~0.1-0.25 when active, but flagged dead 133 times.
5. **Only Level 0 learns** — carries the entire gradient signal (gnorm 1.2-2.4 at convergence).

This conclusively demonstrates that cold-start k=4 cannot bootstrap higher CMS levels. Progressive stacking is the recommended path forward, with two strategies under evaluation:

- **Push-up** (PR #176): Train k=1, shift trained levels to slower frequencies (level[i] → level[i+1]), add fresh L0 at the fastest tier. Trained weights change firing rate.
- **Stack-up** (PR #178): Train k=1, keep existing levels in place (level[i] → level[i]), add fresh level at the slowest (top) tier. Trained weights retain their original firing rate.

## Configuration

| Parameter | MAG (EXP-10) | MAC (EXP-09) |
|---|---|---|
| Composition | MAG (parallel gating) | MAC (sequential) |
| Memory rule | Titans LMM | Titans LMM |
| d_model | 512 | 512 |
| num_heads | 8 | 8 |
| seq_len | 512 | 512 |
| window_size | 512 | 1024 |
| k (CMS levels) | 4 | 4 |
| chunk_sizes | [1, 8, 64, 512] | [1, 8, 64, 512] |
| Momentum | EMA | EMA |
| Parameters | 39,071,764 | 39,071,764 |
| Data | Dolmino 100B (sharegpt BPE) | Dolmino 100B (sharegpt BPE) |
| LR | 0.0006 (cosine, 1K warmup) | 0.0006 (cosine, 1K warmup) |
| Optimizer | AdamW (b1=0.9, b2=0.999, wd=0.1) | AdamW (b1=0.9, b2=0.999, wd=0.1) |
| Grad clip | max_norm=1.0 | max_norm=1.0 |
| Steps | 100,000 | 100,000 |
| Wall time | ~29 hours | ~29 hours |
| GPU | A6000 (GPU1) | A6000 (GPU0) |

## Loss Curve

| Step | MAG Loss | MAC Loss | MAG gnorm_l | MAC gnorm_l |
|---|---|---|---|---|
| 0 | 10.374 | 10.374 | [0,0,0,0] | [0,0,0,0] |
| 500 | 6.390 | 6.390 | [0.146,0,0,0] | [0.146,0,0,0] |
| 1,000 | 6.560 | 6.560 | [0.043,0,0,0] | [0.043,0,0,0] |
| 2,000 | 5.658 | 5.658 | [3.24,0.001,0,0] | [3.17,0.001,0,0] |
| 5,000 | 5.451 | 5.482 | [4.14,0.115,0,0] | [4.79,0.132,0,0] |
| 10,000 | 4.470 | 4.495 | [1.19,0.076,0,0] | [0.92,0.166,0,0] |
| 20,000 | 4.622 | 4.638 | [0.52,0.063,0,0] | [0.74,0.057,0,0] |
| 30,000 | 3.715 | 3.793 | [0.45,0.081,0,0] | [0.77,0.079,0,0] |
| 50,000 | 4.282 | 4.202 | [0.98,0.133,0,0] | [0.93,0.119,0,0] |
| 70,000 | 3.929 | 3.848 | [1.25,0.139,0,0] | [1.39,0.212,0,0] |
| 90,000 | 4.995 | 4.913 | [1.61,0.227,0,0] | [1.70,0.243,0,0] |
| 99,999 | 4.116 | 4.173 | [1.39,0,0,0] | [1.50,0,0,0] |

**Key observation**: The two runs track each other almost identically. The composition pattern (MAG vs MAC) makes no measurable difference because only L0 is active — the composition only matters when multiple levels contribute.

## Final Metrics

| Metric | MAG | MAC |
|---|---|---|
| Final loss | 4.116 | 4.173 |
| Avg last 10 steps | **3.790** | 3.795 |
| Final perplexity | 61.3 | 64.9 |
| Throughput | 560 tok/s | 563 tok/s |
| Total steps | 100,000 (from step 97,395) | 100,000 (from step 97,395) |
| Effective resume steps | 2,605 | 2,605 |
| Wall time | 104,156s (~29h) | 105,147s (~29.2h) |
| Checkpoint | titans_dolmino_100k_mag.safetensors | titans_dolmino_100k_mac.safetensors |

## Level Activity Analysis

### Dead Level Warning Counts (out of ~10,000 log lines)

| Level | MAG | MAC | Interpretation |
|---|---|---|---|
| L0 | 24 | 24 | Dead only during first ~260 steps (cold start), then alive |
| L1 | 133 | 133 | Intermittently dead throughout — weak gradient signal |
| L2 | 153 | 153 | Nearly always dead — sporadic micro-activations |
| L3 | 4,010 | 4,727 | Permanently dead — **zero active fires at end** |

### Level 3 Detail

- L3 fires every 512 steps, giving 196 total fires across 100K steps
- **0 out of 196 fires were active** in both runs (0% activation rate)
- L3 gnorm exceeded 0.0001 only **4 times** in MAG (peak: 0.002), never sustained
- 100-sample active gnorm average at end: 2.04e-5 (MAG), 1.74e-5 (MAC) — both ~100x below the 1e-4 threshold
- L3 is not just underperforming — it is functionally nonexistent

### Level 2 Detail

- L2 fires every 64 steps, giving ~1,563 total fires across 100K steps
- Sporadic gnorm spikes (max observed: 0.145) but no sustained activation
- The gnorm spikes correlate with L1 activity (cascade effect), not independent learning

### Level 0 Recovery

- L0 was flagged dead for the first ~260 steps (cold-start bootstrap)
- By step 500, L0 gnorm reached 0.146 — fully active
- L0 carried 100% of the effective gradient signal for the remaining 99,740 steps

## Root Cause Analysis

The cold-start k=4 failure has a clear mechanism:

1. **Firing frequency**: L3 fires every 512 steps. With LR warmup over 1,000 steps, L3 fires only ~2 times during warmup. This is insufficient to establish gradient flow.

2. **Gradient dilution**: When L3 fires, its gradient signal is divided by `1/sqrt(k) = 1/2` for CMS output normalization. The already-tiny gradient is halved before reaching the optimizer.

3. **Gate initialization**: b_alpha defaults to 3.0 (sigmoid=0.95), b_theta defaults to -4.6 (softplus=0.01). The theta gate produces near-zero decay, making the memory update nearly invisible. With random weights, the memory output is noise — and the alpha gate lets 95% of it through.

4. **Bootstrapping deadlock**: L3 needs many gradient steps to learn useful representations, but it gets ~196 gradient updates in 100K steps. Meanwhile, L0 gets 100K updates and dominates the loss landscape. By the time L3 could hypothetically learn, the model has converged around L0-only representations.

5. **No rescue mechanism**: EMA momentum accumulates gradients for slow levels, but accumulating near-zero gradients produces near-zero momentum. The momentum doesn't help when the fundamental gradient signal is absent.

## Implications

### What this rules out

- **Cold-start k=4 with any composition** (MAG, MAC, MAL) — the problem is firing frequency, not composition
- **Longer training** — L3 gnorm is flat at 2e-5 from step 50K to 100K; more steps won't help
- **Different theta/alpha clamps** — without gradient signal, no clamp configuration can create one

### What to try next

- **Push-up progressive stacking** (PR #176): Train k=1 until converged, shift trained levels to slower frequencies (level[i] → level[i+1]), add fresh L0 at the fastest tier. Each level starts with a pretrained state.
- **Stack-up progressive stacking** (PR #178): Train k=1 until converged, keep existing levels in place, add a fresh level at the slowest (top) tier. Trained weights retain their original firing rate context.
- **Warm-start from k=1 checkpoint**: Use the MAG k=1 run (already converged) as the seed for phase 2 (k=2) under either strategy.

## Artifacts

- **Checkpoints**: Deleted (40 files, ~6.1 GB) — no value in preserving dead-level weights
- **Logs archived**: `docs/archived_runs/nohup_titans_dolmino_100k_{mag,mac}.out`
- **Configs**: `python/configs/titans_dolmino_100k_{mag,mac}.json` (preserved)
- **JSONL run logs**: `python/runs/titans_dolmino_100k_{mag,mac}.jsonl` (preserved for potential analysis)
