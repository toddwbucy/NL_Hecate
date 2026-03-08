# EXP: Residual Stream Validation — Data Rewind Ablation on Push-Up

```text
CONTRACT
  Purpose  : Validate that the residual stream (PR #180) fixes dead higher CMS
             levels, and determine whether rewinding data on push-up helps level
             bootstrapping
  Expects  : Merged residual stream (residual=true), push-up stacking, Dolmino 100B
  Guarantees: Two matched auto-promote runs differing ONLY in data cursor policy
             at push-up time
  Cost     : 2 GPUs × auto-promote k=1→k=4 ≈ 24-48 GPU-hours
  Trade-off: Smaller model (36M) than Titans paper (170M), but sufficient to
             validate gradient flow fix before scaling
  Position : Experiment — prerequisite to multi-block stacking and 170M scaling
  Source   : Residual stream spec (specs/infrastructure/13_residual_stream.md)
             Rewind math (lines 39-42) is original experimental design, not paper-derived.
```

## Hypothesis

The residual stream provides gradient=1.0 additive path to all CMS levels.
With this fix, higher levels should show nonzero gradients after push-up promotion.

## Independent Variable: Data Rewind Policy on Push-Up

| Arm | GPU | Label | promotion_rewind_pct | Behavior |
|-----|-----|-------|---------------------|----------|
| A | GPU0 | 25% rewind | 0.25 | On push-up, rewind cursor by 25% of tokens consumed during THIS level's phase. New L0 gets a primer from prior data, then moves into fresh data. |
| B | GPU1 | no rewind | 0.0 | Cursor continues forward. New L0 only sees fresh data. Trusts residual stream to carry prior context. |

### Why these two arms

- If B succeeds → residual stream alone is sufficient, no data management needed
- If A succeeds but B fails → new levels need a data primer to bootstrap
- If both fail → problem isn't gradient attenuation (falsification criterion from spec 13)
- If both succeed with A > B → option 1 (full rewind) worth testing next

### What "25% of THIS level's steps" means

When push-up fires (e.g. k=2 saturates at cursor position 4.8M, having started at 3.2M):
- tokens_this_level = 4.8M - 3.2M = 1.6M
- rewind = 25% × 1.6M = 400K
- new cursor = 4.8M - 400K = 4.4M
- New L0 re-reads 400K tokens the k=2 level saw, then continues into fresh data

## Controlled Variables (identical across arms)

- Model: d=512, 8 heads, seq_len=512, vocab=32000, Titans LMM, MAG, EMA
- **residual=true** (the feature under test)
- Auto-promote: target_k=4, cooldown=2000 steps
- Push-up stacking (not stack-up)
- CMS chunk template: [1, 8, 64, 512]
- Optimizer: AdamW GPU, lr=6e-4, warmup=500, wd=0.1, b1=0.9, b2=0.999
- Grad clip: max_norm=1.0
- m_norm_max: [100.0] per level
- Dataset: Dolmino 100B (~950M tokens)
- Steps: 60,000 (enough for 3-4 promotions)

## Success Criteria

1. **Primary**: After first promotion (k=1→k=2), L1 gnorm > 0.01 within 100 steps
   (Previous experiments without residual: L1 gnorm ≈ 0.000002)
2. **Secondary**: After reaching k=4, all 4 levels show nonzero gnorm
3. **Ablation signal**: Detectable difference in level gnorms between arms

## Falsification

If higher levels STILL show zero gradients with residual=true, the dead-level
problem is not gradient attenuation. See spec 13 for alternative hypotheses.

## Config Files

- `residual_rewind25_gpu0.json` — Arm A (25% rewind)
- `residual_norewind_gpu1.json` — Arm B (no rewind)

## Monitoring

- `tape_every: 500` — per-level gnorm in tape summaries
- `eval_every: 500` — loss tracking
- `log_every: 10` — step-level loss
- Check level gnorms after each promotion (first 100 steps of new phase)
