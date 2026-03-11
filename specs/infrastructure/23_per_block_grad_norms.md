# Per-Block Gradient Norms and Depth Specialization Metric

```
CONTRACT
  Purpose:    Expose per-block L2 gradient norms from the stacked optimizer so that
              block-level convergence and depth specialization can be observed during
              training. Provides a coefficient of variation (CV) metric over block
              gnorms that quantifies how much the 4 blocks have diverged in their
              learning dynamics — a proxy for depth specialization readiness.

  Expects:    - GpuStackedGrads with per-block gradient buffers (gpu_stacked_backward.rs)
              - GpuStackedAdamWState with norm_scratch/norm_host (gpu_stacked_optimizer.rs)
              - grad_norm_sq_cuda kernel (core/kernels/grad_norm.cu)
              - GpuStackedModel PyO3 wrapper (python/src/lib.rs)
              - JSONL logging infrastructure (loop.py)

  Guarantees: 1. gpu_stacked_per_block_grad_norms returns Vec<f32> of length n_blocks,
                 computed BEFORE global gradient clipping (same as per-level gnorms
                 in single-block path).
              2. Per-block gnorms are opt-in via collect_block_gnorms flag on step_adamw
                 to avoid overhead on non-logging steps.
              3. block_gnorm_cv (coefficient of variation = std/mean) is computed in
                 Python and logged to JSONL on logging steps.
              4. No new CUDA kernels required — reuses existing grad_norm_sq_cuda.
              5. No change to gradient clipping, optimizer update, or forward/backward
                 paths — this is purely observational.

  Cost:       Per logging step: n_blocks host syncs per block (same pattern as existing
              gpu_stacked_grad_norm, just partitioned). At n_blocks=4 with ~15 tensors
              per block, this is ~60 kernel launches + syncs. Negligible compared to
              the forward/backward pass (~2ms total for d=512).

  Trade-off:  Adding a per-block gnorm vector to the log increases JSONL line size by
              ~40 bytes per logged step. The CV computation is trivial (4 floats).
              The diagnostic value far outweighs the cost.

  Position:   specs/infrastructure/23_per_block_grad_norms.md

  Source:     Spec 12 (metric_driven_promotion) — convergence signals for promotion
              Spec 14 (multi_block_stacking) — stacked architecture
              Spec 19 (stacked_push_up_experiment) — "L0 gnorm per block > 0.01"
              Internal: depth-specialization hypothesis (session 2026-03-10)
```

---

## 1. Motivation: Step Count Is Not a Promotion Signal

Spec 19 uses fixed 20K steps for Phase 1 and 15K for Phase 2-3. Spec 12 established
that step count is a meaningless promotion signal — convergence rate varies by data,
model size, and level frequency. But spec 12's convergence signals operate per-level
(gnorm plateau, saturation EMA). For stacked models, we need a per-block dimension too.

### What per-block gnorms tell us

In a 4-block stacked model, each block occupies a different position in the residual
stream:

```text
Block 0: sees raw embeddings (shallow features — tokenization artifacts, common patterns)
Block 1: sees block 0's output (intermediate features — phrase structure, syntax)
Block 2: sees 2x refined output (deeper features — semantic relationships)
Block 3: sees 3x refined output (deepest features — long-range dependencies, discourse)
```

During training, these blocks should develop different gradient magnitudes because they're
learning different things at different rates. Block 0 learns shallow patterns quickly
(high early gnorm, fast decay). Block 3 needs blocks 0-2 to produce useful features first
(lower early gnorm, delayed peak).

**The key insight**: when the blocks' gnorms have diverged and stabilized, each block has
found its depth-specific role. This is when promotion is meaningful — the L0 at each depth
has learned what it can learn, and the knowledge is ready to be pushed up to a slower level.

### Why global gnorm hides this

The current `gpu_stacked_grad_norm` returns a single scalar: the L2 norm across ALL
parameters of ALL blocks. This masks block-level dynamics. If block 0 is converging (gnorm
dropping) while block 3 is still learning (gnorm rising), the global norm stays flat — it
looks like a plateau but is actually two blocks in different phases.

---

## 2. Metric: Block Gnorm Coefficient of Variation (CV)

### Definition

```text
block_gnorms = [gnorm_block_0, gnorm_block_1, gnorm_block_2, gnorm_block_3]
mean = sum(block_gnorms) / n_blocks
std  = sqrt(sum((g - mean)^2 for g in block_gnorms) / n_blocks)
cv   = std / mean    (if mean > 0, else 0)
```

### Interpretation

| CV value | Interpretation |
|----------|---------------|
| ~0 | All blocks have identical gnorms — no depth specialization yet |
| 0.05–0.15 | Mild differentiation — blocks starting to specialize |
| 0.15–0.30 | Clear specialization — blocks at different convergence rates |
| >0.30 | Strong divergence — blocks have found very different roles |
| Stabilized | CV has stopped changing — specialization complete |

### Promotion readiness signal

Promotion readiness requires BOTH:
1. **CV has stabilized** — rolling window CV change < threshold (block specialization saturated)
2. **All block gnorms above minimum** — no block has collapsed (>0.01 for L0, spec 19)

The CV stability check prevents premature promotion: if blocks are still differentiating,
pushing up would interrupt depth specialization before it completes.

### Phase 2+ usage

At k>=2, per-block gnorms include all levels within each block. The CV measures whether
blocks have diverged in their TOTAL learning dynamics (all levels combined per block).
This is complementary to per-level gnorms (spec 12), which measure whether individual
levels within a block have saturated.

Full promotion readiness at k>=2:
- Per-level saturation EMA (spec 12): has each level extracted available signal?
- Per-block CV stability (this spec): has depth specialization stabilized?
- Both must be met before the next push-up.

---

## 3. Implementation

### 3a. Rust: `gpu_stacked_per_block_grad_norms`

New function in `gpu_stacked_optimizer.rs`:

```rust
/// Compute per-block L2 gradient norms. Returns Vec<f32> of length n_blocks.
/// Called before global clipping so values reflect the true per-block learning
/// signal. Includes SWA projections, LayerNorms, and all memory levels for
/// each block. Shared params (embed, unembed, ln_final) are NOT included
/// because they are shared across all blocks and would add the same energy
/// to every block's norm.
#[cfg(feature = "cuda")]
pub fn gpu_stacked_per_block_grad_norms(
    grads: &GpuStackedGrads,
    state: &mut GpuStackedAdamWState,
) -> Vec<f32> {
    let mut block_norms = Vec::with_capacity(grads.blocks.len());

    for bg in &grads.blocks {
        let mut block_sq = 0.0f64;

        // Closure: accumulate squared norm for one GPU buffer
        let mut accum = |g: &GpuBuf<f32>| {
            let n = g.len() as i32;
            if n == 0 { return; }
            let mut num_blocks: i32 = 0;
            let err = unsafe {
                crate::cuda_ffi::grad_norm_sq_cuda(
                    g.as_ptr(), state.norm_scratch.ptr(),
                    n, &mut num_blocks,
                )
            };
            assert_eq!(err, 0);
            crate::dispatch::cuda_sync();
            let nb = num_blocks as usize;
            state.norm_scratch.slice(0, nb)
                .copy_to_host(&mut state.norm_host[..nb]);
            for i in 0..nb {
                block_sq += state.norm_host[i] as f64;
            }
        };

        // SWA projections for this block
        accum(&bg.d_w_q);
        accum(&bg.d_w_k);
        accum(&bg.d_w_v);
        accum(&bg.d_w_o);
        accum(&bg.d_ln_attn_gamma);
        accum(&bg.d_ln_attn_beta);
        accum(&bg.d_ln_mem_gamma);
        accum(&bg.d_ln_mem_beta);

        // All levels within this block
        for lg in &bg.levels {
            accum(&lg.d_w_k_mem);
            accum(&lg.d_w_v_mem);
            accum(&lg.d_w_q_mem);
            accum(&lg.d_w_alpha);
            accum(&lg.d_b_alpha);
            accum(&lg.d_w_theta);
            accum(&lg.d_b_theta);
            accum(&lg.d_w_eta);
            accum(&lg.d_b_eta);
        }

        // alpha_mem gradients (host-side)
        for &g in &bg.d_alpha_mem {
            block_sq += (g as f64) * (g as f64);
        }

        block_norms.push(block_sq.sqrt() as f32);
    }

    block_norms
}
```

### 3b. PyO3: expose on GpuStackedModel

Add `collect_block_gnorms` flag to `step_adamw` and `block_grad_norms()` accessor:

```rust
// In GpuStackedModel:
last_block_gnorms: Vec<f32>,

// In step_adamw, after backward, before clipping:
if collect_block_gnorms {
    self.last_block_gnorms =
        gpu_stacked_per_block_grad_norms(&grads, state);
} else {
    self.last_block_gnorms.clear();
}

fn block_grad_norms(&self) -> Vec<f32> {
    self.last_block_gnorms.clone()
}
```

### 3c. Python: CV computation and logging

In `loop.py`, on logging steps:

```python
# After step_adamw, on logging steps:
if is_stacked and hasattr(gpu_model, "block_grad_norms"):
    block_gnorms = gpu_model.block_grad_norms()
    if block_gnorms:
        mean_bg = sum(block_gnorms) / len(block_gnorms)
        if mean_bg > 0:
            var_bg = sum((g - mean_bg) ** 2 for g in block_gnorms) / len(block_gnorms)
            block_gnorm_cv = (var_bg ** 0.5) / mean_bg
        else:
            block_gnorm_cv = 0.0
        log_fields["block_grad_norms"] = [round(g, 6) for g in block_gnorms]
        log_fields["block_gnorm_cv"] = round(block_gnorm_cv, 6)
```

### 3d. JSONL output format

```json
{
  "event": "step",
  "step": 5000,
  "loss": 4.21,
  "grad_norm": 2.65,
  "block_grad_norms": [1.82, 1.41, 1.15, 0.93],
  "block_gnorm_cv": 0.216,
  ...
}
```

---

## 4. What Is NOT In This Spec

- **Automated promotion triggering**: This spec adds the METRIC. Spec 12 defines the
  promotion POLICY. The CV metric feeds into spec 12's convergence signals but does not
  trigger promotion on its own.

- **Per-block per-level gnorms**: A full n_blocks x k matrix of gnorms. Useful but
  expensive (4*4=16 norm computations at k=4). Defer until needed. Per-block (summed
  across levels) is sufficient for the depth specialization signal.

- **CV threshold tuning**: The exact CV threshold for "specialization complete" depends
  on architecture, data, and d_model. Phase 1 of EXP-15 (this run) establishes the
  baseline. We observe the CV trajectory, then set thresholds empirically.

---

## 5. Relationship to Existing Diagnostics

| Diagnostic | Granularity | What it measures | Spec |
|-----------|------------|-----------------|------|
| `grad_norm` | Global | Total gradient energy | Built-in |
| `level_grad_norms` | Per-level (single-block) | Level convergence rates | Spec 12 |
| `level_output_gnorms` | Per-level per-block (backward) | Gradient signal entering each level | Spec 15 |
| **`block_grad_norms`** | **Per-block** | **Depth specialization** | **This spec** |
| **`block_gnorm_cv`** | **Scalar** | **Specialization degree** | **This spec** |

---

## 6. Falsification

This diagnostic is uninformative if:

1. **All 4 blocks maintain identical gnorms through training** (CV stays ~0).
   Would mean the residual stream does not differentiate blocks — depth
   specialization is not occurring. This would partially falsify spec 19's
   depth-specialization hypothesis.

2. **CV diverges monotonically without stabilizing**. Would mean blocks never
   reach equilibrium — one block dominates while others collapse. This would
   indicate an architectural problem (unbalanced residual connections).

3. **CV oscillates with high amplitude**. Would mean block specialization is
   unstable — blocks trade roles across training. This would suggest the
   residual stream carries too much information for stable specialization.

---

## 7. Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_stacked_optimizer.rs` | Add `gpu_stacked_per_block_grad_norms()` |
| `python/src/lib.rs` | Add `collect_block_gnorms` to stacked `step_adamw`, add `block_grad_norms()` accessor |
| `python/engine/loop.py` | Compute CV, log `block_grad_norms` and `block_gnorm_cv` to JSONL |

---

## 8. Acceptance Criteria

1. `gpu_stacked_per_block_grad_norms` returns `Vec<f32>` of length `n_blocks`
2. Each entry is the L2 norm of all gradients belonging to that block (SWA + all levels)
3. Shared params (embed, unembed, ln_final) excluded from per-block norms
4. Computed before gradient clipping (pre-clip values)
5. Opt-in: zero overhead when `collect_block_gnorms=false`
6. `block_gnorm_cv` logged to JSONL on logging steps
7. No change to forward/backward/optimizer paths — purely observational
8. Single-block path unaffected (no regressions)

---

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-097-hope-cms-chain | hope_equations | HOPE S5.1 | informs (CMS block structure) |
| eq-070-arch-variant1 | hope_equations | HOPE CMS chain | informs (stacked variant) |

## Code Smells

| Smell | Enforcement | Rationale |
|-------|-------------|-----------|
| CS-04/05/06 | ontological | "blocks" for depth, "levels" for CMS frequency — per-block gnorms respect this |
| CS-10 | behavioral | No mode flag — gnorm collection is opt-in per step, not per mode |
| CS-18 | architectural | Norm computation in Rust tier, CV computation + logging in Python tier |
| CS-40 | architectural | Opt-in — collect_block_gnorms flag, zero overhead when off |
