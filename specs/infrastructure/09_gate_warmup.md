# Gate Activation Warmup Protocol

```text
CONTRACT
  Purpose:    Define a three-phase gate warmup protocol that addresses the
              initialization trap for low-frequency CMS levels (L2, L3). The
              trap arises because softplus-gated write amplitudes start near
              zero and AdamW first-moment momentum decays to near-zero between
              the rare L3 fire events (once per 512 steps). The protocol uses
              (1) explicit per-level b_theta initialization, (2) a transient
              decaying theta_floor scaffold, and (3) a natural equilibrium
              phase. The diagnostic falsification threshold — L2 θ > 0.005 AND
              L3 θ > 0.001 at step 20K — determines whether the warmup was
              sufficient before any pod-scale run is authorized.

  Expects:    - CMS k=4, fire periods [1, 8, 64, 512].
              - CS-39 theta_floor/theta_ceil wired end-to-end: CUDA kernel →
                gpu_forward.rs → lib.rs → config.py → loop.py (PR #146+#150).
              - loop.py can update gpu_params between steps (already the case
                for checkpoint/resume: MAGConfig is rebuilt from config).
              - Outer-loop optimizer: AdamW, β₁=0.9, β₂=0.999, lr configurable.
              - memory_rule=delta (k=4 Delta Rule) for all diagnostic runs.
              - memory_reset=carry_forward (TNT is orthogonal to gate warmup).
              - Dataset: ≥500M tokens, long-form books or deeply nested code
                graphs, lag-MI validated (see §6).

  Guarantees: - At the end of Phase 2 (step gate_warmup_decay_steps), all
                theta_floor values are exactly 0.0 — the scaffold is fully
                removed and gate dynamics are determined by gradient alone.
              - The diagnostic run reaches a go/no-go checkpoint at step 20K:
                both L2 θ_mean > 0.005 AND L3 θ_mean > 0.001 must hold.
              - No new Rust or CUDA code is required. The warmup is implemented
                purely in loop.py via linear interpolation of theta_floor and
                explicit b_theta initialization in the config.
              - CS-39 clamp remains a permanent safety rail even after the
                scaffold decays: theta_floor_final values (≥0.0) prevent
                pathological collapse below physical zero.

  Cost:       - One new config section: gate_warmup (§4). No Rust changes.
              - loop.py: ~15 lines to read the schedule and apply linear
                interpolation of theta_floor at each step during Phase 2.
              - Diagnostic run: k=4, d=256, seq_len=512, 20K steps on A6000.
                Estimated wall time: ~3 hours at ~128 tok/s (d=256 is faster).

  Trade-off:  The decaying theta_floor scaffold temporarily prevents gate
              collapse during early training — but it also prevents the gate
              from exploring values below the floor. If the gradient signal
              is very weak (which is the hypothesis being tested), the gate
              may simply track the scaffold rather than learn. The falsification
              threshold tests for this: if L3 θ ≤ 0.001 at step 20K even with
              the scaffold fully decayed, the gate cannot self-sustain activation
              on this dataset and architecture, and the pod run should not
              proceed. There is no warmup that can rescue a fundamentally
              information-scarce corpus or a broken gradient path.

  Position:   specs/infrastructure/09_gate_warmup.md
  Source:     HOPE (2512.24695) — CMS frequency structure, DGD update, per-level
                inner LR (η_t^(ℓ)).
                HADES: hope_equations/eq-102-gated-memory (write gate θ_t),
                       hope_equations/eq-100-freq-equilibrium (CMS frequency structure),
                       hope_equations/eq-106-freq-weights (per-level η_t^(ℓ)).
              TNT (2511.07343) §3.2 — shard boundaries and initialization trap
                analysis (motivates the frequency-dependent LR mismatch).
                HADES: tnt_equations/eq-006-local-memory-update,
                       tnt_equations/eq-014-n-local-memories-update.
              specs/infrastructure/08_tnt_periodic_reset.md — gate dormancy
                context, L3 fires 48× in 25K steps.
              specs/infrastructure/07_gate_backward.md — theta gate gradient
                flow: ∂θ/∂b_theta = softplus'(b_theta).
              nl_code_smells/CS-39 — clamp_theta as architectural constraint.
```

---

## 1. Problem: Gate Dormancy at Low-Frequency Levels

The CMS k=4 write gate for level ℓ is:

```text
θ_t^(ℓ) = softplus(b_theta^(ℓ))    [outer_loop_param, learned by AdamW]
```

The default initialization is `b_theta = [-4.6, -5.6, -6.6, -7.6]` for
levels 0–3, giving:

```text
θ^(L0) = softplus(-4.6) ≈ 0.010    fires every step      → ~20K updates
θ^(L1) = softplus(-5.6) ≈ 0.004    fires every 8 steps   → ~2500 updates
θ^(L2) = softplus(-6.6) ≈ 0.001    fires every 64 steps  → ~312 updates
θ^(L3) = softplus(-7.6) ≈ 0.0005   fires every 512 steps → ~48 updates
```

Observed outcome in all ablation runs (A/B/C/D and B-TNT/C-TNT/D-TNT):
**L3 b_theta moves less than 0.01 over 25K steps.** The gate is dormant.

---

## 2. Root Cause Analysis

### 2.1 AdamW Momentum Starvation

AdamW with β₁=0.9 accumulates first-moment momentum:

```text
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
```

When L3 fires at step t and then does not fire for another 512 steps, the
first moment decays:

```text
m_{t+512} ≈ β₁^512 · m_t = 0.9^512 ≈ 3.7 × 10^{-24} · m_t
```

The momentum accumulated at step t has essentially vanished by step t+512.
Each L3 update starts from near-zero momentum. AdamW's bias correction
partially compensates, but the effective step size is still ε/(√v_hat),
and with near-zero v_hat the update is noisy but not directional.

### 2.2 Gradient Magnitude at Near-Zero Gate

The gradient of the loss through the gate:

```text
∂L/∂b_theta = (∂L/∂θ) · softplus'(b_theta)
```

At b_theta = -7.6: softplus'(-7.6) = σ(-7.6) ≈ 5 × 10^{-4}. The gradient
is attenuated by 2000× relative to a gate near 0. Even if the loss gradient
∂L/∂θ is large, the signal reaching b_theta is vanishingly small.

### 2.3 Compound Effect

The two effects multiply: gradient attenuation × momentum starvation × rare
updates = initialization trap. The gate cannot escape the near-zero basin
without an explicit intervention.

---

## 3. Three-Phase Warmup Protocol

### Phase 1 — Gate Seeding (steps 0 → phase1_steps)

**Action**: Initialize b_theta per level with values that place each gate in
a practically active range from step 0:

```text
b_theta_init: [-4.6, -5.6, -4.6, -4.6]
                L0     L1    L2    L3
```

This sets:
- L2 θ = softplus(-4.6) ≈ 0.010 (vs default 0.001)
- L3 θ = softplus(-4.6) ≈ 0.010 (vs default 0.0005)

The L2 and L3 gates now start where L0 starts. AdamW can accumulate
meaningful first-moment signal from the first L2 fire (step 64) and the
first L3 fire (step 512).

This is NOT changing the architecture — b_theta is a learned parameter
whose initial value is a hyperparameter. The HOPE paper initializes at
"useful" values throughout (§4.2, DGD initialization discussion). We are
applying the same principle per-level.

**Config key**: `b_theta` in the model section. No new infrastructure needed.

### Phase 2 — Scaffold Warmup (steps 0 → gate_warmup_decay_steps)

**Action**: Apply a linearly-decaying theta_floor schedule. At step t:

```text
theta_floor_t^(ℓ) = theta_floor_init^(ℓ) · max(0, 1 - t / gate_warmup_decay_steps)
```

The floor begins at `theta_floor_init` (the scaffold level) and reaches
exactly 0.0 at step `gate_warmup_decay_steps`. After that, no floor is
applied (or a final safety floor of 0.0 is preserved).

Recommended init values:

```text
theta_floor_init: [0.0, 0.0, 0.005, 0.001]
                   L0   L1    L2     L3
```

L0 and L1 need no scaffold — they fire frequently enough for AdamW to
function normally. L2 and L3 receive a floor at their falsification
threshold values. If the gate gradient is positive (gate wants to open),
the floor does not bind and the gate trains freely. If the gradient is weak
or negative early on, the floor prevents premature collapse below the
measurement threshold.

The scaffold decays linearly so that by the falsification checkpoint (step
20K, with decay_steps=10K), the gate has been operating without support for
10K steps. Any gate value above threshold at step 20K is the result of
genuine learned activation, not scaffold clamping.

**Implementation**: loop.py, between steps 0 and gate_warmup_decay_steps:

```python
if step < cfg.gate_warmup_decay_steps:
    alpha = 1.0 - step / cfg.gate_warmup_decay_steps
    current_floor = [f * alpha for f in cfg.theta_floor_init]
    # Rebuild gpu_params with updated theta_floor (same path as resume)
    update_theta_floor(gpu_params, current_floor)
```

### Phase 3 — Natural Equilibrium (steps gate_warmup_decay_steps → end)

**Action**: theta_floor = [0.0, 0.0, 0.0, 0.0] (or final safety values ≥ 0).
All gate dynamics from this point are entirely gradient-driven. The scaffold
is gone. The falsification checkpoint occurs at step 20K.

If `gate_warmup_decay_steps = 10K`:
- Phase 2 ends at step 10K
- Phase 3 runs 10K–20K+ (10K steps without scaffold)
- Falsification checkpoint at step 20K measures whether the gate maintained
  activation under gradient signal alone

---

## 4. Config Schema

New top-level section in the run config JSON:

```json
{
  "model": {
    "d": 256,
    "k": 4,
    "memory_rule": "delta",
    "memory_reset": "carry_forward",
    "b_theta": [-4.6, -5.6, -4.6, -4.6],
    "b_alpha": [3.0, 4.0, 4.5, 5.0],
    "theta_floor": [0.0, 0.0, 0.0, 0.0],
    "theta_ceil": [1.0, 1.0, 1.0, 1.0]
  },
  "gate_warmup": {
    "theta_floor_init":        [0.0, 0.0, 0.005, 0.001],
    "gate_warmup_decay_steps": 10000,
    "falsification_step":      20000,
    "l2_theta_threshold":      0.005,
    "l3_theta_threshold":      0.001
  },
  "training": {
    "lr": 3e-4,
    "steps": 25000
  }
}
```

### Field Definitions

| Field | Type | Description |
|---|---|---|
| `theta_floor_init` | `list[float]` | Per-level theta_floor at step 0 of Phase 2. Decays to 0 by `gate_warmup_decay_steps`. Length must equal k. |
| `gate_warmup_decay_steps` | `int` | Step at which theta_floor_init reaches 0.0. Phase 3 begins. |
| `falsification_step` | `int` | Step at which go/no-go thresholds are evaluated. Must be ≥ gate_warmup_decay_steps. |
| `l2_theta_threshold` | `float` | L2 gate must exceed this at `falsification_step`. Default: 0.005. |
| `l3_theta_threshold` | `float` | L3 gate must exceed this at `falsification_step`. Default: 0.001. |

The `gate_warmup` section is optional. If absent, no schedule is applied and
theta_floor falls back to the static value in `model.theta_floor`.

---

## 5. Falsification Thresholds

The diagnostic experiment is a falsification test, not an optimization target.
The hypothesis under test: **the gate initialization trap is the primary
cause of L2/L3 dormancy, and a targeted warmup is sufficient to escape it.**

### Go Gate (pod run authorized)

```text
L2 θ_mean > 0.005  AND  L3 θ_mean > 0.001  at step 20K
```

where θ_mean is the mean gate activation across all tokens at step 20K (or
the running mean over the final 100 eval steps).

Equivalently in b_theta space:
```text
b_theta^(L2) > softplus_inv(0.005) ≈ -5.3
b_theta^(L3) > softplus_inv(0.001) ≈ -6.9
```

### No-Go (pod run blocked)

If either threshold fails at step 20K, the pod run is blocked. Possible
diagnoses:

| Failure mode | Likely cause | Next action |
|---|---|---|
| L2 θ decays below init after scaffold removed | Gradient is negative — gate is being pushed closed | Check δL/δθ sign; inspect backward path |
| L3 θ holds at scaffold level but doesn't grow | Gate is not receiving gradient during Phase 3 | Check L3 fires are reaching the backward kernel |
| Both gates track scaffold exactly | theta_floor is binding — gate never trained freely | Reduce theta_floor_init; extend Phase 2 |
| L2 OK, L3 fails | Momentum starvation at 512-step period still dominant | Increase b_theta_init for L3 further; consider per-level AdamW LR |

---

## 6. Dataset Requirements

The warmup protocol is necessary but not sufficient. If the corpus lacks
genuine period-64 and period-512 structure, L2 and L3 cannot learn to
activate even with warmup support. The corpus must be validated before the
diagnostic run.

### Required Composition

- **Minimum size**: ≥ 500M tokens, contiguous (not shuffled)
- **Content type**: long-form books (chapter-level coherence) OR deeply
  nested code graphs (function → class → module hierarchy)
- **Forbidden**: short web documents, shuffled sentence pairs, C4-style
  cleaned web text (structure is destroyed by deduplication/filtering)

### Lag-MI Validation

Run ESTR (Excess Spectral Transfer Ratio) at lags [1, 8, 64, 512, 4096] on
a 10M-token sample:

```text
ESTR(lag) = MI(x_t, x_{t-lag}) / MI(x_t, x_{t-1})
```

**Pass conditions** (corpus is acceptable):

```text
ESTR(64)  / ESTR(4096) > 2.0×    (period-64 signal is genuine)
ESTR(512) / ESTR(4096) > 2.0×    (period-512 signal is genuine)
```

**Fail**: If either ratio ≤ 2.0×, the corpus does not have the multi-scale
structure that L2/L3 are designed to exploit. Dataset must be replaced before
the diagnostic run.

---

## 7. Code Smell Constraints

- **CS-10** (no train/eval): theta_floor schedule is applied at every step
  regardless of context (build, eval, checkpoint). No mode flag. The schedule
  is a function of step count only.

- **CS-39** (clamp learnable decay): theta_floor_init values are the Phase 2
  scaffold, not permanent constraints. They decay to 0 by design. After Phase 2,
  the permanent floor is `model.theta_floor` (which may be 0.0 or a small
  positive safety value). The two are distinct: scaffold is transient,
  safety floor is architectural.

- **CS-40** (opt-in AD): The theta_floor schedule update happens in loop.py
  before the step, outside the `with_tape()` context. It does not participate
  in the Wengert tape. Only the forward pass (which reads theta_floor via
  gpu_params) is tape-recorded.

- **CS-11** (no training loop in memory rule): The schedule logic lives in
  loop.py (Python orchestration), not in any Rust memory rule or CUDA kernel.
  The Rust layer only sees the current scalar theta_floor values; it has no
  knowledge of the schedule.

---

## 8. Relationship to Existing Infrastructure

| Component | Used by gate_warmup | Change required |
|---|---|---|
| `model.theta_floor` (config.py) | Phase 3 final value | None — already exists |
| `model.b_theta` (config.py) | Phase 1 seeding | Set per-level at run time |
| `gpu_forward.rs` clamp logic | Phase 2 scaffold enforcement | None — already wired |
| `loop.py` rebuild path | Schedule update | ~15 lines new code |
| `gpu_params` PyO3 binding | theta_floor setter | None — already exists |
| CS-39 theta_floor/ceil | Phase 3 safety rail | None |

The implementation is purely in `loop.py`. No Rust or CUDA changes are
needed. The gate_warmup config section is read at build start and the schedule
is applied step-by-step in the main loop.

---

## 9. Reference Config: `gate_warmup_diagnostic.json`

```json
{
  "_comment": "Gate activation warmup diagnostic. See specs/infrastructure/09_gate_warmup.md.",
  "model": {
    "d": 256,
    "k": 4,
    "num_heads": 4,
    "seq_len": 512,
    "vocab_size": 32000,
    "memory_rule": "delta",
    "memory_reset": "carry_forward",
    "composition": "mag",
    "b_theta": [-4.6, -5.6, -4.6, -4.6],
    "b_alpha": [3.0, 4.0, 4.5, 5.0],
    "theta_floor": [0.0, 0.0, 0.0, 0.0],
    "theta_ceil":  [1.0, 1.0, 1.0, 1.0],
    "m_norm_max": [100.0, 100.0, 100.0, 100.0]
  },
  "gate_warmup": {
    "theta_floor_init":        [0.0, 0.0, 0.005, 0.001],
    "gate_warmup_decay_steps": 10000,
    "falsification_step":      20000,
    "l2_theta_threshold":      0.005,
    "l3_theta_threshold":      0.001
  },
  "training": {
    "lr": 3e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "steps": 25000,
    "eval_interval": 250,
    "checkpoint_interval": 5000
  },
  "data": {
    "dataset": "books_or_code_validated",
    "min_tokens": 500000000
  }
}
```
