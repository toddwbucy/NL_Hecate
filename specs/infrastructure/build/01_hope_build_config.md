# HOPE Model Build Configuration

<!-- HADES: hope_equations/eq-071-arch-variant2 (§6 Eq 71); hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79); hope_equations/eq-085-phase3-optimization (§8.1 Eq 85) -->
```text
CONTRACT
  Purpose:    Enable building a HOPE model from Python. The Rust core has all
              required primitives (Titans LMM, MAG, CMS k=4, DGD, self-ref
              projections, self-generated values, chunkwise self-ref, tape AD).
              The Python tier (build.py) lacks the configuration plumbing to
              activate them. This spec defines: (1) new BuildConfig and PyO3
              MAGConfig fields, (2) config JSON schema for HOPE models,
              (3) build loop changes for HOPE-specific orchestration.
  Expects:    Existing build.py with BuildConfig, AdamW, cosine_lr.
              Existing PyO3 MAGConfig binding (python/src/lib.rs).
              Rust core with: ProjectionKind::Adaptive, self_generated_values,
              self_ref_chunk_size, MomentumKind::Ema.
  Guarantees: A HOPE model can be built with:
                python build.py --config configs/hope_60m.json --gpu
              All new fields default to Phase 1 behavior (static projections,
              no self-generated values, self_ref_chunk_size=1, no momentum).
              Existing configs produce identical behavior (backward compat).
  Cost:       No runtime cost — purely configuration plumbing. The Rust
              primitives already exist; this spec connects Python to them.
  Trade-off:  Exposing all HOPE knobs in build.py adds config complexity.
              Mitigated by sensible defaults and a reference hope_60m.json.
  Position:   specs/infrastructure/build/01_hope_build_config.md
              Cross-ref: specs/algorithms/self_referential/00_interface.md
                         specs/algorithms/composition_patterns/04_hope.md
                         specs/algorithms/optimization_machinery/08_adamw_outer.md
  Source:     HOPE (2512.24695) §9.2 (experimental setup);
              §8.1 Eqs 79, 84-85 (Phase 2/3 configuration)
```

## Current Gaps

Three areas where Python cannot reach existing Rust primitives:

```text
-- Gap 1: PyO3 MAGConfig hardcodes Phase 1
--   python/src/lib.rs line 444:
--     projection_kind: ProjectionKind::Static,
--     self_generated_values: false,
--   These should be configurable parameters, not hardcoded.

-- Gap 2: BuildConfig missing HOPE fields
--   python/build.py BuildConfig has no fields for:
--     projection_kind, self_generated_values, self_ref_chunk_size,
--     momentum_kind, momentum_d_hidden
--   build.py line 719-732 MAGConfig construction only passes basic fields.

-- Gap 3: No reference HOPE config
--   No configs/hope_Nm.json file exists to demonstrate the full configuration.
```

## PyO3 MAGConfig Extension

<!-- HADES: Derived from hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79, projection_kind); hope_equations/eq-084-phase3-self-generated-values (§8.1 Eq 84) -->
```text
-- Add to MAGConfig.__init__() parameters:
--   projection_kind: str = "static"     -- "static" or "adaptive"
--   self_generated_values: bool = False  -- Phase 3 self-modifying
--   momentum_kind: str = "none"         -- "none" or "ema"
--   momentum_d_hidden: int = 0          -- Hidden dim for momentum MLP (0 = d*d matrix)
--   self_ref_chunk_size: int = 1        -- Already exposed (GAP-N, PR #117)

-- String → enum mapping:
--   "static"   → ProjectionKind::Static  (Phase 1: standard projections)
--   "adaptive" → ProjectionKind::Adaptive (Phase 2+: memory-based projections)
--   "none"     → MomentumKind::None
--   "ema"      → MomentumKind::Ema

-- Validation:
--   self_generated_values requires projection_kind == "adaptive"
--     (Phase 3 needs Phase 2 infrastructure)
--   self_ref_chunk_size > 1 requires projection_kind == "adaptive"
--     (chunkwise self-ref is meaningless without adaptive projections)
--   momentum_d_hidden > 0 requires momentum_kind == "ema"

-- Defaults preserve backward compatibility:
--   projection_kind="static", self_generated_values=false → Phase 1
--   All existing configs and tests produce identical behavior.
```

## BuildConfig Extension

```text
-- Add to BuildConfig (python/build.py):

-- Self-referential projections (Phase 2/3)
projection_kind: str = "static"            -- "static" or "adaptive"
self_generated_values: bool = False         -- Phase 3 self-modifying memory
self_ref_chunk_size: int = 1               -- chunkwise self-ref (1 = sequential)

-- Momentum
momentum_kind: str = "none"                -- "none" or "ema"
momentum_d_hidden: int = 0                 -- momentum MLP hidden dim

-- All defaults match Phase 1 (existing behavior).
```

## Config JSON Schema

```text
-- Reference: configs/hope_60m.json
-- Layout follows existing config convention (model/build/data sections).

{
  "model": {
    "d_model": 512,
    "num_heads": 8,
    "seq_len": 512,
    "window_size": 512,
    "vocab_size": 32000,
    "k": 4,
    "chunk_sizes": [1, 8, 64, 512],
    "memory_rule": "titans",
    "composition": "mag",

    // HOPE-specific fields (new)
    "projection_kind": "adaptive",
    "self_generated_values": true,
    "self_ref_chunk_size": 4,
    "momentum_kind": "ema",
    "momentum_d_hidden": 0
  },
  "build": {
    "lr": 0.0004,
    "steps": 100000,
    "optimizer": "adamw_gpu",
    "warmup_steps": 200,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.999,
    "max_grad_norm": 1.0,
    "save_every": 5000,
    "log_every": 10,
    "eval_every": 1000,
    "eval_max_chunks": 100
  },
  "data": {
    "path": "data/sharegpt",
    "format": "sharegpt"
  }
}

-- Key design decisions:
--   memory_rule: "titans" (Titans LMM = matrix memory + L2 + L2 decay + GD+momentum)
--   composition: "mag" (MAG = memory gates attention)
--   projection_kind: "adaptive" (Phase 2 adaptive projections)
--   self_generated_values: true (Phase 3 self-modifying)
--   self_ref_chunk_size: 4 (chunkwise training for 4× speedup in grad computation)
--   k: 4 with chunk_sizes [1,8,64,512] (standard CMS frequencies)
--   optimizer: "adamw_gpu" (fused GPU AdamW, frequency-aware)
--
-- HOPE §9.2 experimental values:
--   lr: 4e-4 (60M), 3e-4 (90M+)
--   warmup: 200 steps (60M), 500 steps (larger)
--   weight_decay: 0.1, beta1: 0.9, beta2: 0.999
--   grad_clip: 1.0
```

## Build Loop Changes

```text
-- build.py MAGConfig construction (line ~719-732) currently:
--   cfg = nl_hecate.MAGConfig(
--       d_model, num_heads, head_dim, seq_len, window_size, vocab_size,
--       memory_enabled=True, k, chunk_sizes, memory_rule, composition,
--       checkpoint_interval=checkpoint_interval,
--   )
--
-- After this spec:
--   cfg = nl_hecate.MAGConfig(
--       d_model, num_heads, head_dim, seq_len, window_size, vocab_size,
--       memory_enabled=True, k, chunk_sizes, memory_rule, composition,
--       checkpoint_interval=checkpoint_interval,
--       projection_kind=bcfg.projection_kind,
--       self_generated_values=bcfg.self_generated_values,
--       self_ref_chunk_size=bcfg.self_ref_chunk_size,
--       momentum_kind=bcfg.momentum_kind,
--       momentum_d_hidden=bcfg.momentum_d_hidden,
--   )
--
-- No changes to the build loop itself. The forward path (cms_forward)
-- dispatches based on MAGConfig fields internally. The Python orchestrator
-- does not need to know about Phase 2 vs Phase 3 — it passes config, gets
-- loss + gradients. This is CS-18 (forward pass IS the API).
```

## CLI Overrides

```text
-- Add to argparse in build.py main():
--   --projection_kind   str   "static" or "adaptive"
--   --self_generated_values   flag (store_true)
--   --self_ref_chunk_size  int   chunkwise self-ref chunk size
--   --momentum_kind     str   "none" or "ema"
--   --momentum_d_hidden int   momentum MLP hidden dim
--
-- Add to apply_cli() mapping dict.
-- These override config file values, same as existing CLI args.
```

## Parameter Count Impact

```text
-- HOPE adds parameters compared to Phase 1:
--
-- Phase 1 (static projections, current):
--   Per level: w_k_mem[d²] + w_v_mem[d²] + w_q_mem[d²] + gates = ~3d² + small
--   Total: SWA params + k * (~3d² + small)
--
-- Phase 2+ (adaptive projections, after task_3c4e6e wiring):
--   Per level: existing + m_k_init[d²] + m_v_init[d²] + m_q_init[d²]
--              + m_eta_init[d²] + m_alpha_init[d²] + m_mem_init[d²] = +6d²
--   For d=512, k=4: +6 * 262144 * 4 = +6,291,456 params (+6.3M)
--
-- Momentum (EMA):
--   Per level: momentum S state = d² per memory × 6 memories = +6d²
--   S is inner_loop_state (not serialized), so no impact on checkpoint size.
--   But momentum adds one EMA buffer per memory per level during forward.
--
-- Model sizing example (d=512, heads=8, k=4, vocab=32K):
--   SWA:          ~33.6M (embed + Q/K/V/O + unembed)
--   Levels (Ph1): ~3.2M  (3 × d² × 4 + gates)
--   Levels (Ph2): ~9.5M  (9 × d² × 4 + gates — 6 init states added)
--   Total (Ph1):  ~36.8M
--   Total (Ph2):  ~43.1M
--   Overhead:     ~17% more params for full self-referential capabilities
```

## Validation Criteria

```text
-- A HOPE model build is valid when:
--
-- 1. Config loads without error:
--    BuildConfig.from_file("configs/hope_60m.json") succeeds
--
-- 2. MAGConfig construction includes all HOPE fields:
--    cfg.projection_kind == "adaptive"
--    cfg.self_generated_values == true
--    cfg.self_ref_chunk_size == 4
--
-- 3. Forward + backward produce finite gradients:
--    loss is not NaN/Inf after 10 steps
--
-- 4. Loss decreases over 500 steps:
--    avg(last_50_losses) < avg(first_50_losses)
--
-- 5. Checkpoint roundtrip preserves config:
--    Save → load → cfg matches original (including HOPE fields)
--
-- 6. Backward compatibility:
--    Existing configs (toy_60m.json, etc.) produce identical behavior
--    No new fields required in existing configs
```

## Print Banner Extension

```text
-- Add to build.py print section (~line 740):
--   if bcfg.projection_kind == "adaptive":
--       print(f"  SelfRef: projection={bcfg.projection_kind}, "
--             f"self_gen={bcfg.self_generated_values}, "
--             f"chunk_size={bcfg.self_ref_chunk_size}")
--   if bcfg.momentum_kind != "none":
--       print(f"  Momentum: kind={bcfg.momentum_kind}, "
--             f"d_hidden={bcfg.momentum_d_hidden}")
--
-- Also log HOPE fields in the build_start JSONL event.
```

## Implementation Sequence

```text
1. Add new fields to BuildConfig with defaults → validate existing configs still load
2. Add projection_kind, self_generated_values, momentum_kind, momentum_d_hidden
   to PyO3 MAGConfig.__init__() with string→enum mapping → compile
3. Wire new BuildConfig fields into MAGConfig construction in build.py → test with
   existing configs (Phase 1 behavior preserved)
4. Add CLI args + apply_cli mapping → test with --projection_kind adaptive
5. Create configs/hope_60m.json reference config
6. Print banner + JSONL log extension
7. Run 100-step smoke test with hope_60m.json → loss decreases, no NaN
```

## Files Modified

```text
| File | Change |
|------|--------|
| python/src/lib.rs | Add projection_kind, self_generated_values, momentum_kind, momentum_d_hidden params to MAGConfig.__init__() |
| python/build.py | Add 5 fields to BuildConfig, wire into MAGConfig construction, add CLI args, update banner |
| python/configs/hope_60m.json | CREATE: reference HOPE config |
```

## Axiom Compliance

- **CS-18** (forward pass IS the API): No build loop changes — config fields control Rust dispatch.
- **CS-10** (no mode flag): Phase 2/3 behavior determined by config, not runtime mode.
- **NL IS #9** (principled not ad hoc): HOPE config directly mirrors paper experimental setup (§9.2).
