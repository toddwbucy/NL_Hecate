# SelfRefParamGrads Outer-Loop Wiring

<!-- HADES: hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79); hope_equations/eq-085-phase3-optimization (§8.1 Eq 85); hecate_specs/self_ref_param_wiring -->
```text
CONTRACT
  Purpose:    Wire SelfRefParamGrads (dL/dM_{k,0}..dL/dM_{mem,0}) from the
              backward pass into MAGParams so the outer-loop optimizer can
              update the 6 initial memory states M_{square,0}. Currently these
              gradients are computed by self_ref_step_backward() and
              chunkwise_self_ref_step_backward() but discarded with _sr_grads
              in mag.rs (tagged TODO(PR-4)). This spec defines the storage
              layout, gradient application, CMS frequency gating, and
              checkpoint serialization for these outer-loop parameters.
  Expects:    SelfRefParamGrads from backward pass (6 × [d*d] gradient vectors).
              MAGParams with per-level storage for initial memory states.
              Pulse from Conductor indicating which CMS levels are active.
              Outer-loop optimizer (SGD or frequency-aware AdamW).
  Guarantees: All 6 initial states are updated by the outer loop when the
              corresponding CMS level fires. Frozen levels accumulate gradients
              in the error buffer without applying them. Checkpoint/restore
              round-trips all initial states. At C=1, gradient application
              matches manual finite-difference checks. Phase 1 (static
              projections) is unaffected — self-ref params only exist when
              projection_kind == Adaptive.
  Cost:       6 × d² extra floats per CMS level for initial states.
              6 × d² extra floats per CMS level for gradient accumulators.
              Zero cost when projection_kind == Static (no allocation).
  Trade-off:  Storing initial states in MAGParams makes them visible to the
              optimizer and checkpoint system. The alternative (storing them
              separately) would require custom plumbing everywhere. The cost
              is 6*d² per level — for d=128, k=4 levels, init states alone are
              ~1.5MB (6 × 128² × 4 × 4B) and ~3MB including gradient accumulators.
              Comparable to the 3×d² per-level projection matrices (also ~384KB/level).
  Position:   specs/algorithms/self_referential/04_self_ref_param_wiring.md
              Parent: 00_interface.md (self-referential projections)
              Cross-ref: specs/algorithms/optimization_machinery/08_adamw_outer.md
  Source:     HOPE (2512.24695) §8.1 Eqs 79, 82, 85 (M_{square,0} as outer_loop_param);
              00_interface.md State Lifetime Analysis (lines 352-370)
```

## Background: The Missing Gradient Path

Self-referential projections (Phase 2/3) replace static weight matrices W_k, W_v,
W_q with adaptive memory modules M_k, M_v, M_q, M_eta, M_alpha, M_mem. Each
memory module has an initial state M_{square,0} that serves the same role as the
static projection matrix: at t=0, `M_k(x) = M_{k,0} @ x` is identical to
`W_k @ x`. The inner loop (DGD) updates M per-token during forward. The outer
loop must update M_{square,0} across sequences.

Currently, the backward pass computes `SelfRefParamGrads` (6 gradient matrices
dL/dM_{k,0}..dL/dM_{mem,0}) but mag.rs discards them:

```text
-- mag.rs line 298 (current):
let (d_emb, _sr_grads) = self_ref_step_backward(sr_cache, &d_y, ...);
// TODO(PR-4): _sr_grads holds dL/dM_{k,0}..dL/dM_{mem,0}
// Currently dropped because MAGParams doesn't have fields for them yet.
```

This spec defines how to stop dropping them.

## Storage: MemoryLevelParams Extension

<!-- HADES: Derived from hope_equations/eq-082-phase2-read (§8.1 Eq 82, M_{square,0} as outer_loop_param) -->
```text
-- Add to MemoryLevelParams (only allocated when projection_kind == Adaptive):

STRUCT MemoryLevelParams {
    -- ... existing fields (w_k_mem, w_v_mem, w_q_mem, gates, etc.) ...

    -- Self-ref initial states: 6 × [d * d] f32 (UNCONDITIONALLY fp32)
    -- Zero-length when projection_kind == Static.
    m_k_init:     Vec<f32>,   -- [d * d] initial state for key projection memory
    m_v_init:     Vec<f32>,   -- [d * d] initial state for value projection memory
    m_q_init:     Vec<f32>,   -- [d * d] initial state for query projection memory
    m_eta_init:   Vec<f32>,   -- [d * d] initial state for learning rate memory
    m_alpha_init: Vec<f32>,   -- [d * d] initial state for retention memory
    m_mem_init:   Vec<f32>,   -- [d * d] initial state for main projection memory
}

-- Initialization (from self_ref_init_state() in self_ref.rs):
--   m_k_init = (I + small random) / d    -- slightly-perturbed identity
--   Same for all 6 memories.
--   This matches the existing SelfRefState initialization logic.
--
-- NOT bf16: These are outer_loop_params but stored as fp32. The inner loop
-- reads M_{square,0} into fp32 SelfRefState at forward start. bf16 storage
-- would require master/stored split like w_k_mem, but the values are NOT
-- used as matrix multiplication operands in SWA attention — they seed the
-- inner-loop DGD recurrence which is unconditionally fp32 (CS-44).
```

## Gradient Application

<!-- HADES: Derived from hope_equations/eq-085-phase3-optimization (§8.1 Eq 85, outer-loop gradient for M_{square,0}) -->
```text
-- In mag_backward (both single-level and CMS paths):
-- Replace:
--   let (d_emb, _sr_grads) = self_ref_step_backward(...);
--   (MemoryLevelParams::zeros_like(d), d_emb)
-- With:
--   let (d_emb, sr_grads) = self_ref_step_backward(...);
--   let mut level_grads = MemoryLevelParams::zeros_like(d);
--   level_grads.m_k_init     = sr_grads.d_m_k;
--   level_grads.m_v_init     = sr_grads.d_m_v;
--   level_grads.m_q_init     = sr_grads.d_m_q;
--   level_grads.m_eta_init   = sr_grads.d_m_eta;
--   level_grads.m_alpha_init = sr_grads.d_m_alpha;
--   level_grads.m_mem_init   = sr_grads.d_m_mem;
--   (level_grads, d_emb)
--
-- Same pattern for ChunkwiseSelfRef backward.
--
-- MemoryLevelParams::apply_weight_gradients already handles all fields.
-- Adding 6 new Vec<f32> fields → 6 new step() calls in apply_weight_gradients.
-- When projection_kind == Static, all init fields are empty → step() is a no-op.
```

## CMS Frequency Gating

<!-- HADES: hope_equations/eq-071-arch-variant2 (HOPE 2512.24695 §6 Eq 71, per-level gating); nl_code_smells/CS-27; nl_code_smells/CS-28 -->
```text
-- Self-ref initial states are per-level outer_loop_params, subject to
-- the same frequency gating as w_k_mem, w_v_mem, etc. (CS-27/28).

-- Active level: outer-loop gradient is applied normally.
--   level.apply_weight_gradients(&level_grads, lr)
--   → m_k_init[i] -= lr * level_grads.m_k_init[i] for all i

-- Frozen level: gradient accumulates in ErrorBuffer.
--   error_buffers[level].accumulate_sr_grads(sr_grads)
--   On next firing: accumulated gradient applied as a batch.

-- ErrorBuffer extension:
--   Currently ErrorBuffer stores a flat Vec<f32> for level-parameter gradients.
--   The self-ref init gradients have the same shape as level params (they are
--   stored IN MemoryLevelParams). The existing error buffer accumulation
--   operates on MemoryLevelParams::get_flat_weights() / set_flat_weights().
--   Since the new fields are part of MemoryLevelParams, they are automatically
--   included in the flat weight serialization. No separate error buffer needed.

-- AdamW interaction:
--   When using Python-side AdamW (build.py AdamW class), the flat weight vector
--   from params.get_flat_weights() includes the self-ref init states. AdamW
--   moment buffers cover them automatically (num_params increases by 6*d²*k).
--   When using GPU AdamW (GpuAdamWState), the MomentLevel must be extended
--   to include moment buffers for the 6 init fields.
```

## Forward Path: Seeding SelfRefState from MAGParams

<!-- HADES: Derived from hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79, M_{square,0} initialization) -->
```text
-- Currently, self_ref_step() creates SelfRefState with a fixed initialization:
--   SelfRefState::new(d) → identity-like matrices for all 6 memories.
--
-- With this wiring:
--   self_ref_step() receives M_{square,0} from MemoryLevelParams::m_*_init.
--   SelfRefState::from_initial_states(m_k_init, m_v_init, ...) copies these
--   values into the mutable inner-loop state.
--
-- FUNCTION: seed_self_ref_state(level_params: &MemoryLevelParams, d: usize) -> SelfRefState
--   state.m_k  = level_params.m_k_init.clone()   -- [d*d]
--   state.m_v  = level_params.m_v_init.clone()
--   state.m_q  = level_params.m_q_init.clone()
--   state.m_eta   = level_params.m_eta_init.clone()
--   state.m_alpha = level_params.m_alpha_init.clone()
--   state.m_mem   = level_params.m_mem_init.clone()
--   RETURN state
--
-- If m_k_init is empty (projection_kind == Static), fall back to
-- SelfRefState::new(d) (the existing identity-like initialization).
-- This preserves backward compatibility.
--
-- Context memory interaction:
--   In serving mode with ContextStream, M_{square,t} at chunk boundary
--   becomes the seed for the next chunk — NOT M_{square,0}. The initial
--   states from MAGParams are only used at the START of a sequence (or
--   after a document boundary reset). The context_memory mechanism
--   already handles this for the main memory M; the same pattern
--   applies to projection memories.
```

## Checkpoint Serialization

```text
-- MemoryLevelParams is already Serialize + Deserialize.
-- Adding 6 new Vec<f32> fields automatically includes them in JSON
-- checkpoint format. Fields serialize as arrays of f32.
--
-- Backward compatibility:
--   Old checkpoints (without m_*_init fields) deserialize with
--   #[serde(default)] → empty Vec<f32>. On forward, empty init vectors
--   trigger SelfRefState::new(d) fallback. Existing checkpoints work.
--
-- New checkpoints (with m_*_init fields) include the learned initial
-- states. On load, they seed SelfRefState correctly. This enables
-- checkpoint → resume with preserved projection memory initialization.
```

## get_flat_weights / set_flat_weights

```text
-- MemoryLevelParams::{get_flat_weights, set_flat_weights, num_params} must
-- include the 6 new fields. The existing implementation appends all Vec<f32>
-- and Bf16Storage fields in declaration order.
--
-- Ordering: append after existing fields (w_q_conv, b_q_conv), before any
-- future fields. The flat layout is:
--   [w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta,
--    w_eta, b_eta, w_omega, w_freq, b_freq, w_k_conv, b_k_conv,
--    w_q_conv, b_q_conv,
--    m_k_init, m_v_init, m_q_init, m_eta_init, m_alpha_init, m_mem_init]
--
-- This preserves backward compatibility: old checkpoints that load via
-- flat weights have fewer elements; the new fields initialize to their
-- default (identity-like) values on first use.
```

## Interaction with Python build.py

```text
-- The Python-side AdamW creates moment buffers sized by params.num_params().
-- After this change, num_params() increases by 6 * d² * k (for k levels
-- with projection_kind == Adaptive). The Python AdamW automatically covers
-- the new parameters — no Python-side changes needed.
--
-- The flat weight vector exposed via PyO3 get_flat_weights() includes the
-- init states. Python can inspect / modify them like any other parameter.
--
-- For the GPU path (adamw_gpu), the MomentLevel struct in gpu_optimizer.rs
-- must be extended to include moment buffers for the 6 init fields.
-- This is a straightforward extension of the existing pattern.
```

## Tests

```text
| Test | What It Validates |
|------|-------------------|
| test_sr_grads_reach_params          | After backward, level_grads.m_k_init is nonzero (not dropped) |
| test_sr_grads_match_fd              | FD check: perturb m_k_init → loss delta matches analytical grad |
| test_outer_loop_updates_init_states | After apply_weight_gradients, m_k_init has changed |
| test_static_projection_no_init      | When projection_kind==Static, m_*_init fields are empty |
| test_checkpoint_roundtrip_with_init | Save/load preserves m_*_init values exactly |
| test_old_checkpoint_compat          | Old checkpoint (no m_*_init) loads with empty defaults |
| test_cms_frozen_level_accumulates   | Frozen level: sr_grads accumulate in error buffer |
| test_cms_active_level_applies       | Active level: m_*_init changes after gradient step |
| test_num_params_includes_init       | num_params() counts the 6*d² per adaptive level |
| test_flat_weights_includes_init     | get_flat_weights() length matches num_params() |
```

## Implementation Sequence

```text
1. Add 6 m_*_init fields to MemoryLevelParams with #[serde(default)]
   Update: init(), zeros_like(), num_params(), apply_weight_gradients(),
           get_flat_weights(), set_flat_weights()
   → Compile (all existing tests pass, new fields default to empty)

2. Add SelfRefState::from_initial_states() constructor
   Wire: mag_forward() seeds SelfRefState from level_params.m_*_init
   → Existing self-ref tests still pass (behavior unchanged when init is identity)

3. Wire sr_grads into level_grads in mag_backward() (4 sites)
   Remove _sr_grads → sr_grads, populate level_grads.m_*_init fields
   → FD gradient check passes

4. Add tests (checkpoint roundtrip, backward nonzero, FD check)
   → Full test suite passes
```

## Axiom Compliance

- **NL IS #4** (compressing context): The initial states compress the outer-loop's learning about optimal projection initialization. Each build step refines M_{square,0} to better seed in-context adaptation.
- **CS-27/28** (frequency-aware optimizer): Self-ref init gradients are gated by CMS frequency. Frozen levels accumulate in error buffers, not applied directly.
- **CS-44** (fp32 inner loop): Init states are fp32, matching the inner-loop DGD recurrence precision.
