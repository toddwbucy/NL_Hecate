# TNT Periodic Reset: `memory_reset` Config Axis

```text
CONTRACT
  Purpose:    Define the memory_reset configuration axis that controls whether CMS
              level memory state M carries forward across fire boundaries (carry_forward)
              or resets to a shared learnable initial state M_init at each fire boundary
              (periodic). The periodic mode is the TNT (Tokenwise iNner-loop Training)
              approximation from 2511.07343, adapted to the CMS multi-level framework.

  Expects:    - CMS k levels defined with fire periods [P_0, P_1, ..., P_{k-1}]
                (e.g. [1, 8, 64, 512] for k=4).
              - Conductor/Pulse infrastructure delivering per-step active_levels mask
                (specs/infrastructure/scheduling/).
              - GpuContextState.memory: k buffers of shape [d×d], one per level.
              - GpuContextState.momentum: k buffers of shape [d×d], one per level
                (Titans momentum S; zero for rules without momentum).
              - A learnable M_init per level: k buffers of shape [d×d], registered as
                outer_loop_param, initialized to zeros, updated by outer-loop optimizer.
              - memory_reset is a top-level field in the model config section:
                  "memory_reset": "carry_forward"   (default, current behavior)
                  "memory_reset": "periodic"         (TNT mode)

  Guarantees: - carry_forward: M at step t+1 = M at last token of step t (unchanged from
                current behavior). No new parameters. Full backward compatibility.
              - periodic: At each fire boundary for level k (i.e. every P_k steps),
                M_k is reset to M_init_k (learnable) and S_k is reset to 0 (non-learnable).
                Between fire boundaries, chunkwise update proceeds normally with the
                chunk-start state (eq-003, eq-006).
              - M_init_k is an outer_loop_param: persists across build steps, serialized
                in checkpoints, updated by AdamW (weight_decay applies).
              - Forward pass output is deterministic given M_init_k and the input sequence.
                The sequential dependency of M on prior tokens is broken at each fire
                boundary — shards within a level are independent given M_init.
              - Backward pass: gradient flows through M_init_k from all shards in which
                it was used as the start state. Gradients are summed (not averaged) across
                shards, consistent with how chunkwise gradients accumulate within a shard.
              - The carry_forward and periodic modes are mutually exclusive per build.
                No mixed-mode within a single run.

  Cost:       - carry_forward: zero additional parameters, zero additional compute.
              - periodic: k additional d×d fp32 buffers (outer_loop_param). At d=512,
                k=4: 4 × 512 × 512 × 4 bytes = 4 MB. Negligible.
              - Backward cost: M_init gradient accumulation adds one atomicAdd per shard
                per level. At S_L / P_k shards per build: negligible vs per-token grad cost.

  Trade-off:  carry_forward: Full sequential M dependency across the entire build. Maximum
              information propagation but minimum parallelism — shards cannot be computed
              independently. L3 (period=512) receives only ~48 updates in 25K steps;
              initialization trap risk is high.
              periodic: Shards are independent within each level. Enables future parallel
              shard dispatch (Task-level parallelism). Breaks the sequential M dependency
              at shard boundaries — long-range context beyond S_L tokens is not encoded
              in M. M_init_k becomes the shared prior across all shards; it accumulates
              gradient signal from every shard at every fire boundary, giving L3 ~48× more
              effective gradient signal per outer-loop step than carry_forward.

  Position:   specs/infrastructure/08_tnt_periodic_reset.md
  Source:     TNT (2511.07343) §3.2 Local Memory with Periodic Reset
                HADES: tnt_equations/eq-006-local-memory-update      (reset rule)
                       tnt_equations/eq-014-n-local-memories-update  (N local memories)
                       tnt_equations/eq-003-chunkwise-compression    (within-shard update)
              TNT (2511.07343) §3.1 Global Memory
                HADES: tnt_equations/eq-005-global-memory-update     (global memory)
  Related:    specs/infrastructure/05_ablation_study.md (A/B/C/D baselines, carry_forward)
              specs/infrastructure/scheduling/ (Conductor/Pulse, fire boundary detection)
              nl_ethnographic_notes/two_phase_nlm_training_curriculum (curriculum context)
```

---

## Motivation

In the current CMS carry_forward mode, each level's memory M accumulates context
across the entire build sequence. Level 3 (period=512) fires only once every 512
steps — approximately 48 times in a 25K-step build. With gate bias `b_theta=-7.6`
(softplus ≈ 0.0005), L3 is nearly closed by design and receives almost no effective
gradient signal. This is the **initialization trap**: too few updates to escape the
initial prior, not enough signal to open the gate.

The TNT periodic reset breaks the sequential dependency at shard boundaries. Each shard
starts from the same learnable `M_init_k`. Every fire of level k contributes gradient
signal directly to `M_init_k` through the outer-loop optimizer. At 48 fires, L3 receives
48 independent gradient contributions per build — vs. the single sequential chain
in carry_forward mode. The effective gradient signal to L3 initialization increases
dramatically without changing the architecture or adding data.

---

## Definitions

**Shard**: The token span between two consecutive fires of level k. For level k with
fire period P_k and sequence length seq_len, a shard spans P_k × seq_len tokens.

**Shard boundary**: The step at which level k fires. Identified by the Pulse struct:
`pulse.active_levels[k] == true`.

**M_init_k**: Learnable d×d fp32 matrix, one per CMS level. Outer-loop parameter
(outer_loop_param lifetime). Initialized to zeros at build start. Updated by the
outer-loop optimizer (AdamW) via gradients accumulated from all shards.

**S reset**: The Titans momentum buffer S_k (d×d fp32, inner_loop_state) is reset to
zero at each shard boundary. Non-learnable — no gradient flows through S.

---

## Reset Rule (TNT eq-006 adapted for CMS)

For CMS level k with fire period P_k, at global step t:

```text
if pulse.active_levels[k]:                          // shard boundary
    M_k  ← M_init_k                                // reset to learnable prior
    S_k  ← 0                                       // reset momentum (non-learnable)
else:
    // chunkwise update from chunk-start state (within shard, unchanged)
    M_k  ← M_k - sum_{tau=chunk_start}^{t} eta_tau * grad_M L(f(M_chunk_start_k, k_tau), v_tau)
```

This matches TNT eq-006 exactly:

```
W_t ← W_init   if  0 ≡ t (mod S_L)
W_t ← W_{t-1} - sum_{tau=xi(t,C_L)}^{t} eta_tau grad_W L(f(W_{xi(t,C_L)}, k_tau), v_tau)   otherwise
```

where `S_L = P_k` (shard length in steps) and `C_L = 1` (chunk size = one step in
our current per-step chunkwise formulation).

**Level-0 special case**: P_0 = 1, so L0 fires every step. In periodic mode, L0 resets
M_0 to M_init_0 at every step — equivalent to L0 always starting from M_init_0. This
degrades L0 to a stateless projection (no memory accumulation within a shard). For
ablation purposes, periodic reset on L0 is valid but expected to hurt performance.
A future spec may restrict periodic reset to levels k ≥ 1 only.

---

## N-Local-Memories Mapping (TNT eq-014)

TNT's generalized formulation supports N local memories, each with its own shard length
S_{L_i} and chunk size C_{L_i}. This maps directly onto CMS k=4 levels:

```
Level 0: S_{L_0} = P_0 = 1,   C_{L_0} = 1,  M_init_0 ∈ R^{d×d}
Level 1: S_{L_1} = P_1 = 8,   C_{L_1} = 1,  M_init_1 ∈ R^{d×d}
Level 2: S_{L_2} = P_2 = 64,  C_{L_2} = 1,  M_init_2 ∈ R^{d×d}
Level 3: S_{L_3} = P_3 = 512, C_{L_3} = 1,  M_init_3 ∈ R^{d×d}
```

Each level is an independent local memory shard system. The Conductor/Pulse already
delivers the `active_levels` mask identifying which levels fire at each step —
shard boundary detection requires no additional infrastructure.

---

## Config Schema

```json
{
  "model": {
    "memory_reset": "carry_forward"
  }
}
```

Valid values:
- `"carry_forward"` (default): current behavior, M carries across all steps
- `"periodic"`: TNT-style, M resets to M_init_k at each fire boundary for level k

The field is optional. If absent, defaults to `"carry_forward"` (backward compatible).

---

## Parameter Lifecycle

| Parameter | Lifetime | Serialized | Optimizer |
|-----------|----------|------------|-----------|
| `M_init_k` (k=0..K-1) | outer_loop_param | yes | AdamW (weight_decay applies) |
| `M_k` (context) | context_memory | yes (checkpoint) | — (inner-loop state) |
| `S_k` (momentum) | inner_loop_state | no | — (reset at shard boundary) |

`M_init_k` is NOT the same as the current carry-forward `M_k`. It is a separate d×d
buffer per level. At shard boundary, `M_k ← M_init_k` (copy, not alias).

---

## Gradient Flow

During backward pass for a shard starting at step t_0:

```text
d_M_init_k += d_M_k_at_t0    // gradient at shard start = gradient w.r.t. M_init_k
```

Since `M_init_k` is shared across all shards, gradients from all S fires of level k
accumulate into `d_M_init_k` over the build. The outer-loop optimizer then applies one
AdamW update to `M_init_k` using the accumulated gradient. This is identical in
structure to how W_K and W_V gradients accumulate across all tokens in the current
carry_forward implementation.

---

## Ablation Comparison

The TNT ablation series (tasks task_d29203, task_b91493, task_2e43ec) runs
carry_forward and periodic modes under identical conditions:

| Config | memory_reset | Expected effect |
|--------|-------------|-----------------|
| ABLATION-B     | carry_forward | Baseline k=1 Titans |
| ABLATION-B-TNT | periodic      | L0 reset each step — likely hurt |
| ABLATION-C     | carry_forward | Baseline k=4 CMS |
| ABLATION-C-TNT | periodic      | L2/L3 get more gradient signal |
| ABLATION-D     | carry_forward | Baseline k=4 CMS + DGD |
| ABLATION-D-TNT | periodic      | Full HOPE with TNT reset |

Primary question: does periodic reset help L2/L3 escape the initialization trap?
If ppl_C-TNT < ppl_C, periodic reset is the lever. If not, the curriculum
(two-phase training, task_01a831) is the next experiment.

---

## Code Smell Constraints

- **CS-10** (no model.train/eval): `memory_reset` mode does not change forward code
  path. The same `cms_forward()` runs regardless of `memory_reset` value. Mode is
  structural, not behavioral.
- **CS-11** (no training loop in memory rule): The shard boundary detection and reset
  logic lives in the Python-tier orchestration loop (loop.py), not inside the memory
  rule or Rust forward pass. The Rust layer receives `m_initial` as an input buffer
  — whether that buffer contains M_init or the previous M is decided in Python.
- **CS-32** (observe-then-advance): The reset happens at the shard boundary, AFTER
  the previous shard's final M is observed and before the new shard's first advance.
  Correct phasing: observe (read final M), reset (M ← M_init), advance (next shard).
- **CS-39** (clamp learnable decay): M_init_k is bounded in magnitude by the existing
  m_norm_clamp applied post-forward. No additional clamping needed.
- **CS-40** (opt-in AD): M_init_k participates in the Wengert tape only when
  `with_tape()` is active. The reset copy `M_k ← M_init_k` is recorded as a tape
  operation when tape is live.
