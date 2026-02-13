# Multi-GPU Distribution

```
CONTRACT
  Purpose:    CS-44: "DDP inflates reported throughput." The NL inner loop
              creates asymmetric computation patterns that break DDP's
              assumptions. This spec defines how NL_Hecate distributes
              across GPUs without inheriting DDP's problems.
  Expects:    Multiple GPU devices. A model with CMS frequency scheduling.
              Inner-loop state that is rank-local. Outer-loop params that sync.
  Guarantees: Outer-loop gradients are synchronized correctly across ranks.
              Inner-loop state is NEVER synchronized (it's rank-local).
              CMS frequency scheduling is globally consistent (same Pulse).
              No parameter staleness between ranks.
  Cost:       Communication: one allreduce per active CMS level per step.
              At most steps, only level 0 is active → 1 allreduce.
              At CMS sync points (all levels active), k allreduces.
  Trade-off:  Simpler than DDP (no hooks into autograd graph), but requires
              manual gradient synchronization. The benefit: we control
              exactly what gets synced and when.
  Position:   specs/infrastructure/distribution/00_multi_gpu.md
              Addresses: CS-43, CS-44, CS-45, nl_toolchain tool-07
  Source:     CS-43 (GPU utilization != throughput); CS-44 (DDP inflates);
              CS-45 (NL cannot fill high-end GPUs); Track A DDP experience
```

## Why Not DDP

```
PROBLEM 1: DDP syncs ALL gradients every step.
  -- In CMS, frozen levels have NO gradients — zero compute, zero grad.
  -- DDP doesn't know this. It waits for all parameters to produce gradients.
  -- With find_unused_parameters=True, it works but wastes communication.
  -- With find_unused_parameters=False, it deadlocks (frozen params never fire).

PROBLEM 2: DDP treats the model as a black box.
  -- DDP hooks into autograd.Function to intercept gradients.
  -- NL's inner loop is INSIDE the forward pass — DDP can't see it.
  -- DDP's gradient bucketing doesn't align with CMS levels.

PROBLEM 3: DDP synchronizes things that shouldn't be synchronized.
  -- Inner-loop state (memory, momentum) is rank-local by design.
  -- Each rank processes different context → different memory states.
  -- DDP has no concept of "this state is local, that state is global."

PROBLEM 4: DDP inflates throughput metrics (CS-44).
  -- DDP reports tokens/sec across all ranks.
  -- But NL's actual throughput is bottlenecked by the rare all-levels-active steps.
  -- At step 512 (all 4 levels active), computation is ~4x a normal step.
  -- DDP's "average tokens/sec" hides this spike.
```

## The NL Distribution Model

```
LAYER 1: Data Parallelism (chunk-level)
  -- Each rank processes a DIFFERENT chunk from the ContextStream.
  -- Each rank has its OWN ContextMemory (divergent by design).
  -- Each rank runs the SAME model with the SAME outer-loop params.

LAYER 2: Gradient Synchronization (CMS-aware)
  -- Only synchronize gradients for ACTIVE levels at each step.
  -- Frozen levels: no gradient, no communication.
  -- Active levels: allreduce gradient across ranks, then apply.

  fn sync_gradients(grads: &mut [Tensor], pulse: &Pulse, world: &ProcessGroup) {
    for level in 0..k {
      if pulse.is_active(level) {
        // Only sync active levels — frozen levels have no gradient
        allreduce(&mut grads[level], world, ReduceOp::Mean);
      }
    }
  }

LAYER 3: State Isolation
  -- OuterLoopParam: SYNCHRONIZED across ranks (all ranks have same weights)
  -- InnerLoopState: RANK-LOCAL (never leaves the GPU)
  -- ContextMemory: RANK-LOCAL (each rank processes different context)
  -- CMS error buffers: RANK-LOCAL (accumulated before sync)

  -- The ONLY thing that crosses rank boundaries: outer-loop gradients.
```

## Communication Pattern

```
Typical step (only level 0 active):
  1. Each rank computes forward pass with own context chunk
  2. Each rank computes Enzyme backward for level 0 params only
  3. Allreduce level 0 gradients across ranks
  4. Each rank applies optimizer update to level 0 params
  → 1 allreduce per step

Every 8th step (levels 0 and 1 active):
  1-2. Same as above, but for levels 0 AND 1
  3. Allreduce level 0 gradients; allreduce level 1 gradients
  4. Apply updates for both levels
  → 2 allreduces

Every 512th step (all 4 levels active):
  1-2. All levels compute forward and backward
  3. 4 allreduces (one per level)
  4. Apply updates for all levels
  → 4 allreduces (rare: happens 1/512 of steps)

Average allreduces per step: 1 + 1/8 + 1/64 + 1/512 ≈ 1.14
Compare to DDP: 1 allreduce per step (but over ALL parameters, even frozen ones)
```

## Gradient Checkpointing Warning (CS-42)

```
CS-42: "Gradient checkpointing hurts NL models."

In conventional models: gradient checkpointing saves memory by
recomputing activations in the backward pass. Net effect: less memory,
more compute. Usually a good trade-off.

In NL models: the inner loop runs during the forward pass.
Recomputing activations means RE-RUNNING the inner loop.
But the inner loop is STATEFUL — re-running it with different
random state gives different results. This breaks reproducibility (CS-46).

RECOMMENDATION: Do NOT use gradient checkpointing in NL models.
Instead, reduce memory pressure by:
  1. CMS itself reduces memory (frozen levels = zero activation memory)
  2. Smaller chunk sizes (fewer tokens in flight at once)
  3. Mixed precision (bf16 activations, fp32 accumulation)

If gradient checkpointing is absolutely required:
  -- The inner loop must be made DETERMINISTIC (fixed seed per chunk)
  -- This is possible but adds complexity and runtime cost
  -- Document this trade-off explicitly if used
```

## Conductor Synchronization

```
-- The Conductor MUST produce the SAME Pulse on all ranks.
-- This means: same global_step, same active_levels, same phase.
-- The Conductor is NOT distributed — it runs identically on each rank.
-- As long as all ranks start from the same state and advance together,
-- the Pulse is automatically consistent.

-- The advance() call happens AFTER gradient sync:
for_each_step {
  // 1. Each rank processes its own chunk (divergent context)
  // 2. Each rank computes gradients (divergent inner state, convergent outer grad)
  // 3. Allreduce outer gradients (now all ranks have same gradient)
  // 4. Apply optimizer update (all ranks apply same update → same params)
  // 5. Conductor.advance() (all ranks advance together → same Pulse)
}
```

## Pulse Reconciliation (Committee Finding 2) — PHASE 2

```
STATUS: DEFERRED TO PHASE 2

  Phase 1 uses synchronous advancement: all ranks call allreduce at every step,
  then all ranks call Conductor.advance(). Since the allreduce is a barrier,
  all ranks enter advance() at the same time with the same state.
  Pulse skew CANNOT occur in Phase 1.

  The committee identified a real concern — CMS asymmetric workloads create
  idle time when a fast rank waits for a slow rank at the barrier. But the
  correct Phase 1 response is to accept the idle time and measure it
  (see Throughput Reporting below), not to add asynchronous advancement
  complexity that introduces the very skew problem it tries to solve.

  See 00_conductor.md "Pulse Reconciliation Protocol" for the full Phase 2
  protocol (PulseEnvelope, sync timeouts, max skew bounds) to be implemented
  when asynchronous gradient updates are introduced.
```

## Throughput Reporting (CS-43, CS-44)

```
RULE: Report throughput as tokens-per-second-per-GPU, not aggregate.

RULE: Report BOTH average throughput AND worst-case throughput.
  -- Average: ~T_base * 1.14 (typical CMS overhead)
  -- Worst-case: T_base * k (all levels active, every 512th step)

RULE: Distinguish computation throughput from communication throughput.
  -- Computation: how fast does one GPU process tokens?
  -- Communication: how much time is spent in allreduce?
  -- NL's allreduce is smaller than DDP's (only active levels, not all params).
```

## Axiom Compliance

- **CS-43** (GPU utilization != throughput): Addressed by honest throughput reporting
- **CS-44** (DDP inflates): Replaced DDP with CMS-aware gradient sync
- **CS-45** (NL cannot fill high-end GPUs): CMS's asymmetric workload acknowledged
- **NL IS #2** (multi-level parallel): Each CMS level syncs independently
