# Conductor (Pulse Generator)

```
CONTRACT
  Purpose:    Committee Finding 2: "No scheduler/conductor." The Conductor
              creates and advances the Pulse that flows through every component.
              It is the single source of truth for timing, frequency scheduling,
              and phase transitions.
  Expects:    CMS configuration (k levels, chunk sizes, frequencies).
              Phase (Build, Test, Stream).
  Guarantees: A Pulse struct passed to every operation.
              Pulse is READ-ONLY — components never modify it.
              Global step advances monotonically.
              Active levels computed correctly per step.
  Cost:       O(k) per step for Pulse generation (trivial).
  Trade-off:  Centralized timing vs distributed. Centralized is simpler
              and prevents timing bugs (CS-32: observe then advance).
              But adds a single point of coordination.
  Position:   specs/infrastructure/scheduling/00_conductor.md
              Addresses: Committee Finding 2, CS-32, nl_toolchain tool-08
  Source:     Committee vCritique review; CS-32 "Observe then advance"
```

## The Pulse

```
STRUCT: Pulse
  global_step: u64              -- monotonic counter, never resets
  active_levels: Vec<bool>      -- which CMS frequency levels fire this step
  chunk_boundaries: Vec<u64>    -- chunk boundary per level (for parallelization)
  phase: Phase                  -- Build | Test | Stream

ENUM: Phase
  Build    -- outer loop is active (Enzyme AD computes gradients)
  Test     -- outer loop is frozen (only inner loop runs)
  Stream   -- continuous context processing (inner loop runs indefinitely)

  -- Note: there is no "train" or "eval" phase. (CS-38: Build not Train)
  -- The inner loop runs identically in ALL phases.
  -- The ONLY difference: Build phase runs Enzyme AD on the outer loop.
  -- Test and Stream phases skip outer-loop gradient computation.
```

## The Conductor

```
STRUCT: Conductor
  config: CMSConfig             -- frequency levels, chunk sizes
  pulse: Pulse                  -- current timing state
  step_observers: Vec<Box<dyn StepObserver>>  -- probes, loggers, etc.

TRAIT: StepObserver
  fn observe(&self, pulse: &Pulse, model: &Model)
  -- Called BEFORE the pulse advances (CS-32: observe then advance)

IMPL Conductor:
  fn new(config: CMSConfig) -> Self
    pulse = Pulse {
      global_step: 0,
      active_levels: active_levels(0, config.chunk_sizes),
      chunk_boundaries: compute_boundaries(0, config),
      phase: Phase::Build,
    }

  fn advance(&mut self, model: &Model)
    -- Step 1: Observe CURRENT state (CS-32: observe before mutate)
    for observer in &self.step_observers:
      observer.observe(&self.pulse, model)

    -- Step 2: Advance the pulse
    self.pulse.global_step += 1
    self.pulse.active_levels = active_levels(self.pulse.global_step,
                                              self.config.chunk_sizes)
    self.pulse.chunk_boundaries = compute_boundaries(self.pulse.global_step,
                                                      self.config)

  fn pulse(&self) -> &Pulse
    -- READ-ONLY access. Components cannot modify the pulse.
    return &self.pulse

  fn set_phase(&mut self, phase: Phase)
    self.pulse.phase = phase
```

## Integration Pattern

Every component receives the Pulse:

```
-- Memory update rule:
fn step(&mut self, x: &Tensor, outer: &Params, pulse: &Pulse) -> Tensor {
    if !pulse.is_active(self.level) {
        return self.read(x);  // frozen: read only
    }
    // active: full update
    ...
}

-- Composition pattern:
fn forward(&mut self, x: &Tensor, ..., pulse: &Pulse) -> Tensor {
    // passes pulse to memory.step() and attention.forward()
}

-- Frequency scheduler:
fn cms_update(&mut self, ..., pulse: &Pulse) {
    // uses pulse.active_levels to gate parameter updates
}
```

## CS-32 Enforcement: Observe Then Advance

```
-- WRONG (observe after advance):
conductor.advance()      // step = 5
probe.check(model)       // observes state that was modified by step 5
                         // but probe thinks it's observing step 4's result

-- CORRECT (observe then advance):
probe.check(model)       // observes state at step 4 (before modification)
conductor.advance()      // step = 5, modifies model state

-- The Conductor enforces this: step_observers run BEFORE advance().
```

## Pulse Reconciliation Protocol (Committee Finding 2) — PHASE 2

```
STATUS: DEFERRED TO PHASE 2
  Phase 1 uses the simple model: all ranks advance together via barrier sync
  (see Conductor Synchronization in 00_multi_gpu.md). This is correct and
  sufficient because the Conductor runs identically on every rank and all
  ranks call advance() after the same allreduce barrier.

  The committee identified a POTENTIAL problem: CMS asymmetric workloads could
  cause pulse skew if ranks advance independently. But Phase 1 does not allow
  independent advancement — ranks synchronize at every step via allreduce.
  The problem only arises if we later introduce asynchronous gradient updates
  or local-only steps for high-frequency levels.

  When that happens, the following protocol applies:

PROBLEM: Pulse Skew (arises ONLY with asynchronous advancement)
  GPU 0: completes pulse t in 10ms (only level 0 active)
  GPU 1: completes pulse t in 200ms (levels 0,1,2,3 all active)
  If GPU 0 advances independently, messages arrive at GPU 1 tagged
  with future pulse IDs. Without reconciliation, this causes confusion.

PHASE 2 PROTOCOL: Pulse Reconciliation

  STRUCT: PulseEnvelope
    pulse_id: u64,              -- which pulse this message belongs to
    source_rank: usize,         -- which GPU sent it
    cms_level: usize,           -- which CMS level this gradient belongs to
    timestamp_ns: u64,          -- wall-clock time for skew detection

  RULES (not needed until asynchronous advancement is implemented):
    1. Every inter-GPU message carries a PulseEnvelope
    2. Receiving rank classifies: FUTURE (queue), PAST (reject), PRESENT (process)
    3. Sync timeout: PULSE_SYNC_TIMEOUT_MS = 30_000 (configurable)
    4. Max skew bound: MAX_PULSE_SKEW = 4 pulses
    5. High-frequency levels MAY take local inner-loop steps without sync
       (but NOT outer-loop param updates)

  TRIGGER: Implement this protocol when any of these are added:
    - Asynchronous gradient updates (ranks advance at different rates)
    - Local-only steps for level 0 (CMS Independent variant)
    - Pipeline parallelism across CMS levels
```

## Phase Transitions

```
-- Build phase → Test phase:
conductor.set_phase(Phase::Test)
-- Inner loop continues identically.
-- Enzyme AD is disabled (no outer-loop gradients).
-- Memory still updates. Model still self-modifies.
-- This is NOT "freezing the model" — it's just stopping outer-loop learning.

-- Test phase → Stream phase:
conductor.set_phase(Phase::Stream)
-- Context streams indefinitely.
-- Global step keeps advancing.
-- Memory accumulates without bound (until retention mechanisms decay it).

-- There is NO mode distinction in the inner loop. (CS-10, CS-38)
-- The phase affects ONLY whether Enzyme AD runs on the outer loop.
```
