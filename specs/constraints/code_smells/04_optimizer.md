# Optimizer Code Smells (CS-27 through CS-30)

```
CONTRACT
  Purpose:    The optimizer is not a separate entity in NL — it IS the model
              (NL IS #6). These smells enforce the coupling between optimizer
              and architecture that makes NL work. An optimizer that doesn't
              know about CMS frequencies will fight the architecture.
  Expects:    All optimizer implementations and parameter group configurations.
  Guarantees: No optimizer operates independently of the frequency schedule.
              The feedback loop (architecture generates gradients, optimizer
              applies them, architecture uses the result) is closed and testable.
  Cost:       Zero runtime cost — optimizer interface constraints only.
  Trade-off:  Cannot use off-the-shelf optimizers without CMS frequency awareness.
  Position:   specs/constraints/code_smells/04_optimizer.md
  Source:     NL IS #6, IS NOT #5; HOPE Section 9; CS-27 through CS-30
```

## CS-27: Optimizer frequency must match architecture frequency

```
SMELL: optimizer = AdamW(model.parameters(), lr=1e-4)
       // Single optimizer for ALL parameters, regardless of CMS level
WHY:   CMS creates k parameter groups at different frequencies.
       Level 0 updates every step. Level 3 updates every 512th step.
       A single optimizer with one learning rate ignores this structure.
       The optimizer MUST have separate state (momentum, learning rate)
       per CMS level.
USE:   M3 optimizer: k-level momentum accumulators, one per CMS frequency.
       Each level has its own learning rate and momentum buffer.
       The frequency scheduler gates which levels the optimizer touches.
TRACE: NL IS #6; HOPE Section 9 (M3); Eq 71 (frequency gating)
```

## CS-28: Optimizer must be frequency-aware

```
SMELL: // Using standard Adam/SGD/AdamW without modification
WHY:   This was corrected from the v1 interpretation.
       The v1 reading was "Adam/SGD/AdamW are forbidden" — WRONG.
       HOPE Section 9.2 explicitly uses AdamW as the outer-loop optimizer.
       Eq 71 says f(.) is arbitrary — ANY optimizer can be used per level.

       The REAL constraint: the optimizer must be FREQUENCY-AWARE.
       It must know which CMS levels are active and only update those.
       Standard Adam applied to all parameters every step VIOLATES this.
       Standard Adam applied per-CMS-level with frequency gating SATISFIES this.

CORRECT READING:
  -- ANY optimizer per level (AdamW, SGD, LAMB, etc.)
  -- But ONLY applied when the level's frequency gate fires
  -- With separate momentum/state per level
  -- This is what M3 formalizes

TRACE: NL IS #6; HOPE Eq 71 (f is arbitrary); Section 9.2 (AdamW usage)
```

## CS-29: Feedback loop is closed

```
SMELL: // Architecture and optimizer designed independently
       // "We'll tune the optimizer later"
WHY:   In NL, the architecture GENERATES gradients (inner loop) that the
       optimizer APPLIES (outer loop) to parameters that the architecture
       USES (next forward pass). This is a closed loop.
       If the optimizer doesn't understand the architecture's frequency
       structure, it applies updates at the wrong times.
       If the architecture doesn't account for the optimizer's momentum,
       it can't predict how quickly parameters will change.
       The loop must be designed as a UNIT, not as separate components.
USE:   M3 closes the loop: CMS frequencies define the optimizer's update
       schedule. Momentum accumulators are frequency-aligned.
       The optimizer IS part of the NeuralLearningModule (CS-31).
TRACE: NL IS #6; NL IS NOT #5 (not optimizers as just optimizers)
```

## CS-30: Harmony between optimizer and architecture is testable

```
SMELL: // "The optimizer should work with the architecture"
       // No test for this claim
WHY:   The claim "optimizer and architecture are harmonious" must be
       TESTABLE, not aspirational. Specifically:

       TEST 1: Gradient magnitude stability
         For each CMS level, the gradient magnitude should be
         roughly consistent across steps when that level fires.
         Diverging magnitudes indicate the optimizer is fighting
         the frequency schedule.

       TEST 2: Error buffer health
         Frozen levels accumulate error in buffers. When they fire,
         the accumulated error should be comparable to a single-step
         gradient (not 100x larger — that indicates the frozen period
         is too long for the learning rate).

       TEST 3: Level interaction
         After a slow level fires, the fast level's loss should improve
         (or at least not worsen). If it worsens, the transfer mechanism
         is broken (CS-24).

USE:   Invariant probes that check these three conditions during building.
       If any fails, the CMS configuration or learning rate is wrong.
TRACE: CS-30 definition; probe patterns from HOPE implementation
```
