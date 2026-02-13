# Code Smell Enforcement Index

```
CONTRACT
  Purpose:    48 code smells define what NL_Hecate code must NOT look like.
              These are the immune system — if code violates a smell, it
              introduces foreign concepts from conventional ML frameworks.
              Organized into 6 categories below.
  Expects:    All code in the NL_Hecate repository.
  Guarantees: Every smell is enforceable (grep-able, lint-able, or trait-bounded).
              No code that violates a smell passes review.
              Smells trace to paper axioms or empirical lessons.
  Cost:       Review overhead. Some smells require custom lints.
  Trade-off:  47 rules is a lot. But each one was discovered the hard way
              (either from paper analysis or from bugs during implementation).
              The cost of enforcement is lower than the cost of debugging
              framework-contaminated code.
  Position:   specs/constraints/code_smells/00_index.md
  Source:     nl_code_smells collection (HADES), hope_code_smells collection
```

## Category Index

```
01_ontological.md     CS-01 through CS-09, CS-20, CS-21, CS-37, CS-38
  -- What concepts EXIST in NL vs conventional ML
  -- "There is no MemoryModule, no RecurrentLayer, no HybridArchitecture"

02_mode_and_phase.md  CS-10 through CS-17, CS-19
  -- The NL model has no mode distinction
  -- "There is no train/test, no epochs, no phase transitions"

03_structural.md      CS-18, CS-22 through CS-26, CS-31, CS-32
  -- How the model is structured and wired
  -- "Forward pass is the only API, NLM is indivisible, observe then advance"

04_optimizer.md       CS-27 through CS-30
  -- Optimizer/architecture coupling
  -- "Optimizer must be frequency-aware, feedback loop is closed"

05_miras.md           CS-33 through CS-36
  -- MIRAS design space constraints
  -- "Don't force same attentional bias, don't restrict memory structure"

06_infrastructure.md  CS-39 through CS-48
  -- Implementation-level constraints from Track A experience
  -- "Clamp decay, opt-in AD, DDP inflates, compile can't trace"
```

## Quick Reference: All 47 Smells

```
ONTOLOGICAL (13 smells):
  CS-01  No MemoryModule class
  CS-02  Weight update = memory storage (not separate)
  CS-03  CMS levels are learning processes, not cache tiers
  CS-04  No store/retrieve API
  CS-05  No model vs optimizer state partition
  CS-06  nn.Parameter vs nn.Buffer is a false hierarchy
  CS-07  Frozen is a frequency statement, not exclusion
  CS-08  Capacity accounting must span all frequency levels
  CS-09  NeuralLearningModule is the unit, not Model
  CS-20  No RecurrentLayer class
  CS-21  No HybridArchitecture class
  CS-37  Use "levels" not "layers" for frequency hierarchy
  CS-38  Use "build" not "train", "test" not "eval"

MODE AND PHASE (9 smells):
  CS-10  No mode distinction (no train/test mode)
  CS-11  No TrainingLoop or DataLoader class
  CS-12  State machine is exactly two states: RECEIVING_INPUT and PROCESSING
  CS-13  The word "training" is a code smell
  CS-14  Persistence is the variable, not mode
  CS-15  Context window size is the only variable between phases
  CS-16  Disconnecting transfer between levels is the original sin
  CS-17  End of pre-building is an arbitrary stop, not a phase transition
  CS-19  Frequency rate is configuration, not architecture

STRUCTURAL (8 smells):
  CS-18  Forward pass IS the only external API
  CS-22  Augment blocks, don't replace them
  CS-23  Level count is the architecture knob
  CS-24  Knowledge transfer is a mandatory wiring check
  CS-25  Transfer mechanism = meta-learning the initial state
  CS-26  No transfer = disconnected levels = root cause of gap
  CS-31  The NeuralLearningModule is indivisible
  CS-32  Observe then advance (stateful counters mutate AFTER observers)

OPTIMIZER (4 smells):
  CS-27  Optimizer frequency must match architecture frequency
  CS-28  Optimizer must be frequency-aware (not optimizer-identity)
  CS-29  Feedback loop (arch gradients -> optimizer update) is closed
  CS-30  Harmony between optimizer and architecture is testable

MIRAS (4 smells):
  CS-33  Don't force same attentional bias across models
  CS-34  Don't restrict memory to matrix-valued
  CS-35  Don't assume GD is the only memory algorithm
  CS-36  Don't restrict retention to quadratic/linear only

INFRASTRUCTURE (10 smells):
  CS-39  Learnable decay parameters must be clamped [CRITICAL]
  CS-40  Differentiation is opt-in, not opt-out [CRITICAL]
  CS-41  GPU utilization != throughput
  CS-42  Gradient checkpointing hurts NL
  CS-43  DDP inflates reported throughput
  CS-44  Optimization polarity flips per hardware
  CS-45  NL cannot fill high-end GPUs
  CS-46  Graph tracing cannot trace NL inner loops
  CS-47  In-place modification destroys reproducibility [CRITICAL]
  CS-48  Shared retention parameters across CMS levels
```

## Enforcement Strategy

```
LEVEL 1: Naming (grep-able)
  -- CS-01, CS-04, CS-10, CS-11, CS-13, CS-19, CS-20, CS-21, CS-37, CS-38
  -- Enforce: grep for forbidden class names, method names, words
  -- Automated: CI lint rule

LEVEL 2: Structural (AST-checkable)
  -- CS-12, CS-18, CS-22, CS-23, CS-31
  -- Enforce: check trait implementations, public API surface
  -- Automated: custom Rust lint (clippy plugin)

LEVEL 3: Semantic (review-required)
  -- CS-02, CS-03, CS-05, CS-06, CS-07, CS-08, CS-09, CS-14 through CS-17,
  -- CS-24 through CS-30, CS-33 through CS-36, CS-48
  -- Enforce: code review against smell descriptions
  -- Semi-automated: review checklist

LEVEL 4: Runtime (test-required)
  -- CS-32, CS-39, CS-40, CS-42, CS-43, CS-44, CS-45, CS-46, CS-47
  -- Enforce: invariant probes, runtime assertions, integration tests
  -- Automated: test suite includes smell-specific tests
```
