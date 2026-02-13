# Structural Code Smells (CS-18, CS-22 through CS-26, CS-31, CS-32)

```
CONTRACT
  Purpose:    These smells define how the NL model is structurally organized.
              The NeuralLearningModule is indivisible. The forward pass is
              the only API. Knowledge transfer between levels is mandatory.
              Observation happens before state mutation.
  Expects:    All module boundaries, public APIs, and wiring patterns.
  Guarantees: No broken abstractions, no silent disconnection between levels,
              no state mutation order bugs.
  Position:   specs/constraints/code_smells/03_structural.md
  Source:     NL IS #2, #6, #7; CS-18 through CS-32 (structural subset)
```

## CS-18: Forward pass IS the only external API

```
SMELL: model.update_memory(key, value)
       model.set_learning_rate(0.001)
       model.get_hidden_state()
WHY:   The NLM has ONE public method: forward (process input, produce output).
       Everything else — memory updates, learning rate, state — is internal.
       Exposing internal methods creates bypass routes around the NLM's
       self-modifying behavior. If you can poke memory directly, you can
       break the invariants that the inner loop maintains.
USE:   fn process(&mut self, input: &Tensor, pulse: &Pulse) -> Tensor
       This is the ONLY external API. Everything else is private.
TRACE: NL IS #7 (self-modifying — the model controls its own modification)
```

## CS-22: Augment blocks, don't replace them

```
SMELL: model.layers[3] = CMSBlock(model.layers[3])  // replace
WHY:   CMS wraps existing blocks — it doesn't replace them.
       The original Transformer block (attention + MLP) still exists.
       CMS adds frequency gating AROUND the MLP. The attention is unchanged.
       Replacing blocks loses the original functionality.
       Augmenting blocks preserves it while adding multi-scale behavior.
USE:   CMSFrequencyGate wraps the MLP. Attention is passed through unchanged.
       The original block's forward() is called inside the CMS wrapper.
TRACE: HOPE Section 7 (CMS applied to existing architectures)
```

## CS-23: Level count is the architecture knob

```
SMELL: struct Config { hidden_size: 2048, num_heads: 16, num_layers: 36, ... }
WHY:   In conventional Transformers, the knobs are hidden_size, num_heads,
       num_layers. In NL, the PRIMARY knob is k (number of CMS levels).
       k=1 is a standard Transformer. k=4 is a 4-timescale CMS model.
       The paper proves this: k is what controls the continuum memory
       approximation quality.
USE:   CMSConfig { n_levels: 4, frequencies: [1, 8, 64, 512], ... }
       n_levels IS the architecture. Everything else is secondary.
TRACE: NL IS #8 (continuum memory); HOPE Section 7.1
```

## CS-24: Knowledge transfer is a mandatory wiring check

```
SMELL: // No explicit transfer mechanism between CMS levels
       // "It should work because they share the same input"
WHY:   CMS levels don't automatically share information. They have
       INDEPENDENT parameter groups updated at different frequencies.
       Without EXPLICIT transfer, level 3 (slow) never benefits from
       level 0's (fast) discoveries, and vice versa.
       This check must be a BUILD-TIME verification, not a runtime hope.
USE:   The wiring validator checks that every pair of adjacent levels
       has a transfer mechanism (error accumulation, M3 cascade, or
       shared initial state).
TRACE: CS-26 (no transfer = disconnected = root cause of gap)
```

## CS-25: Transfer mechanism = meta-learning the initial state

```
SMELL: def transfer(fast_level, slow_level):
         slow_level.params += alpha * fast_level.params
WHY:   Transfer between CMS levels is NOT parameter averaging.
       It's META-LEARNING: the slow level's initial state is optimized
       to be a good starting point for the fast level.
       The outer loop (Enzyme AD) learns initial params such that
       the inner loop (fast) can quickly adapt from them.
       This is MAML-like: learn an initialization, not a model.
USE:   Outer-loop params include M_init — the initial memory state
       that the inner loop starts from. Enzyme optimizes M_init
       so that the inner loop produces good outputs.
TRACE: NL IS #7 (self-modifying); HOPE Section 8 (meta-learning framing)
```

## CS-26: No transfer = disconnected levels

```
SMELL: for level in 0..k:
         level.step(input)  // each level runs independently
WHY:   If levels don't transfer information, they're just k independent
       models that happen to share input. This defeats the purpose of CMS.
       The paper's error accumulation mechanism (frozen levels accumulate
       gradients, active levels apply them) IS the transfer mechanism.
       Without it, there's no multi-scale learning — just parallelism.
       Disconnected levels ARE the root cause of the build/test gap
       in conventional pretrain → finetune pipelines.
USE:   Error buffers + M3 momentum cascade + shared initial state.
TRACE: NL IS #2 (nested — nesting implies connection); CS-16
```

## CS-31: The NeuralLearningModule is indivisible

```
SMELL: memory_module = model.get_memory()
       optimizer = model.get_optimizer()
       attention = model.get_attention()
       // Now manipulate them independently
WHY:   The NLM is a UNIT. You cannot extract the memory subsystem,
       the optimizer subsystem, or the attention subsystem and
       manipulate them independently. They are coupled by design:
       - The optimizer IS memory (IS #6)
       - Memory is updated by the inner loop
       - Attention processes memory output (composition pattern)
       Breaking the unit breaks the coupling.
USE:   NeuralLearningModule::process() — one method, one unit.
       Internal components are private.
TRACE: NL IS #6 (optimizer IS memory); NL IS #7 (self-modifying unit)
```

## CS-32: Observe then advance

```
SMELL: conductor.advance()         // step = 5
       probe.check(model)          // observes step-5 state
       // Bug: probe thinks it's observing step-4 result

WHY:   Stateful counters (global_step, CMS frequencies) must mutate
       AFTER all observers have read the current state.
       If the conductor advances before probes observe, probes see
       a state that doesn't match the step they think they're at.
       This was a real bug in v1 (probe reported wrong step's metrics).

CORRECT:
       probe.check(model)          // observes step-4 state
       conductor.advance()         // NOW step = 5

USE:   The Conductor's advance() method calls all StepObservers FIRST,
       then advances the Pulse. Observers always see pre-mutation state.
TRACE: CS-32 definition; Conductor spec (observe then advance)
```
