# Ontological Code Smells (CS-01 through CS-09, CS-20, CS-21, CS-37, CS-38)

```
CONTRACT
  Purpose:    These smells define what concepts EXIST in NL.
              Conventional ML has MemoryModule, RecurrentLayer, HybridArchitecture.
              NL has NeuralLearningModule, frequency levels, context flow.
              Using the wrong ontology produces framework-contaminated code
              that looks like NL but behaves like a patched Transformer.
  Expects:    All type definitions, class names, and module structures.
  Guarantees: No conventional ML class names in the codebase.
              Paper-native vocabulary used throughout.
  Position:   specs/constraints/code_smells/01_ontological.md
  Source:     NL IS/IS NOT containers; HOPE paper Section 1-3
```

## CS-01: No MemoryModule class

```
SMELL: struct MemoryModule { ... }
WHY:   Memory is not a module. Memory IS the parameters.
       The weight matrix W IS the memory. The update rule IS memorization.
       Wrapping memory in a "MemoryModule" creates a false separation
       between "the model" and "the model's memory."
USE:   NeuralLearningModule — the indivisible unit that contains
       parameters (= memory) and update rules (= memorization).
TRACE: NL IS #6 (optimizers ARE associative memory modules)
```

## CS-02: Weight update = memory storage

```
SMELL: fn store_memory(key, value) { self.memory_bank.insert(key, value); }
WHY:   Weight updates ARE memory storage. There is no separate mechanism.
       When the inner loop updates M = M - eta * grad, that IS storing
       the gradient information into the memory matrix.
       A separate "store" operation implies memory is distinct from parameters.
USE:   The write() method on MemoryUpdateRule. This IS the storage mechanism.
TRACE: NL IS #6 (optimizers ARE memory)
```

## CS-03: CMS levels are learning processes, not cache tiers

```
SMELL: struct CacheLevel { tier: CacheTier, eviction_policy: ... }
WHY:   CMS frequency levels are NOT L1/L2/L3 cache tiers.
       Each level is a LEARNING PROCESS running at a different timescale.
       Level 0 learns fast features. Level 3 learns slow, persistent features.
       They're not "caching" — they're LEARNING at different rates.
USE:   CMSLevel with frequency, parameters, and update rule.
TRACE: NL IS #2 (nested, multi-level, parallel optimization)
```

## CS-04: No store/retrieve API

```
SMELL: fn store(&mut self, key: Tensor, value: Tensor)
       fn retrieve(&self, query: Tensor) -> Tensor
WHY:   Store/retrieve implies a database metaphor. NL memory is not a database.
       The inner loop's gradient descent IS the storage mechanism.
       The read (M @ q) IS the retrieval mechanism.
       Renaming write/read to store/retrieve imports a wrong ontology.
USE:   write() and read() — the MemoryUpdateRule trait methods.
TRACE: NL IS #6 (memory IS optimization)
```

## CS-05: No model vs optimizer state partition

```
SMELL: model_params = [...]; optimizer_state = {...}
WHY:   In NL, the optimizer IS part of the model (IS #6).
       Momentum accumulators are associative memory modules.
       Partitioning "model params" from "optimizer state" creates
       the exact separation NL dissolves.
USE:   All state is owned by the NeuralLearningModule.
       Momentum is a frequency level, not an optimizer detail.
TRACE: NL IS #6; NL IS NOT #5 (not optimizers as just optimizers)
```

## CS-06: nn.Parameter vs nn.Buffer is a false hierarchy

```
SMELL: self.register_buffer("momentum", torch.zeros(...))
WHY:   The Parameter/Buffer distinction assumes "params get gradients,
       buffers don't." In NL, momentum IS memory (IS #6).
       Whether a tensor gets outer-loop gradients depends on its
       STATE LIFECYCLE (outer_loop_param vs inner_loop_state),
       not on a Parameter/Buffer classification.
USE:   The three lifetimes from contract Section 3.
TRACE: NL IS #6; contract Section 3 (state lifecycle)
```

## CS-07: Frozen is a frequency statement, not exclusion

```
SMELL: if not self.frozen: self.update(grad)
WHY:   "Frozen" in NL means "this level's frequency hasn't fired yet."
       It's a timing condition, not a permanent exclusion.
       A frozen level accumulates error and will fire LATER.
       Using "frozen" as a boolean exclusion flag loses the temporal semantics.
USE:   pulse.is_active(level) — the frequency gate from Eq 71.
TRACE: NL IS #8 (continuum memory); Eq 71 (frequency scheduling)
```

## CS-08: Capacity accounting must span all frequency levels

```
SMELL: model_params = sum(p.numel() for p in model.parameters())
WHY:   In CMS, "parameter count" is misleading. A 4-level CMS model
       has 4x the parameters of a single level, but only ~1.14x the
       computation (because most levels are frozen most of the time).
       Reporting capacity as total parameter count over-counts.
USE:   Report: total params, active params per step (average),
       and active params at synchronization points (worst case).
TRACE: NL IS #8 (continuum); CMS parallelism implications
```

## CS-09: NeuralLearningModule is the unit, not Model

```
SMELL: class MyModel(nn.Module): ...
WHY:   The unit of NL is the NeuralLearningModule (NLM), not "a model."
       An NLM contains: parameters (memory), update rule (memorization),
       frequency schedule (CMS), and inner/outer loop structure.
       "Model" is the conventional ML word. NLM is the paper's word.
USE:   NeuralLearningModule as the top-level struct.
TRACE: NL IS #1 (new learning paradigm); Definition 1
```

## CS-20: No RecurrentLayer class

```
SMELL: struct RecurrentLayer { hidden_state: Tensor, ... }
WHY:   NL's inner loop looks like a recurrence but ISN'T a RecurrentLayer.
       A RecurrentLayer implies RNN semantics (h_t = f(h_{t-1}, x_t)).
       NL's inner loop is GRADIENT DESCENT on a memory matrix.
       The recurrence is an optimization loop, not a sequential layer.
USE:   MemoryUpdateRule::step() — the inner optimization step.
TRACE: NL IS #2 (nested optimization); NL IS NOT #1 (not single-level)
```

## CS-21: No HybridArchitecture class

```
SMELL: struct HybridArchitecture { attention: ..., memory: ..., fusion: ... }
WHY:   "Hybrid" implies two separate systems bolted together.
       NL uses COMPOSITION PATTERNS (MAC/MAG/MAL) that define how
       memory and attention interact. The composition is not "fusion" —
       it's a specific, paper-defined protocol.
USE:   CompositionPattern trait with MAC, MAG, MAL implementations.
TRACE: Titans Section 3.2; MIRAS compositional identity
```

## CS-37: Use "levels" not "layers" for frequency hierarchy

```
SMELL: cms_layers = [CMSLayer(freq=1), CMSLayer(freq=8)]
WHY:   "Layers" implies depth (layer 0 feeds layer 1 feeds layer 2).
       CMS levels are PARALLEL — they run at different frequencies,
       not in sequence. Level 0 and level 3 are both active at step 512.
       "Layers" imports the wrong geometric intuition.
USE:   "levels" — CMS levels, frequency levels, optimization levels.
TRACE: NL IS #2 (parallel); CMS structure
```

## CS-38: Use "build" not "train"

```
SMELL: fn train(&mut self, data: &Dataset)
WHY:   "Train" implies a conventional training loop with epochs,
       a train/test split, and a training phase that ends.
       NL has a BUILD phase (outer loop active) and a TEST/STREAM phase
       (outer loop frozen). The inner loop runs identically in all phases.
       "Build" captures what the outer loop does: build the initial state.
USE:   "build" for the outer-loop phase. "test" for assessment. "stream" for serving.
TRACE: CS-13 (word "training" is a smell); NL IS #5 (ICL emerges)
```
