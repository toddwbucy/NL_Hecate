# Mode and Phase Code Smells (CS-10 through CS-17, CS-19)

```
CONTRACT
  Purpose:    These smells enforce NL's most radical claim: there is no
              mode distinction. The model does NOT behave differently in
              "training" vs "inference." The inner loop runs identically
              in ALL phases. The ONLY difference: whether Enzyme AD
              computes outer-loop gradients (Build phase) or not (Test/Stream).
  Expects:    All control flow, API design, and phase management code.
  Guarantees: No code paths that diverge based on "training" vs "inference."
              No training loops, no epochs, no DataLoaders.
              The word "training" does not appear in the codebase.
  Position:   specs/constraints/code_smells/02_mode_and_phase.md
  Source:     NL IS #5, IS NOT #3; CS-10 through CS-17, CS-19
```

## CS-10: No mode distinction

```
SMELL: model.set_mode(Train) / model.set_mode(Test)
       if self.is_building { ... } else { ... }  // divergent behavior
WHY:   NL IS NOT #3: not static/fixed update rules.
       The inner loop is ALWAYS the same. It processes context identically
       regardless of whether we're building, testing, or streaming.
       A mode flag that changes inner-loop behavior violates this axiom.

ALLOWED: conductor.set_phase(Phase::Build) — this controls the CONDUCTOR,
         which decides whether Enzyme AD runs on the OUTER loop.
         The model itself never checks what phase it's in.
TRACE: NL IS NOT #3; NL IS #5 (ICL is the same mechanism)
```

## CS-11: No TrainingLoop or DataLoader class

```
SMELL: struct TrainingLoop { epochs: usize, dataloader: DataLoader, ... }
       struct DataLoader { batch_size: usize, shuffle: bool, ... }
WHY:   There is no training loop. There is a context processing loop.
       There is no DataLoader. There is a ContextStream.
       The conceptual difference: a training loop implies "see all data K times."
       A context stream implies "process tokens as they arrive."
USE:   ContextStream trait (specs/infrastructure/context_stream/)
TRACE: CS-13 (word "training"); NL IS #5 (continuous learning)
```

## CS-12: State machine is exactly two states

```
SMELL: enum ModelState { Initializing, Training, Validating, Saving, Inference }
WHY:   The NL model has exactly TWO states:
         RECEIVING_INPUT: waiting for the next token/chunk
         PROCESSING: running the inner loop on the current input
       Everything else (saving, loading, evaluating) is EXTERNAL to the model.
       The model doesn't know it's being saved. It doesn't know it's being tested.
       It processes input. Period.
TRACE: CS-18 (forward pass is the only API)
```

## CS-13: The word "training" is a code smell

```
SMELL: training_loss, training_step, is_training, train_model()
WHY:   "Training" carries baggage: epochs, train/test split, convergence criteria,
       learning rate schedules, early stopping.
       NL has NONE of these. The outer loop BUILDS initial parameters.
       The inner loop processes context. Neither is "training."
USE:   "build" (outer loop), "process" (inner loop), "stream" (serving).
       build_loss, process_step, is_building.
TRACE: NL IS #1 (new paradigm — new vocabulary)
ENFORCEMENT: grep -r "train" src/ — zero hits (except "trait" and "constraint")
```

## CS-14: Persistence is the variable, not mode

```
SMELL: if mode == Train: persist_gradients()
       if mode == Test: discard_gradients()
WHY:   What varies between Build and Test phase is NOT "mode" — it's
       whether outer-loop gradients PERSIST (are applied to params).
       In Build: Enzyme computes gradients, optimizer applies them.
       In Test: no Enzyme, no optimizer. Model processes identically.
       The variable is "gradient persistence," not "operating mode."
USE:   Phase enum on the Pulse. The Conductor decides gradient persistence.
       The model never checks.
TRACE: Contract Section 3 (state lifecycle); NL IS #5
```

## CS-15: Context window size is the only variable between phases

```
SMELL: build_config = Config(lr=0.001, batch_size=32, augment=True)
       test_config = Config(lr=0.0, batch_size=1, augment=False)
WHY:   Between Build and Test/Stream, the ONLY configuration that changes
       is the context window (how many tokens to process).
       Build processes finite contexts (build budget).
       Stream processes infinite contexts.
       Everything else — chunk size, CMS frequencies, model params — stays the same.
USE:   Phase::Build with a step budget. Phase::Stream without one.
TRACE: NL IS #5; CS-15 (the only variable is context size)
```

## CS-16: Disconnecting transfer between levels is the original sin

```
SMELL: for level in levels: level.train_independently()
WHY:   If CMS levels don't transfer knowledge between each other,
       they're just independent models running at different speeds.
       The POINT of CMS is that slow levels INFORM fast levels
       (error accumulation, initial state transfer).
       Disconnecting them is the root cause of the build/test gap.
USE:   Error accumulation (Eq 71) + M3 momentum cascade + transfer via
       shared initial state (meta-learning, CS-25).
TRACE: CS-24, CS-25, CS-26; NL IS #2 (nested, multi-level)
```

## CS-17: End of building is an arbitrary stop, not a phase transition

```
SMELL: fn on_build_complete(&mut self) { self.freeze(); self.compile(); }
WHY:   When the Build phase ends, NOTHING changes in the model's behavior.
       The outer loop just stops running. The inner loop continues.
       There is no "freeze" — the model was always processing the same way.
       There is no "compile" — the model was always the same code.
       The "end of building" is the human deciding to stop the outer loop.
USE:   conductor.set_phase(Phase::Test) — a Conductor method, not a model method.
TRACE: NL IS #5 (ICL emerges — the model keeps learning via inner loop)
```

## CS-19: Frequency rate is configuration, not architecture

```
SMELL: struct PreTrainer { ... }
       struct InferenceEngine { ... }
       pretraining_model vs inference_model
WHY:   The difference between "pre-training" and "in-context learning" is
       frequency_config — one codebase parameterized by frequency.
       Pre-training = running the outer loop at low frequency (many tokens
       between outer-loop updates). ICL = the inner loop running at token
       frequency. Both are the SAME model, the SAME code, the SAME process.
       The only difference is the Conductor's CMS frequency schedule.
       A separate "pretraining" system introduces a false architectural boundary.
USE:   CMSConfig with different frequency schedules.
       build_config.frequencies = [1, 8, 64, 512]  // build phase
       stream_config.frequencies = [1, 8, 64, 512]  // same — inner loop is identical
TRACE: NL IS #2, IS #3, IS NOT #1; NL reframing: "Pre-training Is ICL at Low Frequency"
ENFORCEMENT: grep -r "PreTrainer\|InferenceEngine\|pretraining_model\|inference_model" src/
```
