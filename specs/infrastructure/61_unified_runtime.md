# Unified Runtime — One Binary, One Config, Phase List

```text
CONTRACT
  Purpose:    Define the single entry point for all NLM execution. One JSON config
              file, one `nl_hecate run` invocation. The config contains a `phases`
              array — an ordered list of (data, duration) pairs executed sequentially.
              The model processes tokens identically in every phase. The only variables
              are: which data directory, how long, and optional per-phase overrides.
  Expects:    Rust CLI binary with step_adamw(), prefill(), decode_token() on GpuStackedModel.
              JSON config file with model, build defaults, and phases array.
              Pre-tokenized data directories (dolmino format).
  Guarantees: (1) Backward pass runs on every input in every phase.
              (2) No train/eval mode distinction — CS-10 strictly enforced.
              (3) No special "flashcard" or "reinforcement" primitive — all phases
                  are the same operation: feed tokens to step_adamw().
              (4) A single checkpoint format works across all phases.
              (5) Phase transitions are checkpoint boundaries.
  Cost:       Every token pays full backward cost. This is correct — the model
              learns from everything it sees (Titans §3).
  Trade-off:  No way to skip backward for "cheap eval." This is deliberate.
              A future freeze_weights flag (spec 55) is the sanctioned escape
              hatch, but backward itself is non-negotiable.
  Position:   specs/infrastructure/61_unified_runtime.md
  Source:     CS-10 (no train/eval), CS-18 (forward pass is the only API),
              Spec 55 (two-path inference), Titans (2501.00663) §3,
              HOPE (2512.24695) §4.1 — the optimizer IS part of the model
```

## Core Principle: Phases, Not Modes

An NLM has exactly two execution paths (spec 55):

1. **Learn** — `step_adamw()`: forward + backward + optimizer.
2. **Speak** — `prefill() + decode_token()`: autoregressive generation.

Every phase uses Learn. Chat additionally uses Speak. But the runtime doesn't
distinguish between "training on a corpus," "reinforcement on math tables," or
"learning from user input." They are all: **feed tokens to step_adamw()**.

A curriculum is an ordered list of phases. Each phase is a data source and a
duration. The runtime walks the list.

## JSON Config Schema

```json
{
    "description": "Gear 1: foundations + math reinforcement, d=1024",

    "model": {
        "d_model": 1024,
        "num_heads": 32,
        "seq_len": 512,
        "window_size": 512,
        "vocab_size": 49152,
        "memory_rule": "titans",
        "composition": "mag",
        "hope_variant": "chained",
        "k": 4,
        "chunk_sizes": [1, 8, 64, 512],
        "m_norm_max": [100.0, 100.0, 100.0, 100.0],
        "error_clip": [0.0, 0.0, 0.0, 0.0],
        "residual": true,
        "n_blocks": 8,
        "parallel_strategy": "tnt_hierarchical",
        "tnt_global_chunk_size": 64,
        "tnt_local_chunk_size": 8,
        "memory_reset": "periodic",
        "reset_intervals": [1, 8, 64, 512],
        "tape_multiplier": 1,
        "tape_strategies": ["exact", "proxy", "proxy", "proxy"]
    },

    "build": {
        "optimizer": {
            "type": "adamw",
            "lr": 0.0003,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.1
        },
        "warmup_steps": 500,
        "max_grad_norm": 1.0,
        "alpha_floor": [0.0, 0.0, 0.0, 0.0],
        "theta_ceil": [1.0, 1.0, 1.0, 1.0],
        "batch_size": 8,
        "log_every": 8,
        "save_every": 5000,
        "gpu": true,
        "seed": 42,
        "run_dir": "runs/gear1_d1024",
        "load": null,
        "cms_sidecar": true
    },

    "phases": [
        {"data": "data/gear1_foundations", "steps": 10000},
        {"data": "data/math_tables",      "think_rounds": 3},
        {"data": "data/gear1_foundations", "steps": 10000},
        {"data": "data/philosophy_quotes", "think_rounds": 2, "optimizer": {"type": "adamw", "lr": 0.0001}},
        {"data": "data/gear1_foundations", "steps": 40000}
    ]
}
```

## The Phase Primitive

A phase is the only scheduling unit. It has two required fields:

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `data` | string | yes | Path to tokenized data directory |
| `steps` | int | yes (unless think_rounds) | Process N segments, then advance |
| `think_rounds` | int | alternative to steps | Learn→speak→feed back, N iterations (see below) |

Plus optional per-phase overrides. Every override reverts to the `build` default
when the phase completes. If it's in `build`, it's overridable per phase.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `label` | string | none | Human-readable name (logged, not used by runtime) |
| `optimizer` | object | inherit | Full optimizer block override (see below) |
| `batch_size` | int | inherit | Batch size |
| `seq_len` | int | inherit | Sequence length (tokens per segment) |
| `save_every` | int | inherit | Checkpoint interval (steps) |
| `log_every` | int | inherit | Metrics logging interval (steps) |
| `max_grad_norm` | float | inherit | Gradient clipping threshold |
| `warmup_steps` | int | inherit | LR warmup (applies to steps within this phase) |

The rule is simple: **`build` is the default, `phase` is the override.** Anything
you might want to change between a foundations pass and a reinforcement pass —
or between a short-context warmup and a long-context ramp — is overridable.
Anything structural to the model (d_model, num_heads, k) is NOT overridable
per phase — that would require a different model, not a different phase.

### `steps` — streaming consumption

A phase with `steps` streams through the data directory using ContextStream,
processing exactly N segments via `step_adamw()` before moving on. If the data
is exhausted before N steps, the cursor wraps to the beginning. This is the
standard build operation — one-way token consumption.

### `think_rounds` — iterative self-refinement

A phase with `think_rounds` uses both execution paths from spec 55. The model
learns from the data, then speaks (generates output), and its output becomes
the input for the next round. Each round builds on the model's own increasingly
refined understanding.

```rust
input = load(data)
for round in 0..think_rounds {
    // LEARN — process input through step_adamw (forward + backward + optimizer)
    step_adamw(input, ...)

    // SPEAK — generate output from what was just learned
    output = prefill(input) → decode_token() loop

    // REDIRECT — the model's output becomes the next round's input
    input = output
}
```

Round 1: raw data → model learns, produces first attempt
Round 2: first attempt → model learns from its own output, produces refinement
Round 3: refinement → model corrects its correction, produces final output

This is the same self-refinement mechanism as the chat `/think` tiers, applied
to curated data. The model doesn't see the math tables three times — it sees
the tables once, then learns from its own increasingly accurate reproduction.

**`steps` and `think_rounds` are mutually exclusive.** A phase uses one or the
other. `steps` is for streaming large corpora (one-way). `think_rounds` is for
focused refinement on curated material (iterative).

### Why `think_rounds` matters

The distinction is fundamental:
- **steps** = feed tokens, model learns passively (absorb)
- **think_rounds** = feed tokens, model learns actively by examining its own output (reflect)

Passive absorption builds breadth. Active reflection builds depth. A curriculum
interleaves both: foundations for breadth, think_rounds for depth on critical material.

## The Optimizer Block

The optimizer is a nested object in `build`, not a flat string. Each optimizer
type declares its own parameters. The `type` field selects the optimizer; all
other fields are optimizer-specific.

### AdamW (current)

```json
"optimizer": {
    "type": "adamw",
    "lr": 0.0003,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.1
}
```

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `type` | string | required | `"adamw"` |
| `lr` | float | 0.0003 | Learning rate |
| `beta1` | float | 0.9 | First moment decay |
| `beta2` | float | 0.999 | Second moment decay |
| `weight_decay` | float | 0.1 | Decoupled weight decay |

### M3 (future — HOPE §4.1)

The M3 optimizer from the HOPE paper is a learned optimizer — it uses the model's
own memory system to compute parameter updates. When implemented, it will have
a different parameter set:

```json
"optimizer": {
    "type": "m3",
    "lr": 0.0003,
    "meta_lr": 0.001,
    "inner_steps": 5
}
```

The exact M3 parameter set will be defined in its own spec when implementation
begins. The point is that the optimizer block is extensible — new optimizer types
add new fields without touching the phase system or the rest of `build`.

### SGD (minimal baseline)

```json
"optimizer": {
    "type": "sgd",
    "lr": 0.001,
    "momentum": 0.9
}
```

### Per-Phase Optimizer Override

A phase can override the entire optimizer block or just individual fields.
When a phase specifies an `optimizer` object, it fully replaces the `build`
default for that phase's duration. On phase completion, the default is restored.

```json
"phases": [
    {"data": "data/foundations", "steps": 10000},
    {"data": "data/math_tables", "think_rounds": 3, "optimizer": {"type": "adamw", "lr": 0.0001}},
    {"data": "data/experimental", "steps": 5000, "optimizer": {"type": "m3", "meta_lr": 0.001}}
]
```

Phase 1 uses the default optimizer from `build`. Phase 2 uses AdamW at a lower lr.
Phase 3 switches to M3 entirely. Phase boundaries restore the default.

### Why the optimizer is in `build`, not `model`

The optimizer is *how* weights are updated — an operational concern. The model
block defines *what* the model is — a structural concern. Changing the optimizer
doesn't reshape weight tensors or reallocate GPU memory. Changing d_model does.

This distinction matters because:
- **`model`** fields are fixed for the lifetime of a checkpoint. Changing them
  means starting over or running a conversion.
- **`build`** fields (including optimizer) can change between phases, between
  gears, between sessions. The checkpoint carries optimizer state (moments),
  but the optimizer type itself is config-driven.

When switching from AdamW to M3 mid-curriculum, the AdamW moments are discarded
and M3 initializes fresh. This is expected — different optimizers have different
state shapes.

## Runtime Execution

```rust
for (i, phase) in config.phases.iter().enumerate() {
    let opt = phase.optimizer.as_ref().unwrap_or(&config.build.optimizer);
    let batch = phase.batch_size.unwrap_or(config.build.batch_size);
    let seq_len = phase.seq_len.unwrap_or(config.model.seq_len);

    eprintln!("[phase {i}: {} — optimizer={}, lr={}]",
              phase.label.as_deref().unwrap_or(&phase.data), opt.type_, opt.lr);

    if let Some(steps) = phase.steps {
        // ── Steps mode: streaming consumption ──
        let stream = ContextStream::open(&phase.data, seq_len);
        for step in 0..steps {
            let segment = stream.next_segment(seq_len);
            let (loss, grad_norm) = step_optimizer(segment, &opt, ...);
            if global_step % save_every == 0 { checkpoint(); }
            global_step += 1;
        }

    } else if let Some(rounds) = phase.think_rounds {
        // ── Think rounds: iterative self-refinement ──
        let mut input = load_tokenized(&phase.data);
        for round in 0..rounds {
            eprintln!("  [think round {round}/{rounds}]");

            // LEARN from current input
            let (loss, grad_norm) = step_optimizer(input, &opt, ...);

            // SPEAK — generate output from what was just learned
            let logits = prefill(input, pulse);
            let output = decode_loop(logits, max_gen_tokens);

            // REDIRECT — output becomes next round's input
            input = output;
            global_step += 1;
        }
    }

    // Forced checkpoint at phase boundary
    checkpoint();
    eprintln!("[phase {i} complete — checkpoint at step {global_step}]");
}
```

Key details:
- **`global_step`** is continuous across phases — it never resets
- **Warmup** applies to global_step, not per-phase step count
- **Phase boundary = forced checkpoint** — always safe to resume from a phase transition
- **Log file** is continuous — phase transitions appear as label entries in metrics.jsonl
- **Overrides revert** — if phase 2 sets `lr: 0.0001`, phase 3 goes back to `build.lr`
- **Think round output** is tokenized in-memory — never written to disk unless explicitly logged
- **`step_optimizer`** dispatches to the appropriate optimizer based on `opt.type_` — currently only AdamW, future: M3, SGD, etc. The function signature is the same regardless of optimizer type; optimizer-specific params come from the optimizer block

## Single-Phase Backward Compatibility

A config with no `phases` array degrades to the current behavior:

```json
{
    "model": { ... },
    "build": { "steps": 60000, ... },
    "data": { "path": "data/gear1_foundations", "format": "dolmino" }
}
```

When `phases` is absent, the runtime synthesizes a single phase from `data.path`
and `build.steps`. Existing configs continue to work unchanged.

## Chat as a Phase (Future)

Chat is not yet implemented in the Rust CLI. When it is, it will be a special
phase type:

```json
{"chat": true, "tokenizer": "SmolLM2", "max_gen_tokens": 512, "temperature": 0.8}
```

A chat phase runs an interactive loop: learn from user input via `step_adamw()`,
then speak via `prefill() + decode_token()`. It terminates on user exit or EOF.

Chat phases can appear in the phase list like any other:

```json
"phases": [
    {"data": "data/foundations", "steps": 10000},
    {"chat": true, "tokenizer": "SmolLM2"},
    {"data": "data/foundations", "steps": 10000}
]
```

Build 10K steps, drop into interactive chat (model learns from conversation),
then resume building on foundations. The checkpoint carries all state across
phase types.

### Backward Pass in Chat

The backward pass is **essential** in chat (spec 55, spec 52):
- Without backward, M accumulates via forward recurrence with no gradient feedback
- Periodic reset never fires (it lives inside step_adamw's backward path)
- M norm clamp never fires
- Gates get no gradient signal — they drift to whatever their biases initialized
- Result: degenerate repetition

The backward pass IS the model's self-correction mechanism. An NLM without
backward is not doing test-time learning — it's just a broken recurrence.

### Future: freeze_weights

For enterprise production deployment where chat latency matters and the model
should not drift during conversation, a `freeze_weights` flag may be added to
`step_adamw()` (per spec 55). This would run forward + backward (M gets gradient
feedback, resets fire, clamps fire) but skip the AdamW parameter update.

This is NOT part of the current research configuration. **Do not expose this
flag in the config schema until it is needed.**

## Checkpoint Compatibility

All phases produce and consume the same checkpoint format (safetensors):
- Model weights (W_K, W_V, W_Q, gates, MLP params)
- Build state (optimizer moments, global step, LR schedule position)
- Context memory (M matrices, momentum S, conductor step)
- CMS sidecar (`.cms.json`, if `cms_sidecar: true`)
- Phase index (which phase was active when checkpoint was written)

The `load` field in `build` resumes from any checkpoint. If the checkpoint was
written mid-phase, execution resumes at that phase and step. If at a phase
boundary, execution begins the next phase.

## Example Configs

### Gear 1: Foundations with math reinforcement

```json
{
    "description": "Gear 1: foundations corpus interleaved with math reinforcement",
    "model": { "d_model": 1024, "num_heads": 32, "k": 4, "..." : "..." },
    "build": { "optimizer": {"type": "adamw", "lr": 0.0003}, "batch_size": 8, "save_every": 5000 },
    "phases": [
        {"label": "foundations-warmup",  "data": "data/gear1_foundations",  "steps": 10000},
        {"label": "think-math",          "data": "data/math_tables",       "think_rounds": 3, "optimizer": {"type": "adamw", "lr": 0.0001}},
        {"label": "foundations-main",    "data": "data/gear1_foundations",  "steps": 20000},
        {"label": "think-quotes",        "data": "data/philosophy_quotes", "think_rounds": 2, "optimizer": {"type": "adamw", "lr": 0.0001}},
        {"label": "foundations-final",   "data": "data/gear1_foundations",  "steps": 30000}
    ]
}
```

### Gear 2: Long-context activation

```json
{
    "description": "Gear 2: same data, longer sequences, L2/L3 activation",
    "model": { "d_model": 1024, "num_heads": 32, "k": 4, "seq_len": 32768, "..." : "..." },
    "build": { "optimizer": {"type": "adamw", "lr": 0.0001}, "batch_size": 1, "save_every": 1000, "load": "runs/gear1/checkpoints/model_step60000.safetensors" },
    "phases": [
        {"label": "long-context-warmup", "data": "data/gear1_foundations", "steps": 2000, "seq_len": 8192},
        {"label": "long-context-main",   "data": "data/gear1_foundations", "steps": 10000}
    ]
}
```

### Simple single-phase (backward compatible)

```json
{
    "description": "Simple build — no phase list needed",
    "model": { "..." : "..." },
    "build": { "steps": 60000, "..." : "..." },
    "data": { "path": "data/gear1_foundations", "format": "dolmino" }
}
```

## Command Line Interface

```bash
# Everything goes through one command
nl_hecate run --config configs/gear1_d1024.json
```

One binary. One verb. One config. The `phases` array describes the entire
curriculum. No subcommands, no mode flags, no `--eval`, no `--flashcard`.

## Constraint Compliance

- **CS-10**: No train/eval distinction. Every phase calls `step_adamw()`. There
  is no "reinforcement mode" vs "training mode" — they are the same operation.
- **CS-11**: No gradient tape on/off switch. Backward always runs.
- **CS-18**: Forward pass is the only external API. All phases enter through
  `step_adamw()` (which wraps forward+backward+optimizer).
- **CS-22**: Single entry point for sequence processing.
- **CS-32**: Observe-then-advance — the model sees tokens before updating.

## Implementation Sequence

1. **Phase list parser** — deserialize `phases` array in config.rs, with validation
   (exactly one of steps/rounds per phase, data path exists)
2. **Phase loop in run.rs** — outer loop over phases, inner loop is existing step loop.
   Per-phase overrides applied/reverted around each phase.
3. **Phase-boundary checkpointing** — forced save at every phase transition, log
   phase label to metrics.jsonl
4. **Single-phase fallback** — synthesize phases from `data` + `build.steps` when
   `phases` is absent
5. **Resume logic** — checkpoint stores phase index + step-within-phase for mid-phase
   resume
6. **Chat phase** (future) — special phase type with stdin tokenizer + generation loop
