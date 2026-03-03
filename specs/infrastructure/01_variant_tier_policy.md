## CONTRACT

**Purpose**: Define the canonical variant tier taxonomy for NL-Hecate, establish the "build-a-bear"
model configuration surface exposed to users, and specify the validation rules that enforce tier
guarantees at config load time. This is the architectural decision gate for all future GPU kernel
work and variant expansion.

**Expects**:
- `BuildConfig` in `python/engine/config.py` is the user-facing config struct loaded from JSON
- `MemoryRuleKind`, `CompositionKind`, `MomentumKind`, `AttentionalBias`, `RetentionKind` are
  defined in `core/src/model.rs` and `core/src/retention.rs`
- `gpu_forward.rs` houses the GPU dispatch path (currently single shared `gpu_memory_forward()`)
- CUDA kernel pairs (forward + backward) exist for: DeltaRule, TitansLMM, HebbianRule (S2-M1)
- No GPU kernels exist for: Moneta, YAAD, MEMORA, LatticeOSR, Trellis, AtlasOmega

**Guarantees**:
- Every combination expressible in a config JSON maps to exactly one tier
- `BuildConfig.validate_gpu_tier()` raises a `ValueError` with tier context if GPU is requested
  and the combination has no GPU kernels. This is called after CLI overrides are applied so that
  `--cpu` takes effect before the check (V-05 is not in `from_file()`).
- A `hecate.py --validate-config <path>` dry-run flag exists to check configs before runs
- The "blessed" production config (TitansLMM + MAG + k=4 + EMA momentum) is the normative Tier 1
  reference and will always run on GPU without modification
- Tier 2 and Tier 3 combinations always run correctly on CPU regardless of GPU availability
- No variant silently falls back from GPU to CPU — the tier is explicit and user-visible

**Cost**:
- Named-variant mode (current BuildConfig style) remains the primary interface; raw MIRAS knob
  mode is deferred to a future spec revision
- Tier 1 is intentionally narrow at v0.4.x — exactly the combinations that have been
  training-validated on real data
- Tier graduation (Tier 2 → Tier 1) requires a training run, not just unit tests

**Trade-off**:
- A narrow Tier 1 is honest and prevents silent CPU fallback surprises. The alternative (broad Tier 1
  with "GPU-capable" variants that have never been training-validated) creates false confidence.
- The build-a-bear surface works best when every config the user can write produces a model that
  runs at the tier they requested. An explicit tier annotation in error messages preserves user
  autonomy without hiding the implementation state.

**Position**:
- This spec governs `BuildConfig` and `hecate.py` CLI. It does not modify core Rust enums.
- Algorithm and retention combination constraints (which MIRAS knob combinations are mathematically
  valid) remain in `specs/algorithms/` — this spec governs implementation tier, not mathematical
  validity.
- When a Tier 2 variant graduates to Tier 1 (new training run validates it), this spec is updated
  and a new entry is added to the `hecate_specs` HADES node.

**Source**:
- MIRAS framework (arxiv 2504.13173): 4-knob combinatorial memory design space
- HOPE / Nested Learning (arxiv 2512.24695): CMS frequency hierarchy
- Internal: S2-M1 GPU kernel delivery, `fineweb_edu_k4_v2.json` training validation

---

## The Build-a-Bear Model

NL-Hecate is not a single model — it is a composition framework. A user constructs an NLM by
selecting values for six orthogonal knobs. The config JSON is the bill of materials.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    NLM Bill of Materials                            │
├─────────────────────┬───────────────────────────────────────────────┤
│ Knob                │ Options                                       │
├─────────────────────┼───────────────────────────────────────────────┤
│ memory_rule         │ titans_lmm · delta_rule · hebbian ·           │
│                     │ moneta · yaad · memora · lattice_osr ·        │
│                     │ trellis · atlas_omega · swiglu_mlp            │
├─────────────────────┼───────────────────────────────────────────────┤
│ composition         │ mag · mac · mal                               │
├─────────────────────┼───────────────────────────────────────────────┤
│ attentional_bias    │ l2 · l1 · lp · kl · huber                    │
├─────────────────────┼───────────────────────────────────────────────┤
│ retention           │ l2_weight_decay · kl_divergence ·             │
│                     │ elastic_net · sphere_normalization             │
├─────────────────────┼───────────────────────────────────────────────┤
│ momentum            │ none · ema · delta_momentum · deep_momentum   │
├─────────────────────┼───────────────────────────────────────────────┤
│ k (CMS levels)      │ 1 · 2 · 4                                     │
└─────────────────────┴───────────────────────────────────────────────┘
```

This is `2504.13173`'s 4-knob MIRAS framework plus composition (Titans §4) and CMS scheduling
(HOPE §3). Every named variant in `MemoryRuleKind` is a pre-composed combination of these knobs
with a canonical set of defaults.

### Named Variants vs. Raw Knob Mode

**v0.4.x**: The user selects a `memory_rule` (named variant). The other knobs default to the
canonical values for that variant and can be overridden within the constraints documented per
variant.

**Future (v0.5+)**: A `model_structure` block will allow direct MIRAS knob selection without
naming a variant. The named variant will become syntactic sugar.

---

## Tier Definitions

### Tier 1 — Production

**Guarantee**: Runs on GPU. Has been training-validated on real corpora (FineWeb-Edu or
equivalent). All forward/backward paths pass gradient checks. Supported in production.

**What "training-validated" means**: A full training run to convergence (or a deliberate
stability sweep of ≥100K steps) on real token data, producing a finite loss curve with no
NaN/Inf. The run must be recorded as a closed HADES task with a `fineweb_edu_k4` or equivalent
config reference.

**Current Tier 1 members**:

| memory_rule | composition | momentum | k | Validated config |
|---|---|---|---|---|
| `titans_lmm` | `mag` | `ema` | 4 | `fineweb_edu_k4_v2.json` |

Everything else that runs on GPU (DeltaRule, Hebbian) has CUDA kernels but has not been
training-validated — it is Tier 2a, not Tier 1.

### Tier 2 — Supported

**Guarantee**: Runs correctly. CPU path is complete and unit-tested. May have GPU kernels (2a)
or be CPU-only (2b). Not training-validated. Supported for research and experimentation.

**Tier 2a — GPU-capable, not training-validated**:

| memory_rule | composition | k | Status |
|---|---|---|---|
| `delta_rule` | `mag` · `mac` · `mal` | 1 · 2 · 4 | Has CUDA kernels (S2-M1), not trained |
| `hebbian` | `mag` · `mac` · `mal` | 1 · 2 · 4 | Has CUDA kernels (S2-M1), not trained |

**Tier 2b — CPU-only, no GPU kernels**:

| memory_rule | composition | k | Status |
|---|---|---|---|
| `moneta` | `mag` · `mac` · `mal` | 1 · 2 · 4 | CPU-complete, no GPU kernels |
| `yaad` | `mag` · `mac` · `mal` | 1 · 2 · 4 | CPU-complete, no GPU kernels |
| `memora` | `mag` · `mac` · `mal` | 1 · 2 · 4 | CPU-complete, no GPU kernels |
| `trellis` | `mag` · `mac` · `mal` | 1 · 2 · 4 | CPU-complete, no GPU kernels |

### Tier 3 — Research Stub

**Guarantee**: Spec is complete. Some unit tests pass. NOT for production or extended research
use. No GPU path. Not all forward/backward paths tested. No support commitment.

| memory_rule | Blocker to Tier 2 |
|---|---|
| `lattice_osr` | Sphere normalization retention + complex slot routing |
| `atlas_omega` | M3 optimizer not fully wired |
| `swiglu_mlp` | No inner-loop M state — different architecture class |

---

## Config Schema (the User-Facing Knobs)

The following fields in `BuildConfig` constitute the build-a-bear surface. All are optional with
defaults that resolve to the Tier 1 canonical config when combined.

```json
{
  "model": {
    "memory_rule":       "titans_lmm",
    "composition":       "mag",
    "attentional_bias":  "l2",
    "retention":         "l2_weight_decay",
    "momentum_kind":     "ema",
    "k":                 4,
    "chunk_sizes":       [1, 8, 64, 512]
  }
}
```

### Per-Knob Defaults and Constraints

**`memory_rule`** (string, default: `"titans_lmm"`)
- Selects the inner-loop memory optimizer. Each named variant has canonical defaults for
  `attentional_bias`, `retention`, and `momentum`. Overrides are permitted within the
  constraints documented in `specs/algorithms/memory_update_rules/`.
- Canonical variant defaults:
  | Variant | bias | retention | momentum |
  |---|---|---|---|
  | `titans_lmm` | `l2` | `l2_weight_decay` | `ema` |
  | `delta_rule` | `l2` | `l2_weight_decay` | `none` |
  | `hebbian` | `l2` | `l2_weight_decay` | `none` |
  | `moneta` | `l1` | `l2_weight_decay` | `none` |
  | `yaad` | `l2` | `l2_weight_decay` | `none` |
  | `memora` | `kl` | `kl_divergence` | `none` |
  | `lattice_osr` | `l2` | `sphere_normalization` | `none` |
  | `trellis` | `l2` | `l2_weight_decay` | `none` |
  | `atlas_omega` | `l2` | `l2_weight_decay` | `delta_momentum` |

**`composition`** (string, default: `"mag"`)
- How memory connects to attention. `mag` is the widest-tested path.
- MAC requires `window_size >= 2 * seq_len` (validated at config load).

**`attentional_bias`** (string, default: per variant)
- Inner-loop loss function. `l2` is the numerical stability baseline.
- `lp` requires an additional `lp_p: float` field (default: 3.0).

**`retention`** (string, default: per variant)
- Memory decay mechanism. `sphere_normalization` is only valid with `lattice_osr`.
- `kl_divergence` is only valid with `memora`.

**`momentum`** (string, default: per variant)
- `ema` and `delta_momentum` require `titans_lmm` or `atlas_omega` memory rules.
- `none` is valid for all rules.
- `deep_momentum` is Tier 3 regardless of memory rule.

**`k`** (integer, default: `4`)
- Number of CMS frequency levels. `chunk_sizes` must have exactly `k` entries.
- `k=1` is the degenerate case (no multi-scale memory, single-level).

---

## Validation Rules

`BuildConfig.from_file()` enforces V-01, V-02, V-03, V-04, and V-06 at load time.
V-05 (GPU tier check) is enforced by `BuildConfig.validate_gpu_tier()` after CLI overrides
are applied, so `--cpu` takes effect before the check. All errors include the tier context
and a suggested fix.

### Rule V-01: Retention–Rule compatibility
`sphere_normalization` is only valid with `memory_rule: lattice_osr`.
`kl_divergence` is only valid with `memory_rule: memora`.
Error template:
```text
ConfigError: retention 'sphere_normalization' requires memory_rule 'lattice_osr'
  You requested memory_rule '{rule}'. Use retention 'l2_weight_decay' for {rule}.
```

### Rule V-02: Momentum–Rule compatibility
`ema` and `delta_momentum` require `memory_rule` in `{titans_lmm, atlas_omega}`.
`deep_momentum` is Tier 3 regardless of memory rule.
Error template:
```text
ConfigError: momentum 'ema' requires memory_rule 'titans_lmm' or 'atlas_omega'
  You requested memory_rule '{rule}'. Use momentum 'none' for {rule}.
```

### Rule V-03: MAC window size
When `composition: mac`, `window_size >= 2 * seq_len` must hold.
Error template:
```text
ConfigError: composition 'mac' requires window_size >= 2 * seq_len
  Got window_size={w}, seq_len={s}. Set window_size >= 2 * {s}.
```

### Rule V-04: k / chunk_sizes coherence
`len(chunk_sizes) == k`.
Error template:
```text
ConfigError: k={k} but len(chunk_sizes)={n}. chunk_sizes must have exactly k entries.
```

### Rule V-05: GPU tier check (when device=cuda requested)
When `device: "cuda"` (explicit or inferred from CUDA_VISIBLE_DEVICES), the combination
must be Tier 1 or Tier 2a (has GPU kernels). Tier 2b and Tier 3 combinations trigger:
```text
ConfigError: '{memory_rule}' is Tier {tier} — no GPU kernels available.
  This combination runs on CPU only. Either:
    (a) set device: "cpu" to run intentionally on CPU, or
    (b) use a Tier 1 or Tier 2a memory_rule (titans_lmm, delta_rule, hebbian)
       to run on GPU.
  See specs/infrastructure/01_variant_tier_policy.md for the full tier matrix.
```

### Rule V-06: Tier 3 always warns
Any Tier 3 combination (lattice_osr, atlas_omega, swiglu_mlp, deep_momentum) emits a
non-fatal warning at config load time regardless of device:
```text
ConfigWarning: '{field}' is Tier 3 (research stub). Not production-ready.
  Proceeding on CPU. See specs/infrastructure/01_variant_tier_policy.md.
```

---

## The `validate-config` Subcommand

`hecate.py --validate-config <path>` performs a dry-run config check without launching training.
It runs all V-01 through V-06 rules and reports:

```text
$ python hecate.py --validate-config configs/my_experiment.json

Config: configs/my_experiment.json
  memory_rule:      moneta          (Tier 2b — CPU only)
  composition:      mag             ✓
  attentional_bias: l1              ✓
  retention:        l2_weight_decay ✓
  momentum:         none            ✓
  k:                4               ✓
  device:           cuda            ✗

ERROR V-05: 'moneta' is Tier 2b — no GPU kernels available.
  Set device: "cpu" or switch to a Tier 1/2a memory_rule.

1 error, 0 warnings. Config is INVALID for cuda.
```

---

## The Blessed Production Config

The canonical Tier 1 reference. All integration tests and stability sweeps run against this
configuration:

```json
{
  "swa": {
    "d_model": 512,
    "num_heads": 8,
    "head_dim": 64,
    "seq_len": 512,
    "window_size": 64,
    "vocab_size": 32000
  },
  "model": {
    "memory_rule":       "titans_lmm",
    "composition":       "mag",
    "attentional_bias":  "l2",
    "retention":         "l2_weight_decay",
    "momentum_kind":     "ema",
    "k":                 4,
    "chunk_sizes":       [1, 8, 64, 512],
    "m_norm_max":        [100.0, 100.0, 100.0, 100.0]
  },
  "build": {
    "lr": 0.0006,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0
  }
}
```

Reference file: `python/configs/fineweb_edu_k4_v2.json`.

---

## Tier Graduation Protocol

A combination graduates from Tier 2 → Tier 1 when:

1. A training run of ≥ 100K steps completes without NaN/Inf on a real corpus
2. The loss curve is finite and trending downward
3. The run is recorded as a closed HADES task with `type: "training_run"`
4. A `hecate_artifacts` node exists for the resulting checkpoint
5. This spec is updated to add the variant to the Tier 1 table

A combination graduates from Tier 3 → Tier 2b when:
1. All CPU `step()` + `step_backward()` paths pass gradient checks
2. A unit test suite exists covering the rule's canonical configuration
3. A code smell compliance audit passes (relevant CS-## edges in HADES)

---

## Implementation Subtasks

The following tasks implement this spec. The epic (`task_1a0431`) does not close until all are done.

| Task key | Description | Depends on |
|---|---|---|
| `task_af2834` | Write this spec (current task) | — |
| `task_cbbf32` | Wire clamp_theta: DeltaRule, Trellis, Moneta CPU | this spec |
| `task_31cfb6` | Wire clamp_theta: gate_compute_cuda GPU | task_cbbf32 |
| TBD | Add `attentional_bias`, `retention`, `momentum` fields to BuildConfig | this spec |
| TBD | Implement V-01 through V-06 validation in BuildConfig.from_file() | above |
| TBD | Implement `hecate.py validate-config` subcommand | V-01..V-06 |
| TBD | Training run: delta_rule + mag + k=4 → graduate to Tier 1 | clamp tasks |
| TBD | Training run: hebbian + mag + k=4 → graduate to Tier 1 | clamp tasks |
