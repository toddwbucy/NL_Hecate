"""BuildConfig and learning rate schedules."""

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

# Tier taxonomy for V-05/V-06 validation (spec: 01_variant_tier_policy.md)
_TIER_1 = {"titans"}
_TIER_2A = {"delta", "hebbian"}
_TIER_2B = {"moneta", "yaad", "memora", "trellis"}
_TIER_3_RULES = {"lattice", "atlas", "atlas_omega", "swiglu_mlp", "swiglu"}
_GPU_CAPABLE = _TIER_1 | _TIER_2A  # Tier 1 + Tier 2a have full CUDA support
# TODO: promote moneta+yaad to _GPU_CAPABLE after config plumbing PR lands
# (GpuContextState sizing for MLP memory, frozen-level MLP readout)
# Rules that support ema / delta_momentum (V-02)
_MOMENTUM_RULES = {"titans", "atlas", "atlas_omega"}
# Enum sets for direct field validation
_ATTENTIONAL_BIASES = {"l2", "l1", "lp", "kl", "huber"}
_RETENTION_KINDS = {"l2_weight_decay", "kl_divergence", "elastic_net", "sphere_normalization"}


def _deprecated_checkpoint_fields(flat: dict) -> None:
    """Spec 32: migrate eval_every/tape_every → save_every with deprecation warnings."""
    if "eval_every" in flat:
        if "save_every" not in flat:
            flat["save_every"] = flat["eval_every"]
            warnings.warn(
                f"eval_every={flat['eval_every']} is deprecated (spec 32). "
                f"Mapped to save_every={flat['save_every']}. "
                "Use save_every directly in config.",
                DeprecationWarning, stacklevel=3,
            )
        else:
            warnings.warn(
                f"eval_every={flat['eval_every']} ignored — save_every={flat['save_every']} "
                "takes precedence (spec 32).",
                DeprecationWarning, stacklevel=3,
            )
    for old_field in ("tape_every", "eval_max_chunks"):
        if old_field in flat and flat[old_field]:
            warnings.warn(
                f"{old_field} is deprecated (spec 32) and ignored. "
                "Tape diagnostics fire at checkpoint cadence (save_every).",
                DeprecationWarning, stacklevel=3,
            )
    for old_field in ("val_path", "val_doc_starts_path",
                      "window_local_val", "window_val_tokens"):
        if old_field in flat:
            warnings.warn(
                f"{old_field} is deprecated (spec 32) and ignored. "
                "Coherence samples use the build stream (CS-10).",
                DeprecationWarning, stacklevel=3,
            )
            del flat[old_field]


class BuildConfig:
    """Validated build configuration. Replaces raw dict parsing.

    Loads from JSON/YAML config file, validates all fields, applies CLI
    overrides. All fields have sensible defaults for quick experimentation.
    """

    # Model
    d_model: int = 64
    num_heads: int = 4
    seq_len: int = 32
    window_size: int = 16
    vocab_size: int = 256
    k: int = 1
    chunk_sizes: list[int] | None = None
    memory_rule: str = "delta"
    composition: str = "mag"

    # HOPE (self-referential projections)
    projection_kind: str = "static"         # "static" or "adaptive"
    self_generated_values: bool = False      # Phase 3 self-modifying memory
    self_ref_chunk_size: int = 1             # chunkwise self-ref (1 = sequential)
    momentum_kind: str = "none"             # "none", "ema", "delta_momentum", "deep_momentum"
    momentum_d_hidden: int = 0              # momentum MLP hidden dim (0 = d*d matrix)
    attentional_bias: str | None = None     # inner-loop loss: "l2", "l1", "lp", "kl", "huber"
    retention: str | None = None            # decay mechanism: "l2_weight_decay", "kl_divergence",
    #                                       #   "elastic_net", "sphere_normalization"

    # Memory enable flag (False = SWA-only baseline, no CMS memory modules)
    memory_enabled: bool = True

    # Per-level alpha retention gate clamps (empty = unclamped, CS-39 style)
    alpha_floor: list[float] | None = None  # per-level sigmoid lower bound (prevents catastrophic forgetting)
    alpha_ceil: list[float] | None = None   # per-level sigmoid upper bound (prevents memory stasis)

    # Per-level theta gate clamps (empty = unclamped, CS-39 style)
    theta_floor: list[float] | None = None  # per-level softplus lower bound
    theta_ceil: list[float] | None = None   # per-level softplus upper bound

    # Per-level M Frobenius norm ceiling (empty = disabled)
    m_norm_max: list[float] | None = None   # straight-through clamp after M update
    # Per-level per-token error clip ceiling (empty = disabled, spec 17)
    error_clip: list[float] | None = None   # clip ‖e_t‖₂ to this value in CUDA kernels

    # Per-level gate bias initialization overrides
    b_alpha_init: list[float] | None = None  # override default b_alpha (3.0,4.0,...) per level
    b_theta_init: list[float] | None = None  # override default b_theta (-4.6,-5.6,...) per level

    # SwiGluMlp / Llama level stacking (HOPE §7.3)
    intermediate_size: int = 0             # 0 for matrix rules; 8192 for Llama-3.2-1B
    donor_layers: list[int] | None = None  # which Llama layers to transplant (e.g. [0,5,10,15])
    donor_weights: str | None = None       # path to extracted donor weights .pt file

    # Parallelization strategy (TNT chunkwise, associative scan, etc.)
    parallel_strategy: str | None = None   # "tnt_hierarchical", "chunkwise", or None (sequential)
    tnt_global_chunk_size: int = 64        # shard size for TNT
    tnt_local_chunk_size: int = 8          # local chunk size within TNT shard

    # Build
    lr: float = 0.01
    steps: int = 500
    seed: int = 42
    save_path: str = "checkpoints/model.json"
    save_every: int = 0
    log_every: int = 10

    # Optimizer
    optimizer: str = "sgd"
    warmup_steps: int = 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    max_grad_norm: float = 0.0  # 0 = disabled

    # M3 optimizer (spec 34): multi-scale momentum with NS orthogonalization
    # Only used when optimizer="m3". See specs/infrastructure/34_m3_gpu_integration.md
    m3_beta1: float = 0.9      # fast momentum coefficient
    m3_beta2: float = 0.999    # second moment coefficient
    m3_beta3: float = 0.99     # slow momentum coefficient
    m3_alpha: float = 0.5      # weight of slow momentum in combined update
    m3_chunk_size: int = 8     # Ĉ — slow momentum (M2) update frequency
    m3_ns_iterations: int = 5  # Newton-Schulz iterations T
    m3_eps: float = 1e-8       # epsilon for 1D Adam-style V division

    # Data
    data_path: str | None = None
    data_format: str = "byte"  # "byte", "sharegpt", or "dolmino"
    doc_starts_path: str | None = None  # byte offsets of document boundaries
    val_path: str | None = None  # byte-level val corpus
    val_doc_starts_path: str | None = None  # val doc boundaries

    # Checkpoint event (spec 32): save + tape + coherence sample fire together
    coher_sample: bool = True    # Decode coherence sample at each checkpoint event

    # Probe tuning (probes fire at each checkpoint event; these control their cost)
    probe_max_tokens: int = 20   # tokens generated per probe (was hardcoded 60)
    probe_prompts: int = 1       # how many EVAL_PROMPTS to run for probe1 (1-4)

    # Deprecated fields (backward compat — mapped to save_every, logged as warnings)
    eval_every: int = 0          # DEPRECATED: use save_every
    eval_max_chunks: int = 100   # DEPRECATED: ignored (single-chunk coherence sample)
    tape_every: int = 0          # DEPRECATED: tape fires at checkpoint cadence

    # Parameter saturation detection (task_962e72)
    # Tracks per-level EMA of gradient norm at each slow_level_fire event.
    # saturation_ratio = ema_gnorm / peak_gnorm → 1.0 at peak, → 0 at saturation.
    # Emits level_saturation JSONL event when ratio < threshold for `saturation_window` fires.
    saturation_ema_alpha: float = 0.1    # EMA decay — higher = more reactive, lower = smoother
    saturation_threshold: float = 0.15  # ratio below which a level is considered saturating
    saturation_window: int = 5           # consecutive fires below threshold to confirm saturation

    # Gradient checkpointing (VRAM optimization for memory rules)
    checkpoint_interval: int | None = None  # None = full trajectory; C = store M every C steps
    # Tape multiplier (spec 25): how many CMS cycles of cache to retain.
    # 1 = one cycle (default, minimum for backward).
    # N = N cycles (deeper gradient flow, more memory).
    tape_multiplier: int = 1

    # Per-level tape strategy (spec 27): controls M/S trajectory in backward.
    # None/empty = auto (L0=exact, L1+=proxy). If set, length must equal k.
    # "exact" = full M trajectory. "proxy" = M_final only (truncated BPTT).
    tape_strategies: list | None = None

    # Batching
    batch_size: int = 1  # number of sequences per step (GPU batching)

    # TNT periodic reset (2511.07343 §3.2)
    # "carry_forward": M carries across all steps (default, current behavior)
    # "periodic": M resets to zeros at each CMS level fire boundary (TNT mode)
    memory_reset: str = "carry_forward"

    # Gate warmup protocol (specs/infrastructure/09_gate_warmup.md)
    # theta_floor_init: per-level scaffold floor at step 0; decays linearly to 0
    # gate_warmup_decay_steps: step at which theta_floor_init reaches 0 (Phase 2 end)
    # gate_warmup_falsification_step: step at which go/no-go thresholds are checked
    # gate_warmup_l2_threshold / l3_threshold: falsification pass thresholds
    gate_warmup_theta_floor_init: list[float] | None = None
    gate_warmup_decay_steps: int = 0
    gate_warmup_falsification_step: int = 20000
    gate_warmup_l2_threshold: float = 0.005
    gate_warmup_l3_threshold: float = 0.001

    # Level stacking (specs/infrastructure/07_push_up_level_stacking.md)
    extend_k: int | None = None       # Target k (must be loaded_k + 1 for push_up/stack_up)
    push_up: bool = False             # Shift existing levels to slower frequencies
    stack_up: bool = False            # Keep existing levels, add fresh level at top
    clone_to: int | None = None       # Clone expansion: duplicate levels to fill target k (any k > loaded_k)
    push_up_init: str = "random"     # L0 init on push-up: "random" (Xavier, spec 28) or "clone" (legacy)
    freeze_embed: bool = False       # Freeze w_embed and w_unembed (spec 28)
    freeze_embed_after: int | None = None  # Step at which to freeze embeddings (None = immediate if freeze_embed)
    dormancy_floors: list[float] | None = None  # Per-level M-diff floor for dormancy detection (spec 28)
    dormancy_consecutive: int = 5     # Consecutive below-floor steps to trigger "dormant" status
    data_seek: int | None = None      # Override data cursor to this token position

    # Auto-promotion (specs/infrastructure/12_metric_driven_promotion.md)
    auto_promote: bool = False       # Enable convergence-driven level promotion
    target_k: int = 4                # Maximum k to promote to
    promotion_cooldown: int = 2000   # Min steps at current k before allowing promotion
    promotion_stability_window: int = 50   # Ratio samples to compute rolling stdev over
    promotion_stability_streak: int = 50   # Consecutive low-stdev samples to confirm plateau
    promotion_stability_threshold: float = 0.025  # Stdev below this = ratio has stabilized
    promotion_rewind_pct: float = 0.0  # Fraction of THIS level's steps to rewind data cursor on push-up
    #                                  # 0.0 = no rewind (cursor continues), 0.25 = 25% rewind, 1.0 = full rewind

    # Window-local validation (specs/infrastructure/12_metric_driven_promotion.md §5)
    window_local_val: bool = False   # Carve val from current training window (not fixed val set)
    window_val_tokens: int = 50000   # Tokens reserved for window-local validation

    # Residual stream + pre-LayerNorm (specs/infrastructure/13_residual_stream.md)
    residual: bool = False  # True = additive residual, no sigmoid gating

    # Multi-block stacking (specs/infrastructure/14_multi_block_stacking.md)
    n_blocks: int = 1  # Number of stacked SWA+CMS blocks (1 = single-block legacy)

    # Stacked tape diagnostics (specs/infrastructure/15_stacked_tape_diagnostics.md)
    # "off" = no tape overhead (default), "gpu" = GPU-resident per-(block,level) gnorms,
    # "cpu" = full Wengert tape on CPU (DGD delta, M state readout — slow)
    tape_device: str = "off"

    # Run directory (unified output location)
    # When set, all outputs go under this directory:
    #   {run_dir}/config.json       — frozen config copy
    #   {run_dir}/metrics.jsonl     — training log
    #   {run_dir}/checkpoints/      — model checkpoints + cursor sidecars
    #   Console output: redirect nohup to {run_dir}/output.log
    # Overrides save_path and log_file. Backward compat: leave None to use
    # save_path/log_file directly (old behavior).
    run_dir: str | None = None

    # Runtime
    gpu: bool = True  # GPU by default; --cpu to override
    load: str | None = None
    log_file: str | None = None

    def __init__(self, **kwargs: Any):
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(f"Unknown config key: {key}")
        self._validate()

    def _validate(self):
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.optimizer not in ("sgd", "adamw", "adamw_gpu", "adamw_gpu_stacked", "m3"):
            raise ValueError(
                f"optimizer must be 'sgd', 'adamw', 'adamw_gpu', 'adamw_gpu_stacked', or 'm3', got '{self.optimizer}'")
        if self.optimizer == "m3":
            if self.m3_ns_iterations < 1:
                raise ValueError(f"m3_ns_iterations must be >= 1, got {self.m3_ns_iterations}")
            if self.m3_chunk_size < 1:
                raise ValueError(f"m3_chunk_size must be >= 1, got {self.m3_chunk_size}")
            for name, val in [("m3_beta1", self.m3_beta1), ("m3_beta2", self.m3_beta2),
                              ("m3_beta3", self.m3_beta3)]:
                if not (0.0 < val < 1.0):
                    raise ValueError(f"{name} must be in (0, 1), got {val}")
            if self.m3_alpha < 0.0:
                raise ValueError(f"m3_alpha must be >= 0, got {self.m3_alpha}")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.chunk_sizes is None:
            self.chunk_sizes = [1] * self.k
        if len(self.chunk_sizes) != self.k:
            raise ValueError(
                f"chunk_sizes length {len(self.chunk_sizes)} must match k={self.k}")
        if self.projection_kind not in ("static", "adaptive"):
            raise ValueError(
                f"projection_kind must be 'static' or 'adaptive', got '{self.projection_kind}'")
        if self.momentum_kind not in ("none", "ema", "delta_momentum", "deep_momentum"):
            raise ValueError(
                f"momentum_kind must be 'none', 'ema', 'delta_momentum', or 'deep_momentum', "
                f"got '{self.momentum_kind}'")
        if self.self_ref_chunk_size < 1:
            raise ValueError(
                f"self_ref_chunk_size must be >= 1, got {self.self_ref_chunk_size}")
        if not isinstance(self.tape_multiplier, int) or self.tape_multiplier < 1:
            raise ValueError(
                f"tape_multiplier must be an int >= 1, got {self.tape_multiplier!r}")
        # Spec 27: normalize tape_strategies
        if self.tape_strategies is None:
            self.tape_strategies = []
        if self.tape_strategies and len(self.tape_strategies) != self.k:
            raise ValueError(
                f"tape_strategies length ({len(self.tape_strategies)}) must equal k ({self.k})")
        if self.momentum_d_hidden < 0:
            raise ValueError(
                f"momentum_d_hidden must be >= 0, got {self.momentum_d_hidden}")
        if self.self_generated_values and self.projection_kind != "adaptive":
            raise ValueError("self_generated_values requires projection_kind='adaptive'")
        if self.self_ref_chunk_size > 1 and self.projection_kind != "adaptive":
            raise ValueError("self_ref_chunk_size > 1 requires projection_kind='adaptive'")
        # Enum validation for new MIRAS knob fields
        if self.attentional_bias is not None and self.attentional_bias not in _ATTENTIONAL_BIASES:
            raise ValueError(
                f"attentional_bias must be one of {sorted(_ATTENTIONAL_BIASES)}, "
                f"got '{self.attentional_bias}'")
        if self.retention is not None and self.retention not in _RETENTION_KINDS:
            raise ValueError(
                f"retention must be one of {sorted(_RETENTION_KINDS)}, "
                f"got '{self.retention}'")
        # V-01: retention–rule compatibility
        if self.retention == "sphere_normalization" and self.memory_rule != "lattice":
            raise ValueError(
                f"retention 'sphere_normalization' requires memory_rule 'lattice', "
                f"got '{self.memory_rule}'. Use retention 'l2_weight_decay' for '{self.memory_rule}'.")
        if self.retention == "kl_divergence" and self.memory_rule != "memora":
            raise ValueError(
                f"retention 'kl_divergence' requires memory_rule 'memora', "
                f"got '{self.memory_rule}'. Use retention 'l2_weight_decay' for '{self.memory_rule}'.")
        # V-02: momentum–rule compatibility
        if self.momentum_kind in ("ema", "delta_momentum"):
            if self.memory_rule not in _MOMENTUM_RULES:
                raise ValueError(
                    f"momentum '{self.momentum_kind}' requires memory_rule 'titans' or "
                    f"'atlas'/'atlas_omega', got '{self.memory_rule}'. "
                    f"Use momentum_kind 'none' for '{self.memory_rule}'.")
        # V-06: Tier 3 always warns (non-fatal)
        if self.memory_rule in _TIER_3_RULES:
            warnings.warn(
                f"memory_rule '{self.memory_rule}' is Tier 3 (research stub). Not production-ready. "
                f"Proceeding on CPU. See specs/infrastructure/01_variant_tier_policy.md.",
                stacklevel=3)
        if self.momentum_kind == "deep_momentum":
            warnings.warn(
                f"momentum 'deep_momentum' is Tier 3 (research stub). Not production-ready. "
                f"Proceeding on CPU. See specs/infrastructure/01_variant_tier_policy.md.",
                stacklevel=3)
        if self.data_format not in ("byte", "sharegpt", "dolmino", "smollm"):
            raise ValueError(
                f"data_format must be 'byte', 'sharegpt', 'dolmino', or 'smollm', "
                f"got '{self.data_format}'")
        if self.alpha_floor is not None and len(self.alpha_floor) != self.k:
            raise ValueError(
                f"alpha_floor length {len(self.alpha_floor)} must match k={self.k}")
        if self.alpha_ceil is not None and len(self.alpha_ceil) != self.k:
            raise ValueError(
                f"alpha_ceil length {len(self.alpha_ceil)} must match k={self.k}")
        if self.theta_floor is not None and len(self.theta_floor) != self.k:
            raise ValueError(
                f"theta_floor length {len(self.theta_floor)} must match k={self.k}")
        if self.theta_ceil is not None and len(self.theta_ceil) != self.k:
            raise ValueError(
                f"theta_ceil length {len(self.theta_ceil)} must match k={self.k}")
        if self.m_norm_max is not None and len(self.m_norm_max) != self.k:
            raise ValueError(
                f"m_norm_max length {len(self.m_norm_max)} must match k={self.k}")
        if self.error_clip is not None and len(self.error_clip) != self.k:
            raise ValueError(
                f"error_clip length {len(self.error_clip)} must match k={self.k}")
        if self.b_alpha_init is not None and self.b_alpha_init and len(self.b_alpha_init) != self.k:
            raise ValueError(
                f"b_alpha_init length {len(self.b_alpha_init)} must match k={self.k}")
        if self.b_theta_init is not None and self.b_theta_init and len(self.b_theta_init) != self.k:
            raise ValueError(
                f"b_theta_init length {len(self.b_theta_init)} must match k={self.k}")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be >= 1 or None, got {self.checkpoint_interval}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.memory_reset not in ("carry_forward", "periodic"):
            raise ValueError(
                f"memory_reset must be 'carry_forward' or 'periodic', got '{self.memory_reset}'")
        if self.gate_warmup_theta_floor_init is not None:
            if self.k < 4:
                raise ValueError(
                    f"gate_warmup requires k >= 4 (CMS L2/L3 levels), got k={self.k}")
            if len(self.gate_warmup_theta_floor_init) != self.k:
                raise ValueError(
                    f"gate_warmup_theta_floor_init length "
                    f"{len(self.gate_warmup_theta_floor_init)} must match k={self.k}")
            if self.gate_warmup_decay_steps <= 0:
                raise ValueError(
                    "gate_warmup_decay_steps must be > 0 when gate_warmup_theta_floor_init is set")
            if self.gate_warmup_falsification_step < 0:
                raise ValueError(
                    f"gate_warmup_falsification_step must be >= 0, "
                    f"got {self.gate_warmup_falsification_step}")
            if (self.gate_warmup_falsification_step > 0
                    and self.gate_warmup_falsification_step <= self.gate_warmup_decay_steps):
                raise ValueError(
                    f"gate_warmup_falsification_step ({self.gate_warmup_falsification_step}) "
                    f"must be > gate_warmup_decay_steps ({self.gate_warmup_decay_steps})")
            if self.gate_warmup_l2_threshold <= 0:
                raise ValueError(
                    f"gate_warmup_l2_threshold must be > 0, got {self.gate_warmup_l2_threshold}")
            if self.gate_warmup_l3_threshold <= 0:
                raise ValueError(
                    f"gate_warmup_l3_threshold must be > 0, got {self.gate_warmup_l3_threshold}")
        if not (0.0 < self.saturation_ema_alpha <= 1.0):
            raise ValueError(
                f"saturation_ema_alpha must be in (0, 1], got {self.saturation_ema_alpha}")
        if not (0.0 < self.saturation_threshold < 1.0):
            raise ValueError(
                f"saturation_threshold must be in (0, 1), got {self.saturation_threshold}")
        if self.saturation_window < 1:
            raise ValueError(
                f"saturation_window must be >= 1, got {self.saturation_window}")
        # Level stacking validation
        if self.extend_k is not None:
            if self.load is None:
                raise ValueError("extend_k requires 'load' to be set (checkpoint to extend from)")
            if self.extend_k < 2:
                raise ValueError(f"extend_k must be >= 2, got {self.extend_k}")
            if not self.push_up and not self.stack_up:
                raise ValueError("extend_k requires either push_up=true or stack_up=true")
            if self.push_up and self.stack_up:
                raise ValueError("push_up and stack_up are mutually exclusive")
        if self.freeze_embed_after is not None and self.freeze_embed_after < 0:
            raise ValueError(
                f"freeze_embed_after must be >= 0, got {self.freeze_embed_after}")
        if self.dormancy_floors is not None and len(self.dormancy_floors) != self.k:
            raise ValueError(
                f"dormancy_floors length {len(self.dormancy_floors)} must match k={self.k}")
        if self.dormancy_consecutive < 1:
            raise ValueError(
                f"dormancy_consecutive must be >= 1, got {self.dormancy_consecutive}")
        if self.push_up_init not in ("random", "clone"):
            raise ValueError(
                f"push_up_init must be 'random' or 'clone', got '{self.push_up_init}'")
        if self.push_up and self.extend_k is None:
            raise ValueError("push_up=true requires extend_k to be set")
        if self.stack_up and self.extend_k is None:
            raise ValueError("stack_up=true requires extend_k to be set")
        if self.data_seek is not None and self.data_seek < 0:
            raise ValueError(f"data_seek must be >= 0, got {self.data_seek}")
        # Auto-promotion validation
        if self.auto_promote:
            if self.target_k < 2:
                raise ValueError(
                    f"target_k must be >= 2 for auto_promote, got {self.target_k}")
            # extend_k is compatible with auto_promote: manual extend at startup,
            # then auto_promote handles subsequent promotions at runtime
            if self.promotion_cooldown < 0:
                raise ValueError(
                    f"promotion_cooldown must be >= 0, got {self.promotion_cooldown}")
            if not (0.0 <= self.promotion_rewind_pct <= 1.0):
                raise ValueError(
                    f"promotion_rewind_pct must be in [0.0, 1.0], got {self.promotion_rewind_pct}")
        # Multi-block stacking guards
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.n_blocks > 1:
            if not self.residual:
                raise ValueError("n_blocks > 1 requires residual=true (pre-LN residual stream)")
            if self.composition.lower() == "mac":
                raise ValueError(
                    "n_blocks > 1 does not yet support composition='mac' "
                    "(MAC persistent tokens have no stacked slot). Use 'mag'.")
        # Tape device validation
        if self.tape_device not in ("off", "gpu", "cpu"):
            raise ValueError(
                f"tape_device must be 'off', 'gpu', or 'cpu', got '{self.tape_device}'")
        # Window-local val validation
        if self.window_local_val and self.window_val_tokens < 1:
            raise ValueError(
                f"window_val_tokens must be >= 1 when window_local_val is enabled, "
                f"got {self.window_val_tokens}")
        # Clone expansion validation
        if self.clone_to is not None:
            if not self.load:
                raise ValueError("clone_to requires load (a checkpoint path to clone from)")
            if self.extend_k is not None:
                raise ValueError("clone_to and extend_k are mutually exclusive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @classmethod
    def from_file(cls, path: str) -> "BuildConfig":
        """Load config from JSON file."""
        with open(path) as f:
            raw = json.load(f)
        flat: dict[str, Any] = {}
        for section in ("model", "build", "data"):
            if section in raw:
                sub = raw[section]
                if section == "data":
                    if "path" in sub:
                        flat["data_path"] = sub["path"]
                    if "format" in sub:
                        flat["data_format"] = sub["format"]
                    if "doc_starts" in sub:
                        flat["doc_starts_path"] = sub["doc_starts"]
                    if "val_path" in sub:
                        flat["val_path"] = sub["val_path"]
                    if "val_doc_starts" in sub:
                        flat["val_doc_starts_path"] = sub["val_doc_starts"]
                else:
                    flat.update(sub)
        # gate_warmup section (09_gate_warmup.md)
        if "gate_warmup" in raw:
            gw = raw["gate_warmup"]
            if "theta_floor_init" in gw:
                flat["gate_warmup_theta_floor_init"] = gw["theta_floor_init"]
            if "gate_warmup_decay_steps" in gw:
                flat["gate_warmup_decay_steps"] = gw["gate_warmup_decay_steps"]
            if "falsification_step" in gw:
                flat["gate_warmup_falsification_step"] = gw["falsification_step"]
            if "l2_theta_threshold" in gw:
                flat["gate_warmup_l2_threshold"] = gw["l2_theta_threshold"]
            if "l3_theta_threshold" in gw:
                flat["gate_warmup_l3_threshold"] = gw["l3_theta_threshold"]
        # Top-level overrides (for flat configs)
        # promotion section (12_metric_driven_promotion.md)
        if "promotion" in raw:
            pm = raw["promotion"]
            for pk in ("auto_promote", "target_k", "promotion_cooldown",
                       "promotion_stability_window", "promotion_stability_streak",
                       "promotion_stability_threshold", "promotion_rewind_pct"):
                if pk in pm:
                    flat[pk] = pm[pk]
        for key in list(raw.keys()):
            if key not in ("model", "build", "data", "gate_warmup", "promotion",
                           "notes", "description"):
                flat[key] = raw[key]
        # Rename head_dim if present (derived, not stored)
        flat.pop("head_dim", None)
        flat.pop("format", None)
        # Auto-load vocab_size from meta/manifest for BPE formats
        if flat.get("data_format") in ("sharegpt", "dolmino") and "data_path" in flat:
            meta_path = Path(flat["data_path"]) / "meta.json"
            if meta_path.exists() and "vocab_size" not in flat:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("vocab_size") is not None:
                    flat["vocab_size"] = meta["vocab_size"]
                else:
                    raise ValueError(
                        f"meta.json at {meta_path} missing 'vocab_size' key")
        if flat.get("data_format") == "smollm" and "data_path" in flat:
            manifest_path = Path(flat["data_path"]) / "manifest.json"
            if manifest_path.exists() and "vocab_size" not in flat:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                vs = manifest.get("tokenizer", {}).get("vocab_size")
                if vs is not None:
                    flat["vocab_size"] = vs
                else:
                    raise ValueError(
                        f"manifest.json at {manifest_path} missing "
                        f"'tokenizer.vocab_size' key")
        # Spec 32 backward compat: eval_every → save_every migration
        _deprecated_checkpoint_fields(flat)
        return cls(**flat)

    def apply_cli(self, args: argparse.Namespace):
        """Apply CLI argument overrides (only non-None values)."""
        mapping = {
            "data": "data_path", "d_model": "d_model", "num_heads": "num_heads",
            "seq_len": "seq_len", "window_size": "window_size", "k": "k",
            "chunk_sizes": "chunk_sizes", "memory_rule": "memory_rule",
            "composition": "composition", "lr": "lr", "steps": "steps",
            "seed": "seed", "save_path": "save_path", "save_every": "save_every",
            "log_every": "log_every", "optimizer": "optimizer",
            "warmup_steps": "warmup_steps", "weight_decay": "weight_decay",
            "beta1": "beta1", "beta2": "beta2", "max_grad_norm": "max_grad_norm",
            "load": "load", "log_file": "log_file",
            # eval_every handled separately below (must not overwrite explicit --save_every)
            "checkpoint_interval": "checkpoint_interval",
            "batch_size": "batch_size",
            "projection_kind": "projection_kind",
            "self_ref_chunk_size": "self_ref_chunk_size",
            "momentum_kind": "momentum_kind",
            "momentum_d_hidden": "momentum_d_hidden",
            "attentional_bias": "attentional_bias",
            "retention": "retention",
            "extend_k": "extend_k",
            "data_seek": "data_seek",
            "run_dir": "run_dir",
        }
        for cli_name, cfg_name in mapping.items():
            val = getattr(args, cli_name, None)
            if val is not None:
                if cli_name == "chunk_sizes" and isinstance(val, str):
                    val = [int(x) for x in val.split(",") if x]
                setattr(self, cfg_name, val)
        # --eval_every backward compat: only apply if --save_every not explicitly set
        cli_eval = getattr(args, "eval_every", None)
        if cli_eval is not None and getattr(args, "save_every", None) is None:
            warnings.warn(
                f"--eval_every={cli_eval} is deprecated (spec 32). "
                f"Mapped to save_every={cli_eval}. Use --save_every directly.",
                DeprecationWarning, stacklevel=2,
            )
            self.save_every = cli_eval
        # --cpu overrides the default GPU mode
        if getattr(args, "cpu", False):
            self.gpu = False
        elif getattr(args, "gpu", False):
            self.gpu = True  # backward compat: --gpu still works
        # store_true with default=None: only override if explicitly passed
        if getattr(args, "self_generated_values", None) is not None:
            self.self_generated_values = args.self_generated_values
        if getattr(args, "push_up", None) is not None:
            self.push_up = args.push_up
        if getattr(args, "stack_up", None) is not None:
            self.stack_up = args.stack_up
        self._validate()
        self.validate_gpu_tier()  # V-05: checked after --cpu/--gpu are applied

    def validate_gpu_tier(self) -> None:
        """V-05: raise if GPU is requested but the rule has no GPU kernels.

        Separate from _validate() so --cpu CLI override is applied first.
        Call explicitly after from_file() when no CLI overrides are used
        (e.g., in the validate-config subcommand).
        """
        if self.gpu and self.memory_rule not in _GPU_CAPABLE:
            tier = "3" if self.memory_rule in _TIER_3_RULES else "2b"
            raise ValueError(
                f"'{self.memory_rule}' is Tier {tier} — no GPU kernels available.\n"
                f"  This combination runs on CPU only. Either:\n"
                f"    (a) add \"gpu\": false to your config build section, or\n"
                f"    (b) pass --cpu at the command line, or\n"
                f"    (c) use a GPU-capable memory_rule (titans, delta, hebbian)\n"
                f"       to run on GPU.\n"
                f"  See specs/infrastructure/01_variant_tier_policy.md for the full tier matrix.")


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_peak: float,
              lr_min: float = 0.0) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return lr_peak * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))
