"""BuildConfig and learning rate schedules."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


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

    # Memory enable flag (False = SWA-only baseline, no CMS memory modules)
    memory_enabled: bool = True

    # Per-level theta gate clamps (empty = unclamped, CS-39 style)
    theta_floor: list[float] | None = None  # per-level softplus lower bound
    theta_ceil: list[float] | None = None   # per-level softplus upper bound

    # Per-level M Frobenius norm ceiling (empty = disabled)
    m_norm_max: list[float] | None = None   # straight-through clamp after M update

    # SwiGluMlp / Llama level stacking (HOPE §7.3)
    intermediate_size: int = 0             # 0 for matrix rules; 8192 for Llama-3.2-1B
    donor_layers: list[int] | None = None  # which Llama layers to transplant (e.g. [0,5,10,15])
    donor_weights: str | None = None       # path to extracted donor weights .pt file

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

    # Data
    data_path: str | None = None
    data_format: str = "byte"  # "byte" or "sharegpt"
    doc_starts_path: str | None = None  # byte offsets of document boundaries
    val_path: str | None = None  # byte-level val corpus
    val_doc_starts_path: str | None = None  # val doc boundaries

    # Eval
    eval_every: int = 0  # 0 = disabled; evaluate on val set every N steps
    eval_max_chunks: int = 100  # max chunks per eval pass

    # Gradient checkpointing (VRAM optimization for memory rules)
    checkpoint_interval: int | None = None  # None = full trajectory; C = store M every C steps

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
        if self.optimizer not in ("sgd", "adamw", "adamw_gpu"):
            raise ValueError(
                f"optimizer must be 'sgd', 'adamw', or 'adamw_gpu', got '{self.optimizer}'")
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
        if self.momentum_d_hidden < 0:
            raise ValueError(
                f"momentum_d_hidden must be >= 0, got {self.momentum_d_hidden}")
        if self.self_generated_values and self.projection_kind != "adaptive":
            raise ValueError("self_generated_values requires projection_kind='adaptive'")
        if self.self_ref_chunk_size > 1 and self.projection_kind != "adaptive":
            raise ValueError("self_ref_chunk_size > 1 requires projection_kind='adaptive'")
        if self.data_format not in ("byte", "sharegpt"):
            raise ValueError(
                f"data_format must be 'byte' or 'sharegpt', got '{self.data_format}'")
        if self.theta_floor is not None and len(self.theta_floor) != self.k:
            raise ValueError(
                f"theta_floor length {len(self.theta_floor)} must match k={self.k}")
        if self.theta_ceil is not None and len(self.theta_ceil) != self.k:
            raise ValueError(
                f"theta_ceil length {len(self.theta_ceil)} must match k={self.k}")
        if self.m_norm_max is not None and len(self.m_norm_max) != self.k:
            raise ValueError(
                f"m_norm_max length {len(self.m_norm_max)} must match k={self.k}")
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
        for key in list(raw.keys()):
            if key not in ("model", "build", "data", "gate_warmup", "notes", "description"):
                flat[key] = raw[key]
        # Rename head_dim if present (derived, not stored)
        flat.pop("head_dim", None)
        flat.pop("format", None)
        # Auto-load vocab_size from meta.json for sharegpt format
        if flat.get("data_format") == "sharegpt" and "data_path" in flat:
            meta_path = Path(flat["data_path"]) / "meta.json"
            if meta_path.exists() and "vocab_size" not in flat:
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("vocab_size") is not None:
                    flat["vocab_size"] = meta["vocab_size"]
                else:
                    raise ValueError(
                        f"meta.json at {meta_path} missing 'vocab_size' key")
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
            "eval_every": "eval_every", "eval_max_chunks": "eval_max_chunks",
            "checkpoint_interval": "checkpoint_interval",
            "batch_size": "batch_size",
            "projection_kind": "projection_kind",
            "self_ref_chunk_size": "self_ref_chunk_size",
            "momentum_kind": "momentum_kind",
            "momentum_d_hidden": "momentum_d_hidden",
        }
        for cli_name, cfg_name in mapping.items():
            val = getattr(args, cli_name, None)
            if val is not None:
                if cli_name == "chunk_sizes" and isinstance(val, str):
                    val = [int(x) for x in val.split(",") if x]
                setattr(self, cfg_name, val)
        # --cpu overrides the default GPU mode
        if getattr(args, "cpu", False):
            self.gpu = False
        elif getattr(args, "gpu", False):
            self.gpu = True  # backward compat: --gpu still works
        # store_true with default=None: only override if explicitly passed
        if getattr(args, "self_generated_values", None) is not None:
            self.self_generated_values = args.self_generated_values
        self._validate()


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_peak: float,
              lr_min: float = 0.0) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return lr_peak * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))
