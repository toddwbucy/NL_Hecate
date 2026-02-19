#!/usr/bin/env python3
"""
NL-Hecate build script: train a model on text using stateful CMS, save checkpoint.

Usage:
    python build.py --config configs/toy_60m.json                    # from config file
    python build.py --config configs/toy_60m.json --lr 0.0005        # config + CLI override
    python build.py --data sample.txt --steps 500 --d_model 64       # pure CLI args
    python build.py --steps 200                                      # uses built-in demo text

All math stays in Rust. This script is pure orchestration (CS-18).
"""

import argparse
import json
import math
import mmap
import os
from pathlib import Path
import random
import struct
import time
from typing import Any, Optional

import nl_hecate


# ── Pydantic-style BuildConfig (no external dep — stdlib only) ───────

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

    # Eval
    eval_every: int = 0  # 0 = disabled; evaluate on val set every N steps
    eval_max_chunks: int = 100  # max chunks per eval pass

    # Runtime
    gpu: bool = False
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
        assert self.d_model > 0, "d_model must be positive"
        assert self.num_heads > 0, "num_heads must be positive"
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.window_size > 0, "window_size must be positive"
        assert self.k >= 1, "k must be >= 1"
        assert self.optimizer in ("sgd", "adamw", "adamw_gpu"), \
            f"optimizer must be 'sgd', 'adamw', or 'adamw_gpu', got '{self.optimizer}'"
        assert self.lr > 0, "lr must be positive"
        assert self.max_grad_norm >= 0, "max_grad_norm must be >= 0"
        if self.chunk_sizes is None:
            self.chunk_sizes = [1] * self.k
        assert len(self.chunk_sizes) == self.k, \
            f"chunk_sizes length {len(self.chunk_sizes)} must match k={self.k}"

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
                else:
                    flat.update(sub)
        # Top-level overrides (for flat configs)
        for key in list(raw.keys()):
            if key not in ("model", "build", "data", "notes", "description"):
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
                flat["vocab_size"] = meta["vocab_size"]
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
        }
        for cli_name, cfg_name in mapping.items():
            val = getattr(args, cli_name, None)
            if val is not None:
                if cli_name == "chunk_sizes" and isinstance(val, str):
                    val = [int(x) for x in val.split(",") if x]
                setattr(self, cfg_name, val)
        if getattr(args, "gpu", False):
            self.gpu = True
        self._validate()


# ── AdamW optimizer (Python-side, operates on flat weight arrays) ────

class AdamW:
    """Decoupled weight decay optimizer (Loshchilov & Hutter, 2019).

    Maintains first/second moment estimates per parameter. Works on flat
    weight arrays from MAGParams.get_weights() / set_weights().
    """

    def __init__(self, num_params: int, lr: float = 4e-4,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.1):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [0.0] * num_params  # first moment
        self.v = [0.0] * num_params  # second moment
        self.t = 0  # step counter for bias correction

    def step(self, params: list[float], grads: list[float], lr: float) -> list[float]:
        """One AdamW update. Returns updated params."""
        self.t += 1
        b1, b2, eps, wd = self.beta1, self.beta2, self.eps, self.weight_decay
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        for i in range(len(params)):
            g = grads[i]
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            # AdamW: decoupled weight decay applied to param directly
            params[i] -= lr * (m_hat / (math.sqrt(v_hat) + eps) + wd * params[i])
        return params


def cosine_lr(step: int, warmup_steps: int, total_steps: int, lr_peak: float,
              lr_min: float = 0.0) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return lr_peak * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))


def grad_norm(grads: list[float]) -> float:
    """Compute L2 norm of gradient vector."""
    return math.sqrt(sum(g * g for g in grads))


def clip_grad_norm(grads: list[float], max_norm: float) -> tuple[list[float], float]:
    """Clip gradient vector to max L2 norm. Returns (clipped_grads, original_norm)."""
    norm = grad_norm(grads)
    if norm > max_norm > 0:
        scale = max_norm / norm
        return [g * scale for g in grads], norm
    return grads, norm


# ── Byte-level tokenizer ────────────────────────────────────────────

def encode(text: str) -> list[int]:
    """Encode text to byte-level token IDs (vocab_size=256)."""
    return list(text.encode("utf-8"))


def decode(token_ids: list[int]) -> str:
    """Decode byte-level token IDs back to text."""
    out = []
    for b in token_ids:
        if 32 <= b < 127:
            out.append(chr(b))
        elif b == 10:
            out.append("\n")
        else:
            out.append("?")
    return "".join(out)


# ── File-backed token stream ────────────────────────────────────────

class MmapTokenStream:
    """Memory-mapped token stream for datasets that exceed RAM.

    Maps a binary file (one byte = one token) via mmap. Random access
    without loading the full file. Implements the same interface as
    a list[int] for indexing.
    """

    def __init__(self, path: str):
        self._f = open(path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._len = self._mm.size()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self._mm[idx])
        return self._mm[idx]

    def close(self):
        self._mm.close()
        self._f.close()


# ── BPE data loader for ShareGPT format ────────────────────────────

class BpeDataLoader:
    """Load pre-tokenized ShareGPT data (numpy arrays) and serve chunks.

    Manages a position cursor into the flat token/target arrays.
    Returns (input_ids, target_ids) per chunk, wrapping at corpus end.
    Masked targets: -1 in numpy → vocab_size as Python int (triggers
    kernel skip via target >= vocab).
    """

    def __init__(self, data_dir: str, split: str = "train"):
        import numpy as np
        data_path = Path(data_dir)
        self.tokens = np.load(data_path / f"{split}_tokens.npy")
        self.targets = np.load(data_path / f"{split}_targets.npy")
        assert len(self.tokens) == len(self.targets), \
            f"tokens ({len(self.tokens)}) != targets ({len(self.targets)})"

        with open(data_path / "meta.json") as f:
            self.meta = json.load(f)
        self.vocab_size = self.meta["vocab_size"]
        self.position = 0
        self.total_tokens = len(self.tokens)

    def next_chunk(self, seq_len: int) -> tuple[list[int], list[int]] | None:
        """Get next chunk of (input_ids, target_ids).

        Returns None if remaining tokens < seq_len (wraps on next call).
        Masked targets (-1) are converted to vocab_size for the kernel.
        """
        if self.position + seq_len > self.total_tokens:
            self.position = 0  # wrap
        if self.total_tokens < seq_len:
            return None

        end = self.position + seq_len
        input_ids = self.tokens[self.position:end].tolist()
        raw_targets = self.targets[self.position:end]

        # Convert -1 (masked) → vocab_size (kernel skip sentinel)
        target_ids = []
        for t in raw_targets:
            target_ids.append(int(t) if t >= 0 else self.vocab_size)

        self.position = end
        return input_ids, target_ids

    def __len__(self) -> int:
        return self.total_tokens


# ── JSONL logger ────────────────────────────────────────────────────

class JSONLLogger:
    """Append-only structured logger. One JSON object per line."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")

    def log(self, **fields):
        fields["timestamp"] = time.time()
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ── Default demo text ───────────────────────────────────────────────

DEMO_TEXT = (
    "the cat sat on the mat. "
    "the dog ran in the park. "
    "birds fly high in the sky. "
    "fish swim deep in the sea. "
) * 10


def load_binary_tokens(path: str) -> list[int]:
    """Load a binary file where each byte IS a token ID."""
    with open(path, "rb") as f:
        return list(f.read())


def evaluate(gpu_model, bcfg: BuildConfig, val_loader: "BpeDataLoader",
             max_chunks: int) -> tuple[float, float]:
    """Run forward-only eval on val set. Returns (avg_loss, perplexity).

    Uses a fresh Conductor per eval (independent CMS state) so eval
    doesn't corrupt training context. No backward pass, no weight update.
    """
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
    context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)

    # Reset val loader position for deterministic eval
    val_loader.position = 0

    total_loss = 0.0
    n_chunks = 0

    for _ in range(max_chunks):
        chunk = val_loader.next_chunk(bcfg.seq_len)
        if chunk is None:
            break
        input_ids, target_ids = chunk
        pulse = conductor.pulse()

        if gpu_model is not None:
            loss, _ = gpu_model.forward(input_ids, target_ids, pulse)
        else:
            # CPU eval path
            params_ref = None  # caller must handle CPU case separately
            raise NotImplementedError("CPU eval not yet implemented for BPE")

        if math.isnan(loss) or math.isinf(loss):
            continue

        total_loss += loss
        n_chunks += 1
        conductor.advance()

    if n_chunks == 0:
        return 0.0, 1.0

    avg_loss = total_loss / n_chunks
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


# Fixed prompts for sampling at checkpoints (tests different capabilities)
SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain how a neural network learns in simple terms.",
    "Write a short poem about the ocean.",
]


def generate_samples(gpu_model, cfg, tokenizer, step: int,
                     temperature: float = 0.7,
                     max_tokens: int = 128) -> list[dict]:
    """Generate sample completions at checkpoint time.

    Uses serve.generate() for autoregressive decoding. Returns list of
    dicts with prompt, completion, and token count for JSONL logging.
    """
    from serve import generate

    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        prompt_tokens = tokenizer.encode(prompt_text)
        output_tokens = generate(
            params=None,  # not needed when gpu_model is provided
            cfg=cfg,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cms=True,
            gpu_model=gpu_model,
        )
        # Decode only the generated portion
        gen_tokens = output_tokens[len(prompt_tokens):]
        completion = tokenizer.decode(gen_tokens)
        samples.append({
            "prompt": prompt_text,
            "completion": completion,
            "gen_tokens": len(gen_tokens),
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description="NL-Hecate build script")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (overrides CLI defaults)")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (.txt or .bin)")
    parser.add_argument("--steps", type=int, default=None, help="Build steps")
    parser.add_argument("--d_model", type=int, default=None, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length")
    parser.add_argument("--window_size", type=int, default=None, help="SWA window size")
    parser.add_argument("--k", type=int, default=None, help="CMS frequency levels")
    parser.add_argument("--chunk_sizes", type=str, default=None,
                        help="Comma-separated chunk sizes per level (e.g. '1,8')")
    parser.add_argument("--memory_rule", type=str, default=None, help="Memory rule")
    parser.add_argument("--composition", type=str, default=None, help="Composition pattern")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Checkpoint save path")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--log_every", type=int, default=None, help="Log every N steps")
    parser.add_argument("--load", type=str, default=None,
                        help="Resume from a build checkpoint (saved with --save_every)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU-resident model (all math on GPU)")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Optimizer: 'sgd' or 'adamw' (default: sgd)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="LR warmup steps (0 = no warmup)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="AdamW weight decay")
    parser.add_argument("--beta1", type=float, default=None, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=None, help="AdamW beta2")
    parser.add_argument("--max_grad_norm", type=float, default=None,
                        help="Max gradient L2 norm for clipping (0 = disabled)")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path for structured JSONL log (e.g. runs/build.jsonl)")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Evaluate on val set every N steps (0 = disabled)")
    parser.add_argument("--eval_max_chunks", type=int, default=None,
                        help="Max chunks per eval pass (default: 100)")
    args = parser.parse_args()

    # ── Build config: file → defaults, then CLI overrides ─────────────
    if args.config:
        bcfg = BuildConfig.from_file(args.config)
        print(f"Loaded config: {args.config}")
    else:
        bcfg = BuildConfig()
    bcfg.apply_cli(args)

    # ── Load data ─────────────────────────────────────────────────────
    use_bpe = (bcfg.data_format == "sharegpt")
    bpe_loader: BpeDataLoader | None = None
    token_ids: list[int] | MmapTokenStream | None = None

    val_loader: BpeDataLoader | None = None

    if use_bpe:
        bpe_loader = BpeDataLoader(bcfg.data_path, split="train")
        print(f"Loaded ShareGPT BPE data: {len(bpe_loader):,} tokens, "
              f"vocab={bpe_loader.vocab_size}")
        if len(bpe_loader) < bcfg.seq_len:
            print(f"Error: data too short ({len(bpe_loader)} tokens < seq_len={bcfg.seq_len})")
            return
        # Load val set if eval is enabled
        if bcfg.eval_every > 0:
            val_path = Path(bcfg.data_path) / "val_tokens.npy"
            if val_path.exists():
                val_loader = BpeDataLoader(bcfg.data_path, split="val")
                print(f"Loaded val set: {len(val_loader):,} tokens")
            else:
                print("Warning: eval_every set but no val data found, disabling eval")
                bcfg.eval_every = 0
    elif bcfg.data_path:
        if bcfg.data_path.endswith(".bin"):
            fsize = os.path.getsize(bcfg.data_path)
            if fsize > 500_000_000:  # >500MB: use mmap
                token_ids = MmapTokenStream(bcfg.data_path)
                print(f"Memory-mapped {len(token_ids):,} byte tokens from {bcfg.data_path}")
            else:
                token_ids = load_binary_tokens(bcfg.data_path)
                print(f"Loaded {len(token_ids):,} byte tokens from {bcfg.data_path}")
        else:
            with open(bcfg.data_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Loaded {len(text):,} chars from {bcfg.data_path}")
            token_ids = encode(text)
    else:
        text = DEMO_TEXT
        print(f"Using built-in demo text ({len(text):,} chars)")
        token_ids = encode(text)

    if not use_bpe and token_ids is not None and len(token_ids) < bcfg.seq_len + 1:
        print(f"Error: text too short ({len(token_ids)} tokens < seq_len+1={bcfg.seq_len + 1})")
        return

    # ── Load tokenizer for sample generation ──────────────────────────
    tokenizer = None
    if use_bpe and bcfg.save_every > 0:
        from serve import load_tokenizer, BpeTokenizer
        tokenizer = load_tokenizer(data_dir=bcfg.data_path)
        tok_type = "BPE" if isinstance(tokenizer, BpeTokenizer) else "byte-level"
        print(f"Tokenizer for samples: {tok_type}")

    # ── Resume from checkpoint or init fresh ──────────────────────────
    resume_step = 0
    if bcfg.load:
        print(f"Loading build checkpoint: {bcfg.load}")
        params, cfg, build_state = nl_hecate.load_build_checkpoint(bcfg.load)
        if build_state is None:
            print("Error: checkpoint has no build state (not a build checkpoint)")
            return
        resume_step = build_state["global_step"]
        bcfg.d_model = cfg.d_model
        bcfg.num_heads = cfg.num_heads
        bcfg.k = cfg.k
        bcfg.chunk_sizes = list(cfg.chunk_sizes)
        bcfg.seq_len = cfg.seq_len
        print(f"  Resuming from step {resume_step}")
        print(f"  Stream position: {build_state['stream_position']}")
    else:
        cfg = nl_hecate.MAGConfig(
            d_model=bcfg.d_model,
            num_heads=bcfg.num_heads,
            head_dim=bcfg.head_dim,
            seq_len=bcfg.seq_len,
            window_size=bcfg.window_size,
            vocab_size=bcfg.vocab_size,
            memory_enabled=True,
            k=bcfg.k,
            chunk_sizes=bcfg.chunk_sizes,
            memory_rule=bcfg.memory_rule,
            composition=bcfg.composition,
        )
        params = nl_hecate.mag_init_params(cfg, bcfg.seed)

    print(f"\n{'=' * 60}")
    print("NL-Hecate Build")
    print(f"{'=' * 60}")
    print(f"  Model:    d={bcfg.d_model}, heads={bcfg.num_heads}, "
          f"seq_len={bcfg.seq_len}, vocab={bcfg.vocab_size}")
    print(f"  Memory:   rule={cfg.memory_rule}, composition={cfg.composition}, k={bcfg.k}")
    print(f"  CMS:      chunk_sizes={bcfg.chunk_sizes}")
    print(f"  Params:   {params.num_params():,}")
    data_len = len(bpe_loader) if use_bpe else len(token_ids)
    print(f"  Data:     {data_len:,} tokens" +
          (f" (ShareGPT BPE, {bcfg.data_format})" if use_bpe else ""))
    use_gpu = bcfg.gpu and hasattr(nl_hecate, "GpuModel")
    if bcfg.optimizer == "adamw_gpu" and not use_gpu:
        raise RuntimeError(
            "optimizer=adamw_gpu requires --gpu and a CUDA-enabled build"
        )
    if bcfg.load and use_gpu:
        raise RuntimeError(
            "GPU resume with context restore is not yet implemented. "
            "Use CPU resume (--gpu omitted) or start a fresh GPU build."
        )
    print(f"  Build:    {bcfg.steps} steps (from step {resume_step}), lr={bcfg.lr}")
    print(f"  Optimizer: {bcfg.optimizer}" +
          (f" (b1={bcfg.beta1}, b2={bcfg.beta2}, wd={bcfg.weight_decay}, warmup={bcfg.warmup_steps})"
           if bcfg.optimizer in ("adamw", "adamw_gpu") else ""))
    if bcfg.max_grad_norm > 0:
        print(f"  Grad clip: max_norm={bcfg.max_grad_norm}")
    print(f"  Device:   {'GPU' if use_gpu else 'CPU'}")
    if bcfg.eval_every > 0:
        print(f"  Eval:     every {bcfg.eval_every} steps, {bcfg.eval_max_chunks} max chunks")
    if bcfg.log_file:
        print(f"  Log:      {bcfg.log_file}")
    print(f"{'=' * 60}\n")

    # ── Stateful CMS build loop ───────────────────────────────────────
    if use_bpe:
        # ShareGPT BPE: Conductor in pulse-only mode (no VecStream).
        # BpeDataLoader handles data; Conductor only generates CMS pulses.
        conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
        context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
    else:
        # Byte-level: VecStream + Conductor for integrated data + pulse.
        if isinstance(token_ids, MmapTokenStream):
            mm = token_ids
            token_ids = list(mm)
            mm.close()
        if bcfg.load:
            conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
            stream = nl_hecate.VecStream(token_ids)
            conductor.attach_stream(stream)
            conductor.restore_from_dict(build_state)
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
            context.set_memory(build_state["context_memory"])
        else:
            conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
            stream = nl_hecate.VecStream(token_ids)
            conductor.attach_stream(stream)
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)

    # GPU-resident model: upload params once, all math on device
    gpu_model = None
    if use_gpu:
        gpu_model = nl_hecate.GpuModel.from_params(params, cfg)

    error_buffers = nl_hecate.ErrorBufferList(bcfg.k, bcfg.d_model)

    # Initialize optimizer
    adamw_opt = None
    use_adamw_gpu = (bcfg.optimizer == "adamw_gpu")
    if bcfg.optimizer == "adamw":
        adamw_opt = AdamW(
            num_params=params.num_params(), lr=bcfg.lr,
            beta1=bcfg.beta1, beta2=bcfg.beta2, weight_decay=bcfg.weight_decay,
        )

    # Structured logger
    jsonl: Optional[JSONLLogger] = None
    if bcfg.log_file:
        jsonl = JSONLLogger(bcfg.log_file)
        jsonl.log(event="build_start", config={
            "d_model": bcfg.d_model, "num_heads": bcfg.num_heads,
            "seq_len": bcfg.seq_len, "k": bcfg.k, "memory_rule": bcfg.memory_rule,
            "composition": bcfg.composition, "optimizer": bcfg.optimizer,
            "lr": bcfg.lr, "steps": bcfg.steps, "params": params.num_params(),
        })

    losses = []
    t_start = time.perf_counter()
    end_step = resume_step + bcfg.steps

    for step in range(resume_step, end_step):
        if use_bpe:
            # ShareGPT BPE path: data from BpeDataLoader, pulse from Conductor
            chunk = bpe_loader.next_chunk(bcfg.seq_len)
            if chunk is None:
                break
            input_ids, target_ids = chunk
            pulse = conductor.pulse()
        else:
            # Byte-level path: integrated data + pulse from Conductor+VecStream
            result = conductor.next_chunk(bcfg.seq_len)
            if result is None:
                break
            input_ids, target_ids, pulse = result

            # Skip truncated chunks at corpus boundary
            if len(input_ids) != bcfg.seq_len:
                conductor.advance()
                continue

        # Compute current learning rate
        use_cosine = (adamw_opt is not None or use_adamw_gpu)
        current_lr = cosine_lr(step, bcfg.warmup_steps, end_step, bcfg.lr) if use_cosine else bcfg.lr

        g_norm = 0.0

        if gpu_model is not None and use_adamw_gpu:
            # Full GPU AdamW: forward + backward + optimizer all on device
            loss, g_norm = gpu_model.step_adamw(
                input_ids, target_ids, pulse, current_lr,
                beta1=bcfg.beta1, beta2=bcfg.beta2, eps=1e-8,
                weight_decay=bcfg.weight_decay,
                max_grad_norm=bcfg.max_grad_norm,
            )
        elif gpu_model is not None and adamw_opt is None:
            # GPU path with SGD: forward + backward + update in one call
            loss = gpu_model.step(input_ids, target_ids, pulse, current_lr)
        elif gpu_model is not None and adamw_opt is not None:
            # Hybrid GPU+AdamW: GPU forward+backward, Python optimizer
            loss, grad_params = gpu_model.backward_only(input_ids, target_ids, pulse)
            p_flat = params.get_flat_weights()
            g_flat = grad_params.get_flat_weights()
            if bcfg.max_grad_norm > 0:
                g_flat, g_norm = clip_grad_norm(g_flat, bcfg.max_grad_norm)
            else:
                g_norm = grad_norm(g_flat)
            p_flat = adamw_opt.step(p_flat, g_flat, current_lr)
            params.set_flat_weights(p_flat)
            # Weight tying: sync w_unembed^T → w_embed (same as SGD path)
            nl_hecate.mag_apply_weight_gradients(params, grad_params, 0.0)
            gpu_model.upload_params(params)
            error_buffers.apply_for_active(params, pulse, current_lr)
        else:
            # CPU path: tape-based forward + backward (single call)
            loss, grads = nl_hecate.cms_compute_gradients(
                params, cfg, input_ids, target_ids, pulse, context,
                error_buffers)
            if adamw_opt:
                p_flat = params.get_flat_weights()
                g_flat = grads.get_flat_weights()
                if bcfg.max_grad_norm > 0:
                    g_flat, g_norm = clip_grad_norm(g_flat, bcfg.max_grad_norm)
                else:
                    g_norm = grad_norm(g_flat)
                p_flat = adamw_opt.step(p_flat, g_flat, current_lr)
                params.set_flat_weights(p_flat)
                # Weight tying: sync w_unembed^T → w_embed
                nl_hecate.mag_apply_weight_gradients(params, grads, 0.0)
            else:
                nl_hecate.mag_apply_weight_gradients(params, grads, current_lr)
            error_buffers.apply_for_active(params, pulse, current_lr)

        # NaN/Inf guard
        if math.isnan(loss) or math.isinf(loss):
            print(f"  step {step:5d}  loss={loss} — ABORTING (NaN/Inf detected)")
            if jsonl:
                jsonl.log(event="abort", step=step, reason="nan_inf", loss=float(loss))
            break

        # Advance conductor (CS-32: observe-then-advance)
        conductor.advance()
        losses.append(loss)

        # Compute perplexity
        ppl = math.exp(min(loss, 20.0))

        # Logging
        if step % bcfg.log_every == 0 or step == end_step - 1:
            msg = f"  step {step:5d}  loss={loss:.4f}  ppl={ppl:.1f}"
            if g_norm > 0:
                msg += f"  gnorm={g_norm:.4f}"
            if adamw_opt or use_adamw_gpu:
                msg += f"  lr={current_lr:.6f}"
            print(msg)

        # Structured JSONL log
        if jsonl and (step % bcfg.log_every == 0 or step == end_step - 1):
            log_fields: dict[str, Any] = dict(
                event="step", step=step, loss=loss, ppl=ppl,
                grad_norm=g_norm, lr=current_lr,
                elapsed=time.perf_counter() - t_start,
                active_levels=pulse.active_levels,
            )
            # Masked token ratio (BPE only — byte-level has no masking)
            if use_bpe:
                n_masked = sum(1 for t in target_ids if t >= bcfg.vocab_size)
                log_fields["masked_ratio"] = n_masked / len(target_ids)
            # Gate biases from GPU (small D2H: 3 floats per level)
            if gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                log_fields["gate_biases"] = gpu_model.gate_biases()
            jsonl.log(**log_fields)

        # Periodic eval on val set
        if (bcfg.eval_every > 0 and val_loader is not None
                and step > 0 and step % bcfg.eval_every == 0):
            eval_loss, eval_ppl = evaluate(
                gpu_model, bcfg, val_loader, bcfg.eval_max_chunks)
            print(f"  [eval] step {step:5d}  loss={eval_loss:.4f}  ppl={eval_ppl:.1f}")
            if jsonl:
                jsonl.log(event="eval", step=step, eval_loss=eval_loss,
                          eval_ppl=eval_ppl, eval_chunks=bcfg.eval_max_chunks)

        # Periodic checkpoint
        if bcfg.save_every > 0 and step > 0 and step % bcfg.save_every == 0:
            # Download from GPU if needed
            if gpu_model is not None:
                params = gpu_model.to_host_params()
                context = gpu_model.to_host_context()
            p = Path(bcfg.save_path)
            ckpt_path = str(p.with_stem(f"{p.stem}_step{step}"))
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            if use_bpe:
                # BPE mode: no VecStream, save params-only checkpoint
                nl_hecate.save_checkpoint(ckpt_path, params, cfg)
            else:
                # Byte-level: resumable checkpoint with build state
                nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)
            print(f"  [checkpoint saved: {ckpt_path}]")

            # Generate samples at checkpoint time
            if tokenizer is not None and gpu_model is not None:
                try:
                    samples = generate_samples(gpu_model, cfg, tokenizer, step)
                    for s in samples:
                        preview = s["completion"][:80].replace("\n", " ")
                        print(f"  [sample] {s['prompt'][:40]}... → {preview}...")
                    if jsonl:
                        jsonl.log(event="sample", step=step, samples=samples)
                except Exception as e:
                    print(f"  [sample generation failed: {e}]")

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    total_tokens = len(losses) * bcfg.seq_len
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # ── Final checkpoint ──────────────────────────────────────────────
    if gpu_model is not None:
        params = gpu_model.to_host_params()
    os.makedirs(os.path.dirname(bcfg.save_path) or ".", exist_ok=True)
    nl_hecate.save_checkpoint(bcfg.save_path, params, cfg)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Build complete")
    print(f"{'=' * 60}")
    print(f"  Steps:     {len(losses)}")
    print(f"  Time:      {elapsed:.2f}s")
    print(f"  Tok/s:     {tok_per_sec:,.0f}")
    if losses:
        print(f"  Loss:      {losses[0]:.4f} -> {losses[-1]:.4f}")
        avg_first = sum(losses[:10]) / min(10, len(losses))
        avg_last = sum(losses[-10:]) / min(10, len(losses))
        print(f"  Avg loss:  first10={avg_first:.4f}, last10={avg_last:.4f}")
    print(f"  Saved:     {bcfg.save_path}")
    print(f"{'=' * 60}")

    if jsonl:
        try:
            jsonl.log(event="build_end", steps=len(losses), elapsed=elapsed,
                      tok_per_sec=tok_per_sec,
                      loss_first=losses[0] if losses else None,
                      loss_last=losses[-1] if losses else None)
        finally:
            jsonl.close()


if __name__ == "__main__":
    main()
