"""Build loop: stateful CMS training with optional GPU acceleration."""

import gc
import json
import sys
from pathlib import Path as _Path

# Ensure the python/ directory is on sys.path so `engine` is importable
# regardless of where the script is invoked from.
_pkg_root = _Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import nl_hecate

from engine.config import BuildConfig, cosine_lr
from engine.data import BpeTokenStream, CursorMismatchError, CursorOutOfBounds, DEMO_TEXT, MmapTokenStream
from engine.evaluation import (
    evaluate, evaluate_numpy, print_level_metrics, print_tape_summary,
    eval_coherence_samples, generate_samples,
    full_snapshot, full_restore,
    probe_within_generation, probe_cross_exposure, probe_context_value,
    probe_memory_vocab,
    EVAL_PROMPTS, SAMPLE_PROMPTS,
)
from engine.logging_utils import JSONLLogger, rss_mb
from engine.tokenizer import ByteTokenizer, BpeTokenizer, load_tokenizer


def _encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _safetensors_path(path_str: str) -> str:
    """Convert .json checkpoint path to .safetensors (transparent migration).
    Non-.json paths are returned unchanged."""
    if path_str.endswith(".json"):
        return path_str[:-5] + ".safetensors"
    return path_str


# Data formats that use the numpy/BPE loader path (BpeTokenStream + sidecar cursor).
# The byte-level path (VecStream / MmapTokenStream) uses a separate resume mechanism
# and is unaffected by this constant.
_NUMPY_LOADER_FORMATS = frozenset({"sharegpt", "dolmino"})


def run_build(bcfg: BuildConfig):
    """Execute the full build loop. All state managed internally."""

    import numpy as np

    # ── Load data ─────────────────────────────────────────────────────
    use_bpe = bcfg.data_format in _NUMPY_LOADER_FORMATS
    active_loader: BpeTokenStream | None = None
    token_ids: list[int] | MmapTokenStream | None = None

    val_stream: BpeTokenStream | None = None

    if use_bpe:
        active_loader = BpeTokenStream(bcfg.data_path, split="train")
        print(f"Loaded {bcfg.data_format} BPE data: {len(active_loader):,} tokens, "
              f"vocab={active_loader.vocab_size}")
        if len(active_loader) < bcfg.seq_len:
            print(f"Error: data too short ({len(active_loader)} tokens < seq_len={bcfg.seq_len}, "
                  f"format={bcfg.data_format})")
            return
        if bcfg.eval_every > 0:
            val_path = Path(bcfg.data_path) / "val_tokens.npy"
            if val_path.exists():
                val_stream = BpeTokenStream(bcfg.data_path, split="val")
                print(f"Loaded val set: {len(val_stream):,} tokens")
            else:
                print("Warning: eval_every set but no val data found, disabling eval")
                bcfg.eval_every = 0  # safe: bcfg is consumed only by this function
    elif bcfg.data_path:
        if bcfg.data_path.endswith(".bin"):
            fsize = os.path.getsize(bcfg.data_path)
            if fsize > 500_000_000:  # >500MB: use mmap
                token_ids = MmapTokenStream(bcfg.data_path)
                print(f"Memory-mapped {len(token_ids):,} byte tokens from {bcfg.data_path}")
            else:
                from engine.data import load_binary_tokens
                token_ids = load_binary_tokens(bcfg.data_path)
                print(f"Loaded {len(token_ids):,} byte tokens from {bcfg.data_path}")
        else:
            with open(bcfg.data_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Loaded {len(text):,} chars from {bcfg.data_path}")
            token_ids = _encode_bytes(text)
    else:
        text = DEMO_TEXT
        print(f"Using built-in demo text ({len(text):,} chars)")
        token_ids = _encode_bytes(text)

    # ── Load document boundaries (for doc-aware memory reset) ──────
    doc_starts = None
    if bcfg.doc_starts_path:
        doc_starts = np.load(bcfg.doc_starts_path).astype(np.uint64)
        print(f"Loaded {len(doc_starts):,} document boundaries from {bcfg.doc_starts_path}")

    # ── Load byte-level val data ─────────────────────────────────────
    val_bytes: bytes | None = None
    val_doc_starts = None
    if not use_bpe and bcfg.eval_every > 0 and bcfg.val_path:
        if os.path.exists(bcfg.val_path):
            with open(bcfg.val_path, "rb") as f:
                val_bytes = f.read()
            print(f"Loaded val corpus: {len(val_bytes):,} bytes from {bcfg.val_path}")
            if bcfg.val_doc_starts_path and os.path.exists(bcfg.val_doc_starts_path):
                val_doc_starts = np.load(bcfg.val_doc_starts_path).astype(np.uint64)
                print(f"Loaded {len(val_doc_starts):,} val document boundaries")
            val_stream = val_bytes  # evaluate() accepts bytes for byte-level
        else:
            print(f"Warning: val_path {bcfg.val_path} not found, disabling eval")
            bcfg.eval_every = 0

    if not use_bpe and token_ids is not None and len(token_ids) < bcfg.seq_len + 1:
        print(f"Error: text too short ({len(token_ids)} tokens < seq_len+1={bcfg.seq_len + 1})")
        return

    # ── Load tokenizer for sample generation ──────────────────────────
    tokenizer = None
    if use_bpe and (bcfg.save_every > 0 or bcfg.eval_every > 0):
        tokenizer = load_tokenizer(data_dir=bcfg.data_path)
        tok_type = "BPE" if isinstance(tokenizer, BpeTokenizer) else "byte-level"
        print(f"Tokenizer for samples: {tok_type}")

    # ── Resume from checkpoint or init fresh ──────────────────────────
    resume_step = 0
    build_state = None
    _restored_level_start_cursor: int | None = None
    _stacked_params_json: str | None = None  # set by stacked checkpoint load
    if bcfg.load:
        print(f"Loading checkpoint: {bcfg.load}")
        # Detect stacked vs single-block checkpoint
        _is_stacked_ckpt = (
            hasattr(nl_hecate, "is_stacked_checkpoint")
            and nl_hecate.is_stacked_checkpoint(bcfg.load)
        )
        if _is_stacked_ckpt:
            # Stacked checkpoint load (spec 22)
            result = nl_hecate.load_stacked_checkpoint(bcfg.load)
            _stacked_params_json = result["params_json"]
            cfg = result["config"]
            bcfg.n_blocks = result["n_blocks"]
            build_state = result["build_state"]
            # Stacked models use _stacked_params_json for GPU construction.
            # param count is read from the GpuStackedModel after construction.
            params = None
            if build_state is not None:
                resume_step = build_state["global_step"]
                print(f"  Stacked checkpoint: n_blocks={result['n_blocks']}, k={cfg.k}")
                print(f"  Resuming from step {resume_step}")
            else:
                print(f"  Stacked checkpoint: n_blocks={result['n_blocks']}, k={cfg.k} (no build state)")
            # Cursor sidecar for data position
            sidecar = Path(str(bcfg.load) + ".cursor.json")
            if sidecar.exists() and active_loader is not None:
                cursor = json.loads(sidecar.read_text())
                is_multi_slot = isinstance(cursor, dict) and isinstance(cursor.get("slots"), list)
                if not is_multi_slot:
                    try:
                        active_loader.restore(cursor)
                        print(f"  Stream position: {cursor['position']:,} / {cursor['total_tokens']:,} tokens")
                    except (CursorMismatchError, CursorOutOfBounds) as e:
                        print(f"  ERROR: cursor mismatch — {e}")
                        return
                if isinstance(cursor, dict) and "level_start_cursor" in cursor:
                    _restored_level_start_cursor = cursor["level_start_cursor"]
        elif use_bpe:
            # BPE path: try build checkpoint first, fall back to serving checkpoint.
            # If a cursor sidecar exists, this is a true resume (same data, same run).
            # If no sidecar, this is a warm-start (different run / donor weights).
            try:
                params, cfg, build_state = nl_hecate.load_build_checkpoint(bcfg.load)
                resume_step = build_state["global_step"] if build_state else 0
            except Exception:
                params, cfg = nl_hecate.load_checkpoint(bcfg.load)
                resume_step = 0
            sidecar = Path(str(bcfg.load) + ".cursor.json")
            if sidecar.exists() and active_loader is not None:
                cursor = json.loads(sidecar.read_text())
                # Multi-slot sidecars contain {"slots": [...]} — skip flat restore
                # here; the per-slot restore block below (bpe_loaders section)
                # handles them. Calling active_loader.restore() on a slots dict
                # would raise KeyError before that block could run.
                is_multi_slot = isinstance(cursor, dict) and isinstance(cursor.get("slots"), list)
                if not is_multi_slot:
                    try:
                        active_loader.restore(cursor)
                        print(f"  Resuming from step {resume_step}")
                        print(f"  Stream position: {cursor['position']:,} / {cursor['total_tokens']:,} tokens")
                    except (CursorMismatchError, CursorOutOfBounds) as e:
                        print(f"  ERROR: cursor mismatch — {e}")
                        return
                if is_multi_slot and len(cursor["slots"]) != bcfg.batch_size:
                    print(
                        "  ERROR: cursor sidecar was saved with "
                        f"{len(cursor['slots'])} slot(s) but current batch_size={bcfg.batch_size}")
                    return
                # Restore level_start_cursor if persisted (for correct rewind after resume).
                if isinstance(cursor, dict) and "level_start_cursor" in cursor:
                    _restored_level_start_cursor = cursor["level_start_cursor"]
                # else: multi-slot sidecar — handled in the bpe_loaders block below
            else:
                print("  Loaded checkpoint as warm-start (no cursor sidecar — data position reset to 0)")
        else:
            params, cfg, build_state = nl_hecate.load_build_checkpoint(bcfg.load)
            if build_state is None:
                print("Error: checkpoint has no build state (not a build checkpoint)")
                return
            resume_step = build_state["global_step"]
            print(f"  Resuming from step {resume_step}")
            print(f"  Stream position: {build_state['stream_position']}")
        # ── Push-up level stacking ────────────────────────────────────
        if bcfg.extend_k is not None:
            loaded_k = cfg.k
            target_k = bcfg.extend_k
            if target_k != loaded_k + 1:
                print(f"  ERROR: extend_k={target_k} must be loaded_k+1={loaded_k + 1}")
                return
            chunk_template = [1, 8, 64, 512]
            if target_k > len(chunk_template):
                print(f"  ERROR: extend_k={target_k} exceeds max supported k={len(chunk_template)}")
                return
            if bcfg.stack_up:
                # Stack-up: preserve donor's chunk sizes, append next tier
                canonical_prefix = chunk_template[:len(cfg.chunk_sizes)]
                if list(cfg.chunk_sizes) != canonical_prefix:
                    print(f"  ERROR: stack-up requires canonical chunk_sizes prefix "
                          f"{canonical_prefix}, got {list(cfg.chunk_sizes)}")
                    return
                new_chunks = list(cfg.chunk_sizes) + [chunk_template[target_k - 1]]
            else:
                # Push-up: use canonical template (levels shift frequencies)
                new_chunks = chunk_template[:target_k]
            # Extend m_norm_max to target_k if needed (prefer checkpoint, normalize [] → None)
            ext_m_norm = (bcfg.m_norm_max or None) if bcfg.m_norm_max is not None else (
                list(cfg.m_norm_max) if hasattr(cfg, 'm_norm_max') and cfg.m_norm_max else None
            )
            if ext_m_norm is not None and len(ext_m_norm) < target_k:
                ext_m_norm = list(ext_m_norm) + [ext_m_norm[-1]] * (target_k - len(ext_m_norm))
            # Extend error_clip to target_k if needed (prefer checkpoint, normalize [] → None)
            ext_error_clip = (bcfg.error_clip or None) if bcfg.error_clip is not None else (
                list(cfg.error_clip) if hasattr(cfg, 'error_clip') and cfg.error_clip else None
            )
            if ext_error_clip is not None and len(ext_error_clip) < target_k:
                ext_error_clip = list(ext_error_clip) + [ext_error_clip[-1]] * (target_k - len(ext_error_clip))
            # Rebuild MAGConfig with the new k (carry all other fields from loaded cfg)
            new_cfg = nl_hecate.MAGConfig(
                d_model=cfg.d_model, num_heads=cfg.num_heads,
                head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                memory_enabled=cfg.memory_enabled, k=target_k,
                chunk_sizes=new_chunks,
                memory_rule=cfg.memory_rule, composition=cfg.composition,
                checkpoint_interval=bcfg.checkpoint_interval,
                tape_multiplier=bcfg.tape_multiplier,
                projection_kind=cfg.projection_kind,
                self_generated_values=cfg.self_generated_values,
                self_ref_chunk_size=cfg.self_ref_chunk_size,
                momentum_kind=cfg.momentum_kind,
                momentum_d_hidden=cfg.momentum_d_hidden,
                attentional_bias=(
                    bcfg.attentional_bias
                    if bcfg.attentional_bias is not None
                    else getattr(cfg, "attentional_bias", None)
                ),
                retention=(
                    bcfg.retention
                    if bcfg.retention is not None
                    else getattr(cfg, "retention", None)
                ),
                intermediate_size=bcfg.intermediate_size,
                alpha_floor=bcfg.alpha_floor,
                alpha_ceil=bcfg.alpha_ceil,
                theta_floor=bcfg.theta_floor,
                theta_ceil=bcfg.theta_ceil,
                m_norm_max=ext_m_norm,
                error_clip=ext_error_clip,
                parallel_strategy=(
                    bcfg.parallel_strategy
                    if bcfg.parallel_strategy is not None
                    else getattr(cfg, "parallel_strategy", None)
                ),
                tnt_global_chunk_size=bcfg.tnt_global_chunk_size,
                tnt_local_chunk_size=bcfg.tnt_local_chunk_size,
                residual=bcfg.residual,
            )
            if _stacked_params_json is not None:
                # Stacked extend_k (spec 22): per-block level shift
                if not bcfg.push_up:
                    print("  ERROR: stacked extend_k only supports push_up (not stack_up)")
                    return
                result = nl_hecate.extend_stacked_push_up(
                    bcfg.load, new_cfg, bcfg.seed)
                _stacked_params_json = result["params_json"]
                print(f"  Stacked push-up: k={loaded_k} → k={target_k}, "
                      f"n_blocks={result['n_blocks']}, chunks={new_chunks}")
            elif bcfg.push_up:
                params = nl_hecate.extend_params_push_up(params, new_cfg, bcfg.seed)
                print(f"  Push-up: k={loaded_k} → k={target_k}, "
                      f"chunks={new_chunks}")
            elif bcfg.stack_up:
                params = nl_hecate.extend_params_stack_up(params, new_cfg, bcfg.seed)
                print(f"  Stack-up: k={loaded_k} → k={target_k}, "
                      f"chunks={new_chunks}")
            else:
                print("  ERROR: extend_k set but neither push_up nor stack_up — "
                      "set one of them to true")
                return
            cfg = new_cfg
            bcfg.k = target_k
            bcfg.chunk_sizes = new_chunks
            if ext_m_norm is not None:
                bcfg.m_norm_max = ext_m_norm
            if ext_error_clip is not None:
                bcfg.error_clip = ext_error_clip
            resume_step = 0
            build_state = None
            # New phase — persisted level_start_cursor from prior phase is stale.
            _restored_level_start_cursor = None

        # ── Data cursor override (for push-up or manual reposition) ──
        if bcfg.data_seek is not None and active_loader is not None:
            if bcfg.batch_size > 1:
                print("  ERROR: data_seek with batch_size>1 is not supported yet")
                return
            try:
                active_loader.restore({
                    "position": bcfg.data_seek,
                    "total_tokens": active_loader.total_tokens,
                    "content_hash": 0,  # skip hash validation on manual seek
                    "chunk_id": 0,
                    "seq_len": bcfg.seq_len,
                })
            except (CursorMismatchError, CursorOutOfBounds) as e:
                print(f"  ERROR: data_seek failed — {e}")
                return
            print(f"  Data cursor overridden → position {bcfg.data_seek:,}")
            # Manual seek starts a new phase — stale level_start_cursor would
            # include pre-seek tokens in rewind calculation.
            _restored_level_start_cursor = None

        bcfg.d_model = cfg.d_model
        bcfg.num_heads = cfg.num_heads
        bcfg.k = cfg.k
        bcfg.chunk_sizes = list(cfg.chunk_sizes)
        bcfg.seq_len = cfg.seq_len
        bcfg.projection_kind = cfg.projection_kind
        bcfg.momentum_kind = cfg.momentum_kind
        bcfg.self_generated_values = cfg.self_generated_values
        bcfg.self_ref_chunk_size = cfg.self_ref_chunk_size
        bcfg.momentum_d_hidden = cfg.momentum_d_hidden
        bcfg.residual = cfg.residual
        # Apply theta clamps from BuildConfig onto loaded cfg (allows
        # adding clamps to an existing checkpoint that didn't have them).
        # MAGConfig is frozen, so rebuild if clamps changed.
        if bcfg.alpha_floor is not None or bcfg.alpha_ceil is not None or bcfg.theta_floor is not None or bcfg.theta_ceil is not None or bcfg.m_norm_max is not None or bcfg.error_clip is not None:
            # Use bcfg value when explicitly set; fall back to loaded cfg so we
            # never silently wipe clamp values that were already baked into the
            # checkpoint (e.g. resuming without --m_norm_max still preserves it).
            alpha_floor = bcfg.alpha_floor if bcfg.alpha_floor is not None else list(cfg.alpha_floor)
            alpha_ceil  = bcfg.alpha_ceil  if bcfg.alpha_ceil  is not None else list(cfg.alpha_ceil)
            theta_floor = bcfg.theta_floor if bcfg.theta_floor is not None else list(cfg.theta_floor)
            theta_ceil  = bcfg.theta_ceil  if bcfg.theta_ceil  is not None else list(cfg.theta_ceil)
            m_norm_max  = bcfg.m_norm_max  if bcfg.m_norm_max  is not None else list(cfg.m_norm_max)
            error_clip  = bcfg.error_clip  if bcfg.error_clip  is not None else list(cfg.error_clip)
            cfg = nl_hecate.MAGConfig(
                d_model=cfg.d_model, num_heads=cfg.num_heads,
                head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                memory_enabled=cfg.memory_enabled, k=cfg.k,
                chunk_sizes=list(cfg.chunk_sizes),
                memory_rule=cfg.memory_rule, composition=cfg.composition,
                checkpoint_interval=bcfg.checkpoint_interval,
                tape_multiplier=bcfg.tape_multiplier,
                projection_kind=cfg.projection_kind,
                self_generated_values=cfg.self_generated_values,
                self_ref_chunk_size=cfg.self_ref_chunk_size,
                momentum_kind=cfg.momentum_kind,
                momentum_d_hidden=cfg.momentum_d_hidden,
                attentional_bias=(
                    bcfg.attentional_bias
                    if bcfg.attentional_bias is not None
                    else getattr(cfg, "attentional_bias", None)
                ),
                retention=(
                    bcfg.retention
                    if bcfg.retention is not None
                    else getattr(cfg, "retention", None)
                ),
                intermediate_size=bcfg.intermediate_size,
                alpha_floor=alpha_floor,
                alpha_ceil=alpha_ceil,
                theta_floor=theta_floor,
                theta_ceil=theta_ceil,
                m_norm_max=m_norm_max,
                error_clip=error_clip,
                parallel_strategy=(
                    bcfg.parallel_strategy
                    if bcfg.parallel_strategy is not None
                    else getattr(cfg, "parallel_strategy", None)
                ),
                tnt_global_chunk_size=(
                    bcfg.tnt_global_chunk_size
                    if bcfg.tnt_global_chunk_size is not None
                    else getattr(cfg, "tnt_global_chunk_size", None)
                ),
                tnt_local_chunk_size=(
                    bcfg.tnt_local_chunk_size
                    if bcfg.tnt_local_chunk_size is not None
                    else getattr(cfg, "tnt_local_chunk_size", None)
                ),
                residual=bcfg.residual,
            )
    else:
        cfg = nl_hecate.MAGConfig(
            d_model=bcfg.d_model,
            num_heads=bcfg.num_heads,
            head_dim=bcfg.head_dim,
            seq_len=bcfg.seq_len,
            window_size=bcfg.window_size,
            vocab_size=bcfg.vocab_size,
            memory_enabled=bcfg.memory_enabled,
            k=bcfg.k,
            chunk_sizes=bcfg.chunk_sizes,
            memory_rule=bcfg.memory_rule,
            composition=bcfg.composition,
            checkpoint_interval=bcfg.checkpoint_interval,
            tape_multiplier=bcfg.tape_multiplier,
            projection_kind=bcfg.projection_kind,
            self_generated_values=bcfg.self_generated_values,
            self_ref_chunk_size=bcfg.self_ref_chunk_size,
            momentum_kind=bcfg.momentum_kind,
            momentum_d_hidden=bcfg.momentum_d_hidden,
            attentional_bias=bcfg.attentional_bias,
            retention=bcfg.retention,
            alpha_floor=bcfg.alpha_floor,
            alpha_ceil=bcfg.alpha_ceil,
            theta_floor=bcfg.theta_floor,
            theta_ceil=bcfg.theta_ceil,
            intermediate_size=bcfg.intermediate_size,
            m_norm_max=bcfg.m_norm_max,
            error_clip=bcfg.error_clip,
            parallel_strategy=bcfg.parallel_strategy,
            tnt_global_chunk_size=bcfg.tnt_global_chunk_size,
            tnt_local_chunk_size=bcfg.tnt_local_chunk_size,
            residual=bcfg.residual,
        )
        params = nl_hecate.mag_init_params(cfg, bcfg.seed)
        if bcfg.donor_weights is not None:
            from engine.donor import load_llama_donor
            load_llama_donor(bcfg.donor_weights, params, cfg, bcfg.k)

    print(f"\n{'=' * 60}")
    print("NL-Hecate Build")
    print(f"{'=' * 60}")
    print(f"  Model:    d={bcfg.d_model}, heads={bcfg.num_heads}, "
          f"seq_len={bcfg.seq_len}, vocab={bcfg.vocab_size}")
    print(f"  Memory:   rule={cfg.memory_rule}, composition={cfg.composition}, k={bcfg.k}")
    print(f"  CMS:      chunk_sizes={bcfg.chunk_sizes}")
    if bcfg.checkpoint_interval:
        print(f"  GradCkpt: interval={bcfg.checkpoint_interval}")
    if bcfg.projection_kind == "adaptive":
        print(f"  SelfRef:  projection={bcfg.projection_kind}, "
              f"self_gen={bcfg.self_generated_values}, "
              f"chunk_size={bcfg.self_ref_chunk_size}")
    if bcfg.momentum_kind != "none":
        print(f"  Momentum: kind={bcfg.momentum_kind}, "
              f"d_hidden={bcfg.momentum_d_hidden}")
    if bcfg.intermediate_size:
        print(f"  SwiGLU:   intermediate_size={bcfg.intermediate_size}")
    if bcfg.donor_weights:
        print(f"  Donor:    {bcfg.donor_weights}")
    if bcfg.alpha_floor is not None or bcfg.alpha_ceil is not None:
        print(f"  α clamps: floor={bcfg.alpha_floor}, ceil={bcfg.alpha_ceil}")
    if bcfg.theta_floor is not None or bcfg.theta_ceil is not None:
        print(f"  θ clamps: floor={bcfg.theta_floor}, ceil={bcfg.theta_ceil}")
    if bcfg.gate_warmup_theta_floor_init is not None:
        print(f"  GateWarmup: theta_floor_init={bcfg.gate_warmup_theta_floor_init}, "
              f"decay_steps={bcfg.gate_warmup_decay_steps}, "
              f"falsification_step={bcfg.gate_warmup_falsification_step}")
    if len(cfg.m_norm_max) > 0:
        print(f"  M-norm:   max={list(cfg.m_norm_max)}")
    if len(cfg.error_clip) > 0:
        print(f"  ErrClip:  max={list(cfg.error_clip)}")
    if params is not None:
        print(f"  Params:   {params.num_params():,}")
    else:
        print(f"  Params:   (stacked — reported after GPU upload)")
    data_len = len(active_loader) if use_bpe else len(token_ids)
    print(f"  Data:     {data_len:,} tokens" +
          (f" ({bcfg.data_format} BPE)" if use_bpe else ""))
    use_gpu = bcfg.gpu and hasattr(nl_hecate, "GpuModel")
    is_stacked = getattr(bcfg, "n_blocks", 1) > 1
    if is_stacked and not use_gpu:
        raise RuntimeError(
            "n_blocks > 1 requires GPU mode (gpu=true). "
            "Stacked multi-block builds are GPU-only."
        )
    if is_stacked and not hasattr(nl_hecate, "GpuStackedModel"):
        raise RuntimeError("n_blocks > 1 requires a CUDA-enabled build with GpuStackedModel")
    # Auto-promote adamw → adamw_gpu when on GPU (no reason to round-trip to CPU)
    if use_gpu and bcfg.optimizer == "adamw":
        bcfg.optimizer = "adamw_gpu"
    if is_stacked and bcfg.optimizer == "adamw_gpu":
        bcfg.optimizer = "adamw_gpu_stacked"
    if is_stacked and bcfg.optimizer != "adamw_gpu_stacked":
        raise RuntimeError(
            f"n_blocks > 1 requires optimizer='adamw_gpu_stacked' (got '{bcfg.optimizer}'). "
            "GpuStackedModel only supports step_adamw()."
        )
    if bcfg.optimizer == "adamw_gpu" and not use_gpu:
        raise RuntimeError(
            "optimizer=adamw_gpu requires GPU and a CUDA-enabled build"
        )
    # Note: GPU + residual=true is supported for training (gpu_cms_forward).
    # Serving paths (prefill/single_token) still assert !residual until adapted.
    # Stacked checkpoint loading and extend_k are supported (spec 22).
    # Guards removed: stacked models can now load and push-up extend.
    if is_stacked and getattr(bcfg, "donor_weights", None) is not None:
        raise RuntimeError(
            "donor_weights is not supported with n_blocks > 1. "
            "GpuStackedModel initializes fresh from seed."
        )
    if is_stacked and getattr(bcfg, "auto_promote", False):
        raise RuntimeError(
            "auto_promote is not supported with n_blocks > 1. "
            "Promotion rebuilds through single-block extend_params_push_up/GpuModel.from_params."
        )
    if bcfg.load and use_gpu and not use_bpe:
        raise RuntimeError(
            "GPU resume with context restore is not yet implemented for byte-level builds. "
            "Use CPU resume (--cpu) or start a fresh GPU build."
        )
    print(f"  Build:    {bcfg.steps} steps (from step {resume_step}), lr={bcfg.lr}")
    print(f"  Optimizer: {bcfg.optimizer}" +
          (f" (b1={bcfg.beta1}, b2={bcfg.beta2}, wd={bcfg.weight_decay}, warmup={bcfg.warmup_steps})"
           if bcfg.optimizer in ("adamw", "adamw_gpu", "adamw_gpu_stacked") else ""))
    if bcfg.max_grad_norm > 0:
        print(f"  Grad clip: max_norm={bcfg.max_grad_norm}")
    print(f"  Device:   {'GPU' if use_gpu else 'CPU'}")
    if bcfg.eval_every > 0:
        tape_eff_disp = bcfg.tape_every if bcfg.tape_every > 0 else bcfg.eval_every
        print(f"  Eval:     every {bcfg.eval_every} steps, {bcfg.eval_max_chunks} max chunks")
        print(f"  Probes:   {bcfg.probe_prompts} prompt(s), {bcfg.probe_max_tokens} tokens each  "
              f"| tape every {tape_eff_disp} steps")
    if bcfg.log_file:
        print(f"  Log:      {bcfg.log_file}")
    print(f"{'=' * 60}\n")

    # ── Stateful CMS build loop ───────────────────────────────────────
    if use_bpe:
        conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
        if build_state is not None and "context_memory" in build_state:
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
            context.set_memory(build_state["context_memory"])
            # Sync conductor step so pulse scheduling matches resumed position.
            target_step = int(build_state.get("conductor_step", 0))
            while conductor.step < target_step:
                conductor.advance()
        else:
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
    else:
        if isinstance(token_ids, MmapTokenStream):
            token_ids.close()
            with open(bcfg.data_path, "rb") as f:
                raw_bytes = f.read()
            stream = nl_hecate.VecStream.from_bytes(raw_bytes)
            del raw_bytes
            token_ids = None
        else:
            stream = nl_hecate.VecStream(token_ids)
        if bcfg.load and build_state is not None:
            conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
            conductor.attach_stream(stream)
            conductor.restore_from_dict(build_state)
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
            context.set_memory(build_state["context_memory"])
        else:
            conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
            conductor.attach_stream(stream)
            context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)

    if bcfg.projection_kind == "adaptive" and not bcfg.load:
        context.seed_self_ref(params)

    # Per-slot loaders for context continuity (spec 06_batch_context_continuity.md).
    # Each slot b has its own sequential M stream through sub-corpus b.
    # Slot b starts at position b * (total_tokens // batch_size) and strides by
    # batch_size * seq_len per step, giving dense sequential coverage within 1/B
    # of the corpus. batch_size=1 falls through to the scalar active_loader path.
    bpe_loaders: list[BpeTokenStream] = []
    if use_bpe and bcfg.batch_size > 1 and active_loader is not None:
        slot_size = active_loader.total_tokens // bcfg.batch_size
        for b in range(bcfg.batch_size):
            loader_b = BpeTokenStream(bcfg.data_path, split="train")
            loader_b.position = b * slot_size
            bpe_loaders.append(loader_b)
        # If a cursor sidecar exists, restore per-slot positions.
        # The sidecar may be from: (a) a multi-slot run {"slots": [...]},
        # or (b) a single-slot run (flat dict) — the latter can't restore
        # per-slot positions so we fall back to partition-start offsets.
        if bcfg.load:
            sidecar = Path(str(bcfg.load) + ".cursor.json")
            if sidecar.exists():
                saved = json.loads(sidecar.read_text())
                slot_cursors = (
                    saved.get("slots")
                    if isinstance(saved, dict) and isinstance(saved.get("slots"), list)
                    else None
                )
                if slot_cursors is not None and len(slot_cursors) == len(bpe_loaders):
                    try:
                        for loader_b, cur in zip(bpe_loaders, slot_cursors, strict=True):
                            loader_b.restore(cur)
                        print(f"  Resuming {len(bpe_loaders)} slots from saved positions")
                    except (CursorMismatchError, CursorOutOfBounds) as e:
                        print(f"  ERROR: slot cursor mismatch — {e}")
                        return
                else:
                    print("  Warning: slot count mismatch in sidecar — "
                          "resetting all slots to partition start")

    gpu_model = None
    if use_gpu:
        if is_stacked:
            n_blocks = bcfg.n_blocks
            periodic = (bcfg.memory_reset == "periodic")
            if _stacked_params_json is not None:
                # Loaded from stacked checkpoint (or extended via push-up)
                gpu_model = nl_hecate.GpuStackedModel.from_params_json(
                    _stacked_params_json, cfg, n_blocks,
                    batch_size=bcfg.batch_size, memory_reset=periodic)
            else:
                # Fresh init
                gpu_model = nl_hecate.GpuStackedModel(
                    cfg, n_blocks, seed=bcfg.seed if hasattr(bcfg, "seed") else 42,
                    batch_size=bcfg.batch_size, memory_reset=periodic)
            print(f"  Stacked:  {n_blocks} blocks x k={bcfg.k} CMS levels"
                  f"  ({gpu_model.total_params():,} params)")
        else:
            periodic = (bcfg.memory_reset == "periodic")
            gpu_model = nl_hecate.GpuModel.from_params(
                params, cfg, batch_size=bcfg.batch_size, memory_reset=periodic)

    error_buffers = nl_hecate.ErrorBufferList(bcfg.k, bcfg.d_model)

    next_doc_idx = 1
    if doc_starts is not None and resume_step > 0:
        byte_pos = resume_step * bcfg.seq_len
        next_doc_idx = int(np.searchsorted(doc_starts, byte_pos, side="right"))

    adamw_opt = None
    use_adamw_gpu = (bcfg.optimizer in ("adamw_gpu", "adamw_gpu_stacked"))
    if bcfg.optimizer == "adamw":
        adamw_opt = nl_hecate.FrequencyAwareAdamW(
            params, beta1=bcfg.beta1, beta2=bcfg.beta2,
            weight_decay=bcfg.weight_decay,
        )

    jsonl: Optional[JSONLLogger] = None
    if bcfg.log_file:
        jsonl = JSONLLogger(bcfg.log_file)
        jsonl.log(event="build_start", config={
            "d_model": bcfg.d_model, "num_heads": bcfg.num_heads,
            "seq_len": bcfg.seq_len, "k": bcfg.k, "memory_rule": bcfg.memory_rule,
            "composition": bcfg.composition, "optimizer": bcfg.optimizer,
            "lr": bcfg.lr, "steps": bcfg.steps, "params": params.num_params() if params is not None else 0,
            "n_blocks": getattr(bcfg, "n_blocks", 1),
        })

    # ── S4-M7 validation state ────────────────────────────────────────
    level_fire_counts = [0] * bcfg.k
    level3_total_fires = 0
    level3_active_fires = 0
    level3_prev_fires = 0
    level3_prev_active = 0
    phase_boundaries = {15000, 25000, 45000, 55000}
    phase_val_data: dict[str, tuple] = {}
    min_stories_loss: float | None = None

    # ── task_cc7eda: CMS health monitoring ───────────────────────────
    # Rolling gnorm window per level (active-step only — ignores inactive-step zeros
    # that would false-trigger dead-level detection for low-frequency CMS levels).
    _DEAD_LEVEL_WINDOW = 100
    _DEAD_LEVEL_THRESHOLD = 1e-4
    _DEAD_LEVEL_MIN_SAMPLES = 4  # require ≥4 active samples before declaring dead
    level_gnorm_history: list[deque] = [deque(maxlen=_DEAD_LEVEL_WINDOW) for _ in range(bcfg.k)]
    # Per-level gnorms for current step (empty until first step_adamw call).
    level_gnorms: list[float] = []

    # Saturation tracking (task_962e72) — per-level EMA, peak, below-threshold streak.
    # Updated at every slow_level_fire event (L2+ fires). Emits level_saturation JSONL
    # event when saturation_ratio < threshold for saturation_window consecutive fires.
    _sat_ema: list[float] = [0.0] * bcfg.k
    _sat_peak: list[float] = [0.0] * bcfg.k
    _sat_below_count: list[int] = [0] * bcfg.k
    _sat_announced: list[bool] = [False] * bcfg.k  # fire once per level
    _last_promotion_step: int = resume_step  # auto-promote cooldown anchor
    _level_start_cursor: int = (
        _restored_level_start_cursor if _restored_level_start_cursor is not None
        else (active_loader.cursor()["position"] if active_loader is not None else 0)
    )
    # Ratio-stability promotion: track rolling ratio history for stdev computation.
    # Promotion fires when stdev(ratio) < threshold for `streak` consecutive samples.
    _sat_ratio_history: list[deque] = [
        deque(maxlen=bcfg.promotion_stability_window) for _ in range(bcfg.k)]
    _sat_stability_streak: list[int] = [0] * bcfg.k
    # Initial per-level M norms captured before training for drift tracking.
    # Captured here (pre-loop) so it reflects the true restored/initialized state,
    # not state after the first optimizer step.
    level_param_norms_init: list[float] = []
    if gpu_model is not None and hasattr(gpu_model, "memory_norms"):
        level_param_norms_init = list(gpu_model.memory_norms())

    losses = []
    t_start = time.perf_counter()
    t_window_start = t_start
    window_step_start = resume_step
    end_step = resume_step + bcfg.steps

    for step in range(resume_step, end_step):
        if use_bpe:
            if bcfg.batch_size > 1:
                if gpu_model is None or not use_adamw_gpu:
                    raise RuntimeError(
                        "batch_size > 1 currently requires GPU with optimizer=adamw_gpu"
                    )
                # Per-slot chunk collection: each loader_b yields its own sequential
                # chunk from sub-corpus b, giving each slot a dense M stream.
                all_input: list[int] = []
                all_target: list[int] = []
                for loader_b in bpe_loaders:
                    chunk = loader_b.next_chunk(bcfg.seq_len)
                    if chunk is None:
                        break
                    all_input.extend(chunk[0])
                    all_target.extend(chunk[1])
                if not all_input:
                    break
                input_ids, target_ids = all_input, all_target
            else:
                chunk = active_loader.next_chunk(bcfg.seq_len)
                if chunk is None:
                    break
                input_ids, target_ids = chunk
            pulse = conductor.pulse()
        else:
            result = conductor.next_chunk(bcfg.seq_len)
            if result is None:
                break
            input_ids, target_ids, pulse = result
            if len(input_ids) != bcfg.seq_len:
                conductor.advance()
                continue

        # ── S4-M7: Track level fire counts ──────────────────────────────
        for lev, active in enumerate(pulse.active_levels):
            if active:
                level_fire_counts[lev] += 1

        if (bcfg.k >= 4 and len(pulse.active_levels) > 3
                and pulse.active_levels[3]):
            level3_total_fires += 1
            if gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                biases = gpu_model.gate_biases()
                if len(biases) > 3:
                    b_theta_l3 = biases[3][1]
                    if b_theta_l3 > 20.0:
                        theta_val = b_theta_l3
                    elif b_theta_l3 < -20.0:
                        theta_val = math.exp(b_theta_l3)
                    else:
                        theta_val = math.log1p(math.exp(b_theta_l3))
                    if theta_val > 0.001:
                        level3_active_fires += 1

        # ── Gate warmup schedule (09_gate_warmup.md) ─────────────────────
        # Phase 2: linearly decay theta_floor_init → 0 over gate_warmup_decay_steps.
        # Applied before the forward pass so the clamp is live for this step.
        if (bcfg.gate_warmup_theta_floor_init is not None
                and step < bcfg.gate_warmup_decay_steps):
            alpha = 1.0 - step / bcfg.gate_warmup_decay_steps
            warmup_floor = [f * alpha for f in bcfg.gate_warmup_theta_floor_init]
            if gpu_model is not None:
                gpu_model.update_theta_floor(warmup_floor)
            else:
                cfg = nl_hecate.MAGConfig(
                    d_model=cfg.d_model, num_heads=cfg.num_heads,
                    head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                    window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                    memory_enabled=cfg.memory_enabled, k=cfg.k,
                    chunk_sizes=list(cfg.chunk_sizes),
                    memory_rule=cfg.memory_rule, composition=cfg.composition,
                    checkpoint_interval=bcfg.checkpoint_interval,
                    tape_multiplier=bcfg.tape_multiplier,
                    projection_kind=cfg.projection_kind,
                    self_generated_values=cfg.self_generated_values,
                    self_ref_chunk_size=cfg.self_ref_chunk_size,
                    momentum_kind=cfg.momentum_kind,
                    momentum_d_hidden=cfg.momentum_d_hidden,
                    attentional_bias=getattr(cfg, "attentional_bias", None),
                    retention=getattr(cfg, "retention", None),
                    intermediate_size=bcfg.intermediate_size,
                    alpha_floor=list(cfg.alpha_floor) if list(cfg.alpha_floor) else None,
                    alpha_ceil=list(cfg.alpha_ceil) if list(cfg.alpha_ceil) else None,
                    theta_floor=warmup_floor,
                    theta_ceil=list(cfg.theta_ceil) if list(cfg.theta_ceil) else None,
                    m_norm_max=list(cfg.m_norm_max) if list(cfg.m_norm_max) else None,
                    error_clip=list(cfg.error_clip) if list(cfg.error_clip) else None,
                    parallel_strategy=bcfg.parallel_strategy,
                    tnt_global_chunk_size=bcfg.tnt_global_chunk_size,
                    tnt_local_chunk_size=bcfg.tnt_local_chunk_size,
                    residual=bcfg.residual,
                )
        elif (bcfg.gate_warmup_theta_floor_init is not None
              and step == bcfg.gate_warmup_decay_steps):
            # Phase 3 start: floor is fully decayed — restore permanent floor
            final_floor = bcfg.theta_floor if bcfg.theta_floor is not None else [0.0] * bcfg.k
            if gpu_model is not None:
                gpu_model.update_theta_floor(final_floor)
            else:
                cfg = nl_hecate.MAGConfig(
                    d_model=cfg.d_model, num_heads=cfg.num_heads,
                    head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                    window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                    memory_enabled=cfg.memory_enabled, k=cfg.k,
                    chunk_sizes=list(cfg.chunk_sizes),
                    memory_rule=cfg.memory_rule, composition=cfg.composition,
                    checkpoint_interval=bcfg.checkpoint_interval,
                    tape_multiplier=bcfg.tape_multiplier,
                    projection_kind=cfg.projection_kind,
                    self_generated_values=cfg.self_generated_values,
                    self_ref_chunk_size=cfg.self_ref_chunk_size,
                    momentum_kind=cfg.momentum_kind,
                    momentum_d_hidden=cfg.momentum_d_hidden,
                    attentional_bias=getattr(cfg, "attentional_bias", None),
                    retention=getattr(cfg, "retention", None),
                    intermediate_size=bcfg.intermediate_size,
                    alpha_floor=list(cfg.alpha_floor) if list(cfg.alpha_floor) else None,
                    alpha_ceil=list(cfg.alpha_ceil) if list(cfg.alpha_ceil) else None,
                    theta_floor=final_floor,
                    theta_ceil=list(cfg.theta_ceil) if list(cfg.theta_ceil) else None,
                    m_norm_max=list(cfg.m_norm_max) if list(cfg.m_norm_max) else None,
                    error_clip=list(cfg.error_clip) if list(cfg.error_clip) else None,
                    parallel_strategy=bcfg.parallel_strategy,
                    tnt_global_chunk_size=bcfg.tnt_global_chunk_size,
                    tnt_local_chunk_size=bcfg.tnt_local_chunk_size,
                    residual=bcfg.residual,
                )

        use_cosine = (adamw_opt is not None or use_adamw_gpu)
        current_lr = cosine_lr(step, bcfg.warmup_steps, end_step, bcfg.lr) if use_cosine else bcfg.lr

        g_norm = 0.0
        # Compute log_this before step_adamw so we can pass collect_level_gnorms
        # only when we'll actually use the per-level norms this step.
        log_this = (step % bcfg.log_every == 0 or step == end_step - 1
                    or (step < 100 and step % 10 == 0))

        # Detect slow-level fires (L2+) to emit targeted gradient events regardless of log_every.
        # LCM(log_every=250, L2=64)=8000 → only 1 logged step per 8000 catches L2.
        # LCM(250, 512)=64000 → L3 is NEVER visible in a 25K run without this.
        slow_level_fires = any(
            pulse.active_levels[i]
            for i in range(2, len(pulse.active_levels))
        ) if len(pulse.active_levels) > 2 else False
        need_gnorms = log_this or slow_level_fires

        if gpu_model is not None and use_adamw_gpu:
            if is_stacked:
                loss, g_norm = gpu_model.step_adamw(
                    input_ids, target_ids, pulse, current_lr,
                    beta1=bcfg.beta1, beta2=bcfg.beta2, eps=1e-8,
                    weight_decay=bcfg.weight_decay,
                    max_grad_norm=bcfg.max_grad_norm,
                    collect_block_gnorms=(log_this and jsonl is not None),
                )
            else:
                loss, g_norm = gpu_model.step_adamw(
                    input_ids, target_ids, pulse, current_lr,
                    beta1=bcfg.beta1, beta2=bcfg.beta2, eps=1e-8,
                    weight_decay=bcfg.weight_decay,
                    max_grad_norm=bcfg.max_grad_norm,
                    collect_level_gnorms=need_gnorms,
                )
            # Component 1+2: collect per-level gnorms for monitoring (active steps only)
            if need_gnorms and hasattr(gpu_model, "level_grad_norms"):
                level_gnorms = gpu_model.level_grad_norms()
                for i, n in enumerate(level_gnorms):
                    if (i < len(level_gnorm_history)
                            and i < len(pulse.active_levels)
                            and pulse.active_levels[i]):
                        level_gnorm_history[i].append(n)
                # Emit a dedicated event for slow-level fires so their gradient
                # dynamics are visible even when this step isn't a log_every step.
                # Also update per-level saturation EMA and emit level_saturation
                # when a level's ratio drops below threshold for the configured window.
                if slow_level_fires and jsonl:
                    # Compute saturation metrics for all active levels this fire.
                    sat_ema_snap: list[float] = []
                    sat_ratio_snap: list[float] = []
                    for i, g in enumerate(level_gnorms):
                        if i < bcfg.k and pulse.active_levels[i]:
                            _sat_ema[i] = (bcfg.saturation_ema_alpha * g
                                           + (1 - bcfg.saturation_ema_alpha) * _sat_ema[i])
                            _sat_peak[i] = max(_sat_peak[i], _sat_ema[i])
                            ratio = (_sat_ema[i] / _sat_peak[i]
                                     if _sat_peak[i] > 1e-10 else 1.0)
                            if ratio < bcfg.saturation_threshold:
                                _sat_below_count[i] += 1
                            else:
                                _sat_below_count[i] = 0
                            # Announce saturation onset once per level.
                            if (_sat_below_count[i] >= bcfg.saturation_window
                                    and not _sat_announced[i]):
                                _sat_announced[i] = True
                                jsonl.log(
                                    event="level_saturation",
                                    step=step,
                                    level=i,
                                    saturation_ratio=round(ratio, 4),
                                    peak_gnorm=round(_sat_peak[i], 8),
                                    ema_gnorm=round(_sat_ema[i], 8),
                                )
                        sat_ema_snap.append(round(_sat_ema[i], 8) if i < bcfg.k else 0.0)
                        sat_ratio_snap.append(
                            round(_sat_ema[i] / _sat_peak[i], 4)
                            if _sat_peak[i] > 1e-10 else 1.0
                        )
                    jsonl.log(
                        event="slow_level_fire",
                        step=step,
                        active_levels=list(pulse.active_levels),
                        level_grad_norms=[round(n, 6) for n in level_gnorms],
                        saturation_ema=sat_ema_snap,
                        saturation_ratio=sat_ratio_snap,
                    )
                # Auto-promote: track L0 ratio stability when slow levels don't fire (k<3)
                if bcfg.auto_promote and not slow_level_fires:
                    for i, g in enumerate(level_gnorms):
                        if i < bcfg.k and i < len(pulse.active_levels) and pulse.active_levels[i]:
                            _sat_ema[i] = (bcfg.saturation_ema_alpha * g
                                           + (1 - bcfg.saturation_ema_alpha) * _sat_ema[i])
                            _sat_peak[i] = max(_sat_peak[i], _sat_ema[i])
                            ratio = (_sat_ema[i] / _sat_peak[i]
                                     if _sat_peak[i] > 1e-10 else 1.0)
                            _sat_ratio_history[i].append(ratio)
                            # Stability check: trimmed stdev of ratio window.
                            # Trim top/bottom 10% to filter gnorm outliers from
                            # mixed-difficulty data (easy boilerplate vs hard passages).
                            if len(_sat_ratio_history[i]) >= bcfg.promotion_stability_window:
                                rh = sorted(_sat_ratio_history[i])
                                trim_n = max(1, len(rh) // 10)
                                rh = rh[trim_n:-trim_n]
                                mean_r = sum(rh) / len(rh)
                                var_r = sum((x - mean_r) ** 2 for x in rh) / (len(rh) - 1)
                                stdev_r = var_r ** 0.5
                                if stdev_r < bcfg.promotion_stability_threshold:
                                    _sat_stability_streak[i] += 1
                                else:
                                    _sat_stability_streak[i] = 0
                                if (_sat_stability_streak[i] >= bcfg.promotion_stability_streak
                                        and not _sat_announced[i]):
                                    _sat_announced[i] = True
                                    if jsonl:
                                        jsonl.log(
                                            event="level_plateau",
                                            step=step, level=i,
                                            ratio_stdev=round(stdev_r, 6),
                                            ratio_mean=round(mean_r, 4),
                                            peak_gnorm=round(_sat_peak[i], 8),
                                            ema_gnorm=round(_sat_ema[i], 8),
                                        )
        elif gpu_model is not None and adamw_opt is None:
            loss = gpu_model.step(input_ids, target_ids, pulse, current_lr)
        elif gpu_model is not None and adamw_opt is not None:
            loss, grad_params = gpu_model.backward_only(input_ids, target_ids, pulse)
            g_norm = adamw_opt.step(params, grad_params, pulse, current_lr,
                                    max_grad_norm=bcfg.max_grad_norm)
            nl_hecate.mag_apply_weight_gradients(params, grad_params, 0.0)
            gpu_model.upload_params(params)
        else:
            loss, grads = nl_hecate.cms_compute_gradients(
                params, cfg, input_ids, target_ids, pulse, context,
                error_buffers)
            if adamw_opt:
                g_norm = adamw_opt.step(params, grads, pulse, current_lr,
                                        max_grad_norm=bcfg.max_grad_norm)
                nl_hecate.mag_apply_weight_gradients(params, grads, 0.0)
            else:
                nl_hecate.mag_apply_weight_gradients(params, grads, current_lr)
                error_buffers.apply_for_active(params, pulse, current_lr)

        if math.isnan(loss) or math.isinf(loss):
            print(f"  step {step:5d}  loss={loss} — ABORTING (NaN/Inf detected)")
            if jsonl:
                jsonl.log(event="abort", step=step, reason="nan_inf", loss=float(loss))
            break

        conductor.advance()

        if doc_starts is not None:
            byte_pos = (step + 1) * bcfg.seq_len
            prev_idx = next_doc_idx
            while next_doc_idx < len(doc_starts) and byte_pos >= doc_starts[next_doc_idx]:
                next_doc_idx += 1
            if next_doc_idx > prev_idx:
                if gpu_model is not None:
                    gpu_model.reset_context()
                else:
                    context.reset()
                error_buffers.reset()

        losses.append(loss)

        if step % 100 == 0:
            gc.collect()

        ppl = math.exp(min(loss, 20.0))

        if log_this:
            t_now = time.perf_counter()
            window_steps = (step + 1) - window_step_start  # steps [window_start, step] inclusive
            dt = t_now - t_window_start
            if window_steps > 0 and dt > 0:
                tok_per_sec = window_steps * len(input_ids) / dt
            else:
                tok_per_sec = 0.0
            t_window_start = t_now
            window_step_start = step + 1  # next window starts after this step
            msg = f"  step {step:5d}  loss={loss:.4f}  ppl={ppl:.1f}"
            if tok_per_sec > 0:
                msg += f"  tok/s={tok_per_sec:.0f}"
            if g_norm > 0:
                msg += f"  gnorm={g_norm:.4f}"
            # Component 1: per-level gnorm breakdown (e.g. gnorm_l=[2.1,0.8,0.3,0.02])
            if level_gnorms:
                lgnorm_str = ",".join(f"{n:.3f}" for n in level_gnorms)
                msg += f"  gnorm_l=[{lgnorm_str}]"
            if adamw_opt or use_adamw_gpu:
                msg += f"  lr={current_lr:.6f}"
            msg += f"  rss={rss_mb():.0f}MB"
            print(msg)

            # Component 2: dead level detection (active-step samples only).
            # level_gnorm_history only contains entries for steps where the level
            # fired, so no inactive-step zeros can false-trigger the warning.
            for i, hist in enumerate(level_gnorm_history):
                if len(hist) < _DEAD_LEVEL_MIN_SAMPLES:
                    continue  # too few active samples — defer judgment
                win_avg = sum(hist) / len(hist)
                if win_avg < _DEAD_LEVEL_THRESHOLD:
                    print(f"  WARNING: Level {i} dead — "
                          f"{len(hist)}-sample active gnorm avg {win_avg:.2e} "
                          f"< {_DEAD_LEVEL_THRESHOLD:.0e} — STOP THE LINE")

        if jsonl and (step % bcfg.log_every == 0 or step == end_step - 1):
            log_fields: dict[str, Any] = dict(
                event="step", step=step, loss=loss, ppl=ppl,
                grad_norm=g_norm, lr=current_lr,
                elapsed=time.perf_counter() - t_start,
                active_levels=pulse.active_levels,
            )
            if use_bpe:
                n_masked = sum(1 for t in target_ids if t >= bcfg.vocab_size)
                log_fields["masked_ratio"] = n_masked / len(target_ids)
            if gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                log_fields["gate_biases"] = gpu_model.gate_biases()
            log_fields["level_fires"] = list(level_fire_counts)
            # Component 1: per-level gnorms in JSONL
            if level_gnorms:
                log_fields["level_grad_norms"] = [round(n, 6) for n in level_gnorms]
            if (bcfg.eval_every > 0 and step % bcfg.eval_every == 0
                    and gpu_model is not None
                    and hasattr(gpu_model, "memory_norms")):
                log_fields["memory_norms"] = [
                    round(n, 6) for n in gpu_model.memory_norms()]
            # Spec 23: per-block gradient norms + depth specialization CV
            if is_stacked and hasattr(gpu_model, "block_grad_norms"):
                block_gnorms = gpu_model.block_grad_norms()
                if block_gnorms:
                    log_fields["block_grad_norms"] = [
                        round(g, 6) for g in block_gnorms]
                    mean_bg = sum(block_gnorms) / len(block_gnorms)
                    if mean_bg > 0:
                        var_bg = sum(
                            (g - mean_bg) ** 2 for g in block_gnorms
                        ) / len(block_gnorms)
                        log_fields["block_gnorm_cv"] = round(
                            var_bg ** 0.5 / mean_bg, 6)
                    else:
                        log_fields["block_gnorm_cv"] = 0.0
                # L0-only per-block gnorms for promotion floor check (spec 19)
                if hasattr(gpu_model, "l0_block_grad_norms"):
                    l0_bg = gpu_model.l0_block_grad_norms()
                    if l0_bg:
                        log_fields["l0_block_grad_norms"] = [
                            round(g, 6) for g in l0_bg]
            jsonl.log(**log_fields)

        # Gate warmup falsification checkpoint (09_gate_warmup.md §5)
        if (bcfg.gate_warmup_theta_floor_init is not None
                and bcfg.gate_warmup_falsification_step > 0
                and step == bcfg.gate_warmup_falsification_step
                and gpu_model is not None
                and hasattr(gpu_model, "gate_biases")):
            biases = gpu_model.gate_biases()
            def _softplus(x: float) -> float:
                # Numerically stable: avoids exp() overflow for large positive x
                return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))
            l2_theta = _softplus(biases[2][1]) if len(biases) > 2 else 0.0
            l3_theta = _softplus(biases[3][1]) if len(biases) > 3 else 0.0
            l2_pass = l2_theta > bcfg.gate_warmup_l2_threshold
            l3_pass = l3_theta > bcfg.gate_warmup_l3_threshold
            verdict = "GO" if (l2_pass and l3_pass) else "NO-GO"
            print(f"\n  ── Gate warmup falsification @ step {step} ──")
            print(f"     L2 θ={l2_theta:.5f}  threshold={bcfg.gate_warmup_l2_threshold}  {'PASS' if l2_pass else 'FAIL'}")
            print(f"     L3 θ={l3_theta:.5f}  threshold={bcfg.gate_warmup_l3_threshold}  {'PASS' if l3_pass else 'FAIL'}")
            print(f"     Verdict: {verdict}\n")
            if jsonl:
                jsonl.log(event="gate_warmup_falsification", step=step,
                          l2_theta=round(l2_theta, 6), l3_theta=round(l3_theta, 6),
                          l2_pass=l2_pass, l3_pass=l3_pass, verdict=verdict)
            if verdict != "GO":
                raise RuntimeError(
                    f"Gate warmup falsification FAILED at step {step}: "
                    f"L2 θ={l2_theta:.5f} ({'PASS' if l2_pass else 'FAIL'}), "
                    f"L3 θ={l3_theta:.5f} ({'PASS' if l3_pass else 'FAIL'}). "
                    f"Pod run blocked. See specs/infrastructure/09_gate_warmup.md §5.")

        if (jsonl and bcfg.k >= 4 and step > 0 and step % 1000 == 0):
            l3_fires_delta = level3_total_fires - level3_prev_fires
            l3_active_delta = level3_active_fires - level3_prev_active
            jsonl.log(event="level3_activity", step=step,
                      fires=l3_fires_delta,
                      active=l3_active_delta)
            level3_prev_fires = level3_total_fires
            level3_prev_active = level3_active_fires

        # Component 4: level activity heatmap at eval intervals.
        # Stats computed over active-step samples only (same as dead level check).
        if (jsonl and bcfg.eval_every > 0 and step > 0
                and step % bcfg.eval_every == 0):
            heatmap_levels = []
            for i in range(bcfg.k):
                hist = level_gnorm_history[i]
                n_samples = len(hist)
                win_avg = sum(hist) / n_samples if n_samples else 0.0
                win_min = min(hist) if n_samples else 0.0
                win_max = max(hist) if n_samples else 0.0
                is_dead = (n_samples >= _DEAD_LEVEL_MIN_SAMPLES
                           and win_avg < _DEAD_LEVEL_THRESHOLD)
                heatmap_levels.append({
                    "level": i,
                    "fires": level_fire_counts[i],
                    "active_samples": n_samples,
                    "gnorm_avg": round(win_avg, 6),
                    "gnorm_min": round(win_min, 6),
                    "gnorm_max": round(win_max, 6),
                    "dead": is_dead,
                })
            jsonl.log(event="level_heatmap", step=step, levels=heatmap_levels)

        if (bcfg.eval_every > 0 and val_stream is not None
                and step > 0 and step % bcfg.eval_every == 0
                and not is_stacked):
            saved_ctx = None
            try:
                if gpu_model is not None:
                    saved_ctx = gpu_model.to_host_context()
                    gpu_model.reset_context()
                eval_loss, eval_ppl = evaluate(
                    gpu_model, bcfg, val_stream, bcfg.eval_max_chunks,
                    val_doc_starts=val_doc_starts)
            finally:
                if gpu_model is not None and saved_ctx is not None:
                    gpu_model.upload_context(saved_ctx)
            print(f"  [eval] step {step:5d}  loss={eval_loss:.4f}  ppl={eval_ppl:.1f}")
            if gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                print_level_metrics(gpu_model, bcfg.k)
            tape_device = getattr(bcfg, "tape_device", "off")
            tape_enabled = tape_device != "off"
            tape_eff = bcfg.tape_every if bcfg.tape_every > 0 else bcfg.eval_every
            if (tape_eff > 0 and step % tape_eff == 0
                    and gpu_model is not None
                    and tape_enabled
                    and input_ids is not None and target_ids is not None
                    and max(target_ids) < bcfg.vocab_size):
                # Skip silently when batch contains masked targets (vocab_size sentinel).
                # The Rust binding rejects target_ids >= vocab_size; masked batches are
                # normal (all-user-turn chunks) and should not generate warning noise.
                try:
                    if is_stacked and tape_device == "cpu":
                        # Stacked model -- CPU Wengert tape for full gradient observability
                        if not hasattr(gpu_model, "cpu_stacked_tape_summary"):
                            raise RuntimeError(
                                "tape_device='cpu' requested for stacked model but "
                                "cpu_stacked_tape_summary() is unavailable"
                            )
                        tape_sum = gpu_model.cpu_stacked_tape_summary(
                            input_ids, target_ids, pulse
                        )
                    elif is_stacked and hasattr(gpu_model, "gpu_stacked_tape_summary"):
                        # Stacked model -- per-(block, level) diagnostics (GPU fast path)
                        tape_sum = gpu_model.gpu_stacked_tape_summary(
                            input_ids, target_ids, pulse
                        )
                    elif tape_device == "gpu" and hasattr(gpu_model, "gpu_tape_forward_summary"):
                        tape_sum = gpu_model.gpu_tape_forward_summary(
                            input_ids, target_ids, pulse
                        )
                    elif tape_device == "cpu" and hasattr(gpu_model, "tape_forward_summary"):
                        tape_sum = gpu_model.tape_forward_summary(
                            input_ids, target_ids, pulse
                        )
                    elif hasattr(gpu_model, "gpu_tape_forward_summary"):
                        # Fallback: prefer GPU if available
                        tape_sum = gpu_model.gpu_tape_forward_summary(
                            input_ids, target_ids, pulse
                        )
                    elif hasattr(gpu_model, "tape_forward_summary"):
                        tape_sum = gpu_model.tape_forward_summary(
                            input_ids, target_ids, pulse
                        )
                    else:
                        tape_sum = None
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    print(f"  [tape] WARNING: tape summary failed at step {step}: {exc}")
                else:
                    if tape_sum is not None:
                        print_tape_summary(tape_sum, step)
                        if jsonl:
                            jsonl.log(event="tape_summary", step=step, **tape_sum)
            if bcfg.k > 1:
                fires_str = "  ".join(f"L{i}:{level_fire_counts[i]}" for i in range(bcfg.k))
                print(f"    [fires] {fires_str}")
                level_fire_counts = [0] * bcfg.k
            # ── Learning probes (CS-10: model learns during eval) ─────
            snapshot = None
            if gpu_model is not None and tokenizer is not None:
                try:
                    snapshot = full_snapshot(gpu_model)
                    # Probe 1: within-generation learning curve
                    # Restore between probes: step_generate modifies params
                    n_prompts = max(1, min(bcfg.probe_prompts, len(EVAL_PROMPTS)))
                    for prompt_text in EVAL_PROMPTS[:n_prompts]:
                        full_restore(gpu_model, snapshot)
                        gpu_model.reset_optimizer()
                        prompt_ids = tokenizer.encode(prompt_text)
                        result = probe_within_generation(
                            gpu_model, cfg, prompt_ids, tokenizer,
                            max_tokens=bcfg.probe_max_tokens, temperature=0.7, lr=bcfg.lr)
                        preview = result["generated_text"][:60].replace("\n", "\\n")
                        print(f"    [probe1] \"{prompt_text}\" → \"{preview}\"")
                        print(f"      loss: {result['loss_first10_avg']:.4f} → "
                              f"{result['loss_last10_avg']:.4f}  "
                              f"slope={result['loss_slope']:.6f}")
                        if jsonl:
                            jsonl.log(event="learning_probe",
                                      probe="within_generation", step=step,
                                      prompt=prompt_text,
                                      loss_first10=result["loss_first10_avg"],
                                      loss_last10=result["loss_last10_avg"],
                                      loss_slope=result["loss_slope"],
                                      n_tokens=result["n_tokens"])

                    # Probe 2: cross-exposure adaptation (first prompt only)
                    full_restore(gpu_model, snapshot)
                    gpu_model.reset_optimizer()  # probe1 corrupts AdamW moments
                    prompt_text = EVAL_PROMPTS[0]
                    prompt_ids = tokenizer.encode(prompt_text)
                    xresult = probe_cross_exposure(
                        gpu_model, cfg, prompt_ids, tokenizer,
                        max_tokens=max(10, bcfg.probe_max_tokens // 2),
                        temperature=0.7, lr=bcfg.lr)
                    print(f"    [probe2] \"{prompt_text}\" "
                          f"run1={xresult['run1_avg_loss']:.4f} → "
                          f"run2={xresult['run2_avg_loss']:.4f}  "
                          f"Δ={xresult['improvement']:.4f} "
                          f"({xresult['improvement_pct']:.1f}%)")
                    if jsonl:
                        jsonl.log(event="learning_probe",
                                  probe="cross_exposure", step=step,
                                  prompt=prompt_text,
                                  run1_loss=xresult["run1_avg_loss"],
                                  run2_loss=xresult["run2_avg_loss"],
                                  improvement=xresult["improvement"],
                                  improvement_pct=xresult["improvement_pct"])
                except Exception as e:
                    print(f"    [learning probe failed: {e}]")
                finally:
                    if snapshot is not None:
                        full_restore(gpu_model, snapshot)
                        gpu_model.reset_optimizer()  # probes corrupt AdamW moments
            # ── Memory vocab probe (logit lens for CMS levels) ────────
            if gpu_model is not None and tokenizer is not None:
                try:
                    # Reuse params/context already downloaded by full_snapshot.
                    # Guard: full_snapshot may have failed or been skipped; re-acquire if needed.
                    if snapshot is None:
                        snapshot = full_snapshot(gpu_model)
                    vprobe = probe_memory_vocab(
                        snapshot["params"], snapshot["context"],
                        cfg, tokenizer, step)
                    for lv in vprobe["levels"]:
                        top3 = " ".join(t["tok"] for t in lv["top20"][:3])
                        print(f"    [vocab-probe] L{lv['level']}  "
                              f"‖M‖={lv['m_norm']:.4f}  top3={top3}")
                    if jsonl:
                        jsonl.log(event="memory_vocab_probe", **vprobe)
                except Exception as e:
                    print(f"    [vocab probe failed: {e}]")
            if jsonl:
                jsonl.log(event="eval", step=step, eval_loss=eval_loss,
                          eval_ppl=eval_ppl, eval_chunks=bcfg.eval_max_chunks)

        # ── S4-M7: Phase boundary curriculum probe ────────────────────
        if (step in phase_boundaries and gpu_model is not None
                and use_bpe and bcfg.data_path and not is_stacked):
            if not phase_val_data:
                data_dir = Path(bcfg.data_path)
                for pname in ("stories", "conversation", "reasoning"):
                    tk = data_dir / f"val_{pname}_tokens.npy"
                    tg = data_dir / f"val_{pname}_targets.npy"
                    if tk.exists() and tg.exists():
                        phase_val_data[pname] = (np.load(tk), np.load(tg))
                if phase_val_data:
                    print(f"  [phase probe] Loaded per-phase val data: "
                          f"{list(phase_val_data.keys())}")

            if phase_val_data:
                probe_ctx = gpu_model.to_host_context()
                try:
                    gpu_model.reset_context()

                    phase_losses = {}
                    for pname, (p_toks, p_tgts) in phase_val_data.items():
                        pl, _pp = evaluate_numpy(
                            gpu_model, bcfg, p_toks, p_tgts, max_chunks=10)
                        phase_losses[pname] = pl
                        gpu_model.reset_context()
                finally:
                    gpu_model.upload_context(probe_ctx)

                if "stories" in phase_losses and step <= 25000:
                    sl = phase_losses["stories"]
                    if min_stories_loss is None or sl < min_stories_loss:
                        min_stories_loss = sl

                print(f"  [phase probe] step {step}: "
                      + ", ".join(f"{k}={v:.4f}" for k, v in phase_losses.items()))
                if jsonl:
                    log_entry: dict[str, Any] = {
                        "event": "phase_boundary", "step": step}
                    if min_stories_loss is not None:
                        log_entry["min_stories_loss"] = min_stories_loss
                    for pname, pl in phase_losses.items():
                        log_entry[f"{pname}_loss"] = pl
                    jsonl.log(**log_entry)

        # ── Stacked tape diagnostics (independent of eval block) ──────
        if is_stacked and gpu_model is not None:
            tape_eff_stacked = bcfg.tape_every if bcfg.tape_every > 0 else bcfg.eval_every
            tape_device_stacked = getattr(bcfg, "tape_device", "off")
            if (tape_device_stacked != "off" and tape_eff_stacked > 0
                    and step > 0 and step % tape_eff_stacked == 0
                    and input_ids is not None and target_ids is not None
                    and max(target_ids) < bcfg.vocab_size):
                try:
                    if tape_device_stacked == "cpu":
                        if not hasattr(gpu_model, "cpu_stacked_tape_summary"):
                            raise RuntimeError(
                                "tape_device='cpu' requested for stacked model but "
                                "cpu_stacked_tape_summary() is unavailable"
                            )
                        tape_sum = gpu_model.cpu_stacked_tape_summary(
                            input_ids, target_ids, pulse
                        )
                    elif hasattr(gpu_model, "gpu_stacked_tape_summary"):
                        tape_sum = gpu_model.gpu_stacked_tape_summary(
                            input_ids, target_ids, pulse
                        )
                    else:
                        tape_sum = None
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    print(f"  [tape] WARNING: stacked tape summary failed at step {step}: {exc}")
                else:
                    if tape_sum is not None:
                        print_tape_summary(tape_sum, step)
                        if jsonl:
                            jsonl.log(event="tape_summary", step=step, **tape_sum)

        # Periodic checkpoint
        if bcfg.save_every > 0 and step > 0 and step % bcfg.save_every == 0:
            if is_stacked and gpu_model is not None:
                p = Path(_safetensors_path(bcfg.save_path))
                ckpt_path = str(p.with_stem(f"{p.stem}_step{step}"))
                os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                nl_hecate.save_stacked_checkpoint(
                    ckpt_path, gpu_model,
                    conductor=conductor, context=context)
                sidecar = Path(str(ckpt_path) + ".cursor.json")
                if bpe_loaders:
                    sidecar.write_text(json.dumps(
                        {"slots": [loader.cursor() for loader in bpe_loaders],
                         "level_start_cursor": _level_start_cursor}, indent=2))
                elif active_loader is not None:
                    cursor_data = active_loader.cursor()
                    cursor_data["level_start_cursor"] = _level_start_cursor
                    sidecar.write_text(json.dumps(cursor_data, indent=2))
                print(f"  [checkpoint saved: {ckpt_path}]")
            elif not is_stacked:
                if gpu_model is not None:
                    params = gpu_model.to_host_params()
                    context = gpu_model.to_host_context()
                p = Path(_safetensors_path(bcfg.save_path))
                ckpt_path = str(p.with_stem(f"{p.stem}_step{step}"))
                os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
                if use_bpe:
                    nl_hecate.save_checkpoint_with_context(ckpt_path, params, cfg, conductor, context)
                else:
                    nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)
                sidecar = Path(str(ckpt_path) + ".cursor.json")
                if bpe_loaders:
                    sidecar.write_text(json.dumps(
                        {"slots": [loader.cursor() for loader in bpe_loaders],
                         "level_start_cursor": _level_start_cursor}, indent=2))
                elif active_loader is not None:
                    cursor_data = active_loader.cursor()
                    cursor_data["level_start_cursor"] = _level_start_cursor
                    sidecar.write_text(json.dumps(cursor_data, indent=2))
                print(f"  [checkpoint saved: {ckpt_path}]")

                # Component 3: per-level parameter drift (||M_t||_F vs ||M_0||_F)
                # Uses memory_norms as proxy for parameter magnitude per level.
                # ||M_t - M_0|| is approximated by tracking norm evolution over time
                # since storing full initial M tensors would require d*d*k floats.
                if jsonl and gpu_model is not None and hasattr(gpu_model, "memory_norms"):
                    cur_norms = list(gpu_model.memory_norms())
                    drift_info = []
                    for i, cur_n in enumerate(cur_norms):
                        init_n = level_param_norms_init[i] if i < len(level_param_norms_init) else 0.0
                        drift_info.append({
                            "level": i,
                            "norm_init": round(init_n, 6),
                            "norm_now": round(cur_n, 6),
                            "norm_ratio": round(cur_n / init_n, 4) if init_n > 1e-8 else None,
                        })
                    jsonl.log(event="level_param_drift", step=step, levels=drift_info)

                # S4-M7: Checkpoint roundtrip verification
                if use_gpu:
                    v_model = None
                    try:
                        if use_bpe:
                            v_params, v_cfg = nl_hecate.load_checkpoint(ckpt_path)
                        else:
                            v_params, v_cfg, _ = nl_hecate.load_build_checkpoint(ckpt_path)
                        v_model = nl_hecate.GpuModel.from_params(v_params, v_cfg, batch_size=bcfg.batch_size)
                        # Save context before verification forward passes
                        # Slice to single seq_len chunk -- forward() expects exactly
                        # seq_len tokens regardless of batch_size used in step_adamw
                        rt_input = list(input_ids[:bcfg.seq_len])
                        rt_target = list(target_ids[:bcfg.seq_len])
                        rt_ctx = gpu_model.to_host_context()
                        try:
                            v_model.upload_context(rt_ctx)
                            train_fwd, _ = gpu_model.forward(rt_input, rt_target, pulse)
                            verify_fwd, _ = v_model.forward(rt_input, rt_target, pulse)
                        finally:
                            # Restore context after verification (forward modifies M)
                            gpu_model.upload_context(rt_ctx)
                        delta = abs(verify_fwd - train_fwd)
                        if jsonl:
                            jsonl.log(event="checkpoint_roundtrip", step=step,
                                      delta=delta, loss=train_fwd,
                                      verify_loss=verify_fwd)
                        if delta > 1e-6:
                            print(f"  [WARNING] checkpoint roundtrip "
                                  f"delta={delta:.2e}")
                        else:
                            print(f"  [checkpoint roundtrip OK, "
                                  f"delta={delta:.2e}]")
                    except (OSError, RuntimeError, ValueError) as e:
                        print(f"  [checkpoint roundtrip failed: {e}]")
                    finally:
                        del v_model

                # ── Checkpoint learning samples + Probe 3 ─────────────────
                if tokenizer is not None and gpu_model is not None:
                    ckpt_snapshot = full_snapshot(gpu_model)
                    try:
                        # Learning samples (generate_learning, not frozen)
                        # Restore between samples: each 128-step generate_learning
                        # heavily modifies params toward one prompt's pattern.
                        from engine.generation import generate_learning
                        for prompt_text in SAMPLE_PROMPTS:
                            full_restore(gpu_model, ckpt_snapshot)
                            gpu_model.reset_optimizer()
                            gpu_model.reset_context()
                            prompt_ids = tokenizer.encode(prompt_text)
                            tokens, losses, _ = generate_learning(
                                gpu_model, cfg, prompt_ids,
                                max_tokens=128, temperature=0.7, lr=bcfg.lr)
                            gen_text = tokenizer.decode(tokens[len(prompt_ids):])
                            preview = gen_text[:80].replace("\n", " ")
                            valid = [v for v in losses if not math.isnan(v)]
                            avg_loss = sum(valid) / len(valid) if valid else float('nan')
                            n_gen = len(tokens) - len(prompt_ids)
                            print(f"  [sample] {prompt_text[:40]}... → {preview}...")
                            print(f"    avg_loss={avg_loss:.4f} over {n_gen} tokens"
                                  f" ({len(valid)}/{len(losses)} valid)")
                        if jsonl:
                            jsonl.log(event="sample", step=step,
                                      mode="learning", n_prompts=len(SAMPLE_PROMPTS))

                        # Probe 3: accumulated context vs cold start (first prompt)
                        full_restore(gpu_model, ckpt_snapshot)
                        gpu_model.reset_optimizer()  # prior probes/samples corrupt AdamW moments
                        prompt_text = EVAL_PROMPTS[0]
                        prompt_ids = tokenizer.encode(prompt_text)
                        cresult = probe_context_value(
                            gpu_model, cfg, prompt_ids, ckpt_snapshot,
                            max_tokens=30, temperature=0.7, lr=bcfg.lr)
                        print(f"  [probe3] cold={cresult['cold_avg_loss']:.4f} "
                              f"warm={cresult['warm_avg_loss']:.4f} "
                              f"benefit={cresult['context_benefit']:.4f}")
                        if jsonl:
                            jsonl.log(event="learning_probe",
                                      probe="context_value", step=step,
                                      cold_loss=cresult["cold_avg_loss"],
                                      warm_loss=cresult["warm_avg_loss"],
                                      context_benefit=cresult["context_benefit"])
                    except Exception as e:
                        print(f"  [checkpoint samples/probe3 failed: {e}]")
                    finally:
                        full_restore(gpu_model, ckpt_snapshot)
                        gpu_model.reset_optimizer()  # probes corrupt AdamW moments

        # ── Auto-promotion: L0 saturated → push up to k+1 ──────────
        if (bcfg.auto_promote and _sat_announced[0]
                and bcfg.k < bcfg.target_k
                and step - _last_promotion_step >= bcfg.promotion_cooldown):
            old_k = bcfg.k
            new_k = old_k + 1
            print(f"\n{'=' * 60}")
            print(f"  AUTO-PROMOTION: L0 saturated at step {step}")
            print(f"  Extending k={old_k} → k={new_k}")
            print(f"{'=' * 60}")

            # Save pre-promotion checkpoint
            params = gpu_model.to_host_params()
            promo_ctx = gpu_model.to_host_context()
            p = Path(_safetensors_path(bcfg.save_path))
            promo_ckpt = str(p.with_stem(f"{p.stem}_pre_k{new_k}_step{step}"))
            os.makedirs(os.path.dirname(promo_ckpt) or ".", exist_ok=True)
            nl_hecate.save_checkpoint_with_context(
                promo_ckpt, params, cfg, conductor, promo_ctx)
            promo_sidecar = Path(str(promo_ckpt) + ".cursor.json")
            if bpe_loaders:
                promo_sidecar.write_text(json.dumps(
                    {"slots": [loader.cursor() for loader in bpe_loaders],
                     "level_start_cursor": _level_start_cursor}, indent=2))
            elif active_loader is not None:
                promo_cursor = active_loader.cursor()
                promo_cursor["level_start_cursor"] = _level_start_cursor
                promo_sidecar.write_text(json.dumps(promo_cursor, indent=2))
            print(f"  Checkpoint: {promo_ckpt}")

            if jsonl:
                cursor_pos = (active_loader.cursor()["position"]
                              if active_loader is not None else 0)
                jsonl.log(event="auto_promotion", step=step,
                          old_k=old_k, new_k=new_k,
                          cursor_position=cursor_pos,
                          saturation_ema=list(_sat_ema[:old_k]),
                          saturation_peak=list(_sat_peak[:old_k]))

            # Build new MAGConfig(k+1)
            chunk_template = [1, 8, 64, 512]
            new_chunks = chunk_template[:new_k]
            # Prefer bcfg values, fall back to live cfg (normalize [] → None)
            src_m_norm = (bcfg.m_norm_max or None) if bcfg.m_norm_max is not None else (
                list(cfg.m_norm_max) if hasattr(cfg, 'm_norm_max') and cfg.m_norm_max else None
            )
            new_m_norm = [*src_m_norm, src_m_norm[-1]] if src_m_norm else None
            src_error_clip = (bcfg.error_clip or None) if bcfg.error_clip is not None else (
                list(cfg.error_clip) if hasattr(cfg, 'error_clip') and cfg.error_clip else None
            )
            new_error_clip = [*src_error_clip, src_error_clip[-1]] if src_error_clip else None
            new_cfg = nl_hecate.MAGConfig(
                d_model=cfg.d_model, num_heads=cfg.num_heads,
                head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                memory_enabled=cfg.memory_enabled, k=new_k,
                chunk_sizes=new_chunks,
                memory_rule=cfg.memory_rule, composition=cfg.composition,
                checkpoint_interval=bcfg.checkpoint_interval,
                tape_multiplier=bcfg.tape_multiplier,
                projection_kind=cfg.projection_kind,
                self_generated_values=cfg.self_generated_values,
                self_ref_chunk_size=cfg.self_ref_chunk_size,
                momentum_kind=cfg.momentum_kind,
                momentum_d_hidden=cfg.momentum_d_hidden,
                attentional_bias=getattr(cfg, "attentional_bias", None),
                retention=getattr(cfg, "retention", None),
                intermediate_size=bcfg.intermediate_size,
                alpha_floor=None, alpha_ceil=None,
                theta_floor=None, theta_ceil=None,
                m_norm_max=new_m_norm,
                error_clip=new_error_clip,
                parallel_strategy=getattr(cfg, "parallel_strategy", None),
                tnt_global_chunk_size=bcfg.tnt_global_chunk_size,
                tnt_local_chunk_size=bcfg.tnt_local_chunk_size,
                residual=bcfg.residual,
            )

            # Push-up: shift trained levels to slower frequencies, fresh L0
            params = nl_hecate.extend_params_push_up(params, new_cfg, bcfg.seed)
            cfg = new_cfg
            print(f"  Push-up complete: chunks={new_chunks}")

            # Rebuild GPU model (fresh optimizer state for new parameter layout)
            del gpu_model
            gc.collect()
            periodic = (bcfg.memory_reset == "periodic")
            gpu_model = nl_hecate.GpuModel.from_params(
                params, cfg, batch_size=bcfg.batch_size, memory_reset=periodic)

            # Fresh conductor and context for new k
            conductor = nl_hecate.Conductor(new_k, new_chunks)
            context = nl_hecate.ContextState(new_k, bcfg.d_model)
            error_buffers = nl_hecate.ErrorBufferList(new_k, bcfg.d_model)

            # Update config state
            bcfg.k = new_k
            bcfg.chunk_sizes = new_chunks
            if new_m_norm is not None:
                bcfg.m_norm_max = new_m_norm
            if new_error_clip is not None:
                bcfg.error_clip = new_error_clip

            # Reset all per-level tracking for new k
            _sat_ema = [0.0] * new_k
            _sat_peak = [0.0] * new_k
            _sat_below_count = [0] * new_k
            _sat_announced = [False] * new_k
            _sat_ratio_history = [
                deque(maxlen=bcfg.promotion_stability_window) for _ in range(new_k)]
            _sat_stability_streak = [0] * new_k
            _last_promotion_step = step
            level_gnorm_history = [deque(maxlen=_DEAD_LEVEL_WINDOW) for _ in range(new_k)]
            level_fire_counts = [0] * new_k
            level_gnorms = []
            if hasattr(gpu_model, "memory_norms"):
                level_param_norms_init = list(gpu_model.memory_norms())
            if new_k >= 4:
                level3_total_fires = 0
                level3_active_fires = 0
                level3_prev_fires = 0
                level3_prev_active = 0

            if active_loader is not None:
                cursor = active_loader.cursor()
                cur_pos = cursor["position"]
                if bcfg.promotion_rewind_pct > 0.0:
                    if bcfg.batch_size > 1:
                        raise RuntimeError(
                            "promotion_rewind_pct with batch_size > 1 is not implemented; "
                            "rewind must be applied to each slot loader"
                        )
                    # Rewind by a fraction of tokens consumed during THIS level's phase only.
                    tokens_this_level = cur_pos - _level_start_cursor
                    rewind_tokens = int(tokens_this_level * bcfg.promotion_rewind_pct)
                    new_pos = max(0, cur_pos - rewind_tokens)
                    active_loader.restore({
                        "position": new_pos,
                        "total_tokens": cursor["total_tokens"],
                        "content_hash": 0,
                        "chunk_id": 0,
                        "seq_len": bcfg.seq_len,
                        "dataset_path": cursor["dataset_path"],
                    })
                    print(f"  Data cursor: {cur_pos:,} → {new_pos:,} "
                          f"(rewound {bcfg.promotion_rewind_pct:.0%} of {tokens_this_level:,} "
                          f"tokens from this level)")
                else:
                    print(f"  Data cursor: {cur_pos:,} (continues naturally)")
                _level_start_cursor = (active_loader.cursor()["position"]
                                       if active_loader is not None else 0)
            print(f"  Promotion complete, continuing at step {step + 1}\n")

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    total_tokens = len(losses) * bcfg.seq_len * (bcfg.batch_size if use_bpe else 1)
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    if bcfg.k >= 4:
        print(f"\n  Level 3 activity: {level3_active_fires} active / "
              f"{level3_total_fires} total fires")
        if level3_active_fires < 25:
            print("  WARNING: Level 3 activity < 25 — STOP THE LINE")
        elif level3_active_fires < 50:
            print("  WARNING: Level 3 activity < 50 — below threshold")
        if jsonl:
            jsonl.log(event="level3_summary",
                      total_fires=level3_total_fires,
                      active_fires=level3_active_fires)

    # ── Final checkpoint ──────────────────────────────────────────────
    final_path = _safetensors_path(bcfg.save_path)
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    if is_stacked and gpu_model is not None:
        nl_hecate.save_stacked_checkpoint(
            final_path, gpu_model,
            conductor=conductor, context=context)
    else:
        if gpu_model is not None:
            params = gpu_model.to_host_params()
            context = gpu_model.to_host_context()
        if use_bpe:
            nl_hecate.save_checkpoint_with_context(final_path, params, cfg, conductor, context)
        else:
            nl_hecate.save_build_checkpoint(final_path, params, cfg, conductor, context)
    sidecar = Path(str(final_path) + ".cursor.json")
    if bpe_loaders:
        sidecar.write_text(json.dumps(
            {"slots": [loader.cursor() for loader in bpe_loaders],
             "level_start_cursor": _level_start_cursor}, indent=2))
    elif active_loader is not None:
        final_cursor = active_loader.cursor()
        final_cursor["level_start_cursor"] = _level_start_cursor
        sidecar.write_text(json.dumps(final_cursor, indent=2))

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
    print(f"  Saved:     {final_path}")
    print(f"{'=' * 60}")

    if jsonl:
        try:
            jsonl.log(event="build_end", steps=len(losses), elapsed=elapsed,
                      tok_per_sec=tok_per_sec,
                      loss_first=losses[0] if losses else None,
                      loss_last=losses[-1] if losses else None)
        finally:
            jsonl.close()
