"""Build loop: stateful CMS training with optional GPU acceleration."""

import gc
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import nl_hecate

from engine.config import BuildConfig, cosine_lr
from engine.data import BpeDataLoader, CursorMismatchError, CursorOutOfBounds, DEMO_TEXT, MmapTokenStream
from engine.evaluation import (
    evaluate, evaluate_numpy, print_level_metrics,
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


def run_build(bcfg: BuildConfig):
    """Execute the full build loop. All state managed internally."""

    import numpy as np

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
        if bcfg.eval_every > 0:
            val_path = Path(bcfg.data_path) / "val_tokens.npy"
            if val_path.exists():
                val_loader = BpeDataLoader(bcfg.data_path, split="val")
                print(f"Loaded val set: {len(val_loader):,} tokens")
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
            val_loader = val_bytes  # evaluate() accepts bytes for byte-level
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
    if bcfg.load:
        print(f"Loading checkpoint: {bcfg.load}")
        if use_bpe:
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
            if sidecar.exists() and bpe_loader is not None:
                cursor = json.loads(sidecar.read_text())
                try:
                    bpe_loader.restore(cursor)
                    print(f"  Resuming from step {resume_step}")
                    print(f"  Stream position: {cursor['position']:,} / {cursor['total_tokens']:,} tokens")
                except (CursorMismatchError, CursorOutOfBounds) as e:
                    print(f"  ERROR: cursor mismatch — {e}")
                    return
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
        # Apply theta clamps from BuildConfig onto loaded cfg (allows
        # adding clamps to an existing checkpoint that didn't have them).
        # MAGConfig is frozen, so rebuild if clamps changed.
        if bcfg.theta_floor is not None or bcfg.theta_ceil is not None or bcfg.m_norm_max is not None:
            # Use bcfg value when explicitly set; fall back to loaded cfg so we
            # never silently wipe clamp values that were already baked into the
            # checkpoint (e.g. resuming without --m_norm_max still preserves it).
            theta_floor = bcfg.theta_floor if bcfg.theta_floor is not None else list(cfg.theta_floor)
            theta_ceil  = bcfg.theta_ceil  if bcfg.theta_ceil  is not None else list(cfg.theta_ceil)
            m_norm_max  = bcfg.m_norm_max  if bcfg.m_norm_max  is not None else list(cfg.m_norm_max)
            cfg = nl_hecate.MAGConfig(
                d_model=cfg.d_model, num_heads=cfg.num_heads,
                head_dim=cfg.head_dim, seq_len=cfg.seq_len,
                window_size=cfg.window_size, vocab_size=cfg.vocab_size,
                memory_enabled=cfg.memory_enabled, k=cfg.k,
                chunk_sizes=list(cfg.chunk_sizes),
                memory_rule=cfg.memory_rule, composition=cfg.composition,
                checkpoint_interval=bcfg.checkpoint_interval,
                projection_kind=cfg.projection_kind,
                self_generated_values=cfg.self_generated_values,
                self_ref_chunk_size=cfg.self_ref_chunk_size,
                momentum_kind=cfg.momentum_kind,
                momentum_d_hidden=cfg.momentum_d_hidden,
                intermediate_size=bcfg.intermediate_size,
                theta_floor=theta_floor,
                theta_ceil=theta_ceil,
                m_norm_max=m_norm_max,
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
            projection_kind=bcfg.projection_kind,
            self_generated_values=bcfg.self_generated_values,
            self_ref_chunk_size=bcfg.self_ref_chunk_size,
            momentum_kind=bcfg.momentum_kind,
            momentum_d_hidden=bcfg.momentum_d_hidden,
            theta_floor=bcfg.theta_floor,
            theta_ceil=bcfg.theta_ceil,
            intermediate_size=bcfg.intermediate_size,
            m_norm_max=bcfg.m_norm_max,
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
    if bcfg.theta_floor is not None or bcfg.theta_ceil is not None:
        print(f"  θ clamps: floor={bcfg.theta_floor}, ceil={bcfg.theta_ceil}")
    if bcfg.gate_warmup_theta_floor_init is not None:
        print(f"  GateWarmup: theta_floor_init={bcfg.gate_warmup_theta_floor_init}, "
              f"decay_steps={bcfg.gate_warmup_decay_steps}, "
              f"falsification_step={bcfg.gate_warmup_falsification_step}")
    if len(cfg.m_norm_max) > 0:
        print(f"  M-norm:   max={list(cfg.m_norm_max)}")
    print(f"  Params:   {params.num_params():,}")
    data_len = len(bpe_loader) if use_bpe else len(token_ids)
    print(f"  Data:     {data_len:,} tokens" +
          (f" (ShareGPT BPE, {bcfg.data_format})" if use_bpe else ""))
    use_gpu = bcfg.gpu and hasattr(nl_hecate, "GpuModel")
    # Auto-promote adamw → adamw_gpu when on GPU (no reason to round-trip to CPU)
    if use_gpu and bcfg.optimizer == "adamw":
        bcfg.optimizer = "adamw_gpu"
    if bcfg.optimizer == "adamw_gpu" and not use_gpu:
        raise RuntimeError(
            "optimizer=adamw_gpu requires GPU and a CUDA-enabled build"
        )
    if bcfg.load and use_gpu and not use_bpe:
        raise RuntimeError(
            "GPU resume with context restore is not yet implemented for byte-level builds. "
            "Use CPU resume (--cpu) or start a fresh GPU build."
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
        conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
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
    # of the corpus. batch_size=1 falls through to the scalar bpe_loader path.
    bpe_loaders: list[BpeDataLoader] = []
    if use_bpe and bcfg.batch_size > 1 and bpe_loader is not None:
        slot_size = bpe_loader.total_tokens // bcfg.batch_size
        for b in range(bcfg.batch_size):
            loader_b = BpeDataLoader(bcfg.data_path, split="train")
            loader_b.position = b * slot_size
            bpe_loaders.append(loader_b)

    gpu_model = None
    if use_gpu:
        periodic = (bcfg.memory_reset == "periodic")
        gpu_model = nl_hecate.GpuModel.from_params(
            params, cfg, batch_size=bcfg.batch_size, memory_reset=periodic)

    error_buffers = nl_hecate.ErrorBufferList(bcfg.k, bcfg.d_model)

    next_doc_idx = 1
    if doc_starts is not None and resume_step > 0:
        byte_pos = resume_step * bcfg.seq_len
        next_doc_idx = int(np.searchsorted(doc_starts, byte_pos, side="right"))

    adamw_opt = None
    use_adamw_gpu = (bcfg.optimizer == "adamw_gpu")
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
            "lr": bcfg.lr, "steps": bcfg.steps, "params": params.num_params(),
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
                chunk = bpe_loader.next_chunk(bcfg.seq_len)
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
                and step < resume_step + bcfg.gate_warmup_decay_steps):
            alpha = 1.0 - (step - resume_step) / bcfg.gate_warmup_decay_steps
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
                    projection_kind=cfg.projection_kind,
                    self_generated_values=cfg.self_generated_values,
                    self_ref_chunk_size=cfg.self_ref_chunk_size,
                    momentum_kind=cfg.momentum_kind,
                    momentum_d_hidden=cfg.momentum_d_hidden,
                    intermediate_size=bcfg.intermediate_size,
                    theta_floor=warmup_floor,
                    theta_ceil=list(cfg.theta_ceil) if list(cfg.theta_ceil) else None,
                    m_norm_max=list(cfg.m_norm_max) if list(cfg.m_norm_max) else None,
                )
        elif (bcfg.gate_warmup_theta_floor_init is not None
              and step == resume_step + bcfg.gate_warmup_decay_steps
              and gpu_model is not None):
            # Phase 3 start: floor is fully decayed — restore permanent floor
            final_floor = bcfg.theta_floor if bcfg.theta_floor is not None else [0.0] * bcfg.k
            gpu_model.update_theta_floor(final_floor)

        use_cosine = (adamw_opt is not None or use_adamw_gpu)
        current_lr = cosine_lr(step, bcfg.warmup_steps, end_step, bcfg.lr) if use_cosine else bcfg.lr

        g_norm = 0.0
        # Compute log_this before step_adamw so we can pass collect_level_gnorms
        # only when we'll actually use the per-level norms this step.
        log_this = (step % bcfg.log_every == 0 or step == end_step - 1
                    or (step < 100 and step % 10 == 0))

        if gpu_model is not None and use_adamw_gpu:
            loss, g_norm = gpu_model.step_adamw(
                input_ids, target_ids, pulse, current_lr,
                beta1=bcfg.beta1, beta2=bcfg.beta2, eps=1e-8,
                weight_decay=bcfg.weight_decay,
                max_grad_norm=bcfg.max_grad_norm,
                collect_level_gnorms=log_this,
            )
            # Component 1+2: collect per-level gnorms for monitoring (active steps only)
            if log_this and hasattr(gpu_model, "level_grad_norms"):
                level_gnorms = gpu_model.level_grad_norms()
                for i, n in enumerate(level_gnorms):
                    if (i < len(level_gnorm_history)
                            and i < len(pulse.active_levels)
                            and pulse.active_levels[i]):
                        level_gnorm_history[i].append(n)
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
            jsonl.log(**log_fields)

        # Gate warmup falsification checkpoint (09_gate_warmup.md §5)
        if (bcfg.gate_warmup_theta_floor_init is not None
                and bcfg.gate_warmup_falsification_step > 0
                and step == bcfg.gate_warmup_falsification_step
                and gpu_model is not None
                and hasattr(gpu_model, "gate_biases")):
            biases = gpu_model.gate_biases()
            l2_theta = math.log1p(math.exp(biases[2][1])) if len(biases) > 2 else 0.0
            l3_theta = math.log1p(math.exp(biases[3][1])) if len(biases) > 3 else 0.0
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

        if (bcfg.eval_every > 0 and val_loader is not None
                and step > 0 and step % bcfg.eval_every == 0):
            saved_ctx = None
            try:
                if gpu_model is not None:
                    saved_ctx = gpu_model.to_host_context()
                    gpu_model.reset_context()
                eval_loss, eval_ppl = evaluate(
                    gpu_model, bcfg, val_loader, bcfg.eval_max_chunks,
                    val_doc_starts=val_doc_starts)
            finally:
                if gpu_model is not None and saved_ctx is not None:
                    gpu_model.upload_context(saved_ctx)
            print(f"  [eval] step {step:5d}  loss={eval_loss:.4f}  ppl={eval_ppl:.1f}")
            if gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                print_level_metrics(gpu_model, bcfg.k)
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
                    for prompt_text in EVAL_PROMPTS:
                        full_restore(gpu_model, snapshot)
                        gpu_model.reset_optimizer()
                        prompt_ids = tokenizer.encode(prompt_text)
                        result = probe_within_generation(
                            gpu_model, cfg, prompt_ids, tokenizer,
                            max_tokens=60, temperature=0.7, lr=bcfg.lr)
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
                        max_tokens=30, temperature=0.7, lr=bcfg.lr)
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
                and use_bpe and bcfg.data_path):
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

        # Periodic checkpoint
        if bcfg.save_every > 0 and step > 0 and step % bcfg.save_every == 0:
            if gpu_model is not None:
                params = gpu_model.to_host_params()
                context = gpu_model.to_host_context()
            p = Path(_safetensors_path(bcfg.save_path))
            ckpt_path = str(p.with_stem(f"{p.stem}_step{step}"))
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            if use_bpe:
                nl_hecate.save_checkpoint(ckpt_path, params, cfg)
            else:
                nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)
            if bpe_loader is not None:
                sidecar = Path(str(ckpt_path) + ".cursor.json")
                sidecar.write_text(json.dumps(bpe_loader.cursor(), indent=2))
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
                    # Slice to single seq_len chunk — forward() expects exactly
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
    if gpu_model is not None:
        params = gpu_model.to_host_params()
        if not use_bpe:
            context = gpu_model.to_host_context()
    final_path = _safetensors_path(bcfg.save_path)
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    if use_bpe:
        nl_hecate.save_checkpoint(final_path, params, cfg)
    else:
        nl_hecate.save_build_checkpoint(final_path, params, cfg, conductor, context)
    if bpe_loader is not None:
        sidecar = Path(str(final_path) + ".cursor.json")
        sidecar.write_text(json.dumps(bpe_loader.cursor(), indent=2))

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
