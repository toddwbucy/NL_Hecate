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
import os
from pathlib import Path
import time

import nl_hecate


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


# ── Default demo text ───────────────────────────────────────────────

DEMO_TEXT = (
    "the cat sat on the mat. "
    "the dog ran in the park. "
    "birds fly high in the sky. "
    "fish swim deep in the sea. "
) * 10


def load_config_file(path: str) -> dict:
    """Load a JSON config file and return as dict."""
    with open(path, "r") as f:
        return json.load(f)


def load_binary_tokens(path: str) -> list[int]:
    """Load a binary file where each byte IS a token ID."""
    with open(path, "rb") as f:
        return list(f.read())


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
    args = parser.parse_args()

    # ── Merge config file + CLI args ─────────────────────────────────
    # Config file provides defaults; CLI args override.
    file_cfg = {}
    if args.config:
        file_cfg = load_config_file(args.config)
        print(f"Loaded config: {args.config}")

    m = file_cfg.get("model", {})
    b = file_cfg.get("build", {})
    d = file_cfg.get("data", {})

    # Resolve each parameter: CLI arg > config file > hardcoded default
    data_path   = args.data        or d.get("path")
    steps       = args.steps       if args.steps is not None else b.get("steps", 500)
    d_model     = args.d_model     if args.d_model is not None else m.get("d_model", 64)
    num_heads   = args.num_heads   if args.num_heads is not None else m.get("num_heads", 4)
    seq_len     = args.seq_len     if args.seq_len is not None else m.get("seq_len", 32)
    window_size = args.window_size if args.window_size is not None else m.get("window_size", 16)
    k           = args.k           if args.k is not None else m.get("k", 1)
    memory_rule = args.memory_rule or m.get("memory_rule", "delta")
    composition = args.composition or m.get("composition", "mag")
    lr          = args.lr          if args.lr is not None else b.get("lr", 0.01)
    seed        = args.seed        if args.seed is not None else b.get("seed", 42)
    save_path   = args.save_path   or b.get("save_path", "checkpoints/model.json")
    save_every  = args.save_every  if args.save_every is not None else b.get("save_every", 0)
    log_every   = args.log_every   if args.log_every is not None else b.get("log_every", 10)

    # chunk_sizes: CLI string "1,8" > config list [1, 8] > default [1]*k
    if args.chunk_sizes is not None:
        chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x]
    elif "chunk_sizes" in m:
        chunk_sizes = m["chunk_sizes"]
    else:
        chunk_sizes = [1] * k

    if len(chunk_sizes) != k:
        print(f"Error: chunk_sizes length {len(chunk_sizes)} must match k={k}")
        return

    # ── Load data ────────────────────────────────────────────────────
    if data_path:
        if data_path.endswith(".bin"):
            token_ids = load_binary_tokens(data_path)
            print(f"Loaded {len(token_ids):,} byte tokens from {data_path}")
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Loaded {len(text):,} chars from {data_path}")
            token_ids = encode(text)
    else:
        text = DEMO_TEXT
        print(f"Using built-in demo text ({len(text):,} chars)")
        token_ids = encode(text)

    if len(token_ids) < seq_len + 1:
        print(f"Error: text too short ({len(token_ids)} tokens < seq_len+1={seq_len + 1})")
        return

    # ── Config ───────────────────────────────────────────────────────
    if d_model % num_heads != 0:
        print(f"Error: d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        return
    head_dim = d_model // num_heads

    cfg = nl_hecate.MAGConfig(
        d_model=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        window_size=window_size,
        vocab_size=256,
        memory_enabled=True,
        k=k,
        chunk_sizes=chunk_sizes,
        memory_rule=memory_rule,
        composition=composition,
    )
    params = nl_hecate.mag_init_params(cfg, seed)

    print(f"\n{'=' * 60}")
    print("NL-Hecate Build")
    print(f"{'=' * 60}")
    print(f"  Model:    d={d_model}, heads={num_heads}, "
          f"seq_len={seq_len}, vocab=256")
    print(f"  Memory:   rule={memory_rule}, composition={composition}, k={k}")
    print(f"  CMS:      chunk_sizes={chunk_sizes}")
    print(f"  Params:   {params.num_params():,}")
    print(f"  Data:     {len(token_ids):,} tokens")
    print(f"  Build:    {steps} steps, lr={lr}")
    print(f"{'=' * 60}\n")

    # ── Stateful CMS build loop ──────────────────────────────────────
    conductor = nl_hecate.Conductor(k, chunk_sizes)
    stream = nl_hecate.VecStream(token_ids)
    conductor.attach_stream(stream)
    context = nl_hecate.ContextState(k, d_model)
    error_buffers = nl_hecate.ErrorBufferList(k, d_model)

    losses = []
    t_start = time.perf_counter()

    for step in range(steps):
        result = conductor.next_chunk(seq_len)
        if result is None:
            break
        input_ids, target_ids, pulse = result

        # Skip truncated chunks at corpus boundary
        if len(input_ids) != seq_len:
            conductor.advance()
            continue

        # Forward + backward (all math in Rust)
        loss, cache = nl_hecate.cms_forward(
            params, cfg, input_ids, target_ids, pulse, context)
        grads = nl_hecate.cms_backward(
            params, cfg, cache, input_ids, target_ids, error_buffers)

        # Outer-loop weight update
        nl_hecate.mag_apply_weight_gradients(params, grads, lr)

        # Apply frozen-level error buffers when levels activate
        error_buffers.apply_for_active(params, pulse, lr)

        # Advance conductor (CS-32: observe-then-advance)
        conductor.advance()
        losses.append(loss)

        # Logging
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss={loss:.4f}")

        # Periodic checkpoint (resumable — includes build state)
        if save_every > 0 and step > 0 and step % save_every == 0:
            p = Path(save_path)
            ckpt_path = str(p.with_stem(f"{p.stem}_step{step}"))
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            nl_hecate.save_build_checkpoint(ckpt_path, params, cfg, conductor, context)
            print(f"  [build checkpoint saved: {ckpt_path}]")

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    total_tokens = len(losses) * seq_len
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # ── Final checkpoint ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    nl_hecate.save_checkpoint(save_path, params, cfg)

    # ── Summary ──────────────────────────────────────────────────────
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
    print(f"  Saved:     {save_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
