#!/usr/bin/env python3
"""
NL-Hecate build script: train a model on text using stateful CMS, save checkpoint.

Usage:
    python build.py --data sample.txt --steps 500 --d_model 64 --seq_len 32 --lr 0.01
    python build.py --steps 200  # uses built-in demo text

All math stays in Rust. This script is pure orchestration (CS-18).
"""

import argparse
import math
import os
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


def main():
    parser = argparse.ArgumentParser(description="NL-Hecate build script")
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--steps", type=int, default=500, help="Build steps")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--window_size", type=int, default=16, help="SWA window size")
    parser.add_argument("--k", type=int, default=1, help="CMS frequency levels")
    parser.add_argument("--memory_rule", type=str, default="delta", help="Memory rule")
    parser.add_argument("--composition", type=str, default="mag", help="Composition pattern")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="checkpoints/model.json",
                        help="Checkpoint save path")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────
    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text):,} chars from {args.data}")
    else:
        text = DEMO_TEXT
        print(f"Using built-in demo text ({len(text):,} chars)")

    token_ids = encode(text)
    if len(token_ids) < args.seq_len + 1:
        print(f"Error: text too short ({len(token_ids)} tokens < seq_len+1={args.seq_len + 1})")
        return

    # ── Config ───────────────────────────────────────────────────────
    head_dim = args.d_model // args.num_heads
    chunk_sizes = [1] * args.k  # level 0 fires every step

    cfg = nl_hecate.MAGConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        head_dim=head_dim,
        seq_len=args.seq_len,
        window_size=args.window_size,
        vocab_size=256,
        memory_enabled=True,
        k=args.k,
        chunk_sizes=chunk_sizes,
        memory_rule=args.memory_rule,
        composition=args.composition,
    )
    params = nl_hecate.mag_init_params(cfg, args.seed)

    print(f"\n{'=' * 60}")
    print("NL-Hecate Build")
    print(f"{'=' * 60}")
    print(f"  Model:    d={args.d_model}, heads={args.num_heads}, "
          f"seq_len={args.seq_len}, vocab=256")
    print(f"  Memory:   rule={args.memory_rule}, composition={args.composition}, k={args.k}")
    print(f"  Params:   {params.num_params():,}")
    print(f"  Data:     {len(token_ids):,} tokens")
    print(f"  Build:    {args.steps} steps, lr={args.lr}")
    print(f"{'=' * 60}\n")

    # ── Stateful CMS build loop ──────────────────────────────────────
    conductor = nl_hecate.Conductor(args.k, chunk_sizes)
    stream = nl_hecate.VecStream(token_ids)
    conductor.attach_stream(stream)
    context = nl_hecate.ContextState(args.k, args.d_model)
    error_buffers = nl_hecate.ErrorBufferList(args.k, args.d_model)

    losses = []
    t_start = time.perf_counter()

    for step in range(args.steps):
        result = conductor.next_chunk(args.seq_len)
        if result is None:
            # Should not happen with VecStream (auto-wraps), but be defensive
            break
        input_ids, target_ids, pulse = result

        # Skip truncated chunks at corpus boundary
        if len(input_ids) != args.seq_len:
            conductor.advance()
            continue

        # Forward + backward (all math in Rust)
        loss, cache = nl_hecate.cms_forward(
            params, cfg, input_ids, target_ids, pulse, context)
        grads = nl_hecate.cms_backward(
            params, cfg, cache, input_ids, target_ids, error_buffers)

        # Outer-loop weight update
        nl_hecate.mag_apply_weight_gradients(params, grads, args.lr)

        # Apply frozen-level error buffers when levels activate
        error_buffers.apply_for_active(params, pulse, args.lr)

        # Advance conductor (CS-32: observe-then-advance)
        conductor.advance()
        losses.append(loss)

        # Logging
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss={loss:.4f}")

        # Periodic checkpoint
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt_path = args.save_path.replace(".json", f"_step{step}.json")
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            nl_hecate.save_checkpoint(ckpt_path, params, cfg)
            print(f"  [checkpoint saved: {ckpt_path}]")

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    total_tokens = len(losses) * args.seq_len
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # ── Final checkpoint ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    nl_hecate.save_checkpoint(args.save_path, params, cfg)

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
    print(f"  Saved:     {args.save_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
