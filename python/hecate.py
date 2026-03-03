#!/usr/bin/env python3
"""
NL-Hecate: unified entry point for build + generation.

No train/eval distinction (CS-10). The only difference is where
tokens come from: data pipeline (--build) or keyboard (--chat/--prompt).

Usage:
    # Build (stream data from files — GPU by default)
    python hecate.py --build --config configs/hope_60m.json
    python hecate.py --build --config configs/hope_60m.json --load checkpoints/model_step5000.json

    # Chat (interactive, with optional learning)
    python hecate.py --chat --checkpoint checkpoints/model.json
    python hecate.py --chat --checkpoint checkpoints/model.json --learn --lr 0.0006

    # One-shot generation
    python hecate.py --prompt "Once upon a time" --checkpoint checkpoints/model.json

    # Raw REPL
    python hecate.py --interactive --checkpoint checkpoints/model.json

    # Force CPU (when GPU is available but you want CPU anyway)
    python hecate.py --build --config configs/hope_60m.json --cpu

All math stays in Rust. This script is pure orchestration (CS-18).
"""

import argparse
import os
import random
import sys
import time
import warnings

import nl_hecate
from engine.config import BuildConfig
from engine.tokenizer import BpeTokenizer, load_tokenizer
from engine.generation import generate
from engine.chat import run_chat
from engine.loop import run_build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NL-Hecate: build, chat, or generate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  --build          Stream data from files (requires --config)
  --chat           Interactive multi-turn chat (requires --checkpoint)
  --interactive    Raw REPL mode (requires --checkpoint)
  --prompt TEXT    One-shot generation (requires --checkpoint)
""",
    )

    # ── Mode flags ─────────────────────────────────────────────────────
    mode = parser.add_argument_group("mode (pick one)")
    mode.add_argument("--build", action="store_true",
                      help="Build mode: stream data from files")
    mode.add_argument("--chat", action="store_true",
                      help="Interactive chat mode with ChatML formatting")
    mode.add_argument("--interactive", action="store_true",
                      help="Raw REPL mode (no ChatML)")
    mode.add_argument("--prompt", type=str, default=None,
                      help="One-shot generation from a text prompt")
    mode.add_argument("--validate-config", type=str, default=None, metavar="CONFIG",
                      help="Dry-run config validation (V-01..V-06) without launching build")

    # ── Build arguments ────────────────────────────────────────────────
    build = parser.add_argument_group("build options")
    build.add_argument("--config", type=str, default=None,
                       help="Path to JSON config file")
    build.add_argument("--data", type=str, default=None,
                       help="Path to data file (.txt or .bin)")
    build.add_argument("--steps", type=int, default=None, help="Build steps")
    build.add_argument("--d_model", type=int, default=None)
    build.add_argument("--num_heads", type=int, default=None)
    build.add_argument("--seq_len", type=int, default=None)
    build.add_argument("--window_size", type=int, default=None)
    build.add_argument("--k", type=int, default=None, help="CMS frequency levels")
    build.add_argument("--chunk_sizes", type=str, default=None,
                       help="Comma-separated chunk sizes per level")
    build.add_argument("--memory_rule", type=str, default=None)
    build.add_argument("--composition", type=str, default=None)
    build.add_argument("--attentional_bias", type=str, default=None,
                       help="Inner-loop loss: l2, l1, lp, kl, huber")
    build.add_argument("--retention", type=str, default=None,
                       help="Memory decay: l2_weight_decay, kl_divergence, elastic_net, sphere_normalization")
    build.add_argument("--checkpoint_interval", type=int, default=None,
                       help="Gradient checkpointing interval")
    build.add_argument("--save_path", type=str, default=None)
    build.add_argument("--save_every", type=int, default=None,
                       help="Save checkpoint every N steps (0 = only at end)")
    build.add_argument("--log_every", type=int, default=None)
    build.add_argument("--load", type=str, default=None,
                       help="Resume from a build checkpoint")
    build.add_argument("--log_file", type=str, default=None,
                       help="Path for structured JSONL log")
    build.add_argument("--eval_every", type=int, default=None)
    build.add_argument("--eval_max_chunks", type=int, default=None)

    # ── Model / optimizer (shared between build modes) ─────────────────
    optim = parser.add_argument_group("optimizer")
    optim.add_argument("--optimizer", type=str, default=None,
                       help="'sgd', 'adamw', or 'adamw_gpu'")
    optim.add_argument("--lr", type=float, default=None, help="Learning rate")
    optim.add_argument("--warmup_steps", type=int, default=None)
    optim.add_argument("--weight_decay", type=float, default=None)
    optim.add_argument("--beta1", type=float, default=None)
    optim.add_argument("--beta2", type=float, default=None)
    optim.add_argument("--max_grad_norm", type=float, default=None,
                       help="Max gradient L2 norm for clipping (0 = disabled)")

    # ── HOPE / self-referential ────────────────────────────────────────
    hope = parser.add_argument_group("HOPE self-referential")
    hope.add_argument("--projection_kind", type=str, default=None,
                      help="'static' or 'adaptive'")
    hope.add_argument("--self_generated_values", action="store_true", default=None,
                      help="Enable Phase 3 self-generated values")
    hope.add_argument("--no_self_generated_values", action="store_false",
                      dest="self_generated_values")
    hope.add_argument("--self_ref_chunk_size", type=int, default=None)
    hope.add_argument("--momentum_kind", type=str, default=None)
    hope.add_argument("--momentum_d_hidden", type=int, default=None)

    # ── Generation arguments ───────────────────────────────────────────
    gen = parser.add_argument_group("generation options")
    gen.add_argument("--checkpoint", type=str, default=None,
                     help="Path to model checkpoint (.json)")
    gen.add_argument("--max_tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top_k", type=int, default=50)
    gen.add_argument("--stateless", action="store_true",
                     help="Disable CMS memory in chat mode")
    gen.add_argument("--learn", action="store_true",
                     help="Enable continuous outer-loop learning during generation (CS-10)")
    gen.add_argument("--tokenizer", type=str, default=None,
                     help="Path to BPE tokenizer.json")
    gen.add_argument("--data_dir", type=str, default=None,
                     help="Data directory containing tokenizer.json")

    # ── Runtime ────────────────────────────────────────────────────────
    rt = parser.add_argument_group("runtime")
    rt.add_argument("--cpu", action="store_true",
                    help="Force CPU mode (default: GPU when available)")
    rt.add_argument("--gpu", action="store_true",
                    help="(deprecated, now the default) Use GPU-resident model")
    rt.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducible generation")

    return parser


def _load_model(args):
    """Load checkpoint, tokenizer, and optionally GPU model."""
    if not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading checkpoint: {args.checkpoint}")
    params, cfg = nl_hecate.load_checkpoint(args.checkpoint)
    print(f"  Model: d={cfg.d_model}, heads={cfg.num_heads}, "
          f"seq_len={cfg.seq_len}, vocab={cfg.vocab_size}")
    print(f"  Memory: rule={cfg.memory_rule}, composition={cfg.composition}, k={cfg.k}")
    print(f"  Params: {params.num_params():,}")

    tokenizer = load_tokenizer(args.tokenizer, args.data_dir)
    tok_type = "BPE" if isinstance(tokenizer, BpeTokenizer) else "byte-level"
    print(f"  Tokenizer: {tok_type}")

    gpu_model = None
    use_gpu = not args.cpu and hasattr(nl_hecate, "GpuModel")
    if use_gpu:
        gpu_model = nl_hecate.GpuModel.from_params(params, cfg)
        print("  Device: GPU")
    else:
        print("  Device: CPU")

    return params, cfg, tokenizer, gpu_model


def _validate_config_cmd(path: str) -> int:
    """Dry-run config validation (V-01..V-06) without launching a build.

    Loads the config, captures any ValueError / warnings, and prints a
    human-readable report. Returns 1 if errors exist, 0 otherwise.
    """
    from engine.config import (
        _GPU_CAPABLE, _TIER_2B, _TIER_3_RULES,
    )

    errors: list[str] = []
    warn_msgs: list[str] = []

    bcfg: "BuildConfig | None" = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            bcfg = BuildConfig.from_file(path)
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
        if bcfg is not None:
            try:
                bcfg.validate_gpu_tier()  # V-05: explicit after load (no CLI overrides)
            except ValueError as exc:
                errors.append(str(exc))
        for w in caught:
            warn_msgs.append(str(w.message))

    print(f"\nConfig: {path}")

    if bcfg is not None:
        rule = bcfg.memory_rule
        if rule in _GPU_CAPABLE:
            tier_label = "Tier 1" if rule == "titans" else "Tier 2a — GPU-capable"
        elif rule in _TIER_2B:
            tier_label = "Tier 2b — CPU only"
        elif rule in _TIER_3_RULES:
            tier_label = "Tier 3 — research stub"
        else:
            tier_label = "unknown tier"

        def check(ok: bool) -> str:
            return "✓" if ok else "✗"
        print(f"  memory_rule:      {rule:<20} ({tier_label})")
        print(f"  composition:      {bcfg.composition:<20} {check(True)}")
        bias = bcfg.attentional_bias or "(default)"
        ret  = bcfg.retention or "(default)"
        print(f"  attentional_bias: {bias:<20} {check(True)}")
        print(f"  retention:        {ret:<20} {check(True)}")
        print(f"  momentum:         {bcfg.momentum_kind:<20} {check(True)}")
        print(f"  k:                {bcfg.k:<20} {check(True)}")
        device = "cuda" if bcfg.gpu else "cpu"
        gpu_ok = not bcfg.gpu or rule in _GPU_CAPABLE
        print(f"  device:           {device:<20} {check(gpu_ok)}")

    print()
    for msg in warn_msgs:
        print(f"WARNING: {msg}")
    for msg in errors:
        print(f"ERROR: {msg}")

    n_err = len(errors)
    n_warn = len(warn_msgs)
    device = "cuda" if (bcfg is not None and bcfg.gpu) else "cpu"
    if n_err == 0:
        print(f"\n{n_err} errors, {n_warn} warnings. Config is VALID for {device}.")
        return 0
    else:
        print(f"\n{n_err} error(s), {n_warn} warning(s). Config is INVALID.")
        return 1


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── validate-config dry-run ────────────────────────────────────────
    if args.validate_config:
        rc = _validate_config_cmd(args.validate_config)
        sys.exit(rc)

    # ── Validate mode ─────────────────────────────────────────────────
    mode_count = sum([args.build, args.chat, args.interactive, args.prompt is not None])
    if mode_count == 0:
        parser.print_help()
        sys.exit(0)
    if mode_count > 1:
        print("Error: pick exactly one mode: --build, --chat, --interactive, or --prompt")
        sys.exit(1)

    # ── Build mode ────────────────────────────────────────────────────
    if args.build:
        if args.config:
            bcfg = BuildConfig.from_file(args.config)
            print(f"Loaded config: {args.config}")
        else:
            bcfg = BuildConfig()
        bcfg.apply_cli(args)
        run_build(bcfg)
        return

    # ── Generation modes require --checkpoint ─────────────────────────
    if not args.checkpoint:
        print("Error: --chat, --interactive, and --prompt require --checkpoint")
        sys.exit(1)

    params, cfg, tokenizer, gpu_model = _load_model(args)

    # Default lr for learn mode
    lr = args.lr if args.lr is not None else 0.0006
    wd = args.weight_decay if args.weight_decay is not None else 0.1
    learn_kwargs = dict(lr=lr, weight_decay=wd) if args.learn else None

    # ── Chat mode ─────────────────────────────────────────────────────
    if args.chat:
        run_chat(
            params, cfg, tokenizer, gpu_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stateless=args.stateless,
            learn=args.learn,
            learn_kwargs=learn_kwargs,
        )
        return

    # ── Interactive REPL ──────────────────────────────────────────────
    if args.interactive:
        print(f"\nInteractive mode (temp={args.temperature}, "
              f"max_tokens={args.max_tokens})")
        print("Type a prompt and press Enter. Ctrl+C or 'quit' to exit.\n")

        while True:
            try:
                prompt = input(">>> ")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if prompt.strip().lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not prompt:
                continue

            prompt_tokens = tokenizer.encode(prompt)
            t0 = time.perf_counter()
            output = generate(params, cfg, prompt_tokens,
                              args.max_tokens, args.temperature,
                              top_k=args.top_k, use_cms=True,
                              gpu_model=gpu_model,
                              learn=args.learn, learn_kwargs=learn_kwargs)
            t1 = time.perf_counter()

            text = tokenizer.decode(output)
            gen_tokens = len(output) - len(prompt_tokens)
            tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0
            print(f"{text}")
            print(f"  [{gen_tokens} tokens, {tps:.0f} tok/s]\n")
        return

    # ── One-shot prompt ───────────────────────────────────────────────
    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
        print(f"\nPrompt: {repr(args.prompt)}")
        print(f"Temperature: {args.temperature}")
        if args.learn:
            print(f"Learning: lr={lr}, weight_decay={wd}")

        t0 = time.perf_counter()
        output = generate(params, cfg, prompt_tokens,
                          args.max_tokens, args.temperature,
                          top_k=args.top_k, use_cms=True,
                          gpu_model=gpu_model,
                          learn=args.learn, learn_kwargs=learn_kwargs)
        t1 = time.perf_counter()

        text = tokenizer.decode(output)
        gen_tokens = len(output) - len(prompt_tokens)
        tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

        print(f"\nOutput: {text}")
        print(f"\n  [{gen_tokens} tokens in {t1-t0:.3f}s, {tps:.0f} tok/s]")
        return


if __name__ == "__main__":
    main()
