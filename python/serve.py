#!/usr/bin/env python3
"""
NL-Hecate serve script: load a checkpoint and generate text from prompts.

Usage:
    python serve.py --checkpoint checkpoints/model.json --prompt "the cat" --max_tokens 64
    python serve.py --checkpoint checkpoints/model.json --interactive

All math stays in Rust. This script is pure orchestration (CS-18).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import nl_hecate


# ── Tokenizer abstraction ──────────────────────────────────────────

class ByteTokenizer:
    """Byte-level tokenizer (vocab_size=256). No external dependencies."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        out = []
        for b in token_ids:
            if 32 <= b < 127:
                out.append(chr(b))
            elif b == 10:
                out.append("\n")
            else:
                out.append("?")
        return "".join(out)


class BpeTokenizer:
    """BPE tokenizer loaded from a tokenizers JSON file."""

    def __init__(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)


def load_tokenizer(tokenizer_path: str | None = None,
                   data_dir: str | None = None) -> ByteTokenizer | BpeTokenizer:
    """Load the appropriate tokenizer. BPE if path provided, else byte-level."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        return BpeTokenizer(tokenizer_path)
    if data_dir:
        bpe_path = Path(data_dir) / "tokenizer.json"
        if bpe_path.exists():
            return BpeTokenizer(str(bpe_path))
    return ByteTokenizer()


# ── Byte-level encode/decode (backward compat) ────────────────────

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


# ── Generation ──────────────────────────────────────────────────────

def generate(
    params,
    cfg,
    prompt_tokens: list[int],
    max_tokens: int = 64,
    temperature: float = 0.8,
    use_cms: bool = False,
    gpu_model=None,
) -> list[int]:
    """
    Autoregressive byte generation.

    Args:
        params: MAGParams checkpoint
        cfg: MAGConfig
        prompt_tokens: encoded prompt
        max_tokens: tokens to generate
        temperature: 0 = greedy, >0 = softmax sampling
        use_cms: if True, use stateful cms_forward with persistent memory
        gpu_model: optional GpuModel for GPU-accelerated generation
    """
    seq = list(prompt_tokens)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len

    # CMS conductor for pulse generation (needed for both CPU CMS and GPU paths)
    conductor = None
    context = None
    if use_cms or gpu_model is not None:
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)
        if not gpu_model:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)

    for _ in range(max_tokens):
        # Take last seq_len tokens as context window
        ctx = seq[-seq_len:]
        # Pad if shorter than seq_len
        while len(ctx) < seq_len:
            ctx = [0, *ctx]

        # Forward pass — target_ids unused for generation, use ctx as dummy
        if gpu_model is not None:
            pulse = conductor.pulse()
            _loss, logits = gpu_model.forward(ctx, ctx, pulse)
            conductor.advance()
            last_logits = logits[(seq_len - 1) * vocab: seq_len * vocab]
        elif use_cms:
            pulse = conductor.pulse()
            _loss, cache = nl_hecate.cms_forward(params, cfg, ctx, ctx, pulse, context)
            conductor.advance()
            logits = cache.get_logits()
            last_logits = logits[(seq_len - 1) * vocab: seq_len * vocab]
        else:
            _loss, cache = nl_hecate.mag_forward(params, cfg, ctx, ctx)
            logits = cache.get_logits()
            last_logits = logits[(seq_len - 1) * vocab: seq_len * vocab]

        # Sample next token
        if temperature <= 0:
            # Greedy
            next_tok = max(range(vocab), key=lambda i: last_logits[i])
        else:
            # Temperature-scaled softmax sampling
            max_l = max(last_logits)
            exps = [math.exp((logit - max_l) / temperature) for logit in last_logits]
            total = sum(exps)
            probs = [e / total for e in exps]
            r = random.random()
            cumsum = 0.0
            next_tok = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    next_tok = i
                    break

        seq.append(next_tok)

    return seq


def main():
    parser = argparse.ArgumentParser(description="NL-Hecate serve script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.json)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--use_cms", action="store_true",
                        help="Use stateful CMS forward (memory-augmented generation)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU-resident model for fast generation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to BPE tokenizer.json (auto-detected from data dir)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory containing tokenizer.json")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    params, cfg = nl_hecate.load_checkpoint(args.checkpoint)
    print(f"  Model: d={cfg.d_model}, heads={cfg.num_heads}, "
          f"seq_len={cfg.seq_len}, vocab={cfg.vocab_size}")
    print(f"  Memory: rule={cfg.memory_rule}, composition={cfg.composition}, k={cfg.k}")
    print(f"  Params: {params.num_params():,}")

    # ── Load tokenizer ─────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer, args.data_dir)
    tok_type = "BPE" if isinstance(tokenizer, BpeTokenizer) else "byte-level"
    print(f"  Tokenizer: {tok_type}")

    gpu_model = None
    if args.gpu and hasattr(nl_hecate, "GpuModel"):
        gpu_model = nl_hecate.GpuModel.from_params(params, cfg)
        print("  Device: GPU")
    else:
        print("  Device: CPU")

    mode = "CMS (memory-augmented)" if args.use_cms else "stateless"
    if gpu_model:
        mode = "GPU (CMS)"
    print(f"  Generation mode: {mode}")

    # ── Generate ─────────────────────────────────────────────────────
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
                              args.max_tokens, args.temperature, args.use_cms,
                              gpu_model=gpu_model)
            t1 = time.perf_counter()

            text = tokenizer.decode(output)
            gen_tokens = len(output) - len(prompt_tokens)
            tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0
            print(f"{text}")
            print(f"  [{gen_tokens} tokens, {tps:.0f} tok/s]\n")

    elif args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt)
        print(f"\nPrompt: {repr(args.prompt)}")
        print(f"Temperature: {args.temperature}")

        t0 = time.perf_counter()
        output = generate(params, cfg, prompt_tokens,
                          args.max_tokens, args.temperature, args.use_cms,
                          gpu_model=gpu_model)
        t1 = time.perf_counter()

        text = tokenizer.decode(output)
        gen_tokens = len(output) - len(prompt_tokens)
        tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

        print(f"\nOutput: {text}")
        print(f"\n  [{gen_tokens} tokens in {t1-t0:.3f}s, {tps:.0f} tok/s]")

    else:
        print("\nError: provide --prompt or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
