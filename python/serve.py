#!/usr/bin/env python3
"""
NL-Hecate serve script: load a checkpoint and generate text from prompts.

Usage:
    python serve.py --checkpoint checkpoints/model.json --prompt "the cat" --max_tokens 64
    python serve.py --checkpoint checkpoints/model.json --interactive
    python serve.py --checkpoint checkpoints/model.json --chat --gpu --data_dir data/sharegpt

Chat mode wraps user/assistant exchanges in ChatML format. By default, CMS
memory carries conversation context across turns (stateful). Pass --stateless
to disable memory and re-send full conversation history each turn instead.

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

# ChatML special token IDs (must match tokenizer training in prepare_sharegpt.py)
IM_START = 0  # <|im_start|>
IM_END = 1    # <|im_end|>
PAD = 2       # <|pad|>


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


# ── Cached generation (KV cache for GPU decode) ───────────────────

def generate_cached(
    gpu_model,
    cfg,
    prompt_tokens: list[int],
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    stop_token: int | None = None,
    conductor=None,
) -> list[int]:
    """
    KV-cached autoregressive generation on GPU.

    Processes the prompt once (prefill), then generates one token at a time
    using cached K/V projections. ~100-500x faster per decode step vs full forward.
    """
    seq = list(prompt_tokens)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len

    if conductor is None:
        conductor = nl_hecate.Conductor(
            cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    try:
        # Pad/truncate prompt to seq_len for prefill
        ctx = seq[-seq_len:]
        while len(ctx) < seq_len:
            ctx = [PAD, *ctx]

        # Prefill: process full prompt, populate KV cache
        pulse = conductor.pulse()
        last_logits = gpu_model.prefill(ctx, pulse)
        conductor.advance()

        for _ in range(max_tokens):
            # Sample next token from last-position logits
            next_tok = _sample_token(last_logits, vocab, temperature, top_k)

            if stop_token is not None and next_tok == stop_token:
                break

            seq.append(next_tok)

            # Decode: single-token forward using KV cache
            pulse = conductor.pulse()
            last_logits = gpu_model.decode_token(next_tok, pulse)
            conductor.advance()

    finally:
        # Always reset cache on exit
        gpu_model.reset_cache()

    return seq


# ── Generation ──────────────────────────────────────────────────────

def generate(
    params,
    cfg,
    prompt_tokens: list[int],
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    stop_token: int | None = None,
    use_cms: bool = False,
    gpu_model=None,
    conductor=None,
    context=None,
) -> list[int]:
    """
    Autoregressive generation with optional top-k sampling and stop token.

    Args:
        params: MAGParams checkpoint
        cfg: MAGConfig
        prompt_tokens: encoded prompt
        max_tokens: tokens to generate
        temperature: 0 = greedy, >0 = softmax sampling
        top_k: if >0, sample only from top-k logits
        stop_token: stop generation when this token is sampled
        use_cms: if True, use stateful cms_forward with persistent memory
        gpu_model: optional GpuModel for GPU-accelerated generation
        conductor: external Conductor (for persistent state across calls)
        context: external ContextState (for persistent state across calls)
    """
    # Delegate to KV-cached path for GPU models
    if gpu_model is not None:
        return generate_cached(
            gpu_model, cfg, prompt_tokens, max_tokens,
            temperature, top_k, stop_token, conductor,
        )

    seq = list(prompt_tokens)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len

    # CMS conductor for pulse generation — use external if provided
    if conductor is None and use_cms:
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)
        owns_conductor = True
        if context is None:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)

    for _ in range(max_tokens):
        # Take last seq_len tokens as context window
        ctx = seq[-seq_len:]
        # Pad if shorter than seq_len
        while len(ctx) < seq_len:
            ctx = [PAD, *ctx]

        # Forward pass — target_ids unused for generation, use ctx as dummy
        if use_cms:
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
        next_tok = _sample_token(last_logits, vocab, temperature, top_k)

        if stop_token is not None and next_tok == stop_token:
            break

        seq.append(next_tok)

    return seq


def _sample_token(logits: list[float], vocab: int, temperature: float,
                  top_k: int) -> int:
    """Sample a single token from logits with temperature and optional top-k."""
    if temperature <= 0:
        return max(range(vocab), key=lambda i: logits[i])

    # Build (index, logit) pairs
    indexed = list(enumerate(logits[:vocab]))

    # Top-k filtering
    if top_k > 0:
        indexed.sort(key=lambda x: x[1], reverse=True)
        indexed = indexed[:top_k]

    # Temperature-scaled softmax
    max_logit = max(logit for _, logit in indexed)
    weighted = [(idx, math.exp((logit - max_logit) / temperature)) for idx, logit in indexed]
    total = sum(w for _, w in weighted)

    r = random.random() * total  # noqa: S311
    cumsum = 0.0
    for idx, w in weighted:
        cumsum += w
        if r < cumsum:
            return idx
    return weighted[-1][0]


# ── ChatML helpers ────────────────────────────────────────────────

def chatml_encode_turn(tokenizer, role: str, content: str) -> list[int]:
    """Encode a single ChatML turn using explicit special token IDs.

    Mirrors prepare_sharegpt.py's format_chatml() to avoid BPE splitting
    special token strings into sub-tokens.
    """
    ids = [IM_START]
    ids.extend(tokenizer.encode(f"{role}\n"))
    ids.extend(tokenizer.encode(content))
    ids.append(IM_END)
    ids.extend(tokenizer.encode("\n"))
    return ids


def chatml_encode_prompt(tokenizer, role: str) -> list[int]:
    """Encode the start of a turn (no content, no end): <|im_start|>role\\n"""
    ids = [IM_START]
    ids.extend(tokenizer.encode(f"{role}\n"))
    return ids


# ── Chat mode ─────────────────────────────────────────────────────

def run_chat(
    params, cfg, tokenizer, gpu_model,
    max_tokens: int, temperature: float, top_k: int, stateless: bool,
):
    """
    Interactive multi-turn chat with ChatML formatting.

    Stateful mode (default): CMS memory carries conversation context.
      Only the new user message is fed each turn. Memory accumulates state.
      Prompt size: constant per turn.

    Stateless mode (--stateless): Full conversation history re-sent each turn.
      No CMS memory persistence. Traditional transformer-style chat.
      Prompt size: grows linearly with conversation length.
    """
    seq_len = cfg.seq_len
    history_tokens: list[int] = []  # accumulated token history (stateless mode)
    turn_count = 0

    # Persistent CMS state for stateful mode
    conductor = None
    context = None
    if not stateless:
        conductor = nl_hecate.Conductor(
            cfg.k,
            list(cfg.chunk_sizes) if hasattr(cfg, "chunk_sizes") else [1] * cfg.k,
        )
        if gpu_model is None:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)

    mode_label = "stateless (full history)" if stateless else "stateful (CMS memory)"
    print(f"\n{'─' * 60}")
    print(f"  NL-Hecate Chat")
    print(f"  Mode: {mode_label}")
    print(f"  temp={temperature}, top_k={top_k}, max_tokens={max_tokens}")
    print(f"{'─' * 60}")
    print("  Commands: /quit  /clear  /mode  /stats")
    print(f"{'─' * 60}\n")

    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        # ── Slash commands ──
        if stripped.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if stripped.lower() == "/clear":
            history_tokens.clear()
            turn_count = 0
            if not stateless:
                # Reset CMS state
                conductor = nl_hecate.Conductor(
                    cfg.k,
                    list(cfg.chunk_sizes) if hasattr(cfg, "chunk_sizes") else [1] * cfg.k,
                )
                if gpu_model is None:
                    context = nl_hecate.ContextState(cfg.k, cfg.d_model)
            print("  [conversation cleared]\n")
            continue

        if stripped.lower() == "/mode":
            print(f"  Mode: {mode_label}")
            print(f"  History: {len(history_tokens)} tokens, {turn_count} turns")
            if not stateless:
                print(f"  CMS: memory persists across turns (constant prompt size)")
            else:
                print(f"  No memory: full history re-sent each turn")
            print()
            continue

        if stripped.lower() == "/stats":
            print(f"  Turns: {turn_count}")
            print(f"  History tokens: {len(history_tokens)}")
            if not stateless and gpu_model is not None and hasattr(gpu_model, "gate_biases"):
                biases = gpu_model.gate_biases()
                for i, (ba, bt, be) in enumerate(biases):
                    print(f"  Level {i}: b_alpha={ba:.2f} b_theta={bt:.2f} b_eta={be:.2f}")
            print()
            continue

        # ── Build prompt tokens ──
        user_turn = chatml_encode_turn(tokenizer, "user", stripped)
        assistant_start = chatml_encode_prompt(tokenizer, "assistant")

        if stateless:
            # Append user turn to history, then prompt for assistant
            history_tokens.extend(user_turn)
            prompt_tokens = history_tokens + assistant_start
            # Truncate to fit seq_len (keep most recent)
            if len(prompt_tokens) > seq_len:
                prompt_tokens = prompt_tokens[-seq_len:]
        else:
            # Stateful: only send this turn + assistant prompt
            # CMS memory carries prior conversation context
            prompt_tokens = user_turn + assistant_start

        # ── Generate ──
        t0 = time.perf_counter()
        if stateless:
            # Fresh conductor each turn (no persistent memory)
            output = generate(
                params, cfg, prompt_tokens, max_tokens, temperature,
                top_k=top_k, stop_token=IM_END, gpu_model=gpu_model,
            )
        else:
            # Use persistent conductor/context
            output = generate(
                params, cfg, prompt_tokens, max_tokens, temperature,
                top_k=top_k, stop_token=IM_END, gpu_model=gpu_model,
                conductor=conductor, context=context,
            )
        t1 = time.perf_counter()

        # Extract only the generated tokens (after the prompt)
        gen_tokens = output[len(prompt_tokens):]
        response_text = tokenizer.decode(gen_tokens).strip()

        # Update history for stateless mode
        if stateless:
            response_turn = chatml_encode_turn(tokenizer, "assistant", response_text)
            history_tokens.extend(response_turn)

        turn_count += 1
        gen_count = len(gen_tokens)
        tps = gen_count / (t1 - t0) if (t1 - t0) > 0 else 0
        prompt_size = len(prompt_tokens)

        print(f"\033[1;32mAssistant:\033[0m {response_text}")
        print(f"  \033[2m[{gen_count} tokens, {tps:.0f} tok/s, "
              f"prompt={prompt_size} tokens]\033[0m\n")


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
                        help="Interactive REPL mode (raw prompts, no ChatML)")
    parser.add_argument("--chat", action="store_true",
                        help="Interactive chat mode with ChatML conversation formatting")
    parser.add_argument("--stateless", action="store_true",
                        help="Disable CMS memory in chat mode; re-send full history each turn")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU-resident model for fast generation")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 = disabled, default 50)")
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
    if args.chat:
        run_chat(
            params, cfg, tokenizer, gpu_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stateless=args.stateless,
        )
        return

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
                              top_k=args.top_k, use_cms=args.use_cms,
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
                          args.max_tokens, args.temperature,
                          top_k=args.top_k, use_cms=args.use_cms,
                          gpu_model=gpu_model)
        t1 = time.perf_counter()

        text = tokenizer.decode(output)
        gen_tokens = len(output) - len(prompt_tokens)
        tps = gen_tokens / (t1 - t0) if (t1 - t0) > 0 else 0

        print(f"\nOutput: {text}")
        print(f"\n  [{gen_tokens} tokens in {t1-t0:.3f}s, {tps:.0f} tok/s]")

    else:
        print("\nError: provide --prompt, --interactive, or --chat")
        sys.exit(1)


if __name__ == "__main__":
    main()
