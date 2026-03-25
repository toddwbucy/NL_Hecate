"""Autoregressive generation: standard, KV-cached, stacked, and learning modes."""

import math
import random

import nl_hecate

# ChatML special token IDs (must match tokenizer training in prepare_sharegpt.py)
IM_START = 0  # <|im_start|>
IM_END = 1    # <|im_end|>
PAD = 2       # <|pad|>


def _safe_pad_token(prompt_tokens: list[int], vocab_size: int) -> int:
    """Choose a padding token that avoids special-token memory instability (CS-50).

    Special tokens (id 0-2) cause Titans inner loop divergence when 29+
    identical tokens appear. Use the prompt's first regular token, or
    fallback to token 3 (first BPE token after specials). Clamped to
    vocab_size - 1 to avoid OOV crashes on tiny-vocab test models.
    """
    if prompt_tokens and prompt_tokens[0] >= 3:
        return prompt_tokens[0]
    return min(3, vocab_size - 1)


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


# ── KV-cached generation (GPU decode) ────────────────────────────

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

    safe_pad = _safe_pad_token(prompt_tokens, vocab)

    try:
        # Pad/truncate prompt to seq_len for prefill
        ctx = seq[-seq_len:]
        while len(ctx) < seq_len:
            ctx = [safe_pad, *ctx]

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


# ── Learning generation (CS-10: continuous outer-loop during inference) ──

def generate_learning(
    gpu_model,
    cfg,
    prompt_tokens: list[int],
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 0,
    stop_token: int | None = None,
    conductor=None,
    lr: float = 0.0006,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
) -> tuple[list[int], list[float], list[float]]:
    """
    Autoregressive generation with continuous outer-loop learning (CS-10).

    Two-path architecture: step_adamw() to learn, prefill()+decode_token() to speak.
    The model learns from the context via step_adamw, then generates output via
    the KV-cached decode path. No step_generate() hybrid — learn and speak are
    separate operations.

    Returns (all_tokens, losses, grad_norms).
    """
    seq = list(prompt_tokens)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len
    losses: list[float] = []
    grad_norms: list[float] = []

    if conductor is None:
        conductor = nl_hecate.Conductor(
            cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    # Phase 1: Learn from prompt via step_adamw (chunked if prompt > seq_len)
    if len(seq) > seq_len:
        for start in range(0, len(seq) - seq_len, seq_len):
            chunk = seq[start:start + seq_len + 1]
            if len(chunk) < seq_len + 1:
                break
            input_ids = chunk[:seq_len]
            target_ids = chunk[1:seq_len + 1]
            pulse = conductor.pulse()
            loss, gnorm = gpu_model.step_adamw(
                input_ids, target_ids, pulse, lr,
                beta1=beta1, beta2=beta2, eps=eps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
            )
            conductor.advance()
            if math.isnan(loss) or math.isinf(loss):
                return seq, losses, grad_norms
            losses.append(loss)
            grad_norms.append(gnorm)
            print(f"  [learn] prompt chunk: loss={loss:.4f} gnorm={gnorm:.4f}")

    safe_pad = _safe_pad_token(prompt_tokens, vocab)

    # Phase 2: Learn from current context, then generate via prefill+decode
    # Each iteration: step_adamw on context (learn), then prefill+decode (speak)
    for i in range(max_tokens):
        # Build context window from tail of sequence
        ctx = seq[-seq_len:] if len(seq) >= seq_len else seq[:]

        # Left-pad if shorter than seq_len
        pad_len = seq_len - len(ctx)
        if pad_len > 0:
            ctx = [safe_pad] * pad_len + ctx

        # Target: shifted by 1, last position masked (vocab_size = OOV, skipped by kernel)
        if len(seq) >= seq_len + 1:
            target_src = seq[-(seq_len - 1):]
            target_ids = [*target_src, vocab]
        else:
            n_real = min(len(seq), seq_len)
            if n_real > 1:
                shifted = list(seq[1:])
                n_masked_prefix = seq_len - n_real
                target_ids = [vocab] * n_masked_prefix + shifted[: seq_len - n_masked_prefix]
                while len(target_ids) < seq_len:
                    target_ids.append(vocab)
                target_ids[-1] = vocab
            else:
                target_ids = [vocab] * seq_len

        # LEARN: step_adamw on the context
        pulse = conductor.pulse()
        loss, gnorm = gpu_model.step_adamw(
            ctx, target_ids, pulse, lr,
            beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        conductor.advance()

        if math.isnan(loss) or math.isinf(loss):
            break

        losses.append(loss)
        grad_norms.append(gnorm)

        # SPEAK: prefill context + decode one token
        try:
            speak_pulse = conductor.pulse()
            last_logits = gpu_model.prefill(ctx, speak_pulse)
            conductor.advance()
        finally:
            gpu_model.reset_cache()

        next_tok = _sample_token(last_logits, vocab, temperature, top_k)

        if stop_token is not None and next_tok == stop_token:
            break

        seq.append(next_tok)

    return seq, losses, grad_norms


# ── Stacked model generation (GpuStackedModel.forward) ───────────

def generate_stacked(
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
    Autoregressive generation using GpuStackedModel with prefill/decode_token.

    Uses single-token decode path (spec 47): one prefill call processes the full
    prompt and populates per-block KV caches, then each subsequent token runs
    through the O(d^2) decode path instead of O(seq_len * d^2) full forward.

    The memory state M is already accumulated during prefill — decode_token reads
    it for gating without re-running the full sequence. CS-10 compliant: same
    forward math, just s=1.
    """
    seq = list(prompt_tokens)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len

    if conductor is None:
        conductor = nl_hecate.Conductor(
            cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    safe_pad = _safe_pad_token(prompt_tokens, vocab)

    if max_tokens <= 0:
        return seq

    try:
        # ── Prefill: process full prompt ──────────────────────────────
        ctx = list(prompt_tokens[-seq_len:])
        while len(ctx) < seq_len:
            ctx = [safe_pad, *ctx]

        pulse = conductor.pulse()
        last_logits = gpu_model.prefill(ctx, pulse)
        conductor.advance()

        # Sample first token from prefill logits
        next_tok = _sample_token(last_logits, vocab, temperature, top_k)
        if stop_token is not None and next_tok == stop_token:
            return seq
        seq.append(next_tok)

        # ── Decode: one token at a time ───────────────────────────────
        for _ in range(max_tokens - 1):
            pulse = conductor.pulse()
            logits = gpu_model.decode_token(next_tok, pulse)
            conductor.advance()

            next_tok = _sample_token(logits, vocab, temperature, top_k)
            if stop_token is not None and next_tok == stop_token:
                break
            seq.append(next_tok)
    finally:
        gpu_model.reset_cache()
        gpu_model.reset_context()

    return seq


# ── Unified generation entry point ───────────────────────────────

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
    learn: bool = False,
    learn_kwargs: dict | None = None,
) -> list[int]:
    """
    Autoregressive generation with optional top-k sampling and stop token.

    Routes to the appropriate backend: learning, KV-cached GPU, or CPU.
    """
    is_stacked = hasattr(gpu_model, 'n_blocks') if gpu_model is not None else False

    # Delegate to learning path for GPU models with --learn
    if gpu_model is not None and learn:
        kw = learn_kwargs or {}
        seq, _losses, _gnorms = generate_learning(
            gpu_model, cfg, prompt_tokens, max_tokens,
            temperature, top_k, stop_token, conductor, **kw,
        )
        return seq

    # Stacked models use prefill/decode_token path (spec 47)
    if gpu_model is not None and is_stacked:
        return generate_stacked(
            gpu_model, cfg, prompt_tokens, max_tokens,
            temperature, top_k, stop_token, conductor,
        )

    # Delegate to KV-cached path for single-block GPU models
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
        if context is None:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)
            if params is not None and getattr(cfg, 'projection_kind', 'static') == 'adaptive':
                context.seed_self_ref(params)

    safe_pad = _safe_pad_token(prompt_tokens, vocab)

    for _ in range(max_tokens):
        # Take last seq_len tokens as context window
        ctx = seq[-seq_len:]
        # Pad if shorter than seq_len
        while len(ctx) < seq_len:
            ctx = [safe_pad, *ctx]

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
