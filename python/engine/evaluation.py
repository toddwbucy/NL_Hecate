"""Evaluation utilities: val loss, coherence samples, checkpoint-time generation."""

import math

import nl_hecate
from engine.data import BpeDataLoader


# Fixed prompts for sampling at checkpoints (tests different capabilities)
SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain how a neural network learns in simple terms.",
    "Write a short poem about the ocean.",
]

# Short prompts for eval-time coherence check (byte-level friendly)
EVAL_PROMPTS = [
    "Once upon a time",
    "The meaning of life is",
    "In the beginning",
]


def evaluate(gpu_model, bcfg, val_loader,
             max_chunks: int, val_doc_starts=None) -> tuple[float, float]:
    """Run forward-only on val set. Returns (avg_loss, perplexity).

    Uses a fresh Conductor + ContextState so eval doesn't corrupt
    training memory. Same forward path as training — no mode flag (CS-10).
    Document boundary resets apply if val_doc_starts is provided.
    """
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
    context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)

    total_loss = 0.0
    n_chunks = 0

    if isinstance(val_loader, BpeDataLoader):
        val_loader.position = 0
        for _ in range(max_chunks):
            chunk = val_loader.next_chunk(bcfg.seq_len)
            if chunk is None:
                break
            input_ids, target_ids = chunk
            pulse = conductor.pulse()

            if gpu_model is not None:
                loss, _ = gpu_model.forward(input_ids, target_ids, pulse)
            else:
                raise NotImplementedError("CPU eval not yet implemented for BPE")

            if not (math.isnan(loss) or math.isinf(loss)):
                total_loss += loss
                n_chunks += 1
            conductor.advance()
    else:
        # Byte-level eval: VecStream over val corpus
        val_stream = nl_hecate.VecStream.from_bytes(val_loader)
        val_conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
        val_conductor.attach_stream(val_stream)

        next_doc_idx = 1
        for chunk_i in range(max_chunks):
            result = val_conductor.next_chunk(bcfg.seq_len)
            if result is None:
                break
            input_ids, target_ids, pulse = result
            if len(input_ids) != bcfg.seq_len:
                val_conductor.advance()
                continue

            if gpu_model is not None:
                loss, _ = gpu_model.forward(input_ids, target_ids, pulse)
            else:
                raise NotImplementedError("CPU byte-level eval not yet wired")

            if not (math.isnan(loss) or math.isinf(loss)):
                total_loss += loss
                n_chunks += 1

            val_conductor.advance()

            # Document boundary reset (same as training)
            if val_doc_starts is not None:
                byte_pos = (chunk_i + 1) * bcfg.seq_len
                prev_idx = next_doc_idx
                while (next_doc_idx < len(val_doc_starts)
                       and byte_pos >= val_doc_starts[next_doc_idx]):
                    next_doc_idx += 1
                if next_doc_idx > prev_idx:
                    gpu_model.reset_context()

    if n_chunks == 0:
        return 0.0, 1.0

    avg_loss = total_loss / n_chunks
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def evaluate_numpy(gpu_model, bcfg, tokens_np, targets_np,
                   max_chunks: int = 10) -> tuple[float, float]:
    """Evaluate on raw numpy arrays (for per-phase curriculum probes).

    Creates a fresh Conductor so eval pulse doesn't corrupt training state.
    NOTE: Callers must save/restore gpu_model context externally — this
    function uses whatever context is currently on the model.
    Returns (avg_loss, perplexity).
    """
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)
    total_loss = 0.0
    n_chunks = 0
    pos = 0

    for _ in range(max_chunks):
        if pos + bcfg.seq_len > len(tokens_np):
            break
        input_ids = tokens_np[pos:pos + bcfg.seq_len].tolist()
        raw_targets = targets_np[pos:pos + bcfg.seq_len]
        target_ids = [int(t) if t >= 0 else bcfg.vocab_size for t in raw_targets]
        pulse = conductor.pulse()
        loss, _ = gpu_model.forward(input_ids, target_ids, pulse)
        if not (math.isnan(loss) or math.isinf(loss)):
            total_loss += loss
            n_chunks += 1
        conductor.advance()
        pos += bcfg.seq_len

    if n_chunks == 0:
        return 0.0, 1.0
    avg = total_loss / n_chunks
    return avg, math.exp(min(avg, 20.0))


def print_level_metrics(gpu_model, k):
    """Print per-level gate activations and memory norms at eval time."""
    biases = gpu_model.gate_biases()  # list of (b_alpha, b_theta, b_eta)
    mem_norms = gpu_model.memory_norms()  # list of floats

    for lev in range(min(k, len(biases))):
        ba, bt, be = biases[lev]
        # Activated values (what the model actually uses)
        alpha = 1.0 / (1.0 + math.exp(-ba))      # sigmoid -> forget gate
        if bt > 20.0:
            theta = bt
        elif bt < -20.0:
            theta = math.exp(bt)
        else:
            theta = math.log1p(math.exp(bt))      # softplus -> lr gate
        eta = 1.0 / (1.0 + math.exp(-be))         # sigmoid -> momentum gate
        mnorm = mem_norms[lev] if lev < len(mem_norms) else 0.0
        print(f"    L{lev}: \u03b1={alpha:.4f} \u03b8={theta:.6f} \u03b7={eta:.4f} \u2016M\u2016={mnorm:.4f}")


def eval_coherence_samples(gpu_model, cfg, max_tokens: int = 30,
                           tokenizer=None):
    """Generate short completions from fixed prompts to eyeball coherence.

    Uses greedy decoding (temperature=0) for deterministic output.
    If tokenizer is provided (e.g. BpeTokenizer), uses it; otherwise
    falls back to ByteTokenizer for byte-level models.
    """
    from engine.generation import generate
    from engine.tokenizer import ByteTokenizer
    tok = tokenizer if tokenizer is not None else ByteTokenizer()
    results = []
    for prompt in EVAL_PROMPTS:
        prompt_ids = tok.encode(prompt)
        out_ids = generate(
            params=None, cfg=cfg, prompt_tokens=prompt_ids,
            max_tokens=max_tokens, temperature=0.0,
            gpu_model=gpu_model,
        )
        gen_text = tok.decode(out_ids[len(prompt_ids):])
        results.append((prompt, gen_text))
    return results


def generate_samples(gpu_model, cfg, tokenizer, step: int,
                     temperature: float = 0.7,
                     max_tokens: int = 128) -> list[dict]:
    """Generate sample completions at checkpoint time.

    Uses autoregressive decoding. Returns list of dicts with prompt,
    completion, and token count for JSONL logging.
    """
    from engine.generation import generate

    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        prompt_tokens = tokenizer.encode(prompt_text)
        output_tokens = generate(
            params=None,
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
            "step": step,
        })
    return samples
