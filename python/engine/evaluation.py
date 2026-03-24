"""Evaluation utilities: val loss, learning probes, checkpoint-time generation."""

import math

import nl_hecate
from engine.data import BpeTokenStream


# Fixed prompts for sampling at checkpoints (tests different capabilities)
SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Explain how a neural network learns in simple terms.",
    "Write a short poem about the ocean.",
]

# Eval prompts matched to FineWeb-Edu domain (educational text)
EVAL_PROMPTS = [
    "The process of",
    "In mathematics,",
    "Scientists discovered that",
    "The history of",
]


# ── Full state snapshot/restore (CS-49 superset) ─────────────────

def full_snapshot(gpu_model):
    """Save complete model state for later restoration.

    Captures params (outer-loop), context (inner-loop M matrices),
    and optimizer moments (spec 33) so probes don't corrupt training state.
    """
    return {
        "params": gpu_model.to_host_params(),
        "context": gpu_model.to_host_context(),
        "optimizer": gpu_model.snapshot_optimizer(),
    }


def full_restore(gpu_model, snapshot):
    """Restore complete model state from snapshot."""
    gpu_model.upload_params(snapshot["params"])
    gpu_model.upload_context(snapshot["context"])
    opt = snapshot.get("optimizer")
    if opt is not None:
        gpu_model.restore_optimizer(opt)


# ── Learning probes (CS-10 compliant eval) ────────────────────────

def probe_within_generation(gpu_model, cfg, prompt_ids, tokenizer,
                            max_tokens=60, temperature=0.7, lr=0.0006,
                            conductor=None):
    """Probe 1: Does the model learn during generation?

    Runs generate_learning() and returns per-token loss trajectory.
    The model updates params on every generated token — if loss decreases
    over the span, the NL mechanism is working.
    """
    from engine.generation import generate_learning

    gpu_model.reset_context()
    tokens, losses, gnorms = generate_learning(
        gpu_model, cfg, list(prompt_ids),
        max_tokens=max_tokens, temperature=temperature,
        conductor=conductor, lr=lr,
    )

    gen_text = tokenizer.decode(tokens[len(prompt_ids):]) if tokenizer else ""

    # Compute summary stats (filter NaN before aggregation)
    valid_losses = [v for v in losses if math.isfinite(v)]
    n = len(valid_losses)
    if n >= 10:
        first10 = sum(valid_losses[:10]) / 10
        last10 = sum(valid_losses[-10:]) / 10
        # Simple linear regression for slope
        mean_x = (n - 1) / 2.0
        mean_y = sum(valid_losses) / n
        num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(valid_losses))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
    else:
        first10 = sum(valid_losses) / max(n, 1)
        last10 = first10
        slope = 0.0

    return {
        "token_losses": losses,
        "token_grad_norms": gnorms,
        "generated_text": gen_text,
        "loss_slope": slope,
        "loss_first10_avg": first10,
        "loss_last10_avg": last10,
        "n_tokens": n,
    }


def probe_cross_exposure(gpu_model, cfg, prompt_ids, tokenizer,
                         max_tokens=30, temperature=0.7, lr=0.0006,
                         conductor_factory=None):
    """Probe 2: Does the model adapt across repeated exposures?

    Runs generate_learning() twice on the same prompt WITHOUT restoring
    params between runs (only resets context). If run 2 produces lower
    loss, the outer-loop updates from run 1 transferred.

    This is the definitive NL test — no transformer can do this.
    """
    from engine.generation import generate_learning

    def make_conductor():
        if conductor_factory:
            return conductor_factory()
        return nl_hecate.Conductor(
            cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    # Run 1: cold start
    gpu_model.reset_context()
    tokens1, losses1, _ = generate_learning(
        gpu_model, cfg, list(prompt_ids),
        max_tokens=max_tokens, temperature=temperature,
        conductor=make_conductor(), lr=lr,
    )
    text1 = tokenizer.decode(tokens1[len(prompt_ids):]) if tokenizer else ""
    valid1 = [v for v in losses1 if not (math.isnan(v) or math.isinf(v))]
    avg1 = sum(valid1) / max(len(valid1), 1)

    # Run 2: reset context but KEEP updated params
    gpu_model.reset_context()
    tokens2, losses2, _ = generate_learning(
        gpu_model, cfg, list(prompt_ids),
        max_tokens=max_tokens, temperature=temperature,
        conductor=make_conductor(), lr=lr,
    )
    text2 = tokenizer.decode(tokens2[len(prompt_ids):]) if tokenizer else ""
    valid2 = [v for v in losses2 if not (math.isnan(v) or math.isinf(v))]
    avg2 = sum(valid2) / max(len(valid2), 1)

    improvement = avg1 - avg2
    improvement_pct = (improvement / avg1 * 100) if (math.isfinite(avg1) and avg1 > 0) else float("nan")

    return {
        "run1_avg_loss": avg1,
        "run2_avg_loss": avg2,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "run1_text": text1,
        "run2_text": text2,
    }


def probe_context_value(gpu_model, cfg, prompt_ids, snapshot,
                        max_tokens=30, temperature=0.7, lr=0.0006,
                        conductor_factory=None):
    """Probe 3: Does accumulated training context help generation?

    Compares generate_learning() with fresh M (cold) vs accumulated
    training M (warm). If warm produces lower loss, the memory built
    during training is contributing to generation quality.
    """
    from engine.generation import generate_learning

    def make_conductor():
        if conductor_factory:
            return conductor_factory()
        return nl_hecate.Conductor(
            cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    # Cold start: fresh M
    gpu_model.reset_context()
    _, cold_losses, _ = generate_learning(
        gpu_model, cfg, list(prompt_ids),
        max_tokens=max_tokens, temperature=temperature,
        conductor=make_conductor(), lr=lr,
    )
    valid_cold = [v for v in cold_losses if not (math.isnan(v) or math.isinf(v))]
    cold_avg = sum(valid_cold) / max(len(valid_cold), 1)

    # Restore full state (params + context + optimizer moments)
    full_restore(gpu_model, snapshot)

    # Warm start: accumulated training M
    _, warm_losses, _ = generate_learning(
        gpu_model, cfg, list(prompt_ids),
        max_tokens=max_tokens, temperature=temperature,
        conductor=make_conductor(), lr=lr,
    )
    valid_warm = [v for v in warm_losses if not (math.isnan(v) or math.isinf(v))]
    warm_avg = sum(valid_warm) / max(len(valid_warm), 1)

    return {
        "cold_avg_loss": cold_avg,
        "warm_avg_loss": warm_avg,
        "context_benefit": cold_avg - warm_avg,
    }


def evaluate(gpu_model, bcfg, val_stream,
             max_chunks: int, val_doc_starts=None) -> tuple[float, float]:
    """Run forward-only on val set. Returns (avg_loss, perplexity).

    Uses a fresh Conductor so eval doesn't corrupt training pulse state.
    Context is managed by the caller (gpu_model.reset_context() / upload_context()).
    Same forward path as training — no mode flag (CS-10).
    Document boundary resets apply if val_doc_starts is provided.
    """
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)

    total_loss = 0.0
    n_chunks = 0

    if isinstance(val_stream, BpeTokenStream):
        val_stream.position = 0
        for _ in range(max_chunks):
            chunk = val_stream.next_chunk(bcfg.seq_len)
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
        val_stream = nl_hecate.VecStream.from_bytes(val_stream)
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
    if gpu_model is None:
        raise ValueError("evaluate_numpy requires a non-None gpu_model")
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
        line = f"    L{lev}: \u03b1={alpha:.4f} \u03b8={theta:.6f} \u03b7={eta:.4f} \u2016M\u2016={mnorm:.4f}"
        # Per-head M norms (spec 50): show head differentiation
        if hasattr(gpu_model, "memory_norms_per_head"):
            ph_norms = gpu_model.memory_norms_per_head()
            if lev < len(ph_norms) and ph_norms[lev]:
                fmt = lambda n: f"{n:.2e}" if 0 < n < 0.01 else f"{n:.4f}"
                heads_str = " ".join(fmt(n) for n in ph_norms[lev])
                line += f"  heads=[{heads_str}]"
        print(line)


def print_tape_summary(tape_summary: dict, step: int) -> None:
    """Log per-level tape diagnostics from gpu_model.tape_forward_summary().

    Called at eval_every intervals alongside standard level metrics.
    Each level line shows block count, output grad norm, and DGD delta norm.

    Interpretation:
      block_count == 0               Level did not fire this step (CMS frequency gate)
      output_grad_norm == 0, blocks > 0  Gradient not flowing — initialization trap
      output_grad_norm > 0, blocks > 0   Level active and receiving gradient (healthy)
      dgd_delta_norm == 0, blocks > 0    Active but DGD delta collapsed (inner loop issue)
    """
    print(f"  [tape] step={step}  loss={tape_summary['loss']:.4f}"
          f"  total_blocks={tape_summary['total_blocks']}")
    for lvl in tape_summary["levels"]:
        line = (
            f"    L{lvl['level']} [{lvl['opaque_key']}]"
            f"  blocks={lvl['block_count']}"
            f"  out_gnorm={lvl['output_grad_norm']:.4e}"
            f"  dgd_delta={lvl['dgd_delta_norm']:.4e}"
        )
        # Append m_norm to level line if present
        if "m_norm" in lvl:
            mn = lvl["m_norm"]
            line += f"  m_norm={'NaN' if (isinstance(mn, float) and math.isnan(mn)) else f'{mn:.1f}'}"
        # Frozen flag
        if lvl.get("is_frozen", False):
            line += "  FROZEN"
        # Frequency gate (learned schedule only — NaN for Fixed)
        if "freq_gate_value" in lvl:
            fgv = lvl["freq_gate_value"]
            if isinstance(fgv, float) and not math.isnan(fgv):
                line += f"  freq_gate={fgv:.4f}"
        # Shard M-diff (spec 28): proxy-compatible level differentiation metric
        if "m_shard_diff" in lvl:
            line += f"  \u0394M={lvl['m_shard_diff']:.4f}"
        if "m_shard_diff_relative" in lvl:
            line += f"  \u0394M_rel={lvl['m_shard_diff_relative']:.4f}"
        if "dormancy_status" in lvl:
            line += f"  dormancy={lvl['dormancy_status']}"
        print(line)
        # Per-head M norms (spec 50): head differentiation signal
        if "head_m_norms" in lvl and lvl["head_m_norms"]:
            heads = lvl["head_m_norms"]
            fmt = lambda n: f"{n:.2e}" if 0 < n < 0.01 else f"{n:.4f}"
            heads_str = " ".join(fmt(n) for n in heads)
            print(f"           M_heads=[{heads_str}]")
        # Alpha (retention/forgetting gate) — before theta
        if "alpha" in lvl and lvl["alpha"] is not None:
            a = lvl["alpha"]
            floor_pct = a.get("frac_at_floor", 0.0) * 100
            p99_val = a.get("p99", a.get("p99_max", 0.0))
            print(
                f"           \u03b1  mean={a['mean']:.4f}  "
                f"p99={p99_val:.4f}  "
                f"max={a['max']:.4f}  "
                f"@floor={floor_pct:.1f}%"
            )
        # Theta (inner-loop learning rate)
        if "theta" in lvl and lvl["theta"] is not None:
            t = lvl["theta"]
            ceil_pct = t.get("frac_at_ceil", 0.0) * 100
            # Aggregated stacked view uses p99_max; single-block uses p99
            p99_val = t.get("p99", t.get("p99_max", 0.0))
            print(
                f"           \u03b8  mean={t['mean']:.4f}  "
                f"p99={p99_val:.4f}  "
                f"max={t['max']:.4f}  "
                f"@ceil={ceil_pct:.1f}%"
            )
        # Eta (momentum gate) — Titans only, no bound indicator
        if "eta" in lvl and lvl["eta"] is not None:
            e = lvl["eta"]
            p99_val = e.get("p99", e.get("p99_max", 0.0))
            print(
                f"           \u03b7  mean={e['mean']:.4f}  "
                f"p99={p99_val:.4f}  "
                f"max={e['max']:.4f}"
            )


def eval_coherence_samples(gpu_model, cfg, max_tokens: int = 30,
                           tokenizer=None):
    """Generate short completions from fixed prompts to eyeball coherence.

    Requires gpu_model (generate routes to KV-cached path with params=None).
    Uses temperature=0.7 sampling for varied output.
    If tokenizer is provided (e.g. BpeTokenizer), uses it; otherwise
    falls back to ByteTokenizer for byte-level models.
    """
    if gpu_model is None:
        raise ValueError("eval_coherence_samples requires a non-None gpu_model")
    from engine.generation import generate
    from engine.tokenizer import ByteTokenizer
    tok = tokenizer if tokenizer is not None else ByteTokenizer()
    results = []
    for prompt in EVAL_PROMPTS:
        prompt_ids = tok.encode(prompt)
        out_ids = generate(
            params=None, cfg=cfg, prompt_tokens=prompt_ids,
            max_tokens=max_tokens, temperature=0.7,
            gpu_model=gpu_model,
        )
        gen_text = tok.decode(out_ids[len(prompt_ids):])
        results.append((prompt, gen_text))
    return results


def probe_memory_vocab(host_params, host_context, cfg, tokenizer, step: int,
                       donor_params=None) -> dict:
    """Logit lens for CMS memory states: project M_l through W_unembed.

    For each CMS level l, computes mean(M_l @ W_unembed, axis=0) to get a
    single vocabulary distribution — asking 'what does this memory level know?'

    Args:
        host_params:   MAGParams downloaded from GPU (snapshot["params"]).
        host_context:  ContextState downloaded from GPU (snapshot["context"]).
        cfg:           MAGConfig (provides d_model, vocab_size, k).
        tokenizer:     BpeTokenizer (for token-id → string decoding).
        step:          Current training step (for logging).
        donor_params:  Optional donor MAGParams; if given, computes KL from
                       donor W_unembed projection for each level.

    Returns dict with keys:
        step, levels (per-level top-20 + M-norm + optional kl_from_donor),
        js_divergence (JS div between every level pair).
    """
    import math
    import numpy as np

    d = cfg.d_model
    v = cfg.vocab_size
    k = cfg.k

    # w_unembed layout: [d_model, vocab_size] row-major
    # (index: w_unembed[i * vocab + j], reshaped to [d, v])
    weights = host_params.get_weights()
    w_u = np.array(weights["w_unembed"], dtype=np.float32).reshape(d, v)

    donor_w_u = None
    if donor_params is not None:
        dw = donor_params.get_weights()
        donor_w_u = np.array(dw["w_unembed"], dtype=np.float32).reshape(d, v)

    def _softmax(x: "np.ndarray") -> "np.ndarray":
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-30)

    def _top20(probs: "np.ndarray") -> list:
        top_ids = np.argsort(probs)[-20:][::-1]
        out = []
        for tid in top_ids:
            tok_str = ""
            try:
                tok_str = tokenizer.decode([int(tid)])
            except (ValueError, TypeError):
                tok_str = f"<{int(tid)}>"
            out.append({"id": int(tid), "prob": round(float(probs[tid]), 6),
                        "tok": repr(tok_str)})
        return out

    def _kl(p: "np.ndarray", q: "np.ndarray") -> float:
        eps = 1e-10
        p = p + eps
        q = q + eps
        return float(np.sum(p * np.log(p / q)))

    level_probs: list = []
    levels_data: list = []

    memory = host_context.memory  # list[k] of flat Vec<f32> length d*d each
    for level_idx in range(k):
        M_l = np.array(memory[level_idx], dtype=np.float32).reshape(d, d)
        m_norm = float(np.linalg.norm(M_l, "fro"))

        # Project: [d, d] @ [d, v] → [d, v]; mean over rows → [v]
        logits_l = M_l @ w_u           # [d, v]
        mean_logits = logits_l.mean(axis=0)  # [v]
        probs_l = _softmax(mean_logits)

        level_probs.append(probs_l)

        entry: dict = {
            "level": level_idx,
            "m_norm": round(m_norm, 6),
            "top20": _top20(probs_l) if m_norm > 1e-6 else [],
        }

        if donor_w_u is not None:
            donor_logits = M_l @ donor_w_u
            donor_mean = donor_logits.mean(axis=0)
            donor_probs = _softmax(donor_mean)
            entry["kl_from_donor"] = round(_kl(probs_l, donor_probs), 6)

        levels_data.append(entry)

    # Jensen-Shannon divergence between every level pair
    js_pairs: list = []
    for i in range(k):
        for j in range(i + 1, k):
            eps = 1e-10
            p = level_probs[i] + eps
            q = level_probs[j] + eps
            m = 0.5 * (p + q)
            js = float(0.5 * np.sum(p * np.log(p / m))
                       + 0.5 * np.sum(q * np.log(q / m)))
            js_pairs.append({"levels": f"{i}-{j}", "js_div": round(js, 6)})

    return {
        "step": step,
        "levels": levels_data,
        "js_divergence": js_pairs,
    }


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
