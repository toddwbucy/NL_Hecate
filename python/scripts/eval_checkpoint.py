#!/usr/bin/env python3
"""Standalone checkpoint evaluator — runs on a separate GPU without stopping training.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/eval_checkpoint.py \
        runs/k4_chain_dolmino_d768_12h/checkpoints/model_step32000.safetensors \
        --data data/dolmino_100b \
        --max-tokens 50

Loads a stacked checkpoint at batch_size=1 and runs step_generate (spec 52):
forward + backward + optimizer + logit extraction in a single call. There is
no separate "inference" mode — batch_size=1 IS inference in an NLM.

Evaluations:
  1. Held-out loss via step_generate (proper backward pass, M gets gradient feedback)
  2. Coherence generation via step_generate (model learns from prompt, generates next tokens)
  3. Per-head M norm snapshot (after step_generate has properly populated M)
  4. Gate biases
"""

import argparse
import math
import random
import sys
from pathlib import Path

import nl_hecate
from engine.tokenizer import load_tokenizer


EVAL_PROMPTS = [
    "The process of",
    "In mathematics,",
    "Scientists discovered that",
    "The history of",
]

# Default optimizer settings — matched to training configs
DEFAULT_LR = 0.0003
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_MAX_GRAD_NORM = 1.0


def load_model(ckpt_path: str):
    """Load a stacked checkpoint at batch_size=1."""
    result = nl_hecate.load_stacked_checkpoint(ckpt_path)
    cfg = result["config"]
    params_json = result["params_json"]
    n_blocks = result["n_blocks"]
    build_state = result["build_state"]

    gpu_model = nl_hecate.GpuStackedModel.from_params_json(
        params_json, cfg, n_blocks,
        batch_size=1, memory_reset=True)

    step = build_state["global_step"] if build_state else 0
    return gpu_model, cfg, n_blocks, step


def eval_loss(gpu_model, cfg, data_path: str, n_chunks: int = 20,
              offset_pct: float = 0.9, lr: float = DEFAULT_LR):
    """Run step_generate on held-out data — the exact same path as training.

    Each chunk runs forward + backward + optimizer update at batch_size=1.
    """
    from engine.data import BpeTokenStream
    loader = BpeTokenStream(data_path, split="train")
    total = loader.total_tokens

    seek_pos = int(total * offset_pct)
    loader._position = seek_pos

    seq_len = cfg.seq_len
    losses = []
    grad_norms = []
    conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

    for i in range(n_chunks):
        chunk = loader.next_chunk(seq_len)
        if chunk is None:
            break
        input_ids, target_ids = chunk
        pulse = conductor.pulse()
        try:
            loss, g_norm, _logits = gpu_model.step_generate(
                list(input_ids), list(target_ids), pulse, lr,
                beta1=DEFAULT_BETA1, beta2=DEFAULT_BETA2, eps=DEFAULT_EPS,
                weight_decay=DEFAULT_WEIGHT_DECAY,
                max_grad_norm=DEFAULT_MAX_GRAD_NORM,
                freeze_embed=False,
            )
            losses.append(loss)
            grad_norms.append(g_norm)
        except Exception as e:
            print(f"  chunk {i} failed: {e}")
            break
        conductor.advance()

    return losses, grad_norms


def sample_token(logits: list[float], vocab: int, temperature: float = 0.7):
    """Sample a single token from logits with temperature scaling."""
    if temperature <= 0:
        return max(range(vocab), key=lambda i: logits[i])
    max_logit = max(logits[:vocab])
    weights = [math.exp((l - max_logit) / temperature) for l in logits[:vocab]]
    total = sum(weights)
    r = random.random() * total
    cumsum = 0.0
    for j, w in enumerate(weights):
        cumsum += w
        if r < cumsum:
            return j
    return len(weights) - 1


def generate(gpu_model, cfg, prompt_ids: list[int],
             max_tokens: int = 50, lr: float = DEFAULT_LR,
             temperature: float = 0.7):
    """Generate tokens via step_generate — same path as training at batch_size=1.

    Each token: build context window, call step_generate (forward + backward +
    optimizer update + logit extraction), sample next token. The model learns
    from its own output as it generates. No mode flags, no forward-only shortcut.
    """
    seq = list(prompt_ids)
    vocab = cfg.vocab_size
    seq_len = cfg.seq_len
    conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

    # Safe pad token (avoid special tokens 0-2)
    safe_pad = prompt_ids[0] if prompt_ids and prompt_ids[0] >= 3 else min(3, vocab - 1)

    for i in range(max_tokens):
        # Build context window from tail of sequence
        ctx = seq[-seq_len:] if len(seq) >= seq_len else list(seq)
        pad_len = seq_len - len(ctx)
        if pad_len > 0:
            ctx = [safe_pad] * pad_len + ctx

        # Target: shifted by 1, last position masked with vocab (OOV, skipped by kernel)
        if len(seq) >= seq_len + 1:
            target_src = seq[-(seq_len - 1):]
            target_ids = list(target_src) + [vocab]
        elif len(seq) > 1:
            n_real = min(len(seq), seq_len)
            shifted = list(seq[1:])
            n_masked_prefix = seq_len - n_real
            target_ids = [vocab] * n_masked_prefix + shifted[:seq_len - n_masked_prefix]
            while len(target_ids) < seq_len:
                target_ids.append(vocab)
            target_ids[-1] = vocab
        else:
            target_ids = [vocab] * seq_len

        pulse = conductor.pulse()

        # Single call: forward + backward + optimizer + logits (spec 52)
        loss, g_norm, last_logits = gpu_model.step_generate(
            ctx, target_ids, pulse, lr,
            beta1=DEFAULT_BETA1, beta2=DEFAULT_BETA2, eps=DEFAULT_EPS,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            max_grad_norm=DEFAULT_MAX_GRAD_NORM,
            freeze_embed=False,
        )
        conductor.advance()

        if math.isnan(loss) or math.isinf(loss):
            print(f"  NaN/inf at token {i}, stopping generation")
            break

        next_tok = sample_token(last_logits, vocab, temperature)
        seq.append(next_tok)

    return seq


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on GPU")
    parser.add_argument("checkpoint", help="Path to .safetensors checkpoint")
    parser.add_argument("--data", default="data/dolmino_100b",
                        help="Data directory for loss evaluation")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens to generate per coherence sample")
    parser.add_argument("--n-chunks", type=int, default=20,
                        help="Number of step_generate chunks for loss eval")
    parser.add_argument("--held-out-pct", type=float, default=0.9,
                        help="Start loss eval at this fraction of corpus (default: last 10%%)")
    parser.add_argument("--n-recall", type=int, default=3,
                        help="Number of training-data chunks for recall test")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Learning rate for step_generate during eval")
    args = parser.parse_args()

    ckpt = args.checkpoint
    print(f"{'=' * 60}")
    print(f"Checkpoint Evaluation (step_generate, batch_size=1)")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Mode: step_generate (same path as training, no train/eval distinction)")

    # Load model
    gpu_model, cfg, n_blocks, step = load_model(ckpt)
    print(f"  Step: {step}")
    print(f"  Architecture: {n_blocks} blocks, k={cfg.k}, d={cfg.d_model}, "
          f"nh={cfg.num_heads}, hd={cfg.head_dim}")
    print(f"  Params: {gpu_model.total_params():,}")
    print(f"  LR: {args.lr}")
    print()

    # 1. Loss on held-out data via step_generate
    print(f"── Loss via step_generate (last {(1-args.held_out_pct)*100:.0f}%%, "
          f"{args.n_chunks} chunks) ──")
    try:
        losses, gnorms = eval_loss(
            gpu_model, cfg, args.data,
            n_chunks=args.n_chunks,
            offset_pct=args.held_out_pct,
            lr=args.lr)
        if losses:
            avg_loss = sum(losses) / len(losses)
            avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            avg_gnorm = sum(gnorms) / len(gnorms)
            print(f"  Chunks: {len(losses)}")
            print(f"  Avg loss: {avg_loss:.4f}  (ppl={avg_ppl:.1f})")
            print(f"  Min loss: {min(losses):.4f}  Max loss: {max(losses):.4f}")
            print(f"  Avg grad norm: {avg_gnorm:.4f}")
            first5 = losses[:5]
            last5 = losses[-5:]
            print(f"  First 5 losses: {[round(l, 4) for l in first5]}")
            print(f"  Last 5 losses:  {[round(l, 4) for l in last5]}")
        else:
            print("  No chunks evaluated")
    except Exception as e:
        print(f"  Loss eval failed: {e}")
    print()

    # 2. Per-head M norms (after step_generate has properly populated M)
    print("── Memory State (post-step_generate) ──")
    try:
        m_norms = list(gpu_model.memory_norms())
        fmt = lambda n: f"{n:.2e}" if 0 < n < 0.01 else f"{n:.4f}"
        print(f"  Aggregate M norms: [{', '.join(fmt(n) for n in m_norms)}]")

        ph_norms = gpu_model.memory_norms_per_head()
        for lev, heads in enumerate(ph_norms):
            if heads:
                vals = [v for v in heads if v > 0]
                if vals:
                    heads_str = " ".join(fmt(n) for n in heads)
                    spread = max(vals) / min(vals) if min(vals) > 0 else float('inf')
                    print(f"  L{lev} heads: [{heads_str}]  spread={spread:.1f}x")
                else:
                    print(f"  L{lev} heads: all zero")
            else:
                print(f"  L{lev} heads: empty")
    except Exception as e:
        print(f"  M norms failed: {e}")
    print()

    # 3. Gate biases
    print("── Gate Biases ──")
    try:
        biases = gpu_model.gate_biases()
        for lev, bias_tuple in enumerate(biases):
            b_alpha, b_theta = bias_tuple[0], bias_tuple[1]
            alpha_eff = 1.0 / (1.0 + math.exp(-b_alpha)) if abs(b_alpha) < 20 else (1.0 if b_alpha > 0 else 0.0)
            theta_eff = math.log1p(math.exp(b_theta)) if abs(b_theta) < 20 else (b_theta if b_theta > 0 else 0.0)
            extra = f"  b_eta={bias_tuple[2]:.4f}" if len(bias_tuple) > 2 else ""
            print(f"  L{lev}: b_alpha={b_alpha:.4f} (eff={alpha_eff:.4f})  "
                  f"b_theta={b_theta:.4f} (eff={theta_eff:.6f}){extra}")
    except Exception as e:
        print(f"  Gate biases failed: {e}")
    print()

    # 4. Recall test: feed chunks from EARLY training data (model saw these).
    # Split each chunk: first 480 tokens as context, last 32 as held-back.
    # Run step_generate on the full chunk to get loss (should be lower than
    # held-out if the model learned from this data). Then generate a continuation
    # from the 480-token prefix and compare against the actual next tokens.
    print(f"── Recall Test (training data, {args.n_recall} chunks) ──")
    tokenizer = load_tokenizer(data_dir=args.data)
    try:
        from engine.data import BpeTokenStream
        recall_loader = BpeTokenStream(args.data, split="train")
        # Seek to very early in corpus — data the model definitely saw
        recall_loader._position = 0

        seq_len = cfg.seq_len
        recall_losses = []
        recall_conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        n_gen = 32  # tokens to generate and compare
        n_ctx = seq_len - n_gen  # context tokens from the real data

        for i in range(args.n_recall):
            gpu_model.reset_context()
            chunk = recall_loader.next_chunk(seq_len)
            if chunk is None:
                break
            input_ids, target_ids = chunk

            # Step 1: Run step_generate on the FULL chunk (like training)
            pulse = recall_conductor.pulse()
            loss, g_norm, _logits = gpu_model.step_generate(
                list(input_ids), list(target_ids), pulse, args.lr,
                beta1=DEFAULT_BETA1, beta2=DEFAULT_BETA2, eps=DEFAULT_EPS,
                weight_decay=DEFAULT_WEIGHT_DECAY,
                max_grad_norm=DEFAULT_MAX_GRAD_NORM,
                freeze_embed=False,
            )
            recall_conductor.advance()
            recall_losses.append(loss)

            # Step 2: Decode the actual text for display
            actual_text = tokenizer.decode(list(target_ids[-n_gen:]))
            context_text = tokenizer.decode(list(input_ids[:80]))  # first 80 tokens for display

            # Step 3: Generate continuation from the first n_ctx tokens
            gpu_model.reset_context()
            ctx_ids = list(input_ids[:n_ctx])
            out_ids = generate(
                gpu_model, cfg, ctx_ids,
                max_tokens=n_gen, lr=args.lr, temperature=0.5)
            gen_text = tokenizer.decode(out_ids[n_ctx:])

            # Step 4: Compare — how many tokens match exactly?
            gen_tokens = out_ids[n_ctx:]
            actual_tokens = list(target_ids[n_ctx - 1:n_ctx - 1 + n_gen])
            n_match = sum(1 for g, a in zip(gen_tokens, actual_tokens) if g == a)

            ctx_preview = context_text[:100].replace("\n", "\\n")
            actual_preview = actual_text[:80].replace("\n", "\\n")
            gen_preview = gen_text[:80].replace("\n", "\\n")

            print(f"  Chunk {i}: loss={loss:.4f}  exact_match={n_match}/{min(len(gen_tokens), n_gen)}")
            print(f"    Context: \"{ctx_preview}...\"")
            print(f"    Actual:  \"{actual_preview}\"")
            print(f"    Model:   \"{gen_preview}\"")
            print()

        if recall_losses:
            avg_recall = sum(recall_losses) / len(recall_losses)
            avg_recall_ppl = math.exp(avg_recall) if avg_recall < 20 else float('inf')
            print(f"  Recall avg loss: {avg_recall:.4f} (ppl={avg_recall_ppl:.1f})")
            if losses:
                avg_heldout = sum(losses) / len(losses)
                delta = avg_recall - avg_heldout
                print(f"  Held-out avg loss: {avg_heldout:.4f}")
                print(f"  Delta (recall - held-out): {delta:+.4f}")
                if delta < -0.1:
                    print(f"  --> Model remembers training data (lower loss on seen data)")
                elif delta > 0.1:
                    print(f"  --> Surprising: higher loss on seen data")
                else:
                    print(f"  --> Similar loss — model may have generalized rather than memorized")
    except Exception as e:
        print(f"  Recall test failed: {e}")
    print()

    print(f"{'=' * 60}")
    print("Done")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
