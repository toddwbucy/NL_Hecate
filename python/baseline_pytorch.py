#!/usr/bin/env python3
"""
Phase 4: PyTorch regression baseline for NL-Hecate.

Implements the exact same SWA architecture as the Rust core using raw PyTorch
tensors (no nn.Module, no torch.optim). Trains both pipelines on identical data
with identical weights, comparing loss curves and throughput.

Usage:
    python baseline_pytorch.py              # comparison + timing
    python baseline_pytorch.py --compare    # comparison only
    python baseline_pytorch.py --timing     # timing only
"""

import sys
import time
import math
import argparse

import torch
import torch.nn.functional as F
import nl_hecate


# ── Config ────────────────────────────────────────────────────────────

SEQ_LEN = 32
D_MODEL = 64
NUM_HEADS = 4
HEAD_DIM = 16
WINDOW_SIZE = 16
VOCAB_SIZE = 256


def make_config():
    return nl_hecate.SWAConfig(D_MODEL, NUM_HEADS, HEAD_DIM, SEQ_LEN, WINDOW_SIZE, VOCAB_SIZE)


# ── Training data (identical to demo.py) ──────────────────────────────

TEXT = "the cat sat on the mat. " * 20
BYTES = [b for b in TEXT.encode("ascii")]


def make_chunks(data, seq_len):
    chunks = []
    for i in range(0, len(data) - seq_len, seq_len):
        inp = data[i : i + seq_len]
        tgt = data[i + 1 : i + seq_len + 1]
        if len(tgt) == seq_len:
            chunks.append((inp, tgt))
    return chunks


CHUNKS = make_chunks(BYTES, SEQ_LEN)


# ── PyTorch SWA model ─────────────────────────────────────────────────

def load_weights_from_rust(params):
    """Extract Rust weights and convert to PyTorch tensors."""
    w = params.get_weights()
    return {
        "w_embed": torch.tensor(w["w_embed"], dtype=torch.float32).reshape(VOCAB_SIZE, D_MODEL).requires_grad_(True),
        "w_q": torch.tensor(w["w_q"], dtype=torch.float32).reshape(D_MODEL, D_MODEL).requires_grad_(True),
        "w_k": torch.tensor(w["w_k"], dtype=torch.float32).reshape(D_MODEL, D_MODEL).requires_grad_(True),
        "w_v": torch.tensor(w["w_v"], dtype=torch.float32).reshape(D_MODEL, D_MODEL).requires_grad_(True),
        "w_o": torch.tensor(w["w_o"], dtype=torch.float32).reshape(D_MODEL, D_MODEL).requires_grad_(True),
        "w_unembed": torch.tensor(w["w_unembed"], dtype=torch.float32).reshape(D_MODEL, VOCAB_SIZE).requires_grad_(True),
    }


def build_swa_mask(seq_len, window_size):
    """Build causal sliding window attention mask.

    Position q attends to [max(0, q+1-window_size), q] inclusive.
    Returns [seq_len, seq_len] with 0.0 for valid and -inf for masked.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))
    for q in range(seq_len):
        win_start = max(0, q + 1 - window_size)
        mask[q, win_start : q + 1] = 0.0
    return mask


# Pre-build the mask once
SWA_MASK = build_swa_mask(SEQ_LEN, WINDOW_SIZE)


def forward_pytorch(weights, input_ids, target_ids):
    """Full forward pass matching the Rust 6-stage pipeline exactly.

    Returns loss scalar (with grad graph attached).
    """
    s = SEQ_LEN
    d = D_MODEL
    nh = NUM_HEADS
    hd = HEAD_DIM

    inp = torch.tensor(input_ids, dtype=torch.long)
    tgt = torch.tensor(target_ids, dtype=torch.long)

    # Stage 1: Embedding lookup — [s, d]
    embedded = weights["w_embed"][inp]

    # Stage 2: QKV projections — X @ W^T  (W is [d, d])
    q = embedded @ weights["w_q"].T  # [s, d]
    k = embedded @ weights["w_k"].T
    v = embedded @ weights["w_v"].T

    # Stage 3: Multi-head SWA attention
    # Reshape to [nh, s, hd]
    q_heads = q.reshape(s, nh, hd).permute(1, 0, 2)   # [nh, s, hd]
    k_heads = k.reshape(s, nh, hd).permute(1, 0, 2)
    v_heads = v.reshape(s, nh, hd).permute(1, 0, 2)

    # Scaled dot-product: [nh, s, s]
    scale = 1.0 / math.sqrt(hd)
    scores = torch.bmm(q_heads, k_heads.transpose(1, 2)) * scale

    # Apply SWA mask (broadcast over heads)
    scores = scores + SWA_MASK.unsqueeze(0)

    # Softmax over key dimension
    attn_weights = torch.softmax(scores, dim=-1)

    # Weighted sum: [nh, s, hd]
    attn_out_heads = torch.bmm(attn_weights, v_heads)

    # Reshape back to [s, d]
    attn_out = attn_out_heads.permute(1, 0, 2).reshape(s, d)

    # Stage 4: Output projection — attn_out @ W_O^T
    projected = attn_out @ weights["w_o"].T

    # Stage 5: Unembed — projected @ W_unembed (no transpose, [d, vocab])
    logits = projected @ weights["w_unembed"]  # [s, vocab]

    # Stage 6: Cross-entropy loss
    loss = F.cross_entropy(logits, tgt, reduction="mean")

    return loss


def sgd_step_pytorch(weights, lr):
    """Manual SGD: param -= lr * grad, then zero grads."""
    with torch.no_grad():
        for w in weights.values():
            if w.grad is not None:
                w -= lr * w.grad
                w.grad = None
    # Re-enable grads (in-place sub detaches)
    for key in weights:
        weights[key].requires_grad_(True)


# ── Comparison ────────────────────────────────────────────────────────

def run_comparison(num_steps=100, lr=0.05):
    """Train both pipelines in lockstep, comparing per-step loss."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Rust vs PyTorch, {num_steps} steps, lr={lr}")
    print(f"{'='*60}\n")

    cfg = make_config()

    # Initialize Rust params
    rust_params = nl_hecate.init_params(cfg, seed=42)

    # Load same weights into PyTorch
    pt_weights = load_weights_from_rust(rust_params)

    max_rel_err = 0.0
    all_finite = True
    rust_decreasing = True
    pt_decreasing = True
    prev_rust_loss = None
    prev_pt_loss = None
    first_rel_err = None

    print(f"  {'step':>4s}  {'rust_loss':>10s}  {'pt_loss':>10s}  {'rel_err':>10s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}")

    for step in range(num_steps):
        chunk_idx = step % len(CHUNKS)
        inp, tgt = CHUNKS[chunk_idx]

        # Rust forward + backward + SGD
        rust_loss, rust_grads = nl_hecate.compute_gradients(rust_params, cfg, inp, tgt)
        nl_hecate.sgd_step(rust_params, rust_grads, lr)

        # PyTorch forward + backward + SGD
        pt_loss_tensor = forward_pytorch(pt_weights, inp, tgt)
        pt_loss = pt_loss_tensor.item()
        pt_loss_tensor.backward()
        sgd_step_pytorch(pt_weights, lr)

        # Check finite
        if not math.isfinite(rust_loss) or not math.isfinite(pt_loss):
            all_finite = False

        # Relative error
        denom = max(abs(rust_loss), abs(pt_loss), 1e-12)
        rel_err = abs(rust_loss - pt_loss) / denom

        if step == 0:
            first_rel_err = rel_err
        max_rel_err = max(max_rel_err, rel_err)

        # Monotonicity check (first 10 steps can be noisy with cycling data)
        if step > 10:
            if prev_rust_loss is not None and rust_loss > prev_rust_loss * 1.1:
                rust_decreasing = False
            if prev_pt_loss is not None and pt_loss > prev_pt_loss * 1.1:
                pt_decreasing = False
        prev_rust_loss = rust_loss
        prev_pt_loss = pt_loss

        if step < 5 or step % 10 == 0 or step == num_steps - 1:
            print(f"  {step:4d}  {rust_loss:10.6f}  {pt_loss:10.6f}  {rel_err:10.2e}")

    print(f"\n  Results:")
    print(f"    Initial loss rel error (step 0): {first_rel_err:.2e}")
    print(f"    Max relative error:              {max_rel_err:.2e}")
    print(f"    All losses finite:               {all_finite}")

    # Pass/fail
    passed = True
    if first_rel_err > 1e-6:
        print(f"    FAIL: Initial loss rel error {first_rel_err:.2e} > 1e-6")
        passed = False
    if max_rel_err > 1e-4:
        print(f"    FAIL: Max rel error {max_rel_err:.2e} > 1e-4")
        passed = False
    if not all_finite:
        print(f"    FAIL: Non-finite losses detected")
        passed = False

    if passed:
        print(f"\n    PASS: Rust and PyTorch produce identical loss curves")
    else:
        print(f"\n    FAIL: Loss curves diverged beyond tolerance")

    return passed, max_rel_err


# ── Timing ────────────────────────────────────────────────────────────

def run_timing(num_steps=500, lr=0.05):
    """Measure tok/s for each pipeline independently."""
    print(f"\n{'='*60}")
    print(f"  TIMING: {num_steps} steps each, seq_len={SEQ_LEN}")
    print(f"{'='*60}\n")

    cfg = make_config()
    total_tokens = num_steps * SEQ_LEN

    # --- Rust timing ---
    rust_params = nl_hecate.init_params(cfg, seed=42)
    t0 = time.perf_counter()
    for step in range(num_steps):
        inp, tgt = CHUNKS[step % len(CHUNKS)]
        _loss, grads = nl_hecate.compute_gradients(rust_params, cfg, inp, tgt)
        nl_hecate.sgd_step(rust_params, grads, lr)
    rust_elapsed = time.perf_counter() - t0
    rust_tps = total_tokens / rust_elapsed

    # --- PyTorch timing ---
    pt_params = nl_hecate.init_params(cfg, seed=42)
    pt_weights = load_weights_from_rust(pt_params)
    t0 = time.perf_counter()
    for step in range(num_steps):
        inp, tgt = CHUNKS[step % len(CHUNKS)]
        loss = forward_pytorch(pt_weights, inp, tgt)
        loss.backward()
        sgd_step_pytorch(pt_weights, lr)
    pt_elapsed = time.perf_counter() - t0
    pt_tps = total_tokens / pt_elapsed

    print(f"  Rust pipeline:")
    print(f"    {rust_elapsed:.3f}s, {rust_tps:,.0f} tok/s")
    print(f"  PyTorch pipeline:")
    print(f"    {pt_elapsed:.3f}s, {pt_tps:,.0f} tok/s")
    print(f"  Ratio: Rust is {rust_tps / pt_tps:.1f}x {'faster' if rust_tps > pt_tps else 'slower'}")

    return rust_tps, pt_tps


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: PyTorch regression baseline")
    parser.add_argument("--compare", action="store_true", help="Run comparison only")
    parser.add_argument("--timing", action="store_true", help="Run timing only")
    args = parser.parse_args()

    if not args.compare and not args.timing:
        # Default: run both
        args.compare = True
        args.timing = True

    exit_code = 0
    if args.compare:
        passed, _ = run_comparison(num_steps=100)
        if not passed:
            exit_code = 1
    if args.timing:
        run_timing(num_steps=500)

    sys.exit(exit_code)
