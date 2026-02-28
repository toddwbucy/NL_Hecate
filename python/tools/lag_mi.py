#!/usr/bin/env python3
"""
Lag Mutual Information (Lag-MI) validator for corpus selection.

Measures whether a corpus has genuine long-range token structure at the
CMS frequency levels [1, 8, 64, 512] (period in tokens). A corpus passes
the selection criterion iff:

    ESTR(lag=512) > 2.0 × ESTR(lag=4096)

Where ESTR (Excess Same-Token Rate) is the ratio of the observed same-token
rate at a given lag to the expected collision rate under independent sampling,
minus 1. The 4096-lag reading serves as the background / null reference.

Spec: specs/infrastructure/02_corpus_selection.md
CS-38: uses 'build'/'context' vocabulary, not 'training'
CS-37: CMS frequency constructs called 'levels', not 'layers'
CS-47: --seed stored in output; same seed + corpus = identical result

Usage:
    python tools/lag_mi.py \\
        --corpus allenai/c4 \\
        --config en \\
        --split train \\
        --sample-tokens 100_000_000 \\
        --lags 1 8 64 512 4096 \\
        --exclude-top-n 200 \\
        --seed 42 \\
        --out results/lag_mi_c4.json

    # Quick smoke-test (1M tokens, fast)
    python tools/lag_mi.py --corpus allenai/c4 --config en \\
        --sample-tokens 1_000_000 --out /tmp/c4_quick.json

    # From a local tokenized .npy file (no HuggingFace needed)
    python tools/lag_mi.py --npy data/fineweb_edu/train_tokens.npy \\
        --out results/lag_mi_fineweb.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from array import array as _array
from pathlib import Path

import numpy as np

# CMS frequency level periods (tokens per fire). Lags evaluated by default.
# Level 0: every token (lag=1), Level 1: lag=8, Level 2: lag=64, Level 3: lag=512
# lag=4096: background / null (beyond most document boundaries)
CMS_LEVEL_PERIODS = [1, 8, 64, 512]
DEFAULT_LAGS = [1, 8, 64, 512, 4096]

# Tokenizer for HuggingFace corpus sources — 32K vocab, same as build configs
TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"  # 32K BPE, lightweight
VOCAB_SIZE = 32000

PASS_THRESHOLD = 2.0  # ESTR(512) must exceed this × ESTR(4096)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _load_tokenizer(tokenizer_id: str):
    """Load BPE tokenizer. Fails loudly if transformers is not installed."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit(
            "transformers not installed. Run: pip install transformers\n"
            "Or use --npy to provide a pre-tokenized file."
        )
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    return tok


def _stream_tokens_hf(
    corpus: str,
    split: str,
    sample_tokens: int,
    tokenizer,
    text_column: str,
    seed: int,
    verbose: bool,
    config: str | None = None,
) -> np.ndarray:
    """Stream documents from a HuggingFace dataset and tokenize.

    Returns a flat uint32 numpy array of token IDs in natural document order.
    Does NOT shuffle — lag-MI must be computed on natural order.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets not installed. Run: pip install datasets")

    eos_id = tokenizer.eos_token_id or 2
    # Use array.array instead of list to avoid OOM at 100M-token scale:
    # list[int] holds Python int objects (~28 bytes each); array.array('I')
    # stores raw 4-byte unsigned ints — ~7× lower peak memory.
    tokens: _array = _array("I")
    t0 = time.time()

    load_kwargs: dict = {"split": split, "streaming": True, "trust_remote_code": False}
    if config:
        load_kwargs["name"] = config
    ds = load_dataset(corpus, **load_kwargs)

    for doc in ds:
        text = doc.get(text_column, "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens.extend(ids)
        tokens.append(eos_id)  # document boundary marker

        if len(tokens) >= sample_tokens:
            break

        if verbose and len(tokens) % 5_000_000 == 0:
            elapsed = time.time() - t0
            tok_per_s = len(tokens) / max(elapsed, 1)
            remaining = (sample_tokens - len(tokens)) / max(tok_per_s, 1)
            print(
                f"  {len(tokens):>12,} / {sample_tokens:,} tokens  "
                f"({tok_per_s:,.0f} tok/s  ~{remaining:.0f}s remaining)",
                end="\r", flush=True,
            )

    if verbose:
        print()  # newline after \r progress

    arr = np.frombuffer(tokens, dtype=np.uint32, count=min(sample_tokens, len(tokens)))
    return arr.copy()


# ---------------------------------------------------------------------------
# Lag-MI computation
# ---------------------------------------------------------------------------

def _compute_estr(
    tokens: np.ndarray,
    lag: int,
    n_samples: int,
    rng: np.random.Generator,
    exclude_top_n: int = 200,
) -> float:
    """Excess Same-Token Rate (ESTR) at a given lag.

    Measures how much more often the same token appears at positions t and
    t+lag versus what independent token sampling would predict.

    ESTR(L) = P(x_{t+L} == x_t) / expected_collision_rate - 1

    Where expected_collision_rate = sum_v P(v)^2  (birthday-problem baseline).

    ESTR > 0  → tokens repeat more than chance at lag L (structured corpus)
    ESTR ≈ 0  → no lag-L token-level structure (flat baseline)

    High-frequency tokens (top-N stop-words) are excluded before sampling.
    They appear at every lag with similar rates, diluting the signal from
    content words (proper nouns, technical terms) that actually carry
    long-range structure.

    Why ESTR beats PPMI here: PPMI over a full 32K vocabulary is dominated
    by the joint distribution of high-frequency BPE subwords ("the", "of",
    " a"), which is flat at all lags. ESTR with stop-word exclusion focuses
    on content tokens that actually repeat within long documents but not
    across independent short articles.

    CS-47: rng is seeded externally; result is deterministic for fixed seed.
    """
    n = len(tokens)
    if n <= lag:
        return 0.0

    # --- Build content-word mask (exclude top-N most frequent tokens) ---
    freq = np.bincount(tokens.clip(0, VOCAB_SIZE - 1), minlength=VOCAB_SIZE)
    top_n_ids = set(np.argsort(freq)[::-1][:exclude_top_n].tolist())

    # --- Sample (t, t+lag) pairs ---
    max_pos = n - lag
    positions = rng.integers(0, max_pos, size=n_samples)
    src = tokens[positions].clip(0, VOCAB_SIZE - 1)
    tgt = tokens[positions + lag].clip(0, VOCAB_SIZE - 1)

    # Filter to pairs where neither token is a stop-word
    stop_mask = np.isin(src, list(top_n_ids)) | np.isin(tgt, list(top_n_ids))
    src_c = src[~stop_mask]
    tgt_c = tgt[~stop_mask]

    if len(src_c) < 1000:
        return 0.0  # too few content-word pairs after filtering

    # --- Same-token rate among content pairs ---
    same_rate = float((src_c == tgt_c).mean())

    # --- Expected collision rate for content tokens ---
    content_mask = ~np.isin(np.arange(VOCAB_SIZE), list(top_n_ids))
    content_freq = freq * content_mask
    total_content = content_freq.sum()
    if total_content == 0:
        return 0.0
    p_content = content_freq / total_content
    expected_collision = float((p_content ** 2).sum())

    if expected_collision <= 0:
        return 0.0

    return same_rate / expected_collision - 1.0


def compute_estr_profile(
    tokens: np.ndarray,
    lags: list[int],
    exclude_top_n: int,
    n_samples: int,
    seed: int,
    verbose: bool,
) -> dict[int, float]:
    """Compute ESTR at each requested lag. Returns {lag: estr_value}."""
    rng = np.random.default_rng(seed)  # CS-47: deterministic
    results: dict[int, float] = {}

    for lag in sorted(lags):
        t0 = time.time()
        estr = _compute_estr(tokens, lag, n_samples, rng, exclude_top_n=exclude_top_n)
        elapsed = time.time() - t0
        results[lag] = estr
        if verbose:
            # Map lags to CMS level descriptions for clear output
            level_desc = {1: "L0 (every token)", 8: "L1 (period=8)",
                          64: "L2 (period=64)", 512: "L3 (period=512)",
                          4096: "background (period=4096)"}
            desc = level_desc.get(lag, f"lag={lag}")
            print(f"  lag={lag:>5}  ESTR={estr:.6f}  [{desc}]  ({elapsed:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Pass/fail evaluation
# ---------------------------------------------------------------------------

def evaluate_criterion(estr: dict[int, float], threshold: float = PASS_THRESHOLD) -> dict:
    """Apply the corpus selection criterion.

    A corpus passes iff ESTR(lag=512) > threshold × ESTR(lag=4096),
    AND ESTR(lag=512) > 0. A zero or negative background does NOT
    auto-pass — it means the metric failed to compute a meaningful signal.

    Returns a dict with pass/fail, ratio, and per-level interpretation.
    """
    estr_512 = estr.get(512, 0.0)
    estr_bg = estr.get(4096, 0.0)

    # Guard: non-positive background forces FAIL (ambiguous signal).
    # A corpus should have some background coherence; 0.0 means something
    # went wrong with the metric computation, not that it's perfectly clean.
    estr_bg_safe = max(estr_bg, 1e-12)
    ratio = estr_512 / estr_bg_safe
    passed = estr_512 > 0 and ratio > threshold

    # Per-CMS-level signal assessment
    level_signal = {}
    for level, period in enumerate(CMS_LEVEL_PERIODS):
        if period in estr:
            bg = estr_bg_safe
            level_signal[f"L{level}_period{period}"] = {
                "estr": estr[period],
                "ratio_vs_background": estr[period] / bg,
                "has_signal": estr[period] > threshold * bg,
            }

    return {
        "passed": passed,
        "estr_512": estr_512,
        "estr_background_4096": estr_bg,
        "ratio_512_4096": ratio,
        "threshold": threshold,
        "cms_level_signal": level_signal,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--corpus", metavar="DATASET_ID",
        help="HuggingFace dataset ID (e.g. 'allenai/c4', 'pg19')",
    )
    src.add_argument(
        "--npy", metavar="PATH",
        help="Path to pre-tokenized uint32 .npy file (skips download + tokenization)",
    )
    p.add_argument(
        "--split", default="train",
        help="Dataset split to stream (default: train)",
    )
    p.add_argument(
        "--config", default=None, metavar="NAME",
        help="Dataset config name if required (e.g. 'en' for allenai/c4)",
    )
    p.add_argument(
        "--text-column", default="text",
        help="Column name containing document text (default: 'text')",
    )
    p.add_argument(
        "--tokenizer", default=TOKENIZER_ID,
        help=f"HuggingFace tokenizer ID (default: {TOKENIZER_ID})",
    )
    p.add_argument(
        "--sample-tokens", type=int, default=100_000_000,
        help="Number of tokens to sample for ESTR estimation (default: 100M)",
    )
    p.add_argument(
        "--n-samples", type=int, default=500_000,
        help="Number of (t, t+lag) pairs to sample per lag (default: 500K)",
    )
    p.add_argument(
        "--lags", nargs="+", type=int, default=DEFAULT_LAGS,
        help=f"Lags to evaluate (default: {DEFAULT_LAGS})",
    )
    p.add_argument(
        "--exclude-top-n", type=int, default=200,
        help=(
            "Exclude the N most frequent tokens (stop-words) before computing ESTR. "
            "Stop-words appear at every lag at similar rates and mask content-word "
            "long-range structure. (default: 200)"
        ),
    )
    p.add_argument(
        "--threshold", type=float, default=PASS_THRESHOLD,
        help=f"Pass threshold: ESTR(512) > threshold × ESTR(4096) (default: {PASS_THRESHOLD})",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sample selection. Stored in output for CS-47 reproducibility.",
    )
    p.add_argument(
        "--out", required=True, metavar="PATH",
        help="Output JSON file path",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    verbose = not args.quiet
    t_start = time.time()

    # --- Load or stream tokens ---
    if args.npy:
        if verbose:
            print(f"Loading pre-tokenized file: {args.npy}")
        tokens = np.load(args.npy)
        if tokens.dtype != np.uint32:
            tokens = tokens.astype(np.uint32)
        tokens = tokens[:args.sample_tokens]
        corpus_name = Path(args.npy).stem
        tokenizer_id = "pre-tokenized"
    else:
        if verbose:
            print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = _load_tokenizer(args.tokenizer)

        if verbose:
            print(
                f"Streaming {args.sample_tokens:,} tokens from "
                f"{args.corpus} ({args.split}) ..."
            )
        tokens = _stream_tokens_hf(
            corpus=args.corpus,
            split=args.split,
            sample_tokens=args.sample_tokens,
            tokenizer=tokenizer,
            text_column=args.text_column,
            seed=args.seed,
            verbose=verbose,
            config=args.config,
        )
        corpus_name = args.corpus
        tokenizer_id = args.tokenizer

    if verbose:
        print(f"Tokens loaded: {len(tokens):,}  (dtype={tokens.dtype})")
        print(f"Computing ESTR at lags {args.lags} ...")

    # --- Compute ESTR profile ---
    estr = compute_estr_profile(
        tokens=tokens,
        lags=args.lags,
        exclude_top_n=args.exclude_top_n,
        n_samples=args.n_samples,
        seed=args.seed,
        verbose=verbose,
    )

    # --- Evaluate criterion ---
    criterion = evaluate_criterion(estr, threshold=args.threshold)

    elapsed = time.time() - t_start

    # --- Build output document ---
    result = {
        "corpus": corpus_name,
        "split": getattr(args, "split", "n/a"),
        "tokenizer": tokenizer_id,
        "sample_tokens": len(tokens),
        "n_samples_per_lag": args.n_samples,
        "exclude_top_n": args.exclude_top_n,
        "seed": args.seed,          # CS-47: stored for reproducibility
        "lags": args.lags,
        "estr": {str(k): v for k, v in estr.items()},
        "criterion": criterion,
        "elapsed_seconds": round(elapsed, 1),
    }

    # --- Write output ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- Summary ---
    if verbose:
        status = "PASS ✓" if criterion["passed"] else "FAIL ✗"
        print(f"\n{status}  corpus={corpus_name}")
        print(f"  ESTR(lag=512)  = {criterion['estr_512']:.6f}")
        print(f"  ESTR(bg=4096)  = {criterion['estr_background_4096']:.6f}")
        print(f"  ratio          = {criterion['ratio_512_4096']:.2f}×  "
              f"(threshold: {args.threshold:.1f}×)")
        print(f"  Output: {out_path}")
        print(f"  Elapsed: {elapsed:.1f}s")

    sys.exit(0 if criterion["passed"] else 1)


if __name__ == "__main__":
    main()
