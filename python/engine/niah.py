"""Needle-in-a-Haystack (NIAH) evaluation for CMS memory retrieval.

Simple 4K-token retrieval test: plant a synthetic fact early in a haystack
of real corpus text, query it at the end, and check whether the model
assigns higher log-probability to the correct answer than to random
alternatives.

At 60M parameters, passing NIAH at 4K is strong evidence the CMS memory
hierarchy enables retrieval beyond the attention window.

Spec: specs/infrastructure/12_metric_driven_promotion.md §6
"""

import math
import random

import nl_hecate


# ── Synthetic needles ────────────────────────────────────────────
# Each needle is (statement, query, answer).  Answers are short tokens
# that are unambiguous and unlikely to appear in natural corpus text.

NEEDLES = [
    (
        "The secret code for project alpha is 7492.",
        "What is the secret code for project alpha?",
        "7492",
    ),
    (
        "The identification number for laboratory zeta is 3841.",
        "What is the identification number for laboratory zeta?",
        "3841",
    ),
    (
        "The access key for server omega is 6205.",
        "What is the access key for server omega?",
        "6205",
    ),
    (
        "The reference number for experiment delta is 1738.",
        "What is the reference number for experiment delta?",
        "1738",
    ),
    (
        "The serial number for device sigma is 9054.",
        "What is the serial number for device sigma?",
        "9054",
    ),
]


def run_niah(gpu_model, cfg, haystack_tokens: list[int],
             tokenizer, num_trials: int = 5,
             haystack_len: int = 4096,
             seed: int = 42) -> dict:
    """Run NIAH evaluation over multiple needles.

    Args:
        gpu_model:       GPU model with .forward() method.
        cfg:             MAGConfig (provides k, chunk_sizes, seq_len, vocab_size).
        haystack_tokens: Flat list of token IDs from real corpus (must be
                         longer than haystack_len + needle/query overhead).
        tokenizer:       Tokenizer with encode/decode methods.
        num_trials:      Number of needle trials to run (max len(NEEDLES)).
        haystack_len:    Target haystack length in tokens (default 4096).
        seed:            RNG seed for haystack segment selection.

    Returns:
        dict with keys: trials (list of per-trial results), pass_rate,
        mean_lift, haystack_len.
    """
    rng = random.Random(seed)
    trials = min(num_trials, len(NEEDLES))
    seq_len = cfg.seq_len

    # We need enough corpus to carve distinct haystack segments
    min_corpus = haystack_len * (trials + 1)
    if len(haystack_tokens) < min_corpus:
        return {
            "error": f"Corpus too short: need {min_corpus} tokens, have {len(haystack_tokens)}",
            "trials": [],
            "pass_rate": 0.0,
            "mean_lift": 0.0,
            "haystack_len": haystack_len,
        }

    # Pick non-overlapping haystack start positions
    max_start = len(haystack_tokens) - haystack_len
    starts = []
    for _ in range(trials):
        for _attempt in range(100):
            s = rng.randint(0, max_start)
            # Ensure no overlap with existing segments
            if all(abs(s - prev) >= haystack_len for prev in starts):
                starts.append(s)
                break
        else:
            # Fallback: sequential non-overlapping
            starts.append(len(starts) * haystack_len)

    results = []
    for i in range(trials):
        needle_stmt, needle_query, needle_answer = NEEDLES[i]
        result = _run_single_trial(
            gpu_model, cfg, tokenizer,
            haystack_tokens[starts[i]:starts[i] + haystack_len],
            needle_stmt, needle_query, needle_answer,
            seq_len,
        )
        result["trial"] = i
        result["haystack_start"] = starts[i]
        results.append(result)

    passing = [r for r in results if r["pass"]]
    mean_lift = (sum(r["lift"] for r in results) / len(results)) if results else 0.0

    return {
        "trials": results,
        "pass_rate": len(passing) / len(results) if results else 0.0,
        "mean_lift": mean_lift,
        "haystack_len": haystack_len,
        "num_trials": len(results),
        "num_pass": len(passing),
    }


def _run_single_trial(gpu_model, cfg, tokenizer,
                       haystack_segment: list[int],
                       needle_stmt: str, needle_query: str,
                       needle_answer: str,
                       seq_len: int) -> dict:
    """Run a single NIAH trial: plant needle, forward, score retrieval.

    The sequence is: [needle_tokens] + [haystack_tokens] + [query_tokens]
    We process this in seq_len chunks (the model learns as it goes via
    the CMS memory), then score the answer at the final position.
    """
    vocab = cfg.vocab_size

    # Encode needle and query
    needle_tokens = tokenizer.encode(needle_stmt)
    query_tokens = tokenizer.encode(needle_query)
    answer_tokens = tokenizer.encode(needle_answer)

    # Build full sequence: needle at the start, query at the end
    # Trim haystack so total fits reasonably
    haystack_budget = len(haystack_segment) - len(needle_tokens) - len(query_tokens)
    if haystack_budget < 100:
        return {"pass": False, "lift": 0.0, "error": "haystack too short after needle+query"}

    full_seq = needle_tokens + haystack_segment[:haystack_budget] + query_tokens
    needle_to_query_distance = haystack_budget

    # Process sequence through the model in chunks.
    # Save and restore context so NIAH doesn't corrupt training state.
    saved_ctx = gpu_model.to_host_context()
    gpu_model.reset_context()

    conductor = nl_hecate.Conductor(
        cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    try:
        # Forward through all chunks except the last (which contains the query)
        last_logits = None
        pos = 0
        while pos + seq_len <= len(full_seq):
            chunk_input = full_seq[pos:pos + seq_len]
            # Target: shifted by 1
            if pos + seq_len + 1 <= len(full_seq):
                chunk_target = full_seq[pos + 1:pos + seq_len + 1]
            else:
                # Last chunk: target is next tokens where available, pad rest
                remaining = full_seq[pos + 1:]
                chunk_target = list(remaining) + [vocab] * (seq_len - len(remaining))

            pulse = conductor.pulse()
            loss, logits_flat = gpu_model.forward(chunk_input, chunk_target, pulse)
            conductor.advance()

            # Keep last-position logits from the final chunk
            last_logits = logits_flat[(seq_len - 1) * vocab: seq_len * vocab]
            pos += seq_len

        # Handle remaining tokens (partial final chunk)
        remainder = len(full_seq) - pos
        if remainder > 0 and remainder < seq_len:
            # Pad to seq_len
            chunk_input = full_seq[pos:] + [full_seq[-1]] * (seq_len - remainder)
            chunk_target = [vocab] * seq_len  # all masked
            if pos + 1 < len(full_seq):
                real_targets = full_seq[pos + 1:]
                for j, t in enumerate(real_targets):
                    chunk_target[j] = t

            pulse = conductor.pulse()
            loss, logits_flat = gpu_model.forward(chunk_input, chunk_target, pulse)
            conductor.advance()

            # Logits at the position just before we'd predict the answer
            answer_pos = remainder - 1
            last_logits = logits_flat[answer_pos * vocab: (answer_pos + 1) * vocab]

        if last_logits is None:
            return {"pass": False, "lift": 0.0, "error": "no logits produced"}

        # Score: log-probability of answer tokens at the final position
        answer_logprob = _logprob_of_tokens(last_logits, answer_tokens, vocab)

        # Baseline: average log-probability of 10 random number strings
        rng = random.Random(42)
        baseline_logprobs = []
        for _ in range(10):
            random_num = str(rng.randint(1000, 9999))
            random_tokens = tokenizer.encode(random_num)
            bp = _logprob_of_tokens(last_logits, random_tokens, vocab)
            baseline_logprobs.append(bp)
        baseline_logprob = sum(baseline_logprobs) / len(baseline_logprobs)

        lift = answer_logprob - baseline_logprob

        return {
            "pass": lift > 0,
            "lift": lift,
            "answer_logprob": answer_logprob,
            "baseline_logprob": baseline_logprob,
            "needle_to_query_tokens": needle_to_query_distance,
            "answer": needle_answer,
            "answer_tokens": answer_tokens,
        }

    finally:
        # Restore training context
        gpu_model.upload_context(saved_ctx)


def _logprob_of_tokens(logits: list[float], token_ids: list[int],
                        vocab: int) -> float:
    """Compute log-probability of the first token in token_ids from logits.

    Uses log-softmax over the vocabulary. Only scores the first token
    since that's what the last-position logits predict.
    """
    if not token_ids:
        return 0.0

    target = token_ids[0]
    if target >= vocab or target < 0:
        return float("-inf")

    # Log-softmax: log(exp(x_i) / sum(exp(x_j)))
    max_logit = max(logits[:vocab])
    log_sum_exp = max_logit + math.log(
        sum(math.exp(logits[j] - max_logit) for j in range(vocab))
    )
    return logits[target] - log_sum_exp


def print_niah_results(results: dict, step: int) -> None:
    """Pretty-print NIAH results to console."""
    if "error" in results:
        print(f"  [niah] step={step}  ERROR: {results['error']}")
        return

    print(f"  [niah] step={step}  haystack={results['haystack_len']}"
          f"  pass={results['num_pass']}/{results['num_trials']}"
          f"  rate={results['pass_rate']:.0%}"
          f"  mean_lift={results['mean_lift']:.3f}")

    for t in results["trials"]:
        status = "PASS" if t["pass"] else "FAIL"
        print(f"    trial {t['trial']}: {status}"
              f"  lift={t['lift']:.3f}"
              f"  answer={t.get('answer', '?')}"
              f"  distance={t.get('needle_to_query_tokens', 0)} tokens")
