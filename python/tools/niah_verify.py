"""Standalone NIAH verification tool — CMS memory retrieval probe.

Loads a checkpoint, plants synthetic factoids at random positions within
real-corpus haystacks, and measures whether the model retrieves them at
configurable distances. Designed to run offline between push-up phases.

This is NOT part of the training loop — it is an offline evaluation tool.

Usage:
    python -m tools.niah_verify \
        --checkpoint model.safetensors \
        --data data/fineweb_edu \
        --distances 1024,2048,4096 \
        --num_trials 5

Spec: specs/infrastructure/26_niah_verification.md
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Ensure the python/ directory is on sys.path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import nl_hecate
from engine.data import BpeTokenStream
from engine.generation import generate_cached
from engine.tokenizer import load_tokenizer


# ── Synthetic needles ────────────────────────────────────────────────
# Each needle is (statement, query, answer).

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

# ── Coherence probes ───────────────────────────────────────────────
# Simple text-completion prompts for qualitative generation assessment.
# At 60M params we expect limited fluency but recognizably English output
# with local coherence. These run alongside NIAH to sanity-check that
# the model hasn't collapsed while showing positive retrieval signal.

COHERENCE_PROMPTS = [
    "The cat sat on the",
    "Once upon a time there was a",
    "Scientists recently discovered that",
]


def _run_coherence(gpu_model, cfg, tokenizer, max_tokens: int = 50) -> list[dict]:
    """Generate text from each coherence prompt for qualitative review."""
    results = []
    for prompt_text in COHERENCE_PROMPTS:
        gpu_model.reset_context()
        prompt_tokens = tokenizer.encode(prompt_text)

        generated = generate_cached(
            gpu_model, cfg, prompt_tokens,
            max_tokens=max_tokens, temperature=0.8, top_k=40)

        output_text = tokenizer.decode(generated)
        results.append({
            "prompt": prompt_text,
            "generation": output_text,
            "num_tokens": len(generated) - len(prompt_tokens),
        })
    return results


def _logprob_at_position(logits_flat: list[float], position: int,
                         token_id: int, vocab: int) -> float:
    """Extract log-probability of token_id at a specific position.

    Delegates to Rust binding (nl_hecate.logprob_at_position) for
    numerically-stable log-sum-exp. Python only validates inputs.
    """
    if token_id < 0:
        return float("-inf")
    return nl_hecate.logprob_at_position(logits_flat, position, token_id, vocab)


def _build_trial_sequence(corpus_tokens: list[int],
                          needle_tokens: list[int],
                          query_tokens: list[int],
                          retrieval_distance: int,
                          haystack_size: int,
                          min_prefix: int,
                          min_suffix: int,
                          rng: random.Random,
                          corpus_offset: int) -> dict | None:
    """Build a trial sequence with randomized needle depth.

    Returns dict with 'sequence', 'needle_depth', 'query_answer_pos',
    or None if corpus is too short.
    """
    needle_len = len(needle_tokens)
    query_len = len(query_tokens)
    overhead = needle_len + query_len

    # Budget for corpus around the needle and query
    prefix_budget = haystack_size - retrieval_distance - overhead - min_suffix
    if prefix_budget < min_prefix:
        return None

    # Random needle depth within the prefix budget
    needle_depth = rng.randint(min_prefix, prefix_budget)

    # Total sequence length
    suffix_len = min_suffix
    total_len = needle_depth + needle_len + retrieval_distance + query_len + suffix_len

    # Check corpus has enough tokens from the offset
    if corpus_offset + needle_depth + retrieval_distance + suffix_len > len(corpus_tokens):
        return None

    # Build the sequence
    prefix = corpus_tokens[corpus_offset:corpus_offset + needle_depth]
    gap_start = corpus_offset + needle_depth
    gap = corpus_tokens[gap_start:gap_start + retrieval_distance]
    suffix_start = gap_start + retrieval_distance
    suffix = corpus_tokens[suffix_start:suffix_start + suffix_len]

    sequence = prefix + needle_tokens + gap + query_tokens + suffix

    # Position in the sequence where the answer should be predicted
    # (last token of the query — model predicts next token here)
    query_answer_pos = needle_depth + needle_len + retrieval_distance + query_len - 1

    return {
        "sequence": sequence,
        "needle_depth": needle_depth,
        "query_answer_pos": query_answer_pos,
        "total_len": len(sequence),
    }


def _run_trial(gpu_model, cfg, sequence: list[int],
               query_answer_pos: int,
               answer_token_id: int,
               baseline_rng: random.Random,
               tokenizer) -> dict:
    """Run a single NIAH trial: forward through sequence, score retrieval."""
    seq_len = cfg.seq_len
    vocab = cfg.vocab_size

    # Reset CMS context for a clean trial
    gpu_model.reset_context()

    conductor = nl_hecate.Conductor(
        cfg.k, list(cfg.chunk_sizes) if hasattr(cfg, 'chunk_sizes') else [1] * cfg.k)

    # Forward through the full sequence in seq_len chunks.
    # Keep logits for the chunk containing query_answer_pos.
    answer_logits_flat = None
    answer_chunk_start = None
    pos = 0

    while pos < len(sequence):
        chunk_end = min(pos + seq_len, len(sequence))
        chunk_len = chunk_end - pos

        if chunk_len < seq_len:
            # Pad final chunk
            chunk_input = sequence[pos:chunk_end] + [sequence[-1]] * (seq_len - chunk_len)
            chunk_target = [vocab] * seq_len
            # Fill real targets where available
            if pos + 1 < len(sequence):
                real_targets = sequence[pos + 1:chunk_end]
                for j, t in enumerate(real_targets):
                    chunk_target[j] = t
        else:
            chunk_input = sequence[pos:chunk_end]
            if chunk_end < len(sequence):
                chunk_target = sequence[pos + 1:chunk_end + 1]
            else:
                chunk_target = sequence[pos + 1:chunk_end] + [vocab]

        pulse = conductor.pulse()
        loss, logits_flat = gpu_model.forward(chunk_input, chunk_target, pulse)
        conductor.advance()

        # Keep logits if this chunk contains the answer position
        if pos <= query_answer_pos < pos + seq_len:
            answer_logits_flat = logits_flat
            answer_chunk_start = pos

        pos = chunk_end

    if answer_logits_flat is None:
        return {"pass": False, "lift": 0.0,
                "error": f"answer_pos {query_answer_pos} not covered by any chunk "
                         f"(sequence len {len(sequence)}, seq_len {seq_len})"}

    # Find the answer position within the answer chunk's logits
    pos_in_chunk = query_answer_pos - answer_chunk_start

    # Score the correct answer
    answer_logprob = _logprob_at_position(
        answer_logits_flat, pos_in_chunk, answer_token_id, vocab)

    if math.isnan(answer_logprob):
        return {"pass": False, "lift": 0.0,
                "error": "answer logprob is NaN (bad logits at this position)"}

    # Score 10 random baselines
    baseline_logprobs = []
    for _ in range(10):
        random_num = str(baseline_rng.randint(1000, 9999))
        random_tokens = tokenizer.encode(random_num)
        if random_tokens:
            bp = _logprob_at_position(
                answer_logits_flat, pos_in_chunk, random_tokens[0], vocab)
            if not math.isnan(bp):
                baseline_logprobs.append(bp)

    if not baseline_logprobs:
        return {"pass": False, "lift": 0.0, "error": "no valid baselines scored"}

    baseline_logprob = sum(baseline_logprobs) / len(baseline_logprobs)
    lift = answer_logprob - baseline_logprob

    return {
        "pass": lift > 0,
        "lift": lift,
        "answer_logprob": answer_logprob,
        "baseline_logprob": baseline_logprob,
    }


def run_niah_verify(checkpoint_path: str, data_path: str,
                    distances: list[int], num_trials: int = 5,
                    haystack_size: int = 8192, min_prefix: int = 512,
                    min_suffix: int = 256, seed: int = 42,
                    gpu_device: int = 0,
                    coherence: bool = True) -> dict:
    """Run full NIAH verification against a checkpoint.

    Returns results dict with per-distance pass rates and trial details.
    """
    # Load checkpoint — detect stacked vs single-block
    is_stacked = nl_hecate.is_stacked_checkpoint(checkpoint_path)

    if is_stacked:
        result = nl_hecate.load_stacked_checkpoint(checkpoint_path)
        cfg = result["config"]
        n_blocks = result["n_blocks"]
        params_json = result["params_json"]
    else:
        try:
            params, cfg, build_state = nl_hecate.load_build_checkpoint(checkpoint_path)
        except Exception:
            params, cfg = nl_hecate.load_checkpoint(checkpoint_path)

    print(f"NIAH Verification: {checkpoint_path}")
    print(f"  Model: d={cfg.d_model}, k={cfg.k}, "
          f"chunks={list(cfg.chunk_sizes)}")
    if is_stacked:
        print(f"  Stacked: {n_blocks} blocks")
    else:
        print(f"  Params: {params.num_params():,}")

    # Load corpus
    loader = BpeTokenStream(data_path, split="train")
    corpus_tokens = loader.tokens.tolist()
    print(f"  Data:  {data_path} ({len(corpus_tokens):,} tokens)")

    # Load tokenizer
    tokenizer = load_tokenizer(data_dir=data_path)

    # Upload to GPU
    if hasattr(nl_hecate, "set_cuda_device"):
        nl_hecate.set_cuda_device(gpu_device)
    if is_stacked:
        gpu_model = nl_hecate.GpuStackedModel.from_params_json(
            params_json, cfg, n_blocks)
        print(f"  GPU:   device {gpu_device} ({gpu_model.total_params():,} params)")
    else:
        gpu_model = nl_hecate.GpuModel.from_params(params, cfg)
        print(f"  GPU:   device {gpu_device}")

    print(f"  Distances: {distances}")
    print(f"  Trials: {num_trials} per distance\n")

    rng = random.Random(seed)
    baseline_rng = random.Random(seed + 1000)

    all_results = {}

    for distance in distances:
        distance_results = []

        # Auto-scale haystack: need at least distance + min_prefix + overhead + min_suffix
        # Use max(user haystack, distance * 2) so there's room for needle placement
        effective_haystack = max(haystack_size, distance + min_prefix + min_suffix + 512)

        for trial_idx in range(num_trials):
            needle_idx = trial_idx % len(NEEDLES)
            needle_stmt, needle_query, needle_answer = NEEDLES[needle_idx]

            needle_tokens = tokenizer.encode(needle_stmt)
            query_tokens = tokenizer.encode(needle_query)
            answer_tokens = tokenizer.encode(needle_answer)

            if not answer_tokens:
                distance_results.append({
                    "needle_idx": needle_idx,
                    "pass": False, "lift": 0.0,
                    "error": "answer tokenizes to empty"
                })
                continue

            # Pick a random corpus offset for this trial
            max_offset = len(corpus_tokens) - effective_haystack - 1000
            if max_offset < 0:
                distance_results.append({
                    "needle_idx": needle_idx,
                    "pass": False, "lift": 0.0,
                    "error": "corpus too short"
                })
                continue
            corpus_offset = rng.randint(0, max_offset)

            trial_seq = _build_trial_sequence(
                corpus_tokens, needle_tokens, query_tokens,
                distance, effective_haystack, min_prefix, min_suffix,
                rng, corpus_offset)

            if trial_seq is None:
                distance_results.append({
                    "needle_idx": needle_idx,
                    "pass": False, "lift": 0.0,
                    "error": "could not build trial sequence"
                })
                continue

            result = _run_trial(
                gpu_model, cfg, trial_seq["sequence"],
                trial_seq["query_answer_pos"],
                answer_tokens[0],
                baseline_rng, tokenizer)

            result["needle_idx"] = needle_idx
            result["needle_depth"] = trial_seq["needle_depth"]
            result["retrieval_distance"] = distance
            result["answer"] = needle_answer
            distance_results.append(result)

        # Aggregate (skip NaN lifts)
        passing = [r for r in distance_results if r.get("pass", False)]
        lifts = [r["lift"] for r in distance_results
                 if "lift" in r and not math.isnan(r["lift"])]
        pass_rate = len(passing) / len(distance_results) if distance_results else 0.0
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0

        all_results[str(distance)] = {
            "pass_rate": pass_rate,
            "mean_lift": mean_lift,
            "num_pass": len(passing),
            "num_trials": len(distance_results),
            "trials": distance_results,
        }

        # Console output
        print(f"  distance={distance}  pass={len(passing)}/{len(distance_results)}"
              f"  mean_lift={mean_lift:.3f}")
        for r in distance_results:
            status = "PASS" if r.get("pass", False) else "FAIL"
            depth = r.get("needle_depth", "?")
            lift = r.get("lift", 0.0)
            answer = r.get("answer", "?")
            error = r.get("error", "")
            if error:
                print(f"    trial: {status}  error={error}")
            else:
                print(f"    trial: {status}  depth={depth}  lift={lift:.3f}"
                      f"  answer={answer}")
        print()

    # Summary
    print("  Summary:")
    for d in distances:
        dr = all_results[str(d)]
        print(f"    {d}: {dr['pass_rate']:.0%} pass, "
              f"mean_lift={dr['mean_lift']:.3f}")

    # Coherence generation (skip for stacked models — no prefill/reset_cache yet)
    coherence_results = []
    if coherence and hasattr(gpu_model, 'prefill'):
        print("\n  Coherence probes:")
        coherence_results = _run_coherence(gpu_model, cfg, tokenizer)
        for cr in coherence_results:
            print(f"    prompt: \"{cr['prompt']}\"")
            print(f"    output: \"{cr['generation']}\"")
            print()
    elif coherence:
        print("\n  Coherence probes: skipped (stacked model — no prefill support yet)")

    return {
        "checkpoint": checkpoint_path,
        "model": {
            "d_model": cfg.d_model,
            "k": cfg.k,
            "chunk_sizes": list(cfg.chunk_sizes),
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
        },
        "config": {
            "haystack_size": haystack_size,
            "min_prefix": min_prefix,
            "min_suffix": min_suffix,
            "seed": seed,
            "num_trials": num_trials,
        },
        "distances": all_results,
        "coherence": coherence_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NIAH Verification — standalone CMS retrieval probe")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to safetensors checkpoint")
    parser.add_argument("--data", required=True,
                        help="Path to BPE corpus directory")
    parser.add_argument("--distances", default="4096",
                        help="Comma-separated retrieval distances (default: 4096)")
    parser.add_argument("--num_trials", type=int, default=5,
                        help="Trials per distance (default: 5)")
    parser.add_argument("--haystack_size", type=int, default=8192,
                        help="Total haystack tokens (default: 8192)")
    parser.add_argument("--min_prefix", type=int, default=512,
                        help="Min corpus tokens before needle (default: 512)")
    parser.add_argument("--min_suffix", type=int, default=256,
                        help="Min corpus tokens after query (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--output", default=None,
                        help="JSON output file path (default: stdout)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--no-coherence", action="store_true",
                        help="Skip coherence generation probes")

    args = parser.parse_args()

    try:
        distances = [int(d.strip()) for d in args.distances.split(",") if d.strip()]
    except ValueError as e:
        parser.error(f"--distances must be comma-separated integers: {e}")
    if not distances:
        parser.error("--distances must contain at least one value")

    t_start = time.perf_counter()

    results = run_niah_verify(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        distances=distances,
        num_trials=args.num_trials,
        haystack_size=args.haystack_size,
        min_prefix=args.min_prefix,
        min_suffix=args.min_suffix,
        seed=args.seed,
        gpu_device=args.gpu,
        coherence=not args.no_coherence,
    )

    elapsed = time.perf_counter() - t_start
    results["elapsed_seconds"] = round(elapsed, 1)
    print(f"\n  Elapsed: {elapsed:.1f}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Results: {output_path}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
