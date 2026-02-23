#!/usr/bin/env python3
"""
Prepare curriculum-ordered data for NL-Hecate training.

Loads sources from local storage (/bulk-store/ and HF cache), formats each
as ChatML with user-turn masking, then blends into a curriculum-ordered
stream using cosine crossfade:

    Phase 1 (0–15K steps):   TinyStories (simple narratives)
    Fade   (15K–25K):        Stories → Conversation
    Phase 2 (25K–45K):       StackExchange + Dolly + tulu_flan (conversation)
    Fade   (45K–55K):        Conversation → Reasoning
    Phase 3 (55K–100K):      GSM8K + OpenMathInstruct (math reasoning)

All sources are local:
    - TinyStories:       HF cache (~1GB)
    - StackExchange:     /bulk-store/training-datasets/dolma/v1.7/stackexchange-*.json.gz
    - tulu_flan:         /bulk-store/training-datasets/dolma/v1.7/tulu_flan-*.json.gz
    - Dolly:             HF cache (~7MB)
    - GSM8K:             HF cache (~3MB)
    - OpenMathInstruct:  HF cache (~3.7GB)

Uses the same tokenizer as ShareGPT (data/sharegpt/tokenizer.json) for
clean A/B comparison.  Output format matches BpeDataLoader expectations.

Usage:
    python data/prepare_curriculum.py \\
        --tokenizer data/sharegpt/tokenizer.json \\
        --output data/curriculum
"""

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


# ── Special token IDs (must match prepare_sharegpt.py) ──────────────
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
IM_START_ID = 0
IM_END_ID = 1
PAD_ID = 2

# ── Local data paths ────────────────────────────────────────────────
DOLMA_DIR = Path("/bulk-store/training-datasets/dolma/v1.7")


# ── ChatML formatting (copied from prepare_sharegpt.py for independence) ─

def load_tokenizer(path: str):
    """Load a previously trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def format_chatml(conversation: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Format a conversation as ChatML and generate token/target arrays.

    Returns:
        tokens:  list[int] — token IDs for the full conversation
        targets: list[int] — next-token targets (-1 for masked user turns)
    """
    tokens = []
    is_assistant = []

    for turn in conversation:
        role = turn.get("from", "").lower()
        value = turn.get("value", "")
        if not value:
            continue

        if role in ("human", "user"):
            role_str = "user"
            is_train = False
        elif role in ("gpt", "assistant", "chatgpt"):
            role_str = "assistant"
            is_train = True
        else:
            continue

        header = f"{role_str}\n"
        header_ids = tokenizer.encode(header).ids
        content_ids = tokenizer.encode(value).ids
        newline_ids = tokenizer.encode("\n").ids

        tokens.append(IM_START_ID)
        is_assistant.append(False)

        tokens.extend(header_ids)
        is_assistant.extend([False] * len(header_ids))

        tokens.extend(content_ids)
        is_assistant.extend([is_train] * len(content_ids))

        tokens.append(IM_END_ID)
        is_assistant.append(is_train)
        tokens.extend(newline_ids)
        is_assistant.extend([False] * len(newline_ids))

    if len(tokens) < 2:
        return [], []

    input_tokens = tokens[:-1]
    target_tokens = []
    for i in range(1, len(tokens)):
        if is_assistant[i]:
            target_tokens.append(tokens[i])
        else:
            target_tokens.append(-1)

    return input_tokens, target_tokens


# ── Dolma JSONL.gz reader ────────────────────────────────────────────

def read_dolma_gz(pattern: str, max_docs: int = 0) -> list[dict]:
    """Read JSONL.gz files matching a glob pattern under DOLMA_DIR.

    Returns list of dicts with at least a 'text' field.
    """
    files = sorted(DOLMA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {DOLMA_DIR}/{pattern}")
    print(f"    Found {len(files)} files matching {pattern}")

    docs = []
    for fpath in files:
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
                if max_docs and len(docs) >= max_docs:
                    print(f"    Read {len(docs):,} docs (hit max_docs limit)")
                    return docs
    print(f"    Read {len(docs):,} docs total")
    return docs


# ── Source loaders / formatters ──────────────────────────────────────

def load_tinystories() -> list[dict]:
    """Load TinyStories from HF cache (no download needed)."""
    from datasets import load_dataset
    print("  Loading TinyStories (HF cache)...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    return list(ds)


def format_tinystories(dataset, tokenizer, min_tokens=20, max_tokens=400):
    """Wrap stories as user-prompt + assistant-story ChatML pairs."""
    examples = []
    skipped = 0
    for row in dataset:
        text = row.get("text", "").strip()
        if not text:
            skipped += 1
            continue
        conv = [
            {"from": "user", "value": "Tell me a story."},
            {"from": "assistant", "value": text},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < min_tokens:
            skipped += 1
            continue
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
            tgts = tgts[:max_tokens]
        examples.append((toks, tgts))
    print(f"    TinyStories: {len(examples)} examples, {skipped} skipped")
    return examples


def load_stackexchange(max_docs: int = 50_000) -> list[dict]:
    """Load StackExchange Q&A from Dolma (local)."""
    print("  Loading StackExchange (Dolma local)...")
    # One shard is plenty — each has ~100K+ docs
    return read_dolma_gz("stackexchange-0000.json.gz", max_docs=max_docs)


def format_stackexchange(dataset, tokenizer, min_tokens=30, max_tokens=512):
    """Parse Q:/A: format into user/assistant ChatML pairs."""
    examples = []
    skipped = 0
    for row in dataset:
        text = row.get("text", "")
        # StackExchange Dolma format: "Q: question\n\nA: answer"
        if "\n\nA: " not in text:
            skipped += 1
            continue
        parts = text.split("\n\nA: ", 1)
        question = parts[0]
        if question.startswith("Q: "):
            question = question[3:]
        question = question.strip()
        answer = parts[1].strip()

        if not question or not answer or len(answer) < 50:
            skipped += 1
            continue

        conv = [
            {"from": "user", "value": question},
            {"from": "assistant", "value": answer},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < min_tokens:
            skipped += 1
            continue
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
            tgts = tgts[:max_tokens]
        examples.append((toks, tgts))
    print(f"    StackExchange: {len(examples)} examples, {skipped} skipped")
    return examples


def load_tulu_flan(max_docs: int = 30_000) -> list[dict]:
    """Load tulu_flan instruction-following data from Dolma (local)."""
    print("  Loading tulu_flan (Dolma local)...")
    return read_dolma_gz("tulu_flan-0000.json.gz", max_docs=max_docs)


def format_tulu_flan(dataset, tokenizer, min_tokens=30, max_tokens=512):
    """Parse tulu_flan Q/A pairs. Various formats — extract first Q/A pair."""
    examples = []
    skipped = 0
    for row in dataset:
        text = row.get("text", "").strip()
        if not text:
            skipped += 1
            continue

        # Try to split into question/answer parts
        # Common patterns: "q: ... a: ...", "QUESTION: ... SOLUTION: ...",
        # "question in book: ... standard solution: ..."
        question = None
        answer = None

        # Pattern 1: lines starting with q:/a: or Q:/A:
        for sep_q, sep_a in [("\nq: ", "\na: "), ("\nQ: ", "\nA: "),
                              ("\nQUESTION: ", "\nSOLUTION: "),
                              ("\nQUESTION: ", "\nAnswer: ")]:
            if sep_a in text:
                # Find last Q/A pair (some docs have multiple)
                idx_a = text.rfind(sep_a)
                idx_q = text.rfind(sep_q, 0, idx_a)
                if idx_q >= 0:
                    question = text[idx_q + len(sep_q):idx_a].strip()
                    answer = text[idx_a + len(sep_a):].strip()
                    break

        # Pattern 2: starts with "q: " at beginning
        if question is None and text.lower().startswith("q: "):
            for sep_a in ["\na: ", "\nA: "]:
                if sep_a in text:
                    idx_a = text.find(sep_a)
                    question = text[2:idx_a].strip()  # skip "q: "
                    answer = text[idx_a + len(sep_a):].strip()
                    break

        if not question or not answer or len(answer) < 30:
            skipped += 1
            continue

        # Truncate very long answers (take first answer if multiple Q/A in text)
        # Cut at next question marker if present
        for marker in ["\nq: ", "\nQ: ", "\nQUESTION:", "\n[TEACHER]"]:
            cut = answer.find(marker)
            if cut > 0:
                answer = answer[:cut].strip()
                break

        conv = [
            {"from": "user", "value": question},
            {"from": "assistant", "value": answer},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < min_tokens:
            skipped += 1
            continue
        if len(toks) > max_tokens:
            toks = toks[:max_tokens]
            tgts = tgts[:max_tokens]
        examples.append((toks, tgts))
    print(f"    tulu_flan: {len(examples)} examples, {skipped} skipped")
    return examples


def load_dolly() -> list[dict]:
    """Load Databricks Dolly 15K from HF cache."""
    from datasets import load_dataset
    print("  Loading Dolly (HF cache)...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return list(ds)


def format_dolly(dataset, tokenizer):
    """instruction (+ context) → user, response → assistant."""
    examples = []
    skipped = 0
    for row in dataset:
        instruction = row.get("instruction", "").strip()
        context = row.get("context", "").strip()
        response = row.get("response", "").strip()
        if not instruction or not response:
            skipped += 1
            continue

        user_text = instruction
        if context:
            user_text = f"{instruction}\n\nContext: {context}"

        conv = [
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": response},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < 10:
            skipped += 1
            continue
        examples.append((toks, tgts))
    print(f"    Dolly: {len(examples)} examples, {skipped} skipped")
    return examples


def load_gsm8k() -> list[dict]:
    """Load GSM8K from HF cache."""
    from datasets import load_dataset
    print("  Loading GSM8K (HF cache)...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return list(ds)


def format_gsm8k(dataset, tokenizer):
    """question → user, chain-of-thought answer → assistant."""
    examples = []
    skipped = 0
    for row in dataset:
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            skipped += 1
            continue

        conv = [
            {"from": "user", "value": question},
            {"from": "assistant", "value": answer},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < 10:
            skipped += 1
            continue
        examples.append((toks, tgts))
    print(f"    GSM8K: {len(examples)} examples, {skipped} skipped")
    return examples


def load_openmath(sample_n: int = 100_000, seed: int = 42) -> list[dict]:
    """Load OpenMathInstruct-1 from HF cache (sample from 1.8M)."""
    from datasets import load_dataset
    print(f"  Loading OpenMathInstruct-1 (HF cache, sampling {sample_n:,})...")
    ds = load_dataset("nvidia/OpenMathInstruct-1", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(sample_n, len(ds))))
    return list(ds)


def format_openmath(dataset, tokenizer):
    """question → user, generated_solution → assistant."""
    examples = []
    skipped = 0
    for row in dataset:
        problem = row.get("question", "").strip()
        solution = row.get("generated_solution", "").strip()
        if not problem or not solution:
            skipped += 1
            continue

        conv = [
            {"from": "user", "value": problem},
            {"from": "assistant", "value": solution},
        ]
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) < 10:
            skipped += 1
            continue
        examples.append((toks, tgts))
    print(f"    OpenMath: {len(examples)} examples, {skipped} skipped")
    return examples


# ── Curriculum blending ──────────────────────────────────────────────

def cosine_fade(position: float, start: float, end: float) -> float:
    """Smooth 1→0 fade over [start, end] using cosine interpolation."""
    if position <= start:
        return 1.0
    if position >= end:
        return 0.0
    t = (position - start) / (end - start)
    return 0.5 * (1.0 + math.cos(math.pi * t))


def phase_weights(token_pos: int, total_tokens: int) -> tuple[float, float, float]:
    """Compute (stories_w, conversation_w, reasoning_w) at a given token position.

    Schedule (in steps, 512 tokens/step):
        0–15K:   100% stories
        15K–25K: fade stories→conversation
        25K–45K: 100% conversation
        45K–55K: fade conversation→reasoning
        55K–100K: 100% reasoning
    """
    frac = token_pos / max(total_tokens, 1)

    stories_fade_start = 15_000 / 100_000   # 0.15
    stories_fade_end = 25_000 / 100_000     # 0.25
    conv_fade_start = 45_000 / 100_000      # 0.45
    conv_fade_end = 55_000 / 100_000        # 0.55

    stories_w = cosine_fade(frac, stories_fade_start, stories_fade_end)
    reasoning_w = 1.0 - cosine_fade(frac, conv_fade_start, conv_fade_end)
    conv_w = max(0.0, 1.0 - stories_w - reasoning_w)

    total = stories_w + conv_w + reasoning_w
    if total > 0:
        stories_w /= total
        conv_w /= total
        reasoning_w /= total

    return stories_w, conv_w, reasoning_w


def blend_curriculum(
    stories_pool: list[tuple[list[int], list[int]]],
    conversation_pool: list[tuple[list[int], list[int]]],
    reasoning_pool: list[tuple[list[int], list[int]]],
    total_tokens: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Interleave examples from three source pools into curriculum-ordered arrays.

    Draws examples round-robin within each pool (wrapping at pool end).
    At each position, picks from pools proportional to phase weights.

    Returns (tokens: np.ndarray[uint32], targets: np.ndarray[int32])
    """
    rng = np.random.RandomState(seed)

    pools = [stories_pool, conversation_pool, reasoning_pool]
    pool_names = ["stories", "conversation", "reasoning"]
    cursors = [0, 0, 0]

    # Pre-shuffle each pool
    for i, pool in enumerate(pools):
        order = rng.permutation(len(pool))
        pools[i] = [pool[j] for j in order]

    # Preallocate with slack (total_tokens + 512 max overshoot)
    capacity = total_tokens + 512
    tokens_arr = np.empty(capacity, dtype=np.uint32)
    targets_arr = np.empty(capacity, dtype=np.int32)
    write_pos = 0
    emitted = 0
    pool_counts = [0, 0, 0]

    while emitted < total_tokens:
        sw, cw, rw = phase_weights(emitted, total_tokens)
        weights = [sw, cw, rw]

        r = rng.random()
        cumulative = 0.0
        chosen = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r < cumulative:
                chosen = i
                break

        pool = pools[chosen]
        if len(pool) == 0:
            for i in range(3):
                if len(pools[i]) > 0:
                    chosen = i
                    pool = pools[i]
                    break
            else:
                break

        idx = cursors[chosen] % len(pool)
        cursors[chosen] = idx + 1
        toks, tgts = pool[idx]

        if emitted + len(toks) > total_tokens + 512:
            found = False
            for attempt in range(10):
                idx2 = (idx + attempt + 1) % len(pool)
                t2, g2 = pool[idx2]
                if emitted + len(t2) <= total_tokens + 512:
                    toks, tgts = t2, g2
                    cursors[chosen] = idx2 + 1
                    found = True
                    break
            if not found:
                break

        n = len(toks)
        tokens_arr[write_pos:write_pos + n] = toks
        targets_arr[write_pos:write_pos + n] = tgts
        write_pos += n
        emitted += n
        pool_counts[chosen] += 1

    # Trim to actual length
    tokens_arr = tokens_arr[:write_pos]
    targets_arr = targets_arr[:write_pos]

    for name, count in zip(pool_names, pool_counts, strict=True):
        print(f"    {name}: {count} examples drawn")
    print(f"    Total: {write_pos:,} tokens")

    return tokens_arr, targets_arr


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare curriculum-ordered data for NL-Hecate")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to existing tokenizer.json")
    parser.add_argument("--output", type=str, default="data/curriculum",
                        help="Output directory (default: data/curriculum)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction held out for validation (default: 0.05)")
    parser.add_argument("--total_tokens", type=int, default=51_200_000,
                        help="Total training tokens (default: 51.2M = 100K steps * 512)")
    parser.add_argument("--openmath_sample", type=int, default=100_000,
                        help="OpenMathInstruct examples to sample (default: 100K)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load tokenizer ───────────────────────────────────────
    print("Step 1: Loading tokenizer...")
    if not Path(args.tokenizer).exists():
        print(f"ERROR: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size}")

    # ── Step 2: Load + format all sources (all local) ────────────────
    print("\nStep 2: Loading and formatting sources (all local)...")
    t0 = time.time()

    # Phase 1: Stories
    raw_stories = load_tinystories()

    # Phase 2: Conversation (StackExchange + tulu_flan + Dolly — all local)
    raw_stackex = load_stackexchange(max_docs=50_000)
    raw_tulu = load_tulu_flan(max_docs=30_000)
    raw_dolly = load_dolly()

    # Phase 3: Reasoning (GSM8K + OpenMath — both in HF cache)
    raw_gsm8k = load_gsm8k()
    raw_openmath = load_openmath(sample_n=args.openmath_sample, seed=args.seed)

    print(f"\n  All sources loaded in {time.time() - t0:.1f}s")
    print("  Formatting as ChatML...")
    t1 = time.time()

    stories = format_tinystories(raw_stories, tokenizer)
    stackex = format_stackexchange(raw_stackex, tokenizer)
    tulu = format_tulu_flan(raw_tulu, tokenizer)
    dolly = format_dolly(raw_dolly, tokenizer)
    gsm8k = format_gsm8k(raw_gsm8k, tokenizer)
    openmath = format_openmath(raw_openmath, tokenizer)

    print(f"  Formatting complete in {time.time() - t1:.1f}s")

    # ── Step 3: Split train/val per source ───────────────────────────
    print(f"\nStep 3: Splitting train/val (val_ratio={args.val_ratio})...")
    rng = np.random.RandomState(args.seed)

    def split_pool(pool, name):
        order = rng.permutation(len(pool))
        n_val = max(1, int(len(pool) * args.val_ratio))
        val = [pool[i] for i in order[:n_val]]
        train = [pool[i] for i in order[n_val:]]
        print(f"    {name}: {len(train)} train, {len(val)} val")
        return train, val

    stories_train, stories_val = split_pool(stories, "TinyStories")
    stackex_train, stackex_val = split_pool(stackex, "StackExchange")
    tulu_train, tulu_val = split_pool(tulu, "tulu_flan")
    dolly_train, dolly_val = split_pool(dolly, "Dolly")
    gsm8k_train, gsm8k_val = split_pool(gsm8k, "GSM8K")
    openmath_train, openmath_val = split_pool(openmath, "OpenMath")

    # Merge pools per phase
    conversation_train = stackex_train + tulu_train + dolly_train
    conversation_val = stackex_val + tulu_val + dolly_val
    reasoning_train = gsm8k_train + openmath_train
    reasoning_val = gsm8k_val + openmath_val

    print(f"    Conversation pool (SE+tulu+Dolly): {len(conversation_train)} train")
    print(f"    Reasoning pool (GSM8K+OpenMath): {len(reasoning_train)} train")

    # ── Step 4: Blend curriculum (train) ─────────────────────────────
    print(f"\nStep 4: Blending curriculum ({args.total_tokens:,} tokens)...")
    t2 = time.time()

    train_tokens, train_targets = blend_curriculum(
        stories_train, conversation_train, reasoning_train,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )
    print(f"  Train blend complete in {time.time() - t2:.1f}s")

    # ── Step 5: Concatenate val examples ─────────────────────────────
    print("\nStep 5: Building validation set...")
    val_pools = [("stories", stories_val), ("conversation", conversation_val),
                 ("reasoning", reasoning_val)]
    val_total = sum(len(t) for _, pool in val_pools for t, _ in pool)
    val_tokens = np.empty(val_total, dtype=np.uint32)
    val_targets = np.empty(val_total, dtype=np.int32)
    val_pos = 0
    val_counts = {}
    for name, pool in val_pools:
        for toks, tgts in pool:
            n = len(toks)
            val_tokens[val_pos:val_pos + n] = toks
            val_targets[val_pos:val_pos + n] = tgts
            val_pos += n
        val_counts[name] = len(pool)
    print(f"  Validation: {len(val_tokens):,} tokens "
          f"(stories={val_counts['stories']}, conv={val_counts['conversation']}, "
          f"reasoning={val_counts['reasoning']})")

    # ── Step 5b: Save per-phase val data for curriculum probes ──────
    print("\nStep 5b: Saving per-phase val data for curriculum probes...")
    for phase_name, pool in [("stories", stories_val),
                              ("conversation", conversation_val),
                              ("reasoning", reasoning_val)]:
        # Take up to 50 examples (~5K tokens) for 10 eval chunks at seq_len=512
        subset = pool[:50]
        p_tokens, p_targets = [], []
        for toks, tgts in subset:
            p_tokens.extend(toks)
            p_targets.extend(tgts)
        if p_tokens:
            np.save(out_dir / f"val_{phase_name}_tokens.npy",
                    np.array(p_tokens, dtype=np.uint32))
            np.save(out_dir / f"val_{phase_name}_targets.npy",
                    np.array(p_targets, dtype=np.int32))
            print(f"  val_{phase_name}: {len(p_tokens):,} tokens "
                  f"({len(subset)} examples)")

    # ── Step 6: Save ─────────────────────────────────────────────────
    print("\nStep 6: Saving output files...")
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "train_targets.npy", train_targets)
    np.save(out_dir / "val_tokens.npy", val_tokens)
    np.save(out_dir / "val_targets.npy", val_targets)

    print(f"  train_tokens.npy: {train_tokens.nbytes / 1e6:.1f} MB")
    print(f"  train_targets.npy: {train_targets.nbytes / 1e6:.1f} MB")
    print(f"  val_tokens.npy: {val_tokens.nbytes / 1e6:.1f} MB")
    print(f"  val_targets.npy: {val_targets.nbytes / 1e6:.1f} MB")

    # Compute stats
    train_valid = int(np.sum(train_targets >= 0))
    train_masked = int(np.sum(train_targets < 0))
    val_valid = int(np.sum(val_targets >= 0))
    val_masked = int(np.sum(val_targets < 0))

    # Symlink tokenizer
    tok_link = out_dir / "tokenizer.json"
    tok_source = Path(args.tokenizer).resolve()
    if tok_link.exists() or tok_link.is_symlink():
        tok_link.unlink()
    tok_link.symlink_to(tok_source)
    print(f"  tokenizer.json → {tok_source}")

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
        "special_tokens": {
            "<|im_start|>": IM_START_ID,
            "<|im_end|>": IM_END_ID,
            "<|pad|>": PAD_ID,
        },
        "train": {
            "total_tokens": len(train_tokens),
            "valid_targets": train_valid,
            "masked_targets": train_masked,
            "mask_ratio": train_masked / max(train_masked + train_valid, 1),
        },
        "val": {
            "total_tokens": len(val_tokens),
            "valid_targets": val_valid,
            "masked_targets": val_masked,
            "mask_ratio": val_masked / max(val_masked + val_valid, 1),
        },
        "curriculum": {
            "schedule": "cosine_crossfade",
            "phases": [
                {"name": "stories", "source": "roneneldan/TinyStories (HF cache)",
                 "steps": "0-15K", "fade_out": "15K-25K"},
                {"name": "conversation",
                 "sources": ["dolma/stackexchange (local)", "dolma/tulu_flan (local)",
                             "databricks-dolly-15k (HF cache)"],
                 "steps": "25K-45K", "fade_in": "15K-25K", "fade_out": "45K-55K"},
                {"name": "reasoning",
                 "sources": ["openai/gsm8k (HF cache)", "nvidia/OpenMathInstruct-1 (HF cache)"],
                 "steps": "55K-100K", "fade_in": "45K-55K"},
            ],
            "total_steps": 100_000,
            "tokens_per_step": 512,
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  meta.json saved")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Curriculum data preparation complete")
    print("=" * 60)
    print(f"  Output:        {out_dir}")
    print(f"  Vocab:         {vocab_size:,}")
    print(f"  Train tokens:  {len(train_tokens):,}")
    print(f"  Val tokens:    {len(val_tokens):,}")
    print(f"  Train valid:   {train_valid:,} ({train_valid/len(train_tokens):.1%})")
    print("  Schedule:      TinyStories → SE+tulu+Dolly → GSM8K+OpenMath")
    print("=" * 60)


if __name__ == "__main__":
    main()
