#!/usr/bin/env python3
"""
Prepare reasoning/Socratic dataset for NL-Hecate pre-training.

Design principle: NL models have persistent memory M — they don't need
role markers, turn delimiters, or <think> blocks. Strip all scaffolding,
concatenate raw text as a continuous token stream. M tracks context internally.

Sources:
  - am-r1-1.4m: DeepSeek-R1 reasoning traces (strip <think> blocks, keep Q→A)
  - big-reasoning-traces: DeepSeek reasoning traces (677K, strip <think>)
  - general-thought-430k: Q&A with model reasoning (strip metadata)
  - SocraTeach: Multi-turn Socratic math tutoring dialogues
  - cot_collection: CoT rationale chains (source→rationale→target)
  - flan_cot: NLI/reasoning with chain-of-thought labels

Output: data/reasoning/
  train_tokens.npy   uint32[N]   — continuous token stream
  train_targets.npy  int32[N]    — next-token targets (shifted by 1)
  val_tokens.npy
  val_targets.npy
  meta.json

Usage:
  python scripts/prepare_reasoning.py
  python scripts/prepare_reasoning.py --target_tokens 200_000_000 --output data/reasoning
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

TOKENIZER_PATH = "data/fineweb_edu/tokenizer.json"
OUTPUT_DIR     = "data/reasoning"
TARGET_TOKENS  = 300_000_000  # 300M tokens
VAL_RATIO      = 0.05
SEED           = 42
VOCAB_SIZE     = 32000
EOT_ID         = 3  # <|endoftext|>

# Source mix ratios — must sum to 1.0
MIX = {
    "am_r1":              0.25,   # DeepSeek-R1 distilled reasoning
    "big_deepseek":       0.20,   # More DeepSeek reasoning traces
    "general_thought":    0.15,   # diverse Q&A with reasoning
    "socrateach":         0.15,   # Socratic dialogue (high signal density)
    "cot_collection":     0.15,   # CoT rationales
    "flan_cot":           0.10,   # NLI/reasoning with CoT
}

# Source paths
AM_R1_PATH           = "/bulk-store/training-datasets/reasoning-traces/am-r1-1.4m"
BIG_DEEPSEEK_PATH    = "/bulk-store/training-datasets/reasoning-traces/big-reasoning-traces/DeepSeek"
GENERAL_THOUGHT_PATH = "/bulk-store/training-datasets/reasoning-traces/general-thought-430k"
SOCRATEACH_PATH      = "/bulk-store/training-datasets/socratic/SocraticLM/data"
COT_COLLECTION_PATH  = "/bulk-store/training-datasets/chain_of_thought/cot_collection/cot_collection.jsonl"
FLAN_COT_PATH        = "/bulk-store/training-datasets/chain_of_thought/flan_cot/flan_cot.jsonl"


# ── Text cleaning ─────────────────────────────────────────────────────────────

# Patterns to strip — all the transformer scaffolding
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ROLE_MARKERS_RE = re.compile(
    r"<\|(?:im_start|im_end|user|assistant|system|human|bot|eot_id|"
    r"start_header_id|end_header_id|begin_of_text|end_of_text)\|>"
)
TURN_MARKERS_RE = re.compile(
    r"^(?:###?\s*(?:Human|Assistant|User|System|Teacher|Student)\s*:?\s*)",
    re.MULTILINE,
)
# DeepSeek-specific markers
DEEPSEEK_RE = re.compile(r"<\|(?:begin|end)_of_(?:thought|solution)\|>")


def clean_text(text: str) -> str:
    """Strip all role/turn/think markup, collapse whitespace."""
    text = THINK_BLOCK_RE.sub(" ", text)
    text = ROLE_MARKERS_RE.sub("", text)
    text = TURN_MARKERS_RE.sub("", text)
    text = DEEPSEEK_RE.sub("", text)
    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize_docs(tok: Tokenizer, docs: list[str], budget: int) -> list[int]:
    """Tokenize documents into a continuous stream, separated by EOT."""
    all_ids: list[int] = []
    for doc in docs:
        if len(all_ids) >= budget:
            break
        cleaned = clean_text(doc)
        if len(cleaned) < 20:  # skip trivially short
            continue
        ids = tok.encode(cleaned).ids
        all_ids.extend(ids)
        all_ids.append(EOT_ID)
    return all_ids[:budget]


# ── Source: am-r1-1.4m (DeepSeek-R1 reasoning traces) ────────────────────────

def load_am_r1(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load am-r1 reasoning traces. Strip <think> blocks → pure Q→A."""
    print("  [am_r1] Loading...")

    # Use the smaller sample first, then the full files
    docs = []
    for fname in sorted(os.listdir(AM_R1_PATH)):
        if not fname.endswith(".jsonl"):
            continue
        if "sample" in fname:
            continue  # skip sample file, use full
        fpath = os.path.join(AM_R1_PATH, fname)
        print(f"    Reading {fname}...")
        with open(fpath) as f:
            for line in f:
                if len(docs) * 150 > budget * 1.2:  # rough estimate: ~150 tok/doc
                    break
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = entry.get("messages", [])
                if not messages:
                    continue
                # Concatenate all message content, stripping roles
                parts = []
                for msg in messages:
                    content = msg.get("content", "").strip()
                    if content:
                        parts.append(content)
                if parts:
                    docs.append("\n\n".join(parts))
        if len(docs) * 150 > budget * 1.2:
            break

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [am_r1] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Source: big-reasoning-traces/DeepSeek (677K reasoning traces) ─────────────

def load_big_deepseek(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load big DeepSeek reasoning traces from parquet. Strip <think> blocks."""
    print("  [big_deepseek] Loading...")
    import pyarrow.parquet as pq

    docs = []
    for fname in sorted(os.listdir(BIG_DEEPSEEK_PATH)):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(BIG_DEEPSEEK_PATH, fname)
        table = pq.read_table(fpath, columns=["prompt", "response"])
        data = table.to_pydict()
        for prompt, response in zip(data["prompt"], data["response"]):
            prompt = str(prompt or "").strip()
            response = str(response or "").strip()
            if not prompt or not response:
                continue
            # Combine prompt and response as flowing text
            docs.append(f"{prompt}\n\n{response}")
        if len(docs) * 300 > budget * 1.2:
            break

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [big_deepseek] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Source: flan_cot (NLI/reasoning with chain-of-thought) ────────────────────

def load_flan_cot(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load flan CoT data. Combine inputs→labels as continuous reasoning text."""
    print("  [flan_cot] Loading...")
    docs = []

    with open(FLAN_COT_PATH, "rb") as f:
        for line in f:
            if len(docs) * 80 > budget * 1.2:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            inputs = entry.get("inputs", "").strip()
            labels = entry.get("labels", "").strip()
            if not inputs or not labels:
                continue
            docs.append(f"{inputs}\n\n{labels}")

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [flan_cot] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Source: general-thought-430k ──────────────────────────────────────────────

def load_general_thought(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load general-thought Q&A traces from parquet. Strip metadata, keep Q+reasoning+A."""
    print("  [general_thought] Loading...")
    import pyarrow.parquet as pq

    docs = []
    for fname in sorted(os.listdir(GENERAL_THOUGHT_PATH)):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(GENERAL_THOUGHT_PATH, fname)
        table = pq.read_table(fpath, columns=["question", "model_reasoning", "model_answer"])
        data = table.to_pydict()
        for q, reasoning, answer in zip(
            data["question"], data["model_reasoning"], data["model_answer"]
        ):
            q = str(q or "").strip()
            reasoning = str(reasoning or "").strip()
            answer = str(answer or "").strip()
            if not q or not answer:
                continue
            # Combine: question then answer (reasoning is the think block — strip it
            # but its content IS the reasoning, so keep it as raw text without markers)
            parts = [q]
            if reasoning:
                parts.append(reasoning)
            parts.append(answer)
            docs.append("\n\n".join(parts))
        if len(docs) * 200 > budget * 1.2:
            break

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [general_thought] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Source: SocraTeach (Socratic math tutoring) ───────────────────────────────

def load_socrateach(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load SocraTeach multi-turn dialogues. Flatten teacher↔student into text flow."""
    print("  [socrateach] Loading...")
    docs = []

    # Multi-turn dialogues (richer, preferred)
    multi_path = os.path.join(SOCRATEACH_PATH, "SocraTeach_multi.json")
    with open(multi_path) as f:
        multi = json.load(f)

    for key, entry in multi.items():
        question = entry.get("question", "").strip()
        dialogues = entry.get("dialogues", {})
        answer = entry.get("answer", "").strip()

        for dial_key, turns in dialogues.items():
            parts = [question]  # start with the math problem
            for turn in turns:
                if isinstance(turn, dict):
                    sys_msg = turn.get("system", "").strip()
                    usr_msg = turn.get("user", "").strip()
                    if sys_msg:
                        parts.append(sys_msg)
                    if usr_msg:
                        parts.append(usr_msg)
                    # Check for END marker
                    if "END" in turn:
                        break
            if answer:
                parts.append(f"The answer is {answer}.")
            docs.append("\n\n".join(parts))

    # Single-turn exchanges (supplement)
    single_path = os.path.join(SOCRATEACH_PATH, "SocraTeach_single.json")
    with open(single_path) as f:
        single = json.load(f)

    for key, entry in single.items():
        prompt = entry.get("prompt", "").strip()
        response = entry.get("response", "").strip()
        history = entry.get("history", [])

        parts = []
        for h_pair in history:
            for h in h_pair:
                h = str(h).strip()
                if h:
                    parts.append(h)
        if prompt:
            parts.append(prompt)
        if response:
            parts.append(response)
        if parts:
            docs.append("\n\n".join(parts))

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [socrateach] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Source: CoT collection ────────────────────────────────────────────────────

def load_cot_collection(tok: Tokenizer, budget: int, rng: random.Random) -> list[int]:
    """Load CoT rationale chains. Combine source→rationale→target as flowing text."""
    print("  [cot_collection] Loading...")
    docs = []

    with open(COT_COLLECTION_PATH) as f:
        for line in f:
            if len(docs) * 100 > budget * 1.2:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = entry.get("source", "").strip()
            rationale = entry.get("rationale", "").strip()
            target = entry.get("target", "").strip()
            if not source or not target:
                continue
            parts = [source]
            if rationale:
                parts.append(rationale)
            parts.append(target)
            docs.append("\n\n".join(parts))

    rng.shuffle(docs)
    result = tokenize_docs(tok, docs, budget)
    print(f"  [cot_collection] {len(result):,} tokens from {len(docs):,} documents")
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reasoning dataset for NL-Hecate pre-training"
    )
    parser.add_argument("--target_tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = Tokenizer.from_file(args.tokenizer)
    assert tok.get_vocab_size() == VOCAB_SIZE, (
        f"Tokenizer vocab_size={tok.get_vocab_size()}, expected {VOCAB_SIZE}"
    )

    budgets = {src: int(args.target_tokens * ratio) for src, ratio in MIX.items()}

    print(f"\nReasoning dataset prep — {args.target_tokens:,} tokens target")
    print(f"  Mix:     { {k: f'{v*100:.0f}%' for k,v in MIX.items()} }")
    print(f"  Budgets: { {k: f'{v:,}' for k,v in budgets.items()} }\n")

    t0 = time.time()

    loaders = {
        "am_r1":           (load_am_r1,           rng),
        "big_deepseek":    (load_big_deepseek,     random.Random(args.seed + 4)),
        "general_thought": (load_general_thought,  random.Random(args.seed + 1)),
        "socrateach":      (load_socrateach,       random.Random(args.seed + 2)),
        "cot_collection":  (load_cot_collection,   random.Random(args.seed + 3)),
        "flan_cot":        (load_flan_cot,         random.Random(args.seed + 5)),
    }

    source_tokens: dict[str, list[int]] = {}
    for name, (loader, src_rng) in loaders.items():
        source_tokens[name] = loader(tok, budgets[name], src_rng)

    # Combine all sources
    print("\nMixing and shuffling...")
    all_tok: list[int] = []
    for name in MIX:
        all_tok.extend(source_tokens[name])

    # Shuffle in 512-token blocks to preserve local coherence
    chunk_size = 512
    n_chunks = len(all_tok) // chunk_size
    chunk_order = list(range(n_chunks))
    rng.shuffle(chunk_order)

    tok_shuf: list[int] = []
    for idx in chunk_order:
        s = idx * chunk_size
        tok_shuf.extend(all_tok[s:s + chunk_size])

    total = len(tok_shuf)
    n_val = max(512, int(total * args.val_ratio))
    n_train = total - n_val

    # Build next-token prediction pairs (no masking — learn from everything)
    train_tokens = np.array(tok_shuf[:n_train], dtype=np.uint32)
    val_tokens = np.array(tok_shuf[n_train:], dtype=np.uint32)

    # Targets: shifted by 1 (standard next-token prediction)
    train_targets = np.empty(n_train, dtype=np.int32)
    train_targets[:-1] = train_tokens[1:]
    train_targets[-1] = EOT_ID

    val_targets = np.empty(n_val, dtype=np.int32)
    val_targets[:-1] = val_tokens[1:]
    val_targets[-1] = EOT_ID

    print(f"Saving to {out_dir}...")
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "train_targets.npy", train_targets)
    np.save(out_dir / "val_tokens.npy", val_tokens)
    np.save(out_dir / "val_targets.npy", val_targets)

    # Copy tokenizer
    import shutil
    shutil.copy2(args.tokenizer, out_dir / "tokenizer.json")

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "format": "sharegpt",
        "description": (
            "Reasoning/Socratic dataset for NL pre-training. "
            "All role/turn/think markup stripped — raw text flow for M to track context."
        ),
        "train": {
            "total_tokens": int(n_train),
        },
        "val": {
            "total_tokens": int(n_val),
        },
        "sources": {
            name: {
                "tokens": len(source_tokens[name]),
                "budget": budgets[name],
                "ratio": MIX[name],
            }
            for name in MIX
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print("Reasoning dataset preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:         {out_dir}")
    print(f"  Total tokens:   {total:,}")
    print(f"  Train tokens:   {n_train:,}")
    print(f"  Val tokens:     {n_val:,}")
    print("  Sources:")
    for name in MIX:
        print(f"    {name:20s} {len(source_tokens[name]):>12,} tokens ({MIX[name]*100:.0f}%)")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
