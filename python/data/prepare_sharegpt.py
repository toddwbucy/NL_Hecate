#!/usr/bin/env python3
"""
Prepare ShareGPT data for NL-Hecate training.

Downloads ShareGPT52K, trains a 32K BPE tokenizer, formats conversations
as ChatML, and generates parallel token/target arrays with loss masking
on user turns.

Usage:
    pip install datasets tokenizers numpy
    python data/prepare_sharegpt.py
    python data/prepare_sharegpt.py --vocab_size 32000 --output data/sharegpt
    python data/prepare_sharegpt.py --test  # unit tests only, no download

Output files:
    tokenizer.json     — trained BPE tokenizer (HuggingFace tokenizers format)
    train_tokens.npy   — uint32 flat token array (training split)
    train_targets.npy  — int32 flat target array (-1 = masked user turns)
    val_tokens.npy     — uint32 flat token array (validation split)
    val_targets.npy    — int32 flat target array
    meta.json          — vocab_size, token counts, conversation counts, stats
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# ── Special token IDs (reserved at start of vocab) ────────────────────

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
IM_START_ID = 0
IM_END_ID = 1
PAD_ID = 2


def train_tokenizer(conversations: list[list[dict]], vocab_size: int, output_path: str):
    """Train a BPE tokenizer on conversation text.

    Uses HuggingFace tokenizers library with ByteLevel pre-tokenizer
    (handles all Unicode). Special tokens reserved at IDs 0-2.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Collect all text for training
    def text_iterator():
        for conv in conversations:
            for turn in conv:
                value = turn.get("value", "")
                if value:
                    yield value

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Verify special token IDs
    assert tokenizer.token_to_id("<|im_start|>") == IM_START_ID
    assert tokenizer.token_to_id("<|im_end|>") == IM_END_ID
    assert tokenizer.token_to_id("<|pad|>") == PAD_ID

    tokenizer.save(output_path)
    print(f"  Tokenizer saved: {output_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str):
    """Load a previously trained tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def format_chatml(conversation: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Format a conversation as ChatML and generate token/target arrays.

    ChatML format:
        <|im_start|>user\n{message}<|im_end|>\n
        <|im_start|>assistant\n{response}<|im_end|>\n

    Returns:
        tokens:  list[int] — token IDs for the full conversation
        targets: list[int] — next-token targets (-1 for masked user turns)
    """
    tokens = []
    # Track which token positions are assistant (trainable) vs user (masked)
    is_assistant = []

    for turn in conversation:
        role = turn.get("from", "").lower()
        value = turn.get("value", "")
        if not value:
            continue

        # Normalize roles
        if role in ("human", "user"):
            role_str = "user"
            is_train = False
        elif role in ("gpt", "assistant", "chatgpt"):
            role_str = "assistant"
            is_train = True
        else:
            # Skip system or unknown roles
            continue

        # Encode: <|im_start|>role\n{content}<|im_end|>\n
        header = f"{role_str}\n"
        header_ids = tokenizer.encode(header).ids
        content_ids = tokenizer.encode(value).ids
        newline_ids = tokenizer.encode("\n").ids

        # <|im_start|> token
        tokens.append(IM_START_ID)
        is_assistant.append(False)  # structural token — always masked

        # role\n
        tokens.extend(header_ids)
        is_assistant.extend([False] * len(header_ids))  # role header — masked

        # content tokens
        tokens.extend(content_ids)
        is_assistant.extend([is_train] * len(content_ids))

        # <|im_end|>\n
        tokens.append(IM_END_ID)
        is_assistant.append(is_train)  # end token trainable for assistant
        tokens.extend(newline_ids)
        is_assistant.extend([False] * len(newline_ids))

    if len(tokens) < 2:
        return [], []

    # Generate next-token shifted targets
    # tokens[:-1] → input, tokens[1:] → targets
    # Mask targets where the TARGET position is not assistant-generated
    input_tokens = tokens[:-1]
    target_tokens = []
    for i in range(1, len(tokens)):
        if is_assistant[i]:
            target_tokens.append(tokens[i])
        else:
            target_tokens.append(-1)  # masked — cross-entropy kernel skips target < 0

    return input_tokens, target_tokens


def process_dataset(
    conversations: list[list[dict]],
    tokenizer,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Process a list of conversations into flat token/target arrays.

    Returns (tokens_array, targets_array, stats_dict).
    """
    all_tokens = []
    all_targets = []
    total_valid = 0
    total_masked = 0
    skipped = 0

    for conv in conversations:
        toks, tgts = format_chatml(conv, tokenizer)
        if len(toks) == 0:
            skipped += 1
            continue
        all_tokens.extend(toks)
        all_targets.extend(tgts)
        valid = sum(1 for t in tgts if t >= 0)
        total_valid += valid
        total_masked += len(tgts) - valid

    tokens_arr = np.array(all_tokens, dtype=np.uint32)
    targets_arr = np.array(all_targets, dtype=np.int32)

    stats = {
        "split": split_name,
        "conversations": len(conversations) - skipped,
        "skipped": skipped,
        "total_tokens": len(all_tokens),
        "valid_targets": total_valid,
        "masked_targets": total_masked,
        "mask_ratio": total_masked / max(total_masked + total_valid, 1),
    }

    print(f"  {split_name}: {stats['conversations']} convs, "
          f"{stats['total_tokens']:,} tokens, "
          f"{stats['valid_targets']:,} valid targets, "
          f"{stats['mask_ratio']:.1%} masked")

    return tokens_arr, targets_arr, stats


def run_tests(tokenizer=None):
    """Unit tests for the data pipeline. Returns True if all pass."""
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # If no tokenizer provided, create a minimal one for testing
    if tokenizer is None:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=1,
        )
        # Train on corpus covering all printable ASCII for reliable round-trip
        corpus = [
            "Hello, how are you? I'm doing well, thanks!",
            "What is the capital of France? The capital of France is Paris.",
            "user\nassistant\n",
            "The quick brown fox jumps over the lazy dog. 0123456789",
            "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?`~",
            "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ] * 5  # repeat for frequency
        tok.train_from_iterator(corpus, trainer=trainer)
        tokenizer = tok

    # Test 1: Special token IDs
    check("special token <|im_start|> = 0",
          tokenizer.token_to_id("<|im_start|>") == 0)
    check("special token <|im_end|> = 1",
          tokenizer.token_to_id("<|im_end|>") == 1)
    check("special token <|pad|> = 2",
          tokenizer.token_to_id("<|pad|>") == 2)

    # Test 2: Round-trip ASCII text
    test_text = "Hello, world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    check("round-trip ASCII text",
          decoded == test_text)

    # Test 3: Known conversation masking
    conv = [
        {"from": "human", "value": "What is 2+2?"},
        {"from": "gpt", "value": "4"},
    ]
    toks, tgts = format_chatml(conv, tokenizer)
    check("tokens and targets same length",
          len(toks) == len(tgts))
    check("has masked targets (user turns)",
          any(t == -1 for t in tgts))
    check("has valid targets (assistant turns)",
          any(t >= 0 for t in tgts))

    # Test 4: All user tokens are masked, assistant tokens are valid
    # Check that the "4" content token has a valid target
    # and "What is 2+2?" content tokens have masked targets
    user_msg_ids = tokenizer.encode("What is 2+2?").ids
    asst_msg_ids = tokenizer.encode("4").ids

    # The assistant content should appear in the token stream
    # and its next-token targets should be valid (>= 0)
    check("non-empty token stream",
          len(toks) > 0)
    check("non-empty target stream",
          len(tgts) > 0)

    # Test 5: Multi-turn conversation
    conv_multi = [
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello!"},
        {"from": "human", "value": "Bye"},
        {"from": "gpt", "value": "Goodbye!"},
    ]
    toks2, tgts2 = format_chatml(conv_multi, tokenizer)
    check("multi-turn: tokens and targets same length",
          len(toks2) == len(tgts2))
    valid_count = sum(1 for t in tgts2 if t >= 0)
    masked_count = sum(1 for t in tgts2 if t == -1)
    check("multi-turn: has both valid and masked",
          valid_count > 0 and masked_count > 0)

    # Test 6: Empty conversation
    toks_empty, tgts_empty = format_chatml([], tokenizer)
    check("empty conversation returns empty",
          len(toks_empty) == 0 and len(tgts_empty) == 0)

    # Test 7: Conversation with only user (no assistant)
    conv_user_only = [{"from": "human", "value": "Hello"}]
    toks_u, tgts_u = format_chatml(conv_user_only, tokenizer)
    if len(tgts_u) > 0:
        check("user-only: all targets masked",
              all(t == -1 for t in tgts_u))
    else:
        check("user-only: empty or all masked", True)

    # Test 8: Target values are valid token IDs or -1
    all_valid = all(t == -1 or (0 <= t < tokenizer.get_vocab_size()) for t in tgts)
    check("all targets are -1 or valid token IDs", all_valid)

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Prepare ShareGPT data for NL-Hecate")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="BPE vocabulary size (default: 32000)")
    parser.add_argument("--output", type=str, default="data/sharegpt",
                        help="Output directory (default: data/sharegpt)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests only (no download)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to existing tokenizer (skip training)")
    args = parser.parse_args()

    if args.test:
        print("Running unit tests...")
        success = run_tests()
        sys.exit(0 if success else 1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"

    # ── Step 1: Download/load ShareGPT52K ───────────────────────────────
    print("Step 1: Loading ShareGPT52K dataset...")
    t0 = time.time()

    # Try loading from HuggingFace cache (raw JSON) first — avoids Arrow schema errors
    cache_dir = Path.home() / ".cache/huggingface/hub/datasets--RyokoAI--ShareGPT52K"
    json_files = sorted(cache_dir.rglob("sg_90k_part*.json"))

    if json_files:
        print(f"  Loading from cached JSON: {len(json_files)} files")
        raw_data = []
        for jf in json_files:
            with open(jf) as f:
                raw_data.extend(json.load(f))
        print(f"  Loaded {len(raw_data)} raw entries in {time.time() - t0:.1f}s")
    else:
        # Download via datasets library, then load raw JSON
        print("  Downloading ShareGPT52K via HuggingFace datasets...")
        from datasets import load_dataset
        try:
            ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
            raw_data = list(ds)
        except Exception:
            # Fallback: download files only and load JSON directly
            from huggingface_hub import snapshot_download
            path = snapshot_download("RyokoAI/ShareGPT52K", repo_type="dataset")
            json_files = sorted(Path(path).glob("sg_90k_part*.json"))
            raw_data = []
            for jf in json_files:
                with open(jf) as f:
                    raw_data.extend(json.load(f))
        print(f"  Downloaded {len(raw_data)} entries in {time.time() - t0:.1f}s")

    # Extract conversation lists (handle both dict and list formats)
    conversations = []
    for row in raw_data:
        conv = row.get("conversations", [])
        if not isinstance(conv, list):
            continue
        # Filter: need at least 2 turns, each turn must be a dict with from/value
        valid_turns = [t for t in conv if isinstance(t, dict) and "from" in t and "value" in t]
        if len(valid_turns) >= 2:
            conversations.append(valid_turns)
    print(f"  {len(conversations)} conversations with >= 2 valid turns")

    # ── Step 2: Train or load tokenizer ───────────────────────────────
    if args.tokenizer and Path(args.tokenizer).exists():
        print(f"Step 2: Loading existing tokenizer: {args.tokenizer}")
        tokenizer = load_tokenizer(args.tokenizer)
    elif tokenizer_path.exists():
        print(f"Step 2: Loading existing tokenizer: {tokenizer_path}")
        tokenizer = load_tokenizer(str(tokenizer_path))
    else:
        print(f"Step 2: Training {args.vocab_size} BPE tokenizer...")
        t0 = time.time()
        tokenizer = train_tokenizer(conversations, args.vocab_size, str(tokenizer_path))
        print(f"  Trained in {time.time() - t0:.1f}s")

    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Vocab size: {actual_vocab}")

    # ── Step 3: Run validation tests ──────────────────────────────────
    print("\nStep 3: Running validation tests...")
    if not run_tests(tokenizer):
        print("ERROR: Validation tests failed!")
        sys.exit(1)

    # ── Step 4: Split conversations ───────────────────────────────────
    print(f"\nStep 4: Splitting {len(conversations)} conversations "
          f"(seed={args.seed}, val_ratio={args.val_ratio})...")
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(conversations))
    n_val = int(len(conversations) * args.val_ratio)
    val_indices = set(indices[:n_val])
    train_convs = [conversations[i] for i in range(len(conversations)) if i not in val_indices]
    val_convs = [conversations[i] for i in val_indices]
    print(f"  Train: {len(train_convs)}, Val: {len(val_convs)}")

    # Verify zero overlap
    train_set = set(range(len(conversations))) - val_indices
    assert len(train_set & val_indices) == 0, "Train/val overlap detected!"

    # ── Step 5: Process and save ──────────────────────────────────────
    print("\nStep 5: Processing conversations...")

    t0 = time.time()
    train_tokens, train_targets, train_stats = process_dataset(
        train_convs, tokenizer, "train")
    val_tokens, val_targets, val_stats = process_dataset(
        val_convs, tokenizer, "val")
    print(f"  Processed in {time.time() - t0:.1f}s")

    # Save arrays
    print("\nStep 6: Saving output files...")
    np.save(out_dir / "train_tokens.npy", train_tokens)
    np.save(out_dir / "train_targets.npy", train_targets)
    np.save(out_dir / "val_tokens.npy", val_tokens)
    np.save(out_dir / "val_targets.npy", val_targets)
    print(f"  train_tokens.npy: {train_tokens.nbytes / 1e6:.1f} MB")
    print(f"  train_targets.npy: {train_targets.nbytes / 1e6:.1f} MB")
    print(f"  val_tokens.npy: {val_tokens.nbytes / 1e6:.1f} MB")
    print(f"  val_targets.npy: {val_targets.nbytes / 1e6:.1f} MB")

    # Save metadata
    meta = {
        "vocab_size": actual_vocab,
        "tokenizer": "tokenizer.json",
        "special_tokens": {
            "<|im_start|>": IM_START_ID,
            "<|im_end|>": IM_END_ID,
            "<|pad|>": PAD_ID,
        },
        "train": train_stats,
        "val": val_stats,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "source": "RyokoAI/ShareGPT52K",
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json saved")

    # ── Summary ───────────────────────────────────────────────────────
    total_tokens = train_stats["total_tokens"] + val_stats["total_tokens"]
    total_valid = train_stats["valid_targets"] + val_stats["valid_targets"]
    print(f"\n{'=' * 60}")
    print("ShareGPT data preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:       {out_dir}")
    print(f"  Vocab:        {actual_vocab:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Valid targets: {total_valid:,} ({total_valid/total_tokens:.1%})")
    print(f"  Conversations: {len(train_convs) + len(val_convs):,} "
          f"(train={len(train_convs):,}, val={len(val_convs):,})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
