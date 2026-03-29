#!/usr/bin/env python3
"""
Prepare a Gutenberg flashcard deck from sedthh/gutenberg_english (HuggingFace).

Filters books by subject/bookshelf/author keywords, tokenizes with the same
49K BPE tokenizer used by Dolmino, and writes flat numpy arrays compatible
with BpeTokenStream. No train/val split — NLMs don't distinguish.

The output is a single contiguous token stream with EOT separators between
books. Books are ordered by category then size (largest first) so the
flashcard system sees deep immersion in each sub-domain.

Usage:
    python scripts/prepare_gutenberg_deck.py --deck philosophy_ethics
    python scripts/prepare_gutenberg_deck.py --deck philosophy_ethics --min-tokens 12288
    python scripts/prepare_gutenberg_deck.py --deck philosophy_ethics --output data/gear3_phil
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Deck definitions ─────────────────────────────────────────────────────
# Each deck is a dict of filter categories. A book matching ANY category
# is included. Within each category, ALL keywords must match (for subject
# filters) or ANY keyword must match (for author filters).

DECKS = {
    "philosophy_ethics": {
        "description": "Philosophy, ethics, moral reasoning, theology — Gear 3 deck 1",
        "subject_keywords": [
            'philosophy', 'ethics', 'moral', 'metaphysics', 'logic',
            'epistemology', 'political science', 'theology', 'religion',
            'virtue', 'justice', 'stoic', 'utilitari', 'plato',
            'aristotle', 'kant', 'nietzsche', 'hegel', 'schopenhauer',
            'spinoza', 'descartes', 'locke', 'hobbes', 'rousseau',
            'voltaire', 'reason', 'natural law', 'social contract',
            'phenomenol', 'existential', 'dialectic',
            # Expanded: social/historical themes that build moral reasoning context
            'social classes', 'social conditions', 'social life', 'social problems',
            'slavery', 'abolition', 'civil rights', "women's rights", 'suffrage',
            'democracy', 'liberty', 'freedom', 'tyranny', 'revolution',
            'war and society', 'peace', 'poverty', 'wealth', 'inequality',
            'education', 'civilization', 'culture', 'law', 'government',
            'biography', 'autobiography', 'history',
            'conduct of life', 'character', 'conscience',
        ],
        "literary_keywords": [
            'psychological fiction', 'political fiction', 'social and moral',
            'allegories', 'utopias', 'dystopias', 'satire',
            # Expanded: literary forms that carry moral/social themes
            'domestic fiction', 'bildungsroman', 'social fiction',
            'manners and customs', 'married people', 'family',
            'love stories', 'adventure stories',
        ],
        "authors": [
            # Core ethics-by-example authors
            'dickens', 'dostoevsk', 'tolstoy', 'hugo', 'twain',
            'hawthorne', 'melville', 'eliot, george', 'bronte',
            'austen', 'orwell', 'huxley', 'swift', 'voltaire',
            'shelley, mary', 'wilde, oscar', 'kafka', 'camus',
            # Expanded: major literary voices on moral/social themes
            'hardy, thomas', 'zola', 'balzac', 'stendhal', 'flaubert',
            'trollope', 'thackeray', 'gaskell', 'conrad', 'james, henry',
            'scott, walter', 'stevenson, robert', 'defoe', 'fielding',
            'richardson, samuel', 'sterne', 'smollett', 'goldsmith',
            'cervantes', 'dumas', 'sand, george', 'maupassant',
            'turgenev', 'chekhov', 'gogol', 'pushkin',
            'goethe', 'schiller', 'mann, thomas',
            'ibsen', 'strindberg',
            'poe', 'london, jack', 'crane, stephen', 'dreiser',
            'wharton', 'sinclair', 'norris, frank',
            'shaw, bernard', 'wells, h. g.', 'kipling',
            'chesterton', 'belloc',
            # Philosophers whose primary works are in Gutenberg
            'mill, john', 'bentham', 'hume, david', 'bacon, francis',
            'emerson', 'thoreau', 'whitman', 'montaigne',
            'machiavelli', 'more, thomas', 'cicero', 'seneca',
            'marcus aurelius', 'epictetus',
        ],
        # Ordering: ethics by literary example first, then formal philosophy
        "category_order": ["author_ethics", "literary_ethics", "philosophy"],
    },
    "foundations": {
        "description": "Mathematics, logic, philosophy, ethics — unified foundational reasoning. Gear 1 L0/L1 deck.",
        "math_keywords": [
            'mathematics', 'geometry', 'algebra', 'calculus', 'arithmetic',
            'trigonometry', 'number theory', 'probability', 'statistics',
            'mathematical', 'euclid', 'equations', 'theorem',
            'science', 'scientific', 'natural philosophy', 'physics',
            'astronomy', 'mechanics', 'optics', 'cosmology',
        ],
        "subject_keywords": [
            # Mathematics & formal reasoning
            'mathematics', 'geometry', 'algebra', 'calculus', 'arithmetic',
            'trigonometry', 'number theory', 'probability', 'statistics',
            'mathematical', 'euclid', 'equations', 'theorem',
            # Logic & scientific method
            'logic', 'reasoning', 'deduction', 'induction', 'syllogism',
            'fallac', 'proof', 'axiom', 'proposition',
            'science', 'scientific', 'natural philosophy', 'physics',
            'astronomy', 'mechanics', 'optics', 'cosmology',
            # Philosophy & epistemology
            'philosophy', 'ethics', 'moral', 'metaphysics',
            'epistemology', 'ontology', 'phenomenol', 'existential',
            'dialectic', 'reason', 'truth', 'knowledge',
            'political science', 'theology', 'religion',
            'virtue', 'justice', 'stoic', 'utilitari',
            'natural law', 'social contract',
            # Expanded social/historical reasoning
            'social classes', 'social conditions', 'social problems',
            'slavery', 'abolition', 'civil rights', "women's rights",
            'democracy', 'liberty', 'freedom', 'tyranny',
            'education', 'civilization', 'law', 'government',
            'conduct of life', 'character', 'conscience',
            'biography', 'autobiography', 'history',
        ],
        "literary_keywords": [
            'psychological fiction', 'political fiction', 'social and moral',
            'allegories', 'utopias', 'dystopias', 'satire',
            'domestic fiction', 'bildungsroman', 'social fiction',
            'manners and customs',
        ],
        "authors": [
            # Mathematics & natural philosophy
            'euclid', 'newton, isaac', 'leibniz', 'euler', 'gauss',
            'laplace', 'lagrange', 'riemann', 'hilbert', 'poincare',
            'descartes', 'pascal', 'fermat', 'bernoulli',
            'archimedes', 'ptolemy', 'copernicus', 'galileo', 'kepler',
            'boole', 'de morgan', 'whitehead', 'russell, bertrand',
            'frege', 'cantor', 'dedekind', 'peano',
            'darwin', 'faraday', 'maxwell', 'helmholtz',
            # Logic & formal reasoning
            'aristotle', 'bacon, francis', 'mill, john',
            'jevons', 'peirce', 'dewey',
            # Core philosophy
            'plato', 'aristotle', 'cicero', 'seneca', 'marcus aurelius',
            'epictetus', 'plotinus', 'augustine', 'aquinas',
            'machiavelli', 'more, thomas', 'montaigne',
            'hobbes', 'locke', 'hume, david', 'berkeley',
            'spinoza', 'leibniz', 'kant', 'hegel',
            'schopenhauer', 'nietzsche', 'kierkegaard',
            'emerson', 'thoreau', 'whitman',
            'bentham', 'mill, john', 'sidgwick',
            'james, william', 'royce', 'santayana',
            'voltaire', 'rousseau', 'diderot',
            # Ethics through literature
            'dickens', 'dostoevsk', 'tolstoy', 'hugo', 'twain',
            'hawthorne', 'melville', 'eliot, george', 'bronte',
            'austen', 'orwell', 'huxley', 'swift',
            'shelley, mary', 'wilde, oscar', 'kafka',
            'hardy, thomas', 'zola', 'balzac', 'flaubert',
            'conrad', 'james, henry', 'cervantes',
            'goethe', 'schiller', 'mann, thomas',
            'shaw, bernard', 'wells, h. g.', 'chesterton',
        ],
        "category_order": ["mathematics", "philosophy", "literary_ethics", "author_ethics"],
    },
}

DEFAULT_TOKENIZER = "data/dolmino_100b/tokenizer.json"
METADATA_PATH = "/bulk-store/training-datasets/gutenberg_hf/metadata.jsonl"


def load_candidates(metadata_path: str, deck_name: str,
                    min_tokens: int) -> list[dict]:
    """Filter books matching deck criteria and minimum token count."""
    deck = DECKS[deck_name]
    records = [json.loads(line) for line in open(metadata_path)]

    subject_kw = deck["subject_keywords"]
    literary_kw = deck["literary_keywords"]
    author_kw = deck["authors"]

    seen = set()
    candidates = []

    for r in records:
        if r['token_est'] < min_tokens:
            continue

        combined = (r['subjects'] + ' ' + r['bookshelves']).lower()
        auth_lower = r['authors'].lower()

        category = None
        math_kw = deck.get("math_keywords", [])
        if math_kw and any(kw in combined for kw in math_kw):
            category = "mathematics"
        elif any(kw in combined for kw in subject_kw):
            category = "philosophy"
        elif any(kw in combined for kw in literary_kw):
            category = "literary_ethics"
        elif any(a in auth_lower for a in author_kw):
            category = "author_ethics"

        if category and r['idx'] not in seen:
            seen.add(r['idx'])
            r['deck_category'] = category
            candidates.append(r)

    # Order by deck category priority, then largest first within category
    order = deck.get("category_order", ["philosophy", "literary_ethics", "author_ethics"])
    cat_rank = {c: i for i, c in enumerate(order)}
    candidates.sort(key=lambda b: (cat_rank.get(b['deck_category'], 99),
                                   -b['token_est']))
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Gutenberg flashcard deck for NL-Hecate")
    parser.add_argument("--deck", required=True, choices=list(DECKS.keys()),
                        help="Deck name to prepare")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/gear3_{deck})")
    parser.add_argument("--metadata", type=str, default=METADATA_PATH,
                        help="Path to gutenberg metadata.jsonl")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="Path to BPE tokenizer.json")
    parser.add_argument("--min-tokens", type=int, default=12288,
                        help="Minimum estimated tokens per book (default: 12288)")
    parser.add_argument("--max-books", type=int, default=0,
                        help="Max books to include (0 = all qualifying)")
    parser.add_argument("--target-tokens", type=int, default=0,
                        help="Target token count — stop adding books once reached (0 = all)")
    args = parser.parse_args()

    out_dir = Path(args.output or f"data/gear3_{args.deck}")
    tokenizer_path = Path(args.tokenizer)

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Filter candidates ─────────────────────────────────────
    print(f"Step 1: Filtering {args.deck} candidates (min {args.min_tokens:,} tokens)...")
    candidates = load_candidates(args.metadata, args.deck, args.min_tokens)

    if args.max_books > 0:
        candidates = candidates[:args.max_books]

    if args.target_tokens > 0:
        trimmed = []
        running = 0
        for c in candidates:
            trimmed.append(c)
            running += c['token_est']
            if running >= args.target_tokens:
                break
        candidates = trimmed

    by_cat = {}
    for c in candidates:
        cat = c['deck_category']
        by_cat.setdefault(cat, []).append(c)

    print(f"  Total: {len(candidates)} books, ~{sum(c['token_est'] for c in candidates):,} est. tokens")
    for cat, books in by_cat.items():
        print(f"    {cat}: {len(books)} books, ~{sum(b['token_est'] for b in books):,} tokens")

    # ── Step 2: Load tokenizer ────────────────────────────────────────
    print(f"\nStep 2: Loading tokenizer: {tokenizer_path}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None:
        print("ERROR: No <|endoftext|> token in tokenizer.", file=sys.stderr)
        sys.exit(1)
    print(f"  Vocab: {vocab_size:,}, EOT: {eot_id}")

    # ── Step 3: Load HF dataset and tokenize ──────────────────────────
    print(f"\nStep 3: Loading Gutenberg dataset and tokenizing...")
    from datasets import load_dataset
    ds = load_dataset('sedthh/gutenberg_english', split='train')

    # Build index set of candidate row indices
    candidate_indices = [c['idx'] for c in candidates]
    idx_to_order = {c['idx']: i for i, c in enumerate(candidates)}

    t0 = time.time()
    all_tokens = []
    book_count = 0
    total_book_tokens = 0

    # Process in deck order
    for i, cand in enumerate(candidates):
        row = ds[cand['idx']]
        text = row['TEXT']

        encoded = tokenizer.encode(text)
        tokens = encoded.ids

        if len(tokens) < args.min_tokens:
            continue  # actual tokens below threshold (estimate was off)

        all_tokens.extend(tokens)
        all_tokens.append(eot_id)
        book_count += 1
        total_book_tokens += len(tokens)

        if (i + 1) % 100 == 0 or i == len(candidates) - 1:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(candidates)}] {book_count} books, "
                  f"{total_book_tokens:,} tokens ({elapsed:.0f}s)",
                  end="\r")

    print(f"\n  Tokenized {book_count} books → {len(all_tokens):,} tokens "
          f"in {time.time()-t0:.1f}s")

    # ── Step 4: Build arrays ──────────────────────────────────────────
    print(f"\nStep 4: Building numpy arrays...")
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]

    tokens_arr = np.array(input_tokens, dtype=np.uint32)
    targets_arr = np.array(target_tokens, dtype=np.int32)

    # ── Step 5: Save ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nStep 5: Saving to {out_dir}/...")

    np.save(out_dir / "train_tokens.npy", tokens_arr)
    np.save(out_dir / "train_targets.npy", targets_arr)

    import shutil
    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")

    meta = {
        "vocab_size": vocab_size,
        "tokenizer": "tokenizer.json",
        "special_tokens": {"<|endoftext|>": eot_id},
        "train": {
            "split": "train",
            "documents": book_count,
            "total_tokens": len(input_tokens),
        },
        "source": f"sedthh/gutenberg_english ({args.deck})",
        "deck": args.deck,
        "min_tokens": args.min_tokens,
        "category_counts": {cat: len(books) for cat, books in by_cat.items()},
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save manifest for curriculum ordering reference
    manifest = []
    for c in candidates:
        manifest.append({
            "title": c['title'],
            "authors": c['authors'],
            "category": c['deck_category'],
            "token_est": c['token_est'],
            "text_id": c['text_id'],
        })
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    mb = (len(input_tokens) * 4) / 1e6
    print(f"\n{'='*60}")
    print(f"Deck: {args.deck}")
    print(f"{'='*60}")
    print(f"  Books:  {book_count}")
    print(f"  Tokens: {len(input_tokens):,} ({mb:.0f} MB)")
    print(f"  Output: {out_dir}")
    for cat, books in by_cat.items():
        print(f"  {cat}: {len(books)} books")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
