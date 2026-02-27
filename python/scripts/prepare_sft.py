#!/usr/bin/env python3
"""
Prepare Phase 2 SFT dataset for NL-Hecate.

Sources (mix ratios configurable):
  - ise-uiuc/Magicoder-Evol-Instruct-110K  (coding, instruction following)
  - meta-math/MetaMathQA                    (math reasoning, chain-of-thought)
  - Salesforce/xlam-function-calling-60k    (tool use, structured function calls)
  - HADES custom                            (knowledge graph tool use, grounded)

Output: data/sft_phase2/
  train_tokens.npy   uint32[N]   — input token IDs
  train_targets.npy  int32[N]    — target IDs; MASK_SENTINEL=32000 for non-assistant tokens
  val_tokens.npy
  val_targets.npy
  meta.json

Loss masking: targets for system/user turns are set to MASK_SENTINEL (vocab_size=32000).
The Rust cross-entropy kernel already skips targets >= vocab_size (see engine/loop.py:449).

Tool call format (XML, maps to actual MCP tool signatures):
  <tool_call>
  {"name": "hades_query", "parameters": {"text": "...", "database": "NL"}}
  </tool_call>
  <tool_result>
  [result content]
  </tool_result>

Usage:
  python scripts/prepare_sft.py
  python scripts/prepare_sft.py --target_tokens 15_000_000 --output data/sft_phase2
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# ── Constants ──────────────────────────────────────────────────────────────────

TOKENIZER_PATH = "data/fineweb_edu/tokenizer.json"
OUTPUT_DIR     = "data/sft_phase2"
TARGET_TOKENS  = 15_000_000
VAL_RATIO      = 0.05
SEED           = 42

VOCAB_SIZE     = 32000
MASK_SENTINEL  = VOCAB_SIZE   # 32000 — skipped by Rust cross-entropy kernel
EOT_ID         = 3            # <|endoftext|>

# Source mix ratios — must sum to 1.0
MIX = {
    "magicoder": 0.30,
    "metamath":  0.30,
    "xlam":      0.20,
    "hades":     0.20,
}

HADES_DB = "NL"   # database passed to `hades --db NL`


# ── Tokenizer + ChatML utilities ───────────────────────────────────────────────

def load_tokenizer(path: str) -> Tokenizer:
    tok = Tokenizer.from_file(path)
    assert tok.get_vocab_size() == VOCAB_SIZE, (
        f"Tokenizer vocab_size={tok.get_vocab_size()}, expected {VOCAB_SIZE}. "
        f"Use data/fineweb_edu/tokenizer.json."
    )
    return tok


def encode_chatml(
    tok: Tokenizer,
    turns: list[dict],
) -> tuple[list[int], list[int]]:
    """
    Encode a conversation in ChatML format with SFT loss masking.

    turns: [{"role": "system"|"user"|"assistant", "content": "..."}]

    Returns (tokens, targets) where:
      - tokens[i]  = input token at position i
      - targets[i] = MASK_SENTINEL  for positions that precede system/user content
                   = next token ID  for positions that precede assistant content

    The final token has no target and is dropped (standard next-token prediction).
    """
    full_ids: list[int] = []
    is_assistant: list[bool] = []   # True if this token is INSIDE an assistant turn

    for turn in turns:
        role    = turn["role"]
        content = turn["content"]

        # ChatML wrapping: <|im_start|>{role}\n{content}<|im_end|>\n
        header_ids  = tok.encode(f"<|im_start|>{role}\n").ids
        content_ids = tok.encode(content).ids
        footer_ids  = tok.encode("<|im_end|>\n").ids

        if role == "assistant":
            # Header (<|im_start|>assistant\n): masked
            full_ids.extend(header_ids)
            is_assistant.extend([False] * len(header_ids))
            # Content + footer: learnable
            full_ids.extend(content_ids)
            is_assistant.extend([True] * len(content_ids))
            full_ids.extend(footer_ids)
            is_assistant.extend([True] * len(footer_ids))
        else:
            # System / user: fully masked
            full_ids.extend(header_ids + content_ids + footer_ids)
            is_assistant.extend([False] * (len(header_ids) + len(content_ids) + len(footer_ids)))

    # Next-token prediction pairs: drop last token (no target)
    tokens  = full_ids[:-1]
    # Target at position i is full_ids[i+1]; mask if that position is non-assistant
    targets = [
        full_ids[i + 1] if is_assistant[i + 1] else MASK_SENTINEL
        for i in range(len(full_ids) - 1)
    ]

    return tokens, targets


# ── HADES query utilities ──────────────────────────────────────────────────────

def hades_aql(aql: str) -> list:
    """Execute an AQL query via `hades --db NL db aql`, return result list.

    hades returns: {"success": true, "data": {"results": [...], "count": N}}
    """
    result = subprocess.run(
        ["hades", "--db", HADES_DB, "db", "aql", aql],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hades db aql failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    return payload.get("data", {}).get("results", [])


def hades_query(text: str, limit: int = 10) -> list[dict]:
    """Semantic search via `hades --db NL db query`.

    Returns list of result dicts with keys: text, arxiv_id, title, similarity.
    hades envelope: {"data": {"results": [...], ...}}
    """
    result = subprocess.run(
        ["hades", "--db", HADES_DB, "db", "query", text, "--limit", str(limit)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hades db query failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    return payload.get("data", {}).get("results", [])


# ── Source: Magicoder (coding) ─────────────────────────────────────────────────

SYSTEM_CODING = (
    "You are an expert software engineer. Solve the coding task clearly and correctly. "
    "Provide working code with brief explanations of key design decisions."
)


def load_magicoder(tok: Tokenizer, budget: int, rng: random.Random) -> tuple[list[int], list[int]]:
    from datasets import load_dataset
    print("  [magicoder] Loading from HuggingFace...")
    ds       = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
    examples = list(ds)
    rng.shuffle(examples)

    all_tokens, all_targets = [], []
    for ex in examples:
        if len(all_tokens) >= budget:
            break
        instruction = ex.get("instruction", "").strip()
        response    = ex.get("response", "").strip()
        if not instruction or not response:
            continue

        t, tgt = encode_chatml(tok, [
            {"role": "system",    "content": SYSTEM_CODING},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ])
        all_tokens.extend(t)
        all_targets.extend(tgt)

    print(f"  [magicoder] {len(all_tokens):,} tokens from {len(examples):,} examples")
    return all_tokens[:budget], all_targets[:budget]


# ── Source: MetaMath (math reasoning) ─────────────────────────────────────────

SYSTEM_MATH = (
    "You are an expert mathematician. Solve problems step by step, showing all reasoning. "
    "Use clear notation and verify your answer at the end."
)


def load_metamath(tok: Tokenizer, budget: int, rng: random.Random) -> tuple[list[int], list[int]]:
    from datasets import load_dataset
    print("  [metamath] Loading from HuggingFace...")
    ds       = load_dataset("meta-math/MetaMathQA", split="train")
    examples = list(ds)
    rng.shuffle(examples)

    all_tokens, all_targets = [], []
    for ex in examples:
        if len(all_tokens) >= budget:
            break
        query    = ex.get("query", "").strip()
        response = ex.get("response", "").strip()
        if not query or not response:
            continue

        t, tgt = encode_chatml(tok, [
            {"role": "system",    "content": SYSTEM_MATH},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": response},
        ])
        all_tokens.extend(t)
        all_targets.extend(tgt)

    print(f"  [metamath] {len(all_tokens):,} tokens from {len(examples):,} examples")
    return all_tokens[:budget], all_targets[:budget]


# ── Source: XLAM (tool use) ────────────────────────────────────────────────────

SYSTEM_TOOL = """\
You are a helpful assistant with access to external tools. When you need to retrieve \
information or perform an action, call the appropriate tool using this format:

<tool_call>
{"name": "<tool_name>", "parameters": {<params>}}
</tool_call>

Wait for the <tool_result> before continuing your response."""


def load_xlam(tok: Tokenizer, budget: int, rng: random.Random) -> tuple[list[int], list[int]]:
    from datasets import load_dataset
    print("  [xlam] Loading from HuggingFace...")
    ds       = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    examples = list(ds)
    rng.shuffle(examples)

    all_tokens, all_targets = [], []
    for ex in examples:
        if len(all_tokens) >= budget:
            break
        query       = ex.get("query", "").strip()
        answers_raw = ex.get("answers", "[]")
        tools_raw   = ex.get("tools", "[]")

        if not query:
            continue
        try:
            answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
            tools   = json.loads(tools_raw)   if isinstance(tools_raw,   str) else tools_raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not answers:
            continue

        # Format each answer as a <tool_call> block
        call_blocks = []
        for a in answers:
            if not isinstance(a, dict):
                continue
            call_blocks.append(
                f'<tool_call>\n'
                f'{json.dumps({"name": a.get("name", ""), "parameters": a.get("arguments", {})}, indent=2)}\n'
                f'</tool_call>'
            )
        if not call_blocks:
            continue

        # Include tool schema in system prompt for grounding
        tool_schema = json.dumps(tools, indent=2) if tools else "[]"
        system = f"{SYSTEM_TOOL}\n\nAvailable tools:\n```json\n{tool_schema}\n```"

        t, tgt = encode_chatml(tok, [
            {"role": "system",    "content": system},
            {"role": "user",      "content": query},
            {"role": "assistant", "content": "\n\n".join(call_blocks)},
        ])
        all_tokens.extend(t)
        all_targets.extend(tgt)

    print(f"  [xlam] {len(all_tokens):,} tokens from {len(examples):,} examples")
    return all_tokens[:budget], all_targets[:budget]


# ── Source: HADES custom (knowledge graph tool use) ────────────────────────────

SYSTEM_HADES = """\
You are a research assistant with access to HADES, a semantic knowledge graph containing \
papers, code, equations, and research notes on Nested Learning (NL). Use HADES tools to \
answer questions accurately and to store information you want to remember later.

Available tools:
  hades_query      — semantic search over the knowledge base
  hades_db_aql     — run a structured AQL graph query (spec→equation, code→smell, etc.)

Tool call format:
<tool_call>
{"name": "<tool_name>", "parameters": {<params>}}
</tool_call>"""

PAPER_TITLES = {
    "2501.00663": "Titans",
    "2504.13173": "MIRAS",
    "2512.24695": "HOPE / Nested Learning",
    "2504.05646": "Lattice",
    "2505.23735": "Atlas",
    "2511.07343": "TNT",
    "2512.23852": "Trellis",
}

LOOKUP_QUESTIONS = [
    "What does the {paper} paper say about {topic}?",
    "Can you search the NL knowledge base for information about {topic}?",
    "I need to understand {topic} from the {paper} paper — look it up.",
    "Find relevant content about {topic} in the knowledge base.",
    "What is the NL research group's position on {topic}?",
    "Look up {topic} in HADES and summarize what you find.",
]


AQL_EXAMPLES = [
    # ── Spec → Equation traces ──────────────────────────────────────────────
    (
        "Which spec files implement the HOPE equations?",
        'FOR e IN nl_hecate_trace_edges FILTER e.rel == "implements" RETURN {spec: e._from, equation: e._to}',
        "I'll query the spec-to-equation trace graph.",
    ),
    (
        "What equations in the HOPE paper does the DGD spec implement?",
        'FOR e IN nl_hecate_trace_edges FILTER e._from == "hecate_specs/dgd" RETURN e._to',
        "I'll look up the trace edges from the DGD spec node.",
    ),
    (
        "Which equations does the Titans LMM spec trace to?",
        'FOR e IN nl_hecate_trace_edges FILTER e._from == "hecate_specs/titans_lmm" RETURN {equation: e._to, rel: e.rel}',
        "I'll traverse the trace edges from the titans_lmm spec.",
    ),
    (
        "Show me all spec files that cite the MIRAS paper.",
        'FOR s IN hecate_specs FILTER "2504.13173" IN s.paper_source RETURN {key: s._key, title: s.title, purpose: s.purpose}',
        "I'll search hecate_specs for specs that reference the MIRAS arxiv ID.",
    ),
    (
        "What equations from the Titans paper are referenced in the NL knowledge graph?",
        'FOR e IN nl_hecate_trace_edges FILTER CONTAINS(e._to, "titans_equations") RETURN DISTINCT e._to',
        "I'll find all Titans equation nodes that appear as targets in trace edges.",
    ),
    # ── Equation → paper chunk source traces ───────────────────────────────
    (
        "Which paper chunks are the source for the Titans gradient-as-matmul equation?",
        'FOR e IN nl_equation_source_edges FILTER e._from == "titans_equations/eq-017-gradient-as-matmul" RETURN {chunk: e._to, similarity: e.similarity}',
        "I'll traverse the equation source edges from that equation node.",
    ),
    (
        "Find the paper source chunks for HOPE equation 94.",
        'FOR e IN nl_equation_source_edges FILTER e._from == "hope_equations/eq-094-hope-forward" RETURN {chunk: e._to, similarity: e.similarity, name: e.equation_name}',
        "I'll look up the source chunks linked to that HOPE equation.",
    ),
    (
        "Which equations have the highest similarity to their source paper chunks?",
        'FOR e IN nl_equation_source_edges SORT e.similarity DESC LIMIT 5 RETURN {equation: e._from, chunk: e._to, similarity: e.similarity}',
        "I'll query equation source edges ordered by similarity score.",
    ),
    # ── Code → smell compliance traces ─────────────────────────────────────
    (
        "Which code files have compliance edges to CS-40?",
        'FOR e IN nl_smell_compliance_edges FILTER e._to == "nl_code_smells/smell-040-tape-opt-in" RETURN e._from',
        "Let me query the compliance edge graph for CS-40.",
    ),
    (
        "What code smells does the swiglu_forward.cu file comply with?",
        'FOR e IN nl_smell_compliance_edges FILTER e._from == "arxiv_metadata/swiglu-forward-cu" RETURN {smell: e._to, enforcement: e.enforcement}',
        "I'll traverse the compliance edges from the swiglu_forward.cu node.",
    ),
    (
        "Show all code files that comply with the no-DDP constraint (CS-34).",
        'FOR e IN nl_smell_compliance_edges FILTER CONTAINS(e._to, "smell-034") RETURN e._from',
        "I'll search compliance edges for the no-DDP smell node.",
    ),
    # ── Code smells ────────────────────────────────────────────────────────
    (
        "What code smells are classified as ontological constraints?",
        'FOR doc IN nl_code_smells FILTER doc.category == "Ontological" RETURN {id: doc.id, description: doc.description}',
        "Let me search the code smell graph for ontological constraints.",
    ),
    # ── Full chain: spec → equation → paper chunk ──────────────────────────
    (
        "Trace the DGD spec all the way to the paper text it implements.",
        (
            'LET eq_ids = (FOR e IN nl_hecate_trace_edges FILTER e._from == "hecate_specs/dgd" RETURN e._to) '
            'FOR eq_id IN eq_ids '
            '  FOR e2 IN nl_equation_source_edges FILTER e2._from == eq_id '
            '    FOR chunk IN arxiv_abstract_chunks FILTER chunk._id == e2._to '
            '    RETURN {equation: eq_id, chunk_id: chunk._id, text: LEFT(chunk.text, 300)}'
        ),
        "I'll do a multi-hop traversal: spec → trace edges → equation → equation source edges → paper chunk.",
    ),
    (
        "Find all papers connected to the Titans paper in the citation graph.",
        'FOR v IN 1..2 OUTBOUND "arxiv_metadata/2501.00663" paper_graph RETURN DISTINCT v._key',
        "I'll traverse the paper citation graph starting from Titans.",
    ),
]


def generate_hades_examples(
    tok: Tokenizer,
    budget: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    print("  [hades] Querying knowledge base via hades CLI...")

    # Pull real content from HADES using the hades CLI
    chunks, smells, specs = [], [], []
    try:
        chunks = hades_aql(
            "FOR doc IN arxiv_abstract_chunks"
            " FILTER LENGTH(doc.text) > 200"
            " SORT RAND() LIMIT 600"
            " RETURN {text: doc.text, arxiv_id: doc.arxiv_id, title: doc.title}"
        )
        smells = hades_aql(
            "FOR doc IN nl_code_smells"
            " RETURN {id: doc.id, description: doc.description, category: doc.category}"
        )
        specs = hades_aql(
            "FOR doc IN hecate_specs"
            " FILTER doc.purpose != null"
            " RETURN {key: doc._key, title: doc.title, purpose: doc.purpose, category: doc.category}"
        )
        print(f"  [hades] Pulled {len(chunks)} paper chunks, {len(smells)} code smells, {len(specs)} specs")
    except Exception as e:
        print(f"  [hades] WARNING: hades CLI unavailable ({e}). Using template-only examples.")

    all_tokens, all_targets = [], []

    def add(turns: list[dict]) -> None:
        t, tgt = encode_chatml(tok, turns)
        all_tokens.extend(t)
        all_targets.extend(tgt)

    # ── A. Grounded paper chunk lookup ──────────────────────────────────────
    for chunk in rng.sample(chunks, min(len(chunks), 400)):
        if len(all_tokens) >= budget:
            break
        text      = chunk.get("text", "").strip()
        arxiv_id  = chunk.get("arxiv_id", "")
        paper     = PAPER_TITLES.get(arxiv_id, "NL")
        topic     = text.split(".")[0][:80].strip()
        if not topic:
            continue

        question = rng.choice(LOOKUP_QUESTIONS).format(paper=paper, topic=topic.lower())
        result_snippet = text[:500]

        assistant = (
            f"I'll search the HADES knowledge base for this.\n\n"
            f"<tool_call>\n"
            f'{{"name": "hades_query", "parameters": {{"text": "{topic}", "database": "NL", "limit": 3}}}}\n'
            f"</tool_call>\n\n"
            f"<tool_result>\n{result_snippet}\n</tool_result>\n\n"
            f"Based on the {paper} paper: {text[:300].strip()}"
        )

        add([
            {"role": "system",    "content": SYSTEM_HADES},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant},
        ])

    # ── B. Code smell constraint checks (query-only) ───────────────────────
    smell_questions = [
        ("Is it okay to have a separate train/eval mode in the model?",
         "train eval mode distinction"),
        ("Can I use DDP directly for gradient synchronization?",
         "DDP distributed data parallel gradient sync"),
        ("Should the optimizer run outside the forward pass?",
         "optimizer external forward pass"),
        ("Is gradient checkpointing helpful for this architecture?",
         "gradient checkpointing memory"),
        ("Can I refer to CMS frequency levels as layers?",
         "layers levels terminology"),
        ("Does the forward pass need to be identical at build and inference time?",
         "forward pass identical build inference"),
        ("Is epoch-based training appropriate for this model?",
         "epoch training context stream"),
        ("Can I add a MemoryModule class to organize memory rules?",
         "MemoryModule class memory rule"),
        ("What does CS-32 say about the observe-then-advance pattern?",
         "observe then advance CS-32"),
        ("Should I use bf16 inside the memory inner loop?",
         "fp32 bf16 inner loop precision memory"),
    ]

    for question, search_term in smell_questions * max(1, budget // (len(smell_questions) * 800)):
        if len(all_tokens) >= budget:
            break

        relevant = [s for s in smells if search_term.split()[0].lower() in
                    s.get("description", "").lower()]
        smell_snippet = relevant[0].get("description", "")[:400] if relevant else \
                        (smells[0].get("description", "")[:400] if smells else
                         "Refer to the code smell specifications in nl_code_smells.")

        assistant = (
            f"Let me check the NL-Hecate code smell constraints.\n\n"
            f"<tool_call>\n"
            f'{{"name": "hades_query", "parameters": {{"text": "{search_term}", "database": "NL", "limit": 3}}}}\n'
            f"</tool_call>\n\n"
            f"<tool_result>\n{smell_snippet}\n</tool_result>\n\n"
            f"According to the NL-Hecate constraints: {smell_snippet[:250]}"
        )

        add([
            {"role": "system",    "content": SYSTEM_HADES},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant},
        ])

    # ── C. Spec lookups — trace spec → purpose + equations ─────────────────
    spec_questions = [
        "What does the {title} spec do?",
        "Describe the purpose of {title} in the NL codebase.",
        "What equations does the {title} spec implement?",
        "Look up {title} in the spec graph and tell me what it traces to.",
    ]
    for spec in rng.sample(specs, min(len(specs), 200)) * max(1, budget // (max(len(specs), 1) * 600)):
        if len(all_tokens) >= budget:
            break
        title   = spec.get("title", "")
        key     = spec.get("key", "")
        purpose = spec.get("purpose", "")[:300]
        if not title or not key:
            continue
        question = rng.choice(spec_questions).format(title=title)
        aql_trace = f'FOR e IN nl_hecate_trace_edges FILTER e._from == "hecate_specs/{key}" RETURN {{rel: e.rel, target: e._to}}'
        assistant = (
            f"I'll look up the spec node and its trace edges.\n\n"
            f"<tool_call>\n"
            f'{{"name": "hades_db_aql", "parameters": {{"query": "{aql_trace}", "database": "NL"}}}}\n'
            f"</tool_call>\n\n"
            f"<tool_result>\n[Trace edges for {key}]\n</tool_result>\n\n"
            f"The **{title}** spec: {purpose}"
        )
        add([
            {"role": "system",    "content": SYSTEM_HADES},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant},
        ])

    # ── D. Graph traversal — AQL (spec→eq, eq→chunk, code→smell) ───────────
    repeat = max(1, budget // (len(AQL_EXAMPLES) * 400))
    for question, aql, thought in AQL_EXAMPLES * repeat:
        if len(all_tokens) >= budget:
            break

        # Escape aql for JSON embedding (remove literal newlines)
        aql_inline = aql.replace("\n", " ").replace('"', '\\"') if isinstance(aql, str) else aql

        assistant = (
            f"{thought}\n\n"
            f"<tool_call>\n"
            f'{{"name": "hades_db_aql", "parameters": {{"query": "{aql_inline}", "database": "NL"}}}}\n'
            f"</tool_call>\n\n"
            f"<tool_result>\n[Graph query results returned]\n</tool_result>\n\n"
            f"The knowledge graph traversal returned the relevant nodes and edges, "
            f"showing the direct relationships in the NL research graph."
        )

        add([
            {"role": "system",    "content": SYSTEM_HADES},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant},
        ])

    print(f"  [hades] {len(all_tokens):,} tokens generated")
    return all_tokens[:budget], all_targets[:budget]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 2 SFT dataset for NL-Hecate")
    parser.add_argument("--target_tokens",   type=int,   default=TARGET_TOKENS)
    parser.add_argument("--output",          type=str,   default=OUTPUT_DIR)
    parser.add_argument("--tokenizer",       type=str,   default=TOKENIZER_PATH)
    parser.add_argument("--seed",            type=int,   default=SEED)
    parser.add_argument("--val_ratio",       type=float, default=VAL_RATIO)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = load_tokenizer(args.tokenizer)

    # Per-source token budgets
    budgets = {src: int(args.target_tokens * ratio) for src, ratio in MIX.items()}

    print(f"\nPhase 2 SFT data prep — {args.target_tokens:,} tokens target")
    print(f"  Mix:     { {k: f'{v*100:.0f}%' for k,v in MIX.items()} }")
    print(f"  Budgets: { {k: f'{v:,}' for k,v in budgets.items()} }\n")

    t0 = time.time()

    mag_tok,  mag_tgt  = load_magicoder(tok, budgets["magicoder"], random.Random(args.seed))
    math_tok, math_tgt = load_metamath( tok, budgets["metamath"],  random.Random(args.seed + 1))
    xl_tok,   xl_tgt   = load_xlam(     tok, budgets["xlam"],      random.Random(args.seed + 2))
    hd_tok,   hd_tgt   = generate_hades_examples(
        tok, budgets["hades"], random.Random(args.seed + 3))

    # Combine and shuffle in seq_len=512 blocks to preserve conversational coherence
    print("\nMixing and shuffling...")
    all_tok = mag_tok + math_tok + xl_tok + hd_tok
    all_tgt = mag_tgt + math_tgt + xl_tgt + hd_tgt

    chunk_size  = 512
    n_chunks    = len(all_tok) // chunk_size
    chunk_order = list(range(n_chunks))
    rng.shuffle(chunk_order)

    tok_shuf, tgt_shuf = [], []
    for idx in chunk_order:
        s = idx * chunk_size
        tok_shuf.extend(all_tok[s:s + chunk_size])
        tgt_shuf.extend(all_tgt[s:s + chunk_size])

    total   = len(tok_shuf)
    n_val   = max(512, int(total * args.val_ratio))
    n_train = total - n_val

    train_tokens  = np.array(tok_shuf[:n_train], dtype=np.uint32)
    train_targets = np.array(tgt_shuf[:n_train], dtype=np.int32)
    val_tokens    = np.array(tok_shuf[n_train:],  dtype=np.uint32)
    val_targets   = np.array(tgt_shuf[n_train:],  dtype=np.int32)

    n_masked_train = int((train_targets == MASK_SENTINEL).sum())
    n_masked_val   = int((val_targets   == MASK_SENTINEL).sum())
    mask_ratio     = n_masked_train / max(1, n_train)

    print(f"Saving to {out_dir}...")
    np.save(out_dir / "train_tokens.npy",  train_tokens)
    np.save(out_dir / "train_targets.npy", train_targets)
    np.save(out_dir / "val_tokens.npy",    val_tokens)
    np.save(out_dir / "val_targets.npy",   val_targets)

    meta = {
        "vocab_size":      VOCAB_SIZE,
        "tokenizer":       str(Path(args.tokenizer).resolve()),
        "mask_sentinel":   MASK_SENTINEL,
        "special_tokens":  {"<|im_start|>": 0, "<|im_end|>": 1, "<|pad|>": 2, "<|endoftext|>": 3},
        "format":          "chatml_sft",
        "tool_format":     "xml_tool_call",
        "phase":           2,
        "train": {
            "split":          "train",
            "total_tokens":   int(n_train),
            "valid_targets":  int(n_train - n_masked_train),
            "masked_targets": int(n_masked_train),
            "mask_ratio":     round(mask_ratio, 4),
        },
        "val": {
            "split":          "val",
            "total_tokens":   int(n_val),
            "valid_targets":  int(n_val - n_masked_val),
            "masked_targets": int(n_masked_val),
            "mask_ratio":     round(n_masked_val / max(1, n_val), 4),
        },
        "sources": {
            src: {"tokens": len(t), "budget": budgets[src], "ratio": ratio}
            for src, ratio, t in [
                ("magicoder", MIX["magicoder"], mag_tok),
                ("metamath",  MIX["metamath"],  math_tok),
                ("xlam",      MIX["xlam"],      xl_tok),
                ("hades",     MIX["hades"],     hd_tok),
            ]
        },
        "seed":      args.seed,
        "val_ratio": args.val_ratio,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print("Phase 2 SFT data preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output:         {out_dir}")
    print(f"  Total tokens:   {total:,}")
    print(f"  Train tokens:   {n_train:,}")
    print(f"    Learnable:    {n_train - n_masked_train:,}  ({1 - mask_ratio:.1%} of train)")
    print(f"    Masked:       {n_masked_train:,}  ({mask_ratio:.1%} of train)")
    print(f"  Val tokens:     {n_val:,}")
    print(f"  Sources:")
    print(f"    magicoder:    {len(mag_tok):,}")
    print(f"    metamath:     {len(math_tok):,}")
    print(f"    xlam:         {len(xl_tok):,}")
    print(f"    hades:        {len(hd_tok):,}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
