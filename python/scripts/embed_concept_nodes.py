#!/usr/bin/env python3
"""
Embed all concept nodes in the NL database inline (embedding field on each doc).

Calls the HADES embedder service over /run/hades/embedder.sock and patches
documents in ArangoDB at /run/metis/readwrite/arangod.sock.

Idempotent: skips docs that already have an embedding field.
GPU: embedder service is pinned to GPU 2 via /etc/hades/embedder.conf.

Usage:
    python scripts/embed_concept_nodes.py [--dry-run] [--collection <name>]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import sys
import time
import http.client
from typing import Any


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARANGO_SOCK  = "/run/metis/readwrite/arangod.sock"
EMBEDDER_SOCK = "/run/hades/embedder.sock"
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
DATABASE = "NL"
EMBED_TASK = "retrieval.passage"
BATCH_SIZE = 8        # texts per embedder call (RTX 2000 Ada: ~5GB free after model load)
EMBED_DIM  = 2048


# ---------------------------------------------------------------------------
# Collection config: (text_fields in priority order, join strategy)
#   "join"  — concatenate all present fields with ". "
#   "first" — use the first non-empty field only
# ---------------------------------------------------------------------------

def _eq_fields() -> list[str]:
    return ["name", "description", "latex"]

def _abs_fields() -> list[str]:
    return ["name", "title", "description"]

def _axiom_fields() -> list[str]:
    # hope_axioms: name + description
    # tnt/lattice/trellis axioms: _key-derived name + principles list
    return ["name", "description", "assumption", "principles"]

COLLECTION_CONFIG: dict[str, dict[str, Any]] = {
    # Code smells — embed_text is the canonical field when present
    "nl_code_smells":       {"fields": ["embed_text", "name", "description"],         "strategy": "join"},
    # Equations — name + description + LaTeX gives the richest signal
    "hope_equations":       {"fields": _eq_fields(), "strategy": "join"},
    "miras_equations":      {"fields": _eq_fields(), "strategy": "join"},
    "titans_equations":     {"fields": _eq_fields(), "strategy": "join"},
    "atlas_equations":      {"fields": _eq_fields(), "strategy": "join"},
    "tnt_equations":        {"fields": _eq_fields(), "strategy": "join"},
    "lattice_equations":    {"fields": _eq_fields(), "strategy": "join"},
    "trellis_equations":    {"fields": _eq_fields(), "strategy": "join"},
    # Abstractions
    "hope_abstractions":    {"fields": _abs_fields(), "strategy": "join"},
    "miras_abstractions":   {"fields": _abs_fields(), "strategy": "join"},
    "titans_abstractions":  {"fields": _abs_fields(), "strategy": "join"},
    "atlas_abstractions":   {"fields": _abs_fields(), "strategy": "join"},
    "tnt_abstractions":     {"fields": _abs_fields(), "strategy": "join"},
    "lattice_abstractions": {"fields": _abs_fields(), "strategy": "join"},
    "trellis_abstractions": {"fields": _abs_fields(), "strategy": "join"},
    # Axioms
    "hope_axioms":          {"fields": _axiom_fields(), "strategy": "join"},
    "miras_axioms":         {"fields": _axiom_fields(), "strategy": "join"},
    "titans_axioms":        {"fields": _axiom_fields(), "strategy": "join"},
    "atlas_axioms":         {"fields": _axiom_fields(), "strategy": "join"},
    "tnt_axioms":           {"fields": _axiom_fields(), "strategy": "join"},
    "lattice_axioms":       {"fields": _axiom_fields(), "strategy": "join"},
    "trellis_axioms":       {"fields": _axiom_fields(), "strategy": "join"},
    # Definitions
    "hope_definitions":     {"fields": ["name", "description", "latex"], "strategy": "join"},
    "miras_definitions":    {"fields": ["name", "description", "latex"], "strategy": "join"},
    "titans_definitions":   {"fields": ["name", "description", "latex"], "strategy": "join"},
    # Specs
    "hecate_specs":         {"fields": ["title", "purpose"],             "strategy": "join"},
    # Reframings — prefer the NL-specific framing text
    "nl_reframings":        {"fields": ["name", "nl_reframe", "verbatim_definition", "description", "implication"], "strategy": "join"},
    "hope_nl_reframings":   {"fields": ["name", "nl_reframe", "verbatim_definition", "description"],               "strategy": "join"},
    # Optimizers
    "nl_optimizers":        {"fields": ["name", "description", "limitations", "objective"], "strategy": "join"},
    # Ethnographic notes — note is the primary field
    "nl_ethnographic_notes":{"fields": ["note", "observation", "insight", "context"],      "strategy": "join"},
    # Algorithms, assumptions, probes
    "hope_algorithms":      {"fields": ["name", "description", "pseudocode"],              "strategy": "join"},
    "hope_assumptions":     {"fields": ["name", "assumption", "description"],              "strategy": "join"},
    "nl_probe_patterns":    {"fields": ["name", "description"],                            "strategy": "join"},
}


# ---------------------------------------------------------------------------
# Unix socket HTTP helpers
# ---------------------------------------------------------------------------

class UnixSocketHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.socket_path)
        self.sock = s


def _auth_headers() -> dict[str, str]:
    creds = base64.b64encode(f"root:{ARANGO_PASSWORD}".encode()).decode()
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {creds}",
    }


def arango_request(method: str, path: str, body: Any = None) -> tuple[int, Any]:
    conn = UnixSocketHTTPConnection(ARANGO_SOCK)
    full_path = f"/_db/{DATABASE}{path}"
    body_bytes = json.dumps(body).encode() if body is not None else None
    conn.request(method, full_path, body=body_bytes, headers=_auth_headers())
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data)


def embedder_request(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    conn = UnixSocketHTTPConnection(EMBEDDER_SOCK)
    payload = {"texts": texts, "task": EMBED_TASK}
    if batch_size is not None:
        payload["batch_size"] = batch_size
    body_bytes = json.dumps(payload).encode()
    conn.request("POST", "/embed", body=body_bytes, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"Embedder returned {resp.status}: {data}")
    return data["embeddings"]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def build_embed_text(doc: dict, fields: list[str], strategy: str) -> str:
    parts = []
    for field in fields:
        val = doc.get(field)
        if not val:
            continue
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, list):
            # Flatten list-of-strings (e.g. tnt_axioms.principles)
            items = [str(v).strip() for v in val if isinstance(v, str) and str(v).strip()]
            if items:
                parts.append(". ".join(items))

    if strategy == "first":
        return parts[0] if parts else ""

    # "join" — concatenate all present
    text = ". ".join(parts)

    # Fallback: join all top-level string/list values, then use _key as label
    if not text:
        fallback = []
        for k, v in doc.items():
            if k.startswith("_"):
                continue  # skip _key, _id, _rev
            if isinstance(v, str) and v.strip():
                fallback.append(v.strip())
            elif isinstance(v, list):
                items = [str(i).strip() for i in v if isinstance(i, str) and str(i).strip()]
                if items:
                    fallback.append(". ".join(items))
        if not fallback:
            # Last resort: humanise the _key
            fallback = [doc.get("_key", "").replace("-", " ")]
        text = ". ".join(fallback[:5])

    return text


# ---------------------------------------------------------------------------
# Fetch / patch helpers
# ---------------------------------------------------------------------------

def fetch_unembedded(collection: str) -> list[dict]:
    """Return all docs in collection that have no 'embedding' field."""
    aql = f"FOR d IN {collection} FILTER !HAS(d, 'embedding') RETURN d"
    status, result = arango_request("POST", "/_api/cursor",
                                    body={"query": aql, "batchSize": 500, "count": True})
    if status not in (200, 201):
        print(f"  [WARN] AQL failed for {collection}: {status} {result.get('errorMessage','')}")
        return []
    docs = result.get("result", [])
    # Handle cursor pagination
    while result.get("hasMore"):
        cursor_id = result["id"]
        status, result = arango_request("PUT", f"/_api/cursor/{cursor_id}")
        if status not in (200, 201):
            break
        docs.extend(result.get("result", []))
    return docs


def patch_embedding(collection: str, key: str, embedding: list[float]) -> bool:
    """PATCH a single document with its embedding."""
    status, _ = arango_request(
        "PATCH",
        f"/_api/document/{collection}/{key}",
        body={"embedding": embedding},
    )
    return status in (200, 201, 202)


# ---------------------------------------------------------------------------
# Embed one collection
# ---------------------------------------------------------------------------

def embed_collection(collection: str, config: dict, dry_run: bool) -> tuple[int, int]:
    """Returns (embedded, skipped_errors)."""
    fields   = config["fields"]
    strategy = config["strategy"]

    docs = fetch_unembedded(collection)
    total = len(docs)
    if total == 0:
        print(f"  {collection}: already complete (0 unembedded)")
        return 0, 0

    print(f"  {collection}: {total} docs to embed", flush=True)
    if dry_run:
        # Show sample text for first doc
        sample = build_embed_text(docs[0], fields, strategy)
        print(f"    [dry-run] sample text: {sample[:120]!r}")
        return 0, 0

    embedded = 0
    errors   = 0
    t0 = time.time()

    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch_docs = docs[i : i + BATCH_SIZE]
        texts = [build_embed_text(d, fields, strategy) for d in batch_docs]

        # Skip docs with no extractable text
        valid_indices = [j for j, t in enumerate(texts) if t.strip()]
        if not valid_indices:
            errors += len(batch_docs)
            continue

        valid_texts = [texts[j] for j in valid_indices]
        try:
            vectors = embedder_request(valid_texts, batch_size=BATCH_SIZE)
        except RuntimeError as e:
            print(f"\n  [ERROR] embedding batch {i//BATCH_SIZE}: {e}")
            errors += len(valid_indices)
            continue

        # Verify dimension
        for vj, vec in zip(valid_indices, vectors):
            if len(vec) != EMBED_DIM:
                print(f"\n  [WARN] unexpected embedding dim {len(vec)} for {batch_docs[vj]['_key']}")
                errors += 1
                continue
            ok = patch_embedding(collection, batch_docs[vj]["_key"], vec)
            if ok:
                embedded += 1
            else:
                errors += 1

        elapsed = time.time() - t0
        rate = embedded / elapsed if elapsed > 0 else 0
        print(f"    {min(i + BATCH_SIZE, total)}/{total} "
              f"({embedded} embedded, {errors} errors, {rate:.1f} docs/s)",
              end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"    Done: {embedded} embedded, {errors} errors in {elapsed:.1f}s{' ' * 20}")
    return embedded, errors


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_coverage() -> None:
    print("\n=== Coverage verification ===")
    total_missing = 0
    for col in COLLECTION_CONFIG:
        aql = f"RETURN LENGTH(FOR d IN {col} FILTER !HAS(d,'embedding') RETURN 1)"
        status, result = arango_request("POST", "/_api/cursor", body={"query": aql})
        missing = result.get("result", [None])[0] if status in (200, 201) else "ERR"
        flag = " ✓" if missing == 0 else f" ← {missing} missing"
        print(f"  {col}: {flag}")
        if isinstance(missing, int):
            total_missing += missing
    print(f"\nTotal missing: {total_missing}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed concept nodes into NL database")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch counts and show sample texts, don't write anything")
    parser.add_argument("--collection", metavar="NAME",
                        help="Process only this collection")
    parser.add_argument("--verify", action="store_true",
                        help="Only run coverage verification, no embedding")
    args = parser.parse_args()

    # Verify ArangoDB connection
    status, result = arango_request("GET", "/_api/database/current")
    if status != 200:
        print(f"ERROR: Cannot connect to ArangoDB: {status} {result}")
        sys.exit(1)
    print(f"ArangoDB: connected to {result['result']['name']}")

    # Verify embedder health
    conn = UnixSocketHTTPConnection(EMBEDDER_SOCK)
    conn.request("GET", "/health", headers={"Accept": "application/json"})
    resp = conn.getresponse()
    health = json.loads(resp.read())
    conn.close()
    if health.get("status") not in ("ready", "idle"):
        print(f"ERROR: Embedder service not ready: {health.get('status')}")
        sys.exit(1)
    print(f"Embedder: {health['status']} on {health.get('device','?')} "
          f"(model: {health.get('model_name','?')})")

    if args.verify:
        verify_coverage()
        return

    collections = (
        {args.collection: COLLECTION_CONFIG[args.collection]}
        if args.collection
        else COLLECTION_CONFIG
    )

    if args.collection and args.collection not in COLLECTION_CONFIG:
        print(f"ERROR: unknown collection '{args.collection}'. "
              f"Known: {', '.join(COLLECTION_CONFIG)}")
        sys.exit(1)

    print(f"\nEmbedding {len(collections)} collection(s) "
          f"{'[DRY RUN] ' if args.dry_run else ''}"
          f"batch_size={BATCH_SIZE} dim={EMBED_DIM}\n")

    total_embedded = 0
    total_errors   = 0
    t_start = time.time()

    for col, cfg in collections.items():
        embedded, errors = embed_collection(col, cfg, dry_run=args.dry_run)
        total_embedded += embedded
        total_errors   += errors

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total: {total_embedded} embedded, {total_errors} errors in {elapsed:.1f}s")

    if not args.dry_run and total_embedded > 0:
        verify_coverage()


if __name__ == "__main__":
    main()
