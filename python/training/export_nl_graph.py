"""
export_nl_graph.py -- Export NL knowledge graph from ArangoDB to PyG HeteroData.

Reads the hecate_knowledge_graph from ArangoDB and produces:
  data/nl_graph.pt           -- serialized PyG HeteroData (torch.save)
  data/nl_graph_node_maps.json -- {collection: {arango_key: int_index}} for round-tripping

Node feature strategy:
  - Concept nodes (equations, axioms, smells, specs, etc.): inline 'embedding' field
    set by python/scripts/embed_concept_nodes.py (2048d)
  - arxiv_metadata and arxiv_abstract_chunks: embeddings are stored in the
    arxiv_abstract_embeddings collection keyed by doc_key; we aggregate per document.
  - Missing embeddings raise ValueError (not silent zeros).

Edge index construction:
  - Each edge collection is split by (from_collection, to_collection) endpoint types.
  - Only edges whose both endpoints appear in our node maps are included.
  - nl_smell_compliance_edges gets train/val/test boolean masks.

Usage:
  python python/training/export_nl_graph.py --database NL --output data/nl_graph.pt --seed 42
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import socket
import sys
import time
import http.client
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# ArangoDB connection (Unix socket, same pattern as embed_concept_nodes.py)
# ---------------------------------------------------------------------------

ARANGO_SOCK = "/run/metis/readwrite/arangod.sock"
ARANGO_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
EMBED_DIM = 2048


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


def arango_request(database: str, method: str, path: str, body=None):
    conn = UnixSocketHTTPConnection(ARANGO_SOCK)
    full_path = f"/_db/{database}{path}"
    body_bytes = json.dumps(body).encode() if body is not None else None
    conn.request(method, full_path, body=body_bytes, headers=_auth_headers())
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, json.loads(data)


def aql(database: str, query: str, bind_vars: dict | None = None) -> list:
    """Run an AQL query and return all results (handles cursor pagination)."""
    body: dict = {"query": query, "batchSize": 10000}
    if bind_vars:
        body["bindVars"] = bind_vars
    status, result = arango_request(database, "POST", "/_api/cursor", body)
    if status not in (200, 201):
        raise RuntimeError(f"AQL failed ({status}): {result.get('errorMessage', result)}")
    docs = result.get("result", [])
    while result.get("hasMore"):
        cursor_id = result["id"]
        status, result = arango_request(database, "PUT", f"/_api/cursor/{cursor_id}")
        if status not in (200, 201):
            raise RuntimeError(
                f"AQL cursor pagination failed ({status}) for query: {query[:120]!r}: "
                f"{result.get('errorMessage', '')}"
            )
        docs.extend(result.get("result", []))
    return docs


# ---------------------------------------------------------------------------
# Graph definition: collections to export
# ---------------------------------------------------------------------------

# All vertex collections to include. Must match NL_NODE_TYPES in rgcn_model.py.
VERTEX_COLLECTIONS = [
    "arxiv_metadata",
    "nl_code_smells",
    "hecate_specs",
    "hope_equations",
    "miras_equations",
    "titans_equations",
    "atlas_equations",
    "tnt_equations",
    "lattice_equations",
    "trellis_equations",
    "hope_abstractions",
    "miras_abstractions",
    "titans_abstractions",
    "atlas_abstractions",
    "tnt_abstractions",
    "lattice_abstractions",
    "trellis_abstractions",
    "hope_axioms",
    "miras_axioms",
    "titans_axioms",
    "atlas_axioms",
    "tnt_axioms",
    "lattice_axioms",
    "trellis_axioms",
    "hope_definitions",
    "miras_definitions",
    "titans_definitions",
    "nl_axioms",
    "nl_reframings",
    "hope_nl_reframings",
    "nl_ethnographic_notes",
    "nl_optimizers",
    "nl_probe_patterns",
    "arxiv_abstract_chunks",
]

# Edge collections to include (prediction target nl_smell_compliance_edges is last).
EDGE_COLLECTIONS = [
    "nl_hecate_trace_edges",
    "nl_signature_equation_edges",
    "nl_axiom_basis_edges",
    "nl_definition_source_edges",
    "nl_structural_embodiment_edges",
    "nl_validated_against_edges",
    "nl_equation_depends_edges",
    "nl_lineage_chain_edges",
    "nl_migration_edges",
    "nl_axiom_inherits_edges",
    "nl_equation_source_edges",
    "nl_smell_source_edges",
    "nl_smell_compliance_edges",  # prediction target -- gets train/val/test masks
]

# Collections that use inline 'embedding' field (set by embed_concept_nodes.py)
INLINE_EMBEDDING_COLLECTIONS = {
    "nl_code_smells", "hecate_specs",
    "hope_equations", "miras_equations", "titans_equations", "atlas_equations",
    "tnt_equations", "lattice_equations", "trellis_equations",
    "hope_abstractions", "miras_abstractions", "titans_abstractions", "atlas_abstractions",
    "tnt_abstractions", "lattice_abstractions", "trellis_abstractions",
    "hope_axioms", "miras_axioms", "titans_axioms", "atlas_axioms",
    "tnt_axioms", "lattice_axioms", "trellis_axioms",
    "hope_definitions", "miras_definitions", "titans_definitions",
    "nl_axioms", "nl_reframings", "hope_nl_reframings",
    "nl_ethnographic_notes", "nl_optimizers", "nl_probe_patterns",
}

# Collections that need embedding from arxiv_abstract_embeddings (by doc_key)
CHUNK_EMBEDDING_COLLECTIONS = {"arxiv_metadata", "arxiv_abstract_chunks"}

TARGET_EDGE_COLLECTION = "nl_smell_compliance_edges"


# ---------------------------------------------------------------------------
# Node loading
# ---------------------------------------------------------------------------

def load_vertex_collection(
    database: str,
    collection: str,
    verbose: bool = True,
) -> tuple[list[str], torch.Tensor]:
    """
    Load all nodes from a vertex collection.

    Returns:
        keys: list of ArangoDB _key strings (defines index ordering)
        features: float32 tensor [N, 2048]
    """
    if verbose:
        print(f"  Loading {collection}...", end="", flush=True)

    if collection in INLINE_EMBEDDING_COLLECTIONS:
        docs = aql(database, f"FOR d IN {collection} RETURN {{k: d._key, e: d.embedding}}")
        keys = []
        embeddings = []
        missing = []
        for doc in docs:
            k = doc["k"]
            e = doc["e"]
            if e is None:
                missing.append(k)
            else:
                keys.append(k)
                embeddings.append(e)
        if missing:
            raise ValueError(
                f"Missing embeddings in {collection} for keys: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} ({len(missing)} total). "
                f"Run python/scripts/embed_concept_nodes.py first."
            )
        if not keys:
            if verbose:
                print(f" empty")
            return [], torch.zeros(0, EMBED_DIM)
        feat = torch.tensor(embeddings, dtype=torch.float32)

    elif collection in CHUNK_EMBEDDING_COLLECTIONS:
        # Get all doc keys in the collection first
        docs = aql(database, f"FOR d IN {collection} RETURN d._key")
        if not docs:
            if verbose:
                print(f" empty")
            return [], torch.zeros(0, EMBED_DIM)
        keys = docs  # list of _key strings

        # Fetch embeddings from arxiv_abstract_embeddings (keyed by chunk _key or doc_key)
        # arxiv_abstract_chunks: _key is the chunk key, each has doc_key pointing to arxiv_metadata
        # arxiv_metadata: aggregate mean over all chunks with matching doc_key
        # Embedding key convention (HADES ingest pipeline):
        #   chunk _key = "{doc_id}_chunk_{N}"
        #   embedding _key = "{doc_id}_chunk_{N}_emb"
        # Neither chunks nor embeddings have a doc_key field — the relationship
        # is encoded in the key prefix.

        if collection == "arxiv_abstract_chunks":
            # Direct lookup: embedding _key = chunk _key + "_emb"
            emb_keys = [k + "_emb" for k in keys]
            emb_docs = aql(
                database,
                "FOR e IN arxiv_abstract_embeddings "
                "FILTER e._key IN @emb_keys "
                "RETURN {ek: e._key, emb: e.embedding}",
                bind_vars={"emb_keys": emb_keys},
            )
            emb_map = {d["ek"]: d["emb"] for d in emb_docs}
            embeddings = []
            valid_keys = []
            missing = []
            for k in keys:
                emb = emb_map.get(k + "_emb")
                if emb is None:
                    missing.append(k)
                else:
                    embeddings.append(emb)
                    valid_keys.append(k)
            if missing:
                print(
                    f"\n  [WARN] {collection}: {len(missing)} chunks missing embeddings "
                    f"(will be skipped). Example: {missing[:3]}"
                )
            keys = valid_keys
            if not keys:
                if verbose:
                    print(f" no embeddings found")
                return [], torch.zeros(0, EMBED_DIM)
            feat = torch.tensor(embeddings, dtype=torch.float32)

        else:  # arxiv_metadata
            # Aggregate chunk embeddings for each document via mean pooling.
            # For a document with key "adamw-rs", its chunk embedding keys are:
            #   "adamw-rs_chunk_0_emb", "adamw-rs_chunk_1_emb", ...
            # Fetch all embedding docs whose _key starts with "{doc_key}_chunk_"
            # by retrieving all embeddings and filtering by prefix.
            # This is efficient for 58 docs × avg 5 chunks = ~290 lookups.
            all_embs = aql(
                database,
                "FOR e IN arxiv_abstract_embeddings "
                "RETURN {ek: e._key, emb: e.embedding}",
            )
            # Build a map: doc_key -> list of chunk embeddings
            doc_emb_map: dict[str, list[list[float]]] = defaultdict(list)
            for e in all_embs:
                ek = e["ek"]
                # Parse "doc_key_chunk_N_emb" -> "doc_key"
                # Find the last occurrence of "_chunk_" to handle doc_keys with underscores
                marker = "_chunk_"
                idx = ek.rfind(marker)
                if idx == -1:
                    continue
                doc_key = ek[:idx]
                if e["emb"] is not None:
                    doc_emb_map[doc_key].append(e["emb"])

            embeddings = []
            valid_keys = []
            missing = []
            for k in keys:
                chunks = doc_emb_map.get(k, [])
                if not chunks:
                    missing.append(k)
                else:
                    emb = torch.tensor(chunks, dtype=torch.float32).mean(dim=0)
                    embeddings.append(emb)
                    valid_keys.append(k)
            if missing:
                print(
                    f"\n  [WARN] {collection}: {len(missing)} docs missing chunk embeddings "
                    f"(will be skipped). Example: {missing[:3]}"
                )
            keys = valid_keys
            if not keys:
                if verbose:
                    print(f" no embeddings found")
                return [], torch.zeros(0, EMBED_DIM)
            feat = torch.stack(embeddings, dim=0)  # [N, 2048]
    else:
        raise ValueError(f"Unknown embedding strategy for collection: {collection}")

    if verbose:
        print(f" {len(keys)} nodes, shape {tuple(feat.shape)}")

    # Validate
    if feat.shape[1] != EMBED_DIM:
        raise ValueError(f"{collection}: expected {EMBED_DIM}d, got {feat.shape[1]}d")
    if torch.isnan(feat).any() or torch.isinf(feat).any():
        raise ValueError(f"{collection}: NaN or Inf in embeddings")

    return keys, feat


# ---------------------------------------------------------------------------
# Edge loading
# ---------------------------------------------------------------------------

def load_edge_collection(
    database: str,
    collection: str,
    node_maps: dict[str, dict[str, int]],
    verbose: bool = True,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """
    Load all edges from an edge collection and return as a dict of
    {(src_collection, edge_collection, dst_collection): edge_index [2, E]}.

    Edges are split by (from_type, to_type) endpoint because PyG HeteroData
    requires a separate edge_index per (src_type, rel, dst_type) triplet.
    Edges whose endpoints aren't in node_maps are silently skipped.
    """
    if verbose:
        print(f"  Loading {collection}...", end="", flush=True)

    docs = aql(database, f"FOR e IN {collection} RETURN {{f: e._from, t: e._to}}")

    # Group by (from_collection, to_collection)
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    skipped = 0

    for doc in docs:
        from_full = doc["f"]   # e.g. "hope_equations/eq-001"
        to_full = doc["t"]     # e.g. "arxiv_abstract_chunks/chunk-abc"
        from_col = from_full.split("/")[0]
        from_key = from_full.split("/")[1]
        to_col = to_full.split("/")[0]
        to_key = to_full.split("/")[1]

        # Look up local node indices
        from_map = node_maps.get(from_col, {})
        to_map = node_maps.get(to_col, {})
        if from_key not in from_map or to_key not in to_map:
            skipped += 1
            continue

        groups[(from_col, to_col)].append((from_map[from_key], to_map[to_key]))

    result: dict[tuple[str, str, str], torch.Tensor] = {}
    for (from_col, to_col), pairs in groups.items():
        key = (from_col, collection, to_col)
        srcs = [p[0] for p in pairs]
        dsts = [p[1] for p in pairs]
        result[key] = torch.tensor([srcs, dsts], dtype=torch.long)

    total = sum(ei.size(1) for ei in result.values())
    if verbose:
        print(f" {len(docs)} docs -> {total} edges ({skipped} skipped), "
              f"{len(result)} triplets")

    return result


# ---------------------------------------------------------------------------
# Train / val / test masks for the prediction target
# ---------------------------------------------------------------------------

def make_compliance_masks(
    num_edges: int,
    seed: int = 42,
    train_ratio: float = 0.83,
    val_ratio: float = 0.085,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (train_mask, val_mask, test_mask) boolean tensors of length num_edges.
    Splits: ~83% train, ~8.5% val, ~8.5% test (rounds to nearest int).
    """
    rng = random.Random(seed)
    perm = list(range(num_edges))
    rng.shuffle(perm)
    n_train = int(num_edges * train_ratio)
    n_val = int(num_edges * val_ratio)

    train_idx = set(perm[:n_train])
    val_idx = set(perm[n_train: n_train + n_val])
    test_idx = set(perm[n_train + n_val:])

    train_mask = torch.tensor([i in train_idx for i in range(num_edges)], dtype=torch.bool)
    val_mask = torch.tensor([i in val_idx for i in range(num_edges)], dtype=torch.bool)
    test_mask = torch.tensor([i in test_idx for i in range(num_edges)], dtype=torch.bool)
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(database: str, output: Path, seed: int = 42, verbose: bool = True) -> None:
    t0 = time.time()

    # Verify DB connection
    status, result = arango_request(database, "GET", "/_api/database/current")
    if status != 200:
        print(f"ERROR: Cannot connect to ArangoDB: {status} {result}")
        sys.exit(1)
    if verbose:
        print(f"Connected to ArangoDB database: {result['result']['name']}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    node_map_path = output.parent / (output.stem + "_node_maps.json")

    # Phase 1: Load vertex collections
    if verbose:
        print(f"\n=== Loading vertex collections ===")
    node_maps: dict[str, dict[str, int]] = {}
    data = HeteroData()

    for col in VERTEX_COLLECTIONS:
        try:
            keys, feat = load_vertex_collection(database, col, verbose=verbose)
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
        if keys:
            data[col].x = feat
            node_maps[col] = {k: i for i, k in enumerate(keys)}

    # Phase 2: Load edge collections
    if verbose:
        print(f"\n=== Loading edge collections ===")
    compliance_triplets: list[tuple[str, str, str]] = []

    for ecol in EDGE_COLLECTIONS:
        triplet_map = load_edge_collection(database, ecol, node_maps, verbose=verbose)
        for triplet, ei in triplet_map.items():
            src_type, edge_col, dst_type = triplet
            data[src_type, edge_col, dst_type].edge_index = ei
            if edge_col == TARGET_EDGE_COLLECTION:
                compliance_triplets.append(triplet)

    # Phase 3: Add train/val/test masks to compliance edges
    if not compliance_triplets:
        raise RuntimeError(
            f"No '{TARGET_EDGE_COLLECTION}' edges exported. "
            "Check that both endpoints have embeddings and appear in node_maps."
        )

    for triplet in compliance_triplets:
        src_type, edge_col, dst_type = triplet
        ei = data[triplet].edge_index
        num_edges = ei.size(1)
        train_mask, val_mask, test_mask = make_compliance_masks(num_edges, seed=seed)
        data[triplet].train_mask = train_mask
        data[triplet].val_mask = val_mask
        data[triplet].test_mask = test_mask
        if verbose:
            print(f"\n  {edge_col} ({src_type}->{dst_type}): {num_edges} edges, "
                  f"split {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()} "
                  f"(train/val/test)")

    # Phase 4: Save
    torch.save(data, output)
    with open(node_map_path, "w") as f:
        json.dump(node_maps, f, indent=2)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n=== Export complete ({elapsed:.1f}s) ===")
        print(f"  Graph:     {output}")
        print(f"  Node maps: {node_map_path}")
        # Summary stats
        total_nodes = sum(data[nt].x.size(0) for nt in data.node_types if hasattr(data[nt], 'x') and data[nt].x is not None)
        total_edges = sum(data[et].edge_index.size(1) for et in data.edge_types if hasattr(data[et], 'edge_index'))
        print(f"  Nodes:     {total_nodes} across {len(data.node_types)} types")
        print(f"  Edges:     {total_edges} across {len(data.edge_types)} triplets")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(output: Path, verbose: bool = True) -> bool:
    """Quick consistency check on the exported graph."""
    data = torch.load(output, weights_only=False)
    node_map_path = output.parent / (output.stem + "_node_maps.json")
    with open(node_map_path) as f:
        node_maps = json.load(f)

    ok = True
    if verbose:
        print("\n=== Verification ===")

    # 1. Node types match node_maps
    for nt in data.node_types:
        if not hasattr(data[nt], 'x') or data[nt].x is None:
            continue
        n = data[nt].x.size(0)
        mapped_n = len(node_maps.get(nt, {}))
        if n != mapped_n:
            if verbose:
                print(f"  [FAIL] {nt}: HeteroData has {n} nodes but node_map has {mapped_n}")
            ok = False
        # Check shape
        if data[nt].x.shape[1] != EMBED_DIM:
            if verbose:
                print(f"  [FAIL] {nt}: expected dim {EMBED_DIM}, got {data[nt].x.shape[1]}")
            ok = False
        # Check NaN/Inf
        if torch.isnan(data[nt].x).any() or torch.isinf(data[nt].x).any():
            if verbose:
                print(f"  [FAIL] {nt}: NaN or Inf in features")
            ok = False

    # 2. Compliance masks
    for triplet in data.edge_types:
        _, edge_col, _ = triplet
        if edge_col == TARGET_EDGE_COLLECTION:
            ei = data[triplet]
            n = ei.edge_index.size(1)
            tm = ei.train_mask.sum().item()
            vm = ei.val_mask.sum().item()
            tsm = ei.test_mask.sum().item()
            total_masked = tm + vm + tsm
            if total_masked != n:
                if verbose:
                    print(f"  [FAIL] compliance masks sum {total_masked} != {n} edges")
                ok = False
            if verbose:
                print(f"  Compliance edges: {n} total, {tm}/{vm}/{tsm} (train/val/test)")

    if ok and verbose:
        print("  All checks passed.")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export NL knowledge graph to PyG HeteroData")
    parser.add_argument("--database", default="NL", help="ArangoDB database name")
    parser.add_argument("--output", default="data/nl_graph.pt", help="Output .pt file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split")
    parser.add_argument("--verify", action="store_true", help="Verify existing output without re-exporting")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    output = Path(args.output)
    verbose = not args.quiet

    if args.verify:
        if not output.exists():
            print(f"ERROR: {output} does not exist. Run without --verify first.")
            sys.exit(1)
        ok = verify(output, verbose=verbose)
        sys.exit(0 if ok else 1)

    export(database=args.database, output=output, seed=args.seed, verbose=verbose)
    verify(output, verbose=verbose)


if __name__ == "__main__":
    main()
