#!/usr/bin/env python3
"""
Memory Manifold Analysis — post-hoc CMS level specialization diagnostic.

Spec: specs/infrastructure/10_memory_manifold_analysis.md
Task: task_4c3c03

Tests the manifold hypothesis: M_l states converge onto level-specific
low-dimensional submanifolds of R^{d×d}, and their vocabulary projections
trace semantically distinct, level-specific regions in the token embedding space.

Modules:
  js      JS divergence trajectory from JSONL (no checkpoint needed)
  rank    Effective rank of M_l via PCA on row vectors (build checkpoint only)
  cluster Vocabulary semantic clustering via embedding k-NN (W_embed + JSONL)
  align   Vocabulary distribution PCA across steps (JSONL only)

Usage:
    cd python/
    python tools/memory_manifold_analysis.py \\
        --log runs/gate_warmup_diagnostic.jsonl \\
        --checkpoint checkpoints/gate_warmup_diagnostic.safetensors \\
        --tokenizer data/c4/tokenizer.json \\
        --module js rank cluster align \\
        --out results/gate_warmup_manifold/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────

JS_FALSIFICATION_THRESHOLD = 0.1   # nats: JS(L0,L3) must exceed this at step 20K
CLUSTER_TOP_K = 20                  # tokens per vocab probe event (matches eval)
SEMANTIC_KNN = 20                   # neighbours per token in semantic graph
RANK_TOP_N_PCS = 32                 # PCs to keep for subspace comparisons


# ── JSONL parsing ──────────────────────────────────────────────────────────

def _load_jsonl_events(path: str, event_type: str) -> list[dict]:
    """Read all events of a given type from a training JSONL log."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("event") == event_type:
                    events.append(d)
            except json.JSONDecodeError:
                continue
    return events


# ── Checkpoint loading ─────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: str) -> tuple:
    """Load checkpoint. Returns (params, cfg, context_or_None).

    Tries load_build_checkpoint first (includes context_memory in build_state),
    falls back to load_checkpoint (params + cfg only, context=None).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import nl_hecate

    try:
        params, cfg, build_state = nl_hecate.load_build_checkpoint(ckpt_path)
        if build_state and "context_memory" in build_state:
            context = nl_hecate.ContextState(cfg.k, cfg.d_model)
            context.set_memory(build_state["context_memory"])
            return params, cfg, context
        return params, cfg, None
    except Exception as e:
        build_load_err = e

    try:
        params, cfg = nl_hecate.load_checkpoint(ckpt_path)
        return params, cfg, None
    except Exception as e:
        raise RuntimeError(
            f"Cannot load checkpoint {ckpt_path}. "
            f"build loader failed: {build_load_err}; fallback loader failed: {e}"
        ) from e


def _get_weights(params) -> dict:
    """Extract weight dict from MAGParams."""
    return params.get_weights()


# ── Tokenizer ──────────────────────────────────────────────────────────────

def _load_tokenizer(tok_path: Optional[str]):
    """Load BPE tokenizer if path provided. Returns callable decode(ids)->str."""
    if tok_path is None:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from engine.tokenizer import BpeTokenizer
        return BpeTokenizer(tok_path)
    except Exception as e:
        warnings.warn(f"Tokenizer load failed ({e}); token strings will be IDs only.")
        return None


def _decode_tok(tokenizer, tid: int) -> str:
    if tokenizer is None:
        return f"<{tid}>"
    try:
        s = tokenizer.decode([tid])
        # repr gives clean single-quoted string without control chars
        return repr(s)
    except Exception:
        return f"<{tid}>"


# ── Module 1: JS Divergence Trajectory ────────────────────────────────────

def module_js(jsonl_path: str, out_dir: str, falsification_step: int = 20000) -> dict:
    """Read memory_vocab_probe events; build per-step JS divergence matrix.

    Returns dict with 'csv_path', 'verdict', 'js_at_falsification'.
    """
    events = _load_jsonl_events(jsonl_path, "memory_vocab_probe")
    if not events:
        return {"error": "No memory_vocab_probe events found in JSONL"}

    k = max(e["levels"][-1]["level"] for e in events) + 1
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    rows = []
    js_at_step: dict[int, dict] = {}

    for evt in events:
        step = evt["step"]
        js_data = {f"{a}-{b}": None for a, b in pairs}
        for entry in evt.get("js_divergence", []):
            js_data[entry["levels"]] = entry["js_div"]
        rows.append({"step": step, **js_data})
        js_at_step[step] = js_data

    csv_path = os.path.join(out_dir, "js_trajectory.csv")
    fieldnames = ["step"] + [f"{a}-{b}" for a, b in pairs]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Falsification check: find the closest logged step to falsification_step
    l0_l_last = f"0-{k-1}"
    verdict = "PENDING"
    js_at_f = None

    if js_at_step:
        closest_step = min(js_at_step, key=lambda s: abs(s - falsification_step))
        js_val = js_at_step[closest_step].get(l0_l_last)
        js_at_f = {"step": closest_step, "pair": l0_l_last, "js": js_val}
        if js_val is not None:
            max_delta = max(1, int(0.1 * falsification_step))
            if abs(closest_step - falsification_step) > max_delta:
                verdict = f"PENDING — run at step {closest_step}, falsification at {falsification_step}"
            elif js_val >= JS_FALSIFICATION_THRESHOLD:
                verdict = f"PASS — JS(L0,L{k-1})={js_val:.4f} >= {JS_FALSIFICATION_THRESHOLD} at step {closest_step}"
            else:
                verdict = f"FAIL — JS(L0,L{k-1})={js_val:.4f} < {JS_FALSIFICATION_THRESHOLD} at step {closest_step}"

    return {
        "csv_path": csv_path,
        "k": k,
        "steps": len(events),
        "verdict": verdict,
        "js_at_falsification": js_at_f,
        "latest_js": rows[-1] if rows else None,
    }


# ── Module 2: Memory Effective Rank ───────────────────────────────────────

def module_rank(ckpt_path: str, out_dir: str) -> dict:
    """Compute SVD-based effective rank of M_l matrices.

    Requires a build checkpoint with context_memory in build_state.
    Returns error dict if context not available.
    """
    _, cfg, context = _load_checkpoint(ckpt_path)
    if context is None:
        return {
            "error": (
                "context_memory not available in this checkpoint. "
                "BPE checkpoints (save_checkpoint path) do not persist M_l. "
                "Module rank requires a build checkpoint saved with "
                "save_build_checkpoint (non-BPE / stream runs)."
            )
        }

    d = cfg.d_model
    k = cfg.k
    rows = []

    for level_idx in range(k):
        M_l = np.array(context.memory[level_idx], dtype=np.float32).reshape(d, d)
        m_fro = float(np.linalg.norm(M_l, "fro"))
        m_spec = float(np.linalg.norm(M_l, ord=2))  # largest singular value

        _, S, _ = np.linalg.svd(M_l, full_matrices=False)

        # Stable rank: ‖M‖_F² / ‖M‖_2² — invariant to scaling
        stable_rank = float((m_fro ** 2) / (m_spec ** 2 + 1e-30))

        # Spectral entropy rank: exp(H(p)) where p = S / S.sum()
        s_sum = S.sum() + 1e-30
        p = S / s_sum
        entropy = float(-np.sum(p * np.log(p + 1e-30)))
        spec_entropy_rank = float(np.exp(entropy))

        rows.append({
            "level": level_idx,
            "stable_rank": round(stable_rank, 3),
            "spec_entropy_rank": round(spec_entropy_rank, 3),
            "m_frobenius": round(m_fro, 6),
            "m_spectral": round(m_spec, 6),
            "top_singular_value": round(float(S[0]), 6),
        })

    csv_path = os.path.join(out_dir, "rank_profile.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Check rank gradient prediction: rank(L0) > rank(L1) > ... > rank(L_{k-1})
    stable_ranks = [r["stable_rank"] for r in rows]
    rank_gradient_ok = all(
        stable_ranks[i] >= stable_ranks[i + 1] for i in range(len(stable_ranks) - 1)
    )

    return {
        "csv_path": csv_path,
        "ranks": rows,
        "rank_gradient_prediction": "PASS" if rank_gradient_ok else "FAIL",
        "rank_gradient_values": stable_ranks,
    }


# ── Module 3: Vocabulary Semantic Clustering ──────────────────────────────

def _build_semantic_graph_fallback(w_embed: np.ndarray, k_nn: int) -> "np.ndarray":
    """Numpy fallback: exact cosine k-NN in batches of 1000. Returns [v, k_nn] indices."""
    v = w_embed.shape[0]
    norm = np.linalg.norm(w_embed, axis=1, keepdims=True) + 1e-10
    W = w_embed / norm  # [v, d] normalized

    nn_indices = np.zeros((v, k_nn), dtype=np.int32)
    batch = 1000
    for start in range(0, v, batch):
        end = min(start + batch, v)
        sims = W[start:end] @ W.T  # [batch, v]
        sims[:, start:end] -= 2 * np.eye(end - start, v, start)  # mask self
        top_k = np.argpartition(-sims, k_nn, axis=1)[:, :k_nn]
        nn_indices[start:end] = top_k
    return nn_indices


def _build_semantic_graph(w_embed: np.ndarray, cache_path: str, k_nn: int) -> np.ndarray:
    """Build or load k-NN semantic graph. Returns [v, k_nn] neighbour index array."""
    if os.path.exists(cache_path):
        return np.load(cache_path)["nn_indices"]

    v, d = w_embed.shape
    print(f"  Building semantic graph: {v} tokens, {d} dims, k={k_nn}...", flush=True)

    try:
        import faiss
        norm = np.linalg.norm(w_embed, axis=1, keepdims=True) + 1e-10
        W = (w_embed / norm).astype(np.float32)
        index = faiss.IndexFlatIP(d)
        index.add(W)
        _, nn_indices = index.search(W, k_nn + 1)   # +1 because self is included
        nn_indices = nn_indices[:, 1:].astype(np.int32)  # drop self-match
        print("    (used FAISS)", flush=True)
    except ImportError:
        print("    (FAISS not available, using numpy batched cosine — slow for large vocab)",
              flush=True)
        nn_indices = _build_semantic_graph_fallback(w_embed, k_nn)

    np.savez_compressed(cache_path, nn_indices=nn_indices)
    print(f"  Cached to {cache_path}", flush=True)
    return nn_indices


def module_cluster(
    jsonl_path: str,
    ckpt_path: str,
    out_dir: str,
    tokenizer,
    no_semantic_graph: bool = False,
) -> dict:
    """Measure vocabulary semantic clustering of memory probe top-k tokens.

    For each level at each logged step, measures how coherent the top-20 activated
    tokens are in the embedding similarity graph vs. a random baseline.
    """
    events = _load_jsonl_events(jsonl_path, "memory_vocab_probe")
    if not events:
        return {"error": "No memory_vocab_probe events found in JSONL"}

    # Load W_embed from checkpoint
    params, cfg, _ = _load_checkpoint(ckpt_path)
    weights = _get_weights(params)
    d = cfg.d_model
    v = cfg.vocab_size

    w_embed_flat = np.array(weights["w_embed"], dtype=np.float32)
    w_embed = w_embed_flat.reshape(v, d)

    nn_indices = None
    if not no_semantic_graph:
        cache_path = os.path.join(out_dir, "semantic_graph.npz")
        nn_indices = _build_semantic_graph(w_embed, cache_path, SEMANTIC_KNN)

    k = cfg.k
    rows = []

    for evt in events:
        step = evt["step"]
        for lv in evt["levels"]:
            level = lv["level"]
            m_norm = lv.get("m_norm", 0.0)
            top20 = lv.get("top20", [])

            if not top20 or m_norm < 1e-6:
                rows.append({
                    "step": step, "level": level, "m_norm": m_norm,
                    "coherence_ratio": 0.0, "uniform_probs": True,
                    "top5_tokens": "",
                })
                continue

            top_ids = [e["id"] for e in top20[:CLUSTER_TOP_K]]
            top5_str = " | ".join(_decode_tok(tokenizer, tid) for tid in top_ids[:5])

            if nn_indices is not None:
                # Count edges within the activated top-k set
                top_id_set = set(top_ids)
                edges_in = sum(
                    sum(1 for nb in nn_indices[tid] if nb in top_id_set)
                    for tid in top_ids
                )
                n = len(top_ids)
                max_possible = n * SEMANTIC_KNN
                density = edges_in / (max_possible + 1e-10)

                # Baseline: expected density for random n tokens
                # Each token has SEMANTIC_KNN neighbours; P(random token in top_n) = n/v
                baseline_density = n / v
                coherence_ratio = density / (baseline_density + 1e-10)
            else:
                coherence_ratio = float("nan")

            rows.append({
                "step": step, "level": level, "m_norm": round(m_norm, 6),
                "coherence_ratio": round(coherence_ratio, 3),
                "uniform_probs": False,
                "top5_tokens": top5_str,
            })

    csv_path = os.path.join(out_dir, "vocab_clustering.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "level", "m_norm",
                                           "coherence_ratio", "uniform_probs",
                                           "top5_tokens"])
        w.writeheader()
        w.writerows(rows)

    # Summarise latest step per level
    latest_by_level: dict[int, dict] = {}
    for r in rows:
        if not r["uniform_probs"]:
            latest_by_level[r["level"]] = r

    # Check tertiary prediction: coherence_ratio(L_{k-1}) > coherence_ratio(L0)
    if nn_indices is None:
        coherence_pred = "PENDING"
    elif len(latest_by_level) >= 2 and 0 in latest_by_level and (k - 1) in latest_by_level:
        r_l0 = latest_by_level[0]["coherence_ratio"]
        r_lk = latest_by_level[k - 1]["coherence_ratio"]
        coherence_pred = "PASS" if r_lk >= r_l0 else "FAIL"
    else:
        coherence_pred = "PENDING"

    return {
        "csv_path": csv_path,
        "rows": len(rows),
        "semantic_graph_built": nn_indices is not None,
        "latest_by_level": latest_by_level,
        "coherence_gradient_prediction": coherence_pred,
    }


# ── Module 4: Vocabulary Distribution PCA (level subspace alignment) ──────

def module_align(jsonl_path: str, out_dir: str) -> dict:
    """Measure how vocabulary probe distributions evolve and differ across levels.

    For each level, stacks the top-k sparse probability vectors across all logged
    steps into a matrix [steps × vocab] (sparse) and computes PCA to find the
    low-dimensional manifold each level traces through vocabulary space.

    Cross-level: computes cosine similarity between level PC matrices to test
    whether levels occupy distinct vocabulary subspaces.
    """
    events = _load_jsonl_events(jsonl_path, "memory_vocab_probe")
    if not events:
        return {"error": "No memory_vocab_probe events found in JSONL"}

    if len(events) < 3:
        return {"error": f"Need ≥3 probe events for PCA; only {len(events)} found"}

    k = max(e["levels"][-1]["level"] for e in events) + 1
    steps = [e["step"] for e in events]
    n_steps = len(steps)

    # Determine vocab size from events
    token_ids = [
        e2["id"]
        for evt in events
        for lv in evt["levels"]
        for e2 in lv.get("top20", [])
    ]
    if not token_ids:
        return {"error": "No top20 token data found; cannot compute alignment PCA"}
    v = max(token_ids) + 1

    # Build sparse probability matrix per level [n_steps × v] (dense for small v)
    # For large vocab (32K), use sparse representation: store only top-20 entries
    level_matrices: dict[int, np.ndarray] = {}

    for lv_idx in range(k):
        mat = np.zeros((n_steps, v), dtype=np.float32)
        for si, evt in enumerate(events):
            lv_data = next((lv for lv in evt["levels"] if lv["level"] == lv_idx), None)
            if lv_data and lv_data.get("top20"):
                for entry in lv_data["top20"]:
                    mat[si, entry["id"]] = entry["prob"]
        level_matrices[lv_idx] = mat

    # PCA on each level matrix — find top-r principal components
    r = min(RANK_TOP_N_PCS, n_steps, 64)  # can't exceed n_steps
    level_pcs: dict[int, np.ndarray] = {}
    level_explained: dict[int, list] = {}
    rows = []

    for lv_idx in range(k):
        mat = level_matrices[lv_idx]
        # Center
        mean = mat.mean(axis=0)
        centered = mat - mean
        # SVD (n_steps × v, n_steps typically << v so use economy SVD)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # U: [n_steps, n_steps], S: [n_steps], Vt: [n_steps, v]
        # Principal directions in vocabulary space: rows of Vt
        top_r = min(r, len(S))
        level_pcs[lv_idx] = Vt[:top_r]  # [r, v]

        total_var = float((S ** 2).sum()) + 1e-30
        explained = [(float(s ** 2) / total_var) for s in S[:top_r]]
        level_explained[lv_idx] = explained

        cumulative_80 = next(
            (i + 1 for i, _ in enumerate(np.cumsum(explained)) if _ >= 0.80),
            len(explained)
        )
        rows.append({
            "level": lv_idx,
            "n_steps": n_steps,
            "top1_explained_var": round(explained[0], 4) if explained else 0.0,
            "top3_explained_var": round(sum(explained[:3]), 4) if len(explained) >= 3 else 0.0,
            "dims_for_80pct_var": cumulative_80,
        })

    # Cross-level subspace alignment: cosine similarity between PC matrices
    cross_rows = []
    for i in range(k):
        for j in range(i + 1, k):
            pci = level_pcs[i]       # [r, v]
            pcj = level_pcs[j]       # [r, v]
            # Grassmann-like: nuclear norm of PCs' gram matrix
            # Normalise each PC vector
            pi_norm = pci / (np.linalg.norm(pci, axis=1, keepdims=True) + 1e-10)
            pj_norm = pcj / (np.linalg.norm(pcj, axis=1, keepdims=True) + 1e-10)
            gram = pi_norm @ pj_norm.T   # [r, r]
            cos_angles = np.linalg.svd(gram, compute_uv=False)
            cos_angles = np.clip(cos_angles, -1, 1)
            principal_angles = np.arccos(cos_angles)
            subspace_dist = float(principal_angles.sum())
            max_angle = float(np.max(principal_angles))

            cross_rows.append({
                "level_pair": f"{i}-{j}",
                "mean_principal_angle_rad": round(float(principal_angles.mean()), 4),
                "max_principal_angle_rad": round(max_angle, 4),
                "subspace_distance": round(subspace_dist, 4),
            })

    csv_path = os.path.join(out_dir, "subspace_alignment.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level", "n_steps", "top1_explained_var",
                                           "top3_explained_var", "dims_for_80pct_var"])
        w.writeheader()
        w.writerows(rows)

    cross_csv = os.path.join(out_dir, "cross_level_alignment.csv")
    with open(cross_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["level_pair", "mean_principal_angle_rad",
                                           "max_principal_angle_rad", "subspace_distance"])
        w.writeheader()
        w.writerows(cross_rows)

    # Prediction: L0-L_{k-1} subspace distance > L0-L1 distance
    if cross_rows and k >= 2:
        dist_01 = next((r["subspace_distance"] for r in cross_rows if r["level_pair"] == "0-1"), None)
        dist_0k = next((r["subspace_distance"] for r in cross_rows
                        if r["level_pair"] == f"0-{k-1}"), None)
        if dist_01 is not None and dist_0k is not None and dist_01 > 0:
            align_pred = ("PASS" if dist_0k >= dist_01
                          else f"FAIL — L0-L{k-1} dist={dist_0k:.3f} < L0-L1 dist={dist_01:.3f}")
        else:
            align_pred = "PENDING"
    else:
        align_pred = "PENDING"

    return {
        "pca_csv": csv_path,
        "cross_csv": cross_csv,
        "level_dims": rows,
        "cross_level": cross_rows,
        "distant_levels_prediction": align_pred,
    }


# ── Report renderer ────────────────────────────────────────────────────────

def _render_report(
    run_name: str,
    modules: list[str],
    results: dict,
    out_dir: str,
    falsification_step: int,
) -> str:
    lines = [
        f"{'=' * 60}",
        f"Memory Manifold Analysis: {run_name}",
        f"{'=' * 60}",
        "",
    ]

    if "js" in modules and "js" in results:
        r = results["js"]
        lines += ["[Module 1: JS Divergence Trajectory]"]
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
        else:
            lines.append(f"  Steps logged: {r['steps']}  |  Levels: {r['k']}")
            if r["latest_js"]:
                step = r["latest_js"]["step"]
                js_vals = {k: v for k, v in r["latest_js"].items() if k != "step" and v is not None}
                lines.append(f"  Latest step {step}:")
                for pair, val in sorted(js_vals.items()):
                    lines.append(f"    L{pair}: {val:.4f}")
            lines.append(f"  Falsification (step {falsification_step}): {r['verdict']}")
        lines.append("")

    if "rank" in modules and "rank" in results:
        r = results["rank"]
        lines += ["[Module 2: Memory Effective Rank]"]
        if "error" in r:
            lines.append(f"  SKIPPED: {r['error']}")
        else:
            for row in r["ranks"]:
                lines.append(
                    f"  L{row['level']}: stable_rank={row['stable_rank']:.1f}"
                    f"  spec_entropy={row['spec_entropy_rank']:.1f}"
                    f"  ‖M‖_F={row['m_frobenius']:.4f}"
                )
            lines.append(f"  Rank gradient (L0>L1>...>L_k): {r['rank_gradient_prediction']}")
        lines.append("")

    if "cluster" in modules and "cluster" in results:
        r = results["cluster"]
        lines += ["[Module 3: Vocabulary Semantic Clustering]"]
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
        else:
            if not r["semantic_graph_built"]:
                lines.append("  (semantic graph disabled — coherence_ratio unavailable)")
            for lv, row in sorted(r["latest_by_level"].items()):
                cr = row["coherence_ratio"]
                cr_str = f"{cr:.2f}" if not math.isnan(cr) else "N/A"
                lines.append(
                    f"  L{lv}: coherence={cr_str}  ‖M‖={row['m_norm']:.4f}"
                    f"  top5: {row['top5_tokens']}"
                )
            lines.append(f"  Coherence gradient (L_k>L0): {r['coherence_gradient_prediction']}")
        lines.append("")

    if "align" in modules and "align" in results:
        r = results["align"]
        lines += ["[Module 4: Vocabulary Subspace Alignment]"]
        if "error" in r:
            lines.append(f"  {'SKIPPED' if 'Need' in r['error'] else 'ERROR'}: {r['error']}")
        else:
            for row in r["level_dims"]:
                lines.append(
                    f"  L{row['level']}: dims_for_80pct={row['dims_for_80pct_var']}"
                    f"  top3_var={row['top3_explained_var']:.3f}"
                )
            if r["cross_level"]:
                lines.append("  Cross-level subspace distances:")
                for row in r["cross_level"]:
                    lines.append(f"    L{row['level_pair']}: {row['subspace_distance']:.3f} rad")
            lines.append(f"  Distant-levels prediction (L0-Lk > L0-L1): {r['distant_levels_prediction']}")
        lines.append("")

    # Overall verdict
    lines += ["[Overall Verdict]"]
    verdicts = []
    if "js" in results and "verdict" in results["js"]:
        verdicts.append(("JS differentiation", results["js"]["verdict"]))
    if "rank" in results and "rank_gradient_prediction" in results["rank"]:
        verdicts.append(("Rank gradient", results["rank"]["rank_gradient_prediction"]))
    if "cluster" in results and "coherence_gradient_prediction" in results["cluster"]:
        verdicts.append(("Coherence gradient", results["cluster"]["coherence_gradient_prediction"]))
    if "align" in results and "distant_levels_prediction" in results["align"]:
        verdicts.append(("Subspace distance", results["align"]["distant_levels_prediction"]))

    for name, v in verdicts:
        lines.append(f"  {name}: {v}")

    report = "\n".join(lines) + "\n"
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    return report


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory Manifold Analysis — CMS level specialization diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--log", required=True,
                   help="Path to training JSONL log (runs/*.jsonl)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint safetensors path (required for rank, cluster modules)")
    p.add_argument("--tokenizer", default=None,
                   help="Path to BPE tokenizer JSON (optional; enables token decoding)")
    p.add_argument("--module", nargs="+", default=["js", "cluster", "align"],
                   choices=["js", "rank", "cluster", "align"],
                   help="Which analysis modules to run (default: js cluster align)")
    p.add_argument("--out", default="results/memory_manifold",
                   help="Output directory for CSVs and report")
    p.add_argument("--no-semantic-graph", action="store_true",
                   help="Skip k-NN semantic graph in cluster module (faster)")
    p.add_argument("--falsification-step", type=int, default=20000,
                   help="Step at which to evaluate the JS falsification criterion (default: 20000)")
    p.add_argument("--run-name", default=None,
                   help="Name for the run in the report (default: log filename stem)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_name = args.run_name or Path(args.log).stem
    modules = args.module

    print(f"Memory Manifold Analysis: {run_name}", flush=True)
    print(f"  log:        {args.log}", flush=True)
    print(f"  checkpoint: {args.checkpoint or '(none)'}", flush=True)
    print(f"  modules:    {' '.join(modules)}", flush=True)
    print(f"  out:        {args.out}", flush=True)
    print(flush=True)

    tokenizer = _load_tokenizer(args.tokenizer) if args.tokenizer else None

    results: dict = {}
    needs_ckpt = {"rank", "cluster"}

    if needs_ckpt & set(modules) and args.checkpoint is None:
        print("WARNING: modules [rank, cluster] require --checkpoint. Skipping.", flush=True)
        modules = [m for m in modules if m not in needs_ckpt]

    # Module 1: JS trajectory
    if "js" in modules:
        print("[js] JS divergence trajectory...", flush=True)
        results["js"] = module_js(args.log, args.out, args.falsification_step)
        v = results["js"].get("verdict", "")
        print(f"  verdict: {v}", flush=True)

    # Module 2: Effective rank
    if "rank" in modules:
        print("[rank] Memory effective rank...", flush=True)
        results["rank"] = module_rank(args.checkpoint, args.out)
        if "error" in results["rank"]:
            print(f"  SKIPPED: {results['rank']['error'][:80]}", flush=True)
        else:
            print(f"  rank gradient: {results['rank']['rank_gradient_prediction']}", flush=True)

    # Module 3: Vocabulary clustering
    if "cluster" in modules:
        print("[cluster] Vocabulary semantic clustering...", flush=True)
        results["cluster"] = module_cluster(
            args.log, args.checkpoint, args.out, tokenizer,
            no_semantic_graph=args.no_semantic_graph,
        )
        if "error" in results["cluster"]:
            print(f"  ERROR: {results['cluster']['error']}", flush=True)
        else:
            print(f"  coherence gradient: {results['cluster']['coherence_gradient_prediction']}",
                  flush=True)

    # Module 4: Subspace alignment via vocabulary PCA
    if "align" in modules:
        print("[align] Vocabulary distribution PCA...", flush=True)
        results["align"] = module_align(args.log, args.out)
        if "error" in results["align"]:
            print(f"  {'SKIPPED' if 'Need' in results['align']['error'] else 'ERROR'}: "
                  f"{results['align']['error']}", flush=True)
        else:
            print(f"  distant-levels prediction: {results['align']['distant_levels_prediction']}",
                  flush=True)

    # Render report
    report = _render_report(run_name, modules, results, args.out, args.falsification_step)
    print(flush=True)
    print(report, flush=True)
    print(f"Report written to: {args.out}/report.txt", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
