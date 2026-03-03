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
  rank    Effective rank of M_l SVD spectrum (checkpoint with context required)
  cluster Vocabulary semantic clustering via embedding k-NN (W_embed + JSONL)
  align   M_l SVD subspace alignment between checkpoints (Grassmann distance)

Usage:
    cd python/
    python tools/memory_manifold_analysis.py \\
        --log runs/gate_warmup_diagnostic.jsonl \\
        --checkpoint checkpoints/gate_warmup_diagnostic.safetensors \\
        --tokenizer data/c4/tokenizer.json \\
        --module js rank cluster align \\
        --out results/gate_warmup_manifold/ \\
        [--step 20000] \\
        [--compare-steps 5000 10000 20000]

Multi-run cross-run comparison:
    python tools/memory_manifold_analysis.py \\
        --logs runs/ablation_A.jsonl runs/ablation_C.jsonl runs/ablation_D.jsonl \\
        --module js rank \\
        --out results/ablation_manifold/
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
RANK_TOP_N_PCS = 8                  # principal components for subspace comparisons (Module 4)


# ── JSONL parsing ──────────────────────────────────────────────────────────

def _load_jsonl_events(path: str, event_type: str) -> list[dict]:
    """Read all events of a given type from a training JSONL log."""
    events = []
    with open(path, encoding="utf-8") as f:
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
        warnings.warn(f"Tokenizer load failed ({e}); token strings will be IDs only.", stacklevel=2)
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

    level_ids = [lv["level"] for e in events for lv in e.get("levels", []) if "level" in lv]
    if not level_ids:
        return {"error": "No level metadata found in memory_vocab_probe events"}
    k = max(level_ids) + 1
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

    # Plot JS trajectory
    png_path = os.path.join(out_dir, "js_trajectory.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        steps_sorted = sorted(js_at_step.keys())
        for a, b in pairs:
            pair_key = f"{a}-{b}"
            xs = [s for s in steps_sorted if js_at_step[s].get(pair_key) is not None]
            ys = [js_at_step[s][pair_key] for s in xs]
            if xs:
                ax.plot(xs, ys, label=f"L{a}-L{b}")
        ax.axhline(JS_FALSIFICATION_THRESHOLD, color="red", linestyle="--",
                   label=f"threshold={JS_FALSIFICATION_THRESHOLD}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("JS divergence (nats)")
        ax.set_title("JS Divergence Trajectory")
        ax.legend()
        fig.savefig(png_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        png_path = None

    return {
        "csv_path": csv_path,
        "png_path": png_path,
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
                "Module rank requires a checkpoint saved with "
                "save_checkpoint_with_context (BPE path) or "
                "save_build_checkpoint (stream path)."
            )
        }

    d = cfg.d_model
    k = cfg.k
    rows = []
    stable_ranks_raw: list = []

    for level_idx in range(k):
        M_l = np.array(context.memory[level_idx], dtype=np.float32).reshape(d, d)
        m_fro = float(np.linalg.norm(M_l, "fro"))
        m_spec = float(np.linalg.norm(M_l, ord=2))  # largest singular value

        _, S, _ = np.linalg.svd(M_l, full_matrices=False)

        # Stable rank: ‖M‖_F² / ‖M‖_2² — invariant to scaling
        stable_rank = float((m_fro ** 2) / (m_spec ** 2 + 1e-30))
        stable_ranks_raw.append(stable_rank)

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

    # Check rank gradient prediction using raw (unrounded) values to avoid
    # masking small inversions at rounding boundaries.
    rank_gradient_ok = all(
        stable_ranks_raw[i] >= stable_ranks_raw[i + 1]
        for i in range(len(stable_ranks_raw) - 1)
    )

    return {
        "csv_path": csv_path,
        "ranks": rows,
        "rank_gradient_prediction": "PASS" if rank_gradient_ok else "FAIL",
        "rank_gradient_values": [round(x, 3) for x in stable_ranks_raw],
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
    v_expected = w_embed.shape[0]
    if os.path.exists(cache_path):
        cached = np.load(cache_path)["nn_indices"]
        if cached.shape == (v_expected, k_nn):
            return cached
        print(
            f"  WARNING: cached semantic graph shape {cached.shape} does not match "
            f"expected ({v_expected}, {k_nn}); rebuilding.",
            flush=True,
        )

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
                "coherence_ratio": coherence_ratio,  # raw; rounded at CSV write time
                "uniform_probs": False,
                "top5_tokens": top5_str,
            })

    csv_path = os.path.join(out_dir, "vocab_clustering.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "level", "m_norm",
                                           "coherence_ratio", "uniform_probs",
                                           "top5_tokens"])
        w.writeheader()
        csv_rows = [
            {**r, "coherence_ratio": (
                round(r["coherence_ratio"], 6)
                if not r["uniform_probs"] and not (isinstance(r["coherence_ratio"], float)
                                                   and r["coherence_ratio"] != r["coherence_ratio"])
                else r["coherence_ratio"]
            )}
            for r in rows
        ]
        w.writerows(csv_rows)

    # Summarise latest step per level (max-step wins to avoid ordering dependency)
    latest_by_level: dict[int, dict] = {}
    for r in rows:
        if not r["uniform_probs"]:
            if r["level"] not in latest_by_level or r["step"] > latest_by_level[r["level"]]["step"]:
                latest_by_level[r["level"]] = r

    # Check tertiary prediction: coherence_ratio(L_{k-1}) > coherence_ratio(L0)
    if nn_indices is None:
        coherence_pred = "PENDING"
    elif len(latest_by_level) >= 2 and 0 in latest_by_level and (k - 1) in latest_by_level:
        r_l0 = latest_by_level[0]["coherence_ratio"]
        r_lk = latest_by_level[k - 1]["coherence_ratio"]
        if math.isnan(r_l0) or math.isnan(r_lk):
            coherence_pred = "PENDING"
        else:
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


# ── Module 4: Level Subspace Alignment (Grassmann distance) ───────────────

def module_align(
    ckpt_paths: list,
    out_dir: str,
    compare_steps: Optional[list] = None,
) -> dict:
    """Compute per-level Grassmann distance between M_l principal subspaces.

    For each consecutive checkpoint pair (T1, T2), SVD-decomposes M_l at both
    steps, takes the top-r=8 left singular vectors, and computes the Grassmann
    distance (sum of principal angles between the two r-dimensional subspaces).

    If only one checkpoint is provided, compares against the zero-init baseline
    (zeroed M_l), giving the total rotation from initialisation.

    Spec: Module 4, specs/infrastructure/10_memory_manifold_analysis.md
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import nl_hecate

    if not ckpt_paths:
        return {"error": "No checkpoint paths provided; --checkpoint required for Module 4"}

    # Validate --compare-steps cardinality if provided
    if compare_steps is not None and len(compare_steps) != len(ckpt_paths):
        return {
            "error": (
                f"--compare-steps has {len(compare_steps)} entries but "
                f"--checkpoint has {len(ckpt_paths)} paths; they must match."
            )
        }

    # Load (step_label, d, k, context) for each checkpoint
    loaded = []
    for i, path in enumerate(ckpt_paths):
        _, cfg, context = _load_checkpoint(path)
        if context is None:
            return {
                "error": (
                    f"context_memory not available in checkpoint {path}. "
                    "Module align requires a checkpoint saved with "
                    "save_checkpoint_with_context (BPE path) or "
                    "save_build_checkpoint (stream path)."
                )
            }
        # Validate d/k consistency across checkpoints
        if loaded:
            ref_d, ref_k = loaded[0][1], loaded[0][2]
            if cfg.d_model != ref_d or cfg.k != ref_k:
                return {
                    "error": (
                        f"Checkpoint {path} has d_model={cfg.d_model}, k={cfg.k} "
                        f"but first checkpoint has d_model={ref_d}, k={ref_k}; "
                        "all checkpoints must share the same configuration."
                    )
                }
        step_label = compare_steps[i] if compare_steps else i
        loaded.append((step_label, cfg.d_model, cfg.k, context))

    # If only one checkpoint, prepend a zero-init baseline at step 0
    if len(loaded) == 1:
        _, d, k, _ = loaded[0]
        zero_ctx = nl_hecate.ContextState(k, d)
        loaded = [(0, d, k, zero_ctx)] + loaded

    d = loaded[0][1]
    k = loaded[0][2]
    r = RANK_TOP_N_PCS  # principal components per spec

    rows = []
    for idx in range(len(loaded) - 1):
        step1, _, _, ctx1 = loaded[idx]
        step2, _, _, ctx2 = loaded[idx + 1]
        step_pair = f"{step1}-{step2}"

        for level_idx in range(k):
            M1 = np.array(ctx1.memory[level_idx], dtype=np.float32).reshape(d, d)
            M2 = np.array(ctx2.memory[level_idx], dtype=np.float32).reshape(d, d)

            U1, _, _ = np.linalg.svd(M1, full_matrices=False)  # [d, d]
            U2, _, _ = np.linalg.svd(M2, full_matrices=False)

            top_r = min(r, d)
            cos_angles = np.linalg.svd(
                U1[:, :top_r].T @ U2[:, :top_r], compute_uv=False
            )
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            principal_angles = np.arccos(cos_angles)
            grassmann_dist = float(principal_angles.sum())

            rows.append({
                "step_pair": step_pair,
                "level": level_idx,
                "grassmann_distance": round(grassmann_dist, 4),
                "max_principal_angle_rad": round(float(np.max(principal_angles)), 4),
                "mean_principal_angle_rad": round(float(principal_angles.mean()), 4),
            })

    csv_path = os.path.join(out_dir, "subspace_alignment.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step_pair", "level", "grassmann_distance",
                                           "max_principal_angle_rad", "mean_principal_angle_rad"])
        w.writeheader()
        w.writerows(rows)

    return {
        "csv_path": csv_path,
        "rows": rows,
        "n_pairs": len(loaded) - 1,
        "n_levels": k,
    }


# ── Report renderer ────────────────────────────────────────────────────────

def _render_report(
    run_name: str,
    modules: list,
    results: dict,
    out_dir: str,
    step: int,
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
                latest_step = r["latest_js"]["step"]
                js_vals = {k: v for k, v in r["latest_js"].items() if k != "step" and v is not None}
                lines.append(f"  Latest step {latest_step}:")
                for pair, val in sorted(js_vals.items()):
                    lines.append(f"    L{pair}: {val:.4f}")
            lines.append(f"  Falsification (step {step}): {r['verdict']}")
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
        lines += ["[Module 4: Level Subspace Alignment]"]
        if "error" in r:
            lines.append(f"  SKIPPED: {r['error']}")
        else:
            for row in r["rows"]:
                lines.append(
                    f"  {row['step_pair']} L{row['level']}: "
                    f"grassmann={row['grassmann_distance']:.3f} rad"
                )
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
    if "align" in results and "rows" in results["align"]:
        n = len(results["align"]["rows"])
        verdicts.append(("Subspace alignment", f"{n} level×pair rows written to subspace_alignment.csv"))

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
    # Log inputs — mutually exclusive: single log or multi-run
    log_grp = p.add_mutually_exclusive_group(required=True)
    log_grp.add_argument("--log",
                         help="Path to training JSONL log (runs/*.jsonl)")
    log_grp.add_argument("--logs", nargs="+",
                         help="Multiple JSONL paths for cross-run comparison "
                              "(adds run_name column; --module js rank only)")
    p.add_argument("--checkpoint", nargs="*", default=None,
                   help="Checkpoint safetensors path(s). Single path: rank/cluster + "
                        "Module 4 vs zero-init baseline. Multiple paths: Module 4 "
                        "compares consecutive pairs.")
    p.add_argument("--tokenizer", default=None,
                   help="Path to BPE tokenizer JSON (optional; enables token decoding)")
    p.add_argument("--module", nargs="+", default=["js", "cluster", "align"],
                   choices=["js", "rank", "cluster", "align"],
                   help="Which analysis modules to run (default: js cluster align)")
    p.add_argument("--out", default="results/memory_manifold",
                   help="Output directory for CSVs and report")
    p.add_argument("--no-semantic-graph", action="store_true",
                   help="Skip k-NN semantic graph in cluster module (faster)")
    p.add_argument("--step", type=int, default=20000, dest="step",
                   help="Step at which to evaluate the JS falsification criterion "
                        "(default: 20000)")
    p.add_argument("--falsification-step", type=int, dest="step",
                   help=argparse.SUPPRESS)  # backward-compat alias for --step
    p.add_argument("--compare-steps", nargs="+", type=int, default=None,
                   help="Step labels for --checkpoint files (Module 4 temporal "
                        "comparison; must match number of --checkpoint paths)")
    p.add_argument("--run-name", default=None,
                   help="Name for the run in the report (default: log filename stem)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Normalise log inputs: --log or --logs → list of (run_name, path) pairs
    if args.logs:
        log_pairs = [(Path(p).stem, p) for p in args.logs]
    else:
        log_pairs = [(args.run_name or Path(args.log).stem, args.log)]

    # Normalise checkpoint list
    ckpt_paths: list = args.checkpoint or []
    primary_ckpt: Optional[str] = ckpt_paths[0] if ckpt_paths else None

    modules = list(args.module)

    # Warn if checkpoint-requiring modules lack a checkpoint
    needs_ckpt = {"rank", "cluster"}
    if needs_ckpt & set(modules) and primary_ckpt is None:
        print("WARNING: modules [rank, cluster] require --checkpoint. Skipping.", flush=True)
        modules = [m for m in modules if m not in needs_ckpt]

    if "align" in modules and not ckpt_paths:
        print("WARNING: module [align] requires --checkpoint. Skipping.", flush=True)
        modules = [m for m in modules if m != "align"]

    # ── Run analysis for each log (multi-run mode iterates; single-run is one pass) ──
    multi_run = len(log_pairs) > 1
    all_results: list[tuple[str, dict, str]] = []
    for run_name, jsonl_path in log_pairs:
        # Each run writes to its own subdirectory in multi-run mode
        run_out = os.path.join(args.out, run_name) if multi_run else args.out
        os.makedirs(run_out, exist_ok=True)

        print(f"Memory Manifold Analysis: {run_name}", flush=True)
        print(f"  log:        {jsonl_path}", flush=True)
        print(f"  checkpoint: {primary_ckpt or '(none)'}", flush=True)
        print(f"  modules:    {' '.join(modules)}", flush=True)
        print(f"  out:        {run_out}", flush=True)
        print(flush=True)

        tokenizer = _load_tokenizer(args.tokenizer) if args.tokenizer else None
        results: dict = {}

        # Module 1: JS trajectory
        if "js" in modules:
            print("[js] JS divergence trajectory...", flush=True)
            results["js"] = module_js(jsonl_path, run_out, args.step)
            print(f"  verdict: {results['js'].get('verdict', '')}", flush=True)

        # Module 2: Effective rank
        if "rank" in modules:
            print("[rank] Memory effective rank...", flush=True)
            results["rank"] = module_rank(primary_ckpt, run_out)
            if "error" in results["rank"]:
                print(f"  SKIPPED: {results['rank']['error'][:80]}", flush=True)
            else:
                print(f"  rank gradient: {results['rank']['rank_gradient_prediction']}",
                      flush=True)

        # Module 3: Vocabulary clustering
        if "cluster" in modules:
            print("[cluster] Vocabulary semantic clustering...", flush=True)
            results["cluster"] = module_cluster(
                jsonl_path, primary_ckpt, run_out, tokenizer,
                no_semantic_graph=args.no_semantic_graph,
            )
            if "error" in results["cluster"]:
                print(f"  ERROR: {results['cluster']['error']}", flush=True)
            else:
                print(f"  coherence gradient: "
                      f"{results['cluster']['coherence_gradient_prediction']}", flush=True)

        # Module 4: Level subspace alignment (Grassmann distance between checkpoint pairs)
        if "align" in modules:
            print("[align] Level subspace alignment (Grassmann distance)...", flush=True)
            results["align"] = module_align(ckpt_paths, run_out, args.compare_steps)
            if "error" in results["align"]:
                print(f"  SKIPPED: {results['align']['error']}", flush=True)
            else:
                print(f"  pairs={results['align']['n_pairs']}  "
                      f"levels={results['align']['n_levels']}", flush=True)

        all_results.append((run_name, results, run_out))

    # Render report for the primary (or only) run
    primary_run_name, primary_results, primary_out = all_results[0]
    report = _render_report(primary_run_name, modules, primary_results, primary_out, args.step)
    print(flush=True)
    print(report, flush=True)
    print(f"Report written to: {primary_out}/report.txt", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
