"""
evaluate_rgcn.py -- Evaluate a saved RGCN checkpoint on the NL compliance graph.

Loads checkpoints/rgcn_best.pt (or any .pt produced by train_rgcn.py) and
reports AUC, AP, and per-smell breakdown on the test split.

Usage:
  CUDA_VISIBLE_DEVICES=2 python python/training/evaluate_rgcn.py \\
      --data data/nl_graph.pt \\
      --checkpoint checkpoints/rgcn_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from compliance_predictor import CompliancePredictor, sample_negatives
from rgcn_model import HeteroRGCN


COMPLIANCE_TRIPLET = (
    "arxiv_metadata",
    "nl_smell_compliance_edges",
    "nl_code_smells",
)


# ---------------------------------------------------------------------------
# Helpers (duplicated here so evaluate_rgcn.py is self-contained)
# ---------------------------------------------------------------------------


def run_forward(
    model: HeteroRGCN,
    predictor: CompliancePredictor,
    data,
    pos_ei: torch.Tensor,
    device: torch.device,
    neg_ratio: int = 20,
    seed: int = 99,
) -> dict[str, float]:
    """Return AUC and AP for a set of positive edges."""
    model.train(False)
    predictor.train(False)

    with torch.no_grad():
        node_embs = model(data)

    num_src = node_embs["arxiv_metadata"].size(0)
    neg_ei = sample_negatives(pos_ei, num_src, neg_ratio=neg_ratio, seed=seed).to(device)

    pos_logits = predictor.predict_edges(node_embs, pos_ei).detach().cpu().numpy()
    neg_logits = predictor.predict_edges(node_embs, neg_ei).detach().cpu().numpy()

    scores = list(pos_logits) + list(neg_logits)
    labels = [1] * len(pos_logits) + [0] * len(neg_logits)

    return {
        "auc": roc_auc_score(labels, scores),
        "ap":  average_precision_score(labels, scores),
    }


def cosine_auc(
    data,
    test_ei: torch.Tensor,
    num_neg_per_pos: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Cosine similarity baseline (raw 2048d Jina embeddings)."""
    src_norm = F.normalize(data["arxiv_metadata"].x.cpu(), dim=-1)
    dst_norm = F.normalize(data["nl_code_smells"].x.cpu(), dim=-1)
    num_src = src_norm.size(0)

    pos_scores = (src_norm[test_ei[0].cpu()] * dst_norm[test_ei[1].cpu()]).sum(-1)

    gen = torch.Generator().manual_seed(seed)
    neg_srcs = torch.randint(num_src, (test_ei.size(1) * num_neg_per_pos,), generator=gen)
    neg_dsts = test_ei[1].cpu().repeat(num_neg_per_pos)
    neg_scores = (src_norm[neg_srcs] * dst_norm[neg_dsts]).sum(-1)

    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)

    return {
        "cosine_auc": roc_auc_score(labels, scores),
        "cosine_ap":  average_precision_score(labels, scores),
    }


def per_smell_breakdown(
    model: HeteroRGCN,
    predictor: CompliancePredictor,
    data,
    pos_ei: torch.Tensor,
    node_maps_path: Path | None,
    device: torch.device,
    neg_ratio: int = 20,
) -> dict[str, dict[str, float]]:
    """AUC + AP per unique destination smell in pos_ei."""
    model.train(False)
    predictor.train(False)

    with torch.no_grad():
        node_embs = model(data)

    smell_idx_to_key: dict[int, str] = {}
    if node_maps_path and node_maps_path.exists():
        with open(node_maps_path) as f:
            node_maps = json.load(f)
        smell_map = node_maps.get("nl_code_smells", {})
        smell_idx_to_key = {v: k for k, v in smell_map.items()}

    num_src = node_embs["arxiv_metadata"].size(0)
    gen = torch.Generator().manual_seed(0)
    results: dict[str, dict[str, float]] = {}

    for dst_idx in pos_ei[1].unique().tolist():
        mask = pos_ei[1] == dst_idx
        local_pos = pos_ei[:, mask].to(device)
        n_pos = local_pos.size(1)

        neg_srcs = torch.randint(num_src, (n_pos * neg_ratio,), generator=gen)
        neg_dsts = torch.full((n_pos * neg_ratio,), dst_idx, dtype=torch.long)
        local_neg = torch.stack([neg_srcs, neg_dsts]).to(device)

        pl = predictor.predict_edges(node_embs, local_pos).detach().cpu().numpy()
        nl = predictor.predict_edges(node_embs, local_neg).detach().cpu().numpy()

        scores = list(pl) + list(nl)
        labels = [1] * n_pos + [0] * (n_pos * neg_ratio)

        if len(set(labels)) < 2:
            continue
        try:
            auc = roc_auc_score(labels, scores)
            ap  = average_precision_score(labels, scores)
        except ValueError:
            continue

        key = smell_idx_to_key.get(int(dst_idx), str(int(dst_idx)))
        results[key] = {"auc": auc, "ap": ap, "pos": n_pos}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved RGCN checkpoint")
    parser.add_argument("--data",       default="data/nl_graph.pt")
    parser.add_argument("--checkpoint", default="checkpoints/rgcn_best.pt")
    parser.add_argument("--neg-ratio",  type=int, default=20, dest="neg_ratio")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load graph
    print(f"Loading graph: {args.data}")
    data = torch.load(args.data, weights_only=False)
    data = data.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})

    # Rebuild model from saved args
    hidden_dim = saved_args.get("hidden_dim", 256)
    num_bases  = saved_args.get("num_bases", 4)
    num_layers = saved_args.get("num_layers", 2)
    dropout    = saved_args.get("dropout", 0.1)

    model = HeteroRGCN(
        hidden_dim=hidden_dim, num_layers=num_layers, num_bases=num_bases, dropout=dropout
    ).to(device)
    predictor = CompliancePredictor(hidden_dim=hidden_dim, dropout=dropout).to(device)

    model.load_state_dict(ckpt["model_state"])
    predictor.load_state_dict(ckpt["predictor_state"])

    print(f"  Checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val_auc={ckpt.get('val_auc', '?'):.4f})")

    # Select split
    compliance_ei = data[COMPLIANCE_TRIPLET].edge_index
    mask_attr = f"{args.split}_mask"
    mask = getattr(data[COMPLIANCE_TRIPLET], mask_attr)
    pos_ei = compliance_ei[:, mask]
    print(f"\nEvaluating {args.split} split: {pos_ei.size(1)} positive edges")

    # Cosine baseline
    baseline = cosine_auc(data, pos_ei, num_neg_per_pos=args.neg_ratio)
    print(f"\nCosine baseline:  AUC={baseline['cosine_auc']:.4f}  AP={baseline['cosine_ap']:.4f}")

    # RGCN evaluation
    rgcn_m = run_forward(model, predictor, data, pos_ei, device,
                         neg_ratio=args.neg_ratio)
    print(f"RGCN:             AUC={rgcn_m['auc']:.4f}  AP={rgcn_m['ap']:.4f}")

    delta_auc = rgcn_m["auc"] - baseline["cosine_auc"]
    delta_ap  = rgcn_m["ap"]  - baseline["cosine_ap"]
    print(f"Delta (RGCN - cosine): ΔAUC={delta_auc:+.4f}  ΔAP={delta_ap:+.4f}")

    # Per-smell breakdown
    print(f"\nPer-smell breakdown ({args.split} split):")
    node_map_path = Path(args.data).with_name(Path(args.data).stem + "_node_maps.json")
    breakdown = per_smell_breakdown(
        model, predictor, data, pos_ei, node_map_path, device, neg_ratio=args.neg_ratio
    )
    for smell_key, m in sorted(breakdown.items(), key=lambda x: -x[1]["auc"]):
        print(f"  {smell_key:40s}  AUC={m['auc']:.3f}  AP={m['ap']:.3f}  pos={m['pos']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
