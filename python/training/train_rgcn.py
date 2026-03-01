"""
train_rgcn.py -- Train RGCN compliance link predictor on the NL knowledge graph.

Reads data/nl_graph.pt (HeteroData produced by export_nl_graph.py), trains an
HeteroRGCN + CompliancePredictor to predict nl_smell_compliance_edges, and
saves the best checkpoint to checkpoints/rgcn_best.pt.

Also reports a cosine similarity baseline before training begins.

Usage:
  CUDA_VISIBLE_DEVICES=2 python python/training/train_rgcn.py \\
      --data data/nl_graph.pt \\
      --hidden-dim 256 \\
      --num-bases 4 \\
      --epochs 200 \\
      --lr 1e-3 \\
      --output checkpoints/rgcn_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# Allow running from repo root as:  python python/training/train_rgcn.py
sys.path.insert(0, str(Path(__file__).parent))

from compliance_predictor import CompliancePredictor, sample_negatives
from rgcn_model import HeteroRGCN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPLIANCE_TRIPLET = (
    "arxiv_metadata",
    "nl_smell_compliance_edges",
    "nl_code_smells",
)


# ---------------------------------------------------------------------------
# Cosine similarity baseline
# ---------------------------------------------------------------------------


def cosine_baseline(
    data,
    test_pos_ei: torch.Tensor,
    all_pos_ei: torch.Tensor | None = None,
    num_neg_per_pos: int = 20,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    Score each (code_file, smell) pair by cosine similarity of raw Jina embeddings.

    Generates num_neg_per_pos negatives per positive (random source, same destination),
    filtering out all known positive edges to avoid label noise.
    """
    src_feat = data["arxiv_metadata"].x.to(device)   # [N_src, 2048]
    dst_feat = data["nl_code_smells"].x.to(device)    # [N_dst, 2048]

    src_norm = F.normalize(src_feat, dim=-1)  # [N_src, 2048]
    dst_norm = F.normalize(dst_feat, dim=-1)  # [N_dst, 2048]
    num_src = src_norm.size(0)

    # Build exclusion set from all known positives
    ref_ei = all_pos_ei if all_pos_ei is not None else test_pos_ei
    all_pos_set: set[tuple[int, int]] = set(
        zip(ref_ei[0].tolist(), ref_ei[1].tolist())
    )

    # Positive scores
    pos_src_emb = src_norm[test_pos_ei[0]]   # [P, 2048]
    pos_dst_emb = dst_norm[test_pos_ei[1]]   # [P, 2048]
    pos_scores = (pos_src_emb * pos_dst_emb).sum(dim=-1).cpu()  # [P]

    # Negative scores: random source, same destination, filtered against all_pos_set
    gen = torch.Generator().manual_seed(seed)
    neg_src_list: list[int] = []
    neg_dst_list: list[int] = []
    for dst in test_pos_ei[1].tolist():
        count = 0
        attempts = 0
        while count < num_neg_per_pos and attempts < num_neg_per_pos * 20:
            src = int(torch.randint(num_src, (1,), generator=gen).item())
            if (src, dst) not in all_pos_set:
                neg_src_list.append(src)
                neg_dst_list.append(dst)
                count += 1
            attempts += 1
    neg_srcs = torch.tensor(neg_src_list, dtype=torch.long)
    neg_dsts = torch.tensor(neg_dst_list, dtype=torch.long)
    neg_src_emb = src_norm[neg_srcs]
    neg_dst_emb = dst_norm[neg_dsts]
    neg_scores = (neg_src_emb * neg_dst_emb).sum(dim=-1).cpu()

    scores = torch.cat([pos_scores, neg_scores]).numpy()
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    return {"cosine_auc": auc, "cosine_ap": ap}


# ---------------------------------------------------------------------------
# Per-smell breakdown
# ---------------------------------------------------------------------------


def per_smell_breakdown(
    model: HeteroRGCN,
    predictor: CompliancePredictor,
    data,
    pos_ei: torch.Tensor,          # [2, P] -- test positive edges
    node_maps_path: Path | None,
    device: torch.device,
    neg_ratio: int = 20,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """
    For each unique destination smell in pos_ei, compute AUC and AP for that smell.
    Returns {smell_key: {"auc": float, "ap": float, "pos": int}}.
    """
    model.train(False)
    predictor.train(False)

    with torch.no_grad():
        node_embs = model(data)

    # Load smell key map for human-readable names
    smell_idx_to_key: dict[int, str] = {}
    if node_maps_path and node_maps_path.exists():
        with open(node_maps_path) as f:
            node_maps = json.load(f)
        smell_map = node_maps.get("nl_code_smells", {})
        smell_idx_to_key = {v: k for k, v in smell_map.items()}

    num_src = node_embs["arxiv_metadata"].size(0)
    results: dict[str, dict[str, float]] = {}
    gen = torch.Generator().manual_seed(seed)

    # Build exclusion set from all known positives to avoid label noise
    all_pos_set: set[tuple[int, int]] = set(
        zip(pos_ei[0].tolist(), pos_ei[1].tolist())
    )

    # Group positive edges by destination smell index
    unique_dsts = pos_ei[1].unique().tolist()
    for dst_idx in unique_dsts:
        mask = pos_ei[1] == dst_idx
        local_pos_ei = pos_ei[:, mask].to(device)
        n_pos = local_pos_ei.size(1)

        neg_srcs_list: list[int] = []
        for _ in range(n_pos * neg_ratio):
            for _ in range(100):  # max attempts per sample
                src = int(torch.randint(num_src, (1,), generator=gen).item())
                if (src, int(dst_idx)) not in all_pos_set:
                    neg_srcs_list.append(src)
                    break
        neg_srcs = torch.tensor(neg_srcs_list, dtype=torch.long)
        neg_dsts = torch.full((len(neg_srcs_list),), dst_idx, dtype=torch.long)
        local_neg_ei = torch.stack([neg_srcs, neg_dsts], dim=0).to(device)

        pos_logits = predictor.predict_edges(node_embs, local_pos_ei).detach().cpu().numpy()
        neg_logits = predictor.predict_edges(node_embs, local_neg_ei).detach().cpu().numpy()

        scores = list(pos_logits) + list(neg_logits)
        labels = [1] * n_pos + [0] * len(neg_srcs_list)

        if len(set(labels)) < 2:
            continue  # can't compute AUC with only one class

        try:
            auc = roc_auc_score(labels, scores)
            ap  = average_precision_score(labels, scores)
        except ValueError:
            continue

        smell_key = smell_idx_to_key.get(int(dst_idx), str(int(dst_idx)))
        results[smell_key] = {"auc": auc, "ap": ap, "pos": n_pos}

    return results


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def evaluate(
    model: HeteroRGCN,
    predictor: CompliancePredictor,
    data,
    pos_ei: torch.Tensor,             # [2, P] -- positive edges for this split
    device: torch.device,
    neg_ratio: int = 5,
    seed: int = 0,
    all_pos_ei: torch.Tensor | None = None,  # [2, E_all] -- full pos set for exclusion
) -> dict[str, float]:
    """Run full-graph forward pass and return AUC + AP for a split."""
    model.train(False)
    predictor.train(False)

    with torch.no_grad():
        node_embs = model(data)

    num_src = node_embs["arxiv_metadata"].size(0)
    neg_ei = sample_negatives(
        pos_ei, num_src, neg_ratio=neg_ratio, seed=seed, all_pos_edge_index=all_pos_ei
    )
    neg_ei = neg_ei.to(device)

    pos_logits = predictor.predict_edges(node_embs, pos_ei).detach().cpu().numpy()
    neg_logits = predictor.predict_edges(node_embs, neg_ei).detach().cpu().numpy()

    scores = list(pos_logits) + list(neg_logits)
    labels = [1] * len(pos_logits) + [0] * len(neg_logits)

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    return {"auc": auc, "ap": ap}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}, {props.total_memory // 1_048_576}MB")

    # ------------------------------------------------------------------ Data
    print(f"\nLoading graph: {args.data}")
    data = torch.load(args.data, weights_only=False)
    data = data.to(device)

    # Compliance edge index + masks
    compliance_ei = data[COMPLIANCE_TRIPLET].edge_index     # [2, E]
    train_mask    = data[COMPLIANCE_TRIPLET].train_mask     # [E] bool
    val_mask      = data[COMPLIANCE_TRIPLET].val_mask
    test_mask     = data[COMPLIANCE_TRIPLET].test_mask

    train_ei = compliance_ei[:, train_mask]   # [2, n_train]
    val_ei   = compliance_ei[:, val_mask]     # [2, n_val]
    test_ei  = compliance_ei[:, test_mask]    # [2, n_test]

    num_src = data["arxiv_metadata"].x.size(0)
    num_dst = data["nl_code_smells"].x.size(0)
    total_nodes = sum(
        data[nt].x.size(0)
        for nt in data.node_types
        if hasattr(data[nt], "x") and data[nt].x is not None
    )
    print(f"  Nodes:        {total_nodes}")
    print(f"  Edge types:   {len(data.edge_types)}")
    print(f"  Compliance:   {compliance_ei.size(1)} total "
          f"({train_ei.size(1)} train / {val_ei.size(1)} val / {test_ei.size(1)} test)")
    print(f"  arxiv_metadata: {num_src} nodes, nl_code_smells: {num_dst} nodes")

    # -------------------------------------------- Cosine similarity baseline
    print("\n=== Cosine similarity baseline ===")
    baseline = cosine_baseline(
        data, test_ei.cpu(), all_pos_ei=compliance_ei.cpu(), device=torch.device("cpu")
    )
    print(f"  Test AUC: {baseline['cosine_auc']:.4f}  AP: {baseline['cosine_ap']:.4f}")

    # ------------------------------------------------------------ Build model
    model = HeteroRGCN(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
    ).to(device)
    predictor = CompliancePredictor(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = (
        sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in predictor.parameters())
    )
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # --------------------------------------------------------- Training loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_auc     = float("-inf")
    patience_counter = 0
    history: list[dict] = []

    print(f"\n=== Training ({args.epochs} epochs, patience={args.patience}) ===")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train(True)
        predictor.train(True)
        optimizer.zero_grad()

        # Full-graph forward pass — compliance edges excluded by HeteroRGCN edge list
        node_embs = model(data)

        # Fresh negatives each epoch (seed varies so negatives are not static)
        neg_ei = sample_negatives(train_ei, num_src, neg_ratio=args.neg_ratio, seed=epoch)
        neg_ei = neg_ei.to(device)

        loss, metrics = predictor.loss(node_embs, train_ei, neg_ei)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters()),
                args.grad_clip,
            )

        optimizer.step()
        scheduler.step()

        # Evaluate val set (cheap — full-graph, <1000 nodes)
        val_m = evaluate(model, predictor, data, val_ei, device,
                         neg_ratio=args.neg_ratio, seed=epoch + 10000,
                         all_pos_ei=compliance_ei)
        val_auc = val_m["auc"]
        val_ap  = val_m["ap"]

        record = {
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "val_auc": val_auc,
            "val_ap": val_ap,
            "pos_logit_mean": metrics["pos_logit_mean"],
            "neg_logit_mean": metrics["neg_logit_mean"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)

        # Save checkpoint when validation improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "predictor_state": predictor.state_dict(),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "args": vars(args),
                    "history": history,
                },
                output_path,
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch <= 5:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}  loss={metrics['loss']:.4f}  "
                  f"val_auc={val_auc:.4f}  val_ap={val_ap:.4f}  "
                  f"patience={patience_counter}/{args.patience}  "
                  f"[{elapsed:.1f}s]")

        if patience_counter >= args.patience:
            print(f"\n  Early stop at epoch {epoch} (patience {args.patience} exceeded)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s. Best val AUC: {best_val_auc:.4f}")

    # --------------------------------------------------------- Test evaluation
    print("\n=== Test evaluation (best checkpoint) ===")
    ckpt = torch.load(output_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    predictor.load_state_dict(ckpt["predictor_state"])

    test_m = evaluate(model, predictor, data, test_ei, device, neg_ratio=20, seed=99,
                      all_pos_ei=compliance_ei)
    print(f"  RGCN   — Test AUC: {test_m['auc']:.4f}  AP: {test_m['ap']:.4f}")
    print(f"  Cosine — Test AUC: {baseline['cosine_auc']:.4f}  AP: {baseline['cosine_ap']:.4f}")

    # ---------------------------------------------------- Per-smell breakdown
    print("\n=== Per-smell AUC breakdown ===")
    node_map_path = Path(args.data).with_name(Path(args.data).stem + "_node_maps.json")
    smell_breakdown = per_smell_breakdown(
        model, predictor, data, test_ei, node_map_path, device, neg_ratio=20
    )
    for smell_key, m in sorted(smell_breakdown.items(), key=lambda x: -x[1]["auc"]):
        print(f"  {smell_key:40s}  AUC={m['auc']:.3f}  AP={m['ap']:.3f}  pos={m['pos']}")

    # ----------------------------------------------------------- Save results
    results = {
        "baseline": baseline,
        "best_val_auc": best_val_auc,
        "test_auc": test_m["auc"],
        "test_ap": test_m["ap"],
        "best_epoch": ckpt["epoch"],
        "total_epochs": len(history),
        "history": history,
        "smell_breakdown": smell_breakdown,
        "args": vars(args),
    }

    results_path = output_path.with_suffix(".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {results_path}")
    print(f"Best checkpoint → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RGCN compliance predictor")
    parser.add_argument("--data",         default="data/nl_graph.pt",
                        help="Path to HeteroData .pt file")
    parser.add_argument("--hidden-dim",   type=int, default=256)
    parser.add_argument("--num-bases",    type=int, default=4,
                        help="RGCN basis decomposition rank")
    parser.add_argument("--num-layers",   type=int, default=2,
                        help="Number of RGCN message-passing layers")
    parser.add_argument("--dropout",      type=float, default=0.1)
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--neg-ratio",    type=int, default=5, dest="neg_ratio",
                        help="Negatives per positive edge during training")
    parser.add_argument("--patience",     type=int, default=20,
                        help="Early-stopping patience on val AUC")
    parser.add_argument("--grad-clip",    type=float, default=1.0, dest="grad_clip",
                        help="Gradient norm clipping (0 = disabled)")
    parser.add_argument("--output",       default="checkpoints/rgcn_best.pt",
                        help="Path for best checkpoint")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
