"""
CompliancePredictor: Link prediction head for nl_smell_compliance_edges.

Predicts whether a code file (arxiv_metadata) should comply with a smell (nl_code_smells).

Architecture:
  src_emb [B, hidden_dim] + dst_emb [B, hidden_dim]
      -> concat [B, 2*hidden_dim]
      -> Linear(2*hidden_dim -> hidden_dim) + ReLU
      -> Linear(hidden_dim -> 1)
      -> squeeze -> logits [B]

Loss: Binary cross-entropy with logits. Positive weight 5.0 compensates for 1:5
positive:negative ratio without over-correcting the gradient signal.

Negative sampling: corrupt the source (code file) endpoint while keeping the smell
fixed. This tests whether the model learns which code files comply with a given smell.
See docs/graphsage-architecture.md section 4.4 for rationale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompliancePredictor(nn.Module):
    """
    MLP link-prediction head for (arxiv_metadata, nl_smell_compliance_edges, nl_code_smells).

    Takes per-node embeddings from HeteroRGCN and predicts compliance edge existence.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pos_weight: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # pos_weight compensates for class imbalance; should match neg_ratio in training
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self,
        src_emb: torch.Tensor,  # [B, hidden_dim] -- code file embeddings
        dst_emb: torch.Tensor,  # [B, hidden_dim] -- smell embeddings
    ) -> torch.Tensor:
        """Returns logits [B]."""
        x = torch.cat([src_emb, dst_emb], dim=-1)  # [B, 2*hidden_dim]
        return self.decoder(x).squeeze(-1)          # [B]

    def predict_edges(
        self,
        node_embeddings: dict[str, torch.Tensor],
        edge_index: torch.Tensor,  # [2, E] in local per-type node ID space
        src_type: str = "arxiv_metadata",
        dst_type: str = "nl_code_smells",
    ) -> torch.Tensor:
        """Given edge_index in local space, return logits [E]."""
        src_emb = node_embeddings[src_type][edge_index[0]]
        dst_emb = node_embeddings[dst_type][edge_index[1]]
        return self.forward(src_emb, dst_emb)

    def loss(
        self,
        node_embeddings: dict[str, torch.Tensor],
        pos_edge_index: torch.Tensor,  # [2, P] -- positive compliance edges
        neg_edge_index: torch.Tensor,  # [2, N] -- corrupted negative edges
        src_type: str = "arxiv_metadata",
        dst_type: str = "nl_code_smells",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute BCE loss with logits over positive and negative edges.

        Returns:
            (loss_scalar, metrics_dict)
            metrics_dict keys: loss, pos_loss, neg_loss, pos_logit_mean, neg_logit_mean
        """
        pos_logits = self.predict_edges(node_embeddings, pos_edge_index, src_type, dst_type)
        neg_logits = self.predict_edges(node_embeddings, neg_edge_index, src_type, dst_type)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            pos_weight=self.pos_weight,
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
        )
        total_loss = pos_loss + neg_loss

        metrics = {
            "loss": total_loss.item(),
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "pos_logit_mean": pos_logits.detach().mean().item(),
            "neg_logit_mean": neg_logits.detach().mean().item(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Negative sampling utilities
# ---------------------------------------------------------------------------


def sample_negatives(
    pos_edge_index: torch.Tensor,  # [2, P]
    num_src: int,                  # total arxiv_metadata nodes
    neg_ratio: int = 5,
    seed: int | None = None,
    all_pos_edge_index: torch.Tensor | None = None,  # [2, E_all] -- full positive set
) -> torch.Tensor:
    """
    Generate negative edges by corrupting the source (code file) endpoint.

    For each positive edge (src, dst), sample neg_ratio source nodes that are
    NOT connected to dst in ANY known positive edge. Returns edge_index [2, P*neg_ratio].

    Pass all_pos_edge_index to exclude cross-split positives (e.g., train positives
    when sampling negatives for the val/test split).
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    pos_src = pos_edge_index[0]  # [P]
    pos_dst = pos_edge_index[1]  # [P]

    # Build exclusion set from ALL known positives (not just current split)
    if all_pos_edge_index is not None:
        all_src = all_pos_edge_index[0].tolist()
        all_dst = all_pos_edge_index[1].tolist()
        pos_set: set[tuple[int, int]] = set(zip(all_src, all_dst))
    else:
        pos_set = set(zip(pos_src.tolist(), pos_dst.tolist()))

    neg_srcs: list[int] = []
    neg_dsts: list[int] = []

    for dst in pos_dst.tolist():
        count = 0
        attempts = 0
        while count < neg_ratio and attempts < neg_ratio * 20:
            src = int(torch.randint(num_src, (1,), generator=gen).item())
            if (src, dst) not in pos_set:
                neg_srcs.append(src)
                neg_dsts.append(dst)
                count += 1
            attempts += 1

    if not neg_srcs:
        # Fallback: sample random pairs, filtering out known positives
        candidates = torch.randint(
            num_src, (len(pos_dst) * neg_ratio * 2,), generator=gen
        ).tolist()
        fallback_dsts = pos_dst.repeat(neg_ratio * 2).tolist()
        for src, dst in zip(candidates, fallback_dsts):
            if (src, dst) not in pos_set:
                neg_srcs.append(src)
                neg_dsts.append(dst)
                if len(neg_srcs) >= len(pos_dst) * neg_ratio:
                    break
        # If still empty (truly dense graph), accept any pairs as last resort
        if not neg_srcs:
            neg_srcs = candidates[:len(pos_dst) * neg_ratio]
            neg_dsts = fallback_dsts[:len(pos_dst) * neg_ratio]

    device = pos_edge_index.device
    return torch.tensor([neg_srcs, neg_dsts], dtype=torch.long, device=device)


def split_edges(
    pos_edge_index: torch.Tensor,  # [2, P]
    train_ratio: float = 0.83,
    val_ratio: float = 0.085,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split positive compliance edges into train / val / test.

    Ratios: train=0.83 (~88/106), val=0.085 (~9/106), test=remainder (~9/106).
    Returns (train_ei, val_ei, test_ei), each [2, split_size].
    """
    P = pos_edge_index.size(1)
    perm = torch.randperm(P, generator=torch.Generator().manual_seed(seed))
    n_train = int(P * train_ratio)
    n_val = int(P * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        pos_edge_index[:, train_idx],
        pos_edge_index[:, val_idx],
        pos_edge_index[:, test_idx],
    )
