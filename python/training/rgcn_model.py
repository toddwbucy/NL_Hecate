"""
HeteroRGCN: Relational Graph Convolutional Network over the NL knowledge graph.

Architecture overview (see docs/graphsage-architecture.md for full rationale):
  - Per-type input projection: Linear(2048 -> hidden_dim, bias=False)
  - Flatten all node types into a homogeneous tensor
  - RGCNConv layers with basis decomposition (B=4) to regularize 12 relation matrices
  - LayerNorm + ReLU + Dropout between layers
  - Split outputs back to per-type dicts

The prediction target edge type (nl_smell_compliance_edges) is NOT included in the
message-passing graph -- it is the label-only signal used by CompliancePredictor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Default node types and edge types for the NL knowledge graph
# ---------------------------------------------------------------------------

NL_NODE_TYPES: list[str] = [
    "arxiv_metadata",        # code files + papers (58 nodes)
    "nl_code_smells",        # smell constraints (50 nodes)
    "hecate_specs",          # spec documents (85 nodes)
    "hope_equations",        # equations per paper
    "miras_equations",
    "titans_equations",
    "atlas_equations",
    "tnt_equations",
    "lattice_equations",
    "trellis_equations",
    "hope_abstractions",     # high-level abstractions per paper
    "miras_abstractions",
    "titans_abstractions",
    "atlas_abstractions",
    "tnt_abstractions",
    "lattice_abstractions",
    "trellis_abstractions",
    "hope_axioms",           # axioms per paper
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
    "nl_reframings",         # PyTorch->NL concept mappings (34)
    "hope_nl_reframings",
    "nl_ethnographic_notes", # field notes (63)
    "nl_optimizers",         # optimizer catalog (14)
    "nl_probe_patterns",     # probe patterns (5)
    "arxiv_abstract_chunks", # paper/code chunks (272)
]

# Edge types for message passing (nl_smell_compliance_edges is EXCLUDED -- it's the
# prediction target). Each entry is (src_type, relation_name, dst_type).
NL_EDGE_TYPES: list[tuple[str, str, str]] = [
    # specs -> paper concepts
    ("hecate_specs", "nl_hecate_trace_edges", "hope_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "miras_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "titans_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "atlas_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "tnt_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "lattice_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "trellis_equations"),
    ("hecate_specs", "nl_hecate_trace_edges", "hope_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "miras_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "titans_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "atlas_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "tnt_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "lattice_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "trellis_axioms"),
    ("hecate_specs", "nl_hecate_trace_edges", "nl_axioms"),
    # abstractions -> axioms
    ("hope_abstractions", "nl_axiom_basis_edges", "hope_axioms"),
    ("miras_abstractions", "nl_axiom_basis_edges", "miras_axioms"),
    ("titans_abstractions", "nl_axiom_basis_edges", "titans_axioms"),
    ("atlas_abstractions", "nl_axiom_basis_edges", "atlas_axioms"),
    ("tnt_abstractions", "nl_axiom_basis_edges", "tnt_axioms"),
    ("lattice_abstractions", "nl_axiom_basis_edges", "lattice_axioms"),
    ("trellis_abstractions", "nl_axiom_basis_edges", "trellis_axioms"),
    # definitions -> equations
    ("hope_definitions", "nl_definition_source_edges", "hope_equations"),
    ("miras_definitions", "nl_definition_source_edges", "miras_equations"),
    ("titans_definitions", "nl_definition_source_edges", "titans_equations"),
    # axioms -> nl_axioms (inheritance)
    ("hope_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("miras_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("titans_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("atlas_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("tnt_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("lattice_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    ("trellis_axioms", "nl_axiom_inherits_edges", "nl_axioms"),
    # equations -> chunks (source tracing)
    ("hope_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("miras_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("titans_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("atlas_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("tnt_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("lattice_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    ("trellis_equations", "nl_equation_source_edges", "arxiv_abstract_chunks"),
    # smells -> chunks (source tracing)
    ("nl_code_smells", "nl_smell_source_edges", "arxiv_abstract_chunks"),
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HeteroRGCN(nn.Module):
    """
    Heterogeneous RGCN for the NL knowledge graph.

    Message passing strategy:
      1. Project each node type from 2048d -> hidden_dim (per-type Linear, no bias)
      2. Concatenate all projected nodes into a flat tensor
      3. Run RGCNConv layers (basis decomposition, B=num_bases)
      4. Split back to per-type dicts

    The flattening approach works because RGCNConv already treats each edge type as a
    separate relation -- we just need to remap local per-type node IDs to global IDs.
    """

    def __init__(
        self,
        node_types: list[str] | None = None,
        edge_types: list[tuple[str, str, str]] | None = None,
        in_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bases: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_types = node_types or NL_NODE_TYPES
        self.edge_types = edge_types or NL_EDGE_TYPES
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Per-type input projection: in_dim -> hidden_dim
        self.projections = nn.ModuleDict({
            nt: nn.Linear(in_dim, hidden_dim, bias=False)
            for nt in self.node_types
        })

        num_relations = len(self.edge_types)

        # RGCN layers with basis decomposition
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Map (src_type, rel, dst_type) -> integer relation ID
        self._edge_type_to_id: dict[tuple[str, str, str], int] = {
            et: i for i, et in enumerate(self.edge_types)
        }

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """
        Run RGCN message passing over the heterogeneous graph.

        Args:
            data: PyG HeteroData with .x tensors [N_type, in_dim] per node type
                  and .edge_index tensors [2, E] per edge type.
                  Only node types with a non-None .x attribute are processed.

        Returns:
            Dict mapping node type -> output tensor [N_type, hidden_dim]
        """
        # Step 1: Project each node type into shared embedding space
        projected: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in data.node_types and hasattr(data[nt], "x") and data[nt].x is not None:
                projected[nt] = self.projections[nt](data[nt].x)

        if not projected:
            return {}

        # Step 2: Concatenate all projected nodes; record per-type offsets
        node_offsets: dict[str, int] = {}
        parts: list[torch.Tensor] = []
        offset = 0
        for nt in self.node_types:
            if nt in projected:
                node_offsets[nt] = offset
                parts.append(projected[nt])
                offset += projected[nt].size(0)

        all_h = torch.cat(parts, dim=0)  # [N_total, hidden_dim]
        device = all_h.device

        # Step 3: Build global edge_index and edge_type tensors
        ei_parts: list[torch.Tensor] = []
        et_parts: list[torch.Tensor] = []

        for (src_type, rel, dst_type), rel_id in self._edge_type_to_id.items():
            key = (src_type, rel, dst_type)
            if key not in data.edge_types:
                continue
            ei = data[key].edge_index.clone().to(device)  # [2, E]
            if src_type not in node_offsets or dst_type not in node_offsets:
                continue
            ei[0] += node_offsets[src_type]
            ei[1] += node_offsets[dst_type]
            ei_parts.append(ei)
            et_parts.append(
                torch.full((ei.size(1),), rel_id, dtype=torch.long, device=device)
            )

        if ei_parts:
            edge_index = torch.cat(ei_parts, dim=1)  # [2, E_total]
            edge_type = torch.cat(et_parts, dim=0)   # [E_total]
        else:
            # No matching edges in data -- return projected features unchanged
            return {nt: projected[nt] for nt in projected}

        # Step 4: RGCN message passing with LayerNorm + ReLU + Dropout
        for conv, norm in zip(self.convs, self.norms):
            all_h = conv(all_h, edge_index, edge_type)
            all_h = norm(all_h)
            all_h = F.relu(all_h)
            all_h = F.dropout(all_h, p=self.dropout, training=self.training)

        # Step 5: Split back into per-type output embeddings
        out: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt in node_offsets:
                off = node_offsets[nt]
                n = projected[nt].size(0)
                out[nt] = all_h[off: off + n]

        return out
