"""Donor weight loading for HOPE §7.3 ad-hoc level stacking.

Loads pre-extracted Llama MLP weights into MAGParams for SwiGluMlp levels.
Weights are extracted by python/scripts/extract_llama_donor.py and stored
as a .pt file with keys "level_0", "level_1", ... each containing a dict
with "gate_proj", "up_proj", "down_proj" tensors (fp32).
"""

from __future__ import annotations


def load_llama_donor(donor_path: str, params: object, cfg: object, k: int) -> None:
    """Load extracted Llama MLP weights into MAGParams levels 0..k-1.

    Args:
        donor_path: Path to .pt file produced by extract_llama_donor.py.
        params: MAGParams PyO3 object with set_level_mlp() method.
        cfg: MAGConfig PyO3 object (needed by set_level_mlp for dimension validation).
        k: Number of CMS levels to populate.
    """
    import torch
    donor = torch.load(donor_path, weights_only=True)
    for level in range(k):
        key = f"level_{level}"
        if key not in donor:
            raise KeyError(
                f"Donor file '{donor_path}' missing key '{key}'. "
                f"Expected keys: {[f'level_{i}' for i in range(k)]}"
            )
        d = donor[key]
        params.set_level_mlp(
            cfg,
            level,
            d["gate_proj"].float().numpy().flatten().tolist(),
            d["up_proj"].float().numpy().flatten().tolist(),
            d["down_proj"].float().numpy().flatten().tolist(),
        )
