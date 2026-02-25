#!/usr/bin/env python3
"""Extract Llama-3.2-1B MLP weights for HOPE §7.3 ad-hoc level stacking.

Usage:
    python python/scripts/extract_llama_donor.py [config_json]

If config_json is provided, reads donor_layers from the config.
Otherwise uses the default [0, 5, 10, 15] for k=4 stacking.

Output: checkpoints/llama_mlp_donor.pt
  Keys: "level_0", "level_1", ..., "level_{k-1}"
  Each: {"gate_proj": Tensor[inter x d], "up_proj": Tensor[inter x d],
          "down_proj": Tensor[d x inter]}  (fp32)

Memory: ~768MB for 4 full Llama-3.2-1B MLP blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_donor(
    model_id: str = "unsloth/Llama-3.2-1B",
    donor_layers: list[int] | None = None,
    output_path: str = "checkpoints/llama_mlp_donor.pt",
) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    if donor_layers is None:
        donor_layers = [0, 5, 10, 15]

    print(f"Loading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    donor: dict[str, dict[str, torch.Tensor]] = {}
    for level_idx, layer_idx in enumerate(donor_layers):
        mlp = model.model.layers[layer_idx].mlp
        key = f"level_{level_idx}"
        donor[key] = {
            "gate_proj": mlp.gate_proj.weight.float().cpu().detach().clone(),
            "up_proj":   mlp.up_proj.weight.float().cpu().detach().clone(),
            "down_proj": mlp.down_proj.weight.float().cpu().detach().clone(),
        }
        g = donor[key]["gate_proj"]
        print(f"  level_{level_idx} <- layer {layer_idx}: "
              f"gate_proj={list(g.shape)}, up_proj={list(donor[key]['up_proj'].shape)}, "
              f"down_proj={list(donor[key]['down_proj'].shape)}")

    del model

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(donor, out)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Saved: {out}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Path to config JSON")
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B",
                        help="HF model ID (default: unsloth/Llama-3.2-1B)")
    parser.add_argument("--output", default="checkpoints/llama_mlp_donor.pt",
                        help="Output path for donor weights")
    parser.add_argument("--layers", nargs="+", type=int,
                        help="Layer indices to extract (overrides config)")
    args = parser.parse_args()

    donor_layers = None
    if args.layers:
        donor_layers = args.layers
    elif args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        donor_layers = cfg.get("donor_layers") or cfg.get("model", {}).get("donor_layers")

    extract_donor(
        model_id=args.model,
        donor_layers=donor_layers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
