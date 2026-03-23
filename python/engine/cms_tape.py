"""CMS Tape Accumulator — cross-step CMS level metric collection.

Spec 49: specs/infrastructure/differentiation/05_cms_observability_application.md

Accumulates per-level metrics from tape summary calls across training steps,
then flushes as .cms.json sidecar files alongside checkpoints. Also provides
a live probe API for Jupyter/callback consumers.

CS-32: Observe-then-advance — accumulator only reads tape summary dicts.
CS-40: Opt-in — accumulator is only active when tape_device != "off".
CS-10: No train/eval — data comes from the build stream's tape diagnostic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# Sidecar schema version — bump when changing the JSON structure
_SCHEMA_VERSION = 1

# Per-level fields extracted from each tape summary
_LEVEL_FIELDS = ("m_norm", "output_grad_norm", "dgd_delta_norm", "freq_gate_value")

# Per-level boolean fields
_LEVEL_BOOL_FIELDS = ("is_frozen",)

# Per-block fields (stacked models only)
_BLOCK_FIELDS = ("m_norm", "output_grad_norm", "dgd_delta_norm",
                 "m_shard_diff", "alpha_mean", "theta_mean")


class CmsTape:
    """Accumulates per-level CMS metrics across training steps.

    Usage:
        tape = CmsTape(k=4, n_blocks=4, num_heads=16,
                        chunk_sizes=[1, 8, 64, 512])
        # Each checkpoint interval:
        tape.record(tape_summary_dict, step)  # called at each tape diagnostic
        sidecar = tape.flush(from_step, to_step)  # at checkpoint time
        Path("model.safetensors.cms.json").write_text(json.dumps(sidecar))
    """

    def __init__(
        self,
        k: int,
        n_blocks: int = 1,
        num_heads: int = 1,
        chunk_sizes: list[int] | None = None,
        l0_sample_rate: float = 0.125,
        l0_per_head_sample_rate: float = 0.0625,
    ):
        if not (0 < l0_sample_rate <= 1.0):
            raise ValueError(
                f"l0_sample_rate must be in (0, 1], got {l0_sample_rate}")
        if not (0 < l0_per_head_sample_rate <= 1.0):
            raise ValueError(
                f"l0_per_head_sample_rate must be in (0, 1], got {l0_per_head_sample_rate}")

        self.k = k
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        self.chunk_sizes = chunk_sizes or [1] + [8 ** i for i in range(1, k)]
        self.l0_sample_rate = l0_sample_rate
        self.l0_per_head_sample_rate = l0_per_head_sample_rate

        # Compute L0 sampling periods (deterministic modular, not random)
        self._l0_period = 1 if l0_sample_rate == 1.0 else max(1, round(1.0 / l0_sample_rate))
        self._l0_per_head_period = (1 if l0_per_head_sample_rate == 1.0
                                    else max(1, round(1.0 / l0_per_head_sample_rate)))

        # Per-level accumulators: parallel arrays
        self._levels: list[dict[str, list]] = []
        for _ in range(k):
            level_data: dict[str, list] = {"steps": []}
            for f in _LEVEL_FIELDS:
                level_data[f] = []
            for f in _LEVEL_BOOL_FIELDS:
                level_data[f] = []
            # Per-head norms: separate step index (different sampling rate for L0)
            level_data["head_steps"] = []
            level_data["head_m_norms"] = []
            # Per-block arrays (only populated for stacked models)
            if n_blocks > 1:
                level_data["blocks"] = [
                    {**{f: [] for f in _BLOCK_FIELDS},
                     "head_m_norms": []}
                    for _ in range(n_blocks)
                ]
            self._levels.append(level_data)

        self._count = 0

    def record(self, tape_summary: dict[str, Any], step: int) -> None:
        """Append one tape summary to the accumulator.

        Respects sampling: L0 records at l0_sample_rate, L1+ always full density.
        """
        levels = tape_summary.get("levels", [])
        blocks = tape_summary.get("blocks")

        for lvl_dict in levels:
            level_idx = lvl_dict.get("level", 0)
            if level_idx >= self.k:
                continue

            # L0 sampling: skip if not on the sampling cadence
            if level_idx == 0 and self._l0_period > 1:
                if step % self._l0_period != 0:
                    continue

            acc = self._levels[level_idx]
            acc["steps"].append(step)
            for f in _LEVEL_FIELDS:
                val = lvl_dict.get(f, float("nan"))
                acc[f].append(val if not (isinstance(val, float) and math.isnan(val))
                              else None)
            for f in _LEVEL_BOOL_FIELDS:
                acc[f].append(lvl_dict.get(f, False))

            # Per-head norms: separate sampling gate for L0
            head_norms = lvl_dict.get("head_m_norms", [])
            if head_norms:
                record_head = True
                if level_idx == 0 and self._l0_per_head_period > 1:
                    record_head = (step % self._l0_per_head_period == 0)
                if record_head:
                    acc["head_steps"].append(step)
                    acc["head_m_norms"].append(list(head_norms))

        # Per-block data (stacked models)
        if blocks and self.n_blocks > 1:
            for block_dict in blocks:
                bi = block_dict.get("block_index", 0)
                if bi >= self.n_blocks:
                    continue
                for blvl in block_dict.get("levels", []):
                    level_idx = blvl.get("level", 0)
                    if level_idx >= self.k:
                        continue
                    # Same L0 sampling gate
                    if level_idx == 0 and self._l0_period > 1:
                        if step % self._l0_period != 0:
                            continue
                    block_acc = self._levels[level_idx]["blocks"][bi]
                    for f in _BLOCK_FIELDS:
                        if f == "alpha_mean":
                            alpha = blvl.get("alpha")
                            block_acc[f].append(alpha["mean"] if alpha else None)
                        elif f == "theta_mean":
                            theta = blvl.get("theta")
                            block_acc[f].append(theta["mean"] if theta else None)
                        else:
                            block_acc[f].append(blvl.get(f))
                    # Per-head norms at block level (same sampling gate as aggregate)
                    blk_head = blvl.get("head_m_norms", [])
                    if blk_head:
                        record_head = True
                        if level_idx == 0 and self._l0_per_head_period > 1:
                            record_head = (step % self._l0_per_head_period == 0)
                        if record_head:
                            block_acc["head_m_norms"].append(list(blk_head))

        self._count += 1

    def flush(self, from_step: int, to_step: int) -> dict[str, Any]:
        """Return accumulated data as sidecar-ready dict and reset.

        The returned dict can be directly serialized to .cms.json.
        """
        result: dict[str, Any] = {
            "version": _SCHEMA_VERSION,
            "from_step": from_step,
            "to_step": to_step,
            "k": self.k,
            "n_blocks": self.n_blocks,
            "num_heads": self.num_heads,
            "chunk_sizes": self.chunk_sizes,
            "l0_sample_rate": self.l0_sample_rate,
            "levels": [],
        }

        for lev in range(self.k):
            acc = self._levels[lev]
            sampled = len(acc["steps"])
            level_out: dict[str, Any] = {
                "level": lev,
                "sampled": sampled,
                "steps": acc["steps"],
            }
            for f in _LEVEL_FIELDS:
                level_out[f] = acc[f]
            for f in _LEVEL_BOOL_FIELDS:
                level_out[f] = acc[f]
            if acc["head_m_norms"]:
                level_out["head_steps"] = acc["head_steps"]
                level_out["head_m_norms"] = acc["head_m_norms"]

            if self.n_blocks > 1 and "blocks" in acc:
                blocks_out = []
                for bi in range(self.n_blocks):
                    block_data = {"block_index": bi}
                    for f in _BLOCK_FIELDS:
                        block_data[f] = acc["blocks"][bi][f]
                    if acc["blocks"][bi]["head_m_norms"]:
                        block_data["head_m_norms"] = acc["blocks"][bi]["head_m_norms"]
                    blocks_out.append(block_data)
                level_out["blocks"] = blocks_out

            result["levels"].append(level_out)

        # Reset accumulators
        self._reset()

        return result

    def probe(self) -> dict[str, Any]:
        """Return current accumulator state without clearing.

        Read-only view for live probes, Jupyter, callbacks.
        """
        # Build snapshot without resetting (use flush's format but skip reset)
        result: dict[str, Any] = {
            "version": _SCHEMA_VERSION,
            "k": self.k,
            "n_blocks": self.n_blocks,
            "samples": self._count,
            "levels": [],
        }
        for lev in range(self.k):
            acc = self._levels[lev]
            level_out: dict[str, Any] = {
                "level": lev,
                "sampled": len(acc["steps"]),
                "steps": list(acc["steps"]),
            }
            for f in _LEVEL_FIELDS:
                level_out[f] = list(acc[f])
            for f in _LEVEL_BOOL_FIELDS:
                level_out[f] = list(acc[f])
            if acc["head_m_norms"]:
                level_out["head_steps"] = list(acc["head_steps"])
                level_out["head_m_norms"] = [list(h) for h in acc["head_m_norms"]]
            if self.n_blocks > 1 and "blocks" in acc:
                blocks_out = []
                for bi in range(self.n_blocks):
                    block_data = {"block_index": bi}
                    for f in _BLOCK_FIELDS:
                        block_data[f] = list(acc["blocks"][bi][f])
                    if acc["blocks"][bi]["head_m_norms"]:
                        block_data["head_m_norms"] = [
                            list(h) for h in acc["blocks"][bi]["head_m_norms"]]
                    blocks_out.append(block_data)
                level_out["blocks"] = blocks_out
            result["levels"].append(level_out)
        return result

    def __len__(self) -> int:
        """Number of recorded tape summary calls."""
        return self._count

    def _reset(self) -> None:
        """Clear all accumulators."""
        for lev in range(self.k):
            acc = self._levels[lev]
            acc["steps"] = []
            for f in _LEVEL_FIELDS:
                acc[f] = []
            for f in _LEVEL_BOOL_FIELDS:
                acc[f] = []
            acc["head_steps"] = []
            acc["head_m_norms"] = []
            if self.n_blocks > 1 and "blocks" in acc:
                for bi in range(self.n_blocks):
                    for f in _BLOCK_FIELDS:
                        acc["blocks"][bi][f] = []
                    acc["blocks"][bi]["head_m_norms"] = []
        self._count = 0

    @staticmethod
    def write_sidecar(ckpt_path: str | Path, data: dict[str, Any]) -> Path:
        """Write .cms.json sidecar alongside a checkpoint file."""
        sidecar = Path(str(ckpt_path) + ".cms.json")
        sidecar.write_text(json.dumps(data))
        return sidecar

    @staticmethod
    def load_sidecar(ckpt_path: str | Path) -> dict[str, Any] | None:
        """Load .cms.json sidecar if it exists, else None."""
        sidecar = Path(str(ckpt_path) + ".cms.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text())
        return None
