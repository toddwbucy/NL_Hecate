"""Tests for CmsTape accumulator (spec 49).

Pure Python tests — no GPU, no nl_hecate import needed.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import sys

# Ensure engine package is importable
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from engine.cms_tape import CmsTape


def _make_tape_summary(step: int, k: int = 4, n_blocks: int = 1, active_levels: list | None = None) -> dict:
    """Build a synthetic tape summary dict."""
    if active_levels is None:
        active_levels = [True] * k
    levels = []
    for lev in range(k):
        levels.append({
            "level": lev,
            "opaque_key": "DeltaRule",
            "block_count": 1 if active_levels[lev] else 0,
            "output_grad_norm": 0.1 * (lev + 1) + step * 0.001,
            "dgd_delta_norm": 0.05 * (lev + 1),
            "m_norm": 10.0 + lev * 2.0 + step * 0.01,
            "freq_gate_value": 0.95 if lev > 0 else float("nan"),
            "is_frozen": not active_levels[lev],
        })
    result = {"loss": 3.0 - step * 0.001, "total_blocks": sum(active_levels), "levels": levels}

    if n_blocks > 1:
        blocks = []
        for bi in range(n_blocks):
            block_levels = []
            for lev in range(k):
                block_levels.append({
                    "level": lev,
                    "m_norm": 10.0 + lev + bi * 0.5,
                    "output_grad_norm": 0.1 * (lev + 1),
                    "dgd_delta_norm": 0.05,
                    "m_shard_diff": 0.01 * bi,
                    "alpha": {"mean": 0.9, "p99": 0.95} if lev < 2 else None,
                    "theta": {"mean": 0.01, "p99": 0.05} if lev < 2 else None,
                })
            blocks.append({"block_index": bi, "levels": block_levels})
        result["blocks"] = blocks
        result["n_blocks"] = n_blocks

    return result


class TestCmsTapeRoundTrip:
    """Spec 49 acceptance: accumulate → flush → verify values match."""

    def test_basic_round_trip(self):
        tape = CmsTape(k=4, chunk_sizes=[1, 8, 64, 512])
        steps = list(range(0, 100, 8))  # every 8 steps
        for s in steps:
            tape.record(_make_tape_summary(s, k=4), s)

        result = tape.flush(0, 100)
        assert result["version"] == 1
        assert result["from_step"] == 0
        assert result["to_step"] == 100
        assert result["k"] == 4
        assert len(result["levels"]) == 4

        # L1-L3 should have all samples (full density)
        for lev in range(1, 4):
            level_data = result["levels"][lev]
            assert level_data["sampled"] == len(steps)
            assert level_data["steps"] == steps

        # After flush, accumulator is empty
        assert len(tape) == 0

    def test_values_match_input(self):
        tape = CmsTape(k=2, l0_sample_rate=1.0)  # no L0 subsampling
        summary = _make_tape_summary(42, k=2)
        tape.record(summary, 42)
        result = tape.flush(0, 42)

        level0 = result["levels"][0]
        assert level0["steps"] == [42]
        assert level0["m_norm"][0] == pytest.approx(10.0 + 42 * 0.01)
        assert level0["output_grad_norm"][0] == pytest.approx(0.1 + 42 * 0.001)


class TestCmsTapeSampling:
    """L0 sampling at reduced rate, L1+ always full density."""

    def test_l0_sampling_rate(self):
        tape = CmsTape(k=2, l0_sample_rate=0.125)  # 1/8
        for s in range(100):
            tape.record(_make_tape_summary(s, k=2), s)

        result = tape.flush(0, 100)
        l0_sampled = result["levels"][0]["sampled"]
        l1_sampled = result["levels"][1]["sampled"]

        # L0 should have ~12-13 samples (100/8)
        assert 10 <= l0_sampled <= 15
        # L1 should have all 100
        assert l1_sampled == 100

    def test_full_density_when_rate_is_1(self):
        tape = CmsTape(k=2, l0_sample_rate=1.0)
        for s in range(50):
            tape.record(_make_tape_summary(s, k=2), s)

        result = tape.flush(0, 50)
        assert result["levels"][0]["sampled"] == 50
        assert result["levels"][1]["sampled"] == 50


class TestCmsTapeProbe:
    """probe() returns data without clearing."""

    def test_probe_does_not_clear(self):
        tape = CmsTape(k=2, l0_sample_rate=1.0)
        for s in range(10):
            tape.record(_make_tape_summary(s, k=2), s)

        assert len(tape) == 10
        snapshot = tape.probe()
        assert len(tape) == 10  # unchanged
        assert snapshot["levels"][0]["sampled"] == 10

    def test_flush_clears(self):
        tape = CmsTape(k=2, l0_sample_rate=1.0)
        for s in range(10):
            tape.record(_make_tape_summary(s, k=2), s)

        tape.flush(0, 10)
        assert len(tape) == 0
        empty = tape.probe()
        assert empty["levels"][0]["sampled"] == 0


class TestCmsTapeSidecar:
    """Sidecar file write/load round-trip."""

    def test_sidecar_write_and_load(self):
        tape = CmsTape(k=2, l0_sample_rate=1.0)
        for s in range(5):
            tape.record(_make_tape_summary(s, k=2), s)

        data = tape.flush(0, 5)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "model_step5.safetensors"
            ckpt.write_text("")  # dummy

            sidecar_path = CmsTape.write_sidecar(ckpt, data)
            assert sidecar_path.name == "model_step5.safetensors.cms.json"
            assert sidecar_path.exists()

            loaded = CmsTape.load_sidecar(ckpt)
            assert loaded is not None
            assert loaded["version"] == 1
            assert loaded["from_step"] == 0
            assert loaded["to_step"] == 5
            assert loaded["levels"][0]["sampled"] == 5

    def test_load_sidecar_missing(self):
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "nonexistent.safetensors"
            assert CmsTape.load_sidecar(ckpt) is None


class TestCmsTapeStacked:
    """Stacked model (n_blocks > 1) per-block data."""

    def test_stacked_block_data(self):
        tape = CmsTape(k=2, n_blocks=4, l0_sample_rate=1.0)
        tape.record(_make_tape_summary(0, k=2, n_blocks=4), 0)
        tape.record(_make_tape_summary(8, k=2, n_blocks=4), 8)

        result = tape.flush(0, 8)
        level0 = result["levels"][0]
        assert "blocks" in level0
        assert len(level0["blocks"]) == 4
        for bi in range(4):
            block = level0["blocks"][bi]
            assert block["block_index"] == bi
            assert len(block["m_norm"]) == 2  # 2 recorded steps


class TestCmsTapeNanHandling:
    """freq_gate_value=NaN should be stored as None in JSON."""

    def test_nan_stored_as_none(self):
        tape = CmsTape(k=1, l0_sample_rate=1.0)
        tape.record(_make_tape_summary(0, k=1), 0)
        result = tape.flush(0, 0)

        # freq_gate_value for level 0 is NaN in the input
        fgv = result["levels"][0]["freq_gate_value"]
        assert len(fgv) == 1
        assert fgv[0] is None  # NaN → None for JSON compatibility

    def test_sidecar_json_serializable(self):
        """Ensure flush output contains no NaN (JSON doesn't support it)."""
        tape = CmsTape(k=2, l0_sample_rate=1.0)
        tape.record(_make_tape_summary(0, k=2), 0)
        data = tape.flush(0, 0)

        # This should not raise — NaN would cause ValueError
        serialized = json.dumps(data)
        assert "NaN" not in serialized
