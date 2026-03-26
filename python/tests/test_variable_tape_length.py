#!/usr/bin/env python3
"""Tests for variable tape length (spec 59).

Validates that the full forward+backward+optimizer pipeline works at
seq_len values beyond the baseline 512: {1024, 2048, 4096}.
Uses a tiny model (d=32) so tests run quickly without GPU memory pressure.
"""

import math
import random

import pytest
import nl_hecate


def _make_cfg(seq_len, k=4):
    """Create a minimal MAGConfig at the given seq_len.

    chunk_sizes=[1, 8, 64, 512] requires seq_len % 512 == 0.
    """
    chunk_sizes = [1, 8, 64, 512][:k]
    return nl_hecate.MAGConfig(
        32, 2, 16, seq_len, 512, 64, True,
        k=k,
        chunk_sizes=chunk_sizes,
        memory_rule="titans",
        composition="mag",
        parallel_strategy="tnt_hierarchical",
    )


def _make_model(seq_len, k=4, reset_intervals=None):
    """Create a stacked model at the given seq_len."""
    cfg = _make_cfg(seq_len, k)
    model = nl_hecate.GpuStackedModel(
        cfg, n_blocks=2, seed=42, batch_size=1,
        memory_reset=True, reset_intervals=reset_intervals,
    )
    return model, cfg


def _random_batch(cfg):
    """Generate random input/target token IDs."""
    sl = cfg.seq_len
    v = cfg.vocab_size
    return (
        [random.randint(0, v - 1) for _ in range(sl)],
        [random.randint(0, v - 1) for _ in range(sl)],
    )


class TestPipelineAtVariousSeqLen:
    """Forward+backward+optimizer should work at seq_len > 512."""

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_step_produces_finite_loss(self, seq_len):
        """One step_adamw at each seq_len should produce finite loss."""
        model, cfg = _make_model(seq_len)
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        random.seed(42)
        inp, tgt = _random_batch(cfg)

        pulse = conductor.pulse()
        loss, gnorm = model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        assert math.isfinite(loss), f"Loss not finite at seq_len={seq_len}: {loss}"
        assert math.isfinite(gnorm), f"Gnorm not finite at seq_len={seq_len}: {gnorm}"
        assert loss > 0, f"Loss should be positive, got {loss}"

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    def test_memory_norms_valid(self, seq_len):
        """memory_norms() should return k finite values after one step."""
        model, cfg = _make_model(seq_len, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        random.seed(42)
        inp, tgt = _random_batch(cfg)

        pulse = conductor.pulse()
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        norms = model.memory_norms()
        assert len(norms) == cfg.k, f"Expected {cfg.k} norms, got {len(norms)}"
        for i, n in enumerate(norms):
            assert math.isfinite(n), f"L{i} norm not finite at seq_len={seq_len}: {n}"


class TestWriteBudgetScaling:
    """Verify that longer tapes give higher levels more fire opportunities."""

    def test_l1_fires_more_at_longer_seq_len(self):
        """L1 (chunk_size=8) should fire more times at seq_len=1024 vs 512."""
        for seq_len, expected_l1_fires in [(512, 64), (1024, 128), (2048, 256)]:
            cfg = _make_cfg(seq_len)
            conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

            l1_fires = 0
            for _ in range(seq_len):
                pulse = conductor.pulse()
                if pulse.active_levels[1]:
                    l1_fires += 1
                conductor.advance()

            assert l1_fires == expected_l1_fires, \
                f"seq_len={seq_len}: expected {expected_l1_fires} L1 fires, got {l1_fires}"

    def test_all_levels_fire_at_step_zero(self):
        """All levels should fire at step 0 regardless of seq_len."""
        for seq_len in [512, 1024, 2048, 4096]:
            cfg = _make_cfg(seq_len)
            conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
            pulse = conductor.pulse()
            assert all(pulse.active_levels), \
                f"Not all levels active at step 0, seq_len={seq_len}: {pulse.active_levels}"


class TestDivisibilityConstraint:
    """seq_len must be divisible by all chunk_sizes."""

    def test_l3_fires_scale_with_seq_len(self):
        """L3 fires exactly seq_len/512 times per tape at divisible lengths."""
        for seq_len, expected in [(512, 1), (1024, 2), (2048, 4), (4096, 8)]:
            conductor = nl_hecate.Conductor(4, [1, 8, 64, 512])
            l3_fires = 0
            for _ in range(seq_len):
                pulse = conductor.pulse()
                if pulse.active_levels[3]:
                    l3_fires += 1
                conductor.advance()
            assert l3_fires == expected, \
                f"seq_len={seq_len}: expected {expected} L3 fires, got {l3_fires}"

    def test_divisible_seq_len_accepted(self):
        """seq_len=2048 is divisible by 512 — should work."""
        cfg = _make_cfg(2048, k=4)
        assert cfg.seq_len == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
