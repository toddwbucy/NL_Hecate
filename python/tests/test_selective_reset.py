#!/usr/bin/env python3
"""Tests for selective periodic reset (spec 57).

Verifies:
1. Backward compat: reset_intervals=None with memory_reset=True behaves like spec-08
2. Selective: [1,8,64,512] only resets levels at their interval boundaries
3. M persistence: higher levels retain M between resets
4. Validation: bad intervals raise errors
5. Parity: explicit [1,1,1,1] == None (all-ones default)
"""

import pytest
import nl_hecate


def _make_cfg(k=4):
    """Minimal k=4 MAGConfig for testing.
    seq_len=512 ensures divisibility by all chunk_sizes (1, 8, 64, 512).
    """
    return nl_hecate.MAGConfig(
        32, 2, 16, 512, 512, 64, True,
        k=k,
        chunk_sizes=[1, 8, 64, 512][:k],
        memory_rule="titans",
        composition="mag",
        parallel_strategy="tnt_hierarchical",
    )


def _make_stacked_model(k=4, memory_reset=True, reset_intervals=None):
    """Create a small stacked model for testing."""
    cfg = _make_cfg(k)
    return nl_hecate.GpuStackedModel(
        cfg, n_blocks=2, seed=42, batch_size=1,
        memory_reset=memory_reset, reset_intervals=reset_intervals,
    ), cfg


def _random_batch(cfg):
    """Generate random input/target token IDs."""
    import random
    sl = cfg.seq_len
    v = cfg.vocab_size
    input_ids = [random.randint(0, v - 1) for _ in range(sl)]
    target_ids = [random.randint(0, v - 1) for _ in range(sl)]
    return input_ids, target_ids


class TestBackwardCompat:
    """reset_intervals=None with memory_reset=True → spec-08 behavior (all-ones)."""

    def test_none_intervals_resets_every_step(self):
        """With default intervals, all levels reset every step (spec-08)."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=None)
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        inp, tgt = _random_batch(cfg)

        # Run one step
        pulse = conductor.pulse()
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        # After spec-08 reset, all levels should have M≈0
        # memory_norms_live() reads current GPU M state (post-reset)
        norms = model.memory_norms_live()
        for i, norm in enumerate(norms):
            assert norm < 1e-3, f"L{i} expected M≈0 after all-ones reset, got {norm}"


class TestSelectiveReset:
    """reset_intervals=[1,8,64,512] preserves M for higher levels."""

    def test_l1_persists_after_first_fire(self):
        """L1 M should persist after step 0 since fire_count(1) < interval(8)."""
        model, cfg = _make_stacked_model(
            memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

        # Step 0: all levels fire (step % chunk_size == 0 for all).
        # With intervals=[1,8,64,512]:
        #   L0: fire_count→1, 1>=1 → RESET, counter→0
        #   L1: fire_count→1, 1<8 → NO reset
        #   L2: fire_count→1, 1<64 → NO reset
        #   L3: fire_count→1, 1<512 → NO reset
        inp, tgt = _random_batch(cfg)
        pulse = conductor.pulse()
        # Verify all levels active at step 0
        assert all(pulse.active_levels), f"Expected all active at step 0: {pulse.active_levels}"
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        # Use memory_norms_live() for post-reset GPU M state
        norms = model.memory_norms_live()
        # L0 should be reset (interval=1)
        assert norms[0] < 1e-3, f"L0 should be reset, got {norms[0]}"
        # L1 should NOT be reset (interval=8, only 1 fire).
        # At d=32 with random init, L1 M norm won't be exactly zero even with
        # small gate biases — the key invariant is that it's NOT zeroed.
        assert norms[1] > 1e-6, f"L1 M should persist (not zeroed), got {norms[1]}"

    def test_l0_always_resets(self):
        """L0 with interval=1 should always reset (same as spec-08 for L0)."""
        model, cfg = _make_stacked_model(
            memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

        for step in range(5):
            inp, tgt = _random_batch(cfg)
            pulse = conductor.pulse()
            model.step_adamw(inp, tgt, pulse, 0.001)
            conductor.advance()
            # memory_norms_live() reads post-reset GPU state
            norms = model.memory_norms_live()
            assert norms[0] < 1e-3, f"L0 should reset every step, got norm={norms[0]} at step {step}"

    def test_l1_resets_at_interval_boundary(self):
        """L1 should reset after 8 fires (fire_count reaches interval=8)."""
        model, cfg = _make_stacked_model(
            memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

        # With chunk_sizes=[1,8,64,512], L1 fires at steps 0,8,16,24...
        # We need 8 fires of L1 → steps 0,8,16,24,32,40,48,56
        # At the 8th fire (step 56), fire_count reaches 8 → RESET
        l1_norms = []
        for step in range(64):
            inp, tgt = _random_batch(cfg)
            pulse = conductor.pulse()
            model.step_adamw(inp, tgt, pulse, 0.001)
            conductor.advance()
            if pulse.active_levels[1]:  # L1 fired
                # Use live norms to see post-reset state
                norms = model.memory_norms_live()
                l1_norms.append((step, norms[1]))

        # Should have fired at steps 0,8,16,24,32,40,48,56
        assert len(l1_norms) == 8, f"Expected 8 L1 fires, got {len(l1_norms)}"
        # After the 8th fire (step 56), L1 should be reset
        assert l1_norms[-1][1] < 1e-3, \
            f"L1 should reset at 8th fire (step {l1_norms[-1][0]}), got norm={l1_norms[-1][1]}"
        # Before the 8th fire, L1 should NOT be reset (norms may be small but accumulating)
        print(f"L1 norms at each fire: {l1_norms}")


class TestNoReset:
    """memory_reset=False → no reset regardless of intervals."""

    def test_no_reset_preserves_m(self):
        model, cfg = _make_stacked_model(memory_reset=False, reset_intervals=None)
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        inp, tgt = _random_batch(cfg)

        pulse = conductor.pulse()
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        # memory_norms_live() for current M state (no reset applied)
        norms = model.memory_norms_live()
        # Without reset, L0 M should persist (non-zero after forward)
        assert norms[0] > 0, f"Expected M > 0 without reset, got {norms[0]}"


class TestValidation:
    """Bad inputs should raise ValueError."""

    def test_wrong_length(self):
        with pytest.raises(Exception, match="reset_intervals length"):
            _make_stacked_model(memory_reset=True, reset_intervals=[1, 8])

    def test_zero_interval(self):
        with pytest.raises(Exception, match="must be >= 1"):
            _make_stacked_model(memory_reset=True, reset_intervals=[1, 0, 64, 512])


class TestParity:
    """Verify that intervals=[1,1,1,1] produces identical results to None."""

    def test_explicit_ones_equals_none(self):
        """Explicit [1,1,1,1] should behave identically to None (all-ones default)."""
        import random
        random.seed(123)

        # Model A: intervals=None (default)
        model_a, cfg_a = _make_stacked_model(memory_reset=True, reset_intervals=None)
        cond_a = nl_hecate.Conductor(cfg_a.k, list(cfg_a.chunk_sizes))

        # Model B: intervals=[1,1,1,1] (explicit)
        model_b, cfg_b = _make_stacked_model(memory_reset=True, reset_intervals=[1, 1, 1, 1])
        cond_b = nl_hecate.Conductor(cfg_b.k, list(cfg_b.chunk_sizes))

        for step in range(10):
            random.seed(42 + step)
            inp, tgt = _random_batch(cfg_a)

            pa = cond_a.pulse()
            _, _ = model_a.step_adamw(inp, tgt, pa, 0.001)
            cond_a.advance()

            pb = cond_b.pulse()
            _, _ = model_b.step_adamw(inp, tgt, pb, 0.001)
            cond_b.advance()

            # Use live norms for both — they should be identical
            norms_a = model_a.memory_norms_live()
            norms_b = model_b.memory_norms_live()

            for li in range(len(norms_a)):
                assert abs(norms_a[li] - norms_b[li]) < 1e-3, \
                    f"Step {step} L{li}: {norms_a[li]} vs {norms_b[li]}"


class TestFireCountsAPI:
    """Verify get/set fire_counts for roundtrip verification (spec 57)."""

    def test_get_fire_counts_initial(self):
        """Fire counts start at zero."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=[1, 8, 64, 512])
        counts = model.get_fire_counts()
        assert counts == [0, 0, 0, 0], f"Expected all zeros, got {counts}"

    def test_fire_counts_advance(self):
        """Fire counts advance after step_adamw."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        inp, tgt = _random_batch(cfg)

        pulse = conductor.pulse()
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        counts = model.get_fire_counts()
        # L0 interval=1: fires and resets → back to 0
        assert counts[0] == 0, f"L0 should reset to 0, got {counts[0]}"
        # L1 interval=8: fires once → count=1
        assert counts[1] == 1, f"L1 should be 1, got {counts[1]}"

    def test_set_fire_counts_restores(self):
        """set_fire_counts restores saved state after multiple advances."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))

        # Advance 3 steps so fire_counts are meaningfully non-zero.
        # At step 0 all levels fire, so L1/L2/L3 counts become 1.
        for _ in range(3):
            inp, tgt = _random_batch(cfg)
            pulse = conductor.pulse()
            model.step_adamw(inp, tgt, pulse, 0.001)
            conductor.advance()

        saved = model.get_fire_counts()
        assert any(c > 0 for c in saved), f"Expected non-zero counts after 3 steps, got {saved}"

        # Advance past step 8 (L1 fires again at step 8) to guarantee counts change.
        # Steps 3..11 = 9 more steps; L1 fires at step 8 → count goes from 1 to 2.
        for _ in range(9):
            inp, tgt = _random_batch(cfg)
            pulse = conductor.pulse()
            model.step_adamw(inp, tgt, pulse, 0.001)
            conductor.advance()
        assert model.get_fire_counts() != saved, "Counts should have changed after crossing L1 fire boundary"

        # Restore and verify
        model.set_fire_counts(saved)
        assert model.get_fire_counts() == saved

    def test_set_fire_counts_wrong_length(self):
        """Wrong length raises ValueError."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=[1, 8, 64, 512])
        with pytest.raises(Exception, match="fire_counts length"):
            model.set_fire_counts([0, 0])


class TestResetContextClearsFireCounts:
    """Verify reset_context() zeros fire_counts (finding 1, spec 57)."""

    def test_reset_context_zeros_fire_counts(self):
        """reset_context() should zero fire_counts since M is being hard-reset."""
        model, cfg = _make_stacked_model(memory_reset=True, reset_intervals=[1, 8, 64, 512])
        conductor = nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))
        inp, tgt = _random_batch(cfg)

        pulse = conductor.pulse()
        model.step_adamw(inp, tgt, pulse, 0.001)
        conductor.advance()

        # fire_counts should be non-zero for L1+ after one step
        counts = model.get_fire_counts()
        assert counts[1] > 0, f"L1 should have fired, got {counts[1]}"

        # reset_context should zero both M and fire_counts
        model.reset_context()
        counts = model.get_fire_counts()
        assert counts == [0, 0, 0, 0], f"Expected all zeros after reset_context, got {counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
