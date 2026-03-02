"""Tests for nl_hecate PyO3 bindings."""

import math
import pytest
import nl_hecate


# ── Config tests ──────────────────────────────────────────────────────

def test_config_creation():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    assert cfg.d_model == 64
    assert cfg.num_heads == 4
    assert cfg.head_dim == 16
    assert cfg.seq_len == 24
    assert cfg.window_size == 16
    assert cfg.vocab_size == 256


def test_config_via_create_config():
    cfg = nl_hecate.create_config(64, 4, 16, 24, 16, 256)
    assert cfg.d_model == 64


def test_config_invalid_head_dim():
    with pytest.raises(ValueError, match="d_model"):
        nl_hecate.SWAConfig(64, 4, 15, 24, 16, 256)  # 4*15=60 != 64


# ── Params tests ──────────────────────────────────────────────────────

def test_params_init():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    params = nl_hecate.init_params(cfg, 42)
    assert params.num_params() > 0


def test_params_deterministic():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    p1 = nl_hecate.init_params(cfg, 42)
    p2 = nl_hecate.init_params(cfg, 42)
    assert p1.num_params() == p2.num_params()


# ── Forward tests ─────────────────────────────────────────────────────

def _make_test_data(cfg):
    input_ids = list(range(cfg.seq_len))
    target_ids = [(t + 1) % cfg.vocab_size for t in range(cfg.seq_len)]
    return input_ids, target_ids


def test_forward_returns_loss():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_test_data(cfg)
    loss, cache = nl_hecate.forward(params, cfg, input_ids, target_ids)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0.0
    # Random init loss should be near ln(256) ≈ 5.55
    assert loss < 20.0


def test_forward_deterministic():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_test_data(cfg)
    loss1, _ = nl_hecate.forward(params, cfg, input_ids, target_ids)
    loss2, _ = nl_hecate.forward(params, cfg, input_ids, target_ids)
    assert loss1 == loss2


# ── Backward / gradient tests ────────────────────────────────────────

def test_backward_returns_grads():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_test_data(cfg)
    _loss, cache = nl_hecate.forward(params, cfg, input_ids, target_ids)
    grads = nl_hecate.backward(params, cfg, cache, input_ids, target_ids)
    assert isinstance(grads, nl_hecate.SWAParams)
    assert grads.num_params() == params.num_params()


def test_compute_gradients():
    cfg = nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_test_data(cfg)
    loss_cg, grads = nl_hecate.compute_gradients(params, cfg, input_ids, target_ids)
    loss_fwd, _cache = nl_hecate.forward(params, cfg, input_ids, target_ids)
    assert loss_cg == loss_fwd
    assert grads.num_params() == params.num_params()


# ── MAG Config tests ─────────────────────────────────────────────

def test_mag_config_creation():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    assert cfg.d_model == 16
    assert cfg.num_heads == 2
    assert cfg.head_dim == 8
    assert cfg.seq_len == 8
    assert cfg.window_size == 8
    assert cfg.vocab_size == 64
    assert cfg.memory_enabled is True


def test_mag_config_via_create():
    cfg = nl_hecate.mag_create_config(16, 2, 8, 8, 8, 64, True)
    assert cfg.d_model == 16


def test_mag_config_invalid_head_dim():
    with pytest.raises(ValueError, match="d_model"):
        nl_hecate.MAGConfig(16, 2, 7, 8, 8, 64, True)  # 2*7=14 != 16


# ── MAG Params tests ─────────────────────────────────────────────

def test_mag_params_init():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    params = nl_hecate.mag_init_params(cfg, 42)
    # MAG has more params than SWA (7 extra memory weight fields)
    assert params.num_params() > 0


def test_mag_params_get_weights():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    params = nl_hecate.mag_init_params(cfg, 42)
    w = params.get_weights()
    expected_keys = {
        "w_embed", "w_q", "w_k", "w_v", "w_o", "w_unembed",
        "w_k_mem", "w_v_mem", "w_q_mem", "w_alpha", "b_alpha", "w_theta", "b_theta",
    }
    assert set(w.keys()) == expected_keys
    d = 16
    v = 64
    assert len(w["w_embed"]) == v * d
    assert len(w["w_q"]) == d * d
    assert len(w["w_k_mem"]) == d * d
    assert len(w["w_v_mem"]) == d * d
    assert len(w["w_q_mem"]) == d * d
    assert len(w["w_alpha"]) == 2 * d
    assert len(w["b_alpha"]) == 1
    assert len(w["w_theta"]) == 2 * d
    assert len(w["b_theta"]) == 1


# ── MAG Forward tests ────────────────────────────────────────────

def _make_mag_test_data(cfg):
    input_ids = list(range(cfg.seq_len))
    target_ids = [(t + 1) % cfg.vocab_size for t in range(cfg.seq_len)]
    return input_ids, target_ids


def test_mag_forward_returns_loss():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    params = nl_hecate.mag_init_params(cfg, 42)
    input_ids, target_ids = _make_mag_test_data(cfg)
    loss, _cache = nl_hecate.mag_forward(params, cfg, input_ids, target_ids)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0.0
    # Random init loss near ln(64) ≈ 4.16
    assert loss < 20.0


def test_mag_forward_deterministic():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    params = nl_hecate.mag_init_params(cfg, 42)
    input_ids, target_ids = _make_mag_test_data(cfg)
    loss1, _ = nl_hecate.mag_forward(params, cfg, input_ids, target_ids)
    loss2, _ = nl_hecate.mag_forward(params, cfg, input_ids, target_ids)
    assert loss1 == loss2


# ── MAG Backward / gradient tests ────────────────────────────────

def test_mag_compute_gradients():
    cfg = nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)
    params = nl_hecate.mag_init_params(cfg, 42)
    input_ids, target_ids = _make_mag_test_data(cfg)
    loss, grads = nl_hecate.mag_compute_gradients(params, cfg, input_ids, target_ids)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert isinstance(grads, nl_hecate.MAGParams)
    assert grads.num_params() == params.num_params()


# ── Vocabulary projection probe (logit lens for CMS memory) ───────────

class _FakeTok:
    """Minimal tokenizer stub: decode returns human-readable placeholder."""
    def decode(self, ids):
        return f"t{ids[0]}"


def _make_probe_fixtures(d=16, vocab=64, k=2):
    """Create a minimal MAGConfig + MAGParams + ContextState for probe tests."""
    cfg = nl_hecate.MAGConfig(d, 2, d // 2, 8, 8, vocab, True, k=k)
    params = nl_hecate.mag_init_params(cfg, 42)
    ctx = nl_hecate.ContextState(k, d)
    # Fill memory with non-zero values so the probe produces a non-trivial distribution.
    import random
    rng = random.Random(99)
    filled = [[rng.gauss(0, 0.1) for _ in range(d * d)] for _ in range(k)]
    ctx.set_memory(filled)
    return cfg, params, ctx


def test_memory_vocab_probe_structure():
    """probe_memory_vocab returns expected structure for k=2 levels."""
    from engine.evaluation import probe_memory_vocab
    cfg, params, ctx = _make_probe_fixtures(d=16, vocab=64, k=2)
    tok = _FakeTok()
    result = probe_memory_vocab(params, ctx, cfg, tok, step=100)

    assert result["step"] == 100
    assert len(result["levels"]) == 2
    assert len(result["js_divergence"]) == 1   # C(2,2) = 1 pair

    for lv in result["levels"]:
        assert "level" in lv
        assert "m_norm" in lv
        assert "top20" in lv
        assert len(lv["top20"]) == 20
        for entry in lv["top20"]:
            assert "id" in entry and "prob" in entry and "tok" in entry
            assert 0.0 <= entry["prob"] <= 1.0

    for pair in result["js_divergence"]:
        assert "levels" in pair and "js_div" in pair
        assert pair["js_div"] >= 0.0


def test_memory_vocab_probe_zero_m_skips_top20():
    """Level with zero M (fresh context) emits empty top20 (m_norm guard)."""
    from engine.evaluation import probe_memory_vocab
    cfg, params, ctx = _make_probe_fixtures(d=16, vocab=64, k=2)
    # Overwrite level 0 with zeros to simulate unactivated level.
    mem = ctx.memory
    mem[0] = [0.0] * (16 * 16)
    ctx.set_memory(mem)
    tok = _FakeTok()
    result = probe_memory_vocab(params, ctx, cfg, tok, step=0)
    assert result["levels"][0]["top20"] == []
    assert len(result["levels"][1]["top20"]) == 20


def test_memory_vocab_probe_probs_sum_to_one():
    """Probability distributions from the probe sum to ≈1.0."""
    from engine.evaluation import probe_memory_vocab
    cfg, params, ctx = _make_probe_fixtures(d=16, vocab=64, k=2)
    tok = _FakeTok()
    result = probe_memory_vocab(params, ctx, cfg, tok, step=50)
    for lv in result["levels"]:
        total_prob = sum(e["prob"] for e in lv["top20"])
        # top-20 of 64 tokens; remaining 44 tokens hold the leftover probability.
        assert total_prob <= 1.0 + 1e-4
