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
