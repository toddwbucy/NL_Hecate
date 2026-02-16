"""End-to-end training loop tests for nl_hecate."""

import nl_hecate


def _make_config():
    return nl_hecate.SWAConfig(64, 4, 16, 24, 16, 256)


def _make_data(cfg):
    input_ids = list(range(cfg.seq_len))
    target_ids = [(t + 1) % cfg.vocab_size for t in range(cfg.seq_len)]
    return input_ids, target_ids


def test_sgd_training_loop():
    """50 steps of SGD on fixed data. Loss at step 50 < loss at step 0."""
    cfg = _make_config()
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_data(cfg)

    initial_loss, _ = nl_hecate.forward(params, cfg, input_ids, target_ids)

    lr = 0.01
    for _ in range(50):
        _loss, grads = nl_hecate.compute_gradients(params, cfg, input_ids, target_ids)
        nl_hecate.apply_weight_gradients(params, grads, lr)

    final_loss, _ = nl_hecate.forward(params, cfg, input_ids, target_ids)
    assert final_loss < initial_loss, (
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_training_loss_trend():
    """Loss decreases on average over windows of 10 steps."""
    cfg = _make_config()
    params = nl_hecate.init_params(cfg, 42)
    input_ids, target_ids = _make_data(cfg)

    lr = 0.01
    losses = []
    for _ in range(50):
        loss, grads = nl_hecate.compute_gradients(params, cfg, input_ids, target_ids)
        losses.append(loss)
        nl_hecate.apply_weight_gradients(params, grads, lr)

    # Compare average of first 10 vs last 10
    avg_first = sum(losses[:10]) / 10
    avg_last = sum(losses[-10:]) / 10
    assert avg_last < avg_first, (
        f"Average loss should decrease: first10={avg_first:.4f}, last10={avg_last:.4f}"
    )


# ══════════════════════════════════════════════════════════════════
#  MAG training tests
# ══════════════════════════════════════════════════════════════════

def _make_mag_config():
    return nl_hecate.MAGConfig(16, 2, 8, 8, 8, 64, True)


def _make_mag_data(cfg):
    input_ids = list(range(cfg.seq_len))
    target_ids = [(t + 1) % cfg.vocab_size for t in range(cfg.seq_len)]
    return input_ids, target_ids


def test_mag_sgd_training_loop():
    """50 steps of MAG SGD on fixed data. Loss at step 50 < loss at step 0."""
    cfg = _make_mag_config()
    params = nl_hecate.mag_init_params(cfg, 42)
    input_ids, target_ids = _make_mag_data(cfg)

    initial_loss, _ = nl_hecate.mag_forward(params, cfg, input_ids, target_ids)

    lr = 0.01
    for _ in range(50):
        _loss, grads = nl_hecate.mag_compute_gradients(params, cfg, input_ids, target_ids)
        nl_hecate.mag_apply_weight_gradients(params, grads, lr)

    final_loss, _ = nl_hecate.mag_forward(params, cfg, input_ids, target_ids)
    assert final_loss < initial_loss, (
        f"MAG loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_mag_training_loss_trend():
    """MAG loss decreases on average over windows of 10 steps."""
    cfg = _make_mag_config()
    params = nl_hecate.mag_init_params(cfg, 42)
    input_ids, target_ids = _make_mag_data(cfg)

    lr = 0.01
    losses = []
    for _ in range(50):
        loss, grads = nl_hecate.mag_compute_gradients(params, cfg, input_ids, target_ids)
        losses.append(loss)
        nl_hecate.mag_apply_weight_gradients(params, grads, lr)

    avg_first = sum(losses[:10]) / 10
    avg_last = sum(losses[-10:]) / 10
    assert avg_last < avg_first, (
        f"MAG average loss should decrease: first10={avg_first:.4f}, last10={avg_last:.4f}"
    )
