"""Regression tests: verify PyTorch baseline matches Rust pipeline."""

import math

torch = __import__("pytest").importorskip("torch")
import nl_hecate

# Import from baseline module (parent dir)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline_pytorch import (
    load_weights_from_rust,
    forward_pytorch,
    sgd_step_pytorch,
    make_config,
    make_chunks,
    TEXT,
    SEQ_LEN,
    VOCAB_SIZE,
    D_MODEL,
)


BYTES = [b for b in TEXT.encode("ascii")]
CHUNKS = make_chunks(BYTES, SEQ_LEN)


def test_get_weights_shapes():
    """get_weights() returns dict with correct flat sizes."""
    cfg = make_config()
    params = nl_hecate.init_params(cfg, 42)
    w = params.get_weights()
    assert set(w.keys()) == {"w_embed", "w_q", "w_k", "w_v", "w_o", "w_unembed"}
    assert len(w["w_embed"]) == VOCAB_SIZE * D_MODEL
    assert len(w["w_q"]) == D_MODEL * D_MODEL
    assert len(w["w_unembed"]) == D_MODEL * VOCAB_SIZE


def test_initial_loss_matches():
    """Step 0 forward loss matches between Rust and PyTorch (rel err < 1e-6)."""
    cfg = make_config()
    params = nl_hecate.init_params(cfg, 42)
    pt_weights = load_weights_from_rust(params)

    inp, tgt = CHUNKS[0]
    rust_loss, _ = nl_hecate.forward(params, cfg, inp, tgt)
    pt_loss = forward_pytorch(pt_weights, inp, tgt).item()

    denom = max(abs(rust_loss), abs(pt_loss), 1e-12)
    rel_err = abs(rust_loss - pt_loss) / denom
    assert rel_err < 1e-6, f"Initial loss mismatch: rust={rust_loss}, pt={pt_loss}, rel_err={rel_err}"


def test_loss_matches_10_steps():
    """10 SGD steps produce matching loss curves (rel err < 1e-4 per step)."""
    cfg = make_config()
    rust_params = nl_hecate.init_params(cfg, 42)
    pt_weights = load_weights_from_rust(rust_params)
    lr = 0.05

    for step in range(10):
        inp, tgt = CHUNKS[step % len(CHUNKS)]

        rust_loss, grads = nl_hecate.compute_gradients(rust_params, cfg, inp, tgt)
        nl_hecate.sgd_step(rust_params, grads, lr)

        pt_loss_t = forward_pytorch(pt_weights, inp, tgt)
        pt_loss = pt_loss_t.item()
        pt_loss_t.backward()
        sgd_step_pytorch(pt_weights, lr)

        assert math.isfinite(rust_loss), f"Rust loss not finite at step {step}"
        assert math.isfinite(pt_loss), f"PyTorch loss not finite at step {step}"

        denom = max(abs(rust_loss), abs(pt_loss), 1e-12)
        rel_err = abs(rust_loss - pt_loss) / denom
        assert rel_err < 1e-4, (
            f"Step {step}: rust={rust_loss:.6f}, pt={pt_loss:.6f}, rel_err={rel_err:.2e}"
        )
