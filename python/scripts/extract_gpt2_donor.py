#!/usr/bin/env python3
"""
Extract GPT-2 Small weights for CMS donor initialization (HOPE Section 5.1).

Maps GPT-2's 12 transformer layers into 4 CMS levels (3 layers per level).
For each level, extracts:
  - MLP effective weight: W_down @ W_up  (768x768)
  - Attention projections: W_Q, W_K, W_V, W_O  (768x768 each)
  - Embedding + unembedding weights

The MLP effective weight M_eff = c_fc @ c_proj composes the up/down
projections into a single 768x768 matrix -- this becomes M_0 for each
CMS level's inner-loop memory.

Per HOPE Section 5.1: "we use the trained parameters of {MLP_pretrained_i}
as the initial state of CMS blocks: MLP^(f_i)_0 = MLP_pretrained_i"

Layer-to-level mapping (3 layers per level):
  Level 0 (every token):    layers 0,1,2   -> average M_eff
  Level 1 (every 8 tokens): layers 3,4,5   -> average M_eff
  Level 2 (every 64 tokens): layers 6,7,8  -> average M_eff
  Level 3 (every 512 tokens): layers 9,10,11 -> average M_eff

Output: safetensors file loadable by NL_Hecate's checkpoint system.
Tensor key names match checkpoint.rs:load_safetensors exactly.

Usage:
    python scripts/extract_gpt2_donor.py --output checkpoints/gpt2_donor.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


def _build_mag_config(d: int, n_heads: int, vocab_size: int, k: int) -> dict:
    """Build a MAGConfig JSON dict matching Rust serde serialization."""
    head_dim = d // n_heads
    return {
        "swa": {
            "d_model": d,
            "num_heads": n_heads,
            "head_dim": head_dim,
            "seq_len": 512,
            "window_size": 512,
            "vocab_size": vocab_size,
        },
        "memory_enabled": True,
        "composition": "MAG",
        "memory_rule": "TitansLMM",
        "k": k,
        "chunk_sizes": [1, 8, 64, 512][:k],
        "d_hidden": 0,
        "lp_p": 2.0,
        "sign_sharpness": 10.0,
        "lq_q": 2.0,
        "lambda_local": 0.0,
        "lambda_2": 0.0,
        "delta": 1.0,
        "m_slots": 0,
        "d_compress": 0,
        "lambda_k": 0.0,
        "lambda_v": 0.0,
        "parallel": None,
        "retention": "L2WeightDecay",
        "m3": None,
        "frequency_schedule": "Fixed",
        "checkpoint_interval": None,
        "hope_variant": "FreqGated",
        "lattice_variant": "Decode",
        "n_persistent": 0,
        "attentional_bias": "L2",
        "kernel_size": 0,
        "momentum_kind": "None",
        "momentum_d_hidden": 0,
        "projection_kind": "Static",
        "self_generated_values": False,
        "self_ref_chunk_size": 1,
        "theta_floor": [],
        "theta_ceil": [],
        "m_norm_max": [100.0, 100.0, 100.0, 100.0][:k],
        "intermediate_size": 0,
        "feature_map": "Identity",
        "residual": True,
    }


def extract_gpt2_weights(output_path: str, d_model: int = 768, k: int = 4):
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        sys.exit("transformers + torch required. pip install transformers torch")

    import torch

    print("Loading GPT-2 Small from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    n_layers = model.config.n_layer  # 12
    n_heads = model.config.n_head    # 12
    vocab_size = model.config.vocab_size  # 50257
    d = model.config.n_embd          # 768

    assert d == d_model, f"GPT-2 d_model={d}, expected {d_model}"
    assert n_layers % k == 0, f"{n_layers} layers not divisible by k={k}"
    layers_per_level = n_layers // k

    print(f"  d_model={d}, n_heads={n_heads}, n_layers={n_layers}")
    print(f"  vocab_size={vocab_size}")
    print(f"  Mapping: {layers_per_level} layers per CMS level (k={k})")

    # ── Scale factor: rescale all GPT-2 weights to Xavier-like norms ──
    # GPT-2's trained weights are 2-23x larger than Xavier init (e.g., embed norm
    # 892 vs Xavier 39, W_Q norm 183 vs Xavier 28). Our CMS architecture expects
    # Xavier-scale weights; larger weights cause NaN on the first forward step.
    # We rescale each weight matrix to its Xavier expected Frobenius norm,
    # preserving the directional structure (trained subspace) while fitting
    # our architecture's scale.
    def xavier_scale(w: np.ndarray, fan_in: int, fan_out: int) -> np.ndarray:
        """Rescale w to have Xavier-expected Frobenius norm."""
        w = w.astype(np.float32)
        norm = np.linalg.norm(w)
        if norm < 1e-8:
            return w
        # Xavier uniform variance = 2/(fan_in+fan_out) * 1/3 * range^2
        # Expected Frobenius norm = sqrt(n_elements * variance)
        target_norm = np.sqrt(fan_in * fan_out * 2.0 / (fan_in + fan_out))
        return w * (target_norm / norm)

    # Extract embedding weights
    w_embed_raw = model.transformer.wte.weight.detach().cpu().numpy()  # [50257, 768]
    w_embed = xavier_scale(w_embed_raw, vocab_size, d)
    # GPT-2 ties embed/unembed, but we store both
    w_unembed = xavier_scale(w_embed_raw.T.copy(), d, vocab_size)  # [768, 50257]

    # Extract per-level MLP effective weights
    level_m_eff = []  # [k] arrays of shape [d, d]
    level_w_k = []
    level_w_v = []
    level_w_q = []

    for level in range(k):
        start_layer = level * layers_per_level
        end_layer = start_layer + layers_per_level

        m_effs = []
        w_ks = []
        w_vs = []
        w_qs = []

        for layer_idx in range(start_layer, end_layer):
            block = model.transformer.h[layer_idx]

            # MLP: c_fc [768, 3072], c_proj [3072, 768]
            # GPT-2 uses Conv1D which stores weights transposed: [in, out]
            c_fc = block.mlp.c_fc.weight.detach().cpu().numpy()      # [768, 3072]
            c_proj = block.mlp.c_proj.weight.detach().cpu().numpy()   # [3072, 768]

            # Effective M = input @ c_fc @ c_proj -> [768, 768]
            # c_fc is [768, 3072], c_proj is [3072, 768]
            m_eff = c_fc @ c_proj  # [768, 768]
            m_effs.append(m_eff)

            # Attention: c_attn [768, 2304] = concat(Q, K, V)
            c_attn = block.attn.c_attn.weight.detach().cpu().numpy()  # [768, 2304]
            w_q_layer = c_attn[:, :d]        # [768, 768]
            w_k_layer = c_attn[:, d:2*d]     # [768, 768]
            w_v_layer = c_attn[:, 2*d:3*d]   # [768, 768]

            w_qs.append(w_q_layer)
            w_ks.append(w_k_layer)
            w_vs.append(w_v_layer)

        # Average across layers in this level's group, then rescale to Xavier norm
        m_eff_avg = xavier_scale(np.mean(m_effs, axis=0), d, d)
        level_m_eff.append(m_eff_avg)
        level_w_k.append(xavier_scale(np.mean(w_ks, axis=0), d, d))
        level_w_v.append(xavier_scale(np.mean(w_vs, axis=0), d, d))
        level_w_q.append(xavier_scale(np.mean(w_qs, axis=0), d, d))

        print(f"  Level {level}: layers {start_layer}-{end_layer-1}")
        print(f"    M_eff norm: {np.linalg.norm(level_m_eff[-1]):.4f}")
        print(f"    W_K norm:   {np.linalg.norm(level_w_k[-1]):.4f}")
        print(f"    W_V norm:   {np.linalg.norm(level_w_v[-1]):.4f}")

    # Extract SWA attention weights from first layer group (level 0)
    # Our model has a single SWA block; use GPT-2 layer 0's attention
    block0 = model.transformer.h[0]
    c_attn_0 = block0.attn.c_attn.weight.detach().cpu().numpy()
    w_o_0 = block0.attn.c_proj.weight.detach().cpu().numpy()  # [768, 768]

    swa_w_q = xavier_scale(c_attn_0[:, :d], d, d)
    swa_w_k = xavier_scale(c_attn_0[:, d:2*d], d, d)
    swa_w_v = xavier_scale(c_attn_0[:, 2*d:3*d], d, d)
    swa_w_o = xavier_scale(w_o_0, d, d)

    # ── Build tensor dict with key names matching checkpoint.rs ──────
    # Key names MUST match what load_safetensors expects (core/src/checkpoint.rs).
    tensors = {}

    # SWA params — checkpoint.rs lines 284-294
    tensors["embed.weight"] = w_embed.astype(np.float32).flatten()
    tensors["swa.w_q"] = swa_w_q.flatten()
    tensors["swa.w_k"] = swa_w_k.flatten()
    tensors["swa.w_v"] = swa_w_v.flatten()
    tensors["swa.w_o"] = swa_w_o.flatten()
    tensors["lm_head.weight"] = w_unembed.astype(np.float32).flatten()

    # LayerNorm — checkpoint.rs lines 280-283
    tensors["ln_attn.gamma"] = np.ones(d, dtype=np.float32)
    tensors["ln_attn.beta"] = np.zeros(d, dtype=np.float32)
    tensors["ln_mem.gamma"] = np.ones(d, dtype=np.float32)
    tensors["ln_mem.beta"] = np.zeros(d, dtype=np.float32)

    # Per-level memory params — checkpoint.rs lines 300-335
    for level in range(k):
        p = f"level.{level}"

        # Memory projection weights (Bf16Storage): checkpoint.rs lines 303-305
        tensors[f"{p}.w_k"] = level_w_k[level].flatten()
        tensors[f"{p}.w_v"] = level_w_v[level].flatten()
        tensors[f"{p}.w_q"] = level_w_q[level].flatten()

        # Gate weights: [2*d] -- Xavier uniform init (same shape as MAGParams::init).
        # Using GPT-2's LN gamma (all ~1.0, norm ~sqrt(d)) causes NaN because
        # gate pre-sigmoid values become ~d (way too large for stable gates).
        # Shape is 2*d because gates take concatenated [x; mem_output] input.
        rng = np.random.RandomState(42 + level)
        gate_range = np.sqrt(6.0 / (2 * d + 1))
        tensors[f"{p}.gate.alpha"] = rng.uniform(-gate_range, gate_range, 2 * d).astype(np.float32)
        tensors[f"{p}.gate.theta"] = rng.uniform(-gate_range, gate_range, 2 * d).astype(np.float32)
        tensors[f"{p}.gate.eta"] = rng.uniform(-gate_range, gate_range, 2 * d).astype(np.float32)

        # Gate biases: warm L2/L3 b_theta per gate warmup protocol (09_gate_warmup.md)
        # Default [-4.6,-5.6,-6.6,-7.6] kills L2/L3: softplus'(-7.6)≈5e-4 attenuates
        # gradients 2000x. Seeding L2/L3 at -4.6 places them in the active range.
        b_alpha_defaults = [3.0, 4.0, 4.5, 5.0]
        b_theta_defaults = [-4.6, -5.6, -4.6, -4.6]
        b_eta_defaults = [0.0, 0.0, 0.0, 0.0]
        tensors[f"{p}.gate.b_alpha"] = np.array([b_alpha_defaults[level]], dtype=np.float32)
        tensors[f"{p}.gate.b_theta"] = np.array([b_theta_defaults[level]], dtype=np.float32)
        tensors[f"{p}.gate.b_eta"] = np.array([b_eta_defaults[level]], dtype=np.float32)

        # Atlas Omega projection: zero-init, d * 2*d (required non-empty by GPU path)
        tensors[f"{p}.w_omega"] = np.zeros(d * 2 * d, dtype=np.float32)

        # M_eff as donor initialization for inner-loop memory (HOPE Section 5.1)
        # checkpoint.rs line 329: level.{i}.m_state.mem -> m_mem_init
        # Conductor.init_from_params copies this to context.memory[level]
        tensors[f"{p}.m_state.mem"] = level_m_eff[level].flatten()

    # CMS aggregation weights — checkpoint.rs lines 354-356
    tensors["alpha_mem"] = np.zeros(k, dtype=np.float32)
    tensors["alpha_refl"] = np.zeros(k, dtype=np.float32)

    # ── Build metadata with serialized MAGConfig ─────────────────────
    # checkpoint.rs lines 240-244: config is deserialized from __metadata__.config
    mag_config = _build_mag_config(d, n_heads, vocab_size, k)

    metadata = {
        "version": "2",
        "format": "nl_hecate_v2",
        "created_at": f"epoch:{int(time.time())}",
        "config": json.dumps(mag_config, separators=(",", ":")),
        "build_state": "null",
    }

    _write_safetensors(output_path, tensors, metadata)

    total_params = sum(v.size for v in tensors.values())
    print(f"\nDonor checkpoint written to {output_path}")
    print(f"  Total params: {total_params:,}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")


def _write_safetensors(path: str, tensors: dict[str, np.ndarray],
                       metadata: dict[str, str]):
    """Write safetensors format matching NL_Hecate's load_safetensors."""

    # Build header
    header = {}
    offset = 0
    data_parts = []

    for name, arr in tensors.items():
        raw = arr.astype(np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": [arr.size],  # flat [n_elems] — matches checkpoint.rs line 174
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    header["__metadata__"] = metadata

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad to 8-byte alignment
    pad_len = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * pad_len

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GPT-2 donor weights")
    parser.add_argument("--output", default="checkpoints/gpt2_donor.safetensors",
                        help="Output safetensors path")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    extract_gpt2_weights(args.output)
