#!/usr/bin/env python3
"""
Profile per-step timing for k=1 vs k=4 to understand where time goes.

Runs a small number of steps on each config and breaks down:
  - Data loading
  - Conductor pulse generation
  - Forward pass (GPU)
  - Backward pass (GPU)
  - Optimizer step (GPU)
  - Total step

Usage:
    CUDA_VISIBLE_DEVICES=2 python profile_step.py
"""

import os
import sys
import time

# Add python/ root to sys.path so engine/ is importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nl_hecate
from engine.config import BuildConfig
from engine.data import BpeTokenStream


def profile_config(config_path: str, label: str, n_steps: int = 50,
                   warmup_steps: int = 10):
    """Profile step timing for a given config."""
    bcfg = BuildConfig.from_file(config_path)

    # Setup
    loader = BpeTokenStream(bcfg.data_path, split="train")
    cfg = nl_hecate.MAGConfig(
        d_model=bcfg.d_model, num_heads=bcfg.num_heads,
        head_dim=bcfg.head_dim, seq_len=bcfg.seq_len,
        window_size=bcfg.window_size, vocab_size=bcfg.vocab_size,
        memory_enabled=True, k=bcfg.k, chunk_sizes=bcfg.chunk_sizes,
        memory_rule=bcfg.memory_rule, composition=bcfg.composition,
        checkpoint_interval=bcfg.checkpoint_interval,
        projection_kind=bcfg.projection_kind,
        self_generated_values=bcfg.self_generated_values,
        self_ref_chunk_size=bcfg.self_ref_chunk_size,
        momentum_kind=bcfg.momentum_kind,
        momentum_d_hidden=bcfg.momentum_d_hidden,
    )
    params = nl_hecate.mag_init_params(cfg, 42)
    gpu_model = nl_hecate.GpuModel.from_params(params, cfg)
    conductor = nl_hecate.Conductor(bcfg.k, bcfg.chunk_sizes)

    # Seed self-ref
    context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
    if bcfg.projection_kind == "adaptive":
        context.seed_self_ref(params)

    print(f"\n{'=' * 60}")
    print(f"Profiling: {label}")
    print(f"  Config: {config_path}")
    print(f"  k={bcfg.k}, params={params.num_params():,}")
    print(f"  Steps: {n_steps} (+ {warmup_steps} warmup)")
    print(f"{'=' * 60}")

    # Timing accumulators
    t_data = []
    t_pulse = []
    t_step_adamw = []
    t_total = []

    for i in range(warmup_steps + n_steps):
        t0 = time.perf_counter()

        # Data load
        td0 = time.perf_counter()
        chunk = loader.next_chunk(bcfg.seq_len)
        if chunk is None:
            raise RuntimeError(f"Dataset too short for seq_len={bcfg.seq_len}")
        input_ids, target_ids = chunk
        td1 = time.perf_counter()

        # Pulse
        tp0 = time.perf_counter()
        pulse = conductor.pulse()
        tp1 = time.perf_counter()

        # step_adamw (forward + backward + optimizer in one GPU call)
        ts0 = time.perf_counter()
        loss, g_norm = gpu_model.step_adamw(
            input_ids, target_ids, pulse, 0.0006,
            beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.1, max_grad_norm=1.0,
        )
        ts1 = time.perf_counter()

        conductor.advance()

        t1 = time.perf_counter()

        if i >= warmup_steps:
            t_data.append(td1 - td0)
            t_pulse.append(tp1 - tp0)
            t_step_adamw.append(ts1 - ts0)
            t_total.append(t1 - t0)

    # Report
    def stats(times):
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        return avg, mn, mx

    print(f"\nResults ({n_steps} steps):")
    print(f"  {'Component':<25} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'% Total':>10}")
    print(f"  {'-' * 65}")

    total_avg = stats(t_total)[0]
    for name, times in [
        ("Data loading", t_data),
        ("Conductor pulse", t_pulse),
        ("step_adamw (fwd+bwd+opt)", t_step_adamw),
        ("Total step", t_total),
    ]:
        avg, mn, mx = stats(times)
        pct = (avg / total_avg * 100) if total_avg > 0 else 0
        print(f"  {name:<25} {avg*1000:>10.3f} {mn*1000:>10.3f} {mx*1000:>10.3f} {pct:>9.1f}%")

    steps_per_sec = 1.0 / total_avg if total_avg > 0 else 0
    print(f"\n  Steps/sec: {steps_per_sec:.2f}")
    print(f"  ms/step:   {total_avg * 1000:.1f}")

    return total_avg


if __name__ == "__main__":
    t_k4 = profile_config("configs/fineweb_edu_k4.json", "k=4 (4 CMS levels)")
    t_k1 = profile_config("configs/fineweb_edu_k1.json", "k=1 (single level)")

    print(f"\n{'=' * 60}")
    print("Comparison")
    print(f"{'=' * 60}")
    print(f"  k=4: {t_k4*1000:.1f} ms/step")
    print(f"  k=1: {t_k1*1000:.1f} ms/step")
    if t_k1 > 0:
        print(f"  Ratio: k=4 is {t_k4/t_k1:.2f}x slower than k=1")
    print(f"{'=' * 60}")
