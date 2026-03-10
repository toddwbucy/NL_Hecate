#!/usr/bin/env python3
"""
Profile MAC vs MAG composition cost breakdown.

Runs identical Titans LMM k=4 d=512 configs with MAC and MAG compositions,
timing each phase with CUDA synchronization to get accurate GPU timings.

Since step_adamw() is opaque, we use two approaches:
  1. step_adamw()         → total step time (fwd + bwd + opt)
  2. backward_only()      → fwd + bwd only (no optimizer)
  3. Difference (1)-(2)   → optimizer cost

We also time per-pulse patterns: all-levels-active (L3 fire) vs L0-only steps
to see if slow-level memory updates dominate.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/profile_composition.py
    CUDA_VISIBLE_DEVICES=2 python scripts/profile_composition.py --steps 100
"""

import os
import sys
import time
import argparse
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nl_hecate
from engine.config import BuildConfig
from engine.data import BpeTokenStream


def cuda_sync():
    """Force CUDA synchronization for accurate timing."""
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaDeviceSynchronize()
    except OSError:
        pass  # No CUDA runtime — timings may include async overlap


def make_model(config_path: str):
    """Build a GpuModel + Conductor from a config file."""
    bcfg = BuildConfig.from_file(config_path)
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

    # Seed adaptive projections if needed
    context = nl_hecate.ContextState(bcfg.k, bcfg.d_model)
    if bcfg.projection_kind == "adaptive":
        context.seed_self_ref(params)

    return gpu_model, conductor, bcfg


def profile_run(config_path: str, label: str,
                n_steps: int = 50, warmup: int = 10):
    """Profile a single composition config."""
    gpu_model, conductor, bcfg = make_model(config_path)

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  composition={bcfg.composition}, window={bcfg.window_size}, "
          f"rule={bcfg.memory_rule}, momentum={bcfg.momentum_kind}")
    print(f"  d={bcfg.d_model}, k={bcfg.k}, seq_len={bcfg.seq_len}, "
          f"params={nl_hecate.mag_init_params(nl_hecate.MAGConfig(d_model=bcfg.d_model, num_heads=bcfg.num_heads, head_dim=bcfg.head_dim, seq_len=bcfg.seq_len, window_size=bcfg.window_size, vocab_size=bcfg.vocab_size, memory_enabled=True, k=bcfg.k, chunk_sizes=bcfg.chunk_sizes, memory_rule=bcfg.memory_rule, composition=bcfg.composition, checkpoint_interval=bcfg.checkpoint_interval, projection_kind=bcfg.projection_kind, self_generated_values=bcfg.self_generated_values, self_ref_chunk_size=bcfg.self_ref_chunk_size, momentum_kind=bcfg.momentum_kind, momentum_d_hidden=bcfg.momentum_d_hidden), 42).num_params():,}")
    print(f"  {n_steps} measured steps + {warmup} warmup")
    print(f"{'=' * 70}")

    # Buckets: step_adamw total, backward_only, by pulse pattern
    t_adamw_all = []       # all step_adamw times
    t_bwd_only_all = []    # all backward_only times
    t_adamw_l3 = []        # step_adamw when L3 fires
    t_adamw_l0_only = []   # step_adamw when only L0 fires

    # Fresh loader for each run
    loader = BpeTokenStream(bcfg.data_path, split="train")

    total_steps = warmup + n_steps
    for i in range(total_steps):
        chunk = loader.next_chunk(bcfg.seq_len)
        if chunk is None:
            loader = BpeTokenStream(bcfg.data_path, split="train")
            chunk = loader.next_chunk(bcfg.seq_len)
        input_ids, target_ids = chunk
        pulse = conductor.pulse()
        active = pulse.active_levels

        measuring = i >= warmup

        # -- step_adamw timing --
        cuda_sync()
        t0 = time.perf_counter()
        _loss, _g_norm = gpu_model.step_adamw(
            input_ids, target_ids, pulse, 0.0006,
            beta1=0.9, beta2=0.999, eps=1e-8,
            weight_decay=0.1, max_grad_norm=1.0,
        )
        cuda_sync()
        t1 = time.perf_counter()

        conductor.advance()

        if measuring:
            dt = t1 - t0
            t_adamw_all.append(dt)

            # Classify by pulse pattern
            n_active = sum(active)
            if n_active >= 4:  # L3 fire (all levels active)
                t_adamw_l3.append(dt)
            elif n_active == 1:  # L0 only
                t_adamw_l0_only.append(dt)

    # Now run backward_only pass for the same number of steps
    loader2 = BpeTokenStream(bcfg.data_path, split="train")
    gpu_model2, conductor2, _ = make_model(config_path)

    for i in range(total_steps):
        chunk = loader2.next_chunk(bcfg.seq_len)
        if chunk is None:
            loader2 = BpeTokenStream(bcfg.data_path, split="train")
            chunk = loader2.next_chunk(bcfg.seq_len)
        input_ids, target_ids = chunk
        pulse = conductor2.pulse()

        measuring = i >= warmup

        cuda_sync()
        t0 = time.perf_counter()
        _loss, _grads = gpu_model2.backward_only(input_ids, target_ids, pulse)
        cuda_sync()
        t1 = time.perf_counter()

        conductor2.advance()

        if measuring:
            t_bwd_only_all.append(t1 - t0)

    # Report
    def stats(times):
        if not times:
            return 0, 0, 0
        avg = sum(times) / len(times)
        return avg, min(times), max(times)

    avg_adamw, min_adamw, max_adamw = stats(t_adamw_all)
    avg_bwd, min_bwd, max_bwd = stats(t_bwd_only_all)
    avg_opt = avg_adamw - avg_bwd  # estimated optimizer cost

    print(f"\n  {'Component':<30} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print(f"  {'-' * 62}")
    print(f"  {'step_adamw (total)':<30} {avg_adamw*1000:>10.2f} {min_adamw*1000:>10.2f} {max_adamw*1000:>10.2f}")
    print(f"  {'backward_only (fwd+bwd)':<30} {avg_bwd*1000:>10.2f} {min_bwd*1000:>10.2f} {max_bwd*1000:>10.2f}")
    print(f"  {'optimizer (estimated)':<30} {avg_opt*1000:>10.2f}")

    if t_adamw_l3:
        avg_l3 = stats(t_adamw_l3)[0]
        print(f"\n  {'L3 fire steps (all levels)':<30} {avg_l3*1000:>10.2f} ms  (n={len(t_adamw_l3)})")
    if t_adamw_l0_only:
        avg_l0 = stats(t_adamw_l0_only)[0]
        print(f"  {'L0-only steps':<30} {avg_l0*1000:>10.2f} ms  (n={len(t_adamw_l0_only)})")
    if t_adamw_l3 and t_adamw_l0_only:
        ratio = stats(t_adamw_l3)[0] / stats(t_adamw_l0_only)[0]
        print(f"  {'L3/L0 ratio':<30} {ratio:>10.2f}x")

    tok_s = bcfg.seq_len / avg_adamw if avg_adamw > 0 else 0
    print(f"\n  Throughput: {tok_s:.0f} tok/s  ({1/avg_adamw:.1f} steps/s)")

    return {
        "label": label,
        "composition": bcfg.composition,
        "window_size": bcfg.window_size,
        "avg_adamw_ms": avg_adamw * 1000,
        "avg_bwd_ms": avg_bwd * 1000,
        "avg_opt_ms": avg_opt * 1000,
        "avg_l3_ms": stats(t_adamw_l3)[0] * 1000 if t_adamw_l3 else None,
        "avg_l0_ms": stats(t_adamw_l0_only)[0] * 1000 if t_adamw_l0_only else None,
        "tok_s": tok_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile MAC vs MAG composition cost")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of measured steps per config (default: 100)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup steps (default: 20)")
    args = parser.parse_args()

    mac = profile_run("configs/titans_dolmino_100k_mac.json", "Titans MAC",
                       n_steps=args.steps, warmup=args.warmup)
    mag = profile_run("configs/titans_dolmino_100k_mag.json", "Titans MAG",
                       n_steps=args.steps, warmup=args.warmup)

    # Comparison
    print(f"\n{'=' * 70}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30} {'MAC':>12} {'MAG':>12} {'MAC/MAG':>10}")
    print(f"  {'-' * 66}")

    for key, label in [
        ("avg_adamw_ms", "Total step (ms)"),
        ("avg_bwd_ms", "Fwd+Bwd (ms)"),
        ("avg_opt_ms", "Optimizer (ms)"),
        ("avg_l3_ms", "L3 fire step (ms)"),
        ("avg_l0_ms", "L0-only step (ms)"),
        ("tok_s", "Throughput (tok/s)"),
    ]:
        v_mac = mac.get(key)
        v_mag = mag.get(key)
        if v_mac is not None and v_mag is not None and v_mag > 0:
            ratio = v_mac / v_mag
            print(f"  {label:<30} {v_mac:>12.2f} {v_mag:>12.2f} {ratio:>10.2f}x")
        elif v_mac is not None:
            print(f"  {label:<30} {v_mac:>12.2f} {'—':>12}")

    print(f"{'=' * 70}")
    print(f"\n  Window size: MAC={mac['window_size']}, MAG={mag['window_size']}")
    print("  If MAC/MAG ratio ≈ 1.0 on fwd+bwd → attention is NOT the bottleneck")
    print("  If optimizer cost is high → AdamW state updates dominate")
    print("  If L3/L0 ratio > 1.5 → slow-level memory updates are expensive")


if __name__ == "__main__":
    main()
