#!/usr/bin/env python3
"""
Wengert tape profiling: measure CPU tape memory and time scaling with seq_len.

Part A of S4-M7 tape profiling. Uses a small model (d=64, k=2) to isolate
tape behavior. Measures RSS and wall time at seq_len=512 vs seq_len=2048.

Usage:
    python profile_tape.py
"""

import gc
import math
import resource
import time

import nl_hecate


def rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / (1024 * 1024)
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def profile_seq_len(seq_len: int, steps: int = 5) -> dict:
    """Run cms_compute_gradients at a given seq_len, measure RSS and time.

    Uses d=64, k=2, vocab=256 (small model to isolate tape overhead).
    """
    d_model = 64
    num_heads = 4
    k = 2
    chunk_sizes = [1, 8]
    vocab_size = 256

    cfg = nl_hecate.MAGConfig(
        d_model=d_model, num_heads=num_heads, head_dim=d_model // num_heads,
        seq_len=seq_len, window_size=min(seq_len, 256), vocab_size=vocab_size,
        memory_enabled=True, k=k, chunk_sizes=chunk_sizes,
        memory_rule="delta", composition="mag",
    )
    params = nl_hecate.mag_init_params(cfg, 42)
    conductor = nl_hecate.Conductor(k, chunk_sizes)
    context = nl_hecate.ContextState(k, d_model)
    error_buffers = nl_hecate.ErrorBufferList(k, d_model)

    # Generate deterministic data
    input_ids = [t % vocab_size for t in range(seq_len)]
    target_ids = [(t + 1) % vocab_size for t in range(seq_len)]

    # Warmup (1 step, discard)
    gc.collect()
    pulse = conductor.pulse()
    nl_hecate.cms_compute_gradients(params, cfg, input_ids, target_ids, pulse,
                                    context, error_buffers)
    conductor.advance()
    gc.collect()

    # Measure
    rss_before = rss_mb()
    t_fwd_total = 0.0
    t_total_start = time.perf_counter()

    for _ in range(steps):
        pulse = conductor.pulse()
        t0 = time.perf_counter()
        loss, grads = nl_hecate.cms_compute_gradients(
            params, cfg, input_ids, target_ids, pulse, context, error_buffers)
        t_fwd_total += time.perf_counter() - t0
        # Apply gradients to keep params evolving (decorrelates tape patterns)
        nl_hecate.mag_apply_weight_gradients(params, grads, 0.001)
        conductor.advance()

    t_total = time.perf_counter() - t_total_start
    rss_after = rss_mb()

    return {
        "seq_len": seq_len,
        "steps": steps,
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "total_time_s": round(t_total, 3),
        "avg_step_time_s": round(t_fwd_total / steps, 4),
        "loss": round(loss, 4),
    }


def main():
    print("=" * 60)
    print("Wengert Tape Profiling (CPU, d=64, k=2)")
    print("=" * 60)

    # Profile at seq_len=512
    print("\nProfiling seq_len=512...")
    gc.collect()
    r512 = profile_seq_len(512, steps=5)
    print(f"  RSS: {r512['rss_before_mb']:.1f} → {r512['rss_after_mb']:.1f} MB "
          f"(delta={r512['rss_delta_mb']:.1f})")
    print(f"  Avg step time: {r512['avg_step_time_s']:.4f}s")

    # Force cleanup before next profile
    gc.collect()

    # Profile at seq_len=2048
    print("\nProfiling seq_len=2048...")
    gc.collect()
    r2048 = profile_seq_len(2048, steps=5)
    print(f"  RSS: {r2048['rss_before_mb']:.1f} → {r2048['rss_after_mb']:.1f} MB "
          f"(delta={r2048['rss_delta_mb']:.1f})")
    print(f"  Avg step time: {r2048['avg_step_time_s']:.4f}s")

    # Compute ratios
    # Memory ratio: compare peak RSS (rss_after) since tape arena grows during forward
    mem_512 = max(r512["rss_after_mb"], 1.0)
    mem_2048 = max(r2048["rss_after_mb"], 1.0)
    mem_ratio = mem_2048 / mem_512

    time_ratio = r2048["avg_step_time_s"] / max(r512["avg_step_time_s"], 1e-6)

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"  Memory (512):  {r512['rss_after_mb']:.1f} MB")
    print(f"  Memory (2048): {r2048['rss_after_mb']:.1f} MB")
    print(f"  Memory ratio:  {mem_ratio:.3f}x")
    print(f"  Time (512):    {r512['avg_step_time_s']:.4f}s/step")
    print(f"  Time (2048):   {r2048['avg_step_time_s']:.4f}s/step")
    print(f"  Time ratio:    {time_ratio:.3f}x")

    # Thresholds from S4-M7 plan
    if mem_ratio <= 1.1:
        print(f"\n  PASS: memory ratio {mem_ratio:.3f} <= 1.1")
    elif mem_ratio <= 1.2:
        print(f"\n  WARNING: memory ratio {mem_ratio:.3f} > 1.1 (stop at >= 1.2)")
    else:
        print(f"\n  FAIL: memory ratio {mem_ratio:.3f} >= 1.2 — tape leaks with seq_len")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
