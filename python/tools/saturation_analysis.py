"""
saturation_analysis.py — post-hoc parameter saturation analysis from JSONL training logs.

Reads slow_level_fire events and computes per-level saturation metrics:
  - EMA of gradient norm at each firing
  - Peak gnorm (historical max)
  - saturation_ratio = ema_gnorm / peak_gnorm  (1.0 at peak, → 0 at saturation)
  - saturated flag: ratio < threshold for K consecutive firings

Usage:
    python tools/saturation_analysis.py runs/warm_start_10k_raw.jsonl
    python tools/saturation_analysis.py runs/gate_warmup_diagnostic_100k.jsonl --alpha 0.15 --threshold 0.15
    python tools/saturation_analysis.py runs/a.jsonl runs/b.jsonl --labels GPU0 GPU1
"""

import argparse
import json
import sys
from pathlib import Path


def compute_saturation(
    log_path: str,
    alpha: float = 0.1,
    threshold: float = 0.15,
    window: int = 5,
) -> dict:
    """
    Read a JSONL log and compute per-level saturation metrics from slow_level_fire events.

    Returns a dict with per-level trajectories and saturation onset info.
    """
    fires_by_level: dict[int, list[dict]] = {}

    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            if d["event"] != "slow_level_fire":
                continue
            gnorms = d.get("level_grad_norms", [])
            active = d.get("active_levels", [])
            step = d["step"]
            for i, (g, a) in enumerate(zip(gnorms, active)):
                if a and i >= 2:  # only slow levels (L2+) have dedicated fire events
                    fires_by_level.setdefault(i, []).append({"step": step, "gnorm": g})
            # L0/L1 are always active — grab their gnorms from every slow_level_fire
            for i in range(min(2, len(gnorms))):
                fires_by_level.setdefault(i, []).append({"step": step, "gnorm": gnorms[i]})

    results = {}
    for level, fires in sorted(fires_by_level.items()):
        ema = 0.0
        peak = 0.0
        below_count = 0
        saturation_onset = None
        trajectory = []

        for fire in fires:
            g = fire["gnorm"]
            step = fire["step"]

            # Update EMA
            ema = alpha * g + (1 - alpha) * ema
            peak = max(peak, ema)
            ratio = ema / peak if peak > 1e-10 else 1.0

            # Check saturation window
            if ratio < threshold:
                below_count += 1
            else:
                below_count = 0

            saturated = below_count >= window

            if saturated and saturation_onset is None:
                saturation_onset = step

            trajectory.append({
                "step": step,
                "gnorm": g,
                "ema": round(ema, 8),
                "peak": round(peak, 8),
                "ratio": round(ratio, 4),
                "saturated": saturated,
            })

        results[level] = {
            "trajectory": trajectory,
            "saturation_onset_step": saturation_onset,
            "final_ratio": trajectory[-1]["ratio"] if trajectory else None,
            "peak_gnorm": peak,
            "fire_count": len(fires),
        }

    return results


def print_report(label: str, results: dict, show_trajectory: bool = False):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for level, data in sorted(results.items()):
        onset = data["saturation_onset_step"]
        ratio = data["final_ratio"]
        peak = data["peak_gnorm"]
        n = data["fire_count"]

        status = f"SATURATING (onset step {onset})" if onset else f"active  (current ratio {ratio:.3f})"
        print(f"\n  L{level}  fires={n}  peak_gnorm={peak:.6f}  {status}")

        if show_trajectory:
            # Print every 10th entry for brevity
            traj = data["trajectory"]
            step = max(1, len(traj) // 20)
            print(f"  {'fire':>5}  {'step':>7}  {'gnorm':>10}  {'ema':>10}  {'ratio':>7}  sat")
            for i, t in enumerate(traj[::step]):
                sat = "  <<<" if t["saturated"] else ""
                print(f"  {i*step:>5}  {t['step']:>7}  {t['gnorm']:>10.6f}  {t['ema']:>10.6f}  {t['ratio']:>7.3f}{sat}")
        else:
            # Just show ratio trend bucketed
            traj = data["trajectory"]
            bucket_size = max(1, len(traj) // 10)
            print(f"  saturation_ratio trend (buckets of {bucket_size} fires):")
            for i in range(0, len(traj), bucket_size):
                bucket = traj[i:i+bucket_size]
                avg_ratio = sum(t["ratio"] for t in bucket) / len(bucket)
                min_ratio = min(t["ratio"] for t in bucket)
                step_range = f"{bucket[0]['step']}-{bucket[-1]['step']}"
                bar = "█" * int(avg_ratio * 20)
                sat_flag = " <<< SATURATING" if any(t["saturated"] for t in bucket) else ""
                print(f"    steps {step_range:>12}  avg={avg_ratio:.3f}  min={min_ratio:.3f}  {bar}{sat_flag}")


def print_comparison(labels: list[str], all_results: list[dict]):
    """Side-by-side saturation_ratio for L0 across runs."""
    print(f"\n{'='*60}")
    print("  L0 saturation_ratio comparison")
    print(f"{'='*60}")

    # Align by fire index
    trajs = [r.get(0, {}).get("trajectory", []) for r in all_results]
    max_len = max(len(t) for t in trajs)
    step_size = max(1, max_len // 20)

    header = f"  {'fire':>5}  {'step':>7}  " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i in range(0, max_len, step_size):
        row = f"  {i:>5}  "
        step_val = ""
        cols = []
        for traj in trajs:
            if i < len(traj):
                t = traj[i]
                step_val = str(t["step"])
                cols.append(f"{t['ratio']:>10.3f}")
            else:
                cols.append(f"{'—':>10}")
        print(f"  {i:>5}  {step_val:>7}  " + "  ".join(cols))


def main():
    parser = argparse.ArgumentParser(description="Parameter saturation analysis from JSONL logs")
    parser.add_argument("logs", nargs="+", help="JSONL log file(s)")
    parser.add_argument("--labels", nargs="+", help="Labels for each log (default: filenames)")
    parser.add_argument("--alpha", type=float, default=0.1, help="EMA alpha (default: 0.1)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Saturation ratio threshold (default: 0.15)")
    parser.add_argument("--window", type=int, default=5, help="Consecutive firings below threshold to confirm saturation (default: 5)")
    parser.add_argument("--trajectory", action="store_true", help="Show full per-fire trajectory")
    args = parser.parse_args()

    labels = args.labels or [Path(p).stem for p in args.logs]
    if len(labels) < len(args.logs):
        labels += [Path(p).stem for p in args.logs[len(labels):]]

    all_results = []
    for label, log_path in zip(labels, args.logs):
        results = compute_saturation(log_path, alpha=args.alpha, threshold=args.threshold, window=args.window)
        all_results.append(results)
        print_report(label, results, show_trajectory=args.trajectory)

    if len(args.logs) > 1:
        print_comparison(labels, all_results)

    # Summary: predicted saturation onset for L0 in each run
    print(f"\n{'='*60}")
    print("  Summary — L0 saturation onset")
    print(f"{'='*60}")
    for label, results in zip(labels, all_results):
        l0 = results.get(0, {})
        onset = l0.get("saturation_onset_step")
        ratio = l0.get("final_ratio", "?")
        peak = l0.get("peak_gnorm", 0)
        if onset:
            print(f"  {label}: SATURATED at step {onset}  (peak gnorm {peak:.4f})")
        else:
            print(f"  {label}: not yet saturated  current ratio={ratio:.3f}  peak={peak:.4f}")
            # Extrapolate: fit linear decay to last 20% of ratio trajectory
            traj = l0.get("trajectory", [])
            if len(traj) > 20:
                tail = traj[-(len(traj)//5):]
                if len(tail) > 1:
                    r0, r1 = tail[0]["ratio"], tail[-1]["ratio"]
                    s0, s1 = tail[0]["step"], tail[-1]["step"]
                    if r0 > r1 and s1 > s0:
                        decay_per_step = (r0 - r1) / (s1 - s0)
                        if decay_per_step > 0:
                            steps_to_threshold = (r1 - args.threshold) / decay_per_step
                            predicted = int(s1 + steps_to_threshold)
                            print(f"          linear extrapolation → saturation ~step {predicted}")


if __name__ == "__main__":
    main()
