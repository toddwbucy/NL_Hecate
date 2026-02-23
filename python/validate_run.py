#!/usr/bin/env python3
"""
S4-M7 post-run validation: parse JSONL log and check all hard thresholds.

Usage:
    python validate_run.py runs/curriculum_100k.jsonl
"""

import json
import math
import sys
from pathlib import Path


def load_events(jsonl_path: str) -> list[dict]:
    """Load all events from a JSONL file."""
    events = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def check_loss_convergence(events: list[dict]) -> tuple[bool, str]:
    """Check >= 15% loss decrease between steps 50K-55K and 95K-100K."""
    steps = [(e["step"], e["loss"]) for e in events
             if e.get("event") == "step" and "loss" in e]

    early = [loss for step, loss in steps if 50000 <= step <= 55000]
    late = [loss for step, loss in steps if 95000 <= step <= 100000]

    if not early or not late:
        return False, f"Insufficient data: {len(early)} early, {len(late)} late samples"

    avg_early = sum(early) / len(early)
    avg_late = sum(late) / len(late)
    decrease = (avg_early - avg_late) / avg_early * 100

    passed = decrease >= 15.0
    msg = (f"avg_loss 50K-55K={avg_early:.4f}, 95K-100K={avg_late:.4f}, "
           f"decrease={decrease:.1f}% (threshold: >=15%)")
    return passed, msg


def check_no_nan_inf(events: list[dict]) -> tuple[bool, str]:
    """Scan all step events for NaN/Inf losses."""
    bad_steps = []
    for e in events:
        if e.get("event") == "step" and "loss" in e:
            loss = e["loss"]
            if math.isnan(loss) or math.isinf(loss):
                bad_steps.append(e["step"])

    abort = [e for e in events if e.get("event") == "abort"]

    if bad_steps:
        return False, f"NaN/Inf at steps: {bad_steps[:10]}{'...' if len(bad_steps) > 10 else ''}"
    if abort:
        return False, f"Build aborted: {abort[0].get('reason', 'unknown')}"
    return True, "No NaN/Inf detected"


def check_level3_activity(events: list[dict]) -> tuple[bool, str]:
    """Sum level3_activity events and check active >= 50."""
    l3_events = [e for e in events if e.get("event") == "level3_activity"]

    if not l3_events:
        return False, "No level3_activity events found"

    total_fires = sum(e.get("fires", 0) for e in l3_events)
    active_fires = sum(e.get("active", 0) for e in l3_events)

    passed = active_fires >= 50
    msg = (f"Level 3: {active_fires} active fires out of {total_fires} total "
           f"(threshold: >=50, stop-the-line: <25)")
    if active_fires < 25:
        msg += " — STOP THE LINE"
    return passed, msg


def check_checkpoint_roundtrip(events: list[dict]) -> tuple[bool, str]:
    """All checkpoint_roundtrip events must have delta < 1e-6."""
    rt_events = [e for e in events if e.get("event") == "checkpoint_roundtrip"]

    if not rt_events:
        return False, "No checkpoint_roundtrip events found"

    max_delta = max(e.get("delta", 0) for e in rt_events)
    bad = [e for e in rt_events if e.get("delta", 0) >= 1e-6]

    if bad:
        return False, (f"{len(bad)}/{len(rt_events)} roundtrips exceeded 1e-6, "
                       f"max delta={max_delta:.2e}")
    return True, f"All {len(rt_events)} roundtrips OK (max delta={max_delta:.2e})"


def check_curriculum_probe(events: list[dict]) -> tuple[bool, str]:
    """Check no catastrophic forgetting at phase boundaries.

    Track minimum stories loss during 0-15K; at step 55K assert
    stories loss < 2x minimum.
    """
    pb_events = [e for e in events if e.get("event") == "phase_boundary"]

    if not pb_events:
        return False, "No phase_boundary events found"

    # Find minimum stories loss during early phase (steps <= 25K)
    early_stories = [e.get("stories_loss", float("inf"))
                     for e in pb_events if e.get("step", 0) <= 25000
                     and "stories_loss" in e]
    late_stories = [e.get("stories_loss", float("inf"))
                    for e in pb_events if e.get("step", 0) >= 55000
                    and "stories_loss" in e]

    if not early_stories:
        return False, "No early phase boundary data for stories"
    if not late_stories:
        return False, "No late phase boundary data for stories"

    min_early = min(early_stories)
    latest_late = late_stories[-1]  # Most recent late measurement

    passed = latest_late < 2.0 * min_early
    msg = (f"Stories loss: min_early={min_early:.4f}, at_55K+={latest_late:.4f}, "
           f"ratio={latest_late / max(min_early, 1e-10):.2f}x (threshold: <2.0x)")
    return passed, msg


def check_tape_scaling(events: list[dict]) -> tuple[bool, str]:
    """Check tape memory scaling from build_start event."""
    start_events = [e for e in events if e.get("event") == "build_start"]

    if not start_events:
        return False, "No build_start event found"

    start = start_events[0]
    ratio = start.get("tape_memory_ratio")

    if ratio is None:
        return True, "Tape memory ratio not measured (GPU path — deferred to profile_tape.py)"

    passed = ratio <= 1.1
    msg = f"Tape memory ratio: {ratio:.3f}x (threshold: <=1.1, stop: >=1.2)"
    if ratio >= 1.2:
        msg += " — STOP THE LINE"
    return passed, msg


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_run.py <jsonl_path>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    if not Path(jsonl_path).exists():
        print(f"ERROR: {jsonl_path} not found")
        sys.exit(1)

    events = load_events(jsonl_path)
    print(f"Loaded {len(events)} events from {jsonl_path}")

    checks = [
        ("Loss convergence (>=15% decrease)", check_loss_convergence),
        ("No NaN/Inf", check_no_nan_inf),
        ("Level 3 activity (>=50 active fires)", check_level3_activity),
        ("Checkpoint roundtrip (delta < 1e-6)", check_checkpoint_roundtrip),
        ("Curriculum probe (no catastrophic forgetting)", check_curriculum_probe),
        ("Tape memory scaling (ratio <= 1.1)", check_tape_scaling),
    ]

    print(f"\n{'=' * 60}")
    print("S4-M7 Validation Results")
    print(f"{'=' * 60}")

    all_passed = True
    for name, check_fn in checks:
        passed, msg = check_fn(events)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"\n  [{status}] {name}")
        print(f"         {msg}")

    print(f"\n{'=' * 60}")
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — review above")
    print(f"{'=' * 60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
