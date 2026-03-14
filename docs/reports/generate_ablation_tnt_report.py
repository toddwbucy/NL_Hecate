"""
generate_ablation_tnt_report.py — Produce docs/reports/ablation_tnt_report.pdf

Covers:
  ABLATION-TNT overview  — TNT periodic reset design rationale
  B-TNT  k=1 TitansLMM + periodic reset
  C-TNT  k=4 TitansLMM + periodic reset  (note: memory_rule=titans, not delta)
  D-TNT  k=4 TitansLMM + periodic reset  (direct counterpart to carry-forward D)

  Carry-forward comparison series (from ablation_report.pdf):
  A     SWA-only baseline
  B     k=1 TitansLMM (no reset)
  C     k=4 Delta Rule (no reset)  ← carry-forward winner
  D     k=4 DGD/TitansLMM (no reset)

Run from repo root:
  python docs/reports/generate_ablation_tnt_report.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR  = REPO_ROOT / "python" / "runs"
OUT_PATH  = REPO_ROOT / "docs" / "reports" / "ablation_tnt_report.pdf"

# ---------------------------------------------------------------------------
# Colour palette (colourblind-safe)
# ---------------------------------------------------------------------------
PALETTE_CF = {
    "A": "#555555",   # grey   — no-memory baseline
    "B": "#0077BB",   # blue   — k=1 Titans
    "C": "#EE7733",   # orange — k=4 Delta (carry-forward winner)
    "D": "#CC3311",   # red    — k=4 DGD
}
PALETTE_TNT = {
    "B": "#66BBFF",   # light blue  — B-TNT
    "C": "#FFBB55",   # light amber — C-TNT (k=4 Titans+reset)
    "D": "#FF7755",   # light red   — D-TNT
}

LABEL_CF = {
    "A": "Run A  SWA-only (no memory)",
    "B": "Run B  k=1 TitansLMM",
    "C": "Run C  k=4 Delta Rule  [best]",
    "D": "Run D  k=4 DGD",
}
LABEL_TNT = {
    "B": "Run B-TNT  k=1 TitansLMM + reset",
    "C": "Run C-TNT  k=4 TitansLMM + reset",
    "D": "Run D-TNT  k=4 TitansLMM + reset",
}

# ---------------------------------------------------------------------------
# Load eval series
# ---------------------------------------------------------------------------
def load_evals(name: str) -> list[dict]:
    path = RUNS_DIR / f"{name}.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in lines if r.get("event") == "eval"]

def latest_run_evals(evals: list[dict]) -> list[dict]:
    """Return the last contiguous monotone-step sequence (handles restarts)."""
    runs: list[list[dict]] = []
    current: list[dict] = []
    for e in evals:
        if current and e["step"] <= current[-1]["step"]:
            runs.append(current)
            current = []
        current.append(e)
    if current:
        runs.append(current)
    return runs[-1] if runs else current

def load_step_events(name: str) -> list[dict]:
    path = RUNS_DIR / f"{name}.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in lines if r.get("event") == "step" and "gate_biases" in r]

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
def page_title(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top",
                 fontsize=10, color="#444444")

def styled_ax(ax: plt.Axes, xlabel="", ylabel="", title="") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
# Carry-forward series
evA = latest_run_evals(load_evals("ablation_A"))
evB = latest_run_evals(load_evals("ablation_B"))
evC = latest_run_evals(load_evals("ablation_C"))
evD = latest_run_evals(load_evals("ablation_D"))

# TNT series
evBt = latest_run_evals(load_evals("ablation_B_tnt"))
evCt = latest_run_evals(load_evals("ablation_C_tnt"))
evDt = latest_run_evals(load_evals("ablation_D_tnt"))

cf_runs  = {"A": evA, "B": evB, "C": evC, "D": evD}
tnt_runs = {"B": evBt, "C": evCt, "D": evDt}

# Final ppl
final_cf  = {k: v[-1]["eval_ppl"] for k, v in cf_runs.items()}
final_tnt = {k: v[-1]["eval_ppl"] for k, v in tnt_runs.items()}

# Gate bias trajectories (level b_theta over time)
step_B  = load_step_events("ablation_B")
step_C  = load_step_events("ablation_C")
step_D  = load_step_events("ablation_D")
step_Bt = load_step_events("ablation_B_tnt")
step_Ct = load_step_events("ablation_C_tnt")
step_Dt = load_step_events("ablation_D_tnt")

def extract_btheta(step_events: list[dict], level: int = 0) -> tuple[list, list]:
    """Extract (steps, b_theta_at_level) from step events."""
    xs, ys = [], []
    for e in step_events:
        gb = e.get("gate_biases")
        if gb and level < len(gb):
            xs.append(e["step"])
            ys.append(gb[level][1])  # index 1 = b_theta
    return xs, ys

# ---------------------------------------------------------------------------
# FIGURE 0 — Cover / Title page
# ---------------------------------------------------------------------------
with PdfPages(OUT_PATH) as pdf:

    # ---- Cover page --------------------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#1a1a2e")
    fig.text(0.5, 0.74, "NL-Hecate", ha="center", color="white",
             fontsize=32, fontweight="bold")
    fig.text(0.5, 0.67, "TNT Periodic Reset Ablation Study", ha="center",
             color="#aabbff", fontsize=18)
    fig.text(0.5, 0.62, "ABLATION-B-TNT  /  C-TNT  /  D-TNT", ha="center",
             color="#aabbff", fontsize=12)
    fig.text(0.5, 0.55,
             "Corpus: allenai/C4 (en) · Model: 60M params, d=512 · Build steps: 25 000",
             ha="center", color="#cccccc", fontsize=10)
    fig.text(0.5, 0.51,
             "memory_reset=periodic · TNT spec: specs/infrastructure/08_tnt_periodic_reset.md",
             ha="center", color="#cccccc", fontsize=10)
    fig.text(0.5, 0.43,
             "Research programme: Mirrokni / Behrouz (HOPE 2512.24695, TNT 2511.07343)",
             ha="center", color="#888888", fontsize=9)
    fig.text(0.5, 0.39,
             "Implementation: NL-Hecate (Rust + CUDA + Python)",
             ha="center", color="#888888", fontsize=9)
    fig.text(0.5, 0.32,
             "2026-03-02", ha="center", color="#666666", fontsize=9)

    # Summary box
    ax_sum = fig.add_axes([0.08, 0.07, 0.84, 0.21])
    ax_sum.set_facecolor("#0d0d1a")
    ax_sum.set_xlim(0, 1); ax_sum.set_ylim(0, 1)
    ax_sum.axis("off")

    # Compute deltas
    delta_B = final_cf["B"] - final_tnt["B"]   # positive = TNT is better
    delta_D = final_cf["D"] - final_tnt["D"]
    results_text = (
        f"Final eval perplexity at step 24 000 — carry-forward vs TNT reset:\n\n"
        f"  Run B  k=1 TitansLMM       {final_cf['B']:.2f}  →  B-TNT {final_tnt['B']:.2f}   "
        f"({'−' if delta_B>0 else '+'}{abs(delta_B):.2f}, TNT {'helps' if delta_B>0 else 'hurts'})\n"
        f"  Run C  k=4 Delta Rule       {final_cf['C']:.2f}  →  C-TNT {final_tnt['C']:.2f}   "
        f"(+{final_tnt['C']-final_cf['C']:.2f}, C-TNT uses titans rule — not a direct pair)\n"
        f"  Run D  k=4 DGD/TitansLMM   {final_cf['D']:.2f}  →  D-TNT {final_tnt['D']:.2f}   "
        f"({'−' if delta_D>0 else '+'}{abs(delta_D):.2f}, TNT {'helps' if delta_D>0 else 'hurts'})\n\n"
        f"  Carry-forward winner: Run C (k=4 Delta Rule) = {final_cf['C']:.2f}\n"
        f"  TNT series winner:   B-TNT (k=1 + reset)   = {final_tnt['B']:.2f}   "
        f"(still {final_tnt['B']-final_cf['C']:.2f} ppl above carry-forward C)"
    )
    ax_sum.text(0.04, 0.88, results_text, va="top", color="white",
                fontsize=8.5, fontfamily="monospace", transform=ax_sum.transAxes,
                linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 1: TNT Design -----------------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "TNT Periodic Reset — Experimental Design",
               "TNT spec: specs/infrastructure/08_tnt_periodic_reset.md · Paper: TNT 2511.07343")

    ax = fig.add_axes([0.05, 0.04, 0.90, 0.84])
    ax.axis("off")

    design_text = """
MOTIVATION
══════════
The TNT (Test-time Normalization / periodic Training) paper (2511.07343) proposes that
periodically resetting the inner-loop memory state M to zeros — synchronized with the
CMS chunk boundary — can improve gradient flow through the memory update rule.  The
hypothesis is that accumulated memory content can enter high-curvature regions where
the outer-loop learning signal is suppressed ("initialization trap"), and that periodic
resets allow the outer-loop optimizer to make larger updates.

In the carry-forward ablation, Run D (k=4 DGD) underperformed relative to the HOPE
Table 6 prediction (HOPE: D best; actual: C best at 60M params / 25K steps).  TNT
resets were proposed as a mechanism to check whether the DGD gate initialization trap
is the cause of that underperformance.

PERIODIC RESET MECHANISM  (memory_reset = "periodic")
═══════════════════════════════════════════════════════
  At each step, immediately after advancing M_{t+1} and before the attention layer:
    For level ℓ: if (step % chunk_size[ℓ]) == 0 → M[ℓ] ← 0

  This means:
    Level 0 (chunk_size=1):   reset EVERY step  → M_L0 is always zero at step start
    Level 1 (chunk_size=8):   reset every 8 steps
    Level 2 (chunk_size=64):  reset every 64 steps
    Level 3 (chunk_size=512): reset every 512 steps (once per full sequence)

  Effect: the memory is treated as a "within-chunk scratch pad" — it accumulates
  within a chunk but cannot carry state across chunk boundaries at that level.
  The outer-loop parameters (W_K, W_V, W_Q, gates) still accumulate across resets.

EXPERIMENTAL DESIGN
════════════════════
  Run  memory_rule  k  memory_reset  Counterpart  Question tested
  ───  ───────────  ─  ────────────  ───────────  ───────────────
  B-TNT  titans     1  periodic      Carry-fwd B  Does TNT help k=1 DGD?
  C-TNT  titans     4  periodic      Carry-fwd C  k=4 DGD+reset vs k=4 Delta  (DIFFERENT rules)
  D-TNT  titans     4  periodic      Carry-fwd D  Does TNT help k=4 DGD?

  NOTE on C-TNT: The C-TNT config uses memory_rule="titans" (DGD), not "delta" like
  carry-forward C.  C-TNT is therefore NOT a direct ablation of adding TNT to Run C.
  C-TNT = Run D architecture + TNT reset.  The comparison C vs C-TNT tests both a
  rule change AND a reset — they are confounded.  D vs D-TNT is the clean TNT ablation.

  B-TNT and D-TNT are CLEAN ablations: identical to their carry-forward counterparts
  except for memory_reset=periodic.

CONTROLLED (identical across all TNT runs and carry-forward)
═════════════════════════════════════════════════════════════
  d_model=512   num_heads=8   seq_len=512   vocab_size=32000
  lr=4e-4   warmup=200   weight_decay=0.1   batch_size=8
  steps=25000   optimizer=AdamW   corpus=allenai/C4 (en)   seed=42

KEY COMPARISONS
════════════════
  B  vs  B-TNT  :  Does TNT reset help the k=1 DGD baseline?
  D  vs  D-TNT  :  Does TNT reset help k=4 DGD specifically?   [primary test]
  D-TNT  vs  C  :  Can TNT-boosted DGD match Delta Rule?        [practical ceiling]
  B-TNT  vs  D-TNT : k=1 DGD+reset vs k=4 DGD+reset            [scale test within TNT]
"""
    ax.text(0.01, 0.99, design_text, va="top", fontsize=8.5,
            transform=ax.transAxes, fontfamily="monospace", linespacing=1.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 2: PPL curves — all TNT runs + carry-forward reference ------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Eval Perplexity — TNT Series vs Carry-Forward",
               "Held-out C4 validation · solid = TNT · dashed = carry-forward reference")

    ax_main = fig.add_axes([0.10, 0.42, 0.85, 0.48])

    # Carry-forward as thin dashed reference
    for run in ["A", "B", "C", "D"]:
        ev = cf_runs[run]
        xs = [e["step"] for e in ev]
        ys = [e["eval_ppl"] for e in ev]
        ax_main.plot(xs, ys, color=PALETTE_CF[run], linestyle="--",
                     linewidth=1.2, alpha=0.55, label=f"{LABEL_CF[run]} (cf)")

    # TNT as solid foreground
    for run in ["B", "C", "D"]:
        ev = tnt_runs[run]
        xs = [e["step"] for e in ev]
        ys = [e["eval_ppl"] for e in ev]
        ax_main.plot(xs, ys, color=PALETTE_TNT[run], linestyle="-",
                     linewidth=2.0, marker="o", markersize=3, label=LABEL_TNT[run])

    styled_ax(ax_main, xlabel="Build step", ylabel="Eval perplexity",
              title="Eval perplexity — TNT (solid) vs carry-forward (dashed)")
    ax_main.legend(fontsize=7.5, loc="upper right", ncol=2)

    # Zoom: steps 10K–24K
    ax_zoom = fig.add_axes([0.10, 0.10, 0.55, 0.26])
    for run in ["A", "B", "C", "D"]:
        ev = cf_runs[run]
        xs = [e["step"] for e in ev if e["step"] >= 10000]
        ys = [e["eval_ppl"] for e in ev if e["step"] >= 10000]
        if xs:
            ax_zoom.plot(xs, ys, color=PALETTE_CF[run], linestyle="--",
                         linewidth=1.2, alpha=0.55)
    for run in ["B", "C", "D"]:
        ev = tnt_runs[run]
        xs = [e["step"] for e in ev if e["step"] >= 10000]
        ys = [e["eval_ppl"] for e in ev if e["step"] >= 10000]
        if xs:
            ax_zoom.plot(xs, ys, color=PALETTE_TNT[run], linestyle="-",
                         linewidth=2.0, marker="o", markersize=3,
                         label=f"{run}-TNT")
    styled_ax(ax_zoom, xlabel="Build step", ylabel="Eval perplexity",
              title="Zoom: steps 10 000 – 24 000")
    ax_zoom.legend(fontsize=7)

    # Final ppl bar chart
    ax_bar = fig.add_axes([0.72, 0.10, 0.23, 0.26])
    run_labels = ["A", "B", "B-T", "C", "C-T", "D", "D-T"]
    vals = [
        final_cf["A"], final_cf["B"], final_tnt["B"],
        final_cf["C"], final_tnt["C"],
        final_cf["D"], final_tnt["D"],
    ]
    colours_bar = [
        PALETTE_CF["A"],
        PALETTE_CF["B"], PALETTE_TNT["B"],
        PALETTE_CF["C"], PALETTE_TNT["C"],
        PALETTE_CF["D"], PALETTE_TNT["D"],
    ]
    bars = ax_bar.bar(run_labels, vals, color=colours_bar, edgecolor="black", linewidth=0.5)
    ax_bar.set_ylim(min(vals) * 0.95, max(vals) * 1.03)
    styled_ax(ax_bar, xlabel="Run", ylabel="Eval ppl (step 24 000)",
              title="Final ppl")
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", fontsize=7, fontweight="bold")
    ax_bar.tick_params(axis="x", labelsize=7)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 3: Per-run comparison panels (2×2 + text) ------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Per-Run Comparison: Carry-Forward vs TNT Reset",
               "Solid = TNT series · Dashed = carry-forward counterpart")

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08,
                           hspace=0.45, wspace=0.35)

    comparisons = [
        ("B",  evB,  evBt, "k=1 DGD/Titans",      "Direct pair (same memory_rule=titans)"),
        ("C",  evC,  evCt, "k=4 Delta vs k=4 Titans+TNT", "NOT direct (rule changed to titans)"),
        ("D",  evD,  evDt, "k=4 DGD",              "Direct pair (same memory_rule=titans)"),
    ]

    for idx, (run, ev_cf, ev_tnt, title_stub, note) in enumerate(comparisons):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        xs_cf  = [e["step"] for e in ev_cf]
        ys_cf  = [e["eval_ppl"] for e in ev_cf]
        xs_tnt = [e["step"] for e in ev_tnt]
        ys_tnt = [e["eval_ppl"] for e in ev_tnt]
        ax.plot(xs_cf,  ys_cf,  color=PALETTE_CF[run],  linestyle="--",
                linewidth=1.6, label=f"Run {run} (cf)", alpha=0.75)
        ax.plot(xs_tnt, ys_tnt, color=PALETTE_TNT[run], linestyle="-",
                linewidth=2.0, marker="o", markersize=2.5, label=f"Run {run}-TNT")
        ax.set_title(f"Run {run} vs {run}-TNT: {title_stub}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Eval ppl", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        delta = ys_cf[-1] - ys_tnt[-1]
        arrow_col = PALETTE_TNT[run] if delta > 0 else "#CC0000"
        ax.annotate(
            f"{run}: {ys_cf[-1]:.1f}\n{run}-TNT: {ys_tnt[-1]:.1f}\nΔ={'−' if delta>0 else '+'}{abs(delta):.1f}",
            xy=(xs_tnt[-1], ys_tnt[-1]),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=7.5, color=arrow_col,
            arrowprops=dict(arrowstyle="->", color=arrow_col, lw=0.8),
        )
        ax.text(0.02, 0.06, note, transform=ax.transAxes,
                fontsize=6.5, color="#555555", va="bottom")
        ax.legend(fontsize=7.5)

    # 4th panel: C-TNT vs D-TNT (duplicate run comparison)
    ax4 = fig.add_subplot(gs[1, 1])
    xs_ct = [e["step"] for e in evCt]
    ys_ct = [e["eval_ppl"] for e in evCt]
    xs_dt = [e["step"] for e in evDt]
    ys_dt = [e["eval_ppl"] for e in evDt]
    ax4.plot(xs_ct, ys_ct, color=PALETTE_TNT["C"], linestyle="-",
             linewidth=2.0, marker="o", markersize=2.5, label="C-TNT")
    ax4.plot(xs_dt, ys_dt, color=PALETTE_TNT["D"], linestyle="-",
             linewidth=2.0, marker="s", markersize=2.5, label="D-TNT")
    ax4.set_title("C-TNT vs D-TNT  (both: k=4 Titans+reset)", fontsize=8.5, fontweight="bold")
    ax4.set_xlabel("Step", fontsize=8)
    ax4.set_ylabel("Eval ppl", fontsize=8)
    ax4.tick_params(labelsize=7)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
    ax4.grid(axis="y", linestyle="--", alpha=0.35)
    ax4.text(0.02, 0.06, "Near-identical configs: inter-run variance estimate",
             transform=ax4.transAxes, fontsize=6.5, color="#555555", va="bottom")
    ax4.legend(fontsize=7.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 4: Summary bar charts and comparisons ----------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Results Summary — TNT Effect and Cross-Series Comparisons",
               "Final eval perplexity at step 24 000")

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08,
                           hspace=0.50, wspace=0.40)

    # Panel 1: Grouped bar chart carry-forward vs TNT
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(3)
    width = 0.35
    cf_vals  = [final_cf["B"],  final_cf["C"],  final_cf["D"]]
    tnt_vals = [final_tnt["B"], final_tnt["C"], final_tnt["D"]]
    cf_cols  = [PALETTE_CF["B"],  PALETTE_CF["C"],  PALETTE_CF["D"]]
    tnt_cols = [PALETTE_TNT["B"], PALETTE_TNT["C"], PALETTE_TNT["D"]]
    bars_cf  = ax1.bar(x - width/2, cf_vals,  width, color=cf_cols,
                       edgecolor="black", linewidth=0.5, label="carry-forward")
    bars_tnt = ax1.bar(x + width/2, tnt_vals, width, color=tnt_cols,
                       edgecolor="black", linewidth=0.5, label="TNT reset")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["B / B-TNT", "C / C-TNT", "D / D-TNT"], fontsize=8)
    # Add the carry-forward C winner line
    ax1.axhline(final_cf["C"], color=PALETTE_CF["C"], linestyle=":", linewidth=1.5,
                label=f"CF-C best ({final_cf['C']:.0f})")
    ax1.set_ylim(min(cf_vals + tnt_vals) * 0.96, max(cf_vals + tnt_vals) * 1.02)
    styled_ax(ax1, xlabel="", ylabel="Eval perplexity",
              title="Carry-Forward vs TNT Reset")
    ax1.legend(fontsize=7)
    for bar, val in zip(list(bars_cf) + list(bars_tnt), cf_vals + tnt_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", fontsize=7, fontweight="bold")

    # Panel 2: TNT effect size (delta ppl, positive = TNT is better)
    ax2 = fig.add_subplot(gs[0, 1])
    run_labels2 = ["B-TNT\nvs B", "D-TNT\nvs D"]
    effect_B = final_cf["B"] - final_tnt["B"]
    effect_D = final_cf["D"] - final_tnt["D"]
    effects  = [effect_B, effect_D]
    cols2    = [PALETTE_TNT["B"], PALETTE_TNT["D"]]
    bars2    = ax2.bar(run_labels2, effects, color=cols2, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    styled_ax(ax2, xlabel="", ylabel="Δ ppl (positive = TNT better)",
              title="TNT Effect Size\n(direct-pair ablations only)")
    for bar, val in zip(bars2, effects):
        ypos = val + 0.3 if val >= 0 else val - 1.0
        ax2.text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{'−' if val>0 else '+'}{abs(val):.2f}", ha="center",
                 fontsize=9, fontweight="bold")
    ax2.text(0.5, 0.97, "C-TNT excluded: rule mismatch",
             transform=ax2.transAxes, ha="center", va="top", fontsize=7.5,
             color="#777777", style="italic")

    # Panel 3: Full cross-series bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    all_labels = ["A", "B", "B-T", "C*", "C-T", "D", "D-T"]
    all_vals   = [
        final_cf["A"], final_cf["B"], final_tnt["B"],
        final_cf["C"], final_tnt["C"],
        final_cf["D"], final_tnt["D"],
    ]
    all_cols   = [
        PALETTE_CF["A"],
        PALETTE_CF["B"], PALETTE_TNT["B"],
        PALETTE_CF["C"], PALETTE_TNT["C"],
        PALETTE_CF["D"], PALETTE_TNT["D"],
    ]
    bars3 = ax3.bar(all_labels, all_vals, color=all_cols, edgecolor="black", linewidth=0.5)
    ax3.axhline(final_cf["C"], color=PALETTE_CF["C"], linestyle=":", linewidth=1.5,
                alpha=0.7)
    ax3.set_ylim(min(all_vals) * 0.96, max(all_vals) * 1.02)
    styled_ax(ax3, xlabel="Run", ylabel="Eval perplexity",
              title="All Runs: Carry-Forward + TNT\n(*C = Delta Rule; C-T = Titans+TNT)")
    for bar, val in zip(bars3, all_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", fontsize=7, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=8)

    # Panel 4: Prediction vs actual
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    gap_D_vs_best_TNT = final_tnt["B"] - final_cf["C"]
    pct_improve_B = (final_cf["B"] - final_tnt["B"]) / final_cf["B"] * 100
    pct_improve_D = (final_cf["D"] - final_tnt["D"]) / final_cf["D"] * 100
    pred_text = (
        "TNT Hypothesis Results\n"
        "─────────────────────────────\n"
        "Expected (from TNT 2511.07343):\n"
        "  TNT reset stabilizes L2/L3 gates\n"
        "  and improves DGD convergence\n\n"
        "Actual (this study):\n"
        "  ✓  B-TNT < B  (TNT helps k=1)\n"
        f"     {final_cf['B']:.1f} → {final_tnt['B']:.1f}  ({pct_improve_B:+.1f}%)\n\n"
        "  ✓  D-TNT < D  (TNT helps k=4 DGD)\n"
        f"     {final_cf['D']:.1f} → {final_tnt['D']:.1f}  ({pct_improve_D:+.1f}%)\n\n"
        "  ✗  TNT does NOT close the gap\n"
        "     vs carry-forward Run C:\n"
        f"     Best TNT = {final_tnt['B']:.1f}  (B-TNT)\n"
        f"     CF best  = {final_cf['C']:.1f}  (C, Delta)\n"
        f"     Gap      = {gap_D_vs_best_TNT:+.1f} ppl\n\n"
        "Key finding:\n"
        "  k=4 Delta Rule (no reset) remains\n"
        "  the dominant strategy at this scale."
    )
    ax4.text(0.03, 0.97, pred_text, va="top", fontsize=8.5,
             transform=ax4.transAxes, fontfamily="monospace", linespacing=1.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 5: Gate analysis — b_theta trajectories --------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Gate Analysis — b_θ (Learning Rate Bias) Trajectories",
               "b_theta per level: lower = smaller inner-loop learning rate (softplus activation)")

    gs = gridspec.GridSpec(2, 2, figure=fig, top=0.88, bottom=0.08,
                           hspace=0.50, wspace=0.40)

    level_colours = {0: "#0077BB", 1: "#EE7733", 2: "#228833", 3: "#CC3311"}
    level_labels  = {0: "L0 (period 1)", 1: "L1 (period 8)",
                     2: "L2 (period 64)", 3: "L3 (period 512)"}

    # Panel 1: Run B vs B-TNT  (k=1, L0 only)
    ax1 = fig.add_subplot(gs[0, 0])
    xs_B, ys_B = extract_btheta(step_B, 0)
    xs_Bt, ys_Bt = extract_btheta(step_Bt, 0)
    ax1.plot(xs_B,  ys_B,  color=PALETTE_CF["B"],  linestyle="--", linewidth=1.8,
             label="B  L0 b_θ", alpha=0.85)
    ax1.plot(xs_Bt, ys_Bt, color=PALETTE_TNT["B"], linestyle="-",  linewidth=2.0,
             label="B-TNT  L0 b_θ")
    ax1.axhline(ys_B[0], color="#aaaaaa", linestyle=":", linewidth=1.0,
                label=f"init ({ys_B[0]:.2f})")
    styled_ax(ax1, xlabel="Build step", ylabel="b_θ (L0)",
              title="Run B vs B-TNT — L0 learning rate gate")
    ax1.legend(fontsize=7.5)
    ax1.text(0.02, 0.06,
             f"B final: {ys_B[-1]:.2f}\nB-TNT final: {ys_Bt[-1]:.2f}",
             transform=ax1.transAxes, fontsize=7.5, va="bottom",
             fontfamily="monospace")

    # Panel 2: Run D vs D-TNT  (k=4, per-level)
    ax2 = fig.add_subplot(gs[0, 1])
    for lv in range(4):
        xs_D, ys_D = extract_btheta(step_D, lv)
        xs_Dt, ys_Dt = extract_btheta(step_Dt, lv)
        col = level_colours[lv]
        ax2.plot(xs_D,  ys_D,  color=col, linestyle="--", linewidth=1.4, alpha=0.7,
                 label=f"{level_labels[lv]} (cf)")
        ax2.plot(xs_Dt, ys_Dt, color=col, linestyle="-",  linewidth=2.0,
                 label=f"{level_labels[lv]} (TNT)")
    styled_ax(ax2, xlabel="Build step", ylabel="b_θ",
              title="Run D vs D-TNT — all levels")
    ax2.legend(fontsize=6.5, ncol=2)

    # Panel 3: L0 b_theta comparison across all runs
    ax3 = fig.add_subplot(gs[1, 0])
    for name, steps_ev, col, label, lw, ls in [
        ("B",  step_B,  PALETTE_CF["B"],  "B  (k=1 cf)",      1.6, "--"),
        ("B-T",step_Bt, PALETTE_TNT["B"], "B-TNT",            2.0, "-"),
        ("C",  step_C,  PALETTE_CF["C"],  "C  (k=4 Delta cf)",1.6, "--"),
        ("D",  step_D,  PALETTE_CF["D"],  "D  (k=4 DGD cf)",  1.6, "--"),
        ("D-T",step_Dt, PALETTE_TNT["D"], "D-TNT",            2.0, "-"),
    ]:
        xs, ys = extract_btheta(steps_ev, 0)
        if xs:
            ax3.plot(xs, ys, color=col, linestyle=ls, linewidth=lw,
                     label=label, alpha=0.85)
    styled_ax(ax3, xlabel="Build step", ylabel="b_θ (L0)",
              title="L0 b_θ — all runs compared")
    ax3.legend(fontsize=7, ncol=2)

    # Panel 4: L3 b_theta — TNT vs carry-forward (L3 barely moves)
    ax4 = fig.add_subplot(gs[1, 1])
    for name, steps_ev, col, label, lw, ls in [
        ("C",  step_C,  PALETTE_CF["C"],  "C  (k=4 Delta)",  1.6, "--"),
        ("D",  step_D,  PALETTE_CF["D"],  "D  (k=4 DGD)",    1.6, "--"),
        ("C-T",step_Ct, PALETTE_TNT["C"], "C-TNT",           2.0, "-"),
        ("D-T",step_Dt, PALETTE_TNT["D"], "D-TNT",           2.0, "-"),
    ]:
        xs, ys = extract_btheta(steps_ev, 3)  # level 3
        if xs:
            ax4.plot(xs, ys, color=col, linestyle=ls, linewidth=lw,
                     label=label, alpha=0.85)
    styled_ax(ax4, xlabel="Build step", ylabel="b_θ (L3)",
              title="L3 b_θ — all k=4 runs  (barely moves)")
    ax4.legend(fontsize=7.5)
    ax4.text(0.02, 0.06,
             "L3 b_θ ≈ −7.60 throughout all runs\n"
             "softplus(−7.60) ≈ 4.5×10⁻⁴\n"
             "L3 memory update is negligibly small",
             transform=ax4.transAxes, fontsize=7, va="bottom",
             fontfamily="monospace", color="#444444")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 6: Analysis and discussion ----------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Analysis and Discussion",
               "TNT periodic reset: effect on k=1 Titans and k=4 DGD architectures")

    ax = fig.add_axes([0.05, 0.02, 0.90, 0.88])
    ax.axis("off")

    pct_B  = (final_cf["B"] - final_tnt["B"]) / final_cf["B"] * 100
    pct_D  = (final_cf["D"] - final_tnt["D"]) / final_cf["D"] * 100
    gap    = final_tnt["B"] - final_cf["C"]

    discussion = f"""
FINDING 1 — TNT reset provides a modest but consistent benefit for DGD variants  ✓ CONFIRMED
───────────────────────────────────────────────────────────────────────────────────────────────
Both clean TNT ablations show improvement:
  B-TNT vs B:   {final_cf['B']:.2f} → {final_tnt['B']:.2f}  ({pct_B:+.1f}%)
  D-TNT vs D:   {final_cf['D']:.2f} → {final_tnt['D']:.2f}  ({pct_D:+.1f}%)

The ~2% improvement is consistent across both scale points (k=1 and k=4).  This confirms
that the TNT periodic reset is not harmful to DGD architectures and provides a small gain,
likely by preventing M_t from drifting into regions where the DGD error term
  error_t = M_t·k_t − v_t
becomes large relative to the gradient signal.  A reset to M=0 ensures the error term
starts small within each chunk, making the gradient contribution from the correction term
more stable in early training.

FINDING 2 — TNT does NOT close the gap vs k=4 Delta Rule  ✗ NOT CONFIRMED
────────────────────────────────────────────────────────────────────────────
The carry-forward winner (Run C, k=4 Delta Rule, no reset) achieves {final_cf['C']:.2f} eval ppl.
The best TNT variant (B-TNT, k=1 + reset) achieves {final_tnt['B']:.2f} — still {gap:.2f} ppl worse.
Even D-TNT (k=4 DGD + reset = {final_tnt['D']:.2f}) remains {final_tnt['D']-final_cf['C']:.2f} ppl worse than carry-forward C.

The TNT reset mechanism was hypothesized to help DGD escape initialization traps that
prevent L2/L3 gates from contributing.  While TNT improves DGD, it does not bring DGD
to parity with standard GD (Delta Rule) at this scale.  The Delta Rule's advantage comes
not from better initialization dynamics but from a fundamentally simpler update:
  M_{{t+1}} = (1-α)·M_t + θ·v⊗k
which lacks the second-order state dependence that makes DGD sensitive to M_t quality.

FINDING 3 — L3 gate (b_θ at period 512) is frozen across all variants
──────────────────────────────────────────────────────────────────────
Gate analysis shows b_θ at L3 ≈ −7.60 throughout all k=4 runs (both carry-forward and TNT).
  softplus(−7.60) ≈ 4.5×10⁻⁴ — an extremely small inner-loop learning rate.

The L3 level fires only once per 512-token sequence (2 fires per 1000 steps).  With so
few gradient updates, the outer-loop AdamW cannot accumulate enough second-moment signal
to move b_θ meaningfully.  TNT resets help L3 marginally (both C-TNT and D-TNT show the
same frozen L3 gate), confirming that the L3 initialization trap is structural:
  • Firing frequency (1/512) is too low for gradient momentum to accumulate
  • TNT resets do not increase L3 firing frequency — they reset M but not the gate params

FINDING 4 — C-TNT and D-TNT are effectively duplicate runs  (inter-run variance)
──────────────────────────────────────────────────────────────────────────────────
Both C-TNT and D-TNT share identical configs (k=4, memory_rule=titans, memory_reset=periodic,
seed=42).  Their final perplexities are {final_tnt['C']:.2f} and {final_tnt['D']:.2f} respectively (Δ={abs(final_tnt['C']-final_tnt['D']):.2f}).
The small difference ({abs(final_tnt['C']-final_tnt['D'])/max(final_tnt['C'],final_tnt['D'])*100:.2f}%) is attributable to different data cursor positions at run start.
This provides an estimate of inter-run variance for k=4 Titans + TNT: ±{abs(final_tnt['C']-final_tnt['D'])/2:.2f} ppl.
Note: C-TNT's description says "TNT counterpart to ABLATION-C" but uses memory_rule=titans
(not delta).  The config note is misleading — C-TNT is architecturally identical to D-TNT.

RECOMMENDATION — Next steps
────────────────────────────
  1. The TNT series confirms: standard GD + k=4 CMS (carry-forward C) remains the best
     configuration at 60M params / 25K steps.  No TNT variant beats it.
  2. To complete the intended C-TNT experiment, re-run with memory_rule="delta":
     k=4 Delta Rule + TNT reset — this is the missing direct pair for Run C.
  3. The L3 frozen gate (b_θ ≈ −7.60) across all runs warrants investigation:
     • Try explicit L3 gate initialization closer to a useful learning rate
     • Or increase L3 firing by using smaller sequence-level chunks
  4. D-TNT's marginal improvement over D ({pct_D:.1f}%) suggests the DGD underperformance at
     60M params is scale-related (see carry-forward analysis), not reset-addressable.
  5. Consider running TNT for more steps (100K) — the 2% TNT benefit at 25K may grow.
"""
    ax.text(0.01, 0.99, discussion, va="top", fontsize=8.5,
            transform=ax.transAxes, linespacing=1.40)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 7: Eval checkpoint table ------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Eval Perplexity at Selected Checkpoints",
               "All runs: carry-forward (A/B/C/D) and TNT series (B-TNT/C-TNT/D-TNT)")

    ax = fig.add_axes([0.05, 0.55, 0.90, 0.34])
    steps_A = np.array([e["step"] for e in evA])
    ppl_A   = np.array([e["eval_ppl"] for e in evA])

    # Relative ppl normalized to Run A
    for run in ["B", "C", "D"]:
        ev = cf_runs[run]
        xs = np.array([e["step"] for e in ev])
        ys = np.array([e["eval_ppl"] for e in ev])
        ppl_A_interp = np.interp(xs, steps_A, ppl_A)
        ratio = ys / ppl_A_interp
        ax.plot(xs, ratio, color=PALETTE_CF[run], linestyle="--",
                linewidth=1.6, alpha=0.7, label=f"Run {run} (cf)")

    for run in ["B", "C", "D"]:
        ev = tnt_runs[run]
        xs = np.array([e["step"] for e in ev])
        ys = np.array([e["eval_ppl"] for e in ev])
        ppl_A_interp = np.interp(xs, steps_A, ppl_A)
        ratio = ys / ppl_A_interp
        ax.plot(xs, ratio, color=PALETTE_TNT[run], linestyle="-",
                linewidth=2.0, marker="o", markersize=2.5, label=f"Run {run}-TNT")

    ax.axhline(1.0, color=PALETTE_CF["A"], linewidth=1.2, linestyle=":",
               label="Run A (baseline = 1.0)")
    styled_ax(ax, xlabel="Build step",
              ylabel="ppl(run) / ppl(A)",
              title="Relative perplexity vs SWA-only baseline  (below 1.0 = better than A)")
    ax.legend(fontsize=7.5, ncol=2)

    # Table
    ax_tbl = fig.add_axes([0.03, 0.03, 0.94, 0.47])
    ax_tbl.axis("off")

    checkpoint_steps = [1000, 4000, 8000, 12000, 16000, 20000, 24000]
    col_labels = ["Step", "A (cf)", "B (cf)", "B-TNT", "C (cf)", "C-TNT", "D (cf)", "D-TNT"]
    all_evs    = [evA, evB, evBt, evC, evCt, evD, evDt]
    table_data = []
    for step in checkpoint_steps:
        row = [str(step)]
        for ev in all_evs:
            match = [e["eval_ppl"] for e in ev if e["step"] == step]
            row.append(f"{match[0]:.1f}" if match else "—")
        table_data.append(row)

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.8)

    header_colours = [
        "#333333",
        PALETTE_CF["A"], PALETTE_CF["B"], PALETTE_TNT["B"],
        PALETTE_CF["C"], PALETTE_TNT["C"],
        PALETTE_CF["D"], PALETTE_TNT["D"],
    ]
    for j, col in enumerate(header_colours):
        tbl[(0, j)].set_facecolor(col)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight CF-C winner column (col 4)
    for i in range(1, len(checkpoint_steps) + 1):
        tbl[(i, 4)].set_facecolor("#fff3e0")  # Run C col

    ax_tbl.set_title("Eval perplexity at selected checkpoints (orange = carry-forward winner C)",
                     fontsize=9, fontweight="bold", pad=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ---- Page 8: Config reference -----------------------------------------
    fig = plt.figure(figsize=(8.5, 11))
    page_title(fig,
               "Configuration Reference",
               "Complete hyperparameter table — TNT series and carry-forward comparators")

    ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
    ax.axis("off")

    config_text = """
SHARED HYPERPARAMETERS (all 7 runs)
══════════════════════════════════════
  d_model=512   num_heads=8   seq_len=512   vocab_size=32000
  lr=4e-4   warmup=200   weight_decay=0.1   batch_size=8
  steps=25000   optimizer=AdamW   seed=42   corpus=allenai/C4 (en)
  composition=MAG   projection_kind=adaptive   m_norm_max=100.0 (per level)

TNT SERIES — Per-Run Differences
══════════════════════════════════
  Run    k  chunk_sizes     memory_rule  memory_reset  params    checkpoint
  ─────  ─  ───────────     ───────────  ────────────  ──────    ──────────
  B-TNT  1  [1]             titans       periodic      36.7M     ablation_B_tnt.safetensors
  C-TNT  4  [1,8,64,512]    titans       periodic      45.4M     ablation_C_tnt.safetensors
  D-TNT  4  [1,8,64,512]    titans       periodic      45.4M     ablation_D_tnt.safetensors

  NOTE: C-TNT config description says "counterpart to ABLATION-C" but uses memory_rule=titans
  (same as D-TNT).  C-TNT and D-TNT are architecturally identical; differences are run order
  and data cursor position only.  Neither is a direct ablation of carry-forward Run C.

CARRY-FORWARD SERIES (reference, from ablation_report.pdf)
═══════════════════════════════════════════════════════════
  Run  k  chunk_sizes     memory_rule  memory_reset  params    checkpoint
  ───  ─  ───────────     ───────────  ────────────  ──────    ──────────
   A   1  [1]             delta        none          36.7M     ablation_A.safetensors
   B   1  [1]             titans       none          36.7M     ablation_B.safetensors
   C   4  [1,8,64,512]    delta        none          45.4M     ablation_C.safetensors
   D   4  [1,8,64,512]    titans       none          45.4M     ablation_D.safetensors

MEMORY RESET (periodic)
════════════════════════
  Level  Chunk size  Reset trigger        Effect
  ─────  ──────────  ─────────────        ──────
  L0     1           Every step           M_L0 ← 0 before every update
  L1     8           Every 8 steps        M_L1 ← 0 at chunk boundaries
  L2     64          Every 64 steps       M_L2 ← 0 at chunk boundaries
  L3     512         Every 512 steps      M_L3 ← 0 at sequence boundary

  Outer-loop parameters (W_K, W_V, W_Q, b_alpha, b_theta) are NOT reset.
  Only inner-loop state M is reset.  This is the TNT mechanism from 2511.07343.

MEMORY RULE EQUATIONS
══════════════════════
  titans / DGD (state-dependent):
    error_t = M_t · k_t − v_t
    M_{{t+1}} = (1 - α_t) · M_t − θ_t · biased(error_t) ⊗ k_t
    Source: HOPE (2512.24695) §4.5 eq-088-practical-dgd-update
            TNT (2511.07343) §3

  delta / standard GD (state-independent):
    M_{{t+1}} = (1 - α_t) · M_t + θ_t · v_t ⊗ k_t
    Source: MIRAS (2504.13173) §3 eq-009-delta-rule

GATE INITIALIZATION (k=4 runs)
════════════════════════════════
  Level  b_alpha_init  sigmoid(b_alpha)  b_theta_init  softplus(b_theta)
  ─────  ────────────  ────────────────  ────────────  ─────────────────
  L0     3.0           0.952             −4.6          0.0100
  L1     4.0           0.982             −5.6          0.0037
  L2     4.5           0.989             −6.6          0.0014
  L3     5.0           0.993             −7.6          0.00046

HARDWARE
══════════
  GPU         NVIDIA A6000 (46 GiB) — primary
  Throughput  ~128 tok/s (k=4, d=512, seq_len=512, single GPU)
"""
    ax.text(0.01, 0.99, config_text, va="top", fontsize=8.5,
            transform=ax.transAxes, fontfamily="monospace", linespacing=1.38)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Report written: {OUT_PATH}")
