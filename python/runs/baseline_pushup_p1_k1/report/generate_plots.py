#!/usr/bin/env python3
"""Generate all plots for the baseline_pushup_p1_k1 experiment report."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ── Load data ──────────────────────────────────────────────────────────
METRICS = "../metrics.jsonl"
NIAH_FILES = {
    5000:  "../niah_step5000.json",
    10000: "../niah_step10000.json",
    15000: "../niah_step15000.json",
    20000: "../niah_step20000.json",
}

steps, losses, ppls, grad_norms, lrs = [], [], [], [], []
block_gnorms = [[], [], [], []]  # 4 blocks
l0_gnorms = [[], [], [], []]
gnorm_cvs = []

with open(METRICS) as f:
    for line in f:
        d = json.loads(line)
        if d.get("event") != "step":
            continue
        steps.append(d["step"])
        losses.append(d["loss"])
        ppls.append(d["ppl"])
        grad_norms.append(d["grad_norm"])
        lrs.append(d["lr"])
        gnorm_cvs.append(d["block_gnorm_cv"])
        for i in range(4):
            block_gnorms[i].append(d["block_grad_norms"][i])
            l0_gnorms[i].append(d["l0_block_grad_norms"][i])

steps = np.array(steps)
losses = np.array(losses)
ppls = np.array(ppls)
grad_norms = np.array(grad_norms)
lrs = np.array(lrs)
gnorm_cvs = np.array(gnorm_cvs)
block_gnorms = [np.array(b) for b in block_gnorms]
l0_gnorms = [np.array(b) for b in l0_gnorms]

# ── Smoothing helper ──────────────────────────────────────────────────
def ema(data, alpha=0.05):
    """Exponential moving average."""
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
BLOCK_LABELS = ['Block 0', 'Block 1', 'Block 2', 'Block 3']


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Loss Curve (raw + EMA)
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(steps, losses, alpha=0.15, color='#2196F3', linewidth=0.5, label='Raw')
ax1.plot(steps, ema(losses), color='#1565C0', linewidth=1.8, label='EMA (α=0.05)')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 12)

# Annotate phases
ax1.axvspan(0, 5000, alpha=0.06, color='green', label='Phase: rapid descent')
ax1.axvspan(5000, 14000, alpha=0.06, color='orange', label='Phase: plateau')
ax1.axvspan(14000, 20000, alpha=0.06, color='blue', label='Phase: late descent')
ax1.text(2500, 10.5, 'Rapid Descent', ha='center', fontsize=9, color='#2E7D32')
ax1.text(9500, 10.5, 'Plateau', ha='center', fontsize=9, color='#E65100')
ax1.text(17000, 10.5, 'Late Descent', ha='center', fontsize=9, color='#1565C0')

# LR overlay
ax2.plot(steps, lrs, color='#FF9800', linewidth=1.2)
ax2.set_ylabel('Learning Rate')
ax2.set_xlabel('Step')
ax2.set_title('Cosine LR Schedule')
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-4,-4))

fig.tight_layout()
fig.savefig('fig_loss_curve.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_curve.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Perplexity (log scale)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(steps, ppls, alpha=0.15, color='#FF5722', linewidth=0.5)
ax.semilogy(steps, ema(ppls), color='#D84315', linewidth=1.8, label='EMA')
ax.set_ylabel('Perplexity (log scale)')
ax.set_xlabel('Step')
ax.set_title('Training Perplexity')
ax.legend()
fig.tight_layout()
fig.savefig('fig_perplexity.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_perplexity.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Gradient Norms — global + per-block
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Global grad norm
axes[0].plot(steps, grad_norms, alpha=0.15, color='gray', linewidth=0.5)
axes[0].plot(steps, ema(grad_norms), color='#424242', linewidth=1.5, label='Global EMA')
axes[0].set_ylabel('Gradient Norm')
axes[0].set_title('Global Gradient Norm')
axes[0].legend()

# Per-block grad norms
for i in range(4):
    axes[1].plot(steps, ema(block_gnorms[i]), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
axes[1].set_ylabel('Block Gradient Norm')
axes[1].set_xlabel('Step')
axes[1].set_title('Per-Block Gradient Norms (EMA)')
axes[1].legend(ncol=4, loc='upper right')

fig.tight_layout()
fig.savefig('fig_grad_norms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_grad_norms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: L0 Inner-Loop Gradient Norms
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
for i in range(4):
    ax.plot(steps, ema(l0_gnorms[i], alpha=0.03), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
ax.set_ylabel('L0 Inner-Loop Gradient Norm')
ax.set_xlabel('Step')
ax.set_title('Per-Block Inner-Loop (Level 0) Gradient Norms')
ax.legend(ncol=4)
fig.tight_layout()
fig.savefig('fig_l0_gnorms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_l0_gnorms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 5: Block Gradient Norm CV (depth specialization)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(steps, gnorm_cvs, alpha=0.15, color='#9C27B0', linewidth=0.5)
ax.plot(steps, ema(gnorm_cvs), color='#6A1B9A', linewidth=1.5)
ax.set_ylabel('CV (σ/μ)')
ax.set_xlabel('Step')
ax.set_title('Block Gradient Norm Coefficient of Variation (Depth Specialization)')
ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='CV = 0.3 threshold')
ax.legend()
fig.tight_layout()
fig.savefig('fig_gnorm_cv.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_gnorm_cv.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 6: Loss Spikes Visualization
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, losses, alpha=0.3, color='#2196F3', linewidth=0.5)
ax.plot(steps, ema(losses), color='#1565C0', linewidth=1.5)

# Mark spikes
spike_steps, spike_losses = [], []
for i in range(1, len(losses)):
    if losses[i] > losses[i-1] * 2:
        spike_steps.append(steps[i])
        spike_losses.append(losses[i])
ax.scatter(spike_steps, spike_losses, color='red', s=50, zorder=5, label=f'{len(spike_steps)} spikes (>2× prev)')
ax.set_ylabel('Loss')
ax.set_xlabel('Step')
ax.set_title('Loss Instabilities (>2× Spikes Highlighted)')
ax.legend()
ax.set_xlim(10000, 20000)
ax.set_ylim(0, 12)
fig.tight_layout()
fig.savefig('fig_loss_spikes.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_spikes.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 7: NIAH Results Heatmap + Bar Chart
# ══════════════════════════════════════════════════════════════════════
niah_data = {}
for step, path in NIAH_FILES.items():
    with open(path) as f:
        niah_data[step] = json.load(f)

ckpt_steps = sorted(niah_data.keys())
distances = [1024, 2048, 4096, 8192]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Heatmap
pass_matrix = np.array([
    [niah_data[s]["distances"][str(d)]["pass_rate"] for d in distances]
    for s in ckpt_steps
])
im = ax1.imshow(pass_matrix.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax1.set_xticks(range(len(ckpt_steps)))
ax1.set_xticklabels([f'{s//1000}K' for s in ckpt_steps])
ax1.set_yticks(range(len(distances)))
ax1.set_yticklabels([str(d) for d in distances])
ax1.set_xlabel('Checkpoint')
ax1.set_ylabel('Needle Distance (tokens)')
ax1.set_title('NIAH Pass Rate')
for i in range(len(ckpt_steps)):
    for j in range(len(distances)):
        ax1.text(i, j, f'{pass_matrix[i,j]:.0%}', ha='center', va='center', fontsize=10,
                color='white' if pass_matrix[i,j] < 0.35 else 'black')
fig.colorbar(im, ax=ax1, shrink=0.8)

# Bar chart: mean lift by distance at step 20K
final = niah_data[20000]
lifts = [final["distances"][str(d)]["mean_lift"] for d in distances]
bars = ax2.bar([str(d) for d in distances], lifts, color=['#66BB6A','#42A5F5','#AB47BC','#FFA726'])
ax2.set_xlabel('Needle Distance (tokens)')
ax2.set_ylabel('Mean Log-Prob Lift')
ax2.set_title('NIAH Mean Lift at Step 20K')
ax2.axhline(y=0, color='gray', linewidth=0.8)
for bar, v in zip(bars, lifts):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:+.3f}', ha='center', fontsize=9)

fig.tight_layout()
fig.savefig('fig_niah.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_niah.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 8: Training Efficiency — tok/s over time
# ══════════════════════════════════════════════════════════════════════
elapsed = []
with open(METRICS) as f:
    for line in f:
        d = json.loads(line)
        if d.get("event") == "step":
            elapsed.append(d["elapsed"])
elapsed = np.array(elapsed)

# Compute instantaneous tok/s from elapsed deltas
tok_per_step = 512  # seq_len
dt = np.diff(elapsed)
# steps are logged every 8, so 8 steps between logs
tps = (8 * tok_per_step) / dt

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(steps[1:], tps, alpha=0.2, color='#4CAF50', linewidth=0.5)
ax.plot(steps[1:], ema(tps, alpha=0.02), color='#2E7D32', linewidth=1.5)
ax.set_ylabel('Tokens/sec')
ax.set_xlabel('Step')
ax.set_title('Training Throughput')
ax.set_ylim(0, max(tps)*1.2)
fig.tight_layout()
fig.savefig('fig_throughput.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_throughput.pdf")


print("\nAll plots generated.")
