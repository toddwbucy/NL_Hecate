#!/usr/bin/env python3
"""Generate all plots for the baseline_pushup_p2_k2 experiment report."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load p2_k2 data ───────────────────────────────────────────────────
METRICS = "../metrics.jsonl"
P1_METRICS = "../../baseline_pushup_p1_k1/metrics.jsonl"
NIAH_FILES = {
    5000:  "../niah_step5000.json",
    10000: "../niah_step10000.json",
    15000: "../niah_step15000.json",
    20000: "../niah_step20000.json",
}
P1_NIAH_FILES = {
    5000:  "../../baseline_pushup_p1_k1/niah_step5000.json",
    10000: "../../baseline_pushup_p1_k1/niah_step10000.json",
    15000: "../../baseline_pushup_p1_k1/niah_step15000.json",
    20000: "../../baseline_pushup_p1_k1/niah_step20000.json",
}

def load_metrics(path):
    steps, losses, ppls, grad_norms, lrs = [], [], [], [], []
    block_gnorms = [[], [], [], []]
    l0_gnorms = [[], [], [], []]
    gnorm_cvs = []
    level_fires = [[], []]
    active_l1 = []
    elapsed_list = []

    with open(path) as f:
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
            elapsed_list.append(d["elapsed"])
            for i in range(4):
                block_gnorms[i].append(d["block_grad_norms"][i])
                l0_gnorms[i].append(d["l0_block_grad_norms"][i])
            lf = d.get("level_fires", [0, 0])
            level_fires[0].append(lf[0] if len(lf) > 0 else 0)
            level_fires[1].append(lf[1] if len(lf) > 1 else 0)
            al = d.get("active_levels", [True])
            active_l1.append(al[1] if len(al) > 1 else False)

    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'ppls': np.array(ppls),
        'grad_norms': np.array(grad_norms),
        'lrs': np.array(lrs),
        'gnorm_cvs': np.array(gnorm_cvs),
        'block_gnorms': [np.array(b) for b in block_gnorms],
        'l0_gnorms': [np.array(b) for b in l0_gnorms],
        'level_fires': [np.array(lf) for lf in level_fires],
        'active_l1': np.array(active_l1),
        'elapsed': np.array(elapsed_list),
    }

p2 = load_metrics(METRICS)
p1 = load_metrics(P1_METRICS)

# ── Smoothing helper ──────────────────────────────────────────────────
def ema(data, alpha=0.05):
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
# Figure 1: Loss Curve — k=2 with k=1 overlay
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

# k=2
ax1.plot(p2['steps'], p2['losses'], alpha=0.10, color='#FF5722', linewidth=0.5)
ax1.plot(p2['steps'], ema(p2['losses']), color='#D84315', linewidth=1.8, label='k=2 (this run)')
# k=1 reference
ax1.plot(p1['steps'], ema(p1['losses']), color='#1565C0', linewidth=1.2, alpha=0.6, linestyle='--', label='k=1 (baseline)')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss — k=2 Push-Up vs k=1 Baseline')
ax1.legend(loc='upper right')
ax1.set_ylim(0, 12)

# Annotate the regression
ax1.annotate('Immediate regression\n2.79 → ~3.5',
             xy=(500, 3.5), xytext=(3000, 7),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', ha='center')

# LR overlay
ax2.plot(p2['steps'], p2['lrs'], color='#FF9800', linewidth=1.2)
ax2.set_ylabel('Learning Rate')
ax2.set_xlabel('Step')
ax2.set_title('Cosine LR Schedule')
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-4,-4))

fig.tight_layout()
fig.savefig('fig_loss_curve.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_curve.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Loss Comparison — side by side smoothed
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(p1['steps'], ema(p1['losses'], alpha=0.02), color='#1565C0', linewidth=2, label='k=1 baseline')
ax.plot(p2['steps'], ema(p2['losses'], alpha=0.02), color='#D84315', linewidth=2, label='k=2 push-up')
ax.axhline(y=2.84, color='#1565C0', linestyle=':', alpha=0.5, label='k=1 final (2.84)')
ax.axhline(y=3.36, color='#D84315', linestyle=':', alpha=0.5, label='k=2 final (3.36)')
ax.fill_between(p2['steps'], ema(p1['losses'], alpha=0.02), ema(p2['losses'], alpha=0.02),
                where=ema(p2['losses'], alpha=0.02) > ema(p1['losses'], alpha=0.02),
                alpha=0.15, color='red', label='k=2 regression zone')
ax.set_ylabel('Loss (EMA α=0.02)')
ax.set_xlabel('Step')
ax.set_title('Loss Comparison: k=1 vs k=2 (Heavy Smoothing)')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(2, 8)
fig.tight_layout()
fig.savefig('fig_loss_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Perplexity (log scale)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(p2['steps'], p2['ppls'], alpha=0.15, color='#FF5722', linewidth=0.5)
ax.semilogy(p2['steps'], ema(p2['ppls']), color='#D84315', linewidth=1.8, label='k=2 EMA')
ax.semilogy(p1['steps'], ema(p1['ppls']), color='#1565C0', linewidth=1.2, alpha=0.5, linestyle='--', label='k=1 EMA')
ax.set_ylabel('Perplexity (log scale)')
ax.set_xlabel('Step')
ax.set_title('Training Perplexity — k=2 vs k=1')
ax.legend()
fig.tight_layout()
fig.savefig('fig_perplexity.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_perplexity.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: Gradient Norms — global + per-block
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].plot(p2['steps'], p2['grad_norms'], alpha=0.15, color='gray', linewidth=0.5)
axes[0].plot(p2['steps'], ema(p2['grad_norms']), color='#424242', linewidth=1.5, label='k=2 Global EMA')
axes[0].plot(p1['steps'], ema(p1['grad_norms']), color='#90CAF9', linewidth=1, linestyle='--', label='k=1 Global EMA')
axes[0].set_ylabel('Gradient Norm')
axes[0].set_title('Global Gradient Norm')
axes[0].legend()

for i in range(4):
    axes[1].plot(p2['steps'], ema(p2['block_gnorms'][i]), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
axes[1].set_ylabel('Block Gradient Norm')
axes[1].set_xlabel('Step')
axes[1].set_title('Per-Block Gradient Norms (EMA)')
axes[1].legend(ncol=4, loc='upper right')

fig.tight_layout()
fig.savefig('fig_grad_norms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_grad_norms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 5: L0 Inner-Loop Gradient Norms — k=2 vs k=1
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

ax1.set_title('k=1 Baseline — L0 Grad Norms')
for i in range(4):
    ax1.plot(p1['steps'], ema(p1['l0_gnorms'][i], alpha=0.03), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
ax1.set_ylabel('L0 Inner-Loop Gradient Norm')
ax1.set_xlabel('Step')
ax1.legend(ncol=2, fontsize=8)

ax2.set_title('k=2 Push-Up — L0 Grad Norms')
for i in range(4):
    ax2.plot(p2['steps'], ema(p2['l0_gnorms'][i], alpha=0.03), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
ax2.set_xlabel('Step')
ax2.legend(ncol=2, fontsize=8)

fig.tight_layout()
fig.savefig('fig_l0_gnorms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_l0_gnorms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 6: Block Gradient Norm CV comparison
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(p2['steps'], ema(p2['gnorm_cvs']), color='#D84315', linewidth=1.5, label='k=2')
ax.plot(p1['steps'], ema(p1['gnorm_cvs']), color='#1565C0', linewidth=1.2, alpha=0.6, linestyle='--', label='k=1')
ax.set_ylabel('CV (σ/μ)')
ax.set_xlabel('Step')
ax.set_title('Block Gradient Norm CV — Depth Specialization')
ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='CV = 0.3 threshold')
ax.legend()
fig.tight_layout()
fig.savefig('fig_gnorm_cv.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_gnorm_cv.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 7: Level 1 Fire Pattern
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# L1 cumulative fires
ax1.plot(p2['steps'], p2['level_fires'][1], color='#9C27B0', linewidth=1.5)
ax1.set_ylabel('Cumulative L1 Fires')
ax1.set_title('Level 1 Cumulative Fire Count')

# L1 active (boolean)
ax2.fill_between(p2['steps'], p2['active_l1'].astype(float), alpha=0.4, color='#9C27B0', step='mid')
ax2.set_ylabel('L1 Active')
ax2.set_xlabel('Step')
ax2.set_title('Level 1 Active Status (per logged step)')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Inactive', 'Active'])

fig.tight_layout()
fig.savefig('fig_level1_fires.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_level1_fires.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 8: Loss Spikes Visualization
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(p2['steps'], p2['losses'], alpha=0.3, color='#FF5722', linewidth=0.5)
ax.plot(p2['steps'], ema(p2['losses']), color='#D84315', linewidth=1.5)

spike_steps, spike_losses = [], []
for i in range(1, len(p2['losses'])):
    if p2['losses'][i] > p2['losses'][i-1] * 2:
        spike_steps.append(p2['steps'][i])
        spike_losses.append(p2['losses'][i])
ax.scatter(spike_steps, spike_losses, color='red', s=50, zorder=5, label=f'{len(spike_steps)} spikes (>2× prev)')
ax.set_ylabel('Loss')
ax.set_xlabel('Step')
ax.set_title('Loss Instabilities — k=2 Push-Up')
ax.legend()
ax.set_ylim(0, max(spike_losses + [6]) * 1.1 if spike_losses else 6)
fig.tight_layout()
fig.savefig('fig_loss_spikes.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_spikes.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 9: NIAH — k=2 heatmap + comparison with k=1
# ══════════════════════════════════════════════════════════════════════
def load_niah(files):
    data = {}
    for step, path in files.items():
        with open(path) as f:
            data[step] = json.load(f)
    return data

niah_k2 = load_niah(NIAH_FILES)
niah_k1 = load_niah(P1_NIAH_FILES)

ckpt_steps = sorted(niah_k2.keys())
distances = [1024, 2048, 4096, 8192]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# k=1 heatmap
mat_k1 = np.array([
    [niah_k1[s]["distances"][str(d)]["pass_rate"] for d in distances]
    for s in ckpt_steps
])
im1 = axes[0].imshow(mat_k1.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(ckpt_steps)))
axes[0].set_xticklabels([f'{s//1000}K' for s in ckpt_steps])
axes[0].set_yticks(range(len(distances)))
axes[0].set_yticklabels([str(d) for d in distances])
axes[0].set_xlabel('Checkpoint')
axes[0].set_ylabel('Distance (tokens)')
axes[0].set_title('k=1 NIAH Pass Rate')
for i in range(len(ckpt_steps)):
    for j in range(len(distances)):
        axes[0].text(i, j, f'{mat_k1[i,j]:.0%}', ha='center', va='center', fontsize=9,
                    color='white' if mat_k1[i,j] < 0.35 else 'black')

# k=2 heatmap
mat_k2 = np.array([
    [niah_k2[s]["distances"][str(d)]["pass_rate"] for d in distances]
    for s in ckpt_steps
])
im2 = axes[1].imshow(mat_k2.T, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[1].set_xticks(range(len(ckpt_steps)))
axes[1].set_xticklabels([f'{s//1000}K' for s in ckpt_steps])
axes[1].set_yticks(range(len(distances)))
axes[1].set_yticklabels([str(d) for d in distances])
axes[1].set_xlabel('Checkpoint')
axes[1].set_title('k=2 NIAH Pass Rate')
for i in range(len(ckpt_steps)):
    for j in range(len(distances)):
        axes[1].text(i, j, f'{mat_k2[i,j]:.0%}', ha='center', va='center', fontsize=9,
                    color='white' if mat_k2[i,j] < 0.35 else 'black')

# Delta heatmap (k=2 - k=1)
delta = mat_k2 - mat_k1
im3 = axes[2].imshow(delta.T, cmap='RdBu', vmin=-0.4, vmax=0.4, aspect='auto')
axes[2].set_xticks(range(len(ckpt_steps)))
axes[2].set_xticklabels([f'{s//1000}K' for s in ckpt_steps])
axes[2].set_yticks(range(len(distances)))
axes[2].set_yticklabels([str(d) for d in distances])
axes[2].set_xlabel('Checkpoint')
axes[2].set_title('Δ Pass Rate (k=2 − k=1)')
for i in range(len(ckpt_steps)):
    for j in range(len(distances)):
        val = delta[i,j]
        axes[2].text(i, j, f'{val:+.0%}', ha='center', va='center', fontsize=9,
                    color='white' if abs(val) > 0.25 else 'black')
fig.colorbar(im3, ax=axes[2], shrink=0.8, label='Δ')

fig.tight_layout()
fig.savefig('fig_niah.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_niah.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 10: NIAH Mean Lift comparison at 20K
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(distances))
width = 0.35
lifts_k1 = [niah_k1[20000]["distances"][str(d)]["mean_lift"] for d in distances]
lifts_k2 = [niah_k2[20000]["distances"][str(d)]["mean_lift"] for d in distances]
bars1 = ax.bar(x - width/2, lifts_k1, width, label='k=1', color='#42A5F5')
bars2 = ax.bar(x + width/2, lifts_k2, width, label='k=2', color='#FF7043')
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in distances])
ax.set_xlabel('Needle Distance (tokens)')
ax.set_ylabel('Mean Log-Prob Lift')
ax.set_title('NIAH Mean Lift at Step 20K — k=1 vs k=2')
ax.axhline(y=0, color='gray', linewidth=0.8)
ax.legend()
for bar, v in zip(bars1, lifts_k1):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:+.3f}', ha='center', fontsize=8, color='#1565C0')
for bar, v in zip(bars2, lifts_k2):
    ax.text(bar.get_x() + bar.get_width()/2, v - 0.05 if v < 0 else v + 0.02,
            f'{v:+.3f}', ha='center', fontsize=8, color='#D84315')
fig.tight_layout()
fig.savefig('fig_niah_lift.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_niah_lift.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 11: Training Throughput comparison
# ══════════════════════════════════════════════════════════════════════
tok_per_step = 512
dt2 = np.diff(p2['elapsed'])
tps2 = (8 * tok_per_step) / dt2
dt1 = np.diff(p1['elapsed'])
tps1 = (8 * tok_per_step) / dt1

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(p2['steps'][1:], ema(tps2, alpha=0.02), color='#D84315', linewidth=1.5, label='k=2 (612 tok/s avg)')
ax.plot(p1['steps'][1:], ema(tps1, alpha=0.02), color='#1565C0', linewidth=1.2, alpha=0.6, linestyle='--', label='k=1 (675 tok/s avg)')
ax.set_ylabel('Tokens/sec')
ax.set_xlabel('Step')
ax.set_title('Training Throughput — k=2 vs k=1')
ax.legend()
ax.set_ylim(0, 1000)
fig.tight_layout()
fig.savefig('fig_throughput.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_throughput.pdf")


print("\nAll plots generated.")
