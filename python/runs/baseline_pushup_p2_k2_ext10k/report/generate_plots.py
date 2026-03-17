#!/usr/bin/env python3
"""Generate all plots for the baseline_pushup_p2_k2_ext10k experiment report."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────
METRICS = "../metrics.jsonl"
P1_METRICS = "../../baseline_pushup_p1_k1/metrics.jsonl"
P2_METRICS = "../../baseline_pushup_p2_k2/metrics.jsonl"
FRESH_METRICS = "../../fresh_k2_from_scratch/metrics.jsonl"

def load_steps(path):
    steps, losses, ppls, grad_norms, lrs, cvs, elapsed = [], [], [], [], [], [], []
    block_gnorms = [[], [], [], []]
    l0_gnorms = [[], [], [], []]
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
            cvs.append(d["block_gnorm_cv"])
            elapsed.append(d["elapsed"])
            for i in range(4):
                block_gnorms[i].append(d["block_grad_norms"][i])
                l0_gnorms[i].append(d["l0_block_grad_norms"][i])
    return {k: np.array(v) for k, v in {
        'steps': steps, 'losses': losses, 'ppls': ppls,
        'grad_norms': grad_norms, 'lrs': lrs, 'cvs': cvs, 'elapsed': elapsed
    }.items()} | {
        'block_gnorms': [np.array(b) for b in block_gnorms],
        'l0_gnorms': [np.array(b) for b in l0_gnorms],
    }

def load_tape(path):
    tapes = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("event") == "tape_summary":
                tapes.append(d)
    return tapes

def gm(lvl, key):
    v = lvl.get(key, {})
    return v.get('mean', 0.0) if isinstance(v, dict) else v

ext = load_steps(METRICS)
p1 = load_steps(P1_METRICS)
p2 = load_steps(P2_METRICS)
fresh = load_steps(FRESH_METRICS)
ext_tape = load_tape(METRICS)
fresh_tape = load_tape(FRESH_METRICS)

def ema(data, alpha=0.05):
    out = np.zeros_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#fafafa',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12, 'legend.fontsize': 9,
    'figure.dpi': 150,
})
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
BLOCK_LABELS = ['Block 0', 'Block 1', 'Block 2', 'Block 3']


# ══════════════════════════════════════════════════════════════════════
# Figure 1: Loss — all four runs on one plot
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(p1['steps'], ema(p1['losses'], 0.02), color='#1565C0', linewidth=1.5, label='k=1 baseline')
ax.plot(p2['steps'], ema(p2['losses'], 0.02), color='#FF7043', linewidth=1.5, label='k=2 push-up (20K)')
ax.plot(ext['steps'], ema(ext['losses'], 0.02), color='#D84315', linewidth=2, label='k=2 push-up ext (+10K)')
ax.plot(fresh['steps'], ema(fresh['losses'], 0.02), color='#2E7D32', linewidth=1.5, linestyle='--', label='k=2 fresh (in progress)')
ax.set_ylabel('Loss (EMA α=0.02)')
ax.set_xlabel('Step')
ax.set_title('Loss Comparison — All Experiments')
ax.legend(loc='upper right')
ax.set_ylim(2, 8)
ax.set_xlim(0, 32000)
fig.tight_layout()
fig.savefig('fig_loss_all.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_all.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 2: Ext10K loss with spikes
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(ext['steps'], ext['losses'], alpha=0.3, color='#FF5722', linewidth=0.5)
ax1.plot(ext['steps'], ema(ext['losses']), color='#D84315', linewidth=1.8, label='EMA')
spike_s, spike_l = [], []
for i in range(1, len(ext['losses'])):
    if ext['losses'][i] > ext['losses'][i-1] * 2:
        spike_s.append(ext['steps'][i])
        spike_l.append(ext['losses'][i])
if spike_s:
    ax1.scatter(spike_s, spike_l, color='red', s=60, zorder=5, label=f'{len(spike_s)} spike(s) (>2×)')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Extension Run: Loss (Steps 20K–30K)')
ax1.legend()
ax1.set_ylim(1.5, 7)

ax2.plot(ext['steps'], ext['lrs'], color='#FF9800', linewidth=1.2)
ax2.set_ylabel('Learning Rate')
ax2.set_xlabel('Step')
ax2.set_title('Cosine LR Schedule (Extension)')
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-5, -5))
fig.tight_layout()
fig.savefig('fig_loss_ext.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_loss_ext.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 3: Gradient norms + CV
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for i in range(4):
    axes[0].plot(ext['steps'], ema(ext['block_gnorms'][i]), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
axes[0].set_ylabel('Block Gradient Norm')
axes[0].set_title('Per-Block Gradient Norms (EMA)')
axes[0].legend(ncol=4, loc='upper right')

axes[1].plot(ext['steps'], ema(ext['cvs']), color='#D84315', linewidth=1.5, label='ext10k')
axes[1].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='CV=0.3')
axes[1].set_ylabel('CV (σ/μ)')
axes[1].set_xlabel('Step')
axes[1].set_title('Block Gradient Norm CV — Depth Specialization')
axes[1].legend()
fig.tight_layout()
fig.savefig('fig_grad_norms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_grad_norms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 4: L0 inner-loop grad norms
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
for i in range(4):
    ax.plot(ext['steps'], ema(ext['l0_gnorms'][i], 0.03), color=COLORS[i], linewidth=1.3, label=BLOCK_LABELS[i])
ax.set_ylabel('L0 Inner-Loop Gradient Norm')
ax.set_xlabel('Step')
ax.set_title('Per-Block Inner-Loop (Level 0) Gradient Norms')
ax.legend(ncol=4)
fig.tight_layout()
fig.savefig('fig_l0_gnorms.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_l0_gnorms.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 5: THETA oscillation — ext10k vs fresh k=2 (Block 1)
# ══════════════════════════════════════════════════════════════════════
def extract_gate_series(tapes, block_idx, gate_key):
    """Extract L0 and L1 gate values for a given block across tape snapshots."""
    steps, l0_vals, l1_vals = [], [], []
    for t in tapes:
        step = t['step']
        for block in t['blocks']:
            if block['block_index'] != block_idx:
                continue
            levels = block['levels']
            if len(levels) < 2:
                continue
            l1v = gm(levels[1], gate_key)
            if l1v == 0.0:  # L1 not active
                continue
            steps.append(step)
            l0_vals.append(gm(levels[0], gate_key))
            l1_vals.append(l1v)
    return np.array(steps), np.array(l0_vals), np.array(l1_vals)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Block 1 theta — ext vs fresh
s_e, l0_e, l1_e = extract_gate_series(ext_tape, 1, 'theta')
s_f, l0_f, l1_f = extract_gate_series(fresh_tape, 1, 'theta')

axes[0, 0].plot(s_e, l0_e, 'o-', color='#D84315', markersize=4, label='L0 θ')
axes[0, 0].plot(s_e, l1_e, 's-', color='#FF9800', markersize=4, label='L1 θ')
axes[0, 0].fill_between(s_e, l0_e, l1_e, alpha=0.15, color='orange')
axes[0, 0].set_title('Push-Up Ext — Block 1 Theta')
axes[0, 0].set_ylabel('θ (inner LR)')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.1)

axes[0, 1].plot(s_f, l0_f, 'o-', color='#2E7D32', markersize=4, label='L0 θ')
axes[0, 1].plot(s_f, l1_f, 's-', color='#66BB6A', markersize=4, label='L1 θ')
axes[0, 1].fill_between(s_f, l0_f, l1_f, alpha=0.15, color='green')
axes[0, 1].set_title('Fresh k=2 — Block 1 Theta')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1.1)

# Block 3 theta — ext vs fresh
s_e3, l0_e3, l1_e3 = extract_gate_series(ext_tape, 3, 'theta')
s_f3, l0_f3, l1_f3 = extract_gate_series(fresh_tape, 3, 'theta')

axes[1, 0].plot(s_e3, l0_e3, 'o-', color='#D84315', markersize=4, label='L0 θ')
axes[1, 0].plot(s_e3, l1_e3, 's-', color='#FF9800', markersize=4, label='L1 θ')
axes[1, 0].fill_between(s_e3, l0_e3, l1_e3, alpha=0.15, color='orange')
axes[1, 0].set_title('Push-Up Ext — Block 3 Theta')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('θ (inner LR)')
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1.1)

axes[1, 1].plot(s_f3, l0_f3, 'o-', color='#2E7D32', markersize=4, label='L0 θ')
axes[1, 1].plot(s_f3, l1_f3, 's-', color='#66BB6A', markersize=4, label='L1 θ')
axes[1, 1].fill_between(s_f3, l0_f3, l1_f3, alpha=0.15, color='green')
axes[1, 1].set_title('Fresh k=2 — Block 3 Theta')
axes[1, 1].set_xlabel('Step')
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 1.1)

fig.suptitle('Theta (Inner LR) Oscillation: Push-Up Extension vs Fresh k=2', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('fig_theta_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_theta_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 6: ALPHA comparison — ext vs fresh (Block 1)
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

s_e, l0_e, l1_e = extract_gate_series(ext_tape, 1, 'alpha')
s_f, l0_f, l1_f = extract_gate_series(fresh_tape, 1, 'alpha')

ax1.plot(s_e, l0_e, 'o-', color='#D84315', markersize=4, label='L0 α')
ax1.plot(s_e, l1_e, 's-', color='#FF9800', markersize=4, label='L1 α')
ax1.fill_between(s_e, l0_e, l1_e, alpha=0.15, color='orange')
ax1.set_title('Push-Up Ext — Block 1 Alpha (Retention)')
ax1.set_ylabel('α (retention)')
ax1.set_xlabel('Step')
ax1.legend()
ax1.set_ylim(0.4, 1.0)

ax2.plot(s_f, l0_f, 'o-', color='#2E7D32', markersize=4, label='L0 α')
ax2.plot(s_f, l1_f, 's-', color='#66BB6A', markersize=4, label='L1 α')
ax2.fill_between(s_f, l0_f, l1_f, alpha=0.15, color='green')
ax2.set_title('Fresh k=2 — Block 1 Alpha (Retention)')
ax2.set_xlabel('Step')
ax2.legend()
ax2.set_ylim(0.4, 1.0)

fig.suptitle('Alpha (Retention): Push-Up Extension vs Fresh k=2', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('fig_alpha_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_alpha_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 7: DGD ratio over time — ext vs fresh
# ══════════════════════════════════════════════════════════════════════
def extract_dgd_ratio(tapes, block_idx):
    steps, ratios = [], []
    for t in tapes:
        step = t['step']
        for block in t['blocks']:
            if block['block_index'] != block_idx:
                continue
            levels = block['levels']
            if len(levels) < 2:
                continue
            l1d = levels[1].get('dgd_delta_norm', 0)
            if l1d == 0.0:
                continue
            l0d = levels[0].get('dgd_delta_norm', 0)
            steps.append(step)
            ratios.append(l0d / l1d)
    return np.array(steps), np.array(ratios)

fig, ax = plt.subplots(figsize=(10, 4.5))

for bi, ls, c, lbl in [(1, '-', '#D84315', 'Push-up ext'), (1, '--', '#2E7D32', 'Fresh k=2')]:
    tape = ext_tape if 'Push' in lbl else fresh_tape
    s, r = extract_dgd_ratio(tape, bi)
    ax.plot(s, r, f'o{ls}', color=c, markersize=4, linewidth=1.3, label=f'{lbl} Block 1')

for bi, ls, c, lbl in [(3, '-', '#FF9800', 'Push-up ext'), (3, '--', '#66BB6A', 'Fresh k=2')]:
    tape = ext_tape if 'Push' in lbl else fresh_tape
    s, r = extract_dgd_ratio(tape, bi)
    ax.plot(s, r, f's{ls}', color=c, markersize=4, linewidth=1.3, label=f'{lbl} Block 3')

ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Ratio=1 (no differentiation)')
ax.set_ylabel('L0/L1 DGD Delta Norm Ratio')
ax.set_xlabel('Step')
ax.set_title('DGD Update Ratio: Push-Up Extension vs Fresh k=2')
ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(0, 5)
fig.tight_layout()
fig.savefig('fig_dgd_ratio.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_dgd_ratio.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 8: ETA (output gate) comparison
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

s_e, l0_e, l1_e = extract_gate_series(ext_tape, 1, 'eta')
s_f, l0_f, l1_f = extract_gate_series(fresh_tape, 1, 'eta')

ax1.plot(s_e, l0_e, 'o-', color='#D84315', markersize=4, label='L0 η')
ax1.plot(s_e, l1_e, 's-', color='#FF9800', markersize=4, label='L1 η')
ax1.fill_between(s_e, l0_e, l1_e, alpha=0.15, color='orange')
ax1.set_title('Push-Up Ext — Block 1 Eta (Output Gate)')
ax1.set_ylabel('η (output gate)')
ax1.set_xlabel('Step')
ax1.legend()
ax1.set_ylim(0.3, 1.0)

ax2.plot(s_f, l0_f, 'o-', color='#2E7D32', markersize=4, label='L0 η')
ax2.plot(s_f, l1_f, 's-', color='#66BB6A', markersize=4, label='L1 η')
ax2.fill_between(s_f, l0_f, l1_f, alpha=0.15, color='green')
ax2.set_title('Fresh k=2 — Block 1 Eta (Output Gate)')
ax2.set_xlabel('Step')
ax2.legend()
ax2.set_ylim(0.3, 1.0)

fig.suptitle('Eta (Output Gate): Push-Up Extension vs Fresh k=2', fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('fig_eta_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_eta_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 9: Output grad norm ratio (proving identity)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3.5))

for tape, color, label in [(ext_tape, '#D84315', 'Push-up ext'), (fresh_tape, '#2E7D32', 'Fresh k=2')]:
    steps_r, ratios_r = [], []
    for t in tape:
        step = t['step']
        for block in t['blocks']:
            if block['block_index'] != 0:
                continue
            l0g = block['levels'][0].get('output_grad_norm', 0)
            l1g = block['levels'][1].get('output_grad_norm', 0)
            if l1g > 0:
                steps_r.append(step)
                ratios_r.append(l0g / l1g)
    ax.plot(steps_r, ratios_r, 'o-', color=color, markersize=4, label=f'{label} (Block 0)')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
ax.set_ylabel('L0/L1 Output Grad Norm Ratio')
ax.set_xlabel('Step')
ax.set_title('Output Gradient Norm Ratio — Both Experiments Show Ratio = 1.000')
ax.legend()
ax.set_ylim(0.9, 1.1)
fig.tight_layout()
fig.savefig('fig_outgrad_ratio.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_outgrad_ratio.pdf")


# ══════════════════════════════════════════════════════════════════════
# Figure 10: Throughput
# ══════════════════════════════════════════════════════════════════════
dt = np.diff(ext['elapsed'])
tps = (8 * 512) / dt

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(ext['steps'][1:], ema(tps, alpha=0.02), color='#D84315', linewidth=1.5, label='ext10k (609 tok/s)')
ax.set_ylabel('Tokens/sec')
ax.set_xlabel('Step')
ax.set_title('Training Throughput — Extension Run')
ax.legend()
ax.set_ylim(0, 1000)
fig.tight_layout()
fig.savefig('fig_throughput.pdf', bbox_inches='tight')
plt.close(fig)
print("✓ fig_throughput.pdf")


print("\nAll plots generated.")
