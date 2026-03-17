#!/usr/bin/env python3
"""Generate PDF plots for k4_zero_retention_d512 (spec 29) — FAILED RUN.
k=4 from scratch with alpha_floor=0.0. Diverged at step ~3120."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_run(path):
    steps, tapes = [], []
    with open(os.path.join(path, 'metrics.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            ev = d.get('event', '')
            if ev == 'step': steps.append(d)
            elif ev == 'tape_summary': tapes.append(d)
    return steps, tapes

def smooth(values, window=20):
    if len(values) < window: return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def gm(d, key='mean'):
    if isinstance(d, dict): return float(d.get(key, 0))
    return float(d) if d is not None else 0.0

# ============================================================
# Plot 1: Loss trajectory with divergence marker
# ============================================================
def plot_loss_ppl(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = [s['step'] for s in steps]; loss = [s['loss'] for s in steps]
    ppl = [min(s['ppl'], 5000) for s in steps]

    ax1.plot(x, loss, alpha=0.3, color='steelblue', linewidth=0.5)
    xs = x[len(x)-len(smooth(loss)):]; ax1.plot(xs, smooth(loss), color='steelblue', linewidth=1.5, label='Loss (smoothed)')
    ax1.axvline(x=3120, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Divergence onset (~3120)')
    ax1.axhline(y=11.0, color='gray', linestyle=':', alpha=0.5, label='Random init (11.0)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss — k=4 Zero Floor (DIVERGED)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(x, ppl, alpha=0.3, color='darkorange', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(ppl)):]; ax2.plot(xs2, smooth(ppl), color='darkorange', linewidth=1.5, label='PPL (smoothed)')
    ax2.axvline(x=3120, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Divergence onset')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Perplexity'); ax2.set_title('Perplexity (clipped at 5000)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'loss_ppl.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  loss_ppl.pdf')

# ============================================================
# Plot 2: Gradient norm explosion
# ============================================================
def plot_grad_explosion(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = [s['step'] for s in steps]; gnorm = [s['grad_norm'] for s in steps]

    # Linear scale (clipped)
    gnorm_clipped = [min(g, 100) for g in gnorm]
    ax1.plot(x, gnorm_clipped, alpha=0.4, color='crimson', linewidth=0.5)
    xs = x[len(x)-len(smooth(gnorm_clipped)):]; ax1.plot(xs, smooth(gnorm_clipped), color='crimson', linewidth=1.5, label='Grad norm (clipped 100)')
    ax1.axvline(x=3120, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Divergence onset')
    spike_x = [s['step'] for s in steps if s['grad_norm'] > 10 and s['step'] < 3120]
    spike_y = [min(s['grad_norm'], 100) for s in steps if s['grad_norm'] > 10 and s['step'] < 3120]
    if spike_x: ax1.scatter(spike_x, spike_y, color='red', s=20, zorder=5, label=f'Pre-diverge spikes ({len(spike_x)})')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Gradient Norm'); ax1.set_title('Gradient Norms (Linear, Clipped)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Log scale (full range)
    gnorm_log = [max(g, 1e-3) for g in gnorm]
    ax2.semilogy(x, gnorm_log, alpha=0.4, color='crimson', linewidth=0.5)
    ax2.axvline(x=3120, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Divergence onset')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='max_grad_norm=1.0')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Gradient Norm (log)'); ax2.set_title('Gradient Norms (Log Scale)')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'grad_explosion.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  grad_explosion.pdf')

# ============================================================
# Plot 3: Alpha evolution across 4 tape summaries
# ============================================================
def plot_alpha_evolution(tapes):
    fig, ax = plt.subplots(figsize=(10, 5))
    tape_steps = [t['step'] for t in tapes]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['L0 (every token)', 'L1 (every 8)', 'L2 (every 64)', 'L3 (every 512)']

    for li in range(4):
        vals = [gm(t['levels'][li]['alpha'], 'mean') for t in tapes]
        ax.plot(tape_steps, vals, 'o-', color=colors[li], linewidth=2, markersize=8, label=labels[li])

    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Old floor (0.80)')
    ax.axvline(x=3120, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Divergence onset')
    ax.set_xlabel('Step'); ax.set_ylabel('Alpha (mean)'); ax.set_title('Retention Gate (alpha) per Level — Zero Floor')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 1.02)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'alpha_evolution.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  alpha_evolution.pdf')

# ============================================================
# Plot 4: Theta evolution across 4 tape summaries
# ============================================================
def plot_theta_evolution(tapes):
    fig, ax = plt.subplots(figsize=(10, 5))
    tape_steps = [t['step'] for t in tapes]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['L0 (every token)', 'L1 (every 8)', 'L2 (every 64)', 'L3 (every 512)']

    for li in range(4):
        vals = [gm(t['levels'][li]['theta'], 'mean') for t in tapes]
        ax.plot(tape_steps, vals, 'o-', color=colors[li], linewidth=2, markersize=8, label=labels[li])

    ax.axvline(x=3120, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Divergence onset')
    ax.set_xlabel('Step'); ax.set_ylabel('Theta (mean)'); ax.set_title('Write Strength (theta) per Level — Zero Floor')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'theta_evolution.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  theta_evolution.pdf')

# ============================================================
# Plot 5: Eta evolution
# ============================================================
def plot_eta_evolution(tapes):
    fig, ax = plt.subplots(figsize=(10, 5))
    tape_steps = [t['step'] for t in tapes]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['L0', 'L1', 'L2', 'L3']

    for li in range(4):
        vals = [gm(t['levels'][li]['eta'], 'mean') for t in tapes]
        ax.plot(tape_steps, vals, 'o-', color=colors[li], linewidth=2, markersize=8, label=labels[li])

    ax.axvline(x=3120, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Divergence onset')
    ax.set_xlabel('Step'); ax.set_ylabel('Eta (mean)'); ax.set_title('Momentum Gate (eta) per Level')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0.7, 1.02)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'eta_evolution.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  eta_evolution.pdf')

# ============================================================
# Plot 6: Pre- vs post-divergence gate comparison
# ============================================================
def plot_gate_comparison(tapes):
    if len(tapes) < 4: return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gate_names = ['alpha', 'theta', 'eta']
    gate_titles = ['Retention (alpha)', 'Write Strength (theta)', 'Momentum (eta)']
    levels = ['L0', 'L1', 'L2', 'L3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Compare step 3000 (pre-divergence, healthy) vs step 4000 (post-divergence)
    pre = tapes[2]   # step 3000
    post = tapes[3]  # step 4000

    for gi, (gate, title) in enumerate(zip(gate_names, gate_titles)):
        ax = axes[gi]
        x = np.arange(4)
        pre_vals = [gm(pre['levels'][li][gate], 'mean') for li in range(4)]
        post_vals = [gm(post['levels'][li][gate], 'mean') for li in range(4)]

        w = 0.35
        bars1 = ax.bar(x - w/2, pre_vals, w, label='Step 3000 (healthy)', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + w/2, post_vals, w, label='Step 4000 (diverged)', color='crimson', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Gate Values: Pre- vs Post-Divergence', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'gate_comparison.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  gate_comparison.pdf')

# ============================================================
# Plot 7: Throughput
# ============================================================
def plot_throughput(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [s['step'] for s in steps]
    toks = [s.get('tok_per_sec', 0) for s in steps]
    # tok_per_sec may not be in metrics, use elapsed to compute
    if all(t == 0 for t in toks):
        # Estimate from elapsed time
        for i, s in enumerate(steps):
            if i > 0 and s.get('elapsed', 0) > steps[i-1].get('elapsed', 0):
                dt = s['elapsed'] - steps[i-1]['elapsed']
                ds = s['step'] - steps[i-1]['step']
                toks[i] = (ds * 512) / dt if dt > 0 else 0
    toks_valid = [(xi, ti) for xi, ti in zip(x, toks) if ti > 0]
    if toks_valid:
        xv, tv = zip(*toks_valid)
        ax.plot(xv, tv, alpha=0.3, color='teal', linewidth=0.5)
        if len(tv) > 20:
            xs = xv[len(xv)-len(smooth(list(tv))):]; ax.plot(xs, smooth(list(tv)), color='teal', linewidth=1.5, label='tok/s (smoothed)')
        ax.axvline(x=3120, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Divergence onset')
        ax.set_xlabel('Step'); ax.set_ylabel('Tokens/sec'); ax.set_title('Throughput')
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'throughput.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  throughput.pdf')

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Loading metrics...')
    steps, tapes = load_run(RUN_DIR)
    print(f'  {len(steps)} step entries, {len(tapes)} tape summaries')
    print(f'  Steps {steps[0]["step"]}–{steps[-1]["step"]}')
    print('Generating plots...')
    plot_loss_ppl(steps)
    plot_grad_explosion(steps)
    plot_alpha_evolution(tapes)
    plot_theta_evolution(tapes)
    plot_eta_evolution(tapes)
    plot_gate_comparison(tapes)
    plot_throughput(steps)
    print('Done.')
