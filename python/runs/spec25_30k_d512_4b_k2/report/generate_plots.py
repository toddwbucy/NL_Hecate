#!/usr/bin/env python3
"""Generate PDF plots for spec25_30k_d512_4b_k2 experiment report."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_metrics():
    steps, tapes, heatmaps = [], [], []
    with open(os.path.join(RUN_DIR, 'metrics.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            ev = d.get('event', '')
            if ev == 'step':
                steps.append(d)
            elif ev == 'tape_summary':
                tapes.append(d)
            elif ev == 'level_heatmap':
                heatmaps.append(d)
    return steps, tapes, heatmaps

def smooth(values, window=50):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

def plot_loss_ppl(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [s['step'] for s in steps]
    loss = [s['loss'] for s in steps]
    ppl = [s['ppl'] for s in steps]

    ax1.plot(x, loss, alpha=0.15, color='steelblue', linewidth=0.5)
    xs = x[len(x)-len(smooth(loss)):]
    ax1.plot(xs, smooth(loss), color='steelblue', linewidth=1.5, label='Loss (smoothed)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ppl_clipped = [min(p, 500) for p in ppl]
    ax2.plot(x, ppl_clipped, alpha=0.15, color='darkorange', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(ppl_clipped)):]
    ax2.plot(xs2, smooth(ppl_clipped), color='darkorange', linewidth=1.5, label='PPL (smoothed)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity (clipped at 500)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'loss_ppl.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  loss_ppl.pdf')

def plot_grad_norms(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [s['step'] for s in steps]
    gnorm = [s['grad_norm'] for s in steps]
    cv = [s['block_gnorm_cv'] for s in steps]

    ax1.plot(x, gnorm, alpha=0.2, color='crimson', linewidth=0.5)
    xs = x[len(x)-len(smooth(gnorm)):]
    ax1.plot(xs, smooth(gnorm), color='crimson', linewidth=1.5, label='Grad norm (smoothed)')
    # Mark spikes
    spike_x = [s['step'] for s in steps if s['grad_norm'] > 10]
    spike_y = [s['grad_norm'] for s in steps if s['grad_norm'] > 10]
    if spike_x:
        ax1.scatter(spike_x, spike_y, color='red', s=30, zorder=5, label=f'Spikes >10 ({len(spike_x)})')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Global Gradient Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, cv, alpha=0.2, color='teal', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(cv)):]
    ax2.plot(xs2, smooth(cv), color='teal', linewidth=1.5, label='Block gnorm CV (smoothed)')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Specialization threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Depth Specialization (Block Gradient Norm CV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  grad_norms.pdf')

def plot_block_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]
    n_blocks = len(steps[0]['block_grad_norms'])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for b in range(n_blocks):
        vals = [s['block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]
        ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Per-Block Gradient Norms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'block_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  block_grad_norms.pdf')

def plot_l0_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]
    n_blocks = len(steps[0]['l0_block_grad_norms'])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for b in range(n_blocks):
        vals = [s['l0_block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]
        ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b} L0')
    ax.set_xlabel('Step')
    ax.set_ylabel('L0 Gradient Norm')
    ax.set_title('L0 Per-Block Gradient Norms (Inner-Loop Signal)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'l0_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  l0_grad_norms.pdf')

def gm(d, key='mean'):
    """Extract mean from gate diagnostic dict or scalar."""
    if isinstance(d, dict):
        return float(d.get(key, 0))
    return float(d) if d is not None else 0.0

def plot_gate_diagnostics(tapes):
    """Plot theta, alpha, eta evolution for L0 and L1 across all blocks."""
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    # --- Theta ---
    fig, axes = plt.subplots(1, n_blocks, figsize=(14, 4), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi]
        l0_theta = [gm(t['blocks'][bi]['levels'][0].get('theta', {})) for t in tapes]
        l1_theta = [gm(t['blocks'][bi]['levels'][1].get('theta', {})) for t in tapes]
        ax.plot(tape_steps, l0_theta, 'o-', color='steelblue', markersize=3, label='L0')
        ax.plot(tape_steps, l1_theta, 's-', color='darkorange', markersize=3, label='L1')
        ax.set_title(f'Block {bi}')
        ax.set_xlabel('Step')
        if bi == 0:
            ax.set_ylabel('Theta (inner-loop LR)')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Theta Evolution by Block', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'theta_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  theta_evolution.pdf')

    # --- Alpha ---
    fig, axes = plt.subplots(1, n_blocks, figsize=(14, 4), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi]
        l0_alpha = [gm(t['blocks'][bi]['levels'][0].get('alpha', {})) for t in tapes]
        l1_alpha = [gm(t['blocks'][bi]['levels'][1].get('alpha', {})) for t in tapes]
        ax.plot(tape_steps, l0_alpha, 'o-', color='steelblue', markersize=3, label='L0')
        ax.plot(tape_steps, l1_alpha, 's-', color='darkorange', markersize=3, label='L1')
        ax.set_title(f'Block {bi}')
        ax.set_xlabel('Step')
        if bi == 0:
            ax.set_ylabel('Alpha (retention)')
        ax.set_ylim(0.75, 1.02)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Alpha (Retention) Evolution by Block', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'alpha_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  alpha_evolution.pdf')

    # --- Eta ---
    fig, axes = plt.subplots(1, n_blocks, figsize=(14, 4), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi]
        l0_eta = [gm(t['blocks'][bi]['levels'][0].get('eta', {})) for t in tapes]
        l1_eta = [gm(t['blocks'][bi]['levels'][1].get('eta', {})) for t in tapes]
        ax.plot(tape_steps, l0_eta, 'o-', color='steelblue', markersize=3, label='L0')
        ax.plot(tape_steps, l1_eta, 's-', color='darkorange', markersize=3, label='L1')
        ax.set_title(f'Block {bi}')
        ax.set_xlabel('Step')
        if bi == 0:
            ax.set_ylabel('Eta (output gate)')
        ax.set_ylim(0.6, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Eta (Output Gate) Evolution by Block', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'eta_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  eta_evolution.pdf')

def plot_dgd_ratio(tapes):
    """DGD delta norm ratio L0/L1 across blocks."""
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    fig, axes = plt.subplots(1, n_blocks, figsize=(14, 4), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi]
        ratios = []
        for t in tapes:
            d0 = t['blocks'][bi]['levels'][0].get('dgd_delta_norm', 0)
            d1 = t['blocks'][bi]['levels'][1].get('dgd_delta_norm', 0)
            ratios.append(d0 / d1 if d1 > 0 else 0)
        ax.plot(tape_steps, ratios, 'o-', color='purple', markersize=3)
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Healthy threshold (2.0)')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Parity (1.0)')
        ax.set_title(f'Block {bi}')
        ax.set_xlabel('Step')
        if bi == 0:
            ax.set_ylabel('DGD Ratio (L0/L1)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('DGD Delta Norm Ratio (L0/L1) by Block', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dgd_ratio.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  dgd_ratio.pdf')

def plot_output_grad_norms(tapes):
    """Output gradient norms L0 vs L1 — in MAG these should be identical."""
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    fig, axes = plt.subplots(1, n_blocks, figsize=(14, 4), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi]
        l0_og = [t['blocks'][bi]['levels'][0].get('output_grad_norm', 0) for t in tapes]
        l1_og = [t['blocks'][bi]['levels'][1].get('output_grad_norm', 0) for t in tapes]
        ax.plot(tape_steps, l0_og, 'o-', color='steelblue', markersize=3, label='L0')
        ax.plot(tape_steps, l1_og, 's-', color='darkorange', markersize=3, label='L1')
        # Compute ratio
        ratios = [l0/l1 if l1 > 0 else 0 for l0, l1 in zip(l0_og, l1_og)]
        ax.set_title(f'Block {bi} (ratio={np.mean(ratios):.3f})')
        ax.set_xlabel('Step')
        if bi == 0:
            ax.set_ylabel('Output Grad Norm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Output Gradient Norms by Block (MAG identity check)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'output_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  output_grad_norms.pdf')

def plot_throughput(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Compute instantaneous tok/s
    x, tps = [], []
    for i in range(1, len(steps)):
        dt = steps[i]['elapsed'] - steps[i-1]['elapsed']
        if dt > 0:
            ds = steps[i]['step'] - steps[i-1]['step']
            x.append(steps[i]['step'])
            tps.append(ds * 512 / dt)

    ax.plot(x, tps, alpha=0.1, color='green', linewidth=0.5)
    xs = x[len(x)-len(smooth(tps, 100)):]
    ax.plot(xs, smooth(tps, 100), color='green', linewidth=1.5, label='tok/s (smoothed)')
    ax.axhline(y=np.mean(tps), color='darkgreen', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(tps):.0f} tok/s')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Training Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'throughput.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  throughput.pdf')

def plot_lr_schedule(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [s['step'] for s in steps]
    lr = [s['lr'] for s in steps]
    ax.plot(x, lr, color='navy', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Decay)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lr_schedule.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  lr_schedule.pdf')

def plot_level_fires(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [s['step'] for s in steps]
    l0_fires = [s['level_fires'][0] for s in steps]
    l1_fires = [s['level_fires'][1] for s in steps]
    ax.plot(x, l0_fires, color='steelblue', linewidth=1.0, label='L0 fires')
    ax.plot(x, l1_fires, color='darkorange', linewidth=1.0, label='L1 fires')
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Fires')
    ax.set_title('Level Fire Counts (L0: every token, L1: every 8th)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'level_fires.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  level_fires.pdf')

if __name__ == '__main__':
    print('Loading metrics...')
    steps, tapes, heatmaps = load_metrics()
    print(f'  {len(steps)} step events, {len(tapes)} tape summaries, {len(heatmaps)} heatmaps')
    print('Generating plots...')
    plot_loss_ppl(steps)
    plot_grad_norms(steps)
    plot_block_grad_norms(steps)
    plot_l0_grad_norms(steps)
    plot_gate_diagnostics(tapes)
    plot_dgd_ratio(tapes)
    plot_output_grad_norms(tapes)
    plot_throughput(steps)
    plot_lr_schedule(steps)
    plot_level_fires(steps)
    print('Done — all plots saved to', OUT_DIR)
