#!/usr/bin/env python3
"""Generate PDF plots for shakedown_8b_d1024_k2 experiment report."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_metrics():
    steps, tapes, heatmaps = [], [], []
    with open(os.path.join(RUN_DIR, 'metrics.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            ev = d.get('event', '')
            if ev == 'step': steps.append(d)
            elif ev == 'tape_summary': tapes.append(d)
            elif ev == 'level_heatmap': heatmaps.append(d)
    return steps, tapes, heatmaps

def smooth(values, window=30):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')

def gm(d, key='mean'):
    if isinstance(d, dict): return float(d.get(key, 0))
    return float(d) if d is not None else 0.0

def plot_loss_ppl(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = [s['step'] for s in steps]
    loss = [s['loss'] for s in steps]
    ppl = [min(s['ppl'], 500) for s in steps]

    ax1.plot(x, loss, alpha=0.15, color='steelblue', linewidth=0.5)
    xs = x[len(x)-len(smooth(loss)):]
    ax1.plot(xs, smooth(loss), color='steelblue', linewidth=1.5, label='Loss (smoothed)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, ppl, alpha=0.15, color='darkorange', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(ppl)):]
    ax2.plot(xs2, smooth(ppl), color='darkorange', linewidth=1.5, label='PPL (smoothed)')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Perplexity'); ax2.set_title('Perplexity (clipped at 500)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

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
    spike_x = [s['step'] for s in steps if s['grad_norm'] > 10]
    spike_y = [s['grad_norm'] for s in steps if s['grad_norm'] > 10]
    if spike_x:
        ax1.scatter(spike_x, spike_y, color='red', s=30, zorder=5, label=f'Spikes >10 ({len(spike_x)})')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Gradient Norm'); ax1.set_title('Global Gradient Norm')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, cv, alpha=0.2, color='teal', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(cv)):]
    ax2.plot(xs2, smooth(cv), color='teal', linewidth=1.5, label='Block gnorm CV (smoothed)')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='d=512 threshold (0.3)')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Depth Specialization (Block Gradient Norm CV)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  grad_norms.pdf')

def plot_block_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = [s['step'] for s in steps]
    n_blocks = len(steps[0]['block_grad_norms'])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))
    for b in range(n_blocks):
        vals = [s['block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]
        ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b}')
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm')
    ax.set_title('Per-Block Gradient Norms (8 blocks)')
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'block_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  block_grad_norms.pdf')

def plot_l0_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = [s['step'] for s in steps]
    n_blocks = len(steps[0]['l0_block_grad_norms'])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_blocks))
    for b in range(n_blocks):
        vals = [s['l0_block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]
        ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b} L0')
    ax.set_xlabel('Step'); ax.set_ylabel('L0 Gradient Norm')
    ax.set_title('L0 Per-Block Gradient Norms (Inner-Loop Signal)')
    ax.legend(ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'l0_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  l0_grad_norms.pdf')

def plot_gate_diagnostics(tapes):
    """Plot theta, alpha, eta for L0 and L1 across all 8 blocks."""
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    # --- Theta (2 rows: L0 top, L1 bottom) ---
    fig, axes = plt.subplots(2, n_blocks, figsize=(20, 6), sharey='row')
    for bi in range(n_blocks):
        l0_theta = [gm(t['blocks'][bi]['levels'][0].get('theta', {})) for t in tapes]
        l1_theta = [gm(t['blocks'][bi]['levels'][1].get('theta', {})) for t in tapes]
        axes[0, bi].plot(tape_steps, l0_theta, 'o-', color='steelblue', markersize=2.5)
        axes[0, bi].set_title(f'B{bi}', fontsize=9)
        axes[0, bi].set_ylim(-0.05, 1.1)
        axes[0, bi].grid(True, alpha=0.3)
        if bi == 0: axes[0, bi].set_ylabel('L0 Theta')
        axes[1, bi].plot(tape_steps, l1_theta, 's-', color='darkorange', markersize=2.5)
        axes[1, bi].set_ylim(-0.05, 1.1)
        axes[1, bi].grid(True, alpha=0.3)
        axes[1, bi].set_xlabel('Step', fontsize=7)
        if bi == 0: axes[1, bi].set_ylabel('L1 Theta')
    fig.suptitle('Theta (Inner-Loop LR) — L0 (top) vs L1 (bottom)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'theta_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  theta_evolution.pdf')

    # --- Alpha (2 rows) ---
    fig, axes = plt.subplots(2, n_blocks, figsize=(20, 6), sharey='row')
    for bi in range(n_blocks):
        l0_alpha = [gm(t['blocks'][bi]['levels'][0].get('alpha', {})) for t in tapes]
        l1_alpha = [gm(t['blocks'][bi]['levels'][1].get('alpha', {})) for t in tapes]
        axes[0, bi].plot(tape_steps, l0_alpha, 'o-', color='steelblue', markersize=2.5)
        axes[0, bi].set_title(f'B{bi}', fontsize=9)
        axes[0, bi].set_ylim(-0.05, 1.05)
        axes[0, bi].grid(True, alpha=0.3)
        if bi == 0: axes[0, bi].set_ylabel('L0 Alpha')
        axes[1, bi].plot(tape_steps, l1_alpha, 's-', color='darkorange', markersize=2.5)
        axes[1, bi].set_ylim(0.85, 1.01)
        axes[1, bi].grid(True, alpha=0.3)
        axes[1, bi].set_xlabel('Step', fontsize=7)
        if bi == 0: axes[1, bi].set_ylabel('L1 Alpha')
    fig.suptitle('Alpha (Retention) — L0 (top, note full 0-1 scale) vs L1 (bottom, 0.85-1.0 scale)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'alpha_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  alpha_evolution.pdf')

    # --- Eta (2 rows) ---
    fig, axes = plt.subplots(2, n_blocks, figsize=(20, 6), sharey='row')
    for bi in range(n_blocks):
        l0_eta = [gm(t['blocks'][bi]['levels'][0].get('eta', {})) for t in tapes]
        l1_eta = [gm(t['blocks'][bi]['levels'][1].get('eta', {})) for t in tapes]
        axes[0, bi].plot(tape_steps, l0_eta, 'o-', color='steelblue', markersize=2.5)
        axes[0, bi].set_title(f'B{bi}', fontsize=9)
        axes[0, bi].set_ylim(-0.05, 1.1)
        axes[0, bi].grid(True, alpha=0.3)
        if bi == 0: axes[0, bi].set_ylabel('L0 Eta')
        axes[1, bi].plot(tape_steps, l1_eta, 's-', color='darkorange', markersize=2.5)
        axes[1, bi].set_ylim(0.6, 1.05)
        axes[1, bi].grid(True, alpha=0.3)
        axes[1, bi].set_xlabel('Step', fontsize=7)
        if bi == 0: axes[1, bi].set_ylabel('L1 Eta')
    fig.suptitle('Eta (Output Gate) — L0 (top) vs L1 (bottom)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'eta_evolution.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  eta_evolution.pdf')

def plot_dgd_ratio(tapes):
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi // 4, bi % 4]
        ratios = []
        for t in tapes:
            d0 = t['blocks'][bi]['levels'][0].get('dgd_delta_norm', 0)
            d1 = t['blocks'][bi]['levels'][1].get('dgd_delta_norm', 0)
            ratios.append(d0 / d1 if d1 > 0 else 0)
        ax.plot(tape_steps, ratios, 'o-', color='purple', markersize=2.5)
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f'Block {bi}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if bi % 4 == 0: ax.set_ylabel('DGD Ratio (L0/L1)')
    fig.suptitle('DGD Delta Norm Ratio (L0/L1) by Block', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dgd_ratio.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  dgd_ratio.pdf')

def plot_output_grad_norms(tapes):
    tape_steps = [t['step'] for t in tapes]
    n_blocks = len(tapes[0]['blocks'])

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)
    for bi in range(n_blocks):
        ax = axes[bi // 4, bi % 4]
        l0_og = [t['blocks'][bi]['levels'][0].get('output_grad_norm', 0) for t in tapes]
        l1_og = [t['blocks'][bi]['levels'][1].get('output_grad_norm', 0) for t in tapes]
        ax.plot(tape_steps, l0_og, 'o-', color='steelblue', markersize=2.5, label='L0')
        ax.plot(tape_steps, l1_og, 's-', color='darkorange', markersize=2.5, label='L1')
        ax.set_title(f'Block {bi}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if bi == 0: ax.legend(fontsize=7)
        if bi % 4 == 0: ax.set_ylabel('Output Grad Norm')
    fig.suptitle('Output Gradient Norms (MAG identity check)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'output_grad_norms.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  output_grad_norms.pdf')

def plot_throughput(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, tps = [], []
    for i in range(1, len(steps)):
        dt = steps[i]['elapsed'] - steps[i-1]['elapsed']
        if dt > 0:
            ds = steps[i]['step'] - steps[i-1]['step']
            x.append(steps[i]['step'])
            tps.append(ds * 512 / dt)
    ax.plot(x, tps, alpha=0.1, color='green', linewidth=0.5)
    xs = x[len(x)-len(smooth(tps, 50)):]
    ax.plot(xs, smooth(tps, 50), color='green', linewidth=1.5, label='tok/s (smoothed)')
    ax.axhline(y=np.mean(tps), color='darkgreen', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(tps):.0f} tok/s')
    ax.set_xlabel('Step'); ax.set_ylabel('Tokens/sec')
    ax.set_title('Training Throughput (d=1024, 8 blocks)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'throughput.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  throughput.pdf')

def plot_block_specialization_heatmap(tapes):
    """Heatmap of gate values across blocks at final tape snapshot."""
    t = tapes[-1]
    n_blocks = len(t['blocks'])
    metrics = ['L0 alpha', 'L0 theta', 'L0 eta', 'L1 alpha', 'L1 theta', 'L1 eta']
    data = np.zeros((len(metrics), n_blocks))
    for bi, block in enumerate(t['blocks']):
        l0, l1 = block['levels'][0], block['levels'][1]
        data[0, bi] = gm(l0.get('alpha', {}))
        data[1, bi] = gm(l0.get('theta', {}))
        data[2, bi] = gm(l0.get('eta', {}))
        data[3, bi] = gm(l1.get('alpha', {}))
        data[4, bi] = gm(l1.get('theta', {}))
        data[5, bi] = gm(l1.get('eta', {}))

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels([f'Block {i}' for i in range(n_blocks)])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    # Annotate cells
    for i in range(len(metrics)):
        for j in range(n_blocks):
            color = 'white' if data[i, j] < 0.3 or data[i, j] > 0.7 else 'black'
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f'Gate Value Heatmap at Step {t["step"]} (Block Specialization Profile)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'gate_heatmap.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  gate_heatmap.pdf')

def plot_lr_schedule(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [s['step'] for s in steps]
    lr = [s['lr'] for s in steps]
    ax.plot(x, lr, color='navy', linewidth=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Decay)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lr_schedule.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  lr_schedule.pdf')

def plot_comparative_loss():
    """Overlay loss curves: shakedown d=1024 vs spec25 d=512."""
    spec25_path = os.path.join(RUN_DIR, '..', 'spec25_30k_d512_4b_k2', 'metrics.jsonl')
    if not os.path.exists(spec25_path):
        print('  (skipping comparative loss — spec25 not found)')
        return

    with open(spec25_path) as f:
        s25_steps = [json.loads(l) for l in f if json.loads(l).get('event') == 'step']

    with open(os.path.join(RUN_DIR, 'metrics.jsonl')) as f:
        sd_steps = [json.loads(l) for l in f if json.loads(l).get('event') == 'step']

    fig, ax = plt.subplots(figsize=(10, 5))
    # spec25 (first 10K steps for fair comparison)
    s25_10k = [s for s in s25_steps if s['step'] <= 10000]
    x25 = [s['step'] for s in s25_10k]
    l25 = [s['loss'] for s in s25_10k]
    xs25 = x25[len(x25)-len(smooth(l25)):]
    ax.plot(xs25, smooth(l25), color='steelblue', linewidth=1.5, label='spec25 d=512 4b (smoothed)')

    x_sd = [s['step'] for s in sd_steps]
    l_sd = [s['loss'] for s in sd_steps]
    xs_sd = x_sd[len(x_sd)-len(smooth(l_sd)):]
    ax.plot(xs_sd, smooth(l_sd), color='darkorange', linewidth=1.5, label='shakedown d=1024 8b (smoothed)')

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison: d=512 (4 blocks) vs d=1024 (8 blocks) — First 10K Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'comparative_loss.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  comparative_loss.pdf')

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
    plot_block_specialization_heatmap(tapes)
    plot_lr_schedule(steps)
    plot_comparative_loss()
    print('Done — all plots saved to', OUT_DIR)
