#!/usr/bin/env python3
"""Generate PDF plots for pushup_k3_from_spec27_30k experiment report.
Push-up from k=2 (spec27 30K checkpoint) to k=3 with L0=exact, L1/L2=proxy.
Compares with spec25 (k=2 exact) and spec27_d512_4b_k2 (k=2 proxy 130K)."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC27_130K_DIR = os.path.join(RUN_DIR, '..', 'spec27_d512_4b_k2')
SPEC25_DIR = os.path.join(RUN_DIR, '..', 'spec25_30k_d512_4b_k2')
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

def smooth(values, window=50):
    if len(values) < window: return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def gm(d, key='mean'):
    if isinstance(d, dict): return float(d.get(key, 0))
    return float(d) if d is not None else 0.0

COLORS4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ============================================================
# Plot 1: Loss and perplexity
# ============================================================
def plot_loss_ppl(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = [s['step'] for s in steps]; loss = [s['loss'] for s in steps]
    ppl = [min(s['ppl'], 500) for s in steps]

    ax1.plot(x, loss, alpha=0.1, color='steelblue', linewidth=0.5)
    xs = x[len(x)-len(smooth(loss)):]; ax1.plot(xs, smooth(loss), color='steelblue', linewidth=1.5, label='Loss (smoothed)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss — k=3 Push-Up (96K Steps)')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, ppl, alpha=0.1, color='darkorange', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(ppl)):]; ax2.plot(xs2, smooth(ppl), color='darkorange', linewidth=1.5, label='PPL (smoothed)')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Perplexity'); ax2.set_title('Perplexity (clipped at 500)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'loss_ppl.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  loss_ppl.pdf')

# ============================================================
# Plot 2: Comparative loss — k=3 vs k=2 runs
# ============================================================
def plot_comparative_loss(k3_steps, k2_130k_steps, k2_25_steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: all three runs, full trajectory
    for steps, label, color in [
        (k2_25_steps, 'spec25 k=2 exact (30K)', 'steelblue'),
        (k2_130k_steps, 'spec27 k=2 proxy (130K)', 'darkgreen'),
        (k3_steps, 'push-up k=3 proxy (96K)', 'darkorange'),
    ]:
        x = [s['step'] for s in steps]; l = [s['loss'] for s in steps]
        xs = x[len(x)-len(smooth(l)):]; ax1.plot(xs, smooth(l), color=color, linewidth=1.5, label=label, alpha=0.8)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Loss Comparison: k=2 vs k=3')
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # Right: k=3 alone with annotations
    x = [s['step'] for s in k3_steps]; l = [s['loss'] for s in k3_steps]
    ax2.plot(x, l, alpha=0.1, color='darkorange', linewidth=0.5)
    xs = x[len(x)-len(smooth(l)):]; ax2.plot(xs, smooth(l), color='darkorange', linewidth=1.5, label='k=3 push-up')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Loss'); ax2.set_title('k=3 Push-Up Trajectory')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'comparative_loss.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  comparative_loss.pdf')

# ============================================================
# Plot 3: Gradient norms + block CV
# ============================================================
def plot_grad_norms(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = [s['step'] for s in steps]; gnorm = [s['grad_norm'] for s in steps]; cv = [s['block_gnorm_cv'] for s in steps]

    ax1.plot(x, gnorm, alpha=0.15, color='crimson', linewidth=0.5)
    xs = x[len(x)-len(smooth(gnorm)):]; ax1.plot(xs, smooth(gnorm), color='crimson', linewidth=1.5, label='Grad norm (smoothed)')
    spike_x = [s['step'] for s in steps if s['grad_norm'] > 10]; spike_y = [s['grad_norm'] for s in steps if s['grad_norm'] > 10]
    if spike_x: ax1.scatter(spike_x, spike_y, color='red', s=30, zorder=5, label=f'Spikes >10 ({len(spike_x)})')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Gradient Norm'); ax1.set_title('Global Gradient Norm'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, cv, alpha=0.15, color='teal', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(cv)):]; ax2.plot(xs2, smooth(cv), color='teal', linewidth=1.5, label='Block gnorm CV (smoothed)')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Specialization threshold')
    ax2.set_xlabel('Step'); ax2.set_ylabel('CV'); ax2.set_title('Depth Specialization'); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  grad_norms.pdf')

# ============================================================
# Plot 4: Per-block gradient norms
# ============================================================
def plot_block_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]
    for b in range(4):
        vals = [s['block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]; ax.plot(xs, smooth(vals), color=COLORS4[b], linewidth=1.5, label=f'Block {b}')
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm'); ax.set_title('Per-Block Gradient Norms (k=3)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'block_grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  block_grad_norms.pdf')

# ============================================================
# Plot 5: L0 per-block gradient norms
# ============================================================
def plot_l0_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]
    for b in range(4):
        vals = [s['l0_block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]; ax.plot(xs, smooth(vals), color=COLORS4[b], linewidth=1.5, label=f'Block {b} L0')
    ax.set_xlabel('Step'); ax.set_ylabel('L0 Gradient Norm'); ax.set_title('L0 Per-Block Gradient Norms')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'l0_grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  l0_grad_norms.pdf')

# ============================================================
# Plot 6: Gate diagnostics — theta, alpha, eta for all 3 levels
# ============================================================
def plot_gate_evolution_3level(tapes):
    tape_steps = [t['step'] for t in tapes]
    level_colors = ['steelblue', 'darkorange', 'forestgreen']
    level_labels = ['L0 (exact)', 'L1 (proxy)', 'L2 (proxy)']

    for gate_name, ylim, extra_line in [('theta', (-0.05, 1.1), None), ('alpha', (0.75, 1.02), 0.8), ('eta', (0.0, 1.05), None)]:
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
        for bi in range(4):
            ax = axes[bi]
            for li in range(3):
                vals = [gm(t['blocks'][bi]['levels'][li].get(gate_name, {})) for t in tapes]
                ax.plot(tape_steps, vals, '-', color=level_colors[li], linewidth=1.2, label=level_labels[li], alpha=0.8)
            if extra_line is not None:
                ax.axhline(y=extra_line, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_title(f'Block {bi}'); ax.set_xlabel('Step')
            if bi == 0: ax.set_ylabel(gate_name.capitalize())
            ax.set_ylim(*ylim); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        fig.suptitle(f'{gate_name.capitalize()} Evolution — k=3 (L0=exact, L1/L2=proxy)', fontsize=13, y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'{gate_name}_evolution.pdf'), bbox_inches='tight'); plt.close(fig)
        print(f'  {gate_name}_evolution.pdf')

# ============================================================
# Plot 7: Level hierarchy — theta per level across blocks
# ============================================================
def plot_level_hierarchy(tapes):
    tape_steps = [t['step'] for t in tapes]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for li, (ax, title) in enumerate(zip(axes, ['L0 Theta (exact)', 'L1 Theta (proxy)', 'L2 Theta (proxy)'])):
        for bi in range(4):
            vals = [gm(t['blocks'][bi]['levels'][li].get('theta', {})) for t in tapes]
            ax.plot(tape_steps, vals, '-o', color=COLORS4[bi], markersize=2, linewidth=1.2, label=f'Block {bi}')
        ax.set_title(title); ax.set_xlabel('Step')
        if li == 0: ax.set_ylabel('Theta')
        ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Theta by Level — Which Blocks Learn at Which Level?', fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'level_hierarchy_theta.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  level_hierarchy_theta.pdf')

# ============================================================
# Plot 8: Level hierarchy — eta per level across blocks
# ============================================================
def plot_level_hierarchy_eta(tapes):
    tape_steps = [t['step'] for t in tapes]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for li, (ax, title) in enumerate(zip(axes, ['L0 Eta (exact)', 'L1 Eta (proxy)', 'L2 Eta (proxy)'])):
        for bi in range(4):
            vals = [gm(t['blocks'][bi]['levels'][li].get('eta', {})) for t in tapes]
            ax.plot(tape_steps, vals, '-o', color=COLORS4[bi], markersize=2, linewidth=1.2, label=f'Block {bi}')
        ax.set_title(title); ax.set_xlabel('Step')
        if li == 0: ax.set_ylabel('Eta')
        ax.set_ylim(0.6, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Eta by Level — Output Gate Hierarchy', fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'level_hierarchy_eta.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  level_hierarchy_eta.pdf')

# ============================================================
# Plot 9: DGD delta norms — L0 only (L1/L2 are 0.00 under proxy)
# ============================================================
def plot_dgd_norms(tapes):
    tape_steps = [t['step'] for t in tapes]
    fig, ax = plt.subplots(figsize=(10, 5))
    for bi in range(4):
        vals = [gm(t['blocks'][bi]['levels'][0].get('dgd_delta_norm', {})) for t in tapes]
        ax.plot(tape_steps, vals, '-o', color=COLORS4[bi], markersize=2, linewidth=1.2, label=f'Block {bi} L0')
    ax.set_xlabel('Step'); ax.set_ylabel('DGD Delta Norm')
    ax.set_title('L0 DGD Delta Norms (L1/L2 = 0.00 under proxy)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'dgd_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  dgd_norms.pdf')

# ============================================================
# Plot 10: Throughput
# ============================================================
def plot_throughput(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x, tps = [], []
    for i in range(1, len(steps)):
        dt = steps[i]['elapsed'] - steps[i-1]['elapsed']
        if dt > 0:
            ds = steps[i]['step'] - steps[i-1]['step']
            x.append(steps[i]['step']); tps.append(ds * 512 / dt)
    xs = x[len(x)-len(smooth(tps, 100)):]
    ax.plot(xs, smooth(tps, 100), color='darkorange', linewidth=1.5, label=f'Throughput (mean: {np.mean(tps):.0f} tok/s)')
    ax.set_xlabel('Step'); ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput — k=3 Push-Up')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'throughput.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  throughput.pdf')

# ============================================================
# Plot 11: Alpha floor-pinning — all 3 levels
# ============================================================
def plot_alpha_pinning(tapes):
    tape_steps = [t['step'] for t in tapes]
    level_colors = ['steelblue', 'darkorange', 'forestgreen']
    level_labels = ['L0', 'L1', 'L2']

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for bi in range(4):
        ax = axes[bi]
        for li in range(3):
            vals = [gm(t['blocks'][bi]['levels'][li].get('alpha', {})) for t in tapes]
            ax.plot(tape_steps, vals, '-', color=level_colors[li], linewidth=1.2, label=level_labels[li])
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.6, linewidth=1, label='alpha_floor')
        ax.set_title(f'Block {bi}'); ax.set_xlabel('Step')
        if bi == 0: ax.set_ylabel('Alpha')
        ax.set_ylim(0.78, 1.02); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    fig.suptitle('Alpha — All 3 Levels (k=3)', fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'alpha_pinning.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  alpha_pinning.pdf')

# ============================================================
# Plot 12: Block 2 three-level detail
# ============================================================
def plot_block2_detail(tapes):
    tape_steps = [t['step'] for t in tapes]
    level_colors = ['steelblue', 'darkorange', 'forestgreen']
    level_labels = ['L0 (exact)', 'L1 (proxy)', 'L2 (proxy)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for gi, (gate, ylim) in enumerate([('theta', (-0.05, 1.1)), ('alpha', (0.78, 1.02)), ('eta', (0.7, 1.05))]):
        ax = axes[gi]
        for li in range(3):
            vals = [gm(t['blocks'][2]['levels'][li].get(gate, {})) for t in tapes]
            ax.plot(tape_steps, vals, '-', color=level_colors[li], linewidth=1.5, label=level_labels[li])
        if gate == 'alpha':
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f'Block 2 {gate.capitalize()}'); ax.set_xlabel('Step')
        ax.set_ylim(*ylim); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('Block 2 — The Anomalous Block (All 3 Levels)', fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'block2_detail.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  block2_detail.pdf')

# ============================================================
if __name__ == '__main__':
    print('Loading metrics...')
    k3_steps, k3_tapes = load_run(RUN_DIR)
    print(f'  k=3 push-up: {len(k3_steps)} steps, {len(k3_tapes)} tapes')

    # Load comparators
    k2_130k_steps, k2_130k_tapes = [], []
    k2_25_steps, k2_25_tapes = [], []
    try:
        k2_130k_steps, k2_130k_tapes = load_run(SPEC27_130K_DIR)
        print(f'  spec27 k=2 130K: {len(k2_130k_steps)} steps')
    except FileNotFoundError:
        print('  spec27 k=2 130K: not found')
    try:
        k2_25_steps, k2_25_tapes = load_run(SPEC25_DIR)
        print(f'  spec25 k=2: {len(k2_25_steps)} steps')
    except FileNotFoundError:
        print('  spec25 k=2: not found')

    print('Generating plots...')
    plot_loss_ppl(k3_steps)
    if k2_130k_steps or k2_25_steps:
        plot_comparative_loss(k3_steps, k2_130k_steps, k2_25_steps)
    plot_grad_norms(k3_steps)
    plot_block_grad_norms(k3_steps)
    plot_l0_grad_norms(k3_steps)
    plot_gate_evolution_3level(k3_tapes)
    plot_level_hierarchy(k3_tapes)
    plot_level_hierarchy_eta(k3_tapes)
    plot_dgd_norms(k3_tapes)
    plot_throughput(k3_steps)
    plot_alpha_pinning(k3_tapes)
    plot_block2_detail(k3_tapes)
    print('Done — all plots saved to', OUT_DIR)
