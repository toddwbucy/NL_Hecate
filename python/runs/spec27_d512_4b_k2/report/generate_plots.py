#!/usr/bin/env python3
"""Generate PDF plots for spec27_d512_4b_k2 extended 130K experiment report.
Two-phase run: 30K initial + 100K warm restart. Compares with spec27_30k and spec25."""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# ============================================================
# Plot 1: Full loss trajectory with phase markers
# ============================================================
def plot_loss_ppl(steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = [s['step'] for s in steps]; loss = [s['loss'] for s in steps]
    ppl = [min(s['ppl'], 500) for s in steps]

    # Loss
    ax1.plot(x, loss, alpha=0.1, color='steelblue', linewidth=0.5)
    xs = x[len(x)-len(smooth(loss)):]; ax1.plot(xs, smooth(loss), color='steelblue', linewidth=1.5, label='Loss (smoothed)')
    ax1.axvline(x=30000, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Phase boundary (30K)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss — Full 130K Trajectory')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # PPL
    ax2.plot(x, ppl, alpha=0.1, color='darkorange', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(ppl)):]; ax2.plot(xs2, smooth(ppl), color='darkorange', linewidth=1.5, label='PPL (smoothed)')
    ax2.axvline(x=30000, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Phase boundary')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Perplexity'); ax2.set_title('Perplexity (clipped at 500)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'loss_ppl.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  loss_ppl.pdf')

# ============================================================
# Plot 2: Loss overlay — 130K vs 30K-only vs spec25
# ============================================================
def plot_comparative_loss(s130k, s25):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: first 30K overlay — spec25 vs spec27 (this run's first phase)
    for steps, label, color in [
        (s25, 'spec25 (all exact, 30K)', 'steelblue'),
        (s130k, 'spec27 (L1 proxy, first 30K)', 'darkorange'),
    ]:
        x = [s['step'] for s in steps]; l = [s['loss'] for s in steps]
        mask = [(xi, li) for xi, li in zip(x, l) if xi <= 30000]
        if not mask: continue
        xm, lm = zip(*mask)
        xs = xm[len(xm)-len(smooth(lm)):]; ax1.plot(xs, smooth(lm), color=color, linewidth=1.5, label=label, alpha=0.8)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.set_title('First 30K: spec25 vs spec27')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Right: full 130K trajectory
    x = [s['step'] for s in s130k]; l = [s['loss'] for s in s130k]
    ax2.plot(x, l, alpha=0.1, color='darkgreen', linewidth=0.5)
    xs = x[len(x)-len(smooth(l)):]; ax2.plot(xs, smooth(l), color='darkgreen', linewidth=1.5, label='130K trajectory')
    ax2.axvline(x=30000, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Phase boundary')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Loss'); ax2.set_title('Full 130K Trajectory')
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
    ax1.axvline(x=30000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Gradient Norm'); ax1.set_title('Global Gradient Norm'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(x, cv, alpha=0.15, color='teal', linewidth=0.5)
    xs2 = x[len(x)-len(smooth(cv)):]; ax2.plot(xs2, smooth(cv), color='teal', linewidth=1.5, label='Block gnorm CV (smoothed)')
    ax2.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Specialization threshold')
    ax2.axvline(x=30000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_xlabel('Step'); ax2.set_ylabel('CV'); ax2.set_title('Depth Specialization (Block CV)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  grad_norms.pdf')

# ============================================================
# Plot 4: Per-block gradient norms
# ============================================================
def plot_block_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]; colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for b in range(4):
        vals = [s['block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]; ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b}')
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm'); ax.set_title('Per-Block Gradient Norms (130K)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'block_grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  block_grad_norms.pdf')

# ============================================================
# Plot 5: L0 per-block gradient norms
# ============================================================
def plot_l0_grad_norms(steps):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [s['step'] for s in steps]; colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for b in range(4):
        vals = [s['l0_block_grad_norms'][b] for s in steps]
        xs = x[len(x)-len(smooth(vals)):]; ax.plot(xs, smooth(vals), color=colors[b], linewidth=1.5, label=f'Block {b} L0')
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('L0 Gradient Norm'); ax.set_title('L0 Per-Block Gradient Norms')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'l0_grad_norms.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  l0_grad_norms.pdf')

# ============================================================
# Plot 6: Gate diagnostics — theta, alpha, eta (all 130K)
# ============================================================
def plot_gate_diagnostics(tapes):
    tape_steps = [t['step'] for t in tapes]

    for gate_name, ylim, extra_line in [('theta', (-0.05, 1.1), None), ('alpha', (0.75, 1.02), 0.8), ('eta', (0.0, 1.05), None)]:
        fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
        for bi in range(4):
            ax = axes[bi]
            l0 = [gm(t['blocks'][bi]['levels'][0].get(gate_name,{})) for t in tapes]
            l1 = [gm(t['blocks'][bi]['levels'][1].get(gate_name,{})) for t in tapes]
            ax.plot(tape_steps, l0, '-', color='steelblue', linewidth=1.2, label='L0 (exact)', alpha=0.8)
            ax.plot(tape_steps, l1, '-', color='darkorange', linewidth=1.2, label='L1 (proxy)', alpha=0.8)
            ax.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1)
            if extra_line is not None:
                ax.axhline(y=extra_line, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_title(f'Block {bi}'); ax.set_xlabel('Step')
            if bi == 0: ax.set_ylabel(gate_name.capitalize())
            ax.set_ylim(*ylim); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle(f'{gate_name.capitalize()} Evolution — 130K (L0=exact, L1=proxy)', fontsize=13, y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f'{gate_name}_evolution.pdf'), bbox_inches='tight'); plt.close(fig)
        print(f'  {gate_name}_evolution.pdf')

# ============================================================
# Plot 7: L1 eta collapse timeline (the headline)
# ============================================================
def plot_eta_collapse(tapes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    tape_steps = [t['step'] for t in tapes]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

    # Left: L1 eta all blocks
    for bi in range(4):
        l1_eta = [gm(t['blocks'][bi]['levels'][1].get('eta',{})) for t in tapes]
        ax1.plot(tape_steps, l1_eta, '-o', color=colors[bi], markersize=2, linewidth=1.2, label=f'Block {bi}')
    ax1.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Phase boundary')
    ax1.axvline(x=57000, color='purple', linestyle=':', alpha=0.6, linewidth=1.5, label='Collapse onset (~57K)')
    ax1.set_xlabel('Step'); ax1.set_ylabel('L1 Eta'); ax1.set_title('L1 Eta — Block 0 Collapses')
    ax1.set_ylim(0.0, 1.05); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # Right: L0 eta all blocks (for contrast — saturates to 1.0)
    for bi in range(4):
        l0_eta = [gm(t['blocks'][bi]['levels'][0].get('eta',{})) for t in tapes]
        ax2.plot(tape_steps, l0_eta, '-o', color=colors[bi], markersize=2, linewidth=1.2, label=f'Block {bi}')
    ax2.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_xlabel('Step'); ax2.set_ylabel('L0 Eta'); ax2.set_title('L0 Eta — Saturates to 1.0')
    ax2.set_ylim(0.85, 1.005); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'eta_collapse.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  eta_collapse.pdf')

# ============================================================
# Plot 8: Alpha floor-pinning
# ============================================================
def plot_alpha_pinning(tapes):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    tape_steps = [t['step'] for t in tapes]
    for bi in range(4):
        ax = axes[bi]
        l1_alpha = [gm(t['blocks'][bi]['levels'][1].get('alpha',{})) for t in tapes]
        ax.plot(tape_steps, l1_alpha, '-', color='darkorange', linewidth=1.2)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.6, linewidth=1, label='alpha_floor')
        ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_title(f'Block {bi}'); ax.set_xlabel('Step')
        if bi == 0: ax.set_ylabel('L1 Alpha')
        ax.set_ylim(0.78, 1.02); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle('L1 Alpha — Approaches Floor (0.8) Over 130K Steps', fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'alpha_pinning.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  alpha_pinning.pdf')

# ============================================================
# Plot 9: Block CV over time
# ============================================================
def plot_block_cv_trajectory(steps):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = [s['step'] for s in steps]; cv = [s['block_gnorm_cv'] for s in steps]
    ax.plot(x, cv, alpha=0.15, color='teal', linewidth=0.5)
    xs = x[len(x)-len(smooth(cv)):]; ax.plot(xs, smooth(cv), color='teal', linewidth=1.5, label='Block CV (smoothed)')
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Specialization threshold')
    ax.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Phase boundary')
    # Annotate key values
    for target, val in [(5000, '0.55'), (30000, '0.37'), (60000, '0.19'), (129000, '0.18')]:
        ax.annotate(val, xy=(target, float(val)), fontsize=8, color='darkred', ha='center', va='bottom')
    ax.set_xlabel('Step'); ax.set_ylabel('Block Gnorm CV'); ax.set_title('Depth Specialization Trajectory — Decreasing Over 130K')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'block_cv_trajectory.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  block_cv_trajectory.pdf')

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
    ax.plot(xs, smooth(tps, 100), color='darkgreen', linewidth=1.5, label=f'Throughput (mean: {np.mean(tps):.0f} tok/s)')
    ax.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Phase boundary')
    ax.set_xlabel('Step'); ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput — 130K Extended Run')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'throughput.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  throughput.pdf')

# ============================================================
# Plot 11: L1 theta — Block 2 anomaly highlight
# ============================================================
def plot_theta_block2(tapes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    tape_steps = [t['step'] for t in tapes]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

    # Left: L1 theta all blocks
    for bi in range(4):
        l1_theta = [gm(t['blocks'][bi]['levels'][1].get('theta',{})) for t in tapes]
        ax1.plot(tape_steps, l1_theta, '-o', color=colors[bi], markersize=2, linewidth=1.2, label=f'Block {bi}')
    ax1.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1, label='Phase boundary')
    ax1.set_xlabel('Step'); ax1.set_ylabel('L1 Theta'); ax1.set_title('L1 Theta — Block 2 Rises, Others Stay Low')
    ax1.set_ylim(-0.05, 1.0); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # Right: Block 2 L1 theta + L0 theta for contrast
    l1_b2 = [gm(t['blocks'][2]['levels'][1].get('theta',{})) for t in tapes]
    l0_b2 = [gm(t['blocks'][2]['levels'][0].get('theta',{})) for t in tapes]
    ax2.plot(tape_steps, l0_b2, '-', color='steelblue', linewidth=1.2, label='B2 L0 (exact)')
    ax2.plot(tape_steps, l1_b2, '-', color='darkorange', linewidth=1.2, label='B2 L1 (proxy)')
    ax2.axvline(x=30000, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Theta'); ax2.set_title('Block 2 Theta — L1 Continues Rising Under Proxy')
    ax2.set_ylim(-0.05, 1.1); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, 'theta_block2.pdf'), bbox_inches='tight'); plt.close(fig)
    print('  theta_block2.pdf')

# ============================================================
if __name__ == '__main__':
    print('Loading metrics...')
    s130k_steps, s130k_tapes = load_run(RUN_DIR)
    print(f'  130K run: {len(s130k_steps)} steps, {len(s130k_tapes)} tapes')

    # Load comparator
    s25_steps, s25_tapes = [], []
    try:
        s25_steps, s25_tapes = load_run(SPEC25_DIR)
        print(f'  spec25: {len(s25_steps)} steps, {len(s25_tapes)} tapes')
    except FileNotFoundError:
        print('  spec25: not found, skipping comparative')

    print('Generating plots...')
    plot_loss_ppl(s130k_steps)
    if s25_steps:
        plot_comparative_loss(s130k_steps, s25_steps)
    plot_grad_norms(s130k_steps)
    plot_block_grad_norms(s130k_steps)
    plot_l0_grad_norms(s130k_steps)
    plot_gate_diagnostics(s130k_tapes)
    plot_eta_collapse(s130k_tapes)
    plot_alpha_pinning(s130k_tapes)
    plot_block_cv_trajectory(s130k_steps)
    plot_throughput(s130k_steps)
    plot_theta_block2(s130k_tapes)
    print('Done — all plots saved to', OUT_DIR)
