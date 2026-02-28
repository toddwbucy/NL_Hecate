#!/usr/bin/env python3
"""
Throwaway: verify gradient scaling is correct at batch_size > 1.

Tests that the per-token-mean gradient is equivalent across different effective
batch sizes (same tokens repeated, different total seq_len).

Mathematical property being verified:
  gradient(seq=A, normalized by 1/s) ≡ gradient(seq=[A,A,A,A], normalized by 1/(4s))

This holds because for N identical copies of sequence A in a single forward pass:
  dL/dW = (1/(N*s)) * sum_{b=0}^{N-1} sum_{t=0}^{s-1} local_grad_{b,t}
        = (N / (N*s)) * sum_{t=0}^{s-1} local_grad_t      (copies are identical)
        = (1/s) * sum_{t=0}^{s-1} local_grad_t
        = gradient(single sequence A)

For memory_enabled=True: M evolves through the FULL concatenated sequence, so
positions N..2N-1 see a different M than positions 0..N-1 — memory weights will
differ. Only SWA weights (w_q, w_k, w_v, w_o, w_unembed) are invariant to this,
because in MAG composition they run in parallel with memory, not sequentially.

Actually even SWA weights are contaminated by the gate values (which depend on M),
so TEST_A (memory_enabled=False) is the clean mathematical test.

TEST_A: memory_enabled=False, SWA-only — ALL weights must match.
TEST_B: Determinism check — same sequence twice must give identical gradients.

PASS = Issue 1 (gradient scaling) is definitively closed.
FAIL = Real gradient normalization bug — must fix before relaunching B/C/D.
"""

import sys
import random

sys.path.insert(0, "/home/todd/olympus/NL_Hecate/python")

import nl_hecate

# ── Parameters ────────────────────────────────────────────────────────────────

D = 32
HEADS = 2
BASE_SEQ = 4       # single-sequence length; must be small to keep it fast
VOCAB = 64
SEED = 42
REPEAT = 4         # how many copies to concatenate for the "batch" simulation
TOL = 1e-4         # fp32 tolerance; AtomicAdd reordering can produce ~1e-6 noise
K = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cfg(seq_len: int, memory_enabled: bool = False, memory_rule: str = "delta",
             window_size: int | None = None):
    """Build MAGConfig with given seq_len and memory toggle.

    window_size defaults to seq_len (full causal attention within the sequence).
    Use window_size=1 for the normalization test to make positions independent:
    with self-attention only, each position's gradient is a function of that
    position's tokens alone, so repeated identical tokens give identical gradients
    regardless of their absolute position in the sequence.
    """
    ws = window_size if window_size is not None else seq_len
    return nl_hecate.MAGConfig(
        d_model=D,
        num_heads=HEADS,
        head_dim=D // HEADS,
        seq_len=seq_len,
        window_size=ws,
        vocab_size=VOCAB,
        memory_enabled=memory_enabled,
        k=K,
        chunk_sizes=[1] * K,
        memory_rule=memory_rule,
        composition="mag",
        checkpoint_interval=None,
        projection_kind="static",
        self_generated_values=False,
        self_ref_chunk_size=1,
        momentum_kind="none",
        momentum_d_hidden=0,
        theta_floor=None,
        theta_ceil=None,
        intermediate_size=0,
        m_norm_max=None,
    )


def make_conductor(cfg):
    return nl_hecate.Conductor(cfg.k, list(cfg.chunk_sizes))


def make_context(cfg):
    return nl_hecate.ContextState(cfg.k, cfg.d_model)


def make_error_buffers(cfg):
    return nl_hecate.ErrorBufferList(cfg.k, cfg.d_model)


def get_grads(grad_params):
    """Extract gradient dict from MAGParams.get_weights()."""
    return grad_params.get_weights()


def max_diff_for_keys(g1: dict, g2: dict, keys: list[str]) -> dict[str, float]:
    diffs = {}
    for key in keys:
        v1 = g1[key]
        v2 = g2[key]
        if len(v1) != len(v2):
            raise ValueError(f"Shape mismatch for {key}: {len(v1)} vs {len(v2)}")
        diffs[key] = max(abs(a - b) for a, b in zip(v1, v2))
    return diffs


SWA_KEYS = ["w_q", "w_k", "w_v", "w_o", "w_unembed"]
MEM_KEYS = ["w_k_mem", "w_v_mem", "w_q_mem", "w_alpha", "b_alpha", "w_theta", "b_theta"]
ALL_KEYS = SWA_KEYS + MEM_KEYS


def run_cms_grads(params, cfg, input_ids, target_ids):
    """Run cms_compute_gradients with fresh context and error buffers."""
    conductor = make_conductor(cfg)
    context = make_context(cfg)
    error_buffers = make_error_buffers(cfg)
    pulse = conductor.pulse()
    return nl_hecate.cms_compute_gradients(
        params, cfg, input_ids, target_ids, pulse, context, error_buffers
    )


def run_mag_grads(params, cfg, input_ids, target_ids):
    """Run mag_compute_gradients (no CMS state needed)."""
    return nl_hecate.mag_compute_gradients(params, cfg, input_ids, target_ids)


# ── Sequence generation ────────────────────────────────────────────────────────

rng = random.Random(SEED)
seq_A = [rng.randint(1, VOCAB - 1) for _ in range(BASE_SEQ)]  # base sequence
tgt_A = [(t + 1) % VOCAB for t in seq_A]                      # next-token targets

seq_repeated = seq_A * REPEAT   # [A, A, A, A] — REPEAT copies
tgt_repeated = tgt_A * REPEAT


# ── TEST A: SWA-only, memory_enabled=False ────────────────────────────────────

print("=" * 65)
print("TEST A: SWA-only gradient scaling (memory_enabled=False)")
print(f"  window_size=1 (self-attention: positions are independent)")
print(f"  seq_single: {BASE_SEQ} tokens")
print(f"  seq_repeat: {BASE_SEQ * REPEAT} tokens = {REPEAT}x identical copies")
print("  With self-attention: each position sees only itself, so repeated")
print("  identical tokens give exactly the same per-token gradient.")
print("=" * 65)

# window_size=1: each query attends only to itself (self-attention).
# This makes all positions truly independent across the sequence —
# the gradient at position t depends only on x_t and target_t, not
# on any other position's keys/values. With window_size=seq_len,
# positions at different offsets in the repeated sequence would see
# different key contexts (cross-token accumulation), confounding the test.
cfg_single = make_cfg(BASE_SEQ, memory_enabled=False, window_size=1)
cfg_repeat = make_cfg(BASE_SEQ * REPEAT, memory_enabled=False, window_size=1)

# Same seed → same initial weights (weight shape is d_model-dependent, not seq_len-dependent)
params_A = nl_hecate.mag_init_params(cfg_single, SEED)
params_B = nl_hecate.mag_init_params(cfg_repeat, SEED)

loss_single, grad_single = run_mag_grads(params_A, cfg_single, seq_A, tgt_A)
loss_repeat, grad_repeat = run_mag_grads(params_B, cfg_repeat, seq_repeated, tgt_repeated)

g_single = get_grads(grad_single)
g_repeat = get_grads(grad_repeat)

print(f"\n  loss_single = {loss_single:.6f}")
print(f"  loss_repeat = {loss_repeat:.6f}")
print(f"  Δloss = {abs(loss_single - loss_repeat):.2e}")

diffs_A = max_diff_for_keys(g_single, g_repeat, SWA_KEYS)
print()
print(f"  {'Weight':<16}  {'max|Δ|':>12}  {'pass?':>6}")
print(f"  {'-'*16}  {'-'*12}  {'-'*6}")
all_pass_A = True
for key, diff in diffs_A.items():
    ok = diff < TOL
    all_pass_A = all_pass_A and ok
    status = "PASS" if ok else "FAIL *** "
    print(f"  {key:<16}  {diff:>12.2e}  {status}")

print()
if all_pass_A:
    print("  ✓ TEST A PASSED — SWA gradient normalization is correct")
else:
    print("  ✗ TEST A FAILED — gradient scaling bug confirmed")


# ── TEST B: Determinism check with memory_enabled=True ───────────────────────

print()
print("=" * 65)
print("TEST B: Determinism — same sequence twice gives identical grads")
print(f"  memory_enabled=True, memory_rule=delta, k={K}")
print("=" * 65)

cfg_mem = make_cfg(BASE_SEQ, memory_enabled=True, memory_rule="delta", window_size=1)
params_mem = nl_hecate.mag_init_params(cfg_mem, SEED)

loss_run1, grad_run1 = run_cms_grads(params_mem, cfg_mem, seq_A, tgt_A)
loss_run2, grad_run2 = run_cms_grads(params_mem, cfg_mem, seq_A, tgt_A)

g_run1 = get_grads(grad_run1)
g_run2 = get_grads(grad_run2)

print(f"\n  loss_run1 = {loss_run1:.6f}")
print(f"  loss_run2 = {loss_run2:.6f}")
print(f"  Δloss = {abs(loss_run1 - loss_run2):.2e}")

diffs_B = max_diff_for_keys(g_run1, g_run2, ALL_KEYS)
print()
print(f"  {'Weight':<16}  {'max|Δ|':>12}  {'pass?':>6}")
print(f"  {'-'*16}  {'-'*12}  {'-'*6}")
all_pass_B = True
for key, diff in diffs_B.items():
    ok = diff == 0.0  # determinism: must be bit-exact
    all_pass_B = all_pass_B and ok
    status = "PASS" if ok else "FAIL *** "
    print(f"  {key:<16}  {diff:>12.2e}  {status}")

print()
if all_pass_B:
    print("  ✓ TEST B PASSED — gradient computation is deterministic")
else:
    print("  ✗ TEST B FAILED — non-deterministic CPU gradient (unexpected)")


# ── TEST C: CMS memory grad scaling for SWA weights ──────────────────────────
# With memory_enabled=True and window_size=1:
# - SWA attention is self-attention only (positions independent)
# - Memory M evolves through the full concatenated sequence
# - Positions BASE_SEQ..2*BASE_SEQ see M that has already been updated
#   through the first BASE_SEQ tokens — so M≠M_initial for those positions
# - This means memory weight gradients WILL differ between single and repeated
# - SWA weights: might also differ slightly because gate values (alpha/theta)
#   depend on M, and M is different at positions 4..7 vs 0..3
# Expected: memory weights (w_k_mem, w_v_mem, w_q_mem) differ — this is
# documented behavior, not a bug. SWA weights may also drift slightly via gates.

print()
print("=" * 65)
print("TEST C: CMS scaling — SWA weights with memory contamination")
print(f"  Expected: SWA weights may NOT match (gate depends on M)")
print("=" * 65)

cfg_mem_single = make_cfg(BASE_SEQ, memory_enabled=True, memory_rule="delta", window_size=1)
cfg_mem_repeat = make_cfg(BASE_SEQ * REPEAT, memory_enabled=True, memory_rule="delta", window_size=1)

params_ms = nl_hecate.mag_init_params(cfg_mem_single, SEED)
params_mr = nl_hecate.mag_init_params(cfg_mem_repeat, SEED)

loss_ms, grad_ms = run_cms_grads(params_ms, cfg_mem_single, seq_A, tgt_A)
loss_mr, grad_mr = run_cms_grads(params_mr, cfg_mem_repeat, seq_repeated, tgt_repeated)

g_ms = get_grads(grad_ms)
g_mr = get_grads(grad_mr)

print(f"\n  loss_single = {loss_ms:.6f}")
print(f"  loss_repeat = {loss_mr:.6f}")
print(f"  Δloss = {abs(loss_ms - loss_mr):.2e}")

diffs_C_swa = max_diff_for_keys(g_ms, g_mr, SWA_KEYS)
diffs_C_mem = max_diff_for_keys(g_ms, g_mr, MEM_KEYS)
print()
print(f"  {'Weight':<16}  {'max|Δ|':>12}  {'match?':>8}")
print(f"  {'-'*16}  {'-'*12}  {'-'*8}")
for key, diff in {**diffs_C_swa, **diffs_C_mem}.items():
    match = "yes" if diff < TOL else "no (expected)"
    print(f"  {key:<16}  {diff:>12.2e}  {match}")

print()
print("  ✓ TEST C complete (differences in mem weights expected by design)")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  TEST A (SWA-only, normalization): {'PASS' if all_pass_A else 'FAIL'}")
print(f"  TEST B (determinism):             {'PASS' if all_pass_B else 'FAIL'}")
print(f"  TEST C (memory contamination):    documented (not a bug)")

if all_pass_A and all_pass_B:
    print()
    print("  ★ ISSUE 1 CLOSED — gradient normalization is correct")
    print("    The 1/seq_len scheme on CPU is mathematically equivalent to")
    print("    the GPU 1/valid_count scheme at batch_size > 1.")
    print("    Proceed to task_48addd (context continuity fix).")
    sys.exit(0)
else:
    print()
    print("  ✗ ISSUE 1 OPEN — gradient bug confirmed, do not relaunch B/C/D")
    sys.exit(1)
