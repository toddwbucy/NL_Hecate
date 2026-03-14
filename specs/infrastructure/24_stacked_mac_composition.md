# Stacked MAC (Memory As Context) Composition

```text
CONTRACT
  Purpose:    Spec for integrating MAC composition into the stacked multi-block
              architecture. MAC is the most expressive composition — memory provides
              context tokens for full causal attention, then reflects on the output.
              Two memory operations per step create a longer gradient path that
              requires explicit stability controls and motivates GPU-resident tape.
  Expects:    Stacked MAG path (specs 18-23) stable and validated.
              Existing single-block MAC (mac.rs) passing all tests.
              Titans Eqs 21-25 implemented in mac.rs forward/backward.
  Guarantees: N-block stacked MAC with per-block: read-only → assemble → full causal
              attention → extract → reflective step → sigmoid gate → W_O → residual.
              Gradient stability via symmetric forward/backward clamps.
              Same diagnostic visibility as MAG (TapeSummary, per-level grad norms).
  Cost:       Attention: O((N_p + 2s)^2 · d) per block — full causal, not sliding window.
              Memory: 2× per block (read-only + reflective step).
              Backward: full-depth gradient chain per block (13+ tape ops).
  Trade-off:  Most expressive composition (memory influences attention directly) at
              the cost of longer gradient paths and higher compute. The sequential
              data dependency (read-only → attention → reflective) prevents the
              parallel memory/attention overlap that MAG enjoys.
  Position:   specs/infrastructure/24_stacked_mac_composition.md
  Source:     Titans (2501.00663) Section 4.1, Eqs 21-25
              HOPE (2512.24695) Section 5.1, Eq 74 (CMS aggregation)
```

## Paper Equations

| Equation | Collection Key | Description |
|----------|---------------|-------------|
| Eq 21 | `titans_equations/eq-021-memory-retrieval-mac` | `h_t = M*_{t-1}(q_t)` — read-only retrieval, q_t = S^(i) W_Q |
| Eq 22 | `titans_equations/eq-022-mac-assembly` | `S_tilde = [p_1..p_{N_p} ‖ h_t ‖ S^(i)]` — assembled context |
| Eq 23 | `titans_equations/eq-023-mac-attention` | `y_t = Attn(S_tilde)` — full causal attention over assembled |
| Eq 24 | `titans_equations/eq-024-mac-memory-update` | `M_t = M_{t-1}(y_t)` — memory update via forward pass |
| Eq 25 | `titans_equations/eq-025-mac-output-gate` | `o_t = y_t ⊗ M*_t(y_t)` — reflective gate |
| Eq 74 | `hope_equations/eq-074-arch-variant5` | `y_t = Agg(...)` — learnable weighted sum for CMS levels |

## Architecture: Per-Block Data Flow

Each block in the N-block stack executes the following sequential pipeline:

```
block_input (from residual stream)
    │
    ├─ LN_attn ──→ [SWA attention branch — unchanged from MAG]
    │                    │
    │                    ▼
    │               attn_proj = attn_out @ W_O^T
    │
    ├─ LN_mem ──→ embedded (memory branch input)
    │                    │
    │                    ▼
    │              ┌─────────────────────────────────────────────┐
    │              │  Stage 1: Read-Only (Eq 21)                 │
    │              │  h_t = M_{t-1} @ (embedded @ W_Q_mem)       │
    │              │  Per CMS level, frozen M from context_memory │
    │              │                                              │
    │              │  Stage 2: Assembly (Eq 22)                   │
    │              │  assembled = [persistent ‖ h_t ‖ embedded]   │
    │              │  Length: N_p + 2s                             │
    │              │                                              │
    │              │  Stage 3: Full Causal Attention (Eq 23)      │
    │              │  QKV on assembled, full causal (ws ≥ N_p+2s) │
    │              │  y_t = attn_out[N_p+s:]  (extract segment)   │
    │              │                                              │
    │              │  Stage 4: Reflective Step (Eq 24)            │
    │              │  memory.step(y_t) → updates M, produces r_t  │
    │              │                                              │
    │              │  Stage 5: Reflective Gate (Eq 25)            │
    │              │  y_combined = y_t ⊗ sigmoid(r_t)             │
    │              └─────────────────────────────────────────────┘
    │                    │
    │                    ▼
    │              CMS level aggregation: softmax(alpha_mem) weighted sum
    │                    │
    │                    ▼
    │              MAG-style sigmoid gate: gate = σ(y_combined)
    │              gated_out = attn_proj * gate
    │
    └──────────────→ residual_out = block_input + gated_out
```

### Key Structural Differences from MAG

| Aspect | MAG (current) | MAC (this spec) |
|--------|--------------|-----------------|
| Memory branch | Single `memory.step(embedded)` | Read-only → assemble → inner attention → reflective step |
| Attention type | Sliding window (ws tokens) | Full causal over N_p + 2s tokens |
| Memory ops per block | 1 (active step) | 2 (read-only + reflective) |
| Memory branch contains attention | No | Yes — internal full causal attention on assembled sequence |
| Gradient chain depth | Short (memory → sigmoid) | Full network depth (read-only → QKV → attn → reflective → sigmoid) |
| Outer sigmoid gate | σ(y_combined) gates attn_proj | Same — σ(y_combined) gates attn_proj |
| Read-only path gating | N/A | **Ungated** — h_t enters assembly directly |

### What Stays the Same

The outer structure is identical to stacked MAG:
- Shared embeddings and lm_head across blocks
- Pre-norm LN_attn and LN_mem per block
- SWA attention branch with W_O projection (spec 18)
- Outer MAG sigmoid gating (spec 20)
- Learnable alpha_mem CMS aggregation (spec 21)
- Per-block gradient norms (spec 23)
- Residual stream connecting blocks

The MAC-specific logic is entirely **inside the memory branch**, replacing the single
`memory.step(embedded)` call with the 5-stage pipeline above.

## Gradient Stability Analysis

### The Problem: Two Backward Recurrences

MAC's backward path for memory parameters (W_K_mem, W_V_mem, W_Q_mem, gate weights)
accumulates gradient from **two chains** that both traverse full network depth:

**Chain 1 (reflective):**
```
d_loss → d_outer_sigmoid → d_reflective_gate → d_reflective_step.backward()
       → d_y_t → d_inner_attention → d_QKV → d_assembled → d_h_t
       → d_read_only.backward()
```

**Chain 2 (direct):**
```
d_loss → d_outer_sigmoid → d_y_combined → d_alpha_aggregation
       → d_level[l] → d_reflective_step.backward()  [gradient for M params]
```

The read-only parameters at the bottom of Chain 1 receive gradient that has been
multiplied by every operation in the stack. The reflective step's `d_M` accumulates
across `s` timesteps within its opaque backward, same as in MAG — but the upstream
gradient feeding into it is much larger because it passed through attention and QKV.

### Required Stability Controls

All clamps are applied **symmetrically** — same mechanism at the same point in both
forward and backward passes. This is the lesson from the MAG d_M explosion
(ethnographic note: `ethn_symmetry_inference_gap`).

#### 1. M-norm clamp (forward + backward, existing)
- **Forward**: clamp ‖M‖_F ≤ m_norm_max per level per timestep (already implemented)
- **Backward**: clamp ‖d_M‖_F ≤ m_norm_max per level per timestep (same budget)

#### 2. h_t norm clamp (NEW — MAC-specific)
The read-only output `h_t` enters the assembled sequence **ungated**. Unlike MAG where
memory output passes through sigmoid before touching the residual, MAC's h_t is
concatenated directly into the attention input. If M grows, h_t grows, attention
amplifies it.

- **Forward**: clamp ‖h_t‖ per-token ≤ h_norm_max before assembly
- **Backward**: clamp ‖d_h_t‖ per-token ≤ h_norm_max before read-only backward

Config: `"h_norm_max": 50.0` (per-token L2 norm, same scale as m_norm_max=100 but
tighter because h_t feeds directly into attention without sigmoid attenuation).

#### 3. Alpha/theta gate clamps (existing)
Same as MAG — `alpha_clamp` and `theta_clamp` bound the learned gate parameters
to prevent the sigmoid/softplus outputs from saturating or exploding.

#### 4. Per-block gradient clipping (existing, spec 23)
Independent gradient clipping per block prevents one block's explosion from starving
the other blocks' learning signal. Essential for MAC where the longer gradient chain
increases the variance between blocks.

### Why the Outer Sigmoid Gate Is Retained

The MAC architecture has an **inner** reflective gate (Eq 25: `y_t ⊗ sigmoid(r_t)`)
and the stacked architecture adds an **outer** MAG-style gate (spec 20:
`gated_out = attn_proj * sigmoid(y_combined)`). Both are retained:

- **Inner gate** (Eq 25): bounds the memory branch's self-assessment. Prevents the
  reflective step from amplifying the combined output.
- **Outer gate** (spec 20): bounds the memory branch's contribution to the residual
  stream. Same function as in MAG — prevents memory from overwhelming attention.

Double gating is intentional: inner bounds memory-on-memory, outer bounds memory-on-residual.

## Tape Implications

### Per-Block Tape Trace (13+ ops)

```
Op 0:  Opaque{FrozenDeltaRule/TitansLMM}  ← read-only: M @ q_mem → h_t
Op 1:  h_t_norm_clamp                      ← NEW: stability control
Op 2:  Concat{persistent, h_t, embedded}   ← assembly (N_p + 2s, d)
Op 3:  Matmul (W_Q_inner)                  ← inner attention QKV
Op 4:  Matmul (W_K_inner)
Op 5:  Matmul (W_V_inner)
Op 6:  Opaque{SWA_FullCausal}              ← full causal attention
Op 7:  Slice{(N_p+s)*d:}                   ← extract y_t
Op 8:  Opaque{DeltaRule/TitansLMM}         ← reflective step: memory.step(y_t)
Op 9:  Sigmoid                             ← inner reflective gate
Op 10: Mul{y_t, sigmoid}                   ← y_combined per level
       ... CMS aggregation ...
Op 11: Sigmoid                             ← outer MAG gate
Op 12: Mul{attn_proj, outer_gate}          ← gated_out
Op 13: Add{block_input, gated_out}         ← residual
```

Backward rewinds Op 13 → Op 0, with the gradient for the read-only parameters
requiring propagation through the entire chain.

### Why GPU-Resident Tape Matters for MAC

The backward pass for MAC's memory branch includes:
- Full causal attention backward over `(N_p + 2s)` tokens: O((N_p + 2s)^2 · d)
- Three QKV matmul backwards: 3 × O((N_p + 2s) · d^2)
- Two opaque memory backward passes (read-only + reflective)
- Concat/slice/sigmoid/mul elementwise ops

On CPU, this is dominated by the attention backward — a triple-nested loop over
(N_p + 2s)^2 positions. At s=512, N_p=8, that's 1032^2 ≈ 1M attention pairs per
block, times 4 blocks = 4M attention backward computations per segment, all on CPU.

With GPU-resident tape (task_40bb9d), the inner attention backward becomes a GPU
kernel on GPU1, the QKV matmuls become cublas calls, and only the thin dispatch
loop stays on CPU. This transforms MAC backward from "computationally infeasible
at scale on CPU" to "one kernel launch per op."

**MAC is the forcing function for task_40bb9d.** MAG's short memory branch makes
CPU tape tolerable. MAC's full-depth memory branch makes it a bottleneck.

## BlockParams Extensions for MAC

```rust
pub struct BlockParams {
    // ... existing MAG fields (w_q, w_k, w_v, w_o, LN, levels, alpha_mem) ...

    // MAC inner attention (separate from the outer SWA attention)
    pub w_q_inner: Vec<f32>,       // [d, d] — QKV for assembled-sequence attention
    pub w_k_inner: Vec<f32>,       // [d, d]
    pub w_v_inner: Vec<f32>,       // [d, d]

    // Persistent tokens per block (MAC Eq 22)
    pub persistent_tokens: Vec<f32>, // [N_p, d] — outer_loop_param

    // MAC reflective aggregation (separate from alpha_mem for CMS levels)
    // alpha_refl already exists in BlockParams — used here for reflective level weighting
}
```

### Design Decision: Inner Attention Parameters

MAC's assembled-sequence attention (Eq 23) requires its own QKV projections because:
- The **outer** SWA attention operates on `s` tokens from the residual stream
- The **inner** MAC attention operates on `N_p + 2s` tokens from the assembled sequence
- These see different data distributions (residual vs. memory-augmented)
- Sharing projections would conflate two different attention tasks

The outer SWA branch and inner MAC branch run **the same attention kernel** (both are
standard causal attention), but with different learned projections and different
sequence lengths.

## CMS Integration

Each CMS level `l` in a block performs:
1. Read-only from `context_memory[l]` → `h_t_l`
2. h_t norm clamp
3. Assemble with persistent + embedded
4. Inner attention (shared W_Q/K/V_inner across levels? TBD — see open question)
5. Reflective step on y_t_l
6. Reflective gate: y_t_l ⊗ sigmoid(r_t_l)

Level outputs are aggregated via `softmax(alpha_mem)` (spec 21), then passed through
the outer MAG sigmoid gate.

### Open Question: Per-Level vs Shared Inner Attention

Two options for the inner full-causal attention:
- **Shared** (one W_Q/K/V_inner per block): All levels share the same inner attention.
  Assembly differs per level (different h_t_l), but attention projections are shared.
  Lower param count, forces attention to generalize across levels.
- **Per-level** (one W_Q/K/V_inner per level per block): Each level has its own inner
  attention. Higher param count, allows level-specific attention patterns.

**Recommendation**: Start with **shared** inner attention. The levels already differ
in their memory content (h_t_l), CMS frequency, and retention dynamics. Adding per-level
attention projections is a multiplicative param increase (3 × d^2 × k per block) that
can be explored later if shared projections prove insufficient.

## Configuration

```json
{
  "composition": "MAC",
  "n_persistent": 8,
  "h_norm_max": 50.0,
  "mac_inner_attention": {
    "shared_qkv": true,
    "window_size_override": null
  }
}
```

- `composition: "MAC"` switches the memory branch from MAG's single-step to MAC's
  5-stage pipeline. Everything outside the memory branch is unchanged.
- `n_persistent`: Number of persistent tokens per block (outer_loop_param).
- `h_norm_max`: Per-token L2 norm clamp on read-only output h_t.
- `mac_inner_attention.shared_qkv`: Whether inner attention QKV is shared across
  CMS levels (true) or per-level (false).
- `mac_inner_attention.window_size_override`: If set, use this window size for inner
  attention instead of N_p + 2s (for testing with smaller windows).

## Implementation Sequence

This spec is designed to be implemented in phases, not all at once:

### Phase A: Single-Block Stacked MAC (prerequisite: MAG stable)
1. Add `w_q_inner`, `w_k_inner`, `w_v_inner`, `persistent_tokens` to BlockParams
2. Implement MAC memory branch in `gpu_stacked_forward.rs` behind `CompositionKind::MAC`
3. Implement MAC backward in `gpu_stacked_backward.rs`
4. Add h_t norm clamp (forward + backward)
5. Validate: single-block MAC matches `mac.rs` reference output

### Phase B: Multi-Block Stacked MAC
1. Wire N-block residual stream (same as MAG — outer structure unchanged)
2. Per-block gradient clipping (already exists from spec 23)
3. Validate: 4-block MAC trains without NaN for 1K steps

### Phase C: GPU-Resident Tape for MAC (depends on task_40bb9d)
1. TapeStorage enum with Device variant
2. alloc_device() for MAC's inner attention saved buffers
3. GPU1 backward worker for MAC's full-depth chain
4. Validate: same TapeSummary output, no PCIe copies for MAC backward

## Dependencies

| Dependency | Status | Why |
|-----------|--------|-----|
| Spec 18 (W_O projection) | Merged (#188) | Outer attn_proj feeds into gating |
| Spec 20 (MAG sigmoid gating) | Merged (#189) | Outer gate bounds memory contribution |
| Spec 21 (alpha aggregation) | Merged (#190) | CMS level weighting |
| Spec 23 (per-block grad norms) | Merged (#192) | Per-block clipping prevents cross-block contamination |
| MAG gate stability | In progress | Alpha/theta clamps needed for MAC gates too |
| task_40bb9d (GPU-resident tape) | Open | Phase C — MAC backward infeasible on CPU at scale |

## Acceptance Criteria

1. Stacked MAC forward produces same loss as single-block `mac.rs` for N=1
2. h_t norm clamp applied symmetrically in forward and backward
3. d_M norm clamp applied symmetrically (same as MAG fix)
4. 4-block MAC trains without NaN for 1K steps at d=512, k=1
5. TapeSummary reports per-level opaque blocks for both read-only and reflective ops
6. Per-block gradient norms (spec 23) work with MAC's longer gradient chain
7. No regressions in MAG stacked path — composition switch is config-only

## Ontological Compliance

- **CS-18**: All math (inner attention, assembly, gating) in Rust tier, not orchestration
- **CS-10**: No mode flag — MAC executes identically in all phases
- **CS-32**: Observe-then-advance — read-only observes M before reflective advances it
- **CS-40**: Opt-in AD — tape recording only when `with_tape()` active
- **CS-42**: All intermediates stored in arena — inner attention saved for backward
- **CS-47**: Parameter snapshots immune to mutation — inner W_Q/K/V cloned at recording
