# Gate Diagnostics: Alpha, Eta, M-Norm, and Clamp Rates

```text
CONTRACT
Purpose:    Expose the full learned gate distribution for all three inner-loop
            gates (alpha, theta, eta) in the GPU tape summary, plus memory norm
            and clamp hit rates. Currently only theta stats are visible. Alpha
            (retention/forgetting) and eta (momentum) are blind — the model
            has learned gates we cannot see. This spec adds per-(block, level)
            distribution statistics for alpha and eta following the existing
            theta_stats pattern, surfaces the already-computed m_norm, and
            reports clamp hit rates for CS-39 enforcement visibility.

Expects:    Existing GPU tape infrastructure:
            - ThetaStats struct with from_slice(values, ceil) → stats
            - theta_stats() method on GpuMemoryCache with D2H copy + TNT drilling
            - gpu_stacked_tape_summary in python/src/lib.rs wiring per-block + aggregated
            - print_tape_summary in python/engine/evaluation.py displaying theta line
            - Alpha buffer [s] on all variants except SwiGlu
            - Eta buffer [s] on Titans and TitansCkpt only
            - m_norm already computed and in tape dict (not printed)
            - CS-39 alpha_floor and theta_ceil config fields

Guarantees: 1. Alpha stats (mean, p99, max, frac_at_floor) visible per (block, level)
               for every memory rule that has a learned alpha gate.
            2. Eta stats (mean, p99, max) visible per (block, level) for Titans.
               Returns None for rules without momentum (Delta, Hebbian, DGD).
            3. m_norm (Frobenius norm of M) printed in tape summary per level.
            4. Clamp hit rates: frac_at_floor for alpha (CS-39 alpha_floor),
               frac_at_ceil for theta (already present).
            5. No new CUDA kernels. All data is already on GPU in existing
               cache buffers — only D2H copy + host-side statistics.
            6. Near-zero overhead: D2H copy of [seq_len] f32 per gate per level
               per block, only at tape_every intervals.

Cost:       One cudaMemcpy D2H per gate per level per block at tape_every steps.
            At seq_len=512, k=4, n_blocks=4: 48 copies × 2KB = 96KB total.
            Negligible vs the forward/backward compute.

Trade-off:  Alpha and eta stats use the same interpolated-quantile approach as
            theta. This is an approximation (per-block quantiles, not a true
            combined quantile across blocks). The aggregated view reports
            max(per-block p99) as p99_max, same as theta. For diagnostics
            this is sufficient — we want to catch pathological distributions,
            not compute exact percentiles.

Position:   specs/infrastructure/26_gate_diagnostics.md

Source:     Titans (2501.00663) eq-013 — M_t = (1 - α_t) M_{t-1} + S_t
            Alpha is the data-dependent forgetting gate: α_t = σ(w_α · [k_t, v_t] + b_α).
            α→0 preserves memory, α→1 clears it. Bounded [0,1] via sigmoid.
            With CS-39 alpha_floor=0.8, we need: is the model learning to stay
            within bounds naturally, or is the clamp doing all the work?

            Titans (2501.00663) eq-014 — S_t = η_t S_{t-1} - θ_t ∇ℓ(M; x_t)
            Eta is the data-dependent momentum gate: η_t = σ(w_η · [k_t, v_t] + b_η).
            η→0 kills momentum (pure GD), η→1 full momentum carry. Only present
            in Titans LMM (the only rule with momentum accumulator S).

            CS-39 — Clamp learnable decay. alpha_floor prevents catastrophic
            forgetting. Visibility into clamp hit rates tells us whether the
            floor is a safety net (rarely hit) or a load-bearing constraint
            (frequently hit, model wants lower alpha but can't go there).
```

## Design

### Generalized GateStats Struct

Rather than separate AlphaStats, EtaStats, and ThetaStats structs with near-identical
fields, introduce a single `GateStats` struct parameterized by bound type:

```rust
#[derive(Debug, Clone)]
pub struct GateStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
    /// Fraction of tokens at the configured bound (floor or ceil).
    /// For alpha: frac at alpha_floor (CS-39 lower bound).
    /// For theta: frac at theta_ceil (CS-39 upper bound).
    /// For eta: 0.0 (no configured bound).
    pub frac_at_bound: f32,
}
```

`GateStats::from_slice(values, bound, bound_is_floor)` computes stats identically
to the existing `ThetaStats::from_slice` but with directional bound checking:
- `bound_is_floor=true` (alpha): counts values ≤ bound × (1 + ε)
- `bound_is_floor=false` (theta): counts values ≥ bound × (1 - ε)
- `bound=f32::MAX` or `bound=0.0`: frac_at_bound = 0.0 (no bound configured)

ThetaStats is retained as a type alias or left unchanged to avoid churn in
existing code. New alpha/eta methods return GateStats directly.

### Methods on GpuMemoryCache

```rust
impl GpuMemoryCache {
    /// Alpha (retention/forgetting) gate statistics.
    /// Available for: Delta, Titans, Hebbian, DGD (+ Ckpt variants).
    /// Returns None for SwiGlu.
    pub fn alpha_stats(&self, alpha_floor: f32) -> Option<GateStats> { ... }

    /// Eta (momentum) gate statistics.
    /// Available for: Titans, TitansCkpt only.
    /// Returns None for all other rules.
    pub fn eta_stats(&self) -> Option<GateStats> { ... }
}
```

Both follow the same pattern as `theta_stats()`:
1. Match on enum variants to extract the buffer reference
2. D2H copy via `buf.copy_to_host(&mut host)`
3. For TNT: drill into `shard_inner_caches`, concatenate, filter zero-padding
4. Call `GateStats::from_slice()`

### Alpha gate semantics

Alpha = σ(w_α · [k_t, v_t] + b_α), bounded [0,1] via sigmoid.

**IMPORTANT**: In our implementation, alpha represents the *retention* factor,
not the forgetting factor. The memory update is:
```
M_t = α_t · M_{t-1} + S_t
```
So α→1 means full retention (preserve memory), α→0 means full forgetting.
The paper's eq-013 uses `(1 - α_t)` as the retention coefficient, but our
code flips the convention. The alpha_floor=0.8 in CS-39 means "retain at
least 80% of memory" — preventing catastrophic forgetting.

The diagnostic question: what fraction of tokens hit the floor (α = alpha_floor)?
If frac_at_floor is high, the model wants to forget more aggressively than CS-39
allows. If low, the model is naturally conservative and the floor is a safety net.

### Eta gate semantics

Eta = σ(w_η · [k_t, v_t] + b_η), bounded [0,1] via sigmoid.

Controls momentum accumulation: `S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M; x_t)`.
η→0 kills momentum (pure GD per-token), η→1 carries full momentum across tokens.
No configured floor or ceiling — the sigmoid bounds [0,1] are the natural range.
No clamp hit rate needed.

### M-norm display

`m_norm` is already in the per-level tape dict (computed by `memory_norms()` in
python/src/lib.rs). `print_tape_summary` simply needs to read and display it.

### Clamp hit rates

Already covered by frac_at_bound:
- Alpha: `frac_at_floor` using `alpha_floor` from config
- Theta: `frac_at_ceil` using `theta_ceil` from config (already implemented)

## Files to Modify

| File | Change |
|------|--------|
| `core/src/gpu_forward.rs` | Add `GateStats` struct, `alpha_stats()` and `eta_stats()` methods on `GpuMemoryCache` |
| `python/src/lib.rs` | Wire alpha/eta stats into `gpu_stacked_tape_summary` dict, add to aggregated levels |
| `python/engine/evaluation.py` | `print_tape_summary`: add m_norm to level line, add alpha/eta stat lines |

## Tape Summary Output Format

Current:
```
[tape] step=1024  loss=4.86  total_blocks=4
  L0 [TitansLMM]  blocks=4  out_gnorm=6.3e-03  dgd_delta=6.3e+01
         θ  mean=0.5782  p99=1.0000  max=1.0000  @ceil=42.7%
```

After:
```
[tape] step=1024  loss=4.86  total_blocks=4
  L0 [TitansLMM]  blocks=4  out_gnorm=6.3e-03  dgd_delta=6.3e+01  m_norm=45.2
         α  mean=0.9512  p99=0.9998  max=1.0000  @floor=3.2%
         θ  mean=0.5782  p99=1.0000  max=1.0000  @ceil=42.7%
         η  mean=0.4231  p99=0.8912  max=0.9543
```

- Alpha line: `@floor=X%` shows CS-39 clamp hit rate
- Theta line: unchanged (`@ceil=X%` already present)
- Eta line: no bound indicator (sigmoid-bounded naturally)
- Eta line only appears for Titans (rules without momentum skip it)
- m_norm appended to the main level line

## Acceptance Criteria

1. `alpha_stats()` returns `GateStats` for Delta, Titans, Hebbian, DGD (+ Ckpt + TNT)
2. `eta_stats()` returns `GateStats` for Titans, TitansCkpt (+ TNT with Titans inner)
3. Both return `None` for SwiGlu and rules without the respective gate
4. `m_norm` printed on the level line in `print_tape_summary`
5. `frac_at_floor` for alpha uses `alpha_floor` from per-level config
6. GPU tape summary dict includes `"alpha"` and `"eta"` keys per level
7. Aggregated levels include alpha/eta stats (same max-p99 pattern as theta)
8. No new CUDA kernels — pure D2H copy + host-side statistics
9. No regressions in existing tests

## Ontological Compliance

- **CS-10**: No mode flag — gate stats computed identically in all phases.
- **CS-18**: Statistics computation is math in the Rust tier.
- **CS-32**: Observe-only — stats are read from existing buffers, no mutation.
- **CS-39**: This spec provides *visibility* into CS-39 enforcement, not enforcement itself.
- **CS-40**: Stats only extracted when tape is active (tape_every intervals).

## Equations Traced

| Equation | Collection | Source | Relationship |
|----------|-----------|--------|-------------|
| eq-013-forgetting-gate | titans_equations | Titans §3.2 | cites |
| eq-014-momentum-with-forgetting | titans_equations | Titans §3.2 | cites |
| eq-018-momentum-linear-recurrence | titans_equations | Titans §3.3 | cites |
