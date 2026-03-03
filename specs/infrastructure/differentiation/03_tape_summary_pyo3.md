# Tape Summary PyO3 Binding

```text
CONTRACT
  Purpose:    Expose Wengert tape per-level diagnostics to the Python evaluation
              path via a single `tape_forward_summary()` method on PyGpuModel.
              Runs one traced forward pass (no optimizer step, no weight update),
              queries the tape observation API, and returns a lightweight
              TapeSummary struct that Python can log alongside standard eval
              metrics. Primary use: L2/L3 dormancy diagnosis during ablation runs.

  Expects:    - core/src/tape.rs — Wengert tape with full observation API:
                  enumerate_opaque_blocks(), find_opaque_at_level(),
                  opaque_output_grad_norm(), get_saved_by_role()
              - core/src/gradient.rs — tape_compute_gradients() with
                  traced_cms_forward() + tape.backward() wired up
              - core/src/tape.rs obs::DGD_DELTA constant
              - python/src/lib.rs — PyGpuModel struct with existing
                  step_adamw() and level_grad_norms() PyO3 methods
              - CS-40: tape activation is opt-in — tape_forward_summary()
                  activates it only for the duration of its call

  Guarantees: - TapeSummary returned as a Python dict with schema:
                  {
                    "loss": float,
                    "total_blocks": int,
                    "levels": [
                      {
                        "level": int,
                        "opaque_key": str,          # e.g. "DeltaRule"
                        "block_count": int,         # opaque blocks at this level
                        "output_grad_norm": float,  # post-backward, 0.0 if no flow
                        "dgd_delta_norm": float,    # L2 norm of DGD_DELTA saved buf,
                                                    # 0.0 if not present
                      },
                      ...
                    ]
                  }
              - Runs one traced forward + backward; does NOT update weights or
                  optimizer state (read-only diagnostic)
              - Callable from eval path at eval_every intervals
              - No hot-path impact: gate is in Python (only called at eval steps)

  Cost:       - One additional forward+backward per eval call (~same cost as
                  a normal training step but without AdamW update)
              - Tape arena allocation proportional to sequence length and k
              - PCIe traffic: host params and context are transferred GPU→CPU
                  via to_host() before extraction (one-way, read-only copy)

  Trade-off:  - eval_every overhead doubles for steps where tape_forward_summary
                  is called (one GPU forward for loss eval + one traced forward
                  for tape diagnostic). Acceptable for diagnostic runs; can be
                  gated by a separate `tape_every` config key.
              - Tape does not survive the call — summary is extracted and tape
                  is dropped. This is correct: CS-42 prohibits persistent tape.

  Position:   specs/infrastructure/differentiation/03_tape_summary_pyo3.md
              Implements: tape observation API (02_tape_observation.md)
              Used by:    engine/evaluation.py (print_level_metrics extension)
              Blocks:     Dolmino ablation suite (task_e34a74)

  Source:     CS-32 (observe-then-advance) — summary is read after backward,
                before any weight advance
              CS-40 (opt-in tape) — with_tape() only active inside this call
              CS-42 (no gradient checkpointing) — tape arena dropped at call end
              CS-18 (orchestration in Python tier) — summary logging lives in
                evaluation.py, not in Rust
```

---

## 1. Rust: `TapeSummary` Struct

Defined in `core/src/tape_summary.rs` (new file):

```rust
/// Per-level entry in a TapeSummary.
#[derive(Debug, Clone)]
pub struct LevelSummary {
    pub level: usize,
    /// String label for the OpaqueKey variant (e.g. "DeltaRule").
    pub opaque_key: String,
    /// Number of opaque blocks recorded at this level during the forward pass.
    pub block_count: usize,
    /// L2 norm of the first output's accumulated gradient for the last
    /// opaque block at this level (post-backward). 0.0 if no gradient flowed.
    pub output_grad_norm: f32,
    /// L2 norm of the DGD_DELTA saved buffer for the last opaque block at
    /// this level. 0.0 if no DGD_DELTA role present (e.g. non-DGD rules).
    pub dgd_delta_norm: f32,
}

/// Lightweight diagnostic extracted from one traced forward+backward pass.
/// Returned by `tape_forward_summary()` in python/src/lib.rs.
#[derive(Debug, Clone)]
pub struct TapeSummary {
    pub loss: f32,
    pub total_blocks: usize,
    pub levels: Vec<LevelSummary>,
}
```

### Extraction function

Also in `core/src/tape_summary.rs`:

```rust
/// Run one traced forward+backward on `params`/`context` and extract
/// a TapeSummary. Does NOT update weights or optimizer state.
///
/// Reads: enumerate_opaque_blocks(), find_opaque_at_level(),
///        opaque_output_grad_norm(), get_saved_by_role(obs::DGD_DELTA)
pub fn extract_tape_summary(
    params:     &MAGParams,
    cfg:        &MAGConfig,
    input_ids:  &[usize],
    target_ids: &[usize],
    pulse:      &Pulse,
    context:    &mut ContextState,
) -> TapeSummary {
    let registry = register_opaque_vjps();

    with_tape(registry, |tape| {
        // ── Forward ──────────────────────────────────────────────────────
        let (loss, _cache, loss_id, _param_ids) =
            traced_cms_forward(tape, params, cfg, input_ids, target_ids, pulse, context);

        // ── Backward (seed loss gradient = 1.0) ──────────────────────────
        tape.backward(loss_id);

        // ── Extract per-level data ────────────────────────────────────────
        let all_blocks = tape.enumerate_opaque_blocks();
        let total_blocks = all_blocks.len();

        let mut levels: Vec<LevelSummary> = Vec::with_capacity(cfg.k);
        for lev in 0..cfg.k {
            // Find all opaque blocks at this level (any OpaqueKey).
            // Active memory rules register with their OpaqueKey + level.
            // Use the last recorded block per level (most recent segment).
            let level_blocks: Vec<(usize, OpaqueKey)> = all_blocks.iter()
                .filter_map(|&(idx, key, lvl)| {
                    if lvl == Some(lev) { Some((idx, key)) } else { None }
                })
                .collect();

            let block_count = level_blocks.len();

            let (opaque_key_str, output_grad_norm, dgd_delta_norm) =
                if let Some(&(last_idx, last_key)) = level_blocks.last() {
                    let gnorm = tape.opaque_output_grad_norm(last_idx).unwrap_or(0.0);
                    let dgd_norm = tape
                        .get_saved_by_role(last_idx, obs::DGD_DELTA)
                        .map(|buf| buf.iter().map(|x| x * x).sum::<f32>().sqrt())
                        .unwrap_or(0.0);
                    (format!("{:?}", last_key), gnorm, dgd_norm)
                } else {
                    (String::from("None"), 0.0, 0.0)
                };

            levels.push(LevelSummary {
                level: lev,
                opaque_key: opaque_key_str,
                block_count,
                output_grad_norm,
                dgd_delta_norm,
            });
        }

        TapeSummary { loss, total_blocks, levels }
    })
}
```

---

## 2. PyO3 Binding

In `python/src/lib.rs`, on `impl PyGpuModel`:

```rust
/// Run one traced forward+backward and return per-level tape diagnostics.
///
/// Does NOT update weights or optimizer state — pure diagnostic read.
/// Returns a Python dict matching the TapeSummary schema.
///
/// Call at eval_every intervals only. Hot-path cost is ~1 training step
/// (one traced forward + backward, no AdamW update).
///
/// CS-40: tape activation is scoped to this call only.
/// CS-42: tape arena is dropped when this call returns.
/// CS-32: summary is read after backward, before any weight advance.
fn tape_forward_summary(
    &mut self,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    pulse: &Pulse,
    py: Python<'_>,
) -> PyResult<PyObject> {
    let summary = nl_hecate_core::tape_summary::extract_tape_summary(
        &self.params,
        &self.cfg,
        &input_ids,
        &target_ids,
        &pulse.inner,
        &mut self.context,
    );

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("loss", summary.loss)?;
    dict.set_item("total_blocks", summary.total_blocks)?;

    let levels_list = pyo3::types::PyList::empty(py);
    for lvl in &summary.levels {
        let ldict = pyo3::types::PyDict::new(py);
        ldict.set_item("level", lvl.level)?;
        ldict.set_item("opaque_key", &lvl.opaque_key)?;
        ldict.set_item("block_count", lvl.block_count)?;
        ldict.set_item("output_grad_norm", lvl.output_grad_norm)?;
        ldict.set_item("dgd_delta_norm", lvl.dgd_delta_norm)?;
        levels_list.append(ldict)?;
    }
    dict.set_item("levels", levels_list)?;

    Ok(dict.into())
}
```

---

## 3. Python Eval Hook

In `engine/evaluation.py`, extend `print_level_metrics()`:

```python
def print_tape_summary(tape_summary: dict, step: int) -> None:
    """Log per-level tape diagnostics from tape_forward_summary().

    Printed at eval_every intervals alongside standard level metrics.
    Each level line shows: block count, output grad norm, DGD delta norm.
    Zero output_grad_norm at L2/L3 means no gradient flowed through those
    levels — the primary signal for the L2/L3 dormancy diagnostic.
    """
    print(f"  Tape summary (step {step}, loss={tape_summary['loss']:.4f}, "
          f"total_blocks={tape_summary['total_blocks']}):")
    for lvl in tape_summary["levels"]:
        print(
            f"    L{lvl['level']} [{lvl['opaque_key']}]: "
            f"blocks={lvl['block_count']}  "
            f"out_gnorm={lvl['output_grad_norm']:.4e}  "
            f"dgd_delta={lvl['dgd_delta_norm']:.4e}"
        )
```

Call site in `loop.py` at eval steps:

```python
if (bcfg.eval_every > 0 and step > 0 and step % bcfg.eval_every == 0
        and gpu_model is not None):
    tape_sum = gpu_model.tape_forward_summary(
        input_ids, target_ids, pulse
    )
    evaluation.print_tape_summary(tape_sum, step)
    if jsonl:
        log_fields["tape_summary"] = tape_sum
```

---

## 4. Diagnostic Interpretation Guide

| L2/L3 `block_count` | `output_grad_norm` | `dgd_delta_norm` | Interpretation |
|---|---|---|---|
| 0 | — | — | Level never fired (CMS frequency gate: step not a multiple of chunk_size[l]) |
| > 0 | ≈ 0 | > 0 | Gradient not flowing back through level — initialization trap (Hypothesis B) |
| > 0 | > 0 | > 0 | Level active and receiving gradient — functioning correctly |
| > 0 | > 0 | ≈ 0 | Active, gradient flowing, but DGD delta is near zero — collapsed inner loop |

The dormancy signal seen in `ablation_C_tnt` (`gnorm_l=[0.2, 0.0, 0.0, 0.0]`) would
appear here as L1/L2/L3 all having `output_grad_norm ≈ 0` even when `block_count > 0`,
directly confirming Hypothesis B (gradient not flowing through higher levels) vs.
the levels never being activated at all.

---

## 5. Test Classes

```rust
// Guarantee: tape_forward_summary returns a summary with cfg.k level entries
fn test_tape_summary_level_count()

// Guarantee: total_blocks > 0 after a valid forward pass
fn test_tape_summary_has_blocks()

// Numerical: output_grad_norm > 0 for active levels after backward
fn test_tape_summary_active_level_has_nonzero_gnorm()

// Guarantee: extract_tape_summary does not modify params (read-only)
fn test_tape_summary_does_not_update_params()

// Guarantee: loss is finite and positive
fn test_tape_summary_loss_is_finite()
```

---

## 6. HADES Registration

```json
{
  "_key": "tape-summary-pyo3",
  "title": "Tape Summary PyO3 Binding",
  "category": "infrastructure",
  "version": "0.4.0",
  "path": "specs/infrastructure/differentiation/03_tape_summary_pyo3.md",
  "purpose": "Expose Wengert tape per-level opaque block diagnostics to Python eval path for L2/L3 dormancy diagnosis",
  "paper_source": ["2512.24695"],
  "traced_to_equations": [],
  "traced_to_axioms": ["hope_axioms/CS-32", "hope_axioms/CS-40", "hope_axioms/CS-42"],
  "depends_on_specs": ["hecate_specs/tape-observation", "hecate_specs/wengert_tape"],
  "status": "v0.4.0"
}
```
