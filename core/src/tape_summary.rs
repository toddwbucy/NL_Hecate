/// Tape diagnostic summary — per-level opaque block inspection.
///
/// `extract_tape_summary()` runs one traced forward + backward pass and
/// queries the tape observation API to produce a lightweight `TapeSummary`.
/// It does NOT update weights or optimizer state — pure diagnostic read.
///
/// CS-40: tape activation is scoped to the duration of this call only.
/// CS-42: tape arena is dropped when the call returns.
/// CS-32: summary is read after backward, before any weight advance.

use crate::model::{MAGConfig, MAGParams};
use crate::conductor::{Pulse, ContextState};
#[allow(unused_imports)]
use crate::tape::OpaqueKey;
use crate::tape::{with_tape, obs};
use crate::traced_forward::traced_cms_forward;
use crate::opaque_adapters::register_opaque_vjps;

/// Per-level entry in a `TapeSummary`.
#[derive(Debug, Clone)]
pub struct LevelSummary {
    /// CMS level index (0 = fastest, k-1 = slowest).
    pub level: usize,
    /// Debug label of the OpaqueKey variant recorded at this level
    /// (e.g. "DeltaRule", "TitansLMM"). "None" if no block was recorded.
    pub opaque_key: String,
    /// Number of opaque blocks recorded at this level during the forward pass.
    /// 0 means the level did not fire this step (CMS frequency gate).
    pub block_count: usize,
    /// L2 norm of the accumulated gradient on the first output of the last
    /// opaque block at this level (post-backward).
    /// 0.0 if no gradient flowed through this level — the primary dormancy signal.
    pub output_grad_norm: f32,
    /// L2 norm of the `DGD_DELTA` saved buffer for the last opaque block at
    /// this level. 0.0 if no DGD_DELTA role is present (non-DGD rules).
    pub dgd_delta_norm: f32,
}

/// Lightweight diagnostic extracted from one traced forward+backward pass.
#[derive(Debug, Clone)]
pub struct TapeSummary {
    /// Cross-entropy loss for this batch (same value as a normal training step).
    pub loss: f32,
    /// Total opaque blocks recorded across all levels.
    pub total_blocks: usize,
    /// Per-level breakdown, length == cfg.k.
    pub levels: Vec<LevelSummary>,
}

/// Run one traced forward+backward on `params`/`context` and return a
/// `TapeSummary` with per-level opaque block diagnostics.
///
/// Does NOT update weights or optimizer state.
/// Safe to call from the Python eval path at `eval_every` intervals.
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
        // ── Forward: record every operation on the tape ───────────────────
        let (loss, _cache, loss_id, _param_ids) =
            traced_cms_forward(tape, params, cfg, input_ids, target_ids, pulse, context);

        // ── Backward: seed loss gradient = 1.0, replay in reverse ─────────
        tape.backward(loss_id);

        // ── Extract per-level data ────────────────────────────────────────
        let all_blocks = tape.enumerate_opaque_blocks();
        let total_blocks = all_blocks.len();

        let mut levels: Vec<LevelSummary> = Vec::with_capacity(cfg.k);
        for lev in 0..cfg.k {
            // Collect all blocks at this level in forward order.
            let level_blocks: Vec<(usize, String)> = all_blocks.iter()
                .filter_map(|&(idx, key, lvl, _block_idx)| {
                    if lvl == Some(lev) {
                        Some((idx, format!("{:?}", key)))
                    } else {
                        None
                    }
                })
                .collect();

            let block_count = level_blocks.len();

            let (opaque_key_str, output_grad_norm, dgd_delta_norm) =
                if let Some((last_idx, last_key)) = level_blocks.last() {
                    let gnorm = tape.opaque_output_grad_norm(*last_idx).unwrap_or(0.0);
                    let dgd_norm = tape
                        .get_saved_by_role(*last_idx, obs::DGD_DELTA)
                        .map(|buf| buf.iter().map(|x| x * x).sum::<f32>().sqrt())
                        .unwrap_or(0.0);
                    (last_key.clone(), gnorm, dgd_norm)
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

// ── Stacked Tape Summary ──────────────────────────────────────────────

use crate::stacked_model::StackedMAGParams;
use crate::traced_forward::traced_stacked_forward;

/// Per-(block, level) entry in a `StackedTapeSummary`.
#[derive(Debug, Clone)]
pub struct BlockLevelSummary {
    pub block: usize,
    pub level: usize,
    pub opaque_key: String,
    pub block_count: usize,
    pub output_grad_norm: f32,
    pub dgd_delta_norm: f32,
}

/// Per-block summary aggregated across levels.
#[derive(Debug, Clone)]
pub struct BlockTapeSummary {
    pub block: usize,
    pub levels: Vec<BlockLevelSummary>,
}

/// Lightweight diagnostic from one traced forward+backward on stacked model.
#[derive(Debug, Clone)]
pub struct StackedTapeSummary {
    pub loss: f32,
    pub total_blocks: usize,
    pub n_blocks: usize,
    pub blocks: Vec<BlockTapeSummary>,
}

/// Run one traced forward+backward on stacked `params`/`context` and return
/// a `StackedTapeSummary` with per-(block, level) opaque block diagnostics.
///
/// Does NOT update weights or optimizer state.
pub fn extract_stacked_tape_summary(
    params:     &StackedMAGParams,
    cfg:        &MAGConfig,
    input_ids:  &[usize],
    target_ids: &[usize],
    pulse:      &Pulse,
    context:    &mut Vec<Vec<Vec<f32>>>,
) -> StackedTapeSummary {
    let registry = register_opaque_vjps();
    let n_blocks = params.blocks.len();

    with_tape(registry, |tape| {
        let (loss, loss_id, _param_ids) =
            traced_stacked_forward(tape, params, cfg, input_ids, target_ids, pulse, context);

        tape.backward(loss_id);

        let all_ops = tape.enumerate_opaque_blocks();
        let total_blocks = all_ops.len();

        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let mut levels = Vec::with_capacity(cfg.k);
            for lev in 0..cfg.k {
                let level_blocks: Vec<(usize, String)> = all_ops.iter()
                    .filter_map(|&(idx, key, lvl, blk)| {
                        if lvl == Some(lev) && blk == Some(b) {
                            Some((idx, format!("{:?}", key)))
                        } else {
                            None
                        }
                    })
                    .collect();

                let block_count = level_blocks.len();
                let (opaque_key_str, output_grad_norm, dgd_delta_norm) =
                    if let Some((last_idx, last_key)) = level_blocks.last() {
                        let gnorm = tape.opaque_output_grad_norm(*last_idx).unwrap_or(0.0);
                        let dgd_norm = tape
                            .get_saved_by_role(*last_idx, obs::DGD_DELTA)
                            .map(|buf| buf.iter().map(|x| x * x).sum::<f32>().sqrt())
                            .unwrap_or(0.0);
                        (last_key.clone(), gnorm, dgd_norm)
                    } else {
                        (String::from("None"), 0.0, 0.0)
                    };

                levels.push(BlockLevelSummary {
                    block: b,
                    level: lev,
                    opaque_key: opaque_key_str,
                    block_count,
                    output_grad_norm,
                    dgd_delta_norm,
                });
            }
            blocks.push(BlockTapeSummary { block: b, levels });
        }

        StackedTapeSummary { loss, total_blocks, n_blocks, blocks }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conductor::{Conductor, ContextState};

    fn make_ids(seq_len: usize, vocab: usize) -> (Vec<usize>, Vec<usize>) {
        let input:  Vec<usize> = (0..seq_len).map(|i| (i * 3 + 1) % vocab).collect();
        let target: Vec<usize> = (0..seq_len).map(|i| (i * 3 + 2) % vocab).collect();
        (input, target)
    }

    #[test]
    fn test_tape_summary_level_count() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        assert_eq!(summary.levels.len(), cfg.k,
            "TapeSummary must have one LevelSummary per CMS level");
    }

    #[test]
    fn test_tape_summary_has_blocks() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        assert!(summary.total_blocks > 0,
            "At least one opaque block must be recorded during a forward pass");
    }

    #[test]
    fn test_tape_summary_active_level_has_nonzero_gnorm() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        // Step 0: all levels fire (every chunk_size divides step 0).
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        // Level 0 fires every step — must have nonzero output_grad_norm.
        let l0 = &summary.levels[0];
        assert!(l0.block_count > 0, "L0 must record blocks on every step");
        assert!(l0.output_grad_norm > 0.0,
            "L0 output_grad_norm must be > 0 after backward (got {})", l0.output_grad_norm);
    }

    #[test]
    fn test_tape_summary_does_not_update_params() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let w_q_before = params.swa.w_q.clone();
        extract_tape_summary(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);

        assert_eq!(params.swa.w_q, w_q_before,
            "extract_tape_summary must not modify params");
    }

    #[test]
    fn test_tape_summary_loss_is_finite() {
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        assert!(summary.loss.is_finite() && summary.loss > 0.0,
            "loss must be finite and positive (got {})", summary.loss);
    }
}
