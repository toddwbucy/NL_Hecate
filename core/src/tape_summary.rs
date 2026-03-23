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
use crate::tape::{Tape, with_tape, obs};
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
    /// Frobenius norm of M (or equivalent state) from the named `M_STATES`/
    /// `W1_STATES`/`S_STATES`/`SK_STATES` buffer. NaN if no named state buffer
    /// was found (pre-Phase 2 adapters).
    pub m_norm: f32,
    /// Sigmoid output of the learned frequency gate for this level.
    /// NaN if the schedule is Fixed (no learned gate).
    pub freq_gate_value: f32,
    /// True if this level used a Frozen* OpaqueKey variant (read-only M path).
    pub is_frozen: bool,
    /// Per-head Frobenius norms of the block-diagonal M sub-matrices.
    /// Length = num_heads for square-M rules (Delta, Titans, Hebbian, etc.).
    /// Empty for MLP-based rules (non-square M) or when num_heads <= 1.
    pub head_m_norms: Vec<f32>,
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

/// Compute per-head Frobenius norms from a block-diagonal M matrix.
///
/// M is `[d × d]` row-major where `d = num_heads * head_dim`. Head `h` owns
/// the diagonal sub-block at rows `[h*hd..(h+1)*hd]`, cols `[h*hd..(h+1)*hd]`.
/// Returns `Vec<f32>` of length `num_heads`, or empty if decomposition is
/// not applicable (non-square M, num_heads <= 1, d not divisible by num_heads).
fn per_head_m_norms(m: &[f32], d: usize, num_heads: usize) -> Vec<f32> {
    if num_heads <= 1 || d == 0 || m.len() != d * d || d % num_heads != 0 {
        return Vec::new();
    }
    let head_dim = d / num_heads;
    (0..num_heads).map(|h| {
        let row_start = h * head_dim;
        let col_start = h * head_dim;
        let mut sum_sq = 0.0f32;
        for r in 0..head_dim {
            let row_off = (row_start + r) * d + col_start;
            for c in 0..head_dim {
                let val = m[row_off + c];
                sum_sq += val * val;
            }
        }
        sum_sq.sqrt()
    }).collect()
}

/// Extract a single `LevelSummary` from the tape for level `lev`.
///
/// `final_memory`: the carried memory state from `context.memory[lev]` (or
/// `context[block][lev]` for stacked models) AFTER `traced_*_forward()` returns.
/// Used for `m_norm`. Pass `None` if unavailable → `m_norm = NaN`.
///
/// `gate_values`: per-level sigmoid outputs from learned frequency gate.
/// Pass `None` for Fixed schedule → `freq_gate_value = NaN`.
///
/// `num_heads`: number of attention heads for per-head M norm decomposition.
/// Pass 1 to skip per-head norms.
///
/// `d_model`: model dimension (d = num_heads * head_dim). Used to verify that
/// `final_memory` has the expected square-M layout (d*d elements) before
/// attempting per-head decomposition.
///
/// `has_square_m`: whether the memory rule stores M as a square [d×d] matrix.
/// MLP-based rules (Moneta/YAAD/MEMORA/SwiGluMlp) have non-square M —
/// per-head decomposition is skipped regardless of buffer size.
fn extract_level(
    tape: &Tape,
    all_blocks: &[(usize, OpaqueKey, Option<usize>, Option<usize>)],
    lev: usize,
    block_filter: Option<usize>,
    final_memory: Option<&[f32]>,
    gate_values: Option<&[f32]>,
    num_heads: usize,
    d_model: usize,
    has_square_m: bool,
) -> LevelSummary {
    let level_blocks: Vec<(usize, String)> = all_blocks.iter()
        .filter_map(|&(idx, key, lvl, blk)| {
            let level_match = lvl == Some(lev);
            let block_match = block_filter.map_or(true, |b| blk == Some(b));
            if level_match && block_match {
                Some((idx, format!("{:?}", key)))
            } else {
                None
            }
        })
        .collect();

    let block_count = level_blocks.len();

    let (opaque_key_str, output_grad_norm, dgd_delta_norm, is_frozen) =
        if let Some((last_idx, last_key)) = level_blocks.last() {
            let gnorm = tape.opaque_output_grad_norm(*last_idx).unwrap_or(0.0);
            let dgd_norm = tape
                .get_saved_by_role(*last_idx, obs::DGD_DELTA)
                .map(|buf| buf.iter().map(|x| x * x).sum::<f32>().sqrt())
                .unwrap_or(0.0);
            let frozen = last_key.starts_with("Frozen");
            (last_key.clone(), gnorm, dgd_norm, frozen)
        } else {
            (String::from("None"), 0.0, 0.0, false)
        };

    // m_norm: Frobenius norm of the final carried memory (from context, not
    // the full per-timestep trace stored on the tape).
    let m_norm = final_memory
        .map(|mem| mem.iter().map(|x| x * x).sum::<f32>().sqrt())
        .unwrap_or(f32::NAN);

    // Per-head M norms: decompose d×d M into num_heads diagonal blocks.
    // Requires: (1) rule has square M, (2) buffer is exactly d_model² elements.
    let head_m_norms = final_memory
        .filter(|mem| has_square_m && d_model > 0 && mem.len() == d_model * d_model)
        .map(|mem| per_head_m_norms(mem, d_model, num_heads))
        .unwrap_or_default();

    let freq_gate_value = gate_values
        .and_then(|gv| gv.get(lev).copied())
        .unwrap_or(f32::NAN);

    LevelSummary {
        level: lev,
        opaque_key: opaque_key_str,
        block_count,
        output_grad_norm,
        dgd_delta_norm,
        m_norm,
        freq_gate_value,
        is_frozen,
        head_m_norms,
    }
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
        let (loss, cache, loss_id, _param_ids) =
            traced_cms_forward(tape, params, cfg, input_ids, target_ids, pulse, context);
        tape.backward(loss_id);

        let all_blocks = tape.enumerate_opaque_blocks();
        let total_blocks = all_blocks.len();
        let gate_values = cache.freq_cache.as_ref().map(|fc| fc.gate_values.as_slice());

        let levels: Vec<LevelSummary> = (0..cfg.k)
            .map(|lev| {
                let final_mem = context.memory.get(lev).map(|v| v.as_slice());
                extract_level(tape, &all_blocks, lev, None, final_mem, gate_values,
                              cfg.swa.num_heads, cfg.swa.d_model,
                              cfg.memory_rule.has_square_m())
            })
            .collect();

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
    pub m_norm: f32,
    pub freq_gate_value: f32,
    pub is_frozen: bool,
    /// Per-head Frobenius norms (same semantics as `LevelSummary.head_m_norms`).
    pub head_m_norms: Vec<f32>,
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

        // Stacked models don't use learned frequency gates — always NaN.
        let mut blocks = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let levels: Vec<BlockLevelSummary> = (0..cfg.k).map(|lev| {
                let final_mem = context.get(b)
                    .and_then(|block_ctx| block_ctx.get(lev))
                    .map(|v| v.as_slice());
                let ls = extract_level(tape, &all_ops, lev, Some(b), final_mem, None,
                                       cfg.swa.num_heads, cfg.swa.d_model,
                                       cfg.memory_rule.has_square_m());
                BlockLevelSummary {
                    block: b,
                    level: ls.level,
                    opaque_key: ls.opaque_key,
                    block_count: ls.block_count,
                    output_grad_norm: ls.output_grad_norm,
                    dgd_delta_norm: ls.dgd_delta_norm,
                    m_norm: ls.m_norm,
                    freq_gate_value: ls.freq_gate_value,
                    is_frozen: ls.is_frozen,
                    head_m_norms: ls.head_m_norms,
                }
            }).collect();
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

    // ── Phase 2: Named buffer observation tests ──────────────────────
    //
    // Each test verifies that after a traced forward+backward pass,
    // get_saved_by_role() returns non-empty data for the adapter's key
    // saved buffers (spec 48 Gap A closure).

    use crate::model::MemoryRuleKind;
    use crate::tape::obs;

    /// Helper: create a MAGConfig with a specific memory rule for named buffer tests.
    fn test_config_for_rule(rule: MemoryRuleKind) -> MAGConfig {
        let mut cfg = MAGConfig::test_config();
        cfg.memory_rule = rule;
        cfg.retention = crate::retention::default_retention(rule);
        // MLP rules need d_hidden > 0
        match rule {
            MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
                cfg.d_hidden = cfg.swa.d_model;
            }
            MemoryRuleKind::LatticeOSR => {
                cfg.m_slots = 4;
            }
            MemoryRuleKind::Trellis => {
                cfg.d_compress = cfg.swa.d_model;
            }
            _ => {}
        }
        cfg
    }

    /// Run a traced forward+backward for a given rule and verify named buffers.
    fn assert_named_buffers(rule: MemoryRuleKind, expected_roles: &[&str]) {
        let cfg = test_config_for_rule(rule);
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = match rule {
            MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
                let d = cfg.swa.d_model;
                ContextState::new_with_memory_size(cfg.k, d, cfg.d_hidden * d + d * cfg.d_hidden)
            }
            MemoryRuleKind::LatticeOSR => {
                ContextState::new_with_memory_size(cfg.k, cfg.swa.d_model, cfg.m_slots * cfg.swa.d_model)
            }
            MemoryRuleKind::Trellis => {
                let d = cfg.swa.d_model;
                ContextState::new_with_memory_size(cfg.k, d, 2 * cfg.d_compress * d)
            }
            _ => ContextState::new(cfg.k, cfg.swa.d_model),
        };
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let registry = register_opaque_vjps();
        with_tape(registry, |tape| {
            let (_, _cache, loss_id, _param_ids) =
                traced_cms_forward(tape, &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);
            tape.backward(loss_id);

            // Find the memory rule opaque block(s) — exclude SWA
            let all_blocks = tape.enumerate_opaque_blocks();
            let rule_blocks: Vec<_> = all_blocks.iter()
                .filter(|&&(_, key, _, _)| {
                    !matches!(key, crate::tape::OpaqueKey::SWA | crate::tape::OpaqueKey::SwiGluMlp)
                })
                .collect();
            assert!(!rule_blocks.is_empty(),
                "{:?}: no memory rule opaque blocks found", rule);

            let &(op_idx, _, _, _) = rule_blocks[0];
            for &role in expected_roles {
                let data = tape.get_saved_by_role(op_idx, role);
                assert!(data.is_some(),
                    "{:?}: get_saved_by_role({}, {:?}) returned None", rule, op_idx, role);
                assert!(!data.unwrap().is_empty(),
                    "{:?}: get_saved_by_role({}, {:?}) returned empty slice", rule, op_idx, role);
            }
        });
    }

    #[test]
    fn test_named_buffers_delta_rule() {
        assert_named_buffers(MemoryRuleKind::DeltaRule,
            &[obs::M_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::DGD_DELTA]);
    }

    #[test]
    fn test_named_buffers_titans_lmm() {
        assert_named_buffers(MemoryRuleKind::TitansLMM,
            &[obs::M_STATES, obs::S_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::ERROR]);
    }

    #[test]
    fn test_named_buffers_hebbian() {
        assert_named_buffers(MemoryRuleKind::HebbianRule,
            &[obs::M_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA]);
    }

    #[test]
    fn test_named_buffers_moneta() {
        assert_named_buffers(MemoryRuleKind::Moneta,
            &[obs::W1_STATES, obs::W2_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::ERROR]);
    }

    #[test]
    fn test_named_buffers_yaad() {
        assert_named_buffers(MemoryRuleKind::YAAD,
            &[obs::W1_STATES, obs::W2_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::ERROR]);
    }

    #[test]
    fn test_named_buffers_memora() {
        assert_named_buffers(MemoryRuleKind::MEMORA,
            &[obs::W1_STATES, obs::W2_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::ERROR]);
    }

    #[test]
    fn test_named_buffers_lattice_osr() {
        assert_named_buffers(MemoryRuleKind::LatticeOSR,
            &[obs::S_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA]);
    }

    #[test]
    fn test_named_buffers_trellis() {
        assert_named_buffers(MemoryRuleKind::Trellis,
            &[obs::SK_STATES, obs::SV_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA, obs::ERROR_K, obs::ERROR_V]);
    }

    #[test]
    fn test_named_buffers_atlas_omega() {
        assert_named_buffers(MemoryRuleKind::AtlasOmega,
            &[obs::M_STATES, obs::S_STATES, obs::K_MEM, obs::V_MEM, obs::ALPHA, obs::THETA]);
    }

    // ── Phase 3 tests: new LevelSummary fields ──────────────────────

    #[test]
    fn test_m_norm_from_named_buffer() {
        // DeltaRule saves M_STATES via alloc_named → m_norm should be finite and > 0
        let cfg = MAGConfig::test_config(); // default = DeltaRule
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        // Level 0 always fires — should have a finite m_norm
        let lev0 = &summary.levels[0];
        assert!(lev0.m_norm.is_finite(), "m_norm should be finite, got {}", lev0.m_norm);
        assert!(lev0.m_norm > 0.0, "m_norm should be > 0 after a forward pass, got {}", lev0.m_norm);
    }

    #[test]
    fn test_frozen_level_flag() {
        // k=2 at step 1: level 1 (chunk_size=8) is frozen (does not fire).
        // The traced forward should record a Frozen* opaque key.
        let cfg = MAGConfig::test_config_k2();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());

        // Step 0: both levels fire → advance context so level 1 has nonzero M
        let pulse0 = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);
        let _ = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse0, &mut ctx,
        );

        // Advance conductor to step 1: level 1 should be frozen (8 does not divide 1)
        let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        conductor.advance();
        let pulse1 = conductor.pulse();

        // Level 1 should not be active at step 1
        assert!(!pulse1.active_levels[1],
            "Level 1 should be inactive at step 1 (chunk_size=8)");

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse1, &mut ctx,
        );
        let lev1 = &summary.levels[1];
        assert!(lev1.is_frozen, "Level 1 should be frozen at step 1");
    }

    #[test]
    fn test_freq_gate_nan_for_fixed_schedule() {
        // Fixed schedule (default): freq_gate_value should be NaN
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        for lev in &summary.levels {
            assert!(lev.freq_gate_value.is_nan(),
                "freq_gate_value should be NaN for Fixed schedule, got {} at level {}",
                lev.freq_gate_value, lev.level);
        }
    }

    // ── Phase 4: Per-head observation tests (spec 50) ────────────────

    #[test]
    fn test_head_norms_length_matches_num_heads() {
        // DeltaRule with d=64, num_heads=4 → head_m_norms.len() == 4
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        let lev0 = &summary.levels[0];
        assert_eq!(lev0.head_m_norms.len(), cfg.swa.num_heads,
            "head_m_norms length should be num_heads={}, got {}",
            cfg.swa.num_heads, lev0.head_m_norms.len());
    }

    #[test]
    fn test_head_norms_consistent_with_aggregate() {
        // Sum of squared head norms <= m_norm² (equality when cross-head blocks are zero)
        let cfg = MAGConfig::test_config();
        let params = MAGParams::init(&cfg, 42);
        let mut ctx = ContextState::new(cfg.k, cfg.swa.d_model);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        let lev0 = &summary.levels[0];
        let head_sq_sum: f32 = lev0.head_m_norms.iter().map(|n| n * n).sum();
        let agg_sq = lev0.m_norm * lev0.m_norm;
        assert!(head_sq_sum <= agg_sq + 1e-4,
            "sum(head_norm²) = {} should be <= m_norm² = {} (diff = {})",
            head_sq_sum, agg_sq, head_sq_sum - agg_sq);
    }

    #[test]
    fn test_head_norms_empty_for_mlp_rules() {
        // MLP rules (MONETA) have non-square M → head_m_norms should be empty
        let cfg = test_config_for_rule(MemoryRuleKind::Moneta);
        let params = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        let mut ctx = ContextState::new_with_memory_size(
            cfg.k, d, cfg.d_hidden * d + d * cfg.d_hidden);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let pulse = conductor.pulse();
        let (input_ids, target_ids) = make_ids(cfg.swa.seq_len, cfg.swa.vocab_size);

        let summary = extract_tape_summary(
            &params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx,
        );
        let lev0 = &summary.levels[0];
        assert!(lev0.head_m_norms.is_empty(),
            "MLP rules should have empty head_m_norms, got {:?}", lev0.head_m_norms);
    }

    #[test]
    fn test_head_norms_uniform_for_identity_m() {
        // If we seed M as scaled identity, all head norms should be equal
        let cfg = MAGConfig::test_config();
        let d = cfg.swa.d_model;
        let num_heads = cfg.swa.num_heads;
        let head_dim = cfg.swa.head_dim;

        // Build a scaled identity matrix
        let mut m = vec![0.0f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }

        let norms = per_head_m_norms(&m, d, num_heads);
        assert_eq!(norms.len(), num_heads);
        // Each head's diagonal block of identity has norm sqrt(head_dim)
        let expected = (head_dim as f32).sqrt();
        for (h, &n) in norms.iter().enumerate() {
            assert!((n - expected).abs() < 1e-5,
                "head {} norm = {}, expected {}", h, n, expected);
        }
    }

    #[test]
    fn test_per_head_m_norms_helper_edge_cases() {
        // num_heads=1 → empty (no decomposition needed)
        assert!(per_head_m_norms(&[1.0; 4], 2, 1).is_empty());
        // empty M → empty
        assert!(per_head_m_norms(&[], 0, 4).is_empty());
        // non-square M → empty
        assert!(per_head_m_norms(&[1.0; 6], 2, 2).is_empty());
    }

    #[test]
    fn test_head_norms_empty_for_non_square_rule_with_matching_len() {
        // Regression: a non-square rule whose buffer happens to have d_model²
        // elements must NOT get per-head norms. The has_square_m gate prevents
        // misinterpretation of non-square M buffers.
        use crate::model::MemoryRuleKind;
        assert!(!MemoryRuleKind::Moneta.has_square_m());
        assert!(!MemoryRuleKind::YAAD.has_square_m());
        assert!(!MemoryRuleKind::MEMORA.has_square_m());
        assert!(!MemoryRuleKind::SwiGluMlp.has_square_m());
        assert!(MemoryRuleKind::DeltaRule.has_square_m());
        assert!(MemoryRuleKind::TitansLMM.has_square_m());
        assert!(MemoryRuleKind::HebbianRule.has_square_m());
    }
}
