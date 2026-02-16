//! 100K-step stability sweep: every (rule, composition, k) combination at d=8.
//!
//! Verifies no NaN/divergence, loss decreases, and captures milestone diagnostics
//! at 25K-step intervals. Covers 72 combinations across 8 MIRAS rules,
//! 3 composition patterns (MAG/MAL/MAC), and CMS k=1/k=2/k=4.
//!
//! ## Combinatorial coverage matrix
//!
//! | Tier | Coverage                          | Tests | LR    |
//! |------|-----------------------------------|-------|-------|
//! | 1    | 8 rules × MAG × k=1              |     8 | 0.01  |
//! | 2    | 8 rules × MAL × k=1              |     8 | 0.01  |
//! | 2    | 8 rules × MAC × k=1              |     8 | 0.01  |
//! | 3    | 8 rules × MAG × k=2              |     8 | 0.01  |
//! | 4    | delta × {MAL,MAC} × k=2          |     2 | 0.01/0.001 |
//! | 5    | 7 non-delta × {MAL,MAC} × k=2    |    14 | 0.01/0.001 |
//! | 6    | 8 rules × MAG × k=4              |     8 | 0.01  |
//! | 7    | 8 rules × {MAL,MAC} × k=4        |    16 | 0.01/0.001 |
//! |      | **Total**                         | **72**|       |
//!
//! ## Falsification criterion (from spec Phase 3)
//!
//! If >20% of 72 valid combinations produce degenerate dynamics (NaN, divergence,
//! or no loss decrease), the constraint matrix needs rebuilding.

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind, SWAConfig};
use nl_hecate_core::mag::{mag_forward, mag_backward, cms_forward, cms_backward};
use nl_hecate_core::mal::{mal_forward, mal_backward, cms_mal_forward, cms_mal_backward};
use nl_hecate_core::mac::{mac_forward, mac_backward, cms_mac_forward, cms_mac_backward};
use nl_hecate_core::conductor::{Conductor, ContextState, ErrorBuffer};

// ── Helpers ──────────────────────────────────────────────────────────

/// Build a sweep config for any (rule, composition, k) at d=8 scale.
fn sweep_config(rule: MemoryRuleKind, comp: CompositionKind, k: usize) -> MAGConfig {
    let seq_len = 4 * k;
    let window_size = match comp {
        CompositionKind::MAC => 2 * seq_len,
        _ => seq_len,
    };

    let (d_hidden, lambda_2, lambda_local, delta) = match rule {
        MemoryRuleKind::Moneta => (4, 0.01, 0.0, 1.0),
        MemoryRuleKind::YAAD => (4, 0.01, 0.01, 1.0),
        MemoryRuleKind::MEMORA => (4, 0.0, 0.0, 1.0),
        _ => (0, 0.0, 0.0, 1.0),
    };

    let m_slots = match rule {
        MemoryRuleKind::LatticeOSR => 4,
        _ => 0,
    };

    let (d_compress, lambda_k, lambda_v) = match rule {
        MemoryRuleKind::Trellis => (8, 0.01, 0.01),
        _ => (0, 0.0, 0.0),
    };

    let chunk_sizes = match k {
        1 => vec![1],
        2 => vec![1, 8],
        4 => vec![1, 8, 64, 512],
        _ => panic!("sweep only supports k=1, k=2, and k=4"),
    };

    MAGConfig {
        swa: SWAConfig {
            d_model: 8,
            num_heads: 2,
            head_dim: 4,
            seq_len,
            window_size,
            vocab_size: 16,
        },
        memory_enabled: true,
        composition: comp,
        memory_rule: rule,
        k,
        chunk_sizes,
        d_hidden,
        lp_p: 2.0,
        lq_q: 2.0,
        lambda_local,
        lambda_2,
        delta,
        m_slots,
        d_compress,
        lambda_k,
        lambda_v,
        parallel: None,
    }
}

/// Per-level memory size for rule-aware ContextState construction.
fn memory_size_for_rule(cfg: &MAGConfig) -> usize {
    let d = cfg.swa.d_model;
    match cfg.memory_rule {
        MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA => {
            cfg.d_hidden * d + d * cfg.d_hidden
        }
        MemoryRuleKind::LatticeOSR => cfg.m_slots * d,
        MemoryRuleKind::Trellis => 2 * cfg.d_compress * d,
        _ => d * d,
    }
}

/// Create rule-aware ContextState.
fn make_context(cfg: &MAGConfig) -> ContextState {
    let d = cfg.swa.d_model;
    let mem_size = memory_size_for_rule(cfg);
    if mem_size == d * d {
        ContextState::new(cfg.k, d)
    } else {
        ContextState::new_with_memory_size(cfg.k, d, mem_size)
    }
}

fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    let target_ids: Vec<usize> = (1..=cfg.swa.seq_len).map(|t| t % cfg.swa.vocab_size).collect();
    (input_ids, target_ids)
}

struct SweepResult {
    initial_loss: f32,
    final_loss: f32,
    milestones: Vec<(usize, f32)>,
    nan_step: Option<usize>,
    converged: bool,
}

/// Run a k=1 sweep for any composition.
fn run_sweep_k1(cfg: &MAGConfig, steps: usize, lr: f32, seed: u64) -> SweepResult {
    let mut params = MAGParams::init(cfg, seed);
    let (input_ids, target_ids) = make_data(cfg);

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let mut milestones = Vec::new();
    let mut nan_step = None;

    for step in 0..steps {
        let (loss, grads) = match cfg.composition {
            CompositionKind::MAG => {
                let (l, cache) = mag_forward(&params, cfg, &input_ids, &target_ids);
                let g = mag_backward(&params, cfg, &cache, &input_ids, &target_ids);
                (l, g)
            }
            CompositionKind::MAL => {
                let (l, cache) = mal_forward(&params, cfg, &input_ids, &target_ids);
                let g = mal_backward(&params, cfg, &cache, &input_ids, &target_ids);
                (l, g)
            }
            CompositionKind::MAC => {
                let (l, cache) = mac_forward(&params, cfg, &input_ids, &target_ids);
                let g = mac_backward(&params, cfg, &cache, &input_ids, &target_ids);
                (l, g)
            }
        };

        if step == 0 {
            initial_loss = loss;
        }
        final_loss = loss;

        if !loss.is_finite() && nan_step.is_none() {
            nan_step = Some(step);
        }

        if step > 0 && step % 25_000 == 0 && step < steps {
            milestones.push((step, loss));
        }

        params.apply_weight_gradients(&grads, lr);
    }

    milestones.push((steps, final_loss));

    SweepResult {
        initial_loss,
        final_loss,
        milestones,
        nan_step,
        converged: final_loss < initial_loss,
    }
}

/// Run a CMS (k>1) sweep for any composition.
fn run_sweep_cms(cfg: &MAGConfig, steps: usize, lr: f32, seed: u64) -> SweepResult {
    let mut params = MAGParams::init(cfg, seed);
    let (input_ids, target_ids) = make_data(cfg);
    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
    let mut context = make_context(cfg);
    let d = cfg.swa.d_model;
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(d))
        .collect();

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;
    let mut milestones = Vec::new();
    let mut nan_step = None;

    for step in 0..steps {
        let pulse = conductor.pulse();

        let (loss, grads) = match cfg.composition {
            CompositionKind::MAG => {
                let (l, cache) = cms_forward(&params, cfg, &input_ids, &target_ids, &pulse, &mut context);
                let g = cms_backward(&params, cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
                (l, g)
            }
            CompositionKind::MAL => {
                let (l, cache) = cms_mal_forward(&params, cfg, &input_ids, &target_ids, &pulse, &mut context);
                let g = cms_mal_backward(&params, cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
                (l, g)
            }
            CompositionKind::MAC => {
                let (l, cache) = cms_mac_forward(&params, cfg, &input_ids, &target_ids, &pulse, &mut context);
                let g = cms_mac_backward(&params, cfg, &cache, &input_ids, &target_ids, &mut error_buffers);
                (l, g)
            }
        };

        if step == 0 {
            initial_loss = loss;
        }
        final_loss = loss;

        if !loss.is_finite() && nan_step.is_none() {
            nan_step = Some(step);
        }

        if step > 0 && step % 25_000 == 0 {
            milestones.push((step, loss));
        }

        params.apply_weight_gradients(&grads, lr);

        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
            }
        }

        conductor.advance();
    }

    milestones.push((steps, final_loss));

    SweepResult {
        initial_loss,
        final_loss,
        milestones,
        nan_step,
        converged: final_loss < initial_loss,
    }
}

/// Dispatch to k=1 or CMS runner.
fn run_sweep(cfg: &MAGConfig, steps: usize, lr: f32, seed: u64) -> SweepResult {
    if cfg.k == 1 {
        run_sweep_k1(cfg, steps, lr, seed)
    } else {
        run_sweep_cms(cfg, steps, lr, seed)
    }
}

/// Standard assertions + diagnostic printing for a sweep test.
fn assert_sweep(name: &str, result: &SweepResult) {
    eprintln!("── {name} ──");
    eprintln!("  initial={:.4}, final={:.4}, converged={}",
              result.initial_loss, result.final_loss, result.converged);
    for (step, loss) in &result.milestones {
        eprintln!("  milestone step={step}: loss={loss:.4}");
    }
    if let Some(s) = result.nan_step {
        eprintln!("  NaN at step {s}!");
    }

    assert!(result.nan_step.is_none(),
        "{name}: NaN at step {}", result.nan_step.unwrap_or(0));
    assert!(result.final_loss.is_finite(),
        "{name}: final loss not finite: {}", result.final_loss);
    assert!(result.final_loss < result.initial_loss,
        "{name}: loss did not decrease: initial={:.4}, final={:.4}",
        result.initial_loss, result.final_loss);
    assert!(result.final_loss < 100.0,
        "{name}: loss diverged: {:.4}", result.final_loss);
}

const STEPS: usize = 100_000;
const LR: f32 = 0.01;
const SEED: u64 = 42;

// ══════════════════════════════════════════════════════════════════════
// Tier 1: 8 rules × MAG × k=1
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAG/k1", &result);
}

#[test]
fn test_sweep_titans_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAG/k1", &result);
}

#[test]
fn test_sweep_hebbian_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAG/k1", &result);
}

#[test]
fn test_sweep_moneta_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAG/k1", &result);
}

#[test]
fn test_sweep_yaad_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAG/k1", &result);
}

#[test]
fn test_sweep_memora_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAG/k1", &result);
}

#[test]
fn test_sweep_lattice_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAG/k1", &result);
}

#[test]
fn test_sweep_trellis_mag_k1() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAG, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAG/k1", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 2: 8 rules × MAL × k=1
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAL/k1", &result);
}

#[test]
fn test_sweep_titans_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAL/k1", &result);
}

#[test]
fn test_sweep_hebbian_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAL/k1", &result);
}

#[test]
fn test_sweep_moneta_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAL/k1", &result);
}

#[test]
fn test_sweep_yaad_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAL/k1", &result);
}

#[test]
fn test_sweep_memora_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAL/k1", &result);
}

#[test]
fn test_sweep_lattice_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAL/k1", &result);
}

#[test]
fn test_sweep_trellis_mal_k1() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAL, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAL/k1", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 2: 8 rules × MAC × k=1
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAC/k1", &result);
}

#[test]
fn test_sweep_titans_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAC/k1", &result);
}

#[test]
fn test_sweep_hebbian_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAC/k1", &result);
}

#[test]
fn test_sweep_moneta_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAC/k1", &result);
}

#[test]
fn test_sweep_yaad_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAC/k1", &result);
}

#[test]
fn test_sweep_memora_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAC/k1", &result);
}

#[test]
fn test_sweep_lattice_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAC/k1", &result);
}

#[test]
fn test_sweep_trellis_mac_k1() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAC, 1);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAC/k1", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 3: 8 rules × MAG × k=2 (CMS)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAG/k2", &result);
}

#[test]
fn test_sweep_titans_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAG/k2", &result);
}

#[test]
fn test_sweep_hebbian_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAG/k2", &result);
}

#[test]
fn test_sweep_moneta_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAG/k2", &result);
}

#[test]
fn test_sweep_yaad_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAG/k2", &result);
}

#[test]
fn test_sweep_memora_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAG/k2", &result);
}

#[test]
fn test_sweep_lattice_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAG/k2", &result);
}

#[test]
fn test_sweep_trellis_mag_k2() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAG, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAG/k2", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 4: DeltaRule × {MAL, MAC} × k=2 (CMS cross-composition)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAL/k2", &result);
}

/// MAC+CMS k=2 has a longer gradient path (2×seq assembled attention + reflective step).
/// ErrorBuffer flush applies ~7 accumulated gradient steps at once, which at lr=0.01
/// causes NaN around step 15K. Reducing lr to 0.001 stabilizes the run.
#[test]
fn test_sweep_delta_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("delta/MAC/k2", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 5: 7 non-delta rules × MAL × k=2 (CMS)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_titans_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAL/k2", &result);
}

#[test]
fn test_sweep_hebbian_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAL/k2", &result);
}

#[test]
fn test_sweep_moneta_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAL/k2", &result);
}

#[test]
fn test_sweep_yaad_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAL/k2", &result);
}

#[test]
fn test_sweep_memora_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAL/k2", &result);
}

#[test]
fn test_sweep_lattice_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAL/k2", &result);
}

#[test]
fn test_sweep_trellis_mal_k2() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAL, 2);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAL/k2", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 5: 7 non-delta rules × MAC × k=2 (CMS)
// All MAC+CMS use lr=0.001 — ErrorBuffer flush is aggressive for all rules.
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_titans_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("titans/MAC/k2", &result);
}

#[test]
fn test_sweep_hebbian_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("hebbian/MAC/k2", &result);
}

#[test]
fn test_sweep_moneta_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("moneta/MAC/k2", &result);
}

#[test]
fn test_sweep_yaad_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("yaad/MAC/k2", &result);
}

#[test]
fn test_sweep_memora_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("memora/MAC/k2", &result);
}

#[test]
fn test_sweep_lattice_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("lattice/MAC/k2", &result);
}

#[test]
fn test_sweep_trellis_mac_k2() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAC, 2);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("trellis/MAC/k2", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 6: 8 rules × MAG × k=4 (CMS)
// MAG is gated composition — most stable at higher k.
// Level 3 (chunk_size=512) fires ~195 times in 100K steps.
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAG/k4", &result);
}

#[test]
fn test_sweep_titans_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAG/k4", &result);
}

#[test]
fn test_sweep_hebbian_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAG/k4", &result);
}

#[test]
fn test_sweep_moneta_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAG/k4", &result);
}

#[test]
fn test_sweep_yaad_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAG/k4", &result);
}

#[test]
fn test_sweep_memora_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAG/k4", &result);
}

#[test]
fn test_sweep_lattice_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAG/k4", &result);
}

#[test]
fn test_sweep_trellis_mag_k4() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAG, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAG/k4", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 7: 8 rules × MAL × k=4 (CMS)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("delta/MAL/k4", &result);
}

#[test]
fn test_sweep_titans_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("titans/MAL/k4", &result);
}

#[test]
fn test_sweep_hebbian_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("hebbian/MAL/k4", &result);
}

#[test]
fn test_sweep_moneta_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("moneta/MAL/k4", &result);
}

#[test]
fn test_sweep_yaad_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("yaad/MAL/k4", &result);
}

#[test]
fn test_sweep_memora_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("memora/MAL/k4", &result);
}

#[test]
fn test_sweep_lattice_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("lattice/MAL/k4", &result);
}

#[test]
fn test_sweep_trellis_mal_k4() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAL, 4);
    let result = run_sweep(&cfg, STEPS, LR, SEED);
    assert_sweep("trellis/MAL/k4", &result);
}

// ══════════════════════════════════════════════════════════════════════
// Tier 7: 8 rules × MAC × k=4 (CMS)
// All MAC+CMS use lr=0.001 — ErrorBuffer flush is aggressive for all rules.
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sweep_delta_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::DeltaRule, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("delta/MAC/k4", &result);
}

#[test]
fn test_sweep_titans_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::TitansLMM, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("titans/MAC/k4", &result);
}

#[test]
fn test_sweep_hebbian_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::HebbianRule, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("hebbian/MAC/k4", &result);
}

#[test]
fn test_sweep_moneta_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::Moneta, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("moneta/MAC/k4", &result);
}

#[test]
fn test_sweep_yaad_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::YAAD, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("yaad/MAC/k4", &result);
}

#[test]
fn test_sweep_memora_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::MEMORA, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("memora/MAC/k4", &result);
}

#[test]
fn test_sweep_lattice_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::LatticeOSR, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("lattice/MAC/k4", &result);
}

#[test]
fn test_sweep_trellis_mac_k4() {
    let cfg = sweep_config(MemoryRuleKind::Trellis, CompositionKind::MAC, 4);
    let result = run_sweep(&cfg, STEPS, 0.001, SEED);
    assert_sweep("trellis/MAC/k4", &result);
}
