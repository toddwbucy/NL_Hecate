// Gradient checkpointing equivalence tests.
//
// Verifies that checkpointed forward+backward produces identical gradients
// to the full-trajectory path across all three matrix-based memory rules
// (Delta, Titans, Hebbian) and various checkpoint intervals.
//
// Requires: --features cuda
#![cfg(feature = "cuda")]

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind, HopeVariant, LatticeVariant};
use nl_hecate_core::conductor::Pulse;
use nl_hecate_core::gpu_forward::{gpu_cms_forward, checkpoint_count};
use nl_hecate_core::gpu_backward::gpu_cms_backward;
use nl_hecate_core::gpu_params::{GpuMAGParams, GpuContextState};
use serial_test::serial;

fn make_config(rule: MemoryRuleKind, checkpoint_interval: Option<usize>) -> MAGConfig {
    MAGConfig {
        swa: nl_hecate_core::model::SWAConfig {
            d_model: 16,
            num_heads: 2,
            head_dim: 8,
            seq_len: 32,
            window_size: 32,
            vocab_size: 64,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: rule,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0,
        delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        parallel: None,
        retention: nl_hecate_core::retention::default_retention(rule),
        m3: None,
        frequency_schedule: nl_hecate_core::dynamic_freq::FrequencySchedule::Fixed,
        checkpoint_interval,
            hope_variant: HopeVariant::FreqGated,
            lattice_variant: LatticeVariant::Decode,
    }
}

fn make_test_data(s: usize, v: usize) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
    let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
    (input_ids, target_ids)
}

fn single_level_pulse() -> Pulse {
    Pulse { global_step: 0, active_levels: vec![true] }
}

/// Run forward+backward with given config, return gradients downloaded to host.
fn run_fwd_bwd(cfg: &MAGConfig) -> (f32, nl_hecate_core::model::MAGParams) {
    let params = MAGParams::init(cfg, 42);
    let gpu_params = GpuMAGParams::from_host(&params, cfg);
    let mut context = GpuContextState::zeros(cfg);
    let pulse = single_level_pulse();
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);

    let (loss, cache) = gpu_cms_forward(&gpu_params, cfg, &input_ids, &target_ids, &pulse, &mut context);
    let grads = gpu_cms_backward(&gpu_params, cfg, &cache);
    let grads_host = grads.to_host(cfg);

    (loss, grads_host)
}

/// Compare two MAGParams gradient structs element-wise.
fn assert_grads_close(
    g1: &nl_hecate_core::model::MAGParams,
    g2: &nl_hecate_core::model::MAGParams,
    tol: f32,
    label: &str,
) {
    // SWA grads
    assert_vecs_close(&g1.swa.w_embed, &g2.swa.w_embed, tol, &format!("{label}/w_embed"));
    assert_vecs_close(&g1.swa.w_q, &g2.swa.w_q, tol, &format!("{label}/w_q"));
    assert_vecs_close(&g1.swa.w_k, &g2.swa.w_k, tol, &format!("{label}/w_k"));
    assert_vecs_close(&g1.swa.w_v, &g2.swa.w_v, tol, &format!("{label}/w_v"));
    assert_vecs_close(&g1.swa.w_o, &g2.swa.w_o, tol, &format!("{label}/w_o"));
    assert_vecs_close(&g1.swa.w_unembed, &g2.swa.w_unembed, tol, &format!("{label}/w_unembed"));

    // Per-level grads
    for (i, (l1, l2)) in g1.levels.iter().zip(g2.levels.iter()).enumerate() {
        assert_vecs_close(&l1.w_k_mem, &l2.w_k_mem, tol, &format!("{label}/level{i}/w_k_mem"));
        assert_vecs_close(&l1.w_v_mem, &l2.w_v_mem, tol, &format!("{label}/level{i}/w_v_mem"));
        assert_vecs_close(&l1.w_q_mem, &l2.w_q_mem, tol, &format!("{label}/level{i}/w_q_mem"));
    }
}

fn assert_vecs_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{label}: max diff {max_diff} at index {max_idx} (a={}, b={}), tol={tol}",
        a[max_idx], b[max_idx],
    );
}

// ══════════════════════════════════════════════════════════════════════
// Gradient equivalence tests: checkpointed vs full-trajectory
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_delta_ckpt_c4_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::DeltaRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::DeltaRule, Some(4));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Delta C=4 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Delta C=4");
}

#[test]
#[serial]
fn test_delta_ckpt_c8_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::DeltaRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::DeltaRule, Some(8));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Delta C=8 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Delta C=8");
}

#[test]
#[serial]
fn test_titans_ckpt_c4_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::TitansLMM, None);
    let cfg_ckpt = make_config(MemoryRuleKind::TitansLMM, Some(4));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Titans C=4 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Titans C=4");
}

#[test]
#[serial]
fn test_titans_ckpt_c8_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::TitansLMM, None);
    let cfg_ckpt = make_config(MemoryRuleKind::TitansLMM, Some(8));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Titans C=8 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Titans C=8");
}

#[test]
#[serial]
fn test_hebbian_ckpt_c4_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::HebbianRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::HebbianRule, Some(4));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Hebbian C=4 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Hebbian C=4");
}

#[test]
#[serial]
fn test_hebbian_ckpt_c8_matches_full() {
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::HebbianRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::HebbianRule, Some(8));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Hebbian C=8 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Hebbian C=8");
}

// ══════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_ckpt_c1_matches_full() {
    // C=1: every step checkpointed — should produce identical results
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::DeltaRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::DeltaRule, Some(1));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Delta C=1 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Delta C=1");
}

#[test]
#[serial]
fn test_ckpt_c_ge_seqlen_matches_full() {
    // C >= seq_len: single segment — should produce identical results
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::DeltaRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::DeltaRule, Some(1024)); // >> seq_len=32

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Delta C>=s loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Delta C>=s");
}

#[test]
#[serial]
fn test_ckpt_uneven_division() {
    // seq_len=32, C=5 → segments of [5,5,5,5,5,5,2] — last segment shorter
    nl_hecate_core::dispatch::cuda_init();
    let cfg_full = make_config(MemoryRuleKind::DeltaRule, None);
    let cfg_ckpt = make_config(MemoryRuleKind::DeltaRule, Some(5));

    let (loss_full, grads_full) = run_fwd_bwd(&cfg_full);
    let (loss_ckpt, grads_ckpt) = run_fwd_bwd(&cfg_ckpt);

    assert!((loss_full - loss_ckpt).abs() < 1e-5,
        "Delta C=5 loss mismatch: full={loss_full}, ckpt={loss_ckpt}");
    assert_grads_close(&grads_full, &grads_ckpt, 1e-5, "Delta C=5 uneven");
}

// ══════════════════════════════════════════════════════════════════════
// VRAM reduction verification (checkpoint buffer size)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_checkpoint_count() {
    // seq_len=32, C=8 → checkpoints at t=0, t=8, t=16, t=24, t=32 → 5
    assert_eq!(checkpoint_count(32, 8), 5);

    // seq_len=32, C=32 → checkpoints at t=0, t=32 → 2
    assert_eq!(checkpoint_count(32, 32), 2);

    // seq_len=32, C=1 → checkpoints at t=0,1,2,...,32 → 33
    assert_eq!(checkpoint_count(32, 1), 33);

    // seq_len=32, C=5 → t=0, then (t+1)%5==0: 5,10,15,20,25,30 → +6, then t+1=32 → +1
    // Total: 1 + 6 + 1 = 8
    assert_eq!(checkpoint_count(32, 5), 8);

    // seq_len=10, C=3 → t=0, then t+1 in {3,6,9,10} → 5
    assert_eq!(checkpoint_count(10, 3), 5);

    // seq_len=512, C=64 → t=0, then 64,128,192,256,320,384,448,512 → 9
    assert_eq!(checkpoint_count(512, 64), 9);
}

#[test]
fn test_checkpoint_vram_reduction() {
    // Full trajectory: (s+1) * d*d elements
    // Checkpointed: checkpoint_count(s, c) * d*d elements
    let s = 512;
    let d = 1024;
    let dd = d * d;
    let c = 64;

    let full_elements = (s + 1) * dd;
    let ckpt_elements = checkpoint_count(s, c) * dd;

    // Reduction ratio should be ~57x
    let ratio = full_elements as f64 / ckpt_elements as f64;
    assert!(ratio > 50.0, "Expected >50x reduction, got {ratio:.1}x");
    assert!(ratio < 65.0, "Unexpected ratio {ratio:.1}x");

    // Verify absolute sizes (in bytes, f32)
    let full_bytes = full_elements * 4;
    let ckpt_bytes = ckpt_elements * 4;
    assert_eq!(full_bytes, (512 + 1) * 1024 * 1024 * 4); // ~2.05 GB
    assert!(ckpt_bytes < 40_000_000); // < 40 MB
}
