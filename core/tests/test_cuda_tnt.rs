#![cfg(feature = "cuda")]
/// GPU TNT parity tests.
///
/// Verifies that gpu_cms_forward with TNT parallelization produces the same
/// loss as the Rust CPU reference (tnt_forward via cms_forward).
///
/// Requires: --features cuda

use nl_hecate_core::model::{
    MAGConfig, MAGParams, MemoryRuleKind, CompositionKind, HopeVariant,
    LatticeVariant, MomentumKind, ProjectionKind, FeatureMapKind,
};
use nl_hecate_core::conductor::{Pulse, ContextState};
use nl_hecate_core::dispatch;
use nl_hecate_core::mag::cms_forward;
use nl_hecate_core::gpu_forward::gpu_cms_forward;
use nl_hecate_core::gpu_params::{GpuMAGParams, GpuContextState};
use nl_hecate_core::parallel::{ParallelConfig, ParallelStrategy};
use serial_test::serial;

/// Create a TNT-enabled test config.
///
/// d=64, seq_len=32, C_G=16, C_L=4 → N=4 local memories per shard,
/// 2 shards total. Small enough for fast tests, large enough to exercise
/// the shard loop.
fn make_tnt_config(rule: MemoryRuleKind) -> MAGConfig {
    MAGConfig {
        swa: nl_hecate_core::model::SWAConfig {
            d_model: 64,
            num_heads: 4,
            head_dim: 16,
            seq_len: 32,
            window_size: 32,
            vocab_size: 256,
        },
        memory_enabled: true,
        composition: CompositionKind::MAG,
        memory_rule: rule,
        k: 1,
        chunk_sizes: vec![1],
        d_hidden: 0, lp_p: 2.0, sign_sharpness: 10.0, lq_q: 2.0,
        lambda_local: 0.0, lambda_2: 0.0,
        delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        parallel: Some(ParallelConfig {
            strategy: ParallelStrategy::TNTHierarchical,
            chunk_size: 4,
            tnt_global_chunk_size: 16,
            tnt_local_chunk_size: 4,
        }),
        retention: nl_hecate_core::retention::default_retention(rule),
        m3: None,
        frequency_schedule: nl_hecate_core::dynamic_freq::FrequencySchedule::Fixed,
        checkpoint_interval: None,
        tape_multiplier: None,
        hope_variant: HopeVariant::FreqGated,
        lattice_variant: LatticeVariant::Decode,
        n_persistent: 0,
        attentional_bias: Default::default(),
        kernel_size: 0,
        momentum_kind: MomentumKind::None,
        momentum_d_hidden: 0,
        projection_kind: ProjectionKind::Static,
        self_generated_values: false,
        self_ref_chunk_size: 1,
        alpha_floor: vec![],
        alpha_ceil: vec![],
        theta_floor: vec![],
        theta_ceil: vec![],
        intermediate_size: 0,
        m_norm_max: vec![],
        error_clip: vec![],
        feature_map: FeatureMapKind::Identity,
            residual: false,
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

/// Core parity test: GPU TNT forward loss ≈ CPU TNT forward loss.
fn test_tnt_forward_parity(rule: MemoryRuleKind) {
    let cfg = make_tnt_config(rule);
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    // CPU reference (force Rust path — uses tnt_forward internally)
    dispatch::force_rust_reference(true);
    let mut ctx_cpu = ContextState::new(cfg.k, cfg.swa.d_model);
    let (loss_cpu, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_cpu);
    dispatch::force_rust_reference(false);

    // GPU TNT path
    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx_gpu = GpuContextState::new(cfg.k, cfg.swa.d_model, 1, None, 0);
    let (loss_gpu, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_gpu);

    let diff = (loss_cpu - loss_gpu).abs();
    let rel = diff / loss_cpu.abs().max(1e-8);
    eprintln!("[TNT {:?}] CPU loss: {loss_cpu:.6}, GPU loss: {loss_gpu:.6}, diff: {diff:.6}, rel: {rel:.6}",
              rule);
    // TNT has more operations (shard loop, broadcast, summary) so allow 5% relative tolerance
    assert!(rel < 0.05,
        "TNT loss mismatch: CPU={loss_cpu}, GPU={loss_gpu}, rel={rel}");
}

#[test]
#[serial]
fn test_tnt_titans_forward_parity() {
    test_tnt_forward_parity(MemoryRuleKind::TitansLMM);
}

#[test]
#[serial]
fn test_tnt_delta_forward_parity() {
    test_tnt_forward_parity(MemoryRuleKind::DeltaRule);
}

/// Test that GPU TNT produces finite, positive loss.
#[test]
#[serial]
fn test_tnt_forward_finite() {
    let cfg = make_tnt_config(MemoryRuleKind::TitansLMM);
    let params = MAGParams::init(&cfg, 123);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model, 1, None, 0);
    let (loss, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);

    assert!(loss.is_finite(), "TNT GPU loss is not finite: {loss}");
    assert!(loss > 0.0, "TNT GPU loss should be positive: {loss}");
    eprintln!("TNT GPU loss: {loss:.6}");
}

/// Test GPU TNT backward: gradients exist and loss decreases over steps.
#[test]
#[serial]
fn test_tnt_backward_loss_decrease() {
    let cfg = make_tnt_config(MemoryRuleKind::TitansLMM);
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    let mut gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model, 1, None, 0);

    let mut losses = Vec::new();
    for _step in 0..5 {
        let (loss, cache) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);
        let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(&gpu_params, &cfg, &cache, false);
        nl_hecate_core::gpu_backward::gpu_weight_update(&mut gpu_params, &grads, 0.01);
        losses.push(loss);
    }

    eprintln!("TNT GPU multistep losses: {:?}", losses);
    // All losses finite
    for (i, l) in losses.iter().enumerate() {
        assert!(l.is_finite(), "TNT step {i} loss not finite: {l}");
    }
    // Loss should decrease
    assert!(losses[4] < losses[0],
        "TNT loss did not decrease: first={}, last={}", losses[0], losses[4]);
}

/// Test TNT with k=2 CMS levels — both levels use TNT sharding.
#[test]
#[serial]
fn test_tnt_k2_forward() {
    let mut cfg = make_tnt_config(MemoryRuleKind::TitansLMM);
    cfg.k = 2;
    cfg.chunk_sizes = vec![1, 8];
    // seq_len must be >= tnt_global_chunk_size, and both levels active
    cfg.swa.seq_len = 32;
    cfg.swa.window_size = 32;

    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model, 1, None, 0);
    let (loss, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);

    assert!(loss.is_finite(), "TNT k=2 GPU loss not finite: {loss}");
    assert!(loss > 0.0, "TNT k=2 GPU loss should be positive: {loss}");
    eprintln!("TNT k=2 GPU loss: {loss:.6}");
}
