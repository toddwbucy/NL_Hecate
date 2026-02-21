/// GPU-resident model parity tests.
///
/// Verifies that gpu_cms_forward produces the same loss as the Rust reference
/// cms_forward (with force_rust_reference). Tests the full data flow:
/// embedding → projections → SWA → memory → gating → output → loss.
///
/// Requires: --features cuda

use nl_hecate_core::model::{MAGConfig, MAGParams, MemoryRuleKind, CompositionKind, HopeVariant};
use nl_hecate_core::conductor::{Pulse, ContextState};
use nl_hecate_core::dispatch;
use nl_hecate_core::mag::cms_forward;
use nl_hecate_core::gpu_forward::gpu_cms_forward;
use nl_hecate_core::gpu_params::{GpuMAGParams, GpuContextState};
use serial_test::serial;

fn make_test_config(rule: MemoryRuleKind) -> MAGConfig {
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
        d_hidden: 0, lp_p: 2.0, lq_q: 2.0, lambda_local: 0.0, lambda_2: 0.0,
        delta: 1.0, m_slots: 0, d_compress: 0, lambda_k: 0.0, lambda_v: 0.0,
        parallel: None,
        retention: nl_hecate_core::retention::default_retention(rule),
        m3: None,
        frequency_schedule: nl_hecate_core::dynamic_freq::FrequencySchedule::Fixed,
        checkpoint_interval: None,
            hope_variant: HopeVariant::FreqGated,
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

/// Core parity test: GPU forward loss ≈ CPU reference forward loss.
fn test_forward_parity(rule: MemoryRuleKind) {
    let cfg = make_test_config(rule);
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    // CPU reference (force Rust path)
    dispatch::force_rust_reference(true);
    let mut ctx_cpu = ContextState::new(cfg.k, cfg.swa.d_model);
    let (loss_cpu, _) = cms_forward(&params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_cpu);
    dispatch::force_rust_reference(false);

    // GPU-resident
    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx_gpu = GpuContextState::new(cfg.k, cfg.swa.d_model);
    let (loss_gpu, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx_gpu);

    let diff = (loss_cpu - loss_gpu).abs();
    let rel = diff / loss_cpu.abs().max(1e-8);
    eprintln!("[{:?}] CPU loss: {loss_cpu:.6}, GPU loss: {loss_gpu:.6}, diff: {diff:.6}, rel: {rel:.6}",
              rule);
    // Tolerance: bf16 SWA introduces rounding, plus cross-entropy atomicAdd reordering
    assert!(rel < 0.05, "Loss mismatch too large: CPU={loss_cpu}, GPU={loss_gpu}, rel={rel}");
}

#[test]
#[serial]
fn test_delta_forward_parity() {
    test_forward_parity(MemoryRuleKind::DeltaRule);
}

#[test]
#[serial]
fn test_titans_forward_parity() {
    test_forward_parity(MemoryRuleKind::TitansLMM);
}

#[test]
#[serial]
fn test_hebbian_forward_parity() {
    test_forward_parity(MemoryRuleKind::HebbianRule);
}

/// Test multi-step forward: loss should decrease (model is learning).
#[test]
fn test_gpu_multistep_loss_decrease() {
    let cfg = make_test_config(MemoryRuleKind::DeltaRule);
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    let mut gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model);

    let mut losses = Vec::new();
    for _step in 0..5 {
        let (loss, cache) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);
        let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(&gpu_params, &cfg, &cache);
        nl_hecate_core::gpu_backward::gpu_weight_update(&mut gpu_params, &grads, 0.01);
        losses.push(loss);
    }

    eprintln!("GPU multistep losses: {:?}", losses);
    // Loss should generally decrease over 5 steps
    assert!(losses[4] < losses[0],
        "Loss did not decrease: first={}, last={}", losses[0], losses[4]);
}

/// Test CMS k=2: two frequency levels with one active, one frozen.
#[test]
fn test_gpu_k2_forward() {
    let mut cfg = make_test_config(MemoryRuleKind::DeltaRule);
    cfg.k = 2;
    cfg.chunk_sizes = vec![1, 8];
    cfg.swa.seq_len = 8;
    cfg.swa.window_size = 8;

    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);

    // Level 0 active, Level 1 also active (step 0 fires all levels)
    let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };

    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model);
    let (loss, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);

    assert!(loss.is_finite(), "k=2 GPU forward loss is not finite: {loss}");
    assert!(loss > 0.0, "k=2 GPU forward loss should be positive: {loss}");
    eprintln!("k=2 GPU loss: {loss:.6}");
}

/// Test checkpoint round-trip: GPU params → host → GPU → same loss.
#[test]
fn test_gpu_checkpoint_roundtrip() {
    let cfg = make_test_config(MemoryRuleKind::DeltaRule);
    let params = MAGParams::init(&cfg, 42);
    let (input_ids, target_ids) = make_test_data(cfg.swa.seq_len, cfg.swa.vocab_size);
    let pulse = single_level_pulse();

    let gpu_params = GpuMAGParams::from_host(&params);
    let mut ctx = GpuContextState::new(cfg.k, cfg.swa.d_model);
    let (loss1, _) = gpu_cms_forward(&gpu_params, &cfg, &input_ids, &target_ids, &pulse, &mut ctx);

    // Roundtrip: GPU → host → GPU
    let host_params = gpu_params.to_host(&cfg);
    let gpu_params2 = GpuMAGParams::from_host(&host_params);
    let mut ctx2 = GpuContextState::new(cfg.k, cfg.swa.d_model);
    let (loss2, _) = gpu_cms_forward(&gpu_params2, &cfg, &input_ids, &target_ids, &pulse, &mut ctx2);

    let diff = (loss1 - loss2).abs();
    eprintln!("Checkpoint roundtrip: loss1={loss1:.6}, loss2={loss2:.6}, diff={diff:.8}");
    assert!(diff < 1e-6, "Checkpoint roundtrip loss mismatch: {diff}");
}
