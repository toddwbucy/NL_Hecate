/// S4-M1: Weight serialization roundtrip tests.
///
/// Verifies that all param/config structs survive JSON serialization
/// and that save_checkpoint/load_checkpoint produce identical results.

use nl_hecate_core::model::{
    MAGConfig, MAGParams, SWAConfig, SWAParams,
    save_checkpoint, load_checkpoint,
};
use nl_hecate_core::parallel::{ParallelConfig, ParallelStrategy};
use nl_hecate_core::m3::M3Config;
use nl_hecate_core::dynamic_freq::FrequencySchedule;

// ── Helpers ──────────────────────────────────────────────────────────

fn assert_swa_params_eq(a: &SWAParams, b: &SWAParams) {
    assert_eq!(a.w_embed, b.w_embed, "w_embed mismatch");
    assert_eq!(a.w_q, b.w_q, "w_q mismatch");
    assert_eq!(a.w_k, b.w_k, "w_k mismatch");
    assert_eq!(a.w_v, b.w_v, "w_v mismatch");
    assert_eq!(a.w_o, b.w_o, "w_o mismatch");
    assert_eq!(a.w_unembed, b.w_unembed, "w_unembed mismatch");
}

fn assert_mag_params_eq(a: &MAGParams, b: &MAGParams) {
    assert_swa_params_eq(&a.swa, &b.swa);
    assert_eq!(a.levels.len(), b.levels.len(), "levels count mismatch");
    for (i, (la, lb)) in a.levels.iter().zip(b.levels.iter()).enumerate() {
        assert_eq!(la.w_k_mem, lb.w_k_mem, "level[{i}].w_k_mem mismatch");
        assert_eq!(la.w_v_mem, lb.w_v_mem, "level[{i}].w_v_mem mismatch");
        assert_eq!(la.w_q_mem, lb.w_q_mem, "level[{i}].w_q_mem mismatch");
        assert_eq!(la.w_alpha, lb.w_alpha, "level[{i}].w_alpha mismatch");
        assert_eq!(la.b_alpha, lb.b_alpha, "level[{i}].b_alpha mismatch");
        assert_eq!(la.w_theta, lb.w_theta, "level[{i}].w_theta mismatch");
        assert_eq!(la.b_theta, lb.b_theta, "level[{i}].b_theta mismatch");
        assert_eq!(la.w_eta, lb.w_eta, "level[{i}].w_eta mismatch");
        assert_eq!(la.b_eta, lb.b_eta, "level[{i}].b_eta mismatch");
        assert_eq!(la.w_omega, lb.w_omega, "level[{i}].w_omega mismatch");
        assert_eq!(la.w_freq, lb.w_freq, "level[{i}].w_freq mismatch");
        assert_eq!(la.b_freq, lb.b_freq, "level[{i}].b_freq mismatch");
    }
}

fn assert_mag_config_eq(a: &MAGConfig, b: &MAGConfig) {
    assert_eq!(a.swa.d_model, b.swa.d_model);
    assert_eq!(a.swa.num_heads, b.swa.num_heads);
    assert_eq!(a.swa.head_dim, b.swa.head_dim);
    assert_eq!(a.swa.seq_len, b.swa.seq_len);
    assert_eq!(a.swa.window_size, b.swa.window_size);
    assert_eq!(a.swa.vocab_size, b.swa.vocab_size);
    assert_eq!(a.memory_enabled, b.memory_enabled);
    assert_eq!(a.composition, b.composition);
    assert_eq!(a.memory_rule, b.memory_rule);
    assert_eq!(a.k, b.k);
    assert_eq!(a.chunk_sizes, b.chunk_sizes);
    assert_eq!(a.d_hidden, b.d_hidden);
    assert_eq!(a.lp_p, b.lp_p);
    assert_eq!(a.lq_q, b.lq_q);
    assert_eq!(a.lambda_local, b.lambda_local);
    assert_eq!(a.lambda_2, b.lambda_2);
    assert_eq!(a.delta, b.delta);
    assert_eq!(a.m_slots, b.m_slots);
    assert_eq!(a.d_compress, b.d_compress);
    assert_eq!(a.lambda_k, b.lambda_k);
    assert_eq!(a.lambda_v, b.lambda_v);
    assert_eq!(a.retention, b.retention);
    assert_eq!(a.frequency_schedule, b.frequency_schedule);
    // parallel: compare strategy/chunk_size if present
    match (&a.parallel, &b.parallel) {
        (None, None) => {},
        (Some(pa), Some(pb)) => {
            assert_eq!(pa.strategy, pb.strategy);
            assert_eq!(pa.chunk_size, pb.chunk_size);
            assert_eq!(pa.tnt_global_chunk_size, pb.tnt_global_chunk_size);
            assert_eq!(pa.tnt_local_chunk_size, pb.tnt_local_chunk_size);
        },
        _ => panic!("parallel config mismatch: one is None, other is Some"),
    }
    // m3: compare if present
    match (&a.m3, &b.m3) {
        (None, None) => {},
        (Some(ma), Some(mb)) => {
            assert_eq!(ma.k, mb.k);
            assert_eq!(ma.etas, mb.etas);
            assert_eq!(ma.thetas, mb.thetas);
            assert_eq!(ma.weights, mb.weights);
            assert_eq!(ma.frequencies, mb.frequencies);
            assert_eq!(ma.use_newton_schulz, mb.use_newton_schulz);
            assert_eq!(ma.ns_iterations, mb.ns_iterations);
            assert_eq!(ma.ns_dim, mb.ns_dim);
        },
        _ => panic!("m3 config mismatch: one is None, other is Some"),
    }
}

// ── SWAParams roundtrip ──────────────────────────────────────────────

#[test]
fn test_swa_params_roundtrip() {
    let cfg = SWAConfig::test_config();
    let params = SWAParams::init(&cfg, 42);
    let json = serde_json::to_string(&params).unwrap();
    let restored: SWAParams = serde_json::from_str(&json).unwrap();
    assert_swa_params_eq(&params, &restored);
}

// ── MAGParams roundtrip (k=1) ────────────────────────────────────────

#[test]
fn test_mag_params_roundtrip() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let json = serde_json::to_string(&params).unwrap();
    let restored: MAGParams = serde_json::from_str(&json).unwrap();
    assert_mag_params_eq(&params, &restored);
}

// ── MAGParams roundtrip (k=4) ────────────────────────────────────────

#[test]
fn test_mag_params_k4_roundtrip() {
    let cfg = MAGConfig::test_config_k4();
    let params = MAGParams::init(&cfg, 99);
    let json = serde_json::to_string(&params).unwrap();
    let restored: MAGParams = serde_json::from_str(&json).unwrap();
    assert_mag_params_eq(&params, &restored);
    assert_eq!(restored.levels.len(), 4);
    // Verify gate biases survive roundtrip
    assert!((restored.levels[0].b_alpha[0] - 3.0).abs() < 1e-6);
    assert!((restored.levels[3].b_alpha[0] - 5.0).abs() < 1e-6);
    assert!((restored.levels[3].b_theta[0] - (-7.6)).abs() < 1e-6);
}

// ── MAGConfig roundtrip ──────────────────────────────────────────────

#[test]
fn test_mag_config_roundtrip() {
    let cfg = MAGConfig::test_config();
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: MAGConfig = serde_json::from_str(&json).unwrap();
    assert_mag_config_eq(&cfg, &restored);
}

// ── Config with Learned frequency schedule ───────────────────────────

#[test]
fn test_config_with_learned_freq() {
    let cfg = MAGConfig::dynamic_freq_test_config();
    assert!(matches!(cfg.frequency_schedule, FrequencySchedule::Learned(_)));
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: MAGConfig = serde_json::from_str(&json).unwrap();
    assert_mag_config_eq(&cfg, &restored);
    assert!(matches!(restored.frequency_schedule, FrequencySchedule::Learned(_)));
    if let FrequencySchedule::Learned(lfc) = &restored.frequency_schedule {
        assert_eq!(lfc.threshold, 0.5);
        assert_eq!(lfc.anneal_steps, 0);
    }
}

// ── Config with M3 ───────────────────────────────────────────────────

#[test]
fn test_config_with_m3() {
    let mut cfg = MAGConfig::test_config();
    cfg.m3 = Some(M3Config::default_k2());
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: MAGConfig = serde_json::from_str(&json).unwrap();
    assert_mag_config_eq(&cfg, &restored);
    assert!(restored.m3.is_some());
    let m3 = restored.m3.unwrap();
    assert_eq!(m3.k, 2);
    assert_eq!(m3.frequencies, vec![1, 8]);
}

// ── Config with ParallelConfig ───────────────────────────────────────

#[test]
fn test_config_with_parallel() {
    let mut cfg = MAGConfig::test_config();
    cfg.parallel = Some(ParallelConfig::chunkwise(4));
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: MAGConfig = serde_json::from_str(&json).unwrap();
    assert_mag_config_eq(&cfg, &restored);
    let p = restored.parallel.unwrap();
    assert_eq!(p.strategy, ParallelStrategy::ChunkwiseGD);
    assert_eq!(p.chunk_size, 4);
}

// ── File I/O roundtrip ───────────────────────────────────────────────

#[test]
fn test_save_load_checkpoint() {
    let dir = std::env::temp_dir().join("nl_hecate_test_checkpoint");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_ckpt.json");

    let cfg = MAGConfig::test_config_k2();
    let params = MAGParams::init(&cfg, 42);

    save_checkpoint(&path, &params, &cfg).unwrap();
    let (loaded_params, loaded_cfg, _build_state) = load_checkpoint(&path).unwrap();

    assert_mag_params_eq(&params, &loaded_params);
    assert_mag_config_eq(&cfg, &loaded_cfg);

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ── Missing file error ───────────────────────────────────────────────

#[test]
fn test_load_nonexistent_file() {
    let path = std::env::temp_dir().join("nl_hecate_nonexistent_checkpoint_abc123.json");
    let result = load_checkpoint(&path);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

// ── Deterministic JSON output ────────────────────────────────────────

#[test]
fn test_checkpoint_deterministic() {
    let cfg = MAGConfig::test_config();
    let params = MAGParams::init(&cfg, 42);
    let json1 = serde_json::to_string(&params).unwrap();
    let json2 = serde_json::to_string(&params).unwrap();
    assert_eq!(json1, json2, "Same params should produce identical JSON");
}
