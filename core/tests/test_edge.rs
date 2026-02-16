/// Edge deployment integration tests.
///
/// Validates three deployment profiles:
///   Profile 1: Inner-loop only (frozen outer-loop, memory adapts)
///   Profile 2: Full NL (forward + backward + gradient apply)
///   Profile 3: WASM (validated via cross-compilation, not runtime)
///
/// Run: cargo +enzyme test --release --features edge --test test_edge

use nl_hecate_core::edge::{EdgeConfig, EdgeModel, estimate_model_size_bytes};
use nl_hecate_core::model::{CompositionKind, MemoryRuleKind};

fn make_input(seq_len: usize, vocab_size: usize) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();
    let target_ids: Vec<usize> = (1..=seq_len).map(|i| i % vocab_size).collect();
    (input_ids, target_ids)
}

// ── Profile 1: Inner-loop only (frozen outer-loop) ─────────────────

#[test]
fn test_profile1_inner_loop_only() {
    // Profile 1: process() runs inner-loop adaptation without touching outer-loop weights
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    let params_before = model.params.clone();
    let mem_before = model.memory_snapshot();

    let (loss, logits) = model.process(&input, &target);
    assert!(loss.is_finite() && loss > 0.0);
    assert_eq!(logits.len(), cfg.seq_len * cfg.vocab_size);

    // Outer-loop weights unchanged (Profile 1 guarantee)
    assert_eq!(model.params.swa.w_q, params_before.swa.w_q);
    assert_eq!(model.params.swa.w_embed, params_before.swa.w_embed);
    assert_eq!(model.params.levels[0].w_k_mem, params_before.levels[0].w_k_mem);

    // Inner-loop memory changed (adaptation happened)
    let mem_after = model.memory_snapshot();
    assert_ne!(mem_before, mem_after, "memory should adapt during forward pass");
}

#[test]
fn test_profile1_multi_step_adaptation() {
    // Verify memory continues to evolve across multiple process() calls.
    // Use different input each step to ensure memory state diverges.
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);

    let mut memories = Vec::new();
    for step in 0..3 {
        let input: Vec<usize> = (0..cfg.seq_len).map(|i| (i + step * 7) % cfg.vocab_size).collect();
        let target: Vec<usize> = (1..=cfg.seq_len).map(|i| (i + step * 7) % cfg.vocab_size).collect();
        let (loss, _) = model.process(&input, &target);
        assert!(loss.is_finite(), "step {step}: loss should be finite");
        memories.push(model.memory_snapshot());
    }

    // Memory should differ between steps (inner-loop is adapting)
    assert_ne!(memories[0], memories[1], "memory should change between step 0 and 1");
    assert_ne!(memories[1], memories[2], "memory should change between step 1 and 2");
}

// ── Profile 2: Full NL (forward + backward) ────────────────────────

#[test]
fn test_profile2_full_nl() {
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    let (loss, grads) = model.forward_backward(&input, &target);
    assert!(loss.is_finite() && loss > 0.0);

    // Gradients should be non-zero
    let swa_grad_norm: f32 = grads.swa.w_q.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mem_grad_norm: f32 = grads.levels[0].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(swa_grad_norm > 0.0, "SWA gradients should be non-zero");
    assert!(mem_grad_norm > 0.0, "memory gradients should be non-zero");
}

#[test]
fn test_profile2_weight_update() {
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    let w_q_before = model.params.swa.w_q.clone();
    let (_, grads) = model.forward_backward(&input, &target);
    model.apply_gradients(&grads, 0.01);

    // Weights should have changed
    assert_ne!(model.params.swa.w_q, w_q_before, "weights should change after gradient update");
}

#[test]
fn test_profile2_training_reduces_loss() {
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    let (loss_before, grads) = model.forward_backward(&input, &target);
    model.apply_gradients(&grads, 0.001);

    // Fresh context for fair comparison (inner-loop memory resets)
    model.context = nl_hecate_core::conductor::ContextState::new(cfg.k, cfg.d_model);
    model.conductor = nl_hecate_core::conductor::Conductor::new(cfg.k, vec![1]);

    let (loss_after, _) = model.forward_backward(&input, &target);
    assert!(
        loss_after < loss_before,
        "loss should decrease after gradient step: before={loss_before}, after={loss_after}"
    );
}

// ── Binary size assertions ──────────────────────────────────────────

#[test]
fn test_model_size_under_1mb() {
    let cfg = EdgeConfig::micro_d64();
    let model = EdgeModel::new_random(&cfg, 42);
    let total = model.deployment_footprint_bytes();
    assert!(
        total < 1_000_000,
        "d=64 deployment footprint should be < 1MB, got {} bytes ({:.1} KB)",
        total,
        total as f64 / 1024.0
    );
}

#[test]
fn test_model_size_d128_under_1mb() {
    let cfg = EdgeConfig::micro_d128();
    let model = EdgeModel::new_random(&cfg, 42);
    let total = model.deployment_footprint_bytes();
    assert!(
        total < 1_000_000,
        "d=128 deployment footprint should be < 1MB, got {} bytes ({:.1} KB)",
        total,
        total as f64 / 1024.0
    );
}

// ── Throughput smoke tests ──────────────────────────────────────────

#[test]
fn test_throughput_above_1k_tok_s() {
    // Conservative: > 1k tok/s on x86_64 (Criterion shows ~34k for d=64)
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    let start = std::time::Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = model.process(&input, &target);
    }
    let elapsed = start.elapsed();
    let total_tokens = iterations * cfg.seq_len;
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    assert!(
        tok_per_sec > 1_000.0,
        "throughput should exceed 1k tok/s, got {:.0} tok/s",
        tok_per_sec
    );
}

#[test]
fn test_forward_latency_under_1ms() {
    // d=64, seq=16: Criterion shows ~471µs
    let cfg = EdgeConfig::micro_d64();
    let mut model = EdgeModel::new_random(&cfg, 42);
    let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);

    // Warm up
    for _ in 0..10 {
        let _ = model.process(&input, &target);
    }

    let start = std::time::Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = model.process(&input, &target);
    }
    let avg_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    assert!(
        avg_ms < 1.0,
        "d=64 forward latency should be < 1ms, got {:.3} ms",
        avg_ms
    );
}

// ── Memory fits L2 cache ────────────────────────────────────────────

#[test]
fn test_memory_fits_l2_cache() {
    let cfg = EdgeConfig::micro_d64();
    let model = EdgeModel::new_random(&cfg, 42);
    let mem = model.memory_state_bytes();
    assert!(
        mem < 256 * 1024,
        "d=64 memory state should fit in L2 (256KB), got {} bytes",
        mem
    );
}

// ── Size estimation ─────────────────────────────────────────────────

#[test]
fn test_estimate_size_matches_actual() {
    for cfg in [EdgeConfig::micro_d64(), EdgeConfig::micro_d128()] {
        let model = EdgeModel::new_random(&cfg, 42);
        let estimated = estimate_model_size_bytes(&cfg);
        let actual = model.deployment_footprint_bytes();
        assert_eq!(estimated, actual, "d={}: estimate {} != actual {}", cfg.d_model, estimated, actual);
    }
}
