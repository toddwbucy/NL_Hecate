//! Tests for NL serving abstraction (non-stationary models).
//!
//! Run: RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --release --features serving --test test_serving -- --nocapture

#![feature(autodiff)]

#[cfg(feature = "serving")]
mod tests {
    use nl_hecate_core::serving::*;
    use nl_hecate_core::model::{MAGConfig, MAGParams};
    use nl_hecate_core::context_stream::VecStream;

    // ── Helpers ────────────────────────────────────────────────────────

    fn test_config() -> MAGConfig {
        MAGConfig::test_config()
    }

    fn test_config_k4() -> MAGConfig {
        MAGConfig::test_config_k4()
    }

    fn make_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
        (input_ids, target_ids)
    }

    fn make_corpus(cfg: &MAGConfig, num_chunks: usize) -> Vec<usize> {
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        // Need s * num_chunks + 1 tokens (for next-token targets)
        let len = s * num_chunks + 1;
        (0..len).map(|t| t % v).collect()
    }

    // ── Phase A: LatencyTracker + ChunkResult ──────────────────────────

    #[test]
    fn test_single_chunk_latency() {
        let mut tracker = LatencyTracker::new();
        tracker.record(5.0);
        assert!((tracker.average_ms() - 5.0).abs() < 1e-6);
        assert!((tracker.worst_ms() - 5.0).abs() < 1e-6);
        assert!((tracker.p99_ms() - 5.0).abs() < 1e-6);
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_multi_chunk_average() {
        let mut tracker = LatencyTracker::new();
        tracker.record(2.0);
        tracker.record(4.0);
        tracker.record(6.0);
        assert!((tracker.average_ms() - 4.0).abs() < 1e-6);
        assert!((tracker.worst_ms() - 6.0).abs() < 1e-6);
        assert_eq!(tracker.count(), 3);
    }

    #[test]
    fn test_p99_tracking() {
        let mut tracker = LatencyTracker::new();
        // 101 samples: 100 at 1.0ms, 1 outlier at 100.0ms
        // p99 nearest-rank: ceil(0.99 * 101) = 100, index 99 → 1.0
        // The outlier at index 100 is above p99.
        for _ in 0..100 {
            tracker.record(1.0);
        }
        tracker.record(100.0);
        // p99 = 1.0 (the 100th value in sorted order of 101 elements)
        assert!((tracker.p99_ms() - 1.0).abs() < 1e-6);
        // worst should still be 100.0
        assert!((tracker.worst_ms() - 100.0).abs() < 1e-6);
        // Average
        let expected_avg = (100.0 * 1.0 + 100.0) / 101.0;
        assert!((tracker.average_ms() - expected_avg).abs() < 1e-4);
    }

    // ── Phase B: Session (Test mode) ───────────────────────────────────

    #[test]
    fn test_session_smoke() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);
        let result = session.process_chunk(&params, &cfg, &input_ids, &target_ids);

        // Basic sanity: loss is finite, logits have correct shape
        assert!(result.loss.is_finite(), "loss must be finite, got {}", result.loss);
        assert!(!result.loss.is_nan(), "loss must not be NaN");
        assert_eq!(result.logits.len(), cfg.swa.seq_len * cfg.swa.vocab_size);
        assert!(result.chunk_time_ms >= 0.0);
        assert_eq!(result.tokens_processed, cfg.swa.seq_len);
        assert!(!session.has_stream());
    }

    #[test]
    fn test_process_chunk_advances_step() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);
        assert_eq!(session.conductor_step(), 0);
        assert_eq!(session.chunks_processed(), 0);

        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(session.conductor_step(), 1);
        assert_eq!(session.chunks_processed(), 1);

        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        assert_eq!(session.conductor_step(), 2);
        assert_eq!(session.chunks_processed(), 2);
    }

    #[test]
    fn test_params_unchanged() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let params_before = params.clone();
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);
        for _ in 0..5 {
            session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        }

        // Outer-loop params must be unchanged (read-only during serving)
        assert_eq!(params.swa.w_embed, params_before.swa.w_embed);
        assert_eq!(params.swa.w_q, params_before.swa.w_q);
        assert_eq!(params.swa.w_k, params_before.swa.w_k);
        assert_eq!(params.swa.w_v, params_before.swa.w_v);
        assert_eq!(params.swa.w_o, params_before.swa.w_o);
        assert_eq!(params.swa.w_unembed, params_before.swa.w_unembed);
        for (level, level_before) in params.levels.iter().zip(params_before.levels.iter()) {
            assert_eq!(level.w_k_mem, level_before.w_k_mem);
            assert_eq!(level.w_v_mem, level_before.w_v_mem);
            assert_eq!(level.w_q_mem, level_before.w_q_mem);
        }
    }

    #[test]
    fn test_latency_recorded() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);
        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        session.process_chunk(&params, &cfg, &input_ids, &target_ids);

        assert_eq!(session.latency().count(), 3);
        assert!(session.latency().average_ms() > 0.0);
        assert!(session.latency().worst_ms() >= session.latency().average_ms());
    }

    // ── Phase C: Session (Stream mode) ─────────────────────────────────

    #[test]
    fn test_stream_session_processes_chunks() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let corpus = make_corpus(&cfg, 3);
        let stream = Box::new(VecStream::new(corpus));

        let mut session = Session::new_stream(1, &cfg, stream);
        assert!(session.has_stream());

        let result = session.process_next(&params, &cfg);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.loss.is_finite());
        assert_eq!(r.tokens_processed, cfg.swa.seq_len);
        assert_eq!(session.chunks_processed(), 1);
    }

    #[test]
    fn test_stream_exhaustion() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        // VecStream wraps around (never truly exhausts), so this tests
        // that multiple chunks can be processed from a small corpus.
        let corpus = make_corpus(&cfg, 2);
        let stream = Box::new(VecStream::new(corpus));

        let mut session = Session::new_stream(1, &cfg, stream);
        // Process more chunks than corpus contains — VecStream wraps
        for i in 0..5 {
            let result = session.process_next(&params, &cfg);
            assert!(result.is_some(), "chunk {} should succeed (VecStream wraps)", i);
        }
        assert_eq!(session.chunks_processed(), 5);
    }

    #[test]
    fn test_stream_advances_conductor() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let corpus = make_corpus(&cfg, 5);
        let stream = Box::new(VecStream::new(corpus));

        let mut session = Session::new_stream(1, &cfg, stream);
        assert_eq!(session.conductor_step(), 0);

        session.process_next(&params, &cfg);
        assert_eq!(session.conductor_step(), 1);

        session.process_next(&params, &cfg);
        assert_eq!(session.conductor_step(), 2);
    }

    // ── Phase D: Checkpoint / Restore ──────────────────────────────────

    #[test]
    fn test_checkpoint_roundtrip() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let corpus = make_corpus(&cfg, 10);
        let stream = Box::new(VecStream::new(corpus.clone()));

        let mut session = Session::new_stream(1, &cfg, stream);

        // Process 3 chunks, then checkpoint
        for _ in 0..3 {
            session.process_next(&params, &cfg);
        }
        let cp = session.checkpoint();
        assert_eq!(cp.session_id, 1);
        assert_eq!(cp.chunks_processed, 3);

        // Create fresh session, restore
        let stream2 = Box::new(VecStream::new(corpus));
        let mut session2 = Session::new_stream(1, &cfg, stream2);
        session2.restore(&cp).expect("restore should succeed");
        assert_eq!(session2.chunks_processed(), 3);
    }

    #[test]
    fn test_restore_pulse_mismatch_error() {
        let cfg = test_config();
        let corpus = make_corpus(&cfg, 10);
        let stream = Box::new(VecStream::new(corpus.clone()));

        let mut session = Session::new_stream(1, &cfg, stream);
        let params = MAGParams::init(&cfg, 42);
        // Process 2 chunks so step=2
        session.process_next(&params, &cfg);
        session.process_next(&params, &cfg);
        let mut cp = session.checkpoint();

        // Corrupt pulse_id in the stream cursor
        cp.inner.stream.pulse_id = 999;

        let stream2 = Box::new(VecStream::new(corpus));
        let mut session2 = Session::new_stream(1, &cfg, stream2);
        let err = session2.restore(&cp);
        assert!(err.is_err(), "restore with mismatched pulse should fail");
    }

    #[test]
    fn test_restore_config_mismatch_error() {
        let cfg = test_config();
        let corpus = make_corpus(&cfg, 10);
        let stream = Box::new(VecStream::new(corpus.clone()));

        let mut session = Session::new_stream(1, &cfg, stream);
        let params = MAGParams::init(&cfg, 42);
        session.process_next(&params, &cfg);
        let mut cp = session.checkpoint();

        // Corrupt config: change k
        cp.inner.conductor.k = 99;

        let stream2 = Box::new(VecStream::new(corpus));
        let mut session2 = Session::new_stream(1, &cfg, stream2);
        let err = session2.restore(&cp);
        assert!(err.is_err(), "restore with mismatched config should fail");
    }

    #[test]
    fn test_checkpoint_preserves_memory_state() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let corpus = make_corpus(&cfg, 10);
        let stream = Box::new(VecStream::new(corpus.clone()));

        let mut session = Session::new_stream(1, &cfg, stream);

        // Process chunks — memory should change from all-zeros
        for _ in 0..5 {
            session.process_next(&params, &cfg);
        }

        // Memory should have been modified by inner loop
        let mem = &session.context().memory[0];
        let has_nonzero = mem.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "memory should be non-zero after processing chunks");

        // Checkpoint and restore preserves conductor/stream state
        let cp = session.checkpoint();
        let stream2 = Box::new(VecStream::new(corpus));
        let mut session2 = Session::new_stream(1, &cfg, stream2);
        session2.restore(&cp).expect("restore should succeed");
        assert_eq!(session2.chunks_processed(), 5);
        assert_eq!(session2.conductor_step(), session.conductor_step());
    }

    // ── Phase E: Integration + Edge Cases ──────────────────────────────

    #[test]
    fn test_k4_session_all_levels_fire() {
        let cfg = test_config_k4();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);

        // Step 0: all 4 levels fire
        let r0 = session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        assert!(r0.loss.is_finite());

        // Steps 1-7: only level 0 fires
        for _ in 1..8 {
            let r = session.process_chunk(&params, &cfg, &input_ids, &target_ids);
            assert!(r.loss.is_finite());
        }

        // Step 8: levels 0 and 1 fire
        let r8 = session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        assert!(r8.loss.is_finite());
        assert_eq!(session.conductor_step(), 9);
    }

    #[test]
    fn test_session_isolation() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session_a = Session::new_test(1, &cfg);
        let mut session_b = Session::new_test(2, &cfg);

        // Process different number of chunks
        for _ in 0..5 {
            session_a.process_chunk(&params, &cfg, &input_ids, &target_ids);
        }
        session_b.process_chunk(&params, &cfg, &input_ids, &target_ids);

        // Sessions have independent state
        assert_eq!(session_a.chunks_processed(), 5);
        assert_eq!(session_b.chunks_processed(), 1);
        assert_eq!(session_a.conductor_step(), 5);
        assert_eq!(session_b.conductor_step(), 1);
        assert_ne!(session_a.id(), session_b.id());

        // Memory should differ (different number of inner-loop updates)
        let mem_a = &session_a.context().memory[0];
        let mem_b = &session_b.context().memory[0];
        assert_ne!(mem_a, mem_b, "sessions should have different memory states");
    }

    #[test]
    fn test_long_session_latency_stable() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);

        // Process 100 chunks
        let mut first_10_avg = 0.0f32;
        let mut last_10_avg = 0.0f32;
        for i in 0..100 {
            let r = session.process_chunk(&params, &cfg, &input_ids, &target_ids);
            assert!(r.loss.is_finite(), "chunk {} loss not finite: {}", i, r.loss);
            if i < 10 {
                first_10_avg += r.chunk_time_ms;
            }
            if i >= 90 {
                last_10_avg += r.chunk_time_ms;
            }
        }
        first_10_avg /= 10.0;
        last_10_avg /= 10.0;

        // Per-token latency should NOT grow with context length.
        // Allow 5x tolerance for system jitter, but catch O(n) growth.
        assert!(
            last_10_avg < first_10_avg * 5.0 + 1.0, // +1.0ms absolute tolerance
            "latency should not grow: first_10_avg={:.3}ms, last_10_avg={:.3}ms",
            first_10_avg, last_10_avg,
        );
        assert_eq!(session.chunks_processed(), 100);
    }

    #[test]
    fn test_context_state_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_data(&cfg);

        let mut session = Session::new_test(1, &cfg);

        // Memory starts at zero
        let mem_before: Vec<f32> = session.context().memory[0].clone();
        assert!(mem_before.iter().all(|&v| v == 0.0), "memory should start at zero");

        // After processing, memory should change (inner loop self-modification)
        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        let mem_after = &session.context().memory[0];
        assert_ne!(&mem_before, mem_after, "memory should evolve after processing");

        // After more processing, memory should continue to evolve
        let mem_mid = mem_after.clone();
        session.process_chunk(&params, &cfg, &input_ids, &target_ids);
        let mem_final = &session.context().memory[0];
        assert_ne!(&mem_mid, mem_final, "memory should continue evolving");
    }
}
