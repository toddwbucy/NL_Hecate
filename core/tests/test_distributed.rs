//! Tests for CMS-aware multi-GPU gradient synchronization.
//!
//! All tests use MockProcessGroup — no real multi-GPU hardware needed.
//! Run: cargo test --features distributed --test test_distributed

#[cfg(feature = "distributed")]
mod tests {
    use nl_hecate_core::distributed::*;
    use nl_hecate_core::model::{MAGConfig, MAGParams};
    use nl_hecate_core::conductor::{Pulse, Conductor, ContextState, ErrorBuffer};

    // ── Helper functions ──────────────────────────────────────────────

    fn test_config_k2() -> MAGConfig {
        MAGConfig::test_config_k2()
    }

    fn test_config_k4() -> MAGConfig {
        MAGConfig::test_config_k4()
    }

    fn make_test_data(cfg: &MAGConfig) -> (Vec<usize>, Vec<usize>) {
        let s = cfg.swa.seq_len;
        let v = cfg.swa.vocab_size;
        let input_ids: Vec<usize> = (0..s).map(|t| t % v).collect();
        let target_ids: Vec<usize> = (1..=s).map(|t| t % v).collect();
        (input_ids, target_ids)
    }

    // ── Group 1: ProcessGroup (3 tests) ──────────────────────────────

    #[test]
    fn test_mock_allreduce_sum() {
        let group = MockProcessGroup::new_group(4);
        let mut buf = vec![1.0f32, 2.0, 3.0];
        group[0].allreduce_sum(&mut buf);
        // Simulated sum: each element * 4 (world_size)
        assert_eq!(buf, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_mock_world_size_rank() {
        let group = MockProcessGroup::new_group(3);
        assert_eq!(group[0].world_size(), 3);
        assert_eq!(group[1].world_size(), 3);
        assert_eq!(group[2].world_size(), 3);
        assert_eq!(group[0].rank(), 0);
        assert_eq!(group[1].rank(), 1);
        assert_eq!(group[2].rank(), 2);
    }

    #[test]
    fn test_mock_multiple_allreduces() {
        let group = MockProcessGroup::new_group(2);
        let mut buf1 = vec![1.0f32; 4];
        let mut buf2 = vec![2.0f32; 8];
        group[0].allreduce_sum(&mut buf1);
        group[0].allreduce_sum(&mut buf2);
        let log = group[0].call_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].len, 4);
        assert_eq!(log[1].len, 8);
        assert_eq!(log[0].rank, 0);
    }

    // ── Group 2: sync_gradients (4 tests) ────────────────────────────

    #[test]
    fn test_sync_all_active() {
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);
        // Set some non-zero gradients
        grads.swa.w_q[0] = 1.0;
        grads.levels[0].w_k_mem[0] = 2.0;
        grads.levels[1].w_k_mem[0] = 3.0;

        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let group = MockProcessGroup::new_group(2);

        let count = sync_gradients(&mut grads, &pulse, &group[0]);

        // SWA: 6 allreduces + 2 active levels * 9 = 24
        assert_eq!(count, 6 + 2 * 9);

        // After mock allreduce (multiply by 2) then divide by 2: values unchanged
        assert!((grads.swa.w_q[0] - 1.0).abs() < 1e-6);
        assert!((grads.levels[0].w_k_mem[0] - 2.0).abs() < 1e-6);
        assert!((grads.levels[1].w_k_mem[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sync_partial_active() {
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);
        grads.levels[0].w_k_mem[0] = 1.0;
        grads.levels[1].w_k_mem[0] = 1.0;

        // Only level 0 active
        let pulse = Pulse { global_step: 1, active_levels: vec![true, false] };
        let group = MockProcessGroup::new_group(2);

        let count = sync_gradients(&mut grads, &pulse, &group[0]);

        // SWA: 6 + level 0: 9 = 15 (level 1 skipped)
        assert_eq!(count, 6 + 9);
    }

    #[test]
    fn test_sync_preserves_frozen() {
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);
        // Set level 1 grads (which will be frozen)
        grads.levels[1].w_k_mem[0] = 5.0;
        grads.levels[1].w_v_mem[0] = 7.0;
        grads.levels[1].b_alpha[0] = 0.3;

        let original_wk = grads.levels[1].w_k_mem[0];
        let original_wv = grads.levels[1].w_v_mem[0];
        let original_ba = grads.levels[1].b_alpha[0];

        // Level 1 frozen
        let pulse = Pulse { global_step: 1, active_levels: vec![true, false] };
        let group = MockProcessGroup::new_group(4);

        sync_gradients(&mut grads, &pulse, &group[0]);

        // Frozen level grads should be completely untouched
        assert_eq!(grads.levels[1].w_k_mem[0], original_wk);
        assert_eq!(grads.levels[1].w_v_mem[0], original_wv);
        assert_eq!(grads.levels[1].b_alpha[0], original_ba);
    }

    #[test]
    fn test_sync_mean_reduction() {
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);
        grads.swa.w_q[0] = 10.0;
        grads.levels[0].w_k_mem[0] = 20.0;

        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let group = MockProcessGroup::new_group(4);

        sync_gradients(&mut grads, &pulse, &group[0]);

        // Mock allreduce: 10.0 * 4 = 40.0, then /4 = 10.0 (same input since identical ranks)
        assert!((grads.swa.w_q[0] - 10.0).abs() < 1e-6);
        assert!((grads.levels[0].w_k_mem[0] - 20.0).abs() < 1e-6);
    }

    // ── Group 3: ThroughputTracker (3 tests) ─────────────────────────

    #[test]
    fn test_single_step_report() {
        let mut tracker = ThroughputTracker::new(1024); // 1024 tokens/step
        let report = tracker.record(100.0, 15); // 100ms step

        assert_eq!(report.allreduce_count, 15);
        assert!((report.step_time_ms - 100.0).abs() < 1e-3);
        // 1024 tokens / 0.1 sec = 10240 tokens/sec
        assert!((report.tokens_per_sec_per_gpu - 10240.0).abs() < 1.0);
    }

    #[test]
    fn test_multi_step_average() {
        let mut tracker = ThroughputTracker::new(100);
        tracker.record(50.0, 10);  // 2000 tok/s
        tracker.record(100.0, 10); // 1000 tok/s
        tracker.record(50.0, 10);  // 2000 tok/s

        // Average: 300 tokens / 0.2 sec = 1500 tok/s
        let avg = tracker.average_tokens_per_sec();
        assert!((avg - 1500.0).abs() < 1.0);
    }

    #[test]
    fn test_worst_case_tracking() {
        let mut tracker = ThroughputTracker::new(100);
        let r1 = tracker.record(10.0, 5);  // 10000 tok/s
        let r2 = tracker.record(100.0, 5); // 1000 tok/s (worst)
        let r3 = tracker.record(20.0, 5);  // 5000 tok/s

        // After first step, worst = first
        assert!((r1.worst_gpu_tokens_per_sec - 10000.0).abs() < 1.0);
        // After second step, worst = 1000
        assert!((r2.worst_gpu_tokens_per_sec - 1000.0).abs() < 1.0);
        // Third step can't be worse, stays 1000
        assert!((r3.worst_gpu_tokens_per_sec - 1000.0).abs() < 1.0);
    }

    // ── Group 4: distributed_step (3 tests) ──────────────────────────

    #[test]
    fn test_distributed_step_smoke() {
        // 2 ranks with same data — no NaN, finite loss
        let cfg = test_config_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let group = MockProcessGroup::new_group(2);

        let (loss, report) = distributed_step(
            &mut params, &cfg, &input_ids, &target_ids,
            &mut conductor, &mut context, &mut error_buffers,
            &group[0], 0.01,
        );

        assert!(loss.is_finite(), "Loss not finite: {loss}");
        assert!(loss > 0.0, "Loss should be positive: {loss}");
        assert!(report.step_time_ms > 0.0);
        assert!(report.allreduce_count > 0);
        assert!(report.tokens_per_sec_per_gpu > 0.0);

        // Verify all params are finite
        for &v in params.swa.w_q.iter() {
            assert!(v.is_finite(), "Param not finite after distributed step");
        }
    }

    #[test]
    fn test_distributed_step_loss_decreases() {
        // Run 100 steps and verify loss trends downward
        let cfg = test_config_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let group = MockProcessGroup::new_group(2);

        let mut losses = Vec::new();
        for _ in 0..100 {
            let (loss, _) = distributed_step(
                &mut params, &cfg, &input_ids, &target_ids,
                &mut conductor, &mut context, &mut error_buffers,
                &group[0], 0.01,
            );
            losses.push(loss);
        }

        // Compare first 10 average to last 10 average
        let first_10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last_10: f32 = losses[90..].iter().sum::<f32>() / 10.0;
        assert!(
            last_10 < first_10,
            "Loss should decrease: first_10={first_10:.4}, last_10={last_10:.4}"
        );

        // All losses finite
        for (i, &loss) in losses.iter().enumerate() {
            assert!(loss.is_finite(), "Loss[{i}] not finite: {loss}");
        }
    }

    #[test]
    fn test_distributed_step_identical_pulses() {
        // Two ranks with same step counter produce identical pulses
        let cfg = test_config_k2();
        let group = MockProcessGroup::new_group(2);

        // Create two independent conductors (simulating two ranks)
        let mut c0 = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut c1 = Conductor::new(cfg.k, cfg.chunk_sizes.clone());

        for step in 0..20 {
            let p0 = c0.pulse();
            let p1 = c1.pulse();
            assert_eq!(
                p0.active_levels, p1.active_levels,
                "Pulses differ at step {step}: rank0={:?}, rank1={:?}",
                p0.active_levels, p1.active_levels
            );
            assert_eq!(p0.global_step, p1.global_step);
            assert_eq!(p0, p1, "Pulse PartialEq failed at step {step}");
            c0.advance();
            c1.advance();
        }

        let _ = group; // used to verify 2-rank scenario
    }

    // ── Group 5: CMS integration (3 tests) ───────────────────────────

    #[test]
    fn test_k2_frozen_isolation() {
        // Verify sync_gradients does NOT touch frozen levels
        let cfg = test_config_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);
        let mut context = ContextState::new(cfg.k, cfg.swa.d_model);

        // Step 0: both active (initialize memory)
        let pulse0 = Pulse { global_step: 0, active_levels: vec![true, true] };
        let (_, _) = nl_hecate_core::mag::cms_forward(
            &params, &cfg, &input_ids, &target_ids, &pulse0, &mut context,
        );

        // Step 1: Level 1 frozen
        let pulse1 = Pulse { global_step: 1, active_levels: vec![true, false] };
        let (_, cache) = nl_hecate_core::mag::cms_forward(
            &params, &cfg, &input_ids, &target_ids, &pulse1, &mut context,
        );

        let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
            .map(|_| ErrorBuffer::new(cfg.swa.d_model))
            .collect();
        let mut grads = nl_hecate_core::mag::cms_backward(
            &params, &cfg, &cache, &input_ids, &target_ids, &mut error_buffers,
        );

        // Frozen level 1 has zero direct grads, non-zero error buffer
        let l1_norm_before: f32 = grads.levels[1].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l1_norm_before < 1e-12, "Frozen level should have zero direct grads");
        assert!(error_buffers[1].steps_accumulated > 0, "Error buffer should have grads");

        // sync_gradients should NOT touch the zero grads for frozen level
        let group = MockProcessGroup::new_group(2);
        sync_gradients(&mut grads, &pulse1, &group[0]);

        let l1_norm_after: f32 = grads.levels[1].w_k_mem.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(l1_norm_after < 1e-12, "Frozen level grads should stay zero after sync");
    }

    #[test]
    fn test_k4_allreduce_count() {
        // Over 512 steps with k=4 [1,8,64,512], verify ~1.14 level-allreduces/step
        let cfg = test_config_k4();
        let group = MockProcessGroup::new_group(2);

        let mut total_level_allreduces = 0usize;
        let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());

        for _ in 0..512 {
            let pulse = conductor.pulse();
            let mut grads = MAGParams::zeros_like(&cfg);
            let count = sync_gradients(&mut grads, &pulse, &group[0]);

            // Level allreduces = count - 6 (the SWA allreduces)
            let level_count = (count - 6) / 9; // 9 allreduces per active level
            total_level_allreduces += level_count;

            conductor.advance();
        }

        // Expected active levels over 512 steps:
        // L0: 512 (every step), L1: 64 (every 8), L2: 8 (every 64), L3: 1 (every 512)
        // Total active-level instances = 512 + 64 + 8 + 1 = 585
        // Per-step average = 585/512 ≈ 1.142
        let per_step = total_level_allreduces as f32 / 512.0;
        assert!(
            (per_step - 1.142).abs() < 0.02,
            "Expected ~1.142 level allreduces/step, got {per_step}"
        );
    }

    #[test]
    fn test_gradient_equivalence() {
        // Two ranks with same data should converge to identical params
        let cfg = test_config_k2();
        let mut params_r0 = MAGParams::init(&cfg, 42);
        let mut params_r1 = MAGParams::init(&cfg, 42);
        let (input_ids, target_ids) = make_test_data(&cfg);

        let group = MockProcessGroup::new_group(2);

        let mut conductor_r0 = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut conductor_r1 = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let mut context_r0 = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut context_r1 = ContextState::new(cfg.k, cfg.swa.d_model);
        let mut eb_r0: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();
        let mut eb_r1: Vec<ErrorBuffer> = (0..cfg.k).map(|_| ErrorBuffer::new(cfg.swa.d_model)).collect();

        let lr = 0.01;
        for _ in 0..20 {
            let (loss_r0, _) = distributed_step(
                &mut params_r0, &cfg, &input_ids, &target_ids,
                &mut conductor_r0, &mut context_r0, &mut eb_r0,
                &group[0], lr,
            );
            let (loss_r1, _) = distributed_step(
                &mut params_r1, &cfg, &input_ids, &target_ids,
                &mut conductor_r1, &mut context_r1, &mut eb_r1,
                &group[1], lr,
            );

            // Same data + same mock allreduce → identical losses and params
            assert!(
                (loss_r0 - loss_r1).abs() < 1e-6,
                "Losses differ: r0={loss_r0}, r1={loss_r1}"
            );
        }

        // Verify params are identical after 20 steps
        for i in 0..params_r0.swa.w_q.len() {
            assert!(
                (params_r0.swa.w_q[i] - params_r1.swa.w_q[i]).abs() < 1e-5,
                "w_q[{i}] differs: r0={}, r1={}",
                params_r0.swa.w_q[i], params_r1.swa.w_q[i]
            );
        }
        for level in 0..cfg.k {
            for i in 0..params_r0.levels[level].w_k_mem.len() {
                assert!(
                    (params_r0.levels[level].w_k_mem[i] - params_r1.levels[level].w_k_mem[i]).abs() < 1e-5,
                    "Level {level} w_k_mem[{i}] differs"
                );
            }
        }
    }

    // ── Group 6: Edge cases (2 tests) ────────────────────────────────

    #[test]
    fn test_single_rank_noop() {
        // With world_size=1, sync_gradients should be a no-op (values unchanged)
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);
        grads.swa.w_q[0] = 42.0;
        grads.levels[0].w_k_mem[0] = 7.0;
        grads.levels[1].w_k_mem[0] = 13.0;

        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let group = MockProcessGroup::new_group(1);

        sync_gradients(&mut grads, &pulse, &group[0]);

        // world_size=1: allreduce_sum multiplies by 1, then divides by 1
        assert!((grads.swa.w_q[0] - 42.0).abs() < 1e-6);
        assert!((grads.levels[0].w_k_mem[0] - 7.0).abs() < 1e-6);
        assert!((grads.levels[1].w_k_mem[0] - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_gradients() {
        // Syncing zero gradients should produce zero gradients
        let cfg = test_config_k2();
        let mut grads = MAGParams::zeros_like(&cfg);

        let pulse = Pulse { global_step: 0, active_levels: vec![true, true] };
        let group = MockProcessGroup::new_group(4);

        sync_gradients(&mut grads, &pulse, &group[0]);

        // All zeros remain zero
        assert!(grads.swa.w_q.iter().all(|&v| v == 0.0));
        assert!(grads.swa.w_embed.iter().all(|&v| v == 0.0));
        for level in &grads.levels {
            assert!(level.w_k_mem.iter().all(|&v| v == 0.0));
            assert!(level.w_v_mem.iter().all(|&v| v == 0.0));
        }
    }
}
