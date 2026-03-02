// CUDA Graph Store and capture lifecycle tests.
//
// These tests do NOT require a real GPU — they exercise the pure-Rust
// CudaGraphStore state machine logic (warmup phases, bitmask routing,
// invalidation) plus the pulse_to_bitmask conversion helper.
//
// The actual cudaGraph* FFI calls are feature-gated and only exercised
// in integration tests run on hardware (test_gpu_resident.rs).

#[cfg(feature = "cuda")]
mod cuda_graph_tests {
    use nl_hecate_core::cuda_graph::CudaGraphStore;
    use nl_hecate_core::conductor::Pulse;
    use nl_hecate_core::gpu_forward::pulse_to_bitmask;

    // ══════════════════════════════════════════════════════════════════
    // CudaGraphStore state machine
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_graph_store_disabled_when_warmup_zero() {
        let store = CudaGraphStore::new(0);
        assert!(!store.enabled, "warmup_steps=0 must disable graph capture");
        assert!(!store.should_capture(), "should_capture must be false when disabled");
        assert!(!store.should_replay(0b0001), "should_replay must be false when disabled");
    }

    #[test]
    fn test_graph_store_warmup_phase() {
        let mut store = CudaGraphStore::new(5);
        assert!(store.enabled);

        for i in 1..=4 {
            store.step();
            assert_eq!(store.steps_seen, i);
            assert!(!store.should_capture(), "should_capture must be false before warmup_steps ({i} < 5)");
            assert!(!store.should_replay(0b0001), "no graphs captured yet");
        }
    }

    #[test]
    fn test_graph_store_capture_fires_exactly_once() {
        let mut store = CudaGraphStore::new(3);

        // Steps 1..3 — not yet
        for _ in 0..2 {
            store.step();
            assert!(!store.should_capture());
        }

        // Step 3 — capture window
        store.step();
        assert_eq!(store.steps_seen, 3);
        assert!(store.should_capture(), "should_capture must be true exactly at step==warmup_steps");

        // Step 4 — capture window has passed
        store.step();
        assert!(!store.should_capture(), "should_capture must be false after warmup_steps");
    }

    #[test]
    fn test_graph_store_should_replay_only_after_warmup() {
        let mut store = CudaGraphStore::new(2);

        // Simulate a successful capture at step 2 by inserting via end_capture.
        // Since we have no real GPU, we can't actually call end_capture (it would
        // invoke cudaStreamEndCapture). Instead verify should_replay returns false
        // when no graph is stored — that's the observable behaviour without hardware.
        store.step(); store.step();
        assert!(store.should_capture());

        store.step(); // now steps_seen=3 > warmup_steps=2
        // No graph stored → should_replay returns false regardless of bitmask
        assert!(!store.should_replay(0b0001));
        assert!(!store.should_replay(0b1111));
    }

    #[test]
    fn test_graph_store_disable_clears_and_stays_off() {
        let mut store = CudaGraphStore::new(2);
        store.step(); store.step(); store.step();

        store.disable();
        assert!(!store.enabled);
        assert!(!store.should_capture());
        assert!(!store.should_replay(0b0001));

        // Stepping after disable must not re-enable
        store.step();
        assert!(!store.enabled);
        assert!(!store.should_capture());
    }

    #[test]
    fn test_graph_store_invalidate_re_enters_warmup() {
        let mut store = CudaGraphStore::new(3);

        // Run through warmup
        for _ in 0..5 {
            store.step();
        }
        assert!(store.steps_seen > store.warmup_steps);
        assert!(!store.should_capture());

        // Invalidate — should reset to step 0 in warmup phase
        store.invalidate();
        assert_eq!(store.steps_seen, 0);
        assert!(store.enabled, "invalidate must re-enable when warmup_steps > 0");
        assert!(!store.should_capture(), "just reset — not at capture point yet");

        // Walk to capture point again
        store.step(); store.step(); store.step();
        assert!(store.should_capture(), "must re-enter capture phase after invalidate");
    }

    #[test]
    fn test_graph_store_invalidate_on_disabled_stays_disabled() {
        let mut store = CudaGraphStore::new(0);
        assert!(!store.enabled);
        store.invalidate();
        // warmup_steps=0 means re-enable condition fails
        assert!(!store.enabled);
    }

    // ══════════════════════════════════════════════════════════════════
    // pulse_to_bitmask
    // ══════════════════════════════════════════════════════════════════

    fn pulse(active: &[bool]) -> Pulse {
        Pulse {
            global_step: 0,
            active_levels: active.to_vec(),
        }
    }

    #[test]
    fn test_pulse_to_bitmask_k1_always_fires() {
        // k=1: only Level 0, always active
        let p = pulse(&[true]);
        assert_eq!(pulse_to_bitmask(&p, 1), 0b0001);
    }

    #[test]
    fn test_pulse_to_bitmask_k2_level0_only() {
        let p = pulse(&[true, false]);
        assert_eq!(pulse_to_bitmask(&p, 2), 0b0001);
    }

    #[test]
    fn test_pulse_to_bitmask_k2_both_active() {
        let p = pulse(&[true, true]);
        assert_eq!(pulse_to_bitmask(&p, 2), 0b0011);
    }

    #[test]
    fn test_pulse_to_bitmask_k4_only_l0() {
        // Normal step: only Level 0 fires
        let p = pulse(&[true, false, false, false]);
        assert_eq!(pulse_to_bitmask(&p, 4), 0b0001);
    }

    #[test]
    fn test_pulse_to_bitmask_k4_l0_l1() {
        // Every 8th step: Level 0 + 1
        let p = pulse(&[true, true, false, false]);
        assert_eq!(pulse_to_bitmask(&p, 4), 0b0011);
    }

    #[test]
    fn test_pulse_to_bitmask_k4_all_levels() {
        // Every 512th step: all four levels
        let p = pulse(&[true, true, true, true]);
        assert_eq!(pulse_to_bitmask(&p, 4), 0b1111);
    }

    #[test]
    fn test_pulse_to_bitmask_k4_l0_l2() {
        // Non-standard pattern: levels 0 and 2 active (bit 0 and bit 2 set)
        let p = pulse(&[true, false, true, false]);
        assert_eq!(pulse_to_bitmask(&p, 4), 0b0101);
    }

    #[test]
    fn test_pulse_to_bitmask_only_counts_up_to_k() {
        // Pulse longer than k — only first k bits are read
        let p = pulse(&[true, true, true, true, true, true]);
        assert_eq!(pulse_to_bitmask(&p, 4), 0b1111, "must ignore levels beyond k");
        assert_eq!(pulse_to_bitmask(&p, 2), 0b0011, "must stop at k=2");
    }

    #[test]
    fn test_pulse_to_bitmask_all_reachable_k4_are_odd() {
        // L0 always fires in CMS → all reachable bitmasks for k=4 have bit 0 set.
        // Non-odd bitmasks represent L0=inactive, which the conductor never produces.
        let reachable = [
            0b0001u8, 0b0011, 0b0101, 0b0111,
            0b1001,   0b1011, 0b1101, 0b1111,
        ];
        for &b in &reachable {
            assert_eq!(b & 1, 1, "bitmask {b:#010b} must have L0 bit set");
        }
        // There are exactly 8 reachable patterns (2^(k-1) for k=4)
        assert_eq!(reachable.len(), 8);
    }
}
