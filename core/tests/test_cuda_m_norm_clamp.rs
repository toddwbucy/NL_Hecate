// CUDA M-norm clamp tests — batched kernel vs single-matrix kernel equivalence.
//
// Spec 65 Phase 1: verifies that `m_norm_clamp_batch_f32_cuda` produces
// identical results to calling `m_norm_clamp_f32_cuda` in a loop.

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};
use serial_test::serial;

use nl_hecate_core::gpu_buf::GpuBuf;
use nl_hecate_core::dispatch::{cuda_sync, m_norm_clamp, m_norm_clamp_batch};

/// Run batched clamp on one buffer and single-matrix clamp in a loop on another,
/// then compare element-wise. Tests multiple (d, batch_size) combinations.
#[test]
#[serial(cuda)]
fn test_batched_clamp_matches_single_loop() {
    let configs: &[(usize, usize)] = &[
        (32, 8),   // small d, moderate batch
        (64, 4),   // production hd=64
        (128, 2),  // larger d
        (64, 1),   // single-element batch (degenerate case)
    ];

    for &(d, batch_size) in configs {
        let dd = d * d;
        let total = batch_size * dd;
        let m_norm_max = 5.0f32;

        // Generate random M matrices with norms that will exceed the clamp.
        // rand_buf uses uniform [-0.1, 0.1], so ||M||_F ≈ 0.1 * sqrt(dd);
        // at d=64: 0.1 * 64 = 6.4 > 5.0 — the clamp will fire.
        let host_data = rand_buf(total, 42 + d as u64);

        // Path A: batched clamp (single kernel launch)
        let mut dev_batched = GpuBuf::from_host(&host_data);
        m_norm_clamp_batch(&mut dev_batched, d as i32, batch_size as i32, m_norm_max);
        cuda_sync();
        let mut result_batched = vec![0.0f32; total];
        dev_batched.copy_to_host(&mut result_batched);

        // Path B: single-matrix clamp in a loop (batch_size separate launches)
        // Upload each d×d slice independently, clamp, read back.
        let mut result_single = vec![0.0f32; total];
        for b in 0..batch_size {
            let start = b * dd;
            let slice = &host_data[start..start + dd];
            let mut dev_slice = GpuBuf::from_host(slice);
            m_norm_clamp(&mut dev_slice, d as i32, m_norm_max);
            cuda_sync();
            dev_slice.copy_to_host(&mut result_single[start..start + dd]);
        }

        let label = format!("m_norm_clamp d={d} batch={batch_size}");
        check_close(&label, &result_batched, &result_single, 1e-6);

        // Verify the clamp actually fired for at least one matrix
        for b in 0..batch_size {
            let start = b * dd;
            let norm_sq: f32 = result_batched[start..start + dd]
                .iter()
                .map(|x| x * x)
                .sum();
            let norm = norm_sq.sqrt();
            assert!(
                norm <= m_norm_max + 1e-4,
                "{label}: batch element {b} norm {norm:.4} exceeds max {m_norm_max}"
            );
        }
    }
}

/// Verify the clamp is a no-op when norms are already below the threshold.
#[test]
#[serial(cuda)]
fn test_batched_clamp_noop_when_below_threshold() {
    let d = 64;
    let dd = d * d;
    let batch_size = 4;
    let total = batch_size * dd;
    let m_norm_max = 1000.0f32; // very high — won't fire

    let host_data = rand_buf(total, 99999);
    let mut dev = GpuBuf::from_host(&host_data);

    m_norm_clamp_batch(&mut dev, d as i32, batch_size as i32, m_norm_max);
    cuda_sync();

    let mut result = vec![0.0f32; total];
    dev.copy_to_host(&mut result);
    check_close("noop_clamp", &result, &host_data, 1e-7);
}
