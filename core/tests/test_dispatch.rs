// Dispatch tests: Backend enum, GPU detection, backend selection, force override.
//
// Tests are split into:
//   - Always-run: Backend enum, select_backend() logic (no GPU needed)
//   - CUDA-only: detect_gpu(), force_rust_reference() with actual dispatch

use nl_hecate_core::dispatch::{Backend, GpuInfo, select_backend, force_rust_reference, is_rust_forced, detect_gpu};

// ══════════════════════════════════════════════════════════════════════
// Backend enum tests (always run)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_backend_variants() {
    let r = Backend::RustReference;
    let n = Backend::CudaNative;
    let p = Backend::CudaPtx;

    // Debug + PartialEq work
    assert_eq!(r, Backend::RustReference);
    assert_ne!(r, n);
    assert_ne!(n, p);
    assert_eq!(format!("{:?}", r), "RustReference");
    assert_eq!(format!("{:?}", n), "CudaNative");
    assert_eq!(format!("{:?}", p), "CudaPtx");
}

#[test]
fn test_backend_copy_clone() {
    let b = Backend::CudaNative;
    let b2 = b; // Copy
    let b3 = b.clone(); // Clone
    assert_eq!(b, b2);
    assert_eq!(b, b3);
}

// ══════════════════════════════════════════════════════════════════════
// Backend selection logic (no GPU needed)
// ══════════════════════════════════════════════════════════════════════

fn make_gpu(name: &str, major: i32, minor: i32) -> GpuInfo {
    GpuInfo {
        name: name.to_string(),
        compute_major: major,
        compute_minor: minor,
        sm_version: major * 10 + minor,
    }
}

#[test]
fn test_select_backend_no_gpu() {
    assert_eq!(select_backend(&None), Backend::RustReference);
}

#[test]
fn test_select_backend_sm86() {
    let gpu = Some(make_gpu("RTX A6000", 8, 6));
    assert_eq!(select_backend(&gpu), Backend::CudaNative);
}

#[test]
fn test_select_backend_sm89() {
    let gpu = Some(make_gpu("RTX 4090", 8, 9));
    assert_eq!(select_backend(&gpu), Backend::CudaNative);
}

#[test]
fn test_select_backend_sm90() {
    let gpu = Some(make_gpu("H100", 9, 0));
    assert_eq!(select_backend(&gpu), Backend::CudaNative);
}

#[test]
fn test_select_backend_future_arch() {
    // sm_100 — no explicit SASS, but PTX fallback covers it
    let gpu = Some(make_gpu("Future GPU", 10, 0));
    assert_eq!(select_backend(&gpu), Backend::CudaPtx);
}

#[test]
fn test_select_backend_old_gpu() {
    // sm_75 (Turing) — below minimum, no PTX compiled for it
    let gpu = Some(make_gpu("RTX 2080", 7, 5));
    assert_eq!(select_backend(&gpu), Backend::RustReference);
}

#[test]
fn test_select_backend_sm80() {
    // sm_80 (A100) — below minimum sm_86
    let gpu = Some(make_gpu("A100", 8, 0));
    assert_eq!(select_backend(&gpu), Backend::RustReference);
}

// ══════════════════════════════════════════════════════════════════════
// Force override tests (always run — tests the atomic flag)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_force_rust_reference_flag() {
    // Reset to default
    force_rust_reference(false);
    assert!(!is_rust_forced());

    force_rust_reference(true);
    assert!(is_rust_forced());

    force_rust_reference(false);
    assert!(!is_rust_forced());
}

// ══════════════════════════════════════════════════════════════════════
// GPU detection tests (CUDA only)
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use nl_hecate_core::dispatch::delta_forward_dispatch;

    #[test]
    fn test_detect_gpu_returns_info() {
        let gpu = detect_gpu();
        assert!(gpu.is_some(), "Expected at least one GPU when compiled with --features cuda");
    }

    #[test]
    fn test_gpu_info_fields() {
        let gpu = detect_gpu().expect("No GPU detected");
        assert!(!gpu.name.is_empty(), "GPU name should not be empty");
        assert!(gpu.compute_major >= 8, "Expected compute major >= 8, got {}", gpu.compute_major);
        assert!(gpu.sm_version >= 86, "Expected sm_version >= 86, got {}", gpu.sm_version);
    }

    #[test]
    fn test_select_backend_real_gpu() {
        let gpu = detect_gpu();
        let backend = select_backend(&gpu);
        // Our hardware is sm_86 and sm_89, both should be CudaNative
        assert_eq!(backend, Backend::CudaNative,
                   "Expected CudaNative for detected GPU {:?}", gpu);
    }

    #[test]
    fn test_force_rust_overrides_cuda_dispatch() {
        // Run a small Delta forward through CUDA, then force Rust and verify
        // both paths produce the same result.
        let d = 4;
        let seq_len = 2;
        let dd = d * d;
        let k_mem = vec![0.1f32; seq_len * d];
        let v_mem = vec![0.2f32; seq_len * d];
        let q_mem = vec![0.05f32; seq_len * d];
        let alpha = vec![0.01f32; seq_len];
        let theta = vec![0.1f32; seq_len];
        let m_initial = vec![0.0f32; dd];

        // CUDA path
        force_rust_reference(false);
        let mut m_states_cuda = vec![0.0f32; (seq_len + 1) * dd];
        let mut y_cuda = vec![0.0f32; seq_len * d];
        delta_forward_dispatch(
            &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
            &mut m_states_cuda, &mut y_cuda, seq_len, d,
        );

        // Force Rust path
        force_rust_reference(true);
        let mut m_states_rust = vec![0.0f32; (seq_len + 1) * dd];
        let mut y_rust = vec![0.0f32; seq_len * d];
        delta_forward_dispatch(
            &k_mem, &v_mem, &q_mem, &alpha, &theta, &m_initial,
            &mut m_states_rust, &mut y_rust, seq_len, d,
        );

        // Reset
        force_rust_reference(false);

        // Both should produce very close results (< 1e-5 per element)
        for i in 0..y_cuda.len() {
            let diff = (y_cuda[i] - y_rust[i]).abs();
            assert!(diff < 1e-5,
                    "y[{i}] CUDA={} Rust={} diff={}", y_cuda[i], y_rust[i], diff);
        }
    }
}
