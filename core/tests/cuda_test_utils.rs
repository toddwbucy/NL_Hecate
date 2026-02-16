// Shared test utilities for CUDA kernel tests.
//
// Provides random buffer generation and tolerance-checked comparison
// used across all CUDA test files.

#![allow(dead_code)]

use nl_hecate_core::tensor::SimpleRng;

pub fn rand_buf(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = SimpleRng::new(seed);
    let mut buf = vec![0.0f32; len];
    rng.fill_uniform(&mut buf, 0.1);
    buf
}

pub fn check_close(name: &str, a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch {} vs {}", a.len(), b.len());
    if a.is_empty() {
        return;
    }
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        assert!(
            diff.is_finite(),
            "{name}: non-finite diff at idx {i} (a={:.6e}, b={:.6e})",
            a[i], b[i]
        );
        if diff > max_diff { max_diff = diff; max_idx = i; }
    }
    assert!(
        max_diff < tol,
        "{name}: max diff {max_diff:.6e} at idx {max_idx} (a={:.6e}, b={:.6e}), tol={tol:.0e}",
        a[max_idx], b[max_idx]
    );
    eprintln!("  {name}: max_diff={max_diff:.6e} (tol={tol:.0e}) ✓");
}
