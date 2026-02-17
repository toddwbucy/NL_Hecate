// cuBLAS Matmul Parity Tests — S2-M1b
//
// Verifies cuBLAS sgemm produces results matching the Rust reference
// for all three dispatch functions: matmul, matmul_acc, matmul_transb.
//
// Run with:
//   CUDA_PATH=/usr/local/cuda-12.8 cargo test --features cuda --test test_cublas -- --nocapture

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::dispatch::{
    matmul_dispatch, matmul_acc_dispatch, matmul_transb_dispatch, force_rust_reference,
};
use nl_hecate_core::tensor::{matmul_f32, matmul_acc_f32, transpose_f32};
use serial_test::serial;

/// Small exact test: 4×4 @ 4×4. Results should match within fp32 rounding.
#[test]
#[serial]
fn test_cublas_matmul_small() {
    let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let b: Vec<f32> = (17..=32).map(|x| x as f32).collect();

    let mut ref_out = vec![0.0f32; 16];
    matmul_f32(&a, &b, &mut ref_out, 4, 4, 4);

    force_rust_reference(false);
    let mut gpu_out = vec![0.0f32; 16];
    matmul_dispatch(&a, &b, &mut gpu_out, 4, 4, 4);

    check_close("cublas_small_4x4", &ref_out, &gpu_out, 1e-5);
}

/// Production-size QKV projection: 512×2048 @ 2048×2048.
/// This is the dominant matmul in the forward pass.
#[test]
#[serial]
fn test_cublas_matmul_production() {
    let m = 512;
    let k = 2048;
    let n = 2048;
    let a = rand_buf(m * k, 42);
    let b = rand_buf(k * n, 43);

    let mut ref_out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut ref_out, m, k, n);

    force_rust_reference(false);
    let mut gpu_out = vec![0.0f32; m * n];
    matmul_dispatch(&a, &b, &mut gpu_out, m, k, n);

    // Wider tolerance for large matrices: 2048-wide dot products accumulate more rounding.
    check_close("cublas_production_512x2048x2048", &ref_out, &gpu_out, 1e-3);
}

/// Accumulate mode: C += A @ B. Verifies the += semantics are preserved.
#[test]
#[serial]
fn test_cublas_matmul_acc() {
    let m = 64;
    let k = 64;
    let n = 64;
    let a = rand_buf(m * k, 100);
    let b = rand_buf(k * n, 101);
    let c_init = rand_buf(m * n, 102);

    let mut ref_out = c_init.clone();
    matmul_acc_f32(&a, &b, &mut ref_out, m, k, n);

    force_rust_reference(false);
    let mut gpu_out = c_init.clone();
    matmul_acc_dispatch(&a, &b, &mut gpu_out, m, k, n);

    check_close("cublas_acc_64x64", &ref_out, &gpu_out, 1e-4);
}

/// Fused transpose-matmul: C = A @ B^T where B is stored as [n,k].
/// This eliminates separate transpose_f32 + matmul_f32 in the forward pass.
#[test]
#[serial]
fn test_cublas_matmul_transb() {
    let m = 128;
    let k = 64;
    let n = 64;
    let a = rand_buf(m * k, 200);
    let b = rand_buf(n * k, 201); // B is [n,k], we want A @ B^T

    // Reference: explicit transpose + matmul
    let mut bt = vec![0.0f32; k * n];
    transpose_f32(&b, &mut bt, n, k);
    let mut ref_out = vec![0.0f32; m * n];
    matmul_f32(&a, &bt, &mut ref_out, m, k, n);

    // cuBLAS fused
    force_rust_reference(false);
    let mut gpu_out = vec![0.0f32; m * n];
    matmul_transb_dispatch(&a, &b, &mut gpu_out, m, k, n);

    check_close("cublas_transb_128x64x64", &ref_out, &gpu_out, 1e-5);
}

/// Unembed matmul: 512×2048 @ 2048×256. Non-square, tests rectangular dispatch.
#[test]
#[serial]
fn test_cublas_matmul_unembed() {
    let m = 512;
    let k = 2048;
    let n = 256;
    let a = rand_buf(m * k, 300);
    let b = rand_buf(k * n, 301);

    let mut ref_out = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut ref_out, m, k, n);

    force_rust_reference(false);
    let mut gpu_out = vec![0.0f32; m * n];
    matmul_dispatch(&a, &b, &mut gpu_out, m, k, n);

    check_close("cublas_unembed_512x2048x256", &ref_out, &gpu_out, 1e-3);
}
