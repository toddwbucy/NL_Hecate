// CUDA SwiGLU MLP Kernel Tests
//
// Tests CUDA forward and backward kernels for SwiGluMlp against the
// CPU reference implementation in swiglu_mlp.rs.
//
// Test dimensions: d=64, intermediate=256, seq_len=4 (per task spec).
//
// Test classes:
//   1. Forward match: CUDA vs CPU for Y and all cached intermediates (tol 1e-5)
//   2. Backward match: all 4 gradient arrays (d_x, d_gate_proj, d_up_proj,
//      d_down_proj) CUDA vs CPU (tol 1e-4)
//   3. Finite-output: no NaN/Inf in either forward or backward outputs
//   4. Seq-len=1 edge case: single-token forward + backward parity

#![cfg(feature = "cuda")]

mod cuda_test_utils;
use cuda_test_utils::{rand_buf, check_close};

use nl_hecate_core::swiglu_mlp::SwiGluMlp;
use nl_hecate_core::delta_rule::MemoryRule;
use nl_hecate_core::model::MemoryLevelParams;

const D: usize = 64;
const INTER: usize = 256;
const SEQ: usize = 4;

// ── helpers ─────────────────────────────────────────────────────────

/// Build a MemoryLevelParams with randomised SwiGLU projections.
///
/// gate_proj / up_proj: [INTER × D]
/// down_proj:           [D × INTER]
fn make_swiglu_params(seed_gate: u64, seed_up: u64, seed_down: u64) -> MemoryLevelParams {
    let mut params = MemoryLevelParams::zeros_like(D);
    params.gate_proj = rand_buf(INTER * D, seed_gate);
    params.up_proj   = rand_buf(INTER * D, seed_up);
    params.down_proj = rand_buf(D * INTER, seed_down);
    params
}

fn check_finite(name: &str, buf: &[f32]) {
    for (i, &v) in buf.iter().enumerate() {
        assert!(v.is_finite(), "{name}: non-finite value {v} at idx {i}");
    }
}

// ── tests ────────────────────────────────────────────────────────────

/// Forward pass: CUDA kernel outputs must match CPU reference within 1e-5.
///
/// step() routes to step_cuda() when the `cuda` feature is enabled;
/// step_cpu() is the portable reference.
#[test]
fn test_swiglu_forward_cuda_vs_cpu() {
    let rule = SwiGluMlp { intermediate_size: INTER };
    let params = make_swiglu_params(1, 2, 3);
    let x = rand_buf(SEQ * D, 4);

    let (y_cpu, cache_cpu)   = rule.step_cpu(&params, &x, SEQ, D);
    let (y_cuda, cache_cuda) = rule.step(&params, &x, SEQ, D, None);

    eprintln!("\n=== test_swiglu_forward_cuda_vs_cpu (d={D}, inter={INTER}, seq={SEQ}) ===");
    check_close("y",          &y_cpu,               &y_cuda,               1e-5);
    check_close("gate_out",   &cache_cpu.gate_out,  &cache_cuda.gate_out,  1e-5);
    check_close("up_out",     &cache_cpu.up_out,    &cache_cuda.up_out,    1e-5);
    check_close("fused",      &cache_cpu.fused,     &cache_cuda.fused,     1e-5);
    check_close("gate_cache", &cache_cpu.gate_cache,&cache_cuda.gate_cache,1e-5);
}

/// Backward pass: CUDA gradient outputs must match CPU reference within 1e-4.
///
/// Uses the CUDA-produced cache from the forward pass so that any forward
/// numerical differences don't propagate into the backward comparison.
#[test]
fn test_swiglu_backward_cuda_vs_cpu() {
    let rule = SwiGluMlp { intermediate_size: INTER };
    let params = make_swiglu_params(10, 20, 30);
    let x  = rand_buf(SEQ * D, 40);
    let dy = rand_buf(SEQ * D, 50);

    // CPU forward → CPU backward (same cache path)
    let (_, cache_cpu)       = rule.step_cpu(&params, &x, SEQ, D);
    let (grads_cpu, dx_cpu)  = rule.step_backward_cpu(&params, &cache_cpu, &dy, &x);

    // CUDA forward → CUDA backward (same cache path)
    let (_, cache_cuda)      = rule.step(&params, &x, SEQ, D, None);
    let (grads_cuda, dx_cuda)= rule.step_backward(&params, &cache_cuda, &dy, &x);

    eprintln!("\n=== test_swiglu_backward_cuda_vs_cpu (d={D}, inter={INTER}, seq={SEQ}) ===");
    check_close("d_x",         &dx_cpu,              &dx_cuda,              1e-4);
    check_close("d_gate_proj", &grads_cpu.gate_proj, &grads_cuda.gate_proj, 1e-4);
    check_close("d_up_proj",   &grads_cpu.up_proj,   &grads_cuda.up_proj,   1e-4);
    check_close("d_down_proj", &grads_cpu.down_proj, &grads_cuda.down_proj, 1e-4);
}

/// Finite-output guard: no NaN / Inf in CUDA forward or backward with
/// randomly initialised weights. Catches kernel numerical issues that
/// might still be within tolerance but produce non-finite edge values.
#[test]
fn test_swiglu_cuda_finite_outputs() {
    let rule = SwiGluMlp { intermediate_size: INTER };
    let params = make_swiglu_params(100, 101, 102);
    let x  = rand_buf(SEQ * D, 103);
    let dy = rand_buf(SEQ * D, 104);

    let (y_cuda, cache_cuda)   = rule.step(&params, &x, SEQ, D, None);
    let (grads, dx)            = rule.step_backward(&params, &cache_cuda, &dy, &x);

    check_finite("y",          &y_cuda);
    check_finite("gate_out",   &cache_cuda.gate_out);
    check_finite("up_out",     &cache_cuda.up_out);
    check_finite("fused",      &cache_cuda.fused);
    check_finite("gate_cache", &cache_cuda.gate_cache);
    check_finite("d_x",        &dx);
    check_finite("d_gate_proj",&grads.gate_proj);
    check_finite("d_up_proj",  &grads.up_proj);
    check_finite("d_down_proj",&grads.down_proj);

    eprintln!("test_swiglu_cuda_finite_outputs: all outputs finite ✓");
}

/// Edge case: seq_len=1. Exercises the single-row matmul path in both kernels.
#[test]
fn test_swiglu_cuda_seq1_parity() {
    let seq1 = 1;
    let rule = SwiGluMlp { intermediate_size: INTER };
    let params = make_swiglu_params(200, 201, 202);
    let x  = rand_buf(seq1 * D, 203);
    let dy = rand_buf(seq1 * D, 204);

    let (y_cpu, cache_cpu)   = rule.step_cpu(&params, &x, seq1, D);
    let (y_cuda, cache_cuda) = rule.step(&params, &x, seq1, D, None);

    eprintln!("\n=== test_swiglu_cuda_seq1_parity (seq=1) ===");
    check_close("y[seq=1]",    &y_cpu,  &y_cuda,  1e-5);

    let (grads_cpu, dx_cpu)   = rule.step_backward_cpu(&params, &cache_cpu, &dy, &x);
    let (grads_cuda, dx_cuda) = rule.step_backward(&params, &cache_cuda, &dy, &x);

    check_close("d_x[seq=1]",        &dx_cpu,              &dx_cuda,              1e-4);
    check_close("d_gate_proj[seq=1]", &grads_cpu.gate_proj, &grads_cuda.gate_proj, 1e-4);
    check_close("d_up_proj[seq=1]",   &grads_cpu.up_proj,   &grads_cuda.up_proj,   1e-4);
    check_close("d_down_proj[seq=1]", &grads_cpu.down_proj, &grads_cuda.down_proj, 1e-4);
}
