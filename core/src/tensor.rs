/// Minimal tensor utilities for Track Zero-A.
///
/// All operations are free functions on flat f32 slices with explicit dimensions.
/// No generics, no traits on Tensor — keeps Enzyme compatibility straightforward.
/// Row-major layout throughout.

/// Flat f32 tensor with shape metadata.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

// ── bf16 conversion helpers ──────────────────────────────────────────
//
// Rust has no native bf16. These truncate f32 → bf16 → f32 by zeroing
// the low 16 mantissa bits. Used at storage boundaries (Q/K/V/attn_weights)
// in the CUDA path; the Rust reference path stays f32 for FD-checkable
// gradients. Phase 2 CUDA kernels will store in actual bf16.

/// Truncate f32 to bf16 precision (round to nearest even).
#[inline]
pub fn f32_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    // Round to nearest even: add 0x7FFF + bit 16 (the "round" bit)
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    f32::from_bits(round & 0xFFFF_0000)
}

/// Truncate a slice to bf16 precision in-place.
#[allow(dead_code)]
pub fn truncate_to_bf16(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = f32_to_bf16(*v);
    }
}

// ── Free-function math ops on flat slices ────────────────────────────
//
// Kernel-pair registry (Phase 2 — CUDA):
//   Each Rust reference impl below requires a CUDA forward + backward kernel.
//   Status: all pending. Kernel symbols reserved for dispatch.rs.
//
//   | Rust reference     | CUDA forward              | CUDA backward              |
//   |--------------------|---------------------------|----------------------------|
//   | matmul_f32         | matmul_f32_cuda_fwd       | matmul_f32_cuda_bwd        |
//   | matmul_acc_f32     | matmul_acc_f32_cuda_fwd   | matmul_acc_f32_cuda_bwd    |
//   | transpose_f32      | transpose_f32_cuda_fwd    | transpose_f32_cuda_bwd     |
//   | softmax_f32        | softmax_f32_cuda_fwd      | softmax_f32_cuda_bwd       |
//
//   Backward kernels must compute correct analytical gradients matching the
//   Rust signatures and row-major memory layout. See specs/infrastructure/
//   00_enzyme_integration.md for the kernel-pair contract.

/// Matrix multiply: C[M,N] = A[M,K] @ B[K,N].  Row-major.
/// `out` must be pre-allocated with M*N elements (will be overwritten).
pub fn matmul_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Matrix multiply with accumulation: C[M,N] += A[M,K] @ B[K,N].
pub fn matmul_acc_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(out.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] += sum;
        }
    }
}

/// Transpose A[M,K] → out[K,M].
pub fn transpose_f32(a: &[f32], out: &mut [f32], m: usize, k: usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(out.len(), k * m);

    for i in 0..m {
        for j in 0..k {
            out[j * m + i] = a[i * k + j];
        }
    }
}

/// Row-wise softmax: each row of length `cols` in `scores` gets softmaxed into `out`.
/// `rows` * `cols` elements.
pub fn softmax_f32(scores: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(scores.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);

    for r in 0..rows {
        let base = r * cols;
        let row = &scores[base..base + cols];

        // Numerically stable: subtract max
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for c in 0..cols {
            let e = (row[c] - max_val).exp();
            out[base + c] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            for c in 0..cols {
                out[base + c] /= sum_exp;
            }
        }
    }
}

/// Element-wise add: out[i] = a[i] + b[i].
pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

/// Scale: out[i] = a[i] * scalar.
pub fn scale_f32(a: &[f32], scalar: f32, out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] * scalar;
    }
}

/// Cross-entropy loss for next-token prediction.
/// `logits`: [seq_len, vocab_size], `targets`: [seq_len] (token indices).
/// Returns average -log(softmax(logit[t, target[t]])) over valid positions.
pub fn cross_entropy_loss(logits: &[f32], targets: &[usize], seq_len: usize, vocab_size: usize) -> f32 {
    debug_assert_eq!(logits.len(), seq_len * vocab_size);
    debug_assert_eq!(targets.len(), seq_len);

    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for t in 0..seq_len {
        let base = t * vocab_size;
        let row = &logits[base..base + vocab_size];
        let target = targets[t];
        if target >= vocab_size {
            continue;
        }

        // log-softmax: log(exp(x_t) / sum(exp(x)))
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for v in 0..vocab_size {
            sum_exp += (row[v] - max_val).exp();
        }
        let log_softmax = (row[target] - max_val) - sum_exp.ln();
        total_loss -= log_softmax;
        count += 1;
    }

    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

/// Simple xorshift64 PRNG for deterministic weight init. Not crypto-safe.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed.max(1) } // avoid zero state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform in [-scale, scale].
    pub fn uniform(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() as f64) / (u64::MAX as f64);
        (2.0 * u as f32 - 1.0) * scale
    }

    /// Fill slice with uniform random values in [-scale, scale].
    pub fn fill_uniform(&mut self, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *v = self.uniform(scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [1.0, 2.0, 3.0, 4.0f32];
        let mut out = [0.0f32; 4];
        matmul_f32(&a, &b, &mut out, 2, 2, 2);
        assert_eq!(out, b);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
        let mut out = [0.0f32; 4];
        matmul_f32(&a, &b, &mut out, 2, 3, 2);
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let mut out = [0.0f32; 6];
        transpose_f32(&a, &mut out, 2, 3);
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_softmax_single_row() {
        let scores = [1.0, 2.0, 3.0f32];
        let mut out = [0.0f32; 3];
        softmax_f32(&scores, &mut out, 1, 3);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_softmax_uniform() {
        let scores = [5.0, 5.0, 5.0, 5.0f32];
        let mut out = [0.0f32; 4];
        softmax_f32(&scores, &mut out, 1, 4);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_two_rows() {
        let scores = [0.0, 1.0, 1.0, 0.0f32];
        let mut out = [0.0f32; 4];
        softmax_f32(&scores, &mut out, 2, 2);
        assert!((out[0] + out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] + out[3] - 1.0).abs() < 1e-6);
        assert!(out[0] < out[1]);
        assert!(out[2] > out[3]);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let mut logits = vec![0.0f32; 4];
        logits[0] = 10.0; logits[1] = -10.0;
        logits[2] = -10.0; logits[3] = 10.0;
        let targets = [0usize, 1];
        let loss = cross_entropy_loss(&logits, &targets, 2, 2);
        assert!(loss < 0.001, "Perfect prediction should have near-zero loss, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_uniform() {
        let logits = vec![0.0f32; 8];
        let targets = [0usize, 2];
        let loss = cross_entropy_loss(&logits, &targets, 2, 4);
        let expected = (4.0f32).ln();
        assert!((loss - expected).abs() < 0.01,
            "Uniform logits should give loss ≈ ln(V)={}, got {}", expected, loss);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_bf16_truncation() {
        // bf16 has 7-bit mantissa → ~2 decimal digits
        let x = 1.234567f32;
        let bf = f32_to_bf16(x);
        assert!((bf - x).abs() < 0.01, "bf16({x}) = {bf}, expected within 0.01");
        // bf16(0) = 0
        assert_eq!(f32_to_bf16(0.0), 0.0);
        // Negative values
        let neg = f32_to_bf16(-3.14159);
        assert!((neg - (-3.14159)).abs() < 0.03);
        // Subnormals flush toward zero
        let tiny = f32_to_bf16(1e-40);
        assert!(tiny.abs() < 1e-37);
    }

    #[test]
    fn test_rng_fill_range() {
        let mut rng = SimpleRng::new(123);
        let mut buf = vec![0.0f32; 1000];
        rng.fill_uniform(&mut buf, 0.1);
        for &v in &buf {
            assert!(v >= -0.1 && v <= 0.1, "Value {} out of range", v);
        }
    }
}
