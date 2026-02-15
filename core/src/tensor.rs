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
// Kernel-pair registry (CUDA):
//   Each Rust reference impl below may have a CUDA forward + backward kernel.
//   Backward kernels compute correct analytical gradients matching the
//   Rust signatures and row-major memory layout. See specs/infrastructure/
//   00_enzyme_integration.md for the kernel-pair contract.
//
//   | Rust reference         | CUDA kernel pair            | Status       |
//   |------------------------|-----------------------------|--------------|
//   | swa::swa_forward       | swa_forward_f32_cuda        | ✓ Phase 2    |
//   | swa::swa_backward_rust | swa_backward_f32_cuda       | ✓ Phase 2    |
//   | matmul_f32             | matmul_f32_cuda_fwd/bwd     | pending      |
//   | matmul_acc_f32         | matmul_acc_f32_cuda_fwd/bwd | pending      |
//   | transpose_f32          | transpose_f32_cuda_fwd/bwd  | pending      |
//   | softmax_f32            | softmax_f32_cuda_fwd/bwd    | pending      |

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

/// Sigmoid: 1 / (1 + exp(-x)). Clamped to avoid overflow.
#[inline]
pub fn sigmoid_f32(x: f32) -> f32 {
    if x >= 15.0 { return 1.0; }
    if x <= -15.0 { return 0.0; }
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x). Smooth ReLU variant.
#[inline]
pub fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

/// SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).
#[inline]
pub fn silu_prime_f32(x: f32) -> f32 {
    let s = sigmoid_f32(x);
    s + x * s * (1.0 - s)
}

/// Softplus: ln(1 + exp(x)). Numerically stable.
#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    if x >= 15.0 { return x; }
    if x <= -15.0 { return 0.0; }
    (1.0 + x.exp()).ln()
}

/// Element-wise natural log with floor clamp to avoid log(0).
/// Computes out[i] = ln(max(a[i], 1e-8)).
pub fn log_f32(a: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].max(1e-8).ln();
    }
}

/// Outer product: out[d1, d2] = a[d1] * b[d2]. Row-major.
/// `out` must be pre-allocated with d1*d2 elements (will be overwritten).
pub fn outer_product_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let d1 = a.len();
    let d2 = b.len();
    debug_assert_eq!(out.len(), d1 * d2);
    for i in 0..d1 {
        for j in 0..d2 {
            out[i * d2 + j] = a[i] * b[j];
        }
    }
}

/// L2 norm of a vector: sqrt(sum(a[i]^2)).
pub fn vec_norm_f32(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize vector in-place to unit length. No-op if norm < eps.
pub fn vec_normalize_f32(a: &mut [f32]) {
    let norm = vec_norm_f32(a);
    if norm > 1e-8 {
        let inv = 1.0 / norm;
        for x in a.iter_mut() {
            *x *= inv;
        }
    }
}

/// Frobenius dot product: sum_ij A[i,j] * B[i,j].
/// Both A and B are flat slices of the same length.
pub fn frobenius_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
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

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid_f32(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid_f32(100.0) - 1.0).abs() < 1e-6);
        assert!((sigmoid_f32(-100.0) - 0.0).abs() < 1e-6);
        // sigmoid(3.0) ≈ 0.9526
        assert!((sigmoid_f32(3.0) - 0.9526).abs() < 0.001);
    }

    #[test]
    fn test_softplus() {
        // softplus(0) = ln(2) ≈ 0.6931
        assert!((softplus_f32(0.0) - 0.6931).abs() < 0.001);
        // softplus(large) ≈ large
        assert!((softplus_f32(20.0) - 20.0).abs() < 0.01);
        // softplus(-large) ≈ 0
        assert!(softplus_f32(-20.0) < 1e-6);
        // softplus(-4.6) ≈ 0.01
        assert!((softplus_f32(-4.6) - 0.01).abs() < 0.002);
    }

    #[test]
    fn test_outer_product() {
        let a = [1.0, 2.0, 3.0f32];
        let b = [4.0, 5.0f32];
        let mut out = [0.0f32; 6];
        outer_product_f32(&a, &b, &mut out);
        assert_eq!(out, [4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_frobenius_dot() {
        let a = [1.0, 2.0, 3.0, 4.0f32];
        let b = [5.0, 6.0, 7.0, 8.0f32];
        let dot = frobenius_dot_f32(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        assert!((dot - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0 * 0.5 = 0
        assert!((silu_f32(0.0) - 0.0).abs() < 1e-6);
        // silu(large) ≈ large (sigmoid → 1)
        assert!((silu_f32(10.0) - 10.0).abs() < 0.01);
        // silu(-large) ≈ 0 (sigmoid → 0)
        assert!(silu_f32(-10.0).abs() < 0.01);
        // silu(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7311
        assert!((silu_f32(1.0) - 0.7311).abs() < 0.001);
    }

    #[test]
    fn test_silu_prime() {
        // Numerical derivative check: silu'(x) ≈ (silu(x+eps) - silu(x-eps)) / (2*eps)
        for &x in &[-2.0f32, -1.0, 0.0, 0.5, 1.0, 3.0] {
            let eps = 1e-4;
            let numerical = (silu_f32(x + eps) - silu_f32(x - eps)) / (2.0 * eps);
            let analytical = silu_prime_f32(x);
            assert!((analytical - numerical).abs() < 1e-3,
                "silu_prime({x}): analytical={analytical}, numerical={numerical}");
        }
    }

    #[test]
    fn test_log_f32_basic() {
        let a = [1.0f32, std::f32::consts::E, 10.0];
        let mut out = [0.0f32; 3];
        log_f32(&a, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6, "ln(1) should be 0, got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "ln(e) should be 1, got {}", out[1]);
        assert!((out[2] - 10.0f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_log_f32_clamps_near_zero() {
        // Zero and negative values should be clamped to ln(1e-8) ≈ -18.42
        let a = [0.0f32, -1.0, 1e-10];
        let mut out = [0.0f32; 3];
        log_f32(&a, &mut out);
        let expected = 1e-8f32.ln();
        for i in 0..3 {
            assert!(out[i].is_finite(), "log_f32 output[{i}] should be finite");
            assert!(out[i] <= expected + 1e-4, "log_f32 output[{i}]={} should be <= {expected}", out[i]);
        }
    }

    #[test]
    fn test_vec_norm_basic() {
        let a = [3.0f32, 4.0];
        assert!((vec_norm_f32(&a) - 5.0).abs() < 1e-6);
        assert!((vec_norm_f32(&[0.0f32; 4]) - 0.0).abs() < 1e-8);
        assert!((vec_norm_f32(&[1.0f32]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec_normalize_basic() {
        let mut a = [3.0f32, 4.0];
        vec_normalize_f32(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6, "a[0]={}", a[0]);
        assert!((a[1] - 0.8).abs() < 1e-6, "a[1]={}", a[1]);
        let norm = vec_norm_f32(&a);
        assert!((norm - 1.0).abs() < 1e-6, "norm after normalize: {norm}");
    }

    #[test]
    fn test_vec_normalize_zero() {
        let mut a = [0.0f32; 4];
        vec_normalize_f32(&mut a);
        assert!(a.iter().all(|&x| x == 0.0), "Zero vec should stay zero");
    }

    #[test]
    fn test_vec_normalize_already_unit() {
        let mut a = [1.0f32, 0.0, 0.0];
        vec_normalize_f32(&mut a);
        assert!((a[0] - 1.0).abs() < 1e-6);
        assert!(a[1].abs() < 1e-6);
        assert!(a[2].abs() < 1e-6);
    }

    #[test]
    fn test_silu_monotonic_for_positive() {
        // SiLU is monotonically increasing for x > 0
        let mut prev = silu_f32(0.01);
        for i in 1..100 {
            let x = i as f32 * 0.1;
            let y = silu_f32(x);
            assert!(y >= prev, "SiLU not monotonic at x={x}: {y} < {prev}");
            prev = y;
        }
    }
}
