/// Associative Scan — exact parallelization for linear recurrences.
///
/// Blelloch parallel prefix sum: given s_t = a_t * s_{t-1} + b_t for all t,
/// compute all prefix states in O(n) work and O(log n) depth.
///
/// Applies to Hebbian (full: M update is linear in M) and Titans
/// (momentum S only: S update is linear in S, but M depends on S nonlinearly).

use crate::model::{MAGConfig, MemoryLevelParams};
use crate::delta_rule::MemoryRule;
use crate::hebbian_rule::HebbianRule;
use crate::tensor::{matmul_f32, transpose_f32, sigmoid_f32};
use crate::mag::MemoryCache;

// ═════════════════════════════════════════════════════════════════════
// Core associative scan: s_t = a_t * s_{t-1} + b_t
// ═════════════════════════════════════════════════════════════════════

/// A scan element: (a, b) where the recurrence is s' = a*s + b.
/// The associative operator is: (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
#[derive(Clone)]
struct ScanElement {
    /// Scalar decay factor
    a: f32,
    /// Additive term: [state_size]
    b: Vec<f32>,
}

/// Compose two scan elements: (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)
fn compose(left: &ScanElement, right: &ScanElement) -> ScanElement {
    let a = left.a * right.a;
    let b: Vec<f32> = left.b.iter().zip(right.b.iter())
        .map(|(l, r)| right.a * l + r)
        .collect();
    ScanElement { a, b }
}

/// Blelloch inclusive parallel prefix scan.
///
/// Given elements [(a_0, b_0), (a_1, b_1), ...], computes all prefix
/// compositions. With initial state s_init, the output states are:
///   s_1 = a_0 * s_init + b_0
///   s_2 = a_1 * s_1 + b_1 = a_1*a_0*s_init + a_1*b_0 + b_1
///   ...
///
/// Returns states [s_1, s_2, ..., s_n] as flat [n * state_size].
pub fn associative_scan(
    a_seq: &[f32],         // [n] scalar decay factors
    b_seq: &[f32],         // [n * state_size] additive terms
    s_init: &[f32],        // [state_size] initial state
    state_size: usize,
) -> Vec<f32> {
    let n = a_seq.len();
    assert_eq!(b_seq.len(), n * state_size);
    assert_eq!(s_init.len(), state_size);

    if n == 0 {
        return Vec::new();
    }

    // Build scan elements
    let mut elements: Vec<ScanElement> = (0..n).map(|t| {
        ScanElement {
            a: a_seq[t],
            b: b_seq[t * state_size..(t + 1) * state_size].to_vec(),
        }
    }).collect();

    // Up-sweep (reduce phase)
    let mut tree = elements.clone();
    let mut stride = 1;
    while stride < n {
        let mut i = 2 * stride - 1;
        while i < n {
            let left_idx = i - stride;
            let combined = compose(&tree[left_idx], &tree[i]);
            tree[i] = combined;
            i += 2 * stride;
        }
        stride *= 2;
    }

    // Down-sweep: propagate prefixes
    // The last element already has the full prefix.
    // We need to propagate partial prefixes down.
    stride /= 2;
    while stride >= 1 {
        let mut i = 3 * stride - 1;
        while i < n {
            let left_idx = i - stride;
            let combined = compose(&tree[left_idx], &tree[i]);
            tree[i] = combined;
            i += 2 * stride;
        }
        if stride == 1 { break; }
        stride /= 2;
    }

    // Apply initial state to get final states
    let mut states = vec![0.0f32; n * state_size];
    for t in 0..n {
        for j in 0..state_size {
            states[t * state_size + j] = tree[t].a * s_init[j] + tree[t].b[j];
        }
    }

    states
}

/// Backward through associative scan.
///
/// Given d_states [n * state_size], compute d_a_seq [n], d_b_seq [n * state_size],
/// d_s_init [state_size].
pub fn associative_scan_backward(
    a_seq: &[f32],
    _b_seq: &[f32],
    s_init: &[f32],
    states: &[f32],   // forward output [n * state_size]
    d_states: &[f32],  // upstream gradient [n * state_size]
    state_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = a_seq.len();

    let mut d_a = vec![0.0f32; n];
    let mut d_b = vec![0.0f32; n * state_size];
    let mut d_s_init = vec![0.0f32; state_size];

    // Reverse through the recurrence: s_t = a_t * s_{t-1} + b_t
    // ds_{t-1} += a_t * ds_t
    // da_t = ds_t . s_{t-1}
    // db_t = ds_t
    let mut d_s = vec![0.0f32; state_size]; // accumulated from upstream

    for t in (0..n).rev() {
        // Add upstream gradient for this step
        for j in 0..state_size {
            d_s[j] += d_states[t * state_size + j];
        }

        // db_t = d_s
        d_b[t * state_size..(t + 1) * state_size].copy_from_slice(&d_s);

        // s_{t-1}: the state BEFORE this step
        let s_prev = if t > 0 {
            &states[(t - 1) * state_size..t * state_size]
        } else {
            s_init
        };

        // da_t = dot(d_s, s_{t-1})
        let mut da_t = 0.0f32;
        for j in 0..state_size {
            da_t += d_s[j] * s_prev[j];
        }
        d_a[t] = da_t;

        // ds_{t-1} += a_t * d_s
        let a_t = a_seq[t];
        let mut d_s_new = vec![0.0f32; state_size];
        for j in 0..state_size {
            d_s_new[j] = a_t * d_s[j];
        }
        d_s = d_s_new;
    }

    // Final: d_s_init = remaining d_s
    d_s_init.copy_from_slice(&d_s);

    (d_a, d_b, d_s_init)
}

// ═════════════════════════════════════════════════════════════════════
// Hebbian scan: exact parallel forward via associative scan
// ═════════════════════════════════════════════════════════════════════

/// Cache for Hebbian associative scan forward.
pub struct HebbianScanCache {
    pub seq_len: usize,
    pub d: usize,
    /// All memory states [s_1..s_n] as flat [(seq_len) * d * d]
    pub m_states: Vec<f32>,
    /// Per-token projected keys: [seq_len, d]
    pub k_mem: Vec<f32>,
    /// Per-token projected values: [seq_len, d]
    pub v_mem: Vec<f32>,
    /// Per-token projected queries: [seq_len, d]
    pub q_mem: Vec<f32>,
    /// Pre-sigmoid alpha values: [seq_len]
    pub alpha_pre: Vec<f32>,
    /// Sigmoid alpha values (decay = 1-alpha): [seq_len]
    pub alpha: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
    /// Concatenated (k,v) for alpha computation: [seq_len, 2*d]
    pub concat_kv: Vec<f32>,
    /// Scan decay factors a_t = 1-alpha_t: [seq_len]
    pub a_seq: Vec<f32>,
    /// Scan additive terms (outer products): [seq_len * d * d]
    pub b_seq: Vec<f32>,
    /// Initial memory state: [d * d]
    pub m_init: Vec<f32>,
}

/// Hebbian forward using associative scan (exact parallel).
///
/// The Hebbian recurrence M_{t+1} = (1-α_t)·M_t + outer(v_t, k_t) is linear
/// in M, so it can be exactly computed via parallel prefix scan.
pub fn hebbian_scan_forward(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, HebbianScanCache) {
    let dd = d * d;
    debug_assert_eq!(embedded.len(), seq_len * d);

    // Project embedded → k, v, q
    let mut w_k_t = vec![0.0f32; dd];
    let mut w_v_t = vec![0.0f32; dd];
    let mut w_q_t = vec![0.0f32; dd];
    transpose_f32(&level_params.w_k_mem, &mut w_k_t, d, d);
    transpose_f32(&level_params.w_v_mem, &mut w_v_t, d, d);
    transpose_f32(&level_params.w_q_mem, &mut w_q_t, d, d);

    let mut k_mem = vec![0.0f32; seq_len * d];
    let mut v_mem = vec![0.0f32; seq_len * d];
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_k_t, &mut k_mem, seq_len, d, d);
    matmul_f32(embedded, &w_v_t, &mut v_mem, seq_len, d, d);
    matmul_f32(embedded, &w_q_t, &mut q_mem, seq_len, d, d);

    // Compute gates: alpha_t = sigmoid(concat(k,v) @ w_alpha + b_alpha)
    let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
    let mut alpha_pre = vec![0.0f32; seq_len];
    let mut alpha = vec![0.0f32; seq_len];

    for t in 0..seq_len {
        let c_base = t * 2 * d;
        concat_kv[c_base..c_base + d].copy_from_slice(&k_mem[t * d..(t + 1) * d]);
        concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(&v_mem[t * d..(t + 1) * d]);
        let concat_t = &concat_kv[c_base..c_base + 2 * d];

        let mut pre = level_params.b_alpha[0];
        for i in 0..(2 * d) {
            pre += concat_t[i] * level_params.w_alpha[i];
        }
        alpha_pre[t] = pre;
        alpha[t] = sigmoid_f32(pre);
    }

    // Build scan elements: a_t = 1-alpha_t, b_t = outer(v_t, k_t)
    let mut a_seq = vec![0.0f32; seq_len];
    let mut b_seq = vec![0.0f32; seq_len * dd];

    for t in 0..seq_len {
        a_seq[t] = 1.0 - alpha[t];
        let k_t = &k_mem[t * d..(t + 1) * d];
        let v_t = &v_mem[t * d..(t + 1) * d];
        for i in 0..d {
            for j in 0..d {
                b_seq[t * dd + i * d + j] = v_t[i] * k_t[j];
            }
        }
    }

    // Initial M
    let m_init = initial_m.unwrap_or_else(|| vec![0.0f32; dd]);
    debug_assert_eq!(m_init.len(), dd);

    // Run associative scan
    let m_states = associative_scan(&a_seq, &b_seq, &m_init, dd);
    assert_eq!(m_states.len(), seq_len * dd);

    // Compute outputs: y_t = M_{t+1} @ q_t (M_{t+1} = m_states[t])
    let mut y = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let m_t = &m_states[t * dd..(t + 1) * dd];
        let q_t = &q_mem[t * d..(t + 1) * d];
        matmul_f32(m_t, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
    }

    let cache = HebbianScanCache {
        seq_len, d, m_states: m_states.clone(), k_mem, v_mem, q_mem,
        alpha_pre, alpha, y: y.clone(), concat_kv, a_seq, b_seq, m_init,
    };

    (y, cache)
}

/// Hebbian backward using reverse-mode through associative scan.
pub fn hebbian_scan_backward(
    level_params: &MemoryLevelParams,
    cache: &HebbianScanCache,
    d_y: &[f32],
    embedded: &[f32],
) -> (MemoryLevelParams, Vec<f32>) {
    let s = cache.seq_len;
    let d = cache.d;
    let dd = d * d;

    // d_y_t = upstream gradient on y_t
    // y_t = M_{t+1} @ q_t → dM_{t+1} += dy_t @ q_t^T, dq_t += M_{t+1}^T @ dy_t
    let mut d_m_states = vec![0.0f32; s * dd];
    let mut d_q_mem = vec![0.0f32; s * d];

    for t in 0..s {
        let dy_t = &d_y[t * d..(t + 1) * d];
        let q_t = &cache.q_mem[t * d..(t + 1) * d];
        let m_t = &cache.m_states[t * dd..(t + 1) * dd];

        // dM += dy @ q^T
        for i in 0..d {
            for j in 0..d {
                d_m_states[t * dd + i * d + j] += dy_t[i] * q_t[j];
            }
        }

        // dq += M^T @ dy
        for i in 0..d {
            let mut sum = 0.0f32;
            for j in 0..d {
                sum += m_t[j * d + i] * dy_t[j];
            }
            d_q_mem[t * d + i] += sum;
        }
    }

    // Backward through associative scan
    let (d_a, d_b, _d_m_init) = associative_scan_backward(
        &cache.a_seq, &cache.b_seq, &cache.m_init, &cache.m_states,
        &d_m_states, dd,
    );

    // d_a -> d_alpha (a = 1-alpha, so d_alpha = -d_a)
    // d_b -> d_outer -> d_k, d_v
    let mut d_k_mem = vec![0.0f32; s * d];
    let mut d_v_mem = vec![0.0f32; s * d];
    let mut d_alpha = vec![0.0f32; s];

    for t in 0..s {
        d_alpha[t] = -d_a[t];
        let k_t = &cache.k_mem[t * d..(t + 1) * d];
        let v_t = &cache.v_mem[t * d..(t + 1) * d];
        let db_t = &d_b[t * dd..(t + 1) * dd];

        // b = outer(v, k) → dv[i] += sum_j db[i,j]*k[j], dk[j] += sum_i db[i,j]*v[i]
        for i in 0..d {
            for j in 0..d {
                d_v_mem[t * d + i] += db_t[i * d + j] * k_t[j];
                d_k_mem[t * d + j] += db_t[i * d + j] * v_t[i];
            }
        }
    }

    // alpha = sigmoid(pre) → d_pre = d_alpha * sigmoid'(pre) = d_alpha * alpha * (1-alpha)
    let mut d_alpha_pre = vec![0.0f32; s];
    for t in 0..s {
        d_alpha_pre[t] = d_alpha[t] * cache.alpha[t] * (1.0 - cache.alpha[t]);
    }

    // d_w_alpha, d_b_alpha from alpha_pre = concat @ w_alpha + b_alpha
    let mut d_w_alpha = vec![0.0f32; 2 * d];
    let mut d_b_alpha = vec![0.0f32; 1];
    let mut d_concat_kv = vec![0.0f32; s * 2 * d];

    for t in 0..s {
        let concat_t = &cache.concat_kv[t * 2 * d..(t + 1) * 2 * d];
        let dap = d_alpha_pre[t];
        d_b_alpha[0] += dap;
        for i in 0..(2 * d) {
            d_w_alpha[i] += dap * concat_t[i];
            d_concat_kv[t * 2 * d + i] += dap * level_params.w_alpha[i];
        }
    }

    // d_concat_kv -> additional d_k_mem, d_v_mem
    for t in 0..s {
        for i in 0..d {
            d_k_mem[t * d + i] += d_concat_kv[t * 2 * d + i];
            d_v_mem[t * d + i] += d_concat_kv[t * 2 * d + d + i];
        }
    }

    // Backward through projections: k = embedded @ W_K^T
    // d_W_K = embedded^T @ d_k_mem (then transpose)
    // d_embedded += d_k @ W_K + d_v @ W_V + d_q @ W_Q
    let mut d_w_k_mem = vec![0.0f32; dd];
    let mut d_w_v_mem = vec![0.0f32; dd];
    let mut d_w_q_mem = vec![0.0f32; dd];
    let mut d_embedded = vec![0.0f32; s * d];

    // d_W^T = embedded^T @ d_proj → d_W = (d_W^T)^T
    // For d_W_K: d(W_K^T) = embedded^T @ d_k_mem [d×d]
    let mut d_wkt = vec![0.0f32; dd];
    matmul_f32_at_b(embedded, &d_k_mem, &mut d_wkt, s, d, d);
    transpose_f32(&d_wkt, &mut d_w_k_mem, d, d);

    let mut d_wvt = vec![0.0f32; dd];
    matmul_f32_at_b(embedded, &d_v_mem, &mut d_wvt, s, d, d);
    transpose_f32(&d_wvt, &mut d_w_v_mem, d, d);

    let mut d_wqt = vec![0.0f32; dd];
    matmul_f32_at_b(embedded, &d_q_mem, &mut d_wqt, s, d, d);
    transpose_f32(&d_wqt, &mut d_w_q_mem, d, d);

    // d_embedded += d_k @ W_K + d_v @ W_V + d_q @ W_Q
    let mut tmp = vec![0.0f32; s * d];
    matmul_f32(&d_k_mem, &level_params.w_k_mem, &mut tmp, s, d, d);
    for i in 0..s * d { d_embedded[i] += tmp[i]; }
    matmul_f32(&d_v_mem, &level_params.w_v_mem, &mut tmp, s, d, d);
    for i in 0..s * d { d_embedded[i] += tmp[i]; }
    matmul_f32(&d_q_mem, &level_params.w_q_mem, &mut tmp, s, d, d);
    for i in 0..s * d { d_embedded[i] += tmp[i]; }

    let grads = MemoryLevelParams {
        w_k_mem: d_w_k_mem,
        w_v_mem: d_w_v_mem,
        w_q_mem: d_w_q_mem,
        w_alpha: d_w_alpha,
        b_alpha: d_b_alpha,
        w_theta: vec![],
        b_theta: vec![],
        w_eta: vec![],
        b_eta: vec![],
        w_omega: vec![0.0f32; level_params.w_omega.len()],
        w_freq: vec![],
        b_freq: vec![],
        w_k_conv: vec![],
        b_k_conv: vec![],
        w_q_conv: vec![],
        b_q_conv: vec![],
    };

    (grads, d_embedded)
}

// ═════════════════════════════════════════════════════════════════════
// Titans momentum scan: exact parallel for momentum S
// ═════════════════════════════════════════════════════════════════════

/// Compute Titans momentum states via associative scan.
///
/// S_t = eta_t * S_{t-1} - theta_t * grad_t
///
/// This is a linear recurrence in S:
///   a_t = eta_t, b_t = -theta_t * grad_t
pub fn titans_momentum_scan(
    etas: &[f32],       // [n] momentum retention factors
    thetas: &[f32],     // [n] inner-loop learning rates
    grads: &[f32],      // [n * state_size] memory gradients
    s_init: &[f32],     // [state_size] initial momentum
    state_size: usize,
) -> Vec<f32> {
    let n = etas.len();
    assert_eq!(thetas.len(), n);
    assert_eq!(grads.len(), n * state_size);

    // b_t = -theta_t * grad_t
    let mut b_seq = vec![0.0f32; n * state_size];
    for t in 0..n {
        for j in 0..state_size {
            b_seq[t * state_size + j] = -thetas[t] * grads[t * state_size + j];
        }
    }

    associative_scan(etas, &b_seq, s_init, state_size)
}

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

/// A^T @ B where A is [m, n] and B is [m, p], result is [n, p].
fn matmul_f32_at_b(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, p: usize) {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), m * p);
    debug_assert_eq!(out.len(), n * p);
    for i in 0..n {
        for j in 0..p {
            let mut sum = 0.0f32;
            for k in 0..m {
                sum += a[k * n + i] * b[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;

    // ═══════════════════════════════════════════════════════════════
    // Core associative scan tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_scan_identity() {
        // a_t = 1.0, b_t = 0.0 → s_t = s_init for all t
        let n = 5;
        let state_size = 3;
        let a_seq = vec![1.0f32; n];
        let b_seq = vec![0.0f32; n * state_size];
        let s_init = vec![1.0, 2.0, 3.0];

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);
        for t in 0..n {
            for j in 0..state_size {
                assert!((states[t * state_size + j] - s_init[j]).abs() < 1e-6,
                    "t={t} j={j}: expected {} got {}", s_init[j], states[t * state_size + j]);
            }
        }
    }

    #[test]
    fn test_scan_pure_additive() {
        // a_t = 0.0, b_t = t+1 → s_t = b_t (no accumulation)
        let n = 4;
        let state_size = 1;
        let a_seq = vec![0.0f32; n];
        let b_seq: Vec<f32> = (1..=n as u32).map(|t| t as f32).collect();
        let s_init = vec![100.0]; // should be ignored

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);
        for t in 0..n {
            let expected = (t + 1) as f32;
            assert!((states[t] - expected).abs() < 1e-5,
                "t={t}: expected {expected} got {}", states[t]);
        }
    }

    #[test]
    fn test_scan_decay() {
        // a_t = 0.5, b_t = 1.0, s_0 = 0 → s converges to 2.0
        let n = 20;
        let state_size = 1;
        let a_seq = vec![0.5f32; n];
        let b_seq = vec![1.0f32; n];
        let s_init = vec![0.0];

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);
        // s_1 = 1, s_2 = 1.5, s_3 = 1.75, ...
        assert!((states[0] - 1.0).abs() < 1e-5);
        assert!((states[1] - 1.5).abs() < 1e-5);
        // After 20 steps should be close to 2.0
        assert!((states[n - 1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_scan_matches_sequential() {
        // Verify scan matches naive sequential computation
        let n = 7;
        let state_size = 4;
        let mut rng = SimpleRng::new(42);
        let mut a_seq = vec![0.0f32; n];
        rng.fill_uniform(&mut a_seq, 1.0);
        let mut b_seq = vec![0.0f32; n * state_size];
        rng.fill_uniform(&mut b_seq, 1.0);
        let mut s_init = vec![0.0f32; state_size];
        rng.fill_uniform(&mut s_init, 1.0);

        // Scan
        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);

        // Sequential
        let mut s = s_init.clone();
        for t in 0..n {
            let mut s_new = vec![0.0f32; state_size];
            for j in 0..state_size {
                s_new[j] = a_seq[t] * s[j] + b_seq[t * state_size + j];
            }
            // Compare
            for j in 0..state_size {
                let diff = (states[t * state_size + j] - s_new[j]).abs();
                assert!(diff < 1e-4,
                    "t={t} j={j}: scan={} seq={} diff={diff}",
                    states[t * state_size + j], s_new[j]);
            }
            s = s_new;
        }
    }

    #[test]
    fn test_scan_power_of_two() {
        // n=8 (power of 2) — tests clean Blelloch tree
        let n = 8;
        let state_size = 2;
        let a_seq = vec![0.9f32; n];
        let mut b_seq = vec![0.0f32; n * state_size];
        for t in 0..n {
            b_seq[t * state_size] = 1.0;
            b_seq[t * state_size + 1] = 0.5;
        }
        let s_init = vec![0.0, 0.0];

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);

        // Verify sequentially
        let mut s = s_init.clone();
        for t in 0..n {
            let mut s_new = vec![0.0f32; state_size];
            for j in 0..state_size {
                s_new[j] = a_seq[t] * s[j] + b_seq[t * state_size + j];
            }
            for j in 0..state_size {
                assert!((states[t * state_size + j] - s_new[j]).abs() < 1e-4);
            }
            s = s_new;
        }
    }

    #[test]
    fn test_scan_non_power_of_two() {
        // n=5 (not a power of 2)
        let n = 5;
        let state_size = 3;
        let mut rng = SimpleRng::new(99);
        let mut a_seq = vec![0.0f32; n];
        rng.fill_uniform(&mut a_seq, 1.0);
        let mut b_seq = vec![0.0f32; n * state_size];
        rng.fill_uniform(&mut b_seq, 1.0);
        let mut s_init = vec![0.0f32; state_size];
        rng.fill_uniform(&mut s_init, 0.5);

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);

        let mut s = s_init.clone();
        for t in 0..n {
            let mut s_new = vec![0.0f32; state_size];
            for j in 0..state_size {
                s_new[j] = a_seq[t] * s[j] + b_seq[t * state_size + j];
            }
            for j in 0..state_size {
                assert!((states[t * state_size + j] - s_new[j]).abs() < 1e-4,
                    "t={t}: mismatch");
            }
            s = s_new;
        }
    }

    #[test]
    fn test_scan_backward_fd() {
        // FD check of scan backward
        let n = 4;
        let state_size = 2;
        let mut rng = SimpleRng::new(42);
        let mut a_seq = vec![0.0f32; n];
        rng.fill_uniform(&mut a_seq, 0.9);
        for v in &mut a_seq { *v = v.abs().min(0.99); }
        let mut b_seq = vec![0.0f32; n * state_size];
        rng.fill_uniform(&mut b_seq, 1.0);
        let mut s_init = vec![0.0f32; state_size];
        rng.fill_uniform(&mut s_init, 0.5);

        let states = associative_scan(&a_seq, &b_seq, &s_init, state_size);

        let d_states = vec![1.0f32; n * state_size];
        let (d_a, _d_b, d_s_init) = associative_scan_backward(
            &a_seq, &b_seq, &s_init, &states, &d_states, state_size,
        );

        let eps = 1e-3;
        // Check d_a
        for t in 0..n {
            let mut a_plus = a_seq.clone();
            a_plus[t] += eps;
            let states_p = associative_scan(&a_plus, &b_seq, &s_init, state_size);
            let loss_p: f32 = states_p.iter().sum();

            let mut a_minus = a_seq.clone();
            a_minus[t] -= eps;
            let states_m = associative_scan(&a_minus, &b_seq, &s_init, state_size);
            let loss_m: f32 = states_m.iter().sum();

            let fd = (loss_p - loss_m) / (2.0 * eps);
            let rel = (d_a[t] - fd).abs() / d_a[t].abs().max(fd.abs()).max(1e-8);
            assert!(rel < 0.1 || (d_a[t].abs() < 1e-4 && fd.abs() < 1e-4),
                "d_a[{t}]: analytical={} fd={fd} rel={rel}", d_a[t]);
        }

        // Check d_s_init
        for j in 0..state_size {
            let mut si_plus = s_init.clone();
            si_plus[j] += eps;
            let states_p = associative_scan(&a_seq, &b_seq, &si_plus, state_size);
            let loss_p: f32 = states_p.iter().sum();

            let mut si_minus = s_init.clone();
            si_minus[j] -= eps;
            let states_m = associative_scan(&a_seq, &b_seq, &si_minus, state_size);
            let loss_m: f32 = states_m.iter().sum();

            let fd = (loss_p - loss_m) / (2.0 * eps);
            let rel = (d_s_init[j] - fd).abs() / d_s_init[j].abs().max(fd.abs()).max(1e-8);
            assert!(rel < 0.1 || (d_s_init[j].abs() < 1e-4 && fd.abs() < 1e-4),
                "d_s_init[{j}]: analytical={} fd={fd}", d_s_init[j]);
        }
    }

    #[test]
    fn test_scan_single_element() {
        // n=1
        let a_seq = vec![0.8];
        let b_seq = vec![3.0, 4.0];
        let s_init = vec![1.0, 2.0];

        let states = associative_scan(&a_seq, &b_seq, &s_init, 2);
        assert!((states[0] - (0.8 * 1.0 + 3.0)).abs() < 1e-5);
        assert!((states[1] - (0.8 * 2.0 + 4.0)).abs() < 1e-5);
    }

    // ═══════════════════════════════════════════════════════════════
    // Hebbian scan tests
    // ═══════════════════════════════════════════════════════════════

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    #[test]
    fn test_hebbian_scan_matches_sequential() {
        // Hebbian scan should EXACTLY match sequential Hebbian.step()
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        // Scan
        let (y_scan, _) = hebbian_scan_forward(
            &params.levels[0], &embedded, s, d, None,
        );

        // Sequential
        let (y_seq, _) = HebbianRule.step(
            &params.levels[0], &embedded, s, d, None,
        );

        let max_diff: f32 = y_scan.iter().zip(y_seq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4,
            "Hebbian scan should match sequential, max_diff={max_diff}");
    }

    #[test]
    fn test_hebbian_scan_forward_finite() {
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (y, _) = hebbian_scan_forward(
            &params.levels[0], &embedded, s, d, None,
        );
        for &v in &y {
            assert!(v.is_finite(), "Hebbian scan output not finite");
        }
    }

    #[test]
    fn test_hebbian_scan_backward_shapes() {
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (_, cache) = hebbian_scan_forward(
            &params.levels[0], &embedded, s, d, None,
        );

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = hebbian_scan_backward(
            &params.levels[0], &cache, &d_y, &embedded,
        );

        assert_eq!(grads.w_k_mem.len(), d * d);
        assert_eq!(grads.w_v_mem.len(), d * d);
        assert_eq!(grads.w_q_mem.len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }

    #[test]
    fn test_hebbian_scan_backward_finite() {
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (_, cache) = hebbian_scan_forward(
            &params.levels[0], &embedded, s, d, None,
        );
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = hebbian_scan_backward(
            &params.levels[0], &cache, &d_y, &embedded,
        );

        for &v in grads.w_k_mem.iter().chain(grads.w_v_mem.iter()).chain(grads.w_q_mem.iter()) {
            assert!(v.is_finite(), "Hebbian scan backward gradient not finite");
        }
        for &v in &d_emb {
            assert!(v.is_finite(), "d_embedded not finite");
        }
    }

    #[test]
    fn test_hebbian_scan_fd_gradient() {
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let eps = 1e-2;

        let (_y, cache) = hebbian_scan_forward(
            &params.levels[0], &embedded, s, d, None,
        );
        let d_y = vec![1.0f32; s * d];
        let (grads, _) = hebbian_scan_backward(
            &params.levels[0], &cache, &d_y, &embedded,
        );

        // FD check on w_k_mem
        let n_check = 5.min(d * d);
        for idx in 0..n_check {
            let mut lp_plus = params.levels[0].clone();
            lp_plus.w_k_mem[idx] += eps;
            let (y_plus, _) = hebbian_scan_forward(&lp_plus, &embedded, s, d, None);
            let loss_plus: f32 = y_plus.iter().sum();

            let mut lp_minus = params.levels[0].clone();
            lp_minus.w_k_mem[idx] -= eps;
            let (y_minus, _) = hebbian_scan_forward(&lp_minus, &embedded, s, d, None);
            let loss_minus: f32 = y_minus.iter().sum();

            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grads.w_k_mem[idx];
            let denom = analytical.abs().max(fd.abs()).max(1e-8);
            let rel = (analytical - fd).abs() / denom;

            assert!(rel < 0.2 || (analytical.abs() < 5e-4 && fd.abs() < 5e-4),
                "FD check idx={idx}: analytical={analytical:.6e} fd={fd:.6e} rel={rel:.4}");
        }
    }

    #[test]
    fn test_hebbian_scan_outer_loop_weight_descent() {
        // Validates outer-loop gradient flow: tape-computed gradients on
        // projection weights (W_K, W_V, W_Q, W_alpha, b_alpha) decrease a proxy
        // loss when applied as weight updates. This is the outer loop — distinct
        // from the inner loop (memory updates inside the forward pass).
        let cfg = MAGConfig::hebbian_test_config();
        let mut level_params = MAGParams::init(&cfg, 42).levels.into_iter().next().unwrap();
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let lr = 0.1;

        let mut rng = SimpleRng::new(99);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 1.0);
        let mut target = vec![0.0f32; s * d];
        rng.fill_uniform(&mut target, 1.0);

        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for outer_step in 0..100 {
            let (y, cache) = hebbian_scan_forward(&level_params, &embedded, s, d, None);
            let loss: f32 = y.iter().zip(target.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
            if outer_step == 0 { first_loss = loss; }
            if outer_step == 99 { last_loss = loss; }

            let d_y: Vec<f32> = y.iter().zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
            let (grads, _) = hebbian_scan_backward(&level_params, &cache, &d_y, &embedded);

            // Outer-loop weight update (projection weights, not inner-loop memory)
            for (w, g) in level_params.w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
            for (w, g) in level_params.w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
            for (w, g) in level_params.w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
            for (w, g) in level_params.w_alpha.iter_mut().zip(grads.w_alpha.iter()) { *w -= lr * g; }
            for (w, g) in level_params.b_alpha.iter_mut().zip(grads.b_alpha.iter()) { *w -= lr * g; }
        }

        assert!(last_loss < first_loss,
            "Hebbian scan outer-loop weight descent should converge: first={first_loss:.6} last={last_loss:.6}");
    }

    #[test]
    fn test_hebbian_scan_with_initial_m() {
        let cfg = MAGConfig::hebbian_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let mut rng = SimpleRng::new(77);
        let mut initial_m = vec![0.0f32; d * d];
        rng.fill_uniform(&mut initial_m, 0.01);

        let (y_with, _) = hebbian_scan_forward(&params.levels[0], &embedded, s, d, Some(initial_m));
        let (y_without, _) = hebbian_scan_forward(&params.levels[0], &embedded, s, d, None);

        let diff: f32 = y_with.iter().zip(y_without.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(diff > 1e-8, "initial_m should affect output");
    }

    // ═══════════════════════════════════════════════════════════════
    // Titans momentum scan tests
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_titans_momentum_scan_matches_sequential() {
        // Verify momentum scan matches sequential computation
        let n = 8;
        let state_size = 16; // d*d for d=4
        let mut rng = SimpleRng::new(42);

        let mut etas = vec![0.0f32; n];
        rng.fill_uniform(&mut etas, 1.0);
        for e in &mut etas { *e = e.abs().min(0.99); }
        let mut thetas = vec![0.0f32; n];
        rng.fill_uniform(&mut thetas, 0.1);
        let mut grads = vec![0.0f32; n * state_size];
        rng.fill_uniform(&mut grads, 1.0);
        let s_init = vec![0.0f32; state_size];

        let s_scan = titans_momentum_scan(&etas, &thetas, &grads, &s_init, state_size);

        // Sequential
        let mut s = s_init.clone();
        for t in 0..n {
            let mut s_new = vec![0.0f32; state_size];
            for j in 0..state_size {
                s_new[j] = etas[t] * s[j] - thetas[t] * grads[t * state_size + j];
            }
            for j in 0..state_size {
                let diff = (s_scan[t * state_size + j] - s_new[j]).abs();
                assert!(diff < 1e-3,
                    "t={t} j={j}: scan={} seq={}", s_scan[t * state_size + j], s_new[j]);
            }
            s = s_new;
        }
    }

    #[test]
    fn test_titans_momentum_scan_zero_init() {
        let n = 4;
        let state_size = 4;
        let etas = vec![0.9f32; n];
        let thetas = vec![0.01f32; n];
        let mut grads = vec![0.0f32; n * state_size];
        for i in 0..n * state_size { grads[i] = (i as f32 + 1.0) * 0.1; }
        let s_init = vec![0.0f32; state_size];

        let s_states = titans_momentum_scan(&etas, &thetas, &grads, &s_init, state_size);

        // First step: S_1 = 0.9 * 0 - 0.01 * grad_0 = -0.01 * grad_0
        for j in 0..state_size {
            let expected = -0.01 * grads[j];
            assert!((s_states[j] - expected).abs() < 1e-5,
                "S_1[{j}]: expected {expected} got {}", s_states[j]);
        }
    }
}
