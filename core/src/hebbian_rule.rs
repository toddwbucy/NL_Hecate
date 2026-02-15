/// Hebbian Rule memory system — direct association without error correction.
///
/// The simplest MIRAS memory variant: pure associative outer-product write.
/// No error term, no gradient descent, no momentum — just "fire together, wire together".
///
/// MIRAS knobs: matrix structure, dot-product bias, L2 decay retention, direct association.
///
/// Forward (per token):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///   M_{t+1} = (1 - alpha_t) * M_t + outer(v_t, k_t)    // Hebbian write
///   y_t = M_{t+1} @ q_t                                  // read
///
/// Backward: reverse token loop with accumulated d_M.
/// Simpler than Delta Rule — no error/prediction chain.

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32,
    frobenius_dot_f32,
};
use crate::model::MemoryLevelParams;
use crate::delta_rule::{MemoryRule, MemoryState, Gates};

// ── Hebbian Rule implementation ─────────────────────────────────────

/// Hebbian Rule: direct association — outer(v, k) write, no error correction.
pub struct HebbianRule;

/// All intermediate values from a Hebbian forward pass, needed for backward.
pub struct HebbianCache {
    pub seq_len: usize,
    pub d: usize,
    /// Memory matrices M_t for t=0..seq_len: [(seq_len+1) * d * d]
    pub m_states: Vec<f32>,
    /// Per-token projected keys: [seq_len, d]
    pub k_mem: Vec<f32>,
    /// Per-token projected values: [seq_len, d]
    pub v_mem: Vec<f32>,
    /// Per-token projected queries: [seq_len, d]
    pub q_mem: Vec<f32>,
    /// Per-token concatenated (k,v): [seq_len, 2*d]
    pub concat_kv: Vec<f32>,
    /// Pre-sigmoid alpha values: [seq_len]
    pub alpha_pre: Vec<f32>,
    /// Sigmoid alpha values: [seq_len]
    pub alpha: Vec<f32>,
    /// Memory output y_t: [seq_len, d]
    pub y: Vec<f32>,
}

impl MemoryRule for HebbianRule {
    type Cache = HebbianCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] { &["sequential"] }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: vec![0.0f32; d * d], d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) {
        let d = state.d;
        // M = (1-alpha) * M + outer(v, k)  — pure associative, no error
        let retention = 1.0 - gates.alpha;
        for i in 0..d {
            for j in 0..d {
                state.m[i * d + j] = retention * state.m[i * d + j] + v[i] * k[j];
            }
        }
    }

    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) {
        let d = state.d;
        matmul_f32(&state.m, q, out, d, d, 1);
    }

    /// Full sequence forward with cache for backward.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<&[f32]>,
    ) -> (Vec<f32>, HebbianCache) {
        debug_assert_eq!(embedded.len(), seq_len * d);

        // Project embedded → k_mem, v_mem, q_mem via W^T
        let mut w_k_mem_t = vec![0.0f32; d * d];
        let mut w_v_mem_t = vec![0.0f32; d * d];
        let mut w_q_mem_t = vec![0.0f32; d * d];
        transpose_f32(&level_params.w_k_mem, &mut w_k_mem_t, d, d);
        transpose_f32(&level_params.w_v_mem, &mut w_v_mem_t, d, d);
        transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);

        let mut k_mem = vec![0.0f32; seq_len * d];
        let mut v_mem = vec![0.0f32; seq_len * d];
        let mut q_mem = vec![0.0f32; seq_len * d];
        matmul_f32(embedded, &w_k_mem_t, &mut k_mem, seq_len, d, d);
        matmul_f32(embedded, &w_v_mem_t, &mut v_mem, seq_len, d, d);
        matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

        // Allocate cache — seed M_0 from initial_m if provided, else zeros
        let mut m_states = vec![0.0f32; (seq_len + 1) * d * d];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), d * d);
            m_states[..d * d].copy_from_slice(m0);
        }
        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut y = vec![0.0f32; seq_len * d];

        // Sequential token loop
        for t in 0..seq_len {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];

            // Concatenate (k_t, v_t)
            let c_base = t * 2 * d;
            concat_kv[c_base..c_base + d].copy_from_slice(k_t);
            concat_kv[c_base + d..c_base + 2 * d].copy_from_slice(v_t);
            let concat_t = &concat_kv[c_base..c_base + 2 * d];

            // alpha_t = sigmoid(concat @ w_alpha + b_alpha)
            let mut alpha_pre_t = level_params.b_alpha[0];
            for i in 0..(2 * d) {
                alpha_pre_t += concat_t[i] * level_params.w_alpha[i];
            }
            alpha_pre[t] = alpha_pre_t;
            alpha[t] = sigmoid_f32(alpha_pre_t);

            // M_{t+1} = (1 - alpha_t) * M_t + outer(v_t, k_t)
            let retention = 1.0 - alpha[t];
            let m_t_off = t * d * d;
            let m_next_off = (t + 1) * d * d;
            for i in 0..d {
                for j in 0..d {
                    m_states[m_next_off + i * d + j] =
                        retention * m_states[m_t_off + i * d + j] + v_t[i] * k_t[j];
                }
            }

            // y_t = M_{t+1} @ q_t
            let m_next = &m_states[m_next_off..m_next_off + d * d];
            matmul_f32(m_next, q_t, &mut y[t * d..(t + 1) * d], d, d, 1);
        }

        let cache = HebbianCache {
            seq_len, d, m_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, y: y.clone(),
        };

        (y, cache)
    }

    /// Full sequence backward through the Hebbian memory.
    ///
    /// Simpler than Delta Rule: no error/prediction chain, no theta gate.
    /// Only alpha gate and outer-product write need backward.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &HebbianCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like(d);

        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // d_M: accumulated gradient on memory state (two buffers, swap to avoid allocation)
        let mut d_m = vec![0.0f32; d * d];
        let mut d_m_prev = vec![0.0f32; d * d];

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let v_t = &cache.v_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let m_t = &cache.m_states[t * d * d..(t + 1) * d * d];
            let m_next = &cache.m_states[(t + 1) * d * d..(t + 2) * d * d];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let alpha_t = cache.alpha[t];

            // ── y_t = M_{t+1} @ q_t backward ──
            let d_y_t = &d_y[t * d..(t + 1) * d];

            // d_M += outer(d_y_t, q_t)
            for i in 0..d {
                for j in 0..d {
                    d_m[i * d + j] += d_y_t[i] * q_t[j];
                }
            }

            // d_q_t = M_{t+1}^T @ d_y_t
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += m_next[j * d + i] * d_y_t[j];
                }
                d_q_mem[t * d + i] = sum;
            }

            // ── M_{t+1} = (1 - alpha_t) * M_t + outer(v_t, k_t) backward ──

            // d_alpha = -frobenius_dot(d_M, M_t)  (from retention term)
            let d_alpha_scalar = -frobenius_dot_f32(&d_m, m_t);

            // d_v_t from outer product: d_v_t[i] = sum_j(d_M[i,j] * k_t[j])
            for i in 0..d {
                let mut sum = 0.0f32;
                for j in 0..d {
                    sum += d_m[i * d + j] * k_t[j];
                }
                d_v_mem[t * d + i] += sum;
            }

            // d_k_t from outer product: d_k_t[j] = sum_i(d_M[i,j] * v_t[i])
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += d_m[i * d + j] * v_t[i];
                }
                d_k_mem[t * d + j] += sum;
            }

            // d_M_prev = (1 - alpha_t) * d_M  (propagate to previous step)
            let retention = 1.0 - alpha_t;
            for i in 0..(d * d) {
                d_m_prev[i] = retention * d_m[i];
            }

            // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_scalar * sig_deriv;

            // ── w_alpha, b_alpha gradient ──
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;

            // ── concat backward → d_k_mem, d_v_mem ──
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i];
            }

            // Swap buffers: d_m_prev becomes d_m for next (earlier) token
            std::mem::swap(&mut d_m, &mut d_m_prev);
        }

        // ── Projection backward: k_mem = embedded @ W_K_mem^T ──
        let mut d_embedded = vec![0.0f32; s * d];

        let mut d_k_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_k_mem, &mut d_k_mem_t, s, d);
        matmul_f32(&d_k_mem_t, embedded, &mut grads.w_k_mem, d, s, d);

        let mut d_v_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_v_mem, &mut d_v_mem_t, s, d);
        matmul_f32(&d_v_mem_t, embedded, &mut grads.w_v_mem, d, s, d);

        let mut d_q_mem_t = vec![0.0f32; d * s];
        transpose_f32(&d_q_mem, &mut d_q_mem_t, s, d);
        matmul_f32(&d_q_mem_t, embedded, &mut grads.w_q_mem, d, s, d);

        crate::tensor::matmul_acc_f32(&d_k_mem, &level_params.w_k_mem, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_v_mem, &level_params.w_v_mem, &mut d_embedded, s, d, d);
        crate::tensor::matmul_acc_f32(&d_q_mem, &level_params.w_q_mem, &mut d_embedded, s, d, d);

        (grads, d_embedded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

    fn test_config() -> MAGConfig {
        MAGConfig::hebbian_test_config()
    }

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut embedded = vec![0.0f32; s * d];
        rng.fill_uniform(&mut embedded, 0.1);
        embedded
    }

    #[test]
    fn test_hebbian_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = HebbianRule;
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_hebbian_forward_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = HebbianRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;

        let m_0 = &cache.m_states[0..d * d];
        let m_t = &cache.m_states[s * d * d..(s + 1) * d * d];
        let m0_norm: f32 = m_0.iter().map(|x| x * x).sum();
        let mt_norm: f32 = m_t.iter().map(|x| x * x).sum();
        assert!(m0_norm < 1e-12, "M_0 should be zero");
        assert!(mt_norm > 1e-12, "M_T should have evolved, norm={mt_norm}");
    }

    #[test]
    fn test_hebbian_forward_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = HebbianRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
        }
    }

    #[test]
    fn test_hebbian_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = HebbianRule;
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Hebbian rule forward should be deterministic");
    }

    #[test]
    fn test_hebbian_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = HebbianRule;
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.m_states.len(), (s + 1) * d * d);
    }

    #[test]
    fn test_hebbian_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = HebbianRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem), ("w_alpha", &grads.w_alpha),
            ("b_alpha", &grads.b_alpha),
        ] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "d_embedded[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_hebbian_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = HebbianRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        for (name, g) in [
            ("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem),
            ("w_q_mem", &grads.w_q_mem),
        ] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "grad_{name} is all zeros (max_abs={max_abs})");
        }
        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "d_embedded is all zeros");
    }

    #[test]
    fn test_hebbian_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = HebbianRule;
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);

        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);

        assert_eq!(grads.w_k_mem.len(), d * d);
        assert_eq!(grads.w_v_mem.len(), d * d);
        assert_eq!(grads.w_q_mem.len(), d * d);
        assert_eq!(grads.w_alpha.len(), 2 * d);
        assert_eq!(grads.b_alpha.len(), 1);
        assert_eq!(d_emb.len(), s * d);
    }

    // ── Trait API tests ──────────────────────────────────────────────

    #[test]
    fn test_hebbian_init() {
        let rule = HebbianRule;
        let state = rule.init(8);
        assert_eq!(state.m.len(), 64);
        assert_eq!(state.d, 8);
        assert!(state.m.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_hebbian_write_read() {
        let rule = HebbianRule;
        let mut state = rule.init(4);
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.0f32, 1.0, 0.0, 0.0];
        let gates = Gates { alpha: 0.0, theta: 0.0 };

        // Write: M = (1-0)*0 + outer(v, k) = outer(v, k)
        // = [[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]
        rule.write(&mut state, &k, &v, &gates);

        // Read: M @ q should give v when q = k
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        rule.read(&state, &q, &mut out);
        assert!((out[1] - 1.0).abs() < 1e-6, "read should return stored value");
    }

    #[test]
    fn test_hebbian_level_and_parallelization() {
        let rule = HebbianRule;
        assert_eq!(rule.level(), 0);
        assert_eq!(rule.supported_parallelization(), &["sequential"]);
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_hebbian_read_only_zero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let frozen_m = vec![0.0f32; d * d];
        let (y, _q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        assert!(y.iter().all(|&x| x.abs() < 1e-12));
    }

    #[test]
    fn test_hebbian_read_only_nonzero_memory() {
        use crate::delta_rule::delta_rule_read_only;
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let mut frozen_m = vec![0.0f32; d * d];
        for i in 0..d { frozen_m[i * d + i] = 1.0; }
        let (y, q_mem) = delta_rule_read_only(&params.levels[0], &embedded, &frozen_m, s, d);
        for i in 0..(s * d) {
            assert!((y[i] - q_mem[i]).abs() < 1e-6, "y[{i}]={} != q_mem[{i}]={}", y[i], q_mem[i]);
        }
    }
}
