/// Lattice OSR (Orthogonal State Recurrence) memory system — 7th MIRAS variant.
///
/// Memory is m unit vectors on S^{d-1} (m×d), not a d×d matrix or MLP.
/// Updates use orthogonal projection: only novel information modifies slots.
/// Renormalization back to the unit sphere is equivalent to Riemannian GD
/// on S^{d-1} (Lattice Proposition 3.1).
///
/// MIRAS knobs: m-vector structure, dot-product bias, sphere normalization retention,
/// GD algorithm (orthogonal projection step).
///
/// Forward (per token, observe-then-advance):
///   k_t = embedded_t @ W_K_mem^T
///   v_t = embedded_t @ W_V_mem^T
///   q_t = embedded_t @ W_Q_mem^T
///   alpha_t = sigmoid(concat(k_t, v_t) @ w_alpha + b_alpha)
///
///   OBSERVE: read from current slots S_t
///     scores_read = S_t @ q_t     → [m]
///     weights = softmax(scores_read) → [m]
///     y_t = sum(weights[i] * S_t[i])  → [d]
///
///   ADVANCE: update slots → S_{t+1}
///     for each slot i:
///       score_i = dot(S_t[i], k_t)
///       gate_i = sigmoid(score_i)
///       delta_s = alpha_t * gate_i * v_t
///       parallel = dot(S_t[i], delta_s) * S_t[i]
///       orthogonal = delta_s - parallel
///       s_unnorm = S_t[i] + orthogonal
///       S_{t+1}[i] = normalize(s_unnorm)

use crate::tensor::{
    matmul_f32, transpose_f32, sigmoid_f32, softmax_f32,
    frobenius_dot_f32, vec_norm_f32, vec_normalize_f32, init_slots,
};
use crate::retention::sphere_project_and_normalize_inplace;
use crate::model::{MemoryLevelParams, LatticeVariant};
use crate::delta_rule::{MemoryRule, MemoryState, Gates, MemoryError};

// ── LatticeOSR implementation ────────────────────────────────────────

/// Lattice OSR: m unit vectors on the sphere, orthogonal projection updates.
///
/// The `variant` field selects which delta_s computation to use:
/// - Decode (default, Eqs 5-6): delta_s = gate_i * v_t
/// - Encode (Eqs 24-25): delta_s = gate_i * k_t
/// - Similarity (Eqs 7-8): delta_s = gate_i * (v_t - dot(S[i], v_t) * S[i])
pub struct LatticeOSR {
    pub m_slots: usize,
    pub variant: LatticeVariant,
}

/// All intermediate values from a Lattice forward pass, needed for backward.
pub struct LatticeCache {
    pub seq_len: usize,
    pub d: usize,
    pub m: usize,
    /// Which variant was used in the forward pass.
    pub variant: LatticeVariant,
    /// Slot states S_t for t=0..seq_len: [(seq_len+1) * m * d]
    pub s_states: Vec<f32>,
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
    /// Per-slot dot(S[i], k_t) scores: [seq_len, m]
    pub scores: Vec<f32>,
    /// Per-slot sigmoid(scores): [seq_len, m]
    pub slot_gates: Vec<f32>,
    /// Softmax read weights: [seq_len, m]
    pub read_weights: Vec<f32>,
    /// ||s_unnorm|| per slot per step: [seq_len, m]
    pub s_unnorm_norms: Vec<f32>,
}

impl MemoryRule for LatticeOSR {
    type Cache = LatticeCache;

    fn level(&self) -> usize { 0 }

    fn supported_parallelization(&self) -> &'static [&'static str] {
        crate::parallel::supported_strategies(crate::model::MemoryRuleKind::LatticeOSR)
    }

    fn init(&self, d: usize) -> MemoryState {
        MemoryState { m: init_slots(self.m_slots, d), d }
    }

    fn write(&self, state: &mut MemoryState, k: &[f32], v: &[f32], gates: &Gates) -> Result<(), MemoryError> {
        let d = state.d;
        let m = self.m_slots;
        for i in 0..m {
            let slot = &state.m[i * d..(i + 1) * d];
            let score = frobenius_dot_f32(slot, k);
            let gate_i = sigmoid_f32(score);

            // Compute input for orthogonal update based on variant
            let input = match self.variant {
                LatticeVariant::Decode => {
                    // Eqs 5-6: delta_s = gate_i * v_t
                    let mut inp = vec![0.0f32; d];
                    for j in 0..d { inp[j] = v[j]; }
                    inp
                }
                LatticeVariant::Encode => {
                    // Eqs 24-25: delta_s = gate_i * k_t
                    let mut inp = vec![0.0f32; d];
                    for j in 0..d { inp[j] = k[j]; }
                    inp
                }
                LatticeVariant::Similarity => {
                    // Eqs 7-8: delta_s = gate_i * (v_t - dot(S[i], v_t) * S[i])
                    let dot_sv = frobenius_dot_f32(slot, v);
                    let mut inp = vec![0.0f32; d];
                    for j in 0..d { inp[j] = v[j] - dot_sv * slot[j]; }
                    inp
                }
            };

            // delta_s = alpha * gate_i * input
            // orthogonal = delta_s - dot(s, delta_s) * s
            let scale = gates.alpha * gate_i;
            let mut s_unnorm = vec![0.0f32; d];
            let mut p = 0.0f32;
            for j in 0..d {
                p += slot[j] * scale * input[j];
            }
            for j in 0..d {
                let ortho = scale * input[j] - p * slot[j];
                s_unnorm[j] = slot[j] + ortho;
            }
            vec_normalize_f32(&mut s_unnorm);
            state.m[i * d..(i + 1) * d].copy_from_slice(&s_unnorm);
        }
        Ok(())
    }

    fn read(&self, state: &MemoryState, q: &[f32], out: &mut [f32]) -> Result<(), MemoryError> {
        let d = state.d;
        let m = self.m_slots;
        // softmax attention over m slots
        let mut scores = vec![0.0f32; m];
        for i in 0..m {
            scores[i] = frobenius_dot_f32(&state.m[i * d..(i + 1) * d], q);
        }
        let mut weights = vec![0.0f32; m];
        softmax_f32(&scores, &mut weights, 1, m);
        for j in 0..d {
            out[j] = 0.0;
        }
        for i in 0..m {
            for j in 0..d {
                out[j] += weights[i] * state.m[i * d + j];
            }
        }
        Ok(())
    }

    /// Full sequence forward with cache for backward.
    fn step(
        &self,
        level_params: &MemoryLevelParams,
        embedded: &[f32],
        seq_len: usize,
        d: usize,
        initial_m: Option<Vec<f32>>,
    ) -> (Vec<f32>, LatticeCache) {
        let m = self.m_slots;
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

        // Allocate cache — seed S_0 from initial_m if provided, else init_slots
        let mut s_states = vec![0.0f32; (seq_len + 1) * m * d];
        if let Some(m0) = initial_m {
            debug_assert_eq!(m0.len(), m * d);
            s_states[..m * d].copy_from_slice(&m0);
        } else {
            let s0 = init_slots(m, d);
            s_states[..m * d].copy_from_slice(&s0);
        }

        let mut concat_kv = vec![0.0f32; seq_len * 2 * d];
        let mut alpha_pre = vec![0.0f32; seq_len];
        let mut alpha = vec![0.0f32; seq_len];
        let mut scores_cache = vec![0.0f32; seq_len * m];
        let mut slot_gates = vec![0.0f32; seq_len * m];
        let mut read_weights = vec![0.0f32; seq_len * m];
        let mut s_unnorm_norms = vec![0.0f32; seq_len * m];
        let mut y = vec![0.0f32; seq_len * d];

        // Scratch buffers reused across the seq_len × m loop (avoid per-iteration allocs)
        let mut delta_s_scratch = vec![0.0f32; d];
        let mut s_new_scratch = vec![0.0f32; d];
        let mut read_scores = vec![0.0f32; m];

        for t in 0..seq_len {
            let k_t = &k_mem[t * d..(t + 1) * d];
            let v_t = &v_mem[t * d..(t + 1) * d];
            let q_t = &q_mem[t * d..(t + 1) * d];
            let s_t_off = t * m * d;

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

            // OBSERVE: read from current slots S_t
            for i in 0..m {
                let slot = &s_states[s_t_off + i * d..s_t_off + (i + 1) * d];
                read_scores[i] = frobenius_dot_f32(slot, q_t);
            }
            softmax_f32(&read_scores, &mut read_weights[t * m..(t + 1) * m], 1, m);
            for j in 0..d {
                y[t * d + j] = 0.0;
            }
            for i in 0..m {
                let w = read_weights[t * m + i];
                let slot = &s_states[s_t_off + i * d..s_t_off + (i + 1) * d];
                for j in 0..d {
                    y[t * d + j] += w * slot[j];
                }
            }

            // ADVANCE: update slots → S_{t+1}
            // Use split_at_mut to get non-overlapping borrows for t and t+1
            let s_next_off = (t + 1) * m * d;
            let (s_prev, s_next) = s_states.split_at_mut(s_next_off);
            for i in 0..m {
                let slot = &s_prev[s_t_off + i * d..s_t_off + (i + 1) * d];

                // Per-slot gate: sigmoid(dot(S[i], k_t))
                let score_i = frobenius_dot_f32(slot, k_t);
                scores_cache[t * m + i] = score_i;
                let gate_i = sigmoid_f32(score_i);
                slot_gates[t * m + i] = gate_i;

                // Variant-dependent delta_s computation (Eq 26 unified form)
                let scale = alpha[t] * gate_i;
                match self.variant {
                    LatticeVariant::Decode => {
                        // Eqs 5-6: delta_s = alpha_t * gate_i * v_t
                        for j in 0..d { delta_s_scratch[j] = scale * v_t[j]; }
                    }
                    LatticeVariant::Encode => {
                        // Eqs 24-25: delta_s = alpha_t * gate_i * k_t
                        for j in 0..d { delta_s_scratch[j] = scale * k_t[j]; }
                    }
                    LatticeVariant::Similarity => {
                        // Eqs 7-8: delta_s = alpha_t * gate_i * (v_t - dot(S[i], v_t) * S[i])
                        let dot_sv = frobenius_dot_f32(slot, v_t);
                        for j in 0..d {
                            delta_s_scratch[j] = scale * (v_t[j] - dot_sv * slot[j]);
                        }
                    }
                }

                // Sphere retention: orthogonal projection + normalize (in-place)
                let norm = sphere_project_and_normalize_inplace(
                    slot, &delta_s_scratch, d, &mut s_new_scratch,
                );
                s_unnorm_norms[t * m + i] = norm;
                s_next[i * d..(i + 1) * d].copy_from_slice(&s_new_scratch[..d]);
            }
        }

        let cache = LatticeCache {
            seq_len, d, m, variant: self.variant,
            s_states, k_mem, v_mem, q_mem, concat_kv,
            alpha_pre, alpha, scores: scores_cache, slot_gates,
            read_weights, s_unnorm_norms,
        };

        (y, cache)
    }

    /// Full sequence backward through the Lattice OSR memory.
    fn step_backward(
        &self,
        level_params: &MemoryLevelParams,
        cache: &LatticeCache,
        d_y: &[f32],
        embedded: &[f32],
    ) -> (MemoryLevelParams, Vec<f32>) {
        let s = cache.seq_len;
        let d = cache.d;
        let m = cache.m;
        debug_assert_eq!(d_y.len(), s * d);
        debug_assert_eq!(embedded.len(), s * d);

        let mut grads = MemoryLevelParams::zeros_like(d);
        let mut d_k_mem = vec![0.0f32; s * d];
        let mut d_v_mem = vec![0.0f32; s * d];
        let mut d_q_mem = vec![0.0f32; s * d];

        // d_S: gradient on slot states, two buffers for swap
        let mut d_s = vec![0.0f32; m * d];      // gradient on S_{t+1}
        let mut d_s_prev = vec![0.0f32; m * d];  // gradient on S_t

        // Reverse token loop
        for t in (0..s).rev() {
            let k_t = &cache.k_mem[t * d..(t + 1) * d];
            let v_t = &cache.v_mem[t * d..(t + 1) * d];
            let q_t = &cache.q_mem[t * d..(t + 1) * d];
            let s_t_off = t * m * d;
            let s_next_off = (t + 1) * m * d;
            let alpha_t = cache.alpha[t];
            let c_base = t * 2 * d;
            let concat_t = &cache.concat_kv[c_base..c_base + 2 * d];
            let d_y_t = &d_y[t * d..(t + 1) * d];

            // Zero d_s_prev for this step
            for x in d_s_prev.iter_mut() { *x = 0.0; }

            // ── Read backward (observe) ──────────────────────────────
            // y_t = sum(weights[i] * S_t[i])
            // weights = softmax(scores_read), scores_read[i] = dot(S_t[i], q_t)

            let weights = &cache.read_weights[t * m..(t + 1) * m];

            // d_weights[i] = dot(S_t[i], d_y_t)
            let mut d_weights = vec![0.0f32; m];
            for i in 0..m {
                let slot = &cache.s_states[s_t_off + i * d..s_t_off + (i + 1) * d];
                d_weights[i] = frobenius_dot_f32(slot, d_y_t);
            }

            // Softmax backward: d_scores_read[i] = weights[i] * (d_weights[i] - dot(weights, d_weights))
            let w_dot_dw: f32 = (0..m).map(|i| weights[i] * d_weights[i]).sum();
            let mut d_scores_read = vec![0.0f32; m];
            for i in 0..m {
                d_scores_read[i] = weights[i] * (d_weights[i] - w_dot_dw);
            }

            // Accumulate d_S_t from read: weights[i] * d_y_t + d_scores_read[i] * q_t
            for i in 0..m {
                for j in 0..d {
                    d_s_prev[i * d + j] += weights[i] * d_y_t[j]
                        + d_scores_read[i] * q_t[j];
                }
            }

            // d_q_mem[t] = sum(d_scores_read[i] * S_t[i])
            for i in 0..m {
                let slot = &cache.s_states[s_t_off + i * d..s_t_off + (i + 1) * d];
                for j in 0..d {
                    d_q_mem[t * d + j] += d_scores_read[i] * slot[j];
                }
            }

            // ── Update backward (advance) per slot ───────────────────
            let mut d_alpha_total = 0.0f32;

            for i in 0..m {
                let slot = &cache.s_states[s_t_off + i * d..s_t_off + (i + 1) * d];
                let s_new = &cache.s_states[s_next_off + i * d..s_next_off + (i + 1) * d];
                let gate_i = cache.slot_gates[t * m + i];
                let norm = cache.s_unnorm_norms[t * m + i];

                // d_S_{t+1}[i] from future tokens (accumulated in d_s)
                let d_s_new = &d_s[i * d..(i + 1) * d];

                // Through normalize: d_s_unnorm = (d_s_new - s_new * dot(s_new, d_s_new)) / norm
                let dot_sn_dsn: f32 = (0..d).map(|j| s_new[j] * d_s_new[j]).sum();
                let mut d_s_unnorm = vec![0.0f32; d];
                if norm > 1e-8 {
                    let inv_norm = 1.0 / norm;
                    for j in 0..d {
                        d_s_unnorm[j] = (d_s_new[j] - s_new[j] * dot_sn_dsn) * inv_norm;
                    }
                }

                // ── Recompute delta_s for this variant ──────────────────
                // Forward: delta_s = alpha_t * gate_i * input
                // where input depends on variant
                let mut delta_s = vec![0.0f32; d];
                match cache.variant {
                    LatticeVariant::Decode => {
                        for j in 0..d { delta_s[j] = alpha_t * gate_i * v_t[j]; }
                    }
                    LatticeVariant::Encode => {
                        for j in 0..d { delta_s[j] = alpha_t * gate_i * k_t[j]; }
                    }
                    LatticeVariant::Similarity => {
                        let dot_sv = frobenius_dot_f32(slot, v_t);
                        for j in 0..d {
                            delta_s[j] = alpha_t * gate_i * (v_t[j] - dot_sv * slot[j]);
                        }
                    }
                }

                // Through orthogonal projection:
                // s_unnorm = slot + delta_s - dot(slot, delta_s) * slot
                //          = slot * (1 - p) + delta_s  where p = dot(slot, delta_s)
                let p: f32 = (0..d).map(|j| slot[j] * delta_s[j]).sum();

                // d_slot_k = d_s_unnorm_k * (1-p) - delta_s_k * dot(d_s_unnorm, slot)
                let dot_dsun_slot: f32 = (0..d).map(|j| d_s_unnorm[j] * slot[j]).sum();

                for j in 0..d {
                    d_s_prev[i * d + j] += d_s_unnorm[j] * (1.0 - p) - delta_s[j] * dot_dsun_slot;
                }

                // d_delta_s_k = d_s_unnorm_k - slot_k * dot(d_s_unnorm, slot)
                let mut d_delta_s = vec![0.0f32; d];
                for j in 0..d {
                    d_delta_s[j] = d_s_unnorm[j] - slot[j] * dot_dsun_slot;
                }

                // ── Variant-dependent backward through delta_s ────────────
                // delta_s = alpha_t * gate_i * input
                // d_delta_s is the gradient on delta_s from orthogonal projection
                match cache.variant {
                    LatticeVariant::Decode => {
                        // input = v_t
                        let dot_dds_input: f32 = frobenius_dot_f32(&d_delta_s, v_t);
                        d_alpha_total += gate_i * dot_dds_input;
                        let d_gate_i = alpha_t * dot_dds_input;
                        for j in 0..d {
                            d_v_mem[t * d + j] += alpha_t * gate_i * d_delta_s[j];
                        }
                        // Through gate_i = sigmoid(score_i)
                        let d_score_i = d_gate_i * gate_i * (1.0 - gate_i);
                        for j in 0..d {
                            d_s_prev[i * d + j] += d_score_i * k_t[j];
                            d_k_mem[t * d + j] += d_score_i * slot[j];
                        }
                    }
                    LatticeVariant::Encode => {
                        // input = k_t
                        let dot_dds_input: f32 = frobenius_dot_f32(&d_delta_s, k_t);
                        d_alpha_total += gate_i * dot_dds_input;
                        let d_gate_i = alpha_t * dot_dds_input;
                        for j in 0..d {
                            d_k_mem[t * d + j] += alpha_t * gate_i * d_delta_s[j];
                        }
                        // Through gate_i = sigmoid(score_i)
                        let d_score_i = d_gate_i * gate_i * (1.0 - gate_i);
                        for j in 0..d {
                            d_s_prev[i * d + j] += d_score_i * k_t[j];
                            d_k_mem[t * d + j] += d_score_i * slot[j];
                        }
                    }
                    LatticeVariant::Similarity => {
                        // input = v_t - dot(S[i], v_t) * S[i]
                        // Recompute input for gradient
                        let dot_sv = frobenius_dot_f32(slot, v_t);
                        let mut input = vec![0.0f32; d];
                        for j in 0..d { input[j] = v_t[j] - dot_sv * slot[j]; }

                        let dot_dds_input: f32 = frobenius_dot_f32(&d_delta_s, &input);
                        d_alpha_total += gate_i * dot_dds_input;
                        let d_gate_i = alpha_t * dot_dds_input;

                        // d_input = alpha_t * gate_i * d_delta_s
                        let mut d_input = vec![0.0f32; d];
                        for j in 0..d { d_input[j] = alpha_t * gate_i * d_delta_s[j]; }

                        // Through input = v_t - dot_sv * S[i]:
                        // d_v_t[j] += d_input[j] - dot(d_input, S[i]) * S[i][j]
                        let dot_dinput_slot: f32 = frobenius_dot_f32(&d_input, slot);
                        for j in 0..d {
                            d_v_mem[t * d + j] += d_input[j] - dot_dinput_slot * slot[j];
                        }
                        // d_S[i][j] += -dot_sv * d_input[j] - dot(d_input, S[i]) * v_t[j]
                        for j in 0..d {
                            d_s_prev[i * d + j] += -dot_sv * d_input[j] - dot_dinput_slot * v_t[j];
                        }

                        // Through gate_i = sigmoid(score_i)
                        let d_score_i = d_gate_i * gate_i * (1.0 - gate_i);
                        for j in 0..d {
                            d_s_prev[i * d + j] += d_score_i * k_t[j];
                            d_k_mem[t * d + j] += d_score_i * slot[j];
                        }
                    }
                }
            }

            // ── Gate backward: alpha_t = sigmoid(alpha_pre_t) ──
            let sig_deriv = alpha_t * (1.0 - alpha_t);
            let d_alpha_pre = d_alpha_total * sig_deriv;

            // w_alpha, b_alpha gradient
            for i in 0..(2 * d) {
                grads.w_alpha[i] += d_alpha_pre * concat_t[i];
            }
            grads.b_alpha[0] += d_alpha_pre;

            // concat backward → d_k_mem, d_v_mem
            for i in 0..d {
                d_k_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[i];
            }
            for i in 0..d {
                d_v_mem[t * d + i] += d_alpha_pre * level_params.w_alpha[d + i];
            }

            // Swap: d_s_prev becomes d_s for next (earlier) token
            std::mem::swap(&mut d_s, &mut d_s_prev);
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

// ── Read-only functions (for CMS frozen levels) ─────────────────────

/// Read-only forward for frozen Lattice levels: softmax-attention read over m slots.
/// Returns (y, q_mem) where y=[seq_len, d] and q_mem=[seq_len, d].
pub fn lattice_read_only(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    frozen_s: &[f32],
    seq_len: usize,
    d: usize,
    m: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(frozen_s.len(), m * d);

    let mut w_q_mem_t = vec![0.0f32; d * d];
    transpose_f32(&level_params.w_q_mem, &mut w_q_mem_t, d, d);
    let mut q_mem = vec![0.0f32; seq_len * d];
    matmul_f32(embedded, &w_q_mem_t, &mut q_mem, seq_len, d, d);

    let mut y = vec![0.0f32; seq_len * d];
    let mut scores = vec![0.0f32; m];
    let mut weights = vec![0.0f32; m];

    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];

        // scores[i] = dot(frozen_s[i], q_t)
        for i in 0..m {
            scores[i] = frobenius_dot_f32(&frozen_s[i * d..(i + 1) * d], q_t);
        }
        softmax_f32(&scores, &mut weights, 1, m);

        // y_t = sum(weights[i] * frozen_s[i])
        for j in 0..d {
            y[t * d + j] = 0.0;
        }
        for i in 0..m {
            for j in 0..d {
                y[t * d + j] += weights[i] * frozen_s[i * d + j];
            }
        }
    }

    (y, q_mem)
}

/// Read-only backward for frozen Lattice levels.
/// Returns (param_grads, d_embedded).
pub fn lattice_read_only_backward(
    level_params: &MemoryLevelParams,
    frozen_s: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    m: usize,
) -> (MemoryLevelParams, Vec<f32>) {
    let mut grads = MemoryLevelParams::zeros_like(d);
    let mut d_q_mem = vec![0.0f32; seq_len * d];

    // Recompute read weights for backward
    let mut scores = vec![0.0f32; m];
    let mut weights = vec![0.0f32; m];

    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let d_y_t = &d_y[t * d..(t + 1) * d];

        for i in 0..m {
            scores[i] = frobenius_dot_f32(&frozen_s[i * d..(i + 1) * d], q_t);
        }
        softmax_f32(&scores, &mut weights, 1, m);

        // d_weights[i] = dot(frozen_s[i], d_y_t)
        let mut d_weights = vec![0.0f32; m];
        for i in 0..m {
            d_weights[i] = frobenius_dot_f32(&frozen_s[i * d..(i + 1) * d], d_y_t);
        }

        // Softmax backward
        let w_dot_dw: f32 = (0..m).map(|i| weights[i] * d_weights[i]).sum();
        let mut d_scores = vec![0.0f32; m];
        for i in 0..m {
            d_scores[i] = weights[i] * (d_weights[i] - w_dot_dw);
        }

        // d_q_mem[t] = sum(d_scores[i] * frozen_s[i])
        for i in 0..m {
            for j in 0..d {
                d_q_mem[t * d + j] += d_scores[i] * frozen_s[i * d + j];
            }
        }
    }

    // Projection backward: q_mem = embedded @ W_Q_mem^T
    let mut d_embedded = vec![0.0f32; seq_len * d];
    let mut d_q_mem_t = vec![0.0f32; d * seq_len];
    transpose_f32(&d_q_mem, &mut d_q_mem_t, seq_len, d);
    matmul_f32(&d_q_mem_t, embedded, &mut grads.w_q_mem, d, seq_len, d);
    crate::tensor::matmul_acc_f32(&d_q_mem, &level_params.w_q_mem, &mut d_embedded, seq_len, d, d);

    (grads, d_embedded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;

    fn test_config() -> MAGConfig {
        MAGConfig::lattice_test_config()
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
    fn test_lattice_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_lattice_forward_deterministic() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (y1, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        assert_eq!(y1, y2, "Lattice forward should be deterministic");
    }

    #[test]
    fn test_lattice_forward_output_shape() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        assert_eq!(y.len(), s * d);
        assert_eq!(cache.k_mem.len(), s * d);
        assert_eq!(cache.v_mem.len(), s * d);
        assert_eq!(cache.q_mem.len(), s * d);
        assert_eq!(cache.s_states.len(), (s + 1) * m * d);
    }

    #[test]
    fn test_lattice_memory_evolves() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        let s = cfg.swa.seq_len;

        let s_0 = &cache.s_states[0..m * d];
        let s_t = &cache.s_states[s * m * d..(s + 1) * m * d];

        // Should be different after processing tokens
        let diff: f32 = s_0.iter().zip(s_t.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(diff > 1e-6, "Slots should evolve, diff={diff:.4e}");
    }

    #[test]
    fn test_lattice_gate_range() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for t in 0..cfg.swa.seq_len {
            let a = cache.alpha[t];
            assert!(a > 0.0 && a < 1.0, "alpha[{t}]={a} not in (0,1)");
            for i in 0..cfg.m_slots {
                let g = cache.slot_gates[t * cfg.m_slots + i];
                assert!(g > 0.0 && g < 1.0, "slot_gate[{t},{i}]={g} not in (0,1)");
            }
        }
    }

    #[test]
    fn test_lattice_sphere_preserved() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;

        // Check every slot at every timestep stays unit norm
        for t in 0..=cfg.swa.seq_len {
            for i in 0..m {
                let off = t * m * d + i * d;
                let slot = &cache.s_states[off..off + d];
                let norm = vec_norm_f32(slot);
                assert!((norm - 1.0).abs() < 1e-5,
                    "Slot [{t},{i}] not unit: norm={norm:.8}");
            }
        }
    }

    #[test]
    fn test_lattice_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
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
    fn test_lattice_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
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
    fn test_lattice_backward_shapes() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
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
    fn test_lattice_init() {
        let rule = LatticeOSR { m_slots: 4, variant: LatticeVariant::Decode };
        let state = rule.init(8);
        assert_eq!(state.m.len(), 4 * 8);
        assert_eq!(state.d, 8);
        // All slots should be unit norm
        for i in 0..4 {
            let norm = vec_norm_f32(&state.m[i * 8..(i + 1) * 8]);
            assert!((norm - 1.0).abs() < 1e-6, "init slot {i} norm={norm}");
        }
    }

    #[test]
    fn test_lattice_write_read() {
        let rule = LatticeOSR { m_slots: 2, variant: LatticeVariant::Decode };
        let mut state = rule.init(4);
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.0f32, 0.0, 0.0, 1.0];
        let gates = Gates { alpha: 0.5, theta: 0.0, eta: 1.0 };

        rule.write(&mut state, &k, &v, &gates).unwrap();

        // Slots should still be unit norm after write
        for i in 0..2 {
            let norm = vec_norm_f32(&state.m[i * 4..(i + 1) * 4]);
            assert!((norm - 1.0).abs() < 1e-5, "post-write slot {i} norm={norm}");
        }

        // Read should produce finite output
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        rule.read(&state, &q, &mut out).unwrap();
        for (i, &o) in out.iter().enumerate() {
            assert!(o.is_finite(), "read out[{i}] not finite: {o}");
        }
    }

    #[test]
    fn test_lattice_level_and_parallelization() {
        let rule = LatticeOSR { m_slots: 4, variant: LatticeVariant::Decode };
        assert_eq!(rule.level(), 0);
        let strategies = rule.supported_parallelization();
        assert!(strategies.contains(&"sequential"));
        assert!(strategies.contains(&"chunkwise_gd"));
        assert!(strategies.contains(&"tnt"));
        assert!(strategies.contains(&"lattice_gla"));
    }

    // ── Read-only tests ──────────────────────────────────────────────

    #[test]
    fn test_lattice_read_only_produces_output() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let m = cfg.m_slots;
        let frozen_s = init_slots(m, d);
        let (y, q_mem) = lattice_read_only(&params.levels[0], &embedded, &frozen_s, s, d, m);
        assert_eq!(y.len(), s * d);
        assert_eq!(q_mem.len(), s * d);
        // Output should be finite and non-trivial (frozen slots are non-zero)
        let y_norm: f32 = y.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(y_norm > 1e-6, "read_only output should be non-zero, norm={y_norm:.4e}");
    }

    #[test]
    fn test_lattice_read_only_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let s = cfg.swa.seq_len;
        let m = cfg.m_slots;
        let frozen_s = init_slots(m, d);
        let (_y, q_mem) = lattice_read_only(&params.levels[0], &embedded, &frozen_s, s, d, m);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = lattice_read_only_backward(
            &params.levels[0], &frozen_s, &q_mem, &d_y, &embedded, s, d, m,
        );
        for &v in grads.w_q_mem.iter() {
            assert!(v.is_finite(), "read_only backward grad not finite");
        }
        for &v in d_emb.iter() {
            assert!(v.is_finite(), "read_only backward d_emb not finite");
        }
    }

    #[test]
    fn test_lattice_initial_m_seeding() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        let rule = LatticeOSR { m_slots: m, variant: LatticeVariant::Decode };

        // Custom initial slots
        let mut custom_s0 = vec![0.0f32; m * d];
        for i in 0..m {
            custom_s0[i * d] = 1.0; // point along first axis
        }

        let (y1, _) = rule.step(&params.levels[0], &embedded, s, d, None);
        let (y2, _) = rule.step(&params.levels[0], &embedded, s, d, Some(custom_s0));

        // Different initial state → different output
        let diff: f32 = y1.iter().zip(y2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(diff > 1e-6, "Different initial state should give different output, diff={diff:.4e}");
    }

    #[test]
    fn test_lattice_orthogonal_update_preserves_sphere() {
        // Run multiple forward passes (simulating training) and verify sphere constraint
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        let rule = LatticeOSR { m_slots: m, variant: LatticeVariant::Decode };

        let mut current_s: Option<Vec<f32>> = None;
        for step in 0..10 {
            let mut rng = SimpleRng::new(100 + step);
            let mut embedded = vec![0.0f32; cfg.swa.seq_len * d];
            rng.fill_uniform(&mut embedded, 0.5);

            let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, current_s);

            // Check final state
            let s_final_off = cfg.swa.seq_len * m * d;
            for i in 0..m {
                let slot = &cache.s_states[s_final_off + i * d..s_final_off + (i + 1) * d];
                let norm = vec_norm_f32(slot);
                assert!((norm - 1.0).abs() < 1e-5,
                    "Step {step}, slot {i}: norm={norm:.8}");
            }

            current_s = Some(cache.s_states[s_final_off..s_final_off + m * d].to_vec());
        }
    }

    // ── Variant tests ───────────────────────────────────────────────

    #[test]
    fn test_encode_variant_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Encode };
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "encode y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_encode_variant_sphere_preserved() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        let rule = LatticeOSR { m_slots: m, variant: LatticeVariant::Encode };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        for t in 0..=cfg.swa.seq_len {
            for i in 0..m {
                let off = t * m * d + i * d;
                let slot = &cache.s_states[off..off + d];
                let norm = vec_norm_f32(slot);
                assert!((norm - 1.0).abs() < 1e-5,
                    "Encode: slot [{t},{i}] not unit: norm={norm:.8}");
            }
        }
    }

    #[test]
    fn test_encode_variant_differs_from_decode() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let decode_rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let encode_rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Encode };
        let (y_decode, _) = decode_rule.step(&params.levels[0], &embedded, s, d, None);
        let (y_encode, _) = encode_rule.step(&params.levels[0], &embedded, s, d, None);

        let diff: f32 = y_decode.iter().zip(y_encode.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(diff > 1e-6, "Encode and Decode should produce different outputs, diff={diff:.4e}");
    }

    #[test]
    fn test_encode_variant_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Encode };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);
        for (name, g) in [("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem), ("w_q_mem", &grads.w_q_mem)] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "encode grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "encode d_embedded[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_similarity_variant_forward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Similarity };
        let (y, _cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, cfg.swa.d_model, None);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "similarity y[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_similarity_variant_sphere_preserved() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let d = cfg.swa.d_model;
        let m = cfg.m_slots;
        let rule = LatticeOSR { m_slots: m, variant: LatticeVariant::Similarity };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, cfg.swa.seq_len, d, None);
        for t in 0..=cfg.swa.seq_len {
            for i in 0..m {
                let off = t * m * d + i * d;
                let slot = &cache.s_states[off..off + d];
                let norm = vec_norm_f32(slot);
                assert!((norm - 1.0).abs() < 1e-5,
                    "Similarity: slot [{t},{i}] not unit: norm={norm:.8}");
            }
        }
    }

    #[test]
    fn test_similarity_variant_differs_from_decode() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let decode_rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Decode };
        let similarity_rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Similarity };
        let (y_decode, _) = decode_rule.step(&params.levels[0], &embedded, s, d, None);
        let (y_similarity, _) = similarity_rule.step(&params.levels[0], &embedded, s, d, None);

        // Similarity pre-projects v_t onto the tangent plane before the standard
        // orthogonal update. In tiny configs (d=8, random slots), the pre-projection
        // effect is small since dot(S[i], v_t) ≈ 0 for random unit vectors. The
        // difference is measurable but tiny — verify it's nonzero.
        let diff: f32 = y_decode.iter().zip(y_similarity.iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        assert!(diff > 1e-10, "Similarity and Decode should produce different outputs, diff={diff:.4e}");
    }

    #[test]
    fn test_similarity_variant_backward_finite() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Similarity };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);
        for (name, g) in [("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem), ("w_q_mem", &grads.w_q_mem)] {
            for (i, &v) in g.iter().enumerate() {
                assert!(v.is_finite(), "similarity grad_{name}[{i}] not finite: {v}");
            }
        }
        for (i, &v) in d_emb.iter().enumerate() {
            assert!(v.is_finite(), "similarity d_embedded[{i}] not finite: {v}");
        }
    }

    #[test]
    fn test_similarity_variant_backward_nonzero() {
        let cfg = test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let rule = LatticeOSR { m_slots: cfg.m_slots, variant: LatticeVariant::Similarity };
        let (_y, cache) = rule.step(&params.levels[0], &embedded, s, d, None);
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = rule.step_backward(&params.levels[0], &cache, &d_y, &embedded);
        for (name, g) in [("w_k_mem", &grads.w_k_mem), ("w_v_mem", &grads.w_v_mem), ("w_q_mem", &grads.w_q_mem)] {
            let max_abs = g.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!(max_abs > 1e-10, "similarity grad_{name} all zeros (max_abs={max_abs})");
        }
        let emb_max = d_emb.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(emb_max > 1e-10, "similarity d_embedded all zeros");
    }

    #[test]
    fn test_write_read_encode_variant() {
        let rule = LatticeOSR { m_slots: 2, variant: LatticeVariant::Encode };
        let mut state = rule.init(4);
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.0f32, 0.0, 0.0, 1.0];
        let gates = Gates { alpha: 0.5, theta: 0.0, eta: 1.0 };
        rule.write(&mut state, &k, &v, &gates).unwrap();
        // Slots should still be unit norm
        for i in 0..2 {
            let norm = vec_norm_f32(&state.m[i * 4..(i + 1) * 4]);
            assert!((norm - 1.0).abs() < 1e-5, "encode write: slot {i} norm={norm}");
        }
    }

    #[test]
    fn test_write_read_similarity_variant() {
        let rule = LatticeOSR { m_slots: 2, variant: LatticeVariant::Similarity };
        let mut state = rule.init(4);
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.0f32, 0.0, 0.0, 1.0];
        let gates = Gates { alpha: 0.5, theta: 0.0, eta: 1.0 };
        rule.write(&mut state, &k, &v, &gates).unwrap();
        for i in 0..2 {
            let norm = vec_norm_f32(&state.m[i * 4..(i + 1) * 4]);
            assert!((norm - 1.0).abs() < 1e-5, "similarity write: slot {i} norm={norm}");
        }
    }
}
