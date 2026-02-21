/// M3 Multi-Scale Optimizer — CMS applied to the optimizer itself.
///
/// k momentum accumulators at k frequency levels.
/// Frozen levels accumulate gradients in error buffers.
/// Active levels apply accumulated + current gradient with EMA.
///
/// Source: HOPE (2512.24695) Section 7, Eq 71 applied to optimizer.
/// Spec: specs/algorithms/optimization_machinery/02_m3.md

use serde::{Serialize, Deserialize};
use crate::model::{MAGParams, SWAParams, MemoryLevelParams};

// ── Configuration ─────────────────────────────────────────────────────

/// M3 optimizer configuration: one momentum accumulator per frequency level.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct M3Config {
    /// Number of optimizer frequency levels.
    pub k: usize,
    /// Per-level EMA decay: S = eta * S + theta * grad.
    pub etas: Vec<f32>,
    /// Per-level gradient scaling factor.
    pub thetas: Vec<f32>,
    /// Per-level combination weights for the final update vector.
    pub weights: Vec<f32>,
    /// Per-level update frequencies: level fires when step % freq == 0.
    pub frequencies: Vec<usize>,
    /// Apply Newton-Schulz orthogonalization to combined momentum (Muon).
    pub use_newton_schulz: bool,
    /// Number of Newton-Schulz iterations (default 5).
    pub ns_iterations: usize,
    /// Explicit square dimension for NS. Required when use_newton_schulz is true.
    /// NS only applies when ns_dim * ns_dim == total_params. None = skip NS.
    pub ns_dim: Option<usize>,
}

impl M3Config {
    /// Single-level EMA momentum (degenerates to standard momentum).
    pub fn default_k1() -> Self {
        M3Config {
            k: 1,
            etas: vec![0.9],
            thetas: vec![0.1],
            weights: vec![1.0],
            frequencies: vec![1],
            use_newton_schulz: false,
            ns_iterations: 5,
            ns_dim: None,
        }
    }

    /// Two-level: fast (every step, eta=0.9) + slow (every 8th, eta=0.99).
    pub fn default_k2() -> Self {
        M3Config {
            k: 2,
            etas: vec![0.9, 0.99],
            thetas: vec![0.1, 0.01],
            weights: vec![1.0, 1.0],
            frequencies: vec![1, 8],
            use_newton_schulz: false,
            ns_iterations: 5,
            ns_dim: None,
        }
    }

    /// Four-level matching CMS frequencies [1, 8, 64, 512].
    pub fn default_k4() -> Self {
        M3Config {
            k: 4,
            etas: vec![0.9, 0.95, 0.99, 0.999],
            thetas: vec![0.1, 0.05, 0.01, 0.001],
            weights: vec![1.0, 1.0, 1.0, 1.0],
            frequencies: vec![1, 8, 64, 512],
            use_newton_schulz: false,
            ns_iterations: 5,
            ns_dim: None,
        }
    }

    /// Validate configuration consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            return Err("k must be >= 1".into());
        }
        if self.etas.len() != self.k {
            return Err(format!("etas length {} != k {}", self.etas.len(), self.k));
        }
        if self.thetas.len() != self.k {
            return Err(format!("thetas length {} != k {}", self.thetas.len(), self.k));
        }
        if self.weights.len() != self.k {
            return Err(format!("weights length {} != k {}", self.weights.len(), self.k));
        }
        if self.frequencies.len() != self.k {
            return Err(format!("frequencies length {} != k {}", self.frequencies.len(), self.k));
        }
        for (i, &f) in self.frequencies.iter().enumerate() {
            if f == 0 {
                return Err(format!("frequencies[{i}] must be >= 1"));
            }
        }
        if self.use_newton_schulz && self.ns_dim.is_none() {
            return Err("ns_dim must be set when use_newton_schulz is true".into());
        }
        Ok(())
    }
}

// ── State ─────────────────────────────────────────────────────────────

/// M3 optimizer state: k momentum accumulators + k error buffers.
pub struct M3State {
    /// k momentum accumulators, each total_params long.
    pub momentum: Vec<Vec<f32>>,
    /// k error buffers for frozen levels.
    pub error_accum: Vec<Vec<f32>>,
    /// Per-level count of accumulated gradient steps.
    pub steps_accumulated: Vec<usize>,
    /// Global step counter (advanced by caller, not internally).
    pub step: usize,
    /// Total number of parameters being optimized.
    pub total_params: usize,
}

impl M3State {
    /// Allocate k momentum + error buffers for `total_params` parameters.
    pub fn new(cfg: &M3Config, total_params: usize) -> Self {
        M3State {
            momentum: vec![vec![0.0f32; total_params]; cfg.k],
            error_accum: vec![vec![0.0f32; total_params]; cfg.k],
            steps_accumulated: vec![0; cfg.k],
            step: 0,
            total_params,
        }
    }
}

// ── Core step ─────────────────────────────────────────────────────────

/// Check if optimizer level is active at the given step.
#[inline]
pub fn m3_is_active(cfg: &M3Config, step: usize, level: usize) -> bool {
    step % cfg.frequencies[level] == 0
}

/// One M3 optimizer step: accumulate/update per-level, combine, return update vector.
///
/// The returned Vec<f32> is the combined momentum update to be applied as:
///   param[i] -= update[i]
/// (Note: thetas already scale the gradient, so no separate lr needed.)
pub fn m3_step(state: &mut M3State, cfg: &M3Config, grad: &[f32]) -> Vec<f32> {
    assert_eq!(grad.len(), state.total_params,
        "grad length {} != total_params {}", grad.len(), state.total_params);

    let n = state.total_params;

    // Step 1: per-level accumulate or update
    for level in 0..cfg.k {
        if m3_is_active(cfg, state.step, level) {
            // Active: EMA update with accumulated error + current gradient
            let eta = cfg.etas[level];
            let theta = cfg.thetas[level];
            for i in 0..n {
                let combined = state.error_accum[level][i] + grad[i];
                state.momentum[level][i] = eta * state.momentum[level][i] + theta * combined;
            }
            // Reset error buffer
            state.error_accum[level].iter_mut().for_each(|v| *v = 0.0);
            state.steps_accumulated[level] = 0;
        } else {
            // Frozen: accumulate gradient for later
            for i in 0..n {
                state.error_accum[level][i] += grad[i];
            }
            state.steps_accumulated[level] += 1;
        }
    }

    // Step 2: weighted combination across levels
    let mut combined = vec![0.0f32; n];
    for level in 0..cfg.k {
        let w = cfg.weights[level];
        for i in 0..n {
            combined[i] += w * state.momentum[level][i];
        }
    }

    // Step 3: optional Newton-Schulz orthogonalization
    // Only apply when ns_dim is explicitly set and matches total_params.
    // This prevents silently orthogonalizing unrelated flattened tensors.
    if cfg.use_newton_schulz {
        if let Some(d) = cfg.ns_dim {
            assert_eq!(d * d, n,
                "ns_dim={d} does not match total_params={n} (expected {d}x{d}={})", d * d);
            combined = newton_schulz_5(&combined, d, cfg.ns_iterations);
        }
    }

    // Advance step counter
    state.step += 1;

    combined
}

// ── Newton-Schulz orthogonalization ───────────────────────────────────

/// Newton-Schulz iteration for approximate matrix orthogonalization.
///
/// Iterates X_{k+1} = 0.5 * X_k * (3I - X_k^T * X_k) for `iterations` steps.
/// Input: flat d×d matrix S. Output: orthogonalized flat d×d matrix.
///
/// Used by Muon optimizer. Converges when ||X^T X - I|| < 1.
/// We pre-normalize by spectral norm estimate to ensure convergence.
pub fn newton_schulz_5(s: &[f32], d: usize, iterations: usize) -> Vec<f32> {
    assert_eq!(s.len(), d * d, "newton_schulz_5: expected {}x{} matrix, got len {}", d, d, s.len());

    if d == 0 || iterations == 0 {
        return s.to_vec();
    }

    // Estimate spectral norm (Frobenius norm / sqrt(d) as upper bound proxy)
    let frob: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
    if frob < 1e-12 {
        return s.to_vec(); // near-zero matrix, nothing to orthogonalize
    }
    let scale = 1.0 / frob; // conservative normalization

    // X = S / ||S||_F (ensures convergence)
    let mut x: Vec<f32> = s.iter().map(|&v| v * scale).collect();

    for _ in 0..iterations {
        // Compute X^T X (d×d)
        let xtx = matmul_t_a(&x, &x, d);

        // Compute 3I - X^T X
        let mut three_i_minus_xtx = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let idx = i * d + j;
                let identity = if i == j { 3.0 } else { 0.0 };
                three_i_minus_xtx[idx] = identity - xtx[idx];
            }
        }

        // X_new = 0.5 * X * (3I - X^T X)
        let xm = matmul(&x, &three_i_minus_xtx, d);
        for i in 0..x.len() {
            x[i] = 0.5 * xm[i];
        }
    }

    // Rescale back by Frobenius norm
    for v in x.iter_mut() {
        *v *= frob;
    }

    x
}

/// Inner-loop Newton-Schulz iteration with generalized per-iteration zeta step sizes.
///
/// Implements the gradient-descent form from HOPE Eq 44:
///   O_{i+1} = O_i - zeta_{i+1} * (O_i - G + 2 * O_i @ (O_i^T @ O_i - I))
///
/// Unlike `newton_schulz_5` (which uses the classical polynomial form for the outer loop),
/// this keeps G as an anchor throughout the iteration — the update is pulled toward the
/// original input, not just toward orthogonality. This matches Atlas's inner-loop usage
/// where the momentum signal S_t should be orthogonalized while preserving its relationship
/// to the accumulated gradients.
///
/// When all zetas = 1.0, this reduces to:
///   O_{i+1} = 2*O_i - 2*O_i@O_i^T@O_i + G
///
/// # Arguments
/// * `s` — input momentum/gradient matrix [d*d], row-major
/// * `d` — matrix dimension (s must be d×d)
/// * `iterations` — number of NS iterations (Atlas default: 5)
/// * `zetas` — per-iteration step sizes, length must equal `iterations`.
///   **Warning**: `None` defaults to all 1.0, which diverges for non-orthogonal
///   inputs (the gradient term grows cubically in ||O||). Use `zeta ≈ 0.25` for
///   stable convergence matching the classical NS polynomial coefficients (3/2, -1/2).
///
/// Source: HOPE (2512.24695) §4.2 Eq 44; Atlas (2505.23735) §5 Eq 32
pub fn newton_schulz_inner(
    s: &[f32],
    d: usize,
    iterations: usize,
    zetas: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(s.len(), d * d,
        "newton_schulz_inner: expected {}x{} matrix, got len {}", d, d, s.len());

    if let Some(z) = zetas {
        assert_eq!(z.len(), iterations,
            "newton_schulz_inner: zetas length {} != iterations {}", z.len(), iterations);
    }

    if d == 0 || iterations == 0 {
        return s.to_vec();
    }

    // Frobenius normalization for convergence
    let frob: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
    if frob < 1e-12 {
        return s.to_vec();
    }
    let scale = 1.0 / frob;

    // G = normalized input (anchor throughout iteration)
    let g: Vec<f32> = s.iter().map(|&v| v * scale).collect();
    let mut o = g.clone();

    for iter in 0..iterations {
        let zeta = zetas.map_or(1.0, |z| z[iter]);

        // O^T @ O
        let oto = matmul_t_a(&o, &o, d);

        // O^T @ O - I
        let mut oto_minus_i = oto;
        for i in 0..d {
            oto_minus_i[i * d + i] -= 1.0;
        }

        // 2 * O @ (O^T O - I)
        let o_oto_i = matmul(&o, &oto_minus_i, d);

        // O_{i+1} = O_i - zeta * (O_i - G + 2 * O @ (O^T O - I))
        for i in 0..d * d {
            let grad = o[i] - g[i] + 2.0 * o_oto_i[i];
            o[i] -= zeta * grad;
        }
    }

    // Rescale back
    for v in o.iter_mut() {
        *v *= frob;
    }

    o
}

/// Batched Newton-Schulz: apply inner-loop NS independently to C matrices.
///
/// In Atlas's chunk-parallel training (§5.1 Eqs 39-41), each position in a chunk
/// has an independent momentum S_t that needs NS orthogonalization. These C calls
/// are embarrassingly parallel — on GPU this would be a single batched operation.
/// This CPU reference implementation processes them sequentially.
///
/// # Arguments
/// * `batch` — C concatenated d×d matrices [C * d * d], row-major
/// * `c` — batch size (number of matrices)
/// * `d` — matrix dimension per element
/// * `iterations` — NS iterations per matrix
/// * `zetas` — shared per-iteration step sizes (applied to all C matrices)
pub fn newton_schulz_batched(
    batch: &[f32],
    c: usize,
    d: usize,
    iterations: usize,
    zetas: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(batch.len(), c * d * d,
        "newton_schulz_batched: expected {}*{}*{}={} elements, got {}",
        c, d, d, c * d * d, batch.len());

    let dd = d * d;
    let mut out = vec![0.0f32; c * dd];

    for b in 0..c {
        let input = &batch[b * dd..(b + 1) * dd];
        let result = newton_schulz_inner(input, d, iterations, zetas);
        out[b * dd..(b + 1) * dd].copy_from_slice(&result);
    }

    out
}

/// Matrix multiply: C = A * B (all d×d, row-major).
fn matmul(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; d * d];
    for i in 0..d {
        for k in 0..d {
            let a_ik = a[i * d + k];
            for j in 0..d {
                c[i * d + j] += a_ik * b[k * d + j];
            }
        }
    }
    c
}

/// Matrix multiply: C = A^T * B (all d×d, row-major).
fn matmul_t_a(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; d * d];
    for i in 0..d {
        for k in 0..d {
            let a_ki = a[k * d + i]; // A^T[i][k] = A[k][i]
            for j in 0..d {
                c[i * d + j] += a_ki * b[k * d + j];
            }
        }
    }
    c
}

// ── Flatten/Unflatten helpers for MAGParams ───────────────────────────

/// Flatten all MAGParams weight vectors into a single contiguous Vec<f32>.
///
/// Order: SWA (embed, q, k, v, o, unembed) then each level
/// (k_mem, v_mem, q_mem, alpha, b_alpha, theta, b_theta, eta, b_eta, w_omega, w_freq, b_freq).
pub fn flatten_mag_params(params: &MAGParams) -> Vec<f32> {
    let mut flat = Vec::with_capacity(params.num_params());

    // SWA
    flat.extend_from_slice(&params.swa.w_embed);
    flat.extend_from_slice(&params.swa.w_q);
    flat.extend_from_slice(&params.swa.w_k);
    flat.extend_from_slice(&params.swa.w_v);
    flat.extend_from_slice(&params.swa.w_o);
    flat.extend_from_slice(&params.swa.w_unembed);

    // Memory levels
    for level in &params.levels {
        flat.extend_from_slice(&level.w_k_mem);
        flat.extend_from_slice(&level.w_v_mem);
        flat.extend_from_slice(&level.w_q_mem);
        flat.extend_from_slice(&level.w_alpha);
        flat.extend_from_slice(&level.b_alpha);
        flat.extend_from_slice(&level.w_theta);
        flat.extend_from_slice(&level.b_theta);
        flat.extend_from_slice(&level.w_eta);
        flat.extend_from_slice(&level.b_eta);
        flat.extend_from_slice(&level.w_omega);
        flat.extend_from_slice(&level.w_freq);
        flat.extend_from_slice(&level.b_freq);
        flat.extend_from_slice(&level.w_k_conv);
        flat.extend_from_slice(&level.b_k_conv);
        flat.extend_from_slice(&level.w_q_conv);
        flat.extend_from_slice(&level.b_q_conv);
    }

    // CMS aggregation weights
    flat.extend_from_slice(&params.alpha_mem);
    flat.extend_from_slice(&params.alpha_refl);

    // Persistent tokens
    flat.extend_from_slice(&params.persistent_tokens);

    flat
}

/// Split a flat gradient vector back into MAGParams structure.
///
/// Uses `template` to determine the sizes of each field.
pub fn unflatten_to_mag_grads(flat: &[f32], template: &MAGParams) -> MAGParams {
    let expected = template.num_params();
    assert_eq!(flat.len(), expected,
        "unflatten: flat len {} != template num_params {}", flat.len(), expected);

    let mut offset = 0usize;

    fn take(flat: &[f32], offset: &mut usize, len: usize) -> Vec<f32> {
        let slice = flat[*offset..*offset + len].to_vec();
        *offset += len;
        slice
    }

    let swa = SWAParams {
        w_embed: take(flat, &mut offset, template.swa.w_embed.len()),
        w_q: take(flat, &mut offset, template.swa.w_q.len()),
        w_k: take(flat, &mut offset, template.swa.w_k.len()),
        w_v: take(flat, &mut offset, template.swa.w_v.len()),
        w_o: take(flat, &mut offset, template.swa.w_o.len()),
        w_unembed: take(flat, &mut offset, template.swa.w_unembed.len()),
    };

    let mut levels = Vec::with_capacity(template.levels.len());
    for tl in &template.levels {
        levels.push(MemoryLevelParams {
            w_k_mem: take(flat, &mut offset, tl.w_k_mem.len()),
            w_v_mem: take(flat, &mut offset, tl.w_v_mem.len()),
            w_q_mem: take(flat, &mut offset, tl.w_q_mem.len()),
            w_alpha: take(flat, &mut offset, tl.w_alpha.len()),
            b_alpha: take(flat, &mut offset, tl.b_alpha.len()),
            w_theta: take(flat, &mut offset, tl.w_theta.len()),
            b_theta: take(flat, &mut offset, tl.b_theta.len()),
            w_eta: take(flat, &mut offset, tl.w_eta.len()),
            b_eta: take(flat, &mut offset, tl.b_eta.len()),
            w_omega: take(flat, &mut offset, tl.w_omega.len()),
            w_freq: take(flat, &mut offset, tl.w_freq.len()),
            b_freq: take(flat, &mut offset, tl.b_freq.len()),
            w_k_conv: take(flat, &mut offset, tl.w_k_conv.len()),
            b_k_conv: take(flat, &mut offset, tl.b_k_conv.len()),
            w_q_conv: take(flat, &mut offset, tl.w_q_conv.len()),
            b_q_conv: take(flat, &mut offset, tl.b_q_conv.len()),
        });
    }

    let alpha_mem = take(flat, &mut offset, template.alpha_mem.len());
    let alpha_refl = take(flat, &mut offset, template.alpha_refl.len());
    let persistent_tokens = take(flat, &mut offset, template.persistent_tokens.len());

    assert_eq!(offset, flat.len(),
        "unflatten consumed {} of {} elements", offset, flat.len());

    MAGParams { swa, levels, alpha_mem, alpha_refl, persistent_tokens }
}

/// Total parameter count for a MAGParams instance.
pub fn mag_params_count(params: &MAGParams) -> usize {
    params.num_params()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m3_config_defaults() {
        let k1 = M3Config::default_k1();
        assert_eq!(k1.k, 1);
        assert!(k1.validate().is_ok());

        let k2 = M3Config::default_k2();
        assert_eq!(k2.k, 2);
        assert!(k2.validate().is_ok());

        let k4 = M3Config::default_k4();
        assert_eq!(k4.k, 4);
        assert!(k4.validate().is_ok());
    }

    #[test]
    fn test_m3_config_validate_rejects_bad() {
        let mut bad = M3Config::default_k1();
        bad.k = 0;
        assert!(bad.validate().is_err());

        let mut bad2 = M3Config::default_k2();
        bad2.etas = vec![0.9]; // length mismatch
        assert!(bad2.validate().is_err());

        let mut bad3 = M3Config::default_k1();
        bad3.frequencies = vec![0]; // zero freq
        assert!(bad3.validate().is_err());
    }

    #[test]
    fn test_m3_state_init() {
        let cfg = M3Config::default_k2();
        let state = M3State::new(&cfg, 100);
        assert_eq!(state.momentum.len(), 2);
        assert_eq!(state.error_accum.len(), 2);
        assert_eq!(state.momentum[0].len(), 100);
        assert_eq!(state.error_accum[1].len(), 100);
        assert_eq!(state.step, 0);
        assert!(state.momentum[0].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_m3_is_active() {
        let cfg = M3Config::default_k2();
        // Level 0: freq=1, always active
        assert!(m3_is_active(&cfg, 0, 0));
        assert!(m3_is_active(&cfg, 7, 0));
        // Level 1: freq=8, active at 0, 8, 16...
        assert!(m3_is_active(&cfg, 0, 1));
        assert!(!m3_is_active(&cfg, 1, 1));
        assert!(!m3_is_active(&cfg, 7, 1));
        assert!(m3_is_active(&cfg, 8, 1));
        assert!(m3_is_active(&cfg, 16, 1));
    }

    #[test]
    fn test_matmul_identity() {
        let d = 3;
        let mut identity = vec![0.0f32; d * d];
        for i in 0..d { identity[i * d + i] = 1.0; }
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let c = matmul(&a, &identity, d);
        for i in 0..d*d {
            assert!((c[i] - a[i]).abs() < 1e-6, "matmul identity failed at {i}");
        }
    }

    #[test]
    fn test_newton_schulz_identity() {
        let d = 4;
        let mut identity = vec![0.0f32; d * d];
        for i in 0..d { identity[i * d + i] = 1.0; }
        let result = newton_schulz_5(&identity, d, 5);
        // NS preserves Frobenius norm. Identity has frob=2, so result ≈ 2*I.
        // Check that result is proportional to identity (diagonal, off-diag ≈ 0).
        let diag_val = result[0]; // should be frob_norm / sqrt(d) * d... = ~2.0
        assert!(diag_val > 0.5, "NS identity: diagonal should be positive: {diag_val}");
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    assert!((result[i * d + j] - diag_val).abs() < 0.1,
                        "NS identity: [{i},{j}] = {} expected ~{diag_val}", result[i * d + j]);
                } else {
                    assert!(result[i * d + j].abs() < 0.1,
                        "NS identity: off-diag [{i},{j}] = {} expected ~0", result[i * d + j]);
                }
            }
        }
    }

    #[test]
    fn test_newton_schulz_orthogonal() {
        // Start with a non-orthogonal matrix, verify NS moves it toward orthogonal
        let d = 3;
        let m = vec![1.0, 0.5, 0.2,
                     0.3, 1.0, 0.1,
                     0.1, 0.2, 1.0];
        let result = newton_schulz_5(&m, d, 10);

        // Check X^T X ≈ c*I for some scalar c (orthogonal up to scale)
        let xtx = matmul_t_a(&result, &result, d);
        // Off-diagonal elements should be small relative to diagonal
        let diag_avg = (0..d).map(|i| xtx[i * d + i]).sum::<f32>() / d as f32;
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let ratio = xtx[i * d + j].abs() / diag_avg;
                    assert!(ratio < 0.15,
                        "NS orthogonal: off-diag [{i},{j}] ratio = {ratio}");
                }
            }
        }
    }

    // ── Inner-loop Newton-Schulz tests ──────────────────────────────────

    #[test]
    fn test_ns_inner_identity() {
        let d = 4;
        let mut identity = vec![0.0f32; d * d];
        for i in 0..d { identity[i * d + i] = 1.0; }
        let result = newton_schulz_inner(&identity, d, 5, None);
        // Identity is already orthogonal — output should be proportional to identity
        let diag_val = result[0];
        assert!(diag_val > 0.5, "NS inner identity: diag should be positive: {diag_val}");
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    assert!((result[i * d + j] - diag_val).abs() < 0.1,
                        "NS inner: [{i},{j}] = {} expected ~{diag_val}", result[i * d + j]);
                } else {
                    assert!(result[i * d + j].abs() < 0.1,
                        "NS inner: off-diag [{i},{j}] = {} expected ~0", result[i * d + j]);
                }
            }
        }
    }

    #[test]
    fn test_ns_inner_orthogonalizes() {
        // The general form is gradient descent on L_orth + proximity to G.
        // zeta=0.25 gives first-step coefficients matching the classical NS polynomial,
        // so convergence is similar. zeta=1.0 diverges (step too aggressive).
        let d = 3;
        let m = vec![1.0, 0.5, 0.2,
                     0.3, 1.0, 0.1,
                     0.1, 0.2, 1.0];
        let zetas = vec![0.25; 10];
        let result = newton_schulz_inner(&m, d, 10, Some(&zetas));

        // Check off-diagonal of X^T X is small relative to diagonal
        let xtx = matmul_t_a(&result, &result, d);
        let diag_avg = (0..d).map(|i| xtx[i * d + i]).sum::<f32>() / d as f32;
        assert!(diag_avg > 0.0, "diagonal average should be positive, got {diag_avg}");
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    let ratio = xtx[i * d + j].abs() / diag_avg;
                    assert!(ratio < 0.2,
                        "NS inner orthogonal: off-diag [{i},{j}] ratio = {ratio}");
                }
            }
        }
    }

    #[test]
    fn test_ns_inner_custom_zetas() {
        let d = 3;
        let m = vec![1.0, 0.5, 0.2,
                     0.3, 1.0, 0.1,
                     0.1, 0.2, 1.0];
        // Conservative step sizes (smaller zeta = more cautious convergence)
        let zetas = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let result = newton_schulz_inner(&m, d, 5, Some(&zetas));
        // Should still produce finite, non-zero output
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "NS inner zeta: result[{i}] not finite: {v}");
        }
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.1, "NS inner zeta: result should be non-trivial, norm={norm}");
    }

    #[test]
    fn test_ns_inner_zeta_one_deterministic() {
        // Same input, same zetas → same output
        let d = 4;
        let m: Vec<f32> = (0..d*d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let r1 = newton_schulz_inner(&m, d, 5, None);
        let r2 = newton_schulz_inner(&m, d, 5, None);
        assert_eq!(r1, r2, "NS inner should be deterministic");
    }

    #[test]
    fn test_ns_inner_zero_iterations() {
        let d = 3;
        let m = vec![1.0; d * d];
        let result = newton_schulz_inner(&m, d, 0, None);
        assert_eq!(result, m, "0 iterations should return input unchanged");
    }

    #[test]
    fn test_ns_inner_near_zero() {
        let d = 3;
        let m = vec![1e-15; d * d];
        let result = newton_schulz_inner(&m, d, 5, None);
        // Near-zero matrix should be returned as-is
        assert_eq!(result, m);
    }

    // ── Batched Newton-Schulz tests ─────────────────────────────────────

    #[test]
    fn test_ns_batched_matches_individual() {
        let d = 3;
        let c = 4;
        let dd = d * d;

        // Create C distinct matrices
        let mut batch = vec![0.0f32; c * dd];
        for b in 0..c {
            for i in 0..dd {
                batch[b * dd + i] = ((b * dd + i) as f32 + 1.0) * 0.05;
            }
        }

        let batched = newton_schulz_batched(&batch, c, d, 5, None);

        // Compare each element with individual NS call
        for b in 0..c {
            let individual = newton_schulz_inner(&batch[b * dd..(b + 1) * dd], d, 5, None);
            for i in 0..dd {
                let diff = (batched[b * dd + i] - individual[i]).abs();
                assert!(diff < 1e-6,
                    "batch[{b}][{i}]: batched={} vs individual={}, diff={diff}",
                    batched[b * dd + i], individual[i]);
            }
        }
    }

    #[test]
    fn test_ns_batched_with_zetas() {
        let d = 3;
        let c = 2;
        let dd = d * d;
        let mut batch = vec![0.0f32; c * dd];
        for i in 0..c * dd {
            batch[i] = (i as f32 + 1.0) * 0.03;
        }
        let zetas = vec![0.8, 0.8, 0.8];
        let result = newton_schulz_batched(&batch, c, d, 3, Some(&zetas));
        assert_eq!(result.len(), c * dd);
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "batched with zetas: result[{i}] not finite");
        }
    }

    #[test]
    fn test_ns_batched_single_element() {
        let d = 3;
        let m = vec![1.0, 0.5, 0.2,
                     0.3, 1.0, 0.1,
                     0.1, 0.2, 1.0];
        let batched = newton_schulz_batched(&m, 1, d, 5, None);
        let single = newton_schulz_inner(&m, d, 5, None);
        assert_eq!(batched, single, "batch of 1 should match single call");
    }
}
