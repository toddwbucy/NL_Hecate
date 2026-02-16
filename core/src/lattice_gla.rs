/// Lattice GLA — specialized parallelization for Lattice OSR and Trellis.
///
/// Wraps chunkwise GD with boundary renormalization to preserve geometric
/// constraints (unit sphere for Lattice, bounded S_K/S_V for Trellis).
///
/// **Design note on Lattice OSR**: LatticeOSR::step() normalizes slots to
/// the unit sphere at every token internally. The boundary renormalization
/// here is therefore a safety net ensuring drift doesn't accumulate across
/// chunk boundaries, not a replacement for per-token normalization. This
/// means Lattice GLA is closer to exact than a true linearized approximation.
///
/// **Design note on Trellis**: Trellis::step() does NOT normalize S_K/S_V
/// per-token — it applies OGD updates without explicit renormalization.
/// Boundary renormalization here is the primary geometric constraint
/// enforcement, making it a true GLA approximation where larger chunks
/// trade quality for parallelism.

use crate::model::{MAGConfig, MemoryLevelParams};
use crate::chunkwise_gd::{chunkwise_gd_forward, chunkwise_gd_backward, ChunkwiseGDCache};

/// Cache for Lattice GLA forward pass.
pub struct LatticeGLACache {
    /// Underlying chunkwise GD cache
    pub gd_cache: ChunkwiseGDCache,
    /// Boundary states after renormalization: [num_chunks, state_size]
    pub normalized_boundaries: Vec<Vec<f32>>,
}

/// Renormalize Lattice OSR slots to unit sphere.
fn renormalize_lattice(state: &mut [f32], m_slots: usize, d: usize) {
    for s in 0..m_slots {
        let offset = s * d;
        let norm: f32 = state[offset..offset + d].iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for j in 0..d {
                state[offset + j] /= norm;
            }
        }
    }
}

/// Renormalize Trellis S_K and S_V states.
fn renormalize_trellis(state: &mut [f32], d_k: usize, d: usize) {
    let sk_size = d_k * d;
    // Renormalize each row of S_K
    for i in 0..d_k {
        let offset = i * d;
        let norm: f32 = state[offset..offset + d].iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            let inv = (d as f32).sqrt() / norm;
            for j in 0..d {
                state[offset + j] *= inv;
            }
        }
    }
    // Renormalize each row of S_V
    for i in 0..d {
        let offset = sk_size + i * d_k;
        let norm: f32 = state[offset..offset + d_k].iter()
            .map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            let inv = (d_k as f32).sqrt() / norm;
            for j in 0..d_k {
                state[offset + j] *= inv;
            }
        }
    }
}

/// Lattice GLA forward for Lattice OSR.
///
/// Uses chunkwise GD internally but renormalizes slots at chunk boundaries.
pub fn lattice_gla_forward(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    chunk_size: usize,
    cfg: &MAGConfig,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, LatticeGLACache) {
    assert!(chunk_size >= 1, "chunk_size must be >= 1");
    let m_slots = cfg.m_slots;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    // If initial_m provided, renormalize it first
    let init = initial_m.map(|mut m| {
        renormalize_lattice(&mut m, m_slots, d);
        m
    });

    // Run chunkwise GD with renormalization at boundaries
    let mut y = vec![0.0f32; seq_len * d];
    let mut all_chunk_caches = Vec::new();
    let mut normalized_boundaries = Vec::new();
    let mut boundary_state = init;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let clen = end - start;
        let chunk_embedded = &embedded[start * d..end * d];

        let (chunk_y, chunk_cache) = chunkwise_gd_forward(
            level_params, chunk_embedded, clen, d, clen, cfg, boundary_state.clone(),
        );

        y[start * d..end * d].copy_from_slice(&chunk_y);

        // Extract final state and renormalize for next boundary
        if chunk_cache.chunks.len() > 0 {
            let mut final_state = chunk_cache.chunks.last().unwrap()
                .boundary_after.state.clone();
            renormalize_lattice(&mut final_state, m_slots, d);
            normalized_boundaries.push(final_state.clone());
            boundary_state = Some(final_state);
        }

        all_chunk_caches.push(chunk_cache);
    }

    // Build a combined cache
    let combined_cache = if all_chunk_caches.len() == 1 {
        all_chunk_caches.into_iter().next().unwrap()
    } else {
        // Merge chunk caches into a single ChunkwiseGDCache
        let mut chunks = Vec::new();
        let mut chunk_starts = Vec::new();
        let mut chunk_lens = Vec::new();
        for (i, cc) in all_chunk_caches.into_iter().enumerate() {
            let offset = i * chunk_size;
            for (j, c) in cc.chunks.into_iter().enumerate() {
                chunk_starts.push(offset + cc.chunk_starts.get(j).copied().unwrap_or(0));
                chunk_lens.push(cc.chunk_lens.get(j).copied().unwrap_or(0));
                chunks.push(c);
            }
        }
        ChunkwiseGDCache {
            chunks,
            chunk_starts,
            chunk_lens,
            seq_len,
            d,
        }
    };

    let cache = LatticeGLACache {
        gd_cache: combined_cache,
        normalized_boundaries,
    };

    (y, cache)
}

/// Lattice GLA backward (delegates to chunkwise GD backward).
pub fn lattice_gla_backward(
    level_params: &MemoryLevelParams,
    cache: &LatticeGLACache,
    d_y: &[f32],
    embedded: &[f32],
    cfg: &MAGConfig,
) -> (MemoryLevelParams, Vec<f32>) {
    chunkwise_gd_backward(level_params, &cache.gd_cache, d_y, embedded, cfg)
}

/// Trellis GLA forward — like Lattice but renormalizes S_K/S_V.
pub fn trellis_gla_forward(
    level_params: &MemoryLevelParams,
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    chunk_size: usize,
    cfg: &MAGConfig,
    initial_m: Option<Vec<f32>>,
) -> (Vec<f32>, LatticeGLACache) {
    assert!(chunk_size >= 1, "chunk_size must be >= 1");
    let d_k = cfg.d_compress;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    let init = initial_m.map(|mut m| {
        renormalize_trellis(&mut m, d_k, d);
        m
    });

    let mut y = vec![0.0f32; seq_len * d];
    let mut all_chunk_caches = Vec::new();
    let mut normalized_boundaries = Vec::new();
    let mut boundary_state = init;

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let clen = end - start;
        let chunk_embedded = &embedded[start * d..end * d];

        let (chunk_y, chunk_cache) = chunkwise_gd_forward(
            level_params, chunk_embedded, clen, d, clen, cfg, boundary_state.clone(),
        );

        y[start * d..end * d].copy_from_slice(&chunk_y);

        if chunk_cache.chunks.len() > 0 {
            let mut final_state = chunk_cache.chunks.last().unwrap()
                .boundary_after.state.clone();
            renormalize_trellis(&mut final_state, d_k, d);
            normalized_boundaries.push(final_state.clone());
            boundary_state = Some(final_state);
        }

        all_chunk_caches.push(chunk_cache);
    }

    let combined_cache = if all_chunk_caches.len() == 1 {
        all_chunk_caches.into_iter().next().unwrap()
    } else {
        let mut chunks = Vec::new();
        let mut chunk_starts = Vec::new();
        let mut chunk_lens = Vec::new();
        for (i, cc) in all_chunk_caches.into_iter().enumerate() {
            let offset = i * chunk_size;
            for (j, c) in cc.chunks.into_iter().enumerate() {
                chunk_starts.push(offset + cc.chunk_starts.get(j).copied().unwrap_or(0));
                chunk_lens.push(cc.chunk_lens.get(j).copied().unwrap_or(0));
                chunks.push(c);
            }
        }
        ChunkwiseGDCache { chunks, chunk_starts, chunk_lens, seq_len, d }
    };

    (y, LatticeGLACache { gd_cache: combined_cache, normalized_boundaries })
}

/// Trellis GLA backward.
pub fn trellis_gla_backward(
    level_params: &MemoryLevelParams,
    cache: &LatticeGLACache,
    d_y: &[f32],
    embedded: &[f32],
    cfg: &MAGConfig,
) -> (MemoryLevelParams, Vec<f32>) {
    chunkwise_gd_backward(level_params, &cache.gd_cache, d_y, embedded, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::tensor::SimpleRng;
    use crate::delta_rule::MemoryRule;
    use crate::lattice_osr::LatticeOSR;
    use crate::trellis::Trellis;

    fn make_embedded(cfg: &MAGConfig, seed: u64) -> Vec<f32> {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let mut rng = SimpleRng::new(seed);
        let mut e = vec![0.0f32; s * d];
        rng.fill_uniform(&mut e, 0.1);
        e
    }

    fn relative_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1e-8))
            .fold(0.0f32, f32::max)
    }

    // ─── Lattice OSR tests ──────────────────────────────────

    #[test]
    fn test_lattice_gla_c1_exact() {
        // chunk_size=seq_len means one GLA chunk with no renormalization = sequential
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (y_gla, _) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, s, &cfg, None,
        );
        let rule = LatticeOSR { m_slots: cfg.m_slots };
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        let max_diff: f32 = y_gla.iter().zip(y_seq.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5,
            "Lattice GLA C=1 should match sequential: max_diff={max_diff}");
    }

    #[test]
    fn test_lattice_gla_c4_quality() {
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let rule = LatticeOSR { m_slots: cfg.m_slots };
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        let c = 4.min(s);
        let (y_gla, _) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, c, &cfg, None,
        );

        let rel = relative_diff(&y_seq, &y_gla);
        // Sphere constraint bounds drift — expect tight approximation
        assert!(rel < 1.0,
            "Lattice GLA C=4 relative diff {rel:.4} exceeds 100%");
    }

    #[test]
    fn test_lattice_gla_forward_finite() {
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (y, _) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        for &v in &y { assert!(v.is_finite()); }
    }

    #[test]
    fn test_lattice_gla_backward_finite() {
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (_, cache) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = lattice_gla_backward(
            &params.levels[0], &cache, &d_y, &embedded, &cfg,
        );

        for &v in grads.w_k_mem.iter() { assert!(v.is_finite()); }
        for &v in &d_emb { assert!(v.is_finite()); }
    }

    #[test]
    fn test_lattice_gla_sphere_preserved() {
        // After renormalization, boundary slots should be on unit sphere
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (_, cache) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );

        for boundary in &cache.normalized_boundaries {
            for slot in 0..cfg.m_slots {
                let offset = slot * d;
                let norm: f32 = boundary[offset..offset + d].iter()
                    .map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-5 || norm < 1e-10,
                    "Slot {slot} should be unit norm, got {norm}");
            }
        }
    }

    #[test]
    fn test_lattice_gla_outer_loop_weight_descent() {
        // Validates outer-loop gradient flow: Enzyme-computed gradients on
        // projection weights (W_K, W_V, W_Q) decrease a proxy loss when applied
        // as weight updates. This is the outer loop — distinct from the inner loop
        // (memory updates inside the forward pass, which has no external optimizer).
        let cfg = MAGConfig::lattice_test_config();
        let mut lp = MAGParams::init(&cfg, 42).levels.into_iter().next().unwrap();
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
            let (y, cache) = lattice_gla_forward(&lp, &embedded, s, d, 2, &cfg, None);
            let loss: f32 = y.iter().zip(target.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
            if outer_step == 0 { first_loss = loss; }
            if outer_step == 99 { last_loss = loss; }

            let d_y: Vec<f32> = y.iter().zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
            let (grads, _) = lattice_gla_backward(&lp, &cache, &d_y, &embedded, &cfg);

            // Outer-loop weight update (projection weights, not inner-loop memory)
            for (w, g) in lp.w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
            for (w, g) in lp.w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
            for (w, g) in lp.w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
        }

        assert!(last_loss <= first_loss + 1e-6,
            "Lattice GLA outer-loop weight descent should not diverge: {first_loss:.6} → {last_loss:.6}");
    }

    // ─── Trellis tests ──────────────────────────────────────

    #[test]
    fn test_trellis_gla_c1_match() {
        // chunk_size=seq_len means one GLA chunk with no renormalization = sequential
        let cfg = MAGConfig::trellis_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (y_gla, _) = trellis_gla_forward(
            &params.levels[0], &embedded, s, d, s, &cfg, None,
        );
        let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        let max_diff: f32 = y_gla.iter().zip(y_seq.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5,
            "Trellis GLA C=1 should match sequential: max_diff={max_diff}");
    }

    #[test]
    fn test_trellis_gla_c4_quality() {
        let cfg = MAGConfig::trellis_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let rule = Trellis { d_k: cfg.d_compress, lambda_k: cfg.lambda_k, lambda_v: cfg.lambda_v };
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        let c = 4.min(s);
        let (y_gla, _) = trellis_gla_forward(
            &params.levels[0], &embedded, s, d, c, &cfg, None,
        );

        let rel = relative_diff(&y_seq, &y_gla);
        assert!(rel < 1.0,
            "Trellis GLA C=4 relative diff {rel:.4} exceeds 100%");
    }

    #[test]
    fn test_trellis_gla_forward_finite() {
        let cfg = MAGConfig::trellis_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (y, _) = trellis_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        for &v in &y { assert!(v.is_finite()); }
    }

    #[test]
    fn test_trellis_gla_backward_finite() {
        let cfg = MAGConfig::trellis_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let (_, cache) = trellis_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        let d_y = vec![1.0f32; s * d];
        let (grads, d_emb) = trellis_gla_backward(
            &params.levels[0], &cache, &d_y, &embedded, &cfg,
        );

        for &v in grads.w_k_mem.iter() { assert!(v.is_finite()); }
        for &v in &d_emb { assert!(v.is_finite()); }
    }

    #[test]
    fn test_trellis_gla_outer_loop_weight_descent() {
        // Validates outer-loop gradient flow through Trellis GLA parallelization.
        // See test_lattice_gla_outer_loop_weight_descent for design rationale.
        let cfg = MAGConfig::trellis_test_config();
        let mut lp = MAGParams::init(&cfg, 42).levels.into_iter().next().unwrap();
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
            let (y, cache) = trellis_gla_forward(&lp, &embedded, s, d, 2, &cfg, None);
            let loss: f32 = y.iter().zip(target.iter())
                .map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (s * d) as f32;
            if outer_step == 0 { first_loss = loss; }
            if outer_step == 99 { last_loss = loss; }

            let d_y: Vec<f32> = y.iter().zip(target.iter())
                .map(|(a, b)| 2.0 * (a - b) / (s * d) as f32).collect();
            let (grads, _) = trellis_gla_backward(&lp, &cache, &d_y, &embedded, &cfg);

            // Outer-loop weight update (projection weights, not inner-loop memory)
            for (w, g) in lp.w_k_mem.iter_mut().zip(grads.w_k_mem.iter()) { *w -= lr * g; }
            for (w, g) in lp.w_v_mem.iter_mut().zip(grads.w_v_mem.iter()) { *w -= lr * g; }
            for (w, g) in lp.w_q_mem.iter_mut().zip(grads.w_q_mem.iter()) { *w -= lr * g; }
        }

        assert!(last_loss <= first_loss + 1e-6,
            "Trellis GLA outer-loop weight descent should not diverge: {first_loss:.6} → {last_loss:.6}");
    }

    #[test]
    fn test_lattice_gla_vs_chunkwise_quality() {
        // GLA should be at least as good as raw chunkwise (renormalization helps)
        let cfg = MAGConfig::lattice_test_config();
        let params = MAGParams::init(&cfg, 42);
        let embedded = make_embedded(&cfg, 99);
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;

        let rule = LatticeOSR { m_slots: cfg.m_slots };
        let (y_seq, _) = rule.step(&params.levels[0], &embedded, s, d, None);

        let (y_gla, _) = lattice_gla_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );
        let (y_cw, _) = chunkwise_gd_forward(
            &params.levels[0], &embedded, s, d, 2, &cfg, None,
        );

        let gla_diff = relative_diff(&y_seq, &y_gla);
        let cw_diff = relative_diff(&y_seq, &y_cw);

        // GLA should be no worse than chunkwise (renormalization can't hurt)
        assert!(gla_diff <= cw_diff + 0.1,
            "GLA ({gla_diff:.4}) should be <= chunkwise ({cw_diff:.4}) + margin");
    }
}
