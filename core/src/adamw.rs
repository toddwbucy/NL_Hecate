/// Frequency-aware AdamW optimizer for the outer loop.
///
/// Maintains per-CMS-level moment buffers with independent step counters.
/// The optimizer only updates a level when the Conductor's Pulse fires for it.
/// Between firings, moment buffers are frozen — no step increment, no update.
///
/// Bias correction uses the level's OWN step count (how many times that level
/// has fired), not the global step. This is critical for slow levels: Level 3
/// at 512x frequency has only fired twice after 1024 global steps, so its
/// bias correction must reflect 2 updates, not 1024.
///
/// Source: HOPE (2512.24695) §4.1-4.2, §6 Eq 71; Loshchilov & Hutter 2019.
/// Constraint: CS-27 (optimizer frequency matches architecture), CS-28 (frequency-aware).

use crate::conductor::Pulse;

/// AdamW hyperparameters (shared across all levels by default).
#[derive(Clone, Debug)]
pub struct AdamWConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
        }
    }
}

/// Moment buffers for a single parameter group (one flat array of weights).
#[derive(Clone)]
struct MomentBuf {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl MomentBuf {
    fn zeros(n: usize) -> Self {
        MomentBuf { m: vec![0.0; n], v: vec![0.0; n] }
    }
}

/// Per-level optimizer state: moment buffers + level-local step counter.
#[derive(Clone)]
struct LevelState {
    /// Moment buffers for each parameter buffer in MemoryLevelParams.
    /// Order: w_k_mem, w_v_mem, w_q_mem, w_alpha, b_alpha, w_theta, b_theta.
    bufs: Vec<MomentBuf>,
    /// Number of times this level has actually fired (for bias correction).
    level_step: u32,
}

/// SWA parameter optimizer state (always updates, not frequency-gated).
#[derive(Clone)]
struct SwaState {
    /// Moment buffers for: w_embed, w_q, w_k, w_v, w_o, w_unembed.
    bufs: Vec<MomentBuf>,
    step: u32,
}

/// Frequency-aware AdamW: per-level state, Pulse-gated updates.
///
/// SWA parameters always update. CMS level parameters only update when
/// the Pulse indicates that level is active.
pub struct FrequencyAwareAdamW {
    pub config: AdamWConfig,
    swa: SwaState,
    levels: Vec<LevelState>,
}

/// Core AdamW step on a single (params, grads, m, v) group.
///
/// Modifies params, m, v in place. Uses pre-computed bias correction inverses.
#[inline]
fn adamw_step_buf(
    params: &mut [f32],
    grads: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    weight_decay: f32,
) {
    debug_assert_eq!(params.len(), grads.len());
    for i in 0..params.len() {
        let g = grads[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] * bc1_inv;
        let v_hat = v[i] * bc2_inv;
        params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
    }
}

impl FrequencyAwareAdamW {
    /// Create optimizer state from MAGParams shapes.
    pub fn new(params: &crate::model::MAGParams, config: AdamWConfig) -> Self {
        let swa = SwaState {
            bufs: vec![
                MomentBuf::zeros(params.swa.w_embed.len()),
                MomentBuf::zeros(params.swa.w_q.len()),
                MomentBuf::zeros(params.swa.w_k.len()),
                MomentBuf::zeros(params.swa.w_v.len()),
                MomentBuf::zeros(params.swa.w_o.len()),
                MomentBuf::zeros(params.swa.w_unembed.len()),
            ],
            step: 0,
        };

        let levels = params.levels.iter().map(|lp| LevelState {
            bufs: vec![
                MomentBuf::zeros(lp.w_k_mem.len()),
                MomentBuf::zeros(lp.w_v_mem.len()),
                MomentBuf::zeros(lp.w_q_mem.len()),
                MomentBuf::zeros(lp.w_alpha.len()),
                MomentBuf::zeros(lp.b_alpha.len()),
                MomentBuf::zeros(lp.w_theta.len()),
                MomentBuf::zeros(lp.b_theta.len()),
            ],
            level_step: 0,
        }).collect();

        FrequencyAwareAdamW { config, swa, levels }
    }

    /// Frequency-gated AdamW step. SWA params always update; CMS levels
    /// only update when the Pulse fires for that level.
    ///
    /// `grads` contains the gradient for the current step (or accumulated
    /// gradient from the error buffer for frozen levels that just fired).
    pub fn step(
        &mut self,
        params: &mut crate::model::MAGParams,
        grads: &crate::model::MAGParams,
        pulse: &Pulse,
        lr: f32,
    ) {
        let c = &self.config;

        // ── SWA params (always active) ────────────────────────────────
        self.swa.step += 1;
        let t = self.swa.step as f32;
        let bc1_inv = 1.0 / (1.0 - c.beta1.powf(t));
        let bc2_inv = 1.0 / (1.0 - c.beta2.powf(t));

        let swa_pairs: Vec<(&mut [f32], &[f32])> = vec![
            (params.swa.w_embed.as_mut_slice(), grads.swa.w_embed.as_slice()),
            (params.swa.w_q.as_mut_slice(), grads.swa.w_q.as_slice()),
            (params.swa.w_k.as_mut_slice(), grads.swa.w_k.as_slice()),
            (params.swa.w_v.as_mut_slice(), grads.swa.w_v.as_slice()),
            (params.swa.w_o.as_mut_slice(), grads.swa.w_o.as_slice()),
            (params.swa.w_unembed.as_mut_slice(), grads.swa.w_unembed.as_slice()),
        ];
        for (idx, (p, g)) in swa_pairs.into_iter().enumerate() {
            let buf = &mut self.swa.bufs[idx];
            adamw_step_buf(p, g, &mut buf.m, &mut buf.v,
                           lr, c.beta1, c.beta2, c.eps, bc1_inv, bc2_inv, c.weight_decay);
        }

        // ── CMS levels (Pulse-gated) ─────────────────────────────────
        for level in 0..self.levels.len() {
            if level >= pulse.active_levels.len() || !pulse.active_levels[level] {
                continue; // Level frozen: no update, no step increment
            }

            let ls = &mut self.levels[level];
            ls.level_step += 1;
            let lt = ls.level_step as f32;
            let lbc1_inv = 1.0 / (1.0 - c.beta1.powf(lt));
            let lbc2_inv = 1.0 / (1.0 - c.beta2.powf(lt));

            let lp = &mut params.levels[level];
            let lg = &grads.levels[level];

            let level_pairs: Vec<(&mut [f32], &[f32])> = vec![
                (lp.w_k_mem.as_mut_slice(), lg.w_k_mem.as_slice()),
                (lp.w_v_mem.as_mut_slice(), lg.w_v_mem.as_slice()),
                (lp.w_q_mem.as_mut_slice(), lg.w_q_mem.as_slice()),
                (lp.w_alpha.as_mut_slice(), lg.w_alpha.as_slice()),
                (lp.b_alpha.as_mut_slice(), lg.b_alpha.as_slice()),
                (lp.w_theta.as_mut_slice(), lg.w_theta.as_slice()),
                (lp.b_theta.as_mut_slice(), lg.b_theta.as_slice()),
            ];
            for (idx, (p, g)) in level_pairs.into_iter().enumerate() {
                let buf = &mut ls.bufs[idx];
                adamw_step_buf(p, g, &mut buf.m, &mut buf.v,
                               lr, c.beta1, c.beta2, c.eps, lbc1_inv, lbc2_inv, c.weight_decay);
            }
        }

        // ── CMS aggregation weights (always active, like SWA) ─────────
        for (a, &da) in params.alpha_mem.iter_mut().zip(grads.alpha_mem.iter()) {
            *a -= lr * da;
        }
    }

    /// Get the level-local step count for a CMS level.
    pub fn level_step(&self, level: usize) -> u32 {
        self.levels[level].level_step
    }

    /// Get the SWA step count.
    pub fn swa_step(&self) -> u32 {
        self.swa.step
    }
}

/// Cosine annealing with linear warmup (HOPE §9.2, TNT experiments).
///
/// Returns the learning rate for the given global step.
pub fn cosine_lr(step: u32, warmup_steps: u32, total_steps: u32,
                 lr_peak: f32, lr_min: f32) -> f32 {
    if step < warmup_steps {
        return lr_peak * step as f32 / warmup_steps.max(1) as f32;
    }
    let progress = (step - warmup_steps) as f32 / total_steps.saturating_sub(warmup_steps).max(1) as f32;
    let progress = progress.min(1.0);
    lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MAGParams};
    use crate::conductor::Conductor;

    fn test_cfg_k2() -> MAGConfig {
        MAGConfig::titans_test_config_k2()
    }

    #[test]
    fn test_adamw_step_basic() {
        let cfg = test_cfg_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let grads = MAGParams::init(&cfg, 99); // nonzero "gradients"
        let pulse = Pulse {
            global_step: 0,
            active_levels: vec![true, true],
        };
        let mut opt = FrequencyAwareAdamW::new(&params, AdamWConfig::default());

        let w0_before = params.swa.w_embed[0];
        opt.step(&mut params, &grads, &pulse, 1e-3);
        let w0_after = params.swa.w_embed[0];

        assert!((w0_after - w0_before).abs() > 1e-10,
            "SWA params should change after step");
        assert_eq!(opt.swa_step(), 1);
        assert_eq!(opt.level_step(0), 1);
        assert_eq!(opt.level_step(1), 1);
    }

    #[test]
    fn test_adamw_frozen_level_no_update() {
        let cfg = test_cfg_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let grads = MAGParams::init(&cfg, 99);

        // Only Level 0 active, Level 1 frozen
        let pulse = Pulse {
            global_step: 0,
            active_levels: vec![true, false],
        };
        let mut opt = FrequencyAwareAdamW::new(&params, AdamWConfig::default());

        let l1_w_before = params.levels[1].w_k_mem.clone();
        opt.step(&mut params, &grads, &pulse, 1e-3);

        assert_eq!(opt.level_step(0), 1, "Active level should increment");
        assert_eq!(opt.level_step(1), 0, "Frozen level should NOT increment");
        assert_eq!(params.levels[1].w_k_mem, l1_w_before,
            "Frozen level params should be unchanged");
    }

    #[test]
    fn test_adamw_per_level_bias_correction() {
        // Verify that Level 1 (fires every 8 steps) uses its own step count
        // for bias correction, not the global step.
        let cfg = test_cfg_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let grads = MAGParams::init(&cfg, 99);
        let mut conductor = Conductor::new(2, vec![1, 8]);
        let mut opt = FrequencyAwareAdamW::new(&params, AdamWConfig::default());

        // Run 16 global steps
        for _ in 0..16 {
            let pulse = conductor.pulse();
            opt.step(&mut params, &grads, &pulse, 1e-3);
            conductor.advance();
        }

        assert_eq!(opt.swa_step(), 16);
        assert_eq!(opt.level_step(0), 16, "Level 0 fires every step");
        assert_eq!(opt.level_step(1), 2, "Level 1 fires every 8 steps → 2 in 16");
    }

    #[test]
    fn test_adamw_convergence_with_pulse() {
        // Verify optimizer actually reduces a parameter toward zero under constant gradient
        let cfg = test_cfg_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let mut opt = FrequencyAwareAdamW::new(&params, AdamWConfig {
            weight_decay: 0.0, // disable decay for clean test
            ..AdamWConfig::default()
        });

        // Create gradient that pushes w_embed[0] positive
        let mut grads = MAGParams::zeros_like(&cfg);
        grads.swa.w_embed[0] = 1.0;

        let pulse = Pulse {
            global_step: 0,
            active_levels: vec![true, true],
        };

        let initial = params.swa.w_embed[0];
        for _ in 0..100 {
            opt.step(&mut params, &grads, &pulse, 1e-2);
        }
        let final_val = params.swa.w_embed[0];

        // Constant positive gradient should push param negative
        assert!(final_val < initial,
            "100 AdamW steps with positive gradient should decrease param: initial={initial}, final={final_val}");
    }

    #[test]
    fn test_adamw_swa_always_updates() {
        // Even with all levels frozen, SWA params must update
        let cfg = test_cfg_k2();
        let mut params = MAGParams::init(&cfg, 42);
        let grads = MAGParams::init(&cfg, 99);
        let pulse = Pulse {
            global_step: 0,
            active_levels: vec![false, false],
        };
        let mut opt = FrequencyAwareAdamW::new(&params, AdamWConfig::default());

        let before = params.swa.w_embed[0];
        opt.step(&mut params, &grads, &pulse, 1e-3);
        let after = params.swa.w_embed[0];

        assert!((after - before).abs() > 1e-10,
            "SWA should update even when all CMS levels are frozen");
        assert_eq!(opt.swa_step(), 1);
    }

    // ── cosine_lr tests ──────────────────────────────────────────────

    #[test]
    fn test_cosine_lr_warmup() {
        let lr = cosine_lr(50, 100, 1000, 4e-4, 0.0);
        let expected = 4e-4 * 50.0 / 100.0;
        assert!((lr - expected).abs() < 1e-8, "Warmup: lr={lr}, expected {expected}");
    }

    #[test]
    fn test_cosine_lr_peak() {
        let lr = cosine_lr(100, 100, 1000, 4e-4, 0.0);
        assert!((lr - 4e-4).abs() < 1e-8, "At warmup end: lr={lr}");
    }

    #[test]
    fn test_cosine_lr_end() {
        let lr = cosine_lr(1000, 100, 1000, 4e-4, 0.0);
        assert!(lr.abs() < 1e-7, "At end: lr={lr}, should be ~0");
    }

    #[test]
    fn test_cosine_lr_midpoint() {
        // At 50% through cosine phase, lr should be lr_peak/2
        let lr = cosine_lr(550, 100, 1000, 4e-4, 0.0);
        let expected = 0.5 * 4e-4 * (1.0 + (std::f32::consts::PI * 0.5).cos());
        assert!((lr - expected).abs() < 1e-7, "Mid: lr={lr}, expected {expected}");
    }
}
