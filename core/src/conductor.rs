//! Conductor / Pulse timing system for CMS (Continuous Memory Systems).
//!
//! The Conductor generates a Pulse struct each step that tells every component
//! which frequency levels are active. Level 0 fires every step, Level 1 every
//! chunk_sizes[1] steps, etc.

use crate::model::MemoryLevelParams;

/// Timing pulse generated each step. Read-only after creation.
#[derive(Clone, Debug)]
pub struct Pulse {
    pub global_step: usize,
    pub active_levels: Vec<bool>,
}

/// Central timing system. Generates Pulse, manages step counter.
pub struct Conductor {
    pub k: usize,
    pub chunk_sizes: Vec<usize>,
    step: usize,
}

impl Conductor {
    pub fn new(k: usize, chunk_sizes: Vec<usize>) -> Self {
        assert_eq!(chunk_sizes.len(), k, "chunk_sizes length must equal k");
        assert!(k >= 1, "k must be at least 1");
        for (i, &cs) in chunk_sizes.iter().enumerate() {
            assert!(cs >= 1, "chunk_sizes[{i}] must be >= 1");
        }
        Conductor { k, chunk_sizes, step: 0 }
    }

    /// Generate pulse for current step.
    pub fn pulse(&self) -> Pulse {
        let active_levels = self.chunk_sizes.iter()
            .map(|&cs| self.step % cs == 0)
            .collect();
        Pulse {
            global_step: self.step,
            active_levels,
        }
    }

    /// Advance step counter. Call AFTER all observers have read the pulse (CS-32).
    pub fn advance(&mut self) {
        self.step += 1;
    }

    /// Current global step.
    pub fn step(&self) -> usize {
        self.step
    }
}

/// Memory state persisted across forward calls (ContextMemory lifetime).
pub struct ContextState {
    /// Per-level M matrices, each [d*d] (row-major).
    pub memory: Vec<Vec<f32>>,
    pub d: usize,
}

impl ContextState {
    pub fn new(k: usize, d: usize) -> Self {
        let memory = (0..k).map(|_| vec![0.0f32; d * d]).collect();
        ContextState { memory, d }
    }
}

/// Accumulated outer-loop gradients for a frozen level.
pub struct ErrorBuffer {
    pub grads: MemoryLevelParams,
    pub steps_accumulated: usize,
    d: usize,
}

impl ErrorBuffer {
    pub fn new(d: usize) -> Self {
        ErrorBuffer {
            grads: MemoryLevelParams::zeros_like(d),
            steps_accumulated: 0,
            d,
        }
    }

    /// Accumulate gradients from a frozen level's backward pass.
    pub fn accumulate(&mut self, level_grads: &MemoryLevelParams) {
        self.grads.accumulate(level_grads);
        self.steps_accumulated += 1;
    }

    /// Apply accumulated gradients via SGD and reset.
    pub fn apply_and_reset(&mut self, params: &mut MemoryLevelParams, lr: f32) {
        if self.steps_accumulated > 0 {
            params.sgd_step(&self.grads, lr);
            self.grads = MemoryLevelParams::zeros_like(self.d);
            self.steps_accumulated = 0;
        }
    }

    /// Health check: norm_ratio = ||accumulated|| / ||single_step||.
    /// Returns None if single_step norm is zero.
    pub fn health_check(&self, single_step_grads: &MemoryLevelParams) -> Option<f32> {
        let single_norm = single_step_grads.norm();
        if single_norm < 1e-12 {
            return None;
        }
        Some(self.grads.norm() / single_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_k1_always_active() {
        let c = Conductor::new(1, vec![1]);
        for _ in 0..10 {
            let p = c.pulse();
            assert_eq!(p.active_levels, vec![true]);
        }
    }

    #[test]
    fn test_pulse_k2_level0_always() {
        let mut c = Conductor::new(2, vec![1, 8]);
        for _ in 0..20 {
            let p = c.pulse();
            assert!(p.active_levels[0], "Level 0 must always be active");
            c.advance();
        }
    }

    #[test]
    fn test_pulse_k2_level1_frequency() {
        let mut c = Conductor::new(2, vec![1, 8]);
        let mut l1_active_steps = vec![];
        for step in 0..24 {
            let p = c.pulse();
            if p.active_levels[1] {
                l1_active_steps.push(step);
            }
            c.advance();
        }
        assert_eq!(l1_active_steps, vec![0, 8, 16]);
    }

    #[test]
    fn test_conductor_advance() {
        let mut c = Conductor::new(1, vec![1]);
        assert_eq!(c.step(), 0);
        c.advance();
        assert_eq!(c.step(), 1);
        c.advance();
        assert_eq!(c.step(), 2);
    }

    #[test]
    fn test_context_state_init() {
        let ctx = ContextState::new(2, 4);
        assert_eq!(ctx.memory.len(), 2);
        assert_eq!(ctx.memory[0].len(), 16);
        assert_eq!(ctx.memory[1].len(), 16);
        assert!(ctx.memory[0].iter().all(|&v| v == 0.0));
        assert!(ctx.memory[1].iter().all(|&v| v == 0.0));
        assert_eq!(ctx.d, 4);
    }

    #[test]
    fn test_error_buffer_accumulate() {
        let d = 4;
        let mut buf = ErrorBuffer::new(d);
        let mut g1 = MemoryLevelParams::zeros_like(d);
        g1.w_k_mem.iter_mut().for_each(|v| *v = 1.0);
        let mut g2 = MemoryLevelParams::zeros_like(d);
        g2.w_k_mem.iter_mut().for_each(|v| *v = 2.0);

        buf.accumulate(&g1);
        assert_eq!(buf.steps_accumulated, 1);
        buf.accumulate(&g2);
        assert_eq!(buf.steps_accumulated, 2);

        // Should be summed: 1.0 + 2.0 = 3.0
        assert!((buf.grads.w_k_mem[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_buffer_apply_reset() {
        let d = 4;
        let mut buf = ErrorBuffer::new(d);
        let mut g = MemoryLevelParams::zeros_like(d);
        g.w_k_mem.iter_mut().for_each(|v| *v = 1.0);
        buf.accumulate(&g);

        let mut params = MemoryLevelParams::zeros_like(d);
        params.w_k_mem.iter_mut().for_each(|v| *v = 10.0);

        buf.apply_and_reset(&mut params, 0.1);

        // params = 10.0 - 0.1 * 1.0 = 9.9
        assert!((params.w_k_mem[0] - 9.9).abs() < 1e-6);
        assert_eq!(buf.steps_accumulated, 0);
        assert!(buf.grads.w_k_mem.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_error_buffer_health() {
        let d = 4;
        let mut buf = ErrorBuffer::new(d);

        // Accumulate 7 identical gradient steps
        let mut g = MemoryLevelParams::zeros_like(d);
        g.w_k_mem.iter_mut().for_each(|v| *v = 1.0);
        for _ in 0..7 {
            buf.accumulate(&g);
        }

        // Health check: accumulated norm / single step norm
        let ratio = buf.health_check(&g).unwrap();
        // accumulated has 7.0 in each slot, single has 1.0 → ratio = 7.0
        assert!((ratio - 7.0).abs() < 0.5);
    }

    #[test]
    fn test_error_buffer_health_zero_single() {
        let d = 4;
        let buf = ErrorBuffer::new(d);
        let zero = MemoryLevelParams::zeros_like(d);
        assert!(buf.health_check(&zero).is_none());
    }
}
