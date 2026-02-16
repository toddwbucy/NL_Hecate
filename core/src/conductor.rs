//! Conductor / Pulse timing system for CMS (Continuous Memory Systems).
//!
//! The Conductor generates a Pulse struct each step that tells every component
//! which frequency levels are active. Level 0 fires every step, Level 1 every
//! chunk_sizes[1] steps, etc.

use serde::{Serialize, Deserialize};
use crate::model::MemoryLevelParams;
use crate::context_stream::{ContextStream, StreamCursor, TokenChunk, RestoreError};

/// Timing pulse generated each step. Read-only after creation.
#[derive(Clone, Debug, PartialEq)]
pub struct Pulse {
    pub global_step: usize,
    pub active_levels: Vec<bool>,
}

/// Serializable snapshot of Conductor state (for checkpoint).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConductorState {
    pub k: usize,
    pub chunk_sizes: Vec<usize>,
    pub step: usize,
}

/// Atomic checkpoint: Conductor state + stream cursor, pulse_id-verified.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub conductor: ConductorState,
    pub stream: StreamCursor,
}

/// Central timing system. Generates Pulse, manages step counter.
/// Optionally owns a ContextStream for integrated data feeding.
pub struct Conductor {
    pub k: usize,
    pub chunk_sizes: Vec<usize>,
    step: usize,
    stream: Option<Box<dyn ContextStream>>,
}

impl Conductor {
    pub fn new(k: usize, chunk_sizes: Vec<usize>) -> Self {
        assert_eq!(chunk_sizes.len(), k, "chunk_sizes length must equal k");
        assert!(k >= 1, "k must be at least 1");
        for (i, &cs) in chunk_sizes.iter().enumerate() {
            assert!(cs >= 1, "chunk_sizes[{i}] must be >= 1");
        }
        Conductor { k, chunk_sizes, step: 0, stream: None }
    }

    /// Builder: attach a ContextStream for integrated data feeding.
    pub fn with_stream(mut self, stream: Box<dyn ContextStream>) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Whether a stream is attached.
    pub fn has_stream(&self) -> bool {
        self.stream.is_some()
    }

    /// Get next chunk from attached stream + generate pulse. Does NOT advance (CS-32).
    /// Panics if no stream is attached.
    pub fn next_chunk(&mut self, chunk_size: usize) -> Option<(TokenChunk, Pulse)> {
        let pulse = self.pulse();
        let step = self.step as u64;
        let stream = self.stream.as_mut().expect("next_chunk requires an attached stream");
        stream.set_pulse_id(step);
        stream.next_chunk(chunk_size).map(|chunk| (chunk, pulse))
    }

    /// Capture atomic checkpoint (Conductor state + stream cursor).
    /// Syncs pulse_id to current step before capture.
    /// Panics if no stream is attached.
    pub fn checkpoint(&mut self) -> Checkpoint {
        let step = self.step as u64;
        let stream = self.stream.as_mut().expect("checkpoint requires an attached stream");
        stream.set_pulse_id(step);
        let cursor = stream.cursor();
        Checkpoint {
            conductor: ConductorState {
                k: self.k,
                chunk_sizes: self.chunk_sizes.clone(),
                step: self.step,
            },
            stream: cursor,
        }
    }

    /// Restore from a checkpoint. Verifies config and pulse_id consistency.
    pub fn restore(&mut self, checkpoint: &Checkpoint) -> Result<(), RestoreError> {
        // Verify CMS configuration matches
        if checkpoint.conductor.k != self.k
            || checkpoint.conductor.chunk_sizes != self.chunk_sizes
        {
            return Err(RestoreError::ConfigMismatch {
                expected_k: self.k,
                expected_chunk_sizes: self.chunk_sizes.clone(),
                found_k: checkpoint.conductor.k,
                found_chunk_sizes: checkpoint.conductor.chunk_sizes.clone(),
            });
        }
        // Verify pulse_id matches between conductor state and stream cursor
        if checkpoint.stream.pulse_id != checkpoint.conductor.step as u64 {
            return Err(RestoreError::PulseMismatch {
                stream_pulse: checkpoint.stream.pulse_id,
                model_pulse: checkpoint.conductor.step as u64,
            });
        }
        let stream = self.stream.as_mut().expect("restore requires an attached stream");
        stream.restore(&checkpoint.stream)?;
        self.step = checkpoint.conductor.step;
        Ok(())
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

    /// Create with custom memory size per level (for MLP-based rules like MONETA).
    pub fn new_with_memory_size(k: usize, d: usize, mem_size: usize) -> Self {
        let memory = (0..k).map(|_| vec![0.0f32; mem_size]).collect();
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

    /// Apply accumulated gradients as outer-loop weight update and reset.
    pub fn apply_and_reset(&mut self, params: &mut MemoryLevelParams, lr: f32) {
        if self.steps_accumulated > 0 {
            params.apply_weight_gradients(&self.grads, lr);
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

    // ── k=4 Conductor tests ─────────────────────────────────────────

    #[test]
    fn test_pulse_k4_all_levels() {
        let mut c = Conductor::new(4, vec![1, 8, 64, 512]);
        // Step 0: all levels fire
        let p = c.pulse();
        assert_eq!(p.active_levels, vec![true, true, true, true]);

        // Step 1: only level 0
        c.advance();
        let p = c.pulse();
        assert_eq!(p.active_levels, vec![true, false, false, false]);

        // Step 8: levels 0 and 1
        for _ in 1..8 { c.advance(); }
        let p = c.pulse();
        assert_eq!(p.active_levels, vec![true, true, false, false]);

        // Step 64: levels 0, 1, and 2
        for _ in 8..64 { c.advance(); }
        let p = c.pulse();
        assert_eq!(p.active_levels, vec![true, true, true, false]);

        // Step 512: all 4 levels
        for _ in 64..512 { c.advance(); }
        let p = c.pulse();
        assert_eq!(p.active_levels, vec![true, true, true, true]);
    }

    #[test]
    fn test_pulse_k4_level3_frequency() {
        let mut c = Conductor::new(4, vec![1, 8, 64, 512]);
        let mut l3_active_steps = vec![];
        for step in 0..1025 {
            let p = c.pulse();
            if p.active_levels[3] {
                l3_active_steps.push(step);
            }
            c.advance();
        }
        assert_eq!(l3_active_steps, vec![0, 512, 1024]);
    }

    #[test]
    fn test_context_state_k4() {
        let d = 8;
        let ctx = ContextState::new(4, d);
        assert_eq!(ctx.memory.len(), 4);
        for level in 0..4 {
            assert_eq!(ctx.memory[level].len(), d * d);
            assert!(ctx.memory[level].iter().all(|&v| v == 0.0));
        }
    }
}
