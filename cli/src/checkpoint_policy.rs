//! Spec 78: Composable checkpoint trigger policy.
//!
//! Defines WHEN checkpoints fire. Spec 67 defines WHAT happens at each
//! checkpoint (action pipeline). Triggers compose via OR — any trigger
//! firing initiates a checkpoint event, then all triggers reset.

use std::collections::VecDeque;
use std::time::Instant;

use crate::config::{CheckpointNaming, TriggerConfig};

/// Runtime state for checkpoint trigger evaluation.
///
/// Tracks token counts, wall-clock time, and loss history since the last
/// checkpoint. Evaluated once per logical step — just integer comparisons
/// and one clock read, zero overhead when no trigger fires.
pub struct TriggerState {
    last_checkpoint_tokens: u64,
    last_checkpoint_time: Instant,
    last_checkpoint_loss: f32,
    loss_ring: VecDeque<f32>,
}

impl TriggerState {
    pub fn new(initial_tokens: u64) -> Self {
        Self {
            last_checkpoint_tokens: initial_tokens,
            last_checkpoint_time: Instant::now(),
            last_checkpoint_loss: f32::MAX,
            loss_ring: VecDeque::new(),
        }
    }

    /// Evaluate all triggers via OR. Returns true if any trigger fires.
    pub fn should_checkpoint(
        &self,
        triggers: &[TriggerConfig],
        total_tokens: u64,
    ) -> bool {
        triggers.iter().any(|t| match t {
            TriggerConfig::TokenCount { every } => {
                total_tokens - self.last_checkpoint_tokens >= *every
            }
            TriggerConfig::ElapsedMinutes { every } => {
                self.last_checkpoint_time.elapsed()
                    >= std::time::Duration::from_secs(*every as u64 * 60)
            }
            TriggerConfig::LossPlateau { window, min_delta } => {
                let w = *window;
                if self.loss_ring.len() < w {
                    return false;
                }
                let best_recent = self.best_in_window(w);
                let improved = self.last_checkpoint_loss - best_recent > *min_delta;
                !improved
            }
            TriggerConfig::StepCount { every } => {
                // Internal-only: legacy save_every compatibility.
                // Evaluated in feed.rs via the phase_step check — this arm
                // exists for completeness but the feed loop handles it directly.
                let _ = every;
                false
            }
        })
    }

    /// Record that a checkpoint was taken at the given token count and loss.
    /// Resets all trigger state (time, token baseline, loss baseline).
    pub fn record_checkpoint(&mut self, total_tokens: u64, loss: f32) {
        self.last_checkpoint_tokens = total_tokens;
        self.last_checkpoint_time = Instant::now();
        self.last_checkpoint_loss = loss;
    }

    /// Record a loss value for plateau detection. Capped at 1000 entries.
    pub fn record_loss(&mut self, loss: f32) {
        self.loss_ring.push_back(loss);
        if self.loss_ring.len() > 1000 {
            self.loss_ring.pop_front();
        }
    }

    /// Best (lowest) loss in the last `window` entries.
    fn best_in_window(&self, window: usize) -> f32 {
        self.loss_ring.iter()
            .rev()
            .take(window)
            .copied()
            .fold(f32::MAX, f32::min)
    }

    /// Check if tokens have been processed since the last checkpoint.
    pub fn is_stale(&self, total_tokens: u64) -> bool {
        total_tokens > self.last_checkpoint_tokens
    }
}

/// Generate checkpoint filename based on naming policy.
pub fn checkpoint_filename(
    base_path: &str,
    naming: &CheckpointNaming,
    global_step: usize,
    total_tokens: usize,
) -> String {
    match naming {
        CheckpointNaming::Tokens => {
            base_path.replace(
                ".safetensors",
                &format!("_{}tok.safetensors", total_tokens),
            )
        }
        CheckpointNaming::Steps => {
            base_path.replace(
                ".safetensors",
                &format!("_step{}.safetensors", global_step),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_trigger_fires_at_boundary() {
        let state = TriggerState::new(0);
        let triggers = vec![TriggerConfig::TokenCount { every: 100_000 }];

        assert!(!state.should_checkpoint(&triggers, 50_000));
        assert!(!state.should_checkpoint(&triggers, 99_999));
        assert!(state.should_checkpoint(&triggers, 100_000));
        assert!(state.should_checkpoint(&triggers, 150_000));
    }

    #[test]
    fn test_token_trigger_respects_baseline() {
        let mut state = TriggerState::new(0);
        let triggers = vec![TriggerConfig::TokenCount { every: 100_000 }];

        // Fire and record
        assert!(state.should_checkpoint(&triggers, 100_000));
        state.record_checkpoint(100_000, 2.5);

        // Now relative to 100k baseline
        assert!(!state.should_checkpoint(&triggers, 150_000));
        assert!(state.should_checkpoint(&triggers, 200_000));
    }

    #[test]
    fn test_or_composition() {
        let state = TriggerState::new(0);
        let triggers = vec![
            TriggerConfig::TokenCount { every: 1_000_000 },
            TriggerConfig::TokenCount { every: 100_000 },
        ];

        // Smaller trigger fires first
        assert!(state.should_checkpoint(&triggers, 100_000));
    }

    #[test]
    fn test_loss_plateau_fires_when_flat() {
        let mut state = TriggerState::new(0);
        state.record_checkpoint(0, 3.0);

        let triggers = vec![TriggerConfig::LossPlateau {
            window: 5,
            min_delta: 0.01,
        }];

        // Not enough history yet
        for _ in 0..4 {
            state.record_loss(2.99);
        }
        assert!(!state.should_checkpoint(&triggers, 1000));

        // Now 5 entries — plateau (best=2.99, delta from 3.0 = 0.01, not > 0.01)
        state.record_loss(2.99);
        assert!(state.should_checkpoint(&triggers, 1000));
    }

    #[test]
    fn test_loss_plateau_does_not_fire_when_improving() {
        let mut state = TriggerState::new(0);
        state.record_checkpoint(0, 3.0);

        let triggers = vec![TriggerConfig::LossPlateau {
            window: 5,
            min_delta: 0.01,
        }];

        for _ in 0..5 {
            state.record_loss(2.0); // big improvement from 3.0
        }
        assert!(!state.should_checkpoint(&triggers, 1000));
    }

    #[test]
    fn test_record_checkpoint_resets_state() {
        let mut state = TriggerState::new(0);
        let triggers = vec![TriggerConfig::TokenCount { every: 100_000 }];

        assert!(state.should_checkpoint(&triggers, 100_000));
        state.record_checkpoint(100_000, 2.5);

        // Timer and token baseline both reset
        assert!(!state.should_checkpoint(&triggers, 100_001));
        assert_eq!(state.last_checkpoint_loss, 2.5);
    }

    #[test]
    fn test_is_stale() {
        let state = TriggerState::new(1000);
        assert!(!state.is_stale(1000));
        assert!(state.is_stale(1001));
    }

    #[test]
    fn test_loss_ring_capped() {
        let mut state = TriggerState::new(0);
        for i in 0..1500 {
            state.record_loss(i as f32);
        }
        assert_eq!(state.loss_ring.len(), 1000);
    }

    #[test]
    fn test_checkpoint_filename_tokens() {
        let path = checkpoint_filename(
            "checkpoints/model.safetensors",
            &CheckpointNaming::Tokens,
            100,
            5_120_000,
        );
        assert_eq!(path, "checkpoints/model_5120000tok.safetensors");
    }

    #[test]
    fn test_checkpoint_filename_steps() {
        let path = checkpoint_filename(
            "checkpoints/model.safetensors",
            &CheckpointNaming::Steps,
            100,
            5_120_000,
        );
        assert_eq!(path, "checkpoints/model_step100.safetensors");
    }

    #[test]
    fn test_empty_triggers_never_fires() {
        let state = TriggerState::new(0);
        assert!(!state.should_checkpoint(&[], 1_000_000));
    }

    #[test]
    fn test_step_count_trigger_is_noop() {
        // StepCount is handled by the feed loop directly, not by should_checkpoint
        let state = TriggerState::new(0);
        let triggers = vec![TriggerConfig::StepCount { every: 10 }];
        assert!(!state.should_checkpoint(&triggers, 1_000_000));
    }
}
