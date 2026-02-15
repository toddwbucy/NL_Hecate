//! ContextStream: continuous token streaming for NL training (CS-11: no epochs).
//!
//! Replaces DataLoader with a monotonic token stream chunked to CMS frequency
//! boundaries. The stream cursor is checkpointed atomically with Conductor state,
//! enabling exact resume from any training position.

use serde::{Serialize, Deserialize};

// ── Types ──────────────────────────────────────────────────────────────

/// Events emitted at chunk boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BoundaryEvent {
    /// Corpus exhausted, stream wrapped to beginning.
    DocumentEnd,
    /// CMS frequency alignment point.
    ChunkBoundary,
    /// Interactive session complete.
    SessionEnd,
    /// CMS level sync point.
    FrequencySync,
}

/// A chunk of tokens for one forward pass.
pub struct TokenChunk {
    /// Input token IDs, length = chunk_size.
    pub input_ids: Vec<usize>,
    /// Target token IDs (next-token shifted), length = chunk_size.
    pub target_ids: Vec<usize>,
    /// Monotonically increasing chunk counter.
    pub chunk_id: u64,
    /// Boundary event if this chunk triggered one.
    pub boundary: Option<BoundaryEvent>,
}

/// Serializable cursor capturing exact stream position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamCursor {
    /// Token offset in data source.
    pub position: u64,
    /// Monotonic chunk counter.
    pub chunk_id: u64,
    /// Synced with Conductor step for atomic checkpoint.
    pub pulse_id: u64,
    /// Optional RNG state for future shuffled backends.
    pub rng_state: Option<u64>,
    /// FNV-1a hash of last chunk's input_ids (integrity canary).
    pub content_hash: u64,
}

/// Errors from restore operations.
#[derive(Clone, Debug, PartialEq)]
pub enum RestoreError {
    PulseMismatch { stream_pulse: u64, model_pulse: u64 },
    PositionOutOfBounds { position: u64, data_len: u64 },
    ConfigMismatch {
        expected_k: usize,
        expected_chunk_sizes: Vec<usize>,
        found_k: usize,
        found_chunk_sizes: Vec<usize>,
    },
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RestoreError::PulseMismatch { stream_pulse, model_pulse } => {
                write!(f, "pulse mismatch: stream={stream_pulse}, model={model_pulse}")
            }
            RestoreError::PositionOutOfBounds { position, data_len } => {
                write!(f, "position {position} out of bounds (data_len={data_len})")
            }
            RestoreError::ConfigMismatch { expected_k, found_k, .. } => {
                write!(f, "config mismatch: expected k={expected_k}, found k={found_k}")
            }
        }
    }
}

// ── Trait ───────────────────────────────────────────────────────────────

/// Continuous token stream. Object-safe for `Box<dyn ContextStream>`.
pub trait ContextStream {
    /// Get next chunk of `chunk_size` tokens. Returns None only if corpus is empty.
    fn next_chunk(&mut self, chunk_size: usize) -> Option<TokenChunk>;

    /// Reset position to beginning. Does NOT reset chunk_id (monotonic).
    fn reset(&mut self);

    /// Current token offset.
    fn position(&self) -> u64;

    /// Capture full cursor state.
    fn cursor(&self) -> StreamCursor;

    /// Restore from a previously captured cursor.
    fn restore(&mut self, cursor: &StreamCursor) -> Result<(), RestoreError>;

    /// Called by Conductor to sync pulse_id before checkpoint.
    fn set_pulse_id(&mut self, pulse_id: u64);
}

// ── FNV-1a hash ────────────────────────────────────────────────────────

/// FNV-1a hash over token slice (little-endian byte representation).
fn fnv1a_hash(tokens: &[usize]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for &token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

// ── VecStream (in-memory backend) ──────────────────────────────────────

/// In-memory token stream backed by a Vec<usize>.
pub struct VecStream {
    corpus: Vec<usize>,
    pos: usize,
    chunk_id: u64,
    pulse_id: u64,
    last_content_hash: u64,
}

impl VecStream {
    /// Create a new VecStream from a token corpus.
    /// Requires at least 2 tokens (input + one target).
    pub fn new(corpus: Vec<usize>) -> Self {
        assert!(corpus.len() >= 2, "corpus must have at least 2 tokens");
        VecStream {
            corpus,
            pos: 0,
            chunk_id: 0,
            pulse_id: 0,
            last_content_hash: 0,
        }
    }
}

impl ContextStream for VecStream {
    fn next_chunk(&mut self, chunk_size: usize) -> Option<TokenChunk> {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");

        if self.corpus.len() < 2 {
            return None;
        }

        let mut boundary = None;

        // Check if we need to wrap around
        let remaining = self.corpus.len().saturating_sub(self.pos);
        if remaining < 2 {
            // Not enough for even 1 input + 1 target: wrap
            self.pos = 0;
            boundary = Some(BoundaryEvent::DocumentEnd);
        }

        // Determine actual chunk size (may be truncated at end)
        let available = self.corpus.len() - self.pos;
        // Need chunk_size inputs + 1 for the last target = chunk_size + 1 tokens
        let actual_size = if available >= chunk_size + 1 {
            chunk_size
        } else {
            // Truncated: use what's available minus 1 (need at least 1 target)
            let truncated = available - 1;
            if truncated == 0 {
                // Edge case: exactly 1 token left, wrap
                self.pos = 0;
                boundary = Some(BoundaryEvent::DocumentEnd);
                chunk_size.min(self.corpus.len() - 1)
            } else {
                if boundary.is_none() {
                    boundary = Some(BoundaryEvent::DocumentEnd);
                }
                truncated
            }
        };

        let input_ids: Vec<usize> = self.corpus[self.pos..self.pos + actual_size].to_vec();
        let target_ids: Vec<usize> = self.corpus[self.pos + 1..self.pos + actual_size + 1].to_vec();

        self.pos += actual_size;
        // If we hit end, wrap for next call
        if self.pos + 1 >= self.corpus.len() {
            self.pos = 0;
            if boundary.is_none() {
                boundary = Some(BoundaryEvent::DocumentEnd);
            }
        }

        self.chunk_id += 1;
        self.last_content_hash = fnv1a_hash(&input_ids);

        Some(TokenChunk {
            input_ids,
            target_ids,
            chunk_id: self.chunk_id,
            boundary,
        })
    }

    fn reset(&mut self) {
        self.pos = 0;
        // chunk_id is monotonic — not reset
    }

    fn position(&self) -> u64 {
        self.pos as u64
    }

    fn cursor(&self) -> StreamCursor {
        StreamCursor {
            position: self.pos as u64,
            chunk_id: self.chunk_id,
            pulse_id: self.pulse_id,
            rng_state: None,
            content_hash: self.last_content_hash,
        }
    }

    fn restore(&mut self, cursor: &StreamCursor) -> Result<(), RestoreError> {
        if cursor.position as usize >= self.corpus.len() && cursor.position != 0 {
            return Err(RestoreError::PositionOutOfBounds {
                position: cursor.position,
                data_len: self.corpus.len() as u64,
            });
        }
        self.pos = cursor.position as usize;
        self.chunk_id = cursor.chunk_id;
        self.pulse_id = cursor.pulse_id;
        self.last_content_hash = cursor.content_hash;
        Ok(())
    }

    fn set_pulse_id(&mut self, pulse_id: u64) {
        self.pulse_id = pulse_id;
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_deterministic() {
        let tokens = vec![1, 2, 3, 4, 5];
        let h1 = fnv1a_hash(&tokens);
        let h2 = fnv1a_hash(&tokens);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_for_different_input() {
        let h1 = fnv1a_hash(&[1, 2, 3]);
        let h2 = fnv1a_hash(&[3, 2, 1]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_vec_stream_new() {
        let s = VecStream::new(vec![0, 1, 2, 3, 4]);
        assert_eq!(s.position(), 0);
        assert_eq!(s.corpus.len(), 5);
    }

    #[test]
    #[should_panic(expected = "at least 2 tokens")]
    fn test_vec_stream_too_small() {
        VecStream::new(vec![0]);
    }

    #[test]
    fn test_set_pulse_id() {
        let mut s = VecStream::new(vec![0, 1, 2, 3]);
        s.set_pulse_id(42);
        assert_eq!(s.cursor().pulse_id, 42);
    }

    #[test]
    fn test_restore_out_of_bounds() {
        let mut s = VecStream::new(vec![0, 1, 2]);
        let cursor = StreamCursor {
            position: 100,
            chunk_id: 0,
            pulse_id: 0,
            rng_state: None,
            content_hash: 0,
        };
        assert_eq!(
            s.restore(&cursor),
            Err(RestoreError::PositionOutOfBounds { position: 100, data_len: 3 })
        );
    }
}
