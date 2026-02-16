//! Serving abstraction for NL's non-stationary models.
//!
//! In NL, the inner loop runs at test time — the model self-modifies as it
//! processes context. Each session owns isolated ContextState + Conductor.
//! There is NO train/eval mode distinction (CS-10): the same `cms_forward()`
//! used during build is used during serving.
//!
//! Two modes, distinguished by stream attachment (no mode flag):
//! - Test: bounded context (fixed-length document evaluation)
//! - Stream: unbounded context (conversation, continuous document stream)

use serde::{Serialize, Deserialize};
use crate::model::MAGConfig;
use crate::model::MAGParams;
use crate::conductor::{Conductor, ContextState, Checkpoint};
use crate::context_stream::ContextStream;
use crate::context_stream::RestoreError;
use crate::mag::cms_forward;

/// Unique session identifier.
pub type SessionId = u64;

/// Result of processing one chunk.
pub struct ChunkResult {
    pub loss: f32,
    pub logits: Vec<f32>,
    pub chunk_time_ms: f32,
    pub tokens_processed: usize,
}

/// Per-chunk latency tracker for SLA validation.
pub struct LatencyTracker {
    chunk_times_ms: Vec<f32>,
    worst_chunk_ms: f32,
}

impl LatencyTracker {
    pub fn new() -> Self {
        LatencyTracker {
            chunk_times_ms: Vec::new(),
            worst_chunk_ms: 0.0,
        }
    }

    pub fn record(&mut self, chunk_time_ms: f32) {
        if chunk_time_ms > self.worst_chunk_ms {
            self.worst_chunk_ms = chunk_time_ms;
        }
        self.chunk_times_ms.push(chunk_time_ms);
    }

    pub fn average_ms(&self) -> f32 {
        if self.chunk_times_ms.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.chunk_times_ms.iter().sum();
        sum / self.chunk_times_ms.len() as f32
    }

    pub fn worst_ms(&self) -> f32 {
        self.worst_chunk_ms
    }

    pub fn p99_ms(&self) -> f32 {
        if self.chunk_times_ms.is_empty() {
            return 0.0;
        }
        let mut sorted = self.chunk_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // p99: value at the 99th percentile (index such that 99% of values are ≤)
        let rank = (sorted.len() as f64 * 0.99).ceil() as usize;
        let idx = rank.min(sorted.len()) - 1;
        sorted[idx]
    }

    pub fn count(&self) -> usize {
        self.chunk_times_ms.len()
    }
}

/// A serving session. Owns per-session state, references shared params.
pub struct Session {
    id: SessionId,
    conductor: Conductor,
    context: ContextState,
    latency: LatencyTracker,
    chunks_processed: usize,
}

impl Session {
    /// Create a Test session (bounded context, no stream).
    pub fn new_test(id: SessionId, cfg: &MAGConfig) -> Self {
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let context = ContextState::new(cfg.k, cfg.swa.d_model);
        Session {
            id,
            conductor,
            context,
            latency: LatencyTracker::new(),
            chunks_processed: 0,
        }
    }

    /// Create a Stream session (unbounded, with ContextStream).
    pub fn new_stream(id: SessionId, cfg: &MAGConfig, stream: Box<dyn ContextStream>) -> Self {
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone())
            .with_stream(stream);
        let context = ContextState::new(cfg.k, cfg.swa.d_model);
        Session {
            id,
            conductor,
            context,
            latency: LatencyTracker::new(),
            chunks_processed: 0,
        }
    }

    /// Process a chunk of tokens. Calls cms_forward() — same path as build.
    /// Advances conductor after processing (CS-32: observe then advance).
    pub fn process_chunk(
        &mut self,
        params: &MAGParams,
        cfg: &MAGConfig,
        input_ids: &[usize],
        target_ids: &[usize],
    ) -> ChunkResult {
        let start = std::time::Instant::now();

        // Observe: generate pulse for current step
        let pulse = self.conductor.pulse();

        // Forward: same path as build (CS-18)
        let (loss, cache) = cms_forward(params, cfg, input_ids, target_ids, &pulse, &mut self.context);

        // Advance: after all observers have read (CS-32)
        self.conductor.advance();

        let elapsed = start.elapsed();
        let chunk_time_ms = elapsed.as_secs_f32() * 1000.0;
        self.latency.record(chunk_time_ms);
        self.chunks_processed += 1;

        ChunkResult {
            loss,
            logits: cache.logits,
            chunk_time_ms,
            tokens_processed: input_ids.len(),
        }
    }

    /// Process next chunk from attached stream (Stream mode only).
    /// Returns None when stream is exhausted.
    pub fn process_next(
        &mut self,
        params: &MAGParams,
        cfg: &MAGConfig,
    ) -> Option<ChunkResult> {
        let chunk_size = cfg.swa.seq_len;
        let chunk_and_pulse = self.conductor.next_chunk(chunk_size);
        let (chunk, _pulse) = chunk_and_pulse?;

        // Use process_chunk which handles pulse generation, forward, advance, timing
        // But we already consumed a pulse via next_chunk — we need to use the chunk data
        // with process_chunk's own pulse generation. Since next_chunk doesn't advance,
        // and process_chunk generates its own pulse and advances, this is correct.
        // The next_chunk pulse and process_chunk pulse are identical (same step).
        Some(self.process_chunk(params, cfg, &chunk.input_ids, &chunk.target_ids))
    }

    /// Atomic checkpoint: conductor state + stream cursor + session metadata.
    /// Requires stream to be attached (Stream mode).
    pub fn checkpoint(&mut self) -> SessionCheckpoint {
        let inner = self.conductor.checkpoint();
        SessionCheckpoint {
            session_id: self.id,
            inner,
            chunks_processed: self.chunks_processed,
        }
    }

    /// Restore from checkpoint. Verifies config and pulse_id consistency.
    pub fn restore(&mut self, checkpoint: &SessionCheckpoint) -> Result<(), RestoreError> {
        self.conductor.restore(&checkpoint.inner)?;
        self.chunks_processed = checkpoint.chunks_processed;
        Ok(())
    }

    pub fn id(&self) -> SessionId {
        self.id
    }

    pub fn chunks_processed(&self) -> usize {
        self.chunks_processed
    }

    pub fn latency(&self) -> &LatencyTracker {
        &self.latency
    }

    pub fn context(&self) -> &ContextState {
        &self.context
    }

    pub fn conductor_step(&self) -> usize {
        self.conductor.step()
    }

    pub fn has_stream(&self) -> bool {
        self.conductor.has_stream()
    }
}

/// Serializable session checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    pub session_id: SessionId,
    pub inner: Checkpoint,
    pub chunks_processed: usize,
}
