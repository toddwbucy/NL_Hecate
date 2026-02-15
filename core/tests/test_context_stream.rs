//! ContextStream integration tests: streaming, checkpointing, CMS training.

use nl_hecate_core::context_stream::{ContextStream, VecStream, BoundaryEvent, RestoreError};
use nl_hecate_core::conductor::{Conductor, Checkpoint, ContextState, ErrorBuffer};
use nl_hecate_core::model::{MAGConfig, MAGParams};
use nl_hecate_core::mag::{cms_forward, cms_backward};

/// Helper: create a small corpus of tokens [0, 1, 2, ..., n-1].
fn make_corpus(n: usize) -> Vec<usize> {
    (0..n).collect()
}

// ── Test 1: Basic chunk production ─────────────────────────────────────

#[test]
fn test_vec_stream_basic() {
    let mut stream = VecStream::new(make_corpus(20));
    assert_eq!(stream.position(), 0);

    let chunk = stream.next_chunk(4).unwrap();
    assert_eq!(chunk.input_ids, vec![0, 1, 2, 3]);
    assert_eq!(chunk.target_ids, vec![1, 2, 3, 4]);
    assert_eq!(chunk.chunk_id, 1);
    assert_eq!(chunk.input_ids.len(), chunk.target_ids.len());
}

// ── Test 2: Sequential chunks without gaps ─────────────────────────────

#[test]
fn test_vec_stream_sequential_chunks() {
    let mut stream = VecStream::new(make_corpus(20));

    // First chunk: tokens 0..4
    let c1 = stream.next_chunk(4).unwrap();
    assert_eq!(c1.input_ids, vec![0, 1, 2, 3]);
    assert_eq!(c1.target_ids, vec![1, 2, 3, 4]);

    // Second chunk: tokens 4..8 (no gap, no overlap)
    let c2 = stream.next_chunk(4).unwrap();
    assert_eq!(c2.input_ids, vec![4, 5, 6, 7]);
    assert_eq!(c2.target_ids, vec![5, 6, 7, 8]);

    // Third chunk: tokens 8..12
    let c3 = stream.next_chunk(4).unwrap();
    assert_eq!(c3.input_ids, vec![8, 9, 10, 11]);
    assert_eq!(c3.target_ids, vec![9, 10, 11, 12]);

    // Monotonic chunk IDs
    assert_eq!(c1.chunk_id, 1);
    assert_eq!(c2.chunk_id, 2);
    assert_eq!(c3.chunk_id, 3);
}

// ── Test 3: DocumentEnd at corpus boundary ─────────────────────────────

#[test]
fn test_vec_stream_document_end() {
    // 10 tokens, chunk_size=4 → chunks of 4,4 then wrap
    let mut stream = VecStream::new(make_corpus(10));

    let c1 = stream.next_chunk(4).unwrap();
    assert_eq!(c1.input_ids, vec![0, 1, 2, 3]);
    assert!(c1.boundary.is_none());

    let c2 = stream.next_chunk(4).unwrap();
    assert_eq!(c2.input_ids, vec![4, 5, 6, 7]);
    // pos is now 8, only tokens [8,9] remain = 2 tokens. Next full chunk would need 5.
    // This chunk consumed 4 tokens fine, but pos+1=9 < 10, so no wrap yet.

    let c3 = stream.next_chunk(4).unwrap();
    // Only 2 tokens remain at pos=8: [8,9] → truncated to 1 input + 1 target
    // This triggers DocumentEnd
    assert!(c3.boundary == Some(BoundaryEvent::DocumentEnd));

    // After DocumentEnd, stream wraps — next chunk starts from beginning
    let c4 = stream.next_chunk(4).unwrap();
    assert_eq!(c4.input_ids[0], 0, "Should wrap to beginning after DocumentEnd");
}

// ── Test 4: Reset preserves monotonic chunk_id ─────────────────────────

#[test]
fn test_vec_stream_reset() {
    let mut stream = VecStream::new(make_corpus(20));

    // Read some chunks
    stream.next_chunk(4);
    stream.next_chunk(4);
    let pre_reset_cursor = stream.cursor();
    assert_eq!(pre_reset_cursor.chunk_id, 2);

    // Reset
    stream.reset();
    assert_eq!(stream.position(), 0);

    // Chunk ID continues monotonically (NOT reset to 0)
    let chunk = stream.next_chunk(4).unwrap();
    assert_eq!(chunk.chunk_id, 3, "chunk_id must be monotonic across reset");
    assert_eq!(chunk.input_ids, vec![0, 1, 2, 3], "data starts from beginning");
}

// ── Test 5: Cursor roundtrip ───────────────────────────────────────────

#[test]
fn test_vec_stream_cursor_roundtrip() {
    let mut stream = VecStream::new(make_corpus(100));

    // Advance to some position
    stream.next_chunk(10);
    stream.next_chunk(10);
    stream.next_chunk(10);

    // Capture cursor
    let cursor = stream.cursor();
    assert_eq!(cursor.chunk_id, 3);

    // Read a chunk from current position
    let expected_chunk = stream.next_chunk(10).unwrap();

    // Restore to saved cursor
    stream.restore(&cursor).unwrap();
    assert_eq!(stream.position(), cursor.position);

    // Read same chunk again — should match
    let replayed_chunk = stream.next_chunk(10).unwrap();
    assert_eq!(replayed_chunk.input_ids, expected_chunk.input_ids);
    assert_eq!(replayed_chunk.target_ids, expected_chunk.target_ids);
}

// ── Test 6: Conductor with stream ──────────────────────────────────────

#[test]
fn test_conductor_with_stream() {
    let corpus = make_corpus(100);
    let stream = VecStream::new(corpus);
    let mut conductor = Conductor::new(2, vec![1, 8])
        .with_stream(Box::new(stream));

    assert!(conductor.has_stream());

    // First step: both levels active (step 0 % 8 == 0)
    let (chunk, pulse) = conductor.next_chunk(8).unwrap();
    assert_eq!(pulse.global_step, 0);
    assert_eq!(pulse.active_levels, vec![true, true]);
    assert_eq!(chunk.input_ids.len(), 8);
    assert_eq!(chunk.input_ids[0], 0);

    conductor.advance();

    // Second step: only level 0 (step 1 % 8 != 0)
    let (chunk2, pulse2) = conductor.next_chunk(8).unwrap();
    assert_eq!(pulse2.global_step, 1);
    assert_eq!(pulse2.active_levels, vec![true, false]);
    assert_eq!(chunk2.input_ids[0], 8);

    conductor.advance();
}

// ── Test 7: Checkpoint serde roundtrip ─────────────────────────────────

#[test]
fn test_checkpoint_serde_roundtrip() {
    let corpus = make_corpus(100);
    let stream = VecStream::new(corpus);
    let mut conductor = Conductor::new(2, vec![1, 8])
        .with_stream(Box::new(stream));

    // Advance a few steps
    for _ in 0..5 {
        conductor.next_chunk(8);
        conductor.advance();
    }

    let checkpoint = conductor.checkpoint();

    // Serialize to JSON
    let json = serde_json::to_string(&checkpoint).unwrap();
    eprintln!("Checkpoint JSON: {json}");

    // Deserialize back
    let restored: Checkpoint = serde_json::from_str(&json).unwrap();

    assert_eq!(restored.conductor.step, checkpoint.conductor.step);
    assert_eq!(restored.conductor.k, checkpoint.conductor.k);
    assert_eq!(restored.conductor.chunk_sizes, checkpoint.conductor.chunk_sizes);
    assert_eq!(restored.stream.position, checkpoint.stream.position);
    assert_eq!(restored.stream.chunk_id, checkpoint.stream.chunk_id);
    assert_eq!(restored.stream.pulse_id, checkpoint.stream.pulse_id);
    assert_eq!(restored.stream.content_hash, checkpoint.stream.content_hash);
}

// ── Test 8: Checkpoint restore produces identical chunks ────────────────

#[test]
fn test_checkpoint_restore_resume() {
    let corpus = make_corpus(200);

    // Phase 1: Run 10 steps, checkpoint, run 5 more, record chunks
    let mut conductor = Conductor::new(1, vec![1])
        .with_stream(Box::new(VecStream::new(corpus.clone())));

    for _ in 0..10 {
        conductor.next_chunk(8);
        conductor.advance();
    }

    let checkpoint = conductor.checkpoint();

    // Read 5 more chunks
    let mut expected_chunks = vec![];
    for _ in 0..5 {
        let (chunk, _) = conductor.next_chunk(8).unwrap();
        expected_chunks.push(chunk.input_ids.clone());
        conductor.advance();
    }

    // Phase 2: New conductor, restore from checkpoint
    let mut conductor2 = Conductor::new(1, vec![1])
        .with_stream(Box::new(VecStream::new(corpus)));
    conductor2.restore(&checkpoint).unwrap();

    // Read 5 chunks — should match
    for (i, expected) in expected_chunks.iter().enumerate() {
        let (chunk, _) = conductor2.next_chunk(8).unwrap();
        assert_eq!(&chunk.input_ids, expected, "Chunk {i} mismatch after restore");
        conductor2.advance();
    }
}

// ── Test 9: Tampered pulse_id rejected ─────────────────────────────────

#[test]
fn test_pulse_mismatch_rejected() {
    let corpus = make_corpus(100);
    let mut conductor = Conductor::new(1, vec![1])
        .with_stream(Box::new(VecStream::new(corpus)));

    for _ in 0..5 {
        conductor.next_chunk(8);
        conductor.advance();
    }

    let mut checkpoint = conductor.checkpoint();

    // Tamper: set stream pulse_id to something different from conductor step
    checkpoint.stream.pulse_id = 999;

    let result = conductor.restore(&checkpoint);
    assert!(result.is_err());
    match result.unwrap_err() {
        RestoreError::PulseMismatch { stream_pulse, model_pulse } => {
            assert_eq!(stream_pulse, 999);
            assert_eq!(model_pulse, checkpoint.conductor.step as u64);
        }
        other => panic!("Expected PulseMismatch, got {:?}", other),
    }
}

// ── Test 10: Config mismatch rejected ───────────────────────────────────

#[test]
fn test_config_mismatch_rejected() {
    let corpus = make_corpus(100);
    // Create checkpoint from k=2 conductor
    let mut conductor_k2 = Conductor::new(2, vec![1, 8])
        .with_stream(Box::new(VecStream::new(corpus.clone())));
    for _ in 0..5 {
        conductor_k2.next_chunk(8);
        conductor_k2.advance();
    }
    let checkpoint = conductor_k2.checkpoint();

    // Try restoring into k=1 conductor → ConfigMismatch
    let mut conductor_k1 = Conductor::new(1, vec![1])
        .with_stream(Box::new(VecStream::new(corpus)));
    let result = conductor_k1.restore(&checkpoint);
    assert!(result.is_err());
    match result.unwrap_err() {
        RestoreError::ConfigMismatch { expected_k, found_k, .. } => {
            assert_eq!(expected_k, 1);
            assert_eq!(found_k, 2);
        }
        other => panic!("Expected ConfigMismatch, got {:?}", other),
    }
}

// ── Test 11: CMS k=2 training driven by ContextStream ──────────────────

#[test]
fn test_cms_training_via_stream() {
    let cfg = MAGConfig::test_config_k2();
    let mut params = MAGParams::init(&cfg, 42);

    // Build a corpus: repeating pattern so the model can learn
    let corpus_len = cfg.swa.seq_len * 50 + 1; // enough for many chunks
    let corpus: Vec<usize> = (0..corpus_len)
        .map(|i| i % cfg.swa.vocab_size)
        .collect();

    let mut conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone())
        .with_stream(Box::new(VecStream::new(corpus)));

    let mut context = ContextState::new(cfg.k, cfg.swa.d_model);
    let mut error_buffers: Vec<ErrorBuffer> = (0..cfg.k)
        .map(|_| ErrorBuffer::new(cfg.swa.d_model))
        .collect();

    let lr = 0.01;
    let steps = 1000;
    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for _ in 0..steps {
        let (chunk, pulse) = conductor.next_chunk(cfg.swa.seq_len).unwrap();

        let (loss, cache) = cms_forward(
            &params, &cfg, &chunk.input_ids, &chunk.target_ids, &pulse, &mut context,
        );
        if initial_loss.is_none() {
            initial_loss = Some(loss);
        }
        final_loss = loss;

        let grads = cms_backward(
            &params, &cfg, &cache, &chunk.input_ids, &chunk.target_ids, &mut error_buffers,
        );
        params.sgd_step(&grads, lr);

        for level in 0..cfg.k {
            if pulse.active_levels[level] && error_buffers[level].steps_accumulated > 0 {
                error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
            }
        }

        conductor.advance();
    }

    let initial = initial_loss.unwrap();
    eprintln!("Stream-driven CMS k=2: initial={initial:.4}, final={final_loss:.4}");

    assert!(initial.is_finite(), "Initial loss not finite");
    assert!(final_loss.is_finite(), "Final loss not finite");
    assert!(
        final_loss < initial,
        "Loss should decrease: initial={initial:.4}, final={final_loss:.4}"
    );
}
