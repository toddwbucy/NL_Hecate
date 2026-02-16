/// CMS-aware multi-GPU gradient synchronization.
///
/// Standard DDP allreduces ALL parameters every step. NL's CMS architecture
/// only syncs active frequency levels — frozen levels accumulate rank-locally
/// in ErrorBuffer. For k=4, this averages ~1.14 allreduces/step vs DDP's 1
/// over all params.
///
/// Feature-gated: `#[cfg(feature = "distributed")]`

use std::time::Instant;
use crate::model::{MAGConfig, MAGParams};
use crate::conductor::{Pulse, Conductor, ContextState, ErrorBuffer};
use crate::mag::{cms_forward, cms_backward};

// ── ProcessGroup trait ────────────────────────────────────────────────

/// Abstract allreduce backend. MockProcessGroup for tests, NcclProcessGroup future.
pub trait ProcessGroup {
    /// In-place sum allreduce: after call, buf contains element-wise sum across all ranks.
    fn allreduce_sum(&self, buf: &mut [f32]);
    /// Total number of ranks in the group.
    fn world_size(&self) -> usize;
    /// This rank's index (0-based).
    fn rank(&self) -> usize;
}

// ── MockProcessGroup ──────────────────────────────────────────────────

/// Simulates multi-rank allreduce for testing without real GPUs.
///
/// Each "rank" is a MockProcessGroup instance sharing the same CallLog.
/// allreduce_sum records the call and simulates sum by multiplying by world_size
/// (since in tests, all ranks have identical data — sum = data * N).
pub struct MockProcessGroup {
    rank_id: usize,
    world: usize,
    call_log: std::sync::Arc<std::sync::Mutex<Vec<AllreduceCall>>>,
}

/// Record of a single allreduce invocation (for test assertions).
#[derive(Clone, Debug)]
pub struct AllreduceCall {
    pub rank: usize,
    pub len: usize,
}

impl MockProcessGroup {
    /// Create a set of N MockProcessGroup instances sharing one call log.
    pub fn new_group(world_size: usize) -> Vec<Self> {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        (0..world_size)
            .map(|r| MockProcessGroup {
                rank_id: r,
                world: world_size,
                call_log: log.clone(),
            })
            .collect()
    }

    /// Read the call log (for test assertions).
    pub fn call_log(&self) -> Vec<AllreduceCall> {
        self.call_log.lock().unwrap().clone()
    }

    /// Clear the call log.
    pub fn clear_log(&self) {
        self.call_log.lock().unwrap().clear();
    }
}

impl ProcessGroup for MockProcessGroup {
    fn allreduce_sum(&self, buf: &mut [f32]) {
        // Record the call
        self.call_log.lock().unwrap().push(AllreduceCall {
            rank: self.rank_id,
            len: buf.len(),
        });
        // Simulate sum across identical ranks: each rank has same data,
        // so sum = data * world_size.
        for v in buf.iter_mut() {
            *v *= self.world as f32;
        }
    }

    fn world_size(&self) -> usize {
        self.world
    }

    fn rank(&self) -> usize {
        self.rank_id
    }
}

// ── sync_gradients ────────────────────────────────────────────────────

/// CMS-aware gradient synchronization.
///
/// 1. Always allreduce SWA gradients (shared attention weights).
/// 2. Per CMS level: only allreduce if `pulse.active_levels[level]` is true.
/// 3. Divide all synced gradients by world_size (mean reduction).
///
/// Returns the number of allreduce calls made (for throughput reporting).
pub fn sync_gradients(
    grads: &mut MAGParams,
    pulse: &Pulse,
    pg: &dyn ProcessGroup,
) -> usize {
    assert_eq!(
        pulse.active_levels.len(),
        grads.levels.len(),
        "pulse.active_levels length ({}) must match grads.levels length ({})",
        pulse.active_levels.len(),
        grads.levels.len(),
    );
    let ws_usize = pg.world_size();
    assert!(ws_usize > 0, "world_size must be >= 1");
    let ws = ws_usize as f32;
    let mut allreduce_count = 0usize;

    // Always sync SWA gradients
    fn allreduce_and_mean(pg: &dyn ProcessGroup, buf: &mut [f32], ws: f32) {
        pg.allreduce_sum(buf);
        for v in buf.iter_mut() {
            *v /= ws;
        }
    }

    allreduce_and_mean(pg, &mut grads.swa.w_embed, ws);
    allreduce_and_mean(pg, &mut grads.swa.w_q, ws);
    allreduce_and_mean(pg, &mut grads.swa.w_k, ws);
    allreduce_and_mean(pg, &mut grads.swa.w_v, ws);
    allreduce_and_mean(pg, &mut grads.swa.w_o, ws);
    allreduce_and_mean(pg, &mut grads.swa.w_unembed, ws);
    allreduce_count += 6;

    // Per-level: only sync active levels
    for (level, level_grads) in grads.levels.iter_mut().enumerate() {
        if level < pulse.active_levels.len() && pulse.active_levels[level] {
            allreduce_and_mean(pg, &mut level_grads.w_k_mem, ws);
            allreduce_and_mean(pg, &mut level_grads.w_v_mem, ws);
            allreduce_and_mean(pg, &mut level_grads.w_q_mem, ws);
            allreduce_and_mean(pg, &mut level_grads.w_alpha, ws);
            allreduce_and_mean(pg, &mut level_grads.b_alpha, ws);
            allreduce_and_mean(pg, &mut level_grads.w_theta, ws);
            allreduce_and_mean(pg, &mut level_grads.b_theta, ws);
            allreduce_and_mean(pg, &mut level_grads.w_eta, ws);
            allreduce_and_mean(pg, &mut level_grads.b_eta, ws);
            allreduce_count += 9;
        }
        // Frozen levels: gradients stay rank-local (already in ErrorBuffer)
    }

    allreduce_count
}

// ── Throughput Reporting (CS-43, CS-44) ───────────────────────────────

/// Per-step throughput metrics. Per-GPU only (CS-43), worst-case tracked (CS-44).
#[derive(Clone, Debug)]
pub struct ThroughputReport {
    /// Tokens processed per second on this GPU.
    pub tokens_per_sec_per_gpu: f32,
    /// Worst-case tokens/sec across all tracked steps.
    pub worst_gpu_tokens_per_sec: f32,
    /// Number of allreduce calls this step.
    pub allreduce_count: usize,
    /// Wall-clock time for this step in milliseconds.
    pub step_time_ms: f32,
}

/// Accumulates step timings and computes running throughput stats.
pub struct ThroughputTracker {
    tokens_per_step: usize,
    step_times_ms: Vec<f32>,
    worst_tokens_per_sec: f32,
}

impl ThroughputTracker {
    pub fn new(tokens_per_step: usize) -> Self {
        ThroughputTracker {
            tokens_per_step,
            step_times_ms: Vec::new(),
            worst_tokens_per_sec: f32::MAX,
        }
    }

    /// Record a step and produce a report.
    pub fn record(&mut self, step_time_ms: f32, allreduce_count: usize) -> ThroughputReport {
        let tokens_per_sec = if step_time_ms > 0.0 {
            self.tokens_per_step as f32 / (step_time_ms / 1000.0)
        } else {
            0.0
        };

        self.step_times_ms.push(step_time_ms);
        if tokens_per_sec < self.worst_tokens_per_sec && tokens_per_sec > 0.0 {
            self.worst_tokens_per_sec = tokens_per_sec;
        }

        ThroughputReport {
            tokens_per_sec_per_gpu: tokens_per_sec,
            worst_gpu_tokens_per_sec: if self.worst_tokens_per_sec == f32::MAX {
                tokens_per_sec
            } else {
                self.worst_tokens_per_sec
            },
            allreduce_count,
            step_time_ms,
        }
    }

    /// Average tokens/sec across all recorded steps.
    pub fn average_tokens_per_sec(&self) -> f32 {
        if self.step_times_ms.is_empty() {
            return 0.0;
        }
        let total_ms: f32 = self.step_times_ms.iter().sum();
        let total_tokens = self.tokens_per_step * self.step_times_ms.len();
        total_tokens as f32 / (total_ms / 1000.0)
    }
}

// ── distributed_step ──────────────────────────────────────────────────

/// One distributed training step: forward → backward → sync → apply → advance.
///
/// Composes existing primitives (cms_forward, cms_backward, sync_gradients,
/// apply_weight_gradients, ErrorBuffer) into a single step with throughput reporting.
///
/// **Not a public training API (CS-18).** The forward pass remains the only
/// external API surface. This function is an internal orchestration helper
/// behind the `distributed` feature gate for testing and internal use only.
pub fn distributed_step(
    params: &mut MAGParams,
    cfg: &MAGConfig,
    input_ids: &[usize],
    target_ids: &[usize],
    conductor: &mut Conductor,
    context: &mut ContextState,
    error_buffers: &mut [ErrorBuffer],
    pg: &dyn ProcessGroup,
    lr: f32,
) -> (f32, ThroughputReport) {
    let start = Instant::now();

    // 1. Generate pulse (all ranks produce identical pulse from same step counter)
    let pulse = conductor.pulse();

    // 2. Forward pass
    let (loss, cache) = cms_forward(params, cfg, input_ids, target_ids, &pulse, context);

    // 3. Backward pass
    let mut grads = cms_backward(params, cfg, &cache, input_ids, target_ids, error_buffers);

    // 4. CMS-aware gradient sync
    let allreduce_count = sync_gradients(&mut grads, &pulse, pg);

    // 5. Apply weight gradients (outer-loop SGD)
    params.apply_weight_gradients(&grads, lr);

    // 6. Apply error buffers for levels that just became active
    for level in 0..cfg.k {
        if pulse.active_levels[level] {
            error_buffers[level].apply_and_reset(&mut params.levels[level], lr);
        }
    }

    // 7. Advance conductor (CS-32: observe then advance)
    conductor.advance();

    // 8. Throughput report
    let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;
    let tokens_per_sec = if elapsed_ms > 0.0 {
        cfg.swa.seq_len as f32 / (elapsed_ms / 1000.0)
    } else {
        0.0
    };

    let report = ThroughputReport {
        tokens_per_sec_per_gpu: tokens_per_sec,
        worst_gpu_tokens_per_sec: tokens_per_sec,
        allreduce_count,
        step_time_ms: elapsed_ms,
    };

    (loss, report)
}
