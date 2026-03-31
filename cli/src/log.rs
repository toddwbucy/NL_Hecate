use std::fs::{File, OpenOptions};
use std::io::Write;
use serde_json::json;

/// Append-only JSONL metrics logger.
pub struct MetricsLogger {
    file: File,
}

impl MetricsLogger {
    pub fn new(path: &str) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open log file {path}: {e}"))?;
        Ok(MetricsLogger { file })
    }

    /// Log a step event.
    /// `level_firings` reports the actual number of memory updates per level
    /// in this step (seq_len / chunk_size for active levels, 0 for inactive).
    /// `cms_diag` optionally adds per-level CMS diagnostics inline.
    pub fn log_step(
        &mut self,
        step: usize,
        loss: f32,
        grad_norm: f32,
        lr: f32,
        elapsed: f64,
        active_levels: &[bool],
        level_firings: &[usize],
        cms_diag: Option<&CmsDiagnostics>,
        total_tokens: usize,
    ) {
        let ppl = (loss as f64).exp();
        let segments = total_tokens / 512;
        let mut entry = json!({
            "event": "step",
            "step": step,
            "segments": segments,
            "total_tokens": total_tokens,
            "loss": loss,
            "ppl": ppl,
            "grad_norm": grad_norm,
            "lr": lr,
            "elapsed": elapsed,
            "active_levels": active_levels,
            "level_firings": level_firings,
        });
        if let Some(diag) = cms_diag {
            let obj = entry.as_object_mut().unwrap();
            obj.insert("level_gnorms".into(), json!(diag.level_gnorms));
            obj.insert("level_m_norms".into(), json!(diag.level_m_norms));
            obj.insert("level_m_deltas".into(), json!(diag.level_m_deltas));
            obj.insert("dormancy_status".into(), json!(diag.dormancy_status));
        }
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log build start event.
    pub fn log_build_start(
        &mut self,
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        k: usize,
        n_blocks: usize,
        steps: usize,
        lr: f32,
        total_params: usize,
    ) {
        let entry = json!({
            "event": "build_start",
            "config": {
                "d_model": d_model,
                "num_heads": num_heads,
                "seq_len": seq_len,
                "k": k,
                "n_blocks": n_blocks,
                "steps": steps,
                "lr": lr,
                "params": total_params,
            }
        });
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log checkpoint event.
    pub fn log_checkpoint(&mut self, step: usize, path: &str) {
        let entry = json!({
            "event": "checkpoint",
            "step": step,
            "path": path,
        });
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log flashcard event.
    pub fn log_flashcard(&mut self, step: usize, chunks: usize, rounds: usize) {
        let entry = json!({
            "event": "flashcard",
            "step": step,
            "chunks": chunks,
            "rounds": rounds,
        });
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log build end event.
    pub fn log_build_end(&mut self, steps: usize, elapsed: f64, tok_per_sec: f64,
                         loss_first: f32, loss_last: f32, total_tokens: usize) {
        let segments = total_tokens / 512;
        let entry = json!({
            "event": "build_end",
            "steps": steps,
            "segments": segments,
            "total_tokens": total_tokens,
            "elapsed": elapsed,
            "tok_per_sec": tok_per_sec,
            "loss_first": loss_first,
            "loss_last": loss_last,
        });
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log arbitrary JSON event.
    pub fn log_raw(&mut self, value: serde_json::Value) {
        let _ = writeln!(self.file, "{}", value);
    }

    /// Log checkpoint probe results (coherence samples, within-gen learning, cross-exposure).
    pub fn log_probe_results(&mut self, step: usize, results: serde_json::Value) {
        let entry = json!({
            "event": "checkpoint_probes",
            "step": step,
            "probes": results,
        });
        let _ = writeln!(self.file, "{}", entry);
    }

    /// Log a dormancy event (level transitioned to warning or dormant).
    pub fn log_dormancy(&mut self, step: usize, block: usize, level: usize,
                        status: &str, consecutive_count: usize) {
        let entry = json!({
            "event": "dormancy",
            "step": step,
            "block": block,
            "level": level,
            "status": status,
            "consecutive_below_floor": consecutive_count,
        });
        let _ = writeln!(self.file, "{}", entry);
    }
}

/// Per-level CMS diagnostics collected after each training step.
/// Aggregated across blocks (mean of per-block values).
pub struct CmsDiagnostics {
    /// Per-level gradient norms (mean across blocks).
    pub level_gnorms: Vec<f32>,
    /// Per-level M Frobenius norms (mean across blocks).
    pub level_m_norms: Vec<f32>,
    /// Per-level M-norm deltas from previous step (mean across blocks).
    pub level_m_deltas: Vec<f32>,
    /// Per-level dormancy status: "active", "warning", "dormant".
    /// Uses worst-case across blocks (if any block is dormant for a level, report dormant).
    pub dormancy_status: Vec<String>,
}

/// Append-only JSONL logger for step profiling sidecar.
/// Records per-component GPU timing breakdown at configured intervals.
pub struct ProfileLogger {
    file: File,
}

impl ProfileLogger {
    pub fn new(path: &str) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open profile file {path}: {e}"))?;
        Ok(ProfileLogger { file })
    }

    /// Log a step profile with category and per-block breakdown.
    pub fn log_profile(&mut self, step: usize, profile: serde_json::Value) {
        let entry = json!({
            "step": step,
            "profile": profile,
        });
        let _ = writeln!(self.file, "{}", entry);
    }
}

/// Append-only JSONL logger for CMS tape sidecar.
/// Records per-level diagnostics every step, persisted alongside checkpoints.
pub struct CmsTapeLogger {
    file: File,
}

impl CmsTapeLogger {
    pub fn new(path: &str) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open CMS tape file {path}: {e}"))?;
        Ok(CmsTapeLogger { file })
    }

    /// Log per-level CMS diagnostics for one step.
    pub fn log_step(&mut self, step: usize, diag: &CmsDiagnostics) {
        let entry = json!({
            "step": step,
            "level_gnorms": diag.level_gnorms,
            "level_m_norms": diag.level_m_norms,
            "level_m_deltas": diag.level_m_deltas,
            "dormancy_status": diag.dormancy_status,
        });
        let _ = writeln!(self.file, "{}", entry);
    }
}
