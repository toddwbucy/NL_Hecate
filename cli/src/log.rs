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
    pub fn log_step(
        &mut self,
        step: usize,
        loss: f32,
        grad_norm: f32,
        lr: f32,
        elapsed: f64,
        active_levels: &[bool],
    ) {
        let ppl = (loss as f64).exp();
        let entry = json!({
            "event": "step",
            "step": step,
            "loss": loss,
            "ppl": ppl,
            "grad_norm": grad_norm,
            "lr": lr,
            "elapsed": elapsed,
            "active_levels": active_levels,
        });
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
                         loss_first: f32, loss_last: f32) {
        let entry = json!({
            "event": "build_end",
            "steps": steps,
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
}
