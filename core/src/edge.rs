/// Edge deployment module for micro models (d <= 128).
///
/// Three deployment profiles:
///   Profile 1: Inner-loop only (no Enzyme on target, frozen outer-loop weights)
///   Profile 2: Full NL (Enzyme on target for fine-tuning)
///   Profile 3: WASM (browser deployment)
///
/// This module provides a zero-dependency wrapper around the existing
/// cms_forward/cms_backward pipeline. No std::time::Instant — timing
/// is provided externally (benchmarks, serving module).

use crate::model::{MAGConfig, MAGParams, CompositionKind, MemoryRuleKind};
use crate::conductor::{Conductor, ContextState, ErrorBuffer};
use crate::mag::{cms_forward, cms_backward};

/// Configuration for a micro model deployment.
#[derive(Clone, Debug)]
pub struct EdgeConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub k: usize,
    pub composition: CompositionKind,
    pub memory_rule: MemoryRuleKind,
}

impl EdgeConfig {
    /// Build the MAGConfig from edge parameters.
    pub fn to_mag_config(&self) -> MAGConfig {
        let head_dim = self.d_model / self.num_heads;
        assert_eq!(
            self.d_model,
            self.num_heads * head_dim,
            "d_model must be divisible by num_heads"
        );

        // CMS chunk sizes: [1, 8, 64, 512] truncated to k levels
        let all_chunks = [1, 8, 64, 512];
        let chunk_sizes: Vec<usize> = all_chunks[..self.k].to_vec();

        MAGConfig {
            swa: crate::model::SWAConfig {
                d_model: self.d_model,
                num_heads: self.num_heads,
                head_dim,
                seq_len: self.seq_len,
                window_size: self.seq_len,
                vocab_size: self.vocab_size,
            },
            memory_enabled: true,
            composition: self.composition,
            memory_rule: self.memory_rule,
            k: self.k,
            chunk_sizes,
            d_hidden: 0,
            lp_p: 2.0,
            lq_q: 2.0,
            lambda_local: 0.0,
            lambda_2: 0.0,
            delta: 1.0,
            m_slots: 0,
            d_compress: 0,
            lambda_k: 0.0,
            lambda_v: 0.0,
            parallel: None,
        }
    }

    /// Micro model: d=64, 4 heads, seq=16, vocab=256, k=1 DeltaRule MAG.
    pub fn micro_d64() -> Self {
        EdgeConfig {
            d_model: 64,
            num_heads: 4,
            seq_len: 16,
            vocab_size: 256,
            k: 1,
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
        }
    }

    /// Micro model: d=128, 8 heads, seq=16, vocab=256, k=1 DeltaRule MAG.
    pub fn micro_d128() -> Self {
        EdgeConfig {
            d_model: 128,
            num_heads: 8,
            seq_len: 16,
            vocab_size: 256,
            k: 1,
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
        }
    }
}

/// Edge model: loads frozen outer-loop params, runs inner-loop adaptation.
pub struct EdgeModel {
    pub params: MAGParams,
    pub cfg: MAGConfig,
    pub conductor: Conductor,
    pub context: ContextState,
}

impl EdgeModel {
    /// Create a new edge model from config + pre-initialized params.
    pub fn new(edge_cfg: &EdgeConfig, params: MAGParams) -> Self {
        let cfg = edge_cfg.to_mag_config();
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let context = ContextState::new(cfg.k, cfg.swa.d_model);
        EdgeModel { params, cfg, conductor, context }
    }

    /// Create with random initialization (for testing/benchmarking).
    pub fn new_random(edge_cfg: &EdgeConfig, seed: u64) -> Self {
        let cfg = edge_cfg.to_mag_config();
        let params = MAGParams::init(&cfg, seed);
        let conductor = Conductor::new(cfg.k, cfg.chunk_sizes.clone());
        let context = ContextState::new(cfg.k, cfg.swa.d_model);
        EdgeModel { params, cfg, conductor, context }
    }

    /// Process one chunk: forward pass with inner-loop adaptation.
    /// Returns (loss, logits).
    ///
    /// Profile 1: This IS the adaptation step — memory M updates happen
    /// inside cms_forward as part of the inner loop. No separate adapt call needed.
    pub fn process(&mut self, input_ids: &[usize], target_ids: &[usize]) -> (f32, Vec<f32>) {
        let pulse = self.conductor.pulse();
        let (loss, cache) = cms_forward(
            &self.params,
            &self.cfg,
            input_ids,
            target_ids,
            &pulse,
            &mut self.context,
        );
        self.conductor.advance();
        let logits = cache.logits.clone();
        (loss, logits)
    }

    /// Profile 2: Full forward + backward for on-device fine-tuning.
    /// Returns (loss, gradients).
    pub fn forward_backward(
        &mut self,
        input_ids: &[usize],
        target_ids: &[usize],
    ) -> (f32, MAGParams) {
        let pulse = self.conductor.pulse();
        let (loss, cache) = cms_forward(
            &self.params,
            &self.cfg,
            input_ids,
            target_ids,
            &pulse,
            &mut self.context,
        );
        let mut error_buffers: Vec<ErrorBuffer> = (0..self.cfg.k)
            .map(|_| ErrorBuffer::new(self.cfg.swa.d_model))
            .collect();
        let grads = cms_backward(
            &self.params,
            &self.cfg,
            &cache,
            input_ids,
            target_ids,
            &mut error_buffers,
        );
        self.conductor.advance();
        (loss, grads)
    }

    /// Apply outer-loop weight update from gradients.
    pub fn apply_gradients(&mut self, grads: &MAGParams, lr: f32) {
        self.params.apply_weight_gradients(grads, lr);
    }

    /// Total model size in bytes (all parameters, fp32).
    pub fn model_size_bytes(&self) -> usize {
        self.params.num_params() * 4  // f32 = 4 bytes
    }

    /// Memory state size in bytes (inner-loop M matrices, fp32).
    pub fn memory_state_bytes(&self) -> usize {
        self.context.memory.iter().map(|m| m.len() * 4).sum()
    }

    /// Total deployment footprint: params + memory state.
    pub fn deployment_footprint_bytes(&self) -> usize {
        self.model_size_bytes() + self.memory_state_bytes()
    }

    /// Get a snapshot of current memory state (for testing).
    pub fn memory_snapshot(&self) -> Vec<Vec<f32>> {
        self.context.memory.clone()
    }
}

/// Compute model size in bytes for a given edge config (without allocating).
pub fn estimate_model_size_bytes(cfg: &EdgeConfig) -> usize {
    let d = cfg.d_model;
    let v = cfg.vocab_size;
    let k = cfg.k;

    // SWA params: embed(v*d) + q(d*d) + k(d*d) + v(d*d) + o(d*d) + unembed(d*v)
    let swa_params = 2 * v * d + 4 * d * d;

    // Per-level: 3 projections(d*d) + 2 gates(2*d+1) + eta(2*d+1)
    let level_params = 3 * d * d + 3 * (2 * d + 1);

    // Memory state: k matrices of d*d
    let memory_state = k * d * d;

    (swa_params + k * level_params + memory_state) * 4  // fp32 = 4 bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_config_micro_d64() {
        let cfg = EdgeConfig::micro_d64();
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.num_heads, 4);
        let mag = cfg.to_mag_config();
        assert_eq!(mag.swa.head_dim, 16);
        assert_eq!(mag.chunk_sizes, vec![1]);
    }

    #[test]
    fn test_edge_config_micro_d128() {
        let cfg = EdgeConfig::micro_d128();
        assert_eq!(cfg.d_model, 128);
        let mag = cfg.to_mag_config();
        assert_eq!(mag.swa.head_dim, 16);
    }

    #[test]
    fn test_edge_model_smoke() {
        let edge_cfg = EdgeConfig::micro_d64();
        let mut model = EdgeModel::new_random(&edge_cfg, 42);
        let s = edge_cfg.seq_len;
        let input_ids: Vec<usize> = (0..s).map(|i| i % edge_cfg.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=s).map(|i| i % edge_cfg.vocab_size).collect();

        let (loss, logits) = model.process(&input_ids, &target_ids);
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        assert!(loss > 0.0, "loss should be positive");
        assert_eq!(logits.len(), s * edge_cfg.vocab_size);
    }

    #[test]
    fn test_edge_model_size() {
        let edge_cfg = EdgeConfig::micro_d64();
        let model = EdgeModel::new_random(&edge_cfg, 42);
        let size = model.model_size_bytes();
        // d=64, vocab=256: ~200KB for params
        assert!(size > 0);
        assert!(size < 1_000_000, "d=64 model should be < 1MB, got {} bytes", size);

        let footprint = model.deployment_footprint_bytes();
        assert!(footprint >= size, "footprint includes params + memory");
    }

    #[test]
    fn test_edge_estimate_matches_actual() {
        let edge_cfg = EdgeConfig::micro_d64();
        let model = EdgeModel::new_random(&edge_cfg, 42);
        let estimated = estimate_model_size_bytes(&edge_cfg);
        let actual = model.deployment_footprint_bytes();
        assert_eq!(estimated, actual, "estimate should match actual");
    }

    #[test]
    fn test_edge_adaptation_changes_memory() {
        let edge_cfg = EdgeConfig::micro_d64();
        let mut model = EdgeModel::new_random(&edge_cfg, 42);
        let s = edge_cfg.seq_len;
        let input_ids: Vec<usize> = (0..s).map(|i| i % edge_cfg.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=s).map(|i| i % edge_cfg.vocab_size).collect();

        let mem_before = model.memory_snapshot();
        let _ = model.process(&input_ids, &target_ids);
        let mem_after = model.memory_snapshot();

        // Memory should have changed (inner-loop adaptation)
        assert_ne!(mem_before, mem_after, "memory should change after processing");
    }

    #[test]
    fn test_edge_frozen_outer_loop() {
        let edge_cfg = EdgeConfig::micro_d64();
        let mut model = EdgeModel::new_random(&edge_cfg, 42);
        let params_before = model.params.clone();
        let s = edge_cfg.seq_len;
        let input_ids: Vec<usize> = (0..s).map(|i| i % edge_cfg.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=s).map(|i| i % edge_cfg.vocab_size).collect();

        // Profile 1: process without apply_gradients
        let _ = model.process(&input_ids, &target_ids);

        // Outer-loop weights should be unchanged
        assert_eq!(model.params.swa.w_q, params_before.swa.w_q);
        assert_eq!(model.params.swa.w_k, params_before.swa.w_k);
        assert_eq!(model.params.levels[0].w_k_mem, params_before.levels[0].w_k_mem);
    }

    #[test]
    fn test_edge_profile2_full_nl() {
        let edge_cfg = EdgeConfig::micro_d64();
        let mut model = EdgeModel::new_random(&edge_cfg, 42);
        let s = edge_cfg.seq_len;
        let input_ids: Vec<usize> = (0..s).map(|i| i % edge_cfg.vocab_size).collect();
        let target_ids: Vec<usize> = (1..=s).map(|i| i % edge_cfg.vocab_size).collect();

        let (loss, grads) = model.forward_backward(&input_ids, &target_ids);
        assert!(loss.is_finite());
        assert!(loss > 0.0);

        // Gradients should be non-zero
        let grad_norm: f32 = grads.swa.w_q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(grad_norm > 0.0, "gradients should be non-zero");

        // Apply and verify weights changed
        let w_q_before = model.params.swa.w_q.clone();
        model.apply_gradients(&grads, 0.01);
        assert_ne!(model.params.swa.w_q, w_q_before);
    }

    #[test]
    fn test_edge_memory_fits_l2_cache() {
        // d=64 model memory: k=1 * 64*64 * 4 bytes = 16KB
        // Typical L2 cache: 256KB - 1MB
        let edge_cfg = EdgeConfig::micro_d64();
        let model = EdgeModel::new_random(&edge_cfg, 42);
        let mem_bytes = model.memory_state_bytes();
        assert!(
            mem_bytes < 256 * 1024,
            "d=64 memory state should fit in L2 cache (256KB), got {} bytes",
            mem_bytes
        );
    }
}
