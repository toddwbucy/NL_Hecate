/// SWA Transformer model configuration and parameters.
///
/// Track Zero-A: single-block SWA with no memory, no CMS, no inner loop.
/// All weight matrices are flat Vec<f32> in row-major layout.

use crate::tensor::SimpleRng;

/// Model configuration — immutable after construction.
#[derive(Clone, Debug)]
pub struct SWAConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub window_size: usize,
    pub vocab_size: usize,
}

impl SWAConfig {
    /// Test configuration: tiny model for fast iteration.
    pub fn test_config() -> Self {
        SWAConfig {
            d_model: 64,
            num_heads: 4,
            head_dim: 16,  // d_model / num_heads
            seq_len: 24,
            window_size: 16,
            vocab_size: 256,
        }
    }
}

/// All learnable parameters — flat Vec<f32> for Enzyme compatibility.
///
/// Layout (row-major):
///   w_embed:  [vocab_size, d_model]
///   w_q:      [d_model, d_model]
///   w_k:      [d_model, d_model]
///   w_v:      [d_model, d_model]
///   w_o:      [d_model, d_model]
///   w_unembed:[d_model, vocab_size]
#[derive(Clone)]
pub struct SWAParams {
    pub w_embed: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_unembed: Vec<f32>,
}

impl SWAParams {
    /// Initialize with small random values using Xavier-like scaling.
    pub fn init(cfg: &SWAConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let d = cfg.d_model;
        let v = cfg.vocab_size;

        let embed_scale = (1.0 / d as f32).sqrt();
        let proj_scale = (2.0 / (d + d) as f32).sqrt(); // Xavier for d→d
        let unembed_scale = (1.0 / d as f32).sqrt();

        let mut w_embed = vec![0.0f32; v * d];
        rng.fill_uniform(&mut w_embed, embed_scale);

        let mut w_q = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_q, proj_scale);

        let mut w_k = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_k, proj_scale);

        let mut w_v = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_v, proj_scale);

        let mut w_o = vec![0.0f32; d * d];
        rng.fill_uniform(&mut w_o, proj_scale);

        let mut w_unembed = vec![0.0f32; d * v];
        rng.fill_uniform(&mut w_unembed, unembed_scale);

        SWAParams { w_embed, w_q, w_k, w_v, w_o, w_unembed }
    }

    /// Create a zero-initialized shadow for gradient accumulation.
    pub fn zeros_like(cfg: &SWAConfig) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        SWAParams {
            w_embed: vec![0.0f32; v * d],
            w_q: vec![0.0f32; d * d],
            w_k: vec![0.0f32; d * d],
            w_v: vec![0.0f32; d * d],
            w_o: vec![0.0f32; d * d],
            w_unembed: vec![0.0f32; d * v],
        }
    }

    /// Total number of parameters.
    pub fn num_params(&self) -> usize {
        self.w_embed.len() + self.w_q.len() + self.w_k.len()
            + self.w_v.len() + self.w_o.len() + self.w_unembed.len()
    }

    /// Apply SGD: param -= lr * grad for all weight matrices.
    /// Test-only: the forward pass is the sole public API (CS-18).
    #[cfg(test)]
    pub fn sgd_step(&mut self, grads: &SWAParams, lr: f32) {
        fn step(param: &mut [f32], grad: &[f32], lr: f32) {
            for i in 0..param.len() {
                param[i] -= lr * grad[i];
            }
        }
        step(&mut self.w_embed, &grads.w_embed, lr);
        step(&mut self.w_q, &grads.w_q, lr);
        step(&mut self.w_k, &grads.w_k, lr);
        step(&mut self.w_v, &grads.w_v, lr);
        step(&mut self.w_o, &grads.w_o, lr);
        step(&mut self.w_unembed, &grads.w_unembed, lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_consistency() {
        let cfg = SWAConfig::test_config();
        assert_eq!(cfg.d_model, cfg.num_heads * cfg.head_dim);
    }

    #[test]
    fn test_init_deterministic() {
        let cfg = SWAConfig::test_config();
        let p1 = SWAParams::init(&cfg, 42);
        let p2 = SWAParams::init(&cfg, 42);
        assert_eq!(p1.w_q, p2.w_q);
        assert_eq!(p1.w_k, p2.w_k);
        assert_eq!(p1.w_embed, p2.w_embed);
    }

    #[test]
    fn test_param_shapes() {
        let cfg = SWAConfig::test_config();
        let p = SWAParams::init(&cfg, 42);
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        assert_eq!(p.w_embed.len(), v * d);
        assert_eq!(p.w_q.len(), d * d);
        assert_eq!(p.w_k.len(), d * d);
        assert_eq!(p.w_v.len(), d * d);
        assert_eq!(p.w_o.len(), d * d);
        assert_eq!(p.w_unembed.len(), d * v);
    }

    #[test]
    fn test_zeros_like() {
        let cfg = SWAConfig::test_config();
        let z = SWAParams::zeros_like(&cfg);
        assert!(z.w_q.iter().all(|&x| x == 0.0));
        assert!(z.w_embed.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_init_range() {
        let cfg = SWAConfig::test_config();
        let p = SWAParams::init(&cfg, 42);
        // Xavier scale for d=64: sqrt(2/128) ≈ 0.125
        for &v in &p.w_q {
            assert!(v.abs() < 0.2, "Weight {} out of expected range", v);
        }
    }
}
