use serde::Deserialize;

/// Top-level JSON config (matches existing config file format).
#[derive(Deserialize, Debug)]
pub struct Config {
    #[serde(default)]
    pub description: String,
    pub model: ModelConfig,
    pub build: BuildConfig,
    pub data: DataConfig,
}

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_memory_rule")]
    pub memory_rule: String,
    #[serde(default = "default_composition")]
    pub composition: String,
    #[serde(default = "default_hope_variant")]
    pub hope_variant: String,
    #[serde(default = "default_k")]
    pub k: usize,
    pub chunk_sizes: Option<Vec<usize>>,
    pub m_norm_max: Option<Vec<f32>>,
    pub error_clip: Option<Vec<f32>>,
    #[serde(default = "default_true")]
    pub residual: bool,
    #[serde(default = "default_n_blocks")]
    pub n_blocks: usize,
    pub parallel_strategy: Option<String>,
    pub tnt_global_chunk_size: Option<usize>,
    pub tnt_local_chunk_size: Option<usize>,
    #[serde(default)]
    pub memory_reset: Option<String>,
    pub reset_intervals: Option<Vec<usize>>,
    #[serde(default = "default_tape_multiplier")]
    pub tape_multiplier: usize,
    pub tape_strategies: Option<Vec<String>>,
}

#[derive(Deserialize, Debug)]
pub struct BuildConfig {
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_beta1")]
    pub beta1: f32,
    #[serde(default = "default_beta2")]
    pub beta2: f32,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    pub alpha_floor: Option<Vec<f32>>,
    pub theta_ceil: Option<Vec<f32>>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
    #[serde(default = "default_save_every")]
    pub save_every: usize,
    #[serde(default)]
    pub tape_device: Option<String>,
    #[serde(default = "default_true")]
    pub gpu: bool,
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub load: Option<String>,
    pub seq_len_override: Option<usize>,
    pub run_dir: Option<String>,
    pub save_path: Option<String>,
    pub log_file: Option<String>,

    // Flashcard
    #[serde(default)]
    pub flashcard: bool,
    #[serde(default = "default_flashcard_pct")]
    pub flashcard_pct: f32,
    #[serde(default = "default_flashcard_rounds")]
    pub flashcard_rounds: usize,
    #[serde(default = "default_flashcard_gen_tokens")]
    pub flashcard_gen_tokens: usize,
}

#[derive(Deserialize, Debug)]
pub struct DataConfig {
    pub path: String,
    #[serde(default = "default_data_format")]
    pub format: String,
}

// Default functions
fn default_window_size() -> usize { 512 }
fn default_vocab_size() -> usize { 49152 }
fn default_memory_rule() -> String { "titans".into() }
fn default_composition() -> String { "mag".into() }
fn default_hope_variant() -> String { "chained".into() }
fn default_k() -> usize { 1 }
fn default_true() -> bool { true }
fn default_n_blocks() -> usize { 1 }
fn default_tape_multiplier() -> usize { 1 }
fn default_optimizer() -> String { "adamw_gpu_stacked".into() }
fn default_lr() -> f32 { 0.0003 }
fn default_steps() -> usize { 10000 }
fn default_weight_decay() -> f32 { 0.1 }
fn default_beta1() -> f32 { 0.9 }
fn default_beta2() -> f32 { 0.999 }
fn default_max_grad_norm() -> f32 { 1.0 }
fn default_batch_size() -> usize { 1 }
fn default_log_every() -> usize { 8 }
fn default_save_every() -> usize { 1000 }
fn default_seed() -> u64 { 42 }
fn default_flashcard_pct() -> f32 { 10.0 }
fn default_flashcard_rounds() -> usize { 3 }
fn default_flashcard_gen_tokens() -> usize { 64 }
fn default_data_format() -> String { "dolmino".into() }

impl Config {
    pub fn from_file(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config {path}: {e}"))?;
        serde_json::from_str(&text)
            .map_err(|e| format!("Failed to parse config {path}: {e}"))
    }

    /// Effective seq_len (override takes precedence).
    pub fn seq_len(&self) -> usize {
        self.build.seq_len_override.unwrap_or(self.model.seq_len)
    }
}
