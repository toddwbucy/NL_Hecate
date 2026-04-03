use serde::Deserialize;

/// Top-level JSON config (spec 61: unified runtime with phase list).
#[derive(Deserialize, Debug)]
pub struct Config {
    #[serde(default)]
    pub description: String,
    pub model: ModelConfig,
    pub build: BuildConfig,
    /// Legacy single-phase data source — used when `phases` is absent.
    pub data: Option<DataConfig>,
    /// Ordered phase list. If absent, a single phase is synthesized from `data` + `build.steps`.
    pub phases: Option<Vec<PhaseConfig>>,
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
    /// Per-level M-norm delta floor for dormancy detection.
    /// If M-norm delta stays below this threshold for `dormancy_consecutive` steps,
    /// the level is flagged as dormant. Length k (one per CMS level).
    pub dormancy_floor: Option<Vec<f32>>,
    /// Number of consecutive steps below dormancy_floor before a level is flagged dormant.
    /// Default: 0 (disabled).
    #[serde(default)]
    pub dormancy_consecutive: usize,
}

/// Nested optimizer block (spec 61).
/// Each optimizer type declares its own parameters.
/// Fields are Option<T> so we can distinguish "explicitly set" from "absent" —
/// this matters for legacy compat (only promote flat build.lr when nested lr is absent).
#[derive(Deserialize, Debug, Clone)]
pub struct OptimizerConfig {
    #[serde(rename = "type")]
    pub optimizer_type: Option<String>,
    pub lr: Option<f32>,
    // AdamW-specific
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub weight_decay: Option<f32>,
    // Future: M3-specific
    pub meta_lr: Option<f32>,
    pub inner_steps: Option<usize>,
    // Future: SGD-specific
    pub momentum: Option<f32>,
}

impl OptimizerConfig {
    /// Resolve the optimizer type, falling back to a parent's type or "adamw".
    pub fn optimizer_type(&self) -> &str {
        self.optimizer_type.as_deref().unwrap_or("adamw")
    }

    /// Merge with a parent optimizer: fill in any None fields from the parent.
    pub fn merged_with(&self, parent: &OptimizerConfig) -> OptimizerConfig {
        OptimizerConfig {
            optimizer_type: Some(self.optimizer_type.clone()
                .unwrap_or_else(|| parent.optimizer_type().to_string())),
            lr: self.lr.or(parent.lr),
            beta1: self.beta1.or(parent.beta1),
            beta2: self.beta2.or(parent.beta2),
            weight_decay: self.weight_decay.or(parent.weight_decay),
            meta_lr: self.meta_lr.or(parent.meta_lr),
            inner_steps: self.inner_steps.or(parent.inner_steps),
            momentum: self.momentum.or(parent.momentum),
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: Some("adamw".into()),
            lr: None,
            beta1: None,
            beta2: None,
            weight_decay: None,
            meta_lr: None,
            inner_steps: None,
            momentum: None,
        }
    }
}

impl OptimizerConfig {
    /// Learning rate (defaults to 0.0003 if not set).
    pub fn lr(&self) -> f32 { self.lr.unwrap_or(0.0003) }
    /// AdamW beta1 (defaults to 0.9 if not set).
    pub fn beta1(&self) -> f32 { self.beta1.unwrap_or(0.9) }
    /// AdamW beta2 (defaults to 0.999 if not set).
    pub fn beta2(&self) -> f32 { self.beta2.unwrap_or(0.999) }
    /// Weight decay (defaults to 0.1 if not set).
    pub fn weight_decay(&self) -> f32 { self.weight_decay.unwrap_or(0.1) }
}
// Note: Rust allows multiple impl blocks for the same type.
// optimizer_type(), merged_with() are defined above; lr/beta1/beta2/weight_decay here.

#[derive(Deserialize, Debug)]
pub struct BuildConfig {
    /// Nested optimizer block. Also accepts legacy flat string via custom deserialize.
    #[serde(default, deserialize_with = "deserialize_optimizer")]
    pub optimizer: OptimizerConfig,
    // Legacy flat fields — used as fallback if optimizer block doesn't set them.
    // These exist for backward compatibility with old configs that have flat lr/beta1/etc.
    #[serde(default)]
    pub lr: Option<f32>,
    #[serde(default)]
    pub beta1: Option<f32>,
    #[serde(default)]
    pub beta2: Option<f32>,
    #[serde(default)]
    pub weight_decay: Option<f32>,

    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    pub alpha_floor: Option<Vec<f32>>,
    pub theta_ceil: Option<Vec<f32>>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Spec 76: gradient accumulation micro-steps. Effective batch = accum_steps × batch_size.
    #[serde(default = "default_accum_steps")]
    pub accum_steps: usize,
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
    /// Reset step counter to 0 when loading a checkpoint (for gear shifts).
    #[serde(default)]
    pub reset_step: bool,
    pub seq_len_override: Option<usize>,
    pub run_dir: Option<String>,
    pub save_path: Option<String>,
    pub log_file: Option<String>,

    // CMS diagnostic sidecar (written alongside each checkpoint)
    #[serde(default = "default_true")]
    pub cms_sidecar: bool,

    // k-extension: promote a checkpoint from k to k+1 CMS levels
    /// Target k (must be loaded_k + 1). Requires `load` to be set.
    pub extend_k: Option<usize>,
    /// Push-up: shift existing levels to slower frequencies, add fresh L0.
    #[serde(default)]
    pub push_up: bool,
    /// Stack-up: keep existing levels, add new level at top (slowest frequency).
    #[serde(default)]
    pub stack_up: bool,
    /// L0 initialization for push-up: "random" (Xavier) or "clone" (donor projections).
    #[serde(default = "default_push_up_init")]
    pub push_up_init: String,

    /// Profile every N steps (0 = disabled). Writes per-step GPU timing breakdown
    /// to a sidecar JSONL file for heatmap visualization.
    #[serde(default)]
    pub profile_every: usize,

    /// Path to tokenizer.json for checkpoint probes (generation + learning probes).
    /// If set, probes run automatically at each checkpoint.
    pub tokenizer_path: Option<String>,
    /// Max tokens to generate per probe prompt (default: 64).
    #[serde(default = "default_probe_max_tokens")]
    pub probe_max_tokens: usize,
    /// Probe sampling temperature (default: 0.8).
    #[serde(default = "default_probe_temperature")]
    pub probe_temperature: f32,

    // Legacy flashcard fields (deprecated — use phases with think_rounds)
    #[serde(default)]
    pub flashcard: bool,
    #[serde(default = "default_flashcard_pct")]
    pub flashcard_pct: f32,
    #[serde(default = "default_flashcard_rounds")]
    pub flashcard_rounds: usize,
    #[serde(default = "default_flashcard_gen_tokens")]
    pub flashcard_gen_tokens: usize,
}

/// A single phase in the curriculum (spec 61).
/// deny_unknown_fields catches typos and invalid model-structure overrides (d_model, num_heads, k).
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct PhaseConfig {
    /// Path to tokenized data directory.
    pub data: String,
    /// Process exactly N segments, then advance to next phase.
    pub steps: Option<usize>,
    /// Iterative self-refinement: learn→speak→feed back, N iterations.
    pub think_rounds: Option<usize>,
    /// Human-readable label (logged, not used by runtime).
    pub label: Option<String>,
    // Per-phase overrides (revert to build defaults after phase completes)
    pub optimizer: Option<OptimizerConfig>,
    pub batch_size: Option<usize>,
    pub accum_steps: Option<usize>,
    pub seq_len: Option<usize>,
    pub save_every: Option<usize>,
    pub log_every: Option<usize>,
    pub max_grad_norm: Option<f32>,
    pub warmup_steps: Option<usize>,
    // Generation parameters for think_rounds speak phase
    /// Max tokens to generate per speak step (default: seq_len).
    pub max_gen_tokens: Option<usize>,
    /// Sampling temperature for speak step (default: 0.0 = greedy).
    pub temperature: Option<f32>,
    /// Top-k filtering for speak step (default: 0 = disabled).
    pub top_k: Option<usize>,
}

#[derive(Deserialize, Debug)]
pub struct DataConfig {
    pub path: String,
    #[serde(default = "default_data_format")]
    pub format: String,
}

// ── Default functions ────────────────────────────────────────────────
fn default_window_size() -> usize { 512 }
fn default_vocab_size() -> usize { 49152 }
fn default_memory_rule() -> String { "titans".into() }
fn default_composition() -> String { "mag".into() }
fn default_hope_variant() -> String { "chained".into() }
fn default_k() -> usize { 1 }
fn default_true() -> bool { true }
fn default_n_blocks() -> usize { 1 }
fn default_tape_multiplier() -> usize { 1 }
fn default_lr() -> f32 { 0.0003 }
fn default_steps() -> usize { 10000 }
fn default_max_grad_norm() -> f32 { 1.0 }
fn default_batch_size() -> usize { 1 }
fn default_accum_steps() -> usize { 1 }
fn default_log_every() -> usize { 8 }
fn default_save_every() -> usize { 1000 }
fn default_seed() -> u64 { 42 }
fn default_push_up_init() -> String { "random".into() }
fn default_probe_max_tokens() -> usize { 64 }
fn default_probe_temperature() -> f32 { 0.8 }
fn default_flashcard_pct() -> f32 { 10.0 }
fn default_flashcard_rounds() -> usize { 3 }
fn default_flashcard_gen_tokens() -> usize { 64 }
fn default_data_format() -> String { "dolmino".into() }


/// Custom deserializer: accept either a string (legacy) or an object (spec 61).
fn deserialize_optimizer<'de, D>(deserializer: D) -> Result<OptimizerConfig, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OptimizerValue {
        /// Spec 61: nested object with type + params
        Object(OptimizerConfig),
        /// Legacy: bare string like "adamw_gpu_stacked"
        String(String),
    }

    match OptimizerValue::deserialize(deserializer) {
        Ok(OptimizerValue::Object(cfg)) => Ok(cfg),
        Ok(OptimizerValue::String(s)) => {
            // Map legacy string to optimizer type
            let optimizer_type = Some(if s.contains("adamw") {
                "adamw".into()
            } else {
                s
            });
            Ok(OptimizerConfig {
                optimizer_type,
                ..OptimizerConfig::default()
            })
        }
        Err(e) => Err(e),
    }
}

impl Config {
    /// Parse config from a JSON string (used in tests).
    pub fn from_str(text: &str) -> Result<Self, String> {
        let mut cfg: Config = serde_json::from_str(text)
            .map_err(|e| format!("Failed to parse config: {e}"))?;
        Self::apply_legacy_compat(&mut cfg);
        Self::validate(&cfg)?;
        Ok(cfg)
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config {path}: {e}"))?;
        let mut cfg: Config = serde_json::from_str(&text)
            .map_err(|e| format!("Failed to parse config {path}: {e}"))?;

        Self::apply_legacy_compat(&mut cfg);
        Self::validate(&cfg)?;
        Ok(cfg)
    }

    fn validate(cfg: &Config) -> Result<(), String> {
        if cfg.build.accum_steps < 1 {
            return Err("accum_steps must be >= 1".into());
        }
        if let Some(ref phases) = cfg.phases {
            for (i, phase) in phases.iter().enumerate() {
                if phase.accum_steps == Some(0) {
                    let label = phase.label.as_deref().unwrap_or(&phase.data);
                    return Err(format!("phase {i} ({label}): accum_steps must be >= 1"));
                }
            }
        }
        Ok(())
    }

    /// Apply backward-compat: promote flat build.lr/beta1/etc into the optimizer block,
    /// but only when the nested optimizer didn't explicitly set those fields.
    fn apply_legacy_compat(cfg: &mut Config) {
        if let Some(lr) = cfg.build.lr {
            if cfg.build.optimizer.lr.is_none() {
                cfg.build.optimizer.lr = Some(lr);
            }
        }
        if let Some(b1) = cfg.build.beta1 {
            if cfg.build.optimizer.beta1.is_none() {
                cfg.build.optimizer.beta1 = Some(b1);
            }
        }
        if let Some(b2) = cfg.build.beta2 {
            if cfg.build.optimizer.beta2.is_none() {
                cfg.build.optimizer.beta2 = Some(b2);
            }
        }
        if let Some(wd) = cfg.build.weight_decay {
            if cfg.build.optimizer.weight_decay.is_none() {
                cfg.build.optimizer.weight_decay = Some(wd);
            }
        }
    }

    /// Effective seq_len (override takes precedence).
    pub fn seq_len(&self) -> usize {
        self.build.seq_len_override.unwrap_or(self.model.seq_len)
    }

    /// Resolve phases: if `phases` is present, use it. Otherwise, synthesize
    /// a single phase from `data` + `build.steps`.
    pub fn resolved_phases(&self) -> Result<Vec<ResolvedPhase>, String> {
        if let Some(ref phases) = self.phases {
            let mut resolved = Vec::with_capacity(phases.len());
            for (i, phase) in phases.iter().enumerate() {
                // Validate: exactly one of steps or think_rounds
                match (phase.steps, phase.think_rounds) {
                    (Some(_), Some(_)) => {
                        return Err(format!(
                            "Phase {i} has both `steps` and `think_rounds` — pick one"
                        ));
                    }
                    (None, None) => {
                        return Err(format!(
                            "Phase {i} has neither `steps` nor `think_rounds`"
                        ));
                    }
                    _ => {}
                }
                // Merge phase optimizer with build default (inherits type + unset fields)
                let merged_optimizer = phase.optimizer.as_ref()
                    .map(|po| po.merged_with(&self.build.optimizer));

                resolved.push(ResolvedPhase {
                    data: phase.data.clone(),
                    duration: if let Some(s) = phase.steps {
                        PhaseDuration::Steps(s)
                    } else {
                        PhaseDuration::ThinkRounds(phase.think_rounds.unwrap())
                    },
                    label: phase.label.clone().unwrap_or_else(|| phase.data.clone()),
                    optimizer: merged_optimizer,
                    batch_size: phase.batch_size,
                    accum_steps: phase.accum_steps,
                    seq_len: phase.seq_len,
                    save_every: phase.save_every,
                    log_every: phase.log_every,
                    max_grad_norm: phase.max_grad_norm,
                    warmup_steps: phase.warmup_steps,
                    max_gen_tokens: phase.max_gen_tokens,
                    temperature: phase.temperature,
                    top_k: phase.top_k,
                });
            }
            Ok(resolved)
        } else {
            // Single-phase fallback from legacy config
            let data_path = self.data.as_ref()
                .ok_or("Config has no `phases` and no `data` — need at least one")?
                .path.clone();
            Ok(vec![ResolvedPhase {
                data: data_path,
                duration: PhaseDuration::Steps(self.build.steps),
                label: "default".into(),
                optimizer: None,
                batch_size: None,
                accum_steps: None,
                seq_len: None,
                save_every: None,
                log_every: None,
                max_grad_norm: None,
                warmup_steps: None,
                max_gen_tokens: None,
                temperature: None,
                top_k: None,
            }])
        }
    }
}

/// Resolved phase with validated duration.
#[derive(Debug)]
pub struct ResolvedPhase {
    pub data: String,
    pub duration: PhaseDuration,
    pub label: String,
    pub optimizer: Option<OptimizerConfig>,
    pub batch_size: Option<usize>,
    pub accum_steps: Option<usize>,
    pub seq_len: Option<usize>,
    pub save_every: Option<usize>,
    pub log_every: Option<usize>,
    pub max_grad_norm: Option<f32>,
    pub warmup_steps: Option<usize>,
    // Generation parameters for think_rounds speak phase
    pub max_gen_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
}

#[derive(Debug)]
pub enum PhaseDuration {
    Steps(usize),
    ThinkRounds(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_model_json() -> &'static str {
        r#""model": {"d_model": 64, "num_heads": 2, "seq_len": 32}"#
    }

    #[test]
    fn parse_legacy_config() {
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": "adamw_gpu_stacked",
                "lr": 0.001,
                "steps": 5000,
                "batch_size": 4
            }},
            "data": {{"path": "data/test", "format": "dolmino"}}
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        assert_eq!(cfg.build.optimizer.optimizer_type(), "adamw");
        assert_eq!(cfg.build.optimizer.lr(), 0.001);
        assert!(cfg.phases.is_none());

        let phases = cfg.resolved_phases().unwrap();
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].data, "data/test");
        assert!(matches!(phases[0].duration, PhaseDuration::Steps(5000)));
    }

    #[test]
    fn parse_nested_optimizer() {
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "adamw", "lr": 0.0001, "beta1": 0.95, "weight_decay": 0.05}},
                "steps": 1000
            }},
            "data": {{"path": "data/test"}}
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        assert_eq!(cfg.build.optimizer.optimizer_type(), "adamw");
        assert_eq!(cfg.build.optimizer.lr(), 0.0001);
        assert_eq!(cfg.build.optimizer.beta1(), 0.95);
        assert_eq!(cfg.build.optimizer.weight_decay(), 0.05);
        assert_eq!(cfg.build.optimizer.beta2(), 0.999); // default
    }

    #[test]
    fn parse_phase_list() {
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "adamw", "lr": 0.0003}},
                "batch_size": 8,
                "save_every": 500
            }},
            "phases": [
                {{"data": "data/foundations", "steps": 1000}},
                {{"data": "data/math", "think_rounds": 3, "label": "think-math"}},
                {{"data": "data/foundations", "steps": 2000, "optimizer": {{"type": "adamw", "lr": 0.0001}}, "batch_size": 4}}
            ]
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let phases = cfg.resolved_phases().unwrap();
        assert_eq!(phases.len(), 3);

        // Phase 0: steps
        assert!(matches!(phases[0].duration, PhaseDuration::Steps(1000)));
        assert!(phases[0].optimizer.is_none());

        // Phase 1: think_rounds
        assert!(matches!(phases[1].duration, PhaseDuration::ThinkRounds(3)));
        assert_eq!(phases[1].label, "think-math");

        // Phase 2: steps with overrides
        assert!(matches!(phases[2].duration, PhaseDuration::Steps(2000)));
        let opt = phases[2].optimizer.as_ref().unwrap();
        assert_eq!(opt.lr(), 0.0001);
        assert_eq!(phases[2].batch_size, Some(4));
    }

    #[test]
    fn phase_validation_both_steps_and_rounds() {
        let json = format!(r#"{{
            {},
            "build": {{"optimizer": {{"type": "adamw", "lr": 0.0003}}}},
            "phases": [
                {{"data": "data/test", "steps": 100, "think_rounds": 3}}
            ]
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let err = cfg.resolved_phases().unwrap_err();
        assert!(err.contains("both"));
    }

    #[test]
    fn phase_validation_neither_steps_nor_rounds() {
        let json = format!(r#"{{
            {},
            "build": {{"optimizer": {{"type": "adamw", "lr": 0.0003}}}},
            "phases": [
                {{"data": "data/test"}}
            ]
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let err = cfg.resolved_phases().unwrap_err();
        assert!(err.contains("neither"));
    }

    #[test]
    fn no_phases_no_data_errors() {
        let json = format!(r#"{{
            {},
            "build": {{"optimizer": {{"type": "adamw", "lr": 0.0003}}}}
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let err = cfg.resolved_phases().unwrap_err();
        assert!(err.contains("no `phases` and no `data`"));
    }

    #[test]
    fn legacy_compat_does_not_overwrite_explicit_nested() {
        // Edge case: nested optimizer explicitly sets lr=0.0003 (same as default),
        // legacy flat lr=0.001 should NOT overwrite it.
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "adamw", "lr": 0.0003}},
                "lr": 0.001,
                "steps": 1000
            }},
            "data": {{"path": "data/test"}}
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        // Nested optimizer explicitly set lr=0.0003, legacy lr=0.001 should not win
        assert_eq!(cfg.build.optimizer.lr(), 0.0003);
    }

    #[test]
    fn phase_optimizer_inherits_build_type() {
        // Phase optimizer specifies only lr — should inherit type from build
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "adamw", "lr": 0.0003, "weight_decay": 0.1}},
                "batch_size": 8,
                "save_every": 500
            }},
            "phases": [
                {{"data": "data/test", "steps": 100, "optimizer": {{"lr": 0.0001}}}}
            ]
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let phases = cfg.resolved_phases().unwrap();
        let opt = phases[0].optimizer.as_ref().unwrap();
        // Type inherited from build
        assert_eq!(opt.optimizer_type(), "adamw");
        // lr overridden by phase
        assert_eq!(opt.lr(), 0.0001);
        // weight_decay inherited from build
        assert_eq!(opt.weight_decay(), 0.1);
    }

    #[test]
    fn phase_rejects_model_structure_fields() {
        // d_model in a phase should fail — model structure is not overridable per phase
        let json = format!(r#"{{
            {},
            "build": {{"optimizer": {{"type": "adamw", "lr": 0.0003}}}},
            "phases": [
                {{"data": "data/test", "steps": 100, "d_model": 512}}
            ]
        }}"#, minimal_model_json());

        let err = Config::from_str(&json);
        assert!(err.is_err(), "phase with d_model should fail to parse");
        assert!(err.unwrap_err().contains("unknown field"));
    }

    #[test]
    fn think_rounds_generation_params() {
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "adamw", "lr": 0.0003}},
                "batch_size": 1
            }},
            "phases": [
                {{"data": "data/math", "think_rounds": 3, "batch_size": 1,
                  "max_gen_tokens": 128, "temperature": 0.8, "top_k": 50}}
            ]
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        let phases = cfg.resolved_phases().unwrap();
        assert_eq!(phases.len(), 1);
        assert!(matches!(phases[0].duration, PhaseDuration::ThinkRounds(3)));
        assert_eq!(phases[0].max_gen_tokens, Some(128));
        assert_eq!(phases[0].temperature, Some(0.8));
        assert_eq!(phases[0].top_k, Some(50));
    }

    #[test]
    fn future_optimizer_type_parses() {
        let json = format!(r#"{{
            {},
            "build": {{
                "optimizer": {{"type": "m3", "lr": 0.0003, "meta_lr": 0.001, "inner_steps": 5}}
            }},
            "data": {{"path": "data/test"}}
        }}"#, minimal_model_json());

        let cfg = Config::from_str(&json).unwrap();
        assert_eq!(cfg.build.optimizer.optimizer_type(), "m3");
        assert_eq!(cfg.build.optimizer.meta_lr, Some(0.001));
        assert_eq!(cfg.build.optimizer.inner_steps, Some(5));
    }
}
