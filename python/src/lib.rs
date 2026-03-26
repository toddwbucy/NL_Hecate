//! PyO3 bindings for NL-Hecate core.
//!
//! Stateless functional API — mirrors the Rust core exactly.
//! No Python-side math. All computation happens in Rust/CUDA.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use nl_hecate_core::model::{SWAConfig as RustConfig, SWAParams as RustParams};
use nl_hecate_core::model::{MAGConfig as RustMAGConfig, MAGParams as RustMAGParams, MemoryRuleKind, CompositionKind, AttentionalBias as RustAttentionalBias, FeatureMapKind as RustFeatureMapKind};
use nl_hecate_core::retention::{RetentionKind, default_retention};
use nl_hecate_core::m3::M3Config as RustM3Config;
use nl_hecate_core::dynamic_freq::{FrequencySchedule, LearnedFreqConfig};
use nl_hecate_core::model::LevelTapeStrategy;
use nl_hecate_core::cms_variants::{
    DeploymentVariant as RustDeploymentVariant,
    BlockConfig as RustBlockConfig,
    MultiBlockConfig as RustMultiBlockConfig,
};
use nl_hecate_core::gradient::compute_gradients as rust_compute_gradients;
use nl_hecate_core::mag::{mag_forward as rust_mag_forward, MAGForwardCache as RustMAGCache};
use nl_hecate_core::gradient::mag_compute_gradients as rust_mag_compute_gradients;
use nl_hecate_core::mag::{cms_forward as rust_cms_forward, CMSForwardCache as RustCMSCache};
use nl_hecate_core::gradient::cms_compute_gradients as rust_cms_compute_gradients;
use nl_hecate_core::conductor::{Conductor as RustConductor, Pulse as RustPulse, ContextState as RustContextState, ErrorBuffer as RustErrorBuffer, Checkpoint as RustCheckpoint, ConductorState as RustConductorState};
use nl_hecate_core::context_stream::StreamCursor;
use nl_hecate_core::context_stream::VecStream as RustVecStream;
use nl_hecate_core::model::{
    save_checkpoint as rust_save_checkpoint,
    save_build_checkpoint as rust_save_build_checkpoint,
    load_checkpoint as rust_load_checkpoint,
    BuildResumeState as RustBuildResumeState,
};
use nl_hecate_core::checkpoint::{
    save_stacked_safetensors as rust_save_stacked,
    load_stacked_safetensors as rust_load_stacked,
    is_stacked_checkpoint as rust_is_stacked_checkpoint,
};

// ── Tape strategy helpers ─────────────────────────────────────────────

fn parse_tape_strategy(s: &str) -> PyResult<LevelTapeStrategy> {
    match s.to_lowercase().as_str() {
        "exact" => Ok(LevelTapeStrategy::Exact),
        "proxy" => Ok(LevelTapeStrategy::Proxy),
        _ => Err(PyValueError::new_err(format!(
            "Unknown tape_strategy '{s}'. Expected: exact, proxy"
        ))),
    }
}

fn format_tape_strategy(ts: &LevelTapeStrategy) -> &'static str {
    match ts {
        LevelTapeStrategy::Exact => "exact",
        LevelTapeStrategy::Proxy => "proxy",
    }
}

// ── SWAConfig ────────────────────────────────────────────────────────

#[pyclass(frozen)]
struct SWAConfig {
    inner: RustConfig,
}

#[pymethods]
impl SWAConfig {
    #[new]
    fn new(
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        window_size: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        if d_model != num_heads * head_dim {
            return Err(PyValueError::new_err(format!(
                "d_model ({d_model}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
            )));
        }
        Ok(SWAConfig {
            inner: RustConfig {
                d_model,
                num_heads,
                head_dim,
                seq_len,
                window_size,
                vocab_size,
            },
        })
    }

    #[getter]
    fn d_model(&self) -> usize { self.inner.d_model }
    #[getter]
    fn num_heads(&self) -> usize { self.inner.num_heads }
    #[getter]
    fn head_dim(&self) -> usize { self.inner.head_dim }
    #[getter]
    fn seq_len(&self) -> usize { self.inner.seq_len }
    #[getter]
    fn window_size(&self) -> usize { self.inner.window_size }
    #[getter]
    fn vocab_size(&self) -> usize { self.inner.vocab_size }
}

// ── SWAParams ────────────────────────────────────────────────────────

#[pyclass]
struct SWAParams {
    inner: RustParams,
}

#[pymethods]
impl SWAParams {
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Return all weight matrices as a dict of flat lists.
    /// Keys: "w_embed", "w_q", "w_k", "w_v", "w_o", "w_unembed".
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("w_embed", self.inner.w_embed.clone())?;
        dict.set_item("w_q", self.inner.w_q.clone())?;
        dict.set_item("w_k", self.inner.w_k.clone())?;
        dict.set_item("w_v", self.inner.w_v.clone())?;
        dict.set_item("w_o", self.inner.w_o.clone())?;
        dict.set_item("w_unembed", self.inner.w_unembed.clone())?;
        Ok(dict)
    }
}

// ── Free functions ───────────────────────────────────────────────────

#[pyfunction]
fn create_config(
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window_size: usize,
    vocab_size: usize,
) -> PyResult<SWAConfig> {
    SWAConfig::new(d_model, num_heads, head_dim, seq_len, window_size, vocab_size)
}

#[pyfunction]
fn init_params(cfg: &SWAConfig, seed: u64) -> SWAParams {
    SWAParams {
        inner: RustParams::init(&cfg.inner, seed),
    }
}

fn validate_seq_lens(cfg: &SWAConfig, input_ids: &[usize], target_ids: &[usize]) -> PyResult<()> {
    let expected = cfg.inner.seq_len;
    if input_ids.len() != expected {
        return Err(PyValueError::new_err(format!(
            "input_ids length ({}) must equal seq_len ({expected})", input_ids.len()
        )));
    }
    if target_ids.len() != expected {
        return Err(PyValueError::new_err(format!(
            "target_ids length ({}) must equal seq_len ({expected})", target_ids.len()
        )));
    }
    Ok(())
}

#[pyfunction]
fn compute_gradients(
    params: &SWAParams,
    cfg: &SWAConfig,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
) -> PyResult<(f32, SWAParams)> {
    validate_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, grads) = rust_compute_gradients(&params.inner, &cfg.inner, &input_ids, &target_ids);
    Ok((loss, SWAParams { inner: grads }))
}

#[pyfunction]
fn apply_weight_gradients(params: &mut SWAParams, grads: &SWAParams, lr: f32) {
    params.inner.apply_weight_gradients(&grads.inner, lr);
}

// ── M3 Config parsing ─────────────────────────────────────────────

/// Parse M3Config from a Python dict.
/// Required keys: "k" (int). Optional: "etas", "thetas", "weights", "frequencies",
/// "use_newton_schulz", "ns_iterations". Missing optional keys use defaults for the given k.
fn parse_m3_config(d: &Bound<'_, PyDict>) -> PyResult<RustM3Config> {
    let k: usize = d.get_item("k")?
        .ok_or_else(|| PyValueError::new_err("m3 dict requires 'k' key"))?
        .extract()?;
    // Get defaults for this k
    let defaults = match k {
        1 => RustM3Config::default_k1(),
        2 => RustM3Config::default_k2(),
        4 => RustM3Config::default_k4(),
        _ => {
            // Non-standard k: require all fields explicitly
            if d.get_item("etas")?.is_none() || d.get_item("thetas")?.is_none()
               || d.get_item("weights")?.is_none() || d.get_item("frequencies")?.is_none() {
                return Err(PyValueError::new_err(format!(
                    "k={k} is non-standard; must provide 'etas', 'thetas', 'weights', 'frequencies'"
                )));
            }
            RustM3Config::default_k1() // values will be overwritten below
        }
    };
    let etas: Vec<f32> = match d.get_item("etas")? {
        Some(v) => v.extract()?,
        None => defaults.etas,
    };
    let thetas: Vec<f32> = match d.get_item("thetas")? {
        Some(v) => v.extract()?,
        None => defaults.thetas,
    };
    let weights: Vec<f32> = match d.get_item("weights")? {
        Some(v) => v.extract()?,
        None => defaults.weights,
    };
    let frequencies: Vec<usize> = match d.get_item("frequencies")? {
        Some(v) => v.extract()?,
        None => defaults.frequencies,
    };
    let use_newton_schulz: bool = match d.get_item("use_newton_schulz")? {
        Some(v) => v.extract()?,
        None => false,
    };
    let ns_iterations: usize = match d.get_item("ns_iterations")? {
        Some(v) => v.extract()?,
        None => 5,
    };
    let ns_dim: Option<usize> = match d.get_item("ns_dim")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    let cfg = RustM3Config { k, etas, thetas, weights, frequencies, use_newton_schulz, ns_iterations, ns_dim };
    cfg.validate().map_err(PyValueError::new_err)?;
    Ok(cfg)
}

// ── MAGConfig ──────────────────────────────────────────────────────

#[pyclass(frozen)]
struct MAGConfig {
    inner: RustMAGConfig,
}

#[pymethods]
impl MAGConfig {
    #[new]
    #[pyo3(signature = (
        d_model, num_heads, head_dim, seq_len, window_size, vocab_size, memory_enabled,
        k=1, chunk_sizes=None, memory_rule="delta", composition="mag",
        d_hidden=None, lp_p=None, sign_sharpness=None, lq_q=None, lambda_local=None, lambda_2=None,
        delta=None, m_slots=None, d_compress=None, lambda_k=None, lambda_v=None,
        retention=None, m3=None, frequency_schedule=None, checkpoint_interval=None, tape_multiplier=1,
        attentional_bias=None, kernel_size=0, self_ref_chunk_size=1,
        projection_kind="static", self_generated_values=false,
        momentum_kind="none", momentum_d_hidden=0,
        alpha_floor=None, alpha_ceil=None,
        theta_floor=None, theta_ceil=None,
        intermediate_size=0,
        m_norm_max=None,
        error_clip=None,
        feature_map="identity",
        feature_map_sigma=1.0,
        parallel_strategy=None,
        tnt_global_chunk_size=64,
        tnt_local_chunk_size=8,
        residual=false,
        b_alpha_init=None,
        b_theta_init=None,
        tape_strategies=None,
        hope_variant=None,
    ))]
    fn new(
        d_model: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        window_size: usize,
        vocab_size: usize,
        memory_enabled: bool,
        k: usize,
        chunk_sizes: Option<Vec<usize>>,
        memory_rule: &str,
        composition: &str,
        d_hidden: Option<usize>,
        lp_p: Option<f32>,
        sign_sharpness: Option<f32>,
        lq_q: Option<f32>,
        lambda_local: Option<f32>,
        lambda_2: Option<f32>,
        delta: Option<f32>,
        m_slots: Option<usize>,
        d_compress: Option<usize>,
        lambda_k: Option<f32>,
        lambda_v: Option<f32>,
        retention: Option<&str>,
        m3: Option<&Bound<'_, PyDict>>,
        frequency_schedule: Option<&Bound<'_, PyAny>>,
        checkpoint_interval: Option<usize>,
        tape_multiplier: usize,
        attentional_bias: Option<&str>,
        kernel_size: usize,
        self_ref_chunk_size: usize,
        projection_kind: &str,
        self_generated_values: bool,
        momentum_kind: &str,
        momentum_d_hidden: usize,
        alpha_floor: Option<Vec<f32>>,
        alpha_ceil: Option<Vec<f32>>,
        theta_floor: Option<Vec<f32>>,
        theta_ceil: Option<Vec<f32>>,
        intermediate_size: usize,
        m_norm_max: Option<Vec<f32>>,
        error_clip: Option<Vec<f32>>,
        feature_map: &str,
        feature_map_sigma: f32,
        parallel_strategy: Option<&str>,
        tnt_global_chunk_size: usize,
        tnt_local_chunk_size: usize,
        residual: bool,
        b_alpha_init: Option<Vec<f32>>,
        b_theta_init: Option<Vec<f32>>,
        tape_strategies: Option<Vec<String>>,
        hope_variant: Option<&str>,
    ) -> PyResult<Self> {
        if d_model != num_heads * head_dim {
            return Err(PyValueError::new_err(format!(
                "d_model ({d_model}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
            )));
        }
        if k < 1 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        if self_ref_chunk_size < 1 {
            return Err(PyValueError::new_err("self_ref_chunk_size must be >= 1"));
        }
        let chunk_sizes = chunk_sizes.unwrap_or_else(|| vec![1; k]);
        if chunk_sizes.len() != k {
            return Err(PyValueError::new_err(format!(
                "chunk_sizes length ({}) must equal k ({k})", chunk_sizes.len()
            )));
        }
        for (i, &cs) in chunk_sizes.iter().enumerate() {
            if cs < 1 {
                return Err(PyValueError::new_err(format!(
                    "chunk_sizes[{i}] must be >= 1, got {cs}"
                )));
            }
        }
        if let Some(ref v) = m_norm_max {
            if !v.is_empty() && v.len() != k {
                return Err(PyValueError::new_err(format!(
                    "m_norm_max length ({}) must equal k ({k})", v.len()
                )));
            }
        }
        if let Some(ref v) = error_clip {
            if !v.is_empty() && v.len() != k {
                return Err(PyValueError::new_err(format!(
                    "error_clip length ({}) must equal k ({k})", v.len()
                )));
            }
        }
        if let Some(ref v) = b_alpha_init {
            if !v.is_empty() && v.len() != k {
                return Err(PyValueError::new_err(format!(
                    "b_alpha_init length ({}) must equal k ({k})", v.len()
                )));
            }
        }
        if let Some(ref v) = b_theta_init {
            if !v.is_empty() && v.len() != k {
                return Err(PyValueError::new_err(format!(
                    "b_theta_init length ({}) must equal k ({k})", v.len()
                )));
            }
        }
        let comp = match composition.to_lowercase().as_str() {
            "mag" => CompositionKind::MAG,
            "mal" => CompositionKind::MAL,
            "mac" => CompositionKind::MAC,
            _ => return Err(PyValueError::new_err(format!(
                "Unknown composition '{composition}'. Expected: mag, mal, mac"
            ))),
        };
        let rule = match memory_rule.to_lowercase().as_str() {
            "delta" => MemoryRuleKind::DeltaRule,
            "titans" => MemoryRuleKind::TitansLMM,
            "hebbian" => MemoryRuleKind::HebbianRule,
            "moneta" => MemoryRuleKind::Moneta,
            "yaad" => MemoryRuleKind::YAAD,
            "memora" => MemoryRuleKind::MEMORA,
            "lattice" => MemoryRuleKind::LatticeOSR,
            "trellis" => MemoryRuleKind::Trellis,
            "atlas" | "atlas_omega" => MemoryRuleKind::AtlasOmega,
            "swiglu_mlp" | "swiglu" => MemoryRuleKind::SwiGluMlp,
            _ => return Err(PyValueError::new_err(format!(
                "Unknown memory_rule '{memory_rule}'. Expected: delta, titans, hebbian, moneta, yaad, memora, lattice, trellis, atlas, atlas_omega, swiglu_mlp"
            ))),
        };
        let ret_kind = match retention {
            Some(s) => match s.to_lowercase().as_str() {
                "l2" | "l2_weight_decay" => RetentionKind::L2WeightDecay,
                "kl" | "kl_divergence" => RetentionKind::KLDivergence,
                "elastic_net" | "elastic" => RetentionKind::ElasticNet,
                "sphere" | "sphere_normalization" => RetentionKind::SphereNormalization,
                _ => return Err(PyValueError::new_err(format!(
                    "Unknown retention '{s}'. Expected: l2, kl, elastic_net, sphere"
                ))),
            },
            None => default_retention(rule),
        };
        let m3_cfg = match m3 {
            Some(d) => Some(parse_m3_config(d)?),
            None => None,
        };
        let freq_sched = match frequency_schedule {
            None => FrequencySchedule::Fixed,
            Some(val) => {
                if let Ok(s) = val.extract::<String>() {
                    match s.to_lowercase().as_str() {
                        "fixed" => FrequencySchedule::Fixed,
                        "learned" => FrequencySchedule::Learned(LearnedFreqConfig::default()),
                        _ => return Err(PyValueError::new_err(format!(
                            "Unknown frequency_schedule '{}'. Expected: 'fixed', 'learned', or dict", s
                        ))),
                    }
                } else if let Ok(d) = val.downcast::<PyDict>() {
                    let threshold: f32 = d.get_item("threshold")?
                        .map(|v| v.extract()).transpose()?.unwrap_or(0.5);
                    let anneal_steps: usize = d.get_item("anneal_steps")?
                        .map(|v| v.extract()).transpose()?.unwrap_or(0);
                    FrequencySchedule::Learned(LearnedFreqConfig { threshold, anneal_steps })
                } else {
                    return Err(PyValueError::new_err(
                        "frequency_schedule must be a string ('fixed'/'learned') or dict"
                    ));
                }
            }
        };
        let bias_kind = match attentional_bias {
            None | Some("l2") | Some("L2") => RustAttentionalBias::L2,
            Some("l1") | Some("L1") => RustAttentionalBias::L1,
            Some(s) if s.starts_with("lp(") || s.starts_with("Lp(") => {
                let inner = s.trim_start_matches(|c: char| c != '(')
                    .trim_start_matches('(').trim_end_matches(')');
                let p: f32 = inner.parse().map_err(|_| PyValueError::new_err(
                    format!("Invalid Lp parameter: '{s}'. Expected format: 'Lp(3.0)'")
                ))?;
                nl_hecate_core::moneta::normalize_bias(RustAttentionalBias::Lp(p))
            }
            Some(s) => return Err(PyValueError::new_err(format!(
                "Unknown attentional_bias '{s}'. Expected: 'L2', 'L1', or 'Lp(p)'"
            ))),
        };
        let fm_kind = match feature_map.to_lowercase().as_str() {
            "identity" => RustFeatureMapKind::Identity,
            "random_fourier" | "rff" => {
                if !feature_map_sigma.is_finite() || feature_map_sigma <= 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "feature_map_sigma must be a positive finite number, got {feature_map_sigma}"
                    )));
                }
                RustFeatureMapKind::RandomFourier { sigma: feature_map_sigma }
            }
            "elu" => RustFeatureMapKind::ELU,
            _ => return Err(PyValueError::new_err(format!(
                "Unknown feature_map '{feature_map}'. Expected: identity, random_fourier, elu"
            ))),
        };
        let parallel_cfg = match parallel_strategy {
            Some(s) => {
                let pcfg = match s.to_lowercase().as_str() {
                    "sequential" | "none" => nl_hecate_core::parallel::ParallelConfig::sequential(),
                    "chunkwise" | "chunkwise_gd" => {
                        nl_hecate_core::parallel::ParallelConfig::chunkwise(chunk_sizes.get(0).copied().unwrap_or(1))
                    }
                    "tnt" | "tnt_hierarchical" => {
                        if tnt_global_chunk_size == 0 {
                            return Err(PyValueError::new_err("tnt_global_chunk_size must be >= 1"));
                        }
                        if tnt_local_chunk_size == 0 {
                            return Err(PyValueError::new_err("tnt_local_chunk_size must be >= 1"));
                        }
                        if tnt_local_chunk_size > tnt_global_chunk_size {
                            return Err(PyValueError::new_err(format!(
                                "tnt_local_chunk_size ({}) must be <= tnt_global_chunk_size ({})",
                                tnt_local_chunk_size, tnt_global_chunk_size
                            )));
                        }
                        if tnt_global_chunk_size % tnt_local_chunk_size != 0 {
                            return Err(PyValueError::new_err(format!(
                                "tnt_global_chunk_size ({}) must be divisible by tnt_local_chunk_size ({})",
                                tnt_global_chunk_size, tnt_local_chunk_size
                            )));
                        }
                        nl_hecate_core::parallel::ParallelConfig {
                            strategy: nl_hecate_core::parallel::ParallelStrategy::TNTHierarchical,
                            chunk_size: tnt_local_chunk_size,
                            tnt_global_chunk_size,
                            tnt_local_chunk_size,
                        }
                    }
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown parallel_strategy '{s}'. Expected: sequential, chunkwise, tnt_hierarchical"
                    ))),
                };
                Some(pcfg)
            }
            None => None,
        };
        Ok(MAGConfig {
            inner: RustMAGConfig {
                swa: RustConfig {
                    d_model,
                    num_heads,
                    head_dim,
                    seq_len,
                    window_size,
                    vocab_size,
                },
                memory_enabled,
                composition: comp,
                memory_rule: rule,
                k,
                chunk_sizes,
                d_hidden: d_hidden.unwrap_or(0),
                lp_p: lp_p.unwrap_or(2.0),
                sign_sharpness: sign_sharpness.unwrap_or(10.0),
                lq_q: lq_q.unwrap_or(2.0),
                lambda_local: lambda_local.unwrap_or(0.0),
                lambda_2: lambda_2.unwrap_or(0.0),
                delta: delta.unwrap_or(1.0),
                m_slots: m_slots.unwrap_or(0),
                d_compress: d_compress.unwrap_or(0),
                lambda_k: lambda_k.unwrap_or(0.0),
                lambda_v: lambda_v.unwrap_or(0.0),
                parallel: parallel_cfg,
                retention: ret_kind,
                m3: m3_cfg,
                frequency_schedule: freq_sched,
                checkpoint_interval,
                tape_multiplier: tape_multiplier.max(1),
                hope_variant: match hope_variant.unwrap_or("freq_gated").to_lowercase().as_str() {
                    "freq_gated" | "freqgated" => nl_hecate_core::model::HopeVariant::FreqGated,
                    "chained" | "chain" => nl_hecate_core::model::HopeVariant::Chained,
                    "independent" => nl_hecate_core::model::HopeVariant::Independent,
                    "sequential" => nl_hecate_core::model::HopeVariant::Sequential,
                    "nested" => nl_hecate_core::model::HopeVariant::Nested,
                    other => return Err(PyValueError::new_err(format!(
                        "Unknown hope_variant '{other}'. Expected: freq_gated, chained, independent, sequential, nested"
                    ))),
                },
                lattice_variant: nl_hecate_core::model::LatticeVariant::Decode,
                n_persistent: 0,
                attentional_bias: bias_kind,
                kernel_size,
                momentum_kind: match momentum_kind.to_lowercase().as_str() {
                    "none" => nl_hecate_core::model::MomentumKind::None,
                    "ema" => nl_hecate_core::model::MomentumKind::EMA,
                    "delta_momentum" => nl_hecate_core::model::MomentumKind::DeltaMomentum,
                    "deep_momentum" => nl_hecate_core::model::MomentumKind::DeepMomentum,
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown momentum_kind '{momentum_kind}'. Expected: none, ema, delta_momentum, deep_momentum"
                    ))),
                },
                momentum_d_hidden,
                projection_kind: match projection_kind.to_lowercase().as_str() {
                    "static" => nl_hecate_core::model::ProjectionKind::Static,
                    "adaptive" => nl_hecate_core::model::ProjectionKind::Adaptive,
                    _ => return Err(PyValueError::new_err(format!(
                        "Unknown projection_kind '{projection_kind}'. Expected: static, adaptive"
                    ))),
                },
                self_generated_values,
                self_ref_chunk_size,
                alpha_floor: alpha_floor.unwrap_or_default(),
                alpha_ceil: alpha_ceil.unwrap_or_default(),
                theta_floor: theta_floor.unwrap_or_default(),
                theta_ceil: theta_ceil.unwrap_or_default(),
                intermediate_size,
                m_norm_max: m_norm_max.unwrap_or_default(),
                error_clip: error_clip.unwrap_or_default(),
                feature_map: fm_kind,
                residual,
                b_alpha_init: b_alpha_init.unwrap_or_default(),
                b_theta_init: b_theta_init.unwrap_or_default(),
                tape_strategies: match tape_strategies {
                    Some(strs) => {
                        if !strs.is_empty() && strs.len() != k {
                            return Err(PyValueError::new_err(format!(
                                "tape_strategies length ({}) must equal k ({k}) when non-empty",
                                strs.len(),
                            )));
                        }
                        strs.iter().map(|s| parse_tape_strategy(s)).collect::<PyResult<Vec<_>>>()?
                    }
                    None => Vec::new(),
                },
            },
        })
    }

    #[getter]
    fn d_model(&self) -> usize { self.inner.swa.d_model }
    #[getter]
    fn num_heads(&self) -> usize { self.inner.swa.num_heads }
    #[getter]
    fn head_dim(&self) -> usize { self.inner.swa.head_dim }
    #[getter]
    fn seq_len(&self) -> usize { self.inner.swa.seq_len }
    #[getter]
    fn window_size(&self) -> usize { self.inner.swa.window_size }
    #[getter]
    fn vocab_size(&self) -> usize { self.inner.swa.vocab_size }
    #[getter]
    fn memory_enabled(&self) -> bool { self.inner.memory_enabled }
    #[getter]
    fn composition(&self) -> &str {
        match self.inner.composition {
            CompositionKind::MAG => "mag",
            CompositionKind::MAL => "mal",
            CompositionKind::MAC => "mac",
        }
    }
    #[getter]
    fn memory_rule(&self) -> &str {
        match self.inner.memory_rule {
            MemoryRuleKind::DeltaRule => "delta",
            MemoryRuleKind::TitansLMM => "titans",
            MemoryRuleKind::HebbianRule => "hebbian",
            MemoryRuleKind::Moneta => "moneta",
            MemoryRuleKind::YAAD => "yaad",
            MemoryRuleKind::MEMORA => "memora",
            MemoryRuleKind::LatticeOSR => "lattice",
            MemoryRuleKind::Trellis => "trellis",
            MemoryRuleKind::AtlasOmega => "atlas",
            MemoryRuleKind::SwiGluMlp => "swiglu_mlp",
        }
    }
    #[getter]
    fn intermediate_size(&self) -> usize { self.inner.intermediate_size }
    #[getter]
    fn k(&self) -> usize { self.inner.k }
    #[getter]
    fn chunk_sizes(&self) -> Vec<usize> { self.inner.chunk_sizes.clone() }
    #[getter]
    fn projection_kind(&self) -> &str {
        match self.inner.projection_kind {
            nl_hecate_core::model::ProjectionKind::Static => "static",
            nl_hecate_core::model::ProjectionKind::Adaptive => "adaptive",
        }
    }
    #[getter]
    fn momentum_kind(&self) -> &str {
        match self.inner.momentum_kind {
            nl_hecate_core::model::MomentumKind::None => "none",
            nl_hecate_core::model::MomentumKind::EMA => "ema",
            nl_hecate_core::model::MomentumKind::DeltaMomentum => "delta_momentum",
            nl_hecate_core::model::MomentumKind::DeepMomentum => "deep_momentum",
        }
    }
    #[getter]
    fn self_generated_values(&self) -> bool { self.inner.self_generated_values }
    #[getter]
    fn self_ref_chunk_size(&self) -> usize { self.inner.self_ref_chunk_size }
    #[getter]
    fn momentum_d_hidden(&self) -> usize { self.inner.momentum_d_hidden }
    #[getter]
    fn alpha_floor(&self) -> Vec<f32> { self.inner.alpha_floor.clone() }
    #[getter]
    fn alpha_ceil(&self) -> Vec<f32> { self.inner.alpha_ceil.clone() }
    #[getter]
    fn theta_floor(&self) -> Vec<f32> { self.inner.theta_floor.clone() }
    #[getter]
    fn theta_ceil(&self) -> Vec<f32> { self.inner.theta_ceil.clone() }
    #[getter]
    fn m_norm_max(&self) -> Vec<f32> { self.inner.m_norm_max.clone() }
    #[getter]
    fn error_clip(&self) -> Vec<f32> { self.inner.error_clip.clone() }
    #[getter]
    fn feature_map(&self) -> (String, Option<f64>) {
        match self.inner.feature_map {
            RustFeatureMapKind::Identity => ("identity".to_string(), None),
            RustFeatureMapKind::RandomFourier { sigma } => ("random_fourier".to_string(), Some(sigma as f64)),
            RustFeatureMapKind::ELU => ("elu".to_string(), None),
        }
    }

    #[getter]
    fn residual(&self) -> bool { self.inner.residual }
    #[getter]
    fn b_alpha_init(&self) -> Vec<f32> { self.inner.b_alpha_init.clone() }
    #[getter]
    fn b_theta_init(&self) -> Vec<f32> { self.inner.b_theta_init.clone() }
    #[getter]
    fn tape_strategies(&self) -> Vec<String> {
        self.inner.tape_strategies.iter().map(|s| format_tape_strategy(s).to_string()).collect()
    }
    #[getter]
    fn hope_variant(&self) -> &str {
        match self.inner.hope_variant {
            nl_hecate_core::model::HopeVariant::FreqGated => "freq_gated",
            nl_hecate_core::model::HopeVariant::Chained => "chained",
            nl_hecate_core::model::HopeVariant::Independent => "independent",
            nl_hecate_core::model::HopeVariant::Sequential => "sequential",
            nl_hecate_core::model::HopeVariant::Nested => "nested",
        }
    }
}

// ── MAGParams ──────────────────────────────────────────────────────

#[pyclass]
struct MAGParams {
    inner: RustMAGParams,
}

#[pymethods]
impl MAGParams {
    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Return all weight matrices as a dict of flat lists.
    /// Keys: 6 SWA + 7 memory (level 0) = 13 total.
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        // SWA weights
        dict.set_item("w_embed", self.inner.swa.w_embed.clone())?;
        dict.set_item("w_q", self.inner.swa.w_q.clone())?;
        dict.set_item("w_k", self.inner.swa.w_k.clone())?;
        dict.set_item("w_v", self.inner.swa.w_v.clone())?;
        dict.set_item("w_o", self.inner.swa.w_o.clone())?;
        dict.set_item("w_unembed", self.inner.swa.w_unembed.clone())?;
        // Memory weights (level 0 for backward compat)
        dict.set_item("w_k_mem", self.inner.levels[0].w_k_mem.master().to_vec())?;
        dict.set_item("w_v_mem", self.inner.levels[0].w_v_mem.master().to_vec())?;
        dict.set_item("w_q_mem", self.inner.levels[0].w_q_mem.master().to_vec())?;
        dict.set_item("w_alpha", self.inner.levels[0].w_alpha.clone())?;
        dict.set_item("b_alpha", self.inner.levels[0].b_alpha.clone())?;
        dict.set_item("w_theta", self.inner.levels[0].w_theta.clone())?;
        dict.set_item("b_theta", self.inner.levels[0].b_theta.clone())?;
        Ok(dict)
    }

    /// Flatten all params into a single Vec<f32> for Python-side optimizers.
    /// Order: SWA(embed,q,k,v,o,unembed), per-level(k_mem,v_mem,q_mem,alpha,
    /// b_alpha,theta,b_theta,eta,b_eta,omega,freq,b_freq,k_conv,b_k_conv,
    /// q_conv,b_q_conv), then agg(alpha_mem,alpha_refl,persistent_tokens).
    fn get_flat_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.inner.num_params());
        flat.extend_from_slice(&self.inner.swa.w_embed);
        flat.extend_from_slice(&self.inner.swa.w_q);
        flat.extend_from_slice(&self.inner.swa.w_k);
        flat.extend_from_slice(&self.inner.swa.w_v);
        flat.extend_from_slice(&self.inner.swa.w_o);
        flat.extend_from_slice(&self.inner.swa.w_unembed);
        for level in &self.inner.levels {
            flat.extend_from_slice(level.w_k_mem.master());
            flat.extend_from_slice(level.w_v_mem.master());
            flat.extend_from_slice(level.w_q_mem.master());
            flat.extend_from_slice(&level.w_alpha);
            flat.extend_from_slice(&level.b_alpha);
            flat.extend_from_slice(&level.w_theta);
            flat.extend_from_slice(&level.b_theta);
            flat.extend_from_slice(&level.w_eta);
            flat.extend_from_slice(&level.b_eta);
            flat.extend_from_slice(&level.w_omega);
            flat.extend_from_slice(&level.w_freq);
            flat.extend_from_slice(&level.b_freq);
            flat.extend_from_slice(&level.w_k_conv);
            flat.extend_from_slice(&level.b_k_conv);
            flat.extend_from_slice(&level.w_q_conv);
            flat.extend_from_slice(&level.b_q_conv);
            flat.extend_from_slice(&level.m_k_init);
            flat.extend_from_slice(&level.m_v_init);
            flat.extend_from_slice(&level.m_q_init);
            flat.extend_from_slice(&level.m_eta_init);
            flat.extend_from_slice(&level.m_alpha_init);
            flat.extend_from_slice(&level.m_mem_init);
            // SwiGluMlp-specific: empty for all other rules
            flat.extend_from_slice(&level.gate_proj);
            flat.extend_from_slice(&level.up_proj);
            flat.extend_from_slice(&level.down_proj);
        }
        flat.extend_from_slice(&self.inner.alpha_mem);
        flat.extend_from_slice(&self.inner.alpha_refl);
        flat.extend_from_slice(&self.inner.persistent_tokens);
        flat
    }

    /// Restore params from a flat Vec<f32> (inverse of get_flat_weights).
    fn set_flat_weights(&mut self, flat: Vec<f32>) -> PyResult<()> {
        if flat.len() != self.inner.num_params() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("flat weights length {} != num_params {}", flat.len(), self.inner.num_params())));
        }
        let mut offset = 0usize;
        macro_rules! copy_slice {
            ($dst:expr) => {{
                let n = $dst.len();
                $dst.copy_from_slice(&flat[offset..offset + n]);
                offset += n;
            }};
        }
        // Bf16Storage fields: copy to master, then sync bf16 stored copy
        macro_rules! copy_bf16 {
            ($dst:expr) => {{
                let n = $dst.len();
                $dst.master_mut().copy_from_slice(&flat[offset..offset + n]);
                $dst.sync_from_master();
                offset += n;
            }};
        }
        copy_slice!(self.inner.swa.w_embed);
        copy_slice!(self.inner.swa.w_q);
        copy_slice!(self.inner.swa.w_k);
        copy_slice!(self.inner.swa.w_v);
        copy_slice!(self.inner.swa.w_o);
        copy_slice!(self.inner.swa.w_unembed);
        for level in &mut self.inner.levels {
            copy_bf16!(level.w_k_mem);
            copy_bf16!(level.w_v_mem);
            copy_bf16!(level.w_q_mem);
            copy_slice!(level.w_alpha);
            copy_slice!(level.b_alpha);
            copy_slice!(level.w_theta);
            copy_slice!(level.b_theta);
            copy_slice!(level.w_eta);
            copy_slice!(level.b_eta);
            copy_slice!(level.w_omega);
            copy_slice!(level.w_freq);
            copy_slice!(level.b_freq);
            copy_slice!(level.w_k_conv);
            copy_slice!(level.b_k_conv);
            copy_slice!(level.w_q_conv);
            copy_slice!(level.b_q_conv);
            copy_slice!(level.m_k_init);
            copy_slice!(level.m_v_init);
            copy_slice!(level.m_q_init);
            copy_slice!(level.m_eta_init);
            copy_slice!(level.m_alpha_init);
            copy_slice!(level.m_mem_init);
            // SwiGluMlp-specific: empty for all other rules
            copy_slice!(level.gate_proj);
            copy_slice!(level.up_proj);
            copy_slice!(level.down_proj);
        }
        copy_slice!(self.inner.alpha_mem);
        copy_slice!(self.inner.alpha_refl);
        copy_slice!(self.inner.persistent_tokens);
        Ok(())
    }

    /// Load MLP weights into a specific CMS level (SwiGluMlp rule).
    /// gate: [intermediate x d_model], up: same, down: [d_model x intermediate].
    fn set_level_mlp(
        &mut self,
        cfg: &MAGConfig,
        level: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    ) -> PyResult<()> {
        if level >= self.inner.levels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "level {level} out of range (k={})", self.inner.levels.len()
            )));
        }
        let d = cfg.inner.swa.d_model;
        let inter = cfg.inner.intermediate_size;
        if inter == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "set_level_mlp: intermediate_size is 0 — configure MAGConfig with intermediate_size > 0 for SwiGluMlp"
            ));
        }
        let expected_gate = inter * d;
        let expected_down = d * inter;
        if gate.len() != expected_gate {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "gate shape mismatch: expected {inter}×{d}={expected_gate} elements, got {}", gate.len()
            )));
        }
        if up.len() != expected_gate {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "up shape mismatch: expected {inter}×{d}={expected_gate} elements, got {}", up.len()
            )));
        }
        if down.len() != expected_down {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "down shape mismatch: expected {d}×{inter}={expected_down} elements, got {}", down.len()
            )));
        }
        self.inner.levels[level].gate_proj = gate;
        self.inner.levels[level].up_proj = up;
        self.inner.levels[level].down_proj = down;
        Ok(())
    }
}

// ── MAGForwardCache ────────────────────────────────────────────────

#[pyclass]
struct MAGForwardCache {
    inner: RustMAGCache,
}

#[pymethods]
impl MAGForwardCache {
    /// Return logits as flat list: [seq_len * vocab_size], row-major.
    fn get_logits(&self) -> Vec<f32> {
        self.inner.logits.clone()
    }
}

// ── MAG free functions ─────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (
    d_model, num_heads, head_dim, seq_len, window_size, vocab_size, memory_enabled,
    k=1, chunk_sizes=None, memory_rule="delta", composition="mag",
    d_hidden=None, lp_p=None, sign_sharpness=None, lq_q=None, lambda_local=None, lambda_2=None,
    delta=None, m_slots=None, d_compress=None, lambda_k=None, lambda_v=None,
    retention=None, m3=None, frequency_schedule=None, checkpoint_interval=None, tape_multiplier=1,
    attentional_bias=None, kernel_size=0, self_ref_chunk_size=1,
    projection_kind="static", self_generated_values=false,
    momentum_kind="none", momentum_d_hidden=0,
    alpha_floor=None, alpha_ceil=None,
    theta_floor=None, theta_ceil=None,
    intermediate_size=0,
    m_norm_max=None,
    error_clip=None,
    feature_map="identity",
    feature_map_sigma=1.0,
    parallel_strategy=None,
    tnt_global_chunk_size=64,
    tnt_local_chunk_size=8,
    residual=false,
    b_alpha_init=None,
    b_theta_init=None,
    tape_strategies=None,
    hope_variant=None,
))]
fn mag_create_config(
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window_size: usize,
    vocab_size: usize,
    memory_enabled: bool,
    k: usize,
    chunk_sizes: Option<Vec<usize>>,
    memory_rule: &str,
    composition: &str,
    d_hidden: Option<usize>,
    lp_p: Option<f32>,
    sign_sharpness: Option<f32>,
    lq_q: Option<f32>,
    lambda_local: Option<f32>,
    lambda_2: Option<f32>,
    delta: Option<f32>,
    m_slots: Option<usize>,
    d_compress: Option<usize>,
    lambda_k: Option<f32>,
    lambda_v: Option<f32>,
    retention: Option<&str>,
    m3: Option<&Bound<'_, PyDict>>,
    frequency_schedule: Option<&Bound<'_, PyAny>>,
    checkpoint_interval: Option<usize>,
    tape_multiplier: usize,
    attentional_bias: Option<&str>,
    kernel_size: usize,
    self_ref_chunk_size: usize,
    projection_kind: &str,
    self_generated_values: bool,
    momentum_kind: &str,
    momentum_d_hidden: usize,
    alpha_floor: Option<Vec<f32>>,
    alpha_ceil: Option<Vec<f32>>,
    theta_floor: Option<Vec<f32>>,
    theta_ceil: Option<Vec<f32>>,
    intermediate_size: usize,
    m_norm_max: Option<Vec<f32>>,
    error_clip: Option<Vec<f32>>,
    feature_map: &str,
    feature_map_sigma: f32,
    parallel_strategy: Option<&str>,
    tnt_global_chunk_size: usize,
    tnt_local_chunk_size: usize,
    residual: bool,
    b_alpha_init: Option<Vec<f32>>,
    b_theta_init: Option<Vec<f32>>,
    tape_strategies: Option<Vec<String>>,
    hope_variant: Option<&str>,
) -> PyResult<MAGConfig> {
    MAGConfig::new(
        d_model, num_heads, head_dim, seq_len, window_size, vocab_size, memory_enabled,
        k, chunk_sizes, memory_rule, composition,
        d_hidden, lp_p, sign_sharpness, lq_q, lambda_local, lambda_2, delta, m_slots, d_compress, lambda_k, lambda_v,
        retention, m3, frequency_schedule, checkpoint_interval, tape_multiplier, attentional_bias, kernel_size, self_ref_chunk_size,
        projection_kind, self_generated_values, momentum_kind, momentum_d_hidden,
        alpha_floor, alpha_ceil,
        theta_floor, theta_ceil, intermediate_size, m_norm_max, error_clip,
        feature_map, feature_map_sigma,
        parallel_strategy, tnt_global_chunk_size, tnt_local_chunk_size,
        residual,
        b_alpha_init, b_theta_init,
        tape_strategies,
        hope_variant,
    )
}

#[pyfunction]
fn mag_init_params(cfg: &MAGConfig, seed: u64) -> MAGParams {
    MAGParams {
        inner: RustMAGParams::init(&cfg.inner, seed),
    }
}

/// Push-up level stacking: shift existing levels to slower frequencies,
/// insert fresh L0 at the fast end.
///
/// `new_cfg.k` must equal `old.k + 1`. SWA weights and persistent tokens
/// are preserved exactly. Old level[i] → new level[i+1].
#[pyfunction]
#[pyo3(signature = (old, new_cfg, seed, push_up_init="random"))]
fn extend_params_push_up(old: &MAGParams, new_cfg: &MAGConfig, seed: u64, push_up_init: &str) -> PyResult<MAGParams> {
    let init = match push_up_init {
        "random" => nl_hecate_core::model::PushUpInit::Random,
        "clone" => nl_hecate_core::model::PushUpInit::Clone,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("push_up_init must be 'random' or 'clone', got '{push_up_init}'"))),
    };
    Ok(MAGParams {
        inner: old.inner.extend_push_up(&new_cfg.inner, seed, init),
    })
}

/// Stack-up level extension: keep existing levels in place, add fresh level at top.
///
/// `new_cfg.k` must equal `old.k + 1`. SWA weights and persistent tokens
/// are preserved exactly. Old level[i] → new level[i], fresh level[k] at top.
#[pyfunction]
fn extend_params_stack_up(old: &MAGParams, new_cfg: &MAGConfig, seed: u64) -> MAGParams {
    MAGParams {
        inner: old.inner.extend_stack_up(&new_cfg.inner, seed),
    }
}

fn validate_mag_seq_lens(cfg: &MAGConfig, input_ids: &[usize], target_ids: &[usize]) -> PyResult<()> {
    let expected = cfg.inner.swa.seq_len;
    let vocab = cfg.inner.swa.vocab_size;
    if input_ids.len() != expected {
        return Err(PyValueError::new_err(format!(
            "input_ids length ({}) must equal seq_len ({expected})", input_ids.len()
        )));
    }
    if target_ids.len() != expected {
        return Err(PyValueError::new_err(format!(
            "target_ids length ({}) must equal seq_len ({expected})", target_ids.len()
        )));
    }
    // Validate input_ids bounds (OOB embedding reads are unsafe memory access).
    // target_ids are NOT validated: the CUDA cross-entropy kernel safely skips
    // target >= vocab (zeros grad, skips loss). This allows masked targets
    // (vocab_size sentinel) for loss masking on user turns.
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok >= vocab {
            return Err(PyValueError::new_err(format!(
                "input_ids[{i}]={tok} must be < vocab_size ({vocab})"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
fn mag_forward(params: &MAGParams, cfg: &MAGConfig, input_ids: Vec<usize>, target_ids: Vec<usize>) -> PyResult<(f32, MAGForwardCache)> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, cache) = rust_mag_forward(&params.inner, &cfg.inner, &input_ids, &target_ids);
    Ok((loss, MAGForwardCache { inner: cache }))
}

#[pyfunction]
fn mag_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
) -> PyResult<(f32, MAGParams)> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, grads) = rust_mag_compute_gradients(&params.inner, &cfg.inner, &input_ids, &target_ids);
    Ok((loss, MAGParams { inner: grads }))
}

#[pyfunction]
fn mag_apply_weight_gradients(params: &mut MAGParams, grads: &MAGParams, lr: f32) {
    params.inner.apply_weight_gradients(&grads.inner, lr);
    // Weight tying: sync w_unembed^T → w_embed (CPU path)
    params.inner.sync_embed_from_unembed();
}

// ── MultiBlockConfig ──────────────────────────────────────────────────

#[pyclass(frozen)]
struct MultiBlockConfig {
    inner: RustMultiBlockConfig,
}

#[pymethods]
impl MultiBlockConfig {
    /// Create a Basic variant from an existing MAGConfig (backward compatible).
    #[staticmethod]
    fn basic(cfg: &MAGConfig) -> PyResult<Self> {
        let mbc = RustMultiBlockConfig::basic(&cfg.inner)
            .map_err(PyValueError::new_err)?;
        Ok(MultiBlockConfig { inner: mbc })
    }

    /// Create a Sequential variant (k non-decreasing across blocks).
    #[staticmethod]
    #[pyo3(signature = (blocks, d_model, num_heads, seq_len, vocab_size))]
    fn sequential(
        blocks: Vec<Bound<'_, PyDict>>,
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        let rust_blocks = blocks.iter().map(parse_block_config).collect::<PyResult<Vec<_>>>()?;
        let mbc = RustMultiBlockConfig::sequential(rust_blocks, d_model, num_heads, seq_len, vocab_size)
            .map_err(PyValueError::new_err)?;
        Ok(MultiBlockConfig { inner: mbc })
    }

    /// Create a Nested variant (every CMS block must have M3 config).
    #[staticmethod]
    #[pyo3(signature = (blocks, d_model, num_heads, seq_len, vocab_size))]
    fn nested(
        blocks: Vec<Bound<'_, PyDict>>,
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        let rust_blocks = blocks.iter().map(parse_block_config).collect::<PyResult<Vec<_>>>()?;
        let mbc = RustMultiBlockConfig::nested(rust_blocks, d_model, num_heads, seq_len, vocab_size)
            .map_err(PyValueError::new_err)?;
        Ok(MultiBlockConfig { inner: mbc })
    }

    /// Create an Independent variant (per-block independent schedules).
    #[staticmethod]
    #[pyo3(signature = (blocks, d_model, num_heads, seq_len, vocab_size))]
    fn independent(
        blocks: Vec<Bound<'_, PyDict>>,
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        let rust_blocks = blocks.iter().map(parse_block_config).collect::<PyResult<Vec<_>>>()?;
        let mbc = RustMultiBlockConfig::independent(rust_blocks, d_model, num_heads, seq_len, vocab_size)
            .map_err(PyValueError::new_err)?;
        Ok(MultiBlockConfig { inner: mbc })
    }

    /// Create a Hybrid variant (mix of CMS and non-CMS blocks).
    #[staticmethod]
    #[pyo3(signature = (blocks, d_model, num_heads, seq_len, vocab_size))]
    fn hybrid(
        blocks: Vec<Bound<'_, PyDict>>,
        d_model: usize,
        num_heads: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> PyResult<Self> {
        let rust_blocks = blocks.iter().map(parse_block_config).collect::<PyResult<Vec<_>>>()?;
        let mbc = RustMultiBlockConfig::hybrid(rust_blocks, d_model, num_heads, seq_len, vocab_size)
            .map_err(PyValueError::new_err)?;
        Ok(MultiBlockConfig { inner: mbc })
    }

    #[getter]
    fn variant(&self) -> &str {
        match self.inner.variant {
            RustDeploymentVariant::Basic => "basic",
            RustDeploymentVariant::Nested => "nested",
            RustDeploymentVariant::Sequential => "sequential",
            RustDeploymentVariant::Independent => "independent",
            RustDeploymentVariant::Hybrid => "hybrid",
        }
    }

    #[getter]
    fn num_blocks(&self) -> usize { self.inner.blocks.len() }

    #[getter]
    fn d_model(&self) -> usize { self.inner.d_model }

    #[getter]
    fn vocab_size(&self) -> usize { self.inner.vocab_size }

    fn total_params_estimate(&self) -> usize {
        self.inner.total_params_estimate()
    }
}

/// Parse a Python dict into a RustBlockConfig.
/// Required keys: "composition" (str). Optional: "memory_rule", "cms_enabled", "k", "m3".
fn parse_block_config(d: &Bound<'_, PyDict>) -> PyResult<RustBlockConfig> {
    let comp_str: String = d.get_item("composition")?
        .ok_or_else(|| PyValueError::new_err("block dict requires 'composition' key"))?
        .extract()?;
    let comp = match comp_str.to_lowercase().as_str() {
        "mag" => CompositionKind::MAG,
        "mal" => CompositionKind::MAL,
        "mac" => CompositionKind::MAC,
        _ => return Err(PyValueError::new_err(format!(
            "Unknown composition '{comp_str}'. Expected: mag, mal, mac"
        ))),
    };

    let cms_enabled: bool = match d.get_item("cms_enabled")? {
        Some(v) => v.extract()?,
        None => true,
    };

    if !cms_enabled {
        return Ok(RustBlockConfig::default_standard(comp));
    }

    let rule_str: String = match d.get_item("memory_rule")? {
        Some(v) => v.extract()?,
        None => "delta".into(),
    };
    let rule = match rule_str.to_lowercase().as_str() {
        "delta" => MemoryRuleKind::DeltaRule,
        "titans" => MemoryRuleKind::TitansLMM,
        "hebbian" => MemoryRuleKind::HebbianRule,
        "moneta" => MemoryRuleKind::Moneta,
        "yaad" => MemoryRuleKind::YAAD,
        "memora" => MemoryRuleKind::MEMORA,
        "lattice" => MemoryRuleKind::LatticeOSR,
        "trellis" => MemoryRuleKind::Trellis,
        "atlas" | "atlas_omega" => MemoryRuleKind::AtlasOmega,
        "swiglu_mlp" | "swiglu" => MemoryRuleKind::SwiGluMlp,
        _ => return Err(PyValueError::new_err(format!(
            "Unknown memory_rule '{rule_str}'"
        ))),
    };

    let k: usize = match d.get_item("k")? {
        Some(v) => v.extract()?,
        None => 1,
    };

    let mut block = RustBlockConfig::default_cms(k, rule, comp);

    // Optional M3 config
    if let Some(m3_dict) = d.get_item("m3")? {
        let m3_d: &Bound<'_, PyDict> = m3_dict.downcast()
            .map_err(|_| PyValueError::new_err("m3 must be a dict"))?;
        block.m3 = Some(parse_m3_config(m3_d)?);
    }

    Ok(block)
}

// ── Conductor ────────────────────────────────────────────────────────

#[pyclass(unsendable)]
struct Conductor {
    inner: RustConductor,
}

#[pymethods]
impl Conductor {
    #[new]
    fn new(k: usize, chunk_sizes: Vec<usize>) -> PyResult<Self> {
        if k < 1 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        if chunk_sizes.len() != k {
            return Err(PyValueError::new_err(format!(
                "chunk_sizes length ({}) must equal k ({k})", chunk_sizes.len()
            )));
        }
        for (i, &cs) in chunk_sizes.iter().enumerate() {
            if cs < 1 {
                return Err(PyValueError::new_err(format!(
                    "chunk_sizes[{i}] must be >= 1, got {cs}"
                )));
            }
        }
        Ok(Conductor { inner: RustConductor::new(k, chunk_sizes) })
    }

    /// Attach a VecStream for integrated data feeding. Consumes the stream.
    fn attach_stream(&mut self, stream: &mut VecStream) -> PyResult<()> {
        let vs = stream.inner.take().ok_or_else(||
            PyValueError::new_err("VecStream already consumed by another Conductor")
        )?;
        self.inner = RustConductor::new(self.inner.k, self.inner.chunk_sizes.clone())
            .with_stream(Box::new(vs));
        Ok(())
    }

    /// Generate pulse for current step (does NOT advance).
    fn pulse(&self) -> Pulse {
        Pulse { inner: self.inner.pulse() }
    }

    /// Advance step counter. Call AFTER all observers have read the pulse.
    fn advance(&mut self) {
        self.inner.advance();
    }

    /// Get next chunk from attached stream + generate pulse.
    /// Returns (input_ids, target_ids, Pulse) or None if corpus empty.
    fn next_chunk(&mut self, chunk_size: usize) -> PyResult<Option<(Vec<usize>, Vec<usize>, Pulse)>> {
        if !self.inner.has_stream() {
            return Err(PyValueError::new_err("no stream attached; call attach_stream() first"));
        }
        match self.inner.next_chunk(chunk_size) {
            Some((chunk, pulse)) => Ok(Some((
                chunk.input_ids,
                chunk.target_ids,
                Pulse { inner: pulse },
            ))),
            None => Ok(None),
        }
    }

    /// Restore conductor state from a build checkpoint dict.
    fn restore_from_dict(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        // Validate that checkpoint k and chunk_sizes match current Conductor
        if let Some(ck) = state.get_item("conductor_k")? {
            let ck_val: usize = ck.extract()?;
            if ck_val != self.inner.k {
                return Err(PyValueError::new_err(format!(
                    "checkpoint conductor_k ({}) != current k ({})", ck_val, self.inner.k
                )));
            }
        }
        if let Some(ccs) = state.get_item("conductor_chunk_sizes")? {
            let ccs_val: Vec<usize> = ccs.extract()?;
            if ccs_val != self.inner.chunk_sizes {
                return Err(PyValueError::new_err(format!(
                    "checkpoint conductor_chunk_sizes ({:?}) != current ({:?})",
                    ccs_val, self.inner.chunk_sizes
                )));
            }
        }
        let conductor_step: usize = state.get_item("conductor_step")?
            .ok_or_else(|| PyValueError::new_err("missing conductor_step"))?.extract()?;
        let stream_position: u64 = state.get_item("stream_position")?
            .ok_or_else(|| PyValueError::new_err("missing stream_position"))?.extract()?;
        let stream_chunk_id: u64 = state.get_item("stream_chunk_id")?
            .ok_or_else(|| PyValueError::new_err("missing stream_chunk_id"))?.extract()?;
        let stream_pulse_id: u64 = state.get_item("stream_pulse_id")?
            .ok_or_else(|| PyValueError::new_err("missing stream_pulse_id"))?.extract()?;
        let stream_content_hash: u64 = state.get_item("stream_content_hash")?
            .ok_or_else(|| PyValueError::new_err("missing stream_content_hash"))?.extract()?;
        let stream_rng_state: Option<u64> = match state.get_item("stream_rng_state")? {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        };
        let checkpoint = RustCheckpoint {
            conductor: RustConductorState {
                k: self.inner.k,
                chunk_sizes: self.inner.chunk_sizes.clone(),
                step: conductor_step,
            },
            stream: StreamCursor {
                position: stream_position,
                chunk_id: stream_chunk_id,
                pulse_id: stream_pulse_id,
                rng_state: stream_rng_state,
                content_hash: stream_content_hash,
            },
        };
        self.inner.restore(&checkpoint)
            .map_err(|e| PyValueError::new_err(format!("restore failed: {e:?}")))?;
        Ok(())
    }

    #[getter]
    fn step(&self) -> usize { self.inner.step() }

    #[getter]
    fn k(&self) -> usize { self.inner.k }
}

// ── Pulse ────────────────────────────────────────────────────────────

#[pyclass(frozen)]
struct Pulse {
    inner: RustPulse,
}

#[pymethods]
impl Pulse {
    #[getter]
    fn global_step(&self) -> usize { self.inner.global_step }

    #[getter]
    fn active_levels(&self) -> Vec<bool> { self.inner.active_levels.clone() }
}

// ── ContextState ────────────────────────────────────────────────────

#[pyclass]
struct ContextState {
    inner: RustContextState,
}

#[pymethods]
impl ContextState {
    #[new]
    fn new(k: usize, d: usize) -> Self {
        ContextState { inner: RustContextState::new(k, d) }
    }

    /// Per-level M matrices as list of flat f32 lists.
    #[getter]
    fn memory(&self) -> Vec<Vec<f32>> { self.inner.memory.clone() }

    /// Restore memory from a saved state (list of flat f32 lists, one per level).
    fn set_memory(&mut self, memory: Vec<Vec<f32>>) -> PyResult<()> {
        if memory.len() != self.inner.memory.len() {
            return Err(PyValueError::new_err(format!(
                "memory length ({}) must equal k ({})", memory.len(), self.inner.memory.len()
            )));
        }
        for (i, (new, old)) in memory.iter().zip(self.inner.memory.iter()).enumerate() {
            if new.len() != old.len() {
                return Err(PyValueError::new_err(format!(
                    "memory[{}] length ({}) must equal existing shape ({})",
                    i, new.len(), old.len()
                )));
            }
        }
        self.inner.memory = memory;
        Ok(())
    }

    /// Zero all memory matrices in-place. Used at document boundaries.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Seed self-referential state from outer-loop m_*_init parameters.
    /// No-op when projection_kind is Static (m_*_init fields are empty).
    fn seed_self_ref(&mut self, params: &MAGParams) {
        self.inner.seed_self_ref(&params.inner.levels);
    }

    #[getter]
    fn d(&self) -> usize { self.inner.d }
}

// ── OptimizerState (spec 33: optimizer snapshot/restore) ────────────

/// Host-side snapshot of AdamW optimizer moments and step counters.
/// Opaque to Python — created by `GpuModel.snapshot_optimizer()`,
/// consumed by `GpuModel.restore_optimizer()`.
#[cfg(feature = "cuda")]
#[pyclass]
struct OptimizerState {
    inner: nl_hecate_core::gpu_optimizer::HostOptimizerState,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl OptimizerState {
    /// Global optimizer step counter.
    #[getter]
    fn step(&self) -> u32 { self.inner.step }

    /// Number of CMS levels in this snapshot.
    #[getter]
    fn k(&self) -> usize { self.inner.level_steps.len() }
}

// ── ErrorBufferList ─────────────────────────────────────────────────

#[pyclass]
struct ErrorBufferList {
    inner: Vec<RustErrorBuffer>,
    d: usize,
}

#[pymethods]
impl ErrorBufferList {
    #[new]
    fn new(k: usize, d: usize) -> Self {
        let inner = (0..k).map(|_| RustErrorBuffer::new(d)).collect();
        ErrorBufferList { inner, d }
    }

    /// Reset all error buffers (recreate with zero gradients).
    /// Used at document boundaries alongside ContextState::reset().
    fn reset(&mut self) {
        let d = self.d;
        self.inner = (0..self.inner.len()).map(|_| RustErrorBuffer::new(d)).collect();
    }

    /// Number of accumulated gradient steps for a given level.
    fn steps_accumulated(&self, level: usize) -> PyResult<usize> {
        self.inner.get(level)
            .map(|b| b.steps_accumulated)
            .ok_or_else(|| PyValueError::new_err(format!("level {level} out of bounds")))
    }

    /// Apply accumulated gradients for one level and reset its buffer.
    fn apply_and_reset(&mut self, params: &mut MAGParams, level: usize, lr: f32) -> PyResult<()> {
        let buf = self.inner.get_mut(level)
            .ok_or_else(|| PyValueError::new_err(format!("level {level} out of bounds")))?;
        let level_params = params.inner.levels.get_mut(level)
            .ok_or_else(|| PyValueError::new_err(format!("params has no level {level}")))?;
        buf.apply_and_reset(level_params, lr);
        Ok(())
    }

    /// Apply accumulated gradients for all active levels in the pulse.
    fn apply_for_active(&mut self, params: &mut MAGParams, pulse: &Pulse, lr: f32) -> PyResult<()> {
        for level in 0..self.inner.len() {
            if pulse.inner.active_levels.get(level).copied().unwrap_or(false)
                && self.inner[level].steps_accumulated > 0
            {
                let level_params = params.inner.levels.get_mut(level)
                    .ok_or_else(|| PyValueError::new_err(format!("params has no level {level}")))?;
                self.inner[level].apply_and_reset(level_params, lr);
            }
        }
        Ok(())
    }

    #[getter]
    fn len(&self) -> usize { self.inner.len() }
}

// ── VecStream ────────────────────────────────────────────────────────

#[pyclass]
struct VecStream {
    inner: Option<RustVecStream>,
}

#[pymethods]
impl VecStream {
    #[new]
    fn new(corpus: Vec<usize>) -> PyResult<Self> {
        if corpus.len() < 2 {
            return Err(PyValueError::new_err("corpus must have at least 2 tokens"));
        }
        Ok(VecStream { inner: Some(RustVecStream::new(corpus)) })
    }

    /// Create from raw bytes (each byte is a token ID 0-255).
    /// Much faster than converting a billion-element Python list.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        if data.len() < 2 {
            return Err(PyValueError::new_err("data must have at least 2 bytes"));
        }
        let corpus: Vec<usize> = data.iter().map(|&b| b as usize).collect();
        Ok(VecStream { inner: Some(RustVecStream::new(corpus)) })
    }
}

// ── CMSForwardCache ─────────────────────────────────────────────────

#[pyclass]
struct CMSForwardCache {
    inner: RustCMSCache,
}

#[pymethods]
impl CMSForwardCache {
    /// Return logits as flat list: [seq_len * vocab_size], row-major.
    fn get_logits(&self) -> Vec<f32> {
        self.inner.logits.clone()
    }
}

// ── CMS free functions ──────────────────────────────────────────────

#[pyfunction]
fn cms_forward(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    pulse: &Pulse,
    context: &mut ContextState,
) -> PyResult<(f32, CMSForwardCache)> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, cache) = rust_cms_forward(
        &params.inner, &cfg.inner, &input_ids, &target_ids,
        &pulse.inner, &mut context.inner,
    );
    Ok((loss, CMSForwardCache { inner: cache }))
}

/// Compute CMS gradients via the Wengert tape (traced forward + automatic backward).
///
/// Combined forward+backward in one call. Equivalent to `cms_forward` + `cms_backward`
/// but uses the tape-based gradient path for correctness parity with the Rust core.
/// Frozen-level gradients are routed into `error_buffers`; active-level gradients
/// are returned in the gradient params.
#[pyfunction]
fn cms_compute_gradients(
    params: &MAGParams,
    cfg: &MAGConfig,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    pulse: &Pulse,
    context: &mut ContextState,
    error_buffers: &mut ErrorBufferList,
) -> PyResult<(f32, MAGParams)> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, grads) = rust_cms_compute_gradients(
        &params.inner, &cfg.inner, &input_ids, &target_ids,
        &pulse.inner, &mut context.inner, &mut error_buffers.inner,
    );
    Ok((loss, MAGParams { inner: grads }))
}

#[pyfunction]
fn save_checkpoint(path: &str, params: &MAGParams, cfg: &MAGConfig) -> PyResult<()> {
    rust_save_checkpoint(std::path::Path::new(path), &params.inner, &cfg.inner)
        .map_err(|e| PyValueError::new_err(format!("save_checkpoint failed: {e}")))
}

#[pyfunction]
fn save_build_checkpoint(
    path: &str, params: &MAGParams, cfg: &MAGConfig,
    conductor: &mut Conductor, context: &ContextState,
) -> PyResult<()> {
    // Use Conductor::checkpoint() to atomically capture both ConductorState
    // and the real StreamCursor (with correct position, chunk_id, etc.)
    if !conductor.inner.has_stream() {
        return Err(PyValueError::new_err(
            "save_build_checkpoint requires an attached stream on the Conductor"
        ));
    }
    let ckpt = conductor.inner.checkpoint();
    let build_state = RustBuildResumeState {
        conductor: ckpt.conductor,
        stream_cursor: ckpt.stream,
        context: context.inner.clone(),
        global_step: conductor.inner.step(),
    };
    rust_save_build_checkpoint(
        std::path::Path::new(path), &params.inner, &cfg.inner, build_state,
    ).map_err(|e| PyValueError::new_err(format!("save_build_checkpoint failed: {e}")))
}

#[pyfunction]
fn save_checkpoint_with_context(
    path: &str, params: &MAGParams, cfg: &MAGConfig,
    conductor: &Conductor, context: &ContextState,
) -> PyResult<()> {
    // Guard: stream-backed conductors must use save_build_checkpoint, which
    // captures the real StreamCursor. Using this function with an attached
    // stream would silently drop the data-loader position on resume.
    if conductor.inner.has_stream() {
        return Err(PyValueError::new_err(
            "save_checkpoint_with_context is for streamless BPE conductors; \
             use save_build_checkpoint for stream-backed conductors"
        ));
    }
    // For BPE-path runs: conductor has no attached stream (position lives in
    // the sidecar .cursor.json). Serialise M_l matrices with a zeroed
    // StreamCursor — callers must not use it for position resume.
    let build_state = RustBuildResumeState {
        conductor: RustConductorState {
            k: conductor.inner.k,
            chunk_sizes: conductor.inner.chunk_sizes.clone(),
            step: conductor.inner.step(),
        },
        stream_cursor: StreamCursor {
            position: 0,
            chunk_id: 0,
            pulse_id: 0,
            rng_state: None,
            content_hash: 0,
        },
        context: context.inner.clone(),
        global_step: conductor.inner.step(),
    };
    rust_save_build_checkpoint(
        std::path::Path::new(path), &params.inner, &cfg.inner, build_state,
    ).map_err(|e| PyValueError::new_err(format!("save_checkpoint_with_context failed: {e}")))
}

#[pyfunction]
fn load_checkpoint(path: &str) -> PyResult<(MAGParams, MAGConfig)> {
    let (params, config, _build_state) = rust_load_checkpoint(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("load_checkpoint failed: {e}")))?;
    Ok((MAGParams { inner: params }, MAGConfig { inner: config }))
}

#[pyfunction]
fn load_build_checkpoint(py: Python<'_>, path: &str) -> PyResult<(MAGParams, MAGConfig, PyObject)> {
    let (params, config, build_state) = rust_load_checkpoint(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("load_build_checkpoint failed: {e}")))?;
    let bs_obj = match build_state {
        Some(bs) => {
            let dict = PyDict::new(py);
            dict.set_item("global_step", bs.global_step)?;
            dict.set_item("conductor_step", bs.conductor.step)?;
            dict.set_item("conductor_k", bs.conductor.k)?;
            dict.set_item("conductor_chunk_sizes", bs.conductor.chunk_sizes)?;
            dict.set_item("stream_position", bs.stream_cursor.position)?;
            dict.set_item("stream_chunk_id", bs.stream_cursor.chunk_id)?;
            dict.set_item("stream_pulse_id", bs.stream_cursor.pulse_id)?;
            dict.set_item("stream_content_hash", bs.stream_cursor.content_hash)?;
            dict.set_item("stream_rng_state", bs.stream_cursor.rng_state)?;
            dict.set_item("context_d", bs.context.d)?;
            dict.set_item("context_memory", bs.context.memory)?;
            dict.into_any().unbind()
        }
        None => py.None(),
    };
    Ok((MAGParams { inner: params }, MAGConfig { inner: config }, bs_obj))
}

// ── Stacked checkpoint + extend_k ─────────────────────────────────────
// Spec: specs/infrastructure/22_stacked_extend_k_per_block.md

/// Check if a safetensors file is a stacked checkpoint.
#[pyfunction]
fn is_stacked_checkpoint(path: &str) -> PyResult<bool> {
    rust_is_stacked_checkpoint(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("is_stacked_checkpoint failed: {e}")))
}

/// Save stacked model checkpoint (n_blocks > 1).
/// Uses hierarchical keys: shared.embed.weight, block.{n}.level.{m}.*, etc.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (path, gpu_model, conductor=None, context=None))]
fn save_stacked_checkpoint(
    path: &str,
    gpu_model: &GpuStackedModel,
    conductor: Option<&mut Conductor>,
    context: Option<&ContextState>,
) -> PyResult<()> {
    let d = gpu_model.cfg.swa.d_model;
    let v = gpu_model.cfg.swa.vocab_size;
    let k = gpu_model.cfg.k;
    let host_params = gpu_model.params.to_host(d, v, k);

    let build_state = match (conductor, context) {
        (Some(cond), Some(ctx)) => {
            let ckpt = if cond.inner.has_stream() {
                cond.inner.checkpoint()
            } else {
                RustCheckpoint {
                    conductor: RustConductorState {
                        k: cond.inner.k,
                        chunk_sizes: cond.inner.chunk_sizes.clone(),
                        step: cond.inner.step(),
                    },
                    stream: StreamCursor {
                        position: 0, chunk_id: 0, pulse_id: 0, rng_state: None, content_hash: 0,
                    },
                }
            };
            Some(RustBuildResumeState {
                conductor: ckpt.conductor,
                stream_cursor: ckpt.stream,
                context: ctx.inner.clone(),
                global_step: cond.inner.step(),
            })
        }
        (None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "save_stacked_checkpoint requires both conductor and context, or neither",
            ));
        }
    };

    rust_save_stacked(
        std::path::Path::new(path), &host_params, &gpu_model.cfg, build_state.as_ref(),
    ).map_err(|e| PyValueError::new_err(format!("save_stacked_checkpoint failed: {e}")))
}

/// Load stacked model checkpoint.
/// Returns dict with: "config" (MAGConfig), "n_blocks", "params_json", "build_state" (dict|None).
/// The params_json is used to construct GpuStackedModel via from_params_json.
#[pyfunction]
fn load_stacked_checkpoint(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let (params, config, n_blocks, build_state) = rust_load_stacked(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("load_stacked_checkpoint failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("n_blocks", n_blocks)?;

    // Return MAGConfig as a proper PyO3 object
    let py_cfg = MAGConfig { inner: config };
    dict.set_item("config", Py::new(py, py_cfg)?)?;

    // Serialize host params as JSON for GpuStackedModel.from_params_json
    let params_json = serde_json::to_string(&params)
        .map_err(|e| PyValueError::new_err(format!("params serialization failed: {e}")))?;
    dict.set_item("params_json", params_json)?;

    match build_state {
        Some(bs) => {
            let bs_dict = PyDict::new(py);
            bs_dict.set_item("global_step", bs.global_step)?;
            bs_dict.set_item("conductor_step", bs.conductor.step)?;
            bs_dict.set_item("conductor_k", bs.conductor.k)?;
            bs_dict.set_item("conductor_chunk_sizes", bs.conductor.chunk_sizes)?;
            bs_dict.set_item("stream_position", bs.stream_cursor.position)?;
            bs_dict.set_item("stream_chunk_id", bs.stream_cursor.chunk_id)?;
            bs_dict.set_item("stream_pulse_id", bs.stream_cursor.pulse_id)?;
            bs_dict.set_item("stream_content_hash", bs.stream_cursor.content_hash)?;
            bs_dict.set_item("stream_rng_state", bs.stream_cursor.rng_state)?;
            bs_dict.set_item("context_d", bs.context.d)?;
            bs_dict.set_item("context_memory", bs.context.memory)?;
            dict.set_item("build_state", bs_dict)?;
        }
        None => {
            dict.set_item("build_state", py.None())?;
        }
    }

    Ok(dict.into_any().unbind())
}

/// Push-up level stacking for stacked multi-block models.
/// Each block independently shifts level[i] → level[i+1], fresh L0 per block.
#[pyfunction]
#[pyo3(signature = (path, new_cfg, seed, push_up_init="random"))]
fn extend_stacked_push_up(
    py: Python<'_>,
    path: &str,
    new_cfg: &MAGConfig,
    seed: u64,
    push_up_init: &str,
) -> PyResult<PyObject> {
    let init = match push_up_init {
        "random" => nl_hecate_core::model::PushUpInit::Random,
        "clone" => nl_hecate_core::model::PushUpInit::Clone,
        _ => return Err(PyValueError::new_err(
            format!("push_up_init must be 'random' or 'clone', got '{push_up_init}'"))),
    };
    let (params, _config, n_blocks, _build_state) = rust_load_stacked(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("extend_stacked_push_up: load failed: {e}")))?;

    let extended = params.extend_push_up(&new_cfg.inner, seed, init);

    // Serialize the extended params for GpuStackedModel construction
    let params_json = serde_json::to_string(&extended)
        .map_err(|e| PyValueError::new_err(format!("params serialization failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("params_json", params_json)?;
    dict.set_item("n_blocks", n_blocks)?;
    Ok(dict.into_any().unbind())
}

/// Clone expansion: duplicate existing levels to fill a larger k.
/// k=1→k=4: level[0] weights cloned into all 4 slots.
/// Gate biases use new config defaults for frequency-appropriate init.
#[pyfunction]
fn extend_stacked_clone(
    py: Python<'_>,
    path: &str,
    new_cfg: &MAGConfig,
    seed: u64,
) -> PyResult<PyObject> {
    let (params, _config, n_blocks, _build_state) = rust_load_stacked(std::path::Path::new(path))
        .map_err(|e| PyValueError::new_err(format!("extend_stacked_clone: load failed: {e}")))?;

    let extended = params.extend_clone(&new_cfg.inner, seed);

    let params_json = serde_json::to_string(&extended)
        .map_err(|e| PyValueError::new_err(format!("params serialization failed: {e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("params_json", params_json)?;
    dict.set_item("n_blocks", n_blocks)?;
    Ok(dict.into_any().unbind())
}

// ── Tape Summary Helpers (spec 49: consolidate 4 near-duplicate methods) ──

/// Validate input_ids/target_ids for tape summary calls.
#[cfg(feature = "cuda")]
fn validate_tape_inputs(s: usize, v: usize, input_ids: &[usize], target_ids: &[usize]) -> PyResult<()> {
    if s == 0 {
        return Err(PyValueError::new_err("seq_len must be > 0"));
    }
    if input_ids.is_empty() || input_ids.len() % s != 0 || target_ids.len() != input_ids.len() {
        return Err(PyValueError::new_err(format!(
            "input/target length must be batch_size * seq_len {} (got {})", s, input_ids.len()
        )));
    }
    if let Some(&max_id) = input_ids.iter().max() {
        if max_id >= v {
            return Err(PyValueError::new_err(format!(
                "input_ids contains {} >= vocab_size {}", max_id, v
            )));
        }
    }
    // target_ids may contain sentinel values >= vocab_size for masked/padded
    // positions (BPE batching). The cross-entropy loss ignores these positions,
    // so we only validate input_ids against vocab_size.
    Ok(())
}

/// Set the 8 core per-level fields shared by all tape summary methods.
#[cfg(feature = "cuda")]
fn set_core_level_fields(
    ldict: &pyo3::Bound<'_, PyDict>,
    level: usize,
    opaque_key: &str,
    block_count: usize,
    output_grad_norm: f32,
    dgd_delta_norm: f32,
    m_norm: f32,
    freq_gate_value: f32,
    is_frozen: bool,
    head_m_norms: &[f32],
) -> PyResult<()> {
    ldict.set_item("level", level)?;
    ldict.set_item("opaque_key", opaque_key)?;
    ldict.set_item("block_count", block_count)?;
    ldict.set_item("output_grad_norm", output_grad_norm)?;
    ldict.set_item("dgd_delta_norm", dgd_delta_norm)?;
    ldict.set_item("m_norm", m_norm)?;
    ldict.set_item("freq_gate_value", freq_gate_value)?;
    ldict.set_item("is_frozen", is_frozen)?;
    ldict.set_item("head_m_norms", head_m_norms)?;
    Ok(())
}

/// Build a PyDict for ThetaStats (inner-loop learning rate distribution).
#[cfg(feature = "cuda")]
fn make_theta_stats_dict<'py>(
    py: Python<'py>,
    ts: &nl_hecate_core::gpu_forward::ThetaStats,
) -> PyResult<pyo3::Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("count", ts.count)?;
    d.set_item("min", ts.min)?;
    d.set_item("max", ts.max)?;
    d.set_item("mean", ts.mean)?;
    d.set_item("median", ts.median)?;
    d.set_item("p95", ts.p95)?;
    d.set_item("p99", ts.p99)?;
    d.set_item("frac_at_ceil", ts.frac_at_ceil)?;
    Ok(d)
}

/// Build a PyDict for GateStats (alpha/eta gate distribution).
#[cfg(feature = "cuda")]
fn make_gate_stats_dict<'py>(
    py: Python<'py>,
    gs: &nl_hecate_core::gpu_forward::GateStats,
    frac_key: &str,
) -> PyResult<pyo3::Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("count", gs.count)?;
    d.set_item("min", gs.min)?;
    d.set_item("max", gs.max)?;
    d.set_item("mean", gs.mean)?;
    d.set_item("median", gs.median)?;
    d.set_item("p95", gs.p95)?;
    d.set_item("p99", gs.p99)?;
    d.set_item(frac_key, gs.frac_at_bound)?;
    Ok(d)
}

/// Build a PyDict for aggregated gate stats across stacked blocks.
/// Uses p99_max (max of per-block p99) instead of true combined p99.
#[cfg(feature = "cuda")]
fn make_aggregated_stats_dict<'py>(
    py: Python<'py>,
    count: usize,
    min: f32,
    max: f32,
    mean: f32,
    p99_max: f32,
    frac: f32,
    frac_key: &str,
) -> PyResult<pyo3::Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("count", count)?;
    d.set_item("min", min)?;
    d.set_item("max", max)?;
    d.set_item("mean", mean)?;
    d.set_item("p99_max", p99_max)?;
    d.set_item(frac_key, frac)?;
    Ok(d)
}

/// Aggregate gate stats across stacked blocks for one level.
///
/// Extracts (count, min, max, mean, p99, frac) from each block's stats using
/// the provided closure, then builds an aggregated dict with weighted mean,
/// min-of-mins, max-of-maxes, and max-of-p99s.
#[cfg(feature = "cuda")]
fn aggregate_gate_stats<T, F>(
    py: Python<'_>,
    ldict: &pyo3::Bound<'_, PyDict>,
    stats_grid: &[Vec<Option<T>>],
    level: usize,
    n_blocks: usize,
    extract: F,
    key: &str,
    frac_key: &str,
) -> PyResult<()>
where
    F: Fn(&T) -> (usize, f32, f32, f32, f32, f32),
{
    let mut total_count = 0usize;
    let mut total_min = f32::MAX;
    let mut total_max = f32::MIN;
    let mut weighted_sum = 0.0f32;
    let mut max_p99 = 0.0f32;
    let mut frac_sum = 0.0f32;
    let mut found = false;
    for bi in 0..n_blocks {
        if let Some(ref s) = stats_grid[bi][level] {
            let (count, min, max, mean, p99, frac) = extract(s);
            found = true;
            total_count += count;
            if min < total_min { total_min = min; }
            if max > total_max { total_max = max; }
            weighted_sum += mean * count as f32;
            if p99 > max_p99 { max_p99 = p99; }
            frac_sum += frac * count as f32;
        }
    }
    if found && total_count > 0 {
        ldict.set_item(key, make_aggregated_stats_dict(
            py, total_count, total_min, total_max,
            weighted_sum / total_count as f32, max_p99,
            frac_sum / total_count as f32, frac_key,
        )?)?;
    }
    Ok(())
}

// ── GPU-Resident Model ───────────────────────────────────────────────

/// GPU-resident model: all parameters live on GPU.
/// Forward/backward/update happen entirely on device.
/// Only input_ids, target_ids, and loss cross PCIe.
#[cfg(feature = "cuda")]
#[pyclass(unsendable)]
struct GpuModel {
    #[allow(dead_code)]
    params: nl_hecate_core::gpu_params::GpuMAGParams,
    context: nl_hecate_core::gpu_params::GpuContextState,
    cfg: nl_hecate_core::model::MAGConfig,
    adamw_state: Option<nl_hecate_core::gpu_optimizer::GpuAdamWState>,
    m3_state: Option<nl_hecate_core::gpu_optimizer::GpuM3State>,
    kv_cache: Option<nl_hecate_core::gpu_forward::GpuKVCache>,
    decode_workspace: Option<nl_hecate_core::gpu_forward::DecodeWorkspace>,
    /// Per-level reset intervals for selective periodic reset (spec 57).
    /// Empty vec = no reset. [1,1,1,1] = spec-08 all-ones. [1,8,64,512] = gear-shifting.
    reset_intervals: Vec<usize>,
    /// Per-level fire counters for selective periodic reset (spec 57).
    /// fire_counts[k] tracks fires since last reset of level k.
    fire_counts: Vec<usize>,
    /// Per-level gradient norms from the most recent step_adamw call, computed
    /// before global gradient clipping. Empty until the first step_adamw call.
    last_level_gnorms: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl GpuModel {
    /// Selective periodic reset (spec 57): for each active level, increment fire
    /// counter and reset M when counter reaches the level's reset interval.
    /// CS-32: called after observe, before next advance.
    fn maybe_reset_levels(&mut self, pulse: &nl_hecate_core::conductor::Pulse) {
        if self.reset_intervals.is_empty() { return; }
        for (k, &active) in pulse.active_levels.iter().enumerate() {
            if !active { continue; }
            self.fire_counts[k] += 1;
            if self.fire_counts[k] >= self.reset_intervals[k] {
                self.context.periodic_reset_level(k);
                self.fire_counts[k] = 0;
            }
        }
    }

    /// Resolve reset_intervals from memory_reset + optional explicit intervals.
    /// memory_reset=true, intervals=None → [1,1,...,1] (spec-08 compat)
    /// memory_reset=true, intervals=Some([1,8,64,512]) → selective (spec 57)
    /// memory_reset=false → [] (no reset)
    fn resolve_intervals(memory_reset: bool, intervals: Option<Vec<usize>>, k: usize) -> PyResult<Vec<usize>> {
        if !memory_reset {
            return Ok(Vec::new());
        }
        match intervals {
            Some(v) => {
                if v.len() != k {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("reset_intervals length {} != k={}", v.len(), k)));
                }
                for (i, &r) in v.iter().enumerate() {
                    if r < 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("reset_intervals[{}] must be >= 1, got {}", i, r)));
                    }
                }
                Ok(v)
            }
            None => Ok(vec![1usize; k]),
        }
    }
}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuModel {
    /// Create a GPU-resident model from a MAGConfig, random seed, and batch_size.
    /// All parameters are uploaded to GPU once. batch_size determines how many
    /// independent M-state slots are allocated in GpuContextState.
    #[new]
    #[pyo3(signature = (cfg, seed, batch_size=1, memory_reset=false, cuda_graph_warmup=0, reset_intervals=None))]
    fn new(
        cfg: &MAGConfig, seed: u64, batch_size: usize,
        memory_reset: bool, cuda_graph_warmup: usize,
        reset_intervals: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let k = cfg.inner.k;
        let intervals = Self::resolve_intervals(memory_reset, reset_intervals, k)?;
        let host_params = nl_hecate_core::model::MAGParams::init(&cfg.inner, seed);
        let gpu_params = nl_hecate_core::gpu_params::GpuMAGParams::from_host(&host_params);
        let gpu_context = nl_hecate_core::gpu_params::GpuContextState::new(
            cfg.inner.k, cfg.inner.swa.d_model, batch_size,
            Some(&cfg.inner), cuda_graph_warmup,
        );
        Ok(GpuModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            m3_state: None,
            kv_cache: None,
            decode_workspace: None,
            reset_intervals: intervals,
            fire_counts: vec![0usize; k],
            last_level_gnorms: Vec::new(),
        })
    }

    /// Create from existing host params (e.g., loaded from checkpoint).
    /// batch_size controls how many M-state slots are allocated for batched training.
    /// cuda_graph_warmup: steps before graph capture (0 = disabled, 100 = recommended for training).
    #[staticmethod]
    #[pyo3(signature = (params, cfg, batch_size=1, memory_reset=false, cuda_graph_warmup=0, reset_intervals=None))]
    fn from_params(
        params: &MAGParams, cfg: &MAGConfig,
        batch_size: usize, memory_reset: bool, cuda_graph_warmup: usize,
        reset_intervals: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let k = cfg.inner.k;
        let intervals = Self::resolve_intervals(memory_reset, reset_intervals, k)?;
        let gpu_params = nl_hecate_core::gpu_params::GpuMAGParams::from_host(&params.inner);
        let gpu_context = nl_hecate_core::gpu_params::GpuContextState::new(
            cfg.inner.k, cfg.inner.swa.d_model, batch_size,
            Some(&cfg.inner), cuda_graph_warmup,
        );
        Ok(GpuModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            m3_state: None,
            kv_cache: None,
            decode_workspace: None,
            reset_intervals: intervals,
            fire_counts: vec![0usize; k],
            last_level_gnorms: Vec::new(),
        })
    }

    /// One build step: forward + backward + weight update. Returns loss.
    fn step(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
            pulse: &Pulse, lr: f32) -> PyResult<f32> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        if input_ids.len() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input_ids length {} != seq_len {}", input_ids.len(), s)));
        }
        if target_ids.len() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("target_ids length {} != seq_len {}", target_ids.len(), s)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }
        // target_ids not validated: kernel safely skips target >= vocab (masking)
        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );

        let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache, false,
        );

        nl_hecate_core::gpu_backward::gpu_weight_update(
            &mut self.params, &grads, lr,
        );

        // Weight tying: sync w_unembed^T → w_embed after each update.
        // Compensates for vanishing embedding gradient in deep models.
        nl_hecate_core::gpu_backward::gpu_sync_embed_weights(
            &mut self.params,
            self.cfg.swa.d_model,
            self.cfg.swa.vocab_size,
        );

        Ok(loss)
    }

    /// Download parameters to host for checkpointing.
    fn to_host_params(&self) -> PyResult<MAGParams> {
        let host = self.params.to_host(&self.cfg);
        Ok(MAGParams { inner: host })
    }

    /// Download context state to host for checkpointing.
    fn to_host_context(&self) -> PyResult<ContextState> {
        let host = self.context.to_host(self.cfg.k);
        Ok(ContextState { inner: host })
    }

    /// GPU forward + backward only (no weight update). Returns (loss, grad_params).
    /// Used for hybrid GPU+AdamW: Python applies optimizer, then calls upload_params().
    fn backward_only(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
                     pulse: &Pulse) -> PyResult<(f32, MAGParams)> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        if input_ids.len() != s || target_ids.len() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input/target length must be seq_len {}", s)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }
        // target_ids not validated: kernel safely skips target >= vocab (masking)
        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );
        let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache, false,
        );
        // Download gradients to host as MAGParams
        let host_grads = grads.to_host(&self.cfg);
        Ok((loss, MAGParams { inner: host_grads }))
    }

    /// Upload host params to GPU (after Python-side optimizer update).
    fn upload_params(&mut self, params: &MAGParams) -> PyResult<()> {
        self.params = nl_hecate_core::gpu_params::GpuMAGParams::from_host(&params.inner);
        Ok(())
    }

    /// Full GPU build step with AdamW optimizer. Returns (loss, grad_norm).
    /// Zero PCIe traffic for weights — only input_ids, target_ids, and loss cross the bus.
    /// AdamW state is lazily created on first call.
    #[pyo3(signature = (input_ids, target_ids, pulse, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, max_grad_norm=1.0, collect_level_gnorms=false))]
    fn step_adamw(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
                  pulse: &Pulse, lr: f32, beta1: f32, beta2: f32,
                  eps: f32, weight_decay: f32, max_grad_norm: f32,
                  collect_level_gnorms: bool) -> PyResult<(f32, f32)> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        if s == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("seq_len must be > 0"));
        }
        // Accept batch_size * seq_len tokens (batch_size >= 1, derived from input length)
        if input_ids.is_empty() || input_ids.len() % s != 0 || target_ids.len() != input_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input/target length must be batch_size * seq_len {} (got {})", s, input_ids.len())));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }
        // target_ids not validated: kernel safely skips target >= vocab (masking)

        // Forward
        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );

        // Backward
        let mut grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache, false,
        );

        // Lazy-init AdamW state
        if self.adamw_state.is_none() {
            self.adamw_state = Some(
                nl_hecate_core::gpu_optimizer::GpuAdamWState::from_params(&self.params)
            );
        }
        let state = self.adamw_state.as_mut().unwrap();

        // Per-level gradient norms before clipping — opt-in to avoid overhead on
        // steps where the caller won't read them (e.g., non-logging steps).
        if collect_level_gnorms {
            self.last_level_gnorms =
                nl_hecate_core::gpu_optimizer::gpu_per_level_grad_norms(&grads, state);
        } else {
            self.last_level_gnorms.clear();
        }

        // AdamW update (Pulse-gated, with grad clipping)
        let grad_norm = nl_hecate_core::gpu_optimizer::gpu_adamw_update(
            &mut self.params, &mut grads, state,
            &pulse.inner,
            lr, beta1, beta2, eps, weight_decay, max_grad_norm,
        );

        // Weight tying: sync w_unembed^T → w_embed
        nl_hecate_core::gpu_backward::gpu_sync_embed_weights(
            &mut self.params,
            self.cfg.swa.d_model,
            self.cfg.swa.vocab_size,
        );

        // Selective periodic reset (spec 57, extends eq-006).
        // CS-32 compliant: reset happens after observe, before next advance.
        self.maybe_reset_levels(&pulse.inner);

        Ok((loss, grad_norm))
    }

    /// Get current AdamW optimizer step count.
    #[getter]
    fn adamw_step(&self) -> u32 {
        self.adamw_state.as_ref().map_or(0, |s| s.step)
    }

    /// Reset AdamW optimizer state (moments and step counter).
    /// Next step_adamw call will re-initialize from scratch.
    /// Use after learning probes to prevent corrupted moments from affecting training.
    fn reset_optimizer(&mut self) {
        self.adamw_state = None;
    }

    /// Snapshot AdamW optimizer state to host memory (spec 33).
    /// Returns None if optimizer hasn't been initialized yet (no training steps taken).
    ///
    /// NOTE: M3 optimizer state is NOT included in the snapshot — M3 moments and
    /// step counters will be re-initialized from scratch on restore. M3 re-warms
    /// quickly (EMA), so this is acceptable for checkpoint recovery. Full M3
    /// serialization is tracked for a follow-up PR.
    fn snapshot_optimizer(&self) -> Option<OptimizerState> {
        if self.m3_state.is_some() {
            eprintln!("WARNING: snapshot_optimizer() does not include M3 state — \
                       M3 moments/step will be re-initialized on restore");
        }
        self.adamw_state.as_ref().map(|state| {
            OptimizerState { inner: state.to_host() }
        })
    }

    /// Restore AdamW optimizer state from a host snapshot (spec 33).
    /// Replaces the current optimizer state with the snapshot's moments and step counters.
    fn restore_optimizer(&mut self, state: &OptimizerState) -> PyResult<()> {
        self.adamw_state = Some(
            nl_hecate_core::gpu_optimizer::GpuAdamWState::from_host(
                &state.inner, &self.params,
            )
        );
        Ok(())
    }

    /// Initialize M3 optimizer state (spec 34). Call once before step_m3.
    /// Creates GPU-resident M1/M2/V buffers + NS scratch space.
    #[pyo3(signature = (beta1=0.9, beta2=0.999, beta3=0.99, alpha=0.5, chunk_size=8, ns_iterations=5, eps=1e-8))]
    fn init_m3(&mut self, beta1: f32, beta2: f32, beta3: f32,
               alpha: f32, chunk_size: u32, ns_iterations: u32, eps: f32) -> PyResult<()> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be >= 1 (used as modulus in gpu_m3_update)"
            ));
        }
        let config = nl_hecate_core::gpu_optimizer::GpuM3Config {
            beta1, beta2, beta3, alpha, chunk_size, ns_iterations, eps,
        };
        let d = self.cfg.swa.d_model;
        self.m3_state = Some(
            nl_hecate_core::gpu_optimizer::GpuM3State::from_params(
                &self.params, config, d,
            )
        );
        Ok(())
    }

    /// Full GPU build step with M3 optimizer (spec 34). Returns (loss, grad_norm).
    /// M3 state must be initialized via init_m3() first.
    #[pyo3(signature = (input_ids, target_ids, pulse, lr, max_grad_norm=1.0))]
    fn step_m3(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
               pulse: &Pulse, lr: f32, max_grad_norm: f32) -> PyResult<(f32, f32)> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        let d = self.cfg.swa.d_model;
        if s == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("seq_len must be > 0"));
        }
        if input_ids.is_empty() || input_ids.len() % s != 0 || target_ids.len() != input_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input/target length must be batch_size * seq_len {} (got {})", s, input_ids.len())));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }

        // Forward
        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );

        // Backward
        let mut grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache, false,
        );

        // M3 update (Pulse-gated, with grad clipping)
        let state = self.m3_state.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "M3 state not initialized — call init_m3() before step_m3()")
        })?;

        // TODO: gpu_m3_update returns scalar grad_norm only; per-level gnorms
        // (for level_grad_norms() / saturation monitoring) require returning
        // Vec<f32> from the Rust side — tracked for follow-up PR.
        let grad_norm = nl_hecate_core::gpu_optimizer::gpu_m3_update(
            &mut self.params, &mut grads, state,
            &pulse.inner, lr, max_grad_norm, d,
        );

        // Weight tying: sync w_unembed^T → w_embed
        nl_hecate_core::gpu_backward::gpu_sync_embed_weights(
            &mut self.params, d, v,
        );

        // Selective periodic reset (spec 57, extends eq-006).
        self.maybe_reset_levels(&pulse.inner);

        Ok((loss, grad_norm))
    }

    /// Get the current M3 optimizer step count.
    fn m3_step(&self) -> u32 {
        self.m3_state.as_ref().map_or(0, |s| s.step)
    }

    /// Prefill: process full prompt, populate KV cache, return last-position logits.
    /// input_ids must have length == seq_len. Returns logits [vocab_size].
    fn prefill(&mut self, input_ids: Vec<usize>, pulse: &Pulse) -> PyResult<Vec<f32>> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        if input_ids.len() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input_ids length {} != seq_len {}", input_ids.len(), s)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }
        let (logits, kv_cache) = nl_hecate_core::gpu_forward::gpu_prefill_forward(
            &self.params, &self.cfg, &input_ids,
            &pulse.inner, &mut self.context,
        );
        self.kv_cache = Some(kv_cache);
        // Create decode workspace once (reused for every decode_token call)
        let d = self.cfg.swa.d_model;
        self.decode_workspace = Some(nl_hecate_core::gpu_forward::DecodeWorkspace::new(d, v));
        Ok(logits)
    }

    /// Decode one token using the KV cache. Returns logits [vocab_size].
    /// Must call prefill() first to populate the cache.
    fn decode_token(&mut self, token_id: usize, pulse: &Pulse) -> PyResult<Vec<f32>> {
        let v = self.cfg.swa.vocab_size;
        if token_id >= v {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("token_id {} >= vocab_size {}", token_id, v)));
        }
        let kv_cache = self.kv_cache.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("KV cache not initialized — call prefill() first")
        })?;
        let workspace = self.decode_workspace.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Decode workspace not initialized — call prefill() first")
        })?;
        let logits = nl_hecate_core::gpu_forward::gpu_single_token_forward(
            &self.params, &self.cfg, token_id,
            &pulse.inner, &mut self.context, kv_cache, workspace,
        );
        Ok(logits)
    }

    /// Clear the KV cache and decode workspace. Call between generation sequences.
    fn reset_cache(&mut self) {
        self.kv_cache = None;
        self.decode_workspace = None;
    }

    /// Zero all GPU memory matrices in-place (cudaMemset).
    /// Used at document boundaries to prevent stale state across documents.
    /// Also zeros fire_counts (spec 57) since M state is being hard-reset.
    fn reset_context(&mut self) {
        self.context.reset();
        // Fire counts track fires since last M reset — zeroing M means the
        // counters must restart so the next reset cycle is correctly aligned.
        self.fire_counts.iter_mut().for_each(|c| *c = 0);
    }

    /// Update per-level theta_floor values on the live model config.
    /// The new floor is applied starting from the next forward pass.
    /// Length must equal k. Used by the gate warmup schedule in loop.py.
    fn update_theta_floor(&mut self, floor: Vec<f32>) -> PyResult<()> {
        if floor.len() != self.cfg.k {
            return Err(PyValueError::new_err(format!(
                "update_theta_floor: floor length {} != k {}",
                floor.len(), self.cfg.k
            )));
        }
        self.cfg.theta_floor = floor;
        Ok(())
    }

    /// Upload context state from host (e.g., to restore after a read-only run).
    /// Also zeros fire_counts (spec 57) since the M state is being replaced.
    fn upload_context(&mut self, ctx: &ContextState) -> PyResult<()> {
        if ctx.inner.d != self.cfg.swa.d_model || ctx.inner.memory.len() != self.cfg.k {
            return Err(PyValueError::new_err(format!(
                "context shape mismatch: got k={} d={}, expected k={} d={}",
                ctx.inner.memory.len(), ctx.inner.d, self.cfg.k, self.cfg.swa.d_model
            )));
        }
        let mut new_ctx = nl_hecate_core::gpu_params::GpuContextState::from_host_context(
            &ctx.inner, self.context.batch_size,
            self.cfg.swa.num_heads, self.cfg.swa.head_dim,
        );
        // Preserve CUDA graph capture state across host-context restore.
        // from_host_context creates CudaGraphStore::new(0) (disabled), so move the live
        // store + scratch from the old context and call invalidate() to re-enter warmup.
        new_ctx.forward_scratch = self.context.forward_scratch.take();
        new_ctx.level_scratch = std::mem::take(&mut self.context.level_scratch);
        new_ctx.cuda_graph = std::mem::replace(
            &mut self.context.cuda_graph,
            nl_hecate_core::cuda_graph::CudaGraphStore::new(0),
        );
        new_ctx.cuda_graph.invalidate();
        self.context = new_ctx;
        // Fire counts must restart when M state is replaced from host.
        self.fire_counts.iter_mut().for_each(|c| *c = 0);
        Ok(())
    }

    /// Compute Frobenius norm of per-level memory matrices on GPU.
    /// Returns Vec<f32> of length k. Uses mem_dd (per-head: nh*hd*hd) for correct layout.
    fn memory_norms(&self) -> Vec<f32> {
        let mem_dd = self.context.mem_dd(); // nh * hd * hd (or d * d for nh=1)
        let mut norms = Vec::with_capacity(self.context.memory.len());
        for gpu_mem in &self.context.memory {
            // Buffer is batch_size * mem_dd; only slot 0 is the carry-forward M.
            let n = mem_dd.min(gpu_mem.len());
            let mut buf = vec![0.0f32; n];
            gpu_mem.slice(0, n).copy_to_host(&mut buf);
            let norm = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(norm);
        }
        norms
    }

    /// Compute per-head Frobenius norms of M sub-matrices on GPU.
    /// Returns Vec<Vec<f32>> of shape [k][num_heads]. Empty inner Vec for
    /// MLP rules (non-square M) or num_heads <= 1.
    ///
    /// GPU memory layout: packed per-head tiles, head h at offset h*hd*hd.
    fn memory_norms_per_head(&self) -> Vec<Vec<f32>> {
        let nh = self.context.num_heads;
        let hd = self.context.head_dim;
        let mem_dd = self.context.mem_dd(); // nh * hd * hd
        let mut result = Vec::with_capacity(self.context.memory.len());
        for gpu_mem in &self.context.memory {
            if nh <= 1 || mem_dd == 0 || gpu_mem.len() < mem_dd {
                result.push(Vec::new());
                continue;
            }
            let mut buf = vec![0.0f32; mem_dd];
            gpu_mem.slice(0, mem_dd).copy_to_host(&mut buf);
            let tile_size = hd * hd;
            let norms: Vec<f32> = (0..nh).map(|h| {
                let start = h * tile_size;
                buf[start..start + tile_size].iter()
                    .map(|v| v * v).sum::<f32>().sqrt()
            }).collect();
            result.push(norms);
        }
        result
    }

    /// Get per-level fire counts for selective reset (spec 57).
    /// Returns Vec<usize> of length k. Used for checkpoint roundtrip verification.
    fn get_fire_counts(&self) -> Vec<usize> {
        self.fire_counts.clone()
    }

    /// Restore per-level fire counts (spec 57).
    /// Used to undo fire_count side effects from roundtrip verification.
    fn set_fire_counts(&mut self, counts: Vec<usize>) -> PyResult<()> {
        if counts.len() != self.fire_counts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("fire_counts length {} != k={}", counts.len(), self.fire_counts.len())));
        }
        self.fire_counts = counts;
        Ok(())
    }

    /// Read gate biases from GPU: returns Vec of (b_alpha, b_theta, b_eta) per level.
    /// Small D2H transfer: 3 floats per level. Used for monitoring gate behavior.
    fn gate_biases(&self) -> Vec<(f32, f32, f32)> {
        self.params.levels.iter().map(|level| {
            let mut ba = [0.0f32; 1];
            let mut bt = [0.0f32; 1];
            let mut be = [0.0f32; 1];
            level.b_alpha.copy_to_host(&mut ba);
            level.b_theta.copy_to_host(&mut bt);
            level.b_eta.copy_to_host(&mut be);
            (ba[0], bt[0], be[0])
        }).collect()
    }

    /// Per-level gradient norms from the most recent step_adamw call.
    /// Computed before global gradient clipping, so values reflect the true
    /// learning signal per level (not scaled by the global clipping factor).
    /// Returns empty Vec if step_adamw has not been called yet.
    fn level_grad_norms(&self) -> Vec<f32> {
        self.last_level_gnorms.clone()
    }

    /// Run one traced forward+backward and return per-level tape diagnostics.
    ///
    /// Does NOT update weights or optimizer state — pure diagnostic read.
    /// Returns a Python dict with schema:
    ///   {"loss": float, "total_blocks": int, "levels": [
    ///     {"level": int, "opaque_key": str, "block_count": int,
    ///      "output_grad_norm": float, "dgd_delta_norm": float}, ...]}
    ///
    /// Call at eval_every intervals only. Cost is ~1 training step
    /// (traced forward + backward, no AdamW update).
    ///
    /// CS-40: tape is activated only for the duration of this call.
    /// CS-42: tape arena is dropped when this call returns.
    /// CS-32: summary is read after backward, before any weight advance.
    fn tape_forward_summary(
        &mut self,
        input_ids: Vec<usize>,
        target_ids: Vec<usize>,
        pulse: &Pulse,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        validate_tape_inputs(self.cfg.swa.seq_len, self.cfg.swa.vocab_size, &input_ids, &target_ids)?;

        // The traced forward runs on CPU — pull host copies of params and context.
        // Context state is read-only here: the tape diagnostic does not update
        // M states, so we discard the modified host context after the call.
        let host_params = self.params.to_host(&self.cfg);
        let mut host_ctx = self.context.to_host(self.cfg.k);

        let summary = nl_hecate_core::tape_summary::extract_tape_summary(
            &host_params, &self.cfg, &input_ids, &target_ids, &pulse.inner, &mut host_ctx,
        );

        let dict = PyDict::new(py);
        dict.set_item("loss", summary.loss)?;
        dict.set_item("total_blocks", summary.total_blocks)?;

        let levels_list = pyo3::types::PyList::empty(py);
        for lvl in &summary.levels {
            let ldict = PyDict::new(py);
            set_core_level_fields(
                &ldict, lvl.level, &lvl.opaque_key, lvl.block_count,
                lvl.output_grad_norm, lvl.dgd_delta_norm, lvl.m_norm,
                lvl.freq_gate_value, lvl.is_frozen, &lvl.head_m_norms,
            )?;
            levels_list.append(ldict)?;
        }
        dict.set_item("levels", levels_list)?;

        Ok(dict.into())
    }

    /// GPU-resident tape summary — fast path.
    ///
    /// Runs one GPU forward+backward (no optimizer step) and captures per-level
    /// output gradient norms from d_y_combined. Same dict schema as
    /// tape_forward_summary() but avoids the GPU→CPU param copy and CPU traced
    /// forward that make the CPU path ~10x slower.
    ///
    /// dgd_delta_norm is computed from the GPU forward cache via the
    /// dgd_delta_norm CUDA kernel (spec 16).
    fn gpu_tape_forward_summary(
        &mut self,
        input_ids: Vec<usize>,
        target_ids: Vec<usize>,
        pulse: &Pulse,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let s = self.cfg.swa.seq_len;
        validate_tape_inputs(s, self.cfg.swa.vocab_size, &input_ids, &target_ids)?;

        if self.context.batch_size > 1 {
            return Err(PyValueError::new_err(
                "gpu_tape_forward_summary currently requires batch_size == 1"
            ));
        }

        let saved_ctx = self.context.to_host(self.cfg.k);

        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );
        let grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache, true,
        );

        let post_ctx = self.context.to_host(self.cfg.k);
        self.context.upload_memory(&saved_ctx);

        let rule_name = format!("{:?}", self.cfg.memory_rule);
        let dict = PyDict::new(py);
        dict.set_item("loss", loss)?;
        let active_count: usize = (0..self.cfg.k)
            .filter(|&l| pulse.inner.active_levels[l])
            .count();
        dict.set_item("total_blocks", active_count)?;

        let d = self.cfg.swa.d_model;
        let bs = input_ids.len() / s;
        let levels_list = pyo3::types::PyList::empty(py);
        for level in 0..self.cfg.k {
            let ldict = PyDict::new(py);
            let delta_norm = cache.memory_caches[level].as_ref()
                .map(|mc| mc.dgd_delta_norm(s, d, bs, self.cfg.swa.num_heads))
                .unwrap_or(0.0);
            let m_norm = if level < post_ctx.memory.len() {
                post_ctx.memory[level].iter().map(|x| x * x).sum::<f32>().sqrt()
            } else {
                f32::NAN
            };
            // Per-head M norms from post-forward context (packed tile layout)
            let head_norms: Vec<f32> = if level < post_ctx.memory.len() {
                let mem = &post_ctx.memory[level];
                let nh = self.context.num_heads;
                let hd = self.context.head_dim;
                let tile_size = hd * hd;
                if nh > 1 && mem.len() == nh * tile_size {
                    (0..nh).map(|h| {
                        let start = h * tile_size;
                        mem[start..start + tile_size].iter()
                            .map(|v| v * v).sum::<f32>().sqrt()
                    }).collect()
                } else { Vec::new() }
            } else { Vec::new() };
            set_core_level_fields(
                &ldict, level, &rule_name,
                if pulse.inner.active_levels[level] { 1 } else { 0 },
                grads.level_output_gnorms[level], delta_norm, m_norm,
                f32::NAN, !pulse.inner.active_levels[level], &head_norms,
            )?;
            if let Some(ref mc) = cache.memory_caches[level] {
                let tc = self.cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
                if let Some(ts) = mc.theta_stats(tc) {
                    ldict.set_item("theta", make_theta_stats_dict(py, &ts)?)?;
                }
            }
            levels_list.append(ldict)?;
        }
        dict.set_item("levels", levels_list)?;

        Ok(dict.into())
    }
}

// ── FrequencyAwareAdamW (CPU) ─────────────────────────────────────────

/// Frequency-aware AdamW optimizer for the outer loop (CPU path).
///
/// Maintains per-CMS-level moment buffers with independent step counters.
/// SWA and aggregation params always update; CMS level params only update
/// when the Pulse fires for that level.
#[pyclass]
struct FrequencyAwareAdamW {
    inner: nl_hecate_core::adamw::FrequencyAwareAdamW,
}

#[pymethods]
impl FrequencyAwareAdamW {
    /// Create optimizer state from MAGParams shapes.
    #[new]
    #[pyo3(signature = (params, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1))]
    fn new(params: &MAGParams, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        let config = nl_hecate_core::adamw::AdamWConfig { beta1, beta2, eps, weight_decay };
        FrequencyAwareAdamW {
            inner: nl_hecate_core::adamw::FrequencyAwareAdamW::new(&params.inner, config),
        }
    }

    /// Pulse-gated AdamW step. SWA params always update; CMS levels only
    /// update when the Pulse fires for that level.
    ///
    /// When `max_grad_norm > 0`, clips the global gradient L2 norm in-place
    /// before applying updates. This mutates `grads` — callers who need the
    /// original gradient values should clone before calling.
    ///
    /// Returns the pre-clip gradient L2 norm (0.0 if clipping is disabled).
    #[pyo3(signature = (params, grads, pulse, lr, max_grad_norm=0.0))]
    fn step(&mut self, params: &mut MAGParams, grads: &mut MAGParams, pulse: &Pulse,
            lr: f32, max_grad_norm: f32) -> f32 {
        self.inner.step(&mut params.inner, &mut grads.inner, &pulse.inner, lr, max_grad_norm)
    }

    /// Get the SWA (global) step count.
    fn swa_step(&self) -> u32 {
        self.inner.swa_step()
    }

    /// Get the level-local step count for a CMS level.
    fn level_step(&self, level: usize) -> PyResult<u32> {
        if level >= self.inner.level_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "level {level} out of range (optimizer has {} levels)",
                self.inner.level_count()
            )));
        }
        Ok(self.inner.level_step(level))
    }
}

// ── GpuStackedModel (multi-block architecture) ──────────────────────

/// GPU-resident stacked multi-block model.
/// N blocks of [SWA + CMS(k levels)] connected via residual stream.
/// Shared embedding/unembedding + final LayerNorm across all blocks.
#[cfg(feature = "cuda")]
#[pyclass(unsendable)]
struct GpuStackedModel {
    params: nl_hecate_core::gpu_params::GpuStackedParams,
    context: nl_hecate_core::gpu_params::GpuStackedContext,
    cfg: nl_hecate_core::model::MAGConfig,
    adamw_state: Option<nl_hecate_core::gpu_stacked_optimizer::GpuStackedAdamWState>,
    m3_state: Option<nl_hecate_core::gpu_stacked_optimizer::GpuStackedM3State>,
    n_blocks: usize,
    /// Per-level reset intervals for selective periodic reset (spec 57).
    /// Empty vec = no reset. [1,1,1,1] = spec-08 all-ones. [1,8,64,512] = gear-shifting.
    reset_intervals: Vec<usize>,
    /// Per-level fire counters for selective periodic reset (spec 57).
    fire_counts: Vec<usize>,
    /// Per-block aggregate gradient norms from the most recent step_adamw call (spec 23).
    last_block_gnorms: Vec<f32>,
    /// Per-block L0-only gradient norms from the most recent step_adamw call (spec 23).
    last_l0_block_gnorms: Vec<f32>,
    /// GPU profiler for per-step heatmap (spec 38). None when disabled.
    profiler: Option<nl_hecate_core::gpu_profiler::GpuProfiler>,
    /// Per-block KV caches for single-token decode (spec 47).
    kv_caches: Option<Vec<nl_hecate_core::gpu_forward::GpuKVCache>>,
    /// Pre-allocated workspace for single-token decode (spec 47).
    decode_workspace: Option<nl_hecate_core::gpu_stacked_forward::StackedDecodeWorkspace>,
}

#[cfg(feature = "cuda")]
impl GpuStackedModel {
    /// Selective periodic reset (spec 57): for each active level, increment fire
    /// counter and reset M when counter reaches the level's reset interval.
    /// CS-32: called after observe, before next advance.
    fn maybe_reset_levels(&mut self, pulse: &nl_hecate_core::conductor::Pulse) {
        if self.reset_intervals.is_empty() { return; }
        for (k, &active) in pulse.active_levels.iter().enumerate() {
            if !active { continue; }
            self.fire_counts[k] += 1;
            if self.fire_counts[k] >= self.reset_intervals[k] {
                self.context.periodic_reset_level(k);
                self.fire_counts[k] = 0;
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuStackedModel {
    /// Create from StackedMAGParams config.
    #[new]
    #[pyo3(signature = (cfg, n_blocks, seed, batch_size=1, memory_reset=false, reset_intervals=None))]
    fn new(
        cfg: &MAGConfig, n_blocks: usize, seed: u64, batch_size: usize,
        memory_reset: bool, reset_intervals: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("batch_size must be >= 1"));
        }
        if n_blocks == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n_blocks must be >= 1"));
        }
        let k = cfg.inner.k;
        let intervals = GpuModel::resolve_intervals(memory_reset, reset_intervals, k)?;
        let host_params = nl_hecate_core::stacked_model::StackedMAGParams::init(
            &cfg.inner, n_blocks, seed,
        );
        let gpu_params = nl_hecate_core::gpu_params::GpuStackedParams::from_host(&host_params);
        let gpu_context = nl_hecate_core::gpu_params::GpuStackedContext::new(
            n_blocks, cfg.inner.k, cfg.inner.swa.d_model, batch_size,
            Some(&cfg.inner),
        );
        Ok(GpuStackedModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            m3_state: None,
            n_blocks,
            reset_intervals: intervals,
            fire_counts: vec![0usize; k],
            last_block_gnorms: Vec::new(),
            last_l0_block_gnorms: Vec::new(),
            profiler: None,
            kv_caches: None,
            decode_workspace: None,
        })
    }

    /// Create from serialized StackedMAGParams JSON (loaded from checkpoint).
    /// Used by load_stacked_checkpoint + extend_stacked_push_up.
    #[staticmethod]
    #[pyo3(signature = (params_json, cfg, n_blocks, batch_size=1, memory_reset=false, reset_intervals=None))]
    fn from_params_json(
        params_json: &str, cfg: &MAGConfig, n_blocks: usize,
        batch_size: usize, memory_reset: bool, reset_intervals: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("batch_size must be >= 1"));
        }
        if n_blocks == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("n_blocks must be >= 1"));
        }
        let k = cfg.inner.k;
        let intervals = GpuModel::resolve_intervals(memory_reset, reset_intervals, k)?;
        let host_params: nl_hecate_core::stacked_model::StackedMAGParams =
            serde_json::from_str(params_json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                    format!("from_params_json: failed to deserialize: {e}")))?;
        if host_params.blocks.len() != n_blocks {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("from_params_json: params JSON has {} blocks but n_blocks={}",
                        host_params.blocks.len(), n_blocks)));
        }
        for (i, block) in host_params.blocks.iter().enumerate() {
            if block.levels.len() != cfg.inner.k {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("from_params_json: block {} has k={} but cfg.k={}",
                            i, block.levels.len(), cfg.inner.k)));
            }
        }
        let gpu_params = nl_hecate_core::gpu_params::GpuStackedParams::from_host(&host_params);
        let gpu_context = nl_hecate_core::gpu_params::GpuStackedContext::new(
            n_blocks, cfg.inner.k, cfg.inner.swa.d_model, batch_size,
            Some(&cfg.inner),
        );
        Ok(GpuStackedModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            m3_state: None,
            n_blocks,
            reset_intervals: intervals,
            fire_counts: vec![0usize; k],
            last_block_gnorms: Vec::new(),
            last_l0_block_gnorms: Vec::new(),
            profiler: None,
            kv_caches: None,
            decode_workspace: None,
        })
    }

    /// Full GPU build step with AdamW optimizer. Returns (loss, grad_norm).
    #[pyo3(signature = (input_ids, target_ids, pulse, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, max_grad_norm=1.0, collect_block_gnorms=false, freeze_embed=false))]
    fn step_adamw(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
                  pulse: &Pulse, lr: f32, beta1: f32, beta2: f32,
                  eps: f32, weight_decay: f32, max_grad_norm: f32,
                  collect_block_gnorms: bool, freeze_embed: bool) -> PyResult<(f32, f32)> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        if s == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("seq_len must be > 0"));
        }
        if input_ids.is_empty() || input_ids.len() % s != 0 || target_ids.len() != input_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input/target length must be batch_size * seq_len {} (got {})", s, input_ids.len())));
        }
        let bs = input_ids.len() / s;
        if bs != self.context.batch_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("batch mismatch: input has {} samples but context allocated for batch_size={}",
                        bs, self.context.batch_size)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }

        // Start profiler step timer (no-op if profiling disabled)
        if let Some(ref prof) = self.profiler { prof.step_start(); }

        // Forward
        let (loss, cache) = nl_hecate_core::gpu_stacked_forward::gpu_stacked_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context, &mut self.profiler,
        );

        // Backward
        let mut grads = nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward(
            &self.params, &self.cfg, &cache, &mut self.profiler,
        );

        // Lazy-init AdamW state
        if self.adamw_state.is_none() {
            self.adamw_state = Some(
                nl_hecate_core::gpu_stacked_optimizer::GpuStackedAdamWState::from_params(&self.params)
            );
        }
        let state = self.adamw_state.as_mut().unwrap();

        // Per-block gradient norms (spec 23) — computed before clipping
        if collect_block_gnorms {
            let per_block = nl_hecate_core::gpu_stacked_optimizer::gpu_stacked_per_block_grad_norms(&grads, state);
            self.last_block_gnorms = per_block.block_norms;
            self.last_l0_block_gnorms = per_block.l0_block_norms;
        } else {
            self.last_block_gnorms.clear();
            self.last_l0_block_gnorms.clear();
        }

        // AdamW update (grads passed as &mut for in-place clipping)
        let grad_norm = nl_hecate_core::gpu_stacked_optimizer::gpu_stacked_adamw_update(
            &mut self.params, &mut grads, state,
            &pulse.inner,
            lr, beta1, beta2, eps, weight_decay, max_grad_norm,
            freeze_embed, &mut self.profiler,
        );

        // Stop profiler step timer (no-op if profiling disabled)
        if let Some(ref prof) = self.profiler { prof.step_stop(); }

        // Weight tying: sync w_unembed^T → w_embed (same as single-block path).
        // Without this, shared input/output embeddings diverge during training.
        nl_hecate_core::gpu_stacked_optimizer::gpu_stacked_sync_embed_weights(
            &mut self.params,
            self.cfg.swa.d_model,
            self.cfg.swa.vocab_size,
        );

        // Update M-norm tracking for dormancy detection (spec 28).
        self.context.update_m_norm_tracking();

        // Selective periodic reset (spec 57, extends eq-006).
        // CS-32 compliant: reset happens after observe, before next advance.
        self.maybe_reset_levels(&pulse.inner);

        Ok((loss, grad_norm))
    }

    /// Initialize M3 optimizer state for stacked model. Call once before step_m3.
    #[pyo3(signature = (beta1=0.9, beta2=0.999, beta3=0.99, alpha=0.5, chunk_size=8, ns_iterations=5, eps=1e-8))]
    fn init_m3(&mut self, beta1: f32, beta2: f32, beta3: f32,
               alpha: f32, chunk_size: u32, ns_iterations: u32, eps: f32) -> PyResult<()> {
        if chunk_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "chunk_size must be >= 1 (used as modulus in gpu_stacked_m3_update)"
            ));
        }
        let config = nl_hecate_core::gpu_optimizer::GpuM3Config {
            beta1, beta2, beta3, alpha, chunk_size, ns_iterations, eps,
        };
        let d = self.cfg.swa.d_model;
        self.m3_state = Some(
            nl_hecate_core::gpu_stacked_optimizer::GpuStackedM3State::from_params(
                &self.params, config, d,
            )
        );
        Ok(())
    }

    /// Full GPU build step with M3 optimizer for stacked model. Returns (loss, grad_norm).
    #[pyo3(signature = (input_ids, target_ids, pulse, lr, max_grad_norm=1.0, freeze_embed=false))]
    fn step_m3(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
               pulse: &Pulse, lr: f32, max_grad_norm: f32,
               freeze_embed: bool) -> PyResult<(f32, f32)> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        let d = self.cfg.swa.d_model;
        if s == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("seq_len must be > 0"));
        }
        if input_ids.is_empty() || input_ids.len() % s != 0 || target_ids.len() != input_ids.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input/target length must be batch_size * seq_len {} (got {})", s, input_ids.len())));
        }
        let bs = input_ids.len() / s;
        if bs != self.context.batch_size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("batch mismatch: input has {} samples but context allocated for batch_size={}",
                        bs, self.context.batch_size)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }

        // Fail fast if M3 not initialized (before expensive forward/backward)
        if self.m3_state.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "M3 state not initialized — call init_m3() before step_m3()"));
        }

        // Start profiler step timer (no-op if profiling disabled)
        if let Some(ref prof) = self.profiler { prof.step_start(); }

        // Forward
        let (loss, cache) = nl_hecate_core::gpu_stacked_forward::gpu_stacked_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context, &mut self.profiler,
        );

        // Backward
        let mut grads = nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward(
            &self.params, &self.cfg, &cache, &mut self.profiler,
        );

        // M3 update (state guaranteed Some by guard above)
        let state = self.m3_state.as_mut().unwrap();

        let grad_norm = nl_hecate_core::gpu_stacked_optimizer::gpu_stacked_m3_update(
            &mut self.params, &mut grads, state,
            &pulse.inner, lr, max_grad_norm, d, freeze_embed,
            &mut self.profiler,
        );

        // Stop profiler step timer (no-op if profiling disabled)
        if let Some(ref prof) = self.profiler { prof.step_stop(); }

        // Weight tying: sync w_unembed^T → w_embed
        nl_hecate_core::gpu_stacked_optimizer::gpu_stacked_sync_embed_weights(
            &mut self.params, d, v,
        );

        // Update M-norm tracking for dormancy detection (spec 28)
        self.context.update_m_norm_tracking();

        // Selective periodic reset (spec 57, CS-32 compliant)
        self.maybe_reset_levels(&pulse.inner);

        Ok((loss, grad_norm))
    }

    /// Get the current M3 optimizer step count.
    fn m3_step(&self) -> u32 {
        self.m3_state.as_ref().map_or(0, |s| s.step)
    }

    /// Update per-level theta_floor values on the live model config.
    /// The new floor is applied starting from the next forward pass.
    /// Length must equal k. Used by the gate warmup schedule in loop.py.
    fn update_theta_floor(&mut self, floor: Vec<f32>) -> PyResult<()> {
        if floor.len() != self.cfg.k {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "update_theta_floor: floor length {} != k {}",
                floor.len(), self.cfg.k
            )));
        }
        self.cfg.theta_floor = floor;
        Ok(())
    }

    /// Total parameter count (downloads to host for counting -- use sparingly).
    fn total_params(&self) -> usize {
        let d = self.cfg.swa.d_model;
        let v = self.cfg.swa.vocab_size;
        let k = self.cfg.k;
        let host = self.params.to_host(d, v, k);
        host.num_params()
    }

    /// Reset all memory across all blocks.
    /// Also zeros fire_counts (spec 57) since M state is being hard-reset.
    fn reset_context(&mut self) {
        self.context.reset();
        self.fire_counts.iter_mut().for_each(|c| *c = 0);
    }

    /// Prefill: process full prompt through all blocks, populate per-block KV caches,
    /// return last-position logits [vocab_size]. Call once before decode_token loop.
    /// Spec: specs/infrastructure/47_single_token_decode.md
    ///
    /// NOTE: Does NOT call maybe_reset_levels(). Selective periodic reset (spec 57)
    /// is an optimizer-path concern (step_adamw/step_m3). Inference paths observe M
    /// but don't own the reset cycle. CS-11: no training loop logic in memory rules.
    fn prefill(&mut self, input_ids: Vec<usize>, pulse: &Pulse) -> PyResult<Vec<f32>> {
        let s = self.cfg.swa.seq_len;
        let v = self.cfg.swa.vocab_size;
        let d = self.cfg.swa.d_model;
        if self.context.batch_size != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("prefill/decode_token only supports batch_size=1 (got {})", self.context.batch_size)));
        }
        if input_ids.len() != s {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("input_ids length {} != seq_len {}", input_ids.len(), s)));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id >= v {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("input_ids contains {} >= vocab_size {}", max_id, v)));
            }
        }
        let mut kv_caches = Vec::new();
        let logits = nl_hecate_core::gpu_stacked_forward::gpu_stacked_prefill(
            &self.params, &self.cfg, &input_ids,
            &pulse.inner, &mut self.context, &mut kv_caches, &mut self.profiler,
        );
        self.kv_caches = Some(kv_caches);
        self.decode_workspace = Some(
            nl_hecate_core::gpu_stacked_forward::StackedDecodeWorkspace::new(self.n_blocks, d, v),
        );
        Ok(logits)
    }

    /// Decode one token using per-block KV caches + M state. Returns logits [vocab_size].
    /// Must call prefill() first. O(d^2) per token, not O(seq_len * d^2).
    /// Spec: specs/infrastructure/47_single_token_decode.md
    ///
    /// NOTE: Does NOT call maybe_reset_levels(). See prefill() note — inference paths
    /// don't own the reset cycle. CS-11 compliant.
    fn decode_token(&mut self, token_id: usize, pulse: &Pulse) -> PyResult<Vec<f32>> {
        let v = self.cfg.swa.vocab_size;
        if token_id >= v {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("token_id {} >= vocab_size {}", token_id, v)));
        }
        let kv_caches = self.kv_caches.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("KV caches not initialized — call prefill() first")
        })?;
        let workspace = self.decode_workspace.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Decode workspace not initialized — call prefill() first")
        })?;
        let logits = nl_hecate_core::gpu_stacked_forward::gpu_stacked_decode_token(
            &self.params, &self.cfg, token_id,
            &pulse.inner, &mut self.context, kv_caches, workspace,
        );
        Ok(logits)
    }

    /// Clear KV caches and decode workspace. Call between generation sequences.
    fn reset_cache(&mut self) {
        self.kv_caches = None;
        self.decode_workspace = None;
    }

    /// Per-(block, level) M norm deltas from the most recent step (spec 28).
    fn m_norm_deltas(&self) -> Vec<Vec<f32>> {
        self.context.m_norm_deltas.clone()
    }

    /// Per-(block, level) dormancy status: "active", "warning", "dormant" (spec 28).
    fn dormancy_status(&self) -> Vec<Vec<String>> {
        self.context.dormancy_status()
    }

    /// Configure dormancy detection thresholds (spec 28).
    /// floors: per-level M-diff floor (length must equal k, or empty to disable).
    /// consecutive: number of consecutive below-floor steps to trigger "dormant".
    fn set_dormancy_config(&mut self, floors: Vec<f32>, consecutive: usize) -> PyResult<()> {
        if !floors.is_empty() && floors.len() != self.cfg.k {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dormancy_floors length {} != k {}", floors.len(), self.cfg.k
            )));
        }
        self.context.set_dormancy_config(floors, consecutive);
        Ok(())
    }

    /// Number of blocks.
    #[getter]
    fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Per-block aggregate gradient norms from the most recent step_adamw call
    /// (spec 23). Includes SWA + all levels per block. Computed before global
    /// gradient clipping. Returns empty Vec if collect_block_gnorms was false.
    fn block_grad_norms(&self) -> Vec<f32> {
        self.last_block_gnorms.clone()
    }

    /// Per-block L0-only gradient norms from the most recent step_adamw call
    /// (spec 23). Only level[0] memory params per block — used for the
    /// "L0 gnorm per block > 0.01" floor check (spec 19 promotion criteria).
    fn l0_block_grad_norms(&self) -> Vec<f32> {
        self.last_l0_block_gnorms.clone()
    }

    /// Return cached per-level Frobenius norms of M matrices.
    /// These are captured by update_m_norm_tracking() BEFORE periodic reset
    /// zeros the buffers, so they reflect the actual end-of-step M state.
    /// Returns Vec<f32> of length k (max across blocks per level).
    fn memory_norms(&self) -> Vec<f32> {
        let k = self.cfg.k;
        let mut level_norms = vec![0.0f32; k];
        for bn in &self.context.prev_m_norms {
            for (level, &norm) in bn.iter().enumerate() {
                if level < k && norm > level_norms[level] {
                    level_norms[level] = norm;
                }
            }
        }
        level_norms
    }

    /// Return cached per-head Frobenius norms of M sub-matrices.
    /// These are captured by update_m_norm_tracking() BEFORE periodic reset
    /// zeros the buffers. Returns Vec<Vec<f32>> of shape [k][num_heads]
    /// (max across blocks per level). Empty inner Vec for MLP rules or num_heads <= 1.
    fn memory_norms_per_head(&self) -> Vec<Vec<f32>> {
        let k = self.cfg.k;
        let nh = self.cfg.swa.num_heads;
        let mut result: Vec<Vec<f32>> = (0..k).map(|_| vec![0.0f32; nh]).collect();
        let mut has_data = false;
        for bhn in &self.context.cached_per_head_norms {
            for (level, heads) in bhn.iter().enumerate() {
                if level < k && !heads.is_empty() {
                    has_data = true;
                    for (h, &norm) in heads.iter().enumerate() {
                        if h < nh && norm > result[level][h] {
                            result[level][h] = norm;
                        }
                    }
                }
            }
        }
        if !has_data {
            return (0..k).map(|_| Vec::new()).collect();
        }
        result
    }

    /// Get per-level fire counts for selective reset (spec 57).
    fn get_fire_counts(&self) -> Vec<usize> {
        self.fire_counts.clone()
    }

    /// Restore per-level fire counts (spec 57).
    fn set_fire_counts(&mut self, counts: Vec<usize>) -> PyResult<()> {
        if counts.len() != self.fire_counts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("fire_counts length {} != k={}", counts.len(), self.fire_counts.len())));
        }
        self.fire_counts = counts;
        Ok(())
    }

    /// Read live GPU M norms (post-reset). Unlike memory_norms() which returns
    /// cached pre-reset values from update_m_norm_tracking(), this reads the
    /// current GPU memory state. Use for verifying that reset actually zeroed M.
    /// Returns Vec<f32> of length k (max across blocks per level).
    fn memory_norms_live(&self) -> Vec<f32> {
        let k = self.cfg.k;
        let mut level_norms = vec![0.0f32; k];
        for bn in self.context.memory_norms() {
            for (level, norm) in bn.iter().enumerate() {
                if level < k && *norm > level_norms[level] {
                    level_norms[level] = *norm;
                }
            }
        }
        level_norms
    }

    /// Read gate biases from GPU: returns Vec of (b_alpha, b_theta, b_eta) per level.
    /// Uses block 0's parameters (all blocks share the same level params in stacked model).
    fn gate_biases(&self) -> Vec<(f32, f32, f32)> {
        self.params.blocks[0].levels.iter().map(|level| {
            let mut ba = [0.0f32; 1];
            let mut bt = [0.0f32; 1];
            let mut be = [0.0f32; 1];
            level.b_alpha.copy_to_host(&mut ba);
            level.b_theta.copy_to_host(&mut bt);
            level.b_eta.copy_to_host(&mut be);
            (ba[0], bt[0], be[0])
        }).collect()
    }

    /// Get the current tape_multiplier value (spec 25: CMS cycles of cache to retain).
    #[getter]
    fn tape_multiplier(&self) -> usize {
        self.cfg.tape_multiplier
    }

    /// Set tape_multiplier at runtime — takes effect on the next forward call.
    /// Controls how many CMS cycles of cache to retain (spec 25).
    /// 1 = one cycle (default, minimum for backward). N = N cycles.
    #[setter]
    fn set_tape_multiplier(&mut self, value: usize) {
        self.cfg.tape_multiplier = value.max(1);
    }

    /// CPU Wengert tape summary for stacked models — full gradient observability.
    ///
    /// Downloads params + context to host, runs traced forward+backward on CPU,
    /// returns per-(block, level) diagnostics. Slower than GPU path but gives
    /// full tape visibility for debugging. Context is not modified.
    fn cpu_stacked_tape_summary(
        &mut self,
        input_ids: Vec<usize>,
        target_ids: Vec<usize>,
        pulse: &Pulse,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let d = self.cfg.swa.d_model;
        let k = self.cfg.k;
        validate_tape_inputs(self.cfg.swa.seq_len, self.cfg.swa.vocab_size, &input_ids, &target_ids)?;

        let host_params = self.params.to_host(d, self.cfg.swa.vocab_size, k);
        let mut host_ctx: Vec<Vec<Vec<f32>>> = self.context.blocks.iter()
            .map(|gpu_ctx| gpu_ctx.to_host(k).memory)
            .collect();

        let summary = nl_hecate_core::tape_summary::extract_stacked_tape_summary(
            &host_params, &self.cfg, &input_ids, &target_ids, &pulse.inner, &mut host_ctx,
        );

        let dict = PyDict::new(py);
        dict.set_item("loss", summary.loss)?;
        dict.set_item("n_blocks", summary.n_blocks)?;

        let blocks_list = pyo3::types::PyList::empty(py);
        let mut agg_gnorms = vec![0.0f32; k];
        let mut agg_deltas = vec![0.0f32; k];

        for block_sum in &summary.blocks {
            let bdict = PyDict::new(py);
            bdict.set_item("block_index", block_sum.block)?;
            let levels_list = pyo3::types::PyList::empty(py);
            for lvl in &block_sum.levels {
                let ldict = PyDict::new(py);
                set_core_level_fields(
                    &ldict, lvl.level, &lvl.opaque_key, lvl.block_count,
                    lvl.output_grad_norm, lvl.dgd_delta_norm, lvl.m_norm,
                    lvl.freq_gate_value, lvl.is_frozen, &lvl.head_m_norms,
                )?;
                levels_list.append(ldict)?;
                if lvl.output_grad_norm > agg_gnorms[lvl.level] {
                    agg_gnorms[lvl.level] = lvl.output_grad_norm;
                }
                if lvl.dgd_delta_norm > agg_deltas[lvl.level] {
                    agg_deltas[lvl.level] = lvl.dgd_delta_norm;
                }
            }
            bdict.set_item("levels", levels_list)?;
            blocks_list.append(bdict)?;
        }
        dict.set_item("blocks", blocks_list)?;

        // Aggregated levels (backward compat with print_tape_summary)
        let rule_name = format!("{:?}", self.cfg.memory_rule);
        let mut total_active = 0usize;
        let agg_levels_list = pyo3::types::PyList::empty(py);
        for level in 0..k {
            let ldict = PyDict::new(py);
            let active = pulse.inner.active_levels[level];
            let bc = if active { self.n_blocks } else { 0 };
            if active { total_active += self.n_blocks; }
            let max_mnorm = host_ctx.iter().map(|block_ctx| {
                if level < block_ctx.len() {
                    block_ctx[level].iter().map(|x| x * x).sum::<f32>().sqrt()
                } else { 0.0 }
            }).fold(0.0f32, f32::max);
            // Aggregate head_m_norms: empty at aggregate level (per-head data on block dicts)
            let empty_heads: Vec<f32> = Vec::new();
            set_core_level_fields(
                &ldict, level, &rule_name, bc, agg_gnorms[level], agg_deltas[level],
                max_mnorm, f32::NAN, !active, &empty_heads,
            )?;
            agg_levels_list.append(ldict)?;
        }
        dict.set_item("levels", agg_levels_list)?;
        dict.set_item("total_blocks", total_active)?;

        Ok(dict.into())
    }

    /// Enable profiling for the next step. Creates a GpuProfiler that wraps
    /// cudaEvent pairs around every pipeline region (spec 38).
    fn enable_profiling(&mut self) {
        self.profiler = Some(nl_hecate_core::gpu_profiler::GpuProfiler::new(true));
    }

    /// Disable profiling and drop the GpuProfiler (frees cudaEvents).
    fn disable_profiling(&mut self) {
        self.profiler = None;
    }

    /// Whether profiling is currently enabled.
    fn is_profiling(&self) -> bool {
        self.profiler.is_some()
    }

    /// Collect the step profile after a profiled step completes.
    /// Returns a Python dict with total_ms, tok/s, by_category, top components, per_block.
    /// Resets the profiler for the next step.
    fn collect_profile(&mut self, py: pyo3::Python<'_>, seq_len: usize) -> PyResult<pyo3::PyObject> {
        let prof = match self.profiler.as_mut() {
            Some(p) => p,
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "profiling not enabled — call enable_profiling() first")),
        };

        let profile = prof.collect(self.n_blocks);
        prof.reset();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_ms", profile.total_ms)?;
        let toks_per_sec = if profile.total_ms > 0.0 {
            seq_len as f32 / (profile.total_ms / 1000.0)
        } else { 0.0 };
        dict.set_item("tok_per_s", toks_per_sec)?;

        // by_category: {name: {ms, pct}}
        let cat_dict = pyo3::types::PyDict::new(py);
        for ct in &profile.by_category {
            let entry = pyo3::types::PyDict::new(py);
            entry.set_item("ms", ct.ms)?;
            entry.set_item("pct", ct.pct)?;
            cat_dict.set_item(ct.category.as_str(), entry)?;
        }
        dict.set_item("by_category", cat_dict)?;

        // top components: [{name, block, level, ms, pct}]
        let top: Vec<&nl_hecate_core::gpu_profiler::ComponentTiming> =
            profile.components.iter().take(20).collect();
        let top_list = pyo3::types::PyList::empty(py);
        for c in &top {
            let entry = pyo3::types::PyDict::new(py);
            entry.set_item("name", &c.name)?;
            entry.set_item("block", c.block_idx)?;
            entry.set_item("level", c.level_idx)?;
            entry.set_item("ms", c.ms)?;
            entry.set_item("pct", c.pct)?;
            top_list.append(entry)?;
        }
        dict.set_item("top_components", top_list)?;

        // per_block: [{block, fwd_ms, bwd_ms, opt_ms}]
        let block_list = pyo3::types::PyList::empty(py);
        for bt in &profile.per_block {
            let entry = pyo3::types::PyDict::new(py);
            entry.set_item("block", bt.block_idx)?;
            entry.set_item("fwd_ms", bt.fwd_ms)?;
            entry.set_item("bwd_ms", bt.bwd_ms)?;
            entry.set_item("opt_ms", bt.opt_ms)?;
            block_list.append(entry)?;
        }
        dict.set_item("per_block", block_list)?;

        Ok(dict.into())
    }

    /// GPU-resident stacked tape summary -- per-(block, level) gradient diagnostics.
    ///
    /// Runs one GPU forward+backward (no optimizer step) and captures per-level
    /// output gradient norms from d_y_combined in each block. Context is
    /// saved and restored -- diagnostic does not modify model state.
    fn gpu_stacked_tape_summary(
        &mut self,
        input_ids: Vec<usize>,
        target_ids: Vec<usize>,
        pulse: &Pulse,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let s = self.cfg.swa.seq_len;
        validate_tape_inputs(s, self.cfg.swa.vocab_size, &input_ids, &target_ids)?;
        let bs = input_ids.len() / s;
        if bs != self.context.batch_size {
            return Err(PyValueError::new_err(
                format!("batch mismatch: input has {} samples but context allocated for batch_size={}",
                        bs, self.context.batch_size)));
        }

        let k = self.cfg.k;
        let d = self.cfg.swa.d_model;
        let m_norms = self.context.memory_norms();
        let saved_ctx = self.context.deep_clone();

        let mut no_prof: Option<nl_hecate_core::gpu_profiler::GpuProfiler> = None;
        let (loss, cache) = nl_hecate_core::gpu_stacked_forward::gpu_stacked_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context, &mut no_prof,
        );
        let m_norms_post = self.context.memory_norms();
        // Per-head M norms (spec 50).
        let head_norms_post = self.context.memory_norms_per_head();

        // Extract DGD delta norms + gate stats from cache BEFORE backward consumes it
        let mut delta_norms: Vec<Vec<f32>> = Vec::with_capacity(self.n_blocks);
        let mut theta_stats_grid: Vec<Vec<Option<nl_hecate_core::gpu_forward::ThetaStats>>> =
            Vec::with_capacity(self.n_blocks);
        let mut alpha_stats_grid: Vec<Vec<Option<nl_hecate_core::gpu_forward::GateStats>>> =
            Vec::with_capacity(self.n_blocks);
        let mut eta_stats_grid: Vec<Vec<Option<nl_hecate_core::gpu_forward::GateStats>>> =
            Vec::with_capacity(self.n_blocks);
        for block_cache in &cache.block_caches {
            let mut block_deltas = Vec::with_capacity(k);
            let mut block_theta = Vec::with_capacity(k);
            let mut block_alpha = Vec::with_capacity(k);
            let mut block_eta = Vec::with_capacity(k);
            for level in 0..k {
                if let Some(ref mem_cache) = block_cache.memory_caches[level] {
                    block_deltas.push(mem_cache.dgd_delta_norm(s, d, bs, self.cfg.swa.num_heads));
                    let tc = self.cfg.theta_ceil.get(level).copied().unwrap_or(f32::MAX);
                    block_theta.push(mem_cache.theta_stats(tc));
                    let af = self.cfg.alpha_floor.get(level).copied().unwrap_or(0.0);
                    block_alpha.push(mem_cache.alpha_stats(af));
                    block_eta.push(mem_cache.eta_stats());
                } else {
                    block_deltas.push(0.0);
                    block_theta.push(None);
                    block_alpha.push(None);
                    block_eta.push(None);
                }
            }
            delta_norms.push(block_deltas);
            theta_stats_grid.push(block_theta);
            alpha_stats_grid.push(block_alpha);
            eta_stats_grid.push(block_eta);
        }

        let grads = nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward(
            &self.params, &self.cfg, &cache, &mut no_prof,
        );
        self.context = saved_ctx;

        let dict = PyDict::new(py);
        dict.set_item("loss", loss)?;
        dict.set_item("n_blocks", self.n_blocks)?;

        let rule_name = format!("{:?}", self.cfg.memory_rule);
        let dormancy_status = self.context.dormancy_status();
        let dormancy_below = &self.context.dormancy_below_count;

        // Per-block breakdown
        let blocks_list = pyo3::types::PyList::empty(py);
        let mut agg_gnorms = vec![0.0f32; k];

        for (bi, bg) in grads.blocks.iter().enumerate() {
            let bdict = PyDict::new(py);
            bdict.set_item("block_index", bi)?;

            let levels_list = pyo3::types::PyList::empty(py);
            for level in 0..k {
                let ldict = PyDict::new(py);
                let active = pulse.inner.active_levels[level];
                let gnorm = bg.level_output_gnorms[level];
                // Per-head M norms (spec 50)
                let empty_head_norms: Vec<f32> = Vec::new();
                let head_norms: &Vec<f32> = head_norms_post.get(bi)
                    .and_then(|bl| bl.get(level))
                    .unwrap_or(&empty_head_norms);
                set_core_level_fields(
                    &ldict, level, &rule_name,
                    if active { 1 } else { 0 },
                    gnorm, delta_norms[bi][level], m_norms_post[bi][level],
                    f32::NAN, !active, head_norms,
                )?;
                // Shard M-diff (spec 28)
                let m_diff_abs = (m_norms_post[bi][level] - m_norms[bi][level]).abs();
                let m_diff_rel = if m_norms[bi][level] > 1e-8 {
                    m_diff_abs / m_norms[bi][level]
                } else { 0.0 };
                ldict.set_item("m_shard_diff", m_diff_abs)?;
                ldict.set_item("m_shard_diff_relative", m_diff_rel)?;
                // Dormancy
                if bi < dormancy_status.len() && level < dormancy_status[bi].len() {
                    ldict.set_item("dormancy_status", &dormancy_status[bi][level])?;
                    ldict.set_item("dormancy_below_count", dormancy_below[bi][level])?;
                }
                // Gate stats
                if let Some(ref as_) = alpha_stats_grid[bi][level] {
                    ldict.set_item("alpha", make_gate_stats_dict(py, as_, "frac_at_floor")?)?;
                }
                if let Some(ref ts) = theta_stats_grid[bi][level] {
                    ldict.set_item("theta", make_theta_stats_dict(py, ts)?)?;
                }
                if let Some(ref es) = eta_stats_grid[bi][level] {
                    ldict.set_item("eta", make_gate_stats_dict(py, es, "frac_at_bound")?)?;
                }
                levels_list.append(ldict)?;
                if gnorm > agg_gnorms[level] { agg_gnorms[level] = gnorm; }
            }
            bdict.set_item("levels", levels_list)?;
            blocks_list.append(bdict)?;
        }
        dict.set_item("blocks", blocks_list)?;

        // Aggregated levels (backward compat with print_tape_summary)
        let mut total_active = 0usize;
        let agg_levels_list = pyo3::types::PyList::empty(py);
        for level in 0..k {
            let ldict = PyDict::new(py);
            let active = pulse.inner.active_levels[level];
            let bc = if active { self.n_blocks } else { 0 };
            if active { total_active += self.n_blocks; }
            let max_delta = delta_norms.iter().map(|bd| bd[level]).fold(0.0f32, f32::max);
            let max_mnorm = m_norms_post.iter().map(|bn| bn[level]).fold(0.0f32, f32::max);
            // Aggregate per-head M norms: max across blocks per head
            let nh = self.cfg.swa.num_heads;
            let agg_heads: Vec<f32> = if nh > 1 {
                let mut agg = vec![0.0f32; nh];
                for bhn in &head_norms_post {
                    if let Some(heads) = bhn.get(level) {
                        for (h, &n) in heads.iter().enumerate() {
                            if h < nh && n > agg[h] { agg[h] = n; }
                        }
                    }
                }
                agg
            } else {
                Vec::new()
            };
            set_core_level_fields(
                &ldict, level, &rule_name, bc, agg_gnorms[level], max_delta,
                max_mnorm, f32::NAN, !active, &agg_heads,
            )?;
            // Aggregate shard M-diff (spec 28)
            let max_diff_abs = (0..self.n_blocks).map(|bi| {
                (m_norms_post[bi][level] - m_norms[bi][level]).abs()
            }).fold(0.0f32, f32::max);
            let max_diff_rel = (0..self.n_blocks).map(|bi| {
                let pre = m_norms[bi][level];
                if pre > 1e-8 { (m_norms_post[bi][level] - pre).abs() / pre } else { 0.0 }
            }).fold(0.0f32, f32::max);
            ldict.set_item("m_shard_diff", max_diff_abs)?;
            ldict.set_item("m_shard_diff_relative", max_diff_rel)?;
            // Worst dormancy
            let mut worst_rank = 0u8;
            for bi in 0..self.n_blocks {
                if bi < dormancy_status.len() && level < dormancy_status[bi].len() {
                    let rank = match dormancy_status[bi][level].as_str() {
                        "dormant" => 2, "warning" => 1, _ => 0,
                    };
                    if rank > worst_rank { worst_rank = rank; }
                }
            }
            ldict.set_item("dormancy_status",
                match worst_rank { 2 => "dormant", 1 => "warning", _ => "active" })?;
            // Aggregate gate stats across blocks
            aggregate_gate_stats(py, &ldict, &theta_stats_grid, level, self.n_blocks,
                |ts| (ts.count, ts.min, ts.max, ts.mean, ts.p99, ts.frac_at_ceil),
                "theta", "frac_at_ceil")?;
            aggregate_gate_stats(py, &ldict, &alpha_stats_grid, level, self.n_blocks,
                |gs| (gs.count, gs.min, gs.max, gs.mean, gs.p99, gs.frac_at_bound),
                "alpha", "frac_at_floor")?;
            aggregate_gate_stats(py, &ldict, &eta_stats_grid, level, self.n_blocks,
                |gs| (gs.count, gs.min, gs.max, gs.mean, gs.p99, gs.frac_at_bound),
                "eta", "frac_at_bound")?;
            agg_levels_list.append(ldict)?;
        }
        dict.set_item("levels", agg_levels_list)?;
        dict.set_item("total_blocks", total_active)?;

        Ok(dict.into())
    }

}

// ── NIAH utilities ───────────────────────────────────────────────────

/// Compute log-probability of a specific token at a position in logits.
///
/// logits_flat: [seq_len * vocab] row-major from model.forward()
/// position: which token position to score
/// token_id: which vocab token to get the log-prob for
/// vocab: vocabulary size
///
/// Returns log p(token_id | logits[position]) using numerically-stable
/// log-sum-exp. Returns -inf for out-of-range token_id, NaN for bad logits.
#[pyfunction]
fn logprob_at_position(
    logits_flat: Vec<f32>,
    position: usize,
    token_id: usize,
    vocab: usize,
) -> PyResult<f64> {
    if token_id >= vocab {
        return Ok(f64::NEG_INFINITY);
    }
    let row_start = position.checked_mul(vocab).ok_or_else(|| {
        PyValueError::new_err(format!(
            "position {position} * vocab {vocab} overflows usize"
        ))
    })?;
    let row_end = row_start.checked_add(vocab).ok_or_else(|| {
        PyValueError::new_err(format!(
            "position {position} out of range (logits has {} positions, vocab={vocab})",
            logits_flat.len() / vocab.max(1)
        ))
    })?;
    if row_end > logits_flat.len() {
        return Err(PyValueError::new_err(format!(
            "position {position} out of range (logits has {} positions, vocab={vocab})",
            logits_flat.len() / vocab.max(1)
        )));
    }
    let row = &logits_flat[row_start..row_end];

    let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_logit.is_nan() || max_logit.is_infinite() {
        return Ok(f64::NAN);
    }

    let exp_sum: f64 = row.iter().map(|&x| ((x - max_logit) as f64).exp()).sum();
    if exp_sum <= 0.0 {
        return Ok(f64::NAN);
    }

    let log_sum_exp = (max_logit as f64) + exp_sum.ln();
    Ok((row[token_id] as f64) - log_sum_exp)
}

// ── Module ───────────────────────────────────────────────────────────

#[pymodule]
fn nl_hecate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SWAConfig>()?;
    m.add_class::<SWAParams>()?;
    m.add_function(wrap_pyfunction!(create_config, m)?)?;
    m.add_function(wrap_pyfunction!(init_params, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(apply_weight_gradients, m)?)?;
    // MAG
    m.add_class::<MAGConfig>()?;
    m.add_class::<MAGParams>()?;
    m.add_class::<MAGForwardCache>()?;
    m.add_function(wrap_pyfunction!(mag_create_config, m)?)?;
    m.add_function(wrap_pyfunction!(mag_init_params, m)?)?;
    m.add_function(wrap_pyfunction!(extend_params_push_up, m)?)?;
    m.add_function(wrap_pyfunction!(extend_params_stack_up, m)?)?;
    m.add_function(wrap_pyfunction!(mag_forward, m)?)?;
    m.add_function(wrap_pyfunction!(mag_compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(mag_apply_weight_gradients, m)?)?;
    // CMS Variants
    m.add_class::<MultiBlockConfig>()?;
    // Stateful CMS build loop
    m.add_class::<Conductor>()?;
    m.add_class::<Pulse>()?;
    m.add_class::<ContextState>()?;
    m.add_class::<ErrorBufferList>()?;
    m.add_class::<VecStream>()?;
    m.add_class::<CMSForwardCache>()?;
    m.add_function(wrap_pyfunction!(cms_forward, m)?)?;
    m.add_function(wrap_pyfunction!(cms_compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(save_build_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(save_checkpoint_with_context, m)?)?;
    m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_build_checkpoint, m)?)?;
    // Stacked checkpoint + extend_k
    m.add_function(wrap_pyfunction!(is_stacked_checkpoint, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(save_stacked_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_stacked_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(extend_stacked_push_up, m)?)?;
    m.add_function(wrap_pyfunction!(extend_stacked_clone, m)?)?;
    // CPU frequency-aware AdamW optimizer
    m.add_class::<FrequencyAwareAdamW>()?;
    // GPU-resident model
    #[cfg(feature = "cuda")]
    m.add_class::<GpuModel>()?;
    #[cfg(feature = "cuda")]
    m.add_class::<GpuStackedModel>()?;
    // NIAH utilities
    m.add_function(wrap_pyfunction!(logprob_at_position, m)?)?;
    Ok(())
}
