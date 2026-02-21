//! PyO3 bindings for NL-Hecate core.
//!
//! Stateless functional API — mirrors the Rust core exactly.
//! No Python-side math. All computation happens in Rust/CUDA.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use nl_hecate_core::model::{SWAConfig as RustConfig, SWAParams as RustParams};
use nl_hecate_core::model::{MAGConfig as RustMAGConfig, MAGParams as RustMAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::retention::{RetentionKind, default_retention};
use nl_hecate_core::m3::M3Config as RustM3Config;
use nl_hecate_core::dynamic_freq::{FrequencySchedule, LearnedFreqConfig};
use nl_hecate_core::cms_variants::{
    DeploymentVariant as RustDeploymentVariant,
    BlockConfig as RustBlockConfig,
    MultiBlockConfig as RustMultiBlockConfig,
};
use nl_hecate_core::forward::{forward as rust_forward, ForwardCache as RustCache};
use nl_hecate_core::backward::backward_full as rust_backward_full;
use nl_hecate_core::gradient::compute_gradients as rust_compute_gradients;
use nl_hecate_core::mag::{mag_forward as rust_mag_forward, MAGForwardCache as RustMAGCache, mag_backward as rust_mag_backward};
use nl_hecate_core::gradient::mag_compute_gradients as rust_mag_compute_gradients;
use nl_hecate_core::mag::{cms_forward as rust_cms_forward, cms_backward as rust_cms_backward, CMSForwardCache as RustCMSCache};
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

// ── ForwardCache ─────────────────────────────────────────────────────

#[pyclass]
struct ForwardCache {
    inner: RustCache,
}

#[pymethods]
impl ForwardCache {
    /// Return logits as flat list: [seq_len * vocab_size], row-major.
    fn get_logits(&self) -> Vec<f32> {
        self.inner.logits.clone()
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
fn forward(params: &SWAParams, cfg: &SWAConfig, input_ids: Vec<usize>, target_ids: Vec<usize>) -> PyResult<(f32, ForwardCache)> {
    validate_seq_lens(cfg, &input_ids, &target_ids)?;
    let (loss, cache) = rust_forward(&params.inner, &cfg.inner, &input_ids, &target_ids);
    Ok((loss, ForwardCache { inner: cache }))
}

#[pyfunction]
fn backward(
    params: &SWAParams,
    cfg: &SWAConfig,
    cache: &ForwardCache,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
) -> PyResult<SWAParams> {
    validate_seq_lens(cfg, &input_ids, &target_ids)?;
    let grads = rust_backward_full(&params.inner, &cfg.inner, &cache.inner, &input_ids, &target_ids);
    Ok(SWAParams { inner: grads })
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
        retention=None, m3=None, frequency_schedule=None, checkpoint_interval=None,
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
    ) -> PyResult<Self> {
        if d_model != num_heads * head_dim {
            return Err(PyValueError::new_err(format!(
                "d_model ({d_model}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
            )));
        }
        if k < 1 {
            return Err(PyValueError::new_err("k must be >= 1"));
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
            _ => return Err(PyValueError::new_err(format!(
                "Unknown memory_rule '{memory_rule}'. Expected: delta, titans, hebbian, moneta, yaad, memora, lattice, trellis, atlas, atlas_omega"
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
                parallel: None,
                retention: ret_kind,
                m3: m3_cfg,
                frequency_schedule: freq_sched,
                checkpoint_interval,
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
        }
    }
    #[getter]
    fn k(&self) -> usize { self.inner.k }
    #[getter]
    fn chunk_sizes(&self) -> Vec<usize> { self.inner.chunk_sizes.clone() }
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
        dict.set_item("w_k_mem", self.inner.levels[0].w_k_mem.clone())?;
        dict.set_item("w_v_mem", self.inner.levels[0].w_v_mem.clone())?;
        dict.set_item("w_q_mem", self.inner.levels[0].w_q_mem.clone())?;
        dict.set_item("w_alpha", self.inner.levels[0].w_alpha.clone())?;
        dict.set_item("b_alpha", self.inner.levels[0].b_alpha.clone())?;
        dict.set_item("w_theta", self.inner.levels[0].w_theta.clone())?;
        dict.set_item("b_theta", self.inner.levels[0].b_theta.clone())?;
        Ok(dict)
    }

    /// Flatten all params into a single Vec<f32> for Python-side optimizers.
    /// Order: SWA(embed,q,k,v,o,unembed) then per-level(k_mem,v_mem,q_mem,alpha,b_alpha,theta,b_theta,eta,b_eta,omega,freq,b_freq).
    fn get_flat_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.inner.num_params());
        flat.extend_from_slice(&self.inner.swa.w_embed);
        flat.extend_from_slice(&self.inner.swa.w_q);
        flat.extend_from_slice(&self.inner.swa.w_k);
        flat.extend_from_slice(&self.inner.swa.w_v);
        flat.extend_from_slice(&self.inner.swa.w_o);
        flat.extend_from_slice(&self.inner.swa.w_unembed);
        for level in &self.inner.levels {
            flat.extend_from_slice(&level.w_k_mem);
            flat.extend_from_slice(&level.w_v_mem);
            flat.extend_from_slice(&level.w_q_mem);
            flat.extend_from_slice(&level.w_alpha);
            flat.extend_from_slice(&level.b_alpha);
            flat.extend_from_slice(&level.w_theta);
            flat.extend_from_slice(&level.b_theta);
            flat.extend_from_slice(&level.w_eta);
            flat.extend_from_slice(&level.b_eta);
            flat.extend_from_slice(&level.w_omega);
            flat.extend_from_slice(&level.w_freq);
            flat.extend_from_slice(&level.b_freq);
        }
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
        copy_slice!(self.inner.swa.w_embed);
        copy_slice!(self.inner.swa.w_q);
        copy_slice!(self.inner.swa.w_k);
        copy_slice!(self.inner.swa.w_v);
        copy_slice!(self.inner.swa.w_o);
        copy_slice!(self.inner.swa.w_unembed);
        for level in &mut self.inner.levels {
            copy_slice!(level.w_k_mem);
            copy_slice!(level.w_v_mem);
            copy_slice!(level.w_q_mem);
            copy_slice!(level.w_alpha);
            copy_slice!(level.b_alpha);
            copy_slice!(level.w_theta);
            copy_slice!(level.b_theta);
            copy_slice!(level.w_eta);
            copy_slice!(level.b_eta);
            copy_slice!(level.w_omega);
            copy_slice!(level.w_freq);
            copy_slice!(level.b_freq);
        }
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
    retention=None, m3=None, frequency_schedule=None, checkpoint_interval=None,
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
) -> PyResult<MAGConfig> {
    MAGConfig::new(
        d_model, num_heads, head_dim, seq_len, window_size, vocab_size, memory_enabled,
        k, chunk_sizes, memory_rule, composition,
        d_hidden, lp_p, sign_sharpness, lq_q, lambda_local, lambda_2, delta, m_slots, d_compress, lambda_k, lambda_v,
        retention, m3, frequency_schedule, checkpoint_interval,
    )
}

#[pyfunction]
fn mag_init_params(cfg: &MAGConfig, seed: u64) -> MAGParams {
    MAGParams {
        inner: RustMAGParams::init(&cfg.inner, seed),
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
fn mag_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &MAGForwardCache,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
) -> PyResult<MAGParams> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let grads = rust_mag_backward(&params.inner, &cfg.inner, &cache.inner, &input_ids, &target_ids);
    Ok(MAGParams { inner: grads })
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

    #[getter]
    fn d(&self) -> usize { self.inner.d }
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

#[pyfunction]
fn cms_backward(
    params: &MAGParams,
    cfg: &MAGConfig,
    cache: &CMSForwardCache,
    input_ids: Vec<usize>,
    target_ids: Vec<usize>,
    error_buffers: &mut ErrorBufferList,
) -> PyResult<MAGParams> {
    validate_mag_seq_lens(cfg, &input_ids, &target_ids)?;
    let grads = rust_cms_backward(
        &params.inner, &cfg.inner, &cache.inner,
        &input_ids, &target_ids, &mut error_buffers.inner,
    );
    Ok(MAGParams { inner: grads })
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
    kv_cache: Option<nl_hecate_core::gpu_forward::GpuKVCache>,
    decode_workspace: Option<nl_hecate_core::gpu_forward::DecodeWorkspace>,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl GpuModel {
    /// Create a GPU-resident model from a MAGConfig and random seed.
    /// All parameters are uploaded to GPU once.
    #[new]
    fn new(cfg: &MAGConfig, seed: u64) -> PyResult<Self> {
        let host_params = nl_hecate_core::model::MAGParams::init(&cfg.inner, seed);
        let gpu_params = nl_hecate_core::gpu_params::GpuMAGParams::from_host(&host_params);
        let gpu_context = nl_hecate_core::gpu_params::GpuContextState::new(cfg.inner.k, cfg.inner.swa.d_model);
        Ok(GpuModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            kv_cache: None,
            decode_workspace: None,
        })
    }

    /// Create from existing host params (e.g., loaded from checkpoint).
    #[staticmethod]
    fn from_params(params: &MAGParams, cfg: &MAGConfig) -> PyResult<Self> {
        let gpu_params = nl_hecate_core::gpu_params::GpuMAGParams::from_host(&params.inner);
        let gpu_context = nl_hecate_core::gpu_params::GpuContextState::new(cfg.inner.k, cfg.inner.swa.d_model);
        Ok(GpuModel {
            params: gpu_params,
            context: gpu_context,
            cfg: cfg.inner.clone(),
            adamw_state: None,
            kv_cache: None,
            decode_workspace: None,
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
            &self.params, &self.cfg, &cache,
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
            &self.params, &self.cfg, &cache,
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

    /// Forward-only pass: returns (loss, logits_flat).
    /// logits_flat is [seq_len * vocab_size] in row-major order.
    fn forward(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
               pulse: &Pulse) -> PyResult<(f32, Vec<f32>)> {
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
        let mut logits = vec![0.0f32; s * v];
        cache.logits.copy_to_host(&mut logits);
        Ok((loss, logits))
    }

    /// Full GPU build step with AdamW optimizer. Returns (loss, grad_norm).
    /// Zero PCIe traffic for weights — only input_ids, target_ids, and loss cross the bus.
    /// AdamW state is lazily created on first call.
    #[pyo3(signature = (input_ids, target_ids, pulse, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1, max_grad_norm=1.0))]
    fn step_adamw(&mut self, input_ids: Vec<usize>, target_ids: Vec<usize>,
                  pulse: &Pulse, lr: f32, beta1: f32, beta2: f32,
                  eps: f32, weight_decay: f32, max_grad_norm: f32) -> PyResult<(f32, f32)> {
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

        // Forward
        let (loss, cache) = nl_hecate_core::gpu_forward::gpu_cms_forward(
            &self.params, &self.cfg, &input_ids, &target_ids,
            &pulse.inner, &mut self.context,
        );

        // Backward
        let mut grads = nl_hecate_core::gpu_backward::gpu_cms_backward(
            &self.params, &self.cfg, &cache,
        );

        // Lazy-init AdamW state
        if self.adamw_state.is_none() {
            self.adamw_state = Some(
                nl_hecate_core::gpu_optimizer::GpuAdamWState::from_params(&self.params)
            );
        }
        let state = self.adamw_state.as_mut().unwrap();

        // AdamW update (with grad clipping)
        let grad_norm = nl_hecate_core::gpu_optimizer::gpu_adamw_update(
            &mut self.params, &mut grads, state,
            lr, beta1, beta2, eps, weight_decay, max_grad_norm,
        );

        // Weight tying: sync w_unembed^T → w_embed
        nl_hecate_core::gpu_backward::gpu_sync_embed_weights(
            &mut self.params,
            self.cfg.swa.d_model,
            self.cfg.swa.vocab_size,
        );

        Ok((loss, grad_norm))
    }

    /// Get current AdamW optimizer step count.
    #[getter]
    fn adamw_step(&self) -> u32 {
        self.adamw_state.as_ref().map_or(0, |s| s.step)
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
    fn reset_context(&mut self) {
        self.context.reset();
    }

    /// Upload context state from host (e.g., to restore after a read-only run).
    fn upload_context(&mut self, ctx: &ContextState) -> PyResult<()> {
        if ctx.inner.d != self.cfg.swa.d_model || ctx.inner.memory.len() != self.cfg.k {
            return Err(PyValueError::new_err(format!(
                "context shape mismatch: got k={} d={}, expected k={} d={}",
                ctx.inner.memory.len(), ctx.inner.d, self.cfg.k, self.cfg.swa.d_model
            )));
        }
        self.context = nl_hecate_core::gpu_params::GpuContextState::from_host_context(&ctx.inner);
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
}

// ── Module ───────────────────────────────────────────────────────────

#[pymodule]
fn nl_hecate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SWAConfig>()?;
    m.add_class::<SWAParams>()?;
    m.add_class::<ForwardCache>()?;
    m.add_function(wrap_pyfunction!(create_config, m)?)?;
    m.add_function(wrap_pyfunction!(init_params, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(backward, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(apply_weight_gradients, m)?)?;
    // MAG
    m.add_class::<MAGConfig>()?;
    m.add_class::<MAGParams>()?;
    m.add_class::<MAGForwardCache>()?;
    m.add_function(wrap_pyfunction!(mag_create_config, m)?)?;
    m.add_function(wrap_pyfunction!(mag_init_params, m)?)?;
    m.add_function(wrap_pyfunction!(mag_forward, m)?)?;
    m.add_function(wrap_pyfunction!(mag_backward, m)?)?;
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
    m.add_function(wrap_pyfunction!(cms_backward, m)?)?;
    m.add_function(wrap_pyfunction!(cms_compute_gradients, m)?)?;
    m.add_function(wrap_pyfunction!(save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(save_build_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(load_build_checkpoint, m)?)?;
    // GPU-resident model
    #[cfg(feature = "cuda")]
    m.add_class::<GpuModel>()?;
    Ok(())
}
