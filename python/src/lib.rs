//! PyO3 bindings for NL-Hecate core.
//!
//! Stateless functional API — mirrors the Rust core exactly.
//! No Python-side math. All computation happens in Rust/CUDA.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use nl_hecate_core::model::{SWAConfig as RustConfig, SWAParams as RustParams};
use nl_hecate_core::model::{MAGConfig as RustMAGConfig, MAGParams as RustMAGParams, MemoryRuleKind, CompositionKind};
use nl_hecate_core::forward::{forward as rust_forward, ForwardCache as RustCache};
use nl_hecate_core::backward::backward_full as rust_backward_full;
use nl_hecate_core::gradient::compute_gradients as rust_compute_gradients;
use nl_hecate_core::mag::{mag_forward as rust_mag_forward, MAGForwardCache as RustMAGCache, mag_backward as rust_mag_backward};
use nl_hecate_core::gradient::mag_compute_gradients as rust_mag_compute_gradients;

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
        d_hidden=None, lp_p=None, lq_q=None, lambda_local=None, lambda_2=None,
        delta=None, m_slots=None, d_compress=None, lambda_k=None, lambda_v=None,
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
        lq_q: Option<f32>,
        lambda_local: Option<f32>,
        lambda_2: Option<f32>,
        delta: Option<f32>,
        m_slots: Option<usize>,
        d_compress: Option<usize>,
        lambda_k: Option<f32>,
        lambda_v: Option<f32>,
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
            _ => return Err(PyValueError::new_err(format!(
                "Unknown memory_rule '{memory_rule}'. Expected: delta, titans, hebbian, moneta, yaad, memora, lattice, trellis"
            ))),
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
                lq_q: lq_q.unwrap_or(2.0),
                lambda_local: lambda_local.unwrap_or(0.0),
                lambda_2: lambda_2.unwrap_or(0.0),
                delta: delta.unwrap_or(1.0),
                m_slots: m_slots.unwrap_or(0),
                d_compress: d_compress.unwrap_or(0),
                lambda_k: lambda_k.unwrap_or(0.0),
                lambda_v: lambda_v.unwrap_or(0.0),
                parallel: None,
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
        }
    }
    #[getter]
    fn k(&self) -> usize { self.inner.k }
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
    d_hidden=None, lp_p=None, lq_q=None, lambda_local=None, lambda_2=None,
    delta=None, m_slots=None, d_compress=None, lambda_k=None, lambda_v=None,
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
    lq_q: Option<f32>,
    lambda_local: Option<f32>,
    lambda_2: Option<f32>,
    delta: Option<f32>,
    m_slots: Option<usize>,
    d_compress: Option<usize>,
    lambda_k: Option<f32>,
    lambda_v: Option<f32>,
) -> PyResult<MAGConfig> {
    MAGConfig::new(
        d_model, num_heads, head_dim, seq_len, window_size, vocab_size, memory_enabled,
        k, chunk_sizes, memory_rule, composition,
        d_hidden, lp_p, lq_q, lambda_local, lambda_2, delta, m_slots, d_compress, lambda_k, lambda_v,
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
    for (i, &tok) in input_ids.iter().enumerate() {
        if tok >= vocab {
            return Err(PyValueError::new_err(format!(
                "input_ids[{i}]={tok} must be < vocab_size ({vocab})"
            )));
        }
    }
    for (i, &tok) in target_ids.iter().enumerate() {
        if tok >= vocab {
            return Err(PyValueError::new_err(format!(
                "target_ids[{i}]={tok} must be < vocab_size ({vocab})"
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
    Ok(())
}
