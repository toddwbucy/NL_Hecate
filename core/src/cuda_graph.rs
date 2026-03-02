/// CUDA Graph capture and replay for CMS forward pass kernel dispatch.
///
/// After `warmup_steps` steps, each distinct CMS pulse bitmask is lazily captured into
/// a `cudaGraphExec_t` on its first occurrence. Subsequent occurrences replay the graph
/// via `cudaGraphLaunch`, eliminating CPU→GPU kernel dispatch overhead.
///
/// Feature-gated: only available with `--features cuda`.
///
/// See: specs/infrastructure/cuda_graph_capture.md

#[cfg(feature = "cuda")]
use crate::gpu_buf::GpuBuf;
#[cfg(feature = "cuda")]
use crate::model::{MAGConfig, MemoryRuleKind};

// ══════════════════════════════════════════════════════════════════════
// CUDA FFI — graph capture API
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
extern "C" {
    /// Begin capturing all work submitted to `stream` into a graph.
    /// mode=0 → cudaStreamCaptureModeGlobal
    fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;

    /// End capture; returns the captured graph via `*graph_out`.
    fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph_out: *mut *mut std::ffi::c_void,
    ) -> i32;

    /// Instantiate a graph into an executable graph.
    fn cudaGraphInstantiate(
        exec_out: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        flags: u64,
    ) -> i32;

    /// Launch a graph exec on `stream` (0 = default stream).
    fn cudaGraphLaunch(
        exec: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// Destroy an instantiated graph exec (frees GPU resources).
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;

    /// Destroy a raw graph (call after instantiation; exec is independent).
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
}

// ══════════════════════════════════════════════════════════════════════
// CudaGraphExec — RAII wrapper around a single instantiated graph
// ══════════════════════════════════════════════════════════════════════

/// Owns one `cudaGraphExec_t` handle. Calls `cudaGraphExecDestroy` on drop.
#[cfg(feature = "cuda")]
pub struct CudaGraphExec {
    handle: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        let rc = unsafe { cudaGraphExecDestroy(self.handle) };
        debug_assert_eq!(rc, 0, "cudaGraphExecDestroy failed: {rc}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// CudaGraphStore — per-bitmask graph handle store
// ══════════════════════════════════════════════════════════════════════

/// Manages captured `cudaGraphExec_t` handles, one per CMS pulse bitmask.
///
/// For k=4 there are 8 reachable bitmasks (L0 always fires → only odd bitmasks 1..15).
/// The store allocates 16 slots (indexed by bitmask u8) but populates only the reachable ones.
///
/// Lifecycle:
///   - warmup (steps < warmup_steps): `should_capture()` is false, standard dispatch
///   - capture (step == warmup_steps): caller must capture each reachable pattern
///   - replay (steps > warmup_steps): `replay()` launches the captured graph
///   - if capture fails: `enabled = false`, falls through to standard dispatch permanently
#[cfg(feature = "cuda")]
pub struct CudaGraphStore {
    /// Indexed by pulse bitmask (u8, 0..=15). None = not yet captured or disabled.
    graphs: [Option<CudaGraphExec>; 16],
    pub warmup_steps: usize,
    pub steps_seen: usize,
    pub enabled: bool,
}

#[cfg(feature = "cuda")]
impl CudaGraphStore {
    pub fn new(warmup_steps: usize) -> Self {
        CudaGraphStore {
            graphs: std::array::from_fn(|_| None),
            warmup_steps,
            steps_seen: 0,
            enabled: warmup_steps > 0,  // disabled when warmup_steps=0
        }
    }

    /// Increment step counter. Call once per `gpu_cms_forward` invocation.
    pub fn step(&mut self) {
        self.steps_seen += 1;
    }

    /// True exactly at step == warmup_steps (single capture opportunity).
    pub fn should_capture(&self) -> bool {
        self.enabled && self.steps_seen == self.warmup_steps
    }

    /// True for steps > warmup_steps when a graph has been captured for this bitmask.
    pub fn should_replay(&self, bitmask: u8) -> bool {
        let idx = bitmask as usize;
        if idx >= self.graphs.len() {
            return false;
        }
        self.enabled
            && self.steps_seen > self.warmup_steps
            && self.graphs[idx].is_some()
    }

    /// Capture the default stream into a new graph exec for `bitmask`.
    ///
    /// Caller is responsible for:
    ///   1. Calling `begin_capture()` before any kernel launches.
    ///   2. Submitting all kernel launches to the default stream.
    ///   3. Calling `end_capture(bitmask)` after all launches.
    ///
    /// Returns `false` if capture failed (caller should log a warning and call `disable()`).
    pub fn begin_capture(&self) -> bool {
        // cudaStreamCaptureModeGlobal = 0
        let rc = unsafe { cudaStreamBeginCapture(std::ptr::null_mut(), 0) };
        rc == 0
    }

    /// Finalize capture for `bitmask`. Instantiates the graph and stores the exec.
    /// Returns `false` on failure.
    pub fn end_capture(&mut self, bitmask: u8) -> bool {
        let mut raw_graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            cudaStreamEndCapture(std::ptr::null_mut(), &mut raw_graph)
        };
        if rc != 0 || raw_graph.is_null() {
            return false;
        }

        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc2 = unsafe { cudaGraphInstantiate(&mut exec, raw_graph, 0) };
        // Always destroy the raw graph (exec is independent).
        unsafe { cudaGraphDestroy(raw_graph) };

        if rc2 != 0 || exec.is_null() {
            return false;
        }

        let idx = bitmask as usize;
        if idx >= self.graphs.len() {
            unsafe { cudaGraphExecDestroy(exec) };
            return false;
        }
        self.graphs[idx] = Some(CudaGraphExec { handle: exec });
        true
    }

    /// Replay the captured graph for `bitmask` on the default stream.
    /// Returns `false` if no graph is stored for this bitmask.
    pub fn replay(&self, bitmask: u8) -> bool {
        let idx = bitmask as usize;
        if idx >= self.graphs.len() {
            return false;
        }
        if let Some(exec) = &self.graphs[idx] {
            let rc = unsafe { cudaGraphLaunch(exec.handle, std::ptr::null_mut()) };
            rc == 0
        } else {
            false
        }
    }

    /// Permanently disable graph capture/replay (e.g., after a capture failure).
    /// Falls through to standard dispatch for all subsequent steps.
    pub fn disable(&mut self) {
        self.enabled = false;
        for slot in &mut self.graphs {
            *slot = None;
        }
    }

    /// Invalidate all captured graphs (e.g., on checkpoint resume with different d or seq_len).
    /// Re-enters warmup phase.
    pub fn invalidate(&mut self) {
        for slot in &mut self.graphs {
            *slot = None;
        }
        self.steps_seen = 0;
        // Re-enable if warmup_steps > 0 (was set at construction).
        self.enabled = self.warmup_steps > 0;
    }
}

// ══════════════════════════════════════════════════════════════════════
// GpuLevelScratch — pre-allocated activation buffers for one CMS level
// ══════════════════════════════════════════════════════════════════════

/// Pre-allocated GPU scratch buffers for one CMS level's forward activations.
///
/// These buffers have FIXED device addresses after allocation. CUDA graphs capture
/// kernel launches using these addresses; replay reuses the same addresses, filling
/// them with updated values from the current step.
///
/// The backward pass reads from these via non-owning `GpuBuf` views in `GpuMemoryCache`.
/// Safety invariant: the backward pass is called before the next forward (which would
/// overwrite scratch), so the non-owning views are always valid during backward.
#[cfg(feature = "cuda")]
pub struct GpuLevelScratch {
    /// Key/value/query projections — [bs * s * d]
    pub k_mem: GpuBuf<f32>,
    pub v_mem: GpuBuf<f32>,
    pub q_mem: GpuBuf<f32>,
    /// Retention gates — [bs * s]
    pub alpha: GpuBuf<f32>,
    pub theta: GpuBuf<f32>,
    /// Momentum gate (Titans only, allocated for all to simplify pattern matching) — [bs * s]
    pub eta: GpuBuf<f32>,
    /// Full M trajectory — [bs * (s+1) * d*d]
    pub m_states: GpuBuf<f32>,
    /// Momentum trajectory (Titans only) — [bs * (s+1) * d*d]; zeros(1) for others
    pub s_states: GpuBuf<f32>,
    /// Per-level output — [bs * s * d]
    pub y: GpuBuf<f32>,
    /// Whether s_states is a real allocation (TitansLMM) or a dummy zeros(1).
    pub has_s_states: bool,
    /// Initial momentum state for Titans — [bs * d*d]; zeros(1) for non-Titans.
    /// Persistent (not transient) so captured graph nodes reference a stable device pointer.
    pub s_initial: GpuBuf<f32>,
}

#[cfg(feature = "cuda")]
impl GpuLevelScratch {
    /// Allocate scratch buffers for one level.
    ///
    /// `has_s_states` should be `true` only when `memory_rule == TitansLMM` — avoids
    /// the ~537MB s_states allocation for non-Titans rules.
    pub fn new(bs: usize, s: usize, d: usize, has_s_states: bool) -> Self {
        let dd = d * d;
        GpuLevelScratch {
            k_mem:    GpuBuf::zeros(bs * s * d),
            v_mem:    GpuBuf::zeros(bs * s * d),
            q_mem:    GpuBuf::zeros(bs * s * d),
            alpha:    GpuBuf::zeros(bs * s),
            theta:    GpuBuf::zeros(bs * s),
            eta:      GpuBuf::zeros(bs * s),
            m_states: GpuBuf::zeros(bs * (s + 1) * dd),
            s_states: if has_s_states {
                GpuBuf::zeros(bs * (s + 1) * dd)
            } else {
                GpuBuf::zeros(1)  // dummy — never used
            },
            y:        GpuBuf::zeros(bs * s * d),
            has_s_states,
            s_initial: if has_s_states {
                GpuBuf::zeros(bs * dd)  // Titans: persistent zero buffer, zeroed each step
            } else {
                GpuBuf::zeros(1)  // dummy — never used
            },
        }
    }

    /// Allocate scratch buffers using MAGConfig to determine which fields are needed.
    pub fn from_cfg(cfg: &MAGConfig, bs: usize) -> Self {
        let s = cfg.swa.seq_len;
        let d = cfg.swa.d_model;
        let has_s_states = matches!(cfg.memory_rule, MemoryRuleKind::TitansLMM);
        Self::new(bs, s, d, has_s_states)
    }
}

// ══════════════════════════════════════════════════════════════════════
// ForwardScratch — pre-allocated buffers for gpu_cms_forward
// ══════════════════════════════════════════════════════════════════════

/// Pre-allocated GPU scratch buffers for the non-level portions of `gpu_cms_forward`.
///
/// Includes embeddings, QKV projections, attention outputs, gating buffers, logits.
/// Combined with `GpuLevelScratch` (per-level), covers the entire forward pass.
#[cfg(feature = "cuda")]
pub struct ForwardScratch {
    pub d_input_ids:  GpuBuf<f32>,    // [bs * s] — stores i32 reinterpreted
    pub d_target_ids: GpuBuf<f32>,    // [bs * s]
    pub embedded:     GpuBuf<f32>,    // [bs * s * d]
    pub q_f32:        GpuBuf<f32>,    // [bs * s * d]
    pub k_f32:        GpuBuf<f32>,    // [bs * s * d]
    pub v_f32:        GpuBuf<f32>,    // [bs * s * d]
    pub q_bf16:       GpuBuf<u16>,    // [bs * s * d]
    pub k_bf16:       GpuBuf<u16>,    // [bs * s * d]
    pub v_bf16:       GpuBuf<u16>,    // [bs * s * d]
    pub attn_out_bf16:    GpuBuf<u16>,  // [bs * s * d]
    pub attn_weights_bf16: GpuBuf<u16>, // [bs * nh * s * ws]
    pub attn_out:     GpuBuf<f32>,    // [bs * s * d]
    pub y_combined:   GpuBuf<f32>,    // [bs * s * d]
    pub gate:         GpuBuf<f32>,    // [bs * s * d]
    pub gated_out:    GpuBuf<f32>,    // [bs * s * d]
    pub projected:    GpuBuf<f32>,    // [bs * s * d]
    pub logits:       GpuBuf<f32>,    // [bs * s * v]
    pub loss_gpu:     GpuBuf<f32>,    // [1]
    /// Per-level output buffers (one GpuBuf<f32> per CMS level, [bs*s*d] each).
    pub y_per_level:  Vec<GpuBuf<f32>>,
    /// Per-level q_mem temporaries for the frozen-level path — [bs*s*d] each.
    /// Persistent so frozen-path kernel nodes captured in the graph reference stable pointers.
    pub q_tmp_per_level: Vec<GpuBuf<f32>>,
}

#[cfg(feature = "cuda")]
impl ForwardScratch {
    pub fn from_cfg(cfg: &MAGConfig, bs: usize) -> Self {
        let s  = cfg.swa.seq_len;
        let d  = cfg.swa.d_model;
        let v  = cfg.swa.vocab_size;
        let nh = cfg.swa.num_heads;
        let ws = cfg.swa.window_size;
        let k  = cfg.k;

        ForwardScratch {
            d_input_ids:       GpuBuf::zeros(bs * s),
            d_target_ids:      GpuBuf::zeros(bs * s),
            embedded:          GpuBuf::zeros(bs * s * d),
            q_f32:             GpuBuf::zeros(bs * s * d),
            k_f32:             GpuBuf::zeros(bs * s * d),
            v_f32:             GpuBuf::zeros(bs * s * d),
            q_bf16:            GpuBuf::zeros(bs * s * d),
            k_bf16:            GpuBuf::zeros(bs * s * d),
            v_bf16:            GpuBuf::zeros(bs * s * d),
            attn_out_bf16:     GpuBuf::zeros(bs * s * d),
            attn_weights_bf16: GpuBuf::zeros(bs * nh * s * ws),
            attn_out:          GpuBuf::zeros(bs * s * d),
            y_combined:        GpuBuf::zeros(bs * s * d),
            gate:              GpuBuf::zeros(bs * s * d),
            gated_out:         GpuBuf::zeros(bs * s * d),
            projected:         GpuBuf::zeros(bs * s * d),
            logits:            GpuBuf::zeros(bs * s * v),
            loss_gpu:          GpuBuf::zeros(1),
            y_per_level:       (0..k).map(|_| GpuBuf::zeros(bs * s * d)).collect(),
            q_tmp_per_level:   (0..k).map(|_| GpuBuf::zeros(bs * s * d)).collect(),
        }
    }
}
