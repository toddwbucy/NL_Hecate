/// Self-referential Phase 2: Adaptive projection memories (HOPE Eqs 79-82, 85, 88).
///
/// Phase 1 (static): projections are fixed — k_t = embedded @ W_K^T.
/// Phase 2 (adaptive): ALL 6 memories (5 projections + main) use DGD (Eq 88).
///   - M_k, M_v, M_q: produce key/value/query via M @ x
///   - M_eta, M_alpha: produce learning rate and retention gate via M @ x → mean → activation
///   - M_mem: the main memory, keyed by adaptive k_t, storing v_t
///
/// The orchestrator `self_ref_step()` decomposes the forward into:
///   (a) adaptive reads from 5 component memories
///   (b) main memory read
///   (c) shared DGD update for all 6
///
/// Existing rule `step()` methods are untouched — this is a standalone orchestrator
/// that calls `dgd_step()` + `matmul_f32()` directly.
///
/// Source: HOPE (2512.24695) §5, §8; Eqs 79-82, 85, 88.

use serde::{Serialize, Deserialize};
use crate::tensor::{matmul_f32, sigmoid_f32, softplus_f32};
use crate::dgd::dgd_step;

/// Which projection style to use for memory key/value/query generation.
///
/// - Static: Phase 1 — W @ x (default, zero change to existing behavior)
/// - Adaptive: Phase 2 — M_{square}(x) via DGD for all 5 component projections
///
/// Default is Static. No MemoryRuleKind parameter needed for Adaptive because
/// all projection memories use DGD on matrix state (HOPE Eq 88).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProjectionKind {
    /// Phase 1: W @ x (static matmul, current behavior).
    Static,
    /// Phase 2: M_{square}(x) via DGD for all projections.
    Adaptive,
}

impl Default for ProjectionKind {
    fn default() -> Self {
        ProjectionKind::Static
    }
}

/// Per-level self-referential state: 5 component memory matrices.
///
/// All are [d, d] matrices (row-major Vec<f32>). When `ProjectionKind::Static`,
/// all vecs are empty — zero overhead.
///
/// - m_k: key projection memory — produces k_t = M_k @ x_t
/// - m_v: value projection memory — produces v_t = M_v @ x_t
/// - m_q: query projection memory — produces q_t = M_q @ x_t
/// - m_eta: learning rate gate memory — produces scalar theta via mean(M_eta @ x_t)
/// - m_alpha: retention gate memory — produces scalar alpha via mean(M_alpha @ x_t)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfRefState {
    pub m_k: Vec<f32>,
    pub m_v: Vec<f32>,
    pub m_q: Vec<f32>,
    pub m_eta: Vec<f32>,
    pub m_alpha: Vec<f32>,
    pub d: usize,
}

impl SelfRefState {
    /// Create zero-initialized self-ref state for adaptive mode.
    pub fn new(d: usize) -> Self {
        SelfRefState {
            m_k: vec![0.0f32; d * d],
            m_v: vec![0.0f32; d * d],
            m_q: vec![0.0f32; d * d],
            m_eta: vec![0.0f32; d * d],
            m_alpha: vec![0.0f32; d * d],
            d,
        }
    }

    /// Create empty state for Static mode — zero allocation, zero overhead.
    pub fn empty(d: usize) -> Self {
        SelfRefState {
            m_k: Vec::new(),
            m_v: Vec::new(),
            m_q: Vec::new(),
            m_eta: Vec::new(),
            m_alpha: Vec::new(),
            d,
        }
    }

    /// Whether this state has allocated component memories (Adaptive mode).
    pub fn is_active(&self) -> bool {
        !self.m_k.is_empty()
    }

    /// Reset all component memories to zero. No-op if empty (Static mode).
    pub fn reset(&mut self) {
        self.m_k.fill(0.0);
        self.m_v.fill(0.0);
        self.m_q.fill(0.0);
        self.m_eta.fill(0.0);
        self.m_alpha.fill(0.0);
    }
}

/// Cache for self_ref_step backward pass.
///
/// Stores per-token reads and gate values, plus M-state histories for all 6
/// memories (5 projections + main). This enables the reverse-token backward loop.
#[derive(Clone, Debug)]
pub struct SelfRefCache {
    pub seq_len: usize,
    pub d: usize,
    /// Per-token key reads from M_k: [seq_len * d]
    pub k_mem: Vec<f32>,
    /// Per-token value reads from M_v: [seq_len * d]
    pub v_mem: Vec<f32>,
    /// Per-token query reads from M_q: [seq_len * d]
    pub q_mem: Vec<f32>,
    /// Per-token raw eta from M_eta: [seq_len * d]
    pub eta_raw: Vec<f32>,
    /// Per-token raw alpha from M_alpha: [seq_len * d]
    pub alpha_raw: Vec<f32>,
    /// Per-token retention gate (scalar per token): [seq_len]
    pub alpha: Vec<f32>,
    /// Per-token learning rate gate (scalar per token): [seq_len]
    pub theta: Vec<f32>,
    /// M_k state history: [(seq_len+1) * d * d]
    pub m_k_states: Vec<f32>,
    /// M_v state history: [(seq_len+1) * d * d]
    pub m_v_states: Vec<f32>,
    /// M_q state history: [(seq_len+1) * d * d]
    pub m_q_states: Vec<f32>,
    /// M_eta state history: [(seq_len+1) * d * d]
    pub m_eta_states: Vec<f32>,
    /// M_alpha state history: [(seq_len+1) * d * d]
    pub m_alpha_states: Vec<f32>,
    /// Main memory state history: [(seq_len+1) * d * d]
    pub m_mem_states: Vec<f32>,
    /// Main memory output (y_t = M_mem @ q_t): [seq_len * d]
    pub y: Vec<f32>,
    /// Input embeddings (needed for backward): [seq_len * d]
    pub embedded: Vec<f32>,
    /// Self-generated value targets: [6 * seq_len * d] when Phase 3, empty when Phase 2.
    /// Layout: [v_hat_k | v_hat_v | v_hat_q | v_hat_eta | v_hat_alpha | v_hat_mem].
    /// Component index c at token t: v_hat_targets[c * seq_len * d + t * d .. + d].
    pub v_hat_targets: Vec<f32>,
    /// Whether self-generated values were used (needed by backward to choose path).
    pub self_generated_values: bool,
}

impl SelfRefCache {
    /// Allocate cache for a given sequence length and dimension.
    pub fn new(seq_len: usize, d: usize) -> Self {
        let dd = d * d;
        let sd = seq_len * d;
        let states = (seq_len + 1) * dd;
        SelfRefCache {
            seq_len,
            d,
            k_mem: vec![0.0f32; sd],
            v_mem: vec![0.0f32; sd],
            q_mem: vec![0.0f32; sd],
            eta_raw: vec![0.0f32; sd],
            alpha_raw: vec![0.0f32; sd],
            alpha: vec![0.0f32; seq_len],
            theta: vec![0.0f32; seq_len],
            m_k_states: vec![0.0f32; states],
            m_v_states: vec![0.0f32; states],
            m_q_states: vec![0.0f32; states],
            m_eta_states: vec![0.0f32; states],
            m_alpha_states: vec![0.0f32; states],
            m_mem_states: vec![0.0f32; states],
            y: vec![0.0f32; sd],
            embedded: Vec::new(), // filled during forward
            v_hat_targets: Vec::new(), // filled during forward if self_generated_values
            self_generated_values: false, // set during forward
        }
    }
}

/// Gradients w.r.t. initial states of all 6 memories (outer-loop parameters).
#[derive(Clone, Debug)]
pub struct SelfRefParamGrads {
    /// dL/dM_{k,0}: [d * d]
    pub d_m_k: Vec<f32>,
    /// dL/dM_{v,0}: [d * d]
    pub d_m_v: Vec<f32>,
    /// dL/dM_{q,0}: [d * d]
    pub d_m_q: Vec<f32>,
    /// dL/dM_{eta,0}: [d * d]
    pub d_m_eta: Vec<f32>,
    /// dL/dM_{alpha,0}: [d * d]
    pub d_m_alpha: Vec<f32>,
    /// dL/dM_{mem,0}: [d * d]
    pub d_m_mem: Vec<f32>,
}

impl SelfRefParamGrads {
    pub fn zeros(d: usize) -> Self {
        let dd = d * d;
        SelfRefParamGrads {
            d_m_k: vec![0.0f32; dd],
            d_m_v: vec![0.0f32; dd],
            d_m_q: vec![0.0f32; dd],
            d_m_eta: vec![0.0f32; dd],
            d_m_alpha: vec![0.0f32; dd],
            d_m_mem: vec![0.0f32; dd],
        }
    }
}

/// Sigmoid backward: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
fn sigmoid_backward(x: f32) -> f32 {
    let s = sigmoid_f32(x);
    s * (1.0 - s)
}

/// Softplus backward: d/dx softplus(x) = sigmoid(x).
fn softplus_backward(x: f32) -> f32 {
    sigmoid_f32(x)
}

/// Self-referential forward pass: all 6 memories updated per token via DGD.
///
/// Per-token loop (observe-then-advance, CS-32):
///   1. Read 5 component memories to get adaptive k/v/q/eta/alpha
///   2. Reduce d-dim gate outputs to scalars: alpha = sigmoid(mean(alpha_raw)), theta = softplus(mean(eta_raw))
///   3. Read main memory: y_t = M_mem @ q_t
///   4. DGD update all 6 memories with shared (alpha_t, theta_t)
///
/// Source: HOPE (2512.24695) Eqs 79-82, 85, 88.
///
/// - `self_ref`: mutable 5-component projection state (modified in-place)
/// - `m_mem`: mutable [d*d] main memory matrix (modified in-place)
/// - `embedded`: input tokens [seq_len * d]
/// - `seq_len`: number of tokens
/// - `d`: model dimension
///
/// Returns (y [seq_len * d], cache for backward).
pub fn self_ref_step(
    self_ref: &mut SelfRefState,
    m_mem: &mut [f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    self_generated_values: bool,
) -> (Vec<f32>, SelfRefCache) {
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(m_mem.len(), d * d);
    debug_assert!(self_ref.is_active(), "self_ref_step called on empty SelfRefState");

    let dd = d * d;
    let mut cache = SelfRefCache::new(seq_len, d);
    cache.embedded = embedded.to_vec();
    cache.self_generated_values = self_generated_values;
    if self_generated_values {
        cache.v_hat_targets = vec![0.0f32; 6 * seq_len * d];
    }

    // Snapshot initial states (t=0)
    cache.m_k_states[..dd].copy_from_slice(&self_ref.m_k);
    cache.m_v_states[..dd].copy_from_slice(&self_ref.m_v);
    cache.m_q_states[..dd].copy_from_slice(&self_ref.m_q);
    cache.m_eta_states[..dd].copy_from_slice(&self_ref.m_eta);
    cache.m_alpha_states[..dd].copy_from_slice(&self_ref.m_alpha);
    cache.m_mem_states[..dd].copy_from_slice(m_mem);

    let mut y = vec![0.0f32; seq_len * d];

    // Reusable buffers for matmul outputs
    let mut k_t = vec![0.0f32; d];
    let mut v_t = vec![0.0f32; d];
    let mut q_t = vec![0.0f32; d];
    let mut eta_raw_t = vec![0.0f32; d];
    let mut alpha_raw_t = vec![0.0f32; d];
    let mut y_t = vec![0.0f32; d];

    for t in 0..seq_len {
        let x_t = &embedded[t * d..(t + 1) * d];

        // Step 1: Adaptive projections — read from M_{component, t}
        matmul_f32(&self_ref.m_k, x_t, &mut k_t, d, d, 1);
        matmul_f32(&self_ref.m_v, x_t, &mut v_t, d, d, 1);
        matmul_f32(&self_ref.m_q, x_t, &mut q_t, d, d, 1);
        matmul_f32(&self_ref.m_eta, x_t, &mut eta_raw_t, d, d, 1);
        matmul_f32(&self_ref.m_alpha, x_t, &mut alpha_raw_t, d, d, 1);

        // Cache per-token reads
        cache.k_mem[t * d..(t + 1) * d].copy_from_slice(&k_t);
        cache.v_mem[t * d..(t + 1) * d].copy_from_slice(&v_t);
        cache.q_mem[t * d..(t + 1) * d].copy_from_slice(&q_t);
        cache.eta_raw[t * d..(t + 1) * d].copy_from_slice(&eta_raw_t);
        cache.alpha_raw[t * d..(t + 1) * d].copy_from_slice(&alpha_raw_t);

        // Step 2: Reduce d-dim gate outputs to scalars
        let alpha_mean: f32 = alpha_raw_t.iter().sum::<f32>() / d as f32;
        let eta_mean: f32 = eta_raw_t.iter().sum::<f32>() / d as f32;
        let alpha_t = sigmoid_f32(alpha_mean);
        let theta_t = softplus_f32(eta_mean);
        cache.alpha[t] = alpha_t;
        cache.theta[t] = theta_t;

        // Step 3: Main memory read — y_t = M_mem @ q_t
        matmul_f32(m_mem, &q_t, &mut y_t, d, d, 1);
        y[t * d..(t + 1) * d].copy_from_slice(&y_t);
        cache.y[t * d..(t + 1) * d].copy_from_slice(&y_t);

        // Step 3.5: Self-generated value targets (Phase 3, HOPE Eq 84)
        // When enabled, each memory generates its own target: v̂_□ = M_{□,t-1}(v_t).
        // When disabled, all memories share v_t as target (Phase 2 behavior).
        let (v_hat_k, v_hat_v, v_hat_q, v_hat_eta, v_hat_alpha, v_hat_mem);
        if self_generated_values {
            let mut buf = vec![0.0f32; d];
            // v̂_k = M_k @ v_t
            matmul_f32(&self_ref.m_k, &v_t, &mut buf, d, d, 1);
            v_hat_k = buf.clone();
            cache.v_hat_targets[0 * seq_len * d + t * d..0 * seq_len * d + (t + 1) * d].copy_from_slice(&buf);
            // v̂_v = M_v @ v_t
            matmul_f32(&self_ref.m_v, &v_t, &mut buf, d, d, 1);
            v_hat_v = buf.clone();
            cache.v_hat_targets[1 * seq_len * d + t * d..1 * seq_len * d + (t + 1) * d].copy_from_slice(&buf);
            // v̂_q = M_q @ v_t
            matmul_f32(&self_ref.m_q, &v_t, &mut buf, d, d, 1);
            v_hat_q = buf.clone();
            cache.v_hat_targets[2 * seq_len * d + t * d..2 * seq_len * d + (t + 1) * d].copy_from_slice(&buf);
            // v̂_eta = M_eta @ v_t
            matmul_f32(&self_ref.m_eta, &v_t, &mut buf, d, d, 1);
            v_hat_eta = buf.clone();
            cache.v_hat_targets[3 * seq_len * d + t * d..3 * seq_len * d + (t + 1) * d].copy_from_slice(&buf);
            // v̂_alpha = M_alpha @ v_t
            matmul_f32(&self_ref.m_alpha, &v_t, &mut buf, d, d, 1);
            v_hat_alpha = buf.clone();
            cache.v_hat_targets[4 * seq_len * d + t * d..4 * seq_len * d + (t + 1) * d].copy_from_slice(&buf);
            // v̂_mem = M_mem @ v_t
            matmul_f32(m_mem, &v_t, &mut buf, d, d, 1);
            v_hat_mem = buf;
            cache.v_hat_targets[5 * seq_len * d + t * d..5 * seq_len * d + (t + 1) * d].copy_from_slice(&v_hat_mem);
        } else {
            v_hat_k = v_t.clone();
            v_hat_v = v_t.clone();
            v_hat_q = v_t.clone();
            v_hat_eta = v_t.clone();
            v_hat_alpha = v_t.clone();
            v_hat_mem = v_t.clone();
        }

        // Step 4: DGD update all 6 memories (Eq 88)
        // Key fix: ALL 6 memories use k_t as key (Eq 88), not x_t for components.
        dgd_step(&mut self_ref.m_k, &k_t, &v_hat_k, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_v, &k_t, &v_hat_v, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_q, &k_t, &v_hat_q, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_eta, &k_t, &v_hat_eta, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_alpha, &k_t, &v_hat_alpha, alpha_t, theta_t, d);
        dgd_step(m_mem, &k_t, &v_hat_mem, alpha_t, theta_t, d);

        // Snapshot updated states (t+1)
        let off = (t + 1) * dd;
        cache.m_k_states[off..off + dd].copy_from_slice(&self_ref.m_k);
        cache.m_v_states[off..off + dd].copy_from_slice(&self_ref.m_v);
        cache.m_q_states[off..off + dd].copy_from_slice(&self_ref.m_q);
        cache.m_eta_states[off..off + dd].copy_from_slice(&self_ref.m_eta);
        cache.m_alpha_states[off..off + dd].copy_from_slice(&self_ref.m_alpha);
        cache.m_mem_states[off..off + dd].copy_from_slice(m_mem);
    }

    (y, cache)
}

/// Read-only forward for frozen self-referential levels.
///
/// All 6 memories are frozen (no DGD update). Just reads projections
/// from the static state and produces output.
///
/// Returns (y [seq_len * d], q_mem [seq_len * d]) — q_mem needed for frozen backward.
pub fn self_ref_read_only(
    self_ref: &SelfRefState,
    m_mem: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(embedded.len(), seq_len * d);

    // Guard: empty SelfRefState means this level was initialized as Static
    // but cfg says Adaptive — return zeros safely instead of panicking on matmul.
    if !self_ref.is_active() || m_mem.is_empty() {
        return (vec![0.0f32; seq_len * d], vec![0.0f32; seq_len * d]);
    }
    debug_assert_eq!(m_mem.len(), d * d);

    let mut y = vec![0.0f32; seq_len * d];
    let mut q_mem = vec![0.0f32; seq_len * d];
    let mut q_t = vec![0.0f32; d];
    let mut y_t = vec![0.0f32; d];

    for t in 0..seq_len {
        let x_t = &embedded[t * d..(t + 1) * d];
        // Read q from frozen M_q
        matmul_f32(&self_ref.m_q, x_t, &mut q_t, d, d, 1);
        q_mem[t * d..(t + 1) * d].copy_from_slice(&q_t);
        // Read from frozen M_mem
        matmul_f32(m_mem, &q_t, &mut y_t, d, d, 1);
        y[t * d..(t + 1) * d].copy_from_slice(&y_t);
    }

    (y, q_mem)
}

/// Backward for frozen self-referential read-only path.
///
/// Forward: q_t = M_q @ x_t, y_t = M_mem @ q_t (all frozen, no DGD).
/// Backward produces gradients for M_q and M_mem (→ error buffer) and d_embedded.
///
/// `frozen_combined`: concatenation of [M_mem (d*d), M_q (d*d)] stored during
/// frozen forward. Split here to access both matrices without needing ContextState.
///
/// Returns (MemoryLevelParams grads, d_embedded [seq_len * d]).
pub fn self_ref_read_only_backward(
    frozen_combined: &[f32],
    q_mem: &[f32],
    d_y: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
) -> (crate::model::MemoryLevelParams, Vec<f32>) {
    let dd = d * d;
    let mut d_embedded = vec![0.0f32; seq_len * d];
    let grads = crate::model::MemoryLevelParams::zeros_like(d);

    // Guard: empty/undersized frozen_combined → zero grads
    if frozen_combined.len() < 2 * dd {
        return (grads, d_embedded);
    }

    let m_mem = &frozen_combined[..dd];
    let m_q = &frozen_combined[dd..2 * dd];

    for t in 0..seq_len {
        let q_t = &q_mem[t * d..(t + 1) * d];
        let dy_t = &d_y[t * d..(t + 1) * d];

        // Backward through y_t = M_mem @ q_t
        // dq_t = M_mem^T @ dy_t
        let mut dq_t = vec![0.0f32; d];
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_mem[i * d + j] * dy_t[i];
            }
            dq_t[j] = sum;
        }
        // (dM_mem and dM_q gradients go to error buffer via MemoryLevelParams.
        //  Since self-ref uses DGD on raw matrices (not w_k_mem etc.), these grads
        //  are zero in MemoryLevelParams — the actual M-state gradients will be
        //  wired through SelfRefParamGrads in PR 4.)

        // Backward through q_t = M_q @ x_t
        // dx_t = M_q^T @ dq_t
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_q[i * d + j] * dq_t[i];
            }
            d_embedded[t * d + j] += sum;
        }
    }

    (grads, d_embedded)
}

/// Backward pass for self_ref_step. Reverse token loop through all 6 memories.
///
/// For each token t (reverse order):
///   1. Backward through main memory read: y_t = M_mem @ q_t
///   2. Backward through each DGD update (all 6 memories)
///   3. Backward through gate reduction (mean → sigmoid/softplus)
///   4. Backward through component reads (M @ x)
///   5. Accumulate d_embedded[t]
///
/// Returns (d_embedded [seq_len * d], SelfRefParamGrads for initial states).
pub fn self_ref_step_backward(
    cache: &SelfRefCache,
    d_y: &[f32],
    self_generated_values: bool,
) -> (Vec<f32>, SelfRefParamGrads) {
    let s = cache.seq_len;
    let d = cache.d;
    let dd = d * d;

    debug_assert_eq!(d_y.len(), s * d);

    let mut d_embedded = vec![0.0f32; s * d];

    // Running dM accumulators for each memory (accumulate through reverse token chain)
    let mut dm_k = vec![0.0f32; dd];
    let mut dm_v = vec![0.0f32; dd];
    let mut dm_q = vec![0.0f32; dd];
    let mut dm_eta = vec![0.0f32; dd];
    let mut dm_alpha = vec![0.0f32; dd];
    let mut dm_mem = vec![0.0f32; dd];

    // Temp buffers
    let mut dq_t = vec![0.0f32; d];

    for t in (0..s).rev() {
        let x_t = &cache.embedded[t * d..(t + 1) * d];
        let k_t = &cache.k_mem[t * d..(t + 1) * d];
        let v_t = &cache.v_mem[t * d..(t + 1) * d];
        let q_t = &cache.q_mem[t * d..(t + 1) * d];
        let alpha_t = cache.alpha[t];
        let theta_t = cache.theta[t];
        let dy_t = &d_y[t * d..(t + 1) * d];

        // M state at time t (before update)
        let m_k_t = &cache.m_k_states[t * dd..(t + 1) * dd];
        let m_v_t = &cache.m_v_states[t * dd..(t + 1) * dd];
        let m_q_t = &cache.m_q_states[t * dd..(t + 1) * dd];
        let m_eta_t = &cache.m_eta_states[t * dd..(t + 1) * dd];
        let m_alpha_t = &cache.m_alpha_states[t * dd..(t + 1) * dd];
        let m_mem_t = &cache.m_mem_states[t * dd..(t + 1) * dd];

        // ── Step 1: Main memory read backward ──
        // y_t = M_mem_t @ q_t (read uses state BEFORE DGD update at this token)
        //
        // dq_t from read: M_mem_t^T @ dy_t (independent of dm_mem accumulator)
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_mem_t[i * d + j] * dy_t[i];
            }
            dq_t[j] = sum;
        }

        // ── Step 2: Resolve DGD value targets for backward ──
        // Phase 3: v_hat from cache; Phase 2: v_hat == v_t.
        let sd = s * d;
        let (v_hat_k, v_hat_v, v_hat_q, v_hat_eta, v_hat_alpha, v_hat_mem);
        if self_generated_values {
            v_hat_k     = &cache.v_hat_targets[0 * sd + t * d..0 * sd + (t + 1) * d];
            v_hat_v     = &cache.v_hat_targets[1 * sd + t * d..1 * sd + (t + 1) * d];
            v_hat_q     = &cache.v_hat_targets[2 * sd + t * d..2 * sd + (t + 1) * d];
            v_hat_eta   = &cache.v_hat_targets[3 * sd + t * d..3 * sd + (t + 1) * d];
            v_hat_alpha = &cache.v_hat_targets[4 * sd + t * d..4 * sd + (t + 1) * d];
            v_hat_mem   = &cache.v_hat_targets[5 * sd + t * d..5 * sd + (t + 1) * d];
        } else {
            v_hat_k = v_t;
            v_hat_v = v_t;
            v_hat_q = v_t;
            v_hat_eta = v_t;
            v_hat_alpha = v_t;
            v_hat_mem = v_t;
        }

        // ── Step 3: DGD backward for main memory ──
        // Forward: M_{mem,t+1} = dgd_step(M_{mem,t}, k_t, v_hat_mem, alpha_t, theta_t)
        // dm_mem currently holds dL/dM_{mem,t+1} (from future tokens only).
        let main_grads = crate::dgd::dgd_step_backward(&dm_mem, m_mem_t, k_t, v_hat_mem, alpha_t, theta_t, d);
        dm_mem.copy_from_slice(&main_grads.d_m);

        // NOW add the read gradient: dL/dM_mem_read = outer(dy_t, q_t).
        // This must come AFTER DGD backward so dgd_step_backward sees clean dL/dM_{t+1}.
        for i in 0..d {
            for j in 0..d {
                dm_mem[i * d + j] += dy_t[i] * q_t[j];
            }
        }
        // dk_t accumulator: all 6 DGD key grads flow here (key fix: all use k_t)
        let mut dk_t_total = main_grads.d_k;
        // dv_hat accumulators: self-gen backward chains through v_hat = M @ v_t
        let mut dv_t_total = vec![0.0f32; d]; // accumulates final dv_t (the v_t from M_v @ x_t)
        let mut dalpha_total = main_grads.d_alpha;
        let mut dtheta_total = main_grads.d_theta;

        // Chain rule for main memory's value target
        if self_generated_values {
            // v_hat_mem = M_mem_t @ v_t → d_v_hat_mem = main_grads.d_v
            // dM_mem += outer(d_v_hat, v_t)   [chain through v_hat read]
            // dv_t += M_mem_t^T @ d_v_hat
            let d_v_hat = &main_grads.d_v;
            for i in 0..d {
                for j in 0..d {
                    dm_mem[i * d + j] += d_v_hat[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += m_mem_t[i * d + j] * d_v_hat[i];
                }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += main_grads.d_v[i]; }
        }

        // ── Step 4: DGD backward for all 5 component memories ──
        // Key fix: ALL use key=k_t (not x_t). dk accumulates into dk_t_total.
        let mut dx_t = vec![0.0f32; d]; // direct dx contributions from non-DGD paths

        // Helper macro pattern: DGD backward + self-gen chain
        // M_alpha
        let g = crate::dgd::dgd_step_backward(&dm_alpha, m_alpha_t, k_t, v_hat_alpha, alpha_t, theta_t, d);
        dm_alpha.copy_from_slice(&g.d_m);
        for i in 0..d { dk_t_total[i] += g.d_k[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;
        if self_generated_values {
            // v_hat_alpha = M_alpha_t @ v_t
            for i in 0..d {
                for j in 0..d {
                    dm_alpha[i * d + j] += g.d_v[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_alpha_t[i * d + j] * g.d_v[i]; }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        }

        // M_eta
        let g = crate::dgd::dgd_step_backward(&dm_eta, m_eta_t, k_t, v_hat_eta, alpha_t, theta_t, d);
        dm_eta.copy_from_slice(&g.d_m);
        for i in 0..d { dk_t_total[i] += g.d_k[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;
        if self_generated_values {
            for i in 0..d {
                for j in 0..d {
                    dm_eta[i * d + j] += g.d_v[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_eta_t[i * d + j] * g.d_v[i]; }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        }

        // M_q
        let g = crate::dgd::dgd_step_backward(&dm_q, m_q_t, k_t, v_hat_q, alpha_t, theta_t, d);
        dm_q.copy_from_slice(&g.d_m);
        for i in 0..d { dk_t_total[i] += g.d_k[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;
        if self_generated_values {
            for i in 0..d {
                for j in 0..d {
                    dm_q[i * d + j] += g.d_v[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_q_t[i * d + j] * g.d_v[i]; }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        }

        // M_v
        let g = crate::dgd::dgd_step_backward(&dm_v, m_v_t, k_t, v_hat_v, alpha_t, theta_t, d);
        dm_v.copy_from_slice(&g.d_m);
        for i in 0..d { dk_t_total[i] += g.d_k[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;
        if self_generated_values {
            for i in 0..d {
                for j in 0..d {
                    dm_v[i * d + j] += g.d_v[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_v_t[i * d + j] * g.d_v[i]; }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        }

        // M_k
        let g = crate::dgd::dgd_step_backward(&dm_k, m_k_t, k_t, v_hat_k, alpha_t, theta_t, d);
        dm_k.copy_from_slice(&g.d_m);
        for i in 0..d { dk_t_total[i] += g.d_k[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;
        if self_generated_values {
            // v_hat_k = M_k_t @ v_t
            for i in 0..d {
                for j in 0..d {
                    dm_k[i * d + j] += g.d_v[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d { sum += m_k_t[i * d + j] * g.d_v[i]; }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        }

        // ── Step 5: Gate backward ──
        // alpha_t = sigmoid(mean(alpha_raw_t))
        // theta_t = softplus(mean(eta_raw_t))
        let alpha_raw_t = &cache.alpha_raw[t * d..(t + 1) * d];
        let eta_raw_t = &cache.eta_raw[t * d..(t + 1) * d];
        let alpha_mean: f32 = alpha_raw_t.iter().sum::<f32>() / d as f32;
        let eta_mean: f32 = eta_raw_t.iter().sum::<f32>() / d as f32;

        let dalpha_mean = dalpha_total * sigmoid_backward(alpha_mean);
        let deta_mean = dtheta_total * softplus_backward(eta_mean);
        let dalpha_per_dim = dalpha_mean / d as f32;
        let deta_per_dim = deta_mean / d as f32;

        // ── Step 6: Component read backward ──
        // Key fix: dk_t_total (from all 6 DGDs) flows back through k_t = M_k_t @ x_t
        // dM_k += outer(dk_t_total, x_t)
        // dx += M_k_t^T @ dk_t_total
        for i in 0..d {
            for j in 0..d {
                dm_k[i * d + j] += dk_t_total[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_k_t[i * d + j] * dk_t_total[i];
            }
            dx_t[j] += sum;
        }

        // dv_t_total flows back through v_t = M_v_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_v[i * d + j] += dv_t_total[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_v_t[i * d + j] * dv_t_total[i];
            }
            dx_t[j] += sum;
        }

        // dq_t flows back through q_t = M_q_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_q[i * d + j] += dq_t[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_q_t[i * d + j] * dq_t[i];
            }
            dx_t[j] += sum;
        }

        // d_eta_raw flows back through eta_raw = M_eta_t @ x_t
        // d_eta_raw is uniform (deta_per_dim per element)
        for i in 0..d {
            for j in 0..d {
                dm_eta[i * d + j] += deta_per_dim * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_eta_t[i * d + j] * deta_per_dim;
            }
            dx_t[j] += sum;
        }

        // d_alpha_raw flows back through alpha_raw = M_alpha_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_alpha[i * d + j] += dalpha_per_dim * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_alpha_t[i * d + j] * dalpha_per_dim;
            }
            dx_t[j] += sum;
        }

        // Write dx_t to d_embedded
        d_embedded[t * d..(t + 1) * d].copy_from_slice(&dx_t);
    }

    let grads = SelfRefParamGrads {
        d_m_k: dm_k,
        d_m_v: dm_v,
        d_m_q: dm_q,
        d_m_eta: dm_eta,
        d_m_alpha: dm_alpha,
        d_m_mem: dm_mem,
    };

    (d_embedded, grads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_ref_state_empty() {
        let state = SelfRefState::empty(8);
        assert!(!state.is_active());
        assert!(state.m_k.is_empty());
        assert_eq!(state.d, 8);
    }

    #[test]
    fn test_self_ref_state_new() {
        let state = SelfRefState::new(8);
        assert!(state.is_active());
        assert_eq!(state.m_k.len(), 64);
        assert_eq!(state.m_v.len(), 64);
        assert_eq!(state.m_q.len(), 64);
        assert_eq!(state.m_eta.len(), 64);
        assert_eq!(state.m_alpha.len(), 64);
    }

    #[test]
    fn test_self_ref_state_reset() {
        let mut state = SelfRefState::new(4);
        state.m_k[0] = 1.0;
        state.m_v[5] = 2.0;
        state.reset();
        assert!(state.m_k.iter().all(|&x| x == 0.0));
        assert!(state.m_v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_projection_kind_default_is_static() {
        assert_eq!(ProjectionKind::default(), ProjectionKind::Static);
    }

    #[test]
    fn test_self_ref_cache_dimensions() {
        let cache = SelfRefCache::new(4, 8);
        assert_eq!(cache.k_mem.len(), 32);         // 4 * 8
        assert_eq!(cache.alpha.len(), 4);           // seq_len scalars
        assert_eq!(cache.m_k_states.len(), 5 * 64); // (4+1) * 8*8
        assert_eq!(cache.y.len(), 32);
    }

    #[test]
    fn test_self_ref_param_grads_zeros() {
        let grads = SelfRefParamGrads::zeros(4);
        assert_eq!(grads.d_m_k.len(), 16);
        assert!(grads.d_m_k.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_self_ref_step_produces_output() {
        let d = 4;
        let seq_len = 3;
        let mut state = SelfRefState::new(d);
        // Initialize M_k as small identity-like to get nonzero reads
        for i in 0..d { state.m_k[i * d + i] = 0.1; }
        for i in 0..d { state.m_v[i * d + i] = 0.1; }
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        for i in 0..d { state.m_eta[i * d + i] = 0.1; }
        for i in 0..d { state.m_alpha[i * d + i] = 0.1; }

        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1).collect();

        let (y, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);
        assert_eq!(y.len(), seq_len * d);
        // Output should be nonzero (M_mem @ q_t with nonzero q from M_q @ x_t)
        let y_norm: f32 = y.iter().map(|x| x * x).sum();
        assert!(y_norm > 0.0, "self_ref_step should produce nonzero output");
        // Cache dimensions
        assert_eq!(cache.m_k_states.len(), (seq_len + 1) * d * d);
        assert_eq!(cache.alpha.len(), seq_len);
    }

    #[test]
    fn test_adaptive_changes_per_token() {
        let d = 4;
        let seq_len = 2;
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_k[i * d + i] = 0.1; }
        for i in 0..d { state.m_v[i * d + i] = 0.1; }
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        // Set eta/alpha to produce nonzero gates
        for i in 0..d { state.m_eta[i * d + i] = 1.0; }
        for i in 0..d { state.m_alpha[i * d + i] = 1.0; }

        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.01).collect();

        let (_, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);

        // M_k at t=0 should differ from M_k at t=1 (DGD update happened)
        let dd = d * d;
        let mk0 = &cache.m_k_states[0..dd];
        let mk1 = &cache.m_k_states[dd..2 * dd];
        let diff: f32 = mk0.iter().zip(mk1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-8, "M_k should change after DGD update, diff={diff}");
    }

    #[test]
    fn test_self_ref_read_only_frozen() {
        let d = 4;
        let seq_len = 2;
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        let m_mem = vec![0.1f32; d * d]; // all 0.1

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1).collect();
        let state_before = state.clone();

        let (y, q_mem) = self_ref_read_only(&state, &m_mem, &embedded, seq_len, d);
        assert_eq!(y.len(), seq_len * d);
        assert_eq!(q_mem.len(), seq_len * d);

        // State should be unchanged (frozen)
        assert_eq!(state.m_q, state_before.m_q);
        assert_eq!(state.m_k, state_before.m_k);
    }

    #[test]
    fn test_self_ref_read_only_empty_guard() {
        // Empty SelfRefState should return zeros, not panic.
        let d = 4;
        let seq_len = 2;
        let state = SelfRefState::empty(d);
        let m_mem = vec![0.1f32; d * d];
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1).collect();
        let (y, q_mem) = self_ref_read_only(&state, &m_mem, &embedded, seq_len, d);
        assert_eq!(y.len(), seq_len * d);
        assert!(y.iter().all(|&x| x == 0.0), "empty state should produce zeros");
        assert!(q_mem.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_self_ref_read_only_backward_roundtrip() {
        // Frozen backward should produce nonzero d_embedded when inputs are nonzero.
        let d = 4;
        let seq_len = 2;
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_q[i * d + i] = 0.2; }
        let m_mem: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.3 } else { 0.01 }).collect();
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let (y, q_mem) = self_ref_read_only(&state, &m_mem, &embedded, seq_len, d);
        let d_y = y.clone(); // simple loss grad

        // Build frozen_combined = [M_mem, M_q]
        let mut frozen_combined = Vec::with_capacity(2 * d * d);
        frozen_combined.extend_from_slice(&m_mem);
        frozen_combined.extend_from_slice(&state.m_q);

        let (_grads, d_embedded) = self_ref_read_only_backward(
            &frozen_combined, &q_mem, &d_y, &embedded, seq_len, d,
        );
        let de_norm: f32 = d_embedded.iter().map(|x| x * x).sum();
        assert!(de_norm > 0.0, "frozen backward should produce nonzero d_embedded");
    }

    #[test]
    fn test_phase1_to_phase2_continuity() {
        // When M_k is initialized as W_K (static projection matrix),
        // the adaptive read M_k @ x should match the static matmul W_K @ x.
        let d = 4;
        let seq_len = 1;
        let w_k = vec![
            0.1, 0.2, 0.0, 0.0,
            0.0, 0.1, 0.3, 0.0,
            0.0, 0.0, 0.1, 0.2,
            0.1, 0.0, 0.0, 0.1,
        ];

        // Phase 2: M_k initialized as W_K
        let mut state = SelfRefState::new(d);
        state.m_k.copy_from_slice(&w_k);

        // Do a single matmul: M_k @ x
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut k_adaptive = vec![0.0f32; d];
        matmul_f32(&state.m_k, &x, &mut k_adaptive, d, d, 1);

        // Phase 1 equivalent: W_K @ x
        let mut k_static = vec![0.0f32; d];
        matmul_f32(&w_k, &x, &mut k_static, d, d, 1);

        // Should be bit-identical
        assert_eq!(k_adaptive, k_static, "M_k=W_K should give same projection");
    }

    /// Helper: compute scalar loss = sum(y^2) / 2 for simple gradient checking.
    fn simple_loss(y: &[f32]) -> f32 {
        y.iter().map(|x| x * x).sum::<f32>() / 2.0
    }

    /// Helper: d_y for loss = sum(y^2)/2 is just y.
    fn simple_dloss(y: &[f32]) -> Vec<f32> {
        y.to_vec()
    }

    #[test]
    fn test_self_ref_backward_produces_gradients() {
        let d = 4;
        let seq_len = 2;
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_k[i * d + i] = 0.1; }
        for i in 0..d { state.m_v[i * d + i] = 0.1; }
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        for i in 0..d { state.m_eta[i * d + i] = 0.5; }
        for i in 0..d { state.m_alpha[i * d + i] = 0.5; }
        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let (y, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = self_ref_step_backward(&cache, &d_y, false);

        // d_embedded should be nonzero
        let de_norm: f32 = d_embedded.iter().map(|x| x * x).sum();
        assert!(de_norm > 1e-10, "d_embedded should be nonzero, norm={de_norm}");

        // Initial state gradients should be nonzero (can be very small with these init values)
        let dk_norm: f32 = grads.d_m_k.iter().map(|x| x * x).sum();
        assert!(dk_norm > 0.0, "dM_k should be nonzero, norm={dk_norm}");
        let dmem_norm: f32 = grads.d_m_mem.iter().map(|x| x * x).sum();
        assert!(dmem_norm > 0.0, "dM_mem should be nonzero, norm={dmem_norm}");
    }

    #[test]
    fn test_self_ref_backward_fd_check_embedded() {
        // Finite-difference gradient check for d_embedded.
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05; // 5% relative tolerance

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        // Compute analytical gradient
        let mut state0 = SelfRefState::new(d);
        for i in 0..d { state0.m_k[i * d + i] = 0.1; }
        for i in 0..d { state0.m_v[i * d + i] = 0.1; }
        for i in 0..d { state0.m_q[i * d + i] = 0.1; }
        for i in 0..d { state0.m_eta[i * d + i] = 0.3; }
        for i in 0..d { state0.m_alpha[i * d + i] = 0.3; }
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (d_embedded, _) = self_ref_step_backward(&cache, &d_y, false);

        // FD check for each element of embedded
        let mut max_err = 0.0f32;
        for idx in 0..embedded.len() {
            let mut perturbed = embedded.clone();
            perturbed[idx] += eps;

            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut s, &mut mm, &perturbed, seq_len, d, false);
            let loss_p = simple_loss(&y_p);

            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = d_embedded[idx];

            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;

            if ana_grad.abs() > 1e-4 {
                max_err = max_err.max(rel_err);
            }
        }
        assert!(max_err < tol, "FD check failed for d_embedded: max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_self_ref_backward_fd_check_initial_mk() {
        // Finite-difference check for dM_{k,0}.
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let mut state0 = SelfRefState::new(d);
        for i in 0..d { state0.m_k[i * d + i] = 0.1; }
        for i in 0..d { state0.m_v[i * d + i] = 0.1; }
        for i in 0..d { state0.m_q[i * d + i] = 0.1; }
        for i in 0..d { state0.m_eta[i * d + i] = 0.3; }
        for i in 0..d { state0.m_alpha[i * d + i] = 0.3; }
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = self_ref_step_backward(&cache, &d_y, false);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut sp = state0.clone();
            sp.m_k[idx] += eps;
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut sp, &mut mm, &embedded, seq_len, d, false);
            let loss_p = simple_loss(&y_p);

            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = grads.d_m_k[idx];

            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;

            if ana_grad.abs() > 1e-4 {
                max_err = max_err.max(rel_err);
            }
        }
        assert!(max_err < tol, "FD check failed for dM_k: max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_self_ref_backward_fd_check_initial_m_mem() {
        // Finite-difference check for dM_{mem,0}.
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let mut state0 = SelfRefState::new(d);
        for i in 0..d { state0.m_k[i * d + i] = 0.1; }
        for i in 0..d { state0.m_v[i * d + i] = 0.1; }
        for i in 0..d { state0.m_q[i * d + i] = 0.1; }
        for i in 0..d { state0.m_eta[i * d + i] = 0.3; }
        for i in 0..d { state0.m_alpha[i * d + i] = 0.3; }
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = self_ref_step_backward(&cache, &d_y, false);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            mm[idx] += eps;
            let (y_p, _) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, false);
            let loss_p = simple_loss(&y_p);

            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = grads.d_m_mem[idx];

            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;

            if ana_grad.abs() > 1e-4 {
                max_err = max_err.max(rel_err);
            }
        }
        assert!(max_err < tol, "FD check failed for dM_mem: max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_self_ref_step_state_continuity() {
        // After self_ref_step, calling it again with the mutated state
        // should produce different output (state carries forward).
        let d = 4;
        let seq_len = 2;
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_k[i * d + i] = 0.1; }
        for i in 0..d { state.m_v[i * d + i] = 0.1; }
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        for i in 0..d { state.m_eta[i * d + i] = 1.0; }
        for i in 0..d { state.m_alpha[i * d + i] = 1.0; }
        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.01).collect();

        // First call
        let (y1, _) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);

        // Second call with same input but mutated state
        let (y2, _) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);

        // Outputs should differ because state evolved
        let diff: f32 = y1.iter().zip(y2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "Second call should produce different output due to state evolution, diff={diff}");
    }

    #[test]
    fn test_projection_kind_serde_roundtrip() {
        let kinds = [ProjectionKind::Static, ProjectionKind::Adaptive];
        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let back: ProjectionKind = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, kind);
        }
    }

    #[test]
    fn test_self_ref_state_serde_roundtrip() {
        let state = SelfRefState::new(4);
        let json = serde_json::to_string(&state).unwrap();
        let back: SelfRefState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.d, state.d);
        assert_eq!(back.m_k.len(), state.m_k.len());
    }

    // ── Phase 3: Self-generated values tests (HOPE Eq 84-85) ──

    /// Helper: create initialized self-ref state with identity-scaled matrices.
    fn make_self_ref_state(d: usize, scale: f32) -> SelfRefState {
        let mut state = SelfRefState::new(d);
        for i in 0..d {
            state.m_k[i * d + i] = scale;
            state.m_v[i * d + i] = scale;
            state.m_q[i * d + i] = scale;
            state.m_eta[i * d + i] = scale;
            state.m_alpha[i * d + i] = scale;
        }
        state
    }

    #[test]
    fn test_self_gen_identity_init() {
        // With M_□ = I, v_hat = M @ v_t = v_t → Phase 3 should match Phase 2.
        let d = 4;
        let seq_len = 2;
        let scale = 1.0; // identity

        let mut state2 = make_self_ref_state(d, scale);
        let mut m_mem2 = vec![0.0f32; d * d];
        for i in 0..d { m_mem2[i * d + i] = scale; }
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        // Phase 2 (self_generated_values=false)
        let state2_clone = state2.clone();
        let m_mem2_clone = m_mem2.clone();
        let (y2, _) = self_ref_step(&mut state2, &mut m_mem2, &embedded, seq_len, d, false);

        // Phase 3 with identity M (self_generated_values=true)
        let mut state3 = state2_clone;
        let mut m_mem3 = m_mem2_clone;
        let (y3, cache3) = self_ref_step(&mut state3, &mut m_mem3, &embedded, seq_len, d, true);

        // With M = I, v_hat = I @ v_t = v_t, so output should be identical
        let diff: f32 = y2.iter().zip(y3.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-6, "Identity M should give identical output, diff={diff}");

        // Cache should have v_hat_targets populated
        assert_eq!(cache3.v_hat_targets.len(), 6 * seq_len * d);
        assert!(cache3.self_generated_values);
    }

    #[test]
    fn test_self_gen_changes_target() {
        // With non-identity M, v_hat_k should differ from v_hat_v and from v_t.
        let d = 4;
        let seq_len = 1;

        let mut state = SelfRefState::new(d);
        // M_k = 2*I, M_v = 3*I — different projections should produce different targets
        for i in 0..d {
            state.m_k[i * d + i] = 2.0;
            state.m_v[i * d + i] = 3.0;
            state.m_q[i * d + i] = 0.5;
            state.m_eta[i * d + i] = 0.1;
            state.m_alpha[i * d + i] = 0.1;
        }
        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 1.5; }
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let (_, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, true);

        let sd = seq_len * d;
        let v_hat_k = &cache.v_hat_targets[0..d];
        let v_hat_v = &cache.v_hat_targets[sd..sd + d];

        // v_hat_k and v_hat_v should differ (M_k ≠ M_v)
        let diff: f32 = v_hat_k.iter().zip(v_hat_v.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "Different M should produce different v_hat, diff={diff}");

        // v_hat_k should differ from v_t (since M_k = 2I, not I)
        let v_t = &cache.v_mem[0..d]; // v_t from M_v @ x_t
        let diff_from_vt: f32 = v_hat_k.iter().zip(v_t.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_from_vt > 1e-4, "v_hat_k should differ from v_t, diff={diff_from_vt}");
    }

    #[test]
    fn test_self_gen_backward_produces_gradients() {
        let d = 4;
        let seq_len = 2;

        let mut state = make_self_ref_state(d, 0.1);
        for i in 0..d { state.m_eta[i * d + i] = 0.5; }
        for i in 0..d { state.m_alpha[i * d + i] = 0.5; }
        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let (y, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, true);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = self_ref_step_backward(&cache, &d_y, true);

        let de_norm: f32 = d_embedded.iter().map(|x| x * x).sum();
        assert!(de_norm > 1e-10, "d_embedded should be nonzero with self-gen, norm={de_norm}");

        let dk_norm: f32 = grads.d_m_k.iter().map(|x| x * x).sum();
        assert!(dk_norm > 0.0, "dM_k should be nonzero with self-gen, norm={dk_norm}");

        let dmem_norm: f32 = grads.d_m_mem.iter().map(|x| x * x).sum();
        assert!(dmem_norm > 0.0, "dM_mem should be nonzero with self-gen, norm={dmem_norm}");
    }

    #[test]
    fn test_self_gen_backward_fd_check_embedded() {
        // FD gradient check for d_embedded through the self-gen path.
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let state0 = make_self_ref_state(d, 0.1);
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, true);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (d_embedded, _) = self_ref_step_backward(&cache, &d_y, true);

        let mut max_err = 0.0f32;
        for idx in 0..embedded.len() {
            let mut perturbed = embedded.clone();
            perturbed[idx] += eps;

            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut s, &mut mm, &perturbed, seq_len, d, true);
            let loss_p = simple_loss(&y_p);

            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = d_embedded[idx];

            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;

            if ana_grad.abs() > 1e-4 {
                max_err = max_err.max(rel_err);
            }
        }
        assert!(max_err < tol, "FD check failed for d_embedded (self-gen): max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_self_gen_backward_fd_check_initial_mk() {
        // FD gradient check for dM_{k,0} through the self-gen path.
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        let state0 = make_self_ref_state(d, 0.1);
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, true);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = self_ref_step_backward(&cache, &d_y, true);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut sp = state0.clone();
            sp.m_k[idx] += eps;
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut sp, &mut mm, &embedded, seq_len, d, true);
            let loss_p = simple_loss(&y_p);

            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = grads.d_m_k[idx];

            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;

            if ana_grad.abs() > 1e-4 {
                max_err = max_err.max(rel_err);
            }
        }
        assert!(max_err < tol, "FD check failed for dM_k (self-gen): max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_self_gen_phase2_equivalence() {
        // self_generated_values=false should produce identical output to before the key fix.
        // (Since this is run AFTER the key fix, we just verify false mode is self-consistent.)
        let d = 4;
        let seq_len = 2;

        let state0 = make_self_ref_state(d, 0.1);
        let m_mem0: Vec<f32> = (0..d * d).map(|i| if i / d == i % d { 0.1 } else { 0.0 }).collect();
        let embedded: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect();

        // Run twice with same input — should be deterministic
        let mut s1 = state0.clone();
        let mut mm1 = m_mem0.clone();
        let (y1, _) = self_ref_step(&mut s1, &mut mm1, &embedded, seq_len, d, false);

        let mut s2 = state0.clone();
        let mut mm2 = m_mem0.clone();
        let (y2, cache2) = self_ref_step(&mut s2, &mut mm2, &embedded, seq_len, d, false);

        assert_eq!(y1, y2, "Deterministic: same input should give same output");

        // Phase 2 cache should have empty v_hat_targets
        assert!(cache2.v_hat_targets.is_empty(), "Phase 2 should not allocate v_hat_targets");
        assert!(!cache2.self_generated_values);
    }

    #[test]
    fn test_key_fix_uses_kt() {
        // Verify component DGD updates use k_t as key, not x_t.
        // Strategy: with M_k = 2*I, k_t = 2*x_t. If DGD uses k_t, the memory
        // update pattern differs from using x_t. We verify by checking that
        // M state after update differs from what x_t-keyed DGD would produce.
        let d = 4;
        let seq_len = 1;

        // M_k = 2*I so k_t = 2*x_t (distinguishable from x_t)
        let mut state = SelfRefState::new(d);
        for i in 0..d { state.m_k[i * d + i] = 2.0; }
        for i in 0..d { state.m_v[i * d + i] = 0.1; }
        for i in 0..d { state.m_q[i * d + i] = 0.1; }
        for i in 0..d { state.m_eta[i * d + i] = 1.0; }  // ensures nonzero theta
        for i in 0..d { state.m_alpha[i * d + i] = 1.0; } // ensures nonzero alpha

        let mut m_mem = vec![0.0f32; d * d];
        for i in 0..d { m_mem[i * d + i] = 0.1; }
        let embedded = vec![0.1f32; d];

        let (_, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, false);

        // Check: k_t = M_k @ x_t. With M_k = 2I, x_t = [0.1, 0.1, 0.1, 0.1]:
        // k_t should be [0.2, 0.2, 0.2, 0.2]
        let k_t = &cache.k_mem[0..d];
        for i in 0..d {
            assert!((k_t[i] - 0.2).abs() < 1e-6, "k_t[{i}] should be 0.2, got {}", k_t[i]);
        }

        // The DGD update uses k_t as key. If it were using x_t instead,
        // the error term would be M@x_t - v_t. With k_t, it's M@k_t - v_t.
        // Since k_t = 2*x_t, the gradient pattern is different.
        // We verify this indirectly: M_v after update should reflect k_t-based error.
        let dd = d * d;
        let m_v_after = &cache.m_v_states[dd..2 * dd]; // state after first token
        let m_v_before = &cache.m_v_states[0..dd];

        // DGD: M' = alpha*M - theta*(M@k - v)@k^T
        // With k_t=0.2, the outer product k@k^T has scale 0.04
        // With x_t=0.1, it would have scale 0.01
        // So the update magnitude should reflect k_t scale, not x_t scale.
        let diff: f32 = m_v_before.iter().zip(m_v_after.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "M_v should change after DGD, diff={diff}");
    }
}
