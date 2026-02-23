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
) -> (Vec<f32>, SelfRefCache) {
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(m_mem.len(), d * d);
    debug_assert!(self_ref.is_active(), "self_ref_step called on empty SelfRefState");

    let dd = d * d;
    let mut cache = SelfRefCache::new(seq_len, d);
    cache.embedded = embedded.to_vec();

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

        // Step 4: DGD update all 6 memories (Eq 88)
        // Component memories: keyed by x_t, target v_t
        dgd_step(&mut self_ref.m_k, x_t, &v_t, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_v, x_t, &v_t, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_q, x_t, &v_t, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_eta, x_t, &v_t, alpha_t, theta_t, d);
        dgd_step(&mut self_ref.m_alpha, x_t, &v_t, alpha_t, theta_t, d);
        // Main memory: keyed by adaptive k_t, target v_t
        dgd_step(m_mem, &k_t, &v_t, alpha_t, theta_t, d);

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
pub fn self_ref_read_only(
    self_ref: &SelfRefState,
    m_mem: &[f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
) -> Vec<f32> {
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(m_mem.len(), d * d);

    let mut y = vec![0.0f32; seq_len * d];
    let mut q_t = vec![0.0f32; d];
    let mut y_t = vec![0.0f32; d];

    for t in 0..seq_len {
        let x_t = &embedded[t * d..(t + 1) * d];
        // Read q from frozen M_q
        matmul_f32(&self_ref.m_q, x_t, &mut q_t, d, d, 1);
        // Read from frozen M_mem
        matmul_f32(m_mem, &q_t, &mut y_t, d, d, 1);
        y[t * d..(t + 1) * d].copy_from_slice(&y_t);
    }

    y
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
        // y_t = M_mem_t @ q_t (note: read uses state BEFORE DGD update at this t,
        // but our forward does read then update, so M_mem_t is the correct state)
        //
        // Wait: actually the forward reads from the M_mem BEFORE the DGD update at time t,
        // but the DGD updates happen AFTER the read. So we need M_mem at the state
        // that was used for reading. That IS m_mem_t (the state at time t before update).
        //
        // However, the dm_mem we're carrying is dL/dM_{mem,t+1}. We need to add the
        // read gradient to it, then flow through DGD backward.

        // dL/dM_mem_read = outer(dy_t, q_t) — gradient from the read y_t = M @ q_t
        // This adds to dm_mem (which already has dL/dM_{t+1} from the next token's DGD backward)
        for i in 0..d {
            for j in 0..d {
                dm_mem[i * d + j] += dy_t[i] * q_t[j];
            }
        }

        // dq_t from read: M_mem_t^T @ dy_t
        // (transposed matmul: q contributes to y via M @ q, so dq = M^T @ dy)
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_mem_t[i * d + j] * dy_t[i];
            }
            dq_t[j] = sum;
        }

        // ── Step 2: DGD backward for main memory ──
        // Forward: M_{mem,t+1} = dgd_step(M_{mem,t}, k_t, v_t, alpha_t, theta_t)
        // dm_mem currently holds dL/dM_{mem,t+1}
        let main_grads = crate::dgd::dgd_step_backward(&dm_mem, m_mem_t, k_t, v_t, alpha_t, theta_t, d);
        dm_mem.copy_from_slice(&main_grads.d_m);
        let mut dk_t_from_main = main_grads.d_k;
        let mut dv_t_total = main_grads.d_v;
        let mut dalpha_total = main_grads.d_alpha;
        let mut dtheta_total = main_grads.d_theta;

        // ── Step 3: DGD backward for all 5 component memories ──
        // All use key=x_t, value=v_t
        // M_alpha
        let g = crate::dgd::dgd_step_backward(&dm_alpha, m_alpha_t, x_t, v_t, alpha_t, theta_t, d);
        dm_alpha.copy_from_slice(&g.d_m);
        let mut dx_t = g.d_k; // dk for component = dx contribution
        for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;

        // M_eta
        let g = crate::dgd::dgd_step_backward(&dm_eta, m_eta_t, x_t, v_t, alpha_t, theta_t, d);
        dm_eta.copy_from_slice(&g.d_m);
        for i in 0..d { dx_t[i] += g.d_k[i]; }
        for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;

        // M_q
        let g = crate::dgd::dgd_step_backward(&dm_q, m_q_t, x_t, v_t, alpha_t, theta_t, d);
        dm_q.copy_from_slice(&g.d_m);
        for i in 0..d { dx_t[i] += g.d_k[i]; }
        for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;

        // M_v
        let g = crate::dgd::dgd_step_backward(&dm_v, m_v_t, x_t, v_t, alpha_t, theta_t, d);
        dm_v.copy_from_slice(&g.d_m);
        for i in 0..d { dx_t[i] += g.d_k[i]; }
        for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;

        // M_k
        let g = crate::dgd::dgd_step_backward(&dm_k, m_k_t, x_t, v_t, alpha_t, theta_t, d);
        dm_k.copy_from_slice(&g.d_m);
        for i in 0..d { dx_t[i] += g.d_k[i]; }
        for i in 0..d { dv_t_total[i] += g.d_v[i]; }
        dalpha_total += g.d_alpha;
        dtheta_total += g.d_theta;

        // ── Step 4: Gate backward ──
        // alpha_t = sigmoid(mean(alpha_raw_t))
        // theta_t = softplus(mean(eta_raw_t))
        let alpha_raw_t = &cache.alpha_raw[t * d..(t + 1) * d];
        let eta_raw_t = &cache.eta_raw[t * d..(t + 1) * d];
        let alpha_mean: f32 = alpha_raw_t.iter().sum::<f32>() / d as f32;
        let eta_mean: f32 = eta_raw_t.iter().sum::<f32>() / d as f32;

        // dalpha_total → dalpha_mean = dalpha_total * sigmoid'(alpha_mean)
        let dalpha_mean = dalpha_total * sigmoid_backward(alpha_mean);
        // dtheta_total → deta_mean = dtheta_total * softplus'(eta_mean) = dtheta_total * sigmoid(eta_mean)
        let deta_mean = dtheta_total * softplus_backward(eta_mean);

        // d_alpha_raw[i] = dalpha_mean / d (from mean reduction)
        // d_eta_raw[i] = deta_mean / d
        let dalpha_per_dim = dalpha_mean / d as f32;
        let deta_per_dim = deta_mean / d as f32;

        // ── Step 5: Component read backward ──
        // k_t = M_k_t @ x_t → dk from main memory contributes to dM_k_read and dx
        // v_t = M_v_t @ x_t → dv_total contributes to dM_v_read and dx
        // q_t = M_q_t @ x_t → dq_t contributes to dM_q_read and dx
        // eta_raw = M_eta_t @ x_t → deta_per_dim per element contributes to dM_eta_read and dx
        // alpha_raw = M_alpha_t @ x_t → dalpha_per_dim per element contributes to dM_alpha_read and dx

        // dk_t_from_main flows back through k_t = M_k_t @ x_t
        // dM_k += outer(dk_t_from_main, x_t)
        // dx += M_k_t^T @ dk_t_from_main
        for i in 0..d {
            for j in 0..d {
                dm_k[i * d + j] += dk_t_from_main[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_k_t[i * d + j] * dk_t_from_main[i];
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

        let (y, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d);
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

        let (_, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d);

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

        let y = self_ref_read_only(&state, &m_mem, &embedded, seq_len, d);
        assert_eq!(y.len(), seq_len * d);

        // State should be unchanged (frozen)
        assert_eq!(state.m_q, state_before.m_q);
        assert_eq!(state.m_k, state_before.m_k);
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

        let (y, cache) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = self_ref_step_backward(&cache, &d_y);

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
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (d_embedded, _) = self_ref_step_backward(&cache, &d_y);

        // FD check for each element of embedded
        let mut max_err = 0.0f32;
        for idx in 0..embedded.len() {
            let mut perturbed = embedded.clone();
            perturbed[idx] += eps;

            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut s, &mut mm, &perturbed, seq_len, d);
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
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = self_ref_step_backward(&cache, &d_y);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut sp = state0.clone();
            sp.m_k[idx] += eps;
            let mut mm = m_mem0.clone();
            let (y_p, _) = self_ref_step(&mut sp, &mut mm, &embedded, seq_len, d);
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
        let (y, cache) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = self_ref_step_backward(&cache, &d_y);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            mm[idx] += eps;
            let (y_p, _) = self_ref_step(&mut s, &mut mm, &embedded, seq_len, d);
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
        let (y1, _) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d);

        // Second call with same input but mutated state
        let (y2, _) = self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d);

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
}
