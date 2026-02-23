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
