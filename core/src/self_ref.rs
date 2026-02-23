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
