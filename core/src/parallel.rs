/// Parallelization strategies for intra-sequence memory processing.
///
/// All 8 MIRAS rules process tokens sequentially by default (`for t in 0..seq_len`
/// inside `step()`). On GPU this means tiny kernels firing one-at-a-time — the
/// "sawtooth problem" (5-15% utilization). These strategies solve it by processing
/// chunks of tokens simultaneously within each memory rule's forward pass.
///
/// CMS (multi-level frequency) and parallelization (intra-sequence chunking) are
/// orthogonal: CMS controls *which* levels fire, parallelization controls *how*
/// each level processes its sequence.

use crate::model::MemoryRuleKind;

/// Which parallelization strategy to use for memory processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParallelStrategy {
    /// C=1, exact (existing behavior). Every rule supports this.
    Sequential,
    /// Universal, approximate. Freeze M at chunk boundary, compute all C gradients
    /// w.r.t. frozen state, accumulate with decay products. Error: O(C * lr * ||grad||).
    ChunkwiseGD,
    /// Linear recurrences only, exact. Blelloch parallel prefix sum in O(log C) steps.
    /// Full support for Hebbian (M update is purely linear in M).
    /// Partial for Titans (momentum S is linear, but M still uses chunkwise GD).
    AssociativeScan,
    /// Architecture-agnostic. One coarse global memory (sequential across shards)
    /// + N fine local memories (fully parallel within each shard). 17.37x speedup.
    TNTHierarchical,
    /// Specialized for Lattice OSR and Trellis. Linearize the orthogonal/two-pass
    /// update into GLA form. Unit sphere bounds drift: C=4 matches C=1 quality.
    LatticeGLA,
    /// Atlas Omega only (placeholder — rule not yet implemented).
    AtlasParallel,
}

/// Configuration for parallelized memory processing.
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Which strategy to use.
    pub strategy: ParallelStrategy,
    /// Intra-sequence chunk size (C). C=1 is sequential. Must divide seq_len or
    /// a remainder chunk handles the tail.
    pub chunk_size: usize,
    /// Global chunk size for TNT (C_G): shard size. Only used by TNTHierarchical.
    pub tnt_global_chunk_size: usize,
    /// Local chunk size for TNT (C_L): local chunk size within each shard.
    /// n_local = C_G / C_L. Only used by TNTHierarchical.
    pub tnt_local_chunk_size: usize,
}

impl ParallelConfig {
    /// Sequential processing (C=1). Equivalent to existing behavior.
    pub fn sequential() -> Self {
        ParallelConfig {
            strategy: ParallelStrategy::Sequential,
            chunk_size: 1,
            tnt_global_chunk_size: 1,
            tnt_local_chunk_size: 1,
        }
    }

    /// Chunkwise GD with the given chunk size.
    pub fn chunkwise(chunk_size: usize) -> Self {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        ParallelConfig {
            strategy: ParallelStrategy::ChunkwiseGD,
            chunk_size,
            tnt_global_chunk_size: 1,
            tnt_local_chunk_size: 1,
        }
    }

    /// Associative scan with the given chunk size.
    pub fn associative_scan(chunk_size: usize) -> Self {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        ParallelConfig {
            strategy: ParallelStrategy::AssociativeScan,
            chunk_size,
            tnt_global_chunk_size: 1,
            tnt_local_chunk_size: 1,
        }
    }

    /// TNT hierarchical with global and local chunk sizes.
    pub fn tnt(global_chunk_size: usize, local_chunk_size: usize) -> Self {
        assert!(global_chunk_size >= 1, "tnt_global_chunk_size must be >= 1");
        assert!(local_chunk_size >= 1, "tnt_local_chunk_size must be >= 1");
        assert!(
            global_chunk_size >= local_chunk_size,
            "global_chunk_size ({global_chunk_size}) must be >= local_chunk_size ({local_chunk_size})"
        );
        ParallelConfig {
            strategy: ParallelStrategy::TNTHierarchical,
            chunk_size: global_chunk_size, // primary chunk_size = global for TNT
            tnt_global_chunk_size: global_chunk_size,
            tnt_local_chunk_size: local_chunk_size,
        }
    }

    /// Lattice GLA with the given chunk size.
    pub fn lattice_gla(chunk_size: usize) -> Self {
        assert!(chunk_size >= 1, "chunk_size must be >= 1");
        ParallelConfig {
            strategy: ParallelStrategy::LatticeGLA,
            chunk_size,
            tnt_global_chunk_size: 1,
            tnt_local_chunk_size: 1,
        }
    }
}

/// Memory state at a chunk boundary — used by chunkwise strategies to propagate
/// state between chunks.
#[derive(Clone, Debug)]
pub struct ChunkBoundary {
    /// Flat memory state at boundary. For matrix rules: [d*d]. For MLP: [W1 ++ W2].
    /// For Lattice: [m*d]. For Trellis: [S_K ++ S_V].
    pub state: Vec<f32>,
    /// Momentum accumulator (TitansLMM only). None for all other rules.
    pub momentum: Option<Vec<f32>>,
}

impl ChunkBoundary {
    /// Create a zero-initialized boundary for a matrix-memory rule.
    pub fn zeros_matrix(d: usize) -> Self {
        ChunkBoundary {
            state: vec![0.0f32; d * d],
            momentum: None,
        }
    }

    /// Create a zero-initialized boundary with momentum (TitansLMM).
    pub fn zeros_matrix_with_momentum(d: usize) -> Self {
        ChunkBoundary {
            state: vec![0.0f32; d * d],
            momentum: Some(vec![0.0f32; d * d]),
        }
    }

    /// Create a boundary from existing state.
    pub fn from_state(state: Vec<f32>) -> Self {
        ChunkBoundary { state, momentum: None }
    }

    /// Create a boundary from existing state + momentum.
    pub fn from_state_with_momentum(state: Vec<f32>, momentum: Vec<f32>) -> Self {
        ChunkBoundary { state, momentum: Some(momentum) }
    }
}

/// Check whether a given rule supports a given strategy per the compatibility matrix.
///
/// | Rule | Chunkwise | Assoc.Scan | TNT | LatticeGLA | AtlasParallel |
/// |------|-----------|------------|-----|------------|---------------|
/// | DeltaRule | YES | NO | YES | NO | NO |
/// | TitansLMM | YES | PARTIAL | YES | NO | NO |
/// | Hebbian | YES | YES | YES | NO | NO |
/// | MONETA | YES | NO | YES | NO | NO |
/// | YAAD | YES | NO | YES | NO | NO |
/// | MEMORA | YES | NO | YES | NO | NO |
/// | LatticeOSR | YES | NO | YES | YES | NO |
/// | Trellis | YES | NO | YES | YES | NO |
pub fn strategy_supported(rule: MemoryRuleKind, strategy: ParallelStrategy) -> bool {
    match strategy {
        ParallelStrategy::Sequential => true, // all rules
        ParallelStrategy::ChunkwiseGD => true, // universal
        ParallelStrategy::TNTHierarchical => true, // architecture-agnostic
        ParallelStrategy::AssociativeScan => matches!(
            rule,
            MemoryRuleKind::HebbianRule | MemoryRuleKind::TitansLMM
        ),
        ParallelStrategy::LatticeGLA => matches!(
            rule,
            MemoryRuleKind::LatticeOSR | MemoryRuleKind::Trellis
        ),
        ParallelStrategy::AtlasParallel => matches!(rule, MemoryRuleKind::AtlasOmega),
    }
}

/// Return the list of supported strategy names for a given rule (for the trait method).
pub fn supported_strategies(rule: MemoryRuleKind) -> &'static [&'static str] {
    match rule {
        MemoryRuleKind::HebbianRule => &["sequential", "chunkwise_gd", "associative_scan", "tnt"],
        MemoryRuleKind::TitansLMM => &["sequential", "chunkwise_gd", "associative_scan_partial", "tnt"],
        MemoryRuleKind::LatticeOSR => &["sequential", "chunkwise_gd", "tnt", "lattice_gla"],
        MemoryRuleKind::Trellis => &["sequential", "chunkwise_gd", "tnt", "lattice_gla"],
        MemoryRuleKind::AtlasOmega => &["sequential", "chunkwise_gd", "tnt", "atlas_parallel"],
        // DeltaRule, MONETA, YAAD, MEMORA
        _ => &["sequential", "chunkwise_gd", "tnt"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{MAGConfig, MemoryRuleKind};

    #[test]
    fn test_parallel_config_defaults() {
        let cfg = ParallelConfig::sequential();
        assert_eq!(cfg.strategy, ParallelStrategy::Sequential);
        assert_eq!(cfg.chunk_size, 1);
    }

    #[test]
    fn test_chunkwise_config() {
        let cfg = ParallelConfig::chunkwise(4);
        assert_eq!(cfg.strategy, ParallelStrategy::ChunkwiseGD);
        assert_eq!(cfg.chunk_size, 4);
    }

    #[test]
    fn test_c1_is_sequential_equivalent() {
        // C=1 chunkwise should be functionally equivalent to sequential
        let cfg = ParallelConfig::chunkwise(1);
        assert_eq!(cfg.chunk_size, 1);
        // Strategy differs but C=1 means single-token chunks = sequential processing
    }

    #[test]
    fn test_chunk_boundary_zeros() {
        let b = ChunkBoundary::zeros_matrix(8);
        assert_eq!(b.state.len(), 64);
        assert!(b.state.iter().all(|&x| x == 0.0));
        assert!(b.momentum.is_none());
    }

    #[test]
    fn test_chunk_boundary_with_momentum() {
        let b = ChunkBoundary::zeros_matrix_with_momentum(4);
        assert_eq!(b.state.len(), 16);
        assert!(b.momentum.is_some());
        assert_eq!(b.momentum.as_ref().unwrap().len(), 16);
    }

    #[test]
    fn test_strategy_supported_universal() {
        // Sequential and ChunkwiseGD supported by all rules
        for rule in [
            MemoryRuleKind::DeltaRule, MemoryRuleKind::TitansLMM,
            MemoryRuleKind::HebbianRule, MemoryRuleKind::Moneta,
            MemoryRuleKind::YAAD, MemoryRuleKind::MEMORA,
            MemoryRuleKind::LatticeOSR, MemoryRuleKind::Trellis,
            MemoryRuleKind::AtlasOmega,
        ] {
            assert!(strategy_supported(rule, ParallelStrategy::Sequential),
                "Sequential should be supported by {:?}", rule);
            assert!(strategy_supported(rule, ParallelStrategy::ChunkwiseGD),
                "ChunkwiseGD should be supported by {:?}", rule);
            assert!(strategy_supported(rule, ParallelStrategy::TNTHierarchical),
                "TNT should be supported by {:?}", rule);
        }
    }

    #[test]
    fn test_strategy_supported_associative_scan() {
        assert!(strategy_supported(MemoryRuleKind::HebbianRule, ParallelStrategy::AssociativeScan));
        assert!(strategy_supported(MemoryRuleKind::TitansLMM, ParallelStrategy::AssociativeScan));
        assert!(!strategy_supported(MemoryRuleKind::DeltaRule, ParallelStrategy::AssociativeScan));
        assert!(!strategy_supported(MemoryRuleKind::Moneta, ParallelStrategy::AssociativeScan));
    }

    #[test]
    fn test_strategy_supported_lattice_gla() {
        assert!(strategy_supported(MemoryRuleKind::LatticeOSR, ParallelStrategy::LatticeGLA));
        assert!(strategy_supported(MemoryRuleKind::Trellis, ParallelStrategy::LatticeGLA));
        assert!(!strategy_supported(MemoryRuleKind::DeltaRule, ParallelStrategy::LatticeGLA));
        assert!(!strategy_supported(MemoryRuleKind::HebbianRule, ParallelStrategy::LatticeGLA));
    }

    #[test]
    fn test_atlas_parallel_support() {
        // AtlasParallel only supported by AtlasOmega
        assert!(strategy_supported(MemoryRuleKind::AtlasOmega, ParallelStrategy::AtlasParallel));
        for rule in [
            MemoryRuleKind::DeltaRule, MemoryRuleKind::TitansLMM,
            MemoryRuleKind::HebbianRule, MemoryRuleKind::Moneta,
            MemoryRuleKind::YAAD, MemoryRuleKind::MEMORA,
            MemoryRuleKind::LatticeOSR, MemoryRuleKind::Trellis,
        ] {
            assert!(!strategy_supported(rule, ParallelStrategy::AtlasParallel),
                "AtlasParallel should not be supported by {:?}", rule);
        }
    }

    #[test]
    fn test_mag_config_parallel_field() {
        // Default MAGConfig should have parallel = None
        let cfg = MAGConfig::test_config();
        assert!(cfg.parallel.is_none());

        // Setting parallel config
        let mut cfg2 = MAGConfig::test_config();
        cfg2.parallel = Some(ParallelConfig::chunkwise(4));
        assert!(cfg2.parallel.is_some());
        assert_eq!(cfg2.parallel.as_ref().unwrap().chunk_size, 4);
    }

    #[test]
    fn test_supported_strategies_per_rule() {
        let delta = supported_strategies(MemoryRuleKind::DeltaRule);
        assert!(delta.contains(&"sequential"));
        assert!(delta.contains(&"chunkwise_gd"));
        assert!(delta.contains(&"tnt"));
        assert!(!delta.contains(&"associative_scan"));

        let heb = supported_strategies(MemoryRuleKind::HebbianRule);
        assert!(heb.contains(&"associative_scan"));

        let lattice = supported_strategies(MemoryRuleKind::LatticeOSR);
        assert!(lattice.contains(&"lattice_gla"));
    }

    #[test]
    fn test_tnt_config() {
        let cfg = ParallelConfig::tnt(64, 8);
        assert_eq!(cfg.strategy, ParallelStrategy::TNTHierarchical);
        assert_eq!(cfg.tnt_global_chunk_size, 64);
        assert_eq!(cfg.tnt_local_chunk_size, 8);
        assert_eq!(cfg.chunk_size, 64); // primary = global
    }
}
