/// Compile-time marker traits and runtime composition validation.
///
/// Implements Committee Finding 3 from specs/constraints/trait_system/00_valid_compositions.md:
/// "Don't let the interface lie about orthogonality. Use Rust's type system to enforce
/// the composition matrix. Make invalid models impossible to compile."
///
/// Four marker traits classify what each memory rule provides:
/// - ProbabilitySimplex: memory state lives on the probability simplex
/// - UnitSphere: memory state lives on the unit sphere
/// - LinearRecurrence: memory update is a linear recurrence (scan-parallelizable)
/// - StateIndependentMomentum: momentum depends only on current gradient
///
/// The MemoryRuleBuilder validates MIRAS 4-knob combinations against the constraint
/// matrices from the spec, producing compile-friendly error messages for invalid pairings.

use crate::model::{MemoryRuleKind, CompositionKind};
use crate::retention::RetentionKind;
use crate::parallel::ParallelStrategy;

// ── Marker Traits (Axis Classification) ─────────────────────────────

/// Memory state lives on the probability simplex: non-negative, rows sum to 1.
/// Implies KL retention is valid. Required for KL attentional bias.
/// Implementors: MEMORA.
pub trait ProbabilitySimplex {}

/// Memory state lives on the unit sphere: ||M_slot|| = 1.
/// Implies sphere normalization retention is valid.
/// Implementors: LatticeOSR.
pub trait UnitSphere {}

/// Memory update is a linear recurrence in M (or momentum S).
/// Enables exact parallelization via associative scan (Blelloch prefix sum).
/// Implementors: HebbianRule (M update), TitansLMM (S update only).
pub trait LinearRecurrence {}

/// Momentum accumulator depends only on current gradient, not accumulated state.
/// Enables Atlas parallel (state-independent Omega rule).
/// Implementors: AtlasOmega.
pub trait StateIndependentMomentum {}

// ── Marker Trait Implementations ────────────────────────────────────

// MEMORA: KL retention + KL bias, operates on probability simplex
impl ProbabilitySimplex for crate::memora::MEMORA {}

// LatticeOSR: sphere normalization retention, unit sphere manifold
impl UnitSphere for crate::lattice_osr::LatticeOSR {}

// HebbianRule: M_t = (1-alpha)*M_{t-1} + v@k^T is linear in M
impl LinearRecurrence for crate::hebbian_rule::HebbianRule {}

// TitansLMM: momentum S_t = beta*S_{t-1} + grad is linear in S
// (M update is still nonlinear, so only momentum is scan-parallelizable)
impl LinearRecurrence for crate::titans_lmm::TitansLMM {}

// AtlasOmega: Newton-Schulz momentum depends only on current gradient
impl StateIndependentMomentum for crate::atlas_omega::AtlasOmega {}

// ── Composition Validation ──────────────────────────────────────────

/// Error describing why a MIRAS 4-knob combination is invalid.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositionError {
    pub axis_a: &'static str,
    pub axis_b: &'static str,
    pub reason: String,
}

impl std::fmt::Display for CompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid composition: {} × {} — {}", self.axis_a, self.axis_b, self.reason)
    }
}

/// Validates a MIRAS configuration against the constraint matrices.
///
/// Returns Ok(()) for valid combinations, Err with all violations for invalid ones.
/// This is the runtime counterpart to the compile-time marker traits —
/// it catches violations that can't be expressed as trait bounds because
/// the configuration is determined at runtime via enums.
pub fn validate_composition(
    rule: MemoryRuleKind,
    retention: RetentionKind,
    _composition: CompositionKind, // reserved for PS-TC-02: MAC/MAG/MAL constraints
    parallel: Option<ParallelStrategy>,
) -> Result<(), Vec<CompositionError>> {
    let mut errors = Vec::new();

    // ── Memory Structure × Retention ────────────────────────────────
    // MLP rules + KL/ElasticNet/Sphere: not validated (MIRAS Table 2)
    let is_mlp = matches!(rule, MemoryRuleKind::Moneta | MemoryRuleKind::YAAD | MemoryRuleKind::MEMORA);
    if is_mlp && matches!(retention, RetentionKind::SphereNormalization) {
        errors.push(CompositionError {
            axis_a: "MLP memory structure",
            axis_b: "SphereNormalization retention",
            reason: "sphere normalization assumes M is a tensor, not a network. \
                     Applying to MLP weights changes semantics (regularizing weights \
                     vs regularizing memory state).".into(),
        });
    }
    // KL retention requires probability simplex — only MEMORA maintains that invariant.
    // LatticeOSR gets a specific error (manifold mismatch); all others get a general error.
    if matches!(retention, RetentionKind::KLDivergence) && !matches!(rule, MemoryRuleKind::MEMORA) {
        if matches!(rule, MemoryRuleKind::LatticeOSR) {
            errors.push(CompositionError {
                axis_a: "SphereNormalization (LatticeOSR)",
                axis_b: "KLDivergence retention",
                reason: "unit sphere and probability simplex are different manifolds. \
                         SphereNormalization projects to ||M||=1, KL requires non-negative \
                         rows summing to 1.".into(),
            });
        } else {
            errors.push(CompositionError {
                axis_a: rule_name(rule),
                axis_b: "KLDivergence retention",
                reason: "KL retention requires probability simplex state. \
                         Only MEMORA maintains simplex invariant.".into(),
            });
        }
    }

    // ── Memory Algorithm × Parallelization ──────────────────────────
    if let Some(strategy) = parallel {
        validate_parallel(rule, strategy, &mut errors);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate memory rule × parallelization strategy.
fn validate_parallel(
    rule: MemoryRuleKind,
    strategy: ParallelStrategy,
    errors: &mut Vec<CompositionError>,
) {
    match strategy {
        ParallelStrategy::Sequential | ParallelStrategy::ChunkwiseGD => {
            // All rules support sequential and chunkwise GD
        }
        ParallelStrategy::TNTHierarchical => {
            // All rules support TNT (it's architecture-agnostic)
        }
        ParallelStrategy::AssociativeScan => {
            // Requires linear recurrence. Only Hebbian (full) and Titans (momentum only).
            let has_linear = matches!(
                rule,
                MemoryRuleKind::HebbianRule | MemoryRuleKind::TitansLMM
            );
            if !has_linear {
                errors.push(CompositionError {
                    axis_a: rule_name(rule),
                    axis_b: "AssociativeScan",
                    reason: format!(
                        "{} does not implement LinearRecurrence. \
                         AssociativeScan requires M (or S) to be a linear recurrence.",
                        rule_name(rule)
                    ),
                });
            }
        }
        ParallelStrategy::LatticeGLA => {
            // Requires the update to be expressible as linear scan with decay matrix.
            // GD + L2 retention (linear in M), FTRL for certain regularizers,
            // Lattice OSR, Trellis.
            let supports_gla = matches!(
                rule,
                MemoryRuleKind::DeltaRule
                | MemoryRuleKind::LatticeOSR
                | MemoryRuleKind::Trellis
            );
            if !supports_gla {
                errors.push(CompositionError {
                    axis_a: rule_name(rule),
                    axis_b: "LatticeGLA",
                    reason: format!(
                        "{} update cannot be linearized into GLA form. \
                         LatticeGLA requires M_{{t+1}} = decay * M_t + write_t (linear in M).",
                        rule_name(rule)
                    ),
                });
            }
        }
        ParallelStrategy::AtlasParallel => {
            // Requires state-independent momentum (Newton-Schulz / Omega rule).
            if !matches!(rule, MemoryRuleKind::AtlasOmega) {
                errors.push(CompositionError {
                    axis_a: rule_name(rule),
                    axis_b: "AtlasParallel",
                    reason: format!(
                        "{} does not implement StateIndependentMomentum. \
                         AtlasParallel requires momentum that depends only on \
                         current gradient, not accumulated state.",
                        rule_name(rule)
                    ),
                });
            }
        }
    }
}

/// Human-readable rule name for error messages.
fn rule_name(rule: MemoryRuleKind) -> &'static str {
    match rule {
        MemoryRuleKind::DeltaRule => "DeltaRule",
        MemoryRuleKind::TitansLMM => "TitansLMM",
        MemoryRuleKind::HebbianRule => "HebbianRule",
        MemoryRuleKind::Moneta => "Moneta",
        MemoryRuleKind::YAAD => "YAAD",
        MemoryRuleKind::MEMORA => "MEMORA",
        MemoryRuleKind::LatticeOSR => "LatticeOSR",
        MemoryRuleKind::Trellis => "Trellis",
        MemoryRuleKind::AtlasOmega => "AtlasOmega",
    }
}

// ── MemoryRuleBuilder ───────────────────────────────────────────────

/// Builder for validated MIRAS configurations.
///
/// Collects the 4 knobs + composition + parallelization, then validates
/// all constraint matrix cells before returning the configuration.
///
/// ```ignore
/// let cfg = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule)
///     .retention(RetentionKind::L2WeightDecay)
///     .composition(CompositionKind::MAG)
///     .parallel(ParallelStrategy::ChunkwiseGD)
///     .build()?;
/// ```
pub struct MemoryRuleBuilder {
    rule: MemoryRuleKind,
    retention: Option<RetentionKind>,
    composition: Option<CompositionKind>,
    parallel: Option<ParallelStrategy>,
}

impl MemoryRuleBuilder {
    pub fn new(rule: MemoryRuleKind) -> Self {
        MemoryRuleBuilder {
            rule,
            retention: None,
            composition: None,
            parallel: None,
        }
    }

    pub fn retention(mut self, r: RetentionKind) -> Self {
        self.retention = Some(r);
        self
    }

    pub fn composition(mut self, c: CompositionKind) -> Self {
        self.composition = Some(c);
        self
    }

    pub fn parallel(mut self, p: ParallelStrategy) -> Self {
        self.parallel = Some(p);
        self
    }

    /// Validate the configuration and return the components if valid.
    pub fn build(self) -> Result<ValidatedComposition, Vec<CompositionError>> {
        let retention = self.retention.unwrap_or_else(||
            crate::retention::default_retention(self.rule)
        );
        let composition = self.composition.unwrap_or(CompositionKind::MAG);

        validate_composition(self.rule, retention, composition, self.parallel)?;

        Ok(ValidatedComposition {
            rule: self.rule,
            retention,
            composition,
            parallel: self.parallel,
        })
    }
}

/// A MIRAS configuration that has passed constraint validation.
#[derive(Debug, Clone)]
pub struct ValidatedComposition {
    pub rule: MemoryRuleKind,
    pub retention: RetentionKind,
    pub composition: CompositionKind,
    pub parallel: Option<ParallelStrategy>,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Valid paper configurations (all should pass) ────────────────

    #[test]
    fn test_titans_mac_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAC)
            .parallel(ParallelStrategy::ChunkwiseGD)
            .build();
        assert!(result.is_ok(), "Titans-MAC should be valid: {:?}", result.err());
    }

    #[test]
    fn test_titans_mag_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::ChunkwiseGD)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_atlas_mag_omega_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::AtlasOmega)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AtlasParallel)
            .build();
        assert!(result.is_ok(), "Atlas-MAG-Omega should be valid: {:?}", result.err());
    }

    #[test]
    fn test_memora_kl_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::MEMORA)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_ok(), "MEMORA+KL should be valid: {:?}", result.err());
    }

    #[test]
    fn test_lattice_sphere_gla_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::LatticeOSR)
            .retention(RetentionKind::SphereNormalization)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::LatticeGLA)
            .build();
        assert!(result.is_ok(), "Lattice+Sphere+GLA should be valid: {:?}", result.err());
    }

    #[test]
    fn test_delta_rule_sequential_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_hebbian_assoc_scan_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::HebbianRule)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AssociativeScan)
            .build();
        assert!(result.is_ok(), "Hebbian+AssocScan should be valid: {:?}", result.err());
    }

    #[test]
    fn test_titans_assoc_scan_valid() {
        // Titans momentum S is linear — AssocScan valid (for momentum)
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AssociativeScan)
            .build();
        assert!(result.is_ok(), "Titans+AssocScan should be valid: {:?}", result.err());
    }

    #[test]
    fn test_trellis_valid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Trellis)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::ChunkwiseGD)
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_rules_accept_chunkwise() {
        for rule in [
            MemoryRuleKind::DeltaRule, MemoryRuleKind::TitansLMM,
            MemoryRuleKind::HebbianRule, MemoryRuleKind::Moneta,
            MemoryRuleKind::YAAD, MemoryRuleKind::MEMORA,
            MemoryRuleKind::LatticeOSR, MemoryRuleKind::Trellis,
            MemoryRuleKind::AtlasOmega,
        ] {
            let retention = crate::retention::default_retention(rule);
            let result = validate_composition(
                rule, retention, CompositionKind::MAG,
                Some(ParallelStrategy::ChunkwiseGD),
            );
            assert!(result.is_ok(), "{:?} should accept ChunkwiseGD: {:?}", rule, result.err());
        }
    }

    // ── Invalid combinations (all should fail) ─────────────────────

    #[test]
    fn test_lattice_kl_invalid() {
        // Sphere manifold ≠ probability simplex
        let result = MemoryRuleBuilder::new(MemoryRuleKind::LatticeOSR)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "LatticeOSR+KL should be invalid");
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.reason.contains("manifold")));
    }

    #[test]
    fn test_moneta_kl_invalid() {
        // Moneta is MLP but not on simplex — KL retention is invalid
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Moneta)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Moneta+KL should be invalid");
    }

    #[test]
    fn test_mlp_sphere_invalid() {
        // MLP + sphere normalization: different semantics
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Moneta)
            .retention(RetentionKind::SphereNormalization)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Moneta+Sphere should be invalid");
    }

    #[test]
    fn test_delta_assoc_scan_invalid() {
        // DeltaRule is not a linear recurrence — no AssocScan
        let result = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AssociativeScan)
            .build();
        assert!(result.is_err(), "Delta+AssocScan should be invalid");
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.reason.contains("LinearRecurrence")));
    }

    #[test]
    fn test_moneta_assoc_scan_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Moneta)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AssociativeScan)
            .build();
        assert!(result.is_err(), "Moneta+AssocScan should be invalid");
    }

    #[test]
    fn test_delta_atlas_parallel_invalid() {
        // DeltaRule doesn't have state-independent momentum
        let result = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AtlasParallel)
            .build();
        assert!(result.is_err(), "Delta+AtlasParallel should be invalid");
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.reason.contains("StateIndependentMomentum")));
    }

    #[test]
    fn test_titans_atlas_parallel_invalid() {
        // Titans momentum is state-dependent (S = beta*S + grad depends on S)
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::AtlasParallel)
            .build();
        assert!(result.is_err(), "Titans+AtlasParallel should be invalid");
    }

    #[test]
    fn test_titans_lattice_gla_invalid() {
        // Titans has momentum recurrence on top of memory — can't linearize both
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::LatticeGLA)
            .build();
        assert!(result.is_err(), "Titans+LatticeGLA should be invalid");
    }

    #[test]
    fn test_delta_kl_invalid() {
        // DeltaRule doesn't maintain simplex — KL retention invalid
        let result = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Delta+KL should be invalid");
    }

    #[test]
    fn test_titans_kl_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::TitansLMM)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Titans+KL should be invalid");
    }

    #[test]
    fn test_hebbian_kl_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::HebbianRule)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Hebbian+KL should be invalid");
    }

    #[test]
    fn test_trellis_kl_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Trellis)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Trellis+KL should be invalid");
    }

    #[test]
    fn test_atlas_kl_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::AtlasOmega)
            .retention(RetentionKind::KLDivergence)
            .composition(CompositionKind::MAG)
            .build();
        assert!(result.is_err(), "Atlas+KL should be invalid");
    }

    #[test]
    fn test_moneta_lattice_gla_invalid() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::Moneta)
            .retention(RetentionKind::L2WeightDecay)
            .composition(CompositionKind::MAG)
            .parallel(ParallelStrategy::LatticeGLA)
            .build();
        assert!(result.is_err(), "Moneta+LatticeGLA should be invalid");
    }

    // ── Builder defaults ────────────────────────────────────────────

    #[test]
    fn test_builder_defaults() {
        // With no explicit retention/composition, uses defaults
        let result = MemoryRuleBuilder::new(MemoryRuleKind::DeltaRule).build();
        assert!(result.is_ok());
        let vc = result.unwrap();
        assert_eq!(vc.retention, RetentionKind::L2WeightDecay);
        assert_eq!(vc.composition, CompositionKind::MAG);
        assert!(vc.parallel.is_none());
    }

    #[test]
    fn test_builder_memora_defaults() {
        let result = MemoryRuleBuilder::new(MemoryRuleKind::MEMORA).build();
        assert!(result.is_ok());
        let vc = result.unwrap();
        // MEMORA defaults to KLDivergence
        assert_eq!(vc.retention, RetentionKind::KLDivergence);
    }

    // ── Marker trait compile-time checks ────────────────────────────

    #[test]
    fn test_memora_is_probability_simplex() {
        // Compile-time: MEMORA implements ProbabilitySimplex
        fn assert_simplex<T: ProbabilitySimplex>() {}
        assert_simplex::<crate::memora::MEMORA>();
    }

    #[test]
    fn test_lattice_is_unit_sphere() {
        fn assert_sphere<T: UnitSphere>() {}
        assert_sphere::<crate::lattice_osr::LatticeOSR>();
    }

    #[test]
    fn test_hebbian_is_linear_recurrence() {
        fn assert_linear<T: LinearRecurrence>() {}
        assert_linear::<crate::hebbian_rule::HebbianRule>();
    }

    #[test]
    fn test_titans_is_linear_recurrence() {
        fn assert_linear<T: LinearRecurrence>() {}
        assert_linear::<crate::titans_lmm::TitansLMM>();
    }

    #[test]
    fn test_atlas_is_state_independent_momentum() {
        fn assert_simo<T: StateIndependentMomentum>() {}
        assert_simo::<crate::atlas_omega::AtlasOmega>();
    }

    // ── Exhaustive constraint matrix coverage ───────────────────────

    #[test]
    fn test_all_rules_all_strategies() {
        // Verify the full constraint matrix from the spec
        use MemoryRuleKind::*;
        use ParallelStrategy::*;

        let expect_valid = vec![
            // (rule, strategy)
            // Sequential: all valid
            (DeltaRule, Sequential), (TitansLMM, Sequential), (HebbianRule, Sequential),
            (Moneta, Sequential), (YAAD, Sequential), (MEMORA, Sequential),
            (LatticeOSR, Sequential), (Trellis, Sequential), (AtlasOmega, Sequential),
            // ChunkwiseGD: all valid
            (DeltaRule, ChunkwiseGD), (TitansLMM, ChunkwiseGD), (HebbianRule, ChunkwiseGD),
            (Moneta, ChunkwiseGD), (YAAD, ChunkwiseGD), (MEMORA, ChunkwiseGD),
            (LatticeOSR, ChunkwiseGD), (Trellis, ChunkwiseGD), (AtlasOmega, ChunkwiseGD),
            // TNTHierarchical: all valid
            (DeltaRule, TNTHierarchical), (TitansLMM, TNTHierarchical),
            (HebbianRule, TNTHierarchical), (Moneta, TNTHierarchical),
            (YAAD, TNTHierarchical), (MEMORA, TNTHierarchical),
            (LatticeOSR, TNTHierarchical), (Trellis, TNTHierarchical),
            (AtlasOmega, TNTHierarchical),
            // AssociativeScan: only linear recurrence rules
            (HebbianRule, AssociativeScan), (TitansLMM, AssociativeScan),
            // LatticeGLA: only linearizable rules
            (DeltaRule, LatticeGLA), (LatticeOSR, LatticeGLA), (Trellis, LatticeGLA),
            // AtlasParallel: only state-independent momentum
            (AtlasOmega, AtlasParallel),
        ];

        for (rule, strategy) in &expect_valid {
            let retention = crate::retention::default_retention(*rule);
            let result = validate_composition(*rule, retention, CompositionKind::MAG, Some(*strategy));
            assert!(result.is_ok(),
                "{:?} × {:?} should be VALID but got: {:?}",
                rule, strategy, result.err()
            );
        }

        let expect_invalid = vec![
            // AssociativeScan: non-linear rules
            (DeltaRule, AssociativeScan), (Moneta, AssociativeScan),
            (YAAD, AssociativeScan), (MEMORA, AssociativeScan),
            (LatticeOSR, AssociativeScan), (Trellis, AssociativeScan),
            (AtlasOmega, AssociativeScan),
            // LatticeGLA: non-linearizable rules
            (TitansLMM, LatticeGLA), (HebbianRule, LatticeGLA),
            (Moneta, LatticeGLA), (YAAD, LatticeGLA),
            (MEMORA, LatticeGLA), (AtlasOmega, LatticeGLA),
            // AtlasParallel: state-dependent momentum
            (DeltaRule, AtlasParallel), (TitansLMM, AtlasParallel),
            (HebbianRule, AtlasParallel), (Moneta, AtlasParallel),
            (YAAD, AtlasParallel), (MEMORA, AtlasParallel),
            (LatticeOSR, AtlasParallel), (Trellis, AtlasParallel),
        ];

        for (rule, strategy) in &expect_invalid {
            let retention = crate::retention::default_retention(*rule);
            let result = validate_composition(*rule, retention, CompositionKind::MAG, Some(*strategy));
            assert!(result.is_err(),
                "{:?} × {:?} should be INVALID but passed validation",
                rule, strategy
            );
        }
    }
}
