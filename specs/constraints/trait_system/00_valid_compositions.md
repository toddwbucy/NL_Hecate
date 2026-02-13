# Valid Composition Pairings (Trait System)

```
CONTRACT
  Purpose:    Not all combinations of memory update rules, composition patterns,
              retention mechanisms, and parallelization strategies are valid.
              The Rust trait system enforces valid pairings at COMPILE TIME.
              This spec catalogs what composes with what, and why invalid
              pairings are invalid.
  Expects:    The 4-knob MIRAS framework (memory structure, attentional bias,
              retention mechanism, memory algorithm).
              The 3 composition patterns (MAC, MAG, MAL).
              The 5 parallelization strategies.
  Guarantees: Invalid combinations produce compile errors, not runtime bugs.
              Valid combinations produce correct gradient flow.
              The constraint matrix below is exhaustive for known paper variants.
  Cost:       Compile-time only. Zero runtime cost for constraint checking.
  Trade-off:  Rigid constraints prevent exploration of untested combinations.
              But: untested combinations have unknown gradient properties.
              The trait system makes the boundary explicit: tested = compiles,
              untested = doesn't compile until validated and trait bounds added.
  Position:   specs/constraints/trait_system/00_valid_compositions.md
  Source:     MIRAS Table 2; paper-specific variant definitions;
              Atlas Eq 26 (generalization proof); Lattice Proposition 3.1
```

## Enforcement Mechanism (Committee Finding 3)

```
-- Committee Finding: "Don't let the interface lie about orthogonality.
-- Use Rust's type system to enforce the composition matrix.
-- Make invalid models impossible to compile."

-- This document defines the RULES. The ENFORCEMENT lives in the type system:
--   1. Marker traits: ProbabilitySimplex, UnitSphere, LinearRecurrence,
--      StateIndependentMomentum — classify what each component provides/requires.
--   2. Associated types: each trait has type RequiredManifold, type UpdateOp, etc.
--      The compiler checks compatibility between associated types.
--   3. Builder pattern: MemoryRuleBuilder prevents direct instantiation.
--      Generic bounds on builder methods reject invalid combinations at compile time.

-- See memory_update_rules/00_interface.md "Compile-Time Composition Safety"
-- for the full type-level enforcement specification.

-- Every "NO" cell in the matrices below corresponds to a compile error.
-- Every "YES" cell corresponds to a valid trait bound resolution.
-- Untested-but-valid cells compile but log a warning.
```

## The Four Trait Axes

```
AXIS 1: MemoryStructure (Knob #1)
  -- Vector:  M is [d_v] — simplest, one-dimensional
  -- Matrix:  M is [d_k, d_v] — standard, supports outer product
  -- MLP:     M is a multi-layer network — richest, supports backprop

AXIS 2: AttentionalBias (Knob #2)
  -- L2:           ||M @ k - v||^2 — standard squared error
  -- DotProduct:   -k^T @ M @ v — associative, no error signal
  -- Huber:        Huber(M @ k - v) — bounded gradient
  -- LpNorm:       ||M @ k - v||_p^p — parameterized norm
  -- KLDivergence: KL(softmax(M @ k) || softmax(v)) — probability space

AXIS 3: RetentionMechanism (Knob #3)
  -- L2WeightDecay:      lambda * M — exponential forgetting
  -- KLDivergence:       KL(M || M_ref) — simplex constraint
  -- ElasticNet:         alpha * ||M||_1 + (1-alpha) * ||M||_2^2 — sparse
  -- FDivergence:        general f-divergence framework
  -- SphereNormalization: M / ||M|| — implicit via projection

AXIS 4: MemoryAlgorithm (Knob #4)
  -- GradientDescent:    M = M - eta * grad
  -- GDWithMomentum:     S = beta * S + grad; M = M - eta * S
  -- NewtonSchulz:       S updated via Newton-Schulz iteration
  -- FTRL:               Follow The Regularized Leader
  -- OnlineMirrorDescent: gradient step on manifold (Lattice)
```

## Constraint Matrix: Memory Structure x Attentional Bias

```
                    L2     DotProduct   Huber    LpNorm   KL
Vector              YES    YES          YES      YES      YES*
Matrix              YES    YES          YES      YES      YES
MLP                 YES    NO**         YES      YES      NO**

* Vector + KL: requires softmax over vector elements (valid but unusual)
** MLP + DotProduct: dot product requires M @ k, but MLP uses forward(k),
   not matrix multiply. The bias must use the MLP's output, not M @ k.
   → Use L2 with MLP output instead: ||MLP(k) - v||^2
** MLP + KL: requires softmax of MLP output, which is valid but
   the "reference distribution" is unclear for an MLP.
```

## Constraint Matrix: Memory Structure x Retention Mechanism

```
                    L2Decay  KL       ElasticNet  FDiv    Sphere
Vector              YES      YES*     YES         YES     YES
Matrix              YES      YES      YES         YES     YES
MLP                 YES**    NO***    NO***       NO***   NO***

* Vector + KL retention: vector must be non-negative (probability simplex)
** MLP + L2 decay: applied to MLP weights, standard weight decay
*** MLP + KL/ElasticNet/FDiv/Sphere: these assume M is a tensor, not a network.
    Applying them to MLP weights is possible but changes the semantics
    (regularizing weights vs. regularizing the memory state).
    → Currently not validated. Requires explicit opt-in if implemented.
```

## Constraint Matrix: Retention x Attentional Bias (Key Couplings)

```
KL retention + KL bias:
  -- VALID and POWERFUL: both operate on the probability simplex
  -- This is MEMORA (Samba-like): softmax naturally emerges
  -- Proven by MIRAS Proposition 3.1

KL retention + L2 bias:
  -- VALID but UNUSUAL: retention constrains to simplex,
  -- but L2 bias doesn't operate on probabilities
  -- May cause tension between bias and retention gradients

Sphere normalization + L2 bias:
  -- VALID: this is Lattice OSR
  -- L2 loss on unit sphere, normalization = implicit forgetting
  -- Proven by Lattice Proposition 3.1 (Riemannian GD)

Sphere normalization + KL bias:
  -- INVALID: sphere normalization projects to unit sphere,
  -- KL requires probability simplex (non-negative, sum to 1)
  -- Unit sphere and probability simplex are different manifolds
  -- → Compile error: SphereNormalization does not impl ProbabilitySimplex
```

## Constraint Matrix: Memory Algorithm x Parallelization

```
                      ChunkwiseGD  AssocScan  TNTHierarchical  LatticeGLA  AtlasParallel
GradientDescent       YES          NO*        YES              YES**       NO***
GDWithMomentum        YES          YES****    YES              NO*****     NO***
NewtonSchulz          YES          NO*        YES              NO*****     YES
FTRL                  YES          NO*        YES              YES**       NO***
OnlineMirrorDescent   YES          NO*        YES              YES         NO***

*   AssocScan requires LINEAR recurrence. GD without momentum is not a
    recurrence (each step is independent given the gradient). Only momentum
    (S = beta * S + grad) creates a linear recurrence suitable for scan.

**  LatticeGLA requires the update to be expressible as a linear scan
    with a decay matrix. GD with L2 retention satisfies this:
    M_{t+1} = lambda * M_t - eta * grad_t (linear in M).
    FTRL also satisfies this for certain regularizers.

*** AtlasParallel requires state-independent momentum (Omega rule).
    Only NewtonSchulz momentum is state-independent.
    Standard momentum S = beta * S + grad is state-DEPENDENT (on S).
    Atlas Eq 14: the Newton-Schulz iteration depends only on the
    CURRENT gradient, not on S's history. This is what makes it parallel.

**** Momentum + AssocScan: S_t = beta * S_{t-1} + grad_t IS linear.
     This is the only algorithm + parallelization pair that uses
     exact parallel prefix computation.

***** LatticeGLA + Momentum: momentum adds a second recurrence on top
      of the memory recurrence. LatticeGLA linearizes ONE recurrence,
      not two. Combining them requires either:
      (a) Fusing momentum into the memory update (Atlas approach)
      (b) Running momentum sequentially (loses GLA's parallelism)
      → Currently not validated as a pair.

SPECIAL CASE: Hebbian Rule
  Hebbian has no algorithm (no gradient, no optimization step).
  Its memory update is DIRECTLY a linear recurrence:
    M_t = (1 - alpha_t) * M_{t-1} + v_t @ k_t^T
  This fits the associative scan form (a_t = 1-alpha_t, b_t = v@k^T).
  Hebbian supports: ChunkwiseGD (trivially), AssociativeScan (YES),
  TNTHierarchical (YES). It is the only rule where the full memory
  update (not just momentum) is parallelizable via associative scan.
```

## Composition Pattern Constraints

```
MAC (Memory As Context):
  -- Requires: attention module (full causal, not SWA)
  -- Memory reads produce tokens prepended to attention's KV
  -- No constraint on memory update rule or retention
  -- The reflective gate (Eq 25) is MAC-specific

MAG (Memory As Gate):
  -- Requires: memory output in [0, 1] (sigmoid gate)
  -- Retention mechanism must preserve [0, 1] range OR
  --   the gate sigmoid must be applied AFTER retention
  -- Uses SWA (not full causal)

MAL (Memory As Layer):
  -- Requires: memory output has same dimensionality as attention input
  -- The memory preprocesses input BEFORE attention sees it
  -- Uses SWA (not full causal)
  -- Information bottleneck: memory output is the ONLY thing attention sees

All patterns:
  -- Any memory update rule can plug into any composition pattern
  -- The constraint is on the OUTPUT FORMAT, not the internal mechanism
  -- MAC: output = list of key-value tokens (variable length)
  -- MAG: output = gate tensor in [0, 1]
  -- MAL: output = processed tensor (same shape as input)
```

## Named Configurations (Paper-Validated)

```
CONFIGURATION: Titans-MAC
  Structure: Matrix
  Bias: L2
  Retention: L2WeightDecay
  Algorithm: GDWithMomentum
  Parallelization: ChunkwiseGD
  Composition: MAC
  STATUS: Paper-validated (Titans 2501.00663)

CONFIGURATION: Titans-MAG
  Structure: Matrix
  Bias: L2
  Retention: L2WeightDecay
  Algorithm: GDWithMomentum
  Parallelization: ChunkwiseGD
  Composition: MAG
  STATUS: Paper-validated (Titans 2501.00663)

CONFIGURATION: Atlas-MAG-Omega
  Structure: Matrix
  Bias: L2
  Retention: L2WeightDecay
  Algorithm: NewtonSchulz
  Parallelization: AtlasParallel
  Composition: MAG
  STATUS: Paper-validated (Atlas 2505.23735)

CONFIGURATION: MONETA
  Structure: MLP (2-layer)
  Bias: LpNorm
  Retention: LpNorm (L_q)
  Algorithm: GradientDescent (backprop through MLP)
  Parallelization: ChunkwiseGD
  Composition: MAG
  STATUS: Paper-validated (MIRAS 2504.13173)

CONFIGURATION: YAAD
  Structure: MLP (2-layer)
  Bias: Huber
  Retention: L2WeightDecay (decoupled: local + global)
  Algorithm: GradientDescent
  Parallelization: ChunkwiseGD
  Composition: MAG
  STATUS: Paper-validated (MIRAS 2504.13173)

CONFIGURATION: MEMORA (Samba)
  Structure: MLP (2-layer, probability simplex)
  Bias: L2 (||MLP(k) - v||^2)
  Retention: KLDivergence
  Algorithm: GradientDescent (with closed-form KL solution)
  Parallelization: ChunkwiseGD
  Composition: MAG
  STATUS: Paper-validated (MIRAS 2504.13173)

CONFIGURATION: Lattice-Decode
  Structure: Matrix
  Bias: L2
  Retention: SphereNormalization
  Algorithm: OnlineMirrorDescent
  Parallelization: LatticeGLA
  Composition: MAG
  STATUS: Paper-validated (Lattice 2504.05646)

CONFIGURATION: Trellis
  Structure: Matrix (two-pass: K then V)
  Bias: L2
  Retention: L2WeightDecay (state decay)
  Algorithm: GradientDescent (OGD inner)
  Parallelization: ChunkwiseGD
  Composition: MAG
  STATUS: Paper-validated (Trellis 2512.23852)
```

## Adding New Combinations

```
To validate a new combination:
  1. Check the constraint matrices above — if any cell says NO, stop
  2. If all cells say YES, the combination COMPILES
  3. Write a test that verifies:
     (a) Forward pass produces valid output (not NaN)
     (b) Backward pass produces valid gradients (not NaN)
     (c) Gradient magnitudes are reasonable (not diverging)
     (d) A short build run converges (loss decreases)
  4. If all 4 pass, add the configuration to the named list above
  5. Update the constraint matrices if the combination reveals new constraints

The trait system prevents invalid combinations at compile time.
Untested-but-valid combinations compile but require manual validation.
```

## Axiom Compliance

- **MIRAS IS #1** (orthogonal design choices): 4 axes ARE the orthogonal choices
- **MIRAS IS NOT #2** (not single optimal config): Multiple valid configurations
- **NL IS #2** (nested, multi-level): CMS wraps any valid configuration
- **NL IS #6** (optimizer IS memory): Algorithm axis IS the optimizer
