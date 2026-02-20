# Memory Update Rule Interface

```
CONTRACT
  Purpose:    Defines the trait that ALL memory update rules must satisfy.
              Every named variant (Titans LMM, Delta, Hebbian, MONETA, YAAD,
              MEMORA, Lattice OSR, Trellis) is a configuration of this trait.
  Expects:    A valid MIRAS 4-knob configuration.
  Guarantees: Any implementor can be plugged into any CompositionPattern
              and any ParallelizationStrategy it declares support for.
  Cost:       Zero — this is a trait definition, not a runtime object.
  Trade-off:  Generality vs specialization. The trait is intentionally
              minimal. Specialized behavior lives in implementors.
  Position:   Root of specs/algorithms/memory_update_rules/.
              Parent of all memory update rule implementations.
  Source:     MIRAS (2504.13173) Table 1 + Figure 1
```

## The Four Knobs (MIRAS Framework)

Every memory update rule is fully specified by four independent choices:

| Knob | What It Controls | Options | Trait |
|---|---|---|---|
| **Memory Structure** | How M is parameterized | vector, matrix, 2-layer MLP | `MemoryStructure` |
| **Attentional Bias** | Loss function (what counts as "surprise") | L2, dot-product, Huber, l_p | `AttentionalBias` |
| **Retention** | How old memories fade | L2 decay, KL, elastic net, Bregman, none | `RetentionMechanism` |
| **Algorithm** | How to solve the loss | GD, GD+momentum, Newton-Schulz, FTRL | `MemoryAlgorithm` |

These are MOSTLY orthogonal — each knob can be varied independently in MOST
cases. However, some combinations are mathematically invalid.
(Code smell CS-33: Don't force same bias across models.
 Code smell CS-34: Don't restrict memory to matrix.
 Code smell CS-35: Don't assume GD is the only algorithm.
 Code smell CS-36: Don't restrict retention to L2.)

**Committee Finding 3**: The claim of full orthogonality is a dangerous
oversimplification. The constraint matrix in `constraints/trait_system/`
documents which pairings are invalid. The Rust type system MUST enforce
these constraints at compile time — not rely on developers memorizing
a markdown file. See "Compile-Time Composition Safety" section below.

## Trait Definition

```
TRAIT: MemoryUpdateRule
  REQUIRES: MemoryStructure + AttentionalBias + RetentionMechanism + MemoryAlgorithm
          + EnzymeOpaque  // Committee Finding 1: compiler-enforced barrier

  TYPES:
    State          -- opaque, rule-specific internal state
    Config         -- the four knobs plus hyperparameters
    OuterParams    -- projection matrices, gate parameters, conv weights (outer_loop_param lifetime)
                   -- Includes: W_K, W_V, W_Q, gate weights (W_alpha, W_eta, W_theta),
                   --   w_k_conv [d_model, kernel_size], w_q_conv [d_model, kernel_size]
                   --   (short causal conv preprocessing, see 02_short_conv.md)

  INIT(config: Config) -> (State, OuterParams)
    Create initial memory state and outer-loop parameters from configuration.
    State includes: memory weights, momentum (if any).
    OuterParams includes: W_K, W_V, W_Q, gate weights.

  WRITE(state: &mut State, k: &Tensor, v: &Tensor, gates: &Gates, pulse: &Pulse)
    Update memory given a key-value pair.
    gates contains data-dependent scalars: alpha (retain), eta (momentum), theta (lr).
    Mutates state in place. Does NOT produce output.
    The Pulse determines whether this level is active (frequency scheduling).

  READ(state: &State, q: &Tensor) -> Tensor
    Query memory with q, return output.
    Does NOT mutate state. Pure function.

  STEP(state: &mut State, x: &Tensor, outer: &OuterParams, pulse: &Pulse) -> Tensor
    Combined operation: project x to k,v,q; apply short conv; compute gates; WRITE; READ.
    The k and q inputs to WRITE/READ are post-convolution outputs — already
    preprocessed by the short causal Conv1D (see specs/infrastructure/attention/02_short_conv.md).
    Convenience method. Equivalent to:
      k_raw = x @ outer.W_K^T
      q_raw = x @ outer.W_Q^T
      k = causal_conv1d(k_raw, outer.w_k_conv)   -- short conv preprocessing
      q = causal_conv1d(q_raw, outer.w_q_conv)   -- short conv preprocessing
      v = x @ outer.W_V^T                         -- values NOT convolved
      gates = compute_gates(k, v, outer.gate_params)
      IF pulse.is_active(self.level()):
        WRITE(state, k, v, gates, pulse)
      y = READ(state, q)
      return y

  LEVEL() -> usize
    Returns this rule's CMS frequency level index.

  SUPPORTED_PARALLELIZATION() -> Vec<ParallelizationKind>
    Declares which parallelization strategies this rule supports.
    Used by the trait system to enforce valid compositions at compile time.
```

## Gate Computation

All gates are data-dependent — computed from the current input, not fixed hyperparameters.
This is what makes the "learning rate" an emergent property.

```
FUNCTION: compute_gates(k: &Tensor, v: &Tensor, gate_params: &GateParams) -> Gates
  -- gate_params are outer_loop_param lifetime (learned via Enzyme AD)
  -- gate VALUES are data-dependent (inner_loop signals)

  alpha_t = sigmoid(linear(concat(k, v), gate_params.W_alpha))  -- retain gate [0,1]
  eta_t   = sigmoid(linear(concat(k, v), gate_params.W_eta))    -- momentum gate [0,1]
  theta_t = softplus(linear(concat(k, v), gate_params.W_theta)) -- learning rate (positive)

  return Gates { alpha: alpha_t, eta: eta_t, theta: theta_t }
```

## Gradient Computation

Inner-loop gradients are ANALYTICAL. No Enzyme needed here.

```
FUNCTION: compute_inner_gradient(state: &State, k: &Tensor, v: &Tensor, bias: &dyn AttentionalBias) -> Tensor
  -- This function is #[enzyme_opaque] — Enzyme CANNOT trace into it.
  -- The gradient is derived from the paper equation, not computed via AD.
  -- Committee Finding 1: #[enzyme_opaque] is mandatory, not optional.

  MATCH bias:
    L2:          grad = 2 * (state.M @ k - v) @ k^T          -- Titans Eq 12
    DotProduct:  grad = v @ k^T                                -- MIRAS Eq 8 (Hebbian)
    Huber:       grad = huber_grad(state.M @ k - v) @ k^T     -- MIRAS (YAAD)
    Lp:          grad = lp_grad(state.M @ k - v, p) @ k^T     -- MIRAS (MONETA)

  return grad
```

## State Lifecycle

```
outer_loop_param:   W_K, W_V, W_Q, gate_params, persistent_memory
                    Owned by the model struct. Enzyme differentiates through these.
                    Serialized in checkpoints.

inner_loop_state:   M (memory matrix), S (momentum accumulator)
                    Scoped to the forward pass or context stream.
                    Created from init params, modified by WRITE, dropped when scope ends.
                    NOT serialized (these are "thoughts", not "memories").

context_memory:     Memory state at chunk boundaries.
                    Explicitly transferred between forward calls via move semantics.
                    Serialized only for context continuation (not checkpoint).
```

## Named Configurations

| Name | Structure | Bias | Retention | Algorithm | Source | Family |
|---|---|---|---|---|---|---|
| Titans LMM | matrix | L2 | L2 decay | GD + momentum | Titans Eqs 8-15 | titans |
| Delta Rule | matrix | L2 | L2 decay | GD (no momentum) | Titans Eq 34 | titans |
| Hebbian | matrix | dot-product | L2 decay | direct (no grad) | MIRAS Eq 8 | titans |
| MONETA | 2-layer MLP | l_p | L_q + L2 global | GD | MIRAS Eqs 24-25 | miras |
| YAAD | 2-layer MLP | Huber | L2 local + global | GD | MIRAS Eq 26 | miras |
| MEMORA | 2-layer MLP | L2 | KL divergence | GD | MIRAS Eq 27 | miras |
| Lattice OSR | matrix (sphere) | L2 or dot | normalization | Riemannian GD | Lattice Eqs 9-10 | compression |
| Trellis | matrix | normalized SiLU | L2 state decay | OGD | Trellis Eqs 13-14 | compression |

## Compile-Time Composition Safety (Committee Finding 3)

```
-- Committee Finding: "Don't let the interface lie about orthogonality."
-- The traits are MOSTLY independent, but some pairings are mathematically invalid.
-- The documentation says "orthogonal" but the math says "constrained."
-- The Rust type system must enforce the constraints. Invalid models must not compile.

-- MECHANISM 1: Marker traits for compatibility classes

TRAIT: ProbabilitySimplex          // memory state lives on the probability simplex
TRAIT: UnitSphere                  // memory state lives on the unit sphere
TRAIT: LinearRecurrence            // update is linear in memory state (M_{t+1} = A*M_t + b)
TRAIT: StateIndependentMomentum    // momentum depends only on current grad, not history

-- Example: KL retention REQUIRES ProbabilitySimplex
impl RetentionMechanism for KLDivergenceRetention {
  type RequiredManifold = ProbabilitySimplex;
}

-- Example: Sphere normalization provides UnitSphere, NOT ProbabilitySimplex
impl RetentionMechanism for SphereNormalization {
  type RequiredManifold = UnitSphere;
}

-- Invalid combination caught at compile time:
--   SphereNormalization + KLDivergence bias → compile error
--   because KL bias requires ProbabilitySimplex, sphere provides UnitSphere

-- MECHANISM 2: Associated types on traits

trait MemoryStructure {
  type Manifold;           // what geometric space the memory lives in
  type UpdateOp;           // what kind of update operations it supports
}

trait AttentionalBias {
  type RequiredOp;         // what update operations the bias needs from memory
}

-- The compiler checks: MemoryStructure::UpdateOp implements AttentionalBias::RequiredOp
-- If not → compile error with a clear message about the incompatibility.

-- MECHANISM 3: Builder pattern (prevents direct instantiation)

STRUCT: MemoryRuleBuilder
  fn new() -> Self
  fn structure<S: MemoryStructure>(self, s: S) -> MemoryRuleBuilder<S, _, _, _>
  fn bias<B: AttentionalBias>(self, b: B) -> MemoryRuleBuilder<_, B, _, _>
    WHERE B::RequiredOp: From<S::UpdateOp>     // compile-time check
  fn retention<R: RetentionMechanism>(self, r: R) -> MemoryRuleBuilder<_, _, R, _>
    WHERE R::RequiredManifold: Compatible<S::Manifold>  // compile-time check
  fn algorithm<A: MemoryAlgorithm>(self, a: A) -> MemoryRuleBuilder<_, _, _, A>
  fn build(self) -> Box<dyn MemoryUpdateRule>
  -- build() is infallible: all composition checks happen at compile time
  -- via the generic bounds on structure(), bias(), retention(), algorithm().
  -- If you reached build(), the composition is valid by construction.

-- Users MUST go through the builder. Direct struct instantiation is private.
-- The builder's generic constraints make invalid compositions unrepresentable:

  // This compiles:
  MemoryRuleBuilder::new()
    .structure(MatrixMemory::new(d_k, d_v))
    .bias(L2Bias::new())
    .retention(L2WeightDecay::new(0.99))
    .algorithm(GDWithMomentum::new(0.9))
    .build()

  // This does NOT compile (sphere + KL retention = manifold mismatch):
  MemoryRuleBuilder::new()
    .structure(MatrixMemory::new(d_k, d_v))
    .bias(KLDivergenceBias::new())           // requires ProbabilitySimplex
    .retention(SphereNormalization::new())     // provides UnitSphere
    //  ^^^ compile error: UnitSphere does not implement Compatible<ProbabilitySimplex>
    .algorithm(GradientDescent::new())
    .build()

-- The full constraint matrix lives in constraints/trait_system/00_valid_compositions.md
-- but the ENFORCEMENT lives in the type system, not the documentation.
```

## What This Interface Does NOT Specify

- **Chunking**: How tokens are grouped for parallel processing (see parallelization/)
- **Frequency**: How often this rule fires (see frequency_scheduling/)
- **Composition**: How this rule plugs into a model alongside attention (see composition_patterns/)
- **Precision**: fp32 vs bf16 (implementation detail, but inner loop requires fp32)
- **Differentiation engine**: Enzyme handles outer loop only; inner loop is analytical
