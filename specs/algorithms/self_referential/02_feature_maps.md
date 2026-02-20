# Higher-Order Feature Maps phi(k)

<!-- HADES: hope_equations/eq-051-higher-order-features (§4.4 Eq 51); hope_equations/eq-005-fwp-update (§2 Eq 5) -->
```text
CONTRACT
  Purpose:    Higher-order feature maps transform the key vector before it
              interacts with memory. Instead of the raw key k_t entering the
              memory update, a feature map phi(k_t) provides a richer
              representation. In Fast Weight Programmers (Eq 5), phi maps keys
              to a feature space where the outer product v @ phi(k)^T has higher
              capacity. In GGD momentum (Eq 51), phi transforms gradient keys
              to enhance the momentum accumulator's memory capacity. Feature
              maps bridge linear associative memory (phi = identity) to
              kernel-based memory (phi = random features, polynomial features,
              or learned transforms).
  Expects:    Key vector k_t [d, 1]. Feature map phi: R^d -> R^d_phi where
              d_phi >= d (expansion) or d_phi = d (same-dimensional transform).
              Memory state M [d, d_phi] (widened if d_phi > d).
  Guarantees: phi(k) replaces k in all memory operations: update, read, gradient.
              When phi = identity, all equations reduce to the standard (no
              feature map) case. The MIRAS 4-knob framework applies unchanged
              to the feature-mapped keys — phi is a preprocessing step before
              the attentional bias loss is computed.
  Cost:       O(d * d_phi) for the feature map evaluation, plus the standard
              memory cost with d_phi replacing d in the key dimension.
              Identity: O(0) (no-op). Linear: O(d * d_phi). MLP: O(d * d_hidden).
              Random features: O(d * d_phi) (one matmul, fixed weights).
  Trade-off:  Feature maps increase memory capacity (more associations per
              matrix entry) but expand the key dimension, increasing the cost
              of the outer product and memory query. The capacity-cost tradeoff
              is controlled by d_phi — larger d_phi means more capacity but
              O(d * d_phi) cost per operation instead of O(d^2).
  Position:   specs/algorithms/self_referential/02_feature_maps.md
              Parent: 00_interface.md (self-referential framework)
              Sibling of: 01_self_generated_values.md (Phase 3 targets)
  Source:     HOPE (2512.24695) §2 Eq 5 (FWP with phi), §4.4 Eq 51 (higher-order
              momentum); extends to MIRAS attentional bias framework via key
              preprocessing
```

## Feature Maps in Fast Weight Programmers (Eq 5)

The original feature map appears in the Fast Weight Programmer update rule:

<!-- HADES: hope_equations/eq-005-fwp-update (§2 Eq 5) -->
```text
-- Fast Weight Programmer (HOPE Eq 5):
M_t = alpha_t * M_{t-1} + v_t @ phi(k_t)^T

-- Components:
--   M: memory matrix [d, d_phi]
--   v_t: value vector [d, 1]
--   k_t: key vector [d, 1]
--   phi: feature map R^d -> R^d_phi
--   alpha_t: retention gate

-- Without feature map (phi = identity):
--   M_t = alpha * M_{t-1} + v_t @ k_t^T
--   This is the standard Hebbian outer-product update.
--   Memory capacity: O(d) associations (linear in dimension).

-- With feature map (phi = nonlinear):
--   M_t = alpha * M_{t-1} + v_t @ phi(k_t)^T
--   The outer product v @ phi(k)^T stores the association in a
--   higher-dimensional feature space.
--   Memory capacity: O(d_phi) associations (can exceed d).

-- Memory read also uses phi:
--   y_t = M_t @ phi(q_t)
--   The query is mapped through the SAME feature map.
--   This ensures write and read operate in the same feature space.
```

## Higher-Order Feature Map Momentum (Eq 51)

In the context of GGD momentum, phi transforms gradient keys:

<!-- HADES: hope_equations/eq-051-higher-order-features (§4.4 Eq 51) -->
```text
-- Higher-order feature map momentum (HOPE Eq 51):
W_{i+1} = W_i + m_{i+1}
m_{i+1} = alpha_{i+1} * m_i - eta_t * P_i @ phi(grad_L(W_i; x_i))

-- Components:
--   W: outer-loop weight (the parameter being optimized)
--   m: momentum accumulator
--   P_i: projection or preconditioner
--   phi: feature map applied to the gradient
--   alpha: momentum decay

-- phi transforms the gradient BEFORE it enters the momentum accumulator.
-- The momentum m acts as a memory of past phi-transformed gradients.
-- This gives the optimizer a richer view of the gradient landscape:
--   phi = identity → standard momentum (stores raw gradients)
--   phi = nonlinear → momentum stores transformed gradients with
--                      higher capacity for gradient patterns

-- Connection to memory: m is a memory matrix, phi(grad) is the key,
-- P_i is the value. The momentum update IS an associative memory update
-- with feature-mapped keys. (HOPE §4.4: "phi may be learned through
-- its internal objective.")
```

## Feature Map Types

Different choices of phi provide different capacity-cost tradeoffs:

<!-- HADES: Derived from hope_equations/eq-005-fwp-update (§2 Eq 5), feature map taxonomy -->
```text
-- Type 1: Identity (standard, no feature map)
phi(k) = k                              -- d_phi = d
-- Cost: O(0) (no-op)
-- Capacity: O(d) associations per memory matrix
-- This is the current NL_Hecate default for all variants.

-- Type 2: Linear projection
phi(k) = W_phi @ k                      -- d_phi arbitrary
-- Cost: O(d * d_phi) per evaluation
-- W_phi: outer_loop_param [d_phi, d], learned by AD
-- Expands or compresses the key dimension.
-- At d_phi > d: increases capacity at the cost of larger M.
-- At d_phi < d: compresses keys (reduces memory but may lose info).

-- Type 3: Random Fourier features (kernel approximation)
phi(k) = sqrt(2/d_phi) * cos(W_rand @ k + b_rand)
-- Cost: O(d * d_phi) per evaluation
-- W_rand: FIXED random matrix (not learned), b_rand: fixed random bias
-- Approximates the Gaussian kernel: phi(k)^T phi(q) ≈ exp(-||k-q||^2 / 2sigma^2)
-- Provides infinite-dimensional kernel capacity with finite d_phi features.
-- Pro: no learnable parameters in phi. Con: phi is not adapted to data.

-- Type 4: Polynomial features
phi(k) = [k; k ⊗ k; ...]              -- d_phi = d + d^2 + ...
-- Cost: O(d^2) for degree-2, O(d^p) for degree-p
-- Explicit polynomial expansion of the key vector.
-- k ⊗ k is the Kronecker product (all pairwise products).
-- Degree 2: captures quadratic key interactions.
-- Pro: deterministic, no parameters. Con: d_phi grows exponentially with degree.

-- Type 5: Learned MLP (self-referential)
phi(k) = k + W_1 @ sigma(W_2 @ k)      -- d_phi = d (residual MLP)
-- Cost: O(d * d_hidden) per evaluation
-- W_1, W_2: outer_loop_params, learned by AD
-- This is the same architecture as Eq 89 (practical MLP memory).
-- The feature map IS itself a memory module (when adaptive, Phase 2+).
-- Pro: maximally expressive. Con: adds learnable parameters and cost.

-- Type 6: ELU-based (from linear attention literature)
phi(k) = elu(k) + 1                     -- d_phi = d
-- Cost: O(d) per evaluation (element-wise)
-- Ensures phi(k) > 0 for all k (non-negative features).
-- Used in linear attention (Katharopoulos et al. 2020) where
-- non-negative features enable causal linear attention.
-- Pro: cheap, non-negative. Con: limited expressiveness.
```

## Integration with Memory Update Rules

Feature maps are a preprocessing step — they slot into the existing MIRAS
framework without changing the 4-knob structure:

<!-- HADES: Derived from hope_equations/eq-005-fwp-update (§2 Eq 5), integration with MIRAS update rules -->
```text
-- Standard memory update (no feature map):
--   error = M @ k_t - v_t                          -- read with raw key
--   grad = bias_gradient(error) @ k_t^T             -- gradient in raw key space
--   M = retention(M) - eta * grad                   -- update raw memory

-- Feature-mapped memory update:
--   phi_k = phi(k_t)                                -- transform key
--   error = M @ phi_k - v_t                         -- read with mapped key
--   grad = bias_gradient(error) @ phi_k^T           -- gradient in feature space
--   M = retention(M) - eta * grad                   -- update in feature space

-- The ONLY change: k_t → phi(k_t) wherever the key appears.
-- Attentional bias (l_p, KL) applies to the error in feature space.
-- Retention (L2, KL, elastic net, L_q) applies to M in feature space.
-- Algorithm (GD, GD+momentum, Newton-Schulz, FTRL) operates on feature-space grad.
-- Memory structure (matrix, MLP) uses M [d, d_phi] instead of M [d, d].

-- Query also uses feature map:
--   y_t = M @ phi(q_t)                             -- read with mapped query
-- Write and read MUST use the same phi for associative retrieval to work.
```

## Gradient Through Feature Maps

The feature map phi participates in the gradient computation:

<!-- HADES: Derived from hope_equations/eq-051-higher-order-features (§4.4 Eq 51), VJP through feature map -->
```text
-- Forward:
--   phi_k_t = phi(k_t)                              -- [d_phi, 1]
--   error_t = M_{t-1} @ phi_k_t - v_t               -- [d, 1]
--   grad_t = bias_gradient(error_t) @ phi_k_t^T     -- [d, d_phi]
--   M_t = retention(M_{t-1}) - eta_t * grad_t       -- [d, d_phi]

-- Given: dL/dM_t (upstream gradient)
-- Need: dL/dk_t (through phi), dL/dphi_params (if learnable)

-- Through memory update (standard, unchanged):
dL/dM_{t-1} = retention_backward(dL/dM_t)
dL/dgrad_t = -eta_t * dL/dM_t

-- Through outer product grad = g @ phi_k^T:
dL/dphi_k_t (through grad) = bias_grad^T @ dL/dgrad_t    -- [d_phi, 1]

-- Through error = M @ phi_k - v:
dL/dphi_k_t (through error) = M_{t-1}^T @ dL/derror_t    -- [d_phi, 1]

-- Total gradient w.r.t. phi_k_t:
dL/dphi_k_t = dL/dphi_k_t (through grad) + dL/dphi_k_t (through error)

-- Through feature map phi:
--   If phi is fixed (random features): dL/dk_t = W_rand^T @ dL/dphi_k_t
--   If phi is learned (linear): dL/dk_t = W_phi^T @ dL/dphi_k_t,
--                                dL/dW_phi = dL/dphi_k_t @ k_t^T
--   If phi is MLP: standard MLP backward through residual connection

-- The tape records phi as an opaque VJP block. Fixed-weight feature maps
-- (random features, ELU) propagate gradients to k_t but have no learnable
-- parameters. Learned feature maps (linear, MLP) accumulate gradients
-- for their own parameters via the same tape mechanism.
```

## Connection to Self-Referential Projections

Feature maps and self-referential projections (00_interface.md) serve
related but distinct purposes:

<!-- HADES: Derived from hope_equations/eq-005-fwp-update (§2 Eq 5); hope_equations/eq-079-phase2-adaptive-projections (§8.1 Eq 79), comparison -->
```text
-- Self-referential projections (Phase 2+):
--   k_t = M_k(x_t)
--   The projection ITSELF is a memory module that adapts per-token.
--   M_k replaces the static W_k projection.
--   The key is produced by an adaptive memory applied to x_t.

-- Feature maps:
--   phi_k_t = phi(k_t)
--   A fixed or learned transformation applied AFTER the key is produced.
--   phi does not adapt per-token (unless it is itself a memory module).
--   The key is transformed before it enters the memory update.

-- Combined (Phase 2+ with feature maps):
--   k_t = M_k(x_t)           -- adaptive projection produces key
--   phi_k_t = phi(k_t)       -- feature map transforms key
--   error = M @ phi_k_t - v  -- memory operates in feature space
--   This composes: the key is first adaptively generated, then
--   feature-mapped. Both layers of processing can be adaptive.

-- Extreme case: phi IS a self-referential memory (Phase 3 feature map)
--   phi = M_phi (a MemoryRule instance that updates per-token)
--   phi_k_t = M_phi(k_t)
--   This merges feature maps with self-referential projections.
--   The feature map adapts to the token stream, not just to pre-training.
--   This is not explicitly in the HOPE paper but follows from composing
--   §4.4 (feature maps) with §8.1 (self-referential projections).
```

## Implementation Notes

1. **Current state**: NL_Hecate uses phi = identity for all variants. The key
   k_t enters the memory update directly. This spec documents the feature map
   extension that the S3b infrastructure will support as a configurable
   preprocessing step.

2. **Memory dimension change**: When d_phi != d, the memory matrix M changes
   shape from [d, d] to [d, d_phi]. This affects all memory operations
   (update, read, backward) and the memory's parameter count. The existing
   MemoryRule trait must accept d_phi as a configuration parameter alongside d.

3. **Shared phi for write and read**: The same feature map MUST be used for
   both the update key phi(k_t) and the query phi(q_t). Using different maps
   breaks associative retrieval — the memory stores associations in phi-space,
   so queries must be in the same space. This is enforced by configuration:
   a single phi instance is shared across all memory operations.

4. **Feature map as configuration, not knob**: phi is NOT a MIRAS knob — it is
   a preprocessing step that sits outside the 4-knob framework. The MIRAS paper
   does not parameterize feature maps. The HOPE paper introduces phi as an
   extension (§4.4) that composes with any MIRAS configuration. Accordingly,
   phi registers as a separate configuration field alongside the 4 knobs.

5. **Pluggable dispatch**: Feature maps register as a `FeatureMapKind` enum:
   `Identity`, `Linear(d_phi)`, `RandomFourier(d_phi, sigma)`, `ELU`,
   `MLP(d_hidden)`. The dispatch evaluates phi once per key, caches the result,
   and passes phi(k) to all subsequent memory operations for that token.

6. **Interaction with CMS**: Feature maps are per-level configuration — each
   CMS level MAY use a different feature map. Fast levels might use identity
   (cheap, standard), while slow levels use learned MLP features (expressive,
   higher capacity for long-horizon associations).

## Axiom Compliance

- **NL IS #4** (compressing context): Feature maps can compress (d_phi < d) or expand (d_phi > d) the key representation, giving explicit control over the information bottleneck between input and memory.
- **NL IS #6** (optimizers are associative memory): Feature-mapped memory IS kernel-based associative memory — phi(k)^T phi(q) is a kernel function, and memory retrieval M @ phi(q) is kernel regression.
- **NL IS #9** (principled not ad hoc): Feature maps derive from the kernel methods literature (random Fourier features, Mercer's theorem) and the Fast Weight Programmer lineage (Schmidhuber 1992). The extension to MIRAS preserves all theoretical guarantees because phi is a pure preprocessing step.
- **MIRAS IS #1** (orthogonal design choices): Feature maps are orthogonal to all four MIRAS knobs. Any phi composes with any (structure, bias, retention, algorithm) combination — the knobs see phi(k) as "the key" without knowing how it was produced.
