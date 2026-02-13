# Sphere Normalization Retention

```
CONTRACT
  Purpose:    Implicit retention via projection back to the unit sphere.
              Used by Lattice OSR. There is no explicit decay gate —
              normalization IS the forgetting mechanism. When new information
              is added and the vector is renormalized, old information's
              relative contribution decreases.
  Expects:    Memory state on unit sphere (||s|| = 1).
              Update direction delta_s.
  Guarantees: Memory stays on unit sphere after update.
              Forgetting rate is proportional to update magnitude.
              No tunable forgetting hyperparameter — forgetting emerges
              from the geometry of the sphere.
  Cost:       O(d) for normalization (compute norm, divide).
  Trade-off:  No forgetting hyperparameter (one less thing to tune).
              But forgetting rate is not directly controllable — it's
              determined by the magnitude of new information.
              Cannot independently control forgetting and learning rates.
  Position:   specs/algorithms/retention_mechanisms/05_sphere_normalization.md
  Source:     Lattice (2504.05646) Proposition 3.1, implicit in Eqs 5-10
```

## Mechanism

```
FUNCTION: sphere_normalize_retention(s: &Tensor, delta_s: &Tensor,
                                      beta: f32) -> Tensor
  -- s: current state, ||s|| = 1
  -- delta_s: update direction (NOT on sphere)
  -- beta: state-dependent gating scalar

  -- Step 1: Scale update
  update = beta * delta_s

  -- Step 2: Add to current state (leaves sphere)
  s_unnormalized = s + update

  -- Step 3: Project back to sphere (THIS IS THE RETENTION)
  s_new = s_unnormalized / norm(s_unnormalized)

  return s_new
```

## Why Normalization IS Forgetting

Consider a 2D example:

```
-- s = [1.0, 0.0]  (memory knows "feature 1")
-- delta_s = [0.0, 0.5]  (new info: "feature 2")

-- Before normalization: s_unnorm = [1.0, 0.5]
-- After normalization:  s_new = [0.894, 0.447]

-- Feature 1 went from 1.0 to 0.894 (decayed)
-- Feature 2 went from 0.0 to 0.447 (learned)
-- The decay of feature 1 happened WITHOUT an explicit forgetting gate.
-- Normalization REDISTRIBUTED capacity from old to new.
```

The larger delta_s is, the more old information is "forgotten" (its relative
contribution shrinks when renormalized). This is an emergent forgetting rate
that depends on the INPUT, not a hyperparameter.

## Relationship to Riemannian GD

Lattice Proposition 3.1 proves that this process — update then normalize — is
gradient descent on the Riemannian manifold S^{d-1} (the unit sphere) with
the metric inherited from R^d.

The orthogonal projection step (removing the parallel component before
normalizing) is the Riemannian correction: it ensures the update stays
in the tangent plane of the sphere.

```
-- Euclidean GD:   W_new = W - lr * grad                (flat space)
-- Riemannian GD:  s_new = normalize(s - lr * proj_tangent(grad))  (curved space)

-- The sphere's curvature creates implicit forgetting.
-- In flat space (L2 retention), forgetting must be explicit: (1-alpha)*W.
-- On the sphere, the curvature does it for free.
```

## MIRAS Decomposition

Sphere normalization doesn't fit neatly into the D_t + G_t decomposition:

```
-- It's NOT D_t = ||s_new - s_old||^2 (that's L2 in embedding space)
-- It's NOT G_t = ||s||^2 (that's always 1 on the sphere)
-- Instead, it's an implicit constraint: s lives on S^{d-1}

-- The "retention" is the manifold constraint itself.
-- There is no separate penalty term — the geometry does the work.
```

## Axiom Compliance

- **Lattice IS #5** (Riemannian GD): Normalization is GD on the sphere manifold
- **Lattice IS #7** (normalization = forgetting): THIS IS the axiom
- **NL IS #4** (compressing context): Fixed-size unit vector compresses unbounded context
