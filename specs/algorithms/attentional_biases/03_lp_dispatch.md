# Generic l_p Attentional Bias Dispatch

```text
CONTRACT
  Purpose:    The l_p attentional bias family unifies all norm-based inner-loop
              loss functions under a single parameterized dispatch. The exponent
              p controls how the memory penalizes prediction errors: p=2 (L2,
              standard squared error), p=1 (L1, sign-based value-less memory),
              p=3 (MONETA design target, intermediate sensitivity). Rather than
              implementing each p as a separate code path, the dispatch computes
              the general l_p gradient using smooth approximators and routes to
              optimized fast-paths when p matches a known specialization.
  Expects:    Memory state W, key k_t, value v_t, learning rate eta_t,
              exponent p >= 1. Memory structure supports W @ k.
  Guarantees: Gradient is p * Sign(Wk - v) ⊙ |Wk - v|^{p-1} @ k^T for all
              p >= 1. At p=2: recovers standard Delta rule gradient. At p=1:
              recovers sign-based gradient (01_l1_sign.md). The dispatch is
              fully differentiable via smooth approximators (tanh for Sign,
              (x^2 + eps)^{.} for absolute power). KL bias (02_kl_objective.md)
              is NOT subsumed — it requires softmax output, a different family.
  Cost:       O(d^2) for the matrix-vector product plus O(d) for the element-wise
              sign and power operations. The smooth approximators add negligible
              overhead (tanh and power are fused element-wise).
  Trade-off:  A single dispatch covers all l_p specializations with runtime p.
              But general p requires both Sign and absolute-power approximators,
              whereas p=1 needs only Sign and p=2 needs neither. Fast-paths
              eliminate unnecessary approximator overhead for common p values.
  Position:   specs/algorithms/attentional_biases/03_lp_dispatch.md
              Parent of: 01_l1_sign.md (p=1 specialization)
              Sibling of: 02_kl_objective.md (KL bias — separate family, not l_p)
  Source:     MIRAS (2504.13173) §5.1 Eqs 10-11 (general l_p framework);
              §5.1 Eq 12 (p=1 specialization); §5.1 Remark 5 (smooth approx);
              §5.3 Eqs 24-25 (MONETA (p,q) parametrization)
```

## The l_p Attentional Bias (Eq 10)

The attentional bias is the inner-loop loss function that drives memory updates.
The l_p family measures error using the p-th power of the L_p norm:

<!-- HADES: miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10) -->
```text
-- l_p attentional bias (MIRAS Eq 10):
L(W; k_t, v_t) = ||M(W, k_t) - v_t||_p^p = sum_j |M(W, k_t) - v_t|_j^p

-- For matrix memory M(W, k) = W @ k:
L(W; k_t, v_t) = sum_j |(W k_t)_j - (v_t)_j|^p

-- The exponent p controls error sensitivity:
--   p = 1: All errors weighted equally (sign-based, value-less memory)
--   p = 2: Errors weighted proportionally (standard squared error)
--   p = 3: Large errors weighted quadratically more than small ones
--   p → ∞: Only the largest error component matters (L_inf limit)

-- The l_p family is one of two attentional bias families in this codebase:
--   l_p (this spec): norm-based, output in R^d (unbounded)
--   KL (02_kl_objective.md): information-theoretic, output on simplex
-- They are NOT interchangeable — KL requires softmax output transformation.
```

## Closed-Form Gradient (Eq 11)

The gradient of the l_p loss has a closed form for all p >= 1:

<!-- HADES: miras_equations/eq-011-lp-closed-form (§5.1 Eq 11) -->
```text
-- l_p gradient (MIRAS Eq 11):
FUNCTION: lp_gradient(W: &Tensor, k: &Tensor, v: &Tensor, p: f32) -> Tensor
  -- W: memory state [d, d]
  -- k: key vector [d, 1]
  -- v: value vector [d, 1]

  error = W @ k - v                                     -- [d, 1]
  grad = p * (Sign(error) ⊙ |error|^{p-1}) @ k^T       -- [d, d]

  -- Decomposition:
  --   Sign(error): direction of each error component (±1)
  --   |error|^{p-1}: magnitude scaling per component
  --   k^T: projects gradient into weight space (outer product)

  -- The gradient has two factors that depend on p:
  --   1. The scalar multiplier p (from d/dx |x|^p = p|x|^{p-1} sign(x))
  --   2. The power |error|^{p-1} that controls magnitude sensitivity
```

## Specializations by p Value

Different values of p produce qualitatively different memory behaviors:

<!-- HADES: miras_equations/eq-011-lp-closed-form (§5.1 Eq 11, specializations) -->
```text
| p   | |error|^{p-1} | Gradient behavior             | Memory converges to     |
|-----|---------------|-------------------------------|-------------------------|
| 1   | |e|^0 = 1     | Constant (sign only)          | Sign(v) — binary dirs   |
| 2   | |e|^1 = |e|   | Linear in error               | v — faithful recall     |
| 3   | |e|^2 = e^2   | Quadratic in error            | v — stronger on outliers|
| 4   | |e|^3 = |e|^3 | Cubic in error                | v — peak-sensitive      |

-- p = 1 (L1): Value-less memory (01_l1_sign.md)
--   grad = Sign(error) @ k^T
--   The |error|^{p-1} = |error|^0 = 1 term vanishes completely.
--   Only the Sign approximator is needed (simplest non-L2 case).
--   See 01_l1_sign.md for full treatment.

-- p = 2 (L2): Standard Delta rule gradient
--   grad = 2 * error @ k^T
--   Sign(error) ⊙ |error|^1 = Sign(error) ⊙ |error| = error (identity).
--   No approximators needed — the product reduces to the raw error.
--   This is the default for Titans LMM, Delta Rule, Hebbian.

-- p = 3 (MONETA design target):
--   grad = 3 * Sign(error) ⊙ error^2 @ k^T
--   Large errors get quadratically more gradient than small ones.
--   This amplifies correction of large mispredictions while still
--   updating for small ones (unlike L1 which treats all equally).
--   MONETA pairs p=3 with L_q retention q=4 (see 07_lq_norm.md).
```

## Smooth Approximators (Remark 5)

Sign and absolute value are non-differentiable at zero. MIRAS Remark 5
specifies smooth replacements for tape integration:

<!-- HADES: miras_equations/eq-012-value-less-memory (§5.1 Remark 5, smooth approximators for l_p) -->
```text
-- Sign approximator:
Sign(x) ≈ tanh(a * x)
-- a = 10 (practical default)
-- tanh(10 * 0.01) ≈ 0.1 (smooth near 0, avoids discontinuity)
-- tanh(10 * 0.5)  ≈ 1.0 (saturated away from 0, matches true Sign)
-- Gradient: d/dx tanh(a*x) = a * sech^2(a*x) = a * (1 - tanh(a*x)^2)

-- Absolute power approximator:
|x|^{p-1} ≈ (x^2 + eps)^{(p-1)/2}
-- eps = 1e-6 (practical default, prevents division-by-zero at x=0)
-- At large |x|: (x^2 + eps)^{(p-1)/2} ≈ |x|^{p-1} (negligible eps)
-- At x = 0: eps^{(p-1)/2} (small but nonzero, smooth)
-- Gradient: (p-1) * x * (x^2 + eps)^{(p-3)/2}

-- Combined smooth l_p gradient:
grad_smooth = p * tanh(a * error) ⊙ (error^2 + eps)^{(p-1)/2} @ k^T

-- The smooth gradient equals the exact gradient everywhere except
-- in a small neighborhood around error = 0, where it interpolates
-- smoothly instead of having a discontinuity.
```

## Fast-Path Dispatch

The generic l_p gradient is correct for all p >= 1, but specific p values
admit optimizations that eliminate unnecessary computation:

<!-- HADES: Derived from miras_equations/eq-011-lp-closed-form (§5.1 Eq 11), dispatch optimization -->
```text
FUNCTION: lp_bias_step(W: &mut Tensor, k: &Tensor, v: &Tensor,
                        alpha_t: f32, eta_t: f32, p: f32) -> ()
  error = W @ k - v                                     -- [d, 1]

  -- Dispatch on p for optimized gradient computation
  IF p == 2.0:
    -- L2 fast-path: no approximators needed
    grad = 2.0 * error @ k^T                            -- [d, d]
    -- Sign(e) ⊙ |e| = e, so the full expression simplifies to 2*e@k^T.
    -- This is the standard Delta rule gradient — no tanh, no power.

  ELSE IF p == 1.0:
    -- L1 fast-path: only Sign approximator needed
    grad = tanh(a * error) @ k^T                         -- [d, d]
    -- |error|^{p-1} = |error|^0 = 1 vanishes.
    -- Only the Sign(error) factor survives. See 01_l1_sign.md.

  ELSE:
    -- General path: both approximators required
    sign_approx = tanh(a * error)                        -- [d, 1]
    power_approx = (error^2 + eps)^{(p-1)/2}            -- [d, 1]
    grad = p * (sign_approx ⊙ power_approx) @ k^T       -- [d, d]

  -- Retention + update (same for all p)
  W = (1 - alpha_t) * W - eta_t * grad
```

## Connection to MONETA (p, q) Parametrization

MONETA pairs the l_p attentional bias with L_q norm retention, creating a
joint (p, q) design space:

<!-- HADES: miras_equations/eq-024-025-moneta-spec (§5.3 Eqs 24-25, (p,q) parametrization) -->
```text
-- MONETA (p, q) design:
--   p (attentional bias): controls error sensitivity in the loss
--   q (retention, see 07_lq_norm.md): controls memory magnitude distribution
--   Design target: (p, q) = (3, 4)
--   Current implementation: lp_p = 2.0, lq_q not wired into retention

-- The l_p gradient feeds into the L_q retention step:
--   1. Compute l_p gradient: grad_lp = p * Sign(e) ⊙ |e|^{p-1} @ k^T
--   2. Accumulate with decay: A = alpha * A - eta * grad_lp
--   3. L_q normalize: W = A / ||A||_q^{q-2}

-- At (p, q) = (2, 2): standard Delta rule + standard L2 decay (current default)
-- At (p, q) = (3, 4): MONETA design target — quadratic error emphasis + peak suppression
-- At (p, q) = (1, 2): L1 sign-based loss + standard L2 decay

-- The (p, q) parametrization is fully orthogonal: any p >= 1 composes
-- with any q >= 1. The l_p bias dispatch (this spec) handles the p axis;
-- L_q retention (07_lq_norm.md) handles the q axis independently.
```

## Gradient Derivation (for tape integration)

The VJP for the general l_p bias step, using smooth approximators throughout:

<!-- HADES: Derived from miras_equations/eq-011-lp-closed-form (§5.1 Eq 11), analytical VJP for l_p dispatch -->
```text
-- Forward (with smooth approximators):
--   error_t = W_{t-1} @ k_t - v_t                                   -- [d, 1]
--   sign_approx_t = tanh(a * error_t)                                -- [d, 1]
--   power_approx_t = (error_t^2 + eps)^{(p-1)/2}                    -- [d, 1]
--   grad_t = p * (sign_approx_t ⊙ power_approx_t) @ k_t^T          -- [d, d]
--   W_t = (1 - alpha_t) * W_{t-1} - eta_t * grad_t                  -- [d, d]

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dW_{t-1}, dL/dk_t, dL/dv_t, dL/dalpha_t, dL/deta_t

-- Step 1: Through retention + update
dL/dW_{t-1} (through decay) = (1 - alpha_t) * dL/dW_t
dL/dgrad_t = -eta_t * dL/dW_t

-- Step 2: Through outer product grad_t = p * combined @ k^T
--   Let combined_t = sign_approx_t ⊙ power_approx_t
dL/dcombined_t = p * dL/dgrad_t @ k_t                               -- [d, 1]
dL/dk_t (through grad) = p * combined_t^T @ dL/dgrad_t              -- [d, 1]

-- Step 3: Through element-wise product combined = sign ⊙ power
dL/dsign_approx_t = dL/dcombined_t ⊙ power_approx_t
dL/dpower_approx_t = dL/dcombined_t ⊙ sign_approx_t

-- Step 4: Through smooth approximators
--   d(tanh(a*x))/dx = a * (1 - tanh(a*x)^2)
dL/derror_t (through sign) = a * (1 - sign_approx_t^2) ⊙ dL/dsign_approx_t

--   d((x^2+eps)^{(p-1)/2})/dx = (p-1) * x * (x^2+eps)^{(p-3)/2}
dL/derror_t (through power) = (p-1) * error_t ⊙ (error_t^2 + eps)^{(p-3)/2}
                               ⊙ dL/dpower_approx_t

dL/derror_t = dL/derror_t (through sign) + dL/derror_t (through power)

-- Step 5: Through error = W @ k - v
dL/dW_{t-1} (through error) = dL/derror_t @ k_t^T
dL/dk_t (through error) = W_{t-1}^T @ dL/derror_t
dL/dv_t = -dL/derror_t

-- Combine:
dL/dW_{t-1} = (1 - alpha_t) * dL/dW_t + dL/derror_t @ k_t^T
dL/dk_t = p * combined_t^T @ dL/dgrad_t + W_{t-1}^T @ dL/derror_t

-- Gate gradients (scalars):
dL/dalpha_t = -trace(W_{t-1}^T @ dL/dW_t)
dL/deta_t = -trace(grad_t^T @ dL/dW_t)

-- Fast-path VJP at p = 2:
--   combined_t = error_t (no approximators)
--   dL/derror_t = 2 * dL/dgrad_t @ k_t (direct, no tanh/power backward)
--   This eliminates Steps 3-4 entirely.

-- Fast-path VJP at p = 1:
--   power_approx_t = 1 (constant, zero gradient through power)
--   dL/derror_t = a * (1 - sign_approx_t^2) ⊙ dL/dcombined_t
--   The power backward (Step 4 second term) vanishes.
```

## Implementation Notes

1. **Existing code**: `core/src/moneta.rs` already implements the general l_p
   gradient with configurable `lp_p` field. The `lp_gradient()` helper computes
   `p * sign(e) * |e|^(p-1)` using the smooth approximators. The dispatch
   described here formalizes the fast-path optimizations that eliminate
   unnecessary approximator computation at p=1 and p=2.

2. **Fast-path detection**: Compare `p` against 1.0 and 2.0 using exact float
   equality. These are configuration values set at construction, not computed
   quantities — exact comparison is reliable. An epsilon-based comparison
   (e.g., `|p - 2.0| < 1e-6`) adds complexity without benefit.

3. **Sharpness parameter a**: Shared with the l_1 bias (01_l1_sign.md). The
   default `a = 10` provides sharp approximation for |x| > 0.1 with manageable
   gradient magnitude. Larger `a` improves approximation fidelity but causes
   vanishing gradients in the tanh backward (sech^2 decays exponentially).

4. **Epsilon for power**: The `eps = 1e-6` in `(x^2 + eps)^{(p-1)/2}` prevents
   NaN at x=0 for p < 2 (where |x|^{p-1} has a singularity). At p >= 2, the
   power is well-defined at x=0 without epsilon, but retaining eps costs nothing
   and simplifies the code path.

5. **Pluggable bias dispatch**: Registers as `AttentionalBias::Lp(p)` in the
   S3b-M2 infrastructure. The enum variant carries the exponent p as a runtime
   parameter. The l_1 specialization (01_l1_sign.md) is subsumed by
   `AttentionalBias::Lp(1.0)` with the Sign-only fast-path. KL bias
   (02_kl_objective.md) remains a separate enum variant — it is NOT an l_p
   specialization.

6. **Interaction with CMS (Continuous Memory System)**: The l_p exponent is
   fixed per memory level — it does not change with the Conductor's Pulse.
   Different CMS levels MAY use different p values (e.g., fast levels with p=2
   for faithful recall, slow levels with p=1 for robust sign-based accumulation),
   but this is a configuration choice, not a runtime dispatch.

## Axiom Compliance

- **NL IS #4** (compressing context): Higher p compresses context more aggressively — large-error associations dominate the memory update, while small errors are progressively ignored.
- **NL IS #9** (principled not ad hoc): The l_p family derives from the l_p norm, a well-defined mathematical object. The exponent p is a continuous knob, not a discrete architectural choice.
- **MIRAS IS #1** (orthogonal design choices): The attentional bias (p) is independent of memory structure, retention (q), and algorithm. The l_p dispatch composes with any valid combination of the other three knobs.
