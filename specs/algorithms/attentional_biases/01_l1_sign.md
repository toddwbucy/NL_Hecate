# l_1 Attentional Bias (Value-Less Associative Memory)

```text
CONTRACT
  Purpose:    The l_1 attentional bias measures memory error using the L1 norm
              instead of L2. Where L2 (standard) squares the error — amplifying
              large errors and smoothing small ones — L1 takes the absolute value,
              treating all error magnitudes equally. At p=1, the gradient becomes
              Sign(M(k) - v) @ k^T: the memory update direction depends only on
              the SIGN of the error, not its magnitude. MIRAS calls this "value-less
              associative memory" because the memory maps entities to +1 or -1
              directions rather than storing actual values.
  Expects:    Memory state W, key k_t, value v_t, learning rate eta_t.
              Memory structure must support matrix-vector product M(W, k) = W @ k.
  Guarantees: Gradient is Sign(W k - v) @ k^T — constant magnitude per element.
              Robust to outlier values: a single extreme v_t does not dominate the
              update (unlike L2 where large errors produce large gradients).
              At convergence, memory stores sign(v) rather than v itself.
  Cost:       O(d^2) — same asymptotic cost as L2. The Sign operation is cheaper
              than the multiply in L2 (no squaring), but requires smooth
              approximation for differentiability.
  Trade-off:  Robust to noise and outliers (sign-based updates are bounded).
              But loses fine-grained value information — memory converges to
              {-1, +1} patterns rather than continuous values. Best suited for
              classification-like memories or entity detection.
  Position:   specs/algorithms/attentional_biases/01_l1_sign.md
              Sibling of: L2 (standard, in memory_update_rules variants),
                          Huber (S3b future, combines L1 and L2)
  Source:     MIRAS (2504.13173) §5.1 Eqs 10-12, Remark 5
```

## The l_p Attentional Bias Family

The l_1 bias is the p=1 specialization of the general l_p family:

<!-- HADES: miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10) -->
```text
-- l_p attentional bias (MIRAS Eq 10):
L(W; k_t, v_t) = ||M(W, k_t) - v_t||_p^p

-- For matrix memory M(W, k) = W @ k:
L(W; k_t, v_t) = ||W k_t - v_t||_p^p = sum_{j} |W k_t - v_t|_j^p

-- Special cases:
--   p = 2: L2 (standard). L = ||W k - v||^2. Gradient = 2*(W k - v) @ k^T
--   p = 1: L1 (this spec). L = ||W k - v||_1. Gradient = Sign(W k - v) @ k^T
--   p = 3: L3 (MONETA design target). Intermediate sensitivity.
--   p → ∞: L_inf. Only the largest error component matters.
```

## l_1 Closed-Form Gradient

Setting p=1 in the general l_p gradient (Eq 11):

<!-- HADES: miras_equations/eq-011-lp-closed-form (§5.1 Eq 11, p=1 specialization) -->
```text
-- General l_p gradient (MIRAS Eq 11):
grad_lp = p * (Sign(W k_t - v_t) ⊙ |W k_t - v_t|^{p-1}) @ k_t^T

-- At p = 1:
--   p * |W k - v|^{p-1} = 1 * |W k - v|^0 = 1 (everywhere)
--   The magnitude term vanishes completely.

-- l_1 gradient:
grad_l1 = Sign(W k_t - v_t) @ k_t^T

-- Each element of the gradient is ±1 scaled by the key projection.
-- The gradient magnitude is INDEPENDENT of the error magnitude.
-- A tiny error and a huge error produce the same update direction.
```

## Value-Less Associative Memory (Eq 12)

The full update rule with retention:

<!-- HADES: miras_equations/eq-012-value-less-memory (§5.1 Eq 12) -->
```text
-- Value-less memory update (MIRAS Eq 12):
FUNCTION: l1_memory_step(W: &mut Tensor, k: &Tensor, v: &Tensor,
                          alpha_t: f32, eta_t: f32) -> ()
  -- W: memory state [d, d]
  -- k: key vector [d, 1]
  -- v: value vector [d, 1]

  -- Error signal
  error = W @ k - v                         -- [d, 1]

  -- Sign-based gradient (constant magnitude)
  grad = Sign(error) @ k^T                  -- [d, d]

  -- Retention + update (with L2 decay as example)
  W = (1 - alpha_t) * W - eta_t * grad

  -- Properties:
  --   The update direction is determined entirely by Sign(error).
  --   Positive error components → decrease W in that direction.
  --   Negative error components → increase W in that direction.
  --   The actual magnitude |W k - v| does NOT affect the update size.
```

## Why "Value-Less"?

<!-- HADES: miras_equations/eq-012-value-less-memory (§5.1 Eq 12, interpretation) -->
```text
-- Standard L2 memory (p=2):
--   Converges to W such that W @ k ≈ v (stores the actual value v).
--   Memory faithfully reconstructs the value associated with each key.

-- Value-less L1 memory (p=1):
--   Converges to W such that Sign(W @ k) = Sign(v).
--   Memory stores only the DIRECTION (+/-) of each value component.
--   The magnitude of W @ k is unconstrained (only sign matters).

-- Interpretation (MIRAS §5.1):
--   "Similar to a coping mechanism in humans where memory does not
--    store values for extreme events" (Loftus 1993).
--   The memory records THAT something happened (and its polarity),
--   not HOW MUCH it happened.

-- Use cases:
--   Entity detection: "Was this entity present?" (+1) or absent (-1)
--   Binary classification in memory: categories, not magnitudes
--   Noise resilience: outlier values do not distort the memory
```

## Robustness Properties

```text
-- L2 gradient: grad = 2 * (W k - v) @ k^T
--   Gradient ∝ error magnitude. Large errors → large gradients.
--   One outlier v_t can dominate many normal updates.
--   Sensitivity: gradient variance = O(var(v)^2)

-- L1 gradient: grad = Sign(W k - v) @ k^T
--   Gradient is ±1 regardless of error size. Bounded.
--   An outlier v_t produces the SAME gradient as a normal v_t.
--   Sensitivity: gradient variance = 0 (constant magnitude)

-- This makes L1 a natural choice for:
--   Noisy value streams (sensor data, user behavior)
--   Long-tail distributions (rare events with extreme values)
--   Memories where direction matters more than magnitude
```

## Smooth Approximation (Remark 5)

Sign and absolute value are non-differentiable at zero. MIRAS Remark 5
specifies smooth approximators for tape integration:

<!-- HADES: miras_equations/eq-012-value-less-memory (§5.1 Remark 5, smooth approximators) -->
```text
-- Sign approximator:
Sign(x) ≈ tanh(a * x)
-- a controls sharpness: large a → approaches true Sign
-- a = 10 (practical default): tanh(10 * 0.01) ≈ 0.1 (smooth near 0)
--                              tanh(10 * 0.5) ≈ 1.0 (saturated away from 0)

-- Gradient of smooth Sign:
d/dx tanh(a*x) = a * (1 - tanh(a*x)^2) = a * sech^2(a*x)

-- For l_1 specifically (p=1), the |x|^{p-1} = |x|^0 = 1 term vanishes,
-- so only the Sign approximator is needed. No absolute-value approximator
-- required (unlike general l_p where both are needed).

-- The smooth l_1 gradient:
grad_l1_smooth = tanh(a * (W k_t - v_t)) @ k_t^T

-- This is fully differentiable and the Wengert tape handles it natively.
```

## Connection to Huber Loss

The Huber loss (MIRAS §5.1 Eqs 13-16) bridges L2 and L1:

<!-- HADES: miras_equations/eq-013-huber-function (§5.1 Eq 13, connection to l_1) -->
```text
-- Huber loss:
H_delta(x) = { x^2 / 2          if |x| <= delta
             { delta * |x| - delta^2/2   if |x| > delta

-- At small errors (|x| <= delta): behaves like L2 (smooth, proportional)
-- At large errors (|x| > delta): behaves like L1 (bounded gradient)

-- The Huber coordinate-wise update (Eq 14) uses indicator functions:
--   I(|error| <= delta): apply L2 gradient
--   I(|error| > delta): apply L1 (Sign-based) gradient

-- L1 bias is the limit of Huber as delta → 0:
--   ALL errors are treated as "large" → pure Sign-based update.
-- L2 bias is the limit as delta → ∞:
--   ALL errors are treated as "small" → standard squared gradient.
--
-- This places l_1 and L2 at opposite ends of the robustness spectrum,
-- with Huber as a tunable interpolation between them.
```

## Gradient Derivation (for tape integration)

<!-- HADES: Derived from miras_equations/eq-012-value-less-memory (§5.1 Eq 12), analytical VJP -->
```text
-- Forward (with smooth approximation):
--   error_t = W_{t-1} @ k_t - v_t                        -- [d, 1]
--   sign_approx_t = tanh(a * error_t)                     -- [d, 1]
--   grad_t = sign_approx_t @ k_t^T                        -- [d, d]
--   W_t = (1 - alpha_t) * W_{t-1} - eta_t * grad_t       -- [d, d]

-- Given: dL/dW_t (upstream gradient)
-- Need: dL/dW_{t-1}, dL/dk_t, dL/dv_t, dL/dalpha_t, dL/deta_t

-- Step 1: Through retention + update
dL/dW_{t-1} (through decay) = (1 - alpha_t) * dL/dW_t
dL/dgrad_t = -eta_t * dL/dW_t

-- Step 2: Through outer product grad_t = sign_approx @ k^T
dL/dsign_approx_t = dL/dgrad_t @ k_t                      -- [d, 1]
dL/dk_t (through grad) = sign_approx_t^T @ dL/dgrad_t     -- [d, 1]

-- Step 3: Through tanh approximation
--   d(tanh(a*x))/dx = a * (1 - tanh(a*x)^2)
dL/derror_t = a * (1 - sign_approx_t^2) ⊙ dL/dsign_approx_t

-- Step 4: Through error = W @ k - v
dL/dW_{t-1} (through error) = dL/derror_t @ k_t^T
dL/dk_t (through error) = W_{t-1}^T @ dL/derror_t
dL/dv_t = -dL/derror_t

-- Combine:
dL/dW_{t-1} = (1 - alpha_t) * dL/dW_t + dL/derror_t @ k_t^T
dL/dk_t = sign_approx_t^T @ dL/dgrad_t + W_{t-1}^T @ dL/derror_t

-- Gate gradients (scalars):
dL/dalpha_t = -trace(W_{t-1}^T @ dL/dW_t)
dL/deta_t = -trace(grad_t^T @ dL/dW_t)
```

## Implementation Notes

1. **Sharpness parameter a**: The tanh sharpness `a` controls approximation
   quality. At a=10, `tanh(10x)` approximates Sign(x) well for |x| > 0.1.
   Larger a gives sharper approximation but can cause vanishing gradients
   (sech^2 decays exponentially). a=10 is a practical default.

2. **Simpler than general l_p**: At p=1, the |x|^{p-1} term vanishes (|x|^0 = 1),
   so only the Sign approximator is needed. General l_p needs both Sign and
   absolute-value-power approximators. This makes l_1 the simplest non-L2 bias.

3. **Existing code**: The l_p infrastructure in `core/src/moneta.rs` already
   supports configurable `lp_p`. Setting `lp_p = 1.0` should yield the l_1
   bias, but the smooth approximators may need tuning (the general l_p path
   computes |x|^{p-1} which at p=1 is |x|^0 — verify this fast-paths to 1.0).

4. **Interaction with retention**: l_1 bias is orthogonal to retention choice.
   It pairs with any retention mechanism (L2 decay, KL, elastic net, L_q,
   sigmoid-bounded). The MIRAS framework guarantees all (bias, retention)
   combinations are valid.

5. **Pluggable bias dispatch**: Registers as `AttentionalBias::L1Sign` in the
   S3b-M2 infrastructure. The generic l_p dispatch (S3b-S11) will subsume this
   as the p=1 specialization with the fast-path optimization.

6. **Interaction with CMS (Continuous Memory System)**: At slow CMS levels
   (see `specs/infrastructure/scheduling/00_conductor.md`), the bounded gradient
   of l_1 means slow-frequency memories accumulate sign information without
   magnitude distortion from outliers in the token stream.

## Axiom Compliance

- **NL IS #4** (compressing context): The Sign operation compresses continuous error signals to binary directions — maximum compression of the error space.
- **NL IS #9** (principled not ad hoc): The l_1 bias emerges from the l_p framework at p=1, not from a heuristic choice to "just take the sign."
- **MIRAS IS #1** (orthogonal design choices): Attentional bias is independent of memory structure, retention, and algorithm — l_1 composes with any valid combination.
