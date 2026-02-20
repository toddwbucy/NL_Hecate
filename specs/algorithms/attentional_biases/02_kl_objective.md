# KL Attentional Bias (Cross-Entropy Objective)

```text
CONTRACT
  Purpose:    The KL attentional bias uses KL divergence (cross-entropy) as the
              inner-loop loss function instead of L2 or L1. Where L2 measures
              squared error and L1 measures absolute error, KL measures
              information-theoretic divergence between the memory output and the
              target value. This requires memory output to be a probability
              distribution (via softmax), making it natural for classification-like
              or next-token-prediction memories. KL bias connects the inner-loop
              memory to the same loss landscape used by the outer-loop language
              model — both minimize cross-entropy.
  Expects:    Memory state W, key k_t, value distribution p_t (target probabilities),
              learning rate eta_t. Memory output must be transformable to a
              probability distribution: q_t = softmax(W @ k_t).
  Guarantees: Gradient is (q_t - p_t) @ k_t^T where q_t = softmax(W @ k_t).
              The softmax Jacobian is well-defined and the Wengert tape handles
              it natively. Loss is bounded below by 0 (when q = p).
  Cost:       O(d^2) for the matrix-vector product plus O(d) for softmax and
              cross-entropy. The softmax adds one exp + normalization pass
              compared to L2.
  Trade-off:  Natural for probabilistic memories (same loss as outer loop).
              But restricts memory output to simplex — cannot store arbitrary
              real-valued associations. Requires target v_t to be interpretable
              as a distribution (or be converted via softmax(v_t / tau)).
  Position:   specs/algorithms/attentional_biases/02_kl_objective.md
              Sibling of: 01_l1_sign.md (L1 bias), L2 (standard, in variants)
  Source:     Novel extension of MIRAS (2504.13173) §5.1 attentional bias
              framework. MIRAS documents L2, L1, Huber, l_p as bias choices;
              this spec extends to KL divergence (cross-entropy) by analogy.
              Note: MIRAS §5.2 Eq 18 applies f-divergence to RETENTION, not
              attentional bias — this spec applies KL to the BIAS knob instead.
```

## Motivation: Why KL as Attentional Bias?

<!-- HADES: Derived from miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10), extension beyond l_p to information-theoretic losses -->
```text
-- The MIRAS framework specifies the attentional bias as a free design choice.
-- Standard options (L2, L1, Huber) all measure error in Euclidean space.
-- KL divergence measures error in INFORMATION space.

-- Why this matters:
-- 1. The outer-loop language model minimizes cross-entropy (KL from data).
--    Using KL as the inner-loop bias aligns the memory's objective with
--    the model's objective — both optimize the same quantity.
--
-- 2. Softmax attention produces a probability distribution over keys:
--    softmax(Q K^T / sqrt(d)) normalizes dot products into weights
--    that sum to 1 — a categorical distribution over positions.
--    KL bias makes memory output a distribution too, aligning the
--    representational form of recurrent memory with attention.
--
-- 3. When memory stores next-token predictions (common in LM heads),
--    KL is the natural loss — it measures how many bits of information
--    the memory prediction is wrong by.
```

## The KL Attentional Bias

<!-- HADES: Derived from miras_equations/eq-010-lp-attentional-bias (§5.1), KL specialization of attentional bias knob -->
```text
-- KL attentional bias:
L(W; k_t, v_t) = KL(p_t || q_t)
               = sum_j p_{t,j} * log(p_{t,j} / q_{t,j})

-- where:
--   p_t = target distribution (derived from v_t)
--   q_t = softmax(W @ k_t) = memory's predicted distribution
--   j indexes the output dimensions

-- Equivalently (dropping the p log p entropy term, which is constant w.r.t. W):
L(W; k_t, v_t) = -sum_j p_{t,j} * log(q_{t,j}) + const
               = H(p_t, q_t)  (cross-entropy)

-- Minimizing KL(p || q) = minimizing cross-entropy H(p, q)
-- This is the SAME objective used by the outer-loop language model.
```

## Gradient and Update Rule

<!-- HADES: Derived from miras_equations/eq-011-lp-closed-form (§5.1 Eq 11), KL gradient via softmax Jacobian -->
```text
-- KL bias gradient:
FUNCTION: kl_bias_step(W: &mut Tensor, k: &Tensor, p: &Tensor,
                        alpha_t: f32, eta_t: f32) -> ()
  -- W: memory state [d, d]
  -- k: key vector [d, 1]
  -- p: target distribution [d, 1] (sums to 1, non-negative)

  -- Step 1: Memory output as distribution
  logits = W @ k                             -- [d, 1]
  q = softmax(logits)                        -- [d, 1], sums to 1

  -- Step 2: Cross-entropy gradient w.r.t. logits
  --   d(H(p, q)) / d(logits) = q - p
  --   This is the well-known softmax + cross-entropy gradient.
  grad_logits = q - p                        -- [d, 1]

  -- Step 3: Gradient w.r.t. W (outer product)
  grad_W = grad_logits @ k^T                 -- [d, d]

  -- Step 4: Retention + update
  W = (1 - alpha_t) * W - eta_t * grad_W

  -- Properties:
  --   When q = p (perfect prediction): grad = 0 (no update needed).
  --   When q is far from p: grad drives W to produce q closer to p.
  --   The gradient magnitude is bounded: ||q - p||_1 <= 2 (distributions).
```

## Comparison with L2 and L1

<!-- HADES: Derived from miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10), cross-bias comparison -->
```text
| Property          | L2 (standard)           | L1 (value-less)      | KL (cross-entropy)        |
|-------------------|-------------------------|----------------------|---------------------------|
| Loss              | ||Wk - v||^2            | ||Wk - v||_1         | KL(p || softmax(Wk))      |
| Gradient          | 2(Wk - v) @ k^T        | Sign(Wk - v) @ k^T  | (softmax(Wk) - p) @ k^T  |
| Output space      | R^d (unbounded)         | R^d (unbounded)      | Simplex (probabilities)   |
| Grad magnitude    | ∝ error magnitude       | Constant (±1)        | Bounded (≤ 2)             |
| Converges to      | W s.t. Wk ≈ v          | W s.t. Sign(Wk)=Sign(v) | W s.t. softmax(Wk) ≈ p |
| Best for          | General regression      | Entity detection     | Distribution matching     |
| Outer-loop align  | No (outer uses CE)      | No                   | Yes (same loss)           |
```

## Target Distribution Construction

The KL bias requires a target distribution p_t. There are several ways to
construct it from the raw value vector v_t:

<!-- HADES: Derived from miras_equations/eq-010-lp-attentional-bias (§5.1), target construction for KL bias -->
```text
-- Option 1: v_t is already a distribution (e.g., teacher softmax output)
p_t = v_t                                   -- directly from outer loop

-- Option 2: Temperature-scaled softmax of v_t
p_t = softmax(v_t / tau)                    -- tau controls sharpness
-- tau → 0: one-hot (hard target). tau → ∞: uniform (soft target).
-- tau = 1.0: standard softmax of value vector.

-- Option 3: One-hot from argmax of v_t
p_t = one_hot(argmax(v_t))                  -- hard classification target

-- Option 4: Label smoothing
p_t = (1 - eps) * one_hot(argmax(v_t)) + eps / d
-- eps = 0.1 (standard): prevents overconfident memory.

-- The choice of target construction is orthogonal to the KL bias itself.
-- It determines what "correct memory output" means for each token.
```

## Connection to MEMORA

KL attentional bias and KL retention (MEMORA) are distinct MIRAS knob choices
that can be combined:

<!-- HADES: miras_equations/eq-021-kl-softmax-update (§5.2 Eq 21, KL retention); miras_equations/eq-010-lp-attentional-bias (§5.1 Eq 10, attentional bias) -->
```text
-- KL retention (MEMORA, Eq 21): constrains MEMORY STATE to simplex
--   Ret_t(W, W') = KL(W || W')
--   Controls HOW MEMORY FORGETS (multiplicative decay via softmax).

-- KL attentional bias (this spec): uses KL as LOSS FUNCTION
--   L(W; k, v) = KL(p || softmax(W @ k))
--   Controls WHAT MEMORY LEARNS (match output to target distribution).

-- Combined: KL bias + KL retention
--   Memory state on simplex AND output loss is cross-entropy.
--   Doubly information-theoretic: both learning and forgetting
--   operate in the same space (distributions, bits).

-- Combined: KL bias + L2 retention (also valid)
--   Memory state unconstrained, but output loss is cross-entropy.
--   More representational freedom at the cost of losing the
--   simplex structure on the memory state itself.
```

## Gradient Derivation (for tape integration)

<!-- HADES: Derived from miras_equations/eq-010-lp-attentional-bias (§5.1), analytical VJP for KL bias -->
```text
-- Forward:
--   logits_t = W_{t-1} @ k_t                             -- [d, 1]
--   q_t = softmax(logits_t)                               -- [d, 1]
--   loss = -sum(p_t ⊙ log(q_t))                          -- scalar (cross-entropy)
--   grad_W_t = (q_t - p_t) @ k_t^T                       -- [d, d]
--   W_t = (1 - alpha_t) * W_{t-1} - eta_t * grad_W_t     -- [d, d]

-- Given: dL/dW_t (upstream gradient from outer loop)
-- Need: dL/dW_{t-1}, dL/dk_t, dL/dp_t, dL/dalpha_t, dL/deta_t

-- Step 1: Through retention + update
dL/dW_{t-1} (through decay) = (1 - alpha_t) * dL/dW_t
dL/dgrad_W_t = -eta_t * dL/dW_t

-- Step 2: Through outer product grad_W = (q - p) @ k^T
dL/d(q_t - p_t) = dL/dgrad_W_t @ k_t                     -- [d, 1]
dL/dk_t (through grad) = (q_t - p_t)^T @ dL/dgrad_W_t    -- [d, 1]

-- Step 3: Through softmax q = softmax(logits)
--   Softmax Jacobian: dq_i/dlogit_j = q_i * (delta_ij - q_j)
--   dL/dlogits = q_t ⊙ dL/dq_t - q_t * (q_t^T @ dL/dq_t)
--   where dL/dq_t = dL/d(q_t - p_t) (p_t is constant w.r.t. W)
dL/dlogits_t = q_t ⊙ dL/dq_t - q_t * dot(q_t, dL/dq_t)

-- Step 4: Through logits = W @ k
dL/dW_{t-1} (through logits) = dL/dlogits_t @ k_t^T
dL/dk_t (through logits) = W_{t-1}^T @ dL/dlogits_t

-- Combine:
dL/dW_{t-1} = (1 - alpha_t) * dL/dW_t + dL/dlogits_t @ k_t^T
dL/dk_t = (q_t - p_t)^T @ dL/dgrad_W_t + W_{t-1}^T @ dL/dlogits_t

-- Gate gradients (scalars):
dL/dalpha_t = -trace(W_{t-1}^T @ dL/dW_t)
dL/deta_t = -trace(grad_W_t^T @ dL/dW_t)
```

## Implementation Notes

1. **Theoretical extension**: KL attentional bias is not a named variant in
   the MIRAS paper. The paper explicitly states that attentional bias is a
   free design choice (§5.1) and documents L2, L1, Huber, and l_p. This spec
   extends the framework to KL divergence, which is natural for language
   modeling where cross-entropy is the standard outer-loop loss.

2. **Softmax numerics**: The softmax computation requires the log-sum-exp
   trick for numerical stability: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
   This is standard and well-optimized on all platforms.

3. **Target distribution**: The target p_t must be a valid probability
   distribution (non-negative, sums to 1). If v_t is a raw value vector,
   it must be converted via softmax or one-hot encoding before use as a
   KL target. Invalid targets (negative entries, non-normalized) produce
   undefined KL values.

4. **Bounded gradient**: The gradient `(q - p)` has L1 norm bounded by 2
   (since both q and p are distributions summing to 1). This provides
   natural gradient clipping without explicit thresholds — similar to l_1's
   bounded gradient property but emerging from the simplex constraint.

5. **Pluggable bias dispatch**: Registers as `AttentionalBias::KLDivergence`
   in the S3b-M2 infrastructure. The generic l_p dispatch (S3b-S11) does NOT
   subsume KL — it is a separate bias family requiring softmax output.

6. **Interaction with CMS (Continuous Memory System)**: At slow CMS levels
   (see `specs/infrastructure/scheduling/00_conductor.md`), KL bias memories
   accumulate distributional knowledge over longer horizons. Slow-frequency
   memories with KL bias naturally learn the long-run token distribution,
   while fast-frequency memories track local distributional shifts.

## Axiom Compliance

- **#4** (compressing context): KL bias compresses context into a probability distribution — the most information-efficient representation for categorical data.
- **#6** (optimizers are associative memory): Memory learns a distribution over values, making the connection to probabilistic associative memory explicit.
- **#9** (principled not ad hoc): KL divergence is the unique loss satisfying the axioms of information theory (Gibbs' inequality), making it maximally principled as a bias choice.
