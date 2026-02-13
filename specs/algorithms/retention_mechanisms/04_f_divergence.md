# f-Divergence Retention (General Framework)

```
CONTRACT
  Purpose:    The most general retention framework. ALL other retention
              mechanisms are special cases of f-divergence. Provides the
              theoretical foundation that unifies L2, KL, elastic net,
              and any future retention mechanism.
  Expects:    A convex function f with f(1)=0. Memory states W, W_prev.
  Guarantees: Every f-divergence defines a valid retention mechanism.
              The choice of f determines the geometry of forgetting.
              L2 = f(t) = (t-1)^2. KL = f(t) = t*log(t). Chi-squared = f(t) = (t-1)^2/t.
  Cost:       Depends on f. Can range from O(1) (L2) to O(d) (KL).
  Trade-off:  Maximum generality. But abstract — must be instantiated to
              a specific f before it can be implemented. The generality
              is for the specification, not the runtime.
  Position:   specs/algorithms/retention_mechanisms/04_f_divergence.md
  Source:     MIRAS (2504.13173) Table 1, general framework
```

## Definition

```
FUNCTION: f_divergence(P, Q, f) -> Scalar
  -- D_f(P || Q) = sum_i Q[i] * f(P[i] / Q[i])
  -- where f is convex with f(1) = 0

  return sum(Q * f(P / Q))
```

## Instantiation Table

| f(t) | D_f | Name | Resulting Retention | Used By |
|---|---|---|---|---|
| (t-1)^2 | Chi-squared | L2 weight decay | (1-alpha)*W | Titans, Atlas, Trellis |
| t*log(t) | KL divergence | KL retention | softmax update | MEMORA |
| -log(t) | Reverse KL | — | — | (theoretical) |
| t^p - 1 | Alpha-divergence | l_p retention | L_q norm decay | MONETA |
| \|t-1\| | Total variation | Hard thresholding | Sign-based forgetting | (theoretical) |

## MIRAS Decomposition under f-Divergence

```
Ret_t(W, W_{t-1}) = D_f(W || W_{t-1})  +  G_f(W)

-- D_f is the LOCAL retention: how far can W move from W_{t-1}?
-- G_f is the GLOBAL retention: how constrained is W overall?
-- The geometry of the constraint set depends on f.

-- L2 (f = (t-1)^2):  Euclidean ball around W_{t-1}
-- KL (f = t*log(t)):  Probability simplex near W_{t-1}
-- TV (f = |t-1|):     Diamond (L1 ball) around W_{t-1}
```

## Why This Matters

The f-divergence framework proves that retention is a DESIGN CHOICE, not a fixed
mechanism. Every convex f gives a valid retention mechanism. The papers explore
three points in this space (L2, KL, elastic net). There are infinitely many others.

For NL_Hecate, this means the RetentionMechanism trait is OPEN — new f-divergences
can be implemented without modifying any existing code. The trait system ensures
any valid implementation can plug into any MemoryUpdateRule.

## Trait Implication

```
-- Any convex f with f(1)=0 defines a valid RetentionMechanism.
-- This is the "open for extension" guarantee of the trait system.

TRAIT: FDivergenceRetention implements RetentionMechanism
  f: Fn(f32) -> f32                    -- convex, f(1)=0
  f_prime: Fn(f32) -> f32              -- derivative of f
  f_star: Fn(f32) -> f32              -- convex conjugate (for closed-form solutions)

  COMPUTE_RETENTION(W, W_prev, gates):
    return gates.eta * f_divergence(W, W_prev, self.f)

  APPLY_RETENTION(W_prev, gates):
    -- Closed form depends on specific f
    -- Must be provided by each instantiation
```

## Axiom Compliance

- **MIRAS IS #1** (orthogonal design choices): Retention is independent of bias/structure/algorithm
- CS-36 compliance: The general framework; cannot be restricted to any single f
