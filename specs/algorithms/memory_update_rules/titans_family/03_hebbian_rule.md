# Hebbian Rule

```
CONTRACT
  Purpose:    Simplest memory update. No gradient computation — direct
              outer product association. The "grandmother cell" of memory rules.
  Expects:    Input token embedding x_t.
              Pulse with frequency level activity status.
  Guarantees: Output y_t. Memory updated via direct k-v association.
              No gradient computation needed (not even analytical).
              Cheapest per-token cost of any memory rule.
  Cost:       Per token: O(d_out * d_in) for outer product + read.
              No gradient: saves the matrix-vector multiply for prediction.
              Memory: O(d_out * d_in) for M only.
  Trade-off:  Fastest and simplest, but cannot correct errors. Once an
              association is written, it persists until decayed. Titans LMM
              can overwrite wrong associations; Hebbian cannot.
  Position:   specs/algorithms/memory_update_rules/titans_family/03_hebbian_rule.md
              Child of 01_titans_lmm.md (special case: eta=0, no gradient)
  Source:     MIRAS (2504.13173) Eq 8, Table 1 row "Hebbian"
```

## MIRAS Configuration

| Knob | Setting |
|---|---|
| Memory Structure | Matrix M in R^{d_out x d_in} |
| Attentional Bias | Dot-product similarity (not L2 regression) |
| Retention | L2 weight decay: (1 - alpha_t) * M |
| Algorithm | Direct association (no gradient, no optimization) |

CS-33 compliance: Hebbian uses dot-product, NOT L2. This is a fundamentally different
learning rule — not a simplified Titans LMM. The MIRAS framework distinguishes them.

## State

```
STATE: HebbianState
  M: Tensor(d_out, d_in)     -- memory matrix (inner_loop_state)

OUTER_PARAMS: HebbianParams
  W_K: Tensor(d_in, d_model)
  W_V: Tensor(d_out, d_model)
  W_Q: Tensor(d_out, d_model)
  W_alpha: Tensor(1, d_in + d_out)   -- retain gate projection (only gate needed)
```

## Pseudocode

```
ALGORITHM: hebbian_step(state: &mut HebbianState, x_t: &Tensor,
                        outer: &HebbianParams, pulse: &Pulse) -> Tensor
  k_t = x_t @ outer.W_K^T
  v_t = x_t @ outer.W_V^T
  q_t = x_t @ outer.W_Q^T

  alpha_t = sigmoid(concat(k_t, v_t) @ outer.W_alpha^T)

  IF NOT pulse.is_active(self.level):
    return state.M @ q_t

  -- Direct association: v @ k^T (MIRAS Eq 8)
  -- No gradient computation. No loss function evaluation.
  -- "See k, store v alongside it."
  state.M = (1 - alpha_t) * state.M + outer_product(v_t, k_t)

  y_t = state.M @ q_t
  return y_t
```

## Relationship to Titans LMM

```
-- Titans LMM inner loop:
grad = 2 * (M @ k - v) @ k^T     -- compute gradient of L2 loss
S = eta * S - theta * grad         -- accumulate momentum
M = (1 - alpha) * M + S           -- update memory

-- Hebbian: skip ALL of the above, replace with:
M = (1 - alpha) * M + v @ k^T    -- direct association

-- Mathematically: Hebbian is NOT eta=0 Delta Rule.
-- Delta Rule: M += -theta * 2(Mk-v)@k^T  (corrects for existing associations)
-- Hebbian:    M += v @ k^T                (ignores existing associations)
-- Hebbian treats every token as if memory were empty.
```

## Parallelization Support

```
SUPPORTED_PARALLELIZATION:
  - ChunkwiseGD:      YES  (trivially — no gradient to approximate)
  - AssociativeScan:   YES  (update IS a linear recurrence — see below)
  - TNTHierarchical:   YES  (cheap enough to be a local memory in TNT)

AssociativeScan compatibility:
  M_t = (1 - alpha_t) * M_{t-1} + v_t @ k_t^T

  This has the form s_t = a_t * s_{t-1} + b_t where:
    a_t = (1 - alpha_t)       -- data-dependent, NOT state-dependent
    b_t = v_t @ k_t^T         -- data-dependent, NOT state-dependent

  alpha_t comes from the gate (function of k_t, v_t), NOT from M.
  The update is LINEAR in M — exactly the "linear attention decay" form
  listed in the associative scan spec (02_associative_scan.md).
  Hebbian is the ONLY Titans-family rule where the full memory update
  (not just momentum) can be parallelized via associative scan.
```

## When to Use Hebbian

- When compute budget is extremely tight (no gradient compute)
- When associations are clean (no conflicting key-value pairs)
- As a fast "capture everything" initial memory, refined by a slower rule at another CMS level
- MIRAS empirical finding: Hebbian variants perform surprisingly well in hybrid architectures

## Axiom Compliance

- **NL IS #4** (compressing context): Direct compression of k-v pairs into matrix
- **NL IS #6** (optimizers are memory): The outer product IS the "optimizer" — it's just one with zero learning rate
- **NL IS NOT #3** (not static): Still data-dependent gating on alpha_t
