# Short Causal Conv1D on Keys and Queries

<!-- HADES: Derived from atlas_equations/eq-001-causal-attention (Atlas §2.1, preprocessing convention); hope_equations/eq-071-arch-variant2 (HOPE §6 architecture block); H3 (Fu et al. 2023); Hyena (Poli et al. 2023); Based (Arora et al. 2024); Mamba (Gu & Dao 2024) -->
```text
CONTRACT
  Purpose:    A short depthwise causal convolution (kernel size 3-4) applied to
              keys and queries BEFORE they enter the memory module. This is not a
              novel contribution of the NL papers — it is a standard preprocessing
              step adopted from the SSM literature (H3, Hyena, Mamba, Based,
              DeltaNet, Gated Linear Attention). Its purpose: inject local token
              interactions into the key/query streams that pointwise linear
              projections alone cannot capture. Without it, each token's key is
              a purely local function of its own embedding — the conv provides a
              small receptive field (3-4 neighboring tokens) for richer keys.

              Why keys and queries, not values?
              Keys determine WHAT gets stored in memory (M += v @ k^T).
              Queries determine WHAT gets retrieved (y = M @ q).
              Values are the CONTENT being stored — they benefit less from
              local mixing because the content is already token-specific.
              Applying conv to values would blur the signal being stored.

              Why causal?
              Autoregressive: token t must not see token t+1. The convolution
              uses only current and past positions (left-padded, no future leak).

              Why depthwise?
              Each head dimension is convolved independently. No cross-channel
              mixing — that is the projection matrices' job. Depthwise keeps the
              parameter count minimal: kernel_size * d_model (not kernel_size * d^2).

  Expects:    Projected key tensor [batch, seq_len, d_model] (after W_K @ x).
              Projected query tensor [batch, seq_len, d_model] (after W_Q @ x).
              Causal conv weights [d_model, kernel_size] (outer_loop_param).
              Optional: separate conv weights for keys and queries (recommended).
  Guarantees: Output shape identical to input: [batch, seq_len, d_model].
              Causal: output at position t depends only on positions [t-kernel_size+1, t].
              Left-padded with zeros for the first kernel_size-1 positions.
              Gradients flow through the tape to the conv weights and the upstream
              projection weights — the conv is a differentiable operation on the tape.
  Cost:       O(seq_len * d_model * kernel_size) per forward pass.
              With kernel_size = 4: 4× the cost of the identity (negligible vs
              O(d^2) memory rule or O(T*w) attention).
              Parameters: 2 * d_model * kernel_size (separate key/query convs).
              For d=1024, kernel=4: 8,192 params — negligible vs millions in projections.
  Trade-off:  Minimal cost for measurable quality improvement. The conv is so cheap
              that there is no reason not to include it. The only question is kernel
              size: 3 (H3/Hyena) vs 4 (Mamba/Based/Atlas convention). We default to 4
              following the majority convention.
  Position:   specs/infrastructure/attention/02_short_conv.md
              Sibling of: 00_attention.md (standard attention)
              Referenced by: memory_update_rules (all rules that take k, q inputs)
  Source:     Atlas (2505.23735) architecture convention;
              H3 (Fu et al. 2023, arXiv 2212.14052) — originated short conv for SSMs;
              Hyena (Poli et al. 2023, arXiv 2302.10866) — generalized to long conv;
              Mamba (Gu & Dao 2024, arXiv 2312.00752) — standardized kernel_size=4;
              Based (Arora et al. 2024, arXiv 2402.18668) — conv before linear attention;
              DeltaNet (Yang et al. 2024) — conv on keys before delta rule
```

## Why Short Convolution Helps Memory

<!-- HADES: Derived from atlas_equations/eq-001-causal-attention (Atlas §2.1); H3 (Fu et al. 2023), Section 3.1 -->
```text
-- Without conv, keys are strictly pointwise:
--   k_t = W_K @ x_t          -- each key depends only on its own token
--
-- With conv, keys see a local window:
--   k_raw_t = W_K @ x_t      -- project first
--   k_t = conv1d(k_raw, kernel_size=4)[t]  -- then mix locally
--       = sum_{j=0}^{3} w[j] * k_raw_{t-j}
--
-- Why this matters for memory:
--   Memory stores via:  M += v_t @ k_t^T  (or gradient variant)
--   Memory reads via:   y_t = M @ q_t
--
--   If k_t depends only on x_t, the memory's addressing resolution is
--   limited to single-token features. The conv gives keys a 4-token
--   receptive field, enabling the memory to distinguish patterns like
--   "this token following those 3 tokens" vs "this token in isolation."
--
--   Empirically, this consistently improves perplexity by 0.1-0.5 across
--   SSM architectures (H3, Mamba, Based, DeltaNet, Atlas) at negligible cost.
--
-- The conv is applied AFTER projection but BEFORE the memory module.
-- This is the standard convention across all SSM-family architectures.
```

## Depthwise Causal Conv1D Operation

<!-- HADES: Derived from H3 (Fu et al. 2023, arXiv 2212.14052); Mamba (Gu & Dao 2024, arXiv 2312.00752) -->
```text
FUNCTION: causal_conv1d(x: &Tensor,           -- [batch, seq_len, d_model]
                        w: &Tensor,            -- [d_model, kernel_size]
                        bias: Option<&Tensor>,  -- [d_model] optional
                       ) -> Tensor             -- [batch, seq_len, d_model]

  -- Depthwise: each channel convolved independently (no cross-channel)
  -- Causal: left-padded so output[t] depends only on input[t-K+1..t]

  LET K = w.shape[1]  -- kernel_size (typically 4)

  -- Left-pad input with K-1 zeros for causality
  x_padded = zero_pad_left(x, K - 1)   -- [batch, seq_len + K - 1, d_model]

  -- Depthwise convolution
  FOR b in 0..batch:
    FOR t in 0..seq_len:
      FOR c in 0..d_model:
        out[b, t, c] = sum_{j=0}^{K-1} w[c, j] * x_padded[b, t + j, c]
        IF bias.is_some():
          out[b, t, c] += bias[c]

  -- Activation: SiLU (Swish) — standard in Mamba/Based convention
  out = out * sigmoid(out)

  RETURN out

-- Note on activation placement:
--   H3/Hyena: no activation after conv (conv is purely linear mixing)
--   Mamba/Based/Atlas: SiLU after conv (adds nonlinearity to local mixing)
--   We follow the Mamba convention (SiLU) as the default but make it
--   configurable. The activation is a minor detail — both work.
```

## Integration Point: Before Memory, After Projection

<!-- HADES: Derived from atlas_equations/eq-001-causal-attention (Atlas §2.1); hope_equations/eq-071-arch-variant2 (HOPE §6 architecture) -->
```text
-- Data flow for a single NL block (any composition pattern):
--
-- 1. Input embedding: x_t [batch, seq_len, d_model]
--
-- 2. Linear projections (outer_loop_param):
--    k_raw = x @ W_K    -- [batch, seq_len, d_model]
--    q_raw = x @ W_Q    -- [batch, seq_len, d_model]
--    v     = x @ W_V    -- [batch, seq_len, d_model]
--
-- 3. Short causal conv1d (THIS SPEC):
--    k = causal_conv1d(k_raw, w_k_conv)   -- local token mixing on keys
--    q = causal_conv1d(q_raw, w_q_conv)   -- local token mixing on queries
--    -- v is NOT convolved (values are token-specific content)
--
-- 4. Memory module (per-token loop):
--    FOR t in 0..seq_len:
--      y_t = memory_read(M, q[t])         -- retrieve
--      memory_update(M, k[t], v[t])       -- store
--
-- 5. Composition pattern (MAC/MAG/MAL):
--    Combines memory output y with attention output
--
-- The conv is a simple preprocessing step that slots between steps 2 and 4.
-- It does not change the memory module's interface or the composition pattern.
-- All memory rules (Titans, Delta, Hebbian, Omega, ...) receive the same
-- convolved keys and queries regardless of which rule is active.
```

## Separate vs Shared Conv Weights

<!-- HADES: Derived from H3 (Fu et al. 2023); Based (Arora et al. 2024) -->
```text
-- Two options for conv weight sharing:

-- OPTION A: Separate conv weights for keys and queries (RECOMMENDED)
--   w_k_conv: [d_model, kernel_size]   -- key-specific local mixing
--   w_q_conv: [d_model, kernel_size]   -- query-specific local mixing
--   Total: 2 * d_model * kernel_size params
--   Reasoning: keys and queries serve different roles — keys are WRITTEN
--   to memory, queries are used to READ from memory. Different local
--   mixing patterns may be optimal for writing vs reading.

-- OPTION B: Shared conv weights
--   w_conv: [d_model, kernel_size]     -- shared across keys and queries
--   Total: d_model * kernel_size params
--   Reasoning: half the parameters. May be sufficient when d_model is small.

-- Default: Option A (separate). The parameter overhead is negligible
-- (8K params for d=1024, kernel=4) and gives the model more flexibility.
-- Option B available as configuration for memory-constrained settings.

-- Per-CMS-level configuration:
--   Each CMS level has its own memory module with its own key/query projections.
--   Each level gets its own conv weights — separate_per_level = true (default).
--   This is consistent with all other per-level outer_loop_params (W_K, W_V, gates).
--   Fast levels (Level 0) may benefit from longer kernels (more local context).
--   Slow levels (Level 3) may benefit from shorter kernels (less noise aggregation).
--   Default: kernel_size = 4 for all levels (override per level if profiling shows benefit).
```

## State Lifetime and Tape Integration

<!-- HADES: Derived from specs/infrastructure/state_lifecycle/00_state_ownership.md; specs/infrastructure/differentiation/00_wengert_tape.md -->
```text
-- Conv weights are outer_loop_param:
--   - Persist across forward calls
--   - Modified by outer-loop optimizer (AdamW/AdaMuon)
--   - Serialized with checkpoint
--   - Frequency-gated: only updated at active CMS levels (CS-27/28)

-- The conv operation is differentiable and recorded on the Wengert tape:
--   Forward: tape records conv inputs (x_padded, w) and output
--   Backward: standard conv1d backward produces dL/dx and dL/dw
--     dL/dw[c, j] = sum_{b,t} dL/dout[b, t, c] * x_padded[b, t + j, c]
--     dL/dx_padded[b, t + j, c] += dL/dout[b, t, c] * w[c, j]
--     (then strip the padding from dL/dx_padded to get dL/dx)
--   If SiLU activation is included, the backward chains through it first.

-- The conv backward is lightweight — same O(seq_len * d_model * kernel_size)
-- as forward. No special handling needed for the tape.

-- No inner_loop_state: the conv is purely feedforward (no recurrence).
-- No context_memory: nothing persists across chunks for the conv itself.
-- The conv is stateless — it transforms each chunk independently.
```

## VJP Gradient Derivation

<!-- HADES: Derived from standard depthwise conv1d backward (no paper-specific equation — this is textbook) -->
```text
-- Forward:
--   x_pad = zero_pad_left(x, K-1)                     -- [B, T+K-1, D]
--   z[b,t,c] = sum_{j=0}^{K-1} w[c,j] * x_pad[b,t+j,c] + bias[c]
--   out = z * sigmoid(z)                               -- SiLU activation

-- Given: dL/dout [B, T, D]

-- Step 1: Backward through SiLU
--   sigmoid_z = sigmoid(z)
--   dL/dz = dL/dout * (sigmoid_z + z * sigmoid_z * (1 - sigmoid_z))
--         = dL/dout * (sigmoid_z * (1 + z * (1 - sigmoid_z)))

-- Step 2: Backward through depthwise conv
--   dL/dw[c, j] = sum_{b=0}^{B-1} sum_{t=0}^{T-1} dL/dz[b,t,c] * x_pad[b,t+j,c]
--
--   dL/dx_pad[b, t+j, c] += dL/dz[b,t,c] * w[c,j]
--     for all (b, t, c, j) — this is the transposed (correlation) conv
--
--   dL/dx[b, t, c] = dL/dx_pad[b, t + K - 1, c]      -- strip left padding
--
--   dL/dbias[c] = sum_{b,t} dL/dz[b,t,c]              -- if bias is used

-- Step 3: Tape propagation
--   dL/dx flows upstream to the projection backward: dL/dW_K, dL/dx_input
--   dL/dw and dL/dbias are outer_loop_param gradients for the optimizer

-- Kernel-pair pattern:
--   Rust reference: straightforward implementation of above
--   CUDA kernel (optional): fused depthwise conv1d forward + SiLU
--   CUDA backward kernel: fused conv1d backward + SiLU backward
--   The CUDA kernels are well-known (used in Mamba's open-source implementation)
```

## Configuration Defaults

<!-- HADES: Derived from Mamba (Gu & Dao 2024, arXiv 2312.00752) experimental setup; Based (Arora et al. 2024) -->
```text
-- Default configuration:
--   kernel_size:      4        (Mamba/Based/Atlas convention)
--   activation:       silu     (Mamba convention; "none" also valid)
--   separate_kq:      true     (separate conv weights for keys and queries)
--   use_bias:         true     (small bias term per channel)
--   per_level:        true     (separate weights per CMS level)
--
-- Initialization:
--   Conv weights: Kaiming uniform (fan_in = kernel_size)
--   Bias: zeros
--   These are standard initializations — nothing NL-specific here.
--
-- When to increase kernel_size:
--   For long-range local patterns (e.g., code with consistent indentation),
--   kernel_size = 7-8 may help. But longer kernels add cost linearly and
--   risk overfitting local patterns. Profile before increasing beyond 4.
--
-- When to disable (kernel_size = 1):
--   kernel_size = 1 reduces to a pointwise scaling (no local mixing).
--   This effectively removes the conv. Useful as a baseline comparison.
--   With bias and SiLU, kernel_size=1 is equivalent to a gated linear unit.
```

## Why Not Convolve Values?

<!-- HADES: Derived from Based (Arora et al. 2024, arXiv 2402.18668); DeltaNet (Yang et al. 2024) -->
```text
-- The decision to convolve keys and queries but NOT values is deliberate:
--
-- Keys control memory ADDRESSING:
--   M += v_t @ k_t^T   → k_t determines WHERE in memory v_t is stored
--   Local mixing in keys helps: "store this based on the local pattern"
--   gives better addressing than "store this based on this token alone."
--
-- Queries control memory RETRIEVAL:
--   y_t = M @ q_t      → q_t determines WHAT is retrieved from memory
--   Local mixing in queries helps: "retrieve based on the local context"
--   gives more precise retrieval than single-token queries.
--
-- Values are CONTENT:
--   v_t is the information being stored in memory.
--   Convolving values would blur the stored content across neighbors.
--   This is usually undesirable — you want to store each token's content
--   precisely, not a mixture of nearby tokens' content.
--
-- Exception: Some architectures (Hyena, early H3) convolve all three.
-- The empirical consensus (Mamba, Based, DeltaNet, Atlas) is keys+queries only.
-- We follow this consensus but the conv function can be applied to values
-- via configuration if experimentation shows benefit.
```

## Implementation Notes

1. **Reuses standard conv1d**: This is a textbook depthwise causal convolution.
   No novel algorithm. Existing CUDA kernels from Mamba's `causal-conv1d` package
   can serve as reference for the CUDA kernel pair.

2. **Fused kernel opportunity**: The projection (W_K @ x) followed by conv1d
   followed by SiLU could be fused into a single kernel. This eliminates two
   intermediate tensor materializations. The Mamba codebase demonstrates this
   pattern. Not required for correctness — optimize after profiling.

3. **Streaming state**: For serving (continuous token processing), the conv
   needs to remember the last `kernel_size - 1` tokens' projected keys/queries
   across chunks. This is a tiny buffer (3 * d_model floats for kernel_size=4)
   stored in `context_memory`. During training with fixed chunk sizes, left-padding
   with zeros at each chunk boundary is acceptable (the boundary artifacts are
   small and decrease with larger chunk sizes).

4. **Multi-head layout**: If using per-head key/query dimensions (d_k = d_model / n_heads),
   the conv operates on the full d_model before head splitting. Alternatively,
   reshape to [batch, seq_len, n_heads, d_k] and convolve per-head. Both are valid —
   the full-d_model approach is simpler and standard.

5. **Interaction with feature maps (phi)**: For Atlas Omega with polynomial feature
   maps (Eq 11), the order is: project → conv → feature map → memory. The conv
   operates on the raw projected keys, and phi expands the dimensionality afterward.
   This avoids convolving in the expanded feature space (which would be more expensive
   and less effective since phi introduces nonlinearity that should not be smoothed).

## Axiom Compliance

- **MIRAS IS #1** (orthogonal design choices): The short conv is orthogonal to all four MIRAS knobs and to the composition pattern. It is a preprocessing step applied uniformly to keys/queries regardless of which memory rule, bias, retention, or algorithm is selected.
- **NL IS NOT #5** (not optimizers as just optimizers): The conv weights are standard outer_loop_params — they are NOT part of the memory's inner loop. The conv is a fixed preprocessing step, not a memory mechanism.
- **CS-18** (forward pass IS the API): The conv is part of the forward data flow. No separate preprocessing phase — it runs inline during the forward pass.
- **CS-40** (opt-in AD): The conv is recorded on the tape when `with_tape()` is active. When tape is off (inference), the conv still runs (forward-only) but no backward is recorded.
