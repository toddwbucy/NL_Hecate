# L2 Key/Query Normalization for Memory Update Rules

## CONTRACT

- **Purpose**: Normalize projected keys and queries to unit L2 norm before memory operations, bounding memory update magnitude independent of d_model.
- **Expects**: Projected k_mem, q_mem vectors of shape [seq_len, d] after linear projection (and optional conv1d).
- **Guarantees**: ||k_t||_2 = 1 and ||q_t||_2 = 1 for all tokens t. Memory update outer product ||error * k_t^T||_F = ||error|| (no d-scaling). Gradients flow through normalization via standard Jacobian.
- **Cost**: O(seq_len * d) per forward/backward — negligible vs O(seq_len * d^2) memory update.
- **Trade-off**: Loses magnitude information in keys/queries. Gates (alpha, theta, eta) receive normalized inputs, changing their effective range. This is the paper-specified behavior.
- **Position**: Between projection+conv1d and the memory token loop. Applied to k_mem and q_mem only (not v_mem — values carry magnitude information for storage).
- **Source**: Titans (arXiv 2501.00663), Section "Architectural Details": "we use SiLU(.) activation as the non-linear activation for computing query, key, and values and normalize queries and keys using l_2-norm."

## Motivation

Without normalization, `||k_t|| ~ O(sqrt(d))`. The memory update contains the outer product:
```text
grad = (M @ k_t - v_t) * k_t^T
```
`||grad||_F` scales as `||k_t||^2 ~ d`. At d=1024, memory updates are ~1024x larger than intended, causing M_norm divergence and NaN within ~1300 steps. At d=512 the `m_norm_clamp` catches it; at d=1024 it overwhelms the clamp.

With normalization, `||k_t|| = 1` regardless of d, and `||grad||_F` depends only on the prediction error magnitude.

## Per-Row L2 Normalization

Forward (per token row):
```text
norm_t = ||k_raw_t||_2
k_norm_t = k_raw_t / max(norm_t, eps)
```
`eps` = 1e-8 (prevents division by zero for degenerate rows).

Backward (per token row):
```text
dot_t = <d_k_norm_t, k_norm_t>
d_k_raw_t = (d_k_norm_t - k_norm_t * dot_t) / max(norm_t, eps)
```

## Application Points

### CPU Path (titans_lmm.rs)
- Forward: After projection+conv1d, before token loop. Normalize k_mem and q_mem in-place, store norms in TitansLMMCache.
- Backward: After reverse token loop and conv1d backward, before projection backward. Apply normalization Jacobian to d_k_mem and d_q_mem using cached norms.

### GPU Path (gpu_forward.rs / gpu_backward.rs)
- Forward: After cuBLAS projection, before gate computation. CUDA kernel normalizes k_mem and q_mem in-place, returns norms buffer.
- Backward: After CUDA memory backward, before weight gradient computation. CUDA kernel applies normalization Jacobian to d_k_mem and d_q_mem.

### CUDA Kernels (l2_normalize.cu)
- `l2_normalize_rows_f32_cuda`: Grid=(n_rows), Block=(min(d, 1024)). Warp reduction for norm, then broadcast divide.
- `l2_normalize_backward_f32_cuda`: Same grid/block. Warp reduction for dot product, then Jacobian application.

## Scope

Applied to all matrix-memory rules (Titans LMM, Delta Rule, Hebbian, DGD) since all share the outer-product update structure. The normalization is applied at the projection level in gpu_forward.rs, which is shared across all rules.
