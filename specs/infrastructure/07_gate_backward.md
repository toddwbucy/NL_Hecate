# Gate Backward: CUDA Gradient Accumulation for Alpha/Theta/Eta Gates

```
CONTRACT
  Purpose: Implement GPU-resident backward pass for gate_compute_cuda —
           accumulate d_w_alpha, d_b_alpha, d_w_theta, d_b_theta, d_w_eta,
           d_b_eta from per-token gate upstream gradients.
  Expects: Per-token gate outputs (alpha, theta, eta) cached from forward pass.
           Per-token scalar upstream gradients (d_alpha, d_theta, d_eta) from
           CUDA memory-rule backward kernels.
           Per-token gate inputs k_mem [bs*s, d], v_mem [bs*s, d].
  Guarantees: d_w_alpha[i] = sum_t( d_alpha[t] * alpha'(t) * concat(k_t,v_t)[i] )
              d_b_alpha[0] = sum_t( d_alpha[t] * alpha'(t) )
              and analogously for theta and eta.
              No forward-pass changes required (see Insight below).
  Cost: 1 CUDA kernel launch per active CMS level per backward step.
        O(T * 2d) work, T = bs * seq_len.
  Trade-off: Block size = min(2*d, 1024). Each thread handles one weight index.
             All threads redundantly accumulate the bias scalar (avoids branching);
             thread 0 writes the final bias value.
  Position: Called from accumulate_projection_grads in gpu_backward.rs after
            projection weight grads are computed.
  Source: MIRAS §2.2 gate definitions; Titans §A.1 momentum gate.
```

## Gate Definitions (forward)

```
alpha_t = sigmoid(W_alpha @ concat(k_t, v_t) + b_alpha)
theta_t = softplus(W_theta @ concat(k_t, v_t) + b_theta)
eta_t   = sigmoid(W_eta   @ concat(k_t, v_t) + b_eta)   [Titans only]
```

## Backward Derivations

### Alpha (sigmoid gate)

```
sigmoid'(logit) = sigmoid(logit) * (1 - sigmoid(logit)) = alpha * (1 - alpha)
d_w_alpha[i] = sum_t( d_alpha[t] * alpha_t * (1 - alpha_t) * concat(k_t, v_t)[i] )
d_b_alpha    = sum_t( d_alpha[t] * alpha_t * (1 - alpha_t) )
```

### Theta (softplus gate)

```
softplus'(logit) = sigmoid(logit)

Key insight: sigmoid(logit) = 1 - exp(-softplus(logit)) = 1 - exp(-theta)
  Proof: sigmoid(x) = 1/(1+exp(-x)) = 1 - 1/(1+exp(x)) = 1 - exp(-log(1+exp(x)))
       = 1 - exp(-softplus(x))  ✓

Therefore NO logit caching needed — theta (the softplus output) is sufficient.

d_w_theta[i] = sum_t( d_theta[t] * (1 - exp(-theta_t)) * concat(k_t, v_t)[i] )
d_b_theta    = sum_t( d_theta[t] * (1 - exp(-theta_t)) )
```

### Eta (sigmoid gate, Titans only)

```
Same as alpha — sigmoid derivative from the cached sigmoid output.
d_w_eta[i] = sum_t( d_eta[t] * eta_t * (1 - eta_t) * concat(k_t, v_t)[i] )
d_b_eta    = sum_t( d_eta[t] * eta_t * (1 - eta_t) )
```

## CUDA Kernel: gate_backward_kernel

```
Grid:  (1)          — single block; weight grad is a global reduction
Block: (min(2*d, 1024))  — one thread per weight index (d <= 512 in current ablations)

Thread i [0, 2*d):
  x_i  = k_mem[t*d + i]       for i < d
          v_mem[t*d + (i-d)]   for i >= d
  wa   += da_scalar * x_i     over t in [0, T)
  wt   += dt_scalar * x_i     over t in [0, T)  (if has_theta)
  we   += de_scalar * x_i     over t in [0, T)  (if has_eta)

All threads also accumulate bias sums (uniform per-token scalar — no branch divergence
from i; all threads reach same value; only thread 0 writes the result).

Memory access pattern:
  - alpha[t], theta[t], eta[t]: T reads, broadcast across warp (L1 cache hit)
  - k_mem[t*d+0..d-1]: coalesced across threads 0..d-1 at each t
  - v_mem[t*d+0..d-1]: coalesced across threads d..2d-1 at each t
```

## Files Modified

| File | Change |
|------|--------|
| `core/kernels/gate_backward.cu` | New kernel (Option A) |
| `core/build.rs` | Add gate_backward.cu to compilation |
| `core/src/cuda_ffi.rs` | `gate_backward_cuda` extern declaration |
| `core/src/dispatch.rs` | `gate_backward_dd` safe wrapper |
| `core/src/gpu_backward.rs` | `accumulate_projection_grads`: add k_mem/v_mem/alpha/theta/eta params; call gate_backward_dd |

## Interface

```rust
// dispatch.rs
pub fn gate_backward_dd(
    d_alpha: &GpuBuf<f32>, alpha: &GpuBuf<f32>,
    d_theta: Option<&GpuBuf<f32>>, theta: Option<&GpuBuf<f32>>,
    d_eta: Option<&GpuBuf<f32>>, eta: Option<&GpuBuf<f32>>,
    k_mem: &GpuBuf<f32>, v_mem: &GpuBuf<f32>,
    d_w_alpha: &mut GpuBuf<f32>, d_b_alpha: &mut GpuBuf<f32>,
    d_w_theta: &mut GpuBuf<f32>, d_b_theta: &mut GpuBuf<f32>,
    d_w_eta: &mut GpuBuf<f32>, d_b_eta: &mut GpuBuf<f32>,
    T: usize, d: usize,
)
```

## Rule-to-Gate Mapping

| Rule    | alpha | theta | eta |
|---------|-------|-------|-----|
| Delta   | yes   | yes   | no  |
| Titans  | yes   | yes   | yes |
| Hebbian | yes   | no    | no  |
| DGD     | yes   | yes   | no  |
