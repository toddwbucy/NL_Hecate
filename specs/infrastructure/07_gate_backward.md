# Gate Backward: CUDA Gradient Accumulation for Alpha/Theta/Eta Gates

```text
CONTRACT
  Purpose: Implement GPU-resident backward pass for gate_compute_cuda —
           accumulate d_w_alpha, d_b_alpha, d_w_theta, d_b_theta, d_w_eta,
           d_b_eta from per-token gate upstream gradients.
  Expects: Per-token gate outputs (alpha, theta, eta) cached from forward pass.
           Per-token scalar upstream gradients (d_alpha, d_theta, d_eta) from
           CUDA memory-rule backward kernels.
           Per-token gate inputs k_mem [bs*s, d], v_mem [bs*s, d].
           d <= 512 (enforced by assert in gate_backward_cuda): the single-block
           design assigns one thread per weight index in [0, 2*d); 2*d must fit
           within the 1024-thread CUDA block limit.
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
  Source: MIRAS §2.2 gate definitions (NL HADES: miras_equations/table1-titans-lmm-write
          covers the Titans LMM write rule; no dedicated node for §2.2 alpha/theta
          definitions — they appear inline in Table 1 entries).
          Titans §A.1 momentum gate (NL HADES: titans_equations — no dedicated node;
          eta gate defined inline in §A.1 alongside the momentum accumulator S).
```

## Gate Definitions (forward)

```text
alpha_t = sigmoid(W_alpha @ concat(k_t, v_t) + b_alpha)
theta_t = softplus(W_theta @ concat(k_t, v_t) + b_theta)
eta_t   = sigmoid(W_eta   @ concat(k_t, v_t) + b_eta)   [Titans only]
```

## Backward Derivations

### Alpha (sigmoid gate)

```text
sigmoid'(logit) = sigmoid(logit) * (1 - sigmoid(logit)) = alpha * (1 - alpha)
d_w_alpha[i] = sum_t( d_alpha[t] * alpha_t * (1 - alpha_t) * concat(k_t, v_t)[i] )
d_b_alpha    = sum_t( d_alpha[t] * alpha_t * (1 - alpha_t) )
```

### Theta (softplus gate)

```text
softplus'(logit) = sigmoid(logit)

Key insight: sigmoid(logit) = 1 - exp(-softplus(logit)) = 1 - exp(-theta)
  Proof: sigmoid(x) = 1/(1+exp(-x)) = 1 - 1/(1+exp(x)) = 1 - exp(-log(1+exp(x)))
       = 1 - exp(-softplus(x))  ✓

Therefore NO logit caching needed — theta (the softplus output) is sufficient.

d_w_theta[i] = sum_t( d_theta[t] * (1 - exp(-theta_t)) * concat(k_t, v_t)[i] )
d_b_theta    = sum_t( d_theta[t] * (1 - exp(-theta_t)) )
```

### Eta (sigmoid gate, Titans only)

```text
Same as alpha — sigmoid derivative from the cached sigmoid output.
d_w_eta[i] = sum_t( d_eta[t] * eta_t * (1 - eta_t) * concat(k_t, v_t)[i] )
d_b_eta    = sum_t( d_eta[t] * eta_t * (1 - eta_t) )
```

## CUDA Kernel: gate_backward_kernel

Grid: `(1)` — single block. Block: `(2*d)` — one thread per weight index, d <= 512.

```rust
// Pseudocode — mirrors gate_backward_kernel in gate_backward.cu exactly.
// CUDA maps each thread to one i; the loop over t is serial within each thread.
fn accumulate_weights(
    d_alpha: &[f32],               // [T] upstream gate gradients
    alpha:   &[f32],               // [T] cached sigmoid outputs
    d_theta: Option<&[f32]>,       // [T] upstream; None for Hebbian
    theta:   Option<&[f32]>,       // [T] cached softplus outputs; None for Hebbian
    d_eta:   Option<&[f32]>,       // [T] upstream; None for Delta/DGD/Hebbian
    eta:     Option<&[f32]>,       // [T] cached sigmoid outputs; None for Delta/DGD/Hebbian
    k_mem:   &[f32],               // [T, d] gate input — first half of concat(k,v)
    v_mem:   &[f32],               // [T, d] gate input — second half of concat(k,v)
    d_w_alpha: &mut [f32],         // [2*d] weight grad output
    d_b_alpha: &mut [f32],         // [1]   bias grad output
    d_w_theta: Option<&mut [f32]>, // [2*d] weight grad; None if !has_theta
    d_b_theta: Option<&mut [f32]>, // [1]   bias grad; None if !has_theta
    d_w_eta:   Option<&mut [f32]>, // [2*d] weight grad; None if !has_eta
    d_b_eta:   Option<&mut [f32]>, // [1]   bias grad; None if !has_eta
    d: usize,                      // hidden dim; must satisfy d <= 512
    T: usize,                      // total tokens = batch_size * seq_len
    thread_idx: usize,             // CUDA threadIdx.x in [0, 2*d)
) {
    let i = thread_idx;

    // Private per-thread weight accumulators — no cross-thread reduction needed.
    // Each i in [0, 2*d) is owned exclusively by one thread (no write conflict).
    let (mut wa, mut wt, mut we) = (0.0_f32, 0.0_f32, 0.0_f32);
    // Bias accumulators — per-token scalars are uniform across i, so every
    // thread reaches the same value; only thread 0 writes the final result.
    let (mut ba, mut bt, mut be) = (0.0_f32, 0.0_f32, 0.0_f32);

    for t in 0..T {
        // Load concat(k_t, v_t)[i]: first d dims from k_mem, next d from v_mem.
        let x_i = if i < d { k_mem[t * d + i] } else { v_mem[t * d + (i - d)] };

        // Alpha (sigmoid): da_scalar = d_alpha[t] * alpha[t] * (1 - alpha[t])
        let a  = alpha[t];
        let da_scalar = d_alpha[t] * a * (1.0 - a);
        wa += da_scalar * x_i;
        ba += da_scalar;

        // Theta (softplus): dt_scalar = d_theta[t] * (1 - exp(-theta[t]))
        // Identity: sigmoid(logit) = 1 - exp(-softplus(logit)) — no logit cache needed.
        if let (Some(dt_up), Some(th)) = (d_theta, theta) {
            let dt_scalar = dt_up[t] * (1.0 - (-th[t]).exp());
            wt += dt_scalar * x_i;
            bt += dt_scalar;
        }

        // Eta (sigmoid, Titans only): de_scalar = d_eta[t] * eta[t] * (1 - eta[t])
        if let (Some(de_up), Some(et)) = (d_eta, eta) {
            let e  = et[t];
            let de_scalar = de_up[t] * e * (1.0 - e);
            we += de_scalar * x_i;
            be += de_scalar;
        }
    }

    // Each thread writes its own index directly — no atomics, no shared memory reduction.
    d_w_alpha[i] = wa;
    if let Some(dw) = d_w_theta { dw[i] = wt; }
    if let Some(dw) = d_w_eta   { dw[i] = we; }

    // Bias: all threads computed the same scalar; thread 0 is the single writer.
    if i == 0 {
        d_b_alpha[0] = ba;
        if let Some(db) = d_b_theta { db[0] = bt; }
        if let Some(db) = d_b_eta   { db[0] = be; }
    }
}
```

Memory access pattern:
- `alpha[t]`, `theta[t]`, `eta[t]`: T scalar reads per thread, broadcast across warp (L1 cache hit)
- `k_mem[t*d + 0..d-1]`: coalesced across threads `0..d-1` at each `t`
- `v_mem[t*d + 0..d-1]`: coalesced across threads `d..2*d-1` at each `t`

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
