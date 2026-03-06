// Gate Backward CUDA Kernel
//
// Accumulates weight and bias gradients for alpha/theta/eta gate projections.
//
// Forward (gate_compute_cuda in elementwise.cu):
//   alpha_t = sigmoid(W_alpha @ concat(k_t, v_t) + b_alpha)
//   theta_t = softplus(W_theta @ concat(k_t, v_t) + b_theta)
//   eta_t   = sigmoid(W_eta   @ concat(k_t, v_t) + b_eta)   [Titans only]
//
// Backward:
//   d_w_alpha[i] = sum_t( d_alpha[t] * alpha_t*(1-alpha_t) * concat(k_t,v_t)[i] )
//   d_b_alpha    = sum_t( d_alpha[t] * alpha_t*(1-alpha_t) )
//
//   d_w_theta[i] = sum_t( d_theta[t] * (1-exp(-theta_t)) * concat(k_t,v_t)[i] )
//   d_b_theta    = sum_t( d_theta[t] * (1-exp(-theta_t)) )
//   Note: softplus'(x) = sigmoid(x) = 1 - exp(-softplus(x)) — no logit cache needed.
//
//   d_w_eta[i] = sum_t( d_eta[t] * eta_t*(1-eta_t) * concat(k_t,v_t)[i] ) [has_eta]
//   d_b_eta    = sum_t( d_eta[t] * eta_t*(1-eta_t) )
//
// Grid: (1), Block: (min(2*D, 1024)).
// Strided loop supports D > 512 (2*D > 1024).

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void gate_backward_kernel(
    // Per-token upstream gradients (scalars, one per token t in [0, T))
    const float* __restrict__ d_alpha,   // [T]
    const float* __restrict__ alpha,     // [T] sigmoid outputs
    const float* __restrict__ d_theta,   // [T] (ignored if !has_theta)
    const float* __restrict__ theta,     // [T] softplus outputs (ignored if !has_theta)
    const float* __restrict__ d_eta,     // [T] (ignored if !has_eta)
    const float* __restrict__ eta,       // [T] sigmoid outputs (ignored if !has_eta)
    // Gate input projections
    const float* __restrict__ k_mem,     // [T, d]
    const float* __restrict__ v_mem,     // [T, d]
    // Output: weight and bias gradients
    float* __restrict__ d_w_alpha,       // [2*d]
    float* __restrict__ d_b_alpha,       // [1]
    float* __restrict__ d_w_theta,       // [2*d] (written only if has_theta)
    float* __restrict__ d_b_theta,       // [1]   (written only if has_theta)
    float* __restrict__ d_w_eta,         // [2*d] (written only if has_eta)
    float* __restrict__ d_b_eta,         // [1]   (written only if has_eta)
    int T, int D, int has_theta, int has_eta)
{
    int tid = (int)threadIdx.x;
    int twoD = 2 * D;

    // Strided loop over weight indices (supports D > 512)
    for (int i = tid; i < twoD; i += blockDim.x) {
        // Per-weight accumulators for d_w
        float wa = 0.0f, wt = 0.0f, we = 0.0f;

        for (int t = 0; t < T; t++) {
            // Concatenated gate input at dimension i
            float x_i = (i < D) ? k_mem[t * D + i] : v_mem[t * D + (i - D)];

            // ── Alpha (sigmoid): sigma'(logit) = alpha * (1 - alpha) ──────
            float a = alpha[t];
            float da = d_alpha[t] * a * (1.0f - a);
            wa += da * x_i;

            // ── Theta (softplus): softplus'(logit) = sigmoid(logit) ───────
            if (has_theta) {
                float dt = d_theta[t] * (1.0f - expf(-theta[t]));
                wt += dt * x_i;
            }

            // ── Eta (sigmoid, Titans only): same as alpha ─────────────────
            if (has_eta) {
                float e = eta[t];
                float de = d_eta[t] * e * (1.0f - e);
                we += de * x_i;
            }
        }

        // Write weight gradients (each iteration writes its own index)
        d_w_alpha[i] = wa;
        if (has_theta) d_w_theta[i] = wt;
        if (has_eta)   d_w_eta[i]   = we;
    }

    // Bias gradients — only thread 0 computes and writes.
    // Bias is a scalar sum over tokens (independent of weight index i).
    if (tid == 0) {
        float ba = 0.0f, bt = 0.0f, be = 0.0f;
        for (int t = 0; t < T; t++) {
            float a = alpha[t];
            ba += d_alpha[t] * a * (1.0f - a);

            if (has_theta) {
                bt += d_theta[t] * (1.0f - expf(-theta[t]));
            }
            if (has_eta) {
                float e = eta[t];
                be += d_eta[t] * e * (1.0f - e);
            }
        }
        d_b_alpha[0] = ba;
        if (has_theta) d_b_theta[0] = bt;
        if (has_eta)   d_b_eta[0]   = be;
    }
}

extern "C" void gate_backward_cuda(
    const float* d_alpha, const float* alpha,
    const float* d_theta, const float* theta,
    const float* d_eta,   const float* eta,
    const float* k_mem, const float* v_mem,
    float* d_w_alpha, float* d_b_alpha,
    float* d_w_theta, float* d_b_theta,
    float* d_w_eta,   float* d_b_eta,
    int T, int D, int has_theta, int has_eta)
{
    if (D <= 0) {
        fprintf(stderr, "gate_backward_cuda: D=%d must be > 0.\n", D);
        exit(1);
    }
    int block = 2 * D;
    if (block > 1024) block = 1024;

    gate_backward_kernel<<<1, block>>>(
        d_alpha, alpha,
        d_theta, theta,
        d_eta,   eta,
        k_mem, v_mem,
        d_w_alpha, d_b_alpha,
        d_w_theta, d_b_theta,
        d_w_eta,   d_b_eta,
        T, D, has_theta, has_eta
    );
}
