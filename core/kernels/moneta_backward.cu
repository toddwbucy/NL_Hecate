// MLP Memory Backward CUDA Kernel — MONETA + YAAD (Templated)
//
// Reverse token loop with accumulated d_W1, d_W2 gradient accumulators.
// Recomputes forward intermediates from cached W1/W2 states.
// Templated over bias mode (l_p vs Huber) — same as forward kernel.
//
// Grid=(1), Block=(min(d, 1024)), power-of-2 rounded.
// Block size capped at d (not d_hidden) to keep register pressure
// manageable (~128 regs/thread at d=512).
//
// All fp32. d_W1, d_W2 gradient accumulators in global memory
// (workspace allocated by C wrapper — too large for shared memory).
//
// Shared memory: 4*d + 4*d_hidden floats.
//   k_buf[d] + pre_act[dh] + hidden[dh] + error_buf[d] + lp_g_buf[d]
//   + gp_dgh[dh] + dbuf_dh[dh] + dbuf_d[d]
// At d=512, d_hidden=2048: ~40 KB — fits in 48 KB default smem.
//
// NOTE: This implements the q=2 (standard L2) retention backward only.
// The L_q normalization backward (q > 2) adds lq_normalize_backward at
// each step and is deferred to a follow-up. Most experiments use q=2.
//
// Source: MIRAS (2504.13173) Eqs 24-26;
//         Spec: specs/infrastructure/cuda/07_mlp_memory_kernels.md §5
//         Rust ref: core/src/moneta.rs step_backward()

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Device helper functions (shared with moneta_forward.cu)
// ══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_bwd(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ __forceinline__ float silu_prime_bwd(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

// MONETA l_p gradient (forward path — recomputed during backward)
__device__ __forceinline__ float lp_grad_bwd(float e, float p, float a) {
    if (fabsf(p - 2.0f) < 1e-6f) {
        return 2.0f * e;
    } else if (fabsf(p - 1.0f) < 1e-6f) {
        return tanhf(a * e);
    } else {
        float smooth_sign = tanhf(a * e);
        float power_approx = powf(e * e + 1e-6f, (p - 1.0f) * 0.5f);
        return p * smooth_sign * power_approx;
    }
}

// Derivative of lp_grad w.r.t. error: d/de [p * Sign(e) * |e|^{p-1}]
// Source: core/src/moneta.rs lp_grad_deriv()
__device__ __forceinline__ float lp_grad_deriv_dev(float e, float p, float a) {
    if (fabsf(p - 2.0f) < 1e-6f) {
        return 2.0f;
    } else if (fabsf(p - 1.0f) < 1e-6f) {
        float tanh_ae = tanhf(a * e);
        return a * (1.0f - tanh_ae * tanh_ae);
    } else {
        float tanh_ae = tanhf(a * e);
        float sech2 = 1.0f - tanh_ae * tanh_ae;
        float e2_eps = e * e + 1e-6f;
        float power = powf(e2_eps, (p - 1.0f) * 0.5f);
        float d_power = (p - 1.0f) * e * powf(e2_eps, (p - 3.0f) * 0.5f);
        return p * (a * sech2 * power + tanh_ae * d_power);
    }
}

// YAAD Huber gradient
__device__ __forceinline__ float huber_grad_bwd(float e, float delta) {
    return (fabsf(e) < delta) ? e : ((e > 0.0f) ? delta : -delta);
}

// YAAD Huber gradient derivative: 1 if |e| < delta, 0 otherwise
__device__ __forceinline__ float huber_grad_deriv_dev(float e, float delta) {
    return (fabsf(e) < delta) ? 1.0f : 0.0f;
}

static inline void check_cuda_launch(const char* kernel_name, int d, int smem_bytes) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s launch failed (d=%d, smem=%d): %s\n",
                kernel_name, d, smem_bytes, cudaGetErrorString(err));
        abort();
    }
}

static inline void check_cuda_alloc(const char* tag, cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s: %s\n", tag, cudaGetErrorString(err));
        abort();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Backward kernel — templated over bias mode
// ══════════════════════════════════════════════════════════════════════

#define MLP_LP    0
#define MLP_HUBER 1

template <int MODE>
__global__ void mlp_backward_kernel(
    const float* __restrict__ k_mem,       // [seq_len, d]
    const float* __restrict__ v_mem,       // [seq_len, d]
    const float* __restrict__ q_mem,       // [seq_len, d]
    const float* __restrict__ alpha,       // [seq_len]
    const float* __restrict__ theta,       // [seq_len]
    const float* __restrict__ w1_states,   // [(seq_len+1) * d_hidden * d]
    const float* __restrict__ w2_states,   // [(seq_len+1) * d * d_hidden]
    const float* __restrict__ d_y,         // [seq_len, d]
    float* __restrict__ d_k_mem,           // [seq_len, d]
    float* __restrict__ d_v_mem,           // [seq_len, d]
    float* __restrict__ d_q_mem,           // [seq_len, d]
    float* __restrict__ d_alpha_out,       // [seq_len]
    float* __restrict__ d_theta_out,       // [seq_len]
    float* __restrict__ d_w1_initial,      // [d_hidden * d]
    float* __restrict__ d_w2_initial,      // [d * d_hidden]
    // Gradient accumulators (workspace, allocated by C wrapper)
    float* __restrict__ d_W1,              // [d_hidden * d]
    float* __restrict__ d_W2,              // [d * d_hidden]
    // Boundary snapshots (YAAD only, null for MONETA)
    const float* __restrict__ w1_boundary,
    const float* __restrict__ w2_boundary,
    // Dimensions
    int seq_len, int d, int d_hidden,
    // MONETA params
    float lp_p, float sign_sharpness, float lambda_2,
    // YAAD params
    float huber_delta, float lambda_local)
{
    int tid = threadIdx.x;
    int w1_size = d_hidden * d;
    int w2_size = d * d_hidden;

    // ── Shared memory layout ──────────────────────────────────────────
    // 4*d + 4*d_hidden floats total
    extern __shared__ float smem[];
    float* k_buf      = smem;                              // [d]
    float* pre_act    = smem + d;                           // [dh]
    float* hidden     = smem + d + d_hidden;                // [dh]
    float* error_buf  = smem + d + 2 * d_hidden;            // [d]
    float* lp_g_buf   = smem + 2 * d + 2 * d_hidden;       // [d]
    float* gp_dgh     = smem + 3 * d + 2 * d_hidden;       // [dh] grad_pre → d_grad_pre → d_grad_h
    float* dbuf_dh    = smem + 3 * d + 3 * d_hidden;       // [dh] d_q_pre / reduce_buf / d_h_gw2→d_pre_act
    float* dbuf_d     = smem + 3 * d + 4 * d_hidden;       // [d]  d_lp_g → d_err

    // ── Initialize d_W1, d_W2 accumulators to zero ──────────────────
    for (int idx = tid; idx < w1_size; idx += blockDim.x) {
        d_W1[idx] = 0.0f;
    }
    for (int idx = tid; idx < w2_size; idx += blockDim.x) {
        d_W2[idx] = 0.0f;
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // Reverse token loop
    // ══════════════════════════════════════════════════════════════════
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];
        const float* d_y_t = d_y + t * d;

        const float* W1_t    = w1_states + t * w1_size;
        const float* W2_t    = w2_states + t * w2_size;
        const float* W1_next = w1_states + (t + 1) * w1_size;
        const float* W2_next = w2_states + (t + 1) * w2_size;

        // ══════════════════════════════════════════════════════════════
        // PHASE 1: Readout backward
        // y_t = W2_{t+1} @ silu(W1_{t+1} @ q_t)
        // ══════════════════════════════════════════════════════════════

        // Load q_t → k_buf (reusing for readout phase)
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = q_t[i];
        }
        __syncthreads();

        // Recompute: q_pre = W1_next @ q_t
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += W1_next[row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;  // q_pre
        }
        __syncthreads();

        // q_hid = silu(q_pre)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = silu_bwd(pre_act[i]);
        }
        __syncthreads();

        // d_W2 += outer(d_y_t, q_hid)
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            int i = idx / d_hidden;
            int j = idx % d_hidden;
            d_W2[idx] += d_y_t[i] * hidden[j];
        }
        __syncthreads();

        // d_q_hid = W2_next^T @ d_y_t, fused with silu' → d_q_pre
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += W2_next[j * d_hidden + i] * d_y_t[j];
            }
            dbuf_dh[i] = sum * silu_prime_bwd(pre_act[i]);  // d_q_pre
        }
        __syncthreads();

        // d_W1 += outer(d_q_pre, q_t)
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_W1[idx] += dbuf_dh[i] * k_buf[j];
        }
        __syncthreads();

        // d_q_mem[t] = W1_next^T @ d_q_pre
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += W1_next[i * d + col] * dbuf_dh[i];
            }
            d_q_mem[t * d + col] = sum;
        }
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // PHASE 2: Recompute forward intermediates from cached W states
        // ══════════════════════════════════════════════════════════════

        // Load k_t → k_buf
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = k_t[i];
        }
        __syncthreads();

        // pre_act = W1_t @ k_t
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += W1_t[row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // hidden = silu(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = silu_bwd(pre_act[i]);
        }
        __syncthreads();

        // prediction → error_buf, then error = prediction - v_t
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += W2_t[row * d_hidden + j] * hidden[j];
            }
            error_buf[row] = sum - v_t[row];  // error = prediction - v_t
        }
        __syncthreads();

        // lp_g = bias_grad(error) → lp_g_buf (error_buf kept intact)
        if (MODE == MLP_LP) {
            for (int i = tid; i < d; i += blockDim.x) {
                lp_g_buf[i] = lp_grad_bwd(error_buf[i], lp_p, sign_sharpness);
            }
        } else {
            for (int i = tid; i < d; i += blockDim.x) {
                lp_g_buf[i] = huber_grad_bwd(error_buf[i], huber_delta);
            }
        }
        __syncthreads();

        // Recompute fwd_grad_pre = (W2^T @ lp_g) * silu'(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float gh = 0.0f;
            for (int j = 0; j < d; j++) {
                gh += W2_t[j * d_hidden + i] * lp_g_buf[j];
            }
            gp_dgh[i] = gh * silu_prime_bwd(pre_act[i]);  // fwd_grad_pre
        }
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // PHASE 3: Gate gradients (Frobenius dot reductions)
        // d_alpha = frob_dot(d_W1, W1_t) + frob_dot(d_W2, W2_t)
        // d_theta = -frob_dot(d_W, grad_W + ret_grad)
        // ══════════════════════════════════════════════════════════════

        // Use dbuf_dh[0..blockDim.x] as reduce_buf (dbuf_dh is dead here)
        float* reduce_buf = dbuf_dh;

        // ── d_alpha ──
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                local_sum += d_W1[idx] * W1_t[idx];
            }
            for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                local_sum += d_W2[idx] * W2_t[idx];
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) d_alpha_out[t] = reduce_buf[0];
            __syncthreads();
        }

        // ── d_theta ──
        // d_theta = -sum(d_W1 * (grad_W1 + ret_W1)) - sum(d_W2 * (grad_W2 + ret_W2))
        // grad_W1[i,j] = gp_dgh[i] * k_buf[j]  (fwd_grad_pre * k)
        // grad_W2[i,j] = lp_g_buf[i] * hidden[j]
        {
            float local_sum = 0.0f;
            if (MODE == MLP_LP) {
                // MONETA: ret_W = lambda_2 * 2 * W_t
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    float grad_w1 = gp_dgh[i] * k_buf[j];
                    float ret_w1 = lambda_2 * 2.0f * W1_t[idx];
                    local_sum -= d_W1[idx] * (grad_w1 + ret_w1);
                }
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    int i = idx / d_hidden;
                    int j = idx % d_hidden;
                    float grad_w2 = lp_g_buf[i] * hidden[j];
                    float ret_w2 = lambda_2 * 2.0f * W2_t[idx];
                    local_sum -= d_W2[idx] * (grad_w2 + ret_w2);
                }
            } else {
                // YAAD: ret = lambda_local * 2 * (W - W_bnd) + lambda_2 * 2 * W
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    float grad_w1 = gp_dgh[i] * k_buf[j];
                    float ret_local = lambda_local * 2.0f
                                    * (W1_t[idx] - w1_boundary[idx]);
                    float ret_global = lambda_2 * 2.0f * W1_t[idx];
                    local_sum -= d_W1[idx] * (grad_w1 + ret_local + ret_global);
                }
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    int i = idx / d_hidden;
                    int j = idx % d_hidden;
                    float grad_w2 = lp_g_buf[i] * hidden[j];
                    float ret_local = lambda_local * 2.0f
                                    * (W2_t[idx] - w2_boundary[idx]);
                    float ret_global = lambda_2 * 2.0f * W2_t[idx];
                    local_sum -= d_W2[idx] * (grad_w2 + ret_local + ret_global);
                }
            }
            reduce_buf[tid] = local_sum;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) d_theta_out[t] = reduce_buf[0];
            __syncthreads();
        }

        // ══════════════════════════════════════════════════════════════
        // PHASE 4: MLP backward — second-order gradients through
        // the update rule. Uses -theta * d_W inline (no workspace).
        //
        // Computes d_lp_g, d_grad_h, d_err, d_pre_act, and input
        // gradient contributions to d_k_mem, d_v_mem.
        // ══════════════════════════════════════════════════════════════

        // Step A: d_lp_g[i] = -theta * sum_j d_W2[i,j] * hidden[j]
        // (grad_W2 = outer(lp_g, h) backward: d_lp_g from d_grad_W2)
        for (int i = tid; i < d; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += d_W2[i * d_hidden + j] * hidden[j];
            }
            dbuf_d[i] = -theta_t * sum;  // d_lp_g
        }
        __syncthreads();

        // Step B: d_h_gw2[j] = -theta * sum_i d_W2[i,j] * lp_g[i]
        // (grad_W2 backward: d_hidden contribution from d_grad_W2)
        for (int j = tid; j < d_hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += d_W2[i * d_hidden + j] * lp_g_buf[i];
            }
            dbuf_dh[j] = -theta_t * sum;  // d_h_from_gw2
        }
        __syncthreads();

        // Step C: d_k contribution from grad_W1 backward
        // d_k[j] += -theta * sum_i d_W1[i,j] * fwd_grad_pre[i]
        // (Must be computed BEFORE step D overwrites gp_dgh/fwd_grad_pre)
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += d_W1[i * d + col] * gp_dgh[i];
            }
            d_k_mem[t * d + col] += -theta_t * sum;
        }
        __syncthreads();

        // Step D: d_grad_pre[i] = -theta * sum_j d_W1[i,j] * k_buf[j]
        // Overwrites gp_dgh (fwd_grad_pre is dead after step C)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += d_W1[i * d + j] * k_buf[j];
            }
            gp_dgh[i] = -theta_t * sum;  // d_grad_pre (overwrites fwd_grad_pre)
        }
        __syncthreads();

        // Step E: d_grad_h = d_grad_pre * silu'(pre_act)
        // Overwrites gp_dgh in-place (d_grad_pre consumed)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            gp_dgh[i] = gp_dgh[i] * silu_prime_bwd(pre_act[i]);
            // gp_dgh now holds d_grad_h
        }
        __syncthreads();

        // Step F: d_lp_g += W2_t @ d_grad_h (contribution from grad_h path)
        // grad_h = W2^T @ lp_g → backward: d_lp_g += W2 @ d_grad_h
        for (int j = tid; j < d; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += W2_t[j * d_hidden + i] * gp_dgh[i];
            }
            dbuf_d[j] += sum;  // d_lp_g updated
        }
        __syncthreads();

        // Step G: d_err = d_lp_g * bias_grad_deriv(error)
        if (MODE == MLP_LP) {
            for (int i = tid; i < d; i += blockDim.x) {
                dbuf_d[i] = dbuf_d[i] * lp_grad_deriv_dev(error_buf[i], lp_p, sign_sharpness);
            }
        } else {
            for (int i = tid; i < d; i += blockDim.x) {
                dbuf_d[i] = dbuf_d[i] * huber_grad_deriv_dev(error_buf[i], huber_delta);
            }
        }
        __syncthreads();
        // dbuf_d now holds d_err

        // Step H: d_v -= d_err
        for (int i = tid; i < d; i += blockDim.x) {
            d_v_mem[t * d + i] -= dbuf_d[i];
        }
        __syncthreads();

        // Step I: d_h_total = d_h_gw2 + W2_t^T @ d_err
        for (int j = tid; j < d_hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += W2_t[i * d_hidden + j] * dbuf_d[i];
            }
            dbuf_dh[j] += sum;  // d_h_total = d_h_gw2 + W2^T @ d_err
        }
        __syncthreads();

        // Step J: d_pre_act = d_h_total * silu'(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            dbuf_dh[i] = dbuf_dh[i] * silu_prime_bwd(pre_act[i]);
            // dbuf_dh now holds d_pre_act
        }
        __syncthreads();

        // Step K: d_k += W1_t^T @ d_pre_act
        for (int col = tid; col < d; col += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += W1_t[i * d + col] * dbuf_dh[i];
            }
            d_k_mem[t * d + col] += sum;
        }
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // PHASE 5: Propagation — scale d_W by coeff and add
        // MLP backward contributions (outer products).
        //
        // d_W1 = coeff * d_W1 + outer(d_pre_act, k)
        // d_W2 = coeff * d_W2 + outer(d_err, h) + outer(lp_g, d_grad_h)
        // ══════════════════════════════════════════════════════════════

        float coeff;
        if (MODE == MLP_LP) {
            coeff = alpha_t - theta_t * lambda_2 * 2.0f;
        } else {
            coeff = alpha_t - theta_t * (lambda_local + lambda_2) * 2.0f;
        }

        // d_W1 propagation
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            int i = idx / d;
            int j = idx % d;
            d_W1[idx] = coeff * d_W1[idx]
                       + dbuf_dh[i] * k_buf[j];  // outer(d_pre_act, k)
        }

        // d_W2 propagation
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            int i = idx / d_hidden;
            int j = idx % d_hidden;
            d_W2[idx] = coeff * d_W2[idx]
                       + dbuf_d[i] * hidden[j]    // outer(d_err, h)
                       + lp_g_buf[i] * gp_dgh[j]; // outer(lp_g, d_grad_h)
        }
        __syncthreads();
    }

    // ══════════════════════════════════════════════════════════════════
    // After processing all tokens: d_W1/d_W2 hold dL/dW_0 = dL/d_initial
    // Copy to output buffers.
    // ══════════════════════════════════════════════════════════════════
    for (int idx = tid; idx < w1_size; idx += blockDim.x) {
        d_w1_initial[idx] = d_W1[idx];
    }
    for (int idx = tid; idx < w2_size; idx += blockDim.x) {
        d_w2_initial[idx] = d_W2[idx];
    }
}

// ══════════════════════════════════════════════════════════════════════
// C wrappers
// ══════════════════════════════════════════════════════════════════════

extern "C" void mlp_backward_lp_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* w1_states, const float* w2_states,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta,
    float* d_w1_initial, float* d_w2_initial,
    int seq_len, int d, int d_hidden,
    float lp_p, float sign_sharpness, float lambda_2, float lq_q)
{
    if (d <= 0 || d_hidden <= 0) {
        fprintf(stderr, "mlp_backward_lp_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    // LQ backward (q > 2) is not implemented — reject at the CUDA level.
    if (fabsf(lq_q - 2.0f) > 1e-6f) {
        fprintf(stderr, "mlp_backward_lp_f32_cuda: lq_q=%.4f but only q=2.0 (L2 retention) "
                "is supported. LQ backward (q > 2) is deferred.\n", lq_q);
        exit(1);
    }

    // Block size capped at d for register pressure (matching Titans backward).
    // Round DOWN to largest power-of-2 <= d to ensure reduce_buf (aliased to
    // dbuf_dh[0..blockDim.x]) never exceeds d_hidden elements.
    int block_size = (d < 1024) ? d : 1024;
    int rounded = 1;
    while ((rounded << 1) <= block_size) rounded <<= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: 4*d + 4*d_hidden floats
    int smem_bytes = (4 * d + 4 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "mlp_backward_lp_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    check_cuda_alloc("mlp_backward_lp: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(mlp_backward_kernel<MLP_LP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // Allocate d_W1, d_W2 workspace
    int w1_size = d_hidden * d;
    int w2_size = d * d_hidden;
    float* d_W1_work = nullptr;
    float* d_W2_work = nullptr;
    check_cuda_alloc("mlp_backward_lp: cudaMalloc d_W1_work",
                     cudaMalloc(&d_W1_work, (size_t)w1_size * sizeof(float)));
    check_cuda_alloc("mlp_backward_lp: cudaMalloc d_W2_work",
                     cudaMalloc(&d_W2_work, (size_t)w2_size * sizeof(float)));

    mlp_backward_kernel<MLP_LP><<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta,
        w1_states, w2_states, d_y,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta,
        d_w1_initial, d_w2_initial,
        d_W1_work, d_W2_work,
        nullptr, nullptr,         // no boundary (MONETA)
        seq_len, d, d_hidden,
        lp_p, sign_sharpness, lambda_2,
        0.0f, 0.0f);             // no Huber/lambda_local
    check_cuda_launch("mlp_backward_kernel<MLP_LP>", d, smem_bytes);

    check_cuda_alloc("mlp_backward_lp: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());

    cudaFree(d_W1_work);
    cudaFree(d_W2_work);
}

extern "C" void mlp_backward_huber_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* w1_states, const float* w2_states,
    const float* w1_boundary, const float* w2_boundary,
    const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta,
    float* d_w1_initial, float* d_w2_initial,
    int seq_len, int d, int d_hidden,
    float huber_delta, float lambda_local, float lambda_2)
{
    if (d <= 0 || d_hidden <= 0) {
        fprintf(stderr, "mlp_backward_huber_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    // Round DOWN to largest power-of-2 <= d (same as lp variant).
    int block_size = (d < 1024) ? d : 1024;
    int rounded = 1;
    while ((rounded << 1) <= block_size) rounded <<= 1;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = (4 * d + 4 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "mlp_backward_huber_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    check_cuda_alloc("mlp_backward_huber: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(mlp_backward_kernel<MLP_HUBER>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    int w1_size = d_hidden * d;
    int w2_size = d * d_hidden;
    float* d_W1_work = nullptr;
    float* d_W2_work = nullptr;
    check_cuda_alloc("mlp_backward_huber: cudaMalloc d_W1_work",
                     cudaMalloc(&d_W1_work, (size_t)w1_size * sizeof(float)));
    check_cuda_alloc("mlp_backward_huber: cudaMalloc d_W2_work",
                     cudaMalloc(&d_W2_work, (size_t)w2_size * sizeof(float)));

    mlp_backward_kernel<MLP_HUBER><<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta,
        w1_states, w2_states, d_y,
        d_k_mem, d_v_mem, d_q_mem,
        d_alpha, d_theta,
        d_w1_initial, d_w2_initial,
        d_W1_work, d_W2_work,
        w1_boundary, w2_boundary,
        seq_len, d, d_hidden,
        0.0f, 0.0f, lambda_2,    // no l_p params
        huber_delta, lambda_local);
    check_cuda_launch("mlp_backward_kernel<MLP_HUBER>", d, smem_bytes);

    check_cuda_alloc("mlp_backward_huber: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());

    cudaFree(d_W1_work);
    cudaFree(d_W2_work);
}
