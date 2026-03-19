// MLP Memory Forward CUDA Kernel — MONETA + YAAD (Templated)
//
// Two-layer MLP memory (W1[d_hidden, d], W2[d, d_hidden]) with templated
// bias function and retention rule. Covers MIRAS variants:
//   MLP_LP:    MONETA — l_p attentional bias + L_q/L2 retention
//   MLP_HUBER: YAAD — Huber attentional bias + decoupled L2 retention
//
// Grid=(1), Block=(min(d_hidden, 1024)), power-of-2 rounded.
// All fp32. W1, W2 in global memory (too large for shared memory at any d).
//
// Shared memory: 2*d + 3*d_hidden floats.
//   k_buf[d] + pre_act[d_hidden] + hidden[d_hidden] + error_lp_g[d]
//   + grad_pre[d_hidden]
// At d=512, d_hidden=2048: ~28 KB — fits in 48 KB default smem.
//
// No Ampere cp.async: bottleneck is W1/W2 global reads (4 MB each),
// not small vector loads. Prefetching k/v/q (2 KB) would not help.
//
// Source: MIRAS (2504.13173) Eqs 24-26, Table 2.
//         Spec: specs/infrastructure/cuda/07_mlp_memory_kernels.md
//         Rust ref: core/src/moneta.rs (step()), core/src/yaad.rs (step())

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Device helper functions
// ══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float silu_dev(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ __forceinline__ float silu_prime_dev(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

// MONETA l_p gradient: p * smooth_sign(e) * |e|^(p-1)
// Fast paths for p=1 (L1) and p=2 (L2).
// Source: MIRAS §5.1 Remark 5; core/src/moneta.rs lp_grad()
__device__ __forceinline__ float lp_grad_dev(float e, float p, float a) {
    if (fabsf(p - 2.0f) < 1e-6f) {
        // L2: derivative of e^2 = 2*e
        return 2.0f * e;
    } else if (fabsf(p - 1.0f) < 1e-6f) {
        // L1: smooth sign only (|e|^0 = 1 vanishes)
        return tanhf(a * e);
    } else {
        // General l_p: p * tanh(a*e) * (e^2 + eps)^{(p-1)/2}
        float smooth_sign = tanhf(a * e);
        float power_approx = powf(e * e + 1e-6f, (p - 1.0f) * 0.5f);
        return p * smooth_sign * power_approx;
    }
}

// YAAD Huber gradient: e if |e| < delta, delta * sign(e) otherwise.
// Source: MIRAS Eq 26; core/src/yaad.rs huber_grad()
__device__ __forceinline__ float huber_grad_dev(float e, float delta) {
    if (fabsf(e) < delta) {
        return e;
    } else {
        return (e > 0.0f) ? delta : -delta;
    }
}

// ══════════════════════════════════════════════════════════════════════
// Error checking helpers (match existing kernel pattern)
// ══════════════════════════════════════════════════════════════════════

static inline void check_cuda_launch(const char* kernel_name, int d_hidden, int smem_bytes) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s launch failed (d_hidden=%d, smem=%d): %s\n",
                kernel_name, d_hidden, smem_bytes, cudaGetErrorString(err));
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
// Forward kernel — templated over bias mode
//
// MODE=0 (MLP_LP):    MONETA — l_p bias, L_q/L2 retention
// MODE=1 (MLP_HUBER): YAAD — Huber bias, decoupled L2 retention
//
// The token loop processes sequentially (like Titans forward).
// Gradient computation and retention update are fused — grad_W1 and
// grad_W2 are never materialized as full matrices, computed element-wise
// and applied directly in the update step.
// ══════════════════════════════════════════════════════════════════════

#define MLP_LP    0
#define MLP_HUBER 1

template <int MODE>
__global__ void mlp_forward_kernel(
    const float* __restrict__ k_mem,       // [seq_len, d]
    const float* __restrict__ v_mem,       // [seq_len, d]
    const float* __restrict__ q_mem,       // [seq_len, d]
    const float* __restrict__ alpha,       // [seq_len]
    const float* __restrict__ theta,       // [seq_len]
    const float* __restrict__ w1_initial,  // [d_hidden * d]
    const float* __restrict__ w2_initial,  // [d * d_hidden]
    float* __restrict__ w1_states,         // [(seq_len+1) * d_hidden * d]
    float* __restrict__ w2_states,         // [(seq_len+1) * d * d_hidden]
    float* __restrict__ y_out,             // [seq_len, d]
    // LQ workspace (MONETA only when q > 2, null otherwise)
    float* a1_work,                        // [d_hidden * d]
    float* a2_work,                        // [d * d_hidden]
    // Boundary snapshots (YAAD only, null for MONETA)
    const float* __restrict__ w1_boundary, // [d_hidden * d]
    const float* __restrict__ w2_boundary, // [d * d_hidden]
    // Dimensions
    int seq_len, int d, int d_hidden,
    // MONETA params (ignored by YAAD path)
    float lp_p, float sign_sharpness, float lambda_2, float lq_q,
    // YAAD params (ignored by MONETA path)
    float huber_delta, float lambda_local)
{
    int tid = threadIdx.x;
    int w1_size = d_hidden * d;
    int w2_size = d * d_hidden;
    bool use_lq = (MODE == MLP_LP) && (fabsf(lq_q - 2.0f) >= 1e-6f);

    // ── Shared memory layout ──────────────────────────────────────────
    // k_buf[d] + pre_act[d_hidden] + hidden[d_hidden] + error_lp_g[d]
    // + grad_pre[d_hidden]
    // Total: 2*d + 3*d_hidden floats
    //
    // pre_act is also reused as reduce_buf[blockDim.x] during LQ
    // normalization (pre_act is dead at that point, and d_hidden >= blockDim.x).
    extern __shared__ float smem[];
    float* k_buf      = smem;                          // [d]
    float* pre_act    = smem + d;                      // [d_hidden]
    float* hidden     = smem + d + d_hidden;           // [d_hidden]
    float* error_lp_g = smem + d + 2 * d_hidden;      // [d]
    float* grad_pre   = smem + 2 * d + 2 * d_hidden;  // [d_hidden]

    // ── Initialize w1_states[0] and w2_states[0] from initial ──────
    for (int idx = tid; idx < w1_size; idx += blockDim.x) {
        w1_states[idx] = w1_initial[idx];
    }
    for (int idx = tid; idx < w2_size; idx += blockDim.x) {
        w2_states[idx] = w2_initial[idx];
    }
    // Initialize LQ accumulators: A_0 = W_0
    if (use_lq) {
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            a1_work[idx] = w1_initial[idx];
        }
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            a2_work[idx] = w2_initial[idx];
        }
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // Token loop — sequential recurrence
    // ══════════════════════════════════════════════════════════════════
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_mem + t * d;
        const float* v_t = v_mem + t * d;
        const float* q_t = q_mem + t * d;
        float alpha_t = alpha[t];
        float theta_t = theta[t];

        const float* W1_t = w1_states + t * w1_size;
        const float* W2_t = w2_states + t * w2_size;
        float* W1_next = w1_states + (t + 1) * w1_size;
        float* W2_next = w2_states + (t + 1) * w2_size;

        // ── Load k_t into shared memory ──────────────────────────────
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = k_t[i];
        }
        __syncthreads();

        // ── MLP forward: pre_act = W1 @ k_t ─────────────────────────
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += W1_t[row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // ── hidden = silu(pre_act) ───────────────────────────────────
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = silu_dev(pre_act[i]);
        }
        __syncthreads();

        // ── prediction = W2 @ hidden → error_lp_g buffer ────────────
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += W2_t[row * d_hidden + j] * hidden[j];
            }
            error_lp_g[row] = sum;  // prediction stored in error_lp_g
        }
        __syncthreads();

        // ── error = prediction - v_t (in-place) ─────────────────────
        for (int i = tid; i < d; i += blockDim.x) {
            error_lp_g[i] -= v_t[i];
        }
        __syncthreads();

        // ── Bias gradient (template-dispatched) ──────────────────────
        // Overwrites error with lp_g in-place (error is dead)
        if (MODE == MLP_LP) {
            for (int i = tid; i < d; i += blockDim.x) {
                error_lp_g[i] = lp_grad_dev(error_lp_g[i], lp_p, sign_sharpness);
            }
        } else {  // MLP_HUBER
            for (int i = tid; i < d; i += blockDim.x) {
                error_lp_g[i] = huber_grad_dev(error_lp_g[i], huber_delta);
            }
        }
        __syncthreads();

        // ── grad_h = W2^T @ lp_g, fused with silu'(pre_act) → grad_pre
        // grad_h is never materialized — computed inline and multiplied
        // by silu'(pre_act) to produce grad_pre directly.
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float gh = 0.0f;
            for (int j = 0; j < d; j++) {
                gh += W2_t[j * d_hidden + i] * error_lp_g[j];
            }
            grad_pre[i] = gh * silu_prime_dev(pre_act[i]);
        }
        __syncthreads();
        // pre_act is now DEAD (silu' already consumed it)

        // ══════════════════════════════════════════════════════════════
        // Fused retention update: compute grad_W element-wise and
        // apply the update rule in the same loop. Never materializes
        // the full grad_W1 [d_hidden, d] or grad_W2 [d, d_hidden].
        // ══════════════════════════════════════════════════════════════

        // ── W1 update ────────────────────────────────────────────────
        // grad_W1[i,j] = grad_pre[i] * k_buf[j]
        if (MODE == MLP_LP) {
            if (use_lq) {
                // LQ accumulator: A1_next = alpha*A1 - theta*(grad + l2*2*W1)
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    float grad_w1 = grad_pre[i] * k_buf[j];
                    float ret_grad = lambda_2 * 2.0f * W1_t[idx];
                    a1_work[idx] = alpha_t * a1_work[idx]
                                 - theta_t * (grad_w1 + ret_grad);
                }
            } else {
                // Standard L2: W1_next = alpha*W1 - theta*(grad + l2*2*W1)
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    int i = idx / d;
                    int j = idx % d;
                    float grad_w1 = grad_pre[i] * k_buf[j];
                    float ret_grad = lambda_2 * 2.0f * W1_t[idx];
                    W1_next[idx] = alpha_t * W1_t[idx]
                                 - theta_t * (grad_w1 + ret_grad);
                }
            }
        } else {  // MLP_HUBER (YAAD)
            // Decoupled L2: local boundary anchor + global L2
            for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                int i = idx / d;
                int j = idx % d;
                float grad_w1 = grad_pre[i] * k_buf[j];
                float ret_local = lambda_local * 2.0f
                                * (W1_t[idx] - w1_boundary[idx]);
                float ret_global = lambda_2 * 2.0f * W1_t[idx];
                W1_next[idx] = alpha_t * W1_t[idx]
                             - theta_t * (grad_w1 + ret_local + ret_global);
            }
        }
        __syncthreads();

        // ── W2 update ────────────────────────────────────────────────
        // grad_W2[i,j] = lp_g[i] * hidden[j]
        if (MODE == MLP_LP) {
            if (use_lq) {
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    int i = idx / d_hidden;
                    int j = idx % d_hidden;
                    float grad_w2 = error_lp_g[i] * hidden[j];
                    float ret_grad = lambda_2 * 2.0f * W2_t[idx];
                    a2_work[idx] = alpha_t * a2_work[idx]
                                 - theta_t * (grad_w2 + ret_grad);
                }
            } else {
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    int i = idx / d_hidden;
                    int j = idx % d_hidden;
                    float grad_w2 = error_lp_g[i] * hidden[j];
                    float ret_grad = lambda_2 * 2.0f * W2_t[idx];
                    W2_next[idx] = alpha_t * W2_t[idx]
                                 - theta_t * (grad_w2 + ret_grad);
                }
            }
        } else {  // MLP_HUBER (YAAD)
            for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                int i = idx / d_hidden;
                int j = idx % d_hidden;
                float grad_w2 = error_lp_g[i] * hidden[j];
                float ret_local = lambda_local * 2.0f
                                * (W2_t[idx] - w2_boundary[idx]);
                float ret_global = lambda_2 * 2.0f * W2_t[idx];
                W2_next[idx] = alpha_t * W2_t[idx]
                             - theta_t * (grad_w2 + ret_local + ret_global);
            }
        }
        __syncthreads();

        // ══════════════════════════════════════════════════════════════
        // LQ normalization (MONETA only, q > 2)
        // W = A / ||A||_q^{q-2}
        // Uses pre_act[0..blockDim.x] as reduce_buf (pre_act is dead).
        // ══════════════════════════════════════════════════════════════
        if (use_lq) {
            float* reduce_buf = pre_act;

            // ── Normalize W1 ────────────────────────────────────────
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    float a = fabsf(a1_work[idx]);
                    local_sum += powf(a, lq_q);
                }
                reduce_buf[tid] = local_sum;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                    __syncthreads();
                }

                if (tid == 0) {
                    float norm_q = powf(reduce_buf[0], 1.0f / lq_q);
                    float scale = (norm_q > 1e-12f)
                                ? powf(norm_q, -(lq_q - 2.0f)) : 1.0f;
                    reduce_buf[0] = scale;
                }
                __syncthreads();

                float scale = reduce_buf[0];
                for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                    W1_next[idx] = a1_work[idx] * scale;
                }
            }
            __syncthreads();

            // ── Normalize W2 ────────────────────────────────────────
            {
                float local_sum = 0.0f;
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    float a = fabsf(a2_work[idx]);
                    local_sum += powf(a, lq_q);
                }
                reduce_buf[tid] = local_sum;
                __syncthreads();

                for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
                    __syncthreads();
                }

                if (tid == 0) {
                    float norm_q = powf(reduce_buf[0], 1.0f / lq_q);
                    float scale = (norm_q > 1e-12f)
                                ? powf(norm_q, -(lq_q - 2.0f)) : 1.0f;
                    reduce_buf[0] = scale;
                }
                __syncthreads();

                float scale = reduce_buf[0];
                for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                    W2_next[idx] = a2_work[idx] * scale;
                }
            }
            __syncthreads();
        }

        // ══════════════════════════════════════════════════════════════
        // Readout: y_t = W2_next @ silu(W1_next @ q_t)
        // Reuses k_buf for q_t, pre_act for q_pre, hidden for q_hid.
        // ══════════════════════════════════════════════════════════════

        // Load q_t → k_buf (k_buf is dead after retention update)
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = q_t[i];
        }
        __syncthreads();

        // q_pre = W1_next @ q_t
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += W1_next[row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // q_hid = silu(q_pre)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = silu_dev(pre_act[i]);
        }
        __syncthreads();

        // y_t = W2_next @ q_hid → output buffer
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += W2_next[row * d_hidden + j] * hidden[j];
            }
            y_out[t * d + row] = sum;
        }
        __syncthreads();
    }
}

// ══════════════════════════════════════════════════════════════════════
// C wrappers — match spec signatures exactly
// ══════════════════════════════════════════════════════════════════════

extern "C" void mlp_forward_lp_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* w1_initial, const float* w2_initial,
    float* w1_states, float* w2_states, float* y,
    int seq_len, int d, int d_hidden,
    float lp_p, float sign_sharpness, float lambda_2, float lq_q)
{
    if (d_hidden <= 0 || d <= 0) {
        fprintf(stderr, "mlp_forward_lp_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    // Block size: d_hidden threads (one per hidden unit), capped at 1024
    int block_size = (d_hidden < 1024) ? d_hidden : 1024;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    // Shared memory: 2*d + 3*d_hidden floats
    int smem_bytes = (2 * d + 3 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "mlp_forward_lp_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    check_cuda_alloc("mlp_forward_lp: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(mlp_forward_kernel<MLP_LP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    // Allocate LQ accumulator workspace (only when q > 2)
    bool use_lq = fabsf(lq_q - 2.0f) >= 1e-6f;
    float* a1_work = nullptr;
    float* a2_work = nullptr;
    int w1_size = d_hidden * d;
    int w2_size = d * d_hidden;
    if (use_lq) {
        check_cuda_alloc("mlp_forward_lp: cudaMalloc a1_work",
                         cudaMalloc(&a1_work, (size_t)w1_size * sizeof(float)));
        check_cuda_alloc("mlp_forward_lp: cudaMalloc a2_work",
                         cudaMalloc(&a2_work, (size_t)w2_size * sizeof(float)));
    }

    mlp_forward_kernel<MLP_LP><<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta,
        w1_initial, w2_initial, w1_states, w2_states, y,
        a1_work, a2_work,         // LQ workspace
        nullptr, nullptr,         // no boundary snapshots (MONETA)
        seq_len, d, d_hidden,
        lp_p, sign_sharpness, lambda_2, lq_q,
        0.0f, 0.0f);             // no Huber/lambda_local (MONETA)
    check_cuda_launch("mlp_forward_kernel<MLP_LP>", d_hidden, smem_bytes);

    check_cuda_alloc("mlp_forward_lp: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());

    if (use_lq) {
        cudaFree(a1_work);
        cudaFree(a2_work);
    }
}

extern "C" void mlp_forward_huber_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta,
    const float* w1_initial, const float* w2_initial,
    const float* w1_boundary, const float* w2_boundary,
    float* w1_states, float* w2_states, float* y,
    int seq_len, int d, int d_hidden,
    float huber_delta, float lambda_local, float lambda_2)
{
    if (d_hidden <= 0 || d <= 0) {
        fprintf(stderr, "mlp_forward_huber_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    int block_size = (d_hidden < 1024) ? d_hidden : 1024;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(1);
    dim3 block(block_size);

    int smem_bytes = (2 * d + 3 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "mlp_forward_huber_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    check_cuda_alloc("mlp_forward_huber: cudaFuncSetAttribute",
                     cudaFuncSetAttribute(mlp_forward_kernel<MLP_HUBER>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mlp_forward_kernel<MLP_HUBER><<<grid, block, smem_bytes>>>(
        k_mem, v_mem, q_mem, alpha, theta,
        w1_initial, w2_initial, w1_states, w2_states, y,
        nullptr, nullptr,         // no LQ workspace (YAAD)
        w1_boundary, w2_boundary, // boundary snapshots for local retention
        seq_len, d, d_hidden,
        0.0f, 0.0f, lambda_2, 2.0f,  // no l_p params (YAAD)
        huber_delta, lambda_local);
    check_cuda_launch("mlp_forward_kernel<MLP_HUBER>", d_hidden, smem_bytes);

    check_cuda_alloc("mlp_forward_huber: cudaDeviceSynchronize",
                     cudaDeviceSynchronize());
}
