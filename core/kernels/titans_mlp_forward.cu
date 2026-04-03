// TitansLMM MLP Memory Forward CUDA Kernel (Spec 75, Phase B)
//
// Deep neural memory: M = {W₁, b₁, W₂, b₂} packed into a flat buffer.
// Forward: M(x) = W₂ @ σ(W₁ @ x + b₁) + b₂
// Inner-loop update: GD + EMA momentum, L2 bias (identity), L2 retention.
//
// Extends moneta_forward.cu with:
//   - Biases (b₁, b₂) in the MLP
//   - EMA momentum accumulator S (like titans_forward.cu)
//   - GELU activation (default; SiLU, ReLU also supported via template)
//   - Packed buffer layout (all params concatenated)
//   - Batch support via grid dimension (for multi-head stacked path)
//   - Per-token M-norm projection (spec 74)
//
// Grid=(batch_size), Block=(min(d_hidden, 1024)), power-of-2 rounded.
// All fp32. Weights in global memory, activations in shared memory.
//
// Shared memory: 2*d + 3*d_hidden floats.
//   k_buf[d] + pre_act[d_hidden] + hidden[d_hidden] + error_g[d]
//   + grad_pre[d_hidden]
// At d=64, d_hidden=256: ~4 KB. At d=512, d_hidden=2048: ~28 KB.
//
// Source: Titans (2501.00663) Eqs 12-15, MIRAS (2504.13173) §5.
//         Spec: specs/infrastructure/75_mlp_memory_module.md
//         Rust ref: core/src/titans_lmm.rs (step_mlp())

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "m_norm_project.cuh"

// ══════════════════════════════════════════════════════════════════════
// Activation device helpers
// ══════════════════════════════════════════════════════════════════════

// GELU (PyTorch tanh approximation)
__device__ __forceinline__ float gelu_dev(float x) {
    float c = 0.7978845608028654f; // sqrt(2/pi)
    float inner = c * (x + 0.044715f * x * x * x);
    float t = tanhf(inner);
    return 0.5f * x * (1.0f + t);
}

__device__ __forceinline__ float gelu_prime_dev(float x) {
    float c = 0.7978845608028654f;
    float a = 0.044715f;
    float inner = c * (x + a * x * x * x);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * a * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner;
}

// SiLU (re-declared; can't share across .cu files without LTO)
__device__ __forceinline__ float titans_mlp_silu_dev(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ __forceinline__ float titans_mlp_silu_prime_dev(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
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
// Forward kernel — templated over activation function
//
// ACT=0 (GELU):  GELU tanh approximation (default for TitansLMM MLP)
// ACT=1 (SiLU):  SiLU/Swish activation
// ACT=2 (ReLU):  ReLU activation
//
// The token loop processes sequentially (M evolves per-token).
// Gradient computation, momentum, and retention are fused — grad_W and
// grad_b are never materialized as full buffers. Computed element-wise
// and applied directly in the momentum+retention update.
// ══════════════════════════════════════════════════════════════════════

#define ACT_GELU 0
#define ACT_SILU 1
#define ACT_RELU 2

template <int ACT>
__device__ __forceinline__ float act_fn(float x) {
    if (ACT == ACT_GELU) return gelu_dev(x);
    if (ACT == ACT_SILU) return titans_mlp_silu_dev(x);
    /* ACT_RELU */ return fmaxf(0.0f, x);
}

template <int ACT>
__device__ __forceinline__ float act_prime_fn(float x) {
    if (ACT == ACT_GELU) return gelu_prime_dev(x);
    if (ACT == ACT_SILU) return titans_mlp_silu_prime_dev(x);
    /* ACT_RELU */ return (x > 0.0f) ? 1.0f : 0.0f;
}

template <int ACT>
__global__ void titans_mlp_forward_kernel(
    const float* __restrict__ k_mem,       // [batch_size, seq_len, d]
    const float* __restrict__ v_mem,       // [batch_size, seq_len, d]
    const float* __restrict__ q_mem,       // [batch_size, seq_len, d]
    const float* __restrict__ alpha,       // [batch_size, seq_len]
    const float* __restrict__ theta,       // [batch_size, seq_len]
    const float* __restrict__ eta,         // [batch_size, seq_len]
    const float* __restrict__ m_initial,   // [batch_size, state_size]
    const float* __restrict__ s_initial,   // [batch_size, state_size]
    float* __restrict__ m_states,          // [batch_size, (seq_len+1)*state_size]
    float* __restrict__ s_states,          // [batch_size, (seq_len+1)*state_size]
    float* __restrict__ y_out,             // [batch_size, seq_len, d]
    int seq_len, int d, int d_hidden,
    int input_stride,   // seq_len * d
    int m_stride,       // (seq_len + 1) * state_size
    float m_norm_max)
{
    int b = blockIdx.x;   // batch index
    int tid = threadIdx.x;

    // ── Packed buffer offsets (L_M=2) ────────────────────────────────
    // W1[d_hidden, d], b1[d_hidden], W2[d, d_hidden], b2[d]
    int w1_size = d_hidden * d;
    int b1_size = d_hidden;
    int w2_size = d * d_hidden;
    int b2_size = d;
    int w1_off = 0;
    int b1_off = w1_size;
    int w2_off = w1_size + b1_size;
    int b2_off = w2_off + w2_size;
    int state_size = w1_size + b1_size + w2_size + b2_size;

    // ── Batch-strided pointers ───────────────────────────────────────
    const float* k_b = k_mem + b * input_stride;
    const float* v_b = v_mem + b * input_stride;
    const float* q_b = q_mem + b * input_stride;
    const float* alpha_b = alpha + b * seq_len;
    const float* theta_b = theta + b * seq_len;
    const float* eta_b = eta + b * seq_len;
    float* m_b = m_states + b * m_stride;
    float* s_b = s_states + b * m_stride;
    float* y_b = y_out + b * input_stride;

    // ── Shared memory layout ─────────────────────────────────────────
    // k_buf[d] + pre_act[d_hidden] + hidden[d_hidden] + error_g[d]
    // + grad_pre[d_hidden]
    // Total: 2*d + 3*d_hidden floats
    extern __shared__ float smem[];
    float* k_buf    = smem;                          // [d]
    float* pre_act  = smem + d;                      // [d_hidden]
    float* hidden   = smem + d + d_hidden;           // [d_hidden]
    float* error_g  = smem + d + 2 * d_hidden;      // [d]
    float* grad_pre = smem + 2 * d + 2 * d_hidden;  // [d_hidden]

    // ── Initialize m_states[0] and s_states[0] from initial ──────────
    const float* m0 = m_initial + b * state_size;
    const float* s0 = s_initial + b * state_size;
    for (int idx = tid; idx < state_size; idx += blockDim.x) {
        m_b[idx] = m0[idx];
        s_b[idx] = s0[idx];
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // Token loop — sequential recurrence
    // ══════════════════════════════════════════════════════════════════
    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_b + t * d;
        const float* v_t = v_b + t * d;
        const float* q_t = q_b + t * d;
        float alpha_t = alpha_b[t];
        float theta_t = theta_b[t];
        float eta_t   = eta_b[t];

        float* M_t    = m_b + t * state_size;
        float* S_t    = s_b + t * state_size;
        float* M_next = m_b + (t + 1) * state_size;
        float* S_next = s_b + (t + 1) * state_size;

        // ── Load k_t into shared memory ──────────────────────────────
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = k_t[i];
        }
        __syncthreads();

        // ── 1. MLP forward: pre_act = W1 @ k_t + b1 ────────────────
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = M_t[b1_off + row]; // bias
            for (int j = 0; j < d; j++) {
                sum += M_t[w1_off + row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // ── 2. hidden = act(pre_act) ─────────────────────────────────
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = act_fn<ACT>(pre_act[i]);
        }
        __syncthreads();

        // ── 3. prediction = W2 @ hidden + b2 → error_g buffer ───────
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = M_t[b2_off + row]; // bias
            for (int j = 0; j < d_hidden; j++) {
                sum += M_t[w2_off + row * d_hidden + j] * hidden[j];
            }
            error_g[row] = sum;  // prediction stored in error_g
        }
        __syncthreads();

        // ── 4. error = prediction - v_t ──────────────────────────────
        // L2 attentional bias is the identity (no 2× factor — the learning
        // rate theta absorbs it). Matches apply_attentional_bias(L2) in
        // core/src/moneta.rs:1164.
        for (int i = tid; i < d; i += blockDim.x) {
            error_g[i] = error_g[i] - v_t[i];
        }
        __syncthreads();

        // ── 5. Backprop through W2 and activation ───────────────────
        // grad_h = W2^T @ error_g, then grad_pre = grad_h * act'(pre)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float gh = 0.0f;
            for (int j = 0; j < d; j++) {
                gh += M_t[w2_off + j * d_hidden + i] * error_g[j];
            }
            grad_pre[i] = gh * act_prime_fn<ACT>(pre_act[i]);
        }
        __syncthreads();
        // pre_act is now DEAD (act' consumed it)

        // ══════════════════════════════════════════════════════════════
        // 6. Fused gradient + EMA momentum + retention update
        //
        // For each param p in {W1, b1, W2, b2}:
        //   S_{t+1}[p] = eta * S_t[p] - theta * grad[p]
        //   M_{t+1}[p] = (1 - alpha) * M_t[p] + S_{t+1}[p]
        //
        // Gradients are NOT materialized as full buffers — computed
        // inline per-element using shared-memory vectors.
        // ══════════════════════════════════════════════════════════════
        float retention = 1.0f - alpha_t;

        // ── W1 update: grad = grad_pre[row] * k_buf[col] ────────────
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            int row = idx / d;
            int col = idx % d;
            float grad = grad_pre[row] * k_buf[col];
            float s_new = eta_t * S_t[w1_off + idx] - theta_t * grad;
            S_next[w1_off + idx] = s_new;
            M_next[w1_off + idx] = retention * M_t[w1_off + idx] + s_new;
        }

        // ── b1 update: grad = grad_pre[i] ───────────────────────────
        for (int i = tid; i < b1_size; i += blockDim.x) {
            float grad = grad_pre[i];
            float s_new = eta_t * S_t[b1_off + i] - theta_t * grad;
            S_next[b1_off + i] = s_new;
            M_next[b1_off + i] = retention * M_t[b1_off + i] + s_new;
        }

        // ── W2 update: grad = error_g[row] * hidden[col] ────────────
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            int row = idx / d_hidden;
            int col = idx % d_hidden;
            float grad = error_g[row] * hidden[col];
            float s_new = eta_t * S_t[w2_off + idx] - theta_t * grad;
            S_next[w2_off + idx] = s_new;
            M_next[w2_off + idx] = retention * M_t[w2_off + idx] + s_new;
        }

        // ── b2 update: grad = error_g[i] ────────────────────────────
        for (int i = tid; i < b2_size; i += blockDim.x) {
            float grad = error_g[i];
            float s_new = eta_t * S_t[b2_off + i] - theta_t * grad;
            S_next[b2_off + i] = s_new;
            M_next[b2_off + i] = retention * M_t[b2_off + i] + s_new;
        }
        __syncthreads();

        // ── 7. M-norm projection (Frobenius norm over entire state) ──
        // Reuses grad_pre (dead) as scratch for warp reduction.
        m_norm_project_inplace(M_next, grad_pre, state_size, tid, m_norm_max);

        // ══════════════════════════════════════════════════════════════
        // 8. Readout: y_t = M_{t+1}(q_t)
        // Reuses k_buf for q_t, pre_act for q_pre, hidden for q_hid.
        // ══════════════════════════════════════════════════════════════

        // Load q_t → k_buf (k_buf is dead after update)
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = q_t[i];
        }
        __syncthreads();

        // q_pre = W1_next @ q_t + b1_next
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = M_next[b1_off + row];
            for (int j = 0; j < d; j++) {
                sum += M_next[w1_off + row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // q_hid = act(q_pre)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = act_fn<ACT>(pre_act[i]);
        }
        __syncthreads();

        // y_t = W2_next @ q_hid + b2_next
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = M_next[b2_off + row];
            for (int j = 0; j < d_hidden; j++) {
                sum += M_next[w2_off + row * d_hidden + j] * hidden[j];
            }
            y_b[t * d + row] = sum;
        }
        __syncthreads();
    }
}

// ══════════════════════════════════════════════════════════════════════
// C wrapper — dispatches to template instantiation by activation
// ══════════════════════════════════════════════════════════════════════

extern "C" void titans_mlp_forward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_initial, const float* s_initial,
    float* m_states, float* s_states, float* y,
    int seq_len, int d, int d_hidden, int batch_size,
    int input_stride, int m_stride,
    int activation, float m_norm_max)
{
    if (d_hidden <= 0 || d <= 0) {
        fprintf(stderr, "titans_mlp_forward_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    // Block size: d_hidden threads (one per hidden unit), capped at 1024
    int block_size = (d_hidden < 1024) ? d_hidden : 1024;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded = 1024;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared memory: 2*d + 3*d_hidden floats
    int smem_bytes = (2 * d + 3 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "titans_mlp_forward_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    // Validate activation ID before launching
    if (activation != ACT_GELU && activation != ACT_SILU && activation != ACT_RELU) {
        fprintf(stderr, "titans_mlp_forward_f32_cuda: invalid activation=%d "
                "(expected 0=GELU, 1=SiLU, 2=ReLU).\n", activation);
        abort();
    }

    // Dispatch by activation template
    if (activation == ACT_GELU) {
        check_cuda_alloc("titans_mlp_forward: cudaFuncSetAttribute",
                         cudaFuncSetAttribute(titans_mlp_forward_kernel<ACT_GELU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        titans_mlp_forward_kernel<ACT_GELU><<<grid, block, smem_bytes>>>(
            k_mem, v_mem, q_mem, alpha, theta, eta,
            m_initial, s_initial, m_states, s_states, y,
            seq_len, d, d_hidden, input_stride, m_stride, m_norm_max);
        check_cuda_launch("titans_mlp_forward_kernel<GELU>", d_hidden, smem_bytes);
    } else if (activation == ACT_SILU) {
        check_cuda_alloc("titans_mlp_forward: cudaFuncSetAttribute",
                         cudaFuncSetAttribute(titans_mlp_forward_kernel<ACT_SILU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        titans_mlp_forward_kernel<ACT_SILU><<<grid, block, smem_bytes>>>(
            k_mem, v_mem, q_mem, alpha, theta, eta,
            m_initial, s_initial, m_states, s_states, y,
            seq_len, d, d_hidden, input_stride, m_stride, m_norm_max);
        check_cuda_launch("titans_mlp_forward_kernel<SiLU>", d_hidden, smem_bytes);
    } else {
        check_cuda_alloc("titans_mlp_forward: cudaFuncSetAttribute",
                         cudaFuncSetAttribute(titans_mlp_forward_kernel<ACT_RELU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        titans_mlp_forward_kernel<ACT_RELU><<<grid, block, smem_bytes>>>(
            k_mem, v_mem, q_mem, alpha, theta, eta,
            m_initial, s_initial, m_states, s_states, y,
            seq_len, d, d_hidden, input_stride, m_stride, m_norm_max);
        check_cuda_launch("titans_mlp_forward_kernel<ReLU>", d_hidden, smem_bytes);
    }
}
