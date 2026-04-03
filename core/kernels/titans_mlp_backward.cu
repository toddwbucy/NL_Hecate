// TitansLMM MLP Memory Backward CUDA Kernel (Spec 75, Phase C)
//
// Analytical backward through deep neural memory with EMA momentum.
// Combines:
//   - Titans backward: d_M / d_S EMA recurrence, d_alpha/d_theta/d_eta gates
//   - MONETA backward: MLP chain rule (W1/b1/W2/b2 through activation)
//
// Forward recap (from titans_mlp_forward.cu):
//   M = {W1, b1, W2, b2} packed flat.
//   For each token t:
//     prediction = W2_t @ σ(W1_t @ k_t + b1_t) + b2_t
//     error = prediction - v_t  (L2 bias = identity)
//     grad = analytical MLP backward of error
//     S_{t+1} = η·S_t - θ·grad
//     M_{t+1} = (1-α)·M_t + S_{t+1}
//     y_t = MLP(M_{t+1}, q_t)
//
// Backward: reverse token loop computing d_k, d_v, d_q, d_alpha, d_theta,
// d_eta, d_m_initial, d_s_initial.
//
// Grid=(batch_size), Block=(power_of_2_round(min(d, 1024))).
// Shared memory: (3*d + 4*d_hidden) * sizeof(float).
//
// Source: Titans (2501.00663) Eqs 12-15; MIRAS (2504.13173) §5.
//         Rust ref: core/src/titans_lmm.rs (mlp_inner_backward)

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ══════════════════════════════════════════════════════════════════════
// Activation device helpers (re-declared; can't share across .cu)
// ══════════════════════════════════════════════════════════════════════

#define ACT_GELU 0
#define ACT_SILU 1
#define ACT_RELU 2

__device__ __forceinline__ float bw_gelu(float x) {
    float c = 0.7978845608028654f;
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float bw_gelu_prime(float x) {
    float c = 0.7978845608028654f;
    float a = 0.044715f;
    float inner = c * (x + a * x * x * x);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = c * (1.0f + 3.0f * a * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner;
}

__device__ __forceinline__ float bw_silu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ __forceinline__ float bw_silu_prime(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

template <int ACT>
__device__ __forceinline__ float bw_act(float x) {
    if (ACT == ACT_GELU) return bw_gelu(x);
    if (ACT == ACT_SILU) return bw_silu(x);
    return fmaxf(0.0f, x);
}

template <int ACT>
__device__ __forceinline__ float bw_act_prime(float x) {
    if (ACT == ACT_GELU) return bw_gelu_prime(x);
    if (ACT == ACT_SILU) return bw_silu_prime(x);
    return (x > 0.0f) ? 1.0f : 0.0f;
}

// ══════════════════════════════════════════════════════════════════════
// Error checking helpers
// ══════════════════════════════════════════════════════════════════════

static inline void check_launch(const char* name, int d, int smem) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s launch failed (d=%d, smem=%d): %s\n",
                name, d, smem, cudaGetErrorString(err));
        abort();
    }
}

static inline void check_alloc(const char* tag, cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[NL_Hecate FATAL] %s: %s\n", tag, cudaGetErrorString(err));
        abort();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Backward kernel — templated over activation function
// ══════════════════════════════════════════════════════════════════════

template <int ACT>
__launch_bounds__(1024, 1)
__global__ void titans_mlp_backward_kernel(
    const float* __restrict__ k_mem,      // [batch, seq_len, d]
    const float* __restrict__ v_mem,      // [batch, seq_len, d]
    const float* __restrict__ q_mem,      // [batch, seq_len, d]
    const float* __restrict__ alpha,      // [batch, seq_len]
    const float* __restrict__ theta,      // [batch, seq_len]
    const float* __restrict__ eta,        // [batch, seq_len]
    const float* __restrict__ m_states,   // [batch, (seq_len+1)*state_size]
    const float* __restrict__ s_states,   // [batch, (seq_len+1)*state_size]
    const float* __restrict__ d_y,        // [batch, seq_len, d]
    float* __restrict__ d_k_mem,          // [batch, seq_len, d]
    float* __restrict__ d_v_mem,          // [batch, seq_len, d]
    float* __restrict__ d_q_mem,          // [batch, seq_len, d]
    float* __restrict__ d_alpha,          // [batch, seq_len]
    float* __restrict__ d_theta,          // [batch, seq_len]
    float* __restrict__ d_eta,            // [batch, seq_len]
    float* __restrict__ d_m_initial,      // [state_size] — summed across batch (atomicAdd)
    float* __restrict__ d_s_initial,      // [state_size] — summed across batch (atomicAdd)
    float* __restrict__ d_M,              // [batch, state_size] — workspace
    float* __restrict__ d_S,              // [batch, state_size] — workspace
    int seq_len, int d, int d_hidden,
    int input_stride,   // seq_len * d
    int m_stride)       // (seq_len + 1) * state_size
{
    int b = blockIdx.x;
    int tid = threadIdx.x;

    // Packed buffer offsets
    int w1_size = d_hidden * d;
    int b1_size = d_hidden;
    int w2_size = d * d_hidden;
    int b2_size = d;
    int w1_off = 0;
    int b1_off = w1_size;
    int w2_off = w1_size + b1_size;
    int b2_off = w2_off + w2_size;
    int state_size = w1_size + b1_size + w2_size + b2_size;

    // Batch-strided pointers
    const float* k_b = k_mem + b * input_stride;
    const float* v_b = v_mem + b * input_stride;
    const float* q_b = q_mem + b * input_stride;
    const float* al_b = alpha + b * seq_len;
    const float* th_b = theta + b * seq_len;
    const float* et_b = eta + b * seq_len;
    const float* m_b = m_states + b * m_stride;
    const float* s_b = s_states + b * m_stride;
    const float* dy_b = d_y + b * input_stride;
    float* dk_b = d_k_mem + b * input_stride;
    float* dv_b = d_v_mem + b * input_stride;
    float* dq_b = d_q_mem + b * input_stride;
    float* da_b = d_alpha + b * seq_len;
    float* dt_b = d_theta + b * seq_len;
    float* de_b = d_eta + b * seq_len;
    float* dM = d_M + b * state_size;
    float* dS = d_S + b * state_size;

    // ── Shared memory layout ─────────────────────────────────────────
    // k_buf[d] + pre_act[dh] + hidden[dh] + error_buf[BS] + d_err[BS]
    // + grad_pre[dh] + d_h_buf[dh]
    // Total: d + 2*BS + 4*d_hidden   (BS = blockDim.x, may be > d after rounding)
    int BS = blockDim.x;
    extern __shared__ float smem[];
    float* k_buf     = smem;                                // [d]
    float* pre_act   = smem + d;                            // [dh]
    float* hidden    = smem + d + d_hidden;                 // [dh]
    float* error_buf = smem + d + 2 * d_hidden;             // [BS]
    float* d_err     = smem + d + 2 * d_hidden + BS;        // [BS]
    float* grad_pre  = smem + d + 2 * d_hidden + 2 * BS;   // [dh]
    float* d_h_buf   = smem + d + 2 * d_hidden + 2 * BS + d_hidden; // [dh]

    // Init d_M = 0, d_S = 0
    for (int idx = tid; idx < state_size; idx += blockDim.x) {
        dM[idx] = 0.0f;
        dS[idx] = 0.0f;
    }
    __syncthreads();

    // ══════════════════════════════════════════════════════════════════
    // Reverse token loop
    // ══════════════════════════════════════════════════════════════════
    for (int t = seq_len - 1; t >= 0; t--) {
        const float* k_t = k_b + t * d;
        const float* v_t = v_b + t * d;
        const float* q_t = q_b + t * d;
        const float* dy_t = dy_b + t * d;
        const float* M_t = m_b + t * state_size;
        const float* M_next = m_b + (t + 1) * state_size;
        const float* S_t = s_b + t * state_size;
        float alpha_t = al_b[t];
        float theta_t = th_b[t];
        float eta_t = et_b[t];
        float retention = 1.0f - alpha_t;

        // ──────────────────────────────────────────────────────────────
        // PHASE 1: Readout backward — y_t = MLP(M_{t+1}, q_t)
        // ──────────────────────────────────────────────────────────────

        // Load q_t into k_buf (reused for readout)
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = q_t[i];
        }
        __syncthreads();

        // Recompute readout: q_pre = W1_next @ q_t + b1_next
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = M_next[b1_off + row];
            for (int j = 0; j < d; j++) {
                sum += M_next[w1_off + row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // q_hid = σ(q_pre)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = bw_act<ACT>(pre_act[i]);
        }
        __syncthreads();

        // Accumulate d_M from readout: W2 and b2
        // d_M[W2][i,j] += d_y_t[i] * q_hid[j]
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            int row = idx / d_hidden;
            int col = idx % d_hidden;
            dM[w2_off + idx] += dy_t[row] * hidden[col];
        }
        // d_M[b2][i] += d_y_t[i]
        for (int i = tid; i < b2_size; i += blockDim.x) {
            dM[b2_off + i] += dy_t[i];
        }
        __syncthreads();

        // Backprop through σ for readout: d_q_hid = W2_next^T @ d_y_t
        for (int j = tid; j < d_hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += M_next[w2_off + i * d_hidden + j] * dy_t[i];
            }
            // d_q_pre = d_q_hid * σ'(q_pre)
            d_h_buf[j] = sum * bw_act_prime<ACT>(pre_act[j]);
        }
        __syncthreads();

        // d_M[W1][i,j] += d_q_pre[i] * q_t[j]
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            int row = idx / d;
            int col = idx % d;
            dM[w1_off + idx] += d_h_buf[row] * k_buf[col];
        }
        // d_M[b1][i] += d_q_pre[i]
        for (int i = tid; i < b1_size; i += blockDim.x) {
            dM[b1_off + i] += d_h_buf[i];
        }
        __syncthreads();

        // d_q_mem[t] = W1_next^T @ d_q_pre
        for (int j = tid; j < d; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += M_next[w1_off + i * d + j] * d_h_buf[i];
            }
            dq_b[t * d + j] = sum;
        }
        __syncthreads();

        // ──────────────────────────────────────────────────────────────
        // PHASE 2: Retention backward
        //   M_{t+1} = (1-α)·M_t + S_{t+1}
        //   d_S += d_M, d_alpha = -<M_t, d_M>, d_M *= (1-α)
        // ──────────────────────────────────────────────────────────────

        // d_S += d_M
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            dS[idx] += dM[idx];
        }
        __syncthreads();

        // d_alpha = -Σ(M_t[i] * d_M[i]) — Frobenius dot on packed state
        // Reuse error_buf[0..blockDim] as reduce_buf
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                local_sum += M_t[idx] * dM[idx];
            }
            error_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) error_buf[tid] += error_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) da_b[t] = -error_buf[0];
            __syncthreads();
        }

        // d_eta = Σ(S_t[i] * d_S[i]) — Frobenius dot on packed state
        {
            float local_sum = 0.0f;
            for (int idx = tid; idx < state_size; idx += blockDim.x) {
                local_sum += S_t[idx] * dS[idx];
            }
            error_buf[tid] = local_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) error_buf[tid] += error_buf[tid + s];
                __syncthreads();
            }
            if (tid == 0) de_b[t] = error_buf[0];
            __syncthreads();
        }

        // ──────────────────────────────────────────────────────────────
        // PHASE 3: Recompute forward intermediates from M_t
        // ──────────────────────────────────────────────────────────────

        // Load k_t into k_buf (overwrite q_t)
        for (int i = tid; i < d; i += blockDim.x) {
            k_buf[i] = k_t[i];
        }
        __syncthreads();

        // pre_act = W1_t @ k_t + b1_t
        for (int row = tid; row < d_hidden; row += blockDim.x) {
            float sum = M_t[b1_off + row];
            for (int j = 0; j < d; j++) {
                sum += M_t[w1_off + row * d + j] * k_buf[j];
            }
            pre_act[row] = sum;
        }
        __syncthreads();

        // hidden = σ(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            hidden[i] = bw_act<ACT>(pre_act[i]);
        }
        __syncthreads();

        // prediction → error_buf = W2_t @ hidden + b2_t - v_t
        for (int row = tid; row < d; row += blockDim.x) {
            float sum = M_t[b2_off + row];
            for (int j = 0; j < d_hidden; j++) {
                sum += M_t[w2_off + row * d_hidden + j] * hidden[j];
            }
            error_buf[row] = sum - v_t[row];
        }
        __syncthreads();

        // grad_h = W2_t^T @ error → grad_pre = grad_h * σ'(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float gh = 0.0f;
            for (int j = 0; j < d; j++) {
                gh += M_t[w2_off + j * d_hidden + i] * error_buf[j];
            }
            grad_pre[i] = gh * bw_act_prime<ACT>(pre_act[i]);
        }
        __syncthreads();

        // ──────────────────────────────────────────────────────────────
        // PHASE 4: d_theta = -<grad, d_S> where grad is the analytical
        // MLP gradient (packed).
        //
        // grad_W1[i,j] = grad_pre[i] * k[j]
        // grad_b1[i]   = grad_pre[i]
        // grad_W2[i,j] = error[i] * hidden[j]
        // grad_b2[i]   = error[i]
        // ──────────────────────────────────────────────────────────────
        {
            float local_sum = 0.0f;
            // W1 contribution: Σ grad_pre[row]*k[col] * d_S[w1_off+idx]
            for (int idx = tid; idx < w1_size; idx += blockDim.x) {
                int row = idx / d;
                int col = idx % d;
                local_sum += grad_pre[row] * k_buf[col] * dS[w1_off + idx];
            }
            // b1 contribution: Σ grad_pre[i] * d_S[b1_off+i]
            for (int i = tid; i < b1_size; i += blockDim.x) {
                local_sum += grad_pre[i] * dS[b1_off + i];
            }
            // W2 contribution: Σ error[row]*hidden[col] * d_S[w2_off+idx]
            for (int idx = tid; idx < w2_size; idx += blockDim.x) {
                int row = idx / d_hidden;
                int col = idx % d_hidden;
                local_sum += error_buf[row] * hidden[col] * dS[w2_off + idx];
            }
            // b2 contribution: Σ error[i] * d_S[b2_off+i]
            for (int i = tid; i < b2_size; i += blockDim.x) {
                local_sum += error_buf[i] * dS[b2_off + i];
            }
            // Reduce
            d_err[tid] = local_sum;  // reuse d_err as reduce_buf
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) d_err[tid] += d_err[tid + s];
                __syncthreads();
            }
            if (tid == 0) dt_b[t] = -d_err[0];
            __syncthreads();
        }

        // ──────────────────────────────────────────────────────────────
        // PHASE 5: MLP backward (second-order)
        //
        // d_grad = -θ·d_S. Backprop through analytical gradient to get
        // d_k, d_v, and propagation terms for d_M.
        //
        // Following MONETA backward Phases 4A-4K, adapted for biases.
        // ──────────────────────────────────────────────────────────────

        // Step A: d_error from W2 + b2 paths
        // d_err[i] = -θ·(Σ_j d_S[w2+i*dh+j]*hidden[j] + d_S[b2+i])
        for (int i = tid; i < d; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += dS[w2_off + i * d_hidden + j] * hidden[j];
            }
            sum += dS[b2_off + i];
            d_err[i] = -theta_t * sum;
        }
        __syncthreads();

        // Step B: d_hidden from W2 path
        // d_h_buf[j] = -θ·Σ_i d_S[w2+i*dh+j]*error[i]
        for (int j = tid; j < d_hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += dS[w2_off + i * d_hidden + j] * error_buf[i];
            }
            d_h_buf[j] = -theta_t * sum;
        }
        __syncthreads();

        // Step C: d_k from W1 path (using grad_pre which is still live)
        // d_k[j] = -θ·Σ_i d_S[w1+i*d+j]*grad_pre[i]
        for (int j = tid; j < d; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += dS[w1_off + i * d + j] * grad_pre[i];
            }
            dk_b[t * d + j] = -theta_t * sum;
        }
        __syncthreads();

        // Step D: d_grad_pre from W1 + b1 paths (overwrites grad_pre)
        // d_gp[i] = -θ·(Σ_j d_S[w1+i*d+j]*k[j] + d_S[b1+i])
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += dS[w1_off + i * d + j] * k_buf[j];
            }
            sum += dS[b1_off + i];
            grad_pre[i] = -theta_t * sum;  // now d_grad_pre
        }
        __syncthreads();

        // Step E: d_grad_h = d_grad_pre * σ'(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            grad_pre[i] = grad_pre[i] * bw_act_prime<ACT>(pre_act[i]);
        }
        __syncthreads();
        // grad_pre is now d_grad_h

        // Step F: d_err += W2_t @ d_grad_h
        for (int i = tid; i < d; i += blockDim.x) {
            float sum = 0.0f;
            for (int j = 0; j < d_hidden; j++) {
                sum += M_t[w2_off + i * d_hidden + j] * grad_pre[j];
            }
            d_err[i] += sum;
        }
        __syncthreads();

        // Step G: d_v = -d_err
        for (int i = tid; i < d; i += blockDim.x) {
            dv_b[t * d + i] = -d_err[i];
        }

        // Step H: d_h_total = d_h_buf + W2_t^T @ d_err
        for (int j = tid; j < d_hidden; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += M_t[w2_off + i * d_hidden + j] * d_err[i];
            }
            d_h_buf[j] += sum;
        }
        __syncthreads();

        // Step I: d_pre_act = d_h_total * σ'(pre_act)
        for (int i = tid; i < d_hidden; i += blockDim.x) {
            d_h_buf[i] = d_h_buf[i] * bw_act_prime<ACT>(pre_act[i]);
        }
        __syncthreads();
        // d_h_buf is now d_pre_act

        // Step J: d_k += W1_t^T @ d_pre_act
        for (int j = tid; j < d; j += blockDim.x) {
            float sum = 0.0f;
            for (int i = 0; i < d_hidden; i++) {
                sum += M_t[w1_off + i * d + j] * d_h_buf[i];
            }
            dk_b[t * d + j] += sum;
        }
        __syncthreads();

        // ──────────────────────────────────────────────────────────────
        // PHASE 6: Propagation — update d_M and d_S accumulators
        //
        // d_M *= (1-α)  (retention backward, deferred from Phase 2)
        // d_M += [outer(d_pre_act, k), d_pre_act,
        //         outer(d_err, hidden) + outer(error, d_grad_h), d_err]
        // d_S *= η      (EMA backward)
        //
        // Note: outer(error, d_grad_h) is the cross-term from W2
        // gradient: grad_W2 = outer(error, hidden), and d_grad_h comes
        // from backprop through grad_pre → σ' → W2^T @ error.
        // grad_pre is currently d_grad_h (Step E overwrote it).
        // ──────────────────────────────────────────────────────────────

        // W1: d_M = retention*d_M + outer(d_pre_act, k)
        for (int idx = tid; idx < w1_size; idx += blockDim.x) {
            int row = idx / d;
            int col = idx % d;
            dM[w1_off + idx] = retention * dM[w1_off + idx]
                             + d_h_buf[row] * k_buf[col];
        }

        // b1: d_M = retention*d_M + d_pre_act
        for (int i = tid; i < b1_size; i += blockDim.x) {
            dM[b1_off + i] = retention * dM[b1_off + i] + d_h_buf[i];
        }

        // W2: d_M = retention*d_M + outer(d_err, hidden) + outer(error, d_grad_h)
        for (int idx = tid; idx < w2_size; idx += blockDim.x) {
            int row = idx / d_hidden;
            int col = idx % d_hidden;
            dM[w2_off + idx] = retention * dM[w2_off + idx]
                             + d_err[row] * hidden[col]
                             + error_buf[row] * grad_pre[col];
        }

        // b2: d_M = retention*d_M + d_err
        for (int i = tid; i < b2_size; i += blockDim.x) {
            dM[b2_off + i] = retention * dM[b2_off + i] + d_err[i];
        }

        // d_S *= η
        for (int idx = tid; idx < state_size; idx += blockDim.x) {
            dS[idx] = eta_t * dS[idx];
        }
        __syncthreads();
    }

    // ── Accumulate d_m_initial, d_s_initial across batch (atomicAdd) ──
    for (int idx = tid; idx < state_size; idx += blockDim.x) {
        atomicAdd(&d_m_initial[idx], dM[idx]);
        atomicAdd(&d_s_initial[idx], dS[idx]);
    }
}

// ══════════════════════════════════════════════════════════════════════
// C wrapper — dispatch to template instantiation + workspace alloc
// ══════════════════════════════════════════════════════════════════════

extern "C" void titans_mlp_backward_f32_cuda(
    const float* k_mem, const float* v_mem, const float* q_mem,
    const float* alpha, const float* theta, const float* eta,
    const float* m_states, const float* s_states, const float* d_y,
    float* d_k_mem, float* d_v_mem, float* d_q_mem,
    float* d_alpha, float* d_theta, float* d_eta,
    float* d_m_initial, float* d_s_initial,
    int seq_len, int d, int d_hidden, int batch_size,
    int input_stride, int m_stride,
    int activation, float m_norm_max)
{
    if (d <= 0 || d_hidden <= 0) {
        fprintf(stderr, "titans_mlp_backward_f32_cuda: d=%d, d_hidden=%d must be > 0.\n",
                d, d_hidden);
        exit(1);
    }

    // Block size: d threads (NOT d_hidden) — backward has higher register
    // pressure, so we cap at d to stay within register budget.
    int block_size = (d < 1024) ? d : 1024;
    int rounded = 1;
    while (rounded < block_size) rounded <<= 1;
    if (rounded > 1024) rounded >>= 1;
    block_size = rounded;

    dim3 grid(batch_size);
    dim3 block(block_size);

    // Shared memory: d + 2*block_size + 4*d_hidden floats
    // (reduction buffers error_buf/d_err use block_size, which may be > d after rounding)
    int smem_bytes = (d + 2 * block_size + 4 * d_hidden) * (int)sizeof(float);

    if (smem_bytes > 163840) {
        fprintf(stderr, "titans_mlp_backward_f32_cuda: d=%d dh=%d requires %d bytes "
                "shared memory (limit 163840).\n", d, d_hidden, smem_bytes);
        exit(1);
    }

    // Validate activation ID
    if (activation != ACT_GELU && activation != ACT_SILU && activation != ACT_RELU) {
        fprintf(stderr, "titans_mlp_backward_f32_cuda: invalid activation=%d "
                "(expected 0=GELU, 1=SiLU, 2=ReLU).\n", activation);
        abort();
    }

    // Allocate per-batch d_M and d_S workspaces
    int state_size = d_hidden * d + d_hidden + d * d_hidden + d;
    float* d_M_work = nullptr;
    float* d_S_work = nullptr;
    check_alloc("titans_mlp_backward: cudaMalloc d_M_work",
                cudaMalloc(&d_M_work, (size_t)batch_size * state_size * sizeof(float)));
    check_alloc("titans_mlp_backward: cudaMalloc d_S_work",
                cudaMalloc(&d_S_work, (size_t)batch_size * state_size * sizeof(float)));

    // Dispatch by activation
    #define LAUNCH_BW(ACTV, LABEL) do { \
        check_alloc("titans_mlp_backward: cudaFuncSetAttribute", \
                     cudaFuncSetAttribute(titans_mlp_backward_kernel<ACTV>, \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)); \
        titans_mlp_backward_kernel<ACTV><<<grid, block, smem_bytes>>>( \
            k_mem, v_mem, q_mem, alpha, theta, eta, \
            m_states, s_states, d_y, \
            d_k_mem, d_v_mem, d_q_mem, \
            d_alpha, d_theta, d_eta, \
            d_m_initial, d_s_initial, \
            d_M_work, d_S_work, \
            seq_len, d, d_hidden, input_stride, m_stride); \
        check_launch("titans_mlp_backward_kernel<" LABEL ">", d, smem_bytes); \
    } while(0)

    if (activation == ACT_GELU) {
        LAUNCH_BW(ACT_GELU, "GELU");
    } else if (activation == ACT_SILU) {
        LAUNCH_BW(ACT_SILU, "SiLU");
    } else {
        LAUNCH_BW(ACT_RELU, "ReLU");
    }

    #undef LAUNCH_BW

    cudaDeviceSynchronize();
    cudaFree(d_M_work);
    cudaFree(d_S_work);
}
