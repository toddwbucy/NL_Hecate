/// Chunkwise training for self-referential Titans (HOPE §8.2, Eqs 90-93).
///
/// Sequential self-ref processes tokens one-by-one: each DGD update computes
/// `error = M_{t-1} @ k - v` using the current M state. Chunkwise freezes M
/// at chunk boundaries: `error = M_frozen @ k - v`. Within a chunk, all C tokens
/// compute errors from the same frozen snapshot, enabling parallel gradient
/// computation. The DGD recurrence (retention + update) still runs sequentially
/// on current M.
///
/// For C=1, chunkwise is bit-identical to sequential (M_frozen == M_{t-1}).
///
/// Source: HOPE (2512.24695) §8.2 Eqs 90-93.

use crate::tensor::{matmul_f32, sigmoid_f32, softplus_f32};
use crate::dgd::{dgd_error, dgd_update};
use crate::self_ref::{SelfRefState, SelfRefCache, SelfRefParamGrads};

/// Cache for chunkwise self-referential backward pass.
#[derive(Clone, Debug)]
pub struct ChunkwiseSelfRefCache {
    /// Standard per-token cache (reused from self_ref.rs — all M histories, reads, gates).
    pub inner: SelfRefCache,
    /// Chunk size used during forward.
    pub chunk_size: usize,
    /// Number of chunks (including remainder).
    pub num_chunks: usize,
    /// Frozen M snapshots at chunk boundaries: [num_chunks * 6 * d * d].
    /// Component order: m_k(0), m_v(1), m_q(2), m_eta(3), m_alpha(4), m_mem(5).
    pub frozen_snapshots: Vec<f32>,
}

/// Sigmoid backward: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
fn sigmoid_backward(x: f32) -> f32 {
    let s = sigmoid_f32(x);
    s * (1.0 - s)
}

/// Softplus backward: d/dx softplus(x) = sigmoid(x).
fn softplus_backward(x: f32) -> f32 {
    sigmoid_f32(x)
}

/// Chunkwise self-referential forward pass (HOPE §8.2).
///
/// Same per-token structure as `self_ref_step()` with two changes:
///   1. At chunk boundaries (t % chunk_size == 0), snapshot all 6 M states.
///   2. DGD error uses frozen M, DGD recurrence uses current M.
///
/// For C=1, every token is a chunk boundary, so frozen == current == sequential.
///
/// Returns (y [seq_len * d], ChunkwiseSelfRefCache).
pub fn chunkwise_self_ref_step(
    self_ref: &mut SelfRefState,
    m_mem: &mut [f32],
    embedded: &[f32],
    seq_len: usize,
    d: usize,
    chunk_size: usize,
    self_generated_values: bool,
) -> (Vec<f32>, ChunkwiseSelfRefCache) {
    debug_assert_eq!(embedded.len(), seq_len * d);
    debug_assert_eq!(m_mem.len(), d * d);
    debug_assert!(self_ref.is_active(), "chunkwise_self_ref_step called on empty SelfRefState");
    assert!(chunk_size >= 1, "chunk_size must be >= 1");

    let dd = d * d;
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    let mut cache = SelfRefCache::new(seq_len, d);
    cache.embedded = embedded.to_vec();
    cache.self_generated_values = self_generated_values;
    if self_generated_values {
        cache.v_hat_targets = vec![0.0f32; 6 * seq_len * d];
    }

    // Snapshot initial states (t=0)
    cache.m_k_states[..dd].copy_from_slice(&self_ref.m_k);
    cache.m_v_states[..dd].copy_from_slice(&self_ref.m_v);
    cache.m_q_states[..dd].copy_from_slice(&self_ref.m_q);
    cache.m_eta_states[..dd].copy_from_slice(&self_ref.m_eta);
    cache.m_alpha_states[..dd].copy_from_slice(&self_ref.m_alpha);
    cache.m_mem_states[..dd].copy_from_slice(m_mem);

    // Frozen snapshots: [num_chunks * 6 * dd]
    let mut frozen_snapshots = vec![0.0f32; num_chunks * 6 * dd];

    let mut y = vec![0.0f32; seq_len * d];

    // Reusable buffers
    let mut k_t = vec![0.0f32; d];
    let mut v_t = vec![0.0f32; d];
    let mut q_t = vec![0.0f32; d];
    let mut eta_raw_t = vec![0.0f32; d];
    let mut alpha_raw_t = vec![0.0f32; d];
    let mut y_t = vec![0.0f32; d];
    let mut v_hat_buf = vec![0.0f32; d];
    let mut error_buf = vec![0.0f32; d];
    let sd = seq_len * d;

    for t in 0..seq_len {
        let chunk_idx = t / chunk_size;
        let x_t = &embedded[t * d..(t + 1) * d];

        // ── Chunk boundary: snapshot frozen M ──
        if t % chunk_size == 0 {
            let base = chunk_idx * 6 * dd;
            frozen_snapshots[base..base + dd].copy_from_slice(&self_ref.m_k);
            frozen_snapshots[base + dd..base + 2 * dd].copy_from_slice(&self_ref.m_v);
            frozen_snapshots[base + 2 * dd..base + 3 * dd].copy_from_slice(&self_ref.m_q);
            frozen_snapshots[base + 3 * dd..base + 4 * dd].copy_from_slice(&self_ref.m_eta);
            frozen_snapshots[base + 4 * dd..base + 5 * dd].copy_from_slice(&self_ref.m_alpha);
            frozen_snapshots[base + 5 * dd..base + 6 * dd].copy_from_slice(m_mem);
        }

        // Step 1: Adaptive projections — read from current M
        matmul_f32(&self_ref.m_k, x_t, &mut k_t, d, d, 1);
        matmul_f32(&self_ref.m_v, x_t, &mut v_t, d, d, 1);
        matmul_f32(&self_ref.m_q, x_t, &mut q_t, d, d, 1);
        matmul_f32(&self_ref.m_eta, x_t, &mut eta_raw_t, d, d, 1);
        matmul_f32(&self_ref.m_alpha, x_t, &mut alpha_raw_t, d, d, 1);

        // Cache per-token reads
        cache.k_mem[t * d..(t + 1) * d].copy_from_slice(&k_t);
        cache.v_mem[t * d..(t + 1) * d].copy_from_slice(&v_t);
        cache.q_mem[t * d..(t + 1) * d].copy_from_slice(&q_t);
        cache.eta_raw[t * d..(t + 1) * d].copy_from_slice(&eta_raw_t);
        cache.alpha_raw[t * d..(t + 1) * d].copy_from_slice(&alpha_raw_t);

        // Step 2: Reduce d-dim gate outputs to scalars
        let alpha_mean: f32 = alpha_raw_t.iter().sum::<f32>() / d as f32;
        let eta_mean: f32 = eta_raw_t.iter().sum::<f32>() / d as f32;
        let alpha_t = sigmoid_f32(alpha_mean);
        let theta_t = softplus_f32(eta_mean);
        cache.alpha[t] = alpha_t;
        cache.theta[t] = theta_t;

        // Step 3: Main memory read — y_t = M_mem @ q_t (current M, not frozen)
        matmul_f32(m_mem, &q_t, &mut y_t, d, d, 1);
        y[t * d..(t + 1) * d].copy_from_slice(&y_t);
        cache.y[t * d..(t + 1) * d].copy_from_slice(&y_t);

        // Get frozen M slices for error computation
        let base = chunk_idx * 6 * dd;
        let frozen_mk = &frozen_snapshots[base..base + dd];
        let frozen_mv = &frozen_snapshots[base + dd..base + 2 * dd];
        let frozen_mq = &frozen_snapshots[base + 2 * dd..base + 3 * dd];
        let frozen_meta = &frozen_snapshots[base + 3 * dd..base + 4 * dd];
        let frozen_malpha = &frozen_snapshots[base + 4 * dd..base + 5 * dd];
        let frozen_mmem = &frozen_snapshots[base + 5 * dd..base + 6 * dd];

        // Step 3.5: Self-generated value targets (Phase 3, HOPE Eq 84)
        // When enabled, v̂_□ = M_frozen_{□} @ v_t (frozen, not current — part of gradient freeze).
        if self_generated_values {
            let t_off = t * d;
            matmul_f32(frozen_mk, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[0 * sd + t_off..0 * sd + t_off + d].copy_from_slice(&v_hat_buf);
            matmul_f32(frozen_mv, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[1 * sd + t_off..1 * sd + t_off + d].copy_from_slice(&v_hat_buf);
            matmul_f32(frozen_mq, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[2 * sd + t_off..2 * sd + t_off + d].copy_from_slice(&v_hat_buf);
            matmul_f32(frozen_meta, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[3 * sd + t_off..3 * sd + t_off + d].copy_from_slice(&v_hat_buf);
            matmul_f32(frozen_malpha, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[4 * sd + t_off..4 * sd + t_off + d].copy_from_slice(&v_hat_buf);
            matmul_f32(frozen_mmem, &v_t, &mut v_hat_buf, d, d, 1);
            cache.v_hat_targets[5 * sd + t_off..5 * sd + t_off + d].copy_from_slice(&v_hat_buf);
        }

        // Step 4: DGD update all 6 memories using FROZEN M for error, CURRENT M for recurrence
        // For each memory: error = M_frozen @ k - v_hat, then M_current = (1-alpha)*M_current - theta*outer(error, k)
        let memories: [(&mut [f32], &[f32]); 6] = unsafe {
            // SAFETY: All 6 memories are non-overlapping slices.
            // We need mutable references to 5 fields of self_ref + m_mem simultaneously.
            // The frozen snapshots are read-only immutable borrows from a separate buffer.
            // This is safe because none of the mutable slices overlap with each other or with
            // the frozen slices.
            let m_k_ptr = self_ref.m_k.as_mut_ptr();
            let m_v_ptr = self_ref.m_v.as_mut_ptr();
            let m_q_ptr = self_ref.m_q.as_mut_ptr();
            let m_eta_ptr = self_ref.m_eta.as_mut_ptr();
            let m_alpha_ptr = self_ref.m_alpha.as_mut_ptr();
            let m_mem_ptr = m_mem.as_mut_ptr();
            [
                (std::slice::from_raw_parts_mut(m_k_ptr, dd), frozen_mk),
                (std::slice::from_raw_parts_mut(m_v_ptr, dd), frozen_mv),
                (std::slice::from_raw_parts_mut(m_q_ptr, dd), frozen_mq),
                (std::slice::from_raw_parts_mut(m_eta_ptr, dd), frozen_meta),
                (std::slice::from_raw_parts_mut(m_alpha_ptr, dd), frozen_malpha),
                (std::slice::from_raw_parts_mut(m_mem_ptr, dd), frozen_mmem),
            ]
        };

        for (comp_idx, (m_current, m_frozen)) in memories.into_iter().enumerate() {
            let v_target = if self_generated_values {
                let t_off = t * d;
                &cache.v_hat_targets[comp_idx * sd + t_off..comp_idx * sd + t_off + d]
            } else {
                &v_t[..]
            };
            crate::dgd::dgd_error_into(m_frozen, &k_t, v_target, d, &mut error_buf);
            dgd_update(m_current, &error_buf, &k_t, alpha_t, theta_t, d);
        }

        // Snapshot updated states (t+1)
        let off = (t + 1) * dd;
        cache.m_k_states[off..off + dd].copy_from_slice(&self_ref.m_k);
        cache.m_v_states[off..off + dd].copy_from_slice(&self_ref.m_v);
        cache.m_q_states[off..off + dd].copy_from_slice(&self_ref.m_q);
        cache.m_eta_states[off..off + dd].copy_from_slice(&self_ref.m_eta);
        cache.m_alpha_states[off..off + dd].copy_from_slice(&self_ref.m_alpha);
        cache.m_mem_states[off..off + dd].copy_from_slice(m_mem);
    }

    let outer_cache = ChunkwiseSelfRefCache {
        inner: cache,
        chunk_size,
        num_chunks,
        frozen_snapshots,
    };

    (y, outer_cache)
}

/// Frozen DGD backward: splits gradient into retention chain (→ M_prev) and error chain (→ M_frozen).
///
/// In standard DGD backward, both retention and error chains flow to M_prev:
///   d_m_prev = (1-alpha) * d_m_out - theta * outer(d_m_out @ k, k)
///
/// In frozen backward, the error chain redirects to M_frozen:
///   d_m_prev   = (1-alpha) * d_m_out                          (retention only)
///   d_m_frozen = -theta * outer(d_m_out @ k, k)               (error chain)
///   d_k        = M_frozen^T @ d_error - theta * E^T @ d_m_out (M_frozen replaces M_prev)
///   d_v, d_alpha, d_theta unchanged from standard backward.
///
/// Returns (d_m_prev, d_m_frozen, d_k, d_v, d_alpha, d_theta).
fn dgd_frozen_backward(
    d_m_out: &[f32],
    m_prev: &[f32],
    m_frozen: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    theta: f32,
    d: usize,
) -> FrozenDgdGrads {
    let dd = d * d;

    // Prediction error from frozen M: error = M_frozen @ k - v
    let error = dgd_error(m_frozen, k, v, d);

    // dm_out_k = d_m_out @ k: [d]
    let mut dm_out_k = vec![0.0f32; d];
    matmul_f32(d_m_out, k, &mut dm_out_k, d, d, 1);

    // d_m_prev = (1-alpha) * d_m_out (retention chain only)
    let mut d_m_prev = vec![0.0f32; dd];
    for i in 0..dd {
        d_m_prev[i] = (1.0 - alpha) * d_m_out[i];
    }

    // d_m_frozen = -theta * outer(dm_out_k, k) (error chain redirected)
    let mut d_m_frozen = vec![0.0f32; dd];
    for i in 0..d {
        let scaled = -theta * dm_out_k[i];
        for j in 0..d {
            d_m_frozen[i * d + j] = scaled * k[j];
        }
    }

    // d_k = -theta * (M_frozen^T @ dm_out_k + E^T @ d_m_out)
    // Term 1: M_frozen^T @ dm_out_k
    let mut mft_dm_k = vec![0.0f32; d];
    for j in 0..d {
        let mut sum = 0.0f32;
        for i in 0..d {
            sum += m_frozen[i * d + j] * dm_out_k[i]; // M_frozen^T
        }
        mft_dm_k[j] = sum;
    }

    // Term 2: E^T @ d_m_out: [1,d] @ [d,d] → [1,d]
    let mut et_dm = vec![0.0f32; d];
    for j in 0..d {
        let mut sum = 0.0f32;
        for i in 0..d {
            sum += error[i] * d_m_out[i * d + j];
        }
        et_dm[j] = sum;
    }

    let mut d_k = vec![0.0f32; d];
    for i in 0..d {
        d_k[i] = -theta * (mft_dm_k[i] + et_dm[i]);
    }

    // d_v = theta * dm_out_k (same as standard)
    let mut d_v = vec![0.0f32; d];
    for i in 0..d {
        d_v[i] = theta * dm_out_k[i];
    }

    // d_alpha = -frobenius_dot(m_prev, d_m_out) (uses m_prev for retention chain)
    let mut d_alpha = 0.0f32;
    for i in 0..dd {
        d_alpha -= m_prev[i] * d_m_out[i];
    }

    // d_theta = -frobenius_dot(outer(error, k), d_m_out) (error from frozen M)
    let mut d_theta = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            d_theta -= error[i] * k[j] * d_m_out[i * d + j];
        }
    }

    FrozenDgdGrads { d_m_prev, d_m_frozen, d_k, d_v, d_alpha, d_theta }
}

struct FrozenDgdGrads {
    d_m_prev: Vec<f32>,
    d_m_frozen: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
    d_alpha: f32,
    d_theta: f32,
}

/// Backward pass for chunkwise self-referential step.
///
/// Same reverse-token-loop structure as `self_ref_step_backward()`. The key
/// difference: DGD backward for each memory uses `dgd_frozen_backward` which
/// splits the gradient into retention chain (→ M_prev) and error chain (→ M_frozen).
/// At chunk boundaries, the accumulated frozen gradients merge with the recurrence.
///
/// Returns (d_embedded [seq_len * d], SelfRefParamGrads for initial states).
pub fn chunkwise_self_ref_step_backward(
    cache: &ChunkwiseSelfRefCache,
    d_y: &[f32],
) -> (Vec<f32>, SelfRefParamGrads) {
    let c = &cache.inner;
    let s = c.seq_len;
    let d = c.d;
    let dd = d * d;
    let chunk_size = cache.chunk_size;

    debug_assert_eq!(d_y.len(), s * d);
    let self_generated_values = c.self_generated_values;

    let mut d_embedded = vec![0.0f32; s * d];

    // Running dM accumulators for each memory
    let mut dm_k = vec![0.0f32; dd];
    let mut dm_v = vec![0.0f32; dd];
    let mut dm_q = vec![0.0f32; dd];
    let mut dm_eta = vec![0.0f32; dd];
    let mut dm_alpha = vec![0.0f32; dd];
    let mut dm_mem = vec![0.0f32; dd];

    // Frozen gradient accumulators (per-chunk, reset at chunk boundaries)
    let mut dfrozen_k = vec![0.0f32; dd];
    let mut dfrozen_v = vec![0.0f32; dd];
    let mut dfrozen_q = vec![0.0f32; dd];
    let mut dfrozen_eta = vec![0.0f32; dd];
    let mut dfrozen_alpha = vec![0.0f32; dd];
    let mut dfrozen_mem = vec![0.0f32; dd];

    let mut dq_t = vec![0.0f32; d];
    let sd = s * d;

    // Track which chunk we're in (going backward)
    let mut current_chunk = if s > 0 { (s - 1) / chunk_size } else { 0 };

    for t in (0..s).rev() {
        let chunk_idx = t / chunk_size;

        // Check if we've crossed a chunk boundary going backward.
        // When we finish processing all tokens in chunk `current_chunk` and move
        // to chunk `chunk_idx` (which is one less), we merge the accumulated
        // frozen gradients from the chunk we just finished into the dM accumulators.
        if chunk_idx < current_chunk {
            // Merge frozen grads into recurrence dM accumulators
            for i in 0..dd {
                dm_k[i] += dfrozen_k[i];
                dm_v[i] += dfrozen_v[i];
                dm_q[i] += dfrozen_q[i];
                dm_eta[i] += dfrozen_eta[i];
                dm_alpha[i] += dfrozen_alpha[i];
                dm_mem[i] += dfrozen_mem[i];
            }
            // Reset frozen accumulators for the new chunk
            dfrozen_k.fill(0.0);
            dfrozen_v.fill(0.0);
            dfrozen_q.fill(0.0);
            dfrozen_eta.fill(0.0);
            dfrozen_alpha.fill(0.0);
            dfrozen_mem.fill(0.0);
            current_chunk = chunk_idx;
        }

        let x_t = &c.embedded[t * d..(t + 1) * d];
        let k_t = &c.k_mem[t * d..(t + 1) * d];
        let v_t = &c.v_mem[t * d..(t + 1) * d];
        let q_t_val = &c.q_mem[t * d..(t + 1) * d];
        let alpha_t = c.alpha[t];
        let theta_t = c.theta[t];
        let dy_t = &d_y[t * d..(t + 1) * d];

        // M state at time t (before update)
        let m_k_t = &c.m_k_states[t * dd..(t + 1) * dd];
        let m_v_t = &c.m_v_states[t * dd..(t + 1) * dd];
        let m_q_t = &c.m_q_states[t * dd..(t + 1) * dd];
        let m_eta_t = &c.m_eta_states[t * dd..(t + 1) * dd];
        let m_alpha_t = &c.m_alpha_states[t * dd..(t + 1) * dd];
        let m_mem_t = &c.m_mem_states[t * dd..(t + 1) * dd];

        // Frozen M for this chunk
        let base = chunk_idx * 6 * dd;
        let frozen_mk = &cache.frozen_snapshots[base..base + dd];
        let frozen_mv = &cache.frozen_snapshots[base + dd..base + 2 * dd];
        let frozen_mq = &cache.frozen_snapshots[base + 2 * dd..base + 3 * dd];
        let frozen_meta = &cache.frozen_snapshots[base + 3 * dd..base + 4 * dd];
        let frozen_malpha = &cache.frozen_snapshots[base + 4 * dd..base + 5 * dd];
        let frozen_mmem = &cache.frozen_snapshots[base + 5 * dd..base + 6 * dd];

        // ── Step 1: Main memory read backward ──
        // y_t = M_mem_t @ q_t → dq_t = M_mem_t^T @ dy_t
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d {
                sum += m_mem_t[i * d + j] * dy_t[i];
            }
            dq_t[j] = sum;
        }

        // ── Step 2: Resolve value targets ──
        let (v_hat_k, v_hat_v, v_hat_q, v_hat_eta, v_hat_alpha, v_hat_mem);
        if self_generated_values {
            v_hat_k     = &c.v_hat_targets[0 * sd + t * d..0 * sd + (t + 1) * d];
            v_hat_v     = &c.v_hat_targets[1 * sd + t * d..1 * sd + (t + 1) * d];
            v_hat_q     = &c.v_hat_targets[2 * sd + t * d..2 * sd + (t + 1) * d];
            v_hat_eta   = &c.v_hat_targets[3 * sd + t * d..3 * sd + (t + 1) * d];
            v_hat_alpha = &c.v_hat_targets[4 * sd + t * d..4 * sd + (t + 1) * d];
            v_hat_mem   = &c.v_hat_targets[5 * sd + t * d..5 * sd + (t + 1) * d];
        } else {
            v_hat_k = v_t;
            v_hat_v = v_t;
            v_hat_q = v_t;
            v_hat_eta = v_t;
            v_hat_alpha = v_t;
            v_hat_mem = v_t;
        }

        // ── Step 3: Frozen DGD backward for main memory ──
        let fg = dgd_frozen_backward(&dm_mem, m_mem_t, frozen_mmem, k_t, v_hat_mem, alpha_t, theta_t, d);
        dm_mem.copy_from_slice(&fg.d_m_prev);
        for i in 0..dd { dfrozen_mem[i] += fg.d_m_frozen[i]; }

        // Add read gradient: dL/dM_mem_read = outer(dy_t, q_t)
        for i in 0..d {
            for j in 0..d {
                dm_mem[i * d + j] += dy_t[i] * q_t_val[j];
            }
        }

        let mut dk_t_total = fg.d_k;
        let mut dv_t_total = vec![0.0f32; d];
        let mut dalpha_total = fg.d_alpha;
        let mut dtheta_total = fg.d_theta;

        // Self-gen chain for main memory
        if self_generated_values {
            let d_v_hat = &fg.d_v;
            for i in 0..d {
                for j in 0..d {
                    // v_hat_mem = M_frozen_mem @ v_t → dM_frozen_mem += outer(d_v_hat, v_t)
                    dfrozen_mem[i * d + j] += d_v_hat[i] * v_t[j];
                }
            }
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += frozen_mmem[i * d + j] * d_v_hat[i];
                }
                dv_t_total[j] += sum;
            }
        } else {
            for i in 0..d { dv_t_total[i] += fg.d_v[i]; }
        }

        // ── Step 4: Frozen DGD backward for 5 component memories ──
        let mut dx_t = vec![0.0f32; d];

        // Helper: process one component memory's frozen DGD backward
        macro_rules! component_backward {
            ($dm:expr, $dfrozen:expr, $m_t:expr, $frozen_m:expr, $v_hat:expr) => {
                let fg = dgd_frozen_backward($dm, $m_t, $frozen_m, k_t, $v_hat, alpha_t, theta_t, d);
                $dm.copy_from_slice(&fg.d_m_prev);
                for i in 0..dd { $dfrozen[i] += fg.d_m_frozen[i]; }
                for i in 0..d { dk_t_total[i] += fg.d_k[i]; }
                dalpha_total += fg.d_alpha;
                dtheta_total += fg.d_theta;
                if self_generated_values {
                    for i in 0..d {
                        for j in 0..d {
                            $dfrozen[i * d + j] += fg.d_v[i] * v_t[j];
                        }
                    }
                    for j in 0..d {
                        let mut sum = 0.0f32;
                        for i in 0..d { sum += $frozen_m[i * d + j] * fg.d_v[i]; }
                        dv_t_total[j] += sum;
                    }
                } else {
                    for i in 0..d { dv_t_total[i] += fg.d_v[i]; }
                }
            };
        }

        component_backward!(&mut dm_alpha, &mut dfrozen_alpha, m_alpha_t, frozen_malpha, v_hat_alpha);
        component_backward!(&mut dm_eta, &mut dfrozen_eta, m_eta_t, frozen_meta, v_hat_eta);
        component_backward!(&mut dm_q, &mut dfrozen_q, m_q_t, frozen_mq, v_hat_q);
        component_backward!(&mut dm_v, &mut dfrozen_v, m_v_t, frozen_mv, v_hat_v);
        component_backward!(&mut dm_k, &mut dfrozen_k, m_k_t, frozen_mk, v_hat_k);

        // ── Step 5: Gate backward ──
        let alpha_raw_t = &c.alpha_raw[t * d..(t + 1) * d];
        let eta_raw_t = &c.eta_raw[t * d..(t + 1) * d];
        let alpha_mean: f32 = alpha_raw_t.iter().sum::<f32>() / d as f32;
        let eta_mean: f32 = eta_raw_t.iter().sum::<f32>() / d as f32;

        let dalpha_mean = dalpha_total * sigmoid_backward(alpha_mean);
        let deta_mean = dtheta_total * softplus_backward(eta_mean);
        let dalpha_per_dim = dalpha_mean / d as f32;
        let deta_per_dim = deta_mean / d as f32;

        // ── Step 6: Component read backward ──
        // dk_t_total → through k_t = M_k_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_k[i * d + j] += dk_t_total[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_k_t[i * d + j] * dk_t_total[i]; }
            dx_t[j] += sum;
        }

        // dv_t_total → through v_t = M_v_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_v[i * d + j] += dv_t_total[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_v_t[i * d + j] * dv_t_total[i]; }
            dx_t[j] += sum;
        }

        // dq_t → through q_t = M_q_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_q[i * d + j] += dq_t[i] * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_q_t[i * d + j] * dq_t[i]; }
            dx_t[j] += sum;
        }

        // d_eta_raw → through eta_raw = M_eta_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_eta[i * d + j] += deta_per_dim * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_eta_t[i * d + j] * deta_per_dim; }
            dx_t[j] += sum;
        }

        // d_alpha_raw → through alpha_raw = M_alpha_t @ x_t
        for i in 0..d {
            for j in 0..d {
                dm_alpha[i * d + j] += dalpha_per_dim * x_t[j];
            }
        }
        for j in 0..d {
            let mut sum = 0.0f32;
            for i in 0..d { sum += m_alpha_t[i * d + j] * dalpha_per_dim; }
            dx_t[j] += sum;
        }

        d_embedded[t * d..(t + 1) * d].copy_from_slice(&dx_t);
    }

    // Merge the final chunk's frozen gradients (chunk 0, processed last)
    for i in 0..dd {
        dm_k[i] += dfrozen_k[i];
        dm_v[i] += dfrozen_v[i];
        dm_q[i] += dfrozen_q[i];
        dm_eta[i] += dfrozen_eta[i];
        dm_alpha[i] += dfrozen_alpha[i];
        dm_mem[i] += dfrozen_mem[i];
    }

    let grads = SelfRefParamGrads {
        d_m_k: dm_k,
        d_m_v: dm_v,
        d_m_q: dm_q,
        d_m_eta: dm_eta,
        d_m_alpha: dm_alpha,
        d_m_mem: dm_mem,
    };

    (d_embedded, grads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::self_ref::{self_ref_step, self_ref_step_backward};

    /// Helper: create initialized self-ref state with identity-scaled matrices.
    fn make_state(d: usize, scale: f32) -> SelfRefState {
        let mut state = SelfRefState::new(d);
        for i in 0..d {
            state.m_k[i * d + i] = scale;
            state.m_v[i * d + i] = scale;
            state.m_q[i * d + i] = scale;
            state.m_eta[i * d + i] = scale;
            state.m_alpha[i * d + i] = scale;
        }
        state
    }

    fn make_m_mem(d: usize, scale: f32) -> Vec<f32> {
        let mut m = vec![0.0f32; d * d];
        for i in 0..d { m[i * d + i] = scale; }
        m
    }

    fn make_embedded(seq_len: usize, d: usize) -> Vec<f32> {
        (0..seq_len * d).map(|i| (i as f32) * 0.1 + 0.05).collect()
    }

    fn simple_loss(y: &[f32]) -> f32 {
        y.iter().map(|x| x * x).sum::<f32>() / 2.0
    }

    fn simple_dloss(y: &[f32]) -> Vec<f32> {
        y.to_vec()
    }

    // ── C=1 correctness tests ──

    #[test]
    fn test_c1_matches_sequential() {
        let d = 4;
        let seq_len = 4;
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        // Sequential
        let mut s1 = state0.clone();
        let mut mm1 = m_mem0.clone();
        let (y_seq, _) = self_ref_step(&mut s1, &mut mm1, &embedded, seq_len, d, false);

        // Chunkwise C=1
        let mut s2 = state0.clone();
        let mut mm2 = m_mem0.clone();
        let (y_chunk, _) = chunkwise_self_ref_step(&mut s2, &mut mm2, &embedded, seq_len, d, 1, false);

        // Bit-identical output
        assert_eq!(y_seq.len(), y_chunk.len());
        for i in 0..y_seq.len() {
            assert!((y_seq[i] - y_chunk[i]).abs() < 1e-10,
                "C=1 output mismatch at {i}: seq={} chunk={}", y_seq[i], y_chunk[i]);
        }

        // Final states should be identical
        assert_eq!(s1.m_k, s2.m_k, "M_k final state mismatch");
        assert_eq!(s1.m_v, s2.m_v, "M_v final state mismatch");
        assert_eq!(mm1, mm2, "M_mem final state mismatch");
    }

    #[test]
    fn test_c1_self_gen_matches_sequential() {
        let d = 4;
        let seq_len = 4;
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        // Sequential with self-gen
        let mut s1 = state0.clone();
        let mut mm1 = m_mem0.clone();
        let (y_seq, _) = self_ref_step(&mut s1, &mut mm1, &embedded, seq_len, d, true);

        // Chunkwise C=1 with self-gen
        let mut s2 = state0.clone();
        let mut mm2 = m_mem0.clone();
        let (y_chunk, _) = chunkwise_self_ref_step(&mut s2, &mut mm2, &embedded, seq_len, d, 1, true);

        for i in 0..y_seq.len() {
            assert!((y_seq[i] - y_chunk[i]).abs() < 1e-10,
                "C=1 self-gen output mismatch at {i}: seq={} chunk={}", y_seq[i], y_chunk[i]);
        }
    }

    // ── C>1 forward tests ──

    #[test]
    fn test_c2_forward_finite() {
        let d = 4;
        let seq_len = 4;
        let mut state = make_state(d, 0.1);
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, 2, false);
        assert!(y.iter().all(|x| x.is_finite()), "C=2 output should be finite");
        assert_eq!(cache.num_chunks, 2);
        assert_eq!(cache.chunk_size, 2);
    }

    #[test]
    fn test_c4_forward_finite() {
        let d = 4;
        let seq_len = 7; // not divisible by 4 → remainder chunk
        let mut state = make_state(d, 0.1);
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, 4, false);
        assert!(y.iter().all(|x| x.is_finite()), "C=4 remainder output should be finite");
        assert_eq!(cache.num_chunks, 2); // ceil(7/4) = 2
    }

    // ── Backward shape and finiteness tests ──

    #[test]
    fn test_backward_shapes() {
        let d = 4;
        let seq_len = 4;
        let mut state = make_state(d, 0.1);
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, 2, false);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);

        assert_eq!(d_embedded.len(), seq_len * d);
        assert_eq!(grads.d_m_k.len(), d * d);
        assert_eq!(grads.d_m_v.len(), d * d);
        assert_eq!(grads.d_m_q.len(), d * d);
        assert_eq!(grads.d_m_eta.len(), d * d);
        assert_eq!(grads.d_m_alpha.len(), d * d);
        assert_eq!(grads.d_m_mem.len(), d * d);
    }

    #[test]
    fn test_backward_finite() {
        let d = 4;
        let seq_len = 4;
        let mut state = make_state(d, 0.1);
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, 2, false);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);

        assert!(d_embedded.iter().all(|x| x.is_finite()), "d_embedded not finite");
        assert!(grads.d_m_k.iter().all(|x| x.is_finite()), "dM_k not finite");
        assert!(grads.d_m_mem.iter().all(|x| x.is_finite()), "dM_mem not finite");
    }

    #[test]
    fn test_backward_nonzero() {
        let d = 4;
        let seq_len = 4;
        let mut state = make_state(d, 0.1);
        // Set gate memories to produce nonzero gates
        for i in 0..d { state.m_eta[i * d + i] = 0.5; }
        for i in 0..d { state.m_alpha[i * d + i] = 0.5; }
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, 2, false);
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);

        let de_norm: f32 = d_embedded.iter().map(|x| x * x).sum();
        assert!(de_norm > 1e-10, "d_embedded should be nonzero, norm={de_norm}");
        let dk_norm: f32 = grads.d_m_k.iter().map(|x| x * x).sum();
        assert!(dk_norm > 0.0, "dM_k should be nonzero, norm={dk_norm}");
        let dmem_norm: f32 = grads.d_m_mem.iter().map(|x| x * x).sum();
        assert!(dmem_norm > 0.0, "dM_mem should be nonzero, norm={dmem_norm}");
    }

    // ── FD gradient checks ──

    #[test]
    fn test_c1_fd_check_embedded() {
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded = make_embedded(seq_len, d);
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = chunkwise_self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, 1, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (d_embedded, _) = chunkwise_self_ref_step_backward(&cache, &d_y);

        let mut max_err = 0.0f32;
        for idx in 0..embedded.len() {
            let mut perturbed = embedded.clone();
            perturbed[idx] += eps;
            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            let (y_p, _) = chunkwise_self_ref_step(&mut s, &mut mm, &perturbed, seq_len, d, 1, false);
            let loss_p = simple_loss(&y_p);
            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = d_embedded[idx];
            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;
            if ana_grad.abs() > 1e-4 { max_err = max_err.max(rel_err); }
        }
        assert!(max_err < tol, "C=1 FD check d_embedded: max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_c1_fd_check_initial_mk() {
        let d = 4;
        let seq_len = 2;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded = make_embedded(seq_len, d);
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = chunkwise_self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, 1, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (_, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);

        let mut max_err = 0.0f32;
        for idx in 0..(d * d) {
            let mut sp = state0.clone();
            sp.m_k[idx] += eps;
            let mut mm = m_mem0.clone();
            let (y_p, _) = chunkwise_self_ref_step(&mut sp, &mut mm, &embedded, seq_len, d, 1, false);
            let loss_p = simple_loss(&y_p);
            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = grads.d_m_k[idx];
            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;
            if ana_grad.abs() > 1e-4 { max_err = max_err.max(rel_err); }
        }
        assert!(max_err < tol, "C=1 FD check dM_k: max_rel_err={max_err:.4e}");
    }

    #[test]
    fn test_c2_fd_check_embedded() {
        let d = 4;
        let seq_len = 4;
        let eps = 1e-3f32;
        let tol = 0.05;

        let embedded = make_embedded(seq_len, d);
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);

        let mut s = state0.clone();
        let mut mm = m_mem0.clone();
        let (y, cache) = chunkwise_self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, 2, false);
        let loss0 = simple_loss(&y);
        let d_y = simple_dloss(&y);
        let (d_embedded, _) = chunkwise_self_ref_step_backward(&cache, &d_y);

        let mut max_err = 0.0f32;
        for idx in 0..embedded.len() {
            let mut perturbed = embedded.clone();
            perturbed[idx] += eps;
            let mut s = state0.clone();
            let mut mm = m_mem0.clone();
            let (y_p, _) = chunkwise_self_ref_step(&mut s, &mut mm, &perturbed, seq_len, d, 2, false);
            let loss_p = simple_loss(&y_p);
            let fd_grad = (loss_p - loss0) / eps;
            let ana_grad = d_embedded[idx];
            let abs_diff = (fd_grad - ana_grad).abs();
            let denom = fd_grad.abs().max(ana_grad.abs()).max(1e-8);
            let rel_err = abs_diff / denom;
            if ana_grad.abs() > 1e-4 { max_err = max_err.max(rel_err); }
        }
        assert!(max_err < tol, "C=2 FD check d_embedded: max_rel_err={max_err:.4e}");
    }

    // ── Approximation quality tests ──

    #[test]
    fn test_c2_approx_bounded() {
        let d = 4;
        let seq_len = 4;
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        // Sequential
        let mut s1 = state0.clone();
        let mut mm1 = m_mem0.clone();
        let (y_seq, _) = self_ref_step(&mut s1, &mut mm1, &embedded, seq_len, d, false);

        // Chunkwise C=2
        let mut s2 = state0.clone();
        let mut mm2 = m_mem0.clone();
        let (y_c2, _) = chunkwise_self_ref_step(&mut s2, &mut mm2, &embedded, seq_len, d, 2, false);

        let diff: f32 = y_seq.iter().zip(y_c2.iter()).map(|(a, b)| (a - b).abs()).sum();
        let norm: f32 = y_seq.iter().map(|x| x.abs()).sum::<f32>().max(1e-8);
        let rel_diff = diff / norm;
        // C=2 should be close but not identical to sequential
        assert!(rel_diff < 0.5, "C=2 output too far from sequential: rel_diff={rel_diff:.4e}");
    }

    #[test]
    fn test_quality_degrades_with_c() {
        let d = 4;
        let seq_len = 8;
        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        // Sequential reference
        let mut s_ref = state0.clone();
        let mut mm_ref = m_mem0.clone();
        let (y_ref, _) = self_ref_step(&mut s_ref, &mut mm_ref, &embedded, seq_len, d, false);

        // C=2
        let mut s2 = state0.clone();
        let mut mm2 = m_mem0.clone();
        let (y_c2, _) = chunkwise_self_ref_step(&mut s2, &mut mm2, &embedded, seq_len, d, 2, false);

        // C=4
        let mut s4 = state0.clone();
        let mut mm4 = m_mem0.clone();
        let (y_c4, _) = chunkwise_self_ref_step(&mut s4, &mut mm4, &embedded, seq_len, d, 4, false);

        let diff2: f32 = y_ref.iter().zip(y_c2.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff4: f32 = y_ref.iter().zip(y_c4.iter()).map(|(a, b)| (a - b).abs()).sum();

        assert!(diff2 <= diff4 + 1e-10,
            "C=2 diff ({diff2}) should be <= C=4 diff ({diff4})");
    }

    #[test]
    fn test_self_gen_uses_frozen_for_vhat() {
        // Verify v_hat at t>0 within a chunk uses M_frozen, not M_current.
        let d = 4;
        let seq_len = 2;
        let chunk_size = 2; // Both tokens in one chunk

        let mut state = make_state(d, 0.2);
        // Make M_k distinguishable so v_hat changes with M
        for i in 0..d { state.m_k[i * d + i] = 0.3; }
        for i in 0..d { state.m_eta[i * d + i] = 1.0; }
        for i in 0..d { state.m_alpha[i * d + i] = 1.0; }
        let mut m_mem = make_m_mem(d, 0.2);
        let embedded = make_embedded(seq_len, d);

        let (_, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, chunk_size, true);

        // v_hat_k at t=0 and t=1 should both use frozen M_k (snapshot at t=0)
        // Since M_k evolves after t=0 via DGD, if we were using current M,
        // v_hat at t=1 would differ from what frozen M produces.
        let v_hat_k_t0 = &cache.inner.v_hat_targets[0..d];
        let v_hat_k_t1 = &cache.inner.v_hat_targets[d..2 * d];

        // Verify the frozen snapshot exists
        assert_eq!(cache.num_chunks, 1);
        let frozen_mk = &cache.frozen_snapshots[0..d * d];

        // Manually compute what frozen M_k @ v_t should give for both tokens
        let v_t0 = &cache.inner.v_mem[0..d];
        let v_t1 = &cache.inner.v_mem[d..2 * d];
        let mut expected0 = vec![0.0f32; d];
        let mut expected1 = vec![0.0f32; d];
        matmul_f32(frozen_mk, v_t0, &mut expected0, d, d, 1);
        matmul_f32(frozen_mk, v_t1, &mut expected1, d, d, 1);

        for i in 0..d {
            assert!((v_hat_k_t0[i] - expected0[i]).abs() < 1e-10,
                "v_hat_k[0][{i}] should use frozen M_k");
            assert!((v_hat_k_t1[i] - expected1[i]).abs() < 1e-10,
                "v_hat_k[1][{i}] should use frozen M_k");
        }
    }

    #[test]
    fn test_outer_loop_convergence() {
        // 100 outer-loop steps with C=2, loss should not increase monotonically.
        let d = 4;
        let seq_len = 4;
        let lr = 0.01f32;

        let state0 = make_state(d, 0.1);
        let m_mem0 = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let mut state = state0.clone();
        let mut m_mem_init = m_mem0.clone();
        let mut first_loss = 0.0f32;
        let mut last_loss = 0.0f32;

        for step in 0..100 {
            let mut s = state.clone();
            let mut mm = m_mem_init.clone();
            let (y, cache) = chunkwise_self_ref_step(&mut s, &mut mm, &embedded, seq_len, d, 2, false);
            let loss = simple_loss(&y);
            if step == 0 { first_loss = loss; }
            last_loss = loss;

            let d_y = simple_dloss(&y);
            let (_, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);

            // Apply outer-loop gradients to initial states
            for i in 0..d * d {
                state.m_k[i] -= lr * grads.d_m_k[i];
                state.m_v[i] -= lr * grads.d_m_v[i];
                state.m_q[i] -= lr * grads.d_m_q[i];
                state.m_eta[i] -= lr * grads.d_m_eta[i];
                state.m_alpha[i] -= lr * grads.d_m_alpha[i];
                m_mem_init[i] -= lr * grads.d_m_mem[i];
            }
        }

        // Loss should decrease (or at least not blow up)
        assert!(last_loss <= first_loss * 1.1,
            "Loss should not increase significantly: first={first_loss:.4e}, last={last_loss:.4e}");
    }

    #[test]
    fn test_remainder_chunk() {
        let d = 4;
        let seq_len = 7;
        let chunk_size = 4;

        let mut state = make_state(d, 0.1);
        let mut m_mem = make_m_mem(d, 0.1);
        let embedded = make_embedded(seq_len, d);

        let (y, cache) = chunkwise_self_ref_step(&mut state, &mut m_mem, &embedded, seq_len, d, chunk_size, false);
        assert_eq!(y.len(), seq_len * d);
        assert_eq!(cache.num_chunks, 2); // [4, 3]
        assert!(y.iter().all(|x| x.is_finite()));

        // Backward should also work
        let d_y = simple_dloss(&y);
        let (d_embedded, grads) = chunkwise_self_ref_step_backward(&cache, &d_y);
        assert_eq!(d_embedded.len(), seq_len * d);
        assert!(d_embedded.iter().all(|x| x.is_finite()));
        assert!(grads.d_m_k.iter().all(|x| x.is_finite()));
    }
}
